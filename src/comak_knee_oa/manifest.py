"""Per-stage input manifests.

Every pipeline stage records the SHA256 of each file it consumed and each file
it produced, into ``_manifest.json`` at the root of its stage directory. Before
a stage runs, it re-hashes its declared inputs and refuses to start if anything
disagrees with what the upstream stage recorded.

The problem this solves
-----------------------
A pipeline of cached intermediates has a quiet failure mode: an input changes,
a derived file computed from the old version stays on disk, and every later
stage reads it without complaint -- the numbers that come out are wrong but
entirely plausible, and nothing raises. Two properties close that route here:

1. Inputs are *vendored* -- copied into the data root, never referenced in
   place -- so nothing upstream can change underneath a cache (see
   ``paths.py``).
2. Every stage hashes what it consumed. A cache cannot go stale without the
   next stage noticing, because "stale" now means "hash mismatch" and that
   raises.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .paths import STAGES, data_root, declaration, stage_dir

__all__ = [
    "MANIFEST_NAME",
    "FileRecord",
    "StageManifest",
    "sha256_file",
    "record_files",
    "write_manifest",
    "read_manifest",
    "verify_inputs",
    "StaleInputError",
    "PINNED_MARKER",
    "environment_record",
]

MANIFEST_NAME = "_manifest.json"

_CHUNK = 1 << 20  # 1 MiB

# Repositories whose commit determines the numbers. Recorded per stage so a
# result can always be traced to the code that produced it. A repo a given stage
# does not import is skipped rather than recorded as missing, so this list can
# name every repo the pipeline uses without every stage needing all of them
# installed -- pyKneeSPM, for instance, is imported only by Figure 5's SPM stage.
_CODE_REPOS = ("comak_knee_oa", "nsosim", "pycomak", "NSM", "pymskt", "pyKneeSPM")

# A pinned library runs from an export of one commit (``git archive``), which
# has no git history. The export carries this file, holding the commit, so the
# version can still be recorded. See ``comak_knee_oa.pins``.
PINNED_MARKER = "PINNED_COMMIT"


class StaleInputError(RuntimeError):
    """An input's hash disagrees with what the upstream stage recorded."""


@dataclass(frozen=True)
class FileRecord:
    path: str  # relative to the data root, always POSIX-style
    sha256: str
    size: int
    # Where a vendored file was copied from, as an absolute path on the machine
    # that vendored it. Only the vendoring stage sets this: its entire job is to
    # record what a file was before it became ours. Every later stage reads its
    # inputs from the data root, where provenance is the upstream manifest, so
    # for those records this is None and is omitted from the JSON.
    source: str | None = None

    @classmethod
    def of(
        cls,
        path: Path,
        root: Path | None = None,
        source: str | None = None,
    ) -> "FileRecord":
        root = root or data_root()
        return cls(
            path=path.resolve().relative_to(root).as_posix(),
            sha256=sha256_file(path),
            size=path.stat().st_size,
            source=source,
        )

    def as_dict(self) -> dict:
        """JSON form, dropping ``source`` when unset.

        Most records have no source, and writing ``"source": null`` on tens of
        thousands of them makes the manifests larger and harder to read for no
        information gain.
        """
        d = {"path": self.path, "sha256": self.sha256, "size": self.size}
        if self.source is not None:
            d["source"] = self.source
        return d


@dataclass
class StageManifest:
    stage: str
    created: str
    inputs: list[FileRecord] = field(default_factory=list)
    outputs: list[FileRecord] = field(default_factory=list)
    code: dict[str, str] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)

    def input_map(self) -> dict[str, str]:
        return {r.path: r.sha256 for r in self.inputs}

    def output_map(self) -> dict[str, str]:
        return {r.path: r.sha256 for r in self.outputs}


def sha256_file(path: Path) -> str:
    """SHA256 of a file, read in chunks so multi-GB meshes do not blow memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def record_files(paths, root: Path | None = None) -> list[FileRecord]:
    """Hash an iterable of files into FileRecords, sorted by path.

    Directories are expanded recursively. Sorting keeps manifests diffable and
    makes two runs over the same tree byte-identical.

    Already-built ``FileRecord``s pass through unhashed. The vendoring stage
    hashes each file as it copies it, so re-reading ~100 GB purely to hash it a
    second time would double the stage's I/O for no added confidence.
    """
    root = root or data_root()
    ready: list[FileRecord] = []
    expanded: list[Path] = []
    for p in paths:
        if isinstance(p, FileRecord):
            ready.append(p)
            continue
        p = Path(p)
        if p.is_dir():
            expanded.extend(q for q in p.rglob("*") if q.is_file())
        elif p.is_file():
            expanded.append(p)
        else:
            raise FileNotFoundError(p)
    ready.extend(FileRecord.of(p, root) for p in expanded)
    return sorted(ready, key=lambda r: r.path)


def _git_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` containing .git or a pinned-export marker.

    Walked rather than assumed: a src/ layout puts the package two levels below
    the repository root and a flat layout one, and guessing wrong would record
    no version at all, without an error.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / PINNED_MARKER).is_file():
            return candidate
    return None


def _package_dir(name: str) -> Path | None:
    """Where ``name`` would be imported from, without importing it.

    Importing to find out would pull Torch into every run record just to write
    down a path.
    """
    module = sys.modules.get(name)
    if module is not None and getattr(module, "__file__", None):
        return Path(module.__file__).resolve().parent
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or spec.origin in ("built-in", "frozen"):
        return None
    return Path(spec.origin).resolve().parent


def _repo_version(repo_dir: Path) -> str | None:
    marker = repo_dir / PINNED_MARKER
    if marker.is_file():
        return marker.read_text().split()[0]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        if not dirty:
            return sha
        # Two different uncommitted states must not record the same version:
        # the suffix hashes the diff plus the names of untracked files.
        diff = subprocess.run(
            ["git", "-C", str(repo_dir), "diff", "HEAD"],
            capture_output=True, timeout=60, check=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "-C", str(repo_dir), "ls-files", "--others", "--exclude-standard"],
            capture_output=True, timeout=60, check=True,
        ).stdout
        digest = hashlib.sha256(diff + b"\0" + untracked).hexdigest()[:12]
        return f"{sha}-dirty-{digest}"
    except Exception:
        return None


def _code_versions(repos=_CODE_REPOS) -> dict[str, str]:
    """Commit SHA for each repo that determines the numbers.

    A checkout with uncommitted changes records ``<sha>-dirty-<hash of the
    diff>``. A pinned export records the commit it was exported from.
    """
    out: dict[str, str] = {}
    for repo in repos:
        package = _package_dir(repo)
        repo_dir = _git_root(package) if package is not None else None
        if repo_dir is None:
            continue
        version = _repo_version(repo_dir)
        if version:
            out[repo] = version
    return out


# ------------------------------------------------------------ environment ----

# Thread and floating-point settings that can change a solver's numbers.
_NUMERIC_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "OPENBLAS_CORETYPE", "MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS",
)
_SLURM_ENV = (
    "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_JOB_PARTITION",
)
_LIBRARIES = ("comak_knee_oa", "nsosim", "pycomak", "NSM", "pymskt", "pyKneeSPM", "opensim")
_native_md5: dict[str, str] = {}


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _md5(path: str) -> str:
    if path not in _native_md5:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                h.update(chunk)
        _native_md5[path] = h.hexdigest()
    return _native_md5[path]


# Native libraries whose identity can move a solver's numbers: the OpenSim and
# Simbody libraries, and the BLAS/LAPACK they call. The ones hashed are those
# not covered by a pinned commit: the solver (built from source) and the BLAS
# (which picks its kernels per CPU at run time). MKL is numpy's BLAS in the
# paper's environment, so it is recorded and hashed too.
_NATIVE = ("libosim", "libSimTK", "openblas", "lapack", "libmkl_")
_NATIVE_HASHED = ("libosimJAM", "libopenblas", "libmkl_")
# A library accepted in place of the paper's by `pins check --accept` (the
# variable pins.ACCEPT_ENV names); recorded so an accepted build shows in every
# run record, not only in a job log.
_PINS_ACCEPT_ENV = "COMAK_PINS_ACCEPT"


def _loaded_native_libraries() -> dict[str, dict]:
    """The solver-side native libraries this process actually loaded.

    Read from the process's own memory map, so it reports the library that ran,
    not the one a path variable suggests. Empty until OpenSim is imported.
    """
    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError:
        return {}
    paths = sorted({line.split()[-1] for line in maps.splitlines()
                    if line.split()[-1].startswith("/")
                    and any(tag in os.path.basename(line.split()[-1]) for tag in _NATIVE)})
    out = {}
    for path in paths:
        entry = {"path": path}
        if os.path.basename(path).startswith(_NATIVE_HASHED):
            entry["md5"] = _md5(path)
            entry["size"] = os.path.getsize(path)
        out[os.path.basename(path)] = entry
    return out


def environment_record() -> dict:
    """Where and how this process ran: host, CPU, scheduler job, data, libraries.

    COMAK is deterministic per CPU type but not across types, so the CPU model
    is what decides whether two results should agree exactly.
    """
    try:
        root = data_root()
        decl = declaration(root) or {}
    except FileNotFoundError:
        root, decl = None, {}
    libraries = {}
    for name in _LIBRARIES:
        package = _package_dir(name)
        if package is not None:
            libraries[name] = str(package)
    return {
        "host": socket.gethostname(),
        "cpu_model": cpu_model(),
        "slurm": {k: os.environ[k] for k in _SLURM_ENV if k in os.environ},
        "data_root": str(root) if root else None,
        "run_name": decl.get("run_name"),
        "python": sys.version.split()[0],
        "pythonpath": os.environ.get("PYTHONPATH"),
        "numeric_env": {k: os.environ.get(k) for k in _NUMERIC_ENV},
        "libraries": libraries,
        "native_libraries": _loaded_native_libraries(),
        "pins_accept": os.environ.get(_PINS_ACCEPT_ENV) or None,
    }


def write_manifest(
    stage: str,
    inputs,
    outputs=(),
    notes: dict | None = None,
) -> Path:
    """Hash a stage's inputs and outputs and write ``_manifest.json``."""
    root = data_root()
    manifest = StageManifest(
        stage=stage,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        inputs=record_files(inputs, root),
        outputs=record_files(outputs, root),
        code=_code_versions(),
        notes=notes or {},
    )
    payload = {
        "stage": manifest.stage,
        "created": manifest.created,
        "inputs": [r.as_dict() for r in manifest.inputs],
        "outputs": [r.as_dict() for r in manifest.outputs],
        "code": manifest.code,
        "environment": environment_record(),
        "notes": manifest.notes,
    }
    target = stage_dir(stage, create=True) / MANIFEST_NAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, target)  # atomic: never leave a half-written manifest
    return target


def read_manifest(stage: str) -> StageManifest:
    path = stage_dir(stage) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{stage!r} has no {MANIFEST_NAME}. Run that stage before the one "
            "that depends on it."
        )
    raw = json.loads(path.read_text())
    return StageManifest(
        stage=raw["stage"],
        created=raw["created"],
        inputs=[FileRecord(**r) for r in raw["inputs"]],
        outputs=[FileRecord(**r) for r in raw["outputs"]],
        code=raw.get("code", {}),
        notes=raw.get("notes", {}),
    )


def verify_inputs(stage: str, upstream: str) -> None:
    """Check this stage's inputs still match what ``upstream`` produced.

    Call at the top of a stage, before any expensive work.

    Raises
    ------
    StaleInputError
        If any file the upstream stage produced has changed, gone missing, or
        is a file this stage cannot account for. The message names the files
        rather than just reporting a count -- when this fires it is usually a
        surprise, and the first question is always "which ones".
    """
    root = data_root()
    expected = read_manifest(upstream).output_map()

    missing: list[str] = []
    changed: list[str] = []
    for rel, sha in sorted(expected.items()):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != sha:
            changed.append(rel)

    if not (missing or changed):
        return

    def _sample(label: str, items: list[str], limit: int = 10) -> str:
        if not items:
            return ""
        shown = "\n".join(f"    {i}" for i in items[:limit])
        more = f"\n    ... and {len(items) - limit} more" if len(items) > limit else ""
        return f"\n  {label} ({len(items)}):\n{shown}{more}"

    raise StaleInputError(
        f"Stage {stage!r} cannot run: its inputs no longer match what stage "
        f"{upstream!r} recorded."
        + _sample("changed", changed)
        + _sample("missing", missing)
        + f"\n\nRe-run stage {upstream!r}, or restore the data root from the "
        "published dataset. Do not bypass this check: results derived from a "
        "mismatched intermediate are wrong in ways that raise no error."
    )


# ------------------------------------------------------------------ CLI ----


def _resolve_stage(name: str) -> str:
    """Accept either the short key ('derived') or the directory ('03_derived')."""
    if name in STAGES:
        return name
    by_dir = {v: k for k, v in STAGES.items()}
    if name in by_dir:
        return by_dir[name]
    raise SystemExit(
        f"unknown stage {name!r}\n"
        f"expected one of: {', '.join(sorted(STAGES))}\n"
        f"           or  : {', '.join(sorted(STAGES.values()))}"
    )


def _cmd_verify(stage: str) -> int:
    """Re-hash a stage's own outputs against its manifest."""
    key = _resolve_stage(stage)
    manifest = read_manifest(key)
    root = data_root()

    bad: list[str] = []
    for rel, sha in sorted(manifest.output_map().items()):
        path = root / rel
        if not path.is_file():
            bad.append(f"missing  {rel}")
        elif sha256_file(path) != sha:
            bad.append(f"changed  {rel}")

    n = len(manifest.outputs)
    if bad:
        print(f"FAIL  {key}: {len(bad)} of {n} file(s) do not match the manifest")
        for line in bad[:20]:
            print(f"  {line}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
        return 1

    print(f"OK    {key}: {n} file(s) match the manifest (written {manifest.created})")
    if manifest.code:
        for repo, sha in sorted(manifest.code.items()):
            print(f"        {repo} {sha}")
    return 0


def _cmd_show(stage: str) -> int:
    key = _resolve_stage(stage)
    m = read_manifest(key)
    print(f"stage    {m.stage}")
    print(f"created  {m.created}")
    print(f"inputs   {len(m.inputs)} file(s)")
    print(f"outputs  {len(m.outputs)} file(s)")
    for repo, sha in sorted(m.code.items()):
        print(f"code     {repo} {sha}")
    for k, v in sorted(m.notes.items()):
        print(f"note     {k}: {v}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m comak_knee_oa.manifest",
        description="Inspect and verify pipeline stage manifests.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser(
        "verify", help="re-hash a stage's outputs against its manifest"
    )
    p_verify.add_argument("stage", help="e.g. derived, or 03_derived")

    p_show = sub.add_parser("show", help="summarise a stage's manifest")
    p_show.add_argument("stage", help="e.g. derived, or 03_derived")

    args = parser.parse_args(argv)
    return {"verify": _cmd_verify, "show": _cmd_show}[args.command](args.stage)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
