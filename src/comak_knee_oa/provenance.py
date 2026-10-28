"""Skip work only when what is on disk was made from the same inputs.

Several stages skip work whose output already exists -- scaling a knee to a gait
donor, rebuilding a model, transferring pressure -- because re-running a
partially finished cohort should not redo hours of finished work. Existence is
the wrong test. An output built from a different donor body, an older model, a
different library commit or a different option looks like a finished one, and
skipping it would carry the old inputs into the new run without warning: a fix
switched on that does nothing.

So an output unit -- a directory holding one piece of work -- carries a record
of what made it, ``_provenance.json``: the hash of every input, the options,
and the commits of the code that matters. Before skipping, a stage recomputes
that fingerprint and compares:

=================================  ==============================================
on disk                            decision
=================================  ==============================================
nothing                            build
record matches, marked complete    skip
record matches, marked started     build (this run's own interrupted attempt)
record differs                     raise -- name what changed; ``force`` rebuilds
output present, no record          raise -- not made by this code (an as-run
                                   file); ``force`` rebuilds
=================================  ==============================================

Raising rather than rebuilding on a mismatch is deliberate. A mismatch means
either the inputs really changed, and the operator should know that before
hundreds of outputs are replaced, or something is pointing at the wrong place.

    fp = provenance.fingerprint(inputs={"body_model": path}, options=kwargs,
                                code=("nsosim",))
    if provenance.decide(unit, fp, complete=is_complete(unit), force=force,
                         what="scaled model") == "skip":
        return
    provenance.mark_started(unit, fp)
    ... build ...
    provenance.mark_complete(unit, fp)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .paths import data_root

__all__ = [
    "RECORD_NAME",
    "ProvenanceMismatch",
    "fingerprint",
    "read_record",
    "differences",
    "decide",
    "mark_started",
    "mark_complete",
    "hash_input",
]

RECORD_NAME = "_provenance.json"
_CHUNK = 1 << 20


class ProvenanceMismatch(RuntimeError):
    """An output exists but was not made from the inputs now asked for."""


# ----------------------------------------------------------------- hashing ----


def _file_hashes(path: Path) -> tuple[str, str, int]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), size


def _display(path: Path) -> dict:
    """Where an input is: data-root-relative when inside the root, so records
    survive a move, plus the real location when a link points elsewhere -- an
    input borrowed from a baseline root says so."""
    path = Path(path)
    real = path.resolve()
    try:
        shown = path.absolute().relative_to(data_root()).as_posix()
    except (ValueError, FileNotFoundError):
        shown = str(path)
    out = {"path": shown}
    if real != path.absolute():
        out["realpath"] = str(real)
    return out


def hash_input(path) -> dict:
    """Identity of one input: a file's sha256 and md5, or a directory's tree hash.

    A directory hashes to the sha256 of its sorted (relative path, file sha256)
    list, so renaming, adding, removing or changing any file changes it.
    """
    path = Path(path)
    if path.is_file():
        sha, md5, size = _file_hashes(path)
        return {**_display(path), "sha256": sha, "md5": md5, "size": size}
    if path.is_dir():
        tree = hashlib.sha256()
        files = sorted(p for p in path.rglob("*") if p.is_file())
        for f in files:
            sha, _md5, _size = _file_hashes(f)
            tree.update(f"{f.relative_to(path).as_posix()}\0{sha}\n".encode())
        return {**_display(path), "sha256": tree.hexdigest(), "files": len(files)}
    raise FileNotFoundError(f"provenance input not found: {path}")


def _options_hash(options: dict) -> str:
    text = json.dumps(options, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(*, inputs: dict, options: dict | None = None, code=("nsosim",)) -> dict:
    """What a piece of work is made from.

    Parameters
    ----------
    inputs : {role: path}
        Every file or directory the work reads that could change its result.
    options : dict
        The parameters it was run with -- exactly what is passed to the builder,
        so adding an option changes the fingerprint without anyone remembering
        to add it here.
    code : names of the repositories whose commit matters (``manifest`` reads
        them, including pinned exports that carry no git history).
    """
    from .manifest import _code_versions

    options = dict(options or {})
    versions = _code_versions(code)
    missing = [repo for repo in code if repo not in versions]
    if missing:
        # An unrecorded commit would compare equal to any other unrecorded commit.
        raise RuntimeError(
            f"cannot record the version of {missing}: not importable, or not a git "
            "checkout or a pinned export. Provenance needs it to tell two builds apart."
        )
    return {
        "inputs": {role: hash_input(path) for role, path in sorted(inputs.items())},
        "options": options,
        "options_sha256": _options_hash(options),
        "code": versions,
    }


# ------------------------------------------------------------------ record ----


def _record_path(unit: Path) -> Path:
    return Path(unit) / RECORD_NAME


def read_record(unit) -> dict | None:
    path = _record_path(unit)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def differences(recorded: dict, expected: dict) -> list[str]:
    """Human-readable list of what differs between two fingerprints."""
    out = []
    old_inputs, new_inputs = recorded.get("inputs", {}), expected["inputs"]
    for role in sorted(set(old_inputs) | set(new_inputs)):
        old, new = old_inputs.get(role), new_inputs.get(role)
        if old is None or new is None:
            out.append(f"input {role!r}: {'added' if old is None else 'removed'}")
        elif old.get("sha256") != new.get("sha256"):
            out.append(
                f"input {role!r} changed: {old.get('path')} sha256 "
                f"{old.get('sha256', '')[:12]} -> {new.get('path')} {new['sha256'][:12]}"
            )
    if recorded.get("options_sha256") != expected["options_sha256"]:
        old_opts, new_opts = recorded.get("options", {}), expected["options"]
        keys = sorted(k for k in set(old_opts) | set(new_opts)
                      if old_opts.get(k, "<unset>") != new_opts.get(k, "<unset>"))
        out.append("options changed: " + ", ".join(
            f"{k}: {old_opts.get(k, '<unset>')!r} -> {new_opts.get(k, '<unset>')!r}"
            for k in keys) if keys else "options changed")
    for repo, sha in expected["code"].items():
        if recorded.get("code", {}).get(repo) != sha:
            out.append(f"code {repo}: {recorded.get('code', {}).get(repo)} -> {sha}")
    return out


def _has_outputs(unit: Path) -> bool:
    unit = Path(unit)
    return unit.is_dir() and any(p.name != RECORD_NAME for p in unit.iterdir())


def decide(unit, expected: dict, *, complete: bool, force: bool = False,
           what: str = "output") -> str:
    """``"skip"`` or ``"build"`` for one output unit; raises on a mismatch.

    ``complete`` is the caller's own check that the outputs are all there.
    """
    unit = Path(unit)
    recorded = read_record(unit)
    if recorded is None:
        if not _has_outputs(unit) or force:
            return "build"
        raise ProvenanceMismatch(
            f"{what} at {unit} exists but carries no {RECORD_NAME}: it was not "
            "produced by provenance-recording code -- an as-run or copied output, "
            "or one written before provenance existed. Refusing to reuse it. Pass "
            "--force to rebuild it, or move it aside if it must be kept."
        )
    changed = differences(recorded, expected)
    if changed:
        if force:
            return "build"
        raise ProvenanceMismatch(
            f"{what} at {unit} was made from different inputs than this run asks for:\n  "
            + "\n  ".join(changed)
            + "\nRefusing to reuse it or to replace it silently. Pass --force to rebuild."
        )
    if recorded.get("state") == "complete" and complete:
        return "skip"
    return "build"


def _write(unit: Path, expected: dict, state: str) -> Path:
    unit = Path(unit)
    unit.mkdir(parents=True, exist_ok=True)
    record = dict(expected)
    record["state"] = state
    record["recorded"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target = _record_path(unit)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n")
    os.replace(tmp, target)
    return target


def mark_started(unit, expected: dict) -> Path:
    """Record that this fingerprint's build has begun (before any output is written)."""
    return _write(unit, expected, "started")


def mark_complete(unit, expected: dict) -> Path:
    """Record that this fingerprint's build finished and its outputs are whole."""
    return _write(unit, expected, "complete")
