"""The library commits a run must use, and how a launcher enforces them.

``docs/pinned_versions.md`` explains *why* each pin is what it is. This module
is the machine-readable half: the commits themselves, a way to export each one
into a directory that cannot drift, and a check a launcher runs before any work.

Why exports rather than the installed checkouts
-----------------------------------------------
The development environment imports nsosim, pycomak, NSM and pymskt from live
git checkouts. A checkout is shared: switching its branch, or editing it,
changes the code under every job that imports it next -- including jobs
already queued. An export of one commit (``git archive``) is a directory nobody
edits, and put first on ``PYTHONPATH`` it wins over the installed copy. The
export carries ``PINNED_COMMIT`` so the run records still name the commit.

Two of these pins are load-bearing for the numbers in a way that does not
raise: the shape models do not load correctly under a newer NSM, and the
pre-pin pymskt meshes differently. A launcher that forgets the pins produces
plausible, wrong output. So the cluster launcher runs ``check --strict`` and
refuses to start on a mismatch.

    python -m comak_knee_oa.pins export --dest <dir> \\
        --repo NSM=<checkout> --repo pymskt=<checkout> \\
        --repo nsosim=<checkout> --repo pycomak=<checkout>
    export PYTHONPATH="$(python -m comak_knee_oa.pins pythonpath --dest <dir>)"
    python -m comak_knee_oa.pins check --strict

Repository locations are arguments, never constants: they are properties of a
machine, not of the pipeline.

What ``check`` compares
-----------------------
- the commit of each pinned library, as imported;
- the md5 of the JAM solver library OpenSim loads (``JAM``);
- the md5 of every BLAS and LAPACK library the process loads (``BLAS``): numpy's
  (MKL), SciPy's own OpenBLAS, and the OpenBLAS the solver's Simbody calls. The
  environment's BLAS moves simulation results by about as much as a change of
  CPU type does, and nothing else would notice it.

A library built or installed differently from the paper's -- your own build of
the JAM solver, whose md5 depends on the compiler and the build paths -- is
accepted only by naming its file and md5 explicitly::

    python -m comak_knee_oa.pins check --strict --accept libosimJAM.so=<md5>
    export COMAK_PINS_ACCEPT=libosimJAM.so=<md5>        # the same, for jobs

``check`` prints every accepted difference. A run record names the md5 of each
of these libraries that the simulation loaded, so a run made with an accepted
difference shows it there too.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from .manifest import PINNED_MARKER, _code_versions, _loaded_native_libraries, _package_dir

__all__ = ["PINS", "JAM", "BLAS", "ACCEPT_ENV", "accepted", "check", "export"]

# Library commits of the v3 run (docs/pinned_versions.md). nsosim and pycomak
# include changes that are each switched by a config key or are output-neutral:
# the AddBiomechanics-generic check (``scaling.allow_generic_mismatch``), the
# per-body mass rule (``scaling.mass_rule``; ``as_run`` is byte-identical to v2;
# ``volume`` gives marker-less bodies a neighbour's factors for mass), the fat
# pad's clearance rays (``fatpad_clearance_rays``), and linked rather than
# copied geometry (results bit-identical either way). nsosim and pycomak are
# tagged ``paper1-v3`` at these commits.
PINS: dict[str, str] = {
    "NSM": "bb2c6a3b17bd2cdb6b3378ea11528763e2698bf2",
    "pymskt": "88b947e3075d270f345bf7f8b545ead7daefd125",
    "nsosim": "29037fbfd71db9d5b3f2874214cff4eab2a79dbf",
    "pycomak": "bc1a59fed07529840a8b4f267161b249c6414c4f",
}

# The OpenSim-JAM solver library. Built from source, not installed from an
# index, so the binary itself is the identifier. It is built from commit
# c4b81de, which the tag paper1-v3 on github.com/gattia/opensim-core points at.
JAM = {
    "commit": "c4b81decaac5bcaeaeea1a2943826ecd6b82ad7e",
    "library": "libosimJAM.so",
    "md5": "705cca876f52b54538537800a3bf021a",
    "size": 92902520,
}


# The BLAS and LAPACK libraries of the paper's environment (conda env `comak`,
# which the docs create as `comak-knee-oa` from environment.lock.txt), by file name -> md5. Identity matters here:
# the same simulation in an environment whose BLAS is OpenBLAS 0.3.34 instead
# moves by up to 1.3 deg and 0.4 % tibial pressure on the same CPU, with every
# other check passing.
BLAS = {
    # conda-forge libopenblas 0.3.21 (pthreads_h78a6416_3). The solver's Simbody
    # links libopenblas.so.0, and the environment's activation puts this first.
    "libopenblasp-r0.3.21.so": "acc0da672279cf8ebf5fe47d01cb73f6",
    # SciPy 1.13.1's own OpenBLAS (PyPI wheel).
    "libopenblasp-r0-01191904.3.27.so": "97452884f82a712b9b65e75601c00244",
    # MKL 2022.1.0 (anaconda, hc2b9512_224): numpy's BLAS, through conda-forge
    # libblas 3.9.0=16_linux64_mkl. MKL loads its kernels for the CPU at run
    # time, so every kernel library of the package is listed.
    "libmkl_rt.so.2": "1deb7da3462c6a8f7e786ac2a58f55a5",
    "libmkl_core.so.2": "6fce53a17445224bb66fe17a6db312c1",
    "libmkl_gf_lp64.so.2": "6571e7191d6dac014e650689009ca0e8",
    "libmkl_intel_lp64.so.2": "114320132601387b18eb765bc1153d39",
    "libmkl_intel_thread.so.2": "d0fa7a36f35847c7ce8c9ee573bbca1c",
    "libmkl_gnu_thread.so.2": "fb5b04d0a3ad769b76b097e26e29f4e1",
    "libmkl_sequential.so.2": "0ac732c2d06146d22a912693ee481176",
    "libmkl_tbb_thread.so.2": "1647f5bc52a0020c1147be92ad9a3acb",
    "libmkl_def.so.2": "9f81bcc3edf5ccbaaff54c606b2191f1",
    "libmkl_mc.so.2": "6b79c347ac488731046e4b67b481b885",
    "libmkl_mc3.so.2": "caa8a7540050c7f9e17f1a1cf71133c7",
    "libmkl_avx.so.2": "7bdaf4d2b99edec555d69215d393094f",
    "libmkl_avx2.so.2": "680db3cab99a1a8ffbb564c7a0d9b965",
    "libmkl_avx512.so.2": "dc1ee26927679353046adc91b3d6b23b",
    "libmkl_vml_def.so.2": "295835b95837dd3223b13c6d9d830f8d",
    "libmkl_vml_mc.so.2": "33b3943b03019cb432f961e72b0912ea",
    "libmkl_vml_mc2.so.2": "ad8d4998ea58324d5ae0e586f0fc1137",
    "libmkl_vml_mc3.so.2": "dc9785e3183da795510bba17ce89c12a",
    "libmkl_vml_avx.so.2": "865651d03534d74267d54b5a589b1048",
    "libmkl_vml_avx2.so.2": "f2d991088ee55abce6a89d7be2db0450",
    "libmkl_vml_avx512.so.2": "9709275c004fa402f13f4fd57883db83",
    "libmkl_vml_cmpt.so.2": "a7da9a3c0b008761e67dbd7169a545f9",
}
# A loaded library is a BLAS/LAPACK library if its file name contains one of these.
_BLAS_TAGS = ("openblas", "libmkl_", "libblas", "libcblas", "liblapack", "libflexiblas", "libblis")

# Explicitly accepted differences, "<library file>=<md5>[,<library file>=<md5>...]".
ACCEPT_ENV = "COMAK_PINS_ACCEPT"


def _short(sha: str) -> str:
    return sha[:7]


# ------------------------------------------------------------------ export ----


def export(dest: Path, repos: dict[str, Path]) -> dict[str, Path]:
    """Export each pinned commit of ``repos`` into ``dest/<name>_<sha7>``.

    An existing export is kept if its marker names the same commit, and is an
    error otherwise -- an export is replaced by making a new directory, never
    by rewriting one a running job may be importing from.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, repo in sorted(repos.items()):
        if name not in PINS:
            raise KeyError(f"no pin for {name!r}; pinned: {sorted(PINS)}")
        sha = PINS[name]
        target = dest / f"{name}_{_short(sha)}"
        marker = target / PINNED_MARKER
        if target.exists():
            recorded = marker.read_text().split()[0] if marker.is_file() else None
            if recorded != sha:
                raise RuntimeError(f"{target} exists but holds {recorded!r}, not {sha}")
            out[name] = target
            continue
        resolved = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", f"{sha}^{{commit}}"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if resolved != sha:
            raise RuntimeError(f"{repo} resolves {sha} to {resolved}")
        partial = dest / f".{target.name}.partial"
        partial.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", sha],
            capture_output=True, check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(partial)], input=archive, check=True)
        origin = subprocess.run(
            ["git", "-C", str(repo), "remote", "get-url", "origin"],
            capture_output=True, text=True,
        ).stdout.strip()
        (partial / PINNED_MARKER).write_text(f"{sha}\n{name} {origin}\n")
        # Byte-compile now: once read-only, imports cannot cache it. Best effort --
        # examples or scripts written for another Python version may not compile.
        subprocess.run([sys.executable, "-m", "compileall", "-q", str(partial)],
                       capture_output=True)
        # Read-only, so nothing edits a pinned export in place.
        os.rename(partial, target)
        for path in [*sorted(target.rglob("*"), reverse=True), target]:
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        out[name] = target
    return out


def pythonpath(dest: Path) -> str:
    """The ``PYTHONPATH`` that puts every pinned export first."""
    parts = []
    for name, sha in sorted(PINS.items()):
        target = Path(dest) / f"{name}_{_short(sha)}"
        if not target.is_dir():
            raise FileNotFoundError(f"no export for {name} at {target}; run `pins export`")
        parts.append(str(target))
    return os.pathsep.join(parts)


# ------------------------------------------------------------------- check ----


def accepted(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The explicitly accepted native libraries: ``$COMAK_PINS_ACCEPT`` plus ``extra``.

    Raises on a malformed entry: an override that accepts nothing, or
    everything, without saying so is worse than none.
    """
    out: dict[str, str] = {}
    for item in filter(None, (x.strip() for x in os.environ.get(ACCEPT_ENV, "").split(","))):
        name, sep, md5 = item.partition("=")
        if not sep or not name or len(md5) != 32 or any(c not in "0123456789abcdef" for c in md5):
            raise ValueError(f"${ACCEPT_ENV}: {item!r} is not '<library file>=<md5>'")
        out[name] = md5
    for name, md5 in (extra or {}).items():
        if len(md5) != 32 or any(c not in "0123456789abcdef" for c in md5):
            raise ValueError(f"--accept {name}={md5}: not an md5")
        out[name] = md5
    return out


def _load_blas() -> None:
    """Load the BLAS libraries a run loads: numpy's (and its CPU kernels), SciPy's."""
    import numpy as np

    a = np.ones((64, 64))
    a @ a  # MKL loads its kernel library for this CPU on the first call
    try:
        import scipy.linalg  # noqa: F401  -- SciPy's own OpenBLAS
    except ImportError:
        pass


def _blas_problems(loaded: dict[str, str], accept: dict[str, str], notes: list[str],
                   expect_solver: bool) -> list[str]:
    """Differences between the loaded BLAS libraries ({file name: md5}) and ``BLAS``."""
    problems = []
    for name, md5 in sorted(loaded.items()):
        if accept.get(name) == md5:
            if BLAS.get(name) != md5:
                notes.append(f"{name} md5 {md5}: ACCEPTED in place of "
                             f"{BLAS.get(name, 'no library of this name')} (override)")
            continue
        if name not in BLAS:
            problems.append(f"{name} (md5 {md5}) is loaded; the paper's environment has no "
                            "BLAS library of that name (environment.lock.txt recreates it)")
        elif BLAS[name] != md5:
            problems.append(f"{name} has md5 {md5} (want {BLAS[name]})")
    if "libmkl_rt.so.2" not in loaded:
        problems.append("numpy does not run on MKL (libmkl_rt.so.2 is not loaded), as it did for the paper")
    if expect_solver and "libopenblasp-r0.3.21.so" not in loaded and not any(
            "openblas" in n and accept.get(n) == loaded[n] for n in loaded):
        problems.append("the solver's OpenBLAS is not libopenblasp-r0.3.21.so")
    return problems


def _loaded_blas() -> dict[str, str]:
    """{file name: md5} of every BLAS/LAPACK library in this process's memory map."""
    from .manifest import _md5

    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError:
        return {}
    paths = {line.split()[-1] for line in maps.splitlines() if line.split()[-1].startswith("/")}
    return {os.path.basename(p): _md5(p) for p in sorted(paths)
            if any(t in os.path.basename(p) for t in _BLAS_TAGS) and os.path.isfile(p)}


def check(load_opensim: bool = True, accept: dict[str, str] | None = None,
          check_blas: bool = True, notes: list[str] | None = None) -> list[str]:
    """Every way the running environment differs from the pins (empty: matches).

    ``accept`` ({library file: md5}, added to ``$COMAK_PINS_ACCEPT``) names native
    libraries accepted in place of the paper's; each acceptance is appended to
    ``notes``.
    """
    notes = notes if notes is not None else []
    accept = accepted(accept)
    problems = []
    versions = _code_versions(tuple(PINS))
    for name, sha in sorted(PINS.items()):
        found = versions.get(name)
        where = _package_dir(name)
        if found is None:
            problems.append(f"{name}: no recordable version at {where} (want {sha})")
        elif found != sha:
            problems.append(f"{name}: {found} at {where} (want {sha})")
    opensim_loaded = False
    if load_opensim:
        try:
            import opensim  # noqa: F401  -- loads the solver libraries
        except Exception as exc:
            problems.append(f"opensim failed to import: {exc}")
        else:
            opensim_loaded = True
            lib = _loaded_native_libraries().get(JAM["library"])
            if lib is None:
                problems.append(f"{JAM['library']} is not loaded after importing opensim")
            elif lib.get("md5") != JAM["md5"]:
                if accept.get(JAM["library"]) == lib.get("md5"):
                    notes.append(f"{JAM['library']} at {lib['path']} md5 {lib['md5']}: ACCEPTED in place "
                                 f"of the paper's {JAM['md5']} (override)")
                else:
                    problems.append(
                        f"{JAM['library']} at {lib['path']} has md5 {lib.get('md5')} "
                        f"(want {JAM['md5']}, built from JAM {JAM['commit'][:7]}; a build of your own "
                        f"is accepted with --accept {JAM['library']}=<its md5>)"
                    )
    if check_blas:
        _load_blas()
        problems += _blas_problems(_loaded_blas(), accept, notes, expect_solver=opensim_loaded)
    return problems


# --------------------------------------------------------------------- CLI ----


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m comak_knee_oa.pins", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="export each pinned commit into --dest")
    p.add_argument("--dest", required=True, type=Path)
    p.add_argument("--repo", action="append", default=[], metavar="NAME=PATH",
                   help="a local checkout holding the pinned commit; repeat per library")

    p = sub.add_parser("pythonpath", help="print the PYTHONPATH for the exports in --dest")
    p.add_argument("--dest", required=True, type=Path)

    p = sub.add_parser("check", help="compare the running environment with the pins")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any difference")
    p.add_argument("--no-opensim", action="store_true", help="skip the solver-library check")
    p.add_argument("--no-blas", action="store_true", help="skip the BLAS-library check")
    p.add_argument("--accept", action="append", default=[], metavar="LIBRARY=MD5",
                   help=f"accept this native library file with this md5 in place of the paper's, e.g. "
                        f"your own build of {JAM['library']} (also ${ACCEPT_ENV}); printed when used")

    sub.add_parser("show", help="print the pins as JSON")

    args = parser.parse_args(argv)
    if args.command == "export":
        repos = {}
        for item in args.repo:
            name, _, path = item.partition("=")
            repos[name] = Path(path)
        for name, target in export(args.dest, repos).items():
            print(f"{name:8s} {target}")
        return 0
    if args.command == "pythonpath":
        print(pythonpath(args.dest))
        return 0
    if args.command == "show":
        print(json.dumps({"libraries": PINS, "jam": JAM}, indent=2))
        return 0
    extra = {}
    for item in args.accept:
        name, sep, md5 = item.partition("=")
        if not sep:
            parser.error(f"--accept {item!r}: expected LIBRARY=MD5")
        extra[name] = md5
    notes: list[str] = []
    problems = check(load_opensim=not args.no_opensim, accept=extra,
                     check_blas=not args.no_blas, notes=notes)
    for note in notes:
        print(f"pins: {note}", file=sys.stderr)
    if problems:
        print("PIN MISMATCH:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1 if args.strict else 0
    print("pins OK: " + ", ".join(f"{k} {v[:7]}" for k, v in sorted(PINS.items()))
          + (f", {JAM['library']} {JAM['md5'][:8]}" if not args.no_opensim else "")
          + (", BLAS as the paper's" if not args.no_blas else "")
          + (f" ({len(notes)} accepted difference(s), above)" if notes else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
