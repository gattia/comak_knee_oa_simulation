"""Where the data lives.

This is the one place in the repository that decides a data path. Everything
else asks for a stage directory by name. There are no absolute paths anywhere
else in the codebase, and ``tests/test_no_absolute_paths.py`` enforces that.

Resolution order for ``DATA_ROOT``
----------------------------------
1. ``$COMAK_DATA_ROOT``          -- explicit override, e.g. a one-off scratch run
2. ``<repo>/data_root.txt``      -- gitignored, one line; how this machine is set up
3. ``<repo>/data``               -- the default; what a HuggingFace download produces

The resolved root must already exist. It is never created implicitly.

Why it fails instead of falling back
------------------------------------
This pipeline runs under SLURM. If a batch job lost its environment and the
root defaulted to ``<repo>/data`` without a check, that directory would be
created and the job would write to the wrong place with no error, producing
plausible but misplaced results. A failure at startup is cheap; that one is
not.

The run a data root holds
-------------------------
A data root that is being written declares which run it holds, in
``<root>/_run.json`` (``python -m comak_knee_oa.new_root`` writes it). One root
holds one run: the run name is part of the names of what the run produces -- the
built models' file names, each simulation's results folder, the quality-control
keys -- and configs spell it ``{run}``. Keeping it in the root rather than in
the repository means a checkout cannot disagree with the data it writes.

Read-only roots, and the write guard
------------------------------------
A finished dataset is never written to. Importing this module installs a Python
audit hook that refuses any write -- ``open`` for writing, ``mkdir``, rename,
remove, copy, ``rmtree`` -- whose *resolved* target lies inside a read-only
root. Resolving matters: a new root borrows its inputs from a baseline root
through symlinks, and a write through such a link would otherwise land in the
baseline tree under an innocent-looking path. Read-only means:

* the active root, if its declaration says ``"read_only": true``;
* the active root, if it declares no run but already holds produced outputs (a
  stage manifest in ``01_models`` .. ``04_results``) -- a finished dataset that
  nobody has declared writable;
* every root the active root's declaration lists under ``read_only_roots``;
* every root in ``$COMAK_READ_ONLY_ROOTS`` (``os.pathsep``-separated).

Also refused: a write that reaches outside the active root through a symlink
from inside it. Content borrowed through a link is read-only, wherever it
points.

The hook sees Python-level writes only. Native libraries (VTK, HDF5, the OpenSim
solver) write through C and are not seen -- but they write into directories that
Python created, and that ``mkdir`` is seen.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "INPUTS",
    "MODELS",
    "SIMULATIONS",
    "DERIVED",
    "RESULTS",
    "DECLARATION_NAME",
    "ReadOnlyRootError",
    "UndeclaredRunError",
    "data_root",
    "stage_dir",
    "declaration",
    "run_name",
    "protected_roots",
    "require_writable",
    "check_run_name",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

_POINTER_FILE = REPO_ROOT / "data_root.txt"
_ENV_VAR = "COMAK_DATA_ROOT"
_READ_ONLY_ENV = "COMAK_READ_ONLY_ROOTS"

# The run declaration a writable data root carries. See the module docstring.
DECLARATION_NAME = "_run.json"

# A run name becomes part of file and directory names, so it is kept to
# characters that are safe in both and cannot introduce a path separator.
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class ReadOnlyRootError(RuntimeError):
    """A write would land inside a read-only data root.

    Not an ``OSError``, so that code that tolerates I/O failures with
    ``except OSError`` must not swallow this one.
    """


class UndeclaredRunError(RuntimeError):
    """A config asked for ``{run}`` on a data root that declares no run."""

# Stage directories, in pipeline order. The names are part of the released
# data layout: a user downloads this tree from HuggingFace.
STAGES: dict[str, str] = {
    "inputs": "00_inputs",
    "models": "01_models",
    "simulations": "02_simulations",
    "derived": "03_derived",
    "results": "04_results",
}


def _candidates() -> list[tuple[str, Path]]:
    """(source description, path) in resolution order, skipping unset sources."""
    found: list[tuple[str, Path]] = []

    env = os.environ.get(_ENV_VAR)
    if env:
        found.append((f"${_ENV_VAR}", Path(env).expanduser()))

    if _POINTER_FILE.is_file():
        for line in _POINTER_FILE.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                found.append((str(_POINTER_FILE), Path(line).expanduser()))
                break

    found.append(("default", REPO_ROOT / "data"))
    return found


def data_root() -> Path:
    """Resolve and validate DATA_ROOT.

    Raises
    ------
    FileNotFoundError
        If no candidate exists on disk. The message lists every candidate that
        was tried and where it came from, so a failing SLURM job says what is
        misconfigured.
    """
    tried = _candidates()
    for _source, path in tried:
        if path.is_dir():
            return path.resolve()

    lines = [f"    {path}   (from {source})" for source, path in tried]
    raise FileNotFoundError(
        "Could not locate the data root. Tried, in order:\n"
        + "\n".join(lines)
        + "\n\nFix by either:\n"
        f"  - writing the path into {_POINTER_FILE} (one line), or\n"
        f"  - setting {_ENV_VAR}, or\n"
        "  - downloading the dataset into <repo>/data (see README.md).\n"
        "The directory is never created automatically -- writing simulation "
        "output to an unintended location is worse than failing here."
    )


def stage_dir(stage: str, *parts: str, create: bool = False) -> Path:
    """Path to a pipeline stage directory, optionally with sub-parts.

    >>> stage_dir("derived", "thickness")        # doctest: +SKIP
    PosixPath('.../03_derived/thickness')

    ``create=True`` makes the sub-parts (never the data root itself), and is for
    stages writing their own output. It refuses a read-only root up front, with
    the reason, rather than leaving the write guard to stop the first file.
    """
    try:
        name = STAGES[stage]
    except KeyError:
        raise KeyError(
            f"unknown stage {stage!r}; expected one of {sorted(STAGES)}"
        ) from None

    path = data_root().joinpath(name, *parts)
    if create:
        require_writable(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------- the run ----


def check_run_name(name: str) -> str:
    """Validate a run name, returning it unchanged."""
    if not isinstance(name, str) or not _RUN_NAME.fullmatch(name):
        raise ValueError(
            f"invalid run name {name!r}: use letters, digits, '_', '-' or '.', "
            "starting with a letter or digit, at most 64 characters"
        )
    return name


def declaration(root: Path | None = None) -> dict | None:
    """The run declaration of ``root`` (default: the active data root), or None."""
    root = Path(root) if root is not None else data_root()
    path = root / DECLARATION_NAME
    if not path.is_file():
        return None
    try:
        decl = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        # A declaration that cannot be read is not an absent one: guessing
        # "no run" here would make the root writable or unnamed by accident.
        raise RuntimeError(f"unreadable run declaration {path}: {exc}") from exc
    if not isinstance(decl, dict):
        raise RuntimeError(f"run declaration {path} is not a JSON object")
    return decl


def run_name() -> str:
    """The run the active data root holds -- what ``{run}`` in a config becomes.

    Raises
    ------
    UndeclaredRunError
        If the root declares no run. The v2 root has none: its folder names
        follow no single pattern, so it is read through
        ``cohorts.condition_config``, which fills in its layout.
    """
    root = data_root()
    decl = declaration(root)
    if not decl or not decl.get("run_name"):
        raise UndeclaredRunError(
            f"the data root {root} declares no run (no {DECLARATION_NAME}), and a "
            "config asked for {run}.\n"
            "  - To run the pipeline, give it its own root:\n"
            "        python -m comak_knee_oa.new_root create --root <new root> "
            "--run-name <name> --baseline <published root>\n"
            "    and point COMAK_DATA_ROOT (or data_root.txt) at it.\n"
            "  - To read a published root produced before run names existed, use "
            "comak_knee_oa.cohorts, which carries that layout (cohorts.V2_LAYOUT)."
        )
    return check_run_name(decl["run_name"])


# --------------------------------------------------------- read-only roots ----

# Stages whose manifest marks a root as holding produced outputs. Inputs are
# excluded: a fresh root borrows a populated 00_inputs from its baseline.
_PRODUCED_STAGES = ("models", "simulations", "derived", "results")


def _real(path) -> str:
    return os.path.realpath(os.fspath(path))


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _holds_outputs(root: Path) -> bool:
    return any((root / STAGES[s] / "_manifest.json").is_file() for s in _PRODUCED_STAGES)


def _active_root() -> Path | None:
    """The data root, or None on a machine where none is configured."""
    try:
        return data_root()
    except FileNotFoundError:
        return None


def _guard_context() -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """(active root's real path, ((read-only real path, why), ...))."""
    protected: list[tuple[str, str]] = []
    root = _active_root()
    active = _real(root) if root is not None else None
    if root is not None:
        decl = declaration(root)
        if decl is None:
            if _holds_outputs(root):
                protected.append(
                    (active, "it declares no run and already holds produced outputs "
                             "(a finished dataset)")
                )
        else:
            if decl.get("read_only"):
                protected.append((active, f"its {DECLARATION_NAME} marks it read_only"))
            for other in decl.get("read_only_roots", []):
                protected.append(
                    (_real(other), f"the active root's {DECLARATION_NAME} lists it read-only")
                )
    for other in filter(None, os.environ.get(_READ_ONLY_ENV, "").split(os.pathsep)):
        protected.append((_real(other), f"${_READ_ONLY_ENV} lists it"))
    return active, tuple(protected)


# The context is cached against everything it is computed from, so a write-heavy
# stage does not re-read the declaration once per file, yet a changed
# environment or a newly written declaration is picked up immediately.
_context_cache: dict = {}


def _context_key():
    def stamp(path: Path):
        try:
            st = os.stat(path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    root = _active_root()
    marks = None
    if root is not None:
        marks = (
            str(root),
            stamp(root / DECLARATION_NAME),
            tuple(stamp(root / STAGES[s] / "_manifest.json") is not None
                  for s in _PRODUCED_STAGES),
        )
    return (os.environ.get(_ENV_VAR), os.environ.get(_READ_ONLY_ENV),
            stamp(_POINTER_FILE), marks)


def _cached_context():
    key = _context_key()
    if _context_cache.get("key") != key:
        _context_cache["value"] = _guard_context()
        _context_cache["key"] = key
    return _context_cache["value"]


def protected_roots() -> list[Path]:
    """The read-only roots in force for this process, as real paths."""
    _active, protected = _cached_context()
    return [Path(p) for p, _why in protected]


def _violation(target: str, logical: str) -> str | None:
    """Why writing ``target`` (a real path) is refused, or None if it is allowed.

    ``logical`` is the path as written, made absolute but not resolved.
    """
    active, protected = _cached_context()
    for root, why in protected:
        if _within(target, root):
            return f"{target} is inside the read-only root {root}: {why}"
    if active and _within(logical, active) and not _within(target, active):
        return (
            f"{logical} resolves to {target}, outside the data root {active}, "
            "through a symlink. Content borrowed through a link is read-only; "
            "replace the link with a real directory if this run must write there."
        )
    return None


def require_writable(path) -> Path:
    """Return ``path`` if writing to it is allowed; raise ReadOnlyRootError if not."""
    logical = os.path.abspath(os.fspath(path))
    problem = _violation(_real(logical), logical)
    if problem:
        raise ReadOnlyRootError(problem)
    return Path(path)


# ------------------------------------------------------------ write guard ----
#
# Audit events that change the filesystem, and which positional arguments are
# paths. "follow": the operation acts on a symlink's target (open, chmod), so
# the target is checked. "link": it acts on the link itself (rename, remove,
# mkdir), so the parent is resolved and the final name is not.

_EVENTS: dict[str, tuple[tuple[int, str, int | None], ...]] = {
    # event: ((path index, mode, dir_fd index), ...)
    "open": ((0, "follow", None),),
    "os.mkdir": ((0, "link", 2),),
    "os.rename": ((0, "link", 2), (1, "link", 3)),
    "os.remove": ((0, "link", 1),),
    "os.rmdir": ((0, "link", 1),),
    "os.symlink": ((1, "link", 2),),
    "os.link": ((1, "link", 3),),
    "os.truncate": ((0, "follow", None),),
    "os.chmod": ((0, "follow", 2),),
    "os.chown": ((0, "follow", 3),),
    "os.utime": ((0, "follow", 3),),
    "os.setxattr": ((0, "follow", None),),
    "os.removexattr": ((0, "follow", None),),
    "shutil.copyfile": ((1, "follow", None),),
    "shutil.copytree": ((1, "link", None),),
    "shutil.rmtree": ((0, "link", 1),),
    "shutil.move": ((0, "link", None), (1, "link", None)),
}

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_guard = threading.local()


def _is_write_open(args) -> bool:
    mode = args[1] if len(args) > 1 else None
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    flags = args[2] if len(args) > 2 else 0
    return bool(isinstance(flags, int) and flags & _WRITE_FLAGS)


def _as_path(value, dir_fd) -> str | None:
    if isinstance(value, int) or value is None:
        return None  # an open file descriptor: its path was checked when opened
    try:
        text = os.fsdecode(os.fspath(value))
    except TypeError:
        return None
    if not os.path.isabs(text) and isinstance(dir_fd, int):
        try:
            text = os.path.join(os.readlink(f"/proc/self/fd/{dir_fd}"), text)
        except OSError:
            pass
    return os.path.abspath(text)


def _target(logical: str, mode: str) -> str:
    if mode == "follow":
        return _real(logical)
    parent, name = os.path.split(logical.rstrip(os.sep) or os.sep)
    return os.path.join(_real(parent), name)


def _audit(event, args):
    spec = _EVENTS.get(event)
    if spec is None or getattr(_guard, "busy", False):
        return
    if event == "open" and not _is_write_open(args):
        return
    _guard.busy = True
    try:
        if event == "os.link":
            # A hard link to a read-only file makes it writable under a new
            # name that this guard would then pass.
            source = _as_path(args[0], args[2] if len(args) > 2 else None)
            if source:
                problem = _violation(_real(source), source)
                if problem:
                    raise ReadOnlyRootError(f"{event}: hard link to read-only content: {problem}")
        for index, mode, fd_index in spec:
            if index >= len(args):
                continue
            dir_fd = args[fd_index] if fd_index is not None and fd_index < len(args) else None
            logical = _as_path(args[index], dir_fd)
            if logical is None:
                continue
            problem = _violation(_target(logical, mode), logical)
            if problem:
                raise ReadOnlyRootError(f"{event} refused: {problem}")
    finally:
        _guard.busy = False


def _install_write_guard() -> None:
    # Audit hooks cannot be removed, and this module can be imported under two
    # names (src layout vs installed); one hook per process is enough.
    if getattr(sys, "_comak_write_guard", False):
        return
    sys.addaudithook(_audit)
    sys._comak_write_guard = True


_install_write_guard()


def __getattr__(name: str) -> Path:
    """Module-level stage constants, resolved lazily.

    Lazy so that merely importing this module -- which the no-absolute-paths
    test and any ``--help`` invocation do -- cannot fail on an unconfigured
    machine. The error surfaces when a path is actually used.
    """
    key = name.lower()
    if key in STAGES:
        return stage_dir(key)
    if name == "DATA_ROOT":
        return data_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
