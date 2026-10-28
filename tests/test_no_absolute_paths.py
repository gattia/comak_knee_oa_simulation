"""No absolute paths outside ``comak_knee_oa.paths``.

Everything this pipeline reads lives in one versioned data root, and every
path is resolved against it in one module. A hardcoded absolute path
anywhere else would reach outside that root -- an input nothing declares,
hashes, or controls -- so this test keeps them out.

``paths.py`` is the single exception: deciding where the data lives is its job.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The scanned tree. Anything outside this (docs, notebooks) may mention real
# paths in prose without breaking reproduction.
SCANNED = ("src", "figures", "pipeline", "tests")

# Only paths.py may name a location. Its own defaults are relative to the repo.
EXEMPT = {REPO_ROOT / "src" / "comak_knee_oa" / "paths.py"}

# Absolute roots that would tie the code to one machine or one person's account.
# Written to catch the *shape* of the mistake, not one site's directory names:
# any absolute POSIX path with at least two components, plus Windows drives.
_PATTERNS = {
    "absolute POSIX path": re.compile(
        r"""['"](/(?!tmp/|dev/|proc/|usr/|etc/|opt/|bin/)[A-Za-z0-9._-]+/[A-Za-z0-9._/-]+)['"]"""
    ),
    "home directory": re.compile(r"""['"]~?/home/[A-Za-z0-9._/-]+['"]"""),
    "Windows drive": re.compile(r"""['"][A-Za-z]:[\\/][A-Za-z0-9._\\/-]+['"]"""),
}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for top in SCANNED:
        root = REPO_ROOT / top
        if root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
    return [f for f in files if f not in EXEMPT]


def _offences(path: Path) -> list[str]:
    # Read as UTF-8 explicitly rather than at whatever encoding the ambient
    # locale reports. VTK resets the process's C locale to "C" when it reads a
    # mesh, so once any earlier test has touched the geometry this would decode
    # as ASCII and raise on every source file containing a degree sign, an em
    # dash or a superscript. That made the whole suite fail on eleven files
    # while each one passed in isolation, which reads as eleven broken figures
    # rather than one missing keyword argument.
    found: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        code = line.split("#", 1)[0]
        if not code.strip():
            continue
        for label, pattern in _PATTERNS.items():
            for match in pattern.finditer(code):
                found.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: {label}: "
                    f"{match.group(0)}"
                )
    return found


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_file_has_no_absolute_paths(path: Path) -> None:
    offences = _offences(path)
    assert not offences, (
        "Hardcoded absolute path(s):\n  "
        + "\n  ".join(offences)
        + "\n\nResolve data locations through comak_knee_oa.paths.stage_dir() "
        "instead. If this genuinely needs a machine-specific location, it "
        "belongs in the gitignored data_root.txt, not in the source."
    )


def test_scan_actually_covers_something() -> None:
    """Guard against the lint passing because it scanned nothing.

    A test that vacuously passes is worse than no test -- it reads as coverage.
    """
    assert _python_files(), f"no Python files found under {SCANNED} in {REPO_ROOT}"
