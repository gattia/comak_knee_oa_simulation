#!/usr/bin/env python3
"""Render every manuscript figure from ``04_results``.

    python figures/render_all.py
    python figures/render_all.py --list
    python figures/render_all.py figure5          # just the ones that match

Each figure is a no-argument entry point; this runs the whole set in one
command and ends with a summary that separates two outcomes:

  missing input   the data is not there: download ``04_results``, or run that
                  figure's derivation.
  failed          the script raised an error.

Figures are discovered by globbing ``figures/figure*.py``, so a new figure
script is included without updating a list.

Rendering only: the derivations (``pipeline/derive_figure*.py``) read the
simulation tree and the mixed models (``figures/figure*_*.R``) need R; both
run separately. See README.md for the entry points into the pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent

# What a script prints when its 04_results input is absent. shared.results
# raises MissingFigureInput with this wording; matching on it lets the summary
# tell "you have not downloaded the data" apart from "this is broken".
MISSING_INPUT_MARKERS = ("MissingFigureInput", "is missing", "derived input")


def discover(patterns: list[str] | None) -> list[Path]:
    """Figure entry points, in figure-number order.

    Sorted by name so the report reads 1, 2, 3 ... (then the Extended Data
    figures) rather than in whatever order the filesystem returns.
    """
    scripts = sorted(p for p in FIGURES_DIR.glob("figure*.py") if p.is_file())
    if patterns:
        scripts = [p for p in scripts if any(pat in p.name for pat in patterns)]
    return scripts


def outputs_snapshot(output_dir: Path) -> dict[Path, float]:
    if not output_dir.is_dir():
        return {}
    return {p: p.stat().st_mtime for p in output_dir.iterdir() if p.is_file()}


def written_since(output_dir: Path, before: dict[Path, float]) -> list[Path]:
    """Files a script created or rewrote, so the report names real artifacts."""
    return sorted(
        p for p, mtime in outputs_snapshot(output_dir).items()
        if before.get(p) != mtime
    )


def run_one(script: Path, output_dir: Path) -> tuple[str, list[Path], str]:
    """Render one figure in its own process -> (status, files written, detail).

    A subprocess rather than an import: the figure scripts set the matplotlib
    backend, insert onto ``sys.path`` and hold module-level state, and one
    figure's setup should not be able to change the next one's. It also means a
    hard crash in one figure costs that figure and not the run.
    """
    before = outputs_snapshot(output_dir)
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    elapsed = time.monotonic() - started
    written = written_since(output_dir, before)

    if completed.returncode == 0:
        return "ok", written, f"{elapsed:.1f}s"

    combined = completed.stdout + completed.stderr
    status = ("missing input"
              if any(marker in combined for marker in MISSING_INPUT_MARKERS)
              else "failed")
    tail = [line for line in combined.strip().splitlines() if line.strip()]
    return status, written, "\n".join(tail[-12:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("patterns", nargs="*",
                        help="substrings of script names to render (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="show what would be rendered, then exit")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(FIGURES_DIR))
    from shared import results  # noqa: E402  - after sys.path is set up

    scripts = discover(args.patterns)
    if not scripts:
        target = " matching " + ", ".join(args.patterns) if args.patterns else ""
        print(f"No figure scripts found in {FIGURES_DIR}{target}.")
        return 1

    if args.list:
        for script in scripts:
            print(script.relative_to(REPO_ROOT))
        return 0

    output_dir = results.output_dir()
    print(f"Rendering {len(scripts)} figure script(s) into {output_dir}\n")

    outcomes: list[tuple[Path, str, list[Path], str]] = []
    for script in scripts:
        print(f"  {script.name} ... ", end="", flush=True)
        status, written, detail = run_one(script, output_dir)
        print(status if status != "ok" else f"ok ({detail})")
        outcomes.append((script, status, written, detail))

    ok = [o for o in outcomes if o[1] == "ok"]
    missing = [o for o in outcomes if o[1] == "missing input"]
    failed = [o for o in outcomes if o[1] == "failed"]

    print(f"\n{len(ok)} rendered, {len(missing)} missing input, {len(failed)} failed")
    for _, _, written, _ in ok:
        for path in written:
            # $COMAK_FIGURE_OUTPUT_DIR may put the renders outside the repository
            shown = path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path
            print(f"  {shown}")

    if missing:
        print("\nMissing 04_results input — download that stage, or run the "
              "derivation each script names:")
        for script, _, _, detail in missing:
            print(f"\n  {script.name}\n" + "\n".join(f"    {l}" for l in detail.splitlines()))

    if failed:
        print("\nFailed:")
        for script, _, _, detail in failed:
            print(f"\n  {script.name}\n" + "\n".join(f"    {l}" for l in detail.splitlines()))

    # Only a script that raised sets a non-zero exit status; missing data does not.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
