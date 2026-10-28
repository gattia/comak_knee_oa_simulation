"""Locating what a figure reads, and failing usefully when it is absent.

Every figure reads from ``04_results/<figure name>/``. Nothing else. A figure
script that reached back into ``02_simulations`` would need the simulation tree
(terabytes) to redraw a plot, and would re-derive its inputs on every run -- so the derivation is a pipeline step and the figure is pure rendering.

The one job here beyond resolving a path is the error message. A figure whose
input is missing must say which pipeline command produces it, because the person
hitting that error is usually reproducing the manuscript rather than developing
the pipeline, and "FileNotFoundError" tells them nothing actionable.
"""

from __future__ import annotations

import os
from pathlib import Path

from comak_knee_oa.paths import stage_dir

__all__ = ["figure_dir", "require", "output_dir"]


class MissingFigureInput(FileNotFoundError):
    """A figure's derived input has not been built yet."""


def figure_dir(name: str, create: bool = False) -> Path:
    """``04_results/<name>/`` -- where one figure's derived inputs live."""
    return stage_dir("results", name, create=create)


def require(name: str, *files: str, produced_by: str) -> list[Path]:
    """Resolve files under ``04_results/<name>/``, or raise naming the fix.

    Parameters
    ----------
    name
        Figure directory under ``04_results``.
    files
        Filenames required, relative to that directory.
    produced_by
        The exact command that builds them. This is the whole point of the
        function: it turns "file not found" into "run this".
    """
    base = figure_dir(name)
    resolved = [base / f for f in files]
    missing = [p for p in resolved if not p.exists()]
    if missing:
        listed = "\n".join(f"    {p}" for p in missing)
        raise MissingFigureInput(
            f"Figure {name!r} is missing {len(missing)} of {len(resolved)} "
            f"derived input(s):\n{listed}\n\n"
            f"Build them with:\n    {produced_by}\n\n"
            "Or download the published 04_results stage, which contains them "
            "already (see README.md)."
        )
    return resolved


def output_dir(create: bool = True) -> Path:
    """Where rendered figures are written: ``<repo>/figures/output/[<run>/]``.

    Inside the repository and gitignored, not in the data root. A rendered
    figure is a *product* of this code, so running the code again should
    replace it; keeping it beside the code makes that obvious, and keeps the
    downloadable data stages free of things the reader can regenerate
    (Figure 1 takes several minutes, the rest seconds).

    On a data root that declares a run (``paths.run_name``), renders go to a
    subfolder named for it, ``figures/output/v3/``; a root that declares none
    renders into ``figures/output/`` itself. So one run's figures sit beside
    another's instead of overwriting them -- the folder is gitignored, so an
    overwrite would leave no history.

    ``$COMAK_FIGURE_OUTPUT_DIR``, if set, redirects the renders elsewhere (e.g.
    trial renders that must not overwrite the existing ones).
    """
    from comak_knee_oa.paths import REPO_ROOT

    override = os.environ.get("COMAK_FIGURE_OUTPUT_DIR")
    if override:
        path = Path(override).expanduser()
    else:
        path = REPO_ROOT / "figures" / "output"
        run = _declared_run()
        if run:
            path = path / run
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _declared_run() -> str | None:
    """The data root's run, or None if it declares none or no root is configured."""
    from comak_knee_oa.paths import declaration, run_name

    try:
        decl = declaration()
    except FileNotFoundError:  # no data root: a schematic needs none
        return None
    return run_name() if decl and decl.get("run_name") else None
