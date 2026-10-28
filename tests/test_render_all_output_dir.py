"""render_all's summary lists what each figure wrote, including when the renders go outside the repository.

``$COMAK_FIGURE_OUTPUT_DIR`` redirects the renders (``shared.results.output_dir``). A summary that lists every
written file relative to the repository would raise ``ValueError`` after a complete render into a redirected
directory, and the command would exit 1 with nothing wrong.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_render_all():
    spec = importlib.util.spec_from_file_location("render_all", REPO_ROOT / "figures" / "render_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(monkeypatch, output_dir, written):
    render_all = load_render_all()
    monkeypatch.setenv("COMAK_FIGURE_OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(render_all, "discover", lambda patterns: [render_all.FIGURES_DIR / "figureX.py"])
    monkeypatch.setattr(render_all, "run_one", lambda script, out: ("ok", [written], "0.1s"))
    monkeypatch.setattr(sys, "argv", ["render_all.py"])
    return render_all.main()


def test_a_render_outside_the_repository_succeeds_and_lists_absolute_paths(tmp_path, monkeypatch, capsys):
    written = tmp_path / "figureX.png"
    assert render(monkeypatch, tmp_path, written) == 0
    assert f"  {written}\n" in capsys.readouterr().out


def test_a_render_inside_the_repository_is_listed_relative_to_it(tmp_path, monkeypatch, capsys):
    written = REPO_ROOT / "figures" / "output" / "figureX.png"
    assert render(monkeypatch, tmp_path, written) == 0
    assert f"  {Path('figures/output/figureX.png')}\n" in capsys.readouterr().out
