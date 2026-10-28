"""Figure 2's derivation refuses a run with too few conditions to ensemble, before writing anything.

Its per-knee ensemble keeps a knee only if the knee was simulated in at least ``MIN_GAITS_FOR_ENSEMBLE`` (five) of
the ensembled conditions. A one-knee test root with two conditions therefore has an empty ensemble, and a script
that extracted and wrote the curves first would crash half-way through the summaries (``IndexError``), leaving a
partial ``04_results/figure2_compartment_mechanics`` behind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "pipeline", "figures"):
    sys.path.insert(0, str(REPO_ROOT / sub))

fig2 = pytest.importorskip("derive_figure2_data")


@pytest.fixture
def root(tmp_path, monkeypatch):
    def make(measured):
        for stage in ("00_inputs", "01_models", "02_simulations", "03_derived", "04_results"):
            (tmp_path / stage).mkdir(exist_ok=True)
        (tmp_path / "_run.json").write_text(json.dumps({"run_name": "t", "measured_gaits": list(measured)}))
        monkeypatch.setenv("COMAK_DATA_ROOT", str(tmp_path))
        monkeypatch.delenv("COMAK_READ_ONLY_ROOTS", raising=False)
        return tmp_path
    return make


def test_a_two_condition_root_is_refused_and_nothing_is_written(root):
    r = root(["Subject_121"])
    with pytest.raises(SystemExit, match=r"needs at least 5 gait conditions .* this run has 2: reference, Subject_121"):
        fig2.derive()
    assert list((r / "04_results").iterdir()) == []


def test_gaits_selecting_too_few_is_refused(root):
    r = root(["Subject_102", "Subject_107", "Subject_109", "Subject_114", "Subject_119", "Subject_121"])
    with pytest.raises(SystemExit, match="this run has 3"):
        fig2.derive(gaits=["reference", "Subject_102", "Subject_107"])
    assert list((r / "04_results").iterdir()) == []


def test_the_measured_ensemble_counts_measured_conditions_only(monkeypatch):
    monkeypatch.setattr(fig2, "measured_gaits", lambda: ["A", "B", "C", "D"])
    conditions = [("reference", "default")] + [(g, g) for g in ("A", "B", "C", "D")]
    fig2.require_enough_conditions(conditions, "all")              # five: enough
    with pytest.raises(SystemExit, match=r"\(--ensemble measured\); this run has 4"):
        fig2.require_enough_conditions(conditions, "measured")
