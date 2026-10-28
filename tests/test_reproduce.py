"""``run_knee.py reproduce`` compares with a published run outside the current root, with that run's own config.

A reader works in a run root of their own and never writes into the downloaded one, so the result it compares
with is in another root: ``--published`` names its results folder, or ``reproduce`` finds the same knee and gait in the
root's baseline under the run the baseline declares. The config defaults to the one recorded with that run,
because a shipped config can change after a run and only the recorded one reproduces it. These tests compare
existing trees (``--compare-only``); no solver runs.

Also here: ``run_addbiomechanics.py knee-factor``'s folder handling (the factor itself needs OpenSim).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

run_knee = pytest.importorskip("run_knee")
import run_addbiomechanics as rab  # noqa: E402

from comak_knee_oa import compare  # noqa: E402

KNEE, KNEE_DIR = "9003316_RIGHT", "9003316_00m_RIGHT"
CONFIG = {
    "run": {"condition": "reference", "gait": "reference", "results_dir": "02_simulations/reference/{knee}/comak_{run}"},
    "model": {"built_model": "01_models/{run}/{knee}/model.osim"},
    "comak_timing": {"comak_window_pad_s": 0.005},
}


def _outputs(d: Path, shift=0.0, cpu="CPU A"):
    t = np.linspace(0.0, 0.65, 20)
    for rel in ("comak/_force.sto", "comak-inverse-dynamics/inverse-dynamics.sto"):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("x\nendheader\ntime\tv\n" + "\n".join(f"{x:.6f}\t{x * 2:.6f}" for x in t) + "\n")
    (d / compare.JOINT_MECHANICS).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / compare.JOINT_MECHANICS, "w") as f:
        f["time"] = t
        for c in compare.ROTATIONS + compare.TRANSLATIONS:
            f[f"model/coordinateset/{c}/value"] = np.sin(t) + shift
    return d


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A published root (run 'pubrun', declared) and the reader's root (run 'mine', baseline = published)."""
    pub, mine = tmp_path / "published", tmp_path / "mine"
    results = pub / "02_simulations" / "reference" / KNEE_DIR / "comak_pubrun"
    _outputs(results)
    config = json.loads(json.dumps(CONFIG).replace("{knee}", KNEE_DIR))
    (results / "run_record.json").write_text(json.dumps(
        {"knee": KNEE, "config": config, "environment": {"cpu_model": "CPU A"}}))
    (pub / "_run.json").write_text(json.dumps({"run_name": "pubrun"}))
    for stage in ("00_inputs", "01_models", "02_simulations", "03_derived", "04_results"):
        (mine / stage).mkdir(parents=True, exist_ok=True)
    (mine / "_run.json").write_text(json.dumps(
        {"run_name": "mine", "baseline": str(pub), "read_only_roots": [str(pub)]}))
    cfg = tmp_path / "reference_gait.json"
    cfg.write_text(json.dumps(dict(CONFIG, comak_timing={"comak_window_pad_s": 0.0})))
    monkeypatch.setenv("COMAK_DATA_ROOT", str(mine))
    monkeypatch.delenv("COMAK_READ_ONLY_ROOTS", raising=False)
    return pub, mine, results, cfg


def _scratch(mine: Path) -> Path:
    return mine / "02_simulations" / "_reproduction" / "reference" / KNEE_DIR / "comak"


def test_published_folder_and_recorded_config(roots, capsys):
    pub, mine, results, cfg = roots
    _outputs(_scratch(mine))
    (_scratch(mine) / "run_record.json").write_text(json.dumps({"environment": {"cpu_model": "CPU A"}}))
    code = run_knee.main(["reproduce", "--published", str(results), "--knee", KNEE, "--compare-only"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "as the published run recorded it" in out and "REPRODUCED" in out


def test_published_found_in_the_baseline(roots, capsys):
    pub, mine, results, cfg = roots
    _outputs(_scratch(mine))
    code = run_knee.main(["reproduce", "--config", str(cfg), "--knee", KNEE, "--compare-only"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert f"published: {results}" in out
    # The shipped config's value differs from the one recorded with that run; the recorded one wins.
    assert "comak_timing.comak_window_pad_s = 0.0; the published run used 0.005" in out


def test_the_baseline_needs_a_config_or_published(roots):
    with pytest.raises(SystemExit):
        run_knee.main(["reproduce", "--knee", KNEE, "--compare-only"])


def test_a_baseline_without_a_run_name_asks_for_published(roots, monkeypatch):
    pub, mine, results, cfg = roots
    monkeypatch.setenv("COMAK_DATA_ROOT", str(pub))   # the reader's root protects the baseline
    (pub / "_run.json").unlink()
    monkeypatch.setenv("COMAK_DATA_ROOT", str(mine))
    with pytest.raises(SystemExit, match="--published"):
        run_knee.main(["reproduce", "--config", str(cfg), "--knee", KNEE, "--compare-only"])


def test_same_cpu_not_identical_is_not_reproduced(roots, capsys):
    pub, mine, results, cfg = roots
    _outputs(_scratch(mine), shift=0.001)
    (_scratch(mine) / "run_record.json").write_text(json.dumps({"environment": {"cpu_model": "CPU A"}}))
    assert run_knee.main(["reproduce", "--published", str(results), "--knee", KNEE, "--compare-only"]) == 1
    assert "NOT REPRODUCED" in capsys.readouterr().out


def test_other_cpu_within_the_envelope_agrees(roots, capsys):
    pub, mine, results, cfg = roots
    _outputs(_scratch(mine), shift=0.001)
    (_scratch(mine) / "run_record.json").write_text(json.dumps({"environment": {"cpu_model": "CPU B"}}))
    assert run_knee.main(["reproduce", "--published", str(results), "--knee", KNEE, "--compare-only"]) == 0
    assert "AGREES" in capsys.readouterr().out


def test_a_record_of_another_knee_is_refused(roots, capsys):
    pub, mine, results, cfg = roots
    record = json.loads((results / "run_record.json").read_text())
    shutil.copytree(results, mine / "elsewhere")
    record["knee"] = "9999999_LEFT"
    (mine / "elsewhere" / "run_record.json").write_text(json.dumps(record))
    assert run_knee.main(["reproduce", "--published", str(mine / "elsewhere"), "--knee", KNEE,
                          "--compare-only"]) == 1


# ------------------------------------------------------------ knee-factor ----


def test_knee_factor_finds_models_in_vendored_and_run_folders(tmp_path):
    for folder in (tmp_path / "vendored" / "Models", tmp_path / "run" / rab.OUTPUT_NAME / "Models"):
        folder.mkdir(parents=True)
        (folder / "match_markers_and_physics.osim").write_text("<OpenSimDocument/>")
    assert rab.models_dir(tmp_path / "vendored") == tmp_path / "vendored" / "Models"
    assert rab.models_dir(tmp_path / "run") == tmp_path / "run" / rab.OUTPUT_NAME / "Models"
    with pytest.raises(FileNotFoundError):
        rab.models_dir(tmp_path)


def test_knee_factor_prints_differences_from_the_first(monkeypatch, capsys):
    values = {"a": (0.98, 0.97), "b": (0.97, 0.96)}
    monkeypatch.setattr(rab, "knee_factor", lambda f: {
        "folder": str(f), "femur_r": values[str(f)][0], "tibia_r": values[str(f)][1],
        "s_wa": sum(values[str(f)]) / 2})
    rab.print_knee_factors(["a", "b"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("s_wa 0.975000000") and "vs the first" not in lines[0]
    assert lines[1].startswith("s_wa 0.965000000") and f"{100 * (0.965 / 0.975 - 1):+.3f} % vs the first" in lines[1]
