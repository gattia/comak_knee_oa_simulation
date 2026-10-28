"""``run_knee.py reproduce`` writes into the gait folder of the simulations it reproduces.

The reference gait's condition key is ``default`` (``run.gait``), but its simulations live under
``02_simulations/reference/``. A reproduction mirrors that layout, ``02_simulations/_reproduction/reference/``,
not ``_reproduction/default/``. These tests compare existing trees (``--compare-only``);
no solver runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

run_knee = pytest.importorskip("run_knee")

from comak_knee_oa import compare  # noqa: E402

KNEE, KNEE_DIR = "9003316_RIGHT", "9003316_00m_RIGHT"


def _outputs(d: Path) -> None:
    t = np.linspace(0.0, 0.65, 20)
    for rel in ("comak/_force.sto", "comak-inverse-dynamics/inverse-dynamics.sto"):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text("x\nendheader\ntime\tv\n" + "\n".join(f"{x:.6f}\t{x * 2:.6f}" for x in t) + "\n")
    (d / compare.JOINT_MECHANICS).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / compare.JOINT_MECHANICS, "w") as f:
        f["time"] = t
        for c in compare.ROTATIONS + compare.TRANSLATIONS:
            f[f"model/coordinateset/{c}/value"] = np.sin(t)
    (d / "run_record.json").write_text(json.dumps({"environment": {"cpu_model": "CPU A"}}))


@pytest.mark.parametrize("gait_key, folder", [("default", "reference"), ("Subject_121", "Subject_121")])
def test_reproduction_lands_in_the_published_gait_folder(tmp_path, monkeypatch, capsys, gait_key, folder):
    pub, mine = tmp_path / "published", tmp_path / "mine"
    results = pub / "02_simulations" / folder / KNEE_DIR / "comak_pubrun"
    _outputs(results)
    config = {"run": {"condition": "x", "gait": gait_key,
                      "results_dir": f"02_simulations/{folder}/{KNEE_DIR}/comak_{{run}}"},
              "model": {"built_model": "01_models/{run}/{knee}/model.osim"}}
    record = json.loads((results / "run_record.json").read_text())
    (results / "run_record.json").write_text(json.dumps(dict(record, knee=KNEE, config=config)))
    for stage in ("00_inputs", "01_models", "02_simulations", "03_derived", "04_results"):
        (mine / stage).mkdir(parents=True, exist_ok=True)
    (mine / "_run.json").write_text(json.dumps({"run_name": "mine", "read_only_roots": [str(pub)]}))
    monkeypatch.setenv("COMAK_DATA_ROOT", str(mine))
    monkeypatch.delenv("COMAK_READ_ONLY_ROOTS", raising=False)

    expected = mine / "02_simulations" / "_reproduction" / folder / KNEE_DIR / "comak"
    _outputs(expected)
    code = run_knee.main(["reproduce", "--published", str(results), "--knee", KNEE, "--compare-only"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert f"re-run    {expected}" in out and "REPRODUCED" in out
    assert not (mine / "02_simulations" / "_reproduction" / "default").exists()
