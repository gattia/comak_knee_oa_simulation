"""The comparison tools: a re-run simulation judged on the paper's quantities, and a rebuilt model file by file.

A re-run on another CPU type is never bit-identical, and an element-wise tolerance cannot tell agreement from a
broken install: a triangle at the edge of a contact patch switching on is a 100 % change in one value. So
``compare_simulations`` reports knee kinematics in degrees and millimetres and contact pressure and force in
percent, and judges them against the largest difference measured between two CPU types. ``compare_models``
compares two builds with their run names masked. Both run here on small synthetic trees.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comak_knee_oa import compare  # noqa: E402

N = 20


def _write_sim(d: Path, *, flex_shift_deg=0.0, ty_shift_m=0.0, pressure_scale=1.0, frames=N,
               patella=(0.0, 0.0, 0.0), cpu="CPU A", blas_md5="a" * 32) -> Path:
    """A results folder with the three compared outputs, a patella decision and a run record."""
    t = np.linspace(0.0, 0.65, frames)
    for rel in ("comak/_force.sto", "comak-inverse-dynamics/inverse-dynamics.sto"):
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"{x:.6f}\t{np.sin(x) * pressure_scale:.9f}" for x in t)
        f.write_text(f"name\nendheader\ntime\tvalue\n{rows}\n")
    h5 = d / compare.JOINT_MECHANICS
    h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5, "w") as f:
        f["time"] = t
        for c in compare.ROTATIONS:
            f[f"model/coordinateset/{c}/value"] = 20 * np.sin(t) + (flex_shift_deg if c == "knee_flex_r" else 0)
        for c in compare.TRANSLATIONS:
            f[f"model/coordinateset/{c}/value"] = 0.001 * np.cos(t) + (ty_shift_m if c == "knee_ty_r" else 0)
        for _, contact, surface in compare.CONTACTS:
            g = f"model/forceset/Smith2018ArticularContactForce/{contact}/{surface}/"
            f[g + "total_mean_pressure"] = 2e6 * (1 + np.sin(3 * t) ** 2) * pressure_scale
            force = np.zeros((frames, 3))
            force[:, 1] = 1000 * (1 + np.sin(2 * t) ** 2) * pressure_scale
            f[g + "total_contact_force"] = force
    patella_file = d / compare.PATELLA_DECISION
    patella_file.parent.mkdir(parents=True, exist_ok=True)
    patella_file.write_text(json.dumps(list(patella)))
    (d / "run_record.json").write_text(json.dumps({"environment": {
        "cpu_model": cpu, "native_libraries": {"libopenblasp-r0.3.21.so": {"path": "/x", "md5": blas_md5}}}}))
    return d


def test_identical_trees_are_bit_identical(tmp_path):
    a, b = _write_sim(tmp_path / "a"), _write_sim(tmp_path / "b")
    c = compare.compare_simulations(a, b)
    assert c.exact and c.verdict == "BIT-IDENTICAL"
    assert c.metrics["knee flexion (deg)"] == 0.0
    assert compare.main(["sims", str(a), str(b)]) == 0


def test_units_are_degrees_and_millimetres(tmp_path):
    a = _write_sim(tmp_path / "a")
    b = _write_sim(tmp_path / "b", flex_shift_deg=0.5, ty_shift_m=0.002)
    c = compare.compare_simulations(a, b)
    assert not c.exact
    assert c.metrics["knee flexion (deg)"] == pytest.approx(0.5)
    assert c.metrics["knee_ty_r"] == pytest.approx(2.0)          # 2 mm, stored in metres
    assert c.metrics["translations (mm)"] == pytest.approx(2.0)
    assert c.metrics["other rotations (deg)"] == 0.0
    assert c.verdict == "WITHIN CROSS-CPU ENVELOPE"


def test_contact_metrics_are_percent_changes(tmp_path):
    a = _write_sim(tmp_path / "a")
    b = _write_sim(tmp_path / "b", pressure_scale=1.01)
    c = compare.compare_simulations(a, b)
    for label, _, _ in compare.CONTACTS:
        for metric in compare.CONTACT_METRICS:
            assert c.metrics[f"{label}: {metric}"] == pytest.approx(1.0)


def test_beyond_the_envelope_is_flagged(tmp_path):
    a = _write_sim(tmp_path / "a")
    b = _write_sim(tmp_path / "b", pressure_scale=1.10)   # 10 %: beyond every tibiofemoral envelope
    c = compare.compare_simulations(a, b)
    assert c.verdict == "BEYOND CROSS-CPU ENVELOPE"
    assert "tibiofemoral (tibia): mean_pressure" in c.beyond
    assert "<- beyond" in compare.report_simulation(c)
    assert compare.main(["sims", str(a), str(b)]) == 1


def test_a_patella_decision_flip_is_reported_apart(tmp_path):
    a = _write_sim(tmp_path / "a")
    b = _write_sim(tmp_path / "b", flex_shift_deg=0.1, patella=(0.0, 0.004, 0.0))
    c = compare.compare_simulations(a, b)
    assert c.patella_flipped and c.verdict == "PATELLA DECISION DIFFERS"
    assert "DIFFERENT DECISION" in compare.report_simulation(c)


def test_different_time_grids_are_not_compared(tmp_path):
    a, b = _write_sim(tmp_path / "a"), _write_sim(tmp_path / "b", frames=N + 1)
    c = compare.compare_simulations(a, b)
    assert c.verdict == "NOT COMPARABLE" and "comak_window_pad_s" in c.problem


def test_environment_differences_name_cpu_and_libraries(tmp_path):
    a = _write_sim(tmp_path / "a")
    b = _write_sim(tmp_path / "b", flex_shift_deg=0.1, cpu="CPU B", blas_md5="b" * 32)
    c = compare.compare_simulations(a, b)
    assert any(d.startswith("CPU model") for d in c.environment)
    assert any("libopenblasp-r0.3.21.so" in d and "b" * 32 in d for d in c.environment)
    same_cpu = compare.compare_simulations(a, _write_sim(tmp_path / "c", flex_shift_deg=0.1, blas_md5="b" * 32))
    assert "different native libraries" in compare.report_simulation(same_cpu)


def test_envelope_covers_every_judged_metric():
    keys = {"knee flexion (deg)", "other rotations (deg)", "translations (mm)"}
    keys |= {f"{label}: {m}" for label, _, _ in compare.CONTACTS for m in compare.CONTACT_METRICS}
    assert set(compare.CROSS_CPU_ENVELOPE) == keys
    assert all(v > 0 for v in compare.CROSS_CPU_ENVELOPE.values())


# ------------------------------------------------------------------ models ----


def _model_tree(root: Path, run: str, released: Path, slack="0.0425", extra_digit="") -> Path:
    knee = root / "01_models" / run / "9003316_00m_RIGHT"
    model = knee / "custom" / f"knee_{run}"
    model.mkdir(parents=True)
    (model / f"knee_{run}.osim").write_text(
        f'<Model name="knee_{run}"><slack_length>{slack}</slack_length>'
        f'<mesh_file>../Geometry/femur.vtp</mesh_file><note>version v30 of {run}</note></Model>\n')
    (knee / "labels.vtk").write_text(f"POINTS 2 float\n0.1 0.2{extra_digit} 0.3\n")
    (knee / "run_record.json").write_text(json.dumps({"built_in": str(root)}))
    # A link to a released file, through a different relative path in each root.
    (knee / "fit.npy").symlink_to(os.path.relpath(released, knee))
    return knee


def test_models_identical_apart_from_the_run_name(tmp_path):
    released = tmp_path / "released" / "fit.npy"
    released.parent.mkdir()
    released.write_bytes(b"\x01\x02")
    a = _model_tree(tmp_path / "pub", "v3", released)
    b = _model_tree(tmp_path / "deeper" / "mine", "cleanroom", released)
    c = compare.compare_models(a, b)
    assert c.same, compare.report_models(c)
    assert c.identical == 2 and c.links_identical == 1
    assert [rel for rel, _ in c.expected] == ["run_record.json"]
    assert compare.main(["models", str(a), str(b)]) == 0


def test_models_report_numeric_only_differences(tmp_path):
    released = tmp_path / "released" / "fit.npy"
    released.parent.mkdir()
    released.write_bytes(b"\x01")
    a = _model_tree(tmp_path / "pub", "v3", released)
    b = _model_tree(tmp_path / "mine", "cleanroom", released, slack="0.04250000000000001", extra_digit="0001")
    c = compare.compare_models(a, b)
    assert not c.same
    details = dict(c.differing)
    assert "numbers only" in details["custom/knee_{run}/knee_{run}.osim"]
    assert "numbers only" in details["labels.vtk"]
    assert compare.main(["models", str(a), str(b)]) == 1


def test_run_name_masking_respects_token_boundaries():
    assert compare._mask("knee_v3.osim v30 abv3 comak_v3/", "v3") == "knee_{run}.osim v30 abv3 comak_{run}/"
    assert compare._mask(b"_v3_g2.osim", "v3_g2") == b"_{run}.osim"
