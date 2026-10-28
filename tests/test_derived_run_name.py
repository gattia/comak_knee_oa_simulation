"""Derived stages name and reuse their outputs by the data root's run.

* The quality-control era is the root's run: on a declared root it defaults to
  that run and refuses any other; on the v2 root (no run) it must be named,
  and only ``v2`` is accepted. The figures' QC module follows the same rule, so
  a root with any declared run name (``v3_g1``) is readable without an edit.
* The pressure transfer reuses a knee-bone's output only when its provenance
  record shows the same inputs. An output with no record -- an as-run or copied
  file -- stops the run, as does one made from other inputs.
* The pressure track reads geometry from the root's own models run.

Everything runs on miniature roots under tmp_path. CPU only; seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_REPO / "figures"))

import derive_pressure  # noqa: E402
import derive_qc  # noqa: E402
from comak_knee_oa import provenance  # noqa: E402
from comak_knee_oa.config import Knee  # noqa: E402
# Imported here, under the session's data root: the module fixes ERA when first
# imported, and a test's scratch root must not be what it remembers.
from shared import qc  # noqa: E402

KNEE = Knee.parse("9000001_RIGHT")


@pytest.fixture
def declared(tmp_path, monkeypatch):
    root = tmp_path / "declared"
    (root / "03_derived").mkdir(parents=True)
    (root / "_run.json").write_text(json.dumps({"run_name": "v3_g1"}))
    monkeypatch.setenv("COMAK_DATA_ROOT", str(root))
    monkeypatch.delenv("COMAK_QC_ERA", raising=False)
    return root


@pytest.fixture
def undeclared(tmp_path, monkeypatch):
    root = tmp_path / "undeclared"
    (root / "03_derived").mkdir(parents=True)
    monkeypatch.setenv("COMAK_DATA_ROOT", str(root))
    monkeypatch.delenv("COMAK_QC_ERA", raising=False)
    return root


# ------------------------------------------------------------------ era ----


def test_era_on_a_declared_root_is_its_run(declared):
    assert derive_qc.resolve_era(None) == "v3_g1"
    assert derive_qc.resolve_era("v3_g1") == "v3_g1"
    with pytest.raises(SystemExit, match="holds run 'v3_g1'"):
        derive_qc.resolve_era("v3")
    with pytest.raises(SystemExit, match="holds run 'v3_g1'"):
        derive_qc.resolve_era("v2")


def test_era_on_the_published_root_must_be_named(undeclared):
    assert derive_qc.resolve_era("v2") == "v2"
    with pytest.raises(SystemExit, match="Pass --era v2"):
        derive_qc.resolve_era(None)
    with pytest.raises(SystemExit, match="Pass --era v2"):
        derive_qc.resolve_era("v3")


def test_qc_keys_carry_the_run(declared):
    assert derive_qc.qc_key("Subject_102", "v3_g1") == "Subject_102_v3_g1"


def test_figures_qc_era_follows_the_root(declared, monkeypatch):
    assert qc._resolve_era() == "v3_g1"
    monkeypatch.setenv("COMAK_QC_ERA", "v3_g1")
    assert qc._resolve_era() == "v3_g1"
    monkeypatch.setenv("COMAK_QC_ERA", "v2")
    with pytest.raises(ValueError, match="another run's table"):
        qc._resolve_era()


def test_figures_qc_era_on_the_published_root(undeclared, monkeypatch):
    assert qc._resolve_era() == qc.CANONICAL_ERA == "v2"
    monkeypatch.setenv("COMAK_QC_ERA", "v1")
    assert qc._resolve_era() == "v1"
    monkeypatch.setenv("COMAK_QC_ERA", "v3")
    with pytest.raises(ValueError, match="own run's root"):
        qc._resolve_era()


# ------------------------------------------------------------- models run ----


def test_pressure_track_reads_the_roots_own_models(declared):
    assert derive_pressure.models_run() == "v3_g1"
    path = derive_pressure.articular_surface_path(KNEE, "femur")
    assert path.is_relative_to(declared / "01_models" / "v3_g1" / KNEE.directory)


def test_published_root_reads_its_own_models(undeclared):
    assert derive_pressure.models_run() == "OARSI_May.22.2026_Full_run"


# ------------------------------------------------------- transfer reuse ----


def _fingerprint(tmp: Path, value: str) -> dict:
    source = tmp / "pressure_stats.vtk"
    source.write_text(value)
    return provenance.fingerprint(inputs={"pressure_stats": source},
                                  options={"T_reg_to_647": [[1.0]]}, code=())


def _write_outputs(output_dir: Path) -> None:
    for path in derive_pressure.transfer_outputs(output_dir, KNEE, "femur"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("transferred")


def test_transfer_reuses_only_its_own_matching_output(declared, tmp_path):
    output_dir = declared / "03_derived" / "pressure_on_reference_Subject_102"
    unit = derive_pressure.transfer_record_unit(output_dir, KNEE, "femur")
    expected = _fingerprint(tmp_path, "a")

    # Nothing on disk: build, without hashing anything.
    assert derive_pressure.transfer_decision(
        output_dir, KNEE, "femur", lambda: pytest.fail("hashed for nothing"), False) == "build"

    # An output with no record (an as-run file) is refused, or rebuilt on --force.
    _write_outputs(output_dir)
    with pytest.raises(provenance.ProvenanceMismatch, match="carries no record"):
        derive_pressure.transfer_decision(output_dir, KNEE, "femur", expected, False)
    assert derive_pressure.transfer_decision(output_dir, KNEE, "femur", expected, True) == "build"

    # Its own complete output, from the same inputs: skipped.
    provenance.mark_complete(unit, expected)
    assert derive_pressure.transfer_decision(output_dir, KNEE, "femur", expected, False) == "skip"

    # Made from other inputs: refused, or rebuilt on --force.
    changed = _fingerprint(tmp_path, "b")
    with pytest.raises(provenance.ProvenanceMismatch, match="pressure_stats"):
        derive_pressure.transfer_decision(output_dir, KNEE, "femur", changed, False)
    assert derive_pressure.transfer_decision(output_dir, KNEE, "femur", changed, True) == "build"

    # Its own interrupted attempt: built again.
    provenance.mark_started(unit, expected)
    assert derive_pressure.transfer_decision(output_dir, KNEE, "femur", expected, False) == "build"


def test_transfer_records_live_beside_the_layout_readers_glob(declared):
    output_dir = declared / "03_derived" / "pressure_on_reference_Subject_102"
    unit = derive_pressure.transfer_record_unit(output_dir, KNEE, "femur")
    assert unit.parent.parent == output_dir / "_provenance"
    vtk, npz = derive_pressure.transfer_outputs(output_dir, KNEE, "femur")
    assert vtk.parent == output_dir / "femur" and vtk.name.endswith("_femur_pressure_ref.vtk")
    assert np.array_equal(sorted([vtk.suffix, npz.suffix]), [".npz", ".vtk"])
