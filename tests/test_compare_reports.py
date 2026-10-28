"""What ``compare sims`` prints must agree with what it found.

Two cases where a report could mislead:

- An older run record that does not list MKL libraries, though its run loaded MKL. Compared with a newer record,
  each MKL library must read as "not recorded" there, not as an environment difference beside a BIT-IDENTICAL
  verdict.
- Two runs on different time grids are NOT COMPARABLE. A dataset of another shape is not compared, and the
  per-file line must say so rather than report ``max abs 0.000e+00`` with every dataset as ``nan``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comak_knee_oa import compare  # noqa: E402

N = 20
OPENBLAS = {"libopenblasp-r0.3.21.so": {"path": "/x", "md5": "a" * 32}}
MKL = {"libmkl_core.so.2": {"path": "/y", "md5": "c" * 32}, "libmkl_rt.so.2": {"path": "/y", "md5": "d" * 32}}


def _write_sim(d: Path, environment: dict, frames: int = N, extra_static: bool = False) -> Path:
    t = np.linspace(0.0, 0.65, frames)
    for rel in ("comak/_force.sto", "comak-inverse-dynamics/inverse-dynamics.sto"):
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("name\nendheader\ntime\tvalue\n" + "\n".join(f"{x:.6f}\t{np.sin(x):.9f}" for x in t) + "\n")
    h5 = d / compare.JOINT_MECHANICS
    h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5, "w") as f:
        f["time"] = t
        for c in compare.ROTATIONS + compare.TRANSLATIONS:
            f[f"model/coordinateset/{c}/value"] = np.sin(t)
        if extra_static:
            f["model/constant"] = np.arange(3.0)
    (d / "run_record.json").write_text(json.dumps({"environment": environment}))
    return d


def _env(libraries: dict | None, cpu: str = "CPU A", new_format: bool = True) -> dict:
    env = {"cpu_model": cpu}
    if libraries is not None:
        env["native_libraries"] = libraries
    if new_format:
        env["pins_accept"] = None
    return env


# ------------------------------------------------------- environment (F4) ----


def test_mkl_missing_from_an_older_record_is_not_a_difference(tmp_path):
    published = _write_sim(tmp_path / "pub", _env(OPENBLAS, new_format=False))   # no MKL, no pins_accept
    rerun = _write_sim(tmp_path / "rerun", _env({**OPENBLAS, **MKL}))
    c = compare.compare_simulations(published, rerun)
    assert c.verdict == "BIT-IDENTICAL"
    assert c.environment == []
    assert sorted(line.split(":")[0] for line in c.unrecorded) == sorted(MKL)
    text = compare.report_simulation(c)
    assert "not by the published one" not in text
    assert "libmkl_core.so.2: not recorded in the published run" in text


def test_mkl_missing_from_a_record_that_records_it_is_a_difference(tmp_path):
    published = _write_sim(tmp_path / "pub", _env(OPENBLAS))                     # records MKL, loaded none
    rerun = _write_sim(tmp_path / "rerun", _env({**OPENBLAS, **MKL}))
    c = compare.compare_simulations(published, rerun)
    assert c.unrecorded == []
    assert any(d.startswith("libmkl_core.so.2: loaded by this run") for d in c.environment)


def test_a_record_listing_some_mkl_records_mkl(tmp_path):
    older = _env({**OPENBLAS, "libmkl_rt.so.2": MKL["libmkl_rt.so.2"]}, new_format=False)
    c = compare.compare_simulations(_write_sim(tmp_path / "pub", older),
                                    _write_sim(tmp_path / "rerun", _env({**OPENBLAS, **MKL})))
    assert any(d.startswith("libmkl_core.so.2: loaded by this run") for d in c.environment)


def test_a_record_without_native_libraries_records_none(tmp_path):
    c = compare.compare_simulations(_write_sim(tmp_path / "pub", _env(None, new_format=False)),
                                    _write_sim(tmp_path / "rerun", _env({**OPENBLAS, **MKL})))
    assert c.environment == [] and len(c.unrecorded) == 3


def test_the_newer_record_missing_a_library_reads_the_other_way(tmp_path):
    c = compare.compare_simulations(_write_sim(tmp_path / "pub", _env({**OPENBLAS, **MKL})),
                                    _write_sim(tmp_path / "rerun", _env(OPENBLAS, new_format=False)))
    assert c.environment == []
    assert all("not recorded in this run's record" in line for line in c.unrecorded)


def test_a_changed_md5_is_still_a_difference(tmp_path):
    changed = {"libopenblasp-r0.3.21.so": {"path": "/x", "md5": "b" * 32}}
    c = compare.compare_simulations(_write_sim(tmp_path / "pub", _env(OPENBLAS, new_format=False)),
                                    _write_sim(tmp_path / "rerun", _env({**changed, **MKL})))
    assert c.environment == [f"libopenblasp-r0.3.21.so: md5 {'a' * 32} published, {'b' * 32} this run"]
    assert len(c.unrecorded) == 2


# ---------------------------------------------------- per-file lines (F6) ----


def test_a_different_time_grid_reports_shapes_not_zero_differences(tmp_path):
    published = _write_sim(tmp_path / "pub", _env(OPENBLAS))
    rerun = _write_sim(tmp_path / "rerun", _env(OPENBLAS), frames=N + 1)
    c = compare.compare_simulations(published, rerun)
    assert c.verdict == "NOT COMPARABLE"
    h5 = next(f for f in c.files if f.relative_path == compare.JOINT_MECHANICS)
    assert not h5.ok and np.isnan(h5.max_abs)
    n = 1 + len(compare.ROTATIONS + compare.TRANSLATIONS)
    assert h5.detail == f"{n} datasets: {n} differ in shape (e.g. time ({N},) vs ({N + 1},)); none compared"
    text = compare.report_simulation(c)
    h5_line = next(line for line in text.splitlines() if compare.JOINT_MECHANICS in line and "DIFFER" in line)
    assert "max abs" not in h5_line
    assert "nan" not in text.split("CPU:")[0]
    assert f"model/coordinateset/knee_flex_r/value: shape ({N},) vs ({N + 1},)" in text
    sto = next(f for f in c.files if f.relative_path == "comak/_force.sto")
    assert sto.detail.startswith("shape differs") and "values not compared" in sto.detail


def test_datasets_of_equal_shape_are_still_compared(tmp_path):
    published = _write_sim(tmp_path / "pub", _env(OPENBLAS), extra_static=True)
    rerun = _write_sim(tmp_path / "rerun", _env(OPENBLAS), frames=N + 1, extra_static=True)
    h5 = next(f for f in compare.compare_simulations(published, rerun).files
              if f.relative_path == compare.JOINT_MECHANICS)
    assert h5.max_abs == 0.0
    assert h5.detail.endswith("; 1 compared, 0 of them differ in value")


def test_identical_files_read_as_before(tmp_path):
    c = compare.compare_simulations(_write_sim(tmp_path / "pub", _env(OPENBLAS)),
                                    _write_sim(tmp_path / "rerun", _env(OPENBLAS)))
    h5 = next(f for f in c.files if f.relative_path == compare.JOINT_MECHANICS)
    assert h5.ok and h5.max_abs == 0.0 and h5.detail.endswith(": 0 differ in value")
    assert "max abs 0.000e+00" in compare.report(c.files)
