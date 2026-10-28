"""The quality-control detectors reproduce the stored verdicts.

The spatial detector is run over the stored combined matrices into a scratch
data root (inputs linked, outputs real), so the shared table is never written.
Every simulation's verdict must equal the stored table's ``flagged_spatial``.

The time-series detector reads all 7,500 joint-mechanics files, which is too
slow for a test; over the full cohort it reproduces every flag, with tibia
metrics string-identical.

CPU only; about a minute.
"""

import csv
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import derive_qc  # noqa: E402
from comak_knee_oa.paths import data_root  # noqa: E402


def _true(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def test_spatial_detector_reproduces_published(tmp_path, monkeypatch):
    real = data_root()
    published = real / "03_derived" / "qc" / "pressure_outliers_unified_v2.csv"
    combined = real / "03_derived" / "combined"
    if not published.is_file() or not combined.is_dir():
        pytest.skip("no published QC table or combined matrices in the data root")
    with published.open() as handle:
        expected = {(r["cohort_tag"], r["knee"]): _true(r["flagged_spatial"])
                    for r in csv.DictReader(handle)}

    root = tmp_path / "root"
    (root / "03_derived").mkdir(parents=True)
    (root / "00_inputs").symlink_to(real / "00_inputs")
    (root / "03_derived" / "combined").symlink_to(combined)
    monkeypatch.setenv("COMAK_DATA_ROOT", str(root))

    assert derive_qc.main(["spatial", "--era", "v2"]) == 0
    output = root / "03_derived" / "qc" / "pressure_outliers_v2.csv"
    with output.open() as handle:
        rebuilt = {(r["cohort_tag"], r["knee"]): _true(r["is_outlier"])
                   for r in csv.DictReader(handle)}

    assert rebuilt, "the detector wrote no rows"
    missing = set(rebuilt) - set(expected)
    assert not missing, f"{len(missing)} simulations absent from the published table"
    disagree = sorted(k for k in rebuilt if rebuilt[k] != expected[k])
    assert not disagree, f"{len(disagree)} verdicts differ, e.g. {disagree[:3]}"
    assert sum(rebuilt.values()) == sum(expected[k] for k in rebuilt)


def test_sharded_extract_cannot_flag_in_one_step():
    """A shard holds part of each condition, and the MAD thresholds need all of it."""
    with pytest.raises(SystemExit, match="merge"):
        derive_qc.main(["timeseries", "--era", "v2", "--mode", "both", "--n-shards", "4"])
