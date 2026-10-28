"""`run_addbiomechanics.py stage` takes your own trial files as well as the provider's zips.

Both routes must stage the same run folder, so a new subject's AB run is the run the
paper's donors had. No real data: a tiny fake trial, staged both ways.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import run_addbiomechanics as rab  # noqa: E402

DONOR = "Subject_121"  # any donor with an `addbiomechanics` section in its config


@pytest.fixture
def trial(tmp_path):
    trial_name = rab.subject_entries(DONOR)["trial"]
    markers = b"PathFileType\t4\t(X/Y/Z)\tmarkers.trc\nfake marker rows\n"
    grf = b"grf\nversion=1\nfake force rows\n"
    own = tmp_path / "own"
    own.mkdir()
    (own / "walk.trc").write_bytes(markers)
    (own / "walk_grf.mot").write_bytes(grf)
    zpath = tmp_path / "provider-001.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(f"OA_GaitRetrainingAddB/{DONOR}/trials/{trial_name}/markers.trc", markers)
        zf.writestr(f"OA_GaitRetrainingAddB/{DONOR}/trials/{trial_name}/grf.mot", grf)
    skeleton = tmp_path / "skeleton.osim"
    skeleton.write_text("<OpenSimDocument/>")
    return own, zpath, skeleton


def _staged(folder: Path) -> dict:
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name != "ab_stage_record.json"}


def test_own_files_stage_the_same_folder_as_the_provider_zips(trial, tmp_path):
    own, zpath, skeleton = trial
    a = rab.stage(DONOR, skeleton, tmp_path / "from_zips", zips=str(zpath))
    b = rab.stage(DONOR, skeleton, tmp_path / "from_files",
                  markers=own / "walk.trc", grf=own / "walk_grf.mot")
    assert _staged(a) == _staged(b)
    rec = json.loads((b / "ab_stage_record.json").read_text())
    assert rec["trial"]["files"]["markers.trc"]["file"] == str((own / "walk.trc").resolve())
    assert "zip" not in rec["trial"]["files"]["markers.trc"]
    zrec = json.loads((a / "ab_stage_record.json").read_text())
    for name in ("markers.trc", "grf.mot"):
        assert rec["trial"]["files"][name]["md5"] == zrec["trial"]["files"][name]["md5"]


@pytest.mark.parametrize("kwargs", [
    {},                                                    # neither
    {"zips": "x.zip", "markers": Path("a.trc"), "grf": Path("b.mot")},  # both
    {"markers": Path("a.trc")},                            # markers without grf
])
def test_stage_needs_exactly_one_source(trial, tmp_path, kwargs):
    _, _, skeleton = trial
    with pytest.raises(ValueError):
        rab.stage(DONOR, skeleton, tmp_path / "out", **kwargs)


def test_missing_own_file_is_named(trial, tmp_path):
    own, _, skeleton = trial
    with pytest.raises(FileNotFoundError, match="nope.mot"):
        rab.stage(DONOR, skeleton, tmp_path / "out", markers=own / "walk.trc", grf=own / "nope.mot")
