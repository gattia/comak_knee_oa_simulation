"""The matrix builder admits the knees the stored matrices hold.

The builder admits a knee only when it finds its thinning rates and its
pressure tree under the names on disk; a wrong tree name admits no knee at all.
Each condition's admission check is run here (``--check``, which reads and
writes nothing else) and its count compared with the stored matrix's.

A full rebuild of two conditions reproduces every data array of the stored
matrices bit for bit; it takes ten minutes, so it is not run here.

CPU only; about half a minute.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import combine_reference_data  # noqa: E402
from comak_knee_oa.config import load_config  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402
from derive_pressure import condition_name  # noqa: E402

CONFIGS = [_REPO / "config" / "reference_gait.json"] + sorted(
    (_REPO / "config" / "measured_gait").glob("*.json")
)


@pytest.mark.parametrize("config_file", CONFIGS, ids=lambda p: p.stem)
def test_admission_matches_published_matrix(config_file, capsys):
    condition = condition_name(load_config(config_file))
    published = stage_dir("derived", "combined") / f"combined_data_{condition}.npz"
    if not published.is_file():
        pytest.skip(f"no published matrix for {condition}")
    with np.load(published, allow_pickle=True) as data:
        expected = len(data["subjects"])

    assert combine_reference_data.main(["matrices", "--check", "--config", str(config_file)]) == 0
    match = re.search(rf"{condition}: (\d+) knees complete", capsys.readouterr().out)
    assert match, "the builder did not report its admission count"
    assert int(match.group(1)) == expected
