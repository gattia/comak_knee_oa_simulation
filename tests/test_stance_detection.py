"""Stance detection must reproduce the runs that already exist.

Stance detection decides the time window every measured-gait simulation is run
over, so a difference here would move every number downstream without any
error: the simulation would still succeed, on a slightly different stride.

Each completed simulation wrote a record of the window it used and how many
stance phases it found. This re-derives both from the vendored ground reaction
forces and requires exact agreement. Not a tolerance: the same arithmetic on
the same file, so anything other than equality means the logic changed.

Skips when there is nothing to compare against, as on a fresh clone, and says
so in the skip reason: a test that passes because it found no data reads as
coverage it does not have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from comak_knee_oa.config import Knee, load_config  # noqa: E402
from comak_knee_oa.paths import data_root, stage_dir  # noqa: E402

CONFIG_DIR = REPO_ROOT / "config" / "measured_gait"
MAX_KNEES_PER_GAIT = 3  # enough to catch a logic change; this reads real files


def _records() -> list[tuple[Path, Knee, Path]]:
    """(config, knee, run record) for completed measured-gait simulations."""
    try:
        data_root()
    except FileNotFoundError:
        return []

    found = []
    for config_path in sorted(CONFIG_DIR.glob("*.json")):
        gait = json.loads(config_path.read_text())["run"]["gait"]
        gait_dir = stage_dir("simulations") / gait
        if not gait_dir.is_dir():
            continue
        taken = 0
        for knee_dir in sorted(gait_dir.iterdir()):
            if taken >= MAX_KNEES_PER_GAIT:
                break
            records = sorted(knee_dir.glob("*/run_config.json"))
            if not records:
                continue
            found.append((config_path, Knee.parse(knee_dir.name), records[0]))
            taken += 1
    return found


RECORDS = _records()


@pytest.mark.skipif(not RECORDS, reason="no completed simulations to compare against")
@pytest.mark.parametrize(
    "config_path,knee,record_path",
    RECORDS,
    ids=[f"{c.stem}:{k.name}" for c, k, _ in RECORDS],
)
def test_stance_detection_reproduces_the_recorded_window(config_path, knee, record_path):
    from simulate_gait import stance_windows, timing_from_stance

    record = json.loads(record_path.read_text())
    if "derived_timing" not in record:
        pytest.skip("record predates timing being written down")

    config = load_config(config_path).for_knee(knee)
    section = "measured_gait"
    spans = stance_windows(
        config.path(section, "grf"),
        config.get(section, "grf_vertical_column", "ground_force_calcn_r_vy"),
        config.get(section, "force_threshold_N", 10.0),
        config.get(section, "min_stance_frames", 40),
    )
    timing = timing_from_stance(
        spans,
        config.get(section, "stance_index", 0),
        config.get(section, "ik_pad_s", 0.1),
    )

    assert len(spans) == record["n_stance_windows"], (
        f"found {len(spans)} stance phases, the run recorded "
        f"{record['n_stance_windows']} -- detection has changed"
    )
    expected = record["derived_timing"]
    for key in (
        "start_time_comak",
        "stop_time_comak",
        "start_time_ik",
        "stop_time_ik",
        "stance_n_frames",
        "stance_peak_force_N",
    ):
        assert timing[key] == pytest.approx(expected[key], abs=0, rel=1e-12), (
            f"{key}: re-derived {timing[key]}, the run used {expected[key]}"
        )


def test_there_is_something_to_compare_against():
    """Report, rather than hide, the case where this suite checked nothing."""
    if not RECORDS:
        pytest.skip(
            "no completed measured-gait simulations under the data root -- this "
            "suite verified nothing on this machine"
        )
    assert RECORDS


def test_storage_reader_handles_the_reference_trial_too():
    """Both storage-file dialects.

    The measured trials are written as bare tab-separated columns; the reference
    trial is column-aligned with padding. A reader that assumed the first could
    not open the second, so stance would not be detected on the reference trial
    and its time window would go unchecked against the ground reaction force.
    """
    from simulate_gait import read_storage, stance_windows

    try:
        reference = (
            stage_dir("inputs", "reference")
            / "data/reference_data/motion_analysis/overground_17_grf.mot"
        )
    except FileNotFoundError:
        pytest.skip("no data root")
    if not reference.is_file():
        pytest.skip("reference gait not vendored")

    columns, values = read_storage(reference)
    assert columns[0] == "time", f"first column read as {columns[0]!r}"
    assert values.shape[1] == len(columns)

    # Force plate 2 carries the right foot for this trial.
    spans = stance_windows(reference, "ground_force_2_vy", 10.0, 40)
    assert len(spans) == 1, f"expected one right-foot stance, found {len(spans)}"
