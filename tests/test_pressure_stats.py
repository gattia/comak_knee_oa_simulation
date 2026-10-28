"""The per-vertex pressure summary reproduces the stored one.

``derive_pressure.py stats`` collapses a simulation's per-frame contact
surfaces into ``{bone}_baseline_pressure_stats.vtk``. Here one stored
simulation's joint-mechanics output is linked into a scratch results
directory, re-summarised there, and compared against the stored file array
for array. Agreement is exact, not approximate: the arithmetic is the same as
the stored file's, so any difference means the logic changed.

The scratch directory keeps the data root read-only: the summary is written
beside the simulation, and the simulation here is a link. The simulation is
found through ``cohorts.condition_config``, which on the v2 root (no run
declared) gives each condition's stored location.

CPU only; about ten seconds per knee.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import derive_pressure  # noqa: E402
from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import Knee  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402

CONDITIONS = {"reference": cohorts.REFERENCE, "measured": "Subject_102"}


def _published_simulation(condition: str):
    """(knee, results_dir) of the first cohort knee with a published summary."""
    config = cohorts.condition_config(condition)
    knees = json.loads((stage_dir("inputs") / "cohort.json").read_text())["knees"]
    for name in knees:
        knee = Knee.parse(name)
        results_dir = config.for_knee(knee).path("run", "results_dir")
        if all(
            derive_pressure.pressure_stats_path(results_dir, bone).is_file()
            for bone in derive_pressure.TF_PRESSURE_SCALAR
        ):
            return knee, results_dir
    return None


@pytest.mark.parametrize("arm", sorted(CONDITIONS))
def test_pressure_stats_reproduce_published(arm, tmp_path):
    import pyvista as pv

    found = _published_simulation(CONDITIONS[arm])
    if found is None:
        pytest.skip(f"no published {arm} pressure summary in the data root")
    knee, published_dir = found

    scratch = tmp_path / knee.directory
    scratch.mkdir()
    (scratch / "joint-mechanics").symlink_to(published_dir / "joint-mechanics")
    n_frames, time_step = derive_pressure.read_time_base(
        scratch / "joint-mechanics" / "joint_mechanics.h5"
    )

    for bone in derive_pressure.TF_PRESSURE_SCALAR:
        output = derive_pressure.stats_one(scratch, bone, n_frames, time_step)
        assert output.parent == scratch, "the summary must land beside the simulation"
        published = pv.read(derive_pressure.pressure_stats_path(published_dir, bone))
        rebuilt = pv.read(output)
        assert np.array_equal(published.points, rebuilt.points), f"{knee.name} {bone}: points"
        assert np.array_equal(published.faces, rebuilt.faces), f"{knee.name} {bone}: faces"
        assert set(published.point_data) == set(rebuilt.point_data), f"{knee.name} {bone}: arrays"
        for name in published.point_data:
            assert np.array_equal(
                np.asarray(published.point_data[name]), np.asarray(rebuilt.point_data[name])
            ), f"{knee.name} {bone}: {name} differs"


def test_frame_count_mismatch_raises(tmp_path):
    """A summary over a partial set of frames must fail, not average over fewer."""
    found = _published_simulation(CONDITIONS["reference"])
    if found is None:
        pytest.skip("no published reference pressure summary in the data root")
    _, published_dir = found
    scratch = tmp_path / "knee"
    scratch.mkdir()
    (scratch / "joint-mechanics").symlink_to(published_dir / "joint-mechanics")
    n_frames, time_step = derive_pressure.read_time_base(
        scratch / "joint-mechanics" / "joint_mechanics.h5"
    )
    with pytest.raises(ValueError, match="frames"):
        derive_pressure.stats_one(scratch, "tibia", n_frames + 1, time_step)
