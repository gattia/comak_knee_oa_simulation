"""Stage 03's inputs resolve, and the derivations reproduce the data.

Same principle as ``test_reconstruction_inputs.py``: resolving paths proves a
config points somewhere, so each test here also re-derives a stored number
from the vendored inputs and requires agreement at numerical precision.

- The per-knee cross-model transform is recomputed from the vendored meshes,
  fits, and segmentations, and compared entry-for-entry against the stored
  transform file. This exercises the whole chain, including the left-knee
  reflection pair, and proves the vendored files are the ones the stored
  transforms were computed from.
- The thinning regression is recomputed from the thickness-on-reference
  arrays and compared against the stored thinning-rate maps.
- The cached walks are checked for internal consistency (displacement equals
  positions minus the reference), and one knee's pressure transfer is run
  end to end into a temporary directory.

Everything here is CPU-only.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import derive_pressure  # noqa: E402
import derive_thickness  # noqa: E402
from comak_knee_oa.config import Knee  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402


def _vendored_model_knees() -> list[Knee]:
    """Knees whose stage-01 geometry is present in the data root."""
    knees = []
    for name in derive_thickness.cohort():
        knee = Knee.parse(name)
        if derive_pressure.aligned_mesh_path(knee, "femur").is_file():
            knees.append(knee)
    return knees


# -------------------------------------------------------------- thickness ----


def test_every_cohort_knee_has_a_baseline_fit():
    coverage = derive_thickness.visit_coverage(visits=("V00",))
    missing = sorted(set(derive_thickness.cohort()) - set(coverage))
    assert not missing, f"{len(missing)} knees without a V00 fit: {missing[:5]}"


def test_reference_surfaces_are_the_20k_meshes():
    import pyvista as pv

    for bone in derive_thickness.BONES:
        surface = pv.read(derive_thickness.reference_surface_path(bone))
        assert surface.n_points == 20_000, f"{bone}: {surface.n_points} points"


def test_thinning_regression_reproduces_the_stored_maps():
    """Recompute a sample of thinning-rate maps from their thickness arrays.

    The stored map records how many visits its regression used; the
    recomputation must find the same visits and land on the same slopes.
    """
    from pymskt.mesh import Mesh

    candidates = [
        ("thickness", stage_dir("derived", "thickness")),
        ("thickness_merged", stage_dir("derived", "thickness_merged")),
    ]
    tree = next(
        (root for _, root in candidates if (root / "thinning_rates").is_dir()), None
    )
    if tree is None:
        pytest.skip("no thinning-rate maps in the data root yet")

    def paths_in_tree(knee, visit, bone):
        directory = tree / "thickness_on_reference" / visit
        base = f"{knee}_{bone}_thickness_ref"
        return (
            directory / f"{base}.vtk",
            directory / f"{base}.npy",
            directory / f"{base}_nsm.npy",
        )

    original = derive_thickness.on_reference_paths
    derive_thickness.on_reference_paths = paths_in_tree
    try:
        stored_maps = sorted((tree / "thinning_rates").glob("*_thinning_rate.vtk"))[:4]
        assert stored_maps, "thinning_rates directory is empty"
        for stored in stored_maps:
            knee, bone = stored.stem.replace("_thinning_rate", "").rsplit("_", 1)
            available = [
                visit
                for visit in derive_thickness.VISIT_YEARS
                if paths_in_tree(knee, visit, bone)[1].is_file()
            ]
            recomputed = derive_thickness.thinning_scalars(knee, bone, available)
            assert recomputed is not None, f"{knee} {bone}: fewer than two visits"

            mesh = Mesh(str(stored))
            stored_n = float(np.asarray(mesh.get_scalar("n_timepoints")).max())
            assert stored_n == float(recomputed["n_timepoints"][0]), (
                f"{knee} {bone}: stored map used {stored_n:.0f} visits, "
                f"recomputation found {recomputed['n_timepoints'][0]:.0f}"
            )
            rate = recomputed["thinning_rate (mm/year)"]
            written = np.where(np.isfinite(rate), rate, 0.0)  # maps store NaN as 0
            stored_rate = np.asarray(mesh.get_scalar("thinning_rate (mm/year)"))
            worst = np.abs(written - stored_rate).max()
            assert worst < 1e-9, f"{knee} {bone}: slopes differ by {worst:.3e}"
    finally:
        derive_thickness.on_reference_paths = original


# --------------------------------------------------------------- pressure ----


def test_transforms_reproduce_the_stored_file():
    """Recompute the cross-model transform for every vendored knee, both bones.

    Exact agreement with the stored file proves the vendored segmentation
    meshes, aligned meshes, shape fits, and (for left knees) the segmentation
    headers and flipped surfaces are the same files the stored transforms
    came from -- the whole chain, reflections included.
    """
    stored_file = derive_pressure.transform_file()
    if not stored_file.is_file():
        pytest.skip("no stored transform file in the data root yet")
    knees = _vendored_model_knees()
    if not knees:
        pytest.skip("no stage-01 geometry in the data root yet")

    stored = json.loads(stored_file.read_text())["transforms"]
    checked = 0
    for knee in knees:
        correction = derive_pressure.flip_correction(knee)
        if knee.side == "LEFT":
            assert correction is not None, f"{knee.name}: flip inputs did not resolve"
        for bone in derive_thickness.BONES:
            recorded = stored[bone]["T_reg_to_647"].get(knee.name)
            if recorded is None:
                continue
            T_alignment, source, residual = derive_pressure.recover_alignment(
                derive_pressure.segmentation_mesh_path(knee, bone),
                derive_pressure.aligned_mesh_path(knee, bone),
            )
            assert residual < 1e-6, f"{knee.name} {bone}: alignment residual {residual:.3e}"
            recon = derive_thickness.recon_results_path(knee.name, "V00", bone)
            T_647 = np.array(json.loads(recon.read_text())["icp_transform"])
            if correction is not None:
                T = T_647 @ correction @ np.linalg.inv(T_alignment)
            else:
                T = T_647 @ np.linalg.inv(T_alignment)
            worst = np.abs(T - np.array(recorded)).max()
            assert worst < 1e-9, f"{knee.name} {bone}: transform differs by {worst:.3e}"
            checked += 1
    assert checked, "no vendored knee appears in the stored transform file"


def test_cached_walks_are_internally_consistent():
    import pyvista as pv

    knees = _vendored_model_knees()
    if not knees:
        pytest.skip("no stage-01 geometry in the data root yet")
    for bone in derive_thickness.BONES:
        reference = np.asarray(
            pv.read(derive_thickness.reference_surface_path(bone)).points, dtype=float
        )
        for knee in knees:
            xyz_path = derive_pressure.walked_out_path(knee, bone)
            if not xyz_path.is_file():
                pytest.skip(f"no cached walk for {knee.name} {bone}")
            xyz = np.load(xyz_path)
            displacement = np.load(
                xyz_path.with_name(f"{knee.subject}_{knee.side}_{bone}_displacement.npy")
            )
            assert xyz.shape == (20_000, 3), f"{knee.name} {bone}: {xyz.shape}"
            worst = np.abs((xyz - reference) - displacement).max()
            assert worst < 1e-6, f"{knee.name} {bone}: displacement inconsistent ({worst:.3e})"


def test_pressure_transfer_end_to_end(tmp_path):
    """Run one knee-bone's transfer into a scratch directory and check the contract."""
    from pymskt.mesh import Mesh

    from comak_knee_oa import cohorts

    # Through cohorts, so the v2 root (no run declared) is read where its
    # simulations are, and a run's root where its own are.
    config = cohorts.condition_config("Subject_102")
    transforms_file = derive_pressure.transform_file()
    if not transforms_file.is_file():
        pytest.skip("no stored transform file in the data root yet")
    transforms = json.loads(transforms_file.read_text())["transforms"]

    chosen = None
    for knee in _vendored_model_knees():
        results_dir = config.for_knee(knee).path("run", "results_dir")
        pressure_vtk = results_dir / "femur_baseline_pressure_stats.vtk"
        if (
            pressure_vtk.is_file()
            and derive_pressure.walked_out_path(knee, "femur").is_file()
            and knee.name in transforms["femur"]["T_reg_to_647"]
        ):
            chosen = (knee, pressure_vtk)
            break
    if chosen is None:
        pytest.skip("no vendored knee has pressure statistics yet")

    knee, pressure_vtk = chosen
    output = derive_pressure.transfer_one(
        knee,
        "femur",
        pressure_vtk,
        np.array(transforms["femur"]["T_reg_to_647"][knee.name]),
        derive_pressure.femur_reference_center(),
        Mesh(str(derive_thickness.reference_surface_path("femur"))),
        tmp_path,
    )
    result = Mesh(str(output))
    has_data = np.asarray(result.get_scalar("has_data"))
    assert has_data.shape == (20_000,)
    covered = has_data.mean()
    assert 0.3 < covered <= 1.0, f"implausible coverage: {covered:.2f}"
    for name in ("mean_tf_pressure", "max_tf_pressure", "cumulative_tf_pressure"):
        values = np.asarray(result.get_scalar(name))
        assert values is not None and np.isfinite(values).all(), name
        assert values[has_data > 0.5].max() > 0, f"{name}: no positive values"
