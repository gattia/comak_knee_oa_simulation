"""Stage 01's inputs resolve, and the vendored inputs and models agree.

Resolving a path proves the config points somewhere, not that it points at the
right thing. So besides existence checks over the whole cohort, this file
re-derives a number that ties ``00_inputs`` to ``01_models``: every aligned
mesh in a knee's model directory must be an exact similarity transform of the
corresponding segmentation mesh in ``00_inputs`` -- the alignment is computed
deterministically from that mesh, before any stochastic fitting. If the
vendored meshes and the vendored models came from different sources, the
least-squares residual is large and the test fails; if they are the true pair,
it is at floating-point precision.

Everything here is CPU-only and runs in seconds.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from comak_knee_oa.config import Knee, load_config  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402
from reconstruct_knee import SHAPE_MODELS, subject_mesh_files  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config" / "reconstruction.json"

# The aligned mesh is the segmentation mesh after one similarity transform, so
# the least-squares recovery should reproduce it to numerical precision. A
# millimetre-scale residual means the two files are not the pair they claim.
ALIGNMENT_RESIDUAL_MM = 1e-6


def _config():
    return load_config(CONFIG)


def _cohort() -> list[str]:
    return json.loads((stage_dir("inputs") / "cohort.json").read_text())["knees"]


def test_reconstruction_inputs_resolve():
    config = _config().for_knee(Knee.parse(_cohort()[0]))
    problems = config.check_inputs(
        ("inputs", "subject_meshes"),
        ("inputs", "reference_meshes"),
        ("inputs", "shape_models"),
        ("inputs", "labeled_bones"),
        ("inputs", "ligament_attachments"),
        ("inputs", "base_model"),
        ("inputs", "smith2019_model"),
    )
    assert not problems, "\n".join(problems)


def test_every_cohort_knee_has_its_meshes():
    """All eight segmentation meshes, for every knee the cohort lists."""
    config = _config()
    missing = []
    for name in _cohort():
        knee = Knee.parse(name)
        subject_dir = config.for_knee(knee).path("inputs", "subject_meshes")
        for filename in subject_mesh_files(knee).values():
            if not (subject_dir / filename).is_file():
                missing.append(str(subject_dir / filename))
    assert not missing, f"{len(missing)} missing meshes, first: {missing[:5]}"


def test_shape_models_are_complete():
    config = _config()
    models_dir = config.path("inputs", "shape_models")
    for bone, model in SHAPE_MODELS.items():
        weights = models_dir / model / "model" / "2000.pth"
        settings = models_dir / model / "model_params_config.json"
        assert weights.is_file(), f"{bone}: {weights}"
        parsed = json.loads(settings.read_text())
        assert "latent_size" in parsed, f"{bone}: no latent_size in {settings}"


def test_base_model_is_exactly_one_osim_with_geometry():
    config = _config()
    base = config.path("inputs", "base_model")
    models = list(base.glob("*.osim"))
    assert len(models) == 1, f"expected one .osim in {base}, found {len(models)}"
    assert (base / "Geometry").is_dir(), f"no Geometry/ beside {models[0]}"


def test_reference_alignment_provides_the_femur_center():
    config = _config()
    alignment = json.loads(
        (config.path("inputs", "reference_meshes") / "femur" / "ref_femur_alignment.json")
        .read_text()
    )
    center = np.asarray(alignment["mean_orig"], dtype=float)
    assert center.shape == (3,) and np.all(np.isfinite(center))


def _vendored_model_knees(config) -> list[Knee]:
    """Knees whose stage-01 output is present in the data root."""
    knees = []
    for name in _cohort():
        knee = Knee.parse(name)
        output_dir = config.for_knee(knee).path("run", "output_dir")
        if (output_dir / "geometries_nsm_similarity").is_dir():
            knees.append(knee)
    return knees


def test_aligned_meshes_are_similarity_transforms_of_the_inputs():
    """Re-derive the alignment linking each vendored model to its meshes.

    For every knee present in ``01_models``, recover the transform between the
    ``00_inputs`` segmentation mesh and the model directory's aligned copy by
    least squares, and require the residual to be at numerical precision.
    """
    import pyvista as pv

    config = _config()
    knees = _vendored_model_knees(config)
    if not knees:
        pytest.skip("no stage-01 output in the data root yet")

    failures = []
    for knee in knees:
        resolved = config.for_knee(knee)
        subject_dir = resolved.path("inputs", "subject_meshes")
        aligned_dir = resolved.path("run", "output_dir") / "geometries_nsm_similarity"
        for bone in ("femur", "tibia", "patella"):
            original = subject_dir / f"{knee.subject}_{knee.side}_{bone}.vtk"
            aligned = aligned_dir / f"{knee.subject}_{knee.side}_{bone}_aligned.vtk"
            if not aligned.is_file():
                failures.append(f"{knee.name} {bone}: aligned mesh missing")
                continue
            source = np.asarray(pv.read(original).points, dtype=float)
            target = np.asarray(pv.read(aligned).points, dtype=float)
            if source.shape != target.shape:
                failures.append(
                    f"{knee.name} {bone}: {source.shape} vs {target.shape} points"
                )
                continue
            homogeneous = np.hstack([source, np.ones((len(source), 1))])
            transform, *_ = np.linalg.lstsq(homogeneous, target, rcond=None)
            residual = np.abs(homogeneous @ transform - target).max()
            if residual > ALIGNMENT_RESIDUAL_MM:
                failures.append(f"{knee.name} {bone}: residual {residual:.3e} mm")
    assert not failures, "\n".join(failures)
