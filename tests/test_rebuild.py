"""A rebuild builds from the saved fit, into its own tree, and nowhere else.

``reconstruct_knee.rebuild`` re-runs model building on a knee's saved
shape-model fit. Its value is that it changes nothing but model building, and
its danger is where it writes: model building saves files under the same names
it reads, and a new data root reaches the baseline's fits through a symlink. These
tests pin both:

* the output must resolve inside the data root and must not overlap the fit --
  including an output that would follow the symlink into the baseline tree;
* the model builder receives exactly the saved surfaces, latents, alignment and
  labelled bones, and the fit's own records are linked (not copied) into the
  rebuilt tree;
* an existing rebuild is accepted only when its run record matches.

CPU only; the shape models and the model builder are replaced by stubs. Needs
the data root, for one knee's saved fit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from comak_knee_oa.config import Knee, load_config  # noqa: E402

RECONSTRUCTION = REPO_ROOT / "config" / "reconstruction.json"
KNEE = Knee.parse("9003316_RIGHT")


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    """A data root declaring run ``test``, whose inputs and published fits are
    symlinks into the real one (which it lists read-only)."""
    from comak_knee_oa import paths

    real = paths.data_root()
    published = json.loads(RECONSTRUCTION.read_text())["inputs"]["saved_fits"].split("/{knee}")[0]
    if not (real / published / KNEE.directory).is_dir():
        pytest.skip(f"no saved fit for {KNEE.name} under {real / published}")
    root = tmp_path / "root"
    (root / "01_models").mkdir(parents=True)
    (root / "00_inputs").symlink_to(real / "00_inputs")
    (root / published).symlink_to(real / published)
    (root / "_run.json").write_text(json.dumps({"run_name": "test", "read_only_roots": [str(real)]}))
    monkeypatch.setenv("COMAK_DATA_ROOT", str(root))
    return root


def _config(tmp_path, output_dir):
    data = json.loads(RECONSTRUCTION.read_text())
    data["run"]["output_dir"] = output_dir
    path = tmp_path / "rebuild.json"
    path.write_text(json.dumps(data))
    return load_config(path)


def test_shipped_config_rebuilds_into_the_run_tree(scratch_root, tmp_path):
    """The shipped config builds into 01_models/<run>/, beside the fit, never over it."""
    from reconstruct_knee import built_model_path, rebuild

    config = load_config(RECONSTRUCTION)
    assert config.require("run", "output_dir") != config.require("inputs", "saved_fits")
    assert rebuild(config, KNEE, check_only=True) is True
    knee = config.for_knee(KNEE)
    assert knee.path("run", "output_dir") == scratch_root / "01_models" / "test" / KNEE.directory
    assert built_model_path(knee).stem.endswith(f"{KNEE.directory}_test")
    # A root that names no run cannot say where the run's models go: no rebuild.
    (scratch_root / "_run.json").unlink()
    assert rebuild(config, KNEE, check_only=True) is False


@pytest.mark.parametrize("output_dir", [
    "01_models/OARSI_May.22.2026_Full_run/{knee}",           # the fit itself
    "01_models/OARSI_May.22.2026_Full_run/{knee}/rebuilt",   # through the symlink
])
def test_output_that_reaches_the_fit_is_refused(scratch_root, tmp_path, output_dir):
    from reconstruct_knee import rebuild

    assert rebuild(_config(tmp_path, output_dir), KNEE, check_only=True) is False
    with pytest.raises(FileNotFoundError):
        rebuild(_config(tmp_path, output_dir), KNEE)


def test_rebuild_hands_the_builder_the_saved_fit(scratch_root, tmp_path, monkeypatch):
    import nsosim.model_building
    import nsosim.utils

    import reconstruct_knee as rk

    config = _config(tmp_path, "01_models/rebuilt/{knee}")
    assert rk.rebuild(config, KNEE, check_only=True) is True

    built = []
    monkeypatch.setattr(rk, "_require_gpu", lambda: None)
    monkeypatch.setattr(rk, "require_center_mode_support", lambda mode: None)
    monkeypatch.setattr(nsosim.utils, "load_model", lambda cfg, path: ("model", path))

    def fake_build(**kw):
        built.append(kw)
        model = Path(kw["save_dir"]) / kw["model_name"] / f"{kw['model_name']}.osim"
        model.parent.mkdir(parents=True)
        model.write_text("<OpenSimDocument/>")
        return str(model)

    monkeypatch.setattr(nsosim.model_building, "build_joint_model", fake_build)

    assert rk.rebuild(config, KNEE) is True
    (kw,) = built
    knee_config = config.for_knee(KNEE)
    fit = knee_config.path("inputs", "saved_fits") / "geometries_nsm_similarity"
    out = knee_config.path("run", "output_dir")

    # The saved surfaces, as saved.
    import pyvista as pv

    for bone, surfaces in rk.SAVED_SURFACES.items():
        assert set(kw["bone_meshes"][bone]) == set(surfaces)
        for role, name in surfaces.items():
            saved = pv.read(fit / name)
            np.testing.assert_array_equal(kw["bone_meshes"][bone][role].point_coords, saved.points)

    # The saved latents and alignment, the cohort's labelled bones.
    for bone in rk.SHAPE_MODELS:
        subject = kw["dict_bones"][bone]["subject"]
        alignment = json.loads((fit / bone / f"{bone}_alignment.json").read_text())
        np.testing.assert_array_equal(
            np.squeeze(subject["recon_latent"]), np.squeeze(np.load(fit / bone / f"{bone}_latent.npy"))
        )
        np.testing.assert_array_equal(subject["recon_dict"]["icp_transform"],
                                      np.array(alignment["linear_transform"]))
        assert subject["recon_dict"]["scale"] == alignment["scale"]
        assert (kw["dict_bones"][bone]["wrap"]["path_labeled_bone"]
                == str(rk.labeled_bone_path(knee_config, bone)))
    assert kw["config"]["meniscus_center_mode"] == rk.meniscus_center_mode(knee_config)
    assert Path(kw["save_dir"]).resolve().is_relative_to(out.resolve())

    # The fit's records are links into the fit, not copies.
    for name in rk.fit_record_files(KNEE):
        link = out / "geometries_nsm_similarity" / name
        assert link.is_symlink(), name
        assert link.resolve() == (fit / name).resolve()

    record = json.loads((out / "run_record.json").read_text())
    assert record["stage"] == "rebuild"
    assert record["rebuilt_from"] == knee_config.require("inputs", "saved_fits")
    assert set(record["saved_fit_sha256"]) == set(rk.saved_fit_files(KNEE))
    # The machine is recorded, and never part of what must match.
    assert record["environment"]["run_name"] == "test"
    assert record["environment"]["host"] and record["environment"]["cpu_model"]
    assert "environment" not in rk._rebuild_record(knee_config, KNEE, "midline", {})

    # A second run accepts the matching build without building again ...
    assert rk.rebuild(config, KNEE) is True
    assert len(built) == 1
    # ... and refuses one built under other options.
    record["build_options"]["meniscus_center_mode"] = "something else"
    (out / "run_record.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="does not match"):
        rk.rebuild(config, KNEE)
