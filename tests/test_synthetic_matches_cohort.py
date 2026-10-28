"""Synthetic knees are built and simulated under the cohort's assumptions.

Figure 3 compares synthetic knees with each other along a shape axis, and the
paper reads the result against the cohort. That only holds if a synthetic knee
is built and walked like a cohort knee. A builder with its own copy of the
model-building call can drift from the cohort in many ways (base model and
contact search, wrap-surface anchoring, coronary ligaments, fat-pad search,
meniscal inertia after scaling). These tests pin the invariant:

* the allowed differences are only the three location/version/trim-centre
  keys for a build and the three path keys for a simulation -- pinned here as
  literals, so widening a module's table fails this file;
* the synthetic build config is ``config/reconstruction.json`` with only those
  keys changed, and every synthetic simulation config is the donor's
  measured-gait config with only its path keys changed;
* ``build_synthetic_knee.build`` itself -- not just the shared helper -- hands
  the model builder the cohort's arguments, the configured trim centre, the
  cohort's labelled bones and resampling, and the knee's own descriptors.

CPU only; the decoder and model builder are replaced by recorders. Every test
runs on a scratch data root that declares a run (the configs name their outputs
by it) and borrows ``00_inputs`` from the configured root, read-only.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from comak_knee_oa.config import ConfigError, Knee, Label, load_config  # noqa: E402

SYNTH = REPO_ROOT / "config" / "synthetic_shape_axis.json"
RECONSTRUCTION = REPO_ROOT / "config" / "reconstruction.json"
LABEL = "fig5_sigma_p0"


@pytest.fixture(autouse=True)
def declared_root(tmp_path, monkeypatch):
    """A scratch root that declares run ``test``; the configured root is read-only."""
    from comak_knee_oa import paths

    real = paths.data_root()
    root = tmp_path / "root"
    (root / "02_simulations").mkdir(parents=True)
    (root / "00_inputs").symlink_to(real / "00_inputs")
    (root / "_run.json").write_text(json.dumps({"run_name": "test", "read_only_roots": [str(real)]}))
    monkeypatch.setenv("COMAK_DATA_ROOT", str(root))
    return root


def _donor_params():
    """One parameter per synthetic donor; a donor whose config has not landed is xfail."""
    for gait in json.loads(SYNTH.read_text())["simulate"]["gaits"]:
        missing = not (REPO_ROOT / "config" / "measured_gait" / f"{gait}.json").is_file()
        yield pytest.param(gait, id=gait, marks=pytest.mark.xfail(
            missing, strict=True, raises=FileNotFoundError,
            reason=f"no config/measured_gait/{gait}.json yet"))


def _flatten(data, prefix=()):
    for key, value in data.items():
        if isinstance(value, dict):
            yield from _flatten(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def _differences(a: dict, b: dict) -> set[tuple]:
    fa, fb = dict(_flatten(a)), dict(_flatten(b))
    return {key for key in fa.keys() | fb.keys() if fa.get(key) != fb.get(key)}


def test_label_fills_only_the_knee_placeholder():
    assert Label("fig5_sigma_m2").substitute("x/{knee}/y") == "x/fig5_sigma_m2/y"
    config = load_config(RECONSTRUCTION).for_knee(Label(LABEL))
    # A label has no subject, so a subject path keeps its placeholder and is refused.
    with pytest.raises(ConfigError):
        config.path("inputs", "subject_meshes")


def test_override_tables_are_the_contract():
    from build_synthetic_knee import BUILD_OVERRIDES
    from run_synthetic import SIMULATE_OVERRIDES

    assert BUILD_OVERRIDES == {
        ("run", "output_dir"),
        ("reconstruction", "model_version"),
        ("reconstruction", "meniscus_center_mode"),
    }
    assert SIMULATE_OVERRIDES == {
        ("run", "results_dir"),
        ("model", "built_model"),
        ("model", "scaled_model"),
    }


def test_build_config_is_the_cohort_config():
    from build_synthetic_knee import BUILD_OVERRIDES, build_config

    synth = load_config(SYNTH)
    cohort = load_config(RECONSTRUCTION)
    changed = {key[:2] for key in _differences(cohort.data, build_config(synth).data)}
    assert changed <= BUILD_OVERRIDES, f"unexpected build differences: {changed - BUILD_OVERRIDES}"
    # The synthetic knees are trimmed about the same centre as the cohort's.
    built = build_config(synth)
    assert (built.get("reconstruction", "meniscus_center_mode")
            == cohort.get("reconstruction", "meniscus_center_mode"))


def test_build_section_rejects_other_keys(tmp_path):
    from build_synthetic_knee import build_config

    data = json.loads(SYNTH.read_text())
    data["build"]["fatpad_max_proximity_m"] = 0.015
    data["build"]["reconstruction_config"] = str(RECONSTRUCTION)
    bad = tmp_path / "synthetic_bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ConfigError):
        build_config(load_config(bad))


@pytest.mark.parametrize("gait", list(_donor_params()))
def test_simulate_config_is_the_donor_config(gait):
    from run_synthetic import SIMULATE_OVERRIDES, simulate_config

    synth = load_config(SYNTH)
    donor = load_config(REPO_ROOT / "config" / "measured_gait" / f"{gait}.json")
    changed = {key[:2] for key in _differences(donor.data, simulate_config(synth, LABEL, gait).data)}
    assert changed <= SIMULATE_OVERRIDES, f"unexpected solver differences: {changed - SIMULATE_OVERRIDES}"


def test_model_builder_gets_the_cohort_arguments(monkeypatch):
    import nsosim.model_building as model_building

    from build_synthetic_knee import build_config
    from reconstruct_knee import build_knee_model

    calls = []
    monkeypatch.setattr(model_building, "build_joint_model", lambda **kw: calls.append(kw) or "x")

    cohort = load_config(RECONSTRUCTION).for_knee(Knee.parse("9003316_RIGHT"))
    synthetic = build_config(load_config(SYNTH)).for_knee(Label(LABEL))
    build_knee_model(cohort, bones={}, osim_meshes={}, center_mode="as_run")
    build_knee_model(synthetic, bones={}, osim_meshes={}, center_mode="as_run")
    cohort_kw, synthetic_kw = calls

    import numpy as np

    location = {"save_dir", "folder_save_bones", "model_name"}
    for key in cohort_kw.keys() | synthetic_kw.keys():
        if key in location:
            continue
        a, b = cohort_kw[key], synthetic_kw[key]
        same = np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
        assert same, f"build_joint_model({key}=...) differs"

    assert synthetic_kw["project_coronary"] is False
    assert Path(synthetic_kw["path_base_osim_model"]).name == "May.22.2026_Full_run"
    options = synthetic_kw["config"]
    assert options["smith2019_osim_path"].endswith("smith2019.osim")
    assert options["wrap_fit_mode"] == "lbfgs_smith2019_anchor"
    assert options["fatpad_max_proximity"] == 0.008


def test_build_entry_point_passes_cohort_arguments(monkeypatch, tmp_path, declared_root):
    """Drive build_synthetic_knee.build() end to end with the GPU parts stubbed."""
    import numpy as np
    import nsosim.decode
    import nsosim.model_building
    import nsosim.transforms
    import nsosim.utils

    import build_synthetic_knee as bsk
    import reconstruct_knee
    from derive_shape_axis import INPUT_FILES

    # The scratch data root: 00_inputs from the real one, an empty synthetic tree.
    root = declared_root

    synth_data = json.loads(SYNTH.read_text())
    synth_data["build"]["reconstruction_config"] = str(RECONSTRUCTION)
    synth_file = tmp_path / "synthetic.json"
    synth_file.write_text(json.dumps(synth_data))
    synth = load_config(synth_file)

    inputs = root / synth_data["run"]["root"] / "synthetic_shape_axis" / LABEL / "input_descriptors"
    inputs.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in INPUT_FILES:
        shape = (4, 4) if name.startswith("T_") else (
            {"femur": 1024, "tibia": 512, "patella": 512}[name.split("_")[0]],)
        np.save(inputs / name, rng.standard_normal(shape))

    decoded, built = [], []
    monkeypatch.setattr(bsk, "_require_gpu", lambda: None)
    monkeypatch.setattr(bsk, "require_center_mode_support", lambda mode: None)
    monkeypatch.setattr(nsosim.utils, "load_model", lambda cfg, path: ("model", path))
    monkeypatch.setattr(nsosim.decode, "decode_joint_from_descriptors",
                        lambda **kw: decoded.append(kw) or {"decoded": True})
    monkeypatch.setattr(nsosim.transforms, "recover_bone_transform", lambda rel, fem: rel @ fem)
    monkeypatch.setattr(
        nsosim.model_building, "build_dict_bones_for_interpolation",
        lambda **kw: {b: {"wrap": {"path_labeled_bone": "placeholder"}} for b in kw["latents"]},
    )

    def fake_build(**kw):
        built.append(kw)
        model = Path(kw["save_dir"]) / kw["model_name"] / f"{kw['model_name']}.osim"
        model.parent.mkdir(parents=True)
        model.write_text("<OpenSimDocument/>")
        return str(model)

    monkeypatch.setattr(nsosim.model_building, "build_joint_model", fake_build)

    assert bsk.build(synth, LABEL) is True
    (kw,) = built
    (dec,) = decoded
    config = bsk.build_config(synth).for_knee(Label(LABEL))

    # The run record names the machine and the run.
    record = json.loads((config.path("run", "output_dir") / "run_record.json").read_text())
    assert record["environment"]["run_name"] == "test"
    assert record["environment"]["host"] and record["environment"]["cpu_model"]

    # The configured trim centre reaches the builder, with the cohort's options.
    assert kw["config"] == reconstruct_knee.build_options(
        config, reconstruct_knee.meniscus_center_mode(config))
    assert kw["project_coronary"] is False
    assert kw["seed"] == config.get("reconstruction", "seed", 0)
    # The cohort's labelled bones, not the helper's naming convention.
    for bone in reconstruct_knee.LABELED_BONES:
        expected = reconstruct_knee.labeled_bone_path(config, bone)
        assert kw["dict_bones"][bone]["wrap"]["path_labeled_bone"] == str(expected)
    # The knee's own descriptors and the cohort's resampling.
    np.testing.assert_array_equal(dec["femur_latent"], np.load(inputs / "femur_latent.npy"))
    np.testing.assert_array_equal(dec["T_rel_tib"], np.load(inputs / "T_rel_tib.npy"))
    assert dec["clusters"] == {"femur": {"bone": 20000}, "tibia": {"bone": 20000},
                               "patella": {"bone": 10000}}
    assert dec["seed"] == 0
    assert kw["bone_meshes"] == {"decoded": True}

    # A second call accepts the model only because the record matches ...
    assert bsk.build(synth, LABEL) is True
    # ... and refuses it once the descriptors change underneath it.
    np.save(inputs / "tibia_latent.npy", rng.standard_normal(512))
    with pytest.raises(RuntimeError, match="descriptors differ"):
        bsk.build(synth, LABEL)
