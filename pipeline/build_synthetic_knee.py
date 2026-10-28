"""Figure 3: build one synthetic knee's OpenSim model from its shape-axis descriptor.

A synthetic knee has no MRI. Its surfaces come from decoding the three latents
and transforms that ``derive_shape_axis.py`` wrote, instead of from fitting the
shape models to segmentation meshes. From there it is built as a cohort knee
is: the same base model, ligament attachments, contact surfaces, fat
pad, wrap surfaces and ligament stiffnesses, through the one model-building
call in ``reconstruct_knee.py``. The build config is the cohort's own
``config/reconstruction.json``, and a synthetic run may change only the three
keys in ``BUILD_OVERRIDES`` -- where the model goes, its version tag, and the
meniscus trim centre. Anything else raises, so a synthetic knee cannot be built
under different assumptions from the cohort it is compared with.

Output: the knee's model under the synthetic tree
(``02_simulations/synthetic_shape_axis/<label>``), with a run record. Decoding runs on
a GPU (a few minutes per knee). Labels are named ``fig5_sigma_<k>``.

    python pipeline/run_synthetic.py build --config config/synthetic_shape_axis.json \\
        --label fig5_sigma_p0
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa.config import Config, ConfigError, Label, load_config  # noqa: E402
from derive_shape_axis import INPUT_FILES  # noqa: E402
from reconstruct_knee import (  # noqa: E402
    LABELED_BONES,
    SHAPE_MODELS,
    _require_gpu,
    build_input_problems,
    build_knee_model,
    build_options,
    built_model_path,
    femur_reference_center,
    labeled_bone_path,
    library_paths,
    meniscus_center_mode,
    require_center_mode_support,
)

__all__ = ["build_config", "build", "library_paths", "synthetic_tree", "BUILD_OVERRIDES"]

# The only reconstruction settings a synthetic build may set. Everything else is
# the cohort's.
BUILD_OVERRIDES = {
    ("run", "output_dir"),
    ("reconstruction", "model_version"),
    ("reconstruction", "meniscus_center_mode"),
}


def synthetic_tree(synth: Config) -> str:
    """The synthetic tree's root, as a data-root-relative path."""
    return f"{synth.require('run', 'root')}/synthetic_shape_axis"


def build_config(synth: Config) -> Config:
    """The cohort reconstruction config, with only ``BUILD_OVERRIDES`` changed."""
    section = synth.section("build")
    source = Path(synth.origin).parent / section.get(
        "reconstruction_config", "reconstruction.json"
    )
    cohort = load_config(source)
    data = copy.deepcopy(cohort.data)

    overrides = {("run", "output_dir"): f"{synthetic_tree(synth)}/{{knee}}"}
    for key in ("model_version", "meniscus_center_mode"):
        if key in section:
            overrides[("reconstruction", key)] = section[key]
    allowed = {"reconstruction_config", "model_version", "meniscus_center_mode"}
    unknown = sorted(k for k in section if k not in allowed and not k.startswith("_"))
    if unknown:
        raise ConfigError(
            f"build section may set only model_version and meniscus_center_mode; "
            f"got {unknown}. Model-building options come from {source.name}, so the "
            "synthetic knees stay comparable with the cohort."
        )
    for (sec, key), value in overrides.items():
        if (sec, key) not in BUILD_OVERRIDES:  # defensive: the table is the contract
            raise ConfigError(f"{sec}.{key} may not be overridden for a synthetic build")
        data.setdefault(sec, {})[key] = value
    return Config(data, origin=source)


def build(synth: Config, label: str, check_only: bool = False) -> bool:
    """Decode and build one synthetic knee. Returns True when its model exists."""
    config = build_config(synth).for_knee(Label(label))
    output_dir = config.path("run", "output_dir")
    inputs_dir = output_dir / "input_descriptors"
    center_mode = meniscus_center_mode(config)

    problems = build_input_problems(config)
    for filename in INPUT_FILES:
        if not (inputs_dir / filename).is_file():
            problems.append(f"descriptor input not found: {inputs_dir / filename} "
                            "(run `run_synthetic.py axis` first)")
    model_file = built_model_path(config)
    if problems:
        message = "\n  ".join(problems)
        if check_only:
            print(f"MISSING INPUTS for {label}:\n  {message}")
            return False
        raise FileNotFoundError(f"inputs missing for {label}:\n  {message}")
    if check_only:
        print(f"OK {label}: all build inputs resolve")
        print(f"   meniscus trim centre: {center_mode}")
        print(f"   base model {config.path('inputs', 'base_model')}")
        print(f"   model      {model_file}")
        return True

    if model_file.exists():
        _check_existing_build(output_dir, config, center_mode, inputs_dir)
        print(f"already built: {model_file}")
        return True
    # The model builder reuses whatever it finds in a non-empty model folder, so
    # a half-finished build must be removed by hand rather than overwritten.
    if model_file.parent.is_dir() and any(model_file.parent.iterdir()):
        raise RuntimeError(
            f"{model_file.parent} holds a partial build; remove it and rerun"
        )

    _require_gpu()
    require_center_mode_support(center_mode)
    from nsosim.decode import decode_joint_from_descriptors
    from nsosim.model_building import build_dict_bones_for_interpolation
    from nsosim.transforms import recover_bone_transform
    from nsosim.utils import load_model

    section = config.section("reconstruction")
    seed = section.get("seed", 0)
    (output_dir / "geometries_nsm_similarity").mkdir(parents=True, exist_ok=True)
    _write_run_record(output_dir, config, synth, label, center_mode, inputs_dir)

    started = time.time()
    models, model_configs = {}, {}
    for bone, name in SHAPE_MODELS.items():
        model_dir = config.path("inputs", "shape_models") / name
        model_configs[bone] = json.loads((model_dir / "model_params_config.json").read_text())
        models[bone] = load_model(model_configs[bone], str(model_dir / "model" / "2000.pth"))

    arrays = {name: np.load(inputs_dir / name) for name in INPUT_FILES}
    latents = {bone: np.squeeze(arrays[f"{bone}_latent.npy"]) for bone in SHAPE_MODELS}
    t_fem = arrays["T_fem.npy"]
    t_rel = {"tibia": arrays["T_rel_tib.npy"], "patella": arrays["T_rel_pat.npy"]}

    print(f"decoding {label} (seed {seed})")
    osim_meshes = decode_joint_from_descriptors(
        femur_latent=latents["femur"],
        tibia_latent=latents["tibia"],
        patella_latent=latents["patella"],
        T_fem=t_fem,
        T_rel_tib=t_rel["tibia"],
        T_rel_pat=t_rel["patella"],
        models=models,
        model_configs=model_configs,
        fem_ref_center=femur_reference_center(config),
        # Same resampling as a cohort knee: bones only, cartilage and menisci as decoded.
        clusters={
            "femur": {"bone": section.get("bone_clusters_femur_tibia", 20_000)},
            "tibia": {"bone": section.get("bone_clusters_femur_tibia", 20_000)},
            "patella": {"bone": section.get("bone_clusters_patella", 10_000)},
        },
        seed=seed,
    )
    print(f"  decoded in {time.time() - started:.0f} s")

    bones = build_dict_bones_for_interpolation(
        models=models,
        latents=latents,
        transforms={
            "femur": t_fem,
            "tibia": recover_bone_transform(t_rel["tibia"], t_fem),
            "patella": recover_bone_transform(t_rel["patella"], t_fem),
        },
        labeled_bone_dir=str(config.path("inputs", "labeled_bones")),
    )
    # The helper names each labelled bone by convention; use the same files as a
    # cohort knee (``tibia_labeled_v2.vtk`` carries the plateau-centre labels).
    for bone in LABELED_BONES:
        bones[bone]["wrap"]["path_labeled_bone"] = str(labeled_bone_path(config, bone))

    print("building the OpenSim model")
    built = build_knee_model(config, bones=bones, osim_meshes=osim_meshes,
                             center_mode=center_mode)
    print(f"built {built} in {time.time() - started:.0f} s total")
    if not model_file.exists():
        raise RuntimeError(f"model building reported {built} but no model is at {model_file}")
    return True


def _descriptor_hashes(inputs_dir: Path) -> dict[str, str]:
    return {name: hashlib.sha256((inputs_dir / name).read_bytes()).hexdigest()
            for name in INPUT_FILES}


def _check_existing_build(output_dir: Path, config: Config, center_mode: str,
                          inputs_dir: Path) -> None:
    """Refuse an existing model built from other descriptors or other options.

    The model path encodes the label and version only, so without this a knee
    rebuilt under a changed trim centre -- or after the axis was re-derived --
    would keep its old model.
    """
    record_file = output_dir / "run_record.json"
    if not record_file.is_file():
        raise RuntimeError(f"{output_dir} has a model but no run_record.json; "
                           "cannot tell what it was built from")
    record = json.loads(record_file.read_text())
    expected = json.loads(json.dumps(build_options(config, center_mode), default=str))
    problems = []
    if record.get("build_options") != expected:
        keys = sorted(k for k in expected.keys() | record.get("build_options", {}).keys()
                      if expected.get(k) != record.get("build_options", {}).get(k))
        problems.append(f"build options differ: {keys}")
    if record.get("descriptor_sha256") != _descriptor_hashes(inputs_dir):
        problems.append("input descriptors differ from those the model was built from")
    if problems:
        raise RuntimeError(
            f"existing model in {output_dir} does not match this build: "
            + "; ".join(problems)
            + ". Use a new build.model_version or run.root."
        )


def _write_run_record(output_dir: Path, config: Config, synth: Config, label: str,
                      center_mode: str, inputs_dir: Path) -> None:
    """Record what produced this build, next to it: config, resolved options, code, machine."""
    from comak_knee_oa.manifest import _code_versions, environment_record

    record = {
        "label": label,
        "synthetic_config_file": str(synth.origin) if synth.origin else None,
        "config_file": str(config.origin) if config.origin else None,
        "config": config.data,
        "build_options": build_options(config, center_mode),
        "descriptor_sha256": _descriptor_hashes(inputs_dir),
        "project_coronary": False,
        "node": socket.gethostname(),
        "code": _code_versions(),
        "libraries": library_paths(),
        "environment": environment_record(),
    }
    (output_dir / "run_record.json").write_text(json.dumps(record, indent=2, default=str) + "\n")

