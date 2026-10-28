"""Stage 01: reconstruct one knee's anatomy and build its OpenSim model.

From a knee's segmentation meshes (bones, cartilage, menisci) this stage:

1. rigidly aligns the knee to the reference anatomy and fits a neural shape
   model per bone, producing a watertight subject-specific reconstruction;
2. decodes each reconstruction into OpenSim-frame surface meshes;
3. builds the subject's simulation model into the base COMAK model --
   articular contact surfaces, menisci, ligaments and their attachments, wrap
   surfaces, and the prefemoral fat pad.

Inputs: the knee's eight segmentation meshes (``inputs.subject_meshes``) and the
reference anatomy, shape models and base model under ``00_inputs/reference``.
Output: ``run.output_dir`` (``01_models/<run>/<knee>``), holding the shape-model
fit (``geometries_nsm_similarity/``), the OpenSim model and a run record --
everything stage 02 needs to simulate this knee walking. Fitting runs on a GPU.

A knee that has already been fitted can be rebuilt without fitting again
(``rebuild``): step 3 runs on the surfaces step 2 saved, so a change to model
building reaches every knee without re-running the shape models. Run names:
``v3`` is the run this paper reports; ``v2`` is an earlier release of the
dataset that the code can still read. The v3 models are rebuilt from the v2
shape-model fits (``inputs.saved_fits``).

Reproducibility
---------------
Each run seeds every random generator the fit uses (``reconstruction.seed``:
Python, NumPy, PyTorch and CUDA, with deterministic cuDNN kernels), so latent
initialisation and point sampling repeat. A refit is still not bit-identical:
the shape model's GPU backward pass has no deterministic kernel, and mesh
extraction (marching cubes, surface resampling) yields vertices that differ in
count and order while describing the same surface. Compare reconstructions by
surface distance, not vertex by vertex.

Running it
----------
    python pipeline/run_knee.py reconstruct --config config/reconstruction.json \\
        --knee 9003316_RIGHT

    python pipeline/run_knee.py reconstruct --config ... --knee ... --check

    python pipeline/run_knee.py rebuild --config config/reconstruction.json \\
        --knee 9003316_RIGHT
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comak_knee_oa.config import Config, Knee  # noqa: E402
from comak_knee_oa.paths import data_root  # noqa: E402

__all__ = [
    "reconstruct",
    "rebuild",
    "SAVED_SURFACES",
    "saved_fit_files",
    "fit_record_files",
    "library_paths",
    "built_model_path",
    "subject_mesh_files",
    "build_input_problems",
    "build_options",
    "build_knee_model",
    "meniscus_center_mode",
    "require_center_mode_support",
    "femur_reference_center",
    "labeled_bone_path",
    "SHAPE_MODELS",
]

# Neural shape model per bone, named by its directory in the released NSM
# weights (00_inputs/reference/nsm_models); the leading number (568, 650, 648)
# identifies the trained model. The thickness stage (derive_thickness.py) uses
# femur model 647 and the same tibia model 650. The femur model 568 is a joint
# model: it fits the femur bone together with its cartilage and both menisci,
# which is why the meniscus meshes are required inputs even though menisci get
# no model of their own here.
SHAPE_MODELS = {
    "femur": "568_nsm_femur_bone_cart_men_v0.0.1",
    "tibia": "650_nsm_tibia_v0.0.1",
    "patella": "648_nsm_patella_v0.0.1",
}

# Labelled reference bones (wrap-surface and landmark labels), one per bone. The
# tibia file also carries the tibial plateau-centre labels that the meniscus trim
# origin uses (see MENISCUS_CENTER_MODES).
LABELED_BONES = {
    "femur": "femur_labeled.vtk",
    "tibia": "tibia_labeled_v2.vtk",
    "patella": "patella_labeled.vtk",
}

# Origin of the radial trim that cuts each meniscus contact surface back to its
# articular face (``reconstruction.meniscus_center_mode``):
#   midline -- one point midway between the tibial plateau centres, used for
#              both menisci (the default, and the shipped config's setting);
#   as_run  -- one point per meniscus, the centroid of that meniscus's centre
#              label on the labelled tibia (the trim centres of the v2 models).
# Contact surfaces from the two agree to within a few percent of their area.
MENISCUS_CENTER_MODES = ("midline", "as_run")

# Reference bone meshes the subject is aligned to, one per bone.
REFERENCE_BONES = {
    "femur": "smith2019-R-femur-bone_processed.vtk",
    "tibia": "smith2019-R-tibia-bone_processed.vtk",
    "patella": "smith2019-R-patella-bone_processed.vtk",
}


def subject_mesh_files(knee: Knee) -> dict[str, str]:
    """The segmentation meshes one knee needs, keyed by role.

    All eight must exist: the femur shape model consumes the meniscus meshes,
    so a knee without them cannot be reconstructed at all.
    """
    prefix = f"{knee.subject}_{knee.side}"
    return {
        "femur": f"{prefix}_femur.vtk",
        "femur_cart": f"{prefix}_femur_cart.vtk",
        "tibia": f"{prefix}_tibia.vtk",
        "tibia_cart": f"{prefix}_tibia_cart.vtk",
        "patella": f"{prefix}_patella.vtk",
        "patella_cart": f"{prefix}_patella_cart.vtk",
        "med_men": f"{prefix}_med_men.vtk",
        "lat_men": f"{prefix}_lat_men.vtk",
    }


def built_model_path(config: Config) -> Path:
    """Where the built model lands, following the layout stage 02 reads.

    ``<output_dir>/custom_nsm_<name>_model/<name>_nsm_<knee>_<version>/...``
    """
    output_dir = config.path("run", "output_dir")
    name = config.get("reconstruction", "model_name", "full_body_healthy_knee")
    version = config.require("reconstruction", "model_version")
    knee = config.knee.directory
    specific = f"{name}_nsm_{knee}_{version}"
    return output_dir / f"custom_nsm_{name}_model" / specific / f"{specific}.osim"


def _bone_definitions(config: Config, knee: Knee, subject_dir: Path | None = None) -> dict:
    """The per-bone input table the shape-model fitter consumes.

    ``subject_dir`` overrides where the segmentation meshes are read from (the
    staged links ``reconstruct`` fits from); by default, ``inputs.subject_meshes``.
    """
    subject_dir = subject_dir or config.path("inputs", "subject_meshes")
    reference_dir = config.path("inputs", "reference_meshes")
    models_dir = config.path("inputs", "shape_models")
    labeled_dir = config.path("inputs", "labeled_bones")
    meshes = subject_mesh_files(knee)

    def model_paths(bone):
        model_dir = models_dir / SHAPE_MODELS[bone]
        return {
            "path_model_state": str(model_dir / "model" / "2000.pth"),
            "path_model_config": str(model_dir / "model_params_config.json"),
        }

    definitions = {}
    for bone in ("femur", "tibia", "patella"):
        subject = {
            "folder": str(subject_dir),
            "bone_filename": meshes[bone],
            "cart_filename": meshes[f"{bone}_cart"],
        }
        if bone == "femur":
            subject["med_men_filename"] = meshes["med_men"]
            subject["lat_men_filename"] = meshes["lat_men"]
        definitions[bone] = {
            "ref": {
                "folder": str(reference_dir / bone),
                "bone_filename": REFERENCE_BONES[bone],
            },
            "subject": subject,
            "model": model_paths(bone),
            "wrap": {"path_labeled_bone": str(labeled_dir / LABELED_BONES[bone])},
        }
    definitions["meniscus"] = {
        "ref": {
            "folder": str(labeled_dir / "menisci"),
            "med_men_filename": "med_meniscus_processed.vtk",
            "lat_men_filename": "lat_meniscus_processed.vtk",
        }
    }
    return definitions


def _ligament_stiffness(config: Config) -> dict:
    """Patellar-tendon stiffness table for the model builder.

    The tendon is stiffened relative to its literature default to keep the
    patellofemoral joint stable through the whole gait cycle; the builder
    wants the default and the multiplier separately, per bundle.
    """
    stiffness = config.get("material_properties", "patellar_tendon_stiffness_N", 12_000)
    default = config.get("material_properties", "patellar_tendon_default_stiffness_N", 3_000)
    return {
        f"PT{i}": {"default_stiffness": default, "update_factor": stiffness / default}
        for i in range(1, 7)
    }


def _require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "reconstruction requires a CUDA GPU and none is available on this "
            "machine. Submit to a GPU node, or use --check to validate inputs "
            "without fitting."
        )


# ------------------------------------------------------------ model building ----
#
# Everything below is shared by every knee this project builds: the cohort knees
# reconstructed from MRI (``reconstruct``) and the synthetic knees decoded from
# shape-axis descriptors (``build_synthetic_knee.py``). They differ in how their
# surfaces are obtained and in nothing after that. Keeping one call site for the
# model builder guarantees it: the base model, contact search distances, wrap
# surface anchoring, coronary ligament attachments, fat pad and meniscal inertia
# are the same for both, because there is only one place they are set.


def meniscus_center_mode(config: Config) -> str:
    """The configured origin of the meniscus radial trim (``MENISCUS_CENTER_MODES``)."""
    return config.get("reconstruction", "meniscus_center_mode", "midline")


def labeled_bone_path(config: Config, bone: str) -> Path:
    """The labelled reference bone one knee's wrap surfaces are fitted from."""
    return config.path("inputs", "labeled_bones") / LABELED_BONES[bone]


def femur_reference_center(config: Config) -> np.ndarray:
    """Centre of the reference femur: the origin every decoded surface is placed about."""
    alignment = json.loads(
        (config.path("inputs", "reference_meshes") / "femur" / "ref_femur_alignment.json")
        .read_text()
    )
    return np.array(alignment["mean_orig"])


def build_input_problems(config: Config) -> list[str]:
    """Missing or invalid inputs to model building, for any knee.

    ``reconstruct`` adds the knee's own segmentation meshes on top of this.
    """
    problems = config.check_inputs(
        ("inputs", "reference_meshes"),
        ("inputs", "shape_models"),
        ("inputs", "labeled_bones"),
        ("inputs", "ligament_attachments"),
        ("inputs", "base_model"),
        ("inputs", "smith2019_model"),
    )
    for bone, model in SHAPE_MODELS.items():
        weights = config.path("inputs", "shape_models") / model / "model" / "2000.pth"
        if not weights.is_file():
            problems.append(f"{bone} shape-model weights not found: {weights}")
    for bone in LABELED_BONES:
        labeled = labeled_bone_path(config, bone)
        if not labeled.is_file():
            problems.append(f"{bone} labelled reference bone not found: {labeled}")
    center_mode = meniscus_center_mode(config)
    if center_mode not in MENISCUS_CENTER_MODES:
        problems.append(
            f"reconstruction.meniscus_center_mode must be one of "
            f"{MENISCUS_CENTER_MODES}, got {center_mode!r}"
        )
    return problems


def require_center_mode_support(center_mode: str) -> None:
    """Refuse an nsosim that lacks the ``meniscus_center_mode`` option.

    An nsosim without the trim-centre option would ignore the key and build
    as_run whatever the config says. The pinned nsosim (docs/pinned_versions.md)
    has it; the commit named in the error is the first that added it.
    """
    from nsosim import model_building

    if center_mode not in getattr(model_building, "MENISCUS_CENTER_MODES", ()):
        raise RuntimeError(
            f"the installed nsosim does not support meniscus_center_mode={center_mode!r}; "
            "it needs nsosim e108ca9 or later (see docs/pinned_versions.md)."
        )


def build_options(config: Config, center_mode: str) -> dict:
    """Every model-building option, from the reconstruction config and nowhere else."""
    section = config.section("reconstruction")
    materials = config.section("material_properties")
    return {
        "triangle_density": section.get("triangle_density", 3_000_000),
        "lig_normal_shift": section.get("ligament_normal_shift_m", 5e-4),
        "dict_lig_stiffness": _ligament_stiffness(config),
        # The knee starts each simulation from the neutral pose; all six knee
        # coordinates are zeroed in the built model.
        "dict_joints_coords_to_update": {
            "knee_r": {i: 0.0 for i in range(6)}
        },
        "fatpad_elastic_modulus": materials.get("fatpad_elastic_modulus_Pa", 4e6),
        "fatpad_poissons_ratio": materials.get("fatpad_poissons_ratio", 0.45),
        "fatpad_thickness": materials.get("fatpad_thickness_m", 0.01),
        "fatpad_min_proximity": materials.get("fatpad_min_proximity_m", 0.0),
        "fatpad_max_proximity": materials.get("fatpad_max_proximity_m", 0.008),
        # Direction of the rays that stop the fat pad short of the patella:
        # "dilation_normals" (along the direction the fat pad grows; the shipped
        # config's setting) or "auto_oriented" (normals recomputed on the open
        # bone patch; nsosim's default, used when the key is absent, and how the
        # v2 models were built).
        "fatpad_clearance_rays": section.get("fatpad_clearance_rays", "auto_oriented"),
        "wrap_n_restarts": section.get("wrap_n_restarts", 1),
        "wrap_jitter_scale": section.get("wrap_jitter_scale", 1e-6),
        # Wrap surfaces are initialised from the corresponding surface of the
        # generic model, transferred to the subject and regularised toward it
        # where the fit is otherwise unconstrained.
        "smith2019_osim_path": str(config.path("inputs", "smith2019_model")),
        "wrap_fit_mode": section.get("wrap_fit_mode", "lbfgs_smith2019_anchor"),
        "label_correspondence_transform_kind": section.get(
            "label_correspondence_transform_kind", "auto"
        ),
        "meniscus_center_mode": center_mode,
    }


def build_knee_model(config: Config, *, bones: dict, osim_meshes: dict, center_mode: str) -> str:
    """Assemble one knee's OpenSim model from its decoded surfaces.

    ``config`` is already substituted for the knee (``Config.for_knee``). The
    base model, the ligament attachments, every option and the output layout
    come from it; only the surfaces and their shape-model records are passed in.
    """
    from nsosim.model_building import build_joint_model

    section = config.section("reconstruction")
    output_dir = config.path("run", "output_dir")
    name = section.get("model_name", "full_body_healthy_knee")
    return build_joint_model(
        bone_meshes=osim_meshes,
        dict_bones=bones,
        ref_data_paths={"folder_ref_recons": str(config.path("inputs", "reference_meshes"))},
        dict_lig_musc_attach_params=json.loads(
            config.path("inputs", "ligament_attachments").read_text()
        ),
        fem_ref_center=femur_reference_center(config),
        save_dir=str(output_dir / f"custom_nsm_{name}_model"),
        folder_save_bones=str(output_dir / "geometries_nsm_similarity"),
        model_name=built_model_path(config).stem,
        path_base_osim_model=str(config.path("inputs", "base_model")),
        config=build_options(config, center_mode),
        project_meniscal_to_tibia=section.get("project_meniscal_to_tibia", False),
        # Coronary ligaments keep their shape-model-interpolated tibial
        # attachment. Projecting them onto the bone shortens them to a few
        # millimetres and pins the meniscus to the tibia.
        project_coronary=False,
        seed=section.get("seed", 0),
        # The template's ~163 generic meshes are linked into the model's Geometry
        # folder instead of copied (25 MB per model); the knee's own meshes are
        # real files. Storage only: the model is identical.
        geometry_staging="link",
    )


def reconstruct(config: Config, knee: Knee, check_only: bool = False, force: bool = False):
    """Reconstruct one knee and build its OpenSim model.

    Returns True on success. Refuses to overwrite an existing built model
    unless ``force`` is set: a cohort's models are the input to every
    simulation stage, and replacing one accidentally invalidates whatever was
    simulated from it.
    """
    config = config.for_knee(knee)

    # Collect every missing input and config problem before any fitting starts.
    problems = config.check_inputs(("inputs", "subject_meshes"))
    problems += build_input_problems(config)
    subject_dir = config.path("inputs", "subject_meshes")
    for filename in subject_mesh_files(knee).values():
        if not (subject_dir / filename).is_file():
            problems.append(f"subject mesh not found: {subject_dir / filename}")
    center_mode = meniscus_center_mode(config)
    if problems:
        message = "\n  ".join(problems)
        if check_only:
            print(f"MISSING INPUTS for {knee.name}:\n  {message}")
            return False
        raise FileNotFoundError(f"inputs missing for {knee.name}:\n  {message}")

    model_file = built_model_path(config)
    if check_only:
        print(f"OK {knee.name}: all reconstruction inputs resolve")
        print(f"   meniscus trim centre: {center_mode}")
        print(f"   meshes {subject_dir}")
        print(f"   model  {model_file}")
        return True

    if model_file.exists() and not force:
        print(f"already built: {model_file}")
        return True

    _require_gpu()
    require_center_mode_support(center_mode)
    from nsosim.nsm_fitting import align_knee_osim_fit_nsm, nsm_recon_to_osim

    section = config.section("reconstruction")
    seed = section.get("seed", 0)
    output_dir = config.path("run", "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = output_dir / "geometries_nsm_similarity"
    geometry_dir.mkdir(exist_ok=True)

    femur_center = femur_reference_center(config)

    _write_run_record(output_dir, config, knee)

    # The fitter saves copies of the meniscus meshes beside the meshes it reads
    # (nsosim save_intermediate_men, on by default). Fitting from links in this
    # knee's own tree keeps those writes out of 00_inputs.
    staged = _stage_subject_meshes(knee, subject_dir, output_dir / "subject_meshes")

    started = time.time()
    print(f"fitting shape models for {knee.name} (seed {seed})")
    bones = align_knee_osim_fit_nsm(
        dict_bones=_bone_definitions(config, knee, subject_dir=staged),
        folder_save_bones=str(geometry_dir),
        n_samples_latent_recon=section.get("n_samples_latent_recon", 20_000),
        convergence_patience=section.get("convergence_patience", 10),
        rigid_reg_type=section.get("rigid_registration", "similarity"),
        acs_align=False,
        seed=seed,
    )
    print(f"  fitted in {time.time() - started:.0f} s")

    clusters = {
        "femur": section.get("bone_clusters_femur_tibia", 20_000),
        "tibia": section.get("bone_clusters_femur_tibia", 20_000),
        "patella": section.get("bone_clusters_patella", 10_000),
    }
    osim_meshes = {}
    for bone in ("femur", "tibia", "patella"):
        print(f"decoding {bone}")
        osim_meshes[bone] = nsm_recon_to_osim(
            bone=bone,
            dict_bones=bones,
            fem_ref_center=femur_center,
            bone_clusters=clusters[bone],
            cart_clusters=None,
            seed=seed,
        )

    print("building the OpenSim model")
    built = build_knee_model(
        config, bones=bones, osim_meshes=osim_meshes, center_mode=center_mode
    )
    print(f"built {built} in {time.time() - started:.0f} s total")

    if not model_file.exists():
        raise RuntimeError(
            f"model building reported {built} but the expected model is not at "
            f"{model_file}"
        )
    return True


def _stage_subject_meshes(knee: Knee, source: Path, staged: Path) -> Path:
    """Link a knee's segmentation meshes into ``staged`` and return it.

    Fitting reads from here, and writes its intermediate copies here under names
    that are not among the links, so nothing is written through a link into the
    inputs.
    """
    staged.mkdir(parents=True, exist_ok=True)
    for name in subject_mesh_files(knee).values():
        link = staged / name
        if not link.is_symlink():
            link.symlink_to((source / name).resolve())
    return staged


def _write_run_record(output_dir: Path, config: Config, knee: Knee) -> None:
    """Record what produced this reconstruction, next to it: config, code, machine."""
    from comak_knee_oa.manifest import _code_versions, environment_record

    record = {
        "knee": knee.name,
        "config_file": str(config.origin) if config.origin else None,
        "config": config.data,
        "code": _code_versions(),
        "environment": environment_record(),
    }
    (output_dir / "run_record.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n"
    )


def library_paths() -> dict[str, str]:
    """Where each numerical library was imported from.

    The shape-model and mesh libraries are run from exported snapshots at their
    pinned commits (``docs/pinned_versions.md``), which carry no git history, so
    the import location is what records which version ran.
    """
    import importlib

    out = {}
    for name in ("NSM", "pymskt", "nsosim", "pycomak", "torch", "opensim"):
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # recorded, not fatal: the record should still be written
            out[name] = f"import failed: {exc}"
            continue
        version = getattr(module, "__version__", "")
        out[name] = f"{Path(module.__file__).resolve().parent} {version}".strip()
    return out


# ----------------------------------------------------------------- rebuild ----
#
# A rebuild repeats model building (step 3) for a knee whose shape models were
# already fitted, starting from the surfaces that fit decoded. The fit is not
# repeated: its latents and alignment are what the knee *is*, and refitting would
# move every surface by the run-to-run variation of the fit itself, burying
# whatever change to model building the rebuild was for.

# The decoded surfaces model building consumes, as a build saved them under
# geometries_nsm_similarity/. The VTK copies, not the STLs beside them: an STL
# round trip re-lists the faces, and the fat pad -- built by casting rays along
# the bone's normals -- moves by ~0.2 mm when it does.
SAVED_SURFACES = {
    "femur": {
        "bone": "femur/femur_nsm_recon_osim.vtk",
        "cart": "femur/femur_cartilage_nsm_recon_osim.vtk",
        "med_men": "femur/med_men_osim.vtk",
        "lat_men": "femur/lat_men_osim.vtk",
    },
    "tibia": {
        "bone": "tibia/tibia_nsm_recon_osim.vtk",
        "cart": "tibia/tibia_cartilage_nsm_recon_osim.vtk",
    },
    # Before centring: model building centres the patella itself.
    "patella": {
        "bone": "patella/patella_nsm_recon_osim_original_position.vtk",
        "cart": "patella/patella_cartilage_nsm_recon_osim_original_position.vtk",
    },
}


def saved_fit_files(knee: Knee) -> list[str]:
    """Every saved file a rebuild reads, relative to ``geometries_nsm_similarity/``."""
    files = [path for surfaces in SAVED_SURFACES.values() for path in surfaces.values()]
    for bone in SHAPE_MODELS:
        files += [f"{bone}/{bone}_latent.npy", f"{bone}/{bone}_alignment.json"]
    return files


def fit_record_files(knee: Knee) -> list[str]:
    """What the shape-model fit itself wrote, relative to ``geometries_nsm_similarity/``.

    The subject meshes as aligned for fitting, each bone's latent and alignment,
    and its reconstruction in millimetres. A rebuild links these into its own tree
    so the rebuilt knee carries its fit like a reconstructed one -- the pressure
    transfer, for one, recovers its cross-model transform from the aligned meshes.
    """
    prefix = f"{knee.subject}_{knee.side}"
    files = [name.replace(".vtk", "_aligned.vtk") for name in subject_mesh_files(knee).values()]
    for bone in SHAPE_MODELS:
        files += [f"{bone}/{bone}_latent.npy", f"{bone}/{bone}_alignment.json"]
        files += [f"{bone}/{prefix}_{bone}_nsm_recon_mm.vtk",
                  f"{bone}/{prefix}_{bone}_cart_nsm_recon_mm.vtk"]
    files += [f"femur/{prefix}_{men}_nsm_recon_mm.vtk" for men in ("med_men", "lat_men")]
    return files


def _require_own_tree(output_dir: Path, fits_dir: Path) -> None:
    """Refuse a rebuild that could write over the fit it is reading, or out of the data root.

    Both checks are on resolved paths. A data root may reach the saved fits
    through a symlink into another root, and a write that followed it would land
    in that root -- so the output must resolve inside this data root itself.
    """
    root = data_root()
    out = output_dir.resolve()
    fits = fits_dir.resolve()
    if not out.is_relative_to(root):
        raise RuntimeError(
            f"rebuild output {output_dir} resolves to {out}, outside the data root "
            f"{root} (through a symlink?). A rebuild writes only into its own data "
            "root; set run.output_dir to a directory that lives there."
        )
    if out == fits or out.is_relative_to(fits) or fits.is_relative_to(out):
        raise RuntimeError(
            f"rebuild output {output_dir} overlaps the saved fits it reads ({fits_dir}). "
            "Model building writes files with the same names as its inputs; set "
            "run.output_dir to a new run."
        )


def _stat(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(config: Config, knee: Knee, check_only: bool = False) -> bool:
    """Rebuild one knee's OpenSim model from its saved shape-model fit.

    Reads the fit at ``inputs.saved_fits`` and builds into ``run.output_dir``
    with this config's model-building options -- the trim centre, the fat pad,
    everything ``build_options`` reads. Never fits.

    An existing model is accepted only if its run record shows it was rebuilt
    from the same saved files with the same options; anything else raises. A
    half-finished build must be removed by hand, because the model builder reuses
    whatever it finds in a non-empty model folder. Returns True on success.
    """
    config = config.for_knee(knee)
    problems = build_input_problems(config)
    fits_dir = output_dir = None
    try:
        fits_dir = config.path("inputs", "saved_fits")
        output_dir = config.path("run", "output_dir")
    except Exception as exc:  # ConfigError; reported with the rest
        problems.append(str(exc))
    if fits_dir is not None:
        geometry = fits_dir / "geometries_nsm_similarity"
        for name in saved_fit_files(knee) + fit_record_files(knee):
            if not (geometry / name).is_file():
                problems.append(f"saved fit file not found: {geometry / name}")
        if output_dir is not None:
            try:
                _require_own_tree(output_dir, fits_dir)
            except RuntimeError as exc:
                problems.append(str(exc))
    center_mode = meniscus_center_mode(config)
    if problems:
        message = "\n  ".join(dict.fromkeys(problems))
        if check_only:
            print(f"CANNOT REBUILD {knee.name}:\n  {message}")
            return False
        raise FileNotFoundError(f"cannot rebuild {knee.name}:\n  {message}")

    model_file = built_model_path(config)
    if check_only:
        print(f"OK {knee.name}: saved fit and build inputs resolve")
        print(f"   meniscus trim centre: {center_mode}")
        print(f"   fit    {fits_dir}")
        print(f"   model  {model_file}")
        return True

    geometry = fits_dir / "geometries_nsm_similarity"
    hashes = {name: _sha256(geometry / name) for name in saved_fit_files(knee)}
    if model_file.exists():
        _check_existing_rebuild(output_dir, config, center_mode, hashes)
        print(f"already built: {model_file}")
        return True
    if model_file.parent.is_dir() and any(model_file.parent.iterdir()):
        raise RuntimeError(f"{model_file.parent} holds a partial build; remove it and rerun")

    _require_gpu()
    require_center_mode_support(center_mode)
    from nsosim.model_building import build_dict_bones_for_interpolation
    from nsosim.utils import load_model
    from pymskt.mesh import Mesh

    started = time.time()
    linked = {name: _stat(geometry / name) for name in fit_record_files(knee)}
    _link_fit_records(knee, geometry, output_dir / "geometries_nsm_similarity")
    _write_rebuild_record(output_dir, config, knee, center_mode, fits_dir, hashes)

    models, latents, transforms, alignments = {}, {}, {}, {}
    for bone, name in SHAPE_MODELS.items():
        model_dir = config.path("inputs", "shape_models") / name
        model_config = json.loads((model_dir / "model_params_config.json").read_text())
        models[bone] = load_model(model_config, str(model_dir / "model" / "2000.pth"))
        latents[bone] = np.load(geometry / bone / f"{bone}_latent.npy")
        alignments[bone] = json.loads((geometry / bone / f"{bone}_alignment.json").read_text())
        transforms[bone] = np.array(alignments[bone]["linear_transform"])
    bones = build_dict_bones_for_interpolation(
        models=models,
        latents=latents,
        transforms=transforms,
        labeled_bone_dir=str(config.path("inputs", "labeled_bones")),
    )
    for bone in SHAPE_MODELS:
        # The fit's own scale and centre, rather than the helper's identity
        # defaults (the cohort's fits all happen to have the identity).
        recon = bones[bone]["subject"]["recon_dict"]
        recon["scale"] = alignments[bone]["scale"]
        recon["center"] = np.array(alignments[bone]["center"])
        bones[bone]["wrap"]["path_labeled_bone"] = str(labeled_bone_path(config, bone))

    osim_meshes = {
        bone: {role: Mesh(str(geometry / name)) for role, name in surfaces.items()}
        for bone, surfaces in SAVED_SURFACES.items()
    }

    print(f"rebuilding {knee.name} from {fits_dir}")
    built = build_knee_model(config, bones=bones, osim_meshes=osim_meshes, center_mode=center_mode)
    print(f"built {built} in {time.time() - started:.0f} s")
    # The fit's records are linked into the rebuilt tree, and a write to a linked
    # name would land in the fit itself. No builder output shares a name with
    # them; this proves it stayed that way for this knee.
    changed = [name for name, digest in hashes.items() if _sha256(geometry / name) != digest]
    changed += [name for name, stat in linked.items() if _stat(geometry / name) != stat]
    if changed:
        raise RuntimeError(f"model building modified the saved fit it read: {changed}")
    if not model_file.exists():
        raise RuntimeError(
            f"model building reported {built} but the expected model is not at {model_file}"
        )
    return True


def _link_fit_records(knee: Knee, source: Path, target: Path) -> None:
    """Link the fit's own files into the rebuilt tree (relative links, never copies)."""
    for name in fit_record_files(knee):
        link = target / name
        link.parent.mkdir(parents=True, exist_ok=True)
        real = (source / name).resolve()
        if link.is_symlink() and link.resolve() == real:
            continue
        link.symlink_to(os.path.relpath(real, link.parent.resolve()))


def _rebuild_record(config: Config, knee: Knee, center_mode: str,
                    hashes: dict[str, str]) -> dict:
    """What a rebuild depends on; an existing build must match all of it."""
    return {
        "knee": knee.name,
        "rebuilt_from": config.require("inputs", "saved_fits"),
        "saved_fit_sha256": hashes,
        "build_options": json.loads(json.dumps(build_options(config, center_mode), default=str)),
        "project_coronary": False,
    }


def _write_rebuild_record(output_dir: Path, config: Config, knee: Knee, center_mode: str,
                          fits_dir: Path, hashes: dict[str, str]) -> None:
    """Record what produced this rebuild, next to it: inputs, options, code, machine.

    The machine (``environment``: host, CPU, scheduler job, data root, run,
    library locations) is recorded, never compared: ``_rebuild_record`` alone
    decides whether an existing build matches.
    """
    from comak_knee_oa.manifest import _code_versions, environment_record

    record = _rebuild_record(config, knee, center_mode, hashes)
    record.update({
        "stage": "rebuild",
        "rebuilt_from_resolved": str(fits_dir.resolve()),
        "config_file": str(config.origin) if config.origin else None,
        "config": config.data,
        "node": socket.gethostname(),
        "gpu": _gpu_name(),
        "code": _code_versions(),
        "libraries": library_paths(),
        "environment": environment_record(),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_record.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


def _check_existing_rebuild(output_dir: Path, config: Config, center_mode: str,
                            hashes: dict[str, str]) -> None:
    """Refuse an existing model rebuilt from other saved files or other options."""
    record_file = output_dir / "run_record.json"
    if not record_file.is_file():
        raise RuntimeError(f"{output_dir} has a model but no run_record.json; "
                           "cannot tell what it was built from")
    record = json.loads(record_file.read_text())
    if record.get("stage") != "rebuild":
        raise RuntimeError(f"{output_dir} holds a model that was not rebuilt from a saved fit")
    expected = _rebuild_record(config, config.knee, center_mode, hashes)
    differ = sorted(key for key in expected if record.get(key) != expected[key])
    if differ:
        raise RuntimeError(
            f"existing model in {output_dir} does not match this rebuild ({', '.join(differ)} "
            "differ). Rebuild into a new run.output_dir."
        )


def _gpu_name() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None
