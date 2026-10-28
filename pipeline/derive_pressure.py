"""Stage 03, pressure: simulated contact pressure on the reference surface.

The simulations write per-vertex pressure on each knee's own articular
surface, in its own coordinate frame. Analysis needs those fields on the same
shared 20,000-vertex reference surface the thickness track uses, so that
vertex *j* means the same anatomical location everywhere. A first subcommand
summarises each simulation over stance, and three more build that bridge:

``stats``
    Collapse one gait condition's per-frame tibiofemoral contact surfaces into
    one mesh per bone carrying per-vertex maximum, mean and cumulative
    (pressure-time integral) pressure over stance, saved beside the
    simulation as ``{bone}_baseline_pressure_stats.vtk``. The frame count and
    time step come from the simulation's own ``joint_mechanics.h5``, because
    the stance window differs between gait donors. ``transfer`` reads these.
    (CPU)

``transform``
    The simulation anatomy and the thickness track are separate shape-model
    fits (femur model 568 against 647; the tibia model 650 is shared, but
    fitted and registered separately), so their frames differ per subject.
    This recovers a per-subject 4x4 transform, ``T_reg_to_647``, from the
    simulation model's registered frame to the thickness fit's frame, through
    the original MRI frame:
    the knee's deterministic alignment is re-derived by least squares from
    the (segmentation mesh, aligned mesh) pair, composed with the thickness
    fit's recorded registration, and validated by round trip. Left knees get
    an extra reflection pair, because the two pipelines mirrored left volumes
    about different planes. (CPU)

``walk``
    One walk of the reference surface through thickness-model latent space
    from the population mean to each subject, cached as vertex positions.
    (The thickness track walks the other way, subject to mean.)
    The walk depends only on the subject's shape -- not on any gait -- so
    computing it once lets every gait condition share the identical
    correspondence operator. (GPU)

``transfer``
    Carry one gait condition's pressure onto the reference: place the
    subject's articular surface (carrying the gait's pressure values) into
    thickness-model space via the per-subject transform, then transfer the
    values to the cached walked surface by weighted nearest neighbours and
    read them back onto the reference by vertex index. (CPU -- this is why
    nine of ten gait conditions never need a GPU here.)

Inputs: the simulations under ``02_simulations``, the knees' models under
``01_models`` (aligned meshes, true-size articular surfaces), the segmentation
meshes and the thickness-model fits under ``00_inputs`` (``STANFORD`` in their
paths is a fixed directory level of the vendored fit tree). The reference gait
condition is named ``default`` in derived outputs. Outputs, under
``03_derived``: ``pressure_on_reference/`` (the transforms
``T_reg_to_647_per_subject.json`` and the cached walks ``<bone>/walked_out_xyz/``)
and ``pressure_on_reference_<gait>/<bone>/<knee>_<bone>_pressure_ref.{vtk,npz}``
per condition, where unsimulated reference vertices carry -1.

Running it
----------
    python pipeline/derive_pressure.py stats --config config/measured_gait/Subject_102.json
    python pipeline/derive_pressure.py transform
    python pipeline/derive_pressure.py walk
    python pipeline/derive_pressure.py transfer --config config/measured_gait/Subject_102.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa import cohorts, provenance  # noqa: E402
from comak_knee_oa.config import Config, Knee, load_config  # noqa: E402
from comak_knee_oa.paths import declaration, run_name, stage_dir  # noqa: E402
from derive_thickness import (  # noqa: E402
    BONES,
    cohort,
    latent_path,
    recon_results_path,
    reference_surface_path,
    _load_shape_model,
)

# The stage-01 run whose geometry anchors the pressure track -- the aligned
# meshes the transform is recovered from, and the true-size articular surfaces
# the transfer reads -- is the run the data root holds (``models_run``). A root
# that declares no run is the v2 root (an earlier release of the dataset that
# the code can still read; v3 is the run this paper reports), and its models
# are under cohorts.V2_MODELS_RUN.

# The transform must reproduce the direct path to numerical precision -- it is
# an exact composition of exact transforms, so anything above rounding error
# means the wrong pair of files was combined.
ROUNDTRIP_TOLERANCE_MM = 1e-9

# A benign triangle re-listing leaves every vertex's normalized centroid
# distance unchanged; a genuine vertex permutation moves it far more. Two
# percent of the mean radius cleanly separates the cases.
RADIUS_SIGNATURE_TOLERANCE = 0.02


# ------------------------------------------------------------------ paths ----

# The run configs call the reference condition "reference"; the cohort module,
# the derived trees, the combined matrices and the QC table call it "default".
# Mapped here, once, for every stage-03 step that names a condition's output.
_CANONICAL_CONDITION = {"reference": cohorts.REFERENCE}


def condition_name(config: Config) -> str:
    """The name a gait condition's derived outputs are filed under."""
    gait = config.require("run", "gait")
    return _CANONICAL_CONDITION.get(gait, gait)


def models_run() -> str:
    """The ``01_models`` run the pressure track reads geometry from.

    The active root's own run, so pressure lands on the surfaces that root's
    own simulations used; on a root that declares no run (the v2 root), the run
    that produced it.
    """
    decl = declaration()
    return run_name() if decl and decl.get("run_name") else cohorts.V2_MODELS_RUN


def segmentation_mesh_path(knee: Knee, bone: str) -> Path:
    return (
        stage_dir("inputs", "meshes", knee.timepoint, knee.subject)
        / f"{knee.subject}_{knee.side}_{bone}.vtk"
    )


def aligned_mesh_path(knee: Knee, bone: str) -> Path:
    return (
        stage_dir("models", models_run(), knee.directory, "geometries_nsm_similarity")
        / f"{knee.subject}_{knee.side}_{bone}_aligned.vtk"
    )


def articular_surface_path(knee: Knee, bone: str) -> Path:
    return (
        stage_dir("models", models_run(), knee.directory, "geometries_nsm_similarity", bone)
        / f"{bone}_articular_surface_osim.vtk"
    )


def left_segmentation_path(knee: Knee) -> Path:
    return (
        stage_dir("inputs", "segmentations", knee.timepoint, knee.subject)
        / "SAG_3D_DESS_LEFT_orig_med_lat_tib_cart_no_correction-label.nrrd"
    )


def flipped_femur_path(knee: Knee) -> Path | None:
    parent = stage_dir(
        "inputs", "latents_thickness", "V00", "STANFORD", "sub_region_surfaces"
    )
    matches = sorted(parent.glob(f"{knee.subject}_LEFT_*/{knee.subject}_LEFT_femur_mesh_flipped.vtk"))
    return matches[0] if matches else None


def pressure_cache_dir() -> Path:
    return stage_dir("derived", "pressure_on_reference")


def transform_file() -> Path:
    return pressure_cache_dir() / "T_reg_to_647_per_subject.json"


def walked_out_path(knee: Knee, bone: str) -> Path:
    return (
        pressure_cache_dir() / bone / "walked_out_xyz"
        / f"{knee.subject}_{knee.side}_{bone}_xyz.npy"
    )


def femur_reference_center() -> np.ndarray:
    """The reference femur's center, used for both bones.

    The model's tibia body frame has its origin at the knee joint -- the femur
    center -- so the tibia's own center is the wrong offset here.
    """
    alignment = json.loads(
        (stage_dir("inputs", "reference", "nsm_meshes", "femur") / "ref_femur_alignment.json")
        .read_text()
    )
    return np.array(alignment["mean_orig"])


# ------------------------------------------------------------------ stats ----

# The post-COMAK joint-mechanics step writes one contact surface per frame,
# numbered by frame index.
CONTACT_FRAME_GLOB = "*contact_{bone}_cartilage_dynamic_ground_ground_*.vtp"

# The tibiofemoral contact's pressure, as carried by each cartilage surface:
# the tibia is the contact's casting mesh, the femur its target.
TF_PRESSURE_SCALAR = {
    "tibia": "casting_triangle_pressure_tf_contact",
    "femur": "target_triangle_pressure_tf_contact",
}


def pressure_stats_path(results_dir: Path, bone: str) -> Path:
    return results_dir / f"{bone}_baseline_pressure_stats.vtk"


def read_time_base(h5: Path) -> tuple[int, float]:
    """(frame count, time step in s) from a simulation's joint-mechanics h5.

    Read rather than assumed: the stance window, and so the frame count,
    differs between gait donors.
    """
    import h5py

    with h5py.File(h5, "r") as handle:
        time_s = handle["time"][:]
    if len(time_s) < 2:
        raise ValueError(f"time vector has {len(time_s)} samples in {h5}")
    return int(len(time_s)), float(time_s[1] - time_s[0])


def contact_frames(results_dir: Path, bone: str) -> list[Path]:
    """The bone's per-frame contact surfaces, in frame order (not name order)."""
    frames = (results_dir / "joint-mechanics").glob(CONTACT_FRAME_GLOB.format(bone=bone))
    return sorted(frames, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def stats_one(results_dir: Path, bone: str, n_frames: int, time_step: float) -> Path:
    """Per-vertex max, cumulative and mean TF pressure over stance, for one bone.

    The output is the first frame's surface (its point data included) with the
    three summaries added. Cumulative is the pressure-time integral, the sum of
    pressure x time step; mean is over all frames, contacting or not.
    """
    import pymskt as mskt

    frames = contact_frames(results_dir, bone)
    if len(frames) != n_frames:
        raise ValueError(
            f"h5 reports {n_frames} frames, found {len(frames)} contact surfaces"
        )
    scalar = TF_PRESSURE_SCALAR[bone]
    for i, frame in enumerate(frames):
        mesh = mskt.mesh.Mesh(str(frame))
        mesh.transfer_cell_data_to_points()
        values = mesh.get_scalar(scalar)
        if i == 0:
            summary = mesh.copy()
            max_pressure = values.copy()
            cumulative = np.zeros_like(values)
            mean = np.zeros_like(values)
        else:
            max_pressure = np.maximum(max_pressure, values)
        cumulative += values * time_step
        mean += values * (1.0 / n_frames)

    summary.set_scalar("max_tf_pressure", max_pressure)
    summary.set_scalar("cumulative_tf_pressure", cumulative)
    summary.set_scalar("mean_tf_pressure", mean)
    output = pressure_stats_path(results_dir, bone)
    summary.save_mesh(str(output))
    return output


def run_stats(args) -> int:
    """Summarise one gait condition's simulations, for every knee that has one.

    Recomputes by default; ``--skip-existing`` passes over knees whose two
    summaries both exist, so a sweep over a cohort still in flight touches only
    new knees.
    """
    config = load_config(args.config)
    gait = config.require("run", "gait")
    knees = [Knee.parse(name) for name in (args.knee or cohort())]
    if args.n_shards > 1:
        if not 0 <= args.shard < args.n_shards:
            raise SystemExit(f"--shard {args.shard} out of range for --n-shards {args.n_shards}")
        knees = knees[args.shard :: args.n_shards]

    work, existing, missing = [], 0, 0
    for knee in knees:
        results_dir = config.for_knee(knee).path("run", "results_dir")
        if args.skip_existing and all(
            pressure_stats_path(results_dir, bone).is_file() for bone in TF_PRESSURE_SCALAR
        ):
            existing += 1
        elif not (results_dir / "joint-mechanics" / "joint_mechanics.h5").is_file():
            missing += 1
        else:
            work.append((knee, results_dir))
    print(
        f"{gait}: {len(work)} to summarise, {existing} already done, "
        f"{missing} without joint_mechanics.h5"
    )
    if args.limit:
        work = work[: args.limit]
    if args.check or not work:
        return 0

    failed = 0
    started = time.time()
    for i, (knee, results_dir) in enumerate(work, 1):
        try:
            n_frames, time_step = read_time_base(
                results_dir / "joint-mechanics" / "joint_mechanics.h5"
            )
        except Exception as exc:
            failed += 1
            print(f"FAILED {knee.name}: {exc}")
            continue
        for bone in TF_PRESSURE_SCALAR:
            try:
                stats_one(results_dir, bone, n_frames, time_step)
            except Exception as exc:
                failed += 1
                print(f"FAILED {knee.name} {bone}: {exc}")
        if i % 10 == 0:
            rate = (time.time() - started) / i
            print(f"  {i}/{len(work)} ({rate:.1f} s each)", flush=True)
    print(f"summarised {len(work)} knees, {failed} knee-bone failures")
    return 1 if failed else 0


# ------------------------------------------------------------- transforms ----


def hom(points: np.ndarray) -> np.ndarray:
    return np.hstack([points, np.ones((len(points), 1))])


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return (hom(points) @ T.T)[:, :3]


def recover_alignment(original: Path, aligned: Path):
    """Recover the similarity transform between a mesh and its aligned copy.

    The alignment is deterministic and applied vertex-for-vertex, so a least
    squares fit over the vertex pair reproduces it to rounding error; the residual is
    the proof that these two files are the pair they claim to be.

    Returns (4x4 transform, original points, max residual in mm).
    """
    import pyvista as pv

    source = pv.read(original).points.astype(np.float64)
    target = pv.read(aligned).points.astype(np.float64)
    solution, *_ = np.linalg.lstsq(hom(source), target, rcond=None)
    T = np.eye(4)
    T[:3, :] = solution.T
    residual = np.linalg.norm(apply_transform(source, T) - target, axis=1).max()
    return T, source, float(residual)


def volume_flip_matrix(segmentation: Path) -> np.ndarray:
    """The reflection the meshing pipeline applied to left knee volumes.

    Left segmentation volumes were flipped along the sagittal slice axis. In
    physical space that is a reflection about the plane through the volume
    center with the slice direction as its normal -- for an oblique scan this
    is not a pure x-flip, which is why it is computed from the image header
    rather than assumed.
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(str(segmentation))
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    size = np.array(image.GetSize())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    normal = direction[:, 2]
    center = origin + direction @ (spacing * np.array([0, 0, (size[2] - 1) / 2.0]))
    M = np.eye(4)
    M[:3, :3] = np.eye(3) - 2 * np.outer(normal, normal)
    M[:3, 3] = 2 * (normal @ center) * normal
    return M


def thickness_flip_matrix(flipped_femur: Path) -> np.ndarray:
    """The reflection the thickness fits applied to left knees.

    A pure x-reflection about the femur's x-centroid. The flip preserves that
    centroid, so it can be read directly from the flipped mesh.
    """
    import pyvista as pv

    cx = float(pv.read(flipped_femur).points[:, 0].mean())
    M = np.eye(4)
    M[0, 0] = -1
    M[0, 3] = 2 * cx
    return M


def flip_correction(knee: Knee) -> np.ndarray | None:
    """For a left knee, the reflection pair bridging the two mirror conventions.

    Both component matrices are self-inverse reflections; their composition
    undoes the volume flip and applies the thickness track's flip. The femur's
    correction serves every bone, since both pipelines mirrored the whole knee
    about femur-derived geometry.
    """
    if knee.side != "LEFT":
        return None
    segmentation = left_segmentation_path(knee)
    femur = flipped_femur_path(knee)
    if not segmentation.is_file() or femur is None:
        return None
    return thickness_flip_matrix(femur) @ volume_flip_matrix(segmentation)


def run_transform(args) -> int:
    """Compute and validate T_reg_to_647 for every knee, per bone."""
    output_dir = stage_dir("derived", args.output_subdir, create=not args.check)
    knees = [Knee.parse(name) for name in cohort()]

    corrections: dict[str, np.ndarray] = {}
    transforms: dict[str, dict] = {bone: {} for bone in BONES}
    validation: dict[str, list] = {bone: [] for bone in BONES}
    skipped: dict[str, list] = {bone: [] for bone in BONES}

    ready = [
        knee for knee in knees
        if aligned_mesh_path(knee, "femur").is_file()
        and recon_results_path(knee.name, "V00", "femur").is_file()
    ]
    print(f"{len(ready)}/{len(knees)} knees have both an aligned mesh and a V00 fit")
    if args.limit:
        ready = ready[: args.limit]
    if args.check or not ready:
        return 0

    started = time.time()
    for i, knee in enumerate(ready, 1):
        if knee.side == "LEFT" and knee.name not in corrections:
            correction = flip_correction(knee)
            if correction is not None:
                corrections[knee.name] = correction
        for bone in BONES:
            original = segmentation_mesh_path(knee, bone)
            aligned = aligned_mesh_path(knee, bone)
            recon = recon_results_path(knee.name, "V00", bone)
            missing = [str(p) for p in (original, aligned, recon) if not p.is_file()]
            if missing:
                skipped[bone].append({"knee": knee.name, "reason": f"missing: {missing[0]}"})
                continue
            if knee.side == "LEFT" and knee.name not in corrections:
                skipped[bone].append(
                    {"knee": knee.name, "reason": "no flip correction (missing inputs)"}
                )
                continue
            try:
                T_alignment, source, residual = recover_alignment(original, aligned)
                T_647 = np.array(json.loads(recon.read_text())["icp_transform"])
                correction = corrections.get(knee.name)
                if correction is not None:
                    T = T_647 @ correction @ np.linalg.inv(T_alignment)
                    direct = apply_transform(apply_transform(source, correction), T_647)
                else:
                    T = T_647 @ np.linalg.inv(T_alignment)
                    direct = apply_transform(source, T_647)
                chained = apply_transform(apply_transform(source, T_alignment), T)
                roundtrip = float(np.linalg.norm(chained - direct, axis=1).max())
                transforms[bone][knee.name] = T.tolist()
                validation[bone].append(
                    {
                        "knee": knee.name,
                        "alignment_residual_mm": residual,
                        "roundtrip_max_error_mm": roundtrip,
                    }
                )
            except Exception as exc:
                skipped[bone].append({"knee": knee.name, "reason": f"error: {exc}"})
        if i % 100 == 0:
            print(f"  {i}/{len(ready)} ({time.time() - started:.0f} s)", flush=True)

    left = [knee for knee in ready if knee.side == "LEFT"]
    if left and not corrections:
        raise SystemExit(
            f"{len(left)} left knees processed but zero flip corrections computed -- "
            "the flip inputs are not resolving. A transform file covering only "
            "right knees is wrong for half the cohort, so nothing was written."
        )

    worst = max(
        (entry["roundtrip_max_error_mm"] for bone in BONES for entry in validation[bone]),
        default=0.0,
    )
    print(f"worst round-trip error: {worst:.2e} mm (tolerance {ROUNDTRIP_TOLERANCE_MM})")

    payload = {
        "description": "Per-knee 4x4 transforms from the aligned simulation frame "
        "to the thickness models' space. Row-vector convention: "
        "points_647 = (hom(points) @ T.T)[:, :3].",
        "bones": list(BONES),
        "n_left_flip_corrections": len(corrections),
        "transforms": {
            bone: {"n_subjects": len(transforms[bone]), "T_reg_to_647": transforms[bone]}
            for bone in BONES
        },
    }
    (output_dir / "T_reg_to_647_per_subject.json").write_text(json.dumps(payload) + "\n")
    (output_dir / "cross_model_transform_validation.json").write_text(
        json.dumps(
            {"tolerance_mm": ROUNDTRIP_TOLERANCE_MM, "validation": validation, "skipped": skipped},
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {output_dir / 'T_reg_to_647_per_subject.json'}")
    return 0 if worst < ROUNDTRIP_TOLERANCE_MM else 1


# ------------------------------------------------------------------- walk ----


def run_walk(args) -> int:
    """Cache the mean-to-subject walk of the reference surface, per knee-bone."""
    from pymskt.mesh import Mesh

    bones = BONES if args.bone == "both" else (args.bone,)
    work, skipped, missing = [], 0, 0
    for name in cohort():
        knee = Knee.parse(name)
        for bone in bones:
            if walked_out_path(knee, bone).is_file():
                skipped += 1
            elif not latent_path(knee.name, "V00", bone).is_file():
                missing += 1
            else:
                work.append((knee, bone))
    print(f"{len(work)} to walk, {skipped} cached, {missing} without a V00 fit")
    if args.limit:
        work = work[: args.limit]
    if args.check or not work:
        return 0

    from NSM.mesh.interpolate import interpolate_mesh

    needed = sorted({bone for _, bone in work})
    models = {bone: _load_shape_model(bone) for bone in needed}
    references = {bone: Mesh(str(reference_surface_path(bone))) for bone in needed}
    failed = 0
    started = time.time()
    for i, (knee, bone) in enumerate(work, 1):
        try:
            latent = np.load(latent_path(knee.name, "V00", bone)).flatten()
            # Mean to subject, vertex identity preserved. smooth=False is
            # required: the surfaces this correspondence serves are open
            # meshes, and VTK's Laplacian smoothing crashes on open meshes.
            walked = interpolate_mesh(
                model=models[bone],
                latent1=np.zeros_like(latent),
                latent2=latent,
                n_steps=100,
                mesh=references[bone].copy(),
                surface_idx=1,
                verbose=False,
                spherical=False,
                smooth=False,
            )
            positions = walked.point_coords.astype(np.float64)
            output = walked_out_path(knee, bone)
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, positions)
            np.save(
                output.with_name(f"{knee.subject}_{knee.side}_{bone}_displacement.npy"),
                positions - references[bone].point_coords,
            )
        except Exception as exc:
            failed += 1
            print(f"FAILED {knee.name} {bone}: {exc}")
        if i % 25 == 0:
            rate = (time.time() - started) / i
            print(f"  {i}/{len(work)} ({rate:.0f} s each)", flush=True)
    print(f"walked {len(work) - failed}, failed {failed}")
    return 1 if failed else 0


# --------------------------------------------------------------- transfer ----


def _radius_signature(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0)
    radius = np.linalg.norm(centered, axis=1)
    return radius / radius.mean()


def verify_point_order(pressure_mesh, articular_mesh) -> tuple[bool, str]:
    """Prove pressure-VTK vertex i is articular-VTK vertex i before indexing by it.

    The transfer assigns pressure onto the articular geometry by vertex index,
    and an equal point count does not prove equal order -- a mesh can round-trip
    through STL with a small fraction of its vertices permuted. Identical face
    arrays settle it immediately; otherwise the per-vertex radius signature is
    compared, which is invariant to the two meshes being in different frames.
    """
    faces_pressure = np.asarray(pressure_mesh.mesh.faces)
    faces_articular = np.asarray(articular_mesh.mesh.faces)
    if np.array_equal(faces_pressure, faces_articular):
        return True, "faces identical"
    deviation = float(
        np.abs(
            _radius_signature(pressure_mesh.point_coords)
            - _radius_signature(articular_mesh.point_coords)
        ).max()
    )
    if deviation <= RADIUS_SIGNATURE_TOLERANCE:
        return True, f"faces re-listed, point order intact (deviation {deviation:.4f})"
    return False, (
        f"point order differs (radius-signature deviation {deviation:.3f} > "
        f"{RADIUS_SIGNATURE_TOLERANCE}); assigning by index would scramble the field"
    )


def transfer_one(knee: Knee, bone: str, pressure_vtk: Path, T: np.ndarray,
                 femur_center: np.ndarray, reference, output_dir: Path) -> Path:
    """Carry one knee-bone's gait pressure onto the reference surface."""
    import pyvista as pv
    from nsosim.nsm_fitting import convert_OSIM_to_nsm_
    from pymskt.mesh import Mesh
    from scipy.spatial import cKDTree

    pressure_mesh = Mesh(str(pressure_vtk))
    articular_mesh = Mesh(str(articular_surface_path(knee, bone)))
    if pressure_mesh.point_coords.shape[0] != articular_mesh.point_coords.shape[0]:
        raise ValueError(
            f"point counts differ: pressure {pressure_mesh.point_coords.shape[0]}, "
            f"articular {articular_mesh.point_coords.shape[0]}"
        )
    ok, reason = verify_point_order(pressure_mesh, articular_mesh)
    if not ok:
        raise ValueError(reason)

    # The pressure values, on the true-size articular geometry, carried into
    # the thickness models' space: local frame -> registered mm -> per-subject
    # transform. convert_OSIM_to_nsm_ mutates its argument; hand it a copy.
    pressure_mesh.point_coords = articular_mesh.point_coords.copy()
    registered = convert_OSIM_to_nsm_(pressure_mesh.point_coords.copy(), femur_center)
    points_647 = apply_transform(registered, T)

    source_poly = pv.PolyData(points_647.copy(), pressure_mesh.mesh.faces)
    for name in pressure_mesh.scalar_names:
        values = pressure_mesh.get_scalar(name)
        if values is not None:
            source_poly[name] = values
    source = Mesh(mesh=source_poly)

    walked_xyz = np.load(walked_out_path(knee, bone))
    if walked_xyz.shape[0] != reference.point_coords.shape[0]:
        raise ValueError(
            f"cached walk has {walked_xyz.shape[0]} points, reference has "
            f"{reference.point_coords.shape[0]}"
        )
    walked = reference.copy()
    walked.point_coords = walked_xyz

    max_dist = 1.1 * np.max(source.edge_lengths)
    walked.copy_scalars_from_other_mesh_to_current(
        source, weighted_avg=True, n_closest=3, max_dist=max_dist
    )

    # Back onto the reference by vertex index -- valid because the walk
    # preserved vertex identity. A vertex farther than max_dist from any
    # source point has no data; -1 marks it, since zero pressure is a real
    # value meaning "no contact".
    result = reference.copy()
    for name in walked.scalar_names:
        values = walked.get_scalar(name)
        if values is not None:
            result.set_scalar(name, values)

    # The reference surface is closed and the simulated articular surface is
    # open, so a zero can mean "simulated, no contact" or "not simulated".
    # Vertices farther than max_dist from the simulated surface are the second
    # kind and are marked -1 (has_data = 0).
    distances, _ = cKDTree(source.point_coords).query(walked.point_coords)
    has_data = (distances <= max_dist).astype(np.float64)
    no_data = has_data == 0

    arrays = {}
    for name in result.scalar_names:
        values = result.get_scalar(name)
        if values is not None:
            cleaned = np.where(np.isfinite(values), values, 0.0)
            cleaned[no_data] = -1.0
            result.set_scalar(name, cleaned)
            arrays[name] = cleaned
    result.set_scalar("has_data", has_data)

    output_dir_bone = output_dir / bone
    output_dir_bone.mkdir(parents=True, exist_ok=True)
    base = f"{knee.subject}_{knee.side}_{bone}_pressure_ref"
    result.save_mesh(str(output_dir_bone / f"{base}.vtk"))
    np.savez(output_dir_bone / f"{base}.npz", **arrays, has_data=has_data)
    return output_dir_bone / f"{base}.vtk"


# Each transferred knee-bone carries a provenance record, so a later run reuses an
# output only if it was made from the same inputs (comak_knee_oa.provenance).
# The outputs share one directory per bone -- the layout every reader globs --
# so the records live beside them, one unit per knee-bone.
TRANSFER_CODE = ("nsosim", "pymskt")


def transfer_outputs(output_dir: Path, knee: Knee, bone: str) -> list[Path]:
    base = output_dir / bone / f"{knee.subject}_{knee.side}_{bone}_pressure_ref"
    return [base.with_suffix(".vtk"), base.with_suffix(".npz")]


def transfer_record_unit(output_dir: Path, knee: Knee, bone: str) -> Path:
    return output_dir / "_provenance" / bone / f"{knee.subject}_{knee.side}"


def transfer_fingerprint(knee: Knee, bone: str, pressure_vtk: Path, T: np.ndarray,
                         femur_center: np.ndarray) -> dict:
    """What one knee-bone's transfer is made from: its four input files, the
    knee's own cross-model transform, the reference centre, the libraries."""
    return provenance.fingerprint(
        inputs={
            "pressure_stats": pressure_vtk,
            "articular_surface": articular_surface_path(knee, bone),
            "walked_out": walked_out_path(knee, bone),
            "reference_surface": reference_surface_path(bone),
        },
        options={
            "function": "derive_pressure.transfer_one",
            "T_reg_to_647": np.asarray(T, dtype=np.float64).tolist(),
            "femur_center": np.asarray(femur_center, dtype=np.float64).tolist(),
        },
        code=TRANSFER_CODE,
    )


def transfer_decision(output_dir: Path, knee: Knee, bone: str, expected, force: bool) -> str:
    """``"skip"`` or ``"build"`` for one knee-bone; raises ProvenanceMismatch.

    ``expected`` is the fingerprint, or a callable that computes it: inputs are
    hashed only when there is something on disk to compare them with.
    """
    unit = transfer_record_unit(output_dir, knee, bone)
    outputs = transfer_outputs(output_dir, knee, bone)
    if provenance.read_record(unit) is None:
        present = [o for o in outputs if o.exists()]
        if present and not force:
            raise provenance.ProvenanceMismatch(
                f"transferred pressure {present[0]} exists but carries no record at "
                f"{unit}: not made by this code (an as-run or copied output). Refusing "
                "to reuse it. Pass --force to transfer again, or move it aside."
            )
        return "build"
    expected = expected() if callable(expected) else expected
    return provenance.decide(unit, expected, complete=all(o.is_file() for o in outputs),
                             force=force, what="transferred pressure")


def run_transfer(args) -> int:
    """Transfer one gait condition's pressure, for every knee that has it.

    The gait is named by the same config file that ran its simulations, so
    a condition's location is defined in one place only. A knee-bone
    whose output exists is skipped only if its provenance record shows the same
    inputs; an output made from anything else, or with no record, stops the
    run before any work starts (``--force`` transfers it again).
    """
    from pymskt.mesh import Mesh

    config = load_config(args.config)
    gait = condition_name(config)
    output_dir = stage_dir("derived", f"pressure_on_reference_{gait}", create=not args.check)

    transforms = json.loads(transform_file().read_text())["transforms"]
    bones = BONES if args.bone == "both" else (args.bone,)
    femur_center = femur_reference_center()

    work, skipped, missing, refused = [], 0, 0, []
    for name in cohort():
        knee = Knee.parse(name)
        results_dir = config.for_knee(knee).path("run", "results_dir")
        for bone in bones:
            pressure_vtk = results_dir / f"{bone}_baseline_pressure_stats.vtk"
            if (
                not pressure_vtk.is_file()
                or not articular_surface_path(knee, bone).is_file()
                or not walked_out_path(knee, bone).is_file()
                or knee.name not in transforms[bone]["T_reg_to_647"]
            ):
                missing += 1
                continue
            T = np.array(transforms[bone]["T_reg_to_647"][knee.name])
            try:
                decision = transfer_decision(
                    output_dir, knee, bone,
                    lambda: transfer_fingerprint(knee, bone, pressure_vtk, T, femur_center),
                    force=args.force)
            except provenance.ProvenanceMismatch as exc:
                refused.append(f"{knee.name} {bone}: {exc}")
                continue
            if decision == "skip":
                skipped += 1
            else:
                work.append((knee, bone, pressure_vtk, T))
    print(
        f"{gait}: {len(work)} to transfer, {skipped} already done, "
        f"{missing} missing an input (pressure stats, geometry, walk, or transform)"
        + (f", {len(refused)} REFUSED" if refused else "")
    )
    if refused:
        raise provenance.ProvenanceMismatch(
            f"{len(refused)} existing transfer output(s) were not made from this run's "
            "inputs; nothing was transferred. First:\n  " + "\n  ".join(refused[:5])
        )
    if args.limit:
        work = work[: args.limit]
    if args.check or not work:
        return 0

    references = {
        bone: Mesh(str(reference_surface_path(bone)))
        for bone in sorted({bone for _, bone, _, _ in work})
    }
    failed = 0
    started = time.time()
    for i, (knee, bone, pressure_vtk, T) in enumerate(work, 1):
        unit = transfer_record_unit(output_dir, knee, bone)
        try:
            expected = transfer_fingerprint(knee, bone, pressure_vtk, T, femur_center)
            provenance.mark_started(unit, expected)
            transfer_one(knee, bone, pressure_vtk, T, femur_center,
                         references[bone], output_dir)
            provenance.mark_complete(unit, expected)
        except Exception as exc:
            failed += 1
            print(f"FAILED {knee.name} {bone}: {exc}")
        if i % 50 == 0:
            rate = (time.time() - started) / i
            print(f"  {i}/{len(work)} ({rate:.1f} s each)", flush=True)
    print(f"transferred {len(work) - failed}, failed {failed}")
    return 1 if failed else 0


# -------------------------------------------------------------------- cli ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="derive_pressure.py",
        description="Contact pressure on the reference surface.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="one gait's per-vertex pressure over stance")
    p_stats.add_argument(
        "--config", required=True,
        help="the gait condition's run config; names the simulations to read",
    )
    p_stats.add_argument(
        "--knee", nargs="+", default=None,
        help="only these knees (subject_SIDE); default the whole cohort",
    )
    p_stats.add_argument(
        "--skip-existing", action="store_true",
        help="pass over knees whose femur and tibia summaries both exist",
    )
    p_stats.add_argument("--n-shards", type=int, default=1,
                         help="split the knees into this many independent shards")
    p_stats.add_argument("--shard", type=int, default=0,
                         help="which shard this run processes (0-based)")

    p_transform = sub.add_parser("transform", help="per-knee cross-model transforms")
    p_transform.add_argument(
        "--output-subdir", default="pressure_on_reference",
        help="directory under 03_derived to write into",
    )

    p_walk = sub.add_parser("walk", help="cache the mean-to-subject walks (GPU)")
    p_walk.add_argument("--bone", choices=["femur", "tibia", "both"], default="both")

    p_transfer = sub.add_parser("transfer", help="one gait's pressure onto the reference")
    p_transfer.add_argument(
        "--config", required=True,
        help="the gait condition's run config; names the simulations to read",
    )
    p_transfer.add_argument("--bone", choices=["femur", "tibia", "both"], default="both")
    p_transfer.add_argument(
        "--force", action="store_true",
        help="transfer again even where an output exists without a matching provenance record",
    )

    for p, runner in (
        (p_stats, run_stats),
        (p_transform, run_transform),
        (p_walk, run_walk),
        (p_transfer, run_transfer),
    ):
        p.add_argument("--limit", type=int, default=0, help="stop after N items")
        p.add_argument(
            "--check", action="store_true",
            help="report the work list and stop; runs nothing",
        )
        p.set_defaults(func=runner)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
