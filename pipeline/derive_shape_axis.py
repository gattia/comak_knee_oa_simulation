"""Figure 3: the healthy-to-mild-OA shape axis and the synthetic knees placed on it.

Every cohort knee is described by one 2069-value joint descriptor: the three
neural-shape-model latents (femur with menisci 1024, tibia 512, patella 512)
and the pose of each bone as a deviation from the population mean (Euler
angles in degrees, translation in mm and a scale ratio, 7 values per bone;
femur absolute, tibia and patella relative to the femur).

The axis is the unit vector from the mean healthy (KL 0-1) descriptor to the
mean mild OA (KL 2) descriptor, in raw descriptor units -- nothing is
standardised, so latent units, degrees and millimetres mix as they are.
Every knee is projected onto it; the population mean and standard
deviation of that projection (numpy default, ddof = 0) set the scale. Synthetic
knees are then placed along the axis:

* sigma steps  -- population mean descriptor + k * sd * axis, k = -3 ... +3;
* group anchors -- population mean descriptor + (group mean score - mu) * axis,
  one per disease group.

Each synthetic descriptor is split back into three latents and three 4 x 4
transforms, which is what ``build_synthetic_knee.py`` decodes.

This stage is CPU-only and takes seconds. It reads the cohort's fitted latents
and alignments from ``01_models`` and never writes there; it writes
``shape_axis.json`` and each synthetic knee's descriptor arrays into the
synthetic tree.

Running it
----------
    python pipeline/run_synthetic.py axis --config config/synthetic_shape_axis.json
    python pipeline/run_synthetic.py axis --config ... --verify

``--verify`` re-derives the axis from the cohort's fits and checks it against
the synthetic tree on disk, writing nothing: the projection mean and standard
deviation, every shape's projected score, and every descriptor array the knees
were built from must agree to floating-point precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comak_knee_oa.config import Config, ConfigError, Knee  # noqa: E402

__all__ = [
    "collect_descriptors",
    "compute_axis",
    "synthetic_shapes",
    "descriptor_to_inputs",
    "derive_shape_axis",
    "INPUT_FILES",
]

BONES = ("femur", "tibia", "patella")
LATENT_DIMS = {"femur": 1024, "tibia": 512, "patella": 512}

# Where each part of the 2069-value descriptor lives.
COL_MAP = {
    "femur_latent": [0, 1024],
    "tibia_latent": [1024, 1536],
    "patella_latent": [1536, 2048],
    "femur_euler_deg": [2048, 2051],
    "femur_translation_mm": [2051, 2054],
    "femur_scale_ratio": [2054, 2055],
    "trel_tib_euler_deg": [2055, 2058],
    "trel_tib_translation_mm": [2058, 2061],
    "trel_tib_scale_ratio": [2061, 2062],
    "trel_pat_euler_deg": [2062, 2065],
    "trel_pat_translation_mm": [2065, 2068],
    "trel_pat_scale_ratio": [2068, 2069],
}

# The files one synthetic knee is decoded from.
INPUT_FILES = (
    "femur_latent.npy",
    "tibia_latent.npy",
    "patella_latent.npy",
    "T_fem.npy",
    "T_rel_tib.npy",
    "T_rel_pat.npy",
)

ANCHOR_NAMES = {
    "kl0_1_healthy": "healthy",
    "kl0_1_progressor": "progressor",
    "kl2": "kl2",
}

# Descriptor arrays reproduce to rounding; anything larger means the inputs,
# the cohort or the method changed.
VERIFY_ATOL = 1e-12


# ------------------------------------------------------------ descriptors ----


def _knee_dir(models_dir: Path, knee_name: str) -> Path:
    return models_dir / Knee.parse(knee_name).directory / "geometries_nsm_similarity"


def collect_descriptors(models_dir: Path, groups: dict[str, list[str]], allow_missing: bool = False):
    """The (N, 2069) descriptor matrix for every cohort knee.

    Returns ``(descriptors, knees, knee_groups, means, skipped)``. Knees are in
    sorted order; ``means`` holds the population pose means needed to turn a
    descriptor back into transforms. The relative transforms are computed in
    memory only.

    A knee without a fit for every bone raises: the axis, its scale and every
    synthetic knee depend on the whole cohort, so a partial ``01_models`` would
    otherwise move all of them unreported. ``allow_missing`` drops such
    knees instead, and the caller records which.
    """
    from nsosim.transforms import (
        compute_T_rel,
        compute_transform_deviations,
        decompose_similarity,
        deviations_to_transform,
    )

    group_of = {knee: group for group, knees in groups.items() for knee in knees}
    knees = []
    for knee in sorted(group_of):
        folder = _knee_dir(models_dir, knee)
        if all(
            (folder / bone / f"{bone}_alignment.json").is_file()
            and (folder / bone / f"{bone}_latent.npy").is_file()
            for bone in BONES
        ):
            knees.append(knee)
    skipped = sorted(set(group_of) - set(knees))
    if skipped and not allow_missing:
        raise FileNotFoundError(
            f"{len(skipped)} cohort knee(s) lack a shape-model fit for every bone in "
            f"{models_dir}: {skipped[:10]}. The axis would be computed without them; "
            "pass --allow-missing to do that deliberately."
        )
    if skipped:
        print(f"  {len(skipped)} knee(s) lack a fit for every bone and are left out: {skipped[:5]}")

    n = len(knees)
    t_fem = np.empty((n, 4, 4))
    t_rel = {"tibia": np.empty((n, 4, 4)), "patella": np.empty((n, 4, 4))}
    latents = {bone: np.empty((n, LATENT_DIMS[bone])) for bone in BONES}
    for i, knee in enumerate(knees):
        folder = _knee_dir(models_dir, knee)
        transforms = {
            bone: np.array(
                json.loads((folder / bone / f"{bone}_alignment.json").read_text())[
                    "linear_transform"
                ]
            )
            for bone in BONES
        }
        t_fem[i] = transforms["femur"]
        for bone in ("tibia", "patella"):
            t_rel[bone][i] = compute_T_rel(transforms["femur"], transforms[bone])
        for bone in BONES:
            latents[bone][i] = np.load(folder / bone / f"{bone}_latent.npy").squeeze()

    mean_fem_scale = float(np.mean([decompose_similarity(t)[0] for t in t_fem]))
    deviations = {
        "femur": compute_transform_deviations(t_fem, mean_fem_scale),
        "trel_tib": compute_transform_deviations(t_rel["tibia"], mean_fem_scale),
        "trel_pat": compute_transform_deviations(t_rel["patella"], mean_fem_scale),
    }

    # Every transform must be recoverable from its deviations; this is the
    # decomposition the synthetic knees rely on, checked on the whole cohort.
    worst = 0.0
    for key, originals in (("femur", t_fem), ("trel_tib", t_rel["tibia"]),
                           ("trel_pat", t_rel["patella"])):
        dev = deviations[key]
        for i in range(n):
            rebuilt = deviations_to_transform(
                dev["euler_angles_deg"][i], dev["translations_mm"][i], dev["scale_ratios"][i],
                dev["R_mean"], dev["t_mean"], dev["s_mean"], mean_fem_scale,
            )
            worst = max(worst, float(np.abs(rebuilt - originals[i]).max()))
    if worst > 1e-8:
        raise RuntimeError(f"pose deviations do not round-trip: worst error {worst:.2e}")

    columns = [latents[bone] for bone in BONES]
    for key in ("femur", "trel_tib", "trel_pat"):
        dev = deviations[key]
        columns += [dev["euler_angles_deg"], dev["translations_mm"], dev["scale_ratios"][:, None]]
    descriptors = np.column_stack(columns)
    if descriptors.shape != (n, 2069):
        raise RuntimeError(f"expected ({n}, 2069) descriptors, got {descriptors.shape}")

    means = {"mean_fem_scale": mean_fem_scale, "n_knees": n, "knees_missing_fits": skipped,
             "col_map": COL_MAP}
    for key in ("femur", "trel_tib", "trel_pat"):
        dev = deviations[key]
        means[key] = {
            "R_mean": np.asarray(dev["R_mean"]).tolist(),
            "t_mean": np.asarray(dev["t_mean"]).tolist(),
            "s_mean": float(dev["s_mean"]),
        }
    return descriptors, knees, np.array([group_of[k] for k in knees]), means, skipped


# ------------------------------------------------------------------- axis ----


def compute_axis(descriptors: np.ndarray, knee_groups: np.ndarray,
                 from_group: str, to_group: str) -> dict:
    """The unit axis from one group's mean descriptor to another's, and the projection."""
    for group in (from_group, to_group):
        if not (knee_groups == group).any():
            raise ConfigError(f"no knees in group {group!r}")
    desc_mean = descriptors.mean(axis=0)
    raw_diff = descriptors[knee_groups == to_group].mean(0) - descriptors[knee_groups == from_group].mean(0)
    diff_norm = float(np.linalg.norm(raw_diff))
    direction = raw_diff / (diff_norm + 1e-12)
    projection = descriptors @ direction
    return {
        "direction": direction,
        "desc_mean": desc_mean,
        "diff_norm": diff_norm,
        "projection": projection,
        "mu": float(projection.mean()),
        "sigma": float(projection.std()),
    }


def _sigma_token(k: float) -> str:
    """-1 -> 'm1', 0 -> 'p0', 1.5 -> 'p1p5'."""
    return ("m" if k < 0 else "p") + f"{abs(k):g}".replace(".", "p")


def synthetic_shapes(axis: dict, knee_groups: np.ndarray, sigmas, anchor_groups):
    """The synthetic descriptors: sigma steps first, then one anchor per group."""
    shapes = []
    for k in sorted(sigmas):
        shapes.append((
            f"fig5_sigma_{_sigma_token(k)}",
            axis["desc_mean"] + (k * axis["sigma"]) * axis["direction"],
            {"type": "sigma_step", "k_sigma": float(k),
             "projected_score_raw": axis["mu"] + k * axis["sigma"]},
        ))
    for group in anchor_groups:
        score = float(axis["projection"][knee_groups == group].mean())
        shapes.append((
            f"fig5_anchor_{ANCHOR_NAMES[group]}",
            axis["desc_mean"] + (score - axis["mu"]) * axis["direction"],
            {"type": "group_anchor", "group": group, "projected_score_raw": score,
             "k_sigma_equiv": (score - axis["mu"]) / axis["sigma"]},
        ))
    for _, descriptor, meta in shapes:
        meta["projected_score_check"] = float(descriptor @ axis["direction"])
    return shapes


def descriptor_to_inputs(descriptor: np.ndarray, means: dict) -> dict[str, np.ndarray]:
    """Split one descriptor into the six arrays a synthetic knee is decoded from."""
    from nsosim.transforms import deviations_to_transform

    cols = means["col_map"]

    def part(key):
        start, stop = cols[key]
        return descriptor[start:stop]

    def transform(prefix, pose_key):
        pose = means[pose_key]
        return deviations_to_transform(
            part(f"{prefix}_euler_deg"), part(f"{prefix}_translation_mm"),
            part(f"{prefix}_scale_ratio")[0],
            np.array(pose["R_mean"]), np.array(pose["t_mean"]), pose["s_mean"],
            means["mean_fem_scale"],
        )

    return {
        "femur_latent.npy": np.asarray(part("femur_latent"), dtype=np.float64),
        "tibia_latent.npy": np.asarray(part("tibia_latent"), dtype=np.float64),
        "patella_latent.npy": np.asarray(part("patella_latent"), dtype=np.float64),
        "T_fem.npy": transform("femur", "femur"),
        "T_rel_tib.npy": transform("trel_tib", "trel_tib"),
        "T_rel_pat.npy": transform("trel_pat", "trel_pat"),
    }


# ------------------------------------------------------------------ stage ----


def derive_shape_axis(config: Config, tree: Path, verify: bool = False,
                      allow_missing: bool = False, force: bool = False) -> dict:
    """Write the shape axis and every synthetic knee's inputs under ``tree``.

    ``tree`` is the synthetic tree's root (``.../synthetic_shape_axis``).
    Returns the ``shape_axis.json`` content. Existing descriptor arrays that
    differ from the new ones raise unless ``force`` is set: models already
    built from them would otherwise be paired with the new axis unnoticed.
    """
    section = config.section("axis")
    models_dir = config.path("axis", "descriptor_models")
    groups = json.loads(config.path("axis", "groups").read_text())
    from_group = section.get("from_group", "kl0_1_healthy")
    to_group = section.get("to_group", "kl2")
    anchors = section.get("anchor_groups", ["kl0_1_healthy", "kl0_1_progressor", "kl2"])
    sigmas = section.get("sigmas", [-3, -2, -1, 0, 1, 2, 3])

    print(f"collecting descriptors from {models_dir}")
    descriptors, knees, knee_groups, means, skipped = collect_descriptors(
        models_dir, groups, allow_missing=allow_missing
    )
    print(f"  {descriptors.shape[0]} knees x {descriptors.shape[1]} values")
    axis = compute_axis(descriptors, knee_groups, from_group, to_group)
    print(f"  axis {from_group} -> {to_group}: projection mu {axis['mu']:.6f}, "
          f"sd {axis['sigma']:.6f}")
    shapes = synthetic_shapes(axis, knee_groups, sigmas, anchors)
    inputs = {label: descriptor_to_inputs(descriptor, means) for label, descriptor, _ in shapes}
    summary = {
        "model_version": config.get("build", "model_version"),
        "axis": f"unit(mean({to_group}) - mean({from_group}))",
        "proj_mu": axis["mu"],
        "proj_sigma": axis["sigma"],
        "n_knees": len(knees),
        "knees_missing_fits": skipped,
        "n_shapes": len(shapes),
        "shapes": {label: meta for label, _, meta in shapes},
    }
    for label, _, meta in shapes:
        print(f"  {label:24s} score {meta['projected_score_raw']:+.6f}")

    if verify:
        # A check, not a build: the tree on disk is what every knee was built from.
        _verify(tree, summary, inputs)
        return summary

    tree.mkdir(parents=True, exist_ok=True)
    record_dir = tree / "_axis"
    record_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        record_dir / "joint_descriptors.npz",
        descriptors=descriptors, subjects=np.array(knees), groups=knee_groups,
    )
    (record_dir / "joint_transform_means.json").write_text(json.dumps(means, indent=2) + "\n")
    np.savez(
        record_dir / "axis.npz",
        direction=axis["direction"], desc_mean=axis["desc_mean"],
        diff_norm=np.float64(axis["diff_norm"]), proj=axis["projection"],
        proj_mu=np.float64(axis["mu"]), proj_sigma=np.float64(axis["sigma"]),
        subjects=np.array(knees), groups=knee_groups,
        sigma_steps=np.array(sorted(sigmas), dtype=np.float64),
    )

    changed = []
    for label, arrays in inputs.items():
        for filename, array in arrays.items():
            existing = tree / label / "input_descriptors" / filename
            if existing.is_file():
                old = np.load(existing)
                if old.shape != array.shape or np.abs(old - array).max() > VERIFY_ATOL:
                    changed.append(f"{label}/{filename}")
    if changed and not force:
        raise RuntimeError(
            f"{len(changed)} existing descriptor array(s) under {tree} differ from the new "
            f"axis (e.g. {changed[:3]}); knees built from them would no longer match "
            "shape_axis.json. Use a new run.root, or pass --force to overwrite."
        )
    for label, arrays in inputs.items():
        out = tree / label / "input_descriptors"
        out.mkdir(parents=True, exist_ok=True)
        for filename, array in arrays.items():
            np.save(out / filename, array)

    (tree / "shape_axis.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _verify(tree: Path, summary: dict, inputs: dict) -> None:
    """Compare a freshly derived axis with the tree on disk; raise on any disagreement."""
    published = tree
    if not (published / "shape_axis.json").is_file():
        raise FileNotFoundError(f"--verify needs {published / 'shape_axis.json'}; "
                                "run the axis step without --verify first")
    reference = json.loads((published / "shape_axis.json").read_text())
    failures = []

    def close(a, b, what, tol=1e-12):
        if not np.isclose(a, b, rtol=tol, atol=tol):
            failures.append(f"{what}: {a!r} vs published {b!r}")

    close(summary["proj_mu"], reference["proj_mu"], "projection mean")
    close(summary["proj_sigma"], reference["proj_sigma"], "projection sd")
    if set(summary["shapes"]) != set(reference["shapes"]):
        failures.append(
            f"shape labels differ: {sorted(summary['shapes'])} vs {sorted(reference['shapes'])}"
        )
    worst = 0.0
    for label, meta in summary["shapes"].items():
        ref = reference["shapes"].get(label)
        if ref is None:
            continue
        close(meta["projected_score_raw"], ref["projected_score_raw"], f"{label} score", 1e-9)
        for filename in INPUT_FILES:
            ours = inputs[label][filename]
            theirs = np.load(published / label / "input_descriptors" / filename)
            if ours.shape != theirs.shape:
                failures.append(f"{label}/{filename}: shape {ours.shape} vs {theirs.shape}")
                continue
            delta = float(np.abs(ours - theirs).max())
            worst = max(worst, delta)
            if delta > VERIFY_ATOL:
                failures.append(f"{label}/{filename}: max |difference| {delta:.3e}")
    if failures:
        raise RuntimeError("shape axis does not match the published tree:\n  "
                           + "\n  ".join(failures))
    print(f"VERIFIED against {published}: mu, sd, {len(summary['shapes'])} shape scores, "
          f"{len(summary['shapes']) * len(INPUT_FILES)} descriptor arrays "
          f"(worst |difference| {worst:.1e})")
