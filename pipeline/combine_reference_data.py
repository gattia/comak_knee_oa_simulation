"""Stage 03, assembly: anatomical regions and the combined per-vertex matrices.

Two subcommands finish the derived stage:

``labels``
    One anatomical region label per reference vertex, shared by every
    analysis. Each knee's cartilage subregions were labelled on its own
    segmentation; those labels already live in the thickness models' space,
    so they transfer to the reference surface by categorical nearest
    neighbours, and a per-vertex majority vote across the cohort gives one
    gait-invariant label map. Femur regions collapse to medial, lateral, and
    trochlea; tibia to medial and lateral. (CPU)

``matrices``
    Stack every knee's pressure and thinning fields into one compressed
    archive per gait condition -- the file the statistics and figures read.
    Rows are knees, columns are the 20,000 reference vertices. A knee enters
    only with all four of its fields present (pressure and thinning, both
    bones); validity masks ride along so downstream code can tell "no data"
    from a measured zero. Shape -- the cached walks -- is gait-invariant, so it
    is written once, beside the reference condition's matrix and over the same
    knees, as ``combined_data_shape.npz``. (CPU)

Outputs, under ``03_derived``: ``region_labels/<bone>_region_labels.{npz,vtk}``
and ``combined/combined_data_<gait>.npz`` (plus the shape archive). Inputs: the
pressure on the reference (``pressure_on_reference_<gait>``), the thinning rates
(``thickness_merged``) and the cached walks, all from ``derive_pressure.py`` and
``derive_thickness.py``.

Running it
----------
    python pipeline/combine_reference_data.py labels
    python pipeline/combine_reference_data.py matrices --config config/measured_gait/Subject_102.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import Knee, load_config  # noqa: E402
from comak_knee_oa.paths import data_root, stage_dir  # noqa: E402
from derive_pressure import condition_name, pressure_cache_dir, walked_out_path  # noqa: E402
from derive_thickness import (  # noqa: E402
    BONES,
    cohort,
    reference_surface_path,
    thinning_rate_of_record_path,
    thinning_rates_of_record_dir,
)

# Cartilage subregion labels as drawn on the segmentations.
FINE_LABEL_NAMES = {
    8: "med_tibia",
    9: "lat_tibia",
    10: "trochlea",
    11: "med_wb_femur",
    12: "lat_wb_femur",
    13: "med_post_femur",
    14: "lat_post_femur",
}
BONE_FINE_LABELS = {"femur": [10, 11, 12, 13, 14], "tibia": [8, 9]}

# Coarse regions: medial=0, lateral=1, trochlea=2 (femur only), -1 = unlabelled.
COLLAPSE = {
    "femur": {10: 2, 11: 0, 13: 0, 12: 1, 14: 1},
    "tibia": {8: 0, 9: 1},
}

# Label transfer settings: five categorical neighbours, capped at 0.08 in
# model space -- cartilage sits about 0.02-0.06 from its bone surface there.
LABEL_NEIGHBOURS = 5
LABEL_MAX_DIST = 0.08
MIN_LABELLED_POINTS = 50


def labelled_mesh_path(knee: str, bone: str) -> Path:
    return (
        stage_dir("inputs", "latents_thickness", "V00", "STANFORD", f"{bone}_meshes_nsm_space")
        / f"RECON_{knee}_{bone}_mesh_nsm_space.vtk"
    )


def region_labels_path(bone: str) -> Path:
    return stage_dir("derived", "region_labels") / f"{bone}_region_labels.npz"


def groups() -> dict[str, str]:
    """Knee -> disease group, from the vendored assignment."""
    table = json.loads((stage_dir("inputs") / "groups.json").read_text())
    return {knee: group for group, knees in table.items() for knee in knees}


# ----------------------------------------------------------------- labels ----


def _transfer_labels(reference, knee: str, bone: str) -> np.ndarray | None:
    """One knee's subregion labels onto the reference vertices, or None.

    A knee abstains when its mesh has no labels or too few labelled points to
    trust the transfer.
    """
    import pyvista as pv
    from pymskt.mesh.meshTools import (
        transfer_mesh_scalars_get_weighted_average_n_closest as transfer_scalars,
    )

    path = labelled_mesh_path(knee, bone)
    if not path.is_file():
        return None
    mesh = pv.read(path)
    if "labels" not in mesh.point_data:
        return None
    labelled = mesh.extract_points(
        np.asarray(mesh.point_data["labels"]) != 0, adjacent_cells=False
    )
    if labelled.n_points < MIN_LABELLED_POINTS:
        return None
    source = pv.PolyData(labelled.points)
    source.point_data["labels"] = np.asarray(labelled.point_data["labels"])
    result = transfer_scalars(
        reference,
        source,
        n=LABEL_NEIGHBOURS,
        categorical={"labels": True},
        max_dist=LABEL_MAX_DIST,
    )
    return np.asarray(result["labels"]).ravel()


def run_labels(args) -> int:
    import pyvista as pv

    bones = BONES if args.bone == "both" else (args.bone,)
    knees = cohort()
    output_dir = stage_dir("derived", "region_labels", create=not args.check)
    if args.check:
        resolvable = sum(
            labelled_mesh_path(knee, bone).is_file() for knee in knees for bone in bones
        )
        print(f"{resolvable}/{len(knees) * len(bones)} labelled meshes resolve")
        return 0

    for bone in bones:
        reference = pv.read(reference_surface_path(bone))
        votes, contributors = [], []
        for i, knee in enumerate(knees, 1):
            labels = _transfer_labels(reference, knee, bone)
            if labels is not None:
                votes.append(labels.astype(np.int16))
                contributors.append(knee)
            if i % 100 == 0:
                print(f"  {bone}: {i}/{len(knees)}", flush=True)
        if not votes:
            raise SystemExit(f"{bone}: no knee contributed labels -- inputs not resolving")

        stacked = np.stack(votes, axis=0)  # (n_knees, 20000)
        n_vertices = stacked.shape[1]
        fine = np.zeros(n_vertices, dtype=np.int64)
        agreement = np.zeros(n_vertices)
        n_votes = np.zeros(n_vertices, dtype=np.int64)
        allowed = set(BONE_FINE_LABELS[bone])
        for v in range(n_vertices):
            column = stacked[:, v]
            column = column[(column != 0) & np.isin(column, list(allowed))]
            n_votes[v] = len(column)
            if len(column) == 0:
                continue
            values, counts = np.unique(column, return_counts=True)
            winner = int(np.argmax(counts))
            fine[v] = int(values[winner])
            agreement[v] = counts[winner] / len(column)

        coarse = np.array(
            [COLLAPSE[bone].get(int(label), -1) for label in fine], dtype=np.int64
        )
        print(
            f"{bone}: {len(contributors)} knees voted; "
            f"mean agreement {agreement[n_votes > 0].mean():.3f}; "
            f"{(coarse < 0).sum()} unlabelled vertices"
        )

        np.savez(
            region_labels_path(bone),
            fine_labels=fine,
            coarse_code=coarse,
            agreement=agreement,
            n_votes=n_votes,
            ref_points=np.asarray(reference.points, dtype=np.float32),
            subjects=np.array(contributors),
            bone=bone,
            n_closest=LABEL_NEIGHBOURS,
            max_dist=LABEL_MAX_DIST,
            fine_label_names=json.dumps(FINE_LABEL_NAMES),
            collapse_name=json.dumps({"medial": 0, "lateral": 1, "trochlea": 2}),
        )
        labelled_reference = reference.copy()
        labelled_reference.point_data["fine_label"] = fine
        labelled_reference.point_data["region_code"] = coarse
        labelled_reference.point_data["agreement"] = agreement
        labelled_reference.point_data["n_votes"] = n_votes
        labelled_reference.save(str(output_dir / f"{bone}_region_labels.vtk"))
    return 0


# --------------------------------------------------------------- matrices ----

PRESSURE_FIELDS = {
    "pressure_mean": "mean_tf_pressure",
    "pressure_max": "max_tf_pressure",
    "pressure_cum": "cumulative_tf_pressure",
}
THINNING_FIELDS = {
    "thinning": "thinning_rate (mm/year)",
    "thinning_pct": "thinning_rate (pct/year)",
    "baseline_thickness": "baseline_thickness (mm)",
}


def _relative(path: Path) -> str:
    """A data path as recorded inside an archive: relative to the data root."""
    return str(path.relative_to(data_root()))


def run_matrices(args) -> int:
    from pymskt.mesh import Mesh

    config = load_config(args.config)
    condition = condition_name(config)
    pressure_dir = stage_dir("derived", f"pressure_on_reference_{condition}")
    thinning_dir = thinning_rates_of_record_dir()
    output_dir = stage_dir("derived", "combined", create=not args.check)
    group_of = groups()
    # Shape is gait-invariant: one archive, built with the reference condition
    # over the same knees as its matrix.
    with_shape = condition == cohorts.REFERENCE

    def pressure_path(knee: Knee, bone: str) -> Path:
        return pressure_dir / bone / f"{knee.subject}_{knee.side}_{bone}_pressure_ref.vtk"

    # A knee enters only complete: pressure and thinning for both bones.
    admitted, absent = [], 0
    for name in cohort():
        knee = Knee.parse(name)
        needed = [pressure_path(knee, bone) for bone in BONES]
        needed += [thinning_rate_of_record_path(name, bone) for bone in BONES]
        if all(path.is_file() for path in needed):
            admitted.append(knee)
        else:
            absent += 1
    print(f"{condition}: {len(admitted)} knees complete, {absent} missing a field")
    if args.check or not admitted:
        return 0 if admitted or args.check else 1

    # The shape fields come from the cached walks and must be complete too --
    # a zero-filled row is indistinguishable from real geometry at the origin.
    missing_walks = [
        f"{knee.name} {bone}"
        for knee in (admitted if with_shape else [])
        for bone in BONES
        if not walked_out_path(knee, bone).is_file()
    ]
    if missing_walks:
        raise FileNotFoundError(
            f"{len(missing_walks)} cached walk(s) missing (first: {missing_walks[0]}); "
            "run 'derive_pressure.py walk' first"
        )

    n_vertices = 20_000
    names = [knee.name for knee in admitted]
    data: dict[str, np.ndarray] = {
        "subjects": np.array(names),
        "groups": np.array([group_of.get(name, "") for name in names]),
        "bones": np.array(list(BONES)),
        "gait": np.array(condition),
        # The figures check this before treating region_mask as anatomical
        # compartments rather than the solver's axis-sign split.
        "region_convention": np.array("anatomical"),
        "pressure_dir": np.array(_relative(pressure_dir)),
        "thinning_dir": np.array(_relative(thinning_dir)),
    }
    shape: dict[str, np.ndarray] = {
        "subjects": np.array(names),
        "bones": np.array(list(BONES)),
        "walked_out_dir": np.array(_relative(pressure_cache_dir())),
    }

    for bone in BONES:
        labels_file = region_labels_path(bone)
        if not labels_file.is_file():
            raise FileNotFoundError(
                f"{labels_file} not found; run 'combine_reference_data.py labels' first"
            )
        labels = np.load(labels_file, allow_pickle=True)
        data[f"{bone}_region_mask"] = labels["coarse_code"].astype(np.int8)
        data[f"{bone}_fine_label"] = labels["fine_labels"].astype(np.int32)
        data[f"{bone}_n_verts"] = np.array(n_vertices)
        shape[f"{bone}_n_verts"] = np.array(n_vertices)

        arrays = {
            key: np.zeros((len(admitted), n_vertices), dtype=np.float32)
            for key in list(PRESSURE_FIELDS) + list(THINNING_FIELDS)
        }
        has_pressure = np.zeros((len(admitted), n_vertices), dtype=bool)
        has_thinning = np.zeros((len(admitted), n_vertices), dtype=bool)
        if with_shape:
            xyz = np.zeros((len(admitted), n_vertices, 3), dtype=np.float32)
            displacement = np.zeros((len(admitted), n_vertices, 3), dtype=np.float32)

        for row, knee in enumerate(admitted):
            pressure = Mesh(str(pressure_path(knee, bone)))
            valid = np.asarray(pressure.get_scalar("has_data")) > 0.5
            has_pressure[row] = valid
            for key, scalar in PRESSURE_FIELDS.items():
                values = np.asarray(pressure.get_scalar(scalar), dtype=np.float32)
                values[~valid | (values < 0)] = 0.0
                arrays[key][row] = values

            thinning = Mesh(str(thinning_rate_of_record_path(knee.name, bone)))
            valid = np.asarray(thinning.get_scalar("has_data")) > 0.5
            has_thinning[row] = valid
            for key, scalar in THINNING_FIELDS.items():
                values = np.asarray(thinning.get_scalar(scalar), dtype=np.float32)
                values[~valid] = 0.0
                arrays[key][row] = values

            if with_shape:
                xyz[row] = np.load(walked_out_path(knee, bone))
                displacement[row] = np.load(
                    walked_out_path(knee, bone).with_name(
                        f"{knee.subject}_{knee.side}_{bone}_displacement.npy"
                    )
                )
            if (row + 1) % 100 == 0:
                print(f"  {bone}: {row + 1}/{len(admitted)}", flush=True)

        for key, matrix in arrays.items():
            data[f"{bone}_{key}"] = matrix
        data[f"{bone}_has_pressure"] = has_pressure
        data[f"{bone}_has_thinning"] = has_thinning
        if with_shape:
            shape[f"{bone}_shape_xyz"] = xyz
            shape[f"{bone}_shape_displacement"] = displacement
            # Every row has its walk (checked above); kept so a reader need not
            # assume it.
            shape[f"{bone}_has_shape"] = np.ones(len(admitted), dtype=bool)

    combined = output_dir / f"combined_data_{condition}.npz"
    np.savez_compressed(combined, **data)
    print(f"wrote {combined} ({len(admitted)} knees)")
    if with_shape:
        np.savez_compressed(output_dir / "combined_data_shape.npz", **shape)
        print(f"wrote {output_dir / 'combined_data_shape.npz'} ({len(admitted)} knees)")
    return 0


# -------------------------------------------------------------------- cli ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="combine_reference_data.py",
        description="Region labels and the combined per-vertex matrices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_labels = sub.add_parser("labels", help="anatomical region labels on the reference")
    p_labels.add_argument("--bone", choices=["femur", "tibia", "both"], default="both")
    p_labels.set_defaults(func=run_labels)

    p_matrices = sub.add_parser("matrices", help="one combined archive per gait condition")
    p_matrices.add_argument(
        "--config", required=True,
        help="the gait condition's run config; names which pressure to stack",
    )
    p_matrices.set_defaults(func=run_matrices)

    for p in (p_labels, p_matrices):
        p.add_argument(
            "--check", action="store_true",
            help="report the work list and stop; runs nothing",
        )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
