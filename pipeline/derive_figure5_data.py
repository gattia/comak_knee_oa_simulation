#!/usr/bin/env python3
"""Derive Figure 5's and Extended Data Figure 2's inputs: 03_derived ->
04_results/figure5_spatial_colocalization.

Figure 5 (cumulative pressure) and Extended Data Figure 2 (peak pressure) ask
where on the joint surface contact pressure and subsequent cartilage thickness
change coincide. It is a surface figure: every panel is the
same 20,000-vertex reference cartilage surface, colored by a different quantity,
under two walking conditions -- the reference gait and one measured
osteoarthritis gait -- so that a spatial claim can be seen to hold under both.

This step does the expensive work; the figure script only composes images.

Two stages
----------
``spm``
    Permutation inference over the surface: 40 independent cells, one per
    invocation, so they can run as a job array (about five hours as a serial
    loop). 32 correlation cells for the manuscript panel (2 conditions x 4 subsets x
    2 bones x 2 pressure types) and 8 for the thickness-change group
    comparison (2 omnibus + 6 pairwise). Writes cluster maps and per-cell summaries under
    ``04_results/figure5_spatial_colocalization/spm/``.
``maps``
    Everything else: quality control, per-group vertex statistics, the surface
    renders, and the cluster tables the manuscript quotes. Reads the ``spm``
    stage's output; does not run it.

    python pipeline/derive_figure5_data.py spm --cell $SLURM_ARRAY_TASK_ID
    python pipeline/derive_figure5_data.py maps

Keeping the stages separate also means re-rendering the figure cannot re-roll
the statistics underneath it.

The permutation null is seeded, so these p-values reproduce. Each cell runs at
``SPM_SEED + its index``; the seed is written into every cell summary and into
``spm/_seeds.json``. Without a seed, pyKneeSPM draws from an unseeded global RNG
and a cell run twice on identical data gives different p-values (0.0059 and
0.0048 in one measured case).

A seed fixes which null is drawn, but the null is still random, so a cluster
near the threshold can flip under a different seed. With 10,000 permutations
the p-value resolution is 1e-4. When no cluster survives, the tables write
``n.s.``; a literal ``0`` would read as the strongest result.

Missing values in the correlation cells: ``--missing``
-------------------------------------------------------
A knee can lack a value at a vertex -- almost always no thickness change there,
mostly near the cartilage rim. The reported analysis (``--missing zero``, the
default) writes both of that knee's values as 0 and keeps it in the vertex's
correlation: a zero-filled knee sits at the origin and pulls r toward whatever
the origin implies. Measured, it is small -- every knee has some such vertices
(median 2 % femur / 4 % tibia of included vertices), but under 0.1 % of knees
inside the significant clusters -- and the default stays the zero fill, because
every table and panel of the figure is computed under it.

``--missing complete-case`` is the opt-in sensitivity run: a missing value is
stored as NaN, and each vertex's Pearson r, its n and the t's degrees of
freedom come from only the knees with both values there. Everything else is
held fixed so that the missing-value rule is the only thing that moves:

* the same included vertices (at least ``VERTEX_COVERAGE`` of knees with a
  finite, non-zero thickness change -- the same set the zero fill admits);
* the same permutation matrix -- same seed, same generator, pressure shuffled
  across knees with its NaN pattern travelling with its row;
* the same clustering, cluster-extent threshold and cluster p-values.

It writes only to ``04_results/figure5_spatial_colocalization/spm_complete_case/``,
never to ``spm/``, and supports only the 32 correlation cells (the
thickness-change group tests have no pairing to break). Nothing reads that tree: the figure, the
cluster tables and the manuscript all read ``spm/``.

    python pipeline/derive_figure5_data.py spm --cell 7 --missing complete-case

pyKneeSPM is a pinned external library, so the complete-case statistic is
implemented here as subclasses of its correlation classes.

What it produces
----------------
``panels/{pressure}/{bone}/*.png``
    The surface renders, already cropped to a shared bounding box and flipped
    into the figure's anatomical convention. The figure script composes these;
    it does not do 3D, so rendering the manuscript needs five Python packages
    and no VTK stack (Figure 3's decoded anatomy works the same way).
``panels/supplement/*.png``
    The thickness-change group comparison's omnibus and pairwise surfaces. These
    are a data output; no figure script composes them.
``layout.json``
    Color limits, condition labels, panel filenames and the section structure.
    The figure reads its analysis choices from here, so the two cannot
    disagree.
``population.json``
    Per condition x bone: matrix n, how many knees quality control dropped and
    why, the surviving n, and its split by disease group.
``spm_clusters.csv`` / ``supplement_clusters.csv``
    One row per SPM cell: cluster count, areas, p-values, and the n it ran at.
    The manuscript quotes its numbers from these tables.
``region_composition.csv``
    Which anatomical regions the thickness-change group comparison's significant
    clusters fall in, split by the sign of the group-mean difference, because a
    positive medial lobe and a negative trochlear one have no single average.

Color limits are pooled across both bones and both conditions
-------------------------------------------------------------
One colorbar per quantity for the whole figure, so a color means the same value
on the femur as on the tibia and under either gait. The comparison is spatial,
and per-panel scaling would make every panel look alike, so pooling is done
here, over every group and both bones, before anything is rendered.

Quality control
---------------
Exclusions come from the shared table (``figures/shared/qc.py``), union policy,
plus the chronic-knee list. No per-script pressure threshold is applied: an
absolute threshold is not scale-invariant, so a change in model scaling that
moves contact area by several percent would move which knees sit either side
of it for
reasons unrelated to being an artifact.
The shared table is also the union of a spatial and a time-series detector,
which disagree often enough (Jaccard about 0.5) that either alone misses cases.

The maps and the SPM use the same population, by construction: both filter
through the same accessor with the same policy.

Usage
-----
    python pipeline/derive_figure5_data.py maps
    python pipeline/derive_figure5_data.py maps --pressure cum --bone femur
    python pipeline/derive_figure5_data.py spm --list-cells
    python pipeline/derive_figure5_data.py spm --cell 7
    python pipeline/derive_figure5_data.py spm --cell 7 --missing complete-case
"""

from __future__ import annotations

import argparse
import csv
import functools
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
# The quality-control table is shared with the figures, so that "which
# simulations exist" has one answer.
sys.path.insert(0, str(_REPO / "figures"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402
from derive_thickness import reference_surface_path  # noqa: E402
from shared import qc  # noqa: E402

FIGURE = "figure5_spatial_colocalization"

BONES = ("femur", "tibia")
PRESSURES = ("cum", "max")

# The knee-OA example gait, the second column: the ninth measured donor. The v3
# run uses Subject_121; the v2 run used the pilot recording RSubject_121. The
# data root's declared measured donors (cohorts.measured()) must contain one of
# the two, which selects it; both configs are kept so either data root works.
OA_EXAMPLE_CANDIDATES = ("Subject_121", "RSubject_121")


def _oa_example_gait() -> str:
    donors = cohorts.measured()
    present = [g for g in OA_EXAMPLE_CANDIDATES if g in donors]
    if len(present) != 1:
        raise RuntimeError(
            f"Figure 5's knee-OA example gait is whichever of {OA_EXAMPLE_CANDIDATES} "
            f"is among the data root's measured donors, and exactly one must be; the root "
            f"declares {list(donors)}. The v3 run uses Subject_121 in place of RSubject_121."
        )
    return present[0]


OA_EXAMPLE_GAIT = _oa_example_gait()

# The two walking conditions this figure contrasts. Both columns show the same
# OAI knees and differ only in gait donor, so the labels name the gait;
# "OA Subject" would read as a different population.
CONDITIONS = ("default", OA_EXAMPLE_GAIT)
CONDITION_LABEL = {"default": "Healthy Gait", OA_EXAMPLE_GAIT: "OA Gait"}

# Subsets as the SPM filenames spell them, and the group keys they map to.
SUBSETS = ("all", "healthy", "progressor", "kl2")
SUBSET_GROUP = {
    "all": "all",
    "healthy": "kl0_1_healthy",
    "progressor": "kl0_1_progressor",
    "kl2": "kl2",
}
GROUP_SUBSET = {v: k for k, v in SUBSET_GROUP.items()}
# Row order everywhere: the pooled row first, then the three disease groups.
GROUP_ROWS = ("all", "kl0_1_healthy", "kl0_1_progressor", "kl2")
DISEASE_GROUPS = ("kl0_1_healthy", "kl0_1_progressor", "kl2")

# Thickness-change group contrasts, in filename order. The stored difference is g1 - g2.
PAIRS = (("healthy", "progressor"), ("healthy", "kl2"), ("progressor", "kl2"))

# Group contrasts drawn as delta-pressure maps on the manuscript panel.
CONTRASTS = (
    ("progressor_vs_healthy", "kl0_1_progressor", "kl0_1_healthy"),
    ("kl2_vs_healthy", "kl2", "kl0_1_healthy"),
)

# Camera per bone. Femur is viewed from below through the articular surface,
# tibia from above; both with anterior up.
VIEW = {"femur": "inferior", "tibia": "superior"}

# A vertex is drawn only where at least this many knees contributed. Below it
# the group mean is one or two subjects and reads as signal.
MIN_KNEES_PER_VERTEX = 10

# Beta-slope color limits use the 20th/80th percentiles (not 2nd/98th) so the
# central slopes, which are the finding, keep their contrast; the edge hotspots
# may saturate.
BETA_CLIM_PCT = (20, 80)

# SPM parameters. The cluster-forming threshold is t at alpha = 0.01 for the
# correlation and pairwise tests; the omnibus derives its F threshold from its
# own degrees of freedom.
MC_ITERATIONS = 10000
CLUSTER_FORMING_T = 2.33
VERTEX_COVERAGE = 0.5
ALPHA_CLUSTER_FORMING = 0.01

# The permutation seed. Each cell runs at SPM_SEED + its index in the fixed
# cell order, so every cluster p-value in this figure reproduces.
#
# Each cell has its own seed. A cell's cluster-extent threshold is the 95th
# percentile of its own null draw, so a draw that happens to land low
# makes that cell slightly liberal. Sharing one draw across cells would make that
# error common to all of them and correlated in sign -- the forty cells would
# stop being forty independent tests. Distinct seeds keep the errors independent,
# which is how the figure's rows and columns are read. (Cells with different n
# draw different streams regardless; this matters for the same-n ones, which is
# most of them.)
#
# A seed fixes which null is drawn; each of the 10,000 draws still advances the
# generator. It makes results reproducible, not more stable: a cluster that
# survives under only one seed is not a finding. The base value is arbitrary.
SPM_SEED = 20260829

# How a knee with no value at a vertex enters that vertex's correlation. "zero"
# is the reported analysis; "complete-case" is the opt-in sensitivity run (see
# the module docstring) and writes to its own tree. The first entry is the
# default, and must stay "zero": every reported number is computed under it.
MISSING_POLICIES = ("zero", "complete-case")

# Panel render size, and the crop margin left around the surface silhouette.
WINDOW_MAIN = (600, 500)
WINDOW_SUPPLEMENT = (900, 900)
CROP_PADDING_PX = 8
# Pixels at or above this value in all three channels are background. Masked
# vertices render light grey (about 211) and must survive the crop, so the
# threshold sits above them.
#
# One threshold for the manuscript panels and the thickness-change group panels.
# A different value moves one row of antialiased silhouette edge inside or
# outside the crop; no number depends on it.
BACKGROUND_THRESHOLD = 245


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def matrix_path(condition: str) -> Path:
    """``03_derived/combined/combined_data_<condition>.npz``."""
    path = stage_dir("derived", "combined") / f"combined_data_{condition}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"No combined matrix for condition {condition!r}: {path}\n\n"
            "Build it with:\n"
            "    python pipeline/combine_reference_data.py\n\n"
            "Or download the published 03_derived stage (docs/data_provenance.md)."
        )
    return path


def load_group_map() -> dict[str, str]:
    """``{knee: group key}`` from the vendored group definition."""
    raw = json.loads((stage_dir("inputs") / "groups.json").read_text())
    return {knee: group for group, knees in raw.items() for knee in knees}


def region_labels(bone: str) -> np.ndarray:
    """Coarse anatomical region code per reference vertex."""
    path = stage_dir("derived", "region_labels") / f"{bone}_region_labels.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Region labels missing: {path}\n\n"
            "Build them with:\n"
            "    python pipeline/combine_reference_data.py\n"
        )
    return np.asarray(np.load(path, allow_pickle=True)["coarse_code"])


REGION_NAMES = {
    "femur": {0: "medial", 1: "lateral", 2: "trochlea"},
    "tibia": {0: "medial", 1: "lateral"},
}


def results_dir(create: bool = False) -> Path:
    return stage_dir("results", FIGURE, create=create)


def spm_dir(create: bool = False) -> Path:
    return stage_dir("results", FIGURE, "spm", create=create)


def spm_complete_case_dir(create: bool = False) -> Path:
    """Where ``--missing complete-case`` writes. Nothing downstream reads it."""
    return stage_dir("results", FIGURE, "spm_complete_case", create=create)


# --------------------------------------------------------------------------- #
# Quality control                                                             #
# --------------------------------------------------------------------------- #
def excluded_knees(condition: str) -> tuple[set[str], dict]:
    """Knees this condition drops, and the policy that produced them.

    ``load_outlier_set`` raises on a condition key the table has never seen,
    because an empty exclusion set is indistinguishable from a
    clean cohort, so a figure pointed at an unquality-controlled condition
    would apply no exclusions while recording that it had applied the union
    policy.
    """
    tag = cohorts.resolve_qc_tags(qc.known_cohort_tags(), [condition])[0]
    per_simulation = qc.load_outlier_set(cohort_tags=[tag], source="union")
    chronic = qc.load_chronic_knees()
    flagged = {knee for _, knee in per_simulation}
    policy = {
        "source": "union",
        "era": qc.ERA,
        "table": qc.unified_csv().name,
        "qc_tag": tag,
        "n_flagged_this_condition": len(flagged),
        "n_chronic": len(chronic),
    }
    return flagged | chronic, policy


def _drop_reason(knee, flagged, chronic) -> str:
    if knee in chronic:
        return "chronic knee (flagged under most conditions, or manually listed)"
    if knee in flagged:
        return "per-simulation artifact under this condition"
    return "excluded"


# --------------------------------------------------------------------------- #
# Per-group vertex statistics                                                 #
# --------------------------------------------------------------------------- #
def group_mean(values, has_data):
    """Vertex-wise mean over the knees that have data there, and their count.

    Dividing by the per-vertex count (not the group size) keeps the surface
    edges, where only part of the group has coverage, from reading as low
    pressure.
    """
    counts = has_data.sum(axis=0)
    total = np.where(has_data, values, 0.0).sum(axis=0)
    return total / counts.clip(min=1), counts


def vertex_slope(pressure, thinning, has_pressure, has_thinning):
    """Per-vertex OLS slope of thickness change on pressure, across knees.

    Returns ``(slope, n_knees)``. A vertex is fitted only where at least
    ``MIN_KNEES_PER_VERTEX`` knees have both measurements *and* non-zero
    pressure -- a zero is "no contact here", not "no load here", and including
    it would drag every slope toward the origin.
    """
    pressure = pressure.astype(np.float64)
    thinning = thinning.astype(np.float64)
    valid = has_pressure & has_thinning & (pressure > 0)
    counts = valid.sum(axis=0)

    slope = np.zeros(pressure.shape[1])
    for v in range(pressure.shape[1]):
        mask = valid[:, v]
        if mask.sum() < MIN_KNEES_PER_VERTEX:
            continue
        x = pressure[mask, v]
        if x.std() < 1e-8:
            continue
        covariance = np.cov(x, thinning[mask, v], ddof=0)
        slope[v] = covariance[0, 1] / covariance[0, 0]
    return slope, counts


def condition_statistics(condition, bone, pressure_type, group_map):
    """Every per-group surface this condition contributes, plus its population.

    One pass per (condition, bone, pressure type). The quality-control filter is
    applied to the whole matrix first and the group indices are rebuilt inside
    the survivors, so a group's mean can never include a knee the manifest says
    was dropped.
    """
    data = np.load(matrix_path(condition), allow_pickle=True)
    knees = [str(k) for k in data["subjects"]]

    drop, policy = excluded_knees(condition)
    flagged = {k for k in drop if k not in qc.load_chronic_knees()}
    chronic = qc.load_chronic_knees()
    keep = np.array([knee not in drop for knee in knees])
    kept_knees = [knee for knee in knees if knee not in drop]
    dropped = [
        {"knee": knee, "reason": _drop_reason(knee, flagged, chronic)}
        for knee in knees
        if knee in drop
    ]

    pressure_key = f"{bone}_pressure_{pressure_type}"
    pressure = data[pressure_key][keep]
    thinning = data[f"{bone}_thinning"][keep]
    has_pressure = data[f"{bone}_has_pressure"][keep]
    has_thinning = data[f"{bone}_has_thinning"][keep]

    surfaces = {}
    for group in GROUP_ROWS:
        if group == "all":
            index = np.arange(len(kept_knees))
        else:
            index = np.array(
                [i for i, knee in enumerate(kept_knees) if group_map.get(knee) == group]
            )
        if len(index) == 0:
            continue
        mean_pressure, pressure_n = group_mean(pressure[index], has_pressure[index])
        mean_thinning, thinning_n = group_mean(thinning[index], has_thinning[index])
        slope, slope_n = vertex_slope(
            pressure[index], thinning[index],
            has_pressure[index], has_thinning[index],
        )
        surfaces[group] = {
            "n": int(len(index)),
            "pressure": mean_pressure, "pressure_n": pressure_n,
            "thinning": mean_thinning, "thinning_n": thinning_n,
            "slope": slope, "slope_n": slope_n,
            "spm": load_spm_map(condition, bone, pressure_type, group),
        }

    population = {
        "matrix_n": len(knees),
        "n_dropped": len(dropped),
        "n": len(kept_knees),
        "groups": {
            group: surfaces[group]["n"] for group in DISEASE_GROUPS if group in surfaces
        },
        "qc_policy": policy,
        "dropped": dropped,
    }

    missing = [g for g, s in surfaces.items() if s["spm"] is None]
    if missing:
        # A missing SPM map would render as an empty panel with no error, so
        # report it rather than drawing nothing.
        print(
            f"    warning: no SPM for {condition}/{bone}/{pressure_type}: {missing}\n"
            f"       expected under {spm_dir()/'correlation'/condition/bone}\n"
            f"       build it with: python pipeline/derive_figure5_data.py spm --all"
        )
    return surfaces, population, kept_knees


def load_spm_map(condition, bone, pressure_type, group):
    """``{stat, significant}`` for one correlation cell, or ``None`` if absent."""
    import pyvista as pv

    subset = GROUP_SUBSET.get(group)
    if subset is None:
        return None
    path = (
        spm_dir() / "correlation" / condition / bone
        / f"corr_{bone}_thinning_vs_{pressure_type}_{subset}_results.vtk"
    )
    if not path.is_file():
        return None
    mesh = pv.read(str(path))
    return {
        "stat": np.asarray(mesh.point_data["stat_value"], dtype=np.float64),
        "significant": np.asarray(
            mesh.point_data.get("is_significant", np.zeros(mesh.n_points)), dtype=bool
        ),
    }


# --------------------------------------------------------------------------- #
# Color limits                                                                #
# --------------------------------------------------------------------------- #
def robust_limits(values, percentiles=(2, 98), symmetric=False):
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (-1.0, 1.0)
    low, high = np.percentile(finite, percentiles)
    if symmetric:
        extent = max(abs(low), abs(high))
        return (-extent, extent)
    return (float(low), float(high))


def pooled_limits(statistics):
    """One set of color limits for the whole figure, over both bones.

    ``statistics`` is ``{bone: {condition: {group: surfaces}}}``. Pooling over
    all panels lets a single colorbar caption the figure.
    """
    pressure, thinning, slope, spm = [], [], [], []
    for per_condition in statistics.values():
        for surfaces in per_condition.values():
            for entry in surfaces.values():
                pressure.append(entry["pressure"][entry["pressure_n"] >= MIN_KNEES_PER_VERTEX])
                thinning.append(entry["thinning"][entry["thinning_n"] >= MIN_KNEES_PER_VERTEX])
                slope.append(entry["slope"][entry["slope_n"] >= MIN_KNEES_PER_VERTEX])
                if entry["spm"] is not None:
                    spm.append(entry["spm"]["stat"])

    slope_pooled = np.concatenate(slope)
    low, high = BETA_CLIM_PCT
    slope_extent = max(
        abs(np.percentile(slope_pooled, low)), abs(np.percentile(slope_pooled, high))
    )
    return {
        "pressure": robust_limits(np.concatenate(pressure), (0, 99.5)),
        "thinning": robust_limits(np.concatenate(thinning), (2, 98), symmetric=True),
        "slope": (-float(slope_extent), float(slope_extent)),
        "spm": (robust_limits(np.concatenate(spm), (2, 98), symmetric=True)
                if spm else (-3.0, 3.0)),
    }


def pooled_contrast_limit(statistics):
    """Symmetric delta-pressure limits, pooled across both bones."""
    differences = []
    for per_condition in statistics.values():
        for surfaces in per_condition.values():
            for _name, first, second in CONTRASTS:
                if first not in surfaces or second not in surfaces:
                    continue
                a, b = surfaces[first], surfaces[second]
                valid = ((a["pressure_n"] >= MIN_KNEES_PER_VERTEX)
                         & (b["pressure_n"] >= MIN_KNEES_PER_VERTEX))
                differences.append((a["pressure"] - b["pressure"])[valid])
    if not differences:
        return (-1.0, 1.0)
    pooled = np.concatenate(differences)
    extent = max(abs(np.percentile(pooled, 5)), abs(np.percentile(pooled, 95)))
    return (-float(extent), float(extent))


# --------------------------------------------------------------------------- #
# Surface rendering                                                           #
# --------------------------------------------------------------------------- #
def _plotter_camera(plotter, mesh, view):
    """Position the camera for one of the two anatomical views.

    The reference frame is the canonical right knee: +x medial, +y anterior.
    Both views put anterior up, which is what makes femur and tibia panels
    stackable in one column.
    """
    x0, x1, y0, y1, z0, z1 = mesh.bounds
    center = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))
    span = max(x1 - x0, y1 - y0, z1 - z0)
    distance = 3 * span
    offset = -distance if view == "inferior" else distance
    plotter.camera_position = [
        (center[0], center[1], center[2] + offset),
        center,
        (0, 1, 0),
    ]
    plotter.camera.zoom(1.4)


def render_surface(mesh, scalars, limits, colormap, out_png, window,
                   outline=None, view="inferior"):
    """One surface render. ``outline`` traces a significant cluster in black."""
    import pyvista as pv

    surface = mesh.copy()
    surface["scalar"] = scalars
    plotter = pv.Plotter(off_screen=True, window_size=window)
    plotter.add_mesh(surface, scalars="scalar", cmap=colormap, clim=limits,
                     nan_color="lightgray", show_scalar_bar=False)
    if outline is not None and outline.any():
        marked = surface.copy()
        marked["significant"] = outline.astype(np.float32)
        region = marked.threshold(0.5, scalars="significant")
        if region.n_points > 0:
            edges = region.extract_feature_edges(
                boundary_edges=True, non_manifold_edges=False,
                feature_edges=False, manifold_edges=False,
            )
            plotter.add_mesh(edges, color="black", line_width=2)
    _plotter_camera(plotter, mesh, view)
    plotter.screenshot(str(out_png), transparent_background=False)
    plotter.close()


def content_box(png_path):
    """Bounding box of non-background pixels, or ``None``."""
    from PIL import Image

    pixels = np.asarray(Image.open(png_path).convert("RGB"))
    mask = (pixels < BACKGROUND_THRESHOLD).any(axis=2)
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    columns = np.where(mask.any(axis=0))[0]
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def crop_and_orient(sources, targets, bone, padding=CROP_PADDING_PX):
    """Crop every panel of one bone to a shared box, flip it, and save it.

    One box per bone: cropping each panel independently shifts panels by a
    pixel or two wherever their content differs, which is visible once they
    are stacked in a grid.

    The flip is applied here so that a panel on disk is the panel as printed,
    and the figure script needs no knowledge of the rendering convention.
    """
    from PIL import Image

    boxes = [content_box(p) for p in sources if Path(p).is_file()]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return
    left = min(b[0] for b in boxes)
    upper = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    lower = max(b[3] for b in boxes)
    width, height = Image.open(sources[0]).size
    box = (max(0, left - padding), max(0, upper - padding),
           min(width, right + padding), min(height, lower + padding))

    for source, target in zip(sources, targets):
        image = Image.open(source).crop(box)
        # Femur is viewed from below and tibia from above, so they need
        # different flips to land in the same medial-right convention.
        image = image.transpose(Image.ROTATE_180) if bone == "femur" \
            else image.transpose(Image.FLIP_TOP_BOTTOM)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        image.save(target)


# --------------------------------------------------------------------------- #
# The SPM stage                                                               #
# --------------------------------------------------------------------------- #
def spm_cells() -> list[dict]:
    """The 40 independent permutation cells, in a fixed order.

    Fixed so that ``--cell N`` means the same thing on every machine and a
    failed array task can be run again by index. The order also sets the seeds:
    a cell's permutation seed is ``SPM_SEED + index``, so reordering this list
    would change every p-value in the figure without any error.
    Append new cells at the end.
    """
    cells = []
    for condition in CONDITIONS:
        for subset in SUBSETS:
            for bone in BONES:
                for pressure_type in PRESSURES:
                    cells.append({
                        "kind": "correlation", "condition": condition,
                        "bone": bone, "pressure_type": pressure_type,
                        "subset": subset,
                        "name": f"corr_{bone}_thinning_vs_{pressure_type}_{subset}",
                    })
    for bone in BONES:
        cells.append({"kind": "anova", "bone": bone,
                      "name": f"anova_{bone}_thinning"})
    for bone in BONES:
        for first, second in PAIRS:
            cells.append({
                "kind": "ttest", "bone": bone, "groups": (first, second),
                "name": f"ttest_{bone}_thinning_{first}_vs_{second}",
            })
    for index, cell in enumerate(cells):
        cell["index"] = index
        cell["seed"] = SPM_SEED + index
    return cells


def _import_spm():
    try:
        import pyKneeSPM  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pyKneeSPM is required for the `spm` stage and is not installed.\n\n"
            "    pip install git+https://github.com/gattia/pyKneeSPM.git\n\n"
            "Install the commit docs/pinned_versions.md names. Earlier commits "
            "have no `seed` parameter, so they will run but will not reproduce "
            "these p-values.\n\n"
            "The `maps` stage does not need it -- it reads the cluster maps the "
            "`spm` stage already wrote."
        ) from exc
    # Checked on the class that declares the parameter. The concrete
    # SimpleCorrelation / TwoSampleComparison take (*args, **kwargs) and
    # would report no `seed` however new the install is.
    from pyKneeSPM.knee_spm import SingleStatisticSPM

    if "seed" not in inspect.signature(SingleStatisticSPM.__init__).parameters:
        # An unseeded pyKneeSPM runs without error and draws a different null
        # every time. Fail here instead of writing p-values that no one can
        # reproduce and nothing downstream can detect.
        raise SystemExit(
            "The installed pyKneeSPM predates the permutation seed "
            f"({Path(pyKneeSPM.__file__).parent}).\n\n"
            "Its nulls are unseeded, so this stage would write p-values that do "
            "not reproduce -- and nothing downstream could tell. Update to the "
            "commit in docs/pinned_versions.md."
        )
    return pyKneeSPM


def _pykneespm_provenance() -> dict:
    """Where pyKneeSPM came from and at which commit, for the stage metadata."""
    import subprocess

    pyKneeSPM = _import_spm()
    path = Path(pyKneeSPM.__file__).resolve().parent
    record = {"path": str(path), "commit": None, "dirty": None}
    try:
        record["commit"] = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        record["dirty"] = bool(subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip())
    except Exception:
        pass  # an installed (non-git) copy still records its path, and the seed
    return record


def write_seed_metadata(missing: str = "zero") -> Path:
    """Record the seed plan for the whole stage, next to the cluster maps.

    Every array task writes it, so a partially repeated array still leaves a
    correct record. The content is a pure function of the
    cell plan and the installed pyKneeSPM, so concurrent writers agree; the write
    is atomic so a reader never sees half a file.

    Per-cell seeds also land in each cell's own summary. This file exists so that
    "what seed produced this tree" is answerable without opening forty of them.

    ``missing="complete-case"`` writes the same plan, restricted to the 32
    correlation cells and naming the missing-value rule, into
    ``spm_complete_case/`` instead. The seeds are the same numbers on purpose:
    the sensitivity run draws the identical permutations.
    """
    payload = {
        "base_seed": SPM_SEED,
        "seed_rule": "SPM_SEED + cell index, in the order spm_cells() returns",
        "n_permutations": MC_ITERATIONS,
        "pykneespm": _pykneespm_provenance(),
        "cells": [{"index": c["index"], "name": c["name"], "seed": c["seed"]}
                  for c in spm_cells()],
        "note": ("Cluster p-values are reproducible given these seeds and this "
                 "pyKneeSPM commit. Earlier pyKneeSPM commits drew the permutations "
                 "from an unseeded global RNG, so their p-values do not repeat."),
    }
    if missing == "zero":
        target = spm_dir(create=True) / "_seeds.json"
    else:
        payload["missing_policy"] = missing
        payload["cells"] = [c for c, cell in zip(payload["cells"], spm_cells())
                            if cell["kind"] == "correlation"]
        payload["note"] += (" This tree is the complete-case sensitivity run of the "
                            "correlation cells; the figure reads spm/, not this.")
        target = spm_complete_case_dir(create=True) / "_seeds.json"
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False,
                                     suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2)
        temporary = Path(handle.name)
    # NamedTemporaryFile creates at 0600. Every other file in the data root is
    # world-readable and this one ships: a reader who cannot open it cannot
    # check which seed produced the maps sitting beside it.
    temporary.chmod(0o644)
    temporary.replace(target)
    return target


def _cell_output_dir(cell, root: Path | None = None) -> Path:
    """Where one cell writes: under ``spm/`` unless ``root`` says otherwise."""
    root = spm_dir() if root is None else root
    if cell["kind"] == "correlation":
        return root / "correlation" / cell["condition"] / cell["bone"]
    return root / "thinning" / cell["kind"] / cell["bone"]


# --------------------------------------------------------------------------- #
# Complete-case correlation (--missing complete-case)                         #
# --------------------------------------------------------------------------- #
def complete_case_correlation(data):
    """Per-vertex Pearson r and t over the knees with both values finite.

    ``data`` is ``(knees, vertices, 2)`` -- thickness change, pressure -- with
    NaN wherever a knee has no value. Returns ``(t, r, n)``: ``n`` is each
    vertex's count of complete knees, and ``t = r * sqrt((n - 2) / (1 - r**2))``,
    so the degrees of freedom are that vertex's own. Non-finite t and r (fewer
    than three complete knees, a constant column) are returned as they are and
    zeroed by the caller, as pyKneeSPM zeroes its own.

    The arithmetic follows pyKneeSPM's ``CorrelationTTest`` step for step --
    same dtype, same summation order, and the float64 detour numpy's own
    ``mean`` takes when it divides by an integer count -- so that on data with
    nothing missing this returns the zero-fill statistic bit for bit. That is
    what lets a comparison of the two trees attribute every difference to the
    missing-value rule: a statistic that differed in its last bit could carry a
    vertex across the cluster-forming threshold by itself.
    """
    # Contiguous working copies, zeroed and centred in place: this runs once per
    # permutation, and in-place masking halves its cost against np.where copies
    # (measured at n = 728: 1.3 s vs 2.7 s, against 1.6 s for pyKneeSPM's own).
    # (Always copies: a column of (knees, vertices, 2) is never contiguous. The
    # caller's array -- the permutation batch, the observed data -- is not touched.)
    first = np.array(data[:, :, 0], order="C", copy=True)
    second = np.array(data[:, :, 1], order="C", copy=True)
    incomplete = ~(np.isfinite(first) & np.isfinite(second))
    n = first.shape[0] - incomplete.sum(axis=0)
    np.copyto(first, 0, where=incomplete)
    np.copyto(second, 0, where=incomplete)
    with np.errstate(divide="ignore", invalid="ignore"):
        # np.mean on float32: a float32 sum, divided by an intp count through a
        # float64 loop, cast back. The per-vertex count takes the scalar's place.
        mean_first = first.sum(axis=0)
        np.true_divide(mean_first, n, out=mean_first, casting="unsafe")
        mean_second = second.sum(axis=0)
        np.true_divide(mean_second, n, out=mean_second, casting="unsafe")
        # An incomplete knee's residual is set to 0, and adding 0.0
        # leaves a running sum unchanged -- so each sum runs over the complete
        # knees only, in the same order the zero-fill sum visits them.
        first -= mean_first
        second -= mean_second
        np.copyto(first, 0, where=incomplete)
        np.copyto(second, 0, where=incomplete)
        numerator = np.sum(first * second, axis=0)
        denominator = (np.sqrt(np.sum(first ** 2, axis=0))
                       * np.sqrt(np.sum(second ** 2, axis=0)))
        r = numerator / denominator
        # The count cast to the data's dtype: pyKneeSPM divides a Python int
        # by a float32 array, which stays float32; an int64 array would not.
        t = r * np.sqrt((n - 2).astype(data.dtype) / (1 - r ** 2))
    return t, r, n


def complete_case_coverage(thinning) -> np.ndarray:
    """Knees per vertex that count toward ``VERTEX_COVERAGE``.

    pyKneeSPM counts ``thickness change != 0``. With NaN for missing, that
    count would include every missing knee (``nan != 0`` is True), so finiteness
    is required explicitly -- which makes the included vertices the same ones
    the zero fill admits, where a missing knee's 0 did not count either.
    """
    return np.sum(np.isfinite(thinning) & (thinning != 0), axis=0)


@functools.lru_cache(maxsize=None)
def _complete_case_spm_class():
    """pyKneeSPM's ``SimpleCorrelation``, with the complete-case statistic.

    Three subclasses, and only the statistic changes. pyKneeSPM names its t
    class as a module global inside two methods (the observed map and the Monte
    Carlo loop), so those two methods are restated here with the complete-case
    class in its place. Copying them, instead of patching the library's module
    global, leaves the zero-fill path untouched. Everything
    else is inherited: the permutation scheme (``PermutationCorrelationTTest``,
    so the same seed draws the same matrix), the clustering, and the
    cluster-extent threshold.

    Built on first use because pyKneeSPM is needed only by the ``spm`` stage,
    and the ``maps`` stage imports this module without it.
    """
    _import_spm()
    from pyKneeSPM import test_statistics
    from pyKneeSPM.clustering import Cluster
    from pyKneeSPM.knee_spm import SimpleCorrelation
    from pyKneeSPM.monte_carlo import MonteCarloClusterCorrelationTest
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    class CompleteCaseCorrelationTTest(test_statistics.CorrelationTTest):
        """``CorrelationTTest`` over complete knees; output handling verbatim."""

        def compute_statistics_per_node(self):
            t_statistic, correlations, _ = complete_case_correlation(self.data)

            if self.idx_not_to_include is not None:
                t_statistic[self.idx_not_to_include] = 0
                correlations[self.idx_not_to_include] = 0

            t_statistic = test_statistics.replace_non_finite_values(
                t_statistic, replace_value=0.)
            correlations = test_statistics.replace_non_finite_values(
                correlations, replace_value=0.)

            t_scalars = numpy_to_vtk(t_statistic)
            t_scalars.SetName('t_statistic')
            corr_scalars = numpy_to_vtk(correlations)
            corr_scalars.SetName('correlation')
            mesh = (self.test_statistics_mesh if self.return_new_mesh is True
                    else self.mesh)
            mesh.GetPointData().AddArray(t_scalars)
            mesh.GetPointData().AddArray(corr_scalars)

    class CompleteCaseMonteCarlo(MonteCarloClusterCorrelationTest):
        """The permutation null, with the complete-case t per permutation.

        ``perform_monte_carlo`` is pyKneeSPM's (monte_carlo.py, commit f8ac9fa)
        with one name changed. Pressure is column 1, so the permutation carries
        each knee's NaNs with its pressure row; thickness change stays put.
        """

        def perform_monte_carlo(self):
            iter_count = 0
            for batch_start in range(0, self.n_iterations, self.permutation_batch_size):
                batch_end = min(batch_start + self.permutation_batch_size, self.n_iterations)
                batch = self.perm_test.apply_permutation_batch(
                    self.data_to_simulate, batch_start, batch_end)

                for i in range(batch.shape[0]):
                    if (self.idx_to_include is not None) & (self.method == 'permutation'):
                        change_array = np.zeros_like(self.data)
                        change_array[:, self.idx_to_include] = batch[i]
                    elif (self.idx_to_include is None) & (self.method == 'permutation'):
                        change_array = batch[i]
                    else:
                        raise Exception('only the permutation method is implemented')

                    t_test = CompleteCaseCorrelationTTest(
                        change_array, self.ref_mesh, return_new_mesh=True,
                        idx_not_to_include=self.idx_not_to_include)
                    t_test.compute_statistics_per_node()
                    t_mesh = t_test.get_statistics_mesh()
                    t_mesh.GetPointData().SetActiveScalars(self.map_name)

                    t_scalars = vtk_to_numpy(t_mesh.GetPointData().GetArray(self.map_name))
                    self.max_test_statistics[self.map_name]['min'].append(np.nanmin(t_scalars))
                    self.max_test_statistics[self.map_name]['max'].append(np.nanmax(t_scalars))
                    self.max_test_statistics[self.map_name]['abs_max'].append(
                        np.nanmax(np.abs(t_scalars)))

                    clust = Cluster(statistic_mesh=t_mesh,
                                    statistic_threshold=self.threshold,
                                    threshold_type='two_sided',
                                    clust_names='cluster',
                                    clust_idx=0)
                    self.cluster_sizes[self.map_name]['all'].append(clust.get_areas())
                    iter_count += 1
                    print('Completed iteration: {}/{}'.format(iter_count, self.n_iterations))

                del batch

    class CompleteCaseSimpleCorrelation(SimpleCorrelation):
        """``SimpleCorrelation`` whose missing values are NaN, not 0."""

        def compute_test_statistics(self):
            coverage = complete_case_coverage(self.change_values[:, :, 0])
            self.idx_no_data = np.where(
                coverage < self.percent_participants_with_data_to_include_vertex
                * self.change_values.shape[0])
            test = CompleteCaseCorrelationTTest(self.change_values,
                                                self.reference_mesh['mesh'],
                                                return_new_mesh=True,
                                                idx_not_to_include=self.idx_no_data)
            test.compute_statistics_per_node()
            test_mesh = test.get_statistics_mesh()
            test_mesh.GetPointData().SetActiveScalars(self.map_name)
            self.test_statistic_maps[self.map_name] = test_mesh

        def compute_mc_thresholds(self):
            # Kept per threshold so a check can confirm the permutations are the
            # zero-fill run's own; the matrix is small next to one batch.
            self.permutation_matrices = {}
            for threshold in self.map_threshold:
                mc_sim = CompleteCaseMonteCarlo(self.reference_mesh['mesh'],
                                                self.change_values,
                                                method=self.mc_cluster_method,
                                                threshold=threshold,
                                                n_iterations=self.n_monte_carlo_iterations,
                                                idx_not_to_include=self.idx_no_data,
                                                idx_to_include=None,
                                                seed=self.seed)
                mc_sim.update()
                self.permutation_matrices[threshold] = (
                    mc_sim.perm_test.get_permutation_indices())
                self.threshold_cluster_distribution[threshold] = (
                    mc_sim.get_distribution_of_max_clustersizes())
                self.threshold_cluster_size[threshold] = mc_sim.get_threshold_clustersize(
                    threshold=self.mc_cluster_extent_significance)
                self.threshold_test_statistic[threshold] = mc_sim.get_threshold_test_statistic(
                    threshold=self.mc_point_significance)

    return CompleteCaseSimpleCorrelation


def _complete_case_mean(values, complete):
    """Mean over the complete knees per vertex; 0 where there are none."""
    counts = complete.sum(axis=0)
    total = np.where(complete, values, 0.0).sum(axis=0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(total / counts, nan=0.0).astype(values.dtype)


def _write_text_atomic(target: Path, text: str) -> None:
    """Write ``text`` so that a concurrent reader never sees half of it."""
    # UTF-8 explicitly: VTK resets the C locale once it has read a mesh, after
    # which the default encoding is ASCII.
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False,
                                     suffix=".tmp", encoding="utf-8") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    temporary.replace(target)


def write_complete_case_manifest(analysis_name: str, rows) -> Path:
    """The inclusion manifest for a complete-case cell, inside its own tree.

    ``qc.write_inclusion_manifest`` is not used: it writes to the shared
    ``04_results/inclusion/`` under the same name the zero-fill cell uses, and a
    sensitivity run must not overwrite the reported run's record. (The knees
    are the same ones -- only the per-vertex use of them differs.)
    """
    out_dir = spm_complete_case_dir(create=True) / "inclusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{analysis_name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["knee", "group", "condition"])
        writer.writeheader()
        writer.writerows(rows)
    return path


COMPLETE_CASE_README = """\
# spm_complete_case/ -- complete-case sensitivity run of Figure 5's SPM

**Nothing reads this tree.** The figure, `spm_clusters.csv`, every other table
and the manuscript read `../spm/`. This is an opt-in sensitivity analysis.

## What it is

The 32 correlation cells of `../spm/correlation/` (thickness change vs contact
pressure, per vertex, across knees), re-run with one change: a knee with no
value at a vertex is left out of that vertex's correlation (complete-case),
instead of entering it as pressure = 0, thickness change = 0 (the published
zero fill). Each vertex's r, its n and the t's degrees of freedom come from the
knees with both values there.

Held fixed, so the missing-value rule is the only difference:

- the included vertices (>= {coverage:.0%} of knees with a finite, non-zero
  thickness change -- the same set the zero fill admits);
- the permutation matrix (same seeds, `_seeds.json`; pressure shuffled across
  knees with its missing pattern travelling with it);
- {n_permutations:,} permutations, cluster-forming |t| >= {threshold}, cluster extent =
  surface area, 95th-percentile extent threshold, and the same rule for a
  cluster's p-value (its extent's rank in the permutation null);
- the knees in each cell (`inclusion/` -- identical to the published cells').

Every summary JSON records `"missing_policy": "complete-case"`. The supplement's
group-difference cells (32-39) are not here: they have no pairing to break.

## Regenerate

From the repository root, one cell per array task:

    python pipeline/derive_figure5_data.py spm --cell N --missing complete-case   # N = 0..31
"""


def write_complete_case_readme() -> Path:
    target = spm_complete_case_dir(create=True) / "README.md"
    _write_text_atomic(target, COMPLETE_CASE_README.format(
        coverage=VERTEX_COVERAGE, n_permutations=MC_ITERATIONS,
        threshold=CLUSTER_FORMING_T))
    return target


def save_spm_results(spm, output_dir: Path, run_name: str, extra_scalars=None):
    """Write one cell's cluster map and summary.

    Scalars on the saved surface: the raw statistic, the clusters that formed at
    the vertex threshold, those that then survived permutation inference, a
    binary mask of the latter, and each surviving cluster's p-value. The mask is
    what the figure draws; the rest is what makes the map auditable.
    """
    import pyvista as pv
    from scipy.spatial import cKDTree
    from vtk.util.numpy_support import vtk_to_numpy

    output_dir.mkdir(parents=True, exist_ok=True)
    statistic_maps = spm.get_test_statistic_maps()
    if not statistic_maps:
        raise RuntimeError(f"{run_name}: the SPM produced no statistic map")

    key = list(statistic_maps.keys())[0]
    surface = pv.wrap(statistic_maps[key]).copy()
    statistic = np.array(surface.point_data[key], dtype=np.float64)
    surface["stat_value"] = statistic
    tree = cKDTree(surface.points)

    def vertices_of(cluster_mesh):
        points = vtk_to_numpy(cluster_mesh.GetPoints().GetData())
        distances, indices = tree.query(points)
        return indices[distances < 1e-6]

    clustered = spm.get_clustered_statistic_maps()
    surviving = spm.get_significant_clusters()
    threshold_sizes = spm.get_threshold_cluster_size()
    null_distributions = spm.get_cluster_distributions()

    formed = np.zeros(surface.n_points, dtype=np.int32)
    formed_id = 1
    for stat_key in clustered:
        for threshold_key in clustered[stat_key]:
            for cluster in clustered[stat_key][threshold_key].values():
                if "mesh" in cluster:
                    formed[vertices_of(cluster["mesh"])] = formed_id
                formed_id += 1
    surface["thresholded_cluster"] = formed

    null = next((np.array(v) for v in null_distributions.values()), None)
    significant = np.zeros(surface.n_points, dtype=np.int32)
    clusters, cluster_id = [], 1
    for stat_key in surviving:
        for threshold_key in surviving[stat_key]:
            for cluster in surviving[stat_key][threshold_key].values():
                area = float(cluster["area"])
                vertices = vertices_of(cluster["mesh"]) if "mesh" in cluster else []
                significant[vertices] = cluster_id
                clusters.append({
                    "cluster_id": cluster_id,
                    "area": area,
                    # The rank of this cluster's extent in the permutation null.
                    "p_value_approx": (float(np.mean(null >= area))
                                       if null is not None and len(null) else None),
                    "n_vertices": int(len(vertices)),
                })
                cluster_id += 1

    surface["significant_cluster"] = significant
    surface["is_significant"] = (significant > 0).astype(np.float32)
    surface["significant_stat"] = np.where(significant > 0, statistic, 0.0)
    per_vertex_p = np.full(surface.n_points, np.nan)
    for cluster in clusters:
        if cluster["p_value_approx"] is not None:
            per_vertex_p[significant == cluster["cluster_id"]] = cluster["p_value_approx"]
    surface["cluster_p_value"] = per_vertex_p

    for label, values in (extra_scalars or {}).items():
        surface[label] = np.where(significant > 0, values, 0.0)

    surface.save(str(output_dir / f"{run_name}_results.vtk"))
    summary = {
        "run_name": run_name,
        "n_significant_clusters": len(clusters),
        "significant_cluster_areas": [c["area"] for c in clusters],
        "total_significant_area": sum(c["area"] for c in clusters),
        "significant_clusters": clusters,
        "threshold_cluster_sizes": {str(k): float(v) for k, v in threshold_sizes.items()},
        "n_thresholded_clusters": int(formed_id - 1),
    }
    return summary, surface


def run_correlation_cell(cell, missing: str = "zero") -> dict:
    """Pressure-to-thickness-change correlation over the surface, one cell.

    ``missing`` is the missing-value rule (see the module docstring). "zero" is
    the reported analysis and writes under ``spm/``; "complete-case" writes
    under ``spm_complete_case/`` and records the rule in the cell's summary.
    """
    pyKneeSPM = _import_spm()
    started = time.time()
    condition, bone = cell["condition"], cell["bone"]
    pressure_type, subset = cell["pressure_type"], cell["subset"]

    data = np.load(matrix_path(condition), allow_pickle=True)
    knees = [str(k) for k in data["subjects"]]
    group_map = load_group_map()
    drop, policy = excluded_knees(condition)

    wanted = SUBSET_GROUP[subset]
    index = np.array([
        i for i, knee in enumerate(knees)
        if knee not in drop and knee in group_map
        and (wanted == "all" or group_map[knee] == wanted)
    ])
    print(f"  n = {len(index)} ({condition} / {bone} / {pressure_type} / {subset})")

    pressure = data[f"{bone}_pressure_{pressure_type}"][index].copy()
    thinning = data[f"{bone}_thinning"][index].copy()
    both = data[f"{bone}_has_pressure"][index] & data[f"{bone}_has_thinning"][index]
    if missing == "zero":
        pressure[~both] = 0
        thinning[~both] = 0
        correlation_class = pyKneeSPM.SimpleCorrelation
        root = spm_dir()
    else:
        # Missing values are NaN, so a missing knee drops out of that vertex's
        # statistic.
        pressure[~both] = np.nan
        thinning[~both] = np.nan
        correlation_class = _complete_case_spm_class()
        root = spm_complete_case_dir(create=True)

    spm = correlation_class(
        map_threshold=CLUSTER_FORMING_T,
        n_monte_carlo_iterations=MC_ITERATIONS,
        percent_participants_with_data_to_include_vertex=VERTEX_COVERAGE,
        seed=cell["seed"],
    )
    spm.add_reference_change_mesh(str(reference_surface_path(bone)))
    spm.add_change_data_directly(np.stack([thinning, pressure], axis=2))
    spm.update()

    # Pearson r alongside the t map: the t statistic answers "is there an
    # effect", r answers "how strong", and a reader needs both.
    stacked = np.stack([thinning, pressure], axis=2)
    if missing == "zero":
        residual = stacked - stacked.mean(axis=0)
        numerator = np.sum(residual[:, :, 0] * residual[:, :, 1], axis=0)
        spread = (np.sqrt(np.sum(residual[:, :, 0] ** 2, axis=0))
                  * np.sqrt(np.sum(residual[:, :, 1] ** 2, axis=0)))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_values = np.nan_to_num(numerator / spread, nan=0.0)
        mean_pressure = pressure.mean(axis=0)
        mean_thinning = thinning.mean(axis=0)
    else:
        _, r_values, n_complete = complete_case_correlation(stacked)
        r_values = np.nan_to_num(r_values, nan=0.0)
        complete = np.isfinite(thinning) & np.isfinite(pressure)
        mean_pressure = _complete_case_mean(pressure, complete)
        mean_thinning = _complete_case_mean(thinning, complete)

    summary, _ = save_spm_results(
        spm, _cell_output_dir(cell, root), cell["name"],
        extra_scalars={
            f"mean_pressure_{pressure_type}": mean_pressure,
            "mean_thinning": mean_thinning,
            "correlation_r": r_values,
        },
    )
    summary.update(condition=condition, bone=bone, pressure_type=pressure_type,
                   subset=subset, n_subjects=int(len(index)), qc_policy=policy,
                   permutation_seed=cell["seed"], cell_index=cell["index"],
                   n_permutations=MC_ITERATIONS,
                   elapsed_seconds=time.time() - started)
    manifest_name = f"figure5_spm_{condition}_{bone}_{pressure_type}_{subset}"
    manifest_rows = [{"knee": knees[i], "group": group_map[knees[i]],
                      "condition": condition} for i in index]
    if missing == "zero":
        qc.write_inclusion_manifest(manifest_name, manifest_rows)
    else:
        # How many knees each analysed vertex actually used -- the quantity the
        # zero fill held at n and this run lets vary.
        included = np.ones(n_complete.shape, dtype=bool)
        included[spm.idx_no_data] = False
        used = n_complete[included]
        summary.update(
            missing_policy=missing,
            complete_knees_per_included_vertex={
                "min": int(used.min()) if used.size else None,
                "median": float(np.median(used)) if used.size else None,
                "max": int(used.max()) if used.size else None,
                "n_included_vertices": int(included.sum()),
            },
        )
        write_complete_case_manifest(manifest_name, manifest_rows)
    return summary


def run_thinning_cell(cell) -> dict:
    """Group differences in cartilage thickness change (8 cells; a data output).

    Pressure quality control is not applied here. The shared
    table flags contact-pressure artifacts; thickness change is an MRI
    measurement that does not depend on the solve, so excluding on it would
    shrink n for no methodological reason. The population is instead bounded by
    the matrix, which requires both measurements.

    Thickness change does not depend on gait, so this result is computed once
    and serves every walking condition.
    """
    pyKneeSPM = _import_spm()
    from scipy import stats as scipy_stats

    started = time.time()
    bone = cell["bone"]
    # The reference condition's matrix, used only for its thickness arrays and
    # its knee list -- the values are identical in every condition's matrix.
    data = np.load(matrix_path(CONDITIONS[0]), allow_pickle=True)
    knees = [str(k) for k in data["subjects"]]
    group_map = load_group_map()

    if cell["kind"] == "anova":
        index = np.array([i for i, k in enumerate(knees) if k in group_map])
        labels = np.array([group_map[knees[i]] for i in index])
        n_groups = len(np.unique(labels))
        threshold = float(scipy_stats.f.isf(
            ALPHA_CLUSTER_FORMING, n_groups - 1, len(index) - n_groups))
        comparison = pyKneeSPM.ANOVAComparison
        extra = None
    else:
        first, second = cell["groups"]
        first_key, second_key = SUBSET_GROUP[first], SUBSET_GROUP[second]
        index, labels = [], []
        for i, knee in enumerate(knees):
            group = group_map.get(knee)
            if group == first_key:
                index.append(i), labels.append(0)
            elif group == second_key:
                index.append(i), labels.append(1)
        index, labels = np.array(index), np.array(labels)
        threshold = CLUSTER_FORMING_T
        comparison = pyKneeSPM.TwoSampleComparison
        extra = "pair"

    values = data[f"{bone}_thinning"][index].copy()
    values[~data[f"{bone}_has_thinning"][index]] = 0
    print(f"  n = {len(index)} ({cell['name']}), threshold = {threshold:.3f}")

    spm = comparison(
        map_threshold=threshold,
        n_monte_carlo_iterations=MC_ITERATIONS,
        percent_participants_with_data_to_include_vertex=VERTEX_COVERAGE,
        seed=cell["seed"],
    )
    spm.add_reference_change_mesh(str(reference_surface_path(bone)))
    spm.add_change_data_directly(values, labels)
    spm.update()

    if extra is None:
        scalars = {f"group_mean_{g.replace('kl0_1_', '')}": values[labels == g].mean(axis=0)
                   for g in np.unique(labels)}
    else:
        first, second = cell["groups"]
        a, b = values[labels == 0], values[labels == 1]
        scalars = {
            f"group_mean_{first}": a.mean(axis=0),
            f"group_mean_{second}": b.mean(axis=0),
            # The direction the omnibus F map cannot carry. Stored as
            # first - second, so a negative value means the first group thins
            # faster; the figure states that in its caption.
            f"group_mean_{first}_minus_{second}": a.mean(axis=0) - b.mean(axis=0),
        }

    summary, _ = save_spm_results(spm, _cell_output_dir(cell), cell["name"],
                                  extra_scalars=scalars)
    summary.update(bone=bone, kind=cell["kind"], n_subjects=int(len(index)),
                   map_threshold=threshold, elapsed_seconds=time.time() - started,
                   permutation_seed=cell["seed"], cell_index=cell["index"],
                   n_permutations=MC_ITERATIONS,
                   qc_policy={"applied": False,
                              "reason": "thickness change is an MRI measurement, "
                                        "not a contact-pressure one"})
    return summary


def run_spm(cells, missing: str = "zero") -> None:
    if missing not in MISSING_POLICIES:
        raise SystemExit(f"--missing must be one of {MISSING_POLICIES}, not {missing!r}")
    if missing == "zero":
        write_seed_metadata()
        root = spm_dir()
    else:
        unsupported = [f"{c['index']} ({c['name']})" for c in cells
                       if c["kind"] != "correlation"]
        if unsupported:
            # Checked before anything is written, so a mistyped array range
            # cannot leave a half-built tree behind.
            raise SystemExit(
                f"--missing {missing} applies to the correlation cells only "
                f"(0-{sum(c['kind'] == 'correlation' for c in spm_cells()) - 1}). "
                "The thickness-change group tests have no pressure column and no "
                "missing pairs to drop; their result is the one under spm/thinning/.\n"
                "Not supported: " + ", ".join(unsupported))
        write_seed_metadata(missing=missing)
        write_complete_case_readme()
        root = spm_complete_case_dir()
    for cell in cells:
        print(f"\n=== {cell['name']} === "
              f"[cell {cell['index']}, seed {cell['seed']}]"
              + ("" if missing == "zero" else f" [missing: {missing}]"))
        summary = (run_correlation_cell(cell, missing) if cell["kind"] == "correlation"
                   else run_thinning_cell(cell))
        path = _cell_output_dir(cell, root) / f"{cell['name']}_summary.json"
        path.write_text(json.dumps(summary, indent=2))
        n = summary["n_significant_clusters"]
        smallest = min(
            (c["p_value_approx"] for c in summary["significant_clusters"]
             if c["p_value_approx"] is not None), default=None)
        print(f"  {n} surviving cluster(s)"
              + (f", smallest p ≈ {smallest:.4f}" if smallest is not None else "")
              + f"  [{summary['elapsed_seconds']:.0f}s]")


# --------------------------------------------------------------------------- #
# The maps stage                                                              #
# --------------------------------------------------------------------------- #
def read_summary(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def format_p(clusters) -> str:
    """Smallest cluster p, as ``n.s.``, ``<0.0001``, or a number.

    ``n.s.`` when no cluster survived (a bare 0 would read as the strongest
    possible result), ``<0.0001`` below the resolution of 10,000 permutations,
    and the value otherwise.
    """
    values = [c["p_value_approx"] for c in clusters if c.get("p_value_approx") is not None]
    if not values:
        return "n.s."
    smallest = min(values)
    # 10,000 permutations resolve to 1e-4; below that, no permutation reached
    # this cluster's extent, and the result is reported as a bound.
    return f"{smallest:.4f}" if smallest >= 1.0 / MC_ITERATIONS else "<0.0001"


def significant_vertices(results_vtk: Path) -> int:
    """Vertices inside a surviving cluster, counted from the mask itself.

    Counted from the map: two clusters can share a boundary vertex, which a
    sum of per-cluster counts counts twice and the mask counts once. The
    difference is one or two vertices in a few thousand,
    but the mask is what gets drawn, so it is the number that describes the
    figure.
    """
    import pyvista as pv

    if not results_vtk.is_file():
        return 0
    mesh = pv.read(str(results_vtk))
    return int(np.asarray(mesh.point_data["is_significant"], float).astype(bool).sum())


def write_cluster_tables() -> None:
    """The manuscript's numbers, one row per SPM cell."""
    rows = []
    for condition in CONDITIONS:
        for bone in BONES:
            for pressure_type in PRESSURES:
                for subset in SUBSETS:
                    name = f"corr_{bone}_thinning_vs_{pressure_type}_{subset}"
                    cell_dir = spm_dir() / "correlation" / condition / bone
                    summary = read_summary(cell_dir / f"{name}_summary.json")
                    clusters = summary.get("significant_clusters", [])
                    rows.append({
                        "condition": condition,
                        "condition_label": CONDITION_LABEL[condition],
                        "bone": bone,
                        "pressure_type": pressure_type,
                        "subset": subset,
                        "n_knees": summary.get("n_subjects", ""),
                        "n_clusters": summary.get("n_significant_clusters", ""),
                        "n_significant_vertices": significant_vertices(
                            cell_dir / f"{name}_results.vtk"),
                        "total_significant_area": round(
                            float(summary.get("total_significant_area", 0)), 5),
                        "min_cluster_p": format_p(clusters),
                        "cluster_p_values": ";".join(
                            f"{c['p_value_approx']:.4f}" for c in clusters
                            if c.get("p_value_approx") is not None),
                    })
    _write_csv(results_dir() / "spm_clusters.csv", rows)
    print(f"wrote spm_clusters.csv: {len(rows)} cells")

    supplement = []
    for bone in BONES:
        stem = spm_dir() / "thinning" / "anova" / bone / f"anova_{bone}_thinning"
        supplement.append(_supplement_row(bone, "omnibus_anova", stem))
        for first, second in PAIRS:
            stem = (spm_dir() / "thinning" / "ttest" / bone
                    / f"ttest_{bone}_thinning_{first}_vs_{second}")
            supplement.append(_supplement_row(bone, f"{first}_vs_{second}", stem))
    _write_csv(results_dir() / "supplement_clusters.csv", supplement)
    print(f"wrote supplement_clusters.csv: {len(supplement)} cells")


def _supplement_row(bone, test, stem: Path):
    summary = read_summary(stem.with_name(stem.name + "_summary.json"))
    clusters = summary.get("significant_clusters", [])
    return {
        "bone": bone,
        "test": test,
        "n_knees": summary.get("n_subjects", ""),
        "n_clusters": summary.get("n_significant_clusters", ""),
        "n_significant_vertices": significant_vertices(
            stem.with_name(stem.name + "_results.vtk")),
        "total_significant_area": round(float(summary.get("total_significant_area", 0)), 5),
        "min_cluster_p": format_p(clusters),
        "cluster_p_values": ";".join(
            f"{c['p_value_approx']:.4f}" for c in clusters
            if c.get("p_value_approx") is not None),
    }


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_region_composition() -> None:
    """Where the thickness-change group comparison's significant clusters sit.

    Split by the sign of the group-mean difference before summarising. A
    contrast whose significant vertices include a positive medial lobe and a
    negative trochlear one has no single composition, and averaging the two
    would hide both.
    """
    import pyvista as pv

    rows = []
    for bone in BONES:
        coarse = region_labels(bone)
        names = REGION_NAMES[bone]

        def composition(mask):
            total = int(mask.sum())
            if total == 0:
                return {}
            return {name: 100.0 * float(((coarse == code) & mask).sum()) / total
                    for code, name in names.items()}

        paths = sorted((spm_dir() / "thinning" / "anova" / bone).glob("*_results.vtk"))
        paths += sorted((spm_dir() / "thinning" / "ttest" / bone).glob("*_results.vtk"))
        if len(paths) != 1 + len(PAIRS):
            raise SystemExit(
                f"expected {1 + len(PAIRS)} supplement result files for {bone}, "
                f"found {len(paths)} under {spm_dir()/'thinning'} -- the SPM "
                "array is incomplete. Refusing to write a table that would look "
                "complete."
            )
        for path in paths:
            mesh = pv.read(str(path))
            significant = np.asarray(mesh.point_data["is_significant"]).astype(bool)
            run = path.name.replace("_results.vtk", "")
            if run.startswith("anova"):
                rows.append((bone, run, "all", int(significant.sum()), None,
                             composition(significant)))
                continue
            difference_names = [k for k in mesh.point_data.keys() if "_minus_" in k]
            if len(difference_names) != 1:
                raise ValueError(f"{path}: expected one difference scalar, "
                                 f"found {difference_names}")
            difference = np.asarray(mesh.point_data[difference_names[0]])
            for sign, label in ((difference > 0, "positive"), (difference < 0, "negative")):
                mask = significant & sign
                if not mask.any():
                    continue
                rows.append((
                    bone, run,
                    f"{label} ({difference_names[0].replace('group_mean_', '')})",
                    int(mask.sum()), 1000.0 * float(difference[mask].mean()),
                    composition(mask),
                ))

    path = results_dir() / "region_composition.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bone", "run", "side", "n_vertices", "mean_delta_um_per_year",
                         "pct_medial", "pct_lateral", "pct_trochlea"])
        for bone, run, side, n, delta, parts in rows:
            writer.writerow([bone, run, side, n,
                             "" if delta is None else round(delta, 2),
                             round(parts.get("medial", 0), 1),
                             round(parts.get("lateral", 0), 1),
                             round(parts.get("trochlea", 0), 1)])
    print(f"wrote region_composition.csv: {len(rows)} rows")


def render_main_panels(statistics, limits, contrast_limit, pressure_type, bones):
    """Render every panel the manuscript figure places, and return their names.

    Only the panels the figure uses. The statistics above cover all four group
    rows because the color limits pool over them, but the figure layout
    draws the pooled row across all four quantities, the three disease groups'
    thickness change, and the group contrasts.
    """
    import pyvista as pv

    panels = {}
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for bone in bones:
            mesh = pv.read(str(reference_surface_path(bone)))
            view = VIEW[bone]
            out_dir = results_dir(create=True) / "panels" / pressure_type / bone
            out_dir.mkdir(parents=True, exist_ok=True)
            sources, targets, names = [], [], []

            def queue(name, scalars, clim, colormap):
                source = scratch / f"{bone}_{name}.png"
                render_surface(mesh, scalars, clim, colormap, source,
                               WINDOW_MAIN, view=view)
                sources.append(source)
                targets.append(out_dir / f"{name}.png")
                names.append(name)

            for condition in CONDITIONS:
                surfaces = statistics[bone][condition]
                pooled = surfaces["all"]

                masked = pooled["pressure"].copy()
                masked[pooled["pressure_n"] < MIN_KNEES_PER_VERTEX] = np.nan
                queue(f"{condition}__all__pressure", masked, limits["pressure"], "cividis")

                masked = pooled["slope"].copy()
                masked[pooled["slope_n"] < MIN_KNEES_PER_VERTEX] = np.nan
                queue(f"{condition}__all__slope", masked, limits["slope"], "RdBu_r")

                if pooled["spm"] is not None:
                    # Only the surviving clusters are colored; everything else
                    # is grey. A t map drawn everywhere invites reading colour
                    # where no cluster reached significance.
                    statistic = np.where(pooled["spm"]["significant"],
                                         pooled["spm"]["stat"], np.nan)
                    queue(f"{condition}__all__spm", statistic, limits["spm"], "RdBu_r")

                # Group contrasts in pressure, one pair of columns per contrast.
                for name, first, second in CONTRASTS:
                    if first not in surfaces or second not in surfaces:
                        continue
                    a, b = surfaces[first], surfaces[second]
                    valid = ((a["pressure_n"] >= MIN_KNEES_PER_VERTEX)
                             & (b["pressure_n"] >= MIN_KNEES_PER_VERTEX))
                    difference = np.where(valid, a["pressure"] - b["pressure"], np.nan)
                    queue(f"{condition}__contrast__{name}", difference,
                          contrast_limit, "RdBu_r")

            # Thickness change does not depend on gait, so it is drawn once,
            # from the reference condition, for the pooled row and each group.
            reference = statistics[bone][CONDITIONS[0]]
            for group in ("all",) + DISEASE_GROUPS:
                if group not in reference:
                    continue
                entry = reference[group]
                masked = entry["thinning"].copy()
                masked[entry["thinning_n"] < MIN_KNEES_PER_VERTEX] = np.nan
                queue(f"thinning__{group}", masked, limits["thinning"], "RdBu_r")

            crop_and_orient(sources, targets, bone)
            panels[bone] = names
            print(f"  {bone}: {len(names)} panels -> {out_dir}")
    return panels


def render_supplement_panels():
    """The thickness-change group panels: an omnibus surface and three contrasts.

    A data output; no figure script composes these panels.

    Its color limits come from the significant vertices only. Scaling to the
    whole surface would be dominated by the non-significant remainder and
    flatten the clusters that are the point of the figure.
    """
    import pyvista as pv

    statistic, mask, difference, sig = {}, {}, {}, {}
    missing = []
    for bone in BONES:
        path = (spm_dir() / "thinning" / "anova" / bone
                / f"anova_{bone}_thinning_results.vtk")
        if not path.is_file():
            missing.append(str(path))
            continue
        mesh = pv.read(str(path))
        statistic[bone] = np.asarray(mesh.point_data["stat_value"], float)
        mask[bone] = np.asarray(mesh.point_data["is_significant"], float) > 0.5
        for first, second in PAIRS:
            name = f"ttest_{bone}_thinning_{first}_vs_{second}"
            path = spm_dir() / "thinning" / "ttest" / bone / f"{name}_results.vtk"
            if not path.is_file():
                missing.append(str(path))
                continue
            mesh = pv.read(str(path))
            # mm/yr on disk, µm/yr on the figure.
            difference[(bone, first, second)] = 1e3 * np.asarray(
                mesh.point_data[f"group_mean_{first}_minus_{second}"], float)
            sig[(bone, first, second)] = np.asarray(
                mesh.point_data["is_significant"], float) > 0.5

    if missing:
        # Each panel is a separate array cell, so a partial array would render a
        # figure quietly missing panels.
        raise SystemExit(
            f"{len(missing)} of {len(BONES) * (1 + len(PAIRS))} supplement SPM "
            f"files are missing -- the array is incomplete. First:\n  "
            + "\n  ".join(missing[:4])
        )

    # With no significant vertex anywhere (a null result, e.g. a small test root),
    # fall back to the whole surface: the panels then show the null result instead
    # of the run stopping before layout.json, which the main figure also needs.
    if any(sig[k].any() for k in difference):
        pooled = np.concatenate([np.abs(difference[k][sig[k]]) for k in difference
                                 if sig[k].any()])
    else:
        pooled = np.concatenate([np.abs(difference[k]) for k in difference])
    difference_limit = (-float(np.percentile(pooled, 98)),
                        float(np.percentile(pooled, 98)))
    if any(mask[b].any() for b in BONES):
        f_pool = np.concatenate([statistic[b][mask[b]] for b in BONES if mask[b].any()])
    else:
        f_pool = np.concatenate([statistic[b] for b in BONES])
    f_limit = (0.0, float(np.percentile(f_pool, 99)))

    out_dir = results_dir(create=True) / "panels" / "supplement"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for bone in BONES:
            mesh = pv.read(str(reference_surface_path(bone)))
            sources, targets = [], []

            source = scratch / f"{bone}_omnibus.png"
            render_surface(mesh, statistic[bone], f_limit, "cividis", source,
                           WINDOW_SUPPLEMENT, outline=mask[bone], view=VIEW[bone])
            sources.append(source)
            targets.append(out_dir / f"{bone}__omnibus.png")

            for first, second in PAIRS:
                source = scratch / f"{bone}_{first}_{second}.png"
                render_surface(mesh, difference[(bone, first, second)],
                               difference_limit, "RdBu_r", source,
                               WINDOW_SUPPLEMENT,
                               outline=sig[(bone, first, second)], view=VIEW[bone])
                sources.append(source)
                targets.append(out_dir / f"{bone}__{first}_vs_{second}.png")

            crop_and_orient(sources, targets, bone)
    print(f"  supplement: {len(BONES) * (1 + len(PAIRS))} panels -> {out_dir}")

    return {
        "f_limit": list(f_limit),
        "difference_limit": list(difference_limit),
        "significant_vertices": {
            f"{bone}__omnibus": int(mask[bone].sum()) for bone in BONES
        } | {
            f"{bone}__{first}_vs_{second}": int(sig[(bone, first, second)].sum())
            for bone in BONES for first, second in PAIRS
        },
    }


def derive_maps(bones=BONES, pressures=PRESSURES) -> Path:
    out_dir = results_dir(create=True)
    group_map = load_group_map()
    print(f"Quality control: era {qc.ERA}, union policy, "
          f"table {qc.unified_csv().name}")

    layout = {
        "conditions": list(CONDITIONS),
        "condition_labels": CONDITION_LABEL,
        "group_rows": list(GROUP_ROWS),
        "disease_groups": list(DISEASE_GROUPS),
        "contrasts": [{"name": n, "first": a, "second": b} for n, a, b in CONTRASTS],
        "min_knees_per_vertex": MIN_KNEES_PER_VERTEX,
        "pressures": {},
    }
    population = {}

    for pressure_type in pressures:
        print(f"\n=== {pressure_type} pressure ===")
        statistics = {}
        for bone in bones:
            statistics[bone] = {}
            for condition in CONDITIONS:
                print(f"  {condition} / {bone}")
                surfaces, counts, kept = condition_statistics(
                    condition, bone, pressure_type, group_map)
                statistics[bone][condition] = surfaces
                population[f"{condition}__{bone}__{pressure_type}"] = counts
                qc.write_inclusion_manifest(
                    f"figure5_{condition}_{bone}_{pressure_type}",
                    [{"knee": knee, "group": group_map.get(knee, "unlabelled"),
                      "condition": condition} for knee in kept],
                )
                print(f"    matrix {counts['matrix_n']} - {counts['n_dropped']} "
                      f"dropped = {counts['n']}  {counts['groups']}")

        limits = pooled_limits(statistics)
        contrast_limit = pooled_contrast_limit(statistics)
        print(f"  pooled limits: { {k: [round(v, 4) for v in lim] for k, lim in limits.items()} }")
        print(f"  contrast limit: {[round(v, 4) for v in contrast_limit]}")

        panels = render_main_panels(statistics, limits, contrast_limit,
                                    pressure_type, bones)
        layout["pressures"][pressure_type] = {
            "limits": {k: list(v) for k, v in limits.items()},
            "contrast_limit": list(contrast_limit),
            "panels": panels,
        }

    layout["supplement"] = render_supplement_panels()
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=2))
    (out_dir / "population.json").write_text(json.dumps(population, indent=2))
    write_cluster_tables()
    write_region_composition()
    print(f"\nwrote layout.json, population.json")
    return out_dir


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    spm = stages.add_parser(
        "spm", help="permutation inference over the surface (expensive; one cell "
                    "per invocation, for a job array)")
    spm.add_argument("--cell", type=int,
                     help="cell index; see --list-cells")
    spm.add_argument("--all", action="store_true",
                     help="run every cell in this process (hours; prefer an array)")
    spm.add_argument("--list-cells", action="store_true",
                     help="print the cell plan and exit")
    spm.add_argument("--missing", choices=MISSING_POLICIES, default=MISSING_POLICIES[0],
                     help="how a knee with no value at a vertex enters that vertex's "
                          "correlation: 'zero' (default; the published analysis, "
                          "writes spm/) or 'complete-case' (opt-in sensitivity run of "
                          "cells 0-31, writes spm_complete_case/)")

    maps = stages.add_parser(
        "maps", help="quality control, group surfaces, renders and cluster tables")
    maps.add_argument("--bone", nargs="+", default=list(BONES), choices=list(BONES))
    maps.add_argument("--pressure", nargs="+", default=list(PRESSURES),
                      choices=list(PRESSURES))

    args = parser.parse_args()

    if args.stage == "spm":
        cells = spm_cells()
        if args.list_cells:
            for i, cell in enumerate(cells):
                extra = (f"{cell['condition']} " if cell["kind"] == "correlation" else "")
                print(f"{i:>3}  {cell['kind']:<12} {extra}{cell['name']}")
            print(f"\n{len(cells)} cells. Run as an array:\n"
                  f"    --array=0-{len(cells) - 1}  ->  "
                  f"python pipeline/derive_figure5_data.py spm "
                  f"--cell $SLURM_ARRAY_TASK_ID")
            return
        if args.all:
            if args.missing != "zero":
                print(f"--missing {args.missing}: running the correlation cells "
                      "only; the supplement cells do not take this option.")
                cells = [c for c in cells if c["kind"] == "correlation"]
            run_spm(cells, args.missing)
        elif args.cell is not None:
            if not 0 <= args.cell < len(cells):
                raise SystemExit(f"--cell must be in 0..{len(cells) - 1}")
            run_spm([cells[args.cell]], args.missing)
        else:
            raise SystemExit("pass --cell N, --all, or --list-cells")
        return

    out_dir = derive_maps(bones=args.bone, pressures=args.pressure)
    print(f"\nFigure 3 inputs in {out_dir}")
    print("Next: figures/figure5_spatial_colocalization.py")


if __name__ == "__main__":
    main()
