#!/usr/bin/env python3
"""Derive Figure 8's and Extended Data Figure 3's inputs: 03_derived ->
04_results/figure8_pressure_roi_biomarker.

Figure 8 (femur) and Extended Data Figure 3 (tibia) ask whether a region of interest chosen by *simulated contact pressure*
is a better longitudinal biomarker of cartilage thinning than the standard
anatomical region trials use. This step builds that region for every knee
and computes the two things the figure reports.

The region
----------
Each knee is simulated under ten gait conditions and the thinning outcome is the
same in all of them, so a per-condition region would be ten answers to one
question. Instead each condition proposes a region -- the high-contact part of
the medial compartment -- and a vertex joins the knee's consensus region if at
least half of that knee's conditions proposed it:

    per condition:  anatomical medial  and  contacted  and  top 90% by cumulative pressure
    consensus:      in at least half of this knee's conditions

Top 90%, not a tighter focal threshold, and fixed rather than tuned per group. It
keeps essentially the whole contacted medial region, dropping only the
lowest-pressure fringe, and a large region is a reproducible one. Choosing the
threshold per group would be selecting on the outcome.

The two measures, which are not the same thing
----------------------------------------------
responsiveness
    Standardized response mean, mean(delta) / SD(delta) over knees. How reliably
    a region detects thinning within a group.
discrimination
    Cohen's d of thinning, disease against healthy. Whether that thinning
    separates the groups.

They come apart here, which is why both are computed. Healthy knees thin a little
but consistently, giving a respectable SRM that makes early disease look no more
responsive; the groups only separate once the disease groups accelerate. A
biomarker for a trial needs the second, so reporting only the first would flatter
the region.

Each is computed for the pressure region and, on the same knees, for the standard
comparator -- paired, so the difference is tested within knee rather than between
two independently estimated numbers. The femur comparator is the curated
weight-bearing sub-region trials use; the tibia has no weight-bearing sub-label,
so it is the whole medial plateau, and the pressure region is a strict subset of
it. That asymmetry is anatomical, not a choice, and it is why the tibia's numbers
are not directly comparable to the femur's.

What it produces, per bone
--------------------------
``consensus_roi_{bone}.npz``      per-knee consensus frequency maps and their mean
``gait_averaged_srm_{bone}.csv``  responsiveness by group and interval, both
                                  regions, with paired difference and bootstrap p
``responsiveness_{bone}.csv``     per group, region and interval: mean change in
                                  micrometres, SD, SRM and its interval
``discrimination_{bone}.csv``     Cohen's d disease-against-healthy, both regions,
                                  and the paired difference between them
``paired_srm_{bone}.csv``         paired SRM difference per group, for the figure's
                                  significance annotations
``meta_{bone}.json``              the region parameters and the population, so the
                                  figure reads the analysis choices instead of
                                  restating them

Population -- and why it is not the cohort size
-----------------------------------------------
Every other pressure figure is bounded by the combined per-vertex matrices. This
one never reads a matrix: it reads transferred pressure and the per-visit
thickness tree directly, so it is bounded by *follow-up MRI availability*. A knee
needs a baseline and the relevant follow-up visit to contribute an interval at
all, and the two intervals therefore have different populations. Quoting one
number for this figure is wrong; the counts are written into ``meta_{bone}.json``.

Quality control
---------------
Exclusions come from the shared table (``figures/shared/qc.py``), never a
threshold invented here. Two levels, and the distinction matters: a knee that
solves badly under one condition is dropped *for that condition* and keeps its
other nine, while a knee flagged under most conditions is dropped entirely. Only
the first stage can express the per-condition drop, since it is the one that
reads pressure; the second stage inherits its knee list verbatim, so the two
cannot disagree about who was included.

The thickness side has its own, much smaller table:
``03_derived/qc/thickness_visit_exclusions.csv`` (``qc.load_thickness_visit_exclusions``)
lists single (knee, visit, bone) fits that failed -- the knee's fitted size jumps
at that one visit. An excluded visit is simply absent, so the knee keeps its other
intervals; the 48-month one is skipped. It lists two mild-OA knees' V06 fits.

Usage
-----
    python pipeline/derive_figure8_data.py
    python pipeline/derive_figure8_data.py --bone femur
    python pipeline/derive_figure8_data.py --keep-outliers   # sensitivity only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
# The quality-control table is shared with the figures rather than duplicated
# here. The filtering happens at derivation time, so the derivation is what must
# import it -- a second copy of the policy is how two halves of one analysis end
# up disagreeing about which simulations exist.
sys.path.insert(0, str(_REPO / "figures"))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402
from shared import qc  # noqa: E402

FIGURE = "figure8_pressure_roi_biomarker"
BONES = ("femur", "tibia")

# Cumulative pressure over stance: the region is defined on total dose, not on an
# instantaneous peak, because thinning accumulates.
PRESSURE_FIELD = "cumulative_tf_pressure"

MEDIAL = 0  # coarse region code

# Standard comparator per bone. Femur: the curated weight-bearing sub-region
# (fine label 11). Tibia: the whole medial plateau (fine label 8) -- no
# weight-bearing sub-label exists for the tibia.
STATIC_FINE_LABEL = {"femur": 11, "tibia": 8}
STATIC_REGION_NAME = {"femur": "Standard Medial Femur", "tibia": "Standard Medial Tibia"}

BASELINE = "V00"
INTERVALS = (("24mo", "V03"), ("48mo", "V06"))

# Region definition. Fixed, not swept: see the module docstring.
QUANTILE = 0.90
CONSENSUS_FRACTION = 0.5
MIN_GAITS = 8
MIN_ROI_VERTICES = 30

N_BOOTSTRAP = 5000
SEED = 12345

# Groups, in the order they are drawn. Keys index stored data and are left alone;
# what a reader sees comes from figures/shared/palette.py.
GROUPS = {"healthy": "kl0_1_healthy", "progressor": "kl0_1_progressor", "kl2": "kl2"}

# Group subsets reported in the responsiveness table. 'kl0_1' pools the two
# groups that start at the same radiographic grade and differ only in outcome.
GROUP_SUBSETS = {
    "all": None,
    "kl0_1": {"kl0_1_healthy", "kl0_1_progressor"},
    "healthy": {"kl0_1_healthy"},
    "progressor": {"kl0_1_progressor"},
    "kl2": {"kl2"},
}


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def thickness_root() -> Path:
    """The per-visit thickness tree this figure measures its outcome on.

    ``thickness_merged``, not ``thickness``. The merged tree holds the baseline
    visit fitted to the same segmentation as the follow-up visits, together with
    those follow-up fits. The outcome here is follow-up minus baseline, so a
    baseline fit on a different segmentation would bias every knee's change in
    the term that matters most.
    """
    return stage_dir("derived", "thickness_merged", "thickness_on_reference")


def thickness_path(knee: str, visit: str, bone: str) -> Path:
    return thickness_root() / visit / f"{knee}_{bone}_thickness_ref.npy"


def region_labels_path(bone: str) -> Path:
    return stage_dir("derived", "region_labels") / f"{bone}_region_labels.npz"


def groups_by_knee() -> dict[str, str]:
    table = json.loads((stage_dir("inputs") / "groups.json").read_text())
    return {knee: group for group, knees in table.items() for knee in knees}


def require_inputs(bone: str) -> dict[str, int]:
    """Resolve every input tree and assert none of them is empty."""
    counts = cohorts.check_pressure_trees(bone)
    root = thickness_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Merged thickness tree not found: {root}\n\n"
            "Build it with:\n"
            "    python pipeline/derive_thickness.py\n\n"
            "Or download the published 03_derived stage."
        )
    for visit in (BASELINE,) + tuple(v for _, v in INTERVALS):
        n = len(list((root / visit).glob(f"*_{bone}_thickness_ref.npy")))
        if n == 0:
            raise RuntimeError(f"visit {visit} holds zero {bone} thickness files under {root}")
        counts[f"thickness_{visit}"] = n
    labels = region_labels_path(bone)
    if not labels.is_file():
        raise FileNotFoundError(
            f"Region labels not found: {labels}\n\n"
            "Build them with:\n"
            "    python pipeline/combine_reference_data.py labels\n\n"
            "Or download the published 03_derived stage."
        )
    return counts


_VISIT_EXCLUSIONS: set[tuple[str, str, str]] | None = None


def visit_exclusions() -> set[tuple[str, str, str]]:
    global _VISIT_EXCLUSIONS
    if _VISIT_EXCLUSIONS is None:
        _VISIT_EXCLUSIONS = qc.load_thickness_visit_exclusions()
    return _VISIT_EXCLUSIONS


def load_thickness(knee: str, bone: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{visit: (thickness, valid)}``. A vertex is valid where thickness > 0.

    Visits in the thickness exclusion table are left out (see Quality control).
    """
    out = {}
    for visit in (BASELINE,) + tuple(v for _, v in INTERVALS):
        if (knee, visit, bone) in visit_exclusions():
            continue
        path = thickness_path(knee, visit, bone)
        if path.is_file():
            values = np.load(path).astype(np.float64)
            out[visit] = (values, values > 0)
    return out


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def region_change(mask, thickness, baseline, follow_up):
    """Mean thickness change over ``mask``, on vertices valid at both visits."""
    if baseline not in thickness or follow_up not in thickness or mask.sum() == 0:
        return np.nan
    t0, valid0 = thickness[baseline]
    t1, valid1 = thickness[follow_up]
    selected = mask & valid0 & valid1
    if selected.sum() == 0:
        return np.nan
    return float(np.mean(t1[selected] - t0[selected]))


def srm(values) -> float:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) < 3 or values.std(ddof=1) == 0:
        return np.nan
    return values.mean() / values.std(ddof=1)


def srm_with_interval(values, rng, b=N_BOOTSTRAP):
    """SRM and a knee-resampling 95% interval.

    The resampling unit is the knee, which is what the SRM is a statistic over.
    """
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 3 or values.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan, n
    point = values.mean() / values.std(ddof=1)
    sample = values[rng.integers(0, n, size=(b, n))]
    sd = sample.std(axis=1, ddof=1)
    sd[sd == 0] = np.nan
    draws = sample.mean(axis=1) / sd
    return point, np.nanpercentile(draws, 2.5), np.nanpercentile(draws, 97.5), n


def paired_srm_difference(a, b_values, rng, b=N_BOOTSTRAP):
    """SRM(a) - SRM(b) on the same knees, with interval and two-sided p.

    Paired on purpose: the two regions are measured on one knee, so resampling
    knees jointly removes the between-knee variance that both share.
    """
    a = np.asarray(a, float)
    b_values = np.asarray(b_values, float)
    keep = ~np.isnan(a) & ~np.isnan(b_values)
    a, b_values = a[keep], b_values[keep]
    n = len(a)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, n
    point = a.mean() / a.std(ddof=1) - b_values.mean() / b_values.std(ddof=1)
    index = rng.integers(0, n, size=(b, n))
    sa, sb = a[index], b_values[index]
    sda, sdb = sa.std(axis=1, ddof=1), sb.std(axis=1, ddof=1)
    sda[sda == 0] = np.nan
    sdb[sdb == 0] = np.nan
    draws = sa.mean(axis=1) / sda - sb.mean(axis=1) / sdb
    draws = draws[~np.isnan(draws)]
    p = min(2.0 * min((draws <= 0).mean(), (draws >= 0).mean()), 1.0)
    return point, np.percentile(draws, 2.5), np.percentile(draws, 97.5), p, n


def cohens_d(a, b_values) -> float:
    a = np.asarray(a, float)
    b_values = np.asarray(b_values, float)
    a = a[~np.isnan(a)]
    b_values = b_values[~np.isnan(b_values)]
    if len(a) < 2 or len(b_values) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b_values) - 1) * b_values.var(ddof=1))
        / (len(a) + len(b_values) - 2)
    )
    return (a.mean() - b_values.mean()) / pooled if pooled > 0 else np.nan


# --------------------------------------------------------------------------- #
# Stage 1: the consensus region, and gait-averaged responsiveness              #
# --------------------------------------------------------------------------- #
def build_consensus_regions(bone, outlier_set, chronic, tag_of, verbose=True):
    """Per-knee consensus region and its thickness change, both regions.

    Returns ``(knees, frequency maps, gaits used, per-knee changes, counts)``.
    """
    import pyvista as pv

    labels = np.load(region_labels_path(bone), allow_pickle=True)
    anatomical_medial = labels["coarse_code"] == MEDIAL
    static = labels["fine_labels"] == STATIC_FINE_LABEL[bone]
    n_vertices = anatomical_medial.shape[0]

    group_of = groups_by_knee()
    candidates = [
        knee for knee in sorted(group_of)
        if thickness_path(knee, BASELINE, bone).is_file()
    ]

    knees, frequency_maps, gaits_used = [], [], []
    roi_rows, static_rows = [], []
    dropped_chronic = dropped_few_gaits = dropped_small_roi = 0
    dropped_simulations = 0

    for knee in candidates:
        if knee in chronic:
            dropped_chronic += 1
            continue

        per_gait = []
        for gait in cohorts.ALL:
            if (tag_of.get(gait), knee) in outlier_set:
                # This knee's solve under this condition is an artifact. It keeps
                # its other conditions.
                dropped_simulations += 1
                continue
            path = cohorts.pressure_dir(gait, bone) / f"{knee}_{bone}_pressure_ref.vtk"
            if not path.is_file():
                continue
            cumulative = pv.read(path).point_data.get(PRESSURE_FIELD)
            if cumulative is None:
                continue
            cumulative = np.asarray(cumulative)
            contacted = anatomical_medial & (cumulative > 0)
            if contacted.sum() < MIN_ROI_VERTICES:
                continue
            threshold = np.quantile(cumulative[contacted], 1.0 - QUANTILE)
            region = np.zeros(n_vertices, dtype=bool)
            region[contacted] = cumulative[contacted] >= threshold
            per_gait.append(region)

        if len(per_gait) < MIN_GAITS:
            dropped_few_gaits += 1
            continue
        frequency = np.mean(per_gait, axis=0)
        consensus = frequency >= CONSENSUS_FRACTION
        if consensus.sum() < MIN_ROI_VERTICES:
            dropped_small_roi += 1
            continue

        thickness = load_thickness(knee, bone)
        if BASELINE not in thickness:
            continue

        knees.append(knee)
        frequency_maps.append(frequency.astype(np.float32))
        gaits_used.append(len(per_gait))
        roi_rows.append(
            dict(knee=knee, group=group_of[knee], roi_size=int(consensus.sum()),
                 **{name: region_change(consensus, thickness, BASELINE, visit)
                    for name, visit in INTERVALS})
        )
        static_rows.append(
            dict(knee=knee, group=group_of[knee],
                 **{name: region_change(static, thickness, BASELINE, visit)
                    for name, visit in INTERVALS})
        )

    counts = dict(
        candidates=len(candidates), included=len(knees),
        dropped_chronic=dropped_chronic, dropped_few_gaits=dropped_few_gaits,
        dropped_small_roi=dropped_small_roi,
        dropped_simulations_qc=dropped_simulations,
        gaits_min=int(min(gaits_used)) if gaits_used else 0,
        gaits_median=int(np.median(gaits_used)) if gaits_used else 0,
        gaits_max=int(max(gaits_used)) if gaits_used else 0,
        roi_vertices_mean=float(np.mean([r["roi_size"] for r in roi_rows])) if roi_rows else np.nan,
        roi_vertices_sd=float(np.std([r["roi_size"] for r in roi_rows], ddof=1)) if len(roi_rows) > 1 else np.nan,
        anatomical_medial_vertices=int(anatomical_medial.sum()),
        static_vertices=int(static.sum()),
    )
    if verbose:
        print(f"[{bone}] {len(knees)} of {len(candidates)} knees carry a consensus region "
              f"({dropped_chronic} chronic, {dropped_few_gaits} with fewer than "
              f"{MIN_GAITS} conditions, {dropped_small_roi} with too small a region)")
        print(f"[{bone}] {dropped_simulations} individual (condition, knee) simulations "
              f"excluded; conditions per retained knee: min {counts['gaits_min']} "
              f"median {counts['gaits_median']} max {counts['gaits_max']}")
        print(f"[{bone}] consensus region {counts['roi_vertices_mean']:.0f} "
              f"+/- {counts['roi_vertices_sd']:.0f} vertices "
              f"(anatomical medial {counts['anatomical_medial_vertices']}, "
              f"{STATIC_REGION_NAME[bone].lower()} {counts['static_vertices']})")
    if not knees:
        # Every statistic downstream needs at least one knee per group; without
        # this the run dies later inside mannwhitneyu with "x and y must be of
        # nonzero size", which does not say why.
        raise RuntimeError(
            f"[{bone}] no knee carries a consensus region: {len(candidates)} candidates, "
            f"{dropped_chronic} chronic, {dropped_few_gaits} with fewer than {MIN_GAITS} "
            f"conditions, {dropped_small_roi} with too small a region. A root with fewer "
            f"than {MIN_GAITS} simulated conditions admits no knee."
        )
    return knees, frequency_maps, gaits_used, roi_rows, static_rows, counts, static


def gait_averaged_srm(roi_rows, static_rows):
    """Responsiveness by group and interval, both regions, paired.

    A fresh generator seeded here rather than one shared across the whole module:
    the intervals are part of the reported result, so the draws that produce
    them have to be reproducible independently of what else the run does.
    """
    rng = np.random.default_rng(SEED)
    out = []
    for subset, keep in GROUP_SUBSETS.items():
        roi = [r for r in roi_rows if keep is None or r["group"] in keep]
        static = [r for r in static_rows if keep is None or r["group"] in keep]
        for interval, _ in INTERVALS:
            roi_values = [r[interval] for r in roi]
            static_values = [r[interval] for r in static]
            s_roi, roi_lo, roi_hi, n = srm_with_interval(roi_values, rng)
            s_static, static_lo, static_hi, _ = srm_with_interval(static_values, rng)

            by_knee = {r["knee"]: r[interval] for r in static}
            paired_roi = [r[interval] for r in roi if r["knee"] in by_knee]
            paired_static = [by_knee[r["knee"]] for r in roi if r["knee"] in by_knee]
            delta, delta_lo, delta_hi, p, n_paired = paired_srm_difference(
                paired_roi, paired_static, rng)

            out.append(dict(
                group=subset, interval=interval, n=n,
                roi_srm=s_roi, roi_ci_lo=roi_lo, roi_ci_hi=roi_hi,
                static_srm=s_static, static_ci_lo=static_lo, static_ci_hi=static_hi,
                delta_srm=delta, delta_ci_lo=delta_lo, delta_ci_hi=delta_hi,
                boot_p=p, n_paired=n_paired,
            ))
    return out


# --------------------------------------------------------------------------- #
# Stage 2: responsiveness and discrimination, in micrometres                  #
# --------------------------------------------------------------------------- #
def per_knee_changes(knees, frequency_maps, bone, static):
    """``{interval: {'roi': {knee: um}, 'static': {...}, 'group': {...}}}``.

    Recomputed from the consensus maps rather than reusing stage 1's millimetre
    values, so this stage owns the unit conversion and the two stages can be read
    independently.
    """
    group_of = groups_by_knee()
    record = {name: {"roi": {}, "static": {}, "group": {}} for name, _ in INTERVALS}
    for knee, frequency in zip(knees, frequency_maps):
        consensus = frequency >= CONSENSUS_FRACTION
        thickness = load_thickness(knee, bone)
        if BASELINE not in thickness:
            continue
        for interval, visit in INTERVALS:
            if visit not in thickness:
                continue
            for name, mask in (("roi", consensus), ("static", static)):
                change = region_change(mask, thickness, BASELINE, visit)
                record[interval][name][knee] = change * 1000.0 if not np.isnan(change) else np.nan
            record[interval]["group"][knee] = group_of.get(knee)
    return record


def responsiveness_table(record, rng):
    rows = []
    for interval, _ in INTERVALS:
        for key, group in GROUPS.items():
            for region in ("roi", "static"):
                values = np.array([
                    record[interval][region][knee]
                    for knee in record[interval][region]
                    if record[interval]["group"][knee] == group
                ])
                values = values[~np.isnan(values)]
                _, lo, hi, _ = srm_with_interval(values, rng)
                rows.append(dict(
                    interval=interval, region=region, group=key,
                    srm=srm(values), srm_ci_lo=lo, srm_ci_hi=hi, n=len(values),
                    mean_um=values.mean() if len(values) else np.nan,
                    sd_um=values.std(ddof=1) if len(values) > 1 else np.nan,
                ))
    return rows


def discrimination_table(record, rng):
    """Cohen's d disease-against-healthy, both regions, and their difference.

    The two groups are resampled independently -- they are different knees -- but
    the *same* resampled knees supply both regions' d, which is what makes the
    difference a paired quantity rather than a comparison of two separate fits.
    """
    from scipy.stats import mannwhitneyu

    rows = []
    for interval, _ in INTERVALS:
        healthy = [k for k in record[interval]["roi"]
                   if record[interval]["group"][k] == GROUPS["healthy"]]
        for key in ("progressor", "kl2"):
            diseased = [k for k in record[interval]["roi"]
                        if record[interval]["group"][k] == GROUPS[key]]
            disease_roi = np.array([record[interval]["roi"][k] for k in diseased])
            disease_static = np.array([record[interval]["static"][k] for k in diseased])
            healthy_roi = np.array([record[interval]["roi"][k] for k in healthy])
            healthy_static = np.array([record[interval]["static"][k] for k in healthy])

            d_roi = cohens_d(disease_roi, healthy_roi)
            d_static = cohens_d(disease_static, healthy_static)

            n_disease, n_healthy = len(diseased), len(healthy)
            roi_draws, static_draws, difference_draws = [], [], []
            for _ in range(N_BOOTSTRAP):
                i = rng.integers(0, n_disease, n_disease)
                j = rng.integers(0, n_healthy, n_healthy)
                a = cohens_d(disease_roi[i], healthy_roi[j])
                b = cohens_d(disease_static[i], healthy_static[j])
                roi_draws.append(a)
                static_draws.append(b)
                difference_draws.append(a - b)
            roi_draws = np.array(roi_draws)
            static_draws = np.array(static_draws)
            difference_draws = np.array(difference_draws)

            rows.append(dict(
                interval=interval, contrast=f"{key}_vs_healthy",
                n_disease=n_disease, n_healthy=n_healthy,
                d_roi=d_roi,
                d_roi_ci_lo=np.nanpercentile(roi_draws, 2.5),
                d_roi_ci_hi=np.nanpercentile(roi_draws, 97.5),
                p_roi=mannwhitneyu(disease_roi[~np.isnan(disease_roi)],
                                   healthy_roi[~np.isnan(healthy_roi)]).pvalue,
                d_static=d_static,
                d_static_ci_lo=np.nanpercentile(static_draws, 2.5),
                d_static_ci_hi=np.nanpercentile(static_draws, 97.5),
                p_static=mannwhitneyu(disease_static[~np.isnan(disease_static)],
                                      healthy_static[~np.isnan(healthy_static)]).pvalue,
                d_difference=d_roi - d_static,
                d_difference_ci_lo=np.nanpercentile(difference_draws, 2.5),
                d_difference_ci_hi=np.nanpercentile(difference_draws, 97.5),
                p_difference=min(2 * min((difference_draws <= 0).mean(),
                                         (difference_draws >= 0).mean()), 1.0),
            ))
    return rows


def paired_srm_table(record, rng):
    """Paired SRM difference per group and interval -- the figure's annotations."""
    rows = []
    for interval, _ in INTERVALS:
        for key, group in GROUPS.items():
            knees = [k for k in record[interval]["roi"]
                     if record[interval]["group"][k] == group]
            delta, lo, hi, p, _ = paired_srm_difference(
                [record[interval]["roi"][k] for k in knees],
                [record[interval]["static"][k] for k in knees],
                rng,
            )
            rows.append(dict(group=key, interval=interval, delta_srm=delta,
                             delta_ci_lo=lo, delta_ci_hi=hi, p_difference=p))
    return rows


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def derive(bone: str, keep_outliers=False) -> Path:
    out_dir = stage_dir("results", FIGURE, create=True)
    input_counts = require_inputs(bone)

    if keep_outliers:
        outlier_set, chronic, tag_of = set(), set(), {}
        print("[qc] --keep-outliers: no exclusions applied (sensitivity run only)")
    else:
        # Resolved against the table's own keys rather than assumed, so a
        # rename there cannot select nothing without an error.
        tags = cohorts.resolve_qc_tags(qc.known_cohort_tags())
        tag_of = dict(zip(cohorts.ALL, tags))
        outlier_set = qc.load_outlier_set(cohort_tags=tags, source="union")
        chronic = qc.load_chronic_knees()
        print(f"[qc] era {qc.ERA}, union policy: {len(outlier_set)} per-simulation "
              f"exclusions over {len(cohorts.ALL)} conditions; "
              f"{len(chronic)} knees dropped entirely: {sorted(chronic)}")

    knees, frequency_maps, gaits_used, roi_rows, static_rows, counts, static = (
        build_consensus_regions(bone, outlier_set, chronic, tag_of))

    np.savez_compressed(
        out_dir / f"consensus_roi_{bone}.npz",
        knees=np.array(knees),
        frequency=np.array(frequency_maps),
        mean_frequency=np.mean(frequency_maps, axis=0),
        n_gaits=np.array(gaits_used),
    )

    write_csv(out_dir / f"gait_averaged_srm_{bone}.csv",
              gait_averaged_srm(roi_rows, static_rows))

    # A second generator, seeded identically and consumed by this stage alone.
    # Stage 2's intervals are reported numbers too, so they must not depend on
    # how many draws stage 1 happened to take.
    rng = np.random.default_rng(SEED)
    record = per_knee_changes(knees, frequency_maps, bone, static)
    write_csv(out_dir / f"responsiveness_{bone}.csv", responsiveness_table(record, rng))
    write_csv(out_dir / f"discrimination_{bone}.csv", discrimination_table(record, rng))
    write_csv(out_dir / f"paired_srm_{bone}.csv", paired_srm_table(record, rng))

    per_interval = {
        interval: sum(1 for k in record[interval]["roi"]
                      if not np.isnan(record[interval]["roi"][k]))
        for interval, _ in INTERVALS
    }
    per_interval_group = {
        interval: {
            key: sum(1 for k in record[interval]["roi"]
                     if record[interval]["group"][k] == group
                     and not np.isnan(record[interval]["roi"][k]))
            for key, group in GROUPS.items()
        }
        for interval, _ in INTERVALS
    }

    meta = dict(
        bone=bone,
        static_region=STATIC_REGION_NAME[bone],
        static_fine_label=STATIC_FINE_LABEL[bone],
        quantile=QUANTILE, consensus_fraction=CONSENSUS_FRACTION,
        min_gaits=MIN_GAITS, n_gait_conditions=len(cohorts.ALL),
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
        qc_era=qc.ERA, qc_policy="union", keep_outliers=bool(keep_outliers),
        qc_tags=tag_of,
        thickness_visit_exclusions=sorted(
            "/".join(row) for row in visit_exclusions() if row[2] == bone),
        intervals={name: visit for name, visit in INTERVALS},
        n_knees_with_region=len(knees),
        n_by_interval=per_interval,
        n_by_interval_group=per_interval_group,
        input_counts=input_counts,
        **counts,
    )
    (out_dir / f"meta_{bone}.json").write_text(json.dumps(meta, indent=2, default=str))

    # One row per (knee, interval): the intervals have different populations,
    # and the figure quotes both.
    qc.write_inclusion_manifest(
        f"{FIGURE}_{bone}",
        [dict(cohort_tag=f"gait_averaged_10:{interval}", knee=knee,
              group=record[interval]["group"][knee],
              change_um=record[interval]["roi"][knee])
         for interval, _ in INTERVALS for knee in record[interval]["roi"]
         if not np.isnan(record[interval]["roi"][knee])],
        policy=f"union/{qc.ERA}; per-simulation plus chronic; "
               f"region top {QUANTILE:.0%} in >= {CONSENSUS_FRACTION:.0%} of >= "
               f"{MIN_GAITS} conditions",
    )
    print(f"[{bone}] knees with a region {len(knees)}; per interval {per_interval}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bone", nargs="+", default=list(BONES), choices=list(BONES))
    parser.add_argument(
        "--keep-outliers", action="store_true",
        help="sensitivity only: skip the shared quality-control exclusions",
    )
    args = parser.parse_args()

    for bone in args.bone:
        print(f"\n=== {bone} ===")
        out_dir = derive(bone, keep_outliers=args.keep_outliers)
    print(f"\nFigure 8 inputs in {out_dir}")
    print("Next: figures/figure8_pressure_roi_biomarker.py")


if __name__ == "__main__":
    main()
