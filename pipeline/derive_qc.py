"""Stage 03, quality control: which simulations' contact pressure is an artifact.

A small fraction of simulations produce contact pressure that is a numerical
artifact rather than mechanics -- almost always a frame where the contact patch
collapses to a few triangles, so force over area explodes into hundreds of
megapascals. The blow-ups are specific to one gait condition (a knee that spikes
under one gait is fine under the others), so they are flagged per simulation,
not per knee. Every analysis that reads contact pressure selects its exclusions
from the one table this builds, through ``figures/shared/qc.py``.

Two detectors look at the same simulations in two representations, and a
third step reconciles them:

``spatial``
    Over the per-vertex pressure summaries in the combined matrices. Flags a
    simulation if any tibiofemoral compartment's peak vertex pressure (95th
    percentile of the per-vertex maximum over stance) exceeds 15 MPa, or if its
    medial cumulative pressure lies more than 5 robust standard deviations
    (1.4826 x MAD) above that condition's median. (CPU, minutes)

``timeseries``
    Over the per-triangle, per-frame contact output in ``joint_mechanics.h5``,
    which keeps the single bad frame a temporal summary can dilute. Flags a
    simulation if any compartment's peak p95 pressure exceeds 20 MPa, its peak
    mean pressure 15 MPa, or its peak contact force 5 kN, or if its medial
    pressure-time integral lies more than 6 robust standard deviations above
    the condition's median. The single-triangle maximum is recorded but never
    excludes -- one bad contact edge is not a failed solve. ``--mode extract``
    reads the h5 files (the slow part, about 75 minutes for the cohort,
    shardable by knee); ``--mode flag`` applies the rule to the extracted
    metrics; ``--mode merge`` joins shards. (CPU)

``unify``
    One row per simulation with both verdicts, their union, their
    intersection, and a fused robust-z score. The policy is the union: bad in
    either representation is excluded everywhere, because the two detectors
    agree only moderately and the intersection lets obvious artifacts through.

Inputs: the combined matrices (``03_derived/combined``), the simulations'
``joint_mechanics.h5`` and the region labels. Outputs, in ``03_derived/qc/``:
``pressure_outliers_<era>.csv`` (spatial), ``pressure_outliers_timeseries_<era>.csv``
and ``pressure_outliers_unified_<era>.csv``, the table analyses read.

The thresholds are absolute (15/20/15 MPa, 5 kN) and are not scale-invariant,
so a change to body scaling changes which simulations blow up. Run this again
after any new simulation rather than retuning it: the rule is calibrated to sit
in the gap between the real distribution and the catastrophic tail.

Compartments are anatomical. Each contact triangle is labelled medial, lateral
or trochlea by carrying its centroid into the shared reference space through
the pressure coregistration's own transform and taking the label of the
nearest reference vertex.

Eras
----
A table belongs to one era and is never merged with another: "chronic" means
flagged under at least five distinct conditions, so a table holding two eras'
keys would double every knee's count. The era is the data root's run: on a root
that declares one (``_run.json``) it is that run's name, ``--era`` may be
omitted, and rows are keyed ``<condition>_<run>`` (``cohorts.QC_TAG``). The v2
root declares none; there ``--era v2`` is required, and rebuilds the v2 table
keyed by the run names it used. Inputs come from whatever the
data root holds -- the era names the output, it does not select the simulations.

Running it
----------
    python pipeline/derive_qc.py spatial                             # era = the root's run
    python pipeline/derive_qc.py timeseries                          # extract + flag
    python pipeline/derive_qc.py timeseries --mode extract --n-shards 8 --shard 0
    python pipeline/derive_qc.py timeseries --mode merge --n-shards 8
    python pipeline/derive_qc.py timeseries --mode flag
    python pipeline/derive_qc.py unify
    python pipeline/derive_qc.py spatial --era v2                    # the v2 root (declares no run)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import Config, Knee, load_config  # noqa: E402
from comak_knee_oa.paths import declaration, run_name, stage_dir  # noqa: E402
from derive_pressure import (  # noqa: E402
    articular_surface_path,
    condition_name,
    femur_reference_center,
    transform_file,
)

# The one era a root that declares no run can hold: the v2 table.
LEGACY_ERA = "v2"

MEDIAL, LATERAL, TROCHLEA = 0, 1, 2

# Spatial rule.
PEAK_CEILING_MPA = 15.0
SPATIAL_MAD_K = 5.0
MIN_COMPARTMENT_VERTICES = 5

# Time-series rule. The ceilings sit in the gap between the real distribution
# (peak mean p99 ~7 MPa, peak p95 p99 ~13 MPa) and the catastrophic tail
# (p99.9 = 256-576 MPa, failed solves).
P95_CEIL, MEAN_CEIL, FORCE_CEIL = 20.0, 15.0, 5000.0
MAXSPIKE_INFO = 100.0  # recorded, never an exclusion
TS_MAD_K = 6.0
TS_COMPARTMENTS = (
    ("femur", MEDIAL, "femur_med"),
    ("femur", LATERAL, "femur_lat"),
    ("tibia", MEDIAL, "tibia_med"),
    ("tibia", LATERAL, "tibia_lat"),
)
TS_METRICS = ("peakmean", "peakp95", "peakmax", "peakforce", "pti")
N_GRID = 101

# Unify: the joint score's threshold, on the Euclidean fusion of each source's
# maximum one-sided robust z. Single-triangle maxima are left out of it, as
# they are left out of the time-series rule.
SPATIAL_SCORE_METRICS = (
    "femur_med_cum", "femur_med_peak", "femur_lat_peak",
    "tibia_med_cum", "tibia_med_peak", "tibia_lat_peak",
)
TS_SCORE_METRICS = tuple(
    f"{c}_{m}" for c in ("femur_med", "femur_lat", "tibia_med", "tibia_lat")
    for m in ("peakmean", "peakp95", "peakforce", "pti")
)
JOINT_THRESH = 6.0
CHRONIC_MIN_CONDITIONS = 5

_TF_GROUP = "model/forceset/Smith2018ArticularContactForce/tf_contact/{cart}"

# nsosim's OpenSim -> shape-model axis swap (x -> -y, y -> z, z -> -x).
_OSIM_TO_NSM = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)


# ------------------------------------------------------------- conditions ----


def conditions() -> list[tuple[str, Config]]:
    """``[(condition, run config), ...]`` -- the reference, then the root's nine measured.

    The list is the data root's (``cohorts.ALL``), not every file under
    config/: a checkout holds the configs of more than one run's donors. Each
    condition is read from the run config that produced its simulations, the
    one description of where its output lives.
    """
    found = []
    for condition in cohorts.ALL:
        config = cohorts.condition_config(condition)
        if condition_name(config) != condition:
            raise KeyError(
                f"{config.origin}: run.gait names condition {condition_name(config)!r}, "
                f"but the file is condition {condition!r}'s config"
            )
        found.append((condition, config))
    return found


def resolve_era(requested: str | None) -> str:
    """The era this data root's tables belong to, checked against ``requested``.

    A root that declares a run holds that run's tables only. The v2 root
    declares none and holds the v2 table, rebuilt only on request.
    """
    decl = declaration()
    run = run_name() if decl and decl.get("run_name") else None
    if run is None:
        if requested != LEGACY_ERA:
            raise SystemExit(
                f"this data root declares no run (no _run.json): it has the earlier run's layout, "
                f"whose table is era {LEGACY_ERA!r}. Pass --era {LEGACY_ERA} to rebuild it"
                + (f" (got --era {requested})." if requested else ".")
            )
        return LEGACY_ERA
    if requested not in (None, run):
        raise SystemExit(
            f"this data root holds run {run!r}, so its QC era is {run!r}, not {requested!r}. "
            "A root's tables belong to its own run; omit --era."
        )
    return run


def qc_key(condition: str, era: str) -> str:
    """The key a condition's rows are filed under in ``era``'s table.

    The active root's key (``cohorts.QC_TAG``): on the v2 root, the run
    names the v2 table used, so a rebuild compares row for row; on a declared
    root, ``<condition>_<run>``. ``era`` is the root's, from ``resolve_era``;
    it is checked once there, and kept here so callers name the table they key.
    """
    return cohorts.QC_TAG[condition]


def knee_list() -> list[str]:
    """Every cohort knee, sorted: the cohort of record, not a completion list.

    A list of knees that finished some earlier run would make a knee that
    simulated perfectly well under one condition invisible to the detector.
    """
    return sorted(json.loads((stage_dir("inputs") / "cohort.json").read_text(encoding="utf-8"))["knees"])


def group_map() -> dict[str, str]:
    table = json.loads((stage_dir("inputs") / "groups.json").read_text(encoding="utf-8"))
    return {knee: group for group, knees in table.items() for knee in knees}


def qc_dir(create: bool = False) -> Path:
    return stage_dir("derived", "qc", create=create)


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _robust_threshold(values, k: float) -> float:
    """median + k x 1.4826 x MAD over the finite values; +inf if there are none."""
    v = np.array([x for x in values if np.isfinite(x)])
    if not len(v):
        return np.inf
    m = np.median(v)
    return m + k * 1.4826 * np.median(np.abs(v - m))


# ---------------------------------------------------------------- spatial ----


def matrix_path(condition: str) -> Path:
    return stage_dir("derived", "combined") / f"combined_data_{condition}.npz"


def _spatial_metrics(data, bone: str):
    """Per knee: medial mean cumulative pressure (MPa s), medial and lateral p95-of-max (MPa)."""
    mask = data[f"{bone}_region_mask"]
    cum = data[f"{bone}_pressure_cum"]
    peak = data[f"{bone}_pressure_max"]
    has = data[f"{bone}_has_pressure"]
    n = cum.shape[0]
    med_cum = np.full(n, np.nan)
    med_peak = np.full(n, np.nan)
    lat_peak = np.full(n, np.nan)
    medial = mask == MEDIAL
    lateral = mask == LATERAL
    for i in range(n):
        m = medial & has[i]
        lat = lateral & has[i]
        if m.sum() >= MIN_COMPARTMENT_VERTICES:
            med_cum[i] = np.mean(cum[i, m]) * 1e-6
            med_peak[i] = np.percentile(peak[i, m], 95) * 1e-6
        if lat.sum() >= MIN_COMPARTMENT_VERTICES:
            lat_peak[i] = np.percentile(peak[i, lat], 95) * 1e-6
    return med_cum, med_peak, lat_peak


def _r4(x) -> float | str:
    return round(float(x), 4) if (x is not None and np.isfinite(x)) else ""


def run_spatial(args) -> int:
    # Measured conditions first, then the reference: the v2 table's row
    # order, so a rebuild can be diffed against it line by line.
    found = conditions()
    order = found[1:] + found[:1]
    missing = [str(matrix_path(c)) for c, _ in order if not matrix_path(c).is_file()]
    if missing:
        raise FileNotFoundError(
            "combined matrix missing for:\n  " + "\n  ".join(missing) + "\n"
            "Skipping a condition would hand every downstream figure zero exclusions "
            "for it. Build them first: pipeline/combine_reference_data.py matrices."
        )
    if args.check:
        print(f"{len(order)} combined matrices present")
        return 0

    rows = []
    for condition, _ in order:
        key = qc_key(condition, args.era)
        data = np.load(matrix_path(condition), allow_pickle=True)
        knees = list(data["subjects"])
        femur = _spatial_metrics(data, "femur")
        tibia = _spatial_metrics(data, "tibia")
        f_thr = _robust_threshold(femur[0], SPATIAL_MAD_K)
        t_thr = _robust_threshold(tibia[0], SPATIAL_MAD_K)
        n_flagged = 0
        for i, knee in enumerate(knees):
            f_cum, f_peak, f_lat = femur[0][i], femur[1][i], femur[2][i]
            t_cum, t_peak, t_lat = tibia[0][i], tibia[1][i], tibia[2][i]
            reasons = []
            peaks = {"femur_med": f_peak, "femur_lat": f_lat,
                     "tibia_med": t_peak, "tibia_lat": t_lat}
            for name, value in peaks.items():
                if np.isfinite(value) and value > PEAK_CEILING_MPA:
                    reasons.append(f"{name}_peak>{PEAK_CEILING_MPA:.0f}MPa({value:.0f})")
            if np.isfinite(f_cum) and f_cum > f_thr:
                reasons.append(f"femur_med_cum>{f_thr:.2f}({f_cum:.2f})")
            if np.isfinite(t_cum) and t_cum > t_thr:
                reasons.append(f"tibia_med_cum>{t_thr:.2f}({t_cum:.2f})")
            n_flagged += bool(reasons)
            rows.append({
                "cohort_tag": key, "gait": condition, "knee": knee,
                "femur_med_cum": _r4(f_cum), "femur_med_peak": _r4(f_peak),
                "femur_lat_peak": _r4(f_lat), "tibia_med_cum": _r4(t_cum),
                "tibia_med_peak": _r4(t_peak), "tibia_lat_peak": _r4(t_lat),
                "is_outlier": bool(reasons), "reason": ";".join(reasons),
            })
        print(f"  {condition:14s} n={len(knees)}  flagged={n_flagged}  "
              f"(cum thr femur={f_thr:.2f}, tibia={t_thr:.2f} MPa·s)")
        del data

    out_dir = qc_dir(create=True)
    output = out_dir / f"pressure_outliers_{args.era}.csv"
    _write_csv(output, rows, list(rows[0]))
    print(f"wrote {output} ({len(rows)} simulations, {sum(r['is_outlier'] for r in rows)} flagged)")
    _spatial_summary(rows, out_dir / f"qc_summary_{args.era}.txt")
    _spatial_plot(rows, out_dir / f"qc_distributions_{args.era}.png")
    return 0


def _spatial_summary(rows: list[dict], path: Path) -> None:
    by_knee = defaultdict(list)
    for r in rows:
        if r["is_outlier"]:
            by_knee[r["knee"]].append(r["gait"])
    multi = {k: v for k, v in by_knee.items() if len(v) >= 3}
    n_flagged = sum(r["is_outlier"] for r in rows)
    lines = [
        "Pressure-outlier QC summary",
        f"rule: peak vertex pressure (p95 of max) > {PEAK_CEILING_MPA:.0f} MPa  OR  medial cum > "
        f"per-cohort median + {SPATIAL_MAD_K:.0f}*MAD",
        f"total simulations: {len(rows)};  flagged: {n_flagged} ({100 * n_flagged / len(rows):.2f}%)",
        f"distinct knees touched: {len(by_knee)};  flagged in >=3 gaits: {len(multi)}",
        "",
        "Per-cohort flagged counts:",
    ]
    by_condition = defaultdict(lambda: [0, 0])
    for r in rows:
        by_condition[r["gait"]][0] += 1
        by_condition[r["gait"]][1] += int(r["is_outlier"])
    lines += [f"  {g:14s} {o}/{n}" for g, (n, o) in by_condition.items()]
    if multi:
        lines += ["", "Knees flagged in >=3 gaits (inspect — possible chronic geometry issue):"]
        lines += [f"  {k}: {len(v)} gaits ({', '.join(v)})"
                  for k, v in sorted(multi.items(), key=lambda x: -len(x[1]))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))


def _spatial_plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaits = list(dict.fromkeys(r["gait"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    panels = [("tibia_med_cum", "Tibia medial cumulative pressure (MPa·s)", None),
              ("tibia_med_peak", "Tibia medial peak pressure p95 (MPa)", PEAK_CEILING_MPA)]
    for ax, (key, label, ceiling) in zip(axes, panels):
        for gi, g in enumerate(gaits):
            values = np.array([r[key] for r in rows if r["gait"] == g and r[key] != ""], float)
            flagged = np.array([r["is_outlier"] for r in rows if r["gait"] == g and r[key] != ""])
            x = np.full(len(values), gi) + (np.linspace(-0.3, 0.3, len(values)) if len(values) else 0)
            ax.scatter(x[~flagged], values[~flagged], s=4, color="steelblue", alpha=0.3)
            ax.scatter(x[flagged], values[flagged], s=22, color="red", marker="x")
        if ceiling:
            ax.axhline(ceiling, color="red", ls="--", lw=1, label=f"{ceiling:.0f} MPa ceiling")
            ax.legend()
        ax.set_xticks(range(len(gaits)))
        ax.set_xticklabels([g.replace("Subject_", "S") for g in gaits], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Pressure-outlier QC — red × = flagged simulation (per cohort)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ------------------------------------------------- time series: geometry ----


def _face_geometry(mesh):
    """(centroids, areas, unit normals) of an all-triangle surface.

    Normals follow the mesh winding; their global sign does not matter, since
    contact force is the norm of a vector sum.
    """
    faces = mesh.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("contact surface must be all triangles")
    v = mesh.points
    a, b, c = v[faces[:, 1]], v[faces[:, 2]], v[faces[:, 3]]
    cross = np.cross(b - a, c - a)
    twice_area = np.linalg.norm(cross, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / twice_area[:, None]
    return (a + b + c) / 3.0, 0.5 * twice_area, np.where(np.isfinite(normals), normals, 0.0)


class TriangleRegions:
    """Labels one bone's contact triangles medial / lateral / trochlea, per knee.

    The labelling depends only on the knee's anatomy, so it is computed once per
    knee and reused for every condition. Nothing is cached to disk: a cache keyed
    on knees outlives the cohort and transform it was built from.
    """

    def __init__(self, bone: str):
        from scipy.spatial import cKDTree

        self.bone = bone
        labels = np.load(stage_dir("derived", "region_labels", f"{bone}_region_labels.npz"),
                         allow_pickle=True)
        self.codes = labels["coarse_code"].astype(np.int64)
        self.tree = cKDTree(labels["ref_points"].astype(np.float64))
        raw = json.loads(transform_file().read_text(encoding="utf-8"))["transforms"][bone]["T_reg_to_647"]
        self.transforms = {knee: np.array(T) for knee, T in raw.items()}
        self.centre = femur_reference_center()

    def geometry(self, knee: Knee):
        """``(region, area, normal)`` at reference size, or None if an input is missing."""
        import pyvista as pv

        surface = articular_surface_path(knee, self.bone)
        if not surface.is_file() or knee.name not in self.transforms:
            return None
        centroids, area, normal = _face_geometry(pv.read(surface))
        # OpenSim local (m) -> femur-aligned mm (femur centre for both bones,
        # since the tibia body's origin is the knee joint) -> reference space.
        aligned = centroids.astype(np.float64) @ _OSIM_TO_NSM * 1000.0 - self.centre
        homogeneous = np.hstack([aligned, np.ones((len(aligned), 1))])
        reference = (homogeneous @ self.transforms[knee.name].T)[:, :3]
        _, nearest = self.tree.query(reference)
        # Area and normal in float64, as the v2 table's geometry was
        # held, so the scaled areas and forces round identically.
        return (self.codes[nearest].astype(np.int16),
                area.astype(np.float64), normal.astype(np.float64))


def condition_scale(results_dir: Path) -> float:
    """The isotropic size factor a measured condition applied to the knee; 1.0 if unscaled.

    Contact area scales by its square; pressure, being intensive, does not.
    """
    report = results_dir.parent / "scaled_model" / "scaled_scaling_report.json"
    if not report.is_file():
        return 1.0
    content = json.loads(report.read_text(encoding="utf-8"))
    if "wa_scale" not in content:
        raise KeyError(
            f"{report} has no 'wa_scale'. Treating the knee as unscaled would mis-size "
            "every contact area (and so every force) without an error."
        )
    return float(content["wa_scale"])


def verify_triangle_order(h5_path: Path, bone: str, area_ref, scale: float, tol: float = 0.02):
    """Check the h5 triangle order against the reference geometry's.

    The h5 holds no areas, but the first contact surface beside it does and
    shares its order: a per-triangle area ratio near ``scale**2`` everywhere
    proves the order matches, a permutation scatters it. ``(ok, info)``;
    unchecked (True) when no surface is present.
    """
    import pyvista as pv

    surfaces = sorted(h5_path.parent.glob(f"*contact_{bone}_cartilage*.vtp"))
    if not surfaces:
        return True, "unchecked(no-vtp)"
    mesh = pv.read(surfaces[0])
    if mesh.n_cells != len(area_ref):
        return False, f"n_cells {mesh.n_cells} != ref {len(area_ref)}"
    area = np.asarray(mesh.compute_cell_sizes(length=False, area=True, volume=False)["Area"])
    expected = area_ref * (scale ** 2)
    good = expected > 0
    ratio = area[good] / expected[good]
    median = float(np.median(ratio))
    within = float(np.mean(np.abs(ratio - median) < tol * median))
    ok = abs(median - 1.0) < 0.05 and within > 0.98
    return ok, f"ratio_med={median:.4f} frac_within_{int(tol * 100)}%={within:.4f}"


def aggregate(h5_path: Path, bone: str, region, area_ref, normal_ref, scale: float):
    """Per-compartment traces over stance, replicating the solver's contact statistics.

    Mean pressure divides by the number of contacting triangles (proximity > 0),
    force is the norm of the vector sum of pressure x area x normal, and p95 is
    the 95th percentile over contacting triangles. Returns
    ``{region: {metric: (frames,)}}`` at the h5's own frame count.
    """
    import h5py

    group = _TF_GROUP.format(cart=f"{bone}_cartilage")
    with h5py.File(h5_path, "r") as handle:
        pressure = handle[f"{group}/triangle_pressure"][:]
        proximity = handle[f"{group}/triangle_proximity"][:]
    n_frames, n_triangles = pressure.shape
    if n_triangles != len(region):
        raise ValueError(f"h5 n_tri {n_triangles} != labelled {len(region)}")
    area = area_ref * (scale ** 2)

    out = {}
    for code in (MEDIAL, LATERAL):
        selected = region == code
        if not selected.any():
            out[code] = {m: np.zeros(n_frames) for m in ("mean", "p95", "max", "force")}
            continue
        p = pressure[:, selected]
        contacting = proximity[:, selected] > 0.0
        n_contact = contacting.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = p.sum(axis=1) / n_contact
        masked = np.where(contacting, p, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p95 = np.nanpercentile(masked, 95, axis=1)
        force = np.linalg.norm((p * area[selected][None, :]) @ normal_ref[selected], axis=1)
        out[code] = {
            "mean": np.where(n_contact > 0, mean, 0.0),
            "p95": np.where(n_contact > 0, np.nan_to_num(p95), 0.0),
            "max": np.abs(p).max(axis=1),
            "force": force,
        }
    return out


def _resample(values, n: int = N_GRID):
    values = np.asarray(values).reshape(-1)
    if len(values) == n:
        return values
    return np.interp(np.linspace(0, 100, n), np.linspace(0, 100, len(values)), values)


_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def compartment_metrics(series) -> dict[str, float]:
    """``{bone: {region: {metric: (101,)}}}`` -> the per-simulation metrics."""
    out = {}
    for bone, code, name in TS_COMPARTMENTS:
        s = series[bone][code]
        out[f"{name}_peakmean"] = float(np.nanmax(s["mean"])) / 1e6
        out[f"{name}_peakp95"] = float(np.nanmax(s["p95"])) / 1e6
        out[f"{name}_peakmax"] = float(np.nanmax(s["max"])) / 1e6
        out[f"{name}_peakforce"] = float(np.nanmax(s["force"]))
        out[f"{name}_pti"] = float(_trapezoid(s["mean"], dx=1 / 100)) / 1e6
    return out


# --------------------------------------------- time series: extract / flag ----


def raw_path(era: str, shard: str = "") -> Path:
    return qc_dir() / f"_timeseries_metrics_raw_{era}{shard}.csv"


def _shard_suffix(shard: int, n_shards: int) -> str:
    return f"__shard{shard}of{n_shards}" if n_shards > 1 else ""


RAW_FIELDS = ["cohort_tag", "gait", "knee", "group"] + [
    f"{name}_{m}" for _, _, name in TS_COMPARTMENTS for m in TS_METRICS
]


def _format(value) -> str:
    """Five significant figures, empty for non-finite -- the v2 raw format."""
    if isinstance(value, float):
        return f"{value:.5g}" if np.isfinite(value) else ""
    return value


def run_extract(args) -> int:
    found = conditions()
    order = {condition: i for i, (condition, _) in enumerate(found)}
    knees = [Knee.parse(name) for name in knee_list()]
    if args.n_shards > 1:
        if not 0 <= args.shard < args.n_shards:
            raise SystemExit(f"--shard {args.shard} out of range for --n-shards {args.n_shards}")
        knees = knees[args.shard :: args.n_shards]
    print(f"extracting {len(knees)} knees x {len(found)} conditions")
    if args.check:
        return 0

    groups = group_map()
    labellers = {bone: TriangleRegions(bone) for bone in ("femur", "tibia")}
    rows, skipped = [], []
    started = time.time()
    for index, knee in enumerate(knees, 1):
        geometry = {bone: labellers[bone].geometry(knee) for bone in labellers}
        for condition, config in found:
            key = qc_key(condition, args.era)
            results_dir = config.for_knee(knee).path("run", "results_dir")
            h5_path = results_dir / "joint-mechanics" / "joint_mechanics.h5"
            reason = None
            if any(g is None for g in geometry.values()):
                reason = "no articular surface or registration transform"
            elif not h5_path.is_file():
                reason = "no joint_mechanics.h5"
            if reason is None:
                # Outside the per-simulation guard: a malformed scaling report
                # is systematic, and a skipped simulation would be kept unflagged.
                scale = condition_scale(results_dir)
                try:
                    series = {}
                    for bone, (region, area, normal) in geometry.items():
                        # The unscaled reference shares its triangle order with
                        # the articular surface by construction; a scaled
                        # condition is checked.
                        if abs(scale - 1.0) > 1e-9:
                            ok, info = verify_triangle_order(h5_path, bone, area, scale)
                            if not ok:
                                raise ValueError(f"{bone} triangle-order guard failed ({info})")
                        traces = aggregate(h5_path, bone, region, area, normal, scale)
                        series[bone] = {code: {m: _resample(v) for m, v in t.items()}
                                        for code, t in traces.items()}
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    reason = f"error: {exc}"
            if reason is not None:
                skipped.append({"cohort_tag": key, "knee": knee.name, "reason": reason})
                continue
            row = {"cohort_tag": key, "gait": condition, "knee": knee.name,
                   "group": groups.get(knee.name, "none")}
            row.update(compartment_metrics(series))
            rows.append(row)
        if index % 25 == 0:
            rate = (time.time() - started) / index
            print(f"  {index}/{len(knees)} knees ({rate:.1f} s/knee)", flush=True)

    # Condition-major, as the v2 table: extraction runs knee-major so
    # each knee's labelling is computed once.
    rows.sort(key=lambda r: (order[r["gait"]], r["knee"]))
    suffix = _shard_suffix(args.shard, args.n_shards)
    out_dir = qc_dir(create=True)
    target = raw_path(args.era, suffix)
    _write_csv(target, [{k: _format(r[k]) for k in RAW_FIELDS} for r in rows], RAW_FIELDS)
    skipped_path = out_dir / f"_timeseries_skipped_{args.era}{suffix}.csv"
    _write_csv(skipped_path, skipped, ["cohort_tag", "knee", "reason"])
    by_condition = defaultdict(lambda: [0, 0])
    for r in rows:
        by_condition[r["cohort_tag"]][0] += 1
    for s in skipped:
        by_condition[s["cohort_tag"]][1] += 1
    for key, (n_ok, n_skip) in by_condition.items():
        print(f"  {key:32s} extracted {n_ok}, skipped {n_skip}")
    print(f"wrote {target} ({len(rows)} simulations); skips in {skipped_path.name}")
    return 0


def run_merge(args) -> int:
    """Join shard caches into the era's raw cache; refuses a partial set.

    A short raw cache yields a table that covers fewer simulations unnoticed,
    which downstream reads as "clean".
    """
    if args.n_shards < 2:
        raise SystemExit("--mode merge needs --n-shards N (N >= 2)")
    suffixes = [_shard_suffix(i, args.n_shards) for i in range(args.n_shards)]
    paths = [raw_path(args.era, s) for s in suffixes]
    absent = [str(p) for p in paths if not p.is_file()]
    if absent:
        raise FileNotFoundError("shard raw CSV(s) missing:\n  " + "\n  ".join(absent))
    order = {qc_key(c, args.era): i for i, (c, _) in enumerate(conditions())}
    rows, skipped = [], []
    for path, suffix in zip(paths, suffixes):
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != RAW_FIELDS:
                raise ValueError(f"shard header mismatch in {path}")
            rows.extend(reader)
        skip_file = qc_dir() / f"_timeseries_skipped_{args.era}{suffix}.csv"
        if skip_file.is_file():
            with skip_file.open(encoding="utf-8") as handle:
                skipped.extend(csv.DictReader(handle))
    keys = [(r["cohort_tag"], r["knee"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate simulations across shards -- shards overlap")
    rows.sort(key=lambda r: (order[r["cohort_tag"]], r["knee"]))
    _write_csv(raw_path(args.era), rows, RAW_FIELDS)
    _write_csv(qc_dir() / f"_timeseries_skipped_{args.era}.csv", skipped,
               ["cohort_tag", "knee", "reason"])
    print(f"wrote {raw_path(args.era)} ({len(rows)} simulations from {len(paths)} shards)")
    return 0


def _load_raw(era: str) -> list[dict]:
    path = raw_path(era)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run --mode extract (and merge) first")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        for _, _, name in TS_COMPARTMENTS:
            for m in TS_METRICS:
                k = f"{name}_{m}"
                r[k] = float(r[k]) if r[k] not in ("", None) else np.nan
    return rows


def run_flag(args) -> int:
    """Apply the rule to the extracted metrics.

    Always reads the written cache rather than in-memory values, so a table
    built in one go and one built from merged shards are identical.
    """
    rows = _load_raw(args.era)
    by_condition = defaultdict(list)
    for r in rows:
        by_condition[r["cohort_tag"]].append(r)
    threshold = {}
    for key, members in by_condition.items():
        for name in ("femur_med", "tibia_med"):
            threshold[(key, name)] = _robust_threshold([r[f"{name}_pti"] for r in members], TS_MAD_K)

    out = []
    for r in rows:
        reasons, max_spike = [], False
        for _, _, name in TS_COMPARTMENTS:
            pm, pp = r[f"{name}_peakmean"], r[f"{name}_peakp95"]
            px, pf = r[f"{name}_peakmax"], r[f"{name}_peakforce"]
            if np.isfinite(pp) and pp > P95_CEIL:
                reasons.append(f"{name}_p95>{P95_CEIL:.0f}({pp:.1f})")
            if np.isfinite(pm) and pm > MEAN_CEIL:
                reasons.append(f"{name}_mean>{MEAN_CEIL:.0f}({pm:.1f})")
            if np.isfinite(pf) and pf > FORCE_CEIL:
                reasons.append(f"{name}_force>{FORCE_CEIL:.0f}({pf:.0f})")
            if np.isfinite(px) and px > MAXSPIKE_INFO:
                max_spike = True
        for name in ("femur_med", "tibia_med"):
            pti, limit = r[f"{name}_pti"], threshold[(r["cohort_tag"], name)]
            if np.isfinite(pti) and pti > limit:
                reasons.append(f"{name}_pti>cohort_med+{TS_MAD_K:.0f}MAD({pti:.2f}>{limit:.2f})")
        out.append({**r, "is_outlier": bool(reasons), "reason": ";".join(reasons),
                    "max_spike": max_spike})

    fields = RAW_FIELDS + ["is_outlier", "reason", "max_spike"]
    target = qc_dir() / f"pressure_outliers_timeseries_{args.era}.csv"
    _write_csv(target, [{k: _format(r[k]) for k in fields} for r in out], fields)
    n_flagged = sum(r["is_outlier"] for r in out)
    print(f"wrote {target} ({len(out)} sims, {n_flagged} flagged = "
          f"{100 * n_flagged / max(len(out), 1):.2f}%)")
    _timeseries_summary(out, qc_dir() / f"qc_summary_timeseries_{args.era}.txt")
    return 0


def _timeseries_summary(rows: list[dict], path: Path) -> None:
    by_knee = defaultdict(list)
    for r in rows:
        if r["is_outlier"]:
            by_knee[r["knee"]].append(r["gait"])
    chronic = {k: v for k, v in by_knee.items() if len(v) >= CHRONIC_MIN_CONDITIONS}
    by_condition = defaultdict(lambda: [0, 0])
    for r in rows:
        by_condition[r["gait"]][0] += 1
        by_condition[r["gait"]][1] += int(r["is_outlier"])
    n_flagged = sum(r["is_outlier"] for r in rows)
    lines = [
        "TIME-SERIES pressure-outlier QC summary",
        f"rule: any compartment p95>{P95_CEIL:.0f}|mean>{MEAN_CEIL:.0f} MPa|force>{FORCE_CEIL:.0f}N  "
        f"OR  medial PTI > per-cohort median+{TS_MAD_K:.0f}MAD  (peakmax single-triangle = info only)",
        f"total sims: {len(rows)};  flagged: {n_flagged} ({100 * n_flagged / max(len(rows), 1):.2f}%)",
        f"distinct knees touched: {len(by_knee)};  chronic (>={CHRONIC_MIN_CONDITIONS} gaits): {len(chronic)}",
        "", "Per-cohort flagged:",
    ] + [f"  {g:14s} {o}/{n}" for g, (n, o) in by_condition.items()]
    if chronic:
        lines += ["", f"Chronic knees (>={CHRONIC_MIN_CONDITIONS} gaits):"]
        lines += [f"  {k}: {len(v)} ({', '.join(v)})"
                  for k, v in sorted(chronic.items(), key=lambda x: -len(x[1]))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))


def run_timeseries(args) -> int:
    if args.mode == "merge":
        return run_merge(args)
    if args.mode == "both" and args.n_shards > 1:
        raise SystemExit("a sharded run is --mode extract per shard, then merge, then flag")
    if args.mode in ("extract", "both"):
        status = run_extract(args)
        if status or args.check or args.mode == "extract":
            return status
    return run_flag(args)


# ------------------------------------------------------------------ unify ----


def _true(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def _float(value) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _robust_z_by_condition(records: dict, metric: str) -> dict:
    """One-sided robust z per condition: (x - median) / (1.4826 x MAD)."""
    by_condition = defaultdict(list)
    for (key, knee), r in records.items():
        by_condition[key].append((knee, _float(r.get(metric))))
    z = {}
    for key, values in by_condition.items():
        arr = np.array([v for _, v in values], float)
        finite = arr[np.isfinite(arr)]
        if len(finite) < 5:
            for knee, _ in values:
                z[(key, knee)] = np.nan
            continue
        median = np.median(finite)
        mad = 1.4826 * np.median(np.abs(finite - median))
        mad = mad if mad > 0 else np.nan
        for knee, v in values:
            z[(key, knee)] = (v - median) / mad if (np.isfinite(v) and np.isfinite(mad)) else np.nan
    return z


def run_unify(args) -> int:
    directory = qc_dir()
    spatial_path = directory / f"pressure_outliers_{args.era}.csv"
    timeseries_path = directory / f"pressure_outliers_timeseries_{args.era}.csv"
    for label, path in (("spatial", spatial_path), ("time-series", timeseries_path)):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} table for era {args.era} not found: {path}\n"
                f"Run 'derive_qc.py spatial --era {args.era}' and "
                f"'derive_qc.py timeseries --era {args.era}' first."
            )
    def load(path):
        with path.open(encoding="utf-8") as handle:
            return {(r["cohort_tag"], r["knee"]): r for r in csv.DictReader(handle)}

    spatial, timeseries = load(spatial_path), load(timeseries_path)
    only_spatial = set(spatial) - set(timeseries)
    only_timeseries = set(timeseries) - set(spatial)
    if only_spatial or only_timeseries:
        print(f"  note: {len(only_spatial)} sims only in spatial, {len(only_timeseries)} only in "
              "time-series -- kept with the available source only")
    keys = sorted(set(spatial) | set(timeseries))

    z_spatial = {m: _robust_z_by_condition(spatial, m) for m in SPATIAL_SCORE_METRICS}
    z_timeseries = {m: _robust_z_by_condition(timeseries, m) for m in TS_SCORE_METRICS}

    rows = []
    for k in keys:
        rs, rt = spatial.get(k), timeseries.get(k)
        fs = _true(rs["is_outlier"]) if rs else False
        ft = _true(rt["is_outlier"]) if rt else False
        zs_values = [z_spatial[m].get(k, np.nan) for m in SPATIAL_SCORE_METRICS]
        zt_values = [z_timeseries[m].get(k, np.nan) for m in TS_SCORE_METRICS]
        zs = np.nanmax(zs_values) if np.any(np.isfinite(zs_values)) else np.nan
        zt = np.nanmax(zt_values) if np.any(np.isfinite(zt_values)) else np.nan
        joint = float(np.hypot(max(zs, 0) if np.isfinite(zs) else 0,
                               max(zt, 0) if np.isfinite(zt) else 0))
        rows.append(dict(
            cohort_tag=k[0], gait=(rt or rs).get("gait", ""), knee=k[1],
            group=(rt.get("group") if rt else "none"),
            flagged_spatial=fs, flagged_timeseries=ft,
            flagged_union=(fs or ft), flagged_intersection=(fs and ft),
            joint_z_spatial=round(float(zs), 3) if np.isfinite(zs) else "",
            joint_z_timeseries=round(float(zt), 3) if np.isfinite(zt) else "",
            joint_combined=round(joint, 3),
            is_outlier_joint=(joint > JOINT_THRESH),
            reason_spatial=(rs.get("reason", "") if rs else ""),
            reason_timeseries=(rt.get("reason", "") if rt else ""),
        ))
    for r in rows:
        r["is_outlier"] = r["flagged_union"]  # the canonical policy

    output = directory / f"pressure_outliers_unified_{args.era}.csv"
    _write_csv(output, rows, list(rows[0]))
    _unify_summary(rows, directory / f"unified_summary_{args.era}.txt",
                   _manual_chronic(directory / "chronic_knees_manual.csv"))
    print(f"\nwrote {output}")
    return 0


def _manual_chronic(path: Path) -> list[str]:
    """Knees on this root's hand-curated drop-entirely list; none if it has no list.

    The list belongs to one run's root and starts empty: a knee is added only
    after that run's simulations have been inspected, never carried over from an
    earlier run whose simulations differ.
    """
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        return sorted(r["knee"].strip() for r in rows
                      if (r.get("scope") or "all").strip() == "all")


def _unify_summary(rows: list[dict], path: Path, manual: list[str]) -> None:
    n = len(rows)

    def rate(flag):
        count = sum(r[flag] for r in rows)
        return count, 100 * count / n

    lines = ["Unified pressure-outlier reconciliation", f"simulations: {n}", "", "Drop rate by policy:"]
    for policy in ("flagged_spatial", "flagged_timeseries", "flagged_union",
                   "flagged_intersection", "is_outlier_joint"):
        count, pct = rate(policy)
        lines.append(f"  {policy:22s} {count:4d}  ({pct:.2f}%)")
    by_group = defaultdict(lambda: [0, 0])
    for r in rows:
        by_group[r["group"]][0] += 1
        by_group[r["group"]][1] += int(r["is_outlier"])
    lines += ["", "UNION (canonical) drops by disease group:"]
    for g, (total, dropped) in sorted(by_group.items()):
        lines.append(f"  {g:18s} {dropped}/{total}  ({100 * dropped / total:.2f}%)"
                     if total else f"  {g}: 0/0")
    by_knee = defaultdict(set)
    for r in rows:
        if r["is_outlier"]:
            by_knee[r["knee"]].add(r["cohort_tag"])
    chronic = sorted(k for k, tags in by_knee.items() if len(tags) >= CHRONIC_MIN_CONDITIONS)
    lines += ["", f"Chronic knees (union, >={CHRONIC_MIN_CONDITIONS} gaits): {len(chronic)}",
              "  " + ", ".join(chronic)]
    lines += ["", f"Manual drop-entirely list (chronic_knees_manual.csv): {len(manual)}",
              "  " + (", ".join(manual) if manual else
                      "none in this root -- each run starts empty; add a knee only after "
                      "inspecting this run's simulations")]
    joint_only = [r for r in rows if r["is_outlier_joint"] and not r["flagged_union"]]
    union_only = [r for r in rows if r["flagged_union"] and not r["is_outlier_joint"]]
    lines += ["", f"joint(combined>{JOINT_THRESH}) flags {len(joint_only)} sims UNION misses "
              "(weird-in-both but neither binary fired);",
              f"union flags {len(union_only)} the joint misses."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# -------------------------------------------------------------------- cli ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="derive_qc.py",
        description="The shared contact-pressure outlier table.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_spatial = sub.add_parser("spatial", help="detector over the combined matrices")
    p_spatial.set_defaults(func=run_spatial)

    p_timeseries = sub.add_parser("timeseries", help="detector over joint_mechanics.h5")
    p_timeseries.add_argument("--mode", choices=["extract", "flag", "both", "merge"],
                              default="both")
    p_timeseries.add_argument("--n-shards", type=int, default=1,
                              help="extract: split the knees into N shards; merge: join N shards")
    p_timeseries.add_argument("--shard", type=int, default=0,
                              help="extract: which shard this run processes (0-based)")
    p_timeseries.set_defaults(func=run_timeseries)

    p_unify = sub.add_parser("unify", help="reconcile the two detectors into the canonical table")
    p_unify.set_defaults(func=run_unify)

    for p in (p_spatial, p_timeseries, p_unify):
        # Never picked by accident: on a declared root the era is the root's own
        # run and nothing else is accepted; on the v2 root it must be
        # named. Two eras' numbers must never be mistaken for each other.
        p.add_argument("--era", default=None,
                       help="the root's run (the default, on a root that declares one); "
                            f"'{LEGACY_ERA}' on a root that declares no run (the earlier run's "
                            "layout; see README, *Run names*)")
        p.add_argument("--check", action="store_true",
                       help="report the work and stop; writes nothing")

    args = parser.parse_args(argv)
    args.era = resolve_era(args.era)
    if args.command == "unify" and args.check:
        print(f"would reconcile {qc_dir()}/pressure_outliers{{,_timeseries}}_{args.era}.csv")
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
