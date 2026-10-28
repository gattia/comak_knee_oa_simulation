#!/usr/bin/env python3
"""Derive Figure 2's inputs: 02_simulations -> 04_results/figure2_compartment_mechanics.

Figure 2 asks how tibiofemoral contact mechanics differ between disease groups
over stance. Unlike every other cohort figure it has no dependency on the
combined pressure/thickness matrices: it reads ``joint_mechanics.h5`` directly,
so a knee contributes as long as it has a stance waveform -- with or without a
follow-up MRI. Its population is therefore larger than the spatial figures'.

What it produces
----------------
``curves/{gait}__{bone}.npz``
    Every included simulation's compartment traces, resampled to 101 points of
    stance: ``knees``, ``groups``, and ``{region}__{metric}`` -> ``(n, 101)``.
    This is the whole extraction; everything below is computed from it, so the
    outputs cannot describe different populations.
``ensemble_{bone}.npz``
    The per-knee ensemble over all ten conditions (one curve per knee, the
    estimand of the figure) and which knees went into it.
``ensemble_spm_{bone}.json`` / ``per_gait_spm_{bone}.json``
    SPM 1D results: the three group pairs and the omnibus ANOVA on the
    ensemble, and a one-way ANOVA per gait condition (data output). Beside the
    plain ``[start, end]`` extents the figures read, every test records its
    critical statistic and each supra-threshold cluster's extent, cluster-level
    p-value and (for the pairwise tests) the direction of the difference.
``composite_scalar_stats_{bone}.csv``
    Group means at the two stance peaks with pairwise Welch t, Holm-corrected
    within each metric x peak.
``compartment_scalars.csv``
    One row per knee x gait x bone x compartment: peak 1 and peak 2 of every
    metric, plus the two pressure-time integrals. The mixed model reads this.
``cross_gait_similarity_{bone}.json``
    Pairwise Pearson of the per-condition group-mean curves over all ten, and
    each condition against the ensemble of the other nine. This is the
    robustness evidence behind the ensemble: how alike the conditions are, and
    which one is least typical.
``coverage.json`` / ``inclusion_manifest.csv`` / ``inclusion_summary.md``
    Which knees were in, and every exclusion with its reason, generated from
    the same run as the outputs above.

The unit of replication is the knee
-----------------------------------
Each knee is simulated under all ten conditions, so knee x gait rows are
repeated measures and pooling them would be pseudoreplication. The figure
averages each knee across the conditions it survives in -- one curve per
knee, n independent -- and the mixed model keeps the rows but carries
``(1 | knee)``. The two therefore report slightly different n: the
ensemble additionally requires a knee to have survived in at least
``MIN_GAITS_FOR_ENSEMBLE`` of the ten, and the per-simulation model does not.

Why all ten conditions, reference included
------------------------------------------
The reference condition is an overground trial from a healthy population and
carries a higher peak load than the nine measured (treadmill, osteoarthritic)
gaits. That load is common to every knee, so averaged into a per-knee curve it
shifts one condition's level and leaves the between-group contrast unchanged,
which is also why the mixed model pools all ten with gait as a factor. Ensembling all ten
makes the waveform curves and the scalar model describe the same conditions.

The leave-one-out correlations in ``cross_gait_similarity_{bone}.json`` show
that the reference is not the least typical condition (one of the measured gaits
sits further from the rest). ``--ensemble measured`` builds the nine-condition
ensemble, without the reference, for comparison.

Compartments
------------
The femur's medial/lateral split is anatomical (the full condyle, with the
trochlea removed) and differs from the solver's axis-sign regions 4/5. The two agree
completely on the tibia and only 62.6% on the femur, where the axis-sign cut
mixes roughly a third of the patellofemoral trochlea into the tibiofemoral
compartments. Each contact triangle is labelled by transforming its centroid
into the shared reference space through the same chain the pressure
coregistration uses, then matching it to the canonical region labels.

That labelling depends only on the knee's own anatomy, so it is computed once
per knee and reused across all ten conditions, so this script loops over knees
in the outer loop. Nothing is cached to disk, because a cache keyed on cohort
membership goes stale without an error when the cohort changes.

Quality control
---------------
Three layers, in order, all recorded per simulation in the manifest:

1. extraction guards -- peak contact force above 5 kN or peak mean pressure
   above 15 MPa in either compartment. Both are physically impossible and
   indicate a contact patch that collapsed to almost nothing.
2. the shared unified QC table (``figures/shared/qc.py``, union policy), which
   is where every pressure analysis in the manuscript gets its exclusions.
3. ``MIN_GAITS_FOR_ENSEMBLE`` -- too few clean gaits to ensemble. This one
   applies to the ensemble only, not to the per-simulation model.

There are no hardcoded knee exclusions: every exclusion comes from the QC
table, so it is versioned and appears in the manifest.

Usage
-----
    python pipeline/derive_figure2_data.py
    python pipeline/derive_figure2_data.py --bone femur
    python pipeline/derive_figure2_data.py --gaits <at least five conditions>
    python pipeline/derive_figure2_data.py --ensemble measured   # nine-condition ensemble
    python pipeline/derive_figure2_data.py --time-base frame     # stance by frame index
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
# The QC table is shared with every other pressure figure and imported here,
# so every figure applies one exclusion rule to one population.
sys.path.insert(0, str(_REPO / "figures"))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import Knee, load_config  # noqa: E402
from comak_knee_oa.paths import UndeclaredRunError, stage_dir  # noqa: E402
from shared import qc  # noqa: E402

FIGURE = "figure2_compartment_mechanics"

# The models stage holds one reconstruction per knee; the articular surfaces
# there are what the contact meshes were built from, at reference size. Where a
# knee's reconstruction lives is the reconstruction config's run.output_dir. It
# carries the run's name, so it is read from there.
RECONSTRUCTION_CONFIG = _REPO / "config" / "reconstruction.json"

# What this figure's outputs (curve files, tables, coverage) call the reference
# condition. Fixed here: config/reference_gait.json's run.gait may be spelled
# "reference" or "default", the renderers look for this name, and a config
# respelling must not rename the files.
REFERENCE_NAME = "reference"

# How a simulation's frames are placed on the 0-100 % stance grid. "time": by
# each frame's time within the stance window the run recorded. "frame": by frame
# index, first frame = 0 %, last = 100 %. The two differ for simulations that do
# not keep the heel-strike and toe-off frames (the v2 measured gaits): by index,
# the first kept frame is filed as 0 % stance although it is ~1.4 % in.
TIME_BASES = ("time", "frame")

BONES = ("femur", "tibia")
MEDIAL, LATERAL, TROCHLEA = 0, 1, 2
REGION_NAME = {MEDIAL: "medial", LATERAL: "lateral"}
ANALYSED_REGIONS = (MEDIAL, LATERAL)
N_GRID = 101

# metric -> (dataset unit -> display scale, label). Stored in SI; the scale is
# carried here so the figures do not each decide what a pressure is measured in.
METRICS = {
    "mean_pressure": (1e-6, "Mean pressure (MPa)"),
    "max_pressure": (1e-6, "Peak pressure · max (MPa)"),
    "p95_pressure": (1e-6, "Peak pressure · p95 (MPa)"),
    "contact_area": (1e6, "Contact area (mm²)"),
    "contact_force": (1.0, "Contact force (N)"),
}

# Extraction guards. A 50 kg person at seven times body weight is 3.5 kN, and
# real cartilage mean pressure peaks near 5 MPa with a cohort p99 around 7, so
# neither threshold is close to anything physiological.
OUTLIER_PEAK_FORCE_N = 5_000.0
OUTLIER_PEAK_MEAN_PRESSURE_PA = 15e6

# A knee must survive in at least this many of the ensembled conditions to be
# ensembled. Below it the per-knee mean reflects which gaits happened to
# converge more than the knee itself.
MIN_GAITS_FOR_ENSEMBLE = 5

# Which conditions the per-knee ensemble averages over. "all" is the
# manuscript's choice; "measured" is the nine measured gaits without the
# reference, for comparison.
ENSEMBLE_CHOICES = ("all", "measured")

_TF_GROUP = "model/forceset/Smith2018ArticularContactForce/tf_contact/{cart}"

# nsosim's OpenSim -> shape-model axis swap (x -> -y, y -> z, z -> -x),
# replicated here so this step needs no OpenSim runtime.
_OSIM_TO_NSM = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)


# --------------------------------------------------------------------------- #
# The conditions                                                              #
# --------------------------------------------------------------------------- #
def gait_conditions() -> list[tuple[str, str]]:
    """``[(name, condition), ...]`` -- reference first, then measured.

    ``name`` is what this figure's outputs file a condition under, and
    ``condition`` is its key in ``comak_knee_oa.cohorts``, through whose run
    config its results are located. They differ only for the reference
    (``REFERENCE_NAME`` vs ``cohorts.REFERENCE``).

    The measured conditions are the data root's donors (``cohorts.measured()``).
    config/measured_gait/ carries configs for more donors than any one root
    holds (RSubject_121 for the v2 root, Subject_121 for v3), so listing the
    configs would declare a condition the root has no simulations for.
    """
    return [(REFERENCE_NAME, cohorts.REFERENCE)] + [(g, g) for g in cohorts.measured()]


def measured_gaits() -> list[str]:
    return [g for g, _ in gait_conditions()[1:]]


# The shared cohort module and the quality-control table call the reference
# condition "default".
_CANONICAL_GAIT = {REFERENCE_NAME: cohorts.REFERENCE}


def qc_keys(gaits) -> dict[str, str]:
    """``gait -> the key the quality-control table files it under``.

    Resolved against the table's own key set, because the keys carry pipeline
    run names whose spelling can change. A hardcoded spelling would not fail:
    ``load_outlier_set`` raises only on keys it is given, and a membership test against a key that
    is merely never present matches nothing, so the exclusions would become
    empty while the manifest recorded that the union policy had been applied.
    """
    canonical = []
    for gait in gaits:
        name = _CANONICAL_GAIT.get(gait, gait)
        if name not in cohorts.ALIASES:
            raise KeyError(
                f"condition {gait!r} is not one of the ten this study ran "
                f"({list(cohorts.ALL)}). If it is a new condition, add it to "
                "comak_knee_oa.cohorts rather than only to config/."
            )
        canonical.append(name)
    resolved = cohorts.resolve_qc_tags(qc.known_cohort_tags(), canonical)
    return dict(zip(gaits, resolved))


@functools.lru_cache(maxsize=None)
def _condition_config(condition: str):
    return cohorts.condition_config(condition)


def condition_config(condition: str, knee: Knee):
    """The run config that simulated ``knee`` under ``condition``, filled for it.

    Every location this figure reads from a simulation -- its results directory,
    its scaled model -- comes through here, so it is defined once, by the config
    that produced it. On a run-named root ``{run}`` is filled from the root.
    """
    return _condition_config(condition).for_knee(knee)


@functools.lru_cache(maxsize=None)
def _reconstruction_config():
    return load_config(RECONSTRUCTION_CONFIG)


def knee_models_dir(knee: Knee) -> Path:
    """``knee``'s reconstruction in 01_models, where config/reconstruction.json puts it.

    The config names the run (``01_models/{run}/{knee}``). A data root that
    declares none (the v2 root) keeps its reconstructions under
    ``cohorts.V2_MODELS_RUN``, as ``cohorts.V2_LAYOUT`` does for the simulations.
    """
    try:
        return _reconstruction_config().for_knee(knee).path("run", "output_dir")
    except UndeclaredRunError:
        return stage_dir("models", cohorts.V2_MODELS_RUN, knee.directory)


# --------------------------------------------------------------------------- #
# Cohort and groups                                                           #
# --------------------------------------------------------------------------- #
def load_knees() -> list[Knee]:
    """The cohort, from ``00_inputs/cohort.json``.

    This is the cohort of record. A list of knees that finished an earlier run
    would go stale when a knee is recovered or re-admitted, and then analyse
    fewer knees than exist.
    """
    manifest = json.loads((stage_dir("inputs") / "cohort.json").read_text(encoding="utf-8"))
    return [Knee.parse(name, manifest.get("timepoint", "00m")) for name in manifest["knees"]]


def load_groups() -> dict[str, str]:
    """``knee name -> disease group key``, from ``00_inputs/groups.json``."""
    raw = json.loads((stage_dir("inputs") / "groups.json").read_text(encoding="utf-8"))
    return {knee: group for group, knees in raw.items() for knee in knees}


# --------------------------------------------------------------------------- #
# Per-knee geometry: which contact triangle is in which compartment           #
# --------------------------------------------------------------------------- #
def _reference_centre() -> np.ndarray:
    path = stage_dir("inputs", "reference", "nsm_meshes", "femur", "ref_femur_alignment.json")
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))["mean_orig"], dtype=np.float64)


def _registration_transforms(bone: str) -> dict[str, np.ndarray]:
    path = stage_dir("derived", "pressure_on_reference", "T_reg_to_647_per_subject.json")
    if not path.is_file():
        raise FileNotFoundError(
            f"Registration transforms not found: {path}\n\n"
            "They place each knee's anatomy in the shared reference space, and "
            "are written by the pressure derivation:\n"
            "    python pipeline/derive_pressure.py\n\n"
            "Or download the published 03_derived stage."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))["transforms"][bone]["T_reg_to_647"]
    return {knee: np.asarray(matrix, dtype=np.float64) for knee, matrix in raw.items()}


def _region_labels(bone: str):
    path = stage_dir("derived", "region_labels", f"{bone}_region_labels.npz")
    if not path.is_file():
        raise FileNotFoundError(
            f"Region labels not found: {path}\n\n"
            "They define the anatomical medial / lateral / trochlea split on the "
            "shared reference surface, and are part of the published 03_derived "
            "stage."
        )
    labels = np.load(path, allow_pickle=True)
    return labels["ref_points"].astype(np.float64), labels["coarse_code"].astype(np.int16)


def _surface_geometry(path: Path):
    """(centroids, triangle areas, unit normals) of a triangle surface.

    Normals follow the mesh winding. Their global sign does not matter: contact
    force is reported as the norm of the vector sum, which the solver also
    computes with a uniform sign convention.
    """
    import pyvista as pv

    mesh = pv.read(str(path))
    faces = mesh.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError(f"{path}: contact surface must be all triangles")
    verts = mesh.points
    a, b, c = verts[faces[:, 1]], verts[faces[:, 2]], verts[faces[:, 3]]
    cross = np.cross(b - a, c - a)
    twice_area = np.linalg.norm(cross, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / twice_area[:, None]
    return ((a + b + c) / 3.0,
            0.5 * twice_area,
            np.where(np.isfinite(normals), normals, 0.0))


class CompartmentLabeller:
    """Labels a knee's contact triangles medial / lateral / trochlea.

    Holds the per-bone reference cloud and its KD-tree, which are the expensive
    part and are shared across every knee.
    """

    def __init__(self, bone: str):
        from scipy.spatial import cKDTree

        self.bone = bone
        points, self.codes = _region_labels(bone)
        self.tree = cKDTree(points)
        self.transforms = _registration_transforms(bone)
        self.centre = _reference_centre()

    def geometry(self, knee: Knee):
        """``(region, area, normal)`` for one knee, or None with a reason.

        Areas are at reference size; a condition that scales the knee scales
        them by its own factor squared.
        """
        surface = (knee_models_dir(knee) / "geometries_nsm_similarity"
                   / self.bone / f"{self.bone}_articular_surface_osim.vtk")
        if not surface.is_file():
            return None, "no reconstructed articular surface in 01_models"
        if knee.name not in self.transforms:
            return None, "no registration transform to the reference surface"

        centroids, area, normal = _surface_geometry(surface)
        # OpenSim local (m) -> reference-aligned (mm) -> shared reference space.
        aligned = centroids @ _OSIM_TO_NSM * 1000.0 - self.centre
        homogeneous = np.hstack([aligned, np.ones((len(aligned), 1))])
        reference = (homogeneous @ self.transforms[knee.name].T)[:, :3]
        _, nearest = self.tree.query(reference)
        return (self.codes[nearest], area, normal), ""


def condition_scale(config) -> float:
    """The isotropic size factor a condition applied to the knee (``config`` is per knee).

    Each knee is reconstructed once at its own size and scaled to the gait
    donor's body, so contact areas scale by this squared and pressures, being
    intensive, not at all. The reference condition does not scale: 1.0.

    A measured condition with no scaling report, or a report without
    ``wa_scale``, raises. Reading either as 1.0 would leave every contact area
    and force of that simulation at reference size with no error.
    """
    if config.get("run", "condition") != "measured":
        return 1.0
    scaled = config.path("model", "scaled_model")
    report = scaled.parent / f"{scaled.stem}_scaling_report.json"
    if not report.is_file():
        raise FileNotFoundError(
            f"no scaling report for a measured-gait simulation: {report}\n"
            "Its contact areas cannot be put at the scaled knee's size without it."
        )
    content = json.loads(report.read_text(encoding="utf-8"))
    if "wa_scale" not in content:
        raise KeyError(
            f"{report} has no 'wa_scale'. Treating the knee as unscaled would "
            "mis-size every contact area (and so every force) without an error."
        )
    return float(content["wa_scale"])


def stance_window(results_dir: Path) -> tuple[float, float]:
    """The stance a simulation's frames are a percentage of: (start, stop) in trial seconds.

    Taken from what the run recorded, because the kept frames can stop short
    of stance (heel-strike and toe-off may be missing):

    * ``run_record.json`` -- this repository's runs. Its ``timing`` holds the
      COMAK window as run, widened by ``comak_window_pad_s`` on each side; the
      pad is taken back off to recover stance.
    * ``inputs/comak_settings.json`` -- runs without a run record (the v2 data
      root), which ran unpadded, so their COMAK window is stance.
    """
    record = results_dir / "run_record.json"
    if record.is_file():
        timing = json.loads(record.read_text(encoding="utf-8"))["timing"]
        pad = float(timing.get("comak_window_pad_s", 0.0))
        return (float(timing["start_time_comak"]) + pad,
                float(timing["stop_time_comak"]) - pad)
    settings = results_dir / "inputs" / "comak_settings.json"
    if settings.is_file():
        content = json.loads(settings.read_text(encoding="utf-8"))
        return float(content["start_time"]), float(content["stop_time"])
    raise FileNotFoundError(
        f"{results_dir} records no stance window (neither run_record.json nor "
        "inputs/comak_settings.json), so its frames cannot be placed in stance by time. "
        "--time-base frame places them by index instead."
    )


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _resample(values, n=N_GRID, percent=None):
    """``values`` on the 0-100 % stance grid.

    ``percent`` is each frame's position in stance (``stance_percent``). None
    places the frames by index instead, the first at 0 % and the last at 100 %
    (``--time-base frame``). Where the grid runs past the frames a simulation
    kept, the curve holds its first or last value (``np.interp``): by time, a
    simulation that starts ~1.4 % into stance repeats that first frame at its
    0 % and 1 % points.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if percent is None:
        if len(values) == n:
            return values
        return np.interp(np.linspace(0, 100, n), np.linspace(0, 100, len(values)), values)
    return np.interp(np.linspace(0, 100, n), percent, values)


def stance_percent(times, window, h5_path=None) -> np.ndarray:
    """Each frame's position in the stance window, in % (0 = heel strike, 100 = toe-off)."""
    start, stop = window
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    percent = (times - start) / (stop - start) * 100.0
    # Frames are a subset of stance; one more than a frame-step outside it means
    # the recorded window is not this simulation's.
    if (np.any(np.diff(percent) <= 0) or percent[0] < -2.0 or percent[-1] > 102.0):
        raise ValueError(
            f"{h5_path or 'simulation'}: frames at {times[0]:.4f}-{times[-1]:.4f} s do "
            f"not sit inside the recorded stance window {start:.4f}-{stop:.4f} s"
        )
    return percent


def aggregate(h5_path: Path, bone: str, region, area, normal, scale=1.0, window=None):
    """Per-compartment traces, replicating the solver's own contact statistics.

    Mean pressure divides by the number of contacting triangles only, contact
    area sums only contacting triangles, and force is the norm of the vector
    sum of pressure x area x normal. Matching the solver here makes these
    numbers comparable with its per-region output -- checked
    to floating-point equality on the compartments where both are defined.

    ``p95_pressure`` has no solver counterpart. It is the 95th percentile over
    contacting triangles, and the figures plot it as the peak because the plain
    maximum is one triangle and moves visibly frame to frame.

    ``window`` is the simulation's stance window (``stance_window``); the frames
    are placed in it by their time. None places them by index.
    """
    import h5py

    group = _TF_GROUP.format(cart=f"{bone}_cartilage")
    with h5py.File(h5_path, "r") as handle:
        pressure = handle[f"{group}/triangle_pressure"][:]
        proximity = handle[f"{group}/triangle_proximity"][:]
        times = handle["time"][:] if window is not None else None
    percent = None if window is None else stance_percent(times, window, h5_path)

    n_frames, n_triangles = pressure.shape
    if n_triangles != len(region):
        raise ValueError(
            f"{h5_path}: {n_triangles} contact triangles, {len(region)} labelled. "
            "The simulation was run on a different mesh than 01_models holds."
        )
    scaled_area = area * (scale ** 2)

    traces = {}
    for code in ANALYSED_REGIONS:
        selected = region == code
        traces[code] = {}
        if not selected.any():
            for metric in METRICS:
                traces[code][metric] = np.zeros(n_frames)
            continue

        p = pressure[:, selected]
        a = scaled_area[selected]
        n = normal[selected]
        contacting = proximity[:, selected] > 0.0
        n_contact = contacting.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_pressure = p.sum(axis=1) / n_contact
        traces[code]["mean_pressure"] = np.where(n_contact > 0, mean_pressure, 0.0)
        traces[code]["max_pressure"] = np.abs(p).max(axis=1)

        import warnings

        masked = np.where(contacting, p, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p95 = np.nanpercentile(masked, 95, axis=1)
        traces[code]["p95_pressure"] = np.where(n_contact > 0, np.nan_to_num(p95), 0.0)

        traces[code]["contact_area"] = (contacting * a[None, :]).sum(axis=1)
        traces[code]["contact_force"] = np.linalg.norm((p * a[None, :]) @ n, axis=1)

    return {code: {m: _resample(traces[code][m], percent=percent) for m in METRICS}
            for code in ANALYSED_REGIONS}


def guard(traces) -> str:
    """"" if the simulation is physically plausible, else why it is not.

    Applied to both compartments. The figure is medial-only, but a knee whose
    lateral side blew up did not solve, and one guard keeps every output of this
    script on the same set of simulations.
    """
    force = max(float(np.nanmax(traces[c]["contact_force"])) for c in ANALYSED_REGIONS)
    pressure = max(float(np.nanmax(traces[c]["mean_pressure"])) for c in ANALYSED_REGIONS)
    if force > OUTLIER_PEAK_FORCE_N:
        return f"peak contact force {force / 1e3:.1f} kN > {OUTLIER_PEAK_FORCE_N / 1e3:g} kN"
    if pressure > OUTLIER_PEAK_MEAN_PRESSURE_PA:
        return (f"peak mean pressure {pressure / 1e6:.0f} MPa "
                f"> {OUTLIER_PEAK_MEAN_PRESSURE_PA / 1e6:g} MPa")
    return ""


def medial_only_guard(traces) -> str:
    """The same test on the medial compartment alone.

    Kept so the two readings can be compared: how often a medial-only guard and the both-compartment guard disagree is a
    measured property of the data, reported in ``coverage.json``.
    """
    force = float(np.nanmax(traces[MEDIAL]["contact_force"]))
    pressure = float(np.nanmax(traces[MEDIAL]["mean_pressure"]))
    if force > OUTLIER_PEAK_FORCE_N or pressure > OUTLIER_PEAK_MEAN_PRESSURE_PA:
        return "medial guard"
    return ""


# --------------------------------------------------------------------------- #
# Extraction driver                                                           #
# --------------------------------------------------------------------------- #
def extract(bone: str, knees, groups, conditions, flagged, keys, time_base="time"):
    """Every included simulation for one bone.

    Returns ``(curves, manifest, disagreements)`` where ``curves`` maps gait to
    ``{"knees": [...], "groups": [...], (region, metric): [(101,), ...]}``.

    Loops knee-major so each knee's triangle labelling is computed once and
    reused across the ten conditions. Cohort-major would recompute it ten times,
    or need a cache, and a cache keyed on cohort membership goes stale when the
    cohort changes.
    """
    labeller = CompartmentLabeller(bone)
    curves = {gait: {"knees": [], "groups": [],
                     **{(r, m): [] for r in ANALYSED_REGIONS for m in METRICS}}
              for gait, _ in conditions}
    manifest, disagreements = [], []
    started = time.time()

    for index, knee in enumerate(knees, start=1):
        group = groups.get(knee.name)
        if group is None:
            manifest.append(dict(bone=bone, knee=knee.name, group="", gait="",
                                 status="not in cohort", reason="no disease group assigned"))
            continue

        geometry, reason = labeller.geometry(knee)
        if geometry is None:
            for gait, _ in conditions:
                manifest.append(dict(bone=bone, knee=knee.name, group=group, gait=gait,
                                     status="error", reason=reason))
            continue
        region, area, normal = geometry

        for gait, condition in conditions:
            row = dict(bone=bone, knee=knee.name, group=group, gait=gait)
            config = condition_config(condition, knee)
            directory = config.path("run", "results_dir")
            h5_path = directory / "joint-mechanics" / "joint_mechanics.h5"
            if not h5_path.is_file():
                manifest.append({**row, "status": "missing",
                                 "reason": "no joint_mechanics.h5 (simulation did not complete)"})
                continue
            if (keys[gait], knee.name) in flagged:
                manifest.append({**row, "status": "excluded",
                                 "reason": "flagged in the shared QC table (union policy)"})
                continue
            # Outside the try below: a missing size factor or stance window is a
            # defect of the run, not of this one simulation, and must not be
            # filed as a per-simulation "error" row.
            scale = condition_scale(config)
            window = stance_window(directory) if time_base == "time" else None
            try:
                traces = aggregate(h5_path, bone, region, area, normal,
                                   scale=scale, window=window)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                manifest.append({**row, "status": "error", "reason": f"aggregation failed: {exc}"})
                continue

            failed = guard(traces)
            if bool(failed) != bool(medial_only_guard(traces)):
                disagreements.append({**row, "both": failed,
                                      "medial_only": medial_only_guard(traces)})
            if failed:
                manifest.append({**row, "status": "excluded", "reason": failed})
                continue

            cell = curves[gait]
            cell["knees"].append(knee.name)
            cell["groups"].append(group)
            for code in ANALYSED_REGIONS:
                for metric in METRICS:
                    cell[(code, metric)].append(traces[code][metric].astype(np.float32))
            manifest.append({**row, "status": "included", "reason": ""})

        if index % 50 == 0:
            rate = (time.time() - started) / index
            print(f"  {bone}: {index}/{len(knees)} knees "
                  f"({rate:.2f} s/knee, ~{rate * (len(knees) - index) / 60:.0f} min left)",
                  flush=True)

    return curves, manifest, disagreements


def write_curves(out_dir: Path, bone: str, curves) -> dict[str, int]:
    """One npz per condition. Returns the per-condition knee count."""
    directory = out_dir / "curves"
    directory.mkdir(parents=True, exist_ok=True)
    counts = {}
    for gait, cell in curves.items():
        arrays = {"knees": np.array(cell["knees"], dtype=object),
                  "groups": np.array(cell["groups"], dtype=object)}
        for code in ANALYSED_REGIONS:
            for metric in METRICS:
                stack = cell[(code, metric)]
                arrays[f"{REGION_NAME[code]}__{metric}"] = (
                    np.vstack(stack) if stack else np.empty((0, N_GRID), np.float32))
        np.savez_compressed(directory / f"{gait}__{bone}.npz", **arrays)
        counts[gait] = len(cell["knees"])
    return counts


def read_curves(out_dir: Path, gait: str, bone: str):
    return np.load(out_dir / "curves" / f"{gait}__{bone}.npz", allow_pickle=True)


# --------------------------------------------------------------------------- #
# The ensemble                                                                #
# --------------------------------------------------------------------------- #
def build_ensemble(curves, gaits, region=MEDIAL):
    """Average each knee across the given conditions it survived in.

    One curve per knee, so the group comparison is over independent units. A
    knee appearing in fewer than ``MIN_GAITS_FOR_ENSEMBLE`` of them is dropped:
    below that its mean says more about which gaits converged than about the
    knee.
    """
    per_knee: dict[str, dict] = {}
    for gait in gaits:
        cell = curves[gait]
        for i, knee in enumerate(cell["knees"]):
            record = per_knee.setdefault(knee, {"group": cell["groups"][i],
                                                "curves": {m: [] for m in METRICS}})
            for metric in METRICS:
                record["curves"][metric].append(cell[(region, metric)][i])

    ensemble: dict[str, dict[str, list]] = {}
    knees_used: dict[str, list[str]] = {}
    dropped = []
    for knee, record in sorted(per_knee.items()):
        n_gaits = len(record["curves"]["mean_pressure"])
        if n_gaits < MIN_GAITS_FOR_ENSEMBLE:
            dropped.append((knee, n_gaits))
            continue
        group = record["group"]
        bucket = ensemble.setdefault(group, {m: [] for m in METRICS})
        for metric in METRICS:
            bucket[metric].append(np.mean(np.vstack(record["curves"][metric]), axis=0))
        knees_used.setdefault(group, []).append(knee)

    stacked = {g: {m: np.vstack(v[m]).astype(np.float32) for m in METRICS}
               for g, v in ensemble.items()}
    return stacked, knees_used, dropped


def per_knee_by_group(cell, region=MEDIAL):
    """One condition's curves regrouped as ``group -> metric -> (n, 101)``."""
    groups = np.asarray(cell["groups"])
    out = {}
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        out[group] = {m: cell[(region, m)][selected] if len(cell[(region, m)])
                      else np.empty((0, N_GRID), np.float32)
                      for m in METRICS}
    return out


def _cell_from_npz(data):
    """``curves/*.npz`` -> the in-memory shape ``build_ensemble`` expects."""
    cell = {"knees": list(data["knees"]), "groups": list(data["groups"])}
    for code in ANALYSED_REGIONS:
        for metric in METRICS:
            cell[(code, metric)] = data[f"{REGION_NAME[code]}__{metric}"]
    return cell


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
GROUP_ORDER = ("kl0_1_healthy", "kl0_1_progressor", "kl2")
PAIRS = (("kl0_1_healthy", "kl0_1_progressor"),
         ("kl0_1_healthy", "kl2"),
         ("kl0_1_progressor", "kl2"))
SPM_OUTCOMES = ("mean_pressure", "p95_pressure", "contact_area")


def _clusters(mask) -> list[list[int]]:
    """Contiguous runs of True as inclusive ``[start, end]`` percent-of-stance."""
    found, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            found.append([start, i - 1])
            start = None
    if start is not None:
        found.append([start, len(mask) - 1])
    return found


def _record(inference, signed: bool, alpha: float) -> dict:
    """An SPM inference as data: the critical statistic and every cluster.

    Each supra-threshold cluster is stored with its inclusive ``[start, end]``
    percent-of-stance extent -- the cluster's endpoints floored and ceiled to
    the grid, which is how the figures draw their bars -- its
    cluster-level p-value, and, for a signed statistic, the sign of the test
    statistic over it, which is the direction of the group difference. The
    sign is read from the t field spm1d computed, so it cannot disagree with
    the test that found the cluster.
    """
    clusters = []
    if inference.h0reject:
        for cluster in inference.clusters:
            low = int(np.floor(cluster.endpoints[0]))
            high = max(low, int(np.ceil(cluster.endpoints[1])))
            record = {"start": low, "end": high, "p": float(cluster.P)}
            if signed:
                record["sign"] = int(np.sign(np.mean(inference.z[low:high + 1])))
            clusters.append(record)
    return {"alpha": alpha, "zstar": float(inference.zstar), "clusters": clusters}


def _untested(alpha: float) -> dict:
    return {"alpha": alpha, "zstar": None, "clusters": []}


def _mask_from(record) -> np.ndarray:
    """A test record back to the per-frame significance mask the figures draw.

    Two clusters whose rounded extents touch merge into one run here, so the
    ``[start, end]`` lists the figures read can hold one entry where the test
    record holds two; the record is the one that carries the p-values.
    """
    mask = np.zeros(N_GRID, bool)
    for cluster in record["clusters"]:
        mask[cluster["start"]:cluster["end"] + 1] = True
    return mask


def spm_pairwise(ensemble, metric, alpha=0.05) -> dict[str, dict]:
    """Two-tailed SPM t-test for each group pair, ``"a|b" -> test record``.

    The statistic is ``a - b``, so a cluster with ``sign`` +1 is a window over
    which the first group of the pair is higher.
    """
    import spm1d

    out = {}
    for a, b in PAIRS:
        A = ensemble.get(a, {}).get(metric)
        B = ensemble.get(b, {}).get(metric)
        record = _untested(alpha)
        if A is not None and B is not None and len(A) >= 5 and len(B) >= 5:
            inference = spm1d.stats.ttest2(A, B).inference(alpha=alpha, two_tailed=True)
            record = _record(inference, signed=True, alpha=alpha)
        record["test"] = "spm1d ttest2, two-tailed"
        record["n"] = {a: int(len(A)) if A is not None else 0,
                       b: int(len(B)) if B is not None else 0}
        record["positive_sign_means"] = f"{a} > {b}"
        out[f"{a}|{b}"] = record
    return out


def spm_anova(group_arrays, alpha=0.05) -> dict:
    """One-way SPM ANOVA across the groups, as a test record."""
    import spm1d

    arrays = [a for a in group_arrays if a is not None and len(a) >= 5]
    if len(arrays) < 2:
        record = _untested(alpha)
    else:
        observations = np.vstack(arrays)
        labels = np.concatenate([[i] * len(a) for i, a in enumerate(arrays)])
        inference = spm1d.stats.anova1(observations, labels).inference(alpha=alpha)
        record = _record(inference, signed=False, alpha=alpha)
    record["test"] = "spm1d anova1"
    record["n"] = [int(len(a)) for a in arrays]
    return record


def peak_indices(mean_pressure) -> tuple[int, int]:
    """Early- and late-stance mean-pressure peak frames, per knee.

    Every metric is read at these two instants (not at its own maximum), so
    pressure and contact area describe the same moment of stance, as the panel
    compares them.
    """
    return int(np.argmax(mean_pressure[:51])), 50 + int(np.argmax(mean_pressure[50:]))


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, input order preserved."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def scalar_stats(ensemble, bone: str) -> list[dict]:
    """Group means at both peaks with pairwise Welch t, Holm-corrected."""
    from scipy import stats as sps

    at_peak = {}
    for group, metrics in ensemble.items():
        mean_pressure = metrics["mean_pressure"]
        if not len(mean_pressure):
            continue
        rows = np.arange(len(mean_pressure))
        t1 = np.argmax(mean_pressure[:, :51], axis=1)
        t2 = 50 + np.argmax(mean_pressure[:, 50:], axis=1)
        for metric in METRICS:
            values = metrics[metric]
            at_peak[(group, metric, 1)] = values[rows, t1]
            at_peak[(group, metric, 2)] = values[rows, t2]

    stats_rows = []
    for metric in SPM_OUTCOMES:
        scale = METRICS[metric][0]
        for peak in (1, 2):
            block = []
            for a, b in PAIRS:
                va = at_peak.get((a, metric, peak))
                vb = at_peak.get((b, metric, peak))
                if va is None or vb is None:
                    continue
                va, vb = va * scale, vb * scale
                _, p = sps.ttest_ind(va, vb, equal_var=False)
                block.append(dict(bone=bone, metric=metric, peak=peak, pair=f"{a}|{b}",
                                  n_a=len(va), n_b=len(vb),
                                  mean_a=float(np.mean(va)), mean_b=float(np.mean(vb)),
                                  ci95_a=float(1.96 * np.std(va) / np.sqrt(len(va))),
                                  ci95_b=float(1.96 * np.std(vb) / np.sqrt(len(vb))),
                                  p_raw=float(p)))
            for row, adjusted in zip(block, holm([r["p_raw"] for r in block])):
                row["p_holm"] = float(adjusted)
            stats_rows.extend(block)
    return stats_rows


def cross_gait_similarity(curves, gaits, metric="mean_pressure", region=MEDIAL):
    """Pairwise Pearson of the per-condition group-mean curves, over all of them.

    Each condition is summarised as its three group-mean curves concatenated,
    so the correlation asks whether two conditions tell the same between-group
    story through stance. Descriptive only: it shows how alike the averaged
    conditions are. The case for averaging them is the pooled-model argument
    in the module docstring.
    """
    means = {}
    for gait in gaits:
        by_group = per_knee_by_group(curves[gait], region)
        stacked = [by_group[g][metric].mean(axis=0)
                   for g in GROUP_ORDER if g in by_group and len(by_group[g][metric]) >= 5]
        if stacked:
            means[gait] = np.concatenate(stacked)

    pairs = {}
    values = []
    names = sorted(means)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = float(np.corrcoef(means[a], means[b])[0, 1])
            pairs[f"{a}|{b}"] = r
            values.append(r)
    return {"metric": metric, "pairwise_pearson": pairs,
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None}


def condition_versus_rest(curves, gaits, region=MEDIAL) -> dict[str, dict[str, float]]:
    """Leave-one-out: each condition against the ensemble of all the others.

    ``condition -> metric -> r``: the mean over groups of the Pearson r between
    the condition's own group-mean curve and the group-mean curve of the
    per-knee ensemble built from every other condition (same
    ``MIN_GAITS_FOR_ENSEMBLE`` rule). It answers "which condition is least like
    the rest" for every condition at once. On the released data the reference is not the least typical
    condition (a measured gait is), so the ensemble does not single it out.
    """
    out = {}
    for gait in gaits:
        others, _, _ = build_ensemble(curves, [g for g in gaits if g != gait], region)
        own = per_knee_by_group(curves[gait], region)
        out[gait] = {}
        for metric in SPM_OUTCOMES:
            values = []
            for group in GROUP_ORDER:
                a = own.get(group, {}).get(metric)
                b = others.get(group, {}).get(metric)
                if a is not None and b is not None and len(a) >= 5 and len(b) >= 5:
                    values.append(float(np.corrcoef(a.mean(axis=0), b.mean(axis=0))[0, 1]))
            if values:
                out[gait][metric] = float(np.mean(values))
    return out


# --------------------------------------------------------------------------- #
# The scalar table the mixed model reads                                      #
# --------------------------------------------------------------------------- #
def _integrate(trace, dx: float = 1 / 100) -> float:
    """Pressure-time integral over stance: the trapezoid rule, written out.

    Written out because numpy's helper is deprecated under one name and only
    available under the other in numpy 2; this way the number does not depend
    on the numpy version.
    """
    values = np.asarray(trace, dtype=np.float64)
    return float((values.sum() - 0.5 * (values[0] + values[-1])) * dx)


def write_scalar_table(out_dir: Path, conditions, bones) -> int:
    """One row per knee x gait x bone x compartment.

    Both compartments and every metric go in one table. A separate file per
    analysis subset could let a model be fitted on one selection and stored
    under another's name without an error.
    """
    fields = (["knee", "group", "gait", "bone", "compartment"]
              + [f"{m}_peak{p}" for m in METRICS for p in (1, 2)]
              + ["mean_pressure_pti", "p95_pressure_pti"])
    path = out_dir / "compartment_scalars.csv"
    written = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bone in bones:
            for gait, _ in conditions:
                data = read_curves(out_dir, gait, bone)
                cell = _cell_from_npz(data)
                for i, knee in enumerate(cell["knees"]):
                    for code in ANALYSED_REGIONS:
                        row = {"knee": knee, "group": cell["groups"][i], "gait": gait,
                               "bone": bone, "compartment": REGION_NAME[code]}
                        for metric in METRICS:
                            trace = cell[(code, metric)][i]
                            row[f"{metric}_peak1"] = f"{np.nanmax(trace[:51]):.6g}"
                            row[f"{metric}_peak2"] = f"{np.nanmax(trace[50:]):.6g}"
                        for metric in ("mean_pressure", "p95_pressure"):
                            row[f"{metric}_pti"] = f"{_integrate(cell[(code, metric)][i]):.6g}"
                        writer.writerow(row)
                        written += 1
    print(f"wrote {path.name}: {written} rows")
    return written


# --------------------------------------------------------------------------- #
# Manifests                                                                   #
# --------------------------------------------------------------------------- #
def write_manifest(out_dir: Path, rows) -> None:
    """Every knee x gait x bone cell with its status, generated not curated."""
    fields = ["bone", "knee", "group", "gait", "status", "reason"]
    with (out_dir / "inclusion_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    reasons: dict[str, int] = {}
    for row in rows:
        if row["status"] != "included":
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1

    lines = ["# Figure 2 — inclusion", "",
             f"{counts.get('included', 0)} of {len(rows)} simulated bones included.", "",
             "| status | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "## Why cells were left out", "", "| reason | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
    lines += ["", f"QC era: {qc.ERA} · table: {qc.unified_csv().name} · policy: union", ""]
    (out_dir / "inclusion_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def _digest(path: Path) -> str:
    """SHA256 of a file this run depended on.

    Recorded for the quality-control table specifically, because it is shared
    with every other figure and is rebuilt from time to time. This derivation
    reads it once at startup; if it is rewritten mid-run the numbers here are
    still self-consistent, but nothing else would show which version they came
    from; the digest records it.
    """
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_enough_conditions(conditions, ensemble_over: str) -> None:
    """Refuse, before anything is written, a run with too few conditions to ensemble.

    A knee enters the ensemble only if it survived in ``MIN_GAITS_FOR_ENSEMBLE``
    of the ensembled conditions, so with fewer conditions than that the
    ensemble is empty by construction and nothing downstream of it can be
    computed. Checking first avoids writing the curves and then failing
    half-way through the summaries, which would leave a partial ``04_results``.
    """
    ensembled = ensemble_conditions([c[0] for c in conditions], ensemble_over)
    if len(ensembled) < MIN_GAITS_FOR_ENSEMBLE:
        raise SystemExit(
            f"Figure 2 needs at least {MIN_GAITS_FOR_ENSEMBLE} gait conditions to ensemble "
            f"(--ensemble {ensemble_over}); this run has {len(ensembled)}: "
            f"{', '.join(ensembled) or 'none'}. Its per-knee ensemble keeps a knee only if it "
            f"was simulated in at least {MIN_GAITS_FOR_ENSEMBLE} of them, so with fewer it is empty. "
            "Nothing was written. The manuscript ensembles all ten (the reference gait and the "
            "nine measured gaits).")


def derive(bones=BONES, gaits=None, force=False, ensemble_over="all",
           time_base="time") -> Path:
    conditions = [c for c in gait_conditions() if gaits is None or c[0] in gaits]
    if not conditions:
        raise SystemExit(f"no conditions match {gaits}")
    if ensemble_over not in ENSEMBLE_CHOICES:
        raise SystemExit(f"--ensemble must be one of {ENSEMBLE_CHOICES}, not {ensemble_over!r}")
    if time_base not in TIME_BASES:
        raise SystemExit(f"--time-base must be one of {TIME_BASES}, not {time_base!r}")
    require_enough_conditions(conditions, ensemble_over)
    out_dir = stage_dir("results", FIGURE, create=True)

    knees = load_knees()
    groups = load_groups()
    keys = qc_keys([c[0] for c in conditions])
    flagged = qc.load_outlier_set(cohort_tags=list(keys.values()), source="union")
    print(f"{len(knees)} knees · {len(conditions)} conditions · "
          f"{len(flagged)} simulations flagged by QC era {qc.ERA}")

    coverage = {"qc_era": qc.ERA, "qc_table": qc.unified_csv().name,
                "qc_table_sha256": _digest(qc.unified_csv()),
                "qc_keys": keys, "n_flagged_simulations": len(flagged),
                "n_knees_in_cohort": len(knees),
                "min_gaits_for_ensemble": MIN_GAITS_FOR_ENSEMBLE,
                "ensemble_over": ensemble_over,
                "time_base": time_base,
                "conditions": [c[0] for c in conditions], "bones": {}}

    for bone in bones:
        existing = out_dir / "curves" / f"{conditions[-1][0]}__{bone}.npz"
        if existing.exists() and not force:
            print(f"{bone}: curves exist; --force to re-extract")
        else:
            print(f"\n=== extracting {bone} ===", flush=True)
            curves, manifest, disagreements = extract(
                bone, knees, groups, conditions, flagged, keys, time_base=time_base)
            counts = write_curves(out_dir, bone, curves)
            print(f"  {bone} included per condition: {counts}")
            if disagreements:
                print(f"  warning: {len(disagreements)} cells where the medial-only and "
                      f"both-compartment guards disagree")
            coverage["bones"][bone] = {"included_per_condition": counts,
                                       "guard_disagreements": disagreements}
            _replace_manifest_bone(out_dir, bone, manifest)

        summarise(out_dir, bone, conditions, coverage, ensemble_over=ensemble_over)

    (out_dir / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    if all((out_dir / "curves" / f"{c[0]}__{b}.npz").exists()
           for c in conditions for b in bones):
        write_scalar_table(out_dir, conditions, bones)
    return out_dir


def _replace_manifest_bone(out_dir: Path, bone: str, rows) -> None:
    """Swap one bone's rows into the manifest, keeping the other's.

    Re-extracting one bone must not shrink the manifest to that bone, which is
    what rewriting it would do. The manifest is the record of which knees an
    analysis saw, and must not depend on the order the bones were run in.
    """
    path = out_dir / "inclusion_manifest.csv"
    kept: list[dict] = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            kept = [row for row in csv.DictReader(handle) if row["bone"] != bone]
    write_manifest(out_dir, kept + list(rows))


class EmptyConditionError(RuntimeError):
    """A declared condition contributed no simulations at all."""


def require_every_condition_populated(curves, conditions, bone: str) -> None:
    """Stop if a condition this run declared holds nothing.

    Declaring a condition -- the data root listing it (``cohorts.measured()``)
    -- is a claim that its simulations exist. When none do, every number downstream that
    says "ten conditions" is describing nine, and nothing else in this script
    will notice: each absent cell is filed as "no joint_mechanics.h5 (simulation
    did not complete)", which is a status that real, converged-short knees
    produce too. A condition at 749 of 756 and a condition at 0 of 756 are the
    same kind of row, differing only in count, and a shortfall is otherwise
    indistinguishable from a legitimately smaller population.

    So absent cells remain a coverage statistic, reported in
    ``coverage.json``; an absent condition is an error.
    """
    keys = dict(conditions)
    empty = sorted(g for g, cell in curves.items() if not cell["knees"])
    if not empty:
        return
    lines = "\n".join(
        f"    {g:<16} expected under  "
        f"{_condition_config(keys[g]).data['run']['results_dir']}" for g in empty
    )
    raise EmptyConditionError(
        f"{bone}: {len(empty)} of {len(curves)} declared condition(s) contributed "
        f"no simulations:\n{lines}\n\n"
        "A condition the data root declares is a claim that its "
        "simulations exist. Either they were never produced, or they are not "
        "where the config says. Both are defects -- proceeding would publish a "
        "figure whose coverage file names ten conditions and whose numbers come "
        "from fewer.\n\n"
        "Fix by one of:\n"
        "  - vendor or run the missing condition\n"
        "      python pipeline/vendor_inputs.py --group simulations --execute\n"
        "  - or, if it is genuinely out of scope for this run, name the ones you "
        "want\n"
        "      python pipeline/derive_figure2_data.py --gaits "
        + " ".join(g for g in curves if g not in empty)
        + "\n\n"
        "Do not work around this by deleting the condition's run config: that "
        "makes the figure silently describe fewer conditions, which is the "
        "failure this check exists to catch."
    )


def ensemble_conditions(gaits, ensemble_over: str) -> list[str]:
    """Which of the derived conditions the per-knee ensemble averages over."""
    if ensemble_over == "all":
        return list(gaits)
    if ensemble_over == "measured":
        return [g for g in gaits if g in measured_gaits()]
    raise ValueError(f"--ensemble must be one of {ENSEMBLE_CHOICES}, not {ensemble_over!r}")


def summarise(out_dir: Path, bone: str, conditions, coverage,
              ensemble_over: str = "all") -> None:
    """Everything downstream of the extraction: ensemble, SPM, scalars."""
    gaits = [g for g, _ in conditions]
    ensembled = ensemble_conditions(gaits, ensemble_over)
    curves = {g: _cell_from_npz(read_curves(out_dir, g, bone)) for g in gaits}
    require_every_condition_populated(curves, conditions, bone)

    ensemble, knees_used, dropped = build_ensemble(curves, ensembled)
    counts = {g: len(knees_used.get(g, [])) for g in GROUP_ORDER}
    total = sum(counts.values())
    breakdown = " / ".join(f"{g}={counts[g]}" for g in GROUP_ORDER)
    print(f"  {bone} ensemble over {len(ensembled)} conditions ({ensemble_over}): "
          f"n={total} ({breakdown}), {len(dropped)} dropped below "
          f"{MIN_GAITS_FOR_ENSEMBLE} conditions")

    np.savez_compressed(
        out_dir / f"ensemble_{bone}.npz",
        **{f"{g}__{m}": ensemble[g][m] for g in ensemble for m in METRICS},
        **{f"knees__{g}": np.array(knees_used.get(g, []), dtype=object) for g in GROUP_ORDER},
    )

    print(f"  {bone}: SPM on the ensemble ...", flush=True)
    # ``*_clusters_pct_stance`` are the extents the figure draws;
    # ``pairwise_tests`` / ``anova_tests`` are the tests behind them, with the
    # critical statistic, cluster p-values and directions.
    spm = {"coverage": {"n_used": counts, "n_dropped_low_coverage": len(dropped),
                        "min_gaits": MIN_GAITS_FOR_ENSEMBLE,
                        "n_ensemble_conditions": len(ensembled),
                        "ensemble_conditions": list(ensembled)},
           "sig_clusters_pct_stance": {}, "anova_clusters_pct_stance": {},
           "pairwise_tests": {}, "anova_tests": {}}
    for metric in SPM_OUTCOMES:
        pairwise = spm_pairwise(ensemble, metric)
        anova = spm_anova([ensemble.get(g, {}).get(metric) for g in GROUP_ORDER])
        spm["pairwise_tests"][metric] = pairwise
        spm["anova_tests"][metric] = anova
        spm["sig_clusters_pct_stance"][metric] = {
            k: _clusters(_mask_from(v)) for k, v in pairwise.items()}
        spm["anova_clusters_pct_stance"][metric] = _clusters(_mask_from(anova))
        extent = int(_mask_from(pairwise["kl0_1_healthy|kl0_1_progressor"]).sum())
        print(f"    {metric:14s} Healthy vs Incident significant over {extent}% of stance")
    (out_dir / f"ensemble_spm_{bone}.json").write_text(json.dumps(spm, indent=2), encoding="utf-8")

    print(f"  {bone}: per-gait SPM for the supplement ...", flush=True)
    per_gait = {}
    for gait in gaits:
        by_group = per_knee_by_group(curves[gait])
        anova = {m: spm_anova([by_group.get(g, {}).get(m) for g in GROUP_ORDER])
                 for m in SPM_OUTCOMES}
        per_gait[gait] = {
            "n": {g: int(len(by_group.get(g, {}).get("mean_pressure", []))) for g in GROUP_ORDER},
            "anova_clusters_pct_stance": {m: _clusters(_mask_from(r)) for m, r in anova.items()},
            "anova_tests": anova,
        }
    (out_dir / f"per_gait_spm_{bone}.json").write_text(json.dumps(per_gait, indent=2), encoding="utf-8")

    rows = scalar_stats(ensemble, bone)
    with (out_dir / f"composite_scalar_stats_{bone}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    similarity = cross_gait_similarity(curves, gaits)
    similarity["conditions"] = list(gaits)
    similarity["condition_vs_rest_shape_r"] = condition_versus_rest(curves, gaits)
    loo = {g: r["mean_pressure"] for g, r in similarity["condition_vs_rest_shape_r"].items()
           if "mean_pressure" in r}
    least = min(loo, key=loo.get) if loo else None
    similarity["least_typical_condition"] = {
        "metric": "mean_pressure", "condition": least, "r": loo.get(least)}
    (out_dir / f"cross_gait_similarity_{bone}.json").write_text(json.dumps(similarity, indent=2), encoding="utf-8")
    print(f"    conditions agree pairwise at r={similarity['mean']:.3f} "
          f"({similarity['min']:.3f}–{similarity['max']:.3f}); least typical by "
          f"leave-one-out: {least} (r={loo.get(least, float('nan')):.3f})")

    coverage["bones"].setdefault(bone, {})
    coverage["bones"][bone].update({
        "ensemble_over": ensemble_over,
        "ensemble_conditions": list(ensembled),
        "n_ensemble_conditions": len(ensembled),
        "ensemble_n": counts, "ensemble_total": total,
        "n_dropped_low_coverage": len(dropped),
        "dropped_low_coverage": [{"knee": k, "n_gaits": n} for k, n in dropped],
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bone", nargs="+", default=list(BONES), choices=list(BONES))
    parser.add_argument("--gaits", nargs="+", default=None,
                        help="restrict to these conditions; default is every one "
                             "the data root declares")
    parser.add_argument("--force", action="store_true", help="re-extract curves that exist")
    parser.add_argument("--ensemble", choices=list(ENSEMBLE_CHOICES), default="all",
                        help="conditions the per-knee ensemble averages over: 'all' "
                             "(the manuscript; reference + measured) or 'measured' "
                             "(the nine measured gaits only)")
    parser.add_argument("--time-base", choices=list(TIME_BASES), default="time",
                        help="place each simulation's frames in stance by their time "
                             "within the recorded stance window ('time', the default), "
                             "or by frame index ('frame', how v2's published curves "
                             "were made)")
    args = parser.parse_args()

    out_dir = derive(bones=args.bone, gaits=args.gaits, force=args.force,
                     ensemble_over=args.ensemble, time_base=args.time_base)
    print(f"\nFigure 2 inputs in {out_dir}")
    print("Next: figures/figure2_mixed_model.R, then "
          "figures/figure2_compartment_mechanics.py")


if __name__ == "__main__":
    main()
