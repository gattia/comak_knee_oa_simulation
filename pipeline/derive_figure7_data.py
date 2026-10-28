#!/usr/bin/env python3
"""Derive Figure 7's inputs: 01_models + 03_derived -> 04_results/figure7_baseline_roc.

Figure 7 asks whether anything measurable at baseline identifies the knees that
will go on to develop osteoarthritis. It is a two-group classification: healthy
knees that stay healthy against knees that are KL0-1 at baseline and cross to KL2
or worse by 48 months. Mild-OA knees are not in the contrast, so this figure's
population is roughly two thirds of the matrix it reads -- see ``n`` below.

Rather than one classifier, it is a **sweep**: every combination of a shape
representation, a pressure representation and a dimensionality reducer, fitted
independently under each of the ten gait conditions, because that is how to
report a negative result. A single poorly chosen feature set that fails to
predict says nothing; forty-four of them, the best of which reaches 0.69, says
the signal is not there to be found.

The grid
--------
=============  ===========================================================
shape          ``none`` · ``thickness`` (per-vertex baseline cartilage
               thickness) · ``xyz_thick`` (per-vertex cartilage coordinates
               and thickness) · ``latents`` (the knee's shape-model code,
               femur 1024 + tibia 512 + patella 512)
pressure       ``none`` · ``cum`` (cumulative) · ``max`` (peak), both
               per-vertex on the reference surface
reducer        ``pca20`` · ``pca50`` · ``pca_max`` (as many components as the
               training fold has rank for) · ``none`` (raw features, L2)
=============  ===========================================================

4 x 3 x 4 = 48, less the four cells with neither shape nor pressure = **44
configurations**, each fitted under **10 gait conditions** = 440 cells. Every one
is computed. Nothing is copied between conditions.

For ``shape='latents'`` with no pressure, copying would give the same numbers --
the shape-model code is a property of the knee's anatomy, built once and
uniformly scaled per gait, so those features are identical across conditions by
construction. For ``latents + pressure`` it would not: pressure is gait-specific,
and copied cells would make the radar's cross-gait agreement an artifact of the
copying. If this needs to be parallelised differently, parallelise over
conditions; never fill one in from another.

Reading the cross-gait agreement
--------------------------------
The shape-only configurations are gait-invariant *by construction*. Their
agreement across conditions is arithmetic, not biology, and is evidence of
nothing. The meaningful cross-gait comparison is on the pressure-bearing
configurations, and those stay weak in every condition.

What it produces
----------------
``auroc.csv``
    One row per (gait, shape, pressure, reducer): the mean and standard
    deviation of the area under the ROC curve across the ten cross-validation
    repeats, the pooled-score AUROC, a feature count, and the population it was
    fitted on. **This is the file to quote a number from** -- ``n`` differs per
    configuration, and reading it off a panel loses that.

    ``n_features`` is the size of the row's *first* feature block: the shape
    block where the configuration has one, the pressure block otherwise. It is
    therefore not the width of the fitted matrix for a configuration that
    combines both. The full per-block breakdown is in the matching weights file,
    which is where to look if the total is what you want.
``cohort_populations.csv``
    Per gait condition: matrix size, classifier population, and how many knees
    each exclusion rule removed.
``dropped.csv``
    Every excluded knee with its reason, so the exclusions are auditable from
    ``04_results`` alone.
``meta.json``
    Provenance: quality-control era and table, the matrices read, the
    cross-validation settings, the grid, the cell count, and the machine.
``weights/{shape}_{pressure}_{reducer}_{gait}.npz``
    Fold-averaged coefficients in original feature space, the standardiser, the
    per-repeat AUROCs, and the knees and labels fitted. Not read by the figure;
    kept because they are what a "which anatomy does the classifier point at"
    follow-up needs, and refitting to recover them costs hours.
``04_results/inclusion/figure7_baseline_roc_{gait}.csv``
    The shared inclusion manifest, one per condition.

The AUROCs depend on the CPU type, at the third decimal
-------------------------------------------------------
The same inputs and code give bit-identical results on one CPU type, but differ
between types by up to 0.0036 AUROC per cell. Ill-conditioned fits amplify
floating-point differences: ``pca_max`` keeps nearly as many components as
knees, and ``none`` fits the raw features. The reported table is derived on
Broadwell (Xeon E5-2699 v4) and reproduces bit-for-bit only on that CPU type, so
``meta.json`` records the CPU model and host.

Population
----------
Exclusions come only from the shared quality-control table (``figures/shared/qc``):
per-simulation union outliers for that condition, and chronic knees -- flagged
under five or more conditions, plus the manual list -- dropped entirely, which is
the right level for a per-knee classifier. There is no hardcoded subject list.

n is 483-492 per condition, not the 724-741 of the matrix, because two of the
three disease groups make the contrast. It is smaller again for ``xyz_thick``,
which additionally needs a knee to be in the shape matrix; that per-configuration
count is in ``auroc.csv``.

The shape-model latents cache
-----------------------------
The latents are three small files per knee under ``01_models``. No gait condition
carries its own -- the measured conditions reuse the reference-size reconstruction
and scale it -- so all ten read the same ~1,500 files. Uncached, ten parallel
workers issue about 15,000 small reads at once, which stalls on network storage.
So the parent reads them once into a 6 MB cache and
then forks, and the workers inherit it copy-on-write. That is also why this is one
process pool and not an array job: an array has no shared parent to inherit from.

Reproducibility
---------------
Seeded end to end, so two runs on the same data and CPU type give the same
table: the fold assignment and the principal-component analysis are both seeded
(see ``_fit_fold``). ``none`` and ``pca_max`` do not depend on the seed at all:
``pca_max`` keeps ``n_train - 1`` components, above scikit-learn's threshold for
switching to the deterministic solver. ``pca20`` and ``pca50`` use the randomized
solver, so their values depend on the seed: two draws differ by a mean absolute
2.8e-4 (``pca20``) and 1.9e-3 (``pca50``) AUROC, with no systematic shift. Nothing the figure draws comes from ``pca20`` or
``pca50``: the panels report the unreduced model, and the reduction sweep is a
sensitivity analysis quoted to three decimals.

Usage
-----
    python pipeline/derive_figure7_data.py                    # all ten, 10 workers
    python pipeline/derive_figure7_data.py --workers 4
    python pipeline/derive_figure7_data.py --gait default Subject_102
    python pipeline/derive_figure7_data.py --rebuild-latents-cache

The paper's run took 3 h 9 min on ten workers with 64 GB requested (peak memory was
not recorded). Most of it is the twelve
``xyz_thick`` configurations per condition, which hold two 20,000-vertex
coordinate blocks per knee.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
from itertools import product
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import ConfigError, Knee, load_config  # noqa: E402
from comak_knee_oa.manifest import cpu_model  # noqa: E402
from comak_knee_oa.paths import UndeclaredRunError, stage_dir  # noqa: E402
from shared import qc  # noqa: E402

FIGURE = "figure7_baseline_roc"

# Where each knee's reconstruction -- and so its shape-model latents -- lives is
# the reconstruction config's run.output_dir. It carries the run's name, so it is
# read from there rather than spelled here.
RECONSTRUCTION_CONFIG = _REPO / "config" / "reconstruction.json"
LATENT_BONES = ("femur", "tibia", "patella")

# The two disease groups the classifier separates. Mild OA is in the matrices and
# is not in this contrast: the question is who *develops* disease.
HEALTHY, INCIDENT = "kl0_1_healthy", "kl0_1_progressor"

SHAPE_SOURCES = ("none", "thickness", "xyz_thick", "latents")
PRESSURE_SOURCES = ("none", "cum", "max")
REDUCERS = ("pca20", "pca50", "pca_max", "none")

# Cross-validation. Seeded, so the sweep is reproducible run to run; the reported
# dispersion is across the ten repeats, which is fold-assignment stability rather
# than sampling uncertainty about the population.
N_SPLITS, N_REPEATS, SEED, C_L2 = 5, 10, 42, 1.0

# A vertex is used only where at least half the knees have data there. The
# reference surface extends slightly beyond where every subject has cartilage, so
# its rim carries a few columns that are mostly absent and would otherwise become
# a feature encoding "this knee had coverage here".
MIN_VERTEX_COVERAGE = 0.5

# knee -> (2048,) latents. Populated in the parent, inherited by the workers.
_LATENTS: dict[str, np.ndarray] | None = None


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def combined_matrix(gait: str) -> Path:
    """``03_derived/combined/combined_data_<gait>.npz`` -- one condition's data."""
    path = stage_dir("derived", "combined") / f"combined_data_{gait}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"No combined matrix for gait condition {gait!r}: {path}\n\n"
            "Build it with:\n"
            "    python pipeline/combine_reference_data.py matrices\n\n"
            "Or download the published 03_derived stage."
        )
    return path


def shape_matrix() -> Path:
    """``03_derived/combined/combined_data_shape.npz``.

    Per-vertex cartilage coordinates on the reference surface, which only the
    ``xyz_thick`` configurations read. Shape is gait-invariant, so there is one
    file rather than one per condition.
    """
    return stage_dir("derived", "combined") / "combined_data_shape.npz"


def group_map() -> dict[str, str]:
    """``{knee: disease group}`` from the vendored cohort definition."""
    path = stage_dir("inputs") / "groups.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cohort group definition not found: {path}\n\n"
            "It is a vendored input; download the published 00_inputs stage."
        )
    groups = json.loads(path.read_text())
    return {knee: group for group, knees in groups.items() for knee in knees}


def models_tree() -> Path:
    """The 01_models tree holding every knee's reconstruction (``run.output_dir`` less ``{knee}``).

    The config names the run; the v2 data root declares none, and keeps its
    reconstructions under ``cohorts.V2_MODELS_RUN``.
    """
    try:
        pattern = load_config(RECONSTRUCTION_CONFIG).get("run", "output_dir")
    except UndeclaredRunError:
        return stage_dir("models", cohorts.V2_MODELS_RUN)
    head, _, tail = pattern.rpartition("/")
    if tail != "{knee}":
        raise ConfigError(
            f"{RECONSTRUCTION_CONFIG}: run.output_dir is {pattern!r}; expected a tree "
            "ending in /{knee}, one directory per knee"
        )
    return load_config(RECONSTRUCTION_CONFIG).resolve(head, where="run.output_dir")


def latents_dir(knee_name: str) -> Path:
    return models_tree() / Knee.parse(knee_name).directory / "geometries_nsm_similarity"


def read_latents(knee_name: str):
    """One knee's concatenated shape-model code, or None if any bone is absent."""
    base = latents_dir(knee_name)
    pieces = []
    for bone in LATENT_BONES:
        path = base / bone / f"{bone}_latent.npy"
        if not path.is_file():
            return None
        pieces.append(np.load(path).reshape(-1))
    return np.concatenate(pieces).astype(np.float32)


def build_latents_cache(knees, path: Path, rebuild=False) -> Path:
    """Read every knee's latents once into a single npz. See the module docstring."""
    if path.exists() and not rebuild:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    kept, matrices = [], []
    for knee in sorted(set(knees)):
        latents = read_latents(knee)
        if latents is not None:
            kept.append(knee)
            matrices.append(latents)
    if not kept:
        raise FileNotFoundError(
            f"No shape-model latents found under {models_tree()}.\n\n"
            "Figure 7's best-performing feature set reads them, so this would "
            "produce a sweep missing its headline result. Download the "
            "published 01_models stage, or re-run the reconstruction."
        )
    np.savez_compressed(path, subjects=np.array(kept), latents=np.stack(matrices))
    print(f"  latents cache: {len(kept)} knees x {matrices[0].size} -> {path.name}")
    return path


def load_latents_cache(path: Path) -> dict:
    """Populate the module-level cache. Call in the parent process, before forking."""
    global _LATENTS
    if _LATENTS is None:
        with np.load(path, allow_pickle=True) as data:
            _LATENTS = dict(zip(list(data["subjects"]), data["latents"]))
    return _LATENTS


def subject_latents(knee_name: str):
    """Cache first, disk as a fallback for a knee the cache was built without."""
    if _LATENTS:
        hit = _LATENTS.get(knee_name)
        if hit is not None:
            return hit
    return read_latents(knee_name)


# --------------------------------------------------------------------------- #
# Population                                                                  #
# --------------------------------------------------------------------------- #
def population(data, groups, qc_tag, outliers, chronic):
    """The quality-controlled two-group population for one condition.

    The single place this figure decides who is in. Returns
    ``(indices, labels, knees, dropped)`` where ``indices`` index into the
    matrix's own ``subjects`` array -- so the ``n`` this figure quotes comes from
    the matrix that produced it, never from a cohort list or a simulation ledger.

    Not being in the contrast is not an exclusion: a Mild-OA knee is absent
    because the question is about *incident* disease, and it is not counted as a
    drop.
    """
    knees = [str(s) for s in data["subjects"]]
    indices, labels, dropped = [], [], []
    for index, knee in enumerate(knees):
        group = groups.get(knee)
        if group not in (HEALTHY, INCIDENT):
            continue
        if knee in chronic:
            dropped.append((knee, group, "chronic knee"))
            continue
        if (qc_tag, knee) in outliers:
            dropped.append((knee, group, "quality control, union policy"))
            continue
        indices.append(index)
        labels.append(0 if group == HEALTHY else 1)
    return (np.array(indices, dtype=int), np.array(labels, dtype=int),
            knees, dropped)


# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #
def _covered_columns(has_data) -> np.ndarray:
    return has_data.mean(axis=0) >= MIN_VERTEX_COVERAGE


def pressure_features(data, indices, kind):
    """Per-vertex contact pressure, femur then tibia, side by side."""
    blocks = []
    for bone in ("femur", "tibia"):
        values = data[f"{bone}_pressure_{kind}"][indices].astype(np.float32)
        values = values[:, _covered_columns(data[f"{bone}_has_pressure"][indices])]
        values[np.isnan(values)] = 0.0
        blocks.append(values)
    return np.concatenate(blocks, axis=1)


def thickness_features(data, indices):
    """Per-vertex baseline cartilage thickness, femur then tibia."""
    blocks = []
    for bone in ("femur", "tibia"):
        values = data[f"{bone}_baseline_thickness"][indices].astype(np.float32)
        values = values[:, _covered_columns(data[f"{bone}_has_thinning"][indices])]
        values[np.isnan(values)] = 0.0
        blocks.append(values)
    return np.concatenate(blocks, axis=1)


def xyz_thick_features(data, indices, knees):
    """Per-vertex (x, y, z, thickness) -- cartilage geometry and how thick it is.

    Returns ``(X, keep)`` where ``keep`` selects the entries of ``indices`` whose
    knee is in the shape matrix. That matrix covers slightly fewer knees than the
    per-condition matrices, so this configuration has its own, smaller ``n``; the
    caller must apply ``keep`` to the labels and to any other feature block, or
    the rows would stop lining up without any error.
    """
    path = shape_matrix()
    if not path.is_file():
        return None, None

    with np.load(path, allow_pickle=True) as shape:
        shape_knees = [str(s) for s in shape["subjects"]]
        position = {knee: i for i, knee in enumerate(shape_knees)}

        keep = np.array([knees[i] in position for i in indices])
        if not keep.any():
            return None, None
        kept = indices[keep]
        rows = np.array([position[knees[i]] for i in kept])

        blocks = []
        for bone in ("femur", "tibia"):
            if f"{bone}_shape_xyz" not in shape:
                return None, None
            xyz = shape[f"{bone}_shape_xyz"][rows].astype(np.float32)
            thickness = data[f"{bone}_baseline_thickness"][kept].astype(np.float32)
            covered = _covered_columns(data[f"{bone}_has_thinning"][kept])
            xyz = xyz[:, covered, :]
            thickness = thickness[:, covered]
            thickness[np.isnan(thickness)] = 0.0
            xyz[np.isnan(xyz)] = 0.0
            block = np.concatenate([xyz, thickness[:, :, None]], axis=2)
            blocks.append(block.reshape(block.shape[0], -1))
    return np.concatenate(blocks, axis=1), keep


def latent_features(indices, knees):
    """The knee's shape-model code. None if any knee in the population lacks one."""
    rows = []
    for i in indices:
        latents = subject_latents(knees[i])
        if latents is None:
            return None
        rows.append(latents)
    return np.stack(rows)


def build_features(data, indices, labels, knees, shape, pressure):
    """``(X, y, knees_used, blocks)`` for one configuration, or None if unavailable."""
    blocks, matrices = [], []
    keep = np.ones(len(indices), dtype=bool)

    if shape == "thickness":
        X = thickness_features(data, indices)
        matrices.append(X)
        blocks.append({"block": "thickness", "n_features": int(X.shape[1])})
    elif shape == "xyz_thick":
        X, mask = xyz_thick_features(data, indices, knees)
        if X is None:
            return None
        keep &= mask
        matrices.append(X)
        blocks.append({"block": "shape_xyz_thickness", "n_features": int(X.shape[1])})
    elif shape == "latents":
        X = latent_features(indices, knees)
        if X is None:
            return None
        matrices.append(X)
        blocks.append({"block": "shape_model_latents", "n_features": int(X.shape[1])})

    if pressure != "none":
        Xp = pressure_features(data, indices, pressure)
        if not keep.all():
            Xp = Xp[keep]          # follow whatever the shape block dropped
        matrices.append(Xp)
        blocks.append({"block": f"pressure_{pressure}", "n_features": int(Xp.shape[1])})

    if not matrices:
        return None
    used = [knees[i] for i, k in zip(indices, keep) if k]
    return np.concatenate(matrices, axis=1), labels[keep], used, blocks


# --------------------------------------------------------------------------- #
# Fitting                                                                     #
# --------------------------------------------------------------------------- #
def _n_components(reducer, n_train, n_features):
    """Components to keep, or None for no reduction.

    ``pca_max`` takes every component the training fold has rank for, which for
    these feature counts is always ``n_train - 1``. It is in the grid to separate
    two explanations of why PCA-20 does worse than raw features: discarding
    components, or the projection itself. It matches raw closely, so it is
    the discarding.
    """
    if reducer == "none":
        return None
    if reducer == "pca_max":
        return min(n_train - 1, n_features)
    requested = int(reducer.removeprefix("pca"))
    return min(requested, n_train - 1, n_features)


def _fit_fold(X, y, train, test, reducer):
    """Fit one fold; returns ``(test scores, coefficients in feature space)``.

    Standardising inside the fold, on the training rows only, is what keeps the
    held-out rows genuinely held out -- scaling on the full matrix first leaks the
    test fold's distribution into the fit.

    The principal-component analysis is seeded. At these shapes -- a few hundred
    rows, tens of thousands of columns, twenty components -- scikit-learn's
    ``auto`` solver selects its randomized truncated SVD, which draws from the
    global random stream unless told otherwise; unseeded, one cell gave 0.57656,
    0.57688 and 0.57729 over three runs on identical data. Seeding makes the
    number reproducible.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    mean = X[train].mean(axis=0)
    sd = X[train].std(axis=0) + 1e-9
    X_train = (X[train] - mean) / sd
    X_test = (X[test] - mean) / sd

    n_components = _n_components(reducer, X_train.shape[0], X_train.shape[1])
    if n_components is not None:
        if n_components < 1:
            return None
        pca = PCA(n_components=n_components, random_state=SEED)
        Z_train = pca.fit_transform(X_train)
        Z_test = pca.transform(X_test)
        model = LogisticRegression(C=C_L2, max_iter=2000, solver="lbfgs")
        model.fit(Z_train, y[train])
        coefficients = pca.components_.T @ model.coef_[0]
    else:
        model = LogisticRegression(C=C_L2, max_iter=4000, solver="lbfgs")
        model.fit(X_train, y[train])
        Z_test = X_test
        coefficients = model.coef_[0]

    return model.predict_proba(Z_test)[:, 1], coefficients, mean, sd


def fit_config(X, y, reducer):
    """Repeated stratified cross-validation for one configuration.

    Two AUROCs are reported and they answer different questions. The *pooled* one
    scores every knee by its mean out-of-fold prediction and takes one curve
    through all of them. The *per-repeat* mean is one curve per repeat, averaged,
    and its spread across repeats is the only dispersion this design supports --
    it measures how much the fold assignment moves the answer, and is not a
    confidence interval on the population. Both are written out; the figure uses
    the per-repeat mean, and its own error bars are computed across gait
    conditions, which is a third thing again.
    """
    from sklearn.metrics import auc, roc_curve
    from sklearn.model_selection import RepeatedStratifiedKFold

    splitter = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                       random_state=SEED)
    folds = list(splitter.split(X, y))

    pooled = np.zeros(len(y))
    counts = np.zeros(len(y), dtype=int)
    coefficients = np.zeros(X.shape[1])
    scaler_mean = np.zeros(X.shape[1])
    scaler_sd = np.zeros(X.shape[1])
    per_repeat = []

    for repeat in range(N_REPEATS):
        repeat_scores = np.zeros(len(y))
        for train, test in folds[repeat * N_SPLITS:(repeat + 1) * N_SPLITS]:
            fitted = _fit_fold(X, y, train, test, reducer)
            if fitted is None:
                return None
            scores, coefficient, mean, sd = fitted
            repeat_scores[test] = scores
            pooled[test] += scores
            counts[test] += 1
            coefficients += coefficient
            scaler_mean += mean
            scaler_sd += sd
        fpr, tpr, _ = roc_curve(y, repeat_scores)
        per_repeat.append(auc(fpr, tpr))

    n_folds = N_SPLITS * N_REPEATS
    fpr, tpr, _ = roc_curve(y, pooled / np.maximum(counts, 1))
    return {
        "auroc_pooled": float(auc(fpr, tpr)),
        "auroc_repeats": np.array(per_repeat),
        "coefficients": coefficients / n_folds,
        "scaler_mean": scaler_mean / n_folds,
        "scaler_sd": scaler_sd / n_folds,
    }


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #
def configurations():
    """The 44 configurations, in a stable order."""
    return [(shape, pressure, reducer)
            for shape, pressure, reducer
            in product(SHAPE_SOURCES, PRESSURE_SOURCES, REDUCERS)
            if not (shape == "none" and pressure == "none")]


def process_one_gait(job):
    """The whole configuration grid for one gait condition. One pool worker.

    Ten conditions in series is about two days; one condition is a few hours, so
    the pool is over conditions. The matrix is loaded once here rather than once
    per configuration, which is a 44-fold difference on a quarter-gigabyte file.
    """
    gait, qc_tag, outliers, chronic = job
    groups = group_map()
    out_dir = stage_dir("results", FIGURE)
    weights_dir = out_dir / "weights"

    with np.load(combined_matrix(gait), allow_pickle=True) as data:
        indices, labels, knees, dropped = population(
            data, groups, qc_tag, outliers, chronic)
        summary = {
            "gait": gait, "qc_tag": qc_tag,
            "matrix": combined_matrix(gait).name,
            "n_in_matrix": len(knees),
            "n_knees": int(len(indices)),
            "n_healthy": int((labels == 0).sum()),
            "n_incident": int((labels == 1).sum()),
            "n_dropped_simulation": sum(1 for d in dropped if d[2].startswith("quality")),
            "n_dropped_chronic": sum(1 for d in dropped if d[2].startswith("chronic")),
        }
        print(f"[{gait}] matrix {summary['n_in_matrix']} -> population "
              f"{summary['n_knees']} ({summary['n_healthy']} healthy / "
              f"{summary['n_incident']} incident); dropped "
              f"{summary['n_dropped_simulation']} by quality control + "
              f"{summary['n_dropped_chronic']} chronic", flush=True)

        rows = []
        for shape, pressure, reducer in configurations():
            built = build_features(data, indices, labels, knees, shape, pressure)
            if built is None:
                print(f"  [{gait}] {shape}/{pressure}/{reducer}: features "
                      "unavailable", flush=True)
                continue
            X, y, used, blocks = built
            fitted = fit_config(X, y, reducer)
            if fitted is None:
                continue

            np.savez_compressed(
                weights_dir / f"{shape}_{pressure}_{reducer}_{gait}.npz",
                coefficients=fitted["coefficients"],
                scaler_mean=fitted["scaler_mean"], scaler_sd=fitted["scaler_sd"],
                auroc_pooled=fitted["auroc_pooled"],
                auroc_repeats=fitted["auroc_repeats"],
                knees=np.array(used), labels=y,
                blocks=json.dumps(blocks))

            repeats = fitted["auroc_repeats"]
            rows.append({
                "gait": gait, "shape": shape, "pressure": pressure,
                "reducer": reducer,
                "auroc_mean": float(repeats.mean()),
                "auroc_sd": float(repeats.std()),
                "auroc_pooled": fitted["auroc_pooled"],
                "auroc_repeat_p2_5": float(np.percentile(repeats, 2.5)),
                "auroc_repeat_p97_5": float(np.percentile(repeats, 97.5)),
                "n_features": blocks[0]["n_features"],
                "n_knees": int(len(y)),
                "n_healthy": int((y == 0).sum()),
                "n_incident": int((y == 1).sum()),
            })

    print(f"[{gait}] done: {len(rows)}/{len(configurations())} configurations",
          flush=True)
    dropped_rows = [{"gait": gait, "knee": k, "group": g, "reason": r}
                    for k, g, r in dropped]
    included = [{"gait": gait, "knee": knees[i],
                 "group": groups.get(knees[i])} for i in indices]
    return gait, rows, summary, dropped_rows, included


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows, fields) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def derive(gaits=None, workers=10, rebuild_latents=False) -> Path:
    gaits = list(gaits or cohorts.ALL)
    out_dir = stage_dir("results", FIGURE, create=True)
    (out_dir / "weights").mkdir(exist_ok=True)

    # Resolve the quality-control keys from the table's own key set rather than
    # assuming a spelling, and load both exclusion levels once in the parent.
    qc_tags = cohorts.resolve_qc_tags(qc.known_cohort_tags(), gaits)
    outliers = qc.load_outlier_set(source="union", cohort_tags=qc_tags)
    chronic = qc.load_chronic_knees(min_gaits=5, source="union")
    print(f"  quality control: era {qc.ERA}, union policy, "
          f"{len(outliers)} flagged simulations, {len(chronic)} chronic knees")

    if not shape_matrix().is_file():
        print(f"  warning: {shape_matrix().name} absent -- the twelve 'cartilage "
              "shape + thickness' configurations will be reported as "
              "unavailable rather than silently skipped")

    knees = set()
    for gait in gaits:
        with np.load(combined_matrix(gait), allow_pickle=True) as data:
            knees.update(str(s) for s in data["subjects"])
    cache = build_latents_cache(knees, out_dir / "latents_cache.npz",
                               rebuild=rebuild_latents)
    load_latents_cache(cache)
    print(f"  latents loaded for {len(_LATENTS)} knees "
          f"(before forking {workers} workers)")

    jobs = [(gait, tag, outliers, chronic) for gait, tag in zip(gaits, qc_tags)]
    collected = {}
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import Pool
        with Pool(min(workers, len(jobs))) as pool:
            for result in pool.imap_unordered(process_one_gait, jobs):
                collected[result[0]] = result
    else:
        for job in jobs:
            result = process_one_gait(job)
            collected[result[0]] = result

    # Re-key into the canonical condition order. The pool yields whichever
    # condition finishes first, and the figure styles conditions by row order --
    # the reference condition owns a reserved colour, and a completion-ordered
    # table hands it to whoever happened to finish first, differently on every
    # run. Ordering the artifact itself is what makes the figure reproducible.
    rows, populations, dropped, included = [], [], [], []
    for gait in gaits:
        if gait not in collected:
            continue
        _, config_rows, summary, dropped_rows, included_rows = collected[gait]
        rows.extend(config_rows)
        populations.append(summary)
        dropped.extend(dropped_rows)
        included.extend(included_rows)
        qc.write_inclusion_manifest(
            f"{FIGURE}_{gait}",
            [r for r in included_rows],
            policy=(f"era={qc.ERA} source=union chronic_min_gaits=5; "
                    f"contrast=healthy_vs_incident; matrix={summary['matrix']}; "
                    f"dropped_simulation={summary['n_dropped_simulation']} "
                    f"dropped_chronic={summary['n_dropped_chronic']}"),
        )

    write_csv(out_dir / "auroc.csv", rows, [
        "gait", "shape", "pressure", "reducer", "auroc_mean", "auroc_sd",
        "auroc_pooled", "auroc_repeat_p2_5", "auroc_repeat_p97_5",
        "n_features", "n_knees", "n_healthy", "n_incident"])
    write_csv(out_dir / "cohort_populations.csv", populations, [
        "gait", "qc_tag", "matrix", "n_in_matrix", "n_knees", "n_healthy",
        "n_incident", "n_dropped_simulation", "n_dropped_chronic"])
    write_csv(out_dir / "dropped.csv", dropped, ["gait", "knee", "group", "reason"])

    expected = len(configurations()) * len(gaits)
    meta = {
        "figure": FIGURE,
        "written_by": "pipeline/derive_figure7_data.py",
        "contrast": {"positive": INCIDENT, "negative": HEALTHY,
                     "note": "Mild OA is in the matrices and not in this "
                             "contrast; the question is who develops disease."},
        "qc": {"era": qc.ERA, "table": qc.unified_csv().name, "policy": "union",
               "chronic_min_gaits": 5,
               # Totals of dropped.csv, so "did QC actually apply?" is
               # answerable from this file alone -- a zero here means the
               # exclusion table was not applied and must not pass unnoticed.
               "n_dropped_rows": len(dropped),
               "n_dropped_simulation": sum(
                   1 for d in dropped if d["reason"].startswith("quality")),
               "n_dropped_chronic": sum(
                   1 for d in dropped if d["reason"] == "chronic knee"),
               "n_unique_knees_dropped": len({d["knee"] for d in dropped})},
        "groups": "00_inputs/groups.json",
        "shape_matrix": shape_matrix().name if shape_matrix().is_file() else None,
        "models_run": models_tree().name,
        # The AUROCs differ between CPU types (module docstring).
        "machine": {"cpu_model": cpu_model(), "host": socket.gethostname()},
        "gaits": gaits,
        "n_gaits": len(gaits),
        "reference_gait": cohorts.REFERENCE,
        "cross_validation": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS,
                             "seed": SEED, "penalty": "l2", "C": C_L2,
                             "pca_seeded": True},
        "grid": {"shape": list(SHAPE_SOURCES), "pressure": list(PRESSURE_SOURCES),
                 "reducer": list(REDUCERS),
                 "n_configurations": len(configurations())},
        "cells_expected": expected,
        "cells_computed": len(rows),
        "backfilled": 0,
        "note": ("Every cell computed under its own gait condition. Nothing is "
                 "copied between conditions -- see the module docstring."),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n  auroc.csv              {len(rows)} of {expected} cells")
    print(f"  cohort_populations.csv {len(populations)} conditions, n = "
          + ", ".join(str(p["n_knees"]) for p in populations))
    print(f"  dropped.csv            {len(dropped)} exclusions")
    if len(rows) != expected:
        print(f"  warning: {expected - len(rows)} cells missing -- see the "
              "'features unavailable' lines above; this is a real shortfall, "
              "not a rounding of the grid")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gait", nargs="+", default=None, choices=list(cohorts.ALL),
                        help="conditions to sweep (default: all ten)")
    parser.add_argument("--workers", type=int, default=10,
                        help="parallel condition workers (default 10 = one each)")
    parser.add_argument("--rebuild-latents-cache", action="store_true",
                        help="re-read every knee's latents from 01_models")
    args = parser.parse_args()

    out_dir = derive(gaits=args.gait, workers=args.workers,
                     rebuild_latents=args.rebuild_latents_cache)
    print(f"\nFigure 6 inputs in {out_dir}")
    print("Next: figures/figure7_baseline_roc.py")


if __name__ == "__main__":
    main()
