"""Which knees each derived layer covers, against what it should cover.

Every stage after the simulations loops per knee and skips a knee whose input
is missing. So a layer built before the cohort changed -- a knee recovered, a
simulation finished late, a cache warmed mid-sweep -- does not fail. It covers
fewer knees, and downstream that is indistinguishable from "this condition
legitimately has fewer knees", and no code bug is involved. After any change to
membership or geometry, the question is not "does the code exclude this knee?"
but "which layers were built before the change?", and this answers it
mechanically.

Each layer declares its own reference -- the cohort, or the layers upstream of
it -- and is reported short only against that. Not every layer should hold
every knee: a simulation that never converged has no pressure to summarise, and
thinning exists only where the thickness fits do. A knee missing from a layer
*and* from its reference is consistent, not a defect, and a tool that reports
it as one is a tool nobody runs.

Three signals:

coverage
    Knees the reference has and the layer does not -- the real check. The
    reverse, knees the layer has and its reference does not, is reported too:
    the layer was built from an upstream that has since lost them.
staleness
    The layer is older than ``cohort.json``. Not a defect on its own (a change
    can remove knees), but a reason to look.
unaudited
    Derived trees and matrices on disk that no layer here collects. Without it
    this audit has the failure it exists to catch: a clean report would mean
    "the layers I know about are fine", not "the derived stage is fine".

Usage
-----
    python pipeline/audit_coverage.py                 # era = the root's run
    python pipeline/audit_coverage.py --verbose       # name the missing knees
    python pipeline/audit_coverage.py --layer step4_default transfer_default
    python pipeline/audit_coverage.py --era v2        # a root that declares no run

Exit status: 0 clean, 1 a layer is short, 3 a derived artifact has no layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_REPO / "figures"))

from comak_knee_oa.config import Knee  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402
from derive_pressure import (  # noqa: E402
    articular_surface_path,
    pressure_cache_dir,
    pressure_stats_path,
    transform_file,
)
from derive_qc import LEGACY_ERA, conditions, qc_key, resolve_era  # noqa: E402
from derive_thickness import BONES, thinning_rates_of_record_dir  # noqa: E402

# A layer: (name, collector, reference). The reference is None for the cohort,
# or a tuple of layer names whose intersection the layer should cover.
Layer = tuple

# Layers measured against the cohort that are legitimately smaller, and why.
# Anything else short against its reference is a finding.
EXPECTED_SHORTFALL = {
    "thinning_rates": "thickness fits exist for ~742 knees; a property of the imaging, not a cache",
}
SIMULATION_SHORTFALL = "convergence and solver-budget drops; a fact about the solve, not a cache"


# ------------------------------------------------------------- collectors ----


def _stat(path: Path) -> float:
    """mtime if the file exists, else 0."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _flat_members(directory: Path, suffix: str) -> tuple[set[str], float]:
    """Knees implied by ``{knee}{suffix}`` files in one directory, and the newest mtime."""
    members, newest = set(), 0.0
    if not directory.is_dir():
        return members, newest
    for entry in directory.iterdir():
        if entry.name.endswith(suffix):
            members.add(entry.name[: -len(suffix)])
            newest = max(newest, _stat(entry))
    return members, newest


def _both_bones(collect) -> tuple[set[str], float]:
    """A knee counts only if the layer holds it for both bones."""
    found, newest = None, 0.0
    for bone in BONES:
        members, t = collect(bone)
        found = members if found is None else found & members
        newest = max(newest, t)
    return found or set(), newest


def _per_knee(knees, path_of) -> tuple[set[str], float]:
    """Knees whose file(s) exist; ``path_of(knee)`` returns one path or a list."""
    members, newest = set(), 0.0
    for knee in knees:
        paths = path_of(knee)
        paths = paths if isinstance(paths, list) else [paths]
        times = [_stat(p) for p in paths]
        if all(times):
            members.add(knee.name)
            newest = max(newest, *times)
    return members, newest


def collect_cohort() -> tuple[set[str], float]:
    path = stage_dir("inputs") / "cohort.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["knees"]), _stat(path)


def collect_manual_exclusions() -> set[str]:
    """Knees every analysis drops, from the one shared table. Their absence downstream is expected."""
    from shared import qc

    return qc.load_manual_chronic_knees()


def build_layers(era: str, knees: list[Knee]) -> list[Layer]:
    layers: list[Layer] = [
        ("step1_geometry",
         lambda: _per_knee(knees, lambda k: [articular_surface_path(k, b) for b in BONES]), None),
        ("walked_out",
         lambda: _both_bones(lambda b: _flat_members(
             pressure_cache_dir() / b / "walked_out_xyz", f"_{b}_xyz.npy")), None),
        ("T_reg_to_647", collect_transforms, None),
        ("thinning_rates",
         lambda: _both_bones(lambda b: _flat_members(
             thinning_rates_of_record_dir(), f"_{b}_thinning_rate.vtk")), None),
    ]
    qc_tables = {
        "spatial": stage_dir("derived", "qc") / f"pressure_outliers_{era}.csv",
        "ts": stage_dir("derived", "qc") / f"pressure_outliers_timeseries_{era}.csv",
        "unified": stage_dir("derived", "qc") / f"pressure_outliers_unified_{era}.csv",
    }
    for condition, config in conditions():
        def results(k, config=config):
            return config.for_knee(k).path("run", "results_dir")

        key = qc_key(condition, era)
        c = condition
        layers += [
            (f"sims_{c}", lambda r=results: _per_knee(
                knees, lambda k: r(k) / "joint-mechanics" / "joint_mechanics.h5"),
             ("step1_geometry",)),
            (f"step4_{c}", lambda r=results: _per_knee(
                knees, lambda k: [pressure_stats_path(r(k), b) for b in BONES]),
             (f"sims_{c}",)),
            (f"transfer_{c}", lambda c=c: _both_bones(lambda b: _flat_members(
                stage_dir("derived", f"pressure_on_reference_{c}", b), f"_{b}_pressure_ref.vtk")),
             (f"step4_{c}", "walked_out", "T_reg_to_647")),
            (f"matrix_{c}", lambda c=c: collect_matrix(f"combined_data_{c}.npz"),
             (f"transfer_{c}", "thinning_rates")),
            (f"qc_spatial_{c}", lambda key=key: collect_qc(qc_tables["spatial"], key),
             (f"matrix_{c}",)),
            (f"qc_ts_{c}", lambda key=key: collect_qc(qc_tables["ts"], key),
             (f"sims_{c}", "step1_geometry", "T_reg_to_647")),
            (f"qc_unified_{c}", lambda key=key: collect_qc(qc_tables["unified"], key),
             (f"qc_ts_{c}",)),
        ]
        EXPECTED_SHORTFALL[f"sims_{c}"] = SIMULATION_SHORTFALL
    # Shape is gait-invariant and built with the reference condition's matrix.
    reference = conditions()[0][0]
    layers.append(("combined_shape", lambda: collect_matrix("combined_data_shape.npz"),
                   (f"matrix_{reference}",)))
    return layers


def collect_transforms() -> tuple[set[str], float]:
    path = transform_file()
    if not path.is_file():
        return set(), 0.0
    transforms = json.loads(path.read_text(encoding="utf-8"))["transforms"]
    found = set(transforms[BONES[0]]["T_reg_to_647"])
    for bone in BONES[1:]:
        found &= set(transforms[bone]["T_reg_to_647"])
    return found, _stat(path)


def collect_matrix(name: str) -> tuple[set[str], float]:
    import numpy as np

    path = stage_dir("derived", "combined") / name
    if not path.is_file():
        return set(), 0.0
    with np.load(path, allow_pickle=True) as data:
        return {str(s) for s in data["subjects"]}, _stat(path)


def collect_qc(path: Path, key: str) -> tuple[set[str], float]:
    if not path.is_file():
        return set(), 0.0
    with path.open(encoding="utf-8") as handle:
        return {r["knee"] for r in csv.DictReader(handle) if r["cohort_tag"] == key}, _stat(path)


# -------------------------------------------------------------- meta-check ----


def unaudited(era: str) -> list[tuple[str, str]]:
    """Derived trees and matrices on disk that no layer collects.

    Adding a condition or a new derived tree turns this red until a collector
    exists, or until the reason it needs none is recorded here.
    """
    names = [c for c, _ in conditions()]
    known_trees = {"pressure_on_reference"} | {f"pressure_on_reference_{c}" for c in names}
    known_trees |= {
        "combined",          # the matrix_* and combined_shape layers
        "qc",                # the qc_* layers
        "thickness_merged",  # thinning_rates (the tree of record)
        "thickness",         # upstream of thickness_merged; frozen, not re-derived here
        "region_labels",     # two fixed reference-surface files, not a per-knee artifact
    }
    known_matrices = {f"combined_data_{c}.npz" for c in names} | {"combined_data_shape.npz"}

    findings = []
    derived = stage_dir("derived")
    for entry in sorted(derived.iterdir()) if derived.is_dir() else []:
        if entry.is_dir() and entry.name not in known_trees:
            findings.append((f"03_derived/{entry.name}", "no layer audits this tree"))
    combined = stage_dir("derived", "combined")
    for entry in sorted(combined.glob("combined_data*.npz")) if combined.is_dir() else []:
        if entry.name not in known_matrices:
            findings.append((f"03_derived/combined/{entry.name}", "matrix with no layer"))
    return findings


# -------------------------------------------------------------------- cli ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audit_coverage.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", default=None,
                        help="which QC tables the qc_* layers read: the root's run (the "
                             f"default, on a root that declares one); '{LEGACY_ERA}' on a root "
                             "that declares no run (the earlier run's layout; see README, "
                             "*Run names*)")
    parser.add_argument("--verbose", "-v", action="store_true", help="name the missing knees")
    parser.add_argument("--layer", nargs="+", default=None,
                        help="report only these layers (their references are still computed)")
    parser.add_argument("--max-list", type=int, default=8)
    args = parser.parse_args(argv)
    args.era = resolve_era(args.era)

    cohort, cohort_mtime = collect_cohort()
    excluded = collect_manual_exclusions()
    print(f"data root        : {stage_dir('inputs').parent}")
    print(f"cohort of record : {len(cohort)} knees, built "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(cohort_mtime))}")
    if excluded:
        print(f"manual exclusions: {', '.join(sorted(excluded))} -- absence downstream is expected")
    print()

    knees = [Knee.parse(name) for name in sorted(cohort)]
    layers = build_layers(args.era, knees)
    if args.layer:
        unknown = set(args.layer) - {name for name, _, _ in layers}
        if unknown:
            print(f"no such layer(s): {sorted(unknown)}", file=sys.stderr)
            return 2

    found, rows = {}, []
    for name, collect, reference in layers:
        members, built = collect()
        found[name] = members
        if reference is None:
            expected_set, ref_label = cohort, "cohort"
        else:
            expected_set = set.intersection(*(found[r] for r in reference))
            ref_label = " ∩ ".join(reference)
        missing = expected_set - members
        real_missing = missing - excluded
        # An expected shortfall excuses some knees, never all of them: an empty
        # layer is a layer that was not built.
        expected_short = name in EXPECTED_SHORTFALL and bool(members)
        rows.append(dict(
            name=name, n=len(members), ref=ref_label, n_ref=len(expected_set),
            missing=missing, short=bool(real_missing) and not expected_short,
            expected=expected_short, only_excluded=bool(missing) and not real_missing,
            stale=0 < built < cohort_mtime,
            # Knees the layer holds and its reference does not: built from an
            # upstream that has since lost them. Reported, not failed -- the
            # missing direction is the one that shrinks an analysis unnoticed.
            extra=members - expected_set if reference is not None else set(),
        ))
    if args.layer:
        rows = [r for r in rows if r["name"] in set(args.layer)]

    width = max(len(r["name"]) for r in rows) + 1
    print(f"{'layer':<{width}} {'n':>5} {'ref n':>6} {'short':>6}  reference / flags")
    print("-" * (width + 60))
    for r in rows:
        flags = []
        if r["short"]:
            flags.append("SHORT")
        elif r["missing"] and r["only_excluded"]:
            flags.append("ok (manually excluded)")
        elif r["missing"] and r["expected"]:
            flags.append("ok (expected)")
        if r["extra"]:
            flags.append(f"+{len(r['extra'])} not in reference")
        if r["stale"]:
            flags.append("older than cohort.json")
        print(f"{r['name']:<{width}} {r['n']:>5} {r['n_ref']:>6} {len(r['missing']):>6}  "
              f"{r['ref']}{'  ' + ' · '.join(flags) if flags else ''}")
        if args.verbose and r["missing"]:
            for knee in sorted(r["missing"])[: args.max_list]:
                print(f"{'':<{width}}       - {knee}")
            if len(r["missing"]) > args.max_list:
                print(f"{'':<{width}}       … +{len(r['missing']) - args.max_list} more")

    findings = unaudited(args.era)
    print("\nderived artifacts no layer collects:")
    for where, why in findings:
        print(f"  {where}   ({why})")
    if not findings:
        print("  none")

    n_short = sum(r["short"] for r in rows)
    n_stale = sum(r["stale"] for r in rows)
    print(f"\n{len(rows)} layers · {n_short} short against their reference · "
          f"{n_stale} older than cohort.json · {len(findings)} unaudited")
    if n_short:
        print("A layer short against its own upstream was built before that upstream changed: "
              "rebuild it, then everything downstream of it.")
        return 1
    return 3 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
