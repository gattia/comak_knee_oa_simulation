"""The shared contact-pressure quality-control table.

Every analysis that consumes COMAK contact pressure excludes artifact
simulations through this module. Not through a per-script threshold: the same
bad solve shows up as a spatial blow-up in the vertex maps and as a spike in the
time-series, and if each figure rolls its own filter they disagree about which
simulations exist.

Two levels of exclusion
-----------------------
per-simulation
    Drop the specific flagged ``(cohort_tag, knee)`` run. Correct for
    per-simulation analyses -- a knee can solve badly under one gait and fine
    under the other nine.
chronic knee
    A knee flagged under most gaits is a bad reconstruction rather than a bad
    solve, and is dropped entirely. Correct for per-knee analyses. Two sources,
    unioned: the derived count above, and a small hand-curated table for knees
    the detectors structurally cannot see -- see ``load_manual_chronic_knees``.

Policy is the union: a simulation flagged by *either* detector is excluded
everywhere. The two detectors agree only moderately (Jaccard about 0.5), so
intersection would let obvious artifacts through.

Synthetic simulations (Figure 3)
--------------------------------
The synthetic shape-axis simulations are not in this table -- they are not
cohort knees, so there is nothing to key them on. Figure 3 applies its own two
criteria at derivation time, documented in ``pipeline/derive_figure3_data.py``.
This module is what the cohort figures use.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

from comak_knee_oa.paths import data_root, declaration, run_name, stage_dir

__all__ = [
    "ERA", "ERAS", "CANONICAL_ERA", "UnknownCohortTagError",
    "unified_csv", "manual_chronic_csv", "known_cohort_tags",
    "load_outlier_set", "load_chronic_knees", "load_manual_chronic_knees",
    "is_flagged", "write_inclusion_manifest",
    "thickness_visit_exclusions_csv", "load_thickness_visit_exclusions",
]

# Which era's table is read. A root that declares no run (the v2 layout, README
# *Run names*) can hold two tables: 'v2', and 'v1' from that root's first
# simulation pass; COMAK_QC_ERA picks one, and CANONICAL_ERA is the default. On a
# root that declares a run (for example v3, or v3_g1) the era is that run -- the
# only table the root can hold -- and COMAK_QC_ERA may only repeat it. An
# explicit switch rather than "use the newest file that exists": an implicit
# fallback would return a plausible set of exclusions from the wrong era and
# raise nothing.
CANONICAL_ERA = "v2"
LEGACY_ERAS = ("v1", "v2")


def _resolve_era() -> str:
    requested = os.environ.get("COMAK_QC_ERA") or None
    try:
        decl = declaration()
    except FileNotFoundError:  # no data root on this machine: nothing to read anyway
        decl = None
    run = run_name() if decl and decl.get("run_name") else None
    if run is None:
        era = requested or CANONICAL_ERA
        if era not in LEGACY_ERAS:
            raise ValueError(
                f"COMAK_QC_ERA must be one of {LEGACY_ERAS} on a root that declares no "
                f"run, got {era!r}. A later era's table lives in its own run's root."
            )
        return era
    if requested not in (None, run):
        raise ValueError(
            f"the data root holds run {run!r}, so its QC era is {run!r}; COMAK_QC_ERA="
            f"{requested!r} names another run's table, which is not in this root"
        )
    return run


ERA = _resolve_era()
ERAS = LEGACY_ERAS + ((ERA,) if ERA not in LEGACY_ERAS else ())

# Column in the unified table per source. Union is the policy; the others exist
# so a sensitivity check can be run without hand-editing anything.
_SOURCE_COL = {
    "union": "flagged_union",
    "spatial": "flagged_spatial",
    "timeseries": "flagged_timeseries",
    "intersection": "flagged_intersection",
}


class UnknownCohortTagError(KeyError):
    """A cohort tag absent from the QC table.

    Raised rather than returning an empty set. An empty set is indistinguishable
    from "this cohort is clean", so a figure pointed at a cohort that was never
    quality-controlled would apply zero exclusions while its manifest recorded
    that the union policy had been applied.
    """


def unified_csv() -> Path:
    suffix = "" if ERA == "v1" else f"_{ERA}"
    return stage_dir("derived", "qc") / f"pressure_outliers_unified{suffix}.csv"


def manual_chronic_csv() -> Path:
    """The hand-curated drop-entirely table of the active data root.

    Each run's root has its own, and it starts empty: a knee is added only after
    that run's simulations have been inspected. A knee that never converged in
    one run can converge in the next, so entries are never carried forward. A
    root without the file has no manual entries.
    """
    return stage_dir("derived", "qc") / "chronic_knees_manual.csv"


def _rows() -> list[dict]:
    path = unified_csv()
    if not path.exists():
        raise FileNotFoundError(
            f"QC table not found: {path}\n\n"
            "Download the published 03_derived stage, which contains it, or "
            "build it from the simulations:\n"
            f"    python pipeline/derive_qc.py spatial --era {ERA}\n"
            f"    python pipeline/derive_qc.py timeseries --era {ERA}\n"
            f"    python pipeline/derive_qc.py unify --era {ERA}"
        )
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _is_true(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "t")


def known_cohort_tags() -> set[str]:
    """Cohort tags the table covers -- what ``load_outlier_set`` will accept."""
    return {r["cohort_tag"] for r in _rows()}


def load_outlier_set(cohort_tags=None, source="union", allow_unknown_tags=False):
    """``{(cohort_tag, knee), ...}`` to exclude.

    ``cohort_tags`` restricts to specific cohorts; unknown tags raise
    ``UnknownCohortTagError`` unless ``allow_unknown_tags`` is set, which should
    be a deliberate, commented decision at the call site.
    """
    try:
        column = _SOURCE_COL[source]
    except KeyError:
        raise KeyError(
            f"unknown source {source!r}; expected one of {sorted(_SOURCE_COL)}"
        ) from None

    rows = _rows()
    if cohort_tags is not None:
        wanted = set(cohort_tags)
        if not allow_unknown_tags:
            unknown = wanted - known_cohort_tags()
            if unknown:
                raise UnknownCohortTagError(
                    f"cohort tag(s) not in {unified_csv().name}: {sorted(unknown)}\n"
                    f"known tags: {sorted(known_cohort_tags())}\n\n"
                    "The table is keyed by (cohort_tag, gait, knee). A tag it "
                    "has never seen has not been quality-controlled, so no "
                    "exclusion set can be returned for it. Rebuild the table "
                    "for this cohort, or pass allow_unknown_tags=True if you "
                    "genuinely mean 'no exclusions'."
                )
        rows = [r for r in rows if r["cohort_tag"] in wanted]

    return {(r["cohort_tag"], r["knee"]) for r in rows if _is_true(r[column])}


def load_manual_chronic_knees(with_reasons=False):
    """Knees excluded entirely for reasons no detector can see.

    The derived list below counts how many cohorts flagged a knee's *pressure*,
    which can only ever find knees that produced pressure at all. A knee whose
    simulation never converged under any gait has no rows in the table, so it is
    invisible to every detector and would show up only as a smaller n, with no
    error -- the failure the shared table exists to prevent. Those go in a
    versioned CSV instead, and are unioned in here so they reach every inclusion
    manifest.

    Returns a set of knee names, or ``{knee: reason}`` when ``with_reasons``.
    """
    path = manual_chronic_csv()
    if not path.exists():
        return {} if with_reasons else set()
    with path.open() as handle:
        # '#' lines carry the rationale for a human reader, not data.
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        found = {
            row["knee"].strip(): (row.get("reason") or "").strip()
            for row in rows
            if (row.get("scope") or "all").strip() == "all"
        }
    return found if with_reasons else set(found)


def thickness_visit_exclusions_csv() -> Path:
    """Per-visit thickness exclusions (failed shape fits). Not a pressure table:
    it lives beside the pressure QC so every exclusion is in one place."""
    return stage_dir("derived", "qc") / "thickness_visit_exclusions.csv"


def load_thickness_visit_exclusions() -> set[tuple[str, str, str]]:
    """``{(knee, visit, bone), ...}`` whose per-visit thickness must not be used.

    Raises if the table is missing, unlike the manual-chronic table: an empty set
    would put the failed fits back into Figure 8 without an error.
    """
    path = thickness_visit_exclusions_csv()
    if not path.exists():
        raise FileNotFoundError(
            f"thickness visit exclusion table not found: {path}\n\n"
            "Download the published 03_derived stage, which contains it."
        )
    with path.open() as handle:
        rows = csv.DictReader(line for line in handle if not line.startswith("#"))
        return {(r["knee"].strip(), r["visit"].strip(), r["bone"].strip()) for r in rows}


def load_chronic_knees(min_gaits=5, source="union", include_manual=True) -> set[str]:
    """Knees to drop entirely: chronic pressure artifacts plus manual entries.

    Derived: flagged under at least ``min_gaits`` distinct cohorts. A knee that
    produces artifacts under most gaits has a bad reconstruction, not bad luck in
    one solve. Manual: ``load_manual_chronic_knees``. Pass
    ``include_manual=False`` only to inspect the derived half on its own -- an
    analysis that does so is choosing to include a knee that was excluded by
    hand, so it should say why at the call site.
    """
    column = _SOURCE_COL[source]
    per_knee = defaultdict(set)
    for row in _rows():
        if _is_true(row[column]):
            per_knee[row["knee"]].add(row["cohort_tag"])
    derived = {knee for knee, tags in per_knee.items() if len(tags) >= min_gaits}
    return derived | load_manual_chronic_knees() if include_manual else derived


def is_flagged(cohort_tag, knee, outlier_set=None) -> bool:
    """Whether one simulation is excluded. Pass ``outlier_set`` in a loop."""
    if outlier_set is None:
        outlier_set = load_outlier_set()
    return (cohort_tag, knee) in outlier_set


def write_inclusion_manifest(analysis_name, included, policy="union",
                             group_key="group"):
    """Record who was actually in an analysis, to ``04_results/inclusion/``.

    Written by every figure, so "which knees produced this panel" is answerable
    from disk months later instead of by re-reasoning about the filter chain.
    ``included`` is an iterable of dicts with at least a ``knee`` key.
    """
    rows = list(included)
    out_dir = stage_dir("results", "inclusion", create=True)
    path = out_dir / f"{analysis_name}.csv"

    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    for required in ("knee", group_key):
        if required not in seen:
            fields.append(required)
            seen.add(required)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = out_dir / f"{analysis_name}.txt"
    summary.write_text(
        f"analysis: {analysis_name}\n"
        f"qc era:   {ERA}\n"
        f"policy:   {policy}\n"
        f"included: {len(rows)}\n"
        f"table:    {unified_csv().relative_to(data_root())}\n"
    )
    return path
