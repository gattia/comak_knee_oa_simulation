#!/usr/bin/env python3
"""Derive Extended Data Table 1, the cohort table: 00_inputs + 03_derived + a local
OAI clinical download -> 04_results/table_ed1_cohort/.

The table answers three questions a reader of a longitudinal cohort study asks
before trusting anything else: who was studied (baseline demographics), what
happened to them radiographically (the incident-OA outcome that *defines* two of
the three groups), and how much data actually entered each analysis
(longitudinal thickness coverage, simulation convergence, quality-control
exclusions). It is a table rather than a figure, so this one step writes the
finished artefact (CSV + Markdown); there is no separate render script.

How the groups were defined, as coded at selection time
--------------------------------------------------------
Group membership is read from ``00_inputs/groups.json`` and never recomputed.
This step does, however, *re-derive the rule the groups were built with* and
fails if any knee violates it, so the table's definitions are checked against
the OAI source:

    healthy      baseline KL 0/1  and  no incident radiographic OA within 48 months
    incident     baseline KL 0/1  and  incident radiographic OA (KL >= 2) within 48 months
    mild OA      baseline KL 2

"Incident radiographic OA" is OAI's own adjudicated outcome, variable
``V99E{R,L}XIOA`` in ``General_ASCII/Outcomes99.txt`` ("KL0-1 at BL, incident
KL2 at NN mo"), not a per-visit KL comparison. Baseline KL is the central
reading, project 15, in ``kxr_sq_bu00.txt``. A knee whose adjudicated incident
OA falls at 72 or 96 months is therefore *healthy* here, by construction: the
window is 48 months for both groups.

Why the per-visit KL grades are not used for the outcome rows
--------------------------------------------------------------
OAI's per-visit KL files carry two reading projects. Project 15 read each visit
through 48 months; project 37 later re-read a subset longitudinally and, per
OAI's own documentation, revised the 48-month grade in a minority of knees. The
adjudicated outcome reflects those revisions; a naive per-visit comparison
against project 15 does not, and readers also regress grades between visits.
Tabulating the outcome from the adjudicated variable is what makes the table
agree with the definition.

What is, and is not, written
----------------------------
Only group-level aggregates. The OAI clinical files are participant-level data
released under the OAI data-use agreement and are neither vendored into the
data root nor echoed per knee into ``04_results``. To run this step, obtain
"Complete Data - ASCII" from https://nda.nih.gov/oai and point
``$OAI_CLINICAL_ROOT`` or ``<repo>/oai_clinical_root.txt`` at the unzipped
directory (see ``oai_clinical_root.txt.example``). A reader who only wants the
table can download the ``04_results`` stage of the data release, which contains it.

    python pipeline/derive_table_ed1_cohort.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
# The QC table is the one every pressure figure uses; its file is named by its
# era, which qc.py decides.
sys.path.insert(0, str(_REPO / "figures"))

from comak_knee_oa.paths import REPO_ROOT, data_root, stage_dir  # noqa: E402
from shared import qc as shared_qc  # noqa: E402

OUT_NAME = "table_ed1_cohort"

# Internal keys index every result file; the labels are the manuscript's.
GROUPS: "OrderedDict[str, str]" = OrderedDict(
    [("kl0_1_healthy", "Healthy"), ("kl0_1_progressor", "Incident OA"), ("kl2", "Mild OA")]
)

INCIDENT_WINDOW_MONTHS = 48
KL_READING_PROJECT = 15          # OAI central reading through 48 months
KL_PROJECT_RECODE = {42: 37}     # kXR_SQ_BU_Descrip.pdf: READPRJ 42 is project 37

# Thickness visits that enter the thinning-rate fit (V10 = 96 months is linked
# for tree completeness but excluded from the fit; see docs/data_provenance.md).
THICKNESS_VISITS_IN_FIT = ("V00", "V01", "V03", "V05", "V06", "V08")

_POINTER_FILE = REPO_ROOT / "oai_clinical_root.txt"
_ENV_VAR = "OAI_CLINICAL_ROOT"

# Files read from the OAI download, relative to its root.
OAI_FILES = {
    "enrollees": Path("General_ASCII") / "Enrollees.txt",
    "outcomes": Path("General_ASCII") / "Outcomes99.txt",
    "clinical00": Path("AllClinical_ASCII") / "AllClinical00.txt",
    "kxr00": Path("X-Ray Image Assessments_ASCII") / "Semi-Quant Scoring_ASCII" / "kxr_sq_bu00.txt",
}

RACE_LABELS = {"1": "White", "2": "Black", "3": "Asian", "0": "Other", ".": "Missing"}


# --------------------------------------------------------------------------- inputs
def oai_clinical_root() -> Path:
    """Locate the OAI clinical download, or say how to provide it."""
    tried: list[tuple[str, Path]] = []
    env = os.environ.get(_ENV_VAR)
    if env:
        tried.append((f"${_ENV_VAR}", Path(env).expanduser()))
    if _POINTER_FILE.is_file():
        for line in _POINTER_FILE.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                tried.append((str(_POINTER_FILE), Path(line).expanduser()))
                break
    for _source, path in tried:
        if path.is_dir():
            missing = [str(rel) for rel in OAI_FILES.values() if not (path / rel).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"OAI clinical root {path} is missing:\n  " + "\n  ".join(missing)
                    + "\nExpected the unzipped 'Complete Data - ASCII' release."
                )
            return path.resolve()
    lines = [f"    {p}   (from {s})" for s, p in tried] or ["    (nothing configured)"]
    raise FileNotFoundError(
        "Could not locate the OAI clinical download. Tried:\n" + "\n".join(lines)
        + f"\n\nSet {_ENV_VAR} or write the path into {_POINTER_FILE} (see "
        "oai_clinical_root.txt.example). The files are participant-level OAI data "
        "and are deliberately not part of the data root; download them from "
        "https://nda.nih.gov/oai. If you only need the table, the published "
        f"04_results/{OUT_NAME}/ already contains it."
    )


def load_groups() -> "OrderedDict[str, str]":
    """knee -> group key, from 00_inputs/groups.json; asserts the three groups exist."""
    path = stage_dir("inputs", "groups.json")
    raw = json.loads(path.read_text())
    missing = [g for g in GROUPS if g not in raw]
    if missing:
        raise KeyError(f"{path} lacks group(s) {missing}; has {sorted(raw)}")
    knee2g: "OrderedDict[str, str]" = OrderedDict()
    for g in GROUPS:
        for knee in raw[g]:
            if knee in knee2g:
                raise ValueError(f"{knee} appears in both {knee2g[knee]} and {g}")
            knee2g[knee] = g
    return knee2g


def read_oai(root: Path, key: str) -> pd.DataFrame:
    df = pd.read_csv(root / OAI_FILES[key], sep="|", low_memory=False)
    df.columns = df.columns.str.upper()
    return df


def _side_code(knee: str) -> int:
    side = knee.rsplit("_", 1)[1]
    return {"RIGHT": 1, "LEFT": 2}[side]


def _first_code(value) -> str:
    """OAI categorical strings look like '2: Female'; return the code before ':'."""
    s = str(value)
    return s.split(":", 1)[0].strip() if ":" in s else s.strip()


def baseline_and_outcome(root: Path, knee2g: dict) -> pd.DataFrame:
    """One row per knee: baseline demographics, project-15 baseline KL, adjudicated outcome."""
    enrol = read_oai(root, "enrollees").set_index("ID")
    clin = read_oai(root, "clinical00").set_index("ID")
    outc = read_oai(root, "outcomes").set_index("ID")
    kxr = read_oai(root, "kxr00")
    kxr["READPRJ"] = pd.to_numeric(kxr["READPRJ"], errors="coerce").replace(KL_PROJECT_RECODE)
    kxr = kxr[kxr["READPRJ"] == KL_READING_PROJECT].set_index(["ID", "SIDE"])

    rows = []
    for knee, group in knee2g.items():
        pid = int(knee.split("_", 1)[0])
        side = _side_code(knee)
        s = "R" if side == 1 else "L"
        e = enrol.loc[pid]
        c = clin.loc[pid]
        height_mm = pd.to_numeric(c["P01HEIGHT"], errors="coerce")
        weight_kg = pd.to_numeric(c["P01WEIGHT"], errors="coerce")
        kl = pd.to_numeric(kxr.loc[(pid, side), "V00XRKL"], errors="coerce") if (pid, side) in kxr.index else np.nan
        xioa = str(outc.loc[pid, f"V99E{s}XIOA"])
        m = re.search(r"incident KL2 at (\d+)", xioa)
        rows.append(
            dict(
                knee=knee,
                group=group,
                participant=pid,
                side=knee.rsplit("_", 1)[1],
                sex=_first_code(e["P02SEX"]),                 # 1 = male, 2 = female
                race=RACE_LABELS.get(_first_code(e["P02RACE"]), "Other"),
                age_y=pd.to_numeric(c["V00AGE"], errors="coerce"),
                height_mm=height_mm,
                weight_kg=weight_kg,
                bmi=weight_kg / (height_mm / 1000.0) ** 2,    # as in the manuscript's demographics
                womac_pain=pd.to_numeric(c[f"V00WOMKP{s}"], errors="coerce"),
                kl_baseline=kl,
                xioa=xioa,
                oa_months=int(m.group(1)) if m else np.nan,
            )
        )
    return pd.DataFrame(rows).set_index("knee")


def check_group_rule(df: pd.DataFrame) -> dict:
    """Re-derive the selection rule from the OAI files and fail on any violation."""
    kl_lt2 = df.kl_baseline < 2
    incident_in_window = df.oa_months <= INCIDENT_WINDOW_MONTHS
    rule = {
        "kl0_1_healthy": kl_lt2 & ~incident_in_window.fillna(False),
        "kl0_1_progressor": kl_lt2 & incident_in_window.fillna(False),
        "kl2": df.kl_baseline == 2,
    }
    violations = {}
    for g, mask in rule.items():
        bad = df.index[(df.group == g) & ~mask].tolist()
        if bad:
            violations[g] = bad
    if violations:
        raise AssertionError(
            "groups.json does not reproduce from the OAI files under the documented rule: "
            + json.dumps(violations, indent=1)
        )
    return {g: int(((df.group == g) & mask).sum()) for g, mask in rule.items()}


# --------------------------------------------------------------------------- data analysed
def thickness_coverage(knee2g: dict) -> pd.DataFrame:
    """Per knee: has a thinning rate (both bones) and how many fit visits have a thickness map."""
    rates = stage_dir("derived", "thickness_merged", "thinning_rates")
    maps = stage_dir("derived", "thickness_merged", "thickness_on_reference")
    if not rates.is_dir() or not maps.is_dir():
        raise FileNotFoundError(
            f"Need {rates} and {maps}. Download the published 03_derived stage, or run "
            "pipeline/derive_thickness.py."
        )
    rows = []
    for knee in knee2g:
        has_rate = all((rates / f"{knee}_{bone}_thinning_rate.vtk").is_file() for bone in ("femur", "tibia"))
        n_visits = sum((maps / v / f"{knee}_femur_thickness_ref.npy").is_file() for v in THICKNESS_VISITS_IN_FIT)
        rows.append(dict(knee=knee, has_rate=has_rate, n_fit_visits=n_visits))
    df = pd.DataFrame(rows).set_index("knee")
    if df.has_rate.sum() == 0:
        raise RuntimeError(f"no thinning-rate files matched under {rates}; wrong data root?")
    return df


def simulation_coverage(knee2g: dict) -> tuple[pd.DataFrame, int]:
    """Per knee: converged simulations and QC-excluded simulations, from the unified QC table."""
    path = shared_qc.unified_csv()
    if not path.is_file():
        raise FileNotFoundError(f"Need {path}. Download the published 03_derived stage.")
    qc = pd.read_csv(path)
    n_conditions = int(qc["cohort_tag"].nunique())
    qc = qc[qc.knee.isin(knee2g)]
    flagged = qc["flagged_union"].astype(str).str.lower().isin({"true", "1"})
    per = pd.DataFrame(
        {"converged": qc.groupby("knee").size(), "excluded": flagged.groupby(qc.knee).sum()}
    ).reindex(list(knee2g)).fillna(0).astype(int)
    per["analysed"] = per.converged - per.excluded
    return per, n_conditions


# --------------------------------------------------------------------------- the table
def _ms(x: pd.Series, nd: int = 1) -> str:
    return f"{x.mean():.{nd}f} ({x.std(ddof=1):.{nd}f})"


def build_table(df: pd.DataFrame, thick: pd.DataFrame, sims: pd.DataFrame, n_conditions: int):
    """Row label -> {group label: cell}. Also returns footnotes and per-group scalars for meta."""
    table: "OrderedDict[str, dict]" = OrderedDict()
    meta_groups: dict = {}

    def put(label, fn):
        table[label] = {GROUPS[g]: fn(g) for g in GROUPS}

    sub = lambda g: df[df.group == g]  # noqa: E731
    put("**Baseline**", lambda g: "")
    put("Knees (participants)ᵃ", lambda g: f"{len(sub(g))} ({sub(g).participant.nunique()})")
    put("Laterality, L:R", lambda g: f"{(sub(g).side == 'LEFT').sum()}:{(sub(g).side == 'RIGHT').sum()}")
    put("Sex, F:M", lambda g: f"{(sub(g).sex == '2').sum()}:{(sub(g).sex == '1').sum()}")
    put(
        "Race, White / Black / Asian / Other / missing",
        lambda g: " / ".join(str((sub(g).race == r).sum()) for r in ("White", "Black", "Asian", "Other", "Missing")),
    )
    put("Age, years", lambda g: _ms(sub(g).age_y))
    put("BMI, kg/m²", lambda g: _ms(sub(g).bmi))
    put("WOMAC pain (0–20)", lambda g: _ms(sub(g).womac_pain))
    put("Baseline KL, 0 / 1 / 2", lambda g: " / ".join(str(int((sub(g).kl_baseline == k).sum())) for k in (0, 1, 2)))

    put(f"**Radiographic outcome (OAI adjudicated incident KL ≥ 2)**ᵇ", lambda g: "")

    def outcome(g):
        s = sub(g)
        if g == "kl2":
            return f"KL ≥ 2 at baseline ({len(s)})"
        months = s.oa_months.dropna().astype(int)
        if g == "kl0_1_progressor":
            vc = months.value_counts().sort_index()
            return ", ".join(f"{m} mo ({n})" for m, n in vc.items())
        none = int(s.oa_months.isna().sum())
        late = months[months > INCIDENT_WINDOW_MONTHS].value_counts().sort_index()
        return f"none within {INCIDENT_WINDOW_MONTHS} mo ({len(s)}); of these, incident at " + ", ".join(
            f"{m} mo ({n})" for m, n in late.items()
        ) + f"; never ({none})"

    put("Incident radiographic OA, months after baseline (n)", outcome)

    put("**Data analysed**", lambda g: "")
    tk = lambda g: thick.loc[[k for k in thick.index if df.loc[k, 'group'] == g]]  # noqa: E731
    put("Knees with a longitudinal thickness rate (≥ 2 visits)", lambda g: str(int(tk(g).has_rate.sum())))

    def visits(g):
        v = tk(g).loc[lambda d: d.has_rate, "n_fit_visits"]
        return f"{int(v.median())} ({v.min()}–{v.max()})"

    put("Thickness visits per knee, median (range)ᶜ", visits)
    sm = lambda g: sims.loc[[k for k in sims.index if df.loc[k, 'group'] == g]]  # noqa: E731

    def converged(g):
        s = sm(g)
        attempted = len(s) * n_conditions
        return f"{s.converged.sum():,} / {attempted:,} ({100 * s.converged.sum() / attempted:.1f} %)"

    put("Simulations converged / attempted", converged)
    put(
        "Simulations excluded by quality control",
        lambda g: f"{sm(g).excluded.sum()} ({100 * sm(g).excluded.sum() / sm(g).converged.sum():.1f} %)",
    )
    put("Simulations analysed", lambda g: f"{sm(g).analysed.sum():,}")

    for g in GROUPS:
        s, t, q = sub(g), tk(g), sm(g)
        meta_groups[g] = dict(
            label=GROUPS[g],
            knees=int(len(s)),
            participants=int(s.participant.nunique()),
            knees_with_thickness_rate=int(t.has_rate.sum()),
            knees_without_thickness_rate=sorted(t.index[~t.has_rate].tolist()),
            simulations_converged=int(q.converged.sum()),
            simulations_attempted=int(len(q) * n_conditions),
            simulations_excluded=int(q.excluded.sum()),
            simulations_analysed=int(q.analysed.sum()),
            incident_oa_months=(
                {int(k): int(v) for k, v in s.oa_months.dropna().astype(int).value_counts().sort_index().items()}
            ),
            adjudicated_status_counts={k: int(v) for k, v in s.xioa.value_counts().items()},
        )

    n_part = int(df.participant.nunique())
    both = int((df.groupby("participant").size() > 1).sum())
    footnotes = [
        f"ᵃ {len(df)} knees from {n_part} participants; {both} participants contribute both knees, "
        "sometimes to different groups, so per-group participant counts sum to more than the total.",
        f"ᵇ OAI adjudicated incident radiographic OA (Outcomes99 `V99ExXIOA`, first visit with KL ≥ 2). "
        f"Healthy and incident are defined on this outcome with a {INCIDENT_WINDOW_MONTHS}-month window: "
        "a knee whose incident OA falls at 72 or 96 months is healthy by definition. Baseline KL is the "
        f"OAI central reading (project {KL_READING_PROJECT}).",
        "ᶜ Visits entering the thinning-rate fit: baseline, 12, 24, 36, 48 and 72 months (the 96-month "
        "visit is excluded from the fit). Knees with fewer than two thickness maps have no rate.",
    ]
    return table, footnotes, meta_groups


def write_outputs(table, footnotes, meta, out_dir: Path) -> None:
    labels = list(GROUPS.values())
    with (out_dir / f"{OUT_NAME}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row"] + labels)
        for row, cells in table.items():
            w.writerow([row.replace("**", "")] + [cells[l] for l in labels])
    md = [
        "**Extended Data Table 1. Cohort characteristics, radiographic outcome, and data analysed.** "
        "Mean (SD) unless stated.",
        "",
        "| | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for row, cells in table.items():
        md.append(f"| {row} | " + " | ".join(cells[l] for l in labels) + " |")
    md += ["", *footnotes, ""]
    (out_dir / f"{OUT_NAME}.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    root = oai_clinical_root()
    knee2g = load_groups()
    df = baseline_and_outcome(root, knee2g)
    rule_counts = check_group_rule(df)
    thick = thickness_coverage(knee2g)
    sims, n_conditions = simulation_coverage(knee2g)
    table, footnotes, meta_groups = build_table(df, thick, sims, n_conditions)

    meta = dict(
        generated=date.today().isoformat(),
        script="pipeline/derive_table_ed1_cohort.py",
        inputs=dict(
            groups="00_inputs/groups.json",
            oai_clinical_files={k: str(v) for k, v in OAI_FILES.items()},
            oai_variables=dict(
                sex="Enrollees.P02SEX", race="Enrollees.P02RACE", age="AllClinical00.V00AGE",
                height="AllClinical00.P01HEIGHT", weight="AllClinical00.P01WEIGHT",
                bmi="weight / (height/1000)^2", womac_pain="AllClinical00.V00WOMKP{R,L}",
                kl_baseline=f"kxr_sq_bu00.V00XRKL, READPRJ == {KL_READING_PROJECT}",
                incident_oa="Outcomes99.V99E{R,L}XIOA",
            ),
            thickness="03_derived/thickness_merged/{thinning_rates, thickness_on_reference}",
            qc=str(shared_qc.unified_csv().relative_to(data_root())),
        ),
        selection_rule=dict(
            healthy=f"baseline KL < 2 and no adjudicated incident OA within {INCIDENT_WINDOW_MONTHS} months",
            incident=f"baseline KL < 2 and adjudicated incident OA within {INCIDENT_WINDOW_MONTHS} months",
            mild_oa="baseline KL == 2",
            verified_against_groups_json=rule_counts,
        ),
        n_gait_conditions=n_conditions,
        thickness_visits_in_fit=list(THICKNESS_VISITS_IN_FIT),
        totals=dict(
            knees=int(len(df)),
            participants=int(df.participant.nunique()),
            simulations_analysed=int(sims.analysed.sum()),
            simulations_converged=int(sims.converged.sum()),
            simulations_attempted=int(len(sims) * n_conditions),
        ),
        groups=meta_groups,
        note="Aggregates only. No participant-level OAI clinical value is written anywhere.",
    )
    out_dir = stage_dir("results", OUT_NAME, create=True)
    write_outputs(table, footnotes, meta, out_dir)
    print(f"\nwrote {out_dir}/{{{OUT_NAME}.csv, {OUT_NAME}.md, meta.json}}")


if __name__ == "__main__":
    main()
