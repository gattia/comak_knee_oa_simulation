"""Compare a re-run with the run it reproduces: simulations, and rebuilt models.

Used by ``run_knee.py reproduce``, and from the command line::

    python -m comak_knee_oa.compare sims   <published results dir> <re-run results dir>
    python -m comak_knee_oa.compare models <published model dir>   <rebuilt model dir>

Simulations
-----------
Three outputs carry the solver chain's signal::

    comak/_force.sto                            muscle and contact forces
    comak-inverse-dynamics/inverse-dynamics.sto joint moments
    joint-mechanics/joint_mechanics.h5          contact pressures and kinematics

A re-run is first checked for exact equality of all three; on the CPU model the
published run records, in the same environment and from the same model file,
that is the expectation. Anything else is judged on the quantities the paper reads
(``simulation_metrics``): knee and patellar kinematics in degrees and
millimetres, and each contact's stance-mean and peak pressure and force in
percent. Element-wise tolerances cannot make that judgement: a triangle at the
edge of a contact patch switching on or off is a 100 % change in one value
while the compartment's pressure moves by a fraction of a percent.

The judgement is against ``CROSS_CPU_ENVELOPE``: the largest difference
measured between the same simulations run on two CPU types (see there). A
re-run whose patella-repositioning decision (whether the forward simulation
moved the patella before COMAK, ``forsim_results/total_patella_update.json``)
differs from that of the run it reproduces is reported apart: the decision moves the patella by about 4 mm, so its metrics are
not comparable.

Models
------
A rebuilt model is compared file by file and link by link with the run name
masked (``compare_models``). Built on the same GPU and CPU type, in the same
environment, the only files that differ are the ones in ``EXPECTED_DIFFERENT``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "FileComparison", "compare_results", "read_sto", "report",
    "CROSS_CPU_ENVELOPE", "SimulationComparison", "compare_simulations", "simulation_metrics",
    "report_simulation", "compare_environments", "environment_differences",
    "EXPECTED_DIFFERENT", "ModelComparison", "compare_models", "report_models",
]

# Result-tree files compared, relative to each results directory.
COMPARED_FILES = (
    "comak/_force.sto",
    "comak-inverse-dynamics/inverse-dynamics.sto",
    "joint-mechanics/joint_mechanics.h5",
)


@dataclass
class FileComparison:
    """Outcome of comparing one file across the two trees."""

    relative_path: str
    ok: bool
    max_abs: float = float("nan")
    max_rel: float = float("nan")
    detail: str = ""
    worst: str = ""
    per_dataset: list = field(default_factory=list)


def read_sto(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse an OpenSim storage file into (column names, values).

    Handles both header dialects: an explicit ``endheader`` line, and files
    that start the table directly at the ``time`` column row.
    """
    lines = Path(path).read_text().splitlines()
    header_end = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break
    if header_end is None:
        header_end = (
            next(
                i
                for i, line in enumerate(lines)
                if line.lower().lstrip().startswith("time")
            )
            - 1
        )
    columns = lines[header_end + 1].split()
    values = np.asarray(
        [
            [float(x) for x in line.split()]
            for line in lines[header_end + 2 :]
            if line.strip()
        ],
        dtype=float,
    )
    return columns, values


def _compare_sto(reference: Path, candidate: Path, atol, rtol) -> FileComparison:
    rel = str(candidate)
    ref_cols, ref = read_sto(reference)
    new_cols, new = read_sto(candidate)
    if ref.shape != new.shape:
        # Nothing is compared value by value, so no difference is reported.
        return FileComparison(
            rel, False, detail=f"shape differs: {ref.shape} vs {new.shape}, values not compared"
        )
    if ref_cols != new_cols:
        return FileComparison(rel, False, detail="column names differ")
    abs_diff = np.abs(ref - new)
    rel_diff = abs_diff / np.maximum(np.abs(ref), 1e-12)
    worst_column = ref_cols[int(np.argmax(np.max(abs_diff, axis=0)))]
    ok = bool(np.all((abs_diff <= atol) | (rel_diff <= rtol)))
    return FileComparison(
        rel,
        ok,
        max_abs=float(abs_diff.max()),
        max_rel=float(rel_diff.max()),
        worst=worst_column,
        detail=f"shape {ref.shape}",
    )


def _compare_h5(reference: Path, candidate: Path, atol, rtol) -> FileComparison:
    """Every dataset of ``reference`` against the same one in ``candidate``.

    A dataset missing from the candidate, or of another shape, cannot be
    compared value by value. Those are counted and named apart, so the largest
    difference reported is over the datasets actually compared, and a file
    whose datasets all differ in shape reports no difference at all rather
    than a misleading zero.
    """
    import h5py

    rel = str(candidate)
    # (dataset, largest absolute difference) or (dataset, why it was not compared)
    failures: list[tuple[str, float | str]] = []
    max_abs = 0.0
    max_rel = 0.0
    count = 0
    compared = 0
    n_values = 0
    shapes: list[tuple[str, tuple, tuple]] = []
    missing: list[str] = []
    worst = ""

    with h5py.File(reference, "r") as ref, h5py.File(candidate, "r") as new:

        def visit(name, node):
            nonlocal max_abs, max_rel, count, compared, n_values, worst
            if not isinstance(node, h5py.Dataset):
                return
            count += 1
            if name not in new:
                missing.append(name)
                failures.append((name, "missing in the re-run"))
                return
            a = np.asarray(node)
            b = np.asarray(new[name])
            if a.shape != b.shape:
                shapes.append((name, a.shape, b.shape))
                failures.append((name, f"shape {a.shape} vs {b.shape}"))
                return
            if a.dtype.kind not in "fiu":
                return
            compared += 1
            a = a.astype(float)
            b = b.astype(float)
            abs_diff = np.abs(a - b)
            rel_diff = abs_diff / np.maximum(np.abs(a), 1e-12)
            largest = float(abs_diff.max()) if abs_diff.size else 0.0
            if largest > max_abs:
                max_abs, worst = largest, name
            if rel_diff.size:
                max_rel = max(max_rel, float(rel_diff.max()))
            if abs_diff.size and not np.all(
                (abs_diff <= atol) | (rel_diff <= rtol)
            ):
                n_values += 1
                failures.append((name, largest))

        ref.visititems(visit)

    parts = []
    if shapes:
        # The time axis first: when it differs, every time series differs with it.
        name, sa, sb = next((s for s in shapes if s[0] == "time"), shapes[0])
        parts.append(f"{len(shapes)} differ in shape (e.g. {name} {sa} vs {sb})")
    if missing:
        parts.append(f"{len(missing)} missing in the re-run")
    if not (shapes or missing):
        parts.append(f"{n_values} differ in value")
    elif compared:
        parts.append(f"{compared} compared, {n_values} of them differ in value")
    else:
        parts.append("none compared")
    return FileComparison(
        rel,
        ok=not failures,
        # Over the datasets compared; none compared, no number.
        max_abs=max_abs if compared else float("nan"),
        max_rel=max_rel if compared else float("nan"),
        worst=worst,
        detail=f"{count} datasets: " + "; ".join(parts),
        per_dataset=failures[:10],
    )


def compare_results(
    reference_dir: Path,
    candidate_dir: Path,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> list[FileComparison]:
    """Compare the signal-carrying outputs of two result trees.

    A file passes when every value agrees within ``atol`` or ``rtol`` --
    element-wise, so a single divergent muscle force fails the file even if the
    aggregate statistics look similar.
    """
    results = []
    for relative in COMPARED_FILES:
        reference = Path(reference_dir) / relative
        candidate = Path(candidate_dir) / relative
        if not reference.is_file():
            results.append(
                FileComparison(relative, False, detail=f"missing in reference: {reference}")
            )
            continue
        if not candidate.is_file():
            results.append(
                FileComparison(relative, False, detail=f"missing: {candidate}")
            )
            continue
        compare = _compare_h5 if relative.endswith(".h5") else _compare_sto
        outcome = compare(reference, candidate, atol, rtol)
        outcome.relative_path = relative
        results.append(outcome)
    return results


def report(comparisons: list[FileComparison]) -> str:
    """A printable summary, one line per file."""
    lines = []
    for c in comparisons:
        status = "agree " if c.ok else "DIFFER"
        numbers = (
            f"max abs {c.max_abs:.3e}  max rel {c.max_rel:.3e}"
            if np.isfinite(c.max_abs)
            else ""
        )
        worst = f"  worst: {c.worst}" if c.worst and not c.ok and numbers else ""
        lines.append(f"  {status}  {c.relative_path}  " + "  ".join(x for x in (numbers + worst, f"({c.detail})") if x))
        for name, value in c.per_dataset:
            lines.append(f"          {name}: " + (f"{value:.3e}" if isinstance(value, float) else str(value)))
    return "\n".join(lines)


# ------------------------------------------------------- simulation metrics ----

# Coordinates of the knee and the patella. The JointMechanics h5 stores rotations
# in degrees and translations in metres.
ROTATIONS = ("knee_flex_r", "knee_add_r", "knee_rot_r", "pf_flex_r", "pf_rot_r", "pf_tilt_r")
TRANSLATIONS = ("knee_tx_r", "knee_ty_r", "knee_tz_r", "pf_tx_r", "pf_ty_r", "pf_tz_r")
# (label, contact force, surface whose totals are read). The tibiofemoral contact
# twice (both cartilages), the tibial side of each meniscus contact, and the
# patellar side of the patellofemoral contact.
CONTACTS = (
    ("tibiofemoral (tibia)", "tf_contact", "tibia_cartilage"),
    ("tibiofemoral (femur)", "tf_contact", "femur_cartilage"),
    ("medial meniscus (tibia)", "tibia_meniscus_medial_contact", "tibia_cartilage"),
    ("lateral meniscus (tibia)", "tibia_meniscus_lateral_contact", "tibia_cartilage"),
    ("patellofemoral (patella)", "pf_contact", "patella_cartilage"),
)
_CONTACT_GROUP = "model/forceset/Smith2018ArticularContactForce/{}/{}/"
JOINT_MECHANICS = "joint-mechanics/joint_mechanics.h5"
PATELLA_DECISION = "forsim_results/total_patella_update.json"

# Contact metrics, each a percent change of the re-run relative to ``published``.
CONTACT_METRICS = {
    "mean_pressure": "stance mean of the contact's mean pressure",
    "peak_mean_pressure": "peak over stance of the contact's mean pressure",
    "force": "stance mean of the contact force magnitude",
    "peak_force": "peak over stance of the contact force magnitude",
}

# The largest difference measured between the same simulations run on two CPU
# types: 35 published simulations of 30 knees (29 under the reference
# gait, 6 under measured gaits), each re-run with the same code, environment and
# model file on the other of Intel Broadwell and Ice Lake. Excluded: the
# simulations whose patella-repositioning decision flipped (reported apart). Each
# value is that maximum, rounded up; the comment gives the median and 90th
# percentile too. A re-run on another CPU type within every value agrees with the
# published run as well as two CPU types agree with each other; one beyond it is
# worth a look, not proof of an error (35 pairs; other CPU pairs were not
# measured). Kinematics: maximum absolute difference over stance. Contacts:
# absolute percent change (CONTACT_METRICS).
CROSS_CPU_ENVELOPE: dict[str, float] = {
    "knee flexion (deg)": 8.4,  # median 1.25, p90 3.14, max 8.31
    "other rotations (deg)": 5.7,  # median 1.02, p90 2.58, max 5.67
    "translations (mm)": 5.2,  # median 1.02, p90 2.99, max 5.13
    "tibiofemoral (tibia): mean_pressure": 3.3,  # median 0.16, p90 1.17, max 3.22
    "tibiofemoral (tibia): peak_mean_pressure": 2.5,  # median 0.68, p90 1.71, max 2.42
    "tibiofemoral (tibia): force": 1.9,  # median 0.29, p90 1.08, max 1.86
    "tibiofemoral (tibia): peak_force": 3.4,  # median 0.55, p90 1.63, max 3.31
    "tibiofemoral (femur): mean_pressure": 2.4,  # median 0.20, p90 1.09, max 2.39
    "tibiofemoral (femur): peak_mean_pressure": 2.2,  # median 0.60, p90 1.66, max 2.17
    "tibiofemoral (femur): force": 1.9,  # median 0.31, p90 1.10, max 1.90
    "tibiofemoral (femur): peak_force": 3.5,  # median 0.67, p90 1.52, max 3.47
    "medial meniscus (tibia): mean_pressure": 5.1,  # median 0.66, p90 1.80, max 5.05
    "medial meniscus (tibia): peak_mean_pressure": 15.0,  # median 0.76, p90 2.79, max 14.94
    "medial meniscus (tibia): force": 8.7,  # median 0.75, p90 1.86, max 8.68
    "medial meniscus (tibia): peak_force": 17.1,  # median 0.69, p90 2.26, max 17.09
    "lateral meniscus (tibia): mean_pressure": 2.1,  # median 0.39, p90 1.07, max 2.03
    "lateral meniscus (tibia): peak_mean_pressure": 7.3,  # median 0.97, p90 3.51, max 7.30
    "lateral meniscus (tibia): force": 2.3,  # median 0.49, p90 1.65, max 2.28
    "lateral meniscus (tibia): peak_force": 6.6,  # median 0.76, p90 2.52, max 6.53
    "patellofemoral (patella): mean_pressure": 27.9,  # median 1.16, p90 4.22, max 27.83
    "patellofemoral (patella): peak_mean_pressure": 22.1,  # median 0.81, p90 3.36, max 22.06
    "patellofemoral (patella): force": 49.8,  # median 1.33, p90 4.46, max 49.73
    "patellofemoral (patella): peak_force": 21.6,  # median 1.20, p90 4.56, max 21.55
}


def _load_joint_mechanics(path: Path) -> dict:
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        out["time"] = f["time"][()]
        for c in ROTATIONS + TRANSLATIONS:
            key = f"model/coordinateset/{c}/value"
            if key in f:
                out[c] = f[key][()]
        for label, contact, surface in CONTACTS:
            group = _CONTACT_GROUP.format(contact, surface)
            if group + "total_mean_pressure" in f:
                out[(label, "pressure")] = f[group + "total_mean_pressure"][()]
                out[(label, "force")] = np.linalg.norm(f[group + "total_contact_force"][()], axis=1)
    return out


def _pct(new: float, old: float) -> float:
    return float(100.0 * (new / old - 1.0)) if old else float("nan")


def simulation_metrics(published: dict, rerun: dict) -> dict[str, float]:
    """The quantities a re-run is judged on, from two loaded JointMechanics files.

    Keys: ``<coordinate>`` (deg or mm, maximum absolute difference over stance),
    the summaries ``knee flexion (deg)``, ``other rotations (deg)`` and
    ``translations (mm)``, and ``<contact label>: <metric>`` (percent, signed;
    ``CONTACT_METRICS``). Both runs must share the time grid.
    """
    if published["time"].shape != rerun["time"].shape or not np.allclose(published["time"], rerun["time"]):
        raise ValueError(
            f"the two runs have different time grids ({published['time'].size} vs {rerun['time'].size} "
            "frames): they were run with different configs (for example comak_window_pad_s), so their "
            "outputs cannot be compared frame by frame")
    m: dict[str, float] = {}
    for c in ROTATIONS:
        if c in published and c in rerun:
            m[c] = float(np.max(np.abs(rerun[c] - published[c])))
    for c in TRANSLATIONS:
        if c in published and c in rerun:
            m[c] = float(1e3 * np.max(np.abs(rerun[c] - published[c])))
    if "knee_flex_r" in m:
        m["knee flexion (deg)"] = m["knee_flex_r"]
    others = [m[c] for c in ROTATIONS[1:] if c in m]
    if others:
        m["other rotations (deg)"] = max(others)
    trans = [m[c] for c in TRANSLATIONS if c in m]
    if trans:
        m["translations (mm)"] = max(trans)
    for label, _, _ in CONTACTS:
        if (label, "pressure") not in published or (label, "pressure") not in rerun:
            continue
        p, q = published[(label, "pressure")], rerun[(label, "pressure")]
        f, g = published[(label, "force")], rerun[(label, "force")]
        m[f"{label}: mean_pressure"] = _pct(q.mean(), p.mean())
        m[f"{label}: peak_mean_pressure"] = _pct(q.max(), p.max())
        m[f"{label}: force"] = _pct(g.mean(), f.mean())
        m[f"{label}: peak_force"] = _pct(g.max(), f.max())
    return m


def _record(results_dir: Path) -> dict:
    path = Path(results_dir) / "run_record.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _patella_decision(results_dir: Path):
    path = Path(results_dir) / PATELLA_DECISION
    return [round(float(x), 9) for x in json.loads(path.read_text())] if path.is_file() else None


# Older run records list less. A record without ``native_libraries`` lists no
# native libraries, and one without ``pins_accept`` that lists no MKL library
# does not record MKL, although its run may have loaded it. A library missing
# from such a record is "not recorded there", not "not loaded": no difference.
# Library-name prefix -> the environment key that marks records listing it.
_RECORDED_SINCE = {"libmkl_": "pins_accept"}


def _records_library(environment: dict, name: str) -> bool:
    """Whether a run record's ``environment`` would have listed the library ``name`` had its run loaded it."""
    listed = environment.get("native_libraries")
    if listed is None:
        return False
    for prefix, marker in _RECORDED_SINCE.items():
        if name.startswith(prefix):
            return marker in environment or any(n.startswith(prefix) for n in listed)
    return True


def compare_environments(published: dict, rerun: dict) -> tuple[list[str], list[str]]:
    """How two run records' CPU and native libraries differ.

    Returns ``(differences, unrecorded)``. ``differences``: a different CPU model,
    a library only one run loaded, or a library whose md5 differs. ``unrecorded``:
    libraries one record lists and the other's record format does not record
    (``_RECORDED_SINCE``), which therefore cannot be compared and are not a
    difference.
    """
    out, unrecorded = [], []
    a, b = published.get("environment") or {}, rerun.get("environment") or {}
    if a.get("cpu_model") and b.get("cpu_model") and a["cpu_model"] != b["cpu_model"]:
        out.append(f"CPU model: published {a['cpu_model']!r}, this run {b['cpu_model']!r}")
    la, lb = a.get("native_libraries") or {}, b.get("native_libraries") or {}
    for name in sorted(set(la) | set(lb)):
        ma, mb = (la.get(name) or {}).get("md5"), (lb.get(name) or {}).get("md5")
        if name not in lb and ma:
            if _records_library(b, name):
                out.append(f"{name}: loaded by the published run (md5 {ma}), not by this one")
            else:
                unrecorded.append(f"{name}: not recorded in this run's record (published md5 {ma})")
        elif name not in la and mb:
            if _records_library(a, name):
                out.append(f"{name}: loaded by this run (md5 {mb}), not by the published one")
            else:
                unrecorded.append(f"{name}: not recorded in the published run (this run md5 {mb})")
        elif ma and mb and ma != mb:
            out.append(f"{name}: md5 {ma} published, {mb} this run")
    return out, unrecorded


def environment_differences(published: dict, rerun: dict) -> list[str]:
    """How two run records' CPU and native libraries differ (empty: same, or unrecorded)."""
    return compare_environments(published, rerun)[0]


@dataclass
class SimulationComparison:
    published: Path
    rerun: Path
    exact: bool
    files: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    beyond: dict = field(default_factory=dict)
    patella: tuple = (None, None)
    environment: list = field(default_factory=list)
    unrecorded: list = field(default_factory=list)
    problem: str = ""

    @property
    def patella_flipped(self) -> bool:
        a, b = self.patella
        return a is not None and b is not None and a != b

    @property
    def verdict(self) -> str:
        if self.problem:
            return "NOT COMPARABLE"
        if self.exact:
            return "BIT-IDENTICAL"
        if self.patella_flipped:
            return "PATELLA DECISION DIFFERS"
        return "WITHIN CROSS-CPU ENVELOPE" if not self.beyond else "BEYOND CROSS-CPU ENVELOPE"


def compare_simulations(published: Path, rerun: Path) -> SimulationComparison:
    """Exact check of the three signal outputs, then the metrics and their judgement."""
    published, rerun = Path(published), Path(rerun)
    files = compare_results(published, rerun, atol=0.0, rtol=0.0)
    differences, unrecorded = compare_environments(_record(published), _record(rerun))
    out = SimulationComparison(
        published, rerun, exact=all(c.ok for c in files), files=files,
        patella=(_patella_decision(published), _patella_decision(rerun)),
        environment=differences, unrecorded=unrecorded,
    )
    a, b = published / JOINT_MECHANICS, rerun / JOINT_MECHANICS
    if not a.is_file() or not b.is_file():
        out.problem = f"missing {a if not a.is_file() else b}"
        return out
    try:
        out.metrics = simulation_metrics(_load_joint_mechanics(a), _load_joint_mechanics(b))
    except ValueError as exc:
        out.problem = str(exc)
        return out
    out.beyond = {k: v for k, v in out.metrics.items()
                  if k in CROSS_CPU_ENVELOPE and not abs(v) <= CROSS_CPU_ENVELOPE[k]}
    return out


def report_simulation(c: SimulationComparison) -> str:
    """A printable report: exact check, environment, metrics against the envelope, verdict."""
    lines = [f"published {c.published}", f"re-run    {c.rerun}", "", "exact comparison:", report(c.files)]
    cpus = [((_record(d).get("environment") or {}).get("cpu_model") or "not recorded") for d in (c.published, c.rerun)]
    lines += ["", f"CPU: published {cpus[0]}; re-run {cpus[1]}", "environment:"]
    lines += [f"  {d}" for d in c.environment] or ["  same CPU model and native libraries (or not recorded)"]
    if c.unrecorded:
        lines += ["not compared (one run record predates recording these libraries; not a difference):"]
        lines += [f"  {d}" for d in c.unrecorded]
    a, b = c.patella
    lines += ["", f"patella repositioning (m): published {a}, re-run {b}"
              + ("  <- DIFFERENT DECISION" if c.patella_flipped else "")]
    if c.metrics:
        lines += ["", f"  {'metric':46s} {'re-run - published':>18s} {'envelope':>9s}"]
        for k, v in c.metrics.items():
            if k in ROTATIONS + TRANSLATIONS:
                continue
            unit = "" if k.endswith(")") else " %"
            env = CROSS_CPU_ENVELOPE.get(k)
            flag = "  <- beyond" if k in c.beyond else ""
            lines.append(f"  {k:46s} {v:+16.3f}{unit:2s} {'' if env is None else f'{env:9.1f}'}{flag}")
    lines += ["", f"verdict: {c.verdict}"]
    if c.problem:
        lines.append(f"  {c.problem}")
    elif c.exact:
        lines.append("  every value of all three outputs is equal")
    elif c.patella_flipped:
        lines.append("  the forward simulation repositioned the patella differently; this happens in about "
                     "4 % of simulations across CPU types and moves the patella by about 4 mm, so the "
                     "metrics above are not a test of agreement")
    elif not c.beyond:
        lines.append("  not bit-identical, and every metric is within the largest difference measured "
                     "between two CPU types")
    else:
        lines.append(f"  {len(c.beyond)} metric(s) exceed the largest difference measured between two CPU types")
    if not c.exact and not c.problem:
        if not c.environment:
            lines.append("  NOTE: the records show the same CPU model and native libraries"
                         + (" (those both record)" if c.unrecorded else "")
                         + ", where a re-run is expected to be bit-identical: check the model file and the config")
        elif not any(d.startswith("CPU model") for d in c.environment):
            lines.append("  NOTE: same CPU model, different native libraries: the environment is not the "
                         "published run's (environment.lock.txt recreates it), so exactness is not expected")
    return "\n".join(lines)


# ----------------------------------------------------------------- models ----

# Files of a model tree that legitimately differ between two builds, and why.
EXPECTED_DIFFERENT = {
    "run_record.json": "the build's own record: paths, times, host, job ids",
}
_NUMBER = re.compile(rb"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _numeric_difference(a: bytes, b: bytes):
    """(max abs, max rel) over the numbers of two texts that differ only in numbers, else None."""
    if _NUMBER.sub(b"#", a) != _NUMBER.sub(b"#", b):
        return None
    x = np.array([float(t) for t in _NUMBER.findall(a)])
    y = np.array([float(t) for t in _NUMBER.findall(b)])
    if x.size != y.size:
        return None
    d = np.abs(x - y)
    if not d.size:
        return 0.0, 0.0
    return float(d.max()), float((d / np.maximum(np.abs(x), 1e-300)).max())


@dataclass
class ModelComparison:
    published: Path
    rebuilt: Path
    identical: int = 0
    links_identical: int = 0
    expected: list = field(default_factory=list)
    differing: list = field(default_factory=list)
    only_published: list = field(default_factory=list)
    only_rebuilt: list = field(default_factory=list)

    @property
    def same(self) -> bool:
        return not (self.differing or self.only_published or self.only_rebuilt)


def _mask(text, run: str):
    """``text`` with each whole-token occurrence of ``run`` replaced by ``{run}``.

    A token is bounded by anything but a letter or digit, so ``v3`` is masked in
    ``knee_v3.osim`` and ``comak_v3/`` but not in ``v30`` or ``abv3``.
    """
    if isinstance(text, bytes):
        return re.sub(rb"(?<![A-Za-z0-9])" + re.escape(run.encode()) + rb"(?![A-Za-z0-9])", b"{run}", text)
    return re.sub(r"(?<![A-Za-z0-9])" + re.escape(run) + r"(?![A-Za-z0-9])", "{run}", text)


def _entries(root: Path, run: str) -> dict[str, Path]:
    out = {}
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        out[_mask(str(path.relative_to(root)), run)] = path
    return out


def compare_models(published: Path, rebuilt: Path, published_run: str | None = None,
                   rebuilt_run: str | None = None) -> ModelComparison:
    """Compare two model trees (``01_models/<run>/<knee>``) with each run name masked.

    Names and contents are compared after replacing the run name with ``{run}``.
    A link is compared by the contents of the file it reaches (by its target only
    if it dangles), since two roots reach the same released file through
    different relative paths. The run names default to the name of each
    tree's parent directory, which is the run in the ``01_models/<run>/<knee>``
    layout. A differing text file that differs only in numbers reports the largest
    absolute and relative difference.
    """
    published, rebuilt = Path(published), Path(rebuilt)
    published_run = published_run or published.resolve().parent.name
    rebuilt_run = rebuilt_run or rebuilt.resolve().parent.name
    a, b = _entries(published, published_run), _entries(rebuilt, rebuilt_run)
    out = ModelComparison(published, rebuilt,
                          only_published=sorted(set(a) - set(b)), only_rebuilt=sorted(set(b) - set(a)))
    for rel in sorted(set(a) & set(b)):
        x, y = a[rel], b[rel]
        linked = x.is_symlink() or y.is_symlink()
        if linked and not (x.exists() and y.exists()):
            # A dangling link: all there is to compare is where it points.
            tx = _mask(os.readlink(x), published_run) if x.is_symlink() else None
            ty = _mask(os.readlink(y), rebuilt_run) if y.is_symlink() else None
            if tx == ty:
                out.links_identical += 1
            else:
                out.differing.append((rel, f"dangling link: {tx!r} vs {ty!r}"))
            continue
        # A link is compared by the file it reaches: two roots link the same released
        # file through different relative paths, and a link and a copy of the same
        # bytes give the same model.
        bx, by = _mask(x.read_bytes(), published_run), _mask(y.read_bytes(), rebuilt_run)
        if bx == by:
            if linked:
                out.links_identical += 1
            else:
                out.identical += 1
        elif Path(rel).name in EXPECTED_DIFFERENT:
            out.expected.append((rel, EXPECTED_DIFFERENT[Path(rel).name]))
        else:
            num = _numeric_difference(bx, by)
            detail = (f"numbers only: max abs {num[0]:.3g}, max rel {num[1]:.3g}" if num is not None
                      else f"{len(bx)} vs {len(by)} bytes")
            out.differing.append((rel, detail))
    return out


def report_models(c: ModelComparison) -> str:
    lines = [f"published {c.published}", f"rebuilt   {c.rebuilt}",
             f"identical with the run name masked: {c.identical} files, {c.links_identical} links "
             "(compared by the file each reaches)"]
    lines += [f"  expected to differ: {rel} ({why})" for rel, why in c.expected]
    lines += [f"  DIFFERS: {rel} ({detail})" for rel, detail in c.differing]
    lines += [f"  only in published: {rel}" for rel in c.only_published]
    lines += [f"  only in rebuilt:   {rel}" for rel in c.only_rebuilt]
    lines.append("verdict: " + ("IDENTICAL apart from the run name" if c.same else "DIFFERENT"))
    return "\n".join(lines)


# -------------------------------------------------------------------- CLI ----


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m comak_knee_oa.compare",
                                     description="Compare a re-run with a published one.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sims", help="two simulation results directories (comak_<run>)")
    p.add_argument("published", type=Path)
    p.add_argument("rerun", type=Path)
    p = sub.add_parser("models", help="two model directories (01_models/<run>/<knee>)")
    p.add_argument("published", type=Path)
    p.add_argument("rebuilt", type=Path)
    p.add_argument("--published-run", help="run name to mask in the published tree (default: its parent folder)")
    p.add_argument("--rebuilt-run", help="run name to mask in the rebuilt tree (default: its parent folder)")
    args = parser.parse_args(argv)
    if args.command == "sims":
        c = compare_simulations(args.published, args.rerun)
        print(report_simulation(c))
        return 0 if c.verdict in ("BIT-IDENTICAL", "WITHIN CROSS-CPU ENVELOPE") else 1
    c = compare_models(args.published, args.rebuilt, args.published_run, args.rebuilt_run)
    print(report_models(c))
    return 0 if c.same else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
