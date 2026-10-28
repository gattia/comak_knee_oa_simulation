"""Stage 03 thickness, GPU half: decode + interpolate reproduce the record.

``test_derivation_inputs.py`` covers the CPU half (the thinning regression).
This covers the two GPU steps that feed it -- ``decode_one`` (cartilage decoded
from the shape-model latent, thickness by ray-casting to the bone) and
``interpolate_one`` (the latent walk to the population mean and the transfer
onto the 20,000-vertex reference surface) -- on a small, seeded, stratified
sample, and compares each result against the thickness-on-reference arrays the
thinning rates in the data root were fitted from.

Not a cohort run: the sample is six knees x two bones x two visits.

Sample
------
Drawn once with ``random.Random(SAMPLE_SEED)`` -- see ``draw_sample()``, which
reproduces it -- one knee per (group x side) cell, restricted to knees whose
V00 and V08 arrays are both present for both bones and whose V08 arrays are not
in the rebuilt ``03_derived/thickness`` tree, so every item compares against the
``thickness_merged`` record (see ``record_paths``). V00 and V08 items therefore
exercise the decode and interpolation that produced the record for those
visits.

Outputs go to a temporary directory: the path functions are monkeypatched, so
nothing is written into the data root.

Tolerance
---------
GPU floating point and marching cubes mean exact equality is not expected.
Per item, on the reference surface: the covered-vertex masks (thickness > 0)
may differ by at most ``MAX_MASK_DIFF`` vertices, and on vertices covered in
both, the median |difference| must be < 0.01 mm and the 99th percentile
< 0.05 mm.

Running it
----------
    pytest tests/test_thickness_gpu.py                  # skips without CUDA
    python tests/test_thickness_gpu.py --out DIR        # same, plus a summary

Set ``COMAK_THICKNESS_SMOKE_OUT=DIR`` to have the pytest run write the same
per-item summary (``summary.csv`` and ``summary.md``) into DIR.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import itertools
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import derive_thickness  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402

# ------------------------------------------------------------------ sample ---

SAMPLE_SEED = 20260929
SAMPLE_VISITS = ("V00", "V08")
SIDES = ("LEFT", "RIGHT")

# (knee, group) as produced by draw_sample(SAMPLE_SEED); pinned here so the
# test does not depend on directory listings staying the same.
SAMPLE = (
    ("9669990_LEFT", "kl0_1_healthy"),
    ("9503723_RIGHT", "kl0_1_healthy"),
    ("9973199_LEFT", "kl0_1_progressor"),
    ("9454239_RIGHT", "kl0_1_progressor"),
    ("9805712_LEFT", "kl2"),
    ("9290752_RIGHT", "kl2"),
)

ITEMS = [
    (knee, group, visit, bone)
    for (knee, group), visit, bone in itertools.product(
        SAMPLE, SAMPLE_VISITS, derive_thickness.BONES
    )
]

MAX_MASK_DIFF = 20          # vertices, of 20,000
MEDIAN_TOL_MM = 0.01
P99_TOL_MM = 0.05


def record_paths(knee: str, visit: str, bone: str) -> tuple[Path, Path]:
    """(thickness-mm npy, thickness-nsm npy) of record on the reference surface.

    ``thickness_merged`` is the tree the thinning rates were fitted from: the
    V00 rebuild plus the follow-up visits.
    """
    directory = stage_dir("derived", "thickness_merged", "thickness_on_reference", visit)
    base = f"{knee}_{bone}_thickness_ref"
    return directory / f"{base}.npy", directory / f"{base}_nsm.npy"


def record_decoded_mesh(knee: str, visit: str, bone: str) -> Path:
    """The decoded cartilage mesh the record's array was interpolated from.

    Found next to wherever the record's array really lives, so a symlink into
    the development tree leads to that tree's decoded mesh.
    """
    mm_path, _ = record_paths(knee, visit, bone)
    tree = Path(os.path.realpath(mm_path)).parents[2]
    return tree / "thickness_maps" / visit / f"{knee}_{bone}_cart_thickness.vtk"


def draw_sample(seed: int = SAMPLE_SEED) -> list[tuple[str, str]]:
    """Reproduce SAMPLE: one knee per (group x side), seeded."""
    groups = json.loads((stage_dir("inputs") / "groups.json").read_text())
    rebuilt_dir = stage_dir("derived", "thickness", "thickness_on_reference", "V08")
    rebuilt = {p.name for p in rebuilt_dir.iterdir()} if rebuilt_dir.is_dir() else set()

    def eligible(knee):
        for bone in derive_thickness.BONES:
            for visit in SAMPLE_VISITS:
                if not all(p.exists() for p in record_paths(knee, visit, bone)):
                    return False
            if f"{knee}_{bone}_thickness_ref.npy" in rebuilt:
                return False
        return True

    rng = random.Random(seed)
    sample = []
    for group in sorted(groups):
        for side in SIDES:
            pool = sorted(k for k in groups[group] if k.endswith(side) and eligible(k))
            sample.append((rng.choice(pool), group))
    return sample


# ------------------------------------------------------------------ running --


def load_models_and_references():
    from pymskt.mesh import Mesh

    models = {b: derive_thickness._load_shape_model(b) for b in derive_thickness.BONES}
    references = {
        b: Mesh(str(derive_thickness.reference_surface_path(b)))
        for b in derive_thickness.BONES
    }
    return models, references


@contextlib.contextmanager
def outputs_redirected_to(out_dir: Path):
    """Point decode/interpolate's output path functions into ``out_dir``."""
    out_dir = Path(out_dir)

    def thickness_map_path(knee, visit, bone):
        return out_dir / "thickness_maps" / visit / f"{knee}_{bone}_cart_thickness.vtk"

    def on_reference_paths(knee, visit, bone):
        directory = out_dir / "thickness_on_reference" / visit
        base = f"{knee}_{bone}_thickness_ref"
        return (
            directory / f"{base}.vtk",
            directory / f"{base}.npy",
            directory / f"{base}_nsm.npy",
        )

    saved = (derive_thickness.thickness_map_path, derive_thickness.on_reference_paths)
    derive_thickness.thickness_map_path = thickness_map_path
    derive_thickness.on_reference_paths = on_reference_paths
    try:
        # Prove the redirect took before anything is written.
        probe = derive_thickness.thickness_map_path("PROBE_RIGHT", "V00", "femur")
        assert out_dir in probe.parents, probe
        assert out_dir in derive_thickness.on_reference_paths("PROBE_RIGHT", "V00", "femur")[1].parents
        yield
    finally:
        derive_thickness.thickness_map_path, derive_thickness.on_reference_paths = saved


def _compare(new: np.ndarray, old: np.ndarray, prefix: str) -> dict:
    new = np.asarray(new, dtype=float)
    old = np.asarray(old, dtype=float)
    cov_new, cov_old = new > 0, old > 0
    both = cov_new & cov_old
    diff = np.abs(new[both] - old[both])
    return {
        f"{prefix}n_vertices": int(new.size),
        f"{prefix}covered_new": int(cov_new.sum()),
        f"{prefix}covered_record": int(cov_old.sum()),
        f"{prefix}mask_diff": int((cov_new != cov_old).sum()),
        f"{prefix}coverage_agreement": float((cov_new == cov_old).mean()),
        f"{prefix}median_abs": float(np.median(diff)) if diff.size else np.nan,
        f"{prefix}p99_abs": float(np.percentile(diff, 99)) if diff.size else np.nan,
        f"{prefix}max_abs": float(diff.max()) if diff.size else np.nan,
        f"{prefix}pearson_r": float(np.corrcoef(new[both], old[both])[0, 1]) if diff.size > 1 else np.nan,
        f"{prefix}identical": bool(np.array_equal(new, old)),
    }


def _compare_decoded(new_vtk: Path, old_vtk: Path) -> dict:
    import pyvista as pv

    row = {"decoded_record_exists": old_vtk.is_file()}
    new = pv.read(str(new_vtk))
    row["decoded_n_points_new"] = int(new.n_points)
    if not old_vtk.is_file():
        return row
    old = pv.read(str(old_vtk))
    row["decoded_n_points_record"] = int(old.n_points)
    if new.n_points == old.n_points:
        row["decoded_max_abs_xyz"] = float(np.abs(new.points - old.points).max())
        t_new = np.asarray(new.point_data["thickness (mm)"], dtype=float)
        t_old = np.asarray(old.point_data["thickness (mm)"], dtype=float)
        row["decoded_max_abs_thickness_mm"] = float(np.abs(t_new - t_old).max())
    return row


def run_item(knee, group, visit, bone, models, references, out_dir: Path) -> dict:
    """Decode + interpolate one knee-visit-bone into ``out_dir`` and compare."""
    row = {"knee": knee, "side": knee.rsplit("_", 1)[1], "group": group,
           "visit": visit, "bone": bone}
    started = time.time()
    with outputs_redirected_to(out_dir):
        decoded = derive_thickness.decode_one(knee, visit, bone, models[bone])
        derive_thickness.interpolate_one(knee, visit, bone, models[bone], references[bone])
        _, mm_new, nsm_new = derive_thickness.on_reference_paths(knee, visit, bone)
    row["seconds"] = round(time.time() - started, 1)

    mm_old, nsm_old = record_paths(knee, visit, bone)
    row["record_source"] = os.path.realpath(mm_old)
    row.update(_compare(np.load(mm_new), np.load(mm_old), "mm_"))
    row.update(_compare(np.load(nsm_new), np.load(nsm_old), "nsm_"))
    row.update(_compare_decoded(decoded, record_decoded_mesh(knee, visit, bone)))
    row["pass"] = bool(
        row["mm_mask_diff"] <= MAX_MASK_DIFF
        and row["mm_median_abs"] < MEDIAN_TOL_MM
        and row["mm_p99_abs"] < P99_TOL_MM
    )
    return row


# ------------------------------------------------------------------ summary --

SUMMARY_COLUMNS = [
    "knee", "side", "group", "visit", "bone", "pass", "seconds",
    "mm_n_vertices", "mm_covered_new", "mm_covered_record", "mm_mask_diff",
    "mm_coverage_agreement", "mm_median_abs", "mm_p99_abs", "mm_max_abs",
    "mm_pearson_r", "mm_identical",
    "nsm_mask_diff", "nsm_median_abs", "nsm_p99_abs", "nsm_max_abs",
    "nsm_pearson_r", "nsm_identical",
    "decoded_record_exists", "decoded_n_points_new", "decoded_n_points_record",
    "decoded_max_abs_xyz", "decoded_max_abs_thickness_mm", "record_source",
]


def write_summary(rows: list[dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}" if np.isfinite(v) else "nan"
        return str(v)

    shown = ["knee", "group", "visit", "bone", "pass", "mm_mask_diff",
             "mm_median_abs", "mm_p99_abs", "mm_max_abs", "mm_pearson_r",
             "nsm_mask_diff", "nsm_median_abs", "nsm_pearson_r",
             "decoded_n_points_new", "decoded_n_points_record", "decoded_max_abs_xyz"]
    lines = [
        f"# Thickness GPU smoke test ({time.strftime('%Y-%m-%d %H:%M')})",
        "",
        f"Seed {SAMPLE_SEED}; {len(rows)} items; "
        f"{sum(r.get('pass', False) for r in rows)} pass "
        f"(mask diff <= {MAX_MASK_DIFF}, median < {MEDIAN_TOL_MM} mm, p99 < {P99_TOL_MM} mm).",
        "",
        "| " + " | ".join(shown) + " |",
        "|" + "---|" * len(shown),
    ]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in shown) + " |")
    ok = [r for r in rows if "mm_median_abs" in r]
    if ok:
        lines += [
            "",
            "Aggregate (mm, over items): worst median "
            f"{max(r['mm_median_abs'] for r in ok):.3g}, worst p99 "
            f"{max(r['mm_p99_abs'] for r in ok):.3g}, worst max "
            f"{max(r['mm_max_abs'] for r in ok):.3g}, worst mask diff "
            f"{max(r['mm_mask_diff'] for r in ok)}, min r "
            f"{min(r['mm_pearson_r'] for r in ok):.6f}",
        ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------- pytest --


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="needs a CUDA GPU")

_ROWS: list[dict] = []


@pytest.fixture(scope="module")
def models_and_references():
    yield load_models_and_references()
    out = os.environ.get("COMAK_THICKNESS_SMOKE_OUT")
    if out and _ROWS:
        write_summary(_ROWS, Path(out))


def test_sample_matches_draw():
    assert tuple(draw_sample(SAMPLE_SEED)) == SAMPLE


@requires_cuda
@pytest.mark.parametrize(
    "knee,group,visit,bone", ITEMS, ids=lambda v: str(v)
)
def test_decode_and_interpolate_reproduce_record(
    knee, group, visit, bone, models_and_references, tmp_path
):
    models, references = models_and_references
    row = run_item(knee, group, visit, bone, models, references, tmp_path)
    _ROWS.append(row)
    assert row["mm_mask_diff"] <= MAX_MASK_DIFF, (
        f"coverage masks differ at {row['mm_mask_diff']} vertices"
    )
    assert row["mm_median_abs"] < MEDIAN_TOL_MM, f"median |d| {row['mm_median_abs']:.4f} mm"
    assert row["mm_p99_abs"] < P99_TOL_MM, f"p99 |d| {row['mm_p99_abs']:.4f} mm"


# ------------------------------------------------------------------- driver --


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory for summary.csv/.md")
    parser.add_argument("--keep", action="store_true", help="keep the decoded outputs in OUT/work")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--visit", choices=SAMPLE_VISITS, help="only this visit")
    parser.add_argument("--knee", action="append", help="only this knee (repeatable)")
    parser.add_argument("--bone", choices=derive_thickness.BONES, help="only this bone")
    args = parser.parse_args(argv)

    if not _cuda_available():
        print("no CUDA GPU; nothing run")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models, references = load_models_and_references()
    items = [
        item for item in ITEMS
        if (args.visit is None or item[2] == args.visit)
        and (args.knee is None or item[0] in args.knee)
        and (args.bone is None or item[3] == args.bone)
    ]
    items = items[: args.limit] if args.limit else items
    rows = []
    with contextlib.ExitStack() as stack:
        work = out / "work" if args.keep else Path(stack.enter_context(tempfile.TemporaryDirectory()))
        for i, (knee, group, visit, bone) in enumerate(items, 1):
            try:
                row = run_item(knee, group, visit, bone, models, references, work)
            except Exception as exc:  # report and continue
                row = {"knee": knee, "group": group, "visit": visit, "bone": bone,
                       "pass": False, "record_source": f"ERROR: {exc!r}"}
            rows.append(row)
            print(f"[{i}/{len(items)}] {knee} {visit} {bone}: pass={row.get('pass')} "
                  f"median={row.get('mm_median_abs')} p99={row.get('mm_p99_abs')} "
                  f"maskdiff={row.get('mm_mask_diff')}", flush=True)
            write_summary(rows, out)
    print(f"{sum(r.get('pass', False) for r in rows)}/{len(rows)} pass; summary in {out}")
    return 0 if all(r.get("pass") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
