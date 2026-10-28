"""Stage 03, thickness: per-vertex cartilage thickness on the reference surface.

Every knee's cartilage is expressed on one shared 20,000-vertex reference
surface per bone, so that vertex *j* is the same anatomical location in every
knee and at every visit. That correspondence is what makes vertex-level
statistics across subjects and over time possible, and it is built in three
steps, each a subcommand here:

``decode``
    Decode the cartilage surface from each knee-visit's shape-model latent
    vector, and measure thickness on it by casting rays from the cartilage to
    the bone. (GPU)

``interpolate``
    Walk each decoded cartilage mesh through the shape model's latent space
    from the subject to the population mean. Vertex identity is preserved
    along the walk, so the thickness values ride along; they are then
    transferred to the reference surface by weighted nearest neighbours. (GPU)

``rates``
    Fit a per-vertex linear regression of thickness against time across a
    knee's visits: the thinning-rate map. (CPU, fast)

The thickness track uses the two-output bone-and-cartilage shape models fit at
every imaging visit (femur 647, tibia 650; the numbers identify models in the
released NSM weights). The femur model differs from the one that builds the
simulation anatomy in stage 01 (568, which also fits the menisci); the tibia
model 650 is the same, fitted separately at each visit. The pressure track
bridges the two sets of fits with a per-subject transform (see
``derive_pressure.py``).

Visits and their times
----------------------
Visits map to years as V00=0, V01=1, V03=2, V05=3, V06=4, V08=6. V10 is not
used: the scanner's receive coil changed at that visit, which shifts measured
thickness and would enter the regression as signal.

Inputs and outputs
------------------
Inputs: the thickness-model latents and fit records per visit
(``00_inputs/latents_thickness/<visit>/``) and the mean reference cartilage
surfaces (``00_inputs/reference/nsm_meshes_mean_2output``). Each subcommand
covers every visit a knee has fits for and writes under ``03_derived/thickness/``:
``decode`` writes ``thickness_maps/<visit>/`` (decoded cartilage with
thickness), ``interpolate`` writes ``thickness_on_reference/<visit>/``
(thickness on the reference surface, ``.vtk`` and ``.npy``), and ``rates``
writes ``thinning_rates/`` (per-knee rate maps, mm/yr).

Downstream analyses read ``03_derived/thickness_merged/`` instead (see
``THICKNESS_OF_RECORD``), a tree no subcommand here writes: it ships with the
released data as the thickness record.

Running it
----------
    python pipeline/derive_thickness.py decode
    python pipeline/derive_thickness.py interpolate
    python pipeline/derive_thickness.py rates
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comak_knee_oa.paths import stage_dir  # noqa: E402

__all__ = ["visit_coverage", "VISIT_YEARS", "THICKNESS_MODELS"]

# The two-output (bone + cartilage) shape models the thickness track decodes.
THICKNESS_MODELS = {
    "femur": "647_nsm_femur_v0.0.1",
    "tibia": "650_nsm_tibia_v0.0.1",
}
BONES = ("femur", "tibia")

# Imaging visits included in the thinning regression, mapped to years.
# Doubles as the inclusion set: a visit absent here (V10) contributes nothing.
VISIT_YEARS = {
    "V00": 0.0,
    "V01": 1.0,
    "V03": 2.0,
    "V05": 3.0,
    "V06": 4.0,
    "V08": 6.0,
}

# Every visit with vendored shape fits, including the one the regression
# excludes -- thickness maps are still decoded for it.
ALL_VISITS = ("V00", "V01", "V03", "V05", "V06", "V08", "V10")


# ------------------------------------------------------------------ paths ----


def latent_path(knee: str, visit: str, bone: str) -> Path:
    return (
        stage_dir("inputs", "latents_thickness", visit, "STANFORD", f"{bone}_latents")
        / f"RECON_{knee}_{bone}_latent.npy"
    )


def recon_results_path(knee: str, visit: str, bone: str) -> Path:
    return (
        stage_dir("inputs", "latents_thickness", visit, "STANFORD", f"{bone}_recon_results")
        / f"RECON_{knee}_{bone}_recon_results.json"
    )


def thickness_map_path(knee: str, visit: str, bone: str) -> Path:
    return (
        stage_dir("derived", "thickness", "thickness_maps", visit)
        / f"{knee}_{bone}_cart_thickness.vtk"
    )


def on_reference_paths(knee: str, visit: str, bone: str) -> tuple[Path, Path, Path]:
    """(vtk, thickness-mm npy, thickness-nsm npy) on the reference surface."""
    directory = stage_dir("derived", "thickness", "thickness_on_reference", visit)
    base = f"{knee}_{bone}_thickness_ref"
    return (
        directory / f"{base}.vtk",
        directory / f"{base}.npy",
        directory / f"{base}_nsm.npy",
    )


def thinning_rate_path(knee: str, bone: str) -> Path:
    """Where ``rates`` writes a thinning-rate map: this pipeline's own tree."""
    return (
        stage_dir("derived", "thickness", "thinning_rates")
        / f"{knee}_{bone}_thinning_rate.vtk"
    )


# The thickness tree downstream analyses read. It is a released input, not
# written by this script. In the released data, ``thickness`` holds the outputs
# these subcommands produced (every knee at V00, a few knee-visits at the
# follow-up visits); ``thickness_merged`` joins them with the follow-up outputs
# of an earlier derivation by the same method (its _merge_report.json records
# which file came from where) and holds the thinning rates fitted on the joined
# tree. Readers use this one: the released ``thickness`` tree alone is
# incomplete.
THICKNESS_OF_RECORD = "thickness_merged"


def thinning_rates_of_record_dir() -> Path:
    return stage_dir("derived", THICKNESS_OF_RECORD, "thinning_rates")


def thinning_rate_of_record_path(knee: str, bone: str) -> Path:
    """The thinning-rate map every downstream analysis reads."""
    return thinning_rates_of_record_dir() / f"{knee}_{bone}_thinning_rate.vtk"


def reference_surface_path(bone: str) -> Path:
    return (
        stage_dir("inputs", "reference", "nsm_meshes_mean_2output")
        / f"{bone}_mean_cart_surface.vtk"
    )


def cohort() -> list[str]:
    return json.loads((stage_dir("inputs") / "cohort.json").read_text())["knees"]


def visit_coverage(visits=ALL_VISITS) -> dict[str, list[str]]:
    """Which visits each cohort knee has shape fits for, both bones present.

    Derived from the vendored files each time rather than cached: coverage is
    a property of what is on disk, and a stored copy of it is one more thing
    that can disagree.
    """
    coverage: dict[str, list[str]] = {}
    for knee in cohort():
        present = [
            visit
            for visit in visits
            if all(latent_path(knee, visit, bone).is_file() for bone in BONES)
        ]
        if present:
            coverage[knee] = present
    return coverage


# ----------------------------------------------------------------- decode ----


def _load_shape_model(bone: str):
    from nsosim.utils import load_model

    model_dir = stage_dir("inputs", "reference", "nsm_models") / THICKNESS_MODELS[bone]
    config = json.loads((model_dir / "model_params_config.json").read_text())
    return load_model(
        config=config,
        path_model_state=str(model_dir / "model" / "2000.pth"),
        model_type="triplanar",
    )


def _scale_to_mm(recon_results: Path) -> float:
    """Factor converting shape-model-space distances to millimetres.

    The fit stored an ICP transform whose rotation block carries the scale;
    together with the fit's own scale field this recovers physical units.
    """
    recon = json.loads(recon_results.read_text())
    icp = np.array(recon["icp_transform"])
    return recon.get("scale", 1) / np.linalg.norm(icp[:3, 0])


def decode_one(knee: str, visit: str, bone: str, model) -> Path:
    """Decode cartilage from the latent and measure thickness on it."""
    import torch
    from NSM.mesh import create_mesh_adaptive
    from pymskt.mesh import BoneMesh, Mesh

    latent = np.load(latent_path(knee, visit, bone))
    latent_tensor = torch.tensor(latent.flatten(), dtype=torch.float32).to("cuda")
    bone_mesh, cart_mesh = create_mesh_adaptive(
        decoder=model, latent_vector=latent_tensor, objects=2, batch_size=300_000
    )

    scale = _scale_to_mm(recon_results_path(knee, visit, bone))

    # Thickness by ray-casting from the cartilage toward the bone: a 6 mm
    # search length expressed in model space, with a short backward cast so a
    # vertex sitting just inside the bone surface still registers.
    cartilage = Mesh(cart_mesh)
    cartilage.calc_distance_to_other_mesh(
        list_other_meshes=[BoneMesh(bone_mesh.mesh)],
        ray_cast_length=-6.0 / scale,
        percent_ray_length_opposite_direction=-0.025,
        name="thickness (nsm)",
    )
    cartilage.set_scalar("thickness (mm)", cartilage.get_scalar("thickness (nsm)") * scale)
    cartilage.set_active_scalars("thickness (mm)")

    output = thickness_map_path(knee, visit, bone)
    output.parent.mkdir(parents=True, exist_ok=True)
    cartilage.save_mesh(str(output))
    return output


def run_decode(args) -> int:
    bones = BONES if args.bone == "both" else (args.bone,)
    work, skipped, missing = [], 0, 0
    for knee, visits in visit_coverage().items():
        for visit in visits:
            for bone in bones:
                if thickness_map_path(knee, visit, bone).is_file():
                    skipped += 1
                elif not recon_results_path(knee, visit, bone).is_file():
                    missing += 1
                else:
                    work.append((knee, visit, bone))
    print(f"{len(work)} to decode, {skipped} already done, {missing} missing inputs")
    if args.limit:
        work = work[: args.limit]
    if args.check or not work:
        return 0

    models = {bone: _load_shape_model(bone) for bone in sorted({w[2] for w in work})}
    failed = 0
    started = time.time()
    for i, (knee, visit, bone) in enumerate(work, 1):
        try:
            decode_one(knee, visit, bone, models[bone])
        except Exception as exc:  # keep the batch going; report at the end
            failed += 1
            print(f"FAILED {knee} {visit} {bone}: {exc}")
        if i % 25 == 0:
            rate = (time.time() - started) / i
            print(f"  {i}/{len(work)} ({rate:.0f} s each)", flush=True)
    print(f"decoded {len(work) - failed}, failed {failed}")
    return 1 if failed else 0


# ------------------------------------------------------------ interpolate ----


def interpolate_one(knee: str, visit: str, bone: str, model, reference) -> Path:
    """Carry a knee-visit's thickness onto the reference surface."""
    from NSM.mesh.interpolate import interpolate_mesh
    from pymskt.mesh import Mesh

    cartilage = Mesh(str(thickness_map_path(knee, visit, bone)))
    latent = np.load(latent_path(knee, visit, bone)).flatten()

    # Walk the mesh from the subject's shape to the population mean. Vertex
    # identity is preserved, so the thickness scalars describe the same
    # anatomical points when the walk arrives.
    walked = interpolate_mesh(
        model=model,
        latent1=latent,
        latent2=np.zeros_like(latent),
        n_steps=100,
        mesh=cartilage.copy(),
        surface_idx=1,  # cartilage in the two-output model
        verbose=False,
        spherical=False,
    )

    result = reference.copy()
    result.copy_scalars_from_other_mesh_to_current(
        walked,
        weighted_avg=True,
        n_closest=3,
        max_dist=1.1 * np.max(reference.edge_lengths),
    )

    vtk_out, mm_out, nsm_out = on_reference_paths(knee, visit, bone)
    vtk_out.parent.mkdir(parents=True, exist_ok=True)
    result.save_mesh(str(vtk_out))
    np.save(mm_out, result.get_scalar("thickness (mm)"))
    np.save(nsm_out, result.get_scalar("thickness (nsm)"))
    return vtk_out


def run_interpolate(args) -> int:
    from pymskt.mesh import Mesh

    bones = BONES if args.bone == "both" else (args.bone,)
    work, skipped, missing = [], 0, 0
    for knee, visits in visit_coverage().items():
        for visit in visits:
            for bone in bones:
                vtk_out, _, _ = on_reference_paths(knee, visit, bone)
                if vtk_out.is_file():
                    skipped += 1
                elif not thickness_map_path(knee, visit, bone).is_file():
                    missing += 1  # decode has not produced this one yet
                else:
                    work.append((knee, visit, bone))
    print(f"{len(work)} to interpolate, {skipped} already done, {missing} not yet decoded")
    if args.limit:
        work = work[: args.limit]
    if args.check or not work:
        return 0

    needed = sorted({w[2] for w in work})
    models = {bone: _load_shape_model(bone) for bone in needed}
    references = {bone: Mesh(str(reference_surface_path(bone))) for bone in needed}
    failed = 0
    started = time.time()
    for i, (knee, visit, bone) in enumerate(work, 1):
        try:
            interpolate_one(knee, visit, bone, models[bone], references[bone])
        except Exception as exc:
            failed += 1
            print(f"FAILED {knee} {visit} {bone}: {exc}")
        if i % 25 == 0:
            rate = (time.time() - started) / i
            print(f"  {i}/{len(work)} ({rate:.0f} s each)", flush=True)
    print(f"interpolated {len(work) - failed}, failed {failed}")
    return 1 if failed else 0


# ------------------------------------------------------------------ rates ----


def thinning_scalars(knee: str, bone: str, visits: list[str]) -> dict | None:
    """Per-vertex thinning regression across a knee's included visits.

    Returns the scalar arrays to attach to the reference surface, or None when
    fewer than two visits have thickness on the reference. Zero thickness
    marks a vertex outside the articular region and is excluded per vertex,
    so a vertex's regression uses only the visits where cartilage was present.
    """
    from scipy.stats import linregress

    mm_by_visit, nsm_by_visit, times = {}, {}, []
    for visit in visits:
        if visit not in VISIT_YEARS:
            continue
        _, mm_path, nsm_path = on_reference_paths(knee, visit, bone)
        if not mm_path.is_file():
            continue
        mm_by_visit[visit] = np.load(mm_path)
        if nsm_path.is_file():
            nsm_by_visit[visit] = np.load(nsm_path)
        times.append((visit, VISIT_YEARS[visit]))
    if len(times) < 2:
        return None

    times.sort(key=lambda pair: pair[1])
    t_years = np.array([year for _, year in times])
    mm = np.stack([mm_by_visit[visit] for visit, _ in times], axis=0)
    has_nsm = len(nsm_by_visit) == len(times)
    if has_nsm:
        nsm = np.stack([nsm_by_visit[visit] for visit, _ in times], axis=0)

    n_vertices = mm.shape[1]
    baseline = mm[0]
    masked = mm.astype(np.float64).copy()
    masked[masked == 0] = np.nan

    slope_mm = np.full(n_vertices, np.nan)
    slope_nsm = np.full(n_vertices, np.nan)
    slope_pct = np.full(n_vertices, np.nan)
    intercept = np.full(n_vertices, np.nan)
    r_squared = np.full(n_vertices, np.nan)
    p_value = np.full(n_vertices, np.nan)
    std_err = np.full(n_vertices, np.nan)

    for v in range(n_vertices):
        y = mm[:, v]
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() < 2:
            continue
        fit = linregress(t_years[valid], y[valid])
        slope_mm[v] = fit.slope
        intercept[v] = fit.intercept
        r_squared[v] = fit.rvalue**2
        p_value[v] = fit.pvalue
        std_err[v] = fit.stderr
        if has_nsm:
            y_nsm = nsm[:, v]
            valid_nsm = np.isfinite(y_nsm) & (y_nsm > 0)
            if valid_nsm.sum() >= 2:
                slope_nsm[v] = linregress(t_years[valid_nsm], y_nsm[valid_nsm]).slope
        if baseline[v] > 0 and np.isfinite(baseline[v]):
            slope_pct[v] = fit.slope / baseline[v] * 100.0

    scalars = {
        "thinning_rate (mm/year)": slope_mm,
        "thinning_rate (pct/year)": slope_pct,
        "intercept (mm)": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "std_err (mm/year)": std_err,
        "n_timepoints": np.full(n_vertices, len(times), dtype=np.float32),
        "time_span_years": np.full(n_vertices, t_years[-1] - t_years[0], dtype=np.float32),
        "baseline_thickness (mm)": baseline.copy(),
        "mean_thickness (mm)": np.nanmean(masked, axis=0),
        "std_thickness (mm)": np.nanstd(masked, axis=0),
    }
    if has_nsm:
        scalars["thinning_rate (nsm/year)"] = slope_nsm
    for visit, _ in times:
        scalars[f"thickness_{visit} (mm)"] = mm_by_visit[visit]
    return scalars


def run_rates(args) -> int:
    from pymskt.mesh import Mesh

    bones = BONES if args.bone == "both" else (args.bone,)
    work, skipped, insufficient = [], 0, 0
    for knee, visits in visit_coverage().items():
        for bone in bones:
            if thinning_rate_path(knee, bone).is_file() and not args.force:
                skipped += 1
                continue
            available = [
                visit
                for visit in visits
                if visit in VISIT_YEARS
                and on_reference_paths(knee, visit, bone)[1].is_file()
            ]
            if len(available) < 2:
                insufficient += 1
                continue
            work.append((knee, bone, available))
    print(f"{len(work)} to fit, {skipped} already done, {insufficient} with <2 visits")
    if args.limit:
        work = work[: args.limit]
    if args.check or not work:
        return 0

    references = {
        bone: Mesh(str(reference_surface_path(bone)))
        for bone in sorted({w[1] for w in work})
    }
    failed = 0
    for i, (knee, bone, available) in enumerate(work, 1):
        try:
            scalars = thinning_scalars(knee, bone, available)
            if scalars is None:
                failed += 1
                continue
            mesh = references[bone].copy()
            rate = scalars["thinning_rate (mm/year)"]
            # VTK files cannot carry NaN: absent values are written as zero,
            # with a separate mask saying which vertices have data.
            mesh.set_scalar("has_data", np.isfinite(rate).astype(np.float32))
            for name, values in scalars.items():
                mesh.set_scalar(name, np.where(np.isfinite(values), values, 0.0))
            mesh.set_active_scalars("thinning_rate (mm/year)")
            output = thinning_rate_path(knee, bone)
            output.parent.mkdir(parents=True, exist_ok=True)
            mesh.save_mesh(str(output))
        except Exception as exc:
            failed += 1
            print(f"FAILED {knee} {bone}: {exc}")
        if i % 100 == 0:
            print(f"  {i}/{len(work)}", flush=True)
    print(f"fitted {len(work) - failed}, failed {failed}")
    return 1 if failed else 0


# -------------------------------------------------------------------- cli ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="derive_thickness.py",
        description="Cartilage thickness and thinning rates on the reference surface.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, runner in (
        ("decode", run_decode),
        ("interpolate", run_interpolate),
        ("rates", run_rates),
    ):
        p = sub.add_parser(name)
        p.add_argument("--bone", choices=["femur", "tibia", "both"], default="both")
        p.add_argument("--limit", type=int, default=0, help="stop after N items")
        p.add_argument(
            "--check", action="store_true",
            help="report the work list and stop; runs nothing",
        )
        p.add_argument(
            "--force", action="store_true",
            help="redo items whose output already exists (rates only)",
        )
        p.set_defaults(func=runner)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
