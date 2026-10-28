#!/usr/bin/env python3
"""Derive Extended Data Figure 4: how the prefemoral fat-pad contact surface is built.

    python pipeline/derive_figure_ed4_fatpad.py
    python pipeline/derive_figure_ed4_fatpad.py --knee 9805712_00m_RIGHT

The fat pad is made inside the model build by one nsosim call,
``create_prefemoral_fatpad_noboolean``, which returns only the finished surface.
To draw the steps, this script repeats that function stage by stage on one knee's
built geometry, calling nsosim's own helpers for every stage, and keeps what each
stage produced. It then renders the 3-D panels and writes, into
``04_results/figure_ed4_fatpad/``:

    panels/ref.png        reference model: femur bone + its superiorly extended cartilage
    panels/subj.png       the knee's shape-model femur + cartilage
    panels/seed.png       step 1: exposed anterior bone (the candidate region)
    panels/seed_pat.png   step 1 again, with the patella at its MRI position
    panels/vec.png        step 2: the offsets as arrows; red = stopped short at the patella
    panels/dil.png        step 2: the dilated surface coloured by offset
    panels/crop.png       step 3: kept (< 25 mm from the patella) vs discarded
    panels/final.png      step 4: the smoothed, resampled fat pad
    section.npz           sagittal cross-section through the patella centre (polylines, mm)
    summary.json          the knee, the parameters, per-stage counts, the checks below,
                          each panel's crop offset and where panel C's arrow points

``figures/figure_ed4_fatpad.py`` composes these and reads nothing else.

Checks (the script raises if either fails)
------------------------------------------
- The stage-by-stage repeat equals one call of ``create_prefemoral_fatpad_noboolean``
  with the same arguments, so the figure shows what the function does.
- That call reproduces the fat pad saved beside the built model, so the arguments
  are the ones the model was built with. Not bit for bit: the build passes the
  meshes in memory, this reads them back from VTK files, and 1e-12 m of difference
  can re-triangulate the final resample; the tolerance is 0.05 mm
  average surface distance.

The parameters are the fat-pad arguments of ``nsosim.model_building.build_joint_model``
(base 1 mm, top 6 mm, 25 mm patella crop, 5,000 clusters, 10 mm rays, log profile,
100 smoothing iterations); this repository does not override them. The ray direction
(``fatpad_clearance_rays``) is read from the knee's ``run_record.json``, i.e. how
that model was built; a knee with no record (the v2 data root) was built with
``auto_oriented``, nsosim's default.

Which knee
----------
9805712_00m_RIGHT. The patella-clearance limit
engages on it (it shortens the offset of about 360 seed vertices), so the figure
can show that step; on the verification knee 9018389 it never engages. A right
knee, so the femur-frame axes match the reference model's.

Run: ~3 minutes on one CPU. Needs nsosim and pymskt (the pins), and an off-screen
VTK.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import Knee, load_config  # noqa: E402
from comak_knee_oa.paths import UndeclaredRunError, data_root, stage_dir  # noqa: E402

FIGURE = "figure_ed4_fatpad"
KNEE = "9805712_00m_RIGHT"
RECONSTRUCTION_CONFIG = _REPO / "config" / "reconstruction.json"

# The fat-pad arguments of nsosim.model_building.build_joint_model (its cfg() defaults;
# this repository passes none of them, so these are what every model was built with).
PARAMS = dict(
    base_mm=1.0,
    top_mm=6.0,
    max_distance_to_patella_mm=25.0,
    percent_fem_cart_to_keep=0.15,      # create_prefemoral_fatpad_noboolean's default
    resample_clusters_final=5000,
    ray_cast_length=10.0,
    safety_mm=0.0,                      # create_prefemoral_fatpad_noboolean's default
    norm_function="log",
    final_smooth_iter=100,
    scale_percentile=95.0,              # fixed inside create_prefemoral_fatpad_noboolean
)
AS_BUILT_TOLERANCE_MM = 0.05            # average surface distance, repeat vs the saved fat pad

# ----------------------------------------------------------------- rendering -------------
COL = dict(
    bone="#EAE3D3", cart="#8FB8DE", seed="#D64545", seed_pale="#F2C4C4",
    fat="#F2B134", fat_edge="#7A4E00", discard="#A9A9A9",
    arrow="#2B2B2B", arrow_cap="#D64545",
)
WIN = (1500, 1500)
PARALLEL_SCALE = 44.0  # mm half-height shared by every 3-D panel -> identical scale
PAT_OPACITY = (0.30, 0.40)  # bone, cartilage
FOCAL = np.array([22.0, 12.0, 0.0])
VIEW_OBLIQUE = (1.0, 0.35, 0.95)   # antero-lateral, slightly from above
VIEW_ANTERIOR = (1.0, 0.15, 0.25)
VIEW_SIDE = (0.35, 0.30, 1.0)      # more lateral: shows the vectors in profile
BG = "white"


# --------------------------------------------------------------------------- inputs ----
def knee_dir(knee: str) -> Path:
    """The knee's built model, where config/reconstruction.json puts it.

    The v2 data root declares no run and keeps its models under
    ``cohorts.V2_MODELS_RUN``.
    """
    try:
        return load_config(RECONSTRUCTION_CONFIG, Knee.parse(knee)).path("run", "output_dir")
    except UndeclaredRunError:
        return stage_dir("models", cohorts.V2_MODELS_RUN, knee)


def reference_geometry_dir() -> Path:
    """The base model's Geometry folder (the reference femur and its extended cartilage)."""
    return load_config(RECONSTRUCTION_CONFIG).path("inputs", "base_model") / "Geometry"


def input_paths(knee: str) -> dict[str, Path]:
    geom = knee_dir(knee) / "geometries_nsm_similarity"
    ref = reference_geometry_dir()
    paths = {
        "femur_bone": geom / "femur" / "femur_nsm_recon_osim.vtk",
        "femur_cart": geom / "femur" / "femur_cartilage_nsm_recon_osim.vtk",
        # build_joint_model passes the patella at its MRI position; the centred copy is only saved
        "pat_bone": geom / "patella" / "patella_nsm_recon_osim_original_position.vtk",
        "pat_cart": geom / "patella" / "patella_cartilage_nsm_recon_osim_original_position.vtk",
        "fatpad_as_built": geom / "femur" / "femur_prefemoral_fat_pad.vtk",
        "ref_bone": ref / "smith2019-R-femur-bone.stl",
        "ref_cart": ref / "smith2019-R-femur-cartilage.stl",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing input(s):\n    " + "\n    ".join(missing))
    return paths


def clearance_rays_as_built(knee: str) -> tuple[str, str]:
    """(ray direction the knee's fat pad was built with, where that came from)."""
    record = knee_dir(knee) / "run_record.json"
    if record.is_file():
        rays = json.loads(record.read_text()).get("build_options", {}).get("fatpad_clearance_rays")
        if rays:
            return rays, "run_record.json build_options"
    return "auto_oriented", "no run record: nsosim's default, as the published v2 models were built"


def code_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ------------------------------------------------------------------ the repeat ---------
def build_stages(paths: dict[str, Path], clearance_rays: str) -> tuple[dict, dict]:
    """Repeat create_prefemoral_fatpad_noboolean stage by stage -> (meshes, stats).

    Every step is the function's own code or a call to the helper it calls; the
    comments name the step of the function they repeat.
    """
    import pyvista as pv
    from nsosim.articular_surfaces import (
        _finalize_fatpad_mesh,
        _forward_clearance_to_targets,
        _prepare_fatpad_input_meshes,
        _y_profile_mm,
        create_prefemoral_fatpad_noboolean,
        dilate_mesh_with_profile_and_clearance,
    )
    from pymskt.mesh import Mesh

    P = PARAMS
    stats: dict = {}

    # Step 0: load, convert m -> mm
    raw = {k: pv.read(str(paths[k])) for k in ("femur_bone", "femur_cart", "pat_bone", "pat_cart")}
    femur_bone, femur_cart, pat_bone, pat_cart = _prepare_fatpad_input_meshes(
        raw["femur_bone"], raw["femur_cart"], raw["pat_bone"], raw["pat_cart"], "m"
    )
    # Step 1: clean + consistent normals
    femur_bone = femur_bone.clean()
    femur_cart = femur_cart.clean()
    pat_bone = pat_bone.clean()
    pat_cart = pat_cart.clean()
    for m in (femur_bone, femur_cart, pat_bone, pat_cart):
        m.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    for k, m in (("femur_bone", femur_bone), ("femur_cart", femur_cart),
                 ("pat_bone", pat_bone), ("pat_cart", pat_cart)):
        stats[f"n_{k}"] = int(m.n_points)

    # Step 2: exposed anterior bone
    fb = Mesh(femur_bone) if not isinstance(femur_bone, Mesh) else femur_bone
    fb.calc_distance_to_other_mesh(
        femur_cart, ray_cast_length=10.0, percent_ray_length_opposite_direction=0.1,
        name="cart_coverage",
    )
    exposed = fb.get_scalar("cart_coverage") == 0
    anterior = fb.points[:, 0] > np.mean(fb.points[:, 0])
    start_mask = exposed & anterior
    labeled = pv.PolyData(fb.copy())
    labeled.point_data["start_region"] = start_mask.astype(float)
    stats.update(n_exposed=int(exposed.sum()), n_anterior=int(anterior.sum()),
                 n_start=int(start_mask.sum()))
    starting_surface = Mesh(fb.remove_points(~start_mask)[0]).clean()

    # Step 3: profile, clearance, dilation
    ref_filter = lambda pts: pts[:, 0] > np.mean(pts[:, 0])  # noqa: E731
    target = _y_profile_mm(
        starting_surface, P["base_mm"], P["top_mm"], scale_axis=1,
        reference_mesh=femur_cart, reference_axis_filter=ref_filter,
        scale_percentile=P["scale_percentile"], norm_function=P["norm_function"],
    )
    clearance = _forward_clearance_to_targets(
        source_mesh=Mesh(starting_surface.copy()), targets=[pat_bone, pat_cart],
        ray_cast_length=P["ray_cast_length"], safety_mm=P["safety_mm"],
        directions=(np.asarray(starting_surface.point_normals)
                    if clearance_rays == "dilation_normals" else None),
    )
    dilated, actual = dilate_mesh_with_profile_and_clearance(
        femur_bone_mesh=starting_surface, patella_bone_mesh=pat_bone,
        patella_cart_mesh=pat_cart, base_mm=P["base_mm"], top_mm=P["top_mm"],
        scale_axis=1, reference_mesh=femur_cart, reference_axis_filter=ref_filter,
        scale_percentile=P["scale_percentile"], norm_function=P["norm_function"],
        ray_cast_length=P["ray_cast_length"], safety_mm=P["safety_mm"], mask=None,
        clearance_rays=clearance_rays,
    )
    if not np.allclose(actual, np.minimum(target, clearance)):
        raise RuntimeError("the stage-by-stage repeat drifted from nsosim's dilation")
    limited = clearance < target - 1e-9
    start = pv.PolyData(starting_surface.copy())
    dilated = pv.PolyData(dilated.copy())
    for arrs in (start.point_data, dilated.point_data):
        arrs["offset_mm"] = actual
        arrs["clearance_limited"] = limited.astype(float)
    ant_cart_y = femur_cart.points[femur_cart.points[:, 0] > np.mean(femur_cart.points[:, 0]), 1]
    finite = np.where(np.isinf(clearance), np.nan, clearance)
    stats.update(
        y_ramp_start_mm=float(np.percentile(ant_cart_y, P["scale_percentile"])),
        y_top_of_start_surface_mm=float(np.max(starting_surface.points[:, 1])),
        n_clearance_limited=int(limited.sum()),
        clearance_finite_min_mm=(float(np.nanmin(finite)) if np.isfinite(finite).any() else None),
    )

    # Step 4: crop to patella proximity + vertical threshold
    sdf_b = Mesh(pat_bone).get_sdf_pts(dilated.points)
    sdf_c = Mesh(pat_cart).get_sdf_pts(dilated.points)
    d_pat = np.minimum(np.abs(sdf_b), np.abs(sdf_c))
    y_thresh = float(np.percentile(ant_cart_y, (1 - P["percent_fem_cart_to_keep"]) * 100))
    keep = (d_pat < P["max_distance_to_patella_mm"]) & (dilated.points[:, 1] > y_thresh)
    dilated.point_data["keep"] = keep.astype(float)
    cropped = Mesh(dilated.remove_points(~keep)[0])
    kept_off = actual[keep]
    stats.update(
        y_crop_threshold_mm=y_thresh, n_keep=int(keep.sum()),
        kept_offset_min_mm=float(kept_off.min()),
        kept_offset_median_mm=float(np.median(kept_off)),
        kept_offset_max_mm=float(kept_off.max()),
        kept_frac_clearance_limited=float(limited[keep].mean()),
    )

    # Step 5: smooth, resample, largest component
    final = _finalize_fatpad_mesh(
        Mesh(cropped.copy()), P["resample_clusters_final"], "mm", P["final_smooth_iter"], None
    )
    final = pv.PolyData(final.copy())
    stats.update(final_n_points=int(final.n_points), final_n_cells=int(final.n_cells))

    # Check 1: the repeat equals one call of the function
    whole = create_prefemoral_fatpad_noboolean(
        raw["femur_bone"], raw["femur_cart"], raw["pat_bone"], raw["pat_cart"],
        base_mm=P["base_mm"], top_mm=P["top_mm"],
        max_distance_to_patella_mm=P["max_distance_to_patella_mm"],
        percent_fem_cart_to_keep=P["percent_fem_cart_to_keep"],
        resample_clusters_final=P["resample_clusters_final"], units="m",
        ray_cast_length=P["ray_cast_length"], safety_mm=P["safety_mm"],
        norm_function=P["norm_function"], final_smooth_iter=P["final_smooth_iter"],
        clearance_rays=clearance_rays,
    )
    whole.points = whole.points * 1000.0  # units="m": returns metres
    stats["assd_repeat_vs_function_mm"] = float(Mesh(final.copy()).get_assd_mesh(whole))
    if stats["assd_repeat_vs_function_mm"] > 1e-6:
        raise RuntimeError(f"the stage-by-stage repeat differs from create_prefemoral_fatpad_noboolean "
                           f"by {stats['assd_repeat_vs_function_mm']:.3g} mm")

    # Check 2: the function reproduces the fat pad the model was built with
    as_built = Mesh(str(paths["fatpad_as_built"]))
    as_built.points = as_built.points * 1000.0
    stats["assd_repeat_vs_as_built_mm"] = float(Mesh(final.copy()).get_assd_mesh(as_built))
    if stats["assd_repeat_vs_as_built_mm"] > AS_BUILT_TOLERANCE_MM:
        raise RuntimeError(f"the repeat is {stats['assd_repeat_vs_as_built_mm']:.3g} mm from the fat pad "
                           f"saved with the model (> {AS_BUILT_TOLERANCE_MM} mm): the parameters or the "
                           f"ray direction are not the ones it was built with")

    meshes = dict(femur=pv.PolyData(femur_bone.copy()), cart=pv.PolyData(femur_cart.copy()),
                  pat=pv.PolyData(pat_bone.copy()), patc=pv.PolyData(pat_cart.copy()),
                  labeled=labeled, start=start, dil=dilated, final=final)
    return meshes, stats


# ------------------------------------------------------------------- reference -----------
def reference_meshes(paths: dict[str, Path], y_clip: float) -> tuple:
    """Reference femur bone (clipped to the knee's proximal extent) and cartilage, in mm.

    The reference cartilage dips up to ~1.2 mm beneath the reference bone surface, so
    the bone would be drawn over it. For display only, the cartilage-covered bone
    vertices are sunk 2 mm inward (coverage by a normal ray cast on the closed bone,
    before clipping, so the auto-oriented normals are trustworthy). The bone stays
    continuous; no hole, no rim.
    """
    import pyvista as pv
    from pymskt.mesh import Mesh

    ref_bone = pv.read(str(paths["ref_bone"]))
    ref_cart = pv.read(str(paths["ref_cart"]))
    for m in (ref_bone, ref_cart):
        m.points = m.points * 1000.0
    rb = ref_bone.copy()
    rb.compute_normals(inplace=True, auto_orient_normals=True, consistent_normals=True)
    rbm = Mesh(rb)
    rbm.calc_distance_to_other_mesh(ref_cart, ray_cast_length=10.0,
                                    percent_ray_length_opposite_direction=0.3, name="cov")
    covered = rbm.get_scalar("cov") != 0
    pts = rb.points.copy()
    pts[covered] -= 2.0 * rb.point_normals[covered]
    rb.points = pts
    return rb.clip(normal=(0, 1, 0), origin=(0, y_clip, 0)), ref_cart


# ------------------------------------------------------------------- 3-D panels ---------
def _rgb255(c):
    from matplotlib import colors
    return np.round(np.array(colors.to_rgb(c)) * 255).astype(int)


def shoot(p, pad=12):
    """Screenshot, crop to everything that is not background -> (image, (x0, y0)).

    The crop offset lets ``screen_xy`` place an annotation in the saved image's pixels.
    """
    img = p.screenshot(return_img=True)[..., :3]
    p.close()
    diff = np.abs(img.astype(int) - _rgb255(BG)).max(axis=2)
    ys, xs = np.where(diff > 5)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    return img[y0:y1, x0:x1], (int(x0), int(y0))


def screen_xy(points, direction, focal, crop_xy, scale=PARALLEL_SCALE, win=WIN):
    """World (mm) -> pixel coordinates in a cropped screenshot (parallel camera)."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    up = np.array([0.0, 1.0, 0.0])
    up = up - d * np.dot(up, d)
    up /= np.linalg.norm(up)
    right = np.cross(-d, up)          # camera looks along -d
    rel = np.atleast_2d(points) - np.asarray(focal)
    x = win[0] / 2 + rel @ right / scale * win[1] / 2
    y = win[1] / 2 - rel @ up / scale * win[1] / 2
    x0, y0 = crop_xy
    return np.column_stack([x - x0, y - y0])


def new_plotter():
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=WIN)
    p.set_background(BG)
    p.enable_anti_aliasing("ssaa")
    return p


def set_view(p, direction, focal=FOCAL, scale=PARALLEL_SCALE):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    p.camera_position = [tuple(np.asarray(focal) + 400 * d), tuple(focal), (0, 1, 0)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = scale


def add(p, mesh, **kw):
    kw.setdefault("smooth_shading", True)
    kw.setdefault("specular", 0.15)
    kw.setdefault("show_scalar_bar", False)
    return p.add_mesh(mesh, **kw)


def farthest_points(points, n, start=0):
    """Greedy farthest-point sampling: n indices spread evenly over ``points``."""
    if len(points) == 0 or n <= 0:
        return np.array([], dtype=int)
    n = min(n, len(points))
    sel = [start]
    d = np.linalg.norm(points - points[start], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return np.array(sel)


def arrows(points, dirs, lengths, shaft_r=0.45, tip_r=0.95):
    """Solid arrows with absolute radii (mm) so a 1 mm arrow is as thick as a 6 mm one."""
    import pyvista as pv
    parts = []
    for p, d, L in zip(points, dirs, lengths):
        d = np.asarray(d, float)
        d /= np.linalg.norm(d)
        tip = min(0.45 * L, 1.8)
        shaft = L - tip
        if shaft > 0.05:
            parts.append(pv.Cylinder(center=p + d * shaft / 2, direction=d, radius=shaft_r,
                                     height=shaft, resolution=20))
        parts.append(pv.Cone(center=p + d * (shaft + tip / 2), direction=d, height=tip,
                             radius=min(tip_r, 0.9 * L), resolution=20))
    return pv.merge(parts)


def render_panels(g: dict, ref_bone, ref_cart) -> tuple[dict, dict, dict]:
    """The eight 3-D views -> (images, crop offsets, extra stats), keyed by panel name."""
    from matplotlib import colors

    seed_mask = g["labeled"]["start_region"] > 0.5

    def tinted_femur(mask, colour):
        m = g["labeled"].copy()
        rgb = np.tile(np.array(colors.to_rgb(COL["bone"])), (m.n_points, 1))
        rgb[mask] = colors.to_rgb(colour)
        m["rgb"] = (rgb * 255).astype(np.uint8)
        return m

    def add_patella(p):
        add(p, g["pat"], color=COL["bone"], opacity=PAT_OPACITY[0])
        add(p, g["patc"], color=COL["cart"], opacity=PAT_OPACITY[1])

    def bone_and_cart(p):
        add(p, g["femur"], color=COL["bone"])
        add(p, g["cart"], color=COL["cart"])

    out, extra = {}, {}

    p = new_plotter()
    add(p, ref_bone, color=COL["bone"])
    add(p, ref_cart, color=COL["cart"])
    set_view(p, VIEW_ANTERIOR)
    out["ref"] = shoot(p)

    p = new_plotter()
    bone_and_cart(p)
    set_view(p, VIEW_ANTERIOR)
    out["subj"] = shoot(p)

    for name, with_patella in (("seed", False), ("seed_pat", True)):
        p = new_plotter()
        add(p, tinted_femur(seed_mask, COL["seed"]), scalars="rgb", rgb=True)
        add(p, g["cart"], color=COL["cart"])
        if with_patella:
            add_patella(p)
        set_view(p, VIEW_OBLIQUE)
        out[name] = shoot(p)

    # Displacement vectors on the bare seed surface. Length = the offset actually applied;
    # red vectors were shortened by the patella clearance and end at the patella. Only
    # vectors this view sees in profile are drawn (a vector aimed at the camera is a dot).
    start, dil = g["start"], g["dil"]
    p = new_plotter()
    add(p, tinted_femur(seed_mask, COL["seed_pale"]), scalars="rgb", rgb=True)
    add(p, g["cart"], color=COL["cart"])
    limited = start["clearance_limited"] > 0.5
    keep = dil["keep"] > 0.5  # same vertex order as ``start``
    pts, nrm, off = start.points, np.asarray(start.point_normals), start["offset_mm"]
    d = np.asarray(VIEW_SIDE, float)
    d /= np.linalg.norm(d)
    in_profile = np.abs(nrm @ d) < 0.6
    free_idx = np.where(keep & ~limited & in_profile)[0]
    cap_idx = np.where(keep & limited & in_profile)[0]
    sel_free = free_idx[farthest_points(pts[free_idx], 42)]
    sel_cap = cap_idx[farthest_points(pts[cap_idx], 18)]
    if len(sel_free):
        add(p, arrows(pts[sel_free], nrm[sel_free], off[sel_free]), color=COL["arrow"])
    if len(sel_cap):
        add(p, arrows(pts[sel_cap], nrm[sel_cap], off[sel_cap]), color=COL["arrow_cap"])
    add_patella(p)
    set_view(p, VIEW_SIDE)
    out["vec"] = shoot(p)
    extra.update(n_vectors_free=int(len(sel_free)), n_vectors_capped=int(len(sel_cap)))

    p = new_plotter()
    bone_and_cart(p)
    add(p, dil, scalars="offset_mm", cmap="viridis", clim=[1, 6])
    set_view(p, VIEW_OBLIQUE)
    out["dil"] = shoot(p)

    p = new_plotter()
    bone_and_cart(p)
    kept = dil.extract_points(np.where(keep)[0], adjacent_cells=False).extract_surface()
    gone = dil.extract_points(np.where(~keep)[0], adjacent_cells=False).extract_surface()
    add(p, gone, color=COL["discard"], opacity=0.65)
    add(p, kept, color=COL["fat"])
    set_view(p, VIEW_OBLIQUE)
    out["crop"] = shoot(p)

    p = new_plotter()
    bone_and_cart(p)
    add(p, g["final"], color=COL["fat"], show_edges=True, edge_color=COL["fat_edge"], line_width=0.4)
    set_view(p, VIEW_OBLIQUE)
    out["final"] = shoot(p)

    # Panel C's arrow: the centroid of the vertices the patella clearance capped
    capped = dil["clearance_limited"] > 0.5
    if capped.any():
        xy = screen_xy(dil.points[capped].mean(axis=0), VIEW_OBLIQUE, FOCAL, out["dil"][1])[0]
        extra["capped_patch_xy_px"] = [float(xy[0]), float(xy[1])]
    else:
        extra["capped_patch_xy_px"] = None

    return ({k: v[0] for k, v in out.items()}, {k: list(v[1]) for k, v in out.items()}, extra)


# ------------------------------------------------------------------- section -----------
def section(g: dict) -> dict[str, list[np.ndarray]]:
    """Sagittal (z = patellar-cartilage centre) cross-section polylines, (x, y) in mm."""
    zc = float(g["patc"].points[:, 2].mean())

    def lines(mesh):
        sl = mesh.slice(normal=(0, 0, 1), origin=(0, 0, zc)).strip()
        return [np.asarray(sl.get_cell(i).points)[:, :2] for i in range(sl.n_cells)]

    return dict(z_mm=zc, bone=lines(g["femur"]) + lines(g["pat"]),
                cartilage=lines(g["cart"]) + lines(g["patc"]), fatpad=lines(g["final"]))


def save_section(sec: dict, path: Path) -> None:
    arrays = {"z_mm": np.array(sec["z_mm"])}
    for kind in ("bone", "cartilage", "fatpad"):
        for i, pl in enumerate(sec[kind]):
            arrays[f"{kind}_{i:03d}"] = pl
    np.savez_compressed(path, **arrays)


# --------------------------------------------------------------------------- main ------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knee", default=KNEE,
                        help=f"knee to draw (default {KNEE}, where the patella clearance engages)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv

    pv.OFF_SCREEN = True
    paths = input_paths(args.knee)
    rays, rays_source = clearance_rays_as_built(args.knee)
    print(f"{args.knee}: clearance rays = {rays} ({rays_source})")

    meshes, stats = build_stages(paths, rays)
    print(json.dumps(stats, indent=2))
    ref_bone, ref_cart = reference_meshes(paths, y_clip=float(meshes["femur"].bounds[3]))
    images, crops, extra = render_panels(meshes, ref_bone, ref_cart)
    sec = section(meshes)

    out = stage_dir("results", FIGURE, create=True)
    (out / "panels").mkdir(exist_ok=True)
    for name, img in images.items():
        plt.imsave(out / "panels" / f"{name}.png", img)
    save_section(sec, out / "section.npz")

    root = data_root()
    summary = {
        "figure": FIGURE,
        "knee": args.knee,
        "clearance_rays": rays,
        "clearance_rays_source": rays_source,
        "params": PARAMS,
        "stats": stats,
        "panel_crop_offsets_px": crops,
        "render_window_px": list(WIN),
        "parallel_scale_mm": PARALLEL_SCALE,
        **extra,
        "inputs": {k: str(p.relative_to(root)) if root in p.parents else str(p.name)
                   for k, p in paths.items()},
        "code_commit": code_commit(),
        "derived_on": date.today().isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
