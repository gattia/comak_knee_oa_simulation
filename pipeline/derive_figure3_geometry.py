#!/usr/bin/env python3
"""Derive Figure 3's geometry inputs: 02_simulations -> 04_results/figure3_shape_pressure.

The pressure panels come from ``derive_figure3_data.py``. This step makes the two
inputs that describe the *shapes* themselves, from the same synthetic tree:

``shape_metrics.csv``
    Per sigma step: meniscal coverage and extrusion, meniscus volumes and
    surface areas, and the femoral and tibial articular-surface areas (panels
    G-I). Lengths in mm; the meshes are stored in metres.
``morph/{panel}__{label}.png``
    The anatomy strip: femoral cartilage seen from below, tibial cartilage seen
    from above, and tibial cartilage with both menisci, at -3, -2, 0, +2, +3 SD.

Both read each knee's reference-size decoded geometry,
``{label}/geometries_nsm_similarity`` -- not the gait-scaled models -- so the
panels describe shape alone.

Every anatomy column is framed in a single shared frame per row, so all five
columns are at the same scale.

Meniscal coverage and extrusion use pymskt's standard tool,
``BoneMesh.set_menisci().compute_meniscal_outcomes()``: the tibial cartilage is
projected onto the bone (cartilage thickness > 0), split medial/lateral by the
sign of z, and extrusion is the meniscus's mediolateral extent beyond the
tibial cartilage rim in the middle of the anteroposterior range. The ``artic``
columns repeat the coverage on the tibial articular surface instead; they are
kept for comparison and not plotted.

Usage
-----
    conda run -n comak-knee-oa python pipeline/derive_figure3_geometry.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comak_knee_oa.paths import stage_dir  # noqa: E402

FIGURE = "figure3_shape_pressure"
SIM_SUBDIR = "synthetic_shape_axis"

# The anatomy strip's columns: the ends of either fitted window, and the mean.
MORPH_LABELS = ("fig5_sigma_m3", "fig5_sigma_m2", "fig5_sigma_p0",
                "fig5_sigma_p2", "fig5_sigma_p3")

CSV_KEYS = ["sigma", "label",
            "medial_cov_bone_pct", "lateral_cov_bone_pct",
            "medial_cov_artic_pct", "lateral_cov_artic_pct",
            "medial_extrusion_mm", "lateral_extrusion_mm",
            "medial_cov_bone_area_mm2", "lateral_cov_bone_area_mm2",
            "medial_cart_bone_area_mm2", "lateral_cart_bone_area_mm2",
            "medial_cov_artic_area_mm2", "lateral_cov_artic_area_mm2",
            "medial_cart_artic_area_mm2", "lateral_cart_artic_area_mm2",
            "med_men_volume_mm3", "lat_men_volume_mm3",
            "med_men_area_mm2", "lat_men_area_mm2",
            "med_men_upper_area_mm2", "med_men_lower_area_mm2",
            "lat_men_upper_area_mm2", "lat_men_lower_area_mm2",
            "femur_artic_area_mm2", "tibia_artic_area_mm2"]

# Mesh paths inside geometries_nsm_similarity/.
TIBIA_BONE = "tibia/tibia_nsm_recon_osim.vtk"
TIBIA_CART = "tibia/tibia_cartilage_nsm_recon_osim.vtk"
TIBIA_ARTIC = "tibia/tibia_articular_surface_osim.vtk"
FEMUR_CART = "femur/femur_cartilage_nsm_recon_osim.vtk"
FEMUR_ARTIC = "femur/femur_articular_surface_osim.vtk"
MED_MEN = "femur/med_men_osim.vtk"
LAT_MEN = "femur/lat_men_osim.vtk"

# Rendering. The meshes are in the right-knee body frame: z medial(-)/lateral(+),
# x anteroposterior, y superoinferior. Anterior is drawn up.
CART_COLOR = "#c9d3da"
MEN_COLOR = "#d49a6a"
PAD_H, PAD_V = 1.16, 1.04
BASE_HEIGHT_PX = 560
VIEWS = {
    "superior": dict(view_dir=(0.0, -1.0, 0.0), up=(1.0, 0.0, 0.0), h_idx=2, v_idx=0),
    "inferior": dict(view_dir=(0.0, 1.0, 0.0), up=(1.0, 0.0, 0.0), h_idx=2, v_idx=0),
}
# panel -> (frame group, view, [(mesh, colour)])
PANELS = {
    "femur_cart": ("femur", "inferior", [(FEMUR_CART, CART_COLOR)]),
    "tibia_cart": ("tibia", "superior", [(TIBIA_CART, CART_COLOR)]),
    "tibia_men": ("tibia", "superior", [(TIBIA_CART, CART_COLOR), (MED_MEN, MEN_COLOR),
                                        (LAT_MEN, MEN_COLOR)]),
}
# The meshes whose union sets a group's shared frame, so rows in a group align.
GROUP_FRAME = {"femur": [FEMUR_CART], "tibia": [TIBIA_CART, MED_MEN, LAT_MEN]}


def sim_root() -> Path:
    return stage_dir("simulations", SIM_SUBDIR)


def geometry_dir(label: str) -> Path:
    return sim_root() / label / "geometries_nsm_similarity"


def sigma_steps() -> list[tuple[float, str]]:
    axis = json.loads((sim_root() / "shape_axis.json").read_text())
    return sorted((meta["k_sigma"], label) for label, meta in axis["shapes"].items()
                  if meta.get("type") == "sigma_step")


# --------------------------------------------------------------------------- #
# Shape metrics                                                               #
# --------------------------------------------------------------------------- #
def _read_mm(path: Path):
    """A mesh in mm, so areas come out in mm² and volumes in mm³."""
    import pyvista as pv

    mesh = pv.read(str(path)).triangulate()
    mesh.points = mesh.points * 1000.0
    return mesh


def _area(path: Path) -> float:
    return float(_read_mm(path).area) if path.is_file() else np.nan


def _volume(path: Path) -> float:
    return float(_read_mm(path).volume) if path.is_file() else np.nan


def _outcomes_bone(g: Path) -> dict:
    """Coverage and extrusion with pymskt's standard tool (bone-projection labels)."""
    from pymskt.mesh import BoneMesh, Mesh
    from pymskt.mesh.meshTools import get_cartilage_properties_at_points

    bone = BoneMesh(mesh=_read_mm(g / TIBIA_BONE))
    cart = Mesh(mesh=_read_mm(g / TIBIA_CART))
    found = get_cartilage_properties_at_points(bone, cart, ray_cast_length=10.0,
                                               n_intersections=2)
    thickness = np.asarray(found[0] if isinstance(found, (list, tuple)) else found)
    z = bone.points[:, 2]
    labels = np.zeros(bone.n_points)
    labels[(thickness > 0) & (z < 0)] = 1     # medial
    labels[(thickness > 0) & (z >= 0)] = 2    # lateral
    bone.point_data["labels"] = labels
    bone.set_menisci(medial_meniscus=Mesh(mesh=_read_mm(g / MED_MEN)),
                     lateral_meniscus=Mesh(mesh=_read_mm(g / LAT_MEN)),
                     medial_cart_label=1, lateral_cart_label=2)
    return bone.compute_meniscal_outcomes()


def _coverage_articular(g: Path) -> dict:
    """Coverage on the tibial articular surface, split medial/lateral by z."""
    import pyvista as pv
    from pymskt.mesh import Mesh
    from pymskt.mesh.mesh_meniscus import compute_meniscal_coverage

    bone, artic = _read_mm(g / TIBIA_BONE), _read_mm(g / TIBIA_ARTIC)
    n_bone = bone.n_points
    bone_faces = bone.faces.reshape(-1, 4)
    artic_faces = artic.faces.reshape(-1, 4).copy()
    artic_faces[:, 1:] += n_bone
    combined = pv.PolyData(np.vstack([bone.points, artic.points]),
                           np.vstack([bone_faces, artic_faces]).ravel())
    combined["labels"] = np.concatenate(
        [np.zeros(n_bone), np.where(artic.points[:, 2] < 0, 1.0, 2.0)])
    return compute_meniscal_coverage(Mesh(combined), Mesh(_read_mm(g / MED_MEN)),
                                     Mesh(_read_mm(g / LAT_MEN)), medial_cart_label=1,
                                     lateral_cart_label=2, scalar_array_name="labels")


def shape_metrics(label: str) -> dict:
    g = geometry_dir(label)
    missing = [p for p in (TIBIA_BONE, TIBIA_CART, TIBIA_ARTIC, MED_MEN, LAT_MEN)
               if not (g / p).is_file()]
    if missing:
        raise FileNotFoundError(f"{g} lacks {missing}")
    b = _outcomes_bone(g)
    a = _coverage_articular(g)
    return {
        "medial_cov_bone_pct": b.get("medial_coverage_percent", np.nan),
        "lateral_cov_bone_pct": b.get("lateral_coverage_percent", np.nan),
        "medial_cov_bone_area_mm2": b.get("medial_covered_area_mm2", np.nan),
        "lateral_cov_bone_area_mm2": b.get("lateral_covered_area_mm2", np.nan),
        "medial_cart_bone_area_mm2": b.get("medial_total_area_mm2", np.nan),
        "lateral_cart_bone_area_mm2": b.get("lateral_total_area_mm2", np.nan),
        "medial_extrusion_mm": b.get("medial_extrusion_mm", np.nan),
        "lateral_extrusion_mm": b.get("lateral_extrusion_mm", np.nan),
        "medial_cov_artic_pct": a.get("medial_coverage_percent", np.nan),
        "lateral_cov_artic_pct": a.get("lateral_coverage_percent", np.nan),
        "medial_cov_artic_area_mm2": a.get("medial_covered_area_mm2", np.nan),
        "lateral_cov_artic_area_mm2": a.get("lateral_covered_area_mm2", np.nan),
        "medial_cart_artic_area_mm2": a.get("medial_total_area_mm2", np.nan),
        "lateral_cart_artic_area_mm2": a.get("lateral_total_area_mm2", np.nan),
        "med_men_volume_mm3": _volume(g / MED_MEN),
        "lat_men_volume_mm3": _volume(g / LAT_MEN),
        "med_men_area_mm2": _area(g / MED_MEN),   # the closed body's whole surface
        "lat_men_area_mm2": _area(g / LAT_MEN),
        "med_men_upper_area_mm2": _area(g / "femur/med_men_upper_art_surf_osim.vtk"),
        "med_men_lower_area_mm2": _area(g / "femur/med_men_lower_art_surf_osim.vtk"),
        "lat_men_upper_area_mm2": _area(g / "femur/lat_men_upper_art_surf_osim.vtk"),
        "lat_men_lower_area_mm2": _area(g / "femur/lat_men_lower_art_surf_osim.vtk"),
        "femur_artic_area_mm2": _area(g / FEMUR_ARTIC),
        "tibia_artic_area_mm2": _area(g / TIBIA_ARTIC),
    }


def write_shape_metrics(out_dir: Path) -> Path:
    rows = []
    for k, label in sigma_steps():
        print(f"  metrics {label} ({k:+g} SD)")
        rows.append({"sigma": float(k), "label": label, **shape_metrics(label)})
    path = out_dir / "shape_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_KEYS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.name}: {len(rows)} sigma steps")
    return path


# --------------------------------------------------------------------------- #
# Anatomy renders                                                             #
# --------------------------------------------------------------------------- #
def _shared_frame(labels, group: str, view: str):
    """Centre and in-plane half-extents covering every label's meshes in a group."""
    import pyvista as pv

    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for label in labels:
        for rel in GROUP_FRAME[group]:
            bounds = np.array(pv.read(str(geometry_dir(label) / rel)).bounds).reshape(3, 2)
            lo, hi = np.minimum(lo, bounds[:, 0]), np.maximum(hi, bounds[:, 1])
    v = VIEWS[view]
    return (lo + hi) / 2.0, float(hi[v["h_idx"]] - lo[v["h_idx"]]) / 2.0, \
        float(hi[v["v_idx"]] - lo[v["v_idx"]]) / 2.0


def _render(label: str, panel: str, out_png: Path, center, half_w, half_h) -> None:
    import pyvista as pv
    from PIL import Image

    group, view, meshes = PANELS[panel]
    v = VIEWS[view]
    width = int(round(BASE_HEIGHT_PX * (half_w / max(half_h, 1e-9)) * (PAD_H / PAD_V)))
    plotter = pv.Plotter(off_screen=True, window_size=[width, BASE_HEIGHT_PX])
    plotter.background_color = "white"
    for rel, color in meshes:
        plotter.add_mesh(pv.read(str(geometry_dir(label) / rel)), color=color, opacity=1.0,
                         smooth_shading=True, specular=0.25, specular_power=15,
                         ambient=0.32, diffuse=0.72)
    view_dir = np.array(v["view_dir"], float)
    view_dir /= np.linalg.norm(view_dir)
    plotter.enable_parallel_projection()
    plotter.camera.focal_point = tuple(center)
    plotter.camera.position = tuple(center - view_dir * (half_w + half_h) * 5.0)
    plotter.camera.up = tuple(v["up"])
    plotter.camera.parallel_scale = half_h * PAD_V
    plotter.screenshot(str(out_png), transparent_background=False)
    plotter.close()
    if group == "tibia":
        # Mirror left-right so the tibia, seen from above, has medial and lateral
        # on the same sides as the femur seen from below.
        Image.open(out_png).transpose(Image.FLIP_LEFT_RIGHT).save(out_png)


def write_morph(out_dir: Path) -> Path:
    morph_dir = out_dir / "morph"
    morph_dir.mkdir(exist_ok=True)
    for panel, (group, view, _) in PANELS.items():
        center, half_w, half_h = _shared_frame(MORPH_LABELS, group, view)
        for label in MORPH_LABELS:
            _render(label, panel, morph_dir / f"{panel}__{label}.png", center, half_w, half_h)
    print(f"wrote morph/: {len(PANELS) * len(MORPH_LABELS)} renders")
    return morph_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("metrics", "morph"),
                        help="build one of the two outputs")
    args = parser.parse_args()
    out_dir = stage_dir("results", FIGURE, create=True)
    if args.only in (None, "metrics"):
        write_shape_metrics(out_dir)
    if args.only in (None, "morph"):
        write_morph(out_dir)
    print(f"\nFigure 3 geometry inputs in {out_dir}")


if __name__ == "__main__":
    main()
