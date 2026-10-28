#!/usr/bin/env python3
"""Derive Figure 1's rendered panels: the pipeline overview's 3D pictures and maps.

    conda run -n comak-knee-oa python pipeline/derive_figure1_panels.py            # every panel
    conda run -n comak-knee-oa python pipeline/derive_figure1_panels.py body hub   # some of them

Figure 1 is a schematic of the pipeline, drawn around one example knee
(``9018389_00m_RIGHT``, the pipeline's verification subject) and one measured
gait (``Subject_102``). Every picture in it that shows data is rendered here,
from the data root, into ``04_results/figure1_pipeline_overview/``:

    model_probe.json                    body poses, wrap surfaces, muscle paths and
                                        ligament / tendon attachment points of the
                                        knee's built OpenSim model (needs OpenSim)
    panels/seg_v2.png                   the segmentation meshes, flat-shaded (rough)
    panels/hub_knee.png                 the neural-shape-model reconstruction
    panels/exploded_nofat_bigtriads.png the exploded multibody knee: bones, cartilage,
                                        menisci, wrap surfaces, attachment sites and
                                        the four joints' coordinate triads
    panels/body.png                     the full-body model, right leg with muscles
    panels/pressure_{pk1,mid,pk2}.png   femoral cartilage pressure at three instants
    panels/pressure_cum.png             the same cartilage's cumulative pressure
    panels/dthick_{healthy,incident}.png  group-mean Delta cartilage thickness on the
                                        reference surface
    panels/synthetic_{m3,p3}.png        the synthetic femur at -3 and +3 SD along
                                        the healthy -> mild-OA shape axis
    panels/*_anchors.json               pixel positions of named points in the
                                        trimmed image (origin top-left), so the
                                        figure can land arrows on them
    provenance.json                     inputs (relative to the data root), camera,
                                        colour limits, knee, gait, and the QC set

Two kinds of panel are not made here, because their source is not in the data
root: the raw DESS slice and its segmentation overlay (the OAI DICOM is not
vendored) and hand-drawn artwork (walking silhouettes, the crowd and scanner
icons). Those are vendored under ``figures/assets/figure1/``; its README says
where each came from.

Every render uses pyvista off screen (OSMesa in the ``comak`` env). The model
probe imports OpenSim, which in that env loads only through ``conda run``
(or with ``$CONDA_PREFIX/lib`` on ``LD_LIBRARY_PATH``); it runs once and is
cached in ``model_probe.json``, so the renders need no OpenSim afterwards.

Exploded-knee layout is set by ``PATELLA_LIFT`` and ``TRIAD_SIZE``; the fat pad
is not drawn. The Delta-thickness maps exclude knees through the shared QC table
(``figures/shared/qc.py``: the Subject_102 union set plus every chronic knee).
``provenance.json`` also records a narrower filter (this gait's union flags plus
one chronic knee) and how its set differs from the shared one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import Knee, load_config  # noqa: E402
from comak_knee_oa.paths import UndeclaredRunError, data_root, stage_dir  # noqa: E402
from derive_thickness import reference_surface_path  # noqa: E402
from shared import qc  # noqa: E402

FIGURE = "figure1_pipeline_overview"

# The example knee and the gait its pressure is shown under. Where the knee's
# model and simulation live, and the quality-control key of its gait, carry the
# run's name, so they are read from the configs and the QC table (below).
KNEE = "9018389_00m_RIGHT"
KNEE_SIDE = "9018389_RIGHT"
GAIT = "Subject_102"
RECONSTRUCTION_CONFIG = _REPO / "config" / "reconstruction.json"

# Colours: sand bone, cartilage blue, orange menisci,
# red ligament / purple tendon sites, muted green-grey wraps with a darker outline.
BONE = "#c9c3b6"
BONE_SEG = "#b5ad9b"
MENI = "#e8a26a"
CART = "#5aa9e6"
CART_MEN = "#8cc4ee"
LIG = "#c0392b"
TEN = "#7E57C2"
WRAP = "#8FB3A3"
WRAP_EDGE = "#4F7F6C"
TRIAD = "#37474F"
GREY = "#c9c9c9"
MUSCLE = "#a86a6a"
TRI = {"TF": "#7B1FA2", "PF": "#C2185B", "MM": "#EF6C00", "LM": "#EF6C00"}   # keyed to the figure's callout
MAT = dict(smooth_shading=True, specular=0.03, specular_power=10, ambient=0.40, diffuse=0.62)
MATTE = dict(smooth_shading=True, specular=0.02, ambient=0.30, diffuse=0.65)

# Exploded knee: the patella sits close to the femur (no fat pad is drawn), and
# the triads are long enough to read at figure size.
PATELLA_LIFT = 0.118          # m, superior offset of the patella in the explosion
TRIAD_SIZE = 0.034            # m, arrow length (tip and shaft radii scale with it)
ATTACH_MAX_DIST = 0.004       # m, attachment points farther from their bone are not drawn

PRESSURE_CLIM = (0.0, 6.0)    # MPa, the instants
CUM_CLIM = (0.0, 3.0)         # MPa*s, the cumulative map
DTHICK_CLIM = (-50.0, 50.0)   # um/yr; group means run p5..p95 = -45..+33
MIN_KNEES = 10                # per vertex; with fewer contributing knees it is left grey
SYNTHETIC_SHOWN = ("m3", "p3")
SYNTHETIC_FRAMED = ("m3", "m2", "p0", "p2", "p3")   # one camera + scale for all of these

ALL_STEPS = ("probe", "seg", "hub", "exploded", "body", "pressure", "dthick", "synthetic")


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def out_dir() -> Path:
    return stage_dir("results", FIGURE, create=True)


def panels_dir() -> Path:
    path = out_dir() / "panels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _knee() -> Knee:
    return Knee.parse(KNEE)


def knee_dir() -> Path:
    """The knee's reconstruction, where config/reconstruction.json puts it.

    The config names the run; a data root whose config declares no run keeps its
    reconstructions under ``cohorts.V2_MODELS_RUN``.
    """
    try:
        return load_config(RECONSTRUCTION_CONFIG, _knee()).path("run", "output_dir")
    except UndeclaredRunError:
        return stage_dir("models", cohorts.V2_MODELS_RUN, KNEE)


def model_run() -> str:
    """The model version of the built model the figure read: the suffix on its name.

    Read off the model itself, so it is right on a run-named root and on a root
    that declares no run alike.
    """
    prefix = f"full_body_healthy_knee_nsm_{KNEE}_"
    stem = osim_path().stem
    if not stem.startswith(prefix):
        raise ValueError(f"unexpected built-model name {stem!r}; expected {prefix}<version>")
    return stem[len(prefix):]


def cohort_tag() -> str:
    """``GAIT``'s key in the quality-control table, in the spelling the table uses."""
    return cohorts.resolve_qc_tags(qc.known_cohort_tags(), [GAIT])[0]


def nsm_dir() -> Path:
    return knee_dir() / "geometries_nsm_similarity"


def model_dir() -> Path:
    return osim_path().parent


def geometry(name: str) -> Path:
    return model_dir() / "Geometry" / name


def osim_path() -> Path:
    """The built model ``GAIT`` simulated, as its run config names it."""
    return cohorts.condition_config(GAIT).for_knee(_knee()).path("model", "built_model")


def sim_dir() -> Path:
    return cohorts.results_dir(GAIT, _knee())


def synthetic_dir() -> Path:
    return stage_dir("simulations", "synthetic_shape_axis")


def combined_path() -> Path:
    return stage_dir("derived", "combined") / f"combined_data_{GAIT}.npz"


def rel(path) -> str:
    """A path as recorded in provenance: relative to the data root.

    Not resolved through symlinks: a data root borrows unchanged inputs from an
    earlier one through links (a root may link its 00_inputs into another's), and
    the record names the path the pipeline read, which is inside this root.
    """
    return str(Path(os.path.abspath(path)).relative_to(data_root()))


def need(*paths: Path) -> None:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Figure 1 panel input(s) missing:\n    "
            + "\n    ".join(str(p) for p in missing)
            + "\n\nThese are vendored model / simulation / derived files; download "
            "the matching data-root stage (see README.md)."
        )


# --------------------------------------------------------------------------- #
# Render helpers                                                              #
# --------------------------------------------------------------------------- #
def _pv():
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


def plotter(size):
    pv = _pv()
    pl = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    pl.set_background("white")
    return pl


def trim(png: Path, pad: int = 8):
    """Crop a transparent screenshot to its content plus ``pad`` pixels."""
    from PIL import Image

    im = Image.open(png).convert("RGBA")
    alpha = np.array(im)[:, :, 3]
    ys, xs = np.where(alpha > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y0 == 0 or x0 == 0 or y1 == alpha.shape[0] - 1 or x1 == alpha.shape[1] - 1:
        print(f"  ** content touches the render border: {png.name}")
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(alpha.shape[0] - 1, y1 + pad), min(alpha.shape[1] - 1, x1 + pad)
    im.crop((x0, y0, x1 + 1, y1 + 1)).save(png)
    return x0, y0


def lights(pl, c, span, key=(1.0, 1.2, 0.8), intensity=0.55):
    """A key light and a weak fill, placed relative to the scene."""
    pv = _pv()
    pl.add_light(pv.Light(position=tuple(c + np.array(key) * span * 4), focal_point=tuple(c),
                          light_type="scene light", intensity=intensity))
    pl.add_light(pv.Light(position=tuple(c + np.array([-1.0, 0.4, -0.6]) * span * 4),
                          focal_point=tuple(c), light_type="scene light", intensity=0.22))


def lights_camera(pl, c, d, up, span, key=0.85):
    """A soft key above-left of the camera and a weak fill from below."""
    pv = _pv()
    d, up = np.asarray(d, float), np.asarray(up, float)
    left = np.cross(up, d)
    left /= np.linalg.norm(left)
    pl.add_light(pv.Light(position=tuple(c + (d + 0.9 * up + 0.5 * left) * span * 4),
                          focal_point=tuple(c), light_type="scene light", intensity=key))
    pl.add_light(pv.Light(position=tuple(c + (d - 0.6 * up - 0.4 * left) * span * 4),
                          focal_point=tuple(c), light_type="scene light", intensity=0.30))


def project(pl, points):
    """World points -> display pixels (VTK convention: rows counted from the bottom)."""
    renderer = pl.renderer
    out = []
    for p in points:
        renderer.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        out.append((x, y))
    return out


def finish(pl, png: Path, window, anchors=None, post=None):
    """Screenshot, optional post-process, trim; anchors -> ``<png>_anchors.json``.

    Anchor pixels are in the trimmed image, origin top-left.
    """
    from PIL import Image

    display = project(pl, [a[1] for a in anchors]) if anchors else []
    pl.screenshot(str(png), transparent_background=True)
    pl.close()
    if post is not None:
        Image.fromarray(post(np.array(Image.open(png).convert("RGBA")))).save(png)
    x0, y0 = trim(png)
    if anchors:
        height = window[1]
        pixels = {name: [float(x - x0), float(height - 1 - y - y0)]
                  for (name, _), (x, y) in zip(anchors, display)}
        png.with_name(png.stem + "_anchors.json").write_text(json.dumps(pixels, indent=1))
    print(f"wrote {png.name} {Image.open(png).size}")


def rotation(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


# --------------------------------------------------------------------------- #
# The model probe (OpenSim)                                                   #
# --------------------------------------------------------------------------- #
def probe_model() -> dict:
    """Body poses, wraps, muscle paths and attachment points at the default pose.

    What the exploded knee and the body need from the model and cannot read off
    the meshes. Written to ``model_probe.json`` so the renders need no OpenSim.
    """
    try:
        import opensim as osim
    except ImportError as err:
        raise ImportError(
            f"OpenSim failed to import ({err}). In the comak env run this step as\n"
            "    conda run -n comak-knee-oa python pipeline/derive_figure1_panels.py probe"
        ) from None
    need(osim_path())
    osim.Logger.setLevelString("error")
    model = osim.Model(str(osim_path()))
    state = model.initSystem()
    forces = model.getForceSet()
    out = {"bodies": {}, "wraps": [], "muscles": {}, "attachments": []}
    for i in range(forces.getSize()):
        force = forces.get(i)
        lig = osim.Blankevoort1991Ligament.safeDownCast(force)
        mus = osim.Muscle.safeDownCast(force)
        if lig is None and mus is None:
            continue
        path = lig.getGeometryPath() if lig is not None else mus.getGeometryPath()
        if mus is not None:
            current = path.getCurrentPath(state)
            out["muscles"][force.getName()] = [
                [current.get(j).getLocationInGround(state).get(k) for k in range(3)]
                for j in range(current.getSize())
            ]
        points = path.getPathPointSet()
        for j in range(points.getSize()):
            point = points.get(j)
            location = point.getLocation(state)
            out["attachments"].append({
                "kind": "lig" if lig is not None else "mus", "name": force.getName(),
                "frame": point.getParentFrame().getName(),
                "xyz": [location.get(k) for k in range(3)],
            })
    bodies = model.getBodySet()
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        pose = body.getTransformInGround(state)
        R, p = pose.R(), pose.p()
        out["bodies"][body.getName()] = {
            "R": [[R.get(r, c) for c in range(3)] for r in range(3)],
            "p": [p.get(k) for k in range(3)],
        }
        wraps = body.getWrapObjectSet()
        for j in range(wraps.getSize()):
            wrap = wraps.get(j)
            entry = {"body": body.getName(), "name": wrap.getName(), "type": wrap.getConcreteClassName(),
                     "translation": [wrap.get_translation().get(k) for k in range(3)],
                     "rot": [wrap.get_xyz_body_rotation().get(k) for k in range(3)]}
            cylinder = osim.WrapCylinder.safeDownCast(wrap)
            ellipsoid = osim.WrapEllipsoid.safeDownCast(wrap)
            if cylinder is not None:
                entry.update(radius=cylinder.get_radius(), length=cylinder.get_length())
            if ellipsoid is not None:
                entry.update(dimensions=[ellipsoid.get_dimensions().get(k) for k in range(3)])
            out["wraps"].append(entry)
    out["source"] = rel(osim_path())
    (out_dir() / "model_probe.json").write_text(json.dumps(out))
    print(f"wrote model_probe.json: {len(out['bodies'])} bodies, {len(out['wraps'])} wraps, "
          f"{len(out['muscles'])} muscles, {len(out['attachments'])} path points")
    return {"model_probe": {"osim": rel(osim_path()), "pose": "default (initSystem)",
                            "muscles": len(out["muscles"]), "path_points": len(out["attachments"])}}


def load_probe() -> dict:
    path = out_dir() / "model_probe.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- build it first (needs OpenSim):\n"
            "    conda run -n comak-knee-oa python pipeline/derive_figure1_panels.py probe"
        )
    return json.loads(path.read_text())


def body_pose(probe, body) -> np.ndarray:
    entry = probe["bodies"][body]
    T = np.eye(4)
    T[:3, :3] = np.array(entry["R"])
    T[:3, 3] = np.array(entry["p"])
    return T


def wrap_mesh(probe, wrap, cap=0.07, res=64):
    """A wrap object as a mesh in ground; cylinders capped to ``cap`` m for display."""
    pv = _pv()
    rx, ry, rz = wrap["rot"]
    R = rotation("x", rx) @ rotation("y", ry) @ rotation("z", rz)
    if wrap["type"] == "WrapCylinder":
        mesh = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=wrap["radius"],
                           height=min(wrap["length"], cap), resolution=max(res, 24))
    else:
        mesh = pv.ParametricEllipsoid(*wrap["dimensions"], u_res=res, v_res=res, w_res=res)
    local = np.eye(4)
    local[:3, :3] = R
    local[:3, 3] = wrap["translation"]
    mesh = mesh.clean(tolerance=1e-6).compute_normals(auto_orient_normals=True)
    return mesh.transform(body_pose(probe, wrap["body"]) @ local, inplace=False)


def triad(pl, origin, size, color=TRIAD):
    pv = _pv()
    for direction in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        pl.add_mesh(pv.Arrow(start=origin, direction=direction, scale=size, tip_length=0.3,
                             tip_radius=0.11, shaft_radius=0.045),
                    color=color, smooth_shading=True, specular=0.0, ambient=0.5, diffuse=0.5)


# --------------------------------------------------------------------------- #
# Segmentation meshes and the shape-model reconstruction                      #
# --------------------------------------------------------------------------- #
NSM_PARTS = ("femur", "femur_cart", "tibia", "tibia_cart", "patella", "patella_cart", "med_men", "lat_men")


def segmentation_meshes() -> dict:
    pv = _pv()
    paths = {n: nsm_dir() / f"{KNEE_SIDE}_{n}_aligned.vtk" for n in NSM_PARTS}
    need(*paths.values())
    return {n: pv.read(str(p)) for n, p in paths.items()}


def reconstruction_meshes() -> dict:
    pv = _pv()
    folder = {"femur": "femur", "femur_cart": "femur", "tibia": "tibia", "tibia_cart": "tibia",
              "patella": "patella", "patella_cart": "patella", "med_men": "femur", "lat_men": "femur"}
    paths = {n: nsm_dir() / folder[n] / f"{KNEE_SIDE}_{n}_nsm_recon_mm.vtk" for n in NSM_PARTS}
    need(*paths.values())
    return {n: pv.read(str(p)) for n, p in paths.items()}


def knee_axes(seg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Superior, anterior and medial unit vectors of this knee in its NSM mm space.

    Superior runs from the tibia's centroid to the femur's; anterior is the
    patella's offset from the joint centre, made orthogonal to superior; medial
    completes the frame. From the segmentation meshes.
    """
    femur = np.array(seg["femur"].center)
    tibia = np.array(seg["tibia"].center)
    sup = femur - tibia
    sup /= np.linalg.norm(sup)
    ant = np.array(seg["patella"].center) - (femur + tibia) / 2
    ant -= (ant @ sup) * sup
    ant /= np.linalg.norm(ant)
    return sup, ant, np.cross(sup, ant)


def render_seg() -> dict:
    """The segmentation meshes, flat-shaded on purpose: rough, before the shape model."""
    pv = _pv()
    seg = segmentation_meshes()
    sup, ant, med = knee_axes(seg)
    colors = {"femur": BONE_SEG, "tibia": BONE_SEG, "patella": BONE_SEG, "femur_cart": CART,
              "tibia_cart": CART, "patella_cart": CART, "med_men": MENI, "lat_men": MENI}
    window = (1600, 1600)
    # Default (light kit) lighting plus one weak scene light.
    pl = pv.Plotter(off_screen=True, window_size=window)
    pl.set_background("white")
    for name, mesh in seg.items():
        pl.add_mesh(mesh, color=colors[name], smooth_shading=False, specular=0.02, ambient=0.28, diffuse=0.65)
    c = (np.array(seg["femur"].center) + np.array(seg["tibia"].center)) / 2 + sup * 8
    span = 110
    d = 1.0 * ant - 0.3 * sup - 0.45 * med
    d /= np.linalg.norm(d)
    pl.camera_position = [tuple(c + d * 3.2 * span), tuple(c), tuple(sup)]
    pl.camera.zoom(1.35)
    pl.add_light(pv.Light(position=tuple(c + (ant + sup * 0.8 - med * 0.5) * 300),
                          light_type="scene light", intensity=0.12))
    png = panels_dir() / "seg_v2.png"
    pl.screenshot(str(png), transparent_background=True)
    pl.close()
    print(f"wrote {png.name} (untrimmed)")
    return {"seg_v2": {"source": [rel(nsm_dir() / f"{KNEE_SIDE}_{n}_aligned.vtk") for n in NSM_PARTS],
                       "shading": "flat (rough on purpose)", "window": list(window), "zoom": 1.35,
                       "view": "anterior-inferior-medial, NSM mm space",
                       "axes_sup_ant_med": [v.tolist() for v in (sup, ant, med)]}}


def render_hub() -> dict:
    """The shape model's reconstruction of the same knee, smooth, same camera family."""
    sup, ant, med = knee_axes(segmentation_meshes())
    rec = reconstruction_meshes()
    colors = {"femur": BONE, "tibia": BONE, "patella": BONE, "femur_cart": CART, "tibia_cart": CART,
              "patella_cart": CART, "med_men": MENI, "lat_men": MENI}
    order = ("femur", "femur_cart", "tibia", "tibia_cart", "patella", "patella_cart", "med_men", "lat_men")
    c = (np.array(rec["femur"].center) + np.array(rec["tibia"].center)) / 2 + sup * 8
    window = (1500, 1900)
    pl = plotter(window)
    for name in order:
        pl.add_mesh(rec[name], color=colors[name], **MATTE)
    d = 1.0 * ant - 0.3 * sup - 0.45 * med
    d /= np.linalg.norm(d)
    pl.camera_position = [tuple(c + d * 3.2 * 110), tuple(c), tuple(sup)]
    pl.camera.zoom(1.25)
    lights_camera(pl, c, d, sup, 110)
    finish(pl, panels_dir() / "hub_knee.png", window)
    return {"hub_knee": {"source": [rel(nsm_dir() / f"{n.split('_')[0] if n not in ('med_men', 'lat_men') else 'femur'}"
                                        / f"{KNEE_SIDE}_{n}_nsm_recon_mm.vtk") for n in order],
                         "view": "anterior-inferior-medial, NSM mm space", "window": list(window)}}


# --------------------------------------------------------------------------- #
# The exploded multibody knee                                                 #
# --------------------------------------------------------------------------- #
EXPLODED_PARTS = {
    "femur": ("femur_distal_r", "femur_nsm_recon_osim.stl", BONE),
    "tibia": ("tibia_proximal_r", "tibia_nsm_recon_osim.stl", BONE),
    "patella": ("patella_r", "patella_nsm_recon_osim.stl", BONE),
    "femur_cart": ("femur_distal_r", "femur_articular_surface_osim.stl", CART),
    "tibia_cart": ("tibia_proximal_r", "tibia_articular_surface_osim.stl", CART),
    "patella_cart": ("patella_r", "patella_articular_surface_osim.stl", CART),
    "med_men": ("meniscus_medial_r", "med_men_osim.stl", MENI),
    "lat_men": ("meniscus_lateral_r", "lat_men_osim.stl", MENI),
    "med_up": ("meniscus_medial_r", "med_men_upper_art_surf_osim.stl", CART_MEN),
    "lat_up": ("meniscus_lateral_r", "lat_men_upper_art_surf_osim.stl", CART_MEN),
    "med_lo": ("meniscus_medial_r", "med_men_lower_art_surf_osim.stl", CART_MEN),
    "lat_lo": ("meniscus_lateral_r", "lat_men_lower_art_surf_osim.stl", CART_MEN),
}


def render_exploded() -> dict:
    """Bones pulled apart along the joint so every part, site and triad is visible.

    No prefemoral fat pad (the figure's callouts do not name it). Ground frame of
    the default pose: +x anterior, +y superior, +z lateral.
    """
    from PIL import Image

    pv = _pv()
    probe = load_probe()
    need(*(geometry(fn) for _, fn, _ in EXPLODED_PARTS.values()))
    sup, ant, lat = np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    offset = {"femur_distal_r": 0.052 * sup, "femur_r": 0.052 * sup,
              "patella_r": PATELLA_LIFT * sup + 0.026 * ant - 0.014 * lat,
              "tibia_proximal_r": -0.016 * sup, "tibia_r": -0.016 * sup,
              "meniscus_medial_r": 0.020 * sup - 0.017 * lat, "meniscus_lateral_r": 0.020 * sup + 0.017 * lat}
    window = (2000, 2000)
    pl = plotter(window)
    # (no depth peeling: off screen it drops the translucent wrap fills)
    meshes = {}
    for key, (body, filename, color) in EXPLODED_PARTS.items():
        # the meniscus contact sheets sit 0.6 mm proud of the meniscus so they do not z-fight
        extra = 0.0006 * sup if key.endswith("_up") else (-0.0006 * sup if key.endswith("_lo") else 0)
        mesh = pv.read(str(geometry(filename))).transform(body_pose(probe, body), inplace=False)
        mesh = mesh.translate(offset[body] + extra, inplace=False)
        meshes[key] = mesh
        pl.add_mesh(mesh, color=color, **MAT)

    wraps = {}
    for wrap in probe["wraps"]:
        if wrap["body"] in ("femur_distal_r", "patella_r", "tibia_proximal_r") or wrap["name"] == "KnExt_at_fem_r":
            mesh = wrap_mesh(probe, wrap, cap=0.05, res=48).translate(offset[wrap["body"]], inplace=False)
            opacity = 0.50 if wrap["type"] == "WrapCylinder" else 0.62
            pl.add_mesh(mesh, color=WRAP, opacity=opacity, silhouette=dict(color=WRAP_EDGE, line_width=2.6),
                        smooth_shading=True, specular=0.0, ambient=0.45, diffuse=0.55)
            wraps[wrap["name"]] = mesh

    # Attachment sites: path points on the knee bodies within 4 mm of their bone
    # (drops muscle origins along the shaft and fibular insertions); all meniscal ones.
    bones = {"femur_distal_r": meshes["femur"], "tibia_proximal_r": meshes["tibia"], "patella_r": meshes["patella"],
             "femur_r": meshes["femur"], "tibia_r": meshes["tibia"],
             "meniscus_medial_r": meshes["med_men"], "meniscus_lateral_r": meshes["lat_men"]}
    sites = {"lig": [], "mus": []}
    patella_sites = []
    for point in probe["attachments"]:
        if point["frame"] not in bones:
            continue
        T = body_pose(probe, point["frame"])
        x = T[:3, :3] @ np.array(point["xyz"]) + T[:3, 3] + offset[point["frame"]]
        bone = bones[point["frame"]]
        if (np.linalg.norm(bone.points[bone.find_closest_point(x)] - x) < ATTACH_MAX_DIST
                or point["frame"].startswith("meniscus")):
            sites[point["kind"]].append(x)
            if point["frame"] == "patella_r":
                patella_sites.append(x)
    patella_sites = np.array(patella_sites)
    sites = {k: np.array(v) for k, v in sites.items()}
    for kind, color in (("lig", LIG), ("mus", TEN)):
        glyphs = pv.PolyData(sites[kind]).glyph(
            geom=pv.Sphere(radius=0.0016, theta_resolution=14, phi_resolution=14), scale=False, orient=False)
        pl.add_mesh(glyphs, color=color, smooth_shading=True, specular=0.0, ambient=0.45, diffuse=0.55)

    # One coordinate triad per 6-DOF joint: tibiofemoral, patellofemoral, both menisci.
    femur_origin = body_pose(probe, "femur_distal_r")[:3, 3]
    tibia_cart = meshes["tibia_cart"]
    triads = {
        "TF": np.array([tibia_cart.center[0], tibia_cart.bounds[3] + 0.012, tibia_cart.center[2]]),
        "PF": body_pose(probe, "patella_r")[:3, 3] + offset["patella_r"] + 0.026 * sup,
        "MM": np.array(meshes["med_men"].center) - 0.020 * lat + 0.006 * sup,
        "LM": np.array(meshes["lat_men"].center) + 0.012 * lat,
    }
    for key, origin in triads.items():
        triad(pl, origin, TRIAD_SIZE, color=TRI[key])

    c = femur_origin + 0.036 * sup
    span = 0.10
    d = 1.0 * ant + 0.30 * sup - 0.62 * lat
    d /= np.linalg.norm(d)
    pl.camera_position = [tuple(c + d * span * 6), tuple(c), tuple(sup)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.176
    lights(pl, c, span, key=(1.0, 1.4, -0.3), intensity=0.85)
    right2d = np.cross(-d, sup)                       # screen-right; medial (-z) reads right
    right2d /= np.linalg.norm(right2d)

    # Ligament-site anchor: the most superior of the screen-right femoral ligament sites.
    near_femur = np.array([abs(p[1] - (femur_origin + offset["femur_distal_r"])[1]) < 0.06 for p in sites["lig"]])
    femoral = sites["lig"][near_femur]
    screen_x = femoral @ right2d
    candidates = femoral[screen_x > np.percentile(screen_x, 70)]
    lig_femur = candidates[np.argmax(candidates[:, 1])]

    # Cartilage anchor: femoral cartilage that is visible in the render (femur,
    # menisci and patella occlude much of the sheet), found from a key-colour pass
    # on the trochlear half, lateral side.
    key_pl = pv.Plotter(off_screen=True, window_size=window, lighting="none")
    key_pl.set_background("black")
    for key, mesh in meshes.items():
        key_pl.add_mesh(mesh, color=("#00FF00" if key == "femur_cart" else "#202020"), lighting=False)
    key_pl.camera_position = pl.camera_position
    key_pl.camera.parallel_projection = True
    key_pl.camera.parallel_scale = pl.camera.parallel_scale
    with tempfile.TemporaryDirectory() as tmp:
        key_png = Path(tmp) / "cartkey.png"
        key_pl.screenshot(str(key_png))
        key_pl.close()
        key_img = np.array(Image.open(key_png).convert("RGB"))
    visible = (key_img[:, :, 1] > 180) & (key_img[:, :, 0] < 90) & (key_img[:, :, 2] < 90)
    rows, cols = np.where(visible)
    upper = rows < np.percentile(rows, 60)
    if upper.sum() < 200:
        upper = np.ones(len(rows), bool)
    cu, ru = cols[upper], rows[upper]
    target_col = np.percentile(cu, 20)
    band = np.abs(cu - target_col) < 30
    target = np.array([target_col, np.median(ru[band])])
    cart_points = meshes["femur_cart"].points
    dist = np.linalg.norm(np.array(project(pl, cart_points)) - np.array([target[0], window[1] - 1 - target[1]]), axis=1)
    femur_cart_anchor = cart_points[int(np.argmin(dist))]

    tc = meshes["tibia_cart"].points
    tibia_cart_anchor = tc[np.argmin(tc[:, 2] - 0.4 * tc[:, 1])]
    lig_wrap = wraps["Med_Lig_r"].points
    wrap_anchor = lig_wrap[np.argmin(lig_wrap[:, 2] - 0.2 * lig_wrap[:, 1])]
    mu = meshes["med_up"].points
    half = mu[mu[:, 2] < np.median(mu[:, 2])]
    men_top = half[np.argmax(half[:, 1])]
    tp = meshes["tibia"].points
    femur = meshes["femur"]
    anchors = [
        ("femur_cart", femur_cart_anchor), ("tibia_cart", tibia_cart_anchor), ("tibia_wrap", wrap_anchor),
        ("lig_site", lig_femur), ("patella_site", patella_sites[np.argmax(patella_sites @ right2d)]),
        ("med_up", men_top),
        ("triad_TF", triads["TF"]), ("triad_PF", triads["PF"]), ("triad_LM", triads["LM"]), ("triad_MM", triads["MM"]),
        ("femur_top", np.array([femur.center[0], femur.bounds[3], femur.center[2]])),
        ("femur_right", np.array([femur.center[0], femur.center[1] + 0.02, femur.bounds[4]])),
        ("tibia_bottom", np.array([meshes["tibia"].center[0], meshes["tibia"].bounds[2], meshes["tibia"].center[2]])),
        ("tibia_bottom_med", tp[np.argmin(tp[:, 1] + 0.6 * tp[:, 2])]),
        ("tibia_bottom_lat", tp[np.argmin(tp[:, 1] - 0.6 * tp[:, 2])]),
    ]
    camera, scale = pl.camera_position, pl.camera.parallel_scale

    def fade_femur_shaft(img):
        """The femur's cut shaft fades in over 14 mm from its top edge."""
        mask_pl = pv.Plotter(off_screen=True, window_size=window, lighting="none")
        mask_pl.set_background("white")
        mask_pl.add_mesh(femur, color="black", lighting=False)
        mask_pl.camera_position = camera
        mask_pl.camera.parallel_projection = True
        mask_pl.camera.parallel_scale = scale
        with tempfile.TemporaryDirectory() as tmp:
            mask_png = Path(tmp) / "femur_mask.png"
            mask_pl.screenshot(str(mask_png), transparent_background=True)
            mask_pl.close()
            mask = np.array(Image.open(mask_png).convert("RGBA"))[:, :, 3] > 0
        y_top = np.where(mask.any(1))[0].min()
        band_px = 0.014 * window[0] / (2 * scale)
        ramp = np.clip((np.arange(img.shape[0]) - y_top) / band_px, 0, 1) ** 1.3
        alpha = img[:, :, 3].astype(float)
        alpha[mask] = alpha[mask] * ramp[:, None].repeat(img.shape[1], 1)[mask]
        img[:, :, 3] = alpha.astype(np.uint8)
        return img

    finish(pl, panels_dir() / "exploded_nofat_bigtriads.png", window, anchors, post=fade_femur_shaft)
    return {"exploded_nofat_bigtriads": {
        "source": [rel(geometry(fn)) for _, fn, _ in EXPLODED_PARTS.values()],
        "poses_and_wraps": rel(out_dir() / "model_probe.json"),
        "wraps": list(wraps), "wrap_style": f"solid {WRAP}, opacity 0.50 (cylinders) / 0.62 (ellipsoids), silhouette {WRAP_EDGE}",
        "attachment_sites": {k: int(len(v)) for k, v in sites.items()},
        "attachment_max_distance_m": ATTACH_MAX_DIST,
        "fat_pad": "not drawn", "patella_offset_sup_m": PATELLA_LIFT, "triad_size_m": TRIAD_SIZE,
        "camera": {"direction": d.tolist(), "up": sup.tolist(), "parallel_scale": 0.176},
        "cartilage_anchor_px": [float(target[0]), float(target[1])],
    }}


# --------------------------------------------------------------------------- #
# The full-body model                                                         #
# --------------------------------------------------------------------------- #
LOWER_BODY = ("pelvis", "femur_r", "femur_distal_r", "patella_r", "tibia_proximal_r", "meniscus_lateral_r",
              "meniscus_medial_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "femur_l", "patella_l", "tibia_l",
              "talus_l", "calcn_l", "toes_l")


def render_body() -> dict:
    """Pelvis and both legs at the default pose; the right leg's muscles as tubes.

    The right femoral and tibial shafts are the model's generic left-side meshes
    mirrored across, clipped 12 mm into the subject's own distal femur and
    proximal tibia, so the knee reads as the subject's and the shafts as generic.
    """
    pv = _pv()
    probe = load_probe()
    need(osim_path())
    root = ET.parse(osim_path()).getroot()
    window = (1000, 2000)
    pl = plotter(window)
    placed, n_mesh = [], 0
    for body in root.iter("Body"):
        name = body.get("name")
        if name not in probe["bodies"] or name not in LOWER_BODY:
            continue
        T = body_pose(probe, name)
        offsets = {}
        for frame in body.iter("PhysicalOffsetFrame"):
            translation, orientation = frame.find("translation"), frame.find("orientation")
            L = np.eye(4)
            if orientation is not None:
                a = [float(v) for v in orientation.text.split()]
                L[:3, :3] = rotation("x", a[0]) @ rotation("y", a[1]) @ rotation("z", a[2])
            if translation is not None:
                L[:3, 3] = [float(v) for v in translation.text.split()]
            offsets[frame.get("name")] = L
        for mesh_el in body.iter("Mesh"):
            filename = mesh_el.find("mesh_file").text.strip()
            path = geometry(filename)
            if not path.exists():
                continue
            scale_el = mesh_el.find("scale_factors")
            S = np.eye(4)
            if scale_el is not None:
                S[:3, :3] = np.diag([float(v) for v in scale_el.text.split()])
            socket = mesh_el.find("socket_frame")
            L = offsets.get(socket.text.strip().split("/")[-1], np.eye(4)) if socket is not None else np.eye(4)
            mesh = pv.read(str(path)).transform(T @ L @ S, inplace=False)
            if filename.endswith(".vtp"):
                mesh = mesh.triangulate().compute_normals(auto_orient_normals=True, consistent_normals=True)
            if "nsm_recon" in filename:
                color = BONE
            elif "articular_surface" in filename:
                color = CART
            elif "men_osim" in filename:
                color = MENI
            elif "fat_pad" in filename or "art_surf" in filename:
                continue
            else:
                color = GREY
            pl.add_mesh(mesh, color=color, **MATTE)
            placed.append(mesh)
            n_mesh += 1
    shafts = (("femur_l.vtp", "femur_r", "femur_l", "femur_distal_r", "femur_nsm_recon_osim.stl", "top"),
              ("tibia_l.vtp", "tibia_r", "tibia_l", "tibia_proximal_r", "tibia_nsm_recon_osim.stl", "bottom"))
    for source, body, scale_body, nsm_body, nsm_file, side in shafts:
        mesh = pv.read(str(geometry(source))).triangulate()
        scale_el = None
        for body_el in root.iter("Body"):
            if body_el.get("name") == scale_body:
                for mesh_el in body_el.iter("Mesh"):
                    if mesh_el.find("mesh_file").text.strip() == source:
                        scale_el = mesh_el.find("scale_factors")
        S = np.eye(4)
        if scale_el is not None:
            S[:3, :3] = np.diag([float(v) for v in scale_el.text.split()])
        mesh = mesh.transform(S @ np.diag([1.0, 1.0, -1.0, 1.0]), inplace=False)
        nsm = pv.read(str(geometry(nsm_file))).transform(
            np.linalg.inv(body_pose(probe, body)) @ body_pose(probe, nsm_body), inplace=False)
        if side == "top":
            mesh = mesh.clip(normal=(0, -1, 0), origin=(0, nsm.bounds[3] - 0.012, 0))
        else:
            mesh = mesh.clip(normal=(0, 1, 0), origin=(0, nsm.bounds[2] + 0.012, 0))
        mesh = mesh.transform(body_pose(probe, body), inplace=False).compute_normals(
            auto_orient_normals=True, consistent_normals=True)
        pl.add_mesh(mesh, color=GREY, **MATTE)
        placed.append(mesh)
        n_mesh += 1
    n_muscles = 0
    for name, points in probe["muscles"].items():
        points = np.array(points)
        if not name.endswith("_r") or len(points) < 2:
            continue
        pl.add_mesh(pv.lines_from_points(points).tube(radius=0.0055, n_sides=12), color=MUSCLE, **MATTE)
        n_muscles += 1
    bounds = np.array([m.bounds for m in placed])
    lo, hi = bounds[:, ::2].min(0), bounds[:, 1::2].max(0)
    c = (lo + hi) / 2
    d = np.array([1.0, 0.12, 0.85])
    d /= np.linalg.norm(d)
    up = np.array([0, 1.0, 0])
    pl.camera_position = [tuple(c + d * 4.0), tuple(c), tuple(up)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = (hi[1] - lo[1]) * 0.54
    lights_camera(pl, c, d, up, 0.5)
    femur_cart = pv.read(str(geometry("femur_articular_surface_osim.stl"))).transform(
        body_pose(probe, "femur_distal_r"), inplace=False)
    anchors = [("knee", np.array(femur_cart.center)), ("hip", body_pose(probe, "femur_r")[:3, 3]),
               ("ankle", body_pose(probe, "talus_r")[:3, 3])]
    finish(pl, panels_dir() / "body.png", window, anchors)
    return {"body": {"osim": rel(osim_path()), "geometry": rel(model_dir() / "Geometry"),
                     "poses_and_muscles": rel(out_dir() / "model_probe.json"), "meshes": n_mesh,
                     "muscles": f"{n_muscles} right-leg muscle paths",
                     "right_shafts": "femur_l.vtp / tibia_l.vtp mirrored in z onto femur_r / tibia_r (generic, grey, capped)"}}


# --------------------------------------------------------------------------- #
# Contact pressure on the knee's own femoral cartilage                        #
# --------------------------------------------------------------------------- #
def frame_transforms():
    """Body -> ground 4x4 per frame, from the joint-mechanics frame-transform table."""
    path = sim_dir() / "joint-mechanics" / "joint_mechanics_frame_transforms_in_ground.sto"
    need(path)
    lines = path.read_text().split("\n")
    header = [j for j, line in enumerate(lines) if line.strip() == "endheader"][0]
    columns = lines[header + 1].split("\t")
    table = np.array([[float(x) for x in line.split("\t")] for line in lines[header + 2:] if line.strip()])

    def transform(body, frame):
        return np.array([[table[frame, columns.index(f"{body}_T{r}{c}")] for c in (1, 2, 3, 4)]
                         for r in (1, 2, 3, 4)])
    return transform


def stance_frames():
    """Early-stance peak, midstance valley and late-stance peak of TF contact force."""
    import h5py

    path = sim_dir() / "joint-mechanics" / "joint_mechanics.h5"
    need(path)
    with h5py.File(path) as f:
        force = np.linalg.norm(
            f["model/forceset/Smith2018ArticularContactForce/tf_contact/tibia_cartilage/total_contact_force"][:], axis=1)
    n = len(force)
    pk1 = int(np.argmax(force[: n // 2]))
    pk2 = n // 2 + int(np.argmax(force[n // 2:]))
    mid = pk1 + int(np.argmin(force[pk1:pk2]))
    return (pk1, mid, pk2), force


def _femur_camera(pl, center):
    """Inferior view of the femur, anterior up, medial on the right."""
    span = 0.09
    pl.camera_position = [tuple(center + np.array((0, -1.0, 0)) * span * 6), tuple(center), (1, 0, 0)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.05
    lights(pl, center, span, key=(0.5, -1.0, 0.3), intensity=0.6)


def render_pressure() -> dict:
    """Three instants plus the cumulative map, one camera and one crop box for all four."""
    from PIL import Image

    pv = _pv()
    transform = frame_transforms()
    frames, force = stance_frames()
    bone = pv.read(str(geometry("femur_nsm_recon_osim.stl")))              # femur body frame
    cart_ref = pv.read(str(geometry("femur_articular_surface_osim.stl")))
    center = np.array(cart_ref.center)
    window = (1500, 1500)
    raw, instants = {}, {}
    for frame, tag in zip(frames, ("pk1", "mid", "pk2")):
        vtp = sim_dir() / "joint-mechanics" / f"joint_mechanics_contact_femur_cartilage_dynamic_ground_ground_{frame}.vtp"
        need(vtp)
        dynamic = pv.read(str(vtp))
        T = transform("femur_distal_r", frame)
        # The table is body -> ground: the articular STL pushed to ground must land on the dynamic mesh.
        error = np.linalg.norm(np.array(cart_ref.transform(T, inplace=False).center) - np.array(dynamic.center))
        assert error < 0.002, f"frame-transform convention broke: {error * 1000:.2f} mm"
        cart = dynamic.transform(np.linalg.inv(T), inplace=False)
        pressure = np.asarray(dynamic.cell_data["target_triangle_pressure_tf_contact"], float) / 1e6
        cart.clear_data()
        cart.cell_data["pressure_MPa"] = pressure
        cart = cart.cell_data_to_point_data()
        pl = plotter(window)
        pl.add_mesh(bone, color=BONE, **MAT)
        pl.add_mesh(cart, scalars="pressure_MPa", cmap="cividis", clim=PRESSURE_CLIM, show_scalar_bar=False,
                    smooth_shading=True, ambient=0.85, diffuse=0.25, specular=0.0)
        _femur_camera(pl, center)
        raw[tag] = _screenshot_array(pl)
        instants[tag] = {"frame": int(frame), "tf_force_N": float(force[frame]), "peak_MPa": float(pressure.max()),
                         "mean_contact_MPa": float(pressure[pressure > 0].mean())}

    stats_path = sim_dir() / "femur_baseline_pressure_stats.vtk"
    need(stats_path)
    stats = pv.read(str(stats_path))
    assert stats.n_points == cart_ref.n_points and stats.n_cells == cart_ref.n_cells
    cum = np.asarray(stats["cumulative_tf_pressure"], float) / 1e6          # Pa*s -> MPa*s
    cart = cart_ref.copy()
    cart["cum_MPas"] = cum
    pl = plotter(window)
    pl.add_mesh(bone, color=BONE, **MAT)
    pl.add_mesh(cart, scalars="cum_MPas", cmap="cividis", clim=CUM_CLIM, show_scalar_bar=False,
                smooth_shading=True, ambient=0.85, diffuse=0.25, specular=0.0)
    _femur_camera(pl, np.array(cart.center))
    cum_img = _screenshot_array(pl)

    # One crop box, from the three instants, so all four maps share a scale on the page.
    ys, xs = np.where(np.any([r[:, :, 3] > 0 for r in raw.values()], axis=0))
    box = (max(0, ys.min() - 8), ys.max() + 9, max(0, xs.min() - 8), xs.max() + 9)
    for tag, img in list(raw.items()) + [("cum", cum_img)]:
        Image.fromarray(img[box[0]:box[1], box[2]:box[3]]).save(panels_dir() / f"pressure_{tag}.png")
        print(f"wrote pressure_{tag}.png")
    joint_mechanics = sim_dir() / "joint-mechanics"
    return {"pressure": {
        "source": rel(joint_mechanics) + "/joint_mechanics_contact_femur_cartilage_dynamic_ground_ground_{frame}.vtp",
        "scalar": "target_triangle_pressure_tf_contact / 1e6 (MPa), cell -> point", "clim": list(PRESSURE_CLIM),
        "frames": instants, "instants": "argmax / valley / argmax of tibiofemoral contact force",
        "transform": "dynamic mesh pulled back into the femur_distal_r body frame with "
                     + rel(joint_mechanics / "joint_mechanics_frame_transforms_in_ground.sto"),
        "view": "inferior, up = anterior, medial on the right", "common_crop_rows_cols": [int(v) for v in box]},
        "cumulative": {"source": rel(stats_path), "scalar": "cumulative_tf_pressure / 1e6 (MPa*s)",
                       "clim": list(CUM_CLIM), "peak_MPas": float(cum.max()),
                       "p99_MPas": float(np.percentile(cum, 99)), "crop": "the instants' common crop box"}}


def _screenshot_array(pl) -> np.ndarray:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "shot.png"
        pl.screenshot(str(png), transparent_background=True)
        pl.close()
        return np.array(Image.open(png).convert("RGBA"))


# --------------------------------------------------------------------------- #
# Group-mean Delta cartilage thickness on the reference surface               #
# --------------------------------------------------------------------------- #
def render_surface(mesh, png: Path, scalars, cmap, clim, nan_color="#d0d0d0"):
    """Reference-surface map. Frame: +x medial, +y posterior, +z superior.

    Rendered from -z with up = +y (posterior up), then rotated 180 degrees, as the
    femur in ``derive_figure5_data.crop_and_orient``, so anterior is up and medial right.
    """
    from PIL import Image

    pv = _pv()
    pl = plotter((1100, 1000))
    camdir = np.array((0, 0, -1.0))
    shown = mesh.copy()
    shown["display"] = scalars
    pl.add_mesh(shown, scalars="display", cmap=cmap, clim=clim, nan_color=nan_color,
                show_scalar_bar=False, lighting=False)
    c = np.array(mesh.center)
    span = max(np.array(mesh.bounds[1::2]) - np.array(mesh.bounds[::2]))
    pl.camera_position = [tuple(c + camdir * span * 3), tuple(c), (0, 1, 0)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = span * 0.57
    pl.add_light(pv.Light(position=tuple(c + np.array([-1, 1, -3]) * span), focal_point=tuple(c), intensity=0.8))
    pl.screenshot(str(png), transparent_background=True)
    pl.close()
    Image.open(png).transpose(Image.ROTATE_180).save(png)
    trim(png)
    print(f"wrote {png.name} {Image.open(png).size}")


def draft_drop_set(knees) -> set[str]:
    """A narrower filter (union flags for ``GAIT`` plus 9916274_RIGHT), for provenance.

    Only used so ``provenance.json`` can report how it differs from the shared QC set.
    """
    import csv

    drop = {"9916274_RIGHT"}
    with qc.unified_csv().open() as handle:
        for row in csv.DictReader(handle):
            if str(row["flagged_union"]).lower() in ("true", "1") and GAIT in (row.get("cohort_tag") or ""):
                drop.add(row["knee"])
    return drop & set(knees)


def render_dthick() -> dict:
    """Mean OLS thickness-change rate per group, masked where < 10 knees contribute."""
    pv = _pv()
    need(combined_path())
    z = np.load(combined_path(), allow_pickle=True)
    knees = [str(x) for x in z["subjects"]]
    groups = np.array([str(g) for g in z["groups"]])
    rate = np.asarray(z["femur_thinning"], float) * 1000.0             # mm/yr -> um/yr
    has = np.asarray(z["femur_has_thinning"])
    outliers = {knee for _, knee in qc.load_outlier_set(cohort_tags=[cohort_tag()])}
    drop = (outliers | qc.load_chronic_knees()) & set(knees)
    keep = np.array([k not in drop for k in knees])
    surface = pv.read(str(reference_surface_path("femur")))
    assert surface.n_points == rate.shape[1], "reference surface and matrix disagree on vertex count"
    record = {}
    for cohort_key, tag in (("kl0_1_healthy", "healthy"), ("kl0_1_progressor", "incident")):
        members = (groups == cohort_key) & keep
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(np.where(has[members], rate[members], np.nan), 0)
        mean[has[members].sum(0) < MIN_KNEES] = np.nan
        render_surface(surface, panels_dir() / f"dthick_{tag}.png", mean, "RdBu_r", DTHICK_CLIM)
        record[tag] = {"n": int(members.sum()), "knees": sorted(np.array(knees)[members].tolist()),
                       "p5": float(np.nanpercentile(mean, 5)), "p95": float(np.nanpercentile(mean, 95))}
    draft = draft_drop_set(knees)
    return {"dthick": {
        "matrix": rel(combined_path()), "surface": rel(reference_surface_path("femur")),
        "quantity": "femur_thinning x 1000 = OLS Delta cartilage thickness rate (um/yr)",
        "clim": list(DTHICK_CLIM), "cmap": "RdBu_r", "min_knees_per_vertex": MIN_KNEES,
        "qc": f"{rel(qc.unified_csv())}: flagged_union for {cohort_tag()} + qc.load_chronic_knees()",
        "dropped": sorted(drop), "draft_filter_dropped": sorted(draft),
        "dropped_here_not_in_draft": sorted(drop - draft), "dropped_in_draft_not_here": sorted(draft - drop),
        **record}}


# --------------------------------------------------------------------------- #
# Synthetic knees along the shape axis                                        #
# --------------------------------------------------------------------------- #
def render_synthetic() -> dict:
    """The synthetic femur at the ends of the axis, one camera and scale for all."""
    pv = _pv()
    meshes, spans = {}, []
    for tag in SYNTHETIC_FRAMED:
        folder = synthetic_dir() / f"fig5_sigma_{tag}" / "geometries_nsm_similarity" / "femur"
        paths = (folder / "femur_nsm_recon_osim.vtk", folder / "femur_cartilage_nsm_recon_osim.vtk")
        need(*paths)
        meshes[tag] = tuple(pv.read(str(p)) for p in paths)
        bounds = np.array(meshes[tag][0].bounds)
        spans.append(max(bounds[1::2] - bounds[::2]))
    span = max(spans)
    up = np.array([0, 1.0, 0])
    d = np.array([1.0, -0.8, 0.35])
    d /= np.linalg.norm(d)
    c = np.array(meshes["p0"][1].center) + up * 0.08 * span               # p0's cartilage centre
    for tag in SYNTHETIC_SHOWN:
        bone, cart = meshes[tag]
        pl = plotter((1300, 1100))
        pl.add_mesh(bone, color=BONE, **MATTE)
        pl.add_mesh(cart, color=CART, **MATTE)
        pl.camera_position = [tuple(c + d * 3.0 * span), tuple(c), tuple(up)]
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = span * 0.62
        lights_camera(pl, c, d, up, span)
        png = panels_dir() / f"synthetic_{tag}.png"
        pl.screenshot(str(png), transparent_background=True)                 # untrimmed: a common frame
        pl.close()
        print(f"wrote {png.name} (untrimmed)")
    axis_path = synthetic_dir() / "shape_axis.json"
    need(axis_path)
    axis = json.loads(axis_path.read_text())
    return {"synthetic": {
        "source": rel(synthetic_dir()) + "/fig5_sigma_{tag}/geometries_nsm_similarity/femur/femur_{,cartilage_}nsm_recon_osim.vtk",
        "axis": axis.get("axis"), "axis_file": rel(axis_path),
        "k_sigma": {t: axis["shapes"][f"fig5_sigma_{t}"]["k_sigma"] for t in SYNTHETIC_SHOWN},
        "framed_over": list(SYNTHETIC_FRAMED),
        "view": "anterior-inferior-lateral; parallel projection, one camera + scale for all"}}


STEPS = {"probe": probe_model, "seg": render_seg, "hub": render_hub, "exploded": render_exploded,
         "body": render_body, "pressure": render_pressure, "dthick": render_dthick, "synthetic": render_synthetic}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # No ``choices=``: with ``nargs="*"`` Python 3.9's argparse rejects the empty
    # list, i.e. the no-argument run. Checked by hand instead.
    parser.add_argument("steps", nargs="*", metavar="step",
                        help=f"any of {', '.join(ALL_STEPS)} (default: all)")
    args = parser.parse_args()
    unknown = sorted(set(args.steps) - set(ALL_STEPS))
    if unknown:
        parser.error(f"unknown step(s) {unknown}; choose from {', '.join(ALL_STEPS)}")
    steps = args.steps or list(ALL_STEPS)

    provenance_path = out_dir() / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    for step in ALL_STEPS:
        if step in steps:
            print(f"== {step}")
            provenance.update(STEPS[step]())
    provenance.update({"knee": KNEE, "gait_for_pressure": f"{GAIT} ({cohort_tag()})", "model_run": model_run(),
                       "produced_by": "python pipeline/derive_figure1_panels.py", "date": date.today().isoformat()})
    provenance_path.write_text(json.dumps(provenance, indent=1))
    print(f"wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
