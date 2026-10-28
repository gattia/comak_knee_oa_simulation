#!/usr/bin/env python3
"""Derive Extended Data Figure 5: Figure 8's two region definitions, rendered.

    python pipeline/derive_figure_ed5_data.py
    python pipeline/derive_figure_ed5_data.py --bone femur

Figure 8 compares a region chosen by each knee's own simulated contact pressure
against the standard anatomical region. This step draws both, so the reader can
see what is being compared. It reads the subregion labels (``03_derived``), the
reference surfaces (``00_inputs``) and the per-knee consensus regions that
Figure 8's derivation already wrote (``consensus_roi_{bone}.npz``), and writes
into ``04_results/figure_ed5_cartilage_regions/``:

    panels/{bone}/anatomy.png            the anatomical subregions, standard ROI outlined
    panels/{bone}/contact_frequency.png  each vertex's share of gaits retained, averaged over knees
    panels/{bone}/example_{group}.png    one knee per group, its own region filled
    panels/{bone}/group_{group}.png      the group's aggregate region: vertices inside the
                                         region of at least half of that group's knees
    panels/{bone}/anatomy_plain.png      the subregions again, each its own color, no outline
                                         and no names -- for a figure that uses a legend
    panels/{bone}/aggregate_all.png      the whole cohort's aggregate region: vertices whose
                                         share of gaits retained, averaged over all knees, is
                                         at least half (a mean; the group panels count knees)
    summary.json                         vertex counts, overlap statistics, which knees,
                                         and where each subregion's name goes on the image

``figures/figure_ed5_cartilage_regions.py`` composes ``anatomy_plain`` and
``aggregate_all`` (legend + one cohort aggregate) and reads nothing else, so
rendering the figure needs no 3D toolkit. The other panels are written as data.
This is the same split Figure 5 uses, and the camera, crop and flip are imported
from Figure 5's derivation so the two figures share one orientation convention.

Which knees are shown
---------------------
One per group, chosen as the knee whose region size is closest to its group's
mean on both bones at once (smallest summed |z|). Typical by construction, and
deterministic; the choice is recorded in ``summary.json``.

Where the names go
------------------
``anatomy.png`` names each subregion on the render rather than in a legend.
The position is the centroid of the subregion's camera-facing vertices, pushed
through the renderer's own world-to-display projection and then through the
same crop and flip the image gets, so a name lands on its region whatever the
crop box turns out to be. Pixel coordinates in the saved image, in ``summary.json``.

The tibia has no outline
------------------------
The femur's standard comparator is a sub-region of the medial condyle, so its
boundary can be traced. The tibia's is the whole medial plateau, which on the
reference surface is a closed mesh of its own: its boundary is the silhouette.
Nothing is drawn for it, rather than a line around the plateau that would
read as a sub-region.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa.paths import stage_dir  # noqa: E402
from derive_figure5_data import (  # noqa: E402
    CROP_PADDING_PX,
    VIEW,
    WINDOW_SUPPLEMENT,
    _plotter_camera,
    content_box,
    crop_and_orient,
)
from derive_figure8_data import (  # noqa: E402
    FIGURE,
    GROUPS,
    STATIC_FINE_LABEL,
    STATIC_REGION_NAME,
    groups_by_knee,
    region_labels_path,
)
from derive_thickness import reference_surface_path  # noqa: E402
from shared import palette  # noqa: E402

ED_FIGURE = "figure_ed5_cartilage_regions"
# A vertex is in a group's aggregate region if it is inside the region of at
# least this share of the group's knees. Half, like the per-knee consensus.
GROUP_MAJORITY = 0.5
BONES = ("femur", "tibia")

# Subregion names as drawn on the segmentations, in reading order per bone.
FINE_LABEL_NAMES = {
    8: "Medial tibia", 9: "Lateral tibia", 10: "Trochlea",
    11: "Medial weight-bearing femur", 12: "Lateral weight-bearing femur",
    13: "Medial posterior femur", 14: "Lateral posterior femur",
}
BONE_FINE_LABELS = {"femur": (11, 13, 10, 12, 14), "tibia": (8, 9)}
# What is printed on the render. Two lines where one would cross the region edge.
LABEL_TEXT = {
    8: "Medial", 9: "Lateral", 10: "Trochlea",
    11: "Medial\nweight-bearing", 12: "Lateral\nweight-bearing",
    13: "Medial\nposterior", 14: "Lateral\nposterior",
}

# Hue says medial or lateral (the manuscript's compartment colors); saturation
# says whether the subregion is the standard comparator.
ANATOMY_COLORS = {
    11: palette.MEDLAT_COLORS["medial"],   # standard femur ROI
    8: palette.MEDLAT_COLORS["medial"],    # standard tibia ROI
    13: "#A8DCC8",                          # medial posterior, a tint of medial
    12: "#E0CDAA",                          # lateral weight-bearing, a tint of lateral
    14: "#EFE3CF",                          # lateral posterior, a lighter tint
    9: "#E0CDAA",                           # lateral tibia
    10: "#B7C9DA",                          # trochlea, neutral blue-grey
}
# One color per subregion for the legend-based figure, from the manuscript's
# Material palette (palette.GAIT_MEASURED), avoiding the three group colors.
# Medial regions are cool, lateral regions warm; the tibia reuses the femur's
# medial and lateral colors since the row label already names the bone.
ANATOMY_COLORS_DISTINCT = {
    10: "#90A4AE",   # trochlea                      blue-grey, the palette's neutral family
    11: "#43A047",   # medial weight-bearing femur   green
    13: "#00897B",   # medial posterior femur        teal
    12: "#D81B60",   # lateral weight-bearing femur  magenta
    14: "#795548",   # lateral posterior femur       brown
    8: "#43A047",    # medial tibia                  green
    9: "#D81B60",    # lateral tibia                 magenta
}
AGGREGATE_COLOR = "#7E57C2"   # the cohort aggregate region: palette purple, no group meaning
SURFACE_GREY = "lightgray"
OUTLINE_WIDTH = 6

WINDOW = WINDOW_SUPPLEMENT


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def regions_dir(create: bool = False) -> Path:
    path = stage_dir("results", ED_FIGURE)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_consensus(bone: str):
    base = stage_dir("results", FIGURE)
    npz = base / f"consensus_roi_{bone}.npz"
    meta = base / f"meta_{bone}.json"
    for path in (npz, meta):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found -- build Figure 8's inputs first with:\n"
                "    python pipeline/derive_figure8_data.py"
            )
    data = np.load(npz)
    return (
        [str(k) for k in data["knees"]],
        np.asarray(data["frequency"], dtype=np.float32),
        np.asarray(data["mean_frequency"], dtype=np.float64),
        json.loads(meta.read_text()),
    )


def load_labels(bone: str) -> np.ndarray:
    path = region_labels_path(bone)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- build it with:\n"
            "    python pipeline/combine_reference_data.py labels"
        )
    return np.asarray(np.load(path, allow_pickle=True)["fine_labels"], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Choosing the knees to show                                                  #
# --------------------------------------------------------------------------- #
def choose_examples(sizes_by_bone: dict[str, dict[str, int]]) -> dict[str, dict]:
    """One knee per group: closest to the group mean region size on every bone."""
    group_of = groups_by_knee()
    common = set.intersection(*(set(s) for s in sizes_by_bone.values()))
    chosen = {}
    for group, cohort_key in GROUPS.items():
        members = sorted(k for k in common if group_of.get(k) == cohort_key)
        if not members:
            continue
        score = np.zeros(len(members))
        for sizes in sizes_by_bone.values():
            values = np.array([sizes[k] for k in members], dtype=float)
            score += np.abs((values - values.mean()) / values.std(ddof=1))
        knee = members[int(np.argmin(score))]
        chosen[group] = {
            "knee": knee,
            "region_vertices": {bone: int(sizes[knee]) for bone, sizes in sizes_by_bone.items()},
            "summed_abs_z": float(score.min()),
        }
    return chosen


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def hex_to_rgb8(color: str) -> np.ndarray:
    from matplotlib.colors import to_rgb

    return (np.array(to_rgb(color)) * 255).astype(np.uint8)


def region_boundary(mesh, mask: np.ndarray):
    """Boundary edges of the triangles lying entirely inside ``mask``."""
    import pyvista as pv

    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    inside = mask[faces].all(axis=1)
    if not inside.any():
        return None
    cells = np.hstack([np.full((inside.sum(), 1), 3), faces[inside]]).astype(np.int64)
    patch = pv.PolyData(mesh.points, cells).clean()
    edges = patch.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False,
    )
    return edges if edges.n_points > 0 else None


def render(mesh, out_png, bone, *, rgb=None, scalars=None, clim=None, cmap=None,
           outline=None, project=None):
    """One surface render with Figure 5's camera. ``rgb`` or ``scalars``, not both.

    ``project`` maps names to world points; the return value maps the same
    names to (column, row) pixels in the saved window, before any crop.
    """
    import pyvista as pv

    surface = mesh.copy()
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW)
    if rgb is not None:
        surface["rgb"] = rgb
        plotter.add_mesh(surface, scalars="rgb", rgb=True, show_scalar_bar=False)
    else:
        surface["scalar"] = scalars
        plotter.add_mesh(surface, scalars="scalar", cmap=cmap, clim=clim,
                         nan_color=SURFACE_GREY, show_scalar_bar=False)
    if outline is not None:
        edges = region_boundary(mesh, outline)
        if edges is not None:
            plotter.add_mesh(edges, color="black", line_width=OUTLINE_WIDTH)
    _plotter_camera(plotter, mesh, VIEW[bone])
    plotter.screenshot(str(out_png), transparent_background=False)
    pixels = {}
    for name, point in (project or {}).items():
        renderer = plotter.renderer
        renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        # VTK counts display rows from the bottom; images count from the top.
        pixels[name] = (float(x), float(WINDOW[1] - y))
    plotter.close()
    return pixels


def facing_centroids(mesh, fine, bone) -> dict[int, np.ndarray]:
    """Centroid of each subregion's camera-facing vertices, in world space."""
    normals = np.asarray(
        mesh.compute_normals(point_normals=True, cell_normals=False,
                             consistent_normals=True, auto_orient_normals=True)
        .point_data["Normals"]
    )
    facing = normals[:, 2] < 0 if VIEW[bone] == "inferior" else normals[:, 2] > 0
    out = {}
    for label in BONE_FINE_LABELS[bone]:
        selected = facing & (fine == label)
        if selected.any():
            out[label] = mesh.points[selected].mean(axis=0)
    return out


def shared_crop_box(sources, padding=CROP_PADDING_PX):
    """The box ``crop_and_orient`` will use for these panels, recomputed the same way."""
    boxes = [b for b in (content_box(p) for p in sources) if b is not None]
    left = min(b[0] for b in boxes)
    upper = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    lower = max(b[3] for b in boxes)
    return (max(0, left - padding), max(0, upper - padding),
            min(WINDOW[0], right + padding), min(WINDOW[1], lower + padding))


def into_saved_image(pixels, box, bone):
    """Window pixels -> pixels in the cropped, flipped image on disk."""
    left, upper, right, lower = box
    width, height = right - left, lower - upper
    out = {}
    for name, (x, y) in pixels.items():
        x, y = x - left, y - upper
        if bone == "femur":      # ROTATE_180
            x, y = width - x, height - y
        else:                    # FLIP_TOP_BOTTOM
            y = height - y
        out[name] = (round(x, 1), round(y, 1))
    return out


def render_bone(bone, examples, out_root):
    import pyvista as pv

    mesh = pv.read(str(reference_surface_path(bone)))
    fine = load_labels(bone)
    knees, frequency, mean_frequency, meta = load_consensus(bone)
    if fine.shape[0] != mesh.n_points or frequency.shape[1] != mesh.n_points:
        raise RuntimeError(f"{bone}: label, consensus and surface vertex counts disagree")

    standard = fine == STATIC_FINE_LABEL[bone]
    grey = hex_to_rgb8(SURFACE_GREY)

    # The femur's comparator is a sub-region with a traceable boundary; the
    # tibia's is a closed plateau whose boundary is the silhouette (module doc).
    outline = standard if region_boundary(mesh, standard) is not None else None

    names = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        rgb = np.tile(grey, (mesh.n_points, 1))
        for label in BONE_FINE_LABELS[bone]:
            rgb[fine == label] = hex_to_rgb8(ANATOMY_COLORS[label])
        centroids = facing_centroids(mesh, fine, bone)
        window_pixels = render(mesh, tmp / "anatomy.png", bone, rgb=rgb, outline=outline,
                               project={str(k): v for k, v in centroids.items()})
        names.append("anatomy")

        shown = mean_frequency.copy()
        shown[shown <= 0] = np.nan
        render(mesh, tmp / "contact_frequency.png", bone, scalars=shown, clim=(0.0, 1.0),
               cmap=palette.SEQUENTIAL_CMAP, outline=outline)
        names.append("contact_frequency")

        rgb = np.tile(grey, (mesh.n_points, 1))
        for label in BONE_FINE_LABELS[bone]:
            rgb[fine == label] = hex_to_rgb8(ANATOMY_COLORS_DISTINCT[label])
        render(mesh, tmp / "anatomy_plain.png", bone, rgb=rgb)
        names.append("anatomy_plain")

        aggregate = mean_frequency >= GROUP_MAJORITY
        rgb = np.tile(grey, (mesh.n_points, 1))
        rgb[aggregate] = hex_to_rgb8(AGGREGATE_COLOR)
        render(mesh, tmp / "aggregate_all.png", bone, rgb=rgb, outline=outline)
        names.append("aggregate_all")

        group_of = groups_by_knee()
        knee_group = np.array([group_of.get(k, "") for k in knees])
        group_stats = {}
        for group, cohort_key in GROUPS.items():
            members = knee_group == cohort_key
            share = (frequency[members] >= meta["consensus_fraction"]).mean(axis=0)
            region = share >= GROUP_MAJORITY
            rgb = np.tile(grey, (mesh.n_points, 1))
            rgb[region] = hex_to_rgb8(palette.GROUP_COLORS_SHORT_KEY[group])
            render(mesh, tmp / f"group_{group}.png", bone, rgb=rgb, outline=outline)
            names.append(f"group_{group}")
            group_stats[group] = {
                "n_knees": int(members.sum()),
                "region_vertices": int(region.sum()),
                "fraction_inside_standard": float((region & standard).sum() / max(region.sum(), 1)),
                "fraction_of_standard_covered": float((region & standard).sum() / standard.sum()),
                "majority": GROUP_MAJORITY,
            }

        index_of = {k: i for i, k in enumerate(knees)}
        for group, chosen in examples.items():
            region = frequency[index_of[chosen["knee"]]] >= meta["consensus_fraction"]
            rgb = np.tile(grey, (mesh.n_points, 1))
            rgb[region] = hex_to_rgb8(palette.GROUP_COLORS_SHORT_KEY[group])
            render(mesh, tmp / f"example_{group}.png", bone, rgb=rgb, outline=outline)
            names.append(f"example_{group}")

        out_dir = out_root / "panels" / bone
        out_dir.mkdir(parents=True, exist_ok=True)
        sources = [tmp / f"{n}.png" for n in names]
        label_pixels = into_saved_image(window_pixels, shared_crop_box(sources), bone)
        crop_and_orient(sources, [out_dir / f"{n}.png" for n in names], bone)

    summary = summarize(bone, fine, standard, knees, frequency, mean_frequency, meta,
                        outline is not None)
    summary["group_regions"] = group_stats
    summary["aggregate_all"] = {
        "n_knees": len(knees), "majority": GROUP_MAJORITY,
        "region_vertices": int((mean_frequency >= GROUP_MAJORITY).sum()),
    }
    summary["anatomy_legend_distinct"] = [
        {"label": FINE_LABEL_NAMES[label], "color": ANATOMY_COLORS_DISTINCT[label]}
        for label in BONE_FINE_LABELS[bone]
    ]
    summary["label_positions"] = [
        {"label": FINE_LABEL_NAMES[label], "text": LABEL_TEXT[label],
         "standard": label == STATIC_FINE_LABEL[bone],
         "xy": label_pixels[str(label)]}
        for label in BONE_FINE_LABELS[bone] if str(label) in label_pixels
    ]
    return summary


# --------------------------------------------------------------------------- #
# What the caption needs                                                      #
# --------------------------------------------------------------------------- #
def summarize(bone, fine, standard, knees, frequency, mean_frequency, meta, outlined):
    consensus = frequency >= meta["consensus_fraction"]
    sizes = consensus.sum(axis=1)
    overlap = (consensus & standard).sum(axis=1)
    n_standard = int(standard.sum())
    dice = 2.0 * overlap / (sizes + n_standard)
    population = mean_frequency >= 0.5
    composition = {
        FINE_LABEL_NAMES[label]: float(np.mean(
            (consensus & (fine == label)).sum(axis=1) / sizes))
        for label in BONE_FINE_LABELS[bone]
    }
    return {
        "standard_region": STATIC_REGION_NAME[bone],
        "standard_fine_label": STATIC_FINE_LABEL[bone],
        "standard_outlined": outlined,
        "standard_vertices": n_standard,
        "anatomical_medial_vertices": int(meta["anatomical_medial_vertices"]),
        "n_knees": len(knees),
        "quantile": meta["quantile"],
        "consensus_fraction": meta["consensus_fraction"],
        "region_vertices_mean": float(sizes.mean()),
        "region_vertices_sd": float(sizes.std(ddof=1)),
        "population_region_vertices": int(population.sum()),
        "vertices_in_any_region": int((mean_frequency > 0).sum()),
        "dice_vs_standard_mean": float(dice.mean()),
        "dice_vs_standard_sd": float(dice.std(ddof=1)),
        "fraction_of_region_inside_standard_mean": float((overlap / sizes).mean()),
        "fraction_of_standard_covered_mean": float((overlap / n_standard).mean()),
        "region_composition_by_subregion": composition,
        # The legend the figure draws, so the colors have one definition.
        "anatomy_legend": [
            {"label": FINE_LABEL_NAMES[label], "color": ANATOMY_COLORS[label],
             "standard": label == STATIC_FINE_LABEL[bone]}
            for label in BONE_FINE_LABELS[bone]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bone", choices=BONES, help="one bone only (default: both)")
    args = parser.parse_args()
    bones = (args.bone,) if args.bone else BONES

    sizes_by_bone = {}
    for bone in BONES:
        knees, frequency, _, meta = load_consensus(bone)
        sizes_by_bone[bone] = dict(zip(knees, (frequency >= meta["consensus_fraction"]).sum(axis=1)))
    examples = choose_examples(sizes_by_bone)
    for group, chosen in examples.items():
        print(f"[examples] {group}: {chosen['knee']} "
              f"({', '.join(f'{b} {n}' for b, n in chosen['region_vertices'].items())} vertices)")

    out_root = regions_dir(create=True)
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.setdefault("bones", {})
    for bone in bones:
        summary["bones"][bone] = render_bone(bone, examples, out_root)
        s = summary["bones"][bone]
        print(f"[{bone}] standard {s['standard_vertices']} vertices; per-knee region "
              f"{s['region_vertices_mean']:.0f} +/- {s['region_vertices_sd']:.0f}; "
              f"Dice vs standard {s['dice_vs_standard_mean']:.2f}; "
              f"{100 * s['fraction_of_region_inside_standard_mean']:.0f}% of the region "
              f"lies inside the standard one")
    summary["examples"] = examples
    summary["produced_by"] = "python pipeline/derive_figure_ed5_data.py"
    summary["date"] = date.today().isoformat()
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
