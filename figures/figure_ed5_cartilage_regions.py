#!/usr/bin/env python3
"""Extended Data Figure 5 -- anatomical regions, and the simulation-defined one.

    python figures/figure_ed5_cartilage_regions.py

Runs with no arguments, reads ``04_results/figure_ed5_cartilage_regions/``
(rendered by ``pipeline/derive_figure_ed5_data.py``) and writes manuscript
Extended Data Figure 5 as ``figures/output/[<run>/]figure_ed5_cartilage_regions.png``.

One row per bone, two panels.

A  The anatomical subregions drawn on the segmentations, each its own color,
   named in a legend beside the surface rather than on it.
B  The average simulation-defined region. Every knee's region is built
   individually from that knee's own simulations (Figure 8's derivation): a
   vertex is in it if it is retained in at least half of that knee's gaits. The
   panel shows the vertices whose share of gaits retained, averaged over all
   knees, is at least half (``mean_frequency >= 0.5``) -- a mean, hence the
   title. (Counting knees whose region contains the vertex instead, a median,
   gives a slightly larger region: 1,181 vs 1,125 femur vertices.) On the femur
   the anatomically defined medial weight-bearing region is outlined in black
   for comparison; the tibia's anatomical comparator is the whole medial
   plateau, so nothing is outlined there. The panel carries only its title;
   that definition is for the caption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import panels, results  # noqa: E402

FIGURE = "figure_ed5_cartilage_regions"
BUILD_CMD = "python pipeline/derive_figure_ed5_data.py"

BONES = ("femur", "tibia")
BONE_LABELS = {"femur": "Femur", "tibia": "Tibia"}
PANELS = ("anatomy_plain", "aggregate_all")

FS_HEADER = 15
FS_ROW = 16
FS_ORIENT = 15
FS_LEGEND = 11
FS_SUBTITLE = 13


def load_inputs():
    files = ["summary.json"] + [
        f"panels/{bone}/{panel}.png" for bone in BONES for panel in PANELS
    ]
    results.require(FIGURE, *files, produced_by=BUILD_CMD)
    base = results.figure_dir(FIGURE)
    summary = json.loads((base / "summary.json").read_text())
    images = {
        (bone, panel): mpimg.imread(base / "panels" / bone / f"{panel}.png")
        for bone in BONES for panel in PANELS
    }
    return summary, images


def legend_handles(entries, bone):
    """Legend patches in column-major order mirroring the anatomy, and the column count."""
    def patch(entry):
        return mpatches.Patch(facecolor=entry["color"], edgecolor="none",
                              label=entry["label"].replace(f" {bone}", ""))

    spacer = mpatches.Patch(facecolor="none", edgecolor="none", label=" ")
    lateral = [e for e in entries if e["label"].startswith("Lateral")]
    medial = [e for e in entries if e["label"].startswith("Medial")]
    other = [e for e in entries if e not in lateral and e not in medial]
    columns = [lateral, medial]
    if other:
        rows = max(len(lateral), len(medial))
        columns = [lateral, other, medial]
        other = other + [None] * (rows - len(other))
        columns[1] = other
    handles = [patch(e) if e is not None else spacer for column in columns for e in column]
    return handles, len(columns)


def render(summary, images) -> Path:
    fig = plt.figure(figsize=(13, 8.6))
    outer = fig.add_gridspec(
        2, 2, width_ratios=[1, 1], height_ratios=[1, 0.74],
        left=0.085, right=0.955, top=0.82, bottom=0.10, wspace=0.10, hspace=0.42,
    )
    axes = {
        (bone, panel): fig.add_subplot(outer[row, col])
        for row, bone in enumerate(BONES) for col, panel in enumerate(PANELS)
    }
    for key, ax in axes.items():
        ax.imshow(images[key])
        ax.axis("off")

    # The legend for A sits under each surface, one per bone, laid out the way
    # the surface is: lateral entries in the left column, medial in the right,
    # and the trochlea -- anterior -- on top between them. The bone's name is
    # dropped from each entry because the row label already says it.
    for bone in BONES:
        handles, ncol = legend_handles(summary["bones"][bone]["anatomy_legend_distinct"], bone)
        axes[(bone, "anatomy_plain")].legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=ncol,
            frameon=False, fontsize=FS_LEGEND, handlelength=1.3, handleheight=1.0,
            columnspacing=1.2, borderaxespad=0.0,
        )

    fig.canvas.draw()
    box = {key: ax.get_position() for key, ax in axes.items()}
    top = max(b.y1 for b in box.values())
    bottom = min(b.y0 for b in box.values())
    x_left = min(b.x0 for b in box.values())
    x_right = max(b.x1 for b in box.values())

    def centre(key):
        return 0.5 * (box[key].x0 + box[key].x1)

    header_y = top + 0.05
    columns = (
        ("A", ("femur", "anatomy_plain"), "Anatomically defined regions", None),
        # "Average": the panel thresholds each vertex's share of gaits retained,
        # averaged over all knees (module doc). The full definition and the
        # outline's meaning belong to the caption.
        ("B", ("femur", "aggregate_all"), "Average simulation-defined region", None),
    )
    for letter, key, title, subtitle in columns:
        fig.text(centre(key), header_y, title, ha="center", va="bottom",
                 fontsize=FS_HEADER, weight="bold")
        fig.text(box[key].x0 - 0.07, header_y, letter, ha="left", va="bottom",
                 fontsize=panels.DEFAULT_FONTSIZE, weight="bold")
        if subtitle:
            fig.text(centre(key), header_y - 0.012, subtitle, ha="center", va="top",
                     fontsize=FS_SUBTITLE, style="italic", color="#333333", linespacing=1.35)

    for bone in BONES:
        b = box[(bone, "anatomy_plain")]
        fig.text(x_left - 0.012, 0.5 * (b.y0 + b.y1), BONE_LABELS[bone], rotation=90,
                 ha="right", va="center", fontsize=FS_ROW, weight="bold")
    x_mid = 0.5 * (x_left + x_right)
    fig.text(x_mid, 0.985, "Anterior", ha="center", va="top", fontsize=FS_ORIENT, weight="bold")
    fig.text(x_mid, 0.012, "Posterior", ha="center", va="bottom", fontsize=FS_ORIENT, weight="bold")
    fig.text(0.012, 0.5 * (top + bottom), "Lateral", rotation=90, ha="left", va="center",
             fontsize=FS_ORIENT, weight="bold")
    fig.text(0.992, 0.5 * (top + bottom), "Medial", rotation=270, ha="right", va="center",
             fontsize=FS_ORIENT, weight="bold")

    out_path = results.output_dir() / "figure_ed5_cartilage_regions.png"
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> int:
    summary, images = load_inputs()
    out = render(summary, images)
    print(f"wrote {out}")
    for bone in BONES:
        a = summary["bones"][bone]["aggregate_all"]
        print(f"  {bone}: aggregate region {a['region_vertices']:,} vertices "
              f"(share of gaits retained, averaged over {a['n_knees']} knees, "
              f">= {100 * a['majority']:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
