#!/usr/bin/env python3
"""Extended Data Figure 4 -- how the prefemoral fat-pad contact surface is built.

    python figures/figure_ed4_fatpad.py

Runs with no arguments, reads ``04_results/figure_ed4_fatpad/`` (rendered by
``pipeline/derive_figure_ed4_fatpad.py``) and writes manuscript Extended Data
Figure 4 as ``figures/output/[<run>/]figure_ed4_fatpad.{png,pdf}``.

Six panels; A-C are vertical pairs, D-F one row, each in a thin grey frame:

A  reference-model cartilage (extended superiorly)  /  the knee's shape-model cartilage
B  step 1: the candidate region (exposed anterior bone), without / with the patella
C  step 2: the offsets along the surface normals  /  the dilated surface coloured by offset
D  step 3: crop to within 25 mm of the patella
E  step 4: smooth and resample (5,000 vertices)
F  sagittal cross-section through the patella centre
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import results  # noqa: E402

FIGURE = "figure_ed4_fatpad"
BUILD_CMD = "python pipeline/derive_figure_ed4_fatpad.py"
PANELS = ("ref", "subj", "seed", "seed_pat", "vec", "dil", "crop", "final")

COL = dict(
    bone="#EAE3D3", cart="#8FB8DE", seed="#D64545", seed_pale="#F2C4C4",
    fat="#F2B134", fat_edge="#7A4E00", discard="#A9A9A9",
    arrow="#2B2B2B", arrow_cap="#D64545",
    section_bone="#6E6558", section_cart="#2E6FA8",
)
LEG = dict(fontsize=12, frameon=True, framealpha=0.9, edgecolor="none", handlelength=1.6)

# Vertical geometry in figure fraction: three equal-height rows, PAIR_GAP between rows 1-2
# (inside the A/B/C pairs) and BLOCK_GAP between row 2 and row 3.
TOP, BOTTOM, WSPACE = 0.950, 0.050, 0.10
PAIR_GAP, BLOCK_GAP = 0.040, 0.125
LETTER_DX = 0.002
BOX_PAD = 0.008       # frame margin around the panel content (figure fraction)
INNER = 0.010         # content must stay this far inside a frame edge (figure fraction)
ROUND = 0.012         # corner radius of the frames (figure fraction)
FRAME = dict(fc="none", ec="0.72", lw=1.2)


def load_inputs():
    files = ["summary.json", "section.npz"] + [f"panels/{p}.png" for p in PANELS]
    results.require(FIGURE, *files, produced_by=BUILD_CMD)
    base = results.figure_dir(FIGURE)
    summary = json.loads((base / "summary.json").read_text())
    images = {p: mpimg.imread(base / "panels" / f"{p}.png") for p in PANELS}
    with np.load(base / "section.npz") as npz:
        section = {kind: [npz[k] for k in sorted(npz.files) if k.startswith(kind + "_")]
                   for kind in ("bone", "cartilage", "fatpad")}
    return summary, images, section


# ------------------------------------------------------------------- section -----------
def draw_section(ax, section):
    for pl in section["bone"]:
        ax.plot(pl[:, 0], pl[:, 1], color=COL["section_bone"], lw=1.4)
    for pl in section["cartilage"]:
        ax.plot(pl[:, 0], pl[:, 1], color=COL["section_cart"], lw=2.2)
    for pl in section["fatpad"]:
        ax.plot(pl[:, 0], pl[:, 1], color=COL["fat"], lw=3.4, solid_capstyle="round")
    fp_pts = np.vstack(section["fatpad"])
    lo = fp_pts[np.argmin(fp_pts[:, 1])][:2]
    hi = fp_pts[np.argmax(fp_pts[:, 1])][:2]
    ax.annotate("1 mm off the bone", xy=lo, xytext=(lo[0] - 26, lo[1] - 9), fontsize=11,
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), color="0.2")
    ax.annotate("6 mm off the bone", xy=hi, xytext=(hi[0] - 31, hi[1] + 1), fontsize=11,
                arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8), color="0.2")
    ax.set_aspect("equal")
    ax.set_ylim(top=float(fp_pts[:, 1].max()) + 7)
    ax.set_xlabel("anterior  (mm)", fontsize=12)
    ax.set_ylabel("superior  (mm)", fontsize=12)
    ax.tick_params(labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=COL["section_bone"], lw=1.4, label="bone"),
                       Line2D([], [], color=COL["section_cart"], lw=2.2, label="cartilage"),
                       Line2D([], [], color=COL["fat"], lw=3.4, label="fat-pad surface")],
              loc="upper left", bbox_to_anchor=(0.07, 0.70), **LEG)  # inside the femur outline


# ------------------------------------------------------------------- composite ---------
def show(ax, img, title, size=14):
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=size, pad=6)


def letter(fig, cell, s):
    """Bold panel letter at the top-left corner of a GridSpec cell."""
    pos = cell.get_position(fig)
    return fig.text(pos.x0 + LETTER_DX, pos.y1 + 0.010, s, fontsize=24, fontweight="bold",
                    va="bottom", ha="left")


def fit_into_columns(fig, axes_cells, r, inv):
    """Narrow any axes whose tick labels, axis label or title would poke out of its column's
    frame -- panel F's y-axis does, and its title would once the axes moves right. Two
    passes: an equal-aspect axes that is narrowed also gets shorter, which widens a centred
    title's overhang a little."""
    for _ in range(2):
        moved = False
        for ax, cell in axes_cells:
            c = cell.get_position(fig)
            tb = ax.get_tightbbox(r).transformed(inv)
            pos = ax.get_position()  # the aspect-adjusted (drawn) position
            over_l = max(0.0, (c.x0 - BOX_PAD + INNER) - tb.x0)
            over_r = max(0.0, tb.x1 - (c.x1 + BOX_PAD - INNER))
            if over_l > 1e-4 or over_r > 1e-4:
                x0, x1 = pos.x0 + over_l, pos.x1 - over_r
                ax.set_position([x0, c.y0, x1 - x0, c.height], which="both")
                moved = True
        if not moved:
            break
        fig.canvas.draw()


def draw_frames(fig, top_arts, bot_arts, cols, axes_cells):
    """Six frames, placed from the drawn extents of the panel content (titles, letters,
    colourbar, axis labels) so nothing is hand-positioned. The frames share the column
    x-extent of the grid so they line up; y comes from the content."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    fit_into_columns(fig, axes_cells, r, inv)

    def extent(arts):
        bbs = []
        for a in arts:
            bb = a.get_tightbbox(r) if isinstance(a, Axes) else a.get_window_extent(r)
            if bb is not None and bb.width > 0 and bb.height > 0:
                bbs.append(bb)
        return Bbox.union(bbs).transformed(inv)

    top_bb, bot_bb = extent(top_arts), extent(bot_arts)
    for cell in (c.get_position(fig) for c in cols):
        for bb in (top_bb, bot_bb):
            fig.add_artist(FancyBboxPatch(
                (cell.x0 - BOX_PAD, bb.y0 - BOX_PAD), cell.width + 2 * BOX_PAD,
                bb.height + 2 * BOX_PAD, boxstyle=f"round,pad=0,rounding_size={ROUND}",
                transform=fig.transFigure, zorder=-5, **FRAME))


def compose(summary, imgs, section):
    fig = plt.figure(figsize=(16.5, 17.3))
    h = (TOP - BOTTOM - PAIR_GAP - BLOCK_GAP) / 3
    grid = dict(figure=fig, left=0.035, right=0.99, wspace=WSPACE)
    gs_top = GridSpec(2, 3, top=TOP, bottom=TOP - 2 * h - PAIR_GAP, hspace=PAIR_GAP / h, **grid)
    gs_bot = GridSpec(1, 3, top=BOTTOM + h, bottom=BOTTOM, **grid)

    cells = dict(A1=gs_top[0, 0], A2=gs_top[1, 0], B1=gs_top[0, 1], B2=gs_top[1, 1],
                 C1=gs_top[0, 2], C2=gs_top[1, 2], D=gs_bot[0, 0], E=gs_bot[0, 1], F=gs_bot[0, 2])
    axA1 = fig.add_subplot(cells["A1"]); show(axA1, imgs["ref"], "Reference model\nnon-physiologic cartilage", size=16)
    axA2 = fig.add_subplot(cells["A2"]); show(axA2, imgs["subj"], "Subject NSM reconstruction\nphysiologic cartilage", size=16)
    axB1 = fig.add_subplot(cells["B1"]); show(axB1, imgs["seed"], "Step 1  Identify the candidate region\nfor the prefemoral fat pad")
    axB2 = fig.add_subplot(cells["B2"]); show(axB2, imgs["seed_pat"], "Step 1  …the candidate region relative\nto the patella (MRI position)")
    axC1 = fig.add_subplot(cells["C1"]); show(axC1, imgs["vec"], "Step 2  Project each candidate vertex\nalong its surface normal")
    axC2 = fig.add_subplot(cells["C2"]); show(axC2, imgs["dil"], "Step 2  Dilated surface,\ncoloured by projection distance")
    axD = fig.add_subplot(cells["D"]); show(axD, imgs["crop"], "Step 3  Crop to within 25 mm of the patella")
    axE = fig.add_subplot(cells["E"]); show(axE, imgs["final"], "Step 4  Smooth and resample (5,000 vertices)")
    axF = fig.add_subplot(cells["F"]); draw_section(axF, section)
    axF.set_title("Sagittal cross-section through the patella centre", fontsize=14, pad=6)
    axF.set_facecolor("none")

    letters = {s: letter(fig, cells[key], s)
               for key, s in (("A1", "A"), ("B1", "B"), ("C1", "C"), ("D", "D"), ("E", "E"), ("F", "F"))}

    # C2: point at the patch whose offset the patella clearance capped
    xy = summary.get("capped_patch_xy_px")
    if xy is not None:
        hh, ww = imgs["dil"].shape[:2]
        axC2.annotate("projection capped where\nthe ray met the patella", xy=xy,
                      xytext=(0.98 * ww, 0.03 * hh), fontsize=13, ha="right", va="top",
                      color="black", arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4,
                                                     mutation_scale=18, shrinkB=4))

    cax = axC2.inset_axes([0.40, -0.075, 0.50, 0.035])
    cax.set_facecolor("none")
    sm = cm.ScalarMappable(norm=colors.Normalize(1, 6), cmap="viridis")
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("projection distance (mm)", fontsize=12.5)
    cb.ax.tick_params(labelsize=11)
    cb.set_ticks([1, 2, 3, 4, 5, 6])

    # One legend for the B pair, in the band under the lower knee that C's colourbar uses
    seed_handles = [Patch(fc=COL["seed"], label="candidate region: exposed anterior bone"),
                    Patch(fc=COL["cart"], label="cartilage"),
                    Patch(fc=COL["bone"], label="bone")]
    patella_handle = Patch(fc=COL["bone"], alpha=0.4, label="patella (translucent)")
    axB2.legend(handles=seed_handles + [patella_handle], loc="upper center",
                bbox_to_anchor=(0.5, -0.015), **LEG)
    axC1.legend(handles=[Line2D([], [], color=COL["arrow"], lw=5, label="projected 1 mm at the top of the trochlea\n→ 6 mm at the top of the femur"),
                         Line2D([], [], color=COL["arrow_cap"], lw=5, label="stopped short at the patella"),
                         Patch(fc=COL["seed_pale"], label="candidate region")],
                loc="lower left", **LEG)
    axD.legend(handles=[Patch(fc=COL["fat"], label="kept: < 25 mm from the patella"),
                        Patch(fc=COL["discard"], label="discarded")],
               loc="upper left", **LEG)
    axE.legend(handles=[Patch(fc=COL["fat"], ec=COL["fat_edge"], label="prefemoral fat-pad contact surface")],
               loc="upper left", **LEG)

    fig.text(0.035 - BOX_PAD, 0.006,
             f"exemplar knee {summary['knee']}  ·  production Step-1 geometry, femur frame",
             fontsize=9.5, color="0.45", ha="left", va="bottom")

    draw_frames(fig,
                [axA1, axA2, axB1, axB2, axC1, axC2, cax] + [letters[s] for s in "ABC"],
                [axD, axE, axF] + [letters[s] for s in "DEF"],
                [cells[k] for k in ("D", "E", "F")],
                axes_cells=[(ax, cells[k]) for ax, k in
                            ((axA1, "A1"), (axA2, "A2"), (axB1, "B1"), (axB2, "B2"),
                             (axC1, "C1"), (axC2, "C2"), (axD, "D"), (axE, "E"), (axF, "F"))])
    return fig


def main() -> int:
    summary, images, section = load_inputs()
    fig = compose(summary, images, section)
    stem = results.output_dir() / FIGURE
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
