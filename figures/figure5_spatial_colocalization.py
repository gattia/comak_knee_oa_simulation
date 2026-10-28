#!/usr/bin/env python3
"""Figure 5 -- where contact pressure and cartilage thickness change coincide.

One image per pressure measure, each holding femur above tibia in every block,
on the shared 20,000-vertex reference cartilage surface:

    cumulative pressure  manuscript Figure 5
                         figures/output/[<run>/]figure5_spatial_colocalization.png
    peak (max) pressure  manuscript Extended Data Figure 2
                         figures/output/[<run>/]figure_ed2_spatial_colocalization_max.png

    A  mean contact pressure                    both walking conditions
    B  mean cartilage thickness change          gait-invariant, drawn once
    C  per-vertex slope of thickness change on pressure   both conditions
    D  correlation t statistic, surviving clusters only   both conditions
    ---------------------------------------------------------------------
    E  thickness change by disease group        Healthy / Incident / Mild OA
    F  group contrast in pressure               two contrasts x both conditions

The top block is the pooled cohort across all four quantities; the bottom block
is where the groups differ. The horizontal rule between them separates "what the
joint does" from "how it differs by group", which is the figure's argument.

Both gait columns show the same knees and differ only in gait donor, so the
column labels name the gait, and the figure shows that the spatial claim does
not depend on which gait was simulated.

What this script does and does not do
-------------------------------------
It composes images. Every surface was rendered by
``pipeline/derive_figure5_data.py``, already cropped to a shared bounding box and
flipped into anatomical convention, so no 3D toolkit runs here (the same
arrangement Figure 3 uses for its decoded anatomy): this script needs only
matplotlib and NumPy, not a VTK stack.

Color limits are read from ``layout.json`` rather than recomputed. They are
pooled across both bones and both conditions upstream so that one colorbar
captions the whole figure: a color means the same value on the femur as on the
tibia and under either gait. Recomputing them here would let the caption and the
pixels drift apart.

    python figures/figure5_spatial_colocalization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import palette, results  # noqa: E402

FIGURE = "figure5_spatial_colocalization"
# Cumulative pressure is Figure 5; peak pressure is Extended Data Figure 2.
OUTPUT_NAMES = {"cum": "figure5_spatial_colocalization.png",
                "max": "figure_ed2_spatial_colocalization_max.png"}
BONES = ("femur", "tibia")
PRESSURES = ("cum", "max")

PRODUCED_BY = "python pipeline/derive_figure5_data.py maps"

# --- geometry, in inches ----------------------------------------------------
# Laid out in inches from the top rather than in gridspec fractions. Panels are
# pre-cropped images whose aspect ratio differs between bones, so their height
# follows from their width; a fractional grid would stretch one bone.
CELL_WIDTH = 1.85
COLUMN_GAP = 0.06           # between the two gait columns of one block
BONE_GAP = 0.05
# Seven columns all COLUMN_GAP apart would read as one strip, so the gaps
# between lettered blocks -- A|B, B|C, C|D -- are wider and each block stands on
# its own. The bottom block does the same: the E|F gap carries the vertical
# rule, and F's two contrasts are separated by a block gap of their own.
BLOCK_GAP_TOP = 0.55        # between lettered blocks in the top row, and between F's contrasts
BLOCK_GAP = 0.90            # E|F gap, carrying the vertical rule
LEFT_MARGIN = 0.95          # room for the Lateral edge label and bone names
RIGHT_MARGIN = 0.45         # room for the Medial edge label
TOP_MARGIN = 0.62           # Anterior label, above the colorbars
BOTTOM_MARGIN = 0.42        # Posterior label
DIVIDER_GAP = 0.34          # the band the dashed rule sits in
HEADER_TOP = 1.14           # four colorbars, their labels, and gait sublabels
HEADER_BOTTOM = 1.24        # thickness change and delta-pressure, side by side

# Which columns end a block, and how wide the gap after them is. Top row: A is
# columns 0-1, B is 2, C is 3-4, D is 5-6. Bottom row: E is 0-2, F is 3-6 with
# its two contrasts at 3-4 and 5-6.
TOP_BLOCK_COLUMNS = ((0, 1), (2, 2), (3, 4), (5, 6))
TOP_BLOCK_GAPS = {1: BLOCK_GAP_TOP, 2: BLOCK_GAP_TOP, 4: BLOCK_GAP_TOP}
BOTTOM_BLOCK_GAPS = {2: BLOCK_GAP, 4: BLOCK_GAP_TOP}

# --- type -------------------------------------------------------------------
FS_COLORBAR_TICK = 11
FS_COLORBAR_LABEL = 12.5
FS_CONDITION = 12
FS_GROUP = 14
FS_ANATOMY = 16.5
FS_CONTRAST = 14
FS_CONTRAST_HEADER = 14.5
FS_SECTION_LETTER = 18
# Section letters sit this far left of their block. A colorbar label is centred
# on its block and can be wider than it, so a letter flush with the edge would
# be overprinted; a fixed offset keeps it clear whatever the label length.
LETTER_OFFSET = 0.10

INK = "#111111"
MUTED = "#444444"
RULE = "#555555"

# Outcome wording for panels B and E, not ``palette.OUTCOME_LABEL`` ("Cartilage
# thickness change"): the signed quantity reads as a delta, so blue = negative =
# thinning needs no double negative in the caption.
THICKNESS_LABEL = "Δ cartilage thickness\n(µm/yr)"
THICKNESS_LABEL_NARROW = "Δ cartilage\nthickness\n(µm/yr)"


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def load_layout() -> dict:
    (path,) = results.require(FIGURE, "layout.json", produced_by=PRODUCED_BY)
    return json.loads(path.read_text())


def load_population() -> dict:
    (path,) = results.require(FIGURE, "population.json", produced_by=PRODUCED_BY)
    return json.loads(path.read_text())


def panel_path(pressure: str, bone: str, name: str) -> Path:
    return results.figure_dir(FIGURE) / "panels" / pressure / bone / f"{name}.png"


# --------------------------------------------------------------------------- #
# Colorbar wording                                                            #
# --------------------------------------------------------------------------- #
def pressure_words(pressure: str) -> tuple[str, str]:
    """``(name, unit)`` for a pressure measure.

    Cumulative pressure is a time integral over stance, so it carries a second
    of time in its unit; the maximum over stance does not.

    "Max" rather than "Peak" because elsewhere in this manuscript "peak" names
    a *stance* peak -- Figure 2's early- and late-stance peaks -- so the two
    words are not interchangeable here even though they would be in isolation.
    """
    return ({"cum": "Cumulative", "max": "Max"}[pressure],
            "MPa·s" if pressure == "cum" else "MPa")


def colorbar_spec(quantity: str, pressure: str, narrow: bool = False) -> tuple:
    """``(scale, label, tick format)``.

    Panels are colored in SI (Pa·s, mm/yr); the colorbars are *labelled* in the
    units the manuscript quotes. ``scale`` converts the limits for the tick
    labels only -- the color-to-value mapping is untouched, so nothing shifts.

    ``narrow`` wraps the thickness label onto three lines. It is needed in the
    top block and only there: that colorbar spans a single column, since
    thickness change does not depend on gait, while every other block spans two
    or more. At two lines the label fills its column edge to edge and collides
    with the section letters on both sides.
    """
    name, unit = pressure_words(pressure)
    return {
        "pressure": (1e-6, f"{name} pressure\n({unit})", "%.1f"),
        "thinning": (1e3, THICKNESS_LABEL_NARROW if narrow else THICKNESS_LABEL, "%.0f"),
        # mm/yr per Pa(·s) -> µm/yr per MPa(·s): 1e3 for the outcome, 1e6 for
        # the predictor.
        "slope": (1e9, f"β slope\n(µm/yr per {unit})", "%.0f"),
        "spm": (1.0, "SPM corr. t-stat\n(significant only)", "%.1f"),
    }[quantity]


def contrast_header(pressure: str) -> tuple:
    name, unit = pressure_words(pressure)
    return 1e-6, f"Group-contrast Δ {name.lower()} pressure  ({unit})", "%.2f"


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def column_offset(index: int, block_gaps: dict[int, float]) -> float:
    """Left edge of column ``index``, in inches from column 0's left edge.

    Columns are ``CELL_WIDTH`` wide and ``COLUMN_GAP`` apart, except after the
    columns named in ``block_gaps``, where that wider gap replaces the ordinary
    one. The width of a whole row is ``column_offset(last) + CELL_WIDTH``.
    """
    x = index * (CELL_WIDTH + COLUMN_GAP)
    for after, gap in block_gaps.items():
        if index > after:
            x += gap - COLUMN_GAP
    return x


def render(pressure: str, layout: dict, population: dict) -> Path:
    spec = layout["pressures"][pressure]
    limits = spec["limits"]
    contrast_limit = spec["contrast_limit"]
    conditions = layout["conditions"]
    condition_label = layout["condition_labels"]
    contrasts = layout["contrasts"]

    images, aspect = {}, {}
    for bone in BONES:
        for name in spec["panels"][bone]:
            path = panel_path(pressure, bone, name)
            if not path.is_file():
                raise results.MissingFigureInput(
                    f"panel {path} is listed in layout.json but absent.\n\n"
                    f"Rebuild with:\n    {PRODUCED_BY}"
                )
            images[(bone, name)] = mpimg.imread(path)
        sample = images[(bone, spec["panels"][bone][0])]
        aspect[bone] = sample.shape[1] / sample.shape[0]

    bone_height = {b: CELL_WIDTH / aspect[b] for b in BONES}
    two_rows = bone_height["femur"] + BONE_GAP + bone_height["tibia"]

    top_width = column_offset(6, TOP_BLOCK_GAPS) + CELL_WIDTH
    fig_width = LEFT_MARGIN + top_width + RIGHT_MARGIN
    fig_height = (TOP_MARGIN + HEADER_TOP + two_rows + DIVIDER_GAP
                  + HEADER_BOTTOM + two_rows + BOTTOM_MARGIN)

    fig = plt.figure(figsize=(fig_width, fig_height))

    def rect(x, top, width, height):
        return [x / fig_width, 1 - (top + height) / fig_height,
                width / fig_width, height / fig_height]

    def text(x, top, string, **kwargs):
        kwargs.setdefault("va", "top")
        kwargs.setdefault("ha", "center")
        return fig.text(x / fig_width, 1 - top / fig_height, string, **kwargs)

    def top_column_x(index):
        return LEFT_MARGIN + column_offset(index, TOP_BLOCK_GAPS)

    # The bottom block reuses a seven-column grid -- three for thickness change,
    # four for the contrasts -- centred under the top one. Its block gaps differ
    # from the top row's, so the two rows' columns do not align, by design.
    bottom_width = column_offset(6, BOTTOM_BLOCK_GAPS) + CELL_WIDTH
    bottom_x0 = LEFT_MARGIN + (top_width - bottom_width) / 2

    def bottom_column_x(index):
        return bottom_x0 + column_offset(index, BOTTOM_BLOCK_GAPS)

    def section_letter(letter, x, top):
        text(x - LETTER_OFFSET, top, letter, ha="left", va="top",
             fontsize=FS_SECTION_LETTER, weight="bold", color=INK)

    def colorbar(x, width, top, height, colormap, limit, scale, fmt, label,
                 max_label_width=None):
        axes = fig.add_axes(rect(x, top, width, height))
        norm = matplotlib.colors.Normalize(vmin=limit[0] * scale, vmax=limit[1] * scale)
        bar = matplotlib.colorbar.ColorbarBase(
            axes, cmap=plt.get_cmap(colormap), norm=norm, orientation="horizontal")
        bar.ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        bar.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        bar.ax.tick_params(labelsize=FS_COLORBAR_TICK)
        if not label:
            return
        drawn = text(x + width / 2, top - 0.06, label, va="bottom",
                     fontsize=FS_COLORBAR_LABEL, weight="bold")
        # Measure rather than estimate. A guessed font size or line break for a
        # label that must not overprint its neighbours breaks again, with no
        # error, the next time any label's wording changes.
        if max_label_width:
            got = drawn.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
            if got > max_label_width:
                drawn.set_fontsize(FS_COLORBAR_LABEL * max_label_width / got)

    def place_row(top, names, bone, column_x):
        """One bone's row of panels; returns its height so the caller can advance."""
        height = bone_height[bone]
        for i, name in enumerate(names):
            axes = fig.add_axes(rect(column_x(i), top, CELL_WIDTH, height))
            axes.axis("off")
            if name is not None and (bone, name) in images:
                axes.imshow(images[(bone, name)])
            else:
                axes.text(0.5, 0.5, "—", ha="center", va="center",
                          color="lightgray", transform=axes.transAxes)
        fig.text((LEFT_MARGIN - 0.30) / fig_width,
                 1 - (top + height / 2) / fig_height, bone.capitalize(),
                 rotation=90, va="center", ha="center", fontsize=FS_GROUP,
                 weight="bold", color="#222222")
        return height

    # ---------------------------------------------------------------- top ---
    # One colorbar per block of columns that share a scale. Panels within a
    # block are not lettered individually -- their columns are already named by
    # the gait sublabels beneath.
    blocks = [
        ("A", "pressure", palette.SEQUENTIAL_CMAP),
        ("B", "thinning", palette.DIVERGING_CMAP),
        ("C", "slope", palette.DIVERGING_CMAP),
        ("D", "spm", palette.DIVERGING_CMAP),
    ]
    shrink = 0.84
    for (letter, quantity, colormap), (first, last) in zip(blocks, TOP_BLOCK_COLUMNS):
        x0 = top_column_x(first)
        x1 = top_column_x(last) + CELL_WIDTH
        margin = (1 - shrink) / 2 * (x1 - x0)
        scale, label, fmt = colorbar_spec(quantity, pressure, narrow=True)
        colorbar(x0 + margin, (x1 - x0) - 2 * margin, TOP_MARGIN + 0.46, 0.12,
                 colormap, limits[quantity], scale, fmt, label,
                 max_label_width=(x1 - x0) - 2 * (LETTER_OFFSET + 0.16))
        section_letter(letter, x0, TOP_MARGIN - 0.02)

    # Gait sublabels, dropped just above the panels so they clear the ticks.
    for column, condition_index in {0: 0, 1: 1, 3: 0, 4: 1, 5: 0, 6: 1}.items():
        text(top_column_x(column) + CELL_WIDTH / 2, TOP_MARGIN + HEADER_TOP - 0.05,
             condition_label[conditions[condition_index]], va="bottom",
             fontsize=FS_CONDITION, weight="bold", style="italic", color=MUTED)

    top_row_names = (
        [f"{conditions[0]}__all__pressure", f"{conditions[1]}__all__pressure"]
        + ["thinning__all"]
        + [f"{conditions[0]}__all__slope", f"{conditions[1]}__all__slope"]
        + [f"{conditions[0]}__all__spm", f"{conditions[1]}__all__spm"]
    )
    cursor = TOP_MARGIN + HEADER_TOP
    for bone in BONES:
        height = place_row(cursor, top_row_names, bone, top_column_x)
        cursor += height + (BONE_GAP if bone == "femur" else 0)
    top_bottom = cursor

    # ------------------------------------------------------------- bottom ---
    divider_y = top_bottom + DIVIDER_GAP / 2
    bottom_top = top_bottom + DIVIDER_GAP

    left_x0, left_x1 = bottom_column_x(0), bottom_column_x(2) + CELL_WIDTH
    width = 0.40 * (left_x1 - left_x0)
    scale, label, fmt = colorbar_spec("thinning", pressure)
    colorbar((left_x0 + left_x1) / 2 - width / 2, width, bottom_top + 0.42, 0.11,
             palette.DIVERGING_CMAP, limits["thinning"], scale, fmt, label)
    section_letter("E", left_x0, bottom_top + 0.04)
    for i, group in enumerate(layout["disease_groups"]):
        text(bottom_column_x(i) + CELL_WIDTH / 2, bottom_top + HEADER_BOTTOM - 0.05,
             palette.GROUP_LABELS[group], va="bottom", fontsize=FS_GROUP,
             weight="bold", color="#222222")

    right_x0, right_x1 = bottom_column_x(3), bottom_column_x(6) + CELL_WIDTH
    scale, header, fmt = contrast_header(pressure)
    text((right_x0 + right_x1) / 2, bottom_top + 0.04, header, va="top",
         fontsize=FS_CONTRAST_HEADER, weight="bold", color="#222222")
    section_letter("F", right_x0, bottom_top + 0.04)
    width = 0.26 * (right_x1 - right_x0)
    colorbar((right_x0 + right_x1) / 2 - width / 2, width, bottom_top + 0.36, 0.11,
             palette.DIVERGING_CMAP, contrast_limit, scale, fmt, None)

    for index, contrast in enumerate(contrasts):
        first_column = 3 + index * 2
        centre = (bottom_column_x(first_column)
                  + bottom_column_x(first_column + 1) + CELL_WIDTH) / 2
        text(centre, bottom_top + HEADER_BOTTOM - 0.28,
             f"{palette.GROUP_LABELS[contrast['first']]} − "
             f"{palette.GROUP_LABELS[contrast['second']]}",
             va="bottom", fontsize=FS_CONTRAST, weight="bold", color=INK)
        for offset, condition in enumerate(conditions):
            text(bottom_column_x(first_column + offset) + CELL_WIDTH / 2,
                 bottom_top + HEADER_BOTTOM - 0.05, condition_label[condition],
                 va="bottom", fontsize=FS_CONDITION, weight="bold",
                 style="italic", color=MUTED)

    bottom_names = [f"thinning__{g}" for g in layout["disease_groups"]]
    for contrast in contrasts:
        for condition in conditions:
            bottom_names.append(f"{condition}__contrast__{contrast['name']}")

    cursor = bottom_top + HEADER_BOTTOM
    for bone in BONES:
        height = place_row(cursor, bottom_names, bone, bottom_column_x)
        cursor += height + (BONE_GAP if bone == "femur" else 0)
    bottom_bottom = cursor

    # -------------------------------------------------------------- rules ---
    fig.add_artist(plt.Line2D(
        [0.03, 0.97], [1 - divider_y / fig_height] * 2, transform=fig.transFigure,
        color=RULE, lw=2.5, linestyle=(0, (8, 4))))
    vertical_x = (bottom_column_x(2) + CELL_WIDTH + bottom_column_x(3)) / 2
    fig.add_artist(plt.Line2D(
        [vertical_x / fig_width] * 2,
        [1 - (bottom_bottom - 0.02) / fig_height, 1 - (bottom_top + 0.02) / fig_height],
        transform=fig.transFigure, color=RULE, lw=2.0, linestyle=(0, (8, 4))))

    # ------------------------------------------------------------- anatomy ---
    # Once per figure, on the edges, because every panel shares one orientation.
    panels_top = TOP_MARGIN + HEADER_TOP
    middle = (panels_top + fig_height - BOTTOM_MARGIN) / 2
    text(fig_width / 2, 0.16, "Anterior", va="top", fontsize=FS_ANATOMY,
         fontweight="bold", color="#222222")
    text(fig_width / 2, fig_height - 0.10, "Posterior", va="bottom",
         fontsize=FS_ANATOMY, fontweight="bold", color="#222222")
    fig.text(0.006, 1 - middle / fig_height, "Lateral", ha="left", va="center",
             rotation=90, fontsize=FS_ANATOMY, fontweight="bold", color="#222222")
    fig.text(1 - 0.006, 1 - middle / fig_height, "Medial", ha="right", va="center",
             rotation=-90, fontsize=FS_ANATOMY, fontweight="bold", color="#222222")

    out_path = results.output_dir() / OUTPUT_NAMES[pressure]
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)

    counts = {c: population[f"{c}__femur__{pressure}"]["n"] for c in conditions}
    print(f"wrote {out_path}")
    print("  n = " + ", ".join(
        f"{counts[c]} ({condition_label[c]})" for c in conditions))
    return out_path


def main() -> None:
    layout = load_layout()
    population = load_population()
    for pressure in PRESSURES:
        render(pressure, layout, population)


if __name__ == "__main__":
    main()
