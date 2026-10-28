#!/usr/bin/env python3
"""Figure 6 -- what determines how hard a knee is loaded.

    python figures/figure6_loading_determinants.py

Runs with no arguments, reads ``04_results/figure6_loading_determinants/`` and
writes manuscript Figure 6 as ``figures/output/[<run>/]figure6_loading_determinants.png``.

The figure
----------
Every knee is walked through nine measured gait conditions whose donors differ in
knee adduction moment and body mass, so the anatomy is held fixed and the loading
varies -- the mirror of Figure 3, which fixes the loading and varies the anatomy.

A-C  the nine gait conditions through stance: knee flexion angle and the
     external knee flexion and adduction moments, one line per condition
     (median over its knees, inter-quartile band), from ``gait_waveforms.csv``
     built by ``pipeline/derive_figure6_gait_waveforms.py``.
D-F  medial tibial cumulative pressure against each loading covariate. Small dots
     are single simulations coloured by condition; the large outlined dot is that
     condition's mean. The line is fitted through the nine means and the
     annotation is their correlation.
G-I  where on the joint surface that association sits: the per-vertex correlation
     between the nine condition-mean pressure maps and the same nine covariate
     values, femur above tibia, on the full −1..+1 scale.

Panels D-I relate medial tibial pressure, overall and per vertex, to the
loading covariates of the nine conditions.

n = 9, and the point cloud is not the sample
--------------------------------------------
Adduction moment and body mass are properties of the gait donor, not of the knee,
so all of the simulations carry only nine distinct loading conditions. The dots
show the within-condition spread; every statistic is computed across the nine
condition means, and both the fit line and the quoted r use those nine points.

No cluster inference
--------------------
The maps carry no spatial-parametric-mapping significance. A per-simulation
regression would test nine distinct covariate values at the degrees of freedom
of thousands of simulations (pseudoreplication), so panels G-I are descriptive:
nothing is thresholded and nothing is masked. The inferential model is the
crossed mixed model in ``figures/figure6_mundlak_model.R``, whose
between-condition term is the model counterpart of the correlations in D-F; its
output is not an input here, so this figure renders without R.

Inputs
------
Everything comes from ``04_results``. The correlations are read from
``gait_level_correlations.csv``, and the map panels
are pre-rendered by the derivation -- so redrawing this figure needs neither the
simulation tree nor a 3D toolkit.
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
import pandas as pd  # noqa: E402
from matplotlib import cm, colors  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from comak_knee_oa import cohorts  # noqa: E402
from shared import palette, panels, results  # noqa: E402

FIGURE = "figure6_loading_determinants"

BUILD_CMD = "python pipeline/derive_figure6_data.py"
WAVEFORMS_CMD = "python pipeline/derive_figure6_gait_waveforms.py"

# The gait-waveform row: variable -> (title, y-axis label)
WAVEFORM_PANELS = [
    ("knee_flexion_deg", "Knee flexion angle", "Angle (°)"),
    ("kfm_ext_nm", "Knee flexion moment", "Moment (N·m)"),
    ("kam_ext_nm", "Knee adduction moment", "Moment (N·m)"),
]

# One column per covariate, left to right.
COLUMNS = [
    ("kam_peak1", "KAM Peak 1", "KAM peak 1 (N·m)"),
    ("kam_peak2", "KAM Peak 2", "KAM peak 2 (N·m)"),
    ("body_mass", "Body Mass", "Body mass (kg)"),
]

# The scatter's outcome. Medial tibial cartilage is where this cohort's disease
# concentrates, and cumulative pressure is the dose the companion figures use.
OUTCOME = "tibia_med_cum"
OUTCOME_LABEL = "Medial tibia cumulative\npressure (MPa·s)"

BONE_ROWS = ("femur", "tibia")

MAP_CLIM = (-1.0, 1.0)

FS_SUPTITLE = 25
FS_TITLE = 21
FS_LABEL = 15
FS_TICK = 13
FS_LEGEND = 14
FS_ANATOMY = 17
FS_BONE = 20
FS_LETTER = 26
# The letter sits just above each panel's top-left corner. Column 0's rotated
# y-label is tall enough that the shared margin offset would land on top of it.
LETTER_OFFSET = (-4, 8)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_inputs():
    """Everything the figure draws, or a message naming the command that builds it."""
    results.require(
        FIGURE,
        "loading_cells.csv", "gait_means.csv", "gait_level_correlations.csv",
        "derivation_meta.json",
        produced_by=BUILD_CMD,
    )
    base = results.figure_dir(FIGURE)

    cells = pd.read_csv(base / "loading_cells.csv")
    gait_means = pd.read_csv(base / "gait_means.csv")
    stats = pd.read_csv(base / "gait_level_correlations.csv")
    meta = json.loads((base / "derivation_meta.json").read_text(encoding="utf-8"))

    map_dir = base / "maps"
    maps = {p.stem: p for p in map_dir.glob("*.png")} if map_dir.is_dir() else {}
    return cells, gait_means, stats, meta, maps


def load_waveforms():
    """The per-gait waveform table for the gait-condition row."""
    results.require(FIGURE, "gait_waveforms.csv", produced_by=WAVEFORMS_CMD)
    return pd.read_csv(results.figure_dir(FIGURE) / "gait_waveforms.csv")


def draw_gait_waveforms(axes, waveforms, gaits, colours):
    """One line per gait condition (median over its knees), faint IQR band."""
    for ax, (variable, title, ylabel) in zip(axes, WAVEFORM_PANELS):
        sub = waveforms[waveforms.variable == variable]
        for gait in gaits:
            g = sub[sub.gait == gait].sort_values("pct_stance")
            if g.empty:
                continue
            ax.fill_between(g.pct_stance, g.q25, g.q75, color=colours[gait], alpha=0.12, lw=0)
            ax.plot(g.pct_stance, g["median"], color=colours[gait], lw=2.2)
        if variable != "knee_flexion_deg":
            ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=FS_TITLE - 2, weight="bold", pad=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% stance", fontsize=FS_LABEL)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.25)


def gait_style(gaits):
    """``{gait: colour}`` and ``{gait: legend label}``, in a fixed order.

    Colour comes from the shared palette by the condition's position in
    ``cohorts.MEASURED``, so the same condition is the same colour here as in
    every other figure that draws more than one.

    The colours come from the palette, not from another figure's saved output,
    so this figure renders on its own.
    """
    colours = {gait: palette.GAIT_MEASURED[i % len(palette.GAIT_MEASURED)]
               for i, gait in enumerate(cohorts.MEASURED)}
    labels = {gait: gait.replace("Subject_", "S") for gait in cohorts.MEASURED}
    ordered = [gait for gait in cohorts.MEASURED if gait in set(gaits)]
    return ordered, colours, labels


def annotation(stats, covariate) -> str:
    """``r = <r>   (p = <p>)`` -- read from the derivation, not recomputed."""
    row = stats[(stats.outcome == OUTCOME) & (stats.covariate == covariate)]
    if not len(row):
        return ""
    r, p = float(row.r.iloc[0]), float(row.p_value.iloc[0])
    p_text = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    return f"r = {r:.2f}   ({p_text})"


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def draw_scatter(ax, cells, gait_means, stats, covariate, xlabel, gaits, colours,
                 with_ylabel):
    """One simulation per small dot, one condition per large dot, fit through nine.

    The line is fitted to the nine condition means rather than to the cloud, so
    it describes the same regression the annotation quotes. A line through
    ~6,600 points would look better supported and be the pseudoreplicated fit.
    """
    # Drawn one condition at a time, in the shared cohort order -- the same order
    # as the legend and the palette. With 30% alpha the overlap order is
    # cosmetic (nothing is hidden).
    for gait in gaits:
        member = cells.gait == gait
        ax.scatter(cells.loc[member, covariate], cells.loc[member, OUTCOME],
                   s=11, color=colours[gait], alpha=0.30, edgecolor="none", zorder=2)

    means = gait_means.set_index("gait").reindex(gaits)
    x, y = means[covariate].to_numpy(float), means[OUTCOME].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 2:
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        # Drawn across the plotted range rather than the fitted points' range, so
        # the line spans the cloud instead of stopping at the outermost mean.
        low, high = cells[covariate].min(), cells[covariate].max()
        pad = 0.02 * (high - low)
        span = np.linspace(low - pad, high + pad, 50)
        ax.plot(span, slope * span + intercept, "k-", lw=2.2, zorder=4)

    for gait in gaits:
        ax.scatter(means.at[gait, covariate], means.at[gait, OUTCOME], s=120,
                   color=colours[gait], edgecolor="black", linewidth=1.3, zorder=5)

    text = annotation(stats, covariate)
    if text:
        ax.text(0.04, 0.96, text, transform=ax.transAxes, fontsize=FS_LABEL + 1,
                va="top", weight="bold")

    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if with_ylabel:
        ax.set_ylabel(OUTCOME_LABEL, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.3)


def draw_maps(figure, grid, maps):
    """The six correlation panels, femur over tibia, one column per covariate.

    They arrive already oriented, cropped and on a common physical scale -- the
    derivation does that, because putting a femur and a tibia on one scale needs
    all six panels at once and the reference geometry to measure them against.
    """
    axes = []
    for row, bone in enumerate(BONE_ROWS):
        row_axes = []
        for column, (covariate, _header, _xlabel) in enumerate(COLUMNS):
            ax = figure.add_subplot(grid[row, column])
            ax.axis("off")
            row_axes.append(ax)
            image = maps.get(f"{bone}__{covariate}")
            if image is not None:
                ax.imshow(mpimg.imread(image))
            else:
                ax.text(0.5, 0.5, f"run {BUILD_CMD}", ha="center", va="center",
                        color="gray", transform=ax.transAxes)
        axes.append(row_axes)
    return axes


def label_map_block(figure, map_axes):
    """Anatomical directions once around the whole block, not once per panel."""
    boxes = [ax.get_position() for row in map_axes for ax in row]
    left = min(box.x0 for box in boxes)
    right = max(box.x1 for box in boxes)
    bottom = min(box.y0 for box in boxes)
    top = max(box.y1 for box in boxes)
    centre_x, centre_y = (left + right) / 2, (bottom + top) / 2

    style = dict(fontsize=FS_ANATOMY, weight="bold", color="#222")
    figure.text(centre_x, top + 0.005, "Anterior", ha="center", va="bottom", **style)
    figure.text(centre_x, bottom - 0.008, "Posterior", ha="center", va="top", **style)
    figure.text(left - 0.028, centre_y, "Lateral", ha="center", va="center",
                rotation=90, **style)
    figure.text(right + 0.020, centre_y, "Medial", ha="center", va="center",
                rotation=270, **style)

    for row, bone in enumerate(BONE_ROWS):
        row_boxes = [ax.get_position() for ax in map_axes[row]]
        row_centre = (min(b.y0 for b in row_boxes) + max(b.y1 for b in row_boxes)) / 2
        figure.text(left - 0.058, row_centre, bone.capitalize(), rotation=90,
                    ha="center", va="center", fontsize=FS_BONE, weight="bold")
    return left, right, bottom, top


def draw_colorbar(figure, left, right, bottom, top):
    mappable = cm.ScalarMappable(norm=colors.Normalize(*MAP_CLIM),
                                 cmap=palette.DIVERGING_CMAP)
    cax = figure.add_axes([right + 0.048, bottom + 0.04, 0.017, (top - bottom) * 0.80])
    bar = figure.colorbar(mappable, cax=cax)
    # Short. What the correlation is between (nine condition-mean pressure maps
    # and nine covariate values) is stated in the caption.
    bar.set_label("Correlation (r)", fontsize=FS_ANATOMY)
    bar.set_ticks([-1, -0.5, 0, 0.5, 1])
    bar.ax.tick_params(labelsize=FS_TICK + 1)


def render() -> Path:
    cells, gait_means, stats, _, maps = load_inputs()
    gaits, colours, labels = gait_style(gait_means.gait)
    waveforms = load_waveforms()

    left_edge, right_edge = 0.085, 0.860
    # Three bands: gait waveforms, scatters, maps.
    figure = plt.figure(figsize=(16.5, 14.8))
    # The waveform and scatter rows share one panel height.
    wave_grid = figure.add_gridspec(1, len(COLUMNS), left=left_edge, right=right_edge,
                                    top=0.905, bottom=0.725, wspace=0.28)
    scatter_grid = figure.add_gridspec(1, len(COLUMNS), left=left_edge, right=right_edge,
                                       top=0.618, bottom=0.438, wspace=0.17)
    map_grid = figure.add_gridspec(len(BONE_ROWS), len(COLUMNS), left=left_edge,
                                   right=right_edge, top=0.340, bottom=0.030,
                                   hspace=0.02, wspace=0.02)
    wave_axes = [figure.add_subplot(wave_grid[0, c]) for c in range(len(COLUMNS))]
    draw_gait_waveforms(wave_axes, waveforms, gaits, colours)
    # This row's y-labels are short, so its letters can sit left of the axes
    # and clear of the (long) titles.
    for column, ax in enumerate(wave_axes):
        panels.add_panel_letter(ax, panels.LETTERS[column], fontsize=FS_LETTER,
                                offset=(-26, 8))
    letter_base = len(COLUMNS)
    # The title at the top, the gait legend between the gait row and the scatters.
    legend_y, title_y = 0.672, 0.975

    for column, (covariate, header, xlabel) in enumerate(COLUMNS):
        ax = figure.add_subplot(scatter_grid[0, column])
        draw_scatter(ax, cells, gait_means, stats, covariate, xlabel, gaits, colours,
                     with_ylabel=(column == 0))
        ax.set_title(header, fontsize=FS_TITLE, weight="bold", pad=12)
        panels.add_panel_letter(ax, panels.LETTERS[letter_base + column], fontsize=FS_LETTER,
                                offset=LETTER_OFFSET)

    map_axes = draw_maps(figure, map_grid, maps)
    # One letter per map column: a column is the unit here, the
    # same covariate rendered on the femur over the tibia. Anchored on the top row.
    for column in range(len(COLUMNS)):
        panels.add_panel_letter(map_axes[0][column],
                                panels.LETTERS[letter_base + len(COLUMNS) + column],
                                fontsize=FS_LETTER, offset=LETTER_OFFSET)

    figure.canvas.draw()
    draw_colorbar(figure, *label_map_block(figure, map_axes))

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colours[g],
                          markeredgecolor="black", markersize=12, label=labels[g])
               for g in gaits]
    # No legend title: it would sit directly under the middle panel's x-axis
    # label.
    figure.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, legend_y),
                  ncol=min(len(gaits), 9), fontsize=FS_LEGEND, frameon=False,
                  handletextpad=0.3, columnspacing=1.2)

    figure.suptitle("Loading determinants of pressure", fontsize=FS_SUPTITLE,
                    weight="bold", y=title_y)

    out_path = results.output_dir() / f"{FIGURE}.png"
    figure.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return out_path


def main() -> None:
    _, _, stats, meta, _ = load_inputs()
    print(f"Figure 6: {meta['n_gaits']} gait conditions "
          f"({meta['n_simulations']:,} simulations over {meta['n_knees']} knees)")
    for covariate, _header, _xlabel in COLUMNS:
        row = stats[(stats.outcome == OUTCOME) & (stats.covariate == covariate)]
        if len(row):
            print(f"  {OUTCOME} vs {covariate:10s} "
                  f"r = {float(row.r.iloc[0]):+.3f}  p = {float(row.p_value.iloc[0]):.4f}")
    print(f"  wrote {render()}")


if __name__ == "__main__":
    main()
