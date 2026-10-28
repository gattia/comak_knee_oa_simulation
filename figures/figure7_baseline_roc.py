#!/usr/bin/env python3
"""Figure 7 -- how well baseline knee mechanics predict who develops osteoarthritis.

    python figures/figure7_baseline_roc.py

Runs with no arguments, reads ``04_results/figure7_baseline_roc/`` and writes
manuscript Figure 7 as ``figures/output/[<run>/]figure7_baseline_roc.png``.

The figure
----------
A two-group classifier: knees that are KL0-1 at baseline and stay healthy against
knees that are KL0-1 at baseline and cross to KL2 or worse by 48 months. Every
combination of a shape representation and a pressure representation is fitted
independently under each of the ten walking conditions.

A  Every feature set, ranked. One bar per feature set, its height the mean across
   the ten conditions and its interval the 95% confidence interval of that mean;
   the individual conditions are not drawn (they would hide the interval; panel
   C shows them). The bars are one colour: a colour ramp on height would repeat the
   y-axis.
B  The same numbers as a grid, so the two axes of the comparison -- what shape
   information is used, what pressure information is added -- can be read
   against each other. Each cell carries the
   mean and, beneath it, the same across-gait 95% interval panel A draws. It is
   not a bootstrap over knees; see ``across_gaits``.
C  One polygon per walking condition over seven of the feature sets, which is
   where to look for whether the choice of gait changes the answer.

Panels A and B compare feature sets; panel C compares walking conditions.

Three of panel C's seven axes are gait-invariant by construction. A knee's
shape features are read from its one reference-size reconstruction (the
measured conditions scale the simulation model, not these features), so the
shape-only feature sets are the same inputs in every condition and only the
per-condition population, set by quality control, differs. The cross-gait
comparison that carries information is on the four axes involving pressure.

Panels A and B report one pre-specified model -- raw features with L2
regularisation, no dimensionality reduction. The derivation also fits three
principal-component reductions per feature set, which belong in the text as a
sensitivity analysis and are in ``auroc.csv``. Reporting the best reducer per cell
instead would mix models: raw for the shape feature sets and a reduction
for the pressure ones, in one grid.

Every ``n``
-----------
This figure's population is smaller than the matrices it comes from, because
Mild-OA knees are in those matrices and not in this contrast. It is one smaller
again for the twelve cartilage-shape configurations. The per-configuration counts are in ``auroc.csv``; quote them
from there rather than from a panel.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from scipy import stats  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import panels, palette, results  # noqa: E402

FIGURE = "figure7_baseline_roc"

BUILD_CMD = "python pipeline/derive_figure7_data.py"

# Panels A and B show one pre-specified model. See the docstring.
REPORTED_REDUCER = "none"

SHAPE_ORDER = ("none", "thickness", "xyz_thick", "latents")
PRESSURE_ORDER = ("none", "cum", "max")

# "NSM latents" is the shape model's own code for a knee -- the vector the neural
# shape model reconstructs that knee's anatomy from. The manuscript uses this
# wording; the derivation's docstring spells out what it is.
SHAPE_LABELS = {
    "none": "No shape",
    "thickness": "Cartilage thickness",
    "xyz_thick": "Cartilage shape + thickness",
    "latents": "NSM latents",
}
PRESSURE_LABELS = {
    "none": "No pressure",
    "cum": "Cumulative pressure",
    "max": "Peak pressure",
}

# Panel C's axes: the three shape-only feature sets, two shape-plus-pressure, and
# the two pressure-only ones, a readable subset of the 44 configurations.
RADAR_CONFIGS = (
    ("latents", "none"),
    ("xyz_thick", "none"),
    ("thickness", "none"),
    ("latents", "max"),
    ("xyz_thick", "max"),
    ("none", "max"),
    ("none", "cum"),
)

# Legend names for the walking conditions. The measured conditions are numbered
# rather than named: they are gait-trial participant identifiers, which mean
# nothing to a reader and would crowd a ten-entry legend. The reference condition
# is called out because it is the one whose donor is not an osteoarthritis
# patient -- see docs/data_provenance.md.
REFERENCE_LEGEND = "Healthy gait"

CHANCE = 0.50

FS_PANEL_TITLE = 21
FS_AXIS = 17
FS_TICK = 15
FS_LEGEND = 15.5


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_inputs():
    """The sweep table and its provenance, or a message naming what builds them."""
    results.require(FIGURE, "auroc.csv", "meta.json", produced_by=BUILD_CMD)
    base = results.figure_dir(FIGURE)
    auroc = pd.read_csv(base / "auroc.csv")
    meta = json.loads((base / "meta.json").read_text())
    return auroc, meta


def gait_order(auroc, meta) -> list[str]:
    """Conditions in canonical order, reference first.

    Read from ``meta.json`` rather than from the order rows happen to appear in:
    the derivation runs the conditions in parallel, and colours here are assigned
    by position, with index 0 the reference condition's reserved black. Taking the
    order from whichever condition finished first hands that black to an arbitrary
    condition -- and differently on every run.
    """
    present = set(auroc["gait"])
    return [gait for gait in meta["gaits"] if gait in present]


def gait_style(gaits, meta):
    """``{gait: (legend name, colour)}``. The reference condition owns index 0."""
    reference = meta.get("reference_gait")
    style, measured = {}, 0
    for i, gait in enumerate(gaits):
        if gait == reference:
            style[gait] = (REFERENCE_LEGEND, palette.GAIT_PALETTE[0])
        else:
            measured += 1
            style[gait] = (f"Gait {measured}", palette.gait_color(i))
    return style


def reported(auroc):
    """``{(shape, pressure): {gait: mean AUROC}}`` for the pre-specified model."""
    rows = auroc[auroc["reducer"] == REPORTED_REDUCER]
    table = defaultdict(dict)
    for row in rows.itertuples():
        table[(row.shape, row.pressure)][row.gait] = row.auroc_mean
    return table


def label_for(shape, pressure) -> str:
    """Two-line feature-set label, as the bar and radar axes need it."""
    if shape == "none":
        return f"{PRESSURE_LABELS[pressure]}\n(vertex-level)"
    if pressure == "none":
        return SHAPE_LABELS[shape].replace(" + ", "\n+ ")
    return f"{SHAPE_LABELS[shape]}\n+ {PRESSURE_LABELS[pressure]}"


def across_gaits(values):
    """``(mean, half-width of the 95% CI)`` over the walking conditions.

    A t interval on n = 10 conditions. This is a different quantity from the
    per-condition spread across cross-validation repeats that ``auroc.csv``
    carries as ``auroc_sd``: that one measures fold-assignment stability within a
    condition, this one how much the choice of condition moves the answer.
    Neither is a confidence interval over knees. Do not relabel either.
    """
    values = np.asarray(list(values), dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, float(half)


# --------------------------------------------------------------------------- #
# Panels                                                                      #
# --------------------------------------------------------------------------- #
def draw_ranked_bars(ax, table, gaits):
    """A -- every feature set, best first, with its across-gait 95% interval."""
    ranked = sorted(
        ((cfg, across_gaits(per_gait.values()), per_gait)
         for cfg, per_gait in table.items()),
        key=lambda row: -row[1][0],
    )
    x = np.arange(len(ranked))
    means = [row[1][0] for row in ranked]
    intervals = [row[1][1] for row in ranked]

    # One colour: height is already on the y-axis, and a ramp on it would be a
    # second encoding to explain.
    ax.bar(x, means, 0.72, yerr=intervals, capsize=9, color=palette.ACCENT,
           edgecolor="black", linewidth=0.6, alpha=0.9,
           error_kw=dict(ecolor="black", lw=2.6, capthick=2.6), zorder=3)
    # Per-gait points would sit on top of the interval and hide it, so the
    # interval alone is drawn. Panel C shows the per-gait values.

    # Chance is the dashed line; the caption names it (an in-panel word would be
    # too small to read at print size).
    ax.axhline(CHANCE, color="#555", linestyle="--", lw=1.4, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([label_for(*row[0]) for row in ranked], fontsize=15,
                       rotation=22, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, len(ranked) - 0.3)
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel(f"Mean AUROC\n(95% CI across {len(gaits)} gaits)",
                  fontsize=16.5, weight="bold")
    ax.set_title("Predictive value of each feature set", fontsize=FS_PANEL_TITLE,
                 weight="bold", pad=10, loc="left")
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def draw_feature_grid(fig, ax, table):
    """B -- the same means as a shape x pressure grid."""
    grid = np.full((len(SHAPE_ORDER), len(PRESSURE_ORDER)), np.nan)
    half = np.full_like(grid, np.nan)
    for si, shape in enumerate(SHAPE_ORDER):
        for pi, pressure in enumerate(PRESSURE_ORDER):
            per_gait = table.get((shape, pressure))
            if per_gait:
                grid[si, pi], half[si, pi] = across_gaits(per_gait.values())

    # Centred on chance rather than on the data, so colour reads as "how far
    # above a coin flip" -- the quantity the figure is about.
    norm = TwoSlopeNorm(vcenter=CHANCE, vmin=0.30, vmax=0.70)
    cmap = palette.DIVERGING_CMAP
    image = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    for si in range(len(SHAPE_ORDER)):
        for pi in range(len(PRESSURE_ORDER)):
            if np.isnan(grid[si, pi]):
                continue
            rgba = plt.get_cmap(cmap)(norm(grid[si, pi]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.55 else "black"
            ax.text(pi, si - 0.13, f"{grid[si, pi]:.2f}", ha="center", va="center",
                    color=text_color, fontsize=17, weight="bold")
            lo, hi = grid[si, pi] - half[si, pi], grid[si, pi] + half[si, pi]
            ax.text(pi, si + 0.20, f"({lo:.2f}–{hi:.2f})", ha="center", va="center",
                    color=text_color, fontsize=14)

    ax.set_xticks(range(len(PRESSURE_ORDER)))
    ax.set_xticklabels([PRESSURE_LABELS[p].replace(" ", "\n")
                        for p in PRESSURE_ORDER], fontsize=FS_AXIS)
    ax.set_yticks(range(len(SHAPE_ORDER)))
    ax.set_yticklabels([SHAPE_LABELS[s].replace(" + ", "\n+ ")
                        for s in SHAPE_ORDER], fontsize=FS_AXIS)
    ax.set_xlabel("Pressure source", fontsize=FS_AXIS, weight="bold")
    ax.set_ylabel("Shape source", fontsize=FS_AXIS, weight="bold")
    ax.set_title("AUROC by feature type", fontsize=FS_PANEL_TITLE, weight="bold",
                 pad=8, loc="left")

    bar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.025, fraction=0.05,
                       ticks=[0.30, 0.40, 0.50, 0.60, 0.70])
    bar.set_label("AUROC", fontsize=11)
    bar.ax.tick_params(labelsize=10)
    bar.ax.axhline(CHANCE, color="#333", lw=0.8)


def draw_gait_radar(ax, auroc, gaits, style):
    """C -- one polygon per walking condition over seven feature sets."""
    per_config = reported(auroc)
    configs = [cfg for cfg in RADAR_CONFIGS if cfg in per_config]
    angles = np.linspace(0, 2 * np.pi, len(configs), endpoint=False).tolist()
    angles += angles[:1]

    # The reference condition is drawn first so it heads the legend, and heavier
    # and black so it is findable among nine similar polygons.
    reference = next((g for g, (name, _) in style.items()
                      if name == REFERENCE_LEGEND), None)
    ordered = ([reference] if reference in gaits else []) + \
              [g for g in gaits if g != reference]

    for gait in ordered:
        values = [per_config[cfg].get(gait, np.nan) for cfg in configs]
        values += values[:1]
        name, color = style[gait]
        if gait == reference:
            ax.plot(angles, values, "-", color=color, lw=3.4, label=name,
                    marker="o", ms=9, zorder=6)
        else:
            ax.plot(angles, values, "-", color=color, lw=2.4, label=name,
                    marker="o", ms=7)
            ax.fill(angles, values, color=color, alpha=0.13)

    ax.plot(angles, [CHANCE] * len(angles), "--", color="gray", lw=1.0, alpha=0.6)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label_for(*cfg) for cfg in configs], fontsize=16)
    ax.set_ylim(0.45, 0.75)
    ticks = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=FS_TICK)
    ax.grid(True, alpha=0.4)
    ax.spines["polar"].set_alpha(0.4)
    ax.set_title("AUROC by gait", fontsize=FS_PANEL_TITLE, weight="bold",
                 pad=24, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.30, 1.05), fontsize=FS_LEGEND,
              title="Gait cohort", title_fontsize=16)


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def render() -> Path:
    auroc, meta = load_inputs()
    gaits = gait_order(auroc, meta)
    style = gait_style(gaits, meta)
    table = reported(auroc)
    if not table:
        raise SystemExit(
            f"No {REPORTED_REDUCER!r}-reducer rows in auroc.csv -- panels A and "
            "B report that model, so there is nothing to draw. Re-run the "
            f"derivation: {BUILD_CMD}")

    fig = plt.figure(figsize=(20, 14))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[0.85, 1.20], width_ratios=[1.0, 1.55],
        # Panel A's rotated two-line tick labels hang well below its axes; this
        # much vertical space is what keeps them off panel C's title.
        hspace=0.52, wspace=0.18,
        left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_bars = fig.add_subplot(grid[0, :])
    ax_grid = fig.add_subplot(grid[1, 0])
    ax_radar = fig.add_subplot(grid[1, 1], projection="polar")
    # Shift the polar axes left so its right-hand spoke labels clear the legend.
    box = ax_radar.get_position()
    ax_radar.set_position([box.x0 - 0.06, box.y0, box.width, box.height])

    draw_ranked_bars(ax_bars, table, gaits)
    draw_feature_grid(fig, ax_grid, table)
    draw_gait_radar(ax_radar, auroc, gaits, style)

    panels.letter_axes([ax_bars, ax_grid, ax_radar], fontsize=24, offset=(-46, 8))

    out_path = results.output_dir() / f"{FIGURE}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    auroc, meta = load_inputs()
    counts = auroc.groupby("gait")["n_knees"].max()
    print(f"{meta['cells_computed']} of {meta['cells_expected']} cells, "
          f"{meta['n_gaits']} gait conditions, "
          f"n = {counts.min()}-{counts.max()} knees per condition")
    print(f"  wrote {render()}")


if __name__ == "__main__":
    main()
