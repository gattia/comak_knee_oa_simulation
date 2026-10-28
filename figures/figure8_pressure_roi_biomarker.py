#!/usr/bin/env python3
"""Figure 8 -- a contact-pressure region of interest as a thinning biomarker.

    python figures/figure8_pressure_roi_biomarker.py

Runs with no arguments, reads ``04_results/figure8_pressure_roi_biomarker/`` and
writes one image per bone into ``figures/output/[<run>/]``:

    femur  manuscript Figure 8                figure8_pressure_roi_biomarker.png
    tibia  manuscript Extended Data Figure 3  figure_ed3_pressure_roi_biomarker_tibia.png

The figure
----------
Three columns, comparing a region chosen by each knee's own simulated contact
pressure against a standard anatomical region.

A  Mean cartilage thickness against time, one line per group, solid for the
   pressure region and dashed for the standard one. Shaded bands are 95%
   intervals of the mean.
B  Responsiveness -- absolute standardized response mean, at 24 and 48 months.
C  Discrimination -- absolute Cohen's d of thinning, each disease group against
   healthy, at 24 and 48 months.

In B and C a filled dot is the pressure region and an open dot the standard one,
joined by an arrow from standard to pressure region where they do not touch.
Above the pair is the difference of the two absolute values (pressure region
minus standard, positive when the pressure region is larger) with the
significance of the paired test from the derivation; the test is paired because
both regions are measured on the same knee.

Panel A shows the thickness trajectories; B and C compare the two regions on
responsiveness and group discrimination.

Comparators
-----------
The femur's and tibia's standard regions are not the same kind of region. The
femur's is a curated weight-bearing sub-region; the tibia has no weight-bearing
sub-label, so its comparator is the whole medial plateau and the pressure region
is a strict subset of it. The tibia's larger apparent gain follows partly from
that. Which comparator was used is read from ``meta_{bone}.json`` and printed on
the figure rather than assumed here.

The pressure region is the high-contact subset of the medial compartment, and
on the femur it is smaller than the standard comparator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import palette, results  # noqa: E402

FIGURE = "figure8_pressure_roi_biomarker"
# The femur is Figure 8; the tibia is Extended Data Figure 3.
OUTPUT_NAMES = {"femur": "figure8_pressure_roi_biomarker.png",
                "tibia": "figure_ed3_pressure_roi_biomarker_tibia.png"}
BONES = ("femur", "tibia")

BUILD_CMD = "python pipeline/derive_figure8_data.py"

# Group order, and the months plotted in panel A. Month 0 is the baseline every
# change is measured from, so it is exactly zero by construction rather than a
# fitted point.
GROUP_KEYS = ("healthy", "progressor", "kl2")
DISEASE_KEYS = ("progressor", "kl2")
MONTHS = (0, 24, 48)
INTERVALS = (("24mo", "24 months"), ("48mo", "48 months"))

ROI_LABEL = "Pressure-based ROI"

FS_HEADER = 18   # column headers
FS_SUB = 14      # header second line
FS_MONTH = 16    # per-row month titles
FS_AXLAB = 16
FS_TICK = 15
FS_ANN = 14      # the region-difference annotations
FS_LEGEND = 14


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_inputs(bone: str):
    results.require(
        FIGURE,
        f"responsiveness_{bone}.csv", f"discrimination_{bone}.csv",
        f"paired_srm_{bone}.csv", f"meta_{bone}.json",
        produced_by=BUILD_CMD,
    )
    base = results.figure_dir(FIGURE)
    return (
        pd.read_csv(base / f"responsiveness_{bone}.csv"),
        pd.read_csv(base / f"discrimination_{bone}.csv"),
        pd.read_csv(base / f"paired_srm_{bone}.csv"),
        json.loads((base / f"meta_{bone}.json").read_text()),
    )


def significance(p) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def annotation_gap(ax, xlim, area=135, pad_points=6.0):
    """Data-space offset from the widest dot to its annotation -- measured.

    The offset is derived from the axes geometry, so the text clears the dot's
    edge by ``pad_points`` whatever the figure size, column ratios or marker
    size. ``area`` is the scatter
    area in points squared, so the radius is ``sqrt(area / pi)`` points.
    """
    width_points = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    return (np.sqrt(area / np.pi) + pad_points) / width_points * xlim


def draw_trajectory(ax, responsiveness, static_label):
    """Panel A: thickness change against time, both regions, all three groups."""
    def value(region, interval, group, column):
        row = responsiveness[
            (responsiveness.region == region) & (responsiveness.interval == interval)
            & (responsiveness.group == group)
        ]
        return float(row[column].iloc[0]) if len(row) else np.nan

    for group in GROUP_KEYS:
        color = palette.GROUP_COLORS_SHORT_KEY[group]
        for region, style, width, alpha, marker, band in (
            ("roi", "-", 2.8, 1.0, "o", True),
            ("static", "--", 1.6, 0.7, None, False),
        ):
            means = [0.0] + [value(region, iv, group, "mean_um") for iv, _ in INTERVALS]
            ax.plot(MONTHS, means, ls=style, lw=width, color=color, alpha=alpha,
                    marker=marker, ms=6, zorder=3)
            if band:
                half = [0.0] + [
                    1.96 * value(region, iv, group, "sd_um")
                    / np.sqrt(value(region, iv, group, "n"))
                    for iv, _ in INTERVALS
                ]
                ax.fill_between(MONTHS,
                                [m - h for m, h in zip(means, half)],
                                [m + h for m, h in zip(means, half)],
                                color=color, alpha=0.12, zorder=1)

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(MONTHS)
    ax.set_xlabel("Months from baseline", fontsize=FS_AXLAB)
    ax.set_ylabel(f"Mean {palette.OUTCOME_LABEL.lower()} (µm)", fontsize=FS_AXLAB)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25)

    # Both keys live here because they only annotate this panel's curves.
    handles = [mlines.Line2D([], [], color=palette.GROUP_COLORS_SHORT_KEY[g], lw=3.4,
                             label=palette.GROUP_LABELS_SHORT_KEY[g])
               for g in GROUP_KEYS]
    handles += [
        mlines.Line2D([], [], color="0.4", lw=2.8, ls="-", marker="o", label=ROI_LABEL),
        mlines.Line2D([], [], color="0.4", lw=1.6, ls="--", label=static_label),
        mpatches.Patch(color="0.55", alpha=0.25, label="Shaded: mean ± 95% CI"),
    ]
    ax.legend(handles=handles, fontsize=FS_LEGEND - 2, loc="lower left", framealpha=0.92)


def draw_dot_rows(axes, rows_by_interval, xlabel, row_label="{label}"):
    """Panels B and C: one row per group, open dot standard, filled dot ROI.

    ``rows_by_interval`` maps an interval key to
    ``[(group, standard, roi, annotation), ...]`` in the order to draw them.
    """
    widest = max(abs(v) for rows in rows_by_interval.values()
                 for _, standard, roi, _ in rows for v in (standard, roi))
    # The axis ends just past the widest dot, at the next 0.1: the effect sizes
    # never reach 1, and the unused range would make the standard-vs-pressure
    # gap unreadable. The annotation sits above the pair, not in a column to
    # the right.
    xlim = np.ceil((widest + 0.06) * 10) / 10

    for ax, (interval, title) in zip(axes, INTERVALS):
        rows = rows_by_interval[interval]
        ticks, labels = [], []
        for i, (group, standard, roi, annotation) in enumerate(rows):
            y = len(rows) - 1 - i
            color = palette.GROUP_COLORS_SHORT_KEY[group]
            ax.scatter([standard], [y], facecolor="white", edgecolor=color,
                       s=105, lw=2.3, zorder=3)
            ax.scatter([roi], [y], facecolor=color, edgecolor="k",
                       s=135, lw=1.0, zorder=4)
            # An arrow from the standard region's dot to the pressure region's,
            # edge to edge, so the direction of the difference reads at a
            # glance. Omitted when the dots touch.
            r_std = annotation_gap(ax, xlim, area=105, pad_points=1.0)
            r_roi = annotation_gap(ax, xlim, area=135, pad_points=1.0)
            direction = np.sign(roi - standard)
            tail, head = standard + direction * r_std, roi - direction * r_roi
            if direction != 0 and (head - tail) * direction > 0.012 * xlim:
                ax.annotate("", xy=(head, y), xytext=(tail, y),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                            mutation_scale=13, shrinkA=0, shrinkB=0),
                            zorder=2)
            ax.text((standard + roi) / 2, y + 0.30, annotation, va="bottom", ha="center",
                    fontsize=FS_ANN, color=color, weight="bold")
            ticks.append(y)
            labels.append(row_label.format(label=palette.GROUP_LABELS_SHORT_KEY[group]))
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=FS_TICK)
        for tick, (group, *_) in zip(ax.get_yticklabels(), rows):
            tick.set_color(palette.GROUP_COLORS_SHORT_KEY[group])
            tick.set_fontweight("bold")
        ax.set_xlim(0, xlim)
        ax.set_ylim(-0.6, len(rows) - 0.2)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(title, fontsize=FS_MONTH, color="0.20", weight="bold")
    axes[-1].set_xlabel(xlabel, fontsize=FS_AXLAB)


def responsiveness_rows(responsiveness, paired):
    """Absolute SRM per group, both regions, with |ROI| - |standard| and the
    paired test's significance."""
    def srm(region, interval, group):
        row = responsiveness[
            (responsiveness.region == region) & (responsiveness.interval == interval)
            & (responsiveness.group == group)
        ]
        return abs(float(row.srm.iloc[0])) if len(row) else np.nan

    out = {}
    for interval, _ in INTERVALS:
        rows = []
        for group in GROUP_KEYS:
            match = paired[(paired.interval == interval) & (paired.group == group)]
            delta = float(match.delta_srm.iloc[0]) if len(match) else np.nan
            p = float(match.p_difference.iloc[0]) if len(match) else np.nan
            roi, standard = srm("roi", interval, group), srm("static", interval, group)
            # Annotated as the gain in magnitude, so a positive number always
            # means the pressure region did better -- the SRMs themselves are
            # negative, where "better" reads as more negative.
            rows.append((group, standard, roi, f"{roi - standard:+.2f} {significance(p)}"))
        out[interval] = rows
    return out


def discrimination_rows(discrimination):
    out = {}
    for interval, _ in INTERVALS:
        rows = []
        for group in DISEASE_KEYS:
            match = discrimination[
                (discrimination.interval == interval)
                & (discrimination.contrast == f"{group}_vs_healthy")
            ]
            if not len(match):
                continue
            roi = abs(float(match.d_roi.iloc[0]))
            standard = abs(float(match.d_static.iloc[0]))
            p = float(match.p_difference.iloc[0])
            rows.append((group, standard, roi, f"{roi - standard:+.2f} {significance(p)}"))
        out[interval] = rows
    return out


def render(bone: str) -> Path:
    responsiveness, discrimination, paired, meta = load_inputs(bone)
    static_label = meta["static_region"]

    fig = plt.figure(figsize=(18.0, 8.4))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.32, 1.0, 1.04],
                            left=0.065, right=0.965, top=0.805, bottom=0.155,
                            wspace=0.40, hspace=0.42)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]
    ax_c = [fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])]

    draw_trajectory(ax_a, responsiveness, static_label)
    draw_dot_rows(ax_b, responsiveness_rows(responsiveness, paired),
                  "|SRM|   (→ more responsive)")
    draw_dot_rows(ax_c, discrimination_rows(discrimination),
                  "|Cohen's d|   (→ better separation)", row_label="{label}\nvs Healthy")

    # Column headers rather than corner panel letters: B and C each span two
    # stacked axes, so a letter on the upper one would read as labelling that row
    # instead of the column, and the header carries the metric's name with it.
    def centre(ax):
        box = ax.get_position()
        return (box.x0 + box.x1) / 2

    fig.text(centre(ax_a), 0.864, "A  Longitudinal Cartilage Thickness",
             ha="center", fontsize=FS_HEADER, weight="bold")
    for ax, letter, title, subtitle in (
        (ax_b[0], "B", "Responsiveness", "Standardized Response Mean"),
        (ax_c[0], "C", "Discrimination", "Thickness change: group vs Healthy"),
    ):
        fig.text(centre(ax), 0.880, f"{letter}  {title}", ha="center",
                 fontsize=FS_HEADER, weight="bold")
        fig.text(centre(ax), 0.847, subtitle, ha="center", fontsize=FS_SUB,
                 color="0.35")

    key_x = (ax_b[0].get_position().x0 + ax_c[1].get_position().x1) / 2
    fig.text(key_x, 0.058, f"○  {static_label}            ●  {ROI_LABEL}",
             ha="center", va="center", fontsize=FS_LEGEND + 3, color="0.2")
    fig.text(key_x, 0.020,
             f"±x.xx = {ROI_LABEL} − Standard (paired Δ;  + = ROI better)"
             "          * p<0.05    ** p<0.01    *** p<0.001",
             ha="center", va="center", fontsize=FS_LEGEND + 3, color="0.2")

    suffix = "" if bone == "femur" else f"  ({bone})"
    fig.suptitle(
        f"Pressure-based ROI as a longitudinal cartilage-thinning biomarker{suffix}",
        fontsize=18, weight="bold", y=0.965)

    out_path = results.output_dir() / OUTPUT_NAMES[bone]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    for bone in BONES:
        print(f"rendering {bone} ...")
        print(f"  wrote {render(bone)}")


if __name__ == "__main__":
    main()
