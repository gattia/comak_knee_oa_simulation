#!/usr/bin/env python3
"""Figure 2 -- medial tibiofemoral contact mechanics through stance, by disease group.

    python figures/figure2_compartment_mechanics.py

Runs with no arguments, reads ``04_results/figure2_compartment_mechanics/``
and writes manuscript Figure 2 (the tibia) as
``figures/output/[<run>/]figure2_compartment_mechanics.png``.

The figure
----------
Two bands, three columns (mean pressure · 95th-percentile pressure · contact
area):

A-C  the ensemble waveform through stance, group mean and 95% confidence
     interval of the mean (1.96 SD / sqrt(n)),
     with SPM 1D significance bars beneath: the three-group omnibus ANOVA
     first, then each group pair.
D-F  the same three metrics as scalars, read at each knee's own early- and
     late-stance pressure peaks, with every pairwise contrast bracketed.

Panels A-C compare the groups through stance; D-F compare them at the two
pressure peaks.

Why the ensemble, and why all ten conditions
--------------------------------------------
Each knee is simulated under ten walking conditions, so a knee x gait row is a
repeated measure. The panel averages each knee across every condition it
survived in -- the healthy reference gait and the nine measured ones, one
curve per knee -- so the unit of replication is the knee and the group
comparison is over independent observations.

The reference condition is in that average. It carries a higher load than the
measured gaits, but that load is common to every knee, so it shifts one
condition's level rather than a between-group contrast -- the same reason the
peak-scalar mixed model (``figure2_mixed_model.R``) pools all ten with gait as
a factor. Each condition's own SPM windows are in ``per_gait_spm_{bone}.json``
and each condition against the ensemble of the others in
``cross_gait_similarity_{bone}.json``.

The per-knee ensemble curves, the SPM clusters, the scalars and their p-values
are computed by ``pipeline/derive_figure2_data.py``. This script computes only
the group mean and 95% confidence interval of each waveform, and skips a group
with fewer than five curves.

Layout
------
Each column stacks a waveform with its own stance axis, a framed significance
raster beneath it (one row per test, omnibus on top, row-labelled, black bars),
then the scalar panel, which repeats the waveform title unbolded. Only
significant contrasts are bracketed, peak-1 brackets above their points and
peak-2 brackets below theirs, so the y-axis follows the data. One footer line
holds the groups and the star key.

A gait-cycle key sits above the panels: walking silhouettes at heel strike,
midstance and toe-off, labelled with the percent of stance each marks. The
image is ``figures/assets/gait_stance_silhouettes.png``, five stance frames
keyed to transparency from an AI-generated green-screen image; the label
positions are the 1st, 3rd and 5th silhouettes' head x-positions in that image.
"""

from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea  # noqa: E402  # noqa: E402
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import panels, palette, results  # noqa: E402

FIGURE = "figure2_compartment_mechanics"
BONE = "tibia"
N_GRID = 101

BUILD_CMD = "python pipeline/derive_figure2_data.py"

# The three plotted metrics. p95 rather than the plain maximum: the maximum is a
# single triangle of the contact mesh, and its frame-to-frame noise swamps a
# group difference that the 95th percentile shows cleanly.
COLUMNS = [
    ("mean_pressure", "Mean pressure (MPa)", 1e-6),
    ("p95_pressure", "Peak pressure, 95$^{\\mathrm{th}}$ percentile (MPa)", 1e-6),
    ("contact_area", "Contact area (mm²)", 1e6),
]

# The columns are narrow (the row labels widen the left margin), and the one
# long title would otherwise run under the panel letters.
STRIP_TITLES = {
    "p95_pressure": "Peak pressure,\n95$^{\\mathrm{th}}$ percentile (MPa)",
}

# The gait-cycle key: image, the head x-position (fraction of image width) of
# each silhouette for its label, and the figure height (inches) reserved for it.
STANCE_KEY_IMAGE = _REPO / "figures" / "assets" / "gait_stance_silhouettes.png"
# Five frames of stance (heel strike, loading, midstance, terminal stance,
# toe-off), keyed to transparency from an AI-generated green-screen image; the
# label x-positions are the 1st, 3rd and 5th heads as fractions of width.
STANCE_KEY_LABELS = (("Heel strike\n0%", 120 / 1847),
                     ("Midstance\n50%", 938 / 1847),
                     ("Toe-off\n100%", 1748 / 1847))
STANCE_KEY_IMAGE_IN = 2.4      # height of the silhouettes
STANCE_KEY_LABEL_IN = 0.6      # room for the two-line labels above them
# Clearance between the panel titles and the silhouettes (panel B has a
# two-line title).
STANCE_KEY_GAP_IN = 0.95

PAIRS = list(palette.SPM_PAIR_COLORS)

FS_TITLE = 16
FS_LABEL = 15
FS_TICK = 13
FS_STAR = 13
FS_LEGEND = 13
FS_SUPTITLE = 13
FS_LETTER = 22

# Bracket rows in the scalar panels: the two adjacent contrasts sit low and
# side by side, the spanning one above them.
BRACKET_LEVEL = {PAIRS[0]: 0, PAIRS[2]: 0, PAIRS[1]: 1}
PEAK_X = {1: 0.0, 2: 1.4}
GROUP_DODGE = dict(zip(palette.GROUP_KEYS, (-0.26, 0.0, 0.26)))


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_inputs(bone: str):
    """Everything the figure draws, or a message naming the command that builds it."""
    results.require(
        FIGURE,
        f"ensemble_{bone}.npz", f"ensemble_spm_{bone}.json",
        f"composite_scalar_stats_{bone}.csv", f"cross_gait_similarity_{bone}.json",
        produced_by=BUILD_CMD,
    )
    base = results.figure_dir(FIGURE)

    raw = np.load(base / f"ensemble_{bone}.npz", allow_pickle=True)
    ensemble = {}
    for group in palette.GROUP_KEYS:
        curves = {metric: raw[f"{group}__{metric}"]
                  for metric, _, _ in COLUMNS if f"{group}__{metric}" in raw}
        if curves:
            ensemble[group] = curves

    spm = json.loads((base / f"ensemble_spm_{bone}.json").read_text(encoding="utf-8"))
    with (base / f"composite_scalar_stats_{bone}.csv").open(encoding="utf-8") as handle:
        stats = list(csv.DictReader(handle))
    similarity = json.loads((base / f"cross_gait_similarity_{bone}.json").read_text(encoding="utf-8"))
    return ensemble, spm, stats, similarity


def cluster_mask(clusters) -> np.ndarray:
    """``[[start, end], ...]`` percent-of-stance back to a per-frame mask."""
    mask = np.zeros(N_GRID, bool)
    for low, high in clusters:
        mask[low:high + 1] = True
    return mask


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _runs(mask):
    """Contiguous True runs of a per-frame mask, as inclusive (start, end) indices."""
    runs, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _bar_masks(spm, metric):
    """The four significance rows as per-frame masks, omnibus first.

    Omnibus-first so a reader checks "is there any group effect here at all"
    before reading which pair carries it.
    """
    masks = [cluster_mask(spm["anova_clusters_pct_stance"][metric])]
    masks += [cluster_mask(spm["sig_clusters_pct_stance"][metric][f"{a}|{b}"])
              for a, b in PAIRS]
    return masks


STRIP_BAR_COLOR = "#222222"


def _row_label_pieces():
    """Row labels for the strip as ``[(text, colour), ...]`` per row, omnibus
    first: the group names carry the group colours, the rest is neutral."""
    short, colours = palette.GROUP_LABELS, palette.GROUP_COLORS
    neutral = "#333333"
    rows = [[("ANOVA", neutral)]]
    rows += [[(short[a], colours[a]), (" vs ", neutral), (short[b], colours[b])] for a, b in PAIRS]
    return rows


def _coloured_label(ax, y, pieces, fontsize, pad_points=6):
    """A y-axis label made of differently coloured runs of text, right-aligned
    just outside the axes at data height ``y``."""
    children = [TextArea(text, textprops=dict(color=color, fontsize=fontsize))
                for text, color in pieces]
    box = HPacker(children=children, align="baseline", pad=0, sep=0)
    ab = AnnotationBbox(box, (0, y), xycoords=("axes fraction", "data"),
                        xybox=(-pad_points, 0), boxcoords="offset points",
                        box_alignment=(1.0, 0.5), frameon=False, pad=0)
    ax.add_artist(ab)


def draw_strip(ax, spm, metric, label_rows: bool):
    """The significance bars in a framed raster of their own.

    One row per test, omnibus on top, each bar one supra-threshold cluster.
    Taking the bars out of the waveform axes leaves those axes' y-range to the
    data and gives the significance results a box a reader can scan on its
    own. The box shares the waveform's x-axis and sits under its tick labels,
    so it carries no stance
    axis of its own -- only the matching vertical grid. Row labels are drawn on
    the first column only; the colours repeat them in the other two.
    """
    masks = _bar_masks(spm, metric)
    n = len(masks)
    t = np.linspace(0, 100, N_GRID)
    for i, mask in enumerate(masks):
        y = n - 1 - i
        for start, end in _runs(mask):
            ax.plot([t[start], t[end]], [y, y], color=STRIP_BAR_COLOR, lw=7,
                    solid_capstyle="butt")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_yticks(range(n))
    ax.set_yticklabels([])
    if label_rows:
        for i, pieces in enumerate(_row_label_pieces()):
            _coloured_label(ax, n - 1 - i, pieces, fontsize=FS_TICK - 2)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("#f4f4f4")


def draw_waveforms(axes, ensemble):
    """Group mean ± 95 % CI per column.

    The stance axis stays on these panels; the significance raster beneath
    shares it rather than repeating it.
    """
    t = np.linspace(0, 100, N_GRID)
    for ax, (metric, title, scale) in zip(axes, COLUMNS):
        for group, label, color in palette.group_tuples():
            curves = ensemble.get(group, {}).get(metric)
            if curves is None or len(curves) < 5:
                continue
            shown = curves * scale
            mean = shown.mean(axis=0)
            ci = 1.96 * shown.std(axis=0) / np.sqrt(len(shown))
            ax.plot(t, mean, color=color, lw=2.2, label=f"{label} (n={len(shown)})")
            ax.fill_between(t, mean - ci, mean + ci, color=color, alpha=0.15, lw=0)
        ax.set_title(STRIP_TITLES.get(metric, title), fontsize=FS_TITLE, weight="bold")
        ax.set_xlabel("% stance", fontsize=FS_LABEL)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Ensemble waveform\n(mean ± 95% CI)", fontsize=FS_LABEL)


def _bracket(ax, x1, x2, y, label, span, above: bool):
    """One significance bracket. ``above``: ticks hang down and the stars sit on
    top; otherwise the bracket is flipped and the stars sit underneath."""
    tick = 0.02 * span * (1 if above else -1)
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="#333", lw=1.8, zorder=4)
    ax.text((x1 + x2) / 2, y + (0.005 if above else -0.005) * span, label,
            ha="center", va="bottom" if above else "top", fontsize=FS_STAR,
            weight="bold", color="#333")


def draw_scalars(axes, stats):
    """Group means at the two stance peaks, bracketing the significant contrasts.

    Only significant pairs are bracketed; the caption says so. Peak-1 brackets sit above the peak-1 points and peak-2
    brackets below the peak-2 points, i.e. in the band between the two peaks
    where the data leave room, so the y-axis can follow the data instead of
    reserving headroom -- the group differences read larger for it. Where a
    metric has no such band the brackets extend the range.
    """
    by_cell = {}
    for row in stats:
        by_cell.setdefault((row["metric"], int(row["peak"])), []).append(row)

    for ax, (metric, _, _) in zip(axes, COLUMNS):
        x_of, hi_of, lo_of = {}, {}, {}
        highest, lowest = -np.inf, np.inf
        for peak in (1, 2):
            rows = by_cell.get((metric, peak), [])
            means = {}
            for row in rows:
                a, b = row["pair"].split("|")
                means[a] = (float(row["mean_a"]), float(row["ci95_a"]))
                means[b] = (float(row["mean_b"]), float(row["ci95_b"]))
            for group, _, color in palette.group_tuples():
                if group not in means:
                    continue
                mean, ci = means[group]
                x = PEAK_X[peak] + GROUP_DODGE[group]
                ax.errorbar(x, mean, yerr=ci, fmt="o", ms=8, color=color, capsize=4,
                            elinewidth=2.0, zorder=3)
                x_of[(peak, group)] = x
                hi_of[(peak, group)] = mean + ci
                lo_of[(peak, group)] = mean - ci
                highest = max(highest, mean + ci)
                lowest = min(lowest, mean - ci)

        span = max(highest - lowest, 1e-9)
        y_min, y_max = lowest, highest
        for peak in (1, 2):
            adjusted = {tuple(r["pair"].split("|")): float(r["p_holm"])
                        for r in by_cell.get((metric, peak), [])}
            drawn = [pair for pair in PAIRS
                     if pair in adjusted and adjusted[pair] < 0.05 and (peak, pair[0]) in x_of]
            if not drawn:
                continue
            # Adjacent pairs share the first level; the spanning pair goes one
            # level further out only when an adjacent pair is drawn under it.
            adjacent_drawn = any(BRACKET_LEVEL[p] == 0 for p in drawn)
            above = peak == 1
            peak_hi = max(v for (pk, _), v in hi_of.items() if pk == peak)
            peak_lo = min(v for (pk, _), v in lo_of.items() if pk == peak)
            base = peak_hi + 0.07 * span if above else peak_lo - 0.07 * span
            for pair in drawn:
                level = BRACKET_LEVEL[pair] if adjacent_drawn else 0
                offset = (0.03 + level * 0.11) * span
                y = base + offset if above else base - offset
                _bracket(ax, x_of[(peak, pair[0])], x_of[(peak, pair[1])], y,
                         stars(adjusted[pair]), span, above)
                if above:
                    y_max = max(y_max, y + 0.09 * span)
                else:
                    y_min = min(y_min, y - 0.09 * span)

        ax.set_xticks([PEAK_X[1], PEAK_X[2]])
        ax.set_xticklabels(["Peak 1\n(early stance)", "Peak 2\n(late stance)"],
                           fontsize=FS_TICK)
        ax.set_xlim(PEAK_X[1] - 0.55, PEAK_X[2] + 0.55)
        margin = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Ensemble point estimates\n(mean ± 95% CI)", fontsize=FS_LABEL)


# The footer shares one line with the group legend, so the key stays short;
# that only significant contrasts are bracketed is stated in the caption.
STAR_KEY_COMPACT = "* p<0.05    ** p<0.01    *** p<0.001"


def _group_entries(ensemble):
    """Legend handles and labels for the three groups, with their ensemble n."""
    counts = {g: len(ensemble.get(g, {}).get("mean_pressure", [])) for g in palette.GROUP_KEYS}
    handles = [plt.Line2D([0], [0], color=c, lw=4) for _, _, c in palette.group_tuples()]
    labels = [f"{lab[:-1]}; n={counts[key]})" for key, lab, _ in
              palette.group_tuples(labels=palette.GROUP_LABELS_LONG)]
    return handles, labels


def draw_footer_line(fig, ensemble, y: float, left: float, right: float):
    """The one-line footer.

    The three groups, horizontal, start at the panels' left edge; the star key
    for the scalar brackets ends at their right edge. The raster rows are
    labelled, so the significance colours need no legend here.
    """
    handles, labels = _group_entries(ensemble)
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(left, y),
               ncol=len(handles), fontsize=FS_LEGEND - 1, frameon=False,
               columnspacing=1.2, handlelength=2.0, handletextpad=0.6,
               borderaxespad=0.0, borderpad=0.0)
    fig.text(right, y, STAR_KEY_COMPACT, ha="right", va="bottom", fontsize=FS_LEGEND - 1)


def draw_stance_key(fig, x_center: float, y_bottom: float, height: float) -> None:
    """The gait-cycle key: silhouettes at heel strike / midstance / toe-off with
    the percent of stance each marks written above its head. ``y_bottom`` and
    ``height`` are figure fractions; the width follows the image's aspect."""
    image = mpimg.imread(str(STANCE_KEY_IMAGE))
    h_px, w_px = image.shape[:2]
    fig_w, fig_h = fig.get_size_inches()
    width = height * fig_h * (w_px / h_px) / fig_w
    ax = fig.add_axes([x_center - width / 2, y_bottom, width, height])
    ax.imshow(image)
    ax.axis("off")
    for text, x in STANCE_KEY_LABELS:
        ax.text(x, 1.03, text, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_LEGEND, weight="bold", linespacing=1.1)


def render(bone: str = BONE) -> Path:
    ensemble, spm, stats, _ = load_inputs(bone)
    coverage = spm["coverage"]

    gap = STANCE_KEY_GAP_IN
    key = gap + STANCE_KEY_IMAGE_IN + STANCE_KEY_LABEL_IN
    # Per column, top to bottom: waveform with its stance axis, the framed
    # significance raster under it (sharing that axis), then the scalar
    # panel. One footer line: groups on the left, star key on the right.
    left, right = 0.105, 0.985
    base_h = 10.8
    fig_h = base_h + key
    fig = plt.figure(figsize=(5.0 * len(COLUMNS) + 0.6, fig_h))
    panel_top = 1.0 - (0.145 * base_h + key) / fig_h
    panel_bottom = 0.095 * base_h / fig_h
    outer = fig.add_gridspec(2, 1, height_ratios=[1.20, 1.0],
                             left=left, right=right, top=panel_top, bottom=panel_bottom,
                             hspace=0.18)
    upper = outer[0].subgridspec(2, len(COLUMNS), height_ratios=[0.90, 0.30],
                                 hspace=0.60, wspace=0.26)
    lower = outer[1].subgridspec(1, len(COLUMNS), wspace=0.26)
    top = [fig.add_subplot(upper[0, c]) for c in range(len(COLUMNS))]
    strips = [fig.add_subplot(upper[1, c], sharex=top[c]) for c in range(len(COLUMNS))]
    bottom = [fig.add_subplot(lower[0, c]) for c in range(len(COLUMNS))]
    draw_waveforms(top, ensemble)
    for c, (ax, (metric, _, _)) in enumerate(zip(strips, COLUMNS)):
        draw_strip(ax, spm, metric, label_rows=(c == 0))
    # One header for the whole strip row: the three boxes are one family of
    # tests, so a title per box would say the same thing three times.
    s0, s2 = strips[0].get_position(fig), strips[-1].get_position(fig)
    fig.text((s0.x0 + s2.x1) / 2, s0.y1 + 0.006, "Significance testing (SPM 1D, p < 0.05)",
             ha="center", va="bottom", fontsize=FS_LABEL - 1, weight="bold")
    draw_footer_line(fig, ensemble, y=0.02, left=left, right=right)

    draw_scalars(bottom, stats)
    # The scalar panels repeat the waveform titles, unbolded: the columns read
    # as one metric each, top to bottom.
    for ax, (metric, title, _) in zip(bottom, COLUMNS):
        ax.set_title(STRIP_TITLES.get(metric, title), fontsize=FS_TITLE, weight="normal")

    # The gait-cycle key above the panel titles, its labels above it, the
    # suptitle above those.
    draw_stance_key(fig, x_center=(left + right) / 2,
                    y_bottom=panel_top + gap / fig_h, height=STANCE_KEY_IMAGE_IN / fig_h)

    index = panels.letter_axes(top, fontsize=FS_LETTER)
    panels.letter_axes(bottom, start=index, fontsize=FS_LETTER)

    counts = coverage["n_used"]
    per_group = " / ".join(str(counts[g]) for g in palette.GROUP_KEYS)
    names = " / ".join(palette.GROUP_LABELS[g] for g in palette.GROUP_KEYS)
    conditions = coverage["ensemble_conditions"]
    which = "reference + measured" if "reference" in conditions else "measured only"
    title = (f"{bone.capitalize()} — medial compartment (anatomical full condyle) · "
             f"per-knee ensemble of {len(conditions)} gait conditions ({which}; "
             f"n = {per_group} knees, {names}). "
             f"Top: per-knee ensemble waveforms with SPM 1D significant regions. "
             f"Bottom: scalar point estimates at the mean-pressure-peak time "
             f"(Peak 1 / Peak 2).")
    fig.suptitle(textwrap.fill(title, width=112), fontsize=FS_SUPTITLE,
                 weight="bold", y=0.985)

    out_path = results.output_dir() / f"{FIGURE}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    print(f"rendering {BONE} ...")
    print(f"  wrote {render()}")


if __name__ == "__main__":
    main()
