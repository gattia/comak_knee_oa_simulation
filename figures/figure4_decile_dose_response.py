#!/usr/bin/env python3
"""Figure 4 -- does cartilage thin faster where it is loaded harder?

    python figures/figure4_decile_dose_response.py

Runs with no arguments, reads ``04_results/figure4_decile_dose_response/`` and
writes manuscript Figure 4 as ``figures/output/[<run>/]figure4_decile_dose_response.png``.

The figure
----------
Each knee's cartilage is split into ten equal-area-ranked bins of its own contact
pressure, and the mean thickness change of each bin is plotted against the bin.
A downward line means the harder-loaded cartilage thinned faster.

Columns are anatomical compartments, rows are the two pressure summaries
(cumulative over stance, and peak). Lines are the mixed model's marginal means
with 95% intervals, gait conditions averaged out; the faint points behind them
are the observed cell means, so the reader can see the model is describing the
data rather than replacing it.

The lateral columns are a negative control: anatomically comparable surfaces
where the disease process is not concentrated, so a dose-response there as
strong as in the medial compartment would point to the decile construction
rather than mechanics.

The significance box reports the categorical model's three omnibus tests; the
continuous model's per-group slopes are not on the figure. The two can
disagree: a categorical omnibus can be significant while the continuous slope is
near zero, because ten means can differ slightly without trending. The slopes
are reported with the continuous model they come from.

Significance boxes
------------------
Each panel reports its three omnibus tests on one line, top-right, with the
short forms P = pressure percentile, G = group, P×G = interaction, which the
caption defines.

Output
------
``figure4_decile_dose_response.png`` (manuscript Figure 4): medial and lateral
femur, medial and lateral tibia, cumulative and peak pressure.

Inputs
------
Everything comes from ``04_results``. The outcome, its units, and the population
are read from the derivation's and the model's metadata -- including the per-group n in the legend, so a literal cannot go stale
when the cohort changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import palette, panels, results  # noqa: E402

FIGURE = "figure4_decile_dose_response"

BUILD_CMD = (
    "python pipeline/derive_figure4_data.py && "
    "Rscript figures/figure4_decile_dose_response.R"
)

# Trochlea is not shown: it is patellofemoral cartilage and carries no
# tibiofemoral load, so it does not belong beside the compartments that do.
COLUMNS_MAIN = [
    ("Medial femur", "femur", "medial"),
    ("Lateral femur", "femur", "lateral"),
    ("Medial tibia", "tibia", "medial"),
    ("Lateral tibia", "tibia", "lateral"),
]
ROWS = [("cum", "Cumulative pressure"), ("max", "Peak pressure")]

UNITS = {"thinning_pct": "% / yr", "thinning_mm": "mm / yr"}

FS_LETTER = 22
FS_TITLE = 16
FS_LABEL = 15
FS_TICK = 12.5
FS_LEGEND = 14
# The letter sits just above each panel's top-left corner, not out in the
# margin. This figure's rotated y-label is tall enough to reach the letter's
# height and wide enough to occupy the space a margin offset would use.
LETTER_OFFSET = (-4, 6)

GROUPS = palette.group_tuples(labels=palette.GROUP_LABELS_LONG)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_inputs():
    results.require(
        FIGURE,
        "observed_decile_means.csv", "group_counts.csv",
        "derivation_meta.json", "mixed_model_meta.json", "model_summary.csv",
        produced_by=BUILD_CMD,
    )
    base = results.figure_dir(FIGURE)
    return {
        "dir": base,
        "observed": pd.read_csv(base / "observed_decile_means.csv"),
        "counts": pd.read_csv(base / "group_counts.csv").set_index("group"),
        "derivation": json.loads((base / "derivation_meta.json").read_text()),
        "model": json.loads((base / "mixed_model_meta.json").read_text()),
    }


def marginal_means(base: Path, bone: str, region: str, metric: str):
    """The categorical model's fitted means, or None if that cell was not fitted."""
    path = base / "model" / f"emmeans_categorical_{bone}_{region}_{metric}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "decile" not in frame.columns:
        frame["decile"] = frame["decile_f"].astype(int)
    lower = "lower" if "lower" in frame.columns else "asymp.LCL"
    upper = "upper" if "upper" in frame.columns else "asymp.UCL"
    return frame.rename(columns={lower: "lower", upper: "upper"})


def omnibus_tests(base: Path, bone: str, region: str, metric: str):
    """The three categorical omnibus p-values, as display strings.

    Read from the model's own joint-tests table by term name. Parsing a printed
    log instead is one renamed term or one changed print width away from
    reporting the wrong number without an error.
    """
    path = base / "model" / f"joint_tests_categorical_{bone}_{region}_{metric}.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    wanted = {"decile_f": "percentile", "group": "group",
              "decile_f:group": "percentile×group"}
    return {
        label: format_p(frame.loc[frame.term == term, "p_value"].iloc[0])
        for term, label in wanted.items()
        if (frame.term == term).any()
    }


def format_p(value) -> str:
    """Three decimals, with a floor. Beyond that the digits are not meaningful."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"p={value}"
    return "p<0.001" if number < 1e-3 else f"p={number:.3f}"


def outcome_label(meta) -> str:
    outcome = meta["outcome"]
    return f"{palette.OUTCOME_LABEL} ({UNITS.get(outcome, outcome)})"


def percentile(decile) -> np.ndarray:
    """Decile index 1..10 -> the midpoint of the percentile band it summarises.

    Each point stands for a ten-percentile band, so placing it at the midpoint
    puts the markers *between* the axis ticks and reads as "this point represents
    that range" rather than "this point is at that percentile".
    """
    return (np.asarray(decile, dtype=float) - 0.5) * 10.0


# --------------------------------------------------------------------------- #
# Measured text placement                                                     #
# --------------------------------------------------------------------------- #
def _wrap_candidates(text: str) -> list[str]:
    """The text unwrapped, then split at its most balanced two-line break."""
    words = text.split()
    if len(words) < 2:
        return [text]
    target = len(text) / 2
    cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - target))
    return [text, " ".join(words[:cut]) + "\n" + " ".join(words[cut:])]


def fit_ylabel(ax, text, fontsize=FS_LABEL, min_fontsize=9.5):
    """A rotated y-label MEASURED to fit the axes height, wrapping if it must.

    Not simply ``set_ylabel(..., fontsize=15)``. At a fixed size a long outcome
    label such as "Cartilage thickness change" makes the two rows' labels collide
    mid-figure and clips the upper one off the canvas, with no error. A font
    size chosen by hand fits only one string, so this measures the drawn artist
    and shrinks only as far as the overflow requires.
    """
    figure = ax.figure

    def place(candidate, size):
        label = ax.set_ylabel(candidate, fontsize=size)
        figure.canvas.draw()          # window extents need a live renderer
        return label, label.get_window_extent().height

    available, label = None, None
    for candidate in _wrap_candidates(text):
        label, needed = place(candidate, fontsize)
        if available is None:
            available = ax.get_window_extent().height
        if needed <= available:
            return label
        # One measured correction rather than a search: scale by the
        # overflow ratio with a little slack, floored so it stays legible.
        label, needed = place(candidate, max(min_fontsize, fontsize * available / needed * 0.98))
        if needed <= available:
            return label
    return label


def place_row_label(ax, text, fontsize=FS_TITLE, gap=0.06):
    """A bold rotated row label, left of whatever the y-label actually occupies.

    Measured for the same reason as ``fit_ylabel``: a two-line y-label reaches
    further left than any fixed offset chosen against a one-line one.
    """
    figure = ax.figure
    figure.canvas.draw()
    box = ax.get_window_extent()
    label = ax.yaxis.label
    left = ((label.get_window_extent().x0 - box.x0) / box.width) if label.get_text() else -0.20
    return ax.annotate(text, xy=(left - gap, 0.5), xycoords="axes fraction",
                       rotation=90, va="center", ha="center",
                       fontsize=fontsize, weight="bold", color="#222")


# --------------------------------------------------------------------------- #
# Panels                                                                      #
# --------------------------------------------------------------------------- #
def draw_panel(ax, data, bone, region, metric, show_xlabel, show_ylabel):
    means = marginal_means(data["dir"], bone, region, metric)
    if means is None:
        ax.text(0.5, 0.5, "(model not fitted for this cell)", ha="center", va="center",
                color="gray", fontsize=13, transform=ax.transAxes)
        return

    outcome = data["model"]["outcome"]
    observed = data["observed"]
    observed = observed[
        (observed.bone == bone) & (observed.region == region)
        & (observed.pressure_metric == metric)
    ]

    for key, _label, color in GROUPS:
        fitted = means[means.group == key].sort_values("decile")
        if fitted.empty:
            continue
        x = percentile(fitted["decile"].to_numpy())
        ax.plot(x, fitted["emmean"].to_numpy(), "-", color=color, lw=2.2, zorder=3)
        ax.fill_between(x, fitted["lower"].to_numpy(), fitted["upper"].to_numpy(),
                        color=color, alpha=0.16, lw=0, zorder=1)
        points = observed[observed.group == key].sort_values("decile")
        if not points.empty:
            ax.plot(percentile(points["decile"].to_numpy()), points[outcome].to_numpy(),
                    "o", color=color, ms=3.6, alpha=0.55, zorder=2)

    ax.axhline(0, color="gray", ls="--", lw=0.7, alpha=0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.22)

    tests = omnibus_tests(data["dir"], bone, region, metric)
    if tests:
        # One line, same corner in every panel, every term with a short form
        # the caption defines: P = pressure percentile, G = group, P×G = their
        # interaction.
        short = {"percentile": "P", "group": "G", "percentile×group": "P×G"}
        text = ", ".join(f"{short.get(k, k)}: {v}" for k, v in tests.items())
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.10 * (hi - lo))      # headroom for the line
        ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=9.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.85))

    if show_xlabel:
        ax.set_xlabel("Pressure percentile", fontsize=FS_LABEL)
    if show_ylabel:
        fit_ylabel(ax, outcome_label(data["model"]))


def legend_handles(data):
    """Group lines labelled with the n the derivation recorded for each."""
    counts = data["counts"]
    return [
        plt.Line2D([0], [0], color=color, lw=2.4,
                   label=f"{label}; n={int(counts.loc[key, 'n_knees'])}")
        for key, label, color in GROUPS
        if key in counts.index
    ]


def share_row_ylim(axes):
    limits = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_ylim(min(lo for lo, _ in limits), max(hi for _, hi in limits))


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def render_grid(data, columns, out_name, suptitle) -> Path:
    n_rows, n_cols = len(ROWS), len(columns)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows),
                                squeeze=False)
    for row, (metric, row_label) in enumerate(ROWS):
        for col, (title, bone, region) in enumerate(columns):
            ax = axes[row][col]
            draw_panel(ax, data, bone, region, metric,
                       show_xlabel=(row == n_rows - 1), show_ylabel=(col == 0))
            if row == 0:
                ax.set_title(title, fontsize=FS_TITLE, weight="bold", pad=6)
            panels.add_panel_letter(ax, panels.LETTERS[row * n_cols + col],
                                    fontsize=FS_LETTER, offset=LETTER_OFFSET)
        place_row_label(axes[row][0], row_label)
        share_row_ylim(axes[row])

    figure.legend(handles=legend_handles(data), loc="lower center", ncol=3,
                  fontsize=FS_LEGEND, frameon=False, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(suptitle, fontsize=20, weight="bold", y=0.985)
    figure.tight_layout(rect=(0.045, 0.04, 1, 0.965))
    return save(figure, out_name)


def save(figure, name, dpi=160) -> Path:
    path = results.output_dir() / name
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


MAIN_TITLE = "Vertex-wise pressure → cartilage thickness change"


def main() -> None:
    data = load_inputs()
    derivation = data["derivation"]
    population = (f"{derivation['n_knees']} knees, "
                  f"{derivation['n_simulations']:,} simulations, "
                  f"{derivation['n_gaits']} gait conditions")
    print(f"Figure 4: {population}")
    path = render_grid(data, COLUMNS_MAIN, f"{FIGURE}.png", MAIN_TITLE)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
