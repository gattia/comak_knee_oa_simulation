#!/usr/bin/env python3
"""Figure 3 -- what changing only the shape of a knee does to its contact mechanics.

    python figures/figure3_shape_pressure.py

Runs with no arguments, reads ``04_results/figure3_shape_pressure/`` and writes
manuscript Figure 3 (the tibia) as ``figures/output/[<run>/]figure3_shape_pressure.png``.

The figure
----------
A healthy-to-mild-OA axis is fitted in the shape model's latent space. Synthetic
knees are decoded at steps along it and simulated under all nine measured gaits,
so gait, body mass and every other input are held fixed and anatomy is the only
thing that varies. Landscape, two halves:

left   (unlettered) the decoded anatomy at the two ends of the fitted window and
       at the mean -- femoral cartilage, tibial cartilage, tibial cartilage with
       the menisci -- so the reader can see the shape change driving the numbers.
       Arrows at both ends, which the caption refers to: blue double-headed
       across the medial femoral condyle (the larger cartilage surface), orange
       at the retracting medial meniscus.
right  a 3 x 3 grid of plots:
  A-C  medial compartment traces through stance, one line per shape step,
       coloured along the axis (colour bar beside C). Dashed lines mark where
       the early and late peaks of the plotted ensemble curves fall.
  D-F  the dose-response: marginal means with 95% intervals from the mixed
       model, at the early- and late-stance pressure peaks (legend beside F).
  G-I  geometric measurements of that same anatomy -- cartilage surface area,
       meniscal coverage, meniscal extrusion (legends beside I).

The rows show the medial contact mechanics (A-F) and the meniscal and
cartilage geometry (G-I) of the same synthetic knees along the axis.

Inputs and the sigma window
---------------------------
Everything comes from ``04_results``; no simulation tree, no GPU, no 3D
rendering at figure time -- the anatomy is the grey renders already in
``morph/``. The plotted sigma window is read from the model's
``mixed_model_meta.json``, so the panels show the range the model was fitted on.

Regenerate: ``python pipeline/derive_figure3_data.py && python
pipeline/derive_figure3_geometry.py && Rscript figures/figure3_mixed_model.R &&
python figures/figure3_shape_pressure.py``. The geometry step makes
``shape_metrics.csv`` (G-I) and the ``morph/*.png`` anatomy renders from the same
synthetic tree.
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
from matplotlib import cm, colors, ticker, transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from PIL import Image  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import panels, palette, results  # noqa: E402

FIGURE = "figure3_shape_pressure"
BONE = "tibia"
N_GRID = 101

BUILD_CMD = (
    "python pipeline/derive_figure3_data.py && "
    "Rscript figures/figure3_mixed_model.R"
)
# The two inputs BUILD_CMD does not write: the shape metrics and anatomy renders.
GEOMETRY_CMD = "python pipeline/derive_figure3_geometry.py"

# Canvas. The x-positions below are fractions of a 26-inch canvas; the extra two
# inches on the right hold the colour bar and the legends.
FIG_SIZE = (28.0, 11.5)
DPI = 140
SX = 26.0 / FIG_SIZE[0]

# p95 rather than the plain maximum: the maximum is a single triangle of the
# contact mesh and is visibly noisy frame to frame, which makes a real trend
# across shapes hard to see. Worded as in Figure 2; two lines fit the panel.
P95_TITLE = "Peak pressure,\n95$^{\\mathrm{th}}$ percentile (MPa)"

# The three plotted metrics; titles are shared by the waveform and dose rows.
COLUMNS = [
    ("mean_pressure", "Mean pressure (MPa)"),
    ("p95_pressure", P95_TITLE),
    ("contact_area", "Contact area (mm²)"),
]

END_LOW = "#1e3a5f"    # dark blue, healthy end
END_HIGH = "#B34700"   # dark orange, mild-OA end

# Bone colours for the area panel and the anatomy row labels. Strong, and kept
# clear of the group blue/orange, peak teal/purple and medial/lateral green/brown.
FEMUR_COLOR = "#263238"   # charcoal
TIBIA_COLOR = "#AD1457"   # deep magenta

# The anatomy block: three views down, three shape positions across.
MORPH_ROWS = [
    ("femur_cart", "Femoral\ncartilage", FEMUR_COLOR),
    ("tibia_cart", "Tibial\ncartilage", TIBIA_COLOR),
    ("tibia_men", "Tibial cartilage\n& menisci", TIBIA_COLOR),
]

# Arrows over the anatomy; the caption points the reader to them. Each entry
# is (structure, end, style,
# centre, angle, length): end is "lo" or "hi", the low or high sigma column of
# the plotted window (so the arrows follow the window rather than a fixed sigma);
# centre is in fractions of that cropped image (x from the left, y from the
# top); angle is in degrees, anticlockwise from the +x axis; length is tip to tip
# in points.
ARROWS = [
    ("femur_cart", "lo", "surface", (0.7917, 0.7396), 0.0, 69.9),
    ("femur_cart", "hi", "surface", (0.8008, 0.7396), 0.0, 85.7),
    ("tibia_men", "lo", "meniscus", (0.6518, 0.5011), -25.2, 51.7),
    ("tibia_men", "hi", "meniscus", (0.7145, 0.5862), -25.2, 52.2),
]
# Block-arrow shapes, in points, measured on the outline's centre line (lw is
# the outline width). Blue fill with a dark-blue outline for the surface
# arrows, orange fill with a dark-brown outline for the meniscus arrows.
ARROW_STYLES = {
    "surface": dict(double=True, shaft=13.2, head=27.8, head_length=13.9, lw=3.2,
                    face="#156082", edge="#042433"),
    "meniscus": dict(double=False, shaft=11.5, head=32.6, head_length=20.6, lw=3.5,
                     face="#E97132", edge="#622C0F"),
}

# Whether to overlay the three group-mean anchors on the waveform and
# dose-response panels. Off in the manuscript figure: the anchors span only
# -0.33 to +0.44 sigma, so all three land almost on top of each other in the
# middle of the plotted sigma window, adding three dashed lines and six markers that
# clutter the panels without separating. They stay in the data and are reported
# in the text. Set True to inspect them.
SHOW_ANCHORS = False

FS_TITLE = 18
FS_LABEL = 18
FS_TICK = 17
FS_LEGEND = 15
FS_LETTER = 20        # smaller than the titles, which it would otherwise outshout
LETTER_OFFSET = (-36, 6)
FS_MESH_HEADER = 22   # the Healthy / z-score / Mild OA caption and the sigma headers
FS_MESH_ROW = 20      # "Femoral cartilage" etc. beside the anatomy rows
FS_ORIENT = 21        # Lateral / Medial / Posterior around the anatomy


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_inputs(bone: str):
    """Everything the figure draws, or a message naming the command that builds it."""
    results.require(
        FIGURE,
        "shapes.json", f"waveforms_{bone}.npz",
        "mixed_model_emmeans.csv", "mixed_model_meta.json",
        produced_by=BUILD_CMD,
    )
    results.require(FIGURE, "shape_metrics.csv", produced_by=GEOMETRY_CMD)
    base = results.figure_dir(FIGURE)

    shapes = json.loads((base / "shapes.json").read_text())
    waveforms = np.load(base / f"waveforms_{bone}.npz", allow_pickle=True)
    emmeans = pd.read_csv(base / "mixed_model_emmeans.csv")
    meta = json.loads((base / "mixed_model_meta.json").read_text())
    metrics = pd.read_csv(base / "shape_metrics.csv")

    return shapes, waveforms, emmeans, meta, metrics


def morph_images(labels):
    """Paths of the anatomy renders, keyed (structure, shape label)."""
    wanted = {(structure, label): f"morph/{structure}__{label}.png"
              for structure, _, _ in MORPH_ROWS for label in labels}
    paths = results.require(FIGURE, *wanted.values(), produced_by=GEOMETRY_CMD)
    return dict(zip(wanted, paths))


def in_window(shape, meta) -> bool:
    """Sigma steps inside the fitted window; anchors are always inside it."""
    if shape["type"] != "sigma_step":
        return True
    return meta["sigma_min"] - 1e-9 <= shape["k_sigma"] <= meta["sigma_max"] + 1e-9


def gait_mean(waveforms, label, metric, region="medial"):
    """Mean over gaits, or None if no gait survived QC for this shape."""
    stack = waveforms[f"{label}__{region}__{metric}"]
    if stack.size == 0:
        return None
    return np.nanmean(stack, axis=0)


def ensemble_peak_times(waveforms, steps):
    """Early / late peak of each plotted ensemble curve of medial mean pressure.

    These are the peaks a reader sees in A, not the per-simulation sampling
    instants. One value per shape step; the figure draws their median (the
    steps agree to within 1-2 % of stance)."""
    half = N_GRID // 2
    early, late = [], []
    for shape in steps:
        curve = gait_mean(waveforms, shape["label"], "mean_pressure")
        if curve is None:
            continue
        early.append(int(np.nanargmax(curve[:half])))
        late.append(half + int(np.nanargmax(curve[half:])))
    return np.array(early), np.array(late)


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def sigma_xlabel(ax, end_size=14, dy=-58):
    """Two-line sigma axis label that stays inside the panel's own width:
    'z-score (σ)' centred, then coloured 'Healthy' / 'Mild OA' at the ends."""
    ax.set_xlabel("z-score (σ)", fontsize=FS_LABEL)
    ax.annotate("← Healthy", xy=(0, 0), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", ha="left", va="top", fontsize=end_size,
                weight="bold", color=END_LOW, annotation_clip=False)
    ax.annotate("Mild OA →", xy=(1, 0), xycoords="axes fraction", xytext=(0, dy),
                textcoords="offset points", ha="right", va="top", fontsize=end_size,
                weight="bold", color=END_HIGH, annotation_clip=False)


def crop_row(paths, pad=0.02, white=0.97):
    """Crop the three images of one anatomy row with one common bounding box.

    The three renders share a camera frame, so an identical crop keeps their
    relative sizes comparable while removing the letterboxing whitespace. The
    renders differ by a few pixels in width between sigma steps; all three are
    resampled to the first before cropping."""
    first = Image.open(str(paths[0])).convert("RGB")
    w, h = first.size
    imgs = [np.asarray(Image.open(str(p)).convert("RGB").resize((w, h), Image.BILINEAR),
                       dtype=np.float32) / 255.0
            for p in paths]
    rows = np.zeros(h, bool)
    cols = np.zeros(w, bool)
    for im in imgs:
        mask = (im < white).any(axis=2)
        rows |= mask.any(axis=1)
        cols |= mask.any(axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pr, pc = int(pad * h), int(pad * w)
    r0, r1 = max(r0 - pr, 0), min(r1 + pr, h - 1)
    c0, c1 = max(c0 - pc, 0), min(c1 + pc, w - 1)
    return [im[r0:r1 + 1, c0:c1 + 1] for im in imgs]


def block_arrow(length, style):
    """Outline of a block arrow lying along +x and centred on the origin, in points."""
    half, shaft, head = length / 2, style["shaft"] / 2, style["head"] / 2
    base = half - style["head_length"]
    outline = [(base, -shaft), (base, -head), (half, 0), (base, head), (base, shaft)]
    if style["double"]:
        outline += [(-base, shaft), (-base, head), (-half, 0), (-base, -head), (-base, -shaft)]
    else:
        outline += [(-half, shaft), (-half, -shaft)]
    return np.array(outline)


def draw_arrow(fig, ax, image_shape, centre, angle, length, style):
    """One annotation arrow on an anatomy image.

    The centre is pinned to the image, so the arrow stays on the anatomy it
    marks. The shape is laid out in points, so it keeps its size and angle
    however large the image is drawn."""
    h, w = image_shape[:2]
    x, y = centre[0] * w - 0.5, centre[1] * h - 0.5   # imshow pixel coordinates
    a = np.radians(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    outline = block_arrow(length, style) @ rotation.T / 72.0   # inches
    at_centre = fig.dpi_scale_trans + transforms.ScaledTranslation(x, y, ax.transData)
    ax.add_patch(Polygon(outline, closed=True, transform=at_centre, facecolor=style["face"],
                         edgecolor=style["edge"], lw=style["lw"], joinstyle="miter",
                         clip_on=False, zorder=3))


def draw_morph(fig, x0, x1, y_center, images, morph_shapes, max_stack, header_gap=0.045):
    """The decoded-anatomy block: 3 rows x 3 sigma positions.

    Cells are sized to the cropped images so the rows abut. The block
    is centred on ``y_center`` and, if it would be taller than ``max_stack``
    (figure fraction), shrunk about its horizontal centre. Left unlettered -- it
    is a labelled image strip, not a plot panel."""
    labels = [s["label"] for s in morph_shapes]
    cropped = {st: crop_row([images[(st, label)] for label in labels])
               for st, _, _ in MORPH_ROWS}
    aspects = [cropped[st][0].shape[0] / cropped[st][0].shape[1] for st, _, _ in MORPH_ROWS]

    fig_w, fig_h = fig.get_size_inches()
    stack = (x1 - x0) * fig_w / 3.0 * sum(aspects) / fig_h
    if stack > max_stack:
        scale = max_stack / stack
        xc, half = (x0 + x1) / 2, (x1 - x0) * scale / 2
        x0, x1 = xc - half, xc + half
        stack = max_stack
    y0, y1 = y_center - stack / 2, y_center + stack / 2
    grid = fig.add_gridspec(3, 3, left=x0, right=x1, top=y1, bottom=y0,
                            height_ratios=aspects, wspace=0.0, hspace=0.0)

    positions = [s["k_sigma"] for s in morph_shapes]
    sigma_label = {
        positions[0]: (f"−{abs(positions[0]):g}σ", END_LOW),
        0: ("Mean (0σ)", "#444444"),
        positions[-1]: (f"+{positions[-1]:g}σ", END_HIGH),
    }
    window_end = {"lo": positions[0], "hi": positions[-1]}
    for row, (structure, row_label, color) in enumerate(MORPH_ROWS):
        for col, shape in enumerate(morph_shapes):
            ax = fig.add_subplot(grid[row, col])
            ax.axis("off")
            image = cropped[structure][col]
            ax.imshow(image)
            for st, end, style, centre, angle, length in ARROWS:
                if st == structure and np.isclose(window_end[end], shape["k_sigma"]):
                    draw_arrow(fig, ax, image.shape, centre, angle, length,
                               ARROW_STYLES[style])
            if row == 0:
                text, c = sigma_label[shape["k_sigma"]]
                ax.text(0.5, 1.03, text, transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=FS_MESH_HEADER, weight="bold", color=c)
            if col == 0:
                ax.text(-0.04, 0.5, row_label, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", ma="center", fontsize=FS_MESH_ROW,
                        weight="bold", color=color, linespacing=1.0)

    # The axis caption above the sigma headers, and the anatomical orientation
    # once around the block rather than once per image.
    xc, col_w = (x0 + x1) / 2, (x1 - x0) / 3
    for dx, text, c in ((-col_w, "←  Healthy", END_LOW), (0, "z-score (σ)", "#444444"),
                        (col_w, "Mild OA  →", END_HIGH)):
        fig.text(xc + dx, y1 + header_gap + 0.035, text, ha="center", va="bottom",
                 fontsize=FS_MESH_HEADER, weight="bold", color=c)
    middle_y = (y0 + y1) / 2
    orient = dict(fontsize=FS_ORIENT, weight="bold", color="#333333")
    fig.text(x0 - 0.03, middle_y, "Lateral", rotation=90, ha="center", va="center", **orient)
    fig.text(x1 + 0.012, middle_y, "Medial", rotation=270, ha="center", va="center", **orient)
    fig.text(xc, y0 - 0.006, "Posterior", ha="center", va="top", **orient)


def draw_waveforms(axes, steps, anchors, waveforms, cmap, norm):
    for ax, (metric, title) in zip(axes, COLUMNS):
        for shape in steps:
            curve = gait_mean(waveforms, shape["label"], metric)
            if curve is None:
                continue
            ax.plot(np.linspace(0, 100, N_GRID), curve, lw=2,
                    color=cmap(norm(shape["k_sigma"])))
        for shape in anchors if SHOW_ANCHORS else ():
            curve = gait_mean(waveforms, shape["label"], metric)
            if curve is None:
                continue
            ax.plot(np.linspace(0, 100, N_GRID), curve, lw=2.4, ls="--",
                    color=palette.GROUP_COLORS[shape["group"]],
                    label=palette.GROUP_LABELS[shape["group"]])
        ax.set_title(title, fontsize=FS_TITLE, weight="bold")
        ax.set_xlabel("% stance", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Ensemble waveform\n(mean over gaits)", fontsize=FS_LABEL)


def draw_peak_lines(axes, early, late):
    """Dashed lines at the early / late peaks, in the D-F peak colours. D-F's
    legend names the colours, so there is no legend here."""
    for ax in axes:
        for times, peak in ((early, 1), (late, 2)):
            ax.axvline(np.median(times), color=palette.PEAK_COLORS[peak], ls="--",
                       lw=3.0, alpha=0.9, zorder=0.5)


def draw_dose_response(axes, bone, emmeans, anchors, waveforms):
    peak_colors = palette.PEAK_COLORS
    for ax, (metric, title) in zip(axes, COLUMNS):
        for peak in (1, 2):
            sel = emmeans[
                (emmeans.bone == bone) & (emmeans.region == "medial")
                & (emmeans.outcome == metric) & (emmeans.peak == peak)
            ].sort_values("sigma")
            if not len(sel):
                continue
            yerr = np.vstack([sel.emmean - sel.lower, sel.upper - sel.emmean])
            ax.errorbar(sel.sigma, sel.emmean, yerr=yerr,
                        fmt="-o" if peak == 1 else "-s", color=peak_colors[peak],
                        capsize=3, lw=2)

        # The three group anchors, at the axis position each group's mean shape
        # projects to. They are reference points, not part of the fitted axis.
        for shape in anchors if SHOW_ANCHORS else ():
            half = N_GRID // 2
            curve = gait_mean(waveforms, shape["label"], metric)
            if curve is None:
                continue
            position = shape.get("k_sigma_equiv", 0.0)
            for segment in (curve[:half], curve[half:]):
                if np.isfinite(segment).any():
                    ax.scatter([position], [np.nanmax(segment)], marker="D", s=55,
                               zorder=5, edgecolor="k",
                               color=palette.GROUP_COLORS[shape["group"]])

        ax.set_title(title, fontsize=FS_TITLE, weight="bold")
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
        sigma_xlabel(ax)
    axes[0].set_ylabel("Peak marginal mean\n(± 95% CI)", fontsize=FS_LABEL)


def draw_metrics(axes, metrics, meta):
    """Geometry along the axis: cartilage area, meniscal coverage, extrusion."""
    medial = palette.MEDLAT_COLORS["medial"]
    lateral = palette.MEDLAT_COLORS["lateral"]

    rows = metrics[
        (metrics.sigma >= meta["sigma_min"] - 1e-9)
        & (metrics.sigma <= meta["sigma_max"] + 1e-9)
    ].sort_values("sigma")
    sigma = rows["sigma"].to_numpy(float)

    def axis(ax):
        ax.set_xticks(sigma)
        ax.axvline(0, color="0.8", lw=1, zorder=0)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(alpha=0.3)
        sigma_xlabel(ax)

    # Cartilage area on twin axes, because femur and tibia areas differ by
    # roughly threefold; the tick labels carry each bone's colour.
    ax = axes[0]
    ax.plot(sigma, rows["femur_artic_area_mm2"] / 1e3, "-o", color=FEMUR_COLOR,
            lw=2.6, ms=9)
    ax.set_ylabel("Area (×10³ mm²)", fontsize=FS_LABEL)
    ax.tick_params(axis="y", labelcolor=FEMUR_COLOR, labelsize=FS_TICK)
    twin = ax.twinx()
    twin.plot(sigma, rows["tibia_artic_area_mm2"] / 1e3, "-s", color=TIBIA_COLOR,
              lw=2.6, ms=9)
    twin.tick_params(axis="y", labelcolor=TIBIA_COLOR, labelsize=FS_TICK)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    twin.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_title("Cartilage\nsurface area", fontsize=FS_TITLE, weight="bold")
    axis(ax)

    ax = axes[1]
    ax.plot(sigma, rows["medial_cov_bone_pct"], "-o", color=medial, lw=2, ms=8)
    ax.plot(sigma, rows["lateral_cov_bone_pct"], "-s", color=lateral, lw=2, ms=8)
    ax.set_ylabel("Coverage (%)", fontsize=FS_LABEL)
    ax.set_title("Meniscal coverage", fontsize=FS_TITLE, weight="bold")
    axis(ax)

    ax = axes[2]
    ax.plot(sigma, rows["medial_extrusion_mm"], "-o", color=medial, lw=2, ms=8)
    ax.plot(sigma, rows["lateral_extrusion_mm"], "-s", color=lateral, lw=2, ms=8)
    ax.axhline(0, color="0.6", lw=1, ls=":")
    ax.set_ylabel("Extrusion (mm)", fontsize=FS_LABEL)
    ax.set_title("Meniscal extrusion", fontsize=FS_TITLE, weight="bold")
    axis(ax)


def _proxy(color, marker):
    """Legend handle drawn thicker than the plotted line, so the small legends
    read at print size."""
    return Line2D([0], [0], color=color, lw=4.6, marker=marker, ms=10.5)


def draw_legends(dose_axes, metric_axes):
    """Legends to the right of their rows, in the colour bar's column, so every
    row keeps its key in the same place."""
    common = dict(loc="upper left", frameon=False, fontsize=FS_LEGEND, borderaxespad=0.0,
                  handlelength=3.0, handletextpad=0.8, labelspacing=0.7)
    dose_axes[2].legend(
        [_proxy(palette.PEAK_COLORS[1], "o"), _proxy(palette.PEAK_COLORS[2], "s")],
        ["Peak 1 (early)", "Peak 2 (late)"], bbox_to_anchor=(1.03, 1.0), **common)

    ax = metric_axes[2]
    area = ax.legend([_proxy(FEMUR_COLOR, "o"), _proxy(TIBIA_COLOR, "s")],
                     ["Femur (left axis)", "Tibia (right axis)"],
                     bbox_to_anchor=(1.03, 1.0), title="Surface area",
                     title_fontsize=FS_LEGEND, **common)
    area.get_title().set_ha("left")
    sides = ax.legend([_proxy(palette.MEDLAT_COLORS["medial"], "o"),
                       _proxy(palette.MEDLAT_COLORS["lateral"], "s")],
                      ["Medial", "Lateral"], bbox_to_anchor=(1.03, 0.42),
                      title="Coverage, extrusion", title_fontsize=FS_LEGEND, **common)
    sides.get_title().set_ha("left")
    ax.add_artist(area)   # a second .legend() call replaces the first; keep both


def draw_colorbar(fig, reference_ax, cmap, norm):
    """Colour bar for the waveform row, centred in the legends' column. The bar
    is short, so the end words go horizontally above and below it and only
    'z-score (σ)' runs along it."""
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    box = reference_ax.get_position(fig)
    # The legends anchor at 1.03 axes-widths; the bar sits further right of
    # that so it lands mid-column with its words.
    x = box.x1 + 0.03 * box.width + 0.024
    cax = fig.add_axes([x, box.y0, 0.009, box.height])
    bar = fig.colorbar(mappable, cax=cax)
    bar.ax.tick_params(labelsize=15)
    bar.ax.text(3.9, 0.5, "z-score (σ)", transform=bar.ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=17)
    bar.ax.text(0.5, 1.04, "↑ Mild OA", transform=bar.ax.transAxes, ha="center",
                va="bottom", fontsize=15, weight="bold", color=END_HIGH)
    bar.ax.text(0.5, -0.04, "↓ Healthy", transform=bar.ax.transAxes, ha="center",
                va="top", fontsize=15, weight="bold", color=END_LOW)


def render(bone: str) -> Path:
    shapes, waveforms, emmeans, meta, metrics = load_inputs(bone)

    shapes = [s for s in shapes if in_window(s, meta)]
    steps = [s for s in shapes if s["type"] == "sigma_step"]
    anchors = [s for s in shapes if s["type"] == "group_anchor"]
    if not steps:
        raise SystemExit(f"No shape steps inside the fitted window for {bone}.")

    positions = [s["k_sigma"] for s in steps]
    cmap = palette.SIGMA_CMAP
    norm = colors.Normalize(min(positions), max(positions))

    # Anatomy columns follow the plotted window, so a restricted figure shows the
    # anatomy at its own endpoints rather than at some wider range's.
    by_position = {s["k_sigma"]: s for s in steps}
    morph_shapes = [by_position[p] for p in (min(positions), 0, max(positions))]
    images = morph_images([s["label"] for s in morph_shapes])

    fig = plt.figure(figsize=FIG_SIZE)
    outer = fig.add_gridspec(3, 1, left=0.535 * SX, right=0.945 * SX, top=0.94,
                             bottom=0.085, hspace=0.75)
    top_axes = [fig.add_subplot(g) for g in outer[0].subgridspec(1, 3, wspace=0.36)]
    dose_axes = [fig.add_subplot(g) for g in outer[1].subgridspec(1, 3, wspace=0.36)]
    metric_axes = [fig.add_subplot(g) for g in outer[2].subgridspec(1, 3, wspace=0.55)]

    draw_morph(fig, 0.05 * SX, 0.445 * SX, 0.47, images, morph_shapes, max_stack=0.70)
    draw_waveforms(top_axes, steps, anchors, waveforms, cmap, norm)
    early, late = ensemble_peak_times(waveforms, steps)
    print(f"  peak lines at {np.median(early):.0f}% / {np.median(late):.0f}% stance "
          f"(ensemble-curve peaks per shape step: {early.tolist()} / {late.tolist()})")
    draw_peak_lines(top_axes, early, late)
    draw_dose_response(dose_axes, bone, emmeans, anchors, waveforms)
    draw_metrics(metric_axes, metrics, meta)
    draw_legends(dose_axes, metric_axes)

    index = panels.letter_axes(top_axes, fontsize=FS_LETTER, offset=LETTER_OFFSET)
    index = panels.letter_axes(dose_axes, start=index, fontsize=FS_LETTER,
                               offset=LETTER_OFFSET)
    panels.letter_axes(metric_axes, start=index, fontsize=FS_LETTER, offset=LETTER_OFFSET)
    draw_colorbar(fig, top_axes[2], cmap, norm)

    out_path = results.output_dir() / f"{FIGURE}.png"
    fig.savefig(out_path, dpi=DPI, facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    print(f"rendering {BONE} ...")
    print(f"  wrote {render(BONE)}")


if __name__ == "__main__":
    main()
