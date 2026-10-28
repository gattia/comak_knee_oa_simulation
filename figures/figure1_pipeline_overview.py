#!/usr/bin/env python3
"""Figure 1 -- the pipeline overview: build a knee, simulate the cohort, analyse it.

    python figures/figure1_pipeline_overview.py

Runs with no arguments, reads ``04_results/figure1_pipeline_overview/`` (the
rendered panels, from ``pipeline/derive_figure1_panels.py``), the static artwork
in ``figures/assets/figure1/``, and three small result tables that other figures
already derive. Writes manuscript Figure 1 as
``figures/output/[<run>/]figure1_pipeline_overview.{png,pdf}``,
183 x 221.5 mm (two-column width) at 500 dpi.

The figure
----------
A schematic drawn around one example knee (``9018389_00m_RIGHT``) in three rows.
Every coordinate below is millimetres on the page, y downward; type sizes are
true points.

A  Build a personalized biomechanical knee model from one MRI.
   i.   Segment -- a DESS slice over its segmentation, and the rough surface mesh
        the segmentation makes (with its imaging artefacts).
   ii.  Neural shape model -- the smooth reconstruction of the same knee, on a
        disc in the shape-model colour.
   iii. Automated multi-body knee model -- the exploded knee (no fat pad) with
        three callouts: its 24 degrees of freedom, the ligament and tendon
        attachment sites, the wrap surfaces. The callout arrows are aimed at
        pixel masks read off the render, so they land on the objects.
B  Simulate 756 knees under ten distinct gaits. The scaled full-body model,
   COMAK, then the example knee's femoral cartilage pressure at three stance
   instants (walking silhouettes above) and cumulated over stance; beside it the
   cohort: 3 x 252 knees x 10 gait conditions.
C  Analyses across gait, time, surface and shape -- one small tile each.
   i.   Gait analysis -- group-mean medial contact pressure over stance, from
        Figure 2's per-knee ten-gait ensemble (``SNIPPET_BONE``).
   ii.  Longitudinal thickness -- the visit timeline and the 6-year Delta
        cartilage thickness by group (healthy | incident OA).
   iii. Pressure-thinning analysis -- Figure 4 panel C: the medial tibia's
        within-knee decile dose-response (categorical mixed model, 95 %
        intervals). The three groups must be graded from decile 3 up; the
        script asserts it.
   iv.  Synthetic experiment -- the decoded knee at -3 and +3 SD along the
        healthy -> mild-OA shape axis, over two snippets of Figure 3 (panels F
        and D): medial contact area and mean pressure at the late-stance peak,
        mixed-model marginal means over Figure 3's fitted window, -3..+3 sigma
        (``SNIPPET_BONE``).

``SNIPPET_BONE`` picks the bone of the two data snippets C.i and C.iv. It is the
femur. Manuscript Figures 2 and 3 plot the tibia; the femoral and tibial values
of the same contact are close (late-stance peak: contact area 324 vs 307 mm² at
-3 sigma, mean pressure 4.60 vs 4.54 MPa), and setting it to "tibia" plots the
tibial values. Row B's pressure maps and C.iv's decoded knees are femoral
whatever it is set to; they are pictures of the anatomy, not the plotted
quantity.

Inputs
------
``04_results/figure1_pipeline_overview/panels/``
    seg_v2, hub_knee, exploded_nofat_bigtriads (+ _anchors.json), body
    (+ body_anchors.json), pressure_{pk1,mid,pk2,cum}, dthick_{healthy,incident},
    synthetic_{m3,p3}.
``figures/assets/figure1/``
    mri_raw, mri_seg (the OAI DICOM is not vendored), icon_crowd, icon_mri,
    Heelstrike / Midstance / Toeoff. Their origin is in that folder's README.
``04_results/figure2_compartment_mechanics/ensemble_<SNIPPET_BONE>.npz``      C.i
``04_results/figure4_decile_dose_response/model/emmeans_categorical_tibia_medial_cum.csv``  C.iii
``04_results/figure3_shape_pressure/mixed_model_emmeans.csv``                 C.iv

The layout is measured, not only placed: after drawing, the script prints a
report of text widths, arrow lengths and plot extents against their boxes, and
flags ``** OVERFLOW`` / ``** short`` where something does not fit.

Regenerate: ``conda run -n comak-knee-oa python pipeline/derive_figure1_panels.py &&
python figures/figure1_pipeline_overview.py`` (C.i, C.iii and C.iv come from the
Figure 2, 3 and 4 derivations and mixed models).
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
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "figures"))

from shared import results  # noqa: E402

FIGURE = "figure1_pipeline_overview"
BUILD_CMD = "conda run -n comak-knee-oa python pipeline/derive_figure1_panels.py"
ASSETS = _REPO / "figures" / "assets" / "figure1"
OUTPUT_STEM = "figure1_pipeline_overview"

# The bone of the two data snippets: C.i (Figure 2's ensemble) and C.iv (Figure
# 3's marginal means). See the module docstring.
SNIPPET_BONE = "femur"

# Where the other figures' tables come from, for the missing-input message.
FIG2, FIG2_CMD = "figure2_compartment_mechanics", "python pipeline/derive_figure2_data.py"
FIG4, FIG4_CMD = "figure4_decile_dose_response", (
    "python pipeline/derive_figure4_data.py && Rscript figures/figure4_decile_dose_response.R")
FIG3, FIG3_CMD = "figure3_shape_pressure", (
    "python pipeline/derive_figure3_data.py && Rscript figures/figure3_mixed_model.R")
FIG4_CELL = "model/emmeans_categorical_tibia_medial_cum.csv"

PANEL_FILES = (
    "seg_v2.png", "hub_knee.png", "exploded_nofat_bigtriads.png", "exploded_nofat_bigtriads_anchors.json",
    "body.png", "body_anchors.json", "pressure_pk1.png", "pressure_mid.png", "pressure_pk2.png",
    "pressure_cum.png", "dthick_healthy.png", "dthick_incident.png", "synthetic_m3.png", "synthetic_p3.png",
)
ASSET_FILES = ("mri_raw.png", "mri_seg.png", "icon_crowd.png", "icon_mri.png",
               "Heelstrike.png", "Midstance.png", "Toeoff.png")

# Page and type. Liberation Sans is registered from the system font folder when
# it is there (Arial-metric) and embedded as TrueType.
W, H = 183.0, 221.5
DPI = 500
for _face in ("Regular", "Bold", "Italic", "BoldItalic"):
    _ttf = Path("/usr/share/fonts/truetype/liberation") / f"LiberationSans-{_face}.ttf"
    if _ttf.exists():
        font_manager.fontManager.addfont(str(_ttf))
plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42, "ps.fonttype": 42})

INK = "#2b2b2b"
GREY = "#5a5a5a"
RULE = "#b8b8b8"
LIG = "#c0392b"
TEN = "#7E57C2"
MAST = "#E3E6E9"              # every row's masthead band
ACCENT = "#37474F"            # the knee ring in row B
NSM = "#B8860B"               # the shape-model panel's colour (goldenrod)
TITLE = INK                   # sub-panel titles
GROUP = {"healthy": "#2196F3", "incident": "#F44336", "mildoa": "#FF9800"}
SD_LO, SD_HI = "#1565C0", "#D2691E"                        # -3 SD healthy / +3 SD mild OA
TRI_TF, TRI_PF, TRI_MEN = "#7B1FA2", "#C2185B", "#EF6C00"  # must match TRI in derive_figure1_panels.py
S_HEAD, S_PAN, S_LAB, S_SUB, S_TICK = 9.0, 7.4, 8.0, 7.3, 6.8
S_PLOT = 7.3                                               # row-C plot text
S2_SUB, S2_TICK, S2_LAB = 8.0, 7.3, 8.6                    # row-B text, one step up


def mix(hex_col, f, base=(255, 255, 255)):
    """f of the colour on white (0 -> white, 1 -> the colour)."""
    c = tuple(int(hex_col[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(round(b + (a - b) * f)) for a, b in zip(c, base))


TINT_T, EDGE_T = mix(NSM, 0.12), mix(NSM, 0.85)

# The page. One full-bleed axes in millimetres (y downward); the plots in row C
# are extra axes placed on it. Module-level so the drawing helpers stay short.
fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")
fig.patch.set_facecolor("white")
REPORT: list[str] = []


# --------------------------------------------------------------------------- inputs
def load_inputs() -> dict:
    """Resolve every input, failing with the command that builds what is missing."""
    panel_paths = results.require(FIGURE, *(f"panels/{f}" for f in PANEL_FILES), produced_by=BUILD_CMD)
    missing = [f for f in ASSET_FILES if not (ASSETS / f).exists()]
    if missing:
        raise FileNotFoundError(f"figures/assets/figure1/ is missing {missing}; see its README.md")
    (ensemble,) = results.require(FIG2, f"ensemble_{SNIPPET_BONE}.npz", produced_by=FIG2_CMD)
    (fig4,) = results.require(FIG4, FIG4_CELL, produced_by=FIG4_CMD)
    (fig3,) = results.require(FIG3, "mixed_model_emmeans.csv", produced_by=FIG3_CMD)
    return {"panel": dict(zip(PANEL_FILES, panel_paths)),
            "asset": {f: ASSETS / f for f in ASSET_FILES},
            "ensemble": ensemble, "fig4": fig4, "fig3": fig3}


# --------------------------------------------------------------------------- drawing helpers
def load(path, trim=True, recolor=None):
    """An RGBA array, trimmed to its opaque pixels; recolor paints every pixel one colour (alpha kept)."""
    a = np.array(Image.open(path).convert("RGBA"))
    if recolor is not None:
        a[:, :, :3] = np.array(recolor, dtype=np.uint8)
    if trim:
        ys, xs = np.where(a[:, :, 3] > 0)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return a


class Placed:
    """An image placed on the page: converts between its pixels and page millimetres."""

    def __init__(self, a, x, y, w, h):
        self.a, self.x, self.y, self.w, self.h = a, x, y, w, h

    def px(self, xm, ym):
        return (int(round((xm - self.x) / self.w * (self.a.shape[1] - 1))),
                int(round((ym - self.y) / self.h * (self.a.shape[0] - 1))))

    def mm(self, px, py):
        return self.x + px / (self.a.shape[1] - 1) * self.w, self.y + py / (self.a.shape[0] - 1) * self.h

    def left_at(self, ym):
        _, r = self.px(self.x, ym)
        return self.mm(np.where(self.a[r, :, 3] > 40)[0].min(), r)[0]

    def right_at(self, ym):
        _, r = self.px(self.x, ym)
        return self.mm(np.where(self.a[r, :, 3] > 40)[0].max(), r)[0]

    def anchor(self, j, key):
        return self.mm(j[key][0], j[key][1])

    @property
    def cx(self):
        return self.x + self.w / 2


def put(a, x, y, w=None, h=None, z=2):
    ih, iw = a.shape[:2]
    if w is None:
        w = h * iw / ih
    if h is None:
        h = w * ih / iw
    ax.imshow(a, extent=(x, x + w, y + h, y), interpolation="lanczos", zorder=z, aspect="auto")
    return Placed(a, x, y, w, h)


def text(x, y, s, size=S_SUB, color=INK, ha="left", va="top", weight="normal", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=z, **kw)


def twidth(t):
    """A text's width in page mm."""
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    return bb.x1 - bb.x0


def smallcaps(x, y, s, size=S_PAN, color=INK, ha="left", va="top", z=5, maxw=None):
    """Letter-spaced small caps; with maxw, shrink (floor 6.6 pt) until it fits and report the measurement."""
    t = text(x, y, " ".join(s.upper()), size=size, color=color, ha=ha, va=va, weight="bold", z=z)
    if maxw is not None:
        w = twidth(t)
        while w > maxw and size > 6.6:
            size -= 0.2
            t.set_fontsize(size)
            w = twidth(t)
        REPORT.append(f"{s!r}: {w:.1f} mm at {size:.1f} pt (max {maxw:.1f})" + ("  ** OVERFLOW" if w > maxw else ""))
    return t


def arrow(x0, y0, x1, y1, color=INK, lw=0.8, z=4, head=8.0):
    """A thin ink sequence arrow; anything under 8 mm is reported as short."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=0.85,head_width=0.30",
                                 mutation_scale=head, lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))
    length = np.hypot(x1 - x0, y1 - y0)
    if length < 8.0:
        REPORT.append(f"** short arrow {length:.1f} mm at ({x0:.1f},{y0:.1f})")


def pointer(ax_, ay_, tx, ty, color=INK, lw=1.0, z=4, bullet=0.75, stop=1.6, head=7.0, floor=8.0):
    """A callout: a filled bullet at the text end, a line, an arrowhead `stop` mm short of the target point."""
    d = np.array([ax_ - tx, ay_ - ty], float)
    length = np.hypot(*d)
    ex, ey = np.array([tx, ty]) + d / length * (length - stop)
    ax.add_patch(FancyArrowPatch((tx, ty), (ex, ey), arrowstyle="-|>,head_length=0.85,head_width=0.32",
                                 mutation_scale=head, lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))
    ax.add_patch(Circle((tx, ty), bullet, facecolor=color, edgecolor="none", zorder=z + 1))
    REPORT.append(f"pointer to ({ax_:.1f},{ay_:.1f}): {length - stop:.1f} mm" + ("  ** short" if length - stop < floor else ""))


def aim(mask_xy, bullet, gap=0.9, color=INK, lw=1.0, z=4, bullet_r=0.75, head=7.0, floor=8.0, label="", toward=None):
    """A callout aimed at an object: from the bullet toward the object's centroid (or `toward`), the tip
    stopping `gap` mm before the object's first pixel on that ray. mask_xy = (N, 2) page-mm pixel coordinates."""
    b = np.array(bullet, float)
    c = mask_xy.mean(0) if toward is None else np.array(toward, float)
    u = (c - b) / np.linalg.norm(c - b)
    tree = cKDTree(mask_xy)
    hit = None
    for s in np.arange(0.0, np.linalg.norm(c - b) + 0.05, 0.05):
        if tree.query(b + u * s)[0] < 0.30:
            hit = s
            break
    assert hit is not None, f"aim({label}): ray never met the mask"
    tip = b + u * (hit - gap)
    ax.add_patch(FancyArrowPatch(tuple(b), tuple(tip), arrowstyle="-|>,head_length=0.85,head_width=0.32",
                                 mutation_scale=head, lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))
    ax.add_patch(Circle(tuple(b), bullet_r, facecolor=color, edgecolor="none", zorder=z + 1))
    length = hit - gap
    REPORT.append(f"aim {label}: {length:.1f} mm, tip ({tip[0]:.1f},{tip[1]:.1f}), object entry {hit:.1f} mm "
                  f"from the bullet, {int(len(mask_xy))} px" + ("  ** short" if length < floor else ""))


def box(x0, y0, x1, y1, fill, edge=RULE, lw=0.5, z=0, r=1.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fill, edgecolor=edge, lw=lw, zorder=z))


def prefix(x, y, s, size, color):
    """The plain bold 'A.' / 'i.' label before a title; returns where the title starts."""
    tp = text(x, y, s, size=size, color=color, weight="bold", va="center_baseline")
    return x + twidth(tp) + 1.7


def row(x0, y0, x1, y1, n, title, lw=0.9, mast_h=8.5):
    """A row container: white body, grey outline, a masthead band with 'n.  TITLE' in small caps."""
    box(x0, y0, x1, y1, "white", edge=RULE, lw=lw)
    ax.add_patch(FancyBboxPatch((x0 + 0.45, y0 + 0.45), x1 - x0 - 0.9, mast_h - 0.45,
                                boxstyle="round,pad=0,rounding_size=1.3", facecolor=MAST, edgecolor="none", zorder=0.5))
    ym = y0 + mast_h / 2 + 0.2
    xt = prefix(x0 + 3.2, ym, f"{n}.", S_HEAD, INK)
    smallcaps(xt, ym, title, size=S_HEAD, color=INK, va="center_baseline", maxw=x1 - xt - 3)


def panel(label, x, y, title, size=S_PAN, maxw=None):
    """Sub-panel head: bold prefix, then the small-caps title. Returns (title_text, x of the title),
    so continuation lines can align under the title. maxw is the width available from x."""
    xt = prefix(x, y + 1.75, label, size, TITLE)
    t = smallcaps(xt, y + 1.75, title, size=size, color=TITLE, va="center_baseline",
                  maxw=None if maxw is None else maxw - (xt - x))
    return t, xt


def colorbar(x, y, w, h, cmap, ticks, size=S_TICK):
    ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, extent=(x, x + w, y + h, y), aspect="auto", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=GREY, lw=0.35, zorder=4))
    for frac, lab in ticks:
        text(x + w * frac, y + h + 0.9, lab, size=size, color=INK,
             ha="left" if frac == 0 else ("right" if frac == 1 else "center"), va="top")


def fit_axes(a, x_lo, x_hi, label, pad=1.0):
    """Centre a fully-labelled plot's whole extent (tick + axis labels) in [x_lo, x_hi] mm.

    The frame keeps its width; it shrinks only if the extent would not fit at all,
    and the report says so."""
    fig.canvas.draw()
    mm = 25.4 / fig.dpi
    bb = a.get_tightbbox(fig.canvas.get_renderer())
    pos = a.get_position()
    ax_l, ax_w_ = pos.x0 * W, pos.width * W
    left_over, right_over = ax_l - bb.x0 * mm, bb.x1 * mm - (ax_l + ax_w_)   # label overhang beyond the frame
    avail = (x_hi - x_lo) - 2 * pad - left_over - right_over
    new_w = ax_w_ if avail >= ax_w_ else avail
    new_l = x_lo + pad + left_over + (avail - new_w) / 2
    a.set_position([new_l / W, pos.y0, new_w / W, pos.height])
    fig.canvas.draw()
    bb2 = a.get_tightbbox(fig.canvas.get_renderer())
    REPORT.append(f"fit_axes {label}: extent was {bb.x0 * mm:.1f}-{bb.x1 * mm:.1f} (tile {x_lo:.1f}-{x_hi:.1f}) -> "
                  f"{bb2.x0 * mm:.1f}-{bb2.x1 * mm:.1f}; frame {ax_w_:.1f} -> {new_w:.1f} mm"
                  + ("  ** OVERFLOW" if bb2.x0 * mm < x_lo or bb2.x1 * mm > x_hi else ""))


def dot_legend(x, y, items, size=S_SUB, gap=2.0, color=INK, r=0.9, dx=2.5):
    for col, lab in items:
        ax.add_patch(Circle((x + r + 0.05, y), r, facecolor=col, edgecolor="none", zorder=6))
        t = text(x + dx, y, lab, size=size, color=color, va="center")
        x = x + dx + twidth(t) + gap
    return x


def style_plot(a):
    a.tick_params(length=1.5, width=0.4, pad=1.0, colors=GREY, labelsize=S_PLOT)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_linewidth(0.5)
        a.spines[sp].set_color(GREY)
    a.patch.set_alpha(0)


def new_axes(x, y, w, h):
    """A plot axes at page mm (x, y) top-left, w x h."""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


# --------------------------------------------------------------------------- page geometry
R1 = (3.0, 3.0, 180.0, 81.0)          # A build
R2 = (3.0, 85.0, 180.0, 147.5)        # B simulate
R3 = (3.0, 151.5, 180.0, 218.5)       # C analyses
AX0, AX1 = 6.0, 59.0                  # A.i
BX0, BX1 = 62.0, 116.0                # A.ii
CX0, CX1 = 118.5, 177.0               # A.iii
PY0, PY1 = 14.0, 78.0                 # row A's panels, top and bottom
HUB_C, HUB_R = ((BX0 + BX1) / 2, 47.0), 20.0   # the shape model's disc
DX0, DX1, DY0, DY1 = 6.0, 177.0, 96.0, 144.5   # row B's content frame
TY0, TY1 = 162.5, 215.5                        # row C's tiles
TX = [(6.0, 46.5), (49.5, 90.0), (93.0, 133.5), (136.5, 177.0)]
TW = TX[0][1] - TX[0][0]
INSET = 2.5


# --------------------------------------------------------------------------- row A -- build
def draw_row_a(inp):
    pan, asset = inp["panel"], inp["asset"]
    box(AX0, PY0, AX1, PY1, "white", z=1)
    box(BX0, PY0, BX1, PY1, TINT_T, edge=EDGE_T, lw=1.1, z=1)
    box(CX0, PY0, CX1, PY1, "white", z=1)

    # i. Segment: the MRI over its segmentation (an arrow between them names the
    # step), the rough surface mesh to their right, the tissue key as a footer.
    panel("i.", AX0 + 2.5, PY0 + 2.6, "Segment")
    h_img = 17.0
    mri = load(asset["mri_raw.png"], trim=False)
    seg = load(asset["mri_seg.png"], trim=False)
    text(AX0 + 2.5, PY0 + 8.6, "3T DESS MRI", size=S_SUB)
    p_mri = put(mri, AX0 + 2.5, PY0 + 12.2, h=h_img)
    gap_a = 9.0
    y_a0 = p_mri.y + p_mri.h + 1.0
    y_a1 = y_a0 + gap_a
    arrow(p_mri.x + 2.2, y_a0, p_mri.x + 2.2, y_a1 - 0.2)
    for i, line in enumerate(("deep-learning", "segmentation")):
        ta = text(p_mri.x + 4.8, y_a0 + 1.2 + i * 3.5, line, size=S_TICK, color=GREY)
        REPORT.append(f"A.i arrow label {line!r} ends x {p_mri.x + 4.8 + twidth(ta):.1f}")
    p_seg = put(seg, AX0 + 2.5, y_a1 + 1.0, h=h_img)
    REPORT.append(f"A.i: arrow {gap_a - 0.2:.1f} mm (floor 8); stack bottom {p_seg.y + p_seg.h:.1f}, "
                  f"key top {PY1 - 2.6 - 1.3:.1f} (box {PY1})")
    mesh = load(pan["seg_v2.png"])
    h_mesh = 38.0                                 # the mesh is tall: height is the budget
    w_mesh = h_mesh * mesh.shape[1] / mesh.shape[0]
    x_stack = p_mri.x + max(p_mri.w, p_seg.w)
    p_mesh = put(mesh, (x_stack + AX1) / 2 - w_mesh / 2, HUB_C[1] - h_mesh / 2 + 3.0, h=h_mesh)
    text(p_mesh.cx, p_mesh.y - 1.0, "w/ imaging artefacts", size=S_SUB, color=GREY, ha="center", va="bottom")
    text(p_mesh.cx, p_mesh.y - 4.4, "Surface mesh", size=S_LAB, ha="center", va="bottom")
    xe = dot_legend(AX0 + 2.5, PY1 - 2.6, [("#d6c8a8", "bone"), ("#5aa9e6", "cartilage"), ("#e8a26a", "menisci")],
                    size=S_SUB, r=1.1, dx=3.0, gap=2.6)
    REPORT.append(f"A.i mesh right edge {p_mesh.x + p_mesh.w:.1f}, key ends {xe:.1f} (box {AX1}); "
                  f"mesh spans y {p_mesh.y:.1f}-{p_mesh.y + p_mesh.h:.1f}")

    # ii. Neural shape model: the decoded knee on the disc.
    panel("ii.", BX0 + 2.5, PY0 + 2.6, "Neural shape model", maxw=BX1 - BX0 - 5)
    ax.add_patch(Circle(HUB_C, HUB_R, facecolor="white", alpha=0.72, edgecolor=mix(NSM, 0.65), lw=1.2, zorder=1.3))
    hub = load(pan["hub_knee.png"])
    h_hub = 36.0
    put(hub, HUB_C[0] - h_hub * hub.shape[1] / hub.shape[0] / 2, HUB_C[1] - h_hub / 2, h=h_hub, z=3)
    for i, line in enumerate(("Encode & reconstruct", "smooth knee geometry")):
        tb = text(HUB_C[0], HUB_C[1] + HUB_R + 2.6 + i * 3.7, line, size=S_SUB, color=GREY, ha="center")
        REPORT.append(f"A.ii caption {line!r}: {twidth(tb):.1f} mm (box {BX1 - BX0 - 4:.1f}); "
                      f"ends y {HUB_C[1] + HUB_R + 2.6 + i * 3.7 + 3.0:.1f} (box {PY1})")
    arrow(p_mesh.right_at(HUB_C[1]) + 0.8, HUB_C[1], HUB_C[0] - HUB_R - 0.6, HUB_C[1])
    REPORT.append(f"mesh -> NSM arrow {HUB_C[0] - HUB_R - 0.6 - (p_mesh.right_at(HUB_C[1]) + 0.8):.1f} mm (floor 8)")

    # iii. Automated multi-body knee model: the exploded knee and three callouts.
    _, xc = panel("iii.", CX0 + 2.5, PY0 + 2.6, "Automated multi-body", maxw=CX1 - CX0 - 5)
    smallcaps(xc, PY0 + 2.6 + 1.75 + 4.2, "knee model", size=S_PAN, color=TITLE, va="center_baseline",
              maxw=CX1 - xc - 2.5)
    ex = load(pan["exploded_nofat_bigtriads.png"])
    anc = json.loads(pan["exploded_nofat_bigtriads_anchors.json"].read_text())
    p_ex = put(ex, CX0 + 2.0, PY0 + 10.0, h=52.5)
    anchors = {k: p_ex.anchor(anc, k) for k in anc}
    arrow(HUB_C[0] + HUB_R + 0.6, HUB_C[1], p_ex.left_at(HUB_C[1]) - 0.5, HUB_C[1])
    REPORT.append(f"NSM -> build arrow {p_ex.left_at(HUB_C[1]) - 0.5 - (HUB_C[0] + HUB_R + 0.6):.1f} mm (floor 8)")
    draw_callouts(ex, p_ex, anchors)


def draw_callouts(ex, p_ex, anchors):
    """A.iii's three callouts. The targets are pixel masks read off the render, so they follow it."""
    cx = p_ex.x + p_ex.w + 1.8        # bullet column
    ctx = cx + 2.2                    # text column
    r_, g_, b_, al = [ex[:, :, i].astype(int) for i in range(4)]
    ys, xs = np.mgrid[0:ex.shape[0], 0:ex.shape[1]]
    mx = p_ex.x + xs / (ex.shape[1] - 1) * p_ex.w
    my = p_ex.y + ys / (ex.shape[0] - 1) * p_ex.h

    def near(hex_col, tol=60):
        c = tuple(int(hex_col[i:i + 2], 16) for i in (1, 3, 5))
        return (al > 200) & (abs(r_ - c[0]) < tol) & (abs(g_ - c[1]) < tol) & (abs(b_ - c[2]) < tol)

    red = (al > 200) & (r_ > 150) & (g_ < 90) & (b_ < 90)
    purple = near(TEN, 55)
    fem_top = anchors["femur_top"][1]
    tri = anchors["triad_PF"]
    m_triad = near(TRI_PF, 70) & (np.hypot(mx - tri[0], my - tri[1]) < 9.0)         # the PF triad's arrows
    up = m_triad & (my < tri[1] - 1.0)                                              # its upright arm
    triad_up = (float(mx[up].mean()), float(my[up].mean()))
    # The patella's dots, not the triad's shaded magenta (which also passes the red test).
    m_pat = (red | purple) & ~near(TRI_PF, 70) & (my > tri[1] - 1.0) & (my < fem_top - 1.5)
    m_fem = red & (my > fem_top + 3.0) & (my < fem_top + 12.0) & (mx > p_ex.x + 0.62 * p_ex.w)   # femoral cluster
    for name, m in (("PF triad", m_triad), ("patella dots", m_pat), ("femur cluster", m_fem)):
        assert m.sum() > 60, f"{name} mask is nearly empty ({int(m.sum())} px)"
        REPORT.append(f"mask {name}: {int(m.sum())} px, centroid ({mx[m].mean():.1f},{my[m].mean():.1f})")

    def xy(m):
        return np.column_stack([mx[m], my[m]])

    # Each callout sits at its anchor's height, pushed down so the sub-lines never
    # touch, and kept inside the box.
    call = [("triad_PF", "24 DOF", "dof"), ("patella_site", "Attachment sites", "att"), ("tibia_wrap", "Wrap surfaces", "wrap")]
    ys_call = [anchors[k][1] for k, _, _ in call]
    ys_call[0] -= 2.5
    for i in range(1, len(ys_call)):
        ys_call[i] = max(ys_call[i], ys_call[i - 1] + (17.5 if i == 1 else 13.0))
    ys_call[-1] = min(ys_call[-1], PY1 - 8.5)
    REPORT.append("callout y: " + ", ".join(f"{k} {y:.1f} (anchor {anchors[k][1]:.1f})"
                                            for (k, _, _), y in zip(call, ys_call)))
    for (key, title, sub), yc in zip(call, ys_call):
        if key == "tibia_wrap":     # its anchor is close to the text column: lower + indent so the arrow comes in diagonally
            yc, off = PY1 - 6.0, 4.2
        else:
            off = 0.0
        cxo, ctx_ = cx + off, ctx + off
        t = text(ctx_, yc, title, size=S_LAB, weight="bold", va="center")
        REPORT.append(f"callout {title!r}: {twidth(t):.1f} mm (max {CX1 - ctx - 1.5:.1f})")
        if sub == "dof":            # one coordinate system per line, spelled out
            for k, (frag, col) in enumerate((("6 tibiofemoral", TRI_TF), ("6 patellofemoral", TRI_PF), ("12 meniscal", TRI_MEN))):
                td = text(ctx, yc + 2.5 + k * 3.6, frag, size=S_SUB, color=col, weight="bold")
                REPORT.append(f"DOF line {frag!r}: {twidth(td):.1f} mm (max {CX1 - ctx - 1.5:.1f})")
            REPORT.append(f"DOF lines end y {yc + 2.5 + 2 * 3.6 + 3.0:.1f}; Attachment callout at {ys_call[1]:.1f}")
        elif sub == "att":
            text(ctx, yc + 2.5, "91 fibres", size=S_SUB, color=GREY)
            xe = dot_legend(ctx, yc + 6.9, [(LIG, "ligament"), (TEN, "tendon")], size=S_SUB, gap=1.6, r=0.8, dx=2.2, color=GREY)
            REPORT.append(f"ligament/tendon legend ends at {xe:.1f} (box {CX1})")
        elif sub == "wrap":
            te = text(ctx_, yc + 2.5, "ellipsoid · cylinder", size=S_SUB, color=GREY)
            REPORT.append(f"wrap sub-line ends x {ctx_ + twidth(te):.1f}, y {yc + 2.5 + 3.0:.1f} (box {CX1}, {PY1})")
        if key == "triad_PF":
            aim(xy(m_triad), (cxo, yc), gap=1.0, label="24 DOF -> PF triad", toward=triad_up)
        elif key == "patella_site":
            aim(xy(m_pat), (cxo, yc), gap=0.8, label="Attachment -> patella dots")
            # The femoral cluster sits ~5 mm from the text column, so this fork is
            # short by construction: a smaller head keeps a visible shaft, floor 3.5.
            aim(xy(m_fem), (cxo, yc), gap=0.6, head=5.5, floor=3.5, label="Attachment -> femur cluster")
        else:
            pointer(anchors[key][0], anchors[key][1], cxo, yc)


# --------------------------------------------------------------------------- row B -- simulate
def draw_row_b(inp):
    pan, asset = inp["panel"], inp["asset"]
    body = load(pan["body.png"])
    banc = json.loads(pan["body_anchors.json"].read_text())
    p_body = put(body, DX0 + 3.0, 98.8, h=36.5)
    kx, ky = p_body.anchor(banc, "knee")
    ax.add_patch(Circle((kx, ky), 2.6, facecolor="none", edgecolor=ACCENT, lw=0.9, zorder=6))
    tb1 = text(p_body.x, p_body.y + p_body.h + 1.2, "Full-body model", size=S2_LAB, weight="bold")
    tb2 = text(p_body.x, p_body.y + p_body.h + 5.1, "scaled to subject", size=S2_SUB, color=GREY)
    REPORT.append(f"body labels: {twidth(tb1):.1f} / {twidth(tb2):.1f} mm from x {p_body.x:.1f} (first bar tick at 38.0); "
                  f"end y {p_body.y + p_body.h + 5.1 + 3.2:.1f} (box {DY1})")

    # body -> COMAK -> the three stance instants (each with its walker above) and
    # the cumulative map, bottom-aligned so the colour bars share a baseline.
    x_map0, w_map, gap_map = 38.0, 24.0, 2.5
    pm = {t: load(pan[f"pressure_{t}.png"], trim=False) for t in ("pk1", "mid", "pk2", "cum")}
    h_map = w_map * pm["pk1"].shape[0] / pm["pk1"].shape[1]
    y_map0 = 112.5
    y_map1 = y_map0 + h_map
    y_flow = (y_map0 + y_map1) / 2
    hw = 9.5                                       # walker height; its feet sit 4.6 mm above the map
    maps = []
    for i, (t, lab, walker) in enumerate(zip(("pk1", "mid", "pk2"), ("early-stance peak", "midstance", "late-stance peak"),
                                             ("Heelstrike", "Midstance", "Toeoff"))):
        p = put(pm[t], x_map0 + i * (w_map + gap_map), y_map0, w=w_map)
        maps.append(p)
        t_l = text(p.cx, y_map0 - 1.0, lab, size=S2_SUB, color=INK, ha="center", va="bottom")
        a = load(asset[f"{walker}.png"], recolor=(58, 55, 51))
        pw = put(a, p.cx - hw * a.shape[1] / a.shape[0] / 2, y_map0 - 4.6 - hw, h=hw)
        REPORT.append(f"instant {lab!r}: label {twidth(t_l):.1f} mm over a {w_map} mm map; walker top y {pw.y:.1f} (box {DY0})")
    w_cum = 26.0
    x_cum = maps[-1].x + maps[-1].w + 6.0
    h_cum = w_cum * pm["cum"].shape[0] / pm["cum"].shape[1]
    p_cum = put(pm["cum"], x_cum, y_map1 - h_cum, w=w_cum)
    t_c = text(p_cum.cx, p_cum.y - 4.8, "cumulative pressure", size=S2_SUB, color=INK, ha="center", va="bottom", weight="bold")
    text(p_cum.cx, p_cum.y - 1.0, "over stance", size=S2_SUB, color=INK, ha="center", va="bottom", weight="bold")
    REPORT.append(f"cumulative title {twidth(t_c):.1f} mm over a {w_cum} mm map; map spans {p_cum.x:.1f}-{p_cum.x + p_cum.w:.1f} (rule at 151)")
    x_arrow0 = p_body.right_at(y_flow) + 1.0
    arrow(x_arrow0, y_flow, x_map0 - 1.2, y_flow)
    text((x_arrow0 + x_map0 - 1.2) / 2, y_flow - 1.6, "COMAK", size=S2_LAB, weight="bold", ha="center", va="bottom")
    x_bar0, x_bar1 = maps[0].x, maps[-1].x + maps[-1].w
    y_bar = y_map1 + 2.4
    colorbar(x_bar0, y_bar, x_bar1 - x_bar0, 1.7, "cividis", [(0, "0"), (0.5, "3"), (1, "≥ 6")], size=S2_TICK)
    text((x_bar0 + x_bar1) / 2, y_bar + 5.9, "tibiofemoral contact pressure (MPa)", size=S2_SUB, color=INK, ha="center")
    colorbar(p_cum.x, y_bar, p_cum.w, 1.7, "cividis", [(0, "0"), (1, "≥ 3")], size=S2_TICK)
    t_cb = text(p_cum.cx, y_bar + 5.9, "MPa·s", size=S2_SUB, color=INK, ha="center")
    REPORT.append(f"bar labels end at y {y_bar + 5.9 + 3.4:.1f} (box {DY1}); cumulative bar label {twidth(t_cb):.1f} mm")

    # The cohort block, right of a thin rule.
    x_rule = 151.0
    ax.plot([x_rule, x_rule], [DY0 + 3.0, DY1 - 3.0], color=RULE, lw=0.5, zorder=3)
    xc_c = (x_rule + DX1) / 2
    crowd = load(asset["icon_crowd.png"], recolor=(90, 90, 90))
    put(crowd, xc_c - 11.0 * crowd.shape[1] / crowd.shape[0] / 2, DY0 + 3.0, h=11.0)
    text(xc_c, DY0 + 16.5, "756 knees", size=10.0, weight="bold", ha="center")
    text(xc_c, DY0 + 21.7, "706 participants", size=7.8, color=GREY, ha="center")
    for i, (g, lab) in enumerate((("healthy", "252 healthy"), ("incident", "252 incident OA"), ("mildoa", "252 mild OA"))):
        yy = DY0 + 27.5 + i * 4.0
        ax.add_patch(Circle((x_rule + 3.8, yy), 0.95, facecolor=GROUP[g], edgecolor="none", zorder=6))
        tg = text(x_rule + 6.3, yy, lab, size=7.6, color=INK, va="center")
        REPORT.append(f"cohort row {lab!r} ends x {x_rule + 6.3 + twidth(tg):.1f} (box {DX1})")
    text(xc_c, DY0 + 40.5, "×  10 gait cycles", size=7.8, color=INK, ha="center")
    ts = text(xc_c, DY0 + 44.8, "7,560 simulations", size=8.3, weight="bold", ha="center")
    REPORT.append(f"simulations line {twidth(ts):.1f} mm centred at {xc_c:.1f} (rule {x_rule}, box {DX1}); "
                  f"ends y {DY0 + 44.8 + 3.3:.1f} (box {DY1})")


# --------------------------------------------------------------------------- row C -- analyses
def tile_head(i, label, title, title2=None, flush=False):
    """A tile's title (and an optional second line); returns the y where content may start.
    flush=True hangs the second line under the prefix instead of under the title text."""
    x0 = TX[i][0]
    _, xt = panel(label, x0 + INSET, TY0 + 2.6, title, maxw=TW - 2 * INSET)
    y = TY0 + 2.6 + 1.75
    if title2:
        xc = x0 + INSET if flush else xt
        y += 4.2
        smallcaps(xc, y, title2, size=S_PAN, color=TITLE, va="center_baseline", maxw=x0 + TW - INSET - xc)
    return y + 3.5


def draw_gait_tile(inp):
    """C.i: group-mean medial contact pressure over stance, from Figure 2's per-knee ten-gait ensemble."""
    x0 = TX[0][0]
    y_c = tile_head(0, "i.", "Gait analysis")
    ens = np.load(inp["ensemble"], allow_pickle=True)
    ax_x, ax_y, ax_w, ax_h = x0 + INSET + 5.0, y_c + 1.5, TW - 2 * INSET - 6.0, 28.0
    a = new_axes(ax_x, ax_y, ax_w, ax_h)
    pct = np.arange(101)
    ymax = 0.0
    for key, g in (("kl0_1_healthy", "healthy"), ("kl0_1_progressor", "incident"), ("kl2", "mildoa")):
        y = np.asarray(ens[f"{key}__mean_pressure"], float) / 1e6          # stored in Pa
        m = np.nanmean(y, 0)
        se = np.nanstd(y, 0) / np.sqrt(np.isfinite(y).sum(0))
        a.plot(pct, m, color=GROUP[g], lw=1.0, solid_capstyle="round")
        ymax = max(ymax, float(np.nanmax(m + 1.96 * se)))
        REPORT.append(f"C.i {SNIPPET_BONE} {g}: n {y.shape[0]}, mean-pressure peak {np.nanmax(m):.2f} MPa")
    ytop = 6.0 if ymax > 4.4 else 4.0                                     # round ticks
    a.set_xlim(0, 100)
    a.set_ylim(0, ytop * 1.08)
    a.set_xticks([0, 50, 100])
    a.set_xticklabels(["0", "50", "100 %"])
    a.set_yticks([0, ytop / 2, ytop])
    a.set_yticklabels(["0", f"{ytop / 2:g}", f"{ytop:g}"])
    style_plot(a)
    a.set_xlabel("stance", size=S_PLOT, color=GREY, labelpad=1.2)
    a.set_ylabel("MPa", size=S_PLOT, color=GREY, labelpad=1.5)
    fit_axes(a, x0, x0 + TW, "C.i gait", pad=0.7)
    xe = dot_legend(x0 + INSET, ax_y + ax_h + 8.6, [(GROUP["healthy"], "healthy"), (GROUP["incident"], "incident"),
                                                   (GROUP["mildoa"], "mild OA")],
                    size=S_PLOT, gap=1.2, r=0.8, dx=2.0, color=GREY)
    REPORT.append(f"C.i legend ends x {xe:.1f} (box {TX[0][1]}); bottom {ax_y + ax_h + 8.6 + 2.5:.1f} (box {TY1})")


def draw_longitudinal_tile(inp):
    """C.ii: the visit timeline, then the 6-year Delta cartilage thickness by group."""
    x0 = TX[1][0]
    y_c = tile_head(1, "ii.", "Longitudinal", title2="thickness")
    icon = load(inp["asset"]["icon_mri.png"], recolor=(90, 90, 90))
    tl0, tl1 = x0 + 4.0, x0 + TW - 4.0
    y_tl = y_c + 5.4
    ax.plot([tl0, tl1], [y_tl, y_tl], color=GREY, lw=0.6, zorder=3)
    for yr in (0, 1, 2, 3, 4, 6):
        xx = tl0 + (tl1 - tl0) * yr / 6
        put(icon, xx - 2.1, y_tl - 5.0, h=4.2)
        ax.plot([xx, xx], [y_tl - 0.6, y_tl + 0.6], color=GREY, lw=0.6, zorder=3)
        text(xx, y_tl + 1.2, f"{yr} y" if yr == 6 else str(yr), size=S_PLOT, color=GREY, ha="center")
    w_t = 15.0
    xs3 = [x0 + (TW - 2 * w_t - 3.0) / 2, x0 + (TW - 2 * w_t - 3.0) / 2 + w_t + 3.0]
    y_t = y_tl + 9.0
    for tag, lab, xx in zip(("healthy", "incident"), ("healthy", "incident OA"), xs3):
        text(xx + w_t / 2, y_t - 0.9, lab, size=S_PLOT + 0.2, color=GROUP[tag], ha="center", va="bottom", weight="bold")
        pv = put(load(inp["panel"][f"dthick_{tag}.png"]), xx, y_t, w=w_t)
    y_cb = y_t + pv.h + 1.6
    colorbar(xs3[0], y_cb, xs3[1] + w_t - xs3[0], 1.5, "RdBu_r", [(0, "−50"), (0.5, "0"), (1, "+50")], size=S_PLOT)
    te = text((xs3[0] + xs3[1] + w_t) / 2, y_cb + 5.6, "Δ cartilage thickness (μm/yr)", size=S_PLOT, color=GREY, ha="center")
    REPORT.append(f"C.ii: maps y {y_t:.1f}-{y_t + pv.h:.1f}, bar label {twidth(te):.1f} mm (max {TW - 2 * INSET:.1f}); "
                  f"bottom {y_cb + 5.6 + 3.2:.1f} (box {TY1})")


def draw_dose_response_tile(inp):
    """C.iii: Figure 4 panel C -- medial tibia, cumulative pressure. Each knee's cartilage split into
    ten deciles of its own pressure; the categorical model's marginal mean thickness change per decile
    by group, 95 % intervals, gaits averaged out. Downward = harder-loaded cartilage of the same knee
    thinned faster."""
    x0 = TX[2][0]
    y_c = tile_head(2, "iii.", "Pressure–", title2="Thinning analysis", flush=True)
    em = pd.read_csv(inp["fig4"])
    if "decile" not in em.columns:
        em["decile"] = em["decile_f"].astype(int)
    lo_col = "lower" if "lower" in em.columns else "asymp.LCL"
    hi_col = "upper" if "upper" in em.columns else "asymp.UCL"
    ax_x, ax_y, ax_w, ax_h = x0 + INSET + 6.0, y_c + 1.5, TW - 2 * INSET - 7.0, 28.0
    a = new_axes(ax_x, ax_y, ax_w, ax_h)
    lo_all, hi_all = 0.0, 0.0
    by_group = {k: em[em.group == k].sort_values("decile").emmean.to_numpy()
                for k in ("kl0_1_healthy", "kl2", "kl0_1_progressor")}
    # 'Graded' = the three lines fan out in order as pressure rises. They share an
    # origin at decile 1, so the order is asserted from decile 3 up.
    assert (np.all(by_group["kl0_1_healthy"][2:] > by_group["kl2"][2:])
            and np.all(by_group["kl2"][2:] > by_group["kl0_1_progressor"][2:])), "C.iii: groups not graded (deciles 3-10)"
    for key, g in (("kl0_1_healthy", "healthy"), ("kl0_1_progressor", "incident"), ("kl2", "mildoa")):
        f = em[em.group == key].sort_values("decile")
        assert len(f) == 10, (key, len(f))
        xp = (f["decile"].to_numpy(float) - 0.5) * 10.0                  # decile -> its band's midpoint
        a.fill_between(xp, f[lo_col].to_numpy(float), f[hi_col].to_numpy(float), color=GROUP[g], alpha=0.18, lw=0)
        a.plot(xp, f["emmean"].to_numpy(float), color=GROUP[g], lw=1.0, solid_capstyle="round")
        lo_all = min(lo_all, float(f[lo_col].min()))
        hi_all = max(hi_all, float(f[hi_col].max()))
        REPORT.append(f"C.iii {g}: emmean decile 1 {f['emmean'].iloc[0]:+.2f} -> decile 10 {f['emmean'].iloc[-1]:+.2f} % / yr")
    a.axhline(0, color=RULE, ls=(0, (2.5, 2.5)), lw=0.5, zorder=1)
    a.set_xlim(0, 100)
    a.set_ylim(min(-2.4, lo_all - 0.1), max(0.3, hi_all + 0.1))
    a.set_xticks([0, 50, 100])
    a.set_xticklabels(["0", "50", "100 %"])
    a.set_yticks([0, -1, -2])
    a.set_yticklabels(["0", "−1", "−2"])
    style_plot(a)
    a.set_xlabel("pressure percentile", size=S_PLOT, color=GREY, labelpad=1.2)
    a.set_ylabel("Δ thickness (% / yr)", size=S_PLOT, color=GREY, labelpad=1.5)
    fit_axes(a, x0, x0 + TW, "C.iii pressure-thinning", pad=0.7)
    REPORT.append(f"C.iii: x-label bottom ~{ax_y + ax_h + 5.6:.1f} (box {TY1}); ylim {a.get_ylim()}")


def draw_synthetic_tile(inp):
    """C.iv: the decoded knee at -3 / +3 SD, over two snippets of Figure 3 -- along the healthy ->
    mild-OA shape axis the medial contact patch grows and mean pressure falls. Mixed-model marginal
    means at the late-stance peak, SNIPPET_BONE medial, nine measured gaits averaged out."""
    x0 = TX[3][0]
    _, xg = panel("iv.", x0 + INSET, TY0 + 2.6, "Synthetic", maxw=TW - 2 * INSET)
    smallcaps(xg, TY0 + 2.6 + 1.75 + 4.2, "experiment", size=S_PAN, color=TITLE, va="center_baseline",
              maxw=x0 + TW - INSET - xg)
    raw = [np.array(Image.open(inp["panel"][f"synthetic_{t}.png"]).convert("RGBA")) for t in ("m3", "p3")]
    ys, xs = np.where(np.any([r[:, :, 3] > 0 for r in raw], axis=0))
    syn = [r[ys.min():ys.max() + 1, xs.min():xs.max() + 1] for r in raw]   # one common crop: one camera, one scale
    w_syn, gap_syn = 12.5, 3.0
    xs_syn = x0 + TW / 2 - w_syn - gap_syn / 2
    y_syn = TY0 + 11.2
    for k, (a, lab, sub, col) in enumerate(zip(syn, ("−3 SD", "+3 SD"), ("healthy", "mild OA"), (SD_LO, SD_HI))):
        ps = put(a, xs_syn + k * (w_syn + gap_syn), y_syn, w=w_syn)
        text(ps.cx, y_syn + ps.h + 1.0, lab, size=S_SUB, color=INK, weight="bold", ha="center")
        t2 = text(ps.cx, y_syn + ps.h + 4.3, sub, size=S_SUB, color=col, weight="bold", ha="center")
        REPORT.append(f"C.iv label {sub!r}: {twidth(t2):.1f} mm under a {ps.w:.1f} mm knee at {ps.cx:.1f}")
    y_lab = y_syn + ps.h + 4.3
    REPORT.append(f"C.iv knees y {y_syn:.1f}-{y_syn + ps.h:.1f}, SD labels top {y_lab:.1f}")
    em5 = pd.read_csv(inp["fig3"])
    y_ttl = y_lab + 4.3
    ax_y5, ax_h5 = y_ttl + 3.4, 9.0
    snips = [("contact_area", "contact area", "mm²", x0 + INSET + 7.0, 10.0, (300, 540), [300, 500]),
             ("mean_pressure", "pressure", "MPa", x0 + INSET + 25.0, 10.5, (2.5, 4.4), [3, 4])]
    for outcome, ttl, unit, axx, axw, ylim, yticks in snips:
        sel = em5[(em5.bone == SNIPPET_BONE) & (em5.region == "medial") & (em5.outcome == outcome)
                  & (em5.peak == 2)].sort_values("sigma")
        # The snippet spans whatever window Figure 3's model was fitted on, one
        # point per integer sigma step; its ends are the two decoded knees above.
        lo, hi = sel.sigma.min(), sel.sigma.max()
        assert sel.sigma.tolist() == list(range(int(lo), int(hi) + 1)), (outcome, sel.sigma.tolist())
        assert sel.emmean.min() >= ylim[0] and sel.emmean.max() <= ylim[1], (outcome, sel.emmean.min(), sel.emmean.max())
        a5 = new_axes(axx, ax_y5, axw, ax_h5)
        a5.plot(sel.sigma, sel.emmean, color=INK, lw=0.9, solid_capstyle="round", zorder=3)
        a5.plot([sel.sigma.iloc[0]], [sel.emmean.iloc[0]], "o", color=SD_LO, ms=4.2, mec="white", mew=0.4, zorder=4, clip_on=False)
        a5.plot([sel.sigma.iloc[-1]], [sel.emmean.iloc[-1]], "o", color=SD_HI, ms=4.2, mec="white", mew=0.4, zorder=4, clip_on=False)
        a5.set_xlim(lo - 0.5, hi + 0.5)
        a5.set_ylim(*ylim)
        a5.set_xticks([lo, 0, hi])
        a5.set_xticklabels([f"−{abs(lo):g}", "0", f"+{hi:g}"])
        a5.set_yticks(yticks)
        style_plot(a5)
        a5.set_ylabel(unit, size=S_PLOT, color=GREY, labelpad=1.5)
        tt = text(axx + axw / 2, y_ttl, ttl, size=S_PLOT, color=INK, ha="center", va="top")
        REPORT.append(f"C.iv {SNIPPET_BONE} snippet {ttl!r}: title {twidth(tt):.1f} mm at {axx + axw / 2:.1f} "
                      f"(axes {axx:.1f}-{axx + axw:.1f}, box {x0}-{x0 + TW}); "
                      f"{sel.emmean.iloc[0]:.2f} -> {sel.emmean.iloc[-1]:.2f} {unit}")
    tx = text(x0 + TW / 2, ax_y5 + ax_h5 + 3.6, "shape axis (SD)", size=S_PLOT, color=GREY, ha="center", va="top")
    REPORT.append(f"C.iv x-label {twidth(tx):.1f} mm; bottom {ax_y5 + ax_h5 + 3.6 + 2.8:.1f} (box {TY1})")


# --------------------------------------------------------------------------- main
def main() -> int:
    inp = load_inputs()
    row(*R1, "A", "Build a personalized biomechanical knee model from one MRI")
    row(*R2, "B", "Simulate 756 knees under ten distinct gaits")
    row(*R3, "C", "Analyses across gait, time, surface, and shape")
    draw_row_a(inp)
    draw_row_b(inp)
    for x0, x1 in TX:
        box(x0, TY0, x1, TY1, "white", z=1)
    draw_gait_tile(inp)
    draw_longitudinal_tile(inp)
    draw_dose_response_tile(inp)
    draw_synthetic_tile(inp)

    out = results.output_dir()
    png, pdf = out / f"{OUTPUT_STEM}.png", out / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png, dpi=DPI, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    print(f"wrote {png} {Image.open(png).size} and {pdf.name} (snippet bone: {SNIPPET_BONE})")
    print("\n".join(REPORT))
    overflow = [r for r in REPORT if "OVERFLOW" in r]
    if overflow:
        print(f"\n{len(overflow)} layout overflow(s) -- see ** OVERFLOW above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
