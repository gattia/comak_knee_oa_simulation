#!/usr/bin/env python3
"""Extended Data Figure 1: point correspondence by neural shape model latent interpolation.

A schematic with no data behind it. The paths in panel a
are produced by running the pipeline's own update rule on a two-dimensional stand-in for
the shape model. The stand-in blends two analytic implicit functions,

    f(x; t) = (1 - t) f1(x) + t f2(x),

which plays the role of "the signed distance predicted by the network at the interpolated
latent z_t". The latent is stepped in 100 increments and every point takes one Newton
projection per increment,

    x <- x - f(x; z_t) grad f(x; z_t) / ||grad f(x; z_t)||^2,

the same rule ``NSM.mesh.interpolate.interpolate_common`` applies (one ``_project_once`` per
``np.linspace(1/n, 1, n)`` step), so the figure shows what the method does.

The one difference is the latent path: the production code interpolates the latent
vector (spherically for model building, linearly for the thickness/pressure transfers);
here the implicit functions are blended, because the toy has no latent. The per-point
mechanics, which the figure illustrates, are identical.

Writes manuscript Extended Data Figure 1 as
``figures/output/[<run>/]figure_ed1_latent_interpolation.{png,pdf}``. Needs no data root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle  # noqa: E402

# Where every figure renders (shared.results.output_dir): figures/output/, or
# its run's subfolder on a run-named data root. Needs no data root.
_FIGURES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_FIGURES_DIR.parent / "src"))
sys.path.insert(0, str(_FIGURES_DIR))
from shared.results import output_dir  # noqa: E402

OUT_DIR = output_dir(create=False)
OUT_STEM = "figure_ed1_latent_interpolation"

N_STEPS = 100  # the pipeline's n_steps

# ------------------------------------------------------------------ colours ----
# Surface 1 / surface 2 in Okabe-Ito blue / vermillion: the pair survives
# deuteranopia, and that hue difference carries the distinction in panel a.
SURF1 = "#0072B2"      # surface 1 (latent z1)
SURF2 = "#D55E00"      # surface 2 (latent z2)
INTER = "#9e9e9e"      # intermediate level sets
PATH_COLORS = {"i": "#CC79A7", "j": "#009E73", "k": "#000000"}
FAINT = "#b0bec5"
STEP = "#7B1FA2"       # the Newton step in panel b, off the red/orange axis
GRAD = "#0072B2"       # the gradient / normal in panel b
TEXT = "#212121"
MUTED = "#616161"

plt.rcParams.update({
    "font.size": 9.5,
    "axes.linewidth": 0.6,
    "mathtext.fontset": "dejavusans",
    "text.color": TEXT,
})


# ------------------------------------------------------- the 2-D stand-in ------
# Surface 1: a circle (a true signed-distance function).
C1, R1 = np.array([0.0, 0.0]), 1.0
# Surface 2: an offset, bumpy blob |x - c| - R(theta) that encloses surface 1 but is not
# concentric with it, so the paths bend. Its gradient norm is not 1, which is why
# the update divides by ||grad f||^2 instead of stepping a unit normal (the decoder is
# only approximately Eikonal too).
C2, R2, A2, TH0, B2, PH = np.array([0.45, 0.20]), 1.55, 0.12, 0.40, 0.05, 0.80


def f1(p):
    d = p - C1
    r = np.linalg.norm(d, axis=-1)
    return r - R1, d / r[..., None]


def f2(p):
    d = p - C2
    r = np.linalg.norm(d, axis=-1)
    th = np.arctan2(d[..., 1], d[..., 0])
    R = R2 * (1 + A2 * np.cos(2 * (th - TH0)) + B2 * np.cos(3 * th + PH))
    dR = R2 * (-2 * A2 * np.sin(2 * (th - TH0)) - 3 * B2 * np.sin(3 * th + PH))
    grad_r = d / r[..., None]
    grad_th = np.stack([-d[..., 1], d[..., 0]], axis=-1) / (r ** 2)[..., None]
    return r - R, grad_r - dR[..., None] * grad_th


def f(p, t):
    """Signed distance and its spatial gradient at 'latent' t in [0, 1]."""
    v1, g1 = f1(p)
    v2, g2 = f2(p)
    return (1 - t) * v1 + t * v2, (1 - t) * g1 + t * g2


def newton_project(p, t):
    """One Newton projection onto f(.; t) = 0 -- the pipeline's _project_once."""
    v, g = f(p, t)
    n2 = np.sum(g * g, axis=-1)
    return p - (v / n2)[..., None] * g


def flow(p0, n_steps=N_STEPS):
    """interpolate_common's loop: one projection per latent increment."""
    traj = [p0]
    p = p0
    for t in np.linspace(1 / n_steps, 1, n_steps):
        p = newton_project(p, t)
        traj.append(p)
    return np.stack(traj)  # (n_steps + 1, n_points, 2)


def level_set(ax, t, **kw):
    xs = np.linspace(-2.2, 3.0, 700)
    ys = np.linspace(-2.3, 2.5, 700)
    X, Y = np.meshgrid(xs, ys)
    F, _ = f(np.stack([X, Y], axis=-1), t)
    return ax.contour(X, Y, F, levels=[0.0], **kw)


# ---------------------------------------------------------------- panel A ------
LABELLED = {"i": 100.0, "j": 60.0, "k": 20.0}  # degrees on surface 1


def panel_a(ax):
    faint = [a for a in np.arange(0, 360, 18.0)
             if all(abs(a - v) > 8 for v in LABELLED.values())]
    angles = np.deg2rad(np.array(list(LABELLED.values()) + faint))
    p0 = C1 + R1 * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    traj = flow(p0)

    # Intermediate surfaces the points are projected onto along the way.
    for t in (0.25, 0.5, 0.75):
        level_set(ax, t, colors=[INTER], linewidths=0.55, linestyles="--", alpha=0.9)
    level_set(ax, 0.0, colors=[SURF1], linewidths=1.8)
    level_set(ax, 1.0, colors=[SURF2], linewidths=1.8)

    # Faint paths: every vertex flows, not just the three that are labelled.
    for n in range(len(LABELLED), traj.shape[1]):
        ax.plot(traj[:, n, 0], traj[:, n, 1], color=FAINT, lw=0.7, zorder=2)
        ax.plot(*traj[0, n], "o", ms=1.6, color=FAINT, zorder=3)
        ax.plot(*traj[-1, n], "o", ms=1.6, color=FAINT, zorder=3)

    # Labelled paths, drawn as the actual 100-step polylines with an arrowhead.
    start_off = {"i": (-0.17, 0.00), "j": (-0.15, -0.05), "k": (-0.05, -0.17)}
    end_off = {"i": (-0.02, 0.17), "j": (0.11, 0.13), "k": (0.17, 0.00)}
    for n, name in enumerate(LABELLED):
        c = PATH_COLORS[name]
        xy = traj[:, n]
        ax.plot(xy[:-1, 0], xy[:-1, 1], color=c, lw=1.5, zorder=4)
        ax.add_patch(FancyArrowPatch(xy[-3], xy[-1], arrowstyle="-|>", mutation_scale=9,
                                     color=c, lw=1.5, zorder=5, shrinkA=0, shrinkB=0))
        ax.plot(*xy[0], "o", ms=4.5, color=c, mec="white", mew=0.6, zorder=6)
        ax.plot(*xy[-1], "o", ms=4.5, color=c, mec="white", mew=0.6, zorder=6)
        ax.text(xy[0, 0] + start_off[name][0], xy[0, 1] + start_off[name][1], name,
                color=c, fontsize=11.5, fontweight="bold", ha="center", va="center", zorder=7)
        ax.text(xy[-1, 0] + end_off[name][0], xy[-1, 1] + end_off[name][1], name + "′",
                color=c, fontsize=11.5, fontweight="bold", ha="center", va="center", zorder=7)

    ax.text(*C1, "Surface 1\nlatent $\\mathbf{z}_1$", color=SURF1, fontsize=10,
            fontweight="bold", ha="center", va="center")
    ax.text(1.72, -1.08, "Surface 2\nlatent $\\mathbf{z}_2$", color=SURF2, fontsize=10,
            fontweight="bold", ha="left", va="center")
    ax.text(-2.1, 1.62, "intermediate surfaces\n$f(\\mathbf{x};\\mathbf{z}_t)=0$",
            color=MUTED, fontsize=9, ha="left", va="bottom", style="italic", linespacing=1.15)

    ax.set_aspect("equal", anchor="N")
    ax.set_xlim(-2.15, 3.0)
    ax.set_ylim(-1.42, 2.05)
    ax.axis("off")
    return traj, list(LABELLED)


def latent_bar(ax):
    """The latent path under panel A: z1 -> z2 in 100 increments, one tick each."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x0, x1, y = 0.20, 0.80, 0.62
    c1, c2 = np.array(mcolors.to_rgb(SURF1)), np.array(mcolors.to_rgb(SURF2))
    n_ticks = 30  # drawn ticks; the figure speaks of n increments, the caption gives n = 100
    for k, x in enumerate(np.linspace(x0, x1, n_ticks)):
        w = k / (n_ticks - 1)
        ax.plot([x, x], [y - 0.19, y + 0.19], color=(1 - w) * c1 + w * c2, lw=1.1,
                solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((x0 - 0.01, y), (x1 + 0.04, y), arrowstyle="-|>",
                                 mutation_scale=8, color="#424242", lw=0.8))
    ax.text(x0 - 0.03, y, "$\\mathbf{z}_1$", color=SURF1, fontsize=11.5, fontweight="bold",
            ha="right", va="center")
    ax.text(x1 + 0.065, y, "$\\mathbf{z}_2$", color=SURF2, fontsize=11.5, fontweight="bold",
            ha="left", va="center")
    ax.text(0.5, y - 0.32, "latent interpolated in $n$ increments:  "
            "$\\mathbf{z}_t$,  $t = 1/n,\\; 2/n,\\; \\ldots,\\; 1$",
            fontsize=8.5, ha="center", va="top", color="#424242")


# ---------------------------------------------------------------- panel B ------
def panel_b(ax, traj, names, which="k", t_prev=0.40, t_new=0.60):
    """One increment, exaggerated: the point sits on the previous surface, f and grad f are
    evaluated at the new latent, and one Newton step lands it on the new surface."""
    n = names.index(which)
    x = traj[int(round(t_prev * N_STEPS)), n]  # on the level set of t_prev
    v, g = f(x, t_new)
    gnorm = np.linalg.norm(g)
    step = -(v / gnorm ** 2) * g
    x_new = x + step
    u = g / gnorm                      # unit normal, for display
    perp = np.array([-u[1], u[0]])     # to the upper-left of the step

    level_set(ax, t_prev, colors=[INTER], linewidths=0.9, linestyles="--")
    level_set(ax, t_new, colors=[SURF2], linewidths=1.8)

    # The Newton step (thin, long) and the gradient direction (short, fat) share a
    # direction here: with f < 0 the step is along +grad f.
    ax.add_patch(FancyArrowPatch(x, x_new, arrowstyle="-|>", mutation_scale=10, color=STEP,
                                 lw=1.4, zorder=5, shrinkA=0, shrinkB=0))
    L = 0.10
    ax.add_patch(FancyArrowPatch(x, x + L * u, arrowstyle="simple,head_length=0.55,"
                                 "head_width=0.55,tail_width=0.18", mutation_scale=10,
                                 color=GRAD, lw=0, zorder=6, shrinkA=0, shrinkB=0))

    ax.plot(*x, "o", ms=5, color=PATH_COLORS[which], mec="white", mew=0.6, zorder=7)
    ax.plot(*x_new, "o", ms=5, mfc="white", mec=PATH_COLORS[which], mew=1.4, zorder=7)

    # Point labels sit on the side of each curve that is away from the other curve:
    # x to the inside (left) of the previous surface, x' to the outside (right) of the new.
    ax.text(x[0] - 0.022, x[1] - 0.012, "$\\mathbf{x}$", fontsize=12, ha="right", va="top",
            color=TEXT, fontweight="bold")
    ax.text(x[0] - 0.004, x[1] - 0.078, "on the\nprevious\nsurface", fontsize=8.5,
            ha="right", va="top", color=MUTED, linespacing=1.15)
    ax.text(x_new[0] + 0.022, x_new[1] + 0.012, "$\\mathbf{x}'$", fontsize=12, ha="left",
            va="bottom", color=TEXT, fontweight="bold")
    ax.text(x_new[0] + 0.022, x_new[1] + 0.072, "on the new\nsurface", fontsize=8.5,
            ha="left", va="bottom", color=MUTED, linespacing=1.15)

    g_anchor = x + 0.6 * L * u + 0.085 * perp + np.array([0.075, 0.02])
    ax.text(*g_anchor, "$\\nabla f(\\mathbf{x};\\mathbf{z}_t)$",
            color=GRAD, fontsize=10, ha="right", va="center")
    ax.text(*(g_anchor - np.array([0, 0.055])), "surface normal",
            color=GRAD, fontsize=8.5, ha="right", va="center")
    # The step's formula, parked outside the new surface with a thin leader to the arrow.
    mid = x + 0.5 * step
    ax.annotate("$-\\,\\dfrac{f(\\mathbf{x};\\mathbf{z}_t)\\,\\nabla f(\\mathbf{x};\\mathbf{z}_t)}"
                "{\\|\\nabla f(\\mathbf{x};\\mathbf{z}_t)\\|^{2}}$\nNewton step",
                xy=mid, xytext=(x_new[0] + 0.095, x_new[1] - 0.115), color=STEP, fontsize=10,
                ha="left", va="top", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=STEP, lw=0.5, shrinkA=1, shrinkB=2,
                                relpos=(0.0, 0.9)))

    ax.text(0.97, 0.965, "one increment,  $\\Delta t = 1/n$", transform=ax.transAxes,
            fontsize=8.5, color="#757575", ha="right", va="top", style="italic")
    # Which curve is which, as a legend below the frame so it never crosses the drawing.
    handles = [plt.Line2D([], [], color=INTER, lw=0.9, ls="--"),
               plt.Line2D([], [], color=SURF2, lw=1.8)]
    ax.legend(handles, ["previous surface,  $f(\\mathbf{x};\\mathbf{z}_{t-\\Delta t})=0$",
                        "new surface,  $f(\\mathbf{x};\\mathbf{z}_t)=0$"],
              loc="upper left", bbox_to_anchor=(-0.01, -0.005), frameon=False, ncol=1,
              fontsize=9, handlelength=1.8, labelspacing=0.25, borderaxespad=0)

    half = 0.30
    cx, cy = (x + x_new) / 2
    # The window sits a little to the right of the step so the formula fits inside it.
    x_lo, x_hi = cx - 0.95 * half, cx + 1.85 * half
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", anchor="N")
    for s in ax.spines.values():
        s.set_color("#9e9e9e")
    ax.set_xticks([])
    ax.set_yticks([])
    return (x_lo, cy - half), x_hi - x_lo, 2 * half



def step_strip(fig, y):
    """Draw the three numbered step captions on one line."""
    items = [
        (0.03, "left", "1", "Interpolate the latent"),
        (0.50, "center", "2", "Newton-project every point"),
        (0.97, "right", "3", "Read off correspondence"),
    ]
    for x, ha, num, text in items:
        fig.text(x, y, f"$\\bf{{{num}}}$   {text}", fontsize=9.5, ha=ha, va="center",
                 color=TEXT)


def main():
    fig = plt.figure(figsize=(7.2, 3.85))
    ax_a = fig.add_axes([0.01, 0.265, 0.515, 0.675])
    ax_bar = fig.add_axes([0.01, 0.15, 0.515, 0.115])
    ax_b = fig.add_axes([0.575, 0.265, 0.41, 0.675])

    traj, names = panel_a(ax_a)
    latent_bar(ax_bar)
    corner, w, h = panel_b(ax_b, traj, names)

    ax_a.add_patch(Rectangle(corner, w, h, fill=False, ec="#616161", lw=0.8,
                             ls=(0, (2, 1.5)), zorder=8))
    for (xa, ya), (xb, yb) in ((corner[0] + w, corner[1] + h), (0.0, 1.0)), \
                              ((corner[0] + w, corner[1]), (0.0, 0.0)):
        fig.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=ax_a.transData,
                                       xyB=(xb, yb), coordsB=ax_b.transAxes,
                                       color="#9e9e9e", lw=0.7, ls=(0, (2, 1.5)), zorder=0))

    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 1),
                    textcoords="offset points", fontsize=13, fontweight="bold",
                    ha="left", va="bottom", annotation_clip=False)

    step_strip(fig, y=0.055)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, facecolor="white", **kw)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
