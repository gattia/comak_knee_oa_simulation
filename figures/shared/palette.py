"""Colors, group labels and axis wording for every manuscript figure.

One module owns all three, because they travel together: a figure that names a
group also colors it, and the two must agree across the whole manuscript.

Roles
-----
1. GROUP triad   Healthy / Incident / Mild OA        (Material blue / red / orange)
2. GAIT cohort   reference (black) + the nine measured gaits
3. SEQUENTIAL    pressure magnitude                  -> cividis
4. DIVERGING     signed difference / correlation / t -> RdBu_r
5. SECONDARY     early/late peak, medial/lateral, area accent, shape-axis gradient

Naming
------
``Incident`` -- not "progressor". These knees are KL0-1 at baseline and cross to
KL2 or worse by 48 months, which is *incident* radiographic osteoarthritis in the
Osteoarthritis Initiative's own terminology; "progressor" implies worsening of
disease that is already present.

``Mild OA`` -- not "KL2". A radiographic grade is not a reader-facing group name,
though the grade is still given in the long-form labels.

``Cartilage thickness change`` -- not "thinning rate". The quantity is signed, so
a negative "thinning rate" means thickening, a double negative that is easy to
misread.

The internal group keys (``kl0_1_progressor``, ``kl2``) are unchanged: they index data, appear in stored CSVs, and renaming them would break
every file on disk for no gain. Only what a reader sees is renamed here.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "GROUP_HEALTHY", "GROUP_INCIDENT", "GROUP_MILDOA",
    "GROUP_KEYS", "GROUP_COLORS", "GROUP_COLORS_SHORT_KEY",
    "GROUP_LABELS", "GROUP_LABELS_LONG", "GROUP_LABELS_SHORT_KEY",
    "OUTCOME_LABEL", "group_tuples",
    "GAIT_REFERENCE", "GAIT_MEASURED", "GAIT_PALETTE", "gait_color",
    "SEQUENTIAL_CMAP", "DIVERGING_CMAP", "SIGMA_CMAP",
    "PEAK_COLORS", "MEDLAT_COLORS", "ACCENT",
    "SPM_PAIR_COLORS",
]

# --------------------------------------------------------------- 1. GROUP ----
GROUP_HEALTHY = "#2196F3"   # blue
GROUP_INCIDENT = "#F44336"  # red
GROUP_MILDOA = "#FF9800"    # orange

# Canonical order, used wherever groups are drawn.
GROUP_KEYS = ("kl0_1_healthy", "kl0_1_progressor", "kl2")

GROUP_COLORS = {
    "kl0_1_healthy": GROUP_HEALTHY,
    "kl0_1_progressor": GROUP_INCIDENT,
    "kl2": GROUP_MILDOA,
}

# Short display labels: legends, rotated row labels, tick labels.
GROUP_LABELS = {
    "kl0_1_healthy": "Healthy",
    "kl0_1_progressor": "Incident",
    "kl2": "Mild OA",
}

# Grade-qualified labels: standalone legends where the KL grade has to be on the
# figure itself. Both registers are needed -- the short form exists because the
# long one overflows narrow panels, so they cannot be collapsed into one.
GROUP_LABELS_LONG = {
    "kl0_1_healthy": "Healthy (KL0–1)",
    "kl0_1_progressor": "Incident (KL0–1→2+)",
    "kl2": "Mild OA (KL2)",
}

# Some analyses key groups by short name rather than the full cohort key.
_SHORT = {"healthy": "kl0_1_healthy", "progressor": "kl0_1_progressor", "kl2": "kl2"}
GROUP_COLORS_SHORT_KEY = {s: GROUP_COLORS[k] for s, k in _SHORT.items()}
GROUP_LABELS_SHORT_KEY = {s: GROUP_LABELS[k] for s, k in _SHORT.items()}

# Outcome-axis wording. Compose with whatever unit the figure plots, e.g.
# f"{OUTCOME_LABEL} (µm/yr)".
OUTCOME_LABEL = "Cartilage thickness change"


def group_tuples(order=GROUP_KEYS, labels=None):
    """``[(key, label, color), ...]`` in canonical group order.

    This is the shape most figure scripts want, so they can write one line
    instead of a literal table that drifts. Pass ``labels=GROUP_LABELS_LONG``
    for the grade-qualified variant.
    """
    lab = labels or GROUP_LABELS
    return [(k, lab.get(k, k), GROUP_COLORS[k]) for k in order]


# ---------------------------------------------------------------- 2. GAIT ----
# The reference (overground) condition is reserved black so it stands out among
# the nine measured (treadmill) gaits -- it differs from them in protocol, speed
# and population, and several analyses exclude it.
GAIT_REFERENCE = "#000000"
GAIT_MEASURED = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#00897B",  # teal
    "#FF9800",  # orange
    "#7E57C2",  # purple
    "#43A047",  # green
    "#D81B60",  # magenta
    "#795548",  # brown
    "#00BCD4",  # cyan
]
GAIT_PALETTE = [GAIT_REFERENCE] + GAIT_MEASURED


def gait_color(i, include_reference=True):
    """Color for gait index ``i`` (0 = reference unless ``include_reference``)."""
    pal = GAIT_PALETTE if include_reference else GAIT_MEASURED
    return pal[i % len(pal)]


# ------------------------------------------------- 3/4. CONTINUOUS SCALES ----
SEQUENTIAL_CMAP = "cividis"   # pressure magnitude
DIVERGING_CMAP = "RdBu_r"     # signed difference, correlation, t statistic

# ----------------------------------------------------------- 5. SECONDARY ----
PEAK_COLORS = {1: "#00897B", 2: "#8E24AA"}                   # early / late stance
MEDLAT_COLORS = {"medial": "#009E73", "lateral": "#A6761D"}
ACCENT = "#37474F"

# The three group pairs of the SPM contrasts, in display order, each with a
# colour distinguishable from the group triad above.
SPM_PAIR_COLORS = {
    ("kl0_1_healthy", "kl0_1_progressor"): "#000000",
    ("kl0_1_healthy", "kl2"): "#7E57C2",
    ("kl0_1_progressor", "kl2"): "#00897B",
}

# Shape-axis gradient for the synthetic morph (Figure 3). Its endpoints are the
# group colors, so the axis reads as "healthy anatomy -> mild-OA anatomy" without
# needing a separate key.
SIGMA_CMAP = LinearSegmentedColormap.from_list(
    "healthy_mildoa", [GROUP_HEALTHY, "#e8e8e8", GROUP_MILDOA]
)
