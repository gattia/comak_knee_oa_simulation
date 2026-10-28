#!/usr/bin/env python3
"""Per-gait knee waveforms for Figure 6's gait row.

    python pipeline/derive_figure6_gait_waveforms.py
    python pipeline/derive_figure6_gait_waveforms.py --max-knees 100
    python pipeline/derive_figure6_gait_waveforms.py --time-base frame   # the v2 run's axis

Writes ``04_results/figure6_loading_determinants/gait_waveforms.csv`` (long:
gait, variable, pct_stance, median, q25, q75, n_knees) and
``gait_waveforms_meta.json``. The figure reads only these.

What is summarised
------------------
For every knee simulated under each of the nine measured gait conditions:

* ``knee_flexion_deg`` -- the knee flexion angle through stance, from the COMAK
  values file. It is a prescribed (primary) coordinate, so it is the donor's
  own kinematics and is the same for every knee in a condition.
* ``kfm_ext_nm`` / ``kam_ext_nm`` -- the external knee flexion and adduction
  moments from inverse dynamics, i.e. the negated stored generalised forces (the
  same convention ``derive_figure6_data.kam_peaks`` uses). Every knee in a
  condition is scaled to the same donor body and driven by the same ground
  reaction forces, so these vary only slightly between knees.

Each trace is placed on a 101-point stance axis and summarised per condition as
the median with the inter-quartile range over knees. ``--time-base time`` (the
default) places each frame at its time in the stance window the run recorded
(``derive_figure2_data.stance_window``), so 0 % is heel strike whatever the
file's first frame; a padded run's files start a frame before it.
``--time-base frame`` normalises each file to its own time span, as in the v2
run. Knees are taken from the condition's own
simulation tree, asserted non-empty (see ``derive_figure6_data.condition_knees``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comak_knee_oa.paths import stage_dir  # noqa: E402
from derive_figure2_data import TIME_BASES, stance_percent, stance_window  # noqa: E402
from derive_figure6_data import (  # noqa: E402
    FIGURE, condition_knees, gait_conditions, read_sto_column, results_dir,
)

N_GRID = 101
VARIABLES = {
    # variable -> (relative file, column, sign, label)
    "knee_flexion_deg": ("comak/_values.sto", "knee_flex_r", +1.0, "Knee flexion angle (°)"),
    "kfm_ext_nm": ("comak-inverse-dynamics/inverse-dynamics.sto", "knee_flex_r_moment", -1.0,
                   "External knee flexion moment (N·m)"),
    "kam_ext_nm": ("comak-inverse-dynamics/inverse-dynamics.sto", "knee_add_r_moment", -1.0,
                   "External knee adduction moment (N·m)"),
}


def resample(times, values, window=None, path=None):
    """``values`` on the 0-100 % stance grid.

    ``window`` (the run's recorded stance, ``stance_window``) places each frame at
    its time in stance; where the grid runs past the kept frames the trace holds
    its first or last value. None normalises to the file's own span (the v2 run's
    axis).
    """
    if window is None:
        pct = (times - times[0]) / max(times[-1] - times[0], 1e-9) * 100.0
    else:
        pct = stance_percent(times, window, path)
    return np.interp(np.linspace(0, 100, N_GRID), pct, values)


def collect(max_knees: int | None, time_base: str = "time"):
    if time_base not in TIME_BASES:
        raise SystemExit(f"--time-base must be one of {TIME_BASES}, not {time_base!r}")
    out_dir = stage_dir("results", FIGURE, create=True)
    rows, meta = [], {"n_grid": N_GRID, "sign": "external moments = -(stored generalised force)",
                      "time_base": time_base,
                      "variables": {k: v[3] for k, v in VARIABLES.items()}, "n_knees": {}}
    for gait, template in gait_conditions():
        knees = condition_knees(template, gait)
        if max_knees:
            knees = knees[:max_knees]
        stacks = {v: [] for v in VARIABLES}
        for knee in knees:
            directory = results_dir(template, knee)
            window = None
            for var, (rel, column, sign, _) in VARIABLES.items():
                read = read_sto_column(directory / rel, column)
                if read is None:
                    continue
                # Only a trace that exists is placed: a failed simulation's folder can hold
                # no record of its stance window, and is skipped as frame mode skips it.
                if time_base == "time" and window is None:
                    window = stance_window(directory)
                stacks[var].append(sign * resample(*read, window=window, path=directory / rel))
        counts = {v: len(s) for v, s in stacks.items()}
        if min(counts.values()) == 0:
            raise RuntimeError(f"{gait}: no waveforms read for {counts}")
        meta["n_knees"][gait] = counts
        print(f"  {gait:13s} knees read: {counts}")
        grid = np.linspace(0, 100, N_GRID)
        for var, stack in stacks.items():
            arr = np.vstack(stack)
            med, q25, q75 = (np.nanpercentile(arr, q, axis=0) for q in (50, 25, 75))
            for i in range(N_GRID):
                rows.append(dict(gait=gait, variable=var, pct_stance=float(grid[i]),
                                 median=float(med[i]), q25=float(q25[i]), q75=float(q75[i]),
                                 n_knees=int(arr.shape[0])))
    path = out_dir / "gait_waveforms.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (out_dir / "gait_waveforms_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-knees", type=int, default=None, help="subsample knees per condition (default: all)")
    parser.add_argument("--time-base", choices=list(TIME_BASES), default="time",
                        help="place frames by their time in the recorded stance window (default), "
                             "or by each file's own span as v2 did")
    args = parser.parse_args()
    collect(args.max_knees, args.time_base)


if __name__ == "__main__":
    main()
