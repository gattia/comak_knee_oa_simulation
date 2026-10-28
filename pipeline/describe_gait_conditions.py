"""Characterise the ten gait conditions from their own measured data.

The paper simulates every knee under ten gait conditions and compares them, so
how those conditions differ from one another is part of the result, not
background. This derives that from the marker trajectories and ground reaction
forces themselves rather than from filenames or study documentation, which can
be wrong.

What it reports, and why each is measured rather than assumed:

Protocol. During foot-flat the foot does not move relative to the surface it
is on. So if the surface is still the marker is still, and if the surface is a
belt the marker travels at belt speed. One number separates overground from
treadmill, and on a treadmill it *is* the walking speed. Corroborated by the
number of consecutive strides in a trial: a walkway with force plates yields
one or two, an instrumented treadmill yields twenty.

Speed. From belt speed on a treadmill. Overground, from how far the centre
of pressure advances between successive footfalls of the same foot, divided by
the stride time.

Body weight. Not read from the model but derived: over one complete stride,
mean total vertical ground reaction force equals body weight, because the body
returns to the same state and the net vertical impulse is zero. This is the
check that decides whether a model's mass and its force data belong together.

Load. Peak vertical force in body weights, using the body weight derived
above, so that a mass recorded in the wrong units or from the wrong participant
cannot rescale the comparison unnoticed.

It prints one row per condition and the speed-versus-load split; ``--json PATH``
also writes the rows. Run it with no arguments::

    python pipeline/describe_gait_conditions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa import cohorts  # noqa: E402
from comak_knee_oa.config import load_config  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402
from simulate_gait import read_storage, stance_windows  # noqa: E402

GRAVITY = 9.81

# Below this, the foot is not being carried anywhere and the surface is still.
# Well clear of both populations observed: overground gives ~0.01 m/s, treadmill
# 0.7 m/s and up.
STATIONARY_GROUND_MS = 0.30

# The reference trial names its columns and markers by a different convention
# from the measured trials.
REFERENCE = {
    "name": "reference",
    "directory": "data/reference_data/motion_analysis",
    "grf": "overground_17_grf.mot",
    "markers": "overground_17.trc",
    "right_vertical": "ground_force_2_vy",
    "plates": {"right": 2, "left_first": 1, "left_second": 3},
    "foot_markers": ("R.Heel", "R.Ankle"),
}
MEASURED_FOOT_MARKERS = ("r_calc", "r_ankle", "R_calc", "R_ankle")


def read_markers(path: Path) -> tuple[list[str], np.ndarray, float]:
    """Marker names, frames, and the factor converting to metres.

    Units come from the file's own header. The two datasets here disagree --
    one writes millimetres and the other metres -- and guessing from magnitude
    produces a plausible number rather than an error.
    """
    lines = Path(path).read_text().splitlines()
    units = lines[2].split("\t")[4].strip()
    header = next(i for i, line in enumerate(lines) if line.startswith("Frame#"))
    names = [
        n.strip()
        for n in lines[header].split("\t")
        if n.strip() and not n.startswith(("Frame", "Time"))
    ]
    frames = np.array(
        [
            [float(x) if x.strip() else np.nan for x in line.split("\t")]
            for line in lines[header + 2 :]
            if line.split("\t")[0].strip().isdigit()
        ]
    )
    return names, frames, 0.001 if units == "mm" else 1.0


def foot_travel_speed(markers: Path, stances, candidates) -> float | None:
    """How fast the foot is carried during foot-flat, in m/s.

    Sampled over the middle of stance only. Early stance is still rolling onto
    the heel and late stance is pushing off, and in both the marker moves for
    reasons that have nothing to do with the surface.
    """
    names, frames, to_metres = read_markers(markers)
    index = next((names.index(n) for n in candidates if n in names), None)
    if index is None:
        return None
    time = frames[:, 1]
    position = frames[:, 2 + 3 * index : 5 + 3 * index] * to_metres

    speeds = []
    for start, stop in stances[:8]:
        span = stop - start
        window = (time >= start + 0.20 * span) & (time <= stop - 0.35 * span)
        if window.sum() < 4 or not np.isfinite(position[window]).all():
            continue
        travelled = np.linalg.norm(position[window][-1] - position[window][0])
        speeds.append(travelled / (time[window][-1] - time[window][0]))
    return float(np.median(speeds)) if speeds else None


def body_weight_from_grf(columns, values, start: float, stop: float) -> float:
    """Body weight in newtons, from the impulse over one stride.

    Over a complete cycle the body starts and ends in the same state, so the
    vertical forces on it must average out to its weight. Summing every plate
    matters: a single plate only sees one foot.
    """
    time = values[:, columns.index("time")]
    vertical = sum(
        values[:, i] for i, name in enumerate(columns) if name.endswith("_vy")
    )
    window = (time >= start) & (time < stop)
    return float(vertical[window].mean())


def describe_reference() -> dict:
    directory = stage_dir("inputs", "reference") / REFERENCE["directory"]
    grf = directory / REFERENCE["grf"]
    columns, values = read_storage(grf)
    time = values[:, columns.index("time")]

    def first_contact(plate: int) -> tuple[float, float]:
        vertical = values[:, columns.index(f"ground_force_{plate}_vy")]
        loaded = np.where(vertical > 10)[0]
        return float(time[loaded[0]]), float(time[loaded[-1]])

    right = first_contact(REFERENCE["plates"]["right"])
    left_first = first_contact(REFERENCE["plates"]["left_first"])
    left_second = first_contact(REFERENCE["plates"]["left_second"])
    stride = left_second[0] - left_first[0]

    # Overground: the body travels, so speed comes from how far successive
    # footfalls of the same foot are apart.
    def mid_stance_position(plate: int) -> np.ndarray:
        vertical = values[:, columns.index(f"ground_force_{plate}_vy")]
        loaded = np.where(vertical > 50)[0]
        k = loaded[len(loaded) // 2]
        return np.array(
            [values[k, columns.index(f"ground_force_{plate}_p{axis}")] for axis in "xyz"]
        )

    stride_length = float(
        np.linalg.norm(
            mid_stance_position(REFERENCE["plates"]["left_second"])
            - mid_stance_position(REFERENCE["plates"]["left_first"])
        )
    )
    weight = body_weight_from_grf(columns, values, left_first[0], left_second[0])
    peak = float(
        values[:, columns.index(REFERENCE["right_vertical"])][
            (time >= right[0]) & (time <= right[1])
        ].max()
    )
    carried = foot_travel_speed(
        directory / REFERENCE["markers"], [right], REFERENCE["foot_markers"]
    )
    return {
        "condition": REFERENCE["name"],
        "n_stance": 1,
        "carried_ms": carried,
        "protocol": "overground" if carried < STATIONARY_GROUND_MS else "treadmill",
        "speed_ms": stride_length / stride,
        "stride_s": stride,
        "cadence": 120 / stride,
        "stance_s": right[1] - right[0],
        "stance_pct": (right[1] - right[0]) / stride * 100,
        "body_weight_N": weight,
        "implied_mass_kg": weight / GRAVITY,
        "peak_BW": peak / weight,
    }


def describe_measured(config_path: Path) -> dict | None:
    config = load_config(config_path)
    gait = config.get("run", "gait")
    section = "measured_gait"
    grf = stage_dir("inputs") / config.get(section, "grf").split("/", 1)[1]
    markers = stage_dir("inputs") / config.get(section, "markers").split("/", 1)[1]
    vertical_column = config.get(section, "grf_vertical_column")
    threshold = config.get(section, "force_threshold_N", 10.0)
    min_frames = config.get(section, "min_stance_frames", 40)

    stances = stance_windows(grf, vertical_column, threshold, min_frames)
    if len(stances) < 2:
        return None
    columns, values = read_storage(grf)
    time = values[:, columns.index("time")]
    vertical = values[:, columns.index(vertical_column)]

    # stance_windows yields (start, stop, n_frames, peak); only the times are
    # wanted here.
    spans = [(s[0], s[1]) for s in stances]
    heel_strikes = np.array([a for a, _ in spans])
    strides = np.diff(heel_strikes)
    strides = strides[strides < 2.0]  # gaps between bouts are not strides
    stride = float(np.median(strides))
    stance = float(np.median([b - a for a, b in spans]))

    # Body weight over a whole stride, from the first stance that has a
    # successor -- so the window is one whole cycle.
    weight = body_weight_from_grf(columns, values, heel_strikes[0], heel_strikes[0] + stride)
    peak = float(
        np.median([vertical[(time >= a) & (time <= b)].max() for a, b in spans])
    )
    carried = foot_travel_speed(markers, spans, MEASURED_FOOT_MARKERS)
    return {
        "condition": gait,
        "n_stance": len(stances),
        "carried_ms": carried,
        "protocol": "overground" if carried < STATIONARY_GROUND_MS else "treadmill",
        "speed_ms": carried,
        "stride_s": stride,
        "cadence": 120 / stride,
        "stance_s": stance,
        "stance_pct": stance / stride * 100,
        "body_weight_N": weight,
        "implied_mass_kg": weight / GRAVITY,
        "peak_BW": peak / weight,
    }


def report(rows: list[dict]) -> None:
    head = (f"{'condition':16} {'protocol':>11} {'n':>4} {'speed':>7} {'cadence':>8} "
            f"{'stance s':>9} {'stance %':>9} {'mass kg':>8} {'peak/BW':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['condition']:16} {r['protocol']:>11} {r['n_stance']:>4} "
            f"{r['speed_ms']:7.3f} {r['cadence']:8.1f} {r['stance_s']:9.4f} "
            f"{r['stance_pct']:9.1f} {r['implied_mass_kg']:8.2f} {r['peak_BW']:8.2f}"
        )

    reference, measured = rows[0], rows[1:]
    if not measured:
        return
    print("-" * len(head))
    print("\nreference against the measured conditions:")
    for key, label in (
        ("stride_s", "stride time (s)"),
        ("cadence", "cadence (/min)"),
        ("stance_s", "stance (s)"),
        ("stance_pct", "stance (% stride)"),
        ("speed_ms", "speed (m/s)"),
        ("peak_BW", "peak load (BW)"),
    ):
        values = np.array([r[key] for r in measured])
        z = (reference[key] - values.mean()) / values.std()
        inside = values.min() <= reference[key] <= values.max()
        print(
            f"  {label:18} measured {values.min():7.3f}-{values.max():7.3f} "
            f"(mean {values.mean():7.3f})   reference {reference[key]:7.3f}"
            f"  {z:+5.1f} SD   {'inside range' if inside else 'OUTSIDE RANGE'}"
        )

    speed = np.array([r["speed_ms"] for r in measured])
    load = np.array([r["peak_BW"] for r in measured])
    slope, intercept = np.polyfit(speed, load, 1)
    r_value = float(np.corrcoef(speed, load)[0, 1])
    residual_sd = float((load - (intercept + slope * speed)).std(ddof=2))
    predicted = intercept + slope * reference["speed_ms"]
    gap = reference["peak_BW"] - load.mean()
    print(
        f"\nhow much of the load difference is speed?\n"
        f"  within the measured conditions: peak/BW = {intercept:.3f} + {slope:.3f} * speed"
        f"   (r = {r_value:.2f}, n = {len(measured)}, residual SD {residual_sd:.3f})\n"
        f"  at the reference's {reference['speed_ms']:.3f} m/s that predicts {predicted:.3f} BW;"
        f" observed {reference['peak_BW']:.3f} BW\n"
        f"  speed explains {(predicted - load.mean()) / gap * 100:.0f}% of the "
        f"{gap:+.3f} BW gap; the rest is protocol and population, which are\n"
        f"  confounded here -- every treadmill condition is a gait-retraining participant and the\n"
        f"  only overground one is the reference. n is small; read this as a split, not an estimate."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="describe_gait_conditions.py",
        description="Measure protocol, speed, timing and load for every gait condition.",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the rows here")
    args = parser.parse_args(argv)

    rows = [describe_reference()]
    # The data root's measured donors, not every file in config/measured_gait/: that folder also
    # holds configs for donors another run used, whose inputs this root may lack (v3 uses
    # Subject_121, v2 used RSubject_121).
    config_dir = Path(__file__).resolve().parents[1] / "config" / "measured_gait"
    for name in cohorts.measured():
        row = describe_measured(config_dir / f"{name}.json")
        if row:
            rows.append(row)

    report(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
