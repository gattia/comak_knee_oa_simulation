"""Figure 6 places its inverse-dynamics frames in stance by time.

A measured-gait run is padded by half a frame each side (``comak_window_pad_s``), and its
inverse-dynamics file starts a whole frame before heel strike. Splitting the file at
``len // 2`` would move the split with the pad, and for a donor whose adduction moment is
still falling at mid-stance "peak 2" (the maximum over the second half) would move with it.
``--time-base time`` splits at 50 % of the stance window the run recorded; ``--time-base
frame`` splits by index, as the v2 run did.

The synthetic run below copies the frame layout of a real v3 Subject_109 simulation: 73
frames at 0.01 s from one step before heel strike, the toe-off frame stamped 1e-4 s past
the recorded stop (AddBiomechanics' 100.008 Hz time stamps), so the index split falls at
49.3 % of stance and the time split at 50 %.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "pipeline", "figures"):
    sys.path.insert(0, str(REPO_ROOT / sub))

fig7 = pytest.importorskip("derive_figure6_data")
waveforms = pytest.importorskip("derive_figure6_gait_waveforms")

START, STOP, PAD, STEP = 10.0, 10.71, 0.005, 0.01


def external_kam(percent):
    """Rises to 30 N·m at 25 %, falls to 10 at 60 %, a lower late hump (8) at 80 %.

    Rounded to 6 decimals so the storage file (9 decimals) holds it exactly.
    """
    early = 30.0 * np.exp(-((percent - 25.0) / 12.0) ** 2)
    fall = np.interp(percent, [25, 60, 100], [30.0, 10.0, 0.0])
    late = 8.0 * np.exp(-((percent - 80.0) / 6.0) ** 2)
    return np.round(np.where(percent <= 25, early, np.maximum(fall, late)), 6)


def make_run(root: Path, times, kam, pad=PAD) -> Path:
    """A results directory with a run record and an inverse-dynamics file."""
    (root / "comak-inverse-dynamics").mkdir(parents=True)
    record = {"timing": {"start_time_comak": START - pad, "stop_time_comak": STOP + pad,
                         "comak_window_pad_s": pad}}
    (root / "run_record.json").write_text(json.dumps(record))
    rows = "\n".join(f"{t:.6f}\t{-k:.9f}\t0.0" for t, k in zip(times, kam))  # stored = internal
    (root / "comak-inverse-dynamics" / "inverse-dynamics.sto").write_text(
        "Inverse Dynamics\nendheader\ntime\tknee_add_r_moment\tknee_flex_r_moment\n" + rows + "\n")
    return root / "comak-inverse-dynamics" / "inverse-dynamics.sto"


def padded_times():
    times = START - STEP + STEP * np.arange(73)  # one step before heel strike ... toe-off
    times[-1] += 1e-4                            # toe-off stamped just past the recorded stop
    return times


def percent(times):
    return (times - START) / (STOP - START) * 100.0


def test_frame_base_reproduces_v2_split(tmp_path):
    times = padded_times()
    kam = external_kam(percent(times))
    sto = make_run(tmp_path / "run", times, kam)
    half = len(kam) // 2
    assert fig7.kam_peaks(sto, "frame") == (float(kam[:half].max()), float(kam[half:].max()))


def test_time_base_splits_at_half_of_the_recorded_stance(tmp_path):
    times = padded_times()
    pct = percent(times)
    kam = external_kam(pct)
    sto = make_run(tmp_path / "run", times, kam)

    p1, p2 = fig7.kam_peaks(sto, "time")
    assert p1 == pytest.approx(kam[(pct >= 0) & (pct < 50)].max(), abs=0)
    # Still falling at mid-stance: peak 2 is the first frame at or after 50 %.
    first_late = np.flatnonzero(pct >= 50)[0]
    assert p2 == pytest.approx(kam[first_late], abs=0)
    # The index split put the 49.3 % frame in the second half; time does not.
    half = len(kam) // 2
    assert pct[half] < 50 <= pct[half + 1]
    assert fig7.kam_peaks(sto, "frame")[1] == pytest.approx(kam[half], abs=0)
    assert p2 < kam[half]


def test_time_base_does_not_depend_on_the_pad(tmp_path):
    """The same stance, padded (v3) or not (v2's layout), gives the same peaks by time."""
    padded = padded_times()
    unpadded = padded[1:]  # no frame before heel strike
    sto_padded = make_run(tmp_path / "padded", padded, external_kam(percent(padded)))
    sto_unpadded = make_run(tmp_path / "unpadded", unpadded, external_kam(percent(unpadded)), pad=0.0)
    assert fig7.kam_peaks(sto_padded, "time") == fig7.kam_peaks(sto_unpadded, "time")
    assert fig7.kam_peaks(sto_padded, "frame") != fig7.kam_peaks(sto_unpadded, "frame")


def test_frames_outside_stance_are_dropped_and_toe_off_is_kept(tmp_path):
    times = padded_times()
    kam = external_kam(percent(times))
    kam[0] = 99.0    # the pre-heel-strike frame, a whole step outside stance
    kam[-1] = 20.0   # the toe-off frame, 1e-4 s past the recorded stop
    p1, p2 = fig7.kam_peaks(make_run(tmp_path / "run", times, kam), "time")
    assert p1 < 99.0
    assert p2 == 20.0


def test_frames_from_another_window_raise(tmp_path):
    times = padded_times() + 0.5  # not this run's stance
    sto = make_run(tmp_path / "run", times, external_kam(percent(times - 0.5)))
    with pytest.raises(ValueError, match="recorded stance window"):
        fig7.kam_peaks(sto, "time")


def test_unknown_time_base_is_refused(tmp_path):
    times = padded_times()
    sto = make_run(tmp_path / "run", times, external_kam(percent(times)))
    with pytest.raises(ValueError, match="time_base"):
        fig7.kam_peaks(sto, "index")


def test_waveform_axis_starts_at_heel_strike_by_time():
    """By time 0 % is heel strike; by frame it is the file's first (pre-heel-strike) frame."""
    times = padded_times()
    truth = percent(times)  # a trace whose value is its own stance percent
    by_time = waveforms.resample(times, truth, window=(START, STOP))
    assert np.allclose(by_time, np.linspace(0, 100, waveforms.N_GRID), atol=1e-9)
    by_frame = waveforms.resample(times, truth)
    assert by_frame[0] == pytest.approx(truth[0])  # ≈ -1.4 %
    assert by_frame[0] < -1.0


def test_waveforms_skip_a_failed_simulation_with_no_stance_record(tmp_path, monkeypatch):
    """A knee folder with no traces and no run record is skipped by time, as by frame."""
    good = tmp_path / "good"
    times = padded_times()
    make_run(good, times, external_kam(percent(times)))
    (good / "comak").mkdir()
    (good / "comak" / "_values.sto").write_text(
        "values\nendheader\ntime\tknee_flex_r\n" + "\n".join(f"{t:.6f}\t0.5" for t in times) + "\n")
    failed = tmp_path / "failed"
    (failed / "comak").mkdir(parents=True)  # empty, as a v2 simulation that never ran COMAK
    dirs = {"good": good, "failed": failed}
    monkeypatch.setattr(waveforms, "gait_conditions", lambda: [("Subject_102", "x/{knee}")])
    monkeypatch.setattr(waveforms, "condition_knees", lambda template, gait: ["good", "failed"])
    monkeypatch.setattr(waveforms, "results_dir", lambda template, knee: dirs[knee])
    monkeypatch.setattr(waveforms, "stage_dir", lambda *a, **k: tmp_path)
    waveforms.collect(None, "time")
    meta = __import__("json").loads((tmp_path / "gait_waveforms_meta.json").read_text())
    assert meta["n_knees"]["Subject_102"] == {"knee_flexion_deg": 1, "kfm_ext_nm": 1, "kam_ext_nm": 1}
