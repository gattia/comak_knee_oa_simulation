"""``pins check`` compares the BLAS libraries a run loads, and accepts a library only when told its md5.

The environment's BLAS moves a simulation by about as much as a change of CPU type, and nothing else notices
it: an environment built from an unpinned spec can load OpenBLAS 0.3.34 and pass every other check.
A reader's own build of the JAM solver cannot have the paper's md5, so it needs an explicit override, and the
override names the exact file and md5 it accepts. None of these tests imports OpenSim.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comak_knee_oa import pins  # noqa: E402

PAPER_BLAS = {
    "libopenblasp-r0.3.21.so": pins.BLAS["libopenblasp-r0.3.21.so"],
    "libmkl_rt.so.2": pins.BLAS["libmkl_rt.so.2"],
    "libmkl_core.so.2": pins.BLAS["libmkl_core.so.2"],
    "libmkl_avx2.so.2": pins.BLAS["libmkl_avx2.so.2"],
    "libopenblasp-r0-01191904.3.27.so": pins.BLAS["libopenblasp-r0-01191904.3.27.so"],
}
OTHER_MD5 = "0123456789abcdef0123456789abcdef"


def test_the_paper_blas_passes():
    notes = []
    assert pins._blas_problems(dict(PAPER_BLAS), {}, notes, expect_solver=True) == []
    assert notes == []


def test_another_build_under_a_known_name_is_named():
    loaded = dict(PAPER_BLAS, **{"libmkl_core.so.2": OTHER_MD5})
    problems = pins._blas_problems(loaded, {}, [], expect_solver=True)
    assert problems == [f"libmkl_core.so.2 has md5 {OTHER_MD5} (want {pins.BLAS['libmkl_core.so.2']})"]


def test_the_cleanroom_environment_fails():
    # What an environment from the unpinned spec loaded: OpenBLAS 0.3.34 for numpy and the solver alike.
    loaded = {"libopenblasp-r0.3.34.so": OTHER_MD5,
              "libopenblasp-r0-01191904.3.27.so": PAPER_BLAS["libopenblasp-r0-01191904.3.27.so"]}
    problems = pins._blas_problems(loaded, {}, [], expect_solver=True)
    assert any("libopenblasp-r0.3.34.so" in p and "no BLAS library of that name" in p for p in problems)
    assert any("not run on MKL" in p for p in problems)
    assert any("solver's OpenBLAS" in p for p in problems)


def test_solver_openblas_is_only_required_when_opensim_was_loaded():
    loaded = {k: v for k, v in PAPER_BLAS.items() if k != "libopenblasp-r0.3.21.so"}
    assert pins._blas_problems(loaded, {}, [], expect_solver=False) == []
    assert pins._blas_problems(loaded, {}, [], expect_solver=True)


def test_an_accepted_library_passes_and_is_reported():
    loaded = dict(PAPER_BLAS, **{"libopenblasp-r0.3.21.so": OTHER_MD5})
    notes = []
    problems = pins._blas_problems(loaded, {"libopenblasp-r0.3.21.so": OTHER_MD5}, notes, expect_solver=True)
    assert problems == []
    assert len(notes) == 1 and "ACCEPTED" in notes[0] and OTHER_MD5 in notes[0]


def test_an_acceptance_names_one_md5_not_a_library():
    loaded = dict(PAPER_BLAS, **{"libopenblasp-r0.3.21.so": OTHER_MD5})
    accept = {"libopenblasp-r0.3.21.so": "f" * 32}
    assert pins._blas_problems(loaded, accept, [], expect_solver=True)


def test_accept_reads_the_environment_and_refuses_malformed_entries(monkeypatch):
    monkeypatch.setenv(pins.ACCEPT_ENV, f"libosimJAM.so={OTHER_MD5}, libfoo.so={'a' * 32}")
    assert pins.accepted() == {"libosimJAM.so": OTHER_MD5, "libfoo.so": "a" * 32}
    for bad in ("libosimJAM.so", "libosimJAM.so=xyz", f"={OTHER_MD5}", "libosimJAM.so=" + "G" * 32):
        monkeypatch.setenv(pins.ACCEPT_ENV, bad)
        with pytest.raises(ValueError):
            pins.accepted()
    monkeypatch.delenv(pins.ACCEPT_ENV)
    with pytest.raises(ValueError):
        pins.accepted({"libosimJAM.so": "short"})


def _fake_solver(monkeypatch, jam_md5: str):
    """Pinned libraries as pinned, an importable stand-in for opensim, and a loaded JAM with ``jam_md5``."""
    monkeypatch.setattr(pins, "_code_versions", lambda names: dict(pins.PINS))
    monkeypatch.setitem(sys.modules, "opensim", types.ModuleType("opensim"))
    monkeypatch.setattr(pins, "_loaded_native_libraries", lambda: {
        pins.JAM["library"]: {"path": "/opt/jam/libosimJAM.so", "md5": jam_md5}})
    monkeypatch.delenv(pins.ACCEPT_ENV, raising=False)


def test_a_jam_build_of_your_own_needs_the_override(monkeypatch):
    _fake_solver(monkeypatch, OTHER_MD5)
    problems = pins.check(check_blas=False)
    assert len(problems) == 1 and OTHER_MD5 in problems[0] and "--accept" in problems[0]

    notes = []
    assert pins.check(check_blas=False, accept={pins.JAM["library"]: OTHER_MD5}, notes=notes) == []
    assert len(notes) == 1 and "ACCEPTED" in notes[0] and pins.JAM["md5"] in notes[0]

    monkeypatch.setenv(pins.ACCEPT_ENV, f"{pins.JAM['library']}={OTHER_MD5}")
    notes = []
    assert pins.check(check_blas=False, notes=notes) == [] and notes


def test_the_paper_jam_needs_no_override(monkeypatch):
    _fake_solver(monkeypatch, pins.JAM["md5"])
    notes = []
    assert pins.check(check_blas=False, notes=notes) == [] and notes == []


def test_cli_prints_the_accepted_difference(monkeypatch, capsys):
    _fake_solver(monkeypatch, OTHER_MD5)
    assert pins.main(["check", "--strict", "--no-blas"]) == 1
    assert "PIN MISMATCH" in capsys.readouterr().err
    assert pins.main(["check", "--strict", "--no-blas", "--accept", f"{pins.JAM['library']}={OTHER_MD5}"]) == 0
    out = capsys.readouterr()
    assert "ACCEPTED" in out.err and "accepted difference" in out.out
    with pytest.raises(SystemExit):
        pins.main(["check", "--accept", "no-equals-sign"])


def test_blas_table_matches_the_locked_packages():
    """Every BLAS entry comes from a package the conda lock installs, at the build the comment names."""
    lock = (REPO_ROOT / "environment.lock.txt").read_text()
    assert "/libopenblas-0.3.21-pthreads_h78a6416_3." in lock
    assert "/mkl-2022.1.0-hc2b9512_224." in lock
    assert "/libblas-3.9.0-16_linux64_mkl." in lock
    assert all(len(m) == 32 for m in pins.BLAS.values())
