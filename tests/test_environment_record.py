"""Run records name the numerical libraries that ran, and any accepted override."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comak_knee_oa import manifest, pins  # noqa: E402


def test_the_override_variable_is_spelled_the_same_in_both_modules():
    # manifest cannot import pins (pins imports manifest), so it keeps its own copy.
    assert manifest._PINS_ACCEPT_ENV == pins.ACCEPT_ENV


def test_an_accepted_override_is_recorded(monkeypatch):
    monkeypatch.setenv(pins.ACCEPT_ENV, "libosimJAM.so=0123abcd")
    assert manifest.environment_record()["pins_accept"] == "libosimJAM.so=0123abcd"
    monkeypatch.delenv(pins.ACCEPT_ENV)
    assert manifest.environment_record()["pins_accept"] is None


def test_mkl_libraries_are_recorded_with_their_md5():
    import numpy as np

    np.dot(np.ones((64, 64)), np.ones((64, 64)))  # make the BLAS load its CPU kernels
    native = manifest.environment_record()["native_libraries"]
    mkl = {name: entry for name, entry in native.items() if name.startswith("libmkl_")}
    for name, entry in mkl.items():
        assert len(entry.get("md5", "")) == 32, name
        assert os.path.isfile(entry["path"]), name
