"""The environment files: the exact lock, and the readable spec that summarises it.

An environment built from an unpinned spec can run the pipeline on a different BLAS, which moves simulations by
about as much as a CPU-type change with every other check passing, or can lack packages the pinned libraries import
at load time. These tests keep the files consistent with each other and with what the docs promise.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK = (REPO_ROOT / "environment.lock.txt").read_text(encoding="utf-8")
PIP_LOCK = (REPO_ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
SPEC = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")

# Installed by hand (docs/environment.md), never from a lock.
HAND_INSTALLED = ("opensim", "mskt", "nsm", "nsosim", "pycomak", "pykneespm", "sqlite-jobqueue")
# Imported at load time by the pinned libraries or the derivations.
REQUIRED_PIP = {"nibabel": "5.3.2", "scikit-image": "0.24.0", "scikit-learn": "1.5.1", "wandb": "0.17.8",
                "spm1d": "0.4.33"}


def _pip_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def test_conda_lock_is_explicit_with_an_md5_per_file():
    lines = _pip_lines(LOCK)
    assert lines[0] == "@EXPLICIT"
    urls = lines[1:]
    assert len(urls) > 200
    for url in urls:
        assert re.fullmatch(r"https://conda\.anaconda\.org/(conda-forge|anaconda|nvidia|pytorch)/\S+\.(conda|tar\.bz2)"
                            r"#[0-9a-f]{32}", url), url


def test_pip_lock_pins_exact_versions_and_leaves_out_hand_installed():
    entries = dict(l.split("==") for l in _pip_lines(PIP_LOCK))
    norm = {re.sub(r"[-_.]+", "-", k).lower(): v for k, v in entries.items()}
    assert all(v and "@" not in v for v in entries.values())
    for name in HAND_INSTALLED + ("vtk", "pillow", "attrs"):
        assert name not in norm, name
    for name, version in REQUIRED_PIP.items():
        assert norm.get(name) == version, name


def test_spec_names_the_lock_versions():
    for name, version in REQUIRED_PIP.items():
        assert f"- {name}=={version}" in SPEC, name
    lock_files = [u.split("/")[-1] for u in _pip_lines(LOCK)[1:]]
    for spec_pin in ("libblas=3.9.0=16_linux64_mkl", "libopenblas=0.3.21=pthreads_h78a6416_3",
                     "mkl=2022.1.0=hc2b9512_224", "numpy=2.0.2=py39h9cb892a_0",
                     "vtk=9.2.5=osmesa_py39h899ec80_103"):
        assert spec_pin in SPEC, spec_pin
        name, version, build = spec_pin.split("=")
        assert any(f.startswith(f"{name}-{version}-{build}.") for f in lock_files), spec_pin


def test_spec_does_not_reinstall_vtk_from_pypi():
    # The paper ran conda-forge's VTK 9.2.5; a PyPI vtk in the pip block would replace its files.
    assert not re.search(r"^\s*- vtk==", SPEC, re.MULTILINE)
    # That VTK needs libOSMesa.so.8, which newer mesalib releases do not ship (an unpinned solve breaks `import vtk`).
    assert "- mesalib=23.0.2" in SPEC
