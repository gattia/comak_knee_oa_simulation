"""The AddBiomechanics engine's packaging (``tools/addbiomechanics``) and its use by ``run_addbiomechanics.py run``.

The engine needs two things its own environment does not have: OpenSim's ``opensim-cmd``, which writes every
scaled model it outputs, and an importable ``opensim`` module, which it imports but does not use. Both come from
``tools/addbiomechanics``, and ``run`` refuses an engine, nimblephysics or ``opensim-cmd`` other than the ones that
scaled the paper's donors. These tests pin that wiring; none needs the engine itself.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import run_addbiomechanics as rab  # noqa: E402

TOOLS = REPO_ROOT / "tools" / "addbiomechanics"
PAPER_VERSION = rab.ENGINE_PINS["opensim_cmd"]


def _fake_opensim_prefix(root: Path, version: str = PAPER_VERSION) -> Path:
    """A prefix whose ``opensim-cmd`` only reports a version."""
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    exe = root / "bin" / "opensim-cmd"
    exe.write_text(f'#!/bin/sh\necho "OpenSim version {version}, build date 00:00:00 Jan  1 2025"\n')
    exe.chmod(0o755)
    return root


def test_pins_are_the_ones_the_docs_and_environment_name():
    readme = (TOOLS / "README.md").read_text(encoding="utf-8")
    env = (TOOLS / "environment.yml").read_text(encoding="utf-8")
    assert rab.ENGINE_PINS["engine_commit"] in readme
    assert rab.ENGINE_PINS["opensim_cmd"] in readme
    assert f"nimblephysics=={rab.ENGINE_PINS['nimblephysics']}" in env
    assert re.fullmatch(r"[0-9a-f]{40}", rab.ENGINE_PINS["engine_commit"])


def test_engine_environment_takes_opensim_cmd_and_the_stub_from_tools(tmp_path):
    env = rab.engine_environment(tmp_path, base={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path / "elsewhere")})
    assert env["PATH"].split(os.pathsep)[0] == str(TOOLS / "bin")
    # The whole PYTHONPATH, not prepended: nothing from the caller's environment reaches the engine.
    assert env["PYTHONPATH"] == str(TOOLS / "stub")
    assert env["AB_OPENSIM_PREFIX"] == str(tmp_path)


def test_shim_runs_the_prefix_opensim_cmd(tmp_path):
    prefix = _fake_opensim_prefix(tmp_path / "osim")
    env = rab.engine_environment(prefix, base={"PATH": "/usr/bin:/bin"})
    assert rab._opensim_cmd_version(env) == PAPER_VERSION


def test_shim_refuses_without_a_prefix():
    out = subprocess.run([str(TOOLS / "bin" / "opensim-cmd"), "--version"], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"})
    assert out.returncode != 0
    assert "AB_OPENSIM_PREFIX" in out.stderr


def test_stub_imports_and_fails_loudly_on_use():
    code = ("import opensim\n"
            "try:\n    opensim.TimeSeriesTable\nexcept AttributeError as e:\n    print('raised', e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(TOOLS / "stub")})
    assert out.stdout.startswith("raised opensim.TimeSeriesTable")


def test_pin_problems_names_each_difference():
    paper = {"engine_commit": rab.ENGINE_PINS["engine_commit"], "engine_dirty": False,
             "nimblephysics": rab.ENGINE_PINS["nimblephysics"], "opensim_cmd": PAPER_VERSION}
    assert rab.pin_problems(paper) == []
    # The channel's later build writes the same scaled models, so it is accepted too.
    assert rab.pin_problems(dict(paper, opensim_cmd="4.5.2-2025-05-02-fcdbeacb2")) == []
    other = dict(paper, engine_commit="0" * 40, engine_dirty=True, opensim_cmd="4.6")
    problems = rab.pin_problems(other)
    assert len(problems) == 3
    assert any("engine commit" in p for p in problems)
    assert any("uncommitted" in p for p in problems)
    assert any("opensim_cmd 4.6" in p for p in problems)


def _staged(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "_subject.json").write_text("{}\n")
    return run_dir


def _engine_checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "AddBiomechanics"
    (repo / "server" / "engine").mkdir(parents=True)
    (repo / "server" / "engine" / "README.md").write_text("not the engine\n")
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "x"], check=True)
    return repo / "server" / "engine"


def test_run_refuses_unpinned_software_before_writing_anything(tmp_path):
    run_dir = _staged(tmp_path)
    engine = _engine_checkout(tmp_path)
    prefix = _fake_opensim_prefix(tmp_path / "osim")
    with pytest.raises(RuntimeError, match="not what scaled the paper's donors") as err:
        rab.run(run_dir, engine, sys.executable, opensim_prefix=prefix)
    assert "engine commit" in str(err.value) and "--allow-unpinned" in str(err.value)
    assert not (run_dir / "ab_run_record.json").exists()


def test_run_needs_an_opensim_prefix(tmp_path, monkeypatch):
    monkeypatch.delenv("AB_OPENSIM_PREFIX", raising=False)
    with pytest.raises(FileNotFoundError, match="opensim_environment.yml"):
        rab.run(_staged(tmp_path), tmp_path, sys.executable, opensim_prefix=None)
