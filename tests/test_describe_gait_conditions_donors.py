"""describe_gait_conditions describes the data root's measured donors, not every config on disk.

``config/measured_gait/`` holds configs for donors of more than one run (``RSubject_121.json`` for v2 beside
``Subject_121.json`` for v3) while a root holds only its declared donors' inputs, so globbing the configs would stop
the script on the first input it could not find.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "pipeline"):
    sys.path.insert(0, str(REPO_ROOT / sub))

import describe_gait_conditions as describe  # noqa: E402


def test_only_the_declared_donors_are_described(monkeypatch):
    described = []
    monkeypatch.setattr(describe.cohorts, "measured", lambda: ("Subject_102", "Subject_121"))
    monkeypatch.setattr(describe, "describe_reference", lambda: {"name": "reference"})
    monkeypatch.setattr(describe, "describe_measured", lambda path: described.append(path.name))
    monkeypatch.setattr(describe, "report", lambda rows: None)
    assert describe.main([]) == 0
    assert described == ["Subject_102.json", "Subject_121.json"]
    assert (REPO_ROOT / "config" / "measured_gait" / "RSubject_121.json").is_file()  # the case this guards
