"""Facts stated in more than one place must agree.

- ``docs/pinned_versions.md``'s *Current pins* names the commits and the solver md5 that ``comak_knee_oa.pins``
  enforces, so the document cannot go on describing pins the code does not use.
- The AddBiomechanics repeat-run spread is stated once, word for word, in each document that quotes it, so a new
  repeat run is added everywhere or nowhere.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from comak_knee_oa import pins  # noqa: E402


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else None]


def test_current_pins_match_the_enforced_pins():
    current = _section((REPO_ROOT / "docs" / "pinned_versions.md").read_text(encoding="utf-8"), "Current pins (v3)")
    for name, sha in pins.PINS.items():
        assert f"`{name}`" in current and sha in current, name
    assert pins.JAM["commit"] in current
    assert pins.JAM["md5"] in current
    assert f"{pins.JAM['size']:,}" in current


AB_SPREAD = ("Repeat runs on identical inputs, per donor: Subject_119, seven runs (the vendored body among them), knee "
             "scale factor `s_wa` SD 0.25 %, range 0.77 %, the six repeats −0.56 % to +0.21 % from the vendored body; "
             "Subject_121, two repeats, −0.38 % and +0.56 % from the vendored body, 0.94 % apart.")


def _flat(path: Path) -> str:
    """The text with line breaks, Markdown quote markers and repeated spaces folded away."""
    text = path.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text))


def test_the_addbiomechanics_spread_is_stated_once():
    for doc in ("tools/addbiomechanics/README.md",):
        flat = _flat(REPO_ROOT / doc)
        assert AB_SPREAD in flat, doc
        assert "Seven runs of one donor" not in flat, doc
