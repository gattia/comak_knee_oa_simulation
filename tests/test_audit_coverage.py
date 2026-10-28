"""Every condition's pressure chain covers what its upstream holds.

Runs the coverage audit's collectors over the data root and checks the layers
the paper's pressure analyses stand on: a completed simulation has its
per-vertex summary, a summary with a walk and a transform has been carried to
the reference surface, and the combined matrix holds the knees with both
pressure and thinning. A layer short against its upstream is a cache built
before that upstream changed, which is what the audit catches.

The QC tables are not checked here: a downloaded root may carry only the
unified table, and the audit reports that as such.

CPU only; about half a minute (one stat per knee, condition and file).
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import audit_coverage  # noqa: E402
from comak_knee_oa.config import Knee  # noqa: E402

CHAIN = ("step4_", "transfer_", "matrix_")


def test_pressure_chain_covers_its_upstream():
    cohort, _ = audit_coverage.collect_cohort()
    excluded = audit_coverage.collect_manual_exclusions()
    knees = [Knee.parse(name) for name in sorted(cohort)]
    layers = audit_coverage.build_layers("v2", knees)

    found, short = {}, {}
    for name, collect, reference in layers:
        found[name], _ = collect()
        if reference is None or not name.startswith(CHAIN):
            continue
        expected = set.intersection(*(found[r] for r in reference))
        missing = expected - found[name] - excluded
        if missing:
            short[name] = sorted(missing)[:3]
    if not any(found[n] for n in found if n.startswith("matrix_")):
        pytest.skip("no combined matrices in the data root")
    assert not short, f"layers short against their upstream: {short}"
