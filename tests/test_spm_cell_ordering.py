"""Figure 5's SPM cell order determines every seed, so it is pinned here.

Each cell's permutation seed is ``SPM_SEED + its index in spm_cells()``, so the
list's order is not a style choice: reordering it (swapping two loop levels,
alphabetising a constant, inserting a cell anywhere but the end) would change
every published cluster p-value while every name still resolves. Nothing
downstream can detect that; only this test can.

The golden order below is a literal copy, not derived from the module's own
constants: if it were built from ``CONDITIONS``/``SUBSETS``/etc., reordering
those constants would reorder the expectation in lockstep and the test would
pass while the seeds moved.

Cells are identified by ``(condition, name)``, not name alone: the 32
correlation names repeat across the two walking conditions (the condition lives
in its own field), and keying on the name would collapse the 32-cell check to 16.

If this test fails after an intentional change (a new cell appended), extend the
golden list at the end. Changing any existing row means Figure 5's published
p-values do not reproduce and the SPM array must be computed again.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

import derive_figure5_data as fig3  # noqa: E402

BASE = 20260829  # an arbitrary fixed base (a date); no other meaning

# Cells 16-31 run under the knee-OA example gait, the ninth measured donor:
# RSubject_121 in v2, Subject_121 in v3. The swap renames the condition and
# leaves every index, so every seed, in place.
OA = "<knee-OA example gait>"
OA_EXAMPLE_GAITS = ("RSubject_121", "Subject_121")

# (index, condition or None, name). Seed == BASE + index, asserted below.
GOLDEN = [
    (0, "default", "corr_femur_thinning_vs_cum_all"),
    (1, "default", "corr_femur_thinning_vs_max_all"),
    (2, "default", "corr_tibia_thinning_vs_cum_all"),
    (3, "default", "corr_tibia_thinning_vs_max_all"),
    (4, "default", "corr_femur_thinning_vs_cum_healthy"),
    (5, "default", "corr_femur_thinning_vs_max_healthy"),
    (6, "default", "corr_tibia_thinning_vs_cum_healthy"),
    (7, "default", "corr_tibia_thinning_vs_max_healthy"),
    (8, "default", "corr_femur_thinning_vs_cum_progressor"),
    (9, "default", "corr_femur_thinning_vs_max_progressor"),
    (10, "default", "corr_tibia_thinning_vs_cum_progressor"),
    (11, "default", "corr_tibia_thinning_vs_max_progressor"),
    (12, "default", "corr_femur_thinning_vs_cum_kl2"),
    (13, "default", "corr_femur_thinning_vs_max_kl2"),
    (14, "default", "corr_tibia_thinning_vs_cum_kl2"),
    (15, "default", "corr_tibia_thinning_vs_max_kl2"),
    (16, OA, "corr_femur_thinning_vs_cum_all"),
    (17, OA, "corr_femur_thinning_vs_max_all"),
    (18, OA, "corr_tibia_thinning_vs_cum_all"),
    (19, OA, "corr_tibia_thinning_vs_max_all"),
    (20, OA, "corr_femur_thinning_vs_cum_healthy"),
    (21, OA, "corr_femur_thinning_vs_max_healthy"),
    (22, OA, "corr_tibia_thinning_vs_cum_healthy"),
    (23, OA, "corr_tibia_thinning_vs_max_healthy"),
    (24, OA, "corr_femur_thinning_vs_cum_progressor"),
    (25, OA, "corr_femur_thinning_vs_max_progressor"),
    (26, OA, "corr_tibia_thinning_vs_cum_progressor"),
    (27, OA, "corr_tibia_thinning_vs_max_progressor"),
    (28, OA, "corr_femur_thinning_vs_cum_kl2"),
    (29, OA, "corr_femur_thinning_vs_max_kl2"),
    (30, OA, "corr_tibia_thinning_vs_cum_kl2"),
    (31, OA, "corr_tibia_thinning_vs_max_kl2"),
    (32, None, "anova_femur_thinning"),
    (33, None, "anova_tibia_thinning"),
    (34, None, "ttest_femur_thinning_healthy_vs_progressor"),
    (35, None, "ttest_femur_thinning_healthy_vs_kl2"),
    (36, None, "ttest_femur_thinning_progressor_vs_kl2"),
    (37, None, "ttest_tibia_thinning_healthy_vs_progressor"),
    (38, None, "ttest_tibia_thinning_healthy_vs_kl2"),
    (39, None, "ttest_tibia_thinning_progressor_vs_kl2"),
]


def test_base_seed_is_pinned():
    assert fig3.SPM_SEED == BASE, (
        "SPM_SEED changed. Every published Figure 3 cluster p-value was "
        "computed under 20260829 + cell index; changing the base re-rolls "
        "all forty nulls."
    )


def test_oa_example_is_one_of_the_two_ninth_donors():
    assert fig3.OA_EXAMPLE_GAIT in OA_EXAMPLE_GAITS


def test_cell_order_names_and_seeds_are_pinned():
    cells = fig3.spm_cells()
    assert len(cells) == len(GOLDEN) == 40

    for expected, cell in zip(GOLDEN, cells):
        index, condition, name = expected
        if condition == OA:
            condition = fig3.OA_EXAMPLE_GAIT
        where = f"cell {index} ({condition}/{name})"
        assert cell["index"] == index, where
        assert cell.get("condition") == condition, (
            f"{where}: condition moved — the 32 correlation names repeat "
            "across conditions, so order within CONDITIONS is load-bearing"
        )
        assert cell["name"] == name, where
        assert cell["seed"] == BASE + index, (
            f"{where}: seed is not SPM_SEED + index — the published maps "
            "were computed under that rule"
        )


def test_matches_the_shipped_seed_record():
    """Code ↔ artifact: the shipped ``spm/_seeds.json`` describes this order.

    Skipped when the data root is absent (code-only clone); when present, the
    published record must agree with what the code would run today, else one of
    them is stale.
    """
    try:
        from comak_knee_oa.paths import stage_dir
        seeds_path = (stage_dir("results", "figure5_spatial_colocalization")
                      / "spm" / "_seeds.json")
    except Exception:
        pytest.skip("data root not configured")
    if not seeds_path.is_file():
        pytest.skip("spm/_seeds.json not present in this checkout's data root")

    import json
    record = json.loads(seeds_path.read_text())
    assert record["base_seed"] == BASE
    recorded = {(c["index"], c["name"]): c["seed"] for c in record["cells"]}
    assert len(recorded) == 40
    for cell in fig3.spm_cells():
        key = (cell["index"], cell["name"])
        assert recorded.get(key) == cell["seed"], (
            f"{key}: shipped _seeds.json disagrees with spm_cells() — "
            "either the code was reordered after the array ran, or the "
            "record is from another revision"
        )
