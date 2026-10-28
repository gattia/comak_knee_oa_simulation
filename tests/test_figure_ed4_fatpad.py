"""Extended Data Figure 4 draws the fat pad with the arguments the model build used.

``pipeline/derive_figure_ed4_fatpad.py`` repeats ``create_prefemoral_fatpad_noboolean`` stage by stage, so it has
to restate the arguments ``nsosim.model_building.build_joint_model`` passes. If nsosim's defaults moved, the
derivation would still run, and only its as-built check (0.05 mm) would catch it, on the cluster. These tests catch
it without data.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def derive():
    return load("pipeline/derive_figure_ed4_fatpad.py", "derive_figure_ed4_fatpad")


def test_parameters_are_the_ones_the_model_build_passes(derive):
    model_building = pytest.importorskip("nsosim.model_building")
    from nsosim.articular_surfaces import create_prefemoral_fatpad_noboolean

    source = inspect.getsource(model_building)
    call = source[source.index("create_prefemoral_fatpad_noboolean(\n"):]
    call = call[:call.index("\n    )")]
    passed = dict(re.findall(r'cfg\("fatpad_(\w+)", ([^)]+)\)', call))
    for key in ("base_mm", "top_mm", "max_distance_to_patella_mm", "resample_clusters_final",
                "ray_cast_length", "final_smooth_iter"):
        assert float(passed[key].replace("_", "")) == float(derive.PARAMS[key]), key
    assert passed["norm_function"].strip('"') == derive.PARAMS["norm_function"]

    defaults = inspect.signature(create_prefemoral_fatpad_noboolean).parameters
    for key in ("percent_fem_cart_to_keep", "safety_mm"):
        assert key not in passed and defaults[key].default == derive.PARAMS[key], key
    assert "scale_percentile=95.0" in inspect.getsource(create_prefemoral_fatpad_noboolean)


def test_no_config_overrides_the_fat_pad_shape(derive):
    """build_joint_model reads these through cfg(), so a config key of the same name would change the build."""
    for path in (REPO_ROOT / "config").glob("*.json"):
        text = path.read_text()
        for key in derive.PARAMS:
            assert f'"fatpad_{key}"' not in text, (path.name, key)


def test_ray_direction_is_read_from_the_knees_build_record(derive, tmp_path, monkeypatch):
    monkeypatch.setattr(derive, "knee_dir", lambda knee: tmp_path)
    assert derive.clearance_rays_as_built("k")[0] == "auto_oriented"   # no record: a v2 build
    (tmp_path / "run_record.json").write_text(json.dumps(
        {"build_options": {"fatpad_clearance_rays": "dilation_normals"}}))
    assert derive.clearance_rays_as_built("k")[0] == "dilation_normals"


def test_the_figure_names_its_derivation_when_the_input_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("COMAK_DATA_ROOT", str(tmp_path))
    figure = load("figures/figure_ed4_fatpad.py", "figure_ed4_fatpad")
    sys.path.insert(0, str(REPO_ROOT / "figures"))
    from shared.results import MissingFigureInput

    with pytest.raises(MissingFigureInput, match=re.escape(figure.BUILD_CMD)):
        figure.load_inputs()
