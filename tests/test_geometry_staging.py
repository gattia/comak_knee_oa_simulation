"""Model builds and simulations link the meshes they share instead of copying them.

Every built model carries the template's ~163 generic meshes unchanged, and by
default COMAK inverse kinematics copies the model's whole Geometry folder twice per
simulation and the body scaling copies the built model's. The pipeline asks the
libraries to link instead (nsosim ``build_joint_model``/``scale_comak_model
(geometry_staging="link")``, pycomak ``KneeOptimizer``/``COMAKInverseKinematics
(geometry_staging="link")``). Results are identical either way; this pins that the
pipeline asks for links, and that a simulation's model finds every mesh in its own
folder (no process-wide search path).

CPU only; the library constructors are replaced by recorders.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from comak_knee_oa.config import Knee, load_config  # noqa: E402

RECONSTRUCTION = REPO_ROOT / "config" / "reconstruction.json"


@pytest.fixture(autouse=True)
def declared_root(tmp_path, monkeypatch):
    """A scratch root that declares run ``test``; the configured root is read-only."""
    from comak_knee_oa import paths

    real = paths.data_root()
    root = tmp_path / "root"
    (root / "02_simulations").mkdir(parents=True)
    (root / "00_inputs").symlink_to(real / "00_inputs")
    (root / "_run.json").write_text(json.dumps({"run_name": "test", "read_only_roots": [str(real)]}))
    monkeypatch.setenv("COMAK_DATA_ROOT", str(root))
    return root


def test_model_build_links_the_template_meshes(monkeypatch):
    import nsosim.model_building as model_building

    from reconstruct_knee import build_knee_model

    calls = []
    monkeypatch.setattr(model_building, "build_joint_model", lambda **kw: calls.append(kw) or "x")
    config = load_config(RECONSTRUCTION).for_knee(Knee.parse("9003316_RIGHT"))
    build_knee_model(config, bones={}, osim_meshes={}, center_mode="as_run")
    (kw,) = calls
    assert kw["geometry_staging"] == "link"


def test_comak_ik_links_the_model_geometry(monkeypatch, tmp_path):
    from comak_knee_oa import comak_chain

    seen = {}

    class Optimizer:
        intermediate_model_path = str(tmp_path / "intermediate.osim")

        def __init__(self, **kw):
            seen["optimizer"] = kw

        def optimize_patella_location(self):
            return None

    class InverseKinematics:
        def __init__(self, **kw):
            seen["ik"] = kw

        def perform_settle_sim(self):
            raise AssertionError("not reached")

    def timeout(func, budget):
        raise TimeoutError

    class Settings:
        def get(self, section, key, default=None):
            return default

    monkeypatch.setattr(comak_chain, "KneeOptimizer", Optimizer)
    monkeypatch.setattr(comak_chain, "COMAKInverseKinematics", InverseKinematics)
    monkeypatch.setattr(comak_chain, "run_with_timeout", timeout)
    result = comak_chain.run_comak_chain(
        model_to_optimize=str(tmp_path / "model.osim"),
        results_dir=str(tmp_path),
        markerset_file="markers.trc",
        external_loads_file="loads.xml",
        forceset_file="reserves.xml",
        timing={"start_time_ik": 0, "stop_time_ik": 1, "start_time_comak": 0, "stop_time_comak": 1},
        config=Settings(),
    )
    assert result == "settle_sim_timeout_ik"
    assert seen["optimizer"]["geometry_staging"] == "link"
    assert seen["ik"]["geometry_staging"] == "link"


MODEL = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <Model name="m">
    <publications>Brandon, S. C., & Thelen, D. G.</publications>
    <Mesh name="a"><mesh_file>a.stl</mesh_file></Mesh>
    <Smith2018ContactMesh name="b">
      <mesh_file>b.obj</mesh_file>
      <mesh_back_file>c.stl</mesh_back_file>
    </Smith2018ContactMesh>
    <ExternalLoads><datafile>not_a_mesh.mot</datafile></ExternalLoads>
    <Mesh name="d"><mesh_file>d.vtp</mesh_file></Mesh>
  </Model>
</OpenSimDocument>
"""


def test_every_mesh_must_resolve_from_the_models_own_folder(tmp_path):
    from simulate_gait import require_meshes_resolve, unresolved_meshes

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "b.obj").write_text("b")
    model_dir = tmp_path / "model"
    (model_dir / "Geometry").mkdir(parents=True)
    model = model_dir / "model.osim"
    model.write_text(MODEL)
    (model_dir / "Geometry" / "a.stl").write_text("a")                      # real file
    (model_dir / "Geometry" / "b.obj").symlink_to(elsewhere / "b.obj")      # resolving link
    (model_dir / "Geometry" / "c.stl").symlink_to(elsewhere / "missing")    # dangling link
    (model_dir / "d.vtp").write_text("d")                                   # beside the model
    assert unresolved_meshes(model) == ["c.stl"]
    with pytest.raises(FileNotFoundError, match="c.stl"):
        require_meshes_resolve(model)
    (elsewhere / "missing").write_text("c")
    require_meshes_resolve(model)


def test_scaling_links_and_keeps_staging_out_of_provenance(tmp_path, monkeypatch):
    import nsosim.scaling

    import simulate_gait

    calls = []

    def fake_scale(**kw):
        calls.append(kw)
        out = Path(kw["output_osim"])
        out.write_text("<OpenSimDocument/>")
        (out.parent / "Geometry").mkdir()
        (out.parent / "Geometry" / "knee.stl").write_text("baked")
        Path(kw["report_json"]).write_text(json.dumps({"wa_scale": 1.0}))
        return kw["report_json"]

    monkeypatch.setattr(nsosim.scaling, "scale_comak_model", fake_scale)
    built = tmp_path / "built" / "built.osim"
    (built.parent / "Geometry").mkdir(parents=True)
    built.write_text("<OpenSimDocument/>")
    body = tmp_path / "body.osim"
    body.write_text("<OpenSimDocument/>")
    scaled = tmp_path / "scaled_model" / "scaled.osim"
    options = {"mode": "WA", "mass_rule": "volume"}
    simulate_gait.scale_to_gait_body(built, body, scaled, options=options)
    (kw,) = calls
    assert kw["geometry_staging"] == "link"
    from comak_knee_oa import provenance

    recorded = provenance.read_record(scaled.parent)
    assert "geometry_staging" not in recorded["options"]
