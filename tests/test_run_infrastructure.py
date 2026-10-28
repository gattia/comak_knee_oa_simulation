"""The run infrastructure: run declaration, write guard, provenance, run name.

Everything here builds its own miniature data roots under pytest's tmp_path --
a "published" root and a new root that borrows from it -- so the guard is
exercised against real symlinks without any test going near real data.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from comak_knee_oa import cohorts, new_root, paths, provenance  # noqa: E402
from comak_knee_oa.config import Config, Knee, load_config  # noqa: E402
from comak_knee_oa.paths import ReadOnlyRootError, UndeclaredRunError  # noqa: E402


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """(published, new): a finished root, and a v3-style root borrowing from it."""
    published = tmp_path / "published"
    inputs = published / "00_inputs"
    (inputs / "meshes" / "00m" / "9000001").mkdir(parents=True)
    (inputs / "meshes" / "00m" / "9000001" / "mesh.vtk").write_text("mesh")
    (inputs / "gait" / "bases").mkdir(parents=True)
    (inputs / "gait" / "addbiomechanics" / "Subject_102").mkdir(parents=True)
    (inputs / "gait" / "addbiomechanics" / "Subject_102" / "body.osim").write_text("ab")
    (inputs / "cohort.json").write_text(json.dumps({"knees": ["9000001_RIGHT"]}))
    (inputs / "_manifest.json").write_text("{}")
    (published / "01_models").mkdir()
    (published / "01_models" / "_manifest.json").write_text("{}")  # holds outputs
    (published / "02_simulations").mkdir()

    new = tmp_path / "new"
    monkeypatch.delenv("COMAK_READ_ONLY_ROOTS", raising=False)
    new_root.create(new, "vtest", published)
    monkeypatch.setenv("COMAK_DATA_ROOT", str(new))
    return published, new


# ------------------------------------------------------------- new root ----


def test_new_root_layout(roots):
    published, new = roots
    decl = json.loads((new / "_run.json").read_text())
    assert decl["run_name"] == "vtest"
    assert decl["read_only_roots"] == [str(published.resolve())]
    assert (new / "00_inputs").is_dir() and not (new / "00_inputs").is_symlink()
    assert (new / "00_inputs" / "meshes").is_symlink()
    assert (new / "00_inputs" / "gait" / "bases").is_symlink()
    ab = new / "00_inputs" / "gait" / "addbiomechanics"
    assert ab.is_dir() and not ab.is_symlink() and not any(ab.iterdir())
    manifest = new / "00_inputs" / "_manifest.json"
    assert manifest.is_file() and not manifest.is_symlink()
    for stage in ("01_models", "02_simulations", "03_derived", "04_results"):
        assert (new / stage).is_dir()


def test_new_root_refuses_a_populated_directory(roots, tmp_path):
    published, new = roots
    with pytest.raises(FileExistsError):
        new_root.create(new, "other", published)


def test_link_addbiomechanics_for_reproduction(roots, tmp_path):
    published, _ = roots
    g1 = tmp_path / "g1"
    new_root.create(g1, "vtest_g1", published, link_ab=True)
    donor = g1 / "00_inputs" / "gait" / "addbiomechanics" / "Subject_102"
    assert donor.is_symlink() and (donor / "body.osim").read_text() == "ab"


# ----------------------------------------------------------- write guard ----


def test_writes_inside_the_new_root_are_allowed(roots):
    _, new = roots
    target = paths.stage_dir("simulations", "Subject_102", create=True) / "x.txt"
    target.write_text("ok")
    assert target.read_text() == "ok"


def test_reading_the_published_root_is_allowed(roots):
    published, new = roots
    assert (new / "00_inputs" / "meshes" / "00m" / "9000001" / "mesh.vtk").read_text() == "mesh"
    assert (published / "00_inputs" / "cohort.json").read_text()


@pytest.mark.parametrize(
    "write",
    [
        lambda new, pub: (new / "00_inputs" / "meshes" / "new.vtk").write_text("x"),
        lambda new, pub: (new / "00_inputs" / "cohort.json").write_text("x"),
        lambda new, pub: (new / "00_inputs" / "meshes" / "sub").mkdir(),
        lambda new, pub: (pub / "02_simulations" / "x.txt").write_text("x"),
        lambda new, pub: open(pub / "00_inputs" / "cohort.json", "a").close(),
        lambda new, pub: os.open(pub / "00_inputs" / "y", os.O_CREAT | os.O_WRONLY),
        lambda new, pub: shutil.copy(pub / "00_inputs" / "cohort.json", pub / "00_inputs" / "c2"),
        lambda new, pub: os.link(pub / "00_inputs" / "cohort.json", new / "02_simulations" / "h"),
        lambda new, pub: shutil.rmtree(pub / "02_simulations"),
        lambda new, pub: os.remove(pub / "00_inputs" / "cohort.json"),
        lambda new, pub: os.rename(pub / "00_inputs" / "cohort.json", new / "02_simulations" / "c"),
    ],
    ids=["new-file-via-dir-link", "overwrite-via-file-link", "mkdir-via-dir-link",
         "direct-write", "append", "os-open", "copy-into", "hardlink", "rmtree",
         "remove", "rename-out"],
)
def test_writes_reaching_the_published_root_are_refused(roots, write):
    published, new = roots
    before = sorted(str(p) for p in published.rglob("*"))
    with pytest.raises(ReadOnlyRootError):
        write(new, published)
    assert sorted(str(p) for p in published.rglob("*")) == before
    assert (published / "00_inputs" / "cohort.json").read_text().startswith("{")


def test_read_only_roots_carry_over_from_a_run_root_baseline(roots, tmp_path, monkeypatch):
    """A root made from a run root also protects the roots that run root borrows from.

    leaf/00_inputs/meshes -> new/00_inputs/meshes -> published/00_inputs/meshes: a
    write through two links must still be refused, with no environment variable set.
    """
    published, new = roots
    leaf = tmp_path / "leaf"
    new_root.create(leaf, "vleaf", new)
    decl = json.loads((leaf / "_run.json").read_text())
    assert decl["read_only_roots"] == [str(new.resolve()), str(published.resolve())]
    monkeypatch.setenv("COMAK_DATA_ROOT", str(leaf))
    with pytest.raises(ReadOnlyRootError):
        (leaf / "00_inputs" / "meshes" / "new.vtk").write_text("x")
    assert not (published / "00_inputs" / "meshes" / "new.vtk").exists()


def test_replacing_a_link_in_the_new_root_is_allowed(roots):
    """Renaming onto a symlink replaces the link, not the borrowed file."""
    published, new = roots
    tmp = new / "00_inputs" / "cohort.json.tmp"
    tmp.write_text('{"knees": []}')
    os.replace(tmp, new / "00_inputs" / "cohort.json")
    assert not (new / "00_inputs" / "cohort.json").is_symlink()
    assert json.loads((published / "00_inputs" / "cohort.json").read_text())["knees"]


def test_stage_dir_refuses_a_finished_undeclared_root(roots, monkeypatch):
    published, _ = roots
    monkeypatch.setenv("COMAK_DATA_ROOT", str(published))
    with pytest.raises(ReadOnlyRootError, match="finished dataset"):
        paths.stage_dir("results", "figure", create=True)
    assert not (published / "04_results").exists()


def test_scratch_root_without_outputs_stays_writable(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    (scratch / "02_simulations").mkdir(parents=True)
    monkeypatch.setenv("COMAK_DATA_ROOT", str(scratch))
    (paths.stage_dir("simulations", "a", create=True) / "f").write_text("ok")


def test_read_only_declaration_and_env(roots, monkeypatch, tmp_path):
    published, new = roots
    decl = json.loads((new / "_run.json").read_text())
    decl["read_only"] = True
    (new / "_run.json").write_text(json.dumps(decl))
    with pytest.raises(ReadOnlyRootError, match="read_only"):
        paths.stage_dir("results", create=True)

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("COMAK_READ_ONLY_ROOTS", str(other))
    with pytest.raises(ReadOnlyRootError, match="COMAK_READ_ONLY_ROOTS"):
        (other / "x").write_text("x")


def test_symlink_escape_to_an_unlisted_place_is_refused(roots, tmp_path):
    _, new = roots
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (new / "03_derived" / "borrowed").symlink_to(elsewhere)
    with pytest.raises(ReadOnlyRootError, match="through a symlink"):
        (new / "03_derived" / "borrowed" / "f").write_text("x")


# --------------------------------------------------------------- run name ----


def test_run_placeholder_is_filled_from_the_declaration(roots):
    config = Config({"run": {"results_dir": "02_simulations/g/{knee}/comak_{run}"}})
    path = config.for_knee(Knee("9000001", "RIGHT")).path("run", "results_dir")
    assert path == paths.data_root() / "02_simulations/g/9000001_00m_RIGHT/comak_vtest"
    assert config.get("run", "results_dir").endswith("comak_vtest")
    assert config.data["run"]["results_dir"].endswith("comak_{run}")  # raw definition


def test_run_placeholder_on_an_undeclared_root_raises(roots, monkeypatch):
    published, _ = roots
    monkeypatch.setenv("COMAK_DATA_ROOT", str(published))
    config = Config({"run": {"results_dir": "02_simulations/{run}"}, "other": {"k": 1}})
    assert config.get("other", "k") == 1  # values without {run} still read
    with pytest.raises(UndeclaredRunError, match="V2_LAYOUT"):
        config.path("run", "results_dir")


def test_qc_keys_follow_the_run(roots, monkeypatch):
    published, _ = roots
    assert cohorts.QC_TAG["Subject_102"] == "Subject_102_vtest"
    assert cohorts.qc_tags(["default"]) == ["default_vtest"]
    # Another run's table must not resolve on this root.
    with pytest.raises(KeyError):
        cohorts.resolve_qc_tags({"multigait_Subject_102_bts_v2"}, ["Subject_102"])
    monkeypatch.setenv("COMAK_DATA_ROOT", str(published))
    assert cohorts.resolve_qc_tags({"multigait_Subject_102_bts_v2"}, ["Subject_102"]) == [
        "multigait_Subject_102_bts_v2"
    ]


CONFIG = REPO_ROOT / "config"
CONDITION_CONFIGS = [CONFIG / "reference_gait.json"] + sorted((CONFIG / "measured_gait").glob("*.json"))


@pytest.mark.parametrize("path", CONDITION_CONFIGS, ids=lambda p: p.stem)
def test_every_condition_config_names_the_run(path):
    """No run name is hard-coded in a condition config; each uses {run}.

    Every config on disk, whichever root's donor list it belongs to.
    """
    text = path.read_text()
    assert "bts_v2" not in text and "May.22.2026_Full_run" not in text
    data = json.loads(text)
    assert "{run}" in data["run"]["results_dir"]
    assert "{run}" in data["model"]["built_model"]
    gait = cohorts.REFERENCE if path.stem == "reference_gait" else path.stem
    assert data["run"]["gait"] == gait, "run.gait must equal the condition key"


def test_simulation_configs_read_the_model_reconstruction_writes(roots):
    import reconstruct_knee

    knee = Knee("9000001", "RIGHT")
    built = reconstruct_knee.built_model_path(
        load_config(REPO_ROOT / "config" / "reconstruction.json").for_knee(knee))
    assert built.is_relative_to(paths.stage_dir("models", "vtest"))
    for path in CONDITION_CONFIGS:
        assert load_config(path).for_knee(knee).path("model", "built_model") == built, path.stem


# --------------------------------------------------------- measured donors ----

_MISSING_DEFAULT_CONFIGS = [g for g in cohorts.MEASURED_DEFAULT
                            if not (CONFIG / "measured_gait" / f"{g}.json").is_file()]


def test_measured_donors_follow_the_root(roots, monkeypatch, tmp_path):
    published, new = roots
    # The new root records its donors; a run's donors are cohorts.MEASURED_DEFAULT.
    assert json.loads((new / "_run.json").read_text())["measured_gaits"] == list(
        cohorts.MEASURED_DEFAULT)
    assert cohorts.MEASURED == cohorts.MEASURED_DEFAULT
    assert cohorts.ALL == (cohorts.REFERENCE,) + cohorts.MEASURED_DEFAULT
    assert cohorts.QC_TAG["Subject_121"] == "Subject_121_vtest"
    with pytest.raises(KeyError, match="RSubject_121"):
        cohorts.condition_config_path("RSubject_121")
    # A root that declares no run is the v2 root, with the v2 donors.
    monkeypatch.setenv("COMAK_DATA_ROOT", str(published))
    assert cohorts.MEASURED == cohorts.MEASURED_V2
    assert cohorts.QC_TAG["RSubject_121"] == "multigait_RSubject_121_bts_v2"
    with pytest.raises(KeyError, match="Subject_121"):
        cohorts.pressure_dir("Subject_121", "femur")
    # A reproduction root links the baseline's gait inputs and runs its donors.
    g1 = tmp_path / "g1"
    decl = new_root.create(g1, "vtest_g1", published, link_ab=True)
    assert decl["measured_gaits"] == ["Subject_102"]  # the fixture baseline's only donor
    monkeypatch.setenv("COMAK_DATA_ROOT", str(g1))
    assert cohorts.ALL == ("default", "Subject_102")
    assert cohorts.qc_tags() == ["default_vtest_g1", "Subject_102_vtest_g1"]


def test_measured_donors_must_have_inputs_when_linked(roots, tmp_path):
    published, _ = roots
    with pytest.raises(ValueError, match="no gait inputs in the baseline"):
        new_root.create(tmp_path / "bad", "vtest_bad", published, link_ab=True,
                        measured_gaits=["Subject_102", "Subject_121"])
    assert not (tmp_path / "bad").exists()
    for bad in ([], ["default"], ["Subject_102", "Subject_102"], ["a/b"]):
        with pytest.raises(ValueError):
            new_root.create(tmp_path / "bad", "vtest_bad", published, measured_gaits=bad)


def test_set_gaits_records_but_never_changes_a_list(roots, tmp_path):
    published, _ = roots
    old = tmp_path / "old"
    new_root.create(old, "vtest_old", published)
    decl = json.loads((old / "_run.json").read_text())
    del decl["measured_gaits"]  # a root created before the key existed
    (old / "_run.json").write_text(json.dumps(decl))
    new_root.set_measured_gaits(old, ["Subject_102", "Subject_107"])
    decl = json.loads((old / "_run.json").read_text())
    assert decl["measured_gaits"] == ["Subject_102", "Subject_107"]
    assert decl["declaration_log"][0]["set"] == "measured_gaits"
    new_root.set_measured_gaits(old, ["Subject_102", "Subject_107"])  # same list: fine
    with pytest.raises(ValueError, match="already declares"):
        new_root.set_measured_gaits(old, ["Subject_102"])


def test_donor_lists_agree():
    """The run's donors are the same nine in the code, the synthetic arm and vendoring."""
    synthetic = json.loads((CONFIG / "synthetic_shape_axis.json").read_text())
    vendored = json.loads((REPO_ROOT / "pipeline" / "vendor_sources.json").read_text())
    assert tuple(synthetic["simulate"]["gaits"]) == cohorts.MEASURED_DEFAULT
    assert tuple(vendored["gaits"]) == cohorts.MEASURED_DEFAULT
    assert "RSubject_121" not in cohorts.MEASURED_DEFAULT  # Subject_121 in its place
    assert set(cohorts.MEASURED_V2) - set(cohorts.MEASURED_DEFAULT) == {"RSubject_121"}


@pytest.mark.xfail(bool(_MISSING_DEFAULT_CONFIGS), strict=True,
                   reason=f"no measured-gait config yet for {_MISSING_DEFAULT_CONFIGS}: "
                          "config/measured_gait/Subject_121.json is written from its AddBiomechanics run")
def test_every_default_donor_has_a_config():
    assert not _MISSING_DEFAULT_CONFIGS


def test_published_root_is_read_with_its_own_layout(roots, monkeypatch):
    published, new = roots
    knee = Knee("9000001", "RIGHT")
    # A declared root: the run's own names.
    built = cohorts.condition_config("Subject_102").for_knee(knee).path("model", "built_model")
    assert built.is_relative_to(new / "01_models" / "vtest")
    assert cohorts.results_dir("default", knee) == (
        new / "02_simulations" / "reference" / knee.directory / "comak_vtest")
    # The v2 root: the names it was produced with.
    monkeypatch.setenv("COMAK_DATA_ROOT", str(published))
    assert cohorts.results_dir("default", knee) == (
        published / "02_simulations" / "reference" / knee.directory / "comak")
    assert cohorts.results_dir("RSubject_121", knee) == (
        published / "02_simulations" / "RSubject_121" / knee.directory
        / "comak_pathway_b_multigait_RSubject_121_bts_v2")
    # ... and the switches as v2 ran them, not the shipped (v3) values.
    v2 = cohorts.condition_config("Subject_102")
    assert v2.get("comak_timing", "comak_window_pad_s") == 0.0
    assert v2.get("scaling", "mass_rule") == "as_run"
    assert v2.get("scaling", "allow_generic_mismatch") is True
    assert cohorts.condition_config("default").get("comak_timing", "comak_window_pad_s") == 0.0
    # Only through cohorts: a config file loaded directly still refuses to name a run there.
    with pytest.raises(UndeclaredRunError):
        load_config(CONFIG / "reference_gait.json").for_knee(knee).path("run", "results_dir")


# ---------------------------------------------------------------- scaling ----


def test_scaling_options(tmp_path):
    """mass_rule is required (no fallback to the library's as-run default); the other
    options are validated and always passed, so they enter the provenance record."""
    import simulate_gait
    from comak_knee_oa.config import ConfigError

    base = load_config(CONFIG / "measured_gait" / "Subject_102.json")
    options = simulate_gait.scaling_options(base)
    assert options["mass_rule"] == "volume"
    assert options["allow_generic_mismatch"] is False
    assert options["recorded_mass_kg"] == pytest.approx(base.get("addbiomechanics", "mass_kg"))
    with pytest.raises(ConfigError, match="no defaults"):
        simulate_gait.scaling_options(None)

    def with_scaling(section, **extra):
        data = json.loads(json.dumps(base.data))
        data["scaling"] = section
        data.update(extra)
        return Config(data, origin=tmp_path / "x.json")

    as_run = simulate_gait.scaling_options(
        with_scaling({"mass_rule": "as_run", "allow_generic_mismatch": True}))
    assert as_run["mass_rule"] == "as_run" and as_run["allow_generic_mismatch"] is True
    for bad, match in (
        ({}, "mass_rule is required"),
        ({"allow_generic_mismatch": True}, "mass_rule is required"),
        ({"mass_rule": "volumetric"}, "must be one of"),
        ({"mass_rule": "volume", "allow_generic_mismach": True}, "unknown scaling option"),
        ({"mass_rule": "volume", "allow_generic_mismatch": "yes"}, "true or false"),
    ):
        with pytest.raises(ConfigError, match=match):
            simulate_gait.scaling_options(with_scaling(bad))
    with pytest.raises(ConfigError, match="positive number"):
        simulate_gait.scaling_options(with_scaling({"mass_rule": "volume"},
                                                   addbiomechanics={"mass_kg": "62"}))
    no_record = json.loads(json.dumps(base.data))
    no_record.pop("addbiomechanics", None)
    assert simulate_gait.scaling_options(
        Config(no_record, origin=tmp_path / "y.json"))["recorded_mass_kg"] is None
    with pytest.raises(ValueError, match="scaling options are required"):
        simulate_gait.scale_to_gait_body(tmp_path / "a.osim", tmp_path / "b.osim",
                                         tmp_path / "c" / "c.osim", options=None)


@pytest.mark.parametrize("path", CONDITION_CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_carry_the_v3_switches(path):
    """The shipped configs carry the v3 settings: volume mass rule, the AddBiomechanics-generic
    check on, and a window pad of 0.005 s for the measured trials and 0.0 for the overground
    reference trial, which needs none."""
    data = json.loads(path.read_text())
    expected_pad = 0.005 if data["run"]["condition"] == "measured" else 0.0
    assert data["comak_timing"]["comak_window_pad_s"] == expected_pad
    if data["run"]["condition"] == "measured":
        assert data["scaling"]["mass_rule"] == "volume"
        assert data["scaling"].get("allow_generic_mismatch", False) is False
    else:
        assert "scaling" not in data


# ------------------------------------------------------------- provenance ----


def test_provenance_decisions(roots, tmp_path):
    _, new = roots
    source = new / "02_simulations" / "input.txt"
    source.write_text("one")
    unit = new / "02_simulations" / "unit"
    fp = provenance.fingerprint(inputs={"source": source}, options={"a": 1}, code=())

    assert provenance.decide(unit, fp, complete=False) == "build"
    provenance.mark_started(unit, fp)
    (unit / "out.txt").write_text("partial")
    assert provenance.decide(unit, fp, complete=False) == "build"  # resume own work
    provenance.mark_complete(unit, fp)
    assert provenance.decide(unit, fp, complete=True) == "skip"

    changed = provenance.fingerprint(inputs={"source": source}, options={"a": 2}, code=())
    with pytest.raises(provenance.ProvenanceMismatch, match="a: 1 -> 2"):
        provenance.decide(unit, changed, complete=True)
    assert provenance.decide(unit, changed, complete=True, force=True) == "build"

    source.write_text("two")
    moved = provenance.fingerprint(inputs={"source": source}, options={"a": 1}, code=())
    with pytest.raises(provenance.ProvenanceMismatch, match="input 'source' changed"):
        provenance.decide(unit, moved, complete=True)


def test_output_without_a_record_is_never_reused(roots):
    _, new = roots
    unit = new / "02_simulations" / "as_run"
    unit.mkdir()
    (unit / "scaled.osim").write_text("from somewhere else")
    fp = provenance.fingerprint(inputs={}, options={}, code=())
    with pytest.raises(provenance.ProvenanceMismatch, match="no _provenance.json"):
        provenance.decide(unit, fp, complete=True)


def test_directory_inputs_hash_their_contents(tmp_path):
    d = tmp_path / "geometry"
    d.mkdir()
    (d / "a.stl").write_text("a")
    first = provenance.hash_input(d)["sha256"]
    (d / "a.stl").write_text("b")
    assert provenance.hash_input(d)["sha256"] != first
