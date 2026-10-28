"""Reconstruction writes only into its own output tree, never into 00_inputs.

nsosim's fitter saves copies of the meniscus meshes next to the meshes it reads
(``save_intermediate_men``, on by default). Fitting straight from
``00_inputs/meshes`` would therefore add files to the shared inputs, so
``reconstruct`` fits from links staged in the knee's output tree. This pins that.

CPU only; the fitter is replaced by a stub that makes the same write. The
scratch root gives the knee a private mesh folder of links, so even a regression
writes into the temporary tree, not into the data root.
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
KNEE = Knee.parse("9003316_RIGHT")


class _StopAfterFit(Exception):
    pass


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    from comak_knee_oa import paths
    from reconstruct_knee import subject_mesh_files

    real = paths.data_root()
    meshes = real / "00_inputs" / "meshes" / KNEE.timepoint / KNEE.subject
    if not meshes.is_dir():
        pytest.skip(f"no segmentation meshes for {KNEE.name}")
    root = tmp_path / "root"
    inputs = root / "00_inputs"
    inputs.mkdir(parents=True)
    for child in (real / "00_inputs").iterdir():
        if child.name != "meshes":
            (inputs / child.name).symlink_to(child)
    private = inputs / "meshes" / KNEE.timepoint / KNEE.subject
    private.mkdir(parents=True)
    for name in subject_mesh_files(KNEE).values():
        (private / name).symlink_to(meshes / name)
    (root / "01_models").mkdir()
    # The configs name their outputs by the run; the real root stays read-only.
    (root / "_run.json").write_text(json.dumps({"run_name": "test", "read_only_roots": [str(real)]}))
    monkeypatch.setenv("COMAK_DATA_ROOT", str(root))
    return private


def test_fitting_reads_staged_links_and_leaves_inputs_alone(scratch_root, tmp_path, monkeypatch):
    import nsosim.nsm_fitting

    import reconstruct_knee as rk

    data = json.loads(RECONSTRUCTION.read_text())
    data["run"]["output_dir"] = "01_models/reconstruct_test/{knee}"
    config_file = tmp_path / "reconstruction.json"
    config_file.write_text(json.dumps(data))
    config = load_config(config_file)
    before = sorted(p.name for p in scratch_root.iterdir())

    folders = {}

    def fake_fit(dict_bones, **kw):
        for bone, definition in dict_bones.items():
            subject = definition.get("subject")
            if subject is None:
                continue
            folders[bone] = Path(subject["folder"])
            if "med_men_filename" in subject:  # what save_intermediate_men does
                (Path(subject["folder"]) / "femur_med_men.vtk").write_text("copy")
        raise _StopAfterFit

    monkeypatch.setattr(rk, "_require_gpu", lambda: None)
    monkeypatch.setattr(rk, "require_center_mode_support", lambda mode: None)
    monkeypatch.setattr(nsosim.nsm_fitting, "align_knee_osim_fit_nsm", fake_fit)
    with pytest.raises(_StopAfterFit):
        rk.reconstruct(config, KNEE)

    staged = config.for_knee(KNEE).path("run", "output_dir") / "subject_meshes"
    assert set(folders) == {"femur", "tibia", "patella"}
    assert all(folder == staged for folder in folders.values())
    assert sorted(p.name for p in scratch_root.iterdir()) == before
    assert (staged / "femur_med_men.vtk").is_file()
    for name in rk.subject_mesh_files(KNEE).values():
        assert (staged / name).resolve() == (scratch_root / name).resolve()
