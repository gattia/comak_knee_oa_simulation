#!/usr/bin/env python3
"""Run AddBiomechanics locally for one gait donor, and record what ran.

    python pipeline/run_addbiomechanics.py stage  --donor Subject_119 --skeleton SKELETON.osim --zips 'ZIPS*.zip' --out RUN_DIR
    python pipeline/run_addbiomechanics.py stage  --donor MY_SUBJECT --skeleton SKELETON.osim \
        --markers TRIAL.trc --grf TRIAL_grf.mot --out RUN_DIR          # your own trial
    python pipeline/run_addbiomechanics.py run    RUN_DIR --engine AB/server/engine --python AB_ENV/bin/python \
        --opensim-prefix AB_OPENSIM_ENV
    python pipeline/run_addbiomechanics.py vendor RUN_DIR --dest DATA_ROOT/00_inputs/gait/addbiomechanics/Subject_119
    python pipeline/run_addbiomechanics.py knee-factor VENDORED_OR_RUN_DIR [MORE ...]

The measured-gait arm scales every knee to a donor's body as AddBiomechanics (AB) fitted it, and reads the donor's
markers, ground reaction forces and external loads from AB's output folder (see docs/scaling_comak_model.md). This
script replaces the website upload with the AB engine run on this machine, in three steps:

``stage``
    Builds an AB run folder: ``_subject.json``, the custom skeleton as ``unscaled_generic.osim``, and the donor's
    baseline treadmill trial (``markers.trc``, ``grf.mot``), taken from the provider zips or, for your own data,
    from a marker file and a ground-reaction-force file you name (``--markers``, ``--grf``). The subject entries -- mass,
    height, sex, age -- come from the donor's config, ``config/measured_gait/<donor>.json`` -> ``addbiomechanics``,
    which is their single source. The trial folder is named ``markers``, as in the website runs, so the outputs are
    called ``markers_segment_<n>`` there too and the configs point at the same file names.
``run``
    Runs the engine (``src/engine.py <run folder> osim_results``) and writes ``ab_run_record.json``: the engine's git
    commit and whether its tree was dirty, the nimblephysics and ``opensim-cmd`` versions, the interpreter, the exact
    command, the host and CPU, the inputs' md5s, the exit code and the times. The engine's environment comes from
    ``tools/addbiomechanics/`` (README there): ``bin/`` first on PATH, for the ``opensim-cmd`` the engine runs to
    write its scaled models, and ``stub/`` as the whole PYTHONPATH, for the ``opensim`` module it imports but does
    not use. It refuses to start unless the engine commit, nimblephysics and ``opensim-cmd`` are the versions that
    scaled the paper's donors (``ENGINE_PINS``; ``OPENSIM_CMD_ACCEPTED`` adds one OpenSim build shown to write the
    same models); ``--allow-unpinned`` overrides that, and the record says so.
``vendor``
    Copies ``osim_results/{Models,MarkerData,ID,IK}`` and the results archive into a data root, with an md5 for
    every file (``ab_vendor_record.json``). It refuses to write into an existing folder.
``knee-factor``
    Prints the knee scale factor the simulations apply for an AB output (vendored or a run folder): the mean of the
    femur and tibia long-axis factors, each read as the body's mesh scale over the same mesh's scale in AB's
    ``unscaled_generic.osim`` (``nsosim.scaling``; ``docs/scaling_comak_model.md``). With several folders it also
    prints each one's difference from the first, which is how a new AB run is compared with the vendored body.
    Runs in the pipeline environment (it reads the models with OpenSim and the pinned nsosim).

Nothing here is specific to a cluster. A batch wrapper only has to call ``run``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DONOR_CONFIGS = REPO / "config" / "measured_gait"
TOOLS = REPO / "tools" / "addbiomechanics"
TRIAL_NAME = "markers"
OUTPUT_NAME = "osim_results"
VENDORED = ("Models", "MarkerData", "ID", "IK")

# What scaled the paper's gait donors (every vendored ab_run_record.json). AddBiomechanics is not deterministic run
# to run, so these pin the software, not the result: the vendored outputs are the input of record.
ENGINE_PINS = {
    "engine_repository": "https://github.com/keenon/AddBiomechanics.git",
    "engine_commit": "8fe69cd8be3786a13e0cb90f4451fb14d7e3cb5c",
    "nimblephysics": "0.10.52.1",
    "opensim_cmd": "4.5.1-2025-04-01-5cf2dacc8",
}
# OpenSim builds accepted for ``opensim-cmd``, which writes AddBiomechanics' scaled models. The first is the paper's.
# The conda channel also serves a second build under the same package name; replaying the engine's scale-tool calls
# on both gives byte-identical models, so either reproduces the paper's software. Others are refused.
OPENSIM_CMD_ACCEPTED = ("4.5.1-2025-04-01-5cf2dacc8", "4.5.2-2025-05-02-fcdbeacb2")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def subject_entries(donor: str) -> dict:
    """The donor's AB subject entries, from its config."""
    path = DONOR_CONFIGS / f"{donor}.json"
    entries = json.loads(path.read_text()).get("addbiomechanics")
    if not entries:
        raise KeyError(f"{path} has no 'addbiomechanics' section (mass_kg, height_m, sex, age_years, trial)")
    missing = [k for k in ("mass_kg", "height_m", "sex", "age_years", "trial") if k not in entries]
    if missing:
        raise KeyError(f"{path}: addbiomechanics is missing {missing}")
    return entries


def subject_json(entries: dict) -> dict:
    """AB's ``_subject.json`` for a custom skeleton with the dynamics pass on (as for the paper's donors)."""
    return {
        "skeletonPreset": "custom",
        "fitDynamics": True,
        "disableDynamics": False,
        "footBodyNames": ["calcn_r", "calcn_l"],
        "subjectTags": ["healthy"],
        "shiftGRF": False,
        "runMoco": False,
        "exportMoco": False,
        "subjectConsent": True,
        "dataSource": "published",
        "massKg": entries["mass_kg"],
        "heightM": entries["height_m"],
        "ageYears": entries["age_years"],
        "sex": entries["sex"],
    }


def read_trial(donor: str, trial: str, zips: str) -> dict:
    """{file name: bytes} of the donor's trial, from whichever part of the split zip holds it."""
    found = {}
    for z in sorted(glob.glob(zips)):
        with zipfile.ZipFile(z) as zf:
            for name in ("markers.trc", "grf.mot"):
                member = f"OA_GaitRetrainingAddB/{donor}/trials/{trial}/{name}"
                if member in zf.namelist():
                    found[name] = (zf.read(member), z, member)
    if set(found) != {"markers.trc", "grf.mot"}:
        raise FileNotFoundError(f"{donor} {trial}: found {sorted(found)} in {zips}, need markers.trc and grf.mot")
    return found


def read_trial_files(markers: Path, grf: Path) -> dict:
    """{file name: bytes} of a trial given as two files (your own data): a .trc and a GRF .mot."""
    found = {}
    for name, path in (("markers.trc", Path(markers)), ("grf.mot", Path(grf))):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist (needed as the trial's {name})")
        found[name] = (path.read_bytes(), str(path.resolve()), None)
    return found


def stage(donor: str, skeleton: Path, out: Path, zips: str | None = None,
          markers: Path | None = None, grf: Path | None = None) -> Path:
    """Stage one donor's AB run folder, from the provider zips or from your own two trial files."""
    if (zips is None) == (markers is None and grf is None):
        raise ValueError("give either zips (the provider's upload) or both markers and grf (your own trial)")
    if zips is None and (markers is None or grf is None):
        raise ValueError("your own trial needs both a marker file (markers) and a GRF file (grf)")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; stage into a new folder")
    entries = subject_entries(donor)
    if zips is not None:
        trial_files = read_trial(donor, entries["trial"], zips)
    else:
        trial_files = read_trial_files(markers, grf)

    trial_dir = out / "trials" / TRIAL_NAME
    trial_dir.mkdir(parents=True)
    shutil.copyfile(skeleton, out / "unscaled_generic.osim")
    (out / "_subject.json").write_text(json.dumps(subject_json(entries), indent=2) + "\n")
    sources = {}
    for name, (data, source, member) in trial_files.items():
        (trial_dir / name).write_bytes(data)
        where = {"zip": source, "member": member} if member is not None else {"file": source}
        sources[name] = {**where, "md5": hashlib.md5(data).hexdigest()}

    record = {
        "donor": donor,
        "staged": now(),
        "subject_entries": entries,
        "config": str(DONOR_CONFIGS / f"{donor}.json"),
        "skeleton": {"path": str(Path(skeleton).resolve()), "md5": md5(Path(skeleton))},
        "trial": {"source_trial": entries["trial"], "staged_as": TRIAL_NAME, "files": sources},
        "clean_repo_commit": _git(REPO, "rev-parse", "HEAD"),
    }
    (out / "ab_stage_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"staged {donor} in {out}")
    return out


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _nimble_version(python: str) -> str | None:
    code = "import importlib.metadata as m; print(m.version('nimblephysics'))"
    try:
        return subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def engine_environment(opensim_prefix: Path, base: dict | None = None) -> dict:
    """The engine's environment: the caller's, with ``tools/addbiomechanics`` providing opensim-cmd and opensim."""
    env = dict(os.environ if base is None else base)
    env["PATH"] = os.pathsep.join([str(TOOLS / "bin"), env.get("PATH", "")])
    env["PYTHONPATH"] = str(TOOLS / "stub")
    env["AB_OPENSIM_PREFIX"] = str(opensim_prefix)
    return env


def _opensim_cmd_version(env: dict) -> str | None:
    """The version ``opensim-cmd --version`` reports, e.g. ``4.5.1-2025-04-01-5cf2dacc8``."""
    exe = shutil.which("opensim-cmd", path=env["PATH"])
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, env=env, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    head = out.strip().splitlines()[0] if out.strip() else ""
    return head.removeprefix("OpenSim version ").split(",")[0].strip() or None


def pin_problems(found: dict) -> list[str]:
    """Every way the engine software differs from ``ENGINE_PINS`` (empty: matches)."""
    problems = []
    if found.get("engine_commit") != ENGINE_PINS["engine_commit"]:
        problems.append(f"engine commit {found.get('engine_commit')} (want {ENGINE_PINS['engine_commit']})")
    if found.get("engine_dirty"):
        problems.append("engine checkout has uncommitted changes")
    if found.get("nimblephysics") != ENGINE_PINS["nimblephysics"]:
        problems.append(f"nimblephysics {found.get('nimblephysics')} (want {ENGINE_PINS['nimblephysics']})")
    if found.get("opensim_cmd") not in OPENSIM_CMD_ACCEPTED:
        problems.append(f"opensim_cmd {found.get('opensim_cmd')} (want one of {', '.join(OPENSIM_CMD_ACCEPTED)})")
    return problems


def run(run_dir: Path, engine: Path, python: str, opensim_prefix: Path | None = None,
        allow_unpinned: bool = False) -> int:
    run_dir, engine = Path(run_dir).resolve(), Path(engine).resolve()
    if not (run_dir / "_subject.json").is_file():
        raise FileNotFoundError(f"{run_dir} is not a staged run folder (no _subject.json)")
    if (run_dir / OUTPUT_NAME).exists():
        raise FileExistsError(f"{run_dir / OUTPUT_NAME} exists: this folder has already been run; stage a new one")
    opensim_prefix = opensim_prefix or os.environ.get("AB_OPENSIM_PREFIX")
    if not opensim_prefix or not (Path(opensim_prefix) / "bin" / "opensim-cmd").is_file():
        raise FileNotFoundError(
            f"no opensim-cmd under --opensim-prefix {opensim_prefix!r}: pass the prefix of the environment made from "
            f"{TOOLS / 'opensim_environment.yml'} (or set AB_OPENSIM_PREFIX); see {TOOLS / 'README.md'}")
    opensim_prefix = Path(opensim_prefix).resolve()
    env = engine_environment(opensim_prefix)
    software = {
        "engine_commit": _git(engine, "rev-parse", "HEAD"),
        "engine_dirty": bool(_git(engine, "status", "--porcelain")),
        "nimblephysics": _nimble_version(python),
        "opensim_cmd": _opensim_cmd_version(env),
    }
    problems = pin_problems(software)
    if problems and not allow_unpinned:
        raise RuntimeError(
            "the AddBiomechanics software is not what scaled the paper's donors:\n  " + "\n  ".join(problems)
            + f"\nSee {TOOLS / 'README.md'}. Pass --allow-unpinned to run anyway (recorded).")
    # The engine rewrites unscaled_generic.osim in place, so its input is hashed before it runs.
    inputs = {str(p.relative_to(run_dir)): md5(p) for p in sorted(run_dir.rglob("*"))
              if p.is_file() and p.name != "ab_run_record.json"}
    command = [python, "-X", "faulthandler", "src/engine.py", str(run_dir), OUTPUT_NAME]
    record = {
        "command": command,
        "cwd": str(engine),
        "engine_commit": software["engine_commit"],
        "engine_dirty": software["engine_dirty"],
        "nimblephysics": software["nimblephysics"],
        "python": python,
        "opensim_cmd": shutil.which("opensim-cmd", path=env["PATH"]),
        "opensim_cmd_version": software["opensim_cmd"],
        "opensim_prefix": str(opensim_prefix),
        "pythonpath": env["PYTHONPATH"],
        "omp_num_threads": env.get("OMP_NUM_THREADS"),
        "pins": dict(ENGINE_PINS, opensim_cmd_accepted=list(OPENSIM_CMD_ACCEPTED)),
        "unpinned": problems,
        "host": socket.gethostname(),
        "cpu": _cpu_model(),
        "slurm_job_id": env.get("SLURM_JOB_ID"),
        "inputs_md5": inputs,
        "started": now(),
    }
    path = run_dir / "ab_run_record.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("engine_commit", "engine_dirty", "nimblephysics", "opensim_cmd_version",
                                             "host", "cpu")}))
    with open(run_dir / "engine.log", "w") as log:
        code = subprocess.run(command, cwd=engine, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    record.update(finished=now(), exit_code=code)
    archive = run_dir / f"{OUTPUT_NAME}.zip"
    record["results_archive_md5"] = md5(archive) if archive.is_file() else None
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"engine exit {code}; log {run_dir / 'engine.log'}")
    return code


def vendor(run_dir: Path, dest: Path) -> Path:
    run_dir, dest = Path(run_dir), Path(dest)
    results = run_dir / OUTPUT_NAME
    record_path = run_dir / "ab_run_record.json"
    if not record_path.is_file() or json.loads(record_path.read_text()).get("exit_code") != 0:
        raise RuntimeError(f"{run_dir} has no successful run record; run it first")
    if dest.exists():
        raise FileExistsError(f"{dest} exists; vendoring never writes into an existing folder")
    for name in VENDORED:
        if not (results / name).is_dir():
            raise FileNotFoundError(f"{results / name} missing")
    dest.mkdir(parents=True)
    for name in VENDORED:
        shutil.copytree(results / name, dest / name, symlinks=False)
    shutil.copyfile(run_dir / f"{OUTPUT_NAME}.zip", dest / "addbiomechanics_results.zip")
    for name in ("ab_stage_record.json", "ab_run_record.json", "_subject.json"):
        shutil.copyfile(run_dir / name, dest / name)
    files = {str(p.relative_to(dest)): md5(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    (dest / "ab_vendor_record.json").write_text(json.dumps(
        {"vendored": now(), "from": str(run_dir.resolve()), "md5": files}, indent=2) + "\n")
    print(f"vendored {len(files)} files into {dest}")
    return dest


def models_dir(folder: Path) -> Path:
    """``Models/`` of a vendored AB folder or of a run folder (``osim_results/Models``)."""
    for candidate in (Path(folder) / "Models", Path(folder) / OUTPUT_NAME / "Models"):
        if (candidate / "match_markers_and_physics.osim").is_file():
            return candidate
    raise FileNotFoundError(f"{folder}: no Models/match_markers_and_physics.osim (nor under {OUTPUT_NAME}/)")


def knee_factor(folder: Path) -> dict:
    """The knee scale factor ``s_wa`` of one AB output, with the two long-axis factors it averages."""
    from nsosim.scaling.config import LONG_AXIS_INDEX
    from nsosim.scaling.scale_factors import read_ab_factors

    models = models_dir(folder)
    factors = read_ab_factors(models / "match_markers_and_physics.osim", models / "unscaled_generic.osim")
    femur, tibia = factors["femur_r"][LONG_AXIS_INDEX], factors["tibia_r"][LONG_AXIS_INDEX]
    return {"folder": str(folder), "s_wa": (femur + tibia) / 2.0, "femur_r": femur, "tibia_r": tibia}


def print_knee_factors(folders) -> None:
    rows = [knee_factor(Path(f)) for f in folders]
    first = rows[0]["s_wa"]
    for i, r in enumerate(rows):
        diff = "" if i == 0 else f"  {100 * (r['s_wa'] / first - 1):+.3f} % vs the first"
        print(f"s_wa {r['s_wa']:.9f}  (femur_r {r['femur_r']:.6f}, tibia_r {r['tibia_r']:.6f})  "
              f"{r['folder']}{diff}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="build an AB run folder for one donor")
    s.add_argument("--donor", required=True)
    s.add_argument("--skeleton", required=True, type=Path, help="the AB custom skeleton (.osim)")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--zips", default=None,
                   help="glob of the provider's AB upload zips (a split set; each holds "
                        "OA_GaitRetrainingAddB/<donor>/trials/<trial>/). They are raw motion data and are "
                        "not part of the shipped data root")
    s.add_argument("--markers", type=Path, default=None,
                   help="your own trial's marker file (.trc), instead of --zips; marker names must match the "
                        "skeleton's (docs/scaling_comak_model.md)")
    s.add_argument("--grf", type=Path, default=None,
                   help="your own trial's ground reaction forces (.mot), with --markers")
    r = sub.add_parser("run", help="run the AB engine on a staged folder")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--engine", required=True, type=Path, help="AddBiomechanics/server/engine")
    r.add_argument("--python", default=sys.executable, help="interpreter of the engine's environment")
    r.add_argument("--opensim-prefix", type=Path, default=None,
                   help="prefix of the environment made from tools/addbiomechanics/opensim_environment.yml "
                        "(default: $AB_OPENSIM_PREFIX)")
    r.add_argument("--allow-unpinned", action="store_true",
                   help="run with an engine, nimblephysics or opensim-cmd other than the paper's (recorded)")
    v = sub.add_parser("vendor", help="copy a finished run's outputs into a data root")
    v.add_argument("run_dir", type=Path)
    v.add_argument("--dest", required=True, type=Path)
    k = sub.add_parser("knee-factor", help="print the knee scale factor (s_wa) of AB outputs, the first as reference")
    k.add_argument("folders", nargs="+", type=Path, help="vendored AB folders or AB run folders")
    args = p.parse_args(argv)
    if args.cmd == "stage":
        if args.zips is None and (args.markers is None or args.grf is None):
            p.error("stage needs --zips (the provider's upload) or both --markers and --grf (your own trial)")
        if args.zips is not None and (args.markers is not None or args.grf is not None):
            p.error("give --zips or --markers/--grf, not both")
        stage(args.donor, args.skeleton, args.out, args.zips, args.markers, args.grf)
    elif args.cmd == "run":
        sys.exit(run(args.run_dir, args.engine, args.python, args.opensim_prefix, args.allow_unpinned))
    elif args.cmd == "knee-factor":
        print_knee_factors(args.folders)
    else:
        vendor(args.run_dir, args.dest)


if __name__ == "__main__":
    main()
