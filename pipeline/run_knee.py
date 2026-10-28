"""Run one knee through one stage of the pipeline, with no database or scheduler.

Subcommands:

- ``reconstruct``: fit the neural shape models to the knee's segmentation meshes
  and build its subject-specific OpenSim model (GPU). Input: the knee's meshes
  under ``00_inputs``. Output: the model under ``01_models``.
- ``rebuild``: rebuild the OpenSim model from the saved shape-model fit (no fitting).
- ``simulate``: run the gait simulation for one condition (scaling, COMAK
  inverse kinematics, COMAK, inverse dynamics, joint mechanics; CPU). Output:
  ``02_simulations/<gait>/<knee>/comak_<run>/``.
- ``verify``: check on disk that every solver stage left its output.
- ``reproduce``: re-simulate the knee into a scratch tree and compare it with
  a published result.

``run_queue.py`` is an optional worker loop for the ``simulate`` stage: it takes
knees one at a time from a SQLite work list and makes the same ``simulate`` call.
A result never depends on the work list, so one knee can be run or reproduced
with this script alone.

    python pipeline/run_knee.py reconstruct \\
        --config config/reconstruction.json --knee 9003316_RIGHT

    python pipeline/run_knee.py rebuild \\
        --config config/reconstruction.json --knee 9003316_RIGHT

    python pipeline/run_knee.py simulate \\
        --config config/measured_gait/Subject_102.json --knee 9003316_RIGHT

    python pipeline/run_knee.py simulate  --config ... --knee ... --check
    python pipeline/run_knee.py verify    --config ... --knee ...
    python pipeline/run_knee.py reproduce --published <published results dir> --knee ...
    python pipeline/run_knee.py reproduce --config ... --knee ...   # published = the root's baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa.compare import compare_simulations, report_simulation  # noqa: E402
from comak_knee_oa.config import Config, Knee, load_config  # noqa: E402
from comak_knee_oa.paths import data_root, declaration  # noqa: E402
from reconstruct_knee import rebuild, reconstruct  # noqa: E402
from simulate_gait import STAGES, simulate, verify_outputs  # noqa: E402


def run_simulate(args) -> int:
    knee = Knee.parse(args.knee, timepoint=args.timepoint)
    config = load_config(args.config)
    result = simulate(
        config,
        knee,
        stop_after=args.stop_after,
        check_only=args.check,
        force=args.force,
    )
    print(f"\n{knee.name}: {result!r}")
    # True means finished. An "ok_stopped_after_*" marker means it stopped where
    # it was told to. Anything else names how it failed, and is not success.
    if result is True or str(result).startswith("ok_"):
        return 0
    return 1


def run_reconstruct(args) -> int:
    knee = Knee.parse(args.knee, timepoint=args.timepoint)
    config = load_config(args.config)
    ok = reconstruct(config, knee, check_only=args.check, force=args.force)
    return 0 if ok else 1


def run_rebuild(args) -> int:
    knee = Knee.parse(args.knee, timepoint=args.timepoint)
    config = load_config(args.config)
    ok = rebuild(config, knee, check_only=args.check)
    return 0 if ok else 1


def run_verify(args) -> int:
    """Check on disk that every solver stage left its output behind.

    Independent of whatever recorded the run as finished: a stage
    can fail, have its exception caught, and leave the run marked complete with
    nothing behind it -- so completeness is checked against the files.
    """
    knee = Knee.parse(args.knee, timepoint=args.timepoint)
    config = load_config(args.config).for_knee(knee)
    results_dir = config.path("run", "results_dir")
    missing = verify_outputs(results_dir)
    if missing:
        print(f"INCOMPLETE {knee.name} ({results_dir})")
        for item in missing:
            print(f"  missing {item}")
        return 1
    print(f"COMPLETE {knee.name}")
    return 0


def run_reproduce(args) -> int:
    """Re-simulate one knee into a scratch tree, then compare it with the
    published result.

    This is the end-to-end check of the whole artifact: it exercises scaling,
    the solver chain, and the result layout, and passes only if the numbers
    that come out match the numbers that were published.

    *Published* is ``--published`` (a results folder such as
    ``<downloaded root>/02_simulations/reference/<knee>/comak_<run>``), or else
    the same knee and gait in the root's baseline (the downloaded root
    ``new_root create --baseline`` named), under the run that root declares.
    The config defaults to the one the published run recorded in its
    ``run_record.json``: a shipped config file can differ from the one a run
    used, and only the recorded one reproduces it. ``--use-config`` uses ``--config`` instead.

    Nothing published is written to: every output goes under
    ``02_simulations/_reproduction/<gait folder>/<knee>/`` of the current root,
    where the gait folder is the one the published simulations live in
    (``reference``, ``Subject_121``). The model simulated is this root's own
    build of the knee (``run_knee.py rebuild``).

    Expectations (``docs/reproduce.md``): on the CPU model the published run
    record names, in the same environment and from the same model file, every
    value is identical. On another CPU type the result is judged on knee
    kinematics and contact pressure and force against the largest difference
    measured between two CPU types (``comak_knee_oa.compare``).
    """
    knee = Knee.parse(args.knee, timepoint=args.timepoint)
    if args.use_config and not args.config:
        raise SystemExit("reproduce: --use-config needs --config")
    published = _published_dir(args, knee)
    record_file = published / "run_record.json"
    record = json.loads(record_file.read_text()) if record_file.is_file() else None
    if record and record.get("knee") not in (None, knee.name):
        print(f"{record_file} is the record of {record['knee']}, not {knee.name}")
        return 1

    file_config = load_config(args.config).for_knee(knee) if args.config else None
    if record and "config" in record and not args.use_config:
        config = Config(record["config"], origin=record_file).for_knee(knee)
        print(f"config: as the published run recorded it ({record_file})")
        if file_config is not None:
            _report_config_differences(config.data, file_config.data, str(args.config))
    elif file_config is not None:
        config = file_config
        print(f"config: {args.config}" + ("" if record else " (the published run left no run record)"))
    else:
        print(f"{published} has no run_record.json holding a config; pass --config")
        return 1

    # The folder the condition's simulations live in (``reference``, ``Subject_121``), not its
    # ``run.gait`` key, which for the reference gait is ``default``.
    parts = Path(config.data.get("run", {}).get("results_dir", "")).parts
    gait = (parts[parts.index("02_simulations") + 1] if "02_simulations" in parts[:-1]
            else config.get("run", "gait", "reference"))
    scratch_base = f"02_simulations/_reproduction/{gait}/{knee.directory}"
    data = config.data
    data["run"]["results_dir"] = f"{scratch_base}/comak"
    if "scaled_model" in data.get("model", {}):
        data["model"]["scaled_model"] = f"{scratch_base}/scaled_model/scaled.osim"
    scratch_config = Config(data, origin=config.origin).for_knee(knee)
    scratch = scratch_config.path("run", "results_dir")

    if not args.compare_only:
        model = scratch_config.get("model", "built_model")
        if model is not None and not scratch_config.resolve(model).is_file():
            print(f"no model for {knee.name} in this root: {scratch_config.resolve(model)}\n"
                  "Build it first: python pipeline/run_knee.py rebuild --config config/reconstruction.json "
                  f"--knee {knee.name}")
            return 1
        result = simulate(scratch_config, knee)
        print(f"\n{knee.name}: {result!r}")
        if result is not True:
            print("simulation did not complete; nothing to compare")
            return 1

    print()
    _report_scale_factors(published, scratch_config)
    comparison = compare_simulations(published, scratch)
    print(report_simulation(comparison))
    cpu_differs = any(d.startswith("CPU model") for d in comparison.environment)
    if comparison.verdict == "BIT-IDENTICAL":
        print(f"\nREPRODUCED {knee.name}: bit-identical to the published result")
        return 0
    if comparison.verdict == "WITHIN CROSS-CPU ENVELOPE" and cpu_differs:
        print(f"\nAGREES {knee.name}: on another CPU type, within the measured cross-CPU differences")
        return 0
    print(f"\nNOT REPRODUCED {knee.name}: {comparison.verdict.lower()}"
          + ("" if cpu_differs else "; on the published run's CPU model the result should be bit-identical"))
    return 1


def _published_dir(args, knee: Knee) -> Path:
    """The published results folder: ``--published``, else the root's baseline."""
    if args.published:
        published = Path(args.published).absolute()
    else:
        if not args.config:
            raise SystemExit("reproduce: pass --published <results dir>, or --config to find the same "
                             "knee and gait in this root's baseline")
        decl = declaration() or {}
        baseline = decl.get("baseline")
        if not baseline:
            raise SystemExit(f"reproduce: the data root {data_root()} names no baseline; pass --published")
        baseline_decl = declaration(Path(baseline)) or {}
        baseline_run = baseline_decl.get("run_name")
        if not baseline_run:
            raise SystemExit(f"reproduce: the baseline {baseline} declares no run name, so its results folder "
                             "cannot be derived; pass --published <its results dir for this knee>")
        template = load_config(args.config).for_knee(knee).data["run"]["results_dir"]
        published = Path(baseline) / template.replace("{run}", baseline_run)
    if not published.is_dir():
        raise SystemExit(f"reproduce: no published results folder at {published}")
    print(f"published: {published}")
    return published


def _report_config_differences(recorded: dict, given: dict, name: str, prefix: str = "") -> None:
    """Print each value where ``given`` (a config file) differs from the recorded config."""
    for key in sorted(set(recorded) | set(given)):
        if key.startswith("_"):
            continue
        a, b = recorded.get(key), given.get(key)
        where = f"{prefix}{key}"
        if isinstance(a, dict) and isinstance(b, dict):
            _report_config_differences(a, b, name, where + ".")
        elif a != b:
            print(f"  note: {name} has {where} = {b!r}; the published run used {a!r} (using the recorded value)")


def _report_scale_factors(published: Path, scratch_config: Config) -> None:
    """Print the published and reproduced knee scale factors, when the condition scales."""
    value = scratch_config.get("model", "scaled_model")
    if value is None:
        return
    files = {
        "published": published.parent / "scaled_model" / "scaled_scaling_report.json",
        "reproduced": scratch_config.resolve(value).parent / "scaled_scaling_report.json",
    }
    for label, report_file in files.items():
        if report_file.is_file():
            factor = json.loads(report_file.read_text()).get("wa_scale")
            print(f"knee scale factor ({label}): {factor}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_knee.py", description="Run or check one knee."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", required=True, help="path to the run config")
        p.add_argument("--knee", required=True, help="e.g. 9003316_RIGHT")
        p.add_argument("--timepoint", default="00m")
        return p

    p_recon = common(
        sub.add_parser(
            "reconstruct",
            help="fit the shape models to this knee and build its OpenSim model",
        )
    )
    p_recon.add_argument(
        "--check", action="store_true",
        help="resolve and report every input, then stop. Needs no GPU.",
    )
    p_recon.add_argument(
        "--force", action="store_true",
        help="rebuild a model that already exists",
    )
    p_recon.set_defaults(func=run_reconstruct)

    p_rebuild = common(
        sub.add_parser(
            "rebuild",
            help="rebuild this knee's OpenSim model from its saved shape-model fit "
            "(no fitting)",
        )
    )
    p_rebuild.add_argument(
        "--check", action="store_true",
        help="resolve and report every input, then stop. Needs no GPU.",
    )
    p_rebuild.set_defaults(func=run_rebuild)

    p_sim = common(sub.add_parser("simulate", help="simulate this knee walking"))
    p_sim.add_argument(
        "--check", action="store_true",
        help="resolve and report every input, then stop. Runs no solver.",
    )
    p_sim.add_argument(
        "--stop-after", choices=STAGES, default=None,
        help="stop after this solver stage",
    )
    p_sim.add_argument(
        "--force", action="store_true",
        help="redo work that is already complete (currently: body scaling)",
    )
    p_sim.set_defaults(func=run_simulate)

    p_verify = common(sub.add_parser("verify", help="check this knee's outputs on disk"))
    p_verify.set_defaults(func=run_verify)

    p_repro = sub.add_parser(
        "reproduce",
        help="re-simulate this knee into a scratch tree and compare it with the published result",
    )
    p_repro.add_argument("--knee", required=True, help="e.g. 9003316_RIGHT")
    p_repro.add_argument("--timepoint", default="00m")
    p_repro.add_argument(
        "--published", type=Path, default=None,
        help="the published results folder (02_simulations/<gait>/<knee>/comak_<run>); default: the same "
        "knee and gait in this root's baseline, which needs --config",
    )
    p_repro.add_argument(
        "--config", default=None,
        help="a run config; used to find the published folder in the baseline, and to run only with "
        "--use-config (by default the config the published run recorded is used)",
    )
    p_repro.add_argument(
        "--use-config", action="store_true",
        help="simulate with --config instead of the config recorded in the published run_record.json",
    )
    p_repro.add_argument(
        "--compare-only", action="store_true",
        help="skip the simulation and compare an existing scratch tree",
    )
    p_repro.set_defaults(func=run_reproduce)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
