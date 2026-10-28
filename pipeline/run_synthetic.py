"""Figure 3's synthetic knees, end to end: shape axis, build, simulate, analyse.

Synthetic knees are placed along the healthy-to-mild-OA shape axis, built like
cohort knees and walked through the nine measured gaits. Every step reuses the
cohort's code and configuration:

* ``axis``      -- derive the axis and each synthetic knee's descriptor (CPU, seconds);
* ``build``     -- decode one knee and build its model with the cohort's model
                   builder and ``config/reconstruction.json`` (GPU, minutes);
* ``simulate``  -- scale it to one gait donor and run the cohort's solver chain
                   with that donor's ``config/measured_gait/<donor>.json`` (CPU);
* ``cells``     -- print the (index, label, gait) table a job array iterates over;
* ``status``    -- summarise which cells finished;
* ``manifest``  -- hash the finished tree into the simulations stage manifest.

The tree it writes, ``02_simulations/synthetic_shape_axis``, is the one
``pipeline/derive_figure3_data.py`` reads with no options.

A simulation may change only where its model and results live
(``SIMULATE_OVERRIDES``); the solver settings are the donor config's own. The
defaults in the solver code differ from the cohort's, so a config written from
scratch would fall back to them unnoticed; the donor configs are reused instead.

    python pipeline/run_synthetic.py axis --config config/synthetic_shape_axis.json --verify
    python pipeline/run_synthetic.py build --config ... --label fig5_sigma_p0
    python pipeline/run_synthetic.py simulate --config ... --label fig5_sigma_p0 --gait Subject_102
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa.config import Config, ConfigError, Label, load_config  # noqa: E402
from comak_knee_oa.paths import data_root, stage_dir  # noqa: E402

# Keys a synthetic simulation may set in the donor's measured-gait config.
SIMULATE_OVERRIDES = {
    ("run", "results_dir"),
    ("model", "built_model"),
    ("model", "scaled_model"),
}

PUBLISHED_TREE = "synthetic_shape_axis"


# ---------------------------------------------------------------- helpers ----


def _tree(synth: Config) -> Path:
    from build_synthetic_knee import synthetic_tree

    tree = synth.resolve(synthetic_tree(synth), where="run.root")
    published = stage_dir("simulations", PUBLISHED_TREE)
    if tree.resolve() != published.resolve():
        # Not an error -- a scratch run elsewhere is legitimate -- but its results
        # are not what Figure 3 reads, so say so.
        print(f"note: run.root writes {tree}, not the published tree {published}",
              file=sys.stderr)
    return tree


def labels(synth: Config) -> list[str]:
    """The synthetic knees, in shape-axis order (from ``shape_axis.json``)."""
    summary = _tree(synth) / "shape_axis.json"
    if not summary.is_file():
        raise FileNotFoundError(f"{summary} not found; run `run_synthetic.py axis` first")
    return list(json.loads(summary.read_text())["shapes"])


def gaits(synth: Config) -> list[str]:
    return list(synth.require("simulate", "gaits"))


def cells(synth: Config) -> list[tuple[int, str, str]]:
    """Every (index, label, gait) cell, labels outermost."""
    table = [(label, gait) for label in labels(synth) for gait in gaits(synth)]
    return [(i, label, gait) for i, (label, gait) in enumerate(table)]


def simulate_config(synth: Config, label: str, gait: str) -> Config:
    """The donor's measured-gait config, pointed at this synthetic knee."""
    from build_synthetic_knee import build_config, synthetic_tree
    from reconstruct_knee import built_model_path

    gait_dir = Path(synth.origin).parent / synth.get("simulate", "gait_config_dir", "measured_gait")
    donor = load_config(gait_dir / f"{gait}.json")
    if donor.get("run", "condition") != "measured":
        raise ConfigError(f"{donor.origin} is not a measured-gait condition")
    data = copy.deepcopy(donor.data)

    built = built_model_path(build_config(synth).for_knee(Label(label)))
    results = f"{synthetic_tree(synth)}/{label}/gait_{gait}"
    overrides = {
        ("run", "results_dir"): results,
        ("model", "built_model"): str(built.relative_to(data_root())),
        ("model", "scaled_model"): f"{results}/scaled_model/scaled.osim",
    }
    for (section, key), value in overrides.items():
        if (section, key) not in SIMULATE_OVERRIDES:
            raise ConfigError(f"{section}.{key} may not be overridden for a synthetic run")
        data[section][key] = value
    return Config(data, origin=donor.origin)


# ---------------------------------------------------------------- commands ----


def cmd_axis(args) -> int:
    from derive_shape_axis import derive_shape_axis

    synth = load_config(args.config)
    derive_shape_axis(synth, _tree(synth), verify=args.verify,
                      allow_missing=args.allow_missing, force=args.force)
    return 0


def cmd_build(args) -> int:
    from build_synthetic_knee import build

    synth = load_config(args.config)
    if args.label not in labels(synth):
        raise ConfigError(f"unknown label {args.label!r}; known: {labels(synth)}")
    return 0 if build(synth, args.label, check_only=args.check) else 1


def cmd_simulate(args) -> int:
    from simulate_gait import STAGES, simulate, verify_outputs

    if args.stop_after is not None and args.stop_after not in STAGES:
        raise ConfigError(f"--stop-after must be one of {STAGES}")
    synth = load_config(args.config)
    if args.label not in labels(synth):
        raise ConfigError(f"unknown label {args.label!r}")
    if args.gait not in gaits(synth):
        raise ConfigError(f"unknown gait {args.gait!r}; configured: {gaits(synth)}")
    config = simulate_config(synth, args.label, args.gait)
    knee = Label(args.label)
    results_dir = config.for_knee(knee).path("run", "results_dir")

    if args.check:
        result = simulate(config, knee, stop_after=args.stop_after, check_only=True)
        return 0 if result is True else 1

    from build_synthetic_knee import library_paths
    from comak_knee_oa.manifest import _code_versions, cpu_model, environment_record

    started = time.time()
    status = {
        "label": args.label,
        "gait": args.gait,
        "return": "started",
        "problems": [],
        "ok": False,
        "node": socket.gethostname(),
        "cpu_model": cpu_model(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": None,
        "code": _code_versions(),
        "libraries": library_paths(),
        "environment": environment_record(),
    }
    # Written before the solver starts and replaced when it ends, so a cell the
    # scheduler kills reads as "started, no result" rather than "not run" -- and
    # never keeps an earlier attempt's result.
    status_file = results_dir / "status.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(status, indent=2) + "\n")

    try:
        result = simulate(config, knee, stop_after=args.stop_after)
    except Exception as exc:  # recorded in status.json so a crash is not read as "not run"
        import traceback

        traceback.print_exc()
        result = f"exception: {type(exc).__name__}: {exc}"

    problems = [] if args.stop_after else verify_outputs(results_dir)
    status.update(
        {
            "return": result if isinstance(result, (bool, str)) else repr(result),
            "problems": problems,
            "ok": result is True and not problems,
            "elapsed_s": round(time.time() - started, 1),
        }
    )
    status_file.write_text(json.dumps(status, indent=2) + "\n")
    print(f"\n{args.label} x {args.gait}: {result!r}"
          + (f"; missing {problems}" if problems else ""))
    return 0 if status["ok"] or str(result).startswith("ok_") else 1


def cmd_cells(args) -> int:
    for index, label, gait in cells(load_config(args.config)):
        print(f"{index}\t{label}\t{gait}")
    return 0


def cmd_labels(args) -> int:
    for index, label in enumerate(labels(load_config(args.config))):
        print(f"{index}\t{label}")
    return 0


def cmd_status(args) -> int:
    synth = load_config(args.config)
    tree = _tree(synth)
    counts: dict[str, int] = {}
    rows = []
    for index, label, gait in cells(synth):
        status_file = tree / label / f"gait_{gait}" / "status.json"
        if status_file.is_file():
            status = json.loads(status_file.read_text())
            if status.get("ok"):
                state = "ok"
            elif status.get("return") == "started":
                state = "started, no result (running or killed)"
            else:
                state = f"failed:{status.get('return')}"
            elapsed = status.get("elapsed_s")
        else:
            state, elapsed = "not run", None
        counts[state] = counts.get(state, 0) + 1
        rows.append((index, label, gait, state, elapsed))
    if args.verbose:
        for index, label, gait, state, elapsed in rows:
            took = f"{elapsed / 60:.0f} min" if elapsed else ""
            print(f"{index:3d}  {label:24s} {gait:14s} {state:28s} {took}")
    for state, count in sorted(counts.items()):
        print(f"{state:30s} {count}")
    return 0


def cmd_manifest(args) -> int:
    """Hash the synthetic tree into ``02_simulations/_manifest.json``.

    The stage manifest also covers the vendored reference arm, so this is a
    merge: every record under the synthetic tree is replaced by a fresh hash of
    what is on disk now, and every other record is kept as it was. Re-reading
    the reference arm (~0.6 TB) to rewrite records nobody changed would cost
    hours for no added confidence.
    """
    import vendor_inputs as vi
    from comak_knee_oa.manifest import read_manifest, record_files, write_manifest

    synth = load_config(args.config)
    tree = _tree(synth)
    prefix = tree.resolve().relative_to(data_root().resolve()).as_posix() + "/"
    existing = read_manifest("simulations")
    kept = [r for r in existing.outputs if not r.path.startswith(prefix)]
    print(f"hashing {tree} ...")
    fresh = record_files([tree])
    records = sorted(kept + fresh, key=lambda r: r.path)

    notes = {k: v for k, v in existing.notes.items()
             if k not in ("synthetic_vendored", "synthetic_source_roots")}
    notes["synthetic"] = {
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produced_by": "pipeline/run_synthetic.py with config/synthetic_shape_axis.json",
        "layout": ("axis at _axis/ and shape_axis.json; per knee {label}/input_descriptors, "
                   "{label}/geometries_nsm_similarity, {label}/custom_nsm_*/ (the built "
                   "model) and {label}/gait_{gait}/ (one run per gait, no replicates)"),
        "files": len(fresh),
    }
    notes["coverage"] = vi._partial_coverage_note("simulations", records)
    notes["contents"] = vi.stage_coverage("simulations", records)
    path = write_manifest("simulations", inputs=[], outputs=records, notes=notes)
    print(f"wrote {path}: {len(fresh):,} synthetic record(s), "
          f"{len(existing.outputs) - len(kept):,} replaced, {len(kept):,} kept")
    return 0


# ------------------------------------------------------------------- main ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_synthetic.py", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p):
        p.add_argument("--config", required=True, help="e.g. config/synthetic_shape_axis.json")
        return p

    p = with_config(sub.add_parser("axis", help="derive the shape axis and every synthetic descriptor"))
    p.add_argument("--verify", action="store_true",
                   help="re-derive and check against the tree on disk; writes nothing")
    p.add_argument("--allow-missing", action="store_true",
                   help="compute the axis without cohort knees that lack a fit (recorded)")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing descriptor arrays that differ from the new axis")
    p.set_defaults(func=cmd_axis)

    p = with_config(sub.add_parser("build", help="decode and build one synthetic knee (GPU)"))
    p.add_argument("--label", required=True)
    p.add_argument("--check", action="store_true", help="resolve inputs only; needs no GPU")
    p.set_defaults(func=cmd_build)

    # The solver stack is imported by `simulate` only, so the table commands that
    # job arrays parse (`labels`, `cells`) stay free of library output.
    p = with_config(sub.add_parser("simulate", help="walk one synthetic knee through one gait"))
    p.add_argument("--label", required=True)
    p.add_argument("--gait", required=True)
    p.add_argument("--check", action="store_true", help="resolve inputs only; runs no solver")
    p.add_argument("--stop-after", default=None, help="stop after this solver stage")
    p.set_defaults(func=cmd_simulate)

    with_config(sub.add_parser("cells", help="print the cell table")).set_defaults(func=cmd_cells)
    with_config(sub.add_parser("labels", help="print index<TAB>label")).set_defaults(func=cmd_labels)
    p = with_config(sub.add_parser("status", help="summarise finished cells"))
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_status)
    with_config(sub.add_parser("manifest", help="hash the tree into the stage manifest")
                ).set_defaults(func=cmd_manifest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
