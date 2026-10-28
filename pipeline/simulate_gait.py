"""Stage 02: simulate one knee walking, and write what the joint did.

There are two kinds of gait condition:

reference -- one overground gait trial, applied to every knee at the size the
reconstruction produced. Holding the motion fixed isolates the effect of
anatomy: any difference in contact mechanics between knees is a difference
between the knees.

measured -- one gait donor's measured treadmill motion and ground reaction
forces, with each knee scaled to that donor's body. Repeated across donors,
this shows how much the gait moves the answer.

Both go through the same solver chain (``comak_knee_oa.comak_chain``): patella
optimisation, COMAK inverse kinematics, COMAK, inverse dynamics and joint
mechanics. They differ only in three prepared inputs -- the model, the motion
and loads, and the time window -- so they share one file and cannot drift apart.

Inputs: the knee's built model (``01_models``, from stage 01) and the condition
config (``config/reference_gait.json`` or ``config/measured_gait/<donor>.json``),
which names the markers and external loads and, for a measured condition, the
donor's AddBiomechanics body model. Output: ``02_simulations/<gait>/<knee>/comak_<run>/``,
holding the solver outputs and a ``run_record.json``. Runs on CPU.

Scaling, for the measured condition
-----------------------------------
Stage 01 builds each knee once, at the size of the reference anatomy (its
shape models are fitted with similarity registration). For a measured
condition this stage then scales that built model to the gait donor's
AddBiomechanics body, so the personalised knee is sized to the body it walks
in: scaling the body alone would leave a reference-size knee inside a scaled
body, and contact would be computed on the wrong-size surface.

The knee scale factor ``s_wa`` (``wa_scale`` in the scaling report) is the mean
of the femur and tibia long-axis (superior-inferior) scale factors of the
donor's body; nsosim's ``"WA"`` (weighted-average) mode applies it
isotropically to every knee body. The factor belongs to the donor, so any
error in it is the same for every knee in a condition and does not average out
across knees.

Run names: ``v3`` is the run this paper reports; ``v2`` is an earlier release
of the dataset that the code can still read, and the options below that
reproduce it are named for it.

Running it
----------
One knee, no queue::

    python pipeline/run_knee.py simulate --config config/measured_gait/Subject_102.json \\
        --knee 9003316_RIGHT

Resolve and report every input without running a solver::

    python pipeline/run_knee.py simulate --config ... --knee ... --check
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comak_knee_oa.comak_chain import (  # noqa: E402
    STAGES,
    build_muscle_weights,
    reference_strains,
    run_comak_chain,
)
from comak_knee_oa.config import Config, ConfigError, Knee  # noqa: E402

__all__ = ["simulate", "STAGES", "CONDITIONS"]

CONDITIONS = ("reference", "measured")

# The reference gait's time window. Tied to the one trial the reference
# condition uses, so it is a constant of that trial rather than a parameter.
REFERENCE_TIMING = {
    "start_time_ik": 1.16,
    "stop_time_ik": 2.36,
    "start_time_comak": 1.26,
    "stop_time_comak": 1.95,
}


# ------------------------------------------------------- stance detection ----


def read_storage(path: Path) -> tuple[list[str], np.ndarray]:
    """Read an OpenSim storage file into (column names, values).

    Whitespace around fields is stripped rather than assumed absent: the
    motion-capture pipeline writes bare tab-separated columns, while the
    reference trial's file is column-aligned with padding, and both must read
    identically.
    """
    lines = Path(path).read_text().splitlines()
    header = next(
        i for i, line in enumerate(lines) if line.strip().lower().startswith("time")
    )
    columns = [c.strip() for c in lines[header].split("\t") if c.strip()]
    values = np.array(
        [
            [float(x) for x in line.split("\t") if x.strip()]
            for line in lines[header + 1 :]
            if line.strip()
        ]
    )
    return columns, values


def stance_windows(grf_file, vertical_column, threshold_N, min_frames):
    """Contiguous spans where the right foot is loaded above ``threshold_N``.

    Returns ``(t_start, t_stop, n_frames, peak_N)`` per span, in time order.
    Spans shorter than ``min_frames`` are dropped: at typical capture rates a
    handful of samples over threshold is force-plate noise or a brush of the
    plate edge, not a step.
    """
    columns, values = read_storage(grf_file)
    if vertical_column not in columns:
        raise KeyError(
            f"no column {vertical_column!r} in {grf_file}. Present: {columns}"
        )
    time = values[:, columns.index("time")]
    vertical = values[:, columns.index(vertical_column)]
    loaded = vertical > threshold_N

    spans, i = [], 0
    while i < len(loaded):
        if not loaded[i]:
            i += 1
            continue
        j = i
        while j < len(loaded) and loaded[j]:
            j += 1
        if j - i >= min_frames:
            spans.append(
                (float(time[i]), float(time[j - 1]), j - i, float(vertical[i:j].max()))
            )
        i = j
    return spans


def timing_from_stance(spans, stance_index, ik_pad_s):
    """Turn one stance span into the inverse-kinematics and COMAK windows.

    COMAK runs over stance itself. Inverse kinematics runs over a padded window
    so that stance sits strictly inside it and the tracked kinematics are
    settled by the time COMAK starts.
    """
    if not spans:
        raise RuntimeError(
            "no stance found above the force threshold -- check the vertical GRF "
            "column name and the threshold before assuming the trial is bad"
        )
    if stance_index >= len(spans):
        raise IndexError(
            f"stance_index {stance_index} but only {len(spans)} stance span(s) found"
        )
    start, stop, n_frames, peak = spans[stance_index]
    return {
        "start_time_comak": start,
        "stop_time_comak": stop,
        "start_time_ik": start - ik_pad_s,
        "stop_time_ik": stop + ik_pad_s,
        "stance_n_frames": n_frames,
        "stance_peak_force_N": peak,
    }


def pad_comak_window(timing: dict, pad_s: float) -> dict:
    """Widen the COMAK window by ``pad_s`` on each side (inverse kinematics unchanged).

    COMAK does not run on the marker frames. It resamples the inverse-kinematics
    output at 0.01 s steps, starting from the first IK frame, and keeps the
    resampled times inside [start, stop]. A measured trial's frames are 100.007
    Hz, stamped to four decimals, so the resampled grid sits a hair off the
    stance frames: the first lands just *below* the stance start (accumulated
    floating-point error) and the last just *past* the stance stop (the 100.007
    Hz drift). Both are dropped -- every measured knee simulated stance minus
    heel strike and toe-off (70 GRF frames in, 68 COMAK frames out). The
    reference trial is on an exact 100 Hz grid and loses nothing.

    Padding by half the resampling step keeps every grid point that stands for
    a stance frame and admits no other: it recovers the measured arm's two
    frames and leaves the reference arm's 70 unchanged. ``pad_s = 0``
    reproduces the v2 run, which did not pad.
    """
    if not pad_s:
        return timing
    if not 0 < pad_s < 0.01:
        raise ConfigError(
            f"comak_timing.comak_window_pad_s = {pad_s}: must be below COMAK's 0.01 s "
            "resampling step (0.005 = half a step), or the window admits extra frames"
        )
    padded = dict(timing)
    padded["start_time_comak"] = timing["start_time_comak"] - pad_s
    padded["stop_time_comak"] = timing["stop_time_comak"] + pad_s
    padded["comak_window_pad_s"] = pad_s
    if not (padded["start_time_ik"] < padded["start_time_comak"]
            and padded["stop_time_comak"] < padded["stop_time_ik"]):
        raise ConfigError(f"padded COMAK window {padded} is not inside the IK window")
    return padded


# ------------------------------------------------------------------ prep ----


def absolutise_external_loads(source_xml: Path, grf_file: Path, target_xml: Path) -> Path:
    """Copy an external-loads file, pointing its data reference at the GRF.

    The motion-capture pipeline writes a bare filename there, which OpenSim
    resolves against the working directory -- so the same file behaves
    differently depending on where the job was launched from.
    """
    tree = ET.parse(source_xml)
    element = tree.getroot().find(".//datafile")
    if element is None:
        raise ValueError(f"no <datafile> element in {source_xml}")
    element.text = str(grf_file)
    target_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target_xml, xml_declaration=True, encoding="UTF-8")
    return target_xml


def scaling_options(config: Config) -> dict:
    """Every option the body scaling is called with, besides its paths.

    Kept in one place because it is also what the scaled model's provenance
    records: a changed option makes an existing scaled model stale, and the
    skip check below will say so rather than reuse it.

    A measured-gait config's ``scaling`` section must set:

    ``mass_rule`` (required: ``"volume"`` or ``"as_run"``)
        How the scaled model's per-body masses are set (nsosim
        ``scale_comak_model``). ``"volume"``: each body's mass grows with its
        volume (sx * sy * sz) and the total is AddBiomechanics' -- OpenSim's own
        rule, and the one AddBiomechanics uses for its segment masses. ``"as_run"``:
        the rule the v2 models were scaled with. There is no default: a missing
        key raises rather than falling back to the library's as-run behaviour.
        The scaling report records the rule used.

    and may set:

    ``allow_generic_mismatch`` (default false)
        nsosim checks that the generic model the gait donor's body was scaled
        from (AddBiomechanics' ``unscaled_generic.osim``) is the skeleton of the
        model being scaled, because the body's scale factors are ratios to that
        generic, and raises if it is not. True lets the scaling proceed anyway --
        legitimate only to reproduce results made on another generic, as the
        v2 measured arm was. nsosim records the override in the
        scaling report (``ab_generic_check``).

    Both options are always passed, so an nsosim too old to honour either fails
    here instead of scaling without it.

    ``recorded_mass_kg`` is the donor's recorded body mass (the config's
    ``addbiomechanics.mass_kg``, what was typed into AddBiomechanics), passed so
    the scaling report can set it beside AddBiomechanics' own total. It is
    recorded only and does not set any mass.
    """
    if config is None:
        raise ConfigError("body scaling options come from a measured-gait config "
                          "(its scaling section); there are no defaults")
    section = config.section("scaling")
    unknown = sorted(k for k in section if k not in _SCALING_KEYS and not k.startswith("_"))
    if unknown:
        raise ConfigError(f"{config.origin}: unknown scaling option(s) {unknown}; "
                          f"expected {sorted(_SCALING_KEYS)}")
    if "mass_rule" not in section:
        raise ConfigError(
            f"{config.origin}: scaling.mass_rule is required ({' or '.join(MASS_RULES)}). "
            "There is deliberately no default: a missing key would quietly reproduce the "
            "published (v2) masses."
        )
    mass_rule = section["mass_rule"]
    if mass_rule not in MASS_RULES:
        raise ConfigError(f"{config.origin}: scaling.mass_rule must be one of {MASS_RULES}, "
                          f"got {mass_rule!r}")
    allow = section.get("allow_generic_mismatch", False)
    if not isinstance(allow, bool):
        raise ConfigError(f"{config.origin}: scaling.allow_generic_mismatch must be true or "
                          f"false, got {allow!r}")
    recorded = config.get("addbiomechanics", "mass_kg")
    if recorded is not None and (isinstance(recorded, bool)
                                 or not isinstance(recorded, (int, float)) or recorded <= 0):
        raise ConfigError(f"{config.origin}: addbiomechanics.mass_kg must be a positive number, "
                          f"got {recorded!r}")
    return {
        "mode": "WA",  # one isotropic factor s_wa for every knee body (see module docstring)
        "preserve_mass_distribution": True,
        "subject_mass": None,  # read from the donor's body model
        "mass_rule": mass_rule,
        "allow_generic_mismatch": allow,
        "recorded_mass_kg": None if recorded is None else float(recorded),
    }


# The keys a condition config's ``scaling`` section may carry, and the mass rules
# nsosim's scale_comak_model accepts.
_SCALING_KEYS = {"mass_rule", "allow_generic_mismatch"}
MASS_RULES = ("volume", "as_run")


def scale_to_gait_body(built_model: Path, body_model: Path, output_model: Path, *,
                       options: dict, force=False):
    """Scale a reconstructed knee model to the gait donor's body.

    Returns the path to the scaling report, which records the scale factor and
    the resulting mass.

    Skips the work if the scaled model is complete *and* was made from the same
    inputs: the built model and its geometry, the donor's body model, these
    options and the same nsosim commit (see ``comak_knee_oa.provenance``). A
    scaled model made from anything else -- another body model, an as-run file
    -- raises instead of being reused; ``force`` rebuilds it.
    """
    from nsosim.scaling import scale_comak_model

    from comak_knee_oa import provenance

    if options is None:
        raise ValueError("scaling options are required: pass scaling_options(config)")
    options = dict(options)
    output_model = Path(output_model)
    unit = output_model.parent
    report = unit / (output_model.stem + "_scaling_report.json")

    if built_model.resolve() == output_model.resolve():
        raise ValueError(f"scaling in place would bake the factor in: {output_model}")

    # Refuse to scale something that has already been scaled. The factor would
    # be applied twice and the model would look plausible at the wrong size.
    # Any of these beside a model marks it as scaled: a scaling report, or a
    # ``.pre_mass_patch.osim`` (the backup a body-mass correction writes beside
    # a scaled model before rewriting its masses).
    siblings = ("*.scaling.json", "*scaling_report*.json", "*.pre_mass_patch.osim")
    if any(any(built_model.parent.glob(pattern)) for pattern in siblings):
        raise ValueError(
            "refusing to scale a model that has already been scaled -- the factor "
            f"would be applied twice: {built_model}"
        )

    inputs = {"built_model": built_model, "body_model": body_model}
    if (built_model.parent / "Geometry").is_dir():
        inputs["built_geometry"] = built_model.parent / "Geometry"
    expected = provenance.fingerprint(
        inputs=inputs,
        options={"function": "nsosim.scaling.scale_comak_model", **options},
        code=("nsosim",),
    )
    decision = provenance.decide(
        unit, expected, complete=_scaled_model_complete(output_model, report),
        force=force, what="scaled model",
    )
    if decision == "skip":
        print(f"already scaled (same inputs): {output_model}")
        return report

    provenance.mark_started(unit, expected)
    returned = scale_comak_model(
        base_osim=built_model,
        ab_scaled_osim=body_model,
        output_osim=output_model,
        output_geometry_dir=unit / "Geometry",
        report_json=report,
        # The built model's marker set, saved before nsosim replaces it with
        # the markers AddBiomechanics placed on the donor's body.
        backup_markerset_xml=unit / "markers_pre_swap_backup.xml",
        **options,
        # Storage only, so kept out of the provenance options: the meshes the
        # scaling leaves unchanged are links to the built model's; the knee
        # meshes it rewrites are real files. The scaled model is identical.
        geometry_staging="link",
    )
    if not output_model.exists():
        raise RuntimeError(
            f"scaling reported {returned} but wrote no model at {output_model}"
        )
    if not _scaled_model_complete(output_model, report):
        raise RuntimeError(f"scaling left an incomplete model at {unit}")
    provenance.mark_complete(unit, expected)
    return Path(returned)


def _scaled_model_complete(model: Path, report: Path) -> bool:
    """A scaled model counts as done only with its geometry and a scale factor."""
    if not (model.exists() and report.exists()):
        return False
    geometry = model.parent / "Geometry"
    if not geometry.is_dir() or not any(geometry.glob("*.stl")):
        return False
    try:
        return json.loads(report.read_text()).get("wa_scale") is not None
    except (json.JSONDecodeError, OSError):
        return False


MESH_SUFFIXES = (".stl", ".vtp", ".obj")
# A bare '&' in a template's citation text: OpenSim reads it, Python's parser does not.
_BARE_AMPERSAND = re.compile(r"&(?!(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);)")


def unresolved_meshes(model: Path) -> list[str]:
    """Mesh files ``model`` names that its own folder cannot supply.

    OpenSim looks for a bare mesh name in the model's folder, then in
    ``<model folder>/Geometry``, and only then in a process-wide search list. A
    queue worker simulates many knees in one process, so a name missing from the
    first two places would be served from another knee's folder. Every
    mesh must therefore resolve from the model's own folder; this lists those
    that do not (links are followed, so a dangling link counts as missing).
    """
    model = Path(model)
    text = _BARE_AMPERSAND.sub("&amp;", model.read_text(encoding="utf-8", errors="replace"))
    names = {element.text.strip() for element in ET.fromstring(text).iter()
             if element.tag.endswith("_file") and element.text
             and element.text.strip().lower().endswith(MESH_SUFFIXES)}
    folder = model.parent
    return sorted(n for n in names
                  if not ((folder / n).is_file() or (folder / "Geometry" / n).is_file()))


def require_meshes_resolve(model: Path) -> None:
    """Raise unless every mesh ``model`` names resolves from its own folder.

    No geometry search path is set, by design (see ``unresolved_meshes``).
    """
    missing = unresolved_meshes(model)
    if missing:
        raise FileNotFoundError(
            f"{model}: {len(missing)} mesh file(s) do not resolve from the model's own "
            f"folder or its Geometry/: {missing[:10]}"
        )


# ----------------------------------------------------------------- stage ----


def simulate(config: Config, knee: Knee, stop_after=None, check_only=False, force=False):
    """Simulate one knee under one gait condition.

    Returns True, or a failure-bucket string from the solver chain. In
    ``check_only`` mode every input is resolved and reported and no solver runs;
    that is the cheap way to find a misconfigured condition before committing
    a few hundred knee-hours to it.
    """
    config = config.for_knee(knee)
    condition = config.get("run", "condition", "reference")
    if condition not in CONDITIONS:
        raise ConfigError(f"run.condition must be one of {CONDITIONS}, got {condition!r}")

    results_dir = config.path("run", "results_dir")
    built_model = config.path("model", "built_model")
    reserve_actuators = config.path("model", "reserve_actuators")

    section = f"{condition}_gait"
    markers = config.path(section, "markers")
    external_loads = config.path(section, "external_loads")

    required = [("model", "built_model"), ("model", "reserve_actuators"),
                (section, "markers"), (section, "external_loads")]
    if condition == "measured":
        required += [(section, "grf"), (section, "body_model")]

    problems = config.check_inputs(*required)
    if problems:
        message = "\n  ".join(problems)
        if check_only:
            print(f"MISSING INPUTS for {knee.name}:\n  {message}")
            return False
        raise FileNotFoundError(f"inputs missing for {knee.name}:\n  {message}")

    # -- timing ----------------------------------------------------------
    if condition == "measured":
        grf = config.path(section, "grf")
        spans = stance_windows(
            grf,
            config.get(section, "grf_vertical_column", "ground_force_calcn_r_vy"),
            config.get(section, "force_threshold_N", 10.0),
            config.get(section, "min_stance_frames", 40),
        )
        timing = timing_from_stance(
            spans,
            config.get(section, "stance_index", 0),
            config.get(section, "ik_pad_s", 0.1),
        )
        print(
            f"{len(spans)} stance span(s); using index "
            f"{config.get(section, 'stance_index', 0)}: "
            f"{timing['start_time_comak']:.3f}-{timing['stop_time_comak']:.3f} s "
            f"({timing['stance_n_frames']} frames, peak "
            f"{timing['stance_peak_force_N']:.0f} N)"
        )
    else:
        timing = {key: config.get("comak_timing", key, value)
                  for key, value in REFERENCE_TIMING.items()}
        print(
            f"reference window: COMAK {timing['start_time_comak']}-"
            f"{timing['stop_time_comak']} s, IK {timing['start_time_ik']}-"
            f"{timing['stop_time_ik']} s"
        )

    # Required, like scaling.mass_rule, so no default can stand in for a choice.
    timing = pad_comak_window(timing, config.require("comak_timing", "comak_window_pad_s"))
    # Resolved before any work, so --check catches a missing or invalid switch.
    scaling = scaling_options(config) if condition == "measured" else None

    if check_only:
        print(f"OK {knee.name}: condition={condition}, all inputs resolve")
        print(f"   model   {built_model}")
        print(f"   markers {markers}")
        print(f"   results {results_dir}")
        print(f"   COMAK window pad {timing.get('comak_window_pad_s', 0.0)} s"
              + (f"; mass rule {scaling['mass_rule']}" if scaling else ""))
        return True

    # -- stage the model -------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = results_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if condition == "measured":
        scaled_model = config.path("model", "scaled_model")
        scale_to_gait_body(
            built_model, config.path(section, "body_model"), scaled_model,
            force=force, options=scaling,
        )
        simulation_model = scaled_model
    else:
        simulation_model = built_model

    # Every model of this simulation -- the staged copies here and the ones the
    # solver writes beside its results -- finds its meshes through a Geometry
    # link in its own folder, never through OpenSim's process-wide search path.
    staged_model = model_dir / "simulation_model.osim"
    shutil.copy(simulation_model, staged_model)
    geometry_link = model_dir / "Geometry"
    if not geometry_link.exists():
        # Relative, so the tree can move or ship as a whole.
        geometry_link.symlink_to(os.path.relpath(simulation_model.parent / "Geometry",
                                                 os.path.realpath(model_dir)))
    require_meshes_resolve(staged_model)

    forceset = model_dir / "reserve_actuators.xml"
    shutil.copy(reserve_actuators, forceset)

    # The patella optimiser mutates the model it is handed; hand it a copy.
    model_to_optimize = model_dir / "simulation_model_forsim.osim"
    shutil.copy(staged_model, model_to_optimize)

    # The reference trial's external-loads file already points at its own data.
    # A measured trial's does not, so it is rewritten on the way in.
    if condition == "measured":
        staged_loads = absolutise_external_loads(
            external_loads, config.path(section, "grf"), model_dir / "external_loads.xml"
        )
    else:
        staged_loads = external_loads

    # -- ligament reference strains --------------------------------------
    ligament_params = config.get("model", "ligament_parameters")
    strains = reference_strains(ligament_params) if ligament_params else reference_strains()

    # -- provenance ------------------------------------------------------
    record = _write_run_record(results_dir, config, knee, condition, timing)

    started = time.time()
    try:
        result = run_comak_chain(
            model_to_optimize=model_to_optimize,
            results_dir=results_dir,
            markerset_file=markers,
            external_loads_file=staged_loads,
            forceset_file=forceset,
            timing=timing,
            config=config,
            muscle_weights=build_muscle_weights(config),
            contact_energy_weight=config.get("comak_optimizer", "contact_energy_weight", 0),
            forsim_max_duration=config.get("timeouts", "forsim_seconds", 450),
            forsim_max_updates=config.get("timeouts", "forsim_max_updates", 1),
            reference_strain_update=strains,
            stop_after=stop_after,
        )
    except BaseException as exc:
        _finish_run_record(record, f"exception: {type(exc).__name__}: {exc}", started)
        raise
    _finish_run_record(record, result, started)
    return result


def _write_run_record(results_dir: Path, config: Config, knee: Knee, condition, timing):
    """Record what produced this result, next to the result.

    Code versions, and where it ran: COMAK is deterministic per CPU type but
    not across types, so host and CPU model decide which results should be
    bit-identical.
    """
    from comak_knee_oa.manifest import _code_versions, environment_record

    record = {
        "knee": knee.name,
        "condition": condition,
        "config_file": str(config.origin) if config.origin else None,
        "config": config.data,
        "timing": timing,
        "code": _code_versions(),
        "environment": environment_record(),
    }
    path = results_dir / "run_record.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return path


def _finish_run_record(path: Path, result, started: float) -> None:
    """Add how the chain ended -- True, a failure bucket, or an exception."""
    from datetime import datetime, timezone

    record = json.loads(path.read_text())
    record["result"] = result if isinstance(result, (bool, str)) else repr(result)
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["elapsed_s"] = round(time.time() - started, 1)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------ output verification ----

# Artefacts that prove each solver stage actually produced something. A queue
# entry marked done is not evidence: a stage can raise, be caught, and leave the
# entry marked done with nothing behind it. Completion is therefore defined as
# these files existing, being non-empty, and -- for the sweep -- holding more
# than one state: a sweep that wrote only its header row did not sweep.
REQUIRED_OUTPUTS = {
    "sweep": "comak-inverse-kinematics/_secondary_constraint_sweep_states.sto",
    "forward simulation": "forsim_results/joint_mechanics/joint_mechanics.h5",
    "COMAK": "comak/_force.sto",
    "inverse dynamics": "comak-inverse-dynamics/inverse-dynamics.sto",
    "joint mechanics": "joint-mechanics/joint_mechanics.h5",
}

# Storage files whose data must have more than one row.
_MULTI_ROW = ("sweep",)


def storage_rows(path: Path) -> int:
    """Number of data rows in an OpenSim storage file (rows after the header)."""
    rows, in_data = 0, False
    with open(path, errors="replace") as fh:
        for line in fh:
            if not in_data:
                if line.strip().lower().startswith("time"):
                    in_data = True
                continue
            if line.strip():
                rows += 1
    return rows


def verify_outputs(results_dir: Path) -> list[str]:
    """Which required outputs are absent or empty. Empty list: the knee is complete.

    Checks files rather than the solver chain's return value, because a stage
    can fail inside the chain without raising, and the chain then reports
    success with that stage's output missing. A non-empty
    ``inputs/error_log.txt`` is not a failure here: the inverse-dynamics tool
    logs a non-fatal error after writing its output, and treating that as a
    failure discarded complete knees.
    """
    problems = []
    for label, relative in REQUIRED_OUTPUTS.items():
        path = Path(results_dir) / relative
        if not path.is_file():
            problems.append(f"{label}: {relative}")
        elif path.stat().st_size == 0:
            problems.append(f"{label}: {relative} is empty")
        elif label in _MULTI_ROW and storage_rows(path) <= 1:
            problems.append(f"{label}: {relative} holds {storage_rows(path)} row(s)")
    return problems
