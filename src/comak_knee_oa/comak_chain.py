"""The COMAK solver chain. One definition, used by every simulated condition.

    patella optimisation (forward-simulation settle)
      -> COMAK inverse kinematics (settle + secondary-constraint sweep)
      -> COMAK
      -> COMAK inverse dynamics
      -> joint mechanics

The paper's central comparison is between gait conditions on the same knees, so
the chain exists once and cannot drift between them. Everything that
differs between conditions -- which model, which motion and loads, which time
window -- is prepared by the caller and handed in as plain arguments. This
function runs solvers and nothing else.

Failures are values, not exceptions. Each solver stage has a wall-clock budget,
and exceeding it returns a named bucket rather than raising, so a batch over
hundreds of knees records *how* a knee failed instead of losing it in a
traceback. Anything unexpected still propagates.

A note on exact reproduction: the underlying solver is sensitive to CPU
microarchitecture. The same input on a different node moves knee rotations by
about a degree and contact pressures by a few percent. That is floating-point
rounding, it is deterministic per node, and it is documented in
``docs/pinned_versions.md``. Compare like with like.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from pycomak import COMAK, COMAKInverseDynamics, COMAKInverseKinematics, JointMechanics
from pycomak.defaults import (
    forsim_patella_optimization_kinematics,
    forsim_patella_optimization_muscle_activations,
    patella_optimization_criteria,
)
from pycomak.knee_optimizer import KneeOptimizer
from pycomak.utils import copy_file_names_with_strings, run_with_timeout

from .paths import stage_dir

__all__ = [
    "STAGES",
    "DEFAULT_HAMSTRING_MUSCLES",
    "run_comak_chain",
    "build_muscle_weights",
    "reference_strains",
]

# Solver stages in execution order. ``stop_after`` takes one of these.
STAGES = ["patella-opt", "comak-ik", "comak", "comak-id", "joint-mechanics"]

# A fixed settling nudge applied to the patella before optimisation. Not scaled
# with body size: it is a numerical starting condition, not an anatomical one.
PATELLA_POSITION_UPDATE = np.asarray([0.0, -0.004, 0.0])

# Hamstrings are de-emphasised in the COMAK cost so the optimiser does not lean
# on them to satisfy the knee moment.
DEFAULT_HAMSTRING_MUSCLES = ["bflh_r", "bfsh_r", "semiten_r", "semimem_r"]
DEFAULT_HAMSTRING_WEIGHT = 2

# How COMAK IK gives the models it writes access to their meshes: "link" writes one
# relative directory symlink to the model's Geometry folder instead of copying it
# (~50 MB per copy, two copies per simulation). Storage only: results are identical.
GEOMETRY_STAGING = "link"

DEFAULT_LIGAMENT_PARAMS = (
    "ligament_strain_stiffness/"
    "lenhart_defaults_plus_smith_menisci_lig_params_Oct.15.2025.json"
)


@functools.lru_cache(maxsize=8)
def reference_strains(relative_path: str = DEFAULT_LIGAMENT_PARAMS) -> dict:
    """Ligament reference strains, keyed by ligament name.

    Read on demand and cached, so importing this module cannot fail on a data
    root that has not been downloaded yet.
    """
    path = stage_dir("inputs", "reference") / relative_path
    params = json.loads(Path(path).read_text())
    return {name: p["strain"] for name, p in params.items() if "strain" in p}


def build_muscle_weights(config) -> dict | None:
    """COMAK muscle weights: the library defaults, with hamstrings re-weighted.

    Returns None when weighting is disabled, which COMAK reads as uniform.
    Copies the library default rather than updating it in place -- mutating an
    imported default is invisible from the call site and outlives the call.
    """
    if not config.get("muscle_weights", "use_muscle_weights", True):
        return None
    from pycomak.defaults import muscle_weights_dict as library_defaults

    weights = dict(library_defaults)
    weight = config.get("muscle_weights", "hamstring_weight", DEFAULT_HAMSTRING_WEIGHT)
    muscles = config.get("muscle_weights", "hamstring_muscles", DEFAULT_HAMSTRING_MUSCLES)
    weights.update({m: weight for m in muscles})
    return weights


def run_comak_chain(
    *,
    model_to_optimize,
    results_dir,
    markerset_file,
    external_loads_file,
    forceset_file,
    timing,
    config,
    muscle_weights=None,
    contact_energy_weight=0,
    forsim_max_duration=450,
    forsim_max_updates=1,
    reference_strain_update=None,
    stop_after=None,
):
    """Run the chain on an already-prepared model, motion and time window.

    Parameters
    ----------
    model_to_optimize : path
        The model the patella optimiser mutates. Must already be a disposable
        copy; this function will not protect the caller's original.
    results_dir : path
        Where every stage writes.
    markerset_file, external_loads_file, forceset_file : path
        Marker trajectories to track, resolved ground reaction forces, and the
        reserve actuator set.
    timing : dict
        ``start_time_ik``, ``stop_time_ik``, ``start_time_comak``,
        ``stop_time_comak``. How the window was chosen is the caller's business:
        a fixed window for the reference gait, stance detection for measured gait.
    config : Config
        Solver settings -- settle repetitions, sweep times, time budgets,
        optimiser tolerances. Every value has a default here.
    stop_after : str | None
        One of :data:`STAGES`. For working through a knee stage by stage.

    Returns
    -------
    True
        The chain completed.
    str
        A failure bucket (``settle_sim_timeout_ik``, ``sweep_sim_timeout``,
        ``comaktool_timeout``, or a patella-optimiser failure), or an
        ``ok_stopped_after_*`` marker when ``stop_after`` was given.
    """
    results_dir = str(results_dir)
    settle_reps = config.get("model_settling", "settle_sim_reps", 2)
    patella_update = np.asarray(
        config.get(
            "model_settling", "patella_position_update_m", list(PATELLA_POSITION_UPDATE)
        )
    )

    # -- 1. patella optimisation ----------------------------------------
    if stop_after and STAGES.index("patella-opt") > STAGES.index(stop_after):
        return "stopped_before_patella_opt"

    _banner("Patella optimisation")

    # settle_reps + 1: inverse kinematics performs settle_reps - 1 settling
    # simulations and then one more at the start of the sweep. The optimiser
    # never runs the sweep, so it needs an extra repetition to arrive at the
    # same settled state that inverse kinematics will start from.
    optimizer = KneeOptimizer(
        path_model_to_update=str(model_to_optimize),
        results_dir=str(Path(results_dir) / "forsim_results"),
        markerset_file=str(markerset_file),
        dict_kinematics=forsim_patella_optimization_kinematics,
        dict_muscles=forsim_patella_optimization_muscle_activations,
        dict_criteria=patella_optimization_criteria,
        settle_sim_reps=settle_reps + 1,
        patella_position_update=patella_update,
        max_duration=forsim_max_duration,
        max_updates=forsim_max_updates,
        dict_reference_strain_update=reference_strain_update,
        geometry_staging=GEOMETRY_STAGING,
    )
    result = optimizer.optimize_patella_location()
    if isinstance(result, str):
        print(f"Patella optimisation failed: {result}")
        return result
    if stop_after == "patella-opt":
        return "ok_stopped_after_patella_opt"

    # -- 2. COMAK inverse kinematics ------------------------------------
    _banner("COMAK inverse kinematics")

    # The sweep traces the secondary (passive) knee coordinates. A short sweep
    # is much faster and succeeds on most knees; a longer one gives the
    # integrator more simulated time to trace a smooth trajectory and rescues
    # the ones that chatter. Try the fast path first and pay the long wall only
    # on the tail. Under tests, the short and long produce nearly identical
    # results on the same knee, so the fallback is not a compromise.
    sweep_primary = config.get("comak_timing", "secondary_constraint_sim_sweep_time", 0.6)
    sweep_fallback = config.get(
        "comak_timing", "secondary_constraint_sim_sweep_time_fallback", 1.2
    )
    sweep_stop = config.get(
        "comak_timing", "secondary_coupled_coordinate_stop_value", 120
    )
    settle_budget = config.get("timeouts", "settle_sim_seconds", 300)
    sweep_budget = config.get("timeouts", "sweep_sim_seconds", 7200)
    comak_budget = config.get("timeouts", "comak_tool_seconds", 10800)

    def build_ik(sweep_time):
        return COMAKInverseKinematics(
            base_model_path=optimizer.intermediate_model_path,
            results_dir=results_dir,
            start_time_ik=timing["start_time_ik"],
            stop_time_ik=timing["stop_time_ik"],
            markerset_file=str(markerset_file),
            settle_sim_reps=settle_reps,
            secondary_constraint_sim_sweep_time=sweep_time,
            secondary_coupled_coordinate_stop_value=sweep_stop,
            geometry_staging=GEOMETRY_STAGING,
        )

    inverse_kinematics = build_ik(sweep_primary)
    try:
        run_with_timeout(inverse_kinematics.perform_settle_sim, settle_budget)
    except TimeoutError:
        return "settle_sim_timeout_ik"

    try:
        run_with_timeout(inverse_kinematics.perform_sweep_sim, sweep_budget)
    except TimeoutError:
        _banner(
            f"Sweep timed out at {sweep_primary}s -- rebuilding at {sweep_fallback}s"
        )
        # Rebuilt from scratch so the longer sweep time takes effect and the
        # coupled-coordinate function is regenerated. The forward simulation is
        # not repeated; only settle and sweep.
        inverse_kinematics = build_ik(sweep_fallback)
        try:
            run_with_timeout(inverse_kinematics.perform_settle_sim, settle_budget)
        except TimeoutError:
            return "settle_sim_timeout_ik"
        try:
            run_with_timeout(inverse_kinematics.perform_sweep_sim, sweep_budget)
        except TimeoutError:
            return "sweep_sim_timeout"

    inverse_kinematics.perform_inverse_kinematics()
    if stop_after == "comak-ik":
        return "ok_stopped_after_comak_ik"

    # -- 3. COMAK --------------------------------------------------------
    _banner("COMAK")

    # Iteration limit: at the library default of 25, most timesteps hit the
    # limit before converging. The unconverged steps show up as a false spike
    # in muscle force at the end of stance. A higher limit removes the spike
    # but takes about twice as long to run.
    #
    # Convergence test: "udot" checks joint accelerations; "generalized_force"
    # checks the residual scaled by each joint's inertia. The menisci are very
    # light, so small forces give them large accelerations. Under "udot" they
    # decide whether a step converges; "generalized_force" stops that.
    comak = COMAK(
        results_dir=results_dir,
        forceset_file=str(forceset_file),
        model_path=inverse_kinematics.final_model_path,
        external_loads_file=str(external_loads_file),
        start_time=timing["start_time_comak"],
        stop_time=timing["stop_time_comak"],
        contact_energy_weight=contact_energy_weight,
        muscle_weights_dict=muscle_weights,
        max_iterations=config.get("comak_optimizer", "max_iterations", 100),
        ipopt_max_iterations=config.get("comak_optimizer", "ipopt_max_iterations", 500),
        convergence_criterion=config.get(
            "comak_optimizer", "convergence_criterion", "udot"
        ),
        generalized_force_tolerance=config.get(
            "comak_optimizer", "generalized_force_tolerance", 1.0
        ),
        generalized_force_worse_case_tolerance=config.get(
            "comak_optimizer", "generalized_force_worse_case_tolerance", 50.0
        ),
    )
    try:
        run_with_timeout(comak.run, comak_budget)
    except TimeoutError:
        return "comaktool_timeout"
    if stop_after == "comak":
        return "ok_stopped_after_comak"

    # -- 4. COMAK inverse dynamics ---------------------------------------
    _banner("COMAK inverse dynamics")
    COMAKInverseDynamics(
        results_dir,
        start_time=timing["start_time_comak"],
        stop_time=timing["stop_time_comak"],
        model_path=inverse_kinematics.final_model_path,
        external_loads_file=str(external_loads_file),
    ).run()
    if stop_after == "comak-id":
        return "ok_stopped_after_comak_id"

    # -- 5. joint mechanics ----------------------------------------------
    _banner("Joint mechanics")
    joint_mechanics = JointMechanics(
        results_dir=results_dir,
        model_path=inverse_kinematics.final_model_path,
        start_time=timing["start_time_comak"],
        end_time=timing["stop_time_comak"],
    )
    joint_mechanics.run()

    # Surface the cartilage contact results at the top of the joint-mechanics
    # directory, where the analysis stages look for them.
    copy_file_names_with_strings(
        [
            "_contact_femur_cartilage*.vtp",
            "_contact_patella_cartilage*.vtp",
            "_contact_tibia_cartilage*.vtp",
            "_mesh_tibia_bone_dynamic*.vtp",
        ],
        joint_mechanics.jnt_mech_result_dir,
    )
    return True


def _banner(text: str) -> None:
    print("=" * 72)
    print(text)
    print("=" * 72, flush=True)
