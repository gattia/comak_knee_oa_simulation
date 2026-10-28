"""The ten gait conditions, and where each one's derived data lives.

One knee is simulated under ten gait conditions: a reference overground gait and
nine measured from a knee-osteoarthritis gait-retraining trial. Every analysis
that spans conditions needs the same three things -- the list, the directory a
condition's derived pressure lands in, and the key its quality-control rows are
filed under -- so they live here once rather than in each figure.

Directory name and quality-control key
--------------------------------------
A condition's directory is named for the gait donor (``Subject_102``), or
``default`` for the reference. Its quality-control key is whatever the pipeline
run that produced the table called it (``multigait_Subject_102_bts_v2``).
Neither can be built from the other by formatting. A formatted shortcut such as
``f"pressure_on_reference_{cohort_tag}"`` resolves to a directory that does not
exist, and consumers that loop per knee and skip missing files would finish
cleanly over fewer conditions. So ``pressure_dir`` raises on a missing tree and
``check_pressure_trees`` asserts a non-zero knee count per condition.

On a data root that declares a run (``paths.run_name``), a condition's key is
``<gait>_<run>`` (``Subject_102_v3``) and that is its only spelling: a table
carrying another run's keys is another run's table, and resolving to it would
apply that run's exclusions to this run's simulations. On the v2 root, which
declares no run, ``ALIASES`` lists every spelling its table uses and
``resolve_qc_tags`` picks the one on disk. A key the table has never seen
raises rather than returning an empty exclusion set, which would read as "this
condition is clean".

Which nine donors: a property of the data root
-----------------------------------------------
The v2 run's ninth donor is ``RSubject_121``; the donor-selection design chose
``Subject_121``, which a new run uses. Both runs stay readable, so the list is
not a constant of the code. Like the run name, it belongs to the data root:

* a root that declares no run is the v2 root: ``MEASURED_V2``;
* a declared root lists its own in ``_run.json`` ``measured_gaits``
  (``new_root create`` records it: the baseline's donors when the root links
  the baseline's gait inputs, else ``MEASURED_DEFAULT``);
* a declared root without the key, or a machine with no data root, gets
  ``MEASURED_DEFAULT``.

``MEASURED`` and ``ALL`` are therefore computed when read, like ``ALIASES``.
A condition outside the active root's list is refused everywhere here, so one
root's donors cannot be mixed into another root's tables.

Reading the v2 root
-------------------
The v2 root declares no run, so the configs' ``{run}`` cannot be filled there:
its folder names follow no single pattern. On that root ``condition_config``
returns the config with the two ``{run}`` keys (``run.results_dir``,
``model.built_model``) set to the v2 locations (``V2_LAYOUT``) and the
switches whose shipped values differ from v2 set to how v2 ran
(``V2_AS_RUN``). The derivation, quality-control and audit stages can then read
v2, to reproduce it or compare a new run with it, without mistaking v3's
settings for v2's. This is for reading only: the pipeline's entry points load a
config file directly, where ``{run}`` refuses an undeclared root.
"""

from __future__ import annotations

from pathlib import Path

import copy
import re

from comak_knee_oa.paths import REPO_ROOT, declaration, run_name, stage_dir

__all__ = [
    "REFERENCE",
    "MEASURED",
    "ALL",
    "MEASURED_V2",
    "MEASURED_DEFAULT",
    "V2_MODELS_RUN",
    "V2_LAYOUT",
    "V2_AS_RUN",
    "measured",
    "all_conditions",
    "check_measured_gaits",
    "QC_TAG",
    "ALIASES",
    "qc_tags",
    "resolve_qc_tags",
    "condition_config_path",
    "condition_config",
    "results_dir",
    "pressure_dir",
    "check_pressure_trees",
]

# The reference (overground) condition. Held separate from the nine because
# several analyses exclude it: it differs in protocol, speed and
# population. See docs/data_provenance.md.
REFERENCE = "default"

# The nine measured donors of the v2 run.
MEASURED_V2 = (
    "RSubject_121",
    "Subject_102",
    "Subject_107",
    "Subject_109",
    "Subject_114",
    "Subject_119",
    "Subject_140",
    "Subject_145",
    "Subject_167",
)

# The nine a new run simulates unless its root declares otherwise: Subject_121,
# the participant the selection design chose, in place of RSubject_121.
MEASURED_DEFAULT = (
    "Subject_102",
    "Subject_107",
    "Subject_109",
    "Subject_114",
    "Subject_119",
    "Subject_121",
    "Subject_140",
    "Subject_145",
    "Subject_167",
)

# Where the v2 root keeps each condition's output: the v2 values of the config
# keys that carry {run}.
V2_MODELS_RUN = "OARSI_May.22.2026_Full_run"
V2_LAYOUT = {
    "built_model": (
        "01_models/OARSI_May.22.2026_Full_run/{knee}/custom_nsm_full_body_healthy_knee_model/"
        "full_body_healthy_knee_nsm_{knee}_May.22.2026_Full_run/"
        "full_body_healthy_knee_nsm_{knee}_May.22.2026_Full_run.osim"
    ),
    "results_dir": {
        REFERENCE: "02_simulations/reference/{knee}/comak",
        **{gait: f"02_simulations/{gait}/{{knee}}/comak_pathway_b_multigait_{gait}_bts_v2"
           for gait in MEASURED_V2},
    },
}

# How v2 ran, for the switches whose shipped values differ: the COMAK window
# unpadded, and for the measured arm the v2 per-body masses on bodies
# AddBiomechanics scaled from ArmlessRajagopal.
V2_AS_RUN = {
    "all": {("comak_timing", "comak_window_pad_s"): 0.0},
    "measured": {("scaling", "mass_rule"): "as_run",
                 ("scaling", "allow_generic_mismatch"): True},
}

# A donor name becomes a directory name and part of a quality-control key.
_DONOR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

# Every spelling of a condition's quality-control key that the v2 table uses,
# most-preferred first. Used only on a root that declares no run.
LEGACY_ALIASES = {
    "default": ("default_May.22.2026_Full_run", "reference"),
    **{gait: (f"multigait_{gait}_bts_v2", gait) for gait in MEASURED_V2},
}


def _active_declaration() -> dict | None | bool:
    """The active root's declaration; None if it declares none; False if no root."""
    try:
        return declaration()
    except FileNotFoundError:  # no data root on this machine
        return False


def _declared_run() -> str | None:
    """The active root's run, or None for a root that declares none."""
    decl = _active_declaration()
    return run_name() if decl and decl.get("run_name") else None


def check_measured_gaits(gaits, where: str = "measured_gaits") -> tuple[str, ...]:
    """Validate a list of measured donors, returning it as a tuple."""
    if isinstance(gaits, str) or not gaits:
        raise ValueError(f"{where}: expected a non-empty list of donor names, got {gaits!r}")
    gaits = tuple(gaits)
    bad = [g for g in gaits if not isinstance(g, str) or not _DONOR.fullmatch(g) or g == REFERENCE]
    if bad:
        raise ValueError(f"{where}: invalid donor name(s) {bad}")
    if len(set(gaits)) != len(gaits):
        raise ValueError(f"{where}: duplicate donor names in {list(gaits)}")
    return gaits


def measured() -> tuple[str, ...]:
    """The measured conditions of the active data root (see the module docstring)."""
    decl = _active_declaration()
    if decl is None:
        return MEASURED_V2
    if decl and "measured_gaits" in decl:
        return check_measured_gaits(decl["measured_gaits"], where="_run.json measured_gaits")
    return MEASURED_DEFAULT


def all_conditions() -> tuple[str, ...]:
    """The reference, then the active root's measured conditions."""
    return (REFERENCE,) + measured()


def _require_condition(gait: str) -> None:
    conditions = all_conditions()
    if gait not in conditions:
        raise KeyError(
            f"unknown gait condition {gait!r} for this data root; expected one of "
            f"{list(conditions)}. The measured donors are a property of the root "
            "(_run.json measured_gaits; a root that declares no run is the v2 root)."
        )


def aliases() -> dict[str, tuple[str, ...]]:
    """Every acceptable quality-control key per condition, for the active root."""
    run = _declared_run()
    if run is None:
        return dict(LEGACY_ALIASES)
    return {gait: (f"{gait}_{run}",) for gait in all_conditions()}


def qc_tag_map() -> dict[str, str]:
    """``{gait: key}`` -- the key to use when nothing says what the table holds."""
    return {gait: names[0] for gait, names in aliases().items()}


def __getattr__(name: str):
    # These depend on the data root -- its run and its donors -- which is not
    # known at import time, so they are computed when read.
    if name == "ALIASES":
        return aliases()
    if name == "QC_TAG":
        return qc_tag_map()
    if name == "MEASURED":
        return measured()
    if name == "ALL":
        return all_conditions()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def qc_tags(gaits=None) -> list[str]:
    """Quality-control keys for ``gaits``, in the given order.

    Prefer ``resolve_qc_tags`` where the table's own key set is available: this
    returns the default spelling, which is right only if the table on disk uses
    it.
    """
    tags = qc_tag_map()
    return [tags[gait] for gait in (all_conditions() if gaits is None else gaits)]


def resolve_qc_tags(known_tags, gaits=None) -> list[str]:
    """Quality-control keys for ``gaits``, in whichever spelling ``known_tags`` uses.

    ``known_tags`` is the table's own key set (``qc.known_cohort_tags()``). Pass
    it rather than assuming a spelling, so that renaming the table's keys does
    not break every figure at once.
    """
    known = set(known_tags)
    names = aliases()
    resolved, missing = [], []
    for gait in all_conditions() if gaits is None else gaits:
        match = next((name for name in names[gait] if name in known), None)
        if match is None:
            missing.append(gait)
        else:
            resolved.append(match)
    if missing:
        raise KeyError(
            f"quality-control table has no key for condition(s) {missing}. "
            f"Tried {[names[g] for g in missing]}; the table knows "
            f"{sorted(known)}. Either the table was not built for these "
            "conditions (or for this run), or it uses a spelling this module "
            "has not been told about -- add it to LEGACY_ALIASES rather than "
            "renaming the table."
        )
    return resolved


# ------------------------------------------------------- run configs ----


def condition_config_path(gait: str) -> Path:
    """The run config that defines condition ``gait`` on the active root.

    Raises KeyError for a condition outside the root's list, and
    FileNotFoundError for one inside it whose config this checkout lacks.
    """
    _require_condition(gait)
    if gait == REFERENCE:
        return REPO_ROOT / "config" / "reference_gait.json"
    path = REPO_ROOT / "config" / "measured_gait" / f"{gait}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"condition {gait!r} is one of this data root's measured donors, but this "
            f"checkout has no {path.relative_to(REPO_ROOT)}. Each donor needs its "
            "measured-gait config before it can be run or read."
        )
    return path


def condition_config(gait: str):
    """The loaded run config for condition ``gait`` (``{run}`` filled on read).

    On the v2 root, which declares no run, the config as it describes that
    root (``V2_LAYOUT``; see the module docstring).
    """
    from comak_knee_oa.config import Config, load_config

    config = load_config(condition_config_path(gait))
    if _active_declaration() is not None:
        return config
    data = copy.deepcopy(config.data)
    data["run"]["results_dir"] = V2_LAYOUT["results_dir"][gait]
    data["model"]["built_model"] = V2_LAYOUT["built_model"]
    switches = dict(V2_AS_RUN["all"])
    if gait != REFERENCE:
        switches.update(V2_AS_RUN["measured"])
    for (section, key), value in switches.items():
        data.setdefault(section, {})[key] = value
    leftover = sorted(f"{section}.{key}" for section, values in data.items()
                      if isinstance(values, dict)
                      for key, value in values.items() if "{run}" in str(value))
    if leftover:
        raise RuntimeError(
            f"{config.origin} names its run in {leftover}, which the published root's "
            "layout (cohorts.V2_LAYOUT) does not cover; add their v2 values there"
        )
    return Config(data, origin=config.origin)


def results_dir(gait: str, knee) -> Path:
    """Where condition ``gait`` put ``knee``'s simulation results.

    Read through the config that ran the simulation, so the location is defined
    once. ``knee`` is a ``Knee`` or a name such as ``9003316_RIGHT``.
    """
    from comak_knee_oa.config import Knee

    if isinstance(knee, str):
        knee = Knee.parse(knee)
    return condition_config(gait).for_knee(knee).path("run", "results_dir")


def pressure_dir(gait: str, bone: str) -> Path:
    """``03_derived/pressure_on_reference_<gait>/<bone>/``.

    Raises rather than returning a path that would yield zero files.
    """
    if gait not in all_conditions():
        raise KeyError(
            f"unknown gait condition {gait!r}; expected one of {list(all_conditions())}. "
            "Pass the gait donor name, not the quality-control key."
        )
    path = stage_dir("derived", f"pressure_on_reference_{gait}", bone)
    if not path.is_dir():
        raise FileNotFoundError(
            f"No transferred pressure for gait {gait!r}, bone {bone!r}: {path}\n\n"
            "Build it with:\n"
            f"    python pipeline/derive_pressure.py transfer --config "
            f"config/measured_gait/{gait}.json\n\n"
            "Or download the published 03_derived stage."
        )
    return path


def check_pressure_trees(bone: str, gaits=None) -> dict[str, int]:
    """``{gait: n knees}``, raising if any condition resolved to an empty tree.

    Call this before a sweep, not after, so a per-condition loop cannot report
    success over zero knees.
    """
    counts = {}
    for gait in all_conditions() if gaits is None else gaits:
        directory = pressure_dir(gait, bone)
        n = len(list(directory.glob(f"*_{bone}_pressure_ref.vtk")))
        if n == 0:
            raise RuntimeError(
                f"gait {gait!r} resolved to {directory} but holds zero {bone} "
                "pressure files -- refusing to run over an empty condition."
            )
        counts[gait] = n
    return counts
