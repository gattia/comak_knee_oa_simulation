"""Copy every external input into the data root, and record where it came from.

This is the first stage, and the one that makes the rest of the pipeline
self-contained. Inputs are copied into the data root and hashed, so nothing
upstream can change underneath a derived result -- there is no upstream left
to move. ``00_inputs`` and ``01_models`` as this stage writes them are the
dataset a reader downloads.

Where the files come from lives in ``vendor_sources.json`` next to this file,
not in this module. Those paths exist on one machine and describe a filesystem;
the input *set* -- which knees, which visits, which meshes -- is part of the
method, and lives here. The committed file carries placeholders for the six
source roots; the real ones go in the gitignored ``vendor_sources.local.json``
beside it. See ``_load_spec``.

Running it
----------
Reporting is the default. Nothing is copied without ``--execute``, because the
job is roughly 100 GB and a mistaken invocation is expensive in both directions::

    python pipeline/vendor_inputs.py                     # what would be copied
    python pipeline/vendor_inputs.py --group reference   # one group
    python pipeline/vendor_inputs.py --execute           # do it

It is resumable and idempotent. Each copied file is journalled with its hash as
it lands, so an interrupted run re-reads the journal rather than the data, and a
second full run copies nothing. A run after a source file changes does re-copy
it, so the manifest records the changed input rather than hiding it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comak_knee_oa.manifest import (  # noqa: E402
    FileRecord,
    write_manifest,
)
from comak_knee_oa.paths import STAGES, data_root, stage_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SOURCES_FILE = Path(__file__).resolve().parent / "vendor_sources.json"
LOCAL_SOURCES_FILE = Path(__file__).resolve().parent / "vendor_sources.local.json"
JOURNAL_NAME = "_vendor_journal.jsonl"
ADOPT_JOURNAL_NAME = "_stage_hash_journal.jsonl"
COHORT_NAME = "cohort.json"

# Working space inside a stage: real directories that are not published data.
# Excluded from the stage manifest, because a manifest that recorded one user's
# scratch reproduction runs would fail verification on every other machine.
STAGE_IGNORE = {"simulations": ("_reproduction", "_synthetic_rerun")}

_CHUNK = 1 << 22  # 4 MiB -- these are mesh files on a network filesystem


# --------------------------------------------------------------- planning ----


@dataclass(frozen=True)
class Item:
    """One file to vendor: an absolute source, and where it lands in its stage."""

    source: Path
    rel: str  # relative to the stage directory


@dataclass
class Plan:
    name: str
    stage: str
    items: list[Item]
    missing: list[str]  # expected but absent at the source -- a defect
    notes: list[str] = None  # expected-to-vary coverage, reported but not a defect

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    @property
    def size(self) -> int:
        return sum(i.source.stat().st_size for i in self.items)


@dataclass
class Context:
    spec: dict
    roots: dict[str, Path]
    cohort: list[str]  # "{subject}_{SIDE}"

    def knee_dir(self, run: str, knee: str) -> Path:
        subject, side = knee.rsplit("_", 1)
        timepoint = self.spec["cohort"]["timepoint"]
        return self.roots["step1_results"] / run / f"{subject}_{timepoint}_{side}"


def _load_spec() -> dict:
    """The input set, with this machine's source roots merged over the placeholders.

    ``vendor_sources.json`` is committed and ships ``<PLACEHOLDER>`` roots. The
    input *set* is part of the method and belongs in the repository; the six
    directories it is copied from are one institution's filesystem layout and
    are not this repository's to publish. They live beside it in the gitignored ``vendor_sources.local.json``,
    which carries the ``roots`` key and nothing else::

        {"roots": {"oai_meshes": "<absolute path to the OAI mesh tree>", ...}}

    Only this stage reads roots at all, and only when assembling the dataset
    from its original sources -- which needs access to those
    sources anyway. A reader who downloads ``00_inputs`` and ``01_models``
    never resolves one.
    """
    spec = json.loads(SOURCES_FILE.read_text())
    if LOCAL_SOURCES_FILE.exists():
        local = json.loads(LOCAL_SOURCES_FILE.read_text())
        spec["roots"] = {**spec["roots"], **local.get("roots", {})}
    return spec


def _roots(spec: dict) -> dict[str, Path]:
    # Fail on an unresolved placeholder here rather than letting it reach the
    # is_dir() check in main(), which would report it as a missing directory --
    # true, but it sends you looking for the data instead of for the config.
    unset = sorted(k for k, v in spec["roots"].items() if str(v).startswith("<"))
    if unset:
        raise SystemExit(
            f"source root(s) still at their placeholder: {', '.join(unset)}\n"
            f"Set them in {LOCAL_SOURCES_FILE.name} (gitignored), beside "
            f"{SOURCES_FILE.name}:\n"
            f'  {{"roots": {{"{unset[0]}": "<absolute path>"}}}}\n'
            "Only re-assembling the dataset from its original sources needs "
            "these; reading the published dataset does not."
        )
    return {k: Path(v) for k, v in spec["roots"].items()}


def resolve_cohort(spec: dict, roots: dict[str, Path]) -> list[str]:
    """The knees to vendor: those with a completed reconstruction, less exclusions.

    Derived from what is on disk rather than from a checked-in list, because a
    list would be one more thing that can drift out of agreement with the data.
    The resolved result is then written down, so everything downstream reads a
    fixed list and this derivation happens once.
    """
    conf = spec["cohort"]
    run_dir = roots["step1_results"] / conf["step1_run"]
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Step-1 run directory not found: {run_dir}")

    marker = spec["model_tree"]["include"][0]  # the built OpenSim model directory
    suffix = f"_{conf['timepoint']}_"
    excluded = set(conf["exclude"])

    knees = []
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir() or suffix not in entry.name:
            continue
        subject, _, side = entry.name.partition(suffix)
        knee = f"{subject}_{side}"
        if knee in excluded:
            continue
        if (entry / marker).is_dir():
            knees.append(knee)
    return knees


def _tree(root: Path, dest_prefix: str) -> list[Item]:
    """Every file under ``root``, mirrored beneath ``dest_prefix``.

    Walks through directory symlinks and materialises their contents as real
    files: the dataset must stand on its own, and a symlink neither survives an
    upload nor guarantees its target comes along. ``Path.rglob`` does not
    descend into symlinked directories (a linked ``Geometry/``, for one), so
    this walk follows them itself.
    """
    if not root.is_dir():
        return []
    items = []
    for parent, _dirs, files in os.walk(root, followlinks=True):
        for name in files:
            path = Path(parent) / name
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                items.append(Item(path, f"{dest_prefix}/{relative}"))
    return sorted(items, key=lambda item: item.rel)


def _expect(source: Path, rel: str, items: list[Item], missing: list[str]) -> None:
    (items.append(Item(source, rel)) if source.is_file() else missing.append(str(source)))


# ------------------------------------------------------------- the groups ----
#
# Each function returns (items, missing). "missing" means a file the method says
# should exist and does not -- always reported, never shrugged off, because a
# short cohort otherwise passes downstream as a smaller, complete one.


def plan_meshes(ctx: Context) -> tuple[list[Item], list[str]]:
    """Bone, cartilage and meniscus surfaces: the input to reconstruction."""
    items: list[Item] = []
    missing: list[str] = []
    timepoint = ctx.spec["cohort"]["timepoint"]
    for knee in ctx.cohort:
        subject, side = knee.rsplit("_", 1)
        src_dir = ctx.roots["oai_meshes"] / timepoint / subject
        for suffix in ctx.spec["mesh_suffixes"]:
            name = f"{subject}_{side}_{suffix}.vtk"
            _expect(src_dir / name, f"meshes/{timepoint}/{subject}/{name}", items, missing)
    return items, missing


def plan_segmentations(ctx: Context) -> tuple[list[Item], list[str]]:
    """The label maps the meshes were made from.

    Both are vendored. Only one is read at run time (the left-knee mirroring
    correction needs the image geometry), but the pair is what documents the
    two-network provenance, and that is easy to get wrong from filenames alone.
    """
    items: list[Item] = []
    missing: list[str] = []
    timepoint = ctx.spec["cohort"]["timepoint"]
    for knee in ctx.cohort:
        subject, side = knee.rsplit("_", 1)
        src_dir = ctx.roots["oai_segmentations"] / timepoint / subject
        for template in ctx.spec["segmentation_names"]:
            name = template.format(SIDE=side)
            _expect(
                src_dir / name,
                f"segmentations/{timepoint}/{subject}/{name}",
                items,
                missing,
            )
    return items, missing


def plan_shape_fits(ctx: Context) -> tuple[list[Item], list[str], list[str]]:
    """Longitudinal shape-model fits: the thickness measurement's whole basis.

    Per visit: the latent vector and the reconstruction report (which carries
    the rigid transform and the scale factor that returns thickness to
    millimetres). At baseline only, additionally the labelled reconstruction in
    model space, which is where the anatomical region definitions come from,
    and -- for left knees -- the mirrored femur the coordinate correction needs.

    Their upstream copies are re-fit and overwritten in place, so the vendored
    copy is the record of what the thickness track used (see ``FROZEN_GROUPS``).

    Follow-up visits are not a complete grid. Participants missed imaging
    visits, so a knee legitimately has fits at some visits and not others, and
    the thinning rate is fit through whatever visits a knee has. Absences after
    baseline are therefore reported as coverage, not as missing inputs. A
    missing *baseline* fit is a different matter and is treated as a defect:
    baseline is the high-leverage endpoint of every slope.
    """
    items: list[Item] = []
    missing: list[str] = []
    site = ctx.spec["shape_fit_site"]
    bones = ("femur", "tibia", "patella")
    baseline = ctx.spec["visits"][0]
    coverage: dict[str, int] = {}

    for visit in ctx.spec["visits"]:
        fit_dir = ctx.roots["shape_fits"] / visit / "fit_nsm" / site
        present = set()
        for knee in ctx.cohort:
            for bone in bones:
                stem = f"RECON_{knee}_{bone}"
                wanted = [
                    (f"{bone}_latents", f"{stem}_latent.npy"),
                    (f"{bone}_recon_results", f"{stem}_recon_results.json"),
                ]
                if visit == baseline:
                    wanted.append(
                        (f"{bone}_meshes_nsm_space", f"{stem}_mesh_nsm_space.vtk")
                    )
                for sub, name in wanted:
                    src = fit_dir / sub / name
                    if src.is_file():
                        items.append(
                            Item(src, f"latents_thickness/{visit}/{site}/{sub}/{name}")
                        )
                        present.add(knee)
                    elif visit == baseline:
                        missing.append(str(src))
        coverage[visit] = len(present)

    # Left-knee mirrored femur, baseline only. The subject directory carries a
    # trailing identifier that is not derivable from the knee, so it is matched
    # rather than constructed.
    surfaces = ctx.roots["shape_fits"] / baseline / "sub_region_surfaces" / site
    for knee in ctx.cohort:
        subject, side = knee.rsplit("_", 1)
        if side != "LEFT":
            continue
        matches = sorted(surfaces.glob(f"{subject}_{side}_*/{knee}_femur_mesh_flipped.vtk"))
        if not matches:
            missing.append(str(surfaces / f"{subject}_{side}_*" / f"{knee}_femur_mesh_flipped.vtk"))
            continue
        src = matches[0]
        items.append(
            Item(
                src,
                f"latents_thickness/{baseline}/{site}/sub_region_surfaces/"
                f"{src.parent.name}/{src.name}",
            )
        )

    n = len(ctx.cohort)
    notes = [
        "knees with a fit, per visit: "
        + ", ".join(f"{v} {c}/{n}" for v, c in coverage.items())
    ]
    return items, missing, notes


def plan_gait(ctx: Context) -> tuple[list[Item], list[str]]:
    """Measured gait, per donor: the body-scaled base model and the motion data."""
    items: list[Item] = []
    missing: list[str] = []
    bases = ctx.roots["requirements"] / ctx.spec["gait_tree"]["multigait_bases"]
    addb = ctx.roots["requirements"] / ctx.spec["gait_tree"]["addbiomechanics"]
    for gait in ctx.spec["gaits"]:
        for root, label in ((bases / gait, "bases"), (addb / gait, "addbiomechanics")):
            if not root.is_dir():
                missing.append(str(root))
                continue
            items.extend(_tree(root, f"gait/{label}/{gait}"))
    return items, missing


def plan_reference(ctx: Context) -> tuple[list[Item], list[str]]:
    """Shape-model weights, the reference knee model, ligament and contact parameters.

    Mirrored verbatim rather than reorganised. The tree is referenced by
    relative path from several places, and a tidier layout here would buy
    nothing and cost a migration.
    """
    items: list[Item] = []
    missing: list[str] = []
    for entry in ctx.spec["reference_tree"]["include"]:
        src = ctx.roots["requirements"] / entry
        if src.is_dir():
            items.extend(_tree(src, f"reference/{entry}"))
        elif src.is_file():
            items.append(Item(src, f"reference/{entry}"))
        else:
            missing.append(str(src))
    return items, missing


def plan_models(ctx: Context) -> tuple[list[Item], list[str]]:
    """Per-knee reconstructed anatomy and the subject-specific OpenSim model.

    Vendored rather than regenerated so that reproducing the simulations, the
    thickness analysis or the figures needs no GPU. The code and weights to
    rebuild these from the meshes are in the repository for anyone who wants to.
    """
    items: list[Item] = []
    missing: list[str] = []
    run = ctx.spec["cohort"]["step1_run"]
    for knee in ctx.cohort:
        knee_dir = ctx.knee_dir(run, knee)
        for entry in ctx.spec["model_tree"]["include"]:
            src = knee_dir / entry
            if src.is_dir():
                items.extend(_tree(src, f"{run}/{knee_dir.name}/{entry}"))
            elif src.is_file():
                items.append(Item(src, f"{run}/{knee_dir.name}/{entry}"))
            elif entry != "timing.txt":  # advisory only; absent on some knees
                missing.append(str(src))
    return items, missing


def _reference_results_template(spec: dict) -> str:
    """Where one reference-gait knee's results live, relative to the stage.

    Taken from ``config/reference_gait.json`` rather than restated in
    ``vendor_sources.json``, so the directory this stage writes into and the
    directory every derivation reads from are the same string by construction.

    If they differed, every reference-gait knee would be read as "no
    joint_mechanics.h5 (simulation did not complete)" -- a status real knees
    produce too -- and nothing would raise.
    """
    config = REPO / "config" / "reference_gait.json"
    results_dir = json.loads(config.read_text(encoding="utf-8"))["run"]["results_dir"]
    prefix = STAGES["simulations"] + "/"
    if not results_dir.startswith(prefix):
        raise ValueError(
            f"{config} puts the reference condition's results at {results_dir!r}, "
            f"which is not under {prefix!r}. Vendoring writes into the simulations "
            "stage; if the condition now lives somewhere else, this group needs to "
            "be rewritten rather than pointed at a guess."
        )
    return results_dir[len(prefix):]


def plan_simulations(ctx: Context) -> tuple[list[Item], list[str], list[str]]:
    """The reference-gait arm's COMAK output, one results tree per knee.

    Only the reference arm. The measured-gait cohorts are written into this
    stage by ``simulate_gait.py``: they are produced, not vendored.

    A knee with no results directory is a defect: the condition was never
    attempted for it. A knee whose directory exists but holds no
    ``joint_mechanics.h5`` is a simulation that did not converge, which is a
    property of the data and is reported as coverage.
    """
    conf = ctx.spec["simulation_tree"]
    template = _reference_results_template(ctx.spec)
    run, subdir = conf["run"], conf["results_subdir"]

    items: list[Item] = []
    missing: list[str] = []
    converged = 0
    for knee in ctx.cohort:
        knee_dir = ctx.knee_dir(run, knee)
        source = knee_dir / subdir
        if not source.is_dir():
            missing.append(str(source))
            continue
        if (source / conf["required_output"]).is_file():
            converged += 1
        items.extend(_tree(source, template.replace("{knee}", knee_dir.name)))

    n = len(ctx.cohort)
    attempted = n - len(missing)
    notes = [
        f"reference arm: {attempted}/{n} knees simulated, "
        f"{converged}/{n} produced {conf['required_output']} "
        f"({attempted - converged} did not converge)"
    ]
    return items, missing, notes


def plan_group_labels(ctx: Context) -> tuple[list[Item], list[str]]:
    """The cohort's disease-group assignment, one JSON mapping group to knees.

    Every analysis that compares groups reads this vendored copy, so group
    membership is pinned alongside the rest of the inputs.
    """
    conf = ctx.spec["group_labels"]
    items: list[Item] = []
    missing: list[str] = []
    _expect(ctx.roots["queues"] / conf["source"], conf["dest"], items, missing)
    return items, missing


GROUPS = {
    "meshes": ("inputs", plan_meshes),
    "segmentations": ("inputs", plan_segmentations),
    "shape_fits": ("inputs", plan_shape_fits),
    "gait": ("inputs", plan_gait),
    "reference": ("inputs", plan_reference),
    "group_labels": ("inputs", plan_group_labels),
    "models": ("models", plan_models),
    "simulations": ("simulations", plan_simulations),
}

# Groups whose vendored copy is the record of what the paper used, and whose
# upstream source is known to change in place. vendor() trusts its journal on a
# size match, and these files keep the same size when re-fit (a latent is always
# 4,224 bytes), so a re-vendor would import the new versions unnoticed. Copying
# one requires --unfreeze, so it is always an explicit choice.
FROZEN_GROUPS = {
    "shape_fits": (
        "the longitudinal shape-model fits are being re-fit in place upstream "
        "(cohort V00 files already differ from the vendored "
        "copies). The vendored latents_thickness/ tree is what the paper's thickness "
        "track was built from."
    ),
}


def build_plans(ctx: Context, names: list[str]) -> list[Plan]:
    plans = []
    for name in names:
        stage, fn = GROUPS[name]
        result = fn(ctx)
        items, missing = result[0], result[1]
        notes = result[2] if len(result) > 2 else []
        plans.append(Plan(name, stage, items, missing, notes))
    return plans


# ---------------------------------------------------------------- copying ----


def _copy_hashed(src: Path, dst: Path) -> tuple[str, int]:
    """Copy, hashing the bytes on their way past. One read, not two."""
    h = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    size = 0
    try:
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            while chunk := fi.read(_CHUNK):
                h.update(chunk)
                fo.write(chunk)
                size += len(chunk)
        shutil.copystat(src, tmp)
        os.replace(tmp, dst)  # atomic: a killed job leaves no half-file
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return h.hexdigest(), size


def _read_journal(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    done = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                entry = json.loads(line)
                done[entry["rel"]] = entry
    return done


def vendor(plan: Plan, jobs: int, quiet: bool = False) -> list[FileRecord]:
    """Copy a group's files, returning a FileRecord for every file in it.

    Records are returned for skipped files too. The manifest describes the
    stage's contents, not one run's activity, so a resumed run and a fresh one
    must produce the same manifest.
    """
    root = data_root()
    target = stage_dir(plan.stage, create=True)
    journal_path = target / JOURNAL_NAME
    done = _read_journal(journal_path)

    todo = []
    records: list[FileRecord] = []
    for item in plan.items:
        prior = done.get(item.rel)
        dst = target / item.rel
        # Trust the journal only if it agrees with both ends: same source, and a
        # destination whose size still matches what was written.
        if (
            prior
            and prior.get("source") == str(item.source)
            and dst.is_file()
            and dst.stat().st_size == prior["size"]
            and item.source.stat().st_size == prior["size"]
        ):
            records.append(
                FileRecord(
                    path=dst.resolve().relative_to(root).as_posix(),
                    sha256=prior["sha256"],
                    size=prior["size"],
                    source=prior["source"],
                )
            )
        else:
            todo.append(item)

    if not todo:
        if not quiet:
            print(f"  {plan.name}: {len(records)} file(s) already vendored")
        return records

    total = len(todo)
    written = 0
    with open(journal_path, "a") as journal, ThreadPoolExecutor(jobs) as pool:

        def one(item: Item) -> tuple[Item, str, int]:
            sha, size = _copy_hashed(item.source, target / item.rel)
            return item, sha, size

        for item, sha, size in pool.map(one, todo):
            entry = {
                "rel": item.rel,
                "source": str(item.source),
                "sha256": sha,
                "size": size,
            }
            journal.write(json.dumps(entry) + "\n")
            written += 1
            if written % 500 == 0:
                journal.flush()  # so an interrupted run resumes near where it died
                if not quiet:
                    print(f"  {plan.name}: {written}/{total}", flush=True)
            records.append(
                FileRecord(
                    path=(target / item.rel).resolve().relative_to(root).as_posix(),
                    sha256=sha,
                    size=size,
                    source=str(item.source),
                )
            )
    if not quiet:
        print(f"  {plan.name}: {written} copied, {len(records) - written} already present")
    return records


def record_stage_residue(stage: str, produced, jobs: int) -> list[FileRecord]:
    """Hash the stage's files that this run did not copy. Opt-in; see the warning.

    The motivation is real: ``manifest verify <stage>`` should mean what it says,
    and the simulations stage holds ten conditions of which vendoring copies one
    -- the other nine were written in place by ``pipeline/simulate_gait.py``. A
    manifest built from the copy alone reports "OK, N files match" over a tenth
    of the stage, which is the same shape of half-truth as a coverage file
    declaring ten conditions with data for nine.

    Not the default, because of its size. On this dataset the simulations stage
    is 8,657,508 files; at the 549 bytes/record the other manifests run to,
    recording all of them produces a 4.75 GB ``_manifest.json``, and
    ``read_manifest`` parses a manifest with a single ``json.loads``, so every
    downstream ``verify_inputs`` call would need tens of GB of RAM to check its
    inputs. The run itself holds about 29 GB resident and takes two hours. A
    manifest nothing can afford to read does not make the stage verifiable.

    Per-cohort manifests, written by the stage that produces each cohort, would
    avoid one file describing eight million others. Without them, ``main`` by
    default records what was vendored and states in the notes what is not
    covered.

    ``source`` is left unset on these records. It is the vendoring stage's field
    for "what this file was before it became ours", and these files were never
    anywhere else.
    """
    root = data_root()
    target = stage_dir(stage, create=True)
    ignore = STAGE_IGNORE.get(stage, ())
    already = {r.path for r in produced}

    candidates: list[Path] = []
    for parent, dirs, files in os.walk(target, followlinks=True):
        if Path(parent) == target:
            dirs[:] = [d for d in dirs if d not in ignore]
        for name in files:
            if name.endswith((".part", ".jsonl")) or name == "_manifest.json":
                continue
            path = Path(parent) / name
            if path.is_file() and path.resolve().relative_to(root).as_posix() not in already:
                candidates.append(path)

    if not candidates:
        return []

    journal_path = target / ADOPT_JOURNAL_NAME
    done = _read_journal(journal_path)

    records: list[FileRecord] = []
    todo: list[tuple[Path, str, os.stat_result]] = []
    for path in sorted(candidates):
        rel = path.resolve().relative_to(root).as_posix()
        stat = path.stat()
        prior = done.get(rel)
        # Trust the journal only where the file has not moved underneath it.
        if prior and prior["size"] == stat.st_size and prior.get("mtime_ns") == stat.st_mtime_ns:
            records.append(FileRecord(path=rel, sha256=prior["sha256"], size=prior["size"]))
        else:
            todo.append((path, rel, stat))

    total_bytes = sum(s.st_size for _, _, s in todo)
    print(f"  hashing {len(todo):,} file(s) already in {STAGES[stage]} "
          f"({_human(total_bytes)}); {len(records):,} from the journal", flush=True)
    if not todo:
        return records

    started = time.time()
    done_bytes = 0
    with open(journal_path, "a") as journal, ThreadPoolExecutor(jobs) as pool:

        def one(job):
            path, rel, stat = job
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                while chunk := fh.read(_CHUNK):
                    h.update(chunk)
            return rel, h.hexdigest(), stat

        for index, (rel, sha, stat) in enumerate(pool.map(one, todo), start=1):
            journal.write(json.dumps({"rel": rel, "sha256": sha, "size": stat.st_size,
                                      "mtime_ns": stat.st_mtime_ns}) + "\n")
            records.append(FileRecord(path=rel, sha256=sha, size=stat.st_size))
            done_bytes += stat.st_size
            if index % 20_000 == 0:
                journal.flush()
                rate = done_bytes / max(time.time() - started, 1e-9)
                left = (total_bytes - done_bytes) / max(rate, 1e-9) / 60
                print(f"    {index:,}/{len(todo):,} "
                      f"({_human(rate)}/s, ~{left:.0f} min left)", flush=True)
    return records


def _partial_coverage_note(stage: str, records) -> dict[str, object]:
    """State, in the manifest itself, which parts of the stage it does not cover.

    A manifest that describes one tenth of its stage without saying so would
    pass verification while covering little, so the gap is named rather than
    left to be inferred from a file count. Top-level entries only -- one ``listdir``, not
    a walk of eight million files, because a note that costs ten minutes to
    produce is a note people will start skipping.
    """
    target = stage_dir(stage)
    ignore = set(STAGE_IGNORE.get(stage, ()))
    covered = set(stage_coverage(stage, records))

    # Split by whether a directory holds knees. Some directories here are named
    # for cohorts and hold none: simulation workers create them at startup for
    # their error logs, and a cohort is resolved by donor name, not by this tag.
    # Listing them beside the real cohorts would double the "not covered" entries
    # and suggest that a directory named for a cohort is a cohort -- the
    # confusion that makes a sweep over them report a clean finish over zero
    # knees.
    holds_knees, no_knees = set(), set()
    for entry in target.iterdir():
        if not entry.is_dir() or entry.name in ignore or entry.name in covered:
            continue
        (holds_knees if any(entry.glob("*_00m_*")) else no_knees).add(entry.name)

    if not holds_knees and not no_knees:
        return {"scope": "every top-level entry in this stage is covered"}
    return {
        "scope": "PARTIAL -- this manifest covers only what this run vendored",
        "covered": sorted(covered),
        "not_covered": sorted(holds_knees),
        "no_knees": {
            "dirs": sorted(no_knees),
            "note": "named for cohorts, hold none -- worker error-log "
                    "directories, recreated on every run. Not data.",
        },
        "why": (
            "These were produced in place by pipeline/simulate_gait.py rather "
            "than vendored, and that stage does not write manifests. They are "
            "not hashed here because one manifest over the whole stage comes to "
            "4.75 GB, which read_manifest cannot parse affordably -- see "
            "record_stage_residue. Consequence to be explicit about: "
            "'manifest verify simulations' passing proves the vendored "
            "condition is intact and proves nothing about the ones listed under "
            "not_covered."
        ),
    }


def stage_coverage(stage: str, records) -> dict[str, dict[str, int]]:
    """Files and bytes per top-level directory of a stage, for the manifest notes.

    So that "which conditions does this manifest cover" is answerable by
    reading the manifest, rather than by trusting that whoever wrote it covered
    everything.
    """
    coverage: dict[str, dict[str, int]] = {}
    prefix = STAGES[stage] + "/"
    for record in records:
        rest = record.path[len(prefix):] if record.path.startswith(prefix) else record.path
        head = rest.split("/", 1)[0]
        entry = coverage.setdefault(head, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += record.size
    return dict(sorted(coverage.items()))


# ---------------------------------------------------------------- reports ----


def _human(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= step
    return f"{value:,.1f} TB"


def report(plans: list[Plan], cohort: list[str]) -> int:
    print(f"\ncohort: {len(cohort)} knees\n")
    print(f"  {'group':<18} {'stage':<7} {'files':>9} {'size':>12}  missing")
    print(f"  {'-' * 18} {'-' * 7} {'-' * 9} {'-' * 12}  {'-' * 7}")
    files = size = missing = 0
    for plan in plans:
        n, b, m = len(plan.items), plan.size, len(plan.missing)
        files, size, missing = files + n, size + b, missing + m
        flag = f"{m:,}" if m else "-"
        print(f"  {plan.name:<18} {plan.stage:<7} {n:>9,} {_human(b):>12}  {flag}")
    print(f"  {'-' * 18} {'-' * 7} {'-' * 9} {'-' * 12}  {'-' * 7}")
    print(f"  {'total':<18} {'':<7} {files:>9,} {_human(size):>12}  {missing or '-'}")
    print(
        "\n  Sizes are apparent bytes. The filesystem these come off compresses, "
        "so\n  the space actually consumed is smaller -- do not quote these as "
        "disk usage."
    )

    for plan in plans:
        for note in plan.notes:
            print(f"\n  {plan.name}: {note}")

    for plan in plans:
        if plan.missing:
            print(f"\n  {plan.name}: {len(plan.missing)} expected file(s) not at the source")
            for path in plan.missing[:8]:
                print(f"    {path}")
            if len(plan.missing) > 8:
                print(f"    ... and {len(plan.missing) - 8:,} more")
    return missing


# -------------------------------------------------------------------- CLI ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vendor_inputs.py",
        description=(
            "Copy external inputs into the data root and record their provenance. "
            "Reports by default; copies only with --execute."
        ),
    )
    parser.add_argument(
        "--group", action="append", choices=sorted(GROUPS),
        help="vendor only this group (repeatable; default: all)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="actually copy. Without it, nothing is written.",
    )
    parser.add_argument(
        "--jobs", type=int, default=8,
        help="concurrent copies (default: 8). These are network reads; more "
             "threads help up to a point and then contend.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use only the first N knees. For rehearsing the run, not for real use.",
    )
    parser.add_argument(
        "--hash-whole-stage", action="store_true",
        help="also hash files already in the stage that this run did not copy, "
             "so the manifest covers the whole stage. Read "
             "record_stage_residue's docstring first: on the simulations stage "
             "this produces a 4.75 GB manifest that read_manifest cannot "
             "affordably parse.",
    )
    parser.add_argument(
        "--allow-missing", action="store_true",
        help="proceed even though expected source files are absent. Every "
             "absence is reported either way; this only decides whether it stops.",
    )
    parser.add_argument(
        "--unfreeze", action="append", default=[], choices=sorted(FROZEN_GROUPS),
        help="allow --execute to copy a frozen group (see FROZEN_GROUPS). Its new "
             "files replace the copies the paper was built from.",
    )
    args = parser.parse_args(argv)

    spec = _load_spec()
    roots = _roots(spec)
    for name, path in roots.items():
        if not path.is_dir():
            print(f"source root {name!r} not found: {path}", file=sys.stderr)
            return 2

    cohort = resolve_cohort(spec, roots)
    if args.limit:
        cohort = cohort[: args.limit]
    ctx = Context(spec=spec, roots=roots, cohort=cohort)

    names = args.group or list(GROUPS)
    frozen = [n for n in names if n in FROZEN_GROUPS and n not in args.unfreeze]
    if args.execute and frozen:
        raise SystemExit(
            "refusing to vendor frozen group(s): "
            + "; ".join(f"{n}: {FROZEN_GROUPS[n]}" for n in frozen)
            + "\nPass --group for the groups you want, or --unfreeze <group> if "
            "replacing the paper's copies is intended."
        )
    print(f"planning {len(names)} group(s) over {len(cohort)} knees ...", flush=True)
    plans = build_plans(ctx, names)
    missing = report(plans, cohort)

    if not args.execute:
        print("\nnothing copied (--execute to do it)")
        return 0

    if missing and not args.allow_missing:
        print(
            f"\nrefusing to vendor: {missing:,} expected source file(s) are absent.\n"
            "A dataset that is quietly short of what the method calls for is worse "
            "than one that failed to build. Re-run with --allow-missing if the "
            "absences above are understood and intended.",
            file=sys.stderr,
        )
        return 1

    print()
    by_stage: dict[str, list[FileRecord]] = {}
    for plan in plans:
        by_stage.setdefault(plan.stage, []).extend(vendor(plan, args.jobs))

    if args.limit:
        print("\n--limit was used: manifests not written (they would describe a partial stage)")
        return 0

    if "inputs" in by_stage:
        cohort_path = stage_dir("inputs", create=True) / COHORT_NAME
        cohort_path.write_text(
            json.dumps(
                {
                    "timepoint": spec["cohort"]["timepoint"],
                    "excluded": spec["cohort"]["exclude"],
                    "n_knees": len(cohort),
                    "knees": cohort,
                },
                indent=2,
            )
            + "\n"
        )
        by_stage["inputs"].append(FileRecord.of(cohort_path))

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for stage, records in by_stage.items():
        groups = [p.name for p in plans if p.stage == stage]
        notes = {
            "vendored": stamp,
            "groups": groups,
            "n_knees": len(cohort),
            "source_roots": {k: str(v) for k, v in roots.items()},
        }
        if args.hash_whole_stage:
            records = records + record_stage_residue(stage, records, args.jobs)
            notes["coverage"] = "whole stage: every file present was hashed"
        else:
            notes["coverage"] = _partial_coverage_note(stage, records)
        notes["contents"] = stage_coverage(stage, records)
        if STAGE_IGNORE.get(stage):
            notes["not_recorded"] = list(STAGE_IGNORE[stage])
        path = write_manifest(
            stage,
            inputs=[],  # this stage's inputs are external; see each record's source
            outputs=records,
            notes=notes,
        )
        print(f"wrote {path}  ({len(records):,} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
