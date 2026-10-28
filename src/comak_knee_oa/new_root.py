"""Create a new data root for one run, borrowing unchanged inputs from a baseline.

A run writes into a root of its own and never into the baseline. The new root
reuses the baseline's inputs through symlinks -- segmentation meshes,
reference models, the cohort list -- so nothing is copied that does not change,
and declares the baseline read-only so the write guard in ``comak_knee_oa.paths``
refuses any write that would reach it, through a link or otherwise.

    python -m comak_knee_oa.new_root create --root <new root> --run-name v3 \\
        --baseline <published root>

What the new root holds
-----------------------
``_run.json``
    The run declaration: the run name (what ``{run}`` in a config becomes), the
    measured gait donors the run simulates (``measured_gaits``, read by
    ``comak_knee_oa.cohorts``), the baseline, the roots it must never write, and
    every link made.
``00_inputs/``
    A real directory. Each baseline entry is a symlink, except:

    * ``gait/addbiomechanics/`` is a real, empty directory: the gait donors'
      body models and motion are re-vendored for the run. Pass
      ``--link-addbiomechanics`` to link the baseline's instead, as a run that
      reproduces the baseline does.
    * the stage records ``_manifest.json`` and ``_vendor_journal.jsonl`` are
      copied, so re-vendoring extends this root's record rather than the
      baseline's. Entries the run has not re-vendored yet keep the baseline's
      hashes, and ``verify_inputs`` reports any file that is missing.
``01_models/`` .. ``04_results/``
    Empty. The run writes them.

Derived caches the run can reuse unchanged (a thickness tree, a transform
cache) are linked later and individually, once it is established that the run
does not change their inputs::

    python -m comak_knee_oa.new_root link --root <new root> 03_derived/thickness

The measured donors
-------------------
A root that links the baseline's gait inputs can run only the baseline's donors,
so by default it declares those (a reproduction of the v2 run simulates its
nine donors, ``RSubject_121`` included). Otherwise the root declares
``cohorts.MEASURED_DEFAULT``. ``--measured-gaits`` overrides either; a root
created before the key existed records it with ``set-gaits``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from .cohorts import MEASURED_DEFAULT, check_measured_gaits
from .paths import DECLARATION_NAME, STAGES, check_run_name

__all__ = ["create", "link", "set_measured_gaits"]

# Baseline 00_inputs entries copied rather than linked: the stage's own records,
# which a run that re-vendors anything must be able to extend.
_COPIED = ("_manifest.json", "_vendor_journal.jsonl")
_AB = ("gait", "addbiomechanics")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_declaration(root: Path, decl: dict) -> None:
    target = root / DECLARATION_NAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(decl, indent=2) + "\n")
    os.replace(tmp, target)


def _plan_inputs(baseline: Path, link_ab: bool) -> tuple[list, list, list]:
    """(links [(rel, target)], copies [(rel, source)], real dirs [rel])."""
    source = baseline / STAGES["inputs"]
    links, copies, dirs = [], [], [STAGES["inputs"]]
    for entry in sorted(source.iterdir()):
        rel = f"{STAGES['inputs']}/{entry.name}"
        if entry.name in _COPIED:
            copies.append((rel, entry))
        elif entry.name == _AB[0] and entry.is_dir():
            dirs.append(rel)
            for sub in sorted(entry.iterdir()):
                sub_rel = f"{rel}/{sub.name}"
                if sub.name == _AB[1]:
                    dirs.append(sub_rel)
                    if link_ab:
                        for donor in sorted(sub.iterdir()):
                            links.append((f"{sub_rel}/{donor.name}", donor))
                else:
                    links.append((sub_rel, sub))
        else:
            links.append((rel, entry))
    return links, copies, dirs


def _measured_gaits(links, link_ab: bool, requested) -> tuple[tuple[str, ...], str]:
    """(the root's measured donors, where the list came from)."""
    ab_prefix = f"{STAGES['inputs']}/{'/'.join(_AB)}/"
    linked = tuple(rel[len(ab_prefix):] for rel, _ in links if rel.startswith(ab_prefix))
    if requested is not None:
        gaits, source = check_measured_gaits(requested, where="--measured-gaits"), "requested"
    elif link_ab:
        gaits, source = check_measured_gaits(linked, where="baseline gait inputs"), \
            "the baseline's donors (gait inputs linked from the baseline)"
    else:
        gaits, source = MEASURED_DEFAULT, "cohorts.MEASURED_DEFAULT"
    if link_ab:
        missing = [g for g in gaits if g not in linked]
        if missing:
            raise ValueError(
                f"measured donors {missing} have no gait inputs in the baseline, and this "
                "root links the baseline's instead of re-vendoring its own"
            )
    return gaits, source


def create(root: Path, run_name: str, baseline: Path, link_ab: bool = False,
           dry_run: bool = False, note: str | None = None,
           measured_gaits=None) -> dict:
    """Create ``root`` for ``run_name``; returns the declaration written."""
    check_run_name(run_name)
    root = Path(root).absolute()
    baseline = Path(baseline).resolve()
    if not (baseline / STAGES["inputs"]).is_dir():
        raise FileNotFoundError(f"baseline has no {STAGES['inputs']}: {baseline}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} exists and is not empty; a new root starts empty")
    if os.path.realpath(root).startswith(str(baseline) + os.sep):
        raise ValueError(f"{root} is inside the baseline {baseline}")

    links, copies, dirs = _plan_inputs(baseline, link_ab)
    gaits, gaits_source = _measured_gaits(links, link_ab, measured_gaits)
    # A baseline that is itself a run root borrows from older roots through its
    # links, so a write could reach them through two links. They stay read-only
    # here too.
    read_only = [str(baseline)]
    if (baseline / DECLARATION_NAME).is_file():
        inherited = json.loads((baseline / DECLARATION_NAME).read_text()).get("read_only_roots", [])
        read_only = list(dict.fromkeys(read_only + [str(r) for r in inherited]))
    decl = {
        "run_name": run_name,
        "measured_gaits": list(gaits),
        "measured_gaits_source": gaits_source,
        "created": _now(),
        "created_on": socket.gethostname(),
        "read_only": False,
        "baseline": str(baseline),
        "read_only_roots": read_only,
        "addbiomechanics": "linked from baseline" if link_ab else "re-vendored for this run",
        "links": {rel: str(target) for rel, target in links},
        "copied": {rel: str(src) for rel, src in copies},
        "note": note or "",
    }
    if dry_run:
        return decl

    from .manifest import _code_versions

    decl["created_with"] = _code_versions(("comak_knee_oa",))
    root.mkdir(parents=True, exist_ok=True)
    # Declare first: from here on the guard knows the baseline is read-only,
    # for this process and for anything that starts while the layout is built.
    _write_declaration(root, decl)
    for rel in dirs:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for stage in ("models", "simulations", "derived", "results"):
        (root / STAGES[stage]).mkdir(exist_ok=True)
    for rel, target in links:
        (root / rel).symlink_to(target)
    for rel, src in copies:
        shutil.copy2(src, root / rel)
    return decl


def link(root: Path, rel: str, note: str | None = None) -> Path:
    """Link one more baseline entry into ``root``, and record it."""
    root = Path(root).absolute()
    decl = json.loads((root / DECLARATION_NAME).read_text())
    baseline = Path(decl["baseline"])
    rel = Path(rel).as_posix().strip("/")
    target = baseline / rel
    if not target.exists():
        raise FileNotFoundError(f"baseline has no {rel}: {target}")
    dest = root / rel
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"{dest} already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(target)
    decl.setdefault("links", {})[rel] = str(target)
    decl.setdefault("link_log", []).append({"path": rel, "linked": _now(), "note": note or ""})
    _write_declaration(root, decl)
    return dest


def set_measured_gaits(root: Path, gaits, note: str | None = None) -> dict:
    """Record the measured donors of a root created before ``create`` recorded them.

    Refuses to change a list the root already declares: its simulations and
    quality-control keys were made for that list.
    """
    root = Path(root).absolute()
    decl = json.loads((root / DECLARATION_NAME).read_text())
    gaits = list(check_measured_gaits(gaits, where="--measured-gaits"))
    current = decl.get("measured_gaits")
    if current is not None and current != gaits:
        raise ValueError(f"{root} already declares measured_gaits {current}; a root's "
                         "donors are not changed after the fact -- create a new root")
    decl["measured_gaits"] = gaits
    decl.setdefault("measured_gaits_source", "requested")
    decl.setdefault("declaration_log", []).append(
        {"set": "measured_gaits", "value": gaits, "when": _now(), "note": note or ""})
    _write_declaration(root, decl)
    return decl


def _gait_list(text: str) -> list[str]:
    return [g for g in text.split(",") if g]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m comak_knee_oa.new_root",
                                     description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="create a new root for one run")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--run-name", required=True)
    p.add_argument("--baseline", required=True, type=Path,
                   help="the published root whose inputs are borrowed; it becomes read-only")
    p.add_argument("--link-addbiomechanics", action="store_true",
                   help="link the baseline's gait-donor inputs instead of leaving them to be re-vendored")
    p.add_argument("--measured-gaits", type=_gait_list, default=None, metavar="A,B,...",
                   help="the measured donors this run simulates (default: the baseline's "
                        "with --link-addbiomechanics, else cohorts.MEASURED_DEFAULT)")
    p.add_argument("--note", default=None)
    p.add_argument("--dry-run", action="store_true", help="print the plan; create nothing")

    p = sub.add_parser("link", help="link one more baseline entry, e.g. 03_derived/thickness")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("path")
    p.add_argument("--note", default=None)

    p = sub.add_parser("set-gaits", help="record the measured donors of an older root")
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--measured-gaits", required=True, type=_gait_list, metavar="A,B,...")
    p.add_argument("--note", default=None)

    args = parser.parse_args(argv)
    if args.command == "create":
        decl = create(args.root, args.run_name, args.baseline,
                      link_ab=args.link_addbiomechanics, dry_run=args.dry_run, note=args.note,
                      measured_gaits=args.measured_gaits)
        print(json.dumps(decl, indent=2))
        if args.dry_run:
            print("(dry run: nothing created)", file=sys.stderr)
        return 0
    if args.command == "set-gaits":
        decl = set_measured_gaits(args.root, args.measured_gaits, note=args.note)
        print(f"{args.root}: measured_gaits = {decl['measured_gaits']}")
        return 0
    dest = link(args.root, args.path, note=args.note)
    print(f"linked {dest} -> {os.readlink(dest)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
