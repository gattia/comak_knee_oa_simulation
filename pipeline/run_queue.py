"""Worker loop that runs the ``simulate`` stage for knees taken from a work list.

Optional: nothing needs it to reproduce a result. Each invocation is one worker.
It claims one knee at a time from a SQLite work list, makes the same
``simulate`` call as ``run_knee.py simulate``, and records where the knee ended
up. The script submits no jobs itself; to spread a cohort over a cluster, start
several workers (for example one per scheduler job) on one shared work list.
Claiming is atomic, so two workers cannot take the same knee.

Nothing scientific lives here. The queue decides *when* a knee runs, never
*what* running it means. Delete the database and every result is still
reproducible one knee at a time.

Outcomes are recorded by kind, not as a boolean. Across several hundred knees a
handful will exceed a solver's time budget, and "43 failed" is not a usable
answer -- "38 exceeded the sweep budget, 5 exceeded the COMAK budget" is, and it
points at which budget to raise.

    python pipeline/run_queue.py --config config/measured_gait/Subject_102.json \\
        --queue work/Subject_102.db

    python pipeline/run_queue.py --config ... --queue ... --seed cohort.json
    python pipeline/run_queue.py --queue ... --status
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comak_knee_oa.config import Knee, load_config  # noqa: E402
from comak_knee_oa.paths import stage_dir  # noqa: E402
from simulate_gait import simulate, verify_outputs  # noqa: E402

# One row per knee. `state` is where it ended up; for a failure it is the
# solver's own bucket name, so the table answers "how did these fail" directly.
SCHEMA = """
CREATE TABLE IF NOT EXISTS work (
    knee     TEXT PRIMARY KEY,
    state    TEXT NOT NULL DEFAULT 'todo',
    worker   TEXT,
    started  TEXT,
    finished TEXT,
    detail   TEXT
);
CREATE INDEX IF NOT EXISTS work_state ON work (state);
"""


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=300.0, isolation_level=None)
    # Rollback journal, not WAL. WAL keeps its index in shared memory on one
    # host, so workers on several nodes sharing the file over a network
    # filesystem -- the cluster case this script exists for -- can corrupt it.
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.executescript(SCHEMA)
    return connection


def seed(connection, knees: list[str]) -> int:
    """Add knees not already listed. Re-seeding never disturbs finished work."""
    with connection:
        before = connection.execute("SELECT count(*) FROM work").fetchone()[0]
        connection.executemany(
            "INSERT OR IGNORE INTO work (knee) VALUES (?)", [(k,) for k in knees]
        )
        after = connection.execute("SELECT count(*) FROM work").fetchone()[0]
    return after - before


def claim(connection, worker: str) -> str | None:
    """Take one queued knee, atomically.

    The UPDATE ... WHERE state='todo' is what makes the claim exclusive: two
    workers race on the same row and only one changes it, so the loser reads
    back nothing and moves on rather than duplicating an hour of solver time.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connection:
        row = connection.execute(
            "SELECT knee FROM work WHERE state='todo' ORDER BY knee LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        changed = connection.execute(
            "UPDATE work SET state='running', worker=?, started=? "
            "WHERE knee=? AND state='todo'",
            (worker, now, row[0]),
        ).rowcount
    return row[0] if changed else None


def finish(connection, knee: str, state: str, detail: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connection:
        connection.execute(
            "UPDATE work SET state=?, finished=?, detail=? WHERE knee=?",
            (state, now, detail, knee),
        )


def status(connection) -> dict[str, int]:
    return dict(
        connection.execute(
            "SELECT state, count(*) FROM work GROUP BY state ORDER BY count(*) DESC"
        ).fetchall()
    )


def requeue(connection, states: list[str]) -> int:
    """Return knees in the given states to the queue.

    Mostly for 'running': a worker killed by the scheduler leaves its knee
    claimed forever, and nothing else will pick it up.
    """
    marks = ",".join("?" * len(states))
    with connection:
        return connection.execute(
            f"UPDATE work SET state='todo', worker=NULL, started=NULL WHERE state IN ({marks})",
            states,
        ).rowcount


def work_loop(connection, config_path: Path, worker: str, timepoint: str, limit=None) -> int:
    config = load_config(config_path)
    done = 0
    while limit is None or done < limit:
        knee_name = claim(connection, worker)
        if knee_name is None:
            print("nothing left to claim")
            break
        knee = Knee.parse(knee_name, timepoint=timepoint)
        print(f"\n=== {knee.name} ===", flush=True)
        try:
            result = simulate(config, knee)
        except Exception:
            detail = traceback.format_exc()
            print(detail, file=sys.stderr)
            finish(connection, knee_name, "error", detail[-4000:])
            done += 1
            continue

        if result is not True:
            finish(connection, knee_name, str(result))
            done += 1
            continue

        # A completed chain is not the same as a complete result. Check the
        # files before recording it as done -- a stage that failed quietly would
        # otherwise be recorded as a success and only noticed at analysis time.
        results_dir = config.for_knee(knee).path("run", "results_dir")
        missing = verify_outputs(results_dir)
        if missing:
            finish(connection, knee_name, "incomplete_output", "; ".join(missing))
        else:
            finish(connection, knee_name, "done")
        done += 1
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_queue.py",
        description="Run knees from a work list. Optional -- see run_knee.py.",
    )
    parser.add_argument("--queue", required=True, help="SQLite work list")
    parser.add_argument("--config", help="run config (required unless --status/--seed)")
    parser.add_argument("--timepoint", default="00m")
    parser.add_argument(
        "--seed", nargs="?", const="", metavar="COHORT_JSON",
        help="add knees to the work list, then exit. Defaults to the vendored "
             "cohort in the inputs stage.",
    )
    parser.add_argument("--status", action="store_true", help="report counts by state")
    parser.add_argument(
        "--requeue", nargs="*", metavar="STATE",
        help="return knees in these states to the queue (default: running)",
    )
    parser.add_argument("--worker", default=None, help="worker name recorded per claim")
    parser.add_argument("--limit", type=int, default=None, help="stop after N knees")
    args = parser.parse_args(argv)

    queue_path = Path(args.queue)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(connect(queue_path)) as connection:
        if args.seed is not None:
            source = Path(args.seed) if args.seed else stage_dir("inputs") / "cohort.json"
            knees = json.loads(Path(source).read_text())["knees"]
            added = seed(connection, knees)
            print(f"seeded {added} new knee(s) from {source}; {len(knees)} listed")
            return 0

        if args.requeue is not None:
            states = args.requeue or ["running"]
            print(f"returned {requeue(connection, states)} knee(s) to the queue")
            return 0

        if args.status:
            counts = status(connection)
            if not counts:
                print("work list is empty")
            for state, count in counts.items():
                print(f"  {state:<24} {count:>6}")
            return 0

        if not args.config:
            parser.error("--config is required to run work")

        import os
        import socket

        worker = args.worker or f"{socket.gethostname()}:{os.getpid()}"
        count = work_loop(connection, Path(args.config), worker, args.timepoint, args.limit)
        print(f"\n{worker} finished {count} knee(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
