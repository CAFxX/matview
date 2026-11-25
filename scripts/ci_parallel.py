#!/usr/bin/env python3
"""Run independent test commands concurrently, each against its own database.

CI jobs use this to run their test modules (and shards of test_matview_sync.py)
in parallel inside one job without letting them interfere:

* every task gets a database of its own on the live server named by
  ``POSTGRES_DSN`` / ``MYSQL_DSN`` (created before the task starts, dropped
  after it ends), and the task's DSN environment variable is rewritten to point
  at it; tests already honour the database named in the DSN;
* every task gets a private ``TMPDIR`` (SQLite files, scratch files) and a
  private fuzz manifest path (``MATVIEW_FUZZ_MANIFEST``) when it sets none;
* tasks matching ``--one-at-a-time REGEX`` never overlap each other (they
  still overlap other tasks), bounding the server's peak memory;
* tasks listed with ``--serial`` (concurrency tests with deliberate
  synchronization, timing-sensitive tests) run one at a time after the
  parallel phase, alone on the machine.

Each task's output is buffered and printed as one block when the task ends, so
logs never interleave. The script fails if any task fails. A task is never
retried: only the database create/drop around it (infrastructure, not code
under test) is retried, a bounded number of times, for transient errors
(connection refused, MySQL 1205/1213, PostgreSQL 40001/40P01), and each retry
is logged.

Usage::

    python3 scripts/ci_parallel.py --jobs 3 \\
        --task "python3 test_matview_sync.py --suites 1" \\
        --task "pytest -v -rxXs test_native_bulk_statements.py" \\
        --serial "python3 test_concurrent_backfill.py"
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

TRANSIENT_PATTERNS = (
    "connection refused",
    "could not connect",
    "can't connect",
    "lost connection",
    "server has gone away",
    "the database system is starting up",
    "(1205",
    "(1213",
    "40001",
    "40p01",
    "deadlock detected",
    "could not serialize access",
    # PostgreSQL CREATE DATABASE copies template1 and needs exclusive use of it.
    "is being accessed by other users",
)
SETUP_ATTEMPTS = 5


def log(msg: str) -> None:
    print(f"[ci_parallel] {msg}", flush=True)


def is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(p in text for p in TRANSIENT_PATTERNS)


def with_setup_retries(what: str, fn: Callable[[], None]) -> None:
    """Retry a database create/drop (never a test) on transient errors only."""
    for attempt in range(1, SETUP_ATTEMPTS + 1):
        try:
            fn()
            return
        except Exception as exc:  # noqa: BLE001 - classified below
            if attempt == SETUP_ATTEMPTS or not is_transient(exc):
                raise
            delay = 0.5 * 2 ** (attempt - 1)
            log(f"RETRY {what}: attempt {attempt}/{SETUP_ATTEMPTS} hit a transient error, retrying in {delay:.1f}s: {exc}")
            time.sleep(delay)


def _replace_db(dsn: str, db: str) -> str:
    parsed = urlparse(dsn if "://" in dsn else f"mysql://{dsn}")
    return urlunparse(parsed._replace(path=f"/{db}"))


class ServerDatabases:
    """Creates and drops one database per task on a live server."""

    env_var = ""

    def create(self, db: str) -> str:
        """Create ``db`` (dropping a leftover first) and return a DSN naming it."""
        raise NotImplementedError

    def drop(self, db: str) -> None:
        raise NotImplementedError


class PostgresDatabases(ServerDatabases):
    env_var = "POSTGRES_DSN"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.admin_dsn = _replace_db(dsn, "postgres")

    def _exec(self, sql: str) -> None:
        import psycopg2

        conn = psycopg2.connect(self.admin_dsn)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql)
        finally:
            conn.close()

    def create(self, db: str) -> str:
        with_setup_retries(f"DROP DATABASE {db}", lambda: self._exec(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)'))
        with_setup_retries(f"CREATE DATABASE {db}", lambda: self._exec(f'CREATE DATABASE "{db}"'))
        return _replace_db(self.dsn, db)

    def drop(self, db: str) -> None:
        with_setup_retries(f"DROP DATABASE {db}", lambda: self._exec(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)'))


class MySQLDatabases(ServerDatabases):
    env_var = "MYSQL_DSN"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.parsed = urlparse(dsn if "://" in dsn else f"mysql://{dsn}")

    def _exec(self, sql: str) -> None:
        import pymysql

        p = self.parsed
        conn = pymysql.connect(
            host=p.hostname or "127.0.0.1",
            port=p.port or 3306,
            user=p.username or "root",
            password=p.password or "",
            autocommit=True,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
        finally:
            conn.close()

    def create(self, db: str) -> str:
        with_setup_retries(f"DROP DATABASE {db}", lambda: self._exec(f"DROP DATABASE IF EXISTS `{db}`"))
        with_setup_retries(
            f"CREATE DATABASE {db}",
            lambda: self._exec(f"CREATE DATABASE `{db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"),
        )
        return _replace_db(self.dsn, db)

    def drop(self, db: str) -> None:
        with_setup_retries(f"DROP DATABASE {db}", lambda: self._exec(f"DROP DATABASE IF EXISTS `{db}`"))


@dataclass
class Task:
    index: int
    command: str
    serial: bool
    rc: Optional[int] = None
    seconds: float = 0.0
    output: str = ""
    notes: List[str] = field(default_factory=list)


def _servers() -> List[ServerDatabases]:
    servers: List[ServerDatabases] = []
    if os.environ.get("POSTGRES_DSN"):
        servers.append(PostgresDatabases(os.environ["POSTGRES_DSN"]))
    if os.environ.get("MYSQL_DSN"):
        servers.append(MySQLDatabases(os.environ["MYSQL_DSN"]))
    return servers


def run_task(
    task: Task,
    servers: Sequence[ServerDatabases],
    prefix: str,
    print_lock: threading.Lock,
    task_timeout: float,
) -> None:
    db = f"{prefix}{task.index:02d}"
    env: Dict[str, str] = dict(os.environ)
    tmp = tempfile.mkdtemp(prefix=f"{db}-")
    env["TMPDIR"] = tmp
    env.setdefault("MATVIEW_FUZZ_MANIFEST", "")
    if not env["MATVIEW_FUZZ_MANIFEST"]:
        env["MATVIEW_FUZZ_MANIFEST"] = f"fuzz-campaign-task{task.index:02d}.json"
    created: List[ServerDatabases] = []
    start = time.monotonic()
    try:
        for server in servers:
            env[server.env_var] = server.create(db)
            created.append(server)
        # Own process group, so a timeout kills the task's whole tree. On a
        # timeout the partial output is kept and printed: a hang must not lose
        # the log (the job timeout would kill everything unprinted).
        proc = subprocess.Popen(
            task.command,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            task.output, _ = proc.communicate(timeout=task_timeout)
            task.rc = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            task.output, _ = proc.communicate()
            task.rc = 124
            task.notes.append(f"TIMEOUT: killed after {task_timeout:.0f}s (--task-timeout)")
    except Exception as exc:  # database setup failed: the task did not run
        task.rc = 125
        task.output = f"[ci_parallel] database setup for task failed: {type(exc).__name__}: {exc}\n"
    finally:
        task.seconds = time.monotonic() - start
        for server in created:
            try:
                server.drop(db)
            except Exception as exc:  # report, and fail the task: a leaked database is a bug
                task.notes.append(f"dropping database {db} failed: {exc}")
                if task.rc == 0:
                    task.rc = 125
        shutil.rmtree(tmp, ignore_errors=True)
    in_ci = bool(os.environ.get("GITHUB_ACTIONS"))
    status = "ok" if task.rc == 0 else f"FAILED (exit {task.rc})"
    header = f"{task.command}  [{status}, {task.seconds:.0f}s{', serial' if task.serial else ''}]"
    with print_lock:
        if in_ci and task.rc == 0:
            print(f"::group::{header}")
        else:
            print(f"===== {header}")
        sys.stdout.write(task.output)
        for note in task.notes:
            print(f"[ci_parallel] {note}")
        if in_ci and task.rc == 0:
            print("::endgroup::")
        elif in_ci:
            print(f"::error::{task.command} failed (exit {task.rc})")
        sys.stdout.flush()


def run_parallel_phase(
    tasks: Sequence[Task],
    jobs: int,
    is_exclusive: Callable[[Task], bool],
    run: Callable[[Task], None],
) -> None:
    """Run ``tasks`` on ``jobs`` workers, in order, never two exclusive tasks at once.

    A worker takes the first pending task it may start: an exclusive task only
    while no other exclusive task runs. When only blocked exclusive tasks are
    pending, idle workers wait for the running one to finish.
    """
    pending = list(tasks)
    cond = threading.Condition()
    state = {"exclusive_running": False}

    def worker() -> None:
        while True:
            with cond:
                while True:
                    if not pending:
                        return
                    pick = next(
                        (t for t in pending if not (is_exclusive(t) and state["exclusive_running"])), None
                    )
                    if pick is not None:
                        pending.remove(pick)
                        exclusive = is_exclusive(pick)
                        if exclusive:
                            state["exclusive_running"] = True
                        break
                    cond.wait()
            try:
                run(pick)
            finally:
                with cond:
                    if exclusive:
                        state["exclusive_running"] = False
                    cond.notify_all()

    threads = [threading.Thread(target=worker, name=f"ci-parallel-{i}") for i in range(jobs)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, required=True, help="tasks run concurrently in the parallel phase")
    ap.add_argument("--task", action="append", default=[], help="a shell command run in the parallel phase")
    ap.add_argument("--serial", action="append", default=[], help="a shell command run alone after the parallel phase")
    ap.add_argument("--db-prefix", default="mvw_task", help="prefix of the per-task database names")
    ap.add_argument(
        "--task-timeout",
        type=float,
        default=1200.0,
        help="seconds after which a task is killed and fails, with its partial output printed (default 1200)",
    )
    ap.add_argument(
        "--one-at-a-time",
        metavar="REGEX",
        default="",
        help="parallel-phase tasks whose command matches REGEX never run concurrently with each other "
        "(other tasks still fill the remaining slots); used to bound peak server memory",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", args.db_prefix):
        ap.error("--db-prefix must be a lowercase identifier")

    tasks = [Task(i, cmd, False) for i, cmd in enumerate(args.task)]
    tasks += [Task(len(tasks) + i, cmd, True) for i, cmd in enumerate(args.serial)]
    servers = _servers()
    lock = threading.Lock()
    t0 = time.monotonic()
    log(f"{len(args.task)} parallel task(s) with --jobs {args.jobs}, then {len(args.serial)} serial task(s); "
        f"per-task databases on: {', '.join(s.env_var for s in servers) or 'none (SQLite only)'}")
    exclusive = re.compile(args.one_at_a_time) if args.one_at_a_time else None
    run_parallel_phase(
        [t for t in tasks if not t.serial],
        args.jobs,
        (lambda t: bool(exclusive.search(t.command))) if exclusive else (lambda t: False),
        lambda t: run_task(t, servers, args.db_prefix, lock, args.task_timeout),
    )
    for task in (t for t in tasks if t.serial):
        run_task(task, servers, args.db_prefix, lock, args.task_timeout)
    wall = time.monotonic() - t0

    print("\nTask durations (slowest first):")
    for task in sorted(tasks, key=lambda t: -t.seconds):
        mark = "ok  " if task.rc == 0 else "FAIL"
        print(f"  {mark} {task.seconds:7.1f}s  {'[serial] ' if task.serial else ''}{task.command}")
    busy = sum(t.seconds for t in tasks)
    print(f"Wall {wall:.1f}s for {busy:.1f}s of task time (x{busy / wall if wall else 0:.2f}).")
    failed = [t for t in tasks if t.rc != 0]
    if failed:
        print(f"{len(failed)} task(s) failed: " + "; ".join(shlex.quote(t.command) for t in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
