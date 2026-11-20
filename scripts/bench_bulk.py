#!/usr/bin/env python3
"""Bulk-statement maintenance benchmark (bulk maintenance stage 1).

Reproduces the measurement tables of docs/design-bulk-maintenance.md: for each
engine, maintenance configuration, view shape, rows-per-statement and group
count it times five committed statements, in order, each in its own
transaction (commit included):

  ins      INSERT INTO t SELECT <rows> generated rows
  upd_v    UPDATE t SET v = v + 1               (every row, groups unchanged)
  upd_g    UPDATE t SET g = (g + 1) % <groups>  (every row moves group)
  del_half DELETE FROM t WHERE id % 2 = 0
  del_all  DELETE FROM t

Configurations: ``none`` (no materialized view: the cost of the statement
alone) and ``mv`` (the view maintained by the engine's only mechanism:
statement-level triggers with transition tables on PostgreSQL, row triggers on
SQLite and MySQL/MariaDB). Views (as in the design):

  agg  SUM, COUNT(*), AVG by g          mm  MIN, MAX by g (index on (g, v))
  j    SUM, COUNT(*) of t JOIN d by d.name
  ds   SELECT DISTINCT g                pj  plain t JOIN d, no aggregation
  rank RANK() OVER (PARTITION BY g)      sagg STRING_AGG(v) by g (group-scoped)

``--latency N`` (PostgreSQL only) also measures single-row statements
server-side, as in the design: a ``DO`` loop of N single-row INSERT, UPDATE and
DELETE statements against 100k rows in 10k groups, without client round trips,
reported in microseconds per statement.

After every statement of a maintained run the materialized view is compared
with the original view query executed natively (exact multiset comparison,
Decimal scale included); ``--no-verify`` skips that. A mismatch is reported in
the ``exact`` column, never hidden.

This is a measurement tool, not a test. It is not run by CI. Timings depend on
the host, the container runtime and concurrent load; compare configurations
measured in one run, and treat differences under about 30% as noise (see the
design document).

Running against a containerized engine
--------------------------------------
::

    docker run -d --name bench-pg18 -e POSTGRES_PASSWORD=password \\
        -e POSTGRES_DB=matview_test -p 55432:5432 postgres:18
    docker run -d --name bench-my84 -e MYSQL_ROOT_PASSWORD=password \\
        -e MYSQL_DATABASE=matview_test -p 53306:3306 mysql:8.4
    python3 scripts/bench_bulk.py \\
        --postgres-dsn postgresql://postgres:password@127.0.0.1:55432/matview_test \\
        --mysql-dsn mysql://root:password@127.0.0.1:53306/matview_test \\
        --sqlite --rows 1,1000 --groups 1,100 --views agg,mm --json bench.json
    docker rm -f bench-pg18 bench-my84

MariaDB uses ``--mysql-dsn`` too (the product is read from the server). Each
run creates and drops its own schema (PostgreSQL) or database (MySQL/MariaDB);
SQLite runs on a temporary file in WAL mode. ``--statement-timeout`` bounds a
single statement (PostgreSQL ``statement_timeout``; SQLite progress handler;
not enforceable for MySQL/MariaDB DML) and records ``timeout``: row-trigger
maintenance can be quadratic in rows per group (PostgreSQL row triggers were,
before stage 2), so 100k rows in one group can run for a very long time.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matview import Dialect, EngineLimitationWarning, MaterializedViewManager  # noqa: E402

STATEMENTS = ("ins", "upd_v", "upd_g", "del_half", "del_all")

VIEWS: Dict[str, str] = {
    "agg": "SELECT g, SUM(v) AS s, COUNT(*) AS c, AVG(v) AS a FROM t GROUP BY g",
    "mm": "SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM t GROUP BY g",
    "j": "SELECT d.name, SUM(t.v) AS s, COUNT(*) AS c FROM t JOIN d ON t.g = d.g GROUP BY d.name",
    "ds": "SELECT DISTINCT g FROM t",
    "pj": "SELECT t.id, t.v, d.name FROM t JOIN d ON t.g = d.g",
    "rank": "SELECT id, g, RANK() OVER (PARTITION BY g ORDER BY v) AS r FROM t",
    "sagg": "SELECT g, STRING_AGG(CAST(v AS TEXT), ',' ORDER BY v) AS sa FROM t GROUP BY g",
}

DDL: Dict[str, List[str]] = {
    "postgres": [
        "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, v BIGINT NOT NULL)",
        "CREATE INDEX t_g_v ON t (g, v)",
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
    ],
    "mysql": [
        "CREATE TABLE t (id BIGINT PRIMARY KEY, g INT NOT NULL, v BIGINT NOT NULL) ENGINE=InnoDB",
        "CREATE INDEX t_g_v ON t (g, v)",
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL) ENGINE=InnoDB",
    ],
    "sqlite": [
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, v INTEGER NOT NULL)",
        "CREATE INDEX t_g_v ON t (g, v)",
        "CREATE TABLE d (g INTEGER PRIMARY KEY, name VARCHAR(20) NOT NULL)",
    ],
}


class StatementTimeout(Exception):
    pass


@dataclass
class Result:
    engine: str
    version: str
    view: str
    config: str
    rows: int
    groups: int
    seconds: Dict[str, Optional[float]] = field(default_factory=dict)
    status: Dict[str, str] = field(default_factory=dict)
    exact: Optional[bool] = None
    note: str = ""


def _exact(v: Any) -> Tuple[str, Any]:
    if isinstance(v, Decimal):
        return ("decimal", v.as_tuple())
    if isinstance(v, int) and not isinstance(v, bool):
        return ("int", v)
    if isinstance(v, float):
        return ("float", v.hex())
    return (type(v).__name__, v)


class Engine:
    """One isolated namespace on one server."""

    def __init__(self, family: str, dsn: str, statement_timeout: Optional[float]) -> None:
        self.family = family
        self.dsn = dsn
        self.timeout = statement_timeout
        self.ns = f"mvbench_{uuid.uuid4().hex[:10]}"
        self.tmpdir: Optional[str] = None
        self.conn: Any = None
        self.version = ""
        self.deadline: Optional[float] = None

    def _connect(self, database: Optional[str]) -> Any:
        if self.family == "postgres":
            import psycopg2

            conn = psycopg2.connect(self.dsn)
            cur = conn.cursor()
            if database:
                cur.execute(f'SET search_path TO "{database}"')
            if self.timeout:
                cur.execute(f"SET statement_timeout = {int(self.timeout * 1000)}")
            conn.commit()
            return conn
        if self.family == "mysql":
            import pymysql

            p = urlparse(self.dsn)
            return pymysql.connect(
                host=p.hostname or "127.0.0.1", port=p.port or 3306, user=p.username or "root",
                password=p.password or "", database=database or (p.path.lstrip("/") or None), autocommit=False,
            )
        assert self.tmpdir is not None
        conn = sqlite3.connect(os.path.join(self.tmpdir, "bench.sqlite3"))
        MaterializedViewManager.configure_sqlite_connection(conn)
        conn.execute("PRAGMA journal_mode=WAL").fetchall()

        def progress() -> int:
            return 1 if self.deadline is not None and time.monotonic() > self.deadline else 0

        conn.set_progress_handler(progress, 100_000)
        return conn

    def __enter__(self) -> "Engine":
        if self.family == "sqlite":
            self.tmpdir = tempfile.mkdtemp(prefix="mvbench_")
        else:
            admin = self._connect(None)
            cur = admin.cursor()
            cur.execute(f'CREATE SCHEMA "{self.ns}"' if self.family == "postgres" else f"CREATE DATABASE `{self.ns}`")
            admin.commit()
            admin.close()
        self.conn = self._connect(self.ns)
        cur = self.conn.cursor()
        if self.family == "postgres":
            cur.execute("SELECT version()")
        elif self.family == "mysql":
            cur.execute("SELECT VERSION()")
        else:
            cur.execute("SELECT 'SQLite ' || sqlite_version()")
        row = cur.fetchone()
        self.version = str(row[0]).split(" on ")[0].split(" (")[0] if row else "?"
        if self.family == "mysql":
            self.version = ("MariaDB " if "mariadb" in self.version.lower() else "MySQL ") + self.version
            cur.execute("SET SESSION cte_max_recursion_depth = 10000000" if "MySQL" in self.version
                        else "SET SESSION max_recursive_iterations = 10000000")
        self.conn.commit()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.conn.rollback()
            self.conn.close()
        finally:
            if self.family == "sqlite":
                if self.tmpdir:
                    shutil.rmtree(self.tmpdir, ignore_errors=True)
            else:
                admin = self._connect(None)
                cur = admin.cursor()
                cur.execute(f'DROP SCHEMA IF EXISTS "{self.ns}" CASCADE' if self.family == "postgres"
                            else f"DROP DATABASE IF EXISTS `{self.ns}`")
                admin.commit()
                admin.close()

    @property
    def dialect(self) -> Dialect:
        return {"postgres": Dialect.POSTGRES, "mysql": Dialect.MYSQL, "sqlite": Dialect.SQLITE}[self.family]

    def generate_rows_sql(self, rows: int, groups: int) -> str:
        if self.family == "postgres":
            return (f"INSERT INTO t (id, g, v) SELECT i, i % {groups}, (i * 7919) % 100003 "
                    f"FROM generate_series(1, {rows}) AS s(i)")
        return (f"INSERT INTO t (id, g, v) WITH RECURSIVE s(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM s "
                f"WHERE i < {rows}) SELECT i, i % {groups}, (i * 7919) % 100003 FROM s")

    def run_timed(self, sql: str) -> float:
        cur = self.conn.cursor()
        self.deadline = time.monotonic() + self.timeout if self.timeout else None
        start = time.perf_counter()
        try:
            cur.execute(sql)
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            if _is_timeout(self.family, exc):
                raise StatementTimeout(str(exc)) from exc
            raise
        finally:
            self.deadline = None
        return time.perf_counter() - start

    def relation(self, sql: str) -> collections.Counter:
        cur = self.conn.cursor()
        cur.execute(sql)
        out = collections.Counter(tuple(_exact(v) for v in r) for r in cur.fetchall())
        self.conn.commit()
        return out


def _is_timeout(family: str, exc: BaseException) -> bool:
    if family == "postgres":
        return getattr(exc, "pgcode", None) == "57014"
    if family == "sqlite":
        return "interrupted" in str(exc).lower()
    return False


def run_case(family: str, dsn: str, view: str, config: str, rows: int, groups: int,
             verify: bool, statement_timeout: Optional[float]) -> Result:
    with Engine(family, dsn, statement_timeout) as eng:
        res = Result(family, eng.version, view if config != "none" else "-", config, rows, groups)
        cur = eng.conn.cursor()
        for ddl in DDL[family]:
            cur.execute(ddl)
        names = max(1, min(50, groups))
        cur.execute("INSERT INTO d (g, name) " + (
            f"SELECT i, 'n' || (i % {names}) FROM generate_series(0, {groups}) AS s(i)" if family == "postgres" else
            f"WITH RECURSIVE s(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM s WHERE i < {groups}) "
            f"SELECT i, CONCAT('n', i % {names}) FROM s" if family == "mysql" else
            f"WITH RECURSIVE s(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM s WHERE i < {groups}) "
            f"SELECT i, 'n' || (i % {names}) FROM s"))
        eng.conn.commit()
        manager: Optional[MaterializedViewManager] = None
        if config != "none":
            manager = MaterializedViewManager(eng.conn, dialect=eng.dialect)
            with warnings.catch_warnings():
                # MySQL/MariaDB registration must acknowledge the documented
                # MYSQL_NO_TRUNCATE_TRIGGER limitation; this benchmark never truncates.
                warnings.simplefilter("ignore", EngineLimitationWarning)
                manager.register_view(f"CREATE VIEW mv AS {VIEWS[view]};", ignore_limitations=(family == "mysql"))
            eng.conn.commit()
        sqls = {
            "ins": eng.generate_rows_sql(rows, groups),
            "upd_v": "UPDATE t SET v = v + 1",
            "upd_g": f"UPDATE t SET g = (g + 1) % {groups}",
            "del_half": "DELETE FROM t WHERE id % 2 = 0",
            "del_all": "DELETE FROM t",
        }
        exact = True if (verify and manager is not None) else None
        for name in STATEMENTS:
            try:
                res.seconds[name] = eng.run_timed(sqls[name])
                res.status[name] = "ok"
            except StatementTimeout:
                res.seconds[name] = None
                res.status[name] = "timeout"
                res.note = f"{name} hit the statement timeout; later statements ran on the pre-timeout state"
            if exact is not None:
                if eng.relation("SELECT * FROM mv") != eng.relation(VIEWS[view]):
                    exact = False
                    res.note = (res.note + "; " if res.note else "") + f"MISMATCH after {name}"
        res.exact = exact
        if manager is not None:
            manager.close()
        return res


def run_latency(dsn: str, view: str, config: str, n: int) -> Dict[str, Any]:
    """Server-side single-row statement cost (PostgreSQL), as in the design:
    100k rows in 10k groups, N statements of each kind in one DO loop."""
    with Engine("postgres", dsn, None) as eng:
        cur = eng.conn.cursor()
        for ddl in DDL["postgres"]:
            cur.execute(ddl)
        cur.execute("INSERT INTO d (g, name) SELECT i, 'n' || (i % 50) FROM generate_series(0, 10000) AS s(i)")
        cur.execute("INSERT INTO t (id, g, v) SELECT i, i % 10000, i FROM generate_series(1, 100000) AS s(i)")
        eng.conn.commit()
        manager: Optional[MaterializedViewManager] = None
        if config != "none":
            manager = MaterializedViewManager(eng.conn, dialect=eng.dialect)
            manager.register_view(f"CREATE VIEW mv AS {VIEWS[view]};")
            eng.conn.commit()
        cur.execute("ANALYZE")
        eng.conn.commit()
        out: Dict[str, Any] = {"version": eng.version, "view": view if config != "none" else "-", "config": config}
        for key, sql in (
            ("insert_us", "INSERT INTO t (id, g, v) VALUES (200000 + i, i % 10000, i)"),
            ("update_us", "UPDATE t SET v = v + 1 WHERE id = i"),
            ("delete_us", "DELETE FROM t WHERE id = i"),
        ):
            t0 = time.perf_counter()
            cur.execute(f"DO $$ BEGIN FOR i IN 1..{n} LOOP {sql}; END LOOP; END $$")
            eng.conn.commit()
            out[key] = (time.perf_counter() - t0) / n * 1e6
        out["exact"] = (eng.relation("SELECT * FROM mv") == eng.relation(VIEWS[view])) if manager is not None else None
        if manager is not None:
            manager.close()
        return out


def _parse_ints(text: str) -> List[int]:
    return [int(x.replace("_", "").replace("k", "000")) for x in text.split(",") if x]


def _fmt(s: Optional[float], status: str) -> str:
    if status == "timeout":
        return "timeout"
    return "-" if s is None else (f"{s:.3f}" if s >= 0.1 else f"{s:.5f}")


def render_table(results: Sequence[Result]) -> str:
    head = "| engine | view | config | rows | groups | " + " | ".join(STATEMENTS) + " | exact |"
    lines = [head, "|" + "---|" * (6 + len(STATEMENTS))]
    for r in results:
        ex = "n/a" if r.exact is None else ("yes" if r.exact else "**NO**")
        cells = " | ".join(_fmt(r.seconds.get(s), r.status.get(s, "")) for s in STATEMENTS)
        lines.append(f"| {r.version} | {r.view} | {r.config} | {r.rows} | {r.groups} | {cells} | {ex} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--postgres-dsn", default=None, help="PostgreSQL DSN (default: $POSTGRES_DSN with --postgres)")
    ap.add_argument("--mysql-dsn", default=None, help="MySQL/MariaDB DSN (default: $MYSQL_DSN with --mysql)")
    ap.add_argument("--postgres", action="store_true", help="use $POSTGRES_DSN")
    ap.add_argument("--mysql", action="store_true", help="use $MYSQL_DSN")
    ap.add_argument("--sqlite", action="store_true", help="benchmark the in-process sqlite3 library")
    ap.add_argument("--rows", default="1,1000,100000", help="rows per statement (comma list)")
    ap.add_argument("--groups", default="1,100,10000", help="distinct group keys (comma list)")
    ap.add_argument("--views", default="agg,mm,j,ds,pj", help=f"comma list of {sorted(VIEWS)}")
    ap.add_argument("--configs", default="none,mv", help="comma list of none,mv")
    ap.add_argument("--latency", type=int, default=0,
                    help="PostgreSQL: also time N single-row statements of each kind server-side (us/stmt)")
    ap.add_argument("--statement-timeout", type=float, default=None, help="seconds per statement (see docstring)")
    ap.add_argument("--no-verify", action="store_true", help="skip the exact comparison after each statement")
    ap.add_argument("--json", default=None, help="also write the results as JSON to this path")
    args = ap.parse_args(argv)

    engines: List[Tuple[str, str]] = []
    pg = args.postgres_dsn or (os.environ.get("POSTGRES_DSN") if args.postgres else None)
    my = args.mysql_dsn or (os.environ.get("MYSQL_DSN") if args.mysql else None)
    if pg:
        engines.append(("postgres", pg))
    if my:
        engines.append(("mysql", my))
    if args.sqlite:
        engines.append(("sqlite", ""))
    if not engines:
        ap.error("name at least one engine: --postgres-dsn, --mysql-dsn, --postgres, --mysql or --sqlite")
    views = [v for v in args.views.split(",") if v]
    unknown = sorted(set(views) - set(VIEWS))
    if unknown:
        ap.error(f"unknown views {unknown}")
    configs = [c for c in args.configs.split(",") if c]
    if set(configs) - {"none", "mv"}:
        ap.error("configs must be none or mv")

    results: List[Result] = []
    for family, dsn in engines:
        for rows in _parse_ints(args.rows):
            for groups in _parse_ints(args.groups):
                for config in configs:
                    for view in (views if config != "none" else views[:1]):
                        r = run_case(family, dsn, view, config, rows, groups, not args.no_verify,
                                     args.statement_timeout)
                        results.append(r)
                        cells = " ".join(f"{s}={_fmt(r.seconds.get(s), r.status.get(s, ''))}" for s in STATEMENTS)
                        print(f"[bench] {r.version} {r.view} {config} rows={rows} groups={groups} {cells} "
                              f"exact={r.exact} {r.note}", file=sys.stderr, flush=True)
    print(render_table(results))
    latency: List[Dict[str, Any]] = []
    if args.latency and pg:
        for config in configs:
            for view in (views if config != "none" else views[:1]):
                lat = run_latency(pg, view, config, args.latency)
                latency.append(lat)
                print(f"[latency] {lat}", file=sys.stderr, flush=True)
        print()
        print("| engine | view | config | INSERT us | UPDATE us | DELETE us | exact |")
        print("|---|---|---|---|---|---|---|")
        for lat in latency:
            print(f"| {lat['version']} | {lat['view']} | {lat['config']} | {lat['insert_us']:.1f} | "
                  f"{lat['update_us']:.1f} | {lat['delete_us']:.1f} | {lat['exact']} |")
    if args.json:
        Path(args.json).write_text(json.dumps({"bulk": [asdict(r) for r in results], "latency": latency}, indent=2) + "\n")
    return 1 if any(r.exact is False for r in results) or any(lat["exact"] is False for lat in latency) else 0


if __name__ == "__main__":
    sys.exit(main())
