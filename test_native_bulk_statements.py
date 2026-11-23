"""Native bulk-statement differential harness (bulk maintenance stage 1).

Contract under test (docs/design-bulk-maintenance.md, "Goal and contract"):
after every statement, the reader relation of each materialized view equals
the original view query over the current base tables, exactly; maintenance
runs in the writer's transaction, so statement failure, rollback and
savepoint rollback restore the view; concurrent writers on independent
connections stay exact or fail closed with an error that rolls their work
back.

Every statement here is plain client SQL sent through the DB-API driver. No
test uses ``bulk_changes()`` or any other Python-side maintenance path.

Oracle. The expected relation is the original ``SELECT`` text of each view,
executed natively on the same connection right after the materialized reader
is read. It shares nothing with the code under test except the engine. The
comparison is exact: rows are compared as multisets (``collections.Counter``)
so multiplicities count; ``Decimal`` values are compared with their scale
(``Decimal('1.50') != Decimal('1.5')``); ``int`` and ``Decimal`` never compare
equal across types; column names and the driver-reported column type codes
must match. No tolerance is used anywhere (no view here has a floating-point
aggregate).

Engines. One run tests the engines named by the environment:

* ``POSTGRES_DSN``: PostgreSQL (``pg``), maintained by statement-level
  triggers with transition tables, its only mechanism.
* ``MYSQL_DSN``: MySQL or MariaDB (the product is read from the server).
* neither: SQLite through the standard-library driver, on a file database so
  that independent connections exist. SQLite runs only when no server DSN is
  set, so the CI SQLite job is the only job that runs it.

Known defects are strict expected failures (``xfail(strict=True)`` with
``raises=`` the specific mismatch exception, so any other error fails the
test). Each names the TODO.md stage that fixes it; that stage removes the
marker when the test starts passing.

Randomized statements. ``test_randomized_bulk_statements`` runs a
deterministic, seed-driven, schema-aware generator (see ``BulkStatementGenerator``)
over the retained seeds ``RETAINED_SEEDS``. Replay one seed with::

    MATVIEW_BULK_SEEDS=<seed> [POSTGRES_DSN=... | MYSQL_DSN=...] \\
        python3 -m pytest -v test_native_bulk_statements.py -k randomized

``MATVIEW_BULK_STEPS`` overrides the number of statements per seed.

Structural limits (what this harness does NOT reach) are listed in
``GENERATOR_LIMITS`` and in the PR description; in short: one fact table and
one dimension table with fixed columns, the view shapes in ``VIEWS`` only, no
window, float, statistical or multi-level views, no ``COPY`` in the generator,
no ``LOAD DATA``, at most two concurrent writers, and literal-only DML.
"""

from __future__ import annotations

import collections
import io
import os
import random
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pytest

from test_matview_sync import FLOAT_OPT_IN

from matview import (
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    MaterializedViewManager,
)

LOCK_WAIT_TIMEOUT_S = 30.0


class ViewMismatch(AssertionError):
    """The materialized reader differs from the native view query."""


class DeadlockObserved(AssertionError):
    """A deadlock occurred where the contract under test requires none."""


def known_defect(reason: str, raises: type = ViewMismatch) -> pytest.MarkDecorator:
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    family: str  # "postgres" | "mysql" | "sqlite"
    dsn: str

    @property
    def id(self) -> str:
        return "pg" if self.family == "postgres" else self.family

    @property
    def dialect(self) -> Dialect:
        return {"postgres": Dialect.POSTGRES, "mysql": Dialect.MYSQL, "sqlite": Dialect.SQLITE}[self.family]


def _targets() -> List[Target]:
    out: List[Target] = []
    if os.environ.get("POSTGRES_DSN"):
        out.append(Target("postgres", os.environ["POSTGRES_DSN"]))
    if os.environ.get("MYSQL_DSN"):
        out.append(Target("mysql", os.environ["MYSQL_DSN"]))
    if not out:
        out.append(Target("sqlite", ""))
    return out


TARGETS = _targets()


def _params(
    families: Iterable[str] = ("postgres", "mysql", "sqlite"),
    xfail: Optional[Dict[str, pytest.MarkDecorator]] = None,
) -> List[Any]:
    """pytest params for the targets of this run, restricted to `families`.

    `xfail` maps a target id (pg, mysql, sqlite) to its marker."""
    fam = set(families)
    out = []
    for t in TARGETS:
        if t.family not in fam:
            continue
        marks = [xfail[t.id]] if xfail and t.id in xfail else []
        out.append(pytest.param(t, id=t.id, marks=marks))
    if not out:
        out.append(pytest.param(None, id="no-target", marks=pytest.mark.skip(reason=f"needs {sorted(fam)}")))
    return out


# ---------------------------------------------------------------------------
# Exact comparison
# ---------------------------------------------------------------------------


def _exact(v: Any) -> Tuple[str, Any]:
    if isinstance(v, Decimal):
        return ("decimal", v.as_tuple())
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        return ("float", v.hex())
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("bytes", bytes(v))
    return (type(v).__name__, v)


def _relation(cur: Any, sql: str) -> Tuple[List[Tuple[str, Any]], collections.Counter]:
    cur.execute(sql)
    cols = [(str(d[0]).lower(), d[1]) for d in cur.description]
    rows = collections.Counter(tuple(_exact(v) for v in row) for row in cur.fetchall())
    return cols, rows


# ---------------------------------------------------------------------------
# Schema and views
# ---------------------------------------------------------------------------

# t: fact table; d: dimension joined on g. g, v, w, m are nullable; the CHECK
# gives deterministic mid-statement failures.
SCHEMA = {
    "postgres": [
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
        "CREATE TABLE t (id INT PRIMARY KEY, g INT, v BIGINT, w INT, m NUMERIC(12,3), "
        "CONSTRAINT t_v_bound CHECK (v IS NULL OR v < 1000000))",
        "CREATE INDEX t_g_v ON t (g, v)",
    ],
    "mysql": [
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL) ENGINE=InnoDB",
        "CREATE TABLE t (id INT PRIMARY KEY, g INT, v BIGINT, w INT, m DECIMAL(12,3), "
        "CONSTRAINT t_v_bound CHECK (v IS NULL OR v < 1000000)) ENGINE=InnoDB",
        "CREATE INDEX t_g_v ON t (g, v)",
    ],
    "sqlite": [
        "CREATE TABLE d (g INTEGER PRIMARY KEY, name VARCHAR(20) NOT NULL)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, w INTEGER, m INTEGER, "
        "CONSTRAINT t_v_bound CHECK (v IS NULL OR v < 1000000))",
        "CREATE INDEX t_g_v ON t (g, v)",
    ],
}


def _seed_rows(n: int = 40) -> List[Tuple[int, Optional[int], Optional[int], Optional[int], Optional[str]]]:
    rows = []
    for i in range(1, n + 1):
        g = None if i % 13 == 0 else i % 6
        v = None if i % 11 == 0 else (i * 37) % 101 - 30
        w = None if i % 7 == 0 else i % 5
        m = None if i % 9 == 0 else str(Decimal((i * 1234) % 10007) / 1000 - 3)
        rows.append((i, g, v, w, m))
    return rows


SEED_D = [(g, f"n{g % 3}") for g in range(8)]  # groups 8+ and NULL have no dimension row


@dataclass(frozen=True)
class ViewDef:
    name: str
    select: str
    families: Tuple[str, ...] = ("postgres", "mysql", "sqlite")
    join: bool = False  # reads both t and d


VIEWS: List[ViewDef] = [
    # Grouped exact aggregates (BIGINT).
    ViewDef("v_agg", "SELECT g, SUM(v) AS s, COUNT(*) AS c, COUNT(v) AS cv, AVG(v) AS a FROM t GROUP BY g"),
    # Grouped exact aggregates (DECIMAL). SQLite has no exact decimal type.
    ViewDef("v_aggd", "SELECT g, SUM(m) AS s, COUNT(m) AS c FROM t GROUP BY g", ("postgres", "mysql")),
    # AVG(DECIMAL): the PostgreSQL native AVG result scale (see test_avg_decimal_new_group_scale).
    ViewDef("v_avgd", "SELECT g, AVG(m) AS a, COUNT(*) AS c FROM t GROUP BY g", ("postgres", "mysql")),
    # Group-scoped MIN/MAX with a supporting (g, v) index ...
    ViewDef("v_mm", "SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM t GROUP BY g"),
    # ... and without an index on w.
    ViewDef("v_mmw", "SELECT g, MIN(w) AS lo, MAX(w) AS hi, COUNT(*) AS c FROM t GROUP BY g"),
    ViewDef("v_ds", "SELECT DISTINCT g FROM t"),
    # Filtered single-table bag projection without the key (duplicates:
    # multiplicities matter; the hidden source key still drives maintenance).
    ViewDef("v_pr", "SELECT g, v FROM t WHERE w > 0"),
    # The same projection with the key column (not affected by that defect).
    ViewDef("v_prk", "SELECT id, g, v FROM t WHERE w > 0"),
    # Plain join (no aggregation).
    ViewDef("v_pj", "SELECT t.id, t.v, d.name FROM t JOIN d ON t.g = d.g", join=True),
    # Join aggregate grouped by a dimension attribute.
    ViewDef("v_j", "SELECT d.name, SUM(t.v) AS s, COUNT(*) AS c FROM t JOIN d ON t.g = d.g GROUP BY d.name", join=True),
    # WHERE + HAVING.
    ViewDef("v_wh", "SELECT g, SUM(v) AS s, COUNT(*) AS c FROM t WHERE v > 0 GROUP BY g HAVING COUNT(*) > 1"),
    # Aggregated expression.
    ViewDef("v_ex", "SELECT g, SUM(v * 2 + w) AS s, COUNT(*) AS c FROM t GROUP BY g"),
    # Expression group key.
    ViewDef("v_exk", "SELECT id % 7 AS k, SUM(v) AS s, COUNT(*) AS c FROM t GROUP BY id % 7"),
]


# Views left out of the shared list on a target because a dedicated strict
# expected failure covers their known defect there (so the rest of the harness
# keeps testing every other shape on that target). Remove an entry together
# with the marker of the test named next to it.
KNOWN_DEFECT_EXCLUDED_VIEWS: Dict[str, Tuple[str, ...]] = {
}


def _sql_lit(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        if re.fullmatch(r"-?\d+(\.\d+)?", v):
            return v
        return "'" + v.replace("'", "''") + "'"
    return str(int(v))


# ---------------------------------------------------------------------------
# Environment: an isolated namespace with registered views
# ---------------------------------------------------------------------------


class Env:
    def __init__(self, target: Target, views: Optional[Sequence[ViewDef]] = None, schema_sql: Optional[List[str]] = None,
                 seed: bool = True, ignore_limitations: Optional[bool] = None) -> None:
        self.target = target
        self.family = target.family
        self.ns = f"mvb_{uuid.uuid4().hex[:12]}"
        self._tmpdir: Optional[str] = None
        self._extra_conns: List[Any] = []
        self.plans: Dict[str, Any] = {}
        self.limitation_codes: Dict[str, List[str]] = {}
        if self.family == "postgres":
            admin = self._raw_connect(None)
            with admin.cursor() as c:
                c.execute(f'CREATE SCHEMA "{self.ns}"')
            admin.commit()
            admin.close()
        elif self.family == "mysql":
            admin = self._raw_connect(None)
            with admin.cursor() as c:
                c.execute(f"CREATE DATABASE `{self.ns}`")
            admin.commit()
            admin.close()
        else:
            self._tmpdir = tempfile.mkdtemp(prefix="mvb_sqlite_")
        self.conn = self.connect()
        cur = self.conn.cursor()
        if self.family == "sqlite":
            cur.execute("PRAGMA journal_mode=WAL")
            cur.fetchall()
        self.product, self.version = self._server()
        for ddl in schema_sql if schema_sql is not None else SCHEMA[self.family]:
            cur.execute(ddl)
        self.conn.commit()
        if seed:
            self.seed()
        self.manager = MaterializedViewManager(self.conn, dialect=target.dialect)
        if views is None:
            views = [vd for vd in VIEWS if self.family in vd.families
                     and vd.name not in KNOWN_DEFECT_EXCLUDED_VIEWS.get(target.id, ())
                     and vd.name not in KNOWN_DEFECT_EXCLUDED_VIEWS.get(self.product, ())]
        self.views = list(views)
        for vd in self.views:
            self.register(vd, ignore_limitations)

    # -- connections -------------------------------------------------------
    def _raw_connect(self, database: Optional[str]) -> Any:
        if self.family == "postgres":
            import psycopg2

            conn = psycopg2.connect(self.target.dsn)
            if database is not None:
                with conn.cursor() as c:
                    c.execute(f'SET search_path TO "{database}"')
                conn.commit()
            return conn
        if self.family == "mysql":
            import pymysql

            p = urlparse(self.target.dsn)
            return pymysql.connect(
                host=p.hostname or "127.0.0.1", port=p.port or 3306, user=p.username or "root",
                password=p.password or "", database=database or (p.path.lstrip("/") or None), autocommit=False,
            )
        assert self._tmpdir is not None
        conn = sqlite3.connect(os.path.join(self._tmpdir, "db.sqlite3"), timeout=30)
        MaterializedViewManager.configure_sqlite_connection(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def connect(self) -> Any:
        """A new, independent connection to this namespace."""
        conn = self._raw_connect(self.ns)
        self._extra_conns.append(conn)
        return conn

    def _server(self) -> Tuple[str, Tuple[int, ...]]:
        cur = self.conn.cursor()
        if self.family == "postgres":
            cur.execute("SHOW server_version_num")
            n = int(cur.fetchone()[0])
            self.conn.commit()
            return "postgres", (n // 10000, n % 10000)
        if self.family == "mysql":
            cur.execute("SELECT VERSION()")
            text = str(cur.fetchone()[0])
            self.conn.commit()
            nums = tuple(int(x) for x in re.findall(r"\d+", text)[:2])
            return ("mariadb" if "mariadb" in text.lower() else "mysql"), nums
        return "sqlite", tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:2])

    @property
    def is_mariadb(self) -> bool:
        return self.product == "mariadb"

    # -- data and registration ---------------------------------------------
    def seed(self) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO d (g, name) VALUES " + ", ".join(f"({g}, '{n}')" for g, n in SEED_D))
        rows = _seed_rows()
        cur.execute("INSERT INTO t (id, g, v, w, m) VALUES " + ", ".join(
            "(" + ", ".join(_sql_lit(x) for x in r) + ")" for r in rows))
        self.conn.commit()

    def register(
        self, vd: ViewDef, ignore_limitations: Optional[bool] = None,
        accept_limitations: Optional[Sequence[str]] = None, populate: bool = True,
    ) -> Any:
        # MySQL/MariaDB registration requires acknowledging the documented
        # MYSQL_NO_TRUNCATE_TRIGGER limitation; that is asserted, not silenced.
        # Floating-point order-sensitive aggregates need an explicit per-code
        # acceptance (accept_limitations); those codes are asserted too.
        ignore = (self.family == "mysql") if ignore_limitations is None else ignore_limitations
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = self.manager.register_view(
                f"CREATE VIEW {vd.name} AS {vd.select};", ignore_limitations=ignore,
                accept_limitations=accept_limitations, populate=populate,
            )
        self.conn.commit()
        codes = sorted({lim.code for lim in plan.limitations})
        self.limitation_codes[vd.name] = codes
        if accept_limitations:
            assert set(codes) <= {"MYSQL_NO_TRUNCATE_TRIGGER", *accept_limitations}, (vd.name, codes)
            codes = [c for c in codes if c not in accept_limitations]
        if self.family == "mysql" and ignore_limitations is None:
            assert codes == ["MYSQL_NO_TRUNCATE_TRIGGER"], (vd.name, codes)
            assert any(
                issubclass(w.category, EngineLimitationWarning) and "MYSQL_NO_TRUNCATE_TRIGGER" in str(w.message)
                for w in caught
            ), [str(w.message) for w in caught]
        self.plans[vd.name] = plan
        return plan

    # -- execution -----------------------------------------------------------
    def execute(self, sql: str, conn: Any = None) -> int:
        c = (conn or self.conn).cursor()
        c.execute(sql)
        return c.rowcount

    def commit(self, conn: Any = None) -> None:
        (conn or self.conn).commit()

    def rollback(self, conn: Any = None) -> None:
        (conn or self.conn).rollback()

    def snapshot(self, conn: Any = None) -> Dict[str, Any]:
        cur = (conn or self.conn).cursor()
        return {vd.name: _relation(cur, vd.select)[1] for vd in self.views}

    def check(self, label: str, conn: Any = None, views: Optional[Sequence[str]] = None,
              end_tx: bool = True) -> None:
        """Compare every view (or `views`) with its native query.

        `end_tx=False` reads inside the caller's open transaction on `conn`."""
        c = conn or self.conn
        cur = c.cursor()
        problems = []
        for vd in self.views:
            if views is not None and vd.name not in views:
                continue
            mv_cols, mv_rows = _relation(cur, f"SELECT * FROM {vd.name}")
            or_cols, or_rows = _relation(cur, vd.select)
            if mv_cols != or_cols:
                problems.append(f"{vd.name}: columns {mv_cols} != oracle {or_cols}")
            if mv_rows != or_rows:
                missing = or_rows - mv_rows
                extra = mv_rows - or_rows
                problems.append(f"{vd.name}: missing {sorted(missing.items(), key=repr)[:6]} "
                                f"extra {sorted(extra.items(), key=repr)[:6]}")
        if end_tx:
            c.commit()
        if problems:
            raise ViewMismatch(f"[{self.target.id} {self.product} {self.version}] after {label}:\n  " + "\n  ".join(problems))

    def reset(self) -> None:
        """Return to the seed state through plain DML (itself maintained and
        checked), closing every connection a previous test opened."""
        for conn in self._extra_conns:
            if conn is self.conn:
                continue
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
        self._extra_conns = [self.conn]
        self.conn.rollback()
        self.execute("DELETE FROM t")
        self.execute("DELETE FROM d")
        self.commit()
        self.seed()
        self.check("reset to the seed state")

    def close(self) -> None:
        for conn in self._extra_conns:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        try:
            self.manager.close()
        except Exception:
            pass
        if self.family in ("postgres", "mysql"):
            admin = self._raw_connect(None)
            with admin.cursor() as c:
                if self.family == "postgres":
                    c.execute(f'DROP SCHEMA IF EXISTS "{self.ns}" CASCADE')
                else:
                    c.execute(f"DROP DATABASE IF EXISTS `{self.ns}`")
            admin.commit()
            admin.close()
        elif self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)


@pytest.fixture
def make_env():
    envs: List[Env] = []

    def factory(target: Target, **kw: Any) -> Env:
        env = Env(target, **kw)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()


_SHARED: Dict[str, Env] = {}


@pytest.fixture(scope="module", autouse=True)
def _close_shared_envs():
    yield
    while _SHARED:
        _SHARED.popitem()[1].close()


@pytest.fixture
def shared_env():
    """The default environment (every view of the target registered), shared
    by the tests of one target to avoid re-registering the views per test and
    reset to the seed state before each use. When the reset itself does not
    produce an exact view (a previous test left a known-defective state), the
    environment is rebuilt from scratch."""

    def get(target: Target) -> Env:
        env = _SHARED.get(target.id)
        if env is not None:
            try:
                env.reset()
                return env
            except Exception:
                del _SHARED[target.id]
                env.close()
        env = Env(target)
        _SHARED[target.id] = env
        return env

    return get


def _errors(family: str) -> Tuple[type, ...]:
    if family == "postgres":
        import psycopg2

        return (psycopg2.Error,)
    if family == "mysql":
        import pymysql

        return (pymysql.err.MySQLError,)
    return (sqlite3.Error,)


def _integrity_errors(family: str) -> Tuple[type, ...]:
    """Constraint violations (duplicate key, CHECK): the only errors a
    statement of the deterministic catalog may legitimately raise when it is
    re-applied to a changed state."""
    if family == "postgres":
        import psycopg2

        return (psycopg2.IntegrityError,)
    if family == "mysql":
        import pymysql

        return (pymysql.err.IntegrityError,)
    return (sqlite3.IntegrityError,)


# ---------------------------------------------------------------------------
# Deterministic statement catalog
# ---------------------------------------------------------------------------

Fn = Callable[[Env], None]


def _x(sql: str) -> Fn:
    return lambda env: (env.execute(sql), None)[1]


def _by_product(**variants: str) -> Fn:
    """Pick SQL by product (postgres/mysql/mariadb/sqlite), falling back to the family."""

    def run(env: Env) -> None:
        sql = variants.get(env.product) or variants.get(env.family)
        assert sql is not None, (env.product, sorted(variants))
        env.execute(sql)

    return run


@dataclass(frozen=True)
class Stmt:
    name: str
    run: Fn
    families: Tuple[str, ...] = ("postgres", "mysql", "sqlite")
    applies: Callable[[Env], bool] = field(default=lambda env: True)


def _pg_copy(env: Env) -> None:
    data = "201\t1\t5\t\\N\t1.250\n202\t\\N\t-7\t2\t\\N\n203\t9\t5\t1\t-0.001\n204\t1\t5\t0\t1.250\n"
    cur = env.conn.cursor()
    cur.copy_expert("COPY t (id, g, v, w, m) FROM STDIN", io.StringIO(data))


UPSERT_VALUES = "(1, 3, 77, 1, 2.500), (2, NULL, NULL, 4, NULL), (300, 1, 9, 9, 0.125), (301, 1, 9, 9, 0.125)"

STATEMENTS: List[Stmt] = [
    Stmt("insert_values_multi_row", _x(
        "INSERT INTO t (id, g, v, w, m) VALUES (101, 1, 5, NULL, 1.250), (102, 1, 7, 3, -2.500), "
        "(103, NULL, NULL, 1, 0), (104, 9, 100, 2, 3), (105, 1, 5, 0, 1.250), (106, 1, -40, 4, 0.001)")),
    Stmt("insert_select_self", _x(
        "INSERT INTO t (id, g, v, w, m) SELECT id + 1000, (id + 1) % 8, v, w, m FROM t WHERE id % 3 = 0")),
    Stmt("insert_select_into_new_groups", _x(
        "INSERT INTO t (id, g, v, w, m) SELECT id + 2000, 20 + id % 3, v - 1, w, m FROM t WHERE v IS NOT NULL")),
    Stmt("update_values_many_groups", _x("UPDATE t SET v = v + 3, m = m * 2, w = w - 1 WHERE id % 2 = 0")),
    Stmt("update_move_between_groups", _x(
        "UPDATE t SET g = CASE WHEN g IS NULL THEN 2 ELSE (g + 1) % 7 END WHERE id % 3 <> 0")),
    Stmt("update_group_key_to_null", _x("UPDATE t SET g = NULL, w = NULL WHERE id % 4 = 1")),
    Stmt("update_primary_key", _x("UPDATE t SET id = id + 500 WHERE id > 25")),
    Stmt("update_flip_extremes", _x("UPDATE t SET v = -v, w = 10 - w WHERE id % 5 <> 2")),
    Stmt("update_noop", _x("UPDATE t SET v = v, g = g")),
    Stmt("update_all_into_one_group", _x("UPDATE t SET g = 3")),
    Stmt("delete_many", _x("DELETE FROM t WHERE id % 3 = 1")),
    Stmt("delete_group_extremes", _x("DELETE FROM t WHERE v <= 0 OR w = 4")),
    Stmt("delete_all", _x("DELETE FROM t")),
    Stmt("dimension_insert_multi_row", _x("INSERT INTO d (g, name) VALUES (8, 'n0'), (9, 'n2'), (20, 'n1')")),
    Stmt("dimension_update_join_group", _x("UPDATE d SET name = CASE WHEN g % 2 = 0 THEN 'n9' ELSE name END")),
    Stmt("dimension_update_join_key", _x("UPDATE d SET g = g + 100 WHERE g IN (1, 2)")),
    Stmt("dimension_delete_many", _x("DELETE FROM d WHERE g % 2 = 1")),
    # Upserts.
    Stmt("upsert_update", _by_product(
        postgres=f"INSERT INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES} "
        "ON CONFLICT (id) DO UPDATE SET g = EXCLUDED.g, v = t.v + EXCLUDED.v, m = EXCLUDED.m",
        sqlite=f"INSERT INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES} "
        "ON CONFLICT (id) DO UPDATE SET g = excluded.g, v = t.v + excluded.v, m = excluded.m",
        mysql=f"INSERT INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES} AS new "
        "ON DUPLICATE KEY UPDATE g = new.g, v = t.v + new.v, m = new.m",
        mariadb=f"INSERT INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES} "
        "ON DUPLICATE KEY UPDATE g = VALUES(g), v = v + VALUES(v), m = VALUES(m)",
    )),
    Stmt("upsert_ignore", _by_product(
        postgres=f"INSERT INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES} ON CONFLICT DO NOTHING",
        sqlite=f"INSERT OR IGNORE INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES}",
        mysql=f"INSERT IGNORE INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES}",
    )),
    Stmt("replace", _by_product(
        sqlite=f"INSERT OR REPLACE INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES}",
        mysql=f"REPLACE INTO t (id, g, v, w, m) VALUES {UPSERT_VALUES}",
    ), ("mysql", "sqlite")),
    Stmt("merge", _x(
        "MERGE INTO t USING (VALUES (1, 4, NULL::bigint), (2, 2, 50::bigint), (3, NULL, 7::bigint), "
        "(400, 1, 3::bigint), (401, 6, -3::bigint)) AS s(id, g, v) ON t.id = s.id "
        "WHEN MATCHED AND s.v IS NULL THEN DELETE "
        "WHEN MATCHED THEN UPDATE SET g = s.g, v = s.v "
        "WHEN NOT MATCHED THEN INSERT (id, g, v, w, m) VALUES (s.id, s.g, s.v, 1, 1.000)"),
        ("postgres",), lambda env: env.version >= (15, 0)),
    Stmt("copy_from_stdin", _pg_copy, ("postgres",)),
    # Statements that join another table but modify only t.
    Stmt("update_from_join", _by_product(
        postgres="UPDATE t SET v = t.v + d.g, g = d.g + 1 FROM d WHERE t.g = d.g AND d.name = 'n1'",
        sqlite="UPDATE t SET v = t.v + d.g, g = d.g + 1 FROM d WHERE t.g = d.g AND d.name = 'n1'",
        mysql="UPDATE t JOIN d ON t.g = d.g SET t.v = t.v + d.g, t.g = d.g + 1 WHERE d.name = 'n1'",
    ), applies=lambda env: env.family != "sqlite" or env.version >= (3, 33)),
    Stmt("delete_using_join", _by_product(
        postgres="DELETE FROM t USING d WHERE t.g = d.g AND d.name = 'n2'",
        sqlite="DELETE FROM t WHERE g IN (SELECT g FROM d WHERE name = 'n2')",
        mysql="DELETE t FROM t JOIN d ON t.g = d.g WHERE d.name = 'n2'",
    )),
    # MySQL/MariaDB multi-table statements that modify both sources.
    Stmt("multi_table_update_both_sources", _x(
        "UPDATE t JOIN d ON t.g = d.g SET t.v = t.v + 1, d.name = CONCAT(d.name, 'x') WHERE d.g < 3"), ("mysql",)),
    Stmt("multi_table_delete_both_sources", _x(
        "DELETE t, d FROM t JOIN d ON t.g = d.g WHERE d.name = 'n1'"), ("mysql",)),
    # PostgreSQL writable CTEs that modify one source.
    Stmt("writable_cte_move_rows", _x(
        "WITH moved AS (DELETE FROM t WHERE id % 4 = 0 RETURNING id, g, v, w, m) "
        "INSERT INTO t (id, g, v, w, m) SELECT id + 3000, g, v + 1, w, m FROM moved"), ("postgres",)),
    Stmt("writable_cte_update_and_delete", _x(
        "WITH u AS (UPDATE t SET v = v + 10, g = 5 WHERE id % 2 = 0 RETURNING id) "
        "DELETE FROM t WHERE id % 2 = 1 AND id % 5 = 0"), ("postgres",)),
    Stmt("writable_cte_two_inserts_same_table", _x(
        "WITH a AS (INSERT INTO t (id, g, v) VALUES (501, 1, 1), (502, 2, 2) RETURNING id) "
        "INSERT INTO t (id, g, v) SELECT id + 10, 1, 3 FROM a"), ("postgres",)),
    # Zero-row statements.
    Stmt("zero_row_insert_select", _x("INSERT INTO t (id, g, v, w, m) SELECT id, g, v, w, m FROM t WHERE 1 = 0")),
    Stmt("zero_row_update", _x("UPDATE t SET v = v + 1 WHERE id < 0")),
    Stmt("zero_row_delete", _x("DELETE FROM t WHERE id < 0")),
    Stmt("zero_row_dimension_delete", _x("DELETE FROM d WHERE g < 0")),
    Stmt("zero_row_merge", _x(
        "MERGE INTO t USING (SELECT id, g, v FROM t WHERE 1 = 0) AS s ON t.id = s.id "
        "WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT (id, g, v) VALUES (s.id, s.g, s.v)"),
        ("postgres",), lambda env: env.version >= (15, 0)),
]


def _stmt_params() -> List[Any]:
    out = []
    for t in TARGETS:
        for s in STATEMENTS:
            if t.family in s.families:
                out.append(pytest.param(t, s, id=f"{t.id}-{s.name}"))
    return out


@pytest.mark.parametrize("target,stmt", _stmt_params())
def test_bulk_statement(shared_env, target: Target, stmt: Stmt) -> None:
    """Each statement kind, alone, on all view shapes, committed; then a
    second application on the changed state."""
    env = shared_env(target)
    if not stmt.applies(env):
        pytest.skip(f"{stmt.name} not available on {env.product} {env.version}")
    env.check("registration")
    stmt.run(env)
    env.check(f"{stmt.name} (in transaction)", end_tx=False)
    env.commit()
    env.check(stmt.name)
    try:
        stmt.run(env)
    except _integrity_errors(env.family):
        env.rollback()  # e.g. a re-insert of the same keys; the state must be unchanged
    else:
        env.commit()
    env.check(f"{stmt.name} applied twice")


@pytest.mark.parametrize("target", _params())
def test_statement_sequence_on_one_state(shared_env, target: Target) -> None:
    """All statement kinds back to back on one evolving state (interactions)."""
    env = shared_env(target)
    for stmt in STATEMENTS:
        if target.family not in stmt.families or not stmt.applies(env):
            continue
        try:
            stmt.run(env)
        except _integrity_errors(env.family):
            env.rollback()
        else:
            env.commit()
        env.check(stmt.name)


# ---------------------------------------------------------------------------
# Failure, rollback, savepoints
# ---------------------------------------------------------------------------

FAILING: List[Tuple[str, str]] = [
    # The duplicate key is the LAST row, so earlier rows have been maintained.
    ("insert_values_duplicate_last_row",
     "INSERT INTO t (id, g, v) VALUES (601, 1, 1), (602, 2, 2), (603, NULL, 3), (1, 1, 1)"),
    ("insert_select_check_violation_midway",
     "INSERT INTO t (id, g, v) SELECT id + 700, g, CASE WHEN id = 30 THEN 5000000 ELSE v END FROM t"),
    ("update_check_violation_midway", "UPDATE t SET v = CASE WHEN id = 35 THEN 9999999 ELSE v + 1 END, g = 2"),
    ("update_primary_key_collision", "UPDATE t SET id = id + 1 WHERE id BETWEEN 10 AND 20"),
]


@pytest.mark.parametrize("target", _params())
@pytest.mark.parametrize("name,sql", FAILING, ids=[f[0] for f in FAILING])
def test_failing_statement_restores_views(shared_env, target: Target, name: str, sql: str) -> None:
    env = shared_env(target)
    before = env.snapshot()
    env.conn.commit()
    if env.family == "postgres":
        env.execute("SAVEPOINT before_fail")
    with pytest.raises(_errors(env.family)):
        env.execute(sql)
    if env.family == "postgres":
        env.execute("ROLLBACK TO SAVEPOINT before_fail")
    # MySQL/MariaDB/SQLite roll back only the failed statement; PostgreSQL
    # rolled back to the savepoint. The open transaction must be exact.
    env.check(f"{name} (statement rolled back, transaction open)", end_tx=False)
    env.commit()
    env.check(name)
    assert env.snapshot() == before, "the failed statement changed the base tables"
    env.commit()


@pytest.mark.parametrize("target", _params())
def test_transaction_and_savepoint_rollback(shared_env, target: Target) -> None:
    env = shared_env(target)
    before = env.snapshot()
    env.conn.commit()
    if env.family == "sqlite":
        env.execute("BEGIN")
    env.execute("INSERT INTO t (id, g, v, w, m) SELECT id + 800, (id * 7) % 9, v, w, m FROM t")
    env.check("bulk insert in transaction", end_tx=False)
    env.execute("SAVEPOINT s1")
    env.execute("UPDATE t SET g = (g + 2) % 9, v = v * 2 WHERE id % 2 = 0")
    env.execute("DELETE FROM t WHERE id % 5 = 0")
    env.check("update+delete after savepoint", end_tx=False)
    env.execute("SAVEPOINT s2")
    env.execute("DELETE FROM t")
    env.check("delete all after nested savepoint", end_tx=False)
    env.execute("ROLLBACK TO SAVEPOINT s2")
    env.check("rollback to nested savepoint", end_tx=False)
    env.execute("ROLLBACK TO SAVEPOINT s1")
    env.check("rollback to savepoint", end_tx=False)
    env.execute("UPDATE d SET name = 'zz' WHERE g < 4")
    env.check("dimension update after savepoint rollback", end_tx=False)
    env.rollback()
    env.check("full rollback")
    assert env.snapshot() == before
    env.commit()


# ---------------------------------------------------------------------------
# TRUNCATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", _params(("postgres",)))
def test_truncate(shared_env, target: Target) -> None:
    env = shared_env(target)
    env.execute("TRUNCATE t")
    env.check("TRUNCATE t (in transaction)", end_tx=False)
    env.commit()
    env.check("TRUNCATE t")
    env.execute("INSERT INTO t (id, g, v) VALUES (1, 1, 1), (2, 1, 2)")
    env.execute("TRUNCATE d")
    env.commit()
    env.check("TRUNCATE d")
    env.execute("SAVEPOINT s")
    env.execute("TRUNCATE t")
    env.execute("ROLLBACK TO SAVEPOINT s")
    env.commit()
    env.check("TRUNCATE rolled back to savepoint")


@pytest.mark.parametrize("target", _params(("mysql",)))
def test_mysql_truncate_is_a_documented_divergence(make_env, target: Target) -> None:
    """MySQL/MariaDB TRUNCATE bypasses triggers. Registration refuses by default
    (MYSQL_NO_TRUNCATE_TRIGGER) and, when acknowledged, the view diverges.
    This asserts the documented consequence (README / planner limitation);
    the fix is tracked in TODO "Close Unsafe DML Paths That Bypass
    Trigger-Based Maintenance"."""
    env = make_env(target, views=[VIEWS[0]])
    with pytest.raises(EngineLimitationError, match="MYSQL_NO_TRUNCATE_TRIGGER"):
        env.manager.register_view("CREATE VIEW v_default AS SELECT g, COUNT(*) AS c FROM t GROUP BY g;")
    env.rollback()
    env.execute("TRUNCATE t")
    with pytest.raises(ViewMismatch):
        env.check("TRUNCATE t (acknowledged limitation)")


# ---------------------------------------------------------------------------
# FK cascades
# ---------------------------------------------------------------------------

FK_SCHEMA = {
    "postgres": [
        "CREATE TABLE p (pid INT PRIMARY KEY)",
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL, pid INT REFERENCES p (pid) "
        "ON DELETE CASCADE ON UPDATE CASCADE)",
        "CREATE TABLE t (id INT PRIMARY KEY, g INT, v BIGINT, w INT, m NUMERIC(12,3), pid INT REFERENCES p (pid) "
        "ON DELETE CASCADE ON UPDATE CASCADE)",
        "CREATE INDEX t_g_v ON t (g, v)",
        "CREATE INDEX t_pid ON t (pid)",
        "CREATE INDEX d_pid ON d (pid)",
    ],
    "sqlite": [
        "CREATE TABLE p (pid INTEGER PRIMARY KEY)",
        "CREATE TABLE d (g INTEGER PRIMARY KEY, name VARCHAR(20) NOT NULL, pid INTEGER REFERENCES p (pid) "
        "ON DELETE CASCADE ON UPDATE CASCADE)",
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, w INTEGER, m INTEGER, "
        "pid INTEGER REFERENCES p (pid) ON DELETE CASCADE ON UPDATE CASCADE)",
        "CREATE INDEX t_g_v ON t (g, v)",
        "CREATE INDEX t_pid ON t (pid)",
        "CREATE INDEX d_pid ON d (pid)",
    ],
    "mysql": [
        "CREATE TABLE p (pid INT PRIMARY KEY) ENGINE=InnoDB",
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL) ENGINE=InnoDB",
        "CREATE TABLE t (id INT PRIMARY KEY, g INT, v BIGINT, w INT, m DECIMAL(12,3), pid INT, "
        "FOREIGN KEY (pid) REFERENCES p (pid) ON DELETE CASCADE) ENGINE=InnoDB",
    ],
}


def _fk_env(make_env, target: Target, views: Sequence[ViewDef]) -> Env:
    env = make_env(target, views=[], schema_sql=FK_SCHEMA[target.family], seed=False)
    env.execute("INSERT INTO p (pid) VALUES " + ", ".join(f"({i})" for i in range(1, 6)))
    if target.family == "mysql":
        env.execute("INSERT INTO d (g, name) VALUES " + ", ".join(f"({g}, '{n}')" for g, n in SEED_D))
    else:
        env.execute("INSERT INTO d (g, name, pid) VALUES " + ", ".join(f"({g}, '{n}', {g % 5 + 1})" for g, n in SEED_D))
    env.execute("INSERT INTO t (id, g, v, w, m, pid) VALUES " + ", ".join(
        "(" + ", ".join(_sql_lit(x) for x in r) + f", {r[0] % 5 + 1})" for r in _seed_rows()))
    env.commit()
    env.views = list(views)
    for vd in views:
        env.register(vd)
    return env


SINGLE_SOURCE_VIEWS = [vd for vd in VIEWS if not vd.join]
JOIN_VIEWS = [vd for vd in VIEWS if vd.join]


@pytest.mark.parametrize("target", _params(("postgres", "sqlite")))
def test_fk_cascade_into_single_source_views(make_env, target: Target) -> None:
    env = _fk_env(make_env, target, [vd for vd in SINGLE_SOURCE_VIEWS if target.family in vd.families])
    env.execute("UPDATE p SET pid = pid + 10 WHERE pid IN (1, 2)")  # ON UPDATE CASCADE (no view column)
    env.commit()
    env.check("ON UPDATE CASCADE")
    env.execute("DELETE FROM p WHERE pid IN (3, 11)")
    env.commit()
    env.check("ON DELETE CASCADE of many rows")
    env.execute("DELETE FROM p")
    env.commit()
    env.check("ON DELETE CASCADE of all rows")


@pytest.mark.parametrize("target", _params(("postgres", "sqlite")))
def test_fk_cascade_into_both_join_sources(make_env, target: Target) -> None:
    env = _fk_env(make_env, target, JOIN_VIEWS)
    env.execute("DELETE FROM p WHERE pid = 2")
    env.commit()
    env.check("ON DELETE CASCADE into t and d")


@pytest.mark.parametrize("target", _params(("mysql",)))
def test_mysql_fk_cascade_is_refused_by_default(make_env, target: Target) -> None:
    """InnoDB cascades do not fire triggers; registration refuses (default)."""
    env = _fk_env(make_env, target, [])
    with pytest.raises(EngineLimitationError, match="MYSQL_FK_CASCADE_NO_TRIGGER"):
        env.manager.register_view("CREATE VIEW v_fk AS SELECT g, COUNT(*) AS c FROM t GROUP BY g;")
    env.rollback()


# ---------------------------------------------------------------------------
# PostgreSQL: one query modifying both sources of a join view
# ---------------------------------------------------------------------------

CO_MODIFYING = [
    ("cte_insert_both", "WITH a AS (INSERT INTO d (g, name) VALUES (30, 'n1'), (31, 'n2') RETURNING g) "
     "INSERT INTO t (id, g, v) VALUES (900, 30, 5), (901, 31, 7), (902, 1, 11)"),
    ("cte_delete_both", "WITH a AS (DELETE FROM d WHERE g = 1 RETURNING g) DELETE FROM t WHERE g IN (1, 2)"),
    ("cte_update_both", "WITH a AS (UPDATE d SET name = 'n7' WHERE g IN (2, 3) RETURNING g) "
     "UPDATE t SET v = v + 100 WHERE g IN (2, 4)"),
]


@pytest.mark.parametrize("target", _params(("postgres",)))
@pytest.mark.parametrize("name,sql", CO_MODIFYING, ids=[c[0] for c in CO_MODIFYING])
def test_pg_writable_cte_both_sources_non_algebraic_views(shared_env, target: Target, name: str, sql: str) -> None:
    """Single-source views and the full-recompute plain join stay exact."""
    env = shared_env(target)
    env.execute(sql)
    env.commit()
    env.check(name, views=[vd.name for vd in VIEWS if vd.name != "v_j"])


@pytest.mark.parametrize("target", _params(("postgres",)))
@pytest.mark.parametrize("name,sql", CO_MODIFYING, ids=[c[0] for c in CO_MODIFYING])
def test_pg_writable_cte_both_sources_join_aggregate(make_env, target: Target, name: str, sql: str) -> None:
    env = make_env(target, views=[vd for vd in VIEWS if vd.name == "v_j"])
    env.execute(sql)
    env.commit()
    env.check(name)


# ---------------------------------------------------------------------------
# Other divergences found while building this harness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", _params(("postgres", "mysql")))
def test_avg_decimal_new_group_scale(make_env, target: Target) -> None:
    """A group created by an INSERT must have the native AVG(NUMERIC) result
    scale (PostgreSQL row triggers stored `1.0 * NEW.m`; statement-level
    maintenance recomputes AVG(NUMERIC) natively)."""
    env = make_env(target, views=[vd for vd in VIEWS if vd.name == "v_avgd"], seed=False)
    env.execute("INSERT INTO t (id, g, m) VALUES (1, 9, 3.000)")
    env.commit()
    env.check("INSERT creating a group")
    env.execute("INSERT INTO t (id, g, m) VALUES (2, 9, 4.500), (3, 10, 0.001)")
    env.commit()
    env.check("multi-row INSERT creating a group")


_E2 = ("A single-table projection without its source primary key must still maintain hidden-key "
       "identity across primary-key-only updates. TODO.md 'Track Source-Key Changes in Projection Views Without the Key Column'")


BACKFILL_SCHEMA = {
    "postgres": ["CREATE TABLE bt (id INT PRIMARY KEY, g INT NOT NULL, x INT)"],
    "mysql": ["CREATE TABLE bt (id INT PRIMARY KEY, g INT NOT NULL, x INT) ENGINE=InnoDB"],
    "sqlite": ["CREATE TABLE bt (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, x INTEGER)"],
}
# One backfill batch of 20 is folded (ids 1..20, cursor 20) when the writes
# run; the unfolded window is 21..28 (max_pk 28): one row per group, which a
# trigger-side recompute must not read. Each group retracts one extreme by a
# delete or by a pk move out of the window (the old pk is folded, the new one
# beyond max_pk), then its unfolded row is deleted (the trigger skips that
# delete and no later chunk touches the group).
# group: (folded values, retracted value, unfolded value, how)
_BACKFILL_GROUPS = {
    1: ([12, 14], 12, -50, "delete"),  # MIN; the cursor row (id 20, value 13) becomes the MIN
    2: ([12, 13], 12, -50, "move"),  # MIN
    3: ([99, 20, 21], 99, 500, "delete"),  # MAX
    4: ([99, 20], 99, 500, "move"),  # MAX
    5: ([1, 2], 1, 1024, "delete"),  # BIT_OR
    6: ([1, 2], 1, 1024, "move"),  # BIT_OR
    7: ([3, 6], 3, 1, "delete"),  # BIT_AND
    8: ([6, 3], 6, 1, "move"),  # BIT_AND
}


def _backfill_rows_and_writes() -> Tuple[List[Tuple[int, int, int]], List[str]]:
    rows: List[Tuple[int, int, int]] = []
    writes: List[str] = []
    next_id = iter(range(1, 20))
    for g, (folded, retracted, _, how) in _BACKFILL_GROUPS.items():
        for x in folded:
            i = next(next_id)
            rows.append((i, g, x))
            if x == retracted and not any(w.endswith(f"id = {i}") for w in writes):
                writes.append(f"DELETE FROM bt WHERE id = {i}" if how == "delete"
                              else f"UPDATE bt SET id = {900 + g} WHERE id = {i}")
    for i in next_id:
        rows.append((i, 9, i))  # filler up to id 19
    rows.append((20, 1, 13))  # the cursor row
    # 21 = cursor + 1 (group 1, MIN), 28 = max_pk (group 4, MAX): both bounds
    # matter on every engine (SQLite has no BIT_AND/BIT_OR).
    for i, g in enumerate((1, 2, 3, 5, 6, 7, 8, 4), start=21):
        rows.append((i, g, _BACKFILL_GROUPS[g][2]))
    writes.append("DELETE FROM bt WHERE id BETWEEN 21 AND 28")
    return rows, writes


@pytest.mark.parametrize("target", _params())
def test_recompute_during_backfill_ignores_rows_not_yet_folded(make_env, target: Target) -> None:
    """While a pk-range backfill is active the triggers skip source rows in
    the unfolded window, which the backfill folds later. A native recompute
    by a trigger (MIN/MAX/BIT_AND/BIT_OR after a retraction, AVG where it is
    recomputed) must skip them too: otherwise it stores a value that includes
    such a row, and when the row is deleted before its chunk runs (the
    trigger skips that delete) no chunk touches the group again and the
    stale value stays (CI failure of
    test_pgmy_exact_float_aggregates.py::NativeConcurrentWriters on PG 15).
    The rows at the cursor (folded), at cursor + 1 and at max_pk (unfolded)
    check both window bounds.

    Deterministic: the writes run on an independent connection between two
    committed backfill batches (from the progress callback)."""
    env = make_env(target, views=[], seed=False, schema_sql=BACKFILL_SCHEMA[target.family])
    rows, writes = _backfill_rows_and_writes()
    env.execute("INSERT INTO bt (id, g, x) VALUES " + ", ".join(f"({i}, {g}, {x})" for i, g, x in rows))
    env.commit()
    bits = "" if target.family == "sqlite" else ", BIT_AND(x) AS ba, BIT_OR(x) AS bo"  # no SQLite BIT_AND
    vd = ViewDef("v_bt", f"SELECT g, COUNT(*) AS n, AVG(x) AS a, MIN(x) AS mn, MAX(x) AS mx{bits} FROM bt GROUP BY g")
    env.views = [vd]
    plan = env.register(vd, populate=False)
    writer = env.connect()
    done: List[int] = []

    def on_progress(batch: int, _rows: int) -> None:
        if batch != 1:
            return
        for sql in writes:
            env.execute(sql, conn=writer)
            env.commit(conn=writer)
        done.append(batch)

    env.manager.backfill_incrementally(plan, batch_size=20, on_progress=on_progress, snapshot_isolation=False)
    assert done == [1]
    env.check("backfill after recomputes of groups with unfolded rows")


# Writes of group 10 while the watermark does not cover any folded row yet:
# its only claimed row (id 29, the smallest value) is deleted after its MIN and
# MAX were retracted among rows outside the claimed range.
_UNFOLDED_GROUP_WRITES = (
    "INSERT INTO bt (id, g, x) VALUES (1001, 10, 100), (1002, 10, 5)",
    "DELETE FROM bt WHERE id = 1002",  # MIN retracted: recomputed
    "UPDATE bt SET x = 50 WHERE id = 1001",  # MAX retracted: recomputed
    "DELETE FROM bt WHERE id = 29",  # claimed and unfolded: skipped by the trigger
    # Net row-count changes that do not cancel out: a trigger that applied
    # writes it must skip would double-count them once the backfill folds.
    "INSERT INTO bt (id, g, x) VALUES (1003, 10, 60), (1004, 9, 1)",
)


@pytest.mark.parametrize("phase", ["unclaimed", "claimed"])
@pytest.mark.parametrize("target", _params())
def test_recompute_before_the_backfill_folds_anything(make_env, target: Target, phase: str) -> None:
    """The watermark states before the first chunk commits.

    ``unclaimed``: registered without population; the watermark is active
    with no cursor (backfill_cursor_pk IS NULL), so the triggers skip every
    write and the backfill folds everything. ``claimed``: the backfill has
    committed its claim (cursor = min - 1, max_pk) and is held before its
    first chunk by a row lock: the triggers maintain rows outside the claimed
    range, and their recomputes must not read the claimed rows (none is
    folded yet). In both, the writes of the shared setup and of group 10
    must leave the view exact. In the unclaimed state no trigger reaches a
    recompute (the guard skips every row), so the recompute window predicate's
    handling of a NULL cursor is not observable there.

    Deterministic: independent connections; the claimed state is held by a
    row lock the first chunk waits for (checked through the engine's lock
    view), not by sleeps. The claimed state runs on PostgreSQL only (see the
    skip reason)."""
    if phase == "claimed" and target.family != "postgres":
        # SQLite serializes writers; on MySQL/MariaDB the held first chunk
        # already blocks the writers (measured: lock wait timeout 1205), so
        # the state cannot be held while they write.
        pytest.skip("the backfill cannot be held between its claim and first chunk while writers proceed")
    env = make_env(target, views=[], seed=False, schema_sql=BACKFILL_SCHEMA[target.family])
    rows, writes = _backfill_rows_and_writes()
    rows.append((29, 10, -7))
    env.execute("INSERT INTO bt (id, g, x) VALUES " + ", ".join(f"({i}, {g}, {x})" for i, g, x in rows))
    env.commit()
    bits = "" if target.family == "sqlite" else ", BIT_AND(x) AS ba, BIT_OR(x) AS bo"
    vd = ViewDef("v_bt", f"SELECT g, COUNT(*) AS n, AVG(x) AS a, MIN(x) AS mn, MAX(x) AS mx{bits} FROM bt GROUP BY g")
    env.views = [vd]
    plan = env.register(vd, populate=False)
    writer = env.connect()

    def run_writes(sqls: Sequence[str]) -> None:
        for sql in sqls:
            env.execute(sql, conn=writer)
            env.commit(conn=writer)

    if phase == "unclaimed":
        run_writes([*writes, *_UNFOLDED_GROUP_WRITES])
        env.manager.backfill_incrementally(plan, batch_size=20, snapshot_isolation=False)
        env.check("backfill after writes while the watermark was unclaimed")
        return
    blocker, monitor = env.connect(), env.connect()
    sid_bf, sid_blocker = _session_id(env, env.conn), _session_id(env, blocker)
    blocker.cursor().execute("SELECT id FROM bt WHERE id = 1 FOR UPDATE")
    errors: List[BaseException] = []
    finished = threading.Event()

    def backfill() -> None:
        try:
            env.manager.backfill_incrementally(plan, batch_size=20, snapshot_isolation=False)
        except BaseException as exc:  # inspected below
            errors.append(exc)
        finally:
            finished.set()

    th = threading.Thread(target=backfill, daemon=True)
    th.start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_bf, sid_blocker), finished,
                  "the first chunk waits for the row lock after the claim committed")
        run_writes(_UNFOLDED_GROUP_WRITES)
    finally:
        blocker.commit()
        th.join(LOCK_WAIT_TIMEOUT_S)
    if errors:
        raise errors[0]
    env.check("backfill after writes while claimed and not yet advanced")


@pytest.mark.parametrize("target", _params())
def test_projection_without_key_after_primary_key_update(make_env, target: Target) -> None:
    """Minimal repro: change only the key, then delete the row."""
    env = make_env(target, views=[vd for vd in VIEWS if vd.name == "v_pr"], seed=False)
    env.execute("INSERT INTO t (id, g, v, w) VALUES (1, 1, 5, 1), (2, 1, 5, 1), (3, 2, 7, 1)")
    env.commit()
    env.execute("UPDATE t SET id = id + 10 WHERE id < 3")
    env.commit()
    env.check("UPDATE of the primary key only")
    env.execute("DELETE FROM t WHERE id > 10")
    env.commit()
    env.check("DELETE of the re-keyed rows")


AVG_TYPE_SCHEMA = {
    "postgres": ["CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
                 "CREATE TABLE a (id INT PRIMARY KEY, g INT, n3 NUMERIC(12,3), n NUMERIC, r REAL, f DOUBLE PRECISION, "
                 "i INT, s SMALLINT)"],
    "mysql": ["CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL) ENGINE=InnoDB",
              "CREATE TABLE a (id INT PRIMARY KEY, g INT, n3 DECIMAL(12,3), n DECIMAL(30,10), r FLOAT, f DOUBLE, "
              "i INT, s SMALLINT) ENGINE=InnoDB"],
    "sqlite": ["CREATE TABLE d (g INTEGER PRIMARY KEY, name VARCHAR(20) NOT NULL)",
               "CREATE TABLE a (id INTEGER PRIMARY KEY, g INTEGER, n3 INTEGER, n INTEGER, r REAL, f REAL, "
               "i INTEGER, s SMALLINT)"],
}


@pytest.mark.parametrize("target", _params())
@pytest.mark.parametrize("col", ["n3", "n", "r", "f", "i", "s"])
@pytest.mark.parametrize("shape", ["single", "join"])
def test_avg_new_group_result_type_and_scale(make_env, target: Target, col: str, shape: str) -> None:
    """AVG of a group created by an INSERT or by an UPDATE moving a row into a
    new group equals the native AVG (type and decimal scale) for each argument
    type; values are exactly representable, so no float ordering is involved."""
    env = make_env(target, views=[], schema_sql=AVG_TYPE_SCHEMA[target.family], seed=False)
    env.execute("INSERT INTO d (g, name) VALUES " + ", ".join(f"({g}, 'n{g}')" for g in range(1, 9)))
    env.commit()
    if shape == "single":
        vd = ViewDef("av", f"SELECT g, AVG({col}) AS a, COUNT(*) AS c FROM a GROUP BY g")
    else:
        vd = ViewDef("av", f"SELECT d.name, AVG(a.{col}) AS a, COUNT(*) AS c FROM a JOIN d ON a.g = d.g "
                           "GROUP BY d.name", join=True)
    env.views = [vd]
    # AVG over REAL/FLOAT is refused by default (order-sensitive); this test
    # uses exactly representable values and opts in per code.
    env.register(vd, accept_limitations=FLOAT_OPT_IN if col in ("r", "f") else None)
    env.check("registration")
    for label, sql in [
        ("INSERT creating a group", f"INSERT INTO a (id, g, {col}) VALUES (1, 1, 3)"),
        ("multi-row INSERT creating groups", f"INSERT INTO a (id, g, {col}) VALUES (2, 1, 4), (3, 2, 1), (4, 3, NULL)"),
        ("UPDATE moving a row into a new group", "UPDATE a SET g = 4 WHERE id = 2"),
        ("UPDATE of the value in a one-row group", f"UPDATE a SET {col} = 2 WHERE id = 3"),
        ("DELETE emptying a group", "DELETE FROM a WHERE id = 1"),
    ]:
        env.execute(sql)
        env.commit()
        env.check(label)


@pytest.mark.parametrize("target", _params())
def test_expression_group_key_result_type(make_env, target: Target) -> None:
    """Regression: register the view; the reader's column types must equal
    the view's (MariaDB types `id % 7` over INT as BIGINT; the backing column
    was INT)."""
    env = make_env(target, views=[vd for vd in VIEWS if vd.name == "v_exk"])
    env.check("registration")
    env.execute("INSERT INTO t (id, g, v) VALUES (1001, 1, 2), (1002, 2, 3)")
    env.commit()
    env.check("multi-row INSERT")


# Integer expression group keys over every integer column type: the reader's
# column type (driver type code) and values must equal the view's.
INT_KEY_SCHEMA = {
    "postgres": ["CREATE TABLE ik (id INT PRIMARY KEY, i INT, s SMALLINT, b BIGINT)"],
    "mysql": ["CREATE TABLE ik (id INT PRIMARY KEY, i INT, s SMALLINT, b BIGINT, ti TINYINT, mi MEDIUMINT, "
              "u INT UNSIGNED) ENGINE=InnoDB"],
    "sqlite": ["CREATE TABLE ik (id INTEGER PRIMARY KEY, i INTEGER, s SMALLINT, b BIGINT)"],
}
INT_KEY_EXPRS: List[Tuple[str, Tuple[str, ...]]] = [
    ("id % 7", ("postgres", "mysql", "sqlite")),
    ("i % 3", ("postgres", "mysql", "sqlite")),
    ("MOD(i, 3)", ("postgres", "mysql", "sqlite")),
    ("s % 2", ("postgres", "mysql", "sqlite")),
    ("b % 7", ("postgres", "mysql", "sqlite")),
    ("i % b", ("postgres", "mysql", "sqlite")),
    ("ti % 2", ("mysql",)),
    ("mi % 2", ("mysql",)),
    ("u % 7", ("mysql",)),
    ("-i", ("postgres", "mysql", "sqlite")),
    ("i + 1", ("postgres", "mysql", "sqlite")),
    ("s + s", ("postgres", "mysql", "sqlite")),
    ("i * 2", ("postgres", "mysql", "sqlite")),
    ("CASE WHEN i > 0 THEN i ELSE 0 END", ("postgres", "mysql", "sqlite")),
    ("COALESCE(i, 0)", ("postgres", "mysql", "sqlite")),
    ("i DIV 2", ("mysql",)),
]


@pytest.mark.parametrize("target", _params())
@pytest.mark.parametrize("expr", [e for e, _ in INT_KEY_EXPRS], ids=[e for e, _ in INT_KEY_EXPRS])
def test_integer_expression_group_key_result_type(make_env, request, target: Target, expr: str) -> None:
    _check_integer_expression_view(make_env, request, target, expr, lambda e: (
        ViewDef("ik_v", f"SELECT {e} AS k, COUNT(*) AS c FROM ik GROUP BY {e}")))


@pytest.mark.parametrize("target", _params())
@pytest.mark.parametrize("expr", [e for e, _ in INT_KEY_EXPRS], ids=[e for e, _ in INT_KEY_EXPRS])
@pytest.mark.parametrize("agg", ["MIN", "MAX"])
def test_min_max_of_integer_expression_result_type(make_env, request, target: Target, expr: str, agg: str) -> None:
    _check_integer_expression_view(make_env, request, target, expr, lambda e: (
        ViewDef("ik_a", f"SELECT s, {agg}({e}) AS a FROM ik GROUP BY s")))


SUM_INT_EXPRS: List[Tuple[str, Tuple[str, ...]]] = [
    *INT_KEY_EXPRS,
    ("b + 1", ("postgres", "mysql", "sqlite")),
    ("ABS(i)", ("postgres", "mysql", "sqlite")),
    ("CASE WHEN i > 0 THEN i ELSE NULL END", ("postgres", "mysql", "sqlite")),
    ("NULLIF(i, 0)", ("postgres", "mysql", "sqlite")),
    ("NULLIF(i, b)", ("postgres", "mysql", "sqlite")),
    ("GREATEST(i, 0)", ("postgres", "mysql")),
    ("IF(i > 0, i, 0)", ("mysql",)),
    ("CAST(ABS(i) AS UNSIGNED)", ("mysql",)),
]


@pytest.mark.parametrize("target", _params())
@pytest.mark.parametrize("expr", [e for e, _ in SUM_INT_EXPRS], ids=[e for e, _ in SUM_INT_EXPRS])
def test_sum_of_integer_expression_result_type(make_env, request, target: Target, expr: str) -> None:
    _check_integer_expression_view(
        make_env,
        request,
        target,
        expr,
        lambda e: ViewDef("ik_s", f"SELECT s, SUM({e}) AS a FROM ik GROUP BY s"),
        expr_families=dict(SUM_INT_EXPRS),
    )


_E6 = ("MySQL reports a GROUP BY expression over TINYINT/SMALLINT/MEDIUMINT arithmetic as INT (LONG) "
       "while a plain SELECT of it is BIGINT; the backing column is BIGINT. TODO.md "
       "'Integer Expression Result Types Not Yet Derived Exactly'")
# (product, view name, expression) -> reason. Applied at run time because the
# product (MySQL or MariaDB) is read from the server.
KNOWN_INT_EXPR_TYPE_DEFECTS: Dict[Tuple[str, str, str], str] = {
    **{("mysql", "ik_v", e): _E6 for e in ("s % 2", "ti % 2", "mi % 2", "s + s")},
}


def _check_integer_expression_view(make_env, request, target: Target, expr: str,
                                   view: Callable[[str], ViewDef],
                                   expr_families: Optional[Dict[str, Tuple[str, ...]]] = None) -> None:
    families = (expr_families or dict(INT_KEY_EXPRS))[expr]
    if target.family not in families:
        pytest.skip(f"{expr} is not {target.family} syntax")
    env = make_env(target, views=[], schema_sql=INT_KEY_SCHEMA[target.family], seed=False)
    reason = KNOWN_INT_EXPR_TYPE_DEFECTS.get((env.product, view(expr).name, expr))
    if reason:
        request.node.add_marker(known_defect(reason))
    cols = "id, i, s, b" + (", ti, mi, u" if target.family == "mysql" else "")
    rows = []
    for n in range(1, 13):
        base = [n, (n * 5) % 11 - 3 if n % 6 else None, n % 4, n * 1000003]
        if target.family == "mysql":
            base += [n % 3, n * 7, n * 13]
        rows.append("(" + ", ".join("NULL" if x is None else str(x) for x in base) + ")")
    env.execute(f"INSERT INTO ik ({cols}) VALUES " + ", ".join(rows))
    env.commit()
    vd = view(expr)
    env.views = [vd]
    env.register(vd)
    env.check("registration")
    env.execute("UPDATE ik SET i = i + 1, s = s + 1 WHERE id % 2 = 0")
    env.commit()
    env.check("UPDATE")


# ---------------------------------------------------------------------------
# UPDATE of each single column, on view shapes whose stored rows are keyed by
# something other than the updated column
# ---------------------------------------------------------------------------
#
# Contract: an UPDATE of any column of a source table, alone, keeps every view
# exact, including when the column is not in the view's output (primary key
# omitted by a projection; column used only in WHERE, HAVING, a join
# condition, an expression group key or a DISTINCT key) and when the new value
# equals the old one under the column's comparison but is not identical
# (case or trailing-space change under a case-insensitive or PAD SPACE
# collation, numeric scale, the sign of a floating-point zero). A later
# DELETE of the updated rows must retract them (no phantom rows).

UPDATE_COLUMN_VIEWS: List[ViewDef] = [
    ViewDef("u_pr", "SELECT g, v FROM t WHERE w > 0"),
    ViewDef("u_bare", "SELECT v FROM t"),
    ViewDef("u_prm", "SELECT m FROM t WHERE g IS NOT NULL"),
    ViewDef("u_ds", "SELECT DISTINCT g FROM t"),
    ViewDef("u_dsw", "SELECT DISTINCT v FROM t WHERE w > 0"),
    ViewDef("u_wh", "SELECT g, SUM(v) AS s, COUNT(*) AS c FROM t WHERE w > 0 GROUP BY g HAVING COUNT(*) > 1"),
    ViewDef("u_hv", "SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING SUM(w) > 3"),
    ViewDef("u_exk", "SELECT id % 7 AS k, COUNT(*) AS c FROM t GROUP BY id % 7"),
    ViewDef("u_exg", "SELECT g % 3 AS k, SUM(v) AS s, COUNT(*) AS c FROM t GROUP BY g % 3"),
    ViewDef("u_mm", "SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM t WHERE w > 0 GROUP BY g"),
    ViewDef("u_pj", "SELECT t.v, d.name FROM t JOIN d ON t.g = d.g", join=True),
    ViewDef("u_pjw", "SELECT t.g, t.v FROM t JOIN d ON t.g = d.g WHERE t.w > 0", join=True),
    ViewDef("u_ja", "SELECT d.name, COUNT(*) AS c, SUM(t.v) AS s FROM t JOIN d ON t.g = d.g "
                    "WHERE t.w > 0 GROUP BY d.name", join=True),
]

# (label, SQL) of single-column UPDATEs, applied in order to one state; each
# changes one column of some rows (or rewrites it with its own value).
UPDATE_COLUMN_STEPS: List[Tuple[str, str]] = [
    ("t.id changed", "UPDATE t SET id = id + 1000 WHERE id % 3 = 0"),
    ("t.g changed", "UPDATE t SET g = (g + 1) % 7 WHERE id % 4 = 1"),
    ("t.v changed", "UPDATE t SET v = v + 1 WHERE id % 5 = 2"),
    ("t.w changed (filter membership)", "UPDATE t SET w = CASE WHEN w > 0 THEN 0 ELSE 1 END WHERE id % 2 = 0"),
    ("t.m changed", "UPDATE t SET m = m + 1 WHERE id % 3 = 1"),
    ("t.id changed again", "UPDATE t SET id = id + 2000 WHERE id % 2 = 1"),
    ("every t column rewritten with its own value", "UPDATE t SET id = id, g = g, v = v, w = w, m = m"),
    ("d.g changed (join key)", "UPDATE d SET g = g + 100 WHERE g IN (1, 4)"),
    ("d.name changed", "UPDATE d SET name = 'n7' WHERE g IN (2, 104)"),
    ("d.g restored", "UPDATE d SET g = g - 100 WHERE g >= 100"),
    ("DELETE of re-keyed and moved t rows", "DELETE FROM t WHERE id > 1000 OR id % 4 = 1"),
    ("DELETE of d rows", "DELETE FROM d WHERE g IN (2, 3)"),
    ("DELETE of every t row", "DELETE FROM t"),
]


@pytest.mark.parametrize("target", _params())
@pytest.mark.parametrize("view", UPDATE_COLUMN_VIEWS, ids=[vd.name for vd in UPDATE_COLUMN_VIEWS])
def test_single_column_update_on_every_column(make_env, target: Target, view: ViewDef) -> None:
    env = make_env(target, views=[view])
    env.check("registration")
    for label, sql in UPDATE_COLUMN_STEPS:
        env.execute(sql)
        env.commit()
        env.check(label)


# Composite primary key: an UPDATE of one key column only.
COMPOSITE_KEY_SCHEMA = {
    "postgres": ["CREATE TABLE t2 (a INT NOT NULL, b INT NOT NULL, x INT, PRIMARY KEY (a, b))"],
    "mysql": ["CREATE TABLE t2 (a INT NOT NULL, b INT NOT NULL, x INT, PRIMARY KEY (a, b)) ENGINE=InnoDB"],
    "sqlite": ["CREATE TABLE t2 (a INTEGER NOT NULL, b INTEGER NOT NULL, x INTEGER, PRIMARY KEY (a, b))"],
}
COMPOSITE_KEY_VIEWS = [
    ViewDef("c_x", "SELECT x FROM t2"),
    ViewDef("c_ax", "SELECT a, x FROM t2 WHERE x > 0"),
    ViewDef("c_bx", "SELECT b, x FROM t2"),
    ViewDef("c_dx", "SELECT DISTINCT x FROM t2"),
    ViewDef("c_agg", "SELECT a, SUM(x) AS s, COUNT(*) AS c FROM t2 GROUP BY a"),
]


@pytest.mark.parametrize("target", _params())
@pytest.mark.parametrize("view", COMPOSITE_KEY_VIEWS, ids=[vd.name for vd in COMPOSITE_KEY_VIEWS])
def test_composite_key_partial_update(make_env, target: Target, view: ViewDef) -> None:
    env = make_env(target, views=[], schema_sql=COMPOSITE_KEY_SCHEMA[target.family], seed=False)
    env.execute("INSERT INTO t2 (a, b, x) VALUES " + ", ".join(
        f"({i % 4}, {i}, {(i * 7) % 5 - 1})" for i in range(1, 21)))
    env.commit()
    env.views = [view]
    env.register(view)
    env.check("registration")
    for label, sql in [
        ("b only", "UPDATE t2 SET b = b + 100 WHERE b % 3 = 0"),
        ("a only", "UPDATE t2 SET a = a + 10 WHERE b % 2 = 0"),
        ("a and b swapped", "UPDATE t2 SET a = b, b = a WHERE b % 5 = 1"),
        ("both key columns rewritten with their own value", "UPDATE t2 SET a = a, b = b"),
        ("DELETE of re-keyed rows", "DELETE FROM t2 WHERE b > 100 OR a >= 10"),
        ("DELETE of every row", "DELETE FROM t2"),
    ]:
        env.execute(sql)
        env.commit()
        env.check(label)


# UPDATEs whose new value compares equal to the old one but is not identical.
SAME_VALUE_CASES: Dict[str, Tuple[List[str], List[str], List[Tuple[str, str]]]] = {
    # family: (schema, views (select texts), [(label, UPDATE)])
    "postgres": (
        ["CREATE TABLE s (id INT PRIMARY KEY, n NUMERIC, f DOUBLE PRECISION, txt VARCHAR(20))",
         "INSERT INTO s VALUES (1, 1.5, 0.0, 'ab'), (2, 2, 1.0, 'cd'), (3, NULL, NULL, NULL)"],
        ["SELECT n, f, txt FROM s", "SELECT id, n FROM s", "SELECT txt, COUNT(*) AS c FROM s GROUP BY txt"],
        [("numeric scale only", "UPDATE s SET n = n * 1.00"),
         ("float zero sign only", "UPDATE s SET f = -f WHERE f = 0")],
    ),
    "mysql": (
        ["CREATE TABLE s (id INT PRIMARY KEY, txt VARCHAR(20), f DOUBLE) ENGINE=InnoDB",
         "INSERT INTO s VALUES (1, 'ab', 0.0), (2, 'cd', 1.0), (3, NULL, NULL)"],
        ["SELECT txt, f FROM s", "SELECT id, txt FROM s"],
        [("string case only (case-insensitive collation)", "UPDATE s SET txt = UPPER(txt)"),
         ("trailing space only", "UPDATE s SET txt = CONCAT(txt, ' ')"),
         ("float zero sign only", "UPDATE s SET f = -f WHERE f = 0")],
    ),
    "sqlite": (
        ["CREATE TABLE s (id INTEGER PRIMARY KEY, txt TEXT COLLATE NOCASE, r TEXT COLLATE RTRIM, f REAL)",
         "INSERT INTO s VALUES (1, 'ab', 'x', 0.0), (2, 'cd', 'y', 1.0), (3, NULL, NULL, NULL)"],
        ["SELECT txt, r, f FROM s", "SELECT id, txt, r FROM s"],
        [("string case only (NOCASE)", "UPDATE s SET txt = UPPER(txt)"),
         ("trailing space only (RTRIM)", "UPDATE s SET r = r || ' '"),
         ("float zero sign only", "UPDATE s SET f = -f WHERE f = 0")],
    ),
}


@pytest.mark.parametrize("target", _params())
def test_update_to_equal_but_not_identical_value(make_env, target: Target) -> None:
    schema, selects, steps = SAME_VALUE_CASES[target.family]
    env = make_env(target, views=[], schema_sql=schema, seed=False)
    env.commit()
    env.views = [ViewDef(f"s_{i}", sel) for i, sel in enumerate(selects)]
    for vd in env.views:
        env.register(vd)
    env.check("registration")
    for label, sql in steps:
        env.execute(sql)
        env.commit()
        env.check(label)


# ---------------------------------------------------------------------------
# PostgreSQL partitions
# ---------------------------------------------------------------------------

PARTITION_SCHEMA = [
    "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
    "CREATE TABLE t (id INT NOT NULL, g INT NOT NULL, v BIGINT, w INT, m NUMERIC(12,3), PRIMARY KEY (g, id)) "
    "PARTITION BY RANGE (g)",
    "CREATE TABLE t_p0 PARTITION OF t FOR VALUES FROM (MINVALUE) TO (3)",
    "CREATE TABLE t_p1 PARTITION OF t FOR VALUES FROM (3) TO (MAXVALUE)",
    "CREATE INDEX t_g_v ON t (g, v)",
]
PARTITION_VIEWS = [vd for vd in VIEWS if vd.name in ("v_agg", "v_mm", "v_ds", "v_j", "v_pj")]


def _partition_env(make_env, target: Target) -> Env:
    env = make_env(target, views=[], schema_sql=PARTITION_SCHEMA, seed=False)
    env.execute("INSERT INTO d (g, name) VALUES " + ", ".join(f"({g}, '{n}')" for g, n in SEED_D))
    env.execute("INSERT INTO t (id, g, v, w, m) VALUES " + ", ".join(
        f"({i}, {i % 6}, {(i * 37) % 101 - 30}, {i % 5}, {i}.5)" for i in range(1, 41)))
    env.commit()
    env.views = list(PARTITION_VIEWS)
    for vd in PARTITION_VIEWS:
        env.register(vd)
    return env


@pytest.mark.parametrize("target", _params(("postgres",)))
def test_pg_partitioned_parent_writes(make_env, target: Target) -> None:
    env = _partition_env(make_env, target)
    env.check("registration on a partitioned table")
    for label, sql in [
        ("parent multi-row insert", "INSERT INTO t (id, g, v) VALUES (100, 0, 1), (101, 4, 2), (102, 5, NULL)"),
        ("parent cross-partition update", "UPDATE t SET g = (g + 3) % 6 WHERE id % 2 = 0"),
        ("parent delete", "DELETE FROM t WHERE id % 3 = 0"),
    ]:
        env.execute(sql)
        env.commit()
        env.check(label)


@pytest.mark.parametrize("target", _params(("postgres",)))
def test_pg_direct_partition_writes(make_env, target: Target) -> None:
    """Direct writes to a partition fire only the partition's statement
    triggers (finding 4); maintenance installs them on every partition."""
    env = _partition_env(make_env, target)
    for label, sql in [
        ("direct partition insert", "INSERT INTO t_p1 (id, g, v) VALUES (200, 3, 9), (201, 4, -9), (202, 5, 0)"),
        ("direct partition update", "UPDATE t_p0 SET v = v + 7 WHERE id % 2 = 1"),
        ("direct partition delete", "DELETE FROM t_p1 WHERE id % 4 = 0"),
    ]:
        env.execute(sql)
        env.commit()
        env.check(label)


# ---------------------------------------------------------------------------
# Concurrency: independent connections, deliberate synchronization
# ---------------------------------------------------------------------------


def _session_id(env: Env, conn: Any) -> int:
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid()" if env.family == "postgres" else "SELECT CONNECTION_ID()")
    sid = int(cur.fetchone()[0])
    conn.commit()
    return sid


def _wait_for(pred: Callable[[], bool], done: threading.Event, what: str) -> None:
    deadline = time.monotonic() + LOCK_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if pred():
            return
        if done.is_set():
            raise AssertionError(f"the session finished before {what}; the interleaving was not exercised")
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting until {what}")


def _row_lock_waiting(env: Env, monitor: Any, sid: int, blocker: int) -> bool:
    if env.family == "postgres":
        cur = monitor.cursor()
        cur.execute("SELECT wait_event_type, pg_blocking_pids(pid) FROM pg_stat_activity WHERE pid = %s", (sid,))
        row = cur.fetchone()
        monitor.commit()
        return bool(row and row[0] == "Lock" and blocker in (row[1] or []))
    # Reuses the MySQL/MariaDB lock-wait evidence collector, which handles the
    # MariaDB information_schema cache (see its comments).
    from test_avg_concurrent_maintenance import _is_lock_waiting

    return _is_lock_waiting(Dialect.MYSQL, monitor, sid, blocker)


class _Worker:
    """Runs statements on its own connection in a thread, recording errors."""

    def __init__(self, conn: Any, statements: Sequence[str]) -> None:
        self.conn = conn
        self.statements = statements
        self.error: Optional[BaseException] = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            cur = self.conn.cursor()
            for s in self.statements:
                cur.execute(s)
            self.conn.commit()
        except BaseException as exc:  # inspected by the test
            self.error = exc
            try:
                self.conn.rollback()
            except Exception:
                pass
        finally:
            self.done.set()

    def start(self) -> "_Worker":
        self.thread.start()
        return self

    def join(self) -> None:
        self.thread.join(LOCK_WAIT_TIMEOUT_S)
        assert not self.thread.is_alive(), "worker did not finish"


@pytest.mark.parametrize("target", _params(("postgres", "mysql")))
def test_minmax_recompute_after_lock_wait(make_env, target: Target) -> None:
    """A deletes group 1's minimum (holds the group's backing row); B deletes
    another row of group 1 plus rows of other groups and waits for A; A
    commits; B commits. The view must reflect both deletions (finding 1: the
    PostgreSQL recompute runs in a statement after the lock, with a fresh
    READ COMMITTED snapshot)."""
    env = make_env(target, views=[vd for vd in VIEWS if vd.name in ("v_mm", "v_mmw")], seed=False)
    env.execute("INSERT INTO t (id, g, v, w) VALUES (1, 1, 1, 1), (2, 1, 2, 2), (3, 1, 3, 3), (4, 2, 10, 10), "
                "(5, 2, 20, 20), (6, 3, 7, 7)")
    env.commit()
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("DELETE FROM t WHERE id = 1")
    worker = _Worker(b, ["DELETE FROM t WHERE id IN (2, 5, 6)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("concurrent deletes of a group's extremes")


def _install_pause_trigger(env: Env, marker: int, lock_key: int) -> None:
    """Make writer A pause, while it holds some group locks and not others,
    on a lock held by the test.

    PostgreSQL (statement-level maintenance): a BEFORE UPDATE row trigger on
    the ``v_agg`` backing table pauses A's key-ordered upsert after it has
    locked group 1's backing row and before it locks group 2's (only in a
    session that set ``bulk.pause``). MySQL/MariaDB (row triggers): a BEFORE
    INSERT row trigger pauses before the row with id = ``marker`` is
    maintained."""
    if env.family == "postgres":
        store = env.plans["v_agg"].backing_table_name
        env.execute(
            f"CREATE FUNCTION bulk_pause() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
            f"IF current_setting('bulk.pause', true) = 'on' AND OLD.g = 1 THEN "
            f"PERFORM pg_advisory_xact_lock_shared({lock_key}); END IF; "
            f"RETURN NEW; END $$")
        env.execute(f"CREATE TRIGGER bulk_pause BEFORE UPDATE ON {store} FOR EACH ROW EXECUTE FUNCTION bulk_pause()")
        env.commit()
    else:
        env.execute(
            f"CREATE TRIGGER bulk_pause BEFORE INSERT ON t FOR EACH ROW BEGIN "
            f"IF NEW.id = {marker} THEN SET @bulk_pause = GET_LOCK('bulk_pause_{lock_key}', 60); END IF; END")
        env.commit()


def _run_opposite_order_inserts(env: Env) -> Tuple[Optional[BaseException], Optional[BaseException]]:
    """A inserts (group 1, group 2, group 3) and pauses holding group 1's
    maintenance lock but not group 2's; B inserts (group 2, group 1, group 3)
    and must wait for A; then A resumes. Locking groups in statement row order
    (row triggers) makes that a lock-order cycle; key-ordered locking does not."""
    lock_key = 4242
    marker = 1002
    _install_pause_trigger(env, marker, lock_key)
    a, b, ctl, monitor = env.connect(), env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    cur = ctl.cursor()
    if env.family == "postgres":
        cur.execute(f"SELECT pg_advisory_lock({lock_key})")
    else:
        cur.execute(f"SELECT GET_LOCK('bulk_pause_{lock_key}', 10)")
        assert cur.fetchone()[0] == 1
    a_stmts = ["INSERT INTO t (id, g, v) VALUES (1001, 1, 5), (1002, 2, 6), (1003, 3, 7)"]
    if env.family == "postgres":
        a_stmts.insert(0, "SET LOCAL bulk.pause = 'on'")
    wa = _Worker(a, a_stmts).start()
    wb = None
    try:
        def a_paused() -> bool:
            c = monitor.cursor()
            if env.family == "postgres":
                c.execute("SELECT wait_event FROM pg_stat_activity WHERE pid = %s", (sid_a,))
                ok = (c.fetchone() or [None])[0] == "advisory"
            else:
                c.execute("SELECT STATE FROM information_schema.PROCESSLIST WHERE ID = %s", (sid_a,))
                ok = "user lock" in str((c.fetchone() or [""])[0]).lower()
            monitor.commit()
            return ok

        _wait_for(a_paused, wa.done, "A pauses on the test lock")
        wb = _Worker(b, ["INSERT INTO t (id, g, v) VALUES (2001, 2, 8), (2002, 1, 9), (2003, 3, 1)"]).start()
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), wb.done, "B waits for A's group-1 lock")
    finally:
        if env.family == "postgres":
            cur.execute(f"SELECT pg_advisory_unlock({lock_key})")
        else:
            cur.execute(f"SELECT RELEASE_LOCK('bulk_pause_{lock_key}')")
        ctl.commit()
        wa.join()
        if wb is not None:
            wb.join()
    return wa.error, (wb.error if wb else None)


def _is_deadlock(env: Env, exc: BaseException) -> bool:
    if env.family == "postgres":
        return getattr(exc, "pgcode", None) == "40P01"
    return bool(getattr(exc, "args", None)) and exc.args[0] == 1213


@pytest.mark.parametrize("target", _params(("postgres", "mysql")))
def test_opposite_order_multi_row_inserts_fail_closed(make_env, target: Target) -> None:
    """Whatever the outcome, a deadlock is a clean rollback of one whole
    transaction, the other commits, and the views stay exact."""
    env = make_env(target, views=[vd for vd in VIEWS if vd.name in ("v_agg", "v_mm", "v_j")])
    ea, eb = _run_opposite_order_inserts(env)
    for e in (ea, eb):
        if e is not None and not _is_deadlock(env, e):
            raise e
    assert not (ea and eb), "both transactions failed"
    cur = env.conn.cursor()
    cur.execute("SELECT id FROM t WHERE id > 1000 ORDER BY id")
    ids = [r[0] for r in cur.fetchall()]
    env.commit()
    expected = ([] if ea else [1001, 1002, 1003]) + ([] if eb else [2001, 2002, 2003])
    assert ids == sorted(expected), (ids, ea, eb)
    env.check("opposite-order concurrent multi-row inserts")
    if target.family == "mysql":
        # Documented consequence of row triggers on MySQL/MariaDB (design,
        # "Concurrency on these engines"): this interleaving is a lock cycle.
        assert (ea is None) != (eb is None), f"expected exactly one deadlock victim, got {ea!r} / {eb!r}"


@pytest.mark.parametrize("target", _params(("postgres",)))
def test_pg_opposite_order_multi_row_inserts_do_not_deadlock(make_env, target: Target) -> None:
    """PostgreSQL statement-level maintenance locks groups in key order, so
    this interleaving (finding 2) must not deadlock. MySQL/MariaDB keep row
    triggers and their documented deadlocks, so they are not part of this
    contract."""
    env = make_env(target, views=[vd for vd in VIEWS if vd.name in ("v_agg", "v_mm", "v_j")])
    ea, eb = _run_opposite_order_inserts(env)
    for e in (ea, eb):
        if e is not None and _is_deadlock(env, e):
            raise DeadlockObserved(f"deadlock: {e!r}")
        if e is not None:
            raise e
    env.check("opposite-order concurrent multi-row inserts")


@pytest.mark.parametrize("target", _params())
def test_stage4_reader_derived_avg_and_same_group_update_exactness(make_env, target: Target) -> None:
    """Stage 4: reader-derived AVG (standalone, shared with SUM, compound,
    HAVING) and same-group UPDATE fast paths stay exact across NULL
    transitions, value updates, group moves, deletes, and rollbacks."""
    extra_views = [
        ViewDef("v_s4_avg_only", "SELECT g, AVG(v) AS a FROM t GROUP BY g"),
        ViewDef("v_s4_avg_sum", "SELECT g, SUM(v) AS s, COUNT(*) AS c, AVG(v) AS a FROM t GROUP BY g"),
        # Compound reader-derived AVG + HAVING AVG(v) > 0. (Note: AVG(v) * 2 on
        # MySQL 9.7 has a pre-existing intermediate-decimal-precision mismatch
        # vs both stored DECIMAL(p,4) and CAST(sum AS DECIMAL(65,0))/cnt: live
        # AVG(v)*2 returns 46.3333 while 23.1667*2 returns 46.3334.)
        ViewDef("v_s4_avg_comp", "SELECT g, AVG(v) + 10 AS a2, COUNT(*) AS c FROM t GROUP BY g HAVING AVG(v) > 0"),
        ViewDef("v_s4_global_avg", "SELECT COUNT(*) AS c, AVG(v) AS a FROM t"),
        ViewDef("v_s4_avg_dec", "SELECT g, AVG(m) AS am, SUM(m) AS sm, COUNT(*) AS c FROM t GROUP BY g", ("postgres", "mysql")),
    ]
    env = make_env(target, views=extra_views)
    env.check("initial")
    steps = [
        ("same_group_upd_all", "UPDATE t SET v = v + 7 WHERE v IS NOT NULL"),
        ("same_group_null_to_val", "UPDATE t SET v = 13 WHERE v IS NULL"),
        ("same_group_val_to_null", "UPDATE t SET v = NULL WHERE id % 3 = 0"),
        ("same_group_all_null_grp", "UPDATE t SET v = NULL WHERE g = 1"),
        ("same_group_revive_null_grp", "UPDATE t SET v = 42 WHERE g = 1 AND id = (SELECT mid FROM (SELECT MIN(id) AS mid FROM t WHERE g = 1) AS _mv_sub)"),
        ("move_group", "UPDATE t SET g = CASE WHEN g IS NULL THEN 0 ELSE (g + 1) % 6 END"),
        ("delete_half", "DELETE FROM t WHERE id % 2 = 0"),
        ("delete_all", "DELETE FROM t"),
        ("reinsert", "INSERT INTO t (id, g, v, w, m) VALUES (501, 1, 10, 1, NULL), (502, 1, NULL, 1, NULL), (503, 2, -5, 1, NULL)"),
    ]
    for label, sql in steps:
        env.execute(sql)
        env.commit()
        env.check(label)
    # Verify savepoint rollback restores same-group UPDATEs cleanly.
    if target.family == "sqlite":
        env.execute("BEGIN")
    env.execute("UPDATE t SET v = 100 WHERE g = 1")
    env.execute("SAVEPOINT sp_s4")
    env.execute("UPDATE t SET v = NULL WHERE g = 1")
    env.execute("ROLLBACK TO SAVEPOINT sp_s4")
    env.commit()
    env.check("savepoint_rollback_same_group")


@pytest.mark.parametrize("target", _params(("sqlite",)))
def test_stage4_sqlite_join_delete_index_and_cost_budget(target: Target) -> None:
    """Stage 4: SQLite join empty-group deletes use PK index lookups (no SCAN),
    scale independently of total group count, and lean agg/ds row triggers stay
    within a bounded per-row budget relative to the measured trigger floor."""
    con = sqlite3.connect(":memory:")
    try:
        MaterializedViewManager.configure_sqlite_connection(con)
        con.executescript(
            "CREATE TABLE d (g INTEGER PRIMARY KEY, name TEXT NOT NULL);"
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, v INTEGER NOT NULL, w INTEGER NOT NULL);"
            "CREATE INDEX t_g ON t (g);"
        )
        # 1. Measure empty trigger floor on 4,000 rows.
        con.executescript(
            "CREATE TABLE floor_t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, v INTEGER NOT NULL);"
            "CREATE TRIGGER floor_ai AFTER INSERT ON floor_t FOR EACH ROW BEGIN SELECT 1; END;"
            "CREATE TRIGGER floor_au AFTER UPDATE ON floor_t FOR EACH ROW BEGIN SELECT 1; END;"
        )
        n_rows = 4000
        t0 = time.perf_counter()
        con.execute(
            f"INSERT INTO floor_t (id, g, v) WITH RECURSIVE s(i) AS "
            f"(SELECT 1 UNION ALL SELECT i + 1 FROM s WHERE i < {n_rows}) "
            f"SELECT i, i % 100, i FROM s"
        )
        con.commit()
        floor_ins = max(time.perf_counter() - t0, 1e-4)
        t0 = time.perf_counter()
        con.execute("UPDATE floor_t SET v = v + 1")
        con.commit()
        floor_upd = max(time.perf_counter() - t0, 1e-4)

        # 2. Lean single-table agg view cost relative to trigger floor.
        mgr = MaterializedViewManager(con, dialect="sqlite")
        agg_plan = mgr.register_view(
            "CREATE VIEW mv_s4_agg AS SELECT g, SUM(v) AS s, COUNT(*) AS c, AVG(v) AS a FROM t GROUP BY g;"
        )
        con.commit()
        t0 = time.perf_counter()
        con.execute(
            f"INSERT INTO t (id, g, v, w) WITH RECURSIVE s(i) AS "
            f"(SELECT 1 UNION ALL SELECT i + 1 FROM s WHERE i < {n_rows}) "
            f"SELECT i, i % 100, (i * 7919) % 100003, i % 5 FROM s"
        )
        con.commit()
        agg_ins = time.perf_counter() - t0
        t0 = time.perf_counter()
        con.execute("UPDATE t SET v = v + 1")
        con.commit()
        agg_upd_v = time.perf_counter() - t0
        # Exactness vs live view.
        mv_rows = con.execute("SELECT g, s, c, a FROM mv_s4_agg ORDER BY g").fetchall()
        live_rows = con.execute("SELECT g, SUM(v), COUNT(*), AVG(v) FROM t GROUP BY g ORDER BY g").fetchall()
        assert mv_rows == live_rows
        # Generous CI-safe budget catching O(group)/O(table) regressions while
        # tolerating noisy runners (floor is ~2-5ms for 4k rows; lean agg is ~15-25ms).
        assert agg_ins < max(floor_ins * 25.0, 0.25), f"agg ins={agg_ins:.4f}s floor={floor_ins:.4f}s"
        assert agg_upd_v < max(floor_upd * 25.0, 0.25), f"agg upd_v={agg_upd_v:.4f}s floor={floor_upd:.4f}s"
        assert "\n    a " not in agg_plan.backing_table_ddl

        # 3. Join empty-group DELETE scaling: 1,500 groups vs 50 groups, plus
        #    composite-PK EQP proving no table SCAN on mv_table.
        def _measure_join_del(n_groups: int) -> float:
            c2 = sqlite3.connect(":memory:")
            try:
                MaterializedViewManager.configure_sqlite_connection(c2)
                c2.executescript(
                    "CREATE TABLE d (g INTEGER PRIMARY KEY, name TEXT NOT NULL);"
                    "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, w INTEGER NOT NULL);"
                )
                c2.executemany("INSERT INTO d VALUES (?, ?)", [(i, f"g_{i}") for i in range(n_groups)])
                c2.executemany("INSERT INTO t VALUES (?, ?, ?)", [(i, i % n_groups, i % 3) for i in range(1500)])
                c2.commit()
                m2 = MaterializedViewManager(c2, dialect="sqlite")
                jp = m2.register_view(
                    "CREATE VIEW mv_jc AS SELECT d.name, t.w, COUNT(*) AS c "
                    "FROM t JOIN d ON t.g = d.g GROUP BY d.name, t.w;"
                )
                c2.commit()
                for tr in jp.trigger_ddls:
                    if "_ad" not in tr:
                        continue
                    for stmt in tr.split("BEGIN", 1)[1].rsplit("END", 1)[0].split(";"):
                        st = stmt.strip()
                        if st.upper().startswith("DELETE"):
                            probe = st.replace("OLD.g", "1").replace("OLD.name", "'g_1'").replace("OLD.w", "1")
                            eqp = "\n".join(r[3] for r in c2.execute("EXPLAIN QUERY PLAN " + probe).fetchall())
                            assert f"SCAN {jp.backing_table_name}" not in eqp, eqp
                            assert f"SEARCH {jp.backing_table_name}" in eqp, eqp
                ts = time.perf_counter()
                c2.execute("DELETE FROM t WHERE id < 1000")
                c2.commit()
                dt = time.perf_counter() - ts
                assert (
                    c2.execute("SELECT name, w, c FROM mv_jc ORDER BY name, w").fetchall()
                    == c2.execute(
                        "SELECT d.name, t.w, COUNT(*) FROM t JOIN d ON t.g = d.g GROUP BY d.name, t.w ORDER BY d.name, t.w"
                    ).fetchall()
                )
                return dt
            finally:
                c2.close()

        dt_50 = _measure_join_del(50)
        dt_1500 = _measure_join_del(1500)
        # With an O(groups) full-table scan, 30x more groups slows deletes by
        # ~15-30x; with O(1) indexed deletes, 1,500 groups stays within ~4x.
        assert dt_1500 < max(dt_50 * 6.0, 0.10), f"dt_50={dt_50:.4f}s dt_1500={dt_1500:.4f}s"
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Randomized, schema-aware statement generator
# ---------------------------------------------------------------------------

RETAINED_SEEDS = [11, 23, 37, 41, 59, 73]
DEFAULT_STEPS = 30

GENERATOR_LIMITS = (
    "Tables: t(id PK, g, v, w, m) and d(g PK, name) only; no FK, partitions or extra indexes in random runs.",
    "Statements: the KINDS productions below with literal values; no parameters, no COPY/LOAD DATA, "
    "no writable CTE touching d, no TRUNCATE, no concurrency.",
    "Predicates: conjunctions/disjunctions/negations of up to 2 atoms over id/g/v/w (depth <= 2).",
    "Views: the VIEWS shapes only (no window, float, statistical, outer-join or nested views).",
)

KINDS = {
    "insert_values": ("postgres", "mysql", "sqlite"),
    "insert_select": ("postgres", "mysql", "sqlite"),
    "update_values": ("postgres", "mysql", "sqlite"),
    "update_move": ("postgres", "mysql", "sqlite"),
    "update_pk": ("postgres", "mysql", "sqlite"),
    "delete": ("postgres", "mysql", "sqlite"),
    "dim_insert": ("postgres", "mysql", "sqlite"),
    "dim_update": ("postgres", "mysql", "sqlite"),
    "dim_delete": ("postgres", "mysql", "sqlite"),
    "upsert_update": ("postgres", "mysql", "sqlite"),
    "upsert_ignore": ("postgres", "mysql", "sqlite"),
    "replace": ("mysql", "sqlite"),
    "merge": ("postgres",),
    "cte_move": ("postgres",),
    "multi_table_update": ("mysql",),
    "fail_duplicate_key": ("postgres", "mysql", "sqlite"),
    "fail_check": ("postgres", "mysql", "sqlite"),
    "tx_rollback": ("postgres", "mysql", "sqlite"),
    "savepoint_rollback": ("postgres", "mysql", "sqlite"),
}
DISTRIBUTIONS = ("uniform", "hot", "pair", "null_heavy", "new_groups")
ROW_COUNTS = (0, 1, 2, 5, 17, 60, 150)


class BulkStatementGenerator:
    """Seeded generator of literal bulk DML over the harness schema.

    It reads the live keys before each statement (schema- and state-aware), so
    it can aim updates/deletes/upserts at existing rows, collide on purpose,
    or create new groups."""

    def __init__(self, seed: int, env: Env) -> None:
        self.rng = random.Random(seed)
        self.env = env
        self.kinds = [k for k, fams in KINDS.items() if env.family in fams]
        if env.family == "postgres" and env.version < (15, 0):
            self.kinds.remove("merge")
        self.coverage: collections.Counter = collections.Counter()
        self.next_id = 0

    def _offset(self) -> int:
        """A key offset that moves every live id above every live id."""
        return max(self._ids(), default=0) + 1

    def _ids(self) -> List[int]:
        cur = self.env.conn.cursor()
        cur.execute("SELECT id FROM t")
        ids = [int(r[0]) for r in cur.fetchall()]
        self.env.commit()
        return ids

    def _g(self, dist: str) -> Optional[int]:
        r = self.rng
        if dist == "hot":
            return 1
        if dist == "pair":
            return r.choice((2, 3))
        if dist == "null_heavy":
            return None if r.random() < 0.6 else r.randrange(8)
        if dist == "new_groups":
            return r.randrange(8, 40)
        return None if r.random() < 0.1 else r.randrange(12)

    def _v(self) -> Optional[int]:
        r = self.rng
        if r.random() < 0.15:
            return None
        return r.choice((0, 1, -1, 5, 5, 42, -42, 999_999, -999_999, r.randrange(-100, 100)))

    def _m(self) -> Optional[str]:
        r = self.rng
        if r.random() < 0.15:
            return None
        return r.choice(("0.000", "1.250", "-1.250", "99999.999", str(Decimal(r.randrange(-99999, 99999)) / 1000)))

    def _row(self, rid: int, dist: str) -> str:
        w = None if self.rng.random() < 0.2 else self.rng.randrange(-3, 6)
        m = self._m() if self.env.family != "sqlite" else (None if self.rng.random() < 0.2 else str(self.rng.randrange(-9, 9)))
        return "(" + ", ".join(_sql_lit(x) for x in (rid, self._g(dist), self._v(), w, m)) + ")"

    def _atom(self) -> str:
        r = self.rng
        k = r.choice((2, 3, 5, 7))
        return r.choice((
            f"id % {k} = {r.randrange(k)}",
            "g IS NULL",
            f"g IN ({', '.join(str(r.randrange(12)) for _ in range(r.randrange(1, 4)))})",
            f"v BETWEEN {r.randrange(-50, 10)} AND {r.randrange(10, 80)}",
            "v IS NULL",
            f"w = {r.randrange(-1, 5)}",
            f"g < {r.randrange(12)}",
        ))

    def _pred(self) -> str:
        r = self.rng.random()
        if r < 0.45:
            return self._atom()
        if r < 0.7:
            return f"({self._atom()} AND {self._atom()})"
        if r < 0.9:
            return f"({self._atom()} OR {self._atom()})"
        return f"NOT ({self._atom()})"

    def _new_rows(self, n: int, dist: str, dup_existing: bool = False) -> str:
        ids = self._ids()
        rows = []
        for _ in range(n):
            if dup_existing and ids and self.rng.random() < 0.5:
                rid = self.rng.choice(ids)
            else:
                self.next_id = max(self.next_id, max(ids, default=0)) + 1
                rid = self.next_id
            rows.append(self._row(rid, dist))
        return ", ".join(rows)

    def _upsert(self, kind: str, n: int, dist: str) -> str:
        rows = self._new_rows(max(n, 1), dist, dup_existing=True)
        # A multi-row VALUES list must not repeat a key for ON CONFLICT DO UPDATE.
        seen: set = set()
        uniq = []
        for row in rows.split("), ("):
            key = row.strip("()").split(",")[0]
            if key not in seen:
                seen.add(key)
                uniq.append(row.strip("()"))
        vals = ", ".join(f"({u})" for u in uniq)
        e = self.env
        if kind == "upsert_update":
            if e.family == "postgres":
                return (f"INSERT INTO t (id, g, v, w, m) VALUES {vals} ON CONFLICT (id) DO UPDATE "
                        "SET g = EXCLUDED.g, v = EXCLUDED.v, w = t.w")
            if e.family == "sqlite":
                return (f"INSERT INTO t (id, g, v, w, m) VALUES {vals} ON CONFLICT (id) DO UPDATE "
                        "SET g = excluded.g, v = excluded.v, w = t.w")
            if e.is_mariadb:
                return f"INSERT INTO t (id, g, v, w, m) VALUES {vals} ON DUPLICATE KEY UPDATE g = VALUES(g), v = VALUES(v)"
            return f"INSERT INTO t (id, g, v, w, m) VALUES {vals} AS nw ON DUPLICATE KEY UPDATE g = nw.g, v = nw.v"
        if kind == "upsert_ignore":
            if e.family == "postgres":
                return f"INSERT INTO t (id, g, v, w, m) VALUES {vals} ON CONFLICT DO NOTHING"
            return f"INSERT {'OR IGNORE' if e.family == 'sqlite' else 'IGNORE'} INTO t (id, g, v, w, m) VALUES {vals}"
        return f"{'INSERT OR REPLACE' if e.family == 'sqlite' else 'REPLACE'} INTO t (id, g, v, w, m) VALUES {vals}"

    def next(self) -> Tuple[str, List[str], str]:
        """Return (kind, statements, expectation).

        expectation: 'commit' (every statement must succeed; then commit),
        'error' (a statement must fail; then roll back), 'error_or_commit'
        (either, depending on the data), 'rollback' (must succeed; then roll
        back)."""
        r = self.rng
        kind = r.choice(self.kinds)
        dist = r.choice(DISTRIBUTIONS)
        n = r.choice(ROW_COUNTS)
        self.coverage[kind] += 1
        self.coverage[f"dist:{dist}"] += 1
        self.coverage[f"rows:{n}"] += 1
        p = self._pred()
        if kind == "insert_values":
            if n == 0:
                return kind, ["INSERT INTO t (id, g, v, w, m) SELECT id, g, v, w, m FROM t WHERE 1 = 0"], "commit"
            return kind, [f"INSERT INTO t (id, g, v, w, m) VALUES {self._new_rows(n, dist)}"], "commit"
        if kind == "insert_select":
            return kind, [f"INSERT INTO t (id, g, v, w, m) SELECT id + {self._offset()}, "
                          f"{_sql_lit(self._g(dist))}, v, w, m FROM t WHERE {p}"], "commit"
        if kind == "update_values":
            # May exceed the CHECK bound when a row holds +-999999.
            return kind, [f"UPDATE t SET v = v + {r.randrange(-5, 6)}, w = {_sql_lit(r.choice((None, 0, 3)))} "
                          f"WHERE {p}"], "error_or_commit"
        if kind == "update_move":
            return kind, [f"UPDATE t SET g = {_sql_lit(self._g(dist))} WHERE {p}"], "commit"
        if kind == "update_pk":
            return kind, [f"UPDATE t SET id = id + {self._offset()} WHERE {p}"], "commit"
        if kind == "delete":
            return kind, [f"DELETE FROM t WHERE {p}"], "commit"
        if kind == "dim_insert":
            gs = sorted({r.randrange(0, 40) for _ in range(max(n, 1) % 9 + 1)})
            vals = ", ".join(f"({g}, 'n{r.randrange(4)}')" for g in gs)
            if self.env.family == "postgres":
                return kind, [f"INSERT INTO d (g, name) VALUES {vals} ON CONFLICT DO NOTHING"], "commit"
            return kind, [f"INSERT {'OR IGNORE' if self.env.family == 'sqlite' else 'IGNORE'} INTO d (g, name) VALUES {vals}"], "commit"
        if kind == "dim_update":
            return kind, [f"UPDATE d SET name = 'n{r.randrange(4)}' WHERE g % {r.choice((2, 3))} = {r.randrange(2)}"], "commit"
        if kind == "dim_delete":
            return kind, [f"DELETE FROM d WHERE g % {r.choice((3, 5, 7))} = {r.randrange(3)}"], "commit"
        if kind in ("upsert_update", "upsert_ignore", "replace"):
            return kind, [self._upsert(kind, n, dist)], "commit"
        if kind == "merge":
            rows = [x.strip("()") for x in self._new_rows(max(n, 1), dist, dup_existing=True).split("), (")]
            uniq = list({row.split(",")[0]: row for row in rows}.values())
            src = ", ".join(f"({u})" for u in uniq)
            return kind, [
                f"MERGE INTO t USING (VALUES {src}) AS s(id, g, v, w, m) ON t.id = s.id "
                "WHEN MATCHED AND s.v IS NULL THEN DELETE "
                "WHEN MATCHED THEN UPDATE SET g = s.g::int, v = s.v::bigint "
                "WHEN NOT MATCHED THEN INSERT (id, g, v, w, m) "
                "VALUES (s.id, s.g::int, s.v::bigint, s.w::int, s.m::numeric)"], "commit"
        if kind == "cte_move":
            return kind, [f"WITH x AS (DELETE FROM t WHERE {p} RETURNING id, g, v, w, m) "
                          f"INSERT INTO t (id, g, v, w, m) SELECT id + {self._offset()}, g, v, w, m FROM x"], "commit"
        if kind == "multi_table_update":
            return kind, [f"UPDATE t JOIN d ON t.g = d.g SET t.v = t.v + 1, d.name = 'n{r.randrange(4)}' "
                          f"WHERE d.g % {r.choice((2, 3))} = 0"], "error_or_commit"  # v + 1 may hit the CHECK
        if kind == "fail_duplicate_key":
            ids = self._ids()
            dup = r.choice(ids) if ids else None
            if dup is None:
                return kind, ["INSERT INTO t (id, g, v) VALUES (1, 1, 1), (1, 1, 1)"], "error"
            return kind, [f"INSERT INTO t (id, g, v, w, m) VALUES {self._new_rows(max(n, 1), dist)}, ({dup}, 1, 1, 1, NULL)"], "error"
        if kind == "fail_check":
            return kind, [f"UPDATE t SET v = CASE WHEN {p} THEN v + 1 ELSE 5000000 END"], "error_or_commit"
        if kind == "tx_rollback":
            return kind, [f"DELETE FROM t WHERE {p}", f"UPDATE t SET g = {_sql_lit(self._g(dist))} WHERE {self._pred()}",
                          f"INSERT INTO t (id, g, v, w, m) VALUES {self._new_rows(max(n, 1), dist)}"], "rollback"
        assert kind == "savepoint_rollback"
        # v * 2 may exceed the CHECK bound.
        return kind, [f"UPDATE t SET v = v * 2 WHERE {p}", "SAVEPOINT sp",
                      f"DELETE FROM t WHERE {self._pred()}", f"UPDATE t SET g = {_sql_lit(self._g(dist))}",
                      "ROLLBACK TO SAVEPOINT sp"], "error_or_commit"


def _seeds() -> List[int]:
    raw = os.environ.get("MATVIEW_BULK_SEEDS")
    return [int(s) for s in raw.split(",")] if raw else list(RETAINED_SEEDS)


_RANDOM_COVERAGE: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)


def _replay_hint(target: Target, seed: int) -> str:
    env_var = {"postgres": "POSTGRES_DSN=<dsn> ", "mysql": "MYSQL_DSN=<dsn> ", "sqlite": ""}[target.family]
    return (f"replay: MATVIEW_BULK_SEEDS={seed} MATVIEW_BULK_STEPS={os.environ.get('MATVIEW_BULK_STEPS', DEFAULT_STEPS)} "
            f"{env_var}python3 -m pytest -v test_native_bulk_statements.py -k 'randomized and {target.id}'")


@pytest.mark.parametrize("seed", _seeds())
@pytest.mark.parametrize("target", _params())
def test_randomized_bulk_statements(shared_env, target: Target, seed: int) -> None:
    env = shared_env(target)
    gen = BulkStatementGenerator(seed, env)
    steps = int(os.environ.get("MATVIEW_BULK_STEPS", DEFAULT_STEPS))
    log: List[str] = []
    try:
        for step in range(steps):
            kind, stmts, expect = gen.next()
            log.append(f"{step}: [{kind}/{expect}] " + " ; ".join(stmts))
            if env.family == "sqlite" and len(stmts) > 1:
                env.execute("BEGIN")
            failed = False
            error: Optional[BaseException] = None
            try:
                for s in stmts:
                    env.execute(s)
            except _errors(env.family) as exc:
                failed, error = True, exc
            gen.coverage[f"outcome:{kind}:{'error' if failed else 'ok'}"] += 1
            if expect == "error":
                assert failed, "the statement was expected to fail"
            if expect in ("commit", "rollback") and failed:
                raise AssertionError(f"step {step}: unexpected statement error {error!r}")
            if failed or expect == "rollback":
                if not failed:
                    env.check(f"step {step} before rollback", end_tx=False)
                env.rollback()
            else:
                env.check(f"step {step} in transaction", end_tx=False)
                env.commit()
            env.check(f"step {step} ({kind})")
    except AssertionError as exc:
        raise type(exc)(f"{exc}\nseed={seed} {_replay_hint(target, seed)}\nstatements:\n" + "\n".join(log[-8:])) from None
    _RANDOM_COVERAGE[target.id].update(gen.coverage)


@pytest.mark.parametrize("target", _params())
def test_randomized_generator_production_coverage(target: Target) -> None:
    """Every production reachable on the engine family, every key distribution
    and every row-count class was generated by the retained seeds. Must run
    after the randomized test in the same session (default file order)."""
    if os.environ.get("MATVIEW_BULK_SEEDS"):
        pytest.skip("coverage is asserted for the retained seeds only")
    cov = _RANDOM_COVERAGE.get(target.id)
    if not cov:
        pytest.skip("randomized test did not run in this session")
    kinds = [k for k, fams in KINDS.items() if target.family in fams]
    missing = [k for k in kinds if cov[k] == 0] + [f"dist:{d}" for d in DISTRIBUTIONS if cov[f"dist:{d}"] == 0] + [
        f"rows:{n}" for n in ROW_COUNTS if cov[f"rows:{n}"] == 0]
    print(f"[bulk generator coverage {target.id}] " + ", ".join(f"{k}={v}" for k, v in sorted(cov.items())))
    assert not missing, f"unreached productions: {missing}"
