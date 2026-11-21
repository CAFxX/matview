"""Native DML / trigger-interaction support matrix (differential).

Question answered: for every way rows of a source table can change, and for
every way other database objects or settings can interfere with trigger-based
maintenance, does the materialized view stay exactly equal to its original
query, fail closed, or diverge?

Each *cell* of the matrix (``CELLS``) is one interaction scenario with one
recorded outcome per product (``sqlite``, ``postgres``, ``mysql``,
``mariadb``):

* ``SUPPORTED-EXACT``: the scenario runs and every view equals its native
  query after every step (in the open transaction and after commit).
* ``SUPPORTED-WITH-LIMITATION``: the documented limitation is asserted (for
  example: out-of-band DDL leaves the view stale but ``check_schema_drift()``
  reports it and ``repair_view()`` restores exactness, or a documented engine
  setting changes user-visible semantics).
* ``DETECTED-REFUSED``: the product or the engine fails closed with an error
  (registration refused, or the DML statement errors and is rolled back); the
  test asserts the error and that the views are unchanged.
* ``SILENTLY-WRONG``: a real defect. The cell must raise ``ViewMismatch``; it
  is reported as an expected failure that turns into a test failure as soon as
  the view stays exact (strict). Each names its TODO.md item.
* ``N/A``: the engine has no such feature; not run.

The README section "DML and trigger interaction support" is generated from
``CELLS`` (``python3 test_native_dml_interactions.py --print-matrix``) and
``test_readme_matrix_is_current`` fails when it drifts, so the published
matrix is exactly what these tests assert.

Oracle: the original ``SELECT`` of each view, executed natively on the same
connection (``test_native_bulk_statements.Env.check``: exact multiset
comparison with value types, Decimal scale and driver column type codes; no
tolerance). Cells marked "twin" additionally run each statement on an
identical source table without views (``t0``/``s0``) and require the same
affected-row count, the same ``RETURNING`` rows and identical final table
contents: maintenance must not perturb the statement's own semantics.

Engines: as in ``test_native_bulk_statements.py`` (``POSTGRES_DSN``,
``MYSQL_DSN`` (MySQL or MariaDB, read from the server), otherwise SQLite
through the standard-library ``sqlite3`` driver on a file database). The
SQLite compatibility drivers (APSW, SQLCipher, libSQL) are not run by this
module.

Runtime: one shared environment per target (reset by plain DML between
cells) plus a small dedicated environment for cells that need their own
schema; no sleeps.
"""

from __future__ import annotations

import collections
import io
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pytest

import test_native_bulk_statements as h
from matview import EngineLimitationError
from test_native_bulk_statements import TARGETS, Env, Target, ViewDef, ViewMismatch

EXACT = "SUPPORTED-EXACT"
LIMITED = "SUPPORTED-WITH-LIMITATION"
REFUSED = "DETECTED-REFUSED"
WRONG = "SILENTLY-WRONG"
NA = "N/A"
OUTCOMES = (EXACT, LIMITED, REFUSED, WRONG, NA)
PRODUCTS = ("sqlite", "postgres", "mysql", "mariadb")
FAMILY_PRODUCTS = {"sqlite": ("sqlite",), "postgres": ("postgres",), "mysql": ("mysql", "mariadb")}

README_BEGIN = "<!-- BEGIN GENERATED: dml-matrix -->"
README_END = "<!-- END GENERATED: dml-matrix -->"

# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    id: str
    area: str
    what: str
    outcomes: Tuple[Tuple[str, str], ...]
    run: Callable[["Ctx"], None]
    note: str
    since: Tuple[Tuple[str, Tuple[int, int]], ...]
    todo: str  # TODO.md item title (required for SILENTLY-WRONG)
    # Products whose outcome was verified only outside CI (the PR-gate server
    # lacks a setting the cell needs; the test skips there).
    local_only: Tuple[str, ...] = ()

    def outcome(self, product: str) -> str:
        return dict(self.outcomes).get(product, NA)

    def min_version(self, product: str) -> Optional[Tuple[int, int]]:
        return dict(self.since).get(product)


CELLS: List[Cell] = []

AREA_A = "A. Row-changing statements"
AREA_B = "B. Other triggers, objects and settings"
AREA_C = "C. Transactions"


def cell(cid: str, area: str, what: str, note: str = "", since: Optional[Dict[str, Tuple[int, int]]] = None,
         todo: str = "", local_only: Tuple[str, ...] = (), **outcomes: str) -> Callable[[Callable[["Ctx"], None]], Callable[["Ctx"], None]]:
    for p, o in outcomes.items():
        assert p in PRODUCTS and o in OUTCOMES, (cid, p, o)

    def deco(fn: Callable[["Ctx"], None]) -> Callable[["Ctx"], None]:
        CELLS.append(Cell(cid, area, what, tuple(outcomes.items()), fn, note,
                          tuple((since or {}).items()), todo, tuple(local_only)))
        return fn

    return deco


# ---------------------------------------------------------------------------
# Schema, views, environments
# ---------------------------------------------------------------------------

# t: source table with a non-PK UNIQUE column u and a CHECK (deterministic
# mid-statement failures); t0: identical table without views (the "twin");
# d: dimension joined on g.
_BASE_COLS = "id {i} PRIMARY KEY, g {i}, v {b}, w {i}, u {i} UNIQUE, CONSTRAINT {n}_v_bound CHECK (v IS NULL OR v < 1000000)"
BASE_SCHEMA: Dict[str, List[str]] = {
    "postgres": [
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
        "CREATE TABLE t (" + _BASE_COLS.format(i="INT", b="BIGINT", n="t") + ")",
        "CREATE TABLE t0 (" + _BASE_COLS.format(i="INT", b="BIGINT", n="t0") + ")",
        "CREATE INDEX t_g_v ON t (g, v)",
    ],
    "mysql": [
        "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL) ENGINE=InnoDB",
        "CREATE TABLE t (" + _BASE_COLS.format(i="INT", b="BIGINT", n="t") + ") ENGINE=InnoDB",
        "CREATE TABLE t0 (" + _BASE_COLS.format(i="INT", b="BIGINT", n="t0") + ") ENGINE=InnoDB",
        "CREATE INDEX t_g_v ON t (g, v)",
    ],
    "sqlite": [
        "CREATE TABLE d (g INTEGER PRIMARY KEY, name VARCHAR(20) NOT NULL)",
        "CREATE TABLE t (" + _BASE_COLS.format(i="INTEGER", b="INTEGER", n="t") + ")",
        "CREATE TABLE t0 (" + _BASE_COLS.format(i="INTEGER", b="INTEGER", n="t0") + ")",
        "CREATE INDEX t_g_v ON t (g, v)",
    ],
}


def _base_rows() -> List[Tuple[Any, ...]]:
    rows = []
    for i in range(1, 25):
        g = None if i % 13 == 0 else i % 5
        v = None if i % 11 == 0 else (i * 37) % 101 - 30
        w = None if i % 7 == 0 else i % 4
        u = None if i % 8 == 0 else i * 10
        rows.append((i, g, v, w, u))
    return rows


def _values(rows: Sequence[Sequence[Any]]) -> str:
    return ", ".join("(" + ", ".join(h._sql_lit(x) for x in r) + ")" for r in rows)


BASE_D = [(g, f"n{g % 3}") for g in range(6)]

V_AGG = ViewDef("v_agg", "SELECT g, SUM(v) AS s, COUNT(*) AS c, COUNT(v) AS cv FROM t GROUP BY g")
V_MM = ViewDef("v_mm", "SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM t GROUP BY g")
V_PR = ViewDef("v_pr", "SELECT id, g, v, u FROM t WHERE w > 0")
V_J = ViewDef("v_j", "SELECT d.name, SUM(t.v) AS s, COUNT(*) AS c FROM t JOIN d ON t.g = d.g GROUP BY d.name", join=True)
V_PJ = ViewDef("v_pj", "SELECT t.id, t.v, d.name FROM t JOIN d ON t.g = d.g", join=True)
BASE_VIEWS = [V_AGG, V_MM, V_PR, V_J, V_PJ]


class DEnv(Env):
    """An Env whose schema and seed rows are given by the cell."""

    def __init__(self, target: Target, views: Sequence[ViewDef], schema: List[str], seed_sql: Sequence[str],
                 ignore_limitations: Optional[bool] = None) -> None:
        self._seed_sql = list(seed_sql)
        super().__init__(target, views=views, schema_sql=schema, seed=True, ignore_limitations=ignore_limitations)

    def seed(self) -> None:
        cur = self.conn.cursor()
        for sql in self._seed_sql:
            cur.execute(sql)
        self.conn.commit()

    def db_path(self) -> str:
        assert self._tmpdir is not None
        return os.path.join(self._tmpdir, "db.sqlite3")

    def raw_sqlite(self, configure: bool) -> sqlite3.Connection:
        """An application writer connection (optionally NOT configured with
        configure_sqlite_connection)."""
        conn = sqlite3.connect(self.db_path(), timeout=30)
        if configure:
            from matview import MaterializedViewManager

            MaterializedViewManager.configure_sqlite_connection(conn)
        conn.execute("PRAGMA foreign_keys = ON")
        self._extra_conns.append(conn)
        return conn

    def fingerprint(self) -> Tuple[Any, ...]:
        """Objects and trigger states of the namespace (a cell must leave
        the shared environment exactly as it found it)."""
        cur = self.conn.cursor()
        if self.family == "postgres":
            cur.execute(
                "SELECT c.relname, c.relkind, tg.tgname, tg.tgenabled FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "LEFT JOIN pg_trigger tg ON tg.tgrelid = c.oid AND NOT tg.tgisinternal "
                "WHERE n.nspname = current_schema() ORDER BY 1, 2, 3")
            rows = [tuple(r) for r in cur.fetchall()]
            cur.execute("SELECT count(*) FROM pg_rewrite r JOIN pg_class c ON c.oid = r.ev_class "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = current_schema() AND r.rulename <> '_RETURN'")
            rows.append(("rules", cur.fetchone()[0]))
        elif self.family == "mysql":
            cur.execute("SELECT table_name, table_type FROM information_schema.tables "
                        "WHERE table_schema = DATABASE() ORDER BY 1")
            rows = [tuple(r) for r in cur.fetchall()]
            cur.execute("SELECT trigger_name, action_order FROM information_schema.triggers "
                        "WHERE trigger_schema = DATABASE() ORDER BY 1")
            rows += [tuple(r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT type, name FROM sqlite_master ORDER BY 1, 2")
            rows = [tuple(r) for r in cur.fetchall()]
        self.conn.commit()
        return tuple(rows)


def _base_seed() -> List[str]:
    rows = _values(_base_rows())
    return [f"INSERT INTO d (g, name) VALUES {_values(BASE_D)}",
            f"INSERT INTO t (id, g, v, w, u) VALUES {rows}",
            f"INSERT INTO t0 (id, g, v, w, u) VALUES {rows}"]


class BaseEnv(DEnv):
    def __init__(self, target: Target) -> None:
        super().__init__(target, BASE_VIEWS, BASE_SCHEMA[target.family], _base_seed())
        self.fp = self.fingerprint()

    def reset(self) -> None:
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
        for tbl in ("t", "t0", "d"):
            self.execute(f"DELETE FROM {tbl}")
        self.commit()
        self.seed()
        self.check("reset to the seed state")
        if self.fingerprint() != self.fp:
            raise RuntimeError("a cell left objects behind in the shared environment")


_BASE: Dict[str, BaseEnv] = {}


def _base_env(target: Target) -> BaseEnv:
    env = _BASE.get(target.id)
    if env is not None:
        try:
            env.reset()
            return env
        except Exception:
            del _BASE[target.id]
            env.close()
    env = BaseEnv(target)
    _BASE[target.id] = env
    return env


@pytest.fixture(scope="module", autouse=True)
def _close_base_envs():
    yield
    while _BASE:
        _BASE.popitem()[1].close()


# ---------------------------------------------------------------------------
# Cell context
# ---------------------------------------------------------------------------


def _exact_rows(cur: Any) -> Optional[collections.Counter]:
    if cur.description is None:
        return None
    return collections.Counter(tuple(h._exact(v) for v in r) for r in cur.fetchall())


class Ctx:
    def __init__(self, target: Target, c: Cell) -> None:
        self.target = target
        self.cell = c
        self.family = target.family
        self.env: BaseEnv = _base_env(target)
        self.product = self.env.product
        self.version = self.env.version
        self.outcome = c.outcome(self.product)
        self._fresh: List[DEnv] = []
        self._cleanup: List[str] = []
        self.twins: List[Tuple[str, str]] = []

    # -- dialect helpers -----------------------------------------------------
    def pick(self, **variants: str) -> str:
        sql = variants.get(self.product)
        if sql is None:
            sql = variants.get(self.family)
        assert sql is not None, (self.cell.id, self.product, sorted(variants))
        return sql

    @property
    def errors(self) -> Tuple[type, ...]:
        return h._errors(self.family)

    # -- environments ----------------------------------------------------------
    def fresh(self, schema: Dict[str, List[str]], views: Sequence[ViewDef], seed: Sequence[str] = (),
              ignore_limitations: Optional[bool] = None) -> DEnv:
        env = DEnv(self.target, views, schema[self.product] if self.product in schema else schema[self.family],
                   seed, ignore_limitations)
        self._fresh.append(env)
        return env

    def cleanup(self, *sql: str) -> None:
        """SQL run after the cell (also on failure), newest first."""
        self._cleanup.extend(sql)

    def finish(self) -> None:
        env = self.env
        for conn in env._extra_conns:
            if conn is env.conn:
                continue
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
        env._extra_conns = [env.conn]
        env.conn.rollback()
        cur = env.conn.cursor()
        for sql in reversed(self._cleanup):
            cur.execute(sql)
        env.conn.commit()
        # A cell that left the shared views stale (a SILENTLY-WRONG cell) is
        # caught by the next reset, which then rebuilds the environment.
        for f in self._fresh:
            f.close()

    # -- statements --------------------------------------------------------------
    def run(self, sql: str, env: Optional[Env] = None, conn: Any = None) -> Tuple[int, Optional[collections.Counter]]:
        e = env or self.env
        cur = (conn or e.conn).cursor()
        cur.execute(sql)
        return cur.rowcount, _exact_rows(cur)

    def both(self, tmpl: str, env: Optional[Env] = None, pair: Tuple[str, str] = ("t", "t0"), conn: Any = None,
             compare_rowcount: bool = True) -> None:
        """Run `tmpl` (with ``{t}`` for the table) on the twin without views
        and then on the source table; the affected-row count and the returned
        rows must be identical."""
        name, twin = pair
        if (name, twin) not in self.twins:
            self.twins.append((name, twin))
        r0 = self.run(tmpl.replace("{t}", twin), env, conn)
        r1 = self.run(tmpl.replace("{t}", name), env, conn)
        if not compare_rowcount:
            r0, r1 = (0, r0[1]), (0, r1[1])
        assert r1 == r0, f"maintenance perturbed the statement result: {r1} != twin {r0} for {tmpl}"

    def fails(self, sql: str, env: Optional[Env] = None, conn: Any = None, match: str = "") -> BaseException:
        """`sql` must fail; on PostgreSQL inside a savepoint so the open
        transaction survives (other engines roll back the statement only)."""
        e = env or self.env
        c = conn or e.conn
        pg = self.family == "postgres"
        if pg:
            e.execute("SAVEPOINT au_fail", c)
        with pytest.raises(self.errors) as info:
            e.execute(sql, c)
        if pg:
            e.execute("ROLLBACK TO SAVEPOINT au_fail", c)
        if match:
            assert re.search(match, str(info.value), re.I), (match, str(info.value))
        return info.value

    def both_fail(self, tmpl: str, env: Optional[Env] = None, pair: Tuple[str, str] = ("t", "t0")) -> None:
        name, twin = pair
        if (name, twin) not in self.twins:
            self.twins.append((name, twin))
        e0 = self.fails(tmpl.replace("{t}", twin), env)
        e1 = self.fails(tmpl.replace("{t}", name), env)
        assert type(e0) is type(e1), (e0, e1)

    def check(self, label: str, env: Optional[Env] = None, end_tx: bool = True, conn: Any = None) -> None:
        e = env or self.env
        c = conn or e.conn
        for name, twin in self.twins:
            if e is not self.env and name in ("t",):
                continue
            cur = c.cursor()
            a = h._relation(cur, f"SELECT * FROM {name}")
            b = h._relation(cur, f"SELECT * FROM {twin}")
            assert a == b, f"after {label}: maintenance perturbed the base data of {name} (vs twin {twin})"
        e.check(f"{self.cell.id}: {label}", conn=c, end_tx=end_tx)

    def commit_check(self, label: str, env: Optional[Env] = None) -> None:
        self.check(f"{label} (in transaction)", env, end_tx=False)
        (env or self.env).commit()
        self.check(label, env)

    def other_conn(self, env: Optional[Env] = None) -> Any:
        return (env or self.env).connect()


# ---------------------------------------------------------------------------
# A. Row-changing statements
# ---------------------------------------------------------------------------

UPSERT_ROWS = "(1, 3, 77, 1, 10), (2, NULL, NULL, 2, 20), (504, 1, 9, 1, 5040), (505, NULL, -4, 3, NULL)"


@cell("upsert_do_nothing", AREA_A, "INSERT ... ON CONFLICT DO NOTHING (PK and non-PK unique) / INSERT IGNORE",
      "twin; conflict target on the PK, on the UNIQUE column, and untargeted",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _upsert_do_nothing(ctx: Ctx) -> None:
    rows = "(1, 2, 5, 1, 777), (501, 1, 5, 1, 20), (502, 1, 6, 1, 5020), (503, NULL, 7, 2, NULL)"
    if ctx.family == "mysql":
        ctx.both(f"INSERT IGNORE INTO {{t}} (id, g, v, w, u) VALUES {rows}")
    else:
        ctx.both(f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} ON CONFLICT DO NOTHING")
        ctx.both("INSERT INTO {t} (id, g, v, w, u) VALUES (506, 4, 1, 1, 30), (507, 4, 2, 1, 5070) "
                 "ON CONFLICT (u) DO NOTHING")
        ctx.both("INSERT INTO {t} (id, g, v, w, u) VALUES (3, 4, 1, 1, 9999), (508, 4, 2, 1, NULL) "
                 "ON CONFLICT (id) DO NOTHING")
    ctx.commit_check("upsert do nothing")


@cell("upsert_do_update", AREA_A,
      "ON CONFLICT DO UPDATE with EXCLUDED / ON DUPLICATE KEY UPDATE (row alias, VALUES())",
      "twin; updates move rows between groups, to and from NULL",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _upsert_do_update(ctx: Ctx) -> None:
    ctx.both(ctx.pick(
        postgres=f"INSERT INTO {{t}} AS x (id, g, v, w, u) VALUES {UPSERT_ROWS} "
                 "ON CONFLICT (id) DO UPDATE SET g = EXCLUDED.g, v = x.v + EXCLUDED.v, w = EXCLUDED.w",
        sqlite=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} "
               "ON CONFLICT (id) DO UPDATE SET g = excluded.g, v = {t}.v + excluded.v, w = excluded.w",
        mysql=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} AS new "
              "ON DUPLICATE KEY UPDATE g = new.g, v = {t}.v + new.v, w = new.w",
        mariadb=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} "
                "ON DUPLICATE KEY UPDATE g = VALUES(g), v = v + VALUES(v), w = VALUES(w)",
    ))
    ctx.commit_check("upsert do update")


@cell("upsert_where_false", AREA_A,
      "ON CONFLICT DO UPDATE ... WHERE false / ON DUPLICATE KEY UPDATE id = id (no-op)",
      "twin; conflicting rows are untouched, others inserted",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _upsert_where_false(ctx: Ctx) -> None:
    ctx.both(ctx.pick(
        postgres=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} ON CONFLICT (id) DO UPDATE SET v = 0 WHERE false",
        sqlite=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} ON CONFLICT (id) DO UPDATE SET v = 0 WHERE 0",
        mysql=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} ON DUPLICATE KEY UPDATE id = id",
    ))
    ctx.commit_check("upsert where false")


@cell("upsert_where_partial", AREA_A, "ON CONFLICT DO UPDATE ... WHERE <predicate on the existing row>",
      "twin; MySQL/MariaDB express the predicate with IF() in the assignments",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _upsert_where_partial(ctx: Ctx) -> None:
    rows = "(1, 3, 1, 1, 10), (2, 3, 2, 1, 20), (3, 3, 3, 1, 30), (4, 3, 4, 1, 40), (5, 3, 5, 1, 50), (509, 3, 6, 1, NULL)"
    ctx.both(ctx.pick(
        postgres=f"INSERT INTO {{t}} AS x (id, g, v, w, u) VALUES {rows} "
                 "ON CONFLICT (id) DO UPDATE SET g = EXCLUDED.g, v = EXCLUDED.v WHERE x.v > 0",
        sqlite=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} "
               "ON CONFLICT (id) DO UPDATE SET g = excluded.g, v = excluded.v WHERE {t}.v > 0",
        mysql=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} AS new ON DUPLICATE KEY UPDATE "
              "g = IF({t}.v > 0, new.g, {t}.g), v = IF({t}.v > 0, new.v, {t}.v)",
        mariadb=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} ON DUPLICATE KEY UPDATE "
                "g = IF(v > 0, VALUES(g), g), v = IF(v > 0, VALUES(v), v)",
    ))
    ctx.commit_check("upsert with a partial WHERE")


@cell("upsert_unique_non_pk", AREA_A, "Upsert conflicting on a non-PK UNIQUE column that rewrites the primary key",
      "twin; the projection view carries the key",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _upsert_unique_non_pk(ctx: Ctx) -> None:
    rows = "(700, 2, 5, 1, 10), (701, 3, 6, 1, 20), (702, 1, 1, 1, 7020)"
    ctx.both(ctx.pick(
        postgres=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} "
                 "ON CONFLICT (u) DO UPDATE SET id = EXCLUDED.id, g = EXCLUDED.g, v = EXCLUDED.v",
        sqlite=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} "
               "ON CONFLICT (u) DO UPDATE SET id = excluded.id, g = excluded.g, v = excluded.v",
        mysql=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} AS new "
              "ON DUPLICATE KEY UPDATE id = new.id, g = new.g, v = new.v",
        mariadb=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} "
                "ON DUPLICATE KEY UPDATE id = VALUES(id), g = VALUES(g), v = VALUES(v)",
    ))
    ctx.commit_check("upsert on a non-PK unique key")


@cell("upsert_duplicate_keys_in_statement", AREA_A,
      "One upsert statement whose rows repeat the same key",
      "twin; PostgreSQL rejects DO UPDATE touching a row twice (SQLSTATE 21000) and the view is unchanged; "
      "DO NOTHING, SQLite and MySQL/MariaDB apply the rows in order",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _upsert_duplicate_keys(ctx: Ctx) -> None:
    rows = "(800, 1, 1, 1, NULL), (800, 2, 2, 1, NULL), (1, 3, 3, 1, NULL), (1, 4, 4, 2, NULL)"
    if ctx.family == "postgres":
        ctx.both_fail(f"INSERT INTO {{t}} AS x (id, g, v, w, u) VALUES {rows} "
                      "ON CONFLICT (id) DO UPDATE SET g = EXCLUDED.g, v = x.v + EXCLUDED.v")
        ctx.check("rejected DO UPDATE", end_tx=False)
        ctx.both(f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} ON CONFLICT (id) DO NOTHING")
    else:
        ctx.both(ctx.pick(
            sqlite=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} "
                   "ON CONFLICT (id) DO UPDATE SET g = excluded.g, v = {t}.v + excluded.v, w = excluded.w",
            mysql=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} AS new "
                  "ON DUPLICATE KEY UPDATE g = new.g, v = {t}.v + new.v, w = new.w",
            mariadb=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} "
                    "ON DUPLICATE KEY UPDATE g = VALUES(g), v = v + VALUES(v), w = VALUES(w)",
        ))
    ctx.commit_check("upsert with repeated keys")


@cell("merge_all_branches", AREA_A,
      "MERGE with WHEN MATCHED [AND] UPDATE / DELETE / DO NOTHING and WHEN NOT MATCHED [AND] INSERT / DO NOTHING",
      "twin", since={"postgres": (15, 0)}, postgres=EXACT)
def _merge_all(ctx: Ctx) -> None:
    ctx.both(
        "MERGE INTO {t} AS x USING (VALUES (1, 4, NULL::bigint, 1), (2, 2, 50::bigint, 1), (3, NULL, 7::bigint, 1), "
        "(4, 1, 1::bigint, 9), (900, 1, 3::bigint, 1), (901, 6, -3::bigint, 0), (902, 6, 1::bigint, 1)) "
        "AS s(id, g, v, k) ON x.id = s.id "
        "WHEN MATCHED AND s.k = 9 THEN DO NOTHING "
        "WHEN MATCHED AND s.v IS NULL THEN DELETE "
        "WHEN MATCHED THEN UPDATE SET g = s.g, v = s.v "
        "WHEN NOT MATCHED AND s.k = 0 THEN DO NOTHING "
        "WHEN NOT MATCHED THEN INSERT (id, g, v, w, u) VALUES (s.id, s.g, s.v, 1, NULL)")
    ctx.commit_check("MERGE")


@cell("merge_by_source_returning", AREA_A, "MERGE ... WHEN NOT MATCHED BY SOURCE ... RETURNING merge_action()",
      "twin (returned rows compared)", since={"postgres": (17, 0)}, postgres=EXACT)
def _merge_by_source(ctx: Ctx) -> None:
    ctx.both(
        "MERGE INTO {t} AS x USING (SELECT id, g FROM {t} WHERE id % 3 = 0) AS s ON x.id = s.id "
        "WHEN MATCHED THEN UPDATE SET v = x.v + 1 "
        "WHEN NOT MATCHED BY SOURCE AND x.g = 2 THEN DELETE "
        "WHEN NOT MATCHED BY SOURCE AND x.g = 3 THEN UPDATE SET w = 3 "
        "RETURNING merge_action(), x.id, x.v")
    ctx.commit_check("MERGE BY SOURCE")


@cell("sqlite_replace_configured", AREA_A,
      "INSERT OR REPLACE / REPLACE / UPDATE OR REPLACE deleting PK- and UNIQUE-conflicting rows",
      "writer connection configured with configure_sqlite_connection() (recursive_triggers ON); twin",
      sqlite=EXACT)
def _sqlite_replace(ctx: Ctx) -> None:
    ctx.both("INSERT OR REPLACE INTO {t} (id, g, v, w, u) VALUES (1, 4, 5, 1, 20), (600, 1, 1, 1, 30)")
    ctx.both("REPLACE INTO {t} (id, g, v, w, u) VALUES (601, 2, 2, 2, 40), (5, NULL, NULL, 1, 60)")
    ctx.both("UPDATE OR REPLACE {t} SET u = 70 WHERE id = 9")
    ctx.commit_check("REPLACE family")


@cell("sqlite_replace_unconfigured_writer", AREA_A,
      "INSERT OR REPLACE / REPLACE through a writer connection with recursive_triggers OFF (SQLite default)",
      "the replaced rows' DELETE triggers do not fire, so the view keeps them (documented under SQLite writer "
      "connections; nothing detects it)",
      todo="Close Unsafe DML Paths That Bypass Trigger-Based Maintenance", sqlite=WRONG)
def _sqlite_replace_unconfigured(ctx: Ctx) -> None:
    w = ctx.env.raw_sqlite(configure=False)
    assert w.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
    w.execute("INSERT OR REPLACE INTO t (id, g, v, w, u) VALUES (1, 4, 5, 1, 20)")
    w.commit()
    ctx.check("REPLACE on an unconfigured connection")


@cell("sqlite_table_on_conflict_replace", AREA_A,
      "Plain INSERT/UPDATE into a table whose UNIQUE constraint declares ON CONFLICT REPLACE",
      "configured writer connection", sqlite=EXACT)
def _sqlite_table_replace(ctx: Ctx) -> None:
    env = ctx.fresh(
        {"sqlite": ["CREATE TABLE r (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, u INTEGER UNIQUE ON CONFLICT REPLACE)"]},
        [ViewDef("v_r", "SELECT g, SUM(v) AS s, COUNT(*) AS c FROM r GROUP BY g"),
         ViewDef("v_rp", "SELECT id, u FROM r WHERE v > 0")],
        ["INSERT INTO r VALUES " + _values([(i, i % 3, i * 3 - 7, i) for i in range(1, 13)])])
    env.execute("INSERT INTO r (id, g, v, u) VALUES (100, 5, 9, 3), (101, 5, 8, 4)")
    env.execute("UPDATE r SET u = 7 WHERE id = 8")
    ctx.commit_check("table-level ON CONFLICT REPLACE", env)


@cell("sqlite_or_ignore", AREA_A, "INSERT OR IGNORE / UPDATE OR IGNORE (PK, UNIQUE and CHECK conflicts)",
      "twin", sqlite=EXACT)
def _sqlite_or_ignore(ctx: Ctx) -> None:
    ctx.both("INSERT OR IGNORE INTO {t} (id, g, v, w, u) VALUES (1, 1, 1, 1, NULL), (610, 1, 5000000, 1, NULL), "
             "(611, 2, 3, 1, 6110), (612, 2, 4, 1, 10)")
    ctx.both("UPDATE OR IGNORE {t} SET u = 10, g = 4 WHERE id IN (2, 3)")
    ctx.both("UPDATE OR IGNORE {t} SET v = CASE WHEN id = 4 THEN 9999999 ELSE v + 1 END WHERE id < 6")
    ctx.commit_check("OR IGNORE")


@cell("sqlite_or_abort_fail", AREA_A,
      "INSERT/UPDATE OR ABORT (statement rolled back) and OR FAIL (earlier rows of the statement kept)",
      "twin; failures inside an open transaction", sqlite=EXACT)
def _sqlite_or_abort_fail(ctx: Ctx) -> None:
    ctx.both("INSERT INTO {t} (id, g, v, w, u) VALUES (619, 1, 1, 1, NULL)")
    ctx.both_fail("INSERT OR ABORT INTO {t} (id, g, v, w, u) VALUES (620, 1, 1, 1, NULL), (1, 1, 1, 1, NULL)")
    ctx.check("OR ABORT", end_tx=False)
    ctx.both_fail("INSERT OR FAIL INTO {t} (id, g, v, w, u) VALUES (621, 2, 5, 1, NULL), (622, 3, 6, 1, NULL), "
                  "(1, 1, 1, 1, NULL)")
    ctx.check("OR FAIL keeps the earlier rows", end_tx=False)
    ctx.both_fail("UPDATE OR FAIL {t} SET u = 10 WHERE id >= 2")
    ctx.commit_check("UPDATE OR FAIL")


@cell("sqlite_or_rollback", AREA_A, "INSERT OR ROLLBACK (the whole transaction is rolled back)", "",
      sqlite=EXACT)
def _sqlite_or_rollback(ctx: Ctx) -> None:
    env = ctx.env
    before = env.snapshot()
    env.commit()
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (630, 1, 1, 1, NULL)")
    ctx.fails("INSERT OR ROLLBACK INTO t (id, g, v, w, u) VALUES (631, 1, 1, 1, NULL), (1, 1, 1, 1, NULL)")
    assert not env.conn.in_transaction
    ctx.check("OR ROLLBACK")
    assert env.snapshot() == before


@cell("mysql_replace", AREA_A, "REPLACE deleting PK- and UNIQUE-conflicting rows (VALUES and SELECT forms)",
      "twin", mysql=EXACT, mariadb=EXACT)
def _mysql_replace(ctx: Ctx) -> None:
    ctx.both("REPLACE INTO {t} (id, g, v, w, u) VALUES (1, 4, 5, 1, 20), (600, 1, 1, 1, 30), (601, 2, 2, 2, NULL)")
    ctx.both("REPLACE INTO {t} (id, g, v, w, u) SELECT id + 1000, g, v, w, NULL FROM {t} WHERE id < 6")
    ctx.commit_check("REPLACE")


@cell("mysql_insert_ignore_data_errors", AREA_A,
      "INSERT IGNORE / UPDATE IGNORE converting CHECK, range and duplicate-key errors to warnings",
      "twin", mysql=EXACT, mariadb=EXACT)
def _mysql_insert_ignore(ctx: Ctx) -> None:
    ctx.both("INSERT IGNORE INTO {t} (id, g, v, w, u) VALUES (640, 1, 5000000, 1, NULL), "
             "(641, 1, 3, 99999999999, NULL), (642, NULL, 1, 1, 10), (643, 2, 2, 2, 6430)")
    ctx.both("UPDATE IGNORE {t} SET v = CASE WHEN id = 4 THEN 5000000 ELSE v + 1 END, w = 99999999999 WHERE id < 7")
    ctx.commit_check("INSERT/UPDATE IGNORE")


@cell("mysql_update_ignore_duplicate_key", AREA_A,
      "UPDATE IGNORE whose rows hit a duplicate key on a UNIQUE column and are skipped",
      "MariaDB fires the AFTER UPDATE trigger for a row whose update was ignored (reproduced without matview: "
      "the trigger logs a change the table never had), so the view applies it; MySQL does not fire it",
      todo="Exact Maintenance Under MariaDB UPDATE IGNORE (AFTER UPDATE Triggers Fire for Ignored Rows)",
      mysql=EXACT, mariadb=WRONG)
def _mysql_update_ignore_dup(ctx: Ctx) -> None:
    ctx.both("UPDATE IGNORE {t} SET u = 10, g = 4 WHERE id IN (2, 3)")
    ctx.commit_check("UPDATE IGNORE skipping duplicate keys")


_S_COLS = {"postgres": "id INT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, g INT DEFAULT 3, v BIGINT DEFAULT 7, k INT UNIQUE",
           "mysql": "id INT AUTO_INCREMENT PRIMARY KEY, g INT DEFAULT 3, v BIGINT DEFAULT 7, k INT UNIQUE",
           "sqlite": "id INTEGER PRIMARY KEY AUTOINCREMENT, g INTEGER DEFAULT 3, v INTEGER DEFAULT 7, k INTEGER UNIQUE"}
_S_SCHEMA = {f: [f"CREATE TABLE s ({c})" + (" ENGINE=InnoDB" if f == "mysql" else ""),
                 f"CREATE TABLE s0 ({c})" + (" ENGINE=InnoDB" if f == "mysql" else "")] for f, c in _S_COLS.items()}


@cell("defaults_identity_upsert_gaps", AREA_A,
      "Column defaults, DEFAULT VALUES, identity/AUTO_INCREMENT keys and the id gaps upserts leave",
      "twin incl. affected rows and lastval()/LAST_INSERT_ID()/last_insert_rowid() after each statement",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _defaults_identity(ctx: Ctx) -> None:
    env = ctx.fresh(_S_SCHEMA, [ViewDef("v_s", "SELECT g, SUM(v) AS s, COUNT(*) AS c, MAX(id) AS mx FROM s GROUP BY g"),
                                ViewDef("v_sp", "SELECT id, k, v FROM s WHERE g = 3")])
    last = ctx.pick(postgres="SELECT lastval()", mysql="SELECT LAST_INSERT_ID()", sqlite="SELECT last_insert_rowid()")
    stmts = [
        "INSERT INTO {t} (k) VALUES (1), (2), (3)",
        ctx.pick(postgres="INSERT INTO {t} DEFAULT VALUES", sqlite="INSERT INTO {t} DEFAULT VALUES",
                 mysql="INSERT INTO {t} () VALUES ()"),
        ctx.pick(postgres="INSERT INTO {t} (k, v) VALUES (1, 1), (4, 4) ON CONFLICT (k) DO UPDATE SET v = {t}.v + EXCLUDED.v",
                 sqlite="INSERT INTO {t} (k, v) VALUES (1, 1), (4, 4) ON CONFLICT (k) DO UPDATE SET v = {t}.v + excluded.v",
                 mysql="INSERT INTO {t} (k, v) VALUES (1, 1), (4, 4) AS new ON DUPLICATE KEY UPDATE v = {t}.v + new.v",
                 mariadb="INSERT INTO {t} (k, v) VALUES (1, 1), (4, 4) ON DUPLICATE KEY UPDATE v = v + VALUES(v)"),
        ctx.pick(postgres="INSERT INTO {t} (k) VALUES (2), (5) ON CONFLICT DO NOTHING",
                 sqlite="INSERT OR IGNORE INTO {t} (k) VALUES (2), (5)",
                 mysql="INSERT IGNORE INTO {t} (k) VALUES (2), (5)"),
        "INSERT INTO {t} (k, g) VALUES (6, 1)",
    ]
    if ctx.family != "sqlite":
        stmts.append("UPDATE {t} SET v = DEFAULT, g = DEFAULT WHERE k IN (1, 6)")
    ctx.twins.append(("s", "s0"))
    for sql in stmts:
        # Same statement on the twin and on the source: same affected rows and,
        # right after it, the same session "last generated id".
        r0 = ctx.run(sql.replace("{t}", "s0"), env)
        l0 = ctx.run(last, env)
        r1 = ctx.run(sql.replace("{t}", "s"), env)
        l1 = ctx.run(last, env)
        assert (r1, l1) == (r0, l0), f"maintenance perturbed {sql}: {(r1, l1)} != twin {(r0, l0)}"
    ctx.commit_check("defaults, identity and gaps", env)


@cell("update_delete_self_join", AREA_A,
      "UPDATE ... FROM / multi-table UPDATE and DELETE ... USING / multi-table DELETE joining the source to itself",
      "twin", sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _self_join_dml(ctx: Ctx) -> None:
    ctx.both(ctx.pick(
        postgres="UPDATE {t} SET v = s.v + 1, g = s.g FROM {t} AS s WHERE {t}.id = s.id + 1 AND s.w = 1",
        sqlite="UPDATE {t} SET v = s.v + 1, g = s.g FROM {t} AS s WHERE {t}.id = s.id + 1 AND s.w = 1",
        mysql="UPDATE {t} JOIN {t} AS s ON {t}.id = s.id + 1 SET {t}.v = s.v + 1, {t}.g = s.g WHERE s.w = 1",
    ))
    ctx.both(ctx.pick(
        postgres="DELETE FROM {t} USING {t} AS s WHERE {t}.id = s.id + 2 AND s.v < 0",
        sqlite="DELETE FROM {t} WHERE id IN (SELECT id + 2 FROM {t} WHERE v < 0)",
        mysql="DELETE {t} FROM {t} JOIN {t} AS s ON {t}.id = s.id + 2 WHERE s.v < 0",
    ))
    ctx.both(ctx.pick(
        postgres="UPDATE {t} SET v = {t}.v + d.g FROM d WHERE {t}.g = d.g AND d.name = 'n1'",
        sqlite="UPDATE {t} SET v = {t}.v + d.g FROM d WHERE {t}.g = d.g AND d.name = 'n1'",
        mysql="UPDATE {t} JOIN d ON {t}.g = d.g SET {t}.v = {t}.v + d.g WHERE d.name = 'n1'",
    ))
    ctx.commit_check("self-join UPDATE/DELETE")


@cell("one_statement_both_join_sources", AREA_A,
      "One statement modifying both sources of a join view (multi-table UPDATE/DELETE; PostgreSQL writable CTE)",
      "PostgreSQL: BEFORE/AFTER statement bracket detects multi-source overlap and reconciles once via "
      "inclusion-exclusion over staged deltas (Stage 3)",
      postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _both_join_sources(ctx: Ctx) -> None:
    env = ctx.env
    if ctx.family == "postgres":
            env.execute("WITH a AS (INSERT INTO d (g, name) VALUES (30, 'n1'), (31, 'n2') RETURNING g) "
                    "INSERT INTO t (id, g, v) VALUES (900, 30, 5), (901, 31, 7), (902, 1, 11)")
    else:
        env.execute("UPDATE t JOIN d ON t.g = d.g SET t.v = t.v + 1, d.name = CONCAT(d.name, 'x') WHERE d.g < 3")
        env.execute("DELETE t, d FROM t JOIN d ON t.g = d.g WHERE d.name = 'n1x'")
    ctx.commit_check("one statement on both join sources")


def _copy(env: Env, table: str, data: str, options: str = "") -> None:
    cur = env.conn.cursor()
    cur.copy_expert(f"COPY {table} (id, g, v, w, u) FROM STDIN {options}", io.StringIO(data))


_COPY_DATA = "201\t1\t5\t\\N\t2010\n202\t\\N\t-7\t2\t\\N\n203\t9\t5\t1\t2030\n204\t1\t5\t0\t\\N\n"


@cell("pg_copy", AREA_A, "COPY FROM STDIN, and COPY ... (FREEZE) after TRUNCATE in the same transaction",
      "twin", postgres=EXACT)
def _pg_copy(ctx: Ctx) -> None:
    env = ctx.env
    ctx.twins.append(("t", "t0"))
    for tbl in ("t0", "t"):
        _copy(env, tbl, _COPY_DATA)
    ctx.commit_check("COPY")
    for tbl in ("t0", "t"):
        env.execute(f"TRUNCATE {tbl}")
        _copy(env, tbl, _COPY_DATA, "WITH (FREEZE)")
    ctx.commit_check("TRUNCATE + COPY FREEZE")


@cell("mysql_load_data_local", AREA_A, "LOAD DATA LOCAL INFILE (plain, REPLACE and IGNORE)",
      "twin; needs local_infile on the server (the test enables it for its duration)",
      mysql=EXACT, mariadb=EXACT)
def _load_data(ctx: Ctx) -> None:
    import pymysql

    env = ctx.env
    cur = env.conn.cursor()
    cur.execute("SELECT @@GLOBAL.local_infile")
    was = int(cur.fetchone()[0])
    if not was:
        cur.execute("SET GLOBAL local_infile = 1")
        ctx.cleanup("SET GLOBAL local_infile = 0")
    env.commit()
    p = urlparse(env.target.dsn)
    conn = pymysql.connect(host=p.hostname or "127.0.0.1", port=p.port or 3306, user=p.username or "root",
                           password=p.password or "", database=env.ns, autocommit=False, local_infile=True)
    env._extra_conns.append(conn)
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("1,4,5,1,9010\n650,1,2,1,\\N\n651,\\N,3,2,6510\n652,2,-9,1,20\n")
        for kw in ("", "REPLACE", "IGNORE"):
            ctx.both(f"LOAD DATA LOCAL INFILE '{path}' {kw} INTO TABLE {{t}} FIELDS TERMINATED BY ',' "
                     "(id, g, v, w, u)", conn=conn)
            ctx.check(f"LOAD DATA {kw} (in transaction)", conn=conn, end_tx=False)
        conn.commit()
        ctx.check("LOAD DATA")
    finally:
        os.unlink(path)


@cell("insert_select_own_source", AREA_A, "INSERT ... SELECT reading the source table itself", "twin",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _insert_select_self(ctx: Ctx) -> None:
    ctx.both("INSERT INTO {t} (id, g, v, w, u) SELECT id + 1000, (id + 1) % 4, v, w, NULL FROM {t} WHERE id % 3 = 0")
    ctx.commit_check("INSERT ... SELECT from the source")


@cell("dml_reading_the_view", AREA_A,
      "INSERT/UPDATE/DELETE of a source whose statement reads the materialized view itself",
      "MySQL/MariaDB: the maintenance trigger cannot write the backing table the statement reads "
      "(error 1442), the statement fails and is rolled back",
      sqlite=EXACT, postgres=EXACT, mysql=REFUSED, mariadb=REFUSED)
def _dml_reading_view(ctx: Ctx) -> None:
    stmts = ["INSERT INTO t (id, g, v, w, u) SELECT 900 + g, g, s, 1, NULL FROM v_agg WHERE g IS NOT NULL",
             "UPDATE t SET v = v + 1 WHERE g IN (SELECT g FROM v_agg WHERE c > 4)",
             "DELETE FROM t WHERE id IN (SELECT id FROM v_pr WHERE v < 0)"]
    for sql in stmts:
        if ctx.outcome == REFUSED:
            ctx.fails(sql, match="1442|trigger")
        else:
            ctx.env.execute(sql)
        ctx.check(sql + " (in transaction)", end_tx=False)
    ctx.commit_check("DML reading the materialized view")


@cell("returning", AREA_A, "INSERT / UPDATE / DELETE / upsert ... RETURNING", "twin (returned rows compared)",
      since={"sqlite": (3, 35), "mariadb": (10, 5)}, sqlite=EXACT, postgres=EXACT, mariadb=EXACT)
def _returning(ctx: Ctx) -> None:
    if ctx.product == "mariadb":
        ctx.both("INSERT INTO {t} (id, g, v, w, u) VALUES (660, 1, 2, 1, NULL), (661, NULL, 3, 1, 6610) RETURNING id, g, v")
        ctx.both("REPLACE INTO {t} (id, g, v, w, u) VALUES (1, 2, 3, 1, 20) RETURNING id, g, v, u")
        ctx.both("DELETE FROM {t} WHERE v < 0 RETURNING id, v")
    else:
        ctx.both("INSERT INTO {t} (id, g, v, w, u) VALUES (660, 1, 2, 1, NULL), (661, NULL, 3, 1, 6610) RETURNING id, g, v")
        ctx.both("UPDATE {t} SET g = 4, v = v * 2 WHERE id % 4 = 1 RETURNING id, g, v")
        ctx.both("DELETE FROM {t} WHERE v < 0 RETURNING id, v")
        ctx.both(ctx.pick(
            postgres="INSERT INTO {t} AS x (id, g, v, w, u) VALUES (2, 1, 1, 1, NULL), (662, 1, 1, 1, NULL) "
                     "ON CONFLICT (id) DO UPDATE SET v = x.v + 100 RETURNING id, v",
            sqlite="INSERT INTO {t} (id, g, v, w, u) VALUES (2, 1, 1, 1, NULL), (662, 1, 1, 1, NULL) "
                   "ON CONFLICT (id) DO UPDATE SET v = {t}.v + 100 RETURNING id, v"))
    ctx.commit_check("RETURNING")


@cell("truncate", AREA_A,
      "TRUNCATE (PostgreSQL: of both join sources in one statement; SQLite: DELETE without WHERE)",
      "MySQL/MariaDB TRUNCATE fires no triggers: registration refuses with MYSQL_NO_TRUNCATE_TRIGGER unless "
      "acknowledged (ignore_limitations=True, then TRUNCATE must not be used)",
      sqlite=EXACT, postgres=EXACT, mysql=REFUSED, mariadb=REFUSED)
def _truncate(ctx: Ctx) -> None:
    env = ctx.env
    if ctx.family == "mysql":
        with pytest.raises(EngineLimitationError, match="MYSQL_NO_TRUNCATE_TRIGGER"):
            env.manager.register_view("CREATE VIEW v_trunc AS SELECT g, COUNT(*) AS c FROM t GROUP BY g;")
        env.rollback()
        return
    if ctx.family == "postgres":
        env.execute("TRUNCATE t, d")
        ctx.commit_check("TRUNCATE t, d")
        env.execute("INSERT INTO d (g, name) VALUES (1, 'n1'), (2, 'n2')")
        env.execute("INSERT INTO t (id, g, v) VALUES (1, 1, 1), (2, 2, 2)")
        env.execute("TRUNCATE ONLY t RESTART IDENTITY")
    else:
        env.execute("DELETE FROM t")
    ctx.commit_check("TRUNCATE / DELETE all")


# ---------------------------------------------------------------------------
# B. Other triggers, objects and settings
# ---------------------------------------------------------------------------


def _pg_trigger(ctx: Ctx, name: str, timing: str, events: str, level: str, body: str,
                when: str = "", referencing: str = "", table: str = "t") -> None:
    fn = "au_fn_" + re.sub(r"\W", "", name)
    ctx.env.execute(f"CREATE FUNCTION {fn}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN {body} END $$")
    ctx.env.execute(f'CREATE TRIGGER "{name}" {timing} {events} ON {table} {referencing} FOR EACH {level} '
                    f"{('WHEN (' + when + ')') if when else ''} EXECUTE FUNCTION {fn}()")
    ctx.env.commit()
    ctx.cleanup(f"DROP FUNCTION IF EXISTS {fn}() CASCADE")


def _mysql_matview_triggers(ctx: Ctx, table: str, event: str) -> List[str]:
    cur = ctx.env.conn.cursor()
    cur.execute("SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = DATABASE() "
                f"AND event_object_table = '{table}' AND event_manipulation = '{event}' AND action_timing = 'AFTER' "
                "AND trigger_name NOT LIKE 'au%' ORDER BY action_order")
    names = [str(r[0]) for r in cur.fetchall()]
    ctx.env.commit()
    assert names, (table, event)
    return names


@cell("user_before_trigger_rewrites_row", AREA_B,
      "User BEFORE row trigger on the source that rewrites NEW (also during upserts)",
      "SQLite cannot assign NEW (see user_after_trigger_writes_same_table)",
      postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _before_rewrites(ctx: Ctx) -> None:
    env = ctx.env
    if ctx.family == "postgres":
        _pg_trigger(ctx, "au_before", "BEFORE", "INSERT OR UPDATE", "ROW",
                    "NEW.v := NEW.v * 2; NEW.g := COALESCE(NEW.g, 4); RETURN NEW;")
        upsert = "INSERT INTO t AS x (id, g, v, w, u) VALUES (1, NULL, 5, 1, 10), (670, NULL, 6, 1, NULL) " \
                 "ON CONFLICT (id) DO UPDATE SET v = x.v + EXCLUDED.v, g = EXCLUDED.g"
    else:
        for ev in ("INSERT", "UPDATE"):
            env.execute(f"CREATE TRIGGER au_before_{ev.lower()} BEFORE {ev} ON t FOR EACH ROW "
                        "SET NEW.v = NEW.v * 2, NEW.g = COALESCE(NEW.g, 4)")
            ctx.cleanup(f"DROP TRIGGER IF EXISTS au_before_{ev.lower()}")
        upsert = "INSERT INTO t (id, g, v, w, u) VALUES (1, NULL, 5, 1, 10), (670, NULL, 6, 1, NULL) " \
                 "ON DUPLICATE KEY UPDATE v = v + VALUES(v), g = VALUES(g)"
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (671, NULL, 3, 1, NULL), (672, 2, -1, 1, NULL)")
    env.execute("UPDATE t SET v = v + 1, g = NULL WHERE id % 5 = 0")
    env.execute(upsert)
    ctx.commit_check("BEFORE trigger rewriting NEW")


@cell("user_before_trigger_skips_row", AREA_B,
      "User BEFORE row trigger that suppresses some rows (RETURN NULL / RAISE(IGNORE))",
      "MySQL/MariaDB triggers cannot skip a row", sqlite=EXACT, postgres=EXACT)
def _before_skips(ctx: Ctx) -> None:
    env = ctx.env
    if ctx.family == "postgres":
        _pg_trigger(ctx, "au_skip", "BEFORE", "INSERT OR UPDATE OR DELETE", "ROW",
                    "IF TG_OP = 'DELETE' THEN IF OLD.w = 0 THEN RETURN NULL; END IF; RETURN OLD; END IF; "
                    "IF NEW.v = 13 THEN RETURN NULL; END IF; RETURN NEW;")
    else:
        for ev, cond in (("INSERT", "NEW.v = 13"), ("UPDATE", "NEW.v = 13"), ("DELETE", "OLD.w = 0")):
            env.execute(f"CREATE TRIGGER au_skip_{ev.lower()} BEFORE {ev} ON t FOR EACH ROW WHEN {cond} "
                        "BEGIN SELECT RAISE(IGNORE); END")
            ctx.cleanup(f"DROP TRIGGER IF EXISTS au_skip_{ev.lower()}")
        env.commit()
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (680, 1, 13, 1, NULL), (681, 1, 14, 1, NULL)")
    env.execute("UPDATE t SET v = 13 WHERE id % 3 = 0")
    env.execute("DELETE FROM t WHERE id % 2 = 0")
    ctx.commit_check("BEFORE trigger skipping rows")


TRIGGER_ORDER_TODO = "Exact Maintenance When a User Trigger or FK Action Runs Before the Maintenance Trigger"


@cell("user_after_trigger_writes_same_table", AREA_B,
      "User AFTER row trigger that updates the same row and inserts/deletes other rows of the source",
      "SQLite: Stage 4's same-group in-place UPDATE trigger avoids transiently deleting the group when a newer user "
      "AFTER INSERT trigger updates a non-group column on an existing group (still wrong if the user trigger changes "
      "a group key or updates an aggregated column of a brand-new group before INSERT maintenance runs — same-row "
      "updates before outer AFTER STATEMENT also affect PostgreSQL). MySQL/MariaDB forbid a trigger writing its own "
      "table (error 1442)",
      sqlite=EXACT, postgres=EXACT)
def _after_writes_same(ctx: Ctx) -> None:
    env = ctx.env
    if ctx.family == "postgres":
        _pg_trigger(ctx, "au_after", "AFTER", "INSERT", "ROW",
                    "UPDATE t SET w = COALESCE(w, 0) + 1 WHERE id = NEW.id; "
                    "INSERT INTO t (id, g, v, w, u) VALUES (NEW.id + 10000, NEW.g, -NEW.v, 1, NULL); "
                    "DELETE FROM t WHERE id = NEW.id - 685 AND NEW.v > 4; RETURN NULL;", when="NEW.id < 10000")
    else:
        env.execute("CREATE TRIGGER au_after AFTER INSERT ON t FOR EACH ROW WHEN NEW.id < 10000 BEGIN "
                    "UPDATE t SET w = COALESCE(w, 0) + 1 WHERE id = NEW.id; "
                    "INSERT INTO t (id, g, v, w, u) VALUES (NEW.id + 10000, NEW.g, -NEW.v, 1, NULL); "
                    "DELETE FROM t WHERE id = NEW.id - 685 AND NEW.v > 4; END")
        ctx.cleanup("DROP TRIGGER IF EXISTS au_after")
        env.commit()
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (690, 1, 5, 0, NULL), (691, 2, 3, NULL, NULL), (692, NULL, 9, 1, NULL)")
    ctx.commit_check("AFTER trigger writing its own table")


@cell("sqlite_recursive_user_trigger", AREA_B,
      "A user trigger that re-fires itself (unguarded self-UPDATE)",
      "configure_sqlite_connection() turns recursive_triggers ON, so such a trigger recurses until SQLite's depth "
      "limit and the statement fails (it ran once with the SQLite default); the view stays exact either way; "
      "PostgreSQL user triggers recurse regardless (guard with pg_trigger_depth())",
      sqlite=LIMITED)
def _sqlite_recursion(ctx: Ctx) -> None:
    env = ctx.env
    env.execute("CREATE TRIGGER au_touch AFTER UPDATE ON t FOR EACH ROW BEGIN "
                "UPDATE t SET w = COALESCE(w, 0) + 1 WHERE id = NEW.id; END")
    ctx.cleanup("DROP TRIGGER IF EXISTS au_touch")
    env.commit()
    ctx.fails("UPDATE t SET v = v + 1 WHERE id = 3", match="recursion")
    ctx.check("recursion limit on the configured connection", end_tx=False)
    env.rollback()
    w = env.raw_sqlite(configure=False)
    w.execute("UPDATE t SET v = v + 1 WHERE id IN (3, 4)")
    ctx.check("same UPDATE on a connection with recursive_triggers OFF", conn=w, end_tx=False)
    w.commit()
    ctx.check("committed")


@cell("user_trigger_raises", AREA_B,
      "A user trigger raising an error mid-statement (RAISE EXCEPTION / SIGNAL / RAISE(ABORT|FAIL|ROLLBACK))",
      "rollback atomicity of the statement (SQLite FAIL keeps the earlier rows, ROLLBACK ends the transaction)",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _user_trigger_raises(ctx: Ctx) -> None:
    env = ctx.env
    rows = "(700, 1, 5, 1, NULL), (701, 2, 6, 1, NULL), (702, 3, 666, 1, NULL), (703, 4, 7, 1, NULL)"
    if ctx.family == "postgres":
        _pg_trigger(ctx, "au_raise", "AFTER", "INSERT OR UPDATE", "ROW",
                    "IF NEW.v = 666 THEN RAISE EXCEPTION 'au: rejected'; END IF; RETURN NULL;")
    elif ctx.family == "mysql":
        for ev in ("INSERT", "UPDATE"):
            env.execute(f"CREATE TRIGGER au_raise_{ev.lower()} AFTER {ev} ON t FOR EACH ROW "
                        "BEGIN IF NEW.v = 666 THEN SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'au: rejected'; END IF; END")
            ctx.cleanup(f"DROP TRIGGER IF EXISTS au_raise_{ev.lower()}")
    else:
        for ev in ("INSERT", "UPDATE"):
            env.execute(f"CREATE TRIGGER au_raise_{ev.lower()} AFTER {ev} ON t FOR EACH ROW WHEN NEW.v = 666 "
                        "BEGIN SELECT RAISE(ABORT, 'au: rejected'); END")
            env.execute(f"CREATE TRIGGER au_rb_{ev.lower()} AFTER {ev} ON t FOR EACH ROW WHEN NEW.v = 668 "
                        "BEGIN SELECT RAISE(ROLLBACK, 'au: rejected'); END")
            ctx.cleanup(f"DROP TRIGGER IF EXISTS au_raise_{ev.lower()}", f"DROP TRIGGER IF EXISTS au_rb_{ev.lower()}")
        env.commit()
    before = env.snapshot()
    env.commit()
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (699, 1, 1, 1, NULL)")
    ctx.fails(f"INSERT INTO t (id, g, v, w, u) VALUES {rows}", match="rejected")
    ctx.check("trigger error in INSERT", end_tx=False)
    ctx.fails("UPDATE t SET v = CASE WHEN id = 12 THEN 666 ELSE v + 1 END, g = 1", match="rejected")
    ctx.check("trigger error in UPDATE", end_tx=False)
    if ctx.family == "sqlite":
        ctx.fails("UPDATE t SET v = CASE WHEN id = 12 THEN 668 ELSE v + 1 END, g = 1", match="rejected")
        assert not env.conn.in_transaction
        ctx.check("RAISE(ROLLBACK) ended the transaction")
        assert env.snapshot() == before
        return
    ctx.commit_check("user trigger errors")


@cell("sqlite_user_trigger_raise_fail", AREA_B,
      "A user AFTER trigger created after registration raising RAISE(FAIL) on the second row of a statement",
      "FAIL keeps the rows already written, including the failing row itself, but the newer user trigger ran "
      "first and the failing row's maintenance trigger never runs",
      todo=TRIGGER_ORDER_TODO, sqlite=WRONG)
def _sqlite_raise_fail(ctx: Ctx) -> None:
    env = ctx.env
    env.execute("CREATE TRIGGER au_fail AFTER INSERT ON t FOR EACH ROW WHEN NEW.v = 667 "
                "BEGIN SELECT RAISE(FAIL, 'au: rejected'); END")
    ctx.cleanup("DROP TRIGGER IF EXISTS au_fail")
    env.commit()
    ctx.fails("INSERT INTO t (id, g, v, w, u) VALUES (710, 1, 1, 1, NULL), (711, 2, 667, 1, NULL)", match="rejected")
    ctx.commit_check("RAISE(FAIL) keeps the earlier rows")


@cell("sqlite_user_triggers_older_than_views", AREA_B,
      "The same self-writing and RAISE(FAIL) user triggers created before the views were registered",
      "SQLite then fires them after the maintenance triggers", sqlite=EXACT)
def _sqlite_old_triggers(ctx: Ctx) -> None:
    schema = BASE_SCHEMA["sqlite"] + [
        "CREATE TRIGGER au_after AFTER INSERT ON t FOR EACH ROW WHEN NEW.id < 10000 AND NEW.v <> 667 BEGIN "
        "UPDATE t SET w = COALESCE(w, 0) + 1 WHERE id = NEW.id; "
        "INSERT INTO t (id, g, v, w, u) VALUES (NEW.id + 10000, NEW.g, -NEW.v, 1, NULL); "
        "DELETE FROM t WHERE id = NEW.id - 685 AND NEW.v > 4; END",
        "CREATE TRIGGER au_fail AFTER INSERT ON t FOR EACH ROW WHEN NEW.v = 667 "
        "BEGIN SELECT RAISE(FAIL, 'au: rejected'); END"]
    env = ctx.fresh({"sqlite": schema}, [V_AGG, V_MM, V_PR], _base_seed())
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (690, 1, 5, 0, NULL), (691, 2, 3, NULL, NULL), (692, NULL, 9, 1, NULL)")
    ctx.check("self-writing trigger after maintenance", env, end_tx=False)
    ctx.fails("INSERT INTO t (id, g, v, w, u) VALUES (710, 1, 1, 1, NULL), (711, 2, 667, 1, NULL)", env, match="rejected")
    ctx.commit_check("RAISE(FAIL) after maintenance", env)


@cell("pg_statement_trigger_ordering", AREA_B,
      "User statement-level AFTER triggers ordered before and after the maintenance triggers (by name) that write "
      "the source again", "", postgres=EXACT)
def _pg_stmt_ordering(ctx: Ctx) -> None:
    env = ctx.env
    for name in ("!au_first", "~au_last"):
        off = 20000 if name.startswith("!") else 30000
        _pg_trigger(ctx, name, "AFTER", "INSERT", "STATEMENT",
                    f"IF pg_trigger_depth() < 2 THEN INSERT INTO t (id, g, v, w, u) "
                    f"SELECT id + {off}, g, v + 1, w, NULL FROM au_new WHERE id < 10000; "
                    f"UPDATE t SET v = v - 1 WHERE id IN (SELECT id FROM au_new); END IF; RETURN NULL;",
                    referencing="REFERENCING NEW TABLE AS au_new")
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (720, 1, 5, 1, NULL), (721, 2, 6, 1, NULL), (722, NULL, 7, 0, NULL)")
    ctx.commit_check("statement triggers before and after maintenance")


def _other_source_rows() -> str:
    return "(950, 50, 5, 1, NULL), (951, 50, 7, 1, NULL), (952, 51, 1, 1, NULL), (953, 1, 2, 1, NULL)"


@cell("user_trigger_writes_other_source_first", AREA_B,
      "User trigger on one join source that writes the other source, firing BEFORE the maintenance trigger",
      "PostgreSQL: row/statement trigger modifying the other join source runs inside the Stage 3 statement bracket "
      "and reconciles via inclusion-exclusion; SQLite (newest trigger first) and MySQL/MariaDB (PRECEDES) still "
      "count the new pair twice where maintained algebraically",
      todo=TRIGGER_ORDER_TODO,
      sqlite=WRONG, postgres=EXACT, mysql=WRONG, mariadb=WRONG)
def _other_source_first(ctx: Ctx) -> None:
    env = ctx.env
    if ctx.family == "postgres":
        _pg_trigger(ctx, "au_dim", "AFTER", "INSERT", "ROW",
                    "INSERT INTO d (g, name) VALUES (NEW.g, 'nz') ON CONFLICT (g) DO NOTHING; RETURN NULL;",
                    when="NEW.g >= 50")
    elif ctx.family == "mysql":
        first = _mysql_matview_triggers(ctx, "t", "INSERT")[0]
        env.execute(f"CREATE TRIGGER au_dim AFTER INSERT ON t FOR EACH ROW PRECEDES {first} "
                    "BEGIN IF NEW.g >= 50 THEN INSERT IGNORE INTO d (g, name) VALUES (NEW.g, 'nz'); END IF; END")
        ctx.cleanup("DROP TRIGGER IF EXISTS au_dim")
    else:
        env.execute("CREATE TRIGGER au_dim AFTER INSERT ON t FOR EACH ROW WHEN NEW.g >= 50 "
                    "BEGIN INSERT OR IGNORE INTO d (g, name) VALUES (NEW.g, 'nz'); END")
        ctx.cleanup("DROP TRIGGER IF EXISTS au_dim")
        env.commit()
    ctx.cleanup("DELETE FROM d WHERE g >= 50")
    env.execute(f"INSERT INTO t (id, g, v, w, u) VALUES {_other_source_rows()}")
    ctx.commit_check("trigger writing the other join source first")


@cell("user_trigger_writes_other_source_last", AREA_B,
      "User trigger on one join source that writes the other source, firing AFTER the maintenance trigger",
      "PostgreSQL statement trigger sorting after the maintenance triggers; SQLite trigger created before "
      "registration; MySQL/MariaDB FOLLOWS",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _other_source_last(ctx: Ctx) -> None:
    if ctx.family == "sqlite":
        schema = BASE_SCHEMA["sqlite"] + [
            "CREATE TRIGGER au_dim AFTER INSERT ON t FOR EACH ROW WHEN NEW.g >= 50 "
            "BEGIN INSERT OR IGNORE INTO d (g, name) VALUES (NEW.g, 'nz'); END"]
        env: Env = ctx.fresh({"sqlite": schema}, [V_J, V_PJ], _base_seed())
    else:
        env = ctx.env
        ctx.cleanup("DELETE FROM d WHERE g >= 50")
        if ctx.family == "postgres":
            _pg_trigger(ctx, "~au_dim", "AFTER", "INSERT", "STATEMENT",
                        "INSERT INTO d (g, name) SELECT DISTINCT g, 'nz' FROM au_new WHERE g >= 50 "
                        "ON CONFLICT (g) DO NOTHING; RETURN NULL;", referencing="REFERENCING NEW TABLE AS au_new")
        else:
            last = _mysql_matview_triggers(ctx, "t", "INSERT")[-1]
            env.execute(f"CREATE TRIGGER au_dim AFTER INSERT ON t FOR EACH ROW FOLLOWS {last} "
                        "BEGIN IF NEW.g >= 50 THEN INSERT IGNORE INTO d (g, name) VALUES (NEW.g, 'nz'); END IF; END")
            ctx.cleanup("DROP TRIGGER IF EXISTS au_dim")
    env.execute(f"INSERT INTO t (id, g, v, w, u) VALUES {_other_source_rows()}")
    ctx.commit_check("trigger writing the other join source last", env)


@cell("dml_through_views", AREA_B,
      "DML through an updatable view and through an INSTEAD OF trigger on a view over the source",
      "MySQL/MariaDB have no INSTEAD OF; their updatable single-table and join views are used",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _dml_through_views(ctx: Ctx) -> None:
    env = ctx.env
    env.execute("CREATE VIEW au_tv AS SELECT id, g, v, w, u FROM t WHERE g IS NOT NULL")
    env.execute("CREATE VIEW au_tj AS SELECT t.id, t.v, d.name FROM t JOIN d ON t.g = d.g")
    ctx.cleanup("DROP VIEW IF EXISTS au_tv", "DROP VIEW IF EXISTS au_tj")
    if ctx.family == "postgres":
        _pg_trigger(ctx, "au_io", "INSTEAD OF", "INSERT", "ROW",
                    "INSERT INTO t (id, g, v, w) SELECT NEW.id, min(d.g), NEW.v, 1 FROM d WHERE d.name = NEW.name; "
                    "RETURN NEW;", table="au_tj")
    elif ctx.family == "sqlite":
        env.execute("CREATE TRIGGER au_io_ins INSTEAD OF INSERT ON au_tv BEGIN "
                    "INSERT INTO t (id, g, v, w, u) VALUES (NEW.id, NEW.g, NEW.v, NEW.w, NEW.u); END")
        env.execute("CREATE TRIGGER au_io_upd INSTEAD OF UPDATE ON au_tv BEGIN "
                    "UPDATE t SET g = NEW.g, v = NEW.v WHERE id = OLD.id; END")
        env.execute("CREATE TRIGGER au_io_del INSTEAD OF DELETE ON au_tv BEGIN DELETE FROM t WHERE id = OLD.id; END")
        env.execute("CREATE TRIGGER au_io INSTEAD OF INSERT ON au_tj BEGIN "
                    "INSERT INTO t (id, g, v, w) SELECT NEW.id, min(d.g), NEW.v, 1 FROM d WHERE d.name = NEW.name; END")
    env.commit()
    env.execute("INSERT INTO au_tv (id, g, v, w, u) VALUES (730, 1, 4, 1, NULL), (731, 3, -2, 2, NULL)")
    env.execute("UPDATE au_tv SET v = v + 1 WHERE g = 2")
    env.execute("DELETE FROM au_tv WHERE id % 4 = 0")
    if ctx.family == "mysql":
        env.execute("UPDATE au_tj SET v = v * 2 WHERE name = 'n1'")
    else:
        env.execute("INSERT INTO au_tj (id, v, name) VALUES (732, 3, 'n1'), (733, 8, 'n2')")
    ctx.commit_check("DML through views")


@cell("view_as_source", AREA_B, "Registering a materialized view whose FROM names a plain (non-materialized) view",
      "refused at registration: a view has no triggers to maintain it",
      sqlite=REFUSED, postgres=REFUSED, mysql=REFUSED, mariadb=REFUSED)
def _view_as_source(ctx: Ctx) -> None:
    env = ctx.env
    env.execute("CREATE VIEW au_src AS SELECT id, g, v FROM t WHERE g IS NOT NULL")
    env.commit()
    ctx.cleanup("DROP VIEW IF EXISTS au_src")
    with pytest.raises(Exception) as info:
        env.manager.register_view("CREATE VIEW v_over_view AS SELECT g, COUNT(*) AS c FROM au_src GROUP BY g;",
                                  ignore_limitations=True)
    env.rollback()
    assert isinstance(info.value, ValueError), repr(info.value)
    assert "does not reference any known base tables" in str(info.value), str(info.value)
    assert "v_over_view" not in env.manager.get_catalog()
    env.rollback()


@cell("pg_rules", AREA_B, "Rewrite rules on the source: conditional DO INSTEAD (soft delete) and DO ALSO",
      "", postgres=EXACT)
def _pg_rules(ctx: Ctx) -> None:
    env = ctx.env
    env.execute("CREATE RULE au_soft AS ON DELETE TO t WHERE OLD.w = 0 DO INSTEAD UPDATE t SET w = 9 WHERE id = OLD.id")
    env.execute("CREATE RULE au_also AS ON INSERT TO t DO ALSO UPDATE t SET v = v + 1 WHERE id = NEW.id")
    env.commit()
    ctx.cleanup("DROP RULE IF EXISTS au_soft ON t", "DROP RULE IF EXISTS au_also ON t")
    env.execute("DELETE FROM t WHERE id % 3 = 0")
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (740, 1, 5, 1, NULL), (741, NULL, 6, 1, NULL)")
    ctx.commit_check("rules")


def _gen_env(ctx: Ctx, virtual: bool) -> Env:
    kind = "VIRTUAL" if virtual else "STORED"
    if ctx.family == "postgres":
        ddl = f"CREATE TABLE gt (id INT PRIMARY KEY, g INT, v BIGINT, gx BIGINT GENERATED ALWAYS AS (v * 2 + g) {kind})"
    elif ctx.family == "mysql":
        ddl = f"CREATE TABLE gt (id INT PRIMARY KEY, g INT, v BIGINT, gx BIGINT AS (v * 2 + g) {kind}) ENGINE=InnoDB"
    else:
        ddl = f"CREATE TABLE gt (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, gx INTEGER GENERATED ALWAYS AS (v * 2 + g) {kind})"
    views = [ViewDef("v_gs", "SELECT g, SUM(gx) AS s, COUNT(*) AS c FROM gt GROUP BY g"),
             ViewDef("v_gk", "SELECT gx, COUNT(*) AS c FROM gt GROUP BY gx"),
             ViewDef("v_gp", "SELECT id, gx FROM gt WHERE gx > 0")]
    return ctx.fresh({ctx.family: [ddl]}, views,
                     ["INSERT INTO gt (id, g, v) VALUES " + _values([(i, i % 3, (i * 7) % 11 - 4) for i in range(1, 16)])])


_GEN_DML = ["INSERT INTO gt (id, g, v) VALUES (100, 1, 3), (101, NULL, 2), (102, 2, NULL)",
            "UPDATE gt SET v = v + 1 WHERE id % 2 = 0",
            "UPDATE gt SET g = 2 WHERE id % 3 = 0",
            "DELETE FROM gt WHERE id % 5 = 0"]


@cell("generated_stored", AREA_B,
      "Views over a STORED generated column (aggregated, grouped by, filtered) while its inputs change",
      "SQLite fixed here: the maintenance UPDATE trigger was limited to the referenced columns (UPDATE OF), which "
      "skipped updates of the generated column's inputs",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _generated_stored(ctx: Ctx) -> None:
    env = _gen_env(ctx, virtual=False)
    for sql in _GEN_DML:
        env.execute(sql)
        ctx.check(sql + " (in transaction)", env, end_tx=False)
    ctx.commit_check("STORED generated column", env)


@cell("generated_virtual", AREA_B,
      "Views over a VIRTUAL generated column while its inputs change",
      "PostgreSQL 18+ fixed here: statement triggers project non-virtual transition columns explicitly and "
      "compute VIRTUAL generated columns from their catalog generation expressions instead of reading them "
      "from the transition tables",
      since={"postgres": (18, 0)}, sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _generated_virtual(ctx: Ctx) -> None:
    env = _gen_env(ctx, virtual=True)
    for sql in _GEN_DML:
        env.execute(sql)
        ctx.check(sql + " (in transaction)", env, end_tx=False)
    ctx.commit_check("VIRTUAL generated column", env)


@cell("pg_domains", AREA_B,
      "Source columns of domain types over BIGINT, VARCHAR(n), TEXT, TEXT with CHECK, TEXT COLLATE \"C\" and CHAR(n): "
      "grouped by, "
      "aggregated, projected (incl. domain CHECK violations)",
      "registration used to fail (backing column declared as type DOMAIN; for a domain over TEXT or CHAR(n) "
      "the reflected collation \"default\" and the lost CHAR length) and a domain group key was refused by the "
      "canonical group-key function; a domain over an array type still fails (array backing types)",
      postgres=EXACT)
def _pg_domains(ctx: Ctx) -> None:
    env = ctx.fresh({"postgres": ["CREATE DOMAIN au_pos AS BIGINT CHECK (VALUE > 0)",
                                  "CREATE DOMAIN au_label AS VARCHAR(10)",
                                  "CREATE DOMAIN au_txt AS TEXT",
                                  "CREATE DOMAIN au_code AS TEXT CHECK (VALUE <> '')",
                                  "CREATE DOMAIN au_ch3 AS CHAR(3)",
                                  "CREATE DOMAIN au_cc AS TEXT COLLATE \"C\"",
                                  "CREATE TABLE dm (id INT PRIMARY KEY, g au_label, v au_pos, t au_txt, k au_code, "
                                  "c au_ch3, cc au_cc)"]},
                    [ViewDef("v_dm", "SELECT g, SUM(v) AS s, COUNT(*) AS c FROM dm GROUP BY g"),
                     ViewDef("v_dmp", "SELECT id, g, v, t FROM dm WHERE v > 3"),
                     ViewDef("v_dmt", "SELECT t, k, cc, COUNT(*) AS n FROM dm GROUP BY t, k, cc"),
                     ViewDef("v_dmc", "SELECT c, MAX(v) AS hi, COUNT(*) AS n FROM dm GROUP BY c")],
                    ["INSERT INTO dm VALUES " + _values(
                        [(i, f"l{i % 3}", i, f"t{i % 2}", f"k{i % 4}", f"c{i % 3}", "Ab"[i % 2]) for i in range(1, 13)])])
    env.execute("INSERT INTO dm VALUES (100, 'l9', 5, 'tz', 'kz', 'abc', 'a'), (101, NULL, 1, NULL, NULL, NULL, NULL)")
    env.execute("UPDATE dm SET g = 'l1', v = v + 2, t = 'tz', c = 'xy' WHERE id % 2 = 0")
    ctx.fails("UPDATE dm SET v = v - 7", env)
    ctx.fails("UPDATE dm SET k = '' WHERE id = 3", env)
    ctx.commit_check("domains", env)


_FK_ACTION_SCHEMA = {
    "postgres": ["CREATE TABLE p (pid INT PRIMARY KEY)",
                 "CREATE TABLE c (id INT PRIMARY KEY, v BIGINT, "
                 "p1 INT REFERENCES p (pid) ON DELETE SET NULL ON UPDATE CASCADE, "
                 "p2 INT DEFAULT 1 REFERENCES p (pid) ON DELETE SET DEFAULT ON UPDATE SET NULL)"],
    "sqlite": ["CREATE TABLE p (pid INTEGER PRIMARY KEY)",
               "CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER, "
               "p1 INTEGER REFERENCES p (pid) ON DELETE SET NULL ON UPDATE CASCADE, "
               "p2 INTEGER DEFAULT 1 REFERENCES p (pid) ON DELETE SET DEFAULT ON UPDATE SET NULL)"],
    "mysql": ["CREATE TABLE p (pid INT PRIMARY KEY) ENGINE=InnoDB",
              "CREATE TABLE c (id INT PRIMARY KEY, v BIGINT, p1 INT, p2 INT, "
              "FOREIGN KEY (p1) REFERENCES p (pid) ON DELETE SET NULL ON UPDATE CASCADE, "
              "FOREIGN KEY (p2) REFERENCES p (pid) ON DELETE CASCADE) ENGINE=InnoDB"],
}
_FK_SEED = ["INSERT INTO p (pid) VALUES (1), (2), (3), (4), (5)",
            "INSERT INTO c (id, v, p1, p2) VALUES " + _values([(i, i * 3 - 10, i % 5 + 1, (i * 2) % 5 + 1) for i in range(1, 16)])]
_FK_VIEWS = [ViewDef("v_c1", "SELECT p1, COUNT(*) AS c, SUM(v) AS s FROM c GROUP BY p1"),
             ViewDef("v_c2", "SELECT p2, COUNT(*) AS c, MIN(v) AS lo FROM c GROUP BY p2")]


@cell("fk_referential_actions", AREA_B,
      "FK ON DELETE SET NULL / SET DEFAULT / CASCADE and ON UPDATE CASCADE / SET NULL into a view's source",
      "MySQL/MariaDB: InnoDB referential actions fire no triggers, so registration refuses "
      "(MYSQL_FK_CASCADE_NO_TRIGGER, also for SET NULL); InnoDB has no SET DEFAULT",
      sqlite=EXACT, postgres=EXACT, mysql=REFUSED, mariadb=REFUSED)
def _fk_actions(ctx: Ctx) -> None:
    if ctx.family == "mysql":
        env = ctx.fresh(_FK_ACTION_SCHEMA, [], _FK_SEED)
        for vd in _FK_VIEWS:
            with pytest.raises(EngineLimitationError, match="MYSQL_FK_CASCADE_NO_TRIGGER"):
                env.manager.register_view(f"CREATE VIEW {vd.name} AS {vd.select};")
            env.rollback()
        return
    env = ctx.fresh(_FK_ACTION_SCHEMA, _FK_VIEWS, _FK_SEED)
    env.execute("DELETE FROM p WHERE pid IN (2, 3)")
    ctx.check("ON DELETE SET NULL / SET DEFAULT", env, end_tx=False)
    env.execute("UPDATE p SET pid = 50 WHERE pid = 4")
    ctx.commit_check("ON UPDATE CASCADE / SET NULL", env)


_FK_JOIN_SCHEMA = {
    "postgres": ["CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
                 "CREATE TABLE t (id INT PRIMARY KEY, g INT REFERENCES d (g) ON UPDATE CASCADE ON DELETE SET NULL, "
                 "v BIGINT, w INT, u INT)"],
    "sqlite": ["CREATE TABLE d (g INTEGER PRIMARY KEY, name VARCHAR(20) NOT NULL)",
               "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER REFERENCES d (g) ON UPDATE CASCADE ON DELETE SET NULL, "
               "v INTEGER, w INTEGER, u INTEGER)"],
}


@cell("fk_action_into_both_join_sources", AREA_B,
      "An FK action (ON UPDATE CASCADE / ON DELETE SET NULL of the join key) modifying both sources of a join view",
      "PostgreSQL: Stage 3 statement bracket stages both parent and cascaded child deltas and reconciles once "
      "via inclusion-exclusion; SQLite runs the cascade before the parent row's maintenance trigger; MySQL/MariaDB "
      "refuse as in fk_referential_actions",
      todo=TRIGGER_ORDER_TODO, sqlite=WRONG, postgres=EXACT)
def _fk_join(ctx: Ctx) -> None:
    rows = [r for r in _base_rows() if r[1] is not None]
    env = ctx.fresh(_FK_JOIN_SCHEMA, [V_J, V_PJ, V_AGG],
                    [f"INSERT INTO d (g, name) VALUES {_values(BASE_D)}",
                     f"INSERT INTO t (id, g, v, w, u) VALUES {_values(rows)}"])
    env.execute("UPDATE d SET g = g + 100 WHERE g IN (1, 2)")
    ctx.check("ON UPDATE CASCADE of the join key", env, end_tx=False)
    env.execute("DELETE FROM d WHERE g = 3")
    ctx.commit_check("ON DELETE SET NULL of the join key", env)


_FK_LATE_SCHEMA = {
    "postgres": ["CREATE TABLE p (pid INT PRIMARY KEY)", "CREATE TABLE c (id INT PRIMARY KEY, g INT, v BIGINT, pid INT)"],
    "mysql": ["CREATE TABLE p (pid INT PRIMARY KEY) ENGINE=InnoDB",
              "CREATE TABLE c (id INT PRIMARY KEY, g INT, v BIGINT, pid INT) ENGINE=InnoDB"],
}


@cell("fk_cascade_added_after_registration", AREA_B,
      "A cascading FK added to a source table after the view was registered (out-of-band ALTER TABLE)",
      "MySQL/MariaDB: InnoDB cascaded deletes fire no triggers, so the view becomes stale, but check_schema_drift() "
      "reports invalid (re-evaluating MYSQL_FK_CASCADE_NO_TRIGGER), repair_view() and sync_views() refuse while the "
      "cascading FK exists, and dropping the cascading FK + repair_view() restores exactness; SQLite cannot add an FK "
      "to an existing table",
      postgres=EXACT, mysql=LIMITED, mariadb=LIMITED)
def _fk_late(ctx: Ctx) -> None:
    v_select = "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM c GROUP BY g"
    env = ctx.fresh(_FK_LATE_SCHEMA, [ViewDef("v_late", v_select)],
                    ["INSERT INTO p (pid) VALUES (1), (2), (3)",
                     "INSERT INTO c (id, g, v, pid) VALUES " + _values([(i, i % 4, i * 5 - 20, i % 3 + 1) for i in range(1, 13)])])
    other = env.connect()
    env.execute("ALTER TABLE c ADD CONSTRAINT au_fk FOREIGN KEY (pid) REFERENCES p (pid) ON DELETE CASCADE", other)
    other.commit()
    env.execute("DELETE FROM p WHERE pid = 2")
    env.commit()
    drift = env.manager.check_schema_drift()
    env.commit()
    if ctx.outcome == LIMITED:
        from matview.connectivity import dbapi_transaction_state

        assert drift == {"v_late": "invalid"}, drift
        with pytest.raises(EngineLimitationError, match="MYSQL_FK_CASCADE_NO_TRIGGER"):
            env.manager.repair_view()
        assert dbapi_transaction_state(env.conn) is False
        with pytest.raises(EngineLimitationError, match="MYSQL_FK_CASCADE_NO_TRIGGER"):
            env.manager.sync_views(f"CREATE VIEW v_late AS {v_select};")
        assert dbapi_transaction_state(env.conn) is False
        env.execute("ALTER TABLE c DROP FOREIGN KEY au_fk", other)
        other.commit()
        drift_after_drop = env.manager.check_schema_drift()
        env.commit()
        assert drift_after_drop == {"v_late": "ok"}, drift_after_drop
        repaired = env.manager.repair_view()
        env.commit()
        assert repaired == {"v_late": "repaired"}, repaired
        ctx.check("after dropping cascading FK and repair_view()", env)
        return
    assert drift == {"v_late": "drifted"}, drift
    ctx.check("cascade through an FK added after registration", env)


_FKOFF_SCHEMA = {
    "mysql": ["CREATE TABLE p (pid INT PRIMARY KEY, label VARCHAR(10)) ENGINE=InnoDB",
              "CREATE TABLE c (id INT PRIMARY KEY, v BIGINT, pid INT, FOREIGN KEY (pid) REFERENCES p (pid)) ENGINE=InnoDB"],
    "sqlite": ["CREATE TABLE p (pid INTEGER PRIMARY KEY, label VARCHAR(10))",
               "CREATE TABLE c (id INTEGER PRIMARY KEY, v INTEGER, pid INTEGER REFERENCES p (pid) ON DELETE CASCADE)"],
}


@cell("fk_enforcement_off", AREA_B,
      "Writes with FK enforcement switched off (SET foreign_key_checks = 0 / PRAGMA foreign_keys = OFF)",
      "no referential action runs, so only the statement's own rows change; PostgreSQL has no such switch "
      "(see pg_session_replication_role)", sqlite=EXACT, mysql=EXACT, mariadb=EXACT)
def _fk_off(ctx: Ctx) -> None:
    env = ctx.fresh(_FKOFF_SCHEMA,
                    [ViewDef("v_pc", "SELECT p.label, COUNT(*) AS c, SUM(c.v) AS s FROM c JOIN p ON c.pid = p.pid GROUP BY p.label", join=True),
                     ViewDef("v_c", "SELECT pid, COUNT(*) AS c FROM c GROUP BY pid")],
                    ["INSERT INTO p (pid, label) VALUES (1, 'a'), (2, 'b'), (3, 'a')",
                     "INSERT INTO c (id, v, pid) VALUES " + _values([(i, i * 2 - 9, i % 3 + 1) for i in range(1, 13)])])
    if ctx.family == "mysql":
        w = env.connect()
        env.execute("SET foreign_key_checks = 0", w)
    else:
        assert isinstance(env, DEnv)
        w = env.raw_sqlite(configure=True)
        w.execute("PRAGMA foreign_keys = OFF")
    env.execute("DELETE FROM p WHERE pid = 2", w)
    env.execute("INSERT INTO c (id, v, pid) VALUES (100, 4, 99), (101, 5, 3)", w)
    env.execute("UPDATE p SET pid = 30 WHERE pid = 3", w)
    ctx.check("FK enforcement off (in transaction)", env, conn=w, end_tx=False)
    w.commit()
    ctx.check("FK enforcement off", env)


@cell("mariadb_system_versioning", AREA_B,
      "System-versioned (temporal) source table: UPDATE/DELETE create history rows; DELETE HISTORY",
      "the view and its oracle read the current rows", mariadb=EXACT)
def _system_versioning(ctx: Ctx) -> None:
    env = ctx.fresh({"mariadb": ["CREATE TABLE sv (id INT PRIMARY KEY, g INT, v BIGINT) WITH SYSTEM VERSIONING"]},
                    [ViewDef("v_sv", "SELECT g, SUM(v) AS s, COUNT(*) AS c FROM sv GROUP BY g"),
                     ViewDef("v_svp", "SELECT id, v FROM sv WHERE g = 1")],
                    ["INSERT INTO sv (id, g, v) VALUES " + _values([(i, i % 3, i * 4 - 15) for i in range(1, 13)])])
    env.execute("UPDATE sv SET v = v + 1, g = 1 WHERE id % 2 = 0")
    env.execute("DELETE FROM sv WHERE id % 3 = 0")
    env.execute("INSERT INTO sv (id, g, v) VALUES (3, 2, 100)")
    ctx.commit_check("system-versioned DML", env)
    env.execute("DELETE HISTORY FROM sv")
    ctx.commit_check("DELETE HISTORY", env)


@cell("pg_disable_trigger", AREA_B,
      "ALTER TABLE ... DISABLE TRIGGER USER/ALL on a source (as pg_restore --disable-triggers does)",
      "writes are not maintained while disabled; check_schema_drift() reports triggers_disabled (added here; it "
      "reported ok) and repair_view() rebuilds the views with enabled triggers",
      postgres=LIMITED)
def _pg_disable(ctx: Ctx) -> None:
    # Own small environment: the repair rebuilds every view it reports.
    env = ctx.fresh(BASE_SCHEMA, [V_AGG, V_PR], _base_seed())
    other = env.connect()
    env.execute("ALTER TABLE t DISABLE TRIGGER USER", other)
    other.commit()
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (750, 1, 5, 1, NULL)")
    env.commit()
    drift = env.manager.check_schema_drift()
    env.commit()
    if ctx.outcome == LIMITED:
        assert drift == {vd.name: "triggers_disabled" for vd in env.views}, drift
        env.manager.repair_view()
        env.commit()
        ctx.check("after repair_view()", env)
        return
    try:
        ctx.check("writes with the maintenance triggers disabled", env)
    except ViewMismatch as exc:
        raise ViewMismatch(f"{exc}\n  check_schema_drift() = {drift}") from None


@cell("pg_session_replication_role", AREA_B,
      "Writes under session_replication_role = replica (logical-replication apply workers, some restore tools)",
      "ENABLE (origin) triggers do not fire; not detectable afterwards. A real subscription was not exercised: "
      "apply workers use this setting",
      todo="Detect PostgreSQL Maintenance Triggers Disabled by `session_replication_role = replica` or "
           "`ALTER TABLE ... DISABLE TRIGGER`", postgres=WRONG)
def _pg_replica(ctx: Ctx) -> None:
    env = ctx.env
    w = ctx.other_conn()
    env.execute("SET session_replication_role = replica", w)
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (760, 1, 5, 1, NULL)", w)
    w.commit()
    ctx.check("write under session_replication_role = replica")


@cell("mysql_session_skip_flag", AREA_B,
      "An application session setting @__matview_skip_triggers (the bulk_changes() bypass flag)",
      "any session can set it and its writes are silently not maintained (documented)",
      todo="Close Unsafe DML Paths That Bypass Trigger-Based Maintenance", mysql=WRONG, mariadb=WRONG)
def _mysql_skip(ctx: Ctx) -> None:
    env = ctx.env
    w = ctx.other_conn()
    env.execute("SET @__matview_skip_triggers = 1", w)
    env.execute("INSERT INTO t (id, g, v, w, u) VALUES (770, 1, 5, 1, NULL)", w)
    w.commit()
    ctx.check("write with the session bypass flag")


# The four out-of-band DDL cells share one environment (one source table and
# one view each), so each cell pays no environment setup of its own.
_DDL_TABLES = ("xa", "xd", "xr", "xc")
_DDL_COLS = {"postgres": "id INT PRIMARY KEY, g INT, v BIGINT",
             "mysql": "id INT PRIMARY KEY, g INT, v BIGINT",
             "sqlite": "id INTEGER PRIMARY KEY, g INTEGER, v INTEGER"}


def _ddl_table_sql(family: str, table: str) -> str:
    return f"CREATE TABLE {table} ({_DDL_COLS[family]})" + (" ENGINE=InnoDB" if family == "mysql" else "")


def _ddl_view(table: str) -> ViewDef:
    return ViewDef(f"v_{table}", f"SELECT g, SUM(v) AS s, COUNT(*) AS c FROM {table} GROUP BY g")


_DDL_ENVS: Dict[str, DEnv] = {}


@pytest.fixture(scope="module", autouse=True)
def _close_ddl_envs():
    yield
    while _DDL_ENVS:
        _DDL_ENVS.popitem()[1].close()


def _ddl_env(ctx: Ctx, table: str) -> DEnv:
    env = _DDL_ENVS.get(ctx.target.id)
    if env is None:
        env = DEnv(ctx.target, [_ddl_view(t) for t in _DDL_TABLES],
                   [_ddl_table_sql(ctx.family, t) for t in _DDL_TABLES],
                   [f"INSERT INTO {t} (id, g, v) VALUES " + _values([(i, i % 3, i * 3 - 11) for i in range(1, 13)])
                    for t in _DDL_TABLES])
        _DDL_ENVS[ctx.target.id] = env
    return env


def _ddl_check(env: DEnv, table: str, label: str, select: Optional[str] = None) -> None:
    vd = _ddl_view(table)
    saved = env.views
    env.views = [ViewDef(vd.name, select or vd.select)]
    try:
        env.check(label)
    finally:
        env.views = saved


def _ddl_drift(env: DEnv, table: str) -> str:
    drift = env.manager.check_schema_drift()
    env.commit()
    return drift[_ddl_view(table).name]


@cell("ddl_add_unused_column", AREA_B, "Out-of-band ALTER TABLE ADD COLUMN on a source (column unused by the view)",
      "maintenance continues exactly; check_schema_drift() reports drifted (the signature covers every column)",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _ddl_add(ctx: Ctx) -> None:
    env = _ddl_env(ctx, "xa")
    other = env.connect()
    env.execute("ALTER TABLE xa ADD COLUMN extra INT", other)
    other.commit()
    env.execute("INSERT INTO xa (id, g, v, extra) VALUES (100, 1, 4, 9), (101, 2, 5, NULL)")
    env.execute("UPDATE xa SET v = v + 1, extra = 1 WHERE id % 2 = 0")
    env.execute("DELETE FROM xa WHERE id % 5 = 0")
    env.commit()
    _ddl_check(env, "xa", "after ADD COLUMN")
    assert _ddl_drift(env, "xa") == "drifted"


@cell("ddl_drop_used_column", AREA_B, "Out-of-band ALTER TABLE DROP COLUMN of a column the view uses",
      "SQLite refuses the DDL (a trigger references the column); PostgreSQL/MySQL/MariaDB accept it, then every "
      "INSERT/UPDATE of the source fails in the maintenance trigger and check_schema_drift() reports invalid",
      sqlite=REFUSED, postgres=REFUSED, mysql=REFUSED, mariadb=REFUSED)
def _ddl_drop(ctx: Ctx) -> None:
    env = _ddl_env(ctx, "xd")
    other = env.connect()
    if ctx.family == "sqlite":
        ctx.fails("ALTER TABLE xd DROP COLUMN v", env, conn=other)
        other.rollback()
        env.execute("INSERT INTO xd (id, g, v) VALUES (100, 1, 4)")
        env.commit()
        _ddl_check(env, "xd", "after the refused DROP COLUMN")
        return
    env.execute("ALTER TABLE xd DROP COLUMN v", other)
    other.commit()
    ctx.fails("INSERT INTO xd (id, g) VALUES (100, 1)", env)
    ctx.fails("UPDATE xd SET g = 2 WHERE id = 1", env)
    env.rollback()
    assert _ddl_drift(env, "xd") == "invalid"


@cell("ddl_rename_used_column", AREA_B, "Out-of-band ALTER TABLE RENAME COLUMN of a column the view uses",
      "SQLite rewrites the trigger bodies, so maintenance continues on the renamed column while the registered "
      "definition no longer compiles (check_schema_drift() reports invalid); PostgreSQL/MySQL/MariaDB "
      "maintenance fails on every write, so writes fail closed",
      sqlite=LIMITED, postgres=REFUSED, mysql=REFUSED, mariadb=REFUSED)
def _ddl_rename(ctx: Ctx) -> None:
    env = _ddl_env(ctx, "xr")
    other = env.connect()
    env.execute("ALTER TABLE xr RENAME COLUMN v TO v2", other)
    other.commit()
    if ctx.family == "sqlite":
        env.execute("INSERT INTO xr (id, g, v2) VALUES (100, 1, 4), (101, 2, 5)")
        env.execute("UPDATE xr SET v2 = v2 + 1 WHERE id % 2 = 0")
        env.commit()
        _ddl_check(env, "xr", "maintenance on the renamed column",
                   select="SELECT g, SUM(v2) AS s, COUNT(*) AS c FROM xr GROUP BY g")
        assert _ddl_drift(env, "xr") == "invalid"
        return
    ctx.fails("INSERT INTO xr (id, g, v2) VALUES (100, 1, 4)", env)
    ctx.fails("UPDATE xr SET v2 = v2 + 1 WHERE id = 1", env)
    env.rollback()
    assert _ddl_drift(env, "xr") != "ok"


@cell("ddl_drop_recreate_source", AREA_B, "Out-of-band DROP TABLE and re-CREATE of a source, then writes",
      "the new table has no maintenance triggers, so the view is stale; check_schema_drift() reports it and "
      "repair_view() (MySQL/MariaDB: drop_view() + register_view(), they refuse a non-atomic rebuild) restores "
      "exactness",
      sqlite=LIMITED, postgres=LIMITED, mysql=LIMITED, mariadb=LIMITED)
def _ddl_recreate(ctx: Ctx) -> None:
    from matview import UnsafeReplacementError

    env = _ddl_env(ctx, "xc")
    other = env.connect()
    env.execute("DROP TABLE xc", other)
    env.execute(_ddl_table_sql(ctx.family, "xc"), other)
    env.execute("INSERT INTO xc (id, g, v) VALUES (1, 1, 1), (2, 1, 2), (50, 2, 3)", other)
    other.commit()
    assert _ddl_drift(env, "xc") != "ok"
    vd = _ddl_view("xc")
    if ctx.family == "mysql":
        with pytest.raises(UnsafeReplacementError):
            env.manager.repair_view(vd.name)
        env.rollback()
        env.manager.drop_view(vd.name)
        env.register(vd)
    else:
        env.manager.repair_view(vd.name)
    env.commit()
    _ddl_check(env, "xc", "DROP + CREATE of the source after repair")


# ---------------------------------------------------------------------------
# C. Transactions
# ---------------------------------------------------------------------------


@cell("savepoints_with_upserts", AREA_C, "Upserts and REPLACE inside nested savepoints, ROLLBACK TO and RELEASE",
      "twin", sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _savepoints(ctx: Ctx) -> None:
    env = ctx.env
    up = ctx.pick(
        postgres=f"INSERT INTO {{t}} AS x (id, g, v, w, u) VALUES {UPSERT_ROWS} "
                 "ON CONFLICT (id) DO UPDATE SET g = EXCLUDED.g, v = x.v + EXCLUDED.v",
        sqlite=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} "
               "ON CONFLICT (id) DO UPDATE SET g = excluded.g, v = {t}.v + excluded.v",
        mysql=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {UPSERT_ROWS} ON DUPLICATE KEY UPDATE g = VALUES(g), v = v + VALUES(v)")
    repl = ctx.pick(postgres="UPDATE {t} SET u = NULL, g = 4 WHERE id % 3 = 0",
                    sqlite="INSERT OR REPLACE INTO {t} (id, g, v, w, u) VALUES (6, 1, 1, 1, 70), (601, 1, 1, 1, 90)",
                    mysql="REPLACE INTO {t} (id, g, v, w, u) VALUES (6, 1, 1, 1, 70), (601, 1, 1, 1, 90)")
    ctx.both("DELETE FROM {t} WHERE id = 24")
    env.execute("SAVEPOINT s1")
    ctx.both(up)
    env.execute("SAVEPOINT s2")
    ctx.both(repl)
    ctx.check("inside nested savepoints", end_tx=False)
    env.execute("ROLLBACK TO SAVEPOINT s2")
    ctx.check("rolled back to s2", end_tx=False)
    ctx.both(repl)
    env.execute("RELEASE SAVEPOINT s2")
    env.execute("ROLLBACK TO SAVEPOINT s1")
    ctx.check("rolled back to s1", end_tx=False)
    ctx.both(up)
    env.execute("RELEASE SAVEPOINT s1")
    ctx.commit_check("savepoints")


_DEFER_SCHEMA = {
    "postgres": ["CREATE TABLE p (pid INT PRIMARY KEY)",
                 "CREATE TABLE c (id INT PRIMARY KEY, g INT, v BIGINT, k INT UNIQUE DEFERRABLE INITIALLY DEFERRED, "
                 "pid INT REFERENCES p (pid) DEFERRABLE INITIALLY DEFERRED)"],
    "sqlite": ["CREATE TABLE p (pid INTEGER PRIMARY KEY)",
               "CREATE TABLE c (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER, k INTEGER UNIQUE, "
               "pid INTEGER REFERENCES p (pid) DEFERRABLE INITIALLY DEFERRED)"],
}


@cell("deferred_constraints", AREA_C,
      "Deferred UNIQUE/FK constraints failing at COMMIT or at SET CONSTRAINTS ... IMMEDIATE",
      "SQLite: a failed COMMIT keeps the transaction open (rolled back by the test); MySQL/MariaDB have no deferred "
      "constraints", sqlite=EXACT, postgres=EXACT)
def _deferred(ctx: Ctx) -> None:
    env = ctx.fresh(_DEFER_SCHEMA, [ViewDef("v_dc", "SELECT g, SUM(v) AS s, COUNT(*) AS c FROM c GROUP BY g"),
                                    ViewDef("v_dk", "SELECT k, COUNT(*) AS c FROM c GROUP BY k")],
                    ["INSERT INTO p (pid) VALUES (1), (2)",
                     "INSERT INTO c (id, g, v, k, pid) VALUES " + _values([(i, i % 3, i, i, i % 2 + 1) for i in range(1, 10)])])
    before = env.snapshot()
    env.commit()
    env.execute("INSERT INTO c (id, g, v, k, pid) VALUES (100, 1, 5, 200, 99)")  # FK violation, deferred
    env.execute("UPDATE c SET g = 2 WHERE id < 4")
    ctx.check("deferred violation pending", env, end_tx=False)
    with pytest.raises(ctx.errors):
        env.commit()
    env.rollback()
    ctx.check("after the failed COMMIT", env)
    assert env.snapshot() == before
    env.commit()
    if ctx.family == "postgres":
        env.execute("UPDATE c SET k = k + 1")  # transiently duplicate keys, legal when deferred
        env.execute("INSERT INTO c (id, g, v, k, pid) VALUES (101, 2, 1, 5, 1)")
        ctx.fails("SET CONSTRAINTS ALL IMMEDIATE", env)
        ctx.check("SET CONSTRAINTS IMMEDIATE failed (in transaction)", env, end_tx=False)
        env.execute("DELETE FROM c WHERE id = 101")
        env.execute("SET CONSTRAINTS ALL IMMEDIATE")
        ctx.commit_check("deferred unique resolved", env)


@cell("autocommit_writers", AREA_C,
      "Writes from an autocommit connection (each statement its own transaction), including a failing statement",
      "", sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _autocommit(ctx: Ctx) -> None:
    env = ctx.env
    if ctx.family == "postgres":
        w = ctx.other_conn()
        w.autocommit = True
    elif ctx.family == "mysql":
        w = ctx.other_conn()
        w.autocommit(True)
    else:
        w = env.raw_sqlite(configure=True)
        w.isolation_level = None
    up = ctx.pick(postgres="INSERT INTO t (id, g, v, w, u) VALUES (1, 2, 3, 1, 10), (780, 1, 1, 1, NULL) "
                           "ON CONFLICT (id) DO UPDATE SET g = EXCLUDED.g",
                  sqlite="INSERT INTO t (id, g, v, w, u) VALUES (1, 2, 3, 1, 10), (780, 1, 1, 1, NULL) "
                         "ON CONFLICT (id) DO UPDATE SET g = excluded.g",
                  mysql="INSERT INTO t (id, g, v, w, u) VALUES (1, 2, 3, 1, 10), (780, 1, 1, 1, NULL) "
                        "ON DUPLICATE KEY UPDATE g = VALUES(g)")
    env.execute(up, w)
    with pytest.raises(ctx.errors):
        env.execute("INSERT INTO t (id, g, v) VALUES (781, 1, 1), (782, 1, 5000000)", w)
    env.execute("DELETE FROM t WHERE id % 4 = 1", w)
    env.execute("UPDATE t SET g = NULL WHERE id % 3 = 0", w)
    ctx.check("autocommit writes")


@cell("two_phase_commit", AREA_C,
      "Two-phase commit: PREPARE TRANSACTION / COMMIT|ROLLBACK PREPARED; XA START ... XA COMMIT|ROLLBACK",
      "PostgreSQL needs max_prepared_transactions > 0; the CI service runs with the default 0, so the cell is "
      "skipped there (verified locally on 18)", local_only=("postgres",),
      sqlite=NA, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _two_phase(ctx: Ctx) -> None:
    env = ctx.env
    w = ctx.other_conn()
    if ctx.family == "postgres":
        cur = env.conn.cursor()
        cur.execute("SHOW max_prepared_transactions")
        n = int(cur.fetchone()[0])
        env.commit()
        if n == 0:
            pytest.skip("max_prepared_transactions = 0 on this server")
        admin = ctx.other_conn()
        admin.autocommit = True
        try:
            for gid, finish in (("au_gid_r", "ROLLBACK PREPARED"), ("au_gid_c", "COMMIT PREPARED")):
                w.tpc_begin(gid)  # a plain string is used verbatim as the PostgreSQL gid
                env.execute("INSERT INTO t (id, g, v, w, u) VALUES (790, 1, 5, 1, NULL) "
                            "ON CONFLICT (id) DO UPDATE SET v = t.v + 1", w)
                env.execute("DELETE FROM t WHERE id % 5 = 0", w)
                w.tpc_prepare()
                ctx.check(f"prepared {gid} (not yet visible)")
                if finish == "COMMIT PREPARED":
                    w.tpc_commit()
                else:
                    w.tpc_rollback()
                ctx.check(f"{finish} {gid}")
        finally:
            cur = admin.cursor()
            cur.execute("SELECT gid FROM pg_prepared_xacts WHERE database = current_database() AND gid LIKE 'au_gid_%'")
            for (gid,) in cur.fetchall():
                cur.execute(f"ROLLBACK PREPARED '{gid}'")
        return
    w.autocommit(True)
    for xid, finish in (("au_xid_c", "XA COMMIT"), ("au_xid_r", "XA ROLLBACK")):
        env.execute(f"XA START '{xid}'", w)
        env.execute("INSERT INTO t (id, g, v, w, u) VALUES (790, 1, 5, 1, NULL) ON DUPLICATE KEY UPDATE v = v + 1", w)
        env.execute("DELETE FROM t WHERE id % 5 = 0", w)
        env.execute(f"XA END '{xid}'", w)
        env.execute(f"XA PREPARE '{xid}'", w)
        ctx.check(f"XA prepared {xid}")
        env.execute(f"{finish} '{xid}'", w)
        ctx.check(f"{finish} {xid}")


@cell("statement_fails_midway", AREA_C,
      "An upsert failing on its third row (CHECK) inside a multi-statement transaction, then more statements",
      "twin; PostgreSQL continues after ROLLBACK TO SAVEPOINT, the others after the statement-level rollback",
      sqlite=EXACT, postgres=EXACT, mysql=EXACT, mariadb=EXACT)
def _fails_midway(ctx: Ctx) -> None:
    rows = "(1, 1, 1, 1, 10), (2, 2, 2, 1, 20), (3, 3, 5000000, 1, 30), (790, 1, 1, 1, NULL)"
    up = ctx.pick(
        postgres=f"INSERT INTO {{t}} AS x (id, g, v, w, u) VALUES {rows} ON CONFLICT (id) DO UPDATE SET v = x.v + EXCLUDED.v, g = EXCLUDED.g",
        sqlite=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} ON CONFLICT (id) DO UPDATE SET v = {{t}}.v + excluded.v, g = excluded.g",
        mysql=f"INSERT INTO {{t}} (id, g, v, w, u) VALUES {rows} ON DUPLICATE KEY UPDATE v = v + VALUES(v), g = VALUES(g)")
    ctx.both("UPDATE {t} SET v = v + 1 WHERE id % 2 = 0")
    ctx.both_fail(up)
    ctx.check("after the failed upsert", end_tx=False)
    ctx.both("DELETE FROM {t} WHERE id % 7 = 0")
    ctx.commit_check("statement failing midway")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _cell_params() -> List[Any]:
    out = []
    for t in TARGETS:
        for c in CELLS:
            if all(c.outcome(p) == NA for p in FAMILY_PRODUCTS[t.family]):
                continue
            out.append(pytest.param(t, c, id=f"{t.id}-{c.id}"))
    if not out:
        out.append(pytest.param(None, None, id="no-target", marks=pytest.mark.skip(reason="no target")))
    return out


@pytest.mark.parametrize("target,c", _cell_params())
def test_interaction(target: Target, c: Cell) -> None:
    ctx = Ctx(target, c)
    try:
        if ctx.outcome == NA:
            pytest.skip(f"{c.id}: not applicable on {ctx.product}")
        need = c.min_version(ctx.product)
        if need is not None and ctx.version < need:
            pytest.skip(f"{c.id}: needs {ctx.product} {need}, server is {ctx.version}")
        if ctx.outcome == WRONG:
            try:
                c.run(ctx)
            except ViewMismatch as exc:
                pytest.xfail(f"SILENTLY-WRONG (TODO.md '{c.todo}'): {exc}")
            pytest.fail(f"{c.id} is recorded SILENTLY-WRONG on {ctx.product} but the view stayed exact: "
                        "update CELLS and regenerate the README matrix")
        c.run(ctx)
    finally:
        ctx.finish()


# ---------------------------------------------------------------------------
# The published matrix
# ---------------------------------------------------------------------------

_SHORT = {EXACT: "exact", LIMITED: "limited", REFUSED: "refused", WRONG: "**WRONG**", NA: "n/a"}


def render_matrix() -> str:
    lines = [
        "| Cell | Interaction | SQLite | PostgreSQL | MySQL | MariaDB | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    area = None
    for c in CELLS:
        if c.area != area:
            area = c.area
            lines.append(f"| **{area}** | | | | | | |")
        outs = []
        for p in PRODUCTS:
            o = _SHORT[c.outcome(p)]
            need = c.min_version(p)
            if need is not None and c.outcome(p) != NA:
                o += f" ({need[0]}.{need[1]}+)" if p in ("sqlite", "mariadb") else f" ({need[0]}+)"
            if p in c.local_only:
                o += " (verified locally, not in CI)"
            outs.append(o)
        note = c.note.replace("|", "/")
        if c.todo:
            note = (note + "; " if note else "") + f"TODO: {c.todo}"
        lines.append(f"| `{c.id}` | {c.what.replace('|', '/')} | " + " | ".join(outs) + f" | {note} |")
    counts = {p: collections.Counter(c.outcome(p) for c in CELLS) for p in PRODUCTS}
    lines.append("")
    lines.append("Counts per product (" + ", ".join(_SHORT[o].strip("*") for o in OUTCOMES) + "): " + "; ".join(
        f"{p} " + "/".join(str(counts[p][o]) for o in OUTCOMES) for p in PRODUCTS) + ".")
    return "\n".join(lines)


def _readme_block() -> str:
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), encoding="utf-8").read()
    assert README_BEGIN in text and README_END in text, "README.md lacks the generated DML matrix markers"
    return text.split(README_BEGIN, 1)[1].split(README_END, 1)[0].strip("\n")


def test_readme_matrix_is_current() -> None:
    assert _readme_block() == render_matrix(), (
        "README 'DML and trigger interaction support' is stale: run "
        "python3 test_native_dml_interactions.py --print-matrix and paste it between the markers")


def test_cells_are_well_formed() -> None:
    todo = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TODO.md"), encoding="utf-8").read()
    ids = [c.id for c in CELLS]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    for c in CELLS:
        assert any(c.outcome(p) != NA for p in PRODUCTS), c.id
        if any(c.outcome(p) == WRONG for p in PRODUCTS):
            assert c.todo and f"**{c.todo}**" in todo, f"{c.id}: SILENTLY-WRONG needs an existing TODO.md item: {c.todo!r}"
        if any(c.outcome(p) in (LIMITED, REFUSED) for p in PRODUCTS):
            assert c.note, f"{c.id}: limitations and refusals must be explained"


if __name__ == "__main__":
    if "--print-matrix" in sys.argv:
        print(render_matrix())
    else:
        sys.exit(pytest.main([__file__, "-v", "-rxXs"] + sys.argv[1:]))
