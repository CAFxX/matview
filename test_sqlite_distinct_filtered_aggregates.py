#!/usr/bin/env python3
"""SQLite DISTINCT / FILTER on emulated VAR_*/STDDEV_*/BIT_*/BOOL_* aggregates.

SQLite has no native variance, bitwise or boolean aggregate, so the project
lowers them (``_rewrite_sqlite_total_in_sql`` plus the Tier 1 fold). This file
checks the lowered/maintained result against the ORIGINAL view executed
natively by SQLite on an independent mirror database: every base-table
transaction is applied to both databases and, after every checkpoint of the
deterministic scenarios and after every randomized transaction (a transaction
may hold several statements; a rolled-back one is applied to the materialized
database only), each materialized reader must equal the original ``SELECT`` -- same rows, same
multiplicity, same values, same Python/SQLite storage class (``int`` vs
``float``), same sign of zero. There is no tolerance anywhere in this file.

Oracle. The mirror connection carries *reference* aggregates registered as
application-defined functions (SQLite itself applies DISTINCT and FILTER to
them): BIT_*/BOOL_* in Python integer arithmetic, VAR_*/STDDEV_* as the exact
rational variance rounded once (``fractions.Fraction``; the square root with
100 significant digits, then rounded once to binary64). The mirror never sees
the materialized-view machinery and the materialized-view connection never
sees the reference functions, so a lowering that leaked a call to them would
fail at install time rather than pass silently.

Known exclusion (TODO.md "Make SQLite VAR/STDDEV Correctly Rounded"). SQLite
float VAR/STDDEV are computed by shifted sums in binary64, which is not
correctly rounded in general (the same defect as the existing non-DISTINCT
Tier 1 fold). The exact differential therefore draws VAR/STDDEV inputs from
small dyadic rationals and evaluates VAR/STDDEV only for groups whose number
of distinct (filtered) inputs is 0, 1, 2 or 4 (at most 2 for STDDEV_SAMP;
``stat()`` below): then every shifted sum, the mean and ``M2`` are exactly
representable and at most one rounding remains, so the product must be
bit-identical to the correctly rounded reference. For every other group ``stat()`` yields NULL on both sides, and
the unrestricted domain is pinned by
``test_known_defect_float_statistics_are_not_correctly_rounded``, which asserts
that the discrepancy still exists (same groups and NULL pattern, different
float bits) and fails once it is fixed, forcing the exclusion to be removed.

Limits: CPython ``sqlite3`` and SQLCipher only (the reference aggregates need
a DB-API ``create_aggregate``; APSW and libSQL skip), one base table family
(``t``, ``d``, ``tc``), about twenty hand-picked view shapes, seeded random
DML (``MATVIEW_DISTINCT_FILTER_SEEDS`` seeds, default 2, 40 transactions each).
"""

from __future__ import annotations

import math
import os
import random
import re
import sqlite3
import unittest
import warnings
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matview import MaterializedViewManager
from matview.diagnostics import MatviewDiagnosticWarning, Strategy as DiagnosticStrategy
from matview_sync import Dialect, MaintenanceStrategy, UnsupportedMaterializedViewError

def _connect() -> Any:
    """In-memory connection of the driver under test: CPython's ``sqlite3`` by
    default, SQLCipher with ``SQLITE_COMPAT_ENGINE=sqlcipher``. APSW and libSQL
    have no DB-API ``create_aggregate`` for the reference aggregates, so those
    engines skip this file (they are exercised by ``test_matview_sync.py``)."""
    engine = os.environ.get("SQLITE_COMPAT_ENGINE", "").strip().lower()
    if engine == "sqlcipher":
        from sqlcipher3 import dbapi2 as sqlcipher_mod  # type: ignore[import-not-found]  # optional driver

        conn = sqlcipher_mod.connect(":memory:")
        conn.execute("PRAGMA key = 'matview-sqlcipher-secret';")
        return conn
    if engine not in ("", "sqlite3", "sqlite"):
        raise unittest.SkipTest(f"reference aggregates cannot be registered on SQLITE_COMPAT_ENGINE={engine}")
    return sqlite3.connect(":memory:")


def _ensure_sqrt(conn: Any) -> None:
    """The STDDEV lowering calls SQRT(), which SQLite only has when built with
    SQLITE_ENABLE_MATH_FUNCTIONS (some SQLCipher builds lack it; see TODO.md).
    ``math.sqrt`` is the same correctly rounded IEEE operation."""
    try:
        conn.execute("SELECT SQRT(4.0)").fetchall()
    except Exception:  # driver-specific OperationalError class
        conn.create_function("SQRT", 1, lambda v: None if v is None or v < 0 else math.sqrt(v))


SCHEMA = """
CREATE TABLE d (g INTEGER PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE t (
    id INTEGER PRIMARY KEY,
    g INTEGER,
    x REAL,
    b INTEGER,
    flag INTEGER,
    active INTEGER NOT NULL
);
CREATE TABLE tc (id INTEGER PRIMARY KEY, g TEXT, x REAL);
"""


def _same_value(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return bool(a == b)


def _same_rows(mv: Sequence[Sequence[Any]], live: Sequence[Sequence[Any]]) -> bool:
    if len(mv) != len(live):
        return False
    a = sorted((tuple(r) for r in mv), key=repr)
    b = sorted((tuple(r) for r in live), key=repr)
    return all(len(x) == len(y) and all(_same_value(p, q) for p, q in zip(x, y)) for x, y in zip(a, b))


# --------------------------------------------------------------------------
# Reference aggregates (oracle only)
# --------------------------------------------------------------------------

def _exact_variance(vals: Sequence[float], sample: bool) -> Optional[Fraction]:
    n = len(vals)
    if n == 0 or (sample and n < 2):
        return None
    fr = [Fraction(v) for v in vals]
    mean = sum(fr, Fraction(0)) / n
    m2 = sum(((f - mean) ** 2 for f in fr), Fraction(0))
    return m2 / (n - 1 if sample else n)


def _round_sqrt(fr: Fraction) -> float:
    getcontext().prec = 100
    return float((Decimal(fr.numerator) / Decimal(fr.denominator)).sqrt())


def _stat_class(sample: bool, std: bool) -> type:
    class _Stat:
        def __init__(self) -> None:
            self.vals: List[float] = []

        def step(self, x: Any) -> None:
            if x is not None:
                self.vals.append(float(x))

        def finalize(self) -> Optional[float]:
            v = _exact_variance(self.vals, sample)
            if v is None:
                return None
            return _round_sqrt(v) if std else float(v)

    return _Stat


def _sqlite_int(v: Any) -> int:
    """SQLite's operand coercion for ``&``, ``|`` and ``>>``: INTEGER stays, REAL truncates
    toward zero, TEXT uses its leading integer digits (0 when there are none)."""
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("utf-8", "replace")
    if isinstance(v, str):
        m = re.match(r"\s*[+-]?\d+", v)
        return int(m.group(0)) if m else 0
    return int(v)


def _bit_class(op: str) -> type:
    class _Bit:
        def __init__(self) -> None:
            self.acc: Optional[int] = None

        def step(self, x: Any) -> None:
            if x is None:
                return
            v = _sqlite_int(x)
            if self.acc is None:
                self.acc = v
            elif op == "and":
                self.acc &= v
            elif op == "or":
                self.acc |= v
            else:
                self.acc ^= v

        def finalize(self) -> Optional[int]:
            return self.acc

    return _Bit


_NUMERIC_PREFIX = re.compile(r"\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def _sqlite_truth(v: Any) -> bool:
    """SQLite's own boolean coercion (``WHERE v`` / ``CASE WHEN v``): the value is
    converted with NUMERIC affinity and is true iff the result is not zero. Text
    uses its longest leading numeric prefix, so '0', '0.0', '', 'abc' are false
    and '2x', ' 1', '1e0' are true; this is NOT Python's ``bool()``."""
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("utf-8", "replace")
    if isinstance(v, str):
        m = _NUMERIC_PREFIX.match(v)
        return bool(m) and float(m.group(0)) != 0.0
    return bool(v != 0)


def _bool_class(any_: bool) -> type:
    class _Bool:
        def __init__(self) -> None:
            self.truth: List[bool] = []

        def step(self, x: Any) -> None:
            if x is not None:
                self.truth.append(_sqlite_truth(x))

        def finalize(self) -> Optional[int]:
            if not self.truth:
                return None
            return int(any(self.truth) if any_ else all(self.truth))

    return _Bool


def register_reference_aggregates(conn: Any) -> None:
    for name, cls in (
        ("VAR_POP", _stat_class(False, False)),
        ("VAR_SAMP", _stat_class(True, False)),
        ("STDDEV_POP", _stat_class(False, True)),
        ("STDDEV_SAMP", _stat_class(True, True)),
        ("BIT_AND", _bit_class("and")),
        ("BIT_OR", _bit_class("or")),
        ("BIT_XOR", _bit_class("xor")),
        ("BOOL_AND", _bool_class(False)),
        ("BOOL_OR", _bool_class(True)),
    ):
        conn.create_aggregate(name, 1, cls)


# --------------------------------------------------------------------------
# View shapes
# --------------------------------------------------------------------------

def stat(fn: str, arg: str = "x", flt: Optional[str] = None) -> str:
    """VAR/STDDEV over DISTINCT ``arg`` (optionally FILTERed), restricted to the
    domain where the correctly rounded result needs no intermediate rounding in
    the product: 0, 1, 2 or 4 distinct dyadic inputs (n and n - 1 for the
    population/sample divisor must be 1, 2 or 4 for VAR, and the variance must
    itself be exact for STDDEV, so STDDEV_SAMP allows at most 2). NULL otherwise."""
    f = f" FILTER (WHERE {flt})" if flt else ""
    counts = "0, 1, 2" if fn == "STDDEV_SAMP" else "0, 1, 2, 4"
    return f"CASE WHEN COUNT(DISTINCT {arg}){f} IN ({counts}) THEN {fn}(DISTINCT {arg}){f} END"


def _four_stats(arg: str = "x", flt: Optional[str] = None) -> str:
    return ", ".join(
        f"{stat(fn, arg, flt)} AS {al}"
        for fn, al in (("VAR_POP", "vp"), ("VAR_SAMP", "vs"), ("STDDEV_POP", "sp"), ("STDDEV_SAMP", "ss"))
    )


_BIT_BOOL_DISTINCT = (
    "BIT_XOR(DISTINCT b) AS bx, BIT_AND(DISTINCT b) AS ba, BIT_OR(DISTINCT b) AS bo, "
    "BOOL_AND(DISTINCT flag) AS bl_and, BOOL_OR(DISTINCT flag) AS bl_or"
)

VIEWS: Dict[str, str] = {
    # Global aggregates (one row even over an empty table).
    "mv_glob": (
        f"SELECT COUNT(*) AS c, {_four_stats()}, {_BIT_BOOL_DISTINCT}, "
        f"COALESCE({stat('VAR_POP')}, -1.0) AS vp_coal FROM t"
    ),
    "mv_grp": f"SELECT g, {_four_stats()}, {_BIT_BOOL_DISTINCT} FROM t GROUP BY g",
    "mv_filt": (
        f"SELECT g, {stat('VAR_POP', 'x', 'active = 1')} AS vp_f, "
        f"{stat('STDDEV_SAMP', 'x', 'active = 1')} AS ss_f, "
        "BIT_XOR(DISTINCT b) FILTER (WHERE active = 1) AS bx_f, "
        "BIT_AND(b) FILTER (WHERE active = 1) AS ba_f, "
        "BIT_OR(DISTINCT b) FILTER (WHERE active = 0) AS bo_f0, "
        "BOOL_AND(flag) FILTER (WHERE active = 1) AS bla_f, "
        "BOOL_OR(DISTINCT flag) FILTER (WHERE active = 1) AS blo_f, "
        "BOOL_AND(flag) FILTER (WHERE 0) AS bl_never "
        "FROM t GROUP BY g"
    ),
    "mv_filt_glob": (
        "SELECT BIT_XOR(DISTINCT b) FILTER (WHERE active = 1 AND g = 99) AS bx_none, "
        f"{stat('VAR_POP', 'x', 'active = 1 AND g = 99')} AS vp_none, "
        "BOOL_OR(flag) FILTER (WHERE active = 1) AS blo FROM t"
    ),
    "mv_having": f"SELECT g, {stat('VAR_POP')} AS vp FROM t GROUP BY g HAVING {stat('VAR_POP')} > 0.5",
    "mv_expr_g": f"SELECT g + 10 AS gp, {stat('VAR_POP')} AS vp FROM t WHERE g IS NOT NULL GROUP BY gp",
    "mv_ord_g": f"SELECT g, {stat('VAR_SAMP')} AS vs, BIT_XOR(DISTINCT b) AS bx FROM t GROUP BY 1",
    "mv_lj": (
        f"SELECT d.g, d.label, {stat('VAR_POP', 't.x')} AS vp, BIT_XOR(DISTINCT t.b) AS bx "
        "FROM d LEFT JOIN t ON t.g = d.g AND t.active = 1 GROUP BY d.g, d.label"
    ),
    "mv_lj_all": (
        f"SELECT d.g, {stat('STDDEV_POP', 't.x')} AS sp, "
        "BIT_OR(t.b) FILTER (WHERE t.active = 1) AS bo_f, BOOL_OR(DISTINCT t.flag) AS blo "
        "FROM d LEFT JOIN t ON t.g = d.g GROUP BY d.g"
    ),
    "mv_cte": (
        "WITH act AS (SELECT DISTINCT g, x, b FROM t WHERE active = 1) "
        f"SELECT g, {stat('VAR_POP')} AS vp, BIT_XOR(DISTINCT b) AS bx FROM act GROUP BY g"
    ),
    "mv_cte_all": (
        "WITH c AS (SELECT DISTINCT g AS k, x AS val FROM t) "
        f"SELECT k + 1 AS x, {stat('VAR_POP', 'val')} AS vp FROM c GROUP BY x"
    ),
    "mv_cte_every": (
        "WITH every(a, b) AS (SELECT DISTINCT g, x FROM t WHERE g IS NOT NULL) "
        "SELECT a, SUM(b) AS s FROM every GROUP BY a"
    ),
    "mv_corr_sel": (
        f"SELECT d.g, (SELECT {stat('VAR_POP', 't.x')} FROM t WHERE t.g = d.g) AS vp, "
        "(SELECT BIT_XOR(DISTINCT t.b) FROM t WHERE t.g = d.g) AS bx FROM d"
    ),
    "mv_corr_whr": (
        f"SELECT d.g FROM d WHERE (SELECT {stat('VAR_POP', 't.x')} FROM t WHERE t.g = d.g) > 0.5"
    ),
    "mv_nested_from": (
        "SELECT g, BIT_XOR(DISTINCT bo) AS ox, BOOL_OR(DISTINCT ba) AS ob FROM ("
        "SELECT g, active, BIT_OR(b) AS bo, BOOL_AND(flag) AS ba FROM t GROUP BY g, active"
        ") AS sub GROUP BY g"
    ),
    "mv_nested_whr": (
        f"SELECT g, {stat('VAR_POP')} AS vp FROM t "
        f"WHERE x > (SELECT COALESCE({stat('VAR_POP')}, -1.0) FROM t WHERE g = 1) GROUP BY g"
    ),
    "mv_alias": (
        f"SELECT a.g, COUNT(DISTINCT a.id) AS c, {stat('VAR_POP', 'a.x', 'a.active = 1')} AS vp, "
        "BIT_XOR(ALL a.b) AS bx FROM t AS a GROUP BY a.g"
    ),
    "mv_t1_bit_bool": (
        "SELECT g, BIT_AND(b) AS ba, BIT_OR(b) AS bo, BIT_XOR(b) AS bx, "
        "BOOL_AND(flag) AS bland, BOOL_OR(flag) AS blor FROM t GROUP BY g"
    ),
    "mv_t1_glob_filt": (
        "SELECT BIT_AND(b) FILTER (WHERE active = 1) AS ba, BOOL_AND(flag) FILTER (WHERE active = 0) AS bla FROM t"
    ),
}

COLLATED_VIEWS: Dict[str, str] = {
    "mv_coll_ord": f"SELECT SUBSTR(g, 1, 1) AS gp, {stat('VAR_POP')} AS vp FROM tc GROUP BY 1 COLLATE NOCASE",
    "mv_coll_al": f"SELECT SUBSTR(g, 1, 1) AS gp, {stat('VAR_POP')} AS vp FROM tc GROUP BY gp COLLATE NOCASE",
    "mv_al_expr": f"SELECT id % 2 AS gp, {stat('VAR_POP')} AS vp FROM tc GROUP BY gp + 1",
}


class _Env:
    """A materialized-view database and an independent oracle mirror."""

    def __init__(self, views: Dict[str, str], seed_rows: bool = True) -> None:
        self.mv = _connect()
        self.live = _connect()
        MaterializedViewManager.configure_sqlite_connection(self.mv)
        _ensure_sqrt(self.mv)
        self.mv.executescript(SCHEMA)
        self.live.executescript(SCHEMA)
        register_reference_aggregates(self.live)
        self.views = dict(views)
        if seed_rows:
            for conn in (self.mv, self.live):
                conn.executemany("INSERT INTO d VALUES (?, ?)", [(1, "one"), (2, "two"), (3, "three"), (4, "four")])
            self.mv.commit()
            self.live.commit()
        self.mgr = MaterializedViewManager(self.mv, dialect=Dialect.SQLITE)
        self.plans: Dict[str, Any] = {}
        self.diagnostics: List[Any] = []

    def register(self) -> None:
        """Register every view; the announced diagnostics are collected, not printed."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for name, select_sql in self.views.items():
                self.plans[name] = self.mgr.register_view(f"CREATE VIEW {name} AS {select_sql};")
        self.diagnostics = [w.message.diagnostic for w in caught if isinstance(w.message, MatviewDiagnosticWarning)]
        other = [w for w in caught if not isinstance(w.message, MatviewDiagnosticWarning)]
        assert not other, [str(w.message) for w in other]

    def close(self) -> None:
        self.mv.close()
        self.live.close()

    def apply(self, stmts: Sequence[Tuple[str, Sequence[Any]]], commit: bool = True) -> None:
        """Run ``stmts`` as one transaction on the MV database; mirror it on the
        oracle only when committed (a rolled-back transaction changes nothing)."""
        for sql, params in stmts:
            self.mv.execute(sql, params)
        if commit:
            self.mv.commit()
            for sql, params in stmts:
                self.live.execute(sql, params)
            self.live.commit()
        else:
            self.mv.rollback()

    def check(self, tc: unittest.TestCase, step: str, key: Optional[Any] = None) -> None:
        for name, select_sql in self.views.items():
            got = self.mv.execute(f"SELECT * FROM {name}").fetchall()
            want = self.live.execute(select_sql).fetchall()
            if key is not None:
                got = [key(r) for r in got]
                want = [key(r) for r in want]
            tc.assertTrue(
                _same_rows(got, want),
                f"{step}: {name}\n  MV   {sorted(got, key=repr)}\n  LIVE {sorted(want, key=repr)}\n  SQL  {select_sql}",
            )


def _ins(i: int, g: Any, x: Any, b: Any, flag: Any, active: int) -> Tuple[str, Tuple[Any, ...]]:
    return ("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", (i, g, x, b, flag, active))


class TestDistinctAndFilteredStatBitBool(unittest.TestCase):
    env: _Env

    @classmethod
    def setUpClass(cls) -> None:
        # Registration dominates the runtime (about 0.2 s per view): register
        # once on an EMPTY base table (global views must install over it) and
        # let each test start from an emptied ``t``.
        cls.env = _Env(VIEWS)
        cls.env.register()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.env.apply([("DELETE FROM t", ())])

    def test_exact_differential_deterministic_scenarios(self) -> None:
        """Hand-written transitions. The data are chosen so that deduplication
        changes the result (duplicate values in every group) and so that every
        retraction of a duplicate leaves the DISTINCT value, or removes it."""
        env = self.env
        # Global aggregates over an EMPTY table: one row of NULLs / -1.0.
        self.assertEqual(
            env.mv.execute("SELECT * FROM mv_glob").fetchall(),
            env.live.execute(VIEWS["mv_glob"]).fetchall(),
        )
        env.check(self, "empty")
        env.apply(
            [
                _ins(1, 1, 1.0, 3, 1, 1),
                _ins(2, 1, 1.0, 3, 1, 1),  # duplicate of id 1
                _ins(3, 1, 3.0, 5, 1, 1),
                _ins(4, 1, 3.0, 5, 0, 0),  # inactive duplicate value
                _ins(5, 1, None, None, None, 1),  # NULL inputs
                _ins(6, 2, 0.5, 7, 1, 1),
                _ins(7, 2, 0.5, 7, 1, 1),
                _ins(8, 2, 2.5, 7, 1, 1),
                _ins(9, 2, -3.5, 7, 1, 1),
                _ins(10, 3, 4.0, 12, 0, 1),  # single distinct value, FALSE flags
                _ins(11, 3, 4.0, 12, 0, 1),
                _ins(12, 4, None, None, None, 1),  # all-NULL group
                _ins(13, None, 7.25, -9223372036854775808, 1, 1),  # NULL group key, int64 min
                _ins(14, None, 7.25, -1, 1, 1),
                _ins(15, None, -3.5, 2**62 + 1, 3, 1),
                _ins(16, 5, 1.0, 1, 1, 1),  # TRUE and NULL flags only: BOOL_AND must ignore the NULL
                _ins(17, 5, None, None, None, 1),
                _ins(18, 5, 1.0, 1, 1, 1),
                _ins(19, 6, 1.0, 1, "abc", 1),  # TEXT flags use SQLite's numeric coercion, not Python bool()
                _ins(20, 6, 2.5, 1, "2x", 1),
                _ins(21, 7, 1.0, 1, "0.0", 1),
                _ins(22, 7, 1.0, 1, "", 1),
                _ins(23, 7, 0.5, 1, 0.5, 1),
                _ins(24, 8, 1.0, "abc", 1, 1),  # sole row of a group holds a non-integer bit operand
                _ins(25, 9, 2.0, 2.5, 1, 1),
            ]
        )
        env.check(self, "after seed rows")
        # Delete one of two duplicates: DISTINCT result must not move.
        env.apply([("DELETE FROM t WHERE id = ?", (2,))])
        env.check(self, "delete one duplicate")
        # Delete the last copy: the distinct value disappears.
        env.apply([("DELETE FROM t WHERE id = ?", (1,))])
        env.check(self, "delete last copy")
        env.apply([("DELETE FROM t WHERE id IN (3, 4)", ())])
        env.check(self, "group 1 reduced to NULL inputs only")
        # Group move that keeps a value in the old group via a duplicate.
        env.apply([("UPDATE t SET g = 3 WHERE id = 7", ())])  # group 2 keeps id 6's 0.5; group 3 gains it
        env.check(self, "group move of a duplicate")
        env.apply([("UPDATE t SET x = 4.0, b = -2, flag = 2, active = 1 WHERE id = 6", ())])
        env.check(self, "update of aggregate inputs")
        env.apply([("UPDATE t SET active = 0 WHERE g = 2", ())])
        env.check(self, "FILTER predicate flips for a whole group")
        # Rolled-back transaction: nothing may change, MV must still equal the oracle.
        env.apply(
            [
                ("DELETE FROM t WHERE g = 2", ()),
                _ins(100, 2, 1.5, 1, 0, 1),
            ],
            commit=False,
        )
        env.check(self, "after rollback")
        # UPSERT / REPLACE / bulk.
        env.apply(
            [
                (
                    "INSERT INTO t VALUES (6, 3, 0.5, 1, 0, 1) ON CONFLICT(id) DO UPDATE "
                    "SET g = excluded.g, x = excluded.x, b = excluded.b",
                    (),
                ),
                ("INSERT OR REPLACE INTO t VALUES (10, 4, 2.0, 6, NULL, 0)", ()),
                ("INSERT INTO t SELECT id + 1000, g, x, b, flag, active FROM t WHERE id % 2 = 1", ()),
            ]
        )
        env.check(self, "upsert, replace and bulk insert")
        env.apply([("DELETE FROM t", ())])
        env.check(self, "all rows deleted")
        env.apply([_ins(200, 1, 2.0, 1, 1, 1)])
        env.check(self, "single row after empty")

    def test_exact_differential_randomized(self) -> None:
        """Seeded random DML (INSERT, UPDATE, DELETE, UPSERT, bulk, rollback,
        multi-statement transactions) checked after every transaction."""
        seeds = range(int(os.environ.get("MATVIEW_DISTINCT_FILTER_SEEDS", "2")))
        xs: List[Any] = [None, -3.5, -1.0, 0.0, 0.5, 2.0, 2.5, 4.0, 7.25]
        bs: List[Any] = [None, -(2**63), -9, -2, -1, 0, 3, 5, 7, 12, 2**62 + 1, "abc", "", 2.5]
        flags: List[Any] = [None, 0, 1, 2, 3, 10, 0.5, "abc", "", "2x", "0.0 ", " 1"]
        gs: List[Any] = [1, 2, 3, 4, None]
        for seed in seeds:
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                env = self.env
                env.apply([("DELETE FROM t", ())])
                nid = 1

                def row() -> Tuple[Any, ...]:
                    nonlocal nid
                    nid += 1
                    return (nid, rng.choice(gs), rng.choice(xs), rng.choice(bs), rng.choice(flags), rng.randint(0, 1))

                env.apply([("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", row()) for _ in range(rng.randint(0, 6))])
                for step in range(40):
                    ids = [r[0] for r in env.live.execute("SELECT id FROM t")]
                    stmts: List[Tuple[str, Tuple[Any, ...]]] = []
                    for _ in range(rng.choice((1, 1, 1, 2, 3))):
                        k = rng.random()
                        if k < 0.30 or not ids:
                            stmts.append(("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", row()))
                        elif k < 0.45:
                            stmts.append(("DELETE FROM t WHERE id = ?", (rng.choice(ids),)))
                        elif k < 0.58:
                            stmts.append(
                                ("UPDATE t SET x = ?, b = ?, flag = ? WHERE id = ?",
                                 (rng.choice(xs), rng.choice(bs), rng.choice(flags), rng.choice(ids)))
                            )
                        elif k < 0.70:
                            stmts.append(
                                ("UPDATE t SET g = ?, active = ? WHERE id = ?",
                                 (rng.choice(gs), rng.randint(0, 1), rng.choice(ids)))
                            )
                        elif k < 0.78:
                            stmts.append(("UPDATE t SET x = -x WHERE g IS ?", (rng.choice(gs),)))
                        elif k < 0.84:
                            stmts.append(("DELETE FROM t WHERE g IS ? AND active = ?", (rng.choice(gs), rng.randint(0, 1))))
                        elif k < 0.90:
                            r = row()
                            stmts.append(
                                ("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE "
                                 "SET x = excluded.x, g = excluded.g, active = excluded.active",
                                 (rng.choice(ids),) + r[1:])
                            )
                        elif k < 0.95:
                            stmts.append(("INSERT OR REPLACE INTO t VALUES (?, ?, ?, ?, ?, ?)", (rng.choice(ids),) + row()[1:]))
                        else:
                            base = nid  # every existing id is <= nid: copies land in (nid, 2 * nid]
                            nid = 2 * base
                            stmts.append(
                                ("INSERT INTO t SELECT id + ?, g, x, b, flag, active FROM t WHERE id % 3 = ?",
                                 (base, rng.randint(0, 2)))
                            )
                    env.apply(stmts, commit=rng.random() > 0.15)
                    env.check(self, f"seed {seed} step {step}: {stmts}")

    def test_initial_population_over_existing_rows(self) -> None:
        """Views registered over a populated table (initial population / backfill
        SQL, a different code path from trigger maintenance) equal the original
        view, and keep doing so under further DML."""
        names = ("mv_grp", "mv_filt", "mv_filt_glob", "mv_lj", "mv_t1_bit_bool", "mv_t1_glob_filt")
        rng = random.Random(11)
        xs: List[Any] = [None, -3.5, 0.0, 0.5, 2.5, 4.0]
        bs: List[Any] = [None, -(2**63), -2, 3, 5, 12, 2**62 + 1, "abc", ""]
        flags: List[Any] = [None, 0, 1, 2, "abc", "2x", ""]
        rows = [
            (i, rng.choice([1, 2, 3, None]), rng.choice(xs), rng.choice(bs), rng.choice(flags), rng.randint(0, 1))
            for i in range(1, 41)
        ]
        # A group whose only non-NULL flags are TRUE (BOOL_AND must ignore NULL, not read it as FALSE).
        rows += [(41, 4, 1.0, 1, 1, 1), (42, 4, None, None, None, 1), (43, 4, 1.0, 1, 1, 1)]
        env = _Env({n: VIEWS[n] for n in names})
        try:
            for conn in (env.mv, env.live):
                conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?, ?, ?)", rows)
                conn.commit()
            env.register()
            env.check(self, "initial population")
            env.apply([("DELETE FROM t WHERE id = 42", ()), _ins(100, 4, None, None, None, 1)])
            env.check(self, "after DML")
        finally:
            env.close()

    def test_adversarial_names_aliases_and_derived_stars(self) -> None:
        """Shapes the lowering must keep exact or refuse loudly: a derived-table
        ``SELECT *`` whose hidden column collides with a GROUP BY alias, user
        columns named like the generated ones, and any alias named OLD/NEW
        (trigger-record names; a column alias of that name stays fine)."""
        schema = (
            "CREATE TABLE ck (id INTEGER PRIMARY KEY, k INTEGER, x INTEGER);"
            "CREATE TABLE sh (id INTEGER PRIMARY KEY, __mv_g0 INTEGER, __mv_dv INTEGER, x INTEGER);"
            "CREATE TABLE oa (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER);"
        )
        views = {
            "v_star": (
                "SELECT x AS k, BIT_XOR(DISTINCT x) AS bx FROM "
                "(SELECT * FROM ck UNION ALL SELECT * FROM ck WHERE 0) AS d GROUP BY k"
            ),
            "v_shadow": "SELECT __mv_g0, BIT_XOR(DISTINCT x) AS bx, BIT_XOR(DISTINCT __mv_dv) AS bd FROM sh GROUP BY __mv_g0",
            "v_oldcol": "SELECT g AS new, MIN(x) AS m, COUNT(DISTINCT x) AS c FROM oa GROUP BY g",
        }
        mv, live = _connect(), _connect()
        try:
            MaterializedViewManager.configure_sqlite_connection(mv)
            for c in (mv, live):
                c.executescript(schema)
            register_reference_aggregates(live)
            rows = [
                ("INSERT INTO ck VALUES (?, ?, ?)", (1, 1, 1)),
                ("INSERT INTO ck VALUES (?, ?, ?)", (2, 1, 2)),
                ("INSERT INTO sh VALUES (?, ?, ?, ?)", (1, 1, 1, 1)),
                ("INSERT INTO sh VALUES (?, ?, ?, ?)", (2, 2, 4, 2)),
                ("INSERT INTO oa VALUES (?, ?, ?)", (1, 1, 1)),
                ("INSERT INTO oa VALUES (?, ?, ?)", (2, 1, 5)),
                ("INSERT INTO oa VALUES (?, ?, ?)", (3, 2, 0)),
            ]
            for sql, p in rows:
                mv.execute(sql, p)
                live.execute(sql, p)
            mv.commit()
            live.commit()
            mgr = MaterializedViewManager(mv, dialect=Dialect.SQLITE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MatviewDiagnosticWarning)
                for name, sel in views.items():
                    mgr.register_view(f"CREATE VIEW {name} AS {sel};")

            def check(step: str) -> None:
                for name, sel in views.items():
                    got = mv.execute(f"SELECT * FROM {name}").fetchall()
                    want = live.execute(sel).fetchall()
                    self.assertTrue(
                        _same_rows(got, want), f"{step}: {name}\n  MV   {sorted(got, key=repr)}\n  LIVE {sorted(want, key=repr)}"
                    )

            check("initial")
            for sql in (
                "DELETE FROM oa WHERE id = 1",
                "UPDATE sh SET x = 3 WHERE id = 1",
                "INSERT INTO ck VALUES (3, 1, 6)",
                "UPDATE oa SET x = 7 WHERE id = 3",
            ):
                mv.execute(sql)
                live.execute(sql)
                mv.commit()
                live.commit()
                check(sql)
            # Loud refusals instead of silently wrong results: any table/join/subquery/CTE alias
            # named OLD or NEW (any case, any quoting, qualified or not, whatever the aggregate),
            # a FROM-less aggregate with WHERE, and window forms of the emulated aggregates.
            refused = [
                "SELECT g, VAR_POP(DISTINCT x) AS v FROM oa AS NEW GROUP BY g",
                "SELECT new.g AS g, BOOL_OR(DISTINCT new.x) AS r FROM oa AS new GROUP BY new.g",
                "SELECT old.g, BIT_OR(DISTINCT old.x) AS r FROM oa AS old GROUP BY old.g",
                "SELECT OLD.g, MAX(OLD.x) AS m FROM oa AS OLD GROUP BY OLD.g",
                'SELECT g, SUM("new".x) AS s FROM oa AS "new" GROUP BY g',
                "SELECT g, VAR_POP(old.x) AS s FROM oa AS `old` GROUP BY g",
                "SELECT a.g, COUNT(*) AS c FROM oa AS a JOIN oa AS New ON New.id = a.id GROUP BY a.g",
                "SELECT d.g, SUM(d.x) AS s FROM (SELECT g, x FROM oa) AS d GROUP BY d.g HAVING SUM(d.x) > (SELECT MAX(x) FROM oa AS old)",
                "SELECT g, EXISTS (SELECT BIT_XOR(DISTINCT 7) WHERE 0) AS e FROM oa",
                "SELECT id, BOOL_OR(x) OVER w AS r FROM oa WINDOW w AS (PARTITION BY g)",
                "SELECT id, VAR_POP(x) OVER (PARTITION BY g) AS r FROM oa",
                "SELECT id, BIT_XOR(x) OVER (PARTITION BY g) AS r FROM oa",
            ]
            for sel in refused:
                with self.assertRaises(UnsupportedMaterializedViewError, msg=sel):
                    mgr.register_view(f"CREATE VIEW v_refused AS {sel};")
            # A base table literally named old/new (any case, any quoting) is refused too.
            for tname in ('"old"', "New", '"NEW"', "`old`"):
                mv.execute(f"CREATE TABLE {tname} (id INTEGER PRIMARY KEY, g INTEGER, x INTEGER)")
                mv.commit()
                for sel in (
                    f"SELECT g, MAX(x) AS m FROM {tname} GROUP BY g",
                    f"SELECT g, BIT_XOR(DISTINCT x) AS b FROM {tname} GROUP BY g",
                    f"SELECT a.g, COUNT(*) AS c FROM oa AS a JOIN {tname} AS o ON o.id = a.id GROUP BY a.g",
                ):
                    with self.assertRaises(UnsupportedMaterializedViewError, msg=sel):
                        mgr.register_view(f"CREATE VIEW v_refused AS {sel};")
                mv.execute(f"DROP TABLE {tname}")
                mv.commit()
        finally:
            mv.close()
            live.close()

    def test_collated_and_expression_group_by(self) -> None:
        """GROUP BY ordinal/alias with COLLATE NOCASE and an expression alias.

        The key of a NOCASE group is the value of an arbitrary member row (SQL
        leaves it unspecified), so keys are compared case-folded; values are
        compared exactly."""
        env = _Env(COLLATED_VIEWS, seed_rows=False)
        try:
            for conn in (env.mv, env.live):
                conn.executemany(
                    "INSERT INTO tc VALUES (?, ?, ?)",
                    [(1, "a", 1.0), (2, "A", 1.0), (3, "A", 3.0), (4, "b", 0.5), (5, "B", 4.0)],
                )
                conn.commit()
            env.register()

            def fold(r: Tuple[Any, ...]) -> Tuple[Any, ...]:
                return (r[0].lower() if isinstance(r[0], str) else r[0],) + tuple(r[1:])

            env.check(self, "initial", key=fold)
            env.apply([("INSERT INTO tc VALUES (6, 'a', 2.5)", ())])
            env.check(self, "insert into collated group", key=fold)
            env.apply([("DELETE FROM tc WHERE id = 2", ())])
            env.check(self, "delete one of two duplicates in a collated group", key=fold)
            env.apply([("UPDATE tc SET g = 'B' WHERE id = 1", ())])
            env.check(self, "move between collated groups", key=fold)
        finally:
            env.close()

    def test_maintenance_strategy_per_shape(self) -> None:
        """The strategy actually chosen is part of the contract (tier standing
        order: never worse than recorded here). Grouped DISTINCT/FILTER
        VAR/STDDEV/BIT/BOOL are maintained by group-scoped recompute (Tier 2,
        history independent); non-DISTINCT shapes by the Tier 1 fold. Global
        DISTINCT aggregates and views with CTEs, derived tables or correlated
        subqueries are recomputed from the original query on every write
        (Tier 3, exact by construction): a known exclusion (TODO.md, SQLite
        DISTINCT VAR/STDDEV/BIT maintenance tiers) that must shrink, never
        grow, and that is announced at registration (never silent)."""
        tier3 = {
            # A global aggregate without an algebraic rule is announced by the exact
            # routing (UNRECOGNIZED_AGGREGATE_FULL_REFRESH); FULL_REFRESH_FALLBACK only
            # announces full refreshes that no more specific diagnostic names.
            "mv_glob": "no plain GROUP BY",
            "mv_filt_glob": "no plain GROUP BY",
            "mv_cte": "uses a CTE",
            "mv_cte_all": "uses a CTE",
            "mv_cte_every": "uses a CTE",
            "mv_corr_sel": "correlated) scalar subquery",
            "mv_corr_whr": "correlated) scalar subquery",
            "mv_nested_from": "derived table",
        }
        codes = {n: "UNRECOGNIZED_AGGREGATE_FULL_REFRESH" for n in ("mv_glob", "mv_filt_glob")}
        # Cheaper is fine (the set may shrink); a new Tier 3 fallback must fail.
        got = {n for n, p in self.env.plans.items() if p.strategy == MaintenanceStrategy.ATOMIC_QUERY_SYNC}
        self.assertLessEqual(got, set(tier3), "new Tier 3 (atomic_query_sync) fallback")
        lead = "exact result required: falling back to strategy full refresh of the original query. "
        for name, reason in tier3.items():
            if name not in got:
                self.assertEqual(self.env.plans[name].strategy_diagnostics, [], name)
                continue
            diags = self.env.plans[name].strategy_diagnostics
            self.assertEqual([d.code for d in diags], [codes.get(name, "FULL_REFRESH_FALLBACK")], name)
            self.assertEqual(diags[0].view, name)
            self.assertEqual(diags[0].strategy, DiagnosticStrategy.FULL_REFRESH, name)
            self.assertTrue(diags[0].message.startswith(lead), diags[0].message)
            self.assertIn(reason, diags[0].message, name)
            self.assertIn("To change it:", diags[0].message, name)
            self.assertIn("Cost:", diags[0].message, name)
        # Emitted at registration as Python warnings carrying the Diagnostic, once per fallback view.
        # The other announcements are master's exact-routing notes on Tier 2 plans
        # (group-scoped recompute); a full refresh is never announced for them.
        full = [(d.view, d.code) for d in self.env.diagnostics if d.strategy == DiagnosticStrategy.FULL_REFRESH]
        self.assertEqual(sorted(full), sorted((n, codes.get(n, "FULL_REFRESH_FALLBACK")) for n in got))
        for d in self.env.diagnostics:
            if d.strategy == DiagnosticStrategy.FULL_REFRESH:
                continue
            self.assertEqual(d.strategy, DiagnosticStrategy.GROUP_SCOPED_RECOMPUTE, d.view)
            self.assertEqual(self.env.plans[d.view].strategy, MaintenanceStrategy.GROUP_SCOPED_DELTA, d.view)
            self.assertTrue(d.message.startswith("exact result required: falling back to strategy "), d.message)
        for name in ("mv_grp", "mv_filt", "mv_lj", "mv_alias", "mv_having", "mv_expr_g", "mv_ord_g"):
            self.assertIn(
                self.env.plans[name].strategy,
                (MaintenanceStrategy.GROUP_SCOPED_DELTA, MaintenanceStrategy.ALGEBRAIC_DELTA),
                name,
            )
            # Never a full-refresh announcement; a Tier 2 plan may carry the routing
            # note (TODO.md: announce every DISTINCT group recompute consistently).
            self.assertEqual(
                [d.strategy for d in self.env.plans[name].strategy_diagnostics
                 if d.strategy != DiagnosticStrategy.GROUP_SCOPED_RECOMPUTE],
                [],
                name,
            )
        for name in ("mv_t1_bit_bool", "mv_t1_glob_filt"):
            self.assertEqual(self.env.plans[name].strategy, MaintenanceStrategy.ALGEBRAIC_DELTA, name)
            self.assertEqual(self.env.plans[name].strategy_diagnostics, [], name)

    def test_known_defect_float_statistics_are_not_correctly_rounded(self) -> None:
        """TODO.md "Make SQLite VAR/STDDEV Correctly Rounded": over decimal inputs
        and any number of distinct values the shifted-sum result differs from the
        correctly rounded variance in the last bits. Registration, DML and the
        oracle run outside any expected-failure scope (an SQL error is an error);
        the test asserts that the known numeric discrepancy still exists and fails
        when it disappears, which is the signal to remove the ``stat()`` guard
        and this test."""
        views = {
            "mv_dec": (
                "SELECT g, VAR_POP(DISTINCT x) AS vp, VAR_SAMP(DISTINCT x) AS vs, "
                "STDDEV_POP(DISTINCT x) AS sp, STDDEV_SAMP(DISTINCT x) AS ss FROM t GROUP BY g"
            ),
            "mv_dec_f": "SELECT g, VAR_POP(DISTINCT x) FILTER (WHERE active = 1) AS vp_f FROM t GROUP BY g",
        }
        rng = random.Random(7)
        env = _Env(views)
        try:
            env.register()
            mismatched: List[str] = []
            for i in range(1, 40):
                env.apply([_ins(i, rng.choice([1, 2]), round(rng.uniform(-100, 100), 2), None, None, rng.randint(0, 1))])
                for name, select_sql in views.items():
                    got = env.mv.execute(f"SELECT * FROM {name}").fetchall()
                    want = env.live.execute(select_sql).fetchall()
                    # Same groups and the same NULL pattern (structure), different float bits.
                    self.assertEqual(
                        sorted((r[0], tuple(v is None for v in r[1:])) for r in got),
                        sorted((r[0], tuple(v is None for v in r[1:])) for r in want),
                        f"insert {i}: {name} structure",
                    )
                    for g_row in got:
                        for w_row in want:
                            if g_row[0] == w_row[0]:
                                for gv, wv in zip(g_row[1:], w_row[1:]):
                                    self.assertIsInstance(gv, (float, type(None)))
                                    if gv != wv:
                                        mismatched.append(f"{name} insert {i} group {g_row[0]}: {gv!r} != {wv!r}")
            self.assertTrue(
                mismatched,
                "float VAR/STDDEV is now bit-identical to the correctly rounded reference: "
                "remove the stat() domain guard, this test and the README/TODO exclusion",
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
