#!/usr/bin/env python3
"""Group keys must follow the grouped expression's collation (exact vs the live view).

Regression origin: on SQLite, ``GROUP BY g`` over ``g TEXT COLLATE NOCASE``
split the groups ``'a'`` and ``'A'`` in the materialized view (two rows) while
the original view returns one group, even for plain ``COUNT(*)``.

Oracle: the original ``SELECT`` executed natively on an independent mirror
database that receives the same statements (never derived from the
materialization). Counts, sums and other aggregates are compared exactly. SQL
leaves the spelling shown for a case-insensitive group open (the value of some
member row), so keys are compared case-folded and, in addition, the displayed
spelling must be one of the group's current member values.

Native engines: SQLite always; MySQL/MariaDB with ``MYSQL_DSN``
(``mysql://user:pass@host:port/db``, product read from the server).
"""

from __future__ import annotations

import os
import random
import re
import sqlite3
import unittest
import warnings
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from matview import MaterializedViewManager, key_collation
from matview.diagnostics import MatviewDiagnosticWarning
from matview_sync import Dialect, MaintenanceStrategy

# (declared column collation, spellings of one logical key, a different key)
SQLITE_CASES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("COLLATE NOCASE", ("a", "A", "aB", "Ab", "AB"), "b"),
    ("COLLATE RTRIM", ("a", "a ", "a  "), "b"),
    ("COLLATE BINARY", ("a", "A"), "b"),
    ("", ("a", "A"), "b"),
)


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def _fold(key: Any, coll: str) -> Any:
    if not isinstance(key, str):
        return key
    if "NOCASE" in coll:
        return key.lower()
    if "RTRIM" in coll:
        return key.rstrip(" ")
    return key


class _Pair:
    """The materialized database and the mirror holding only base tables."""

    def __init__(self, schema: str) -> None:
        self.mv = sqlite3.connect(":memory:")
        self.live = sqlite3.connect(":memory:")
        MaterializedViewManager.configure_sqlite_connection(self.mv)
        self.mv.executescript(schema)
        self.live.executescript(schema)
        self.mgr = MaterializedViewManager(self.mv, dialect=Dialect.SQLITE)

    def register(self, name: str, select_sql: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mgr.register_view(f"CREATE VIEW {name} AS {select_sql};")

    def run(self, stmts: Sequence[Tuple[str, Sequence[Any]]], commit: bool = True) -> None:
        for sql, params in stmts:
            self.mv.execute(sql, params)
        if commit:
            self.mv.commit()
            for sql, params in stmts:
                self.live.execute(sql, params)
            self.live.commit()
        else:
            self.mv.rollback()

    def close(self) -> None:
        self.mv.close()
        self.live.close()


_SHAPE_SCHEMA = (
    "CREATE TABLE d (k TEXT COLLATE NOCASE PRIMARY KEY, w INTEGER);"
    "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, x INTEGER, k TEXT COLLATE NOCASE);"
    "CREATE TABLE u (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, x INTEGER);"
)
_SHAPE_VIEWS = {
    "ua": "SELECT g, COUNT(*) AS c, SUM(x) AS s FROM (SELECT g, x FROM t UNION ALL SELECT g, x FROM u) AS x GROUP BY g",
    "j2": "SELECT t.g AS g, COUNT(*) AS c FROM t JOIN u ON t.x = u.x GROUP BY t.g",
    "j": "SELECT t.g AS g, COUNT(*) AS c, SUM(d.w) AS s FROM t JOIN d ON t.k = d.k GROUP BY t.g",
    "hv": "SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING COUNT(*) > 1",
    "wh": "SELECT g, SUM(x) AS s FROM t WHERE x > 2 GROUP BY g",
    "gc": "SELECT g, COUNT(DISTINCT x) AS cd, GROUP_CONCAT(DISTINCT x) AS gc FROM t GROUP BY g",
    "mm": "SELECT g, MIN(x) AS mn, MAX(x) AS mx FROM t GROUP BY g",
    "cte": "WITH c AS (SELECT g, x FROM t) SELECT g, SUM(x) AS s FROM c GROUP BY g",
    "st": "SELECT g, AVG(x) AS a, COUNT(DISTINCT x) AS cd FROM t GROUP BY g",
    "nn": "SELECT g, COUNT(*) AS c FROM t WHERE g IS NOT NULL GROUP BY g",
    "dist": "SELECT DISTINCT g FROM t",
}


class TestSqliteGroupKeyCollation(unittest.TestCase):
    def _check(
        self,
        pair: _Pair,
        name: str,
        select_sql: str,
        coll: str,
        step: str,
        key_cols: Tuple[int, ...] = (0,),
        member_sql: Optional[str] = None,
    ) -> None:
        got = pair.mv.execute(f"SELECT * FROM {name}").fetchall()
        want = pair.live.execute(select_sql).fetchall()

        def norm(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
            return sorted(
                (tuple(_fold(v, coll) if i in key_cols else v for i, v in enumerate(r)) for r in rows), key=repr
            )

        ng, nw = norm(got), norm(want)
        self.assertEqual(len(ng), len(nw), f"{step}: {name} row count\n  MV   {got}\n  LIVE {want}")
        for g_row, w_row in zip(ng, nw):
            self.assertTrue(
                len(g_row) == len(w_row) and all(_same(a, b) for a, b in zip(g_row, w_row)),
                f"{step}: {name}\n  MV   {sorted(got, key=repr)}\n  LIVE {sorted(want, key=repr)}",
            )
        if member_sql is not None:  # the displayed spelling is a current member value
            members = {r[0] for r in pair.live.execute(member_sql).fetchall()}
            for row in got:
                self.assertIn(row[0], members, f"{step}: {name} shows a spelling no member row has")

    def test_single_column_key_all_collations(self) -> None:
        for decl, spellings, other in SQLITE_CASES:
            with self.subTest(collation=decl or "default"):
                pair = _Pair(f"CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT {decl}, x INTEGER);")
                try:
                    view = "SELECT g, COUNT(*) AS c, SUM(x) AS s, MIN(x) AS mn, MAX(x) AS mx FROM t GROUP BY g"
                    pair.register("v", view)
                    pair.register("vd", "SELECT g, COUNT(DISTINCT x) AS cd FROM t GROUP BY g")
                    members = "SELECT DISTINCT g FROM t"
                    chk = lambda step: (  # noqa: E731
                        self._check(pair, "v", view, decl, step, member_sql=members),
                        self._check(pair, "vd", "SELECT g, COUNT(DISTINCT x) AS cd FROM t GROUP BY g", decl, step),
                    )
                    chk("empty")
                    nid = 0
                    for sp in spellings:  # INSERT of every spelling, one by one
                        nid += 1
                        pair.run([("INSERT INTO t VALUES (?, ?, ?)", (nid, sp, nid))])
                        chk(f"insert {sp!r}")
                    pair.run([("INSERT INTO t VALUES (?, ?, ?)", (100, other, 7))])
                    chk("other key")
                    first, second = spellings[0], spellings[1]
                    # UPDATE changing only the case/trailing spaces of the key.
                    pair.run([("UPDATE t SET g = ? WHERE id = 1", (second,))])
                    chk("update spelling only")
                    pair.run([("UPDATE t SET g = ?, x = x + 1 WHERE id = 2", (first,))])
                    chk("update spelling and value")
                    # DELETE one of two spellings, then the rest.
                    pair.run([("DELETE FROM t WHERE id = 1", ())])
                    chk("delete one spelling")
                    pair.run([("UPDATE t SET g = ? WHERE id = 100", (spellings[-1],))])
                    chk("move other key into the group")
                    pair.run([("DELETE FROM t WHERE g = ?", (spellings[-1],))], commit=False)
                    chk("rolled back delete")
                    pair.run([("DELETE FROM t", ())])
                    chk("all deleted")
                finally:
                    pair.close()

    def test_expression_and_multi_column_keys_with_explicit_collate(self) -> None:
        pair = _Pair("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, h TEXT COLLATE NOCASE, x INTEGER);")
        try:
            views = {
                "v_expr": "SELECT g COLLATE NOCASE AS k, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY g COLLATE NOCASE",
                "v_lower": "SELECT lower(g) AS k, COUNT(*) AS c FROM t GROUP BY lower(g)",
                "v_multi": "SELECT g, h, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY g, h",
                "v_multi_c": (
                    "SELECT g COLLATE NOCASE AS gk, h, COUNT(*) AS c FROM t GROUP BY g COLLATE NOCASE, h"
                ),
                "v_rtrim": "SELECT g COLLATE RTRIM AS k, COUNT(*) AS c FROM t GROUP BY g COLLATE RTRIM",
            }
            for name, sel in views.items():
                pair.register(name, sel)
            rng = random.Random(5)
            spell = ["a", "A", "ab", "AB", "b", "a ", None]
            nid = 0

            def check(step: str) -> None:
                self._check(pair, "v_expr", views["v_expr"], "NOCASE", step)
                self._check(pair, "v_lower", views["v_lower"], "", step)
                self._check(pair, "v_rtrim", views["v_rtrim"], "RTRIM", step)
                # g is BINARY, h NOCASE: only h folds; compare with h folded.
                for name in ("v_multi", "v_multi_c"):
                    got = pair.mv.execute(f"SELECT * FROM {name}").fetchall()
                    want = pair.live.execute(views[name]).fetchall()
                    fold_cols = (1,) if name == "v_multi" else (0, 1)

                    def nrm(rows: List[Tuple[Any, ...]], cols: Tuple[int, ...] = fold_cols) -> List[Tuple[Any, ...]]:
                        return sorted(
                            (tuple(_fold(v, "NOCASE") if i in cols else v for i, v in enumerate(r)) for r in rows),
                            key=repr,
                        )

                    self.assertEqual(nrm(got), nrm(want), f"{step}: {name}\n  MV {got}\n  LIVE {want}")

            check("empty")
            for step in range(60):
                ids = [r[0] for r in pair.live.execute("SELECT id FROM t")]
                k = rng.random()
                if k < 0.45 or not ids:
                    nid += 1
                    stmt = ("INSERT INTO t VALUES (?, ?, ?, ?)", (nid, rng.choice(spell), rng.choice(spell), rng.randint(0, 9)))
                elif k < 0.65:
                    stmt = ("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
                elif k < 0.9:
                    stmt = ("UPDATE t SET g = ?, h = ? WHERE id = ?", (rng.choice(spell), rng.choice(spell), rng.choice(ids)))
                else:
                    stmt = ("UPDATE t SET x = x + 1 WHERE id = ?", (rng.choice(ids),))
                pair.run([stmt], commit=rng.random() > 0.1)
                check(f"step {step}: {stmt}")
        finally:
            pair.close()

    def test_every_maintenance_strategy_shape_under_nocase(self) -> None:
        """One NOCASE key through every maintenance family (algebraic joins,
        HAVING, WHERE, group-scoped recompute, CTE, MIN/MAX,
        DISTINCT aggregates) under seeded random DML."""
        schema, views = _SHAPE_SCHEMA, _SHAPE_VIEWS
        pair = _Pair(schema)
        try:
            for name, sel in views.items():
                pair.register(name, sel)
            rng = random.Random(3)
            spell = ["a", "A", "b", "B", "ab", None]
            for dk in ("a", "b", "ab"):
                pair.run([("INSERT INTO d VALUES (?, ?)", (dk, rng.randint(1, 5)))])

            def norm(name: str, rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
                out = []
                for r in rows:
                    r = tuple(r)
                    if name == "gc":  # GROUP_CONCAT order is unspecified
                        r = (r[0], r[1], tuple(sorted((r[2] or "").split(","))))
                    out.append((_fold(r[0], "NOCASE"),) + tuple(r[1:]))
                return sorted(out, key=repr)

            nid = 0
            for step in range(80):
                ids = [r[0] for r in pair.live.execute("SELECT id FROM t")]
                k = rng.random()
                u_ids = [r[0] for r in pair.live.execute("SELECT id FROM u")]
                if k < 0.5 or not ids:
                    nid += 1
                    stmt = ("INSERT INTO t VALUES (?, ?, ?, ?)", (nid, rng.choice(spell), rng.randint(0, 5), rng.choice(["a", "A", "b", "B"])))
                elif k < 0.58:
                    nid += 1
                    stmt = ("INSERT INTO u VALUES (?, ?, ?)", (nid, rng.choice(spell), rng.randint(0, 5)))
                elif k < 0.64 and u_ids:
                    stmt = ("DELETE FROM u WHERE id = ?", (rng.choice(u_ids),))
                elif k < 0.7:
                    stmt = ("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
                else:
                    stmt = (
                        "UPDATE t SET g = ?, x = ?, k = ? WHERE id = ?",
                        (rng.choice(spell), rng.randint(0, 5), rng.choice(["a", "A", "B", "b"]), rng.choice(ids)),
                    )
                pair.run([stmt], commit=rng.random() > 0.1)
                for name, sel in views.items():
                    got = norm(name, pair.mv.execute(f"SELECT * FROM {name}").fetchall())
                    want = norm(name, pair.live.execute(sel).fetchall())
                    self.assertEqual(got, want, f"step {step} {stmt}: {name}")
                    # The displayed spelling is a spelling of a current member row.
                    members = {r[0] for r in pair.live.execute("SELECT g FROM t")}
                    for row in pair.mv.execute(f"SELECT * FROM {name}").fetchall():
                        self.assertIn(row[0], members, f"step {step} {stmt}: {name} shows spelling {row[0]!r} no row has")
        finally:
            pair.close()

    def test_displayed_spelling_is_a_member_in_every_shape(self) -> None:
        """Two spellings of one key, then the row that supplied the stored spelling
        leaves (DELETE, key-changing UPDATE, spelling-only UPDATE), through every
        maintenance family: the shown spelling must always be a member's."""
        scenarios = (
            [("INSERT INTO t VALUES (1, 'A', 3, 'a')", ()), ("INSERT INTO t VALUES (2, 'a', 3, 'a')", ())],
            [("INSERT INTO t VALUES (1, 'a', 3, 'a')", ()), ("INSERT INTO t VALUES (2, 'A', 3, 'a')", ())],
        )
        tail = (
            [("DELETE FROM t WHERE id = 1", ())],
            [("UPDATE t SET g = 'b' WHERE id = 1", ())],
            [("UPDATE t SET g = 'a' WHERE id = 1", ()), ("UPDATE t SET g = 'A' WHERE id = 2", ())],
            [("DELETE FROM d WHERE k = 'a'", ())],
            [("DELETE FROM u", ())],
        )
        # One registered environment for every scenario (a scenario ends with the base
        # tables emptied, itself a maintained transition), not one per scenario.
        pair = _Pair(_SHAPE_SCHEMA)
        try:
            for name, sel in _SHAPE_VIEWS.items():
                pair.register(name, sel)
            for si, prelude in enumerate(scenarios):
                for ti, steps in enumerate(tail):
                    base = [("INSERT INTO d VALUES ('a', 5)", ()), ("INSERT INTO u VALUES (1, 'z', 3)", ())]
                    for stmt in [*base, *prelude, *steps]:
                        pair.run([stmt])
                        members = {r[0] for r in pair.mv.execute("SELECT g FROM t UNION ALL SELECT g FROM u")}
                        for name, sel in _SHAPE_VIEWS.items():
                            self.assertEqual(
                                self._oracle_mismatches_nocase(pair, name, sel), [], f"{si}/{ti} {stmt}: {name}"
                            )
                            for row in pair.mv.execute(f"SELECT * FROM {name}").fetchall():
                                self.assertIn(row[0], members, f"{si}/{ti} {stmt}: {name} shows {row[0]!r}")
                    for tbl in ("t", "u", "d"):
                        pair.run([(f"DELETE FROM {tbl}", ())])
        finally:
            pair.close()

    @staticmethod
    def _oracle_mismatches_nocase(pair: "_Pair", name: str, sel: str) -> List[Any]:
        """Rows of the original SELECT (same database) that have no equal MV row once
        keys are compared case-folded (GROUP_CONCAT order is unspecified)."""

        def norm(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
            out = []
            for r in rows:
                r = tuple(r)
                if name == "gc":
                    r = (r[0], r[1], tuple(sorted((r[2] or "").split(","))))
                out.append((_fold(r[0], "NOCASE"),) + tuple(r[1:]))
            return sorted(out, key=repr)

        got = norm(pair.mv.execute(f"SELECT * FROM {name}").fetchall())
        want = norm(pair.mv.execute(sel).fetchall())
        return [] if got == want else [("MV", got), ("LIVE", want)]

    def test_distinct_projection_follows_the_column_collation(self) -> None:
        pair = _Pair("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, x INTEGER);")
        try:
            view = "SELECT DISTINCT g FROM t"
            pair.register("vdist", view)
            for i, sp in enumerate(["a", "A", "b"], start=1):
                pair.run([("INSERT INTO t VALUES (?, ?, 1)", (i, sp))])
                self._check(pair, "vdist", view, "NOCASE", f"insert {sp}")
            pair.run([("DELETE FROM t WHERE id = 1", ())])
            self._check(pair, "vdist", view, "NOCASE", "delete one spelling")
            pair.run([("DELETE FROM t WHERE id = 2", ())])
            self._check(pair, "vdist", view, "NOCASE", "delete last spelling")
        finally:
            pair.close()

    _DISTINCT_SCHEMA = (
        "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, h TEXT COLLATE RTRIM, bb TEXT, x INTEGER);"
    )
    _DISTINCT_VIEWS = {
        "d1": "SELECT DISTINCT g FROM t",
        "d2": "SELECT DISTINCT g, h FROM t",
        "d3": "SELECT DISTINCT h FROM t",
        "d4": "SELECT DISTINCT g, bb FROM t WHERE x > 1",
        "d5": "SELECT DISTINCT lower(bb) AS k FROM t",
        "d6": "SELECT DISTINCT g COLLATE BINARY AS gb FROM t",
        "d7": "SELECT DISTINCT g, x FROM t",
    }

    def _distinct_failures(self, seeds: int, steps: int) -> List[str]:
        """Seeded random DML (inserts, deletes, key/spelling updates, NULLs, rollbacks)
        over single- and multi-column DISTINCT views on NOCASE / RTRIM / BINARY columns.
        Returns the divergences found: a row set that differs from the original view's
        (compared under the engine's own grouping), or a displayed row whose values
        no current member row has (all columns of a row from one member)."""
        spell = ["a", "A", "b", "B", "a ", "A ", "ab", None, "\u00e9", "\u00c9"]
        failures: List[str] = []
        for seed in range(seeds):
            pair = _Pair(self._DISTINCT_SCHEMA)
            try:
                for name, sel in self._DISTINCT_VIEWS.items():
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        pair.mgr.register_view(f"CREATE VIEW {name} AS {sel};")
                    if any("COLLATION_KEY_FALLBACK" in str(w.message) for w in caught):
                        failures.append(f"{name} unexpectedly fell back to a full refresh")
                rng = random.Random(seed)
                nid = 0
                for step in range(steps):
                    ids = [r[0] for r in pair.mv.execute("SELECT id FROM t")]
                    k = rng.random()
                    if k < 0.5 or not ids:
                        nid += 1
                        stmt: Tuple[str, Tuple[Any, ...]] = (
                            "INSERT INTO t VALUES (?, ?, ?, ?, ?)",
                            (nid, rng.choice(spell), rng.choice(spell), rng.choice(spell), rng.randint(0, 3)),
                        )
                    elif k < 0.75:
                        stmt = ("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
                    else:
                        stmt = (
                            "UPDATE t SET g = ?, h = ?, bb = ?, x = ? WHERE id = ?",
                            (rng.choice(spell), rng.choice(spell), rng.choice(spell), rng.randint(0, 3), rng.choice(ids)),
                        )
                    pair.run([stmt], commit=rng.random() > 0.1)
                    for name, sel in self._DISTINCT_VIEWS.items():
                        cols = [d[0] for d in pair.mv.execute(f"SELECT * FROM {name}").description]
                        member = " AND ".join(f'((m."{c}" COLLATE BINARY) IS v."{c}")' for c in cols)
                        orphans = pair.mv.execute(
                            f"SELECT COUNT(*) FROM {name} AS v WHERE NOT EXISTS "
                            f"(SELECT 1 FROM ({sel.replace('DISTINCT ', '', 1)}) AS m WHERE {member})"
                        ).fetchone()[0]
                        if orphans:
                            failures.append(f"seed {seed} step {step} {stmt}: {name} shows {orphans} row(s) no member has")
                        elif self._oracle_mismatches(pair.mv, name, sel):
                            failures.append(f"seed {seed} step {step} {stmt}: {name} differs from the original view")
                    if failures:
                        return failures
            finally:
                pair.close()
        return failures

    def test_spelling_follows_when_another_key_column_is_null(self) -> None:
        """A NULL in one key column must not stop the spelling refresh of the other
        (minimized from the DISTINCT fuzz: seed 2, step 51)."""
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, h TEXT COLLATE NOCASE, x INTEGER);"
        for sel in ("SELECT DISTINCT g, h FROM t", "SELECT g, h, COUNT(*) AS c FROM t GROUP BY g, h"):
            with self.subTest(view=sel):
                pair = _Pair(schema)
                try:
                    pair.register("v", sel)
                    pair.run([("INSERT INTO t VALUES (1, 'a', NULL, 1)", ()), ("INSERT INTO t VALUES (2, 'A', NULL, 1)", ())])
                    pair.run([("DELETE FROM t WHERE id = 1", ())])
                    self.assertEqual([r[0] for r in pair.mv.execute("SELECT g FROM v")], ["A"])
                    pair.run([("INSERT INTO t VALUES (3, 'a', 'b', 1)", ()), ("INSERT INTO t VALUES (4, 'A', 'B', 1)", ())])
                    pair.run([("DELETE FROM t WHERE id = 3", ())])
                    self.assertEqual(sorted(pair.mv.execute("SELECT g, h FROM v WHERE h IS NOT NULL").fetchall()), [("A", "B")])
                finally:
                    pair.close()

    def test_distinct_stays_incremental_and_exact(self) -> None:
        """SELECT DISTINCT over NOCASE/RTRIM/BINARY columns (single and multiple
        columns, expressions, NULLs, WHERE) is maintained incrementally, equals the
        original view after every transition, and shows a spelling a member has."""
        self.assertEqual(self._distinct_failures(seeds=6, steps=40), [])

    def test_distinct_differential_detects_a_byte_exact_key(self) -> None:
        """Mutant: a byte-exact (collation-blind) key splits NOCASE groups; the DISTINCT
        differential must fail."""
        from matview.dialects.base import DialectBaseMixin

        original = DialectBaseMixin._canonical_group_pk_expr
        DialectBaseMixin._canonical_group_pk_expr = (  # type: ignore[method-assign]
            lambda self, group_exprs, *, collation_aware=True: original(self, group_exprs, collation_aware=False)
        )
        try:
            failures = self._distinct_failures(seeds=3, steps=30)
        finally:
            DialectBaseMixin._canonical_group_pk_expr = original  # type: ignore[method-assign]
        self.assertTrue(failures, "a collation-blind key went undetected")

    def test_distinct_spelling_check_detects_a_stale_spelling(self) -> None:
        """Mutant: without the spelling refresh the DISTINCT differential must fail."""
        from matview.maintenance.single import Tier1SingleMixin as SingleTableMixin

        original = SingleTableMixin._key_spelling_refresh_stmts
        SingleTableMixin._key_spelling_refresh_stmts = lambda self, mv_table, record_prefix="OLD": []  # type: ignore[method-assign]
        try:
            failures = self._distinct_failures(seeds=6, steps=40)
        finally:
            SingleTableMixin._key_spelling_refresh_stmts = original  # type: ignore[method-assign]
        self.assertTrue(any("no member has" in f for f in failures), failures)

    # --- engine-oracle tests ------------------------------------------------

    @staticmethod
    def _oracle_mismatches(con: sqlite3.Connection, view: str, select_sql: str) -> int:
        """Number of groups on which the materialized view and the original SELECT (run
        on the same native database state) disagree: the union of both result sets
        is grouped by all columns with SQLite's own GROUP BY (so equal keys under the
        key's collation, and equal aggregates, meet in one bucket); every bucket must
        hold exactly one row of each side."""
        cols = [d[0] for d in con.execute(f"SELECT * FROM {view}").description]
        cs = ", ".join(f'"{c}"' for c in cols)
        sql = (
            f"SELECT COUNT(*) FROM (SELECT {cs}, SUM(src = 1) AS a, SUM(src = 2) AS b FROM ("
            f"SELECT {cs}, 1 AS src FROM {view} UNION ALL SELECT {cs}, 2 AS src FROM ({select_sql})"
            f") GROUP BY {cs}) WHERE a <> 1 OR b <> 1"
        )
        return int(con.execute(sql).fetchone()[0])

    EXPRESSION_KEYS = (
        # (expression over t(g NOCASE, b BINARY, r RTRIM, x), engine collation it compares under)
        "g", "(g)", "+g", "CAST(g AS TEXT)", "CAST(CAST(g AS TEXT) AS TEXT)", "g COLLATE NOCASE", "(g COLLATE BINARY)",
        "b COLLATE NOCASE", "(b COLLATE NOCASE)", "(b COLLATE NOCASE) || 'x'", "'x' || (b COLLATE NOCASE)",
        "b || (b COLLATE NOCASE)", "coalesce(b COLLATE NOCASE, '')", "CAST(b COLLATE NOCASE AS TEXT)",
        "lower(g)", "upper(g)", "g || ''", "g || b", "coalesce(g, '')", "ifnull(g, '')", "nullif(g, 'zz')",
        "trim(g)", "substr(g, 1, 1)", "CASE WHEN x > 2 THEN g END", "iif(x > 2, g, b)", "b", "r", "+r", "b COLLATE RTRIM",
        "CAST(r AS TEXT)", "-x", "x % 3",
    )

    def test_expression_keys_follow_the_expression_collation(self) -> None:
        """GROUP BY over expressions: CAST/unary plus/parentheses/COLLATE at any depth
        keep the collation, functions and operators without an explicit COLLATE
        yield BINARY. Compared against the original SELECT on the same database,
        after seeded random DML including rollbacks; spellings shown are members'."""
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, b TEXT, r TEXT COLLATE RTRIM, x INTEGER);"
        spell = ["a", "A", "b", "B", "a ", "A ", "ab", "\u00e9", "\u00c9", None]
        for expr in self.EXPRESSION_KEYS:
            with self.subTest(expression=expr):
                pair = _Pair(schema)
                try:
                    sel = f"SELECT {expr} AS k, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY {expr}"
                    pair.register("v", sel)
                    rng = random.Random(zlib.crc32(expr.encode()))
                    nid = 0
                    for step in range(40):
                        ids = [r[0] for r in pair.mv.execute("SELECT id FROM t")]
                        k = rng.random()
                        if k < 0.5 or not ids:
                            nid += 1
                            stmt = ("INSERT INTO t VALUES (?, ?, ?, ?, ?)", (nid, rng.choice(spell), rng.choice(spell), rng.choice(spell), rng.randint(0, 5)))
                        elif k < 0.7:
                            stmt = ("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
                        else:
                            stmt = ("UPDATE t SET g = ?, b = ?, r = ?, x = ? WHERE id = ?",
                                    (rng.choice(spell), rng.choice(spell), rng.choice(spell), rng.randint(0, 5), rng.choice(ids)))
                        pair.run([stmt], commit=rng.random() > 0.1)
                        bad = self._oracle_mismatches(pair.mv, "v", sel)
                        self.assertEqual(bad, 0, f"step {step} {stmt}: {expr}\n  MV {pair.mv.execute('SELECT * FROM v').fetchall()}\n  LIVE {pair.mv.execute(sel).fetchall()}")
                finally:
                    pair.close()

    def test_full_refresh_fallbacks_are_announced_and_exact(self) -> None:
        """Shapes whose key collation cannot be kept exact incrementally run as a
        full refresh of the original query, announced with the lead phrase."""
        schema = (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, h TEXT COLLATE NOCASE, x INTEGER);"
            "CREATE TABLE u (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, x INTEGER);"
        )
        views = {
            "setop": "SELECT g FROM t UNION SELECT g FROM u",
            "setop_ex": "SELECT g FROM t EXCEPT SELECT g FROM u",
        }
        pair = _Pair(schema)
        try:
            for name, sel in views.items():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pair.mgr.register_view(f"CREATE VIEW {name} AS {sel};")
                msgs = [str(w.message) for w in caught if "COLLATION_KEY_FALLBACK" in str(w.message)]
                self.assertEqual(len(msgs), 1, f"{name}: {msgs}")
                self.assertIn("exact result required: falling back to strategy full refresh of the original query.", msgs[0])
                self.assertIn("To change it:", msgs[0])
            rng = random.Random(11)
            spell = ["a", "A", "b", "B", "ab", None]
            nid = 0
            for step in range(60):
                ids = [r[0] for r in pair.mv.execute("SELECT id FROM t")]
                k = rng.random()
                if k < 0.5 or not ids:
                    nid += 1
                    stmt = ("INSERT INTO t VALUES (?, ?, ?, ?)", (nid, rng.choice(spell), rng.choice(spell), rng.randint(0, 3)))
                elif k < 0.6:
                    stmt = ("INSERT INTO u VALUES (?, ?, ?)", (10_000 + step, rng.choice(spell), rng.randint(0, 3)))
                elif k < 0.8:
                    stmt = ("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
                else:
                    stmt = ("UPDATE t SET g = ?, x = ? WHERE id = ?", (rng.choice(spell), rng.randint(0, 3), rng.choice(ids)))
                pair.run([stmt], commit=rng.random() > 0.1)
                for name, sel in views.items():
                    self.assertEqual(self._oracle_mismatches(pair.mv, name, sel), 0, f"step {step} {stmt}: {name}")
        finally:
            pair.close()

    def test_distinct_aggregates_follow_the_collation(self) -> None:
        """COUNT(DISTINCT v) counts values distinct under v's collation, per group
        under the group key's collation (NOT NULL key columns take the aux-store path
        that keys groups by their columns instead of the canonical key text)."""
        schema = (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT NOT NULL COLLATE NOCASE, v TEXT COLLATE NOCASE, "
            "w TEXT COLLATE RTRIM, x INTEGER);"
        )
        views = {
            "d1": "SELECT g, COUNT(DISTINCT x) AS c FROM t GROUP BY g",
            "d2": "SELECT g, COUNT(DISTINCT v) AS c, COUNT(DISTINCT w) AS cw FROM t GROUP BY g",
            "d3": "SELECT x % 2 AS p, COUNT(DISTINCT v) AS c, SUM(DISTINCT x) AS s FROM t GROUP BY x % 2",
            "d4": "SELECT COUNT(DISTINCT lower(v)) AS c, COUNT(DISTINCT v) AS c2 FROM t",
        }
        pair = _Pair(schema)
        try:
            for name, sel in views.items():
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pair.mgr.register_view(f"CREATE VIEW {name} AS {sel};")
                self.assertEqual([w for w in caught if "COLLATION_KEY_FALLBACK" in str(w.message)], [], name)
            rng = random.Random(23)
            spell = ["a", "A", "b", "B", "a ", "ab", None]
            nid = 0
            for step in range(80):
                ids = [r[0] for r in pair.mv.execute("SELECT id FROM t")]
                k = rng.random()
                if k < 0.5 or not ids:
                    nid += 1
                    stmt = ("INSERT INTO t VALUES (?, ?, ?, ?, ?)", (nid, rng.choice(spell[:-1]), rng.choice(spell), rng.choice(spell), rng.randint(0, 3)))
                elif k < 0.7:
                    stmt = ("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
                else:
                    stmt = ("UPDATE t SET g = ?, v = ?, w = ?, x = ? WHERE id = ?",
                            (rng.choice(spell[:-1]), rng.choice(spell), rng.choice(spell), rng.randint(0, 3), rng.choice(ids)))
                pair.run([stmt], commit=rng.random() > 0.1)
                for name, sel in views.items():
                    self.assertEqual(self._oracle_mismatches(pair.mv, name, sel), 0, f"step {step} {stmt}: {name}")
        finally:
            pair.close()

    def test_unreproducible_collation_falls_back_to_a_full_refresh(self) -> None:
        """A user-registered collation (here: Unicode case folding, wider than NOCASE)
        cannot be reproduced in a key text; the view is maintained by a full refresh of
        the original query (announced) and stays exact."""

        def fold(a: str, b: str) -> int:
            a, b = a.casefold(), b.casefold()
            return (a > b) - (a < b)

        con = sqlite3.connect(":memory:")
        try:
            con.create_collation("UFOLD", fold)
            MaterializedViewManager.configure_sqlite_connection(con)
            con.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE UFOLD, x INTEGER);")
            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            sel = "SELECT g, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY g"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mgr.register_view(f"CREATE VIEW v AS {sel};")
            msgs = [str(w.message) for w in caught if "COLLATION_KEY_FALLBACK" in str(w.message)]
            self.assertEqual(len(msgs), 1, msgs)
            self.assertIn("UFOLD", msgs[0])
            for i, sp in enumerate(["\u00e9", "\u00c9", "e", "E", "ss", "\u00df"], start=1):
                con.execute("INSERT INTO t VALUES (?, ?, ?)", (i, sp, i))
                con.commit()
                self.assertEqual(self._oracle_mismatches(con, "v", sel), 0, f"insert {sp!r}")
            con.execute("DELETE FROM t WHERE id = 1")
            con.commit()
            self.assertEqual(self._oracle_mismatches(con, "v", sel), 0)
        finally:
            con.close()


class TestPlanOffline(unittest.TestCase):
    """The collation-derived maintenance decision, per engine, without a database."""

    SCHEMAS = {
        "sqlite": (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, b TEXT, n INTEGER);"
            "CREATE TABLE u (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, n INTEGER);"
        ),
        "mysql:8.4": (
            "CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(20) COLLATE utf8mb4_0900_ai_ci, b VARCHAR(20) COLLATE utf8mb4_0900_bin, n INT);"
            "CREATE TABLE u (id INT PRIMARY KEY, g VARCHAR(20) COLLATE utf8mb4_0900_ai_ci, n INT);"
        ),
        "mariadb:11.8": (
            "CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(20) COLLATE utf8mb4_uca1400_ai_ci, b VARCHAR(20) COLLATE utf8mb4_uca1400_ai_cs, n INT);"
            "CREATE TABLE u (id INT PRIMARY KEY, g VARCHAR(20) COLLATE utf8mb4_uca1400_ai_ci, n INT);"
        ),
    }

    @staticmethod
    def _plan(engine: str, schema: str, view: str) -> Any:
        from matview_sync import generate_materialized_view

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return generate_materialized_view(schema, f"CREATE VIEW v AS {view};", dialect=engine, ignore_limitations=True)

    def test_incremental_where_the_key_is_derivable(self) -> None:
        incremental = (
            "SELECT g, COUNT(*) AS c FROM t GROUP BY g",
            "SELECT LOWER(g) AS k, COUNT(*) AS c FROM t GROUP BY LOWER(g)",
            "SELECT CONCAT(g, 'x') AS k, COUNT(*) AS c FROM t GROUP BY CONCAT(g, 'x')",
            "SELECT n, COUNT(*) AS c FROM t GROUP BY n",
            "SELECT n, COUNT(DISTINCT g) AS c FROM t GROUP BY n",
            "SELECT g, COUNT(DISTINCT n) AS c FROM t GROUP BY g",
            "SELECT g FROM t",
            "SELECT DISTINCT g FROM t",
            "SELECT DISTINCT g, n FROM t",
            "SELECT t.g AS g, COUNT(*) AS c FROM t JOIN u ON t.n = u.n GROUP BY t.g",
        )
        for engine, schema in self.SCHEMAS.items():
            for view in incremental:
                if engine == "sqlite" and "CONCAT" in view:
                    continue
                with self.subTest(engine=engine, view=view):
                    plan = self._plan(engine, schema, view)
                    self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
                    self.assertEqual([d.code for d in plan.strategy_diagnostics], [])

    def test_full_refresh_where_the_key_is_not_derivable(self) -> None:
        fallbacks = (
            "SELECT g FROM t UNION SELECT g FROM u",
            "SELECT g FROM t EXCEPT SELECT g FROM u",
        )
        for engine, schema in self.SCHEMAS.items():
            for view in fallbacks:
                with self.subTest(engine=engine, view=view):
                    plan = self._plan(engine, schema, view)
                    self.assertNotEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
                    self.assertEqual([d.code for d in plan.strategy_diagnostics], ["COLLATION_KEY_FALLBACK"])
                    self.assertTrue(plan.strategy_diagnostics[0].message.startswith(
                        "exact result required: falling back to strategy full refresh of the original query."))
        # MariaDB's multi-level collation: weights are not an equality key.
        plan = self._plan("mariadb:11.8", self.SCHEMAS["mariadb:11.8"], "SELECT b, COUNT(*) AS c FROM t GROUP BY b")
        self.assertEqual([d.code for d in plan.strategy_diagnostics], ["COLLATION_KEY_FALLBACK"])
        plan = self._plan("mysql:8.4", self.SCHEMAS["mysql:8.4"], "SELECT b, COUNT(*) AS c FROM t GROUP BY b")
        self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)  # NO PAD binary: byte-exact


    BACKFILL_VIEWS = (
        "SELECT g, COUNT(DISTINCT n) AS c, SUM(n) AS s FROM t GROUP BY g",
        "SELECT LOWER(g) AS k, COUNT(DISTINCT n) AS c FROM t GROUP BY LOWER(g)",
        "SELECT CONCAT(g, 'x') AS k, COUNT(DISTINCT n) AS c FROM t GROUP BY CONCAT(g, 'x')",
        "SELECT g COLLATE utf8mb4_general_ci AS k, COUNT(DISTINCT n) AS c FROM t GROUP BY g COLLATE utf8mb4_general_ci",
        "SELECT g, n, COUNT(DISTINCT id) AS c FROM t GROUP BY g, n",
        "SELECT g, COUNT(*) AS c FROM t GROUP BY g",
    )

    def test_group_key_batch_templates_render_through_the_backfill_path(self) -> None:
        """Every chunk template of the incremental backfill (group-key batches carry
        the group key text, which contains braces, quotes and regular expressions) takes
        its placeholders by replacing the reserved token, never by str.format."""
        from matview.logical_ir import GROUP_KEY_PLACEHOLDERS

        seen = 0
        for engine, schema in self.SCHEMAS.items():
            for view in self.BACKFILL_VIEWS:
                if engine == "sqlite" and ("CONCAT" in view or "utf8mb4" in view):
                    continue
                with self.subTest(engine=engine, view=view):
                    spec = self._plan(engine, schema, view).incremental_backfill_spec
                    self.assertIsNotNone(spec)
                    for tmpl in spec.chunk_sql_templates:  # type: ignore[union-attr]
                        if spec.mode != "group_key_batch":  # type: ignore[union-attr]
                            self.assertNotIn(GROUP_KEY_PLACEHOLDERS, tmpl)
                            continue
                        seen += 1
                        rendered = tmpl.replace(GROUP_KEY_PLACEHOLDERS, "?, ?, ?")
                        self.assertNotIn(GROUP_KEY_PLACEHOLDERS, rendered)
                        self.assertNotIn("{placeholders}", tmpl)
                        if GROUP_KEY_PLACEHOLDERS in tmpl:
                            self.assertEqual(rendered.count("?, ?, ?"), tmpl.count(GROUP_KEY_PLACEHOLDERS))
        self.assertGreater(seen, 20)
        # Group key text carries the grouped expression verbatim, braces included:
        # the reason str.format cannot be used on generated SQL.
        view = "SELECT CONCAT(g, '{0}') AS k, COUNT(DISTINCT n) AS c FROM t GROUP BY CONCAT(g, '{0}')"
        spec = self._plan("mariadb:11.8", self.SCHEMAS["mariadb:11.8"], view).incremental_backfill_spec
        self.assertEqual(spec.mode, "group_key_batch")  # type: ignore[union-attr]
        braced = [t for t in spec.chunk_sql_templates if "'{0}'" in t and GROUP_KEY_PLACEHOLDERS in t]  # type: ignore[union-attr]
        self.assertTrue(braced)
        for tmpl in braced:
            self.assertIn("'{0}'", tmpl.replace(GROUP_KEY_PLACEHOLDERS, "?, ?, ?"))
            with self.assertRaises((IndexError, KeyError, ValueError)):
                tmpl.format(placeholders="?, ?, ?")

    def test_reserved_backfill_token_in_a_view_is_refused(self) -> None:
        from matview.logical_ir import GROUP_KEY_PLACEHOLDERS

        view = f"SELECT g, COUNT(DISTINCT n) AS c FROM t WHERE '{GROUP_KEY_PLACEHOLDERS}' <> 'x' GROUP BY g"
        with self.assertRaises(Exception) as ctx:
            self._plan("mysql:8.4", self.SCHEMAS["mysql:8.4"], view)
        self.assertIn("reserved backfill token", str(ctx.exception))

    def test_mysql_pad_space_uca_collations_are_refused_as_plan_dependent(self) -> None:
        pad_schema = (
            "CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(20) COLLATE {c}, n INT);"
            "CREATE TABLE u (id INT PRIMARY KEY, g VARCHAR(20) COLLATE {c}, n INT);"
        )
        views = (
            "SELECT g, COUNT(*) AS c FROM t GROUP BY g",
            "SELECT DISTINCT g FROM t",
            "SELECT g, COUNT(DISTINCT n) AS c FROM t GROUP BY g",
            "SELECT g, MIN(n) AS lo, MAX(n) AS hi FROM t GROUP BY g",
        )
        for coll in ("utf8mb4_unicode_ci", "utf8mb4_unicode_520_ci", "utf8mb4_danish_ci", "utf8mb3_unicode_ci", "cp1250_general_ci"):
            for view in views:
                with self.subTest(collation=coll, view=view):
                    plan = self._plan("mysql:8.4", pad_schema.format(c=coll), view)
                    self.assertNotEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
                    self.assertEqual([d.code for d in plan.strategy_diagnostics], ["COLLATION_KEY_FALLBACK"])
                    msg = plan.strategy_diagnostics[0].message
                    self.assertTrue(msg.startswith("exact result required: falling back to strategy full refresh"))
                    self.assertIn("plan-dependently", msg)
        for coll in ("utf8mb4_0900_ai_ci", "utf8mb4_0900_as_cs", "utf8mb4_0900_bin", "utf8mb4_bin", "utf8mb4_general_ci", "latin1_swedish_ci"):
            for view in views[:3]:
                with self.subTest(collation=coll, view=view):
                    plan = self._plan("mysql:8.4", pad_schema.format(c=coll), view)
                    self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA, coll)
                    self.assertEqual([d.code for d in plan.strategy_diagnostics], [])
        # MariaDB pads by weight in every plan: not refused.
        plan = self._plan("mariadb:11.8", pad_schema.format(c="utf8mb4_unicode_ci"), views[0])
        self.assertEqual([d.code for d in plan.strategy_diagnostics], [])

    def test_mysql_plan_dependence_rule_table(self) -> None:
        for name, expected in (
            ("utf8mb4_unicode_ci", True), ("utf8mb4_unicode_520_ci", True), ("utf8mb4_danish_ci", True),
            ("UTF8MB4_UNICODE_CI", True), ("utf8mb3_unicode_ci", True), ("ucs2_unicode_ci", True),
            ("utf16_unicode_520_ci", True), ("utf32_swedish_ci", True), ("cp1250_general_ci", True),
            ("utf8mb4_general_ci", False), ("utf8mb4_bin", False), ("utf8mb3_general_ci", False),
            ("latin1_swedish_ci", False), ("latin1_general_cs", False), ("ascii_general_ci", False),
            ("utf8mb4_0900_ai_ci", False), ("utf8mb4_ja_0900_as_cs", False), ("utf8mb4_0900_bin", False),
            ("binary", False), (None, False), ("", False),
        ):
            self.assertEqual(key_collation.mysql_collation_is_plan_dependent(name), expected, name)


def _mysql_dsn() -> Optional[str]:
    return os.environ.get("MYSQL_DSN") or None


# Spellings tried per collation: case, accents, trailing spaces, no-break and
# ideographic spaces, zero-width space, expansions (sharp s / ss, ae-ligature / ae,
# fi-ligature / fi), combining marks, supplementary-plane characters, empty and
# blank strings.
_NATIVE_SPELLINGS: Tuple[str, ...] = (
    "a", "A", "á", "a ", "a  ", "a ", "a　", "a​", "ß", "ss", "æ", "ae",
    "ﬁ", "fi", "\U0001f600", "\U0001f600 ", "á", "", " ", "b", "B ",
)

# name -> (select, text key columns as (expression over the member source, output column),
# member source). The member source holds the rows whose spellings a displayed key may show.
_T = "t AS m"
_NATIVE_SHAPES: Dict[str, Tuple[str, Tuple[Tuple[str, str], ...], str]] = {
    "plain": ("SELECT g, COUNT(*) AS c, SUM(x) AS s, MIN(x) AS mn FROM t GROUP BY g", (("g", "g"),), _T),
    "multi": ("SELECT g, h, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY g, h", (("g", "g"), ("h", "h")), _T),
    "lower": ("SELECT LOWER(g) AS k, COUNT(*) AS c FROM t GROUP BY LOWER(g)", (("LOWER(g)", "k"),), _T),
    "concat": ("SELECT CONCAT(g, 'x') AS k, COUNT(*) AS c FROM t GROUP BY CONCAT(g, 'x')", (("CONCAT(g, 'x')", "k"),), _T),
    "trim": ("SELECT TRIM(g) AS k, COUNT(*) AS c FROM t GROUP BY TRIM(g)", (("TRIM(g)", "k"),), _T),
    "coalesce": ("SELECT COALESCE(g, '') AS k, COUNT(*) AS c FROM t GROUP BY COALESCE(g, '')", (("COALESCE(g, '')", "k"),), _T),
    "cast": ("SELECT CAST(g AS CHAR) AS k, COUNT(*) AS c FROM t GROUP BY CAST(g AS CHAR)", (("CAST(g AS CHAR)", "k"),), _T),
    "substr": ("SELECT SUBSTRING(g, 1, 1) AS k, COUNT(*) AS c FROM t GROUP BY SUBSTRING(g, 1, 1)", (("SUBSTRING(g, 1, 1)", "k"),), _T),
    "explicit": (
        "SELECT g COLLATE utf8mb4_general_ci AS k, COUNT(*) AS c FROM t GROUP BY g COLLATE utf8mb4_general_ci",
        (("g COLLATE utf8mb4_general_ci", "k"),), _T,
    ),
    "distinct": ("SELECT DISTINCT g FROM t", (("g", "g"),), _T),
    "union": ("SELECT g FROM t UNION SELECT g FROM u", (("g", "g"),), "(SELECT g FROM t UNION ALL SELECT g FROM u) AS m"),
    "join1": ("SELECT t.g AS g, COUNT(*) AS c FROM t JOIN u ON t.x = u.id GROUP BY t.g", (("g", "g"),), _T),
    "uall": (
        "SELECT g, COUNT(*) AS c FROM (SELECT g FROM t UNION ALL SELECT g FROM u) AS x GROUP BY g", (("g", "g"),),
        "(SELECT g FROM t UNION ALL SELECT g FROM u) AS m",
    ),
    "countd": ("SELECT MOD(x, 2) AS p, COUNT(DISTINCT g) AS c FROM t GROUP BY MOD(x, 2)", (), _T),
    "join": ("SELECT t.g AS g, COUNT(*) AS c FROM t JOIN u ON t.x = u.x GROUP BY t.g", (("g", "g"),), _T),
}
_NATIVE_SHORT_SPELLINGS: Tuple[str, ...] = ("a", "A", "á", "a ", "a　", "a\u00a0", "", "b", "B ", "\U0001f600")
_NATIVE_LIGHT = ("plain", "distinct")
_NATIVE_MEDIUM = ("plain", "distinct", "countd")


@unittest.skipUnless(_mysql_dsn(), "MYSQL_DSN not set (native MySQL/MariaDB only)")
class TestMysqlGroupKeyCollation(unittest.TestCase):
    """Native MySQL/MariaDB: groups merge exactly where the server's collation
    says two values are equal (case/accent-insensitive, PAD SPACE and NO PAD,
    NBSP/ideographic space/zero-width/expansion spellings), for plain and
    expression keys and the shapes that fall back to a full refresh.

    The oracle is the server itself: the original SELECT runs on the same base
    tables. Every materialized row must (a) have the aggregates the server computes
    for the rows whose key compares equal to the row's key, (b) show a spelling some
    member row currently has, and (c) not duplicate another row's group."""

    # (collation, shapes, shapes announced as COLLATION_KEY_FALLBACK).
    # utf8mb4_unicode_ci is a PAD SPACE UCA collation: MySQL groups it plan-dependently
    # (see test_premise_*), so every text key falls back to a full refresh there. UNION
    # over a spelling-sensitive key stays a full refresh (no spelling refresh exists for
    # set operations).
    _ALL = tuple(_NATIVE_SHAPES)
    MYSQL = (
        ("utf8mb4_0900_ai_ci", _ALL, {"union"}),
        ("utf8mb4_unicode_ci", ("plain", "lower", "distinct", "countd", "explicit"), {"plain", "lower", "distinct", "countd"}),
        ("utf8mb4_general_ci", ("plain", "distinct", "countd"), set()),
        ("utf8mb4_0900_as_cs", _NATIVE_LIGHT, set()),
        ("utf8mb4_bin", _NATIVE_LIGHT, set()),
        ("utf8mb4_0900_bin", _NATIVE_LIGHT, set()),
        (None, _NATIVE_LIGHT, None),  # the server's default collation, none declared: see _run_collation
    )
    MARIADB = (
        ("utf8mb4_uca1400_ai_ci", _ALL, {"union"}),
        ("utf8mb4_unicode_ci", _NATIVE_MEDIUM, set()),
        ("utf8mb4_general_ci", _NATIVE_LIGHT, set()),
        ("utf8mb4_bin", _NATIVE_LIGHT, set()),
        ("utf8mb4_nopad_bin", _NATIVE_LIGHT, set()),
        ("utf8mb4_uca1400_nopad_ai_ci", _NATIVE_LIGHT, set()),
        ("utf8mb4_uca1400_ai_cs", ("plain", "lower"), {"plain", "lower"}),  # multi-level: full refresh
        (None, _NATIVE_LIGHT, set()),
    )

    @classmethod
    def setUpClass(cls) -> None:
        import pymysql

        cls._pymysql = pymysql
        parsed = urlparse(_mysql_dsn() or "")
        cls._conn_args = dict(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") or None,
            charset="utf8mb4",
        )
        probe = cls._connect()
        try:
            cur = probe.cursor()
            cur.execute("SELECT VERSION()")
            cls.is_mariadb = "mariadb" in str(cur.fetchone()[0]).lower()
        finally:
            probe.close()

    @classmethod
    def _connect(cls) -> Any:
        return cls._pymysql.connect(autocommit=False, **cls._conn_args)

    def _check(self, cur: Any, tag: str, shapes: Tuple[str, ...]) -> None:
        for name in shapes:
            sel, text_keys, member_src = _NATIVE_SHAPES[name]
            cur.execute(f"SELECT * FROM gk_v_{name}")
            cols = [d[0] for d in cur.description]
            got = cur.fetchall()
            # Compare text keys under the collation the server derives for the key
            # expression (a full-refresh view's output column carries the table default,
            # which UNION would refuse to mix with the live column's collation).
            coll_of: Dict[str, str] = {}
            for expr, col in text_keys:
                probe = (
                    "(SELECT g, h, x FROM t UNION ALL SELECT NULL, NULL, NULL) AS m"
                    if member_src == _T
                    else member_src.replace(") AS m", " UNION ALL SELECT NULL) AS m")
                )
                cur.execute(f"SELECT COLLATION(({expr})) FROM {probe} LIMIT 1")
                coll_of[col] = cur.fetchone()[0]
            proj = ", ".join(f"`{c}` COLLATE {coll_of[c]} AS `{c}`" if c in coll_of else f"`{c}`" for c in cols)
            col_sql = ", ".join(f"`{c}`" for c in cols)
            # Every group of the live view (as the server's own GROUP BY sees equal
            # keys) matches exactly one row of the materialized view with the same
            # aggregates, and the other way round: group the union of both sides by
            # all columns; each bucket must hold one live and one materialized row.
            cur.execute(
                f"SELECT COUNT(*) FROM (SELECT {col_sql}, SUM(src = 1) AS a, SUM(src = 2) AS b FROM ("
                f"SELECT {proj}, 1 AS src FROM gk_v_{name} UNION ALL SELECT {proj}, 2 AS src FROM ({sel}) AS l"
                f") AS z GROUP BY {col_sql}) AS w WHERE a <> 1 OR b <> 1"
            )
            bad = cur.fetchone()[0]
            if bad:
                cur.execute(sel)
                self.fail(f"{tag}: {name} differs from the live view in {bad} group(s)\n  MV {got}\n  LIVE {cur.fetchall()}")
            # The displayed key is a spelling some member row currently has (all key
            # columns of a row from the same member).
            if text_keys:
                member = " AND ".join(
                    f"CAST(({expr}) AS BINARY) <=> CAST(v.`{col}` AS BINARY)" for expr, col in text_keys
                )
                cur.execute(
                    f"SELECT COUNT(*) FROM gk_v_{name} AS v WHERE NOT EXISTS (SELECT 1 FROM {member_src} WHERE {member})"
                )
                self.assertEqual(cur.fetchone()[0], 0, f"{tag}: {name} shows a spelling no member row has: {got}")

    def _run_collation(self, coll: Optional[str], shapes: Tuple[str, ...], fallbacks: Optional[set]) -> None:
        spellings = _NATIVE_SPELLINGS if len(shapes) > 4 else _NATIVE_SHORT_SPELLINGS
        conn = self._connect()
        cur = conn.cursor()
        decl = f"CHARACTER SET utf8mb4 COLLATE {coll}" if coll else ""
        try:
            for stale in ("t", "u"):
                cur.execute(f"DROP TABLE IF EXISTS {stale}")
            cur.execute(f"CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(20) {decl}, h VARCHAR(20) {decl}, x INT) ENGINE=InnoDB")
            cur.execute(f"CREATE TABLE u (id INT PRIMARY KEY, g VARCHAR(20) {decl}, x INT) ENGINE=InnoDB")
            conn.commit()
            mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for name in shapes:
                    mgr.register_view(f"CREATE VIEW gk_v_{name} AS {_NATIVE_SHAPES[name][0]};")
            diag = {
                w.message.diagnostic.view for w in caught
                if isinstance(w.message, MatviewDiagnosticWarning) and w.message.diagnostic.code == "COLLATION_KEY_FALLBACK"
            }
            tag = coll or "server default"
            if fallbacks is None:  # server default collation: the rule is proven by the premise test
                cur.execute("SELECT @@collation_database")
                dflt = cur.fetchone()[0]
                plan_dependent = (not self.is_mariadb) and key_collation.mysql_collation_is_plan_dependent(dflt)
                fallbacks = set(shapes) if plan_dependent else set()
            self.assertEqual(diag, {f"gk_v_{n}" for n in fallbacks}, f"{tag}: shapes announced as a full refresh")

            self._check(cur, f"{tag} empty", shapes)
            n = len(spellings)
            conn.begin()
            for i, sp in enumerate(spellings, start=1):
                cur.execute("INSERT INTO t VALUES (%s, %s, %s, %s)", (i, sp, spellings[i % n], i % 5))
            for i, sp in enumerate(spellings[:6], start=1):
                cur.execute("INSERT INTO u VALUES (%s, %s, %s)", (i, sp, i % 5))
            conn.commit()
            self._check(cur, f"{tag} bulk insert", shapes)
            cur.execute("UPDATE t SET g = %s WHERE id = 1", (spellings[1],))
            cur.execute("UPDATE t SET g = %s, x = x + 1 WHERE id = 4", (spellings[0],))
            conn.commit()
            self._check(cur, f"{tag} spelling updates", shapes)
            rng = random.Random(7)
            ids = [r for r in range(1, n + 1)]
            rng.shuffle(ids)
            for k, rid in enumerate(ids, start=1):  # delete one row at a time: the shown spelling must follow
                cur.execute("DELETE FROM t WHERE id = %s", (rid,))
                if k % 5 == 0:
                    cur.execute("DELETE FROM u WHERE id = %s", (k // 5,))
                conn.commit()
                if k % 4 == 0 or k == n:
                    self._check(cur, f"{tag} after {k} deletes", shapes)
            cur.execute("INSERT INTO t VALUES (200, %s, %s, 1)", (spellings[0], spellings[1]))
            cur.execute("DELETE FROM u")
            conn.rollback()
            self._check(cur, f"{tag} rolled back", shapes)
        finally:
            conn.rollback()
            for name in reversed(shapes):
                try:
                    mgr.drop_view(f"gk_v_{name}")
                except Exception:
                    pass
            for tbl in ("t", "u"):
                cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            conn.commit()
            conn.close()

    # --- the premise of the MySQL PAD SPACE refusal, measured on the server ----

    # Characters that weigh like U+0020 under some collation (NBSP, the Unicode spaces)
    # and ones that weigh nothing (soft hyphen, zero-width space).
    _SPACE_LIKE = (0xA0, 0x1680, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000, 0x200B, 0xAD)
    _UNICODE_CHARSETS = ("utf8mb4", "utf8mb3", "utf16", "utf32", "ucs2")

    def _divergent_pad_chars(self, cur: Any, charset: str, coll: str) -> Tuple[List[int], List[int]]:
        """``(equal, divergent)``: the characters c for which ``'k' = 'k' || c`` holds,
        and those among them that GROUP BY / DISTINCT do not merge in every plan (no
        index, KEY(g), COUNT(DISTINCT), DISTINCT with and without the index)."""
        cps = list(self._SPACE_LIKE) if charset in self._UNICODE_CHARSETS else [0xA0]
        cur.execute("DROP TABLE IF EXISTS gk_pad")
        cur.execute(
            f"CREATE TABLE gk_pad (id INT AUTO_INCREMENT PRIMARY KEY, k INT, x INT, "
            f"g VARCHAR(16) CHARACTER SET {charset} COLLATE {coll})"
        )
        rows: List[Tuple[int, str, int]] = []
        for k, cp in enumerate(cps):
            rows.append((k, f"k{k:03d}", k))
            rows.append((k, f"k{k:03d}" + chr(cp), k + 1000))
        cur.executemany("INSERT INTO gk_pad (k, g, x) VALUES (%s, %s, %s)", rows)

        def merged(sql: str) -> set:
            cur.execute(sql)
            counts: Dict[str, int] = {}
            for (g, *_rest) in cur.fetchall():
                counts[g[:4]] = counts.get(g[:4], 0) + 1
            return {int(key[1:]) for key, n in counts.items() if n == 1}

        plans = [
            "SELECT g FROM gk_pad GROUP BY g",
            "SELECT g, COUNT(DISTINCT x) FROM gk_pad GROUP BY g",
            "SELECT DISTINCT g FROM gk_pad",
        ]
        results = [merged(q) for q in plans]
        cur.execute("ALTER TABLE gk_pad ADD KEY kg (g)")
        results += [merged(q) for q in plans]
        cur.execute("SELECT DISTINCT a.k FROM gk_pad a JOIN gk_pad b ON a.k = b.k AND a.id < b.id WHERE a.g = b.g")
        equal = {r[0] for r in cur.fetchall()}
        divergent = set()
        for r in results:
            divergent |= r ^ equal
        cur.execute("DROP TABLE gk_pad")
        return sorted(cps[k] for k in equal), sorted(cps[k] for k in divergent)

    def _collations_of_interest(self, cur: Any, full: bool) -> List[Tuple[str, str]]:
        cur.execute(
            "SELECT COLLATION_NAME, CHARACTER_SET_NAME FROM information_schema.COLLATIONS "
            "WHERE CHARACTER_SET_NAME IN ('utf8mb4','utf8mb3','latin1','ascii','ucs2','utf16','utf32') "
            "ORDER BY CHARACTER_SET_NAME, COLLATION_NAME"
        )
        every = [(str(n).lower(), str(c).lower()) for n, c in cur.fetchall()]
        if full:
            return every
        wanted = (
            ("utf8mb4_unicode_ci", "utf8mb4_unicode_520_ci", "utf8mb4_danish_ci", "utf8mb4_general_ci", "utf8mb4_bin",
             "utf8mb4_0900_ai_ci", "utf8mb4_0900_as_cs", "utf8mb4_nopad_bin", "utf8mb4_uca1400_ai_ci",
             "utf8mb4_uca1400_nopad_ai_ci", "utf8mb3_general_ci", "latin1_swedish_ci", "ucs2_unicode_ci")
        )
        return [(n, c) for n, c in every if n in wanted]

    def test_premise_pad_rule_matches_the_server(self) -> None:
        """The refusal rule (``mysql_collation_is_plan_dependent``) is measured, not
        assumed: for sampled collations (every collation with ``MATVIEW_FULL_COLLATION_SWEEP=1``)
        a MySQL collation is plan-dependent exactly when some character other than U+0020
        is ``=``-equal to padding but not merged by every GROUP BY / DISTINCT plan;
        MariaDB merges exactly what ``=`` equates in every plan."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            full = bool(os.environ.get("MATVIEW_FULL_COLLATION_SWEEP"))
            sampled = self._collations_of_interest(cur, full)
            self.assertGreater(len(sampled), 3)
            diverging_seen = 0
            for coll, charset in sampled:
                with self.subTest(collation=coll):
                    equal, divergent = self._divergent_pad_chars(cur, charset, coll)
                    if self.is_mariadb:
                        self.assertEqual(divergent, [], f"{coll}: GROUP BY / DISTINCT disagree with '=' in some plan")
                    else:
                        self.assertEqual(
                            bool(divergent), key_collation.mysql_collation_is_plan_dependent(coll),
                            f"{coll}: divergent characters {[hex(c) for c in divergent]}",
                        )
                        diverging_seen += bool(divergent)
            if not self.is_mariadb:
                self.assertGreater(diverging_seen, 2)
        finally:
            conn.close()

    def test_premise_unicode_ci_groups_plan_dependently(self) -> None:
        """The reproduction behind the refusal: under utf8mb4_unicode_ci the same rows
        make GROUP BY return one group with an index on the key and two without it, so
        the original view itself has no single result (MySQL only)."""
        if self.is_mariadb:
            self.skipTest("MariaDB pads by weight in every plan")
        conn = self._connect()
        try:
            cur = conn.cursor()
            sizes = []
            for key in ("", ", KEY(g)"):
                cur.execute("DROP TABLE IF EXISTS gk_plan")
                cur.execute(f"CREATE TABLE gk_plan (id INT PRIMARY KEY, g VARCHAR(8) COLLATE utf8mb4_unicode_ci, x INT{key})")
                cur.execute("INSERT INTO gk_plan VALUES (1, 'a', 1), (2, %s, 2)", ("a\u3000",))
                cur.execute("SELECT COUNT(*) FROM (SELECT g FROM gk_plan GROUP BY g) AS q")
                sizes.append(cur.fetchone()[0])
            cur.execute("DROP TABLE gk_plan")
            conn.commit()
            self.assertEqual(sizes, [2, 1])
        finally:
            conn.close()

    def test_unicode_ci_view_falls_back_and_stays_exact(self) -> None:
        """utf8mb4_unicode_ci (PAD SPACE UCA) on MySQL: the group-scoped MIN/MAX
        recompute and the grouped views fall back to a full refresh (announced) and equal
        the live view; the same rows under utf8mb4_general_ci stay incremental."""
        conn = self._connect()
        cur = conn.cursor()
        views = {
            "mm": "SELECT g, MIN(x) AS mn, MAX(x) AS mx FROM t GROUP BY g",
            "cd": "SELECT g, COUNT(DISTINCT x) AS c FROM t GROUP BY g",
        }
        try:
            for coll, expect_fallback in (("utf8mb4_unicode_ci", not self.is_mariadb), ("utf8mb4_general_ci", False)):
                with self.subTest(collation=coll):
                    for tbl in ("t", "u"):
                        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
                    cur.execute(f"CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(20) COLLATE {coll}, h VARCHAR(20), x INT)")
                    cur.execute("CREATE TABLE u (id INT PRIMARY KEY, g VARCHAR(20), x INT)")
                    conn.commit()
                    mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        for name, sel in views.items():
                            mgr.register_view(f"CREATE VIEW gk_v_{name} AS {sel};")
                    announced = [w for w in caught if "COLLATION_KEY_FALLBACK" in str(w.message)]
                    self.assertEqual(len(announced), 2 if expect_fallback else 0, [str(w.message) for w in announced])
                    for m in announced:
                        self.assertIn("exact result required: falling back to strategy full refresh", str(m.message))
                    # (1,'a' + ideographic space,3), (2,'a with acute',0), (3,'A  ',0); delete id 3
                    for i, (g, x) in enumerate([("a\u3000", 3), ("\u00e1", 0), ("A  ", 0), ("a\u00a0", 5)], start=1):
                        cur.execute("INSERT INTO t VALUES (%s, %s, NULL, %s)", (i, g, x))
                        conn.commit()
                        self._same_as_live(cur, views)
                    cur.execute("DELETE FROM t WHERE id = 3")
                    conn.commit()
                    self._same_as_live(cur, views)
                    for name in views:
                        mgr.drop_view(f"gk_v_{name}")
        finally:
            conn.rollback()
            for tbl in ("t", "u"):
                cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            conn.commit()
            conn.close()

    def _same_as_live(self, cur: Any, views: Dict[str, str]) -> None:
        """Each materialized view holds exactly the rows (and multiplicities) of its original query."""
        for name, sel in views.items():
            cur.execute(f"SELECT * FROM gk_v_{name}")
            mv_rows = sorted(map(repr, cur.fetchall()))
            cur.execute(sel)
            live = sorted(map(repr, cur.fetchall()))
            self.assertEqual(mv_rows, live, f"{name}")

    def test_incremental_backfill_of_collation_keys(self) -> None:
        """register_view(incremental_backfill=True) over a populated table: the group-key
        batch statements carry the (brace-bearing) key text and must run as generated."""
        conn = self._connect()
        cur = conn.cursor()
        shapes = ("bf_plain", "bf_lower", "bf_countd", "bf_multi")
        sels = {
            "bf_plain": "SELECT g, COUNT(*) AS c, SUM(x) AS s FROM t GROUP BY g",
            "bf_lower": "SELECT LOWER(g) AS k, COUNT(DISTINCT x) AS c FROM t GROUP BY LOWER(g)",
            "bf_countd": "SELECT g, COUNT(DISTINCT x) AS c, SUM(x) AS s FROM t GROUP BY g",
            "bf_multi": "SELECT g, h, COUNT(DISTINCT x) AS c FROM t GROUP BY g, h",
        }
        try:
            for coll in ("utf8mb4_unicode_ci",) if self.is_mariadb else ("utf8mb4_0900_ai_ci",):
                with self.subTest(collation=coll or "server default"):
                    cur.execute("DROP TABLE IF EXISTS t")
                    decl = f"COLLATE {coll}" if coll else ""
                    cur.execute(f"CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(32) {decl}, h VARCHAR(32) {decl}, x INT)")
                    cur.executemany(
                        "INSERT INTO t VALUES (%s, %s, %s, %s)",
                        [(i, _NATIVE_SPELLINGS[i % len(_NATIVE_SPELLINGS)], _NATIVE_SPELLINGS[(i * 7) % len(_NATIVE_SPELLINGS)], i % 5)
                         for i in range(1, 80)],
                    )
                    conn.commit()
                    mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for name in shapes:
                            mgr.register_view(f"CREATE VIEW gk_v_{name} AS {sels[name]};", incremental_backfill=True)
                    try:
                        self._same_as_live_keys(cur, {n: sels[n] for n in shapes})
                        cur.execute("DELETE FROM t WHERE id % 3 = 0")
                        conn.commit()
                        self._same_as_live_keys(cur, {n: sels[n] for n in shapes})
                    finally:
                        for name in shapes:
                            mgr.drop_view(f"gk_v_{name}")
        finally:
            conn.rollback()
            cur.execute("DROP TABLE IF EXISTS t")
            conn.commit()
            conn.close()

    def _same_as_live_keys(self, cur: Any, views: Dict[str, str]) -> None:
        """Group-count equality with the live view under the server's own comparison:
        the union of both sides grouped by all columns leaves one row of each side per bucket."""
        for name, sel in views.items():
            cur.execute(f"SELECT * FROM gk_v_{name}")
            cols = [d[0] for d in cur.description]
            col_sql = ", ".join(f"`{c}`" for c in cols)
            cur.execute(
                f"SELECT COUNT(*) FROM (SELECT {col_sql}, SUM(src = 1) AS a, SUM(src = 2) AS b FROM ("
                f"SELECT {col_sql}, 1 AS src FROM gk_v_{name} UNION ALL SELECT {col_sql}, 2 AS src FROM ({sel}) AS l"
                f") AS z GROUP BY {col_sql}) AS w WHERE a <> 1 OR b <> 1"
            )
            self.assertEqual(cur.fetchone()[0], 0, f"{name} differs from the live view")

    def test_native_differential_detects_injected_faults(self) -> None:
        """Mutants: (1) no spelling refresh (a stale displayed spelling), (2) a
        collation-blind key (groups split). The native differential must fail for both."""
        from matview.dialects.base import DialectBaseMixin
        from matview.maintenance.single import Tier1SingleMixin

        coll = "utf8mb4_unicode_ci" if self.is_mariadb else "utf8mb4_0900_ai_ci"
        shapes = ("plain", "distinct", "countd")
        spelling = Tier1SingleMixin._key_spelling_refresh_stmts
        Tier1SingleMixin._key_spelling_refresh_stmts = lambda self, mv_table, record_prefix="OLD": []  # type: ignore[method-assign]
        try:
            with self.assertRaises(AssertionError) as stale:
                self._run_collation(coll, shapes, set())
        finally:
            Tier1SingleMixin._key_spelling_refresh_stmts = spelling  # type: ignore[method-assign]
        self.assertIn("shows a spelling no member row has", str(stale.exception))
        canon = DialectBaseMixin._canonical_group_pk_expr
        DialectBaseMixin._canonical_group_pk_expr = (  # type: ignore[method-assign]
            lambda self, group_exprs, *, collation_aware=True: canon(self, group_exprs, collation_aware=False)
        )
        try:
            with self.assertRaises(AssertionError) as split:
                self._run_collation(coll, shapes, set())
        finally:
            DialectBaseMixin._canonical_group_pk_expr = canon  # type: ignore[method-assign]
        self.assertIn("differs from the live view", str(split.exception))

    def test_group_merge_follows_the_expression_collation(self) -> None:
        cur = self._connect().cursor()
        cur.execute("SELECT LOWER(COLLATION_NAME) FROM information_schema.COLLATIONS")
        available = {r[0] for r in cur.fetchall()}
        cur.connection.close()
        for coll, shapes, fallbacks in self.MARIADB if self.is_mariadb else self.MYSQL:
            if coll is not None and coll not in available:  # e.g. uca1400 collations are MariaDB 11.3+
                continue
            with self.subTest(collation=coll or "server default"):
                self._run_collation(coll, shapes, fallbacks)


if __name__ == "__main__":
    unittest.main()
