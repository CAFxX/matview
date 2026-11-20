#!/usr/bin/env python3
"""SQLite SUM/TOTAL/AVG maintenance must equal the original view exactly.

Contract: the materialized reader returns exactly what the original view
returns over the current base rows -- same value, same storage class (INTEGER
vs REAL), same sign of zero. There is no numeric tolerance anywhere in this
file: every comparison is exact value + exact Python type.

SQLite accumulates integer inputs exactly and switches to binary floating point
as soon as one input is not an INTEGER; every SQLite column can store REAL
values regardless of its declared type. An add/retract fold over such values
depends on values that were added and later removed (1e16 + 1.5 - 1e16 == 2.0),
so it is not equivalent to the native aggregate. Regression origin: CI fuzz
seed 1586782166 scenario 3 (BIGINT column holding REAL 0.6 after an UPDATE;
SUM 7.000000000000007 vs 7 after deleting the REAL rows).
"""

from __future__ import annotations

import math
import os
import random
import sqlite3
import tempfile
import unittest
from typing import Any, List, Optional, Sequence, Tuple

from matview import install_materialized_view_sqlite


def _same_value(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


def _same_rows(mv: Sequence[Sequence[Any]], live: Sequence[Sequence[Any]]) -> bool:
    if len(mv) != len(live):
        return False
    a = sorted((tuple(r) for r in mv), key=repr)
    b = sorted((tuple(r) for r in live), key=repr)
    return all(len(x) == len(y) and all(_same_value(p, q) for p, q in zip(x, y)) for x, y in zip(a, b))


def _install(conn: sqlite3.Connection, name: str, select_sql: str, schema: str, **kw: Any) -> None:
    # The manager refuses to commit caller-owned work: commit the fixture rows first.
    conn.commit()
    install_materialized_view_sqlite(conn, f"CREATE VIEW {name} AS {select_sql};", schema_sql=schema, **kw)


class ExactAssertions(unittest.TestCase):
    def assertViewExact(self, conn: sqlite3.Connection, name: str, select_sql: str, context: str = "") -> None:
        mv = conn.execute(f"SELECT * FROM {name}").fetchall()
        live = conn.execute(select_sql).fetchall()
        self.assertTrue(
            _same_rows(mv, live),
            f"{context}\n  MV   {sorted(mv, key=repr)}\n  LIVE {sorted(live, key=repr)}",
        )


class TestFuzzRegression(ExactAssertions):
    def test_ci_seed_1586782166_sc3_bigint_column_holding_real(self) -> None:
        # Minimized from the CI failure: qty is BIGINT, the UPDATE stores REAL
        # 0.6 in five rows of the group, the DELETE removes them again.
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, label TEXT, qty BIGINT NOT NULL DEFAULT 0, amount NUMERIC(18, 4));"
        view = "SELECT label, COUNT(*) AS row_count, SUM(qty) AS total_qty, AVG(amount) AS average_amount FROM t GROUP BY label"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        conn.executemany(
            "INSERT INTO t VALUES (?, 'hot', ?, ?)",
            [(2, 91, None), (3, 32, 65.125), (5, 59, 5.125), (6, 96, 86.125), (7, 72, 36.125), (8, 7, 49.125)],
        )
        _install(conn, "v", view, schema)
        conn.execute(
            "UPDATE t SET qty = ((CASE WHEN (ROUND(4)) >= (1) THEN (ROUND(4)) - (1) ELSE (ROUND(4)) END) / 5) "
            "WHERE NOT COALESCE(qty, 0) BETWEEN 5 AND 22"
        )
        self.assertViewExact(conn, "v", view, "after UPDATE storing REAL 0.6")
        # Live SUM is REAL 10.0 here; an INTEGER-affinity backing column used to
        # turn it into INTEGER 10.
        self.assertIs(type(conn.execute("SELECT total_qty FROM v").fetchone()[0]), float)
        conn.execute("DELETE FROM t WHERE id < 8")
        self.assertViewExact(conn, "v", view, "after DELETE of the REAL rows")
        self.assertEqual(conn.execute("SELECT total_qty, typeof(total_qty) FROM v").fetchall(), [(7, "integer")])


class TestCancellationAndHistory(ExactAssertions):
    TYPES = ("REAL", "DOUBLE PRECISION", "FLOAT", "BIGINT", "INTEGER", "NUMERIC(18, 4)", "DECIMAL(10, 2)")

    def _run(self, typ: str, agg: str, ops: Sequence[Tuple[str, Tuple[Any, ...]]], grouped: bool = True) -> None:
        schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x {typ});"
        view = (
            f"SELECT g, COUNT(*) AS n, {agg} AS r FROM t GROUP BY g" if grouped else f"SELECT COUNT(*) AS n, {agg} AS r FROM t"
        )
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        _install(conn, "v", view, schema)
        for i, (sql, args) in enumerate(ops):
            conn.execute(sql, args)
            self.assertViewExact(conn, "v", view, f"{typ} {agg} step {i}: {sql} {args}")

    def test_large_value_retracted(self) -> None:
        ops = [
            ("INSERT INTO t VALUES (1, 'a', 1e16)", ()),
            ("INSERT INTO t VALUES (2, 'a', 1.5)", ()),
            ("DELETE FROM t WHERE id = 1", ()),
        ]
        for typ in self.TYPES:
            for agg in ("SUM(x)", "TOTAL(x)", "AVG(x)"):
                with self.subTest(typ=typ, agg=agg):
                    self._run(typ, agg, ops)
                    self._run(typ, agg, ops, grouped=False)

    def test_decimal_fractions_inserted_and_deleted(self) -> None:
        ops = [("INSERT INTO t VALUES (?, 'a', ?)", (i, v)) for i, v in enumerate([0.1, 0.2, 0.3, 0.7, 7], 1)]
        ops += [("DELETE FROM t WHERE id = ?", (i,)) for i in (2, 4, 1, 3)]
        for typ in self.TYPES:
            for agg in ("SUM(x)", "TOTAL(x)", "AVG(x)"):
                with self.subTest(typ=typ, agg=agg):
                    self._run(typ, agg, ops)

    def test_updates_group_moves_nulls_and_empty_groups(self) -> None:
        ops = [
            ("INSERT INTO t VALUES (1, 'a', 0.1)", ()),
            ("INSERT INTO t VALUES (2, 'a', 3)", ()),
            ("INSERT INTO t VALUES (3, 'b', NULL)", ()),
            ("UPDATE t SET x = 0.25 WHERE id = 2", ()),
            ("UPDATE t SET g = 'b' WHERE id = 1", ()),  # group move carrying a REAL
            ("UPDATE t SET x = 4, g = 'a' WHERE id = 3", ()),
            ("UPDATE t SET x = NULL WHERE id = 2", ()),
            ("DELETE FROM t WHERE g = 'b'", ()),  # group emptied
            ("INSERT INTO t VALUES (5, 'b', 2)", ()),  # group re-created with an integer
            ("UPDATE t SET x = x + 0.5", ()),  # multi-row statement
            ("UPDATE t SET x = CAST(x AS INTEGER)", ()),  # back to integers only
            ("DELETE FROM t", ()),
        ]
        for typ in self.TYPES:
            for agg in ("SUM(x)", "TOTAL(x)", "AVG(x)"):
                with self.subTest(typ=typ, agg=agg):
                    self._run(typ, agg, ops)
                    self._run(typ, agg, ops, grouped=False)

    def test_update_never_counts_new_value_twice(self) -> None:
        # Regression for an intermediate design that decided fold-vs-recompute
        # per step: the OLD step recomputed (magnitude guard) and already saw
        # the NEW row, then the NEW step folded it again (TOTAL 52.0 vs 26.0).
        ops = [
            ("INSERT INTO t VALUES (1, 'a', 9007199254740992)", ()),
            ("UPDATE t SET x = 26 WHERE id = 1", ()),
            ("UPDATE t SET x = 0.5 WHERE id = 1", ()),
            ("UPDATE t SET x = 9007199254740993 WHERE id = 1", ()),
            ("UPDATE t SET x = -3 WHERE id = 1", ()),
        ]
        for typ in ("BIGINT", "INTEGER", "NUMERIC(18, 4)"):
            for agg in ("SUM(x)", "TOTAL(x)", "AVG(x)"):
                with self.subTest(typ=typ, agg=agg):
                    self._run(typ, agg, ops)

    def test_negative_zero_and_storage_class(self) -> None:
        ops = [
            ("INSERT INTO t VALUES (1, 'a', -0.0)", ()),
            ("INSERT INTO t VALUES (2, 'a', 4)", ()),
            ("INSERT INTO t VALUES (3, 'a', 6.0)", ()),
            ("DELETE FROM t WHERE id = 2", ()),
        ]
        for typ in self.TYPES:
            for agg in ("SUM(x)", "TOTAL(x)", "AVG(x)"):
                with self.subTest(typ=typ, agg=agg):
                    self._run(typ, agg, ops)

    def test_beyond_two_pow_53(self) -> None:
        big = 2**61  # partial and final sums stay inside int64, beyond 2**53
        ops = [
            ("INSERT INTO t VALUES (1, 'a', ?)", (big,)),
            ("INSERT INTO t VALUES (2, 'a', ?)", (big - 1,)),
            ("INSERT INTO t VALUES (3, 'a', 1)", ()),
            ("DELETE FROM t WHERE id = 1", ()),
            ("INSERT INTO t VALUES (4, 'a', 0.5)", ()),
            ("DELETE FROM t WHERE id = 4", ()),
        ]
        for typ in ("BIGINT", "INTEGER"):
            for agg in ("SUM(x)", "TOTAL(x)", "AVG(x)"):
                with self.subTest(typ=typ, agg=agg):
                    self._run(typ, agg, ops)


class TestOtherViewShapes(ExactAssertions):
    SCHEMA = (
        "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT NOT NULL);\n"
        "CREATE TABLE t (id INTEGER PRIMARY KEY, pid INTEGER NOT NULL REFERENCES p(id), g TEXT, h INTEGER NOT NULL DEFAULT 0, x REAL, y BIGINT);"
    )
    VIEWS = {
        "join": "SELECT p.g AS g, SUM(t.x) AS s, TOTAL(t.x) AS tt, AVG(t.x) AS a FROM t JOIN p ON t.pid = p.id GROUP BY p.g",
        "union_all": "SELECT g, SUM(x) AS s, TOTAL(x) AS tt, AVG(x) AS a FROM (SELECT g, x FROM t UNION ALL SELECT g, 1.5 AS x FROM p) AS u GROUP BY g",
        "expr_group": "SELECT UPPER(g) AS gg, SUM(x) AS s, TOTAL(y) AS tt, AVG(x) AS a FROM t GROUP BY UPPER(g)",
        "filter": "SELECT g, SUM(x) FILTER (WHERE h > 0) AS s, AVG(y) AS a FROM t GROUP BY g",
        "having": "SELECT g, SUM(x) AS s FROM t GROUP BY g HAVING SUM(x) > 0",
        "where": "SELECT g, SUM(y) AS s, TOTAL(x) AS tt FROM t WHERE h <> 2 GROUP BY g",
        "multi_group": "SELECT g, h, SUM(x + y) AS s, AVG(y) AS a FROM t GROUP BY g, h",
        "distinct": "SELECT g, SUM(DISTINCT x) AS s, TOTAL(DISTINCT y) AS tt, AVG(DISTINCT x) AS a FROM t GROUP BY g",
    }

    def test_view_shapes(self) -> None:
        for name, view in self.VIEWS.items():
            with self.subTest(view=name):
                conn = sqlite3.connect(":memory:")
                conn.executescript(self.SCHEMA)
                conn.execute("INSERT INTO p VALUES (1, 'a'), (2, 'b')")
                _install(conn, "v", view, self.SCHEMA)
                steps = [
                    "INSERT INTO t VALUES (1, 1, 'a', 1, 1e16, 3), (2, 1, 'a', 1, 1.5, 0.5), (3, 2, 'b', 0, 0.1, 7), (4, 2, 'b', 2, 0.2, 2)",
                    "DELETE FROM t WHERE id IN (1, 3)",
                    "UPDATE t SET x = 0.3, h = 1 WHERE id = 4",
                    "UPDATE t SET g = 'b', pid = 2 WHERE id = 2",
                    "INSERT INTO t VALUES (5, 1, NULL, 2, 0.1, NULL)",
                    "UPDATE t SET y = y + 0.25",
                    "DELETE FROM t WHERE id = 2",
                ]
                for i, sql in enumerate(steps):
                    conn.execute(sql)
                    self.assertViewExact(conn, "v", view, f"{name} step {i}: {sql}")


def _random_value(rng: random.Random) -> Any:
    k = rng.random()
    if k < 0.12:
        return None
    if k < 0.40:
        return rng.randint(-50, 50)
    if k < 0.55:
        return rng.choice([0.1, 0.2, 0.3, 0.6, 1.5, -0.0, 2.5, 1e-300])
    if k < 0.62:
        return rng.choice([1e16, -1e16, 2.0**53, 9007199254740993, 1e308])
    if k < 0.85:
        return round(rng.uniform(-1000, 1000), rng.randint(0, 6))
    return rng.uniform(-1e6, 1e6)


RANDOM_VIEWS = (
    "SELECT g, COUNT(*) AS n, SUM(x) AS s, TOTAL(x) AS t, AVG(x) AS a FROM t GROUP BY g",
    "SELECT COUNT(*) AS n, SUM(x) AS s, TOTAL(y) AS t, AVG(x) AS a FROM t",
    "SELECT g, SUM(x + y) AS s, AVG(y) AS a FROM t WHERE y IS NOT NULL GROUP BY g",
    "SELECT g, h, SUM(x) AS s, COUNT(x) AS c, AVG(x) AS a FROM t GROUP BY g, h",
    "SELECT g, SUM(DISTINCT x) AS s, AVG(DISTINCT x) AS a, TOTAL(DISTINCT x) AS t FROM t GROUP BY g",
)


class TestRandomizedDifferential(ExactAssertions):
    """Seeded random DML against several column types; exact comparison after
    every statement, plus rebuild-vs-incremental equivalence at the end.

    Limits: one base table, five fixed view shapes (grouped, ungrouped,
    expression argument with WHERE, multi-column group, DISTINCT), values drawn
    from integers, short decimals, large magnitudes, -0.0 and subnormals. Seeds:
    MATVIEW_EXACT_FLOAT_SEEDS (default 40). A rejected write must make at least one original view raise "integer
    overflow" on a mirror database without materialized views (see
    TestIntegerOverflow); a view that raises on read while its materialized
    reader returns a value fails the test."""

    SEEDS = range(int(os.environ.get("MATVIEW_EXACT_FLOAT_SEEDS", "40")))

    def _run_seed(self, seed: int) -> Optional[str]:
        rng = random.Random(seed)
        types = ("REAL", "BIGINT", "INTEGER", "NUMERIC(18,4)", "DOUBLE PRECISION")
        xt, yt = rng.choice(types), rng.choice(types)
        schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, h INTEGER NOT NULL DEFAULT 0, x {xt}, y {yt});"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        groups = ["a", "b", "c", None]
        nid = 1
        for _ in range(rng.randint(0, 8)):
            conn.execute("INSERT INTO t VALUES (?,?,?,?,?)", (nid, rng.choice(groups), rng.randint(0, 2), _random_value(rng), _random_value(rng)))
            nid += 1
        views = []
        for i, v in enumerate(RANDOM_VIEWS):
            _install(conn, f"v{i}", v, schema)
            views.append((f"v{i}", v))
        # Mirror database without materialized views: every statement is
        # applied to both. A statement the MV database rejects must be one
        # that makes the original view unevaluable ("integer overflow").
        shadow = sqlite3.connect(":memory:")
        shadow.executescript(schema)
        shadow.executemany("INSERT INTO t VALUES (?,?,?,?,?)", conn.execute("SELECT * FROM t ORDER BY id").fetchall())
        for step in range(50):
            k = rng.random()
            ids = [r[0] for r in conn.execute("SELECT id FROM t")]
            if k < 0.35 or not ids:
                stmt = ("INSERT INTO t VALUES (?,?,?,?,?)", (nid, rng.choice(groups), rng.randint(0, 2), _random_value(rng), _random_value(rng)))
                nid += 1
            elif k < 0.55:
                stmt = ("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
            elif k < 0.80:
                stmt = ("UPDATE t SET g = ?, x = ? WHERE id = ?", (rng.choice(groups), _random_value(rng), rng.choice(ids)))
            elif k < 0.90:
                stmt = ("UPDATE t SET x = x * 0.5 + 0.1 WHERE h = ?", (rng.randint(0, 2),))
            else:
                stmt = ("DELETE FROM t WHERE h = ? AND id % 2 = 0", (rng.randint(0, 2),))
            before = conn.execute("SELECT * FROM t ORDER BY id").fetchall()
            try:
                conn.execute(*stmt)
            except sqlite3.OperationalError as exc:
                if "integer overflow" not in str(exc):
                    raise
                if conn.execute("SELECT * FROM t ORDER BY id").fetchall() != before:
                    return f"seed={seed} rejected write was not rolled back"
                shadow.execute("SAVEPOINT probe")
                shadow.execute(*stmt)
                raised = False
                for _, v in views:
                    try:
                        shadow.execute(v).fetchall()
                    except sqlite3.OperationalError as live_exc:
                        raised = raised or "integer overflow" in str(live_exc)
                shadow.execute("ROLLBACK TO probe")
                shadow.execute("RELEASE probe")
                if not raised:
                    return f"seed={seed} step={step} write {stmt} rejected although every original view still evaluates"
                continue
            shadow.execute(*stmt)
            for name, v in views:
                live = conn.execute(v).fetchall()
                mv = conn.execute(f"SELECT * FROM {name}").fetchall()
                if not _same_rows(mv, live):
                    return f"seed={seed} ({xt},{yt}) step={step} {v}\n  MV   {sorted(mv, key=repr)}\n  LIVE {sorted(live, key=repr)}"
        # Metamorphic: a freshly populated view over the final state must equal
        # the incrementally maintained one (and both equal the live view).
        for name, v in views:
            _install(conn, f"{name}_rebuilt", v, schema)
            inc = conn.execute(f"SELECT * FROM {name}").fetchall()
            reb = conn.execute(f"SELECT * FROM {name}_rebuilt").fetchall()
            if not _same_rows(inc, reb):
                return f"seed={seed} rebuild != incremental for {v}\n  INC {inc}\n  REB {reb}"
        return None

    def test_random_seeds(self) -> None:
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                failure = self._run_seed(seed)
                self.assertIsNone(failure, failure)


class TestIntegerFastPathAndBackfill(ExactAssertions):
    def test_integer_only_workload_never_marks_groups_dirty(self) -> None:
        # Integer-only data keeps the O(1) exact fold: no native recompute.
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT NOT NULL, x BIGINT);"
        view = "SELECT g, SUM(x) AS s, TOTAL(x) AS t, AVG(x) AS a FROM t GROUP BY g"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        _install(conn, "v", view, schema)
        recomputes: List[str] = []
        conn.set_trace_callback(lambda sql: recomputes.append(sql) if "__mv_fold_dirty = 0" in sql else None)
        for i in range(1, 40):
            conn.execute("INSERT INTO t VALUES (?, ?, ?)", (i, "ab"[i % 2], i * 1000003))
        conn.execute("UPDATE t SET x = x - 7 WHERE id % 3 = 0")
        conn.execute("DELETE FROM t WHERE id % 5 = 0")
        conn.set_trace_callback(None)
        self.assertViewExact(conn, "v", view)
        store = conn.execute("SELECT name FROM sqlite_master WHERE name LIKE '__matview_store_v%' AND type = 'table'").fetchone()[0]
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({store})")]
        self.assertIn("__mv_fold_dirty", cols)
        acnt_cols = [c for c in cols if c.startswith("__mv_acnt_")]
        self.assertTrue(acnt_cols)
        rows = conn.execute(f"SELECT __mv_fold_dirty, {', '.join(acnt_cols)} FROM {store}").fetchall()
        self.assertTrue(all(all(v == 0 for v in r) for r in rows), rows)

    def test_real_value_marks_dirty_and_is_cleared(self) -> None:
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT NOT NULL, x BIGINT);"
        view = "SELECT g, SUM(x) AS s FROM t GROUP BY g"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        _install(conn, "v", view, schema)
        conn.execute("INSERT INTO t VALUES (1, 'a', 1), (2, 'a', 0.5)")
        store = conn.execute("SELECT name FROM sqlite_master WHERE name LIKE '__matview_store_v%' AND type = 'table'").fetchone()[0]
        acnt = [r[1] for r in conn.execute(f"PRAGMA table_info({store})") if r[1].startswith("__mv_acnt_")][0]
        self.assertEqual(conn.execute(f"SELECT __mv_fold_dirty, {acnt} FROM {store}").fetchall(), [(0, 1)])
        self.assertViewExact(conn, "v", view)

    def test_incremental_backfill_with_concurrent_writer(self) -> None:
        # Independent connection writes between committed backfill batches,
        # inside and outside the already-folded pk range, with REAL and large
        # values. The final state must equal the live view exactly.
        views = (
            "SELECT g, COUNT(*) AS n, SUM(x) AS s, TOTAL(x) AS t, AVG(x) AS a FROM t GROUP BY g",
            "SELECT COUNT(*) AS n, SUM(x) AS s, TOTAL(x) AS t, AVG(x) AS a FROM t",
        )
        with tempfile.TemporaryDirectory() as d:
            for seed in range(int(os.environ.get("MATVIEW_EXACT_FLOAT_BACKFILL_SEEDS", "30"))):
                rng = random.Random(seed)
                xt = rng.choice(["REAL", "BIGINT", "NUMERIC(18,4)"])
                view = rng.choice(views)
                schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT NOT NULL, x {xt});"
                path = os.path.join(d, f"bf{seed}.db")
                c1 = sqlite3.connect(path, isolation_level=None)
                c1.executescript(schema)
                for i in range(1, 25):
                    c1.execute("INSERT INTO t VALUES (?,?,?)", (i, rng.choice("ab"), _random_value(rng)))
                c2 = sqlite3.connect(path, isolation_level=None, timeout=10)
                next_id = [100]

                def writer(_batches: int, _done: int) -> None:
                    for _ in range(rng.randint(1, 3)):
                        ids = [r[0] for r in c2.execute("SELECT id FROM t")]
                        k = rng.random()
                        try:
                            if k < 0.35 or not ids:
                                c2.execute("INSERT INTO t VALUES (?,?,?)", (next_id[0], rng.choice("ab"), _random_value(rng)))
                                next_id[0] += 1
                            elif k < 0.6:
                                c2.execute("DELETE FROM t WHERE id = ?", (rng.choice(ids),))
                            else:
                                c2.execute("UPDATE t SET x = ?, g = ? WHERE id = ?", (_random_value(rng), rng.choice("ab"), rng.choice(ids)))
                        except sqlite3.OperationalError as exc:
                            if "integer overflow" not in str(exc):
                                raise

                _install(c1, "v", view, schema, incremental_backfill=True, batch_size=3, on_progress=writer)
                for _ in range(5):
                    writer(0, 0)
                with self.subTest(seed=seed, typ=xt, view=view):
                    try:
                        live = c1.execute(view).fetchall()
                    except sqlite3.OperationalError as exc:
                        if "integer overflow" in str(exc):
                            continue
                        raise
                    mv = c1.execute("SELECT * FROM v").fetchall()
                    self.assertTrue(_same_rows(mv, live), f"MV {mv}\nLIVE {live}")
                c1.close()
                c2.close()


class TestIntegerOverflow(unittest.TestCase):
    def test_write_that_overflows_native_sum_is_rejected(self) -> None:
        """SQLite's native SUM raises "integer overflow" past int64, so the
        original view cannot be evaluated over such a state; the materialized
        SUM cannot represent that error. The maintained integer fold would
        silently turn into REAL (SQLite arithmetic); instead the group is
        recomputed with the native SUM, which raises, and the write is
        rejected and rolled back. (Policy: explicit rejection rather than a
        divergent value. On origin/master the write succeeded and the reader
        returned 9.223372036854776e+18.)"""
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, x BIGINT);"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        _install(conn, "v", "SELECT SUM(x) AS s, AVG(x) AS a FROM t", schema)
        conn.execute("INSERT INTO t VALUES (1, ?)", (2**62,))
        with self.assertRaisesRegex(sqlite3.OperationalError, "integer overflow"):
            conn.execute("INSERT INTO t VALUES (2, ?)", (2**62,))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM t").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT * FROM v").fetchall(), conn.execute("SELECT SUM(x), AVG(x) FROM t").fetchall())

    def test_avg_alone_never_rejects_writes(self) -> None:
        # Native AVG does not raise on int64 overflow (it continues in floating
        # point); the hidden sum is recomputed with TOTAL, which does not raise.
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, x BIGINT);"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        view = "SELECT COUNT(*) AS n, AVG(x) AS a, TOTAL(x) AS t FROM t"
        _install(conn, "v", view, schema)
        conn.execute("INSERT INTO t VALUES (1, ?), (2, ?), (3, 1)", (2**62, 2**62))
        conn.execute("DELETE FROM t WHERE id = 1")
        live = conn.execute(view).fetchall()
        self.assertTrue(_same_rows(conn.execute("SELECT * FROM v").fetchall(), live), live)


TWO53 = 2**53


class TestBinary64IntegerLimit(ExactAssertions):
    """Integer folds into REAL accumulators (TOTAL, hidden AVG sums) are exact
    only while |partial sums| stay below 2**52. Native TOTAL/AVG over integers
    use an exact int64 sum: TOTAL({2**53, 1, 1}) == 9007199254740994.0, whereas
    a binary64 fold gives 2**53 + 1 -> 2**53 (ties to even) twice."""

    VIEWS = (
        "SELECT TOTAL(x) AS tt FROM t",
        "SELECT AVG(x) AS a FROM t",
        "SELECT g, COUNT(*) AS n, AVG(x) AS a, TOTAL(x) AS tt, SUM(x) AS s FROM t GROUP BY g",
    )
    # Native TOTAL = 2**53 + 4 exactly; a binary64 fold loses both 1s (2**53 + 2).
    ROWS = [(1, "a", TWO53), (2, "a", 1), (3, "a", 1), (4, "a", 2)]

    def test_trigger_folds(self) -> None:
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x BIGINT);"
        for view in self.VIEWS:
            with self.subTest(view=view):
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema)
                _install(conn, "v", view, schema)
                for row in self.ROWS:
                    conn.execute("INSERT INTO t VALUES (?, ?, ?)", row)
                    self.assertViewExact(conn, "v", view, f"after insert {row}")
                for rid in (2, 1, 3):
                    conn.execute("DELETE FROM t WHERE id = ?", (rid,))
                    self.assertViewExact(conn, "v", view, f"after delete {rid}")

    def test_backfill_chunk_merges_with_integer_only_chunks(self) -> None:
        # Regression: a merge recompute over the whole base table counted rows
        # of later chunks, which were then folded again (TOTAL 24.0 vs 12.0).
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x BIGINT);"
        datasets = {
            "limit": self.ROWS,
            "double_count": [(1, "a", 3 * 10**15), (2, "a", -3 * 10**15), (3, "a", 5), (4, "a", 7)],
        }
        for label, rows in datasets.items():
            for view in self.VIEWS:
                for batch_size in (1, 2, 3):
                    with self.subTest(data=label, view=view, batch_size=batch_size):
                        conn = sqlite3.connect(":memory:")
                        conn.executescript(schema)
                        conn.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
                        _install(conn, "v", view, schema, incremental_backfill=True, batch_size=batch_size)
                        self.assertViewExact(conn, "v", view, "after backfill")
                        conn.execute("INSERT INTO t VALUES (100, 'a', 1)")
                        self.assertViewExact(conn, "v", view, "after a post-backfill write")

    def test_backfill_with_writes_between_chunks_beyond_limit(self) -> None:
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x BIGINT);"
        view = "SELECT g, COUNT(*) AS n, AVG(x) AS a, TOTAL(x) AS tt, SUM(x) AS s FROM t GROUP BY g"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lim.db")
            c1 = sqlite3.connect(path, isolation_level=None)
            c1.executescript(schema)
            c1.executemany("INSERT INTO t VALUES (?, ?, ?)", [(i, "ab"[i % 2], TWO53 if i % 3 == 0 else 1) for i in range(1, 13)])
            c2 = sqlite3.connect(path, isolation_level=None, timeout=10)
            nxt = [100]

            def writer(_b: int, _d: int) -> None:
                c2.execute("INSERT INTO t VALUES (?, 'a', ?)", (nxt[0], TWO53 if nxt[0] % 2 else 1))
                c2.execute("UPDATE t SET x = x + 1 WHERE id = 12")
                c2.execute("DELETE FROM t WHERE id = 3")
                nxt[0] += 1

            _install(c1, "v", view, schema, incremental_backfill=True, batch_size=2, on_progress=writer)
            self.assertViewExact(c1, "v", view, "after backfill with concurrent writer")
            c1.close()
            c2.close()


class TestDistinctNumericConversion(ExactAssertions):
    def test_text_and_blob_inputs(self) -> None:
        cases = [
            ("VARCHAR(10)", ["2.25", "5", "abc", "7"]),
            ("TEXT", ["1e3", "0x10", " 4 "]),
            ("BLOB", [b"\x01", b"12"]),
        ]
        views = (
            "SELECT g, SUM(DISTINCT x) AS s, TOTAL(DISTINCT x) AS tt, AVG(DISTINCT x) AS a FROM t GROUP BY g",
            "SELECT g, SUM(x) AS s, TOTAL(x) AS tt, AVG(x) AS a FROM t GROUP BY g",
        )
        for typ, values in cases:
            for view in views:
                with self.subTest(typ=typ, view=view):
                    schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x {typ});"
                    conn = sqlite3.connect(":memory:")
                    conn.executescript(schema)
                    _install(conn, "v", view, schema)
                    for i, value in enumerate(values, 1):
                        conn.execute("INSERT INTO t VALUES (?, ?, ?)", (i, "ab"[i % 2], value))
                        self.assertViewExact(conn, "v", view, f"after insert {value!r}")
                    conn.execute("DELETE FROM t WHERE id = 1")
                    self.assertViewExact(conn, "v", view, "after delete")

    def test_avg_distinct_never_rejects_writes(self) -> None:
        # Native AVG(DISTINCT) does not raise on int64 overflow; the hidden sum
        # is TOTAL(DISTINCT), so the write is accepted as without the view.
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x BIGINT);"
        view = "SELECT g, COUNT(*) AS n, AVG(DISTINCT x) AS a, TOTAL(DISTINCT x) AS tt FROM t GROUP BY g"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        _install(conn, "v", view, schema)
        conn.execute("INSERT INTO t VALUES (1, 'a', 9223372036854775807)")
        conn.execute("INSERT INTO t VALUES (2, 'a', 5)")
        self.assertViewExact(conn, "v", view)

    def test_initial_populate_of_avg_over_overflowing_sum(self) -> None:
        # Native AVG is defined although SUM over the same rows overflows; the
        # hidden sum is populated with TOTAL so installation succeeds.
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, y BIGINT);"
        view = "SELECT g, AVG(y) AS a FROM t GROUP BY g"
        for incremental in (False, True):
            with self.subTest(incremental_backfill=incremental):
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema)
                conn.execute("INSERT INTO t VALUES (1, 'a', 9223372036854775807), (2, 'a', 5)")
                _install(conn, "v", view, schema, incremental_backfill=incremental, batch_size=1)
                self.assertViewExact(conn, "v", view)


class TestUntypedColumns(ExactAssertions):
    """SQLite columns without a declared type have BLOB affinity. Regression:
    the schema parser dropped them, so the native join recompute referenced a
    column it did not know and every write to the table failed with "no such
    column" (found by the randomized join differential); grouping by such a
    column stored numbers as TEXT."""

    def test_parsed_and_introspected_as_blob(self) -> None:
        from matview import introspect_sqlite_schemas
        from matview.sql_frontend import parse_table_schema

        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, x, y NOT NULL, z);"
        parsed = parse_table_schema(schema)["t"].columns
        self.assertEqual({c: parsed[c].data_type for c in ("x", "y", "z")}, {"x": "BLOB", "y": "BLOB", "z": "BLOB"})
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        introspected = introspect_sqlite_schemas(conn)["t"].columns
        self.assertEqual({c: introspected[c].data_type for c in ("x", "y", "z")}, {"x": "BLOB", "y": "BLOB", "z": "BLOB"})

    def test_pragma_fallback_records_blob(self) -> None:
        """Tables whose DDL yields no parsed columns are introspected through
        PRAGMA table_info; a column without a declared type has BLOB affinity
        there too (recording it as TEXT converts grouped numbers to text)."""
        from unittest import mock

        import matview.connectivity as connectivity
        from matview import introspect_sqlite_schemas

        conn = sqlite3.connect(":memory:")
        conn.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY, x, y NOT NULL, z REAL);")
        with mock.patch.object(connectivity, "parse_table_schema", lambda *a, **k: {}):
            cols = introspect_sqlite_schemas(conn)["t"].columns
        self.assertEqual({c: cols[c].data_type for c in ("x", "y", "z")}, {"x": "BLOB", "y": "BLOB", "z": "REAL"})

    def test_views_over_untyped_columns(self) -> None:
        schema = (
            "CREATE TABLE p (pid INTEGER PRIMARY KEY, g TEXT, w);\n"
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, x, h INTEGER NOT NULL DEFAULT 0);"
        )
        views = {
            "join_avg": "SELECT p.g AS g, COUNT(*) AS n, SUM(p.w) AS s, AVG(o.x * p.w) AS a FROM o JOIN p ON o.pid = p.pid GROUP BY p.g",
            "single": "SELECT pid, SUM(x) AS s, TOTAL(x) AS tt, AVG(x) AS a, MAX(x) AS m FROM o GROUP BY pid",
            "group_by_untyped": "SELECT x, COUNT(*) AS n FROM o GROUP BY x",
        }
        steps = [
            "INSERT INTO p VALUES (1, 'a', 2), (2, 'b', 0.5)",
            "INSERT INTO o VALUES (1, 1, 1, 0), (2, 1, 2.5, 0), (3, 2, '7', 0), (4, 2, 1e16, 0)",
            "INSERT INTO o VALUES (3, 1, 0.6, 0) ON CONFLICT(id) DO UPDATE SET x = excluded.x, pid = excluded.pid",
            "DELETE FROM o WHERE id = 4",
            "UPDATE p SET w = '3' WHERE pid = 2",
            "UPDATE o SET x = NULL WHERE id = 1",
        ]
        for name, view in views.items():
            with self.subTest(view=name):
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema)
                _install(conn, "v", view, schema)
                for i, sql in enumerate(steps):
                    conn.execute(sql)
                    self.assertViewExact(conn, "v", view, f"{name} step {i}: {sql}")


class TestInfinityAndNaN(ExactAssertions):
    """inf + -inf is NaN, which SQLite returns as NULL: TOTAL is then NULL,
    not 0.0. The TOTAL backing column was NOT NULL, so the write was rejected
    ("NOT NULL constraint failed"), or, under INSERT OR REPLACE, silently
    stored the column default 0.0 (found by the randomized differential)."""

    SCHEMA = (
        "CREATE TABLE p (pid INTEGER PRIMARY KEY, g TEXT, w DOUBLE PRECISION);\n"
        "CREATE TABLE t (id INTEGER PRIMARY KEY, pid INTEGER, g TEXT, x REAL);"
    )
    VIEWS = {
        "single": "SELECT g, TOTAL(x) AS tt, SUM(x) AS s, AVG(x) AS a FROM t GROUP BY g",
        "scalar": "SELECT TOTAL(x) AS tt, SUM(x) AS s, AVG(x) AS a, COUNT(*) AS n FROM t",
        "join": "SELECT p.g AS g, TOTAL(t.x * p.w) AS tt, SUM(p.w) AS s, AVG(t.x) AS a FROM t JOIN p ON t.pid = p.pid GROUP BY p.g",
        "union_all": "SELECT g, TOTAL(x) AS tt, SUM(x) AS s FROM (SELECT g, x FROM t UNION ALL SELECT g, w AS x FROM p) AS u GROUP BY g",
        "distinct": "SELECT g, TOTAL(DISTINCT x) AS tt, AVG(DISTINCT x) AS a FROM t GROUP BY g",
    }

    def test_nan_sums(self) -> None:
        steps = [
            "INSERT INTO p VALUES (1, 'a', 1.0)",
            "INSERT INTO t VALUES (1, 1, 'a', 1e308 * 10), (2, 1, 'a', 2.5)",
            "INSERT INTO t VALUES (3, 1, 'a', -1e308 * 10)",
            "INSERT OR REPLACE INTO p VALUES (1, 'a', -1e308 * 10)",
            "UPDATE t SET x = 4 WHERE id = 3",
            "INSERT OR REPLACE INTO t VALUES (2, 1, 'a', -1e308 * 10)",
            "DELETE FROM t WHERE id = 1",
            "DELETE FROM t",
        ]
        for name, view in self.VIEWS.items():
            with self.subTest(view=name):
                conn = sqlite3.connect(":memory:")
                conn.executescript(self.SCHEMA)
                _install(conn, "v", view, self.SCHEMA)
                for i, sql in enumerate(steps):
                    conn.execute(sql)
                    self.assertViewExact(conn, "v", view, f"{name} step {i}: {sql}")


class TestStorageClassOnlyUpdates(ExactAssertions):
    """In a column without affinity, UPDATE can change INTEGER 3 into REAL 3.0
    (or back) without changing the value: 3 IS 3.0 is true in SQLite. The
    UPDATE trigger guard skipped such rows, so SUM kept REAL 3.0 where the
    original view returns INTEGER 3 (found by the randomized differential:
    UPSERT ... SET x = excluded.x * 2, then UPDATE ... SET x = CAST(x AS
    INTEGER))."""

    def test_integer_real_round_trip(self) -> None:
        views = (
            "SELECT g, COUNT(*) AS n, SUM(x) AS s, TOTAL(x) AS tt, AVG(x) AS a FROM t GROUP BY g",
            "SELECT SUM(x - h) AS s, SUM(DISTINCT x) AS sd, TOTAL(DISTINCT x) AS td FROM t",
        )
        steps = [
            "INSERT INTO t VALUES (1, 'k1', 'a', 1, 1.5)",
            "INSERT INTO t VALUES (2, 'k1', 'a', 1, 1.5) ON CONFLICT(k) DO UPDATE SET x = excluded.x * 2",
            "UPDATE t SET x = CAST(x AS INTEGER)",
            "UPDATE t SET x = 3.0",
            "INSERT INTO t VALUES (3, 'k3', 'a', 0, 4)",
            "UPDATE t SET x = x * 1.0 WHERE id = 3",
            "UPDATE t SET x = CAST(x AS INTEGER) WHERE id = 3",
        ]
        for typ in ("BLOB", ""):
            for view in views:
                with self.subTest(typ=typ, view=view):
                    schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, k TEXT UNIQUE, g TEXT, h INTEGER NOT NULL DEFAULT 0, x {typ});"
                    conn = sqlite3.connect(":memory:")
                    conn.executescript(schema)
                    _install(conn, "v", view, schema)
                    for i, sql in enumerate(steps):
                        conn.execute(sql)
                        self.assertViewExact(conn, "v", view, f"{typ!r} step {i}: {sql}")


class TestSqliteAffinityRules(unittest.TestCase):
    def test_affinity_rules_in_order(self) -> None:
        from matview.dialects.base import _sqlite_type_affinity

        expected = {
            "INTEGER": "INTEGER", "BIGINT": "INTEGER", "POINT REAL": "INTEGER",
            "FLOATING POINT": "INTEGER", "VARCHAR(10)": "TEXT", "CHARINT": "INTEGER",
            "CLOB": "TEXT", "BLOB": "BLOB", "": "BLOB", "REAL": "REAL", "FLOAT": "REAL",
            "DOUBLE PRECISION": "REAL", "NUMERIC(18,4)": "NUMERIC", "DECIMAL(10,2)": "NUMERIC",
            "BOOLEAN": "NUMERIC", "DATETIME": "NUMERIC",
        }
        for declared, affinity in expected.items():
            self.assertEqual(_sqlite_type_affinity(declared), affinity, declared)

    def test_sum_storage_follows_actual_affinity(self) -> None:
        # "POINT REAL" has INTEGER affinity in SQLite (contains "INT"); its SUM
        # can be INTEGER, so the backing column must not have REAL affinity.
        for declared, stored in (("POINT REAL", "BLOB"), ("REAL", "REAL"), ("DOUBLE", "REAL"), ("BIGINT", "BLOB")):
            with self.subTest(declared=declared):
                schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, x {declared});"
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema)
                _install(conn, "v", "SELECT SUM(x) AS s FROM t", schema)
                ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name LIKE '__matview_store_v%' AND type='table'").fetchone()[0]
                self.assertIn(f"s {stored}", ddl)
                conn.execute("INSERT INTO t VALUES (1, 2), (2, 3)")
                live = conn.execute("SELECT SUM(x) FROM t").fetchall()
                self.assertTrue(_same_rows(conn.execute("SELECT * FROM v").fetchall(), live), live)


class TestUpgradeFromPreviousGenerator(ExactAssertions):
    """Installations created by the previous generator (fixture captured from
    origin/p2/tier2-window-partition-scoping 0542ca6 with SQLite 3.46.1) must be
    rebuilt -- never kept with drifting triggers or repopulated into the old
    backing table."""

    FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "sqlite_install_generator_v1.json")
    VIEWS = {
        "v_single": "SELECT g, SUM(x) AS s, AVG(x) AS a, TOTAL(x) AS tt FROM t GROUP BY g",
        "v_join": "SELECT p.g AS g, SUM(o.x) AS s FROM o JOIN p ON o.pid = p.pid GROUP BY p.g",
    }

    def _old_install(self) -> Tuple[sqlite3.Connection, dict]:
        import json

        with open(self.FIXTURE, encoding="utf-8") as fh:
            fx = json.load(fh)
        conn = sqlite3.connect(":memory:")
        conn.executescript(fx["schema"])
        for ddl in fx["ddl"]:
            conn.execute(ddl)
        conn.executemany(
            f"INSERT INTO __matview_catalog ({', '.join(fx['catalog_cols'])}) VALUES ({', '.join('?' for _ in fx['catalog_cols'])})",
            [tuple(r) for r in fx["catalog"]],
        )
        conn.commit()
        return conn, fx

    def _drift(self, conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO t VALUES (1, 'a', 10000000000000000), (2, 'a', 1.5)")
        conn.execute("DELETE FROM t WHERE id = 1")
        conn.execute("INSERT INTO p VALUES (1, 'x')")
        conn.execute("INSERT INTO o VALUES (1, 1, 1e16), (2, 1, 1.5)")
        conn.execute("DELETE FROM o WHERE id = 1")
        conn.commit()

    def test_old_install_drifts_then_repair_rebuilds(self) -> None:
        from matview.manager import MaterializedViewManager
        from matview import Dialect

        conn, _ = self._old_install()
        self._drift(conn)
        # The old triggers really drift (sanity check of the fixture).
        self.assertEqual(conn.execute("SELECT s FROM v_single").fetchone()[0], 2)
        mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        self.assertEqual(mgr.repair_view(), {"v_single": "rebuilt", "v_join": "rebuilt"})
        for name, view in self.VIEWS.items():
            self.assertViewExact(conn, name, view, f"{name} after repair")
        conn.execute("INSERT INTO t VALUES (3, 'a', 0.1)")
        conn.execute("DELETE FROM t WHERE id = 3")
        conn.execute("INSERT INTO o VALUES (3, 1, 0.1)")
        conn.execute("DELETE FROM o WHERE id = 3")
        for name, view in self.VIEWS.items():
            self.assertViewExact(conn, name, view, f"{name} maintained after repair")

    def test_reregistering_an_old_install_rebuilds_it(self) -> None:
        from matview.manager import MaterializedViewManager
        from matview import Dialect

        conn, _ = self._old_install()
        mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        for name, view in self.VIEWS.items():
            mgr.register_view(f"CREATE VIEW {name} AS {view};")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(__matview_store_v_single)")]
        self.assertIn("__mv_fold_dirty", cols)
        self._drift(conn)
        for name, view in self.VIEWS.items():
            self.assertViewExact(conn, name, view, f"{name} after re-registration")

    def test_old_install_reported_drifted_and_sync_views_rebuilds_it(self) -> None:
        from matview.manager import MaterializedViewManager
        from matview import Dialect

        conn, _ = self._old_install()
        self._drift(conn)
        mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        self.assertEqual(mgr.check_schema_drift(), {"v_single": "drifted", "v_join": "drifted"})
        result = mgr.sync_views([f"CREATE VIEW {name} AS {view};" for name, view in self.VIEWS.items()])
        self.assertEqual(result, {"v_single": "updated", "v_join": "updated"})
        self.assertEqual(mgr.check_schema_drift(), {"v_single": "ok", "v_join": "ok"})
        for name, view in self.VIEWS.items():
            self.assertViewExact(conn, name, view, f"{name} after sync_views")
        conn.execute("INSERT INTO t VALUES (3, 'a', 0.1)")
        conn.execute("DELETE FROM t WHERE id = 3")
        conn.execute("INSERT INTO o VALUES (3, 1, 0.1)")
        conn.execute("DELETE FROM o WHERE id = 3")
        for name, view in self.VIEWS.items():
            self.assertViewExact(conn, name, view, f"{name} maintained after sync_views")

    def _install_with_patched_plan(self, patch_plan: Any = None, generator_version: Optional[str] = None) -> Tuple[sqlite3.Connection, Any]:
        from unittest import mock

        import matview.manager as manager_mod
        from matview import Dialect
        from matview.manager import MaterializedViewManager

        conn = sqlite3.connect(":memory:")
        conn.executescript(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x REAL);"
            "CREATE TABLE p (pid INTEGER PRIMARY KEY, g TEXT);"
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, x REAL);"
        )
        real_generate = manager_mod.generate_materialized_view

        def generate(*args: Any, **kwargs: Any) -> Any:
            plan = real_generate(*args, **kwargs)
            return patch_plan(plan) if patch_plan else plan

        mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        with mock.patch.object(manager_mod, "generate_materialized_view", generate), mock.patch.object(
            manager_mod, "MATVIEW_GENERATOR_VERSION", generator_version or manager_mod.MATVIEW_GENERATOR_VERSION
        ):
            for name, view in self.VIEWS.items():
                mgr.register_view(f"CREATE VIEW {name} AS {view};")
            self.assertEqual(mgr.check_schema_drift(), {"v_single": "ok", "v_join": "ok"})
        return conn, mgr

    def _assert_drifted_then_synced(self, conn: sqlite3.Connection, mgr: Any) -> None:
        self.assertEqual(mgr.check_schema_drift(), {"v_single": "drifted", "v_join": "drifted"})
        result = mgr.sync_views([f"CREATE VIEW {name} AS {view};" for name, view in self.VIEWS.items()])
        self.assertEqual(result, {"v_single": "updated", "v_join": "updated"})
        self.assertEqual(mgr.check_schema_drift(), {"v_single": "ok", "v_join": "ok"})
        self._drift(conn)
        for name, view in self.VIEWS.items():
            self.assertViewExact(conn, name, view, f"{name} after sync_views")

    def test_generator_version_change_alone_rebuilds(self) -> None:
        """A trigger-only generator change is signalled by bumping
        MATVIEW_GENERATOR_VERSION: an installation made under another version,
        with identical view SQL and backing DDL, must be reported drifted and
        rebuilt."""
        conn, mgr = self._install_with_patched_plan(generator_version="previous-generator")
        self._assert_drifted_then_synced(conn, mgr)

    def test_backing_ddl_change_alone_rebuilds(self) -> None:
        """An installation whose backing-table DDL differs from what the current
        generator emits (same view SQL, same generator version) must be
        reported drifted and rebuilt."""
        import dataclasses

        def older_backing(plan: Any) -> Any:
            return dataclasses.replace(plan, backing_table_ddl=plan.backing_table_ddl.rstrip().rstrip(";") + " ;")

        conn, mgr = self._install_with_patched_plan(patch_plan=older_backing)
        self._assert_drifted_then_synced(conn, mgr)


class TestPerformanceSurface(unittest.TestCase):
    def test_group_key_index_recommended_and_cost_documented(self) -> None:
        from matview import generate_materialized_view, introspect_sqlite_schemas

        conn = sqlite3.connect(":memory:")
        conn.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT NOT NULL, x REAL);")
        plan = generate_materialized_view(
            introspect_sqlite_schemas(conn), "CREATE VIEW v AS SELECT g, SUM(x) AS s FROM t GROUP BY g;", dialect="sqlite"
        )
        self.assertIn("CREATE INDEX __matview_base_idx_t__g ON t (g);", plan.recommended_base_indexes)
        notes = plan.diagnostics.details["performance_notes"]
        self.assertTrue(any(n.startswith("sqlite_sum_total_avg_native_recompute_on_real") for n in notes), notes)

    def test_integer_writes_issue_no_base_table_scan(self) -> None:
        # Measured bound (SQLite 3.46.1, 100k rows, 2000 inserts): integer-only
        # groups ~0.005 s total (O(1) fold); REAL groups of 10k rows ~4.9 s with
        # a group-key index; 100-row REAL groups ~0.08 s with an index and
        # ~7 s without one. Here: integer writes must not read the base table.
        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, x BIGINT);"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        conn.executemany("INSERT INTO t (g, x) VALUES (?, ?)", [(i % 10, i) for i in range(5000)])
        _install(conn, "v", "SELECT g, SUM(x) AS s, AVG(x) AS a, TOTAL(x) AS tt FROM t GROUP BY g", schema)
        steps_before = conn.execute("SELECT 1").fetchall()  # warm statement cache
        del steps_before
        vm_steps = []
        conn.set_progress_handler(lambda: vm_steps.append(1) and 0, 1000)
        for i in range(200):
            conn.execute("INSERT INTO t (g, x) VALUES (?, ?)", (i % 10, i))
        conn.set_progress_handler(None, 0)
        # A single base-table scan of 5000 rows costs far more than 1000 VM
        # steps; 200 O(1) folds stay well below one scan per write.
        self.assertLess(len(vm_steps), 200, len(vm_steps))


class TestSqliteStorageClassAndGroupKeyExactness(ExactAssertions):
    """Regression coverage for SQLite storage-class preservation, raw affinity
    parsing, canonical numeric group keys (1 vs 1.0, 0 vs -0.0, +/-2^63),
    and surviving group-key representative refresh on DELETE/UPDATE."""

    def test_parse_table_schema_preserves_sqlite_affinity(self) -> None:
        from matview import Dialect, parse_table_schema
        from matview.dialects.base import _sqlite_type_affinity

        ddl = """
        CREATE TABLE raw_types (
            id INTEGER PRIMARY KEY,
            b BLOB,
            s STRING,
            pr POINT REAL NOT NULL,
            fp FLOATING POINT,
            ubi UNSIGNED BIG INT,
            any_col ANY,
            "primary" BLOB,
            dec DECIMAL(12, 4),
            vc VARCHAR(32)
        );
        """
        schemas = parse_table_schema(ddl, dialect=Dialect.SQLITE)
        cols = schemas["raw_types"].columns
        self.assertEqual(_sqlite_type_affinity(cols["b"].data_type), "BLOB")
        self.assertEqual(_sqlite_type_affinity(cols["s"].data_type), "NUMERIC")
        self.assertEqual(_sqlite_type_affinity(cols["pr"].data_type), "INTEGER")
        self.assertTrue(cols["pr"].not_null)
        self.assertEqual(_sqlite_type_affinity(cols["fp"].data_type), "INTEGER")
        self.assertEqual(_sqlite_type_affinity(cols["ubi"].data_type), "INTEGER")
        self.assertEqual(_sqlite_type_affinity(cols["any_col"].data_type), "BLOB")
        self.assertEqual(_sqlite_type_affinity(cols["primary"].data_type), "BLOB")
        self.assertEqual(_sqlite_type_affinity(cols["dec"].data_type), "NUMERIC")
        self.assertEqual(_sqlite_type_affinity(cols["vc"].data_type), "TEXT")

    def test_minmax_group_key_and_distinct_preserve_storage_classes(self) -> None:
        cases = (
            (
                "max_min_blob_real_4000",
                "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x BLOB);",
                "SELECT g, MAX(x) AS mx, MIN(x) AS mn FROM t GROUP BY g",
                [(1, "a", 4000.0)],
                [
                    ("INSERT INTO t VALUES (2, 'a', 3000.0)", ()),
                    ("DELETE FROM t WHERE id = 2", ()),
                    ("UPDATE t SET x = 5000.0 WHERE id = 1", ()),
                    ("UPDATE t SET x = 5000 WHERE id = 1", ()),
                ],
            ),
            (
                "min_case_real_not_coerced_to_text",
                "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, y REAL, z TEXT);",
                "SELECT g, MIN(CASE WHEN y < 0 THEN y ELSE z END) AS mn FROM t GROUP BY g",
                [(1, "a", -995.788, "hi")],
                [
                    ("INSERT INTO t VALUES (2, 'a', 5.0, 'zz')", ()),
                    ("UPDATE t SET y = -1000.0 WHERE id = 1", ()),
                    ("DELETE FROM t WHERE id = 1", ()),
                ],
            ),
            (
                "max_min_decimal_sum_real",
                "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x DECIMAL(10, 2), y DECIMAL(10, 2));",
                "SELECT g, MAX(x + y) AS mx, MIN(x + y) AS mn FROM t GROUP BY g",
                [(1, "a", 1.5, 2.5)],
                [
                    ("INSERT INTO t VALUES (2, 'a', 0.5, 0.5)", ()),
                    ("DELETE FROM t WHERE id = 2", ()),
                ],
            ),
            (
                "group_by_string_column_numeric_affinity",
                "CREATE TABLE t (id INTEGER PRIMARY KEY, s STRING, x INTEGER);",
                "SELECT s, COUNT(*) AS c, SUM(x) AS sx FROM t GROUP BY s",
                [(1, 1.5, 10)],
                [
                    ("INSERT INTO t VALUES (2, 1.5, 20)", ()),
                    ("INSERT INTO t VALUES (3, 'abc', 5)", ()),
                    ("UPDATE t SET s = 2.5 WHERE id = 2", ()),
                    ("DELETE FROM t WHERE id = 1", ()),
                ],
            ),
            (
                "count_and_sum_distinct_blob_text_neg_zero_vs_int_zero",
                "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, y BLOB);",
                "SELECT g, COUNT(DISTINCT y) AS dc, SUM(DISTINCT y) AS ds FROM t GROUP BY g",
                [(1, "a", "-0"), (2, "a", 0)],
                [
                    ("INSERT INTO t VALUES (3, 'a', 0.0)", ()),
                    ("DELETE FROM t WHERE id = 1", ()),
                    ("DELETE FROM t WHERE id = 2", ()),
                    ("INSERT INTO t VALUES (4, 'a', '-0')", ()),
                ],
            ),
        )
        for name, schema, view, seed_rows, steps in cases:
            with self.subTest(case=name):
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema)
                if seed_rows:
                    placeholders = ", ".join("?" for _ in seed_rows[0])
                    conn.executemany(f"INSERT INTO t VALUES ({placeholders})", seed_rows)
                _install(conn, "v", view, schema)
                self.assertViewExact(conn, "v", view, f"{name} initial")
                for idx, (sql, params) in enumerate(steps):
                    conn.execute(sql, params)
                    self.assertViewExact(conn, "v", view, f"{name} step {idx}: {sql}")
                conn.close()

    def test_numeric_group_key_equivalence_and_boundaries(self) -> None:
        # 1 == 1.0, 0 == 0.0 == -0.0, 2^53 == 2^53.0, -2^63 == -2^63.0 form one
        # group each; 2^53+1, 2^63-1 vs 2^63.0, '1', b'1', '-0', and NULL stay
        # distinct groups.
        for col_decl in ("BLOB", "BLOB NOT NULL"):
            with self.subTest(col_decl=col_decl):
                schema = f"CREATE TABLE t (id INTEGER PRIMARY KEY, g {col_decl}, x INTEGER);"
                view = "SELECT g, COUNT(*) AS c, SUM(x) AS sx FROM t GROUP BY g"
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema)
                _install(conn, "v", view, schema)

                # Equivalent pairs collapse into one group, and deleting one
                # refreshes the stored representative to the surviving class.
                steps: List[Tuple[str, Tuple[Any, ...]]] = [
                    ("INSERT INTO t VALUES (1, ?, 10)", (1,)),
                    ("INSERT INTO t VALUES (2, ?, 20)", (1.0,)),
                    ("DELETE FROM t WHERE id = 1", ()),
                    ("UPDATE t SET g = ? WHERE id = 2", (1,)),
                    ("UPDATE t SET g = ? WHERE id = 2", (1.0,)),
                    ("INSERT INTO t VALUES (3, ?, 3)", (0,)),
                    ("INSERT INTO t VALUES (4, ?, 4)", (-0.0,)),
                    ("DELETE FROM t WHERE id = 3", ()),
                    ("INSERT INTO t VALUES (5, ?, 5)", (2**53,)),
                    ("INSERT INTO t VALUES (6, ?, 6)", (float(2**53),)),
                    ("INSERT INTO t VALUES (7, ?, 7)", (2**53 + 1,)),
                    ("DELETE FROM t WHERE id = 5", ()),
                    ("INSERT INTO t VALUES (8, ?, 8)", (-(2**63),)),
                    ("INSERT INTO t VALUES (9, ?, 9)", (float(-(2**63)),)),
                    ("DELETE FROM t WHERE id = 8", ()),
                    ("INSERT INTO t VALUES (10, ?, 10)", (2**63 - 1,)),
                    ("INSERT INTO t VALUES (11, ?, 11)", (float(2**63),)),
                    ("INSERT INTO t VALUES (12, ?, 12)", ("1",)),
                    ("INSERT INTO t VALUES (13, ?, 13)", (b"1",)),
                    ("INSERT INTO t VALUES (14, ?, 14)", ("-0",)),
                ]
                if "NOT NULL" not in col_decl:
                    steps.append(("INSERT INTO t VALUES (15, ?, 15)", (None,)))
                for idx, (sql, params) in enumerate(steps):
                    conn.execute(sql, params)
                    self.assertViewExact(conn, "v", view, f"{col_decl} step {idx}: {sql} {params}")
                conn.close()

    def test_distinct_setops_join_union_all_and_computed_projections(self) -> None:
        # SELECT DISTINCT over BLOB 1 and 1.0
        schema_d = "CREATE TABLE t (id INTEGER PRIMARY KEY, g BLOB);"
        view_d = "SELECT DISTINCT g FROM t"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_d)
        _install(conn, "v", view_d, schema_d)
        conn.execute("INSERT INTO t VALUES (1, 1)")
        self.assertViewExact(conn, "v", view_d, "distinct ins 1")
        conn.execute("INSERT INTO t VALUES (2, 1.0)")
        self.assertEqual(len(conn.execute("SELECT * FROM v").fetchall()), 1)
        conn.execute("DELETE FROM t WHERE id = 1")
        self.assertViewExact(conn, "v", view_d, "distinct after del 1 -> 1.0")
        conn.close()

        # UNION mixed INT and BLOB + 1 vs 1.0
        schema_u = (
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY, g INTEGER);\n"
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY, g BLOB);"
        )
        view_u = "SELECT g FROM t1 UNION SELECT g FROM t2"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_u)
        _install(conn, "v", view_u, schema_u)
        conn.execute("INSERT INTO t1 VALUES (1, 10), (2, 1)")
        conn.execute("INSERT INTO t2 VALUES (1, 4000.0), (2, 1.0)")
        self.assertEqual(len(conn.execute("SELECT * FROM v").fetchall()), 3)
        conn.execute("DELETE FROM t1 WHERE id = 2")
        self.assertViewExact(conn, "v", view_u, "union after del t1(1) -> 1.0 and 4000.0")
        conn.close()

        # EXCEPT: branch 1 inserts 1.0 first, branch 0 inserts 1, then branch 1 deletes 1.0
        schema_e = (
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY, g BLOB);\n"
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY, g BLOB);"
        )
        view_e = "SELECT g FROM t1 EXCEPT SELECT g FROM t2"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_e)
        _install(conn, "v", view_e, schema_e)
        conn.execute("INSERT INTO t2 VALUES (1, 1.0)")
        conn.execute("INSERT INTO t1 VALUES (1, 1)")
        self.assertViewExact(conn, "v", view_e, "except hidden while matched")
        conn.execute("DELETE FROM t2 WHERE id = 1")
        self.assertViewExact(conn, "v", view_e, "except revealed with branch 0 storage class")
        conn.close()

        # INNER JOIN grouped MIN/MAX over BLOB and 1.0 -> 1 deletion
        schema_j = (
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g BLOB);\n"
            "CREATE TABLE c (id INTEGER PRIMARY KEY, pid INTEGER, x BLOB);"
        )
        view_j = "SELECT p.g AS g, MAX(c.x) AS mx, MIN(c.x) AS mn FROM c JOIN p ON c.pid = p.id GROUP BY p.g"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_j)
        _install(conn, "v", view_j, schema_j)
        conn.execute("INSERT INTO p VALUES (1, 1.0), (2, 1)")
        conn.execute("INSERT INTO c VALUES (1, 1, 4000.0), (2, 2, 3000.0)")
        self.assertEqual(len(conn.execute("SELECT * FROM v").fetchall()), 1)
        conn.execute("DELETE FROM c WHERE id = 1")
        self.assertViewExact(conn, "v", view_j, "join after del 1.0 -> int 1")
        conn.close()

        # UNION ALL grouped MAX over BLOB and 1 -> 1.0 deletion
        schema_ua = (
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY, g BLOB, x BLOB);\n"
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY, g BLOB, x BLOB);"
        )
        view_ua = (
            "SELECT g, MAX(x) AS mx, COUNT(*) AS cnt "
            "FROM (SELECT g, x FROM t1 UNION ALL SELECT g, x FROM t2) AS u GROUP BY g"
        )
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_ua)
        _install(conn, "v", view_ua, schema_ua)
        conn.execute("INSERT INTO t1 VALUES (1, 1, 4000.0)")
        conn.execute("INSERT INTO t2 VALUES (1, 1.0, 3000.0)")
        self.assertEqual(len(conn.execute("SELECT * FROM v").fetchall()), 1)
        conn.execute("DELETE FROM t1 WHERE id = 1")
        self.assertViewExact(conn, "v", view_ua, "union_all_agg after del 1 -> 1.0")
        conn.close()

        # Computed projection over DECIMAL and mixed CASE must not coerce storage class
        schema_p = (
            "CREATE TABLE t (id INTEGER PRIMARY KEY, flag INT NOT NULL, "
            "x DECIMAL(10, 2), y DECIMAL(10, 2), qty INT, rate REAL);"
        )
        view_p = (
            "SELECT id, x + y AS s, "
            "CASE WHEN flag = 1 THEN qty ELSE rate END AS chosen FROM t"
        )
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_p)
        _install(conn, "v", view_p, schema_p)
        conn.execute("INSERT INTO t VALUES (1, 1, 1.5, 2.5, 5, 1.25)")
        conn.execute("INSERT INTO t VALUES (2, 0, 2.0, 3.0, 5, 1.25)")
        self.assertViewExact(conn, "v", view_p, "computed projection exact storage classes")
        conn.close()

    def test_failed_initial_population_overflow_rolls_back_cleanly(self) -> None:
        from matview import MaterializedViewManager

        schema = "CREATE TABLE t (id INTEGER PRIMARY KEY, x BIGINT);"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        conn.execute("INSERT INTO t VALUES (1, ?), (2, ?)", (2**62, 2**62))
        conn.commit()
        mgr = MaterializedViewManager(conn)
        with self.assertRaisesRegex(sqlite3.OperationalError, "integer overflow"):
            mgr.register_view("CREATE VIEW v_ovf AS SELECT SUM(x) AS s FROM t;")
        objs = conn.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE name LIKE '%v_ovf%' OR name LIKE '__matview_trg_%' OR name LIKE '__matview_store_%'"
        ).fetchall()
        self.assertEqual(objs, [], f"failed registration left objects behind: {objs}")
        self.assertEqual(mgr.get_catalog(), {})
        # Subsequent writes to base table must still succeed without stray triggers.
        conn.execute("INSERT INTO t VALUES (3, 1)")
        conn.close()

    def test_adversarial_sqlite_edge_cases(self) -> None:
        # 1. Multiline UNION ALL aggregate with inner WHERE deleting a filtered-out row
        schema_ml = (
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY, g BLOB, x INT);\n"
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY, g BLOB, x INT);"
        )
        view_ml = (
            "SELECT g, COUNT(*) AS cnt FROM (\n"
            "    SELECT g, x FROM t1\n"
            "    WHERE x > 0\n"
            "    UNION ALL\n"
            "    SELECT g, x FROM t2\n"
            "    WHERE x > 0\n"
            ") AS u GROUP BY g"
        )
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_ml)
        _install(conn, "v", view_ml, schema_ml)
        conn.execute("INSERT INTO t1 VALUES (1, 1.0, 10), (2, 1.0, -5)")
        self.assertViewExact(conn, "v", view_ml, "multiline union_all insert")
        conn.execute("DELETE FROM t1 WHERE id = 2")
        self.assertViewExact(conn, "v", view_ml, "multiline union_all delete filtered-out row")
        conn.close()

        # 2. UNION ALL aggregate over multiple branches of the same table with NOT NULL group key
        schema_sb = "CREATE TABLE t1 (id INTEGER PRIMARY KEY, x BLOB NOT NULL, y BLOB NOT NULL);"
        view_sb = (
            "SELECT g, COUNT(*) AS cnt "
            "FROM (SELECT x AS g FROM t1 UNION ALL SELECT y AS g FROM t1) AS u GROUP BY g"
        )
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_sb)
        _install(conn, "v", view_sb, schema_sb)
        conn.execute("INSERT INTO t1 VALUES (1, 1, 1.0)")
        self.assertEqual(len(conn.execute("SELECT * FROM v").fetchall()), 1)
        conn.execute("DELETE FROM t1 WHERE id = 1")
        self.assertViewExact(conn, "v", view_sb, "same-table multi-branch delete last row")
        conn.close()

        # 3. -0.0 <-> 0.0 UPDATE triggers must fire across projection, MIN/MAX, DISTINCT, GROUP BY, UNION
        for label, view_z in (
            ("projection", "SELECT id, x FROM t"),
            ("minmax", "SELECT MIN(x) AS mn, MAX(x) AS mx FROM t"),
            ("distinct", "SELECT DISTINCT x FROM t"),
            ("group_by", "SELECT x, COUNT(*) AS c FROM t GROUP BY x"),
        ):
            with self.subTest(signed_zero_view=label):
                schema_z = "CREATE TABLE t (id INTEGER PRIMARY KEY, x BLOB);"
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema_z)
                _install(conn, "v", view_z, schema_z)
                conn.execute("INSERT INTO t VALUES (1, -0.0)")
                self.assertViewExact(conn, "v", view_z, f"{label} after insert -0.0")
                conn.execute("UPDATE t SET x = 0.0 WHERE id = 1")
                self.assertViewExact(conn, "v", view_z, f"{label} after update -0.0 -> +0.0")
                conn.execute("UPDATE t SET x = -0.0 WHERE id = 1")
                self.assertViewExact(conn, "v", view_z, f"{label} after update +0.0 -> -0.0")
                conn.close()

        # 4. INTERSECT insert order & EXCEPT cnt0: 1 -> 0 -> 1 & UNION aliased branches
        schema_set = (
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY, g BLOB);\n"
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY, g BLOB);"
        )
        view_int = "SELECT g FROM t1 INTERSECT SELECT g FROM t2"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_set)
        _install(conn, "v", view_int, schema_set)
        conn.execute("INSERT INTO t2 VALUES (1, 1.0)")
        conn.execute("INSERT INTO t1 VALUES (1, 1)")
        self.assertViewExact(conn, "v", view_int, "intersect t2(1.0) before t1(1)")
        conn.execute("UPDATE t1 SET g = 1.0 WHERE id = 1")
        self.assertViewExact(conn, "v", view_int, "intersect t1 updated 1 -> 1.0")
        conn.close()

        view_exc = "SELECT g FROM t1 EXCEPT SELECT g FROM t2"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_set)
        _install(conn, "v", view_exc, schema_set)
        conn.execute("INSERT INTO t1 VALUES (1, 1)")
        conn.execute("INSERT INTO t2 VALUES (1, 1.0)")
        conn.execute("DELETE FROM t1 WHERE id = 1")
        conn.execute("INSERT INTO t1 VALUES (2, 1.0)")
        conn.execute("DELETE FROM t2 WHERE id = 1")
        self.assertViewExact(conn, "v", view_exc, "except cnt0 1->0->1 reveals 1.0")
        conn.close()

        view_ualias = "SELECT a.g AS g FROM t1 AS a UNION SELECT b.g AS g FROM t2 AS b"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_set)
        _install(conn, "v", view_ualias, schema_set)
        conn.execute("INSERT INTO t1 VALUES (1, 1)")
        conn.execute("INSERT INTO t2 VALUES (1, 1.0)")
        conn.execute("DELETE FROM t1 WHERE id = 1")
        self.assertViewExact(conn, "v", view_ualias, "union aliased branches delete 1 -> 1.0")
        conn.close()

        # 5. Compound expression starting/ending with quotes in UNION must still canonicalize
        view_qexpr = "SELECT '1' + '0.0' AS g FROM t1 UNION SELECT g FROM t2"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_set)
        _install(conn, "v", view_qexpr, schema_set)
        conn.execute("INSERT INTO t1 VALUES (1, 99)")
        conn.execute("INSERT INTO t2 VALUES (1, 1)")
        self.assertEqual(len(conn.execute("SELECT * FROM v").fetchall()), 1)
        conn.close()

        # 6. Tier 2 compound aggregates and Tier 3 set-op / CTE storage-class preservation
        schema_t2 = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, x BLOB);"
        view_t2 = (
            "SELECT g, COUNT(*) + 1 AS cp1, COALESCE(MAX(x), 0) AS mx "
            "FROM t GROUP BY g ORDER BY g"
        )
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_t2)
        _install(conn, "v", view_t2, schema_t2)
        conn.execute("INSERT INTO t VALUES (1, 'a', 4000.0)")
        self.assertViewExact(conn, "v", view_t2, "tier3 compound aggregate exact types")
        conn.close()

        schema_t3 = (
            "CREATE TABLE t1 (id INTEGER PRIMARY KEY, g INTEGER);\n"
            "CREATE TABLE t2 (id INTEGER PRIMARY KEY, g BLOB);"
        )
        view_t3 = "SELECT g FROM t1 UNION SELECT g FROM t2 ORDER BY g"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_t3)
        _install(conn, "v", view_t3, schema_t3)
        conn.execute("INSERT INTO t1 VALUES (1, 10)")
        conn.execute("INSERT INTO t2 VALUES (1, 4000.0)")
        self.assertViewExact(conn, "v", view_t3, "tier3 union order by preserves float 4000.0")
        conn.close()

        schema_cte = "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER);"
        view_cte = (
            "WITH c AS (SELECT id, x + 1.0 AS x FROM t) "
            "SELECT id, x FROM c ORDER BY id"
        )
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_cte)
        _install(conn, "v", view_cte, schema_cte)
        conn.execute("INSERT INTO t VALUES (1, 1)")
        self.assertViewExact(conn, "v", view_cte, "tier3 cte shadowing base column preserves float")
        conn.close()

        # 7. WHERE membership toggle + incremental_backfill=True with 1 and 1.0
        schema_w = "CREATE TABLE t (id INTEGER PRIMARY KEY, g BLOB, flag INT NOT NULL, x INT);"
        view_w = "SELECT g, COUNT(*) AS c, SUM(x) AS sx FROM t WHERE flag > 0 GROUP BY g"
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_w)
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?)",
            [(1, 1, 1, 10), (2, 1.0, 1, 20), (3, 1.0, 0, 30)],
        )
        _install(conn, "v", view_w, schema_w, incremental_backfill=True, batch_size=1)
        self.assertEqual(len(conn.execute("SELECT * FROM v").fetchall()), 1)
        conn.execute("UPDATE t SET flag = 0 WHERE id = 1")
        self.assertViewExact(conn, "v", view_w, "where exit of int 1 refreshes to float 1.0")
        conn.execute("UPDATE t SET flag = 1 WHERE id = 3")
        self.assertViewExact(conn, "v", view_w, "where entry of id=3")
        conn.close()


if __name__ == "__main__":
    unittest.main()
