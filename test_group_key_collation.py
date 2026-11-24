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
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from matview import MaterializedViewManager
from matview_sync import Dialect, UnsupportedMaterializedViewError

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
        schema = (
            "CREATE TABLE d (k TEXT COLLATE NOCASE PRIMARY KEY, w INTEGER);"
            "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE NOCASE, x INTEGER, k TEXT COLLATE NOCASE);"
        )
        views = {
            "j": "SELECT t.g AS g, COUNT(*) AS c, SUM(d.w) AS s FROM t JOIN d ON t.k = d.k GROUP BY t.g",
            "hv": "SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING COUNT(*) > 1",
            "wh": "SELECT g, SUM(x) AS s FROM t WHERE x > 2 GROUP BY g",
            "gc": "SELECT g, COUNT(DISTINCT x) AS cd, GROUP_CONCAT(DISTINCT x) AS gc FROM t GROUP BY g",
            "mm": "SELECT g, MIN(x) AS mn, MAX(x) AS mx FROM t GROUP BY g",
            "cte": "WITH c AS (SELECT g, x FROM t) SELECT g, SUM(x) AS s FROM c GROUP BY g",
            "st": "SELECT g, AVG(x) AS a, COUNT(DISTINCT x) AS cd FROM t GROUP BY g",
            "nn": "SELECT g, COUNT(*) AS c FROM t WHERE g IS NOT NULL GROUP BY g",
        }
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
                if k < 0.5 or not ids:
                    nid += 1
                    stmt = ("INSERT INTO t VALUES (?, ?, ?, ?)", (nid, rng.choice(spell), rng.randint(0, 5), rng.choice(["a", "A", "b", "B"])))
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
        finally:
            pair.close()

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

    def test_unreproducible_collation_is_refused_explicitly(self) -> None:
        con = sqlite3.connect(":memory:")
        try:
            con.create_collation("REVERSE", lambda a, b: (a < b) - (a > b))
            con.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT COLLATE REVERSE, x INTEGER);")
            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            with self.assertRaises(UnsupportedMaterializedViewError):
                mgr.register_view("CREATE VIEW v AS SELECT g, COUNT(*) AS c FROM t GROUP BY g;")
        finally:
            con.close()


def _mysql_dsn() -> Optional[str]:
    return os.environ.get("MYSQL_DSN") or None


@unittest.skipUnless(_mysql_dsn(), "MYSQL_DSN not set (native MySQL/MariaDB only)")
class TestMysqlGroupKeyCollation(unittest.TestCase):
    """Native MySQL/MariaDB: groups merge exactly where the server's collation
    says two values are equal (case/accent-insensitive, PAD SPACE and NO PAD).

    The oracle is the server itself: the number of materialized rows equals the
    number of groups of the original view, and every materialized row (key
    spelling, aggregates) equals what the original query computes for the rows
    whose key compares equal to that spelling under the column's collation."""

    # (collation, spellings of one logical key)
    MYSQL = (
        ("utf8mb4_0900_ai_ci", ("a", "A", "\u00e1", "a ")),
        ("utf8mb4_0900_as_cs", ("a", "A", "\u00e1")),
        ("utf8mb4_general_ci", ("a", "A", "a ", "a  ")),
        ("utf8mb4_bin", ("a", "A", "a ")),
        ("utf8mb4_0900_bin", ("a", "A", "a ")),
    )
    MARIADB = (
        ("utf8mb4_uca1400_ai_ci", ("a", "A", "\u00e1", "a ")),
        ("utf8mb4_uca1400_as_cs", ("a", "A", "\u00e1")),
        ("utf8mb4_general_ci", ("a", "A", "a ", "a  ")),
        ("utf8mb4_bin", ("a", "A", "a ")),
        ("utf8mb4_nopad_bin", ("a", "A", "a ")),
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

    def _run_collation(self, coll: str, spellings: Tuple[str, ...]) -> None:
        conn = self._connect()
        cur = conn.cursor()
        tbl = f"gk_t_{re.sub('[^a-z0-9]', '_', coll)}"
        view = f"gk_v_{re.sub('[^a-z0-9]', '_', coll)}"
        cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        cur.execute(
            f"CREATE TABLE {tbl} (id INT PRIMARY KEY, g VARCHAR(20) CHARACTER SET utf8mb4 COLLATE {coll}, "
            f"h VARCHAR(20) CHARACTER SET utf8mb4 COLLATE {coll}, x INT) ENGINE=InnoDB"
        )
        conn.commit()
        mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True)
        views = {
            view: f"SELECT g, COUNT(*) AS c, SUM(x) AS s, MIN(x) AS mn FROM {tbl} GROUP BY g",
            f"{view}_m": f"SELECT g, h, COUNT(*) AS c, SUM(x) AS s FROM {tbl} GROUP BY g, h",
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for name, sel in views.items():
                    mgr.register_view(f"CREATE VIEW {name} AS {sel};")

            def check(step: str) -> None:
                for name, sel in views.items():
                    key_cols = ["g"] if name == view else ["g", "h"]
                    cur.execute(sel)
                    want = cur.fetchall()
                    cur.execute(f"SELECT * FROM {name}")
                    got = cur.fetchall()
                    self.assertEqual(len(got), len(want), f"{coll} {step}: {name} group count\n  MV {got}\n  LIVE {want}")
                    for row in got:
                        keys = row[: len(key_cols)]
                        cond = " AND ".join(f"{k} <=> %s" for k in key_cols)
                        agg = "COUNT(*), SUM(x), MIN(x)" if name == view else "COUNT(*), SUM(x)"
                        cur.execute(f"SELECT {agg} FROM {tbl} WHERE {cond}", keys)
                        exp = cur.fetchone()
                        self.assertEqual(tuple(row[len(key_cols):]), tuple(exp), f"{coll} {step}: {name} row {row}")

            check("empty")
            nid = 0
            for sp in spellings:
                nid += 1
                cur.execute(f"INSERT INTO {tbl} VALUES (%s, %s, %s, %s)", (nid, sp, spellings[0], nid))
                conn.commit()
                check(f"insert {sp!r}")
            cur.execute(f"INSERT INTO {tbl} VALUES (100, 'zzz', 'q', 9)")
            conn.commit()
            check("other key")
            cur.execute(f"UPDATE {tbl} SET g = %s WHERE id = 1", (spellings[1],))
            conn.commit()
            check("update spelling only")
            cur.execute(f"UPDATE {tbl} SET g = %s, x = x + 5 WHERE id = 2", (spellings[0],))
            conn.commit()
            check("update spelling and value")
            cur.execute(f"DELETE FROM {tbl} WHERE id = 1")
            conn.commit()
            check("delete one spelling")
            cur.execute(f"UPDATE {tbl} SET g = %s WHERE id = 100", (spellings[-1],))
            conn.commit()
            check("move other key into the group")
            cur.execute(f"DELETE FROM {tbl} WHERE id = 2")
            cur.execute(f"INSERT INTO {tbl} VALUES (200, %s, 'q', 1)", (spellings[0],))
            conn.rollback()
            check("rolled back")
            cur.execute(f"DELETE FROM {tbl}")
            conn.commit()
            check("all deleted")
        finally:
            conn.rollback()
            for name in views:
                try:
                    mgr.drop_view(name)
                except Exception:
                    pass
            cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            conn.commit()
            conn.close()

    def test_group_merge_follows_the_column_collation(self) -> None:
        for coll, spellings in self.MARIADB if self.is_mariadb else self.MYSQL:
            with self.subTest(collation=coll):
                self._run_collation(coll, spellings)


if __name__ == "__main__":
    unittest.main()
