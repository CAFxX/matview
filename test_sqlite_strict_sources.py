#!/usr/bin/env python3
"""SQLite STRICT source tables are supported and exact (differential, no tolerance).

Contract: for every view class below, the materialized view returns exactly what
the original view returns over the current base rows (same rows with their
multiplicities, same values, same storage class per cell, bit-identical floats)
after initial population and after INSERT / UPDATE / DELETE / UPSERT / REPLACE /
bulk / ROLLBACK transitions, whether the source tables are plain or STRICT.
The same schema factory builds both variants (`strict` flag) and every view is
checked against both, so a STRICT-only divergence cannot hide.

STRICT additionally *rejects* wrong-typed values (a statement error that aborts the
user's DML): the rejection must leave the view equal to the live view too.
"""

from __future__ import annotations

import math
import sqlite3
import unittest
from collections import Counter
from typing import Any, List, Sequence, Tuple

from matview import Dialect, MaterializedViewManager

STRICT_OK = sqlite3.sqlite_version_info >= (3, 37, 0)


def schema(strict: bool) -> List[str]:
    """One schema factory for both variants (ANY has no pre-3.37 spelling: BLOB affinity is its twin)."""
    any_t = "ANY" if strict else "BLOB"
    tail = " STRICT" if strict else ""
    return [
        f"CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, v INTEGER, f REAL, x {any_t}, b BLOB){tail}",
        f"CREATE TABLE d (g TEXT NOT NULL PRIMARY KEY, name TEXT NOT NULL, w INTEGER) WITHOUT ROWID{', STRICT' if strict else ''}",
        f"CREATE TABLE u (a INTEGER NOT NULL, b TEXT NOT NULL, n INTEGER, PRIMARY KEY (a, b)){tail}",
    ]


SEED = [
    "INSERT INTO d VALUES ('a','Alpha',1),('b','Beta',2),('c','Gamma',NULL)",
    "INSERT INTO t VALUES (1,'a',10,0.1,5,x'01'),(2,'a',20,0.2,'five',NULL),(3,'b',NULL,0.30000000000000004,2.5,x'02'),"
    "(4,'b',40,-0.0,NULL,x'00'),(5,NULL,9007199254740993,1e300,1,NULL),(6,'c',-7,NULL,x'ff',x'ff')",
    "INSERT INTO u VALUES (1,'p',1),(1,'q',NULL),(2,'p',3)",
]

# (name, select) per supported view class; shapes mirror the classes the README lists.
VIEWS: List[Tuple[str, str]] = [
    ("agg", "SELECT g, COUNT(*) AS c, COUNT(v) AS cv, SUM(v) AS sv, SUM(f) AS sf, AVG(v) AS av, MIN(f) AS mn, MAX(v) AS mx FROM t GROUP BY g"),
    ("agg_total", "SELECT COUNT(*) AS c, SUM(v) AS s, TOTAL(f) AS tf, MAX(g) AS mg FROM t"),
    ("agg_any", "SELECT g, MIN(x) AS mnx, MAX(x) AS mxx, COUNT(x) AS cx FROM t GROUP BY g"),
    ("join", "SELECT t.id, t.g, d.name, t.v + COALESCE(d.w, 0) AS vw FROM t JOIN d ON d.g = t.g"),
    ("join_agg", "SELECT d.name, COUNT(*) AS c, SUM(t.v) AS s FROM t JOIN d ON d.g = t.g GROUP BY d.name"),
    ("left_join", "SELECT t.id, d.name FROM t LEFT JOIN d ON d.g = t.g"),
    ("window", "SELECT id, g, v, SUM(v) OVER (PARTITION BY g) AS sg, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) AS rn FROM t"),
    ("distinct", "SELECT DISTINCT g, v IS NULL AS vn FROM t"),
    ("union_all", "SELECT g AS k, v AS n FROM t UNION ALL SELECT b, n FROM u"),
    ("union", "SELECT g AS k FROM t UNION SELECT name FROM d"),
    ("intersect", "SELECT g FROM t INTERSECT SELECT g FROM d"),
    ("except", "SELECT g FROM d EXCEPT SELECT g FROM t WHERE v IS NOT NULL"),
    ("expr", "SELECT id, CASE WHEN v IS NULL THEN 'n' WHEN v > 15 THEN 'big' ELSE 'small' END AS k, COALESCE(v, -1) AS cv, typeof(x) AS tx, x, f * 2 AS f2 FROM t"),
    ("composite", "SELECT a, COUNT(*) AS c, SUM(n) AS s FROM u GROUP BY a"),
]

# (label, statements, expect_error)
DML: List[Tuple[str, Sequence[str], bool]] = [
    ("insert", ["INSERT INTO t VALUES (7,'a',5,1.5,'s',x'0a')", "INSERT INTO u VALUES (3,'z',9)"], False),
    ("update_group", ["UPDATE t SET g = 'c' WHERE id IN (1,2)", "UPDATE u SET n = n + 1 WHERE a = 1"], False),
    ("update_value", ["UPDATE t SET v = v + 1, f = f + 0.1 WHERE v IS NOT NULL AND v < 100"], False),
    ("upsert", ["INSERT INTO t(id,g,v) VALUES (1,'a',99) ON CONFLICT(id) DO UPDATE SET v = excluded.v, g = 'b'",
                "INSERT INTO d VALUES ('a','Alpha2',7) ON CONFLICT(g) DO UPDATE SET name = excluded.name"], False),
    ("replace", ["INSERT OR REPLACE INTO t VALUES (2,'b',1,1.0,x'00',NULL)", "INSERT OR REPLACE INTO u VALUES (1,'p',100)"], False),
    # numeric text is converted losslessly by both variants (STRICT stores 10, never '10')
    ("numeric_text", ["INSERT INTO t VALUES (70,'a','10','2.5',NULL,NULL)"], False),
    ("bulk_insert", ["INSERT INTO t SELECT id + 100, g, v, f, x, b FROM t"], False),
    ("bulk_update", ["UPDATE t SET v = NULL, x = 3"], False),
    ("delete", ["DELETE FROM t WHERE id = 3", "DELETE FROM d WHERE g = 'c'"], False),
    ("bulk_delete", ["DELETE FROM t", "DELETE FROM u"], False),
    ("reinsert", ["INSERT INTO t SELECT id, g, v, f, x, b FROM t WHERE 0", "INSERT INTO t VALUES (50,'a',1,2.0,NULL,NULL)"], False),
]
# STRICT rejects these (non-numeric text / lossy REAL into INTEGER / text into REAL / NULL into a NOT NULL key):
# the rejected statement must not leave the view inconsistent. On a plain table they are accepted
# (affinity), so they are only exercised against the STRICT variant.
STRICT_REJECTED = [
    "INSERT INTO t VALUES (60,'a','10x',1.0,NULL,NULL)",
    "INSERT INTO t VALUES (61,'a',1,'abc',NULL,NULL)",
    "UPDATE t SET v = 1.5",
    "INSERT INTO u VALUES (9, NULL, 1)",
]


def _cell(v: Any) -> Tuple[Any, ...]:
    if isinstance(v, float):
        return ("float", "nan" if math.isnan(v) else v.hex())  # hex(): bit-exact incl. signed zero
    return (type(v).__name__, v)


def _rows(conn: sqlite3.Connection, sql: str) -> Counter:
    return Counter(tuple(_cell(c) for c in r) for r in conn.execute(sql).fetchall())


class SqliteStrictSourceTests(unittest.TestCase):
    def _run(self, strict: bool) -> Tuple[int, int]:
        conn = sqlite3.connect(":memory:")
        try:
            for stmt in schema(strict) + SEED:
                conn.execute(stmt)
            conn.commit()
            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            plans = {}
            for name, select in VIEWS:
                plans[name] = mgr.register_view(f"CREATE VIEW sv_{name} AS {select};")
            lives = dict(VIEWS)

            def check(stage: str) -> None:
                for name, select in lives.items():
                    got = _rows(conn, f"SELECT * FROM {plans[name].db_view_name}")
                    want = _rows(conn, select)
                    self.assertEqual(got, want, f"strict={strict} view={name} stage={stage}")

            check("initial")
            transitions = 1
            for label, stmts, _ in DML:
                # a rolled-back transaction leaves view == live view (state before the change),
                # also checked inside the open transaction (maintenance sees uncommitted rows)
                conn.execute("BEGIN")
                for s in stmts:
                    conn.execute(s)
                check(label + "+in-txn")
                conn.rollback()
                check(label + "+rollback")
                conn.execute("BEGIN")
                for s in stmts:
                    conn.execute(s)
                conn.commit()
                check(label)
                transitions += 3
            if strict:
                for bad in STRICT_REJECTED:
                    with self.assertRaises(sqlite3.IntegrityError, msg=bad):
                        conn.execute(bad)
                    conn.rollback()
                    check("rejected: " + bad)
                    transitions += 1
            return len(VIEWS), transitions
        finally:
            conn.close()

    @unittest.skipUnless(STRICT_OK, "STRICT tables need SQLite >= 3.37")
    def test_strict_and_plain_sources_match_live_views_exactly(self) -> None:
        for strict in (False, True):
            views, transitions = self._run(strict)
            self.assertEqual(views, len(VIEWS))
            self.assertGreaterEqual(transitions, 21)

    def test_strict_detected_from_ddl_and_live_connection(self) -> None:
        from matview.sql_frontend import parse_table_schema

        ddl = ";\n".join(schema(True)) + ";"
        parsed = parse_table_schema(ddl, dialect=Dialect.SQLITE)
        self.assertEqual({n: ts.strict for n, ts in parsed.items()}, {"t": True, "d": True, "u": True})
        self.assertTrue(parsed["d"].without_rowid)
        plain = parse_table_schema(";\n".join(schema(False)) + ";", dialect=Dialect.SQLITE)
        self.assertEqual({n: ts.strict for n, ts in plain.items()}, {"t": False, "d": False, "u": False})


if __name__ == "__main__":
    unittest.main()
