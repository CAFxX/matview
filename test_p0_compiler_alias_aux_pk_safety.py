#!/usr/bin/env python3
"""Regression tests for reserved aggregate alias quoting, distinct aux types, and SQLite nullable PK projections."""

from __future__ import annotations

import re
import sqlite3
import unittest
from typing import List

from test_matview_sync import FLOAT_OPT_IN
from matview_sync import (
    Dialect,
    MaterializedViewManager,
    generate_materialized_view,
)


def _assert_matches_live(
    tc: unittest.TestCase, conn: sqlite3.Connection, view_name: str, live_sql: str
) -> None:
    actual = sorted(
        conn.execute(f"SELECT * FROM {view_name};").fetchall(),
        key=lambda r: tuple("" if x is None else repr(x) for x in r),
    )
    expected = sorted(
        conn.execute(live_sql).fetchall(),
        key=lambda r: tuple("" if x is None else repr(x) for x in r),
    )
    tc.assertEqual(actual, expected)


class TestCompilerAliasAuxPkSafety(unittest.TestCase):
    def test_reserved_keyword_agg_aliases_not_duplicated_single_table(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER);"
        )
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?);",
            [
                (1, "a", 10),
                (2, "a", 20),
                (3, "b", None),
                (4, "b", 5),
                (5, "c", 100),
            ],
        )
        conn.commit()

        mgr = MaterializedViewManager(conn)
        live_sql = (
            'SELECT grp, COUNT(val) AS "order", SUM(val) AS "group", '
            'AVG(val) AS "select" FROM t GROUP BY grp'
        )
        plan = mgr.register_view(f"CREATE VIEW mv_kw AS {live_sql};", populate=False)
        for stmt in [*plan.initial_populate_sql, *plan.incremental_backfill_spec.chunk_sql_templates]:
            m = re.search(r"INSERT INTO\s+\S+\s*\(([^)]+)\)", stmt, re.IGNORECASE)
            if m:
                cols = [c.strip().strip('"').strip("`").lower() for c in m.group(1).split(",")]
                self.assertEqual(
                    len(cols),
                    len(set(cols)),
                    f"duplicate target column in SQL:\n{stmt}",
                )

        mgr.backfill_incrementally(plan, batch_size=2)
        _assert_matches_live(self, conn, "mv_kw", live_sql)

        conn.execute("INSERT INTO t VALUES (6, 'a', 30);")
        conn.execute("UPDATE t SET val = 15 WHERE id = 3;")
        conn.execute("DELETE FROM t WHERE id = 1;")
        conn.commit()
        _assert_matches_live(self, conn, "mv_kw", live_sql)

        # Also check MySQL and Postgres codegen for reserved keyword aliases
        schema_sql = "CREATE TABLE t (id INTEGER PRIMARY KEY, grp VARCHAR(32) NOT NULL, val INTEGER);"
        view_sql = f"CREATE VIEW mv_kw AS {live_sql};"
        for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
            d_plan = generate_materialized_view(
                schema_sql, view_sql, dialect=dialect, ignore_limitations=(dialect == Dialect.MYSQL), accept_limitations=FLOAT_OPT_IN
            )
            for stmt in [
                *d_plan.initial_populate_sql,
                *d_plan.incremental_backfill_spec.chunk_sql_templates,
            ]:
                m = re.search(r"INSERT INTO\s+\S+\s*\(([^)]+)\)", stmt, re.IGNORECASE)
                if m:
                    cols = [c.strip().strip('"').strip("`").lower() for c in m.group(1).split(",")]
                    self.assertEqual(
                        len(cols),
                        len(set(cols)),
                        f"[{dialect.name}] duplicate target column in SQL:\n{stmt}",
                    )
        conn.close()

    def test_reserved_keyword_agg_aliases_join_and_union_all(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE p (id INTEGER PRIMARY KEY, cat TEXT NOT NULL);")
        conn.execute("CREATE TABLE c1 (id INTEGER PRIMARY KEY, p_id INTEGER NOT NULL, amt INTEGER);")
        conn.execute("CREATE TABLE c2 (id INTEGER PRIMARY KEY, p_id INTEGER NOT NULL, amt INTEGER);")
        conn.executemany("INSERT INTO p VALUES (?, ?);", [(1, "x"), (2, "y")])
        conn.executemany(
            "INSERT INTO c1 VALUES (?, ?, ?);",
            [(1, 1, 10), (2, 1, 20), (3, 2, 5)],
        )
        conn.executemany(
            "INSERT INTO c2 VALUES (?, ?, ?);",
            [(1, 1, 7), (2, 2, 13)],
        )
        conn.commit()

        mgr = MaterializedViewManager(conn)
        join_live = (
            'SELECT p.cat AS cat, COUNT(c1.amt) AS "order", SUM(c1.amt) AS "group" '
            "FROM p INNER JOIN c1 ON c1.p_id = p.id GROUP BY p.cat"
        )
        mgr.register_view(f"CREATE VIEW mv_join_kw AS {join_live};", populate=True)
        _assert_matches_live(self, conn, "mv_join_kw", join_live)
        conn.execute("INSERT INTO c1 VALUES (4, 2, 15);")
        conn.execute("DELETE FROM c1 WHERE id = 1;")
        conn.commit()
        _assert_matches_live(self, conn, "mv_join_kw", join_live)

        uall_live = (
            'SELECT p_id, COUNT(amt) AS "order", SUM(amt) AS "group" FROM ('
            "SELECT p_id, amt FROM c1 UNION ALL SELECT p_id, amt FROM c2"
            ") AS u GROUP BY p_id"
        )
        mgr.register_view(f"CREATE VIEW mv_uall_kw AS {uall_live};", populate=True)
        _assert_matches_live(self, conn, "mv_uall_kw", uall_live)
        conn.execute("INSERT INTO c2 VALUES (3, 1, 100);")
        conn.execute("DELETE FROM c1 WHERE id = 2;")
        conn.commit()
        _assert_matches_live(self, conn, "mv_uall_kw", uall_live)
        conn.close()

    def test_distinct_aux_table_infers_expr_storage_type(self) -> None:
        schema_sql = (
            "CREATE TABLE events (id INTEGER PRIMARY KEY, grp INTEGER NOT NULL, val INTEGER, score REAL);"
        )
        view_sql = (
            "CREATE VIEW mv_dist_expr AS "
            "SELECT (grp + 1) AS g1, "
            "COUNT(DISTINCT (val + 10)) AS dc_int, "
            "SUM(DISTINCT (score * 1.5)) AS ds_float "
            "FROM events GROUP BY (grp + 1);"
        )
        for dialect, expected_int, expected_float in (
            # SQLite stores DISTINCT aux values without affinity (BLOB) so
            # runtime storage classes (e.g. TEXT '-0' vs INTEGER 0) are never
            # coerced on insert.
            (Dialect.SQLITE, "BLOB", "BLOB"),
            # PostgreSQL integer + integer is integer (int4), the native type
            # of the DISTINCT value; other engines' integer arithmetic is 64-bit.
            (Dialect.POSTGRES, "INTEGER", "DOUBLE PRECISION"),
            (Dialect.MYSQL, "BIGINT", "DOUBLE"),
        ):
            plan = generate_materialized_view(
                schema_sql, view_sql, dialect=dialect, ignore_limitations=(dialect == Dialect.MYSQL), accept_limitations=FLOAT_OPT_IN
            )
            aux_ddls = [d for d in plan.auxiliary_table_ddls if "dist_" in d]
            self.assertEqual(len(aux_ddls), 2, f"expected 2 distinct aux tables: {plan.auxiliary_table_ddls}")
            int_aux = next(d for d in aux_ddls if "dc_int" in d)
            flt_aux = next(d for d in aux_ddls if "ds_float" in d)
            self.assertIn(f"__val {expected_int} NOT NULL", int_aux)
            self.assertNotIn("__val TEXT NOT NULL", int_aux)
            self.assertIn(f"__val {expected_float} NOT NULL", flt_aux)
            self.assertNotIn("__val TEXT NOT NULL", flt_aux)

        # Native execution end-to-end
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY, grp INTEGER NOT NULL, val INTEGER, score REAL);"
        )
        conn.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?);",
            [(1, 1, 9, 2.0), (2, 1, 9, 2.0), (3, 1, 10, 4.0)],
        )
        conn.commit()
        mgr = MaterializedViewManager(conn)
        live_sql = (
            "SELECT (grp + 1) AS g1, COUNT(DISTINCT (val + 10)) AS dc_int, "
            "SUM(DISTINCT (score * 1.5)) AS ds_float FROM events GROUP BY (grp + 1)"
        )
        mgr.register_view(f"CREATE VIEW mv_dist_expr AS {live_sql};", populate=True)
        _assert_matches_live(self, conn, "mv_dist_expr", live_sql)
        conn.execute("INSERT INTO events VALUES (4, 1, 10, 2.0);")
        conn.execute("DELETE FROM events WHERE id = 1;")
        conn.commit()
        _assert_matches_live(self, conn, "mv_dist_expr", live_sql)
        conn.close()

    def test_sqlite_nullable_non_rowid_and_composite_pks_avoid_native_pk(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t_int_pk (id INT PRIMARY KEY, grp TEXT, val INTEGER);")
        conn.execute(
            "CREATE TABLE t_comp_pk (k1 INTEGER, k2 INT, val TEXT, PRIMARY KEY (k1, k2));"
        )
        conn.execute(
            "CREATE TABLE t_rowid_pk (id INTEGER PRIMARY KEY, val TEXT);"
        )
        conn.executemany(
            "INSERT INTO t_int_pk VALUES (?, ?, ?);",
            [(None, "g1", 10), (None, "g2", 20), (1, "g1", 30)],
        )
        conn.executemany(
            "INSERT INTO t_comp_pk VALUES (?, ?, ?);",
            [(None, 1, "a"), (1, None, "b"), (None, None, "c"), (1, 1, "d")],
        )
        conn.executemany("INSERT INTO t_rowid_pk VALUES (?, ?);", [(1, "ok"), (2, "good")])
        conn.commit()

        mgr = MaterializedViewManager(conn)

        plan_int = mgr.register_view(
            "CREATE VIEW mv_int_pk AS SELECT id, grp, val FROM t_int_pk;", populate=True
        )
        self.assertIn("__mv_pk", plan_int.backing_table_ddl)
        _assert_matches_live(self, conn, "mv_int_pk", "SELECT id, grp, val FROM t_int_pk")
        conn.execute("INSERT INTO t_int_pk VALUES (NULL, 'g3', 40);")
        conn.execute("UPDATE t_int_pk SET val = 99 WHERE grp = 'g1' AND id IS NULL;")
        conn.execute("DELETE FROM t_int_pk WHERE grp = 'g2';")
        conn.commit()
        _assert_matches_live(self, conn, "mv_int_pk", "SELECT id, grp, val FROM t_int_pk")

        plan_comp = mgr.register_view(
            "CREATE VIEW mv_comp_pk AS SELECT k1, k2, val FROM t_comp_pk;", populate=True
        )
        self.assertIn("__mv_pk", plan_comp.backing_table_ddl)
        _assert_matches_live(self, conn, "mv_comp_pk", "SELECT k1, k2, val FROM t_comp_pk")
        conn.execute("INSERT INTO t_comp_pk VALUES (NULL, 2, 'e');")
        conn.execute("UPDATE t_comp_pk SET val = 'c2' WHERE k1 IS NULL AND k2 IS NULL;")
        conn.execute("DELETE FROM t_comp_pk WHERE k1 = 1 AND k2 IS NULL;")
        conn.commit()
        _assert_matches_live(self, conn, "mv_comp_pk", "SELECT k1, k2, val FROM t_comp_pk")

        plan_rowid = mgr.register_view(
            "CREATE VIEW mv_rowid_pk AS SELECT id, val FROM t_rowid_pk;", populate=True
        )
        self.assertNotIn("__mv_pk", plan_rowid.backing_table_ddl)
        _assert_matches_live(self, conn, "mv_rowid_pk", "SELECT id, val FROM t_rowid_pk")
        conn.close()


if __name__ == "__main__":
    unittest.main()
