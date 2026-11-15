#!/usr/bin/env python3
"""
Tests for Jinja2 procedural trigger and backfill template rendering.

Covers:
  - StrictUndefined fail-fast behavior across all template keys
  - SQLite, PostgreSQL, and MySQL read-only protection templates
  - Incremental backfill templates (PK range-fold & offset-batch, PG MERGE vs
    ON CONFLICT, MySQL 8+ row-alias vs MariaDB VALUES())
  - Trigger templates (SQLite WHEN/UPDATE OF, PG row/truncate/transition
    vectorized & loop modes, MySQL skip-trigger guard)
  - Live SQLite execution of rendered read-only triggers, recursive CTE
    incremental backfills, and DML trigger maintenance with reserved identifiers
"""

from __future__ import annotations

import sqlite3
import unittest

import jinja2

import matview_sync as ms


class TestJinja2ProceduralTemplates(unittest.TestCase):
    def test_strict_undefined_rejects_missing_context(self) -> None:
        self.assertGreaterEqual(len(ms.PROCEDURAL_SQL_TEMPLATES), 18)
        for tmpl_name in ms.PROCEDURAL_SQL_TEMPLATES:
            with self.assertRaises(jinja2.UndefinedError, msg=f"template {tmpl_name} did not enforce StrictUndefined"):
                ms.render_procedural_sql(tmpl_name)

    def test_readonly_triggers_sqlite_live_and_postgres_ddl(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute('CREATE TABLE "order" (id INTEGER PRIMARY KEY, amt INTEGER)')
            conn.execute('INSERT INTO "order" VALUES (1, 10), (2, 20)')
            mgr = ms.MaterializedViewManager(conn, dialect="sqlite")
            plan = mgr.register_view(
                'CREATE VIEW "select" AS SELECT id, amt FROM "order"',
                schema_sql='CREATE TABLE "order" (id INTEGER PRIMARY KEY, amt INTEGER)',
                populate=True,
            )
            self.assertEqual(plan.view_name, "select")
            self.assertEqual(
                conn.execute('SELECT id, amt FROM "select" ORDER BY id').fetchall(),
                [(1, 10), (2, 20)],
            )
            for dml in (
                'INSERT INTO "select" VALUES (3, 30)',
                'UPDATE "select" SET amt = 99 WHERE id = 1',
                'DELETE FROM "select" WHERE id = 1',
            ):
                with self.assertRaises(sqlite3.DatabaseError) as ctx:
                    conn.execute(dml)
                self.assertIn("read-only and maintained automatically", str(ctx.exception))
        finally:
            conn.close()

        pg_ts = ms.parse_table_schema('CREATE TABLE "order" (id BIGINT PRIMARY KEY, amt INT)', dialect="postgres")
        pg_vs = ms.parse_view_definition('CREATE VIEW "select" AS SELECT id, amt FROM "order"', dialect="postgres")
        pg_plan = ms.IncrementalViewCompiler(pg_ts, pg_vs, dialect="postgres").compile()
        ro_joined = "\n".join(pg_plan.readonly_trigger_ddls)
        self.assertIn("CREATE OR REPLACE FUNCTION", ro_joined)
        self.assertIn("RAISE EXCEPTION 'Materialized view % is read-only and maintained automatically'", ro_joined)
        self.assertIn('INSTEAD OF INSERT OR UPDATE OR DELETE ON "select"', ro_joined)

    def test_incremental_backfill_templates_across_dialects_and_versions(self) -> None:
        # SQLite live PK range-fold with WHERE predicate
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER)")
            conn.executemany(
                "INSERT INTO items (id, cat, val) VALUES (?, ?, ?)",
                [(i, f"c{i % 3}", i * 10) for i in range(1, 16)],
            )
            sq_schema_sql = "CREATE TABLE items (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER)"
            sq_ts = ms.parse_table_schema(sq_schema_sql, dialect="sqlite")
            v_grp_vs = ms.parse_view_definition(
                "CREATE VIEW v_grp AS SELECT cat, COUNT(*) AS cnt, SUM(val) AS total FROM items WHERE val >= 50 GROUP BY cat",
                dialect="sqlite",
            )
            plan = ms.IncrementalViewCompiler(
                sq_ts, v_grp_vs, dialect="sqlite", incremental_backfill=True, batch_size=3
            ).compile()
            pop_sql = "\n".join(plan.initial_populate_sqls)
            self.assertIn("WITH RECURSIVE __mv_bounds AS", pop_sql)
            self.assertIn("__mv_batches(lo, hi, max_id) AS", pop_sql)
            for stmt in [plan.backing_table_ddl] + plan.auxiliary_table_ddls + plan.index_ddls + [plan.reader_view_ddl] + plan.initial_populate_sqls:
                conn.execute(stmt)
            expected = conn.execute(
                "SELECT cat, COUNT(*), SUM(val) FROM items WHERE val >= 50 GROUP BY cat ORDER BY cat"
            ).fetchall()
            actual = conn.execute("SELECT cat, cnt, total FROM v_grp ORDER BY cat").fetchall()
            self.assertEqual(actual, expected)

            # Offset-batch template on multi-table JOIN view on live SQLite
            conn.execute("CREATE TABLE cats (id INTEGER PRIMARY KEY, name TEXT)")
            conn.executemany("INSERT INTO cats VALUES (?, ?)", [(1, "c0"), (2, "c1"), (3, "c2")])
            join_ts = ms.parse_table_schema(
                "CREATE TABLE items (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER); "
                "CREATE TABLE cats (id INTEGER PRIMARY KEY, name TEXT);",
                dialect="sqlite",
            )
            v_join_vs = ms.parse_view_definition(
                "CREATE VIEW v_join_batched AS SELECT c.name, COUNT(*) AS cnt FROM items i JOIN cats c ON c.name = i.cat GROUP BY c.name",
                dialect="sqlite",
            )
            join_plan = ms.IncrementalViewCompiler(
                join_ts, v_join_vs, dialect="sqlite", incremental_backfill=True, batch_size=2
            ).compile()
            self.assertIn("WITH __mv_batched_src AS", "\n".join(join_plan.initial_populate_sqls))
            for stmt in [join_plan.backing_table_ddl] + join_plan.auxiliary_table_ddls + join_plan.index_ddls + [join_plan.reader_view_ddl] + join_plan.initial_populate_sqls:
                conn.execute(stmt)
            join_actual = conn.execute("SELECT name, cnt FROM v_join_batched ORDER BY name").fetchall()
            join_expected = conn.execute(
                "SELECT c.name, COUNT(*) FROM items i JOIN cats c ON c.name = i.cat GROUP BY c.name ORDER BY c.name"
            ).fetchall()
            self.assertEqual(join_actual, join_expected)
        finally:
            conn.close()

        # PostgreSQL PK range-fold (global agg fallback) & offset-batch (MERGE PG 16 vs ON CONFLICT PG 14)
        pg_ts = ms.parse_table_schema(
            "CREATE TABLE items (id BIGINT PRIMARY KEY, cat TEXT, val INT)", dialect="postgres"
        )
        pg_global_vs = ms.parse_view_definition("CREATE VIEW v_cnt AS SELECT COUNT(*) AS cnt FROM items", dialect="postgres")
        pg16_global_plan = ms.IncrementalViewCompiler(
            pg_ts, pg_global_vs, dialect="postgres", engine_version="postgres:16", incremental_backfill=True, batch_size=4
        ).compile()
        pg14_global_plan = ms.IncrementalViewCompiler(
            pg_ts, pg_global_vs, dialect="postgres", engine_version="postgres:14", incremental_backfill=True, batch_size=4
        ).compile()
        pg16_global_sql = "\n".join(pg16_global_plan.initial_populate_sqls)
        pg14_global_sql = "\n".join(pg14_global_plan.initial_populate_sqls)
        self.assertIn("DO $$", pg16_global_sql)
        self.assertIn("MERGE INTO __matview_store_v_cnt", pg16_global_sql)
        self.assertIn("IF NOT EXISTS (SELECT 1 FROM __matview_store_v_cnt) THEN", pg16_global_sql)
        self.assertIn("HAVING COUNT(*) > 0", pg14_global_sql)

        pg_dist_vs = ms.parse_view_definition("CREATE VIEW v_d AS SELECT DISTINCT cat FROM items", dialect="postgres")
        pg16_dist = ms.IncrementalViewCompiler(
            pg_ts, pg_dist_vs, dialect="postgres", engine_version="postgres:16", incremental_backfill=True, batch_size=5
        ).compile()
        pg14_dist = ms.IncrementalViewCompiler(
            pg_ts, pg_dist_vs, dialect="postgres", engine_version="postgres:14", incremental_backfill=True, batch_size=5
        ).compile()
        self.assertIn("MERGE INTO", "\n".join(pg16_dist.initial_populate_sqls))
        self.assertNotIn("MERGE INTO", "\n".join(pg14_dist.initial_populate_sqls))
        self.assertIn("ON CONFLICT", "\n".join(pg14_dist.initial_populate_sqls))

        # MySQL 8.4 vs MariaDB 10.11 procedure templates (PK-range WHILE + offset REPEAT)
        my_ts = ms.parse_table_schema(
            "CREATE TABLE items (id BIGINT PRIMARY KEY, cat VARCHAR(32), val INT); "
            "CREATE TABLE cats (id BIGINT PRIMARY KEY, name VARCHAR(32));",
            dialect="mysql",
        )
        my_dist_vs = ms.parse_view_definition("CREATE VIEW v_d AS SELECT DISTINCT cat FROM items", dialect="mysql")
        my_pk_plan = ms.IncrementalViewCompiler(
            my_ts, my_dist_vs, dialect="mysql", engine_version="mysql:8.4",
            incremental_backfill=True, batch_size=5, ignore_limitations=True,
        ).compile()
        my_pk_pop = my_pk_plan.initial_populate_sqls
        self.assertTrue(my_pk_pop[0].startswith("DROP PROCEDURE IF EXISTS"))
        self.assertIn("WHILE __mv_cur_pk < __mv_max_pk DO", my_pk_pop[1])
        self.assertTrue(my_pk_pop[-2].startswith("CALL "))
        self.assertTrue(my_pk_pop[-1].startswith("DROP PROCEDURE IF EXISTS"))

        my_join_vs = ms.parse_view_definition(
            "CREATE VIEW v_j AS SELECT c.name, COUNT(*) AS cnt FROM items i JOIN cats c ON c.name = i.cat GROUP BY c.name",
            dialect="mysql",
        )
        my8_plan = ms.IncrementalViewCompiler(
            my_ts, my_join_vs, dialect="mysql", engine_version="mysql:8.4",
            incremental_backfill=True, batch_size=5, ignore_limitations=True,
        ).compile()
        m10_plan = ms.IncrementalViewCompiler(
            my_ts, my_join_vs, dialect="mysql", engine_version="mariadb:10.11",
            incremental_backfill=True, batch_size=5, ignore_limitations=True,
        ).compile()
        my8_pop = my8_plan.initial_populate_sqls
        self.assertTrue(my8_pop[0].startswith("DROP PROCEDURE IF EXISTS"))
        self.assertIn("REPEAT", my8_pop[1])
        self.assertIn("UNTIL __mv_rows <= 0 END REPEAT;", my8_pop[1])
        self.assertTrue(my8_pop[-2].startswith("CALL "))
        self.assertTrue(my8_pop[-1].startswith("DROP PROCEDURE IF EXISTS"))
        self.assertIn("= __mv_init.", my8_pop[1])
        self.assertIn("VALUES(", m10_plan.initial_populate_sqls[1])

    def test_trigger_templates_sqlite_pg_transition_and_mysql(self) -> None:
        pg_ts = ms.parse_table_schema(
            "CREATE TABLE items (id BIGINT PRIMARY KEY, cat TEXT, val INT)", dialect="postgres"
        )
        pg_vs = ms.parse_view_definition(
            "CREATE VIEW v AS SELECT cat, COUNT(*) AS cnt, SUM(val) AS s FROM items WHERE val > 0 GROUP BY cat",
            dialect="postgres",
        )

        # Row-level PG triggers
        pg_row_plan = ms.IncrementalViewCompiler(pg_ts, pg_vs, dialect="postgres", pg_use_transition_tables=False).compile()
        pg_row_sql = "\n".join(pg_row_plan.trigger_ddls)
        self.assertIn("matview.skip_triggers", pg_row_sql)
        self.assertIn("AFTER TRUNCATE ON", pg_row_sql)
        self.assertIn("IF (TG_OP = 'DELETE') THEN", pg_row_sql)

        # Transition-table vectorized vs loop PG triggers
        pg_unfiltered_vs = ms.parse_view_definition(
            "CREATE VIEW v AS SELECT cat, COUNT(*) AS cnt, SUM(val) AS s FROM items GROUP BY cat",
            dialect="postgres",
        )
        pg_vec_plan = ms.IncrementalViewCompiler(
            pg_ts, pg_unfiltered_vs, dialect="postgres", pg_use_transition_tables=True, pg_vectorize_transition_tables=True
        ).compile()
        pg_loop_plan = ms.IncrementalViewCompiler(
            pg_ts, pg_unfiltered_vs, dialect="postgres", pg_use_transition_tables=True, pg_vectorize_transition_tables=False
        ).compile()
        vec_sql = "\n".join(pg_vec_plan.trigger_ddls)
        loop_sql = "\n".join(pg_loop_plan.trigger_ddls)
        self.assertIn("REFERENCING NEW TABLE AS __mv_new_rows", vec_sql)
        self.assertNotIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", vec_sql)
        self.assertIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", loop_sql)
        self.assertIn("FOR OLD IN SELECT * FROM __mv_old_rows LOOP", loop_sql)

        # MySQL row triggers
        my_ts = ms.parse_table_schema(
            "CREATE TABLE items (id BIGINT PRIMARY KEY, cat VARCHAR(32), val INT)", dialect="mysql"
        )
        my_vs = ms.parse_view_definition(
            "CREATE VIEW v AS SELECT cat, COUNT(*) AS cnt FROM items WHERE val > 0 GROUP BY cat", dialect="mysql"
        )
        my_plan = ms.IncrementalViewCompiler(my_ts, my_vs, dialect="mysql", ignore_limitations=True).compile()
        my_sql = "\n".join(my_plan.trigger_ddls)
        self.assertIn("@__matview_skip_triggers", my_sql)
        self.assertEqual(len(my_plan.trigger_ddls), 3)

        # SQLite filtered WHEN / UPDATE OF live execution
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER, note TEXT)")
            mgr = ms.MaterializedViewManager(conn, dialect="sqlite")
            t_plan = mgr.register_view(
                "CREATE VIEW v_live AS SELECT cat, COUNT(*) AS cnt, SUM(val) AS s FROM items WHERE val > 0 GROUP BY cat",
                schema_sql="CREATE TABLE items (id INTEGER PRIMARY KEY, cat TEXT, val INTEGER, note TEXT)",
                populate=True,
            )
            upd_trig = [d for d in t_plan.trigger_ddls if "AFTER UPDATE" in d][0]
            self.assertIn("AFTER UPDATE OF id, cat, val, note ON items", upd_trig)
            conn.execute("INSERT INTO items VALUES (1, 'a', 10, 'n1'), (2, 'a', -5, 'n2')")
            conn.execute("UPDATE items SET val = 20 WHERE id = 2")
            conn.execute("UPDATE items SET val = 0 WHERE id = 1")
            conn.execute("DELETE FROM items WHERE id = 2")
            self.assertEqual(conn.execute("SELECT cat, cnt, s FROM v_live").fetchall(), [])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
