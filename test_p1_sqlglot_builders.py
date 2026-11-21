#!/usr/bin/env python3
"""Tests for AST-based SQL code generation helpers backed by `sqlglot.exp`."""

import sqlite3
import unittest

import sqlglot
from sqlglot import exp as sqlglot_exp

from matview_sync import (
    Dialect,
    MaterializedViewManager,
    build_alter_rename_table_sql,
    build_create_index_sql,
    build_create_view_sql,
    build_drop_sql,
    build_insert_select_sql,
    build_select_sql,
    generate_materialized_view,
    parse_table_schema,
)


class TestSqlglotExpBuilders(unittest.TestCase):
    """Verifies dialect correctness and execution of `sqlglot.exp` builders."""

    def test_build_drop_sql_dialects_and_quoting(self) -> None:
        self.assertEqual(
            build_drop_sql("VIEW", "mv_sales", dialect=Dialect.SQLITE),
            "DROP VIEW IF EXISTS mv_sales;",
        )
        self.assertEqual(
            build_drop_sql("VIEW", "my sales", dialect=Dialect.POSTGRES, cascade=True),
            'DROP VIEW IF EXISTS "my sales" CASCADE;',
        )
        self.assertEqual(
            build_drop_sql("TABLE", ["user", "old_tbl"], dialect=Dialect.MYSQL),
            "DROP TABLE IF EXISTS `user`, old_tbl;",
        )
        self.assertEqual(
            build_drop_sql("TABLE", "public.mv_tbl", dialect=Dialect.POSTGRES),
            "DROP TABLE IF EXISTS public.mv_tbl;",
        )

    def test_build_alter_rename_table_sql_across_dialects(self) -> None:
        self.assertEqual(
            build_alter_rename_table_sql("shadow_mv", "mv_store", dialect=Dialect.SQLITE),
            "ALTER TABLE shadow_mv RENAME TO mv_store;",
        )
        self.assertEqual(
            build_alter_rename_table_sql("shadow_mv", "mv_store", dialect=Dialect.POSTGRES),
            "ALTER TABLE shadow_mv RENAME TO mv_store;",
        )
        self.assertEqual(
            build_alter_rename_table_sql("shadow mv", "order", dialect=Dialect.MYSQL),
            "ALTER TABLE `shadow mv` RENAME `order`;",
        )

    def test_build_create_view_sql_and_live_sqlite_execution(self) -> None:
        pg_ddl = build_create_view_sql(
            "mv_summary",
            "__matview_store_summary",
            ["grp", "total"],
            where_sql="grp IS NOT NULL",
            dialect=Dialect.POSTGRES,
            replace=True,
        )
        parsed_pg = sqlglot.parse_one(pg_ddl, read="postgres")
        self.assertIsInstance(parsed_pg, sqlglot_exp.Create)
        self.assertTrue(parsed_pg.args.get("replace"))

        my_ddl = build_create_view_sql(
            "mv summary",
            "__matview_store_summary",
            ["`grp`", "`user`"],
            where_sql="EXISTS (SELECT 1)",
            dialect=Dialect.MYSQL,
            algorithm="TEMPTABLE",
        )
        self.assertIn("ALGORITHM = TEMPTABLE", my_ddl)
        self.assertIn("`mv summary`", my_ddl)
        self.assertIn("`user`", my_ddl)

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE backing (grp TEXT, total INT);")
            conn.execute("INSERT INTO backing VALUES ('a', 10), (NULL, 5);")
            sq_ddl = build_create_view_sql(
                "v_live",
                "backing",
                ["grp", "total"],
                where_sql="grp IS NOT NULL",
                dialect=Dialect.SQLITE,
            )
            conn.execute(sq_ddl)
            rows = conn.execute("SELECT grp, total FROM v_live;").fetchall()
            self.assertEqual(rows, [("a", 10)])
        finally:
            conn.close()

    def test_build_create_index_sql_variants_and_live_sqlite(self) -> None:
        pg_idx = build_create_index_sql(
            "idx_grp",
            "mv_tbl",
            ["grp", "sub"],
            dialect=Dialect.POSTGRES,
            unique=True,
            using_method="btree",
            where_clause="grp IS NOT NULL",
            nulls_not_distinct=True,
        )
        self.assertIn("CREATE UNIQUE INDEX idx_grp ON mv_tbl", pg_idx)
        self.assertIn("USING btree", pg_idx)
        self.assertIn("WHERE grp IS NOT NULL", pg_idx)
        self.assertTrue(pg_idx.endswith("NULLS NOT DISTINCT;"))

        my_expr_idx = build_create_index_sql(
            "idx_expr",
            "mv_tbl",
            ["((`a` + `b`)) DESC"],
            dialect=Dialect.MYSQL,
        )
        self.assertNotIn("(((", my_expr_idx)
        self.assertIn("((`a` + `b`) DESC)", my_expr_idx)
        self.assertIn("ON mv_tbl (", my_expr_idx)

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t_idx (id INTEGER PRIMARY KEY, name TEXT, score INT);")
            sq_idx = build_create_index_sql(
                "idx_t_name",
                "t_idx",
                ['"name" COLLATE NOCASE DESC'],
                dialect=Dialect.SQLITE,
                where_clause="score > 0",
            )
            conn.execute(sq_idx)
            names = [r[1] for r in conn.execute("PRAGMA index_list('t_idx');").fetchall()]
            self.assertIn("idx_t_name", names)
        finally:
            conn.close()

    def test_build_select_and_insert_select_sql(self) -> None:
        sel = build_select_sql(
            ["bucket", "COUNT(*) AS cnt", "SUM(amt) AS total"],
            "items",
            where_sql="amt > 0",
            group_by_exprs=["bucket"],
            having_sql="COUNT(*) >= 1",
            order_by_exprs=["total DESC"],
            limit=5,
            dialect=Dialect.POSTGRES,
        )
        parsed_sel = sqlglot.parse_one(sel, read="postgres")
        self.assertIsInstance(parsed_sel, sqlglot_exp.Select)
        self.assertIsNotNone(parsed_sel.args.get("group"))
        self.assertIsNotNone(parsed_sel.args.get("having"))
        self.assertIsNotNone(parsed_sel.args.get("limit"))

        ins = build_insert_select_sql("dst", ["bucket", "cnt", "total"], sel, dialect=Dialect.MYSQL)
        parsed_ins = sqlglot.parse_one(ins, read="mysql")
        self.assertIsInstance(parsed_ins, sqlglot_exp.Insert)

    def test_end_to_end_plan_ddls_use_ast_builders(self) -> None:
        schema = "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT NOT NULL, amt INTEGER NOT NULL);"
        view_sql = (
            "CREATE VIEW mv_sales_region AS "
            "SELECT region, COUNT(*) AS cnt, SUM(amt) AS total FROM sales GROUP BY region;"
        )
        schemas = parse_table_schema(schema, dialect=Dialect.POSTGRES)
        plan = generate_materialized_view(schemas, view_sql, dialect=Dialect.POSTGRES)
        self.assertIsInstance(sqlglot.parse_one(plan.reader_view_ddl, read="postgres"), sqlglot_exp.Create)
        for idx_ddl in plan.index_ddls:
            self.assertIsInstance(sqlglot.parse_one(idx_ddl, read="postgres"), sqlglot_exp.Create)
        for drop_ddl in plan.drop_statements():
            if not drop_ddl.upper().startswith("DROP FUNCTION"):
                self.assertIsInstance(sqlglot.parse_one(drop_ddl, read="postgres"), sqlglot_exp.Drop)

        conn = sqlite3.connect(":memory:")
        try:
            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            mgr.apply_schema_ddl(schema)
            conn.execute("INSERT INTO sales VALUES (1, 'EU', 10), (2, 'US', 20);")
            conn.commit()
            live_plan = mgr.register_view(view_sql)
            self.assertEqual(
                conn.execute(f"SELECT region, cnt, total FROM {live_plan.view_name} ORDER BY region;").fetchall(),
                [("EU", 1, 10), ("US", 1, 20)],
            )
            self.assertTrue(mgr.drop_view(live_plan.view_name))
        finally:
            conn.close()


    def test_special_char_view_names_and_extract_created_object(self) -> None:
        import re

        from matview.sql_frontend import _extract_created_object

        # _extract_created_object must handle quoted names with spaces, single quotes,
        # dots, and escaped quote delimiters.
        self.assertEqual(_extract_created_object('CREATE VIEW "sales summary" AS SELECT 1;'), ("view", "sales summary"))
        self.assertEqual(_extract_created_object('CREATE VIEW "user\'s view" AS SELECT 1;'), ("view", "user's view"))
        self.assertEqual(_extract_created_object('CREATE VIEW "a ""b"" c" AS SELECT 1;'), ("view", 'a "b" c'))
        self.assertEqual(_extract_created_object('CREATE VIEW "my.view" AS SELECT 1;'), ("view", "my.view"))
        self.assertEqual(_extract_created_object('CREATE VIEW "public"."my.view" AS SELECT 1;'), ("view", "my.view"))
        self.assertEqual(_extract_created_object("CREATE VIEW `sales summary` AS SELECT 1;"), ("view", "sales summary"))
        self.assertEqual(_extract_created_object("CREATE TABLE `order items` (id INT);"), ("table", "order items"))
        self.assertEqual(_extract_created_object('CREATE TRIGGER "trg 1" AFTER INSERT ON t;'), ("trigger", "trg 1"))

        # Across all three dialects, special-character view names (and base table names)
        # must produce valid unquoted internal identifiers without collisions.
        for dial in (Dialect.POSTGRES, Dialect.SQLITE, Dialect.MYSQL):
            q = "`" if dial == Dialect.MYSQL else '"'
            sc = parse_table_schema(
                f"CREATE TABLE {q}order items{q} (id INT PRIMARY KEY, g INT, v INT);",
                dialect=dial,
            )
            seen_stores = set()
            vnames = ["sales summary", "sales_summary", "Sales Summary", "sales-summary", "user's view", "my.view", "---"]
            for vn in vnames:
                qvn = f"{q}{vn.replace(q, q + q)}{q}"
                plan = generate_materialized_view(
                    sc,
                    f"CREATE VIEW {qvn} AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM {q}order items{q} GROUP BY g;",
                    dialect=dial,
                    ignore_limitations=True,
                )
                self.assertRegex(plan.backing_table_name, r"^[A-Za-z_][A-Za-z0-9_]*$", (dial, vn))
                self.assertLessEqual(len(plan.backing_table_name), 62)
                self.assertNotIn(plan.backing_table_name, seen_stores, (dial, vn))
                seen_stores.add(plan.backing_table_name)
                for obj_name in (*plan.trigger_names, *plan.index_names, *plan.auxiliary_table_names):
                    self.assertRegex(obj_name, r"^[A-Za-z_][A-Za-z0-9_]*$", (dial, vn, obj_name))
                    self.assertLessEqual(len(obj_name), 62)
                expected_ref = vn if vn == "sales_summary" else qvn
                self.assertIn(expected_ref, plan.reader_view_ddl)
                self.assertTrue(any(expected_ref in stmt for stmt in plan.drop_statements()), (dial, vn, plan.drop_statements()))

        # Trailing-newline bypass and long custom prefix cap in _bounded_ident.
        long_prefix_plan = generate_materialized_view(
            parse_table_schema("CREATE TABLE t (id INT PRIMARY KEY, g INT, v INT);", dialect=Dialect.POSTGRES),
            'CREATE VIEW "sales\n" AS SELECT g, SUM(v) AS s FROM t GROUP BY g;',
            dialect=Dialect.POSTGRES,
            object_prefix="x_" * 20,
        )
        self.assertRegex(long_prefix_plan.backing_table_name, r"^[A-Za-z_][A-Za-z0-9_]*$")
        self.assertLessEqual(len(long_prefix_plan.backing_table_name), 62)
        for obj_name in (*long_prefix_plan.trigger_names, *long_prefix_plan.index_names):
            self.assertRegex(obj_name, r"^[A-Za-z_][A-Za-z0-9_]*$")
            self.assertLessEqual(len(obj_name), 62)

        # PostgreSQL BIT VARYING / BIT VARYING(n) DDL must parse cleanly in parse_table_schema.
        bit_schemas = parse_table_schema(
            "CREATE TABLE tb (id INT PRIMARY KEY, g INT, b4 BIT(4), bv BIT VARYING, bv8 BIT VARYING(8));",
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("tb", bit_schemas)
        self.assertIn("bv", bit_schemas["tb"].columns)
        self.assertIn("bv8", bit_schemas["tb"].columns)

        # Live SQLite end-to-end lifecycle across all adversarial view names on a spaced base table,
        # including diff_views(prune_removed=True) quoted DROP generation.
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute('CREATE TABLE "order items" (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER);')
            conn.execute('INSERT INTO "order items" VALUES (1, 1, 10), (2, 1, 20), (3, 2, 5);')
            conn.commit()
            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            sq_names = ["sales summary", "sales_summary", "sales-summary", "user's view", 'a "b" c', "my.view", "---"]
            for vn in sq_names:
                qvn = '"' + vn.replace('"', '""') + '"'
                mgr.register_view(
                    f'CREATE VIEW {qvn} AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM "order items" GROUP BY g;'
                )
            conn.execute('INSERT INTO "order items" VALUES (4, 2, 15);')
            conn.execute('UPDATE "order items" SET v = 30 WHERE id = 1;')
            conn.execute('DELETE FROM "order items" WHERE id = 3;')
            conn.commit()
            expected = conn.execute(
                'SELECT g, SUM(v), COUNT(*) FROM "order items" GROUP BY g ORDER BY g;'
            ).fetchall()
            prune_report = mgr.diff_views(
                ['CREATE VIEW sales_summary AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM "order items" GROUP BY g;'],
                prune_removed=True,
            )
            for vn in sq_names:
                qvn = '"' + vn.replace('"', '""') + '"'
                if vn != "sales_summary":
                    self.assertIn(vn, prune_report)
                    self.assertTrue(
                        any(f"DROP VIEW IF EXISTS {qvn};" == d for d in prune_report[vn].pending_ddls),
                        (vn, prune_report[vn].pending_ddls),
                    )
                got = conn.execute(f"SELECT g, s, c FROM {qvn} ORDER BY g;").fetchall()
                self.assertEqual(got, expected, vn)
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(f"DELETE FROM {qvn};")
                conn.rollback()
                self.assertEqual(mgr.check_schema_drift()[vn], "ok")
                mgr.repair_view(vn)
                self.assertTrue(mgr.drop_view(vn))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
