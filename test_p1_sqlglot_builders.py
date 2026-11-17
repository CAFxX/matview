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


if __name__ == "__main__":
    unittest.main(verbosity=2)
