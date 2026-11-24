#!/usr/bin/env python3
"""Tests for P1 Bug #12: Dialect-aware SQLGlot tokenization, AST parsing, and plan compilation."""

from __future__ import annotations

import unittest

from matview_sync import (
    Dialect,
    EngineSubdialect,
    EngineVersion,
    MaterializedViewPlan,
    SQLGlotTokenType,
    SQLTokenType,
    _sqlglot_dialect,
    _tokenize_sql,
    generate_materialized_view,
    parse_engine_version,
    parse_sql_ast,
    parse_table_schema,
    parse_view_definition,
    tokenize_sql,
)


class DialectAwareParsingTests(unittest.TestCase):
    def test_sqlglot_dialect_mapping(self) -> None:
        """Verify _sqlglot_dialect maps dialect enums, subdialects, and versions cleanly."""
        self.assertEqual(_sqlglot_dialect(Dialect.SQLITE), "sqlite")
        self.assertEqual(_sqlglot_dialect(Dialect.POSTGRES), "postgres")
        self.assertEqual(_sqlglot_dialect(Dialect.MYSQL), "mysql")

        self.assertEqual(_sqlglot_dialect(EngineSubdialect.SQLITE), "sqlite")
        self.assertEqual(_sqlglot_dialect(EngineSubdialect.POSTGRES), "postgres")
        self.assertEqual(_sqlglot_dialect(EngineSubdialect.MYSQL), "mysql")
        # MariaDB must map to 'mysql' in sqlglot to avoid ValueError: Unknown dialect 'mariadb'
        self.assertEqual(_sqlglot_dialect(EngineSubdialect.MARIADB), "mysql")

        ev_pg = parse_engine_version("postgres:17")
        ev_my = parse_engine_version("mysql:8.4")
        ev_maria = parse_engine_version("mariadb:11.4")
        ev_sq = parse_engine_version("sqlite:3.35.0")

        self.assertEqual(_sqlglot_dialect(ev_pg), "postgres")
        self.assertEqual(_sqlglot_dialect(ev_my), "mysql")
        self.assertEqual(_sqlglot_dialect(ev_maria), "mysql")
        self.assertEqual(_sqlglot_dialect(ev_sq), "sqlite")

        self.assertEqual(_sqlglot_dialect("mariadb"), "mysql")
        self.assertEqual(_sqlglot_dialect("postgresql"), "postgres")
        self.assertEqual(_sqlglot_dialect("sqlite"), "sqlite")
        self.assertEqual(_sqlglot_dialect(None), "sqlite")

    def test_postgres_schema_parsing(self) -> None:
        """Verify PostgreSQL specific data types (SERIAL, JSONB, BYTEA, TEXT[], CITEXT) parse correctly."""
        schema_sql = """
        CREATE TABLE pg_events (
            id SERIAL PRIMARY KEY,
            payload JSONB NOT NULL,
            raw_data BYTEA,
            tags TEXT[],
            metric_label CITEXT,
            price NUMERIC(10, 2) NOT NULL
        );
        """
        schemas = parse_table_schema(schema_sql, dialect=Dialect.POSTGRES)
        self.assertIn("pg_events", schemas)
        cols = schemas["pg_events"].columns
        self.assertTrue(cols["id"].primary_key)
        self.assertEqual(cols["payload"].data_type.upper(), "JSONB")
        self.assertEqual(cols["raw_data"].data_type.upper(), "BYTEA")
        self.assertIn("TEXT[]", cols["tags"].data_type.upper())
        self.assertEqual(cols["metric_label"].data_type.upper(), "CITEXT")
        self.assertTrue(
            "NUMERIC" in cols["price"].data_type.upper()
            or "DECIMAL" in cols["price"].data_type.upper()
        )

    def test_postgres_view_ast_and_plan_generation(self) -> None:
        """Verify PostgreSQL ::type casts and -> / ->> JSON operators compile cleanly end-to-end."""
        schema_sql = """
        CREATE TABLE pg_orders (
            order_id SERIAL PRIMARY KEY,
            details JSONB NOT NULL,
            status TEXT NOT NULL
        );
        """
        view_sql = """
        CREATE VIEW mv_orders_summary AS
        SELECT
            (details->>'region')::TEXT AS region,
            COUNT(*) AS order_cnt,
            SUM(((details->>'amount')::NUMERIC(10, 2))) AS total_amount
        FROM pg_orders
        WHERE status = 'completed'
        GROUP BY (details->>'region')::TEXT;
        """
        schemas = parse_table_schema(schema_sql, dialect=Dialect.POSTGRES)
        view_spec = parse_view_definition(view_sql, known_tables=set(schemas.keys()), dialect=Dialect.POSTGRES)
        self.assertEqual(view_spec.base_table, "pg_orders")
        self.assertEqual(len(view_spec.select_items), 3)

        plan = generate_materialized_view(
            schemas,
            view_sql,
            dialect=Dialect.POSTGRES,
            engine_version="postgres:16",
        )
        self.assertIsInstance(plan, MaterializedViewPlan)
        self.assertEqual(plan.dialect, Dialect.POSTGRES)
        self.assertTrue(len(plan.trigger_ddls) > 0)
        self.assertTrue(any("FUNCTION" in ddl for ddl in plan.trigger_ddls))
        self.assertTrue(any("RETURNS TRIGGER" in ddl for ddl in plan.trigger_ddls))
        self.assertIn("mv_orders_summary", plan.reader_view_ddl)

    def test_mysql_schema_parsing(self) -> None:
        """Verify MySQL / MariaDB specific schema syntax (backticks, UNSIGNED, MEDIUMINT)."""
        schema_sql = """
        CREATE TABLE `my_metrics` (
            `id` INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            `counter` MEDIUMINT UNSIGNED NOT NULL,
            `score` DOUBLE NOT NULL,
            `note` VARCHAR(255)
        );
        """
        schemas = parse_table_schema(schema_sql, dialect=Dialect.MYSQL)
        self.assertIn("my_metrics", schemas)
        cols = schemas["my_metrics"].columns
        self.assertIn("id", cols)
        self.assertTrue(cols["id"].primary_key)
        self.assertIn("UNSIGNED", cols["id"].data_type.upper())
        self.assertIn("MEDIUMINT", cols["counter"].data_type.upper())

    def test_mariadb_schema_parsing_via_engine_version(self) -> None:
        """Verify MariaDB engine version passes down cleanly without unknown dialect errors."""
        schema_sql = """
        CREATE TABLE `sales` (
            `id` BIGINT UNSIGNED NOT NULL PRIMARY KEY,
            `amount` DECIMAL(12, 4) NOT NULL
        );
        """
        ev = parse_engine_version("mariadb:11.4")
        schemas = parse_table_schema(schema_sql, dialect=ev.dialect)
        self.assertIn("sales", schemas)
        cols = schemas["sales"].columns
        self.assertIn("id", cols)
        self.assertTrue(cols["id"].primary_key)

    def test_mysql_view_ast_and_plan_generation(self) -> None:
        """Verify MySQL / MariaDB backtick queries and aggregate compilation."""
        schema_sql = """
        CREATE TABLE `inventory` (
            `item_id` INT UNSIGNED NOT NULL PRIMARY KEY,
            `warehouse_id` INT UNSIGNED NOT NULL,
            `quantity` INT NOT NULL
        );
        """
        view_sql = """
        CREATE VIEW `mv_wh_inventory` AS
        SELECT
            `warehouse_id`,
            COUNT(*) AS `item_types_count`,
            SUM(`quantity`) AS `total_quantity`
        FROM `inventory`
        GROUP BY `warehouse_id`;
        """
        schemas = parse_table_schema(schema_sql, dialect=Dialect.MYSQL)
        plan = generate_materialized_view(
            schemas,
            view_sql,
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4",
            ignore_limitations=True,
        )
        self.assertIsInstance(plan, MaterializedViewPlan)
        self.assertEqual(plan.dialect, Dialect.MYSQL)
        self.assertTrue(len(plan.trigger_ddls) > 0)
        self.assertTrue(any("AFTER INSERT ON inventory" in ddl or "AFTER INSERT ON `inventory`" in ddl for ddl in plan.trigger_ddls))
        self.assertTrue(any("ON DUPLICATE KEY UPDATE" in ddl for ddl in plan.trigger_ddls))
        self.assertIn("mv_wh_inventory", plan.reader_view_ddl)

    def test_sqlite_parsing_and_fallback(self) -> None:
        """Verify SQLite square bracket quoting and standard syntax remain fully supported."""
        schema_sql = """
        CREATE TABLE [t_accounts] (
            [acc_id] INTEGER PRIMARY KEY,
            [balance] REAL NOT NULL
        );
        """
        schemas = parse_table_schema(schema_sql, dialect=Dialect.SQLITE)
        self.assertIn("t_accounts", schemas)
        cols = schemas["t_accounts"].columns
        self.assertIn("acc_id", cols)
        self.assertTrue(cols["acc_id"].primary_key)

        view_sql = """
        CREATE VIEW mv_acc_summary AS
        SELECT COUNT(*) AS total_accounts, TOTAL(balance) AS sum_balance
        FROM [t_accounts];
        """
        plan = generate_materialized_view(
            schemas,
            view_sql,
            dialect=Dialect.SQLITE,
        )
        self.assertIsInstance(plan, MaterializedViewPlan)
        self.assertEqual(plan.dialect, Dialect.SQLITE)
        self.assertTrue(len(plan.trigger_ddls) > 0)
        self.assertTrue(any("AFTER INSERT ON [t_accounts]" in ddl or "AFTER INSERT ON t_accounts" in ddl for ddl in plan.trigger_ddls))

    def test_tokenize_sql_dialect_awareness(self) -> None:
        """Verify tokenize_sql recognizes dialect-specific tokens."""
        # PostgreSQL dollar-quoted string and :: type cast
        pg_sql = "SELECT $$hello$$ AS val, id::INTEGER FROM tbl;"
        toks_pg = tokenize_sql(pg_sql, dialect=Dialect.POSTGRES)
        dollar_tokens = [t for t in toks_pg if t.kind == SQLTokenType.DOLLAR_STRING]
        self.assertTrue(any("$$hello$$" in t.text for t in dollar_tokens))
        self.assertTrue(any(t.kind == SQLTokenType.OP and t.text == "::" for t in toks_pg))

        # MySQL backtick quoting
        my_sql = "SELECT `col_a` FROM `tbl`;"
        toks_my = tokenize_sql(my_sql, dialect=Dialect.MYSQL)
        idents = [t for t in toks_my if t.kind == SQLTokenType.QUOTED_IDENT]
        self.assertEqual([t.unquoted for t in idents], ["col_a", "tbl"])

        # Default fallback to SQLite
        sq_sql = "SELECT [col_x] FROM [tbl_y];"
        toks_sq = tokenize_sql(sq_sql)
        idents_sq = [t for t in toks_sq if t.kind == SQLTokenType.QUOTED_IDENT]
        self.assertEqual([t.unquoted for t in idents_sq], ["col_x", "tbl_y"])

    def test_ast_parsing_preserves_dialect_semantics(self) -> None:
        """Verify parse_sql_ast uses the dialect parser rather than forcing SQLite."""
        # PostgreSQL JSON arrow operator
        pg_expr = "SELECT data->>'key' FROM t;"
        ast_pg = parse_sql_ast(pg_expr, dialect=Dialect.POSTGRES)
        self.assertEqual(ast_pg.sql("postgres"), "SELECT data ->> 'key' FROM t")

        # MySQL backtick quoting
        my_expr = "SELECT `k` FROM `t`;"
        ast_my = parse_sql_ast(my_expr, dialect=Dialect.MYSQL)
        self.assertEqual(ast_my.sql("mysql"), "SELECT `k` FROM `t`")

    def test_mysql_statistical_aggregate_names_round_trip_with_their_native_semantics(self) -> None:
        """The product relies on sqlglot to spell MySQL/MariaDB statistics. At
        sqlglot 30.23.0 (the former declared floor) VAR_POP was generated as the
        non-existent MySQL function VARIANCE_POP and MySQL VARIANCE (population
        variance) was parsed as the sample variance, so the minimum-dependency
        lane compiled a view the server could not run or that meant something
        else. Every MySQL spelling must keep its name and population/sample kind."""
        expected = {
            "VAR_POP": "VAR_POP", "VARIANCE": "VAR_POP", "STD": "STDDEV_POP", "STDDEV": "STDDEV_POP",
            "STDDEV_POP": "STDDEV_POP", "VAR_SAMP": "VAR_SAMP", "STDDEV_SAMP": "STDDEV_SAMP",
        }
        for spelling, native in expected.items():
            with self.subTest(spelling=spelling):
                ast = parse_sql_ast(f"SELECT {spelling}(a) FROM t", dialect=Dialect.MYSQL)
                self.assertEqual(ast.sql("mysql"), f"SELECT {native}(a) FROM t")
                self.assertEqual(ast.sql("postgres"), f"SELECT {native}(a) FROM t")


if __name__ == "__main__":
    unittest.main(verbosity=2)
