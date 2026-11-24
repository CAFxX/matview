"""
Comprehensive tests for Engine & Version-Specific Optimal Code Generation:
1. P1 Bug #11: PostgreSQL NULLIF variance division-by-zero guards & live evaluation.
2. P1 Bug #13: MySQL 8.0.19+/8.4/9.7 row-alias upsert syntax vs MySQL 5.7 vs MariaDB VALUES().
3. PostgreSQL 15+ MERGE atomic set-based refresh/backfill & FILLFACTOR.
4. SQLite clustered composite-PK WITHOUT ROWID backing/aux tables & RETURNING in drop_view.
"""

import math
import os
import sqlite3
import unittest
from typing import Dict
from urllib.parse import urlparse

from test_matview_sync import FLOAT_OPT_IN, pin_native_plan
from matview_sync import (
    Dialect,
    EngineVersion,
    IncrementalViewCompiler,
    MaintenanceStrategy,
    MaterializedViewManager,
    TableSchema,
    generate_materialized_view,
    generate_materialized_views,
    parse_engine_version,
    parse_table_schema,
)


class TestPostgresVarianceNullifGuards(unittest.TestCase):
    """Verifies P1 Bug #11: PostgreSQL NULLIF division-by-zero guards in variance formulas."""

    def test_postgres_variance_sql_contains_nullif_guards(self):
        """PG variance SQL from _variance_formula_sql must contain NULLIF({cnt}, 0) and NULLIF({denom}, 0)."""
        schema_sql = "CREATE TABLE metrics (id INT PRIMARY KEY, grp INT, val DOUBLE PRECISION);"
        schemas = parse_table_schema(schema_sql)
        view_sql = "CREATE VIEW mv_var AS SELECT grp, VAR_SAMP(val) AS metric_val FROM metrics GROUP BY grp;"
        plan = generate_materialized_view(schemas, view_sql, dialect=Dialect.POSTGRES, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
        compiler = IncrementalViewCompiler(schemas, plan.view_spec, dialect=Dialect.POSTGRES, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)

        # 1. PostgreSQL sample variance & stddev
        pg_var_samp = compiler._variance_formula_sql("new_cnt", "new_sum", "new_sum2", is_sample=True, is_stddev=False)
        self.assertIn("NULLIF(new_cnt, 0)", pg_var_samp)
        self.assertIn("NULLIF((new_cnt - 1), 0)", pg_var_samp)
        self.assertIn("DOUBLE PRECISION", pg_var_samp)

        pg_std_samp = compiler._variance_formula_sql("new_cnt", "new_sum", "new_sum2", is_sample=True, is_stddev=True)
        self.assertIn("SQRT(", pg_std_samp)
        self.assertIn("NULLIF((new_cnt - 1), 0)", pg_std_samp)

        # 2. PostgreSQL population variance & stddev
        pg_var_pop = compiler._variance_formula_sql("new_cnt", "new_sum", "new_sum2", is_sample=False, is_stddev=False)
        self.assertIn("NULLIF(new_cnt, 0)", pg_var_pop)

        # 3. MySQL safe guards
        compiler_mysql = IncrementalViewCompiler(schemas, plan.view_spec, dialect=Dialect.MYSQL, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
        my_var = compiler_mysql._variance_formula_sql("c", "s", "s2", is_sample=True, is_stddev=False)
        self.assertIn("NULLIF(c, 0)", my_var)
        self.assertIn("NULLIF((c - 1), 0)", my_var)

        # 4. SQLite safe guards
        compiler_sqlite = IncrementalViewCompiler(schemas, plan.view_spec, dialect=Dialect.SQLITE)
        sq_var = compiler_sqlite._variance_formula_sql("c", "s", "s2", is_sample=True, is_stddev=False)
        self.assertIn("NULLIF((c - 1), 0)", sq_var)

    def test_sqlite_live_sample_variance_evaluation_boundaries(self):
        """Live evaluation of variance/stddev at N=0, N=1, N=2 rows without division-by-zero errors."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE sample_data (id INTEGER PRIMARY KEY, grp INTEGER, val REAL);")

        view_sql = (
            "CREATE VIEW mv_stats AS "
            "SELECT grp, "
            "       COUNT(val) AS cnt, "
            "       SUM(val) AS total, "
            "       VAR_POP(val) AS v_pop, "
            "       VAR_SAMP(val) AS v_samp, "
            "       STDDEV_POP(val) AS s_pop, "
            "       STDDEV_SAMP(val) AS s_samp "
            "FROM sample_data GROUP BY grp;"
        )

        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        manager.register_view(view_sql, populate=True)

        # 1. N = 0 rows: query view, must return empty (no groups)
        rows = conn.execute("SELECT * FROM mv_stats;").fetchall()
        self.assertEqual(len(rows), 0)

        # 2. Insert 1 row: N = 1 (denom = 0 for sample variance)
        conn.execute("INSERT INTO sample_data VALUES (1, 100, 42.0);")
        row = conn.execute("SELECT cnt, total, v_pop, v_samp, s_pop, s_samp FROM mv_stats WHERE grp = 100;").fetchone()
        self.assertIsNotNone(row)
        cnt, total, v_pop, v_samp, s_pop, s_samp = row
        self.assertEqual(cnt, 1)
        self.assertEqual(total, 42.0)
        self.assertEqual(v_pop, 0.0)
        self.assertEqual(s_pop, 0.0)
        self.assertIsNone(v_samp, "VAR_SAMP with N=1 must be NULL, not division by zero")
        self.assertIsNone(s_samp, "STDDEV_SAMP with N=1 must be NULL, not division by zero")

        # 3. Insert 2nd row: N = 2
        conn.execute("INSERT INTO sample_data VALUES (2, 100, 48.0);")
        row = conn.execute("SELECT cnt, total, v_pop, v_samp, s_pop, s_samp FROM mv_stats WHERE grp = 100;").fetchone()
        cnt, total, v_pop, v_samp, s_pop, s_samp = row
        self.assertEqual(cnt, 2)
        self.assertEqual(total, 90.0)
        # vals: 42, 48. mean = 45. diffs: -3, +3. sq diffs: 9, 9.
        # sum sq diffs = 18.
        # pop var = 18 / 2 = 9.0, pop stddev = 3.0
        # samp var = 18 / 1 = 18.0, samp stddev = sqrt(18) = 4.24264...
        self.assertEqual(v_pop, 9.0)
        self.assertEqual(s_pop, 3.0)
        self.assertEqual(v_samp, 18.0)
        self.assertEqual(s_samp, math.sqrt(18.0))

        # 4. Delete 1 row: back to N = 1
        conn.execute("DELETE FROM sample_data WHERE id = 2;")
        row = conn.execute("SELECT cnt, total, v_pop, v_samp, s_pop, s_samp FROM mv_stats WHERE grp = 100;").fetchone()
        cnt, total, v_pop, v_samp, s_pop, s_samp = row
        self.assertEqual(cnt, 1)
        self.assertEqual(total, 42.0)
        self.assertEqual(v_pop, 0.0)
        self.assertIsNone(v_samp)
        self.assertIsNone(s_samp)

        # 5. Delete last row: back to N = 0
        conn.execute("DELETE FROM sample_data WHERE id = 1;")
        rows = conn.execute("SELECT * FROM mv_stats;").fetchall()
        self.assertEqual(len(rows), 0)
        conn.close()


class TestMySQLRowAliasUpsertCodegen(unittest.TestCase):
    """Verifies P1 Bug #13: modern row-alias upsert syntax in MySQL 8.0.19+ vs MySQL 5.7 vs MariaDB."""

    def setUp(self):
        self.schema_sql = (
            "CREATE TABLE sales (\n"
            "    id INT PRIMARY KEY,\n"
            "    dept VARCHAR(32) NOT NULL,\n"
            "    region VARCHAR(32) NOT NULL,\n"
            "    amount DECIMAL(10, 2) NOT NULL\n"
            ");"
        )
        self.schemas = parse_table_schema(self.schema_sql)

    def test_mysql_84_and_97_row_alias_projection_views(self):
        """MySQL >= 8.0.19 emits VALUES (...) AS __mv_new ON DUPLICATE KEY UPDATE col = __mv_new.col."""
        view_sql = "CREATE VIEW mv_proj AS SELECT id, dept, region, amount FROM sales;"
        for ver_str in ("mysql:8.0.20", "mysql:8.4.0", "mysql:9.7.0"):
            plan = generate_materialized_view(
                self.schemas,
                view_sql,
                dialect=Dialect.MYSQL,
                engine_version=ver_str,
                ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
            )
            trigger_sql = "\n".join(plan.trigger_ddls)
            self.assertIn("AS __mv_new", trigger_sql, f"Expected row alias in {ver_str}")
            self.assertIn("__mv_new.", trigger_sql, f"Expected __mv_new column reference in {ver_str}")
            self.assertNotIn("VALUES(", trigger_sql, f"Deprecated VALUES() found in {ver_str}")

    def test_mysql_84_and_97_row_alias_union_all_views(self):
        """MySQL >= 8.0.19 emits SELECT * FROM (...) AS __mv_new for UNION ALL triggers."""
        schema_sql = (
            "CREATE TABLE branch_a (id INT PRIMARY KEY, val INT NOT NULL);\n"
            "CREATE TABLE branch_b (id INT PRIMARY KEY, val INT NOT NULL);"
        )
        schemas = parse_table_schema(schema_sql)
        view_sql = "CREATE VIEW mv_union AS SELECT id, val FROM branch_a UNION ALL SELECT id, val FROM branch_b;"
        for ver_str in ("mysql:8.4.0", "mysql:9.7.0"):
            plan = generate_materialized_view(
                schemas,
                view_sql,
                dialect=Dialect.MYSQL,
                engine_version=ver_str,
                ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
            )
            trigger_sql = "\n".join(plan.trigger_ddls)
            self.assertIn("AS __mv_new", trigger_sql, f"Expected row alias in {ver_str}")
            self.assertIn("__mv_new.val", trigger_sql, f"Expected __mv_new.val in {ver_str}")
            self.assertNotIn("VALUES(", trigger_sql, f"Deprecated VALUES() found in {ver_str}")

    def test_mysql_84_and_97_backfill_chunk_upsert_derived_table_alias(self):
        """MySQL >= 8.0.19 chunk backfill wraps SELECT in derived table AS __mv_src."""
        view_sql = "CREATE VIEW mv_agg AS SELECT dept, SUM(amount) AS total FROM sales GROUP BY dept;"
        for ver_str in ("mysql:8.4.0", "mysql:9.7.0"):
            plan = generate_materialized_view(
                self.schemas,
                view_sql,
                dialect=Dialect.MYSQL,
                engine_version=ver_str,
                incremental_backfill=True,
                ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
            )
            self.assertIsNotNone(plan.incremental_backfill_spec)
            chunk_sql = "\n".join(plan.incremental_backfill_spec.chunk_sql_templates)
            self.assertIn("AS __mv_src", chunk_sql, f"Expected derived table alias in {ver_str}")
            self.assertIn("__mv_src.", chunk_sql, f"Expected __mv_src column reference in {ver_str}")
            self.assertNotIn("VALUES(", chunk_sql, f"Deprecated VALUES() found in {ver_str}")

    def test_mysql_84_and_97_incremental_populate_statements_use_mv_init_alias(self):
        """MySQL >= 8.0.19 get_populate_statements uses __mv_src (pk_range_fold) or __mv_init (offset_batch) without VALUES()."""
        # Non-INT PK projection exercises offset_batch (__mv_init)
        schemas_str_pk = parse_table_schema(
            "CREATE TABLE sales_str (id VARCHAR(32) PRIMARY KEY, dept VARCHAR(32), amount INT);"
        )
        view_str = "CREATE VIEW mv_proj_str AS SELECT id, dept, amount FROM sales_str;"
        # INT PK projection exercises pk_range_fold (__mv_src)
        view_int = "CREATE VIEW mv_proj AS SELECT id, dept, amount FROM sales;"
        for ver_str in ("mysql:8.4.0", "mysql:9.7.0"):
            plan_str = generate_materialized_view(
                schemas_str_pk,
                view_str,
                dialect=Dialect.MYSQL,
                engine_version=ver_str,
                incremental_backfill=True,
                ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
            )
            pop_str = "\n".join(plan_str.get_populate_statements(incremental_backfill=True))
            self.assertIn("__mv_init.", pop_str, f"Expected __mv_init column reference in {ver_str}")
            self.assertNotIn("VALUES(", pop_str, f"Deprecated VALUES() found in {ver_str}")

            plan_int = generate_materialized_view(
                self.schemas,
                view_int,
                dialect=Dialect.MYSQL,
                engine_version=ver_str,
                incremental_backfill=True,
                ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
            )
            pop_int = "\n".join(plan_int.get_populate_statements(incremental_backfill=True))
            self.assertIn("AS __mv_src", pop_int, f"Expected __mv_src derived alias in {ver_str}")
            self.assertIn("__mv_src.", pop_int, f"Expected __mv_src column reference in {ver_str}")
            self.assertNotIn("__mv_new.", pop_int, f"Dangling __mv_new in chunk SELECT in {ver_str}")
            self.assertNotIn("VALUES(", pop_int, f"Deprecated VALUES() found in {ver_str}")

    def test_mysql_57_retains_legacy_values_syntax(self):
        """MySQL 5.7 (which lacks row alias upsert) must retain VALUES() for compatibility."""
        view_sql = "CREATE VIEW mv_proj AS SELECT id, dept, region, amount FROM sales;"
        plan = generate_materialized_view(
            self.schemas,
            view_sql,
            dialect=Dialect.MYSQL,
            engine_version="mysql:5.7.0",
            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        trigger_sql = "\n".join(plan.trigger_ddls)
        self.assertIn("VALUES(", trigger_sql, "Expected legacy VALUES() in MySQL 5.7")
        self.assertNotIn("AS __mv_new", trigger_sql, "MySQL 5.7 must not emit modern row alias")

    def test_mariadb_1011_and_118_retains_values_syntax(self):
        """MariaDB does not support MySQL row alias upsert syntax and must retain VALUES()."""
        view_sql = "CREATE VIEW mv_proj AS SELECT id, dept, region, amount FROM sales;"
        for ver_str in ("mariadb:10.11.0", "mariadb:11.8.0", "mariadb:12.3.0"):
            plan = generate_materialized_view(
                self.schemas,
                view_sql,
                dialect=Dialect.MYSQL,
                engine_version=ver_str,
                ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
            )
            trigger_sql = "\n".join(plan.trigger_ddls)
            self.assertIn("VALUES(", trigger_sql, f"Expected VALUES() in {ver_str}")
            self.assertNotIn("AS __mv_new", trigger_sql, f"{ver_str} must not emit MySQL row alias")


class TestPostgresMergeAndFillfactor(unittest.TestCase):
    """Verifies PostgreSQL 15+ MERGE codegen and configurable FILLFACTOR on backing tables."""

    def setUp(self):
        self.schema_sql = (
            "CREATE TABLE events (\n"
            "    id INT PRIMARY KEY,\n"
            "    category VARCHAR(64) NOT NULL,\n"
            "    metric DOUBLE PRECISION NOT NULL\n"
            ");"
        )
        self.schemas = parse_table_schema(self.schema_sql)

    def test_postgres_15_and_18_uses_merge_for_chunk_backfill(self):
        """PostgreSQL 15+ uses atomic MERGE INTO ... USING (...) AS __mv_src for chunk backfill."""
        view_sql = "CREATE VIEW mv_events AS SELECT category, SUM(metric) AS total FROM events GROUP BY category;"
        for ver_str in ("postgres:15.0", "postgres:16.0", "postgres:17.0", "postgres:18.0"):
            plan = generate_materialized_view(
                self.schemas,
                view_sql,
                dialect=Dialect.POSTGRES,
                engine_version=ver_str,
                incremental_backfill=True, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
            )
            self.assertIsNotNone(plan.incremental_backfill_spec)
            chunk_sql = "\n".join(plan.incremental_backfill_spec.chunk_sql_templates)
            self.assertIn("MERGE INTO", chunk_sql, f"Expected MERGE INTO in {ver_str}")
            self.assertIn("USING (", chunk_sql, f"Expected USING (...) in {ver_str}")
            self.assertIn("AS __mv_src", chunk_sql, f"Expected source alias __mv_src in {ver_str}")
            self.assertIn("WHEN MATCHED THEN", chunk_sql, f"Expected WHEN MATCHED in {ver_str}")
            self.assertIn("WHEN NOT MATCHED THEN", chunk_sql, f"Expected WHEN NOT MATCHED in {ver_str}")
            self.assertNotIn("ON CONFLICT", chunk_sql, f"PostgreSQL 15+ chunk backfill should use MERGE, not ON CONFLICT")

    def test_postgres_14_retains_on_conflict_for_chunk_backfill(self):
        """PostgreSQL 14 retains ON CONFLICT DO UPDATE for chunk backfill."""
        view_sql = "CREATE VIEW mv_events AS SELECT category, SUM(metric) AS total FROM events GROUP BY category;"
        plan = generate_materialized_view(
            self.schemas,
            view_sql,
            dialect=Dialect.POSTGRES,
            engine_version="postgres:14.0",
            incremental_backfill=True, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        self.assertIsNotNone(plan.incremental_backfill_spec)
        chunk_sql = "\n".join(plan.incremental_backfill_spec.chunk_sql_templates)
        self.assertIn("ON CONFLICT", chunk_sql, "Expected ON CONFLICT in PostgreSQL 14")
        self.assertNotIn("MERGE INTO", chunk_sql, "PostgreSQL 14 does not support MERGE")

    def test_postgres_15_populate_statements_use_merge(self):
        """PostgreSQL 15+ batched initial populate loop uses MERGE INTO."""
        view_sql = "CREATE VIEW mv_events AS SELECT category, SUM(metric) AS total FROM events GROUP BY category;"
        plan = generate_materialized_view(
            self.schemas,
            view_sql,
            dialect=Dialect.POSTGRES,
            engine_version="postgres:15.0",
            incremental_backfill=True, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        pop_stmts = plan.get_populate_statements(incremental_backfill=True)
        pop_sql = "\n".join(pop_stmts)
        self.assertIn("MERGE INTO", pop_sql, "Expected MERGE INTO in PG 15 populate statements")
        self.assertIn("WHEN MATCHED THEN", pop_sql)
        self.assertIn("WHEN NOT MATCHED THEN", pop_sql)

    def test_postgres_fillfactor_on_backing_tables_only(self):
        """WITH (fillfactor = 80) is applied to backing store tables, NOT aux tables or readonly views."""
        schema_sql = "CREATE TABLE items (id INT PRIMARY KEY, grp INT, tag VARCHAR(32));"
        schemas = parse_table_schema(schema_sql)
        # View with distinct aggregate produces both backing table and distinct auxiliary table
        view_sql = "CREATE VIEW mv_items AS SELECT grp, COUNT(DISTINCT tag) AS d_cnt FROM items GROUP BY grp;"

        plan = generate_materialized_view(
            schemas,
            view_sql,
            dialect=Dialect.POSTGRES,
            engine_version="postgres:15.0",
            pg_fillfactor=80,
        )

        # Backing table must have fillfactor = 80
        self.assertIn("WITH (fillfactor = 80)", plan.backing_table_ddl)
        # Auxiliary tables must NOT have fillfactor
        for aux_ddl in plan.auxiliary_table_ddls:
            self.assertNotIn("fillfactor", aux_ddl, f"Aux table should not have fillfactor: {aux_ddl}")
        # Reader view must NOT have fillfactor
        self.assertNotIn("fillfactor", plan.reader_view_ddl or "")

    def test_postgres_configurable_fillfactor(self):
        """pg_fillfactor parameter is configurable and respects None."""
        view_sql = "CREATE VIEW mv_events AS SELECT category, SUM(metric) AS total FROM events GROUP BY category;"

        # Configurable to 90
        plan90 = generate_materialized_view(
            self.schemas,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_fillfactor=90, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        self.assertIn("WITH (fillfactor = 90)", plan90.backing_table_ddl)

        # Disabled via None
        plan_none = generate_materialized_view(
            self.schemas,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_fillfactor=None, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        self.assertNotIn("fillfactor", plan_none.backing_table_ddl)


class TestSQLiteWithoutRowidAndReturning(unittest.TestCase):
    """Verifies SQLite clustered composite-PK WITHOUT ROWID backing/aux tables & RETURNING in drop_view."""

    def test_sqlite_composite_pk_backing_tables_emit_without_rowid(self):
        """SQLite backing tables with composite PK (len >= 2) and NOT NULL columns emit WITHOUT ROWID."""
        schema_sql = (
            "CREATE TABLE order_items (\n"
            "    order_id INT NOT NULL,\n"
            "    item_id INT NOT NULL,\n"
            "    price REAL NOT NULL,\n"
            "    PRIMARY KEY (order_id, item_id)\n"
            ");"
        )
        schemas = parse_table_schema(schema_sql)

        # 1. Group-by view on composite key (order_id, item_id)
        view_sql = (
            "CREATE VIEW mv_comp AS "
            "SELECT order_id, item_id, SUM(price) AS total "
            "FROM order_items GROUP BY order_id, item_id;"
        )
        plan = generate_materialized_view(schemas, view_sql, dialect=Dialect.SQLITE)
        self.assertIn("WITHOUT ROWID", plan.backing_table_ddl)

        # 2. Distinct aux table with composite PK also emits WITHOUT ROWID
        view_dist_sql = (
            "CREATE VIEW mv_dist AS "
            "SELECT order_id, COUNT(DISTINCT item_id) AS d_items "
            "FROM order_items GROUP BY order_id;"
        )
        plan_dist = generate_materialized_view(schemas, view_dist_sql, dialect=Dialect.SQLITE)
        self.assertTrue(len(plan_dist.auxiliary_table_ddls) > 0)
        for aux_ddl in plan_dist.auxiliary_table_ddls:
            self.assertIn("WITHOUT ROWID", aux_ddl)

    def test_sqlite_single_pk_backing_tables_do_not_emit_without_rowid(self):
        """SQLite backing tables with single-column PK retain standard rowid B-tree."""
        schema_sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, dept TEXT, score REAL);"
        schemas = parse_table_schema(schema_sql)

        # View grouped by single column
        view_sql = "CREATE VIEW mv_users AS SELECT dept, SUM(score) AS total FROM users GROUP BY dept;"
        plan = generate_materialized_view(schemas, view_sql, dialect=Dialect.SQLITE)
        self.assertNotIn("WITHOUT ROWID", plan.backing_table_ddl)

    def test_sqlite_live_without_rowid_dml_and_drift_check(self):
        """Live SQLite test: WITHOUT ROWID backing table maintains DML and passes check_schema_drift()."""
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE sales (\n"
            "    store_id INTEGER NOT NULL,\n"
            "    dept_id INTEGER NOT NULL,\n"
            "    amount REAL NOT NULL,\n"
            "    PRIMARY KEY (store_id, dept_id)\n"
            ");"
        )

        view_sql = (
            "CREATE VIEW mv_sales AS "
            "SELECT store_id, dept_id, SUM(amount) AS total, COUNT(*) AS cnt "
            "FROM sales GROUP BY store_id, dept_id;"
        )

        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        plan = manager.register_view(view_sql, populate=True)

        # Verify backing table in sqlite_master has WITHOUT ROWID
        master_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?;",
            (plan.backing_table_name,),
        ).fetchone()
        self.assertIsNotNone(master_row)
        self.assertIn("WITHOUT ROWID", master_row[0])

        # Perform live DML: INSERT, UPDATE, DELETE
        conn.execute("INSERT INTO sales VALUES (1, 10, 100.0);")
        conn.execute("INSERT INTO sales VALUES (1, 20, 200.0);")
        conn.execute("INSERT INTO sales VALUES (2, 10, 300.0);")

        rows = conn.execute("SELECT store_id, dept_id, total, cnt FROM mv_sales ORDER BY store_id, dept_id;").fetchall()
        self.assertEqual(len(rows), 3)

        # Update an existing row
        conn.execute("UPDATE sales SET amount = 150.0 WHERE store_id = 1 AND dept_id = 10;")
        val = conn.execute("SELECT total FROM mv_sales WHERE store_id = 1 AND dept_id = 10;").fetchone()[0]
        self.assertEqual(val, 150.0)

        # Delete a row
        conn.execute("DELETE FROM sales WHERE store_id = 2 AND dept_id = 10;")
        rows_after = conn.execute("SELECT * FROM mv_sales WHERE store_id = 2 AND dept_id = 10;").fetchall()
        self.assertEqual(len(rows_after), 0)

        # Verify check_schema_drift() returns "ok"
        drift = manager.check_schema_drift()
        self.assertEqual(drift.get("mv_sales"), "ok")
        conn.close()

    def test_sqlite_returning_in_drop_view(self):
        """SQLite 3.35+ drop_view uses RETURNING view_name to confirm catalog deletion."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INT);")

        manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
        manager.register_view("CREATE VIEW mv_test AS SELECT id, v FROM t;", populate=True)

        # drop_view on existing view returns True
        res = manager.drop_view("mv_test")
        self.assertTrue(res)

        # Catalog record must be gone
        cat_count = conn.execute(f"SELECT COUNT(*) FROM {manager.catalog_table} WHERE view_name = 'mv_test';").fetchone()[0]
        self.assertEqual(cat_count, 0)

        # drop_view on non-existent view returns False
        res_nonexistent = manager.drop_view("mv_nonexistent")
        self.assertFalse(res_nonexistent)
        conn.close()


class TestDerivedAliasContractValidation(unittest.TestCase):
    """Validates via sqlglot AST that every __mv_src / __mv_new ref matches an inner SELECT alias."""

    def _assert_subquery_aliases_cover_refs(self, sql_stmt: str, alias_name: str, sg_dialect: str) -> None:
        import re
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse_one(sql_stmt.replace("%s", "0"), read=sg_dialect)
        subq = None
        for node in parsed.find_all(exp.Subquery):
            if node.alias == alias_name:
                subq = node
                break
        self.assertIsNotNone(subq, f"Missing subquery alias {alias_name} in:\n{sql_stmt}")
        sel = subq.this
        self.assertIsInstance(sel, exp.Select, f"Expected SELECT inside {alias_name}:\n{sql_stmt}")
        projected_aliases = {expr.alias for expr in sel.expressions if expr.alias}
        refs = set(re.findall(rf"\b{re.escape(alias_name)}\.([A-Za-z_][A-Za-z0-9_]*|`[^`]+`|\"[^\"]+\")", sql_stmt))
        cleaned_refs = {r.strip('`"') for r in refs}
        self.assertGreater(len(cleaned_refs), 0, f"Expected column references to {alias_name} in:\n{sql_stmt}")
        self.assertTrue(
            cleaned_refs.issubset(projected_aliases),
            f"Missing inner aliases for {alias_name}: refs={cleaned_refs}, aliases={projected_aliases}\nSQL:\n{sql_stmt}",
        )

    def test_mysql_84_derived_table_aliases_and_global_having_placement(self) -> None:
        schemas = parse_table_schema(
            "CREATE TABLE items (id INT PRIMARY KEY, grp VARCHAR(32) NOT NULL, tag VARCHAR(32), val INT NOT NULL);"
        )
        # 1. pk_range_fold
        p_grp = generate_materialized_view(
            schemas,
            "CREATE VIEW mv_grp AS SELECT grp, SUM(val) AS tot FROM items GROUP BY grp;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        self._assert_subquery_aliases_cover_refs(p_grp.incremental_backfill_spec.chunk_sql_templates[0], "__mv_src", "mysql")

        # 2. global aggregate incremental populate HAVING placement
        p_glob = generate_materialized_view(
            schemas,
            "CREATE VIEW mv_glob AS SELECT SUM(val) AS tot FROM items;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        pop_sql = "\n".join(p_glob.get_populate_statements(incremental_backfill=True))
        self.assertIn("HAVING COUNT(*) > 0\n) AS __mv_src\nON DUPLICATE KEY UPDATE", pop_sql)
        self.assertNotIn(") AS __mv_src\nHAVING COUNT(*) > 0", pop_sql)

        # 3. group_key_batch distinct aux + main
        p_dist = generate_materialized_view(
            schemas,
            "CREATE VIEW mv_dist AS SELECT grp, COUNT(DISTINCT tag) AS dc, SUM(val) AS sv FROM items GROUP BY grp;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        for tmpl in p_dist.incremental_backfill_spec.chunk_sql_templates:
            self._assert_subquery_aliases_cover_refs(tmpl.format(placeholders="'A'"), "__mv_src", "mysql")

    def test_postgres_15_merge_using_aliases_across_pk_and_group_key_batch(self) -> None:
        schemas = parse_table_schema(
            "CREATE TABLE items (id INT PRIMARY KEY, grp VARCHAR(32) NOT NULL, tag VARCHAR(32), val INT NOT NULL);"
        )
        for view_sql, fmt_ph in (
            ("CREATE VIEW mv_grp AS SELECT grp, SUM(val) AS tot FROM items GROUP BY grp;", False),
            ("CREATE VIEW mv_dist AS SELECT grp, COUNT(DISTINCT tag) AS dc, SUM(val) AS sv FROM items GROUP BY grp;", True),
        ):
            plan = generate_materialized_view(
                schemas,
                view_sql,
                dialect=Dialect.POSTGRES,
                engine_version="postgres:15.0",
                incremental_backfill=True,
            )
            for tmpl in plan.incremental_backfill_spec.chunk_sql_templates:
                sql = tmpl.format(placeholders="'A'") if fmt_ph else tmpl
                self._assert_subquery_aliases_cover_refs(sql, "__mv_src", "postgres")

    def test_mysql_84_projection_chunk_upsert_aliases(self) -> None:
        schemas = parse_table_schema(
            "CREATE TABLE items (id INT PRIMARY KEY, grp VARCHAR(32) NOT NULL, val INT NOT NULL);"
        )
        schemas_no_pk = parse_table_schema(
            "CREATE TABLE logs (grp VARCHAR(32) NOT NULL, val INT NOT NULL);"
        )
        p_proj = generate_materialized_view(
            schemas,
            "CREATE VIEW mv_proj_pk AS SELECT id, grp, val FROM items;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        self._assert_subquery_aliases_cover_refs(p_proj.incremental_backfill_spec.chunk_sql_templates[0], "__mv_src", "mysql")
        p_proj_bag = generate_materialized_view(
            schemas,
            "CREATE VIEW mv_proj_bag AS SELECT grp, val FROM items;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        self._assert_subquery_aliases_cover_refs(p_proj_bag.incremental_backfill_spec.chunk_sql_templates[0], "__mv_src", "mysql")
        p_no_pk = generate_materialized_view(
            schemas_no_pk,
            "CREATE VIEW mv_proj_npk AS SELECT grp, val FROM logs;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
        )
        pop_sql = "\n".join(p_no_pk.get_populate_statements(incremental_backfill=True))
        self.assertIn("AS __mv_init", pop_sql)
        self.assertNotIn("VALUES(", pop_sql)


class TestNativeEngineCodegenExecution(unittest.TestCase):
    """Live PostgreSQL & MySQL/MariaDB native codegen verification when DSN env vars are set."""

    def test_native_mysql_row_alias_upsert_and_codegen_dml(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; skipping live MySQL/MariaDB codegen execution test")

        import pymysql

        parsed = urlparse(dsn)
        conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/"),
            autocommit=False,
        )
        base_tbl = "t_cg_my_sales"
        mv_proj = "t_cg_my_mv_proj"
        mv_agg = "t_cg_my_mv_agg"
        mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
        cur = conn.cursor()
        try:
            for vn in (mv_proj, mv_agg):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
                cur.execute(f"DROP VIEW IF EXISTS {vn};")
                cur.execute(f"DROP TABLE IF EXISTS {mgr.object_prefix}store_{vn};")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
            cur.execute(
                f"CREATE TABLE {base_tbl} (id INT PRIMARY KEY, dept VARCHAR(32) NOT NULL, tag VARCHAR(32), amount INT NOT NULL);"
            )
            cur.executemany(
                f"INSERT INTO {base_tbl} VALUES (%s, %s, %s, %s);",
                [(1, "eng", "t1", 10), (2, "eng", "t1", 30), (3, "eng", "t2", 20), (4, "ops", "t1", 50), (5, "hr", None, 100)],
            )
            conn.commit()

            p_proj = mgr.register_view(
                f"CREATE VIEW {mv_proj} AS SELECT id, dept, amount FROM {base_tbl};",
                populate=True,
                incremental_backfill=True,
                batch_size=2,
            )
            p_agg = mgr.register_view(
                f"CREATE VIEW {mv_agg} AS SELECT dept, SUM(amount) AS s, STDDEV_SAMP(amount) AS sd, COUNT(DISTINCT tag) AS dc FROM {base_tbl} GROUP BY dept;",
                populate=True,
                incremental_backfill=True,
                batch_size=2,
            )

            ev = mgr.engine_version
            if ev.supports_mysql_row_alias_upsert:
                self.assertIn("AS __mv_new", "\n".join(p_proj.trigger_ddls))
                self.assertIn("AS __mv_src", "\n".join(p_proj.incremental_backfill_spec.chunk_sql_templates))
                self.assertIn("AS __mv_init", "\n".join(p_agg.get_populate_statements(incremental_backfill=True)))
                self.assertNotIn("VALUES(", "\n".join(p_proj.trigger_ddls))
            else:
                self.assertIn("VALUES(", "\n".join(p_proj.trigger_ddls))

            cur.execute(f"INSERT INTO {base_tbl} VALUES (6, 'ops', 't2', 90), (7, 'hr', 't9', 200);")
            cur.execute(f"UPDATE {base_tbl} SET amount = 40 WHERE id = 1;")
            cur.execute(f"DELETE FROM {base_tbl} WHERE id = 5;")
            conn.commit()

            cur.execute(f"SELECT id, dept, amount FROM {mv_proj} ORDER BY id;")
            mv_proj_rows = cur.fetchall()
            cur.execute(f"SELECT id, dept, amount FROM {base_tbl} ORDER BY id;")
            self.assertEqual(mv_proj_rows, cur.fetchall())

            cur.execute(f"SELECT dept, s, sd, dc FROM {mv_agg} ORDER BY dept;")
            mv_agg_rows = cur.fetchall()
            # Small single-table view, no secondary index: the original view
            # reads in the recompute's order (PostgreSQL: pinned plan).
            pin_native_plan(cur, mgr.dialect)
            cur.execute(
                f"SELECT dept, SUM(amount) AS s, STDDEV_SAMP(amount) AS sd, COUNT(DISTINCT tag) AS dc FROM {base_tbl} GROUP BY dept ORDER BY dept;"
            )
            ora_agg_rows = cur.fetchall()
            self.assertEqual(len(mv_agg_rows), len(ora_agg_rows))
            for r_mv, r_ora in zip(mv_agg_rows, ora_agg_rows):
                self.assertEqual(r_mv[0], r_ora[0])
                self.assertEqual(r_mv[1], r_ora[1])
                self.assertEqual(r_mv[3], r_ora[3])
                if r_ora[2] is None:
                    self.assertIsNone(r_mv[2])
                else:
                    self.assertEqual(r_mv[2], r_ora[2])
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            for vn in (mv_proj, mv_agg):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
            try:
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
                conn.commit()
            except Exception:
                pass
            cur.close()
            conn.close()

    def test_native_postgres_merge_and_nullif_variance_execution(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; skipping live PostgreSQL codegen execution test")

        import psycopg2

        conn = psycopg2.connect(dsn)
        base_tbl = "t_cg_pg_sales"
        mv_proj = "t_cg_pg_mv_proj"
        mv_agg = "t_cg_pg_mv_agg"
        mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
        cur = conn.cursor()
        try:
            for vn in (mv_proj, mv_agg):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
                cur.execute(f"DROP VIEW IF EXISTS {vn} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {mgr.object_prefix}store_{vn} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
            cur.execute(
                f"CREATE TABLE {base_tbl} (id INT PRIMARY KEY, dept TEXT NOT NULL, tag TEXT, amount DOUBLE PRECISION NOT NULL);"
            )
            cur.executemany(
                f"INSERT INTO {base_tbl} VALUES (%s, %s, %s, %s);",
                [(1, "eng", "t1", 10.0), (2, "eng", "t1", 30.0), (3, "eng", "t2", 20.0), (4, "ops", "t1", 50.0), (5, "hr", None, 100.0)],
            )
            conn.commit()

            p_proj = mgr.register_view(
                f"CREATE VIEW {mv_proj} AS SELECT id, dept, amount FROM {base_tbl};",
                populate=True,
                incremental_backfill=True,
                batch_size=2,
            )
            p_agg = mgr.register_view(
                f"CREATE VIEW {mv_agg} AS SELECT dept, SUM(amount) AS s, STDDEV_SAMP(amount) AS sd, COUNT(DISTINCT tag) AS dc FROM {base_tbl} GROUP BY dept;",
                populate=True,
                incremental_backfill=True,
                batch_size=2,
            )

            if mgr.engine_version.supports_pg_merge:
                self.assertIn("MERGE INTO", "\n".join(p_agg.incremental_backfill_spec.chunk_sql_templates))
                self.assertIn("fillfactor = 80", p_agg.backing_table_ddl)

            cur.execute(f"INSERT INTO {base_tbl} VALUES (6, 'ops', 't2', 90.0), (7, 'solo', 's1', 42.0);")
            cur.execute(f"UPDATE {base_tbl} SET amount = 40.0 WHERE id = 1;")
            cur.execute(f"DELETE FROM {base_tbl} WHERE id = 5;")
            conn.commit()

            cur.execute(f"SELECT id, dept, amount FROM {mv_proj} ORDER BY id;")
            mv_proj_rows = cur.fetchall()
            cur.execute(f"SELECT id, dept, amount FROM {base_tbl} ORDER BY id;")
            self.assertEqual(mv_proj_rows, cur.fetchall())

            cur.execute(f"SELECT dept, s, sd, dc FROM {mv_agg} ORDER BY dept;")
            mv_agg_rows = cur.fetchall()
            # Small single-table view, no secondary index: the original view
            # reads in the recompute's order (PostgreSQL: pinned plan).
            pin_native_plan(cur, mgr.dialect)
            cur.execute(
                f"SELECT dept, SUM(amount) AS s, STDDEV_SAMP(amount) AS sd, COUNT(DISTINCT tag) AS dc FROM {base_tbl} GROUP BY dept ORDER BY dept;"
            )
            ora_agg_rows = cur.fetchall()
            self.assertEqual(len(mv_agg_rows), len(ora_agg_rows))
            for r_mv, r_ora in zip(mv_agg_rows, ora_agg_rows):
                self.assertEqual(r_mv[0], r_ora[0])
                self.assertEqual(r_mv[1], r_ora[1])
                self.assertEqual(r_mv[3], r_ora[3])
                if r_ora[2] is None:
                    self.assertIsNone(r_mv[2])
                else:
                    self.assertEqual(r_mv[2], r_ora[2])
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            for vn in (mv_proj, mv_agg):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
            try:
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
                conn.commit()
            except Exception:
                pass
            cur.close()
            conn.close()

    def test_native_mysql_backfill_unreferenced_pk_update(self) -> None:
        """Native MySQL/MariaDB: mid-backfill UPDATE changing an unreferenced integer
        PK across watermark boundaries on an independent connection stays exact for
        SUM-only grouped views, native-PK projections, and aliased incremental populate."""
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; skipping live MySQL/MariaDB unreferenced PK backfill test")
        import threading
        from urllib.parse import urlparse
        import pymysql

        parsed = urlparse(dsn)

        def _connect():
            return pymysql.connect(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 3306,
                user=parsed.username or "root",
                password=parsed.password or "",
                database=parsed.path.lstrip("/"),
                autocommit=False,
            )

        conn = _connect()
        base_tbl = "t_cg_my_unref_pk"
        mv_agg = "v_cg_my_unref_agg"
        mv_proj = "v_cg_my_unref_proj"
        mv_alias = "v_cg_my_unref_alias"
        mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True)
        cur = conn.cursor()
        try:
            for vn in (mv_agg, mv_proj, mv_alias):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
                cur.execute(f"DROP VIEW IF EXISTS {vn};")
                cur.execute(f"DROP TABLE IF EXISTS {mgr.object_prefix}store_{vn};")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
            cur.execute(
                f"CREATE TABLE {base_tbl} (id INT PRIMARY KEY, grp VARCHAR(32) NOT NULL, val INT NOT NULL, note VARCHAR(32));"
            )
            cur.executemany(
                f"INSERT INTO {base_tbl} VALUES (%s, %s, %s, %s);",
                [(i, f"g{i % 3}", i * 10, f"n{i}") for i in range(1, 11)],
            )
            conn.commit()

            agg_sql = f"SELECT grp, SUM(val) AS s FROM {base_tbl} GROUP BY grp"
            proj_sql = f"SELECT val FROM {base_tbl}"
            p_agg = mgr.register_view(f"CREATE VIEW {mv_agg} AS {agg_sql};", populate=False)
            p_proj = mgr.register_view(f"CREATE VIEW {mv_proj} AS {proj_sql};", populate=False)

            def _run_with_moves(plan, moves: tuple[str, ...]) -> None:
                worker_errors: list[BaseException] = []

                def on_progress(batch_idx: int, _rows_done: int) -> None:
                    if batch_idx != 1:
                        return
                    done = threading.Event()

                    def worker() -> None:
                        try:
                            c2 = _connect()
                            try:
                                cur2 = c2.cursor()
                                for st in moves:
                                    cur2.execute(st)
                                c2.commit()
                                cur2.close()
                            finally:
                                c2.close()
                        except BaseException as exc:
                            worker_errors.append(exc)
                        finally:
                            done.set()

                    th = threading.Thread(target=worker, daemon=True)
                    th.start()
                    self.assertTrue(done.wait(timeout=15), "MySQL worker timed out")
                    th.join(timeout=1)
                    if worker_errors:
                        raise worker_errors[0]

                batches = mgr.backfill_incrementally(
                    plan, batch_size=3, on_progress=on_progress, snapshot_isolation=False
                )
                self.assertGreaterEqual(batches, 2)

            _run_with_moves(
                p_agg,
                (
                    f"UPDATE {base_tbl} SET note = 'x' WHERE id = 3;",
                    f"UPDATE {base_tbl} SET id = 100 WHERE id = 8;",
                    f"UPDATE {base_tbl} SET id = 0 WHERE id = 7;",
                    f"UPDATE {base_tbl} SET id = 7 WHERE id = 2;",
                ),
            )
            _run_with_moves(
                p_proj,
                (
                    f"UPDATE {base_tbl} SET id = 105 WHERE id = 9;",
                    f"UPDATE {base_tbl} SET id = 9 WHERE id = 1;",
                ),
            )

            # Aliased projection incremental populate procedure executes cleanly
            p_alias = mgr.register_view(
                f"CREATE VIEW {mv_alias} AS SELECT s.val FROM {base_tbl} AS s WHERE s.val >= 20;",
                populate=False,
            )
            for stmt in p_alias.get_populate_statements(incremental_backfill=True):
                cur.execute(stmt)
            conn.commit()

            cur.execute(f"SELECT grp, s FROM {mv_agg} ORDER BY grp;")
            got_agg = cur.fetchall()
            cur.execute(f"{agg_sql} ORDER BY grp;")
            self.assertEqual(got_agg, cur.fetchall())

            cur.execute(f"SELECT val FROM {mv_proj} ORDER BY val;")
            got_proj = cur.fetchall()
            cur.execute(f"{proj_sql} ORDER BY val;")
            self.assertEqual(got_proj, cur.fetchall())

            cur.execute(f"SELECT val FROM {mv_alias} ORDER BY val;")
            got_alias = cur.fetchall()
            cur.execute(f"SELECT s.val FROM {base_tbl} AS s WHERE s.val >= 20 ORDER BY s.val;")
            self.assertEqual(got_alias, cur.fetchall())
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            for vn in (mv_agg, mv_proj, mv_alias):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
            try:
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl};")
                conn.commit()
            except Exception:
                pass
            cur.close()
            conn.close()

    def test_native_postgres_backfill_unreferenced_pk_update(self) -> None:
        """Native PostgreSQL: mid-backfill UPDATE changing an unreferenced integer PK
        across watermark boundaries on an independent connection stays exact for
        SUM-only grouped views, native-PK projections, and aliased incremental populate."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; skipping live PostgreSQL unreferenced PK backfill test")
        import threading
        import psycopg2

        conn = psycopg2.connect(dsn)
        base_tbl = "t_cg_pg_unref_pk"
        mv_agg = "v_cg_pg_unref_agg"
        mv_proj = "v_cg_pg_unref_proj"
        mv_alias = "v_cg_pg_unref_alias"
        mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
        cur = conn.cursor()
        try:
            for vn in (mv_agg, mv_proj, mv_alias):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
                cur.execute(f"DROP VIEW IF EXISTS {vn} CASCADE;")
                cur.execute(f"DROP TABLE IF EXISTS {mgr.object_prefix}store_{vn} CASCADE;")
            cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
            cur.execute(
                f"CREATE TABLE {base_tbl} (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, val INTEGER NOT NULL, note TEXT);"
            )
            cur.executemany(
                f"INSERT INTO {base_tbl} VALUES (%s, %s, %s, %s);",
                [(i, f"g{i % 3}", i * 10, f"n{i}") for i in range(1, 11)],
            )
            conn.commit()

            agg_sql = f"SELECT grp, SUM(val) AS s FROM {base_tbl} GROUP BY grp"
            proj_sql = f"SELECT val FROM {base_tbl}"
            p_agg = mgr.register_view(f"CREATE VIEW {mv_agg} AS {agg_sql};", populate=False)
            p_proj = mgr.register_view(f"CREATE VIEW {mv_proj} AS {proj_sql};", populate=False)

            def _run_with_moves(plan, moves: tuple[str, ...]) -> None:
                worker_errors: list[BaseException] = []

                def on_progress(batch_idx: int, _rows_done: int) -> None:
                    if batch_idx != 1:
                        return
                    done = threading.Event()

                    def worker() -> None:
                        try:
                            c2 = psycopg2.connect(dsn)
                            try:
                                cur2 = c2.cursor()
                                for st in moves:
                                    cur2.execute(st)
                                c2.commit()
                                cur2.close()
                            finally:
                                c2.close()
                        except BaseException as exc:
                            worker_errors.append(exc)
                        finally:
                            done.set()

                    th = threading.Thread(target=worker, daemon=True)
                    th.start()
                    self.assertTrue(done.wait(timeout=15), "PostgreSQL worker timed out")
                    th.join(timeout=1)
                    if worker_errors:
                        raise worker_errors[0]

                batches = mgr.backfill_incrementally(
                    plan, batch_size=3, on_progress=on_progress, snapshot_isolation=False
                )
                self.assertGreaterEqual(batches, 2)

            _run_with_moves(
                p_agg,
                (
                    f"UPDATE {base_tbl} SET note = 'x' WHERE id = 3;",
                    f"UPDATE {base_tbl} SET id = 100 WHERE id = 8;",
                    f"UPDATE {base_tbl} SET id = 0 WHERE id = 7;",
                    f"UPDATE {base_tbl} SET id = 7 WHERE id = 2;",
                ),
            )
            _run_with_moves(
                p_proj,
                (
                    f"UPDATE {base_tbl} SET id = 105 WHERE id = 9;",
                    f"UPDATE {base_tbl} SET id = 9 WHERE id = 1;",
                ),
            )

            # Aliased projection incremental populate DO block executes cleanly
            p_alias = mgr.register_view(
                f"CREATE VIEW {mv_alias} AS SELECT s.val FROM {base_tbl} AS s WHERE s.val >= 20;",
                populate=False,
            )
            for stmt in p_alias.get_populate_statements(incremental_backfill=True):
                cur.execute(stmt)
            conn.commit()

            cur.execute(f"SELECT grp, s FROM {mv_agg} ORDER BY grp;")
            got_agg = cur.fetchall()
            cur.execute(f"{agg_sql} ORDER BY grp;")
            self.assertEqual(got_agg, cur.fetchall())

            cur.execute(f"SELECT val FROM {mv_proj} ORDER BY val;")
            got_proj = cur.fetchall()
            cur.execute(f"{proj_sql} ORDER BY val;")
            self.assertEqual(got_proj, cur.fetchall())

            cur.execute(f"SELECT val FROM {mv_alias} ORDER BY val;")
            got_alias = cur.fetchall()
            cur.execute(f"SELECT s.val FROM {base_tbl} AS s WHERE s.val >= 20 ORDER BY s.val;")
            self.assertEqual(got_alias, cur.fetchall())
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            for vn in (mv_agg, mv_proj, mv_alias):
                try:
                    mgr.drop_view(vn)
                except Exception:
                    pass
            try:
                cur.execute(f"DROP TABLE IF EXISTS {base_tbl} CASCADE;")
                conn.commit()
            except Exception:
                pass
            cur.close()
            conn.close()


    def test_postgres_join_views_emit_pre_join_view_lock(self) -> None:
        """PostgreSQL algebraic and group-scoped multi-source join views must create
        their per-view lock table and acquire the exclusive view lock row (__mv_id = 0)
        in a separate statement BEFORE joining their transition delta with other sources,
        while single-FROM-item group-scoped views retain per-group locking, and trigger
        names keep a table-independent firing order across sources."""
        schemas = parse_table_schema(
            "CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL);"
            "CREATE TABLE d (g INT PRIMARY KEY, name TEXT NOT NULL);",
            dialect=Dialect.POSTGRES,
        )

        # 1. Algebraic join view (Tier 1B)
        alg_plan = generate_materialized_view(
            schemas,
            "CREATE VIEW v_alg_j AS SELECT d.name, SUM(t.v) AS s, COUNT(*) AS c "
            "FROM t JOIN d ON t.g = d.g GROUP BY d.name;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(alg_plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        lock_tbl = "__matview_lock_v_alg_j"
        self.assertEqual(
            alg_plan.auxiliary_table_names,
            [lock_tbl, "__matview_stg_v_alg_j__t", "__matview_stg_v_alg_j__d"],
        )
        self.assertTrue(
            any(f"CREATE TABLE {lock_tbl}" in ddl for ddl in alg_plan.auxiliary_table_ddls),
            alg_plan.auxiliary_table_ddls,
        )
        fn_ddls = [
            d
            for d in alg_plan.trigger_ddls
            if "CREATE OR REPLACE FUNCTION __matview_fn_sync_" in d and "_bs()" not in d
        ]
        self.assertEqual(len(fn_ddls), 8)  # 2 sources x (ai, au, ad, trunc)
        for ddl in fn_ddls:
            lock_pos = ddl.find(f"INSERT INTO {lock_tbl} (__mv_id, __mv_n) VALUES (0, 1)")
            self.assertNotEqual(lock_pos, -1, f"missing pre-join view lock in:\n{ddl}")
            store_pos = ddl.find(alg_plan.backing_table_name)
            self.assertNotEqual(store_pos, -1, ddl)
            self.assertLess(lock_pos, store_pos, f"view lock must precede backing-store access:\n{ddl}")

        # 2. Group-scoped multi-source join view (Tier 2): exclusive view lock before kset,
        #    NOT FOR SHARE + pre-lock hash, while keeping scoped DELETE/INSERT.
        t2_join_plan = generate_materialized_view(
            schemas,
            "CREATE VIEW v_t2_lj AS SELECT d.name, SUM(t.v) AS s, COUNT(t.v) AS c "
            "FROM d LEFT JOIN t ON t.g = d.g GROUP BY d.name;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(t2_join_plan.strategy, MaintenanceStrategy.GROUP_SCOPED_DELTA)
        t2_lock_tbl = "__matview_lock_v_t2_lj"
        self.assertEqual(t2_join_plan.auxiliary_table_names, [t2_lock_tbl])
        t2_fn_ddls = [
            d
            for d in t2_join_plan.trigger_ddls
            if "CREATE OR REPLACE FUNCTION __matview_fn_sync_" in d and "_bs()" not in d
        ]
        self.assertEqual(len(t2_fn_ddls), 8)
        for ddl in t2_fn_ddls:
            lock_pos = ddl.find(f"INSERT INTO {t2_lock_tbl} (__mv_id, __mv_n) VALUES (0, 1)")
            self.assertNotEqual(lock_pos, -1, ddl)
            self.assertNotIn("FOR SHARE", ddl)
            del_pos = ddl.find(f"DELETE FROM {t2_join_plan.backing_table_name}")
            self.assertLess(lock_pos, del_pos, ddl)

        # 3. Single-table group-scoped view must use bounded hash buckets (1..1024),
        #    fillfactor = 50, NO FOR SHARE on row 0, and full-bucket locks on TRUNCATE
        #    or non-base-table (subquery) triggers.
        t2_single_plan = generate_materialized_view(
            schemas,
            "CREATE VIEW v_t2_single AS SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids "
            "FROM t GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(t2_single_plan.strategy, MaintenanceStrategy.GROUP_SCOPED_DELTA)
        self.assertTrue(
            any("WITH (fillfactor = 50)" in d for d in t2_single_plan.auxiliary_table_ddls),
            t2_single_plan.auxiliary_table_ddls,
        )
        self.assertTrue(
            any("generate_series(0, 1024)" in d for d in t2_single_plan.auxiliary_table_ddls),
            t2_single_plan.auxiliary_table_ddls,
        )
        single_ai = next(
            d for d in t2_single_plan.trigger_ddls if "CREATE OR REPLACE FUNCTION __matview_fn_sync_v_t2_single__t_ai" in d
        )
        self.assertNotIn("FOR SHARE", single_ai)
        self.assertIn("((hashtextextended(", single_ai)
        self.assertIn("& 1023) + 1)", single_ai)
        self.assertIn("ORDER BY 1", single_ai)
        single_trunc = next(
            d for d in t2_single_plan.trigger_ddls if "CREATE OR REPLACE FUNCTION __matview_fn_sync_v_t2_single__t_trunc" in d
        )
        self.assertIn("generate_series(0, 1024)", single_trunc)
        self.assertIn("ORDER BY 1", single_trunc)

        # 3b. Subquery-filtered single-source group-scoped view: writes to t use bounded
        #     group buckets, while writes to u (full refresh) lock all buckets 0..1024.
        sub_schemas = parse_table_schema(
            "CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL);"
            "CREATE TABLE u (g INT PRIMARY KEY);",
            dialect=Dialect.POSTGRES,
        )
        t2_sub_plan = generate_materialized_view(
            sub_schemas,
            "CREATE VIEW v_t2_sub AS SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids "
            "FROM t WHERE g IN (SELECT g FROM u) GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertTrue(
            any("generate_series(0, 1024)" in d for d in t2_sub_plan.auxiliary_table_ddls),
            t2_sub_plan.auxiliary_table_ddls,
        )
        sub_t_ai = next(
            d for d in t2_sub_plan.trigger_ddls if "CREATE OR REPLACE FUNCTION __matview_fn_sync_v_t2_sub__t_ai" in d
        )
        sub_u_ai = next(
            d for d in t2_sub_plan.trigger_ddls if "CREATE OR REPLACE FUNCTION __matview_fn_sync_v_t2_sub__u_ai" in d
        )
        self.assertNotIn("FOR SHARE", sub_t_ai)
        self.assertIn("& 1023) + 1)", sub_t_ai)
        self.assertIn("generate_series(0, 1024)", sub_u_ai)
        self.assertIn("ORDER BY 1", sub_u_ai)

        # 3c. Self-subquery Tier 2 view (base_table also appears in subquery): never uses
        #     per-group buckets, so it must seed and lock only row 0 (not 0..1024).
        t2_self_sub_plan = generate_materialized_view(
            sub_schemas,
            "CREATE VIEW v_t2_self_sub AS SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids "
            "FROM t WHERE v > (SELECT AVG(v) FROM t) GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertTrue(
            any("VALUES (0, 0)" in d for d in t2_self_sub_plan.auxiliary_table_ddls),
            t2_self_sub_plan.auxiliary_table_ddls,
        )
        self.assertFalse(
            any("generate_series(0, 1024)" in d for d in t2_self_sub_plan.auxiliary_table_ddls),
            t2_self_sub_plan.auxiliary_table_ddls,
        )
        self_sub_ai = next(
            d for d in t2_self_sub_plan.trigger_ddls if "CREATE OR REPLACE FUNCTION __matview_fn_sync_v_t2_self_sub__t_ai" in d
        )
        self.assertIn("VALUES (0, 1)", self_sub_ai)
        self.assertNotIn("generate_series", self_sub_ai)

        # 3d. HAVING and SELECT-list subqueries: external table u uses full-bucket lock while
        #     base table t uses bounded group buckets; self-subquery in HAVING uses row 0 only.
        t2_having_u = generate_materialized_view(
            sub_schemas,
            "CREATE VIEW v_t2_hav_u AS SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids "
            "FROM t GROUP BY g HAVING COUNT(*) > (SELECT COUNT(*) FROM u);",
            dialect=Dialect.POSTGRES,
        )
        self.assertTrue(
            any("generate_series(0, 1024)" in d for d in t2_having_u.auxiliary_table_ddls),
            t2_having_u.auxiliary_table_ddls,
        )
        hav_u_ai = next(
            d for d in t2_having_u.trigger_ddls if "CREATE OR REPLACE FUNCTION __matview_fn_sync_v_t2_hav_u__u_ai" in d
        )
        self.assertIn("generate_series(0, 1024)", hav_u_ai)

        t2_having_self = generate_materialized_view(
            sub_schemas,
            "CREATE VIEW v_t2_hav_self AS SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids "
            "FROM t GROUP BY g HAVING SUM(v) > (SELECT AVG(v) FROM t);",
            dialect=Dialect.POSTGRES,
        )
        self.assertFalse(
            any("generate_series(0, 1024)" in d for d in t2_having_self.auxiliary_table_ddls),
            t2_having_self.auxiliary_table_ddls,
        )

        # 4. Trigger firing order between any two views must be identical across every
        #    source table (including underscore/case boundaries and long shared prefixes).
        case_schemas = parse_table_schema(
            'CREATE TABLE "Orders" (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL);'
            "CREATE TABLE dim (g INT PRIMARY KEY, name TEXT NOT NULL);",
            dialect=Dialect.POSTGRES,
        )
        pairs = [
            (
                case_schemas,
                "Orders",
                "dim",
                'CREATE VIEW v AS SELECT dim.name, COUNT(*) AS c FROM "Orders" JOIN dim ON "Orders".g = dim.g GROUP BY dim.name;',
                'CREATE VIEW v_ AS SELECT dim.name, SUM("Orders".v) AS s FROM "Orders" JOIN dim ON "Orders".g = dim.g GROUP BY dim.name;',
            ),
            (
                schemas,
                "t",
                "d",
                "CREATE VIEW very_long_shared_prefix_for_join_view_alpha_one AS "
                "SELECT d.name, COUNT(*) AS c FROM t JOIN d ON t.g = d.g GROUP BY d.name;",
                "CREATE VIEW very_long_shared_prefix_for_join_view_alpha_two AS "
                "SELECT d.name, SUM(t.v) AS s FROM t JOIN d ON t.g = d.g GROUP BY d.name;",
            ),
        ]
        for sc, tbl1, tbl2, sql1, sql2 in pairs:
            p1 = generate_materialized_view(sc, sql1, dialect=Dialect.POSTGRES)
            p2 = generate_materialized_view(sc, sql2, dialect=Dialect.POSTGRES)
            t1_p1 = next(t for t in p1.trigger_names if t.endswith("_ai") and tbl1.lower() in next(d for d in p1.trigger_ddls if f"CREATE TRIGGER {t} " in d).lower())
            t1_p2 = next(t for t in p2.trigger_names if t.endswith("_ai") and tbl1.lower() in next(d for d in p2.trigger_ddls if f"CREATE TRIGGER {t} " in d).lower())
            t2_p1 = next(t for t in p1.trigger_names if t.endswith("_au") and tbl2.lower() in next(d for d in p1.trigger_ddls if f"CREATE TRIGGER {t} " in d).lower())
            t2_p2 = next(t for t in p2.trigger_names if t.endswith("_au") and tbl2.lower() in next(d for d in p2.trigger_ddls if f"CREATE TRIGGER {t} " in d).lower())
            self.assertEqual(t1_p1.lower() < t1_p2.lower(), t2_p1.lower() < t2_p2.lower(), (t1_p1, t1_p2, t2_p1, t2_p2))

    def test_postgres_18_virtual_generated_columns_parsing_codegen_and_fail_closed(self) -> None:
        from matview.engines import UnsupportedMaterializedViewError
        from matview.logical_ir import ColumnDef, TableSchema

        ddl = (
            'CREATE TABLE gv ('
            '  id INT PRIMARY KEY,'
            '  "a b" INT,'
            '  x NUMERIC(10, 2),'
            '  bv BIT VARYING(8),'
            '  v_explicit INT GENERATED ALWAYS AS ("a b" * 2 + 1) VIRTUAL,'
            '  v_default NUMERIC(12, 2) GENERATED ALWAYS AS (x * 1.5),'
            '  v_stored INT GENERATED ALWAYS AS ("a b" + 10) STORED'
            ');'
        )
        schemas = parse_table_schema(ddl, dialect=Dialect.POSTGRES)
        self.assertIn("gv", schemas)
        cols = schemas["gv"].columns
        self.assertTrue(cols["v_explicit"].generated and cols["v_explicit"].generated_virtual)
        self.assertIn('"a b"', cols["v_explicit"].generation_expr or "")
        self.assertTrue(cols["v_default"].generated and cols["v_default"].generated_virtual)
        self.assertIn("1.5", cols["v_default"].generation_expr or "")
        self.assertTrue(cols["v_stored"].generated and not cols["v_stored"].generated_virtual)

        # Trigger functions must NOT expand __mv_n.* / __mv_o.*, and must compute
        # virtual columns via CAST over non-virtual transition columns.
        plan = generate_materialized_view(
            schemas,
            "CREATE VIEW v_gv AS SELECT v_explicit, SUM(v_default) AS s, COUNT(*) AS c FROM gv GROUP BY v_explicit;",
            dialect=Dialect.POSTGRES,
        )
        fn_ddls = [d for d in plan.trigger_ddls if "CREATE OR REPLACE FUNCTION __matview_fn_sync_" in d and "_trunc" not in d]
        self.assertEqual(len(fn_ddls), 3)
        for fn in fn_ddls:
            self.assertNotIn("__mv_n.*", fn)
            self.assertNotIn("__mv_o.*", fn)
            self.assertIn('AS "v_explicit"', fn)
            self.assertIn('AS "v_default"', fn)

        # Standalone PG 18 default-virtual DDL (no VIRTUAL keyword, no BIT VARYING)
        # and partitioned DDL with trailing PARTITION BY LIST (g) must also parse as virtual.
        default_only = parse_table_schema(
            "CREATE TABLE gv_def (id INT PRIMARY KEY, a INT, v INT GENERATED ALWAYS AS (a * 2));"
            "CREATE TABLE gv_p (id INT NOT NULL, g INT NOT NULL, a INT, "
            "v INT GENERATED ALWAYS AS (a + 1) VIRTUAL, PRIMARY KEY (g, id)) PARTITION BY LIST (g);",
            dialect=Dialect.POSTGRES,
        )
        self.assertTrue(default_only["gv_def"].columns["v"].generated_virtual)
        self.assertEqual(default_only["gv_def"].columns["v"].generation_expr, "a * 2")
        self.assertTrue(default_only["gv_p"].columns["v"].generated_virtual)
        self.assertEqual(default_only["gv_p"].columns["v"].generation_expr, "a + 1")

        # Missing generation_expr: unreferenced virtual column (including COUNT(*)) is
        # safely omitted; referenced virtual column fails closed at compile time.
        no_expr_schemas = {
            "gv": TableSchema(
                name="gv",
                columns={
                    "id": ColumnDef("id", "INTEGER", not_null=True, primary_key=True),
                    "g": ColumnDef("g", "INTEGER"),
                    "v": ColumnDef("v", "INTEGER"),
                    "gx": ColumnDef("gx", "INTEGER", generated=True, generated_virtual=True, generation_expr=None),
                },
                pk_columns=("id",),
            )
        }
        for v_sql in (
            "CREATE VIEW v_unref AS SELECT g, SUM(v) AS s FROM gv GROUP BY g;",
            "CREATE VIEW v_cnt_only AS SELECT COUNT(*) AS c FROM gv;",
        ):
            plan_unref = generate_materialized_view(no_expr_schemas, v_sql, dialect=Dialect.POSTGRES)
            for fn in plan_unref.trigger_ddls:
                if "CREATE OR REPLACE FUNCTION __matview_fn_sync_" in fn and "_trunc" not in fn:
                    self.assertNotIn("__mv_n.*", fn)
                    self.assertNotIn("__mv_o.*", fn)
                    self.assertNotIn("gx", fn)

        with self.assertRaises(UnsupportedMaterializedViewError):
            generate_materialized_view(
                no_expr_schemas,
                "CREATE VIEW v_ref_missing AS SELECT g, SUM(gx) AS s FROM gv GROUP BY g;",
                dialect=Dialect.POSTGRES,
            )

    def test_postgres_stage3_multi_source_inclusion_exclusion_codegen(self) -> None:
        """Stage 3: PostgreSQL multi-source algebraic joins emit BEFORE/AFTER
        bracket triggers, minimal-column staging tables, and exact inclusion-
        exclusion reconcile functions (3 terms for 2 tables, 7 terms for 3 tables,
        full-refresh overlap fallback for 5+ tables)."""
        schemas = parse_table_schema(
            "CREATE TABLE d (g INT PRIMARY KEY, name TEXT NOT NULL, unused_col BYTEA);"
            "CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL, extra_unused TEXT);"
            "CREATE TABLE u (uid INT PRIMARY KEY, tid INT NOT NULL, x INT NOT NULL);"
            "CREATE TABLE p4 (p4id INT PRIMARY KEY, uid INT NOT NULL);"
            "CREATE TABLE p5 (p5id INT PRIMARY KEY, p4id INT NOT NULL);",
            dialect=Dialect.POSTGRES,
        )

        # 2-table join: 3 inclusion-exclusion terms (+Δt⋈d, +t⋈Δd, -(Δt⋈Δd))
        p2 = generate_materialized_view(
            schemas,
            "CREATE VIEW v_s3_2 AS SELECT d.name, COUNT(*) AS c, SUM(t.v) AS s, MIN(t.v) AS mn "
            "FROM t JOIN d ON t.g = d.g GROUP BY d.name;",
            dialect=Dialect.POSTGRES,
        )
        reconcile_ddls = [d for d in p2.trigger_ddls if "__matview_fn_s3_" in d and "RETURNS VOID" in d]
        self.assertEqual(len(reconcile_ddls), 1, p2.trigger_ddls)
        rec2 = reconcile_ddls[0]
        # Must include the negative pair correction -(t.__mv_s * d.__mv_s)
        self.assertIn("-(t.__mv_s * d.__mv_s)", rec2)
        self.assertIn("__matview_stg_v_s3_2__t", rec2)
        self.assertIn("__matview_stg_v_s3_2__d", rec2)
        # Unused columns must not be copied into staging tables.
        stg_ddls = "\n".join(d for d in p2.auxiliary_table_ddls if "__matview_stg_" in d)
        self.assertNotIn("unused_col", stg_ddls)
        self.assertNotIn("extra_unused", stg_ddls)
        # BEFORE bracket trigger must be installed on parent AND partitions.
        before_trigs = [
            d
            for d in p2.trigger_ddls
            if d.startswith("CREATE TRIGGER") and "BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE ON" in d
        ]
        self.assertEqual(len(before_trigs), 2)
        install_blocks = [d for d in p2.trigger_ddls if d.startswith("DO $mv_install$")]
        self.assertEqual(len(install_blocks), 2)
        for blk in install_blocks:
            self.assertIn("BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE ON %s", blk)

        # 3-table join: 7 inclusion-exclusion terms including 3 negative pairs and 1 positive triple.
        p3 = generate_materialized_view(
            schemas,
            "CREATE VIEW v_s3_3 AS SELECT d.name, COUNT(*) AS c, SUM(u.x) AS sx "
            "FROM d JOIN t ON t.g = d.g JOIN u ON u.tid = t.id GROUP BY d.name;",
            dialect=Dialect.POSTGRES,
        )
        rec3 = next(d for d in p3.trigger_ddls if "__matview_fn_s3_" in d and "RETURNS VOID" in d)
        self.assertIn("-(d.__mv_s * t.__mv_s)", rec3)
        self.assertIn("-(d.__mv_s * u.__mv_s)", rec3)
        self.assertIn("-(t.__mv_s * u.__mv_s)", rec3)
        self.assertIn("d.__mv_s * t.__mv_s * u.__mv_s", rec3)

        # 5-table join: overlap falls back to locked full refresh without staging tables.
        p5 = generate_materialized_view(
            schemas,
            "CREATE VIEW v_s3_5 AS SELECT d.name, COUNT(*) AS c "
            "FROM d JOIN t ON t.g = d.g JOIN u ON u.tid = t.id "
            "JOIN p4 ON p4.uid = u.uid JOIN p5 ON p5.p4id = p4.p4id GROUP BY d.name;",
            dialect=Dialect.POSTGRES,
        )
        self.assertEqual(p5.auxiliary_table_names, ["__matview_lock_v_s3_5"])
        self.assertFalse(any("__matview_fn_s3_" in d for d in p5.trigger_ddls))
        self.assertEqual(
            sum(
                d.startswith("CREATE TRIGGER") and "BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE ON" in d
                for d in p5.trigger_ddls
            ),
            5,
        )

    def test_stage4_lean_row_triggers_codegen_sqlite_and_mysql(self) -> None:
        """Stage 4: row-trigger bodies on SQLite/MySQL/MariaDB restrict join
        empty-group deletes to touched keys, inline single-row derived tables,
        derive unindexed AVG in the reader view, use VALUES upserts, and emit
        same-group UPDATE fast paths."""
        import re
        import sqlite3
        from matview_sync import MaterializedViewManager

        schema = """
        CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL);
        CREATE TABLE d_str (code VARCHAR(20) PRIMARY KEY, label VARCHAR(20) NOT NULL);
        CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL, w INT, code VARCHAR(20));
        CREATE TABLE u (uid INT PRIMARY KEY, tid INT NOT NULL, x INT NOT NULL);
        """
        unrestricted_del_re = re.compile(
            r"DELETE\s+FROM\s+\S+\s+WHERE\s+\S+\.\S+\s*<=\s*0\s*;",
            re.IGNORECASE,
        )

        # 1. Join views: no unrestricted DELETE, no single-row derived tables,
        #    and 2-table unique-PK-lookup drops GROUP BY / __mv_delta.
        for engine in ("sqlite", "mysql:8.4", "mariadb:10.11"):
            with self.subTest(engine=engine, shape="join_pk"):
                plan = generate_materialized_view(
                    schema,
                    "CREATE VIEW v_j AS SELECT d.name, SUM(t.v) AS s, COUNT(*) AS c "
                    "FROM t JOIN d ON t.g = d.g GROUP BY d.name;",
                    dialect=engine,
                    ignore_limitations=True,
                )
                all_trigs = "\n".join(plan.trigger_ddls)
                self.assertIsNone(unrestricted_del_re.search(all_trigs), all_trigs)
                self.assertNotIn("(SELECT NEW.", all_trigs)
                self.assertNotIn("(SELECT OLD.", all_trigs)
                self.assertNotIn("FROM DUAL", all_trigs)
                t_trigs = [d for d in plan.trigger_ddls if "__t_" in d]
                self.assertTrue(t_trigs)
                for td in t_trigs:
                    self.assertNotIn("GROUP BY", td, td)
                    self.assertNotIn("__mv_delta", td, td)
                d_ad = next(d for d in plan.trigger_ddls if "__d_ad" in d)
                self.assertIn("OLD.name", d_ad)

            # Non-integer VARCHAR PK joined to INT must NOT assume unique 1-row
            # match on MySQL/MariaDB (e.g. '01' = 1 and '1' = 1).
            with self.subTest(engine=engine, shape="join_varchar_pk_keeps_groupby"):
                plan_str = generate_materialized_view(
                    schema,
                    "CREATE VIEW v_js AS SELECT d_str.label, COUNT(*) AS c "
                    "FROM t JOIN d_str ON t.g = d_str.code GROUP BY d_str.label;",
                    dialect=engine,
                    ignore_limitations=True,
                )
                all_str = "\n".join(plan_str.trigger_ddls)
                self.assertIsNone(unrestricted_del_re.search(all_str), all_str)
                self.assertNotIn("(SELECT NEW.", all_str)
                t_ai = next(d for d in plan_str.trigger_ddls if "__t_ai" in d)
                self.assertIn("GROUP BY", t_ai)

            # 3-table join and composite-PK join: inlined, key-restricted delete.
            with self.subTest(engine=engine, shape="join_3t_and_composite"):
                p3 = generate_materialized_view(
                    schema,
                    "CREATE VIEW v_j3 AS SELECT d.name, t.w, COUNT(*) AS c "
                    "FROM d JOIN t ON t.g = d.g JOIN u ON u.tid = t.id "
                    "GROUP BY d.name, t.w;",
                    dialect=engine,
                    ignore_limitations=True,
                )
                all_p3 = "\n".join(p3.trigger_ddls)
                self.assertIsNone(unrestricted_del_re.search(all_p3), all_p3)
                self.assertNotIn("(SELECT NEW.", all_p3)
                self.assertNotIn("(SELECT OLD.", all_p3)
                self.assertNotIn("FROM DUAL", all_p3)
                self.assertNotIn("CROSS JOIN", all_p3)

            # 2. Single-table agg: reader-derived AVG, NOT NULL count reuse,
            #    VALUES upsert, and same-group UPDATE fast path.
            with self.subTest(engine=engine, shape="single_agg"):
                p_agg = generate_materialized_view(
                    schema,
                    "CREATE VIEW v_agg AS SELECT g, SUM(v) AS s, COUNT(*) AS c, AVG(v) AS a "
                    "FROM t GROUP BY g;",
                    dialect=engine,
                    ignore_limitations=True,
                )
                self.assertNotIn("\n    a ", p_agg.backing_table_ddl)
                self.assertNotIn("__mv_cnt_", p_agg.backing_table_ddl)
                self.assertIn("AS a", p_agg.reader_view_ddl)
                t_ai = next(d for d in p_agg.trigger_ddls if d.startswith("CREATE TRIGGER") and "__t_ai" in d)
                self.assertIn("VALUES (", t_ai)
                self.assertNotIn("FROM DUAL", t_ai)
                self.assertNotIn("WHERE TRUE", t_ai)
                if engine == "sqlite":
                    self.assertTrue(any("__t_aus" in d for d in p_agg.trigger_ddls))
                else:
                    self.assertEqual(len(p_agg.trigger_ddls), 3)
                    t_au = next(d for d in p_agg.trigger_ddls if "__t_au" in d)
                    self.assertIn("ELSE", t_au)

            # Compound AVG expression + HAVING AVG(...): still reader-derived.
            with self.subTest(engine=engine, shape="compound_having_avg"):
                p_comp = generate_materialized_view(
                    schema,
                    "CREATE VIEW v_comp AS SELECT g, AVG(v) * 2 AS a2, COUNT(*) AS c "
                    "FROM t GROUP BY g HAVING AVG(v) > 5;",
                    dialect=engine,
                    ignore_limitations=True,
                )
                self.assertNotIn("__mv_elem_", p_comp.backing_table_ddl)
                self.assertIn("AS a2", p_comp.reader_view_ddl)
                self.assertIn("WHERE", p_comp.reader_view_ddl.upper())

            # MariaDB projection trigger uses VALUES without FROM DUAL WHERE TRUE.
            if engine == "mariadb:10.11":
                p_proj = generate_materialized_view(
                    schema,
                    "CREATE VIEW v_proj AS SELECT id, g, v FROM t;",
                    dialect=engine,
                    ignore_limitations=True,
                )
                ai_proj = next(d for d in p_proj.trigger_ddls if "__t_ai" in d)
                self.assertIn("VALUES (", ai_proj)
                self.assertNotIn("FROM DUAL", ai_proj)
                self.assertNotIn("WHERE TRUE", ai_proj)

        # 3. Live SQLite EXPLAIN QUERY PLAN check & adversarial edge cases:
        #    join empty-group DELETE and refresh_grp_stmts use indexed SEARCH
        #    (never SCAN); one-row SUM(d.x)/TOTAL(d.x)/FILTER(WHERE d.x > 0)
        #    never emit phantom rows on unmatched inserts; user index modifiers
        #    / partial / expression / quoted aliases keep AVG stored; AST
        #    reader-view substitution never corrupts CAST types, IS NOT NULL,
        #    or escaped alias quotes; swapped aliases and 4-trigger lifecycle.
        con = sqlite3.connect(":memory:")
        try:
            MaterializedViewManager.configure_sqlite_connection(con)
            con.executescript(
                "CREATE TABLE d (g INTEGER PRIMARY KEY, name TEXT NOT NULL, x INTEGER);"
                "CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, v INTEGER NOT NULL);"
                "INSERT INTO d VALUES (1, 'A', 10);"
            )
            mgr = MaterializedViewManager(con, dialect="sqlite")
            plan_live = mgr.register_view(
                "CREATE VIEW mv_j_eqp AS SELECT d.name, COUNT(*) AS c "
                "FROM t JOIN d ON t.g = d.g GROUP BY d.name;"
            )
            for tr in plan_live.trigger_ddls:
                if "_ad" not in tr and "_au" not in tr:
                    continue
                body = tr.split("BEGIN", 1)[1].rsplit("END", 1)[0]
                for stmt in body.split(";"):
                    st = stmt.strip()
                    if not st.upper().startswith("DELETE"):
                        continue
                    probe = (
                        st.replace("OLD.g", "1")
                        .replace("OLD.name", "'x'")
                        .replace("OLD.id", "1")
                        .replace("OLD.v", "1")
                    )
                    eqp_lines = [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + probe).fetchall()]
                    eqp_text = "\n".join(eqp_lines)
                    self.assertNotIn(f"SCAN {plan_live.backing_table_name}", eqp_text)
                    self.assertIn(f"SEARCH {plan_live.backing_table_name}", eqp_text)

            # One-row SQLite SUM/TOTAL/FILTER referencing joined table `d` must
            # not turn the enclosing `FROM d` SELECT into an ungrouped aggregate.
            # Test GROUP BY t.g first so dropping the subquery scope produces a
            # phantom row caught by assertEqual (killing the curated mutant via
            # assertion rather than IntegrityError on NOT NULL d.name).
            mgr.register_view(
                "CREATE VIEW mv_j_tg AS SELECT t.g, SUM(d.x) AS sx, COUNT(*) AS c "
                "FROM t JOIN d ON t.g = d.g GROUP BY t.g;"
            )
            con.execute("INSERT INTO t VALUES (1, 99, 5);")  # unmatched in d
            con.commit()
            self.assertEqual(con.execute("SELECT * FROM mv_j_tg").fetchall(), [])
            mgr.register_view(
                "CREATE VIEW mv_j_dsum AS SELECT d.name, SUM(d.x) AS sx, TOTAL(d.x) AS tx, "
                "SUM(t.v) FILTER (WHERE d.x > 0) AS sf, COUNT(*) AS c "
                "FROM t JOIN d ON t.g = d.g GROUP BY d.name;"
            )
            con.execute("INSERT INTO t VALUES (10, 99, 5);")  # unmatched in d
            con.commit()
            self.assertEqual(con.execute("SELECT * FROM mv_j_dsum").fetchall(), [])
            con.execute("INSERT INTO t VALUES (2, 1, 7);")   # matched in d
            con.commit()
            self.assertEqual(
                con.execute("SELECT * FROM mv_j_dsum ORDER BY name").fetchall(),
                con.execute(
                    "SELECT d.name, SUM(d.x), TOTAL(d.x), SUM(t.v) FILTER (WHERE d.x > 0), COUNT(*) "
                    "FROM t JOIN d ON t.g = d.g GROUP BY d.name ORDER BY d.name"
                ).fetchall(),
            )

            # User indexes with DESC, partial WHERE, expression, and quoted
            # reserved alias must keep AVG stored in the backing table.
            for vname, vsql in (
                (
                    "mv_idx_desc",
                    "CREATE VIEW mv_idx_desc AS SELECT g, AVG(v) AS a FROM t GROUP BY g;\n"
                    "CREATE INDEX idx_d ON mv_idx_desc (a DESC);",
                ),
                (
                    "mv_idx_part",
                    "CREATE VIEW mv_idx_part AS SELECT g, AVG(v) AS a FROM t GROUP BY g;\n"
                    "CREATE INDEX idx_p ON mv_idx_part (g) WHERE a > 5;",
                ),
                (
                    "mv_idx_expr",
                    "CREATE VIEW mv_idx_expr AS SELECT g, AVG(v) AS a FROM t GROUP BY g;\n"
                    "CREATE INDEX idx_e ON mv_idx_expr ((a * 2));",
                ),
                (
                    "mv_idx_quot",
                    'CREATE VIEW mv_idx_quot AS SELECT g, AVG(v) AS "order" FROM t GROUP BY g;\n'
                    'CREATE INDEX idx_q ON mv_idx_quot ("order");',
                ),
            ):
                p_idx = mgr.register_view(vsql)
                self.assertEqual(
                    con.execute(f"SELECT * FROM {vname} ORDER BY 1").fetchall(),
                    con.execute("SELECT g, AVG(v) FROM t GROUP BY g ORDER BY 1").fetchall(),
                    vname,
                )
                self.assertTrue(p_idx.index_ddls)

            # AST-based reader substitution must not corrupt CAST types,
            # IS NOT NULL, or escaped double-quoted aliases.
            mgr.register_view(
                'CREATE VIEW mv_ast_real AS SELECT g, AVG(v) AS "real", '
                'CAST(SUM(v) AS REAL) AS s_real FROM t GROUP BY g;'
            )
            mgr.register_view(
                'CREATE VIEW mv_ast_null AS SELECT g, AVG(v) AS "null", '
                'SUM(v) AS s FROM t GROUP BY g HAVING SUM(v) IS NOT NULL;'
            )
            mgr.register_view(
                'CREATE VIEW mv_ast_esc AS SELECT g, AVG(v) AS av, '
                'AVG(v) * 2 AS "my ""av""" FROM t GROUP BY g;'
            )
            self.assertEqual(
                con.execute("SELECT * FROM mv_ast_real ORDER BY g").fetchall(),
                con.execute("SELECT g, AVG(v), CAST(SUM(v) AS REAL) FROM t GROUP BY g ORDER BY g").fetchall(),
            )
            self.assertEqual(
                con.execute("SELECT * FROM mv_ast_null ORDER BY g").fetchall(),
                con.execute("SELECT g, AVG(v), SUM(v) FROM t GROUP BY g HAVING SUM(v) IS NOT NULL ORDER BY g").fetchall(),
            )
            self.assertEqual(
                con.execute("SELECT * FROM mv_ast_esc ORDER BY g").fetchall(),
                con.execute("SELECT g, AVG(v), AVG(v) * 2 FROM t GROUP BY g ORDER BY g").fetchall(),
            )

            # Swapped table/alias names (`FROM t AS d JOIN d AS t`).
            mgr.register_view(
                "CREATE VIEW mv_swap_al AS SELECT t.name, SUM(d.v) AS sv, COUNT(*) AS c "
                "FROM t AS d JOIN d AS t ON d.g = t.g GROUP BY t.name;"
            )
            self.assertEqual(
                con.execute("SELECT * FROM mv_swap_al ORDER BY name").fetchall(),
                con.execute(
                    "SELECT t.name, SUM(d.v), COUNT(*) FROM t AS d JOIN d AS t ON d.g = t.g GROUP BY t.name ORDER BY t.name"
                ).fetchall(),
            )

            # 4-trigger SQLite lifecycle (register, bulk_changes, repair_view, drop_view).
            p_life = mgr.register_view(
                "CREATE VIEW mv_life AS SELECT g, SUM(v) AS s, COUNT(*) AS c, AVG(v) AS a FROM t GROUP BY g;"
            )
            self.assertTrue(any("_aus" in tn for tn in p_life.trigger_names))
            with mgr.bulk_changes(["mv_life"]):
                con.execute("INSERT INTO t VALUES (3, 1, 13);")
            mgr.repair_view("mv_life")
            self.assertEqual(
                con.execute("SELECT g, s, c, a FROM mv_life ORDER BY g").fetchall(),
                con.execute("SELECT g, SUM(v), COUNT(*), AVG(v) FROM t GROUP BY g ORDER BY g").fetchall(),
            )
            mgr.drop_view("mv_life")
            leftover = con.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%mv_life%'"
            ).fetchall()
            self.assertEqual(leftover, [])
        finally:
            con.close()

    def test_stage4_mysql_filtered_union_all_agg_preserves_branch_where_guard(self) -> None:
        """Regression: MySQL VALUES (...) upserts in filtered UNION ALL aggregate
        branches must still be guarded by the branch WHERE predicate via IF (...)."""
        schema = parse_table_schema(
            """
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                region VARCHAR(32),
                status VARCHAR(32),
                amount INTEGER
            );
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                tier VARCHAR(32)
            );
            """,
            dialect=Dialect.MYSQL,
        )
        plan = generate_materialized_view(
            schema,
            """
            CREATE VIEW v_uagg_filt AS
            WITH u AS (
                SELECT region AS k, amount AS v FROM orders WHERE status = 'settled'
                UNION ALL
                SELECT tier AS k, 10 AS v FROM customers
            )
            SELECT k, COUNT(*) AS cnt, SUM(v) AS sv FROM u GROUP BY k;
            """,
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        orders_ai = next(t for t in plan.trigger_ddls if "AFTER INSERT ON orders" in t)
        orders_au = next(t for t in plan.trigger_ddls if "AFTER UPDATE ON orders" in t)
        self.assertRegex(orders_ai, r"IF \(NEW\.`?status`? = 'settled'\) THEN")
        self.assertIn("VALUES (", orders_ai)
        self.assertRegex(orders_au, r"IF \(NEW\.`?status`? = 'settled'\) THEN")
        self.assertNotIn("FROM DUAL", orders_ai)

    def test_postgres_repair_snapshot_fence_codegen(self) -> None:
        schema = """
        CREATE TABLE orders (
            id BIGINT PRIMARY KEY,
            d_id BIGINT NOT NULL,
            region TEXT NOT NULL,
            amount BIGINT NOT NULL
        );
        CREATE TABLE dims (
            id BIGINT PRIMARY KEY,
            label TEXT NOT NULL
        );
        """
        # 1. Standalone CLI / generate_materialized_view() without repair_fence_catalog_table
        #    stays self-contained with no catalog dependency.
        standalone = generate_materialized_view(
            schema,
            "CREATE VIEW v_plain AS SELECT region, SUM(amount) AS total FROM orders GROUP BY region;",
            dialect=Dialect.POSTGRES,
        )
        self.assertNotIn("__MV_REPAIR_FENCE_START__", "\n".join(standalone.trigger_ddls))
        self.assertNotIn("FOR SHARE", "\n".join(standalone.trigger_ddls))

        # 2. Single-table view with repair_fence_catalog_table emits the fence in
        #    all statement + truncate functions.
        fenced = generate_materialized_view(
            schema,
            "CREATE VIEW v_fence AS SELECT region, SUM(amount) AS total FROM orders GROUP BY region;",
            dialect=Dialect.POSTGRES,
            repair_fence_catalog_table="__matview_catalog",
        )
        fn_ddls = [d for d in fenced.trigger_ddls if d.lstrip().startswith("CREATE OR REPLACE FUNCTION")]
        self.assertEqual(len(fn_ddls), 4)
        for fn in fn_ddls:
            self.assertIn("__MV_REPAIR_FENCE_START__", fn)
            self.assertIn("pg_catalog.current_setting('transaction_isolation') IN ('repeatable read', 'serializable')", fn)
            self.assertIn("PERFORM 1 FROM __matview_catalog WHERE view_name = 'v_fence' FOR SHARE;", fn)
            self.assertIn("USING ERRCODE = 'serialization_failure'", fn)

        # 3. Stage 3 bracketed multi-source join emits the fence on overlap staging,
        #    truncate overlap, single-source fast path, and bracketed truncate.
        fenced_join = generate_materialized_view(
            schema,
            "CREATE VIEW v_j AS SELECT d.label, SUM(o.amount) AS total FROM orders o JOIN dims d ON d.id = o.d_id GROUP BY d.label;",
            dialect=Dialect.POSTGRES,
            repair_fence_catalog_table="__matview_catalog",
        )
        stmt_fns = [
            d for d in fenced_join.trigger_ddls
            if d.lstrip().startswith("CREATE OR REPLACE FUNCTION")
            and not any(s in d.splitlines()[0] for s in ("_bs()", "_trunc()", "_reconcile()"))
        ]
        trunc_fns = [
            d for d in fenced_join.trigger_ddls
            if d.lstrip().startswith("CREATE OR REPLACE FUNCTION") and "_trunc()" in d.splitlines()[0]
        ]
        self.assertEqual(len(stmt_fns), 6)
        self.assertEqual(len(trunc_fns), 2)
        for fn in stmt_fns:
            self.assertEqual(fn.count("__MV_REPAIR_FENCE_START__"), 3)
            self.assertIn("PERFORM 1 FROM __matview_catalog WHERE view_name = 'v_j' FOR SHARE;", fn)
            # The closing bracket (IF __mv_nd = 0 THEN) fences BOTH trunc_refresh_sql
            # and overlap_close_sql (not only IF __mv_tr <> 0).
            close_idx = fn.index("IF __mv_nd = 0 THEN")
            trunc_branch_idx = fn.index("IF __mv_tr <> 0 THEN", close_idx)
            self.assertIn("__MV_REPAIR_FENCE_START__", fn[close_idx:trunc_branch_idx])
        for fn in trunc_fns:
            self.assertEqual(fn.count("__MV_REPAIR_FENCE_START__"), 1)
            self.assertIn("PERFORM 1 FROM __matview_catalog WHERE view_name = 'v_j' FOR SHARE;", fn)

        # 4. Quoted view name with single quote and custom object_prefix catalog table.
        fenced_quoted = generate_materialized_view(
            schema,
            'CREATE VIEW "v\'s fence" AS SELECT region, COUNT(*) AS cnt FROM orders GROUP BY region;',
            dialect=Dialect.POSTGRES,
            object_prefix="custom_mv_",
            repair_fence_catalog_table="custom_mv_catalog",
        )
        quoted_fns = [d for d in fenced_quoted.trigger_ddls if d.lstrip().startswith("CREATE OR REPLACE FUNCTION")]
        self.assertTrue(quoted_fns)
        for fn in quoted_fns:
            self.assertIn("PERFORM 1 FROM custom_mv_catalog WHERE view_name = 'v''s fence' FOR SHARE;", fn)

        # 5. Multi-view script propagates repair_fence_catalog_table to additional_views
        #    and generate_materialized_views() forwards it symmetrically.
        multi_sql = (
            "CREATE VIEW v_m1 AS SELECT region, COUNT(*) AS c FROM orders GROUP BY region;\n"
            "CREATE VIEW v_m2 AS SELECT label, COUNT(*) AS c FROM dims GROUP BY label;"
        )
        multi = generate_materialized_view(
            schema,
            multi_sql,
            dialect=Dialect.POSTGRES,
            repair_fence_catalog_table="__matview_catalog",
        )
        self.assertEqual(len(multi.sub_plans), 2)
        self.assertIn("WHERE view_name = 'v_m1' FOR SHARE;", "\n".join(multi.sub_plans[0].trigger_ddls))
        self.assertIn("WHERE view_name = 'v_m2' FOR SHARE;", "\n".join(multi.sub_plans[1].trigger_ddls))
        multi_list = generate_materialized_views(
            schema,
            multi_sql,
            dialect=Dialect.POSTGRES,
            repair_fence_catalog_table="__matview_catalog",
        )
        self.assertEqual(len(multi_list), 2)
        self.assertIn("WHERE view_name = 'v_m1' FOR SHARE;", "\n".join(multi_list[0].trigger_ddls))
        self.assertIn("WHERE view_name = 'v_m2' FOR SHARE;", "\n".join(multi_list[1].trigger_ddls))

        # 6. Non-PostgreSQL dialects ignore repair_fence_catalog_table cleanly.
        for non_pg in (Dialect.SQLITE, Dialect.MYSQL):
            plan = generate_materialized_view(
                schema,
                "CREATE VIEW v_nonpg AS SELECT region, COUNT(*) AS c FROM orders GROUP BY region;",
                dialect=non_pg,
                ignore_limitations=(non_pg == Dialect.MYSQL),
                repair_fence_catalog_table="__matview_catalog",
            )
            self.assertNotIn("__MV_REPAIR_FENCE_START__", "\n".join(plan.trigger_ddls))

    def test_postgres_minmax_varchar_collation_name_and_numeric_group_rep_codegen(self) -> None:
        """PostgreSQL MIN/MAX over collated VARCHAR/CHAR, NAME, and explicit COLLATE
        expressions resolve to TEXT/BPCHAR with preserved collation; unconstrained
        NUMERIC group keys emit variable-scale representative refresh while
        fixed-scale NUMERIC(p, s) and integer keys do not."""
        schema = parse_table_schema(
            """
            CREATE TABLE items (
                id BIGINT PRIMARY KEY,
                g INT NOT NULL,
                vc VARCHAR(20),
                vc_c VARCHAR(20) COLLATE "C",
                ch_c CHAR(5) COLLATE "C",
                nm NAME,
                vca VARCHAR(20)[] COLLATE "C",
                num NUMERIC,
                num_nn NUMERIC NOT NULL,
                num_fixed NUMERIC(10, 2)
            );
            """,
            dialect=Dialect.POSTGRES,
        )
        # 1. MIN/MAX over collated VARCHAR, CHAR, NAME, explicit COLLATE, array, and window.
        p_mm = generate_materialized_view(
            schema,
            """
            CREATE VIEW mv_pg_mm AS
            SELECT
                g,
                MIN(vc) AS mn_vc,
                MIN(vc_c) AS mn_vcc,
                MAX(ch_c) AS mx_chc,
                MIN(nm) AS mn_nm,
                MIN(vc COLLATE "C") AS mn_expr_c,
                MAX(vc_c COLLATE "POSIX") AS mx_override,
                MIN(vca) AS mn_vca
            FROM items GROUP BY g;
            """,
            dialect=Dialect.POSTGRES,
        )
        ddl = p_mm.backing_table_ddl
        self.assertRegex(ddl, r"\bmn_vc\s+TEXT\b")
        self.assertRegex(ddl, r'\bmn_vcc\s+TEXT\s+COLLATE\s+"C"')
        self.assertRegex(ddl, r'\bmx_chc\s+BPCHAR\s+COLLATE\s+"C"')
        self.assertRegex(ddl, r'\bmn_nm\s+TEXT\s+COLLATE\s+"C"')
        self.assertRegex(ddl, r'\bmn_expr_c\s+TEXT\s+COLLATE\s+"C"')
        self.assertRegex(ddl, r'\bmx_override\s+TEXT\s+COLLATE\s+"POSIX"')
        self.assertRegex(ddl, r'\bmn_vca\s+VARCHAR\[\]\s+COLLATE\s+"C"')

        p_win = generate_materialized_view(
            schema,
            "CREATE VIEW mv_pg_win_mm AS SELECT id, MIN(vc_c) OVER (PARTITION BY g) AS mn_w FROM items;",
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_win.backing_table_ddl, r'\bmn_w\s+TEXT\s+COLLATE\s+"C"')

        p_proj_coll = generate_materialized_view(
            schema,
            'CREATE VIEW mv_pg_proj_coll AS SELECT id, vc COLLATE "C" AS vc_lit_c FROM items;',
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_proj_coll.backing_table_ddl, r'\bvc_lit_c\s+VARCHAR\s+COLLATE\s+"C"')

        # 2. Unconstrained NUMERIC group key emits representative refresh on UPDATE/DELETE,
        #    while fixed-scale NUMERIC(10, 2) and INT do not.
        p_num = generate_materialized_view(
            schema,
            "CREATE VIEW mv_num AS SELECT num, COUNT(*) AS c FROM items GROUP BY num;",
            dialect=Dialect.POSTGRES,
        )
        num_au = "\n".join(t for t in p_num.trigger_ddls if "_au()" in t)
        num_ad = "\n".join(t for t in p_num.trigger_ddls if "_ad()" in t)
        num_ai = "\n".join(t for t in p_num.trigger_ddls if "_ai()" in t)
        self.assertIn("__mv_gd0", num_au)
        self.assertIn("__mv_gd0", num_ad)
        self.assertNotIn("__mv_gd0", num_ai)

        p_num_nn = generate_materialized_view(
            schema,
            "CREATE VIEW mv_num_nn AS SELECT num_nn, COUNT(*) AS c FROM items GROUP BY num_nn;",
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("__mv_gd0", "\n".join(t for t in p_num_nn.trigger_ddls if "_au()" in t))

        p_fixed = generate_materialized_view(
            schema,
            "CREATE VIEW mv_num_fixed AS SELECT num_fixed, COUNT(*) AS c FROM items GROUP BY num_fixed;",
            dialect=Dialect.POSTGRES,
        )
        self.assertNotIn("__mv_gd0", "\n".join(p_fixed.trigger_ddls))

        p_int = generate_materialized_view(
            schema,
            "CREATE VIEW mv_int AS SELECT g, COUNT(*) AS c FROM items GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertNotIn("__mv_gd0", "\n".join(p_int.trigger_ddls))

        # 3. Adversarial regressions: ARRAY[vc_c, vc_c], COALESCE(MIN(DISTINCT vc_c), 'x'),
        #    COALESCE(vc_c, 'long'), CTE MIN(vc_c), non-decomposed MIN(vc) COLLATE "C",
        #    string expressions over collated columns, NUMERIC[][]/NUMRANGE[], quoted
        #    table/column names, and multi-column correlated LATERAL refresh.
        adv_schema = parse_table_schema(
            """
            CREATE TABLE items (
                id BIGINT PRIMARY KEY,
                g INT NOT NULL,
                vc VARCHAR(20),
                vc_c VARCHAR(20) COLLATE "C",
                vc30_c VARCHAR(30) COLLATE "C",
                num NUMERIC,
                num2 NUMERIC,
                num2d NUMERIC[][],
                nra NUMRANGE[]
            );
            CREATE TABLE "order" (id INT PRIMARY KEY, g INT NOT NULL, num NUMERIC);
            CREATE TABLE "my dim" (g INT PRIMARY KEY, tag TEXT);
            CREATE TABLE t_sp ("my key" NUMERIC PRIMARY KEY, "my num" NUMERIC, v INT);
            CREATE TABLE t_sp2 ("my num" NUMERIC, v INT);
            CREATE TABLE t_int (id INT PRIMARY KEY, num INT, v INT);
            """,
            dialect=Dialect.POSTGRES,
        )
        p_arr = generate_materialized_view(
            adv_schema,
            "CREATE VIEW mv_arr_c AS SELECT id, ARRAY[vc_c, vc_c] AS a FROM items;",
            dialect=Dialect.POSTGRES,
        )
        self.assertNotIn('COLLATE "C"[]', p_arr.backing_table_ddl)
        self.assertRegex(p_arr.backing_table_ddl, r'\ba\s+VARCHAR\[\]\s+COLLATE\s+"C"')

        p_coal_d = generate_materialized_view(
            adv_schema,
            "CREATE VIEW mv_coal_d AS SELECT g, COALESCE(MIN(DISTINCT vc_c), 'x') AS mn FROM items GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_coal_d.backing_table_ddl, r'\bmn\s+TEXT\s+COLLATE\s+"C"')

        p_coal_v = generate_materialized_view(
            adv_schema,
            "CREATE VIEW mv_coal_v AS SELECT id, COALESCE(vc_c, 'a_very_long_string_exceeding_20_chars') AS x FROM items;",
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_coal_v.backing_table_ddl, r'\bx\s+VARCHAR\s+COLLATE\s+"C"')

        p_cte_mm = generate_materialized_view(
            adv_schema,
            'CREATE VIEW mv_cte_mm AS WITH c AS (SELECT g, MIN(vc) AS mn, MIN(vc_c) AS mnc, MIN(vc COLLATE "C") AS mnx FROM items GROUP BY g) SELECT g, mn, mnc, mnx FROM c;',
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_cte_mm.backing_table_ddl, r"\bmn\s+TEXT\b")
        self.assertRegex(p_cte_mm.backing_table_ddl, r'\bmnc\s+TEXT\s+COLLATE\s+"C"')
        self.assertRegex(p_cte_mm.backing_table_ddl, r'\bmnx\s+TEXT\s+COLLATE\s+"C"')

        p_lim_coll = generate_materialized_view(
            adv_schema,
            'CREATE VIEW mv_lim_c AS SELECT g, MIN(vc) COLLATE "C" AS mn, MIN(vc_c) COLLATE "POSIX" AS mnp FROM items GROUP BY g ORDER BY g LIMIT 5;',
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_lim_coll.backing_table_ddl, r'\bmn\s+TEXT\s+COLLATE\s+"C"')
        self.assertRegex(p_lim_coll.backing_table_ddl, r'\bmnp\s+TEXT\s+COLLATE\s+"POSIX"')

        p_str_expr = generate_materialized_view(
            adv_schema,
            "CREATE VIEW mv_str_e AS SELECT g, MIN(UPPER(vc_c)) AS mu, MIN(vc_c || 'x') AS mc, MIN(CASE WHEN id > 0 THEN vc_c ELSE 'z' END) AS mcase FROM items GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_str_expr.backing_table_ddl, r'\bmu\s+TEXT\s+COLLATE\s+"C"')
        self.assertRegex(p_str_expr.backing_table_ddl, r'\bmc\s+TEXT\s+COLLATE\s+"C"')
        self.assertRegex(p_str_expr.backing_table_ddl, r'\bmcase\s+TEXT\s+COLLATE\s+"C"')

        p_uall_coll = generate_materialized_view(
            adv_schema,
            "CREATE VIEW mv_uall_c AS WITH u AS (SELECT g, vc_c AS s FROM items UNION ALL SELECT g, vc30_c AS s FROM items) SELECT g, MIN(s) AS ms FROM u GROUP BY g;",
            dialect=Dialect.POSTGRES,
        )
        self.assertRegex(p_uall_coll.backing_table_ddl, r'\bms\s+TEXT\s+COLLATE\s+"C"')

        # Multi-dimensional NUMERIC[][] and NUMRANGE[] get representative refresh.
        for vn, col in (("mv_n2d", "num2d"), ("mv_nra", "nra")):
            p_arr_n = generate_materialized_view(
                adv_schema,
                f"CREATE VIEW {vn} AS SELECT {col}, COUNT(*) AS c FROM items GROUP BY {col};",
                dialect=Dialect.POSTGRES,
            )
            self.assertIn("__mv_gd0", "\n".join(t for t in p_arr_n.trigger_ddls if "_au()" in t), vn)

        # Quoted/keyword/space table & column names in join, projection, and UNION.
        p_kw_j = generate_materialized_view(
            adv_schema,
            'CREATE VIEW mv_kw_j AS SELECT "order".num, COUNT(*) AS c FROM "order" JOIN "my dim" ON "my dim".g = "order".g GROUP BY "order".num;',
            dialect=Dialect.POSTGRES,
        )
        kw_au = "\n".join(t for t in p_kw_j.trigger_ddls if "_au()" in t)
        self.assertIn("__mv_j_0", kw_au)
        self.assertIn("__mv_j_1", kw_au)
        self.assertNotIn("AS __mv_j_my dim", kw_au)

        p_sp_pk = generate_materialized_view(
            adv_schema,
            'CREATE VIEW mv_sp_pk AS SELECT "my key", v FROM t_sp;',
            dialect=Dialect.POSTGRES,
        )
        self.assertIn('"my key" = EXCLUDED."my key"', "\n".join(p_sp_pk.trigger_ddls))

        p_sp_un = generate_materialized_view(
            adv_schema,
            'CREATE VIEW mv_sp_un AS SELECT "my num" FROM t_sp UNION SELECT "my num" FROM t_sp2;',
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("__mv_vd0", "\n".join(t for t in p_sp_un.trigger_ddls if "_au()" in t))

        # Multi-column variable-scale GROUP BY emits correlated single-row LATERAL refresh.
        p_multi = generate_materialized_view(
            adv_schema,
            "CREATE VIEW mv_multi_n AS SELECT num, num2, COUNT(*) AS c FROM items GROUP BY num, num2;",
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("LEFT JOIN LATERAL", "\n".join(t for t in p_multi.trigger_ddls if "_au()" in t))

        # Widened UNION ALL INT + NUMERIC with extra projected `num + 0` refreshes both columns.
        p_widen = generate_materialized_view(
            adv_schema,
            "CREATE VIEW mv_widen AS WITH u AS (SELECT num, v FROM t_int UNION ALL SELECT num, v FROM items) SELECT num AS a, num + 0 AS c_num, SUM(v) AS s FROM u GROUP BY num;",
            dialect=Dialect.POSTGRES,
        )
        widen_au = "\n".join(t for t in p_widen.trigger_ddls if "_au()" in t)
        self.assertIn("__mv_gd0", widen_au)
        self.assertIn("__mv_gd1", widen_au)
        self.assertIn("LEFT JOIN LATERAL", widen_au)

    def test_postgres_registration_expr_type_probing_complex_shapes_codegen(self) -> None:
        """PostgreSQL _pg_collation_guard_ddl emits registration-time pg_typeof
        probes for CTEs, compound set operations, FROM subqueries, JOIN ... USING,
        GROUPING SETS, DISTINCT ON, named WINDOW clauses, and MIN/MAX."""
        schema = parse_table_schema(
            """
            CREATE TABLE cp1 (id INT PRIMARY KEY, a INT, v INT);
            CREATE TABLE cp2 (id INT PRIMARY KEY, b INT, w INT);
            """,
            dialect=Dialect.POSTGRES,
        )
        cases = [
            (
                "mv_cte_ni",
                "CREATE VIEW mv_cte_ni AS WITH c AS (SELECT DISTINCT id, ROW(a, v) AS rk, v FROM cp1) "
                "SELECT COUNT(*) AS n FROM c GROUP BY rk;",
                ("WITH c AS", "FROM c\nWHERE false", ", true)"),
            ),
            (
                "mv_sub_ni",
                "CREATE VIEW mv_sub_ni AS SELECT COUNT(*) AS n "
                "FROM (SELECT DISTINCT id, ROW(a, v) AS rk, v FROM cp1) AS sub GROUP BY rk;",
                ("AS sub\nWHERE false", "(rk) AS __mv_k", ", true)"),
            ),
            (
                "mv_using",
                "CREATE VIEW mv_using AS SELECT COUNT(*) AS n "
                "FROM cp1 JOIN cp2 USING (id) GROUP BY ROW(cp1.a, cp2.b);",
                ("cp1 JOIN cp2 USING (id)", "ROW(cp1.a, cp2.b)", ", true)"),
            ),
            (
                "mv_union",
                "CREATE VIEW mv_union AS SELECT ROW(a, v) AS r FROM cp1 UNION SELECT ROW(b, w) AS r FROM cp2;",
                ("AS __mv_v(__mv_c0)", "pg_typeof(q.__mv_c0)", ", true)"),
            ),
            (
                "mv_uall_br",
                "CREATE VIEW mv_uall_br AS SELECT COUNT(*) AS n FROM cp1 GROUP BY ROW(a, v) "
                "UNION ALL SELECT COUNT(*) AS n FROM cp2 GROUP BY ROW(b, w);",
                ("ROW(a, v)", "ROW(b, w)", ", true)"),
            ),
            (
                "mv_gsets",
                "CREATE VIEW mv_gsets AS SELECT COUNT(*) AS n FROM cp1 GROUP BY GROUPING SETS ((ROW(a, v)), ());",
                ("ROW(a, v)", ", true)"),
            ),
            (
                "mv_dston",
                "CREATE VIEW mv_dston AS SELECT DISTINCT ON (1) ROW(a, v) AS r, id FROM cp1 ORDER BY 1, id;",
                ("ROW(a, v)", ", true)"),
            ),
            (
                "mv_named_win",
                "CREATE VIEW mv_named_win AS SELECT id, SUM(v) OVER w AS s FROM cp1 WINDOW w AS (PARTITION BY ROW(a, v));",
                ("ROW(a, v)", "FROM cp1\nWHERE false", ", true)"),
            ),
            (
                "mv_cnt_dist",
                "CREATE VIEW mv_cnt_dist AS SELECT a, COUNT(DISTINCT ROW(a, v)) AS c FROM cp1 GROUP BY a;",
                ("ROW(a, v)", ", true)"),
            ),
            (
                "mv_minmax",
                "CREATE VIEW mv_minmax AS SELECT a, MIN(ROW(a, v)) AS m FROM cp1 GROUP BY a;",
                ("ROW(a, v)", ", false)"),
            ),
        ]
        for vname, view_sql, expected_fragments in cases:
            plan = generate_materialized_view(schema, view_sql, dialect=Dialect.POSTGRES)
            guards = [d for d in plan.trigger_ddls if "FOR probe_rec IN" in d]
            self.assertEqual(len(guards), 1, f"{vname}: expected registration probe block in trigger_ddls")
            probe_section = guards[0].split("FOR probe_rec IN", 1)[1].split("END LOOP;", 1)[0]
            for frag in expected_fragments:
                self.assertIn(frag, probe_section, f"{vname}: missing {frag!r} in probe block")


class TestSqleanStatisticsRefusal(unittest.TestCase):
    """sqlean's `stats` extension defines its own VAR_*/STDDEV_* (a single-value
    sample statistic is 0.0, streaming accumulation), which differ from the
    engine-neutral exact definition matview maintains on SQLite. A view that
    uses them on a connection that really exposes them is refused explicitly;
    the global override does not bypass it; detection is never a guess."""

    SCHEMA = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT NOT NULL, v REAL)"

    @staticmethod
    def _flagged() -> EngineVersion:
        import dataclasses

        return dataclasses.replace(parse_engine_version("sqlite:3.50.4"), sqlite_native_statistics=True)

    def test_every_statistical_spelling_is_refused_even_with_ignore_limitations(self) -> None:
        from matview_sync import UnsupportedMaterializedViewError

        schemas = parse_table_schema(self.SCHEMA)
        for fn in ("VAR_POP", "VAR_SAMP", "VARIANCE", "STDDEV_POP", "STDDEV_SAMP", "STDDEV"):
            for shape in (
                f"SELECT g, {fn}(v) AS s FROM t GROUP BY g",
                f"SELECT g, 1 + {fn}(v) AS s FROM t GROUP BY g",
                f"SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING {fn}(v) > 1",
                f"WITH c AS (SELECT g, {fn}(v) AS s FROM t GROUP BY g) SELECT g, s FROM c",
            ):
                with self.subTest(shape=shape):
                    for ignore in (False, True):
                        # VARIANCE/STDDEV are refused on every SQLite connection (their
                        # sample/population meaning is engine-specific): another
                        # message, same explicit refusal.
                        pattern = "sqlean" if fn.startswith(("VAR_", "STDDEV_")) else "."
                        with self.assertRaisesRegex(UnsupportedMaterializedViewError, pattern):
                            generate_materialized_view(
                                schemas, f"CREATE VIEW v AS {shape}", dialect=Dialect.SQLITE,
                                engine_version=self._flagged(), ignore_limitations=ignore,
                                accept_limitations=FLOAT_OPT_IN,
                            )

    def test_other_views_and_unflagged_connections_are_unaffected(self) -> None:
        schemas = parse_table_schema(self.SCHEMA)
        generate_materialized_view(
            schemas, "CREATE VIEW v AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM t GROUP BY g",
            dialect=Dialect.SQLITE, engine_version=self._flagged(),
        )
        generate_materialized_view(
            schemas, "CREATE VIEW v AS SELECT g, VAR_POP(v) AS s FROM t GROUP BY g",
            dialect=Dialect.SQLITE, engine_version=parse_engine_version("sqlite:3.50.4"),
        )

    def test_detection_needs_the_sqlean_driver_and_its_functions(self) -> None:
        from matview.connectivity import detect_engine_version

        stats = [("var_pop",), ("stddev_samp",), ("sum",)]

        def make(module: str, functions):
            class Cursor:
                def __init__(self) -> None:
                    self.rows = []

                def execute(self, sql: str):
                    if sql.startswith("SELECT sqlite_version"):
                        self.rows = [("3.50.4",)]
                    elif sql.startswith("PRAGMA function_list"):
                        self.rows = list(functions)
                    return self

                def fetchone(self):
                    return self.rows[0] if self.rows else None

                def fetchall(self):
                    return list(self.rows)

            class Conn:
                dialect = Dialect.SQLITE

                def cursor(self):
                    return Cursor()

            Conn.__module__ = module
            return Conn()

        self.assertTrue(detect_engine_version(make("sqlean.dbapi2", stats)).sqlite_native_statistics)
        # A UDF of the same name on another driver is the caller's own function.
        self.assertFalse(detect_engine_version(make("apsw", stats)).sqlite_native_statistics)
        self.assertFalse(detect_engine_version(make("sqlite3", stats)).sqlite_native_statistics)
        # sqlean without its stats extension.
        self.assertFalse(detect_engine_version(make("sqlean.dbapi2", [("sum",)])).sqlite_native_statistics)


if __name__ == "__main__":
    unittest.main()
