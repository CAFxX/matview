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

from matview_sync import (
    Dialect,
    EngineVersion,
    IncrementalViewCompiler,
    MaterializedViewManager,
    TableSchema,
    generate_materialized_view,
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
        plan = generate_materialized_view(schemas, view_sql, dialect=Dialect.POSTGRES)
        compiler = IncrementalViewCompiler(schemas, plan.view_spec, dialect=Dialect.POSTGRES)

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
        compiler_mysql = IncrementalViewCompiler(schemas, plan.view_spec, dialect=Dialect.MYSQL, ignore_limitations=True)
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
        self.assertAlmostEqual(total, 42.0)
        self.assertAlmostEqual(v_pop, 0.0)
        self.assertAlmostEqual(s_pop, 0.0)
        self.assertIsNone(v_samp, "VAR_SAMP with N=1 must be NULL, not division by zero")
        self.assertIsNone(s_samp, "STDDEV_SAMP with N=1 must be NULL, not division by zero")

        # 3. Insert 2nd row: N = 2
        conn.execute("INSERT INTO sample_data VALUES (2, 100, 48.0);")
        row = conn.execute("SELECT cnt, total, v_pop, v_samp, s_pop, s_samp FROM mv_stats WHERE grp = 100;").fetchone()
        cnt, total, v_pop, v_samp, s_pop, s_samp = row
        self.assertEqual(cnt, 2)
        self.assertAlmostEqual(total, 90.0)
        # vals: 42, 48. mean = 45. diffs: -3, +3. sq diffs: 9, 9.
        # sum sq diffs = 18.
        # pop var = 18 / 2 = 9.0, pop stddev = 3.0
        # samp var = 18 / 1 = 18.0, samp stddev = sqrt(18) = 4.24264...
        self.assertAlmostEqual(v_pop, 9.0)
        self.assertAlmostEqual(s_pop, 3.0)
        self.assertAlmostEqual(v_samp, 18.0)
        self.assertAlmostEqual(s_samp, math.sqrt(18.0))

        # 4. Delete 1 row: back to N = 1
        conn.execute("DELETE FROM sample_data WHERE id = 2;")
        row = conn.execute("SELECT cnt, total, v_pop, v_samp, s_pop, s_samp FROM mv_stats WHERE grp = 100;").fetchone()
        cnt, total, v_pop, v_samp, s_pop, s_samp = row
        self.assertEqual(cnt, 1)
        self.assertAlmostEqual(total, 42.0)
        self.assertAlmostEqual(v_pop, 0.0)
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
                ignore_limitations=True,
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
                ignore_limitations=True,
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
                ignore_limitations=True,
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
                ignore_limitations=True,
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
                ignore_limitations=True,
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
            ignore_limitations=True,
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
                ignore_limitations=True,
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
                incremental_backfill=True,
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
            incremental_backfill=True,
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
            incremental_backfill=True,
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
            pg_fillfactor=90,
        )
        self.assertIn("WITH (fillfactor = 90)", plan90.backing_table_ddl)

        # Disabled via None
        plan_none = generate_materialized_view(
            self.schemas,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_fillfactor=None,
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
        self.assertAlmostEqual(val, 150.0)

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
            ignore_limitations=True,
        )
        self._assert_subquery_aliases_cover_refs(p_grp.incremental_backfill_spec.chunk_sql_templates[0], "__mv_src", "mysql")

        # 2. global aggregate incremental populate HAVING placement
        p_glob = generate_materialized_view(
            schemas,
            "CREATE VIEW mv_glob AS SELECT SUM(val) AS tot FROM items;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True,
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
            ignore_limitations=True,
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
            ignore_limitations=True,
        )
        self._assert_subquery_aliases_cover_refs(p_proj.incremental_backfill_spec.chunk_sql_templates[0], "__mv_src", "mysql")
        p_proj_bag = generate_materialized_view(
            schemas,
            "CREATE VIEW mv_proj_bag AS SELECT grp, val FROM items;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True,
        )
        self._assert_subquery_aliases_cover_refs(p_proj_bag.incremental_backfill_spec.chunk_sql_templates[0], "__mv_src", "mysql")
        p_no_pk = generate_materialized_view(
            schemas_no_pk,
            "CREATE VIEW mv_proj_npk AS SELECT grp, val FROM logs;",
            dialect=Dialect.MYSQL,
            engine_version="mysql:8.4.0",
            incremental_backfill=True,
            ignore_limitations=True,
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
        mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True)
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
                    self.assertAlmostEqual(float(r_mv[2]), float(r_ora[2]), places=6)
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
        mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES, ignore_limitations=True)
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
                self.assertIn("fillfactor = 80", "\n".join(p_agg.backing_table_ddls))

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
            cur.execute(
                f"SELECT dept, SUM(amount) AS s, STDDEV_SAMP(amount) AS sd, COUNT(DISTINCT tag) AS dc FROM {base_tbl} GROUP BY dept ORDER BY dept;"
            )
            ora_agg_rows = cur.fetchall()
            self.assertEqual(len(mv_agg_rows), len(ora_agg_rows))
            for r_mv, r_ora in zip(mv_agg_rows, ora_agg_rows):
                self.assertEqual(r_mv[0], r_ora[0])
                self.assertAlmostEqual(float(r_mv[1]), float(r_ora[1]), places=6)
                self.assertEqual(r_mv[3], r_ora[3])
                if r_ora[2] is None:
                    self.assertIsNone(r_mv[2])
                else:
                    self.assertAlmostEqual(float(r_mv[2]), float(r_ora[2]), places=6)
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


if __name__ == "__main__":
    unittest.main()
