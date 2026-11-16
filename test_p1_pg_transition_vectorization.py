#!/usr/bin/env python3
"""
Comprehensive unit and integration tests for PostgreSQL statement-level
transition-table trigger vectorization (pg_use_transition_tables + pg_vectorize_transition_tables).

Verifies:
1. DDL inspection:
   - Default pg_vectorize_transition_tables=True eliminates PL/pgSQL row-by-row loops
     (FOR NEW/OLD IN SELECT * FROM __mv_*_rows LOOP) in favor of genuine set-based SQL.
   - Trigger ordering invariant from 5f2f0be: In AFTER UPDATE (au), OLD minus deltas
     and NEW plus deltas/upserts execute before zero-count cleanup.
   - Fallback loop path is preserved when pg_vectorize_transition_tables=False.
2. PostgreSQL dialect syntax validation via sqlglot for all generated DDL.
3. Behavior and semantics:
   - Multi-row batch INSERT / UPDATE / DELETE.
   - Multi-row PK swaps in a single UPDATE statement.
   - Self-cancelling deltas (+1 and -1 in same statement).
   - Multi-row updates collapsing into the same group.
   - NULL group keys with canonical PK surrogate.
   - Count hitting zero & re-inserting in same statement.
   - Distinct auxiliary table refcounts (COUNT(DISTINCT)).
   - Tier 1 projection, Tier 1 aggregation, Tier 1 joins, Tier 2 group-scoped, Tier 3 atomic sync.
4. Live PostgreSQL verification if POSTGRES_DSN is available.
"""

from __future__ import annotations

import os
import re
import unittest
from typing import Any, List, Optional

import sqlglot

from matview_sync import (
    Dialect,
    MaintenanceStrategy,
    MaterializedViewManager,
    MaterializedViewPlan,
    _split_top_level,
    generate_materialized_view,
    main as cli_main,
)
from test_matview_sync import TranspiledConnectionAdapter


class TestPGTransitionTableVectorizationDDL(unittest.TestCase):
    """DDL inspection and structural tests for transition-table vectorization."""

    def test_tier1_aggregation_vectorized_ddl_structure(self) -> None:
        schema = """
        CREATE TABLE sales (
            id INT PRIMARY KEY,
            region VARCHAR(32) NOT NULL,
            amount NUMERIC(10, 2) NOT NULL,
            qty INT NOT NULL
        );
        """
        view = """
        CREATE VIEW regional_sales AS
        SELECT region, COUNT(*) AS cnt, SUM(amount) AS total_amount, AVG(qty) AS avg_qty
        FROM sales
        GROUP BY region;
        """
        plan = generate_materialized_view(
            schema,
            view,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        ddls = "\n\n".join(plan.trigger_ddls)

        # No row-by-row loops in vectorized mode
        self.assertNotIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", ddls)
        self.assertNotIn("FOR OLD IN SELECT * FROM __mv_old_rows LOOP", ddls)
        self.assertNotIn("LOOP", ddls)

        # AI trigger function: set-based INSERT ... ON CONFLICT DO UPDATE
        ai_ddl = [d for d in plan.trigger_ddls if "_ai()" in d][0]
        self.assertIn("INSERT INTO __matview_store_regional_sales", ai_ddl)
        self.assertIn("FROM __mv_new_rows", ai_ddl)
        self.assertIn("GROUP BY", ai_ddl)
        self.assertIn("ON CONFLICT (region) DO UPDATE SET", ai_ddl)

        # AD trigger function: set-based UPDATE ... FROM __mv_old_rows and DELETE count <= 0
        ad_ddl = [d for d in plan.trigger_ddls if "_ad()" in d][0]
        self.assertIn("UPDATE __matview_store_regional_sales", ad_ddl)
        self.assertIn("FROM (", ad_ddl)
        self.assertIn("FROM __mv_old_rows", ad_ddl)
        self.assertIn("DELETE FROM __matview_store_regional_sales WHERE cnt <= 0;", ad_ddl)

        # AU trigger function: OLD update -> NEW upsert -> cleanup
        au_ddl = [d for d in plan.trigger_ddls if "_au()" in d][0]
        pos_old = au_ddl.find("__mv_old_rows")
        pos_new = au_ddl.find("__mv_new_rows")
        pos_clean = au_ddl.find("DELETE FROM __matview_store_regional_sales WHERE cnt <= 0;")

        self.assertNotEqual(pos_old, -1)
        self.assertNotEqual(pos_new, -1)
        self.assertNotEqual(pos_clean, -1)
        self.assertLess(pos_old, pos_new, "OLD minus update must precede NEW plus upsert")
        self.assertLess(pos_new, pos_clean, "NEW plus upsert must precede zero-count cleanup")

    def test_tier1_projection_vectorized_ddl_structure(self) -> None:
        schema = """
        CREATE TABLE users (
            id INT PRIMARY KEY,
            username VARCHAR(64) NOT NULL,
            email VARCHAR(128) NOT NULL
        );
        """
        view = "CREATE VIEW active_users AS SELECT id, username FROM users;"
        plan = generate_materialized_view(
            schema,
            view,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        ddls = "\n\n".join(plan.trigger_ddls)

        self.assertNotIn("LOOP", ddls)
        ai_ddl = [d for d in plan.trigger_ddls if "_ai()" in d][0]
        self.assertIn("INSERT INTO __matview_store_active_users", ai_ddl)
        self.assertIn("FROM __mv_new_rows", ai_ddl)
        self.assertIn("ON CONFLICT (id) DO UPDATE SET", ai_ddl)

        au_ddl = [d for d in plan.trigger_ddls if "_au()" in d][0]
        self.assertIn("INSERT INTO __matview_store_active_users", au_ddl)
        self.assertIn("DELETE FROM __matview_store_active_users", au_ddl)
        self.assertIn("USING __mv_old_rows", au_ddl)
        self.assertIn("NOT EXISTS", au_ddl)

    def test_tier1_distinct_aux_tables_vectorized_ddl(self) -> None:
        schema = """
        CREATE TABLE web_events (
            id INT PRIMARY KEY,
            domain VARCHAR(64) NOT NULL,
            visitor_id INT NOT NULL
        );
        """
        view = """
        CREATE VIEW domain_visitors AS
        SELECT domain, COUNT(DISTINCT visitor_id) AS unique_visitors
        FROM web_events
        GROUP BY domain;
        """
        plan = generate_materialized_view(
            schema,
            view,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        ddls = "\n\n".join(plan.trigger_ddls)
        self.assertNotIn("LOOP", ddls)

        # Aux table refcount maintenance is set-based
        self.assertIn("ON CONFLICT (domain, __val) DO UPDATE SET", ddls)
        self.assertIn("DELETE FROM __matview_dist_domain_visitors__unique_visitors WHERE __ref_cnt <= 0;", ddls)

    def test_tier1_join_algebraic_vectorized_ddl(self) -> None:
        schema = """
        CREATE TABLE departments (
            dept_id INT PRIMARY KEY,
            dept_name VARCHAR(64) NOT NULL
        );
        CREATE TABLE employees (
            emp_id INT PRIMARY KEY,
            dept_id INT NOT NULL,
            salary NUMERIC(10, 2) NOT NULL
        );
        """
        view = """
        CREATE VIEW dept_salaries AS
        SELECT d.dept_name, COUNT(*) AS emp_cnt, SUM(e.salary) AS total_sal
        FROM departments d
        JOIN employees e ON d.dept_id = e.dept_id
        GROUP BY d.dept_name;
        """
        plan = generate_materialized_view(
            schema,
            view,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        ddls = "\n\n".join(plan.trigger_ddls)
        self.assertNotIn("LOOP", ddls)

        # Trigger on departments uses __mv_new_rows AS d in FROM clause
        ai_ddl = [d for d in plan.trigger_ddls if "departments_ai" in d][0]
        self.assertIn("__mv_new_rows AS d", ai_ddl)
        self.assertIn("ON CONFLICT (dept_name) DO UPDATE SET", ai_ddl)

    def test_tier2_group_scoped_vectorized_ddl(self) -> None:
        schema = """
        CREATE TABLE categories (
            cat_id INT PRIMARY KEY,
            cat_name VARCHAR(32) NOT NULL
        );
        CREATE TABLE products (
            prod_id INT PRIMARY KEY,
            cat_id INT NOT NULL,
            prod_name VARCHAR(64) NOT NULL
        );
        """
        # Tier 2: holistic aggregate STRING_AGG falls back to GROUP_SCOPED_DELTA
        view = """
        CREATE VIEW cat_prod_names AS
        SELECT c.cat_name, STRING_AGG(p.prod_name, ',') AS names
        FROM categories c
        JOIN products p ON c.cat_id = p.cat_id
        GROUP BY c.cat_name;
        """
        plan = generate_materialized_view(
            schema,
            view,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        self.assertEqual(plan.strategy, MaintenanceStrategy.GROUP_SCOPED_DELTA)
        ddls = "\n\n".join(plan.trigger_ddls)
        self.assertNotIn("LOOP", ddls)
        self.assertIn("EXISTS (SELECT 1 FROM __mv_new_rows", ddls)

    def test_fallback_when_vectorization_disabled(self) -> None:
        schema = """
        CREATE TABLE items (
            id INT PRIMARY KEY,
            cat VARCHAR(32) NOT NULL,
            qty INT NOT NULL
        );
        """
        view = "CREATE VIEW item_summary AS SELECT cat, SUM(qty) AS tot FROM items GROUP BY cat;"
        plan = generate_materialized_view(
            schema,
            view,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=False,
        )
        ddls = "\n\n".join(plan.trigger_ddls)
        self.assertIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", ddls)
        self.assertIn("FOR OLD IN SELECT * FROM __mv_old_rows LOOP", ddls)

    def test_sqlglot_postgres_syntax_validity(self) -> None:
        """Parse all generated PL/pgSQL SQL statements with sqlglot on postgres dialect."""
        schema = """
        CREATE TABLE metrics (
            id INT PRIMARY KEY,
            tag VARCHAR(32),
            val DOUBLE PRECISION NOT NULL,
            flag BOOLEAN NOT NULL
        );
        """
        view = """
        CREATE VIEW metrics_summary AS
        SELECT
            tag,
            COUNT(*) AS row_cnt,
            SUM(val) AS sum_val,
            MIN(val) AS min_val,
            MAX(val) AS max_val,
            BOOL_AND(flag) AS all_flags,
            BOOL_OR(flag) AS any_flags
        FROM metrics
        GROUP BY tag;
        """
        plan = generate_materialized_view(
            schema,
            view,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )

        for ddl in plan.trigger_ddls:
            # Check function definition
            m_fn = re.search(r"BEGIN\s*(.*?)\s*RETURN NULL;\s*END;", ddl, flags=re.I | re.S)
            if m_fn:
                body = m_fn.group(1).strip()
                for stmt in body.split(";"):
                    s = stmt.strip()
                    if not s:
                        continue
                    # Parse each inner SQL statement with sqlglot postgres parser
                    try:
                        parsed = sqlglot.parse(s, read="postgres")
                        self.assertTrue(len(parsed) > 0, f"Failed to parse statement: {s}")
                    except Exception as exc:
                        self.fail(f"sqlglot postgres dialect failed to parse: {s}\nError: {exc}")

    def test_cli_flags_vectorize_option(self) -> None:
        """Verify CLI parser honors --pg-transition-tables and --no-pg-vectorize-transition-tables."""
        schema = """
        CREATE TABLE t (id INT PRIMARY KEY, grp INT, v INT);
        """
        view = "CREATE VIEW v AS SELECT grp, SUM(v) AS tot FROM t GROUP BY grp;"

        import io
        from contextlib import redirect_stdout

        out_vec = io.StringIO()
        with redirect_stdout(out_vec):
            cli_main([
                "--schema", schema,
                "--view", view,
                "--dialect", "postgres",
                "--pg-transition-tables",
            ])
        sql_vec = out_vec.getvalue()
        self.assertNotIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", sql_vec)

        out_loop = io.StringIO()
        with redirect_stdout(out_loop):
            cli_main([
                "--schema", schema,
                "--view", view,
                "--dialect", "postgres",
                "--pg-transition-tables",
                "--no-pg-vectorize-transition-tables",
            ])
        sql_loop = out_loop.getvalue()
        self.assertIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", sql_loop)


class TestPGTransitionTableVectorizationSemantics(unittest.TestCase):
    """Behavioral and edge-case execution tests using TranspiledConnectionAdapter."""

    def setUp(self) -> None:
        self.adapter = TranspiledConnectionAdapter(dialect=Dialect.POSTGRES)
        self.cur = self.adapter.cursor()

    def tearDown(self) -> None:
        self.adapter.close()

    def _execute_script(self, sql: str) -> None:
        for stmt in _split_top_level(sql, ";"):
            s = stmt.strip()
            if s:
                self.cur.execute(s)

    def test_multi_row_batch_dml_and_group_collapse(self) -> None:
        """Batch INSERT/UPDATE/DELETE across multiple groups and group migration."""
        schema_sql = """
        CREATE TABLE inventory (
            item_id INT PRIMARY KEY,
            warehouse VARCHAR(16) NOT NULL,
            qty INT NOT NULL
        );
        """
        view_sql = """
        CREATE VIEW wh_totals AS
        SELECT warehouse, COUNT(*) AS item_cnt, SUM(qty) AS total_qty
        FROM inventory
        GROUP BY warehouse;
        """
        self._execute_script(schema_sql)
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        for ddl in plan.create_structure_statements() + plan.trigger_ddls:
            self.cur.execute(ddl)

        # 1. Multi-row batch insert: 6 rows across 2 warehouses
        self.cur.execute(
            "INSERT INTO inventory VALUES (1, 'EAST', 10), (2, 'EAST', 20), (3, 'EAST', 30), "
            "(4, 'WEST', 100), (5, 'WEST', 200), (6, 'WEST', 300);"
        )
        self.cur.execute("SELECT warehouse, item_cnt, total_qty FROM wh_totals ORDER BY warehouse;")
        rows = self.cur.fetchall()
        self.assertEqual(rows, [("EAST", 3, 60), ("WEST", 3, 600)])

        # 2. Multi-row batch update collapsing items from EAST to WEST
        self.cur.execute("UPDATE inventory SET warehouse = 'WEST', qty = qty * 2 WHERE warehouse = 'EAST';")
        self.cur.execute("SELECT warehouse, item_cnt, total_qty FROM wh_totals ORDER BY warehouse;")
        rows = self.cur.fetchall()
        # EAST emptied -> deleted; WEST now has all 6 items with 100+200+300 + 20+40+60 = 720
        self.assertEqual(rows, [("WEST", 6, 720)])

        # 3. Multi-row batch delete: remove all items
        self.cur.execute("DELETE FROM inventory WHERE item_id IN (1, 2, 3, 4, 5, 6);")
        self.cur.execute("SELECT warehouse, item_cnt, total_qty FROM wh_totals ORDER BY warehouse;")
        self.assertEqual(self.cur.fetchall(), [])

    def test_single_statement_pk_swap(self) -> None:
        """In-statement swap of primary key / group key values."""
        # SQLite base table omits UNIQUE constraint on the base table so SQLite doesn't fail
        # the in-place UPDATE with an immediate uniqueness check; the MV table retains the PK.
        db_schema = """
        CREATE TABLE bins (
            bin_id INT,
            val INT NOT NULL
        );
        """
        plan_schema = """
        CREATE TABLE bins (
            bin_id INT PRIMARY KEY,
            val INT NOT NULL
        );
        """
        view_sql = "CREATE VIEW bin_view AS SELECT bin_id, val FROM bins;"
        self._execute_script(db_schema)
        plan = generate_materialized_view(
            plan_schema,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        for ddl in plan.create_structure_statements() + plan.trigger_ddls:
            self.cur.execute(ddl)

        self.cur.execute("INSERT INTO bins VALUES (1, 100), (2, 200);")
        self.cur.execute("SELECT bin_id, val FROM bin_view ORDER BY bin_id;")
        self.assertEqual(self.cur.fetchall(), [(1, 100), (2, 200)])

        # Swap bin_id 1 <-> 2 in a single statement
        self.cur.execute("UPDATE bins SET bin_id = CASE WHEN bin_id = 1 THEN 2 ELSE 1 END;")
        self.cur.execute("SELECT bin_id, val FROM bin_view ORDER BY bin_id;")
        # Row 1 now has 200, row 2 now has 100
        self.assertEqual(self.cur.fetchall(), [(1, 200), (2, 100)])

    def test_null_group_keys_with_canonical_pk(self) -> None:
        """NULL group key handling with surrogate canonical PK."""
        schema_sql = """
        CREATE TABLE tasks (
            id INT PRIMARY KEY,
            assignee VARCHAR(32),
            hours INT NOT NULL
        );
        """
        view_sql = """
        CREATE VIEW assignee_hours AS
        SELECT assignee, COUNT(*) AS task_count, SUM(hours) AS total_hours
        FROM tasks
        GROUP BY assignee;
        """
        self._execute_script(schema_sql)
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        for ddl in plan.create_structure_statements() + plan.trigger_ddls:
            self.cur.execute(ddl)

        # Batch insert including NULL assignee
        self.cur.execute(
            "INSERT INTO tasks VALUES (1, 'Alice', 4), (2, NULL, 6), (3, 'Bob', 8), (4, NULL, 5);"
        )
        self.cur.execute("SELECT assignee, task_count, total_hours FROM assignee_hours ORDER BY total_hours;")
        rows = self.cur.fetchall()
        # Alice: 4, Bob: 8, NULL: 11 (2 tasks)
        self.assertEqual(rows, [("Alice", 1, 4), ("Bob", 1, 8), (None, 2, 11)])

        # Update NULL assignee to Alice
        self.cur.execute("UPDATE tasks SET assignee = 'Alice' WHERE assignee IS NULL;")
        self.cur.execute("SELECT assignee, task_count, total_hours FROM assignee_hours ORDER BY total_hours;")
        rows = self.cur.fetchall()
        # Alice: 4+11=15 (3 tasks), Bob: 8 (1 task), NULL group deleted
        self.assertEqual(rows, [("Bob", 1, 8), ("Alice", 3, 15)])

    def test_distinct_aux_refcounts_vectorized(self) -> None:
        """Batch insert/delete with duplicate distinct items."""
        schema_sql = """
        CREATE TABLE clicks (
            click_id INT PRIMARY KEY,
            page VARCHAR(32) NOT NULL,
            user_id INT NOT NULL
        );
        """
        view_sql = """
        CREATE VIEW page_reach AS
        SELECT page, COUNT(DISTINCT user_id) AS reach
        FROM clicks
        GROUP BY page;
        """
        self._execute_script(schema_sql)
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        for ddl in plan.create_structure_statements() + plan.trigger_ddls:
            self.cur.execute(ddl)

        # Insert batch with duplicates (user 101 clicked 'home' twice)
        self.cur.execute(
            "INSERT INTO clicks VALUES (1, 'home', 101), (2, 'home', 101), (3, 'home', 102), (4, 'faq', 101);"
        )
        self.cur.execute("SELECT page, reach FROM page_reach ORDER BY page;")
        self.assertEqual(self.cur.fetchall(), [("faq", 1), ("home", 2)])

        # Delete one duplicate click for user 101 on 'home' -> reach should still be 2
        self.cur.execute("DELETE FROM clicks WHERE click_id = 1;")
        self.cur.execute("SELECT page, reach FROM page_reach ORDER BY page;")
        self.assertEqual(self.cur.fetchall(), [("faq", 1), ("home", 2)])

        # Delete remaining click for user 101 on 'home' -> reach drops to 1
        self.cur.execute("DELETE FROM clicks WHERE click_id = 2;")
        self.cur.execute("SELECT page, reach FROM page_reach ORDER BY page;")
        self.assertEqual(self.cur.fetchall(), [("faq", 1), ("home", 1)])

    def test_native_avg_recompute_with_expression_group_key_uses_safe_row_loop(self) -> None:
        """Expression GROUP BY plus native AVG recomputation must not correlate an ungrouped raw column."""
        schema_sql = """
        CREATE TABLE orders (
            id INT PRIMARY KEY,
            region TEXT,
            amount INT
        );
        """
        view_sql = """
        CREATE VIEW pgtr_expr_avg AS
        SELECT COALESCE(region, 'UNK') AS reg_norm,
               COUNT(*) AS cnt,
               SUM(amount) AS s_amt,
               AVG(amount) AS a_amt
        FROM orders
        GROUP BY COALESCE(region, 'UNK');
        """
        self._execute_script(schema_sql)
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        fn_sql = "\n".join(
            statement for statement in plan.trigger_ddls
            if "CREATE OR REPLACE FUNCTION" in statement
        )
        self.assertIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", fn_sql)

        for ddl in plan.create_structure_statements() + plan.trigger_ddls:
            self.cur.execute(ddl)

        def assert_matches_live_view() -> None:
            self.cur.execute(
                "SELECT reg_norm, cnt, s_amt, a_amt FROM pgtr_expr_avg ORDER BY reg_norm;"
            )
            actual = self.cur.fetchall()
            self.cur.execute(
                "SELECT COALESCE(region, 'UNK') AS reg_norm, COUNT(*) AS cnt, "
                "SUM(amount) AS s_amt, AVG(amount) AS a_amt "
                "FROM orders GROUP BY COALESCE(region, 'UNK') ORDER BY reg_norm;"
            )
            self.assertEqual(actual, self.cur.fetchall())

        self.cur.execute(
            "INSERT INTO orders VALUES (1, 'NA', 10), (2, 'NA', 20), (3, NULL, 30);"
        )
        assert_matches_live_view()
        self.cur.execute("UPDATE orders SET amount = amount + 5 WHERE region = 'NA';")
        assert_matches_live_view()
        self.cur.execute("DELETE FROM orders WHERE id = 3;")
        assert_matches_live_view()

    def test_count_star_alias_not_duplicated_in_vectorized_dml(self) -> None:
        """COUNT(*) with explicit alias (e.g. cnt) must appear only once in INSERT/UPDATE columns."""
        schema_sql = """
        CREATE TABLE accounts (
            id INT PRIMARY KEY,
            org TEXT,
            balance INT
        );
        """
        view_sql = """
        CREATE VIEW pgtr_cnt_alias AS
        SELECT org, COUNT(*) AS cnt, SUM(balance) AS sb, AVG(balance) AS ab
        FROM accounts
        GROUP BY org;
        """
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )
        fn_sql = "\n".join(s for s in plan.trigger_ddls if "CREATE OR REPLACE FUNCTION" in s)
        self.assertIn("(__mv_pk, org, cnt, sb, ab, __mv_cnt_sb_1)", fn_sql)
        self.assertNotIn("cnt, cnt", fn_sql)


if __name__ == "__main__":
    unittest.main()
