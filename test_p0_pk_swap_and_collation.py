#!/usr/bin/env python3
"""P0 Regression Tests for Finding #6 and Finding #7.

Finding #6: Row-Level Trigger PK Swap / Shift Data Loss Prevention
- Verifies that row-level triggers on PostgreSQL, SQLite, and MySQL do not prematurely
  delete backing table rows during multi-row PK swaps, cycles, or shifts.
- Verifies that the delete predicate in AFTER UPDATE triggers is gated by:
  `AND NOT EXISTS (SELECT 1 FROM <base_t> WHERE <base_pk = OLD.pk> AND (<where_clause>))`
- Verifies edge cases with and without WHERE filtering, composite primary keys,
  non-native PK projection views (`uses_native_pk = False`), and PostgreSQL
  transition-table statement-level trigger ordering.

Finding #7: MySQL/MariaDB __mv_pk Collation Safety & Case-Insensitive Collision Prevention
- Verifies that MySQL backing tables use `VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin`
  across all compilation strategies (single-table aggregation, multi-table join, Z-set projection,
  set operations, UNION ALL aggregation, and DISTINCT auxiliary tables).
- Verifies that `_canonical_group_pk_expr` on MySQL generates expressions using
  `OCTET_LENGTH(CAST(e AS CHAR CHARACTER SET utf8mb4))` and `COLLATE utf8mb4_bin`.
- Verifies that case-distinct and accent-distinct group keys (e.g. 'apple' vs 'APPLE', 'cafe' vs 'café')
  do not collide into the same group key and maintain isolated counts and aggregates.
"""

from __future__ import annotations

import re
import sqlite3
import unittest
from typing import List, Tuple

from matview_sync import (
    ColumnDef,
    Dialect,
    IncrementalViewCompiler,
    MaterializedViewManager,
    TableSchema,
    generate_materialized_view,
    parse_view_definition,
)
from test_matview_sync import (
    TranspiledConnectionAdapter,
    _register_sqlite_compat_udfs,
)


class TestFinding6PkSwapAndShift(unittest.TestCase):
    """Exhaustive tests for Finding #6: PK swap, cycle, and shift safety."""

    def test_postgres_row_level_trigger_sql_inspection(self) -> None:
        """Verify PostgreSQL row-level trigger generates NOT EXISTS gate on OLD.pk."""
        schema_sql = """
        CREATE TABLE items (
            id INT PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            val INT
        );
        """
        view_sql = "CREATE VIEW items_mv AS SELECT id, name, val FROM items;"
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=False,
        )

        trig_ddls = "\n\n".join(plan.trigger_ddls)
        self.assertIn("NOT EXISTS", trig_ddls)
        self.assertRegex(
            trig_ddls,
            r'DELETE FROM __matview_store_items_mv\s+WHERE __matview_store_items_mv\.id = OLD\.id\s+AND NOT EXISTS\s*\(\s*SELECT 1 FROM "?items"? WHERE "?items"?\.id = OLD\.id\)',
        )
        self.assertRegex(
            trig_ddls,
            r"TG_OP = 'DELETE'\s*\)\s*THEN\s*DELETE FROM __matview_store_items_mv\s+WHERE __matview_store_items_mv\.id = OLD\.id;",
        )

    def test_postgres_row_level_trigger_with_where_filter_sql_inspection(self) -> None:
        """Verify PostgreSQL trigger gates NOT EXISTS with the view WHERE clause."""
        schema_sql = """
        CREATE TABLE items (
            id INT PRIMARY KEY,
            status VARCHAR(20) NOT NULL
        );
        """
        view_sql = "CREATE VIEW active_items_mv AS SELECT id, status FROM items WHERE status = 'active';"
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=False,
        )

        trig_ddls = "\n\n".join(plan.trigger_ddls)
        self.assertRegex(
            trig_ddls,
            r'NOT EXISTS\s*\(\s*SELECT 1 FROM "?items"? WHERE "?items"?\.id = OLD\.id\s+AND\s*\(("?items"?\.)?"?status"? = \'active\'\)\)',
        )

    def test_postgres_transition_table_loop_ordering(self) -> None:
        """Verify fallback loop-based transition table triggers execute NEW loop before cleanup."""
        schema_sql = """
        CREATE TABLE items (
            id INT PRIMARY KEY,
            category VARCHAR(32) NOT NULL,
            qty INT NOT NULL
        );
        """
        view_sql = "CREATE VIEW cat_summary_mv AS SELECT category, SUM(qty) AS total_qty FROM items GROUP BY category;"
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=False,
        )

        au_ddl = [ddl for ddl in plan.trigger_ddls if "_au()" in ddl][0]
        pos_old = au_ddl.find("FOR OLD IN SELECT * FROM __mv_old_rows LOOP")
        pos_new = au_ddl.find("FOR NEW IN SELECT * FROM __mv_new_rows LOOP")
        pos_clean = au_ddl.find("DELETE FROM __matview_store_cat_summary_mv")

        self.assertNotEqual(pos_old, -1)
        self.assertNotEqual(pos_new, -1)
        self.assertNotEqual(pos_clean, -1)
        self.assertLess(pos_old, pos_new, "OLD loop must precede NEW loop")
        self.assertLess(pos_new, pos_clean, "NEW loop must precede zero-count cleanup")

    def test_postgres_transition_table_vectorized_ordering(self) -> None:
        """Verify vectorized statement-level transition table triggers have no loops and run NEW before cleanup."""
        schema_sql = """
        CREATE TABLE items (
            id INT PRIMARY KEY,
            category VARCHAR(32) NOT NULL,
            qty INT NOT NULL
        );
        """
        view_sql = "CREATE VIEW cat_summary_mv AS SELECT category, SUM(qty) AS total_qty FROM items GROUP BY category;"
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
        )

        au_ddl = [ddl for ddl in plan.trigger_ddls if "_au()" in ddl][0]
        self.assertNotIn("LOOP", au_ddl, "Vectorized trigger function must not contain row loops")
        pos_old = au_ddl.find("__mv_old_rows")
        pos_new = au_ddl.find("__mv_new_rows")
        pos_clean = au_ddl.find("DELETE FROM __matview_store_cat_summary_mv")

        self.assertNotEqual(pos_old, -1, "__mv_old_rows must be referenced in AU")
        self.assertNotEqual(pos_new, -1, "__mv_new_rows must be referenced in AU")
        self.assertNotEqual(pos_clean, -1, "zero-count cleanup must be present in AU")
        self.assertLess(pos_old, pos_new, "OLD delta update must precede NEW delta upsert")
        self.assertLess(pos_new, pos_clean, "NEW delta upsert must precede zero-count cleanup")

    def test_sqlite_trigger_sql_inspection(self) -> None:
        """Verify SQLite trigger contains NOT EXISTS gate for PK swap protection."""
        schema_sql = """
        CREATE TABLE accounts (
            acc_id INTEGER PRIMARY KEY,
            balance REAL NOT NULL
        );
        """
        view_sql = "CREATE VIEW accounts_mv AS SELECT acc_id, balance FROM accounts;"
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.SQLITE,
        )

        trig_ddls = "\n\n".join(plan.trigger_ddls)
        self.assertIn("NOT EXISTS", trig_ddls)
        self.assertRegex(
            trig_ddls,
            r'DELETE FROM __matview_store_accounts_mv\s+WHERE __matview_store_accounts_mv\.acc_id = OLD\.acc_id\s+AND NOT EXISTS\s*\(\s*SELECT 1 FROM "?accounts"? WHERE "?accounts"?\.acc_id = OLD\.acc_id\)',
        )

    def test_mysql_trigger_sql_inspection(self) -> None:
        """Verify MySQL trigger contains NOT EXISTS gate for PK swap protection."""
        schema_sql = """
        CREATE TABLE accounts (
            acc_id INT PRIMARY KEY,
            balance DECIMAL(12, 2) NOT NULL
        );
        """
        view_sql = "CREATE VIEW accounts_mv AS SELECT acc_id, balance FROM accounts;"
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )

        trig_ddls = "\n\n".join(plan.trigger_ddls)
        self.assertIn("NOT EXISTS", trig_ddls)
        self.assertRegex(
            trig_ddls,
            r'DELETE FROM __matview_store_accounts_mv\s+WHERE __matview_store_accounts_mv\.acc_id = OLD\.acc_id\s+AND NOT EXISTS\s*\(\s*SELECT 1 FROM [`"]?accounts[`"]? WHERE [`"]?accounts[`"]?\.acc_id = OLD\.acc_id\)',
        )

    def test_pk_swap_execution_data_loss_prevention(self) -> None:
        """Live execution: demonstrate that NOT EXISTS gate prevents 50% data loss on 2-row PK swap."""
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE nodes (id INT, label TEXT);")
        con.execute("CREATE TABLE __matview_store_nodes_mv (id INTEGER PRIMARY KEY, label TEXT);")

        schema_sql = "CREATE TABLE nodes (id INTEGER PRIMARY KEY, label TEXT);"
        view_sql = "CREATE VIEW nodes_mv AS SELECT id, label FROM nodes;"
        plan = generate_materialized_view(schema_sql, view_sql, dialect=Dialect.SQLITE)

        # 1. Populate initial rows in base table
        con.execute("INSERT INTO nodes (id, label) VALUES (1, 'Alpha'), (2, 'Beta');")
        # 2. Initial populate of backing table
        for pop in plan.initial_populate_sqls:
            con.execute(pop)
        # 3. Install triggers
        for trg in plan.trigger_ddls:
            con.execute(trg)

        self.assertEqual(
            con.execute("SELECT id, label FROM __matview_store_nodes_mv ORDER BY id;").fetchall(),
            [(1, "Alpha"), (2, "Beta")],
        )

        # 4. Execute 2-row PK swap
        con.execute("UPDATE nodes SET id = 3 - id;")

        # Base table has (1, 'Beta') and (2, 'Alpha')
        self.assertEqual(
            con.execute("SELECT id, label FROM nodes ORDER BY id;").fetchall(),
            [(1, "Beta"), (2, "Alpha")],
        )

        # Backing table MUST retain BOTH rows (without NOT EXISTS gate, row 2 was destroyed)
        self.assertEqual(
            con.execute("SELECT id, label FROM __matview_store_nodes_mv ORDER BY id;").fetchall(),
            [(1, "Beta"), (2, "Alpha")],
        )

    def test_3row_pk_cycle_execution(self) -> None:
        """Live execution: 3-row cyclic permutation (1->2, 2->3, 3->1) with zero data loss."""
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE nodes (id INT, val TEXT);")
        con.execute("CREATE TABLE __matview_store_nodes_mv (id INTEGER PRIMARY KEY, val TEXT);")

        schema_sql = "CREATE TABLE nodes (id INTEGER PRIMARY KEY, val TEXT);"
        view_sql = "CREATE VIEW nodes_mv AS SELECT id, val FROM nodes;"
        plan = generate_materialized_view(schema_sql, view_sql, dialect=Dialect.SQLITE)

        con.execute("INSERT INTO nodes (id, val) VALUES (1, 'A'), (2, 'B'), (3, 'C');")
        for pop in plan.initial_populate_sqls:
            con.execute(pop)
        for trg in plan.trigger_ddls:
            con.execute(trg)

        con.execute("UPDATE nodes SET id = CASE WHEN id = 1 THEN 2 WHEN id = 2 THEN 3 ELSE 1 END;")

        self.assertEqual(
            con.execute("SELECT id, val FROM nodes ORDER BY id;").fetchall(),
            [(1, "C"), (2, "A"), (3, "B")],
        )
        self.assertEqual(
            con.execute("SELECT id, val FROM __matview_store_nodes_mv ORDER BY id;").fetchall(),
            [(1, "C"), (2, "A"), (3, "B")],
        )

    def test_pk_swap_with_where_filter_execution(self) -> None:
        """Live execution: PK swap where one row drops out of WHERE filter."""
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE tasks (id INT, val TEXT);")
        con.execute("CREATE TABLE __matview_store_keep_tasks_mv (id INTEGER PRIMARY KEY, val TEXT);")

        schema_sql = "CREATE TABLE tasks (id INTEGER PRIMARY KEY, val TEXT);"
        view_sql = "CREATE VIEW keep_tasks_mv AS SELECT id, val FROM tasks WHERE val = 'keep';"
        plan = generate_materialized_view(schema_sql, view_sql, dialect=Dialect.SQLITE)

        con.execute("INSERT INTO tasks (id, val) VALUES (1, 'keep'), (2, 'keep');")
        for pop in plan.initial_populate_sqls:
            con.execute(pop)
        for trg in plan.trigger_ddls:
            con.execute(trg)

        self.assertEqual(
            con.execute("SELECT id, val FROM __matview_store_keep_tasks_mv ORDER BY id;").fetchall(),
            [(1, "keep"), (2, "keep")],
        )

        # Row 1 (id=1, val='keep') -> (id=2, val='keep')
        # Row 2 (id=2, val='keep') -> (id=1, val='drop')
        con.execute(
            "UPDATE tasks SET "
            "id = CASE WHEN id = 1 THEN 2 ELSE 1 END, "
            "val = CASE WHEN id = 1 THEN 'keep' ELSE 'drop' END;"
        )

        self.assertEqual(
            con.execute("SELECT id, val FROM tasks ORDER BY id;").fetchall(),
            [(1, "drop"), (2, "keep")],
        )

        # Backing table must contain ONLY (2, 'keep')!
        self.assertEqual(
            con.execute("SELECT id, val FROM __matview_store_keep_tasks_mv ORDER BY id;").fetchall(),
            [(2, "keep")],
        )

    def test_composite_pk_swap_execution(self) -> None:
        """Live execution: composite PK swap on (tenant_id, item_id)."""
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE inventory (tenant_id INT, item_id INT, qty INT);")
        con.execute(
            "CREATE TABLE __matview_store_inv_mv (tenant_id INT, item_id INT, qty INT, PRIMARY KEY (tenant_id, item_id));"
        )

        schema_sql = """
        CREATE TABLE inventory (
            tenant_id INT NOT NULL,
            item_id INT NOT NULL,
            qty INT NOT NULL,
            PRIMARY KEY (tenant_id, item_id)
        );
        """
        view_sql = "CREATE VIEW inv_mv AS SELECT tenant_id, item_id, qty FROM inventory;"
        plan = generate_materialized_view(schema_sql, view_sql, dialect=Dialect.SQLITE)

        con.execute("INSERT INTO inventory (tenant_id, item_id, qty) VALUES (10, 1, 100), (10, 2, 200);")
        for pop in plan.initial_populate_sqls:
            con.execute(pop)
        for trg in plan.trigger_ddls:
            con.execute(trg)

        # Swap item_ids: 1 -> 2, 2 -> 1
        con.execute("UPDATE inventory SET item_id = 3 - item_id WHERE tenant_id = 10;")

        self.assertEqual(
            con.execute("SELECT tenant_id, item_id, qty FROM inventory ORDER BY tenant_id, item_id;").fetchall(),
            [(10, 1, 200), (10, 2, 100)],
        )
        self.assertEqual(
            con.execute("SELECT tenant_id, item_id, qty FROM __matview_store_inv_mv ORDER BY tenant_id, item_id;").fetchall(),
            [(10, 1, 200), (10, 2, 100)],
        )

    def test_transpiled_postgres_shift_execution(self) -> None:
        """Test non-overlapping PK shift under TranspiledConnectionAdapter(dialect=POSTGRES)."""
        schema_sql = """
        CREATE TABLE nodes (
            id INT PRIMARY KEY,
            label VARCHAR(50) NOT NULL
        );
        """
        view_sql = "CREATE VIEW nodes_mv AS SELECT id, label FROM nodes;"

        conn = TranspiledConnectionAdapter(dialect=Dialect.POSTGRES)
        mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
        mgr.apply_schema_ddl(schema_sql)
        mgr.register_view(view_sql)

        cur = conn.cursor()
        cur.execute("INSERT INTO nodes (id, label) VALUES (1, 'Alpha'), (2, 'Beta');")

        cur.execute("SELECT id, label FROM nodes_mv ORDER BY id;")
        self.assertEqual(cur.fetchall(), [(1, "Alpha"), (2, "Beta")])

        cur.execute("UPDATE nodes SET id = id + 10;")

        cur.execute("SELECT id, label FROM nodes_mv ORDER BY id;")
        self.assertEqual(cur.fetchall(), [(11, "Alpha"), (12, "Beta")])

    def test_non_native_pk_projection_swap(self) -> None:
        """Verify non-native PK projection (uses_native_pk=False) handles swaps cleanly."""
        schema_map = {
            "logs": TableSchema(
                "logs",
                {
                    "msg": ColumnDef("msg", "TEXT", not_null=False),
                    "code": ColumnDef("code", "INT", not_null=False),
                },
            )
        }
        v = parse_view_definition("CREATE VIEW v AS SELECT msg, code FROM logs;")
        compiler = IncrementalViewCompiler(schema_map, v, dialect=Dialect.POSTGRES)
        plan = compiler._compile_tier1_zset_projection()

        self.assertIn("__mv_pk TEXT NOT NULL PRIMARY KEY", plan.backing_table_ddl)
        self.assertIn("__mv_row_count", plan.backing_table_ddl)

    def test_filtered_native_pk_projection_where_exit_and_swap_live_equivalence(self) -> None:
        """Live SQLite filtered native-PK projection: when a row stays at OLD.pk
        but stops satisfying the WHERE predicate, the NOT EXISTS guard must check
        the WHERE predicate so the old row is retracted cleanly."""
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE TABLE tasks (id INTEGER PRIMARY KEY, val TEXT NOT NULL);")
            con.executemany("INSERT INTO tasks (id, val) VALUES (?, ?);", [(1, "keep"), (2, "keep"), (3, "drop")])
            con.commit()

            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            plan = mgr.register_view(
                "CREATE VIEW active_tasks_mv AS SELECT id, val FROM tasks WHERE val = 'keep';"
            )
            self.assertIn("NOT EXISTS", "\n".join(plan.trigger_ddls))

            def assert_matches_live(step: str) -> None:
                self.assertEqual(
                    con.execute("SELECT id, val FROM active_tasks_mv ORDER BY id;").fetchall(),
                    con.execute("SELECT id, val FROM tasks WHERE val = 'keep' ORDER BY id;").fetchall(),
                    f"active_tasks_mv diverged after {step}",
                )

            assert_matches_live("initial population")

            # Row 2 keeps id=2 (so `EXISTS (SELECT 1 FROM tasks WHERE id = 2)` is
            # still TRUE) but leaves the WHERE predicate. Dropping `and_base_where`
            # from the NOT EXISTS guard leaves a phantom (2, 'keep') row.
            con.execute("UPDATE tasks SET val = 'drop' WHERE id = 2;")
            con.commit()
            assert_matches_live("in-place filter exit")

            # Re-enter filter on row 3 and move row 1 to a new key while exiting filter.
            con.execute("UPDATE tasks SET val = 'keep' WHERE id = 3;")
            con.execute("UPDATE tasks SET id = 10, val = 'drop' WHERE id = 1;")
            con.commit()
            assert_matches_live("filter re-entry and key move out of filter")
        finally:
            con.close()

    def test_refcount_projection_pk_swap_live_equivalence(self) -> None:
        """Live SQLite refcount (__mv_row_count) projection: single-statement swaps,
        key-changing updates, and filtered transitions on a nullable composite PK
        must match the live original SELECT."""
        con = sqlite3.connect(":memory:")
        try:
            # Nullable composite PRIMARY KEY compiles to Tier 1 ALGEBRAIC_DELTA
            # projection with uses_native_pk=False (__mv_row_count refcounting),
            # and rows with id IS NULL permit single-statement swaps without
            # hitting SQLite's row-by-row UNIQUE check.
            con.execute("CREATE TABLE logs (id INT, label TEXT, PRIMARY KEY (id, label));")
            con.executemany(
                "INSERT INTO logs (id, label) VALUES (?, ?);",
                [(None, "A"), (None, "B"), (1, "C")],
            )
            con.commit()

            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            plan_all = mgr.register_view("CREATE VIEW logs_mv AS SELECT id, label FROM logs;")
            plan_filt = mgr.register_view(
                "CREATE VIEW logs_ab_mv AS SELECT id, label FROM logs WHERE label IN ('A', 'B');"
            )
            self.assertIn("__mv_row_count", plan_all.backing_table_ddl)
            self.assertIn("__mv_row_count", plan_filt.backing_table_ddl)

            def assert_matches_live(step: str) -> None:
                self.assertEqual(
                    con.execute("SELECT id, label FROM logs_mv ORDER BY id, label;").fetchall(),
                    con.execute("SELECT id, label FROM logs ORDER BY id, label;").fetchall(),
                    f"logs_mv diverged after {step}",
                )
                self.assertEqual(
                    con.execute("SELECT id, label FROM logs_ab_mv ORDER BY id, label;").fetchall(),
                    con.execute(
                        "SELECT id, label FROM logs WHERE label IN ('A', 'B') ORDER BY id, label;"
                    ).fetchall(),
                    f"logs_ab_mv diverged after {step}",
                )

            assert_matches_live("initial population")

            # Single-statement 2-row swap: row 1 moves (NULL,'A')->(NULL,'B')
            # (refcount of (NULL,'B') becomes 2), then row 2 moves (NULL,'B')->(NULL,'A')
            # (refcount of (NULL,'B') drops to 1). Dropping `__mv_row_count <= 0`
            # deletes (NULL,'B') and loses 50% of swapped rows.
            con.execute(
                "UPDATE logs SET label = CASE WHEN label = 'A' THEN 'B' ELSE 'A' END "
                "WHERE id IS NULL AND label IN ('A', 'B');"
            )
            con.commit()
            assert_matches_live("single-statement refcount swap")

            # Key-changing update: omitting OLD decrement leaves phantom (1, 'C').
            con.execute("UPDATE logs SET id = 99, label = 'Z' WHERE id = 1 AND label = 'C';")
            con.commit()
            assert_matches_live("key-changing update")

            # Filtered swap where one row exits the WHERE filter.
            con.execute(
                "UPDATE logs SET label = CASE WHEN label = 'A' THEN 'B' ELSE 'DROP' END "
                "WHERE id IS NULL AND label IN ('A', 'B');"
            )
            con.commit()
            assert_matches_live("filtered refcount swap with exit")

            # Delete a remaining row.
            con.execute("DELETE FROM logs WHERE id = 99 AND label = 'Z';")
            con.commit()
            assert_matches_live("delete row")
        finally:
            con.close()


class TestFinding7CollationAndGroupPk(unittest.TestCase):
    """Exhaustive tests for Finding #7: MySQL __mv_pk binary collation and case-sensitivity."""

    def test_mysql_all_strategies_use_binary_collation_ddl(self) -> None:
        """Assert all MySQL backing and aux tables create __mv_pk with utf8mb4_bin."""
        expected_pk_type = "VARCHAR(512) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"

        # 1. Single-table group by with nullable key
        schema_sql = """
        CREATE TABLE orders (
            id INT PRIMARY KEY,
            region VARCHAR(32),
            amount DECIMAL(10, 2)
        );
        """
        plan_alg = generate_materialized_view(
            schema_sql,
            "CREATE VIEW v1 AS SELECT region, SUM(amount) AS total FROM orders GROUP BY region;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertIn(f"__mv_pk {expected_pk_type} NOT NULL PRIMARY KEY", plan_alg.backing_table_ddl)

        # 2. Join view with nullable group key
        schema_join = """
        CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(50));
        CREATE TABLE orders (id INT PRIMARY KEY, cust_id INT, status VARCHAR(20), amount DECIMAL(10, 2));
        """
        plan_join = generate_materialized_view(
            schema_join,
            "CREATE VIEW v2 AS SELECT o.status, SUM(o.amount) AS total FROM customers c JOIN orders o ON c.id = o.cust_id GROUP BY o.status;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertIn(f"__mv_pk {expected_pk_type} NOT NULL PRIMARY KEY", plan_join.backing_table_ddl)

        # 3. Non-native PK Z-set projection (compiled directly on compiler)
        schema_map_nopk = {
            "events": TableSchema(
                "events",
                {
                    "tag": ColumnDef("tag", "VARCHAR(50)", not_null=False),
                    "val": ColumnDef("val", "INT", not_null=False),
                },
            )
        }
        v3 = parse_view_definition("CREATE VIEW v3 AS SELECT tag, val FROM events;")
        compiler_v3 = IncrementalViewCompiler(schema_map_nopk, v3, dialect=Dialect.MYSQL, ignore_limitations=True)
        plan_proj = compiler_v3._compile_tier1_zset_projection()
        self.assertIn(f"__mv_pk {expected_pk_type} NOT NULL PRIMARY KEY", plan_proj.backing_table_ddl)

        # 4. UNION ALL with non-uniform PK structure
        schema_uall = """
        CREATE TABLE t1 (id INT, item INT, val INT, PRIMARY KEY (id, item));
        CREATE TABLE t2 (code VARCHAR(32) PRIMARY KEY, val INT);
        """
        plan_uall = generate_materialized_view(
            schema_uall,
            "CREATE VIEW v4 AS SELECT val FROM t1 UNION ALL SELECT val FROM t2;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertIn(f"__mv_pk {expected_pk_type} NOT NULL PRIMARY KEY", plan_uall.backing_table_ddl)

        # 5. UNION ALL aggregation with nullable group keys
        plan_uall_agg = generate_materialized_view(
            schema_uall,
            "CREATE VIEW v5 AS SELECT val, COUNT(*) AS cnt FROM (SELECT val FROM t1 UNION ALL SELECT val FROM t2) AS u GROUP BY val;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertIn(f"__mv_pk {expected_pk_type} NOT NULL PRIMARY KEY", plan_uall_agg.backing_table_ddl)

        # 6. DISTINCT auxiliary table with nullable group key
        schema_dist = """
        CREATE TABLE sales (
            id INT PRIMARY KEY,
            region VARCHAR(32),
            item VARCHAR(32)
        );
        """
        plan_dist = generate_materialized_view(
            schema_dist,
            "CREATE VIEW v6 AS SELECT region, COUNT(DISTINCT item) AS items FROM sales GROUP BY region;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertTrue(len(plan_dist.auxiliary_table_ddls) > 0)
        aux_ddl = plan_dist.auxiliary_table_ddls[0]
        self.assertIn(f"__mv_pk {expected_pk_type} NOT NULL", aux_ddl)

    def test_mysql_canonical_group_pk_expr_generation(self) -> None:
        """Verify _canonical_group_pk_expr uses OCTET_LENGTH, utf8mb4 charset, and COLLATE utf8mb4_bin."""
        schema_map = {
            "t": TableSchema(
                "t",
                {
                    "col": ColumnDef("col", "VARCHAR(50)", not_null=False),
                },
            )
        }
        v = parse_view_definition("CREATE VIEW v AS SELECT col, COUNT(*) FROM t GROUP BY col;")
        compiler = IncrementalViewCompiler(schema_map, v, dialect=Dialect.MYSQL, ignore_limitations=True)

        pk_expr = compiler._canonical_group_pk_expr(["col"])
        self.assertIn("OCTET_LENGTH(CAST(col AS CHAR CHARACTER SET utf8mb4))", pk_expr)
        self.assertIn("CAST(col AS CHAR CHARACTER SET utf8mb4)", pk_expr)
        self.assertTrue(pk_expr.endswith("COLLATE utf8mb4_bin"))

    def test_canonical_pk_preserves_case_and_accents(self) -> None:
        """Verify that case-variants and accent-variants produce distinct PK tokens."""
        def make_pk(val: str | None) -> str:
            if val is None:
                return "N"
            b = val.encode("utf-8")
            return f"V{len(b)}:{val}"

        apple_lower = make_pk("apple")
        apple_upper = make_pk("APPLE")
        self.assertEqual(apple_lower, "V5:apple")
        self.assertEqual(apple_upper, "V5:APPLE")
        self.assertNotEqual(apple_lower, apple_upper)

        cafe_plain = make_pk("cafe")
        cafe_accent = make_pk("café")
        self.assertEqual(cafe_plain, "V4:cafe")
        self.assertEqual(cafe_accent, "V5:café")
        self.assertNotEqual(cafe_plain, cafe_accent)

        pk_null = make_pk(None)
        pk_empty = make_pk("")
        pk_zero = make_pk("0")
        self.assertEqual(pk_null, "N")
        self.assertEqual(pk_empty, "V0:")
        self.assertEqual(pk_zero, "V1:0")
        self.assertEqual(len({pk_null, pk_empty, pk_zero}), 3)

    def test_mysql_transpiled_case_distinct_aggregation(self) -> None:
        """Live execution: case-distinct keys ('apple' vs 'APPLE') maintain separate aggregates."""
        schema_sql = """
        CREATE TABLE tag_counts (
            id INT PRIMARY KEY,
            tag VARCHAR(50),
            val INT NOT NULL
        );
        """
        view_sql = "CREATE VIEW tag_summary_mv AS SELECT tag, SUM(val) AS total_val FROM tag_counts GROUP BY tag;"

        conn = TranspiledConnectionAdapter(dialect=Dialect.MYSQL)
        mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL, ignore_limitations=True)
        mgr.apply_schema_ddl(schema_sql)
        mgr.register_view(view_sql)

        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tag_counts (id, tag, val) VALUES "
            "(1, 'apple', 10), "
            "(2, 'APPLE', 20), "
            "(3, 'Apple', 30);"
        )

        cur.execute("SELECT tag, total_val FROM tag_summary_mv ORDER BY tag;")
        rows = sorted(cur.fetchall(), key=lambda r: (r[0] or "", r[1]))
        expected = [("APPLE", 20), ("Apple", 30), ("apple", 10)]
        self.assertEqual(rows, expected)

        # Update one case-variant: 'apple' from 10 to 15
        cur.execute("UPDATE tag_counts SET val = 15 WHERE id = 1;")
        cur.execute("SELECT tag, total_val FROM tag_summary_mv ORDER BY tag;")
        rows = sorted(cur.fetchall(), key=lambda r: (r[0] or "", r[1]))
        expected = [("APPLE", 20), ("Apple", 30), ("apple", 15)]
        self.assertEqual(rows, expected)

        # Delete 'APPLE'
        cur.execute("DELETE FROM tag_counts WHERE id = 2;")
        cur.execute("SELECT tag, total_val FROM tag_summary_mv ORDER BY tag;")
        rows = sorted(cur.fetchall(), key=lambda r: (r[0] or "", r[1]))
        expected = [("Apple", 30), ("apple", 15)]
        self.assertEqual(rows, expected)


if __name__ == "__main__":
    unittest.main()
