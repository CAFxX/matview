#!/usr/bin/env python3
"""P0 Regression Tests for Finding #6 and Finding #7.

Finding #6: Row-Level Trigger PK Swap / Shift Data Loss Prevention
- Verifies that row-level triggers on PostgreSQL, SQLite, and MySQL do not prematurely
  delete backing table rows during multi-row PK swaps, cycles, or shifts.
- Verifies that the delete predicate in AFTER UPDATE triggers is gated by:
  `AND NOT EXISTS (SELECT 1 FROM <base_t> WHERE <base_pk = OLD.pk> AND (<where_clause>))`
- Verifies edge cases with and without WHERE filtering, composite primary keys,
  and non-native PK projection views (`uses_native_pk = False`). PostgreSQL
  maintenance is statement-level (one signed delta per statement); its native
  PK-permutation tests are in test_pg_statement_maintenance.py.

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

import os
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

    def test_postgres_projection_maintenance_is_statement_level(self) -> None:
        """PostgreSQL has no row-level maintenance: PK swaps, cycles and shifts
        are one signed delta per statement (native behaviour is tested in
        test_pg_statement_maintenance.py::test_primary_key_permutations)."""
        schema_sql = """
        CREATE TABLE items (
            id INT PRIMARY KEY,
            status VARCHAR(20) NOT NULL
        );
        """
        view_sql = "CREATE VIEW active_items_mv AS SELECT id, status FROM items WHERE status = 'active';"
        plan = generate_materialized_view(schema_sql, view_sql, dialect=Dialect.POSTGRES)
        trig_ddls = "\n\n".join(plan.trigger_ddls)
        self.assertNotIn("FOR EACH ROW", trig_ddls)
        self.assertNotIn("LOOP", trig_ddls.replace("END LOOP", "").replace("    LOOP\n", ""))
        au_ddl = [ddl for ddl in plan.trigger_ddls if "_au()" in ddl and "FUNCTION" in ddl][0]
        # A retracted key is deleted only if the same statement does not
        # re-insert it (a PK swap keeps both rows).
        self.assertRegex(au_ddl, r"(?s)DELETE FROM __matview_store_active_items_mv\s+WHERE .* IN \(SELECT .*__mv_old_rows.*\)\s+AND NOT .* IN \(SELECT .*__mv_new_rows")

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

    def test_refcount_projection_duplicate_null_pk_multiplicity_live_equivalence(self) -> None:
        """Live SQLite refcount projection (`uses_native_pk=False`): duplicate NULL-PK
        rows must preserve exact multiplicity in the reader view across initial
        population, inserts, unprojected-column updates, filtered views, deletes,
        large multiplicity (>1000), and read-only enforcement."""
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE TABLE items (id INT PRIMARY KEY, val TEXT, extra TEXT);")
            con.executemany(
                "INSERT INTO items (id, val, extra) VALUES (?, ?, ?);",
                [
                    (None, "a", "e1"),
                    (None, "a", "e1"),
                    (None, "a", "e2"),
                    (None, None, "e3"),
                    (1, "b", "e4"),
                ],
            )
            con.commit()

            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            plan_all = mgr.register_view("CREATE VIEW items_mv AS SELECT id, val FROM items;")
            plan_filt = mgr.register_view(
                "CREATE VIEW items_nonnull_mv AS SELECT id, val FROM items WHERE val IS NOT NULL;"
            )
            self.assertIn("__mv_row_count", plan_all.backing_table_ddl)
            self.assertIn("WITH RECURSIVE", plan_all.reader_view_ddl or "")

            def assert_matches_live(step: str) -> None:
                self.assertEqual(
                    sorted(con.execute("SELECT id, val FROM items_mv;").fetchall(), key=repr),
                    sorted(con.execute("SELECT id, val FROM items;").fetchall(), key=repr),
                    f"items_mv diverged after {step}",
                )
                self.assertEqual(
                    sorted(con.execute("SELECT id, val FROM items_nonnull_mv;").fetchall(), key=repr),
                    sorted(
                        con.execute("SELECT id, val FROM items WHERE val IS NOT NULL;").fetchall(),
                        key=repr,
                    ),
                    f"items_nonnull_mv diverged after {step}",
                )

            assert_matches_live("initial duplicate NULL-PK population")

            # Add a 3rd identical (NULL, 'a', 'e1') row -> backing __mv_row_count = 3.
            con.execute("INSERT INTO items (id, val, extra) VALUES (NULL, 'a', 'e1');")
            con.commit()
            assert_matches_live("insert 3rd duplicate NULL-PK row")

            # Update unprojected column `extra` (which is part of pk_hash_cols):
            # trigger must fire so later DELETE does not leave a phantom row.
            con.execute("UPDATE items SET extra = 'e9' WHERE id IS NULL AND val = 'a' AND extra = 'e2';")
            con.commit()
            assert_matches_live("update unprojected hashed column")

            con.execute("DELETE FROM items WHERE id IS NULL AND val = 'a' AND extra = 'e9';")
            con.commit()
            assert_matches_live("delete after unprojected column update")

            # Delete one of the 3 identical (NULL, 'a', 'e1') rows via rowid.
            con.execute(
                "DELETE FROM items WHERE rowid = ("
                "SELECT MIN(rowid) FROM items WHERE id IS NULL AND val = 'a' AND extra = 'e1');"
            )
            con.commit()
            assert_matches_live("delete 1 of 3 identical duplicates")

            # High multiplicity (>1000) exercises O(log N) binary-doubling CTE.
            con.executemany(
                "INSERT INTO items (id, val, extra) VALUES (?, ?, ?);",
                [(None, "bulk", "x")] * 1200,
            )
            con.commit()
            self.assertEqual(
                con.execute("SELECT COUNT(*) FROM items_mv WHERE val = 'bulk';").fetchone()[0],
                1200,
            )
            assert_matches_live("insert 1200 identical duplicates")

            # Clear all rows -> empty reader view.
            con.execute("DELETE FROM items;")
            con.commit()
            assert_matches_live("delete all rows")

            # Reader view remains read-only (INSERT, UPDATE, DELETE all rejected).
            con.execute("INSERT INTO items (id, val, extra) VALUES (NULL, 'al', 'e1'), (NULL, 'al', 'e1');")
            con.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute("INSERT INTO items_mv (id, val) VALUES (99, 'ro');")
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute("UPDATE items_mv SET val = 'bad';")
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute("DELETE FROM items_mv;")
            con.rollback()

            # Aliased projection on nullable-PK table works cleanly.
            mgr.register_view("CREATE VIEW items_alias_mv AS SELECT t.id, t.val FROM items AS t;")
            self.assertEqual(
                sorted(con.execute("SELECT id, val FROM items_alias_mv;").fetchall(), key=repr),
                sorted(con.execute("SELECT t.id, t.val FROM items AS t;").fetchall(), key=repr),
            )
        finally:
            con.close()

    def test_refcount_projection_and_union_all_incremental_backfill_paths(self) -> None:
        """Incremental backfill (`pk_range_fold` and `offset_batch`) must preserve
        all duplicate NULL-PK rows on nullable INT/TEXT/composite PKs and UNION ALL."""
        con = sqlite3.connect(":memory:")
        try:
            # 1. Nullable INT PRIMARY KEY with duplicate NULL-PK rows + incremental_backfill=True
            con.execute("CREATE TABLE ib (id INT PRIMARY KEY, val TEXT);")
            con.executemany(
                "INSERT INTO ib (id, val) VALUES (?, ?);",
                [(None, "n1"), (None, "n1"), (1, "k1"), (2, "k2")],
            )
            # 2. Nullable TEXT PRIMARY KEY with NULL-PK rows + incremental_backfill=True
            con.execute("CREATE TABLE ibt (id TEXT PRIMARY KEY, val TEXT);")
            con.executemany(
                "INSERT INTO ibt (id, val) VALUES (?, ?);",
                [(None, "t1"), (None, "t1"), ("a", "k1")],
            )
            # 3. Composite nullable PK (offset_batch) + incremental_backfill=True
            con.execute("CREATE TABLE ibc (a INT, b TEXT, PRIMARY KEY (a, b));")
            con.executemany(
                "INSERT INTO ibc (a, b) VALUES (?, ?);",
                [(None, "c1"), (None, "c1"), (1, "c2")],
            )
            con.commit()

            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            p_ib = mgr.register_view(
                "CREATE VIEW ib_mv AS SELECT id, val FROM ib;",
                incremental_backfill=True,
                batch_size=2,
            )
            self.assertEqual(p_ib.incremental_backfill_spec.pk_column, "rowid")
            self.assertEqual(
                sorted(con.execute("SELECT id, val FROM ib_mv;").fetchall(), key=repr),
                sorted(con.execute("SELECT id, val FROM ib;").fetchall(), key=repr),
            )

            p_ibt = mgr.register_view(
                "CREATE VIEW ibt_mv AS SELECT id, val FROM ibt;",
                incremental_backfill=True,
                batch_size=2,
            )
            self.assertEqual(p_ibt.incremental_backfill_spec.pk_column, "rowid")
            self.assertEqual(
                sorted(con.execute("SELECT id, val FROM ibt_mv;").fetchall(), key=repr),
                sorted(con.execute("SELECT id, val FROM ibt;").fetchall(), key=repr),
            )

            # Verify get_populate_statements(incremental_backfill=True) SQL executes
            # cleanly for both pk_range_fold (SQLite recursive batch CTE) and offset_batch
            # (nullable-PK UNION ALL) with zero embedded ON CONFLICT syntax errors.
            p_ibc = mgr.register_view(
                "CREATE VIEW ibc_mv AS SELECT a, b FROM ibc;",
                incremental_backfill=True,
                batch_size=2,
            )
            self.assertEqual(
                sorted(con.execute("SELECT a, b FROM ibc_mv;").fetchall(), key=repr),
                sorted(con.execute("SELECT a, b FROM ibc;").fetchall(), key=repr),
            )

            p_uall = generate_materialized_view(
                "CREATE TABLE ib (id INT PRIMARY KEY, val TEXT);",
                "CREATE VIEW ua_inc_mv AS SELECT id, val FROM ib UNION ALL SELECT id, val FROM ib;",
                dialect=Dialect.SQLITE,
                incremental_backfill=True,
                batch_size=2,
            )
            con.execute(p_uall.backing_table_ddl)
            assert p_uall.reader_view_ddl is not None
            con.execute(p_uall.reader_view_ddl)
            for stmt in p_uall.get_populate_statements(incremental_backfill=True):
                con.execute(stmt)
            self.assertEqual(
                sorted(con.execute("SELECT id, val FROM ua_inc_mv;").fetchall(), key=repr),
                sorted(
                    con.execute("SELECT id, val FROM ib UNION ALL SELECT id, val FROM ib;").fetchall(),
                    key=repr,
                ),
            )
        finally:
            con.close()

    def test_refcount_projection_sqlite_storage_class_preserved(self) -> None:
        """On SQLite, `(NULL, 1)` and `(NULL, 1.0)` in a BLOB/untyped column have
        distinct storage classes and must not collapse in a bag projection."""
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE TABLE vals (id INT PRIMARY KEY, v BLOB);")
            con.executemany(
                "INSERT INTO vals (id, v) VALUES (?, ?);",
                [(None, 1), (None, 1), (None, 1.0), (None, "1"), (None, b"1")],
            )
            con.commit()

            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            mgr.register_view("CREATE VIEW vals_mv AS SELECT id, v FROM vals;")

            def typed_rows(sql: str) -> List[Tuple[str, str, object]]:
                return sorted(
                    [(type(r[0]).__name__, type(r[1]).__name__, r[1]) for r in con.execute(sql).fetchall()],
                    key=repr,
                )

            self.assertEqual(typed_rows("SELECT id, v FROM vals_mv;"), typed_rows("SELECT id, v FROM vals;"))

            con.execute("DELETE FROM vals WHERE rowid = (SELECT MIN(rowid) FROM vals WHERE typeof(v) = 'integer');")
            con.commit()
            self.assertEqual(typed_rows("SELECT id, v FROM vals_mv;"), typed_rows("SELECT id, v FROM vals;"))

            con.execute("INSERT INTO vals (id, v) VALUES (NULL, 1.0);")
            con.commit()
            self.assertEqual(typed_rows("SELECT id, v FROM vals_mv;"), typed_rows("SELECT id, v FROM vals;"))
        finally:
            con.close()

    def test_nullable_pk_union_all_duplicate_null_pk_multiplicity_live_equivalence(self) -> None:
        """UNION ALL over SQLite nullable-PK tables (and mixed PK types) must not
        fail NOT NULL / UNIQUE constraints and must preserve exact row multiplicity."""
        con = sqlite3.connect(":memory:")
        try:
            con.execute("CREATE TABLE a (id INT PRIMARY KEY, val TEXT, extra TEXT);")
            con.execute("CREATE TABLE b (id INT PRIMARY KEY, val TEXT, extra TEXT);")
            con.executemany(
                "INSERT INTO a (id, val, extra) VALUES (?, ?, ?);",
                [(None, "x", "e1"), (None, "x", "e1"), (None, "y", "e2"), (1, "z", "e3")],
            )
            con.executemany(
                "INSERT INTO b (id, val, extra) VALUES (?, ?, ?);",
                [(None, "x", "e1"), (2, "w", "e4")],
            )
            con.commit()

            mgr = MaterializedViewManager(con, dialect=Dialect.SQLITE)
            plan = mgr.register_view(
                "CREATE VIEW ab_uall_mv AS SELECT id, val FROM a UNION ALL SELECT id, val FROM b;"
            )
            self.assertIn("WITH RECURSIVE", plan.reader_view_ddl or "")

            def assert_matches_live(step: str) -> None:
                self.assertEqual(
                    sorted(con.execute("SELECT id, val FROM ab_uall_mv;").fetchall(), key=repr),
                    sorted(
                        con.execute("SELECT id, val FROM a UNION ALL SELECT id, val FROM b;").fetchall(),
                        key=repr,
                    ),
                    f"ab_uall_mv diverged after {step}",
                )

            assert_matches_live("initial population")
            con.execute("INSERT INTO a VALUES (NULL, 'x', 'e1');")
            con.execute("UPDATE a SET extra = 'e9' WHERE id IS NULL AND val = 'y';")
            con.commit()
            assert_matches_live("insert duplicate and update unprojected column")

            con.execute("DELETE FROM a WHERE id IS NULL AND val = 'y';")
            con.execute("DELETE FROM b WHERE id IS NULL;")
            con.commit()
            assert_matches_live("deletes across both branches")

            # Mixed PK types across UNION ALL branches (INTEGER vs TEXT) also work safely.
            con.execute("CREATE TABLE m_int (id INTEGER PRIMARY KEY, val TEXT);")
            con.execute("CREATE TABLE m_txt (id TEXT NOT NULL PRIMARY KEY, val TEXT);")
            con.execute("INSERT INTO m_int VALUES (1, 'i1');")
            con.execute("INSERT INTO m_txt VALUES ('k1', 't1');")
            con.commit()
            mgr.register_view(
                "CREATE VIEW mixed_uall_mv AS SELECT val FROM m_int UNION ALL SELECT val FROM m_txt;"
            )
            self.assertEqual(
                sorted(con.execute("SELECT val FROM mixed_uall_mv;").fetchall()),
                sorted(con.execute("SELECT val FROM m_int UNION ALL SELECT val FROM m_txt;").fetchall()),
            )

            # Aliased UNION ALL branches and case-differing table names share triggers cleanly.
            con.execute("CREATE TABLE CaseTbl (id INT PRIMARY KEY, val TEXT);")
            con.execute("INSERT INTO CaseTbl VALUES (NULL, 'c1'), (NULL, 'c1'), (1, 'c2');")
            con.commit()
            mgr.register_view(
                "CREATE VIEW case_alias_uall_mv AS "
                "SELECT t1.id, t1.val FROM CaseTbl AS t1 "
                "UNION ALL SELECT t2.id, t2.val FROM casetbl AS t2;"
            )
            self.assertEqual(
                sorted(con.execute("SELECT id, val FROM case_alias_uall_mv;").fetchall(), key=repr),
                sorted(
                    con.execute(
                        "SELECT t1.id, t1.val FROM CaseTbl AS t1 "
                        "UNION ALL SELECT t2.id, t2.val FROM casetbl AS t2;"
                    ).fetchall(),
                    key=repr,
                ),
            )
        finally:
            con.close()

    def test_bag_projection_reader_view_codegen_postgres_and_mysql(self) -> None:
        """Verify PostgreSQL and MySQL bag-projection codegen expands __mv_row_count."""
        schema_map = {
            "logs": TableSchema(
                "logs",
                {
                    "msg": ColumnDef("msg", "TEXT", not_null=False),
                    "code": ColumnDef("code", "INT", not_null=False),
                },
            )
        }
        v = parse_view_definition("CREATE VIEW v_bag AS SELECT msg, code FROM logs;")

        pg_plan = IncrementalViewCompiler(schema_map, v, dialect=Dialect.POSTGRES)._compile_tier1_zset_projection()
        self.assertIn("CROSS JOIN LATERAL generate_series(1,", pg_plan.reader_view_ddl or "")
        self.assertIn("GROUP BY", pg_plan.initial_populate_sqls[0])
        self.assertIn("COUNT(*)", pg_plan.initial_populate_sqls[0])

        my_plan = IncrementalViewCompiler(
            schema_map, v, dialect=Dialect.MYSQL, ignore_limitations=True
        )._compile_tier1_zset_projection()
        self.assertIn("CREATE ALGORITHM = TEMPTABLE VIEW", my_plan.reader_view_ddl or "")
        self.assertIn("WITH RECURSIVE __mv_bits(__mv_b)", my_plan.reader_view_ddl or "")
        self.assertIn("AND EXISTS (SELECT 1)", my_plan.reader_view_ddl or "")

    def test_native_postgres_bag_projection_multiplicity_roundtrip(self) -> None:
        """Native PostgreSQL end-to-end verification of LATERAL generate_series
        bag-projection reader view, grouped initial population, statement-level
        counted triggers, and read-only enforcement on a keyless table."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        try:
            import psycopg2 as pg_driver  # type: ignore[import-not-found]
            conn = pg_driver.connect(dsn)
        except ImportError:
            import psycopg as pg_driver  # type: ignore[import-not-found]
            conn = pg_driver.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()
        try:
            cur.execute("DROP VIEW IF EXISTS pg_bag_mv CASCADE;")
            cur.execute("DROP TABLE IF EXISTS __matview_store_pg_bag_mv CASCADE;")
            cur.execute("DROP TABLE IF EXISTS pg_bag_logs CASCADE;")
            cur.execute("CREATE TABLE pg_bag_logs (code INT, msg TEXT);")
            cur.execute(
                "INSERT INTO pg_bag_logs VALUES (1, 'a'), (1, 'a'), (NULL, 'b'), (NULL, 'b'), (2, 'c');"
            )

            schema_map = {
                "pg_bag_logs": TableSchema(
                    "pg_bag_logs",
                    {
                        "code": ColumnDef("code", "INT", not_null=False),
                        "msg": ColumnDef("msg", "TEXT", not_null=False),
                    },
                )
            }
            v = parse_view_definition("CREATE VIEW pg_bag_mv AS SELECT code, msg FROM pg_bag_logs;")
            compiler = IncrementalViewCompiler(schema_map, v, dialect=Dialect.POSTGRES)
            raw_plan = compiler._compile_tier1_zset_projection()
            plan = compiler._finalize_plan_backfill(raw_plan)

            cur.execute(plan.backing_table_ddl)
            for idx_ddl in plan.index_ddls:
                cur.execute(idx_ddl)
            assert plan.reader_view_ddl is not None
            cur.execute(plan.reader_view_ddl)
            for ro_ddl in plan.readonly_trigger_ddls:
                cur.execute(ro_ddl)
            for trig_ddl in plan.trigger_ddls:
                cur.execute(trig_ddl)
            for pop_sql in plan.initial_populate_sqls:
                cur.execute(pop_sql)

            def assert_pg_matches(step: str) -> None:
                cur.execute("SELECT code, msg FROM pg_bag_mv;")
                mv_rows = sorted(cur.fetchall(), key=repr)
                cur.execute("SELECT code, msg FROM pg_bag_logs;")
                live_rows = sorted(cur.fetchall(), key=repr)
                self.assertEqual(mv_rows, live_rows, f"pg_bag_mv diverged after {step}")

            assert_pg_matches("initial duplicate population")
            cur.execute("INSERT INTO pg_bag_logs VALUES (1, 'a'), (NULL, 'b');")
            assert_pg_matches("multi-row duplicate insert")
            cur.execute("UPDATE pg_bag_logs SET msg = 'a2' WHERE code = 1;")
            assert_pg_matches("update all duplicates")
            cur.execute(
                "DELETE FROM pg_bag_logs WHERE ctid = ("
                "SELECT ctid FROM pg_bag_logs WHERE code IS NULL AND msg = 'b' LIMIT 1);"
            )
            assert_pg_matches("delete 1 of 3 NULL duplicates")
            cur.execute("DELETE FROM pg_bag_logs;")
            assert_pg_matches("delete all rows")

            with self.assertRaises(Exception):
                cur.execute("INSERT INTO pg_bag_mv VALUES (9, 'ro');")

            # Also verify literal constants in SELECT + NUMERIC 1.0 vs 1.00 + FLOAT8 0.0 vs -0.0
            # + schema-qualified reader view & INSTEAD OF trigger.
            cur.execute("DROP SCHEMA IF EXISTS mv_bag_sch CASCADE;")
            cur.execute("CREATE SCHEMA mv_bag_sch;")
            cur.execute("DROP VIEW IF EXISTS pg_bag_const_mv CASCADE;")
            cur.execute("DROP TABLE IF EXISTS __matview_store_pg_bag_const_mv CASCADE;")
            cur.execute("DROP TABLE IF EXISTS pg_bag_num CASCADE;")
            cur.execute("CREATE TABLE pg_bag_num (n NUMERIC, f FLOAT8, msg TEXT);")
            cur.execute(
                "INSERT INTO pg_bag_num VALUES "
                "(1.0, 0.0::float8, 'x'), (1.00, -0.0::float8, 'x'), (1.0, 0.0::float8, 'x');"
            )
            num_schemas = {
                "pg_bag_num": TableSchema(
                    "pg_bag_num",
                    {
                        "n": ColumnDef("n", "NUMERIC", not_null=False),
                        "f": ColumnDef("f", "DOUBLE PRECISION", not_null=False),
                        "msg": ColumnDef("msg", "TEXT", not_null=False),
                    },
                )
            }
            v_const = parse_view_definition(
                "CREATE VIEW pg_bag_const_mv AS "
                "SELECT 5 AS k, 'hello' AS tag, NULL::TEXT AS nlit, TRUE AS flg, 1.5 AS decv, "
                "n::text AS ns, f::text AS fs, msg FROM pg_bag_num;"
            )
            comp_const = IncrementalViewCompiler(num_schemas, v_const, dialect=Dialect.POSTGRES)
            plan_const = comp_const._finalize_plan_backfill(comp_const._compile_tier1_zset_projection())
            cur.execute(plan_const.backing_table_ddl)
            assert plan_const.reader_view_ddl is not None
            cur.execute(plan_const.reader_view_ddl)
            for ro_ddl in plan_const.readonly_trigger_ddls:
                cur.execute(ro_ddl)
            for trig_ddl in plan_const.trigger_ddls:
                cur.execute(trig_ddl)
            for pop_sql in plan_const.initial_populate_sqls:
                cur.execute(pop_sql)
            cur.execute("SELECT k, tag, nlit, flg, decv, ns, fs, msg FROM pg_bag_const_mv;")
            mv_const = sorted(cur.fetchall(), key=repr)
            cur.execute(
                "SELECT 5 AS k, 'hello' AS tag, NULL::TEXT AS nlit, TRUE AS flg, 1.5 AS decv, "
                "n::text AS ns, f::text AS fs, msg FROM pg_bag_num;"
            )
            live_const = sorted(cur.fetchall(), key=repr)
            self.assertEqual(mv_const, live_const)
            with self.assertRaises(Exception):
                cur.execute(
                    "INSERT INTO pg_bag_const_mv VALUES (5, 'hello', NULL, TRUE, 1.5, '1', '0', 'x');"
                )

            # Schema-qualified reader view + INSTEAD OF trigger execute cleanly on PostgreSQL.
            from dataclasses import replace as dc_replace
            v_sq = dc_replace(v_const, view_name="mv_bag_sch.pg_bag_sq_v")
            comp_sq = IncrementalViewCompiler(num_schemas, v_sq, dialect=Dialect.POSTGRES)
            sq_view_ddl, sq_ro_trigs, _, _ = comp_sq._build_readonly_view_and_triggers(
                "mv_bag_sch.pg_bag_sq_v",
                plan_const.backing_table_name,
                plan_const.user_columns,
                repeat_count_col="__mv_row_count",
            )
            cur.execute(sq_view_ddl)
            for ro_ddl in sq_ro_trigs:
                cur.execute(ro_ddl)
            cur.execute("SELECT COUNT(*) FROM mv_bag_sch.pg_bag_sq_v;")
            self.assertEqual(cur.fetchone()[0], 3)
            with self.assertRaises(Exception):
                cur.execute(
                    "INSERT INTO mv_bag_sch.pg_bag_sq_v VALUES (5, 'hello', NULL, TRUE, 1.5, '1', '0', 'x');"
                )
        finally:
            cur.execute("DROP SCHEMA IF EXISTS mv_bag_sch CASCADE;")
            cur.execute("DROP VIEW IF EXISTS pg_bag_const_mv CASCADE;")
            cur.execute("DROP VIEW IF EXISTS pg_bag_mv CASCADE;")
            cur.execute("DROP TABLE IF EXISTS __matview_store_pg_bag_mv CASCADE;")
            cur.execute("DROP TABLE IF EXISTS __matview_store_pg_bag_const_mv CASCADE;")
            cur.execute("DROP TABLE IF EXISTS pg_bag_num CASCADE;")
            cur.execute("DROP TABLE IF EXISTS pg_bag_logs CASCADE;")
            cur.close()
            conn.close()


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
