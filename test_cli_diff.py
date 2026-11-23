#!/usr/bin/env python3
"""
test_cli_diff.py — Test Suite for matview diff CLI Command & Compilation Plan Diagnostics
========================================================================================

Verifies:
  1. Structured compilation plan diagnostics (`plan.diagnostics` / `PlanDiagnostics`):
     - Strategy selection and tier assignment (Tier 1 algebraic, Tier 2 group-scoped, Tier 3 atomic).
     - Incremental vs fallback reasons (e.g. holistic aggregate, window function, CTE, join topology).
     - Engine transparency limitations, backing/auxiliary object tracking, and JSON serialization.
  2. MaterializedViewManager diff inspection API (`diff_view`, `diff_views`):
     - Fresh view not in catalog -> CREATE diff.
     - Unchanged registered view -> clean NOOP diff.
     - Modified query / drifted schema -> REPLACE/UPDATE diff.
     - Pruned / stale catalog entry -> DROP diff.
     - Out-of-band schema drift and missing database object detection.
     - Zero side-effects: diff inspection never mutates schema state.
  3. `matview diff` CLI command (`cli_main`):
     - Subcommand (`diff`) and flag (`--diff`) invocation styles.
     - Clean text output formatting and structured `--json` serialization.
     - Exit code contracts: default 0, `--exit-code` (0 for clean NOOP, 1 for changes).
     - Rejection on transparency limitations without `--ignore-limitations` (exit code 2).
     - Limitation override with `--ignore-limitations` (exit code 0 / diff emitted).
     - Argument validation (missing `--db`/`--schema`, missing `--view`).
"""

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict

import matview_sync as mv
from matview_sync import (
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    MaintenanceStrategy,
    MaterializedViewManager,
    PlanDiagnostics,
    ViewDiff,
    ViewDiffReport,
    generate_materialized_view,
    main as cli_main,
)


class TestPlanDiagnostics(unittest.TestCase):
    """Tests compilation plan diagnostics on single-view and multi-view definitions."""

    def setUp(self) -> None:
        self.schemas = mv.parse_table_schema(
            """
            CREATE TABLE orders (
                id INT PRIMARY KEY,
                customer_id INT,
                status TEXT,
                amount INT,
                created_at TEXT
            );
            CREATE TABLE customers (
                id INT PRIMARY KEY,
                region TEXT,
                name TEXT
            );
            """
        )

    def test_tier1_single_table_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_sales AS "
            "SELECT customer_id, COUNT(*) AS cnt, SUM(amount) AS total_amt "
            "FROM orders GROUP BY customer_id;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertIsNotNone(diag)
        self.assertIsInstance(diag, PlanDiagnostics)
        self.assertEqual(diag.strategy, "algebraic_delta")
        self.assertEqual(diag.tier, 1)
        self.assertTrue(diag.is_incremental)
        self.assertEqual(diag.algebraic_mode, "single")
        self.assertIsNone(diag.fallback_reason)
        self.assertIn("single-table", diag.selection_reason)
        self.assertEqual(diag.backing_table, plan.backing_table_name)
        self.assertEqual(diag.triggers, list(plan.trigger_names))
        self.assertEqual(diag.indexes, list(plan.index_names))

    def test_tier1_join_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_cust_orders AS "
            "SELECT c.region, COUNT(*) AS cnt, SUM(o.amount) AS tot "
            "FROM customers c INNER JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.region;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "algebraic_delta")
        self.assertEqual(diag.tier, 1)
        self.assertTrue(diag.is_incremental)
        self.assertEqual(diag.algebraic_mode, "join")
        self.assertIsNone(diag.fallback_reason)
        self.assertIn("INNER/CROSS JOIN", diag.selection_reason)

    def test_tier1_distinct_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_distinct_cust AS "
            "SELECT DISTINCT customer_id FROM orders;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "algebraic_delta")
        self.assertEqual(diag.tier, 1)
        self.assertEqual(diag.algebraic_mode, "distinct")
        self.assertIn("DISTINCT", diag.selection_reason)

    def test_tier2_holistic_aggregate_fallback_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_holistic AS "
            "SELECT customer_id, GROUP_CONCAT(status, ',' ORDER BY id) AS statuses "
            "FROM orders GROUP BY customer_id;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "group_scoped_delta")
        self.assertEqual(diag.tier, 2)
        self.assertTrue(diag.is_incremental)
        self.assertIsNone(diag.algebraic_mode)
        self.assertIsNotNone(diag.fallback_reason)
        self.assertIn("GROUP_CONCAT", diag.fallback_reason)
        self.assertTrue(diag.fallback_reason.startswith("exact result required: falling back to strategy group-scoped recompute. Why: "))

    def test_tier2_left_join_fallback_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_left_join AS "
            "SELECT c.id, COUNT(o.id) AS cnt "
            "FROM customers c LEFT JOIN orders o ON c.id = o.customer_id "
            "GROUP BY c.id;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "group_scoped_delta")
        self.assertEqual(diag.tier, 2)
        self.assertTrue(diag.is_incremental)
        self.assertIsNotNone(diag.fallback_reason)
        self.assertIn("Non-inner join", diag.fallback_reason)

    def test_tier2_window_partition_scoped_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_window AS "
            "SELECT id, customer_id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY id) AS rn "
            "FROM orders;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "group_scoped_delta")
        self.assertEqual(diag.tier, 2)
        self.assertEqual(diag.algebraic_mode, "window_partition")
        self.assertTrue(diag.is_incremental)
        self.assertIn("partition-scoped", diag.selection_reason)
        self.assertIn("PARTITION BY", diag.fallback_reason)

    def test_tier3_window_function_fallback_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_window AS "
            "SELECT id, customer_id, ROW_NUMBER() OVER (ORDER BY id) AS rn "
            "FROM orders;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "atomic_query_sync")
        self.assertEqual(diag.tier, 3)
        self.assertFalse(diag.is_incremental)
        self.assertIsNotNone(diag.fallback_reason)
        self.assertIn("Window function", diag.fallback_reason)

    def test_tier3_cte_fallback_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_cte AS "
            "WITH complex_cte AS ("
            "    SELECT o.customer_id, c.region "
            "    FROM orders o JOIN customers c ON o.customer_id = c.id "
            "    WHERE o.amount > (SELECT AVG(amount) FROM orders)"
            ") "
            "SELECT customer_id, region FROM complex_cte;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "atomic_query_sync")
        self.assertEqual(diag.tier, 3)
        self.assertFalse(diag.is_incremental)
        self.assertIsNotNone(diag.fallback_reason)

    def test_tier3_limit_fallback_diagnostics(self) -> None:
        view_sql = (
            "CREATE VIEW v_top_orders AS "
            "SELECT id, amount FROM orders ORDER BY amount DESC LIMIT 10;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics
        self.assertEqual(diag.strategy, "atomic_query_sync")
        self.assertEqual(diag.tier, 3)
        self.assertFalse(diag.is_incremental)
        self.assertIsNotNone(diag.fallback_reason)
        self.assertIn("LIMIT", diag.fallback_reason)

    def test_diagnostics_dict_and_json_interfaces(self) -> None:
        view_sql = (
            "CREATE VIEW v_sales AS "
            "SELECT customer_id, COUNT(*) AS cnt FROM orders GROUP BY customer_id;"
        )
        plan = generate_materialized_view(self.schemas, view_sql, dialect=Dialect.SQLITE)
        diag = plan.diagnostics

        # Attribute access
        self.assertEqual(diag.strategy, "algebraic_delta")
        self.assertEqual(diag.tier, 1)

        # Mapping / dict interface
        self.assertEqual(diag["strategy"], "algebraic_delta")
        self.assertEqual(diag["tier"], 1)
        self.assertEqual(diag.get("algebraic_mode"), "single")
        self.assertIn("strategy", diag)
        self.assertIn("tier", diag.keys())

        # Serialization
        d_dict = diag.to_dict()
        self.assertIsInstance(d_dict, dict)
        encoded = json.dumps(d_dict)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["strategy"], "algebraic_delta")
        self.assertEqual(decoded["tier"], 1)

    def test_default_fallback_diagnostics_on_unfinalized_plan(self) -> None:
        # If MaterializedViewPlan is constructed directly without diagnostics,
        # __post_init__ automatically builds default diagnostics.
        v_spec = mv.parse_view_definition(
            "CREATE VIEW v_raw AS SELECT customer_id FROM orders;",
            known_tables={"orders"},
        )
        raw_plan = mv.MaterializedViewPlan(
            dialect=Dialect.SQLITE,
            strategy=MaintenanceStrategy.ATOMIC_QUERY_SYNC,
            view_spec=v_spec,
            backing_table_name="__matview_store_v_raw",
            backing_table_ddl="CREATE TABLE __matview_store_v_raw (customer_id INT);",
            auxiliary_table_ddls=[],
            index_ddls=[],
            initial_populate_sqls=[],
            trigger_ddls=[],
        )
        self.assertIsNotNone(raw_plan.diagnostics)
        self.assertEqual(raw_plan.diagnostics.strategy, "atomic_query_sync")
        self.assertEqual(raw_plan.diagnostics.tier, 3)


class TestMaterializedViewManagerDiff(unittest.TestCase):
    """Tests MaterializedViewManager.diff_view and diff_views inspection methods."""

    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.conn = sqlite3.connect(self.tmp.name)
        self.conn.execute("CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, amount INT);")
        self.conn.commit()
        self.mgr = MaterializedViewManager(self.conn, dialect=Dialect.SQLITE)

    def tearDown(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_diff_fresh_view_create(self) -> None:
        view_sql = "CREATE VIEW v_sales AS SELECT customer_id, COUNT(*) AS cnt FROM orders GROUP BY customer_id;"
        diff = self.mgr.diff_view(view_sql)
        self.assertIsInstance(diff, ViewDiff)
        self.assertEqual(diff.view_name, "v_sales")
        self.assertEqual(diff.status, "CREATE")
        self.assertEqual(diff.action, "CREATE")
        self.assertTrue(diff.has_changes)
        self.assertEqual(diff.strategy, "algebraic_delta")
        self.assertIsNone(diff.current_strategy)
        self.assertIsNotNone(diff.signature_hash)
        self.assertIsNone(diff.current_signature_hash)
        self.assertGreater(len(diff.pending_ddls), 0)
        self.assertEqual(diff.pending_changes["backing_table"]["action"], "CREATE")
        self.assertIn("to_create", diff.pending_changes["triggers"])
        self.assertGreater(len(diff.pending_changes["triggers"]["to_create"]), 0)

        # Mapping interface
        self.assertEqual(diff["status"], "CREATE")
        self.assertTrue(diff["has_changes"])
        self.assertIn("diagnostics", diff)

    def test_diff_unchanged_registered_view_noop(self) -> None:
        view_sql = "CREATE VIEW v_sales AS SELECT customer_id, COUNT(*) AS cnt FROM orders GROUP BY customer_id;"
        self.mgr.register_view(view_sql)

        diff = self.mgr.diff_view(view_sql)
        self.assertEqual(diff.status, "NOOP")
        self.assertEqual(diff.action, "NOOP")
        self.assertFalse(diff.has_changes)
        self.assertEqual(len(diff.pending_ddls), 0)
        self.assertEqual(diff.pending_changes["backing_table"]["action"], "KEEP")
        self.assertEqual(len(diff.pending_changes["triggers"]["to_create"]), 0)
        self.assertEqual(len(diff.pending_changes["triggers"]["to_drop"]), 0)
        self.assertEqual(diff.signature_hash, diff.current_signature_hash)
        self.assertIn("no pending changes", diff.summary)

    def test_diff_modified_query_replace(self) -> None:
        view_sql1 = "CREATE VIEW v_sales AS SELECT customer_id, COUNT(*) AS cnt FROM orders GROUP BY customer_id;"
        self.mgr.register_view(view_sql1)

        view_sql2 = "CREATE VIEW v_sales AS SELECT customer_id, COUNT(*) AS cnt, SUM(amount) AS tot FROM orders GROUP BY customer_id;"
        diff = self.mgr.diff_view(view_sql2)
        self.assertEqual(diff.status, "REPLACE")
        self.assertEqual(diff.action, "REPLACE")
        self.assertTrue(diff.has_changes)
        self.assertNotEqual(diff.signature_hash, diff.current_signature_hash)
        self.assertEqual(diff.pending_changes["backing_table"]["action"], "REPLACE")
        self.assertGreater(len(diff.pending_ddls), 0)
        self.assertGreater(len(diff.pending_changes["triggers"]["to_create"]), 0)
        self.assertGreater(len(diff.pending_changes["triggers"]["to_drop"]), 0)

    def test_diff_views_multi_view(self) -> None:
        v1_sql = "CREATE VIEW v_cnt AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
        v2_sql = "CREATE VIEW v_sum AS SELECT customer_id, SUM(amount) AS s FROM orders GROUP BY customer_id;"
        self.mgr.register_view(v1_sql)

        report = self.mgr.diff_views([v1_sql, v2_sql])
        self.assertIsInstance(report, ViewDiffReport)
        self.assertEqual(len(report), 2)
        self.assertTrue(report.has_changes)
        self.assertEqual(report["v_cnt"].status, "NOOP")
        self.assertEqual(report["v_sum"].status, "CREATE")
        self.assertEqual(report.counts["NOOP"], 1)
        self.assertEqual(report.counts["CREATE"], 1)

    def test_diff_views_prune_removed_drop(self) -> None:
        v1_sql = "CREATE VIEW v_keep AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
        v2_sql = "CREATE VIEW v_drop AS SELECT customer_id, SUM(amount) AS s FROM orders GROUP BY customer_id;"
        self.mgr.register_view(v1_sql)
        self.mgr.register_view(v2_sql)

        # Diff with only v_keep declared and prune_removed=True
        report = self.mgr.diff_views([v1_sql], prune_removed=True)
        self.assertEqual(len(report), 2)
        self.assertEqual(report["v_keep"].status, "NOOP")
        self.assertEqual(report["v_drop"].status, "DROP")
        self.assertEqual(report["v_drop"].action, "DROP")
        self.assertTrue(report["v_drop"].has_changes)
        self.assertEqual(report.counts["DROP"], 1)
        self.assertEqual(report.counts["NOOP"], 1)
        self.assertGreater(len(report["v_drop"].pending_ddls), 0)

    def test_diff_out_of_band_schema_drift(self) -> None:
        view_sql = "CREATE VIEW v_star AS SELECT * FROM orders;"
        self.mgr.register_view(view_sql)

        # Baseline diff is NOOP
        diff_pre = self.mgr.diff_view(view_sql)
        self.assertEqual(diff_pre.status, "NOOP")

        # Mutate underlying table out-of-band
        self.conn.execute("ALTER TABLE orders ADD COLUMN priority INT;")
        self.conn.commit()

        # Diff detects schema signature drift
        diff_post = self.mgr.diff_view(view_sql)
        self.assertEqual(diff_post.status, "REPLACE")
        self.assertTrue(diff_post.has_changes)

    def test_diff_out_of_band_missing_backing_table(self) -> None:
        view_sql = "CREATE VIEW v_sales AS SELECT customer_id, COUNT(*) AS cnt FROM orders GROUP BY customer_id;"
        plan = self.mgr.register_view(view_sql)

        # Drop backing table behind the catalog's back
        self.conn.execute(f"DROP TABLE {plan.backing_table_name};")
        self.conn.commit()

        diff = self.mgr.diff_view(view_sql)
        self.assertEqual(diff.status, "REPLACE")
        self.assertTrue(diff.has_changes)

    def test_diff_dry_run_preserves_database_state(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view', 'trigger');")
        initial_objects = set(cur.fetchall())

        view_sql = "CREATE VIEW v_sales AS SELECT customer_id, COUNT(*) AS cnt FROM orders GROUP BY customer_id;"
        diff = self.mgr.diff_view(view_sql)
        self.assertEqual(diff.status, "CREATE")

        cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view', 'trigger');")
        after_objects = set(cur.fetchall())
        self.assertEqual(initial_objects, after_objects, "diff_view must not mutate database schema state.")


class TestCLIDiffCommand(unittest.TestCase):
    """Tests the matview diff CLI command entrypoint."""

    def setUp(self) -> None:
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.conn = sqlite3.connect(self.tmp.name)
        self.conn.execute("CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, amount INT);")
        self.conn.commit()
        self.conn.close()

    def tearDown(self) -> None:
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_cli_diff_subcommand_fresh_view(self) -> None:
        v_sql = "CREATE VIEW v_cli AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["diff", "--db", self.tmp.name, "--view", v_sql])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("[CREATE] v_cli", out)
        self.assertIn("Strategy: algebraic_delta", out)
        self.assertIn("Backing Table: __matview_store_v_cli [CREATE]", out)
        self.assertIn("Summary: 1 view(s) with pending changes", out)

    def test_cli_diff_flag_compatibility(self) -> None:
        v_sql = "CREATE VIEW v_cli AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["--diff", "--db", self.tmp.name, "--view", v_sql])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("[CREATE] v_cli", out)

    def test_cli_diff_noop_view(self) -> None:
        v_sql = "CREATE VIEW v_cli AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
        # Register view first
        rc_apply = cli_main(["--db", self.tmp.name, "--view", v_sql, "--apply"])
        self.assertEqual(rc_apply, 0)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["diff", "--db", self.tmp.name, "--view", v_sql])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("[NOOP] v_cli", out)
        self.assertIn("Up to date; no pending changes.", out)
        self.assertIn("Summary: 0 view(s) with pending changes", out)

    def test_cli_diff_modified_replace_view(self) -> None:
        v_sql1 = "CREATE VIEW v_cli AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
        cli_main(["--db", self.tmp.name, "--view", v_sql1, "--apply"])

        v_sql2 = "CREATE VIEW v_cli AS SELECT customer_id, COUNT(*) AS c, SUM(amount) AS s FROM orders GROUP BY customer_id;"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["diff", "--db", self.tmp.name, "--view", v_sql2])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("[REPLACE] v_cli", out)
        self.assertIn("pending REPLACE", out)

    def test_cli_diff_json_output(self) -> None:
        v_sql = "CREATE VIEW v_cli AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["diff", "--db", self.tmp.name, "--view", v_sql, "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertIn("views", data)
        self.assertIn("summary", data)
        self.assertIn("v_cli", data["views"])
        entry = data["views"]["v_cli"]
        self.assertEqual(entry["status"], "CREATE")
        self.assertTrue(entry["has_changes"])
        self.assertEqual(entry["strategy"], "algebraic_delta")
        self.assertIn("diagnostics", entry)
        self.assertEqual(entry["diagnostics"]["tier"], 1)
        self.assertIn("pending_changes", entry)
        self.assertEqual(entry["pending_changes"]["backing_table"]["action"], "CREATE")

    def test_cli_diff_exit_code_flag(self) -> None:
        v_sql = "CREATE VIEW v_cli AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"

        # 1. Fresh view with differences -> exit code 1 when --exit-code passed
        rc_changes = cli_main(["diff", "--db", self.tmp.name, "--view", v_sql, "--exit-code"])
        self.assertEqual(rc_changes, 1)

        # 2. Apply view -> unchanged NOOP -> exit code 0 when --exit-code passed
        cli_main(["--db", self.tmp.name, "--view", v_sql, "--apply"])
        rc_noop = cli_main(["diff", "--db", self.tmp.name, "--view", v_sql, "--exit-code"])
        self.assertEqual(rc_noop, 0)

    def test_cli_diff_missing_arguments(self) -> None:
        # Missing --db and --schema
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc_nodb = cli_main(["diff"])
        self.assertEqual(rc_nodb, 2)
        self.assertIn("requires --db", err_buf.getvalue())

        # Database with no catalog and no --view
        err_buf2 = io.StringIO()
        with redirect_stderr(err_buf2):
            rc_noview = cli_main(["diff", "--db", self.tmp.name])
        self.assertEqual(rc_noview, 2)
        self.assertIn("--view is required", err_buf2.getvalue())

    def test_cli_diff_limitation_rejection_and_override(self) -> None:
        v_nondet = "CREATE VIEW v_nondet AS SELECT id, customer_id, CURRENT_TIMESTAMP AS ts FROM orders;"

        # Without --ignore-limitations: rejected with exit code 2
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc_reject = cli_main(["diff", "--db", self.tmp.name, "--view", v_nondet])
        self.assertEqual(rc_reject, 2)
        self.assertIn("NONDETERMINISTIC_EXPR", err_buf.getvalue())

        # With --ignore-limitations: succeeds with exit code 0 and records limitations
        out_buf = io.StringIO()
        with redirect_stdout(out_buf):
            rc_override = cli_main([
                "diff", "--db", self.tmp.name, "--view", v_nondet, "--ignore-limitations"
            ])
        self.assertEqual(rc_override, 0)
        self.assertIn("[CREATE] v_nondet", out_buf.getvalue())
        self.assertIn("Limitations (1):", out_buf.getvalue())
        self.assertIn("NONDETERMINISTIC_EXPR", out_buf.getvalue())

    def test_cli_diff_schema_only_without_db(self) -> None:
        schema_sql = "CREATE TABLE items (id INT PRIMARY KEY, grp INT, val INT);"
        view_sql = "CREATE VIEW v_items AS SELECT grp, COUNT(*) AS cnt FROM items GROUP BY grp;"

        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as s_f:
            s_f.write(schema_sql)
            schema_path = s_f.name

        with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as v_f:
            v_f.write(view_sql)
            view_path = v_f.name

        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["diff", "--schema", schema_path, "--view", view_path])
            self.assertEqual(rc, 0)
            self.assertIn("[CREATE] v_items", buf.getvalue())
            self.assertIn("Strategy: algebraic_delta", buf.getvalue())
        finally:
            if os.path.exists(schema_path):
                os.unlink(schema_path)
            if os.path.exists(view_path):
                os.unlink(view_path)

    def test_diff_detects_missing_live_triggers_and_dropped_reader_view(self) -> None:
        mgr = MaterializedViewManager(self.tmp.name)
        try:
            v_sql = "CREATE VIEW v_drift AS SELECT customer_id, COUNT(*) AS c FROM orders GROUP BY customer_id;"
            plan = mgr.register_view(v_sql)

            # Drop ALL triggers in database out-of-band so live_triggers == set()
            for trig_name in plan.trigger_names:
                mgr.conn.execute(f"DROP TRIGGER IF EXISTS {trig_name};")
            mgr.conn.commit()

            rep_trigs = mgr.diff_views(v_sql)
            diff_trigs = rep_trigs["v_drift"]
            self.assertEqual(diff_trigs.status, "ALTER")
            self.assertTrue(diff_trigs.has_changes)
            self.assertEqual(
                set(diff_trigs.pending_changes["triggers"]["to_create"]),
                set(plan.trigger_names),
            )
            self.assertTrue(any("CREATE TRIGGER" in s for s in diff_trigs.pending_ddls))

            # Re-create triggers via repair, then drop only the reader view out-of-band
            mgr.repair_view("v_drift")
            mgr.conn.execute("DROP VIEW IF EXISTS v_drift;")
            mgr.conn.commit()

            rep_view = mgr.diff_views(v_sql)
            diff_view = rep_view["v_drift"]
            self.assertEqual(diff_view.status, "ALTER")
            self.assertTrue(diff_view.has_changes)
            self.assertEqual(diff_view.pending_changes["reader_view"]["action"], "CREATE")
            self.assertTrue(any("CREATE VIEW" in s for s in diff_view.pending_ddls))
        finally:
            mgr.conn.close()



class TestCLICompileLimitationContract(unittest.TestCase):
    """The compile entry point honours the limitation contract (ported from PR #60).

    Default: exit 2, no SQL on stdout, every structured limitation code on
    stderr. `--ignore-limitations`: exit 0, SQL emitted, and exactly one
    EngineLimitationWarning carrying the same codes. The expected codes come
    from the API's structured EngineLimitationError, not from CLI text.
    """

    SCHEMA = "CREATE TABLE t (id INT PRIMARY KEY, grp TEXT, val INT);"

    def _run(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always", EngineLimitationWarning)
            with redirect_stdout(out), redirect_stderr(err):
                rc = cli_main(argv)
        limitation_warnings = [w for w in emitted if issubclass(w.category, EngineLimitationWarning)]
        return rc, out.getvalue(), err.getvalue(), limitation_warnings

    def test_default_rejection_and_explicit_override(self) -> None:
        cases = [
            ("mysql:8.4", "CREATE VIEW v AS SELECT grp, COUNT(*) AS c FROM t GROUP BY grp;"),
            ("mariadb:10.11", "CREATE VIEW v AS SELECT grp, MAX(val) AS m FROM t GROUP BY grp;"),
            ("postgres:18", "CREATE VIEW v AS SELECT grp, RANDOM() AS r FROM t;"),
        ]
        for dialect, view in cases:
            with self.subTest(dialect=dialect, view=view):
                with self.assertRaises(EngineLimitationError) as raised:
                    generate_materialized_view(self.SCHEMA, view, dialect=dialect, ignore_limitations=False)
                codes = {lim.code for lim in raised.exception.limitations}
                self.assertTrue(codes)
                self.assertNotIn("", codes)

                rc, out, err, lims = self._run(["--schema", self.SCHEMA, "--view", view, "--dialect", dialect])
                self.assertEqual(rc, 2)
                self.assertEqual(out, "")
                self.assertEqual(lims, [])
                for code in codes:
                    self.assertIn(f"[{code}]", err)

                rc, out, err, lims = self._run(
                    ["--schema", self.SCHEMA, "--view", view, "--dialect", dialect, "--ignore-limitations"]
                )
                self.assertEqual(rc, 0, err)
                self.assertIn("CREATE", out)
                self.assertEqual(len(lims), 1)
                for code in codes:
                    self.assertIn(code, str(lims[0].message))

    def test_unsupported_statistics_are_rejected_even_with_override(self) -> None:
        for dialect, view, extras in (
            ("sqlite", "CREATE VIEW v AS SELECT grp, VARIANCE(val) AS v FROM t GROUP BY grp;", ([], ["--ignore-limitations"])),
            ("postgres:15", "CREATE VIEW v AS SELECT grp, STD(val) AS s FROM t GROUP BY grp;", ([], ["--ignore-limitations"])),
            # MariaDB statistics it cannot reproduce exactly: the override (needed
            # for every MySQL-family view) must not get past the refusal.
            ("mariadb:10.11", "CREATE VIEW v AS SELECT grp, VAR_SAMP(val) * 2 AS v2 FROM t GROUP BY grp;",
             (["--ignore-limitations"],
              # Past the order-sensitivity opt-in too: MariaDB's own refusal must hold.
              ["--ignore-limitations", "--accept-limitation", "ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE",
               "--accept-limitation", "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX"])),
        ):
            for extra in extras:
                with self.subTest(dialect=dialect, extra=extra):
                    rc, out, err, _ = self._run(
                        ["--schema", self.SCHEMA, "--view", view, "--dialect", dialect, *extra]
                    )
                    self.assertEqual(rc, 2)
                    self.assertEqual(out, "")
                    self.assertRegex(
                        err, r"VAR_POP/VAR_SAMP/STDDEV_POP/STDDEV_SAMP|cannot reproduce exactly|ORDER_SENSITIVE_AGGREGATE"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
