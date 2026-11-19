#!/usr/bin/env python3
"""
PostgreSQL statement-level maintenance (transition tables): code shape and
emulated semantics.

1. DDL structure for every strategy: statement-level triggers only (no row
   trigger, no row loop), an early return on an empty transition table, one
   consolidated signed delta (UPDATE = old -1 UNION ALL new +1), key-ordered
   upserts, empty-group deletes restricted to retracted keys, native
   recomputes in a statement after the lock statement, a per-view lock row
   for group-scoped and full recomputes, and the partition install block.
2. sqlglot parses every generated statement as PostgreSQL.
3. Behaviour through TranspiledConnectionAdapter (an EMULATION: SQLite row
   triggers run the set-based body for one-row deltas). Native semantics are
   in test_native_bulk_statements.py and test_pg_statement_maintenance.py.
4. A native check of expression GROUP BY recomputes when POSTGRES_DSN is set.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import unittest
import uuid
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
from test_matview_sync import FLOAT_OPT_IN, TranspiledConnectionAdapter


def _functions(plan: MaterializedViewPlan) -> dict:
    """Trigger function DDL by suffix (ai, au, ad, trunc) for a single-source view."""
    out = {}
    for ddl in plan.trigger_ddls:
        m = re.match(r"CREATE OR REPLACE FUNCTION \S+_(ai|au|ad|trunc)\(\)", ddl)
        if m and "fn_readonly_" not in ddl:
            out[m.group(1)] = ddl
    return out


def _body(fn_ddl: str) -> str:
    m = re.search(r"END IF;\n(.*)\n    RETURN NULL;\nEND;", fn_ddl, flags=re.S)
    assert m, fn_ddl
    return m.group(1)


SALES = """
CREATE TABLE sales (
    id INT PRIMARY KEY,
    region VARCHAR(32),
    amount NUMERIC(10, 2) NOT NULL,
    qty INT NOT NULL
);
"""


class TestPGStatementLevelDDL(unittest.TestCase):
    def _plan(self, view: str, schema: str = SALES) -> MaterializedViewPlan:
        return generate_materialized_view(schema, view, dialect=Dialect.POSTGRES, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)

    def assert_statement_level_only(self, plan: MaterializedViewPlan) -> None:
        ddls = "\n".join(plan.trigger_ddls)
        self.assertNotIn("FOR EACH ROW", ddls)
        self.assertNotRegex(ddls, r"FOR\s+(NEW|OLD)\s+IN\s+SELECT")
        self.assertNotIn("skip_triggers", ddls)
        triggers = [d for d in plan.trigger_ddls if d.startswith("CREATE TRIGGER")]
        events = sorted(re.search(r"AFTER (\w+)", t).group(1) for t in triggers)  # type: ignore[union-attr]
        n_sources = len(plan.view_spec.referenced_tables)
        self.assertEqual(events, sorted(["DELETE", "INSERT", "TRUNCATE", "UPDATE"] * n_sources))
        for t in triggers:
            self.assertIn("FOR EACH STATEMENT", t)
            if "TRUNCATE" not in t:
                self.assertIn("REFERENCING", t)
        self.assertEqual(sum(d.startswith("DO $mv_install$") for d in plan.trigger_ddls), n_sources)
        for suffix, fn in _functions(plan).items():
            if suffix != "trunc":
                table = "__mv_old_rows" if suffix == "ad" else "__mv_new_rows"
                self.assertIn(f"IF NOT EXISTS (SELECT 1 FROM {table}) THEN\n        RETURN NULL;", fn)

    def test_grouped_aggregate_signed_ordered_restricted(self) -> None:
        plan = self._plan(
            "CREATE VIEW v AS SELECT region, COUNT(*) AS cnt, SUM(amount) AS total, MIN(qty) AS lo "
            "FROM sales GROUP BY region;"
        )
        self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assert_statement_level_only(plan)
        fns = _functions(plan)
        au = _body(fns["au"])
        # One consolidated signed delta: both transition tables in one relation.
        self.assertRegex(au, r"SELECT -1 AS __mv_s, __mv_o\.\* FROM __mv_old_rows AS __mv_o\s+UNION ALL\s+SELECT 1 AS __mv_s")
        upsert, rest = au.split(";", 1)
        self.assertRegex(upsert, r"(?s)GROUP BY .*\n    ORDER BY .*\n    ON CONFLICT")
        # Empty-group delete restricted to retracted keys, after the upsert.
        self.assertRegex(rest, r"DELETE FROM __matview_store_v\s+WHERE __matview_store_v\.__mv_pk IN \(SELECT .*")
        self.assertIn("WHERE __mv_p.__mv_s < 0)", rest)
        # MIN recompute: its own UPDATE statement after the upsert, only for
        # groups whose retracted value reaches the stored minimum.
        self.assertNotIn("SELECT MIN(", upsert)
        self.assertRegex(rest, r"(?s)UPDATE __matview_store_v SET\s+lo = CASE WHEN \(__mv_r\.__mv_r0 IS NOT NULL")
        # Insert-only statements never retract: no delete, no recompute.
        ai = _body(fns["ai"])
        self.assertNotIn("DELETE FROM", ai)
        self.assertNotIn("UPDATE __matview_store_v", ai)
        self.assertNotIn("__mv_old_rows", ai)

    def test_expression_group_key_native_recompute_is_set_based(self) -> None:
        plan = self._plan(
            "CREATE VIEW v AS SELECT COALESCE(region, 'UNK') AS r, AVG(amount) AS a, VAR_POP(qty) AS vq "
            "FROM sales GROUP BY COALESCE(region, 'UNK');"
        )
        self.assert_statement_level_only(plan)
        ai = _body(_functions(plan)["ai"])
        # Natively recomputed per affected group, correlated with the stored
        # group column (not with a raw transition-table column).
        self.assertRegex(ai, r"(?s)UPDATE __matview_store_v SET\s+a = [^\n]*\(SELECT AVG\(amount\) FROM sales WHERE")
        self.assertIn("__matview_store_v.r", ai)

    def test_projection_keyed_by_source_primary_key(self) -> None:
        plan = self._plan("CREATE VIEW v AS SELECT id, region, amount FROM sales WHERE qty > 0;")
        self.assert_statement_level_only(plan)
        au = _body(_functions(plan)["au"])
        self.assertRegex(au, r"(?s)^\s*DELETE FROM __matview_store_v\s+WHERE .* IN \(SELECT .*\)\s+AND NOT .* IN \(SELECT")
        self.assertRegex(au, r"ORDER BY __mv_p\.__mv_k0\n    ON CONFLICT \(id\) DO UPDATE SET")

    def test_distinct_aux_tables_follow_the_group_upsert(self) -> None:
        plan = self._plan("CREATE VIEW v AS SELECT region, COUNT(DISTINCT qty) AS dq FROM sales GROUP BY region;")
        self.assert_statement_level_only(plan)
        au = _body(_functions(plan)["au"])
        aux = plan.auxiliary_table_names[0]
        self.assertLess(au.index("INSERT INTO __matview_store_v"), au.index(f"INSERT INTO {aux}"))
        self.assertIn(f"DELETE FROM {aux}", au)
        self.assertIn(f"{aux}.__ref_cnt <= 0", au)

    def test_join_aggregate_joins_the_signed_delta(self) -> None:
        schema = SALES + "CREATE TABLE regions (region VARCHAR(32) PRIMARY KEY, zone VARCHAR(8) NOT NULL);"
        plan = self._plan(
            "CREATE VIEW v AS SELECT r.zone, SUM(s.amount) AS total, COUNT(*) AS c "
            "FROM sales s JOIN regions r ON s.region = r.region GROUP BY r.zone;",
            schema,
        )
        self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assert_statement_level_only(plan)
        ddls = "\n".join(plan.trigger_ddls)
        self.assertRegex(ddls, r"FROM \(\s+SELECT 1 AS __mv_s, __mv_n\.\* FROM __mv_new_rows AS __mv_n\s+\) AS s\s+INNER JOIN regions AS r")

    def test_recompute_strategies_take_the_view_lock_first(self) -> None:
        for view in (
            "CREATE VIEW v AS SELECT id, region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY id) AS rn FROM sales;",
            "CREATE VIEW v AS SELECT region, STRING_AGG(CAST(id AS TEXT), ',') AS ids FROM sales GROUP BY region;",
        ):
            with self.subTest(view=view):
                plan = self._plan(view)
                self.assertIn(plan.strategy, (MaintenanceStrategy.GROUP_SCOPED_DELTA, MaintenanceStrategy.ATOMIC_QUERY_SYNC))
                self.assert_statement_level_only(plan)
                self.assertIn("__matview_lock_v", plan.auxiliary_table_names)
                for suffix, fn in _functions(plan).items():
                    body = _body(fn) if suffix != "trunc" else fn
                    lock = body.index("INSERT INTO __matview_lock_v")
                    self.assertLess(lock, body.index("DELETE FROM __matview_store_v"), suffix)

    def test_sqlglot_postgres_syntax_validity(self) -> None:
        """Every statement of every trigger function body parses as PostgreSQL."""
        schema = SALES + "CREATE TABLE regions (region VARCHAR(32) PRIMARY KEY, zone VARCHAR(8) NOT NULL);"
        views = [
            "CREATE VIEW v1 AS SELECT region, COUNT(*) AS c, SUM(amount) AS s, AVG(qty) AS a, MAX(qty) AS hi FROM sales GROUP BY region;",
            "CREATE VIEW v2 AS SELECT id, region FROM sales WHERE qty > 1;",
            "CREATE VIEW v3 AS SELECT DISTINCT region FROM sales;",
            "CREATE VIEW v4 AS SELECT r.zone, COUNT(*) AS c FROM sales s JOIN regions r ON s.region = r.region GROUP BY r.zone;",
            "CREATE VIEW v5 AS SELECT region, COUNT(DISTINCT qty) AS dq FROM sales GROUP BY region;",
            "CREATE VIEW v6 AS SELECT region FROM sales UNION SELECT region FROM regions;",
        ]
        for view in views:
            plan = self._plan(view, schema)
            for ddl in plan.trigger_ddls:
                m = re.search(r"AS \$\$(.*)\$\$ LANGUAGE plpgsql", ddl, flags=re.S)
                if not m:
                    continue
                body = re.sub(r"(?s)^\s*BEGIN\s*(IF NOT EXISTS .*?END IF;)?", "", m.group(1))
                body = re.sub(r"(?s)RETURN NULL;\s*END;\s*$", "", body)
                for stmt in _split_top_level(body, ";"):
                    if stmt.strip() and stmt.strip().upper() not in ("BEGIN",):
                        try:
                            sqlglot.parse(stmt, read="postgres")
                        except Exception as exc:  # pragma: no cover - failure path
                            self.fail(f"sqlglot cannot parse {stmt!r}: {exc}")

    def test_cli_rejects_removed_transition_table_flags(self) -> None:
        for flag in ("--pg-transition-tables", "--no-pg-vectorize-transition-tables"):
            with self.subTest(flag=flag):
                with contextlib.redirect_stderr(io.StringIO()) as err, self.assertRaises(SystemExit):
                    cli_main(["--schema", SALES, "--view", "CREATE VIEW v AS SELECT region FROM sales;",
                              "--dialect", "postgres", flag])
                self.assertIn("unrecognized arguments", err.getvalue())


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

    # A single-statement primary-key swap is a statement-level property: the
    # per-row emulation cannot represent it. Native coverage:
    # test_pg_statement_maintenance.py::test_primary_key_permutations.

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

    def test_native_avg_recompute_with_expression_group_key_is_set_based(self) -> None:
        """Expression GROUP BY plus native AVG recomputation must not correlate an ungrouped raw column."""
        schema_sql = """
        CREATE TABLE orders (
            id INT PRIMARY KEY,
            region TEXT,
            amount NUMERIC(12, 2)
        );
        """
        # NUMERIC AVG keeps native recomputation. Integer AVG is derived from
        # SUM/COUNT and needs no correlated recompute; that path is covered by
        # test_integer_avg_with_expression_group_key_is_vectorized_and_derived.
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
        )
        fn_sql = "\n".join(
            statement for statement in plan.trigger_ddls
            if "CREATE OR REPLACE FUNCTION" in statement
        )
        self.assertNotRegex(fn_sql, r"FOR\s+(NEW|OLD)\s+IN\s+SELECT")

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

    def test_integer_avg_with_expression_group_key_is_vectorized_and_derived(self) -> None:
        """Integer AVG with an expression GROUP BY on the transition-table path.

        PostgreSQL AVG(int) is numeric_div(sum::numeric, count::numeric), so it
        is derived from the SUM/COUNT folded in the same statement: no
        correlated base-table recompute (which would force the row loop for an
        expression group key) and no pre-lock snapshot read. Execution here is
        the SQLite transpilation adapter, not native PostgreSQL; native value
        and scale equality is covered by test_avg_concurrent_maintenance.py.
        """
        schema_sql = """
        CREATE TABLE orders_i (
            id INT PRIMARY KEY,
            region TEXT,
            amount INT
        );
        """
        view_sql = """
        CREATE VIEW pgtr_expr_avg_int AS
        SELECT COALESCE(region, 'UNK') AS reg_norm,
               COUNT(*) AS cnt,
               AVG(amount) AS a_amt
        FROM orders_i
        GROUP BY COALESCE(region, 'UNK');
        """
        self._execute_script(schema_sql)
        plan = generate_materialized_view(
            schema_sql,
            view_sql,
            dialect=Dialect.POSTGRES,
        )
        fn_sql = "\n".join(
            statement for statement in plan.trigger_ddls
            if "CREATE OR REPLACE FUNCTION" in statement
        )
        self.assertNotIn("FOR NEW IN SELECT * FROM __mv_new_rows LOOP", fn_sql)
        self.assertNotIn("FOR OLD IN SELECT * FROM __mv_old_rows LOOP", fn_sql)
        self.assertNotRegex(fn_sql, r"\(SELECT AVG\(")
        self.assertIn(")::NUMERIC / (", fn_sql)

        for ddl in plan.create_structure_statements() + plan.trigger_ddls:
            self.cur.execute(ddl)

        def assert_matches_live_view() -> None:
            self.cur.execute("SELECT reg_norm, cnt, a_amt FROM pgtr_expr_avg_int ORDER BY reg_norm;")
            actual = self.cur.fetchall()
            self.cur.execute(
                "SELECT COALESCE(region, 'UNK') AS reg_norm, COUNT(*), AVG(amount) "
                "FROM orders_i GROUP BY COALESCE(region, 'UNK') ORDER BY reg_norm;"
            )
            self.assertEqual(actual, self.cur.fetchall())

        self.cur.execute(
            "INSERT INTO orders_i VALUES (1, 'NA', 10), (2, 'NA', 25), (3, NULL, 30), (4, NULL, NULL);"
        )
        assert_matches_live_view()
        self.cur.execute("UPDATE orders_i SET amount = amount + 5, region = NULL WHERE id = 2;")
        assert_matches_live_view()
        self.cur.execute("DELETE FROM orders_i WHERE id IN (1, 3);")
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
        )
        fn_sql = "\n".join(s for s in plan.trigger_ddls if "CREATE OR REPLACE FUNCTION" in s)
        self.assertIn("(__mv_pk, org, cnt, sb, ab, __mv_cnt_sb_1)", fn_sql)
        self.assertNotIn("cnt, cnt", fn_sql)


@unittest.skipUnless(
    os.environ.get("POSTGRES_DSN"),
    "POSTGRES_DSN is not set; native PostgreSQL integration test only",
)
class TestNativePGTransitionTableVectorizationSemantics(unittest.TestCase):
    """Native PostgreSQL regression for expression-group native aggregates."""

    def test_expression_group_avg_and_variance_match_live_view_after_batch_insert(self) -> None:
        import psycopg2

        schema_name = f"mv_pg_transition_expr_{uuid.uuid4().hex[:12]}"
        conn = psycopg2.connect(os.environ["POSTGRES_DSN"])
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA "{schema_name}"')
                cur.execute(f'SET search_path TO "{schema_name}"')
                schema_sql = """
                CREATE TABLE orders (
                    id INT PRIMARY KEY,
                    region TEXT,
                    amount DOUBLE PRECISION NOT NULL
                );
                """
                view_sql = """
                CREATE VIEW region_stats AS
                SELECT
                    COALESCE(region, 'UNK') AS region_norm,
                    AVG(amount) AS avg_amount,
                    VAR_POP(amount) AS var_amount,
                    STDDEV_POP(amount) AS stddev_amount
                FROM orders
                GROUP BY COALESCE(region, 'UNK');
                """
                cur.execute(schema_sql)
                # AVG/VAR over DOUBLE PRECISION: per-code opt-in (values are
                # small integers, so the result does not depend on the order).
                plan = generate_materialized_view(
                    schema_sql,
                    view_sql,
                    dialect=Dialect.POSTGRES,
                    accept_limitations=FLOAT_OPT_IN,
                )
                for ddl in plan.create_structure_statements() + plan.trigger_ddls:
                    cur.execute(ddl)

                oracle_sql = """
                SELECT
                    COALESCE(region, 'UNK') AS region_norm,
                    AVG(amount) AS avg_amount,
                    VAR_POP(amount) AS var_amount,
                    STDDEV_POP(amount) AS stddev_amount
                FROM orders
                GROUP BY COALESCE(region, 'UNK')
                ORDER BY region_norm;
                """
                batches = (
                    "INSERT INTO orders VALUES "
                    "(1, 'NA', 10.0), (2, 'NA', 20.0), (3, NULL, 30.0), (4, 'APAC', 40.0);",
                    "INSERT INTO orders VALUES "
                    "(5, 'NA', 50.0), (6, 'EU', 60.0), (7, NULL, 90.0);",
                )
                for statement in batches:
                    cur.execute(statement)
                    cur.execute(
                        "SELECT region_norm, avg_amount, var_amount, stddev_amount "
                        "FROM region_stats ORDER BY region_norm;"
                    )
                    materialized_rows = cur.fetchall()
                    cur.execute(oracle_sql)
                    live_rows = cur.fetchall()
                    self.assertEqual(materialized_rows, live_rows)
        finally:
            try:
                with conn.cursor() as cur:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
