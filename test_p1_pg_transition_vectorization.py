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


_GUARD_RE = re.compile(r"\A    IF (NOT EXISTS \(.*?\)) THEN\n        RETURN NULL;\n    END IF;\n", re.S)
# One early-exit statement may OR several conditions (empty transition table,
# unchanged UPDATE): each `NOT EXISTS (...)` operand is one guard.
_GUARD_SPLIT_RE = re.compile(r" OR (?=NOT EXISTS \()")


def _guards_and_body(fn_ddl: str) -> "tuple[List[str], str]":
    """Split a statement-trigger function into its leading early-exit guards
    (``IF NOT EXISTS (<cond>) THEN RETURN NULL; END IF;``, returned as the
    ``NOT EXISTS (...)`` conditions) and the maintenance body that follows."""
    m = re.search(r"BEGIN\n(.*)\n    RETURN NULL;\nEND;", fn_ddl, flags=re.S)
    assert m, fn_ddl
    rest = m.group(1) + "\n"
    guards: List[str] = []
    while True:
        g = _GUARD_RE.match(rest)
        if not g:
            break
        guards.extend(_GUARD_SPLIT_RE.split(g.group(1)))
        rest = rest[g.end():]
    return guards, rest.rstrip("\n")


def _body(fn_ddl: str) -> str:
    """The maintenance statements of a trigger function, without early exits."""
    return _guards_and_body(fn_ddl)[1]


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
        after_triggers = [t for t in triggers if re.search(r"\bAFTER\s+\w+", t)]
        before_triggers = [t for t in triggers if re.search(r"\bBEFORE\s+INSERT\s+OR\s+UPDATE\s+OR\s+DELETE\s+OR\s+TRUNCATE\b", t)]
        self.assertEqual(len(after_triggers) + len(before_triggers), len(triggers))
        events = sorted(re.search(r"AFTER (\w+)", t).group(1) for t in after_triggers)  # type: ignore[union-attr]
        n_sources = len(plan.view_spec.referenced_tables)
        self.assertEqual(events, sorted(["DELETE", "INSERT", "TRUNCATE", "UPDATE"] * n_sources))
        if before_triggers:
            self.assertEqual(len(before_triggers), n_sources)
        for t in triggers:
            self.assertIn("FOR EACH STATEMENT", t)
            if t in after_triggers and "TRUNCATE" not in t:
                self.assertIn("REFERENCING", t)
        self.assertEqual(sum(d.startswith("DO $mv_install$") for d in plan.trigger_ddls), n_sources)
        for suffix, fn in _functions(plan).items():
            if suffix not in ("trunc", "bs", "reconcile"):
                table = "__mv_old_rows" if suffix == "ad" else "__mv_new_rows"
                self.assertIn(f"IF NOT EXISTS (SELECT 1 FROM {table})", fn)

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
        self.assertRegex(
            rest,
            r"(?s)UPDATE __matview_store_v SET\s+lo = CASE WHEN \(__mv_r\.__mv_r0 IS DISTINCT FROM NULL",
        )
        # Insert-only statements never retract: no delete, no recompute.
        ai = _body(fns["ai"])
        self.assertNotIn("DELETE FROM", ai)
        self.assertNotIn("UPDATE __matview_store_v", ai)
        self.assertNotIn("__mv_old_rows", ai)

    def _unchanged_guard(self, plan: MaterializedViewPlan, suffix: str = "au") -> Optional[str]:
        """The UPDATE early exit that skips maintenance when the paired OLD/NEW
        rows are equal in every column the view reads (None if absent)."""
        guards, body = _guards_and_body(_functions(plan)[suffix])
        # Every early exit precedes the first write of the maintenance body.
        self.assertNotRegex("\n".join(guards), r"\b(INSERT|UPDATE|DELETE)\b")
        found = [g for g in guards if "__mv_old_rows" in g and "__mv_new_rows" in g]
        self.assertLessEqual(len(found), 1, guards)
        return found[0] if found else None

    @staticmethod
    def _compared_columns(guard: str, alias: str) -> List[str]:
        """Columns compared for `alias` (old/new): identity-typed columns by
        direct IS DISTINCT FROM, the rest through one JSON-bytes comparison."""
        pairs = re.findall(r"__mv_old\.(\w+) IS DISTINCT FROM __mv_new\.(\w+)", guard)
        assert all(o == n for o, n in pairs), pairs
        direct = [o for o, _ in pairs]
        m = re.search(rf"row_to_json\(ROW\(((?:{alias}\.\w+(?:, )?)+)\)\)", guard)
        return direct + (re.findall(rf"{alias}\.(\w+)", m.group(1)) if m else [])

    def test_unchanged_update_guard_compares_exactly_the_referenced_columns(self) -> None:
        # Reads region and amount; qty is unread. The PK (id) is always compared.
        plan = self._plan("CREATE VIEW v AS SELECT region, amount FROM sales;")
        guard = self._unchanged_guard(plan)
        assert guard is not None
        # OLD/NEW pairs are matched on the primary key; an unmatched row on
        # either side (key change, insert/delete) counts as changed.
        self.assertRegex(guard, r"FROM __mv_old_rows AS __mv_old FULL OUTER JOIN __mv_new_rows AS __mv_new ON __mv_old\.id = __mv_new\.id WHERE ")
        self.assertIn("__mv_old.id IS NULL OR __mv_new.id IS NULL", guard)
        old_cols = self._compared_columns(guard, "__mv_old")
        self.assertEqual(old_cols, self._compared_columns(guard, "__mv_new"))
        self.assertEqual(sorted(old_cols), ["amount", "id", "region"])  # not qty
        # Identity-typed columns (id INT) are compared directly; numeric scale
        # and (possibly collated) text are compared as JSON bytes, never as text.
        self.assertIn("__mv_old.id IS DISTINCT FROM __mv_new.id", guard)
        self.assertRegex(guard, r"convert_to\(pg_catalog\.row_to_json\(ROW\(__mv_old\.region, __mv_old\.amount\)\)")
        self.assertNotIn("__mv_old.region IS DISTINCT FROM", guard)
        self.assertNotIn("__mv_old.amount IS DISTINCT FROM", guard)
        # Only UPDATE statements can pair rows: INSERT/DELETE bodies have no guard.
        self.assertIsNone(self._unchanged_guard(plan, "ai"))
        self.assertIsNone(self._unchanged_guard(plan, "ad"))

    def test_unchanged_update_guard_absent_when_result_may_depend_on_physical_row_order(self) -> None:
        # An UPDATE that changes no referenced column still rewrites rows to new
        # physical positions, so views whose original query may observe scan
        # order must be maintained on every UPDATE (group-scoped and full
        # recompute, order-sensitive aggregates), while order-independent
        # algebraic views keep the early exit.
        schema = "CREATE TABLE t (id INT PRIMARY KEY, g INT, v INT, s TEXT, x INT);"
        guarded = ["CREATE VIEW v AS SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM t GROUP BY g;",
                   "CREATE VIEW v AS SELECT id, g, v FROM t WHERE v > 0;"]
        unguarded = {
            "unordered STRING_AGG": "CREATE VIEW v AS SELECT g, STRING_AGG(s, ',') AS ss FROM t GROUP BY g;",
            "unordered ARRAY_AGG": "CREATE VIEW v AS SELECT g, ARRAY_AGG(v) AS vs FROM t GROUP BY g;",
            "window ties": "CREATE VIEW v AS SELECT id, g, v, ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) AS rn FROM t;",
            "LIMIT ties": "CREATE VIEW v AS SELECT id, v FROM t ORDER BY v LIMIT 5;",
        }
        for view in guarded:
            self.assertIsNotNone(self._unchanged_guard(self._plan(view, schema)), view)
        for name, view in unguarded.items():
            plan = self._plan(view, schema)
            self.assertIsNone(self._unchanged_guard(plan), name)
            self.assertNotIn(MaintenanceStrategy.ALGEBRAIC_DELTA, (plan.strategy,), name)

    def test_unchanged_update_guard_only_where_it_was_measured_to_pay_off(self) -> None:
        # Cheap O(groups) maintenance bodies cost about as much as the guard
        # (measured, see emitters._PG_GUARD_PAYS_OFF_MODES): no guard there.
        schema = "CREATE TABLE t (id INT PRIMARY KEY, g INT, v INT, s TEXT, x INT); CREATE TABLE d (g INT PRIMARY KEY, name TEXT);"
        for view in (
            "CREATE VIEW v AS SELECT g, SUM(v) AS sv, COUNT(*) AS c FROM t GROUP BY g;",
            "CREATE VIEW v AS SELECT g, AVG(v) AS a FROM t GROUP BY g;",
            "CREATE VIEW v AS SELECT d.name, SUM(t.v) AS sv FROM t JOIN d ON t.g = d.g GROUP BY d.name;",
            "CREATE VIEW v AS SELECT DISTINCT g FROM t;",
            "CREATE VIEW v AS SELECT g, SUM(v) AS sv FROM (SELECT g, v FROM t UNION ALL SELECT g, v FROM t) u GROUP BY g;",
        ):
            plan = self._plan(view, schema)
            self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA, view)
            self.assertIsNone(self._unchanged_guard(plan), view)
        for view in (
            "CREATE VIEW v AS SELECT id, g, v FROM t WHERE v > 0;",
            "CREATE VIEW v AS SELECT g, MIN(v) AS lo FROM t GROUP BY g;",
            "CREATE VIEW v AS SELECT g FROM t UNION SELECT g FROM d;",
        ):
            self.assertIsNotNone(self._unchanged_guard(self._plan(view, schema), "au"), view)

    def test_unchanged_update_guard_needs_identity_types_where_equal_values_can_differ(self) -> None:
        # MIN/MAX and set operations return ONE of several values that compare
        # equal (numeric 1.0 vs 1.00, collated text); which one depends on
        # physical row order, which every UPDATE changes: guard only when every
        # column the view reads is an identity type.
        schema = (
            "CREATE TABLE t (id INT PRIMARY KEY, g INT, x NUMERIC, s TEXT, n NUMERIC(10, 2), ts TIMESTAMPTZ);"
            "CREATE TABLE d (id INT PRIMARY KEY, x NUMERIC, g INT);"
        )
        no_guard = [
            "CREATE VIEW v AS SELECT g, MIN(x) AS lo, MAX(x) AS hi FROM t GROUP BY g;",
            "CREATE VIEW v AS SELECT s, MIN(g) AS lo FROM t GROUP BY s;",  # text group key
            "CREATE VIEW v AS SELECT g, MAX(s) AS hi FROM t GROUP BY g;",
            "CREATE VIEW v AS SELECT x FROM t UNION SELECT x FROM d;",
            "CREATE VIEW v AS SELECT x FROM t INTERSECT SELECT x FROM d;",
            "CREATE VIEW v AS SELECT id, ctid::text AS c FROM t;",  # changes on every UPDATE
        ]
        for view in no_guard:
            self.assertIsNone(self._unchanged_guard(self._plan(view, schema)), view)
        # Projections keep every row's own value: numeric/text columns are fine.
        self.assertIsNotNone(self._unchanged_guard(self._plan("CREATE VIEW v AS SELECT id, x, s FROM t;", schema)))
        # Identity-typed MIN/MAX and set operations keep the guard.
        for view in (
            "CREATE VIEW v AS SELECT g, MIN(id) AS lo, MAX(ts) AS hi FROM t GROUP BY g;",
            "CREATE VIEW v AS SELECT g FROM t UNION SELECT g FROM d;",
        ):
            self.assertIsNotNone(self._unchanged_guard(self._plan(view, schema)), view)
        # MIN/MAX under a non-algebraic strategy (HAVING with an aggregate subquery
        # is group-scoped recompute) never gets the guard.
        plan = self._plan(
            "CREATE VIEW v AS SELECT g, MIN(id) AS lo FROM t GROUP BY g HAVING MIN(id) > (SELECT AVG(id) FROM t);",
            schema,
        )
        self.assertNotEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assertIsNone(self._unchanged_guard(plan))

    def test_unchanged_update_guard_is_fail_closed(self) -> None:
        # No primary key, a float column, or a non-built-in type: no guard.
        no_pk = "CREATE TABLE s (region VARCHAR(32), amount NUMERIC(10, 2) NOT NULL);"
        self.assertIsNone(self._unchanged_guard(self._plan("CREATE VIEW v AS SELECT region, amount FROM s;", no_pk)))
        flt = "CREATE TABLE s (id INT PRIMARY KEY, amount DOUBLE PRECISION NOT NULL);"
        self.assertIsNone(self._unchanged_guard(self._plan("CREATE VIEW v AS SELECT id, amount FROM s;", flt)))
        arr = "CREATE TABLE s (id INT PRIMARY KEY, tags TEXT[] NOT NULL);"
        self.assertIsNone(self._unchanged_guard(self._plan("CREATE VIEW v AS SELECT id, tags FROM s;", arr)))
        # Composite primary keys pair on every key column.
        comp = "CREATE TABLE s (a INT, b INT, amount NUMERIC(10, 2) NOT NULL, PRIMARY KEY (a, b));"
        guard = self._unchanged_guard(self._plan("CREATE VIEW v AS SELECT a, b, amount FROM s;", comp))
        assert guard is not None
        self.assertIn("__mv_old.a = __mv_new.a AND __mv_old.b = __mv_new.b", guard)

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
                body = m.group(1)
                # Strip DECLARE block and Stage 3 PL/pgSQL control-flow scaffolding;
                # parse every embedded SQL statement (including staging INSERTs and
                # reconcile inclusion-exclusion upserts/deletes) as PostgreSQL.
                m_begin = re.search(r"\bBEGIN\b", body)
                if m_begin:
                    body = body[m_begin.end() :]
                # Extract any embedded Stage 3 staging INSERT statements before stripping
                # PL/pgSQL bracket control flow, so staging DML is also parsed by sqlglot.
                stage_blocks = re.findall(
                    r"/\*\s*__MV_STAGE_START__\s*\*/(.*?)/\*\s*__MV_STAGE_END__\s*\*/",
                    body,
                    flags=re.S,
                )
                body = re.sub(r"/\*\s*__MV_BRACKET_START__\s*\*/.*?/\*\s*__MV_BRACKET_END__\s*\*/", "", body, flags=re.S)
                body = re.sub(r"/\*\s*__MV_REPAIR_FENCE_START__\s*\*/.*?/\*\s*__MV_REPAIR_FENCE_END__\s*\*/", "", body, flags=re.S)
                body = "\n".join(stage_blocks) + "\n" + body
                body = re.sub(r"(?s)IF NOT EXISTS .*?END IF;", "", body)
                body = re.sub(r"(?s)IF NOT \(.*?END IF;", "", body)
                body = re.sub(r"(?s)IF pg_catalog\.txid_current_if_assigned\(\) IS NULL THEN.*?END IF;", "", body)
                body = re.sub(r"(?s)(?:RETURN(?:\s+NULL)?;\s*)?\bEND;\s*$", "", body)
                for stmt in _split_top_level(body, ";"):
                    st = stmt.strip()
                    st_up = st.upper()
                    if not st or st_up in ("BEGIN", "END", "ELSE", "END IF", "RETURN", "RETURN NULL"):
                        continue
                    if re.match(r"^(?:IF\b|ELSIF\b|PERFORM\b|__MV_\w+\s*:=)", st_up):
                        continue
                    try:
                        sqlglot.parse(st, read="postgres")
                    except Exception as exc:  # pragma: no cover - failure path
                        self.fail(f"sqlglot cannot parse {st!r}: {exc}")

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
