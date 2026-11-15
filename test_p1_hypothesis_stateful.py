#!/usr/bin/env python3
"""
Stateful Property-Based Fuzzing & Automatic Counterexample Shrinking for Materialized Views.

This suite complements deterministic unit testing and random AST fuzzing with
Hypothesis RuleBasedStateMachine exploration (`hypothesis.stateful`).

Core capabilities:
1. MatviewStatefulMachine:
   - Sequences of single and batched INSERT, UPDATE (including PK-changing updates,
     NULL <-> non-NULL transitions, no-op updates), DELETE, bulk_changes(),
     sync_views() / force_rebuild, and transactions with ROLLBACK vs COMMIT.
   - Diverse views: filtered projections (Z-set), GROUP BY aggregates (COUNT, SUM, AVG,
     MIN, MAX, bitwise aggregates BIT_OR/BIT_AND/BIT_XOR, COUNT(DISTINCT ...)), and
     nullable grouping keys.
   - Explicit invariants checked on EVERY step: exact multiset/value equivalence against
     the live SELECT oracle, distinct aux refcount consistency (__matview_dist_* > 0
     matching live base table group counts), and physical catalog/trigger/view consistency.

2. MultiTableJoinStatefulMachine:
   - Multi-table schema covering INNER JOIN aggregate views, UNION ALL aggregate views,
     foreign key re-parenting, orphaning, and cross-table DML.
   - Full live SELECT equivalence and catalog consistency on every step.

3. Automatic Counterexample Shrinking:
   - Unit tests verifying that when a multi-step DML sequence triggers a fault,
     Hypothesis automatically minimizes the sequence to a minimal reproducible
     counterexample (<= 3-4 steps).
"""

from __future__ import annotations

import random
import sqlite3
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

import hypothesis.stateful as hs
from hypothesis import HealthCheck, settings, strategies as st

import matview_sync as mv
from test_matview_sync import (
    Dialect,
    _register_sqlite_compat_udfs,
    _rows_equal,
    _verify_external_db_mv,
)

# Shared deterministic settings for stateful fuzzing
_STATEFUL_SETTINGS = settings(
    max_examples=12,
    stateful_step_count=10,
    derandomize=True,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# =============================================================================
# In-Memory Template Connections (Fast 0.1ms Backup Clones)
# =============================================================================

_MATVIEW_TEMPLATE: Optional[sqlite3.Connection] = None
_JOIN_TEMPLATE: Optional[sqlite3.Connection] = None
_SHRINK_TEMPLATE: Optional[sqlite3.Connection] = None


def _get_matview_template() -> sqlite3.Connection:
    global _MATVIEW_TEMPLATE
    if _MATVIEW_TEMPLATE is None:
        conn = sqlite3.connect(":memory:")
        _register_sqlite_compat_udfs(conn, Dialect.SQLITE)
        mgr = mv.MaterializedViewManager(conn)
        mgr.apply_schema_ddl("""
            CREATE TABLE items (
                id INTEGER PRIMARY KEY,
                category TEXT,
                status TEXT,
                val_int INTEGER,
                weight INTEGER NOT NULL,
                score REAL
            );
        """)
        # 1. Filtered projection (Z-set)
        v1 = "CREATE VIEW v_filtered_proj AS SELECT id, category, val_int, weight FROM items WHERE weight >= 10 AND status = 'active';"
        # 2. GROUP BY aggregate view with nullable group key, COUNT(*), SUM, AVG, MIN, MAX, bitwise, COUNT(DISTINCT)
        v2 = (
            "CREATE VIEW v_grp_aggs AS SELECT category, COUNT(*) AS total_cnt, "
            "COUNT(val_int) AS not_null_cnt, SUM(val_int) AS sum_val, AVG(val_int) AS avg_val, "
            "MIN(val_int) AS min_val, MAX(val_int) AS max_val, BIT_OR(weight) AS bor_w, "
            "BIT_AND(weight) AS band_w, BIT_XOR(weight) AS bxor_w, COUNT(DISTINCT val_int) AS dist_val_cnt "
            "FROM items GROUP BY category;"
        )
        # 3. Ungrouped global aggregate view
        v3 = "CREATE VIEW v_global_agg AS SELECT COUNT(*) AS cnt, SUM(weight) AS sum_w, AVG(score) AS avg_score, MIN(weight) AS min_w, MAX(weight) AS max_w FROM items;"
        mgr.register_view(v1)
        mgr.register_view(v2)
        mgr.register_view(v3)
        _MATVIEW_TEMPLATE = conn
    return _MATVIEW_TEMPLATE


def _get_join_template() -> sqlite3.Connection:
    global _JOIN_TEMPLATE
    if _JOIN_TEMPLATE is None:
        conn = sqlite3.connect(":memory:")
        _register_sqlite_compat_udfs(conn, Dialect.SQLITE)
        mgr = mv.MaterializedViewManager(conn)
        mgr.apply_schema_ddl("""
            CREATE TABLE departments (
                dept_id INTEGER PRIMARY KEY,
                dept_name TEXT NOT NULL,
                active INTEGER NOT NULL
            );
            CREATE TABLE employees (
                emp_id INTEGER PRIMARY KEY,
                dept_id INTEGER,
                emp_name TEXT NOT NULL,
                salary INTEGER NOT NULL,
                role TEXT
            );
            CREATE TABLE sales_online (
                id INTEGER PRIMARY KEY,
                region TEXT,
                amount INTEGER
            );
            CREATE TABLE sales_store (
                id INTEGER PRIMARY KEY,
                region TEXT,
                amount INTEGER
            );
        """)
        # 1. INNER JOIN aggregate view
        v_join = (
            "CREATE VIEW v_dept_summary AS "
            "SELECT d.dept_name, COUNT(*) AS emp_cnt, SUM(e.salary) AS total_sal, "
            "AVG(e.salary) AS avg_sal, MIN(e.salary) AS min_sal, MAX(e.salary) AS max_sal "
            "FROM employees e INNER JOIN departments d ON e.dept_id = d.dept_id "
            "GROUP BY d.dept_name;"
        )
        # 2. UNION ALL aggregate view
        v_union = (
            "CREATE VIEW v_regional_sales AS "
            "SELECT region, COUNT(*) AS txn_cnt, SUM(amount) AS total_sales, AVG(amount) AS avg_sales "
            "FROM ("
            "    SELECT region, amount FROM sales_online "
            "    UNION ALL "
            "    SELECT region, amount FROM sales_store"
            ") u GROUP BY region;"
        )
        mgr.register_view(v_join)
        mgr.register_view(v_union)
        _JOIN_TEMPLATE = conn
    return _JOIN_TEMPLATE


def _get_shrink_template() -> sqlite3.Connection:
    global _SHRINK_TEMPLATE
    if _SHRINK_TEMPLATE is None:
        conn = sqlite3.connect(":memory:")
        _register_sqlite_compat_udfs(conn, Dialect.SQLITE)
        mgr = mv.MaterializedViewManager(conn)
        mgr.apply_schema_ddl("CREATE TABLE shrink_items (id INT PRIMARY KEY, cat TEXT, val INT);")
        mgr.register_view("CREATE VIEW v_shrink AS SELECT cat, SUM(val) AS s, COUNT(*) AS c FROM shrink_items GROUP BY cat;")
        _SHRINK_TEMPLATE = conn
    return _SHRINK_TEMPLATE


# =============================================================================
# State Machine 1: MatviewStatefulMachine
# =============================================================================

class MatviewStatefulMachine(hs.RuleBasedStateMachine):
    """
    Hypothesis RuleBasedStateMachine exercising diverse DML mutations, views,
    and invariants on Materialized Views.
    """

    item_ids = hs.Bundle("item_ids")

    def __init__(self) -> None:
        super().__init__()
        self.conn = sqlite3.connect(":memory:")
        _get_matview_template().backup(self.conn)
        _register_sqlite_compat_udfs(self.conn, Dialect.SQLITE)
        self.mgr = mv.MaterializedViewManager(self.conn)
        self.cur = self.conn.cursor()
        self.catalog = self.mgr.get_catalog()
        self.views = [
            (
                "v_filtered_proj",
                "SELECT id, category, val_int, weight FROM items WHERE weight >= 10 AND status = 'active';",
            ),
            (
                "v_grp_aggs",
                (
                    "SELECT category, COUNT(*) AS total_cnt, COUNT(val_int) AS not_null_cnt, "
                    "SUM(val_int) AS sum_val, AVG(val_int) AS avg_val, MIN(val_int) AS min_val, "
                    "MAX(val_int) AS max_val, BIT_OR(weight) AS bor_w, BIT_AND(weight) AS band_w, "
                    "BIT_XOR(weight) AS bxor_w, COUNT(DISTINCT val_int) AS dist_val_cnt "
                    "FROM items GROUP BY category;"
                ),
            ),
            (
                "v_global_agg",
                "SELECT COUNT(*) AS cnt, SUM(weight) AS sum_w, AVG(score) AS avg_score, MIN(weight) AS min_w, MAX(weight) AS max_w FROM items;",
            ),
        ]
        self._next_id = 100

    @hs.initialize(target=item_ids)
    def init_seed(self) -> Any:
        self.cur.execute(
            "INSERT INTO items VALUES "
            "(1, 'A', 'active', 10, 12, 1.5), "
            "(2, 'B', 'inactive', NULL, 8, 2.0), "
            "(3, NULL, 'active', 25, 16, 3.5);"
        )
        self.conn.commit()
        return hs.multiple(1, 2, 3)

    # --- INSERT Rules ---

    @hs.rule(
        target=item_ids,
        cat=st.sampled_from(["A", "B", "C", None]),
        stat=st.sampled_from(["active", "inactive", "pending"]),
        val=st.sampled_from([None, 0, 5, 20, 50, -5]),
        w=st.integers(1, 30),
        sc=st.floats(0.0, 100.0, allow_nan=False),
    )
    def insert_single(
        self, cat: Optional[str], stat: str, val: Optional[int], w: int, sc: float
    ) -> int:
        self._next_id += 1
        new_id = self._next_id
        self.cur.execute(
            "INSERT INTO items VALUES (?, ?, ?, ?, ?, ?);",
            (new_id, cat, stat, val, w, sc),
        )
        self.conn.commit()
        return new_id

    @hs.rule(target=item_ids, count=st.integers(2, 3))
    def insert_batch(self, count: int) -> Any:
        new_ids = []
        for _ in range(count):
            self._next_id += 1
            new_ids.append(self._next_id)
            self.cur.execute(
                "INSERT INTO items VALUES (?, 'A', 'active', 15, 14, 2.5);",
                (self._next_id,),
            )
        self.conn.commit()
        return hs.multiple(*new_ids)

    # --- UPDATE Rules ---

    @hs.rule(
        i_id=item_ids,
        val=st.sampled_from([None, 0, 12, 99]),
        w=st.integers(1, 30),
        sc=st.floats(0.0, 50.0, allow_nan=False),
    )
    def update_value(self, i_id: int, val: Optional[int], w: int, sc: float) -> None:
        # Exercises NULL <-> non-NULL value transitions
        self.cur.execute(
            "UPDATE items SET val_int = ?, weight = ?, score = ? WHERE id = ?;",
            (val, w, sc, i_id),
        )
        self.conn.commit()

    @hs.rule(
        i_id=item_ids,
        cat=st.sampled_from(["A", "B", "C", None]),
        stat=st.sampled_from(["active", "inactive"]),
    )
    def update_group_key(self, i_id: int, cat: Optional[str], stat: str) -> None:
        # Exercises NULL <-> non-NULL grouping key transitions
        self.cur.execute(
            "UPDATE items SET category = ?, status = ? WHERE id = ?;",
            (cat, stat, i_id),
        )
        self.conn.commit()

    @hs.rule(target=item_ids, i_id=item_ids)
    def update_pk(self, i_id: int) -> int:
        # Exercises Primary Key swap / update
        self._next_id += 1
        new_id = self._next_id
        self.cur.execute("UPDATE items SET id = ? WHERE id = ?;", (new_id, i_id))
        self.conn.commit()
        return new_id

    @hs.rule(i_id=item_ids)
    def update_noop(self, i_id: int) -> None:
        # Exercises no-op updates where OLD == NEW
        self.cur.execute("UPDATE items SET weight = weight WHERE id = ?;", (i_id,))
        self.conn.commit()

    # --- DELETE Rules ---

    @hs.rule(i_id=item_ids)
    def delete_single(self, i_id: int) -> None:
        self.cur.execute("DELETE FROM items WHERE id = ?;", (i_id,))
        self.conn.commit()

    @hs.rule(cat=st.sampled_from(["A", "B", "C", None]))
    def delete_batch(self, cat: Optional[str]) -> None:
        if cat is None:
            self.cur.execute("DELETE FROM items WHERE category IS NULL;")
        else:
            self.cur.execute("DELETE FROM items WHERE category = ?;", (cat,))
        self.conn.commit()

    # --- bulk_changes & sync_views / force_rebuild Rules ---

    @hs.rule(use_shadow_swap=st.booleans())
    def bulk_changes_op(self, use_shadow_swap: bool) -> None:
        with self.mgr.bulk_changes(use_shadow_swap=use_shadow_swap):
            self._next_id += 1
            self.cur.execute(
                "INSERT INTO items VALUES (?, 'C', 'active', 30, 20, 10.0);",
                (self._next_id,),
            )
        self.conn.commit()

    @hs.rule()
    def rebuild_views(self) -> None:
        self.mgr.repair_view()
        self.catalog = self.mgr.get_catalog()

    # --- Transaction ROLLBACK vs COMMIT Rules ---

    @hs.rule(commit=st.booleans())
    def transaction_op(self, commit: bool) -> None:
        self.cur.execute("SAVEPOINT sp_stateful_tx;")
        self._next_id += 1
        self.cur.execute(
            "INSERT INTO items VALUES (?, 'TX', 'pending', 50, 16, 12.0);",
            (self._next_id,),
        )
        self.cur.execute("UPDATE items SET weight = weight + 1 WHERE status = 'active';")
        if commit:
            self.cur.execute("RELEASE sp_stateful_tx;")
            self.conn.commit()
        else:
            self.cur.execute("ROLLBACK TO sp_stateful_tx;")
            self.cur.execute("RELEASE sp_stateful_tx;")

    # --- INVARIANTS (Checked on EVERY Step) ---

    @hs.invariant()
    def check_all_invariants(self) -> None:
        # 1. Exact multiset and value equivalence against live SELECT oracle
        for v_name, live_sql in self.views:
            _verify_external_db_mv(
                self.cur, v_name, live_sql, f"MatviewStatefulMachine:{v_name}"
            )

        # 2. Distinct aux refcount consistency (__matview_dist_* > 0 matching live base table)
        aux_table = "__matview_dist_v_grp_aggs__dist_val_cnt"
        self.cur.execute(f"SELECT COUNT(*) FROM {aux_table} WHERE __ref_cnt <= 0;")
        zero_ref_cnt = self.cur.fetchone()[0]
        assert zero_ref_cnt == 0, f"Found {zero_ref_cnt} rows with __ref_cnt <= 0 in {aux_table}"

        self.cur.execute(f"SELECT __mv_pk, __val, __ref_cnt FROM {aux_table};")
        aux_rows = self.cur.fetchall()
        self.cur.execute(
            "SELECT quote(category), val_int, COUNT(*) FROM items "
            "WHERE val_int IS NOT NULL GROUP BY quote(category), val_int;"
        )
        expected_aux_rows = self.cur.fetchall()
        assert _rows_equal(aux_rows, expected_aux_rows), (
            f"Aux refcount mismatch on {aux_table}!\n"
            f"  Aux:      {aux_rows}\n"
            f"  Expected: {expected_aux_rows}"
        )

        # 3. Physical catalog, trigger, and view consistency in sqlite_master
        self.cur.execute("SELECT type, name FROM sqlite_master;")
        objects = set(self.cur.fetchall())
        for v_name, info in self.catalog.items():
            assert ("view", v_name) in objects, f"Missing reader view {v_name} in sqlite_master"
            assert (
                "table",
                info["backing_table"],
            ) in objects, f"Missing backing table {info['backing_table']} in sqlite_master"
            for trg in info["trigger_names"]:
                assert ("trigger", trg) in objects, f"Missing trigger {trg} in sqlite_master"
            for aux in info["aux_tables"]:
                assert ("table", aux) in objects, f"Missing aux table {aux} in sqlite_master"

    def teardown(self) -> None:
        self.conn.close()


TestMatviewStatefulMachine = MatviewStatefulMachine.TestCase
TestMatviewStatefulMachine.settings = _STATEFUL_SETTINGS


# =============================================================================
# State Machine 2: MultiTableJoinStatefulMachine
# =============================================================================

class MultiTableJoinStatefulMachine(hs.RuleBasedStateMachine):
    """
    Hypothesis RuleBasedStateMachine exercising multi-table schemas:
    - INNER JOIN aggregate views
    - UNION ALL aggregate views
    - FK updates, re-parenting, orphaning, and cross-table DML.
    """

    dept_ids = hs.Bundle("dept_ids")
    emp_ids = hs.Bundle("emp_ids")
    online_ids = hs.Bundle("online_ids")
    store_ids = hs.Bundle("store_ids")

    def __init__(self) -> None:
        super().__init__()
        self.conn = sqlite3.connect(":memory:")
        _get_join_template().backup(self.conn)
        _register_sqlite_compat_udfs(self.conn, Dialect.SQLITE)
        self.mgr = mv.MaterializedViewManager(self.conn)
        self.cur = self.conn.cursor()
        self.catalog = self.mgr.get_catalog()
        self.views = [
            (
                "v_dept_summary",
                (
                    "SELECT d.dept_name, COUNT(*) AS emp_cnt, SUM(e.salary) AS total_sal, "
                    "AVG(e.salary) AS avg_sal, MIN(e.salary) AS min_sal, MAX(e.salary) AS max_sal "
                    "FROM employees e INNER JOIN departments d ON e.dept_id = d.dept_id "
                    "GROUP BY d.dept_name;"
                ),
            ),
            (
                "v_regional_sales",
                (
                    "SELECT region, COUNT(*) AS txn_cnt, SUM(amount) AS total_sales, AVG(amount) AS avg_sales "
                    "FROM ("
                    "    SELECT region, amount FROM sales_online "
                    "    UNION ALL "
                    "    SELECT region, amount FROM sales_store"
                    ") u GROUP BY region;"
                ),
            ),
        ]
        self._next_id = 500

    @hs.initialize(target=dept_ids)
    def init_data(self) -> Any:
        self.cur.execute(
            "INSERT INTO departments VALUES "
            "(1, 'Engineering', 1), (2, 'Sales', 1), (3, 'HR', 0);"
        )
        self.cur.execute(
            "INSERT INTO employees VALUES "
            "(10, 1, 'Alice', 120, 'lead'), (20, 2, 'Bob', 80, 'rep');"
        )
        self.cur.execute(
            "INSERT INTO sales_online VALUES (1, 'North', 100), (2, 'South', 200);"
        )
        self.cur.execute(
            "INSERT INTO sales_store VALUES (1, 'North', 50), (2, 'East', 300);"
        )
        self.conn.commit()
        return hs.multiple(1, 2, 3)

    # --- Department & Employee Rules ---

    @hs.rule(
        target=emp_ids,
        d_id=dept_ids,
        sal=st.integers(30, 200),
        role=st.sampled_from(["eng", "sales", "support"]),
    )
    def insert_emp(self, d_id: int, sal: int, role: str) -> int:
        self._next_id += 1
        e_id = self._next_id
        self.cur.execute(
            "INSERT INTO employees VALUES (?, ?, 'Emp', ?, ?);",
            (e_id, d_id, sal, role),
        )
        self.conn.commit()
        return e_id

    @hs.rule(e_id=emp_ids, d_id=st.sampled_from([1, 2, 3, None, 999]))
    def reparent_emp_dept(self, e_id: int, d_id: Optional[int]) -> None:
        # Re-parent employee to another department, or orphan with NULL/unknown dept_id
        self.cur.execute("UPDATE employees SET dept_id = ? WHERE emp_id = ?;", (d_id, e_id))
        self.conn.commit()

    @hs.rule(e_id=emp_ids, new_sal=st.integers(40, 250))
    def update_emp_salary(self, e_id: int, new_sal: int) -> None:
        self.cur.execute("UPDATE employees SET salary = ? WHERE emp_id = ?;", (new_sal, e_id))
        self.conn.commit()

    @hs.rule(e_id=emp_ids)
    def delete_emp(self, e_id: int) -> None:
        self.cur.execute("DELETE FROM employees WHERE emp_id = ?;", (e_id,))
        self.conn.commit()

    @hs.rule(d_id=dept_ids, new_name=st.sampled_from(["Engineering", "Research", "Marketing", "Finance"]))
    def update_dept_name(self, d_id: int, new_name: str) -> None:
        self.cur.execute("UPDATE departments SET dept_name = ? WHERE dept_id = ?;", (new_name, d_id))
        self.conn.commit()

    # --- UNION ALL Sales Rules ---

    @hs.rule(
        target=online_ids,
        reg=st.sampled_from(["North", "South", "East", "West", None]),
        amt=st.integers(10, 500),
    )
    def insert_online_sale(self, reg: Optional[str], amt: int) -> int:
        self._next_id += 1
        s_id = self._next_id
        self.cur.execute("INSERT INTO sales_online VALUES (?, ?, ?);", (s_id, reg, amt))
        self.conn.commit()
        return s_id

    @hs.rule(
        target=store_ids,
        reg=st.sampled_from(["North", "South", "East", "West", None]),
        amt=st.integers(10, 500),
    )
    def insert_store_sale(self, reg: Optional[str], amt: int) -> int:
        self._next_id += 1
        s_id = self._next_id
        self.cur.execute("INSERT INTO sales_store VALUES (?, ?, ?);", (s_id, reg, amt))
        self.conn.commit()
        return s_id

    @hs.rule(reg=st.sampled_from(["North", "South", "East", "West"]))
    def delete_store_sales_by_region(self, reg: str) -> None:
        self.cur.execute("DELETE FROM sales_store WHERE region = ?;", (reg,))
        self.conn.commit()

    @hs.rule(commit=st.booleans())
    def tx_sales_op(self, commit: bool) -> None:
        self.cur.execute("SAVEPOINT sp_multi_tx;")
        self._next_id += 1
        self.cur.execute("INSERT INTO sales_online VALUES (?, 'West', 999);", (self._next_id,))
        if commit:
            self.cur.execute("RELEASE sp_multi_tx;")
            self.conn.commit()
        else:
            self.cur.execute("ROLLBACK TO sp_multi_tx;")
            self.cur.execute("RELEASE sp_multi_tx;")

    # --- INVARIANTS ---

    @hs.invariant()
    def check_join_invariants(self) -> None:
        for v_name, live_sql in self.views:
            _verify_external_db_mv(
                self.cur, v_name, live_sql, f"MultiTableJoinStatefulMachine:{v_name}"
            )

        self.cur.execute("SELECT type, name FROM sqlite_master;")
        objects = set(self.cur.fetchall())
        for v_name, info in self.catalog.items():
            assert ("view", v_name) in objects, f"Missing view {v_name} in sqlite_master"
            assert (
                "table",
                info["backing_table"],
            ) in objects, f"Missing backing table {info['backing_table']} in sqlite_master"

    def teardown(self) -> None:
        self.conn.close()


TestMultiTableJoinStatefulMachine = MultiTableJoinStatefulMachine.TestCase
TestMultiTableJoinStatefulMachine.settings = _STATEFUL_SETTINGS


# =============================================================================
# Shrinking Verification (Proves Automatic Minimal Counterexample Reduction)
# =============================================================================

class ShrinkCountSumFaultMachine(hs.RuleBasedStateMachine):
    """
    Intentionally faulty state machine designed to prove that Hypothesis shrinks
    a multi-step DML failure sequence to a minimal reproducible counterexample (<= 2-3 steps).
    """

    ids = hs.Bundle("ids")
    execution_history: List[List[Tuple[Any, ...]]] = []

    def __init__(self) -> None:
        super().__init__()
        self.conn = sqlite3.connect(":memory:")
        _get_shrink_template().backup(self.conn)
        _register_sqlite_compat_udfs(self.conn, Dialect.SQLITE)
        self.recorded_steps: List[Tuple[Any, ...]] = []
        self._next_id = 0

    @hs.rule(
        target=ids,
        cat=st.sampled_from(["A", "B", "FAULT"]),
        val=st.integers(1, 40),
    )
    def insert(self, cat: str, val: int) -> int:
        self._next_id += 1
        new_id = self._next_id
        self.recorded_steps.append(("insert", new_id, cat, val))
        self.conn.execute("INSERT INTO shrink_items VALUES (?, ?, ?);", (new_id, cat, val))
        self.conn.commit()
        return new_id

    @hs.rule(i_id=ids, new_val=st.integers(30, 60))
    def update(self, i_id: int, new_val: int) -> None:
        self.recorded_steps.append(("update", i_id, new_val))
        self.conn.execute("UPDATE shrink_items SET val = ? WHERE id = ?;", (new_val, i_id))
        self.conn.commit()

    @hs.rule(i_id=ids)
    def delete(self, i_id: int) -> None:
        self.recorded_steps.append(("delete", i_id))
        self.conn.execute("DELETE FROM shrink_items WHERE id = ?;", (i_id,))
        self.conn.commit()

    @hs.invariant()
    def assert_injected_fault(self) -> None:
        rows = self.conn.execute(
            "SELECT cat, s, c FROM v_shrink WHERE cat = 'FAULT';"
        ).fetchall()
        # Injected failure boundary: requires at least 2 rows and sum >= 50
        if rows and rows[0][1] >= 50 and rows[0][2] >= 2:
            self.execution_history.append(list(self.recorded_steps))
            raise AssertionError(
                f"Injected fault: sum={rows[0][1]}, count={rows[0][2]} "
                f"after {len(self.recorded_steps)} steps"
            )

    def teardown(self) -> None:
        self.conn.close()


class ShrinkInsertUpdateFaultMachine(hs.RuleBasedStateMachine):
    """
    Intentionally faulty state machine testing shrinking of an insert-followed-by-update
    sequence to exactly 2 steps.
    """

    ids = hs.Bundle("ids")
    execution_history: List[List[Tuple[Any, ...]]] = []

    def __init__(self) -> None:
        super().__init__()
        self.conn = sqlite3.connect(":memory:")
        _get_shrink_template().backup(self.conn)
        _register_sqlite_compat_udfs(self.conn, Dialect.SQLITE)
        self.recorded_steps: List[Tuple[Any, ...]] = []
        self._next_id = 0
        self.has_updated = False

    @hs.rule(
        target=ids,
        cat=st.sampled_from(["A", "B", "FAULT"]),
        val=st.integers(1, 20),
    )
    def insert(self, cat: str, val: int) -> int:
        self._next_id += 1
        new_id = self._next_id
        self.recorded_steps.append(("insert", new_id, cat, val))
        self.conn.execute("INSERT INTO shrink_items VALUES (?, ?, ?);", (new_id, cat, val))
        self.conn.commit()
        return new_id

    @hs.rule(i_id=ids, new_val=st.integers(50, 100))
    def update(self, i_id: int, new_val: int) -> None:
        self.has_updated = True
        self.recorded_steps.append(("update", i_id, new_val))
        self.conn.execute("UPDATE shrink_items SET val = ? WHERE id = ?;", (new_val, i_id))
        self.conn.commit()

    @hs.rule(i_id=ids)
    def delete(self, i_id: int) -> None:
        self.recorded_steps.append(("delete", i_id))
        self.conn.execute("DELETE FROM shrink_items WHERE id = ?;", (i_id,))
        self.conn.commit()

    @hs.invariant()
    def assert_injected_fault(self) -> None:
        if self.has_updated:
            rows = self.conn.execute(
                "SELECT cat, s FROM v_shrink WHERE cat = 'FAULT';"
            ).fetchall()
            if rows and rows[0][1] >= 50:
                self.execution_history.append(list(self.recorded_steps))
                raise AssertionError(f"Injected fault: sum={rows[0][1]}")

    def teardown(self) -> None:
        self.conn.close()


class TestHypothesisAutomaticShrinking(unittest.TestCase):
    """
    Unit tests proving Hypothesis's automatic counterexample shrinking works
    on intentionally injected multi-step DML faults.
    """

    def test_multi_insert_failure_shrinks_to_minimal_sequence(self) -> None:
        ShrinkCountSumFaultMachine.execution_history.clear()
        shrink_settings = settings(
            max_examples=30,
            stateful_step_count=15,
            derandomize=True,
            deadline=None,
            database=None,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        )

        with self.assertRaises(AssertionError) as ctx:
            hs.run_state_machine_as_test(
                ShrinkCountSumFaultMachine,
                settings=shrink_settings,
            )

        self.assertIn("Injected fault", str(ctx.exception))
        history = ShrinkCountSumFaultMachine.execution_history
        self.assertTrue(len(history) >= 2, "Hypothesis must have executed multiple shrinking steps")
        initial_steps = len(history[0])
        final_steps = len(history[-1])

        # Initial random sequence took many steps to hit the fault; final shrunk sequence is <= 3
        self.assertGreaterEqual(initial_steps, 5, f"Initial sequence should have >= 5 steps, got {initial_steps}")
        self.assertLessEqual(final_steps, 3, f"Final shrunk sequence must be <= 3 steps, got {final_steps}: {history[-1]}")
        # Verify the shrunk sequence is minimal: exactly 2 inserts of category 'FAULT'
        fault_inserts = [step for step in history[-1] if step[0] == "insert" and step[2] == "FAULT"]
        self.assertEqual(len(fault_inserts), 2, f"Minimal counterexample requires exactly 2 inserts, got {history[-1]}")

    def test_insert_update_failure_shrinks_to_minimal_two_step_sequence(self) -> None:
        ShrinkInsertUpdateFaultMachine.execution_history.clear()
        shrink_settings = settings(
            max_examples=30,
            stateful_step_count=15,
            derandomize=True,
            deadline=None,
            database=None,
            suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        )

        with self.assertRaises(AssertionError) as ctx:
            hs.run_state_machine_as_test(
                ShrinkInsertUpdateFaultMachine,
                settings=shrink_settings,
            )

        self.assertIn("Injected fault", str(ctx.exception))
        history = ShrinkInsertUpdateFaultMachine.execution_history
        self.assertTrue(len(history) >= 2, "Hypothesis must have executed shrinking steps")
        final_steps = len(history[-1])

        # Exactly 2 steps: 1 insert and 1 update
        self.assertEqual(final_steps, 2, f"Final sequence must be shrunk to exactly 2 steps, got {final_steps}: {history[-1]}")
        step_types = [step[0] for step in history[-1]]
        self.assertEqual(step_types, ["insert", "update"])


# =============================================================================
# CLI / Direct Execution
# =============================================================================

if __name__ == "__main__":
    unittest.main()
