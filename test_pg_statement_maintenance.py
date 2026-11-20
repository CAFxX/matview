"""Native PostgreSQL contract tests for statement-level maintenance (bulk stage 2).

PostgreSQL maintains every view with statement-level triggers and transition
tables, its only mechanism (docs/design-bulk-maintenance.md, "PostgreSQL:
statement-level transition-table triggers, the only mechanism"). These tests
cover what the shared harness (``test_native_bulk_statements.py``) does not:

* isolation levels: a concurrent change of a locked group under REPEATABLE
  READ / SERIALIZABLE fails closed with a serialization error (40001), with
  independent connections and deliberate synchronization; disjoint groups
  commit; group-scoped and full-recompute views serialize on their lock row;
* partitions: sub-partitioned sources, TRUNCATE of one partition, a partition
  attached after registration (drift report, repair), and the refusal of
  sources whose statement triggers cannot see every write (a partition, an
  INHERITS parent or child);
* cost structure: one trigger invocation per statement and view; single-row
  statement cost that does not grow with the number of groups (the empty-group
  delete is restricted to the affected keys); a statement whose transition
  tables spill to disk.

The oracle is the original view query executed natively on the same
connection (``Env.check`` from the harness): exact multisets, Decimal scale
and driver column types, no tolerance.

Every test needs ``POSTGRES_DSN`` and is skipped without it.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import pytest

from test_native_bulk_statements import (
    SCHEMA,
    VIEWS,
    Env,
    Target,
    ViewDef,
    ViewMismatch,
    _row_lock_waiting,
    _session_id,
    _wait_for,
    _Worker,
)

PG_DSN = os.environ.get("POSTGRES_DSN")
pytestmark = pytest.mark.skipif(not PG_DSN, reason="needs POSTGRES_DSN (native PostgreSQL)")

SERIALIZATION_FAILURE = "40001"

VIEW_BY_NAME = {vd.name: vd for vd in VIEWS}
PG_SCHEMA = SCHEMA["postgres"]
# Group-scoped recompute (window partition) and full recompute (plain join).
V_WIN = ViewDef("v_win", "SELECT id, g, v, ROW_NUMBER() OVER (PARTITION BY g ORDER BY v, id) AS rn FROM t")
V_STRAGG = ViewDef("v_stragg", "SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids FROM t GROUP BY g")


@pytest.fixture
def make_env():
    envs: List[Env] = []

    def factory(**kw: Any) -> Env:
        assert PG_DSN is not None
        env = Env(Target("postgres", PG_DSN), **kw)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()


def _strategy(env: Env, name: str) -> str:
    return env.plans[name].strategy.value


def _pgcode(exc: Optional[BaseException]) -> Optional[str]:
    return getattr(exc, "pgcode", None)


# ---------------------------------------------------------------------------
# Isolation levels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isolation", ["REPEATABLE READ", "SERIALIZABLE"])
@pytest.mark.parametrize("view", ["v_mm", "v_agg", "v_j"])
def test_snapshot_isolation_conflict_on_a_locked_group_fails_closed(make_env, isolation: str, view: str) -> None:
    """B's snapshot predates A's committed change of group 1; B then changes
    group 1. Its maintenance must fail with a serialization error rather than
    fold or recompute from the stale snapshot; the views stay exact."""
    env = make_env(views=[VIEW_BY_NAME[view]], seed=True)
    b = env.connect()
    cur_b = b.cursor()
    cur_b.execute(f"BEGIN ISOLATION LEVEL {isolation}")
    cur_b.execute("SELECT COUNT(*) FROM t")  # takes B's snapshot
    cur_b.fetchall()
    env.execute("DELETE FROM t WHERE id = 1")  # group 1 (A, autocommitted below)
    env.commit()
    with pytest.raises(Exception) as excinfo:
        cur_b.execute("DELETE FROM t WHERE g = 1 AND id <> 1")
        b.commit()
    b.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)
    env.check(f"{isolation} conflict on group 1 rolled back")


@pytest.mark.parametrize("isolation", ["REPEATABLE READ", "SERIALIZABLE"])
def test_snapshot_isolation_conflict_after_lock_wait_fails_closed(make_env, isolation: str) -> None:
    """A holds group 1 (uncommitted MIN retraction); B (snapshot before A's
    commit) retracts another row of group 1 and waits for A's lock; A commits.
    B must fail with a serialization error, not recompute MIN from its
    snapshot."""
    env = make_env(views=[VIEW_BY_NAME["v_mm"], VIEW_BY_NAME["v_mmw"]], seed=False)
    env.execute("INSERT INTO t (id, g, v, w) VALUES (1, 1, 1, 1), (2, 1, 2, 2), (3, 1, 3, 3), (4, 2, 10, 10)")
    env.commit()
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("DELETE FROM t WHERE id = 1")
    worker = _Worker(b, [f"BEGIN ISOLATION LEVEL {isolation}", "SELECT 1", "DELETE FROM t WHERE id = 2"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's group lock")
    finally:
        a.commit()
        worker.join()
    assert _pgcode(worker.error) == SERIALIZATION_FAILURE, repr(worker.error)
    env.check("serialization failure after a lock wait")


@pytest.mark.parametrize("isolation", ["READ COMMITTED", "REPEATABLE READ", "SERIALIZABLE"])
def test_concurrent_writers_of_disjoint_groups_commit(make_env, isolation: str) -> None:
    """Writers of different groups do not conflict on maintenance rows (both
    under snapshot isolation), and the result is exact."""
    env = make_env(views=[VIEW_BY_NAME["v_agg"], VIEW_BY_NAME["v_mm"]], seed=True)
    a, b = env.connect(), env.connect()
    ca, cb = a.cursor(), b.cursor()
    for c in (ca, cb):
        c.execute(f"BEGIN ISOLATION LEVEL {isolation}")
        c.execute("SELECT 1")
    ca.execute("DELETE FROM t WHERE g = 1 AND v = (SELECT MIN(v) FROM t WHERE g = 1)")
    cb.execute("UPDATE t SET v = v + 1 WHERE g = 2")
    a.commit()
    try:
        b.commit()
    except Exception as exc:
        # SERIALIZABLE may still report a predicate (rw) conflict through the
        # MIN subquery reads; that is a fail-closed outcome, not a divergence.
        b.rollback()
        assert isolation == "SERIALIZABLE" and _pgcode(exc) == SERIALIZATION_FAILURE, repr(exc)
    env.check(f"disjoint groups under {isolation}")


@pytest.mark.parametrize("view", [V_WIN, V_STRAGG, VIEW_BY_NAME["v_pj"]], ids=lambda v: v.name)
def test_recompute_strategies_serialize_concurrent_writers(make_env, view: ViewDef) -> None:
    """Group-scoped and full-recompute views: A and B insert into the same,
    new group under READ COMMITTED. B must wait for A's per-view lock and then
    recompute from a snapshot that includes A (no duplicated or stale group)."""
    env = make_env(views=[view], seed=True)
    assert _strategy(env, view.name) in ("group_scoped_delta", "atomic_query_sync"), _strategy(env, view.name)
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (5001, 1, 1), (5002, 77, 2)")
    worker = _Worker(b, ["INSERT INTO t (id, g, v) VALUES (5003, 77, 3), (5004, 1, 4)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's view lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("concurrent inserts into one new group")


@pytest.mark.parametrize("view", [V_WIN, VIEW_BY_NAME["v_pj"]], ids=lambda v: v.name)
def test_recompute_strategies_fail_closed_under_repeatable_read(make_env, view: ViewDef) -> None:
    env = make_env(views=[view], seed=True)
    b = env.connect()
    cb = b.cursor()
    cb.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
    cb.execute("SELECT COUNT(*) FROM t")
    cb.fetchall()
    env.execute("INSERT INTO t (id, g, v) VALUES (6001, 1, 1)")
    env.commit()
    with pytest.raises(Exception) as excinfo:
        cb.execute("INSERT INTO t (id, g, v) VALUES (6002, 1, 2)")
        b.commit()
    b.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)
    env.check("REPEATABLE READ writer after a concurrent commit")


# ---------------------------------------------------------------------------
# Partitions and inheritance
# ---------------------------------------------------------------------------

SUBPARTITION_SCHEMA = [
    "CREATE TABLE d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
    "CREATE TABLE t (id INT NOT NULL, g INT NOT NULL, v BIGINT, w INT NOT NULL DEFAULT 0, m NUMERIC(12,3), "
    "PRIMARY KEY (g, w, id)) "
    "PARTITION BY RANGE (g)",
    "CREATE TABLE t_lo PARTITION OF t FOR VALUES FROM (MINVALUE) TO (3) PARTITION BY LIST (w)",
    "CREATE TABLE t_lo_a PARTITION OF t_lo FOR VALUES IN (0, 1, 2)",
    "CREATE TABLE t_lo_b PARTITION OF t_lo DEFAULT",
    "CREATE TABLE t_hi PARTITION OF t FOR VALUES FROM (3) TO (10)",
]
PART_VIEWS = [VIEW_BY_NAME[n] for n in ("v_agg", "v_mm", "v_ds", "v_j", "v_pj", "v_prk")]


def _subpartition_env(make_env) -> Env:
    env = make_env(views=[], schema_sql=SUBPARTITION_SCHEMA, seed=False)
    env.execute("INSERT INTO d (g, name) VALUES " + ", ".join(f"({g}, 'n{g % 3}')" for g in range(10)))
    env.execute("INSERT INTO t (id, g, v, w, m) SELECT i, i % 9, (i * 37) % 101 - 30, i % 5, i / 4.0 "
                "FROM generate_series(1, 60) i")
    env.commit()
    env.views = list(PART_VIEWS)
    for vd in PART_VIEWS:
        env.register(vd)
    return env


def test_subpartitioned_source_all_levels(make_env) -> None:
    env = _subpartition_env(make_env)
    env.check("registration")
    for label, sql in [
        ("leaf insert", "INSERT INTO t_lo_a (id, g, v, w) VALUES (101, 1, 5, 1), (102, 2, -5, 0)"),
        ("intermediate insert", "INSERT INTO t_lo (id, g, v, w) VALUES (103, 0, 7, 4), (104, 1, 8, 2)"),
        ("leaf update", "UPDATE t_lo_b SET v = v * 2"),
        ("intermediate update moving between leaves", "UPDATE t_lo SET w = (w + 3) % 5 WHERE id % 2 = 0"),
        ("root update moving between subtrees", "UPDATE t SET g = (g + 4) % 9 WHERE id % 3 = 0"),
        ("leaf delete", "DELETE FROM t_hi WHERE id % 2 = 1"),
        ("truncate one leaf", "TRUNCATE t_lo_b"),
        ("truncate intermediate", "TRUNCATE t_lo"),
        ("root insert after truncates", "INSERT INTO t (id, g, v, w) VALUES (200, 1, 1, 1), (201, 5, 2, 3)"),
        ("truncate root", "TRUNCATE t"),
    ]:
        env.execute(sql)
        env.commit()
        env.check(label)


def test_partition_attached_after_registration_is_reported_and_repaired(make_env) -> None:
    env = _subpartition_env(make_env)
    env.execute("CREATE TABLE t_new (id INT NOT NULL, g INT NOT NULL, v BIGINT, w INT NOT NULL DEFAULT 0, "
                "m NUMERIC(12,3))")
    env.execute("ALTER TABLE t ATTACH PARTITION t_new FOR VALUES FROM (10) TO (20)")
    env.commit()
    drift = env.manager.check_schema_drift()
    env.commit()
    assert {drift[vd.name] for vd in PART_VIEWS} == {"triggers_missing"}, str(drift)
    # Writes through the root are still maintained (the root's triggers see
    # every routed row); a direct write to the new partition is not.
    env.execute("INSERT INTO t (id, g, v, w) VALUES (300, 11, 4, 1)")
    env.commit()
    env.check("root insert routed into the new partition")
    env.execute("INSERT INTO t_new (id, g, v, w) VALUES (301, 12, 5, 1)")
    env.commit()
    with pytest.raises(ViewMismatch):
        env.check("direct write to a partition without triggers (documented: report + repair)")
    results = env.manager.repair_view()
    env.commit()
    assert {results[vd.name] for vd in PART_VIEWS} == {"rebuilt"}, results
    assert set(env.manager.check_schema_drift().values()) == {"ok"}
    env.commit()
    env.execute("INSERT INTO t_new (id, g, v, w) VALUES (302, 13, 6, 2)")
    env.execute("UPDATE t_new SET v = v + 1")
    env.commit()
    env.check("direct writes to the repaired partition")


@pytest.mark.parametrize("shape", ["view_over_partition", "inherits_parent", "inherits_child"])
def test_sources_hidden_from_statement_triggers_are_refused(make_env, shape: str) -> None:
    schema = {
        "view_over_partition": [
            "CREATE TABLE t (id INT NOT NULL, g INT NOT NULL, v BIGINT, PRIMARY KEY (g, id)) PARTITION BY RANGE (g)",
            "CREATE TABLE t_p PARTITION OF t FOR VALUES FROM (0) TO (100)",
        ],
        "inherits_parent": [
            "CREATE TABLE t (id INT PRIMARY KEY, g INT, v BIGINT)",
            "CREATE TABLE t_child (extra INT) INHERITS (t)",
        ],
        "inherits_child": [
            "CREATE TABLE t_parent (id INT PRIMARY KEY, g INT, v BIGINT)",
            "CREATE TABLE t (extra INT) INHERITS (t_parent)",
        ],
    }[shape]
    source = "t_p" if shape == "view_over_partition" else "t"
    env = make_env(views=[], schema_sql=schema, seed=False)
    import psycopg2

    with pytest.raises(psycopg2.Error) as excinfo:
        env.manager.register_view(f"CREATE VIEW v_refused AS SELECT g, COUNT(*) AS c FROM {source} GROUP BY g;")
    env.rollback()
    assert excinfo.value.pgcode == "0A000", repr(excinfo.value)
    assert "statement-level triggers" in str(excinfo.value)
    assert env.manager.get_catalog() == {}
    env.commit()


# ---------------------------------------------------------------------------
# Cost structure
# ---------------------------------------------------------------------------


def test_one_trigger_invocation_per_statement_and_view(make_env) -> None:
    """A statement of N rows runs each maintenance function once (no row
    trigger, no row loop), and a statement on another table none."""
    views = [VIEW_BY_NAME[n] for n in ("v_agg", "v_mm", "v_j", "v_pj")]
    env = make_env(views=views, seed=True)
    env.execute("SET track_functions = 'all'")
    env.commit()
    env.execute("INSERT INTO t (id, g, v) SELECT i, i % 7, i FROM generate_series(1000, 1499) i")
    env.execute("UPDATE t SET v = v + 1 WHERE id >= 1000")
    env.execute("DELETE FROM t WHERE id >= 1250")
    cur = env.conn.cursor()
    cur.execute("SELECT funcname, calls FROM pg_stat_xact_user_functions WHERE funcname LIKE '%%fn_sync_%%'")
    calls = {r[0]: r[1] for r in cur.fetchall()}
    env.check("statements counted", end_tx=False)
    env.commit()
    for vd in views:
        for sfx, n in (("_ai", 1), ("_au", 1), ("_ad", 1)):
            fn = f"__matview_fn_sync_{vd.name}__t{sfx}"
            assert calls.get(fn) == n, (fn, calls)
        assert not any(f"{vd.name}__d_" in f for f in calls), calls


def _per_statement_us(env: Env, sql_template: str, n: int) -> float:
    cur = env.conn.cursor()
    t0 = time.perf_counter()
    cur.execute(
        "DO $$ BEGIN FOR i IN 1.." + str(n) + " LOOP EXECUTE format($q$" + sql_template + "$q$, i); END LOOP; END $$"
    )
    elapsed = time.perf_counter() - t0
    env.rollback()
    return elapsed / n * 1e6


def test_single_row_cost_does_not_grow_with_view_size(make_env) -> None:
    """Single-row maintenance touches only the affected group, so its cost
    must not grow with the number of groups or source rows: an unrestricted
    empty-group delete scans the whole backing table per statement, and a
    MIN/MAX recompute through ``IS NOT DISTINCT FROM`` (nullable group key)
    scans the whole source table. Both views keep two rows per group at
    both sizes, so every UPDATE/DELETE below retracts a group extreme and
    recomputes it. Measured server-side (DO loop, no client round trips); the
    bound is a ratio between the sizes, so it does not depend on the
    machine's speed."""
    env = make_env(views=[VIEW_BY_NAME["v_agg"], VIEW_BY_NAME["v_mm"]], seed=False)
    costs = {}
    for groups in (1000, 50000):
        env.execute("DELETE FROM t")
        env.execute(f"INSERT INTO t (id, g, v, w) SELECT i, i % {groups}, i, i % 5 FROM generate_series(1, {2 * groups}) i")
        env.commit()
        env.execute("ANALYZE t")
        env.commit()
        ins = _per_statement_us(env, "INSERT INTO t (id, g, v) VALUES (1000000 + %1$s, mod(%1$s, 50), 1)", 1000)
        upd = _per_statement_us(env, "UPDATE t SET v = v + 1 WHERE id = %1$s", 1000)
        dele = _per_statement_us(env, "DELETE FROM t WHERE id = %1$s", 1000)
        costs[groups] = (round(ins, 1), round(upd, 1), round(dele, 1))
    print(f"single-row us/stmt (insert, update, delete) by group count: {costs}")
    for small, large in zip(costs[1000], costs[50000]):
        assert large < 3 * small + 50, costs
    env.check("after the measurements")


def test_statement_spilling_transition_tables_to_disk(make_env) -> None:
    """Transition tables beyond work_mem spill to temporary files; the result
    stays exact. ``MATVIEW_PG_LARGE_ROWS`` raises the row count (the stage-2
    report ran 1,000,000)."""
    rows = int(os.environ.get("MATVIEW_PG_LARGE_ROWS", "150000"))
    env = make_env(views=[VIEW_BY_NAME[n] for n in ("v_agg", "v_mm", "v_ds", "v_j")], seed=True)
    env.execute("SET work_mem = '64kB'")
    env.execute(f"INSERT INTO t (id, g, v, w) SELECT i, i % 97, i % 1000, i % 5 FROM generate_series(100, {rows + 99}) i")
    env.execute("UPDATE t SET g = (g + 1) % 97, v = v - 1")
    env.check("large statements with spilled transition tables", end_tx=False)
    env.execute("DELETE FROM t WHERE id % 2 = 0")
    env.commit()
    env.check("large delete")


# ---------------------------------------------------------------------------
# Primary-key and group-key permutations in one statement
# ---------------------------------------------------------------------------

PERMUTATIONS = [
    ("swap", "UPDATE t SET id = CASE WHEN id = 1 THEN 2 WHEN id = 2 THEN 1 END WHERE id IN (1, 2)"),
    ("three_cycle", "UPDATE t SET id = CASE id WHEN 3 THEN 4 WHEN 4 THEN 5 WHEN 5 THEN 3 END WHERE id IN (3, 4, 5)"),
    ("rotate_overlapping", "UPDATE t SET id = CASE WHEN id = 12 THEN 6 ELSE id + 1 END WHERE id BETWEEN 6 AND 12"),
    ("swap_with_filter_flip", "UPDATE t SET id = 23 - id, w = CASE WHEN w > 0 THEN 0 ELSE 3 END WHERE id IN (10, 13)"),
    ("swap_group_keys", "UPDATE t SET g = CASE WHEN g = 1 THEN 2 WHEN g = 2 THEN 1 ELSE g END"),
    ("key_and_group_rotation", "UPDATE t SET id = id + 1000, g = (COALESCE(g, 0) + 1) % 4 WHERE id % 3 = 0"),
]


@pytest.mark.parametrize("name,sql", PERMUTATIONS, ids=[p[0] for p in PERMUTATIONS])
def test_primary_key_permutations(make_env, name: str, sql: str) -> None:
    """One statement that permutes primary keys (or group keys) is one signed
    delta: rows keyed by the source primary key (projections, UNION ALL) and
    groups are exact, including keys that are both retracted and re-inserted.
    The primary key is DEFERRABLE (checked at commit) so that a permutation
    is one statement; the per-row SQLite emulation cannot represent this."""
    views = [VIEW_BY_NAME[n] for n in ("v_prk", "v_pr", "v_agg", "v_mm", "v_ds", "v_pj", "v_j")]
    views.append(ViewDef("v_ua", "SELECT id, g FROM t WHERE w > 2 UNION ALL SELECT id, g FROM t WHERE w <= 2"))
    schema = [ddl.replace("id INT PRIMARY KEY,", "id INT NOT NULL,").replace(
        "CONSTRAINT t_v_bound", "PRIMARY KEY (id) DEFERRABLE INITIALLY DEFERRED, CONSTRAINT t_v_bound")
        for ddl in PG_SCHEMA]
    env = make_env(views=views, schema_sql=schema, seed=True)
    env.execute(sql)
    env.check(f"{name} (in transaction)", end_tx=False)
    env.commit()
    env.check(name)
    env.execute("DELETE FROM t WHERE id % 2 = 0")
    env.commit()
    env.check(f"{name} then delete of half the rows")
