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
import threading
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
    # Commit one write to the same groups first, so the lock rows already
    # exist: a lock statement that does not lock an existing row (for example
    # ON CONFLICT DO NOTHING) must make this test fail, not wait on an
    # uncommitted unique-index entry instead.
    env.execute("INSERT INTO t (id, g, v) VALUES (4999, 1, 0), (4998, 77, 0)")
    env.commit()
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


def test_partition_detached_after_registration_is_reported_and_repaired(make_env) -> None:
    """DETACH keeps the partition's rows in the view and leaves the
    maintenance triggers on the detached table (they belong to the view's
    functions): the drift check reports "triggers_orphaned", repair rebuilds
    (drops them, re-installs on the current tree, repopulates), and writes to
    the detached table no longer change the view."""
    env = _subpartition_env(make_env)
    env.execute("ALTER TABLE t DETACH PARTITION t_hi")
    env.commit()
    drift = env.manager.check_schema_drift()
    env.commit()
    assert {drift[vd.name] for vd in PART_VIEWS} == {"triggers_orphaned"}, str(drift)
    with pytest.raises(ViewMismatch):
        env.check("detached rows are still in the view (documented: report + repair)")
    results = env.manager.repair_view()
    env.commit()
    assert {results[vd.name] for vd in PART_VIEWS} == {"rebuilt"}, results
    assert set(env.manager.check_schema_drift().values()) == {"ok"}
    env.commit()
    env.check("after repair")
    cur = env.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pg_trigger WHERE tgrelid = 't_hi'::regclass AND NOT tgisinternal")
    assert cur.fetchone()[0] == 0
    env.execute("INSERT INTO t_hi (id, g, v, w) VALUES (900, 5, 7, 1)")
    env.execute("UPDATE t_hi SET v = v + 100")
    env.execute("DELETE FROM t_hi WHERE id % 2 = 0")
    env.commit()
    env.check("writes to the detached table do not change the view")
    env.execute("INSERT INTO t (id, g, v, w) VALUES (901, 1, 7, 1)")
    env.commit()
    env.check("writes to the remaining tree are maintained")


def test_partition_created_or_dropped_after_registration(make_env) -> None:
    """CREATE TABLE ... PARTITION OF behaves like ATTACH (reported missing,
    repaired). DROP of a partition fires no trigger (DDL): the view keeps the
    dropped rows until repair_view(), and no catalog or trigger residue is left."""
    env = _subpartition_env(make_env)
    env.execute("CREATE TABLE t_late PARTITION OF t FOR VALUES FROM (10) TO (20)")
    env.commit()
    drift = env.manager.check_schema_drift()
    env.commit()
    assert {drift[vd.name] for vd in PART_VIEWS} == {"triggers_missing"}, str(drift)
    env.manager.repair_view()
    env.commit()
    env.execute("INSERT INTO t_late (id, g, v, w) VALUES (950, 12, 3, 1)")
    env.commit()
    env.check("direct write to the repaired new partition")
    env.execute("DROP TABLE t_hi")
    env.commit()
    assert set(env.manager.check_schema_drift().values()) == {"ok"}
    env.commit()
    with pytest.raises(ViewMismatch):
        env.check("rows of a dropped partition (documented: DROP is DDL, repair_view() fixes it)")
    env.manager.repair_view()
    env.commit()
    env.check("after repair of a dropped partition")
    cur = env.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pg_trigger WHERE tgname LIKE '%%t_hi%%'")
    assert cur.fetchone()[0] == 0
    env.commit()


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
    recomputes it. The group-scoped views (window RANK-like ROW_NUMBER,
    STRING_AGG) recompute only the touched groups through index-usable key
    predicates (a NULL-safe EXISTS would scan the table). Measured server-side
    (DO loop, no client round trips); the bound is a ratio between the sizes,
    so it does not depend on the machine's speed."""
    env = make_env(views=[VIEW_BY_NAME["v_agg"], VIEW_BY_NAME["v_mm"], V_WIN, V_STRAGG], seed=False)
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


# ---------------------------------------------------------------------------
# Group-scoped views: per-group locks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("view", [V_WIN, V_STRAGG], ids=lambda v: v.name)
def test_group_scoped_writers_of_disjoint_groups_neither_block_nor_deadlock(make_env, view: ViewDef) -> None:
    """A updates a row of one group and keeps its transaction open; B updates
    a row of another group: B must not wait for A (no whole-view lock).
    Then A updates B's row: A waits for B's row lock only, B commits, A
    commits; no deadlock, views exact."""
    env = make_env(views=[view], seed=True)
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("UPDATE t SET v = v + 1 WHERE id = 10")  # g = 4
    cur_b = b.cursor()
    cur_b.execute("SET LOCAL lock_timeout = '10s'")
    cur_b.execute("UPDATE t SET v = v + 1 WHERE id = 20")  # g = 2: completes while A is open
    worker = _Worker(a, ["UPDATE t SET v = v + 2 WHERE id = 20"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_a, sid_b), worker.done, "A waits for B's source row")
    finally:
        b.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("disjoint group writers")


def test_group_scoped_same_group_writers_wait_and_stay_exact(make_env) -> None:
    env = make_env(views=[V_WIN, V_STRAGG], seed=True)
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("UPDATE t SET v = v + 1 WHERE id = 10")  # g = 4
    worker = _Worker(b, ["UPDATE t SET v = -v WHERE id = 16"]).start()  # g = 4
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's group lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("same group writers")


# ---------------------------------------------------------------------------
# Aggregate recompute conditions and storage
# ---------------------------------------------------------------------------

# BIGINT argument: BIT_AND/BIT_OR(INT) results are stored as BIGINT (TODO.md,
# pre-existing result-type divergence), which this test does not cover.
V_BITS = ViewDef("v_bits", "SELECT g, BIT_AND(v) AS ba, BIT_OR(v) AS bo, COUNT(*) AS c FROM t GROUP BY g")


def test_bit_and_bit_or_after_retracting_a_contributor(make_env) -> None:
    """BIT_OR/BIT_AND are recomputed after a retraction of a non-NULL value:
    deleting (or updating away) the only row that sets or clears a bit must
    change the result."""
    env = make_env(views=[V_BITS], seed=False)
    env.execute("INSERT INTO t (id, g, v) VALUES (1, 1, 1), (2, 1, 3), (3, 1, 4), (4, 2, 7), (5, 2, 5), (6, 3, 6)")
    env.commit()
    for label, sql in [
        ("delete the only row with bit 4", "DELETE FROM t WHERE id = 3"),
        ("update away the only row without bit 2", "UPDATE t SET v = 7 WHERE id = 5"),
        ("update one value to NULL", "UPDATE t SET v = NULL WHERE id = 4"),
        ("delete the last non-NULL value", "DELETE FROM t WHERE id = 6"),
    ]:
        env.execute(sql)
        env.commit()
        env.check(label)


COUNTED_VIEWS = [
    ViewDef("v_un", "SELECT g FROM t WHERE w > 1 UNION SELECT g FROM t WHERE v > 0"),
    ViewDef("v_in", "SELECT g FROM t WHERE w > 1 INTERSECT SELECT g FROM t WHERE v > 0"),
    ViewDef("v_ex2", "SELECT g FROM t WHERE w > 1 EXCEPT SELECT g FROM t WHERE v > 50"),
    VIEW_BY_NAME["v_ds"], VIEW_BY_NAME["v_agg"], VIEW_BY_NAME["v_j"],
]


def test_no_empty_rows_remain_in_backing_tables(make_env) -> None:
    """Retractions delete rows whose counts reach zero: no backing row with
    all counts <= 0 may remain (they are invisible to readers but leak
    storage and slow every lookup)."""
    env = make_env(views=COUNTED_VIEWS, seed=True)
    for sql in ("UPDATE t SET g = (COALESCE(g, 0) + 1) % 4", "DELETE FROM t WHERE id % 3 = 0",
                "UPDATE t SET w = 0, v = -1 WHERE id % 2 = 0", "DELETE FROM t WHERE g = 1"):
        env.execute(sql)
        env.commit()
        env.check(sql)
    cur = env.conn.cursor()
    for vd in COUNTED_VIEWS:
        store = env.plans[vd.name].backing_table_name
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s "
                    "AND (column_name LIKE '__mv_cnt_%%' OR column_name IN ('__mv_row_count', 'c'))", (store,))
        cols = [r[0] for r in cur.fetchall()]
        assert cols, (vd.name, store)
        cur.execute(f"SELECT COUNT(*) FROM {store} WHERE " + " AND ".join(f"{c} <= 0" for c in cols))
        assert cur.fetchone()[0] == 0, (vd.name, cols)
    env.commit()


# ---------------------------------------------------------------------------
# search_path of the writer
# ---------------------------------------------------------------------------


def test_writer_search_path_does_not_redirect_maintenance(make_env) -> None:
    """The trigger functions run with the search_path of their creation
    (SET search_path FROM CURRENT): a writer whose search_path puts another
    schema with a table `t` first must not make the recomputes read it."""
    env = make_env(views=[VIEW_BY_NAME["v_mm"], V_WIN, VIEW_BY_NAME["v_pj"]], seed=True)
    evil = env.ns + "_evil"
    cur = env.conn.cursor()
    cur.execute(f'CREATE SCHEMA "{evil}"')
    cur.execute(f'CREATE TABLE "{evil}".t (id INT PRIMARY KEY, g INT, v BIGINT, w INT, m NUMERIC(12,3))')
    cur.execute(f'CREATE TABLE "{evil}".d (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)')
    cur.execute(f'INSERT INTO "{evil}".t SELECT i, i % 6, -1000 - i, 1, 0 FROM generate_series(1, 60) i')
    env.commit()
    try:
        w = env.connect()
        wc = w.cursor()
        wc.execute(f'SET search_path TO "{evil}", "{env.ns}"')
        wc.execute(f'DELETE FROM "{env.ns}".t WHERE id IN (SELECT id FROM "{env.ns}".t WHERE v IS NOT NULL ORDER BY v LIMIT 5)')
        wc.execute(f'UPDATE "{env.ns}".t SET v = v + 1 WHERE g = 2')
        wc.execute(f'INSERT INTO "{env.ns}".t (id, g, v) VALUES (700, 3, 1)')
        w.commit()
        env.check("writer with a hostile search_path")
        # TRUNCATE has its own maintenance function (a full recompute).
        wc.execute(f'TRUNCATE "{env.ns}".t')
        wc.execute(f'INSERT INTO "{env.ns}".t (id, g, v) VALUES (701, 1, 2), (702, 2, 3)')
        w.commit()
        env.check("TRUNCATE by a writer with a hostile search_path")
    finally:
        cur.execute(f'DROP SCHEMA "{evil}" CASCADE')
        env.commit()


# ---------------------------------------------------------------------------
# Known limitation: writers of different sources of an algebraic join view
# ---------------------------------------------------------------------------

_JOIN_RACE = (
    "Two READ COMMITTED transactions writing DIFFERENT sources of an algebraic join view each "
    "join their delta with the other table's committed state and miss each other's change. "
    "TODO.md 'Serialize Concurrent Writers of Different Sources of a PostgreSQL Join View'"
)


@pytest.mark.xfail(strict=True, raises=ViewMismatch, reason=_JOIN_RACE)
def test_join_view_writers_of_different_sources_read_committed(make_env) -> None:
    env = make_env(views=[VIEW_BY_NAME["v_j"]], seed=True)
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (3000, 1, 5)")
    worker = _Worker(b, ["UPDATE d SET name = 'n2' WHERE g = 1"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's group lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("concurrent writers of t and d")


def test_join_view_writers_of_different_sources_repeatable_read_fail_closed(make_env) -> None:
    env = make_env(views=[VIEW_BY_NAME["v_j"]], seed=True)
    a, b = env.connect(), env.connect()
    cb = b.cursor()
    cb.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
    cb.execute("SELECT COUNT(*) FROM t")
    cb.fetchall()
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (3000, 1, 5)")
    a.commit()
    with pytest.raises(Exception) as excinfo:
        cb.execute("UPDATE d SET name = 'n2' WHERE g = 1")
        b.commit()
    b.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)
    env.check("REPEATABLE READ writer of the other source")


def test_whole_view_recompute_excludes_group_writers(make_env) -> None:
    """A group-scoped view whose WHERE reads another table: writes to t
    recompute their groups (holding the view lock row in SHARE mode), writes
    to u recompute the whole view (holding it exclusively). The whole-view
    writer must wait for the group writer and then see it."""
    schema = list(PG_SCHEMA) + ["CREATE TABLE u (g INT PRIMARY KEY)"]
    view = ViewDef("v_sub", "SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids FROM t "
                            "WHERE g IN (SELECT g FROM u) GROUP BY g")
    env = make_env(views=[], schema_sql=schema, seed=True)
    env.execute("INSERT INTO u (g) VALUES (1), (2), (3)")
    env.commit()
    env.views = [view]
    env.register(view)
    assert _strategy(env, "v_sub") == "group_scoped_delta"
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (6001, 1, 1)")
    worker = _Worker(b, ["INSERT INTO u (g) VALUES (4)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's view lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("whole-view recompute after a group writer")



def test_group_scoped_single_row_cost_close_to_no_view(make_env) -> None:
    """Group-scoped single-row maintenance against a no-view baseline on the
    same table (100k rows, 10k groups, index on the group key): a table scan
    per statement costs milliseconds, the per-group recompute tens to
    hundreds of microseconds."""
    env = make_env(views=[], seed=False)
    env.execute("INSERT INTO t (id, g, v, w) SELECT i, i % 10000, i, i % 5 FROM generate_series(1, 100000) i")
    env.commit()
    env.execute("ANALYZE t")
    env.commit()
    stmts = ("INSERT INTO t (id, g, v) VALUES (1000000 + %1$s, mod(%1$s, 10000), 1)",
             "UPDATE t SET v = v + 1 WHERE id = %1$s", "DELETE FROM t WHERE id = %1$s")
    base = [_per_statement_us(env, sql, 500) for sql in stmts]
    results = {}
    for view in (V_WIN, V_STRAGG, ViewDef("v_rank", "SELECT id, g, RANK() OVER (PARTITION BY g ORDER BY v) AS r FROM t")):
        env.views = [view]
        env.register(view)
        env.execute("ANALYZE")
        env.commit()
        results[view.name] = [_per_statement_us(env, sql, 500) for sql in stmts]
        env.manager.drop_view(view.name)
        env.commit()
    print(f"group-scoped single-row us/stmt (ins, upd, del): none={base} {results}")
    for name, costs in results.items():
        for b, c in zip(base, costs):
            assert c < 20 * b + 1000, (name, base, costs)


# ---------------------------------------------------------------------------
# Group identity: keys that are equal but print differently
# ---------------------------------------------------------------------------
#
# GROUP BY, DISTINCT and set operations use the key type's equality: NUMERIC
# 1.0 = 1.00, FLOAT8 0 = -0 (and NaN = NaN), INTERVAL '1 day' = '24 hours',
# and a TIMESTAMPTZ instant is one value whatever the session TimeZone. A
# nullable group key is identified in the backing table by a canonical text
# (__mv_pk), and a group-scoped view locks a row whose id hashes that text:
# both must be equal exactly when the keys are equal, in any session.
#
# Which equal value a group reports (1.0 or 1.00) is unspecified for native
# GROUP BY as well, so these checks compare a normalized form of the key
# (KEY_TYPES[...][3]) with exact counts and sums, and require exactly one
# backing row per native group.

# name -> (column type, groups of equal literals, distinct literal, key normalization for the comparison)
KEY_TYPES = {
    "numeric": ("NUMERIC", [["1.0", "1.00", "1", "1.000"], ["-2.50", "-2.5"]], "3.25", "trim_scale({k})::text"),
    "float8": ("DOUBLE PRECISION", [["0::float8", "'-0'::float8"], ["'NaN'::float8", "'Infinity'::float8 - 'Infinity'::float8"]],
               "0.1::float8", "({k} + 0)::text"),
    "real": ("REAL", [["0::real", "'-0'::real"]], "0.1::real", "({k} + 0)::text"),
    "timestamptz": ("TIMESTAMPTZ", [["'2026-01-01 00:00:00+00'", "'2026-01-01 09:00:00+09'", "'2025-12-31 19:00:00-05'"]],
                    "'2026-01-02 00:00:00+00'", "({k} AT TIME ZONE 'UTC')::text"),
    "timestamp": ("TIMESTAMP", [["'2026-01-01 10:00:00'", "'2026-01-01 10:00:00.000'"]], "'2026-01-02 10:00:00'",
                  "{k}::text"),
    "date": ("DATE", [["'2026-03-04'", "'2026-03-04'::timestamp"]], "'2026-03-05'", "{k}::text"),
    "interval": ("INTERVAL", [["'1 day'", "'24 hours'", "'1440 minutes'"], ["'1 mon'", "'30 days'"]], "'1 hour'",
                 "justify_interval({k})::text"),
    "bytea": ("BYTEA", [["'\\x00ff'::bytea", "decode('00ff', 'hex')"]], "'\\x01'::bytea", "encode({k}, 'hex')"),
    "jsonb": ("JSONB", [["'{\"a\": 1.0, \"b\": [2]}'", "'{\"b\": [2.00], \"a\": 1}'"]], "'{\"a\": 2}'",
              "trim_scale(({k}->>'a')::numeric)::text || ':' || trim_scale(({k}->'b'->>0)::numeric)::text"),
    "numeric_array": ("NUMERIC[]", [["'{1.0,NULL,2}'", "'{1,NULL,2.00}'"]], "'{1,2}'",
                      "(SELECT array_agg(trim_scale(x) ORDER BY o) FROM unnest({k}) WITH ORDINALITY u(x, o))::text"),
    "numrange": ("NUMRANGE", [["'[1.0,2.50)'", "'[1,2.5)'"]], "'[1,3)'",
                 "numrange(trim_scale(lower({k})), trim_scale(upper({k})))::text"),
}

# Session settings that change how values print; writes alternate between them.
SESSION_PROFILES = [
    ["SET extra_float_digits = 0", "SET TimeZone = 'Asia/Tokyo'", "SET DateStyle = 'SQL, DMY'",
     "SET IntervalStyle = 'sql_standard'", "SET bytea_output = 'escape'"],
    ["SET extra_float_digits = 3", "SET TimeZone = 'UTC'", "SET DateStyle = 'ISO, MDY'",
     "SET IntervalStyle = 'postgres'", "SET bytea_output = 'hex'"],
]


def _key_views(kind: str) -> List[ViewDef]:
    views = [
        ViewDef("ck_agg", "SELECT k, COUNT(*) AS n, SUM(v) AS s FROM ck GROUP BY k"),
        ViewDef("ck_two", "SELECT k, g, COUNT(*) AS n, SUM(v) AS s FROM ck GROUP BY k, g"),
        ViewDef("ck_str", "SELECT k, STRING_AGG(CAST(v AS TEXT), ',' ORDER BY v) AS a, COUNT(*) AS n FROM ck GROUP BY k"),
        ViewDef("ck_dst", "SELECT DISTINCT k FROM ck"),
        ViewDef("ck_un", "SELECT k FROM ck WHERE v > 0 UNION SELECT k FROM ck WHERE v < 100"),
    ]
    if kind in ("numeric", "float8"):
        views.append(ViewDef("ck_expr", "SELECT k * 2 AS k2, COUNT(*) AS n, SUM(v) AS s FROM ck GROUP BY k * 2"))
    return views


def _check_key_groups(env: Env, kind: str, label: str) -> None:
    """Each view equals its query up to the reported representative of a
    group, and stores exactly one row per group."""
    norm = KEY_TYPES[kind][3]
    cur = env.conn.cursor()
    for vd in env.views:
        key = "k2" if vd.name == "ck_expr" else "k"
        cur.execute(f"SELECT * FROM {vd.name} LIMIT 0")
        cols = [d[0] for d in cur.description]
        sel = ", ".join(norm.format(k=c) if c == key else c for c in cols)
        cur.execute(f"SELECT COUNT(*) FROM {env.plans[vd.name].backing_table_name}")
        stored = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM ({vd.select}) AS q")
        assert stored == cur.fetchone()[0], f"{label}: {vd.name} stores {stored} rows"
        got = []
        for src in (vd.name, f"({vd.select}) AS q"):
            cur.execute(f"SELECT {sel} FROM {src}")
            got.append(sorted(map(repr, cur.fetchall())))
        assert got[0] == got[1], f"{label}: {vd.name}\n view:  {got[0]}\n query: {got[1]}"
    env.conn.commit()


def _key_env(make_env, kind: str) -> Env:
    col_type = KEY_TYPES[kind][0]
    return make_env(views=[], seed=False,
                    schema_sql=[f"CREATE TABLE ck (id INT PRIMARY KEY, k {col_type}, g INT, v INT)"])


@pytest.mark.parametrize("kind", list(KEY_TYPES))
def test_equal_keys_that_print_differently_form_one_group(make_env, kind: str) -> None:
    _, groups, other, _ = KEY_TYPES[kind]
    env = _key_env(make_env, kind)
    cur = env.conn.cursor()
    ids = iter(range(1, 1000))

    def write(sql: str, profile: int) -> None:
        for s in SESSION_PROFILES[profile]:
            cur.execute(s)
        cur.execute(sql)
        env.conn.commit()

    # One existing row per group before registration (initial population).
    for grp in groups:
        write(f"INSERT INTO ck VALUES ({next(ids)}, {grp[0]}, 1, 10)", 0)
    write(f"INSERT INTO ck VALUES ({next(ids)}, NULL, 1, 5)", 0)
    env.views = _key_views(kind)
    for vd in env.views:
        env.register(vd)
    _check_key_groups(env, kind, "registration")
    step = 0
    for grp in groups:
        for lit in grp[1:]:
            step += 1
            write(f"INSERT INTO ck VALUES ({next(ids)}, {lit}, {step % 2}, {step})", step % 2)
            _check_key_groups(env, kind, f"INSERT {lit}")
        rows = ", ".join(f"({next(ids)}, {lit}, 2, {50 + i})" for i, lit in enumerate(grp))
        write(f"INSERT INTO ck VALUES {rows}", 1)
        _check_key_groups(env, kind, f"multi-row INSERT {grp}")
        oid = next(ids)
        write(f"INSERT INTO ck VALUES ({oid}, {other}, 1, 7)", 0)
        write(f"UPDATE ck SET k = {grp[-1]} WHERE id = {oid}", 1)
        _check_key_groups(env, kind, f"UPDATE moving a row into {grp[-1]}")
        write(f"UPDATE ck SET k = {grp[0]} WHERE k = {grp[0]}", 0)
        _check_key_groups(env, kind, f"UPDATE rewriting {grp[0]} in place")
    write("UPDATE ck SET k = NULL WHERE id % 3 = 0", 1)
    _check_key_groups(env, kind, "UPDATE to NULL")
    for i in sorted({r for r in range(1, 60)}):
        cur.execute("DELETE FROM ck WHERE id = %s", (i,))
        env.conn.commit()
        if i % 4 == 0:
            _check_key_groups(env, kind, f"DELETE {i}")
    _check_key_groups(env, kind, "DELETE all")


def test_timestamptz_key_value_independent_of_session_timezone(make_env) -> None:
    """A TIMESTAMPTZ key is stored as TIMESTAMPTZ (SQLAlchemy reflects it as
    TIMESTAMP(timezone=True)), not as a TIMESTAMP whose value would depend on
    the writer's TimeZone."""
    env = _key_env(make_env, "timestamptz")
    env.views = [ViewDef("ck_agg", "SELECT k, COUNT(*) AS n, SUM(v) AS s FROM ck GROUP BY k")]
    env.register(env.views[0])
    cur = env.conn.cursor()
    cur.execute("SET TimeZone = 'Asia/Tokyo'")
    cur.execute("INSERT INTO ck VALUES (1, '2026-01-01 00:00:00+00', 1, 1)")
    env.conn.commit()
    cur.execute("SET TimeZone = 'UTC'")
    cur.execute("SELECT (SELECT k FROM ck_agg) = (SELECT k FROM ck)")
    assert cur.fetchone()[0] is True
    env.conn.commit()


@pytest.mark.parametrize("kind,view", [("numeric", "ck_str"), ("numeric", "ck_agg"), ("float8", "ck_str"),
                                       ("interval", "ck_str")])
def test_concurrent_writers_of_equal_keys_that_print_differently(make_env, kind: str, view: str) -> None:
    """READ COMMITTED: A inserts one spelling of a key and stays open; B
    inserts another spelling of the same key from a session with other print
    settings. They are one group, so B must wait for A's lock (the backing
    row of an algebraic view, the group lock row of a group-scoped one)."""
    _, groups, _, _ = KEY_TYPES[kind]
    env = _key_env(make_env, kind)
    env.views = [vd for vd in _key_views(kind) if vd.name == view]
    env.register(env.views[0])
    env.execute(f"INSERT INTO ck VALUES (1, {groups[0][0]}, 1, 1)")
    env.commit()
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    ca = a.cursor()
    for s in SESSION_PROFILES[0]:
        ca.execute(s)
    ca.execute(f"INSERT INTO ck VALUES (10, {groups[0][0]}, 1, 10)")
    worker = _Worker(b, [*SESSION_PROFILES[1], f"INSERT INTO ck VALUES (11, {groups[0][1]}, 1, 11)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's group lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    _check_key_groups(env, kind, "concurrent writers of one group")


def test_nondeterministic_collation_keys_are_refused_at_registration(make_env) -> None:
    """Under a nondeterministic collation 'A' = 'a': no stored value or
    canonical text identifies the group. Registration refuses (0A000) a view
    that groups, deduplicates or takes MIN/MAX over such a column or
    collation; views not mentioning one, and projections, are maintained."""
    env = make_env(views=[], seed=False, schema_sql=[
        "CREATE COLLATION IF NOT EXISTS mv_test_ci (provider = icu, locale = 'und-u-ks-level2', deterministic = false)",
        "CREATE TABLE tc (id INT PRIMARY KEY, c TEXT COLLATE mv_test_ci, d TEXT, v INT)",
    ])
    env.execute("INSERT INTO tc VALUES (1, 'A', 'A', 1), (2, 'a', 'a', 2)")
    env.commit()
    refused = [
        ViewDef("vc_grp", "SELECT c, COUNT(*) AS n FROM tc GROUP BY c"),
        ViewDef("vc_dst", "SELECT DISTINCT c FROM tc"),
        ViewDef("vc_max", "SELECT v, MAX(c) AS m FROM tc GROUP BY v"),
        ViewDef("vc_coll", "SELECT d COLLATE mv_test_ci AS d2, COUNT(*) AS n FROM tc GROUP BY d COLLATE mv_test_ci"),
        ViewDef("vc_str", "SELECT c, STRING_AGG(d, ',' ORDER BY d) AS a FROM tc GROUP BY c"),
    ]
    for vd in refused:
        with pytest.raises(Exception) as excinfo:
            env.register(vd)
        env.conn.rollback()
        assert _pgcode(excinfo.value) == "0A000", (vd.name, repr(excinfo.value))
        assert "mv_test_ci" in str(excinfo.value), (vd.name, str(excinfo.value))
    allowed = [
        ViewDef("vd_grp", "SELECT d, COUNT(*) AS n, SUM(v) AS s FROM tc GROUP BY d"),
        ViewDef("vc_proj", "SELECT id, c, v FROM tc WHERE v > 0"),
    ]
    env.views = allowed
    for vd in allowed:
        env.register(vd)
    env.execute("INSERT INTO tc VALUES (3, 'B', 'b', 3)")
    env.execute("UPDATE tc SET c = 'b', d = 'B' WHERE id = 1")
    env.commit()
    env.check("views without a nondeterministic key")


def test_unsupported_key_type_fails_closed(make_env) -> None:
    """A multirange-typed nullable group key has no canonical text (yet): the
    write fails with 0A000 instead of possibly splitting a group."""
    env = make_env(views=[], seed=False, schema_sql=[
        "CREATE TABLE cp (id INT PRIMARY KEY, p NUMMULTIRANGE, v INT)",
    ])
    vd = ViewDef("vp", "SELECT p, COUNT(*) AS n FROM cp GROUP BY p")
    env.views = [vd]
    env.register(vd)
    with pytest.raises(Exception) as excinfo:
        env.execute("INSERT INTO cp VALUES (1, '{[1.0,2)}', 1)")
        env.commit()
    env.conn.rollback()
    assert _pgcode(excinfo.value) == "0A000", repr(excinfo.value)
    assert "nummultirange" in str(excinfo.value), str(excinfo.value)
    env.check("after the refused write")


# ---------------------------------------------------------------------------
# Lock-id hash collisions and repair_view under concurrent writers
# ---------------------------------------------------------------------------


def test_group_lock_id_collisions_only_serialize(make_env, monkeypatch) -> None:
    """Every group hashes to the same lock row (forced): writers of different
    groups then wait for each other but stay exact."""
    from matview.compiler import IncrementalViewCompiler

    monkeypatch.setattr(IncrementalViewCompiler, "_pg_group_lock_id_sql", lambda self, canonical_key: "1")
    env = make_env(views=[V_STRAGG, V_WIN], seed=True)
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("UPDATE t SET v = v + 1 WHERE id = 10")  # g = 4
    worker = _Worker(b, ["UPDATE t SET v = v + 1 WHERE id = 20"]).start()  # g = 2
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits on the shared lock row")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("colliding lock ids")


@pytest.mark.parametrize("view", [VIEW_BY_NAME["v_agg"], V_STRAGG, VIEW_BY_NAME["v_pj"]], ids=lambda v: v.name)
def test_repair_view_waits_for_writers_and_writers_wait_for_it(make_env, view: ViewDef) -> None:
    """repair_view() repopulates under SHARE locks on the sources: a repair
    started while a writer is open waits for it and includes its commit; a
    writer started during a repair waits for it and is maintained after."""
    from matview.manager import MaterializedViewManager

    env = make_env(views=[view], seed=True)
    cur = env.conn.cursor()
    cur.execute(f"DELETE FROM {env.plans[view.name].backing_table_name}")  # damage it: repair must repopulate
    env.commit()
    a, r, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_r = _session_id(env, a), _session_id(env, r)
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (7001, 1, 3)")
    repair_mgr = MaterializedViewManager(r, dialect=env.target.dialect)
    result: dict = {}

    def run_repair() -> None:
        try:
            result["status"] = repair_mgr.repair_view(view.name)
            r.commit()
        except BaseException as exc:  # inspected below
            result["error"] = exc
            r.rollback()

    th = threading.Thread(target=run_repair, daemon=True)
    done = threading.Event()
    th.start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_r, sid_a), done, "the repair waits for the writer")
    finally:
        a.commit()
        th.join(60)
    assert "error" not in result, result
    env.check("repair after a concurrent writer")
    # A writer during a repair: X holds the backing table in SHARE mode, so
    # the repair (holding its source locks) waits for X; the writer then waits
    # for the repair.
    store = env.plans[view.name].backing_table_name
    cur.execute(f"DELETE FROM {store}")
    env.commit()
    x = env.connect()
    sid_x = _session_id(env, x)
    x.cursor().execute(f"LOCK TABLE {store} IN SHARE MODE")
    result.clear()
    th = threading.Thread(target=run_repair, daemon=True)
    th.start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_r, sid_x), done, "the repair waits for X")
        worker = _Worker(a, ["INSERT INTO t (id, g, v) VALUES (7002, 2, 4)"]).start()
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_a, sid_r), worker.done, "the writer waits for the repair")
    finally:
        x.commit()
        th.join(60)
    worker.join()
    assert "error" not in result, result
    if worker.error is not None:
        raise worker.error
    env.check("writer during a repair")


def test_key_text_function_is_created_and_dropped_with_the_view(make_env) -> None:
    env = _key_env(make_env, "numeric")
    vd = ViewDef("ck_agg", "SELECT k, COUNT(*) AS n, SUM(v) AS s FROM ck GROUP BY k")
    env.views = [vd]
    env.register(vd)
    plan = env.plans[vd.name]
    key_fns = [f for f in plan.function_names if f.endswith("(anyelement)")]
    assert len(key_fns) == 1, plan.function_names
    cur = env.conn.cursor()
    count = "SELECT COUNT(*) FROM pg_proc WHERE proname = %s AND pronamespace = %s::regnamespace"
    cur.execute(count, (key_fns[0].split("(")[0], env.ns))
    assert cur.fetchone()[0] == 1
    env.conn.commit()
    env.manager.drop_view(vd.name)
    env.views = []
    cur.execute(count, (key_fns[0].split("(")[0], env.ns))
    assert cur.fetchone()[0] == 0
    env.conn.commit()


def test_native_postgres_array_backing_types_and_expressions(make_env, monkeypatch) -> None:
    """PostgreSQL array-typed columns (built-in, parameterized, unbounded VARCHAR/BPCHAR,
    TIMESTAMPTZ, BYTEA, JSONB, NUMRANGE, mixed-case ENUM, domain-over-array, array-of-domain,
    2D arrays, partitioned tables, and array expressions) register cleanly with native backing
    column types, preserve exact values and result OIDs across DML, and work via both
    SQLAlchemy and raw DB-API introspection."""
    import matview.connectivity as _conn_mod

    env = make_env(
        views=[],
        seed=False,
        schema_sql=[
            'CREATE TYPE "MixedArrEnum" AS ENUM (\'alpha\', \'beta\')',
            "CREATE DOMAIN pos_int_dom AS INTEGER CHECK (VALUE > 0)",
            "CREATE DOMAIN int_arr_dom AS INTEGER[]",
            """
            CREATE TABLE arr_all (
                id INT PRIMARY KEY,
                g INT NOT NULL,
                v INT NOT NULL,
                a_int INT[],
                a_int2d INT[][],
                a_num NUMERIC[],
                a_num_ps NUMERIC(10, 2)[],
                a_txt TEXT[],
                a_vc VARCHAR[],
                a_vc20 VARCHAR(20)[],
                a_c3 CHAR(3)[],
                a_bp BPCHAR[],
                a_ts TIMESTAMPTZ[],
                a_bin BYTEA[],
                a_jb JSONB[],
                a_rng NUMRANGE[],
                a_enum "MixedArrEnum"[],
                a_dom_elem pos_int_dom[],
                a_dom_arr int_arr_dom
            )
            """,
            """
            CREATE TABLE arr_part (
                id INT NOT NULL,
                g INT NOT NULL,
                k NUMERIC[],
                v INT NOT NULL,
                PRIMARY KEY (g, id)
            ) PARTITION BY LIST (g)
            """,
            "CREATE TABLE arr_part_1 PARTITION OF arr_part FOR VALUES IN (1)",
            "CREATE TABLE arr_part_2 PARTITION OF arr_part FOR VALUES IN (2)",
        ],
    )
    cur = env.conn.cursor()

    # Verify both SQLAlchemy reflection and raw DB-API fallback produce exact native array DDL types
    # (never bare "ARRAY" or "NULL", and preserving typmods, VARCHAR[] vs TEXT[], TIMESTAMPTZ[], and enums).
    sa_schemas = env.manager.fetch_schema()
    monkeypatch.setattr(_conn_mod, "_can_use_sqlalchemy_inspector", lambda _c: False)
    raw_schemas = env.manager.fetch_schema()
    monkeypatch.undo()

    expected_reflected = {
        "a_int": "integer[]",
        "a_int2d": "integer[]",
        "a_num": "numeric[]",
        "a_num_ps": "numeric(10,2)[]",
        "a_txt": "text[]",
        "a_vc": "character varying[]",
        "a_vc20": "character varying(20)[]",
        "a_c3": "character(3)[]",
        "a_bp": "bpchar[]",
        "a_ts": "timestamp with time zone[]",
        "a_bin": "bytea[]",
        "a_jb": "jsonb[]",
        "a_rng": "numrange[]",
        "a_enum": '"MixedArrEnum"[]',
        "a_dom_elem": "integer[]",
        "a_dom_arr": "integer[]",
    }
    for label, sch_map in (("sqlalchemy", sa_schemas), ("dbapi_fallback", raw_schemas)):
        cols = sch_map["arr_all"].columns
        for cname, exp_dt in expected_reflected.items():
            assert cols[cname].data_type == exp_dt, f"{label}: {cname} got {cols[cname].data_type!r}, expected {exp_dt!r}"
        assert sch_map["arr_part"].columns["k"].data_type == "numeric[]", (
            f"{label}: partitioned table array column got {sch_map['arr_part'].columns['k'].data_type!r}"
        )

    # Savepoint safety: an error inside _PG_NATIVE_ARRAY_TYPES_SQL inside an active
    # transaction must roll back only its own savepoint and never abort the caller's tx.
    cur.execute("SELECT 1")
    monkeypatch.setattr(_conn_mod, "_PG_NATIVE_ARRAY_TYPES_SQL", "SELECT * FROM __nonexistent_pg_arr_probe__")
    assert _conn_mod._pg_introspect_native_array_types(env.conn) == {}
    cur.execute("SELECT 42")
    assert cur.fetchone()[0] == 42
    monkeypatch.undo()
    env.conn.commit()

    cur.execute(
        """
        INSERT INTO arr_all VALUES
        (1, 1, 10, '{1,2}', '{{1,2},{3,4}}', '{1.0,NULL,2}', '{1.50,2.25}', '{"a","b"}',
         '{"x"}', '{"short"}', '{"ab "}', '{"bp"}',
         ARRAY['2026-01-01 00:00:00+00'::timestamptz],
         ARRAY['\\x00ff'::bytea], ARRAY['{"a":1.0}'::jsonb], ARRAY['[1.0,2.50)'::numrange],
         ARRAY['alpha'::"MixedArrEnum"], ARRAY[1,2]::pos_int_dom[], ARRAY[10,20]::int_arr_dom),
        (2, 1, 20, '{1,2}', '{{1,2},{3,4}}', '{1,NULL,2.00}', '{1.5,2.25}', '{"a","b"}',
         '{"x"}', '{"short"}', '{"ab "}', '{"bp"}',
         ARRAY['2026-01-01 09:00:00+09'::timestamptz],
         ARRAY['\\x00ff'::bytea], ARRAY['{"a":1}'::jsonb], ARRAY['[1,2.5)'::numrange],
         ARRAY['alpha'::"MixedArrEnum"], ARRAY[1,2]::pos_int_dom[], ARRAY[10,20]::int_arr_dom),
        (3, 2, 30, '{}', '{}', NULL, '{}', NULL,
         '{}', '{}', '{}', '{}',
         '{}', '{}', '{}', '{}',
         '{}', '{}', '{}')
        """
    )
    cur.execute(
        "INSERT INTO arr_part VALUES (1, 1, '{1.0,2}', 10), (2, 2, '{1,2.00}', 25), (3, 1, NULL, 5)"
    )
    env.conn.commit()

    views = [
        ViewDef(
            "v_arr_proj",
            "SELECT id, a_int, a_int2d, a_num, a_num_ps, a_txt, a_vc, a_vc20, a_c3, a_bp, "
            "a_ts, a_bin, a_jb, a_rng, a_enum, a_dom_elem, a_dom_arr FROM arr_all",
        ),
        ViewDef("v_arr_grp_types", "SELECT a_int, a_vc, a_bp, a_ts, a_jb, a_rng, a_enum, COUNT(*) AS n, SUM(v) AS s FROM arr_all GROUP BY a_int, a_vc, a_bp, a_ts, a_jb, a_rng, a_enum"),
        ViewDef("v_arr_dom_grp", "SELECT a_dom_elem, a_dom_arr, COUNT(*) AS n, SUM(v) AS s FROM arr_all GROUP BY a_dom_elem, a_dom_arr"),
        ViewDef(
            "v_arr_expr",
            "SELECT COALESCE(a_num, ARRAY[]::NUMERIC[]) AS k_co, "
            "CASE WHEN g = 1 THEN a_num_ps ELSE a_num END AS k_case_num, "
            "CASE WHEN g = 1 THEN a_vc ELSE ARRAY['x'] END AS k_case_txt, "
            "CASE WHEN g = 1 THEN a_int ELSE a_num END AS k_case_wide, "
            "COALESCE(ARRAY['x'], a_vc) AS k_co_lit, "
            "a_txt COLLATE \"C\" AS k_coll, "
            "a_num[1] AS first_num, a_num[1:2] AS slice_num, "
            "a_int2d[1][1] AS elem_2d, a_int2d[1] AS elem_1d_of_2d, "
            "a_vc[1] AS elem_vc, a_vc20[1] AS elem_vc20, a_c3[1] AS elem_c3, "
            "ARRAY[g, v] AS made_arr, ARRAY[1, '2'] AS mix_arr, ARRAY[NULL] AS null_arr, "
            "COUNT(*) AS n FROM arr_all "
            "GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16",
        ),
        ViewDef(
            "v_arr_minmax",
            "SELECT g, MIN(a_num_ps) AS mn_num, MAX(a_vc20) AS mx_vc, "
            "COALESCE(MIN(a_num_ps), ARRAY[]::NUMERIC[]) AS co_mn "
            "FROM arr_all GROUP BY g",
        ),
        ViewDef("v_arr_part", "SELECT k, COUNT(*) AS n, SUM(v) AS s FROM arr_part GROUP BY k"),
    ]
    env.views = views
    for vd in views:
        env.register(vd)

    def _norm_val(val):
        if isinstance(val, memoryview):
            return bytes(val)
        if isinstance(val, list):
            return [_norm_val(x) for x in val]
        if isinstance(val, tuple):
            return tuple(_norm_val(x) for x in val)
        return val

    def _assert_views_match_and_oids(label: str) -> None:
        # Check exact result-type OIDs on every non-domain array column against the live SELECT.
        domain_cols = {"a_dom_elem", "a_dom_arr"}
        for vd in views:
            cur.execute(f"SELECT * FROM {vd.name} LIMIT 0")
            mv_desc = [(d.name, d.type_code) for d in cur.description]
            cur.execute(f"SELECT * FROM ({vd.select}) AS q LIMIT 0")
            live_desc = [(d.name, d.type_code) for d in cur.description]
            for (mn, moid), (ln, loid) in zip(mv_desc, live_desc):
                assert mn == ln
                if mn not in domain_cols:
                    assert moid == loid, f"{label}: {vd.name}.{mn} OID mismatch MV={moid} LIVE={loid}"
            sel_cols = ", ".join(
                f"{n}::integer[] AS {n}" if n == "a_dom_elem" else n
                for n, _ in mv_desc
            )
            cur.execute(f"SELECT {sel_cols} FROM {vd.name}")
            mv_rows = cur.fetchall()
            cur.execute(f"SELECT {sel_cols} FROM ({vd.select}) AS q")
            live_rows = cur.fetchall()
            assert len(mv_rows) == len(live_rows), f"{label}: {vd.name} row count MV={len(mv_rows)} LIVE={len(live_rows)}"
            if vd.name == "v_arr_proj":
                # Projection rows must preserve exact per-row representation (including display scale).
                mv_repr = sorted(repr(_norm_val(r)) for r in mv_rows)
                live_repr = sorted(repr(_norm_val(r)) for r in live_rows)
                assert mv_repr == live_repr, f"{label}: {vd.name}\n MV:   {mv_repr}\n LIVE: {live_repr}"
            else:
                # Grouped views must match LIVE with exact multiset equality under PostgreSQL's
                # native equality (equal group keys like 1.0 vs 1 form one group; exact stored
                # representative is arbitrary among equal members, just as in _check_key_groups).
                cur.execute(
                    f"(SELECT {sel_cols} FROM {vd.name} EXCEPT ALL SELECT {sel_cols} FROM ({vd.select}) AS q) "
                    f"UNION ALL "
                    f"(SELECT {sel_cols} FROM ({vd.select}) AS q EXCEPT ALL SELECT {sel_cols} FROM {vd.name})"
                )
                diff = cur.fetchall()
                assert diff == [], f"{label}: {vd.name} symmetric diff={diff}\n MV:   {mv_rows}\n LIVE: {live_rows}"
        env.conn.commit()

    _assert_views_match_and_oids("initial")

    # Mutate arrays across INSERT, UPDATE, and DELETE (including empty arrays and NULLs).
    cur.execute(
        """
        INSERT INTO arr_all VALUES
        (4, 2, 40, '{9,8}', '{{9,8}}', '{3.5}', '{3.50}', '{"z"}',
         '{"y"}', '{"long"}', '{"cd "}', '{"bp2"}',
         ARRAY['2026-06-01 00:00:00+00'::timestamptz],
         ARRAY['\\xabcd'::bytea], ARRAY['{"b":2}'::jsonb], ARRAY['[2,4)'::numrange],
         ARRAY['beta'::"MixedArrEnum"], ARRAY[5]::pos_int_dom[], ARRAY[50]::int_arr_dom)
        """
    )
    cur.execute("UPDATE arr_all SET a_num = '{1.000,NULL,2.0}', v = 99 WHERE id = 3")
    cur.execute("DELETE FROM arr_all WHERE id = 1")
    cur.execute("INSERT INTO arr_part VALUES (4, 2, '{1.00,2.0}', 15)")
    cur.execute("UPDATE arr_part SET k = '{}', v = 77 WHERE id = 3")
    cur.execute("DELETE FROM arr_part WHERE id = 1")
    env.conn.commit()

    _assert_views_match_and_oids("after_dml")

    # Verify secondary search_path schema visibility: a table in a secondary schema on
    # search_path (not current_schema()) must also resolve native array types via pg_table_is_visible.
    sec_sch = f"{env.ns}_sec"
    cur.execute(f'CREATE SCHEMA "{sec_sch}"')
    cur.execute(f'CREATE TABLE "{sec_sch}".sec_arr_t (id INT PRIMARY KEY, k NUMERIC(8, 2)[], v INT)')
    cur.execute(f'SET search_path TO "{env.ns}", "{sec_sch}"')
    env.conn.commit()
    try:
        arr_map = _conn_mod._pg_introspect_native_array_types(env.conn)
        assert arr_map.get(("sec_arr_t", "k")) == "numeric(8,2)[]", arr_map
    finally:
        cur.execute(f'SET search_path TO "{env.ns}"')
        cur.execute(f'DROP SCHEMA IF EXISTS "{sec_sch}" CASCADE')
        env.conn.commit()
