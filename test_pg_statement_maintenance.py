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
# Concurrent writers of different sources of a PostgreSQL join view
# ---------------------------------------------------------------------------


def test_join_view_writers_of_different_sources_read_committed(make_env) -> None:
    """READ COMMITTED: concurrent writers of different sources of an algebraic
    join view serialize on the view's lock row before joining their delta with
    the other source, so the second writer's join takes a fresh snapshot that
    includes the first writer's commit."""
    env = make_env(views=[VIEW_BY_NAME["v_j"]], seed=True)
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (3000, 1, 5)")
    worker = _Worker(b, ["UPDATE d SET name = 'n2' WHERE g = 1"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's view lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("concurrent writers of t and d (t first)")

    # Reverse direction: d writes first, t inserts into the renamed join key concurrently.
    a.cursor().execute("UPDATE d SET name = 'n3' WHERE g = 1")
    worker = _Worker(b, ["INSERT INTO t (id, g, v) VALUES (3001, 1, 7)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits for A's view lock (d first)")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("concurrent writers of d and t (d first)")


def test_join_view_empty_join_race_and_truncate_read_committed(make_env) -> None:
    """READ COMMITTED: each algebraic join shape tested in isolation (no
    multi-view trigger masking): empty-join race, join-key UPDATE + DELETE/INSERT
    races, MIN/AVG recompute join, 3-table join with disjoint join keys,
    concurrent TRUNCATE + INSERT, and lock-table drop lifecycle."""
    # 1. Standalone v_j: empty-join race, concurrent join-key UPDATE, DELETE + INSERT,
    #    and lock-table lifecycle.
    env = make_env(views=[VIEW_BY_NAME["v_j"]], seed=True)
    assert _strategy(env, "v_j") == "algebraic_delta"
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)

    a.cursor().execute("INSERT INTO t (id, g, v, m) VALUES (9901, 99, 13, 1.250)")
    worker = _Worker(b, ["INSERT INTO d (g, name) VALUES (99, 'n99')"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "empty-join B waits for A")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("empty-join race (v_j alone)")

    # Concurrent join-key UPDATE on t + rename on d.
    a.cursor().execute("UPDATE t SET g = 2 WHERE id = 9901")
    worker = _Worker(b, ["UPDATE d SET name = 'n2_upd' WHERE g = 2"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "join-key UPDATE B waits for A")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("concurrent join-key UPDATE on t and rename on d")

    # Concurrent DELETE on d + INSERT into t for the deleted join key.
    a.cursor().execute("DELETE FROM d WHERE g = 99")
    worker = _Worker(b, ["INSERT INTO t (id, g, v, m) VALUES (9904, 99, 5, 1.000)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "DELETE+INSERT B waits for A")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("concurrent DELETE on d and INSERT into t")

    lock_tbl = env.plans["v_j"].auxiliary_table_names[0]
    cur = env.conn.cursor()
    cur.execute("SELECT to_regclass(%s)", (f'"{env.ns}"."{lock_tbl}"',))
    assert cur.fetchone()[0] is not None
    env.conn.commit()
    env.manager.drop_view("v_j")
    env.views = []
    cur.execute("SELECT to_regclass(%s)", (f'"{env.ns}"."{lock_tbl}"',))
    assert cur.fetchone()[0] is None
    env.conn.commit()

    # 2. Standalone algebraic join with MIN/AVG recompute (no other view on t/d).
    v_j_rec = ViewDef(
        "v_j_rec",
        "SELECT d.name, MIN(t.v) AS mn, AVG(t.m) AS av, SUM(t.v) AS s, COUNT(*) AS c "
        "FROM t JOIN d ON t.g = d.g GROUP BY d.name",
    )
    env_rec = make_env(views=[v_j_rec], seed=True)
    assert _strategy(env_rec, "v_j_rec") == "algebraic_delta"
    ar, br, mon_r = env_rec.connect(), env_rec.connect(), env_rec.connect()
    sid_ar, sid_br = _session_id(env_rec, ar), _session_id(env_rec, br)
    ar.cursor().execute("DELETE FROM t WHERE g = 1 AND v = (SELECT MIN(v) FROM t WHERE g = 1)")
    worker = _Worker(br, ["UPDATE d SET name = 'n2' WHERE g = 1"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env_rec, mon_r, sid_br, sid_ar), worker.done, "recompute B waits for A")
    finally:
        ar.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env_rec.check("standalone algebraic join MIN/AVG recompute + concurrent d update")

    # 3. Standalone 3-table join (disjoint join keys) + concurrent TRUNCATE.
    schema3 = list(PG_SCHEMA) + [
        "CREATE TABLE e (name VARCHAR(20) PRIMARY KEY, tag VARCHAR(20) NOT NULL)",
    ]
    v_j3 = ViewDef(
        "v_j3",
        "SELECT e.tag, SUM(t.v) AS s, COUNT(*) AS c "
        "FROM t JOIN d ON t.g = d.g JOIN e ON d.name = e.name GROUP BY e.tag",
    )
    env3 = make_env(views=[], schema_sql=schema3, seed=True)
    env3.execute("INSERT INTO e (name, tag) VALUES ('n0', 't0'), ('n1', 't1'), ('n2', 't2')")
    env3.execute("INSERT INTO d (g, name) VALUES (99, 'n99')")
    env3.commit()
    env3.views = [v_j3]
    env3.register(v_j3)
    assert _strategy(env3, "v_j3") == "algebraic_delta"
    a3, b3, mon3 = env3.connect(), env3.connect(), env3.connect()
    sid_a3, sid_b3 = _session_id(env3, a3), _session_id(env3, b3)

    a3.cursor().execute("INSERT INTO t (id, g, v, m) VALUES (9902, 99, 17, 2.750)")
    worker = _Worker(b3, ["INSERT INTO e (name, tag) VALUES ('n99', 't99')"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env3, mon3, sid_b3, sid_a3), worker.done, "3-table B waits for A")
    finally:
        a3.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env3.check("standalone 3-table join concurrent writers of t and e")

    a3.cursor().execute("INSERT INTO t (id, g, v, m) VALUES (9903, 2, 11, 3.000)")
    worker = _Worker(b3, ["TRUNCATE e"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env3, mon3, sid_b3, sid_a3), worker.done, "TRUNCATE waits for A's view lock")
    finally:
        a3.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env3.check("standalone 3-table join concurrent INSERT into t and TRUNCATE e")


@pytest.mark.parametrize(
    "vd",
    [
        ViewDef(
            "v_lj",
            "SELECT d.name, SUM(t.v) AS s, COUNT(t.v) AS c FROM d LEFT JOIN t ON t.g = d.g GROUP BY d.name",
        ),
        ViewDef(
            "v_j_str",
            "SELECT d.name, STRING_AGG(CAST(t.id AS TEXT), ',' ORDER BY t.id) AS ids "
            "FROM t INNER JOIN d ON t.g = d.g GROUP BY d.name",
        ),
        ViewDef(
            "v_fj",
            "SELECT COALESCE(d.name, '__unmatched__') AS grp, COUNT(t.id) AS c, SUM(t.v) AS s "
            "FROM d FULL OUTER JOIN t ON t.g = d.g GROUP BY COALESCE(d.name, '__unmatched__')",
        ),
    ],
    ids=lambda v: v.name,
)
def test_group_scoped_join_writers_of_different_sources(make_env, vd: ViewDef) -> None:
    """Each group-scoped (Tier 2) join view tested in isolation (no cross-view
    trigger masking): LEFT JOIN, INNER JOIN STRING_AGG, and FULL OUTER JOIN
    serialize concurrent writers across sources before computing affected keys."""
    env = make_env(views=[vd], seed=True)
    assert _strategy(env, vd.name) == "group_scoped_delta", (vd.name, _strategy(env, vd.name))

    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)

    # 1. Empty-join race (d then t): A inserts d(99, 'n99'), B concurrently inserts t(9900, 99, 42).
    a.cursor().execute("INSERT INTO d (g, name) VALUES (99, 'n99')")
    worker = _Worker(b, ["INSERT INTO t (id, g, v) VALUES (9900, 99, 42)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, f"{vd.name} d->t B waits for A")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check(f"{vd.name} empty-join race (d then t)")

    # 2. Reverse empty-join race (t then d): A inserts t(9905, 101, 19), B concurrently inserts d(101, 'n101').
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (9905, 101, 19)")
    worker = _Worker(b, ["INSERT INTO d (g, name) VALUES (101, 'n101')"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, f"{vd.name} t->d B waits for A")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check(f"{vd.name} empty-join race (t then d)")

    # 3. Rename race: A updates d.name while B inserts into t on the same join key.
    a.cursor().execute("UPDATE d SET name = 'n99_renamed' WHERE g = 99")
    worker = _Worker(b, ["INSERT INTO t (id, g, v) VALUES (9901, 99, 8)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, f"{vd.name} rename B waits for A")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check(f"{vd.name} rename race")

    # 4. REPEATABLE READ: concurrent writer of the other source fails closed with 40001.
    cb = b.cursor()
    cb.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
    cb.execute("SELECT COUNT(*) FROM d")
    cb.fetchall()
    a.cursor().execute("INSERT INTO d (g, name) VALUES (100, 'n100')")
    a.commit()
    with pytest.raises(Exception) as excinfo:
        cb.execute("INSERT INTO t (id, g, v) VALUES (9902, 100, 5)")
        b.commit()
    b.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)
    env.check(f"{vd.name} REPEATABLE READ fail closed")


def test_multi_view_join_trigger_order_does_not_deadlock(make_env) -> None:
    """Multiple join views sharing two sources must acquire their per-view lock
    rows in the same order regardless of which source table is written, even
    when view names sit on an underscore/digit boundary ('v' vs 'v_' on
    '0_orders' vs 'dim') or share a long truncated prefix."""
    schema = [
        'CREATE TABLE "0_orders" (id INT PRIMARY KEY, g INT NOT NULL, v INT NOT NULL)',
        "CREATE TABLE dim (g INT PRIMARY KEY, name VARCHAR(20) NOT NULL)",
    ]
    views = [
        ViewDef("v", 'SELECT dim.name, COUNT(*) AS c FROM "0_orders" JOIN dim ON "0_orders".g = dim.g GROUP BY dim.name'),
        ViewDef("v_", 'SELECT dim.name, SUM("0_orders".v) AS s FROM "0_orders" JOIN dim ON "0_orders".g = dim.g GROUP BY dim.name'),
        ViewDef(
            "very_long_shared_prefix_for_join_view_alpha_one",
            'SELECT dim.name, COUNT(*) AS c FROM "0_orders" JOIN dim ON "0_orders".g = dim.g GROUP BY dim.name',
        ),
        ViewDef(
            "very_long_shared_prefix_for_join_view_alpha_two",
            'SELECT dim.name, SUM("0_orders".v) AS s FROM "0_orders" JOIN dim ON "0_orders".g = dim.g GROUP BY dim.name',
        ),
    ]
    env = make_env(views=[], schema_sql=schema, seed=False)
    env.execute("INSERT INTO dim VALUES (1, 'n1'), (2, 'n2')")
    env.execute('INSERT INTO "0_orders" VALUES (1, 1, 10), (2, 2, 20)')
    env.commit()
    env.views = views
    for vd in views:
        env.register(vd)

    # Verify PostgreSQL's own pg_trigger.tgname firing order is identical on both sources.
    trig_to_view = {t.lower(): vd.name for vd in views for t in env.plans[vd.name].trigger_names}
    cur = env.conn.cursor()
    cur.execute(
        "SELECT t.tgname FROM pg_trigger t "
        "WHERE t.tgrelid = '\"0_orders\"'::regclass AND NOT t.tgisinternal AND t.tgname LIKE '%%_ai' "
        "ORDER BY t.tgname"
    )
    ord_orders = [trig_to_view[r[0].lower()] for r in cur.fetchall()]
    cur.execute(
        "SELECT t.tgname FROM pg_trigger t "
        "WHERE t.tgrelid = 'dim'::regclass AND NOT t.tgisinternal AND t.tgname LIKE '%%_au' "
        "ORDER BY t.tgname"
    )
    ord_dim = [trig_to_view[r[0].lower()] for r in cur.fetchall()]
    assert len(ord_orders) == 4 and ord_orders == ord_dim, (ord_orders, ord_dim)
    env.conn.commit()

    # Run concurrent opposite-source writers repeatedly; consistent lock order
    # means zero deadlocks (40P01) and exact results across all 4 views.
    a, b = env.connect(), env.connect()
    for step in range(8):
        start_evt = threading.Event()
        errs: List[Optional[BaseException]] = [None, None]

        def run_a(idx: int = step) -> None:
            start_evt.wait()
            try:
                a.cursor().execute(f'INSERT INTO "0_orders" VALUES ({100 + idx}, 1, {idx + 1})')
                a.commit()
            except BaseException as exc:
                a.rollback()
                errs[0] = exc

        def run_b(idx: int = step) -> None:
            start_evt.wait()
            try:
                b.cursor().execute(f"UPDATE dim SET name = 'n1_{idx}' WHERE g = 1")
                b.commit()
            except BaseException as exc:
                b.rollback()
                errs[1] = exc

        ta = threading.Thread(target=run_a, daemon=True)
        tb = threading.Thread(target=run_b, daemon=True)
        ta.start()
        tb.start()
        start_evt.set()
        ta.join(30)
        tb.join(30)
        assert errs == [None, None], f"step {step} failed (possible deadlock): {errs!r}"
    env.check("multi-view concurrent opposite-source writers")


def test_repair_view_reseeds_lock_row_for_repeatable_read(make_env) -> None:
    """repair_view() deletes auxiliary tables; it must re-seed the lock rows
    (0, and 1..PG_GROUP_LOCK_BUCKETS for group-scoped views) in
    __matview_lock_<v> so REPEATABLE READ writers of subquery-filtered Tier 2
    views still fail closed with 40001 after a repair, and algebraic join views
    remain serialized after a repair."""
    schema = list(PG_SCHEMA) + ["CREATE TABLE u (g INT PRIMARY KEY)"]
    v_sub = ViewDef(
        "v_sub",
        "SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids FROM t "
        "WHERE g IN (SELECT g FROM u) GROUP BY g",
    )
    env = make_env(views=[], schema_sql=schema, seed=True)
    env.execute("INSERT INTO u (g) VALUES (1), (2), (3)")
    env.commit()
    env.views = [VIEW_BY_NAME["v_j"], v_sub]
    for vd in env.views:
        env.register(vd)

    # Damage both backing tables and repair them.
    for vd in env.views:
        env.execute(f"DELETE FROM {env.plans[vd.name].backing_table_name}")
    env.commit()
    for vd in env.views:
        env.manager.repair_view(vd.name)
    env.commit()

    # Verify (0, 0) seed row was restored in both lock tables.
    cur = env.conn.cursor()
    for vd in env.views:
        lock_t = env.plans[vd.name].auxiliary_table_names[0]
        cur.execute(f"SELECT COUNT(*) FROM {lock_t} WHERE __mv_id = 0")
        assert cur.fetchone()[0] == 1, (vd.name, lock_t)
    env.conn.commit()

    # Under REPEATABLE READ after repair_view(): B takes snapshot, A updates u
    # (locks all buckets 0..PG_GROUP_LOCK_BUCKETS) and commits, B writes t
    # (updates its bucket row) -> must fail closed with 40001 rather than
    # seeing an empty lock table.
    a, b = env.connect(), env.connect()
    cb = b.cursor()
    cb.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
    cb.execute("SELECT COUNT(*) FROM t")
    cb.fetchall()
    a.cursor().execute("INSERT INTO u (g) VALUES (4)")
    a.commit()
    with pytest.raises(Exception) as excinfo:
        cb.execute("INSERT INTO t (id, g, v) VALUES (8801, 4, 10)")
        b.commit()
    b.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)
    env.check("post-repair REPEATABLE READ fail closed on v_sub and v_j")


@pytest.mark.parametrize("isolation", ["REPEATABLE READ", "SERIALIZABLE"])
def test_join_view_writers_of_different_sources_repeatable_read_fail_closed(make_env, isolation: str) -> None:
    env = make_env(views=[VIEW_BY_NAME["v_j"]], seed=True)
    a, b = env.connect(), env.connect()
    cb = b.cursor()
    cb.execute(f"BEGIN ISOLATION LEVEL {isolation}")
    cb.execute("SELECT COUNT(*) FROM t")
    cb.fetchall()
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (3000, 1, 5)")
    a.commit()
    with pytest.raises(Exception) as excinfo:
        cb.execute("UPDATE d SET name = 'n2' WHERE g = 1")
        b.commit()
    b.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)
    env.check(f"{isolation} writer of the other source")

    # Also fail closed on the empty-join case (neither t nor d initially had g=99,
    # so A's insert produces an empty Δt ⋈ d and only touches the view lock row).
    cb.execute(f"BEGIN ISOLATION LEVEL {isolation}")
    cb.execute("SELECT COUNT(*) FROM t")
    cb.fetchall()
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (3009, 99, 5)")
    a.commit()
    with pytest.raises(Exception) as excinfo:
        cb.execute("INSERT INTO d (g, name) VALUES (99, 'n99')")
        b.commit()
    b.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)
    env.check(f"{isolation} empty-join writer of the other source")


def test_whole_view_recompute_excludes_group_writers(make_env) -> None:
    """A group-scoped view whose WHERE reads another table: writes to t
    recompute their groups (locking bounded group buckets), writes to u
    recompute the whole view (locking all buckets 0..PG_GROUP_LOCK_BUCKETS).
    The whole-view writer must wait for the group writer and then see it."""
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
    "nummultirange": (
        "NUMMULTIRANGE",
        [["'{[1.0,2.50),[4.00,5)}'::nummultirange", "'{[1,2.5),[4,5.0)}'::nummultirange"], ["'{}'::nummultirange"]],
        "'{[1,3)}'::nummultirange",
        "(SELECT COALESCE(array_agg(numrange(trim_scale(lower(r)), trim_scale(upper(r)), "
        "(CASE WHEN lower_inc(r) THEN '[' ELSE '(' END) || (CASE WHEN upper_inc(r) THEN ']' ELSE ')' END)) ORDER BY o), '{{}}') "
        "FROM unnest({k}) WITH ORDINALITY u(r, o))::text",
    ),
    "tstzmultirange": (
        "TSTZMULTIRANGE",
        [["'{[2026-01-01 00:00:00+00,2026-01-01 02:00:00+00)}'::tstzmultirange",
          "'{[2026-01-01 09:00:00+09,2026-01-01 11:00:00+09)}'::tstzmultirange"]],
        "'{[2026-01-02 00:00:00+00,2026-01-02 01:00:00+00)}'::tstzmultirange",
        "(SELECT COALESCE(array_agg((lower(r) AT TIME ZONE 'UTC')::text || '..' || (upper(r) AT TIME ZONE 'UTC')::text ORDER BY o), '{{}}') "
        "FROM unnest({k}) WITH ORDINALITY u(r, o))::text",
    ),
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
                                       ("interval", "ck_str"), ("nummultirange", "ck_str"), ("nummultirange", "ck_agg")])
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
    """Unsupported grouping key types (such as anonymous ``record`` via
    ``GROUP BY ROW(...)``) and composite/domain/array keys containing a
    nondeterministic collation are refused at registration time (0A000) even
    when the source table is empty."""
    env = make_env(views=[], seed=False, schema_sql=[
        "CREATE TABLE cp (id INT PRIMARY KEY, a INT, v INT)",
    ])
    # 1. Anonymous `record` grouping key (including positional GROUP BY 1 and
    #    SELECT DISTINCT): refused at registration, not at first write.
    rec_views = [
        ViewDef("v_rec_grp", "SELECT COUNT(*) AS n, SUM(v) AS s FROM cp GROUP BY ROW(a, v)"),
        ViewDef("v_rec_pos", "SELECT ROW(a, v) AS r, COUNT(*) AS n FROM cp GROUP BY 1"),
        ViewDef("v_rec_dst", "SELECT DISTINCT ROW(a, v) AS r FROM cp"),
        ViewDef("v_rec_win", "SELECT id, ROW_NUMBER() OVER (PARTITION BY ROW(a, v) ORDER BY id) AS rn FROM cp"),
    ]
    for vd in rec_views:
        with pytest.raises(Exception) as excinfo:
            env.register(vd)
        env.conn.rollback()
        assert _pgcode(excinfo.value) == "0A000", (vd.name, repr(excinfo.value))
        assert "record" in str(excinfo.value).lower(), (vd.name, str(excinfo.value))

    # 2. Composite / nested composite / domain / array / custom range containing
    #    a nondeterministic collation (including output-alias shadowing): refused
    #    at registration with 0A000.
    cur = env.conn.cursor()
    try:
        cur.execute(
            "CREATE COLLATION mv_test_ci_comp (provider = icu, locale = 'und-u-ks-level2', deterministic = false)"
        )
        env.conn.commit()
    except Exception as exc:
        env.conn.rollback()
        pytest.skip(f"ICU nondeterministic collation unavailable: {exc}")

    cur.execute("CREATE TYPE ci_leaf AS (n INT, s TEXT COLLATE mv_test_ci_comp)")
    cur.execute("CREATE TYPE ci_outer AS (id INT, leaf ci_leaf)")
    cur.execute("CREATE DOMAIN ci_dom AS ci_leaf")
    cur.execute("CREATE TYPE ci_range AS RANGE (subtype = text, collation = mv_test_ci_comp)")
    cur.execute(
        "CREATE TABLE tc_comp (id INT PRIMARY KEY, c1 ci_leaf, c2 ci_outer, c3 ci_dom, "
        "c4 ci_leaf[], c5 ci_range, c6 ci_multirange, ok INT, v INT)"
    )
    env.conn.commit()

    for col in ("c1", "c2", "c3", "c4", "c5", "c6"):
        vd = ViewDef(f"v_ci_{col}", f"SELECT {col}, COUNT(*) AS n FROM tc_comp GROUP BY {col}")
        with pytest.raises(Exception) as excinfo:
            env.register(vd)
        env.conn.rollback()
        assert _pgcode(excinfo.value) == "0A000", (col, repr(excinfo.value))
        assert "mv_test_ci_comp" in str(excinfo.value), (col, str(excinfo.value))

    # Output alias `'ok' AS c1` does NOT shadow base column `tc_comp.c1` in PostgreSQL GROUP BY.
    vd_shadow = ViewDef("v_ci_shadow", "SELECT 'ok'::TEXT AS c1, COUNT(*) AS n FROM tc_comp GROUP BY c1")
    with pytest.raises(Exception) as excinfo_sh:
        env.register(vd_shadow)
    env.conn.rollback()
    assert _pgcode(excinfo_sh.value) == "0A000", repr(excinfo_sh.value)
    assert "mv_test_ci_comp" in str(excinfo_sh.value), str(excinfo_sh.value)


def test_composite_and_multirange_canonical_keys(make_env, monkeypatch) -> None:
    """Field-wise composite canonicalization and range-wise multirange
    canonicalization (including nested/quoted/empty/dropped-attribute
    composites, NULL vs ROW(NULL,...), domains over composites, 1D/2D arrays
    of composites, custom float multiranges, concurrent writers, and native
    type/OID preservation via both SQLAlchemy and raw DB-API introspection)."""
    import matview.connectivity as _conn_mod

    env = make_env(
        views=[],
        seed=False,
        schema_sql=[
            'CREATE TYPE comp_pair AS (a NUMERIC, b FLOAT8, "c d" TEXT)',
            'CREATE TYPE "MixedComp" AS (x INT, y comp_pair)',
            "CREATE TYPE empty_comp AS ()",
            "CREATE TYPE dropped_comp AS (keep_a NUMERIC, drop_me TEXT, keep_b FLOAT8)",
            "ALTER TYPE dropped_comp DROP ATTRIBUTE drop_me",
            "CREATE DOMAIN comp_dom AS comp_pair",
            "CREATE TYPE floatrange AS RANGE (subtype = float8)",
            "CREATE TYPE comp_with_mr AS (tag TEXT, mr NUMMULTIRANGE)",
            """
            CREATE TABLE cm (
                id INT PRIMARY KEY,
                k_comp comp_pair,
                k_mcomp "MixedComp",
                k_empty empty_comp,
                k_drop dropped_comp,
                k_dom comp_dom,
                k_dom_arr comp_dom[],
                k_arr comp_pair[],
                k_arr2d comp_pair[][],
                k_fmr floatmultirange,
                k_cmr comp_with_mr,
                v INT
            )
            """,
        ],
    )

    # 1. Verify both SQLAlchemy and raw DB-API introspection preserve native
    #    composite, domain-over-composite, and custom multirange DDL types.
    sa_schemas = env.manager.fetch_schema()
    monkeypatch.setattr(_conn_mod, "_can_use_sqlalchemy_inspector", lambda _c: False)
    raw_schemas = env.manager.fetch_schema()
    monkeypatch.undo()
    expected_types = {
        "k_comp": "comp_pair",
        "k_mcomp": '"MixedComp"',
        "k_empty": "empty_comp",
        "k_drop": "dropped_comp",
        "k_dom": "comp_pair",
        "k_dom_arr": "comp_pair[]",
        "k_arr": "comp_pair[]",
        "k_arr2d": "comp_pair[]",
        "k_fmr": "floatmultirange",
        "k_cmr": "comp_with_mr",
    }
    for label, sch_map in (("sqlalchemy", sa_schemas), ("dbapi_fallback", raw_schemas)):
        cols = sch_map["cm"].columns
        for cname, exp_dt in expected_types.items():
            assert cols[cname].data_type == exp_dt, f"{label}: {cname} got {cols[cname].data_type!r}, expected {exp_dt!r}"

    views = [
        ViewDef("v_comp_agg", "SELECT k_comp, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_comp"),
        ViewDef("v_comp_str", "SELECT k_comp, STRING_AGG(CAST(v AS TEXT), ',' ORDER BY v) AS vs FROM cm GROUP BY k_comp"),
        ViewDef("v_comp_minmax", "SELECT k_comp, MIN(v) AS lo, MAX(v) AS hi, AVG(v) AS av FROM cm GROUP BY k_comp"),
        ViewDef(
            "v_comp_cnt",
            "SELECT (v % 2) AS parity, COUNT(k_comp) AS c_all, COUNT(DISTINCT k_comp) AS c_dist, "
            "COUNT(k_fmr) AS c_fmr, COUNT(DISTINCT k_fmr) AS c_fmr_dist FROM cm GROUP BY (v % 2)",
        ),
        ViewDef("v_comp_proj", "SELECT k_comp, v FROM cm UNION ALL SELECT k_comp, v + 1000 FROM cm"),
        ViewDef("v_mcomp", "SELECT k_mcomp, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_mcomp"),
        ViewDef("v_empty", "SELECT k_empty, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_empty"),
        ViewDef("v_drop", "SELECT k_drop, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_drop"),
        ViewDef("v_dom", "SELECT k_dom, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_dom"),
        ViewDef("v_dom_arr", "SELECT k_dom_arr, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_dom_arr"),
        ViewDef("v_arr", "SELECT k_arr, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_arr"),
        ViewDef("v_arr2d", "SELECT k_arr2d, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_arr2d"),
        ViewDef("v_fmr", "SELECT k_fmr, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_fmr"),
        ViewDef("v_cmr", "SELECT k_cmr, COUNT(*) AS n, SUM(v) AS s FROM cm GROUP BY k_cmr"),
        ViewDef("v_comp_dst", "SELECT DISTINCT k_comp FROM cm"),
    ]
    env.views = views
    for vd in views:
        env.register(vd)

    cur = env.conn.cursor()

    def _assert_all_exact(label: str) -> None:
        for vd in views:
            if vd.name != "v_comp_proj":
                cur.execute(f"SELECT COUNT(*) FROM {env.plans[vd.name].backing_table_name}")
                stored = cur.fetchone()[0]
                cur.execute(f"SELECT COUNT(*) FROM ({vd.select}) AS q")
                live_cnt = cur.fetchone()[0]
                assert stored == live_cnt, f"{label}: {vd.name} stored={stored} live={live_cnt}"
            cur.execute(f"SELECT * FROM {vd.name} LIMIT 0")
            mv_oids = [d.type_code for d in cur.description]
            cols = [d[0] for d in cur.description]
            cur.execute(f"SELECT * FROM ({vd.select}) AS q LIMIT 0")
            live_oids = [d.type_code for d in cur.description]
            # Domain columns intentionally unwrap to their base type in backing tables.
            if vd.name not in ("v_dom", "v_dom_arr"):
                assert mv_oids == live_oids, f"{label}: {vd.name} OIDs {mv_oids} != {live_oids}"
            sel_cols = ", ".join(cols)
            cur.execute(
                f"(SELECT {sel_cols} FROM {vd.name} EXCEPT ALL SELECT {sel_cols} FROM ({vd.select}) AS q) "
                f"UNION ALL "
                f"(SELECT {sel_cols} FROM ({vd.select}) AS q EXCEPT ALL SELECT {sel_cols} FROM {vd.name})"
            )
            diff = cur.fetchall()
            assert diff == [], f"{label}: {vd.name} diff={diff}"
        env.conn.commit()

    # Insert equal spellings across SESSION_PROFILES plus adversarial distinct values:
    # - ROW(1.0, -0.0, 'hi') == ROW(1.00, 0.0, 'hi') -> MUST merge into 1 group
    # - ROW(1.0, NULL, 'hi') == ROW(1.00, NULL, 'hi') -> partially-NULL composite
    #   (both IS NULL and IS NOT NULL are FALSE in PG; must NOT be dropped in Tier 2,
    #    MIN/MAX/AVG recomputes, COUNT(k_comp), COUNT(DISTINCT k_comp), or Z-set PKs!)
    # - NULL::comp_pair vs ROW(NULL, NULL, NULL)::comp_pair -> MUST stay 2 distinct groups
    # - ARRAY[ROW(1.0, 2.0, 'hi')] vs ARRAY[ROW(1.0, 2.0, 'DIFF')] -> MUST stay 2 distinct groups
    # - floatmultirange '{[-0.0, 1.0)}' == '{[0.0, 1.0)}' and '{}' empty multirange
    for s in SESSION_PROFILES[0]:
        cur.execute(s)
    cur.execute("""
        INSERT INTO cm VALUES
        (1, ROW(1.0, -0.0, 'hi')::comp_pair, ROW(7, ROW(1.0, -0.0, NULL)::comp_pair)::"MixedComp",
            ROW()::empty_comp, ROW(2.50, -0.0)::dropped_comp, ROW(1.0, -0.0, 'd')::comp_dom,
            ARRAY[ROW(1.0, -0.0, 'd')::comp_dom, NULL],
            ARRAY[ROW(1.0, -0.0, 'hi')::comp_pair, NULL],
            ARRAY[ARRAY[ROW(1.0, -0.0, 'a')::comp_pair, NULL], ARRAY[ROW(2.0, 3.0, 'b')::comp_pair, ROW(NULL, NULL, NULL)::comp_pair]],
            '{[-0.0,1.0)}'::floatmultirange, ROW('t', '{[1.0,2.50)}'::nummultirange)::comp_with_mr, 10),
        (2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 20),
        (3, ROW(NULL, NULL, NULL)::comp_pair, ROW(NULL, NULL)::"MixedComp",
            ROW()::empty_comp, ROW(NULL, NULL)::dropped_comp, ROW(NULL, NULL, NULL)::comp_dom,
            ARRAY[ROW(1.0, -0.0, 'DIFF')::comp_dom, NULL],
            ARRAY[ROW(1.0, -0.0, 'DIFF')::comp_pair, NULL],
            ARRAY[ARRAY[ROW(1.0, -0.0, 'DIFF')::comp_pair, NULL], ARRAY[ROW(2.0, 3.0, 'b')::comp_pair, ROW(NULL, NULL, NULL)::comp_pair]],
            '{}'::floatmultirange, ROW('t', '{}'::nummultirange)::comp_with_mr, 31),
        (5, ROW(1.0, NULL, 'hi')::comp_pair, ROW(8, ROW(1.0, NULL, 'hi')::comp_pair)::"MixedComp",
            ROW()::empty_comp, ROW(2.50, NULL)::dropped_comp, ROW(1.0, NULL, 'd')::comp_dom,
            ARRAY[ROW(1.0, NULL, 'd')::comp_dom],
            ARRAY[ROW(1.0, NULL, 'hi')::comp_pair],
            ARRAY[ARRAY[ROW(1.0, NULL, 'a')::comp_pair]],
            '{[2.0,3.0)}'::floatmultirange, ROW('p', '{[1.0,2.50)}'::nummultirange)::comp_with_mr, 51)
    """)
    env.conn.commit()
    _assert_all_exact("after initial insert")

    for s in SESSION_PROFILES[1]:
        cur.execute(s)
    cur.execute("""
        INSERT INTO cm VALUES
        (4, ROW(1.00, 0.0, 'hi')::comp_pair, ROW(7, ROW(1, 0.0, NULL)::comp_pair)::"MixedComp",
            ROW()::empty_comp, ROW(2.5, 0.0)::dropped_comp, ROW(1, 0.0, 'd')::comp_dom,
            ARRAY[ROW(1, 0.0, 'd')::comp_dom, NULL],
            ARRAY[ROW(1, 0.0, 'hi')::comp_pair, NULL],
            ARRAY[ARRAY[ROW(1, 0.0, 'a')::comp_pair, NULL], ARRAY[ROW(2, 3.0, 'b')::comp_pair, ROW(NULL, NULL, NULL)::comp_pair]],
            '{[0.0,1.0)}'::floatmultirange, ROW('t', '{[1,2.5)}'::nummultirange)::comp_with_mr, 40),
        (6, ROW(1.00, NULL, 'hi')::comp_pair, ROW(8, ROW(1, NULL, 'hi')::comp_pair)::"MixedComp",
            ROW()::empty_comp, ROW(2.5, NULL)::dropped_comp, ROW(1, NULL, 'd')::comp_dom,
            ARRAY[ROW(1, NULL, 'd')::comp_dom],
            ARRAY[ROW(1, NULL, 'hi')::comp_pair],
            ARRAY[ARRAY[ROW(1, NULL, 'a')::comp_pair]],
            '{[2.0,3.0)}'::floatmultirange, ROW('p', '{[1,2.5)}'::nummultirange)::comp_with_mr, 61)
    """)
    env.conn.commit()
    _assert_all_exact("after equal-spelling merge (including partially-NULL composite)")

    # Verify NULL vs ROW(NULL,NULL,NULL) vs ROW(1,NULL,'hi') vs ROW(1,0,'hi') = 4 distinct groups!
    cur.execute("SELECT COUNT(*) FROM v_comp_agg")
    assert cur.fetchone()[0] == 4
    cur.execute("SELECT COUNT(*) FROM v_comp_str")
    assert cur.fetchone()[0] == 4
    cur.execute("SELECT COUNT(*) FROM v_comp_minmax")
    assert cur.fetchone()[0] == 4
    # Verify composite arrays differing only in field 3 stayed distinct (4 groups: hi, DIFF, partial-null, NULL).
    cur.execute("SELECT COUNT(*) FROM v_arr")
    assert cur.fetchone()[0] == 4
    cur.execute("SELECT COUNT(*) FROM v_arr2d")
    assert cur.fetchone()[0] == 4
    cur.execute("SELECT COUNT(*) FROM v_dom_arr")
    assert cur.fetchone()[0] == 4
    env.conn.commit()

    # Verify _pg_key_text_fn works even when called under a hostile search_path (SET search_path FROM CURRENT).
    key_fn = next(f.split("(")[0] for f in env.plans["v_comp_agg"].function_names if f.startswith("__matview_key_"))
    cur.execute("SET search_path TO pg_catalog")
    cur.execute(
        f'SELECT "{env.ns}".{key_fn}(ROW(1.0, -0.0, \'hi\')::"{env.ns}".comp_pair) '
        f'     = "{env.ns}".{key_fn}(ROW(1, 0.0, \'hi\')::"{env.ns}".comp_pair)'
    )
    assert cur.fetchone()[0] is True
    cur.execute(f'SET search_path TO "{env.ns}"')
    env.conn.commit()

    # Concurrent writers inserting different spellings of the same composite key must serialize on one lock.
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("INSERT INTO cm (id, k_comp, v) VALUES (10, ROW(1.000, -0.0, 'hi')::comp_pair, 100)")
    worker = _Worker(
        b, ["INSERT INTO cm (id, k_comp, v) VALUES (11, ROW(1, 0.0, 'hi')::comp_pair, 110)"]
    ).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "B waits on composite group lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    a.close()
    b.close()
    monitor.close()
    _assert_all_exact("after concurrent composite writers")

    # UPDATE and DELETE retractions including retracting MIN/MAX extremes on partially-NULL & all-NULL composites.
    cur.execute("DELETE FROM cm WHERE id = 5")  # retracts MIN(v)=51 from partially-NULL group ROW(1,NULL,'hi')
    cur.execute("UPDATE cm SET k_comp = ROW(1, 0.0, 'hi')::comp_pair, v = v + 1 WHERE id = 3")
    cur.execute("DELETE FROM cm WHERE id IN (1, 2, 4, 10, 11)")
    env.conn.commit()
    _assert_all_exact("after extreme retraction, update, and delete")

    cur.execute("DELETE FROM cm")
    env.conn.commit()
    _assert_all_exact("after delete all")

    # Verify non-default array lower bounds ('[2:3]={1,2}' vs '{1,2}') remain distinct,
    # and schema-qualified quoted composite types preserve quoting in COALESCE().
    sec_sch = f"{env.ns}_comp_sec"
    cur.execute(f'CREATE SCHEMA "{sec_sch}"')
    cur.execute(f'CREATE TYPE "{sec_sch}"."SecMixedComp" AS (a NUMERIC, b TEXT)')
    cur.execute(
        f'CREATE TABLE cm_extra (id INT PRIMARY KEY, ka INT[], kc1 "{sec_sch}"."SecMixedComp", kc2 "{sec_sch}"."SecMixedComp", v INT)'
    )
    env.conn.commit()
    try:
        vd_lb = ViewDef("v_arr_bounds", "SELECT ka, COUNT(*) AS n FROM cm_extra GROUP BY ka")
        vd_sc = ViewDef("v_sec_coal", "SELECT COALESCE(kc1, kc2) AS k, COUNT(*) AS n FROM cm_extra GROUP BY COALESCE(kc1, kc2)")
        env.register(vd_lb)
        env.register(vd_sc)
        cur.execute(
            f"""
            INSERT INTO cm_extra VALUES
            (1, '{{1,2}}'::int[], ROW(1.0, 'x')::"{sec_sch}"."SecMixedComp", NULL, 1),
            (2, '[2:3]={{1,2}}'::int[], NULL, ROW(1.00, 'x')::"{sec_sch}"."SecMixedComp", 2)
            """
        )
        env.conn.commit()
        cur.execute("SELECT COUNT(*) FROM v_arr_bounds")
        assert cur.fetchone()[0] == 2  # '[2:3]={1,2}' != '{1,2}' in PostgreSQL!
        cur.execute("SELECT COUNT(*) FROM v_sec_coal")
        assert cur.fetchone()[0] == 1  # ROW(1.0,'x') == ROW(1.00,'x') merged cleanly!
        env.conn.commit()
        env.manager.drop_view("v_arr_bounds")
        env.manager.drop_view("v_sec_coal")
    finally:
        cur.execute("DROP TABLE IF EXISTS cm_extra")
        cur.execute(f'DROP SCHEMA IF EXISTS "{sec_sch}" CASCADE')
        env.conn.commit()


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


def test_group_lock_table_stays_bounded_hot_and_zero_multixacts(make_env) -> None:
    """Group-scoped lock table stays bounded at PG_GROUP_LOCK_BUCKETS + 1 rows
    even after thousands of distinct (including transient/deleted) groups are
    written concurrently across independent connections, uses HOT updates, and
    allocates zero MultiXactIds."""
    import concurrent.futures
    from matview.dialects.pg_statement import PG_GROUP_LOCK_BUCKETS

    env = make_env(views=[V_STRAGG], seed=False)
    env.execute("CREATE INDEX idx_t_g ON t (g)")
    env.commit()
    lock_tbl = "__matview_lock_v_stragg"

    stat_conn = env.connect()
    stat_conn.autocommit = True
    scur = stat_conn.cursor()
    scur.execute("SELECT current_setting('is_superuser')")
    is_super = scur.fetchone()[0] == "on"
    scur.execute(
        "SELECT COALESCE(n_tup_upd, 0), COALESCE(n_tup_hot_upd, 0) FROM pg_stat_user_tables "
        "WHERE schemaname = %s AND relname = %s",
        (env.ns, lock_tbl),
    )
    row_before = scur.fetchone()
    upd_before, hot_before = (row_before[0], row_before[1]) if row_before else (0, 0)
    if is_super:
        scur.execute("CHECKPOINT")
        scur.execute("SELECT next_multixact_id::text::bigint FROM pg_control_checkpoint()")
        mx_before = scur.fetchone()[0]
    else:
        mx_before = None

    n_threads = 4
    groups_per_thread = 750  # 3,000 distinct groups (half deleted after insert -> 4,500 lock updates)
    expected_updates = n_threads * (groups_per_thread + (groups_per_thread + 1) // 2)

    def _writer(tid: int) -> None:
        conn = env.connect()
        try:
            cur = conn.cursor()
            base = tid * groups_per_thread + 1
            for i in range(groups_per_thread):
                rid = base + i
                cur.execute("INSERT INTO t (id, g, v) VALUES (%s, %s, %s)", (rid, rid, i))
                conn.commit()
                if i % 2 == 0:
                    cur.execute("DELETE FROM t WHERE id = %s", (rid,))
                    conn.commit()
            # Force this worker backend to flush its pg_stat_user_tables counters immediately.
            cur.execute("SELECT pg_stat_force_next_flush()")
            conn.commit()
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futs = [ex.submit(_writer, t) for t in range(n_threads)]
        for f in futs:
            f.result()

    # Exact MV equivalence after 3,000 concurrent groups + 1,500 transient group deletions.
    env.check("bounded group lock table concurrent transient groups")

    # Lock table row count and id bounds: strictly 0..PG_GROUP_LOCK_BUCKETS.
    scur.execute(f"SELECT COUNT(*), MIN(__mv_id), MAX(__mv_id) FROM {env.ns}.{lock_tbl}")
    row_cnt, min_id, max_id = scur.fetchone()
    assert row_cnt == PG_GROUP_LOCK_BUCKETS + 1, (row_cnt, min_id, max_id)
    assert min_id == 0 and max_id == PG_GROUP_LOCK_BUCKETS, (min_id, max_id)

    # Verify negative hashtextextended results still map strictly into 1..PG_GROUP_LOCK_BUCKETS.
    scur.execute(
        f"SELECT MIN(((h & {PG_GROUP_LOCK_BUCKETS - 1}) + 1)), "
        f"       MAX(((h & {PG_GROUP_LOCK_BUCKETS - 1}) + 1)), "
        f"       COUNT(*) FILTER (WHERE h < 0) "
        f"FROM (SELECT hashtextextended('g_' || i, 0) AS h FROM generate_series(1, 2000) i) s"
    )
    b_min, b_max, neg_cnt = scur.fetchone()
    assert neg_cnt > 0 and b_min >= 1 and b_max <= PG_GROUP_LOCK_BUCKETS, (b_min, b_max, neg_cnt)

    # Verify bounded heap size (<= 512 KiB even under brief xmin pinning), HOT updates,
    # and zero MultiXactId allocations.
    scur.execute(f"SELECT pg_relation_size('{env.ns}.{lock_tbl}')")
    heap_bytes = scur.fetchone()[0]
    assert heap_bytes <= 512 * 1024, f"lock table heap grew unexpectedly: {heap_bytes} bytes"

    deadline = time.monotonic() + 5.0
    upd_delta, hot_delta = 0, 0
    while True:
        scur.execute(
            "SELECT COALESCE(n_tup_upd, 0), COALESCE(n_tup_hot_upd, 0) FROM pg_stat_user_tables "
            "WHERE schemaname = %s AND relname = %s",
            (env.ns, lock_tbl),
        )
        r_after = scur.fetchone()
        if r_after is not None:
            upd_delta = r_after[0] - upd_before
            hot_delta = r_after[1] - hot_before
            if upd_delta >= expected_updates:
                break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    assert upd_delta == expected_updates, f"expected {expected_updates} lock-table updates, saw {upd_delta}"
    assert hot_delta >= int(upd_delta * 0.98), f"expected HOT updates on lock table: {hot_delta}/{upd_delta}"

    if is_super:
        scur.execute("CHECKPOINT")
        scur.execute("SELECT next_multixact_id::text::bigint FROM pg_control_checkpoint()")
        mx_after = scur.fetchone()[0]
        assert mx_after == mx_before, f"group-scoped writers allocated MultiXactIds: {mx_after - mx_before}"
    stat_conn.close()


@pytest.mark.parametrize("isolation", ["REPEATABLE READ", "SERIALIZABLE"])
def test_whole_view_recompute_and_group_writer_symmetric_exclusion(make_env, isolation: str) -> None:
    """Whole-view recomputes (writes to subquery table u or TRUNCATE t) and
    group-scoped writes to t exclude each other symmetrically in BOTH directions:
    under READ COMMITTED (reverse order: whole-view holds lock, group writer
    waits) and under REPEATABLE READ / SERIALIZABLE (both u-then-t and t-then-u
    plus TRUNCATE-then-t fail closed with 40001, including after repair_view())."""
    from matview.dialects.pg_statement import PG_GROUP_LOCK_BUCKETS

    schema = list(PG_SCHEMA) + ["CREATE TABLE u (g INT PRIMARY KEY)"]
    view = ViewDef(
        "v_sub",
        "SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids FROM t "
        "WHERE g IN (SELECT g FROM u) GROUP BY g",
    )
    env = make_env(views=[], schema_sql=schema, seed=True)
    env.execute("INSERT INTO u (g) VALUES (1), (2), (3)")
    env.commit()
    env.views = [view]
    env.register(view)
    assert _strategy(env, "v_sub") == "group_scoped_delta"

    # 1. Reverse READ COMMITTED direction: whole-view writer holds all buckets,
    #    group-scoped writer on t must wait and then see the committed u row.
    a, b, monitor = env.connect(), env.connect(), env.connect()
    sid_a, sid_b = _session_id(env, a), _session_id(env, b)
    a.cursor().execute("INSERT INTO u (g) VALUES (4)")
    worker = _Worker(b, ["INSERT INTO t (id, g, v) VALUES (6101, 4, 1)"]).start()
    try:
        _wait_for(lambda: _row_lock_waiting(env, monitor, sid_b, sid_a), worker.done, "group writer waits for whole-view lock")
    finally:
        a.commit()
        worker.join()
    if worker.error is not None:
        raise worker.error
    env.check("group writer after whole-view recompute")

    # 2. Reverse REPEATABLE READ/SERIALIZABLE direction: B takes snapshot, A writes t
    #    (updates one bucket) and commits, B writes u (locks all buckets 0..B) -> 40001.
    b_iso = env.connect()
    cb = b_iso.cursor()
    cb.execute(f"BEGIN ISOLATION LEVEL {isolation}")
    cb.execute("SELECT COUNT(*) FROM t")
    cb.fetchall()
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (6102, 1, 2)")
    a.commit()
    with pytest.raises(Exception) as excinfo:
        cb.execute("INSERT INTO u (g) VALUES (5)")
        b_iso.commit()
    b_iso.rollback()
    assert _pgcode(excinfo.value) == SERIALIZATION_FAILURE, repr(excinfo.value)

    # 3. Forward REPEATABLE READ/SERIALIZABLE direction with TRUNCATE: B takes snapshot
    #    (without locking table t, since TRUNCATE needs ACCESS EXCLUSIVE on t),
    #    A truncates t (locks all buckets 0..B) and commits, B inserts into t -> 40001.
    cb.execute(f"BEGIN ISOLATION LEVEL {isolation}")
    cb.execute("SELECT COUNT(*) FROM u")
    cb.fetchall()
    a.cursor().execute("TRUNCATE TABLE t")
    a.commit()
    with pytest.raises(Exception) as excinfo2:
        cb.execute("INSERT INTO t (id, g, v) VALUES (6103, 2, 3)")
        b_iso.commit()
    b_iso.rollback()
    assert _pgcode(excinfo2.value) == SERIALIZATION_FAILURE, repr(excinfo2.value)

    # 3b. Reverse REPEATABLE READ/SERIALIZABLE direction with TRUNCATE: B takes snapshot
    #     (without locking t), A inserts into t (updates one bucket) and commits,
    #     B truncates t (locks all buckets 0..B) -> 40001.
    cb.execute(f"BEGIN ISOLATION LEVEL {isolation}")
    cb.execute("SELECT COUNT(*) FROM u")
    cb.fetchall()
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (6105, 3, 5)")
    a.commit()
    with pytest.raises(Exception) as excinfo2b:
        cb.execute("TRUNCATE TABLE t")
        b_iso.commit()
    b_iso.rollback()
    assert _pgcode(excinfo2b.value) == SERIALIZATION_FAILURE, repr(excinfo2b.value)

    # 4. After repair_view(), all PG_GROUP_LOCK_BUCKETS + 1 rows are re-seeded and
    #    reverse REPEATABLE READ/SERIALIZABLE direction still fails closed with 40001.
    env.manager.repair_view("v_sub")
    cur = env.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM __matview_lock_v_sub")
    assert cur.fetchone()[0] == PG_GROUP_LOCK_BUCKETS + 1
    env.conn.commit()

    cb.execute(f"BEGIN ISOLATION LEVEL {isolation}")
    cb.execute("SELECT COUNT(*) FROM t")
    cb.fetchall()
    a.cursor().execute("INSERT INTO t (id, g, v) VALUES (6104, 1, 4)")
    a.commit()
    with pytest.raises(Exception) as excinfo3:
        cb.execute("INSERT INTO u (g) VALUES (6)")
        b_iso.commit()
    b_iso.rollback()
    assert _pgcode(excinfo3.value) == SERIALIZATION_FAILURE, repr(excinfo3.value)
    b_iso.close()
    a.close()
    b.close()
    monitor.close()
    env.check(f"symmetric exclusion ({isolation})")


def test_multi_bucket_opposite_order_batch_and_whole_view_no_deadlock(make_env) -> None:
    """Concurrent multi-row statements touching many buckets in opposite orders
    plus concurrent whole-view recomputes (writes to subquery table u) never
    deadlock and stay exact."""
    import concurrent.futures

    schema = list(PG_SCHEMA) + ["CREATE TABLE u (g INT PRIMARY KEY)"]
    view = ViewDef(
        "v_sub_batch",
        "SELECT g, STRING_AGG(CAST(id AS TEXT), ',' ORDER BY id) AS ids FROM t "
        "WHERE g IN (SELECT g FROM u) GROUP BY g",
    )
    env = make_env(views=[], schema_sql=schema, seed=False)
    env.execute("INSERT INTO u (g) SELECT g FROM generate_series(1, 60) g")
    env.commit()
    env.views = [view]
    env.register(view)

    def _batch_writer(direction: int) -> None:
        conn = env.connect()
        try:
            cur = conn.cursor()
            gs = list(range(1, 61)) if direction > 0 else list(range(60, 0, -1))
            for r in range(15):
                base = (10000 if direction > 0 else 20000) + r * 100
                vals = ", ".join(f"({base + idx}, {g}, {r})" for idx, g in enumerate(gs))
                cur.execute(f"INSERT INTO t (id, g, v) VALUES {vals}")
                conn.commit()
        finally:
            conn.close()

    def _whole_view_writer() -> None:
        conn = env.connect()
        try:
            cur = conn.cursor()
            for r in range(10):
                cur.execute("INSERT INTO u (g) VALUES (%s)", (100 + r,))
                conn.commit()
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futs = [
            ex.submit(_batch_writer, 1),
            ex.submit(_batch_writer, -1),
            ex.submit(_whole_view_writer),
        ]
        for f in futs:
            f.result()

    env.check("multi-bucket opposite-order batch + whole-view stress")


def test_pg18_virtual_generated_columns_all_strategies_and_drift(make_env, monkeypatch) -> None:
    """PostgreSQL 18+ VIRTUAL generated columns (both explicit VIRTUAL and
    default GENERATED ALWAYS AS (...)) work across every maintenance strategy
    (unreferenced views, algebraic aggregates, Tier 2 STRING_AGG/window, joins
    where the joined source has virtual columns, UNION ALL/UNION/DISTINCT,
    filtered projections, partitioned tables), preserve exact values and OIDs
    under both SQLAlchemy and raw DB-API introspection, and report schema drift
    on ALTER TABLE ... ALTER COLUMN ... SET EXPRESSION AS."""
    import matview.connectivity as _conn_mod

    probe_env = make_env(views=[], seed=False)
    pcur = probe_env.conn.cursor()
    pcur.execute("SHOW server_version_num")
    ver_num = int(pcur.fetchone()[0])
    probe_env.conn.commit()
    if ver_num < 180000:
        pytest.skip(f"VIRTUAL generated columns require PostgreSQL 18+ (running {ver_num})")

    env = make_env(
        views=[],
        seed=False,
        schema_sql=[
            """
            CREATE TABLE gv (
                id INT PRIMARY KEY,
                g INT,
                "a b" INT,
                x NUMERIC(10, 2),
                v_default INT GENERATED ALWAYS AS ("a b" * 2 + 1),
                v_num NUMERIC(12, 2) GENERATED ALWAYS AS (x * 1.5) VIRTUAL,
                v_tag TEXT GENERATED ALWAYS AS (CASE WHEN "a b" > 0 THEN 'pos' ELSE 'nonpos' END) VIRTUAL,
                v_stored INT GENERATED ALWAYS AS ("a b" + 10) STORED
            )
            """,
            """
            CREATE TABLE gv_dim (
                g INT PRIMARY KEY,
                raw_name TEXT,
                label TEXT GENERATED ALWAYS AS (UPPER(raw_name) || '_v') VIRTUAL
            )
            """,
            """
            CREATE TABLE gv_part (
                id INT NOT NULL,
                g INT NOT NULL,
                val INT,
                doubled INT GENERATED ALWAYS AS (val * 2) VIRTUAL,
                PRIMARY KEY (g, id)
            ) PARTITION BY LIST (g)
            """,
            "CREATE TABLE gv_part_1 PARTITION OF gv_part FOR VALUES IN (1)",
            "CREATE TABLE gv_part_2 PARTITION OF gv_part FOR VALUES IN (2)",
        ],
    )

    # Verify both SQLAlchemy and raw DB-API introspection populate generated_virtual and generation_expr.
    sa_schemas = env.manager.fetch_schema()
    monkeypatch.setattr(_conn_mod, "_can_use_sqlalchemy_inspector", lambda _c: False)
    raw_schemas = env.manager.fetch_schema()
    monkeypatch.undo()
    for label, sm in (("sqlalchemy", sa_schemas), ("dbapi_fallback", raw_schemas)):
        gv_cols = sm["gv"].columns
        assert gv_cols["v_default"].generated and gv_cols["v_default"].generated_virtual, label
        assert gv_cols["v_default"].generation_expr is not None, label
        assert gv_cols["v_num"].generated and gv_cols["v_num"].generated_virtual, label
        assert gv_cols["v_tag"].generated and gv_cols["v_tag"].generated_virtual, label
        assert gv_cols["v_stored"].generated and not gv_cols["v_stored"].generated_virtual, label
        assert sm["gv_dim"].columns["label"].generated_virtual, label
        assert sm["gv_part"].columns["doubled"].generated_virtual, label

    views = [
        # 1. View that does NOT reference any virtual column (previously crashed on every write!).
        ViewDef("v_gv_unref", 'SELECT g, SUM("a b") AS s, COUNT(*) AS c FROM gv GROUP BY g'),
        # 2. Algebraic aggregate grouping by and aggregating virtual columns.
        ViewDef("v_gv_agg", "SELECT v_tag, SUM(v_default) AS sd, SUM(v_num) AS sn, COUNT(*) AS c FROM gv GROUP BY v_tag"),
        # 3. Tier 2 STRING_AGG and window function over virtual columns.
        ViewDef("v_gv_str", "SELECT g, STRING_AGG(CAST(v_default AS TEXT), ',' ORDER BY id) AS ds FROM gv GROUP BY g"),
        ViewDef("v_gv_win", "SELECT id, v_tag, v_default, ROW_NUMBER() OVER (PARTITION BY v_tag ORDER BY v_default, id) AS rn FROM gv"),
        # 4. Algebraic multi-table join where BOTH sources have virtual columns.
        ViewDef(
            "v_gv_join",
            "SELECT d.label, SUM(gv.v_default) AS sd, COUNT(*) AS c "
            "FROM gv JOIN gv_dim d ON gv.g = d.g GROUP BY d.label",
        ),
        # 5. UNION / DISTINCT / filtered Z-set projection over virtual columns.
        ViewDef("v_gv_dst", "SELECT DISTINCT v_default, v_tag FROM gv"),
        ViewDef("v_gv_proj", "SELECT id, v_default, v_num FROM gv WHERE v_default > 0"),
        ViewDef(
            "v_gv_uall",
            "SELECT id, v_default AS val FROM gv WHERE g = 1 UNION ALL SELECT id, v_stored AS val FROM gv WHERE g = 2",
        ),
        # 6. Partitioned source table with virtual column.
        ViewDef("v_gv_part", "SELECT g, SUM(doubled) AS s, COUNT(*) AS c FROM gv_part GROUP BY g"),
    ]
    env.views = views
    for vd in views:
        env.register(vd)

    cur = env.conn.cursor()

    def _assert_all_exact(label: str) -> None:
        for vd in views:
            cur.execute(f"SELECT * FROM {vd.name} LIMIT 0")
            mv_oids = [d.type_code for d in cur.description]
            cols = [d[0] for d in cur.description]
            cur.execute(f"SELECT * FROM ({vd.select}) AS q LIMIT 0")
            live_oids = [d.type_code for d in cur.description]
            assert mv_oids == live_oids, f"{label}: {vd.name} OIDs {mv_oids} != {live_oids}"
            sel_cols = ", ".join(cols)
            cur.execute(
                f"(SELECT {sel_cols} FROM {vd.name} EXCEPT ALL SELECT {sel_cols} FROM ({vd.select}) AS q) "
                f"UNION ALL "
                f"(SELECT {sel_cols} FROM ({vd.select}) AS q EXCEPT ALL SELECT {sel_cols} FROM {vd.name})"
            )
            diff = cur.fetchall()
            assert diff == [], f"{label}: {vd.name} diff={diff}"
        env.conn.commit()

    cur.execute("INSERT INTO gv_dim (g, raw_name) VALUES (1, 'alpha'), (2, 'beta')")
    cur.execute('INSERT INTO gv (id, g, "a b", x) VALUES (1, 1, 5, 10.00), (2, 1, -2, 3.33), (3, 2, 0, NULL), (4, 2, NULL, 4.50)')
    cur.execute("INSERT INTO gv_part (id, g, val) VALUES (1, 1, 10), (2, 1, 20), (3, 2, -5)")
    env.conn.commit()
    _assert_all_exact("after inserts")

    # UPDATE inputs of virtual columns (including ON CONFLICT DO UPDATE and direct partition write).
    cur.execute('UPDATE gv SET "a b" = "a b" + 3, x = COALESCE(x, 0) + 1.25 WHERE id IN (1, 2, 3)')
    cur.execute(
        'INSERT INTO gv (id, g, "a b", x) VALUES (1, 2, -10, 7.00) '
        'ON CONFLICT (id) DO UPDATE SET g = EXCLUDED.g, "a b" = EXCLUDED."a b", x = EXCLUDED.x'
    )
    cur.execute("UPDATE gv_dim SET raw_name = 'gamma' WHERE g = 1")
    cur.execute("UPDATE gv_part_1 SET val = val + 7 WHERE id = 1")
    env.conn.commit()
    _assert_all_exact("after updates and upsert")

    cur.execute("DELETE FROM gv WHERE id IN (2, 4)")
    cur.execute("DELETE FROM gv_part WHERE id = 3")
    env.conn.commit()
    _assert_all_exact("after deletes")

    cur.execute("TRUNCATE TABLE gv")
    env.conn.commit()
    _assert_all_exact("after truncate")

    # ALTER COLUMN ... SET EXPRESSION AS must be detected as schema_changed and repaired cleanly.
    cur.execute('INSERT INTO gv (id, g, "a b", x) VALUES (10, 1, 4, 2.00)')
    env.conn.commit()
    _assert_all_exact("before alter expression")

    cur.execute('ALTER TABLE gv ALTER COLUMN v_default SET EXPRESSION AS ("a b" * 10)')
    env.conn.commit()
    drift = env.manager.check_schema_drift()
    assert drift["v_gv_agg"] == "drifted", drift
    for vn in ("v_gv_agg", "v_gv_str", "v_gv_win", "v_gv_join", "v_gv_dst", "v_gv_proj", "v_gv_uall"):
        env.manager.repair_view(vn)
    cur.execute('UPDATE gv SET "a b" = 6 WHERE id = 10')
    env.conn.commit()
    _assert_all_exact("after alter expression and repair")

    # Adversarial keyword-named columns, multi-word casts, ARRAY/COLLATE virtual columns,
    # coexisting ENUM column, dropped columns before/after virtual columns, and NULL virtual expression.
    cur.execute("CREATE TYPE gv_status AS ENUM ('lo', 'hi')")
    cur.execute(
        """
        CREATE TABLE gv_adv (
            id INT PRIMARY KEY,
            drop_before INT,
            status gv_status,
            "end" INT,
            "null" INT,
            "both" TEXT,
            "precision" INT,
            "varying" TEXT,
            v_kw INT GENERATED ALWAYS AS (CASE WHEN "end" > 0 THEN "end" + "precision" ELSE COALESCE("null", 0) END) VIRTUAL,
            v_cast DOUBLE PRECISION GENERATED ALWAYS AS (("end"::double precision) + 0.5) VIRTUAL,
            v_sub TEXT GENERATED ALWAYS AS (SUBSTRING("varying"::character varying FROM 1 FOR 3)) VIRTUAL,
            v_arr INT[] GENERATED ALWAYS AS (ARRAY["end", COALESCE("null", 0)]) VIRTUAL,
            v_coll TEXT COLLATE "C" GENERATED ALWAYS AS (LOWER("both")) VIRTUAL,
            v_null INT GENERATED ALWAYS AS (NULL) VIRTUAL,
            drop_after INT
        )
        """
    )
    cur.execute("ALTER TABLE gv_adv DROP COLUMN drop_before, DROP COLUMN drop_after")
    env.conn.commit()

    # Register via raw DB-API fallback too to verify ENUM/ARRAY + COLLATE fidelity.
    monkeypatch.setattr(_conn_mod, "_can_use_sqlalchemy_inspector", lambda _c: False)
    vd_adv = ViewDef(
        "v_gv_adv",
        "SELECT status, v_coll, v_sub, v_arr, SUM(v_kw) AS sk, COUNT(v_cast) AS cc, COUNT(v_null) AS cn, COUNT(*) AS c "
        "FROM gv_adv GROUP BY status, v_coll, v_sub, v_arr",
    )
    env.register(vd_adv)
    monkeypatch.undo()

    cur.execute(
        'INSERT INTO gv_adv (id, status, "end", "null", "both", "precision", "varying") VALUES '
        "(1, 'hi', 5, 1, 'AbC', 2, 'hello'), (2, 'lo', -1, 9, 'XYZ', 4, 'world')"
    )
    cur.execute('UPDATE gv_adv SET status = \'lo\', "end" = -3, "both" = \'ZzZ\', "varying" = \'abcdef\' WHERE id = 1')
    cur.execute("DELETE FROM gv_adv WHERE id = 2")
    env.conn.commit()
    cur.execute(
        f"(SELECT * FROM {vd_adv.name} EXCEPT ALL SELECT * FROM ({vd_adv.select}) AS q) "
        f"UNION ALL "
        f"(SELECT * FROM ({vd_adv.select}) AS q EXCEPT ALL SELECT * FROM {vd_adv.name})"
    )
    assert cur.fetchall() == []
    env.conn.commit()


def test_pg_stage3_multi_source_overlap_bracket_and_inclusion_exclusion(make_env) -> None:
    """Stage 3: exact PostgreSQL maintenance when one query modifies several
    sources of a join view via writable CTEs (2- and 3-source INSERT/UPDATE/
    DELETE mixes), 2- and 3-level FK cascades (including 0-child cascades),
    user BEFORE/AFTER ROW triggers, TRUNCATE CASCADE, partitioned sources,
    SAVEPOINT rollback with an open bracket, sequential PL/pgSQL non-overlap
    (verified with zero staging inserts), Tier 2 multi-source joins, special-
    character view names, and concurrent READ COMMITTED / REPEATABLE READ
    writers."""
    import psycopg2
    import psycopg2.extensions

    schema = [
        "CREATE TABLE s3_d (g INT PRIMARY KEY, name TEXT NOT NULL, x INT NOT NULL DEFAULT 0, unused_blob BYTEA)",
        (
            "CREATE TABLE s3_t ("
            "id INT PRIMARY KEY, "
            "g INT REFERENCES s3_d(g) ON UPDATE CASCADE ON DELETE CASCADE, "
            "v INT, "
            "w INT NOT NULL DEFAULT 0)"
        ),
        (
            "CREATE TABLE s3_u ("
            "uid INT PRIMARY KEY, "
            "tid INT REFERENCES s3_t(id) ON UPDATE CASCADE ON DELETE CASCADE, "
            "amt INT)"
        ),
        # Separate FK-free pair for phantom zero-net-group CTE tests (INSERT d + DELETE orphan t).
        "CREATE TABLE s3_od (g INT PRIMARY KEY, name TEXT NOT NULL)",
        "CREATE TABLE s3_ot (id INT PRIMARY KEY, g INT NOT NULL, v INT)",
        # Partitioned child table joined to s3_d.
        (
            "CREATE TABLE s3_pt ("
            "id INT NOT NULL, g INT NOT NULL, v INT NOT NULL, "
            "PRIMARY KEY (g, id)) PARTITION BY LIST (g)"
        ),
        "CREATE TABLE s3_pt_1 PARTITION OF s3_pt FOR VALUES IN (1, 2, 10, 11, 12)",
        "CREATE TABLE s3_pt_2 PARTITION OF s3_pt DEFAULT",
    ]
    views = [
        # 2-table algebraic join with COUNT/SUM/AVG/MIN/MAX/BOOL_AND/BIT_XOR.
        ViewDef(
            "v_s3_2",
            "SELECT d.name, COUNT(*) AS c, SUM(t.v) AS sv, AVG(t.v) AS av, "
            "MIN(t.v) AS mn, MAX(t.v) AS mx, BOOL_AND(t.v > 0) AS all_pos, BIT_XOR(t.w) AS bx "
            "FROM s3_t t JOIN s3_d d ON t.g = d.g GROUP BY d.name",
        ),
        # Global (ungrouped) 2-table algebraic join.
        ViewDef(
            "v_s3_glob",
            "SELECT COUNT(*) AS c, SUM(t.v) AS sv, MIN(t.v) AS mn "
            "FROM s3_t t JOIN s3_d d ON t.g = d.g",
        ),
        # 3-table algebraic join (exercises all 7 inclusion-exclusion terms including triple term).
        ViewDef(
            "v_s3_3",
            "SELECT d.name, COUNT(*) AS c, SUM(u.amt) AS samt, MIN(u.amt) AS mn, MAX(u.amt) AS mx "
            "FROM s3_d d JOIN s3_t t ON t.g = d.g JOIN s3_u u ON u.tid = t.id GROUP BY d.name",
        ),
        # Tier 2 multi-source LEFT JOIN, FULL OUTER JOIN, STRING_AGG, and cross-table GROUP BY.
        ViewDef(
            "v_s3_left",
            "SELECT d.name, COUNT(t.id) AS c, SUM(t.v) AS sv "
            "FROM s3_d d LEFT JOIN s3_t t ON t.g = d.g GROUP BY d.name",
        ),
        ViewDef(
            "v_s3_str",
            "SELECT (t.w + d.x) AS k, COUNT(*) AS c, STRING_AGG(d.name, ',' ORDER BY t.id) AS names "
            "FROM s3_t t JOIN s3_d d ON t.g = d.g GROUP BY (t.w + d.x)",
        ),
        # FK-free phantom zero-net-group join.
        ViewDef(
            "v_s3_orphan",
            "SELECT od.name, COUNT(*) AS c, SUM(ot.v) AS sv "
            "FROM s3_ot ot JOIN s3_od od ON ot.g = od.g GROUP BY od.name",
        ),
        # Partitioned join view.
        ViewDef(
            "v_s3_part",
            "SELECT d.name, COUNT(*) AS c, SUM(pt.v) AS sv "
            "FROM s3_pt pt JOIN s3_d d ON pt.g = d.g GROUP BY d.name",
        ),
    ]
    env = make_env(views=[], schema_sql=schema, seed=False)
    cur = env.conn.cursor()
    cur.execute("INSERT INTO s3_d (g, name, x) VALUES (1, 'a', 10), (2, 'b', 20), (3, 'c', 30)")
    cur.execute(
        "INSERT INTO s3_t (id, g, v, w) VALUES "
        "(10, 1, 5, 1), (11, 1, 15, 2), (20, 2, -4, 3), (30, 3, 8, 4)"
    )
    cur.execute(
        "INSERT INTO s3_u (uid, tid, amt) VALUES "
        "(100, 10, 7), (101, 10, -2), (200, 20, 9), (300, 30, 11)"
    )
    cur.execute("INSERT INTO s3_od (g, name) VALUES (1, 'ok')")
    cur.execute("INSERT INTO s3_ot (id, g, v) VALUES (1, 1, 10), (2, 99, 50)")
    cur.execute("INSERT INTO s3_pt (id, g, v) VALUES (1, 1, 100), (2, 2, 200)")
    env.conn.commit()
    env.views = views
    for vd in views:
        env.register(vd)
    # Also register a special-character view name join directly to verify GUC/OID/trigger quoting.
    special_select = "SELECT d.name, COUNT(*) AS c, SUM(pt.v) AS sv FROM s3_pt pt JOIN s3_d d ON pt.g = d.g GROUP BY d.name"
    env.manager.register_view(f'CREATE VIEW "my-stage3 join" AS {special_select};')
    env.check("initial registration")

    # 1. Sequential PL/pgSQL statements MUST NOT trigger overlap/staging.
    stg_tables = [t for t in env.plans["v_s3_2"].auxiliary_table_names if "__matview_stg_" in t]
    assert len(stg_tables) == 2, env.plans["v_s3_2"].auxiliary_table_names
    cur.execute(
        """
        DO $$
        BEGIN
            UPDATE s3_d SET x = x + 1 WHERE g = 1;
            UPDATE s3_t SET v = v + 2 WHERE id = 10;
            UPDATE s3_u SET amt = amt + 3 WHERE uid = 100;
        END $$;
        """
    )
    cur.execute(
        "SELECT relname, n_tup_ins FROM pg_stat_xact_user_tables WHERE relname = ANY(%s)",
        (stg_tables,),
    )
    stg_ins = dict(cur.fetchall())
    assert all(cnt == 0 for cnt in stg_ins.values()), f"sequential PL/pgSQL flagged overlap: {stg_ins}"
    env.conn.commit()
    env.check("after sequential PL/pgSQL statements")

    # 2. 2-source writable CTEs: INSERT+INSERT, UPDATE+UPDATE, DELETE+DELETE,
    #    mixed INSERT+UPDATE+DELETE, and phantom zero-net group (INSERT od + DELETE orphan ot).
    cur.execute(
        "WITH a AS (INSERT INTO s3_d (g, name, x) VALUES (4, 'd', 40), (5, 'a', 50) RETURNING g) "
        "INSERT INTO s3_t (id, g, v, w) VALUES (40, 4, 25, 5), (41, 5, -9, 6)"
    )
    env.conn.commit()
    env.check("2-source CTE INSERT+INSERT")

    cur.execute(
        "WITH a AS (UPDATE s3_d SET name = 'a_renamed', x = x + 5 WHERE g = 1 RETURNING g) "
        "UPDATE s3_t SET v = v - 20, w = w + 10 WHERE g IN (SELECT g FROM a)"
    )
    env.conn.commit()
    env.check("2-source CTE UPDATE+UPDATE (group key + extreme change)")

    cur.execute(
        "WITH a AS (INSERT INTO s3_od (g, name) VALUES (99, 'phantom') RETURNING g) "
        "DELETE FROM s3_ot WHERE g IN (SELECT g FROM a)"
    )
    env.conn.commit()
    env.check("2-source CTE phantom zero-net group")

    # 3. 3-source writable CTEs (proving all 7 inclusion-exclusion terms including triple term!).
    cur.execute(
        "WITH a AS (INSERT INTO s3_d (g, name, x) VALUES (6, 'triple', 60) RETURNING g), "
        "     b AS (INSERT INTO s3_t (id, g, v, w) VALUES (60, 6, 33, 7) RETURNING id) "
        "INSERT INTO s3_u (uid, tid, amt) VALUES (600, 60, 100), (601, 60, -50)"
    )
    env.conn.commit()
    env.check("3-source CTE INSERT+INSERT+INSERT")

    cur.execute(
        "WITH a AS (UPDATE s3_d SET name = 'triple2', x = 61 WHERE g = 6 RETURNING g), "
        "     b AS (UPDATE s3_t SET v = v + 1, w = w + 1 WHERE g = 6 RETURNING id) "
        "UPDATE s3_u SET amt = amt + 25 WHERE tid = 60"
    )
    env.conn.commit()
    env.check("3-source CTE UPDATE+UPDATE+UPDATE")

    cur.execute(
        "WITH a AS (DELETE FROM s3_u WHERE uid = 600 RETURNING tid), "
        "     b AS (UPDATE s3_t SET v = -100 WHERE id = 60 RETURNING g) "
        "UPDATE s3_d SET name = 'triple3' WHERE g = 6"
    )
    env.conn.commit()
    env.check("3-source CTE DELETE+UPDATE+UPDATE mix")

    # 4. 2-level and 3-level FK cascades (multi-row ON UPDATE CASCADE, ON DELETE CASCADE,
    #    and 0-child-row cascade where child transition table is empty!).
    cur.execute("INSERT INTO s3_d (g, name, x) VALUES (77, 'childless', 0)")
    env.conn.commit()
    cur.execute("UPDATE s3_d SET g = 78, name = 'childless2' WHERE g = 77")
    cur.execute("DELETE FROM s3_d WHERE g = 78")
    env.conn.commit()
    env.check("FK cascade with 0 matching child rows")

    cur.execute("UPDATE s3_d SET g = g + 100 WHERE g IN (1, 2)")
    env.conn.commit()
    env.check("multi-row FK ON UPDATE CASCADE (d->t)")

    cur.execute("DELETE FROM s3_d WHERE g IN (102, 6)")
    env.conn.commit()
    env.check("3-level FK ON DELETE CASCADE (d->t->u)")

    # 5. Partitioned source in writable CTE + special-char view name + TRUNCATE CASCADE.
    cur.execute(
        "WITH a AS (INSERT INTO s3_d (g, name, x) VALUES (10, 'part_grp', 1) RETURNING g) "
        "INSERT INTO s3_pt (id, g, v) VALUES (10, 10, 55), (11, 10, 45)"
    )
    env.conn.commit()
    env.check("writable CTE into partitioned source")
    cur.execute(
        f'(SELECT * FROM "my-stage3 join" EXCEPT ALL SELECT * FROM ({special_select}) AS q) '
        f'UNION ALL '
        f'(SELECT * FROM ({special_select}) AS q EXCEPT ALL SELECT * FROM "my-stage3 join")'
    )
    assert cur.fetchall() == []
    env.conn.commit()

    # 6. SAVEPOINT + ROLLBACK TO SAVEPOINT mid-bracket (error after BEFORE/staging runs).
    cur.execute("SAVEPOINT sp_s3")
    with pytest.raises(psycopg2.Error):
        cur.execute(
            "WITH a AS (INSERT INTO s3_d (g, name, x) VALUES (88, 'rolled', 1) RETURNING g) "
            "INSERT INTO s3_t (id, g, v, w) VALUES (10, 88, 1, 1)"  # duplicate PK id=10!
        )
    cur.execute("ROLLBACK TO SAVEPOINT sp_s3")
    cur.execute("RELEASE SAVEPOINT sp_s3")
    # Subsequent overlapping and non-overlapping writes in the SAME transaction must remain exact.
    cur.execute(
        "WITH a AS (INSERT INTO s3_d (g, name, x) VALUES (89, 'after_sp', 2) RETURNING g) "
        "INSERT INTO s3_t (id, g, v, w) VALUES (890, 89, 12, 3)"
    )
    cur.execute("UPDATE s3_t SET v = v + 3 WHERE id = 890")
    env.conn.commit()
    env.check("after SAVEPOINT rollback mid-bracket + follow-up writes")

    # Verify staging tables are completely empty after commit (zero leaked rows).
    for st in stg_tables:
        cur.execute(f"SELECT COUNT(*) FROM {st}")
        assert cur.fetchone()[0] == 0, f"leaked staging rows in {st}"
    env.conn.commit()

    # 7. User BEFORE ROW and AFTER ROW triggers on s3_d writing s3_t (multiple rows).
    cur.execute(
        """
        CREATE OR REPLACE FUNCTION s3_user_after_d() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.g >= 200 THEN
                INSERT INTO s3_t (id, g, v, w) VALUES (NEW.g * 10, NEW.g, 17, 2)
                ON CONFLICT (id) DO UPDATE SET v = s3_t.v + 5;
            END IF;
            RETURN NULL;
        END;
        $$;
        CREATE TRIGGER s3_z_user_after_d AFTER INSERT OR UPDATE ON s3_d
            FOR EACH ROW EXECUTE FUNCTION s3_user_after_d();
        """
    )
    env.conn.commit()
    cur.execute("INSERT INTO s3_d (g, name, x) VALUES (201, 'u1', 1), (202, 'u2', 2)")
    cur.execute("UPDATE s3_d SET name = name || '_v2' WHERE g IN (201, 202)")
    cur.execute("DROP TRIGGER s3_z_user_after_d ON s3_d; DROP FUNCTION s3_user_after_d();")
    env.conn.commit()
    env.check("multi-row user AFTER ROW trigger writing other join source")

    # 8. Concurrent READ COMMITTED writer alongside writable CTEs + REPEATABLE READ 40001 check.
    w1, w2 = env.connect(), env.connect()
    try:
        start_evt = threading.Event()
        errs: List[Optional[BaseException]] = [None, None]

        def _cte_worker() -> None:
            start_evt.wait()
            try:
                c1 = w1.cursor()
                for i in range(6):
                    g = 500 + i
                    c1.execute(
                        f"WITH a AS (INSERT INTO s3_d (g, name, x) VALUES ({g}, 'conc_{i % 2}', {i}) RETURNING g) "
                        f"INSERT INTO s3_t (id, g, v, w) VALUES ({g * 10}, {g}, {i + 1}, 1)"
                    )
                    w1.commit()
            except BaseException as exc:
                w1.rollback()
                errs[0] = exc

        def _single_worker() -> None:
            start_evt.wait()
            try:
                c2 = w2.cursor()
                for i in range(6):
                    c2.execute("UPDATE s3_t SET v = v + 1 WHERE id = 10")
                    w2.commit()
            except BaseException as exc:
                w2.rollback()
                errs[1] = exc

        t1 = threading.Thread(target=_cte_worker)
        t2 = threading.Thread(target=_single_worker)
        t1.start()
        t2.start()
        start_evt.set()
        t1.join()
        t2.join()
        assert errs == [None, None], errs
    finally:
        w1.close()
        w2.close()
    env.check("concurrent READ COMMITTED writable CTE + single-table writer")

    # REPEATABLE READ concurrent writable CTE must fail closed with 40001.
    rr1, rr2 = env.connect(), env.connect()
    try:
        rr1.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_REPEATABLE_READ)
        rr2.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_REPEATABLE_READ)
        c1, c2 = rr1.cursor(), rr2.cursor()
        c1.execute("SELECT COUNT(*) FROM s3_d")
        c2.execute("SELECT COUNT(*) FROM s3_d")
        c1.execute(
            "WITH a AS (INSERT INTO s3_d (g, name, x) VALUES (901, 'rr', 1) RETURNING g) "
            "INSERT INTO s3_t (id, g, v, w) VALUES (9010, 901, 5, 1)"
        )
        rr1.commit()
        with pytest.raises(psycopg2.errors.SerializationFailure):
            c2.execute(
                "WITH a AS (INSERT INTO s3_d (g, name, x) VALUES (902, 'rr', 2) RETURNING g) "
                "INSERT INTO s3_t (id, g, v, w) VALUES (9020, 902, 6, 1)"
            )
            rr2.commit()
        rr2.rollback()
    finally:
        rr1.close()
        rr2.close()
    env.check("REPEATABLE READ concurrent writable CTE fail-closed 40001")

    # 9. TRUNCATE s3_d CASCADE (truncates s3_d, s3_t, s3_u in one overlapping bracket).
    cur.execute("TRUNCATE s3_d CASCADE")
    env.conn.commit()
    env.check("after TRUNCATE s3_d CASCADE")

    # Drift check must stay ok across all Stage 3 triggers (including BEFORE triggers on partitions).
    drift = env.manager.check_schema_drift()
    assert all(status == "ok" for status in drift.values()), drift
    env.conn.commit()

    # 10. Adversarial drift detection for BEFORE bracket triggers (_bs) on parent and partition:
    #     disabling or dropping _bs must be reported by check_schema_drift() and repaired cleanly.
    bs_v2_d = next(t for t in env.plans["v_s3_2"].trigger_names if t.endswith("__s3_d_bs"))
    cur.execute(f"ALTER TABLE s3_d DISABLE TRIGGER {bs_v2_d}")
    env.conn.commit()
    assert env.manager.check_schema_drift()["v_s3_2"] == "triggers_disabled"
    env.manager.repair_view("v_s3_2")
    assert env.manager.check_schema_drift()["v_s3_2"] == "ok"

    bs_part = next(t for t in env.plans["v_s3_part"].trigger_names if t.endswith("__s3_pt_bs"))
    cur.execute(f"DROP TRIGGER {bs_part} ON s3_pt_1")
    env.conn.commit()
    assert env.manager.check_schema_drift()["v_s3_part"] == "triggers_missing"
    env.manager.repair_view("v_s3_part")
    assert env.manager.check_schema_drift()["v_s3_part"] == "ok"
    env.conn.commit()

    # 11. Adversarial Stage 3 edge cases:
    #     - Orphan row v -> NULL + parent INSERT with SUM (must store NULL, NOT 0)
    #     - Consecutive overlapping brackets in ONE transaction (bseq=1 then bseq=2)
    #     - PL/pgSQL EXCEPTION WHEN subtransaction rollback mid-bracket
    #     - FULL OUTER JOIN, nullable group keys, WHERE/HAVING/BOOL_OR
    #     - 4-table (15-term inclusion-exclusion) and 5-table (overlap full-refresh fallback) joins
    #     - Tier 2 single-table view with 2nd-table subquery (uses_group_buckets=True)
    #     - Rich staging column types (COLLATE "C", BPCHAR, VARCHAR(20), NUMERIC(10,2), INT[], ENUM, TIMESTAMPTZ)
    #     - Multi-view concurrent s3_d writer alongside (s3_d, s3_t) writable CTE (AB-BA deadlock freedom)
    #     - PG 18+ VIRTUAL generated column in a Stage 3 join source table
    cur.execute("SHOW server_version_num")
    pg_ver = int(cur.fetchone()[0])
    env.conn.commit()

    adv_schema = [
        "CREATE TYPE s3_tag_enum AS ENUM ('lo', 'hi')",
        (
            "CREATE TABLE s3_a ("
            'g INT PRIMARY KEY, name TEXT COLLATE "C", code BPCHAR, tag s3_tag_enum, '
            "ts TIMESTAMPTZ, arr INT[], amt NUMERIC(10, 2))"
        ),
        (
            "CREATE TABLE s3_b ("
            "id INT PRIMARY KEY, g INT, v INT, flag BOOLEAN, "
            + ("v_dbl INT GENERATED ALWAYS AS (v * 2) VIRTUAL" if pg_ver >= 180000 else "v_dbl INT GENERATED ALWAYS AS (v * 2) STORED")
            + ")"
        ),
        "CREATE TABLE s3_c (cid INT PRIMARY KEY, bid INT, cv INT)",
        "CREATE TABLE s3_e (eid INT PRIMARY KEY, cid INT, ev INT)",
        "CREATE TABLE s3_f (fid INT PRIMARY KEY, eid INT, fv INT)",
        "CREATE TABLE s3_sub (g INT PRIMARY KEY)",
    ]
    adv_views = [
        # Nullable group key + COLLATE "C" + BPCHAR + ENUM + ARRAY + VIRTUAL col + SUM/BOOL_OR/HAVING/WHERE.
        ViewDef(
            "v_s3_rich",
            "SELECT a.name, a.code, a.tag, a.arr, SUM(b.v) AS sv, SUM(b.v_dbl) AS sd, "
            "BOOL_OR(b.flag) AS any_f, COUNT(*) AS c "
            "FROM s3_b b JOIN s3_a a ON b.g = a.g "
            "WHERE b.id > 0 GROUP BY a.name, a.code, a.tag, a.arr HAVING COUNT(*) > 0",
        ),
        # Single-table view sharing s3_a (to stress multi-view lock ordering with v_s3_rich).
        ViewDef(
            "v_s3_a_only",
            "SELECT name, COUNT(*) AS c, SUM(amt) AS sa FROM s3_a GROUP BY name",
        ),
        # FULL OUTER JOIN (Tier 2 multi-source).
        ViewDef(
            "v_s3_full",
            "SELECT COALESCE(a.name, 'none') AS nm, COUNT(a.g) AS ca, COUNT(b.id) AS cb, SUM(b.v) AS sv "
            "FROM s3_a a FULL OUTER JOIN s3_b b ON a.g = b.g GROUP BY COALESCE(a.name, 'none')",
        ),
        # Tier 2 single-table view with 2nd-table subquery (uses_group_buckets=True).
        ViewDef(
            "v_s3_t2_sub",
            "SELECT b.g, COUNT(*) AS c, SUM(b.v) AS sv FROM s3_b b "
            "WHERE b.g IN (SELECT s.g FROM s3_sub s) GROUP BY b.g",
        ),
        # 4-table algebraic join (15 inclusion-exclusion terms).
        ViewDef(
            "v_s3_4",
            "SELECT a.name, COUNT(*) AS c, SUM(e.ev) AS sev "
            "FROM s3_a a JOIN s3_b b ON b.g = a.g JOIN s3_c c ON c.bid = b.id "
            "JOIN s3_e e ON e.cid = c.cid GROUP BY a.name",
        ),
        # 5-table algebraic join (overlap full-refresh fallback).
        ViewDef(
            "v_s3_5",
            "SELECT a.name, COUNT(*) AS c, SUM(f.fv) AS sfv "
            "FROM s3_a a JOIN s3_b b ON b.g = a.g JOIN s3_c c ON c.bid = b.id "
            "JOIN s3_e e ON e.cid = c.cid JOIN s3_f f ON f.eid = e.eid GROUP BY a.name",
        ),
    ]
    adv_env = make_env(views=[], schema_sql=adv_schema, seed=False)
    acur = adv_env.conn.cursor()
    # Seed an orphan row in s3_b (g=1, v=10) with NO parent in s3_a yet.
    acur.execute("INSERT INTO s3_b (id, g, v, flag) VALUES (10, 1, 10, TRUE)")
    acur.execute("INSERT INTO s3_sub (g) VALUES (1)")
    adv_env.conn.commit()
    adv_env.views = adv_views
    for vd in adv_views:
        adv_env.register(vd)

    def _check_adv(label: str) -> None:
        for vd in adv_views:
            acur.execute(f"SELECT * FROM {vd.name} LIMIT 0")
            mv_oids = [d.type_code for d in acur.description]
            cols = [d[0] for d in acur.description]
            acur.execute(f"SELECT * FROM ({vd.select}) AS q LIMIT 0")
            live_oids = [d.type_code for d in acur.description]
            assert mv_oids == live_oids, f"{label}: {vd.name} OIDs {mv_oids} != {live_oids}"
            sel_cols = ", ".join(cols)
            acur.execute(
                f"(SELECT {sel_cols} FROM {vd.name} EXCEPT ALL SELECT {sel_cols} FROM ({vd.select}) AS q) "
                f"UNION ALL "
                f"(SELECT {sel_cols} FROM ({vd.select}) AS q EXCEPT ALL SELECT {sel_cols} FROM {vd.name})"
            )
            diff = acur.fetchall()
            assert diff == [], f"{label}: {vd.name} diff={diff}"
        adv_env.conn.commit()

    _check_adv("adv initial registration")

    # Critical #2 regression: INSERT parent s3_a(g=1) + UPDATE orphan s3_b(v=10 -> NULL) in ONE CTE.
    # Surviving group 'alpha' has 1 row with v=NULL, so SUM(b.v) MUST be NULL (not 0!).
    acur.execute(
        "WITH ins_a AS ("
        "  INSERT INTO s3_a (g, name, code, tag, ts, arr, amt) "
        "  VALUES (1, 'alpha', 'LONG_BPCHAR_VALUE', 'hi', '2026-01-01 00:00:00+00', ARRAY[1, 2], 12.34) "
        "  RETURNING g"
        ") UPDATE s3_b SET v = NULL WHERE id = 10"
    )
    adv_env.conn.commit()
    _check_adv("orphan v->NULL + parent INSERT must yield SUM=NULL not 0")
    acur.execute("SELECT sv, sd FROM v_s3_rich WHERE name = 'alpha'")
    assert acur.fetchall() == [(None, None)]
    adv_env.conn.commit()

    # Consecutive overlapping brackets in ONE transaction (bseq=1 then bseq=2) +
    # PL/pgSQL EXCEPTION WHEN subtransaction rollback mid-bracket!
    acur.execute(
        "WITH ins_a AS ("
        "  INSERT INTO s3_a (g, name, code, tag, ts, arr, amt) "
        "  VALUES (2, NULL, 'B2', 'lo', '2026-01-02 00:00:00+00', ARRAY[3], 5.00) RETURNING g"
        "), ins_b AS ("
        "  INSERT INTO s3_b (id, g, v, flag) VALUES (20, 2, 7, FALSE) RETURNING id"
        "), ins_c AS ("
        "  INSERT INTO s3_c (cid, bid, cv) VALUES (200, 20, 11) RETURNING cid"
        "), ins_e AS ("
        "  INSERT INTO s3_e (eid, cid, ev) VALUES (2000, 200, 13) RETURNING eid"
        ") INSERT INTO s3_f (fid, eid, fv) VALUES (20000, 2000, 17)"
    )
    # Second overlapping bracket in the SAME transaction (bseq=2) + Tier 2 subquery table overlap!
    acur.execute(
        "WITH ins_sub AS (INSERT INTO s3_sub (g) VALUES (2) RETURNING g) "
        "UPDATE s3_b SET v = 9 WHERE id = 20"
    )
    # PL/pgSQL EXCEPTION block catching a duplicate-key failure mid-bracket.
    acur.execute(
        """
        DO $$
        BEGIN
            BEGIN
                WITH ins_a AS (INSERT INTO s3_a (g, name, code, tag, ts, arr, amt)
                               VALUES (3, 'rolled_exc', 'R', 'lo', now(), ARRAY[9], 1.00) RETURNING g)
                INSERT INTO s3_b (id, g, v, flag) VALUES (20, 3, 99, TRUE);
            EXCEPTION WHEN unique_violation THEN
                NULL;
            END;
        END $$;
        """
    )
    # 0-row multi-source UPDATE must not fail and must keep exactness.
    acur.execute("WITH u AS (UPDATE s3_a SET amt = 0 WHERE g = -999) UPDATE s3_b SET v = 0 WHERE id = -999")
    adv_env.conn.commit()
    _check_adv("consecutive brackets + 4/5-table joins + Tier 2 subquery + EXCEPTION rollback + 0-row CTE")

    # Multi-view deadlock freedom: concurrent writer updating s3_a (shared by v_s3_rich & v_s3_a_only)
    # while another connection runs (s3_a, s3_b) writable CTEs.
    mv1, mv2 = adv_env.connect(), adv_env.connect()
    try:
        go = threading.Event()
        mv_errs: List[Optional[BaseException]] = [None, None]

        def _mv_cte() -> None:
            go.wait()
            try:
                c1 = mv1.cursor()
                for k in range(10):
                    c1.execute(
                        "WITH ua AS (UPDATE s3_a SET amt = amt + 1 WHERE g = 1 RETURNING g) "
                        "UPDATE s3_b SET v = COALESCE(v, 0) + 1 WHERE id = 10"
                    )
                    mv1.commit()
            except BaseException as exc:
                mv1.rollback()
                mv_errs[0] = exc

        def _mv_single_a() -> None:
            go.wait()
            try:
                c2 = mv2.cursor()
                for k in range(10):
                    c2.execute("UPDATE s3_a SET amt = amt + 2 WHERE g = 1")
                    mv2.commit()
            except BaseException as exc:
                mv2.rollback()
                mv_errs[1] = exc

        th1 = threading.Thread(target=_mv_cte)
        th2 = threading.Thread(target=_mv_single_a)
        th1.start()
        th2.start()
        go.set()
        th1.join()
        th2.join()
        assert mv_errs == [None, None], f"multi-view deadlock/error: {mv_errs}"
    finally:
        mv1.close()
        mv2.close()
    _check_adv("multi-view concurrent shared-source s3_a writer + (s3_a, s3_b) writable CTE")


def test_pg_function_and_trigger_tampering_detected_and_repaired(make_env) -> None:
    """Out-of-band tampering with PostgreSQL maintenance functions, helper
    functions, search_path settings, or triggers (same OID via CREATE OR REPLACE
    FUNCTION, ALTER FUNCTION RESET/SET search_path, dropped helper function,
    repointed leaf-partition trigger, or added WHEN predicate) is reported by
    check_schema_drift() and sync_views(), and repair_view() rebuilds and
    restores exact maintenance."""
    schema = [
        "CREATE TABLE tp (id INT NOT NULL, g INT, v BIGINT, PRIMARY KEY (id, g)) PARTITION BY RANGE (id)",
        "CREATE TABLE tp_lo PARTITION OF tp FOR VALUES FROM (0) TO (100)",
        "CREATE TABLE tp_hi PARTITION OF tp FOR VALUES FROM (100) TO (200)",
        "CREATE TABLE a (g INT PRIMARY KEY, label TEXT NOT NULL)",
        "CREATE TABLE b (id INT PRIMARY KEY, g INT NOT NULL, x BIGINT NOT NULL)",
        "CREATE TABLE ngrp (id INT PRIMARY KEY, g NUMERIC(10, 2), v BIGINT NOT NULL)",
    ]
    seed = [
        "INSERT INTO tp VALUES (1, 1, 10), (101, 2, 20)",
        "INSERT INTO a VALUES (1, 'one'), (2, 'two')",
        "INSERT INTO b VALUES (10, 1, 5), (20, 2, 7)",
        "INSERT INTO ngrp VALUES (1, 1.0, 10), (2, 1.00, 20), (3, NULL, 30)",
    ]
    views = [
        ViewDef("v_part", "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM tp GROUP BY g"),
        ViewDef("v_s3", "SELECT a.g, a.label, COUNT(*) AS c, SUM(b.x) AS sx FROM a JOIN b ON b.g = a.g GROUP BY a.g, a.label"),
        ViewDef("v_key", "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM ngrp GROUP BY g"),
    ]
    env = make_env(views=[], schema_sql=schema, seed=False)
    for stmt in seed:
        env.execute(stmt)
    env.commit()
    env.views = views
    for vd in views:
        env.register(vd)
    env.check("initial")
    assert env.manager.check_schema_drift() == {"v_part": "ok", "v_s3": "ok", "v_key": "ok"}
    env.commit()
    view_sqls = [f"CREATE VIEW {vd.name} AS {vd.select};" for vd in views]

    def _expect_drift_and_repair(
        tampered_view: str, expected_status: str, label: str, post_dml: str
    ) -> None:
        drift = env.manager.check_schema_drift()
        env.commit()
        assert drift[tampered_view] == expected_status, f"{label}: check_schema_drift={drift}"
        for other in ("v_part", "v_s3", "v_key"):
            if other != tampered_view:
                assert drift[other] == "ok", f"{label}: unrelated view drifted: {drift}"
        sync_res = env.manager.sync_views(view_sqls)
        env.commit()
        assert sync_res[tampered_view] == expected_status, f"{label}: sync_views={sync_res}"
        rep = env.manager.repair_view(tampered_view)
        env.commit()
        assert rep == {tampered_view: "rebuilt"}, f"{label}: repair_view={rep}"
        assert env.manager.check_schema_drift() == {"v_part": "ok", "v_s3": "ok", "v_key": "ok"}
        env.commit()
        env.execute(post_dml)
        env.commit()
        env.check(f"after repair ({label})")

    cat = env.manager.get_catalog()
    env.commit()

    # 1. Hand-replace a maintenance trigger function (CREATE OR REPLACE keeps OID).
    part_ins_fn = next(
        fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai")
    )
    cur = env.conn.cursor()
    cur.execute("SELECT oid FROM pg_proc WHERE proname = %s AND pg_function_is_visible(oid)", (part_ins_fn,))
    oid_before = cur.fetchone()[0]
    cur.execute(
        f"CREATE OR REPLACE FUNCTION {part_ins_fn}() RETURNS TRIGGER AS $$ "
        f"BEGIN RETURN NULL; END; $$ LANGUAGE plpgsql SET search_path FROM CURRENT;"
    )
    cur.execute("SELECT oid FROM pg_proc WHERE proname = %s AND pg_function_is_visible(oid)", (part_ins_fn,))
    assert cur.fetchone()[0] == oid_before, "CREATE OR REPLACE FUNCTION must keep the OID"
    env.commit()
    _expect_drift_and_repair("v_part", "triggers_modified", "replace_trigger_fn_same_oid", "INSERT INTO tp VALUES (2, 1, 15)")

    # 2. Strip search_path via ALTER FUNCTION ... RESET search_path.
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_fn = next(
        fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai")
    )
    env.execute(f"ALTER FUNCTION {part_ins_fn}() RESET search_path;")
    env.commit()
    _expect_drift_and_repair("v_part", "triggers_modified", "reset_search_path", "INSERT INTO tp VALUES (3, 1, 25)")

    # 3. Redirect search_path to pg_catalog (drops the view's own schema).
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_fn = next(
        fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai")
    )
    env.execute(f"ALTER FUNCTION {part_ins_fn}() SET search_path TO pg_catalog;")
    env.commit()
    _expect_drift_and_repair("v_part", "triggers_modified", "redirect_search_path", "INSERT INTO tp VALUES (4, 1, 5)")

    # 4. Hand-replace Stage 3 reconcile helper function.
    cat = env.manager.get_catalog()
    env.commit()
    s3_rec_fn = next(fn for fn in cat["v_s3"]["function_names"] if "_reconcile" in fn)
    env.execute(
        f"CREATE OR REPLACE FUNCTION {s3_rec_fn}() RETURNS VOID AS $$ "
        f"BEGIN RETURN; END; $$ LANGUAGE plpgsql SET search_path FROM CURRENT;"
    )
    env.commit()
    _expect_drift_and_repair(
        "v_s3",
        "triggers_modified",
        "replace_s3_reconcile_fn",
        "WITH ia AS (INSERT INTO a VALUES (3, 'three') RETURNING g) INSERT INTO b VALUES (30, 3, 9)",
    )

    # 5. Hand-replace canonical key helper function (__matview_key_*(anyelement)).
    cat = env.manager.get_catalog()
    env.commit()
    key_fn = next(fn for fn in cat["v_key"]["function_names"] if fn.endswith("(anyelement)"))
    key_fn_bare = key_fn[: -len("(anyelement)")]
    env.execute(
        f"CREATE OR REPLACE FUNCTION {key_fn_bare}(v anyelement) RETURNS text "
        f"LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE SET search_path FROM CURRENT AS $$ "
        f"BEGIN RETURN 'tampered'; END; $$;"
    )
    env.commit()
    _expect_drift_and_repair("v_key", "triggers_modified", "replace_key_text_fn", "INSERT INTO ngrp VALUES (4, 1.000, 40)")

    # 6. Hand-replace read-only view trigger function, and verify read-only
    # rejection is actually restored after repair_view().
    cat = env.manager.get_catalog()
    env.commit()
    ro_fn = next(fn for fn in cat["v_key"]["function_names"] if "_fn_readonly_" in fn)
    env.execute(
        f'CREATE OR REPLACE FUNCTION "{ro_fn}"() RETURNS TRIGGER AS $$ '
        f"BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql;"
    )
    env.commit()
    _expect_drift_and_repair("v_key", "triggers_modified", "replace_readonly_fn", "INSERT INTO ngrp VALUES (5, 2.50, 50)")
    with pytest.raises(Exception, match="read-only"):
        env.execute("INSERT INTO v_key VALUES (1.0, 1, 1)")
    env.conn.rollback()

    # 7. Repoint a leaf-partition trigger with the same name to a no-op function.
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_trg = next(
        t for t in cat["v_part"]["trigger_names"] if t.endswith("_ai")
    )
    env.execute(
        "CREATE OR REPLACE FUNCTION __mv_test_noop() RETURNS TRIGGER AS $$ "
        "BEGIN RETURN NULL; END; $$ LANGUAGE plpgsql SET search_path FROM CURRENT;"
    )
    env.execute(f"DROP TRIGGER {part_ins_trg} ON tp_hi;")
    env.execute(
        f"CREATE TRIGGER {part_ins_trg} AFTER INSERT ON tp_hi "
        f"REFERENCING NEW TABLE AS __mv_new FOR EACH STATEMENT EXECUTE FUNCTION __mv_test_noop();"
    )
    env.commit()
    _expect_drift_and_repair("v_part", "triggers_modified", "repoint_leaf_partition_trigger", "INSERT INTO tp_hi VALUES (102, 2, 30)")

    # 8. Recreate trigger pointing to the REAL function but with a WHEN (false) predicate (tgqual IS NOT NULL).
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_trg = next(
        t for t in cat["v_part"]["trigger_names"] if t.endswith("_ai")
    )
    part_ins_fn = next(
        fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai")
    )
    env.execute(f"DROP TRIGGER {part_ins_trg} ON tp_hi;")
    env.execute(
        f"CREATE TRIGGER {part_ins_trg} AFTER INSERT ON tp_hi "
        f"REFERENCING NEW TABLE AS __mv_new FOR EACH STATEMENT WHEN (false) EXECUTE FUNCTION {part_ins_fn}();"
    )
    env.commit()
    _expect_drift_and_repair("v_part", "triggers_modified", "add_when_false_qual_on_partition", "INSERT INTO tp_hi VALUES (103, 2, 35)")

    # 9. BEFORE ... UPDATE OF <col> tampering (tgattr non-empty with identical
    # tgtype/tgoldtable/tgnewtable/tgqual/tgfoid) must be caught.
    cat = env.manager.get_catalog()
    env.commit()
    s3_bs_trg = next(t for t in cat["v_s3"]["trigger_names"] if t.endswith("__a_bs"))
    s3_bs_fn = next(fn for fn in cat["v_s3"]["function_names"] if fn.endswith("__a_bs"))
    env.execute(f"DROP TRIGGER {s3_bs_trg} ON a;")
    env.execute(
        f"CREATE TRIGGER {s3_bs_trg} BEFORE INSERT OR UPDATE OF g OR DELETE OR TRUNCATE ON a "
        f"FOR EACH STATEMENT EXECUTE FUNCTION {s3_bs_fn}();"
    )
    env.commit()
    _expect_drift_and_repair("v_s3", "triggers_modified", "update_of_col_filter_tgattr", "UPDATE a SET label = 'ONE' WHERE g = 1")

    # 10. Cross-schema decoy function with identical name and body must be caught.
    import uuid as _uuid
    decoy_schema = f"mv_decoy_{_uuid.uuid4().hex[:8]}"
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_trg = next(t for t in cat["v_part"]["trigger_names"] if t.endswith("_ai"))
    part_ins_fn = next(fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai"))
    cur.execute(
        "SELECT prosrc FROM pg_proc WHERE proname = %s AND pg_function_is_visible(oid)",
        (part_ins_fn,),
    )
    real_src = cur.fetchone()[0]
    try:
        cur.execute(f'CREATE SCHEMA "{decoy_schema}"')
        cur.execute(
            f'CREATE FUNCTION "{decoy_schema}"."{part_ins_fn}"() RETURNS TRIGGER '
            f'LANGUAGE plpgsql SET search_path TO "{decoy_schema}" AS $decoy${real_src}$decoy$;'
        )
        cur.execute(f"DROP TRIGGER {part_ins_trg} ON tp_hi;")
        cur.execute(
            f'CREATE TRIGGER {part_ins_trg} AFTER INSERT ON tp_hi '
            f'REFERENCING NEW TABLE AS __mv_new_rows FOR EACH STATEMENT '
            f'EXECUTE FUNCTION "{decoy_schema}"."{part_ins_fn}"();'
        )
        env.commit()
        _expect_drift_and_repair("v_part", "triggers_modified", "cross_schema_decoy_fn", "INSERT INTO tp_hi VALUES (104, 2, 40)")
    finally:
        env.conn.rollback()
        cur.execute(f'DROP SCHEMA IF EXISTS "{decoy_schema}" CASCADE')
        env.commit()

    # 11. Redirect search_path to '"$user", public' inside a non-public, non-$user schema.
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_fn = next(fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai"))
    env.execute(f'ALTER FUNCTION {part_ins_fn}() SET search_path TO "$user", public;')
    env.commit()
    _expect_drift_and_repair("v_part", "triggers_modified", "dollar_user_search_path_in_other_schema", "INSERT INTO tp VALUES (6, 1, 6)")

    # 12. SECURITY DEFINER tampering on trigger function.
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_fn = next(fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai"))
    env.execute(f"ALTER FUNCTION {part_ins_fn}() SECURITY DEFINER;")
    env.commit()
    _expect_drift_and_repair("v_part", "triggers_modified", "security_definer_fn", "INSERT INTO tp VALUES (7, 1, 7)")

    # 13. Volatility (VOLATILE) and strictness (CALLED ON NULL INPUT) tampering on __matview_key_*.
    cat = env.manager.get_catalog()
    env.commit()
    key_fn = next(fn for fn in cat["v_key"]["function_names"] if fn.endswith("(anyelement)"))
    env.execute(f"ALTER FUNCTION {key_fn} VOLATILE;")
    env.commit()
    _expect_drift_and_repair("v_key", "triggers_modified", "key_fn_volatile", "INSERT INTO ngrp VALUES (8, 3.00, 8)")

    cat = env.manager.get_catalog()
    env.commit()
    key_fn = next(fn for fn in cat["v_key"]["function_names"] if fn.endswith("(anyelement)"))
    env.execute(f"ALTER FUNCTION {key_fn} CALLED ON NULL INPUT;")
    env.commit()
    _expect_drift_and_repair("v_key", "triggers_modified", "key_fn_not_strict", "INSERT INTO ngrp VALUES (9, NULL, 9)")

    # 14. PROCEDURE substitution for Stage 3 reconcile function (prokind='p' with identical body).
    cat = env.manager.get_catalog()
    env.commit()
    s3_rec_fn = next(fn for fn in cat["v_s3"]["function_names"] if "_reconcile" in fn)
    cur.execute("SELECT prosrc FROM pg_proc WHERE proname = %s AND pg_function_is_visible(oid)", (s3_rec_fn,))
    rec_src = cur.fetchone()[0]
    cur.execute(f"DROP FUNCTION {s3_rec_fn}();")
    cur.execute(
        f"CREATE PROCEDURE {s3_rec_fn}() LANGUAGE plpgsql SET search_path FROM CURRENT AS $p${rec_src}$p$;"
    )
    env.commit()
    drift_proc = env.manager.check_schema_drift()
    env.commit()
    assert drift_proc["v_s3"] == "triggers_modified", drift_proc
    # Clean up the spoofed procedure before repair_view (DROP FUNCTION won't drop a PROCEDURE).
    cur.execute(f"DROP PROCEDURE {s3_rec_fn}();")
    env.commit()
    _expect_drift_and_repair(
        "v_s3",
        "triggers_missing",
        "drop_s3_reconcile_fn",
        "WITH ua AS (UPDATE a SET label = 'THREE' WHERE g = 3 RETURNING g) UPDATE b SET x = 11 WHERE id = 30",
    )

    # 15. Priority precedence: disabled > modified > orphaned; missing > disabled.
    cat = env.manager.get_catalog()
    env.commit()
    part_ins_fn = next(fn for fn in cat["v_part"]["function_names"] if "_fn_sync_" in fn and fn.endswith("_ai"))
    part_ins_trg = next(t for t in cat["v_part"]["trigger_names"] if t.endswith("_ai"))
    env.execute(f"ALTER FUNCTION {part_ins_fn}() RESET search_path;")
    env.execute(f"ALTER TABLE tp_hi DISABLE TRIGGER {part_ins_trg};")
    env.commit()
    assert env.manager.check_schema_drift()["v_part"] == "triggers_disabled"
    env.commit()
    env.execute(f"DROP TRIGGER {part_ins_trg} ON tp_hi;")
    env.commit()
    assert env.manager.check_schema_drift()["v_part"] == "triggers_missing"
    env.commit()
    assert env.manager.repair_view("v_part") == {"v_part": "rebuilt"}
    env.commit()
    assert env.manager.check_schema_drift() == {"v_part": "ok", "v_s3": "ok", "v_key": "ok"}
    env.commit()


def test_pg_dump_restore_fresh_oids_stays_ok_and_exact(make_env) -> None:
    """Recreating every PostgreSQL maintenance function and trigger under
    brand-new OIDs via pg_get_functiondef / pg_get_triggerdef (running
    unconditionally on every PG CI runner) keeps check_schema_drift() == 'ok',
    sync_views() == 'unchanged', read-only enforcement active, and exact DML
    maintenance across partitions, canonical keys, and Stage 3 writable CTEs."""
    schema = [
        "CREATE TABLE tp (id INT NOT NULL, g INT, v BIGINT, PRIMARY KEY (id, g)) PARTITION BY RANGE (id)",
        "CREATE TABLE tp_lo PARTITION OF tp FOR VALUES FROM (0) TO (100)",
        "CREATE TABLE tp_hi PARTITION OF tp FOR VALUES FROM (100) TO (200)",
        "CREATE TABLE a (g INT PRIMARY KEY, label TEXT NOT NULL)",
        "CREATE TABLE b (id INT PRIMARY KEY, g INT NOT NULL, x BIGINT NOT NULL)",
        "CREATE TABLE ngrp (id INT PRIMARY KEY, g NUMERIC(10, 2), v BIGINT NOT NULL)",
    ]
    seed = [
        "INSERT INTO tp VALUES (1, 1, 10), (101, 2, 20)",
        "INSERT INTO a VALUES (1, 'one'), (2, 'two')",
        "INSERT INTO b VALUES (10, 1, 5), (20, 2, 7)",
        "INSERT INTO ngrp VALUES (1, 1.0, 10), (2, 1.00, 20), (3, NULL, 30)",
    ]
    views = [
        ViewDef("v_part", "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM tp GROUP BY g"),
        ViewDef("v_s3", "SELECT a.g, a.label, COUNT(*) AS c, SUM(b.x) AS sx FROM a JOIN b ON b.g = a.g GROUP BY a.g, a.label"),
        ViewDef("v_key", "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM ngrp GROUP BY g"),
    ]
    env = make_env(views=[], schema_sql=schema, seed=False)
    for stmt in seed:
        env.execute(stmt)
    env.commit()
    env.views = views
    for vd in views:
        env.register(vd)
    env.check("source initial")
    view_sqls = [f"CREATE VIEW {vd.name} AS {vd.select};" for vd in views]

    cur = env.conn.cursor()
    cur.execute("SELECT current_schema()")
    src_schema = cur.fetchone()[0]
    cur.execute(
        "SELECT proname, oid FROM pg_proc WHERE pronamespace = %s::regnamespace ORDER BY proname",
        (src_schema,),
    )
    src_fn_oids = dict(cur.fetchall())
    assert src_fn_oids
    env.commit()

    # Recreate every function and trigger from pg_get_functiondef /
    # pg_get_triggerdef after DROP + CREATE (giving every function and trigger
    # a brand-new OID, just as pg_dump/pg_restore does).
    cur.execute(
        "SELECT p.oid, p.proname, pg_catalog.pg_get_functiondef(p.oid),"
        " pg_catalog.pg_get_function_identity_arguments(p.oid)"
        " FROM pg_catalog.pg_proc p WHERE p.pronamespace = %s::regnamespace ORDER BY p.oid",
        (src_schema,),
    )
    fn_defs = cur.fetchall()
    cur.execute(
        "SELECT c.relname, t.tgname, pg_catalog.pg_get_triggerdef(t.oid)"
        " FROM pg_catalog.pg_trigger t JOIN pg_catalog.pg_class c ON c.oid = t.tgrelid"
        " WHERE c.relnamespace = %s::regnamespace AND NOT t.tgisinternal ORDER BY t.oid",
        (src_schema,),
    )
    trg_defs = cur.fetchall()
    for relname, tgname, _ in trg_defs:
        cur.execute(f'DROP TRIGGER "{tgname}" ON "{src_schema}"."{relname}";')
    for _, proname, _, ident_args in fn_defs:
        cur.execute(f'DROP FUNCTION "{src_schema}"."{proname}"({ident_args});')
    for _, _, fdef, _ in fn_defs:
        cur.execute(fdef)
    for _, _, tdef in trg_defs:
        cur.execute(tdef)
    env.commit()

    cur.execute(
        "SELECT proname, oid FROM pg_proc WHERE pronamespace = %s::regnamespace ORDER BY proname",
        (src_schema,),
    )
    new_fn_oids = dict(cur.fetchall())
    assert set(new_fn_oids.keys()) == set(src_fn_oids.keys())
    for fn_name, old_oid in src_fn_oids.items():
        assert new_fn_oids[fn_name] != old_oid, f"function {fn_name} kept old OID {old_oid}"
    env.commit()

    assert env.manager.check_schema_drift() == {"v_part": "ok", "v_s3": "ok", "v_key": "ok"}
    env.commit()
    assert env.manager.sync_views(view_sqls) == {"v_part": "unchanged", "v_s3": "unchanged", "v_key": "unchanged"}
    env.commit()
    with pytest.raises(Exception, match="read-only"):
        env.execute("INSERT INTO v_key VALUES (1.0, 1, 1)")
    env.conn.rollback()
    env.execute("INSERT INTO tp VALUES (5, 1, 11)")
    env.execute("INSERT INTO tp_hi VALUES (105, 2, 22)")
    env.execute("WITH ia AS (INSERT INTO a VALUES (4, 'four') RETURNING g) INSERT INTO b VALUES (40, 4, 13)")
    env.execute("INSERT INTO ngrp VALUES (6, 1.000, 7)")
    env.commit()
    env.check("after fresh-OID recreate")


def test_pg_dump_restore_real_pg_dump_cli(make_env) -> None:
    """End-to-end pg_dump -> psql into a fresh database (explicitly skipped
    when pg_dump/psql client binaries or CREATEDB privilege are absent)."""
    import shutil
    import subprocess
    import uuid
    import psycopg2
    import psycopg2.extensions
    from matview import Dialect, MaterializedViewManager

    pg_dump_bin = shutil.which("pg_dump")
    psql_bin = shutil.which("psql")
    if not (pg_dump_bin and psql_bin and PG_DSN):
        pytest.skip("pg_dump and psql client binaries are required for real CLI dump/restore test")

    schema = [
        "CREATE TABLE tp (id INT NOT NULL, g INT, v BIGINT, PRIMARY KEY (id, g)) PARTITION BY RANGE (id)",
        "CREATE TABLE tp_lo PARTITION OF tp FOR VALUES FROM (0) TO (100)",
        "CREATE TABLE tp_hi PARTITION OF tp FOR VALUES FROM (100) TO (200)",
        "CREATE TABLE a (g INT PRIMARY KEY, label TEXT NOT NULL)",
        "CREATE TABLE b (id INT PRIMARY KEY, g INT NOT NULL, x BIGINT NOT NULL)",
        "CREATE TABLE ngrp (id INT PRIMARY KEY, g NUMERIC(10, 2), v BIGINT NOT NULL)",
    ]
    seed = [
        "INSERT INTO tp VALUES (1, 1, 10), (101, 2, 20)",
        "INSERT INTO a VALUES (1, 'one'), (2, 'two')",
        "INSERT INTO b VALUES (10, 1, 5), (20, 2, 7)",
        "INSERT INTO ngrp VALUES (1, 1.0, 10), (2, 1.00, 20), (3, NULL, 30)",
    ]
    views = [
        ViewDef("v_part", "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM tp GROUP BY g"),
        ViewDef("v_s3", "SELECT a.g, a.label, COUNT(*) AS c, SUM(b.x) AS sx FROM a JOIN b ON b.g = a.g GROUP BY a.g, a.label"),
        ViewDef("v_key", "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM ngrp GROUP BY g"),
    ]
    env = make_env(views=[], schema_sql=schema, seed=False)
    cur = env.conn.cursor()
    cur.execute("SELECT rolcreatedb OR rolsuper FROM pg_roles WHERE rolname = current_user")
    can_createdb = bool(cur.fetchone()[0])
    env.commit()
    if not can_createdb:
        pytest.skip("current PostgreSQL role lacks CREATEDB privilege")

    for stmt in seed:
        env.execute(stmt)
    env.commit()
    env.views = views
    for vd in views:
        env.register(vd)
    env.check("source initial")
    view_sqls = [f"CREATE VIEW {vd.name} AS {vd.select};" for vd in views]
    cur.execute("SELECT current_schema()")
    src_schema = cur.fetchone()[0]
    env.commit()

    dst_db = f"mv_dump_{uuid.uuid4().hex[:8]}"
    dsn_params = psycopg2.extensions.parse_dsn(PG_DSN)
    dst_dsn = psycopg2.extensions.make_dsn(**{**dsn_params, "dbname": dst_db})
    admin_conn = psycopg2.connect(PG_DSN)
    admin_conn.autocommit = True
    try:
        with admin_conn.cursor() as acur:
            acur.execute(f'CREATE DATABASE "{dst_db}"')
        dump_res = subprocess.run(
            [pg_dump_bin, "-d", PG_DSN, "-n", src_schema],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [psql_bin, "-v", "ON_ERROR_STOP=1", "-d", dst_dsn],
            input=dump_res.stdout,
            check=True,
            capture_output=True,
            text=True,
        )
        dst_conn = psycopg2.connect(dst_dsn)
        try:
            dcur = dst_conn.cursor()
            dcur.execute(f'SET search_path TO "{src_schema}", public')
            dst_conn.commit()
            dmgr = MaterializedViewManager(dst_conn, dialect=Dialect.POSTGRES)
            assert dmgr.check_schema_drift() == {"v_part": "ok", "v_s3": "ok", "v_key": "ok"}
            dst_conn.commit()
            assert dmgr.sync_views(view_sqls) == {"v_part": "unchanged", "v_s3": "unchanged", "v_key": "unchanged"}
            dst_conn.commit()
            with pytest.raises(Exception, match="read-only"):
                dcur.execute("INSERT INTO v_key VALUES (1.0, 1, 1)")
            dst_conn.rollback()
            dcur.execute(f'SET search_path TO "{src_schema}", public')
            dcur.execute("INSERT INTO tp VALUES (6, 1, 100)")
            dcur.execute("INSERT INTO tp_hi VALUES (106, 2, 200)")
            dcur.execute("WITH ia AS (INSERT INTO a VALUES (5, 'five') RETURNING g) INSERT INTO b VALUES (50, 5, 19)")
            dcur.execute("INSERT INTO ngrp VALUES (7, 1.0, 99)")
            dst_conn.commit()
            for vd in views:
                dcur.execute(f"SELECT * FROM {vd.name} ORDER BY 1, 2")
                got = dcur.fetchall()
                dcur.execute(f"SELECT * FROM ({vd.select}) _q ORDER BY 1, 2")
                exp = dcur.fetchall()
                assert got == exp, f"pg_dump/restore mismatch on {vd.name}: {got} != {exp}"
            dst_conn.commit()
        finally:
            dst_conn.close()
    finally:
        with admin_conn.cursor() as acur:
            acur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (dst_db,),
            )
            acur.execute(f'DROP DATABASE IF EXISTS "{dst_db}"')
        admin_conn.close()
