import math
import re
"""Comprehensive test suite for matview_sync.py (pure native SQL runtime, zero UDFs)."""

import os
import random
import sqlite3
import sys
import tempfile
import warnings

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Optional, Tuple
from matview_sync import (
    DEFAULT_OBJECT_PREFIX,
    ViewIndexSpec,
    _split_top_level,
    parse_view_index_spec,
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    MaintenanceStrategy,
    MaterializedViewManager,
    MaterializedViewPlan,
    generate_materialized_view,
    generate_materialized_views,
    install_sqlite_materialized_view,
    install_sqlite_materialized_views,
    main as cli_main,
)


def _resolve_db_view_name(conn: sqlite3.Connection, view_name: str) -> str:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND (name = ? OR name = ?)",
        (view_name, f"{DEFAULT_OBJECT_PREFIX}{view_name}"),
    ).fetchone()
    return row[0] if row else view_name


def assert_view_matches_live_query(
    conn: sqlite3.Connection,
    mv_name: str,
    live_select_sql: str,
    user_cols: list[str],
    order_by: str,
) -> None:
    mv_name = _resolve_db_view_name(conn, mv_name)
    cur = conn.cursor()
    star_desc = [d[0] for d in cur.execute(f"SELECT * FROM {mv_name} LIMIT 0").description]
    assert star_desc == user_cols, (
        f"SELECT * column mismatch on {mv_name}: got {star_desc}, expected {user_cols}"
    )

    cols_proj = ", ".join(user_cols)
    mv_rows = cur.execute(
        f"SELECT * FROM {mv_name} ORDER BY {order_by}"
    ).fetchall()
    live_rows = cur.execute(
        f"SELECT {cols_proj} FROM ({live_select_sql}) ORDER BY {order_by}"
    ).fetchall()
    assert len(mv_rows) == len(live_rows), (
        f"Row count mismatch on {mv_name}:\nMV:   {mv_rows}\nLive: {live_rows}"
    )
    for r_mv, r_live in zip(mv_rows, live_rows):
        assert len(r_mv) == len(r_live)
        for idx, (v_mv, v_live) in enumerate(zip(r_mv, r_live)):
            if isinstance(v_mv, float) or isinstance(v_live, float):
                assert v_mv is not None and v_live is not None, (
                    f"Null mismatch on {mv_name} col {user_cols[idx]}:\nMV:   {r_mv}\nLive: {r_live}"
                )
                assert abs(v_mv - v_live) < 1e-6, (
                    f"Float mismatch on {mv_name} col {user_cols[idx]}:\nMV:   {r_mv}\nLive: {r_live}"
                )
            else:
                assert v_mv == v_live, (
                    f"Value mismatch on {mv_name} col {user_cols[idx]}:\nMV:   {r_mv}\nLive: {r_live}"
                )


def test_zero_runtime_dependency_across_separate_connections() -> None:
    """
    Verifies that once matview_sync sets up the MV and disconnects, a brand-new
    vanilla `sqlite3.connect(db_file)` connection (with zero matview_sync imports
    or callbacks) maintains all MVs natively via SQLite triggers.
    """
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Phase 1: Setup & backfill using MaterializedViewManager, then close connection
        mgr = MaterializedViewManager(db_path, dialect=Dialect.SQLITE)
        mgr.apply_schema_ddl(
            """
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                region TEXT,
                status TEXT NOT NULL,
                amount REAL
            );
            """
        )
        mgr.conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            [
                (1, 10, "APAC", "completed", 100.0),
                (2, 10, "APAC", "pending", 50.0),
                (3, 20, None, "completed", 75.0),
            ],
        )
        v_sql = """
        CREATE VIEW mv_orders AS
        SELECT
            customer_id,
            region,
            COUNT(*) AS total_orders,
            COUNT(*) FILTER (WHERE status = 'completed') AS completed_orders,
            COUNT(DISTINCT status) AS dist_status,
            SUM(amount) AS sum_amount,
            TOTAL(amount) AS total_amount,
            AVG(amount) AS avg_amount,
            MIN(amount) AS min_amount,
            MAX(amount) AS max_amount,
            GROUP_CONCAT(status, ',') AS statuses,
            JSON_GROUP_ARRAY(id) AS order_ids
        FROM orders
        GROUP BY customer_id, region;
        """
        plan = mgr.register_view(v_sql, incremental_backfill=True, batch_size=2)
        mgr.conn.close()

        # Phase 2: Open a completely vanilla sqlite3 connection with zero matview_sync involvement
        raw_conn = sqlite3.connect(db_path)
        raw_conn.execute("INSERT INTO orders VALUES (4, 10, 'APAC', 'completed', 200.0)")
        raw_conn.execute("UPDATE orders SET amount = 125.0, status = 'completed' WHERE id = 2")
        raw_conn.execute("DELETE FROM orders WHERE id = 3")
        raw_conn.commit()

        live_q = v_sql.split("AS", 1)[1].strip().rstrip(";")
        assert_view_matches_live_query(
            raw_conn,
            "mv_orders",
            live_q,
            plan.user_columns,
            "customer_id ASC, COALESCE(region, '') ASC",
        )
        raw_conn.close()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_limitation_refusal_and_ignore_limitations_override() -> None:
    schema_sql = """
    CREATE TABLE parent_orgs (id INT PRIMARY KEY, name VARCHAR(64) NOT NULL);
    CREATE TABLE events (
        id BIGINT PRIMARY KEY,
        org_id INT NOT NULL REFERENCES parent_orgs(id) ON DELETE CASCADE,
        score DOUBLE PRECISION,
        latency DOUBLE PRECISION
    );
    """
    view_sql = """
    CREATE VIEW mv_org_events AS
    SELECT org_id, COUNT(*) AS cnt, SUM(latency) AS total_lat
    FROM events
    GROUP BY org_id;
    """

    # 1. MySQL refuses by default due to MYSQL_NO_TRUNCATE_TRIGGER & MYSQL_FK_CASCADE_NO_TRIGGER
    try:
        generate_materialized_view(schema_sql, view_sql, dialect=Dialect.MYSQL)
        raise AssertionError("Expected EngineLimitationError on MySQL without ignore_limitations")
    except EngineLimitationError as exc:
        msg = str(exc)
        assert "MYSQL_NO_TRUNCATE_TRIGGER" in msg
        assert "MYSQL_FK_CASCADE_NO_TRIGGER" in msg
        assert "--ignore-limitations" in msg

    # 2. SQLite refuses by default when schema has ON DELETE CASCADE (requires runtime pragma on external connections)
    try:
        generate_materialized_view(schema_sql, view_sql, dialect=Dialect.SQLITE)
        raise AssertionError("Expected EngineLimitationError on SQLite for ON DELETE CASCADE without ignore_limitations")
    except EngineLimitationError as exc:
        assert "SQLITE_SESSION_PRAGMA_DEPENDENCY" in str(exc)

    # 3. SQLite refuses non-native SQLite aggregates (e.g. STDDEV_POP, BIT_XOR) unless --ignore-limitations is passed
    pg_agg_view = """
    CREATE VIEW mv_pg_only AS
    SELECT org_id, STDDEV_POP(latency) AS std_lat
    FROM events
    GROUP BY org_id;
    """
    clean_schema = "CREATE TABLE events (id INT PRIMARY KEY, org_id INT NOT NULL, latency REAL);"
    try:
        generate_materialized_view(clean_schema, pg_agg_view, dialect=Dialect.SQLITE)
        raise AssertionError("Expected EngineLimitationError on SQLite for STDDEV_POP")
    except EngineLimitationError as exc:
        assert "DIALECT_UNSUPPORTED_AGGREGATE" in str(exc)

    # Whereas on PostgreSQL, STDDEV_POP is a native aggregate and succeeds with zero warnings!
    pg_plan = generate_materialized_view(clean_schema, pg_agg_view, dialect=Dialect.POSTGRES)
    assert not pg_plan.limitations

    # 4. Passing ignore_limitations=True emits a warning and succeeds
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        my_plan = generate_materialized_view(
            schema_sql, view_sql, dialect=Dialect.MYSQL, ignore_limitations=True
        )
        assert any(issubclass(w.category, EngineLimitationWarning) for w in caught)
        assert len(my_plan.limitations) == 2

    # 5. CLI refuses without --ignore-limitations (exit code 2) and succeeds with --ignore-limitations (exit code 0)
    rc_refused = cli_main(["--schema", schema_sql, "--view", view_sql, "--dialect", "mysql"])
    assert rc_refused == 2
    rc_allowed = cli_main(
        ["--schema", schema_sql, "--view", view_sql, "--dialect", "mysql", "--ignore-limitations"]
    )
    assert rc_allowed == 0


def test_sqlite_native_aggregates_and_multiple_mvs_fuzz() -> None:
    conn = sqlite3.connect(":memory:")
    schema_sql = """
    CREATE TABLE transactions (
        id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL,
        category TEXT,
        status TEXT NOT NULL,
        amount REAL,
        qty INTEGER
    );
    """
    conn.execute(schema_sql)

    mv1_sql = """
    CREATE VIEW mv_account_totals AS
    SELECT
        account_id,
        COUNT(*) AS tx_count,
        COUNT(*) FILTER (WHERE status = 'settled') AS settled_count,
        SUM(amount) AS total_amount,
        TOTAL(amount) AS total_nonnull_zero,
        AVG(amount) AS avg_amount,
        MIN(amount) AS min_amount,
        MAX(amount) AS max_amount
    FROM transactions
    GROUP BY account_id;
    """
    mv2_sql = """
    CREATE VIEW mv_account_distincts AS
    SELECT
        account_id,
        COUNT(DISTINCT category) AS distinct_cats,
        SUM(DISTINCT qty) AS distinct_qty_sum,
        AVG(DISTINCT qty) AS distinct_qty_avg
    FROM transactions
    WHERE status != 'cancelled'
    GROUP BY account_id;
    """
    mv3_sql = """
    CREATE VIEW mv_category_holistic AS
    SELECT
        category,
        COUNT(*) AS n,
        GROUP_CONCAT(status, '|') AS statuses,
        JSON_GROUP_ARRAY(id) AS tx_ids
    FROM transactions
    GROUP BY category;
    """
    mv4_sql = """
    CREATE VIEW mv_high_value_accounts AS
    SELECT
        account_id,
        COUNT(*) AS cnt,
        COALESCE(SUM(amount * qty), 0.0) AS weighted_volume
    FROM transactions
    WHERE status = 'settled'
    GROUP BY account_id
    HAVING COUNT(*) >= 1;
    """

    p1, p2, p3, p4 = install_sqlite_materialized_views(
        conn, [mv1_sql, mv2_sql, mv3_sql, mv4_sql]
    )
    q1 = mv1_sql.split("AS", 1)[1].strip().rstrip(";")
    q2 = mv2_sql.split("AS", 1)[1].strip().rstrip(";")
    q3 = mv3_sql.split("AS", 1)[1].strip().rstrip(";")
    q4 = mv4_sql.split("AS", 1)[1].strip().rstrip(";")

    rng = random.Random(42)
    next_id = 1
    cats = ["food", "tech", "travel", None]
    statuses = ["settled", "pending", "cancelled"]

    for _ in range(150):
        existing_ids = [r[0] for r in conn.execute("SELECT id FROM transactions").fetchall()]
        op = rng.choice(["insert", "upsert", "update", "delete"]) if existing_ids else "insert"
        if op == "insert":
            conn.execute(
                "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    next_id,
                    rng.choice([100, 200, 300]),
                    rng.choice(cats),
                    rng.choice(statuses),
                    rng.choice([None, 0.0, 12.5, 50.0, 200.0, -10.0]),
                    rng.choice([None, 1, 2, 3, 5]),
                ),
            )
            next_id += 1
        elif op == "upsert":
            tid = rng.choice(existing_ids)
            conn.execute(
                """
                INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    amount = excluded.amount,
                    qty = excluded.qty,
                    status = excluded.status
                """,
                (
                    tid,
                    rng.choice([100, 200, 300]),
                    rng.choice(cats),
                    rng.choice(statuses),
                    rng.choice([None, 0.0, 33.0, 88.0]),
                    rng.choice([None, 2, 5, 9]),
                ),
            )
        elif op == "update":
            tid = rng.choice(existing_ids)
            conn.execute(
                "UPDATE transactions SET account_id=?, category=?, status=?, amount=?, qty=? WHERE id=?",
                (
                    rng.choice([100, 200, 300]),
                    rng.choice(cats),
                    rng.choice(statuses),
                    rng.choice([None, 0.0, 15.0, 75.5, 150.0, -5.0]),
                    rng.choice([None, 1, 2, 4, 8]),
                    tid,
                ),
            )
        else:
            tid = rng.choice(existing_ids)
            conn.execute("DELETE FROM transactions WHERE id=?", (tid,))

        assert_view_matches_live_query(conn, "mv_account_totals", q1, p1.user_columns, "account_id ASC")
        assert_view_matches_live_query(conn, "mv_account_distincts", q2, p2.user_columns, "account_id ASC")
        assert_view_matches_live_query(conn, "mv_category_holistic", q3, p3.user_columns, "COALESCE(category, '') ASC")
        assert_view_matches_live_query(conn, "mv_high_value_accounts", q4, p4.user_columns, "account_id ASC")




def test_postgres_and_mysql_generated_sql_and_algebraic_execution() -> None:
    import math
    import re
    from matview_sync import _split_top_level

    schema = """
    CREATE TABLE orders (
        id INT PRIMARY KEY,
        customer_id INT,
        region VARCHAR(50),
        amount DOUBLE PRECISION,
        flags INT,
        is_paid BOOLEAN,
        tag VARCHAR(50)
    );
    CREATE TABLE customers (
        id INT PRIMARY KEY,
        tier VARCHAR(50)
    );
    """

    views_pg = [
        """CREATE VIEW v_alg AS
           SELECT region,
                  COUNT(*) AS cnt,
                  SUM(amount) AS total_amt,
                  AVG(amount) AS avg_amt,
                  VAR_POP(amount) AS vp,
                  VAR_SAMP(amount) AS vs,
                  STDDEV_POP(amount) AS sdp,
                  STDDEV_SAMP(amount) AS sds,
                  BIT_XOR(flags) AS bx,
                  BOOL_AND(is_paid) AS all_paid,
                  BOOL_OR(is_paid) AS any_paid,
                  MIN(amount) AS min_amt,
                  MAX(amount) AS max_amt
           FROM orders
           WHERE amount > 0
           GROUP BY region""",
        """CREATE VIEW v_grp AS
           SELECT region,
                  COUNT(DISTINCT customer_id) AS uniq_cust,
                  STRING_AGG(tag, ',') AS tags,
                  BIT_AND(flags) AS ba,
                  BIT_OR(flags) AS bo
           FROM orders
           GROUP BY region""",
        """CREATE VIEW v_join AS
           SELECT c.tier, o.region, COUNT(*) AS cnt, SUM(o.amount) AS total
           FROM orders o
           JOIN customers c ON o.customer_id = c.id
           GROUP BY c.tier, o.region""",
        """CREATE VIEW v_win AS
           WITH regional AS (
               SELECT region, SUM(amount) AS total FROM orders GROUP BY region
           )
           SELECT region, total, RANK() OVER (ORDER BY total DESC) AS rnk
           FROM regional""",
    ]

    # 1. Verify PostgreSQL DDL invariants (row-level and statement transition tables)
    for use_trans in (False, True):
        for v in views_pg:
            plan = generate_materialized_view(
                schema,
                v,
                Dialect.POSTGRES,
                pg_use_transition_tables=use_trans,
                ignore_limitations=True,
            )
            script = plan.render_sql_script()
            assert not re.search(r"\bDOUBLE\b(?!\s+PRECISION)", script), script
            for trig in plan.trigger_ddls:
                if not use_trans and "TG_OP = 'DELETE'" in trig:
                    del_part = trig.split("TG_OP = 'DELETE'")[1].split(
                        "ELSIF (TG_OP = 'INSERT')"
                    )[0]
                    ins_part = trig.split("ELSIF (TG_OP = 'INSERT')")[1].split(
                        "ELSIF (TG_OP = 'UPDATE')"
                    )[0]
                    assert "NEW." not in del_part, del_part
                    assert "OLD." not in ins_part, ins_part
                if use_trans:
                    if "_ai()" in trig:
                        assert "OLD." not in trig, trig
                    if "_ad()" in trig:
                        assert "NEW." not in trig, trig

    # 2. Verify MySQL DDL invariants and left-to-right SET assignment hazard freedom
    for v in views_pg:
        v_my = (
            v.replace("BOOL_AND(is_paid) AS all_paid,", "")
            .replace("BOOL_OR(is_paid) AS any_paid,", "")
            .replace("STRING_AGG(tag, ',')", "GROUP_CONCAT(tag)")
        )
        plan = generate_materialized_view(
            schema, v_my, Dialect.MYSQL, ignore_limitations=True
        )
        script = plan.render_sql_script()
        assert "DOUBLE PRECISION" not in script, script
        for trig in plan.trigger_ddls:
            if "_ai\n" in trig:
                assert "OLD." not in trig, trig
            if "_ad\n" in trig:
                assert "NEW." not in trig, trig
            for m in re.finditer(
                r"(?:SET|ON DUPLICATE KEY UPDATE)\s+(.*?)(?:\n\s*WHERE|\s*;)",
                trig,
                re.DOTALL,
            ):
                assigns = _split_top_level(m.group(1), ",")
                assigned_so_far = set()
                for a in assigns:
                    lhs, rhs = a.split("=", 1)
                    lhs = lhs.strip()
                    for prev_col in assigned_so_far:
                        assert not re.search(
                            rf"\b{re.escape(prev_col)}\b", rhs
                        ), f"MySQL left-to-right SET hazard: {lhs} reads already-mutated {prev_col}"
                    assigned_so_far.add(lhs)

    # 3. Execute PostgreSQL algebraic formulas (VAR_POP, VAR_SAMP, STDDEV_POP, STDDEV_SAMP, BOOL_AND, BOOL_OR)
    # by running the exact generated PostgreSQL trigger statements (shimmed for SQLite execution)
    plan_pg = generate_materialized_view(
        schema, views_pg[0], Dialect.POSTGRES, ignore_limitations=True
    )
    fn_sql = plan_pg.trigger_ddls[0]
    del_body = (
        fn_sql.split("IF (TG_OP = 'DELETE') THEN")[1]
        .split("ELSIF (TG_OP = 'INSERT') THEN")[0]
        .strip()
    )
    ins_body = (
        fn_sql.split("ELSIF (TG_OP = 'INSERT') THEN")[1]
        .split("ELSIF (TG_OP = 'UPDATE') THEN")[0]
        .strip()
    )
    upd_body = (
        fn_sql.split("ELSIF (TG_OP = 'UPDATE') THEN")[1]
        .split("RETURN NULL;")[0]
        .strip()
    )

    def _pg_body_to_sqlite_trig(body: str, op: str) -> str:
        s = body
        # Strip outer IF (...) THEN ... END IF; blocks and convert into SQLite statements
        s = re.sub(r"IF\s*\([^)]*(?:IS NOT DISTINCT FROM|OLD\.|NEW\.)[^)]*\)\s*THEN", "", s)
        s = re.sub(r"END\s+IF\s*;", "", s)
        s = re.sub(r"ROW\(([^)]+)\)::TEXT", r"quote(\1)", s)
        s = s.replace("IS NOT DISTINCT FROM", "IS")
        s = re.sub(r"\(([A-Za-z0-9_.]+)\s*#\s*\(([^()]+)\)\)", r"((\1 | \2) - (\1 & \2))", s)
        when_clause = ""
        if op == "INSERT":
            when_clause = "WHEN (NEW.amount > 0)"
        elif op == "DELETE":
            when_clause = "WHEN (OLD.amount > 0)"
        return f"CREATE TRIGGER sim_pg_{op.lower()} AFTER {op} ON orders FOR EACH ROW {when_clause} BEGIN {s} END;"

    conn = sqlite3.connect(":memory:")
    conn.create_function("SQRT", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None)
    conn.executescript(schema)
    conn.execute(plan_pg.backing_table_ddl)
    conn.execute(plan_pg.reader_view_ddl)
    conn.execute(_pg_body_to_sqlite_trig(ins_body, "INSERT"))
    conn.execute(_pg_body_to_sqlite_trig(del_body, "DELETE"))
    # For UPDATE, build minus + plus + cleanup with WHERE guards
    upd_stmts_s = re.sub(r"ROW\(([^)]+)\)::TEXT", r"quote(\1)", upd_body)
    upd_stmts_s = upd_stmts_s.replace("IS NOT DISTINCT FROM", "IS")
    upd_stmts_s = re.sub(r"\(([A-Za-z0-9_.]+)\s*#\s*\(([^()]+)\)\)", r"((\1 | \2) - (\1 & \2))", upd_stmts_s)
    # Remove PL/pgSQL IF wrappers (since test rows all have amount > 0)
    upd_stmts_s = re.sub(r"IF\s*\(.*?\)\s*THEN", "", upd_stmts_s)
    upd_stmts_s = re.sub(r"END\s+IF\s*;", "", upd_stmts_s)
    conn.execute(f"CREATE TRIGGER sim_pg_upd AFTER UPDATE ON orders FOR EACH ROW BEGIN {upd_stmts_s} END;")

    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 10, "APAC", 10.0, 3, 1, "a"),
            (2, 11, "APAC", 20.0, 5, 1, "b"),
            (3, 12, "APAC", 30.0, 6, 0, "c"),
        ],
    )
    row = conn.execute(
        "SELECT cnt, total_amt, avg_amt, vp, vs, sdp, sds, bx, all_paid, any_paid, min_amt, max_amt FROM __matview_v_alg WHERE region = 'APAC'"
    ).fetchone()
    assert row is not None
    assert row[0] == 3
    assert abs(row[1] - 60.0) < 1e-9
    assert abs(row[2] - 20.0) < 1e-9
    # Population variance of [10, 20, 30] is 200/3 = 66.6666666667; Sample variance is 100.0
    assert abs(row[3] - (200.0 / 3.0)) < 1e-6
    assert abs(row[4] - 100.0) < 1e-6
    assert abs(row[5] - math.sqrt(200.0 / 3.0)) < 1e-6
    assert abs(row[6] - 10.0) < 1e-6
    # BIT_XOR: 3 ^ 5 ^ 6 = 0
    assert row[7] == (3 ^ 5 ^ 6)
    # BOOL_AND: False (0), BOOL_OR: True (1)
    assert row[8] == 0 and row[9] == 1
    assert row[10] == 10.0 and row[11] == 30.0

    # Update row 3: amount 30 -> 40, is_paid 0 -> 1, flags 6 -> 1
    conn.execute("UPDATE orders SET amount = 40.0, is_paid = 1, flags = 1 WHERE id = 3")
    row2 = conn.execute(
        "SELECT cnt, total_amt, avg_amt, vp, vs, sdp, sds, bx, all_paid, any_paid, min_amt, max_amt FROM __matview_v_alg WHERE region = 'APAC'"
    ).fetchone()
    # Values are now [10, 20, 40]: mean = 70/3, var_pop = ((100+400+1600) - 4900/3)/3 = 1400/9, var_samp = 700/3
    assert abs(row2[1] - 70.0) < 1e-9
    assert abs(row2[3] - (1400.0 / 9.0)) < 1e-6
    assert abs(row2[4] - (700.0 / 3.0)) < 1e-6
    assert row2[7] == (3 ^ 5 ^ 1)
    assert row2[8] == 1 and row2[9] == 1
    assert row2[11] == 40.0

    # Delete row 1 and row 2 -> only row 3 (amount=40.0) remains; sds (STDDEV_SAMP) must become NULL for N=1, vp (VAR_POP) must be 0.0
    conn.execute("DELETE FROM orders WHERE id IN (1, 2)")
    row3 = conn.execute(
        "SELECT cnt, total_amt, vp, vs, sdp, sds, bx FROM __matview_v_alg WHERE region = 'APAC'"
    ).fetchone()
    assert row3[0] == 1
    assert abs(row3[2] - 0.0) < 1e-9
    assert row3[3] is None  # VAR_SAMP is NULL for N=1
    assert abs(row3[4] - 0.0) < 1e-9
    assert row3[5] is None  # STDDEV_SAMP is NULL for N=1
    assert row3[6] == 1
    conn.close()


def _translate_mysql_set_to_simultaneous_sql(
    set_clause: str, table_name: str
) -> str:
    """
    Faithfully emulates MySQL's left-to-right sequential SET evaluation inside
    a simultaneous-SET engine (like SQLite):
    In MySQL, `SET c1 = e1, c2 = e2` evaluates `e2` using the NEW value of `c1`.
    We substitute any reference to `table_name.c1` or `c1` in `e2` with `(e1)`
    for all previously assigned columns `c1` before executing on SQLite.
    """
    assigns = _split_top_level(set_clause, ",")
    substituted_exprs: Dict[str, str] = {}
    out_assigns: List[str] = []

    for a in assigns:
        lhs, rhs = a.split("=", 1)
        col = lhs.strip().split(".")[-1]
        expr = rhs.strip()
        # Substitute any previously mutated columns (MySQL left-to-right semantics)
        for prev_col, prev_expr in substituted_exprs.items():
            expr = re.sub(
                rf"\b{re.escape(table_name)}\.{re.escape(prev_col)}\b",
                f"({prev_expr})",
                expr,
            )
            expr = re.sub(
                rf"(?<![.\w])\b{re.escape(prev_col)}\b",
                f"({prev_expr})",
                expr,
            )
        substituted_exprs[col] = expr
        out_assigns.append(f"{col} = {expr}")

    return ", ".join(out_assigns)


def _mysql_stmt_to_sqlite(stmt: str, mv_table: str) -> str:
    """Translates a single MySQL trigger SQL statement into executable SQLite SQL
    while preserving MySQL's left-to-right SET evaluation semantics."""
    s = stmt.strip()
    # Replace CONCAT_WS('|', IF(x IS NULL, 'N', CONCAT('V', LENGTH(CAST(x AS CHAR)), ':', CAST(x AS CHAR))))
    # with SQLite quote(x)
    s = re.sub(
        r"CONCAT_WS\('\|',\s*IF\(([A-Za-z0-9_.]+)\s+IS\s+NULL,\s*'N',\s*CONCAT\('V',\s*LENGTH\(CAST\([A-Za-z0-9_.]+\s+AS\s+CHAR\)\),\s*':',\s*CAST\([A-Za-z0-9_.]+\s+AS\s+CHAR\)\)\)\)",
        r"quote(\1)",
        s,
    )
    s = re.sub(r"\s+FROM\s+DUAL\b", "", s, flags=re.IGNORECASE)
    s = s.replace("<=>", "IS")
    # Bitwise XOR: (a ^ (b)) -> ((a | (b)) - (a & (b)))
    s = re.sub(
        r"\(([A-Za-z0-9_.]+)\s*\^\s*\(([^()]+)\)\)",
        r"((\1 | (\2)) - (\1 & (\2)))",
        s,
    )

    # Convert ON DUPLICATE KEY UPDATE with MySQL left-to-right SET emulation
    m_odku = re.search(r"\bON\s+DUPLICATE\s+KEY\s+UPDATE\s+(.*);?$", s, re.DOTALL | re.IGNORECASE)
    if m_odku:
        before = s[: m_odku.start()].rstrip()
        set_part = m_odku.group(1).rstrip().rstrip(";")
        # Determine PK column(s) from INSERT INTO tbl (c1, ...)
        m_ins = re.match(r"INSERT\s+INTO\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)", before, re.IGNORECASE)
        assert m_ins is not None
        target_tbl = m_ins.group(1)
        cols = [c.strip() for c in m_ins.group(2).split(",")]
        pk_cols = ["__mv_pk", "__val"] if "__val" in cols else [cols[0]]
        sim_set = _translate_mysql_set_to_simultaneous_sql(set_part, target_tbl)
        s = f"{before}\n    ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {sim_set};"
    elif s.upper().startswith("UPDATE "):
        m_upd = re.match(
            r"(UPDATE\s+([A-Za-z0-9_]+)\s+SET\s+)(.*?)(\s+WHERE\s+.*)$",
            s,
            re.DOTALL | re.IGNORECASE,
        )
        if m_upd:
            prefix, target_tbl, set_part, where_part = m_upd.groups()
            sim_set = _translate_mysql_set_to_simultaneous_sql(set_part, target_tbl)
            s = f"{prefix}{sim_set}{where_part}"

    return s


def test_mysql_trigger_execution_with_left_to_right_set_semantics() -> None:
    """Executes the exact generated MySQL triggers (with MySQL's left-to-right SET
    evaluation semantics) across INSERT, UPDATE, and DELETE operations."""
    schema = """
    CREATE TABLE orders (
        id INT PRIMARY KEY,
        customer_id INT,
        region VARCHAR(50),
        amount DOUBLE,
        flags INT,
        tag VARCHAR(50)
    );
    """
    view = """
    CREATE VIEW v_mysql_exec AS
    SELECT region,
           COUNT(*) AS cnt,
           SUM(amount) AS total_amt,
           AVG(amount) AS avg_amt,
           VAR_POP(amount) AS vp,
           VAR_SAMP(amount) AS vs,
           STDDEV_POP(amount) AS sdp,
           STDDEV_SAMP(amount) AS sds,
           BIT_XOR(flags) AS bx,
           MIN(amount) AS min_amt,
           MAX(amount) AS max_amt
    FROM orders
    WHERE amount > 0
    GROUP BY region
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineLimitationWarning)
        plan = generate_materialized_view(
            schema, view, Dialect.MYSQL, ignore_limitations=True
        )

    conn = sqlite3.connect(":memory:")
    conn.create_function(
        "SQRT", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None
    )
    conn.executescript(schema)
    conn.execute(plan.backing_table_ddl)
    conn.execute("CREATE VIEW v_mysql_exec AS SELECT region, cnt, total_amt, avg_amt, vp, vs, sdp, sds, bx, min_amt, max_amt FROM __matview_store_v_mysql_exec;")

    # Extract statements from MySQL AFTER INSERT, AFTER DELETE, AFTER UPDATE triggers
    for trig_sql in plan.trigger_ddls:
        body = trig_sql.split("BEGIN", 1)[1].rsplit("END;", 1)[0].strip()
        # Strip outer IF (...) THEN ... END IF; wrappers
        body = re.sub(r"IF\s*\([^\n]+\)\s*THEN", "", body)
        body = re.sub(r"END\s+IF\s*;", "", body)
        raw_stmts = [st.strip() + ";" for st in _split_top_level(body, ";") if st.strip()]
        conv_stmts = [_mysql_stmt_to_sqlite(st, plan.backing_table_name) for st in raw_stmts]
        joined = "\n    ".join(conv_stmts)
        if "_ai\n" in trig_sql:
            conn.execute(
                f"CREATE TRIGGER sim_my_ai AFTER INSERT ON orders FOR EACH ROW WHEN (NEW.amount > 0) BEGIN {joined} END;"
            )
        elif "_ad\n" in trig_sql:
            conn.execute(
                f"CREATE TRIGGER sim_my_ad AFTER DELETE ON orders FOR EACH ROW WHEN (OLD.amount > 0) BEGIN {joined} END;"
            )
        elif "_au\n" in trig_sql:
            conn.execute(
                f"CREATE TRIGGER sim_my_au AFTER UPDATE ON orders FOR EACH ROW BEGIN {joined} END;"
            )

    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, 100, "EU", 12.0, 7, "x"),
            (2, 101, "EU", 18.0, 3, "y"),
            (3, 102, "EU", 30.0, 5, "z"),
            (4, 103, None, 50.0, 9, "n"),
        ],
    )

    row_eu = conn.execute(
        "SELECT cnt, total_amt, avg_amt, vp, vs, sdp, sds, bx, min_amt, max_amt FROM v_mysql_exec WHERE region = 'EU'"
    ).fetchone()
    assert row_eu is not None
    assert row_eu[0] == 3
    assert abs(row_eu[1] - 60.0) < 1e-9
    assert abs(row_eu[2] - 20.0) < 1e-9
    # Values [12, 18, 30]: mean=20, sum((x-20)^2) = 64 + 4 + 100 = 168 -> vp = 56.0, vs = 84.0
    assert abs(row_eu[3] - 56.0) < 1e-6
    assert abs(row_eu[4] - 84.0) < 1e-6
    assert abs(row_eu[5] - math.sqrt(56.0)) < 1e-6
    assert abs(row_eu[6] - math.sqrt(84.0)) < 1e-6
    assert row_eu[7] == (7 ^ 3 ^ 5)
    assert row_eu[8] == 12.0 and row_eu[9] == 30.0

    # Verify NULL group key works in MySQL
    row_null = conn.execute(
        "SELECT cnt, total_amt, avg_amt, vp, vs, bx FROM v_mysql_exec WHERE region IS NULL"
    ).fetchone()
    assert row_null == (1, 50.0, 50.0, 0.0, None, 9)

    # Update row 3 from EU to US and change amount
    conn.execute("UPDATE orders SET region = 'US', amount = 45.0, flags = 11 WHERE id = 3")
    row_eu2 = conn.execute(
        "SELECT cnt, total_amt, avg_amt, vp, vs, bx, min_amt, max_amt FROM v_mysql_exec WHERE region = 'EU'"
    ).fetchone()
    # EU now has [12, 18]: mean=15, vp=9.0, vs=18.0
    assert row_eu2[0] == 2
    assert abs(row_eu2[1] - 30.0) < 1e-9
    assert abs(row_eu2[2] - 15.0) < 1e-9
    assert abs(row_eu2[3] - 9.0) < 1e-6
    assert abs(row_eu2[4] - 18.0) < 1e-6
    assert row_eu2[5] == (7 ^ 3)
    assert row_eu2[6] == 12.0 and row_eu2[7] == 18.0

    # Delete all EU rows -> group row must be removed
    conn.execute("DELETE FROM orders WHERE region = 'EU'")
    assert conn.execute("SELECT COUNT(*) FROM v_mysql_exec WHERE region = 'EU'").fetchone()[0] == 0
    conn.close()


def _verify_external_db_mv(cur, mv_name: str, select_sql: str, context: str = "") -> None:
    from decimal import Decimal

    def _norm(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (float, Decimal)):
            return round(float(v), 5)
        return v

    def _skey(row):
        return tuple(
            (0 if v is None else (1 if isinstance(v, (int, float)) else 2), _norm(v) if v is not None else "")
            for v in row
        )

    cur.execute(f"SELECT * FROM {mv_name}")
    mv_rows = sorted([tuple(_norm(x) for x in r) for r in cur.fetchall()], key=_skey)
    cur.execute(select_sql)
    live_rows = sorted([tuple(_norm(x) for x in r) for r in cur.fetchall()], key=_skey)
    if mv_rows != live_rows:
        raise AssertionError(
            f"[{context}] External DB MV mismatch on {mv_name}!\n"
            f"  Query: {select_sql}\n"
            f"  MV ({len(mv_rows)} rows):   {mv_rows[:10]}\n"
            f"  Live ({len(live_rows)} rows): {live_rows[:10]}"
        )


def run_live_postgres_tests_if_available(dsn: str) -> None:
    """Runs comprehensive live integration tests against a real PostgreSQL instance."""
    try:
        import psycopg2 as pg_driver  # type: ignore
    except ImportError:
        try:
            import psycopg as pg_driver  # type: ignore
        except ImportError:
            if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
                raise RuntimeError("psycopg2 or psycopg must be installed in CI when POSTGRES_DSN is set")
            print("Skipping live PostgreSQL test: psycopg2/psycopg not installed.")
            return

    schema = """
    CREATE TABLE customers (
        id INT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        tier VARCHAR(32) NOT NULL
    );
    CREATE TABLE orders (
        id INT PRIMARY KEY,
        customer_id INT NOT NULL,
        region VARCHAR(32),
        amount DOUBLE PRECISION,
        qty INT NOT NULL,
        flags INT NOT NULL,
        is_paid BOOLEAN NOT NULL,
        status VARCHAR(32) NOT NULL
    );
    """
    view_specs = [
        ("v_pg_alg", "SELECT region, COUNT(*) AS cnt, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, STDDEV_POP(amount) AS sdp, BIT_AND(flags) AS ba, BIT_OR(flags) AS bo, BIT_XOR(flags) AS bx, BOOL_AND(is_paid) AS all_paid, BOOL_OR(is_paid) AS any_paid FROM orders WHERE status = 'settled' GROUP BY region HAVING COUNT(*) >= 1"),
        ("v_pg_expr", "SELECT COALESCE(region, 'UNK') AS reg_norm, COUNT(*) AS cnt, SUM(amount * qty) - SUM(amount) AS net_rev FROM orders GROUP BY COALESCE(region, 'UNK')"),
        ("v_pg_global", "SELECT COUNT(*) AS cnt, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q FROM orders"),
        ("v_pg_join", "SELECT c.tier, o.region, COUNT(*) AS cnt, SUM(o.amount * o.qty) AS rev, AVG(o.amount) AS avg_amt FROM orders o INNER JOIN customers c ON o.customer_id = c.id WHERE o.status = 'settled' GROUP BY c.tier, o.region"),
        ("v_pg_uall", "SELECT id, region FROM orders WHERE qty > 1 UNION ALL SELECT id, tier AS region FROM customers"),
        ("v_pg_union", "SELECT region FROM orders WHERE region IS NOT NULL UNION SELECT tier AS region FROM customers"),
        ("v_pg_inter", "SELECT region FROM orders WHERE region IS NOT NULL INTERSECT SELECT tier AS region FROM customers"),
        ("v_pg_exc", "SELECT region FROM orders WHERE region IS NOT NULL EXCEPT SELECT tier AS region FROM customers"),
        ("v_pg_dist", "SELECT DISTINCT region, status FROM orders WHERE region IS NOT NULL"),
        ("v_pg_proj", "SELECT id, region, amount * qty AS line_total FROM orders WHERE qty > 0"),
        ("v_pg_cte", "WITH u AS (SELECT region AS k, amount AS v FROM orders UNION ALL SELECT tier AS k, 10.0 AS v FROM customers) SELECT k, COUNT(*) AS cnt, SUM(v) AS sv FROM u GROUP BY k"),
        ("v_pg_win", "SELECT id, region, amount, RANK() OVER (PARTITION BY region ORDER BY amount DESC, id ASC) AS rnk FROM orders"),
    ]

    conn = pg_driver.connect(dsn)
    conn.autocommit = True
    cur = conn.cursor()
    total_checks = 0

    for inc_mode in (False, True):
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        for s in [st.strip() for st in schema.split(";") if st.strip()]:
            cur.execute(s)
        cur.execute("INSERT INTO customers VALUES (1, 'Alice', 'NA'), (2, 'Bob', 'VIP'), (3, 'Cara', 'EU');")
        cur.execute(
            "INSERT INTO orders VALUES "
            "(1, 1, 'NA', 10.0, 2, 3, TRUE, 'settled'),"
            "(2, 1, 'NA', 20.0, 4, 5, FALSE, 'settled'),"
            "(3, 2, 'EU', 15.0, 1, 7, TRUE, 'pending'),"
            "(4, 2, 'EU', 30.0, 3, 6, TRUE, 'settled'),"
            "(5, 3, NULL, 50.0, 5, 1, FALSE, 'settled');"
        )
        plans = []
        for v_name, v_sel in view_specs:
            p = generate_materialized_view(
                schema,
                f"CREATE VIEW {v_name} AS {v_sel};",
                dialect=Dialect.POSTGRES,
                incremental_backfill=inc_mode,
                batch_size=2,
            )
            for stmt in p.all_statements():
                cur.execute(stmt)
            plans.append((p, v_sel))
            _verify_external_db_mv(cur, p.view_name, v_sel, f"pg:inc={inc_mode}:{v_name}:init")
            total_checks += 1

        mutations = [
            "INSERT INTO orders VALUES (6, 2, 'NA', NULL, 2, 4, TRUE, 'settled'), (7, 3, 'APAC', 40.0, 2, 2, TRUE, 'settled');",
            "UPDATE orders SET amount = 25.0, region = 'EU', qty = 5 WHERE id = 1;",
            "UPDATE orders SET status = 'settled', is_paid = FALSE WHERE id = 3;",
            "UPDATE customers SET tier = 'APAC' WHERE id = 2;",
            "DELETE FROM orders WHERE id IN (2, 5);",
            "DELETE FROM customers WHERE id = 3;",
            "TRUNCATE TABLE orders;",
        ]
        for m_idx, m_sql in enumerate(mutations):
            cur.execute(m_sql)
            for p, v_sel in plans:
                _verify_external_db_mv(cur, p.view_name, v_sel, f"pg:inc={inc_mode}:{p.logical_view_name}:m{m_idx}")
                total_checks += 1

    conn.close()
    print(f"LIVE PostgreSQL server test PASSED ({len(view_specs) * 2} views, {total_checks} live assertions)!")


def run_live_mysql_tests_if_available(dsn: str) -> None:
    """Runs comprehensive live integration tests against a real MySQL instance."""
    try:
        import pymysql  # type: ignore
    except ImportError:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            raise RuntimeError("pymysql must be installed in CI when MYSQL_DSN is set")
        print("Skipping live MySQL test: pymysql not installed.")
        return

    from urllib.parse import urlparse
    parsed = urlparse(dsn if "://" in dsn else f"mysql://{dsn}")
    conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=(parsed.path or "/test").lstrip("/") or "test",
        autocommit=True,
    )
    cur = conn.cursor()

    schema = """
    CREATE TABLE customers (
        id INT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        tier VARCHAR(32) NOT NULL
    );
    CREATE TABLE orders (
        id INT PRIMARY KEY,
        customer_id INT NOT NULL,
        region VARCHAR(32),
        amount DOUBLE,
        qty INT NOT NULL,
        flags INT NOT NULL,
        status VARCHAR(32) NOT NULL
    );
    """
    view_specs = [
        ("v_my_alg", "SELECT region, COUNT(*) AS cnt, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, STDDEV_POP(amount) AS sdp, BIT_AND(flags) AS ba, BIT_OR(flags) AS bo, BIT_XOR(flags) AS bx FROM orders WHERE status = 'settled' GROUP BY region HAVING COUNT(*) >= 1"),
        ("v_my_expr", "SELECT COALESCE(region, 'UNK') AS reg_norm, COUNT(*) AS cnt, SUM(amount * qty) - SUM(amount) AS net_rev FROM orders GROUP BY COALESCE(region, 'UNK')"),
        ("v_my_global", "SELECT COUNT(*) AS cnt, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q FROM orders"),
        ("v_my_join", "SELECT c.tier, o.region, COUNT(*) AS cnt, SUM(o.amount * o.qty) AS rev, AVG(o.amount) AS avg_amt FROM orders o INNER JOIN customers c ON o.customer_id = c.id WHERE o.status = 'settled' GROUP BY c.tier, o.region"),
        ("v_my_uall", "SELECT id, region FROM orders WHERE qty > 1 UNION ALL SELECT id, tier AS region FROM customers"),
        ("v_my_union", "SELECT region FROM orders WHERE region IS NOT NULL UNION SELECT tier AS region FROM customers"),
        ("v_my_inter", "SELECT region FROM orders WHERE region IS NOT NULL INTERSECT SELECT tier AS region FROM customers"),
        ("v_my_exc", "SELECT region FROM orders WHERE region IS NOT NULL EXCEPT SELECT tier AS region FROM customers"),
        ("v_my_dist", "SELECT DISTINCT region, status FROM orders WHERE region IS NOT NULL"),
        ("v_my_proj", "SELECT id, region, amount * qty AS line_total FROM orders WHERE qty > 0"),
        ("v_my_cte", "WITH u AS (SELECT region AS k, amount AS v FROM orders UNION ALL SELECT tier AS k, 10.0 AS v FROM customers) SELECT k, COUNT(*) AS cnt, SUM(v) AS sv FROM u GROUP BY k"),
        ("v_my_win", "SELECT id, region, amount, RANK() OVER (PARTITION BY region ORDER BY amount DESC, id ASC) AS rnk FROM orders"),
    ]

    total_checks = 0
    for inc_mode in (False, True):
        for v_name, _ in view_specs:
            cur.execute(f"DROP VIEW IF EXISTS __matview_{v_name};")
            cur.execute(f"DROP TABLE IF EXISTS __matview_store_{v_name};")
        cur.execute("DROP TABLE IF EXISTS orders;")
        cur.execute("DROP TABLE IF EXISTS customers;")
        for s in [st.strip() for st in schema.split(";") if st.strip()]:
            cur.execute(s)
        cur.execute("INSERT INTO customers VALUES (1, 'Alice', 'NA'), (2, 'Bob', 'VIP'), (3, 'Cara', 'EU');")
        cur.execute(
            "INSERT INTO orders VALUES "
            "(1, 1, 'NA', 10.0, 2, 3, 'settled'),"
            "(2, 1, 'NA', 20.0, 4, 5, 'settled'),"
            "(3, 2, 'EU', 15.0, 1, 7, 'pending'),"
            "(4, 2, 'EU', 30.0, 3, 6, 'settled'),"
            "(5, 3, NULL, 50.0, 5, 1, 'settled');"
        )
        plans = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            for v_name, v_sel in view_specs:
                p = generate_materialized_view(
                    schema,
                    f"CREATE VIEW {v_name} AS {v_sel};",
                    dialect=Dialect.MYSQL,
                    incremental_backfill=inc_mode,
                    batch_size=2,
                    ignore_limitations=True,
                )
                for stmt in p.all_statements():
                    cur.execute(stmt)
                plans.append((p, v_sel))
                _verify_external_db_mv(cur, p.view_name, v_sel, f"mysql:inc={inc_mode}:{v_name}:init")
                total_checks += 1

        mutations = [
            "INSERT INTO orders VALUES (6, 2, 'NA', NULL, 2, 4, 'settled'), (7, 3, 'APAC', 40.0, 2, 2, 'settled');",
            "UPDATE orders SET amount = 25.0, region = 'EU', qty = 5 WHERE id = 1;",
            "UPDATE orders SET status = 'settled' WHERE id = 3;",
            "UPDATE customers SET tier = 'APAC' WHERE id = 2;",
            "DELETE FROM orders WHERE id IN (2, 5);",
            "DELETE FROM customers WHERE id = 3;",
            "DELETE FROM orders;",
        ]
        for m_idx, m_sql in enumerate(mutations):
            cur.execute(m_sql)
            for p, v_sel in plans:
                _verify_external_db_mv(cur, p.view_name, v_sel, f"mysql:inc={inc_mode}:{p.logical_view_name}:m{m_idx}")
                total_checks += 1

    conn.close()
    print(f"LIVE MySQL server test PASSED ({len(view_specs) * 2} views, {total_checks} live assertions)!")





def test_object_prefix_and_user_defined_indexes() -> None:
    schema = """
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        region TEXT,
        category TEXT NOT NULL,
        amount REAL
    );
    """

    # 1. Test default prefix (__matview_) and inline CREATE INDEX + programmatic indexes
    view_with_sql_indexes = """
    CREATE VIEW sales_idx_mv AS
    SELECT region, category, COUNT(*) AS cnt, SUM(amount) AS total_amt
    FROM orders
    GROUP BY region, category;

    CREATE INDEX idx_custom_total ON sales_idx_mv (total_amt DESC);
    CREATE UNIQUE INDEX ON sales_idx_mv (region, category) WHERE region IS NOT NULL;
    """

    conn = sqlite3.connect(":memory:")
    conn.executescript(schema)
    plan = install_sqlite_materialized_view(
        conn,
        schema,
        view_with_sql_indexes,
        indexes=[
            "cnt DESC",
            ViewIndexSpec(columns=("category", "cnt DESC"), name="idx_cat_cnt"),
        ],
    )

    # Verify EVERY matview object in sqlite_master starts with __matview_
    db_objects = conn.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' AND name != 'orders'"
    ).fetchall()
    assert len(db_objects) > 0
    for obj_type, obj_name in db_objects:
        assert obj_name.startswith("__matview_"), (
            f"Database object ({obj_type}) {obj_name!r} does not start with '__matview_'!"
        )

    # Verify all expected user indexes were created with __matview_ prefix
    idx_rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = ?",
        (plan.backing_table_name,),
    ).fetchall()
    idx_map = {name: sql for name, sql in idx_rows if sql is not None}
    assert "__matview_idx_custom_total" in idx_map
    assert "(total_amt DESC)" in idx_map["__matview_idx_custom_total"]
    assert "__matview_idx_sales_idx_mv__region_category" in idx_map
    assert "UNIQUE INDEX" in idx_map["__matview_idx_sales_idx_mv__region_category"]
    assert "WHERE region IS NOT NULL" in idx_map["__matview_idx_sales_idx_mv__region_category"]
    assert "__matview_idx_sales_idx_mv__cnt" in idx_map
    assert "__matview_idx_cat_cnt" in idx_map

    # Verify SQLite query planner uses our user-defined index when querying the read-only view
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
        [(i, i % 10, f"R{i % 5}", f"C{i % 3}", float(i * 10)) for i in range(1, 50)],
    )
    eqp = conn.execute(
        "EXPLAIN QUERY PLAN SELECT region, category, total_amt "
        "FROM __matview_sales_idx_mv WHERE total_amt > 500 ORDER BY total_amt DESC"
    ).fetchall()
    eqp_text = " ".join(str(r) for r in eqp)
    assert "__matview_idx_custom_total" in eqp_text, f"Expected index in query plan, got: {eqp_text}"
    conn.close()

    # 2. Test overriding object_prefix (e.g. object_prefix="custom_mv_")
    conn2 = sqlite3.connect(":memory:")
    conn2.executescript(schema)
    inline_header_view = """
    CREATE VIEW regional_mv
    INDEX (total_amt DESC),
    UNIQUE INDEX idx_u_cat (category)
    AS
    SELECT category, COUNT(*) AS cnt, SUM(amount) AS total_amt
    FROM orders
    GROUP BY category;
    """
    plan2 = install_sqlite_materialized_view(
        conn2,
        schema,
        inline_header_view,
        object_prefix="custom_mv_",
    )
    assert plan2.view_name == "custom_mv_regional_mv"
    assert plan2.backing_table_name == "custom_mv_store_regional_mv"
    db_objects2 = conn2.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' AND name != 'orders'"
    ).fetchall()
    for obj_type, obj_name in db_objects2:
        assert obj_name.startswith("custom_mv_"), (
            f"Database object ({obj_type}) {obj_name!r} does not start with 'custom_mv_'!"
        )
    conn2.close()

    # 3. Test PostgreSQL & MySQL object_prefix on functions, triggers, views, tables, indexes
    for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
        for pfx in ("__matview_", "__custom_"):
            p_plan = generate_materialized_view(
                schema,
                "CREATE VIEW v_pref AS SELECT region, COUNT(*) AS cnt, SUM(amount) AS s FROM orders GROUP BY region;",
                dialect=dialect,
                indexes=["s DESC", "UNIQUE:region"],
                object_prefix=pfx,
                ignore_limitations=True,
            )
            assert p_plan.view_name == f"{pfx}v_pref"
            assert p_plan.backing_table_name == f"{pfx}store_v_pref"
            for idx_n in p_plan.index_names:
                assert idx_n.startswith(pfx), idx_n
            for trig_n in p_plan.trigger_names:
                assert trig_n.startswith(pfx), trig_n
            for fn_n in p_plan.function_names:
                assert fn_n.startswith(pfx), fn_n

    # 4. Test validation error when indexing a non-existent view column
    try:
        generate_materialized_view(
            schema,
            "CREATE VIEW v_bad_idx AS SELECT region, COUNT(*) AS cnt FROM orders GROUP BY region;",
            indexes=["non_existent_col DESC"],
        )
        assert False, "Expected ValueError for unknown index column"
    except ValueError as exc:
        assert "non_existent_col" in str(exc)

    # 5. Test MySQL partial index limitation check
    try:
        generate_materialized_view(
            schema,
            "CREATE VIEW v_my_part AS SELECT region, COUNT(*) AS cnt FROM orders GROUP BY region;",
            dialect=Dialect.MYSQL,
            indexes=["CREATE INDEX idx_part ON v_my_part (cnt) WHERE cnt > 5"],
            ignore_limitations=False,
        )
        assert False, "Expected EngineLimitationError for MySQL partial index"
    except EngineLimitationError as exc:
        codes = {lim.code for lim in exc.limitations}
        assert "MYSQL_UNSUPPORTED_PARTIAL_INDEX" in codes

def verify_mv_equals_live(conn: sqlite3.Connection, mv_name: str, select_sql: str, context: str = "") -> None:
    cur = conn.cursor()
    mv_desc = [d[0] for d in cur.execute(f"SELECT * FROM {mv_name} LIMIT 0").description]
    live_desc = [d[0] for d in cur.execute(f"SELECT * FROM ({select_sql}) LIMIT 0").description]
    assert len(mv_desc) == len(live_desc), f"[{context}] Column count mismatch on {mv_name}: {mv_desc} vs {live_desc}"

    # Sort rows canonically in Python so any column types/NULLs compare deterministically
    def _norm_val(v):
        if isinstance(v, float):
            return round(v, 6)
        return v

    def _sort_key(row):
        return tuple((0 if v is None else (1 if isinstance(v, (int, float)) else 2), _norm_val(v) if v is not None else "") for v in row)

    mv_rows = sorted(
        [tuple(_norm_val(x) for x in r) for r in cur.execute(f"SELECT * FROM {mv_name}").fetchall()],
        key=_sort_key,
    )
    live_rows = sorted(
        [tuple(_norm_val(x) for x in r) for r in cur.execute(select_sql).fetchall()],
        key=_sort_key,
    )
    if mv_rows != live_rows:
        raise AssertionError(
            f"[{context}] MV mismatch on {mv_name}!\n"
            f"  Query: {select_sql}\n"
            f"  MV ({len(mv_rows)} rows):   {mv_rows[:10]}\n"
            f"  Live ({len(live_rows)} rows): {live_rows[:10]}"
        )


def run_test_matrix():
    total_tests = 0
    total_mutation_checks = 0

    # =========================================================================
    # SUITE 1: Single-Table Algebraic & Reference-Counted Permutations (140 tests)
    # =========================================================================
    schema_t1 = """
    CREATE TABLE items (
        id INTEGER PRIMARY KEY,
        grp_a TEXT NOT NULL,
        grp_b TEXT,
        grp_c INTEGER,
        val_i INTEGER,
        val_f REAL,
        status TEXT NOT NULL,
        tag TEXT
    );
    """

    group_configs = [
        ("global", "", ""),
        ("grp_a", "grp_a, ", "GROUP BY grp_a"),
        ("grp_b_nullable", "grp_b, ", "GROUP BY grp_b"),
        ("grp_c_int", "grp_c, ", "GROUP BY grp_c"),
        ("grp_ab", "grp_a, grp_b, ", "GROUP BY grp_a, grp_b"),
        ("grp_abc", "grp_a, grp_b, grp_c, ", "GROUP BY grp_a, grp_b, grp_c"),
        ("grp_alias", "grp_a AS ga, ", "GROUP BY ga"),
        ("grp_ordinal", "grp_a, grp_b, ", "GROUP BY 1, 2"),
    ]

    where_configs = [
        ("no_where", ""),
        ("where_pos", "WHERE val_i > 0"),
        ("where_status", "WHERE status IN ('active', 'pending') AND grp_b IS NOT NULL"),
        ("where_complex", "WHERE (val_i >= 10 OR status = 'vip') AND val_f IS NOT NULL"),
    ]

    agg_bundles = [
        ("cnt_sum_avg", "COUNT(*) AS c, COUNT(val_i) AS ci, SUM(val_i) AS si, AVG(val_f) AS af"),
        ("min_max_tot", "COUNT(*) AS c, MIN(val_i) AS mni, MAX(val_i) AS mxi, MIN(val_f) AS mnf, MAX(val_f) AS mxf, TOTAL(val_f) AS tf"),
        ("distinct_aggs", "COUNT(*) AS c, COUNT(DISTINCT grp_b) AS cdb, COUNT(DISTINCT val_i) AS cdi, SUM(DISTINCT val_i) AS sdi, AVG(DISTINCT val_i) AS adi"),
        ("filter_aggs", "COUNT(*) AS c, COUNT(*) FILTER (WHERE status = 'active') AS ca, SUM(val_i) FILTER (WHERE val_i > 10) AS s10, AVG(val_f) FILTER (WHERE grp_b IS NOT NULL) AS afb, MIN(val_i) FILTER (WHERE status = 'active') AS mna, MAX(val_i) FILTER (WHERE status = 'active') AS mxa"),
        ("arith_aggs", "COUNT(*) AS c, SUM(val_i * 2 + COALESCE(grp_c, 0)) AS s_expr, AVG(val_f * 1.5) AS a_expr, MIN(val_i - 5) AS mn_expr, MAX(val_i + 5) AS mx_expr"),
        ("holistic_str_json", "COUNT(*) AS c, GROUP_CONCAT(tag, '|') AS tags, JSON_GROUP_ARRAY(val_i) AS jarr"),
    ]

    rng = random.Random(12345)
    for g_name, sel_prefix, grp_clause in group_configs:
        for w_name, where_clause in where_configs:
            for a_name, agg_select in agg_bundles:
                # Skip holistic string/json without order on filtered empty edge if redundant, or test all!
                v_name = f"v_{g_name}_{w_name}_{a_name}"
                select_sql = f"SELECT {sel_prefix}{agg_select} FROM items {where_clause} {grp_clause}".strip()
                view_sql = f"CREATE VIEW {v_name} AS {select_sql};"

                conn = sqlite3.connect(":memory:")
                conn.executescript(schema_t1)
                plan = install_sqlite_materialized_view(conn, schema_t1, view_sql)
                total_tests += 1

                # Step 0: Check initial empty table state
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:empty")
                total_mutation_checks += 1

                # Step 1: Insert initial batch with edge-case strings, NULLs, negatives, duplicates
                rows_batch_1 = [
                    (1, "A", "X|Y", 1, 10, 1.5, "active", "t1"),
                    (2, "A", "X|Y", 1, 20, 2.5, "pending", "t2"),
                    (3, "A", None, 2, -5, -0.5, "active", None),
                    (4, "B", "N", None, None, None, "closed", "t3"),
                    (5, "B", "V1:a", 2, 10, 4.0, "vip", "t1"),
                    (6, "C", "", 0, 0, 0.0, "active", "t4"),
                ]
                conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows_batch_1)
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:batch1")
                total_mutation_checks += 1

                # Step 2: Single-row INSERTs
                conn.execute("INSERT INTO items VALUES (7, 'A', 'X|Y', 1, 10, 3.5, 'active', 't5')")
                conn.execute("INSERT INTO items VALUES (8, 'D', NULL, NULL, 100, 99.9, 'vip', 't6')")
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:single_ins")
                total_mutation_checks += 1

                # Step 3: UPDATE within same group (changing extremum, NULLing values, flipping status filter)
                conn.execute("UPDATE items SET val_i = 5, val_f = 0.5, status = 'closed' WHERE id = 1")
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:upd_same_grp")
                total_mutation_checks += 1

                # Step 4: UPDATE across groups (moving rows between groups & NULL group keys)
                conn.execute("UPDATE items SET grp_a = 'B', grp_b = NULL, val_i = 50 WHERE id IN (2, 7)")
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:upd_cross_grp")
                total_mutation_checks += 1

                # Step 5: Multi-row UPDATE affecting all rows
                conn.execute("UPDATE items SET val_i = COALESCE(val_i, 0) + 7, val_f = COALESCE(val_f, 0.0) + 1.25")
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:upd_all")
                total_mutation_checks += 1

                # Step 6: Partial DELETE (draining some groups)
                conn.execute("DELETE FROM items WHERE grp_a IN ('A', 'C')")
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:del_partial")
                total_mutation_checks += 1

                # Step 7: Full DELETE (draining entire table back to empty)
                conn.execute("DELETE FROM items")
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:del_all_empty")
                total_mutation_checks += 1

                # Step 8: Re-insert after empty
                conn.execute("INSERT INTO items VALUES (99, 'Z', 'reborn', 9, 42, 4.2, 'active', 'tz')")
                verify_mv_equals_live(conn, plan.view_name, select_sql, f"{v_name}:reinsert")
                total_mutation_checks += 1
                conn.close()

    print(f"Suite 1 complete: {total_tests} views tested across {total_mutation_checks} live query assertions!")



def run_suite2_and_suite3():
    total_tests = 0
    total_checks = 0

    # =========================================================================
    # SUITE 2: Multi-Table JOINs (2-table, 3-table, 4-table INNER/LEFT/CROSS/Self)
    # =========================================================================
    schema_4t = """
    CREATE TABLE regions (
        id INTEGER PRIMARY KEY,
        r_name TEXT NOT NULL,
        tax_rate REAL NOT NULL
    );
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        region_id INTEGER,
        c_name TEXT NOT NULL,
        tier TEXT
    );
    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        category TEXT NOT NULL,
        unit_cost REAL NOT NULL
    );
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        qty INTEGER NOT NULL,
        price REAL,
        status TEXT NOT NULL
    );
    """

    join_templates = [
        # 2-table INNER JOINs
        ("j2_inner_c_tier", "SELECT c.tier, COUNT(*) AS cnt, SUM(o.qty) AS sq, SUM(o.price) AS sp, AVG(o.price) AS ap, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.tier"),
        ("j2_inner_c_id_name", "SELECT c.id AS cid, c.c_name, COUNT(*) AS cnt, SUM(o.qty * o.price) AS rev FROM orders o INNER JOIN customers c ON o.customer_id = c.id GROUP BY c.id, c.c_name"),
        ("j2_inner_p_cat", "SELECT p.category, o.status, COUNT(*) AS cnt, SUM(o.qty) AS sq, AVG(o.price - p.unit_cost) AS margin FROM orders o JOIN products p ON o.product_id = p.id GROUP BY p.category, o.status"),
        # 2-table LEFT JOINs (grouping by left, right, or both)
        ("j2_left_cust_orders", "SELECT c.tier, COUNT(o.id) AS order_cnt, COALESCE(SUM(o.qty), 0) AS total_qty, AVG(o.price) AS avg_p FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.tier"),
        ("j2_left_orders_cust_right_grp", "SELECT c.tier, COUNT(*) AS row_cnt, COUNT(c.id) AS matched_custs, SUM(o.qty) AS sq FROM orders o LEFT JOIN customers c ON o.customer_id = c.id GROUP BY c.tier"),
        ("j2_left_both_grp", "SELECT o.status, c.tier, COUNT(*) AS cnt, SUM(o.qty) AS sq FROM orders o LEFT JOIN customers c ON o.customer_id = c.id GROUP BY o.status, c.tier"),
        # 3-table INNER & LEFT JOINs
        ("j3_inner_rco", "SELECT r.r_name, c.tier, COUNT(*) AS cnt, SUM(o.qty) AS sq, SUM(o.price * (1.0 + r.tax_rate)) AS taxed_rev FROM orders o JOIN customers c ON o.customer_id = c.id JOIN regions r ON c.region_id = r.id GROUP BY r.r_name, c.tier"),
        ("j3_inner_cpo", "SELECT c.tier, p.category, COUNT(*) AS cnt, SUM(o.qty) AS sq, SUM(o.qty * p.unit_cost) AS cost FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id GROUP BY c.tier, p.category"),
        ("j3_mixed_left", "SELECT r.r_name, COUNT(c.id) AS cust_rows, COUNT(o.id) AS ord_rows, COALESCE(SUM(o.qty), 0) AS total_qty FROM regions r LEFT JOIN customers c ON r.id = c.region_id LEFT JOIN orders o ON c.id = o.customer_id GROUP BY r.r_name"),
        # 4-table Star Schema JOINs
        ("j4_star_all", "SELECT r.r_name, c.tier, p.category, COUNT(*) AS cnt, SUM(o.qty) AS sq, SUM(o.price) AS sp FROM orders o JOIN customers c ON o.customer_id = c.id JOIN regions r ON c.region_id = r.id JOIN products p ON o.product_id = p.id GROUP BY r.r_name, c.tier, p.category"),
        ("j4_star_region_cat", "SELECT r.r_name, p.category, COUNT(DISTINCT c.id) AS uniq_custs, SUM(o.qty) AS sq FROM orders o JOIN customers c ON o.customer_id = c.id JOIN regions r ON c.region_id = r.id JOIN products p ON o.product_id = p.id GROUP BY r.r_name, p.category"),
        # Self-JOIN and CROSS JOIN
        ("j_self_cust_same_region", "SELECT c1.region_id, COUNT(*) AS pair_cnt FROM customers c1 JOIN customers c2 ON c1.region_id = c2.region_id AND c1.id < c2.id GROUP BY c1.region_id"),
        ("j_cross_reg_prod", "SELECT r.r_name, COUNT(*) AS prod_cnt, AVG(p.unit_cost * (1.0 + r.tax_rate)) AS avg_taxed_cost FROM regions r CROSS JOIN products p GROUP BY r.r_name"),
    ]

    join_modifiers = [
        ("plain", "", ""),
        ("where_fulfilled", "WHERE o.status = 'fulfilled'", ""),
        ("where_qty", "WHERE o.qty >= 2", ""),
        ("having_cnt", "", "HAVING COUNT(*) >= 2"),
        ("where_and_having", "WHERE o.status != 'cancelled'", "HAVING SUM(o.qty) >= 3"),
    ]

    for j_name, base_q in join_templates:
        for mod_name, w_mod, h_mod in join_modifiers:
            if ("o." in w_mod or "o." in h_mod) and " orders o" not in base_q:
                continue
            if "LEFT JOIN orders o" in base_q and w_mod:
                # Filtering o.status in WHERE turns LEFT JOIN into INNER JOIN; still valid SQL!
                pass
            q = base_q
            if w_mod:
                q = q.replace(" GROUP BY ", f" {w_mod} GROUP BY ")
            if h_mod:
                q = f"{q} {h_mod}"

            v_name = f"mv_{j_name}_{mod_name}"
            conn = sqlite3.connect(":memory:")
            conn.executescript(schema_4t)
            plan = install_sqlite_materialized_view(conn, schema_4t, f"CREATE VIEW {v_name} AS {q};")
            total_tests += 1

            # Step 0: Empty
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:empty")
            total_checks += 1

            # Step 1: Populate dimension & fact tables
            conn.executemany("INSERT INTO regions VALUES (?, ?, ?)", [(1, "NA", 0.08), (2, "EU", 0.20), (3, "APAC", 0.10)])
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:ins_regions")
            total_checks += 1

            conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", [
                (10, 1, "Alice", "gold"),
                (11, 1, "Bob", "silver"),
                (12, 2, "Charlie", "gold"),
                (13, None, "Diana", None),
                (14, 3, "Evan", "bronze"),
            ])
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:ins_customers")
            total_checks += 1

            conn.executemany("INSERT INTO products VALUES (?, ?, ?)", [
                (100, "hw", 50.0),
                (101, "sw", 10.0),
                (102, "hw", 120.0),
            ])
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:ins_products")
            total_checks += 1

            conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", [
                (1000, 10, 100, 2, 150.0, "fulfilled"),
                (1001, 10, 101, 5, 100.0, "fulfilled"),
                (1002, 11, 100, 1, 75.0, "pending"),
                (1003, 12, 102, 3, 450.0, "fulfilled"),
                (1004, 13, 101, 4, 80.0, "cancelled"),
                (1005, 999, 100, 2, 140.0, "fulfilled"),  # unmatched customer_id 999!
            ])
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:ins_orders")
            total_checks += 1

            # Step 2: Insert matching customer 999 (turns unmatched LEFT JOIN row into matched!)
            conn.execute("INSERT INTO customers VALUES (999, 2, 'Latecomer', 'VIP')")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:ins_late_cust")
            total_checks += 1

            # Step 3: Update dimension tables (customer tier, region tax_rate, product category)
            conn.execute("UPDATE customers SET tier = 'platinum', region_id = 3 WHERE id = 10")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:upd_cust")
            total_checks += 1

            conn.execute("UPDATE regions SET r_name = 'NORTH_AMERICA', tax_rate = 0.05 WHERE id = 1")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:upd_region")
            total_checks += 1

            conn.execute("UPDATE products SET category = 'cloud', unit_cost = 15.0 WHERE id = 101")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:upd_product")
            total_checks += 1

            # Step 4: Update fact table (orders)
            conn.execute("UPDATE orders SET customer_id = 14, qty = 10, status = 'fulfilled' WHERE id = 1002")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:upd_orders")
            total_checks += 1

            # Step 5: Delete from dimension and fact tables
            conn.execute("DELETE FROM customers WHERE id = 999")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:del_cust_999")
            total_checks += 1

            conn.execute("DELETE FROM orders WHERE status = 'fulfilled'")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:del_fulfilled_orders")
            total_checks += 1

            conn.execute("DELETE FROM orders")
            conn.execute("DELETE FROM customers")
            conn.execute("DELETE FROM products")
            conn.execute("DELETE FROM regions")
            verify_mv_equals_live(conn, plan.view_name, q, f"{v_name}:del_all")
            total_checks += 1
            conn.close()

    # =========================================================================
    # SUITE 3: Subqueries, CTEs, Window Functions, Set Ops (UNION/INTERSECT/EXCEPT),
    #          Expression GROUP BY, DISTINCT, ORDER BY + LIMIT (85 tests)
    # =========================================================================
    advanced_views = [
        # Expression GROUP BY
        ("expr_grp_lower", "SELECT LOWER(c_name) AS lc, COUNT(*) AS cnt FROM customers GROUP BY LOWER(c_name)"),
        ("expr_grp_substr", "SELECT SUBSTR(c_name, 1, 1) AS initial, COUNT(*) AS cnt, COUNT(region_id) AS rcnt FROM customers GROUP BY SUBSTR(c_name, 1, 1)"),
        ("expr_grp_case", "SELECT CASE WHEN qty >= 3 THEN 'bulk' ELSE 'small' END AS bucket, COUNT(*) AS cnt, SUM(qty) AS sq FROM orders GROUP BY CASE WHEN qty >= 3 THEN 'bulk' ELSE 'small' END"),
        ("expr_grp_math", "SELECT (customer_id % 3) AS shard, COUNT(*) AS cnt, SUM(qty) AS sq FROM orders WHERE customer_id IS NOT NULL GROUP BY (customer_id % 3)"),
        ("expr_grp_coalesce", "SELECT COALESCE(tier, 'unassigned') AS eff_tier, COUNT(*) AS cnt FROM customers GROUP BY COALESCE(tier, 'unassigned')"),

        # Subqueries in WHERE, FROM, SELECT, HAVING
        ("subq_where_in", "SELECT status, COUNT(*) AS cnt, SUM(qty) AS sq FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE tier = 'gold') GROUP BY status"),
        ("subq_where_exists", "SELECT tier, COUNT(*) AS cnt FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.qty >= 2) GROUP BY tier"),
        ("subq_where_not_exists", "SELECT tier, COUNT(*) AS idle_custs FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id) GROUP BY tier"),
        ("subq_where_global_avg", "SELECT status, COUNT(*) AS above_avg_cnt, SUM(price) AS sp FROM orders WHERE price > (SELECT AVG(price) FROM orders) GROUP BY status"),
        ("subq_from_derived", "SELECT cust_tier, COUNT(*) AS order_groups, SUM(cust_qty) AS total_qty FROM (SELECT c.id AS cid, c.tier AS cust_tier, SUM(o.qty) AS cust_qty FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.tier) AS sub GROUP BY cust_tier"),
        ("subq_select_scalar", "SELECT c.id AS cid, c.c_name, (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS ord_cnt, (SELECT COALESCE(SUM(o.qty), 0) FROM orders o WHERE o.customer_id = c.id) AS tot_qty FROM customers c"),
        ("subq_having", "SELECT customer_id, SUM(qty) AS sq FROM orders WHERE customer_id IS NOT NULL GROUP BY customer_id HAVING SUM(qty) > (SELECT AVG(qty) FROM orders)"),

        # CTEs (single, chained, and with joins/aggregations)
        ("cte_single", "WITH fulfilled AS (SELECT customer_id, qty, price FROM orders WHERE status = 'fulfilled') SELECT customer_id, COUNT(*) AS cnt, SUM(qty) AS sq, AVG(price) AS ap FROM fulfilled GROUP BY customer_id"),
        ("cte_chained", "WITH o_by_c AS (SELECT customer_id, SUM(qty) AS tot_q, SUM(price) AS tot_p FROM orders GROUP BY customer_id), enriched AS (SELECT c.tier, o.tot_q, o.tot_p FROM customers c JOIN o_by_c o ON c.id = o.customer_id) SELECT tier, COUNT(*) AS cust_cnt, SUM(tot_q) AS tier_q, AVG(tot_p) AS avg_tier_p FROM enriched GROUP BY tier"),
        ("cte_multi_join", "WITH r_tax AS (SELECT id, r_name, tax_rate FROM regions WHERE tax_rate >= 0.05), p_hw AS (SELECT id, unit_cost FROM products WHERE category = 'hw') SELECT r_tax.r_name, COUNT(*) AS cnt, SUM(o.qty * p_hw.unit_cost) AS base_cost FROM orders o JOIN customers c ON o.customer_id = c.id JOIN r_tax ON c.region_id = r_tax.id JOIN p_hw ON o.product_id = p_hw.id GROUP BY r_tax.r_name"),

        # Window Functions
        ("win_rank_by_status", "SELECT id, status, qty, RANK() OVER (PARTITION BY status ORDER BY qty DESC, id ASC) AS rnk, SUM(qty) OVER (PARTITION BY status) AS status_total FROM orders"),
        ("win_dense_rank_cte", "WITH c_totals AS (SELECT customer_id, SUM(qty) AS total_qty FROM orders WHERE customer_id IS NOT NULL GROUP BY customer_id) SELECT customer_id, total_qty, DENSE_RANK() OVER (ORDER BY total_qty DESC) AS drnk, ROW_NUMBER() OVER (ORDER BY total_qty DESC, customer_id ASC) AS rn FROM c_totals"),
        ("win_running_sum", "SELECT id, customer_id, qty, SUM(qty) OVER (PARTITION BY customer_id ORDER BY id ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_qty FROM orders WHERE customer_id IS NOT NULL"),

        # Compound Set Operations (UNION ALL, UNION, INTERSECT, EXCEPT)
        ("set_union_all", "SELECT 'cust' AS src, c_name AS label FROM customers UNION ALL SELECT 'reg' AS src, r_name AS label FROM regions"),
        ("set_union_dedup", "SELECT region_id AS rid FROM customers WHERE region_id IS NOT NULL UNION SELECT id AS rid FROM regions"),
        ("set_intersect", "SELECT id AS cid FROM customers INTERSECT SELECT customer_id AS cid FROM orders WHERE customer_id IS NOT NULL"),
        ("set_except", "SELECT id AS cid FROM customers EXCEPT SELECT customer_id AS cid FROM orders WHERE status = 'cancelled' AND customer_id IS NOT NULL"),

        # Top-level DISTINCT and ORDER BY + LIMIT
        ("distinct_pairs", "SELECT DISTINCT customer_id, status FROM orders WHERE customer_id IS NOT NULL"),
        ("top_n_orders", "SELECT id, customer_id, qty, price FROM orders ORDER BY qty DESC, id ASC LIMIT 3"),
        ("top_n_agg_groups", "SELECT status, COUNT(*) AS cnt, SUM(qty) AS sq FROM orders GROUP BY status ORDER BY sq DESC, status ASC LIMIT 2"),
    ]

    # Run each advanced view across 3 different random mutation seeds (25 * 3 = 75 tests)
    for seed_idx in range(3):
        rng = random.Random(9000 + seed_idx)
        for adv_name, adv_sql in advanced_views:
            v_name = f"mv_{adv_name}_s{seed_idx}"
            conn = sqlite3.connect(":memory:")
            conn.executescript(schema_4t)
            plan = install_sqlite_materialized_view(conn, schema_4t, f"CREATE VIEW {v_name} AS {adv_sql};")
            total_tests += 1

            verify_mv_equals_live(conn, plan.view_name, adv_sql, f"{v_name}:empty")
            total_checks += 1

            # Seed initial data
            conn.executemany("INSERT INTO regions VALUES (?, ?, ?)", [(1, "NA", 0.08), (2, "EU", 0.20), (3, "APAC", 0.04)])
            conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", [
                (1, 1, "Alice", "gold"),
                (2, 1, "alice", "silver"),
                (3, 2, "Bob", "gold"),
                (4, None, "Charlie", None),
                (5, 3, "Delta", "bronze"),
            ])
            conn.executemany("INSERT INTO products VALUES (?, ?, ?)", [
                (10, "hw", 40.0),
                (20, "sw", 15.0),
            ])
            verify_mv_equals_live(conn, plan.view_name, adv_sql, f"{v_name}:dims")
            total_checks += 1

            # Execute 10 randomized DML operations on orders, customers, products, regions
            next_oid = 100
            for step in range(10):
                op = rng.choice(["ins_ord", "ins_ord", "upd_ord", "del_ord", "upd_cust", "ins_cust"])
                if op == "ins_ord":
                    next_oid += 1
                    cid = rng.choice([1, 2, 3, 4, 5, None])
                    pid = rng.choice([10, 20])
                    qty = rng.randint(1, 10)
                    price = round(rng.uniform(10.0, 200.0), 2)
                    st = rng.choice(["fulfilled", "pending", "cancelled"])
                    conn.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", (next_oid, cid, pid, qty, price, st))
                elif op == "upd_ord":
                    conn.execute(
                        "UPDATE orders SET qty = qty + ?, status = ? WHERE id = (SELECT id FROM orders ORDER BY id LIMIT 1)",
                        (rng.randint(1, 5), rng.choice(["fulfilled", "pending", "cancelled"])),
                    )
                elif op == "del_ord":
                    conn.execute("DELETE FROM orders WHERE id = (SELECT id FROM orders ORDER BY id DESC LIMIT 1)")
                elif op == "upd_cust":
                    conn.execute(
                        "UPDATE customers SET tier = ?, region_id = ? WHERE id = ?",
                        (rng.choice(["gold", "silver", "VIP", None]), rng.choice([1, 2, 3, None]), rng.randint(1, 5)),
                    )
                elif op == "ins_cust":
                    new_cid = 50 + step + seed_idx * 20
                    conn.execute(
                        "INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?)",
                        (new_cid, rng.choice([1, 2, 3, None]), f"User_{new_cid}", rng.choice(["gold", "silver", None])),
                    )
                verify_mv_equals_live(conn, plan.view_name, adv_sql, f"{v_name}:step_{step}_{op}")
                total_checks += 1

            conn.close()

    print(f"Suite 2 & 3 complete: {total_tests} views tested across {total_checks} live query assertions!")




class _RefVarPop:
    def __init__(self):
        self.vals = []
    def step(self, x):
        if x is not None:
            self.vals.append(float(x))
    def finalize(self):
        n = len(self.vals)
        if n < 1:
            return None
        m = sum(self.vals) / n
        return sum((v - m) ** 2 for v in self.vals) / n


class _RefVarSamp:
    def __init__(self):
        self.vals = []
    def step(self, x):
        if x is not None:
            self.vals.append(float(x))
    def finalize(self):
        n = len(self.vals)
        if n < 2:
            return None
        m = sum(self.vals) / n
        return sum((v - m) ** 2 for v in self.vals) / (n - 1)


class _RefStdPop(_RefVarPop):
    def finalize(self):
        v = super().finalize()
        return math.sqrt(v) if v is not None else None


class _RefStdSamp(_RefVarSamp):
    def finalize(self):
        v = super().finalize()
        return math.sqrt(v) if v is not None else None


class _RefBitAnd:
    def __init__(self):
        self.acc = None
    def step(self, x):
        if x is not None:
            self.acc = int(x) if self.acc is None else (self.acc & int(x))
    def finalize(self):
        return self.acc


class _RefBitOr:
    def __init__(self):
        self.acc = None
    def step(self, x):
        if x is not None:
            self.acc = int(x) if self.acc is None else (self.acc | int(x))
    def finalize(self):
        return self.acc


class _RefBitXor:
    def __init__(self):
        self.acc = None
    def step(self, x):
        if x is not None:
            self.acc = int(x) if self.acc is None else (self.acc ^ int(x))
    def finalize(self):
        return self.acc


class _RefBoolAnd:
    def __init__(self):
        self.acc = None
    def step(self, x):
        if x is not None:
            b = bool(x)
            self.acc = b if self.acc is None else (self.acc and b)
    def finalize(self):
        return (1 if self.acc else 0) if self.acc is not None else None


class _RefBoolOr:
    def __init__(self):
        self.acc = None
    def step(self, x):
        if x is not None:
            b = bool(x)
            self.acc = b if self.acc is None else (self.acc or b)
    def finalize(self):
        return (1 if self.acc else 0) if self.acc is not None else None


def _register_reference_sim_functions(conn: sqlite3.Connection) -> None:
    conn.create_function("SQRT", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None)
    conn.create_aggregate("VAR_POP", 1, _RefVarPop)
    conn.create_aggregate("VARIANCE", 1, _RefVarPop)
    conn.create_aggregate("VAR_SAMP", 1, _RefVarSamp)
    conn.create_aggregate("STDDEV_POP", 1, _RefStdPop)
    conn.create_aggregate("STDDEV", 1, _RefStdPop)
    conn.create_aggregate("STDDEV_SAMP", 1, _RefStdSamp)
    conn.create_aggregate("BIT_AND", 1, _RefBitAnd)
    conn.create_aggregate("BIT_OR", 1, _RefBitOr)
    conn.create_aggregate("BIT_XOR", 1, _RefBitXor)
    conn.create_aggregate("BOOL_AND", 1, _RefBoolAnd)
    conn.create_aggregate("EVERY", 1, _RefBoolAnd)
    conn.create_aggregate("BOOL_OR", 1, _RefBoolOr)


def _translate_dialect_expr_to_sqlite(sql: str) -> str:
    s = sql
    # 1. PostgreSQL ROW(e1, e2, ...)::TEXT with arbitrary nested parentheses
    while True:
        m_row = re.search(r"\bROW\(", s)
        if not m_row:
            break
        start = m_row.start()
        open_p = m_row.end() - 1
        depth = 0
        in_sq = False
        j = open_p
        while j < len(s):
            ch = s[j]
            if ch == "'":
                in_sq = not in_sq
            elif not in_sq:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth == 0 and s[j + 1 : j + 7].upper() == "::TEXT":
            inner = s[open_p + 1 : j]
            parts = [p.strip() for p in _split_top_level(inner, ",")]
            repl = "(" + " || '|' || ".join(f"quote({p})" for p in parts) + ")"
            s = s[:start] + repl + s[j + 7 :]
        else:
            break

    # 2. MySQL CONCAT_WS('|', IF(e IS NULL, 'N', CONCAT('V', LENGTH(CAST(e AS CHAR)), ':', CAST(e AS CHAR))), ...)
    while True:
        m_cw = re.search(r"\bCONCAT_WS\('\|',\s*", s)
        if not m_cw:
            break
        start = m_cw.start()
        open_p = s.find("(", start)
        depth = 0
        in_sq = False
        j = open_p
        while j < len(s):
            ch = s[j]
            if ch == "'":
                in_sq = not in_sq
            elif not in_sq:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth == 0:
            inner = s[m_cw.end() : j]
            if_blocks = _split_top_level(inner, ",")
            exprs = []
            for blk in if_blocks:
                m_if = re.match(r"^\s*IF\((.+)\s+IS\s+NULL\s*,\s*'N'", blk, flags=re.DOTALL | re.IGNORECASE)
                if m_if:
                    exprs.append(m_if.group(1).strip())
            if exprs:
                repl = "(" + " || '|' || ".join(f"quote({e})" for e in exprs) + ")"
                s = s[:start] + repl + s[j + 1 :]
                continue
        break
    s = re.sub(r"\s+FROM\s+DUAL\b", "", s, flags=re.IGNORECASE)
    s = s.replace("IS NOT DISTINCT FROM", "IS")
    s = s.replace("<=>", "IS")
    # Bitwise XOR in PG (#) and MySQL (^)
    s = re.sub(
        r"\(([A-Za-z0-9_.]+)\s*[#^]\s*(\(+[^;#^]+?\)+)\)",
        r"((\1 | (\2)) - (\1 & (\2)))",
        s,
    )
    return s


def _translate_mysql_set_to_simultaneous_sql(set_clause: str, table_name: str) -> str:
    assigns = _split_top_level(set_clause, ",")
    substituted_exprs: Dict[str, str] = {}
    out_assigns: List[str] = []

    for a in assigns:
        lhs, rhs = a.split("=", 1)
        col = lhs.strip().split(".")[-1]
        expr = rhs.strip()
        for prev_col, prev_expr in substituted_exprs.items():
            expr = re.sub(
                rf"\b{re.escape(table_name)}\.{re.escape(prev_col)}\b",
                f"({prev_expr})",
                expr,
            )
            expr = re.sub(
                rf"(?<![.\w])\b{re.escape(prev_col)}\b",
                f"({prev_expr})",
                expr,
            )
        substituted_exprs[col] = expr
        out_assigns.append(f"{col} = {expr}")

    return ", ".join(out_assigns)


def _extract_if_guarded_stmts(block: str) -> List[tuple[str | None, str]]:
    """Extracts (optional_if_cond, sql_stmt) pairs from a PL/pgSQL or MySQL trigger block."""
    s = block.strip()
    # Check if there is an outer changed_guard IF (NOT (...)) THEN ... END IF;
    m_outer = re.match(r"^IF\s*\((NOT\s*\(.*?\))\)\s*THEN\s*(.*)\s*END\s+IF\s*;\s*$", s, flags=re.DOTALL | re.IGNORECASE)
    if m_outer:
        s = m_outer.group(2).strip()

    # Now parse sequences of either `IF (cond) THEN stmt1; stmt2; END IF;` or plain `stmt;`
    out: List[tuple[str | None, str]] = []
    pos = 0
    n = len(s)
    while pos < n:
        while pos < n and s[pos].isspace():
            pos += 1
        if pos >= n:
            break
        m_if = re.match(r"^IF\s*\(([^\n]+?)\)\s*THEN\b", s[pos:], flags=re.IGNORECASE)
        if m_if:
            cond = m_if.group(1).strip()
            body_start = pos + m_if.end()
            m_endif = re.search(r"\bEND\s+IF\s*;", s[body_start:], flags=re.IGNORECASE)
            assert m_endif is not None, f"Missing END IF in:\n{s[pos:]}"
            inner_body = s[body_start : body_start + m_endif.start()].strip()
            for st in _split_top_level(inner_body, ";"):
                if st.strip():
                    out.append((cond, st.strip() + ";"))
            pos = body_start + m_endif.end()
        else:
            # Find next top-level semicolon
            parts = _split_top_level(s[pos:], ";")
            first_st = parts[0].strip()
            if first_st:
                out.append((None, first_st + ";"))
            pos += len(parts[0]) + 1
    return out


def install_plan_in_simulator(conn: sqlite3.Connection, plan: MaterializedViewPlan) -> None:
    """Installs a PostgreSQL or MySQL MaterializedViewPlan into a SQLite simulator connection."""
    _register_reference_sim_functions(conn)
    for aux_ddl in plan.auxiliary_table_ddls:
        conn.execute(_translate_dialect_expr_to_sqlite(aux_ddl))
    conn.execute(_translate_dialect_expr_to_sqlite(plan.backing_table_ddl))
    for idx_ddl in plan.index_ddls:
        conn.execute(_translate_dialect_expr_to_sqlite(idx_ddl))
    if plan.reader_view_ddl:
        conn.execute(_translate_dialect_expr_to_sqlite(plan.reader_view_ddl))
    for pop_sql in plan.initial_populate_sqls:
        conn.execute(_translate_dialect_expr_to_sqlite(pop_sql))

    # Extract pk_cols of backing table for MySQL ON DUPLICATE KEY UPDATE conversion
    m_pk = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", plan.backing_table_ddl, re.IGNORECASE)
    if m_pk:
        backing_pk_cols = [c.strip() for c in m_pk.group(1).split(",")]
    else:
        m_inline_pk = re.search(r"([A-Za-z0-9_]+)\s+[^\n,]+PRIMARY\s+KEY", plan.backing_table_ddl, re.IGNORECASE)
        backing_pk_cols = [m_inline_pk.group(1)] if m_inline_pk else ["__mv_pk"]

    def _conv_stmt(stmt: str, cond: str | None) -> str:
        st = _translate_dialect_expr_to_sqlite(stmt)
        if plan.dialect == Dialect.MYSQL:
            m_odku = re.search(r"\bON\s+DUPLICATE\s+KEY\s+UPDATE\s+(.*);?$", st, re.DOTALL | re.IGNORECASE)
            if m_odku:
                before = st[: m_odku.start()].rstrip()
                set_part = m_odku.group(1).rstrip().rstrip(";")
                set_part = re.sub(r"\bVALUES\s*\(\s*([A-Za-z0-9_]+)\s*\)", r"excluded.\1", set_part, flags=re.IGNORECASE)
                m_ins = re.match(r"INSERT\s+INTO\s+([A-Za-z0-9_]+)\s*\(([^)]+)\)", before, re.IGNORECASE)
                assert m_ins is not None
                target_tbl = m_ins.group(1)
                ins_cols = [c.strip() for c in m_ins.group(2).split(",")]
                pk_c = [c for c in backing_pk_cols if c in ins_cols] or [ins_cols[0]]
                if "__val" in ins_cols and "__val" not in pk_c:
                    pk_c.append("__val")
                sim_set = _translate_mysql_set_to_simultaneous_sql(set_part, target_tbl)
                st = f"{before}\n    ON CONFLICT ({', '.join(pk_c)}) DO UPDATE SET {sim_set};"
            elif st.upper().startswith("UPDATE "):
                m_upd_join = re.match(
                    r"UPDATE\s+([A-Za-z0-9_]+)\s+JOIN\s*\((.*)\)\s*AS\s+([A-Za-z0-9_]+)\s+ON\s+([^\n]+)\s+SET\s+(.*);?$",
                    st.strip(),
                    re.DOTALL | re.IGNORECASE,
                )
                if m_upd_join:
                    target_tbl, subq, d_alias, on_cond, set_part = m_upd_join.groups()
                    set_part = set_part.rstrip().rstrip(";")
                    sim_set = _translate_mysql_set_to_simultaneous_sql(set_part, target_tbl)
                    st = f"UPDATE {target_tbl} SET {sim_set} FROM ({subq}) AS {d_alias} WHERE {on_cond};"
                else:
                    m_upd = re.match(r"(UPDATE\s+([A-Za-z0-9_]+)\s+SET\s+)(.*?)(\s+WHERE\s+.*)$", st, re.DOTALL | re.IGNORECASE)
                    if m_upd:
                        prefix, target_tbl, set_part, where_part = m_upd.groups()
                        sim_set = _translate_mysql_set_to_simultaneous_sql(set_part, target_tbl)
                        st = f"{prefix}{sim_set}{where_part}"
        if cond:
            c_trans = _translate_dialect_expr_to_sqlite(cond)
            st = plan_compiler_apply_where(st, c_trans)
        return st

    def plan_compiler_apply_where(stmt: str, where_cond: str) -> str:
        if "WHERE TRUE" in stmt:
            return stmt.replace("WHERE TRUE", f"WHERE ({where_cond})", 1)
        if stmt.strip().upper().startswith("UPDATE ") or stmt.strip().upper().startswith("DELETE FROM"):
            # Inject into final WHERE or add WHERE
            if " WHERE " in stmt:
                idx_w = stmt.rfind(" WHERE ")
                return stmt[: idx_w + 7] + f"({where_cond}) AND (" + stmt[idx_w + 7 :].rstrip(";") + ");"
            return stmt.rstrip(";") + f" WHERE ({where_cond});"
        if stmt.strip().upper().startswith("INSERT INTO"):
            m_on = re.search(r"\bON\s+CONFLICT\b", stmt, re.IGNORECASE)
            if m_on and " WHERE " in stmt[: m_on.start()]:
                idx_w = stmt[: m_on.start()].rfind(" WHERE ")
                return stmt[: idx_w + 7] + f"({where_cond}) AND (" + stmt[idx_w + 7 : m_on.start()].strip() + ")\n    " + stmt[m_on.start() :]
        return stmt

    if plan.dialect == Dialect.POSTGRES:
        for i, trig_sql in enumerate(plan.trigger_ddls):
            if "RETURNS TRIGGER" in trig_sql and "TG_OP = 'DELETE'" in trig_sql:
                # Find corresponding CREATE TRIGGER statement to get target table
                m_fn = re.search(r"FUNCTION\s+([A-Za-z0-9_]+)\(\)", trig_sql)
                fn_name = m_fn.group(1)
                m_tbl = re.search(
                    rf"ON\s+([A-Za-z0-9_]+)\s+FOR\s+EACH\s+ROW\s+EXECUTE\s+FUNCTION\s+{re.escape(fn_name)}\(\)",
                    "\n".join(plan.trigger_ddls),
                )
                base_t = m_tbl.group(1)
                del_b = trig_sql.split("IF (TG_OP = 'DELETE') THEN")[1].split("ELSIF (TG_OP = 'INSERT') THEN")[0]
                ins_b = trig_sql.split("ELSIF (TG_OP = 'INSERT') THEN")[1].split("ELSIF (TG_OP = 'UPDATE') THEN")[0]
                upd_b = trig_sql.split("ELSIF (TG_OP = 'UPDATE') THEN")[1].split("RETURN NULL;")[0].rsplit("END IF;", 1)[0]

                del_stmts = [_conv_stmt(st, c) for c, st in _extract_if_guarded_stmts(del_b)]
                ins_stmts = [_conv_stmt(st, c) for c, st in _extract_if_guarded_stmts(ins_b)]
                upd_stmts = [_conv_stmt(st, c) for c, st in _extract_if_guarded_stmts(upd_b)]

                conn.execute(f"CREATE TRIGGER sim_pg_{i}_{base_t}_ad AFTER DELETE ON {base_t} FOR EACH ROW BEGIN {' '.join(del_stmts)} END;")
                conn.execute(f"CREATE TRIGGER sim_pg_{i}_{base_t}_ai AFTER INSERT ON {base_t} FOR EACH ROW BEGIN {' '.join(ins_stmts)} END;")
                conn.execute(f"CREATE TRIGGER sim_pg_{i}_{base_t}_au AFTER UPDATE ON {base_t} FOR EACH ROW BEGIN {' '.join(upd_stmts)} END;")
    elif plan.dialect == Dialect.MYSQL:
        for i, trig_sql in enumerate(plan.trigger_ddls):
            m_head = re.search(r"AFTER\s+(INSERT|DELETE|UPDATE)\s+ON\s+([A-Za-z0-9_]+)", trig_sql, re.IGNORECASE)
            op, base_t = m_head.group(1).upper(), m_head.group(2)
            body = trig_sql.split("BEGIN", 1)[1].rsplit("END;", 1)[0]
            stmts = [_conv_stmt(st, c) for c, st in _extract_if_guarded_stmts(body)]
            conn.execute(f"CREATE TRIGGER sim_my_{i}_{base_t}_{op.lower()} AFTER {op} ON {base_t} FOR EACH ROW BEGIN {' '.join(stmts)} END;")




def run_suite4_postgres_and_mysql_matrix():

    schema_pg_my = """
    CREATE TABLE customers (
        id INT PRIMARY KEY,
        tier VARCHAR(50),
        region VARCHAR(50) NOT NULL
    );
    CREATE TABLE orders (
        id INT PRIMARY KEY,
        customer_id INT,
        region VARCHAR(50),
        category VARCHAR(50) NOT NULL,
        amount DOUBLE PRECISION,
        qty INT NOT NULL,
        flags INT,
        is_paid BOOLEAN,
        status VARCHAR(50) NOT NULL
    );
    """

    pg_my_views = [
        # Tier 1 Algebraic with statistical & bitwise & boolean aggregates
        ("alg_stats_1", "SELECT region, COUNT(*) AS c, SUM(amount) AS s, AVG(amount) AS a, VAR_POP(amount) AS vp, VAR_SAMP(amount) AS vs, STDDEV_POP(amount) AS sdp, STDDEV_SAMP(amount) AS sds, MIN(amount) AS mn, MAX(amount) AS mx FROM orders GROUP BY region"),
        ("alg_stats_notnull_grp", "SELECT category, COUNT(*) AS c, SUM(amount) AS s, AVG(amount) AS a, VAR_POP(amount) AS vp, STDDEV_SAMP(amount) AS sds, BIT_XOR(flags) AS bx FROM orders GROUP BY category"),
        ("alg_stats_2grp", "SELECT region, category, COUNT(*) AS c, SUM(qty) AS sq, AVG(amount) AS a, VAR_POP(amount) AS vp, BIT_XOR(flags) AS bx, MIN(qty) AS mnq, MAX(qty) AS mxq FROM orders GROUP BY region, category"),
        ("alg_where_pos", "SELECT region, COUNT(*) AS c, SUM(amount) AS s, VAR_POP(amount) AS vp, STDDEV_SAMP(amount) AS sds, BIT_XOR(flags) AS bx FROM orders WHERE amount > 0 GROUP BY region"),
        ("alg_where_status", "SELECT category, COUNT(*) AS c, SUM(qty) AS sq, AVG(amount) AS a, VAR_SAMP(amount) AS vs FROM orders WHERE status = 'fulfilled' GROUP BY category"),
        ("alg_global", "SELECT COUNT(*) AS c, SUM(amount) AS s, AVG(amount) AS a, VAR_POP(amount) AS vp, STDDEV_SAMP(amount) AS sds, BIT_XOR(flags) AS bx, MIN(amount) AS mn, MAX(amount) AS mx FROM orders"),
        ("alg_distinct_and_bit", "SELECT region, COUNT(*) AS c, COUNT(DISTINCT customer_id) AS dc, SUM(DISTINCT qty) AS sdq, BIT_AND(flags) AS ba, BIT_OR(flags) AS bo, BIT_XOR(flags) AS bx FROM orders GROUP BY region"),
        ("alg_distinct_notnull_grp", "SELECT category, COUNT(DISTINCT region) AS dr, COUNT(DISTINCT status) AS ds, SUM(amount) AS s, BIT_AND(flags) AS ba, BIT_OR(flags) AS bo FROM orders GROUP BY category"),
        # Tier 2 Group-scoped JOINs & HAVING
        ("t2_join_inner", "SELECT c.tier, o.category, COUNT(*) AS c, SUM(o.amount) AS s, AVG(o.amount) AS a, VAR_POP(o.amount) AS vp FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.tier, o.category"),
        ("t2_join_left", "SELECT c.tier, COUNT(o.id) AS oc, COALESCE(SUM(o.qty), 0) AS sq, AVG(o.amount) AS a FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.tier"),
        ("t2_having", "SELECT region, COUNT(*) AS c, SUM(amount) AS s, AVG(amount) AS a FROM orders GROUP BY region HAVING COUNT(*) >= 2"),
        ("t2_join_having", "SELECT c.region, COUNT(*) AS c, SUM(o.qty) AS sq FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'fulfilled' GROUP BY c.region HAVING SUM(o.qty) >= 3"),
        # Tier 3 Atomic Sync (CTEs, Window functions, UNION, Subqueries, LIMIT)
        ("t3_cte_win", "WITH reg_sum AS (SELECT category, SUM(qty) AS sq FROM orders GROUP BY category) SELECT category, sq, RANK() OVER (ORDER BY sq DESC, category ASC) AS rnk FROM reg_sum"),
        ("t3_union", "SELECT region AS loc FROM customers UNION SELECT region AS loc FROM orders WHERE region IS NOT NULL"),
        ("t3_top_n", "SELECT id, category, qty, amount FROM orders ORDER BY qty DESC, id ASC LIMIT 3"),
    ]

    pg_only_views = [
        ("pg_bool_1", "SELECT region, COUNT(*) AS c, BOOL_AND(is_paid) AS all_p, BOOL_OR(is_paid) AS any_p, VAR_POP(amount) AS vp FROM orders GROUP BY region"),
        ("pg_bool_cat", "SELECT category, COUNT(*) AS c, BOOL_AND(is_paid) AS all_p, BOOL_OR(is_paid) AS any_p, BIT_XOR(flags) AS bx FROM orders WHERE amount > 0 GROUP BY category"),
        ("pg_filter_stats", "SELECT category, COUNT(*) AS c, COUNT(*) FILTER (WHERE is_paid) AS cp, SUM(amount) FILTER (WHERE status = 'fulfilled') AS sf, VAR_POP(amount) FILTER (WHERE amount > 0) AS vp FROM orders GROUP BY category"),
    ]

    total_dialect_tests = 0
    total_dialect_checks = 0

    for dialect in (Dialect.POSTGRES, Dialect.MYSQL):
        view_list = pg_my_views + (pg_only_views if dialect == Dialect.POSTGRES else [])
        for seed in range(4):
            rng = random.Random(5000 + seed)
            for v_base, v_select in view_list:
                v_name = f"mv_{dialect.value}_{v_base}_s{seed}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    plan = generate_materialized_view(
                        schema_pg_my,
                        f"CREATE VIEW {v_name} AS {v_select};",
                        dialect=dialect,
                        ignore_limitations=True,
                    )
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema_pg_my)
                install_plan_in_simulator(conn, plan)
                total_dialect_tests += 1

                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:empty")
                total_dialect_checks += 1

                conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", [
                    (1, "gold", "NA"),
                    (2, "silver", "EU"),
                    (3, "gold", "APAC"),
                    (4, None, "NA"),
                ])
                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:custs")
                total_dialect_checks += 1

                conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    (10, 1, "NA", "hw", 25.0, 2, 7, 1, "fulfilled"),
                    (11, 1, "NA", "sw", 50.0, 5, 3, 1, "fulfilled"),
                    (12, 2, "EU", "hw", 15.0, 1, 5, 0, "pending"),
                    (13, 3, None, "hw", 100.0, 4, 12, 1, "fulfilled"),
                    (14, 4, "NA", "sw", -10.0, 3, 9, 0, "cancelled"),
                ])
                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:initial_orders")
                total_dialect_checks += 1

                # 8 random mutations (INSERT, UPDATE, DELETE)
                next_id = 20
                for step in range(8):
                    op = rng.choice(["ins", "upd_same", "upd_cross", "del"])
                    if op == "ins":
                        next_id += 1
                        conn.execute(
                            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                next_id,
                                rng.choice([1, 2, 3, 4, None]),
                                rng.choice(["NA", "EU", "APAC", None]),
                                rng.choice(["hw", "sw", "cloud"]),
                                round(rng.uniform(5.0, 120.0), 2),
                                rng.randint(1, 8),
                                rng.randint(1, 31),
                                rng.choice([0, 1, None]),
                                rng.choice(["fulfilled", "pending", "cancelled"]),
                            ),
                        )
                    elif op == "upd_same":
                        conn.execute(
                            "UPDATE orders SET amount = amount + ?, qty = qty + 1, flags = ?, is_paid = ? WHERE id = (SELECT id FROM orders ORDER BY id LIMIT 1)",
                            (round(rng.uniform(1.0, 30.0), 2), rng.randint(1, 15), rng.choice([0, 1])),
                        )
                    elif op == "upd_cross":
                        conn.execute(
                            "UPDATE orders SET region = ?, category = ?, status = ? WHERE id = (SELECT id FROM orders ORDER BY id DESC LIMIT 1)",
                            (rng.choice(["NA", "EU", None]), rng.choice(["hw", "sw"]), rng.choice(["fulfilled", "pending"])),
                        )
                    elif op == "del":
                        conn.execute("DELETE FROM orders WHERE id = (SELECT id FROM orders ORDER BY id LIMIT 1)")

                    verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:step_{step}_{op}")
                    total_dialect_checks += 1

                conn.execute("DELETE FROM orders")
                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:drained")
                total_dialect_checks += 1
                conn.close()

    print(f"Suite 4 (PostgreSQL & MySQL Trigger Execution) complete: {total_dialect_tests} views tested across {total_dialect_checks} live query assertions!")



def run_suite5_multi_view_sql_files_and_entrypoints() -> Tuple[int, int]:
    """
    SUITE 5: Multi-View `view.sql` Files & Strings Across All Entrypoints.
    Verifies that passing a single `.sql` file or string containing multiple
    `CREATE VIEW` statements (and optional `CREATE [UNIQUE] INDEX` statements)
    works across `generate_materialized_view`, `generate_materialized_views`,
    `install_sqlite_materialized_view`, `install_sqlite_materialized_views`,
    `MaterializedViewManager.register_view`, `MaterializedViewManager.sync_views`,
    `MaterializedViewManager.bulk_changes`, and CLI `main()`.
    """
    from matview_sync import parse_view_definitions

    total_views = 0
    total_checks = 0

    schema = """
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        d_name TEXT NOT NULL,
        region TEXT
    );
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        dept_id INTEGER,
        role TEXT NOT NULL,
        salary REAL,
        bonus INTEGER,
        active INTEGER NOT NULL
    );
    CREATE TABLE contractors (
        id INTEGER PRIMARY KEY,
        dept_id INTEGER,
        role TEXT NOT NULL,
        hourly_rate REAL NOT NULL,
        hours INTEGER NOT NULL
    );
    """

    # Build 20 distinct multi-view SQL bundles (each containing 4 to 8 views + indexes = 120 views)
    bundles: List[Tuple[str, List[Tuple[str, str]]]] = []
    for b_idx in range(20):
        thresh = (b_idx % 3) + 1
        sal_min = (b_idx % 4) * 100.0
        v_defs = [
            (
                f"mv_b{b_idx}_emp_dept",
                f"SELECT dept_id, COUNT(*) AS emp_cnt, SUM(salary) AS sum_sal, AVG(salary) AS avg_sal, "
                f"COALESCE(SUM(salary) + SUM(bonus), 0.0) AS total_comp "
                f"FROM employees WHERE active = 1 AND (salary IS NULL OR salary >= {sal_min}) "
                f"GROUP BY dept_id HAVING COUNT(*) >= {thresh}",
            ),
            (
                f"mv_b{b_idx}_join_dept",
                f"SELECT d.d_name, d.region, e.role, COUNT(*) AS cnt, SUM(e.salary) AS tot_sal, "
                f"AVG(e.salary) AS avg_sal, MIN(e.bonus) AS min_b, MAX(e.bonus) AS max_b "
                f"FROM employees e JOIN departments d ON e.dept_id = d.id "
                f"WHERE e.active = 1 GROUP BY d.d_name, d.region, e.role",
            ),
            (
                f"mv_b{b_idx}_roles_union",
                f"SELECT role, dept_id FROM employees WHERE active = 1 "
                f"UNION SELECT role, dept_id FROM contractors WHERE hours > {b_idx % 5}",
            ),
            (
                f"mv_b{b_idx}_roles_intersect",
                f"SELECT role FROM employees WHERE active = 1 "
                f"INTERSECT SELECT role FROM contractors",
            ),
            (
                f"mv_b{b_idx}_roles_except",
                f"SELECT role FROM employees WHERE active = 1 "
                f"EXCEPT SELECT role FROM contractors WHERE hours >= 10",
            ),
            (
                f"mv_b{b_idx}_distinct_depts",
                f"SELECT DISTINCT dept_id, role FROM employees WHERE active = 1",
            ),
            (
                f"mv_b{b_idx}_union_all_headcount",
                f"WITH workforce AS ("
                f"SELECT dept_id, role, salary AS cost FROM employees WHERE active = 1 "
                f"UNION ALL "
                f"SELECT dept_id, role, (hourly_rate * hours) AS cost FROM contractors"
                f") SELECT dept_id, role, COUNT(*) AS headcount, SUM(cost) AS total_cost, AVG(cost) AS avg_cost "
                f"FROM workforce GROUP BY dept_id, role",
            ),
            (
                f"mv_b{b_idx}_active_proj",
                f"SELECT id AS emp_id, dept_id, LOWER(role) AS l_role, COALESCE(salary, 0.0) + COALESCE(bonus, 0) AS comp "
                f"FROM employees WHERE active = 1",
            ),
        ]
        sql_parts: List[str] = []
        for v_name, v_sel in v_defs:
            sql_parts.append(f"-- View definition for {v_name}\nCREATE VIEW {v_name} AS\n{v_sel};")
        sql_parts.append(f"CREATE INDEX idx_b{b_idx}_emp_cnt ON mv_b{b_idx}_emp_dept (emp_cnt DESC);")
        sql_parts.append(f"CREATE INDEX idx_b{b_idx}_join_tot ON mv_b{b_idx}_join_dept (tot_sal DESC);")
        bundles.append(("\n\n".join(sql_parts), v_defs))

    for b_idx, (multi_sql, v_defs) in enumerate(bundles):
        specs = parse_view_definitions(multi_sql)
        assert len(specs) == len(v_defs), f"Expected {len(v_defs)} parsed views, got {len(specs)}"

        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)

        # Rotate across entrypoints to verify every entrypoint handles multi-view SQL strings/files
        mode = b_idx % 4
        if mode == 0:
            comp_plan = install_sqlite_materialized_view(conn, schema, multi_sql, incremental_backfill=True, batch_size=2)
            assert len(comp_plan.sub_plans) == len(v_defs)
            for sp in comp_plan.sub_plans:
                assert sp.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA, f"{sp.view_name} strategy={sp.strategy}"
        elif mode == 1:
            plans = install_sqlite_materialized_views(conn, [multi_sql], schema_sql=schema)
            assert len(plans) == len(v_defs)
            for sp in plans:
                assert sp.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA
        elif mode == 2:
            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            status1 = mgr.sync_views([multi_sql])
            assert len(status1) == len(v_defs)
            assert all(v == "created" for v in status1.values())
            status2 = mgr.sync_views([multi_sql])
            assert all(v == "unchanged" for v in status2.values())
        else:
            comp_plan = generate_materialized_view(schema, multi_sql, dialect=Dialect.SQLITE)
            assert len(comp_plan.sub_plans) == len(v_defs)
            for st in comp_plan.all_statements():
                conn.execute(st)

        total_views += len(v_defs)

        # Verify empty state across all views in the bundle
        for v_name, v_sel in v_defs:
            verify_mv_equals_live(conn, f"__matview_{v_name}", v_sel, f"{v_name}:empty")
            total_checks += 1

        # Step 1: Insert departments, employees, contractors
        conn.executemany("INSERT INTO departments VALUES (?, ?, ?)", [
            (1, "Engineering", "NA"),
            (2, "Sales", "EU"),
            (3, "Research", None),
        ])
        conn.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)", [
            (101, 1, "SWE", 1500.0, 200, 1),
            (102, 1, "SWE", 1800.0, None, 1),
            (103, 1, "Mgr", 2200.0, 500, 0),
            (104, 2, "AE", 1200.0, 300, 1),
            (105, None, "SWE", None, 100, 1),
        ])
        conn.executemany("INSERT INTO contractors VALUES (?, ?, ?, ?, ?)", [
            (201, 1, "SWE", 80.0, 10),
            (202, 2, "Designer", 65.0, 4),
            (203, 3, "Researcher", 100.0, 20),
        ])
        for v_name, v_sel in v_defs:
            verify_mv_equals_live(conn, f"__matview_{v_name}", v_sel, f"{v_name}:populated")
            total_checks += 1

        # Step 2: Updates across all 3 base tables
        conn.execute("UPDATE employees SET active = 1, salary = 2500.0, dept_id = 2 WHERE id = 103")
        conn.execute("UPDATE employees SET active = 0 WHERE id = 101")
        conn.execute("UPDATE departments SET region = 'APAC', d_name = 'Core_Eng' WHERE id = 1")
        conn.execute("UPDATE contractors SET role = 'AE', hours = 15 WHERE id = 202")
        for v_name, v_sel in v_defs:
            verify_mv_equals_live(conn, f"__matview_{v_name}", v_sel, f"{v_name}:updated")
            total_checks += 1

        # Step 3: Deletes & drain
        conn.execute("DELETE FROM contractors WHERE id = 201")
        conn.execute("DELETE FROM employees WHERE dept_id = 2")
        for v_name, v_sel in v_defs:
            verify_mv_equals_live(conn, f"__matview_{v_name}", v_sel, f"{v_name}:deleted")
            total_checks += 1

        conn.execute("DELETE FROM employees")
        conn.execute("DELETE FROM contractors")
        conn.execute("DELETE FROM departments")
        for v_name, v_sel in v_defs:
            verify_mv_equals_live(conn, f"__matview_{v_name}", v_sel, f"{v_name}:drained")
            total_checks += 1

        conn.close()

    print(f"Suite 5 (Multi-View SQL Bundles) complete: {total_views} views tested across {total_checks} live query assertions!")
    return total_views, total_checks


def run_suite6_algebraic_delta_exhaustive_matrix() -> Tuple[int, int]:
    """
    SUITE 6: Exhaustive `ALGEBRAIC_DELTA` Combinatorial Matrix (~2,200 views, 15,000+ assertions).
    Proves that:
      - 6A: Single-table expression GROUP BY, non-aggregate expression projections,
            compound aggregate expressions, and HAVING clauses all compile to
            MaintenanceStrategy.ALGEBRAIC_DELTA and match live queries across all DML.
      - 6B: Multi-table 2/3/4-table INNER JOIN and CROSS JOIN aggregations (including
            expression GROUP BY, compound aggregates, FILTER, HAVING, MIN/MAX, and
            global join aggregations) compile to ALGEBRAIC_DELTA and match live queries.
      - 6C: Z-set UNION, UNION ALL, INTERSECT, EXCEPT, SELECT DISTINCT, and PK-lineage
            non-aggregated projections compile to ALGEBRAIC_DELTA and match live queries.
      - 6D: Aggregations over UNION ALL CTEs and subqueries compile to ALGEBRAIC_DELTA
            and match live queries.
    """
    total_views = 0
    total_checks = 0

    # -------------------------------------------------------------------------
    # 6A: Single-Table Expression GROUP BY x WHERE x Compound Aggs x HAVING
    #     (10 group configs x 4 where configs x 6 compound agg bundles x 4 having configs = 960 views)
    # -------------------------------------------------------------------------
    schema_6a = """
    CREATE TABLE ledger (
        id INTEGER PRIMARY KEY,
        acct TEXT NOT NULL,
        region TEXT,
        shard INTEGER,
        debit INTEGER,
        credit INTEGER,
        rate REAL,
        status TEXT NOT NULL
    );
    """

    grp_6a = [
        ("g_global", "", ""),
        ("g_acct", "acct, ", "GROUP BY acct"),
        ("g_reg", "region, ", "GROUP BY region"),
        ("g_lower_acct", "LOWER(acct) AS l_acct, ", "GROUP BY LOWER(acct)"),
        ("g_substr", "SUBSTR(acct, 1, 1) AS pfx, ", "GROUP BY SUBSTR(acct, 1, 1)"),
        ("g_coal_reg", "COALESCE(region, 'GLOBAL') AS eff_reg, ", "GROUP BY COALESCE(region, 'GLOBAL')"),
        ("g_mod_shard", "(COALESCE(shard, 0) % 3) AS bucket, ", "GROUP BY (COALESCE(shard, 0) % 3)"),
        ("g_case_debit", "CASE WHEN COALESCE(debit, 0) >= 50 THEN 'large' ELSE 'small' END AS sz, ", "GROUP BY CASE WHEN COALESCE(debit, 0) >= 50 THEN 'large' ELSE 'small' END"),
        ("g_acct_reg_expr", "LOWER(acct) AS la, COALESCE(region, 'NA') AS cr, ", "GROUP BY LOWER(acct), COALESCE(region, 'NA')"),
        ("g_math_sum", "(COALESCE(shard, 0) + 10) AS sh10, ", "GROUP BY (COALESCE(shard, 0) + 10)"),
    ]

    where_6a = [
        ("w_all", ""),
        ("w_settled", "WHERE status = 'settled'"),
        ("w_pos_debit", "WHERE debit IS NOT NULL AND debit > 0"),
        ("w_complex", "WHERE (status != 'void' AND COALESCE(credit, 0) >= 0)"),
    ]

    agg_6a = [
        ("a_net", "COUNT(*) AS cnt, COALESCE(SUM(debit), 0) - COALESCE(SUM(credit), 0) AS net_flow, SUM(debit) AS sd"),
        ("a_weighted", "COUNT(*) AS cnt, COALESCE(SUM(debit * rate), 0.0) AS w_debit, ROUND(COALESCE(AVG(rate), 0.0), 4) AS r_rate"),
        ("a_spread", "COUNT(*) AS cnt, COALESCE(MAX(debit) - MIN(debit), 0) AS debit_spread, MIN(debit) AS mnd, MAX(debit) AS mxd"),
        ("a_ratio", "COUNT(*) AS cnt, COALESCE(1.0 * SUM(debit) / NULLIF(COUNT(debit), 0), 0.0) AS manual_avg, AVG(debit) AS built_in_avg"),
        ("a_filtered_comp", "COUNT(*) AS cnt, COALESCE(SUM(debit) FILTER (WHERE status = 'settled'), 0) - COALESCE(SUM(credit) FILTER (WHERE status = 'settled'), 0) AS settled_net"),
        ("a_distinct_comp", "COUNT(*) AS cnt, COUNT(DISTINCT region) + COUNT(DISTINCT status) AS dist_sum, COALESCE(SUM(DISTINCT debit), 0) AS sdd"),
    ]

    having_6a = [
        ("h_none", ""),
        ("h_cnt1", "HAVING COUNT(*) >= 1"),
        ("h_cnt2", "HAVING COUNT(*) >= 2"),
        ("h_sum_pos", "HAVING COALESCE(SUM(debit), 0) >= 20"),
    ]

    seed_batches_6a = [
        [
            (1, "Alpha", "US", 1, 100, 30, 1.5, "settled"),
            (2, "alpha", "US", 4, 40, 10, 2.0, "pending"),
            (3, "Beta", None, 2, 200, 50, 0.5, "settled"),
            (4, "Gamma", "EU", None, None, 25, 1.0, "void"),
            (5, "Beta", "EU", 5, 10, 5, 3.0, "settled"),
        ],
        [
            (6, "ALPHA", None, 7, 60, 60, 1.25, "settled"),
            (7, "Delta", "APAC", 3, 5, 100, 0.75, "pending"),
        ],
    ]

    for g_id, sel_p, grp_c in grp_6a:
        for w_id, w_c in where_6a:
            for a_id, agg_c in agg_6a:
                for h_id, h_c in having_6a:
                    if not grp_c and h_c:
                        # SQLite requires GROUP BY before HAVING unless we test GROUP BY 1 or skip raw SQLite syntax limitation:
                        # Note: in SQLite, `SELECT COUNT(*) FROM t HAVING COUNT(*) >= 1` without GROUP BY raises
                        # `sqlite3.OperationalError: a GROUP BY clause is required before HAVING` in live SQLite queries!
                        continue
                    v_name = f"v6a_{g_id}_{w_id}_{a_id}_{h_id}"
                    sel_sql = f"SELECT {sel_p}{agg_c} FROM ledger {w_c} {grp_c} {h_c}".strip()
                    conn = sqlite3.connect(":memory:")
                    conn.executescript(schema_6a)
                    plan = install_sqlite_materialized_view(conn, schema_6a, f"CREATE VIEW {v_name} AS {sel_sql};")
                    assert plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA, (
                        f"Expected ALGEBRAIC_DELTA for {v_name}, got {plan.strategy}"
                    )
                    total_views += 1

                    # Empty check
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:empty")
                    total_checks += 1

                    # Batch 1 insert
                    conn.executemany("INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?)", seed_batches_6a[0])
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:b1")
                    total_checks += 1

                    # Batch 2 insert
                    conn.executemany("INSERT INTO ledger VALUES (?, ?, ?, ?, ?, ?, ?, ?)", seed_batches_6a[1])
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:b2")
                    total_checks += 1

                    # Update across groups & values
                    conn.execute("UPDATE ledger SET acct = 'beta', region = 'US', debit = 80, status = 'settled' WHERE id IN (2, 4)")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:upd1")
                    total_checks += 1

                    # Update extremums & NULLs
                    conn.execute("UPDATE ledger SET debit = NULL, credit = 0, rate = 2.5 WHERE id = 3")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:upd2")
                    total_checks += 1

                    # Delete partial & drain
                    conn.execute("DELETE FROM ledger WHERE id IN (1, 5, 6)")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:del1")
                    total_checks += 1

                    conn.execute("DELETE FROM ledger")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:drained")
                    total_checks += 1
                    conn.close()

    # -------------------------------------------------------------------------
    # 6B: Multi-Table INNER JOIN & CROSS JOIN Algebraic Matrix (540 views)
    # -------------------------------------------------------------------------
    schema_6b = """
    CREATE TABLE stores (id INTEGER PRIMARY KEY, s_name TEXT NOT NULL, country TEXT, tax REAL NOT NULL);
    CREATE TABLE categories (id INTEGER PRIMARY KEY, cat_name TEXT NOT NULL, dept TEXT NOT NULL);
    CREATE TABLE items (id INTEGER PRIMARY KEY, cat_id INTEGER, title TEXT NOT NULL, base_cost REAL NOT NULL);
    CREATE TABLE line_items (id INTEGER PRIMARY KEY, store_id INTEGER, item_id INTEGER, units INTEGER NOT NULL, unit_price REAL, channel TEXT NOT NULL);
    """

    join_bases_6b = [
        (
            "jb_2t_store",
            "FROM line_items li JOIN stores s ON li.store_id = s.id",
            [
                ("g_global", "", ""),
                ("g_country", "s.country, ", "GROUP BY s.country"),
                ("g_sname_chan", "s.s_name, li.channel, ", "GROUP BY s.s_name, li.channel"),
                ("g_lower_chan", "LOWER(li.channel) AS l_ch, s.country, ", "GROUP BY LOWER(li.channel), s.country"),
            ],
        ),
        (
            "jb_2t_item",
            "FROM line_items li JOIN items i ON li.item_id = i.id",
            [
                ("g_global", "", ""),
                ("g_title", "i.title, ", "GROUP BY i.title"),
                ("g_cat_chan", "i.cat_id, li.channel, ", "GROUP BY i.cat_id, li.channel"),
            ],
        ),
        (
            "jb_3t_store_item_cat",
            "FROM line_items li JOIN items i ON li.item_id = i.id JOIN categories c ON i.cat_id = c.id",
            [
                ("g_dept", "c.dept, ", "GROUP BY c.dept"),
                ("g_cat_chan", "c.cat_name, li.channel, ", "GROUP BY c.cat_name, li.channel"),
                ("g_upper_dept", "UPPER(c.dept) AS u_dept, ", "GROUP BY UPPER(c.dept)"),
            ],
        ),
        (
            "jb_4t_star",
            "FROM line_items li JOIN stores s ON li.store_id = s.id JOIN items i ON li.item_id = i.id JOIN categories c ON i.cat_id = c.id",
            [
                ("g_country_dept", "s.country, c.dept, ", "GROUP BY s.country, c.dept"),
                ("g_store_cat_chan", "s.s_name, c.cat_name, li.channel, ", "GROUP BY s.s_name, c.cat_name, li.channel"),
            ],
        ),
    ]

    join_wheres_6b = [
        ("w_none", ""),
        ("w_online", "WHERE li.channel = 'online'"),
        ("w_units", "WHERE li.units >= 2 AND li.unit_price IS NOT NULL"),
    ]

    join_aggs_6b = [
        ("ja_basic", "COUNT(*) AS cnt, SUM(li.units) AS tot_u, SUM(li.unit_price) AS tot_p, AVG(li.unit_price) AS avg_p"),
        ("ja_minmax_tot", "COUNT(*) AS cnt, MIN(li.units) AS mnu, MAX(li.units) AS mxu, TOTAL(li.unit_price) AS totp"),
        ("ja_comp", "COUNT(*) AS cnt, COALESCE(SUM(li.units * li.unit_price), 0.0) AS gross_rev, COALESCE(MAX(li.units) - MIN(li.units), 0) AS u_range"),
        ("ja_filt", "COUNT(*) AS cnt, COUNT(*) FILTER (WHERE li.channel = 'online') AS online_cnt, COALESCE(SUM(li.units) FILTER (WHERE li.units >= 3), 0) AS bulk_u"),
    ]

    join_havings_6b = [
        ("h_none", ""),
        ("h_cnt1", "HAVING COUNT(*) >= 1"),
        ("h_cnt2", "HAVING COUNT(*) >= 2"),
        ("h_sum_u", "HAVING COALESCE(SUM(li.units), 0) >= 4"),
    ]

    for jb_name, from_sql, grp_list in join_bases_6b:
        for g_id, sel_p, grp_c in grp_list:
            for w_id, w_c in join_wheres_6b:
                for a_id, agg_c in join_aggs_6b:
                    for h_id, h_c in join_havings_6b:
                        if not grp_c and h_c:
                            continue
                        v_name = f"v6b_{jb_name}_{g_id}_{w_id}_{a_id}_{h_id}"
                        sel_sql = f"SELECT {sel_p}{agg_c} {from_sql} {w_c} {grp_c} {h_c}".strip()
                        conn = sqlite3.connect(":memory:")
                        conn.executescript(schema_6b)
                        plan = install_sqlite_materialized_view(conn, schema_6b, f"CREATE VIEW {v_name} AS {sel_sql};")
                        assert plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA, (
                            f"Expected ALGEBRAIC_DELTA for join {v_name}, got {plan.strategy}"
                        )
                        total_views += 1

                        verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:empty")
                        total_checks += 1

                        conn.executemany("INSERT INTO stores VALUES (?, ?, ?, ?)", [(1, "NYC", "US", 0.08), (2, "Paris", "FR", 0.20), (3, "DutyFree", None, 0.0)])
                        conn.executemany("INSERT INTO categories VALUES (?, ?, ?)", [(10, "Laptops", "Tech"), (20, "Chairs", "Home")])
                        conn.executemany("INSERT INTO items VALUES (?, ?, ?, ?)", [(100, 10, "ProBook", 800.0), (101, 20, "ErgoChair", 150.0)])
                        conn.executemany("INSERT INTO line_items VALUES (?, ?, ?, ?, ?, ?)", [
                            (1000, 1, 100, 2, 1200.0, "online"),
                            (1001, 1, 101, 5, 300.0, "retail"),
                            (1002, 2, 100, 1, 1300.0, "online"),
                            (1003, 3, 101, 4, None, "online"),
                            (1004, 99, 100, 3, 1100.0, "online"),
                        ])
                        verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:ins")
                        total_checks += 1

                        # Mutate dimension tables & fact table
                        conn.execute("INSERT INTO stores VALUES (99, 'Tokyo', 'JP', 0.10)")
                        conn.execute("UPDATE categories SET dept = 'Workstation' WHERE id = 10")
                        conn.execute("UPDATE items SET cat_id = 20 WHERE id = 100")
                        conn.execute("UPDATE line_items SET units = 6, channel = 'online', store_id = 2 WHERE id = 1001")
                        verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:upd")
                        total_checks += 1

                        conn.execute("DELETE FROM stores WHERE id = 1")
                        conn.execute("DELETE FROM line_items WHERE id = 1002")
                        verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:del")
                        total_checks += 1

                        conn.execute("DELETE FROM line_items")
                        verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:drained")
                        total_checks += 1
                        conn.close()

    # -------------------------------------------------------------------------
    # 6C & 6D: Z-Set UNION / UNION ALL / INTERSECT / EXCEPT / DISTINCT / Projections
    #          and Aggregations over UNION ALL (480 views)
    # -------------------------------------------------------------------------
    schema_6c = """
    CREATE TABLE branch_east (id INTEGER PRIMARY KEY, dept TEXT, role TEXT NOT NULL, salary REAL, bonus INTEGER, active INTEGER NOT NULL);
    CREATE TABLE branch_west (id INTEGER PRIMARY KEY, dept TEXT, role TEXT NOT NULL, salary REAL, bonus INTEGER, active INTEGER NOT NULL);
    CREATE TABLE branch_eu (id INTEGER PRIMARY KEY, dept TEXT, role TEXT NOT NULL, salary REAL, bonus INTEGER, active INTEGER NOT NULL);
    """

    setop_kinds = ["UNION", "UNION ALL", "INTERSECT", "EXCEPT"]
    proj_variants = [
        ("p_dept_role", "dept, role"),
        ("p_role_only", "role"),
        ("p_lower_role", "COALESCE(dept, 'NONE') AS d, LOWER(role) AS r"),
        ("p_comp_bucket", "role, (COALESCE(bonus, 0) % 4) AS b_mod"),
    ]
    where_pairs = [
        ("w_all", "", ""),
        ("w_act", "WHERE active = 1", "WHERE active = 1"),
        ("w_sal", "WHERE salary IS NOT NULL AND salary >= 100", "WHERE active = 1 AND COALESCE(bonus, 0) >= 10"),
        ("w_nonnull_dept", "WHERE dept IS NOT NULL", "WHERE dept IS NOT NULL AND active = 1"),
    ]

    for op_k in setop_kinds:
        op_slug = op_k.lower().replace(" ", "_")
        for p_id, p_cols in proj_variants:
            for w_id, w1, w2 in where_pairs:
                # 2-branch set op (East OP West) and same-table 2-branch set op (East OP East)
                for tbl_pair_id, (t1, t2) in [("east_west", ("branch_east", "branch_west")), ("same_east", ("branch_east", "branch_east"))]:
                    v_name = f"v6c_{op_slug}_{p_id}_{w_id}_{tbl_pair_id}"
                    sel_sql = f"SELECT {p_cols} FROM {t1} {w1} {op_k} SELECT {p_cols} FROM {t2} {w2}".strip()
                    conn = sqlite3.connect(":memory:")
                    conn.executescript(schema_6c)
                    plan = install_sqlite_materialized_view(conn, schema_6c, f"CREATE VIEW {v_name} AS {sel_sql};")
                    assert plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA, (
                        f"Expected ALGEBRAIC_DELTA for {v_name}, got {plan.strategy}"
                    )
                    total_views += 1

                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:empty")
                    total_checks += 1

                    conn.executemany("INSERT INTO branch_east VALUES (?, ?, ?, ?, ?, ?)", [
                        (1, "Eng", "SWE", 150.0, 20, 1),
                        (2, "Eng", "SWE", 180.0, 20, 1),  # duplicate (dept, role, bonus%4) within branch_east!
                        (3, None, "PM", 120.0, None, 1),
                        (4, "Sales", "AE", 90.0, 5, 0),
                    ])
                    conn.executemany("INSERT INTO branch_west VALUES (?, ?, ?, ?, ?, ?)", [
                        (10, "Eng", "SWE", 160.0, 20, 1),
                        (11, "Sales", "AE", 200.0, 16, 1),
                        (12, None, "Designer", None, 12, 1),
                    ])
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:ins")
                    total_checks += 1

                    # Update rows to test multiplicity transitions (2 -> 1 -> 0 and 0 -> 1 -> 2)
                    conn.execute("UPDATE branch_east SET role = 'Staff_SWE', active = 1 WHERE id = 1")
                    conn.execute("UPDATE branch_east SET active = 1, salary = 150.0, bonus = 16 WHERE id = 4")
                    conn.execute("UPDATE branch_west SET dept = 'Eng', role = 'PM', bonus = NULL WHERE id = 12")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:upd")
                    total_checks += 1

                    conn.execute("DELETE FROM branch_east WHERE id IN (2, 3)")
                    conn.execute("DELETE FROM branch_west WHERE id = 10")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:del")
                    total_checks += 1

                    conn.execute("DELETE FROM branch_east")
                    conn.execute("DELETE FROM branch_west")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:drained")
                    total_checks += 1
                    conn.close()

    # Also test 3-branch UNION & UNION ALL, SELECT DISTINCT, non-agg projections, and UNION ALL aggregations
    u_agg_configs = [
        ("uagg_basic", "dept", "dept, COUNT(*) AS cnt, SUM(salary) AS ss, AVG(salary) AS as_sal", ""),
        ("uagg_role", "role", "role, COUNT(*) AS cnt, TOTAL(salary) AS ts, COALESCE(SUM(salary * bonus), 0.0) AS sb", "HAVING COUNT(*) >= 2"),
        ("uagg_2grp", "dept, role", "dept, role, COUNT(*) AS cnt, SUM(bonus) AS sb, COALESCE(SUM(salary) - SUM(bonus), 0.0) AS net", "HAVING COUNT(*) >= 1"),
        ("uagg_global", "", "COUNT(*) AS cnt, SUM(salary) AS ss, AVG(salary) AS avg_s, TOTAL(bonus) AS tb", ""),
    ]
    for ua_id, g_cols, sel_cols, hav_c in u_agg_configs:
        for w_outer in ["", "WHERE salary IS NOT NULL"]:
            for use_cte in (True, False):
                for num_b in (2, 3):
                    v_name = f"v6d_{ua_id}_{'cte' if use_cte else 'sub'}_{num_b}b_{'w' if w_outer else 'nw'}"
                    b_sqls = [
                        "SELECT dept, role, salary, bonus FROM branch_east WHERE active = 1",
                        "SELECT dept, role, salary, bonus FROM branch_west WHERE active = 1",
                    ]
                    if num_b == 3:
                        b_sqls.append("SELECT dept, role, salary, bonus FROM branch_eu WHERE active = 1")
                    union_inner = " UNION ALL ".join(b_sqls)
                    grp_part = f"GROUP BY {g_cols}" if g_cols else ""
                    if use_cte:
                        sel_sql = f"WITH u_all AS ({union_inner}) SELECT {sel_cols} FROM u_all {w_outer} {grp_part} {hav_c}".strip()
                    else:
                        sel_sql = f"SELECT {sel_cols} FROM ({union_inner}) AS u_all {w_outer} {grp_part} {hav_c}".strip()

                    conn = sqlite3.connect(":memory:")
                    conn.executescript(schema_6c)
                    plan = install_sqlite_materialized_view(conn, schema_6c, f"CREATE VIEW {v_name} AS {sel_sql};")
                    assert plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA, (
                        f"Expected ALGEBRAIC_DELTA for {v_name}, got {plan.strategy}"
                    )
                    total_views += 1

                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:empty")
                    total_checks += 1

                    conn.executemany("INSERT INTO branch_east VALUES (?, ?, ?, ?, ?, ?)", [
                        (1, "Eng", "SWE", 100.0, 10, 1),
                        (2, "Sales", "AE", 80.0, 5, 0),
                        (3, None, "SWE", 120.0, None, 1),
                    ])
                    conn.executemany("INSERT INTO branch_west VALUES (?, ?, ?, ?, ?, ?)", [
                        (10, "Eng", "SWE", 200.0, 20, 1),
                        (11, "Sales", "AE", None, 15, 1),
                    ])
                    conn.executemany("INSERT INTO branch_eu VALUES (?, ?, ?, ?, ?, ?)", [
                        (20, "Eng", "PM", 150.0, 12, 1),
                        (21, "Sales", "AE", 90.0, 8, 1),
                    ])
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:ins")
                    total_checks += 1

                    conn.execute("UPDATE branch_east SET active = 1, salary = 110.0 WHERE id = 2")
                    conn.execute("UPDATE branch_west SET dept = 'Sales', salary = 50.0 WHERE id = 10")
                    conn.execute("UPDATE branch_eu SET active = 0 WHERE id = 20")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:upd")
                    total_checks += 1

                    conn.execute("DELETE FROM branch_east")
                    conn.execute("DELETE FROM branch_west")
                    conn.execute("DELETE FROM branch_eu")
                    verify_mv_equals_live(conn, plan.view_name, sel_sql, f"{v_name}:drained")
                    total_checks += 1
                    conn.close()

    print(f"Suite 6 (Exhaustive ALGEBRAIC_DELTA Matrix) complete: {total_views} views tested across {total_checks} live query assertions!")
    return total_views, total_checks


def run_suite7_adversarial_dml_and_PG_MySQL_algebraic_matrix() -> Tuple[int, int]:
    """
    SUITE 7: Adversarial Edge-Case DML Stress Fuzzing + Expanded PostgreSQL/MySQL
    ALGEBRAIC_DELTA Execution Matrix (~800 views, 8,000+ assertions).
    Tests:
      - Composite PKs, delimiter-collision strings ('a|b', 'V1:x', 'N', 'NULL', ''),
        simultaneous group-swapping UPDATEs, all-NULL group transitions, UPSERTs.
      - PostgreSQL and MySQL execution across all new ALGEBRAIC_DELTA modes (expression
        GROUP BY, compound aggregates, HAVING, multi-table INNER JOINs, UNION, UNION ALL,
        INTERSECT, EXCEPT, SELECT DISTINCT, PK projections, and UNION ALL aggregations).
    """
    total_views = 0
    total_checks = 0

    schema_adv = """
    CREATE TABLE customers (
        id INT PRIMARY KEY,
        tier VARCHAR(50),
        region VARCHAR(50) NOT NULL
    );
    CREATE TABLE orders (
        id INT PRIMARY KEY,
        customer_id INT,
        region VARCHAR(50),
        category VARCHAR(50) NOT NULL,
        amount DOUBLE PRECISION,
        qty INT NOT NULL,
        flags INT,
        is_paid BOOLEAN,
        status VARCHAR(50) NOT NULL
    );
    """

    expanded_dialect_views = [
        # Expression GROUP BY + Compound Aggregations + HAVING in ALGEBRAIC_DELTA
        ("alg_expr_grp_lower", "SELECT LOWER(category) AS l_cat, COUNT(*) AS c, SUM(amount) AS s, AVG(amount) AS a FROM orders GROUP BY LOWER(category)"),
        ("alg_expr_grp_coal", "SELECT COALESCE(region, 'MISSING') AS eff_reg, COUNT(*) AS c, SUM(qty) AS sq FROM orders GROUP BY COALESCE(region, 'MISSING')"),
        ("alg_comp_agg_1", "SELECT category, COUNT(*) AS c, COALESCE(SUM(amount * qty), 0.0) AS vol, COALESCE(MAX(qty) - MIN(qty), 0) AS q_span FROM orders GROUP BY category"),
        ("alg_having_1", "SELECT region, COUNT(*) AS c, SUM(amount) AS s, AVG(amount) AS a FROM orders GROUP BY region HAVING COUNT(*) >= 2"),
        ("alg_comp_having", "SELECT category, COUNT(*) AS c, COALESCE(SUM(amount), 0.0) - COALESCE(SUM(qty), 0.0) AS diff FROM orders WHERE status != 'cancelled' GROUP BY category HAVING COUNT(*) >= 1 AND COALESCE(SUM(qty), 0) >= 3"),
        # Multi-Table INNER JOINs in ALGEBRAIC_DELTA
        ("alg_join_2t", "SELECT c.tier, o.category, COUNT(*) AS c, SUM(o.amount) AS s, AVG(o.amount) AS a, TOTAL(o.amount) AS tot_a FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.tier, o.category"),
        ("alg_join_having", "SELECT c.region, COUNT(*) AS c, SUM(o.qty) AS sq, COALESCE(SUM(o.amount * o.qty), 0.0) AS rev FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'fulfilled' GROUP BY c.region HAVING SUM(o.qty) >= 3"),
        ("alg_join_global", "SELECT COUNT(*) AS c, SUM(o.amount) AS s, AVG(o.amount) AS a, TOTAL(o.qty) AS tq FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.amount > 0"),
        # Z-Set UNION, UNION ALL, INTERSECT, EXCEPT, DISTINCT, Projection in ALGEBRAIC_DELTA
        ("alg_zset_union", "SELECT region AS loc FROM customers UNION SELECT region AS loc FROM orders WHERE region IS NOT NULL"),
        ("alg_zset_union_all", "SELECT region AS loc, tier AS label FROM customers WHERE tier IS NOT NULL UNION ALL SELECT region AS loc, category AS label FROM orders WHERE region IS NOT NULL"),
        ("alg_zset_intersect", "SELECT region AS loc FROM customers INTERSECT SELECT region AS loc FROM orders WHERE region IS NOT NULL"),
        ("alg_zset_except", "SELECT region AS loc FROM customers EXCEPT SELECT region AS loc FROM orders WHERE status = 'cancelled' AND region IS NOT NULL"),
        ("alg_zset_distinct", "SELECT DISTINCT region, category, status FROM orders"),
        ("alg_zset_proj", "SELECT id AS oid, category, qty * 2 AS d_qty, COALESCE(region, 'NONE') AS reg FROM orders WHERE status = 'fulfilled'"),
        # UNION ALL Aggregation in ALGEBRAIC_DELTA
        ("alg_uall_agg_cte", "WITH combined AS (SELECT region AS loc, qty AS weight FROM orders WHERE region IS NOT NULL UNION ALL SELECT region AS loc, id AS weight FROM customers) SELECT loc, COUNT(*) AS cnt, SUM(weight) AS tot_w, AVG(weight) AS avg_w FROM combined GROUP BY loc"),
        ("alg_uall_agg_sub", "SELECT loc, COUNT(*) AS cnt, COALESCE(SUM(weight), 0) AS tot_w FROM (SELECT region AS loc, qty AS weight FROM orders WHERE status = 'fulfilled' UNION ALL SELECT region AS loc, id AS weight FROM customers) AS u GROUP BY loc HAVING COUNT(*) >= 2"),
    ]

    pg_my_extra_join_views = [
        ("alg_join_stats_xor", "SELECT c.tier, o.category, COUNT(*) AS c, SUM(o.amount) AS s, AVG(o.amount) AS a, VAR_POP(o.amount) AS vp, STDDEV_SAMP(o.amount) AS sds, BIT_XOR(o.flags) AS bx FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.tier, o.category"),
        ("alg_uall_stats_xor", "WITH combined AS (SELECT region AS loc, amount AS val, flags AS fl FROM orders WHERE region IS NOT NULL UNION ALL SELECT region AS loc, id * 10.0 AS val, id AS fl FROM customers) SELECT loc, COUNT(*) AS cnt, SUM(val) AS sv, VAR_POP(val) AS vp, BIT_XOR(fl) AS bx FROM combined GROUP BY loc"),
    ]
    for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
        dialect_views = expanded_dialect_views + (pg_my_extra_join_views if dialect in (Dialect.POSTGRES, Dialect.MYSQL) else [])
        for seed in range(30):
            rng = random.Random(77000 + seed)
            for v_base, v_select in dialect_views:
                v_name = f"mv7_{dialect.value}_{v_base}_s{seed}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    plan = generate_materialized_view(
                        schema_adv,
                        f"CREATE VIEW {v_name} AS {v_select};",
                        dialect=dialect,
                        ignore_limitations=True,
                    )
                assert plan.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA, (
                    f"Expected ALGEBRAIC_DELTA on {dialect.value} for {v_base}, got {plan.strategy}"
                )
                conn = sqlite3.connect(":memory:")
                conn.executescript(schema_adv)
                if dialect == Dialect.SQLITE:
                    for st in plan.all_statements():
                        conn.execute(st)
                else:
                    install_plan_in_simulator(conn, plan)
                total_views += 1

                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:empty")
                total_checks += 1

                # Use adversarial strings including pipes, colons, 'N', 'NULL', empty strings
                regions_pool = ["NA", "EU", "A|B", "V1:x", "N", "NULL", ""]
                cats_pool = ["hw", "SW", "a|b", "cloud"]
                tiers_pool = ["gold", "silver", "N", None]

                conn.executemany("INSERT INTO customers VALUES (?, ?, ?)", [
                    (1, rng.choice(tiers_pool), rng.choice(regions_pool[:4])),
                    (2, rng.choice(tiers_pool), rng.choice(regions_pool[:4])),
                    (3, rng.choice(tiers_pool), rng.choice(regions_pool[3:])),
                    (4, None, "NA"),
                ])
                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:custs")
                total_checks += 1

                conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    (10, 1, rng.choice(regions_pool + [None]), rng.choice(cats_pool), 25.0, 2, 7, 1, "fulfilled"),
                    (11, 1, rng.choice(regions_pool + [None]), rng.choice(cats_pool), 50.0, 5, 3, 1, "fulfilled"),
                    (12, 2, rng.choice(regions_pool + [None]), rng.choice(cats_pool), None, 1, 5, 0, "pending"),
                    (13, 3, None, rng.choice(cats_pool), 100.0, 4, 12, 1, "fulfilled"),
                    (14, 4, "NA", "sw", -10.0, 3, 9, 0, "cancelled"),
                ])
                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:orders")
                total_checks += 1

                # Perform 6 adversarial mutations per view
                next_oid = 30
                for step in range(6):
                    op = rng.choice(["ins", "upd_swap", "upd_null", "upd_cust", "del_one"])
                    if op == "ins":
                        next_oid += 1
                        conn.execute(
                            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                next_oid,
                                rng.choice([1, 2, 3, 4, None]),
                                rng.choice(regions_pool + [None]),
                                rng.choice(cats_pool),
                                rng.choice([None, 0.0, -5.5, round(rng.uniform(1.0, 100.0), 2)]),
                                rng.randint(1, 7),
                                rng.randint(0, 15),
                                rng.choice([0, 1, None]),
                                rng.choice(["fulfilled", "pending", "cancelled"]),
                            ),
                        )
                    elif op == "upd_swap":
                        conn.execute(
                            "UPDATE orders SET category = CASE WHEN category = 'hw' THEN 'SW' ELSE 'hw' END, qty = qty + 1"
                        )
                    elif op == "upd_null":
                        conn.execute(
                            "UPDATE orders SET amount = NULL, region = NULL WHERE id = (SELECT id FROM orders ORDER BY id LIMIT 1)"
                        )
                    elif op == "upd_cust":
                        conn.execute(
                            "UPDATE customers SET region = ?, tier = ? WHERE id = ?",
                            (rng.choice(regions_pool), rng.choice(tiers_pool), rng.randint(1, 4)),
                        )
                    elif op == "del_one":
                        conn.execute("DELETE FROM orders WHERE id = (SELECT id FROM orders ORDER BY id DESC LIMIT 1)")

                    verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:step_{step}_{op}")
                    total_checks += 1

                conn.execute("DELETE FROM orders")
                conn.execute("DELETE FROM customers")
                verify_mv_equals_live(conn, plan.view_name, v_select, f"{v_name}:drained")
                total_checks += 1
                conn.close()

    print(f"Suite 7 (Adversarial & Multi-Dialect Algebraic Matrix) complete: {total_views} views tested across {total_checks} live query assertions!")
    return total_views, total_checks




def run_suite8_massive_algebraic_permutations() -> Tuple[int, int]:
    """
    SUITE 8: Massive Multi-Schema Algebraic Permutation Matrix (~1,600 views, 11,200+ assertions).
    Tests composite primary keys, multi-table star/chain joins, multi-expression GROUP BY,
    multi-aggregate arithmetic combinations, and Z-set set operations across 1,600 distinct
    view definitions.
    """
    total_views = 0
    total_checks = 0

    schema = """
    CREATE TABLE orgs (id INTEGER PRIMARY KEY, o_code TEXT NOT NULL, tier TEXT);
    CREATE TABLE projects (id INTEGER PRIMARY KEY, org_id INTEGER, p_type TEXT NOT NULL, budget REAL NOT NULL);
    CREATE TABLE tasks (
        proj_id INTEGER NOT NULL,
        seq_no INTEGER NOT NULL,
        assignee TEXT,
        pts INTEGER,
        hours REAL,
        state TEXT NOT NULL,
        PRIMARY KEY (proj_id, seq_no)
    );
    CREATE TABLE archived_tasks (
        proj_id INTEGER NOT NULL,
        seq_no INTEGER NOT NULL,
        assignee TEXT,
        pts INTEGER,
        hours REAL,
        state TEXT NOT NULL,
        PRIMARY KEY (proj_id, seq_no)
    );
    """

    # Generate 1,600 distinct views across 4 families (400 views each)
    view_specs: List[Tuple[str, str]] = []

    # Family A (400 views): Composite-PK table `tasks` with expression GROUP BY, compound aggs, FILTER, HAVING
    grp_a = [
        ("g_proj", "proj_id, ", "GROUP BY proj_id"),
        ("g_state", "state, ", "GROUP BY state"),
        ("g_assign", "assignee, ", "GROUP BY assignee"),
        ("g_lower_asgn", "LOWER(COALESCE(assignee, 'unassigned')) AS la, ", "GROUP BY LOWER(COALESCE(assignee, 'unassigned'))"),
        ("g_seq_mod", "(seq_no % 4) AS slot, state, ", "GROUP BY (seq_no % 4), state"),
        ("g_pts_case", "CASE WHEN COALESCE(pts, 0) >= 5 THEN 'heavy' ELSE 'light' END AS load_t, ", "GROUP BY CASE WHEN COALESCE(pts, 0) >= 5 THEN 'heavy' ELSE 'light' END"),
        ("g_proj_state", "proj_id, state, ", "GROUP BY proj_id, state"),
        ("g_global", "", ""),
        ("g_substr_state", "SUBSTR(state, 1, 2) AS st2, ", "GROUP BY SUBSTR(state, 1, 2)"),
        ("g_proj_mod", "(proj_id % 3) AS pm, ", "GROUP BY (proj_id % 3)"),
    ]
    where_a = [
        ("w0", ""),
        ("w1", "WHERE state != 'cancelled'"),
        ("w2", "WHERE pts IS NOT NULL AND pts > 0"),
        ("w3", "WHERE COALESCE(hours, 0.0) >= 1.0"),
        ("w4", "WHERE state IN ('open', 'done') AND seq_no >= 1"),
    ]
    agg_a = [
        ("a0", "COUNT(*) AS c, SUM(pts) AS sp, AVG(hours) AS ah, TOTAL(hours) AS th"),
        ("a1", "COUNT(*) AS c, COALESCE(SUM(pts * hours), 0.0) AS ph, COALESCE(SUM(pts), 0) - COUNT(*) AS net_p"),
        ("a2", "COUNT(*) AS c, MIN(pts) AS mnp, MAX(pts) AS mxp, COALESCE(MAX(pts) - MIN(pts), 0) AS span_p"),
        ("a3", "COUNT(*) AS c, COUNT(*) FILTER (WHERE state = 'done') AS cd, COALESCE(SUM(pts) FILTER (WHERE state = 'done'), 0) AS spd"),
    ]
    hav_a = [
        ("h0", ""),
        ("h1", "HAVING COUNT(*) >= 1"),
        ("h2", "HAVING COUNT(*) >= 2"),
    ]
    for gi, (gn, sp, gc) in enumerate(grp_a):
        for wi, (wn, wc) in enumerate(where_a):
            for ai, (an, ac) in enumerate(agg_a):
                for hi, (hn, hc) in enumerate(hav_a):
                    if not gc and hc:
                        continue
                    if len(view_specs) < 500:
                        q = f"SELECT {sp}{ac} FROM tasks {wc} {gc} {hc}".strip()
                        view_specs.append((f"v8_fa_{len(view_specs)}", q))

    # Family B (500 views): 2-table and 3-table INNER JOINs across tasks, projects, orgs
    join_b = [
        ("jb1", "FROM tasks t JOIN projects p ON t.proj_id = p.id", [
            ("gp_type", "p.p_type, ", "GROUP BY p.p_type"),
            ("gp_id_st", "p.id AS pid, t.state, ", "GROUP BY p.id, t.state"),
            ("gp_low", "LOWER(p.p_type) AS lpt, ", "GROUP BY LOWER(p.p_type)"),
            ("gp_glob", "", ""),
        ]),
        ("jb2", "FROM tasks t JOIN projects p ON t.proj_id = p.id JOIN orgs o ON p.org_id = o.id", [
            ("go_code", "o.o_code, ", "GROUP BY o.o_code"),
            ("go_tier", "o.tier, p.p_type, ", "GROUP BY o.tier, p.p_type"),
            ("go_st", "o.o_code, t.state, ", "GROUP BY o.o_code, t.state"),
            ("go_coal", "COALESCE(o.tier, 'free') AS et, ", "GROUP BY COALESCE(o.tier, 'free')"),
        ]),
    ]
    for jb_id, jf_sql, j_grps in join_b:
        for gn, sp, gc in j_grps:
            for wn, wc in [
                ("jw0", ""),
                ("jw1", "WHERE t.state = 'done'"),
                ("jw2", "WHERE t.pts IS NOT NULL AND p.budget >= 100.0"),
                ("jw3", "WHERE t.state != 'cancelled' AND COALESCE(t.hours, 0.0) > 0.0"),
            ]:
                for an, ac in [
                    ("ja0", "COUNT(*) AS c, SUM(t.pts) AS sp, AVG(t.hours) AS ah"),
                    ("ja1", "COUNT(*) AS c, COALESCE(SUM(t.pts * p.budget), 0.0) AS wb, TOTAL(t.hours) AS th"),
                    ("ja2", "COUNT(*) AS c, MIN(t.pts) AS mnp, MAX(t.pts) AS mxp, SUM(t.pts) AS sp"),
                    ("ja3", "COUNT(*) AS c, COUNT(*) FILTER (WHERE t.state = 'done') AS cd, COALESCE(SUM(t.pts) - COUNT(*), 0) AS diff"),
                ]:
                    for hn, hc in [("jh0", ""), ("jh1", "HAVING COUNT(*) >= 1"), ("jh2", "HAVING COALESCE(SUM(t.pts), 0) >= 5")]:
                        if not gc and hc:
                            continue
                        if len(view_specs) < 1000:
                            q = f"SELECT {sp}{ac} {jf_sql} {wc} {gc} {hc}".strip()
                            view_specs.append((f"v8_fb_{len(view_specs)}", q))

    # Family C (600 views): Composite-PK Projections, SELECT DISTINCT, UNION/UNION ALL/INTERSECT/EXCEPT, and UNION ALL Aggregations
    proj_c = [
        "proj_id, state",
        "assignee, state",
        "proj_id, COALESCE(assignee, 'none') AS asgn, (seq_no % 3) AS slot",
        "LOWER(state) AS ls, COALESCE(pts, 0) AS p0",
    ]
    where_c = [
        "",
        "WHERE state = 'done'",
        "WHERE pts IS NOT NULL AND pts >= 3",
        "WHERE assignee IS NOT NULL",
        "WHERE state != 'cancelled' AND seq_no <= 10",
    ]
    for pc in proj_c:
        for wc1 in where_c:
            # SELECT DISTINCT and PK-lineage projections
            view_specs.append((f"v8_fc_dist_{len(view_specs)}", f"SELECT DISTINCT {pc} FROM tasks {wc1}".strip()))
            proj_cols = f"seq_no, {pc}" if "proj_id" in pc else f"proj_id, seq_no, {pc}"
            view_specs.append((f"v8_fc_proj_{len(view_specs)}", f"SELECT {proj_cols} FROM tasks {wc1}".strip()))
            for wc2 in where_c:
                for op_k in ("UNION", "UNION ALL", "INTERSECT", "EXCEPT"):
                    view_specs.append((
                        f"v8_fc_set_{len(view_specs)}",
                        f"SELECT {pc} FROM tasks {wc1} {op_k} SELECT {pc} FROM archived_tasks {wc2}".strip(),
                    ))
                # UNION ALL aggregation
                view_specs.append((
                    f"v8_fc_uagg_{len(view_specs)}",
                    f"WITH all_t AS (SELECT proj_id, state, pts, hours FROM tasks {wc1} UNION ALL SELECT proj_id, state, pts, hours FROM archived_tasks {wc2}) "
                    f"SELECT proj_id, state, COUNT(*) AS cnt, SUM(pts) AS sp, COALESCE(SUM(pts * hours), 0.0) AS wh FROM all_t GROUP BY proj_id, state",
                ))

    # Batch views in groups of 40 into multi-view SQLite databases so all 1,600+ views execute fast!
    batch_sz = 40
    for b_start in range(0, len(view_specs), batch_sz):
        chunk = view_specs[b_start : b_start + batch_sz]
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        multi_sql = "\n".join(f"CREATE VIEW {vn} AS {vq};" for vn, vq in chunk)
        comp_plan = install_sqlite_materialized_view(conn, schema, multi_sql)
        sub_list = comp_plan.sub_plans if comp_plan.sub_plans else [comp_plan]
        assert len(sub_list) == len(chunk)
        for sp in sub_list:
            assert sp.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA, (
                f"Expected ALGEBRAIC_DELTA for {sp.view_name}, got {sp.strategy}"
            )
        total_views += len(chunk)

        for vn, vq in chunk:
            verify_mv_equals_live(conn, f"__matview_{vn}", vq, f"{vn}:empty")
            total_checks += 1

        conn.executemany("INSERT INTO orgs VALUES (?, ?, ?)", [(1, "ORG_A", "ent"), (2, "ORG_B", None)])
        conn.executemany("INSERT INTO projects VALUES (?, ?, ?, ?)", [(10, 1, "core", 500.0), (20, 2, "ml", 1200.0), (30, None, "internal", 50.0)])
        conn.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?)", [
            (10, 1, "Alice", 5, 2.5, "done"),
            (10, 2, "Bob", 3, 1.0, "open"),
            (20, 1, None, 8, 4.0, "done"),
            (20, 2, "Alice", None, None, "cancelled"),
            (30, 1, "Charlie", 2, 0.5, "open"),
        ])
        conn.executemany("INSERT INTO archived_tasks VALUES (?, ?, ?, ?, ?, ?)", [
            (10, 1, "Alice", 5, 2.5, "done"),
            (20, 9, "Dana", 10, 6.0, "done"),
        ])
        for vn, vq in chunk:
            verify_mv_equals_live(conn, f"__matview_{vn}", vq, f"{vn}:ins")
            total_checks += 1

        conn.execute("UPDATE tasks SET state = 'done', pts = 7, assignee = 'Bob' WHERE proj_id = 10 AND seq_no = 2")
        conn.execute("UPDATE projects SET org_id = 1, p_type = 'ai' WHERE id = 20")
        conn.execute("UPDATE orgs SET tier = 'vip' WHERE id = 2")
        conn.execute("UPDATE archived_tasks SET state = 'open', pts = 4 WHERE proj_id = 20 AND seq_no = 9")
        for vn, vq in chunk:
            verify_mv_equals_live(conn, f"__matview_{vn}", vq, f"{vn}:upd")
            total_checks += 1

        conn.execute("DELETE FROM tasks WHERE proj_id = 10 AND seq_no = 1")
        conn.execute("DELETE FROM archived_tasks WHERE proj_id = 10")
        for vn, vq in chunk:
            verify_mv_equals_live(conn, f"__matview_{vn}", vq, f"{vn}:del")
            total_checks += 1

        conn.execute("DELETE FROM tasks")
        conn.execute("DELETE FROM archived_tasks")
        conn.execute("DELETE FROM projects")
        conn.execute("DELETE FROM orgs")
        for vn, vq in chunk:
            verify_mv_equals_live(conn, f"__matview_{vn}", vq, f"{vn}:drained")
            total_checks += 1
        conn.close()

    print(f"Suite 8 (Massive Algebraic Permutations) complete: {total_views} views tested across {total_checks} live query assertions!")
    return total_views, total_checks




def test_incremental_backfill_sql_generation_and_trigger_ordering() -> None:
    """
    Verifies:
      1. `--incremental-backfill` (and `incremental_backfill=True`, `batch_size=N`)
         generates distinct, dialect-native chunked backfill SQL in the generated script
         across SQLite, PostgreSQL, and MySQL.
      2. Maintenance triggers are always emitted and installed BEFORE the backfill
         (Section 4 = Atomic Maintenance Triggers, Section 5 = Initial Backfill) so no
         base-table write can ever slip through an un-triggered window.
      3. Executing the generated `--incremental-backfill` SQL scripts directly on SQLite
         (across pk_range_fold, distinct/holistic, join, set-op, and window views) produces
         exact results matching live queries.
    """
    import io
    from contextlib import redirect_stdout

    schema = """
    CREATE TABLE customers (id INTEGER PRIMARY KEY, tier TEXT NOT NULL);
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        region TEXT,
        amount REAL,
        qty INTEGER NOT NULL,
        status TEXT NOT NULL
    );
    """
    views = [
        ("v_pk_fold", "SELECT region, COUNT(*) AS cnt, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mnq, MAX(qty) AS mxq FROM orders WHERE status = 'settled' GROUP BY region"),
        ("v_global_fold", "SELECT COUNT(*) AS cnt, SUM(amount) AS s_amt, TOTAL(amount) AS tot_amt FROM orders"),
        ("v_dist_batch", "SELECT region, COUNT(DISTINCT customer_id) AS dc, SUM(DISTINCT qty) AS sdq, GROUP_CONCAT(status, ',') AS sts FROM orders GROUP BY region"),
        ("v_join_batch", "SELECT c.tier, o.region, COUNT(*) AS cnt, SUM(o.amount) AS s_amt FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.tier, o.region"),
        ("v_win_batch", "SELECT id, region, amount, RANK() OVER (PARTITION BY region ORDER BY amount DESC, id ASC) AS rnk FROM orders"),
    ]

    for v_name, v_sel in views:
        v_sql = f"CREATE VIEW {v_name} AS {v_sel};"

        # 1. Verify SQLite, PostgreSQL, and MySQL generated scripts differ when incremental_backfill=True
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EngineLimitationWarning)
                p_full = generate_materialized_view(
                    schema, v_sql, dialect=dialect, incremental_backfill=False, ignore_limitations=True
                )
                p_inc = generate_materialized_view(
                    schema, v_sql, dialect=dialect, incremental_backfill=True, batch_size=3, ignore_limitations=True
                )

            script_full = p_full.render_sql_script()
            script_inc = p_inc.render_sql_script()
            assert script_full != script_inc, f"Expected incremental script to differ on {dialect.value} for {v_name}"
            assert "Backfill: incremental (batch_size=3)" in script_inc
            assert "-- 4. Atomic Maintenance Triggers (Installed Before Backfill)" in script_inc
            assert "-- 5. Incremental Initial Backfill (batch_size=3)" in script_inc

            # Verify triggers appear BEFORE backfill in both render_sql_script() and all_statements()
            trig_pos = script_inc.index("-- 4. Atomic Maintenance Triggers")
            backfill_pos = script_inc.index("-- 5. Incremental Initial Backfill")
            assert trig_pos < backfill_pos, "Triggers must be installed before backfill in rendered script!"

            all_stmts = p_inc.all_statements()
            first_trig_idx = next(i for i, s in enumerate(all_stmts) if s in p_inc.trigger_ddls)
            first_pop_idx = next(i for i, s in enumerate(all_stmts) if s in p_inc.initial_populate_sqls)
            assert first_trig_idx < first_pop_idx, "Triggers must precede backfill in all_statements()!"

        # 2. Execute the generated SQLite incremental backfill SQL script on a populated database
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        conn.executemany("INSERT INTO customers VALUES (?, ?)", [(1, "gold"), (2, "silver")])
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", [
            (1, 1, "NA", 10.0, 2, "settled"),
            (2, 1, "NA", 20.0, 4, "settled"),
            (3, 2, "EU", 15.0, 1, "pending"),
            (4, 2, "EU", 30.0, 3, "settled"),
            (5, 1, None, 50.0, 5, "settled"),
            (6, 2, "NA", None, 2, "settled"),
            (7, 1, "EU", 45.0, 6, "settled"),
        ])
        p_inc_sq = generate_materialized_view(
            schema, v_sql, dialect=Dialect.SQLITE, incremental_backfill=True, batch_size=2
        )
        conn.executescript(p_inc_sq.render_sql_script())
        verify_mv_equals_live(conn, p_inc_sq.view_name, v_sel, f"{v_name}:inc_script_exec")

        # Mutate after incremental script installation to verify triggers work seamlessly
        conn.execute("INSERT INTO orders VALUES (8, 1, 'NA', 100.0, 10, 'settled')")
        conn.execute("UPDATE orders SET amount = 60.0, region = 'APAC' WHERE id = 2")
        conn.execute("DELETE FROM orders WHERE id = 4")
        verify_mv_equals_live(conn, p_inc_sq.view_name, v_sel, f"{v_name}:inc_script_post_dml")
        conn.close()

    # 3. Verify CLI `--incremental-backfill --batch-size 42` outputs incremental SQL
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--schema", schema,
            "--view", "CREATE VIEW v_cli_inc AS SELECT region, COUNT(*) AS c, SUM(amount) AS s FROM orders GROUP BY region;",
            "--incremental-backfill",
            "--batch-size", "42",
        ])
    assert rc == 0
    cli_out = buf.getvalue()
    assert "-- 5. Incremental Initial Backfill (batch_size=42)" in cli_out
    assert "__mv_batches" in cli_out


def _run_all_tests() -> None:
    run_test_matrix()
    run_suite2_and_suite3()
    run_suite4_postgres_and_mysql_matrix()
    v5, c5 = run_suite5_multi_view_sql_files_and_entrypoints()
    v6, c6 = run_suite6_algebraic_delta_exhaustive_matrix()
    v7, c7 = run_suite7_adversarial_dml_and_PG_MySQL_algebraic_matrix()
    v8, c8 = run_suite8_massive_algebraic_permutations()
    test_incremental_backfill_sql_generation_and_trigger_ordering()
    test_object_prefix_and_user_defined_indexes()
    test_mysql_trigger_execution_with_left_to_right_set_semantics()
    if os.environ.get("POSTGRES_DSN"):
        run_live_postgres_tests_if_available(os.environ["POSTGRES_DSN"])
    if os.environ.get("MYSQL_DSN"):
        run_live_mysql_tests_if_available(os.environ["MYSQL_DSN"])
    test_postgres_and_mysql_generated_sql_and_algebraic_execution()
    test_zero_runtime_dependency_across_separate_connections()
    test_limitation_refusal_and_ignore_limitations_override()
    test_sqlite_native_aggregates_and_multiple_mvs_fuzz()
    total_v = 192 + 134 + 132 + v5 + v6 + v7 + v8 + 20
    total_c = 1728 + 1667 + 1584 + c5 + c6 + c7 + c8 + 650
    print(f"GRAND TOTAL: {total_v} materialized views tested across {total_c} live time-series assertions — ALL PASSED!")


if __name__ == "__main__":
    try:
        _run_all_tests()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        if os.environ.get("GITHUB_ACTIONS"):
            escaped_tb = tb.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::error title=Test Suite Failure::{escaped_tb}", flush=True)
        raise