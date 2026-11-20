#!/usr/bin/env python3
"""Tests for Tier 2 group-scoped maintenance: window partition scoping and outer-join group scoping."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import unittest

import sqlalchemy as sa

import matview_sync as ms


def _engine_url_for_dialect(dsn: str, dialect: ms.Dialect):
    """Choose an installed DB-API driver when a DSN leaves it implicit."""
    url = sa.engine.make_url(dsn)
    if dialect == ms.Dialect.POSTGRES and url.drivername in {"postgres", "postgresql"}:
        # SQLAlchemy 2.1 defaults bare PostgreSQL URLs to psycopg (v3), but
        # requirements.txt installs psycopg2-binary instead.
        if importlib.util.find_spec("psycopg2") is not None:
            return url.set(drivername="postgresql+psycopg2")
        if importlib.util.find_spec("psycopg") is not None:
            return url.set(drivername="postgresql+psycopg")
    if dialect == ms.Dialect.MYSQL and url.drivername == "mysql":
        # requirements.txt installs PyMySQL, not mysqlclient/MySQLdb.
        return url.set(drivername="mysql+pymysql")
    return url


class TestTier2WindowPartitionScoping(unittest.TestCase):
    SCHEMA = """
    CREATE TABLE scores (
        id INTEGER PRIMARY KEY,
        team TEXT,
        role TEXT NOT NULL,
        score INTEGER NOT NULL
    );
    CREATE TABLE teams (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    CREATE TABLE members (
        id INTEGER PRIMARY KEY,
        team_id INTEGER NOT NULL,
        rating INTEGER NOT NULL
    );
    """

    def test_strategy_selection_partitioned_vs_unpartitioned_and_disjoint(self) -> None:
        p_part = ms.generate_materialized_view(
            self.SCHEMA,
            "CREATE VIEW mv_ranked AS SELECT id, team, ROW_NUMBER() OVER (PARTITION BY team ORDER BY score DESC, id ASC) AS rn FROM scores;",
            dialect=ms.Dialect.SQLITE,
        )
        self.assertEqual(p_part.strategy, ms.MaintenanceStrategy.GROUP_SCOPED_DELTA)
        self.assertEqual(p_part.diagnostics.algebraic_mode, "window_partition")
        self.assertEqual(p_part.diagnostics.tier, 2)
        # Partition-scoped triggers do NOT unconditionally wipe backing store on INSERT/UPDATE/DELETE
        ai_trig = [t for t in p_part.trigger_ddls if "_ai" in t][0]
        self.assertIn("WHERE", ai_trig)
        self.assertNotIn("DELETE FROM __matview_store_mv_ranked;", ai_trig)

        # Nested compatible partitions: (team) & (team, role) -> common partition key is (team)
        p_nested = ms.generate_materialized_view(
            self.SCHEMA,
            """
            CREATE VIEW mv_nested AS
            SELECT id, team, role,
                   RANK() OVER (PARTITION BY team ORDER BY score DESC) AS team_rk,
                   DENSE_RANK() OVER (PARTITION BY team, role ORDER BY score DESC) AS role_drk
            FROM scores;
            """,
            dialect=ms.Dialect.SQLITE,
        )
        self.assertEqual(p_nested.strategy, ms.MaintenanceStrategy.GROUP_SCOPED_DELTA)

        # Disjoint partitions must safely fall back to Tier 3
        p_disjoint = ms.generate_materialized_view(
            self.SCHEMA,
            """
            CREATE VIEW mv_disjoint AS
            SELECT id,
                   ROW_NUMBER() OVER (PARTITION BY team ORDER BY id) AS r1,
                   ROW_NUMBER() OVER (PARTITION BY role ORDER BY id) AS r2
            FROM scores;
            """,
            dialect=ms.Dialect.SQLITE,
        )
        self.assertEqual(p_disjoint.strategy, ms.MaintenanceStrategy.ATOMIC_QUERY_SYNC)

        # Global unpartitioned window must fall back to Tier 3
        p_global = ms.generate_materialized_view(
            self.SCHEMA,
            "CREATE VIEW mv_global AS SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC, id ASC) AS rn FROM scores;",
            dialect=ms.Dialect.SQLITE,
        )
        self.assertEqual(p_global.strategy, ms.MaintenanceStrategy.ATOMIC_QUERY_SYNC)

        # PostgreSQL statement-level codegen stays scoped to __mv_new_rows / __mv_old_rows
        p_pg = ms.generate_materialized_view(
            self.SCHEMA,
            "CREATE VIEW mv_pg_win AS SELECT id, team, LAG(score, 1) OVER (PARTITION BY team ORDER BY id) AS prev_s FROM scores;",
            dialect=ms.Dialect.POSTGRES,
        )
        self.assertEqual(p_pg.strategy, ms.MaintenanceStrategy.GROUP_SCOPED_DELTA)
        joined_pg = "\n".join(p_pg.trigger_ddls)
        self.assertIn("__mv_new_rows", joined_pg)
        self.assertIn("__mv_old_rows", joined_pg)

    def test_live_sqlite_partitioned_windows_with_null_unprojected_and_lag_lead(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(self.SCHEMA)
            conn.executescript(
                """
                INSERT INTO scores VALUES
                    (1, 'alpha', 'eng', 100),
                    (2, 'alpha', 'eng', 80),
                    (3, 'beta',  'ops', 90),
                    (4, NULL,    'ops', 50);
                """
            )
            # View 1: Unprojected partition key (hidden __mv_grp_0) + ROW_NUMBER/RANK/LAG/LEAD/running SUM
            v1_select = """
                SELECT id, score,
                       ROW_NUMBER() OVER (PARTITION BY team ORDER BY score DESC, id ASC) AS rn,
                       RANK() OVER (PARTITION BY team ORDER BY score DESC) AS rk,
                       DENSE_RANK() OVER (PARTITION BY team ORDER BY score DESC) AS drk,
                       COALESCE(LAG(score, 1) OVER (PARTITION BY team ORDER BY id ASC), -1) AS prev_s,
                       COALESCE(LEAD(score, 1) OVER (PARTITION BY team ORDER BY id ASC), -1) AS next_s,
                       SUM(score) OVER (PARTITION BY team ORDER BY id ASC) AS run_s
                FROM scores
            """
            # View 2: Inner join with partitioned window
            v2_select = """
                SELECT m.id AS member_id, t.name AS team_name, m.rating,
                       ROW_NUMBER() OVER (PARTITION BY t.name ORDER BY m.rating DESC, m.id ASC) AS team_rn
                FROM members m
                INNER JOIN teams t ON m.team_id = t.id
            """
            conn.executescript(
                """
                INSERT INTO teams VALUES (1, 'red'), (2, 'blue');
                INSERT INTO members VALUES (10, 1, 50), (11, 1, 70), (12, 2, 40);
                """
            )

            mgr = ms.MaterializedViewManager(conn, dialect=ms.Dialect.SQLITE)
            p1 = mgr.register_view(f"CREATE VIEW mv_scores_win AS {v1_select};")
            p2 = mgr.register_view(f"CREATE VIEW mv_members_win AS {v2_select};")
            self.assertEqual(p1.strategy, ms.MaintenanceStrategy.GROUP_SCOPED_DELTA)
            self.assertEqual(p2.strategy, ms.MaintenanceStrategy.GROUP_SCOPED_DELTA)

            def assert_equiv() -> None:
                for mv_name, raw_sel in (("mv_scores_win", v1_select), ("mv_members_win", v2_select)):
                    mv_rows = sorted(conn.execute(f"SELECT * FROM {mv_name};").fetchall())
                    live_rows = sorted(conn.execute(f"{raw_sel};").fetchall())
                    self.assertEqual(mv_rows, live_rows, f"Mismatch on {mv_name}")

            assert_equiv()
            dml_steps = [
                "INSERT INTO scores VALUES (5, 'alpha', 'ops', 100), (6, NULL, 'eng', 70);",
                "UPDATE scores SET score = 120 WHERE id = 2;",
                # Move row across partitions (including NULL -> non-NULL and non-NULL -> NULL)
                "UPDATE scores SET team = 'beta' WHERE id = 4;",
                "UPDATE scores SET team = NULL WHERE id = 1;",
                "DELETE FROM scores WHERE id IN (3, 6);",
                "INSERT INTO members VALUES (13, 2, 95);",
                "UPDATE members SET team_id = 2, rating = 85 WHERE id = 10;",
                "DELETE FROM members WHERE id = 11;",
            ]
            for stmt in dml_steps:
                conn.execute(stmt)
                assert_equiv()
        finally:
            conn.close()

    def test_postgres_dsn_selects_an_installed_driver(self) -> None:
        if importlib.util.find_spec("psycopg2") is not None:
            expected_driver = "psycopg2"
        elif importlib.util.find_spec("psycopg") is not None:
            expected_driver = "psycopg"
        else:
            self.skipTest("No PostgreSQL DB-API driver is installed")

        eng = sa.create_engine(
            _engine_url_for_dialect(
                "postgresql://user:password@localhost/matview_test",
                ms.Dialect.POSTGRES,
            )
        )
        try:
            self.assertEqual(eng.dialect.name, "postgresql")
            self.assertEqual(eng.dialect.driver, expected_driver)
        finally:
            eng.dispose()

    def test_mysql_dsn_selects_the_installed_pymysql_driver(self) -> None:
        # SQLAlchemy's bare mysql:// default is mysqlclient (MySQLdb), which is
        # intentionally absent from requirements.txt. The project installs PyMySQL.
        eng = sa.create_engine(
            _engine_url_for_dialect(
                "mysql://user:password@localhost/matview_test",
                ms.Dialect.MYSQL,
            )
        )
        try:
            self.assertEqual(eng.dialect.name, "mysql")
            self.assertEqual(eng.dialect.driver, "pymysql")
        finally:
            eng.dispose()

    def _run_live_engine_window_roundtrip(self, dsn: str, dialect: ms.Dialect) -> None:
        eng = sa.create_engine(_engine_url_for_dialect(dsn, dialect))
        view_sel = (
            "SELECT id, team, score, "
            "ROW_NUMBER() OVER (PARTITION BY team ORDER BY score DESC, id ASC) AS rn, "
            "DENSE_RANK() OVER (PARTITION BY team ORDER BY score DESC) AS drk, "
            "COALESCE(LAG(score, 1) OVER (PARTITION BY team ORDER BY id ASC), -1) AS prev_s, "
            "SUM(score) OVER (PARTITION BY team ORDER BY id ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_sum, "
            "COUNT(*) OVER (PARTITION BY team ORDER BY id ASC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_count "
            "FROM p2_win_scores"
        )
        try:
            with eng.begin() as conn:
                conn.exec_driver_sql("DROP VIEW IF EXISTS mv_p2_win")
                conn.exec_driver_sql("DROP TABLE IF EXISTS __matview_store_mv_p2_win")
                conn.exec_driver_sql("DROP TABLE IF EXISTS p2_win_scores")
                conn.exec_driver_sql(
                    "CREATE TABLE p2_win_scores (id INTEGER PRIMARY KEY, team VARCHAR(32), score INTEGER NOT NULL)"
                )
                conn.exec_driver_sql(
                    "INSERT INTO p2_win_scores VALUES (1, 'a', 10), (2, 'a', 20), (3, 'b', 30), (4, NULL, 40)"
                )
            with eng.begin() as conn:
                mgr = ms.MaterializedViewManager(
                    conn,
                    dialect=dialect,
                    ignore_limitations=(dialect == ms.Dialect.MYSQL),
                )
                plan = mgr.register_view(f"CREATE VIEW mv_p2_win AS {view_sel};")
                self.assertEqual(plan.strategy, ms.MaintenanceStrategy.GROUP_SCOPED_DELTA)
                for stmt in (
                    "INSERT INTO p2_win_scores VALUES (5, 'a', 20), (6, NULL, 5)",
                    # Change only a window ORDER BY expression while keeping the
                    # row in the same partition; every window result must converge.
                    "UPDATE p2_win_scores SET score = 5 WHERE id = 2",
                    # Change the primary-key tie-break/order expression without
                    # changing the partition or the row's other projected values.
                    "UPDATE p2_win_scores SET id = 7 WHERE id = 2",
                    # Restore a tied score after the PK/order-key change; rank and
                    # row-number semantics must still match the live query.
                    "UPDATE p2_win_scores SET score = 20 WHERE id = 7",
                    # Change only the partition key, independently of ORDER BY.
                    "UPDATE p2_win_scores SET team = 'b' WHERE id = 1",
                    "DELETE FROM p2_win_scores WHERE id = 4",
                ):
                    conn.exec_driver_sql(stmt)
                    mv_rows = sorted(conn.exec_driver_sql("SELECT * FROM mv_p2_win").fetchall())
                    live_rows = sorted(conn.exec_driver_sql(view_sel).fetchall())
                    self.assertEqual(mv_rows, live_rows)
        finally:
            # drop_view also removes the catalog entry; dropping only the
            # objects left a stale entry with the same signature, so the next
            # run's register_view was a no-op and the view stayed missing.
            with eng.begin() as conn:
                ms.MaterializedViewManager(conn, dialect=dialect).drop_view("mv_p2_win")
            with eng.begin() as conn:
                conn.exec_driver_sql("DROP TABLE IF EXISTS p2_win_scores")
            eng.dispose()

    def test_native_postgres_window_partition_scoping(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_live_engine_window_roundtrip(dsn, ms.Dialect.POSTGRES)

    def test_native_mysql_window_partition_scoping(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_live_engine_window_roundtrip(dsn, ms.Dialect.MYSQL)


class TestTier2OuterJoinGroupScoping(unittest.TestCase):
    """Tier 2 group-scoped maintenance must refresh both matched and NULL-extended
    groups across LEFT/RIGHT/FULL OUTER JOINs, chained and sibling outer joins,
    self-joins, unqualified/quoted/keyword-named columns, FK cascades, and
    WHERE subqueries on SQLite, PostgreSQL, and MySQL/MariaDB."""

    def _assert_sqlite_outer_join(
        self,
        label: str,
        schema: str,
        view_sql: str,
        seed_stmts: list[str],
        mut_stmts: list[str],
        *,
        foreign_keys: bool = False,
        expect_scoped_tables: tuple[str, ...] = (),
        expect_fallback_tables: tuple[str, ...] = (),
    ) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            if foreign_keys:
                conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(schema)
            for s in seed_stmts:
                conn.execute(s)
            conn.commit()
            mgr = ms.MaterializedViewManager(conn, dialect=ms.Dialect.SQLITE)
            plan = mgr.register_view(f"CREATE VIEW mv_oj AS {view_sql};")
            self.assertEqual(
                plan.strategy,
                ms.MaintenanceStrategy.GROUP_SCOPED_DELTA,
                f"{label}: expected Tier 2 GROUP_SCOPED_DELTA, got {plan.strategy}",
            )
            full_del = f"DELETE FROM {plan.backing_table_name};"
            for tbl in expect_scoped_tables:
                trigs = [t for t in plan.trigger_ddls if f"__{tbl}_a" in t]
                self.assertTrue(trigs, f"{label}: missing triggers for {tbl}")
                self.assertFalse(
                    any(full_del in t for t in trigs),
                    f"{label}: expected scoped trigger for {tbl}, got full refresh",
                )
            for tbl in expect_fallback_tables:
                trigs = [t for t in plan.trigger_ddls if f"__{tbl}_a" in t]
                self.assertTrue(trigs, f"{label}: missing triggers for {tbl}")
                self.assertTrue(
                    all(full_del in t for t in trigs),
                    f"{label}: expected full-refresh fallback trigger for {tbl}",
                )
            self.assertEqual(
                sorted(conn.execute("SELECT * FROM mv_oj").fetchall(), key=repr),
                sorted(conn.execute(view_sql).fetchall(), key=repr),
                f"{label}: initial mismatch",
            )
            for idx, op in enumerate(mut_stmts):
                conn.execute(op)
                conn.commit()
                mv_rows = sorted(conn.execute("SELECT * FROM mv_oj").fetchall(), key=repr)
                live_rows = sorted(conn.execute(view_sql).fetchall(), key=repr)
                self.assertEqual(
                    mv_rows,
                    live_rows,
                    f"{label} step {idx} ({op}): MV={mv_rows} LIVE={live_rows}",
                )
        finally:
            conn.close()

    def test_sqlite_left_join_parent_moves_children_between_matched_and_null_groups(self) -> None:
        schema = (
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, amt INTEGER NOT NULL);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT);"
        )
        ops = [
            "DELETE FROM p WHERE id = 1",
            "INSERT INTO p VALUES (1, 'a')",
            "INSERT INTO p VALUES (2, 'b')",
            "UPDATE p SET g = 'c' WHERE id = 1",
            "UPDATE p SET id = 3 WHERE id = 2",
            "DELETE FROM p",
            "INSERT INTO p VALUES (1, 'a'), (2, 'b')",
            "DELETE FROM o WHERE id = 10",
            "INSERT INTO o VALUES (12, 99, 50)",
        ]
        self._assert_sqlite_outer_join(
            "left_join_group_by_p_g",
            schema,
            "SELECT p.g AS g, COUNT(*) AS cnt, SUM(o.amt) AS s FROM o LEFT JOIN p ON o.pid = p.id GROUP BY p.g",
            ["INSERT INTO p VALUES (1, 'a')", "INSERT INTO o VALUES (10, 1, 100), (11, 2, 200)"],
            ops,
        )
        self._assert_sqlite_outer_join(
            "left_join_coalesce_p_g",
            schema,
            "SELECT COALESCE(p.g, 'none') AS g, COUNT(*) AS cnt, SUM(o.amt) AS s "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY COALESCE(p.g, 'none')",
            ["INSERT INTO p VALUES (1, 'a')", "INSERT INTO o VALUES (10, 1, 100), (11, 2, 200)"],
            ops,
        )

    def test_sqlite_outer_join_mixed_sibling_chained_self_and_fk_cascades(self) -> None:
        schema_mix = (
            "CREATE TABLE o (id INTEGER PRIMARY KEY, k TEXT NOT NULL, pid INTEGER, amt INTEGER NOT NULL);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT);"
        )
        mix_ops = [
            "DELETE FROM p WHERE id = 1",
            "INSERT INTO p VALUES (1, 'a')",
            "INSERT INTO p VALUES (2, 'b')",
            "UPDATE p SET g = 'c' WHERE id = 1",
            "UPDATE p SET id = 3 WHERE id = 2",
            "DELETE FROM p",
        ]
        self._assert_sqlite_outer_join(
            "left_join_mixed_o_k_p_g",
            schema_mix,
            "SELECT o.k AS k, p.g AS g, COUNT(*) AS cnt FROM o LEFT JOIN p ON o.pid = p.id GROUP BY o.k, p.g",
            ["INSERT INTO p VALUES (1, 'a')", "INSERT INTO o VALUES (10, 'k1', 1, 100), (11, 'k2', 2, 200)"],
            mix_ops,
        )
        self._assert_sqlite_outer_join(
            "left_join_coalesce_p_g_o_k",
            schema_mix,
            "SELECT COALESCE(p.g, o.k) AS grp, COUNT(*) AS cnt FROM o LEFT JOIN p ON o.pid = p.id GROUP BY COALESCE(p.g, o.k)",
            ["INSERT INTO p VALUES (1, 'a')", "INSERT INTO o VALUES (10, 'k1', 1, 100), (11, 'k2', 2, 200)"],
            mix_ops,
        )

        # Sibling LEFT JOINs: deleting p must NOT nullify q in COALESCE(p.g, q.label)
        self._assert_sqlite_outer_join(
            "sibling_left_joins_coalesce_p_q",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, qid INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT);\n"
            "CREATE TABLE q (id INTEGER PRIMARY KEY, label TEXT);",
            "SELECT COALESCE(p.g, q.label) AS grp, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id LEFT JOIN q ON o.qid = q.id "
            "GROUP BY COALESCE(p.g, q.label)",
            ["INSERT INTO p VALUES (1, 'pg1')", "INSERT INTO q VALUES (10, 'lbl1')", "INSERT INTO o VALUES (1, 1, 10)"],
            [
                "DELETE FROM p WHERE id = 1",
                "INSERT INTO p VALUES (1, 'pg1')",
                "DELETE FROM q WHERE id = 10",
                "DELETE FROM p WHERE id = 1",
                "INSERT INTO q VALUES (10, 'lbl2')",
            ],
        )

        # Chained LEFT JOIN: o -> p -> q grouped by q.label
        self._assert_sqlite_outer_join(
            "chained_left_join_o_p_q",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, qid INTEGER);\n"
            "CREATE TABLE q (id INTEGER PRIMARY KEY, label TEXT);",
            "SELECT q.label AS lbl, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id LEFT JOIN q ON p.qid = q.id GROUP BY q.label",
            ["INSERT INTO q VALUES (100, 'top')", "INSERT INTO p VALUES (1, 100)", "INSERT INTO o VALUES (10, 1), (11, 2)"],
            [
                "DELETE FROM p WHERE id = 1",
                "INSERT INTO p VALUES (1, 100)",
                "DELETE FROM q WHERE id = 100",
                "INSERT INTO q VALUES (100, 'new_top')",
                "UPDATE p SET qid = NULL WHERE id = 1",
                "UPDATE p SET qid = 100 WHERE id = 1",
            ],
        )

        # Dependent non-strict ON condition after LEFT JOIN fails closed safely on p while o stays scoped
        self._assert_sqlite_outer_join(
            "dependent_coalesce_on_after_left_join",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, qid INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT, qid INTEGER);\n"
            "CREATE TABLE q (id INTEGER PRIMARY KEY, label TEXT);",
            "SELECT p.g AS g, q.label AS lbl, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id "
            "INNER JOIN q ON COALESCE(p.qid, o.qid) = q.id "
            "GROUP BY p.g, q.label",
            ["INSERT INTO q VALUES (10, 'q10'), (20, 'q20')", "INSERT INTO p VALUES (1, 'a', 10)", "INSERT INTO o VALUES (1, 1, 20)"],
            ["DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (1, 'a', 10)"],
            expect_scoped_tables=("o",),
            expect_fallback_tables=("p",),
        )

        # Self LEFT JOIN: nodes n LEFT JOIN nodes p GROUP BY p.g (scoped) and GROUP BY n.g, p.g (self-loop fallback)
        self._assert_sqlite_outer_join(
            "self_left_join_p_g",
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, parent_id INTEGER, g TEXT);",
            "SELECT p.g AS pg, COUNT(*) AS cnt FROM nodes n LEFT JOIN nodes p ON n.parent_id = p.id GROUP BY p.g",
            ["INSERT INTO nodes VALUES (1, NULL, 'root'), (2, 1, 'child1'), (3, 1, 'child2')"],
            [
                "UPDATE nodes SET g = 'ROOT2' WHERE id = 1",
                "DELETE FROM nodes WHERE id = 1",
                "INSERT INTO nodes VALUES (1, NULL, 'reborn')",
                "UPDATE nodes SET parent_id = 2 WHERE id = 3",
            ],
        )
        self._assert_sqlite_outer_join(
            "self_left_join_both_n_g_and_p_g_with_self_loop",
            "CREATE TABLE nodes (id INTEGER PRIMARY KEY, parent_id INTEGER, g TEXT);",
            "SELECT n.g AS ng, p.g AS pg, COUNT(*) AS cnt FROM nodes n LEFT JOIN nodes p ON n.parent_id = p.id GROUP BY n.g, p.g",
            ["INSERT INTO nodes VALUES (1, 1, 'loop'), (2, 1, 'child1'), (3, NULL, 'root')"],
            [
                "UPDATE nodes SET g = 'loop_v2' WHERE id = 1",
                "UPDATE nodes SET g = g || '_all'",
                "DELETE FROM nodes WHERE id = 1",
                "INSERT INTO nodes VALUES (1, 1, 'reborn')",
            ],
            expect_fallback_tables=("nodes",),
        )
        self._assert_sqlite_outer_join(
            "multi_alias_same_nullable_table_in_group_by",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, p1 INTEGER, p2 INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, pg TEXT);",
            "SELECT p1t.pg AS g1, p2t.pg AS g2, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p p1t ON o.p1 = p1t.id LEFT JOIN p p2t ON o.p2 = p2t.id "
            "GROUP BY p1t.pg, p2t.pg",
            ["INSERT INTO p VALUES (1, 'A')", "INSERT INTO o VALUES (10, 1, 1)"],
            ["UPDATE p SET pg = 'B' WHERE id = 1", "DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (1, 'C')"],
            expect_scoped_tables=("o",),
            expect_fallback_tables=("p",),
        )

        # Foreign-key cascades (ON DELETE SET NULL, ON DELETE CASCADE, ON UPDATE CASCADE)
        self._assert_sqlite_outer_join(
            "fk_on_delete_set_null_mixed_group",
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT);\n"
            "CREATE TABLE o (id INTEGER PRIMARY KEY, k TEXT NOT NULL, pid INTEGER REFERENCES p(id) ON DELETE SET NULL);",
            "SELECT o.k AS k, p.g AS g, COUNT(*) AS cnt FROM o LEFT JOIN p ON o.pid = p.id GROUP BY o.k, p.g",
            ["INSERT INTO p VALUES (1, 'a')", "INSERT INTO o VALUES (10, 'k1', 1)"],
            ["DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (2, 'b')", "UPDATE o SET pid = 2 WHERE id = 10"],
            foreign_keys=True,
        )
        self._assert_sqlite_outer_join(
            "fk_on_delete_and_update_cascade_mixed_group",
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT);\n"
            "CREATE TABLE o (id INTEGER PRIMARY KEY, k TEXT NOT NULL, pid INTEGER REFERENCES p(id) ON DELETE CASCADE ON UPDATE CASCADE);",
            "SELECT o.k AS k, p.g AS g, COUNT(*) AS cnt FROM o LEFT JOIN p ON o.pid = p.id GROUP BY o.k, p.g",
            ["INSERT INTO p VALUES (1, 'a')", "INSERT INTO o VALUES (10, 'k1', 1)"],
            ["UPDATE p SET id = 3, g = 'new_g' WHERE id = 1", "DELETE FROM p WHERE id = 3"],
            foreign_keys=True,
        )

    def test_sqlite_unqualified_quoted_keyword_subquery_and_shared_col_edge_cases(self) -> None:
        # Unqualified GROUP BY columns in multi-table join must not shadow inner EXISTS
        self._assert_sqlite_outer_join(
            "unqualified_group_by_in_left_join",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, k TEXT NOT NULL, pid INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, grp TEXT);",
            "SELECT grp, k, COUNT(*) AS cnt FROM o LEFT JOIN p ON o.pid = p.id GROUP BY grp, k",
            ["INSERT INTO p VALUES (1, 'a'), (2, 'b')", "INSERT INTO o VALUES (10, 'k1', 1), (11, 'k2', 2)"],
            ["INSERT INTO o VALUES (12, 'k3', 1)", "DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (1, 'a')"],
        )

        # Quoted reserved-keyword table and column names
        self._assert_sqlite_outer_join(
            "quoted_reserved_identifiers_in_left_join",
            'CREATE TABLE "order" (id INTEGER PRIMARY KEY, pid INTEGER, "group" TEXT);\n'
            'CREATE TABLE p (id INTEGER PRIMARY KEY, "select" TEXT);',
            'SELECT "order"."group" AS og, p."select" AS ps, COUNT(*) AS cnt '
            'FROM "order" LEFT JOIN p ON "order".pid = p.id '
            'GROUP BY "order"."group", p."select"',
            ['INSERT INTO p VALUES (1, \'s1\')', 'INSERT INTO "order" VALUES (10, 1, \'g1\')'],
            ["DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (1, 's2')", 'INSERT INTO "order" VALUES (11, 2, \'g2\')'],
        )

        # Keyword-named columns (text, "end") referenced inside CAST / CASE expressions
        self._assert_sqlite_outer_join(
            "keyword_named_columns_with_cast_and_case",
            'CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER);\n'
            'CREATE TABLE p (id INTEGER PRIMARY KEY, text TEXT, "end" TEXT);',
            'SELECT CAST(p.text AS TEXT) AS ct, CASE WHEN p."end" IS NULL THEN \'x\' ELSE p."end" END AS ke, COUNT(*) AS cnt '
            'FROM o LEFT JOIN p ON o.pid = p.id '
            'GROUP BY CAST(p.text AS TEXT), CASE WHEN p."end" IS NULL THEN \'x\' ELSE p."end" END',
            ["INSERT INTO p VALUES (1, 't1', 'e1')", "INSERT INTO o VALUES (10, 1), (11, 2)"],
            ["DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (1, 't2', 'e2')"],
            expect_scoped_tables=("o", "p"),
        )

        # Subqueries in WHERE, HAVING, SELECT list, and JOIN ON clauses fail closed on referenced table
        self._assert_sqlite_outer_join(
            "where_scalar_subquery_on_joined_table",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, amt INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT, val INTEGER);",
            "SELECT p.g AS g, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id "
            "WHERE o.amt > (SELECT MIN(val) FROM p) GROUP BY p.g",
            ["INSERT INTO p VALUES (1, 'a', 50), (2, 'b', 50)", "INSERT INTO o VALUES (1, 1, 100), (2, 2, 100), (3, 2, 20)"],
            ["INSERT INTO p VALUES (3, 'c', 10)", "DELETE FROM p WHERE id = 3"],
            expect_fallback_tables=("p",),
        )
        self._assert_sqlite_outer_join(
            "having_subquery_on_joined_table",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT, val INTEGER);",
            "SELECT p.g AS g, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id "
            "GROUP BY p.g HAVING COUNT(*) >= (SELECT MAX(val) FROM p)",
            ["INSERT INTO p VALUES (1, 'a', 1), (2, 'b', 1)", "INSERT INTO o VALUES (10, 1), (11, 1), (12, 2)"],
            ["INSERT INTO p VALUES (3, 'c', 5)", "DELETE FROM p WHERE id = 3"],
            expect_fallback_tables=("p",),
        )
        self._assert_sqlite_outer_join(
            "select_list_subquery_on_joined_table",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT, bonus INTEGER);",
            "SELECT p.g AS g, COUNT(*) AS cnt, (SELECT MAX(bonus) FROM p) AS max_b "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY p.g",
            ["INSERT INTO p VALUES (1, 'a', 10), (2, 'b', 5)", "INSERT INTO o VALUES (10, 1), (11, 2)"],
            ["INSERT INTO p VALUES (3, 'c', 99)", "DELETE FROM p WHERE id = 3"],
            expect_fallback_tables=("p",),
        )
        self._assert_sqlite_outer_join(
            "own_on_subquery_on_joined_table",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, amt INTEGER);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT, thresh INTEGER);",
            "SELECT p.g AS g, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id AND o.amt > (SELECT COALESCE(MAX(p2.thresh), 0) FROM p p2) "
            "GROUP BY p.g",
            ["INSERT INTO p VALUES (1, 'a', 10)", "INSERT INTO o VALUES (10, 1, 50), (11, 9, 50)"],
            ["INSERT INTO p VALUES (2, 'b', 200)", "DELETE FROM p WHERE id = 2"],
            expect_fallback_tables=("p",),
        )
        self._assert_sqlite_outer_join(
            "later_on_subquery_on_joined_table",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, amt INTEGER, ok TEXT);\n"
            "CREATE TABLE p (pid_pk INTEGER PRIMARY KEY, qid INTEGER, thresh INTEGER, pg TEXT);\n"
            "CREATE TABLE q (qid_pk INTEGER PRIMARY KEY, qg TEXT);",
            "SELECT o.ok, p.pg, q.qg, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.pid_pk "
            "LEFT JOIN q ON p.qid = q.qid_pk AND o.amt > COALESCE((SELECT thresh FROM p WHERE pg = 'P2'), 0) "
            "GROUP BY o.ok, p.pg, q.qg",
            [
                "INSERT INTO q VALUES (10, 'Q1')",
                "INSERT INTO p VALUES (1, 10, 5, 'P1')",
                "INSERT INTO o VALUES (100, 1, 50, 'O1')",
            ],
            ["INSERT INTO p VALUES (2, 10, 1000, 'P2')", "DELETE FROM p WHERE pid_pk = 2"],
            expect_fallback_tables=("p",),
        )

        # Shared unprojected column name (o.g projected, p.g unprojected in GROUP BY)
        self._assert_sqlite_outer_join(
            "shared_column_name_unprojected_group_by",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, g TEXT);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, g TEXT);",
            "SELECT o.g AS og, COUNT(*) AS cnt FROM o LEFT JOIN p ON o.pid = p.id GROUP BY o.g, p.g",
            ["INSERT INTO p VALUES (1, 'p1')", "INSERT INTO o VALUES (10, 1, 'o1')"],
            ["UPDATE p SET g = 'p2' WHERE id = 1", "DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (1, 'p3')"],
        )

    def test_sqlite_right_and_full_outer_join_scoping(self) -> None:
        if sqlite3.sqlite_version_info < (3, 39, 0):
            self.skipTest("RIGHT/FULL OUTER JOIN requires SQLite >= 3.39.0")
        schema = (
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, g TEXT);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, name TEXT);"
        )
        self._assert_sqlite_outer_join(
            "right_join_group_by_o_g",
            schema,
            "SELECT o.g AS og, COUNT(*) AS cnt FROM o RIGHT JOIN p ON o.pid = p.id GROUP BY o.g",
            ["INSERT INTO p VALUES (1, 'p1'), (2, 'p2')", "INSERT INTO o VALUES (10, 1, 'x')"],
            [
                "DELETE FROM o WHERE id = 10",
                "INSERT INTO o VALUES (10, 1, 'x')",
                "INSERT INTO p VALUES (3, 'p3')",
                "DELETE FROM p WHERE id = 2",
                "UPDATE o SET g = 'y' WHERE id = 10",
            ],
        )
        self._assert_sqlite_outer_join(
            "full_outer_join_coalesce_o_g_p_name",
            schema,
            "SELECT COALESCE(o.g, p.name) AS grp, COUNT(*) AS cnt "
            "FROM o FULL OUTER JOIN p ON o.pid = p.id GROUP BY COALESCE(o.g, p.name)",
            ["INSERT INTO p VALUES (1, 'p1'), (2, 'p2')", "INSERT INTO o VALUES (10, 1, 'x'), (11, 9, 'z')"],
            [
                "DELETE FROM o WHERE id = 10",
                "INSERT INTO o VALUES (10, 1, 'x')",
                "DELETE FROM p WHERE id = 1",
                "INSERT INTO p VALUES (1, 'p1')",
                "DELETE FROM p",
            ],
        )
        # RIGHT JOIN + LEFT JOIN: deleting o must NOT nullify q
        self._assert_sqlite_outer_join(
            "right_join_then_left_join_coalesce",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, g TEXT);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, qid INTEGER);\n"
            "CREATE TABLE q (id INTEGER PRIMARY KEY, label TEXT);",
            "SELECT COALESCE(o.g, q.label) AS grp, COUNT(*) AS cnt "
            "FROM o RIGHT JOIN p ON o.pid = p.id LEFT JOIN q ON p.qid = q.id "
            "GROUP BY COALESCE(o.g, q.label)",
            ["INSERT INTO q VALUES (100, 'L1')", "INSERT INTO p VALUES (1, 100)", "INSERT INTO o VALUES (10, 1, 'x')"],
            ["DELETE FROM o WHERE id = 10", "INSERT INTO o VALUES (10, 1, 'x')", "DELETE FROM q WHERE id = 100"],
        )
        # LEFT JOIN + RIGHT JOIN: deleting p nullifies BOTH o and p
        self._assert_sqlite_outer_join(
            "left_join_then_right_join",
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, ok TEXT);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, qid INTEGER, pg TEXT);\n"
            "CREATE TABLE q (id INTEGER PRIMARY KEY, label TEXT);",
            "SELECT o.ok AS ok, p.pg AS pg, q.label AS lbl, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id RIGHT JOIN q ON p.qid = q.id "
            "GROUP BY o.ok, p.pg, q.label",
            ["INSERT INTO q VALUES (100, 'L1')", "INSERT INTO p VALUES (1, 100, 'P1')", "INSERT INTO o VALUES (10, 1, 'O1')"],
            ["DELETE FROM p WHERE id = 1", "INSERT INTO p VALUES (1, 100, 'P1')"],
        )
        # INNER JOIN / CROSS JOIN upstream of RIGHT JOIN and 3-table FULL OUTER JOIN
        inner_cross_schema = (
            "CREATE TABLE o (id INTEGER PRIMARY KEY, pid INTEGER, ok TEXT);\n"
            "CREATE TABLE q (id INTEGER PRIMARY KEY, oid INTEGER, qid INTEGER, qg TEXT);\n"
            "CREATE TABLE p (id INTEGER PRIMARY KEY, pg TEXT);"
        )
        self._assert_sqlite_outer_join(
            "inner_join_before_right_join",
            inner_cross_schema,
            "SELECT o.ok, p.pg, q.qg, COUNT(*) AS cnt "
            "FROM o INNER JOIN q ON o.id = q.oid RIGHT JOIN p ON o.pid = p.id "
            "GROUP BY o.ok, p.pg, q.qg",
            ["INSERT INTO p VALUES (1, 'P1')", "INSERT INTO o VALUES (10, 1, 'O1')", "INSERT INTO q VALUES (100, 10, 1, 'Q1')"],
            ["DELETE FROM q WHERE id = 100", "INSERT INTO q VALUES (100, 10, 1, 'Q1')"],
            expect_scoped_tables=("o", "q", "p"),
        )
        self._assert_sqlite_outer_join(
            "cross_join_before_right_join",
            inner_cross_schema,
            "SELECT o.ok, p.pg, q.qg, COUNT(*) AS cnt "
            "FROM o CROSS JOIN q RIGHT JOIN p ON o.pid = p.id "
            "GROUP BY o.ok, p.pg, q.qg",
            ["INSERT INTO p VALUES (1, 'P1')", "INSERT INTO o VALUES (10, 1, 'O1')", "INSERT INTO q VALUES (100, 10, 1, 'Q1')"],
            ["DELETE FROM q WHERE id = 100", "INSERT INTO q VALUES (100, 10, 1, 'Q1')"],
            expect_scoped_tables=("o", "q", "p"),
        )
        self._assert_sqlite_outer_join(
            "cross_join_before_full_outer_join",
            inner_cross_schema,
            "SELECT o.ok, q.qg, p.pg, COUNT(*) AS cnt "
            "FROM o CROSS JOIN q FULL OUTER JOIN p ON q.qid = p.id "
            "GROUP BY o.ok, q.qg, p.pg",
            ["INSERT INTO p VALUES (1, 'P1')", "INSERT INTO o VALUES (10, 1, 'O1')", "INSERT INTO q VALUES (100, 10, 1, 'Q1')"],
            ["DELETE FROM o WHERE id = 10", "INSERT INTO o VALUES (10, 1, 'O1')"],
            expect_fallback_tables=("o", "q", "p"),
        )
        self._assert_sqlite_outer_join(
            "inner_join_before_full_outer_join",
            inner_cross_schema,
            "SELECT o.ok, q.qg, p.pg, COUNT(*) AS cnt "
            "FROM o INNER JOIN q ON o.id = q.oid FULL OUTER JOIN p ON o.pid = p.id "
            "GROUP BY o.ok, q.qg, p.pg",
            ["INSERT INTO p VALUES (1, 'P1')", "INSERT INTO o VALUES (10, 1, 'O1')", "INSERT INTO q VALUES (100, 10, 1, 'Q1')"],
            ["DELETE FROM q WHERE id = 100", "INSERT INTO q VALUES (100, 10, 1, 'Q1')"],
            expect_fallback_tables=("o", "q", "p"),
        )

    def test_postgres_and_mysql_outer_join_codegen_contracts(self) -> None:
        # 1. PostgreSQL: uppercase unquoted aliases/columns must not get erroneously quoted
        pg_schemas = {
            "o": ms.TableSchema("o", {
                "id": ms.ColumnDef("id", "INTEGER", not_null=True, primary_key=True),
                "pid": ms.ColumnDef("pid", "INTEGER"),
                "ok": ms.ColumnDef("ok", "TEXT"),
                "amt": ms.ColumnDef("amt", "INTEGER", not_null=True),
            }),
            "p": ms.TableSchema("p", {
                "id": ms.ColumnDef("id", "SERIAL", not_null=True, primary_key=True),
                "pg": ms.ColumnDef("pg", "TEXT"),
                "num": ms.ColumnDef("num", "INTEGER"),
                "arr": ms.ColumnDef("arr", "INTEGER[]"),
                "mood": ms.ColumnDef("mood", "USER-DEFINED"),
            }),
        }

        def _compile(view_sql: str, dialect: ms.Dialect = ms.Dialect.POSTGRES) -> ms.MaterializedViewPlan:
            views = ms.parse_view_definitions(
                view_sql,
                known_tables=set(pg_schemas.keys()),
                dialect=dialect,
            )
            return ms.IncrementalViewCompiler(
                pg_schemas,
                views[0],
                dialect=dialect,
                ignore_limitations=(dialect == ms.Dialect.MYSQL),
            ).compile()

        plan_upper = _compile(
            "CREATE VIEW mv_u AS SELECT O.ok, P.pg, COUNT(*) AS cnt "
            "FROM o O LEFT JOIN p P ON O.pid = P.id GROUP BY O.ok, P.pg;"
        )
        trig_sql = "\n".join(plan_upper.trigger_ddls)
        self.assertNotIn('"O".', trig_sql)
        self.assertNotIn('"P".', trig_sql)

        # 2. PostgreSQL: SERIAL column projected/grouped becomes nullable INTEGER in backing table
        plan_serial = _compile(
            "CREATE VIEW mv_ser AS SELECT p.id AS pid_out, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY p.id;"
        )
        self.assertNotIn("SERIAL", plan_serial.backing_table_ddl.upper())
        self.assertIn("INTEGER", plan_serial.backing_table_ddl.upper())

        # 3. PostgreSQL: unprojected & whitespace-mismatched expressions infer native types (not TEXT)
        plan_unproj = _compile(
            "CREATE VIEW mv_unp AS SELECT COUNT(*) AS cnt, SUM(o.amt) AS s "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY ABS(p.num);"
        )
        self.assertIn("__mv_grp_0 INTEGER", plan_unproj.backing_table_ddl)
        plan_ws = _compile(
            "CREATE VIEW mv_ws AS SELECT COALESCE(p.num, 0) AS x, SUM(o.amt) AS s "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY COALESCE(p.num,0);"
        )
        self.assertNotIn("__mv_grp_0", plan_ws.backing_table_ddl)

        # 4. PostgreSQL: boolean/comparison GROUP BY keys are parenthesized and typed BOOLEAN
        plan_bool = _compile(
            "CREATE VIEW mv_b AS SELECT p.num > 0 AS pos, p.num IS NULL AS isn, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY p.num > 0, p.num IS NULL;"
        )
        self.assertIn("pos BOOLEAN", plan_bool.backing_table_ddl)
        self.assertIn("isn BOOLEAN", plan_bool.backing_table_ddl)
        bool_trigs = "\n".join(plan_bool.trigger_ddls)
        self.assertIn("(p.num > 0) = ANY", bool_trigs)
        self.assertIn("(p.num IS NULL) = ANY", bool_trigs)

        # 5. PostgreSQL: ARRAY group key uses IN (SELECT ...) instead of forbidden ARRAY(SELECT ...)
        plan_arr = _compile(
            "CREATE VIEW mv_arr AS SELECT p.arr AS arr, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY p.arr;"
        )
        arr_trigs = "\n".join(plan_arr.trigger_ddls)
        self.assertNotIn("ARRAY(SELECT", arr_trigs)
        self.assertIn("IN (SELECT __mv_kset.__mv_k0", arr_trigs)

        # 6. PostgreSQL: USER-DEFINED enum inside COALESCE fails closed on p
        plan_enum = _compile(
            "CREATE VIEW mv_enum AS SELECT COALESCE(p.mood, 'sad') AS m, COUNT(*) AS cnt "
            "FROM o LEFT JOIN p ON o.pid = p.id GROUP BY COALESCE(p.mood, 'sad');"
        )
        p_enum_trigs = [t for t in plan_enum.trigger_ddls if "__p_" in t]
        self.assertTrue(any(f"DELETE FROM {plan_enum.backing_table_name};" in t for t in p_enum_trigs))

        # 7. PostgreSQL: schema-qualified tables preserved in inner join graph
        plan_schema = _compile(
            "CREATE VIEW mv_sch AS SELECT o.ok, p.pg, COUNT(*) AS cnt "
            "FROM sales.o o LEFT JOIN sales.p p ON o.pid = p.id GROUP BY o.ok, p.pg;"
        )
        self.assertIn("sales.p AS __mv_j_1", "\n".join(plan_schema.trigger_ddls))

        # 8. MySQL: grouped FULL OUTER JOIN is explicitly rejected rather than emitting invalid SQL
        with self.assertRaises(ms.UnsupportedMaterializedViewError):
            _compile(
                "CREATE VIEW mv_my_full AS SELECT o.ok, p.pg, COUNT(*) AS cnt "
                "FROM o FULL OUTER JOIN p ON o.pid = p.id GROUP BY o.ok, p.pg;",
                dialect=ms.Dialect.MYSQL,
            )

    def _run_native_outer_join_matrix(self, dsn: str, dialect: ms.Dialect) -> None:
        eng = sa.create_engine(_engine_url_for_dialect(dsn, dialect))
        is_pg = dialect == ms.Dialect.POSTGRES
        cases: list[tuple[str, tuple[str, ...], list[str], list[str], str, list[str]]] = [
            (
                "mv_oj_pg_g",
                ("oj_o", "oj_p"),
                [
                    "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, g VARCHAR(32))",
                    "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER, amt INTEGER NOT NULL)",
                ],
                ["INSERT INTO oj_p VALUES (1, 'a')", "INSERT INTO oj_o VALUES (10, 1, 100), (11, 2, 200)"],
                "SELECT p.g AS g, COUNT(*) AS cnt, SUM(o.amt) AS s "
                "FROM oj_o o LEFT JOIN oj_p p ON o.pid = p.id GROUP BY p.g",
                [
                    "DELETE FROM oj_p WHERE id = 1",
                    "INSERT INTO oj_p VALUES (1, 'a')",
                    "INSERT INTO oj_p VALUES (2, 'b')",
                    "UPDATE oj_p SET g = 'c' WHERE id = 1",
                    "UPDATE oj_p SET id = 3 WHERE id = 2",
                    "DELETE FROM oj_p",
                ],
            ),
            (
                "mv_oj_coal",
                ("oj_o", "oj_p"),
                [
                    "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, g VARCHAR(32))",
                    "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, k VARCHAR(32) NOT NULL, pid INTEGER)",
                ],
                ["INSERT INTO oj_p VALUES (1, 'a')", "INSERT INTO oj_o VALUES (10, 'k1', 1), (11, 'k2', 2)"],
                "SELECT COALESCE(p.g, o.k) AS grp, COUNT(*) AS cnt "
                "FROM oj_o o LEFT JOIN oj_p p ON o.pid = p.id GROUP BY COALESCE(p.g, o.k)",
                [
                    "DELETE FROM oj_p WHERE id = 1",
                    "INSERT INTO oj_p VALUES (1, 'a')",
                    "INSERT INTO oj_p VALUES (2, 'b')",
                    "DELETE FROM oj_p",
                ],
            ),
            (
                "mv_oj_abs",
                ("oj_o", "oj_p"),
                [
                    "CREATE TABLE oj_p (id SERIAL PRIMARY KEY, num INTEGER)" if is_pg
                    else "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, num INTEGER)",
                    "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER)",
                ],
                ["INSERT INTO oj_p (id, num) VALUES (1, -5)", "INSERT INTO oj_o VALUES (10, 1), (11, 2)"],
                "SELECT ABS(p.num) AS an, COUNT(*) AS cnt "
                "FROM oj_o o LEFT JOIN oj_p p ON o.pid = p.id GROUP BY ABS(p.num)",
                [
                    "DELETE FROM oj_p WHERE id = 1",
                    "INSERT INTO oj_p (id, num) VALUES (1, -5)",
                    "INSERT INTO oj_p (id, num) VALUES (2, -9)",
                    "DELETE FROM oj_p",
                ],
            ),
            (
                "mv_oj_self",
                ("oj_nodes",),
                ["CREATE TABLE oj_nodes (id INTEGER PRIMARY KEY, parent_id INTEGER, g VARCHAR(32))"],
                ["INSERT INTO oj_nodes VALUES (1, NULL, 'root'), (2, 1, 'c1'), (3, 1, 'c2')"],
                "SELECT n.g AS ng, p.g AS pg, COUNT(*) AS cnt "
                "FROM oj_nodes n LEFT JOIN oj_nodes p ON n.parent_id = p.id GROUP BY n.g, p.g",
                [
                    "UPDATE oj_nodes SET g = 'ROOT2' WHERE id = 1",
                    "UPDATE oj_nodes SET g = CONCAT(g, '_v2')",
                    "DELETE FROM oj_nodes WHERE id = 1",
                    "INSERT INTO oj_nodes VALUES (1, NULL, 'reborn')",
                ],
            ),
            (
                "mv_oj_sib",
                ("oj_o", "oj_p", "oj_q"),
                [
                    "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, g VARCHAR(32))",
                    "CREATE TABLE oj_q (id INTEGER PRIMARY KEY, label VARCHAR(32))",
                    "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER, qid INTEGER)",
                ],
                ["INSERT INTO oj_p VALUES (1, 'pg1')", "INSERT INTO oj_q VALUES (10, 'lbl1')", "INSERT INTO oj_o VALUES (1, 1, 10)"],
                "SELECT COALESCE(p.g, q.label) AS grp, COUNT(*) AS cnt "
                "FROM oj_o o LEFT JOIN oj_p p ON o.pid = p.id LEFT JOIN oj_q q ON o.qid = q.id "
                "GROUP BY COALESCE(p.g, q.label)",
                ["DELETE FROM oj_p WHERE id = 1", "INSERT INTO oj_p VALUES (1, 'pg1')", "DELETE FROM oj_q WHERE id = 10"],
            ),
            (
                "mv_oj_rj",
                ("oj_o", "oj_p"),
                [
                    "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, name VARCHAR(32))",
                    "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER, g VARCHAR(32))",
                ],
                ["INSERT INTO oj_p VALUES (1, 'p1'), (2, 'p2')", "INSERT INTO oj_o VALUES (10, 1, 'x')"],
                "SELECT o.g AS og, COUNT(*) AS cnt "
                "FROM oj_o o RIGHT JOIN oj_p p ON o.pid = p.id GROUP BY o.g",
                ["DELETE FROM oj_o WHERE id = 10", "INSERT INTO oj_o VALUES (10, 1, 'x')", "INSERT INTO oj_p VALUES (3, 'p3')"],
            ),
        ]
        if is_pg:
            cases.extend(
                [
                    (
                        "mv_oj_full",
                        ("oj_o", "oj_p"),
                        [
                            "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, name VARCHAR(32))",
                            "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER, g VARCHAR(32))",
                        ],
                        ["INSERT INTO oj_p VALUES (1, 'p1'), (2, 'p2')", "INSERT INTO oj_o VALUES (10, 1, 'x'), (11, 9, 'z')"],
                        "SELECT COALESCE(o.g, p.name) AS grp, COUNT(*) AS cnt "
                        "FROM oj_o o FULL OUTER JOIN oj_p p ON o.pid = p.id GROUP BY COALESCE(o.g, p.name)",
                        ["DELETE FROM oj_o WHERE id = 10", "INSERT INTO oj_o VALUES (10, 1, 'x')", "DELETE FROM oj_p WHERE id = 1"],
                    ),
                    (
                        "mv_oj_fk",
                        ("oj_o", "oj_p"),
                        [
                            "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, g VARCHAR(32))",
                            "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, k VARCHAR(32) NOT NULL, pid INTEGER REFERENCES oj_p(id) ON DELETE SET NULL)",
                        ],
                        ["INSERT INTO oj_p VALUES (1, 'a')", "INSERT INTO oj_o VALUES (10, 'k1', 1)"],
                        "SELECT o.k AS k, p.g AS g, COUNT(*) AS cnt "
                        "FROM oj_o o LEFT JOIN oj_p p ON o.pid = p.id GROUP BY o.k, p.g",
                        ["DELETE FROM oj_p WHERE id = 1", "INSERT INTO oj_p VALUES (2, 'b')", "UPDATE oj_o SET pid = 2 WHERE id = 10"],
                    ),
                    (
                        "mv_oj_pg_expr_bool_upper",
                        ("oj_o", "oj_p"),
                        [
                            "CREATE TABLE oj_p (id SERIAL PRIMARY KEY, num INTEGER, g VARCHAR(32))",
                            "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER, amt INTEGER NOT NULL)",
                        ],
                        [
                            "INSERT INTO oj_p (id, num, g) VALUES (1, 5, 'a'), (2, -3, 'b')",
                            "INSERT INTO oj_o VALUES (10, 1, 100), (11, 2, 200), (12, 3, 50)",
                        ],
                        "SELECT P.num > 0 AS pos, P.num IS NULL AS isn, COUNT(*) AS cnt, SUM(O.amt) AS s "
                        "FROM oj_o O LEFT JOIN oj_p P ON O.pid = P.id GROUP BY P.num > 0, P.num IS NULL",
                        [
                            "UPDATE oj_p SET num = -num",
                            "DELETE FROM oj_p WHERE id IN (1, 2)",
                            "INSERT INTO oj_p (id, num, g) VALUES (1, 10, 'a'), (3, -2, 'c')",
                        ],
                    ),
                    (
                        "mv_oj_pg_unproj_abs_serial",
                        ("oj_o", "oj_p"),
                        [
                            "CREATE TABLE oj_p (id SERIAL PRIMARY KEY, num INTEGER)",
                            "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER, amt INTEGER NOT NULL)",
                        ],
                        [
                            "INSERT INTO oj_p (id, num) VALUES (1, -7), (2, 7)",
                            "INSERT INTO oj_o VALUES (10, 1, 10), (11, 2, 20), (12, 9, 30)",
                        ],
                        "SELECT COUNT(*) AS cnt, SUM(o.amt) AS s "
                        "FROM oj_o o LEFT JOIN oj_p p ON o.pid = p.id GROUP BY ABS(p.num)",
                        [
                            "DELETE FROM oj_p WHERE id = 1",
                            "INSERT INTO oj_p (id, num) VALUES (9, -7)",
                            "DELETE FROM oj_p",
                        ],
                    ),
                    (
                        "mv_oj_pg_inner_right",
                        ("oj_o", "oj_q", "oj_p"),
                        [
                            "CREATE TABLE oj_p (id INTEGER PRIMARY KEY, pg VARCHAR(32))",
                            "CREATE TABLE oj_o (id INTEGER PRIMARY KEY, pid INTEGER, ok VARCHAR(32))",
                            "CREATE TABLE oj_q (id INTEGER PRIMARY KEY, oid INTEGER, qg VARCHAR(32))",
                        ],
                        [
                            "INSERT INTO oj_p VALUES (1, 'P1')",
                            "INSERT INTO oj_o VALUES (10, 1, 'O1')",
                            "INSERT INTO oj_q VALUES (100, 10, 'Q1')",
                        ],
                        "SELECT o.ok, p.pg, q.qg, COUNT(*) AS cnt "
                        "FROM oj_o o INNER JOIN oj_q q ON o.id = q.oid RIGHT JOIN oj_p p ON o.pid = p.id "
                        "GROUP BY o.ok, p.pg, q.qg",
                        [
                            "DELETE FROM oj_q WHERE id = 100",
                            "INSERT INTO oj_q VALUES (100, 10, 'Q1')",
                        ],
                    ),
                ]
            )
        try:
            for vname, tbls, ddls, seeds, view_sql, ops in cases:
                with self.subTest(view=vname, dialect=dialect.value):
                    try:
                        with eng.begin() as conn:
                            ms.MaterializedViewManager(conn, dialect=dialect).drop_view(vname)
                    except Exception:
                        pass
                    with eng.begin() as conn:
                        conn.exec_driver_sql(f"DROP VIEW IF EXISTS {vname}")
                        conn.exec_driver_sql(f"DROP TABLE IF EXISTS __matview_store_{vname}")
                        for t in tbls:
                            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {t}")
                        for d in ddls:
                            conn.exec_driver_sql(d)
                        for s in seeds:
                            conn.exec_driver_sql(s)
                    try:
                        with eng.begin() as conn:
                            mgr = ms.MaterializedViewManager(
                                conn,
                                dialect=dialect,
                                ignore_limitations=(dialect == ms.Dialect.MYSQL),
                            )
                            plan = mgr.register_view(f"CREATE VIEW {vname} AS {view_sql};")
                            self.assertEqual(plan.strategy, ms.MaintenanceStrategy.GROUP_SCOPED_DELTA)
                            for op in ops:
                                conn.exec_driver_sql(op)
                                mv_rows = sorted(conn.exec_driver_sql(f"SELECT * FROM {vname}").fetchall(), key=repr)
                                live_rows = sorted(conn.exec_driver_sql(view_sql).fetchall(), key=repr)
                                self.assertEqual(mv_rows, live_rows, f"{vname} ({op}): MV={mv_rows} LIVE={live_rows}")
                    finally:
                        with eng.begin() as conn:
                            ms.MaterializedViewManager(conn, dialect=dialect).drop_view(vname)
                        with eng.begin() as conn:
                            for t in tbls:
                                conn.exec_driver_sql(f"DROP TABLE IF EXISTS {t}")
        finally:
            eng.dispose()

    def test_native_postgres_outer_join_scoping(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_native_outer_join_matrix(dsn, ms.Dialect.POSTGRES)

    def test_native_mysql_outer_join_scoping(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_native_outer_join_matrix(dsn, ms.Dialect.MYSQL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
