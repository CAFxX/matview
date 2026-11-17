#!/usr/bin/env python3
"""Tests for P2 partition-scoped (Tier 2) materialized view maintenance for window functions."""

from __future__ import annotations

import os
import sqlite3
import unittest

import sqlalchemy as sa

import matview_sync as ms


def _engine_url_for_dialect(dsn: str, dialect: ms.Dialect):
    """Choose an installed DB-API driver when a DSN leaves it implicit."""
    url = sa.engine.make_url(dsn)
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

        # PostgreSQL transition-table vectorized codegen stays scoped to __mv_new_rows / __mv_old_rows
        p_pg = ms.generate_materialized_view(
            self.SCHEMA,
            "CREATE VIEW mv_pg_win AS SELECT id, team, LAG(score, 1) OVER (PARTITION BY team ORDER BY id) AS prev_s FROM scores;",
            dialect=ms.Dialect.POSTGRES,
            pg_use_transition_tables=True,
            pg_vectorize_transition_tables=True,
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
                    pg_use_transition_tables=(dialect == ms.Dialect.POSTGRES),
                    pg_vectorize_transition_tables=True,
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
            with eng.begin() as conn:
                conn.exec_driver_sql("DROP VIEW IF EXISTS mv_p2_win")
                conn.exec_driver_sql("DROP TABLE IF EXISTS __matview_store_mv_p2_win")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
