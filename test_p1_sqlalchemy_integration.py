#!/usr/bin/env python3
"""
Tests for SQLAlchemy unified connectivity, DB-API normalization, parameter
binding, savepoint helpers, and cross-engine schema introspection.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import sqlalchemy as sa

import matview_sync as ms


class TestSqlAlchemyConnectivityAndIntrospection(unittest.TestCase):
    def test_url_normalization_variants(self) -> None:
        cases = [
            ("sqlite:///:memory:", "sqlite", ms.Dialect.SQLITE, {}),
            ("/tmp/example_matview.db", "sqlite", ms.Dialect.SQLITE, {}),
            ("postgres://u:p@localhost:5432/db", "postgresql", ms.Dialect.POSTGRES, {}),
            ("postgresql://u:p@localhost:5432/db", "postgresql", ms.Dialect.POSTGRES, {}),
            ("postgresql+psycopg2://u:p@localhost:5432/db", "postgresql", ms.Dialect.POSTGRES, {}),
            ("mysql://root:pw@127.0.0.1:3306/db", "mysql", ms.Dialect.MYSQL, {}),
            ("mariadb://root:pw@127.0.0.1:3306/db", "mysql", ms.Dialect.MYSQL, {}),
            (
                "mysql://root:pw@localhost/db?unix_socket=/var/run/mysqld/mysqld.sock",
                "mysql",
                ms.Dialect.MYSQL,
                {"unix_socket": "/var/run/mysqld/mysqld.sock"},
            ),
        ]
        for raw_url, exp_backend, exp_dialect, exp_cargs in cases:
            sa_url, dial, cargs = ms.normalize_sqlalchemy_url(raw_url)
            self.assertEqual(sa_url.get_backend_name(), exp_backend, raw_url)
            self.assertEqual(dial, exp_dialect, raw_url)
            self.assertEqual(cargs, exp_cargs, raw_url)

    def test_render_parameterized_sql_preserves_quoted_question_marks(self) -> None:
        sql = "SELECT '?' AS q_lit, \"col?\" FROM tbl WHERE a = ? AND b = ?"
        self.assertEqual(ms.render_parameterized_sql(sql, ms.Dialect.SQLITE), sql)
        expected_pg = "SELECT '?' AS q_lit, \"col?\" FROM tbl WHERE a = %s AND b = %s"
        self.assertEqual(ms.render_parameterized_sql(sql, ms.Dialect.POSTGRES), expected_pg)
        self.assertEqual(ms.render_parameterized_sql(sql, ms.Dialect.MYSQL), expected_pg)

    def test_sqlite_file_url_engine_and_connection_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "mv_sa.db")
            url = f"sqlite:///{db_file}"

            # 1. Manager opened from SQLite URL via SQLAlchemy create_engine
            mgr_url = ms.MaterializedViewManager(url)
            try:
                cur = mgr_url.conn.cursor()
                cur.execute(
                    "CREATE TABLE parent (id INTEGER PRIMARY KEY, label VARCHAR(32) NOT NULL)"
                )
                cur.execute(
                    "CREATE TABLE orders ("
                    "id INTEGER PRIMARY KEY, "
                    "parent_id INTEGER REFERENCES parent(id) ON DELETE CASCADE, "
                    "cat TEXT NOT NULL, "
                    "amount DECIMAL(12, 4))"
                )
                cur.execute("INSERT INTO parent VALUES (1, 'p1'), (2, 'p2')")
                cur.execute("INSERT INTO orders VALUES (1, 1, 'a', 10.25), (2, 2, 'a', 5.75)")
                mgr_url.conn.commit()
                cur.close()

                mgr_url.register_view(
                    "CREATE VIEW v_cat AS SELECT cat, COUNT(*) AS cnt, SUM(amount) AS total FROM orders GROUP BY cat",
                    populate=True,
                    incremental_backfill=True,
                    batch_size=1,
                    ignore_limitations=True,
                )
                with mgr_url.bulk_changes():
                    c2 = mgr_url.conn.cursor()
                    c2.execute("INSERT INTO orders VALUES (3, 1, 'b', 20.00)")
                    c2.close()
            finally:
                mgr_url.close()

            # 2. Manager constructed directly from sqlalchemy.Engine
            engine = sa.create_engine(url)
            try:
                schemas = ms.introspect_database_schemas(engine)
                self.assertIn("orders", schemas)
                self.assertIn("parent", schemas)
                self.assertNotIn("__matview_catalog", schemas)
                self.assertEqual(schemas["orders"].columns["amount"].data_type, "DECIMAL(12, 4)")
                self.assertTrue(schemas["orders"].columns["cat"].not_null)
                self.assertTrue(schemas["orders"].columns["id"].primary_key)
                self.assertIn("ON DELETE CASCADE", schemas["orders"].fk_cascade_actions)

                mgr_eng = ms.MaterializedViewManager(engine)
                try:
                    c3 = mgr_eng.conn.cursor()
                    c3.execute("SELECT cat, cnt, total FROM v_cat ORDER BY cat")
                    self.assertEqual(c3.fetchall(), [("a", 2, 16.0), ("b", 1, 20.0)])
                    c3.close()
                finally:
                    mgr_eng.close()

                # 3. Manager constructed from active sqlalchemy.Connection
                with engine.begin() as sa_conn:
                    mgr_conn = ms.MaterializedViewManager(sa_conn)
                    schemas_from_conn = mgr_conn.fetch_schema()
                    self.assertIn("orders", schemas_from_conn)
                    c4 = mgr_conn.conn.cursor()
                    c4.execute("INSERT INTO orders VALUES (4, 1, 'b', 5.0)")
                    c4.close()

                with engine.connect() as verify_conn:
                    rows = verify_conn.execute(
                        sa.text("SELECT cat, cnt, total FROM v_cat ORDER BY cat")
                    ).fetchall()
                    self.assertEqual([tuple(r) for r in rows], [("a", 2, 16.0), ("b", 2, 25.0)])
            finally:
                engine.dispose()

    def test_raw_sqlite_conn_remains_open_after_sqlalchemy_inspection(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name VARCHAR(24), price DECIMAL(10, 2))")
            conn.execute("INSERT INTO items VALUES (1, 'x', 12.34)")
            schemas = ms.introspect_schemas_with_sqlalchemy(conn, dialect="sqlite")
            self.assertIn("items", schemas)
            self.assertEqual(schemas["items"].columns["price"].data_type, "DECIMAL(10, 2)")
            # Caller connection must stay open after inspection
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0], 1)
        finally:
            conn.close()

    def test_live_postgres_or_mysql_sqlalchemy_url(self) -> None:
        pg_dsn = os.environ.get("POSTGRES_DSN", "").strip()
        my_dsn = os.environ.get("MYSQL_DSN", "").strip()
        if not pg_dsn and not my_dsn:
            self.skipTest("Neither POSTGRES_DSN nor MYSQL_DSN is set")

        for dsn, expected_dialect in ((pg_dsn, ms.Dialect.POSTGRES), (my_dsn, ms.Dialect.MYSQL)):
            if not dsn:
                continue
            mgr = ms.MaterializedViewManager(dsn)
            try:
                self.assertEqual(mgr.dialect, expected_dialect)
                t_name = f"sa_it_tbl_{expected_dialect.value}"
                v_name = f"sa_it_v_{expected_dialect.value}"
                with mgr._cursor() as cur:
                    mgr.drop_view(v_name)
                    cur.execute(f"DROP TABLE IF EXISTS {t_name}")
                    cur.execute(
                        f"CREATE TABLE {t_name} ("
                        "id INT PRIMARY KEY, "
                        "grp VARCHAR(24) NOT NULL, "
                        "amt DECIMAL(14, 4))"
                    )
                    cur.execute(f"INSERT INTO {t_name} VALUES (1, 'g1', 10.2500), (2, 'g1', 2.7500)")
                    mgr._commit()
                schemas = mgr.fetch_schema()
                self.assertIn(t_name, schemas)
                self.assertEqual(schemas[t_name].columns["amt"].data_type, "DECIMAL(14, 4)")
                mgr.register_view(
                    f"CREATE VIEW {v_name} AS SELECT grp, COUNT(*) AS cnt, SUM(amt) AS total FROM {t_name} GROUP BY grp",
                    populate=True,
                    ignore_limitations=True,
                )
                with mgr._cursor() as cur:
                    cur.execute(f"SELECT grp, cnt, total FROM {v_name}")
                    row = cur.fetchone()
                    self.assertEqual(row[0], "g1")
                    self.assertEqual(int(row[1]), 2)
                # End this read transaction: drop_view refuses to run inside a
                # caller-owned transaction.
                mgr._rollback()
                mgr.drop_view(v_name)
                with mgr._cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {t_name}")
                    mgr._commit()
            finally:
                mgr.close()


if __name__ == "__main__":
    unittest.main()
