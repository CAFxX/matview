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

    def test_postgres_array_sa_type_formatting_and_codegen(self) -> None:
        from sqlalchemy.dialects import postgresql as sa_pg
        from sqlalchemy.types import Enum as SaEnum, NullType

        from matview.connectivity import _format_sa_column_type

        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.NUMERIC()), ms.Dialect.POSTGRES),
            "NUMERIC[]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.NUMERIC(10, 2)), ms.Dialect.POSTGRES),
            "DECIMAL(10, 2)[]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.INTEGER(), dimensions=2), ms.Dialect.POSTGRES),
            "INTEGER[][]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.VARCHAR()), ms.Dialect.POSTGRES),
            "VARCHAR[]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.CHAR()), ms.Dialect.POSTGRES),
            "BPCHAR[]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.TIMESTAMP(timezone=True)), ms.Dialect.POSTGRES),
            "TIMESTAMP WITH TIME ZONE[]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.TEXT(collation="C")), ms.Dialect.POSTGRES),
            'TEXT[] COLLATE "C"',
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.TEXT(collation="default")), ms.Dialect.POSTGRES),
            "TEXT[]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(sa_pg.ENUM("a", "b", name="MixedEnum")), ms.Dialect.POSTGRES),
            '"MixedEnum"[]',
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(SaEnum("a", "b", name="GenericEnum")), ms.Dialect.POSTGRES),
            '"GenericEnum"[]',
        )
        self.assertEqual(
            _format_sa_column_type(
                sa_pg.ARRAY(sa_pg.ENUM("a", "b", name="my_enum", schema="custom_sch")),
                ms.Dialect.POSTGRES,
            ),
            '"custom_sch"."my_enum"[]',
        )
        self.assertEqual(
            _format_sa_column_type(
                sa_pg.ARRAY(sa_pg.DOMAIN("pos_int", sa_pg.INTEGER())),
                ms.Dialect.POSTGRES,
            ),
            "INTEGER[]",
        )
        self.assertEqual(
            _format_sa_column_type(
                sa_pg.DOMAIN("int_arr_dom", sa_pg.ARRAY(sa_pg.INTEGER())),
                ms.Dialect.POSTGRES,
            ),
            "INTEGER[]",
        )
        self.assertEqual(
            _format_sa_column_type(sa_pg.ARRAY(NullType()), ms.Dialect.POSTGRES),
            "NULL",
        )

        # Offline DDL codegen preserves PostgreSQL array types per column for columns & expressions,
        # without regressing SQLite/MySQL CAST inference.
        schema = (
            "CREATE TABLE arr_src ("
            "id INT PRIMARY KEY, "
            "k NUMERIC[], "
            "k_ps NUMERIC(10, 2)[], "
            "tags VARCHAR[], "
            "c3 CHAR(3)[], "
            "vals DOUBLE PRECISION[], "
            "m2d INT[][], "
            "v INT NOT NULL"
            ");"
        )
        plan = ms.generate_materialized_view(
            schema,
            "CREATE VIEW v_arr_expr AS "
            "SELECT k, tags, vals, COALESCE(k, ARRAY[]::NUMERIC[]) AS k_co, "
            "CASE WHEN id > 0 THEN k_ps ELSE k END AS k_case, "
            "k[1] AS first_k, k[1:2] AS slice_k, m2d[1][1] AS m_elem, "
            "tags[1] AS tag_elem, c3[1] AS c3_elem, ARRAY[id, v] AS pair, "
            "ARRAY[1, '2'] AS mix_a, ARRAY[NULL] AS null_a, "
            "COUNT(*) AS n FROM arr_src "
            "GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13;",
            dialect=ms.Dialect.POSTGRES,
        )
        ddl = plan.backing_table_ddl
        self.assertNotIn(" ARRAY ", ddl)
        self.assertIn("k DECIMAL[]", ddl)
        self.assertIn("tags VARCHAR[]", ddl)
        self.assertIn("vals DOUBLE PRECISION[]", ddl)
        self.assertIn("k_co NUMERIC[]", ddl)
        self.assertIn("k_case NUMERIC[]", ddl)
        self.assertIn("first_k NUMERIC", ddl)
        self.assertIn("slice_k NUMERIC[]", ddl)
        self.assertIn("m_elem INTEGER", ddl)
        self.assertIn("tag_elem VARCHAR", ddl)
        self.assertIn("c3_elem BPCHAR", ddl)
        self.assertIn("pair INTEGER[]", ddl)
        self.assertIn("mix_a INTEGER[]", ddl)
        self.assertIn("null_a TEXT[]", ddl)

        # Quoted PostgreSQL identifiers (collations like COLLATE "C", mixed-case enum casts)
        # in GROUP BY expressions must keep their double quotes when rewritten.
        coll_plan = ms.generate_materialized_view(
            schema,
            'CREATE VIEW v_arr_coll AS SELECT tags COLLATE "C" AS k_coll, COUNT(*) AS n '
            'FROM arr_src GROUP BY tags COLLATE "C";',
            dialect=ms.Dialect.POSTGRES,
        )
        self.assertIn("k_coll VARCHAR[]", coll_plan.backing_table_ddl)
        self.assertIn('COLLATE "C"', coll_plan.initial_populate_sql)
        self.assertNotIn("COLLATE C,", coll_plan.initial_populate_sql)
        self.assertNotIn("COLLATE C;", coll_plan.initial_populate_sql)

        # Ensure CAST handling on SQLite and MySQL did not regress.
        sq_plan = ms.generate_materialized_view(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, x TEXT);",
            "CREATE VIEW v_sq_cast AS SELECT id, CAST(x AS NUMERIC) AS c FROM t;",
            dialect=ms.Dialect.SQLITE,
        )
        self.assertIn("c BLOB", sq_plan.backing_table_ddl)
        my_plan = ms.generate_materialized_view(
            "CREATE TABLE t (id INT PRIMARY KEY, x INT);",
            "CREATE VIEW v_my_cast AS SELECT id, CAST(x AS UNSIGNED) AS c FROM t;",
            dialect=ms.Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertNotIn("c BIGINT", my_plan.backing_table_ddl)

        # PostgreSQL BIT(n) / BIT VARYING SQLAlchemy formatting and BIT_AND/BIT_OR codegen.
        self.assertEqual(_format_sa_column_type(sa_pg.BIT(length=4), ms.Dialect.POSTGRES), "BIT(4)")
        self.assertEqual(_format_sa_column_type(sa_pg.BIT(varying=True), ms.Dialect.POSTGRES), "BIT VARYING")
        self.assertEqual(
            _format_sa_column_type(sa_pg.BIT(length=8, varying=True), ms.Dialect.POSTGRES),
            "BIT VARYING(8)",
        )
        bit_plan = ms.generate_materialized_view(
            "CREATE TABLE tb (id INT PRIMARY KEY, g INT, s SMALLINT, ss SMALLSERIAL, "
            "s2 SERIAL2, i INT, s4 SERIAL4, b BIGINT, bf BIT(4), vb VARBIT);",
            "CREATE VIEW v_bit AS SELECT g, BIT_AND(s) AS ba_s, BIT_OR(ss) AS bo_ss, "
            "BIT_AND(s2) AS ba_s2, BIT_AND(i) AS ba_i, BIT_OR(s4) AS bo_s4, BIT_OR(b) AS bo_b, "
            "BIT_AND(bf) AS ba_bf, BIT_OR(vb) AS bo_vb, "
            "BIT_AND(COALESCE(bf, B'1111')) AS ba_co_bf, "
            "BIT_AND(bf & B'1111') AS ba_expr_bf "
            "FROM tb GROUP BY g;",
            dialect=ms.Dialect.POSTGRES,
        )
        b_ddl = bit_plan.backing_table_ddl
        self.assertIn("ba_s SMALLINT", b_ddl)
        self.assertIn("bo_ss SMALLINT", b_ddl)
        self.assertIn("ba_s2 SMALLINT", b_ddl)
        self.assertIn("ba_i INTEGER", b_ddl)
        self.assertIn("bo_s4 INTEGER", b_ddl)
        self.assertIn("bo_b BIGINT", b_ddl)
        self.assertIn("ba_bf BIT(4)", b_ddl)
        self.assertIn("bo_vb BIT VARYING", b_ddl)
        self.assertIn("ba_co_bf BIT(4)", b_ddl)
        self.assertIn("ba_expr_bf BIT(4)", b_ddl)

    def test_native_special_char_view_names_and_pg_bit_aggregates(self) -> None:
        pg_dsn = os.environ.get("POSTGRES_DSN", "").strip()
        my_dsn = os.environ.get("MYSQL_DSN", "").strip()
        if not pg_dsn and not my_dsn:
            self.skipTest("Neither POSTGRES_DSN nor MYSQL_DSN is set")

        if pg_dsn:
            import psycopg2
            from unittest import mock

            import matview.connectivity as _conn_mod

            conn = psycopg2.connect(pg_dsn)
            cur = conn.cursor()
            ns = f"__mv_vname_pg_{os.getpid()}"
            try:
                cur.execute(f'DROP SCHEMA IF EXISTS "{ns}" CASCADE; CREATE SCHEMA "{ns}"; SET search_path TO "{ns}";')
                cur.execute(
                    'CREATE TABLE "order items" ('
                    "id INT PRIMARY KEY, g INT, v INT, s SMALLINT, b BIGINT, bf BIT(4), vb VARBIT)"
                )
                cur.execute(
                    'INSERT INTO "order items" VALUES '
                    "(1, 1, 10, 7, 31, B'1010', B'110'), "
                    "(2, 1, 20, 3, 14, B'1100', B'101'), "
                    "(3, 2, 5, 5, 9, B'0110', B'011')"
                )
                conn.commit()
                mgr = ms.MaterializedViewManager(conn, dialect=ms.Dialect.POSTGRES)

                # Verify both SQLAlchemy reflection and raw DB-API fallback preserve BIT(4) width.
                sa_sc = mgr.fetch_schema()
                self.assertEqual(sa_sc["order items"].columns["bf"].data_type, "BIT(4)")
                self.assertEqual(sa_sc["order items"].columns["vb"].data_type, "BIT VARYING")
                with mock.patch.object(_conn_mod, "_can_use_sqlalchemy_inspector", lambda _c: False):
                    raw_sc = mgr.fetch_schema()
                self.assertEqual(raw_sc["order items"].columns["bf"].data_type, "BIT(4)")
                self.assertEqual(raw_sc["order items"].columns["vb"].data_type, "BIT VARYING")

                pg_names = [
                    "sales summary",
                    "sales_summary",
                    "Sales Summary",
                    "SalesSummary",
                    "sales-summary",
                    "user's view",
                    'a "b" c',
                    "my.view",
                    "---",
                ]
                for vn in pg_names:
                    qv = '"' + vn.replace('"', '""') + '"'
                    mgr.register_view(
                        f'CREATE VIEW {qv} AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM "order items" GROUP BY g'
                    )
                sel_bit = (
                    "SELECT g, BIT_AND(s) AS ba_s, BIT_OR(s) AS bo_s, BIT_XOR(s) AS bx_s, "
                    "BIT_AND(v) AS ba_i, BIT_OR(v) AS bo_i, BIT_XOR(v) AS bx_i, "
                    "BIT_AND(b) AS ba_b, BIT_OR(b) AS bo_b, BIT_XOR(b) AS bx_b, "
                    "BIT_AND(bf) AS ba_bf, BIT_OR(bf) AS bo_bf, BIT_XOR(bf) AS bx_bf, "
                    "BIT_AND(COALESCE(bf, B'1111')) AS ba_co_bf, "
                    "BIT_AND(bf & B'1111') AS ba_expr_bf, "
                    'BIT_AND(vb) AS ba_vb FROM "order items" GROUP BY g'
                )
                mgr.register_view(f'CREATE VIEW "bit summary" AS {sel_bit}')

                cur.execute('INSERT INTO "order items" VALUES (4, 2, 15, 1, 7, B\'0101\', B\'001\')')
                cur.execute('UPDATE "order items" SET v = 30, bf = B\'1110\' WHERE id = 1')
                cur.execute('DELETE FROM "order items" WHERE id = 3')
                conn.commit()

                cur.execute('SELECT g, SUM(v), COUNT(*) FROM "order items" GROUP BY g ORDER BY g')
                expected = cur.fetchall()
                conn.commit()

                for vn in pg_names:
                    qv = vn if vn == "SalesSummary" else '"' + vn.replace('"', '""') + '"'
                    cur.execute(f"SELECT g, s, c FROM {qv} ORDER BY g")
                    self.assertEqual(cur.fetchall(), expected, vn)
                    conn.commit()
                    with self.assertRaises(Exception):
                        cur.execute(f"DELETE FROM {qv}")
                    conn.rollback()
                    self.assertEqual(mgr.check_schema_drift()[vn], "ok", vn)
                    mgr.repair_view(vn)
                    self.assertTrue(mgr.drop_view(vn))

                cur.execute('SELECT * FROM "bit summary" ORDER BY g')
                mv_bit = cur.fetchall()
                mv_oids = {d.name: d.type_code for d in cur.description}
                cur.execute(f"SELECT * FROM ({sel_bit}) AS q ORDER BY g")
                live_bit = cur.fetchall()
                live_oids = {d.name: d.type_code for d in cur.description}
                conn.commit()
                self.assertEqual(mv_bit, live_bit)
                for col in (
                    "ba_s", "bo_s", "bx_s",
                    "ba_i", "bo_i", "bx_i",
                    "ba_b", "bo_b", "bx_b",
                    "ba_bf", "bo_bf", "bx_bf",
                    "ba_co_bf", "ba_expr_bf",
                ):
                    self.assertEqual(mv_oids[col], live_oids[col], (col, mv_oids[col], live_oids[col]))
                self.assertEqual(mv_oids["ba_vb"], 1562)
                self.assertTrue(mgr.drop_view("bit summary"))
            finally:
                try:
                    conn.rollback()
                    cur.execute(f'DROP SCHEMA IF EXISTS "{ns}" CASCADE')
                    conn.commit()
                finally:
                    cur.close()
                    conn.close()

        if my_dsn:
            mgr = ms.MaterializedViewManager(my_dsn)
            t_name = f"order items {os.getpid()}"
            my_names = [
                f"sales summary {os.getpid()}",
                f"sales-summary-{os.getpid()}",
                f"user's view {os.getpid()}",
            ]
            try:
                for vn in my_names:
                    mgr.drop_view(vn)
                with mgr._cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS `{t_name}`")
                    cur.execute(f"CREATE TABLE `{t_name}` (id INT PRIMARY KEY, g INT, v INT)")
                    cur.execute(f"INSERT INTO `{t_name}` VALUES (1, 1, 10), (2, 1, 20), (3, 2, 5)")
                    mgr._commit()
                for vn in my_names:
                    mgr.register_view(
                        f"CREATE VIEW `{vn}` AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM `{t_name}` GROUP BY g",
                        ignore_limitations=True,
                    )
                with mgr._cursor() as cur:
                    cur.execute(f"INSERT INTO `{t_name}` VALUES (4, 2, 15)")
                    cur.execute(f"UPDATE `{t_name}` SET v = 30 WHERE id = 1")
                    cur.execute(f"DELETE FROM `{t_name}` WHERE id = 3")
                    mgr._commit()
                    cur.execute(f"SELECT g, SUM(v), COUNT(*) FROM `{t_name}` GROUP BY g ORDER BY g")
                    expected = cur.fetchall()
                    mgr._rollback()
                for vn in my_names:
                    with mgr._cursor() as cur:
                        cur.execute(f"SELECT g, s, c FROM `{vn}` ORDER BY g")
                        self.assertEqual(cur.fetchall(), expected, vn)
                        mgr._rollback()
                    self.assertEqual(mgr.check_schema_drift()[vn], "ok", vn)
                    mgr.repair_view(vn)
                    self.assertTrue(mgr.drop_view(vn))
            finally:
                try:
                    mgr._rollback()
                    for vn in my_names:
                        try:
                            mgr.drop_view(vn)
                        except Exception:
                            pass
                    with mgr._cursor() as cur:
                        cur.execute(f"DROP TABLE IF EXISTS `{t_name}`")
                        mgr._commit()
                finally:
                    mgr.close()


if __name__ == "__main__":
    unittest.main()
