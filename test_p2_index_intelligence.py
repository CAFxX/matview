#!/usr/bin/env python3
"""Tests for P2 index intelligence: base index introspection, leveraging, recommendations, and aux/backing auto-indexes."""

from __future__ import annotations

import os
import sqlite3
import unittest

import sqlalchemy as sa

import matview_sync as ms


class TestBaseTableIndexIntrospectionAndLeveraging(unittest.TestCase):
    def test_parse_table_schema_captures_indexes_unique_and_composite_pk_order(self) -> None:
        ddl = """
        CREATE TABLE orders (
            tenant_id INTEGER NOT NULL,
            order_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            region TEXT NOT NULL,
            code VARCHAR(16) UNIQUE,
            PRIMARY KEY (tenant_id, order_id),
            UNIQUE (region, customer_id)
        );
        CREATE INDEX idx_orders_cust_region ON orders (customer_id, region);
        CREATE INDEX idx_orders_partial ON orders (region) WHERE tenant_id > 0;
        """
        schemas = ms.parse_table_schema(ddl, dialect=ms.Dialect.POSTGRES)
        orders = schemas["orders"]

        self.assertEqual(orders.pk_columns, ("tenant_id", "order_id"))
        self.assertTrue(orders.has_index_covering(["tenant_id"]))
        self.assertTrue(orders.has_index_covering(["tenant_id", "order_id"]))
        # Trailing PK column is not a leftmost prefix
        self.assertFalse(orders.has_index_covering(["order_id"]))

        # Secondary index prefix checks
        self.assertTrue(orders.has_index_covering(["customer_id"]))
        self.assertTrue(orders.has_index_covering(["customer_id", "region"]))
        self.assertIn("idx_orders_cust_region", orders.covering_indexes_for(["customer_id"]))

        # Partial index should not satisfy unconditional full-table lookups unless partial_ok=True
        self.assertTrue(orders.has_index_covering(["region"]))  # via table UNIQUE (region, customer_id)
        self.assertEqual(orders.covering_indexes_for(["code"]), ("uq_orders_code",))

        ddl_only_partial = """
        CREATE TABLE events (id INTEGER PRIMARY KEY, region TEXT);
        CREATE INDEX idx_ev_partial ON events (region) WHERE id > 0;
        """
        ev_schema = ms.parse_table_schema(ddl_only_partial, dialect=ms.Dialect.POSTGRES)["events"]
        self.assertFalse(ev_schema.has_index_covering(["region"]))
        self.assertTrue(ev_schema.has_index_covering(["region"], partial_ok=True))

    def test_introspect_sqlite_and_sqlalchemy_capture_indexes(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(
                """
                CREATE TABLE depts (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE emps (
                    id INTEGER PRIMARY KEY,
                    dept_id INTEGER NOT NULL,
                    email TEXT UNIQUE,
                    salary INTEGER NOT NULL
                );
                CREATE INDEX idx_emps_dept ON emps (dept_id);
                """
            )
            schemas = ms.introspect_database_schemas(conn, dialect=ms.Dialect.SQLITE)
            emps = schemas["emps"]
            self.assertTrue(emps.has_index_covering(["dept_id"]))
            self.assertIn("idx_emps_dept", emps.covering_indexes_for(["dept_id"]))
            self.assertTrue(emps.has_index_covering(["email"]))

            # Also verify PRAGMA fallback path on SQLite when DDL is bypassed or via SQLAlchemy Engine
            eng = sa.create_engine(
                "sqlite:///:memory:",
                creator=lambda: conn,
            )
            # Use a fresh file-less SA engine bound to its own connection to test SA inspector directly
        finally:
            conn.close()

        eng = sa.create_engine("sqlite:///:memory:")
        try:
            with eng.begin() as sa_conn:
                sa_conn.exec_driver_sql(
                    "CREATE TABLE items (tenant INTEGER NOT NULL, id INTEGER NOT NULL, grp TEXT, val INT, PRIMARY KEY (tenant, id));"
                )
                sa_conn.exec_driver_sql("CREATE INDEX idx_items_grp_val ON items (grp, val);")
                sa_schemas = ms.introspect_database_schemas(sa_conn, dialect=ms.Dialect.SQLITE)
                items = sa_schemas["items"]
                self.assertEqual(items.pk_columns, ("tenant", "id"))
                self.assertTrue(items.has_index_covering(["grp"]))
                self.assertTrue(items.has_index_covering(["grp", "val"]))
                self.assertFalse(items.has_index_covering(["val"]))
        finally:
            eng.dispose()

    def test_introspect_postgres_and_mysql_metadata_queries_capture_indexes(self) -> None:
        class _FakeCursor:
            def __init__(self, dialect_tag: str) -> None:
                self.dialect_tag = dialect_tag
                self._rows = []

            def execute(self, sql: str) -> None:
                low = sql.lower()
                if self.dialect_tag == "postgres":
                    if "constraint_type = 'primary key'" in low:
                        self._rows = [("orders", "id")]
                    elif "constraint_type = 'foreign key'" in low:
                        self._rows = []
                    elif "from pg_indexes" in low:
                        self._rows = [
                            ("orders", "orders_pkey", "CREATE UNIQUE INDEX orders_pkey ON orders USING btree (id)"),
                            ("orders", "idx_orders_dept", "CREATE INDEX idx_orders_dept ON orders USING btree (dept_id)"),
                        ]
                    elif "from information_schema.columns" in low:
                        self._rows = [
                            ("orders", "id", "integer", "NO", None),
                            ("orders", "dept_id", "integer", "NO", None),
                        ]
                else:
                    if "referential_constraints" in low:
                        self._rows = []
                    elif "information_schema.statistics" in low:
                        self._rows = [
                            ("orders", "PRIMARY", 0, "id"),
                            ("orders", "idx_orders_dept", 1, "dept_id"),
                        ]
                    elif "information_schema.columns" in low:
                        self._rows = [
                            ("orders", "id", "int", "NO", "PRI", "int"),
                            ("orders", "dept_id", "int", "NO", "MUL", "int"),
                        ]

            def fetchall(self):
                return list(self._rows)

            def close(self) -> None:
                pass

        class _FakeConn:
            def __init__(self, dialect_tag: str) -> None:
                self._dialect_tag = dialect_tag

            def cursor(self):
                return _FakeCursor(self._dialect_tag)

        pg_schemas = ms.introspect_database_schemas(_FakeConn("postgres"), dialect=ms.Dialect.POSTGRES)
        self.assertTrue(pg_schemas["orders"].has_index_covering(["dept_id"]))
        self.assertIn("idx_orders_dept", pg_schemas["orders"].covering_indexes_for(["dept_id"]))

        my_schemas = ms.introspect_database_schemas(_FakeConn("mysql"), dialect=ms.Dialect.MYSQL)
        self.assertEqual(my_schemas["orders"].pk_columns, ("id",))
        self.assertTrue(my_schemas["orders"].has_index_covering(["dept_id"]))
        self.assertIn("idx_orders_dept", my_schemas["orders"].covering_indexes_for(["dept_id"]))


class TestRecommendedBaseIndexesAndAutoIndexes(unittest.TestCase):
    def test_join_proposes_missing_base_index_and_leverages_existing_index(self) -> None:
        base_schema = """
        CREATE TABLE depts (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE emps (id INTEGER PRIMARY KEY, dept_id INTEGER NOT NULL, salary INTEGER NOT NULL);
        """
        view_sql = """
        CREATE VIEW mv_dept_sal AS
        SELECT d.name AS dept_name, COUNT(*) AS emp_cnt, SUM(e.salary) AS total_sal
        FROM emps e
        INNER JOIN depts d ON e.dept_id = d.id
        GROUP BY d.name;
        """
        plan_unindexed = ms.generate_materialized_view(base_schema, view_sql, dialect=ms.Dialect.SQLITE)
        # depts.id is already PK (leveraged); emps.dept_id is unindexed (join key,
        # recommended). SQLite SUM over a join is recomputed per group from the
        # base tables, so the group key depts.name is recommended as well.
        recs = {r["index_name"]: r for r in plan_unindexed.diagnostics.details["base_index_recommendations"]}
        self.assertEqual(
            {(r["table"], tuple(r["columns"]), r["reason"]) for r in recs.values()},
            {
                ("emps", ("dept_id",), "join_on_key (INNER JOIN)"),
                ("depts", ("name",), "sqlite_native_sum_avg_group_recompute"),
            },
        )
        self.assertEqual(len(plan_unindexed.recommended_base_indexes), 2)
        used_tables = {u["table"] for u in plan_unindexed.diagnostics.details["existing_base_indexes_used"]}
        self.assertIn("depts", used_tables)

        # Executing the recommended base-table DDL on live SQLite succeeds
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(base_schema)
            for rec_ddl in plan_unindexed.recommended_base_indexes:
                conn.execute(rec_ddl)
            conn.executescript(plan_unindexed.render_sql_script())
        finally:
            conn.close()

        # Recompiling with existing index eliminates recommendation and records leverage
        indexed_schema = base_schema + "\nCREATE INDEX idx_emps_dept ON emps (dept_id);\nCREATE INDEX idx_depts_name ON depts (name);"
        plan_indexed = ms.generate_materialized_view(indexed_schema, view_sql, dialect=ms.Dialect.SQLITE)
        self.assertEqual(plan_indexed.recommended_base_indexes, [])
        used_idx_names = {
            name
            for entry in plan_indexed.diagnostics.details["existing_base_indexes_used"]
            for name in entry["indexes"]
        }
        self.assertIn("idx_emps_dept", used_idx_names)

    def test_minmax_and_tier2_propose_group_base_indexes_and_skip_mysql_text(self) -> None:
        schema_sql = """
        CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT NOT NULL, amount INTEGER NOT NULL);
        """
        view_sql = """
        CREATE VIEW mv_region_minmax AS
        SELECT region, MIN(amount) AS min_amt, MAX(amount) AS max_amt
        FROM sales
        GROUP BY region;
        """
        plan_sq = ms.generate_materialized_view(schema_sql, view_sql, dialect=ms.Dialect.SQLITE)
        self.assertTrue(
            any("sales" in ddl and "region" in ddl for ddl in plan_sq.recommended_base_indexes),
            f"Expected base index recommendation on sales(region), got {plan_sq.recommended_base_indexes}",
        )

        # MySQL cannot create unprefixed indexes on TEXT columns
        plan_my = ms.generate_materialized_view(
            schema_sql, view_sql, dialect=ms.Dialect.MYSQL, ignore_limitations=True
        )
        self.assertEqual(plan_my.recommended_base_indexes, [])

    def test_auto_indexes_on_distinct_aux_table_and_backing_tables(self) -> None:
        schema_sql = """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            tenant_id INTEGER NOT NULL,
            region TEXT NOT NULL,
            customer_id INTEGER NOT NULL,
            amount INTEGER NOT NULL
        );
        """
        view_distinct = """
        CREATE VIEW mv_dist_cust AS
        SELECT tenant_id, region, COUNT(DISTINCT customer_id) AS dist_custs
        FROM orders
        GROUP BY tenant_id, region;
        """
        plan = ms.generate_materialized_view(schema_sql, view_distinct, dialect=ms.Dialect.SQLITE)
        # Should automatically index trailing group col on __matview_dist_* and __val
        self.assertTrue(any("__matview_dist_" in ddl and "region" in ddl for ddl in plan.index_ddls))
        self.assertTrue(any("__matview_dist_" in ddl and "__val" in ddl for ddl in plan.index_ddls))

        # Execute end-to-end with MaterializedViewManager, bulk_changes shadow swap, drift, and repair
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(schema_sql)
            conn.execute("INSERT INTO orders VALUES (1, 10, 'us', 100, 50), (2, 10, 'us', 101, 25);")
            conn.commit()
            mgr = ms.MaterializedViewManager(conn, dialect=ms.Dialect.SQLITE)
            mgr.register_view(view_distinct)
            live_idxs = mgr._get_live_index_names()
            self.assertIsNotNone(live_idxs)
            for idx_name in plan.index_names:
                self.assertIn(idx_name.lower(), live_idxs)

            # Shadow-swap rebuild should recreate both backing and aux indexes
            with mgr.bulk_changes(["mv_dist_cust"], use_shadow_swap=True):
                conn.execute("INSERT INTO orders VALUES (3, 10, 'eu', 100, 75);")
            live_idxs_after = mgr._get_live_index_names()
            for idx_name in plan.index_names:
                self.assertIn(idx_name.lower(), live_idxs_after)

            # Tier 3 window view auto-indexes PARTITION BY projected column
            win_view = """
            CREATE VIEW mv_ranked AS
            SELECT id, region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY id) AS rn
            FROM orders;
            """
            win_plan = ms.generate_materialized_view(schema_sql, win_view, dialect=ms.Dialect.SQLITE)
            self.assertTrue(
                any("__matview_store_mv_ranked" in ddl and "region" in ddl for ddl in win_plan.index_ddls),
                f"Expected backing auto-index on region, got {win_plan.index_ddls}",
            )
            self.assertTrue(
                any("orders" in ddl and "region" in ddl for ddl in win_plan.recommended_base_indexes),
            )
        finally:
            conn.close()

    def _run_live_engine_index_roundtrip(self, dsn: str, dialect: ms.Dialect) -> None:
        # Use the project's DSN normalization so the declared drivers
        # (psycopg2 / PyMySQL) are selected; a bare postgresql:// or mysql://
        # URL would make SQLAlchemy import undeclared drivers (psycopg, MySQLdb).
        sa_url, _, connect_args = ms.normalize_sqlalchemy_url(dsn)
        eng = sa.create_engine(sa_url, **connect_args)
        try:
            with eng.begin() as conn:
                conn.exec_driver_sql("DROP VIEW IF EXISTS mv_p2_idx_test")
                conn.exec_driver_sql("DROP TABLE IF EXISTS __matview_store_mv_p2_idx_test")
                conn.exec_driver_sql("DROP TABLE IF EXISTS p2_idx_emps")
                conn.exec_driver_sql("DROP TABLE IF EXISTS p2_idx_depts")
                conn.exec_driver_sql(
                    "CREATE TABLE p2_idx_depts (id INTEGER PRIMARY KEY, title VARCHAR(64) NOT NULL)"
                )
                conn.exec_driver_sql(
                    "CREATE TABLE p2_idx_emps (id INTEGER PRIMARY KEY, dept_id INTEGER NOT NULL, salary INTEGER NOT NULL)"
                )
                conn.exec_driver_sql("CREATE INDEX idx_p2_emps_dept ON p2_idx_emps (dept_id)")
            schemas = ms.introspect_database_schemas(eng, dialect=dialect)
            self.assertTrue(schemas["p2_idx_emps"].has_index_covering(["dept_id"]))
            view_sql = """
            CREATE VIEW mv_p2_idx_test AS
            SELECT d.title AS dept_title, COUNT(*) AS cnt, SUM(e.salary) AS total_sal
            FROM p2_idx_emps e
            INNER JOIN p2_idx_depts d ON e.dept_id = d.id
            GROUP BY d.title;
            """
            plan = ms.generate_materialized_view(
                schemas, view_sql, dialect=dialect, ignore_limitations=(dialect == ms.Dialect.MYSQL)
            )
            self.assertEqual(plan.recommended_base_indexes, [])
        finally:
            with eng.begin() as conn:
                conn.exec_driver_sql("DROP TABLE IF EXISTS p2_idx_emps")
                conn.exec_driver_sql("DROP TABLE IF EXISTS p2_idx_depts")
            eng.dispose()

    def test_native_postgres_index_introspection(self) -> None:
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_live_engine_index_roundtrip(dsn, ms.Dialect.POSTGRES)

    def test_native_mysql_index_introspection(self) -> None:
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_live_engine_index_roundtrip(dsn, ms.Dialect.MYSQL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
