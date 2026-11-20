#!/usr/bin/env python3
"""
Tests for dialect-aware AST expression storage type inference across
SQLite, PostgreSQL, and MySQL (CASE, COALESCE, arithmetic, MIN/MAX expressions).
"""

import sqlite3
import unittest
from decimal import Decimal

from matview_sync import (
    Dialect,
    MaterializedViewManager,
    generate_materialized_view,
    parse_table_schema,
)


class TestExprStorageTypeInference(unittest.TestCase):
    SCHEMA = """
    CREATE TABLE ledger (
        id INTEGER PRIMARY KEY,
        grp TEXT NOT NULL,
        flag INT NOT NULL,
        qty INT,
        price DECIMAL(12, 4) NOT NULL,
        rate REAL NOT NULL,
        tag VARCHAR(32)
    );
    """

    def setUp(self) -> None:
        self.schemas = parse_table_schema(self.SCHEMA)

    def test_case_with_decimal_and_zero_fallback_preserves_decimal_across_dialects(self) -> None:
        v_sql = (
            "CREATE VIEW mv_case_dec AS "
            "SELECT grp, "
            "CASE WHEN flag > 0 THEN price ELSE 0 END AS eff_price "
            "FROM ledger;"
        )
        p_pg = generate_materialized_view(self.schemas, v_sql, dialect=Dialect.POSTGRES)
        self.assertIn("eff_price NUMERIC", p_pg.backing_table_ddl)

        p_my = generate_materialized_view(
            self.schemas, v_sql, dialect=Dialect.MYSQL, ignore_limitations=True
        )
        self.assertRegex(p_my.backing_table_ddl, r"eff_price DECIMAL\(12,\s*4\)")

        p_sq = generate_materialized_view(self.schemas, v_sql, dialect=Dialect.SQLITE)
        # SQLite computed expressions have no affinity in live views; storing
        # without affinity (BLOB) prevents coercing REAL results to INTEGER/TEXT.
        self.assertIn("eff_price BLOB", p_sq.backing_table_ddl)
        self.assertNotIn("eff_price BIGINT", p_sq.backing_table_ddl)

    def test_case_column_branches_infer_common_type_not_text(self) -> None:
        v_sql = (
            "CREATE VIEW mv_case_cols AS "
            "SELECT id, "
            "CASE WHEN flag = 1 THEN qty ELSE id END AS chosen_int, "
            "CASE WHEN flag = 1 THEN tag ELSE 'default' END AS chosen_tag, "
            "CASE WHEN flag = 1 THEN qty ELSE rate END AS chosen_real "
            "FROM ledger;"
        )
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            plan = generate_materialized_view(
                self.schemas,
                v_sql,
                dialect=dialect,
                ignore_limitations=(dialect == Dialect.MYSQL),
            )
            ddl = plan.backing_table_ddl.upper()
            if dialect == Dialect.POSTGRES:
                self.assertIn("CHOSEN_INT INTEGER", ddl)
                self.assertIn("CHOSEN_TAG TEXT", ddl)
                self.assertIn("CHOSEN_REAL DOUBLE PRECISION", ddl)
            elif dialect == Dialect.SQLITE:
                self.assertIn("CHOSEN_INT BLOB", ddl)
                self.assertIn("CHOSEN_TAG BLOB", ddl)
                self.assertIn("CHOSEN_REAL BLOB", ddl)
            else:
                self.assertIn("CHOSEN_INT BIGINT", ddl)
                self.assertNotIn("CHOSEN_INT VARCHAR", ddl)
                self.assertNotIn("CHOSEN_INT TEXT", ddl)
                self.assertNotIn("CHOSEN_REAL BIGINT", ddl)

    def test_coalesce_mixed_numeric_promotes_to_wider_type(self) -> None:
        v_sql = (
            "CREATE VIEW mv_coal AS "
            "SELECT id, "
            "COALESCE(qty, price) AS q_or_p, "
            "COALESCE(qty, rate) AS q_or_r "
            "FROM ledger;"
        )
        p_pg = generate_materialized_view(self.schemas, v_sql, dialect=Dialect.POSTGRES)
        self.assertIn("q_or_p NUMERIC", p_pg.backing_table_ddl)
        self.assertIn("q_or_r DOUBLE PRECISION", p_pg.backing_table_ddl)

        p_my = generate_materialized_view(
            self.schemas, v_sql, dialect=Dialect.MYSQL, ignore_limitations=True
        )
        self.assertRegex(p_my.backing_table_ddl, r"q_or_p DECIMAL\(")
        self.assertIn("q_or_r DOUBLE", p_my.backing_table_ddl)

    def test_decimal_arithmetic_preserves_exact_decimal(self) -> None:
        v_sql = (
            "CREATE VIEW mv_mul AS "
            "SELECT id, price * qty AS line_total "
            "FROM ledger;"
        )
        p_pg = generate_materialized_view(self.schemas, v_sql, dialect=Dialect.POSTGRES)
        self.assertIn("line_total NUMERIC", p_pg.backing_table_ddl)

        p_my = generate_materialized_view(
            self.schemas, v_sql, dialect=Dialect.MYSQL, ignore_limitations=True
        )
        self.assertRegex(p_my.backing_table_ddl, r"line_total DECIMAL\(")

    def test_min_max_expression_arg_infers_inner_expr_type(self) -> None:
        v_sql = (
            "CREATE VIEW mv_min_expr AS "
            "SELECT grp, "
            "MIN(UPPER(tag)) AS min_upper_tag, "
            "MAX(COALESCE(tag, 'z')) AS max_tag "
            "FROM ledger GROUP BY grp;"
        )
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            plan = generate_materialized_view(
                self.schemas,
                v_sql,
                dialect=dialect,
                ignore_limitations=(dialect == Dialect.MYSQL),
            )
            ddl = plan.backing_table_ddl.upper()
            self.assertNotIn("DOUBLE", ddl)
            self.assertNotIn("REAL", ddl)

    def test_live_sqlite_roundtrip_without_truncation(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(
                """
                CREATE TABLE ledger (
                    id INTEGER PRIMARY KEY,
                    grp TEXT NOT NULL,
                    flag INT NOT NULL,
                    qty INT,
                    price REAL NOT NULL,
                    tag TEXT
                );
                INSERT INTO ledger VALUES (1, 'a', 1, 2, 12.75, 'alpha');
                INSERT INTO ledger VALUES (2, 'a', 0, NULL, 9.25, NULL);
                """
            )
            mgr = MaterializedViewManager(conn)
            mgr.register_view(
                """
                CREATE VIEW mv_exprs AS
                SELECT grp,
                    MAX(CASE WHEN flag > 0 THEN price ELSE 0 END) AS max_eff_price,
                    MIN(UPPER(COALESCE(tag, 'none'))) AS min_tag
                FROM ledger
                GROUP BY grp;
                """
            )
            rows = conn.execute(
                "SELECT grp, max_eff_price, min_tag FROM mv_exprs;"
            ).fetchall()
            self.assertEqual(rows, [("a", 12.75, "ALPHA")])

            conn.execute("INSERT INTO ledger VALUES (3, 'a', 1, 5, 19.50, 'aa');")
            rows2 = conn.execute(
                "SELECT grp, max_eff_price, min_tag FROM mv_exprs;"
            ).fetchall()
            self.assertEqual(rows2, [("a", 19.5, "AA")])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
