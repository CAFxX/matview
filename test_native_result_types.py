#!/usr/bin/env python3
"""Native result types of materialized projections and aggregates.

Each view below is materialized and compared with a live view of the same
SELECT on the real engine -- the only oracle -- through population and
INSERT/UPDATE/DELETE/rollback: rows exactly (Python types, floats bit-for-bit,
Decimal digits and scale) and result metadata (DB-API description and the
catalog column type of the reader view vs the live view).

The expression matrix targets result-type rules that differ per engine:
integer arithmetic width and `%` (PostgreSQL keeps int2/int4, MySQL widens),
CASE/COALESCE common types with untyped literals, string types
(varchar/bpchar/text), MIN/MAX type modifiers, SUM/BIT_* result families,
aggregates nested in expressions, window aggregates and one-row AVG scale.
Offline tests pin the planned storage types for the PostgreSQL rules, which
were each confirmed natively.

Native engines: POSTGRES_DSN and/or MYSQL_DSN (MySQL or MariaDB, detected
from the server).
"""

from __future__ import annotations

import unittest
import uuid
import warnings
from typing import Any, List, Tuple

from matview_sync import (
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    MatviewDiagnosticWarning,
    MaintenanceStrategy,
    MaterializedViewManager,
    UnsupportedMaterializedViewError,
    generate_materialized_view,
)
from test_matview_sync import FLOAT_OPT_IN
from test_native_statistical_aggregates import (
    _catalog_types,
    _connect,
    _description,
    _exact,
    _metadata_mismatches,
    _native_engines,
)

_TABLE = (
    "CREATE TABLE {t} (id INT PRIMARY KEY, g INT NOT NULL, i INT, s SMALLINT, b BIGINT, "
    "d DECIMAL(12,3), f DOUBLE{dp}, v VARCHAR(16), c CHAR(3))"
)

_SEED = [
    (1, 1, 5, 2, 10, "1.250", 0.5, "ab", "x"),
    (2, 1, -3, 7, 20, "2.500", 1.25, None, "yy"),
    (3, 2, None, 1, None, None, None, "cd", None),
    (4, 2, 8, 4, 40, "-0.125", 3.0, "ef", "z"),
    (5, 3, 0, 0, 0, "0.000", 0.0, "", "w"),
]

_DML = [
    ("insert", "INSERT INTO {t} (id, g, i, s, b, d, f, v, c) VALUES (6, 3, 4, 3, 7, 9.875, 2.5, 'gh', 'q')"),
    ("insert new group", "INSERT INTO {t} (id, g, i, s, b, d, f, v, c) VALUES (7, 4, 50, 5, 9, 1.5, 0.25, 'zz', 'k')"),
    ("update", "UPDATE {t} SET i = i + 1, d = d * 2, v = 'upd' WHERE id IN (1, 4)"),
    ("move group", "UPDATE {t} SET g = 3 WHERE id = 2"),
    ("to null", "UPDATE {t} SET i = NULL, s = NULL, d = NULL, v = NULL WHERE id = 5"),
    ("delete", "DELETE FROM {t} WHERE id = 3"),
]

# (name, projection over {t} rows) -- non-aggregate projections keep id.
_PROJECTIONS = [
    ("int_arith", "i * 2 + s"),
    ("small_arith", "s + s"),
    ("int_mod", "i % 3"),
    ("big_arith", "b + 1"),
    ("dec_mod", "d % 2"),
    ("dec_mul", "d * 2"),
    ("case_lit", "CASE WHEN i > 0 THEN 1 ELSE 0 END"),
    ("case_col_neg", "CASE WHEN i > 0 THEN i ELSE -1 END"),
    ("case_nested", "CASE WHEN i > 0 THEN CASE WHEN s > 1 THEN s ELSE 2 END ELSE i END"),
    ("coalesce_int", "COALESCE(i, 0)"),
    ("coalesce_dec", "COALESCE(d, 0)"),
    ("coalesce_varchar", "COALESCE(v, 'none')"),
    ("case_varchar", "CASE WHEN i > 0 THEN v ELSE 'neg' END"),
    ("upper_varchar", "UPPER(v)"),
    ("win_sum", "SUM(i) OVER (PARTITION BY g)"),
    ("win_min_dec", "MIN(d) OVER (PARTITION BY g)"),
    ("win_count", "COUNT(i) OVER (PARTITION BY g)"),
    ("win_max_small", "MAX(s) OVER (PARTITION BY g)"),
    ("win_min_int", "MIN(i) OVER (PARTITION BY g)"),
    ("win_max_big", "MAX(b) OVER (PARTITION BY g)"),
]

_AGGREGATES = [
    ("sum_mul", "SUM(i * s)"),
    ("sum_case", "SUM(CASE WHEN i > 0 THEN 1 ELSE 0 END)"),
    ("min_dec", "MIN(d)"),
    ("max_varchar", "MAX(v)"),
    ("bit_and_int", "BIT_AND(i)"),
    ("bit_or_small", "BIT_OR(s)"),
    ("bit_xor_big", "BIT_XOR(b)"),
    ("avg_int", "AVG(i)"),
    ("avg_distinct_int", "AVG(DISTINCT i)"),
    ("coalesce_sum", "COALESCE(SUM(i), 0)"),
    ("scaled_sum", "1.0 * SUM(i)"),
    ("count_plus", "COUNT(*) + 1"),
    ("sum_dec_expr", "SUM(d * 2)"),
]


# Grouped aggregate views whose MySQL/MariaDB column nullability is compared
# despite the general MYSQL_UNMODELLED_METADATA exclusion (expression group
# keys such as a CASE that is never NULL are part of the open gap).
_NULLABILITY_CHECKED = {
    "sum_mul", "sum_case", "min_dec", "max_varchar", "bit_and_int", "bit_or_small", "bit_xor_big",
    "avg_int", "avg_distinct_int", "count_plus", "sum_dec_expr",
}


def _views(engine: Dialect, t: str) -> List[Tuple[str, str]]:
    views = [(name, f"SELECT id, {expr} AS r FROM {t}") for name, expr in _PROJECTIONS]
    views += [(name, f"SELECT g, {expr} AS r FROM {t} GROUP BY g") for name, expr in _AGGREGATES]
    views += [
        # Grouped through a temporary table (MySQL sends INT, not BIGINT).
        ("grouped_case_key", f"SELECT CASE WHEN i IS NULL THEN 0 ELSE 1 END AS k, COUNT(*) AS c FROM {t} "
                             f"GROUP BY CASE WHEN i IS NULL THEN 0 ELSE 1 END"),
        # Aggregates read through a CTE / derived UNION ALL.
        ("cte_max_int", f"WITH a AS (SELECT g, i FROM {t} WHERE id > 0) SELECT g, MAX(i) AS r FROM a GROUP BY g"),
        ("cte_coalesce_sum", f"WITH a AS (SELECT g, i FROM {t}) SELECT g, COALESCE(SUM(i), 0) AS r FROM a GROUP BY g"),
        # Not decomposable (LEFT JOIN): the compound expression is stored.
        ("leftjoin_coalesce_sum", f"WITH a AS (SELECT g, i FROM {t}) SELECT d.id, COALESCE(SUM(a.i), 0) AS r "
                                  f"FROM {t} d LEFT JOIN a ON a.g = d.id GROUP BY d.id"),
        ("recursive_cte_depth", f"WITH RECURSIVE tr(id, depth) AS (SELECT id, 0 FROM {t} WHERE g = 1 "
                                f"UNION ALL SELECT x.id, tr.depth + 1 FROM {t} x JOIN tr ON x.g = tr.id + 1 "
                                f"WHERE tr.depth < 3) SELECT id, depth FROM tr"),
        # MIN/MAX of a recursive CTE's integer column (a bare column, not an
        # expression: MariaDB keeps INT; MySQL's derived-table rule gives BIGINT).
        ("recursive_cte_max_depth", f"WITH RECURSIVE tr(id, depth) AS (SELECT id, 0 FROM {t} WHERE g = 1 "
                                    f"UNION ALL SELECT x.id, tr.depth + 1 FROM {t} x JOIN tr ON x.g = tr.id + 1 "
                                    f"WHERE tr.depth < 3) SELECT id, MAX(depth) AS r FROM tr GROUP BY id"),
        ("union_sum", f"SELECT g, SUM(v) AS r FROM (SELECT g, i AS v FROM {t} UNION ALL SELECT g, s AS v FROM {t}) AS u GROUP BY g"),
    ]
    views += [
        # String CASE/COALESCE resolution order (PostgreSQL: CASE resolves its
        # ELSE first; MySQL/MariaDB: common VARCHAR).
        ("case_varchar_char", f"SELECT id, CASE WHEN g = 1 THEN v ELSE c END AS r FROM {t}"),
        ("case_char_varchar", f"SELECT id, CASE WHEN g = 1 THEN c ELSE v END AS r FROM {t}"),
        ("coalesce_char_varchar", f"SELECT id, COALESCE(c, v) AS r FROM {t}"),
    ]
    views += [
        # Window MIN/MAX over CHAR (MySQL returns VARCHAR, PostgreSQL bpchar).
        ("win_max_char", f"SELECT id, MAX(c) OVER (PARTITION BY g) AS r FROM {t}"),
        # MIN/MAX over a recursive CTE's SMALLINT column (MySQL keeps the type;
        # a non-recursive CTE sends BIGINT, see cte_max_int).
        ("recursive_cte_min_s", f"WITH RECURSIVE tr(id, s, depth) AS (SELECT id, s, 0 FROM {t} WHERE g = 1 "
                                f"UNION ALL SELECT x.id, x.s, tr.depth + 1 FROM {t} x JOIN tr ON x.g = tr.id + 1 "
                                f"WHERE tr.depth < 3) SELECT depth, MIN(s) AS r FROM tr GROUP BY depth"),
    ]
    if engine == Dialect.MYSQL:
        views.append(("win_max_enum", f"SELECT id, MAX(e) OVER () AS r FROM {t}"))
    if engine == Dialect.POSTGRES:
        views.append(("min_char", f"SELECT g, MIN(c) AS r FROM {t} GROUP BY g"))
        views.append(("coalesce_char", f"SELECT id, COALESCE(c, 'q') AS r FROM {t}"))
    return views


class PostgresPlannedResultTypeTests(unittest.TestCase):
    """Offline: planned PostgreSQL storage types follow the native rules."""

    SCHEMA = _TABLE.format(t="t", dp=" PRECISION") + ";"

    def _col_type(self, select: str) -> str:
        plan = generate_materialized_view(self.SCHEMA, f"CREATE VIEW v AS {select};", dialect="postgres:18")
        for line in plan.backing_table_ddl.splitlines():
            parts = line.strip().rstrip(",").split(" ", 1)
            if parts[0] == "r":
                return parts[1]
        raise AssertionError(plan.backing_table_ddl)

    def test_native_postgres_types(self) -> None:
        expected = {
            "SELECT id, i * 2 + s AS r FROM t": "INTEGER",
            "SELECT id, s + s AS r FROM t": "SMALLINT",
            "SELECT id, i % 3 AS r FROM t": "INTEGER",
            "SELECT id, b + 1 AS r FROM t": "BIGINT",
            "SELECT id, COALESCE(v, 'none') AS r FROM t": "VARCHAR",
            "SELECT id, COALESCE(c, 'q') AS r FROM t": "BPCHAR",
            "SELECT id, UPPER(v) AS r FROM t": "TEXT",
            "SELECT id, CASE WHEN g = 1 THEN v ELSE c END AS r FROM t": "BPCHAR",
            "SELECT id, CASE WHEN g = 1 THEN c ELSE v END AS r FROM t": "VARCHAR",
            "SELECT id, CASE WHEN g = 1 THEN v ELSE 'z' END AS r FROM t": "VARCHAR",
            "SELECT id, COALESCE(c, v) AS r FROM t": "BPCHAR",
            "SELECT id, SUM(i) OVER (PARTITION BY g) AS r FROM t": "BIGINT",
            "SELECT id, MIN(d) OVER (PARTITION BY g) AS r FROM t": "NUMERIC",
            "SELECT g, SUM(i * s) AS r FROM t GROUP BY g": "BIGINT",
            "SELECT g, SUM(CASE WHEN i > 0 THEN 1 ELSE 0 END) AS r FROM t GROUP BY g": "BIGINT",
            "SELECT g, MIN(d) AS r FROM t GROUP BY g": "NUMERIC",
            "SELECT g, MAX(v) AS r FROM t GROUP BY g": "TEXT",
            "SELECT g, MIN(c) AS r FROM t GROUP BY g": "BPCHAR",
            "SELECT g, BIT_AND(i) AS r FROM t GROUP BY g": "INTEGER",
            "SELECT g, BIT_OR(s) AS r FROM t GROUP BY g": "SMALLINT",
            "SELECT g, BIT_XOR(b) AS r FROM t GROUP BY g": "BIGINT",
        }
        for select, col_type in expected.items():
            with self.subTest(select=select):
                self.assertEqual(self._col_type(select), col_type)

    def test_compound_aggregate_expressions_are_typed(self) -> None:
        plan = generate_materialized_view(
            self.SCHEMA,
            "CREATE VIEW v AS WITH c AS (SELECT g, i FROM t) SELECT g, COALESCE(SUM(i), 0) AS r "
            "FROM c GROUP BY g ORDER BY g LIMIT 5;",
            dialect="postgres:18",
        )
        self.assertIn("r BIGINT", plan.backing_table_ddl)

    def test_recursive_cte_literal_column_is_integer(self) -> None:
        plan = generate_materialized_view(
            self.SCHEMA,
            "CREATE VIEW v AS WITH RECURSIVE tr(id, depth) AS (SELECT id, 0 FROM t WHERE g = 1 "
            "UNION ALL SELECT t.id, tr.depth + 1 FROM t JOIN tr ON t.g = tr.id WHERE tr.depth < 3) "
            "SELECT id, depth FROM tr;",
            dialect="postgres:18",
        )
        self.assertIn("depth INTEGER", plan.backing_table_ddl)

    def test_sum_integer_expression_types_pg_and_mysql(self) -> None:
        import re
        from matview import EngineLimitationError

        pg_schema = (
            "CREATE TABLE ik (id INT PRIMARY KEY, g INT, i INT4, s INT2, b INT8, "
            "d NUMERIC(12, 3), f4 FLOAT4, f8 DOUBLE PRECISION);"
        )
        pg_cases = {
            "SELECT g, SUM(i % 3) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(MOD(i, 3)) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(s + s) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(-i) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(i / 2) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(b) AS r FROM ik GROUP BY g": "NUMERIC",
            "SELECT g, SUM(b + 1) AS r FROM ik GROUP BY g": "NUMERIC",
            "SELECT g, SUM(i % b) AS r FROM ik GROUP BY g": "NUMERIC",
            "SELECT g, SUM(CASE WHEN i > 0 THEN i ELSE NULL END) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(COALESCE(NULL, i)) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(NULLIF(i, 0)) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(NULLIF(i, b)) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(NULLIF(NULL, i)) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(GREATEST(i, 0)) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(LEAST(b, 10)) AS r FROM ik GROUP BY g": "NUMERIC",
            "SELECT g, SUM(ABS(i)) AS r FROM ik GROUP BY g": "BIGINT",
            "SELECT g, SUM(FLOOR(d)) AS r FROM ik GROUP BY g": "NUMERIC",
            "SELECT g, SUM(ROUND(d, 1)) AS r FROM ik GROUP BY g": "NUMERIC",
        }
        for select, expected_type in pg_cases.items():
            with self.subTest(dialect="postgres", select=select):
                plan = generate_materialized_view(pg_schema, f"CREATE VIEW v AS {select};", dialect="postgres:18")
                m = re.search(r'"?r"?\s+([A-Z][A-Z0-9_ ()]*?)(?:\s+NOT\s+NULL)?\s*(?:,|\))', plan.backing_table_ddl)
                assert m is not None, plan.backing_table_ddl
                self.assertEqual(m.group(1).strip(), expected_type)

        # Bare FLOAT4, f4 + f4, and NULLIF(f4, f8) keep REAL under opt-in;
        # NULLIF(i, f4), NULLIF(i, f8), and FLOOR/CEIL/ROUND/TRUNC/SIGN(int) are DOUBLE PRECISION and refused by default.
        for opt_select, opt_type in (
            ("SELECT g, SUM(f4) AS r FROM ik GROUP BY g", "REAL"),
            ("SELECT g, SUM(f4 + f4) AS r FROM ik GROUP BY g", "REAL"),
            ("SELECT g, SUM(NULLIF(f4, f8)) AS r FROM ik GROUP BY g", "REAL"),
            ("SELECT g, SUM(NULLIF(i, f4)) AS r FROM ik GROUP BY g", "DOUBLE PRECISION"),
            ("SELECT g, SUM(FLOOR(i)) AS r FROM ik GROUP BY g", "DOUBLE PRECISION"),
        ):
            with self.subTest(dialect="postgres_float", select=opt_select):
                with self.assertRaises(EngineLimitationError):
                    generate_materialized_view(pg_schema, f"CREATE VIEW v AS {opt_select};", dialect="postgres:18")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    plan_fl = generate_materialized_view(
                        pg_schema,
                        f"CREATE VIEW v AS {opt_select};",
                        dialect="postgres:18",
                        accept_limitations=FLOAT_OPT_IN,
                    )
                self.assertIn(f"r {opt_type}", plan_fl.backing_table_ddl)
        for pg_refused in (
            "SELECT g, SUM(NULLIF(i, f8)) AS r FROM ik GROUP BY g",
            "SELECT g, SUM(CEIL(i)) AS r FROM ik GROUP BY g",
            "SELECT g, SUM(ROUND(i)) AS r FROM ik GROUP BY g",
            "SELECT g, SUM(TRUNC(i)) AS r FROM ik GROUP BY g",
            "SELECT g, SUM(SIGN(i)) AS r FROM ik GROUP BY g",
        ):
            with self.subTest(dialect="postgres_refused", select=pg_refused):
                with self.assertRaises(EngineLimitationError):
                    generate_materialized_view(pg_schema, f"CREATE VIEW v AS {pg_refused};", dialect="postgres:18")

        mysql_schema = (
            "CREATE TABLE ik (id INT PRIMARY KEY, g INT, i INT, s SMALLINT, b BIGINT, "
            "ti TINYINT, mi MEDIUMINT, u INT UNSIGNED, d DECIMAL(12, 3), f DOUBLE, du DOUBLE UNSIGNED);"
        )
        mysql_exact = {
            "SELECT g, SUM(i % 3) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(MOD(i, 3)) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(i DIV 2) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(s + s) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(-i) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(u % 7) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(CASE WHEN i > 0 THEN i ELSE NULL END) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(COALESCE(NULL, i)) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(NULLIF(i, 0)) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(GREATEST(i, 0)) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(IF(i > 0, i, 0)) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(CAST(ABS(i) AS UNSIGNED)) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(FLOOR(d)) AS r FROM ik GROUP BY g": "DECIMAL(65, 0)",
            "SELECT g, SUM(ROUND(d, 1)) AS r FROM ik GROUP BY g": "DECIMAL(65, 1)",
            "SELECT g, SUM(TRUNCATE(d, 1)) AS r FROM ik GROUP BY g": "DECIMAL(65, 1)",
        }
        for select, expected_type in mysql_exact.items():
            with self.subTest(dialect="mysql", select=select):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    plan = generate_materialized_view(
                        mysql_schema,
                        f"CREATE VIEW v AS {select};",
                        dialect="mysql:8.4",
                        ignore_limitations=True,
                    )
                self.assertIn(expected_type, plan.backing_table_ddl)

        # MySQL exact AVG(u) (INT UNSIGNED): unindexed AVG is derived in the reader
        # view with exact scale 0+4=4, while an indexed AVG keeps its DECIMAL(14, 4)
        # backing column.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            avg_unindexed = generate_materialized_view(
                mysql_schema,
                "CREATE VIEW v AS SELECT g, AVG(u) AS r FROM ik GROUP BY g;",
                dialect="mysql:8.4",
                ignore_limitations=True,
            )
            avg_indexed = generate_materialized_view(
                mysql_schema,
                "CREATE VIEW v AS SELECT g, AVG(u) AS r FROM ik GROUP BY g;",
                dialect="mysql:8.4",
                indexes=["CREATE INDEX idx_v_r ON v (r);"],
                ignore_limitations=True,
            )
        self.assertIn("DECIMAL(65, 0)", avg_unindexed.reader_view_ddl)
        self.assertIn(") AS r", avg_unindexed.reader_view_ddl)
        self.assertIn("DECIMAL(14, 4)", avg_indexed.backing_table_ddl)

        # Float branch inside CASE/IF and DOUBLE UNSIGNED must fail closed (never misclassified as exact).
        for approx_select in (
            "SELECT g, SUM(CASE WHEN i > 0 THEN f ELSE NULL END) AS r FROM ik GROUP BY g",
            "SELECT g, SUM(IF(i > 0, f, 0)) AS r FROM ik GROUP BY g",
            "SELECT g, SUM(du) AS r FROM ik GROUP BY g",
            "SELECT g, SUM(du + 1) AS r FROM ik GROUP BY g",
            "SELECT g, AVG(du) AS r FROM ik GROUP BY g",
        ):
            with self.subTest(dialect="mysql_refused", select=approx_select):
                with self.assertRaises(EngineLimitationError):
                    generate_materialized_view(
                        mysql_schema,
                        f"CREATE VIEW v AS {approx_select};",
                        dialect="mysql:8.4",
                        ignore_limitations=True,
                    )


class MySQLPlannedResultTypeTests(unittest.TestCase):
    """Offline guards for MySQL's GROUP BY expression result metadata."""

    SCHEMA = (
        "CREATE TABLE ik (id INT PRIMARY KEY, i INT, s SMALLINT, b BIGINT, "
        "ti TINYINT, mi MEDIUMINT, u INT UNSIGNED, su SMALLINT UNSIGNED);"
    )

    def _key_type(self, dialect: str, select_tail: str, expr: str) -> str:
        import re

        plan = generate_materialized_view(
            self.SCHEMA,
            f"CREATE VIEW v AS SELECT {expr} AS k, COUNT(*) AS c FROM ik {select_tail};",
            dialect=dialect,
            ignore_limitations=True,
        )
        match = re.search(r'(?im)^\s*[\x60"]?k[\x60"]?\s+([A-Z]+(?:\s+UNSIGNED)?)\b', plan.backing_table_ddl)
        self.assertIsNotNone(match, plan.backing_table_ddl)
        assert match is not None
        return match.group(1)

    def test_grouped_integer_arithmetic_follows_the_length_rule(self) -> None:
        # Display length below 10 characters => INT, else BIGINT (natively
        # verified in test_mysql_group_key_integer_type_matches_the_length_model);
        # shapes outside the verified subset stay BIGINT (never a wrong INT).
        cases = {
            "s % 2": "INT", "ti % 2": "INT", "mi % 2": "INT", "s + s": "INT", "s - s": "INT", "s * 2": "INT",
            "su + su": "INT", "(s + s) % 3": "INT", "s DIV 2": "INT", "s + 123456": "INT",
            "i % 3": "BIGINT", "s * s": "BIGINT", "mi + mi": "BIGINT", "i + 1": "BIGINT", "b % 2": "BIGINT",
            "-s": "BIGINT", "su + s": "BIGINT", "(u + u) % 3": "BIGINT",
        }
        for dialect in ("mysql:8.4", "mysql:9.7"):
            for expr, expected_type in cases.items():
                with self.subTest(dialect=dialect, expr=expr):
                    self.assertEqual(self._key_type(dialect, f"GROUP BY {expr}", expr), expected_type)

    def test_unsigned_casts_and_div_are_never_stored_as_strings(self) -> None:
        # CAST(x AS UNSIGNED) and DIV over unsigned columns used to fall through
        # to VARCHAR(255): readers got strings instead of integers (MySQL and MariaDB).
        for dialect in ("mysql:8.4", "mysql:9.7", "mariadb:11.8"):
            for expr in ("CAST(s AS UNSIGNED)", "CAST(i AS UNSIGNED INTEGER)", "u DIV 2", "i DIV u", "s DIV 2"):
                with self.subTest(dialect=dialect, expr=expr):
                    stored = self._key_type(dialect, f"GROUP BY {expr}", expr)
                    self.assertIn(stored.split()[0], ("INT", "BIGINT"), stored)

    def test_grouped_key_matched_through_alias_and_ordinal(self) -> None:
        for dialect in ("mysql:8.4", "mysql:9.7"):
            with self.subTest(dialect=dialect, how="ordinal"):
                self.assertEqual(self._key_type(dialect, "GROUP BY 1", "s % 2"), "INT")
            with self.subTest(dialect=dialect, how="alias"):
                self.assertEqual(self._key_type(dialect, "GROUP BY k", "s % 2"), "INT")
            with self.subTest(dialect=dialect, how="not grouped"):
                plan = generate_materialized_view(
                    self.SCHEMA, "CREATE VIEW v AS SELECT id, s % 2 AS k FROM ik;",
                    dialect=dialect, ignore_limitations=True,
                )
                self.assertRegex(plan.backing_table_ddl, r'(?im)^\s*[\x60"]?k[\x60"]?\s+BIGINT\b')


class NativeResultTypeTests(unittest.TestCase):
    # Views registered on one base table at a time: per-statement trigger
    # memory grows quadratically with the views on a table on MySQL/MariaDB
    # (see test_native_statistical_aggregates._VIEWS_PER_TABLE).
    VIEWS_PER_TABLE = 16

    def _run(self, engine: Dialect, dsn: str, chunk: slice) -> List[str]:
        conn = _connect(engine, dsn)
        cur = conn.cursor()
        cur.execute("SELECT VERSION()")
        version = str(cur.fetchone()[0])
        is_mariadb = "mariadb" in version.lower()
        conn.commit()
        tag = uuid.uuid4().hex[:8]
        schema, t = f"rt_{tag}", f"t_{tag}"
        failures: List[str] = []
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
            dp = " PRECISION" if engine == Dialect.POSTGRES else ""
            cur.execute(_TABLE.format(t=t, dp=dp) + (" ENGINE=InnoDB" if engine == Dialect.MYSQL else ""))
            cur.executemany(f"INSERT INTO {t} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", _SEED)
            if engine == Dialect.MYSQL:
                # ENUM (MySQL family only) for the window MIN/MAX string cases.
                cur.execute(f"ALTER TABLE {t} ADD COLUMN e ENUM('a', 'bb') NULL")
                cur.execute(f"UPDATE {t} SET e = IF(id % 2 = 1, 'a', 'bb')")
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=engine)
            pairs = []
            for name, select in _views(engine, t)[chunk]:
                mv, live = f"mv_{name}_{tag}", f"live_{name}_{tag}"
                cur.execute(f"CREATE VIEW {live} AS {select}")
                conn.commit()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if engine == Dialect.MYSQL:
                        with self.assertRaises(EngineLimitationError):
                            manager.register_view(f"CREATE VIEW {mv} AS {select};", accept_limitations=FLOAT_OPT_IN)
                        conn.rollback()
                        manager.register_view(f"CREATE VIEW {mv} AS {select};", ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
                        codes = [str(w.message) for w in caught if issubclass(w.category, EngineLimitationWarning)]
                        self.assertEqual(len(codes), 1)
                        self.assertIn("MYSQL_NO_TRUNCATE_TRIGGER", codes[0])
                    else:
                        manager.register_view(f"CREATE VIEW {mv} AS {select};", accept_limitations=FLOAT_OPT_IN)
                conn.commit()
                pairs.append((name, mv, live))
                mv_cat = [r[1:] for r in _catalog_types(engine, cur, mv)]
                live_cat = [r[1:] for r in _catalog_types(engine, cur, live)]
                if engine == Dialect.MYSQL and name in _NULLABILITY_CHECKED:
                    # Nullability is a documented MYSQL_UNMODELLED_METADATA gap
                    # in general; for grouped aggregates (NOT NULL group key,
                    # COUNT, nullable SUM/MIN/MAX/AVG/BIT_*) the materialized
                    # columns must already agree with the live view.
                    nulls = []
                    for view in (mv, live):
                        cur.execute(
                            "SELECT COLUMN_NAME, IS_NULLABLE FROM information_schema.COLUMNS "
                            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION",
                            (view,),
                        )
                        nulls.append([tuple(r) for r in cur.fetchall()])
                    if nulls[0] != nulls[1]:
                        failures.append(f"{version} {name}: nullability {nulls[0]} != {nulls[1]}")
                conn.commit()
                if engine == Dialect.MYSQL and not is_mariadb:
                    # MySQL reports some results as `int` in information_schema
                    # but sends them as BIGINT (LONGLONG) on the wire (CASE over
                    # integer literals, SMALLINT arithmetic, MAX of an INT read
                    # through a CTE; checked natively); no stored column type
                    # matches both. The catalog compares int and bigint as one
                    # family here; the exact wire type code (what readers get)
                    # is compared for every step below.
                    mv_cat = [tuple("bigint" if x == "int" else x for x in r) for r in mv_cat]
                    live_cat = [tuple("bigint" if x == "int" else x for x in r) for r in live_cat]
                if mv_cat != live_cat:
                    failures.append(f"{version} {name}: catalog {mv_cat} != {live_cat}")

            def check(step: str) -> None:
                for name, mv, live in pairs:
                    cur.execute(f"SELECT * FROM {mv}")
                    a = sorted((tuple(_exact(x) for x in r) for r in cur.fetchall()), key=repr)
                    da = _description(cur)
                    cur.execute(f"SELECT * FROM {live}")
                    b = sorted((tuple(_exact(x) for x in r) for r in cur.fetchall()), key=repr)
                    db = _description(cur)
                    conn.commit()
                    if a != b:
                        failures.append(f"{version} {name} {step}: rows {a} != {b}")
                    meta = _metadata_mismatches(engine, da, db)
                    if meta:
                        failures.append(f"{version} {name} {step}: {meta}")

            check("initial")
            for step, stmt in _DML:
                try:
                    cur.execute(stmt.format(t=t))
                    conn.commit()
                except Exception as exc:  # a maintenance trigger failing the DML is a defect
                    conn.rollback()
                    failures.append(f"{version} {step}: DML failed: {exc!r}")
                    continue
                check(step)
            cur.execute(f"UPDATE {t} SET i = 100, v = 'rb' WHERE id = 1")
            conn.rollback()
            check("rollback")
        finally:
            conn.rollback()
            if engine == Dialect.POSTGRES:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            else:
                cur.execute(
                    "SELECT TABLE_NAME, TABLE_TYPE FROM information_schema.TABLES "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE %s",
                    (f"%{tag}%",),
                )
                objs = cur.fetchall()
                for name, kind in sorted(objs, key=lambda o: o[1] != "VIEW"):
                    cur.execute(f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} IF EXISTS `{name}`")
            conn.commit()
            conn.close()
        return failures

    def test_mysql_group_key_integer_type_matches_the_length_model(self) -> None:
        """Generated integer arithmetic group keys: the type the compiler stores
        must equal the wire type MySQL sends for the live GROUP BY query, for
        every expression inside the verified subset; expressions outside it
        must fall back to the conservative BIGINT (documented gap in TODO.md,
        never a wrong INT). One table, one compile, one query: all expressions
        are keys of the same views."""
        import random
        import re

        mysql_engines = [(engine, dsn) for engine, dsn in _native_engines() if engine == Dialect.MYSQL]
        if not mysql_engines:
            self.skipTest("set MYSQL_DSN for native MySQL GROUP BY result-type tests")
        cols = ["ti", "tu", "s", "su", "mi", "mu", "i", "iu", "b", "bu"]
        decl = {"ti": "TINYINT", "tu": "TINYINT UNSIGNED", "s": "SMALLINT", "su": "SMALLINT UNSIGNED",
                "mi": "MEDIUMINT", "mu": "MEDIUMINT UNSIGNED", "i": "INT", "iu": "INT UNSIGNED",
                "b": "BIGINT", "bu": "BIGINT UNSIGNED"}
        rng = random.Random(20261123)

        def gen(depth: int) -> str:
            if depth == 0 or rng.random() < 0.3:
                return rng.choice(cols) if rng.random() < 0.75 else str(rng.choice([1, 2, 7, 100, 123456]))
            op = rng.choice(["+", "-", "*", "%", "+", "-"])
            return f"({gen(depth - 1)} {op} {gen(depth - 1)})"

        exprs = [e for e in (gen(rng.randint(1, 3)) for _ in range(160)) if re.search(r"[a-z].*[-+*%]|[-+*%].*[a-z]", e)][:50]
        # Boundaries of the length rule (signed/unsigned, every operator, literals, DIV, nesting).
        exprs += ["s - s", "s * 2", "-s", "-5 + s", "su + su", "su + s", "(s + s) % 3", "(su + su) % 3", "mi + mi",
                  "mi + tu", "mi % 2", "mu + mu", "mu - mu", "i + 1", "i % 2", "iu + 1", "b % 2", "ti * ti",
                  "ti * mi", "s DIV 2", "i DIV 2", "s + 123456", "s * 123456", "tu - tu", "ti + 99999", "s - su",
                  "(mu + mu) % 3", "(ti + ti) * (ti + ti)", "s % su", "su % s", "iu DIV 2", "s DIV su",
                  "CAST(s AS UNSIGNED)", "CAST(i AS SIGNED)"]
        exprs = sorted(set(exprs))
        schema = "CREATE TABLE gl (" + ", ".join(f"{c} {decl[c]}" for c in cols) + ");"
        for engine, dsn in mysql_engines:
            conn = _connect(engine, dsn)
            cur = conn.cursor()
            try:
                cur.execute("SELECT VERSION()")
                version = str(cur.fetchone()[0])
                if "mariadb" in version.lower():
                    continue
                m = re.match(r"(\d+)\.(\d+)", version)
                assert m is not None
                dialect = f"mysql:{m.group(1)}.{m.group(2)}"
                table = f"gl_{uuid.uuid4().hex[:8]}"
                cur.execute(schema.replace("gl", table, 1))
                # No rows: the result metadata does not depend on them, and
                # unsigned subtraction over real values would overflow.
                conn.commit()
                wrong, conservative = [], 0
                # Compile time grows quadratically with the number of keys of one
                # view, so the expressions are checked in small views.
                for start in range(0, len(exprs), 20):
                    chunk = exprs[start:start + 20]
                    keys = ", ".join(chunk)
                    cur.execute(f"SELECT {keys} FROM {table} GROUP BY {keys}")
                    actual = ["BIGINT" if d[1] == 8 else "INT" if d[1] == 3 else str(d[1]) for d in cur.description]
                    plan = generate_materialized_view(
                        schema.replace("gl", table, 1),
                        "CREATE VIEW v AS SELECT "
                        + ", ".join(f"{e} AS k{n}" for n, e in enumerate(chunk))
                        + f", COUNT(*) AS c FROM {table} GROUP BY {keys};",
                        dialect=dialect,
                        ignore_limitations=True,
                    )
                    ddl_types = dict(re.findall(r"(?m)^\s*[`\"]?(k\d+)[`\"]?\s+([A-Z]+)\b", plan.backing_table_ddl))
                    for n, e in enumerate(chunk):
                        stored = ddl_types[f"k{n}"]
                        if stored == actual[n]:
                            continue
                        if stored == "BIGINT":
                            conservative += 1  # outside the verified subset: the engine says INT
                        else:
                            wrong.append((e, stored, actual[n]))
                self.assertEqual(wrong, [], f"{version}: stored INT where the engine sends BIGINT")
                self.assertLess(conservative, len(exprs) * 0.2, f"{version}: subset too narrow ({conservative}/{len(exprs)})")
                cur.execute(f"DROP TABLE {table}")
                conn.commit()
            finally:
                conn.close()

    def test_mysql_group_key_integer_type_is_plan_dependent_engine_behaviour(self) -> None:
        mysql_engines = [(engine, dsn) for engine, dsn in _native_engines() if engine == Dialect.MYSQL]
        if not mysql_engines:
            self.skipTest("set MYSQL_DSN for native MySQL GROUP BY result-type tests")

        tested_mysql = False
        for engine, dsn in mysql_engines:
            conn = _connect(engine, dsn)
            cur = conn.cursor()
            try:
                cur.execute("SELECT VERSION()")
                version = str(cur.fetchone()[0])
                if "mariadb" in version.lower():
                    continue
                tested_mysql = True
                table = f"gk_{uuid.uuid4().hex[:8]}"
                cur.execute(
                    f"CREATE TEMPORARY TABLE {table} ("
                    "s SMALLINT NOT NULL, ti TINYINT NOT NULL, mi MEDIUMINT NOT NULL, "
                    "g_s_mod INT GENERATED ALWAYS AS (s % 2) STORED, "
                    "g_ti_mod INT GENERATED ALWAYS AS (ti % 2) STORED, "
                    "g_mi_mod INT GENERATED ALWAYS AS (mi % 2) STORED, "
                    "g_s_add INT GENERATED ALWAYS AS (s + s) STORED, "
                    "KEY idx_s_mod (g_s_mod), KEY idx_ti_mod (g_ti_mod), "
                    "KEY idx_mi_mod (g_mi_mod), KEY idx_s_add (g_s_add)"
                    ") ENGINE=InnoDB"
                )
                rows = [
                    (n % 65536 - 32768, n % 256 - 128, n % 16777216 - 8388608)
                    for n in range(1, 41)
                ]
                cur.executemany(
                    f"INSERT INTO {table} (s, ti, mi) VALUES (%s, %s, %s)",
                    rows,
                )
                conn.commit()

                cases = (
                    ("s % 2", "g_s_mod", "idx_s_mod"),
                    ("ti % 2", "g_ti_mod", "idx_ti_mod"),
                    ("mi % 2", "g_mi_mod", "idx_mi_mod"),
                    ("s + s", "g_s_add", "idx_s_add"),
                )
                all_indexes = "idx_s_mod, idx_ti_mod, idx_mi_mod, idx_s_add"
                for expr, generated_col, index_name in cases:
                    cur.execute(f"SELECT {expr} AS k FROM {table} LIMIT 0")
                    projection_type = cur.description[0][1]
                    cur.execute(f"SELECT {generated_col} AS k FROM {table} LIMIT 0")
                    expected_group_type = cur.description[0][1]
                    self.assertNotEqual(
                        projection_type,
                        expected_group_type,
                        f"{version}: plain projection unexpectedly has the generated INT type for {expr}",
                    )

                    access_results = []
                    for hint, expected_key in (
                        (f"IGNORE INDEX ({all_indexes})", None),
                        (f"FORCE INDEX ({index_name})", index_name),
                    ):
                        query = (
                            f"SELECT {expr} AS k, COUNT(*) AS c FROM {table} "
                            f"{hint} GROUP BY {expr}"
                        )
                        # Plain EXPLAIN is the tree format on MySQL 9.7, so the
                        # tabular form is requested explicitly and its columns
                        # are selected by name.
                        cur.execute("EXPLAIN FORMAT=TRADITIONAL " + query)
                        explain_names = [str(col[0]).lower() for col in cur.description]
                        key_position = explain_names.index("key")
                        explain_rows = cur.fetchall()
                        observed_keys = {row[key_position] for row in explain_rows}
                        if expected_key is None:
                            self.assertEqual(observed_keys, {None}, f"{version}: {query}")
                        else:
                            self.assertIn(expected_key, observed_keys, f"{version}: {query}")

                        cur.execute(query)
                        description = tuple(cur.description[0][:7])
                        actual_rows = sorted(cur.fetchall(), key=repr)
                        # Engine behaviour (MySQL 8.4 and 9.7, measured natively):
                        # the wire type of the SAME live GROUP BY query depends
                        # on the access path. Without a matching generated-column
                        # index the key is INT (the type the materialized reader
                        # stores); when the optimizer serves the GROUP BY from the
                        # index on the generated column that matches the key
                        # expression, the key is reported as BIGINT. No stored
                        # type matches both; the compiler follows the plan
                        # without such an index (documented exclusion in TODO.md).
                        # Values are identical either way.
                        if index_name in observed_keys:
                            self.assertEqual(
                                description[1],
                                projection_type,
                                f"{version}: grouped {expr} under {hint} (served by {index_name})",
                            )
                        else:
                            self.assertEqual(
                                description[1],
                                expected_group_type,
                                f"{version}: grouped {expr} metadata changed under {hint}",
                            )
                        access_results.append((description, actual_rows))

                    self.assertEqual(
                        access_results[0][1],
                        access_results[1][1],
                        f"{version}: grouped {expr} values depend on the access path",
                    )
                    # The no-matching-index plan is the one the reader mirrors.
                    self.assertEqual(access_results[0][0][1], expected_group_type, f"{version}: {expr}")
            finally:
                try:
                    conn.rollback()
                except Exception:
                    pass
                conn.close()

        if not tested_mysql:
            self.skipTest("MYSQL_DSN resolves to MariaDB; this MySQL-only type rule is not applicable")

    def test_native_result_types_match_live_views(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN for native result-type tests")
        for engine, dsn in engines:
            total, step = len(_views(engine, "t")), self.VIEWS_PER_TABLE
            for start in range(0, total, step):
                with self.subTest(engine=engine.value, views=f"{start}:{start + step}"):
                    failures = self._run(engine, dsn, slice(start, start + step))
                    self.assertEqual(failures, [], "\n".join(failures))


_INT_FAMILY = [
    "TINYINT", "TINYINT UNSIGNED", "SMALLINT", "SMALLINT UNSIGNED", "MEDIUMINT", "MEDIUMINT UNSIGNED",
    "INT", "INT UNSIGNED", "BIGINT", "BIGINT UNSIGNED", "YEAR", "BIT(8)", "BIT(16)", "BOOL", "TINYINT(1)",
]


_SUBQUERY_AGG_VIEWS = [
    "SELECT g, COUNT(*) AS n FROM {t} GROUP BY g HAVING COUNT(*) > (SELECT SUM(i) FROM {t}) - 100",
    "SELECT g, COUNT(*) AS n FROM {t} a GROUP BY g HAVING COUNT(*) > (SELECT SUM(b.i) FROM {t} b WHERE b.g = a.g)",
    "SELECT g, COUNT(*) AS n FROM {t} GROUP BY g HAVING MAX(i) < (SELECT AVG(i) FROM {t}) + 10",
]


class ScalarSubqueryAggregateFallbackTests(unittest.TestCase):
    """An aggregate inside a scalar subquery in HAVING is not decomposed.

    It used to be rewritten to the outer group's stored value: a reader with
    `(select __mv_elem_1 + 0 from t)` ("Subquery returns more than 1 row", or
    silently wrong on a one-row table); the correlated form failed with 1054.
    Now the HAVING clause is left whole and the view takes a fallback
    strategy, which must match the live view after every DML step (the
    uncorrelated subquery depends on every row of the table, not only on the
    touched groups).
    """

    def test_not_decomposed_offline(self) -> None:
        schema = _TABLE.format(t="t", dp=" PRECISION") + ";"
        for engine in ("sqlite", "postgres:18", "mysql:8.4", "mariadb:10.11"):
            for view in _SUBQUERY_AGG_VIEWS:
                with self.subTest(engine=engine, view=view):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        plan = generate_materialized_view(
                            schema.replace("DOUBLE PRECISION", "DOUBLE") if not engine.startswith("postgres") else schema,
                            f"CREATE VIEW v AS {view.format(t='t')};",
                            dialect=engine,
                            ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
                        )
                    self.assertNotEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
                    self.assertNotIn("__mv_elem", plan.reader_view_ddl or "")

    def test_fallback_is_announced(self) -> None:
        schema = _TABLE.format(t="t", dp=" PRECISION") + ";"
        view = _SUBQUERY_AGG_VIEWS[0].format(t="t")
        for engine in ("sqlite", "postgres:18", "mysql:8.4", "mariadb:10.11"):
            with self.subTest(engine=engine):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    plan = generate_materialized_view(
                        schema.replace("DOUBLE PRECISION", "DOUBLE") if not engine.startswith("postgres") else schema,
                        f"CREATE VIEW v AS {view};", dialect=engine, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
                    )
                lead = "exact result required: falling back to strategy group-scoped recompute. "
                msgs = [str(w.message) for w in caught if issubclass(w.category, MatviewDiagnosticWarning)]
                self.assertEqual(len(msgs), 1, msgs)
                self.assertIn(lead, msgs[0])
                self.assertEqual([d.code for d in plan.strategy_diagnostics], ["HAVING_SUBQUERY_AGGREGATE_FALLBACK"])
                self.assertTrue(plan.strategy_diagnostics[0].message.startswith(lead))
                self.assertIn("To change it:", plan.strategy_diagnostics[0].message)
                self.assertEqual("deadlock" in msgs[0], not engine.startswith(("sqlite", "postgres")))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_materialized_view(schema, "CREATE VIEW v AS SELECT g, COUNT(*) AS n FROM t GROUP BY g HAVING COUNT(*) > 1;",
                                       dialect="postgres:18")
        self.assertFalse([w for w in caught if issubclass(w.category, MatviewDiagnosticWarning)])

    def test_projected_subquery_aggregates_stay_refused(self) -> None:
        # Their result types are not derived (a fallback stored
        # SUM(x) + (SELECT MAX(v) ..) as DOUBLE where engines return BIGINT/DECIMAL).
        schema = _TABLE.format(t="t", dp=" PRECISION") + ";"
        for engine in ("sqlite", "postgres:18", "mysql:8.4", "mariadb:10.11"):
            for view in (
                "SELECT g, SUM(i) + (SELECT MAX(b) FROM t) AS s FROM t GROUP BY g",
                "SELECT g, COALESCE(MAX(i) + (SELECT SUM(b) FROM t), -1) AS s FROM t GROUP BY g",
                "SELECT g, SUM(i + (SELECT MAX(i) FROM t)) AS s FROM t GROUP BY g",
            ):
                with self.subTest(engine=engine, view=view):
                    with self.assertRaises(UnsupportedMaterializedViewError):
                        generate_materialized_view(
                            schema.replace("DOUBLE PRECISION", "DOUBLE") if not engine.startswith("postgres") else schema,
                            f"CREATE VIEW v AS {view};", dialect=engine, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
                        )

    def test_matches_live_view_natively(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN")
        steps = [
            ("initial", None),
            ("insert into a group", "INSERT INTO {t} (id, g, i) VALUES (100, 1, 40)"),
            ("insert elsewhere", "INSERT INTO {t} (id, g, i) VALUES (101, 7, 1)"),
            ("update elsewhere", "UPDATE {t} SET i = i + 60 WHERE id = 101"),
            ("delete", "DELETE FROM {t} WHERE id IN (100, 101)"),
        ]
        for engine, dsn in engines:
            conn = _connect(engine, dsn)
            cur = conn.cursor()
            tag = uuid.uuid4().hex[:8]
            t = f"sq_{tag}"
            try:
                if engine == Dialect.POSTGRES:
                    cur.execute(f'CREATE SCHEMA "{t}"')
                    cur.execute(f'SET search_path TO "{t}"')
                dp = " PRECISION" if engine == Dialect.POSTGRES else ""
                cur.execute(_TABLE.format(t=t, dp=dp))
                cur.executemany(f"INSERT INTO {t} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", _SEED)
                conn.commit()
                manager = MaterializedViewManager(conn, dialect=engine)
                for n, view in enumerate(_SUBQUERY_AGG_VIEWS):
                    sql = view.format(t=t)
                    mv, live = f"mv{n}_{tag}", f"lv{n}_{tag}"
                    cur.execute(f"CREATE VIEW {live} AS {sql}")
                    conn.commit()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        manager.register_view(f"CREATE VIEW {mv} AS {sql};", ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
                    conn.commit()
                    for step, stmt in steps:
                        if stmt:
                            cur.execute(stmt.format(t=t))
                            conn.commit()
                        with self.subTest(engine=engine.value, view=view, step=step):
                            cur.execute(f"SELECT g, n FROM {mv} ORDER BY g")
                            got = list(cur.fetchall())
                            cur.execute(f"SELECT g, n FROM {live} ORDER BY g")
                            self.assertEqual(got, list(cur.fetchall()))
                        conn.commit()
                    manager.drop_view(mv)
                    cur.execute(f"DROP VIEW {live}")
                    conn.commit()
            finally:
                conn.rollback()
                if engine == Dialect.POSTGRES:
                    cur.execute(f'DROP SCHEMA IF EXISTS "{t}" CASCADE')
                else:
                    cur.execute(f"DROP TABLE IF EXISTS {t}")
                conn.commit()
                conn.close()


_MINMAX_CONTEXT_TABLE = (
    "CREATE TABLE {t} (id INT PRIMARY KEY, g INT NOT NULL, ti TINYINT, tu TINYINT UNSIGNED, s SMALLINT, "
    "m MEDIUMINT UNSIGNED, i INT, iu INT UNSIGNED, b BIGINT, y YEAR, bt BIT(4), bo BOOL)"
)
_MINMAX_NARROW = ("ti", "tu", "s", "m", "y", "bt", "bo")
_MINMAX_CONTEXTS = {
    "grpco": "SELECT g, COALESCE(MAX({c}), 0) AS a FROM {t} GROUP BY g",
    "grpp": "SELECT g, MAX({c}) + 0 AS a FROM {t} GROUP BY g",
    "winco": "SELECT id, COALESCE(MIN({c}) OVER (PARTITION BY g), 0) AS a FROM {t}",
    "winp": "SELECT id, MAX({c}) OVER (PARTITION BY g) + 1 AS a FROM {t}",
    "dwin": "SELECT d.id, MAX(d.{c}) OVER (PARTITION BY d.g) AS a FROM (SELECT id, g, {c} FROM {t}) d",
    "plain": "SELECT g, MIN({c}) AS a, MAX({c}) AS z FROM {t} GROUP BY g",
    "having": "SELECT g, COUNT(*) AS n FROM {t} GROUP BY g HAVING MAX({c}) + 0 >= 0",
}


def _minmax_refused(mariadb: bool, col: str, ctx: str) -> bool:
    """The measured rule (see MySQLDialectMixin._mysql_untyped_minmax_uses)."""
    if ctx in ("plain", "having"):
        return False
    if mariadb:
        return ctx in ("winco", "winp") and (col in _MINMAX_NARROW or col == "iu")
    return col in _MINMAX_NARROW


def _minmax_context_views() -> List[Tuple[str, str, str]]:
    return [(col, ctx, sql) for col in _MINMAX_NARROW + ("i", "iu", "b") for ctx, sql in _MINMAX_CONTEXTS.items()]


class MinMaxContextTypingTests(unittest.TestCase):
    """MIN/MAX of integer-like arguments: exact where typed, refused where the server types by context.

    Matrix: argument type (TINYINT, TINYINT UNSIGNED, SMALLINT, MEDIUMINT
    UNSIGNED, YEAR, BIT(4), BOOL, INT, INT UNSIGNED, BIGINT) x context
    (COALESCE/arithmetic around grouped and window MIN/MAX, window over a
    derived table, plain projection, HAVING). Natively every view must be
    exactly what `_minmax_refused` predicts: refused (not bypassable), or
    exact in rows and wire type codes through DML.
    """

    def test_offline_decisions(self) -> None:
        schema = _MINMAX_CONTEXT_TABLE.format(t="t") + ";"
        for engine in ("mysql:8.4.13", "mysql:9.7.4", "mariadb:10.11"):
            for col, ctx, sql in _minmax_context_views():
                with self.subTest(engine=engine, col=col, ctx=ctx):
                    compile_ = lambda: generate_materialized_view(
                        schema, f"CREATE VIEW v AS {sql.format(c=col, t='t')};", dialect=engine,
                        ignore_limitations=True, accept_limitations=FLOAT_OPT_IN,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        if _minmax_refused(engine.startswith("mariadb"), col, ctx):
                            with self.assertRaises(UnsupportedMaterializedViewError):
                                compile_()
                        else:
                            compile_()

    def test_native_exact_or_refused(self) -> None:
        engines = [(e, d) for e, d in _native_engines() if e == Dialect.MYSQL]
        if not engines:
            self.skipTest("set MYSQL_DSN")
        for engine, dsn in engines:
            # One fresh table per chunk of views (see NativeResultTypeTests.VIEWS_PER_TABLE).
            all_views = list(enumerate(_minmax_context_views()))
            per_table = NativeResultTypeTests.VIEWS_PER_TABLE
            for start in range(0, len(all_views), per_table):
                with self.subTest(views=f"{start}:{start + per_table}"):
                    conn = _connect(engine, dsn)
                    cur = conn.cursor()
                    cur.execute("SELECT VERSION()")
                    mariadb = "mariadb" in str(cur.fetchone()[0]).lower()
                    conn.commit()
                    tag = uuid.uuid4().hex[:8]
                    t = f"mc_{tag}"
                    failures: List[str] = []
                    try:
                        cur.execute(_MINMAX_CONTEXT_TABLE.format(t=t))
                        cur.execute(
                            f"INSERT INTO {t} VALUES (1, 1, -3, 200, 7, 5, 5, 4000000000, 9, 2001, b'101', 1), "
                            f"(2, 1, 4, 3, -2, 1, -8, 1, -7, 1999, b'1', 0), "
                            f"(3, 2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
                        )
                        conn.commit()
                        manager = MaterializedViewManager(conn, dialect=engine)
                        pairs = []
                        for n, (col, ctx, sql_t) in all_views[start:start + per_table]:
                            sql = sql_t.format(c=col, t=t)
                            refused = _minmax_refused(mariadb, col, ctx)
                            mv, live = f"mv{n}_{tag}", f"lv{n}_{tag}"
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", EngineLimitationWarning)
                                    manager.register_view(f"CREATE VIEW {mv} AS {sql};", ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
                            except UnsupportedMaterializedViewError:
                                conn.rollback()
                                if not refused:
                                    failures.append(f"unexpected refusal: {sql}")
                                continue
                            conn.commit()
                            if refused:
                                failures.append(f"not refused: {sql}")
                            cur.execute(f"CREATE VIEW {live} AS {sql}")
                            conn.commit()
                            pairs.append((sql, mv, live))
                        for step, stmt in (
                            ("initial", None),
                            ("insert", f"INSERT INTO {t} VALUES (4, 2, 0, 0, 0, 0, 0, 0, 0, 2000, b'0', 1), "
                                       f"(5, 1, 100, 250, 30000, 16000000, 2000000000, 4294967295, 5, 2155, b'1111', 0)"),
                            ("update", f"UPDATE {t} SET g = 2 WHERE id = 1"),
                            ("delete", f"DELETE FROM {t} WHERE id = 2"),
                        ):
                            if stmt:
                                cur.execute(stmt)
                                conn.commit()
                            for sql, mv, live in pairs:
                                cur.execute(f"SELECT * FROM {mv}")
                                a = sorted((tuple(_exact(x) for x in r) for r in cur.fetchall()), key=repr)
                                da = _description(cur)
                                cur.execute(f"SELECT * FROM {live}")
                                b = sorted((tuple(_exact(x) for x in r) for r in cur.fetchall()), key=repr)
                                db = _description(cur)
                                conn.commit()
                                if a != b:
                                    failures.append(f"{step} {sql}: rows {a[:3]} != {b[:3]}")
                                meta = _metadata_mismatches(engine, da, db)
                                if meta:
                                    failures.append(f"{step} {sql}: {meta}")
                    finally:
                        conn.rollback()
                        cur.execute(
                            "SELECT TABLE_NAME, TABLE_TYPE FROM information_schema.TABLES "
                            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE %s", (f"%{tag}%",),
                        )
                        for name, kind in sorted(cur.fetchall(), key=lambda o: o[1] != "VIEW"):
                            cur.execute(f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} IF EXISTS `{name}`")
                        conn.commit()
                        conn.close()
                    self.assertEqual(failures, [], "\n".join(failures))


class MySQLBitWindowVersionGateTests(unittest.TestCase):
    """Offline: window MIN/MAX over BIT is typed only for measured MySQL releases."""

    SCHEMA = "CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, b BIT(8));"
    VIEW = "CREATE VIEW v AS SELECT id, MAX(b) OVER (PARTITION BY g) AS r FROM t;"

    def _col(self, engine: str) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            plan = generate_materialized_view(self.SCHEMA, self.VIEW, dialect=engine, ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
        return next(l.strip().rstrip(",") for l in plan.backing_table_ddl.splitlines() if l.strip().startswith("r "))

    def test_measured_releases_are_typed(self) -> None:
        self.assertEqual(self._col("mysql:8.4.13"), "r INT UNSIGNED")
        self.assertEqual(self._col("mysql:9.7.4"), "r BIT(8)")
        self.assertEqual(self._col("mariadb:10.11"), "r BIT(8)")

    def test_unmeasured_releases_are_refused(self) -> None:
        for engine in ("mysql", "mysql:8.0.40", "mysql:9.1", "mysql:9.6.0", "mysql:9.8", "mysql:10.0"):
            with self.subTest(engine=engine):
                with self.assertRaises(UnsupportedMaterializedViewError):
                    self._col(engine)


class NativeMySQLIntegerMinMaxTests(unittest.TestCase):
    """Window / CTE / grouped MIN and MAX over every MySQL integer-like type.

    MySQL widens these results (and MariaDB does not); the materialized view
    must return the type readers of the live view receive.
    """

    def test_integer_minmax_types_match_live_views(self) -> None:
        engines = [(e, d) for e, d in _native_engines() if e == Dialect.MYSQL]
        if not engines:
            self.skipTest("set MYSQL_DSN for the MySQL integer MIN/MAX matrix")
        for engine, dsn in engines:
            # Three or four views per type on one fresh table per type group
            # (see NativeResultTypeTests.VIEWS_PER_TABLE).
            for start in range(0, len(_INT_FAMILY), 4):
                with self.subTest(types=_INT_FAMILY[start:start + 4]):
                    failures = self._run(engine, dsn, range(start, min(start + 4, len(_INT_FAMILY))))
                    self.assertEqual(failures, [], "\n".join(failures))

    def _run(self, engine: Dialect, dsn: str, type_indexes: range) -> List[str]:
        conn = _connect(engine, dsn)
        cur = conn.cursor()
        cur.execute("SELECT VERSION()")
        version = str(cur.fetchone()[0])
        is_mariadb = "mariadb" in version.lower()
        conn.commit()
        tag = uuid.uuid4().hex[:8]
        t = f"im_{tag}"
        cols = ", ".join(f"c{i} {ty}" for i, ty in enumerate(_INT_FAMILY))

        def value(ty: str, n: int) -> str:
            if ty == "YEAR":
                return str(2000 + n)
            if ty.startswith("BIT"):
                return f"b'{n:b}'"
            if ty in ("BOOL", "TINYINT(1)"):
                return str(n % 2)
            return str(n)

        def row(rid: int, g: int, n: int) -> str:
            return f"({rid}, {g}, " + ", ".join(value(ty, n) for ty in _INT_FAMILY) + ")"

        failures: List[str] = []
        try:
            cur.execute(f"CREATE TABLE {t} (id INT PRIMARY KEY, g INT NOT NULL, {cols}) ENGINE=InnoDB")
            cur.execute(f"INSERT INTO {t} VALUES " + ", ".join(row(i, i % 2, i * 3 % 7 + 1) for i in range(1, 7)))
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=engine)
            pairs = []
            for i in type_indexes:
                ty = _INT_FAMILY[i]
                for kind, select in (
                    ("winmin", f"SELECT id, MIN(c{i}) OVER (PARTITION BY g) AS r FROM {t}"),
                    ("winmax", f"SELECT id, MAX(c{i}) OVER (PARTITION BY g) AS r FROM {t}"),
                    ("ctemax", f"WITH a AS (SELECT g, c{i} FROM {t}) SELECT g, MAX(c{i}) AS r FROM a GROUP BY g"),
                    ("grpmin", f"SELECT g, MIN(c{i}) AS r FROM {t} GROUP BY g"),
                ):
                    if kind == "grpmin" and ty.startswith("BIT"):
                        # Pre-existing, loud: grouped MIN(BIT) fails to register
                        # ("Data too long"); recorded in TODO.md.
                        continue
                    name = f"{kind}_{i}"
                    mv, live = f"mv_{name}_{tag}", f"lv_{name}_{tag}"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        manager.register_view(f"CREATE VIEW {mv} AS {select};", ignore_limitations=True, accept_limitations=FLOAT_OPT_IN)
                    conn.commit()
                    cur.execute(f"CREATE VIEW {live} AS {select}")
                    conn.commit()
                    pairs.append((f"{ty} {kind}", mv, live))

            def check(step: str) -> None:
                for label, mv, live in pairs:
                    cur.execute(f"SELECT * FROM {mv}")
                    a = sorted((tuple(_exact(x) for x in r) for r in cur.fetchall()), key=repr)
                    da = _description(cur)
                    cur.execute(f"SELECT * FROM {live}")
                    b = sorted((tuple(_exact(x) for x in r) for r in cur.fetchall()), key=repr)
                    db = _description(cur)
                    conn.commit()
                    if a != b:
                        failures.append(f"{version} {label} {step}: rows {a[:3]} != {b[:3]}")
                    meta = _metadata_mismatches(engine, da, db)
                    if meta:
                        failures.append(f"{version} {label} {step}: {meta}")

            check("initial")
            for step, stmt in (
                ("insert", f"INSERT INTO {t} VALUES " + row(20, 0, 6) + ", " + row(21, 1, 0)),
                ("update", f"UPDATE {t} SET g = 1 - g WHERE id = 2"),
                ("delete", f"DELETE FROM {t} WHERE id IN (1, 20)"),
            ):
                cur.execute(stmt)
                conn.commit()
                check(step)
        finally:
            conn.rollback()
            cur.execute(
                "SELECT TABLE_NAME, TABLE_TYPE FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME LIKE %s", (f"%{tag}%",),
            )
            for name, kind in sorted(cur.fetchall(), key=lambda o: o[1] != "VIEW"):
                cur.execute(f"DROP {'VIEW' if kind == 'VIEW' else 'TABLE'} IF EXISTS `{name}`")
            conn.commit()
            conn.close()
        del is_mariadb
        return failures


def _exact_deep(value: Any) -> Any:
    """``_exact`` that also compares array elements exactly (Decimal scale, float bits, types)."""
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_exact_deep(v) for v in value))
    return _exact(value)


class SetopCommonTypesPgCatalogAndNameTests(unittest.TestCase):
    """Offline and native verification for:
    1. Set-op and UNION ALL aggregate common column types across branches,
    2. PostgreSQL pg_catalog.<agg>(...) parsing and compound/HAVING decomposition,
    3. PostgreSQL NAME result types in COALESCE/CASE/NULLIF.
    """

    def test_setop_and_union_all_agg_common_types_offline(self) -> None:
        from matview import EngineLimitationError

        pg_schema = (
            "CREATE TABLE a (id INT PRIMARY KEY, k SMALLINT NOT NULL, x SMALLINT, d NUMERIC(6, 2), s VARCHAR(5), "
            "tm TIME, arr SMALLINT[], b BIT(2), n NAME);"
            "CREATE TABLE b (id INT PRIMARY KEY, k BIGINT NOT NULL, y BIGINT, f DOUBLE PRECISION, d NUMERIC(12, 4), "
            "s VARCHAR(50), tm TIMETZ, arr BIGINT[], b BIT(8), tx TEXT);"
        )
        p_uall = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_u AS SELECT id, x, d, s, tm, arr, b FROM a UNION ALL SELECT id, y, d, s, tm, arr, b FROM b;",
            dialect="postgres:18",
        )
        self.assertRegex(p_uall.backing_table_ddl, r"\bx BIGINT\b")
        self.assertRegex(p_uall.backing_table_ddl, r"\bd NUMERIC\b")
        self.assertRegex(p_uall.backing_table_ddl, r"\bs VARCHAR\b")
        self.assertRegex(p_uall.backing_table_ddl, r"\btm TIMETZ\b")
        self.assertRegex(p_uall.backing_table_ddl, r"\barr BIGINT\[\]")
        self.assertRegex(p_uall.backing_table_ddl, r'\bb pg_catalog\."bit"(?=[,\s\)])')

        # Positional alignment with differing branch column names (a.x vs b.y),
        # MIN/MAX typmod dropping, BIT_XOR, AVG, compound expressions, and large literals.
        p_uagg = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_ua AS WITH u AS (SELECT k, x, s, d, b FROM a UNION ALL SELECT k, y, s, d, b FROM b) "
            "SELECT k, MIN(x) AS mnx, MAX(x) AS mxx, SUM(x) AS sx, MIN(s) AS mns, MIN(d) AS mnd, "
            "BIT_XOR(x) AS bx, MIN(x + 1) AS mnx1 FROM u GROUP BY k;",
            dialect="postgres:18",
        )
        self.assertRegex(p_uagg.backing_table_ddl, r"\bk BIGINT\b")
        self.assertRegex(p_uagg.backing_table_ddl, r"\bmnx BIGINT\b")
        self.assertRegex(p_uagg.backing_table_ddl, r"\bmxx BIGINT\b")
        self.assertRegex(p_uagg.backing_table_ddl, r"\bsx NUMERIC\b")
        self.assertRegex(p_uagg.backing_table_ddl, r"\bmns TEXT\b")
        self.assertRegex(p_uagg.backing_table_ddl, r"\bmnd NUMERIC\b")
        self.assertRegex(p_uagg.backing_table_ddl, r"\bbx BIGINT\b")
        self.assertRegex(p_uagg.backing_table_ddl, r"\bmnx1 BIGINT\b")

        # Float branch in UNION ALL agg must trigger ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE by default,
        # and widen SUM/AVG/VAR_POP to DOUBLE PRECISION when explicitly accepted.
        float_uagg_sql = (
            "CREATE VIEW v_uf AS WITH u AS (SELECT k, x FROM a UNION ALL SELECT k, f FROM b) "
            "SELECT k, SUM(x) AS sx, AVG(x) AS ax, VAR_POP(x) AS vp FROM u GROUP BY k;"
        )
        with self.assertRaises(EngineLimitationError):
            generate_materialized_view(pg_schema, float_uagg_sql, dialect="postgres:18")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            p_uf = generate_materialized_view(
                pg_schema,
                float_uagg_sql,
                dialect="postgres:18",
                accept_limitations=("ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE",),
            )
        self.assertRegex(p_uf.backing_table_ddl, r"\bsx DOUBLE PRECISION\b")
        self.assertRegex(p_uf.backing_table_ddl, r"\bax DOUBLE PRECISION\b")
        self.assertRegex(p_uf.backing_table_ddl, r"\bvp DOUBLE PRECISION\b")

        # NULL, (NULL), typmod dropping, untyped string literal + NAME, and Tier 3 ORDER BY set-op widening.
        p_null = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_un AS SELECT id, (NULL) AS x, NULL AS d, 'lit' AS n FROM a "
            "UNION ALL SELECT id, y, d, n FROM b JOIN a a2 ON a2.id = b.id;",
            dialect="postgres:18",
        )
        self.assertRegex(p_null.backing_table_ddl, r"\bx BIGINT\b")
        self.assertRegex(p_null.backing_table_ddl, r"\bd NUMERIC\b")
        self.assertRegex(p_null.backing_table_ddl, r"\bn NAME\b")

        p_t3 = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_t3 AS SELECT id, x FROM a UNION ALL SELECT id, y FROM b ORDER BY id;",
            dialect="postgres:18",
        )
        self.assertRegex(p_t3.backing_table_ddl, r"\bx BIGINT\b")

        my_schema = (
            "CREATE TABLE a (id INT PRIMARY KEY, k SMALLINT NOT NULL, x SMALLINT, u INT UNSIGNED, "
            "bu BIGINT UNSIGNED, z BIGINT ZEROFILL, fl BOOL, bt BIT(4), d DECIMAL(8, 2), s VARCHAR(5), "
            "s500 VARCHAR(500), dt DATE);"
            "CREATE TABLE b (id INT PRIMARY KEY, k BIGINT NOT NULL, y BIGINT, u INT, bu BIGINT, z INT, "
            "fl BIGINT, bt BIT(16), d DECIMAL(12, 6), s TEXT, tt TINYTEXT, dt DATETIME(6));"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            pm_uall = generate_materialized_view(
                my_schema,
                "CREATE VIEW v_u AS SELECT id, x, u, bu, z, fl, bt, d, s, s500, dt FROM a "
                "UNION ALL SELECT id, y, u, bu, z, fl, bt, d, s, tt, dt FROM b;",
                dialect="mysql:8.4",
                ignore_limitations=True,
            )
            pm_uagg = generate_materialized_view(
                my_schema,
                "CREATE VIEW v_ua AS WITH u AS (SELECT k, x, d FROM a UNION ALL SELECT k, y, d FROM b) "
                "SELECT k, MIN(x) AS mnx, MAX(x) AS mxx, SUM(x) AS sx, AVG(d) AS ad FROM u GROUP BY k;",
                dialect="mysql:8.4",
                ignore_limitations=True,
            )
            pm_numstr = generate_materialized_view(
                my_schema,
                "CREATE VIEW v_ns AS SELECT id, x FROM a UNION ALL SELECT id, s500 FROM a;",
                dialect="mysql:8.4",
                ignore_limitations=True,
            )
            pm_uagg_int = generate_materialized_view(
                "CREATE TABLE a (id INT PRIMARY KEY, k INT NOT NULL, x SMALLINT);"
                "CREATE TABLE b (id INT PRIMARY KEY, k INT NOT NULL, y INT);",
                "CREATE VIEW v_uai AS WITH u AS (SELECT k, x FROM a UNION ALL SELECT k, y FROM b) "
                "SELECT k, MIN(x) AS mnx, MAX(x) AS mxx FROM u GROUP BY k;",
                dialect="mysql:8.4",
                ignore_limitations=True,
            )
        self.assertRegex(pm_uall.backing_table_ddl, r"\bx BIGINT\b")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bu BIGINT\b")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bbu DECIMAL\(20, 0\)")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bz DECIMAL\(20, 0\)")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bfl BIGINT\b")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bbt BIT\(16\)")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bd DECIMAL\(12, 6\)")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bs TEXT\b")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bs500 TEXT\b")
        self.assertRegex(pm_uall.backing_table_ddl, r"\bdt DATETIME\(6\)")
        self.assertRegex(pm_uagg.backing_table_ddl, r"\bk BIGINT\b")
        self.assertRegex(pm_uagg.backing_table_ddl, r"\bmnx BIGINT\b")
        self.assertRegex(pm_uagg.backing_table_ddl, r"\bmxx BIGINT\b")
        self.assertRegex(pm_uagg.backing_table_ddl, r"\bsx DECIMAL\(65, 0\)")
        self.assertRegex(pm_uagg.backing_table_ddl, r"\bad DECIMAL\(16, 10\)")
        self.assertRegex(pm_numstr.backing_table_ddl, r"\bx VARCHAR\(500\)")
        # MySQL materializes UNION ALL CTEs, keeping the widened INT type for MIN/MAX (not BIGINT).
        self.assertRegex(pm_uagg_int.backing_table_ddl, r"\bmnx INT\b")
        self.assertRegex(pm_uagg_int.backing_table_ddl, r"\bmxx INT\b")

    def test_pg_catalog_aggregates_and_name_coalesce_offline(self) -> None:
        from matview import EngineLimitationError, MaintenanceStrategy, UnsupportedMaterializedViewError, parse_view_definition

        pg_schema = "CREATE TABLE t (id INT PRIMARY KEY, g INT, x INT, f DOUBLE PRECISION, n NAME, c CHAR(4), vc VARCHAR(8), tx TEXT);"

        # Elementary, FILTER, parenthesized, and compound pg_catalog.<agg>(...) calls,
        # including SQLite-default pre-parsed ViewSpec.
        view_sql = (
            "CREATE VIEW v_pgcat AS SELECT g, pg_catalog.var_pop(x) AS vp, pg_catalog.variance(x) AS vs, "
            'pg_catalog."stddev"(x) AS sd, COALESCE(pg_catalog.sum(x), 0) AS cs, '
            "pg_catalog.sum(x) FILTER (WHERE x > 0) AS sf, "
            "COALESCE(pg_catalog.sum(x) FILTER (WHERE x > 0), 0) AS csf "
            "FROM t GROUP BY g HAVING pg_catalog.count(*) > 0 AND pg_catalog.sum(x) FILTER (WHERE x > 0) >= 0;"
        )
        for v_input in (view_sql, parse_view_definition(view_sql)):
            plan = generate_materialized_view(pg_schema, v_input, dialect="postgres:18")
            self.assertEqual(plan.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
            self.assertNotIn("pg_catalog.__mv_elem", plan.reader_view_ddl)
            self.assertNotIn("pg_catalog.", plan.reader_view_ddl)
            self.assertIn("VAR_POP", plan.initial_populate_sql)
            self.assertIn("VAR_SAMP", plan.initial_populate_sql)
            self.assertIn("STDDEV_SAMP", plan.initial_populate_sql)
            self.assertRegex(plan.backing_table_ddl, r"\bsf BIGINT\b")

        # pg_catalog.<agg>(...) inside scalar subqueries must trigger HAVING fallback / outer+subquery refusal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_hav_sub = generate_materialized_view(
                pg_schema,
                "CREATE VIEW v_hs AS SELECT g, COUNT(*) AS c FROM t GROUP BY g "
                "HAVING COUNT(*) > (SELECT pg_catalog.count(*) FROM t);",
                dialect="postgres:18",
            )
        self.assertEqual(p_hav_sub.strategy, MaintenanceStrategy.GROUP_SCOPED_DELTA)
        with self.assertRaises(UnsupportedMaterializedViewError):
            generate_materialized_view(
                pg_schema,
                "CREATE VIEW v_mix AS SELECT g, SUM(x) + (SELECT pg_catalog.max(x) FROM t) AS s FROM t GROUP BY g;",
                dialect="postgres:18",
            )

        # pg_catalog.<agg>(...) OVER (...) on DOUBLE must still raise EngineLimitationError.
        with self.assertRaises(EngineLimitationError):
            generate_materialized_view(
                pg_schema,
                "CREATE VIEW v_win_pgcat AS SELECT id, pg_catalog.sum(f) OVER (PARTITION BY g) AS s FROM t;",
                dialect="postgres:18",
            )

        # Non-decomposed Tier 3 pg_catalog.<agg>(...) preserves exact result types.
        p_t3_pgcat = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_t3_pgcat AS SELECT g, COALESCE(pg_catalog.sum(x), 0) AS cs, "
            "pg_catalog.var_pop(x) + pg_catalog.stddev(x) AS sv FROM t GROUP BY g ORDER BY g LIMIT 5;",
            dialect="postgres:18",
        )
        self.assertRegex(p_t3_pgcat.backing_table_ddl, r"\bcs BIGINT\b")
        self.assertRegex(p_t3_pgcat.backing_table_ddl, r"\bsv NUMERIC\b")

        # pg_catalog.std(x) is undefined on PostgreSQL and must be refused
        with self.assertRaises(UnsupportedMaterializedViewError):
            generate_materialized_view(
                pg_schema,
                "CREATE VIEW v_bad_std AS SELECT g, pg_catalog.std(x) AS s FROM t GROUP BY g;",
                dialect="postgres:18",
            )

        # PostgreSQL NAME result types in COALESCE, CASE, NULLIF, GREATEST/LEAST, and MIN/MAX
        p_name = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_name AS SELECT id, "
            "COALESCE(n, c) AS n_c, "
            "COALESCE(c, n) AS c_n, "
            "COALESCE(vc, n) AS vc_n, "
            "COALESCE(tx, n) AS tx_n, "
            "CASE WHEN id > 0 THEN c ELSE n END AS case_else_n, "
            "CASE WHEN id > 0 THEN n ELSE tx END AS case_else_tx, "
            "NULLIF(n, c) AS nif_nc, "
            "NULLIF(c, n) AS nif_cn, "
            "NULLIF(vc, n) AS nif_vcn, "
            "NULLIF(vc, c) AS nif_vcc, "
            "NULLIF('lit', n) AS nif_litn, "
            "GREATEST(c, n) AS gr_cn, "
            "LEAST(n, tx) AS ls_ntx, "
            "GREATEST(tx, n) AS gr_txn, "
            "GREATEST(c, vc) AS gr_cvc "
            "FROM t;",
            dialect="postgres:18",
        )
        self.assertRegex(p_name.backing_table_ddl, r"\bn_c NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bc_n NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bvc_n NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\btx_n TEXT\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bcase_else_n NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bcase_else_tx TEXT\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bnif_nc NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bnif_cn TEXT\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bnif_vcn TEXT\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bnif_vcc BPCHAR\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bnif_litn NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bgr_cn NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bls_ntx NAME\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bgr_txn TEXT\b")
        self.assertRegex(p_name.backing_table_ddl, r"\bgr_cvc BPCHAR\b")

        p_mm_name = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_mm_n AS SELECT g, MIN(n) AS mnn, MAX(n) AS mxn FROM t GROUP BY g;",
            dialect="postgres:18",
        )
        self.assertRegex(p_mm_name.backing_table_ddl, r"\bmnn TEXT\b")
        self.assertRegex(p_mm_name.backing_table_ddl, r"\bmxn TEXT\b")

    def test_postgres_cross_branch_union_array_scalar_and_collation_widening_offline(self) -> None:
        pg_schema = (
            "CREATE TABLE a ("
            "  id INT PRIMARY KEY, "
            "  ai2 SMALLINT[], ai2_2d SMALLINT[][], ai4 INT[], "
            "  avc5 VARCHAR(5)[], ach4 CHAR(4)[], atxt TEXT[], at_c TEXT[] COLLATE \"C\", "
            "  ab4 BIT(4)[], avb VARBIT(8)[], "
            "  ad DATE[], ats2 TIMESTAMP(2) WITHOUT TIME ZONE[], atm2 TIME(2) WITHOUT TIME ZONE[], "
            "  aiv2 INTERVAL(2)[], acidr CIDR[], "
            "  vc5 VARCHAR(5), ch4 CHAR(4), txt TEXT, t_c TEXT COLLATE \"C\", "
            "  b4 BIT(4), vb VARBIT(8), d DATE, ts2 TIMESTAMP(2) WITHOUT TIME ZONE, "
            "  tm2 TIME(2) WITHOUT TIME ZONE, num NUMERIC(6, 2), cidr_c CIDR, "
            "  vc_unb VARCHAR, avc255 VARCHAR(255)[], x SMALLINT"
            ");"
            "CREATE TABLE b ("
            "  id INT PRIMARY KEY, "
            "  ai8 BIGINT[], ai8_2d BIGINT[][], anum NUMERIC(10, 2)[], af4 REAL[], af8 DOUBLE PRECISION[], "
            "  avc20 VARCHAR(20)[], ach4 CHAR(4)[], atxt TEXT[], at_c TEXT[] COLLATE \"C\", anm NAME[], "
            "  ab8 BIT(8)[], avb VARBIT[], "
            "  ats4 TIMESTAMP(4) WITHOUT TIME ZONE[], atstz4 TIMESTAMP(4) WITH TIME ZONE[], atmtz4 TIME(4) WITH TIME ZONE[], "
            "  aiv4 INTERVAL(4)[], ainet INET[], "
            "  vc20 VARCHAR(20), txt TEXT, t_c TEXT COLLATE \"C\", nm NAME, "
            "  b8 BIT(8), vb VARBIT, ts4 TIMESTAMP(4) WITHOUT TIME ZONE, "
            "  tstz4 TIMESTAMP(4) WITH TIME ZONE, tmtz4 TIME(4) WITH TIME ZONE, "
            "  inet_c INET, vc_unb VARCHAR, avc255 VARCHAR(255)[], y BIGINT"
            ");"
        )

        # 1. Mixed array dimensions, element widening, unknown literals, bit/temporal/network arrays,
        #    and single-branch non-default collation propagation.
        p1 = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_u1 AS "
            "SELECT id, ai2_2d AS arr_dim, ai4 AS arr_num, ai4 AS arr_f4, ai4 AS arr_lit, "
            "  avc5 AS avc_lit, avc5 AS avc_ch, ach4 AS ach_vc, atxt AS atxt_nm, avc5 AS avc_nm, "
            "  at_c AS at_coll, t_c AS t_coll, vc5 AS vc_coll, "
            "  ab4 AS ab_b, ab4 AS ab_vb, avb AS avb_b, b4 AS b_vb, vb AS vb_b, "
            "  ad AS ad_ts, ats2 AS ats_tz, atm2 AS atm_tz, aiv2 AS aiv, acidr AS anet, cidr_c AS net, "
            "  vc5 AS vc_lit, num AS num_lit, vc_unb, avc255 "
            "FROM a "
            "UNION ALL "
            "SELECT id, ai8, anum, af4, '{1,2}', "
            "  '{abcdef}', ach4, avc20, anm, anm, "
            "  atxt, txt, t_c, "
            "  ab8, avb, ab8, vb, b8, "
            "  ats4, atstz4, atmtz4, aiv4, ainet, inet_c, "
            "  'abcdef', '1234567.891', vc_unb, avc255 "
            "FROM b;",
            dialect="postgres:18",
        )
        ddl1 = p1.backing_table_ddl
        self.assertRegex(ddl1, r"\barr_dim BIGINT\[\]")
        self.assertRegex(ddl1, r"\barr_num NUMERIC\[\]")
        self.assertRegex(ddl1, r"\barr_f4 REAL\[\]")
        self.assertRegex(ddl1, r"\barr_lit INT(?:EGER)?\[\]")
        self.assertRegex(ddl1, r"\bavc_lit VARCHAR\[\]")
        self.assertRegex(ddl1, r"\bavc_ch VARCHAR\[\]")
        self.assertRegex(ddl1, r"\bach_vc BPCHAR\[\]")
        self.assertRegex(ddl1, r'\batxt_nm TEXT\[\] COLLATE "C"')
        self.assertRegex(ddl1, r"\bavc_nm NAME\[\]")
        self.assertRegex(ddl1, r'\bat_coll TEXT\[\] COLLATE "C"')
        self.assertRegex(ddl1, r'\bt_coll TEXT COLLATE "C"')
        self.assertRegex(ddl1, r'\bvc_coll VARCHAR COLLATE "C"')
        self.assertRegex(ddl1, r'\bab_b pg_catalog\."bit"\[\]')
        self.assertRegex(ddl1, r'\bab_vb pg_catalog\."bit"\[\]')
        self.assertRegex(ddl1, r"\bavb_b BIT VARYING\[\]")
        self.assertRegex(ddl1, r'\bb_vb pg_catalog\."bit"(?=[,\s\)])')
        self.assertRegex(ddl1, r"\bvb_b BIT VARYING\b")
        self.assertRegex(ddl1, r"\bad_ts TIMESTAMP\[\]")
        self.assertRegex(ddl1, r"\bats_tz TIMESTAMPTZ\[\]")
        self.assertRegex(ddl1, r"\batm_tz TIMETZ\[\]")
        self.assertRegex(ddl1, r"\baiv INTERVAL\[\]")
        self.assertRegex(ddl1, r"\banet INET\[\]")
        self.assertRegex(ddl1, r"\bnet INET\b")
        self.assertRegex(ddl1, r"\bvc_lit VARCHAR\b")
        self.assertRegex(ddl1, r"\bnum_lit NUMERIC\b")
        self.assertRegex(ddl1, r"\bvc_unb VARCHAR\b")
        self.assertNotIn("VARCHAR(255),", ddl1)
        self.assertRegex(ddl1, r"\bavc255 VARCHAR\(255\)\[\]")

        # 2. UNION trigger bodies must cast widened / unknown-literal branches to the
        #    common type before canonical key computation.
        p_un = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_un_cast AS "
            "SELECT d AS dt, ai2_2d AS arr, ch4 AS s FROM a "
            "UNION "
            "SELECT ts4, '{1,2}', vc20 FROM b;",
            dialect="postgres:18",
        )
        self.assertRegex(p_un.backing_table_ddl, r"\bdt TIMESTAMP\b")
        self.assertRegex(p_un.backing_table_ddl, r"\barr SMALLINT\[\]")
        self.assertRegex(p_un.backing_table_ddl, r"\bs BPCHAR\b")
        trig_sql_un = "\n".join(p_un.trigger_ddls)
        self.assertIn("CAST((__mv_d.d) AS TIMESTAMP)", trig_sql_un)
        self.assertIn("CAST(('{1,2}') AS SMALLINT[])", trig_sql_un)
        self.assertIn("CAST((__mv_d.vc20) AS BPCHAR)", trig_sql_un)

        # 3. CTE / derived-subquery UNION widening must not be short-circuited by branch-0 aliases,
        #    and UNION ALL GROUP BY must cast widened branch columns.
        p_cte = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_cte_arr AS "
            "WITH u AS (SELECT id, ai2_2d AS arr, b4 AS bt FROM a UNION ALL SELECT id, ai8 AS arr, b8 AS bt FROM b) "
            "SELECT id, arr, bt FROM u ORDER BY id;",
            dialect="postgres:18",
        )
        self.assertRegex(p_cte.backing_table_ddl, r"\barr BIGINT\[\]")
        self.assertRegex(p_cte.backing_table_ddl, r'\bbt pg_catalog\."bit"(?=[,\s\)])')

        p_sub = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_sub_arr AS "
            "SELECT id, arr FROM (SELECT id, ai2_2d AS arr FROM a UNION ALL SELECT id, ai8 AS arr FROM b) AS sub ORDER BY id;",
            dialect="postgres:18",
        )
        self.assertRegex(p_sub.backing_table_ddl, r"\barr BIGINT\[\]")

        p_uagg_arr = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_uagg_arr AS "
            "WITH u AS (SELECT ai2_2d AS g, ai2 AS a, x FROM a UNION ALL SELECT ai8 AS g, ai8 AS a, y FROM b) "
            "SELECT g, MIN(a) AS mna, SUM(x) AS sx FROM u GROUP BY g;",
            dialect="postgres:18",
        )
        self.assertRegex(p_uagg_arr.backing_table_ddl, r"\bg BIGINT\[\]")
        self.assertRegex(p_uagg_arr.backing_table_ddl, r"\bmna BIGINT\[\]")
        self.assertRegex(p_uagg_arr.backing_table_ddl, r"\bsx NUMERIC\b")
        self.assertIn("CAST((__mv_d.ai2_2d) AS BIGINT[])", "\n".join(p_uagg_arr.trigger_ddls))
        self.assertIn("CAST((__mv_d.ai2) AS BIGINT[])", "\n".join(p_uagg_arr.trigger_ddls))

        # 4. Tier 3 SELECT * UNION widens positionally across branches.
        star_schema = (
            "CREATE TABLE s1 (id INT PRIMARY KEY, arr SMALLINT[][], dt DATE);"
            "CREATE TABLE s2 (id INT PRIMARY KEY, arr BIGINT[], dt TIMESTAMPTZ);"
        )
        p_star = generate_materialized_view(
            star_schema,
            "CREATE VIEW v_star_u AS SELECT * FROM s1 UNION SELECT * FROM s2;",
            dialect="postgres:18",
        )
        self.assertRegex(p_star.backing_table_ddl, r"\barr BIGINT\[\]")
        self.assertRegex(p_star.backing_table_ddl, r"\bdt TIMESTAMPTZ\b")

        # 5. ARRAY[...] element widening across strings, NAME, temporal, network, and nested arrays.
        p_arr_expr = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_arr_expr AS SELECT id, "
            "  ARRAY[vc5, ch4] AS a_vc_ch, "
            "  ARRAY[ch4, vc5] AS a_ch_vc, "
            "  ARRAY[vc5, nm] AS a_vc_nm, "
            "  ARRAY[txt, nm] AS a_txt_nm, "
            "  ARRAY[d, ts2] AS a_d_ts, "
            "  ARRAY[cidr_c, inet_c] AS a_net "
            "FROM a JOIN b USING (id);",
            dialect="postgres:18",
        )
        self.assertRegex(p_arr_expr.backing_table_ddl, r"\ba_vc_ch VARCHAR\[\]")
        self.assertRegex(p_arr_expr.backing_table_ddl, r"\ba_ch_vc BPCHAR\[\]")
        self.assertRegex(p_arr_expr.backing_table_ddl, r"\ba_vc_nm NAME\[\]")
        self.assertRegex(p_arr_expr.backing_table_ddl, r'\ba_txt_nm TEXT\[\] COLLATE "C"')
        self.assertRegex(p_arr_expr.backing_table_ddl, r"\ba_d_ts TIMESTAMP\[\]")
        self.assertRegex(p_arr_expr.backing_table_ddl, r"\ba_net INET\[\]")

        # 6. Adversarial edge cases:
        #    (a) Column names colliding with type / collation tokens in UNION ALL GROUP BY triggers.
        coll_schema = (
            'CREATE TABLE a (id INT PRIMARY KEY, c TEXT COLLATE "C", name NAME, dt DATE, x INT);'
            "CREATE TABLE b (id INT PRIMARY KEY, c TEXT, name VARCHAR(20), dt TIMESTAMP, y BIGINT);"
        )
        p_col_hit = generate_materialized_view(
            coll_schema,
            "CREATE VIEW v_col_hit AS "
            "WITH u AS (SELECT c AS g, name AS nm, dt, x FROM a UNION ALL SELECT c, name, dt, y FROM b) "
            "SELECT g, nm, dt, SUM(x) AS sx FROM u GROUP BY g, nm, dt;",
            dialect="postgres:18",
        )
        trig_hit = "\n".join(p_col_hit.trigger_ddls)
        self.assertNotIn("COLLATE __mv_d.c", trig_hit)
        self.assertNotIn("AS __mv_d.name", trig_hit)
        self.assertIn('COLLATE "C"', trig_hit)
        self.assertIn("AS NAME", trig_hit)

        #    (b) Recursive CTEs must not recurse infinitely across PostgreSQL, MySQL, and MariaDB.
        rec_schema = "CREATE TABLE a (id INT PRIMARY KEY, x SMALLINT);"
        rec_view = (
            "CREATE VIEW v_rec AS "
            "WITH RECURSIVE r(id, depth) AS ("
            "  SELECT id, 0 FROM a UNION ALL SELECT id, depth + 1 FROM r WHERE depth < 5"
            ") SELECT id, depth FROM r;"
        )
        for d_name in ("postgres:18", "mysql:8.4", "mariadb:11.4"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_rec = generate_materialized_view(
                    rec_schema, rec_view, dialect=d_name, ignore_limitations=True
                )
            self.assertIn("depth", p_rec.backing_table_ddl)

        #    (c) Multi-CTE qualifier disambiguation, nested set-op CTE widening, and outer alias shadowing.
        p_multi_cte = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_mcte AS "
            "WITH u1 AS (SELECT id, ai2 AS v FROM a UNION ALL SELECT id, ai8 AS v FROM b), "
            "     u2 AS (SELECT id, txt AS v FROM b UNION ALL SELECT id, txt AS v FROM b) "
            "SELECT u1.v AS v1, u2.v AS v2 FROM u1 JOIN u2 ON u1.id = u2.id;",
            dialect="postgres:18",
        )
        self.assertRegex(p_multi_cte.backing_table_ddl, r"\bv1 BIGINT\[\]")
        self.assertRegex(p_multi_cte.backing_table_ddl, r"\bv2 TEXT\b")

        p_nested_cte = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_ncte AS "
            "WITH u1 AS (SELECT id, ai2 AS v FROM a UNION ALL SELECT id, ai8 AS v FROM b), "
            "     u2 AS (SELECT id, v FROM u1 UNION ALL SELECT id, anum AS v FROM b) "
            "SELECT id, v FROM u2;",
            dialect="postgres:18",
        )
        self.assertRegex(p_nested_cte.backing_table_ddl, r"\bv NUMERIC\[\]")

        p_shadow = generate_materialized_view(
            pg_schema,
            "CREATE VIEW v_shadow AS "
            "WITH u AS (SELECT id, txt FROM a UNION ALL SELECT id, txt FROM b) "
            "SELECT id, txt FROM (SELECT id, COUNT(*) AS txt FROM u GROUP BY id) AS s;",
            dialect="postgres:18",
        )
        self.assertRegex(p_shadow.backing_table_ddl, r"\btxt BIGINT\b")

        #    (d) Asymmetric Tier 3 SELECT col UNION SELECT * widens symmetrically.
        p_star_asym = generate_materialized_view(
            star_schema,
            "CREATE VIEW v_star_asym AS SELECT id, arr, dt FROM s1 UNION SELECT * FROM s2;",
            dialect="postgres:18",
        )
        self.assertRegex(p_star_asym.backing_table_ddl, r"\barr BIGINT\[\]")
        self.assertRegex(p_star_asym.backing_table_ddl, r"\bdt TIMESTAMPTZ\b")

        #    (e) Explicit COLLATE overrides implicit collation; collatable custom domains keep COLLATE;
        #        BIT + BIT(1) keeps BIT(1); TIMESTAMP(2) + NULL drops typmod.
        extra_schema = (
            'CREATE TABLE e1 (id INT PRIMARY KEY, t_c TEXT COLLATE "C", nm NAME, b BIT, d_c my_dom COLLATE "C", ts2 TIMESTAMP(2) WITHOUT TIME ZONE);'
            "CREATE TABLE e2 (id INT PRIMARY KEY, txt TEXT, b1 BIT(1), d_c my_dom COLLATE \"C\");"
        )
        p_extra = generate_materialized_view(
            extra_schema,
            'CREATE VIEW v_extra AS '
            'SELECT id, t_c AS c_exp, nm AS nm_exp, b AS b_one, d_c, ts2 AS ts_null FROM e1 '
            'UNION ALL '
            'SELECT id, txt COLLATE "POSIX", txt COLLATE "POSIX", b1, d_c, NULL FROM e2;',
            dialect="postgres:18",
        )
        self.assertRegex(p_extra.backing_table_ddl, r'\bc_exp TEXT COLLATE "POSIX"')
        self.assertRegex(p_extra.backing_table_ddl, r'\bnm_exp NAME COLLATE "POSIX"')
        self.assertRegex(p_extra.backing_table_ddl, r"\bb_one BIT(?:\(1\))?(?=[,\s\)])")
        self.assertRegex(p_extra.backing_table_ddl, r'\bd_c my_dom COLLATE "C"')
        self.assertRegex(p_extra.backing_table_ddl, r"\bts_null TIMESTAMP(?=[,\s\)])")

        #    (f) Incompatible PostgreSQL UNION / UNION ALL mixes fail closed offline.
        for bad_sql in (
            "CREATE VIEW v_bad1 AS SELECT txt AS v FROM a UNION ALL SELECT ai4 FROM a;",
            "CREATE VIEW v_bad2 AS SELECT ai4 AS v FROM a UNION ALL SELECT x FROM a;",
            "CREATE VIEW v_bad3 AS SELECT txt AS v FROM a UNION ALL SELECT x FROM a;",
            "CREATE VIEW v_bad4 AS SELECT ai4 AS v FROM a UNION ALL SELECT atxt FROM a;",
        ):
            with self.assertRaises(UnsupportedMaterializedViewError):
                generate_materialized_view(pg_schema, bad_sql, dialect="postgres:18")

    def test_postgres_set_operation_quantifiers_nested_collation_and_arm_chain_offline(self) -> None:
        schema = (
            'CREATE TABLE a (id INT PRIMARY KEY, s TEXT, i INT, r REAL, d DOUBLE PRECISION, n BIGINT);'
            'CREATE TABLE b (id DOUBLE PRECISION PRIMARY KEY, s TEXT COLLATE "C", n DOUBLE PRECISION);'
        )
        # An ALL / DISTINCT quantifier must not hide the second arm's types from Tier 3 widening.
        for op in ("EXCEPT ALL", "INTERSECT ALL", "EXCEPT DISTINCT", "UNION DISTINCT"):
            with self.subTest(op=op):
                plan = generate_materialized_view(
                    schema,
                    f"CREATE VIEW v_q AS SELECT id, n FROM a {op} SELECT id, n FROM b;",
                    dialect="postgres:18",
                )
                self.assertRegex(plan.backing_table_ddl, r"\bid DOUBLE PRECISION\b")
                self.assertRegex(plan.backing_table_ddl, r"\bn DOUBLE PRECISION\b")
        plan_star = generate_materialized_view(
            "CREATE TABLE a (id BIGINT PRIMARY KEY); CREATE TABLE b (id DOUBLE PRECISION PRIMARY KEY);",
            "CREATE VIEW v_qs AS SELECT * FROM a EXCEPT ALL SELECT * FROM b;",
            dialect="postgres:18",
        )
        self.assertRegex(plan_star.backing_table_ddl, r"\bid DOUBLE PRECISION\b")

        # An explicit COLLATE below a string function wins over the other arm's implicit collation.
        plan_c = generate_materialized_view(
            schema,
            'CREATE VIEW v_c AS SELECT lower(s COLLATE "POSIX") AS v FROM a UNION ALL SELECT s FROM b;',
            dialect="postgres:18",
        )
        self.assertRegex(plan_c.backing_table_ddl, r'\bv TEXT COLLATE "POSIX"')

        # PostgreSQL converts arms pairwise (INTEGER -> REAL -> DOUBLE PRECISION): refused, not modelled.
        for op in ("UNION", "UNION ALL"):
            with self.subTest(chain=op), self.assertRaises(UnsupportedMaterializedViewError):
                generate_materialized_view(
                    schema,
                    f"CREATE VIEW v_ch AS SELECT i AS v FROM a {op} SELECT r FROM a {op} SELECT d FROM a;",
                    dialect="postgres:18",
                )

    def test_native_postgres_cross_branch_union_array_and_scalar_widening(self) -> None:
        import os

        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        tag = f"uwide_{uuid.uuid4().hex[:8]}"
        ta, tb, ts1, ts2, tcol = f"ua_{tag}", f"ub_{tag}", f"us1_{tag}", f"us2_{tag}", f"uc_{tag}"
        tcol2 = f"uc2_{tag}"
        conn = _connect(Dialect.POSTGRES, dsn)
        cur = conn.cursor()
        mv_names: List[str] = []
        lv_names: List[str] = []
        try:
            cur.execute("SELECT VERSION()")
            ver_row = cur.fetchone()
            self.assertIsNotNone(ver_row)
            assert ver_row is not None
            self.assertIn("PostgreSQL", str(ver_row[0]))

            cur.execute(
                f"CREATE TABLE {ta} ("
                "  id INT PRIMARY KEY, "
                "  x SMALLINT, "
                "  ai2 SMALLINT[], "
                "  ai2_2d SMALLINT[][], "
                "  ai4 INT[], "
                "  avc5 VARCHAR(5)[], "
                "  ach4 CHAR(4)[], "
                "  atxt TEXT[], "
                '  at_c TEXT[] COLLATE "C", '
                "  ab4 BIT(4)[], "
                "  avb VARBIT(8)[], "
                "  ad DATE[], "
                "  ats2 TIMESTAMP(2) WITHOUT TIME ZONE[], "
                "  atm2 TIME(2) WITHOUT TIME ZONE[], "
                "  aiv2 INTERVAL(2)[], "
                "  acidr CIDR[], "
                "  vc5 VARCHAR(5), "
                "  ch4 CHAR(4), "
                "  txt TEXT, "
                '  t_c TEXT COLLATE "C", '
                "  b4 BIT(4), "
                "  vb VARBIT(8), "
                "  d DATE, "
                "  ts2 TIMESTAMP(2) WITHOUT TIME ZONE, "
                "  tm2 TIME(2) WITHOUT TIME ZONE, "
                "  iv2 INTERVAL(2), "
                "  num NUMERIC(6, 2), "
                "  cidr_c CIDR"
                ")"
            )
            cur.execute(
                f"CREATE TABLE {tb} ("
                "  id INT PRIMARY KEY, "
                "  y BIGINT, "
                "  ai8 BIGINT[], "
                "  anum NUMERIC(10, 2)[], "
                "  af4 REAL[], "
                "  avc20 VARCHAR(20)[], "
                "  atxt TEXT[], "
                "  anm NAME[], "
                "  ab8 BIT(8)[], "
                "  avb VARBIT[], "
                "  atstz4 TIMESTAMP(4) WITH TIME ZONE[], "
                "  atmtz4 TIME(4) WITH TIME ZONE[], "
                "  aiv4 INTERVAL(4)[], "
                "  ainet INET[], "
                "  vc20 VARCHAR(20), "
                "  txt TEXT, "
                "  nm NAME, "
                "  b8 BIT(8), "
                "  vb VARBIT, "
                "  ts4 TIMESTAMP(4) WITHOUT TIME ZONE, "
                "  tstz4 TIMESTAMP(4) WITH TIME ZONE, "
                "  tmtz4 TIME(4) WITH TIME ZONE, "
                "  iv4 INTERVAL(4), "
                "  inet_c INET"
                ")"
            )
            cur.execute(f"CREATE TABLE {ts1} (id INT PRIMARY KEY, arr SMALLINT[][], dt DATE)")
            cur.execute(f"CREATE TABLE {ts2} (id INT PRIMARY KEY, arr BIGINT[], dt TIMESTAMPTZ)")
            cur.execute(f'CREATE TABLE {tcol} (id INT PRIMARY KEY, c TEXT COLLATE "C", name NAME, x INT)')
            cur.execute(f"CREATE TABLE {tcol2} (id INT PRIMARY KEY, c TEXT, name VARCHAR(20), y BIGINT)")

            cur.execute(
                f"INSERT INTO {ta} VALUES ("
                "  1, 10, ARRAY[1, 2]::smallint[], ARRAY[ARRAY[1, 2]]::smallint[][], ARRAY[1, 2], "
                "  ARRAY['hi']::varchar(5)[], ARRAY['ab']::char(4)[], ARRAY['t1'], ARRAY['c1'], "
                "  ARRAY[B'1010'], ARRAY[B'101'], ARRAY[DATE '2026-01-01'], "
                "  ARRAY[TIMESTAMP '2026-01-01 00:00:00.12'], ARRAY[TIME '12:30:00.12'], "
                "  ARRAY[INTERVAL '1 hour'], "
                "  ARRAY['10.0.0.0/8'::cidr], 'ab', 'ab', 't1', 'c1', "
                "  B'1010', B'101', DATE '2026-01-01', TIMESTAMP '2026-01-01 00:00:00', "
                "  TIME '12:30:00.12', INTERVAL '1 hour', 12.34, '10.0.0.0/8'::cidr"
                "), ("
                "  2, 20, ARRAY[3, 4]::smallint[], ARRAY[ARRAY[3, 4]]::smallint[][], ARRAY[5, 6], "
                "  NULL, ARRAY['cd']::char(4)[], ARRAY['t2'], ARRAY['c2'], "
                "  ARRAY[B'0101'], ARRAY[B'11'], ARRAY[DATE '2026-02-02'], "
                "  ARRAY[TIMESTAMP '2026-02-02 01:02:03.45'], ARRAY[TIME '08:00:00'], "
                "  ARRAY[INTERVAL '2 hours'], "
                "  ARRAY['172.16.0.0/12'::cidr], 'xy', 'xy', 't2', 'c2', "
                "  B'0101', B'11', DATE '2026-02-02', TIMESTAMP '2026-02-02 00:00:00', "
                "  TIME '08:00:00', INTERVAL '2 hours', 56.78, '172.16.0.0/12'::cidr"
                ")"
            )
            cur.execute(
                f"INSERT INTO {tb} VALUES ("
                "  1, 5000000000, ARRAY[1, 2]::bigint[], ARRAY[1.00, 2.00]::numeric(10,2)[], ARRAY[1.5]::real[], "
                "  ARRAY['ab  ']::varchar(20)[], ARRAY['t1'], ARRAY['n1']::name[], "
                "  ARRAY[B'10101010'], ARRAY[B'10101010'], "
                "  ARRAY[TIMESTAMPTZ '2026-01-01 00:00:00+00'], ARRAY[TIMETZ '12:30:00+00'], "
                "  ARRAY[INTERVAL '30 minutes'], "
                "  ARRAY['10.0.0.0/8'::inet], 'ab  ', 't1', 'n1'::name, "
                "  B'10101010', B'10101010', TIMESTAMP '2026-01-01 00:00:00', "
                "  TIMESTAMPTZ '2026-01-01 00:00:00+00', TIMETZ '12:30:00+00', INTERVAL '30 minutes', '10.0.0.0/8'::inet"
                "), ("
                "  2, 6000000000, ARRAY[9000000000, 10]::bigint[], ARRAY[7.25, 8.50]::numeric(10,2)[], ARRAY[2.5]::real[], "
                "  ARRAY['longervalue']::varchar(20)[], ARRAY['t3'], ARRAY['n2']::name[], "
                "  ARRAY[B'11110000'], ARRAY[B'11110000'], "
                "  ARRAY[TIMESTAMPTZ '2026-03-03 00:00:00+00'], ARRAY[TIMETZ '09:15:00+00'], "
                "  ARRAY[INTERVAL '45 minutes'], "
                "  ARRAY['192.168.1.5/24'::inet], 'zz', 't3', 'n2'::name, "
                "  B'11110000', B'11110000', TIMESTAMP '2026-03-03 00:00:00', "
                "  TIMESTAMPTZ '2026-03-03 00:00:00+00', TIMETZ '09:15:00+00', INTERVAL '45 minutes', '192.168.1.5/24'::inet"
                ")"
            )
            cur.execute(
                f"INSERT INTO {ts1} VALUES (1, ARRAY[ARRAY[1, 2]]::smallint[][], DATE '2026-01-01')"
            )
            cur.execute(
                f"INSERT INTO {ts2} VALUES (2, ARRAY[5000000000]::bigint[], TIMESTAMPTZ '2026-04-04 00:00:00+00')"
            )
            cur.execute(f"INSERT INTO {tcol} VALUES (1, 'k1', 'n1'::name, 10)")
            cur.execute(f"INSERT INTO {tcol2} VALUES (1, 'k1', 'n1', 5000000000)")
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
            specs = [
                (
                    "uall_all",
                    f"SELECT id, ai2_2d AS arr_dim, ai4 AS arr_num, ai4 AS arr_f4, ai4 AS arr_lit, "
                    f"  avc5 AS avc_lit, avc5 AS avc_null, atxt AS atxt_nm, avc5 AS avc_nm, "
                    f"  at_c AS at_coll, t_c AS t_coll, t_c AS t_exp_coll, "
                    f"  b4 AS b_b, ab4 AS ab_vb, avb AS avb_b, "
                    f"  ad AS ad_ts, ats2 AS ats_tz, atm2 AS atm_tz, ts2 AS ts_null, "
                    f"  iv2 AS iv_mix, aiv2 AS aiv_mix, "
                    f"  acidr AS anet, cidr_c AS net, vc5 AS vc_lit, num AS num_lit "
                    f"FROM {ta} "
                    f"UNION ALL "
                    f"SELECT id, ai8, anum, af4, '{{1,2}}', "
                    f"  '{{toolongfor5}}', NULL, anm, anm, "
                    f"  atxt, txt, txt COLLATE \"POSIX\", "
                    f"  b8, avb, ab8, "
                    f"  ARRAY[ts4], atstz4, atmtz4, NULL, "
                    f"  iv4, aiv4, "
                    f"  ainet, inet_c, 'longstring', '12345678.901' "
                    f"FROM {tb}",
                ),
                (
                    "un_arr_dim",
                    f"SELECT ai2_2d AS arr FROM {ta} UNION SELECT ai8 FROM {tb}",
                ),
                (
                    "un_arr_num",
                    f"SELECT ai4 AS arr FROM {ta} UNION SELECT anum FROM {tb}",
                ),
                (
                    "un_arr_lit",
                    f"SELECT ai4 AS arr FROM {ta} UNION SELECT '{{1,2}}' FROM {tb}",
                ),
                (
                    "un_date_ts",
                    f"SELECT d AS dt FROM {ta} UNION SELECT ts4 FROM {tb}",
                ),
                (
                    "un_ch_vc",
                    f"SELECT ch4 AS s FROM {ta} UNION SELECT vc20 FROM {tb}",
                ),
                (
                    "un_vc_ch",
                    f"SELECT vc20 AS s FROM {tb} UNION SELECT ch4 FROM {ta}",
                ),
                (
                    "un_net",
                    f"SELECT cidr_c AS net FROM {ta} UNION SELECT inet_c FROM {tb}",
                ),
                (
                    "un_coll",
                    f"SELECT t_c AS s FROM {ta} UNION SELECT txt FROM {tb}",
                ),
                (
                    "inter_mix",
                    f"SELECT d AS dt, ai2 AS arr FROM {ta} INTERSECT SELECT ts4, ai8 FROM {tb}",
                ),
                (
                    "inter_num_bp_arr",
                    f"SELECT ai4 AS arr, ach4 AS s FROM {ta} INTERSECT SELECT anum, avc20 FROM {tb}",
                ),
                (
                    "except_num_bp",
                    f"SELECT ai4 AS arr, ch4 AS s FROM {ta} EXCEPT SELECT anum, vc20 FROM {tb}",
                ),
                (
                    "except_lit",
                    f"SELECT ai4 AS arr FROM {ta} EXCEPT SELECT '{{1,2}}' FROM {tb}",
                ),
                (
                    "uagg_arr",
                    f"WITH u AS ("
                    f"  SELECT ai2 AS g, ai2 AS a, x FROM {ta} "
                    f"  UNION ALL "
                    f"  SELECT ai8 AS g, ai8 AS a, y FROM {tb}"
                    f") SELECT g, COUNT(*) AS cnt, SUM(x) AS sx, MIN(a) AS mna, MAX(a) AS mxa FROM u GROUP BY g",
                ),
                (
                    "uagg_dt_null",
                    f"WITH u AS ("
                    f"  SELECT d AS g, x FROM {ta} "
                    f"  UNION ALL "
                    f"  SELECT ts4 AS g, NULL AS x FROM {tb}"
                    f") SELECT g, COUNT(*) AS cnt, SUM(x) AS sx FROM u GROUP BY g",
                ),
                (
                    "uagg_col_hit",
                    f"WITH u AS ("
                    f"  SELECT c AS g, name AS nm, x FROM {tcol} "
                    f"  UNION ALL "
                    f"  SELECT c, name, y FROM {tcol2}"
                    f") SELECT g, nm, SUM(x) AS sx FROM u GROUP BY g, nm",
                ),
                (
                    "rec_cte",
                    f"WITH RECURSIVE r(id, depth) AS ("
                    f"  SELECT id, 0 FROM {tcol} "
                    f"  UNION ALL "
                    f"  SELECT id, depth + 1 FROM r WHERE depth < 2"
                    f") SELECT id, depth FROM r",
                ),
                (
                    "cte_sub",
                    f"WITH u AS ("
                    f"  SELECT id, ai2_2d AS arr, b4 AS bt FROM {ta} "
                    f"  UNION ALL "
                    f"  SELECT id, ai8 AS arr, b8 AS bt FROM {tb}"
                    f") SELECT id, arr, bt FROM u ORDER BY id, arr",
                ),
                (
                    "derived_sub",
                    f"SELECT id, arr FROM ("
                    f"  SELECT id, ai2_2d AS arr FROM {ta} "
                    f"  UNION ALL "
                    f"  SELECT id, ai8 AS arr FROM {tb}"
                    f") AS sub ORDER BY id, arr",
                ),
                (
                    "star_u",
                    f"SELECT * FROM {ts1} UNION SELECT * FROM {ts2}",
                ),
                (
                    "star_asym",
                    f"SELECT id, arr, dt FROM {ts1} UNION SELECT * FROM {ts2}",
                ),
                # ALL / DISTINCT quantifiers on Tier 3 `SELECT *` set operations must not hide
                # the second arm's column types from the widening.
                ("star_except_all", f"SELECT * FROM {ts1} EXCEPT ALL SELECT * FROM {ts2}"),
                ("star_intersect_all", f"SELECT * FROM {ts1} INTERSECT ALL SELECT * FROM {ts2}"),
                ("star_union_distinct", f"SELECT * FROM {ts1} UNION DISTINCT SELECT * FROM {ts2}"),
                # An explicit COLLATE below a string function still wins over the other arm's
                # implicit non-default collation.
                ("lower_explicit_collate", f'SELECT lower(c COLLATE "POSIX") AS v FROM {tcol2} UNION ALL SELECT c FROM {tcol}'),
            ]
            pairs = []
            for label, sel in specs:
                mv, lv = f"mv_{label}_{tag}", f"lv_{label}_{tag}"
                mgr.register_view(f"CREATE VIEW {mv} AS {sel};")
                conn.commit()
                mv_names.append(mv)
                cur.execute(f"CREATE VIEW {lv} AS {sel}")
                conn.commit()
                lv_names.append(lv)
                pairs.append((label, mv, lv))

            def _pg_view_attr_sig(view_name: str) -> List[Tuple[str, str, int, int, str]]:
                cur.execute(
                    "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), "
                    "       a.atttypid::int, a.atttypmod::int, COALESCE(co.collname, '-') "
                    "FROM pg_catalog.pg_attribute a "
                    "JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
                    "LEFT JOIN pg_catalog.pg_collation co ON co.oid = a.attcollation "
                    "WHERE c.relname = %s AND pg_catalog.pg_table_is_visible(c.oid) "
                    "  AND a.attnum > 0 AND NOT a.attisdropped "
                    "ORDER BY a.attnum",
                    (view_name,),
                )
                return [(str(r[0]), str(r[1]), int(r[2]), int(r[3]), str(r[4])) for r in cur.fetchall()]

            def check(step: str) -> None:
                for label, mv, lv in pairs:
                    cur.execute(f"SELECT * FROM {mv}")
                    got = sorted((tuple(_exact_deep(v) for v in r) for r in cur.fetchall()), key=repr)
                    da = _description(cur)
                    cur.execute(f"SELECT * FROM {lv}")
                    exp = sorted((tuple(_exact_deep(v) for v in r) for r in cur.fetchall()), key=repr)
                    db = _description(cur)
                    mv_sig = _pg_view_attr_sig(mv)
                    lv_sig = _pg_view_attr_sig(lv)
                    conn.commit()
                    self.assertEqual(got, exp, f"PG {label} {step}: {got} != {exp}")
                    self.assertEqual(_metadata_mismatches(Dialect.POSTGRES, da, db), [], f"PG {label} {step} description")
                    self.assertEqual(mv_sig, lv_sig, f"PG {label} {step} pg_attribute type/mod/collation")

            check("initial")

            # Mutate across INSERT, UPDATE, and DELETE on all tables (including overlapping
            # widened keys, array scale changes, trailing spaces, CIDR/INET, INTERVAL, and NULLs).
            cur.execute(
                f"INSERT INTO {ta} VALUES ("
                "  3, 30, ARRAY[9000, 10]::smallint[], ARRAY[ARRAY[9, 10]]::smallint[][], ARRAY[7, 8], "
                "  ARRAY['ok']::varchar(5)[], ARRAY['zz']::char(4)[], ARRAY['t3'], ARRAY['c3'], "
                "  ARRAY[B'1100'], ARRAY[B'1'], ARRAY[DATE '2026-03-03'], "
                "  ARRAY[TIMESTAMP '2026-03-03 00:00:00'], ARRAY[TIME '09:15:00'], "
                "  ARRAY[INTERVAL '15 minutes'], "
                "  ARRAY['10.0.0.0/8'::cidr], 'zz', 'zz', 't3', 'c3', "
                "  B'1100', B'1', DATE '2026-03-03', TIMESTAMP '2026-03-03 00:00:00', "
                "  TIME '09:15:00', INTERVAL '15 minutes', 99.99, '10.0.0.0/8'::cidr"
                ")"
            )
            cur.execute(
                f"UPDATE {tb} SET ai8 = ARRAY[3, 4]::bigint[], anum = ARRAY[5.00, 6.00]::numeric(10,2)[], "
                "  avc20 = ARRAY['cd  ']::varchar(20)[], "
                "  ts4 = TIMESTAMP '2026-02-02 00:00:00', vc20 = 'cd  ', inet_c = '172.16.0.0/12'::inet "
                "WHERE id = 2"
            )
            cur.execute(f"DELETE FROM {ta} WHERE id = 1")
            cur.execute(f"DELETE FROM {tb} WHERE id = 1")
            cur.execute(f"INSERT INTO {ts1} VALUES (3, ARRAY[ARRAY[5, 6]]::smallint[][], DATE '2026-05-05')")
            cur.execute(f"DELETE FROM {ts2} WHERE id = 2")
            cur.execute(f"INSERT INTO {tcol} VALUES (2, 'k2', 'n2'::name, 20)")
            cur.execute(f"UPDATE {tcol2} SET c = 'k2', name = 'n2', y = 7000000000 WHERE id = 1")
            conn.commit()
            check("after_dml")

            # Negative fail-closed checks: mixing typed array with typed scalar or incompatible
            # scalar categories in UNION / UNION ALL must fail cleanly and roll back.
            for idx, bad_sel in enumerate((
                f"SELECT ai4 AS v FROM {ta} UNION SELECT x FROM {ta}",
                f"SELECT txt AS v FROM {ta} UNION ALL SELECT ai4 FROM {ta}",
                f"SELECT txt AS v FROM {ta} UNION ALL SELECT x FROM {ta}",
                # PostgreSQL pairs arms left to right (int -> REAL -> DOUBLE PRECISION): refused, not
                # silently mis-converted (16777217 would survive instead of becoming 16777216).
                f"SELECT x::int AS v FROM {ta} UNION SELECT x::real FROM {ta} UNION SELECT x::float8 FROM {ta}",
                f"SELECT x::int AS v FROM {ta} UNION ALL SELECT x::real FROM {ta} UNION ALL SELECT x::float8 FROM {ta}",
            )):
                bad_mv = f"mv_bad_u_{idx}_{tag}"
                with self.assertRaises(UnsupportedMaterializedViewError, msg=bad_sel):
                    mgr.register_view(f"CREATE VIEW {bad_mv} AS {bad_sel};")
                conn.rollback()
                cur.execute(
                    "SELECT 1 FROM pg_catalog.pg_class WHERE relname = %s AND pg_catalog.pg_table_is_visible(oid)",
                    (bad_mv,),
                )
                self.assertIsNone(cur.fetchone(), f"Failed registration {bad_mv} must roll back cleanly")
                conn.rollback()  # end the catalog read's transaction before the next registration
        finally:
            conn.rollback()
            for mv in mv_names:
                try:
                    MaterializedViewManager(conn, dialect=Dialect.POSTGRES).drop_view(mv)
                except Exception:
                    conn.rollback()
            for lv in lv_names:
                cur.execute(f"DROP VIEW IF EXISTS {lv} CASCADE")
            for tbl in (ta, tb, ts1, ts2, tcol, tcol2):
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            conn.commit()
            conn.close()

    _TREE_SCHEMA = (
        "CREATE TABLE a (id INT PRIMARY KEY, s TEXT, i4 INT, i2 SMALLINT, r REAL, d DOUBLE PRECISION, n BIGINT, "
        "nu NUMERIC(6,2), ci CITEXT, arrt TEXT[], e TEXT, c TEXT COLLATE \"C\", vc VARCHAR(5), ach CHAR(4)[], aci CITEXT[]);"
        "CREATE TABLE b (id INT PRIMARY KEY, z TEXT COLLATE \"POSIX\", e TEXT);"
    )
    # (view body, accepted?) -- accepted iff no arm reaches the final type through a lossy REAL pair.
    _TREE_VIEWS = [
        # right-nested / parenthesised forms: the inner set operation is itself an operand, so its
        # (DOUBLE PRECISION) result type is what the left arm joins; only a REAL pair that
        # contains the exact arm is lossy.
        ("SELECT i4 AS v FROM a UNION ALL (SELECT r FROM a UNION ALL SELECT d FROM a)", True),
        ("SELECT i4 AS v FROM a UNION (SELECT r FROM a UNION ALL SELECT d FROM a)", True),
        ("SELECT i4 AS v FROM a INTERSECT (SELECT r FROM a UNION ALL SELECT d FROM a)", True),
        ("SELECT n AS v FROM a UNION ALL (SELECT r FROM a UNION ALL SELECT d FROM a)", True),
        ("SELECT nu AS v FROM a UNION ALL (SELECT r FROM a UNION ALL SELECT d FROM a)", True),
        ("SELECT i2 AS v FROM a UNION ALL (SELECT r FROM a UNION ALL SELECT d FROM a)", True),
        ("SELECT i4 AS v FROM a UNION ALL (SELECT d FROM a UNION ALL SELECT r FROM a)", True),
        ("SELECT r AS v FROM a UNION ALL (SELECT i4 FROM a UNION ALL SELECT d FROM a)", True),
        ("SELECT d AS v FROM a UNION ALL (SELECT r FROM a UNION ALL SELECT i4 FROM a)", False),
        ("SELECT d AS v FROM a UNION ALL (SELECT i4 FROM a UNION ALL SELECT r FROM a)", False),
        ("SELECT d AS v FROM a UNION (SELECT r FROM a INTERSECT SELECT n FROM a)", False),
        # flat left-to-right chains
        ("SELECT i4 AS v FROM a UNION ALL SELECT r FROM a UNION ALL SELECT d FROM a", False),
        ("SELECT i4 AS v FROM a UNION SELECT r FROM a UNION SELECT d FROM a", False),
        ("SELECT d AS v FROM a UNION ALL SELECT i4 FROM a UNION ALL SELECT r FROM a", True),
        ("SELECT r AS v FROM a UNION ALL SELECT d FROM a UNION ALL SELECT i4 FROM a", True),
        ("SELECT i4 AS v FROM a UNION ALL SELECT d FROM a UNION ALL SELECT r FROM a UNION ALL SELECT i2 FROM a", True),
        ("SELECT i2 AS v FROM a UNION ALL SELECT r FROM a UNION ALL SELECT d FROM a", True),
        ("SELECT max(v) AS m FROM (SELECT d AS v FROM a UNION ALL SELECT i4 FROM a UNION ALL SELECT r FROM a) q", True),
        ("SELECT max(v) AS m FROM (SELECT i4 AS v FROM a UNION ALL SELECT r FROM a UNION ALL SELECT d FROM a) q", False),
        ("SELECT d AS v FROM a UNION ALL SELECT i4 FROM a UNION ALL SELECT r FROM a ORDER BY 1", True),
    ]

    # A CTE / derived-table OUTPUT column may carry the name of a base-table column of another type:
    # the arm is typed through the relation's own scope (DOUBLE PRECISION here), never from the table.
    _SHADOW_VIEWS = [
        "WITH u AS (SELECT d AS i2 FROM a) SELECT i4 AS w FROM a UNION ALL SELECT i2 FROM u",
        "WITH u AS (SELECT d AS i2 FROM a) SELECT i4 AS w FROM a UNION ALL SELECT u.i2 FROM u",
        "SELECT i4 AS w FROM a UNION ALL SELECT x.i2 FROM (SELECT d AS i2 FROM a) x",
        "SELECT i4 AS w FROM a UNION ALL SELECT i2 FROM (SELECT d AS i2 FROM a) x",
        "WITH u AS (SELECT d + 1 AS i2 FROM a) SELECT i4 AS w FROM a UNION ALL SELECT i2 FROM u",
        "WITH u AS (SELECT d AS i2, id AS uid FROM a) SELECT i4 AS w FROM a UNION ALL SELECT u.i2 FROM u JOIN a p ON p.id = u.uid",
        "WITH u AS (SELECT d AS i2 FROM a) SELECT i2 AS w FROM u UNION ALL SELECT i4 FROM a",
    ]

    def test_postgres_set_operation_pairwise_tree_type_resolution_offline(self) -> None:
        """PostgreSQL resolves set-operation column types pairwise over the parse tree; only a
        lossy REAL intermediate (INTEGER/BIGINT/NUMERIC -> REAL -> DOUBLE PRECISION) is refused."""
        for body, accepted in self._TREE_VIEWS:
            with self.subTest(view=body):
                if accepted:
                    plan = generate_materialized_view(
                        self._TREE_SCHEMA, f"CREATE VIEW mv AS {body};", dialect="postgres:18"
                    )
                    if "max(v)" not in body:
                        self.assertRegex(plan.backing_table_ddl, r"\bv DOUBLE PRECISION\b")
                else:
                    with self.assertRaises(UnsupportedMaterializedViewError):
                        generate_materialized_view(
                            self._TREE_SCHEMA, f"CREATE VIEW mv AS {body};", dialect="postgres:18"
                        )
        for body in self._SHADOW_VIEWS:
            with self.subTest(shadow=body):
                plan = generate_materialized_view(
                    self._TREE_SCHEMA, f"CREATE VIEW mv AS {body};", dialect="postgres:18"
                )
                self.assertRegex(plan.backing_table_ddl, r"\bw DOUBLE PRECISION\b")
        # The accepted nested form widens to the final type of the tree (DOUBLE PRECISION), not the
        # first nested SELECT's REAL.
        plan = generate_materialized_view(
            self._TREE_SCHEMA,
            "CREATE VIEW mv AS SELECT i2 AS v FROM a UNION ALL (SELECT r FROM a UNION ALL SELECT d FROM a);",
            dialect="postgres:18",
        )
        self.assertRegex(plan.backing_table_ddl, r"\bv DOUBLE PRECISION\b")

    def test_postgres_array_scalar_inference_collation_and_unmodelled_arms_offline(self) -> None:
        schema = self._TREE_SCHEMA
        # Array-valued functions and ARRAY[] over an extension-typed column are arrays, so they
        # unify with array arms (no false "cannot mix array and scalar" refusal).
        for sel in (
            "SELECT ARRAY[s] AS v FROM a INTERSECT ALL SELECT string_to_array(e, 'b') FROM a",
            "SELECT arrt AS v FROM a UNION ALL SELECT ARRAY[ci] FROM a",
            "SELECT ARRAY[ci] AS v FROM a UNION ALL SELECT arrt FROM a",
        ):
            with self.subTest(sel=sel):
                plan = generate_materialized_view(schema, f"CREATE VIEW mv AS {sel};", dialect="postgres:18")
                self.assertRegex(plan.backing_table_ddl, r"\bv TEXT\[\]")
        # A function sqlglot does not model (regexp_split_to_array) as a de-duplicating set
        # operation arm: its type is not derived, and the server is asked only for the view's
        # output columns, not for an arm's, so it is refused (exact routing), offline and
        # natively (TODO.md: ask the server for set-operation types).
        with self.assertRaisesRegex(UnsupportedMaterializedViewError, "without asking the server"):
            generate_materialized_view(
                schema,
                "CREATE VIEW mv AS SELECT arrt AS v FROM a UNION SELECT regexp_split_to_array(e, 'b') FROM a;",
                dialect="postgres:18",
            )
        # citext only casts implicitly to text: any other string arm resolves to text.
        plan = generate_materialized_view(
            schema, "CREATE VIEW mv AS SELECT ci AS v FROM a UNION ALL SELECT s FROM a;", dialect="postgres:18"
        )
        self.assertRegex(plan.backing_table_ddl, r"\bv TEXT\b")
        # A numeric literal with an exponent is NUMERIC in PostgreSQL.
        for lit in ("1.5e0", "1e2", "2.5E-3"):
            plan = generate_materialized_view(
                schema, f"CREATE VIEW mv AS SELECT {lit} AS v FROM a;", dialect="postgres:18"
            )
            self.assertRegex(plan.backing_table_ddl, r"\bv NUMERIC\b", lit)
            plan = generate_materialized_view(
                schema, f"CREATE VIEW mv AS WITH u AS (SELECT id, {lit} AS v FROM a) SELECT id, v FROM u;", dialect="postgres:18"
            )
            self.assertRegex(plan.backing_table_ddl, r"\bv NUMERIC\b", f"derived {lit}")
            plan = generate_materialized_view(
                schema,
                f"CREATE VIEW mv AS WITH u AS (SELECT {lit} AS v FROM a UNION ALL SELECT 2.5 FROM a) SELECT v FROM u;",
                dialect="postgres:18",
            )
            self.assertRegex(plan.backing_table_ddl, r"\bv NUMERIC\b", f"cte union {lit}")
        # Arms whose result types the compiler does not model at all are refused, not guessed as TEXT.
        for sel in (
            "SELECT to_jsonb(s) AS v FROM a UNION ALL SELECT to_jsonb(e) FROM b",
            "SELECT e::bytea || '\\x00'::bytea AS v FROM a UNION ALL SELECT e::bytea || '\\x01'::bytea FROM b",
        ):
            with self.subTest(sel=sel), self.assertRaises(UnsupportedMaterializedViewError):
                generate_materialized_view(schema, f"CREATE VIEW mv AS {sel};", dialect="postgres:18")
        # An explicit cast has exactly its target type, modelled or not.
        plan = generate_materialized_view(
            schema,
            "CREATE VIEW mv AS SELECT e::bytea AS v FROM a UNION ALL SELECT e::bytea FROM b;",
            dialect="postgres:18",
        )
        self.assertRegex(plan.backing_table_ddl, r"\bv BYTEA\b")
        # ...while known text-valued expressions keep working.
        for sel in (
            "SELECT s || 'x' AS v FROM a UNION ALL SELECT e || 'y' FROM b",
            "SELECT md5(s) AS v FROM a UNION ALL SELECT md5(e) FROM b",
        ):
            with self.subTest(sel=sel):
                generate_materialized_view(schema, f"CREATE VIEW mv AS {sel};", dialect="postgres:18")
        # An explicit COLLATE names a collation, never a column: the set-operation trigger bodies must
        # not qualify it as a record field (`__mv_d.c COLLATE __mv_d.c`).
        plan = generate_materialized_view(
            schema,
            'CREATE VIEW mv AS SELECT c COLLATE "C" AS v FROM a UNION SELECT z FROM b;',
            dialect="postgres:18",
        )
        trigger_sql = "\n".join(str(t) for t in plan.trigger_ddls)
        self.assertNotRegex(trigger_sql, r"COLLATE\s+__mv_d\.")
        self.assertRegex(trigger_sql, r'__mv_d\.c COLLATE "C"')
        # citext with char(n)/varchar (scalar or array) resolves to the other arm's type, not text: refused.
        for sel in (
            "SELECT ci AS v FROM a UNION ALL SELECT vc FROM a",
            "SELECT vc AS v FROM a UNION SELECT ci FROM a",
            "SELECT ach AS v FROM a UNION ALL (SELECT aci FROM a UNION ALL SELECT arrt FROM a)",
        ):
            with self.subTest(sel=sel), self.assertRaises(UnsupportedMaterializedViewError):
                generate_materialized_view(schema, f"CREATE VIEW mv AS {sel};", dialect="postgres:18")
        # An unmodelled arm named in a refusal is reported as assumed text, not as a bogus type.
        with self.assertRaisesRegex(UnsupportedMaterializedViewError, r"could not be inferred"):
            generate_materialized_view(
                "CREATE TABLE a (id INT PRIMARY KEY, iv INTERVAL, i4 INT);",
                "CREATE VIEW mv AS SELECT iv * 2 AS v FROM a UNION ALL SELECT i4 FROM a;",
                dialect="postgres:18",
            )
        # Clear refusals instead of raw parse/driver errors.
        for sel in (
            "(SELECT i4 AS v FROM a UNION ALL SELECT r FROM a) UNION ALL SELECT d FROM a",
            "SELECT * FROM (SELECT i4 AS x FROM a UNION SELECT i2 FROM a) q",
            "WITH u AS (SELECT i4 AS x FROM a UNION SELECT i2 FROM a) SELECT * FROM u",
            "SELECT p.id, x.* FROM a p LEFT JOIN (SELECT s AS v0 FROM a UNION SELECT e FROM b) x ON x.v0 = p.s",
        ):
            with self.subTest(sel=sel), self.assertRaises(UnsupportedMaterializedViewError):
                generate_materialized_view(schema, f"CREATE VIEW mv AS {sel};", dialect="postgres:18")

    def test_native_postgres_set_operation_tree_types_arrays_and_collate(self) -> None:
        """Native: every accepted view is compared with the live view (values bit-for-bit, pg_attribute
        type/typmod/collation) after population and DML; refused views leave nothing behind."""
        import os
        import re

        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        tag = f"utree_{uuid.uuid4().hex[:8]}"
        ta, tb = f"a_{tag}", f"b_{tag}"
        conn = _connect(Dialect.POSTGRES, dsn)
        cur = conn.cursor()
        mv_names: List[str] = []
        lv_names: List[str] = []
        created_citext = False
        try:
            cur.execute("SELECT VERSION()")
            ver_row = cur.fetchone()
            assert ver_row is not None
            self.assertIn("PostgreSQL", str(ver_row[0]))
            cur.execute("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'citext'")
            had_citext = cur.fetchone() is not None
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS citext")
                conn.commit()
                ci_type = "CITEXT"
                created_citext = not had_citext
            except Exception:
                conn.rollback()
                ci_type = "TEXT"
            cur.execute(
                f"CREATE TABLE {ta} (id INT PRIMARY KEY, s TEXT, i4 INT, i2 SMALLINT, r REAL, d DOUBLE PRECISION, "
                f'n BIGINT, nu NUMERIC(6,2), ci {ci_type}, arrt TEXT[], e TEXT, c TEXT COLLATE "C", vc VARCHAR(5))'
            )
            cur.execute(f'CREATE TABLE {tb} (id INT PRIMARY KEY, z TEXT COLLATE "POSIX", e TEXT)')
            cur.execute(
                f"INSERT INTO {ta} VALUES "
                "(1,'a',16777217,3,1.5,2.25,16777217,12.34,'Ab',ARRAY['x','y'],'xbyb','Y'),"
                "(2,'b',5,4,16777217,3e10,9007199254740993,1.00,'cd',ARRAY['z'],'q','z')"
            )
            cur.execute(f"INSERT INTO {tb} VALUES (1,'Z','abc'),(2,'a','b')")
            conn.commit()

            def sub(body: str) -> str:
                return re.sub(r"\b(FROM|JOIN) a\b", rf"\1 {ta}", re.sub(r"\b(FROM|JOIN) b\b", rf"\1 {tb}", body))

            accepted = [sub(b) for b, ok in self._TREE_VIEWS if ok]
            accepted += [sub(b) for b in self._SHADOW_VIEWS]
            accepted += [
                sub(b)
                for b in (
                    "SELECT ARRAY[s] AS v FROM a INTERSECT ALL SELECT string_to_array(e, 'b') FROM a",
                    "SELECT arrt AS v FROM a UNION ALL SELECT ARRAY[ci] FROM a",
                    "SELECT ARRAY[ci] AS v FROM a UNION ALL SELECT arrt FROM a",
                    "SELECT ci AS v FROM a UNION ALL SELECT s FROM a",
                    "SELECT s AS v FROM a UNION ALL SELECT ci FROM a",
                    "SELECT 1.5e0 AS v FROM a",
                    'SELECT c COLLATE "C" AS v FROM a UNION SELECT z FROM b',
                    'SELECT c COLLATE "C" AS v FROM a UNION ALL SELECT z FROM b',
                    "SELECT s || 'x' AS v FROM a UNION ALL SELECT e || 'y' FROM b",
                )
            ]
            refused = [sub(b) for b, ok in self._TREE_VIEWS if not ok] + [
                sub("SELECT to_jsonb(s) AS v FROM a UNION ALL SELECT to_jsonb(e) FROM b"),
                sub("(SELECT i4 AS v FROM a UNION ALL SELECT r FROM a) UNION ALL SELECT d FROM a"),
                sub("SELECT arrt AS v FROM a UNION SELECT regexp_split_to_array(e, 'b') FROM a"),
            ]
            refused.append(sub("SELECT p.id, x.* FROM a p LEFT JOIN (SELECT s AS v0 FROM a UNION SELECT e FROM b) x ON x.v0 = p.s"))
            if ci_type == "CITEXT":
                refused.append(sub("SELECT ci AS v FROM a UNION ALL SELECT vc FROM a"))
            if ci_type != "CITEXT":
                accepted = [v for v in accepted if "ci" not in v.replace("nu", "")]
            # citext defines the aggregate max(citext), so MAX is overloaded by a user-defined
            # aggregate: the catalog classification routes every max() view to the exact full
            # refresh of the original query (UNCLASSIFIED_FUNCTION_FULL_REFRESH), whose column
            # type is the server-reported one. PostgreSQL itself then evaluates the lossy REAL
            # chain, so the refusal (which guards the incremental casts) does not apply.
            shadowed_max = [v for v in refused if "max(v)" in v] if ci_type == "CITEXT" else []
            self.assertEqual(len(shadowed_max), 1 if ci_type == "CITEXT" else 0, refused)
            refused = [v for v in refused if v not in shadowed_max]
            accepted += shadowed_max
            mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
            pairs = []
            for idx, sel in enumerate(accepted):
                mv, lv = f"mv{idx}_{tag}", f"lv{idx}_{tag}"
                plan = mgr.register_view(f"CREATE VIEW {mv} AS {sel};")
                conn.commit()
                if sel in shadowed_max:
                    self.assertEqual(plan.strategy, MaintenanceStrategy.ATOMIC_QUERY_SYNC, sel)
                    self.assertEqual(
                        [d.code for d in plan.strategy_diagnostics], ["UNCLASSIFIED_FUNCTION_FULL_REFRESH"], sel
                    )
                mv_names.append(mv)
                cur.execute(f"CREATE VIEW {lv} AS {sel}")
                conn.commit()
                lv_names.append(lv)
                pairs.append((sel, mv, lv))

            def attr_sig(name: str) -> List[Tuple[str, str, int, int, str]]:
                cur.execute(
                    "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), a.atttypid::int, "
                    "       a.atttypmod::int, COALESCE(co.collname, '-') "
                    "FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
                    "LEFT JOIN pg_catalog.pg_collation co ON co.oid = a.attcollation "
                    "WHERE c.relname = %s AND pg_catalog.pg_table_is_visible(c.oid) AND a.attnum > 0 "
                    "  AND NOT a.attisdropped ORDER BY a.attnum",
                    (name,),
                )
                return [(str(r[0]), str(r[1]), int(r[2]), int(r[3]), str(r[4])) for r in cur.fetchall()]

            def check(step: str) -> None:
                for sel, mv, lv in pairs:
                    cur.execute(f"SELECT * FROM {mv}")
                    got = sorted((tuple(_exact_deep(v) for v in r) for r in cur.fetchall()), key=repr)
                    cur.execute(f"SELECT * FROM {lv}")
                    exp = sorted((tuple(_exact_deep(v) for v in r) for r in cur.fetchall()), key=repr)
                    mv_sig, lv_sig = attr_sig(mv), attr_sig(lv)
                    conn.commit()
                    self.assertEqual(got, exp, f"{step}: {sel}")
                    self.assertEqual(mv_sig, lv_sig, f"{step}: pg_attribute type/typmod/collation of {sel}")

            check("initial")
            for dml in (
                f"INSERT INTO {ta} VALUES (3,'c',16777219,1,16777219,1e20,16777219,99.99,'Ab',ARRAY['x'],'bb','C')",
                f"UPDATE {ta} SET i4 = 16777221, r = 3.5, c = 'k', ci = 'ZZ' WHERE id = 2",
                f"DELETE FROM {tb} WHERE id = 1",
                f"INSERT INTO {tb} VALUES (3,'zz','xbb')",
                f"DELETE FROM {ta} WHERE id = 1",
                # The maximum now comes from the INTEGER arm, which PostgreSQL converts through
                # REAL (16777221 -> 16777220): a max() view must show the rounded value.
                f"UPDATE {ta} SET r = 0, d = 0",
            ):
                cur.execute(dml)
                conn.commit()
                check(dml[:20])

            for idx, bad in enumerate(refused):
                bad_mv = f"mvbad{idx}_{tag}"
                with self.assertRaises(UnsupportedMaterializedViewError, msg=bad):
                    mgr.register_view(f"CREATE VIEW {bad_mv} AS {bad};")
                conn.rollback()
                cur.execute(
                    "SELECT 1 FROM pg_catalog.pg_class WHERE relname = %s AND pg_catalog.pg_table_is_visible(oid)",
                    (bad_mv,),
                )
                self.assertIsNone(cur.fetchone(), f"Failed registration {bad_mv} must roll back cleanly")
                conn.rollback()
        finally:
            conn.rollback()
            for mv in mv_names:
                try:
                    MaterializedViewManager(conn, dialect=Dialect.POSTGRES).drop_view(mv)
                except Exception:
                    conn.rollback()
            for lv in lv_names:
                cur.execute(f"DROP VIEW IF EXISTS {lv} CASCADE")
            for tbl in (ta, tb):
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            if created_citext:
                # Leave the database as found: citext's min/max(citext) change how every
                # later min()/max() view is classified.
                cur.execute("DROP EXTENSION IF EXISTS citext")
            conn.commit()
            conn.close()

    def test_native_postgres_setop_pg_catalog_and_name_types(self) -> None:
        import os

        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        tag = f"rtfix_{uuid.uuid4().hex[:8]}"
        ta, tb, tn = f"a_{tag}", f"b_{tag}", f"n_{tag}"
        conn = _connect(Dialect.POSTGRES, dsn)
        cur = conn.cursor()
        try:
            cur.execute(f"CREATE TABLE {ta} (id INT PRIMARY KEY, k SMALLINT NOT NULL, x SMALLINT)")
            cur.execute(f"CREATE TABLE {tb} (id INT PRIMARY KEY, k BIGINT NOT NULL, y BIGINT)")
            cur.execute(f"CREATE TABLE {tn} (id INT PRIMARY KEY, g INT, x INT, n NAME, c CHAR(4), vc VARCHAR(8), tx TEXT)")
            cur.execute(f"INSERT INTO {ta} VALUES (1, 1, 10), (2, 2, 20)")
            cur.execute(f"INSERT INTO {tb} VALUES (1, 5000000000, 10000000000), (2, 2, 30000000000)")
            cur.execute(
                f"INSERT INTO {tn} VALUES "
                "(1, 1, 10, 'alpha', 'ab', 'v1', 't1'), "
                "(2, 1, 20, NULL, 'cd', 'v2', 't2'), "
                "(3, 2, 30, 'beta', 'ef', NULL, NULL)"
            )
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
            specs = [
                ("uall", f"SELECT id, x FROM {ta} UNION ALL SELECT id, y FROM {tb}"),
                ("uagg", f"WITH u AS (SELECT k, x FROM {ta} UNION ALL SELECT k, y FROM {tb}) SELECT k, MIN(x) AS mnx, MAX(x) AS mxx, SUM(x) AS sx, BIT_XOR(x) AS bx FROM u GROUP BY k"),
                ("pgcat", f"SELECT g, pg_catalog.var_pop(x) AS vp, pg_catalog.variance(x) AS vs, COALESCE(pg_catalog.sum(x), 0) AS cs, pg_catalog.sum(x) FILTER (WHERE x > 15) AS sf FROM {tn} GROUP BY g HAVING pg_catalog.count(*) > 0"),
                ("name", f"SELECT id, COALESCE(n, c) AS n_c, COALESCE(c, n) AS c_n, COALESCE(vc, n) AS vc_n, COALESCE(tx, n) AS tx_n, CASE WHEN id > 1 THEN c ELSE n END AS case_n, CASE WHEN id > 1 THEN n ELSE tx END AS case_tx, NULLIF(n, c) AS nif_nc, NULLIF(c, n) AS nif_cn, NULLIF(vc, c) AS nif_vcc, GREATEST(c, n) AS gr_cn, LEAST(n, tx) AS ls_ntx FROM {tn}"),
                ("mmn", f"SELECT g, MIN(n) AS mnn, MAX(n) AS mxn FROM {tn} GROUP BY g"),
            ]
            pairs = []
            for label, sel in specs:
                mv, lv = f"mv_{label}_{tag}", f"lv_{label}_{tag}"
                mgr.register_view(f"CREATE VIEW {mv} AS {sel};")
                conn.commit()
                cur.execute(f"CREATE VIEW {lv} AS {sel}")
                conn.commit()
                pairs.append((label, mv, lv))

            def check(step: str) -> None:
                for label, mv, lv in pairs:
                    cur.execute(f"SELECT * FROM {mv}")
                    got = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
                    da = _description(cur)
                    cur.execute(f"SELECT * FROM {lv}")
                    exp = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
                    db = _description(cur)
                    conn.commit()
                    self.assertEqual(got, exp, f"PG {label} {step}: {got} != {exp}")
                    self.assertEqual(_metadata_mismatches(Dialect.POSTGRES, da, db), [], f"PG {label} {step} metadata")

            check("initial")
            cur.execute(f"INSERT INTO {tb} VALUES (3, 9000000000, 40000000000)")
            cur.execute(f"UPDATE {tb} SET y = 50000000000 WHERE id = 1")
            cur.execute(f"DELETE FROM {ta} WHERE id = 1")
            cur.execute(f"INSERT INTO {tn} VALUES (4, 2, 50, NULL, 'gh', 'v4', 't4')")
            cur.execute(f"UPDATE {tn} SET x = 40, n = 'gamma' WHERE id = 2")
            cur.execute(f"DELETE FROM {tn} WHERE id = 1")
            conn.commit()
            check("after_dml")
        finally:
            conn.rollback()
            for label in ("uall", "uagg", "pgcat", "name", "mmn"):
                try:
                    MaterializedViewManager(conn, dialect=Dialect.POSTGRES).drop_view(f"mv_{label}_{tag}")
                except Exception:
                    conn.rollback()
                cur.execute(f"DROP VIEW IF EXISTS lv_{label}_{tag} CASCADE")
            for tbl in (ta, tb, tn):
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            conn.commit()
            conn.close()

    def test_native_postgres_minmax_shadowed_by_user_aggregate_full_refresh_types(self) -> None:
        """A user-defined aggregate overloading MIN/MAX (citext's min/max(citext)) makes every
        min()/max() call unclassifiable, so the view is a full refresh of the original query
        (UNCLASSIFIED_FUNCTION_FULL_REFRESH). Every backing column must then have the type the
        server reports for the live view: a CTE set-operation column the server widens
        (SMALLINT/BIGINT -> BIGINT) and SUM over it (NUMERIC) were typed statically from the
        first arm (registration failed with "smallint out of range", SUM read back as int).
        The server type keeps a non-default collation and unwraps a domain (a NOT NULL
        domain column through an outer join holds NULLs)."""
        import os

        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN not set")
        tag = f"shmx_{uuid.uuid4().hex[:8]}"
        ta, tb, tc, dom = f"a_{tag}", f"b_{tag}", f"c_{tag}", f"pos_{tag}"
        conn = _connect(Dialect.POSTGRES, dsn)
        cur = conn.cursor()
        created_citext = False
        names: List[str] = []
        try:
            cur.execute("SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'citext'")
            if cur.fetchone() is None:
                try:
                    cur.execute("CREATE EXTENSION citext")
                    conn.commit()
                    created_citext = True
                except Exception:
                    conn.rollback()
                    self.skipTest("the citext extension is not available")
            cur.execute(f"CREATE TABLE {ta} (id INT PRIMARY KEY, k SMALLINT NOT NULL, x SMALLINT)")
            cur.execute(f"CREATE TABLE {tb} (id INT PRIMARY KEY, k BIGINT NOT NULL, y BIGINT)")
            cur.execute(f"INSERT INTO {ta} VALUES (1, 1, 10), (2, 2, 20)")
            cur.execute(f"INSERT INTO {tb} VALUES (1, 5000000000, 10000000000), (2, 2, 30000000000)")
            cur.execute(f"CREATE DOMAIN {dom} AS INT NOT NULL CHECK (VALUE > 0)")
            cur.execute(f'CREATE TABLE {tc} (id INT PRIMARY KEY, n TEXT COLLATE "C", dv {dom})')
            cur.execute(f"INSERT INTO {tc} VALUES (1, 'b', 7), (3, 'B', 9)")
            conn.commit()
            mgr = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
            specs = [
                f"WITH u AS (SELECT k, x FROM {ta} UNION ALL SELECT k, y FROM {tb}) "
                f"SELECT k, MIN(x) AS mnx, MAX(x) AS mxx, SUM(x) AS sx FROM u GROUP BY k",
                f"SELECT MAX(v) AS m, SUM(v) AS s FROM (SELECT x AS v FROM {ta} UNION ALL SELECT y FROM {tb}) q",
                f"SELECT n, MIN(id) AS m FROM {tc} GROUP BY n",
                f"SELECT a.id, c.dv, MAX(a.x) AS m FROM {ta} a LEFT JOIN {tc} c ON c.id = a.id GROUP BY a.id, c.dv",
            ]
            pairs = []
            for idx, sel in enumerate(specs):
                mv, lv = f"mv{idx}_{tag}", f"lv{idx}_{tag}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    plan = mgr.register_view(f"CREATE VIEW {mv} AS {sel};")
                conn.commit()
                names.append(mv)
                self.assertEqual(
                    [d.code for d in plan.strategy_diagnostics], ["UNCLASSIFIED_FUNCTION_FULL_REFRESH"], sel
                )
                cur.execute(f"CREATE VIEW {lv} AS {sel}")
                conn.commit()
                pairs.append((sel, mv, lv))

            def collations(name: str) -> List[Tuple[str, str]]:
                cur.execute(
                    "SELECT a.attname, COALESCE(co.collname, '-') FROM pg_catalog.pg_attribute a "
                    "JOIN pg_catalog.pg_class c ON c.oid = a.attrelid "
                    "LEFT JOIN pg_catalog.pg_collation co ON co.oid = a.attcollation "
                    "WHERE c.relname = %s AND pg_catalog.pg_table_is_visible(c.oid) AND a.attnum > 0 "
                    "AND NOT a.attisdropped ORDER BY a.attnum",
                    (name,),
                )
                return [(str(r[0]), str(r[1])) for r in cur.fetchall()]

            for sel, mv, lv in pairs:
                self.assertEqual(collations(mv), collations(lv), f"collations: {sel}")
            conn.commit()

            def check(step: str) -> None:
                for sel, mv, lv in pairs:
                    cur.execute(f"SELECT * FROM {mv}")
                    got = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
                    da = _description(cur)
                    cur.execute(f"SELECT * FROM {lv}")
                    exp = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
                    db = _description(cur)
                    conn.commit()
                    self.assertEqual(got, exp, f"{step}: {sel}")
                    self.assertEqual(_metadata_mismatches(Dialect.POSTGRES, da, db), [], f"{step} metadata: {sel}")

            check("initial")
            cur.execute(f"INSERT INTO {tb} VALUES (3, 9000000000, 40000000000)")
            cur.execute(f"UPDATE {tb} SET y = 50000000000 WHERE id = 1")
            cur.execute(f"DELETE FROM {ta} WHERE id = 1")
            cur.execute(f"INSERT INTO {tc} VALUES (2, 'a', 3)")
            conn.commit()
            check("after_dml")
        finally:
            conn.rollback()
            for mv in names:
                try:
                    MaterializedViewManager(conn, dialect=Dialect.POSTGRES).drop_view(mv)
                except Exception:
                    conn.rollback()
            for idx in range(4):
                cur.execute(f"DROP VIEW IF EXISTS lv{idx}_{tag} CASCADE")
            for tbl in (ta, tb, tc):
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
            cur.execute(f"DROP DOMAIN IF EXISTS {dom}")
            if created_citext:
                cur.execute("DROP EXTENSION IF EXISTS citext")
            conn.commit()
            conn.close()

    def test_native_mysql_setop_common_types(self) -> None:
        import os

        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN not set")
        tag = f"rtfix_{uuid.uuid4().hex[:8]}"
        ta, tb = f"a_{tag}", f"b_{tag}"
        conn = _connect(Dialect.MYSQL, dsn)
        cur = conn.cursor()
        try:
            cur.execute(f"CREATE TABLE {ta} (id INT PRIMARY KEY, k SMALLINT NOT NULL, x SMALLINT, z SMALLINT) ENGINE=InnoDB")
            cur.execute(f"CREATE TABLE {tb} (id INT PRIMARY KEY, k BIGINT NOT NULL, x BIGINT, z INT) ENGINE=InnoDB")
            cur.execute(f"INSERT INTO {ta} VALUES (1, 1, 10, 100), (2, 2, 20, 200)")
            cur.execute(f"INSERT INTO {tb} VALUES (1, 5000000000, 10000000000, 100000), (2, 2, 30000000000, 200000)")
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.MYSQL)
            specs = [
                ("uall", f"SELECT id, x FROM {ta} UNION ALL SELECT id, x FROM {tb}"),
                ("uagg", f"WITH u AS (SELECT k, x, z FROM {ta} UNION ALL SELECT k, x, z FROM {tb}) SELECT k, MIN(x) AS mnx, MAX(x) AS mxx, MIN(z) AS mnz, MAX(z) AS mxz, SUM(x) AS sx FROM u GROUP BY k"),
            ]
            pairs = []
            for label, sel in specs:
                mv, lv = f"mv_{label}_{tag}", f"lv_{label}_{tag}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    mgr.register_view(f"CREATE VIEW {mv} AS {sel};", ignore_limitations=True)
                conn.commit()
                cur.execute(f"CREATE VIEW {lv} AS {sel}")
                conn.commit()
                pairs.append((label, mv, lv))

            def check(step: str) -> None:
                for label, mv, lv in pairs:
                    cur.execute(f"SELECT * FROM {mv}")
                    got = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
                    da = _description(cur)
                    cur.execute(f"SELECT * FROM {lv}")
                    exp = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
                    db = _description(cur)
                    conn.commit()
                    self.assertEqual(got, exp, f"MySQL {label} {step}: {got} != {exp}")
                    self.assertEqual(_metadata_mismatches(Dialect.MYSQL, da, db), [], f"MySQL {label} {step} metadata")

            check("initial")
            cur.execute(f"INSERT INTO {tb} VALUES (3, 9000000000, 40000000000, 300000)")
            cur.execute(f"UPDATE {tb} SET x = 50000000000, z = 400000 WHERE id = 1")
            cur.execute(f"DELETE FROM {ta} WHERE id = 1")
            conn.commit()
            check("after_dml")
        finally:
            conn.rollback()
            for label in ("uall", "uagg"):
                try:
                    MaterializedViewManager(conn, dialect=Dialect.MYSQL).drop_view(f"mv_{label}_{tag}")
                except Exception:
                    conn.rollback()
                cur.execute(f"DROP VIEW IF EXISTS lv_{label}_{tag}")
            for tbl in (ta, tb):
                cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            conn.commit()
            conn.close()


class ExactWindowExpressionTypesTests(unittest.TestCase):
    """Exact storage-type inference and native SQLite preservation for window
    SUM/AVG over expressions, nested aggregates, CTE passthroughs, and derived
    tables (covering SQLite >2**53 integers and MySQL/MariaDB/PostgreSQL exact
    window expressions while keeping approximate inputs refused by default)."""

    def test_sqlite_window_sum_preserves_int64_above_2_53_and_mixed_storage_classes(self) -> None:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, b INTEGER, r REAL)")
            conn.executemany(
                "INSERT INTO t VALUES (?, ?, ?, ?)",
                [
                    (1, "a", 2**53, 1.5),
                    (2, "a", 1, 2.5),
                    (3, "a", 2, 3.0),
                    (4, "b", 2**60, 0.5),
                    (5, "b", 3, 0.5),
                    (6, "c", None, None),
                ],
            )
            conn.commit()

            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            views = [
                (
                    "v_win_t2",
                    "SELECT id, g, SUM(b) OVER (PARTITION BY g ORDER BY id) AS s_run, "
                    "SUM(b) OVER (PARTITION BY g) AS s_part, SUM(b) OVER () AS s_all FROM t",
                ),
                (
                    "v_win_expr",
                    "SELECT id, g, COALESCE(SUM(b) OVER (PARTITION BY g), 0) AS s_coal, "
                    "SUM(b) OVER (PARTITION BY g) + 1 AS s_add FROM t",
                ),
                (
                    "v_win_cte",
                    "WITH c AS (SELECT id, g, SUM(b) OVER (PARTITION BY g) AS s_cte FROM t) "
                    "SELECT id, g, s_cte FROM c",
                ),
                (
                    "v_win_grp",
                    "SELECT g, SUM(COUNT(*)) OVER () AS s_cnt, SUM(SUM(b)) OVER () AS s_sum "
                    "FROM t GROUP BY g",
                ),
            ]
            for vn, q in views:
                mgr.register_view(f"CREATE VIEW {vn} AS {q};")

            def _check(step: str) -> None:
                for vn, q in views:
                    got = conn.execute(f"SELECT * FROM {vn} ORDER BY 1").fetchall()
                    exp = conn.execute(f"{q} ORDER BY 1").fetchall()
                    self.assertEqual(got, exp, f"SQLite {vn} {step}: {got!r} != {exp!r}")
                    got_types = [[type(v) for v in row] for row in got]
                    exp_types = [[type(v) for v in row] for row in exp]
                    self.assertEqual(got_types, exp_types, f"SQLite {vn} {step} types: {got_types!r} != {exp_types!r}")

            _check("initial")
            # Confirm exact 2**53 + 1 and 2**60 + 3 integer values (not rounded floats).
            row_id2 = conn.execute("SELECT s_run, s_part FROM v_win_t2 WHERE id = 2").fetchone()
            self.assertEqual(row_id2, (2**53 + 1, 2**53 + 3))
            self.assertIs(type(row_id2[0]), int)
            self.assertIs(type(row_id2[1]), int)

            # DML: INSERT above 2**53, UPDATE, DELETE, and mixed REAL value in INTEGER column.
            conn.execute("INSERT INTO t VALUES (7, 'a', ?, 1.0)", (2**53 + 5,))
            conn.commit()
            _check("after_insert")
            conn.execute("UPDATE t SET b = ? WHERE id = 2", (2**53 + 7,))
            conn.commit()
            _check("after_update")
            conn.execute("INSERT INTO t VALUES (8, 'b', 10.0, 2.0)")
            conn.commit()
            _check("after_real_into_int_col")
            conn.execute("DELETE FROM t WHERE id = 8")
            conn.commit()
            _check("after_delete_real_back_to_int")
        finally:
            conn.close()

    def test_exact_window_expression_types_and_fail_closed_offline(self) -> None:
        import re

        my_schema = (
            "CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(10), i INT, b BIGINT, "
            "d DECIMAL(10, 2), s VARCHAR(20), dt DATE, tm6 TIME(6), bin VARBINARY(16), f DOUBLE);"
            "CREATE TABLE t2 (id INT PRIMARY KEY, g VARCHAR(10), x DOUBLE);"
        )
        pg_schema = (
            "CREATE TABLE t (id INT PRIMARY KEY, g VARCHAR(10), i INT, b BIGINT, "
            "d NUMERIC(10, 2), s VARCHAR(20), dt DATE, b1 BIT(4), b2 BIT(4), f DOUBLE PRECISION);"
            "CREATE TABLE t2 (id INT PRIMARY KEY, g VARCHAR(10), x DOUBLE PRECISION);"
        )

        def _col_type(plan: Any, col: str) -> str:
            m = re.search(
                rf'(?m)^\s*"?{re.escape(col)}"?\s+([A-Z][A-Z0-9_ (),]*?)(?:\s+NOT\s+NULL)?\s*,?\s*$',
                plan.backing_table_ddl,
            )
            assert m is not None, f"column {col!r} not found in:\n{plan.backing_table_ddl}"
            return m.group(1).strip()

        mysql_cases = [
            ("SELECT id, SUM(i > 0) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 0)"}),
            ("SELECT id, AVG(i > 0) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(5, 4)"}),
            ("SELECT id, SUM(i / 2) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 4)"}),
            ("SELECT id, AVG(i / 2) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(18, 8)"}),
            ("SELECT id, SUM(LENGTH(s)) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 0)"}),
            ("SELECT id, SUM(YEAR(dt)) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 0)"}),
            ("SELECT id, SUM(DATEDIFF(dt, dt)) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 0)"}),
            ("SELECT id, SUM(i & 7) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 0)"}),
            ("SELECT id, ~i AS w FROM t", {"w": "BIGINT UNSIGNED"}),
            ("SELECT id, AVG(BIT_COUNT(bin)) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(23, 4)"}),
            ("SELECT id, SUM(ROUND(d, -1)) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 0)"}),
            ("SELECT id, SUM(TRUNCATE(d, -1)) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 0)"}),
            ("SELECT id, SUM(ROUND(d, 5)) OVER (PARTITION BY g) AS w FROM t", {"w": "DECIMAL(65, 2)"}),
            ("SELECT id, COALESCE(SUM(d) OVER (PARTITION BY g), 0) AS w FROM t", {"w": "DECIMAL(65, 2)"}),
            ("SELECT id, SUM(d) OVER (PARTITION BY g) + 1 AS w FROM t", {"w": "DECIMAL(65, 2)"}),
            ("SELECT id, COALESCE(AVG(i) OVER (PARTITION BY g), 0) AS w FROM t", {"w": "DECIMAL(65, 4)"}),
            ("SELECT g, SUM(COUNT(*)) OVER () AS w FROM t GROUP BY g", {"w": "DECIMAL(65, 0)"}),
            ("SELECT g, AVG(COUNT(*)) OVER () AS w FROM t GROUP BY g", {"w": "DECIMAL(23, 4)"}),
            ("SELECT g, SUM(AVG(d)) OVER () AS w FROM t GROUP BY g", {"w": "DECIMAL(65, 6)"}),
            (
                "WITH c AS (SELECT id, SUM(d) OVER (PARTITION BY g) AS w, AVG(i) OVER (PARTITION BY g) AS a FROM t) "
                "SELECT id, w, a FROM c",
                {"w": "DECIMAL(65, 2)", "a": "DECIMAL(14, 4)"},
            ),
            (
                "WITH w AS (SELECT id, g, LAG(d, 1, 0) OVER (PARTITION BY g ORDER BY id) AS prev_d, "
                "ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) AS rn FROM t) "
                "SELECT id, SUM(prev_d) OVER (PARTITION BY g) AS w, SUM(rn) OVER (PARTITION BY g) AS a FROM w",
                {"w": "DECIMAL(65, 2)", "a": "DECIMAL(65, 0)"},
            ),
            (
                "SELECT id, SUM(d) OVER (PARTITION BY g) AS w FROM (SELECT a.id, a.g, a.i * 2 AS d FROM t a) sub",
                {"w": "DECIMAL(65, 0)"},
            ),
            (
                "WITH c1 AS (SELECT id, g, i * 2 AS x FROM t), c2 AS (SELECT id, g, x + 1 AS y FROM c1) "
                "SELECT id, SUM(y) OVER (PARTITION BY g) AS w, y / 2 AS a FROM c2",
                {"w": "DECIMAL(65, 0)", "a": "DECIMAL(16, 4)"},
            ),
            (
                "WITH c(id, g, x) AS (SELECT id, g, i * 2 FROM t) SELECT id, SUM(x) OVER (PARTITION BY g) AS w FROM c",
                {"w": "DECIMAL(65, 0)"},
            ),
            (
                "WITH c AS (SELECT id, g, LAG(i, 1, 0.5) OVER (PARTITION BY g ORDER BY id) AS w FROM t) "
                "SELECT id, w FROM c",
                {"w": "DECIMAL(11, 1)"},
            ),
            (
                "WITH c AS (SELECT id, g, g AS x FROM t) SELECT id, x AS w, COUNT(*) OVER () AS x FROM c",
                {"w": "VARCHAR(10)"},
            ),
            (
                "WITH RECURSIVE r AS (SELECT id, 1 AS n FROM t UNION ALL SELECT id, n + 1 FROM r WHERE n < 5) "
                "SELECT id, n + 1 AS w FROM r",
                {"w": "BIGINT"},
            ),
        ]
        for q, expected in mysql_cases:
            with self.subTest(engine="mysql:8.4", select=q):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    plan = generate_materialized_view(
                        my_schema,
                        f"CREATE VIEW v AS {q};",
                        dialect="mysql:8.4",
                        accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                    )
                for col, exp_t in expected.items():
                    self.assertEqual(_col_type(plan, col), exp_t)

        pg_cases = [
            ("SELECT id, SUM(LENGTH(s)) OVER (PARTITION BY g) AS w FROM t", {"w": "BIGINT"}),
            ("SELECT id, SUM(i & 7) OVER (PARTITION BY g) AS w FROM t", {"w": "BIGINT"}),
            ("SELECT id, SUM((i / 2) & 7) OVER (PARTITION BY g) AS w FROM t", {"w": "BIGINT"}),
            ("SELECT id, b1 & b2 AS w FROM t", {"w": "BIT(4)"}),
            ("SELECT id, SUM(b & 7) OVER (PARTITION BY g) AS w FROM t", {"w": "NUMERIC"}),
            ("SELECT id, COALESCE(SUM(d) OVER (PARTITION BY g), 0) AS w FROM t", {"w": "NUMERIC"}),
            ("SELECT id, SUM(d) OVER (PARTITION BY g) + 1 AS w FROM t", {"w": "NUMERIC"}),
            ("SELECT g, SUM(COUNT(*)) OVER () AS w FROM t GROUP BY g", {"w": "NUMERIC"}),
            ("SELECT g, AVG(COUNT(*)) OVER () AS w FROM t GROUP BY g", {"w": "NUMERIC"}),
            ("SELECT g, SUM(AVG(d)) OVER () AS w FROM t GROUP BY g", {"w": "NUMERIC"}),
            (
                "WITH c AS (SELECT id, SUM(d) OVER (PARTITION BY g) AS w, AVG(i) OVER (PARTITION BY g) AS a FROM t) "
                "SELECT id, w, a FROM c",
                {"w": "NUMERIC", "a": "NUMERIC"},
            ),
            (
                "SELECT id, SUM(d) OVER (PARTITION BY g) AS w FROM (SELECT a.id, a.g, a.i * 2 AS d FROM t a) sub",
                {"w": "BIGINT"},
            ),
            (
                "WITH c1 AS (SELECT id, g, i * 2 AS x FROM t), c2 AS (SELECT id, g, x + 1 AS y FROM c1) "
                "SELECT id, SUM(y) OVER (PARTITION BY g) AS w FROM c2",
                {"w": "BIGINT"},
            ),
            (
                "WITH c(id, g, x) AS (SELECT id, g, i * 2 FROM t) SELECT id, SUM(x) OVER (PARTITION BY g) AS w FROM c",
                {"w": "BIGINT"},
            ),
            (
                "WITH c AS (SELECT id, g, LAG(i, 1, 0.5) OVER (PARTITION BY g ORDER BY id) AS w FROM t) "
                "SELECT id, w FROM c",
                {"w": "NUMERIC"},
            ),
            (
                "WITH c AS (SELECT id, g, g AS x FROM t) SELECT id, x AS w, COUNT(*) OVER () AS x FROM c",
                {"w": "VARCHAR(10)"},
            ),
            (
                "WITH RECURSIVE r AS (SELECT id, 1 AS n FROM t UNION ALL SELECT id, n + 1 FROM r WHERE n < 5) "
                "SELECT id, n + 1 AS w FROM r",
                {"w": "INTEGER"},
            ),
        ]
        for q, expected in pg_cases:
            with self.subTest(engine="postgres:18", select=q):
                plan = generate_materialized_view(pg_schema, f"CREATE VIEW v AS {q};", dialect="postgres:18")
                for col, exp_t in expected.items():
                    self.assertEqual(_col_type(plan, col), exp_t)

        # Approximate or unprovably exact expressions MUST remain refused by default.
        for q in (
            "SELECT id, SUM(f) OVER (PARTITION BY g) AS w FROM t",
            "SELECT id, SUM(SQRT(i)) OVER (PARTITION BY g) AS w FROM t",
            "SELECT id, SUM(ROUND(f, 1)) OVER (PARTITION BY g) AS w FROM t",
            "SELECT id, SUM(i & f) OVER (PARTITION BY g) AS w FROM t",
            "SELECT id, SUM(f) FILTER (WHERE i > 0) OVER (PARTITION BY g) AS w FROM t",
            "WITH u AS (SELECT id, g, i AS x FROM t UNION ALL SELECT * FROM t2) "
            "SELECT id, SUM(x) OVER (PARTITION BY g) AS w FROM u",
            "SELECT id, SUM(x) OVER (PARTITION BY g) AS w "
            "FROM (SELECT id, g, i AS x FROM t UNION ALL SELECT id, g, f FROM t) sub",
        ):
            with self.subTest(refused_pg=q):
                with self.assertRaises(EngineLimitationError):
                    generate_materialized_view(pg_schema, f"CREATE VIEW v AS {q};", dialect="postgres:18")
            with self.subTest(refused_my=q):
                with self.assertRaises(EngineLimitationError):
                    generate_materialized_view(
                        my_schema,
                        f"CREATE VIEW v AS {q};",
                        dialect="mysql:8.4",
                        accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                    )
        # MySQL/MariaDB-specific fail-closed cases: MariaDB TIME_TO_SEC returns DOUBLE,
        # MySQL BIT_AND(VARBINARY) returns VARBINARY -> SUM converts to DOUBLE,
        # and `||` inside a CTE is string concatenation under PIPES_AS_CONCAT.
        for my_dialect, my_refused in (
            ("mariadb:10.11", "SELECT id, SUM(TIME_TO_SEC(tm6)) OVER (PARTITION BY g) AS w FROM t"),
            ("mysql:8.4", "SELECT g, SUM(BIT_AND(bin)) OVER () AS w FROM t GROUP BY g"),
            ("mysql:8.4", "WITH c AS (SELECT id, g, s || 'x' AS flag FROM t) SELECT id, SUM(flag) OVER (PARTITION BY g) AS w FROM c"),
        ):
            with self.subTest(refused_dialect=my_dialect, select=my_refused):
                with self.assertRaises(EngineLimitationError):
                    generate_materialized_view(
                        my_schema,
                        f"CREATE VIEW v AS {my_refused};",
                        dialect=my_dialect,
                        accept_limitations=("MYSQL_NO_TRUNCATE_TRIGGER",),
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
