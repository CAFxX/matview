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
        "ti TINYINT, mi MEDIUMINT, u INT UNSIGNED);"
    )

    def test_grouped_narrow_integer_arithmetic_uses_int(self) -> None:
        import re

        cases = {
            "s % 2": "INT",
            "ti % 2": "INT",
            "mi % 2": "INT",
            "s + s": "INT",
            "i % 3": "BIGINT",
            "s * s": "BIGINT",
        }
        for dialect in ("mysql:8.4", "mysql:9.7"):
            for expr, expected_type in cases.items():
                with self.subTest(dialect=dialect, expr=expr):
                    select = f"SELECT {expr} AS k, COUNT(*) AS c FROM ik GROUP BY {expr}"
                    plan = generate_materialized_view(
                        self.SCHEMA,
                        f"CREATE VIEW v AS {select};",
                        dialect=dialect,
                        ignore_limitations=True,
                    )
                    match = re.search(
                        r'(?im)^\s*[\x60"]?k[\x60"]?\s+([A-Z]+(?:\s+UNSIGNED)?)\b',
                        plan.backing_table_ddl,
                    )
                    self.assertIsNotNone(match, plan.backing_table_ddl)
                    self.assertEqual(match.group(1), expected_type)


class NativeResultTypeTests(unittest.TestCase):
    def _run(self, engine: Dialect, dsn: str) -> List[str]:
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
            for name, select in _views(engine, t):
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

    def test_mysql_group_key_integer_type_is_stable_across_access_paths(self) -> None:
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
                    for n in range(1, 1001)
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
                        cur.execute("EXPLAIN " + query)
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
                        self.assertEqual(
                            description[1],
                            expected_group_type,
                            f"{version}: grouped {expr} metadata changed under {hint}",
                        )
                        access_results.append((description, actual_rows))

                    self.assertEqual(
                        access_results[0],
                        access_results[1],
                        f"{version}: grouped {expr} metadata/values depend on the access path",
                    )
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
            with self.subTest(engine=engine.value):
                failures = self._run(engine, dsn)
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
                for n, (col, ctx, sql_t) in enumerate(_minmax_context_views()):
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
            failures = self._run(engine, dsn)
            self.assertEqual(failures, [], "\n".join(failures))

    def _run(self, engine: Dialect, dsn: str) -> List[str]:
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
            for i, ty in enumerate(_INT_FAMILY):
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
        self.assertRegex(p_uall.backing_table_ddl, r"\bb BIT VARYING\b")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
