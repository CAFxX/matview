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
    MaterializedViewManager,
    generate_materialized_view,
)
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
        ("union_sum", f"SELECT g, SUM(v) AS r FROM (SELECT g, i AS v FROM {t} UNION ALL SELECT g, s AS v FROM {t}) AS u GROUP BY g"),
    ]
    views += [
        # String CASE/COALESCE resolution order (PostgreSQL: CASE resolves its
        # ELSE first; MySQL/MariaDB: common VARCHAR).
        ("case_varchar_char", f"SELECT id, CASE WHEN g = 1 THEN v ELSE c END AS r FROM {t}"),
        ("case_char_varchar", f"SELECT id, CASE WHEN g = 1 THEN c ELSE v END AS r FROM {t}"),
        ("coalesce_char_varchar", f"SELECT id, COALESCE(c, v) AS r FROM {t}"),
    ]
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
                            manager.register_view(f"CREATE VIEW {mv} AS {select};")
                        conn.rollback()
                        manager.register_view(f"CREATE VIEW {mv} AS {select};", ignore_limitations=True)
                        codes = [str(w.message) for w in caught if issubclass(w.category, EngineLimitationWarning)]
                        self.assertEqual(len(codes), 1)
                        self.assertIn("MYSQL_NO_TRUNCATE_TRIGGER", codes[0])
                    else:
                        manager.register_view(f"CREATE VIEW {mv} AS {select};")
                conn.commit()
                pairs.append((name, mv, live))
                mv_cat = [r[1:] for r in _catalog_types(engine, cur, mv)]
                live_cat = [r[1:] for r in _catalog_types(engine, cur, live)]
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

    def test_native_result_types_match_live_views(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN for native result-type tests")
        for engine, dsn in engines:
            with self.subTest(engine=engine.value):
                failures = self._run(engine, dsn)
                self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
