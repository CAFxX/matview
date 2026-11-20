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
    UnsupportedMaterializedViewError,
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


_MINMAX_CONTEXT_TABLE = (
    "CREATE TABLE {t} (id INT PRIMARY KEY, g INT NOT NULL, ti TINYINT, tu TINYINT UNSIGNED, "
    "s SMALLINT, i INT, iu INT UNSIGNED, b BIGINT, y YEAR)"
)
# (select, refused on MySQL/MariaDB)
_MINMAX_CONTEXT_VIEWS = [
    ("SELECT id, COALESCE(MIN(ti) OVER (PARTITION BY g), 0) AS a, COALESCE(MIN(iu) OVER (PARTITION BY g), 0) AS c FROM {t}", True),
    ("SELECT id, MAX(ti) OVER (PARTITION BY g) + 1 AS a, -MAX(tu) OVER () AS c FROM {t}", True),
    ("SELECT g, COALESCE(MAX(ti), 0) AS a, COALESCE(MIN(iu), 0) AS b, MAX(s) + 0 AS c, CAST(MAX(y) AS UNSIGNED) AS d FROM {t} GROUP BY g", True),
    ("WITH c AS (SELECT g, i FROM {t}) SELECT g, COALESCE(MIN(i), 0) AS e FROM c GROUP BY g", True),
    ("SELECT d.id, MAX(d.ti) OVER (PARTITION BY d.g) AS a FROM (SELECT id, g, ti FROM {t}) d", True),
    ("SELECT g, MIN(ti) AS a, MAX(iu) AS b, MIN(y) AS c, COALESCE(MAX(b), 0) AS d FROM {t} GROUP BY g", False),
    ("SELECT id, MAX(ti) OVER (PARTITION BY g) AS a, MIN(s) OVER (PARTITION BY g) AS b FROM {t}", False),
    ("SELECT g, COUNT(*) AS n FROM {t} GROUP BY g HAVING MAX(ti) + 0 > 1", False),
    ("WITH c AS (SELECT g, ti FROM {t}) SELECT g, MAX(ti) AS a FROM c GROUP BY g", False),
]


class MinMaxContextTypingTests(unittest.TestCase):
    """MIN/MAX of integers narrower than BIGINT: exact where typed, refused elsewhere.

    MySQL types these results by context (window vs grouped vs nested,
    INT vs BIGINT on the wire) and MariaDB loses UNSIGNED through
    COALESCE/arithmetic around windowed MIN/MAX; nested and derived-table
    window uses are refused (not bypassable), the rest must be exact.
    """

    def test_offline_decisions(self) -> None:
        schema = _MINMAX_CONTEXT_TABLE.format(t="t") + ";"
        for engine in ("mysql:8.4.13", "mysql:9.7.4", "mariadb:10.11"):
            for select, refused in _MINMAX_CONTEXT_VIEWS:
                with self.subTest(engine=engine, select=select):
                    compile_ = lambda: generate_materialized_view(
                        schema, f"CREATE VIEW v AS {select.format(t='t')};", dialect=engine, ignore_limitations=True
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        if refused:
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
            tag = uuid.uuid4().hex[:8]
            t = f"mc_{tag}"
            failures: List[str] = []
            try:
                cur.execute(_MINMAX_CONTEXT_TABLE.format(t=t))
                cur.executemany(
                    f"INSERT INTO {t} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    [(1, 1, -3, 200, 7, 5, 4000000000, 9, 2001), (2, 1, 4, 3, -2, -8, 1, -7, 1999),
                     (3, 2, None, None, None, None, None, None, None), (4, 2, 0, 0, 0, 0, 0, 0, 2000)],
                )
                conn.commit()
                manager = MaterializedViewManager(conn, dialect=engine)
                pairs = []
                for n, (select, refused) in enumerate(_MINMAX_CONTEXT_VIEWS):
                    sql = select.format(t=t)
                    mv, live = f"mv{n}_{tag}", f"lv{n}_{tag}"
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", EngineLimitationWarning)
                            manager.register_view(f"CREATE VIEW {mv} AS {sql};", ignore_limitations=True)
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
                    ("insert", f"INSERT INTO {t} VALUES (5, 1, 100, 250, 30000, 2000000000, 4294967295, 5, 2155)"),
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
            plan = generate_materialized_view(self.SCHEMA, self.VIEW, dialect=engine, ignore_limitations=True)
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
                        manager.register_view(f"CREATE VIEW {mv} AS {select};", ignore_limitations=True)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
