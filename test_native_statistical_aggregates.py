#!/usr/bin/env python3
"""Native result semantics and representation of VAR/STDDEV materializations.

Native tests compare the materialized reader against the original view
executed by the engine -- the only oracle -- for initial population,
INSERT/UPDATE/DELETE (including group moves, singleton, emptied and new
groups) and rollback. Values are compared exactly (floats bit-for-bit,
Decimals including their scale: `0` and `0.0` are different results), and
result metadata deliberately: DB-API description fields and the catalog
column type of the reader view versus a live view of the same SELECT.

Native engines are selected with POSTGRES_DSN and/or MYSQL_DSN (MySQL or
MariaDB; the product is read from the server, not trusted from a label).
"""

from __future__ import annotations

import os
import re
import unittest
import uuid
import warnings
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from matview_sync import (
    AggFunc,
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    MaterializedViewManager,
    generate_materialized_view,
    parse_table_schema,
)

_SCHEMA = "CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, x DOUBLE PRECISION);"

_EXPLICIT = {
    "VAR_POP": AggFunc.VAR_POP,
    "VAR_SAMP": AggFunc.VAR_SAMP,
    "STDDEV_POP": AggFunc.STDDEV_POP,
    "STDDEV_SAMP": AggFunc.STDDEV_SAMP,
}


class NativeResultRepresentationPlanTests(unittest.TestCase):
    """Offline guards for the native representation of VAR/STDDEV results.

    Structural only: the native tests below establish the engine semantics
    (PostgreSQL returns a one-row NUMERIC population statistic as `0`, MariaDB
    fixes exact-input result decimals and ROUND(x, 31) is not an identity).
    """

    def _plan(self, engine: str, col_type: str, select: str, **kw: Any):
        dialect = Dialect.POSTGRES if engine.startswith("postgres") else Dialect.MYSQL
        schema = f"CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, x {col_type});"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            return generate_materialized_view(
                parse_table_schema(schema, dialect=dialect),
                f"CREATE VIEW v AS {select};",
                dialect=engine,
                ignore_limitations=kw.pop("ignore_limitations", True),
                **kw,
            )

    def test_postgres_first_row_population_statistic_keeps_native_scale(self) -> None:
        for col_type, zero in (("NUMERIC(12,3)", "0"), ("INTEGER", "0"), ("DOUBLE PRECISION", "0.0")):
            with self.subTest(type=col_type):
                plan = self._plan("postgres:18", col_type, "SELECT g, VAR_POP(x) AS vp, STDDEV_POP(x) AS sp FROM t GROUP BY g")
                trigger = "\n".join(plan.trigger_ddls)
                self.assertIn(f"(CASE WHEN (NEW.x) IS NULL THEN NULL ELSE {zero} END)", trigger)
                other = "0.0" if zero == "0" else "0"
                self.assertNotIn(f"(CASE WHEN (NEW.x) IS NULL THEN NULL ELSE {other} END)", trigger)

    def test_mariadb_reader_rounds_only_fixed_decimal_results(self) -> None:
        select = "SELECT g, VAR_POP(x) AS vp, STDDEV_SAMP(x) AS ss FROM t GROUP BY g"
        for col_type, rounded in (
            ("INT", "4"), ("DECIMAL(12,3)", "7"), ("FLOAT(7,2)", "6"), ("DOUBLE", None), ("FLOAT", None),
        ):
            with self.subTest(type=col_type):
                reader = self._plan("mariadb:10.11", col_type, select).reader_view_ddl
                for col in ("vp", "ss"):
                    if rounded is None:
                        self.assertNotIn(f"ROUND({col}", reader)
                    else:
                        self.assertIn(f"ROUND({col}, {rounded}) AS {col}", reader)
        reader = self._plan("mysql:8.4", "INT", select).reader_view_ddl
        self.assertNotIn("ROUND(", reader)

    def test_mariadb_nested_fixed_decimal_statistics_are_a_limitation(self) -> None:
        cases = [
            ("INT", "SELECT g, VAR_SAMP(x) * 2 AS a FROM t GROUP BY g", True),
            ("DECIMAL(12,3)", "SELECT g, COALESCE(STDDEV_POP(x), -1) AS a FROM t GROUP BY g", True),
            ("FLOAT(7,2)", "SELECT g, VAR_SAMP(x) + 1 AS a FROM t GROUP BY g", True),
            ("INT", "SELECT id, VAR_SAMP(x) OVER (PARTITION BY g) AS a FROM t", True),
            ("INT", "SELECT g, VAR_SAMP(CAST(x AS DOUBLE)) * 2 AS a FROM t GROUP BY g", False),
            ("DOUBLE", "SELECT g, VAR_SAMP(x) * 2 AS a FROM t GROUP BY g", False),
            ("INT", "SELECT g, VAR_SAMP(x) AS a FROM t GROUP BY g", False),
            ("INT", "SELECT g, VAR_SAMP(x * 2) AS a FROM t GROUP BY g", False),
            ("INT", "SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING VAR_SAMP(x) > 1", False),
        ]
        for col_type, select, limited in cases:
            for engine in ("mariadb:10.11", "mariadb:12.3", "mysql:8.4"):
                with self.subTest(engine=engine, type=col_type, select=select):
                    with self.assertRaises(EngineLimitationError) as raised:
                        self._plan(engine, col_type, select, ignore_limitations=False)
                    codes = {lim.code for lim in raised.exception.limitations}
                    expect = limited and engine.startswith("mariadb")
                    self.assertEqual("MARIADB_STAT_RESULT_DECIMALS" in codes, expect, codes)


# ---------------------------------------------------------------------------
# Native differential tests
# ---------------------------------------------------------------------------


def _connect(engine: Dialect, dsn: str):
    if engine == Dialect.POSTGRES:
        import psycopg2

        return psycopg2.connect(dsn)
    import pymysql

    parsed = urlparse(dsn)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=parsed.path.lstrip("/") or None,
        autocommit=False,
    )


def _native_engines() -> List[Tuple[Dialect, str]]:
    out = []
    if os.environ.get("POSTGRES_DSN"):
        out.append((Dialect.POSTGRES, os.environ["POSTGRES_DSN"]))
    if os.environ.get("MYSQL_DSN"):
        out.append((Dialect.MYSQL, os.environ["MYSQL_DSN"]))
    return out


def _exact(value: Any) -> Any:
    """Canonical exact representation; floats are compared bit-for-bit."""
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, Decimal):
        return ("decimal", str(value))
    return (type(value).__name__, value)


def _description(cur) -> List[Tuple[Any, ...]]:
    return [tuple(col[:7]) for col in cur.description]


def _metadata_mismatches(engine: Dialect, mv: Sequence[Any], live: Sequence[Any]) -> List[str]:
    """Compare DB-API result metadata column by column.

    Name, type code, scale and nullability are compared whenever the driver
    reports them. Precision is compared for every PostgreSQL column (psycopg2
    derives it from the type modifier) and for MySQL exact DECIMAL columns.
    PyMySQL's precision of integer and floating columns is the wire display
    width (BIGINT column 20 vs COUNT(*) 21, DOUBLE column 22 vs aggregate 23),
    which is not part of the result type; those columns' types are compared
    through the catalog (`_catalog_types`) instead.
    """
    out: List[str] = []
    if len(mv) != len(live):
        return [f"column count {len(mv)} != {len(live)}"]
    for idx, (a, b) in enumerate(zip(mv, live)):
        fields = [("name", 0), ("type_code", 1), ("scale", 5), ("null_ok", 6)]
        if engine == Dialect.POSTGRES or b[1] in (0, 246):  # MySQL DECIMAL / NEWDECIMAL
            fields.append(("precision", 4))
        for name, pos in fields:
            if a[pos] is None and b[pos] is None:
                continue
            if a[pos] != b[pos]:
                out.append(f"column {idx} ({b[0]}) {name}: mv={a[pos]!r} live={b[pos]!r}")
    return out


def _catalog_types(engine: Dialect, cur, view: str) -> List[Tuple[Any, ...]]:
    if engine == Dialect.POSTGRES:
        cur.execute(
            "SELECT attname, format_type(atttypid, atttypmod), attcollation::regcollation::text "
            "FROM pg_attribute WHERE attrelid = %s::regclass AND attnum > 0 AND NOT attisdropped "
            "ORDER BY attnum",
            (view,),
        )
    else:
        cur.execute(
            "SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, CHARACTER_SET_NAME, COLLATION_NAME "
            "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s "
            "ORDER BY ORDINAL_POSITION",
            (view,),
        )
        return [(r[0], _normalize_mysql_column_type(r[1])) + tuple(r[2:]) for r in cur.fetchall()]
    return [tuple(r) for r in cur.fetchall()]


def _normalize_mysql_column_type(column_type: str) -> str:
    """Drop display widths from a MySQL/MariaDB COLUMN_TYPE, keep everything else.

    Integer display widths (`bigint(20)` vs `bigint(21)` for COUNT) are
    documented as cosmetic and deprecated, and MariaDB reports a floating
    expression's maximum printed length as `double(M,D)`. The fixed number of
    decimals D of a floating result is kept: MariaDB rounds VAR/STDDEV results
    to it, so it is part of the result. Exact DECIMAL(P,S), character lengths,
    signedness, nullability and collation are compared unchanged.
    """
    t = re.sub(r"^(tinyint|smallint|mediumint|int|bigint)\(\d+\)", r"\1", column_type)
    return re.sub(r"^(float|double)\(\d+,(\d+)\)", r"\1(*,\2)", t)


_SEED = [
    (1, 1, "1"), (2, 1, "2"), (3, 1, "4"),      # group 1: three rows
    (4, 2, "7"),                                # group 2: singleton
    (5, 3, None),                               # group 3: only NULL
    (6, 4, "-1.5"), (7, 4, "2.25"),             # group 4: two rows
]
# Integer columns use integral values for the fractional seeds.
_INT_SEED = [(i, g, {"-1.5": "-2", "2.25": "5"}.get(x, x)) for i, g, x in _SEED]

# (label, statement) -- each committed and then compared.
_DML = [
    ("insert into singleton", "INSERT INTO {t} (id, g, x) VALUES (10, 2, 9)"),
    ("insert multi-row", "INSERT INTO {t} (id, g, x) VALUES (11, 5, 3), (12, 5, 3), (13, 3, 8)"),
    ("update value", "UPDATE {t} SET x = 3 WHERE id = 1"),
    ("move row to other group", "UPDATE {t} SET g = 2 WHERE id = 2"),
    ("update to NULL", "UPDATE {t} SET x = NULL WHERE id = 6"),
    ("delete to singleton", "DELETE FROM {t} WHERE id = 10"),
    ("delete whole group", "DELETE FROM {t} WHERE g = 5"),
    ("insert new group", "INSERT INTO {t} (id, g, x) VALUES (14, 6, 6)"),
]


class NativeStatisticalAggregateTests(unittest.TestCase):
    """MV vs native live view for every function, shape, type and DML step."""

    def _server(self, engine: Dialect, conn) -> str:
        cur = conn.cursor()
        cur.execute("SELECT VERSION()")
        version = str(cur.fetchone()[0])
        conn.commit()
        return version

    def _views(self, engine: Dialect, t: str) -> List[Tuple[str, str]]:
        fns = list(_EXPLICIT)
        sel = ", ".join(f"{fn}(x) AS s_{fn.lower()}" for fn in fns)
        views = [("grouped", f"SELECT g, COUNT(x) AS n, {sel} FROM {t} GROUP BY g")]
        # Every shape is checked with sample and population functions.
        pairs = [("VAR_SAMP", "STDDEV_SAMP"), ("VAR_POP", "STDDEV_POP")]
        for var_fn, std_fn in pairs:
            k = f"{var_fn}_{std_fn}".lower()
            views += [
                (f"global_{k}", f"SELECT COUNT(*) AS n, {var_fn}(x) AS v, {std_fn}(x) AS s FROM {t}"),
                (
                    f"nested_{k}",
                    f"SELECT g, {var_fn}(x) * 2 AS v2, {std_fn}(x) + 1 AS s1, "
                    f"COALESCE({var_fn}(x), -1) AS vc FROM {t} GROUP BY g",
                ),
                # Singleton groups have a NULL sample but a 0 population
                # variance: the HAVING filter keeps different groups.
                (f"having_{k}", f"SELECT g, COUNT(*) AS c FROM {t} GROUP BY g HAVING {var_fn}(x) >= 0"),
                (
                    f"join_{k}",
                    f"SELECT d.label, {var_fn}(f.x) AS v, {std_fn}(f.x) AS s FROM {t} f "
                    f"JOIN {t}_d d ON d.g = f.g GROUP BY d.label",
                ),
            ]
            double = "DOUBLE PRECISION" if engine == Dialect.POSTGRES else "DOUBLE"
            views += [
                # Over an approximate argument the result decimals are not
                # fixed on any engine, so nesting is supported everywhere.
                (
                    f"nestedcast_{k}",
                    f"SELECT g, {var_fn}(CAST(x AS {double})) * 2 AS v2, "
                    f"COALESCE({std_fn}(CAST(x AS {double})), -1) AS sc FROM {t} GROUP BY g",
                ),
            ]
            if engine == Dialect.POSTGRES:
                views.append((
                    f"filter_distinct_{k}",
                    f"SELECT g, {var_fn}(x) FILTER (WHERE x > 0) AS vf, {std_fn}(DISTINCT x) AS sd "
                    f"FROM {t} GROUP BY g",
                ))
        return views

    def _assert_equivalent(self, engine, conn, mv: str, live: str, label: str) -> None:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {mv}")
        mv_rows = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
        mv_desc = _description(cur)
        cur.execute(f"SELECT * FROM {live}")
        live_rows = sorted((tuple(_exact(v) for v in r) for r in cur.fetchall()), key=repr)
        live_desc = _description(cur)
        conn.commit()
        self.assertEqual(mv_rows, live_rows, f"{label}: materialized rows != native view rows")
        self.assertEqual(_metadata_mismatches(engine, mv_desc, live_desc), [], f"{label}: result metadata")

    def _run(self, engine: Dialect, dsn: str, col_type: str) -> None:
        conn = _connect(engine, dsn)
        version = self._server(engine, conn)
        is_mariadb = "mariadb" in version.lower()
        cur = conn.cursor()
        tag = uuid.uuid4().hex[:8]
        schema = f"sa_{tag}"
        t = f"t_{tag}"
        try:
            if engine == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{schema}"')
                cur.execute(f'SET search_path TO "{schema}"')
            suffix = " ENGINE=InnoDB" if engine == Dialect.MYSQL else ""
            cur.execute(f"CREATE TABLE {t} (id INT PRIMARY KEY, g INT NOT NULL, x {col_type}){suffix}")
            cur.execute(f"CREATE TABLE {t}_d (g INT PRIMARY KEY, label VARCHAR(8) NOT NULL){suffix}")
            seed = _INT_SEED if "INT" in col_type.upper() else _SEED
            cur.executemany(f"INSERT INTO {t} (id, g, x) VALUES (%s, %s, %s)", seed)
            cur.executemany(
                f"INSERT INTO {t}_d (g, label) VALUES (%s, %s)",
                [(1, "a"), (2, "b"), (3, "a"), (4, "c"), (5, "c"), (6, "d")],
            )
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=engine)
            pairs: List[Tuple[str, str, str]] = []
            for name, select in self._views(engine, t):
                mv, live = f"mv_{name}_{tag}", f"live_{name}_{tag}"
                cur.execute(f"CREATE VIEW {live} AS {select}")
                conn.commit()
                view_sql = f"CREATE VIEW {mv} AS {select};"
                expected = set()
                if engine == Dialect.MYSQL:
                    # MySQL has no TRUNCATE trigger.
                    expected.add("MYSQL_NO_TRUNCATE_TRIGGER")
                    if (
                        is_mariadb
                        and name.startswith("nested_")
                        and "DOUBLE" not in col_type.upper()
                    ):
                        # MariaDB fixes the decimals of VAR/STDDEV over exact
                        # inputs; nested in an expression they cannot be kept.
                        expected.add("MARIADB_STAT_RESULT_DECIMALS")
                if expected:
                    # The default refuses with exactly the expected codes; the
                    # override must warn with exactly those codes.
                    with self.assertRaises(EngineLimitationError) as raised:
                        manager.register_view(view_sql)
                    self.assertEqual({lim.code for lim in raised.exception.limitations}, expected, name)
                    conn.rollback()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        manager.register_view(view_sql, ignore_limitations=True)
                    lim_warnings = [w for w in caught if issubclass(w.category, EngineLimitationWarning)]
                    self.assertEqual(len(lim_warnings), 1)
                    for code in expected:
                        self.assertIn(code, str(lim_warnings[0].message))
                    if "MARIADB_STAT_RESULT_DECIMALS" in expected:
                        # Overridden known divergence: not compared.
                        conn.commit()
                        continue
                else:
                    manager.register_view(view_sql)
                conn.commit()
                pairs.append((name, mv, live))
                self.assertEqual(
                    [r[1:] for r in _catalog_types(engine, cur, mv)],
                    [r[1:] for r in _catalog_types(engine, cur, live)],
                    f"{version} {col_type} {name}: catalog column types",
                )
                conn.commit()

            def check(step: str) -> None:
                for name, mv, live in pairs:
                    self._assert_equivalent(engine, conn, mv, live, f"{version} {col_type} {name} {step}")

            check("initial")
            for step, stmt in _DML:
                cur.execute(stmt.format(t=t))
                conn.commit()
                check(step)
            # A rolled-back change must leave the materialized result unchanged.
            cur.execute(f"UPDATE {t} SET x = x + 100, g = 1 WHERE id IN (3, 4)")
            cur.execute(f"DELETE FROM {t} WHERE id = 7")
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
                for name, kind in objs:
                    if kind == "VIEW":
                        cur.execute(f"DROP VIEW IF EXISTS `{name}`")
                for name, kind in objs:
                    if kind != "VIEW":
                        cur.execute(f"DROP TABLE IF EXISTS `{name}`")
            conn.commit()
            conn.close()

    def test_native_values_and_result_types(self) -> None:
        engines = _native_engines()
        if not engines:
            self.skipTest("set POSTGRES_DSN and/or MYSQL_DSN for native statistical aggregate tests")
        for engine, dsn in engines:
            types = (
                ["INTEGER", "DOUBLE PRECISION", "NUMERIC(12,3)"]
                if engine == Dialect.POSTGRES
                else ["INT", "DOUBLE", "DECIMAL(12,3)"]
            )
            for col_type in types:
                with self.subTest(engine=engine.value, type=col_type):
                    self._run(engine, dsn, col_type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
