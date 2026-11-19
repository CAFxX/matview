#!/usr/bin/env python3
"""Native result semantics and representation of VAR/STDDEV materializations.

Native tests compare the materialized reader against the original view
executed by the engine -- the only oracle -- for initial population,
INSERT/UPDATE/DELETE (including group moves, singleton, emptied and new
groups) and rollback. Values are compared exactly (floats bit-for-bit,
Decimals including their scale: `0` and `0.0` are different results), and
result metadata deliberately: DB-API description fields and the catalog
column type of the reader view versus a live view of the same SELECT.

`VARIANCE`, `STDDEV` and `STD` are not SQL-standard functions; every engine
defines them itself:

* PostgreSQL: `VARIANCE` = `VAR_SAMP`, `STDDEV` = `STDDEV_SAMP` (sample
  statistics); there is no `STD`.
* MySQL / MariaDB: `VARIANCE` = `VAR_POP`, `STDDEV` = `STD` = `STDDEV_POP`
  (population statistics).
* SQLite: none of them exists (a function of that name can only be an
  application-defined UDF whose semantics the compiler cannot know).

The compiler must bind each spelling to the target engine's definition and
reject spellings the engine does not define.

Native engines are selected with POSTGRES_DSN and/or MYSQL_DSN (MySQL or
MariaDB; the product is read from the server, not trusted from a label).
"""

from __future__ import annotations

import os
import re
import sqlite3
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
    UnsupportedMaterializedViewError,
    generate_materialized_view,
    parse_table_schema,
    parse_view_definition,
)
from matview.compiler import IncrementalViewCompiler
import test_matview_sync as _shared_oracle

_SCHEMA = "CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, x DOUBLE PRECISION);"

# (engine version, alias -> expected AggFunc)
_EXPECTED = {
    "postgres:15": {"VARIANCE": AggFunc.VAR_SAMP, "STDDEV": AggFunc.STDDEV_SAMP},
    "postgres:18": {"VARIANCE": AggFunc.VAR_SAMP, "STDDEV": AggFunc.STDDEV_SAMP},
    "mysql:8.4": {"VARIANCE": AggFunc.VAR_POP, "STDDEV": AggFunc.STDDEV_POP, "STD": AggFunc.STDDEV_POP},
    "mysql:9.7": {"VARIANCE": AggFunc.VAR_POP, "STDDEV": AggFunc.STDDEV_POP, "STD": AggFunc.STDDEV_POP},
    "mariadb:10.11": {"VARIANCE": AggFunc.VAR_POP, "STDDEV": AggFunc.STDDEV_POP, "STD": AggFunc.STDDEV_POP},
    "mariadb:12.3": {"VARIANCE": AggFunc.VAR_POP, "STDDEV": AggFunc.STDDEV_POP, "STD": AggFunc.STDDEV_POP},
}
# Alias spellings each native engine defines (MySQL and MariaDB agree).
_NATIVE_ALIASES = {
    Dialect.POSTGRES: ("VARIANCE", "STDDEV"),
    Dialect.MYSQL: ("VARIANCE", "STDDEV", "STD"),
}
_NATIVE_ALIAS_PAIRS = {
    Dialect.POSTGRES: [("VARIANCE", "STDDEV")],
    Dialect.MYSQL: [("VARIANCE", "STDDEV"), ("VARIANCE", "STD")],
}
_EXPLICIT = {
    "VAR_POP": AggFunc.VAR_POP,
    "VAR_SAMP": AggFunc.VAR_SAMP,
    "STDDEV_POP": AggFunc.STDDEV_POP,
    "STDDEV_SAMP": AggFunc.STDDEV_SAMP,
}


def _compiled_items(view_sql_or_spec: Any, engine: str) -> Dict[str, Optional[AggFunc]]:
    ev_dialect = Dialect.POSTGRES if engine.startswith("postgres") else Dialect.MYSQL
    if engine.startswith("sqlite"):
        ev_dialect = Dialect.SQLITE
    schemas = parse_table_schema(_SCHEMA, dialect=ev_dialect)
    spec = (
        parse_view_definition(view_sql_or_spec, known_tables={"t"}, dialect=ev_dialect)
        if isinstance(view_sql_or_spec, str)
        else view_sql_or_spec
    )
    compiler = IncrementalViewCompiler(schemas, spec, dialect=engine, ignore_limitations=True)
    return {it.alias: it.agg_func for it in compiler.view.select_items}


class AliasResolutionTests(unittest.TestCase):
    def test_aliases_bind_to_each_engines_definition(self) -> None:
        for engine, expected in _EXPECTED.items():
            calls = {**expected, **_EXPLICIT}
            sel = ", ".join(f"{fn}(x) AS a_{fn.lower()}" for fn in calls)
            with self.subTest(engine=engine):
                items = _compiled_items(f"CREATE VIEW v AS SELECT g, {sel} FROM t GROUP BY g;", engine)
                for fn, agg in calls.items():
                    self.assertEqual(items[f"a_{fn.lower()}"], agg, f"{engine} {fn}")

    def test_case_and_spacing_do_not_change_the_binding(self) -> None:
        items = _compiled_items(
            "CREATE VIEW v AS SELECT g, variance (x) AS a, StdDev( x ) AS b FROM t GROUP BY g;",
            "postgres:18",
        )
        self.assertEqual(items["a"], AggFunc.VAR_SAMP)
        self.assertEqual(items["b"], AggFunc.STDDEV_SAMP)

    def test_preparsed_spec_is_rebound_to_the_compile_target(self) -> None:
        # parse_view_definition defaults to the SQLite dialect, where the
        # aliases are undefined; compiling the same spec for another engine
        # must bind the aliases to that engine, not to the parse dialect.
        sql = "CREATE VIEW v AS SELECT g, VARIANCE(x) AS v, STDDEV(x) AS s FROM t GROUP BY g;"
        spec = parse_view_definition(sql, known_tables={"t"})
        for engine, var_f, std_f in (
            ("postgres:18", AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
            ("mysql:8.4", AggFunc.VAR_POP, AggFunc.STDDEV_POP),
            ("mariadb:12.3", AggFunc.VAR_POP, AggFunc.STDDEV_POP),
        ):
            with self.subTest(engine=engine):
                items = _compiled_items(spec, engine)
                self.assertEqual((items["v"], items["s"]), (var_f, std_f))
        mysql_spec = parse_view_definition(sql, known_tables={"t"}, dialect=Dialect.MYSQL)
        items = _compiled_items(mysql_spec, "postgres:15")
        self.assertEqual((items["v"], items["s"]), (AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP))

    def test_aliases_nested_in_expressions_and_having_are_bound(self) -> None:
        sql = (
            "CREATE VIEW v AS SELECT g, VARIANCE(x) * 2 AS v2 FROM t GROUP BY g "
            "HAVING STDDEV(x) > 0;"
        )
        for engine, var_f, std_f in (
            ("postgres:18", AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP),
            ("mysql:8.4", AggFunc.VAR_POP, AggFunc.STDDEV_POP),
        ):
            with self.subTest(engine=engine):
                dialect = Dialect.POSTGRES if engine.startswith("postgres") else Dialect.MYSQL
                compiler = IncrementalViewCompiler(
                    parse_table_schema(_SCHEMA, dialect=dialect),
                    parse_view_definition(sql, known_tables={"t"}, dialect=dialect),
                    dialect=engine,
                    ignore_limitations=True,
                )
                funcs = {it.agg_func for it in compiler._working_select_items if it.is_aggregate}
                self.assertIn(var_f, funcs)
                self.assertIn(std_f, funcs)
                wrong = {AggFunc.VAR_POP, AggFunc.STDDEV_POP} if var_f == AggFunc.VAR_SAMP else {
                    AggFunc.VAR_SAMP, AggFunc.STDDEV_SAMP}
                self.assertFalse(funcs & wrong, funcs)

    def test_undefined_spellings_are_rejected_explicitly(self) -> None:
        cases = [
            ("postgres:18", "SELECT g, STD(x) AS s FROM t GROUP BY g"),
            ("postgres:15", "SELECT g, STD(x) + 1 AS s FROM t GROUP BY g"),
            ("postgres:18", "SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING STD(x) > 0"),
            ("postgres:18", "SELECT id, STD(x) OVER (PARTITION BY g) AS s FROM t"),
            ("sqlite", "SELECT g, VARIANCE(x) AS v FROM t GROUP BY g"),
            ("sqlite", "SELECT g, STDDEV(x) AS v FROM t GROUP BY g"),
            ("sqlite", "SELECT g, STD(x) AS v FROM t GROUP BY g"),
            ("sqlite", "SELECT g, COALESCE(VARIANCE(x), 0) AS v FROM t GROUP BY g"),
            ("sqlite", "SELECT g FROM t WHERE x > (SELECT STDDEV(x) FROM t)"),
        ]
        for engine, select in cases:
            for ignore in (False, True):
                with self.subTest(engine=engine, select=select, ignore_limitations=ignore):
                    dialect = Dialect.SQLITE if engine == "sqlite" else Dialect.POSTGRES
                    with self.assertRaises(UnsupportedMaterializedViewError) as raised:
                        generate_materialized_view(
                            parse_table_schema(_SCHEMA, dialect=dialect),
                            f"CREATE VIEW v AS {select};",
                            dialect=engine,
                            ignore_limitations=ignore,
                        )
                    self.assertIn("VAR_POP/VAR_SAMP/STDDEV_POP/STDDEV_SAMP", str(raised.exception))

    def test_names_that_are_not_alias_calls_are_not_rejected(self) -> None:
        schema = "CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, std DOUBLE PRECISION, variance TEXT);"
        for engine, dialect in (("sqlite", Dialect.SQLITE), ("postgres:18", Dialect.POSTGRES)):
            with self.subTest(engine=engine):
                plan = generate_materialized_view(
                    parse_table_schema(schema, dialect=dialect),
                    "CREATE VIEW v AS SELECT g, SUM(std) AS std, COUNT(variance) AS n, "
                    "MAX('STD(') AS lit FROM t GROUP BY g;",
                    dialect=engine,
                    ignore_limitations=True,
                )
                self.assertTrue(plan.view_name)

    def test_sqlite_manager_rejects_before_creating_objects(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, g INTEGER NOT NULL, x REAL)")
            conn.commit()
            manager = MaterializedViewManager(conn, dialect="sqlite")
            conn.commit()
            before = conn.execute("SELECT type, name FROM sqlite_master ORDER BY 1, 2").fetchall()
            with self.assertRaises(UnsupportedMaterializedViewError):
                manager.register_view("CREATE VIEW v AS SELECT g, VARIANCE(x) AS v FROM t GROUP BY g;")
            conn.rollback()
            after = conn.execute("SELECT type, name FROM sqlite_master ORDER BY 1, 2").fetchall()
            self.assertEqual(before, after)
        finally:
            conn.close()


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

    def test_mariadb_unreproducible_statistics_are_refused(self) -> None:
        """Every MariaDB VAR/STDDEV shape is exact or refused, never bypassable.

        Fixed-decimal results (exact or FLOAT(M,D) arguments) are reproducible
        only as top-level projected aggregates of a single query (reader-side
        ROUND to the native decimals) or inside HAVING; window statistics never.
        """
        cases = [
            ("INT", "SELECT g, VAR_SAMP(x) * 2 AS a FROM t GROUP BY g", True),
            ("DECIMAL(12,3)", "SELECT g, COALESCE(STDDEV_POP(x), -1) AS a FROM t GROUP BY g", True),
            ("FLOAT(7,2)", "SELECT g, VAR_SAMP(x) + 1 AS a FROM t GROUP BY g", True),
            ("INT", "SELECT id, VAR_SAMP(x) OVER (PARTITION BY g) AS a FROM t", True),
            ("DOUBLE", "SELECT id, STDDEV_POP(x) OVER (PARTITION BY g) AS a FROM t", True),
            ("INT", "WITH c AS (SELECT g, VARIANCE(x) AS v FROM t GROUP BY g) SELECT g, v FROM c", True),
            ("INT", "SELECT s.g, s.v FROM (SELECT g, STD(x) AS v FROM t GROUP BY g) s", True),
            ("INT", "SELECT g, VAR_POP(x) AS v FROM (SELECT g, x FROM t UNION ALL SELECT g, x FROM t) u GROUP BY g", True),
            ("INT", "SELECT a.g, VARIANCE(b.x) AS v FROM t a JOIN t b ON b.g = a.g GROUP BY a.g", True),
            ("INT", "SELECT g, VAR_SAMP(x) AS v FROM t GROUP BY g ORDER BY v LIMIT 2", True),
            ("INT", "SELECT g, VAR_SAMP(CAST(x AS DOUBLE)) * 2 AS a FROM t GROUP BY g", False),
            ("DOUBLE", "SELECT g, VAR_SAMP(x) * 2 AS a FROM t GROUP BY g", False),
            ("DOUBLE", "WITH c AS (SELECT g, VARIANCE(x) AS v FROM t GROUP BY g) SELECT g, v FROM c", False),
            ("INT", "SELECT g, VAR_SAMP(x) AS a FROM t GROUP BY g", False),
            ("DECIMAL(12,3)", "SELECT g, STDDEV_POP(x) AS a, VARIANCE(x) AS b FROM t GROUP BY g", False),
            ("INT", "SELECT g, VAR_SAMP(x * 2) AS a FROM t GROUP BY g", False),
            ("INT", "SELECT g, COUNT(*) AS c FROM t GROUP BY g HAVING VAR_SAMP(x) > 1", False),
        ]
        for col_type, select, refused in cases:
            for engine in ("mariadb:10.11", "mariadb:12.3", "mysql:8.4"):
                for ignore in (False, True):
                    with self.subTest(engine=engine, type=col_type, select=select, ignore=ignore):
                        expect = refused and engine.startswith("mariadb")
                        # Without the override every MySQL-family view stops at
                        # MYSQL_NO_TRUNCATE_TRIGGER first; the override must
                        # not get past the refusal.
                        if expect and ignore:
                            with self.assertRaises(UnsupportedMaterializedViewError):
                                self._plan(engine, col_type, select, ignore_limitations=ignore)
                        elif ignore:
                            self._plan(engine, col_type, select, ignore_limitations=True)
                        else:
                            with self.assertRaises(EngineLimitationError) as raised:
                                self._plan(engine, col_type, select, ignore_limitations=False)
                            self.assertEqual(
                                {lim.code for lim in raised.exception.limitations}, {"MYSQL_NO_TRUNCATE_TRIGGER"}
                            )


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
    """DB-API result metadata comparison shared with the main differential oracle.

    See test_matview_sync._result_description_mismatches: type code and scale
    always, precision on PostgreSQL, nullability where the driver reports it,
    minus the documented MYSQL_UNMODELLED_METADATA gaps.
    """
    family = "postgres" if engine == Dialect.POSTGRES else "mysql"
    return _shared_oracle._result_description_mismatches(family, mv, live)


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
        # IS_NULLABLE (r[2]) is a documented unmodelled gap
        # (test_matview_sync.MYSQL_UNMODELLED_METADATA).
        return [(r[0], _normalize_mysql_column_type(r[1])) + tuple(r[3:]) for r in cur.fetchall()]
    return [tuple(r) for r in cur.fetchall()]


def _normalize_mysql_column_type(column_type: str) -> str:
    """Drop display widths from a MySQL/MariaDB COLUMN_TYPE, keep everything else.

    Integer display widths (`bigint(20)` vs `bigint(21)` for COUNT) are
    documented as cosmetic and deprecated, and MariaDB reports a floating
    expression's maximum printed length as `double(M,D)`. The fixed number of
    decimals D of a floating result is kept: MariaDB rounds VAR/STDDEV results
    to it, so it is part of the result. DECIMAL scale,
    signedness and collation are compared unchanged; nullability, DECIMAL
    precision and string lengths are documented unmodelled gaps (above).
    """
    t = re.sub(r"^(tinyint|smallint|mediumint|int|bigint)\(\d+\)", r"\1", column_type)
    t = re.sub(r"^(float|double)\(\d+,(\d+)\)", r"\1(*,\2)", t)
    # Documented unmodelled gaps (test_matview_sync.MYSQL_UNMODELLED_METADATA):
    # DECIMAL precision (the scale is kept) and string lengths.
    t = re.sub(r"^decimal\(\d+,(\d+)\)", r"decimal(*,\1)", t)
    return re.sub(r"^(varchar|char|varbinary|binary)\(\d+\)", r"\1(*)", t)


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
        fns = list(_EXPLICIT) + list(_NATIVE_ALIASES[engine])
        sel = ", ".join(f"{fn}(x) AS s_{fn.lower()}" for fn in fns)
        views = [("grouped", f"SELECT g, COUNT(x) AS n, {sel} FROM {t} GROUP BY g")]
        # Every shape is checked with sample and population functions and with
        # each alias pair the engine defines.
        pairs = [("VAR_SAMP", "STDDEV_SAMP"), ("VAR_POP", "STDDEV_POP")] + _NATIVE_ALIAS_PAIRS[engine]
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
                (
                    f"window_{k}",
                    f"SELECT id, {var_fn}(x) OVER (PARTITION BY g) AS w, "
                    f"{std_fn}(x) OVER (PARTITION BY g) AS sw FROM {t}",
                ),
                # Shape audit: on MariaDB each must be exact or refused.
                (f"audit_cte_{k}", f"WITH c AS (SELECT g, {var_fn}(x) AS v FROM {t} GROUP BY g) SELECT g, v FROM c"),
                (f"audit_derived_{k}", f"SELECT s.g, s.v FROM (SELECT g, {std_fn}(x) AS v FROM {t} GROUP BY g) s"),
                (
                    f"audit_union_{k}",
                    f"SELECT g, {var_fn}(x) AS v FROM (SELECT g, x FROM {t} UNION ALL "
                    f"SELECT g, x FROM {t} WHERE id > 2) u GROUP BY g",
                ),
                (f"audit_orderlimit_{k}", f"SELECT g, {std_fn}(x) AS v FROM {t} GROUP BY g ORDER BY g LIMIT 3"),
            ]
            if engine == Dialect.POSTGRES:
                views.append((
                    f"filter_distinct_{k}",
                    f"SELECT g, {var_fn}(x) FILTER (WHERE x > 0) AS vf, {std_fn}(DISTINCT x) AS sd "
                    f"FROM {t} GROUP BY g",
                ))
        # Self-join statistics over DOUBLE are left out on PostgreSQL: its
        # floating VAR/STDDEV accumulation depends on row order and the join
        # recompute visits rows in another order (1-ULP differences observed);
        # tracked in TODO "Make Floating-Point Aggregates Exactly Equal".
        for var_fn, std_fn in pairs:
            k = f"{var_fn}_{std_fn}".lower()
            views.append((
                f"audit_selfjoin_{k}",
                f"SELECT a.g, {var_fn}(b.x) AS v FROM {t} a JOIN {t} b ON b.g = a.g GROUP BY a.g",
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
            refused: List[str] = []
            for name, select in self._views(engine, t):
                mv, live = f"mv_{name}_{tag}", f"live_{name}_{tag}"
                cur.execute(f"CREATE VIEW {live} AS {select}")
                conn.commit()
                view_sql = f"CREATE VIEW {mv} AS {select};"
                if (
                    engine == Dialect.POSTGRES
                    and name.startswith("audit_selfjoin_")
                    and "DOUBLE" in col_type.upper()
                ):
                    continue
                if is_mariadb and name.startswith("audit_"):
                    # Exact or refused: a refusal must hold under the override.
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", EngineLimitationWarning)
                            manager.register_view(view_sql, ignore_limitations=True)
                    except UnsupportedMaterializedViewError:
                        conn.rollback()
                        refused.append(name)
                        continue
                    conn.commit()
                    pairs.append((name, mv, live))
                    continue
                if is_mariadb and (
                    name.startswith("window_")
                    or (name.startswith("nested_") and "DOUBLE" not in col_type.upper())
                ):
                    # MariaDB statistics it cannot reproduce exactly are refused,
                    # also with ignore_limitations.
                    with self.assertRaises(UnsupportedMaterializedViewError, msg=name):
                        manager.register_view(view_sql, ignore_limitations=True)
                    conn.rollback()
                    continue
                if engine == Dialect.MYSQL:
                    # MySQL has no TRUNCATE trigger: the default refuses, the
                    # override must warn with exactly that limitation.
                    with self.assertRaises(EngineLimitationError) as raised:
                        manager.register_view(view_sql)
                    self.assertEqual({lim.code for lim in raised.exception.limitations}, {"MYSQL_NO_TRUNCATE_TRIGGER"}, name)
                    conn.rollback()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        manager.register_view(view_sql, ignore_limitations=True)
                    lim_warnings = [w for w in caught if issubclass(w.category, EngineLimitationWarning)]
                    self.assertEqual(len(lim_warnings), 1)
                    self.assertIn("MYSQL_NO_TRUNCATE_TRIGGER", str(lim_warnings[0].message))
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


# Every statistic spelling x every wrapper of a second copy of the same call
# next to the projected call itself. On MariaDB a fixed-decimal copy that is
# not itself a whole top-level item must be refused (even under the
# override); everything accepted must be exact. MySQL accepts all of them.
_STAT_NAMES = ("VARIANCE", "VAR_POP", "VAR_SAMP", "STD", "STDDEV", "STDDEV_POP", "STDDEV_SAMP")
_WRAPPERS = {
    "plus0": "{c} + 0",
    "case": "CASE WHEN COUNT(*) > 1 THEN {c} END",
    "coalesce": "COALESCE({c}, 0)",
    "round": "ROUND({c}, 2)",
    "cast": "CAST({c} AS DOUBLE)",
    "arith": "{c} * 2 - 1",
    "funcarg": "ABS({c})",
    "dup": "{c}",
    "lower_space": "{lc} + 0",
    "quoted": "{qc} + 0",
}


def _wrapper_views(t: str) -> List[Tuple[str, str, bool]]:
    """(name, select, is_plain_top_level_duplicate) for every name x wrapper."""
    out = []
    for fn in _STAT_NAMES:
        call = f"{fn}(x)"
        lc = f"{fn.lower()}( x )"
        qc = f"{fn}(`x`)"
        for wname, tmpl in _WRAPPERS.items():
            expr = tmpl.format(c=call, lc=lc, qc=qc)
            out.append((f"{fn.lower()}_{wname}", f"SELECT g, {call} AS top, {expr} AS w FROM {t} GROUP BY g", wname == "dup"))
    return out


class StatisticWrapperRefusalTests(unittest.TestCase):
    """Offline: refusal is decided per call position, never by call text."""

    def test_mariadb_refuses_every_wrapped_copy_of_a_fixed_decimal_call(self) -> None:
        for col_type, fixed in (("INT", True), ("DECIMAL(10,2)", True), ("DOUBLE", False)):
            schema = f"CREATE TABLE t (id INT PRIMARY KEY, g INT NOT NULL, x {col_type});"
            for name, select, plain_dup in _wrapper_views("t"):
                for engine in ("mariadb:10.11", "mysql:8.4"):
                    with self.subTest(type=col_type, view=name, engine=engine):
                        refused = fixed and not plain_dup and engine.startswith("mariadb")
                        compile_ = lambda: generate_materialized_view(
                            parse_table_schema(schema, dialect=Dialect.MYSQL),
                            f"CREATE VIEW v AS {select};",
                            dialect=engine,
                            ignore_limitations=True,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", EngineLimitationWarning)
                            if refused:
                                with self.assertRaises(UnsupportedMaterializedViewError):
                                    compile_()
                            else:
                                compile_()


class NativeStatisticWrapperTests(unittest.TestCase):
    """Native: on MySQL/MariaDB every wrapper view is exact or (MariaDB) refused."""

    def test_wrapped_statistics_are_exact_or_refused(self) -> None:
        engines = [(e, d) for e, d in _native_engines() if e == Dialect.MYSQL]
        if not engines:
            self.skipTest("set MYSQL_DSN for the native statistic wrapper matrix")
        for engine, dsn in engines:
            for col_type in ("INT", "DECIMAL(10,2)", "DOUBLE"):
                with self.subTest(type=col_type):
                    self._run(engine, dsn, col_type)

    def _run(self, engine: Dialect, dsn: str, col_type: str) -> None:
        conn = _connect(engine, dsn)
        cur = conn.cursor()
        cur.execute("SELECT VERSION()")
        is_mariadb = "mariadb" in str(cur.fetchone()[0]).lower()
        conn.commit()
        tag = uuid.uuid4().hex[:8]
        t = f"w_{tag}"
        try:
            cur.execute(f"CREATE TABLE {t} (id INT PRIMARY KEY, g INT NOT NULL, x {col_type}) ENGINE=InnoDB")
            seed = _INT_SEED if "INT" in col_type else _SEED
            cur.executemany(f"INSERT INTO {t} (id, g, x) VALUES (%s, %s, %s)", seed)
            conn.commit()
            manager = MaterializedViewManager(conn, dialect=engine)
            pairs, refused = [], []
            for name, select, plain_dup in _wrapper_views(t):
                mv, live = f"mv_{name}_{tag}", f"lv_{name}_{tag}"
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        manager.register_view(f"CREATE VIEW {mv} AS {select};", ignore_limitations=True)
                except UnsupportedMaterializedViewError:
                    conn.rollback()
                    refused.append(name)
                    continue
                conn.commit()
                cur.execute(f"CREATE VIEW {live} AS {select}")
                conn.commit()
                pairs.append((name, mv, live))
            if not is_mariadb or "DOUBLE" in col_type:
                self.assertEqual(refused, [], "nothing may be refused here")
            else:
                # Only the plain top-level duplicates are reproducible.
                self.assertEqual(sorted(n for n, _, _ in pairs), sorted(f"{fn.lower()}_dup" for fn in _STAT_NAMES))
            checker = NativeStatisticalAggregateTests()
            for step, stmt in [("initial", None)] + list(_DML):
                if stmt:
                    cur.execute(stmt.format(t=t))
                    conn.commit()
                for name, mv, live in pairs:
                    checker._assert_equivalent(engine, conn, mv, live, f"{col_type} {name} {step}")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
