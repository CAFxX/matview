#!/usr/bin/env python3
"""Differentially exercise generated SELECTs as maintained materialized views across native engines."""

from __future__ import annotations

import os
import re
import sqlite3
import unittest
import urllib.parse
import uuid
import warnings
from typing import Any, Optional, Sequence

from matview_sync import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineLimitationError,
    EngineLimitationWarning,
    EngineSubdialect,
    MaterializedViewManager,
    detect_engine_version,
    generate_materialized_view,
)
from test_matview_sync import (
    DifferentialSQLFuzzer,
    NumericTolerance,
    VariableSchemaDifferentialFuzzer,
    _query_column_tolerances,
    _register_sqlite_compat_udfs,
    _rows_equal,
)
from test_query_ast_fuzzer import GeneratedQuery, SchemaAwareSelectFuzzer


def _dbapi_description_field(column_descriptor: Any, field_name: str, index: int) -> Any:
    """Extract a named or indexed field from a DB-API cursor.description column tuple/object."""
    if hasattr(column_descriptor, field_name):
        return getattr(column_descriptor, field_name)
    if isinstance(column_descriptor, (tuple, list)) and len(column_descriptor) > index:
        return column_descriptor[index]
    return None


def _is_expression_column_name(name: str) -> bool:
    """Check whether a column name represents an expression or compiler-assigned auto-alias."""
    return any(ch in name for ch in "()*,+-/% ") or name.lower().startswith("col_") or name.lower().startswith("__matview_")


def _result_description_mismatches(
    mv_description: Optional[Sequence[Any]],
    live_description: Optional[Sequence[Any]],
) -> list[str]:
    """Compare result column metadata and types where exposed by DB-API cursor.description.

    SQLite's DB-API driver does not expose result type codes (reporting None), so unavailable
    type codes are deliberately skipped. When both sides expose a type code or field, however,
    it must match. Column count must always match exactly.
    """
    if mv_description is None or live_description is None:
        return []
    if len(mv_description) != len(live_description):
        return [
            f"column count differs (materialized={len(mv_description)}, live={len(live_description)})"
        ]

    mismatches: list[str] = []
    # Compare semantic identity/type/scale fields; exclude PEP 249 `precision`
    # (index 4) because MySQL/PyMySQL populates it from wire-protocol display width
    # (e.g. BIGINT=20 vs COUNT(*)=21, DOUBLE=22 vs SUM/AVG=23) or widened DECIMAL(65, s).
    fields = (
        ("name", 0),
        ("type_code", 1),
        ("scale", 5),
    )
    for index, (mv_column, live_column) in enumerate(
        zip(mv_description, live_description)
    ):
        for field_name, tuple_index in fields:
            mv_value = _dbapi_description_field(mv_column, field_name, tuple_index)
            live_value = _dbapi_description_field(live_column, field_name, tuple_index)
            if mv_value is None or live_value is None:
                continue

            if field_name == "name":
                mv_s = str(mv_value).strip().lower()
                live_s = str(live_value).strip().lower()
                if mv_s != live_s:
                    if not (_is_expression_column_name(mv_s) or _is_expression_column_name(live_s)):
                        mismatches.append(
                            f"column {index} {field_name} differs (materialized={mv_value!r}, live={live_value!r})"
                        )
            elif mv_value != live_value:
                mismatches.append(
                    f"column {index} {field_name} differs (materialized={mv_value!r}, live={live_value!r})"
                )
    return mismatches


def _assert_rows_match(
    actual: Sequence[Sequence[Any]],
    expected: Sequence[Sequence[Any]],
    select_sql: str,
    context: str = "",
) -> None:
    """Compare materialized view rows against live query rows using AST-derived numeric tolerances."""
    column_policies = _query_column_tolerances(select_sql)
    if not _rows_equal(actual, expected, column_tolerances=column_policies):
        raise AssertionError(
            f"[{context}] Materialized view rows do not match live query!\n"
            f"  Query: {select_sql}\n"
            f"  Actual MV ({len(actual)} rows):   {list(actual)[:10]}\n"
            f"  Expected Live ({len(expected)} rows): {list(expected)[:10]}"
        )


class QueryASTMaterializedViewTests(unittest.TestCase):
    def test_generated_selects_match_materialized_views_through_dml(self) -> None:
        """Native SQLite differential execution for recursive query AST materialization."""
        supported = 0
        explicitly_limited = 0
        dml_transitions = 0
        observed_features: set[str] = set()
        ast_signatures: set[tuple[Any, ...]] = set()

        for schema_seed in range(8):
            schema, _, seed_sql, mutations, _ = (
                DifferentialSQLFuzzer.generate_variable_schema_scenario(
                    schema_seed, scenario_idx=1, dialect=Dialect.SQLITE
                )
            )
            for query_index in range(4):
                conn = sqlite3.connect(":memory:")
                try:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    _register_sqlite_compat_udfs(conn)
                    conn.executescript(schema)
                    for statement in seed_sql:
                        conn.execute(statement)
                    conn.commit()

                    generated = SchemaAwareSelectFuzzer(
                        schema_seed * 1009 + query_index
                    ).generate(conn, dialect=Dialect.SQLITE)
                    observed_features.update(generated.features)
                    ast_signatures.add(generated.shape)
                    view_name = f"ast_mv_{schema_seed}_{query_index}"
                    view_sql = f"CREATE VIEW {view_name} AS {generated.sql};"
                    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)

                    try:
                        plan = manager.register_view(view_sql, ignore_limitations=False)
                    except EngineLimitationError as exc:
                        codes = {lim.code for lim in exc.limitations}
                        self.assertTrue(
                            codes and all(codes),
                            f"unsupported generated plan lacked stable limitation codes: {exc}",
                        )
                        explicitly_limited += 1
                        continue

                    supported += 1

                    def assert_matches(stage: str) -> None:
                        c_mv = conn.execute(f"SELECT * FROM {plan.db_view_name}")
                        mv_desc = c_mv.description
                        actual = c_mv.fetchall()

                        c_live = conn.execute(generated.sql)
                        live_desc = c_live.description
                        expected = c_live.fetchall()

                        metadata_diffs = _result_description_mismatches(mv_desc, live_desc)
                        self.assertFalse(
                            metadata_diffs,
                            f"metadata mismatch at {stage}: {'; '.join(metadata_diffs)}; "
                            f"seed={schema_seed}, query={query_index}, SQL={generated.sql!r}",
                        )
                        _assert_rows_match(
                            actual,
                            expected,
                            generated.sql,
                            context=f"sqlite:{schema_seed}:{query_index}:{stage}",
                        )

                    assert_matches("initial population")
                    for mutation_index, (_, statement) in enumerate(mutations):
                        conn.execute(statement)
                        conn.commit()
                        assert_matches(f"mutation {mutation_index}: {statement}")
                        dml_transitions += 1
                finally:
                    conn.close()

        self.assertGreater(supported, 0, "no generated SELECT was tested as a materialized view")
        self.assertEqual(supported + explicitly_limited, 8 * 4)
        self.assertTrue(observed_features)
        self.assertTrue(ast_signatures)
        print(
            f"[NATIVE ENGINE: SQLITE] supported={supported}, explicitly_limited={explicitly_limited}, "
            f"ast_signatures={len(ast_signatures)}, dml_transitions={dml_transitions}, "
            f"features={','.join(sorted(observed_features))}"
        )

    def _connect_native(self, dialect: Dialect, dsn: str):
        """Open a live connection to PostgreSQL or MySQL with autocommit and standard settings."""
        parsed = urllib.parse.urlparse(dsn if "://" in dsn else f"{dialect.value}://{dsn}")
        if dialect == Dialect.POSTGRES:
            try:
                import psycopg2
                conn = psycopg2.connect(dsn)
            except ImportError:
                import psycopg
                conn = psycopg.connect(dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = OFF;")
            return conn

        import pymysql
        conn = pymysql.connect(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 3306,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") or None,
            autocommit=True,
        )
        with conn.cursor() as cur:
            cur.execute("SET SESSION collation_connection = @@collation_database;")
            cur.execute(
                "SET SESSION sql_mode = CONCAT_WS(',', REPLACE(@@SESSION.sql_mode, 'ONLY_FULL_GROUP_BY', ''), 'PIPES_AS_CONCAT', 'ANSI_QUOTES');"
            )
        return conn

    def _run_native_query_ast_suite(self, dialect: Dialect, dsn: str) -> None:
        """Exercise generated variable schemas and recursive SELECT ASTs on a live database."""
        conn = self._connect_native(dialect, dsn)
        version_info = detect_engine_version(conn, hint_dialect=dialect)
        engine_label = f"native {version_info.label} ({version_info.raw_version})"

        suffix = uuid.uuid4().hex[:8]
        pg_root_schema = f"ast_pg_{suffix}"
        supported = 0
        limitation_rejected = 0
        dml_transitions = 0
        observed_features: set[str] = set()
        ast_signatures: set[tuple[Any, ...]] = set()

        num_schemas = 4
        num_queries = 2
        mutations_count = 8

        try:
            cur = conn.cursor()
            orig_mysql_db: Optional[str] = None
            if dialect == Dialect.POSTGRES:
                cur.execute(f'CREATE SCHEMA "{pg_root_schema}";')
                cur.execute(f'SET search_path TO "{pg_root_schema}";')
            elif dialect == Dialect.MYSQL:
                cur.execute("SELECT DATABASE();")
                row = cur.fetchone()
                orig_mysql_db = str(row[0]) if row and row[0] else None

            for schema_seed in range(num_schemas):
                schema, _, seed_sql, mutations, _ = (
                    DifferentialSQLFuzzer.generate_variable_schema_scenario(
                        seed=schema_seed,
                        scenario_idx=schema_seed + 1,
                        dialect=dialect,
                        mutations_count=mutations_count,
                    )
                )

                for query_index in range(num_queries):
                    query_pg_schema = f"{pg_root_schema}_{schema_seed}_{query_index}"
                    query_mysql_db = f"ast_mysql_{suffix}_{schema_seed}_{query_index}"
                    manager: Optional[MaterializedViewManager] = None
                    plan: Any = None

                    try:
                        if dialect == Dialect.POSTGRES:
                            cur.execute(f'CREATE SCHEMA "{query_pg_schema}";')
                            cur.execute(f'SET search_path TO "{query_pg_schema}";')
                        elif dialect == Dialect.MYSQL:
                            cur.execute(
                                f"CREATE DATABASE `{query_mysql_db}` "
                                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
                            )
                            cur.execute(f"USE `{query_mysql_db}`;")

                        # Execute base table DDL and seed statements
                        for stmt in [s.strip() for s in schema.split(";") if s.strip()]:
                            cur.execute(stmt + ";")
                        for stmt in seed_sql:
                            cur.execute(stmt)

                        generated = SchemaAwareSelectFuzzer(
                            schema_seed * 1009 + query_index
                        ).generate(conn, dialect=dialect)
                        observed_features.update(generated.features)
                        ast_signatures.add(generated.shape)

                        view_name = f"ast_mv_{schema_seed}_{query_index}"
                        view_sql = f"CREATE VIEW {view_name} AS {generated.sql};"
                        manager = MaterializedViewManager(conn, dialect=dialect)

                        # Step 1: Attempt compilation with default limitation enforcement
                        try:
                            plan = manager.register_view(view_sql, ignore_limitations=False)
                        except EngineLimitationError as exc:
                            codes = {lim.code for lim in exc.limitations}
                            self.assertTrue(
                                codes and all(codes),
                                f"unsupported {dialect.value} plan lacked stable limitation codes: {exc}",
                            )
                            # On MySQL-family engines: if the only limitations are base-table trigger limitations
                            # (MYSQL_NO_TRUNCATE_TRIGGER, MYSQL_FK_CASCADE_NO_TRIGGER), verify default rejection
                            # first, then recompile with ignore_limitations=True and assert EngineLimitationWarning.
                            if dialect == Dialect.MYSQL and codes.issubset({
                                "MYSQL_NO_TRUNCATE_TRIGGER", "MYSQL_FK_CASCADE_NO_TRIGGER"
                            }):
                                with warnings.catch_warnings(record=True) as emitted:
                                    warnings.simplefilter("always", EngineLimitationWarning)
                                    plan = manager.register_view(view_sql, ignore_limitations=True)
                                lim_warnings = [
                                    w for w in emitted if issubclass(w.category, EngineLimitationWarning)
                                ]
                                self.assertGreaterEqual(
                                    len(lim_warnings), 1,
                                    f"MySQL limitation override for {codes} did not emit EngineLimitationWarning",
                                )
                                warn_text = " ".join(str(w.message) for w in lim_warnings)
                                for code in codes:
                                    self.assertIn(code, warn_text)
                            else:
                                limitation_rejected += 1
                                continue

                        supported += 1

                        def assert_matches(stage: str) -> None:
                            cur.execute(f"SELECT * FROM {plan.db_view_name}")
                            mv_desc = cur.description
                            actual = cur.fetchall()

                            cur.execute(generated.sql)
                            live_desc = cur.description
                            expected = cur.fetchall()

                            metadata_diffs = _result_description_mismatches(mv_desc, live_desc)
                            self.assertFalse(
                                metadata_diffs,
                                f"[{dialect.value}] metadata mismatch at {stage}: {'; '.join(metadata_diffs)}; "
                                f"seed={schema_seed}, query={query_index}, SQL={generated.sql!r}",
                            )
                            _assert_rows_match(
                                actual,
                                expected,
                                generated.sql,
                                context=f"{dialect.value}:{schema_seed}:{query_index}:{stage}",
                            )

                        assert_matches("initial population")
                        for mutation_index, (_, statement) in enumerate(mutations):
                            cur.execute(statement)
                            assert_matches(f"mutation {mutation_index}: {statement}")
                            dml_transitions += 1

                    finally:
                        if dialect == Dialect.POSTGRES:
                            cur.execute(f'SET search_path TO "{pg_root_schema}";')
                            cur.execute(f'DROP SCHEMA IF EXISTS "{query_pg_schema}" CASCADE;')
                        elif dialect == Dialect.MYSQL:
                            if manager is not None and plan is not None:
                                try:
                                    manager.drop_view(plan.view_name)
                                except Exception:
                                    pass
                            try:
                                if orig_mysql_db:
                                    cur.execute(f"USE `{orig_mysql_db}`;")
                                cur.execute(f"DROP DATABASE IF EXISTS `{query_mysql_db}`;")
                            except Exception:
                                pass

        finally:
            try:
                cur = conn.cursor()
                if dialect == Dialect.POSTGRES:
                    cur.execute("SET search_path TO public;")
                    cur.execute(f'DROP SCHEMA IF EXISTS "{pg_root_schema}" CASCADE;')
                elif dialect == Dialect.MYSQL and orig_mysql_db:
                    cur.execute(f"USE `{orig_mysql_db}`;")
            except Exception:
                pass
            conn.close()

        self.assertGreater(supported, 0, f"no generated SELECT was tested as a materialized view on {dialect.value}")
        self.assertEqual(supported + limitation_rejected, num_schemas * num_queries)
        self.assertTrue(ast_signatures)
        print(
            f"[NATIVE ENGINE: {dialect.value.upper()}] engine={engine_label}, "
            f"supported={supported}, limitation_rejected={limitation_rejected}, "
            f"ast_signatures={len(ast_signatures)}, dml_transitions={dml_transitions}, "
            f"features={','.join(sorted(observed_features))}"
        )

    def test_native_postgres_generated_selects_match_materialized_views_through_dml(self) -> None:
        """Native PostgreSQL differential execution for recursive query AST materialization."""
        dsn = os.environ.get("POSTGRES_DSN")
        if not dsn:
            self.skipTest("POSTGRES_DSN is not set; native PostgreSQL test only")
        self._run_native_query_ast_suite(Dialect.POSTGRES, dsn)

    def test_native_mysql_generated_selects_match_materialized_views_through_dml(self) -> None:
        """Native MySQL/MariaDB differential execution for recursive query AST materialization."""
        dsn = os.environ.get("MYSQL_DSN")
        if not dsn:
            self.skipTest("MYSQL_DSN is not set; native MySQL/MariaDB test only")
        self._run_native_query_ast_suite(Dialect.MYSQL, dsn)

    def test_deterministic_variable_schema_scenario_generation_across_dialects(self) -> None:
        """Verify scenario generation and compilation across SQLite, PostgreSQL, and MySQL dialects."""
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            schema, _, seed_sql, mutations, shape = (
                DifferentialSQLFuzzer.generate_variable_schema_scenario(
                    seed=42, scenario_idx=1, dialect=dialect, mutations_count=6
                )
            )
            self.assertTrue(schema)
            self.assertTrue(seed_sql)
            self.assertGreater(len(mutations), 0)
            self.assertIn("topology", shape)

            # Check dialect-appropriate column types and upsert syntax
            if dialect == Dialect.POSTGRES:
                self.assertTrue(any(t in schema for t in ["DOUBLE PRECISION", "NUMERIC", "INTEGER", "BIGINT"]))
                self.assertTrue(any("ON CONFLICT" in m[1] for m in mutations))
            elif dialect == Dialect.MYSQL:
                self.assertTrue(any(t in schema for t in ["DOUBLE", "DECIMAL", "INTEGER", "BIGINT"]))
                self.assertTrue(any("ON DUPLICATE KEY UPDATE" in m[1] for m in mutations))
            else:
                self.assertTrue(any("ON CONFLICT" in m[1] for m in mutations))

            # Verify SchemaAwareSelectFuzzer generates valid SQL for this dialect
            fuzzer = SchemaAwareSelectFuzzer(12345)
            query = fuzzer.generate(
                tables=["vs1_t0", "vs1_t1"],
                foreign_keys=[("vs1_t1", "parent_0_id", "vs1_t0")],
                dialect=dialect,
            )
            self.assertTrue(query.sql.startswith("SELECT") or query.sql.startswith("WITH"))
            self.assertTrue(query.features)
            self.assertTrue(query.shape)
            if dialect == Dialect.MYSQL:
                self.assertIn("`vs1_t", query.sql)
            else:
                self.assertIn('"vs1_t', query.sql)

    def test_deterministic_limitation_enforcement_and_override_contract(self) -> None:
        """Verify default limitation rejection and explicit warning emission on MySQL-family overrides."""
        schema_my, _, _, _, _ = (
            DifferentialSQLFuzzer.generate_variable_schema_scenario(
                seed=0, scenario_idx=1, dialect=Dialect.MYSQL
            )
        )
        fuzzer = SchemaAwareSelectFuzzer(0)
        query = fuzzer.generate(
            tables=["vs1_t0", "vs1_t1", "vs1_t2"],
            foreign_keys=[("vs1_t1", "parent_0_id", "vs1_t0"), ("vs1_t2", "ref_1_id", "vs1_t1")],
            dialect=Dialect.MYSQL,
        )
        view_sql = f"CREATE VIEW v_test_lim AS {query.sql};"

        # Default compilation must reject base-table trigger limitations
        with self.assertRaises(EngineLimitationError) as ctx:
            generate_materialized_view(
                schema_my, view_sql, dialect=Dialect.MYSQL, ignore_limitations=False
            )
        codes = {lim.code for lim in ctx.exception.limitations}
        self.assertTrue(codes)
        self.assertTrue(codes.issubset({"MYSQL_NO_TRUNCATE_TRIGGER", "MYSQL_FK_CASCADE_NO_TRIGGER"}))

        # Explicit override must emit EngineLimitationWarning with the codes
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always", EngineLimitationWarning)
            plan = generate_materialized_view(
                schema_my, view_sql, dialect=Dialect.MYSQL, ignore_limitations=True
            )
            self.assertIsNotNone(plan)
            lim_warnings = [w for w in emitted if issubclass(w.category, EngineLimitationWarning)]
            self.assertGreaterEqual(len(lim_warnings), 1)
            warn_msg = " ".join(str(w.message) for w in lim_warnings)
            for code in codes:
                self.assertIn(code, warn_msg)

    def test_deterministic_result_description_and_row_matching_oracle(self) -> None:
        """Verify DB-API description matching and exact/tolerant row multiset verification."""
        # 1. Matching columns with exact metadata
        desc_a = (("id", 23, None, None, 10, 0, None), ("val", 701, None, None, None, None, None))
        desc_b = (("id", 23, None, None, 10, 0, None), ("val", 701, None, None, None, None, None))
        self.assertEqual(_result_description_mismatches(desc_a, desc_b), [])

        # 2. SQLite-style None type codes pass when column count matches
        desc_lite_a = (("id", None, None, None, None, None, None),)
        desc_lite_b = (("id", None, None, None, None, None, None),)
        self.assertEqual(_result_description_mismatches(desc_lite_a, desc_lite_b), [])

        # 3. Mismatches detected
        desc_diff_count = (("id", 23, None, None, 10, 0, None),)
        self.assertTrue(_result_description_mismatches(desc_a, desc_diff_count))

        desc_diff_type = (("id", 23, None, None, 10, 0, None), ("val", 1700, None, None, None, None, None))
        self.assertTrue(_result_description_mismatches(desc_a, desc_diff_type))

        desc_diff_name = (("other_id", 23, None, None, 10, 0, None), ("val", 701, None, None, None, None, None))
        self.assertTrue(_result_description_mismatches(desc_a, desc_diff_name))

        # 4. Row matching oracle: exact integers
        _assert_rows_match([(1, 10), (2, 20)], [(2, 20), (1, 10)], "SELECT id, qty FROM t")
        with self.assertRaises(AssertionError):
            _assert_rows_match([(1, 10)], [(1, 11)], "SELECT id, qty FROM t")

        # 5. Row matching oracle: tolerant floating aggregates
        _assert_rows_match(
            [(1, 10.000000000001)], [(1, 10.0)],
            "SELECT id, AVG(amount) FROM t GROUP BY id",
        )

    def test_deterministic_schema_introspection_across_dialects(self) -> None:
        """Verify SchemaAwareSelectFuzzer introspection and AST generation across simulated schemas."""
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE vs1_t0 (id INTEGER PRIMARY KEY, label TEXT, qty INTEGER, amount REAL);")
            conn.execute(
                "CREATE TABLE vs1_t1 (id INTEGER PRIMARY KEY, parent_0_id INTEGER REFERENCES vs1_t0(id), "
                "label TEXT, qty INTEGER, amount REAL);"
            )
            tables, fks = SchemaAwareSelectFuzzer._schema(conn)
            self.assertEqual(tables, ["vs1_t0", "vs1_t1"])
            self.assertEqual(len(fks), 1)
            self.assertEqual(fks[0][0], "vs1_t1")
            self.assertEqual(fks[0][1], "parent_0_id")
            self.assertEqual(fks[0][2], "vs1_t0")

            query = SchemaAwareSelectFuzzer(777).generate(conn)
            self.assertTrue(query.sql)
            self.assertTrue(query.features)
        finally:
            conn.close()

    def test_schema_introspection_filters_views_and_grouped_order_by(self) -> None:
        """Ensure _schema excludes views/LIKE wildcards and grouped ORDER BY avoids ungrouped id."""
        for dialect, version_str in (
            (Dialect.POSTGRES, "PostgreSQL 18.1"),
            (Dialect.MYSQL, "8.4.4-MySQL"),
        ):
            executed: list[str] = []

            class _FakeCursor:
                def execute(self, sql: str) -> None:
                    executed.append(sql)
                    self._last = sql

                def fetchall(self):
                    sql_up = self._last.upper()
                    if "VERSION()" in sql_up:
                        return [(version_str,)]
                    if "INFORMATION_SCHEMA.TABLES" in sql_up:
                        return [("vs1_t0",), ("vs1_t1",)]
                    return []

                def fetchone(self):
                    rows = self.fetchall()
                    return rows[0] if rows else None

                def close(self) -> None:
                    pass

            class _FakeConn:
                def cursor(self):
                    return _FakeCursor()

            tables, _ = SchemaAwareSelectFuzzer._schema(_FakeConn())
            self.assertEqual(tables, ["vs1_t0", "vs1_t1"])
            table_queries = [q for q in executed if "information_schema.tables" in q.lower()]
            self.assertEqual(len(table_queries), 1, f"expected table lookup query for {dialect}")
            self.assertIn("BASE TABLE", table_queries[0])
            self.assertIn("vs%#_t%", table_queries[0])
            self.assertIn("ESCAPE '#'", table_queries[0])

        for seed in range(64):
            for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
                q = SchemaAwareSelectFuzzer(seed).generate(
                    tables=["vs1_t0", "vs1_t1"],
                    foreign_keys=[("vs1_t1", "parent_0_id", "vs1_t0")],
                    dialect=dialect,
                )
                if "group_by" in q.features:
                    # Top-level grouped query must not reference ungrouped `id` in ORDER BY.
                    order_idx = q.sql.rfind("ORDER BY ")
                    if order_idx != -1:
                        self.assertNotIn("id", q.sql[order_idx:].lower(), q.sql)


if __name__ == "__main__":
    unittest.main(verbosity=2)
