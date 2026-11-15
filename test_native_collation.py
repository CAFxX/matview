#!/usr/bin/env python3
"""Native-engine collation equivalence regressions for incremental MVs."""

from __future__ import annotations

import os
import uuid
from collections import Counter
from urllib.parse import urlparse

from matview_sync import Dialect, MaterializedViewManager


def _connect(engine: str, dsn: str):
    if engine == "sqlite":
        import sqlite3
        return sqlite3.connect(dsn or ":memory:")
    if engine == "postgres":
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
        charset="utf8mb4",
    )


def run(engine: str, dsn: str) -> None:
    engine = engine.strip().lower()
    if engine not in {"postgres", "mysql", "mariadb", "sqlite"}:
        raise ValueError(f"Unsupported native collation test engine: {engine}")
    if engine == "postgres":
        dialect = Dialect.POSTGRES
    elif engine == "sqlite":
        dialect = Dialect.SQLITE
    else:
        dialect = Dialect.MYSQL
    suffix = uuid.uuid4().hex[:12]
    table = f"collation_rows_{suffix}"
    view = f"collation_totals_{suffix}"
    schema = f"mv_collation_{suffix}"
    conn = _connect(engine, dsn)
    manager = None
    try:
        cur = conn.cursor()
        if engine == "postgres":
            cur.execute(f'CREATE SCHEMA "{schema}";')
            cur.execute(f'SET search_path TO "{schema}";')
            cur.execute(
                f'CREATE TABLE "{table}" ('
                'id BIGINT PRIMARY KEY, label TEXT COLLATE "C" NOT NULL, '
                'amount BIGINT NOT NULL);'
            )
        elif engine == "sqlite":
            cur.execute(
                f'CREATE TABLE "{table}" ('
                'id INTEGER PRIMARY KEY, label TEXT COLLATE BINARY NOT NULL, '
                'amount INTEGER NOT NULL);'
            )
        else:
            cur.execute(
                f"CREATE TABLE `{table}` ("
                "id BIGINT PRIMARY KEY, "
                "label VARCHAR(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL, "
                "amount BIGINT NOT NULL) ENGINE=InnoDB;"
            )
        param_marker = "?" if engine == "sqlite" else "%s"
        cur.executemany(
            f"INSERT INTO {table} (id, label, amount) VALUES ({param_marker}, {param_marker}, {param_marker})",
            [(1, "a", 1), (2, "A", 10), (3, "á", 100), (4, "Á", 1000)],
        )
        conn.commit()

        manager = MaterializedViewManager(
            conn,
            dialect=dialect,
            ignore_limitations=(dialect == Dialect.MYSQL),
        )
        select_sql = (
            f"SELECT label, COUNT(*) AS n, SUM(amount) AS total "
            f"FROM {table} GROUP BY label"
        )
        plan = manager.register_view(
            f"CREATE VIEW {view} AS {select_sql};",
            ignore_limitations=(dialect == Dialect.MYSQL),
        )

        def assert_equivalent(stage: str, expected_groups: int) -> None:
            cur.execute(f"SELECT label, n, total FROM {plan.view_name};")
            actual = Counter(tuple(row) for row in cur.fetchall())
            cur.execute(select_sql)
            expected = Counter(tuple(row) for row in cur.fetchall())
            if actual != expected:
                raise AssertionError(
                    f"{engine} {stage}: materialized rows differ from live SELECT; "
                    f"actual={actual!r}, expected={expected!r}"
                )
            if len(expected) != expected_groups:
                raise AssertionError(
                    f"{engine} {stage}: expected {expected_groups} collation groups, "
                    f"got {len(expected)}: {expected!r}"
                )

        assert_equivalent("initial", 4)
        mutations = (
            ("insert case-variant", f"INSERT INTO {{table}} (id, label, amount) VALUES ({param_marker}, {param_marker}, {param_marker})", (5, "a", 5), 4),
            ("merge case variants", f"UPDATE {{table}} SET label = {param_marker}, amount = {param_marker} WHERE id = {param_marker}", ("a", 11, 2), 3),
            ("delete accent group", f"DELETE FROM {{table}} WHERE id = {param_marker}", (3,), 2),
            ("change remaining accent case", f"UPDATE {{table}} SET label = {param_marker} WHERE id = {param_marker}", ("á", 4), 2),
        )
        for stage, statement, params, group_count in mutations:
            cur.execute(statement.format(table=table), params)
            conn.commit()
            assert_equivalent(stage, group_count)

        manager.drop_view(plan.view_name)
        manager = None
        conn.commit()
        print(
            f"[NATIVE COLLATION VERIFIED] engine={engine} "
            "case-sensitive grouping preserved across INSERT/UPDATE/DELETE"
        )
    finally:
        try:
            if manager is not None:
                manager.drop_view(view)
                conn.commit()
        except Exception:
            conn.rollback()
        try:
            cur = conn.cursor()
            if engine == "postgres":
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
            elif engine == "sqlite":
                cur.execute(f'DROP VIEW IF EXISTS "{view}";')
                cur.execute(f'DROP TABLE IF EXISTS "{table}";')
            else:
                cur.execute(f"DROP VIEW IF EXISTS `{view}`;")
                cur.execute(f"DROP TABLE IF EXISTS `{table}`;")
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    engine = os.environ.get("MATVIEW_NATIVE_ENGINE", "sqlite")
    dsn = os.environ.get("MATVIEW_NATIVE_DSN", ":memory:")
    run(engine, dsn)
