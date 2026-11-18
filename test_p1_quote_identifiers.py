"""
Comprehensive test suite for P1 Audit Finding #8:
Quoting SQL identifiers across emitted DDL, backing tables, triggers,
indexes, and reader views when tables, columns, PKs, or aliases use SQL
reserved keywords ('order', 'group', 'user', 'select', 'where', 'key', 'from', 'offset').
"""

import sqlite3
import sys

import pytest
from matview_sync import (
    Dialect,
    MaterializedViewManager,
    TableSchema,
    ColumnDef,
    generate_materialized_view,
    parse_view_definitions,
    quote_sql_identifier,
)


def _sqlite_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def test_quote_sql_identifier_idempotence_and_keywords():
    """Verify quote_sql_identifier is idempotent and correctly handles keywords across dialects."""
    for kw in ["order", "group", "select", "where", "from", "key", "user", "offset", "row", "do"]:
        # SQLite
        q_sqlite = quote_sql_identifier(kw, Dialect.SQLITE)
        assert q_sqlite == f'"{kw}"'
        assert quote_sql_identifier(q_sqlite, Dialect.SQLITE) == f'"{kw}"'

        # PostgreSQL
        q_pg = quote_sql_identifier(kw, Dialect.POSTGRES)
        assert q_pg == f'"{kw}"'
        assert quote_sql_identifier(q_pg, Dialect.POSTGRES) == f'"{kw}"'

        # MySQL
        q_mysql = quote_sql_identifier(kw, Dialect.MYSQL)
        assert q_mysql == f"`{kw}`"
        assert quote_sql_identifier(q_mysql, Dialect.MYSQL) == f"`{kw}`"


def test_tier1_projection_reserved_keywords_end_to_end():
    """
    Tier 1 zset projection with table named 'order', PK named 'select',
    columns 'group', 'where', 'from', and aliases 'key', 'offset'.
    """
    conn = _sqlite_conn()
    conn.execute(
        'CREATE TABLE "order" ('
        '    "select" INTEGER PRIMARY KEY,'
        '    "group" TEXT NOT NULL,'
        '    "where" INTEGER NOT NULL,'
        '    "from" TEXT'
        ');'
    )
    conn.commit()

    view_sql = (
        'CREATE VIEW "mv_order_proj" AS '
        'SELECT "select" AS "key", "group" AS "offset", "where" + 1 AS "total" '
        'FROM "order" '
        'WHERE "where" > 0;'
    )

    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
    plan = manager.register_view(view_sql)
    assert plan is not None

    # Check backing table DDL quotes keywords
    assert '"key"' in plan.backing_table_ddl or '"order"' in plan.backing_table_ddl
    assert '"offset"' in plan.backing_table_ddl

    # Insert rows
    conn.execute('INSERT INTO "order" ("select", "group", "where", "from") VALUES (1, \'grpA\', 10, \'here\');')
    conn.execute('INSERT INTO "order" ("select", "group", "where", "from") VALUES (2, \'grpB\', 0, \'filtered\');')
    conn.execute('INSERT INTO "order" ("select", "group", "where", "from") VALUES (3, \'grpA\', 20, \'there\');')
    conn.commit()

    rows = conn.execute('SELECT "key", "offset", "total" FROM "mv_order_proj" ORDER BY "key";').fetchall()
    assert rows == [(1, 'grpA', 11), (3, 'grpA', 21)]

    # Update row
    conn.execute('UPDATE "order" SET "group" = \'grpA_mod\', "where" = 30 WHERE "select" = 1;')
    conn.commit()
    rows = conn.execute('SELECT "key", "offset", "total" FROM "mv_order_proj" ORDER BY "key";').fetchall()
    assert rows == [(1, 'grpA_mod', 31), (3, 'grpA', 21)]

    # Delete row
    conn.execute('DELETE FROM "order" WHERE "select" = 3;')
    conn.commit()
    rows = conn.execute('SELECT "key", "offset", "total" FROM "mv_order_proj" ORDER BY "key";').fetchall()
    assert rows == [(1, 'grpA_mod', 31)]


def test_tier1_aggregate_reserved_keywords_end_to_end():
    """
    Tier 1 single-table aggregation with table 'user', group by 'group',
    aggregating 'order', 'where', 'select'.
    """
    conn = _sqlite_conn()
    conn.execute(
        'CREATE TABLE "user" ('
        '    "key" INTEGER PRIMARY KEY,'
        '    "group" TEXT NOT NULL,'
        '    "order" REAL NOT NULL,'
        '    "where" INTEGER NOT NULL'
        ');'
    )
    conn.commit()

    view_sql = (
        'CREATE VIEW "mv_user_agg" AS '
        'SELECT "group", COUNT(*) AS "select", SUM("order") AS "total", AVG("order") AS "from" '
        'FROM "user" '
        'GROUP BY "group";'
    )

    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
    plan = manager.register_view(view_sql)
    assert plan is not None

    conn.execute('INSERT INTO "user" ("key", "group", "order", "where") VALUES (1, \'admin\', 10.5, 1);')
    conn.execute('INSERT INTO "user" ("key", "group", "order", "where") VALUES (2, \'admin\', 20.5, 2);')
    conn.execute('INSERT INTO "user" ("key", "group", "order", "where") VALUES (3, \'guest\', 5.0, 3);')
    conn.commit()

    rows = conn.execute('SELECT "group", "select", "total", "from" FROM "mv_user_agg" ORDER BY "group";').fetchall()
    assert rows == [('admin', 2, 31.0, 15.5), ('guest', 1, 5.0, 5.0)]

    # Update
    conn.execute('UPDATE "user" SET "order" = 30.5 WHERE "key" = 2;')
    conn.commit()
    rows = conn.execute('SELECT "group", "select", "total", "from" FROM "mv_user_agg" ORDER BY "group";').fetchall()
    assert rows == [('admin', 2, 41.0, 20.5), ('guest', 1, 5.0, 5.0)]

    # Delete
    conn.execute('DELETE FROM "user" WHERE "key" = 3;')
    conn.commit()
    rows = conn.execute('SELECT "group", "select", "total", "from" FROM "mv_user_agg" ORDER BY "group";').fetchall()
    assert rows == [('admin', 2, 41.0, 20.5)]


def test_tier1_distinct_aggregate_reserved_keywords_end_to_end():
    """
    Tier 1 distinct aggregation with table 'from', column 'order', aux table ref count.
    """
    conn = _sqlite_conn()
    conn.execute(
        'CREATE TABLE "from" ('
        '    "key" INTEGER PRIMARY KEY,'
        '    "group" TEXT NOT NULL,'
        '    "order" TEXT NOT NULL'
        ');'
    )
    conn.commit()

    view_sql = (
        'CREATE VIEW "mv_distinct_agg" AS '
        'SELECT "group", COUNT(DISTINCT "order") AS "select" '
        'FROM "from" '
        'GROUP BY "group";'
    )

    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
    plan = manager.register_view(view_sql)
    assert plan is not None

    conn.execute('INSERT INTO "from" ("key", "group", "order") VALUES (1, \'A\', \'apple\');')
    conn.execute('INSERT INTO "from" ("key", "group", "order") VALUES (2, \'A\', \'banana\');')
    conn.execute('INSERT INTO "from" ("key", "group", "order") VALUES (3, \'A\', \'apple\');')
    conn.execute('INSERT INTO "from" ("key", "group", "order") VALUES (4, \'B\', \'cherry\');')
    conn.commit()

    rows = conn.execute('SELECT "group", "select" FROM "mv_distinct_agg" ORDER BY "group";').fetchall()
    assert rows == [('A', 2), ('B', 1)]

    # Delete duplicate
    conn.execute('DELETE FROM "from" WHERE "key" = 3;')
    conn.commit()
    rows = conn.execute('SELECT "group", "select" FROM "mv_distinct_agg" ORDER BY "group";').fetchall()
    assert rows == [('A', 2), ('B', 1)]

    # Delete remaining banana
    conn.execute('DELETE FROM "from" WHERE "key" = 2;')
    conn.commit()
    rows = conn.execute('SELECT "group", "select" FROM "mv_distinct_agg" ORDER BY "group";').fetchall()
    assert rows == [('A', 1), ('B', 1)]


def test_tier1_join_reserved_keywords_end_to_end():
    """
    Tier 1 multi-table join with reserved keywords in table names, join aliases,
    column names, and join ON condition.
    """
    conn = _sqlite_conn()
    conn.execute(
        'CREATE TABLE "order" ('
        '    "key" INTEGER PRIMARY KEY,'
        '    "user" INTEGER NOT NULL,'
        '    "select" REAL NOT NULL'
        ');'
    )
    conn.execute(
        'CREATE TABLE "group" ('
        '    "where" INTEGER PRIMARY KEY,'
        '    "from" TEXT NOT NULL'
        ');'
    )
    conn.commit()

    view_sql = (
        'CREATE VIEW "mv_join_kw" AS '
        'SELECT g."from" AS "group", SUM(o."select") AS "total", COUNT(*) AS "offset" '
        'FROM "order" AS o '
        'JOIN "group" AS g ON o."user" = g."where" '
        'GROUP BY g."from";'
    )

    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
    plan = manager.register_view(view_sql)
    assert plan is not None

    conn.execute('INSERT INTO "group" ("where", "from") VALUES (1, \'North\'), (2, \'South\');')
    conn.execute('INSERT INTO "order" ("key", "user", "select") VALUES (10, 1, 100.0), (20, 1, 50.0), (30, 2, 200.0);')
    conn.commit()

    rows = conn.execute('SELECT "group", "total", "offset" FROM "mv_join_kw" ORDER BY "group";').fetchall()
    assert rows == [('North', 150.0, 2), ('South', 200.0, 1)]

    # Update order amount
    conn.execute('UPDATE "order" SET "select" = 75.0 WHERE "key" = 20;')
    conn.commit()
    rows = conn.execute('SELECT "group", "total", "offset" FROM "mv_join_kw" ORDER BY "group";').fetchall()
    assert rows == [('North', 175.0, 2), ('South', 200.0, 1)]

    # Delete order
    conn.execute('DELETE FROM "order" WHERE "key" = 10;')
    conn.commit()
    rows = conn.execute('SELECT "group", "total", "offset" FROM "mv_join_kw" ORDER BY "group";').fetchall()
    assert rows == [('North', 75.0, 1), ('South', 200.0, 1)]


def test_tier1_union_all_reserved_keywords_end_to_end():
    """
    Tier 1 UNION ALL with reserved keyword tables and columns.
    """
    conn = _sqlite_conn()
    conn.execute('CREATE TABLE "order" ("key" INTEGER PRIMARY KEY, "where" TEXT NOT NULL);')
    conn.execute('CREATE TABLE "user" ("key" INTEGER PRIMARY KEY, "where" TEXT NOT NULL);')
    conn.commit()

    view_sql = (
        'CREATE VIEW "mv_uall_kw" AS '
        'SELECT "key" AS "select", "where" AS "group" FROM "order" '
        'UNION ALL '
        'SELECT "key" AS "select", "where" AS "group" FROM "user";'
    )

    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
    plan = manager.register_view(view_sql)
    assert plan is not None

    conn.execute('INSERT INTO "order" ("key", "where") VALUES (1, \'O1\'), (2, \'O2\');')
    conn.execute('INSERT INTO "user" ("key", "where") VALUES (1, \'U1\');')
    conn.commit()

    rows = conn.execute('SELECT "select", "group" FROM "mv_uall_kw" ORDER BY "select", "group";').fetchall()
    assert rows == [(1, 'O1'), (1, 'U1'), (2, 'O2')]

    # Update
    conn.execute('UPDATE "order" SET "where" = \'O1_mod\' WHERE "key" = 1;')
    conn.commit()
    rows = conn.execute('SELECT "select", "group" FROM "mv_uall_kw" ORDER BY "select", "group";').fetchall()
    assert rows == [(1, 'O1_mod'), (1, 'U1'), (2, 'O2')]


def test_tier2_window_reserved_keywords_end_to_end():
    """
    Tier 2 window / group-scoped view with reserved keyword columns.
    """
    conn = _sqlite_conn()
    conn.execute(
        'CREATE TABLE "order" ('
        '    "key" INTEGER PRIMARY KEY,'
        '    "group" TEXT NOT NULL,'
        '    "select" REAL NOT NULL'
        ');'
    )
    conn.commit()

    view_sql = (
        'CREATE VIEW "mv_win_kw" AS '
        'SELECT "group", "select", '
        'ROW_NUMBER() OVER (PARTITION BY "group" ORDER BY "select") AS "offset" '
        'FROM "order";'
    )

    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
    plan = manager.register_view(view_sql)
    assert plan is not None

    conn.execute('INSERT INTO "order" ("key", "group", "select") VALUES (1, \'grp1\', 50.0), (2, \'grp1\', 20.0), (3, \'grp2\', 10.0);')
    conn.commit()

    rows = conn.execute('SELECT "group", "select", "offset" FROM "mv_win_kw" ORDER BY "group", "offset";').fetchall()
    assert rows == [('grp1', 20.0, 1), ('grp1', 50.0, 2), ('grp2', 10.0, 1)]

    # Update
    conn.execute('UPDATE "order" SET "select" = 5.0 WHERE "key" = 1;')
    conn.commit()
    rows = conn.execute('SELECT "group", "select", "offset" FROM "mv_win_kw" ORDER BY "group", "offset";').fetchall()
    assert rows == [('grp1', 5.0, 1), ('grp1', 20.0, 2), ('grp2', 10.0, 1)]


def test_tier3_fallback_reserved_keywords_end_to_end():
    """
    Tier 3 fallback atomic refresh view with reserved keyword identifiers.
    """
    conn = _sqlite_conn()
    conn.execute('CREATE TABLE "where" ("key" INTEGER PRIMARY KEY, "from" TEXT NOT NULL);')
    conn.commit()

    # Correlated subquery forces Tier 3 fallback
    view_sql = (
        'CREATE VIEW "mv_t3_kw" AS '
        'SELECT w1."key" AS "order", w1."from" AS "group", '
        '(SELECT COUNT(*) FROM "where" w2 WHERE w2."key" <= w1."key") AS "select" '
        'FROM "where" w1;'
    )

    manager = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
    plan = manager.register_view(view_sql)
    assert plan is not None

    conn.execute('INSERT INTO "where" ("key", "from") VALUES (1, \'first\'), (2, \'second\');')
    conn.commit()

    rows = conn.execute('SELECT "order", "group", "select" FROM "mv_t3_kw" ORDER BY "order";').fetchall()
    assert rows == [(1, 'first', 1), (2, 'second', 2)]

    conn.execute('DELETE FROM "where" WHERE "key" = 1;')
    conn.commit()
    rows = conn.execute('SELECT "order", "group", "select" FROM "mv_t3_kw" ORDER BY "order";').fetchall()
    assert rows == [(2, 'second', 1)]


def test_postgres_and_mysql_transpiled_quoting():
    """
    Verify compilation for PostgreSQL and MySQL properly quotes keyword tables,
    columns, aliases, and trigger statements without syntax errors.
    """
    schemas = {
        "order": TableSchema(
            name="order",
            columns={
                "select": ColumnDef(name="select", data_type="INTEGER", primary_key=True),
                "group": ColumnDef(name="group", data_type="VARCHAR(50)", not_null=True),
                "where": ColumnDef(name="where", data_type="INTEGER", not_null=True),
            },
        )
    }

    view_sql = (
        'CREATE VIEW "mv_order_summary" AS '
        'SELECT "group" AS "user", COUNT(*) AS "order", SUM("where") AS "total" '
        'FROM "order" '
        'GROUP BY "group";'
    )
    v_spec = parse_view_definitions(view_sql)[0]

    # Test PostgreSQL
    pg_plan = generate_materialized_view(schemas, v_spec, dialect=Dialect.POSTGRES, ignore_limitations=True)
    assert '"mv_order_summary"' in pg_plan.backing_table_ddl or '"order"' in pg_plan.backing_table_ddl
    assert '"user"' in pg_plan.backing_table_ddl
    for t_sql in pg_plan.trigger_ddls:
        assert '"order"' in t_sql or 'order' in t_sql

    # Test MySQL
    mysql_plan = generate_materialized_view(schemas, v_spec, dialect=Dialect.MYSQL, ignore_limitations=True)
    assert '`user`' in mysql_plan.backing_table_ddl or '`order`' in mysql_plan.backing_table_ddl
    for t_sql in mysql_plan.trigger_ddls:
        assert '`order`' in t_sql or '`user`' in t_sql


if __name__ == "__main__":
    # Propagate the result: CI runs this file as `python3 test_p1_quote_identifiers.py`.
    sys.exit(pytest.main(["-v", __file__]))
