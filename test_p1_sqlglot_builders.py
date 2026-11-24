#!/usr/bin/env python3
"""Tests for AST-based SQL code generation helpers backed by `sqlglot.exp`."""

import sqlite3
import unittest

import sqlglot
from sqlglot import exp as sqlglot_exp

from matview_sync import (
    Dialect,
    MaterializedViewManager,
    build_alter_rename_table_sql,
    build_create_index_sql,
    build_create_view_sql,
    build_drop_sql,
    build_insert_select_sql,
    build_select_sql,
    generate_materialized_view,
    parse_table_schema,
)


class TestSqlglotExpBuilders(unittest.TestCase):
    """Verifies dialect correctness and execution of `sqlglot.exp` builders."""

    def test_build_drop_sql_dialects_and_quoting(self) -> None:
        self.assertEqual(
            build_drop_sql("VIEW", "mv_sales", dialect=Dialect.SQLITE),
            "DROP VIEW IF EXISTS mv_sales;",
        )
        self.assertEqual(
            build_drop_sql("VIEW", "my sales", dialect=Dialect.POSTGRES, cascade=True),
            'DROP VIEW IF EXISTS "my sales" CASCADE;',
        )
        self.assertEqual(
            build_drop_sql("TABLE", ["user", "old_tbl"], dialect=Dialect.MYSQL),
            "DROP TABLE IF EXISTS `user`, old_tbl;",
        )
        self.assertEqual(
            build_drop_sql("TABLE", "public.mv_tbl", dialect=Dialect.POSTGRES),
            "DROP TABLE IF EXISTS public.mv_tbl;",
        )

    def test_build_alter_rename_table_sql_across_dialects(self) -> None:
        self.assertEqual(
            build_alter_rename_table_sql("shadow_mv", "mv_store", dialect=Dialect.SQLITE),
            "ALTER TABLE shadow_mv RENAME TO mv_store;",
        )
        self.assertEqual(
            build_alter_rename_table_sql("shadow_mv", "mv_store", dialect=Dialect.POSTGRES),
            "ALTER TABLE shadow_mv RENAME TO mv_store;",
        )
        self.assertEqual(
            build_alter_rename_table_sql("shadow mv", "order", dialect=Dialect.MYSQL),
            "ALTER TABLE `shadow mv` RENAME `order`;",
        )

    def test_build_create_view_sql_and_live_sqlite_execution(self) -> None:
        pg_ddl = build_create_view_sql(
            "mv_summary",
            "__matview_store_summary",
            ["grp", "total"],
            where_sql="grp IS NOT NULL",
            dialect=Dialect.POSTGRES,
            replace=True,
        )
        parsed_pg = sqlglot.parse_one(pg_ddl, read="postgres")
        self.assertIsInstance(parsed_pg, sqlglot_exp.Create)
        self.assertTrue(parsed_pg.args.get("replace"))

        my_ddl = build_create_view_sql(
            "mv summary",
            "__matview_store_summary",
            ["`grp`", "`user`"],
            where_sql="EXISTS (SELECT 1)",
            dialect=Dialect.MYSQL,
            algorithm="TEMPTABLE",
        )
        self.assertIn("ALGORITHM = TEMPTABLE", my_ddl)
        self.assertIn("`mv summary`", my_ddl)
        self.assertIn("`user`", my_ddl)

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE backing (grp TEXT, total INT);")
            conn.execute("INSERT INTO backing VALUES ('a', 10), (NULL, 5);")
            sq_ddl = build_create_view_sql(
                "v_live",
                "backing",
                ["grp", "total"],
                where_sql="grp IS NOT NULL",
                dialect=Dialect.SQLITE,
            )
            conn.execute(sq_ddl)
            rows = conn.execute("SELECT grp, total FROM v_live;").fetchall()
            self.assertEqual(rows, [("a", 10)])
        finally:
            conn.close()

    def test_build_create_index_sql_variants_and_live_sqlite(self) -> None:
        pg_idx = build_create_index_sql(
            "idx_grp",
            "mv_tbl",
            ["grp", "sub"],
            dialect=Dialect.POSTGRES,
            unique=True,
            using_method="btree",
            where_clause="grp IS NOT NULL",
            nulls_not_distinct=True,
        )
        self.assertIn("CREATE UNIQUE INDEX idx_grp ON mv_tbl", pg_idx)
        self.assertIn("USING btree", pg_idx)
        self.assertIn("WHERE grp IS NOT NULL", pg_idx)
        self.assertTrue(pg_idx.endswith("NULLS NOT DISTINCT;"))

        my_expr_idx = build_create_index_sql(
            "idx_expr",
            "mv_tbl",
            ["((`a` + `b`)) DESC"],
            dialect=Dialect.MYSQL,
        )
        self.assertNotIn("(((", my_expr_idx)
        self.assertIn("((`a` + `b`) DESC)", my_expr_idx)
        self.assertIn("ON mv_tbl (", my_expr_idx)

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE t_idx (id INTEGER PRIMARY KEY, name TEXT, score INT);")
            sq_idx = build_create_index_sql(
                "idx_t_name",
                "t_idx",
                ['"name" COLLATE NOCASE DESC'],
                dialect=Dialect.SQLITE,
                where_clause="score > 0",
            )
            conn.execute(sq_idx)
            names = [r[1] for r in conn.execute("PRAGMA index_list('t_idx');").fetchall()]
            self.assertIn("idx_t_name", names)
        finally:
            conn.close()

    def test_build_select_and_insert_select_sql(self) -> None:
        sel = build_select_sql(
            ["bucket", "COUNT(*) AS cnt", "SUM(amt) AS total"],
            "items",
            where_sql="amt > 0",
            group_by_exprs=["bucket"],
            having_sql="COUNT(*) >= 1",
            order_by_exprs=["total DESC"],
            limit=5,
            dialect=Dialect.POSTGRES,
        )
        parsed_sel = sqlglot.parse_one(sel, read="postgres")
        self.assertIsInstance(parsed_sel, sqlglot_exp.Select)
        self.assertIsNotNone(parsed_sel.args.get("group"))
        self.assertIsNotNone(parsed_sel.args.get("having"))
        self.assertIsNotNone(parsed_sel.args.get("limit"))

        ins = build_insert_select_sql("dst", ["bucket", "cnt", "total"], sel, dialect=Dialect.MYSQL)
        parsed_ins = sqlglot.parse_one(ins, read="mysql")
        self.assertIsInstance(parsed_ins, sqlglot_exp.Insert)

    def test_end_to_end_plan_ddls_use_ast_builders(self) -> None:
        schema = "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT NOT NULL, amt INTEGER NOT NULL);"
        view_sql = (
            "CREATE VIEW mv_sales_region AS "
            "SELECT region, COUNT(*) AS cnt, SUM(amt) AS total FROM sales GROUP BY region;"
        )
        schemas = parse_table_schema(schema, dialect=Dialect.POSTGRES)
        plan = generate_materialized_view(schemas, view_sql, dialect=Dialect.POSTGRES)
        self.assertIsInstance(sqlglot.parse_one(plan.reader_view_ddl, read="postgres"), sqlglot_exp.Create)
        for idx_ddl in plan.index_ddls:
            self.assertIsInstance(sqlglot.parse_one(idx_ddl, read="postgres"), sqlglot_exp.Create)
        for drop_ddl in plan.drop_statements():
            if not drop_ddl.upper().startswith("DROP FUNCTION"):
                self.assertIsInstance(sqlglot.parse_one(drop_ddl, read="postgres"), sqlglot_exp.Drop)

        conn = sqlite3.connect(":memory:")
        try:
            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            mgr.apply_schema_ddl(schema)
            conn.execute("INSERT INTO sales VALUES (1, 'EU', 10), (2, 'US', 20);")
            conn.commit()
            live_plan = mgr.register_view(view_sql)
            self.assertEqual(
                conn.execute(f"SELECT region, cnt, total FROM {live_plan.view_name} ORDER BY region;").fetchall(),
                [("EU", 1, 10), ("US", 1, 20)],
            )
            self.assertTrue(mgr.drop_view(live_plan.view_name))
        finally:
            conn.close()


    def test_special_char_view_names_and_extract_created_object(self) -> None:
        import re

        from matview.sql_frontend import _extract_created_object

        # _extract_created_object must handle quoted names with spaces, single quotes,
        # dots, and escaped quote delimiters.
        self.assertEqual(_extract_created_object('CREATE VIEW "sales summary" AS SELECT 1;'), ("view", "sales summary"))
        self.assertEqual(_extract_created_object('CREATE VIEW "user\'s view" AS SELECT 1;'), ("view", "user's view"))
        self.assertEqual(_extract_created_object('CREATE VIEW "a ""b"" c" AS SELECT 1;'), ("view", 'a "b" c'))
        self.assertEqual(_extract_created_object('CREATE VIEW "my.view" AS SELECT 1;'), ("view", "my.view"))
        self.assertEqual(_extract_created_object('CREATE VIEW "public"."my.view" AS SELECT 1;'), ("view", "my.view"))
        self.assertEqual(_extract_created_object("CREATE VIEW `sales summary` AS SELECT 1;"), ("view", "sales summary"))
        self.assertEqual(_extract_created_object("CREATE TABLE `order items` (id INT);"), ("table", "order items"))
        self.assertEqual(_extract_created_object('CREATE TRIGGER "trg 1" AFTER INSERT ON t;'), ("trigger", "trg 1"))

        # Across all three dialects, special-character view names (and base table names)
        # must produce valid unquoted internal identifiers without collisions.
        for dial in (Dialect.POSTGRES, Dialect.SQLITE, Dialect.MYSQL):
            q = "`" if dial == Dialect.MYSQL else '"'
            sc = parse_table_schema(
                f"CREATE TABLE {q}order items{q} (id INT PRIMARY KEY, g INT, v INT);",
                dialect=dial,
            )
            seen_stores = set()
            vnames = ["sales summary", "sales_summary", "Sales Summary", "sales-summary", "user's view", "my.view", "---"]
            for vn in vnames:
                qvn = f"{q}{vn.replace(q, q + q)}{q}"
                plan = generate_materialized_view(
                    sc,
                    f"CREATE VIEW {qvn} AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM {q}order items{q} GROUP BY g;",
                    dialect=dial,
                    ignore_limitations=True,
                )
                self.assertRegex(plan.backing_table_name, r"^[A-Za-z_][A-Za-z0-9_]*$", (dial, vn))
                self.assertLessEqual(len(plan.backing_table_name), 62)
                self.assertNotIn(plan.backing_table_name, seen_stores, (dial, vn))
                seen_stores.add(plan.backing_table_name)
                for obj_name in (*plan.trigger_names, *plan.index_names, *plan.auxiliary_table_names):
                    self.assertRegex(obj_name, r"^[A-Za-z_][A-Za-z0-9_]*$", (dial, vn, obj_name))
                    self.assertLessEqual(len(obj_name), 62)
                expected_ref = vn if vn == "sales_summary" else qvn
                self.assertIn(expected_ref, plan.reader_view_ddl)
                self.assertTrue(any(expected_ref in stmt for stmt in plan.drop_statements()), (dial, vn, plan.drop_statements()))

        # Trailing-newline bypass and long custom prefix cap in _bounded_ident.
        long_prefix_plan = generate_materialized_view(
            parse_table_schema("CREATE TABLE t (id INT PRIMARY KEY, g INT, v INT);", dialect=Dialect.POSTGRES),
            'CREATE VIEW "sales\n" AS SELECT g, SUM(v) AS s FROM t GROUP BY g;',
            dialect=Dialect.POSTGRES,
            object_prefix="x_" * 20,
        )
        self.assertRegex(long_prefix_plan.backing_table_name, r"^[A-Za-z_][A-Za-z0-9_]*$")
        self.assertLessEqual(len(long_prefix_plan.backing_table_name), 62)
        for obj_name in (*long_prefix_plan.trigger_names, *long_prefix_plan.index_names):
            self.assertRegex(obj_name, r"^[A-Za-z_][A-Za-z0-9_]*$")
            self.assertLessEqual(len(obj_name), 62)

        # PostgreSQL BIT VARYING / BIT VARYING(n) DDL must parse cleanly in parse_table_schema.
        bit_schemas = parse_table_schema(
            "CREATE TABLE tb (id INT PRIMARY KEY, g INT, b4 BIT(4), bv BIT VARYING, bv8 BIT VARYING(8));",
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("tb", bit_schemas)
        self.assertIn("bv", bit_schemas["tb"].columns)
        self.assertIn("bv8", bit_schemas["tb"].columns)

        # Live SQLite end-to-end lifecycle across all adversarial view names on a spaced base table,
        # including diff_views(prune_removed=True) quoted DROP generation.
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute('CREATE TABLE "order items" (id INTEGER PRIMARY KEY, g INTEGER, v INTEGER);')
            conn.execute('INSERT INTO "order items" VALUES (1, 1, 10), (2, 1, 20), (3, 2, 5);')
            conn.commit()
            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            sq_names = ["sales summary", "sales_summary", "sales-summary", "user's view", 'a "b" c', "my.view", "---"]
            for vn in sq_names:
                qvn = '"' + vn.replace('"', '""') + '"'
                mgr.register_view(
                    f'CREATE VIEW {qvn} AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM "order items" GROUP BY g;'
                )
            conn.execute('INSERT INTO "order items" VALUES (4, 2, 15);')
            conn.execute('UPDATE "order items" SET v = 30 WHERE id = 1;')
            conn.execute('DELETE FROM "order items" WHERE id = 3;')
            conn.commit()
            expected = conn.execute(
                'SELECT g, SUM(v), COUNT(*) FROM "order items" GROUP BY g ORDER BY g;'
            ).fetchall()
            prune_report = mgr.diff_views(
                ['CREATE VIEW sales_summary AS SELECT g, SUM(v) AS s, COUNT(*) AS c FROM "order items" GROUP BY g;'],
                prune_removed=True,
            )
            for vn in sq_names:
                qvn = '"' + vn.replace('"', '""') + '"'
                if vn != "sales_summary":
                    self.assertIn(vn, prune_report)
                    self.assertTrue(
                        any(f"DROP VIEW IF EXISTS {qvn};" == d for d in prune_report[vn].pending_ddls),
                        (vn, prune_report[vn].pending_ddls),
                    )
                got = conn.execute(f"SELECT g, s, c FROM {qvn} ORDER BY g;").fetchall()
                self.assertEqual(got, expected, vn)
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(f"DELETE FROM {qvn};")
                conn.rollback()
                self.assertEqual(mgr.check_schema_drift()[vn], "ok")
                mgr.repair_view(vn)
                self.assertTrue(mgr.drop_view(vn))
        finally:
            conn.close()

    def test_dotted_base_tables_and_special_char_user_indexes(self) -> None:
        # 1. Offline multi-dialect codegen: dotted base tables & columns, joins,
        # recommended base indexes, compound backing_col_map, and special-char CREATE INDEX
        # (dots, spaces, dashes, apostrophes, doubled quotes/backticks, SQL keywords).
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            q = "`" if dialect == Dialect.MYSQL else '"'
            q_esc = "``" if dialect == Dialect.MYSQL else '""'
            q_col_inner = f"q{q_esc}col"
            schema_sql = (
                f"CREATE TABLE {q}my.orders{q} ("
                f"{q}row.id{q} INT PRIMARY KEY, "
                f"{q}grp.id{q} INT NOT NULL, "
                f"{q}a-b{q} VARCHAR(50), "
                f"{q}desc{q} INT, "
                f"{q}nulls last{q} INT, "
                f"{q}{q_col_inner}{q} INT NOT NULL"
                f");\n"
                f"CREATE TABLE {q}my.dim{q} ("
                f"{q}dim.id{q} INT PRIMARY KEY, "
                f"{q}dim.name{q} VARCHAR(50) NOT NULL"
                f");"
            )
            view_script = (
                f"CREATE VIEW {q}v.1's view{q} AS "
                f"SELECT {q}grp.id{q}, {q}a-b{q}, {q}desc{q}, {q}nulls last{q}, "
                f"SUM({q}{q_col_inner}{q}) AS {q}tot.q{q}, "
                f"SUM({q}{q_col_inner}{q}) + 1 AS {q}net.q{q} "
                f"FROM {q}my.orders{q} "
                f"GROUP BY {q}grp.id{q}, {q}a-b{q}, {q}desc{q}, {q}nulls last{q};\n"
                f"CREATE INDEX {q}idx v.1's{q} ON {q}v.1's view{q} "
                f"({q}desc{q} ASC, {q}nulls last{q} DESC, {q}tot.q{q} DESC);\n"
                f"CREATE INDEX {q}idx net.1{q} ON {q}v.1's view{q} ({q}net.q{q} DESC) WHERE {q}net.q{q} > 0;\n"
                f"CREATE INDEX {q}idx expr.1{q} ON {q}v.1's view{q} (LOWER({q}a-b{q}) DESC) WHERE {q}tot.q{q} > 0;\n"
            )
            try:
                plan = generate_materialized_view(
                    schema_sql,
                    view_script,
                    dialect=dialect,
                    ignore_limitations=True,
                    incremental_backfill=True,
                )
            except Exception as exc:
                self.fail(f"generate_materialized_view failed on {dialect.value}: {exc}")
            all_trigs = "\n".join(plan.trigger_ddls)
            self.assertIn(f"{q}my.orders{q}", all_trigs)
            self.assertNotIn(f"{q}my{q}.{q}orders{q}", all_trigs)
            self.assertIn(f"{q}grp.id{q}", plan.backing_table_ddl)
            self.assertIn(f"{q}nulls last{q}", plan.backing_table_ddl)
            self.assertIn(f"{q}v.1's view{q}", plan.reader_view_ddl)
            idx_joined = "\n".join(plan.index_ddls)
            self.assertIn(f"{q}__matview_idx v.1's{q}", idx_joined)
            self.assertIn(f"{q}desc{q} ASC", idx_joined)
            self.assertIn(f"{q}nulls last{q} DESC", idx_joined)
            # Column named "nulls last" must NOT inject a spurious trailing NULLS LAST clause.
            self.assertNotIn(f"{q}nulls last{q} DESC NULLS LAST", idx_joined)
            # Compound expression alias ("net.q") rewritten cleanly via backing_col_map without trailing AS.
            self.assertIn(f"{q}__matview_idx net.1{q}", idx_joined)
            net_idx_ddl = next(d for d in plan.index_ddls if "__matview_idx net.1" in d)
            self.assertIn(f"{q}tot.q{q} + 1", net_idx_ddl)
            self.assertNotIn(f"AS {q}net.q{q}", net_idx_ddl)
            self.assertNotIn(f"({q}net.q{q}", net_idx_ddl)
            # Incremental backfill SQL must reference the literal dotted base table, not schema.table.
            if plan.incremental_backfill_spec is not None:
                chunk_joined = "\n".join(plan.incremental_backfill_spec.chunk_sql_templates)
                self.assertIn(f"{q}my.orders{q}", chunk_joined)
                self.assertNotIn(f"{q}my{q}.{q}orders{q}", chunk_joined)

            # Join view across two dotted base tables + recommended base indexes.
            join_script = (
                f"CREATE VIEW {q}v.join's{q} AS "
                f"SELECT d.{q}dim.name{q}, SUM(o.{q}{q_col_inner}{q}) AS {q}tot.q{q} "
                f"FROM {q}my.orders{q} o JOIN {q}my.dim{q} d ON o.{q}grp.id{q} = d.{q}dim.id{q} "
                f"GROUP BY d.{q}dim.name{q};"
            )
            jplan = generate_materialized_view(
                schema_sql,
                join_script,
                indexes=[
                    ("dim.name", "tot.q"),
                    {"name": "idx dict.1", "columns": f"{q}tot.q{q} DESC, {q}dim.name{q} ASC"},
                ],
                dialect=dialect,
                ignore_limitations=True,
            )
            jtrigs = "\n".join(jplan.trigger_ddls)
            self.assertIn(f"{q}my.orders{q}", jtrigs)
            self.assertIn(f"{q}my.dim{q}", jtrigs)
            self.assertNotIn(f"{q}my{q}.{q}orders{q}", jtrigs)
            self.assertNotIn(f"{q}my{q}.{q}dim{q}", jtrigs)
            jidxs = "\n".join(jplan.index_ddls)
            self.assertIn(f"{q}__matview_idx dict.1{q}", jidxs)
            self.assertIn(f"{q}dim.name{q}", jidxs)
            # Recommended base index on my.orders(grp.id) must quote literal dotted table & col.
            rec_joined = "\n".join(jplan.recommended_base_indexes)
            self.assertIn(f"ON {q}my.orders{q} ({q}grp.id{q})", rec_joined)
            self.assertNotIn(f"{q}my{q}.{q}orders{q}", rec_joined)

            # Negative validation: unknown column, unknown expr column, table-qualified column, wrong target.
            for bad_idx in (
                f"CREATE INDEX bad1 ON {q}v.1's view{q} ({q}no.such.col{q});",
                f"CREATE INDEX bad2 ON {q}v.1's view{q} (LOWER({q}no.such.col{q}));",
                f"CREATE INDEX bad3 ON {q}v.1's view{q} (t.{q}grp.id{q});",
                f"CREATE INDEX bad4 ON {q}wrong.view{q} ({q}grp.id{q});",
            ):
                with self.assertRaises(ValueError, msg=(dialect, bad_idx)):
                    generate_materialized_view(
                        schema_sql,
                        f"CREATE VIEW {q}v.1's view{q} AS SELECT {q}grp.id{q}, SUM({q}{q_col_inner}{q}) AS {q}tot.q{q} FROM {q}my.orders{q} GROUP BY {q}grp.id{q};",
                        indexes=[bad_idx],
                        dialect=dialect,
                        ignore_limitations=True,
                    )

        # Genuine schema-qualified table names still preserve schema.table when allow_qualified=True.
        self.assertEqual(
            build_create_index_sql("idx_pub", "public.orders", ["grp"], dialect=Dialect.POSTGRES),
            "CREATE INDEX idx_pub ON public.orders (grp);",
        )
        self.assertEqual(
            build_drop_sql("TABLE", "public.orders", dialect=Dialect.POSTGRES),
            "DROP TABLE IF EXISTS public.orders;",
        )

        # 2. Live SQLite end-to-end lifecycle: dotted base table, dotted PK, compound backing_col_map,
        # special-char indexes, incremental backfill, DML, drift check, diff_views, repair_view, drop_view.
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                'CREATE TABLE "my.items" ('
                '"row.id" INTEGER PRIMARY KEY, '
                '"grp.id" INTEGER NOT NULL, '
                '"a-b" TEXT, '
                '"desc" INTEGER, '
                '"nulls last" INTEGER, '
                '"q""col" INTEGER NOT NULL'
                ");"
            )
            conn.executemany(
                'INSERT INTO "my.items" VALUES (?, ?, ?, ?, ?, ?);',
                [(i, i % 3, f"K{i % 4}", i * 10, i * 2, i + 1) for i in range(1, 21)],
            )
            conn.commit()
            mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
            vsql = (
                'CREATE VIEW "v.1\'s summary" AS '
                'SELECT "grp.id", "a-b", "desc", "nulls last", '
                'SUM("q""col") AS "tot.q", '
                'SUM("q""col") + 1 AS "net.q" '
                'FROM "my.items" '
                'GROUP BY "grp.id", "a-b", "desc", "nulls last";\n'
                'CREATE INDEX "idx v.1\'s" ON "v.1\'s summary" ("desc" ASC, "nulls last" DESC, "tot.q" DESC);\n'
                'CREATE INDEX "idx net.1" ON "v.1\'s summary" ("net.q" DESC) WHERE "net.q" > 0;\n'
                'CREATE INDEX "idx expr.1" ON "v.1\'s summary" (LOWER("a-b") DESC) WHERE "tot.q" > 0;'
            )
            plan_live = mgr.register_view(vsql, populate=False, incremental_backfill=True)
            try:
                batches = mgr.backfill_incrementally(plan_live, batch_size=5)
            except sqlite3.OperationalError as exc:
                self.fail(f"backfill_incrementally failed on dotted base table: {exc}")
            self.assertGreaterEqual(batches, 4)

            conn.execute('INSERT INTO "my.items" VALUES (21, 1, \'NEW\', 99, 88, 77);')
            conn.execute('UPDATE "my.items" SET "q""col" = 50 WHERE "row.id" = 1;')
            conn.execute('DELETE FROM "my.items" WHERE "row.id" = 2;')
            conn.commit()

            got = conn.execute(
                'SELECT "grp.id", "a-b", "desc", "nulls last", "tot.q", "net.q" '
                'FROM "v.1\'s summary" ORDER BY "grp.id", "a-b", "desc";'
            ).fetchall()
            exp = conn.execute(
                'SELECT "grp.id", "a-b", "desc", "nulls last", SUM("q""col"), SUM("q""col") + 1 '
                'FROM "my.items" GROUP BY "grp.id", "a-b", "desc", "nulls last" '
                'ORDER BY "grp.id", "a-b", "desc";'
            ).fetchall()
            self.assertEqual(got, exp)

            live_idx_names = {
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index';").fetchall()
            }
            for expected_idx in ("__matview_idx v.1's", "__matview_idx net.1", "__matview_idx expr.1"):
                self.assertIn(expected_idx, live_idx_names)

            self.assertEqual(mgr.check_schema_drift()["v.1's summary"], "ok")
            diff_rep = mgr.diff_views([vsql])
            self.assertEqual(diff_rep["v.1's summary"].status, "NOOP")
            mgr.repair_view("v.1's summary")
            self.assertEqual(mgr.check_schema_drift()["v.1's summary"], "ok")
            self.assertTrue(mgr.drop_view("v.1's summary"))

            # 3. Adversarial edge cases: bare compound index + bare boolean WHERE, AVG+ASC NULLS LAST,
            # SELECT * (expanded & unexpanded LIMIT + tuple indexes), UNION/EXCEPT over dotted tables,
            # HAVING/GROUP BY dotted aliases, long user index names, binary expression parens,
            # parenthesized COLLATE, shorthand multi-expr index, and _normalize_index_column_key.
            from matview.logical_ir import _normalize_index_column_key
            from matview.sql_frontend import parse_view_index_spec

            self.assertEqual(
                _normalize_index_column_key('"my.tbl"."a.b" DESC', frozenset({"a.b"})),
                "a.b",
            )
            self.assertEqual(
                _normalize_index_column_key('"a.b" COLLATE "foo bar" DESC', frozenset({"a.b"})),
                "a.b",
            )
            sh_spec = parse_view_index_spec('("grp.id" + 1), ("tot.q" + 2)', dialect=Dialect.POSTGRES)
            self.assertEqual(len(sh_spec.columns), 2)

            # Bare compound index key ("net.q") and bare boolean compound WHERE ("pos.flag")
            for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
                q = "`" if dialect == Dialect.MYSQL else '"'
                sch_c = f"CREATE TABLE {q}my.t{q} ({q}row.id{q} INT PRIMARY KEY, {q}grp.id{q} INT NOT NULL, {q}a-b{q} VARCHAR(50), {q}v.1{q} INT NOT NULL);"
                long_idx = "idx_" + ("a" * 58)
                v_c = (
                    f"CREATE VIEW {q}v.cmp{q} AS "
                    f"SELECT {q}grp.id{q}, {q}a-b{q}, SUM({q}v.1{q}) AS {q}tot.v{q}, "
                    f"SUM({q}v.1{q}) + 1 AS {q}net.v{q}, (SUM({q}v.1{q}) > 0) AS {q}pos.flag{q} "
                    f"FROM {q}my.t{q} GROUP BY {q}grp.id{q}, {q}a-b{q};\n"
                    f"CREATE INDEX {q}idx_bare_cmp{q} ON {q}v.cmp{q} ({q}net.v{q});\n"
                    f"CREATE INDEX {q}idx_bare_where{q} ON {q}v.cmp{q} ({q}tot.v{q}) WHERE {q}pos.flag{q};\n"
                    f"CREATE INDEX {q}{long_idx}{q} ON {q}v.cmp{q} ({q}tot.v{q});\n"
                    f"CREATE INDEX {q}idx_bin_expr{q} ON {q}v.cmp{q} ((({q}grp.id{q} + 1) + ({q}tot.v{q} + 2)));\n"
                )
                p_c = generate_materialized_view(sch_c, v_c, dialect=dialect, ignore_limitations=True)
                bare_ddl = next(d for d in p_c.index_ddls if "idx_bare_cmp" in d)
                self.assertIn(f"{q}tot.v{q} + 1", bare_ddl, (dialect, bare_ddl))
                self.assertNotIn(f"({q}net.v{q})", bare_ddl, (dialect, bare_ddl))
                if dialect != Dialect.MYSQL:
                    where_ddl = next(d for d in p_c.index_ddls if "idx_bare_where" in d)
                    self.assertIn(f"{q}tot.v{q} > 0", where_ddl, (dialect, where_ddl))
                    self.assertNotIn(f"WHERE {q}pos.flag{q}", where_ddl, (dialect, where_ddl))
                # Bounded user index names on PG (<= 63 bytes) and MySQL (<= 64 chars)
                for ix_n in p_c.index_names:
                    if dialect == Dialect.POSTGRES:
                        self.assertLessEqual(len(ix_n.encode("utf-8")), 63, ix_n)
                    elif dialect == Dialect.MYSQL:
                        self.assertLessEqual(len(ix_n), 64, ix_n)
                # Non-outer-parenthesized binary expression must be wrapped in outer parens
                bin_ddl = next(d for d in p_c.index_ddls if "idx_bin_expr" in d)
                if dialect == Dialect.POSTGRES:
                    self.assertIn(f'(("{q[0]}grp.id{q[0]}" + 1) + ("tot.v" + 2))'.replace('""', '"'), bin_ddl)
                    self.assertIn('((("grp.id" + 1) + ("tot.v" + 2)))', bin_ddl)

            # Parenthesized COLLATE preserved
            p_coll = generate_materialized_view(
                'CREATE TABLE "my.t" ("id" INT PRIMARY KEY, "a-b" TEXT);',
                'CREATE VIEW "v.coll" AS SELECT "a-b", COUNT(*) AS c FROM "my.t" GROUP BY "a-b";\n'
                'CREATE INDEX "idx_coll" ON "v.coll" (("a-b" COLLATE NOCASE) DESC);',
                dialect=Dialect.SQLITE,
            )
            coll_ddl = next(d for d in p_coll.index_ddls if "idx_coll" in d)
            self.assertIn("COLLATE NOCASE", coll_ddl)

            # AVG dotted alias with ASC NULLS LAST stays stored in backing table when indexed
            p_avg = generate_materialized_view(
                'CREATE TABLE "my.t" ("id" INT PRIMARY KEY, "grp.id" INT NOT NULL, "q" INT NOT NULL);',
                'CREATE VIEW "v.avg" AS SELECT "grp.id", AVG("q") AS "avg.q" FROM "my.t" GROUP BY "grp.id";\n'
                'CREATE INDEX "idx_avg" ON "v.avg" ("avg.q" ASC NULLS LAST);',
                dialect=Dialect.SQLITE,
            )
            self.assertIn('"avg.q"', p_avg.backing_table_ddl)

            # Live SQLite: SELECT * (expanded + unexpanded LIMIT + tuple indexes), UNION/EXCEPT, HAVING/GROUP BY alias
            conn.execute('CREATE TABLE "my.items2" ("row.id" INTEGER PRIMARY KEY, "grp.id" INTEGER NOT NULL, "a-b" TEXT);')
            conn.execute('INSERT INTO "my.items2" VALUES (1, 1, \'K1\'), (2, 9, \'Z9\');')
            conn.commit()

            mgr.register_view('CREATE VIEW "v.star" AS SELECT * FROM "my.items";')
            self.assertEqual(
                conn.execute('SELECT * FROM "v.star" ORDER BY "row.id";').fetchall(),
                conn.execute('SELECT * FROM "my.items" ORDER BY "row.id";').fetchall(),
            )
            self.assertEqual(mgr.check_schema_drift()["v.star"], "ok")

            mgr.register_view(
                'CREATE VIEW "v.lim" AS SELECT * FROM "my.items" ORDER BY "row.id" LIMIT 5;',
                indexes=[("grp.id", "desc"), ("a-b",)],
            )
            self.assertEqual(mgr.check_schema_drift()["v.lim"], "ok")
            mgr.repair_view("v.lim")
            self.assertEqual(mgr.check_schema_drift()["v.lim"], "ok")

            try:
                mgr.register_view(
                    'CREATE VIEW "v.union" AS '
                    'SELECT "grp.id", "a-b" FROM "my.items" '
                    'UNION '
                    'SELECT "grp.id", "a-b" FROM "my.items2";'
                )
            except sqlite3.OperationalError as exc:
                self.fail(f"UNION view over dotted tables/columns failed: {exc}")
            self.assertEqual(
                conn.execute('SELECT "grp.id", "a-b" FROM "v.union" ORDER BY 1, 2;').fetchall(),
                conn.execute(
                    'SELECT "grp.id", "a-b" FROM "my.items" UNION SELECT "grp.id", "a-b" FROM "my.items2" ORDER BY 1, 2;'
                ).fetchall(),
            )

            mgr.register_view(
                'CREATE VIEW "v.hav" AS '
                'SELECT o."grp.id" AS "g.out", SUM(o."q""col") AS "tot.q" '
                'FROM "my.items" o GROUP BY o."grp.id" HAVING o."grp.id" > 0 AND SUM(o."q""col") > 5;'
            )
            self.assertEqual(
                conn.execute('SELECT * FROM "v.hav" ORDER BY 1;').fetchall(),
                conn.execute(
                    'SELECT o."grp.id", SUM(o."q""col") FROM "my.items" o '
                    'GROUP BY o."grp.id" HAVING o."grp.id" > 0 AND SUM(o."q""col") > 5 ORDER BY 1;'
                ).fetchall(),
            )

            mgr.register_view(
                'CREATE VIEW "v.gexpr" AS '
                'SELECT LOWER("a-b") AS "lo.ab", SUM("q""col") AS "tot.q" '
                'FROM "my.items" GROUP BY "lo.ab";'
            )
            self.assertEqual(
                conn.execute('SELECT * FROM "v.gexpr" ORDER BY 1;').fetchall(),
                conn.execute(
                    'SELECT LOWER("a-b") AS "lo.ab", SUM("q""col") FROM "my.items" GROUP BY "lo.ab" ORDER BY 1;'
                ).fetchall(),
            )
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
