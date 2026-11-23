#!/usr/bin/env python3
"""Tests for the modular `matview` package architecture, immutable IR, and CLI/facade parity."""

from __future__ import annotations

import contextlib
from dataclasses import FrozenInstanceError
import io
import importlib
import pkgutil
import re
import subprocess
import sys
import unittest
import warnings
from pathlib import Path

import matview
import matview_sync as ms


class TestModularPackageArchitecture(unittest.TestCase):
    def test_expected_submodules_importable(self) -> None:
        modules = [
            "matview.engines",
            "matview.logical_ir",
            "matview.sql_frontend",
            "matview.connectivity",
            "matview.planner",
            "matview.compiler",
            "matview.manager",
            "matview.cli",
            "matview.dialects",
            "matview.dialects.templates",
            "matview.dialects.base",
            "matview.dialects.sqlite",
            "matview.dialects.postgres",
            "matview.dialects.mysql",
            "matview.dialects.emitters",
            "matview.maintenance",
            "matview.maintenance.single",
            "matview.maintenance.join",
            "matview.maintenance.zset",
            "matview.maintenance.union_all_agg",
            "matview.maintenance.tier2",
            "matview.maintenance.tier3",
        ]
        # Discover package modules as well as pinning the established public
        # architecture. This keeps newly added modules inside the import check.
        modules.extend(
            info.name
            for info in pkgutil.walk_packages(matview.__path__, matview.__name__ + ".")
        )
        for mod_name in sorted(set(modules)):
            with self.subTest(module=mod_name):
                mod = importlib.import_module(mod_name)
                self.assertIsNotNone(mod)

    def test_facade_reexports_identical_objects(self) -> None:
        for attr in (
            "Dialect",
            "EngineVersion",
            "CompilationContext",
            "LogicalIR",
            "IncrementalViewCompiler",
            "MaterializedViewManager",
            "MaterializedViewPlan",
            "generate_materialized_view",
            "parse_table_schema",
            "parse_view_definition",
            "main",
        ):
            self.assertIs(getattr(ms, attr), getattr(matview, attr), f"Mismatch for {attr}")

    def test_ir_and_context_are_immutable(self) -> None:
        schemas = ms.parse_table_schema(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, cat TEXT NOT NULL, val INTEGER NOT NULL);"
        )
        vspec = ms.parse_view_definition(
            "CREATE VIEW v_dist AS SELECT DISTINCT cat, val FROM items WHERE val > 0;",
            known_tables=schemas.keys(),
        )
        comp = ms.IncrementalViewCompiler(schemas, vspec, dialect=ms.Dialect.POSTGRES)
        self.assertIsInstance(comp.context, ms.CompilationContext)
        self.assertIsInstance(comp.ir, ms.LogicalIR)

        with self.assertRaises(FrozenInstanceError):
            comp.ir.logical_view_name = "mutated"  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            comp.context.alg_mode = "mutated"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            comp.view = vspec  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            comp.primary_schema = None  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            comp._working_select_items = ()  # type: ignore[misc]
        with self.assertRaises(TypeError):
            comp.schemas["extra"] = schemas["items"]  # type: ignore[index]

    def test_distinct_and_union_all_agg_never_mutate_compiler_ir(self) -> None:
        schemas = ms.parse_table_schema(
            "CREATE TABLE a (id INTEGER PRIMARY KEY, cat TEXT NOT NULL, x INTEGER NOT NULL);\n"
            "CREATE TABLE b (id INTEGER PRIMARY KEY, cat TEXT NOT NULL, x INTEGER NOT NULL);"
        )
        dist_v = ms.parse_view_definition(
            "CREATE VIEW v_d AS SELECT DISTINCT cat, x FROM a;",
            known_tables=schemas.keys(),
        )
        uall_v = ms.parse_view_definition(
            """
            CREATE VIEW v_u AS
            SELECT u.cat, COUNT(*) AS cnt, SUM(u.x) AS tot
            FROM (
                SELECT cat, x FROM a WHERE x > 0
                UNION ALL
                SELECT cat, x FROM b WHERE x > 0
            ) AS u
            GROUP BY u.cat;
            """,
            known_tables=schemas.keys(),
        )
        for vspec in (dist_v, uall_v):
            for dialect in (ms.Dialect.SQLITE, ms.Dialect.POSTGRES, ms.Dialect.MYSQL):
                comp = ms.IncrementalViewCompiler(
                    schemas, vspec, dialect=dialect, ignore_limitations=True
                )
                ctx_before = comp.context
                ir_before = comp.ir
                view_before = comp.view
                work_before = comp._working_select_items
                grp_before = comp._effective_group_by_exprs
                schema_before = comp.primary_schema

                plan = comp.compile()
                self.assertEqual(plan.view_name, vspec.view_name)
                self.assertIs(comp.context, ctx_before)
                self.assertIs(comp.ir, ir_before)
                self.assertIs(comp.view, view_before)
                self.assertIs(comp._working_select_items, work_before)
                self.assertEqual(comp._effective_group_by_exprs, grp_before)
                self.assertIs(comp.primary_schema, schema_before)

    def test_every_public_package_export_is_identical_on_the_legacy_facade(self) -> None:
        public_names = tuple(matview.__all__)
        self.assertTrue(public_names)
        self.assertEqual(
            len(public_names),
            len(set(public_names)),
            "matview.__all__ must not contain duplicate exports",
        )

        package_cli = importlib.import_module("matview.cli")
        self.assertIs(matview.main, package_cli.main)
        for name in public_names:
            with self.subTest(public_export=name):
                self.assertFalse(name.startswith("_"), f"Private name exported: {name}")
                self.assertTrue(hasattr(matview, name), f"Missing package export: {name}")
                self.assertTrue(hasattr(ms, name), f"Missing facade export: {name}")
                self.assertIs(getattr(ms, name), getattr(matview, name))

    def test_package_and_facade_import_cleanly_in_fresh_interpreters(self) -> None:
        # A fresh interpreter is essential here: importlib.reload() in the test
        # process would retain modules imported by the other tests and can hide
        # order-dependent circular-import failures.
        script = r"""
import importlib
import pkgutil
import sys

if sys.argv[1] == "facade-first":
    import matview_sync as facade
    import matview
else:
    import matview
    import matview_sync as facade

module_names = sorted(
    item.name for item in pkgutil.walk_packages(matview.__path__, matview.__name__ + ".")
)
if sys.argv[1] == "package-first":
    module_names.reverse()
for module_name in module_names:
    module = importlib.import_module(module_name)
    assert sys.modules[module_name] is module, module_name

for name in matview.__all__:
    assert getattr(facade, name) is getattr(matview, name), name
"""
        repo_root = Path(__file__).resolve().parent
        for order in ("facade-first", "package-first"):
            with self.subTest(import_order=order):
                result = subprocess.run(
                    [sys.executable, "-c", script, order],
                    cwd=repo_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_output_matches_legacy_api_plan_across_dialects(self) -> None:
        schema_sql = """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            category VARCHAR(32) NOT NULL,
            amount INTEGER NOT NULL
        );
        """
        view_sql = """
        CREATE VIEW order_totals AS
        SELECT category, COUNT(*) AS row_count, SUM(amount) AS total_amount
        FROM orders
        GROUP BY category;
        """
        # Exercise three materially different CLI modes: ordinary SQLite
        # population, PostgreSQL incremental backfill, and MySQL no-populate.
        # MySQL has no AFTER TRUNCATE trigger, so every MySQL view needs the
        # global --ignore-limitations (MYSQL_NO_TRUNCATE_TRIGGER); the CLI and
        # the API must then emit the same limitation warning.
        cases = (
            ("sqlite:3.46.0", False, False, False),
            ("postgres:18.0", True, False, False),
            ("mysql:8.4.0", False, True, True),
        )
        for engine_spec, incremental, no_populate, ignore_limitations in cases:
            with self.subTest(
                engine=engine_spec,
                incremental_backfill=incremental,
                no_populate=no_populate,
                ignore_limitations=ignore_limitations,
            ):
                engine = ms.parse_engine_version(engine_spec)
                schemas = ms.parse_table_schema(schema_sql, dialect=engine.dialect)
                with warnings.catch_warnings(record=True) as api_warnings:
                    warnings.simplefilter("always")
                    api_plan = ms.generate_materialized_view(
                        schemas,
                        view_sql,
                        dialect=engine.dialect,
                        engine_version=engine,
                        object_prefix="compat_",
                        incremental_backfill=incremental,
                        batch_size=7,
                        ignore_limitations=ignore_limitations,
                    )
                expected_sql = api_plan.render_sql_script(
                    include_populate=not no_populate,
                    incremental_backfill=incremental,
                    batch_size=7,
                )

                argv = [
                    "--schema",
                    schema_sql,
                    "--view",
                    view_sql,
                    "--engine-version",
                    engine_spec,
                    "--object-prefix",
                    "compat_",
                    "--batch-size",
                    "7",
                ]
                if incremental:
                    argv.append("--incremental-backfill")
                if no_populate:
                    argv.append("--no-populate")
                if ignore_limitations:
                    argv.append("--ignore-limitations")

                stdout = io.StringIO()
                stderr = io.StringIO()
                with warnings.catch_warnings(record=True) as cli_warnings:
                    warnings.simplefilter("always")
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exit_code = matview.main(argv)

                self.assertEqual(exit_code, 0, stderr.getvalue())
                self.assertEqual(stderr.getvalue(), "")
                self.assertEqual(
                    [(w.category, str(w.message)) for w in cli_warnings],
                    [(w.category, str(w.message)) for w in api_warnings],
                )
                self.assertEqual(bool(api_warnings), ignore_limitations)
                # CLI output is print(plan.render_sql_script(...)), so include
                # exactly the one newline added by print().
                self.assertEqual(stdout.getvalue(), expected_sql + "\n")

    def test_cli_entrypoints_help(self) -> None:
        r1 = subprocess.run(
            [sys.executable, "matview_sync.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        r2 = subprocess.run(
            [sys.executable, "-m", "matview.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("--schema", r1.stdout)
        self.assertIn("--schema", r2.stdout)
        # argparse uses a different executable name for the two entrypoints;
        # compare the actual advertised option surface rather than the usage prefix.
        def long_options(output: str) -> set[str]:
            return set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", output))

        self.assertEqual(long_options(r1.stdout), long_options(r2.stdout))


if __name__ == "__main__":
    unittest.main()
