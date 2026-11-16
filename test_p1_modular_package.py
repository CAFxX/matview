#!/usr/bin/env python3
"""Tests for the modular `matview` package architecture, immutable IR, and CLI/facade parity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
import subprocess
import sys
import unittest

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
        for mod_name in modules:
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


if __name__ == "__main__":
    unittest.main()
