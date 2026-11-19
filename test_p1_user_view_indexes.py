#!/usr/bin/env python3
"""User-declared indexes on materialized views (CREATE INDEX statements and ``indexes=``).

Contract under test (README "CREATE INDEX statements on the view"): an index
declared on a view is accepted by every supported input form, validated against
the view's exposed columns, and created on the view's backing table with its
declared columns, uniqueness, and partial-index predicate.

Regression origin: pyright found that ``PlannerMixin._extract_index_col_base_ident``
was a ``@staticmethod`` that referenced ``self`` (NameError for every view with a
user index) and that CREATE INDEX statements were parsed through a non-existent
``IndexParameters.columns`` attribute (AttributeError). Neither form was covered
by any test, so declaring a view index failed on every engine.
"""

from __future__ import annotations

import contextlib
import sqlite3
import unittest

import matview_sync as ms

SCHEMA = "CREATE TABLE t (id INTEGER PRIMARY KEY, g TEXT, v INTEGER);"
VIEW = "CREATE VIEW mv AS SELECT g, SUM(v) AS s, COUNT(*) AS n FROM t GROUP BY g"
ALL_DIALECTS = ("sqlite", "postgres", "mysql")


def _plan(tc: unittest.TestCase, dialect: str, view_sql: str = VIEW, **kw):
    # MySQL refuses the view without an explicit override (no AFTER TRUNCATE
    # trigger); that limitation is unrelated to index handling, and the
    # override's documented warning is asserted rather than suppressed.
    warned = (
        tc.assertWarns(ms.EngineLimitationWarning) if dialect == "mysql" else contextlib.nullcontext()
    )
    with warned:
        return ms.generate_materialized_view(
            SCHEMA, view_sql, dialect=dialect, ignore_limitations=(dialect == "mysql"), **kw
        )


class TestUserIndexInputForms(unittest.TestCase):
    def test_create_index_statement_on_every_dialect(self) -> None:
        sql = VIEW + "; CREATE UNIQUE INDEX ix_s ON mv (s DESC, g) WHERE s > 0;"
        for dialect in ALL_DIALECTS:
            with self.subTest(dialect=dialect):
                plan = _plan(self, dialect, sql)
                self.assertEqual(
                    list(plan.user_indexes),
                    [ms.ViewIndexSpec(columns=("s DESC", "g"), name="ix_s", unique=True,
                                      where_clause="s > 0", using_method=None, target_view="mv")],
                )
                ddl = [d for d in plan.index_ddls if "ix_s" in d]
                self.assertEqual(len(ddl), 1, plan.index_ddls)
                self.assertIn("UNIQUE INDEX", ddl[0])
                self.assertIn("(s DESC, g)", ddl[0])

    def test_indexes_argument_forms_agree(self) -> None:
        # str shorthand, column sequence, and dict spec all describe the same index.
        forms = ["g", ["g"], {"columns": ["g"]}]
        results = []
        for form in forms:
            plan = _plan(self, "sqlite", indexes=[form])
            self.assertEqual([tuple(u.columns) for u in plan.user_indexes], [("g",)], form)
            results.append(sorted(d for d in plan.index_ddls if "ON __matview_store_mv (g)" in d
                                  and "_grp" not in d))
        self.assertTrue(all(r == results[0] and len(r) == 1 for r in results), results)

    def test_extract_index_col_base_ident_strips_ordering_and_quoting(self) -> None:
        comp = ms.IncrementalViewCompiler(
            ms.parse_table_schema(SCHEMA), ms.parse_view_definition(VIEW, known_tables={"t"})
        )
        cases = {
            "g": "g",
            "s DESC": "s",
            '"g" ASC NULLS LAST': "g",
            "g COLLATE NOCASE": "g",
        }
        for spec, expected in cases.items():
            with self.subTest(spec=spec):
                self.assertEqual(comp._extract_index_col_base_ident(spec), expected)

    def test_index_on_unknown_column_is_rejected_not_crashed(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _plan(self, "sqlite", indexes=["no_such_col"])
        self.assertIn("no_such_col", str(ctx.exception))


class TestUserIndexesOnLiveSQLite(unittest.TestCase):
    """End to end on a real SQLite connection: the declared indexes exist and data stays correct."""

    def test_declared_indexes_are_created_and_view_stays_consistent(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        mgr = ms.MaterializedViewManager(conn)
        mgr.apply_schema_ddl(SCHEMA, auto_sync_views=False)
        mgr.register_view(
            VIEW + "; CREATE UNIQUE INDEX ix_g ON mv (g); CREATE INDEX ix_s ON mv (s DESC) WHERE s > 0;",
            populate=True,
        )
        idx = dict(conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = '__matview_store_mv'"
        ).fetchall())
        self.assertEqual(idx["__matview_ix_g"], "CREATE UNIQUE INDEX __matview_ix_g ON __matview_store_mv (g)")
        self.assertEqual(
            idx["__matview_ix_s"],
            "CREATE INDEX __matview_ix_s ON __matview_store_mv (s DESC) WHERE s > 0",
        )
        conn.executemany("INSERT INTO t VALUES (?, ?, ?)", [(1, "a", 5), (2, "a", 7), (3, "b", -1)])
        conn.commit()
        self.assertEqual(conn.execute("SELECT g, s, n FROM mv ORDER BY g").fetchall(),
                         [("a", 12, 2), ("b", -1, 1)])
        conn.execute("DELETE FROM t WHERE id = 2")
        conn.commit()
        self.assertEqual(conn.execute("SELECT g, s, n FROM mv ORDER BY g").fetchall(),
                         [("a", 5, 1), ("b", -1, 1)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
