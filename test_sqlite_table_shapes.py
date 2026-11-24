#!/usr/bin/env python3
"""Shape of the SQLite tables matview creates: WITHOUT ROWID where nothing needs the rowid.

Rule (matview.dialects.base, measured in docs/design-sqlite-strict.md): a table is WITHOUT ROWID
when it has a NOT NULL PRIMARY KEY that is not already the rowid (a single `INTEGER PRIMARY KEY`),
no AUTOINCREMENT, and narrow rows (estimated <= 128 bytes); otherwise it keeps its rowid and the
registration says why. Exactness over every shape is covered by test_sqlite_strict_sources.py
(plain/STRICT sources x default/STRICT backing) and by the fuzzer; this file pins the rule itself.
"""

from __future__ import annotations

import unittest

from matview import Dialect, generate_materialized_view
from matview.sql_frontend import parse_table_schema

SCHEMA = """
CREATE TABLE narrow (id INTEGER PRIMARY KEY, g TEXT NOT NULL, v INTEGER);
CREATE TABLE wide (id INT NOT NULL PRIMARY KEY, a TEXT, b TEXT, c TEXT, d TEXT, e TEXT);
CREATE TABLE nopk (a INTEGER, b TEXT);
CREATE TABLE comp (x INTEGER NOT NULL, y INTEGER NOT NULL, a TEXT, b TEXT, c TEXT, d TEXT, e TEXT, PRIMARY KEY (x, y));
"""

# (view select, substring expected in the backing DDL or None, reason substring when the rowid is kept)
CASES = [
    ("SELECT g, COUNT(*) AS n, SUM(v) AS s FROM narrow GROUP BY g", "WITHOUT ROWID", None),  # __mv_pk TEXT / group key
    ("SELECT COUNT(*) AS n FROM narrow", None, "already is the rowid"),  # __mv_global_id INTEGER PRIMARY KEY
    ("SELECT id, g FROM narrow", "WITHOUT ROWID", None),  # projection keyed by the (non-alias) copy of the PK
    ("SELECT id, a, b, c, d, e FROM wide", None, "wide"),  # 8 + 5 * 32 bytes estimated > 128
    ("SELECT id, a FROM wide", "WITHOUT ROWID", None),
    ("SELECT a, b FROM nopk", None, "no PRIMARY KEY"),
    ("SELECT x, y, a, b, c, d, e FROM comp", None, "wide"),  # a composite key no longer forces WITHOUT ROWID on wide rows
    ("SELECT x, y, COUNT(*) AS n FROM comp GROUP BY x, y", "WITHOUT ROWID", None),
]


class SqliteTableShapeTests(unittest.TestCase):
    def test_without_rowid_rule(self) -> None:
        schemas = parse_table_schema(SCHEMA, dialect=Dialect.SQLITE)
        for i, (select, expect_wor, reason) in enumerate(CASES):
            plan = generate_materialized_view(schemas, f"CREATE VIEW sh{i} AS {select};", dialect=Dialect.SQLITE)
            ddl = plan.backing_table_ddl
            notes = [n for n in plan.diagnostics.details["performance_notes"] if n.startswith("sqlite_table_shape")]
            if expect_wor:
                self.assertIn("WITHOUT ROWID", ddl, select)
                self.assertEqual([n for n in notes if "keeps its rowid" in n], [], select)
            else:
                self.assertNotIn("WITHOUT ROWID", ddl, select)
                self.assertTrue(any("keeps its rowid" in n and reason in n for n in notes), (select, notes))

    def test_strict_backing_is_opt_in_and_per_table_provable(self) -> None:
        schemas = parse_table_schema(SCHEMA, dialect=Dialect.SQLITE)
        agg = "CREATE VIEW sa AS SELECT g, COUNT(*) AS n, MIN(v) AS m FROM narrow GROUP BY g;"
        proj = "CREATE VIEW sp AS SELECT id, g FROM narrow;"
        default = generate_materialized_view(schemas, agg, dialect=Dialect.SQLITE)
        self.assertNotIn("STRICT", default.backing_table_ddl)
        opted = generate_materialized_view(schemas, agg, dialect=Dialect.SQLITE, sqlite_strict_backing=True)
        self.assertTrue(opted.backing_table_ddl.rstrip(";").endswith("STRICT"), opted.backing_table_ddl)
        self.assertIn(" ANY", opted.backing_table_ddl)  # BLOB (no affinity) <-> ANY
        # a column copied from a base table keeps the base column's affinity: not provable, stays plain, announced
        copied = generate_materialized_view(schemas, proj, dialect=Dialect.SQLITE, sqlite_strict_backing=True)
        self.assertNotIn("STRICT", copied.backing_table_ddl)
        self.assertTrue(
            any("not STRICT although strict backing tables were requested" in n
                for n in copied.diagnostics.details["performance_notes"])
        )

    def test_opt_out_and_version_gate_and_unbounded_note(self) -> None:
        schemas = parse_table_schema(SCHEMA, dialect=Dialect.SQLITE)
        agg = "CREATE VIEW oa AS SELECT g, COUNT(*) AS n FROM narrow GROUP BY g;"
        default = generate_materialized_view(schemas, agg, dialect=Dialect.SQLITE)
        self.assertIn("WITHOUT ROWID", default.backing_table_ddl)
        # the width estimate cannot see value sizes: the note says so and names the opt-out
        notes = default.diagnostics.details["performance_notes"]
        self.assertTrue(any("estimated row width" in n and "sqlite_without_rowid=False" in n for n in notes), notes)
        off = generate_materialized_view(schemas, agg, dialect=Dialect.SQLITE, sqlite_without_rowid=False)
        self.assertNotIn("WITHOUT ROWID", off.backing_table_ddl)
        self.assertTrue(any("keeps its rowid: WITHOUT ROWID was disabled" in n for n in off.diagnostics.details["performance_notes"]))
        # STRICT needs SQLite 3.37: an older target falls back, announced, instead of emitting invalid DDL
        old = generate_materialized_view(
            schemas, agg, dialect=Dialect.SQLITE, engine_version="sqlite:3.30.0", sqlite_strict_backing=True
        )
        self.assertNotIn("STRICT", old.backing_table_ddl)
        self.assertTrue(any("STRICT tables need SQLite 3.37" in n for n in old.diagnostics.details["performance_notes"]))
        new = generate_materialized_view(
            schemas, agg, dialect=Dialect.SQLITE, engine_version="sqlite:3.45.1", sqlite_strict_backing=True
        )
        self.assertIn("STRICT", new.backing_table_ddl)


if __name__ == "__main__":
    unittest.main()
