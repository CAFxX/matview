#!/usr/bin/env python3
"""STRICT-aware diagnostics, introspection and drift on SQLite (>= 3.37).

* A non-STRICT source table that costs the view something gets a NOTE
  (`SQLITE_STRICT_RECOMMENDED`): in `plan.strategy_diagnostics`, in the CLI diff
  text/JSON, as a log record, and never as a Python warning (a recommendation
  changes no guarantee). STRICT sources get no note.
* STRICT-ness comes from `PRAGMA table_list` and is part of the registered
  signature: a table rebuilt in the other shape is reported as drift.
"""

from __future__ import annotations

import json
import sqlite3
import unittest
import warnings
from typing import Any, Dict, List

from matview import Dialect, MaterializedViewManager
from matview.connectivity import introspect_sqlite_schemas
from matview.diagnostics import SQLITE_STRICT_RECOMMENDED, DiagnosticSeverity, NOTE_LEAD

STRICT_OK = sqlite3.sqlite_version_info >= (3, 37, 0)
AGG_VIEW = "CREATE VIEW sv AS SELECT g, SUM(q) AS s, AVG(q) AS a, COUNT(*) AS c FROM o GROUP BY g;"


def _conn(strict: bool, extra: str = "") -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute(f"CREATE TABLE o (id INTEGER PRIMARY KEY, g TEXT, q INTEGER, p REAL){' STRICT' if strict else ''}{extra}")
    c.execute("INSERT INTO o VALUES (1,'a',1,0.5),(2,'a',2,1.5)")
    c.commit()
    return c


def _notes(plan: Any) -> List[Any]:
    return [d for d in plan.strategy_diagnostics if d.code == SQLITE_STRICT_RECOMMENDED]


@unittest.skipUnless(STRICT_OK, "STRICT tables need SQLite >= 3.37")
class SqliteStrictDiagnosticsTests(unittest.TestCase):
    def test_note_for_plain_source_none_for_strict_and_never_a_warning(self) -> None:
        for strict in (False, True):
            conn = _conn(strict)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    plan = MaterializedViewManager(conn, dialect=Dialect.SQLITE).register_view(AGG_VIEW)
                self.assertEqual([w for w in caught if "STRICT" in str(w.message)], [], "a recommendation must not warn")
                notes = _notes(plan)
                if strict:
                    self.assertEqual(notes, [])
                    continue
                self.assertEqual(len(notes), 1)
                n = notes[0]
                self.assertIs(n.severity, DiagnosticSeverity.NOTE)
                self.assertTrue(n.message.startswith(NOTE_LEAD))
                self.assertEqual(n.columns, ("q",))  # only the integer-affinity SUM/AVG argument, not the REAL one
                for needle in ("source table o", "q INT", "STRICT", "12-step", "3.37", "never a wrong result"):
                    self.assertIn(needle, n.message)
                self.assertEqual(plan.diagnostics.details["recommendations"][0]["code"], SQLITE_STRICT_RECOMMENDED)
            finally:
                conn.close()

    def test_no_note_when_nothing_depends_on_the_type(self) -> None:
        conn = _conn(False)
        try:
            plan = MaterializedViewManager(conn, dialect=Dialect.SQLITE).register_view(
                "CREATE VIEW pv AS SELECT g, COUNT(*) AS c, MAX(p) AS m FROM o GROUP BY g;"
            )
            self.assertEqual(_notes(plan), [])
        finally:
            conn.close()

    def test_note_follows_decomposed_aggregates_and_skips_distinct(self) -> None:
        cases = {
            "SELECT g, COALESCE(SUM(q), 0) AS s FROM o GROUP BY g": 1,  # compound expr over an elementary SUM
            "SELECT g, SUM(DISTINCT q) AS s FROM o GROUP BY g": 0,  # DISTINCT always recomputes its group
        }
        for select, expected in cases.items():
            conn = _conn(False)
            try:
                plan = MaterializedViewManager(conn, dialect=Dialect.SQLITE).register_view(f"CREATE VIEW dv AS {select};")
                self.assertEqual(len(_notes(plan)), expected, select)
            finally:
                conn.close()

    def test_watermark_note_on_plain_without_rowid_and_not_on_strict(self) -> None:
        view = "CREATE VIEW wv AS SELECT g, COUNT(*) AS c FROM w GROUP BY g;"
        out: Dict[bool, Any] = {}
        for strict in (False, True):
            conn = sqlite3.connect(":memory:")
            conn.execute(
                "CREATE TABLE w (id INTEGER NOT NULL PRIMARY KEY, g TEXT) WITHOUT ROWID" + (", STRICT" if strict else "")
            )
            conn.execute("INSERT INTO w VALUES (1,'a'),(2,'b')")
            conn.commit()
            try:
                plan = MaterializedViewManager(conn, dialect=Dialect.SQLITE).register_view(
                    view, incremental_backfill=True
                )
                out[strict] = (_notes(plan), plan.incremental_backfill_spec.mode if plan.incremental_backfill_spec else None)
            finally:
                conn.close()
        self.assertEqual(len(out[False][0]), 1)
        self.assertIn("group_key_batch", out[False][0][0].message)
        self.assertEqual(out[False][1], "group_key_batch")
        self.assertEqual(out[True], ([], "pk_range_fold"))

    def test_introspection_reads_strict_and_without_rowid_from_the_live_connection(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            for name, tail in (("p", ""), ("s", " STRICT"), ("w", " WITHOUT ROWID"), ("ws", " WITHOUT ROWID, STRICT")):
                conn.execute(f"CREATE TABLE {name} (id INTEGER NOT NULL PRIMARY KEY, v TEXT){tail}")
            got = {n: (ts.strict, ts.without_rowid) for n, ts in introspect_sqlite_schemas(conn).items()}
            self.assertEqual(got, {"p": (False, False), "s": (True, False), "w": (False, True), "ws": (True, True)})
        finally:
            conn.close()

    def test_strictness_change_after_registration_is_drift_and_repair_restores_ok(self) -> None:
        for to_strict in (False, True):
            conn = _conn(not to_strict)
            try:
                mgr = MaterializedViewManager(conn, dialect=Dialect.SQLITE)
                mgr.register_view(AGG_VIEW)
                self.assertEqual(mgr.check_schema_drift(), {"sv": "ok"})
                # the 12-step rebuild into the other shape (foreign keys off, as SQLite documents)
                tail = " STRICT" if to_strict else ""
                # (the maintenance triggers go away with the table, as in any out-of-band rebuild)
                conn.execute(f"CREATE TABLE o2 (id INTEGER PRIMARY KEY, g TEXT, q INTEGER, p REAL){tail}")
                conn.execute("INSERT INTO o2 SELECT * FROM o")
                conn.execute("DROP TABLE o")
                conn.execute("ALTER TABLE o2 RENAME TO o")
                conn.commit()
                self.assertNotEqual(mgr.check_schema_drift()["sv"], "ok", f"to_strict={to_strict}")
                mgr.repair_view("sv")
                self.assertEqual(mgr.check_schema_drift(), {"sv": "ok"})
                self.assertEqual(
                    sorted(conn.execute("SELECT * FROM sv").fetchall()), sorted(conn.execute(
                        "SELECT g, SUM(q), AVG(q), COUNT(*) FROM o GROUP BY g").fetchall())
                )
            finally:
                conn.close()

    def test_cli_diff_text_and_json_carry_the_recommendation(self) -> None:
        conn = _conn(False)
        try:
            report = MaterializedViewManager(conn, dialect=Dialect.SQLITE).diff_views(
                [AGG_VIEW], schema_sql="CREATE TABLE o (id INTEGER PRIMARY KEY, g TEXT, q INTEGER, p REAL);"
            )
            text = report.render_text()
            self.assertIn("Recommendations (1):", text)
            self.assertIn(SQLITE_STRICT_RECOMMENDED, text)
            data = json.loads(report.to_json())
            recs = data["views"]["sv"]["diagnostics"]["details"]["recommendations"]
            self.assertEqual(recs[0]["severity"], "note")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
