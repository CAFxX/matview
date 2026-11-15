#!/usr/bin/env python3
"""Verify SQLite feature gates against the actual linked runtime."""

from __future__ import annotations

import sqlite3
import unittest

from matview_sync import (
    Dialect,
    EngineSubdialect,
    EngineVersion,
    detect_engine_version,
    parse_engine_version,
)


class SQLiteRuntimeBoundaryTests(unittest.TestCase):
    def test_connection_detection_uses_the_live_sqlite_runtime(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            live_version = conn.execute("SELECT sqlite_version();").fetchone()[0]
            detected = detect_engine_version(conn)
            self.assertEqual(detected.dialect, Dialect.SQLITE)
            self.assertEqual(detected.subdialect, EngineSubdialect.SQLITE)
            self.assertEqual(detected.version, tuple(map(int, live_version.split("."))))
        finally:
            conn.close()

    def test_runtime_gated_sql_features_execute_when_reported_supported(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            detected = detect_engine_version(conn)
            self.assertEqual(detected.version, sqlite3.sqlite_version_info[:3])
            conn.execute(
                "CREATE TABLE upsert_probe (k INTEGER PRIMARY KEY, value INTEGER NOT NULL);"
            )
            conn.execute("INSERT INTO upsert_probe VALUES (1, 10);")

            if detected.supports_sqlite_upsert:
                conn.execute(
                    "INSERT INTO upsert_probe VALUES (1, 20) "
                    "ON CONFLICT(k) DO UPDATE SET value = excluded.value;"
                )
                self.assertEqual(
                    conn.execute("SELECT value FROM upsert_probe WHERE k = 1;").fetchone(),
                    (20,),
                )
            else:
                self.fail(f"SQLite {detected.version_str} is below the UPSERT feature floor")

            if detected.supports_sqlite_window_functions:
                rows = conn.execute(
                    "SELECT k, ROW_NUMBER() OVER (ORDER BY k) FROM upsert_probe;"
                ).fetchall()
                self.assertEqual(rows, [(1, 1)])
            else:
                self.fail(f"SQLite {detected.version_str} is below the window-function floor")

            if detected.supports_sqlite_returning:
                returned = conn.execute(
                    "INSERT INTO upsert_probe VALUES (2, 30) RETURNING k, value;"
                ).fetchall()
                self.assertEqual(returned, [(2, 30)])
        finally:
            conn.close()

    def test_sqlite_feature_version_floors_are_exact(self) -> None:
        cases = (
            ("sqlite:3.23.1", False, False, False),
            ("sqlite:3.24.0", True, False, False),
            ("sqlite:3.24.9", True, False, False),
            ("sqlite:3.25.0", True, True, False),
            ("sqlite:3.34.9", True, True, False),
            ("sqlite:3.35.0", True, True, True),
        )
        for spec, upsert, windows, returning in cases:
            with self.subTest(version=spec):
                version: EngineVersion = parse_engine_version(spec)
                self.assertEqual(version.supports_sqlite_upsert, upsert)
                self.assertEqual(version.supports_sqlite_window_functions, windows)
                self.assertEqual(version.supports_sqlite_returning, returning)


if __name__ == "__main__":
    unittest.main(verbosity=2)
