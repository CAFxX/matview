#!/usr/bin/env python3
"""
test_p1_packaging_version.py — Verification suite for Standardized Engine Version Parsing (`packaging`)

Tests delegation of semantic version parsing, normalization, ordering, and comparison
to `packaging.version.Version` in `EngineVersion`, `parse_engine_version`, and `detect_engine_version`:
  1. Delegation to `packaging.version.Version` (type, properties, backward-compatible tuple & attributes).
  2. PEP 440 pre-releases, release candidates, post-releases, dev-releases, and base versions.
  3. Database vendor banner and version string extractions:
     - PostgreSQL server_version_num (e.g. 170002 -> 17.0.2, 90624 -> 9.6.24, 140005 -> 14.0.5)
     - MariaDB banners (e.g. 5.5.5-10.11.8-MariaDB-1:10.11.8+maria~ubu2204, 11.4.3-MariaDB-ubu2404)
     - Ubuntu/Debian distribution build suffixes (e.g. 8.4.3-0ubuntu0, 8.4.3-0ubuntu0.24.04.1)
     - MySQL (e.g. mysql:9.7.0, mysql:8.4.2, mysql:5.7.44)
     - SQLite (e.g. sqlite:3.46.1, sqlite:3.20.0)
     - Verbose live server strings (e.g. 'PostgreSQL 17.0 on x86_64-pc-linux-gnu, compiled by gcc...')
  4. Rich comparisons, total ordering, sorting, and hashability of `EngineVersion`.
  5. Semantic floor checking via `ev.at_least(...)` across `Version`, strings, tuples, and `EngineVersion`.
  6. Feature flag properties delegating to `at_least`.
  7. Explicit `ValueError` on invalid version specifications.
  8. CLI `--engine-version` integration.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from unittest.mock import MagicMock

from packaging.version import Version

from matview_sync import (
    Dialect,
    EngineSubdialect,
    EngineVersion,
    main as cli_main,
    detect_engine_version,
    parse_engine_version,
    _extract_packaging_and_tuple,
    _extract_version_tuple,
    _to_packaging_version,
)


class TestPackagingVersionDelegation(unittest.TestCase):
    """Tests that EngineVersion delegates semantic version parsing to packaging.version.Version."""

    def test_packaging_version_attribute_and_type(self) -> None:
        ev = parse_engine_version("postgres:17.2")
        self.assertIsInstance(ev.packaging_version, Version)
        self.assertEqual(ev.packaging_version, Version("17.2.0"))
        self.assertEqual(ev.packaging_version.major, 17)
        self.assertEqual(ev.packaging_version.minor, 2)
        self.assertEqual(ev.packaging_version.micro, 0)

    def test_backward_compatible_attributes(self) -> None:
        ev = parse_engine_version("postgres:17.2.1")
        # Legacy .version tuple
        self.assertEqual(ev.version, (17, 2, 1))
        # Legacy major/minor/patch properties
        self.assertEqual(ev.major, 17)
        self.assertEqual(ev.minor, 2)
        self.assertEqual(ev.patch, 1)
        # Legacy version_str and label
        self.assertEqual(ev.version_str, "17.2.1")
        self.assertEqual(ev.label, "postgres 17.2.1")
        # PostgreSQL version number (major * 10000 + minor * 100 + patch)
        self.assertEqual(ev.pg_version_num, 170201)

    def test_pep440_prerelease_and_postrelease_properties(self) -> None:
        ev_rc = parse_engine_version("mysql:8.4.0rc1")
        self.assertTrue(ev_rc.is_prerelease)
        self.assertFalse(ev_rc.is_postrelease)
        self.assertFalse(ev_rc.is_devrelease)
        self.assertEqual(ev_rc.packaging_version.pre, ("rc", 1))
        self.assertEqual(ev_rc.base_version, "8.4.0")
        self.assertEqual(ev_rc.version_str, "8.4.0rc1")
        self.assertEqual(ev_rc.version, (8, 4, 0))

        ev_post = parse_engine_version("10.11.8.post1", default_dialect=Dialect.MYSQL)
        self.assertFalse(ev_post.is_prerelease)
        self.assertTrue(ev_post.is_postrelease)
        self.assertFalse(ev_post.is_devrelease)
        self.assertEqual(ev_post.packaging_version.post, 1)
        self.assertEqual(ev_post.base_version, "10.11.8")
        self.assertEqual(ev_post.version_str, "10.11.8.post1")
        self.assertEqual(ev_post.version, (10, 11, 8))

        ev_dev = parse_engine_version("postgres:17.0.0.dev1")
        self.assertTrue(ev_dev.is_devrelease)
        self.assertFalse(ev_dev.is_postrelease)
        self.assertEqual(ev_dev.base_version, "17.0.0")

    def test_direct_engine_version_construction(self) -> None:
        # Constructing EngineVersion directly without passing packaging_version derives it automatically
        ev = EngineVersion(Dialect.SQLITE, EngineSubdialect.SQLITE, (3, 45, 2))
        self.assertIsInstance(ev.packaging_version, Version)
        self.assertEqual(ev.packaging_version, Version("3.45.2"))
        self.assertEqual(ev.version_str, "3.45.2")
        self.assertEqual(ev.major, 3)
        self.assertEqual(ev.minor, 45)
        self.assertEqual(ev.patch, 2)


class TestVendorBannersAndEdgeCases(unittest.TestCase):
    """Tests parsing of real-world database server banners, package strings, and numbering schemes."""

    def test_postgres_server_version_num(self) -> None:
        # 6-digit PostgreSQL version numbers (PG 10+)
        ev_17 = parse_engine_version("170002", default_dialect=Dialect.POSTGRES)
        self.assertEqual(ev_17.packaging_version, Version("17.0.2"))
        self.assertEqual(ev_17.version, (17, 0, 2))
        self.assertEqual(ev_17.pg_version_num, 170002)

        ev_14 = parse_engine_version("140005", default_dialect=Dialect.POSTGRES)
        self.assertEqual(ev_14.packaging_version, Version("14.0.5"))
        self.assertEqual(ev_14.version, (14, 0, 5))

        # 5-digit PostgreSQL version numbers (PG 9.x)
        ev_96 = parse_engine_version("90624", default_dialect=Dialect.POSTGRES)
        self.assertEqual(ev_96.packaging_version, Version("9.6.24"))
        self.assertEqual(ev_96.version, (9, 6, 24))
        self.assertEqual(ev_96.pg_version_num, 90624)

    def test_mariadb_banners(self) -> None:
        # Debian/Ubuntu MariaDB replication-compat banner with 5.5.5- prefix
        ev1 = parse_engine_version("5.5.5-10.11.8-MariaDB-1:10.11.8+maria~ubu2204")
        self.assertEqual(ev1.dialect, Dialect.MYSQL)
        self.assertEqual(ev1.subdialect, EngineSubdialect.MARIADB)
        self.assertEqual(ev1.packaging_version, Version("10.11.8"))
        self.assertEqual(ev1.version, (10, 11, 8))

        # Modern MariaDB banner without 5.5.5-
        ev2 = parse_engine_version("11.4.3-MariaDB-ubu2404")
        self.assertEqual(ev2.subdialect, EngineSubdialect.MARIADB)
        self.assertEqual(ev2.packaging_version, Version("11.4.3"))
        self.assertEqual(ev2.version, (11, 4, 3))

        # Pre-release MariaDB banner
        ev3 = parse_engine_version("11.5.1-rc-MariaDB")
        self.assertEqual(ev3.subdialect, EngineSubdialect.MARIADB)
        self.assertEqual(ev3.packaging_version, Version("11.5.1rc0"))
        self.assertTrue(ev3.is_prerelease)

    def test_ubuntu_debian_package_suffixes(self) -> None:
        # Ubuntu package build suffixes e.g. 8.4.3-0ubuntu0 or 8.4.3-0ubuntu0.24.04.1
        ev_ubu1 = parse_engine_version("mysql:8.4.3-0ubuntu0")
        self.assertEqual(ev_ubu1.packaging_version, Version("8.4.3"))
        self.assertEqual(ev_ubu1.version, (8, 4, 3))

        ev_ubu2 = parse_engine_version("mysql:8.4.3-0ubuntu0.24.04.1")
        self.assertEqual(ev_ubu2.packaging_version, Version("8.4.3"))
        self.assertEqual(ev_ubu2.version, (8, 4, 3))

        ev_ubu3 = parse_engine_version("8.0.35-0ubuntu0.22.04.1", default_dialect=Dialect.MYSQL)
        self.assertEqual(ev_ubu3.packaging_version, Version("8.0.35"))
        self.assertEqual(ev_ubu3.version, (8, 0, 35))

    def test_mysql_and_sqlite_versions(self) -> None:
        # MySQL 9.7 Innovation Release
        ev_my9 = parse_engine_version("mysql:9.7.0")
        self.assertEqual(ev_my9.packaging_version, Version("9.7.0"))
        self.assertEqual(ev_my9.version, (9, 7, 0))

        # SQLite 3.46.1
        ev_sq = parse_engine_version("sqlite:3.46.1")
        self.assertEqual(ev_sq.packaging_version, Version("3.46.1"))
        self.assertEqual(ev_sq.version, (3, 46, 1))

    def test_verbose_live_version_strings(self) -> None:
        pg_banner = "PostgreSQL 17.0 on x86_64-pc-linux-gnu, compiled by gcc (GCC) 14.2.1, 64-bit"
        ev_pg = parse_engine_version(pg_banner, default_dialect=Dialect.POSTGRES)
        self.assertEqual(ev_pg.packaging_version, Version("17.0.0"))
        self.assertEqual(ev_pg.version, (17, 0, 0))


class TestEngineVersionOrderingAndComparisons(unittest.TestCase):
    """Tests rich comparisons, sorting, total_ordering, and hashing of EngineVersion."""

    def test_pep440_ordering_distinguishes_prereleases(self) -> None:
        ev_dev = parse_engine_version("mysql:8.4.0.dev1")
        ev_b = parse_engine_version("mysql:8.4.0b2")
        ev_rc = parse_engine_version("mysql:8.4.0rc1")
        ev_final = parse_engine_version("mysql:8.4.0")
        ev_post = parse_engine_version("mysql:8.4.0.post1")

        # Crucial: standard tuples cannot distinguish pre-releases: (8, 4, 0) == (8, 4, 0).
        # EngineVersion uses packaging.version.Version, so it respects PEP 440 order:
        self.assertLess(ev_dev, ev_b)
        self.assertLess(ev_b, ev_rc)
        self.assertLess(ev_rc, ev_final)
        self.assertLess(ev_final, ev_post)

        self.assertGreater(ev_final, ev_rc)
        self.assertGreater(ev_post, ev_final)
        self.assertNotEqual(ev_rc, ev_final)

        # Sorting a list of versions
        unordered = [ev_post, ev_dev, ev_final, ev_b, ev_rc]
        ordered = sorted(unordered)
        self.assertEqual(ordered, [ev_dev, ev_b, ev_rc, ev_final, ev_post])

    def test_at_least_semantic_floor(self) -> None:
        ev = parse_engine_version("postgres:17.2.1")

        # Check against packaging.version.Version
        self.assertTrue(ev.at_least(Version("17.0.0")))
        self.assertTrue(ev.at_least(Version("17.2.0")))
        self.assertTrue(ev.at_least(Version("17.2.1")))
        self.assertFalse(ev.at_least(Version("17.3.0")))
        self.assertFalse(ev.at_least(Version("18.0.0")))

        # Check against string
        self.assertTrue(ev.at_least("15.0"))
        self.assertTrue(ev.at_least("17.2"))
        self.assertTrue(ev.at_least("17.2.1"))
        self.assertFalse(ev.at_least("17.2.2"))
        self.assertFalse(ev.at_least("18.0"))

        # Check against tuple
        self.assertTrue(ev.at_least((15, 0, 0)))
        self.assertTrue(ev.at_least((17, 2, 0)))
        self.assertFalse(ev.at_least((17, 3, 0)))

        # Check against another EngineVersion
        ev_pg15 = parse_engine_version("postgres:15.4")
        self.assertTrue(ev.at_least(ev_pg15))
        self.assertFalse(ev_pg15.at_least(ev))

    def test_at_least_with_prerelease(self) -> None:
        ev_rc = parse_engine_version("mysql:8.4.0rc1")
        # Pre-release is strictly less than the final release
        self.assertFalse(ev_rc.at_least("8.4.0"))
        self.assertTrue(ev_rc.at_least("8.4.0rc1"))
        self.assertTrue(ev_rc.at_least("8.0.0"))

    def test_hashability_in_sets_and_dicts(self) -> None:
        ev1 = parse_engine_version("postgres:17.2")
        ev2 = parse_engine_version("postgres:17.2")
        ev3 = parse_engine_version("postgres:17.3")

        self.assertEqual(hash(ev1), hash(ev2))
        s = {ev1, ev2, ev3}
        self.assertEqual(len(s), 2)

        mapping = {ev1: "v17.2"}
        self.assertEqual(mapping[ev2], "v17.2")


class TestEngineVersionFeatureFlags(unittest.TestCase):
    """Tests that EngineVersion feature flags delegate correctly to semantic floors."""

    def test_postgres_feature_flags(self) -> None:
        pg9 = parse_engine_version("postgres:9.6.24")
        pg10 = parse_engine_version("postgres:10.0")
        pg14 = parse_engine_version("postgres:14.5")
        pg15 = parse_engine_version("postgres:15.0")
        pg17 = parse_engine_version("postgres:17.2")

        # Statement-level maintenance: transition tables (PG 10) created with
        # EXECUTE FUNCTION (PG 11)
        self.assertFalse(pg9.supports_pg_statement_maintenance)
        self.assertFalse(pg10.supports_pg_statement_maintenance)
        self.assertTrue(parse_engine_version("postgres:11.0").supports_pg_statement_maintenance)
        self.assertTrue(pg14.supports_pg_statement_maintenance)
        self.assertTrue(pg17.supports_pg_statement_maintenance)

        # NULLS NOT DISTINCT (PG 15+)
        self.assertFalse(pg14.supports_pg_nulls_not_distinct)
        self.assertTrue(pg15.supports_pg_nulls_not_distinct)
        self.assertTrue(pg17.supports_pg_nulls_not_distinct)

        # MERGE statement (PG 15+)
        self.assertFalse(pg14.supports_pg_merge)
        self.assertTrue(pg15.supports_pg_merge)
        self.assertTrue(pg17.supports_pg_merge)

    def test_sqlite_feature_flags(self) -> None:
        sq320 = parse_engine_version("sqlite:3.20.0")
        sq324 = parse_engine_version("sqlite:3.24.0")
        sq325 = parse_engine_version("sqlite:3.25.0")
        sq335 = parse_engine_version("sqlite:3.35.0")

        # UPSERT (SQLite 3.24+)
        self.assertFalse(sq320.supports_sqlite_upsert)
        self.assertTrue(sq324.supports_sqlite_upsert)

        # Window functions (SQLite 3.25+)
        self.assertFalse(sq324.supports_sqlite_window_functions)
        self.assertTrue(sq325.supports_sqlite_window_functions)

        # RETURNING (SQLite 3.35+)
        self.assertFalse(sq325.supports_sqlite_returning)
        self.assertTrue(sq335.supports_sqlite_returning)

    def test_mysql_and_mariadb_feature_flags(self) -> None:
        my57 = parse_engine_version("mysql:5.7.44")
        my80 = parse_engine_version("mysql:8.0.0")
        my8019 = parse_engine_version("mysql:8.0.19")
        maria101 = parse_engine_version("mariadb:10.1.48")
        maria102 = parse_engine_version("mariadb:10.2.39")

        # Window functions (MySQL 8.0+, MariaDB 10.2+)
        self.assertFalse(my57.supports_mysql_window_functions)
        self.assertTrue(my80.supports_mysql_window_functions)
        self.assertFalse(maria101.supports_mysql_window_functions)
        self.assertTrue(maria102.supports_mysql_window_functions)

        # Row alias UPSERT (MySQL 8.0.19+)
        self.assertFalse(my80.supports_mysql_row_alias_upsert)
        self.assertTrue(my8019.supports_mysql_row_alias_upsert)


class TestInvalidVersionHandling(unittest.TestCase):
    """Tests that invalid version specifications raise explicit ValueError."""

    def test_invalid_version_strings_raise_value_error(self) -> None:
        invalid_specs = [
            "postgres:invalid",
            "mysql:--bad--",
            "sqlite:xyz",
            "postgres:",
            "mariadb:none",
            "invalid_version",  # with default_dialect
        ]
        for spec in invalid_specs:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    parse_engine_version(spec, default_dialect=Dialect.POSTGRES)

    def test_unknown_dialect_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_engine_version("oracle:19c")
        self.assertIn("Cannot determine database dialect", str(ctx.exception))


class TestCLIEngineVersionIntegration(unittest.TestCase):
    """Tests CLI invocation with --engine-version."""

    def test_cli_engine_version_valid(self) -> None:
        schema = "CREATE TABLE t (id INT PRIMARY KEY, v INT);"
        view = "CREATE VIEW v_t AS SELECT id, SUM(v) AS sv FROM t GROUP BY id;"
        # Valid --engine-version with PEP 440 pre-release
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = cli_main([
                "--schema", schema,
                "--view", view,
                "--engine-version", "mysql:8.4.0rc1",
                "--ignore-limitations",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("Engine: mysql 8.4.0rc1", out_buf.getvalue())

    def test_cli_engine_version_invalid(self) -> None:
        schema = "CREATE TABLE t (id INT PRIMARY KEY, v INT);"
        view = "CREATE VIEW v_t AS SELECT id, SUM(v) AS sv FROM t GROUP BY id;"
        # Invalid --engine-version should exit with code 2
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            rc = cli_main([
                "--schema", schema,
                "--view", view,
                "--engine-version", "postgres:invalid_version_tag",
            ])
        self.assertEqual(rc, 2)
        self.assertIn("ERROR: Invalid engine version specification", err_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
