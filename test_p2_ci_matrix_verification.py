#!/usr/bin/env python3
"""Focused semantic tests verifying CI matrix variables, engine verification, and verify_engine_matrix.py.

Addresses Codebase Audit P2 Findings:
- Finding #14: .github/workflows/ci.yml sqlite-compat matrix missing python-version definition.
- Finding #15: .github/workflows/ci-deferred.yml matrix jobs omitted verify_engine_matrix.py.
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_DEFERRED_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-deferred.yml"

# Import helper functions from verify_engine_matrix
from verify_engine_matrix import (
    _mysql_version_pair,
    check_mysql_version,
    check_postgres_version,
    verify_mysql,
    verify_postgres,
)


def _parse_yaml_minimal(text: str) -> Dict[str, Any]:
    """Parse YAML content using PyYAML if available or fallback stdlib parser."""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except Exception:
        pass

    # Fallback to test_strength_reporting's _stdlib_yaml_parse
    from test_strength_reporting import _stdlib_yaml_parse
    return _stdlib_yaml_parse(text)


def validate_ci_fuzz_target_matrix_variables(workflow_data: Dict[str, Any]) -> List[str]:
    """Validate that every ${{ matrix.* }} in MATVIEW_FUZZ_TARGET is defined in the job's matrix."""
    errors: List[str] = []
    jobs = workflow_data.get("jobs", {})

    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue

        strategy = job.get("strategy", {})
        matrix_def = strategy.get("matrix", {})
        if not isinstance(matrix_def, dict):
            continue

        # Global matrix keys (e.g. python-version: ["3.11", "3.15"])
        root_keys = set(matrix_def.keys()) - {"include", "exclude"}

        # Include row keys
        includes = matrix_def.get("include", [])
        if includes and isinstance(includes, list):
            include_row_keys = [set(inc.keys()) for inc in includes if isinstance(inc, dict)]
            if include_row_keys:
                # A variable is safe across all jobs only if defined in every include row or at root
                common_include_keys = set.intersection(*include_row_keys)
                all_defined_keys = root_keys | common_include_keys
            else:
                all_defined_keys = root_keys
        else:
            all_defined_keys = root_keys

        # Check every step in the job
        steps = job.get("steps", [])
        for step in steps:
            if not isinstance(step, dict):
                continue
            env_map = step.get("env", {})
            if not isinstance(env_map, dict):
                continue
            fuzz_target = env_map.get("MATVIEW_FUZZ_TARGET", "")
            if not fuzz_target:
                continue

            refs = re.findall(r"\$\{\{\s*matrix\.([a-zA-Z0-9_-]+)\s*\}\}", str(fuzz_target))
            for ref in refs:
                if ref not in all_defined_keys:
                    errors.append(
                        f"Job {job_id!r} step {step.get('name')!r} references 'matrix.{ref}' in "
                        f"MATVIEW_FUZZ_TARGET, but '{ref}' is not universally defined in matrix: {sorted(all_defined_keys)}"
                    )

    return errors


def validate_deferred_engine_verification(workflow_data: Dict[str, Any]) -> List[str]:
    """Validate that deferred database jobs execute verify_engine_matrix.py before tests with explicit version expectations."""
    errors: List[str] = []
    jobs = workflow_data.get("jobs", {})

    target_jobs = ["compatibility-sweep", "dependency-minimum"]
    for job_name in target_jobs:
        job = jobs.get(job_name)
        if not job or not isinstance(job, dict):
            errors.append(f"Missing required deferred job: {job_name!r}")
            continue

        steps = job.get("steps", [])
        verify_step_idx = -1
        test_step_idx = -1
        verify_step = None

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            run_cmd = step.get("run", "")
            if "verify_engine_matrix.py" in run_cmd:
                verify_step_idx = idx
                verify_step = step
            if "test_matview_sync.py" in run_cmd:
                test_step_idx = idx

        if verify_step is None:
            errors.append(f"Job {job_name!r} is missing verify_engine_matrix.py execution step")
            continue

        if test_step_idx >= 0 and verify_step_idx > test_step_idx:
            errors.append(f"Job {job_name!r} executes verify_engine_matrix.py at step index {verify_step_idx}, after test execution at {test_step_idx}")

        env = verify_step.get("env", {})
        if not isinstance(env, dict):
            errors.append(f"Job {job_name!r} verify step has non-dictionary env")
            continue

        # Must support explicit Postgres and MySQL expectations
        if "EXPECTED_PG_MAJOR" not in env:
            errors.append(f"Job {job_name!r} verify step missing EXPECTED_PG_MAJOR in env")
        if "EXPECTED_MYSQL_VERSION" not in env:
            errors.append(f"Job {job_name!r} verify step missing EXPECTED_MYSQL_VERSION in env")

        # Check matrix entries declare expected versions
        strategy = job.get("strategy", {})
        matrix_def = strategy.get("matrix", {})
        includes = matrix_def.get("include", [])
        for row in includes:
            if not isinstance(row, dict):
                continue
            kind = row.get("kind")
            name = row.get("name", "unnamed")
            if kind == "postgres":
                if "expected-pg-major" not in row:
                    errors.append(f"Job {job_name!r} matrix entry {name!r} missing 'expected-pg-major'")
            elif kind == "mysql":
                if "expected-product" not in row or "expected-version" not in row:
                    errors.append(f"Job {job_name!r} matrix entry {name!r} missing 'expected-product' or 'expected-version'")

    return errors


class TestCIMatrixVariableConsistency(unittest.TestCase):
    """Test suite verifying Finding #14: matrix definitions and MATVIEW_FUZZ_TARGET."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.ci_text = CI_WORKFLOW.read_text(encoding="utf-8")
        cls.ci_data = _parse_yaml_minimal(cls.ci_text)

    def test_sqlite_compat_matrix_defines_python_version(self) -> None:
        jobs = self.ci_data.get("jobs", {})
        self.assertIn("sqlite-compat", jobs, "ci.yml must define sqlite-compat job")
        compat_job = jobs["sqlite-compat"]
        matrix = compat_job.get("strategy", {}).get("matrix", {})
        includes = matrix.get("include", [])
        self.assertGreaterEqual(len(includes), 3, "sqlite-compat must have at least 3 implementations")

        for entry in includes:
            label = entry.get("label", "")
            self.assertIn(
                "python-version",
                entry,
                f"sqlite-compat include entry {label!r} must explicitly define python-version",
            )
            self.assertEqual(
                str(entry["python-version"]),
                "3.12",
                f"sqlite-compat include entry {label!r} python-version must be '3.12'",
            )

    def test_sqlite_compat_setup_python_wires_matrix_version(self) -> None:
        jobs = self.ci_data.get("jobs", {})
        compat_job = jobs["sqlite-compat"]
        setup_step = None
        for step in compat_job.get("steps", []):
            if "setup-python" in step.get("uses", ""):
                setup_step = step
                break

        self.assertIsNotNone(setup_step, "sqlite-compat must contain setup-python step")
        with_map = setup_step.get("with", {})
        py_ver = with_map.get("python-version", "")
        self.assertIn(
            "${{ matrix.python-version }}",
            py_ver,
            "setup-python step in sqlite-compat must reference ${{ matrix.python-version }}",
        )

    def test_every_fuzz_target_matrix_reference_is_defined_in_matrix(self) -> None:
        errors = validate_ci_fuzz_target_matrix_variables(self.ci_data)
        self.assertEqual(
            errors,
            [],
            "Every ${{ matrix.* }} in MATVIEW_FUZZ_TARGET in ci.yml must be defined in its job matrix:\n"
            + "\n".join(errors),
        )

    def test_validator_detects_undefined_matrix_variable_in_fuzz_target(self) -> None:
        broken_workflow = {
            "jobs": {
                "broken-job": {
                    "strategy": {
                        "matrix": {
                            "include": [
                                {"label": "Broken 1"},
                                {"label": "Broken 2"},
                            ]
                        }
                    },
                    "steps": [
                        {
                            "name": "Run suite",
                            "env": {
                                "MATVIEW_FUZZ_TARGET": "${{ matrix.label }} / Python ${{ matrix.python-version }}",
                            },
                            "run": "python3 test.py",
                        }
                    ],
                }
            }
        }
        errors = validate_ci_fuzz_target_matrix_variables(broken_workflow)
        self.assertTrue(
            any("python-version" in err for err in errors),
            f"Validator must detect missing matrix.python-version, got: {errors}",
        )

    def test_validator_detects_inconsistent_include_keys(self) -> None:
        broken_workflow = {
            "jobs": {
                "inconsistent-job": {
                    "strategy": {
                        "matrix": {
                            "include": [
                                {"label": "Row 1", "python-version": "3.12"},
                                {"label": "Row 2"},  # Missing python-version
                            ]
                        }
                    },
                    "steps": [
                        {
                            "name": "Run suite",
                            "env": {
                                "MATVIEW_FUZZ_TARGET": "${{ matrix.label }} / Python ${{ matrix.python-version }}",
                            },
                            "run": "python3 test.py",
                        }
                    ],
                }
            }
        }
        errors = validate_ci_fuzz_target_matrix_variables(broken_workflow)
        self.assertTrue(
            any("python-version" in err for err in errors),
            f"Validator must detect inconsistent include keys across rows: {errors}",
        )


class TestCIDeferredEngineVerification(unittest.TestCase):
    """Test suite verifying Finding #15: ci-deferred.yml engine verification."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.deferred_text = CI_DEFERRED_WORKFLOW.read_text(encoding="utf-8")
        cls.deferred_data = _parse_yaml_minimal(cls.deferred_text)

    def test_deferred_workflow_runs_verify_engine_matrix_across_all_live_jobs(self) -> None:
        errors = validate_deferred_engine_verification(self.deferred_data)
        self.assertEqual(
            errors,
            [],
            "ci-deferred.yml must verify native engine server versions before running tests:\n"
            + "\n".join(errors),
        )

    def test_compatibility_sweep_matrix_covers_explicit_versions(self) -> None:
        jobs = self.deferred_data.get("jobs", {})
        compat_job = jobs.get("compatibility-sweep", {})
        matrix_includes = compat_job.get("strategy", {}).get("matrix", {}).get("include", [])

        # PostgreSQL entries
        pg_entries = [e for e in matrix_includes if e.get("kind") == "postgres"]
        self.assertGreaterEqual(len(pg_entries), 3, "compatibility-sweep must test at least 3 PostgreSQL targets")
        for entry in pg_entries:
            self.assertIn("expected-pg-major", entry)
            self.assertIn(str(entry["expected-pg-major"]), {"16", "17", "18"})

        # MySQL-family entries
        mysql_entries = [e for e in matrix_includes if e.get("kind") == "mysql"]
        self.assertGreaterEqual(len(mysql_entries), 5, "compatibility-sweep must test at least 5 MySQL-family targets")
        for entry in mysql_entries:
            self.assertIn("expected-product", entry)
            self.assertIn("expected-version", entry)
            self.assertIn(entry["expected-product"], {"mysql", "mariadb", "percona"})

    def test_dependency_minimum_matrix_covers_explicit_versions(self) -> None:
        jobs = self.deferred_data.get("jobs", {})
        dep_job = jobs.get("dependency-minimum", {})
        matrix_includes = dep_job.get("strategy", {}).get("matrix", {}).get("include", [])
        self.assertEqual(len(matrix_includes), 2)

        kinds = {e.get("kind") for e in matrix_includes}
        self.assertEqual(kinds, {"postgres", "mysql"})
        for entry in matrix_includes:
            if entry["kind"] == "postgres":
                self.assertEqual(str(entry["expected-pg-major"]), "18")
            else:
                self.assertEqual(entry["expected-product"], "mysql")
                self.assertEqual(str(entry["expected-version"]), "9.7")

    def test_fuzz_stress_verifies_engine_version(self) -> None:
        jobs = self.deferred_data.get("jobs", {})
        fuzz_job = jobs.get("fuzz-stress", {})
        steps = fuzz_job.get("steps", [])
        verify_step = None
        for step in steps:
            if "verify_engine_matrix.py" in step.get("run", ""):
                verify_step = step
                break

        self.assertIsNotNone(verify_step, "fuzz-stress must contain verify_engine_matrix.py step")
        self.assertIn("EXPECTED_PG_MAJOR", verify_step.get("env", {}))
        self.assertIn("EXPECTED_MYSQL_VERSION", verify_step.get("env", {}))

    def test_validator_detects_omitted_verify_step(self) -> None:
        broken_workflow = {
            "jobs": {
                "compatibility-sweep": {
                    "strategy": {"matrix": {"include": [{"kind": "postgres", "name": "PG 16"}]}},
                    "steps": [{"name": "Run suite", "run": "python3 test_matview_sync.py"}],
                },
                "dependency-minimum": {
                    "strategy": {"matrix": {"include": [{"kind": "mysql", "name": "MySQL 9.7"}]}},
                    "steps": [{"name": "Run suite", "run": "python3 test_matview_sync.py"}],
                },
            }
        }
        errors = validate_deferred_engine_verification(broken_workflow)
        self.assertTrue(any("missing verify_engine_matrix.py" in err for err in errors))

    def test_validator_detects_misplaced_verify_step(self) -> None:
        broken_workflow = {
            "jobs": {
                "compatibility-sweep": {
                    "strategy": {"matrix": {"include": [{"kind": "postgres", "name": "PG 16", "expected-pg-major": "16"}]}},
                    "steps": [
                        {"name": "Run suite", "run": "python3 test_matview_sync.py"},
                        {"name": "Late verify", "run": "python3 verify_engine_matrix.py", "env": {"EXPECTED_PG_MAJOR": "16", "EXPECTED_MYSQL_VERSION": "8.4"}},
                    ],
                },
                "dependency-minimum": {
                    "strategy": {"matrix": {"include": [{"kind": "mysql", "name": "MySQL 9.7", "expected-product": "mysql", "expected-version": "9.7"}]}},
                    "steps": [
                        {"name": "Verify", "run": "python3 verify_engine_matrix.py", "env": {"EXPECTED_PG_MAJOR": "18", "EXPECTED_MYSQL_VERSION": "9.7"}},
                        {"name": "Run suite", "run": "python3 test_matview_sync.py"},
                    ],
                },
            }
        }
        errors = validate_deferred_engine_verification(broken_workflow)
        self.assertTrue(any("after test execution" in err for err in errors))


class TestVerifyEngineMatrixUnit(unittest.TestCase):
    """Unit tests for verify_engine_matrix.py logic without live database connections."""

    def test_postgres_version_matching_validates_cleanly(self) -> None:
        # Test PostgreSQL 15, 16, 17, 18
        cases = [
            ("PostgreSQL 15.6 on x86_64-pc-linux-gnu", 150006, 15),
            ("PostgreSQL 16.2 on x86_64-pc-linux-gnu", 160002, 16),
            ("PostgreSQL 17.0 on x86_64-pc-linux-gnu", 170000, 17),
            ("PostgreSQL 18.0 on x86_64-pc-linux-gnu", 180000, 18),
            # Percona Distribution for PostgreSQL 18
            ("PostgreSQL 18.1 (Percona Distribution) on x86_64-pc-linux-gnu", 180001, 18),
        ]
        for ver_text, ver_num, expected in cases:
            major = check_postgres_version(ver_text, ver_num, expected_major=expected)
            self.assertEqual(major, expected)
            # Also test string expected_major
            major_str = check_postgres_version(ver_text, ver_num, expected_major=str(expected))
            self.assertEqual(major_str, expected)

    def test_postgres_version_derivation_from_label_and_image(self) -> None:
        # Expected major is None or empty: derive from label or requested image
        major16 = check_postgres_version(
            "PostgreSQL 16.1", 160001, expected_major=None, label="PostgreSQL 16"
        )
        self.assertEqual(major16, 16)

        major17 = check_postgres_version(
            "PostgreSQL 17.0", 170000, expected_major="", requested_image="postgres:17"
        )
        self.assertEqual(major17, 17)

        major18 = check_postgres_version(
            "PostgreSQL 18.0", 180000, expected_major=None, requested_image="percona/percona-distribution-postgresql:18"
        )
        self.assertEqual(major18, 18)

    def test_postgres_version_mismatch_raises_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            check_postgres_version("PostgreSQL 15.6", 150006, expected_major=16)
        self.assertIn("requested PostgreSQL major 16, but server reports major 15", str(ctx.exception))

    def test_postgres_extension_verification(self) -> None:
        # Available extension
        check_postgres_version(
            "PostgreSQL 16.0",
            160000,
            expected_major=16,
            label="PostgreSQL with TimescaleDB",
            available_extensions={"timescaledb"},
        )

        # Missing extension
        with self.assertRaises(RuntimeError) as ctx:
            check_postgres_version(
                "PostgreSQL 16.0",
                160000,
                expected_major=16,
                label="PostgreSQL with PostGIS",
                available_extensions=set(),
            )
        self.assertIn("extension 'postgis'", str(ctx.exception))

    def test_mysql_version_pair_parsing(self) -> None:
        # Oracle MySQL
        self.assertEqual(_mysql_version_pair("8.4.3", "MySQL Community Server - GPL"), (8, 4))
        self.assertEqual(_mysql_version_pair("9.7.0", "MySQL Community Server - GPL"), (9, 7))
        self.assertEqual(_mysql_version_pair("26.10.0", "MySQL Innovation Server - GPL"), (26, 10))

        # Percona Server for MySQL
        self.assertEqual(
            _mysql_version_pair("8.4.3-3", "Percona Server (GPL), Release 3, Revision abcdef"), (8, 4)
        )
        self.assertEqual(
            _mysql_version_pair("9.7.0-1", "Percona Server (GPL), Release 1"), (9, 7)
        )

        # MariaDB
        self.assertEqual(
            _mysql_version_pair("10.11.8-MariaDB-log", "mariadb.org binary distribution"), (10, 11)
        )
        self.assertEqual(
            _mysql_version_pair("11.4.5-MariaDB-1:11.4.5+maria~ubu2404", "mariadb.org binary distribution"), (11, 4)
        )
        self.assertEqual(
            _mysql_version_pair("11.8.1-MariaDB", "mariadb.org binary distribution"), (11, 8)
        )
        self.assertEqual(
            _mysql_version_pair("12.3.1-MariaDB", "mariadb.org binary distribution"), (12, 3)
        )
        self.assertEqual(
            _mysql_version_pair("13.1.0-MariaDB", "mariadb.org binary distribution"), (13, 1)
        )

    def test_mysql_check_version_matching(self) -> None:
        # Oracle MySQL 8.4 LTS
        prod, pair = check_mysql_version("8.4.3", "MySQL Community Server - GPL", "mysql", "8.4")
        self.assertEqual(prod, "mysql")
        self.assertEqual(pair, (8, 4))

        # Oracle MySQL 9.7 LTS
        prod, pair = check_mysql_version("9.7.0", "MySQL Community Server - GPL", "mysql", "9.7")
        self.assertEqual(prod, "mysql")
        self.assertEqual(pair, (9, 7))

        # MySQL 26.10 Innovation
        prod, pair = check_mysql_version("26.10.0", "MySQL Community Server - GPL", "mysql", "26.10")
        self.assertEqual(prod, "mysql")
        self.assertEqual(pair, (26, 10))

        # Percona Server 8.4
        prod, pair = check_mysql_version("8.4.3-3", "Percona Server (GPL), Release 3", "percona", "8.4")
        self.assertEqual(prod, "percona")
        self.assertEqual(pair, (8, 4))

        # Percona Server 9.7
        prod, pair = check_mysql_version("9.7.0-1", "Percona Server (GPL), Release 1", "percona", "9.7")
        self.assertEqual(prod, "percona")
        self.assertEqual(pair, (9, 7))

        # MariaDB 11.4 LTS
        prod, pair = check_mysql_version("11.4.5-MariaDB", "mariadb.org binary distribution", "mariadb", "11.4")
        self.assertEqual(prod, "mariadb")
        self.assertEqual(pair, (11, 4))

        # MariaDB 13.1 Rolling
        prod, pair = check_mysql_version("13.1.0-MariaDB", "mariadb.org binary distribution", "mariadb", "13.1")
        self.assertEqual(prod, "mariadb")
        self.assertEqual(pair, (13, 1))

    def test_mysql_check_version_derivation_from_image(self) -> None:
        prod, pair = check_mysql_version(
            "11.4.5-MariaDB", "mariadb.org binary distribution", requested_image="mariadb:11.4"
        )
        self.assertEqual(prod, "mariadb")
        self.assertEqual(pair, (11, 4))

        prod, pair = check_mysql_version(
            "8.4.3-3", "Percona Server (GPL)", requested_image="percona/percona-server:8.4"
        )
        self.assertEqual(prod, "percona")
        self.assertEqual(pair, (8, 4))

    def test_mysql_product_mismatch_raises_runtime_error(self) -> None:
        # Requested mysql, server is MariaDB
        with self.assertRaises(RuntimeError) as ctx:
            check_mysql_version("11.4.5-MariaDB", "mariadb.org", expected_product="mysql", expected_version="11.4")
        self.assertIn("requested mysql, but server identifies as", str(ctx.exception))

        # Requested mariadb, server is Oracle MySQL
        with self.assertRaises(RuntimeError) as ctx:
            check_mysql_version("8.4.3", "MySQL Community Server", expected_product="mariadb", expected_version="8.4")
        self.assertIn("requested mariadb, but server identifies as", str(ctx.exception))

        # Requested percona, server is Oracle MySQL
        with self.assertRaises(RuntimeError) as ctx:
            check_mysql_version("8.4.3", "MySQL Community Server", expected_product="percona", expected_version="8.4")
        self.assertIn("requested percona, but server identifies as", str(ctx.exception))

    def test_mysql_version_mismatch_raises_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            check_mysql_version("8.0.36", "MySQL Community Server", expected_product="mysql", expected_version="8.4")
        self.assertIn("requested mysql 8.4, but server reports", str(ctx.exception))

    def test_main_entrypoint_requires_dsn(self) -> None:
        import verify_engine_matrix
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                if hasattr(verify_engine_matrix, "__name__"):
                    if not os.environ.get("POSTGRES_DSN") and not os.environ.get("MYSQL_DSN"):
                        raise SystemExit("Set POSTGRES_DSN or MYSQL_DSN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
