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


# ==============================================================================
# Fuzz tiering: fixed-seed PR gate (ci.yml) vs random-seed extended (ci-extended.yml)
# ==============================================================================

CI_EXTENDED_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-extended.yml"
FUZZ_CAMPAIGN_KEYS = (
    "MATVIEW_FUZZ_SCALE",
    "MATVIEW_FUZZ_SEEDS",
    "MATVIEW_FUZZ_RANDOM_SEEDS",
    "MATVIEW_FUZZ_SCENARIOS",
    "MATVIEW_FUZZ_VIEWS_PER_SCHEMA",
    "MATVIEW_FUZZ_MUTATIONS",
)
# Every deterministic fuzz catch in CI history (2026-11-05..18) came from this
# (seed, scenario) of the former scale-2 gate campaign; the reduced gate must keep it.
HISTORICAL_CATCH = (20261106, 1)
_FUZZ_CLI_FLAGS = ("--fuzz-scale", "--fuzz-seeds", "--fuzz-scenarios", "--fuzz-views", "--fuzz-mutations")


def _no_random(*_: Any) -> int:
    raise AssertionError("the PR-gate fuzz campaign must not draw random seeds")


def _workflow_triggers(workflow: Dict[str, Any]) -> Dict[str, Any]:
    # PyYAML (YAML 1.1) parses the key `on` as boolean True.
    on = workflow.get("on", workflow.get(True))
    return on if isinstance(on, dict) else {}


def gate_fuzz_campaign_errors(workflow: Dict[str, Any]) -> List[str]:
    """Check the PR gate's fuzz campaign as GitHub Actions would configure it.

    Requirements: the campaign is set once at workflow level (no job/step/CLI
    override), draws no random seeds, contains HISTORICAL_CATCH, and every
    (seed, scenario) it runs generates exactly the scenario that the former
    scale-2 CI campaign generated for that pair.
    """
    import test_matview_sync as tms

    errors: List[str] = []
    wf_env = {str(k): str(v) for k, v in (workflow.get("env") or {}).items()}
    if wf_env.get("PYTHONUNBUFFERED") != "1":
        errors.append("ci.yml must set PYTHONUNBUFFERED=1 at workflow level")
    for job_id, job in (workflow.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        overridden = set(job.get("env") or {}) & set(FUZZ_CAMPAIGN_KEYS)
        if overridden:
            errors.append(f"job {job_id!r} overrides the gate fuzz campaign: {sorted(overridden)}")
        for step in job.get("steps") or []:
            if not isinstance(step, dict):
                continue
            overridden = set(step.get("env") or {}) & set(FUZZ_CAMPAIGN_KEYS)
            if overridden:
                errors.append(f"job {job_id!r} step {step.get('name')!r} overrides {sorted(overridden)}")
            run = str(step.get("run", ""))
            if any(flag in run for flag in _FUZZ_CLI_FLAGS) or any(k in run for k in FUZZ_CAMPAIGN_KEYS):
                errors.append(f"job {job_id!r} step {step.get('name')!r} changes the fuzz campaign in its command")

    gate_env = {k: v for k, v in wf_env.items() if k in FUZZ_CAMPAIGN_KEYS}
    gate_env["CI"] = "true"
    try:
        gate = tms.fuzz_campaign_config(gate_env, random_seed_source=_no_random)
    except AssertionError as exc:
        return errors + [str(exc)]
    reference = tms.fuzz_campaign_config({"CI": "true", "MATVIEW_FUZZ_RANDOM_SEEDS": "0"}, random_seed_source=_no_random)
    if reference["scale"] != 2:
        errors.append(f"reference CI campaign is expected at scale 2, got {reference['scale']}")

    if gate["random_seeds"]:
        errors.append(f"gate campaign draws random seeds: {gate['random_seeds']}")
    seed, scenario = HISTORICAL_CATCH
    if seed not in gate["seeds"] or gate["scenarios_per_seed"] < scenario:
        errors.append(f"gate campaign {gate['seeds']} x {gate['scenarios_per_seed']} scenarios drops {HISTORICAL_CATCH}")
    extra = [s for s in gate["seeds"] if s not in reference["seeds"]]
    if extra:
        errors.append(f"gate seeds {extra} are not fixed seeds of the scale-2 campaign {reference['seeds']}")
    if gate["scenarios_per_seed"] > reference["scenarios_per_seed"]:
        errors.append("gate runs more scenarios per seed than the scale-2 campaign")

    # Scenario identity: generate both and compare (generation is cheap; execution is not).
    for s_val in gate["seeds"]:
        fz = tms.DifferentialSQLFuzzer(seed=s_val)
        for sc in range(1, min(gate["scenarios_per_seed"], reference["scenarios_per_seed"]) + 1):
            for dialect in (tms.Dialect.SQLITE, tms.Dialect.POSTGRES, tms.Dialect.MYSQL):
                a = fz.generate_scenario(sc, dialect, views_count=gate["views_per_schema"],
                                         mutations_count=gate["mutations_per_scenario"])
                b = fz.generate_scenario(sc, dialect, views_count=reference["views_per_schema"],
                                         mutations_count=reference["mutations_per_scenario"])
                if a != b:
                    errors.append(f"gate scenario (seed={s_val}, sc={sc}, {dialect.value}) differs from the scale-2 scenario")
        var_gate = tms.VariableSchemaDifferentialFuzzer(seed=s_val).generate_campaign(
            gate["scenarios_per_seed"], candidates_per_scenario=6)[0]
        var_ref = tms.VariableSchemaDifferentialFuzzer(seed=s_val).generate_campaign(
            reference["scenarios_per_seed"], candidates_per_scenario=6)[0]
        for idx in range(min(len(var_gate), len(var_ref))):
            if var_gate[idx][:4] != var_ref[idx][:4]:
                errors.append(f"gate variable-schema scenario (seed={s_val}, sc={idx + 1}) differs from scale-2")
    return errors


def extended_workflow_errors(extended: Dict[str, Any], gate: Dict[str, Any]) -> List[str]:
    """Structural contract of the non-required random-seed fuzz workflow."""
    from test_readme_examples import render_matrix_rows

    errors: List[str] = []
    triggers = _workflow_triggers(extended)
    if set(triggers) != {"push", "schedule", "workflow_dispatch"}:
        errors.append(f"ci-extended.yml triggers must be push/schedule/workflow_dispatch, got {sorted(triggers)}")
    push = triggers.get("push") or {}
    if not isinstance(push, dict) or list(push.get("branches") or []) != ["master"]:
        errors.append("ci-extended.yml push trigger must be restricted to master")
    if not triggers.get("schedule"):
        errors.append("ci-extended.yml needs a schedule (nightly) trigger")

    concurrency = extended.get("concurrency") or {}
    if "github.event_name" not in str(concurrency.get("group", "")):
        errors.append("ci-extended.yml concurrency group must separate events (push must not cancel the nightly run)")
    if str(concurrency.get("cancel-in-progress")).lower() != "false":
        errors.append("ci-extended.yml must not cancel in-progress exploration runs")

    perms = extended.get("permissions") or {}
    if perms != {"contents": "read"}:
        errors.append(f"ci-extended.yml workflow-level permissions must be contents: read only, got {perms}")

    env = {str(k): str(v) for k, v in (extended.get("env") or {}).items()}
    if env.get("MATVIEW_FUZZ_SCALE") != "2":
        errors.append("ci-extended.yml must run the scale-2 campaign")
    if int(env.get("MATVIEW_FUZZ_RANDOM_SEEDS", "4")) < 1:
        errors.append("ci-extended.yml must draw random seeds")
    if "MATVIEW_FUZZ_SEEDS" in env:
        errors.append("ci-extended.yml must not pin MATVIEW_FUZZ_SEEDS (that disables random seeds)")
    if env.get("PYTHONUNBUFFERED") != "1":
        errors.append("ci-extended.yml must set PYTHONUNBUFFERED=1")

    jobs = extended.get("jobs") or {}
    if any(isinstance(j, dict) and j.get("name") == "CI success" for j in jobs.values()):
        errors.append("ci-extended.yml must not define the required `CI success` check")
    fuzz_jobs = {jid: j for jid, j in jobs.items() if isinstance(j, dict) and any(
        "test_matview_sync.py --fuzz" in str(s.get("run", "")) for s in j.get("steps") or [])}
    for jid, job in fuzz_jobs.items():
        steps = job.get("steps") or []
        runs = [str(s.get("run", "")) for s in steps]
        fuzz_idx = next(i for i, r in enumerate(runs) if "test_matview_sync.py --fuzz" in r)
        if any(flag in runs[fuzz_idx] for flag in _FUZZ_CLI_FLAGS):
            errors.append(f"ci-extended.yml {jid}: fuzz command must use the workflow campaign env")
        native = bool(job.get("services")) or any("docker run" in r for r in runs)
        if native:
            verify = [i for i, r in enumerate(runs) if "verify_engine_matrix.py" in r]
            if not verify or verify[0] > fuzz_idx:
                errors.append(f"ci-extended.yml {jid}: verify_engine_matrix.py must run before the fuzzer")
            else:
                venv = steps[verify[0]].get("env") or {}
                if "EXPECTED_PG_MAJOR" not in venv and "EXPECTED_MYSQL_VERSION" not in venv:
                    errors.append(f"ci-extended.yml {jid}: verify step needs explicit version expectations")
        after = steps[fuzz_idx + 1:]
        if not any("fuzz_failure_report.py" in str(s.get("run", "")) and s.get("if") == "always()" for s in after):
            errors.append(f"ci-extended.yml {jid}: missing always() failing-seed report step after the fuzzer")
        uploads = [s for s in after if "upload-artifact" in str(s.get("uses", "")) and s.get("if") == "always()"]
        if not uploads or not all("fuzz-campaign.json" in str((u.get("with") or {}).get("path", ""))
                                  and "fuzz-report" in str((u.get("with") or {}).get("path", "")) for u in uploads):
            errors.append(f"ci-extended.yml {jid}: must upload fuzz-campaign.json and fuzz-report/ with if: always()")
        if not job.get("timeout-minutes"):
            errors.append(f"ci-extended.yml {jid}: needs timeout-minutes")

    gate_rows = {r[2]: r[1] for r in render_matrix_rows(gate)}
    ext_rows = render_matrix_rows(extended)
    ext_dbs = {db for _, _, db in ext_rows}
    expected = {"SQLite bundled with CPython (`sqlite3`)", "`postgres:18`", "`mysql:8.4`", "`mariadb:12.3`"}
    if ext_dbs != expected:
        errors.append(f"ci-extended.yml targets must be {sorted(expected)}, got {sorted(ext_dbs)}")
    for name, py, db in ext_rows:
        if db.startswith("`") and db not in gate_rows:
            errors.append(f"ci-extended.yml {name}: image {db} is not a PR-gate image")
        if py != "3.12":
            errors.append(f"ci-extended.yml {name}: expected Python 3.12, got {py}")

    reporters = [(jid, j) for jid, j in jobs.items() if isinstance(j, dict) and any(
        "github-script" in str(s.get("uses", "")) for s in j.get("steps") or [])]
    if len(reporters) != 1:
        errors.append("ci-extended.yml needs exactly one issue-reporting job")
    else:
        jid, job = reporters[0]
        if set(job.get("needs") or []) != set(fuzz_jobs):
            errors.append(f"ci-extended.yml {jid} must depend on every fuzz job")
        if (job.get("permissions") or {}).get("issues") != "write":
            errors.append(f"ci-extended.yml {jid} needs permissions issues: write")
        cond = str(job.get("if", ""))
        for needle in ("!cancelled()", "'push'", "'schedule'", "'failure'"):
            if needle not in cond:
                errors.append(f"ci-extended.yml {jid} condition must contain {needle}: {cond}")
        script = "".join(str((s.get("with") or {}).get("script", "")) for s in job.get("steps") or [])
        for needle in ("listForRepo", "createComment", "issues.create", "context.runId"):
            if needle not in script:
                errors.append(f"ci-extended.yml {jid} script must use {needle} (dedupe + run link)")
    return errors


class TestFuzzTiering(unittest.TestCase):
    """The PR gate runs a fixed-seed subset of the scale-2 campaign; random seeds run in ci-extended.yml."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = _parse_yaml_minimal(CI_WORKFLOW.read_text(encoding="utf-8"))
        cls.extended = _parse_yaml_minimal(CI_EXTENDED_WORKFLOW.read_text(encoding="utf-8"))

    def test_gate_fuzz_campaign_is_fixed_and_identical_to_scale2_scenarios(self) -> None:
        self.assertEqual(gate_fuzz_campaign_errors(self.gate), [])

    def test_gate_validator_detects_random_seeds(self) -> None:
        import copy
        wf = copy.deepcopy(self.gate)
        del wf["env"]["MATVIEW_FUZZ_RANDOM_SEEDS"]
        errors = gate_fuzz_campaign_errors(wf)
        self.assertTrue(any("random seeds" in e for e in errors), errors)

    def test_gate_validator_detects_unpinned_scenario_sizes(self) -> None:
        # Scale 1 defaults (60 views, 45 mutations) generate different mutations for
        # (20261106, 1): the historical catching scenario would silently disappear.
        import copy
        wf = copy.deepcopy(self.gate)
        del wf["env"]["MATVIEW_FUZZ_MUTATIONS"]
        errors = gate_fuzz_campaign_errors(wf)
        self.assertTrue(any("seed=20261106, sc=1" in e for e in errors), errors)

    def test_gate_validator_detects_dropped_historical_seed_and_overrides(self) -> None:
        import copy
        wf = copy.deepcopy(self.gate)
        wf["env"]["MATVIEW_FUZZ_SEEDS"] = "1337"
        wf["jobs"]["postgres"]["steps"][0].setdefault("env", {})["MATVIEW_FUZZ_SCALE"] = "3"
        errors = gate_fuzz_campaign_errors(wf)
        self.assertTrue(any("drops (20261106, 1)" in e for e in errors), errors)
        self.assertTrue(any("'postgres'" in e and "MATVIEW_FUZZ_SCALE" in e for e in errors), errors)

    def test_extended_workflow_contract(self) -> None:
        self.assertEqual(extended_workflow_errors(self.extended, self.gate), [])

    def test_extended_validator_detects_pull_request_trigger_and_missing_issue_permission(self) -> None:
        import copy
        wf = copy.deepcopy(self.extended)
        triggers = _workflow_triggers(wf)
        triggers["pull_request"] = None
        wf["jobs"]["report-failure"]["permissions"] = {"contents": "read"}
        wf["concurrency"]["cancel-in-progress"] = True
        errors = extended_workflow_errors(wf, self.gate)
        self.assertTrue(any("triggers" in e for e in errors), errors)
        self.assertTrue(any("issues: write" in e for e in errors), errors)
        self.assertTrue(any("cancel" in e for e in errors), errors)

    def test_extended_validator_detects_image_not_in_gate(self) -> None:
        import copy
        wf = copy.deepcopy(self.extended)
        wf["jobs"]["fuzz-mysql"]["strategy"]["matrix"]["include"][1]["image"] = "mariadb:13.1"
        errors = extended_workflow_errors(wf, self.gate)
        self.assertTrue(any("mariadb:13.1" in e for e in errors), errors)


class TestFuzzFailureReport(unittest.TestCase):
    """scripts/fuzz_failure_report.py: failing seeds and replay commands from a manifest."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import fuzz_failure_report  # type: ignore
        import test_matview_sync as tms

        cls.ffr = fuzz_failure_report
        cls.tms = tms

    def _manifest(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        rnd = iter([111, 222, 333, 444])
        cfg = self.tms.fuzz_campaign_config({"CI": "true", "MATVIEW_FUZZ_SCALE": "2"}, random_seed_source=lambda: next(rnd))
        return dict(cfg, schema_version=1, commit="abc", python="3.12.0", default_dialect="postgres", runs=runs)

    def test_replay_env_reproduces_the_campaign_outside_ci(self) -> None:
        manifest = self._manifest([])
        for seed in (20261106, 222):
            env = self.ffr.replay_env(manifest, [seed])
            replayed = self.tms.fuzz_campaign_config(env, random_seed_source=_no_random)  # no CI variable
            self.assertEqual(replayed["seeds"], [seed])
            for key in ("scale", "scenarios_per_seed", "views_per_schema", "mutations_per_scenario"):
                self.assertEqual(replayed[key], manifest[key], key)
            self.assertEqual(self.ffr.replay_env(manifest, [seed]), self.tms.fuzz_replay_env(manifest, [seed]))

    def test_failed_and_interrupted_batches_are_reported_once_per_seed(self) -> None:
        manifest = self._manifest([
            {"suite": "Suite 12a (seed=20261106 sc=1)", "seed": 20261106, "scenario": 1, "status": "passed"},
            {"suite": "Suite 12a (seed=222 sc=2)", "seed": 222, "scenario": 2, "status": "failed",
             "error_type": "AssertionError", "error": "MV mismatch | on v1\nline2"},
            {"suite": "Suite 12c (seed=222 sc=2)", "seed": 222, "scenario": 2, "status": "failed",
             "error_type": "AssertionError", "error": "x"},
            {"suite": "Suite 12e (seed=444 sc=3)", "seed": 444, "scenario": 3, "status": "running"},
        ])
        self.assertEqual(self.ffr.failing_seeds(manifest), [222, 444])
        md = self.ffr.render(manifest, target="PostgreSQL 18", engine_env="POSTGRES_DSN=pg://x", run_url="https://run")
        self.assertNotIn("| 20261106 |", md)
        self.assertIn("| 222 | random | 2 |", md)
        self.assertIn("interrupted", md)
        self.assertIn("MV mismatch \\| on v1 line2", md)
        self.assertIn("POSTGRES_DSN=pg://x MATVIEW_FUZZ_SCALE=2 MATVIEW_FUZZ_SCENARIOS=3 "
                      "MATVIEW_FUZZ_VIEWS_PER_SCHEMA=70 MATVIEW_FUZZ_MUTATIONS=60 MATVIEW_FUZZ_SEEDS=222 "
                      "python3 test_matview_sync.py --fuzz", md)
        self.assertEqual(md.count("MATVIEW_FUZZ_SEEDS=222 "), 1)
        self.assertIn("https://run", md)

    def test_missing_manifest_and_clean_campaign_are_explained(self) -> None:
        self.assertIn("No fuzz-campaign.json manifest", self.ffr.render(None, target="t"))
        clean = self.ffr.render(self._manifest([{"seed": 1337, "scenario": 1, "status": "passed"}]), target="t")
        self.assertIn("No Suite 12 batch failed", clean)
        self.assertNotIn("```sh", clean)

    def test_main_writes_outputs_and_step_summary(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            mpath = Path(tmp) / "fuzz-campaign.json"
            mpath.write_text(json.dumps(self._manifest([
                {"suite": "s", "seed": 333, "scenario": 1, "status": "failed", "error_type": "E", "error": "e"}])))
            summary = Path(tmp) / "summary.md"
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}), \
                    patch("sys.stdout", new_callable=io.StringIO):
                rc = self.ffr.main(["--manifest", str(mpath), "--target", "SQLite", "--out-dir", str(Path(tmp) / "out")])
            self.assertEqual(rc, 0)
            self.assertEqual((Path(tmp) / "out" / "failing-seeds.txt").read_text(), "333\n")
            self.assertIn("MATVIEW_FUZZ_SEEDS=333", (Path(tmp) / "out" / "fuzz-failures.md").read_text())
            self.assertIn("MATVIEW_FUZZ_SEEDS=333", summary.read_text())
            with patch("sys.stdout", new_callable=io.StringIO):
                rc = self.ffr.main(["--manifest", str(Path(tmp) / "absent.json"), "--out-dir", str(Path(tmp) / "out2")])
            self.assertEqual(rc, 0)
            self.assertEqual((Path(tmp) / "out2" / "failing-seeds.txt").read_text(), "")



if __name__ == "__main__":
    unittest.main(verbosity=2)
