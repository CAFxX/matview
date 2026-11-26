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
from typing import Any, Dict, List, Sequence, Set, Tuple
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

    errors += issue_reporter_errors("ci-extended.yml", extended, set(fuzz_jobs), ("'push'", "'schedule'"),
                                    "fuzz-extended-*")
    return errors


CI_FAILURE_ISSUE_JS = REPO_ROOT / "scripts" / "ci_failure_issue.js"


def issue_reporter_errors(fname: str, workflow: Dict[str, Any], expected_needs: set,
                          event_needles: Sequence[str], artifact_pattern: str) -> List[str]:
    """One job opens/updates ONE issue through scripts/ci_failure_issue.js."""
    errors: List[str] = []
    jobs = workflow.get("jobs") or {}
    reporters = [(jid, j) for jid, j in jobs.items() if isinstance(j, dict) and any(
        "github-script" in str(s.get("uses", "")) for s in j.get("steps") or [])]
    if len(reporters) != 1:
        return [f"{fname} needs exactly one issue-reporting job"]
    jid, job = reporters[0]
    if set(job.get("needs") or []) != set(expected_needs):
        errors.append(f"{fname} {jid} must depend on exactly {sorted(expected_needs)}, got {job.get('needs')}")
    if (job.get("permissions") or {}).get("issues") != "write":
        errors.append(f"{fname} {jid} needs permissions issues: write")
    if not job.get("timeout-minutes"):
        errors.append(f"{fname} {jid}: needs timeout-minutes")
    cond = str(job.get("if", ""))
    for needle in ("!cancelled()", "'failure'", *event_needles):
        if needle not in cond:
            errors.append(f"{fname} {jid} condition must contain {needle}: {cond}")
    steps = job.get("steps") or []
    uses = [str(s.get("uses", "")) for s in steps]
    script_idx = next(i for i, u in enumerate(uses) if "github-script" in u)
    if not any(u.startswith("actions/checkout@") for u in uses[:script_idx]):
        errors.append(f"{fname} {jid}: must check out the repository before requiring the shared script")
    downloads = [s for s in steps[:script_idx] if "download-artifact" in str(s.get("uses", ""))]
    if not downloads or str((downloads[0].get("with") or {}).get("pattern")) != artifact_pattern:
        errors.append(f"{fname} {jid}: must download the {artifact_pattern} report artifacts")
    script = str((steps[script_idx].get("with") or {}).get("script", ""))
    if 'require("./scripts/ci_failure_issue.js")' not in script:
        errors.append(f"{fname} {jid}: must use scripts/ci_failure_issue.js")
    if "NEEDS_JSON" not in (steps[script_idx].get("env") or {}):
        errors.append(f"{fname} {jid}: must pass NEEDS_JSON (job results) to the script")
    for key in ("title:", "label:", "reportsDir:", "reportName:"):
        if key not in script:
            errors.append(f"{fname} {jid}: script must set {key}")
    shared = CI_FAILURE_ISSUE_JS.read_text(encoding="utf-8")
    for needle in ("listForRepo", "createComment", "issues.create", "context.runId"):
        if needle not in shared:
            errors.append(f"scripts/ci_failure_issue.js must use {needle} (dedupe + run link)")
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



# ==============================================================================
# Workflow hygiene: timeouts, every test file runs in CI, deferred sweep contract
# ==============================================================================

CI_DEFERRED_WEEKLY_CRON = "17 3 * * 0"
CI_DEFERRED_NIGHTLY_CRON = "17 3 * * 1-6"
GATE_JOB_TIMEOUT_MAX = 25
# Temporary per-job stopgaps above GATE_JOB_TIMEOUT_MAX. mysql-compat: the MariaDB
# jobs ran at 23.7-25 min and were cancelled at 25 after the group-key collation
# change; lower back to the budget once the slowdown is removed and headroom proven.
GATE_JOB_TIMEOUT_EXCEPTIONS = {"mysql-compat": 40}
DEFERRED_JOB_TIMEOUT_MAX = {"python-deferred": 60, "compatibility-sweep": 60, "dependency-minimum": 60,
                            "fuzz-stress": 90}
DEFERRED_REPORTED_JOBS = {"python-deferred", "compatibility-sweep", "dependency-minimum", "fuzz-stress",
                          "deferred-success"}


def _all_workflows() -> Dict[str, Dict[str, Any]]:
    return {p.name: _parse_yaml_minimal(p.read_text(encoding="utf-8"))
            for p in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))}


def timeout_errors(workflows: Dict[str, Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for fname, wf in workflows.items():
        for jid, job in (wf.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            tmo = job.get("timeout-minutes")
            if not str(tmo or "").isdigit() or int(tmo) <= 0:
                errors.append(f"{fname} {jid}: needs a positive integer timeout-minutes, got {tmo!r}")
                continue
            gate_max = GATE_JOB_TIMEOUT_EXCEPTIONS.get(jid, GATE_JOB_TIMEOUT_MAX)
            if fname == "ci.yml" and int(tmo) > gate_max:
                errors.append(f"ci.yml {jid}: timeout {tmo} exceeds the gate budget {gate_max}")
            if fname == "ci-deferred.yml" and jid in DEFERRED_JOB_TIMEOUT_MAX and int(tmo) > DEFERRED_JOB_TIMEOUT_MAX[jid]:
                errors.append(f"ci-deferred.yml {jid}: timeout {tmo} exceeds {DEFERRED_JOB_TIMEOUT_MAX[jid]}")
    return errors


def unexecuted_test_files(workflows: Dict[str, Dict[str, Any]], root: Path = REPO_ROOT) -> List[str]:
    """Test files in the repository that no workflow step executes."""
    commands = "\n".join(
        str(step.get("run", ""))
        for wf in workflows.values()
        for job in (wf.get("jobs") or {}).values() if isinstance(job, dict)
        for step in job.get("steps") or [] if isinstance(step, dict)
    )
    missing = []
    for path in sorted([*root.glob("test_*.py"), *root.glob("scripts/test_*.py")]):
        rel = path.relative_to(root).as_posix()
        module = rel[:-3].replace("/", ".")
        if not re.search(rf"(?<![\w/.-]){re.escape(rel)}\b|-m unittest\b[^\n]*\b{re.escape(module)}\b", commands):
            missing.append(rel)
    return missing


# Test modules that mention live-engine env-var names only to verify workflow
# definitions, presubmit commands, docs, or mocked env behavior (never opening
# a live database connection).
LIVE_JOB_NON_LIVE_MODULES: Dict[str, str] = {
    "scripts/test_presubmit.py": "verifies presubmit/workflow commands without opening live DB connections",
    "test_oracle_adversarial_boundaries.py": "patches SQLITE_COMPAT_ENGINE in-process with unittest.mock",
    "test_p2_ci_matrix_verification.py": "validates workflow definitions and env-var names statically",
    "test_readme_examples.py": "validates README links, snippets, and workflow consistency",
}


def _job_run_text(workflow: Dict[str, Any], job_id: str) -> str:
    job = (workflow.get("jobs") or {}).get(job_id)
    if not isinstance(job, dict):
        return ""
    return "\n".join(
        str(step.get("run", ""))
        for step in job.get("steps") or []
        if isinstance(step, dict)
    )


def _module_in_commands(rel: str, commands: str) -> bool:
    module = rel[:-3].replace("/", ".")
    return bool(re.search(rf"(?<![\w/.-]){re.escape(rel)}\b|-m unittest\b[^\n]*\b{re.escape(module)}\b", commands))


def _live_engine_requirements(path: Path) -> Tuple[bool, bool, bool]:
    """Inspect a test file's AST (excluding docstrings) to determine whether it
    contains live PostgreSQL, live MySQL/MariaDB, or SQLite-compat tests."""
    import ast

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path.as_posix())
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    consts: set[str] = set()
    imports_bulk_targets = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_nodes:
            if node.value in ("POSTGRES_DSN", "MYSQL_DSN", "SQLITE_COMPAT_ENGINE"):
                consts.add(node.value)
        elif isinstance(node, ast.ImportFrom) and node.module == "test_native_bulk_statements":
            if any(alias.name == "TARGETS" for alias in node.names):
                imports_bulk_targets = True
        elif isinstance(node, ast.Attribute) and node.attr == "TARGETS":
            if isinstance(node.value, ast.Name) and node.value.id in ("test_native_bulk_statements", "h"):
                imports_bulk_targets = True

    uses_pg = "POSTGRES_DSN" in consts or imports_bulk_targets
    uses_my = "MYSQL_DSN" in consts or imports_bulk_targets
    uses_sc = "SQLITE_COMPAT_ENGINE" in consts
    return uses_pg, uses_my, uses_sc


def live_job_module_errors(
    workflows: Dict[str, Dict[str, Any]],
    root: Path = REPO_ROOT,
) -> List[str]:
    """Verify every test module with live PostgreSQL, MySQL/MariaDB, or
    SQLite-compat tests is executed by the matching live CI job(s) in ci.yml,
    and that mysql and mysql-compat execute the same live test modules."""
    ci = workflows.get("ci.yml") or {}
    pg_cmds = _job_run_text(ci, "postgres")
    my_cmds = _job_run_text(ci, "mysql")
    mc_cmds = _job_run_text(ci, "mysql-compat")
    sc_cmds = _job_run_text(ci, "sqlite-compat")

    errors: List[str] = []
    if root == REPO_ROOT:
        for rel in sorted(LIVE_JOB_NON_LIVE_MODULES):
            if not (root / rel).exists():
                errors.append(f"LIVE_JOB_NON_LIVE_MODULES references missing file {rel!r}")

    mysql_modules: set[str] = set()
    mariadb_modules: set[str] = set()
    for path in sorted([*root.glob("test_*.py"), *root.glob("scripts/test_*.py")]):
        rel = path.relative_to(root).as_posix()
        if _module_in_commands(rel, my_cmds):
            mysql_modules.add(rel)
        if _module_in_commands(rel, mc_cmds):
            mariadb_modules.add(rel)
        if rel in LIVE_JOB_NON_LIVE_MODULES:
            continue
        uses_pg, uses_my, uses_sc = _live_engine_requirements(path)
        if uses_pg and not _module_in_commands(rel, pg_cmds):
            errors.append(f"{rel}: uses POSTGRES_DSN/TARGETS but is not run by ci.yml job 'postgres'")
        if uses_my and not _module_in_commands(rel, my_cmds):
            errors.append(f"{rel}: uses MYSQL_DSN/TARGETS but is not run by ci.yml job 'mysql'")
        if uses_my and not _module_in_commands(rel, mc_cmds):
            errors.append(f"{rel}: uses MYSQL_DSN/TARGETS but is not run by ci.yml job 'mysql-compat'")
        if uses_sc and not _module_in_commands(rel, sc_cmds):
            errors.append(f"{rel}: uses SQLITE_COMPAT_ENGINE but is not run by ci.yml job 'sqlite-compat'")

    only_mysql = sorted(mysql_modules - mariadb_modules)
    only_mariadb = sorted(mariadb_modules - mysql_modules)
    if only_mysql:
        errors.append(f"ci.yml job 'mysql' runs modules absent from 'mysql-compat': {only_mysql}")
    if only_mariadb:
        errors.append(f"ci.yml job 'mysql-compat' runs modules absent from 'mysql': {only_mariadb}")
    return errors


def deferred_workflow_errors(deferred: Dict[str, Any], requirements: str) -> List[str]:
    errors: List[str] = []
    triggers = _workflow_triggers(deferred)
    crons = sorted(str(c.get("cron")) for c in triggers.get("schedule") or [] if isinstance(c, dict))
    if crons != sorted([CI_DEFERRED_NIGHTLY_CRON, CI_DEFERRED_WEEKLY_CRON]):
        errors.append(f"ci-deferred.yml schedules must be nightly {CI_DEFERRED_NIGHTLY_CRON!r} + weekly "
                      f"{CI_DEFERRED_WEEKLY_CRON!r}, got {crons}")
    if "workflow_dispatch" not in triggers:
        errors.append("ci-deferred.yml must keep workflow_dispatch")
    concurrency = deferred.get("concurrency") or {}
    if "github.event_name" not in str(concurrency.get("group", "")) or \
            str(concurrency.get("cancel-in-progress")).lower() != "false":
        errors.append("ci-deferred.yml concurrency must separate events and never cancel a running sweep")
    if str((deferred.get("env") or {}).get("PYTHONUNBUFFERED")) != "1":
        errors.append("ci-deferred.yml must set PYTHONUNBUFFERED=1")

    jobs = deferred.get("jobs") or {}
    fuzz = jobs.get("fuzz-stress") or {}
    if CI_DEFERRED_WEEKLY_CRON not in str(fuzz.get("if", "")):
        errors.append("ci-deferred.yml fuzz-stress must be gated to the weekly schedule / manual runs")
    for jid, job in jobs.items():
        if jid != "fuzz-stress" and isinstance(job, dict) and CI_DEFERRED_WEEKLY_CRON in str(job.get("if", "")):
            errors.append(f"ci-deferred.yml {jid} must run nightly, not only weekly")
    agg = "".join(str(s.get("run", "")) for s in (jobs.get("deferred-success") or {}).get("steps") or [])
    if CI_DEFERRED_WEEKLY_CRON not in agg:
        errors.append("ci-deferred.yml deferred-success must recognise the weekly schedule")

    # Scale-3 fuzz: failing-seed report with replay env, artifacts like ci-extended.yml.
    steps = fuzz.get("steps") or []
    runs = [str(s.get("run", "")) for s in steps]
    fuzz_idx = next((i for i, r in enumerate(runs) if "test_matview_sync.py --fuzz" in r), None)
    if fuzz_idx is None:
        errors.append("ci-deferred.yml fuzz-stress has no fuzz step")
    else:
        after = steps[fuzz_idx + 1:]
        report = [s for s in after if "fuzz_failure_report.py" in str(s.get("run", "")) and s.get("if") == "always()"]
        if not report or "--engine-env" not in str(report[0].get("run", "")):
            errors.append("ci-deferred.yml fuzz-stress: missing always() failing-seed report with --engine-env")
        uploads = [s for s in after if "upload-artifact" in str(s.get("uses", "")) and s.get("if") == "always()"]
        if not uploads or not all(
                "fuzz-campaign.json" in str((u.get("with") or {}).get("path", ""))
                and "fuzz-report" in str((u.get("with") or {}).get("path", ""))
                and str((u.get("with") or {}).get("name", "")).startswith("fuzz-deferred-") for u in uploads):
            errors.append("ci-deferred.yml fuzz-stress: must upload fuzz-campaign.json and fuzz-report/ "
                          "as fuzz-deferred-* with if: always()")
    expected_replay = {"sqlite": "", "apsw": "SQLITE_COMPAT_ENGINE=apsw",
                       "postgres": "POSTGRES_DSN=postgresql://postgres:password@127.0.0.1:5432/matview_test",
                       "mysql": "MYSQL_DSN=mysql://root:password@127.0.0.1:3306/matview_test"}
    for row in ((fuzz.get("strategy") or {}).get("matrix") or {}).get("include") or []:
        if str(row.get("replay-env", None)) != expected_replay.get(str(row.get("backend"))):
            errors.append(f"ci-deferred.yml fuzz-stress {row.get('label')}: replay-env {row.get('replay-env')!r} "
                          f"does not select backend {row.get('backend')!r}")

    # Minimum dependencies: pins derived from requirements.txt, never hand-copied.
    dep_runs = "\n".join(str(s.get("run", "")) for s in (jobs.get("dependency-minimum") or {}).get("steps") or [])
    if "sed -E 's/>=/==/' requirements.txt > min-requirements.txt" not in dep_runs or \
            "pip install -r min-requirements.txt" not in dep_runs:
        errors.append("ci-deferred.yml dependency-minimum must derive its exact pins from requirements.txt")
    if re.search(r"[\"'][A-Za-z0-9_.-]+==\d", dep_runs):
        errors.append("ci-deferred.yml dependency-minimum must not hand-copy version pins")
    for line in requirements.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and ">=" not in line:
            errors.append(f"requirements.txt entry without a >= floor: {line!r}")

    # No sweep row may fail silently: known divergences are refused by the
    # product (and pinned by a strict native test), never allowed failures.
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        if job.get("continue-on-error") not in (None, False):
            errors.append(f"ci-deferred.yml job {job_name} must not set continue-on-error: {job['continue-on-error']!r}")
        for row in ((job.get("strategy") or {}).get("matrix") or {}).get("include") or []:
            if "allow-failure" in row:
                errors.append(f"ci-deferred.yml row {row.get('name')!r} must not be allow-failure; refuse the divergent shape in the product and pin it with a native test")
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        for row in ((job.get("strategy") or {}).get("matrix") or {}).get("include") or []:
            for key in ("image", "pg-image", "mysql-image"):
                if re.search(r"(?i)[-.]?rc\b|alpha|beta", str(row.get(key, ""))):
                    errors.append(f"ci-deferred.yml: pre-release image {row.get(key)!r} is not a support claim")

    errors += issue_reporter_errors("ci-deferred.yml", deferred, DEFERRED_REPORTED_JOBS, ("'schedule'",),
                                    "fuzz-deferred-*")
    return errors


def _deferred_success_script(deferred: Dict[str, Any]) -> str:
    run = "".join(str(s.get("run", "")) for s in deferred["jobs"]["deferred-success"]["steps"])
    return run.split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0]


class TestWorkflowHygiene(unittest.TestCase):
    """Timeouts, CI coverage of every test file, and the deferred sweep's contract."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workflows = _all_workflows()
        cls.deferred = cls.workflows["ci-deferred.yml"]
        cls.requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    def test_every_job_has_a_bounded_timeout(self) -> None:
        self.assertEqual(timeout_errors(self.workflows), [])

    def test_timeout_validator_detects_missing_and_excessive_timeouts(self) -> None:
        import copy
        wfs = copy.deepcopy(self.workflows)
        del wfs["ci.yml"]["jobs"]["postgres"]["timeout-minutes"]
        wfs["ci.yml"]["jobs"]["mysql"]["timeout-minutes"] = 360
        errors = timeout_errors(wfs)
        self.assertTrue(any("ci.yml postgres" in e for e in errors), errors)
        self.assertTrue(any("ci.yml mysql" in e and "exceeds" in e for e in errors), errors)

    def test_every_test_file_is_executed_by_a_workflow(self) -> None:
        self.assertEqual(unexecuted_test_files(self.workflows), [])

    def test_unexecuted_test_file_detection(self) -> None:
        import copy
        import tempfile
        wfs = copy.deepcopy(self.workflows)
        for job in wfs["ci.yml"]["jobs"].values():
            job["steps"] = [st for st in job.get("steps") or [] if "scripts/test_presubmit.py" not in str(st.get("run", ""))]
        self.assertEqual(unexecuted_test_files(wfs), ["scripts/test_presubmit.py"])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "test_new_feature.py").write_text("")
            self.assertEqual(unexecuted_test_files(self.workflows, Path(tmp)), ["test_new_feature.py"])

    def test_every_live_engine_test_module_runs_in_its_live_ci_jobs(self) -> None:
        self.assertEqual(live_job_module_errors(self.workflows), [])

    def test_live_job_module_inventory_detects_missing_and_asymmetric_modules(self) -> None:
        import copy
        import tempfile

        wfs = copy.deepcopy(self.workflows)
        for st in wfs["ci.yml"]["jobs"]["postgres"]["steps"]:
            if "run" in st:
                st["run"] = str(st["run"]).replace("test_p0_pk_swap_and_collation.py", "test_removed_placeholder.py")
        for st in wfs["ci.yml"]["jobs"]["mysql-compat"]["steps"]:
            if "run" in st:
                st["run"] = str(st["run"]).replace("test_group_key_collation.py", "test_removed_placeholder.py")
        errors = live_job_module_errors(wfs)
        self.assertTrue(any("test_p0_pk_swap_and_collation.py" in e and "'postgres'" in e for e in errors), errors)
        self.assertTrue(any("test_group_key_collation.py" in e and "'mysql-compat'" in e for e in errors), errors)
        self.assertTrue(any("absent from 'mysql-compat'" in e for e in errors), errors)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "test_new_pg.py").write_text('import os\nDSN = os.environ.get("POSTGRES_DSN")\n')
            (tmp_path / "test_new_my.py").write_text('import os\nDSN = os.environ.get("MYSQL_DSN")\n')
            (tmp_path / "test_new_sc.py").write_text('import os\nENG = os.environ.get("SQLITE_COMPAT_ENGINE")\n')
            (tmp_path / "test_new_doc_only.py").write_text('"""Mentions POSTGRES_DSN and MYSQL_DSN only in docstring."""\n')
            tmp_errors = live_job_module_errors(self.workflows, tmp_path)
            self.assertTrue(any("test_new_pg.py" in e and "'postgres'" in e for e in tmp_errors), tmp_errors)
            self.assertTrue(any("test_new_my.py" in e and "'mysql'" in e for e in tmp_errors), tmp_errors)
            self.assertTrue(any("test_new_my.py" in e and "'mysql-compat'" in e for e in tmp_errors), tmp_errors)
            self.assertTrue(any("test_new_sc.py" in e and "'sqlite-compat'" in e for e in tmp_errors), tmp_errors)
            self.assertFalse(any("test_new_doc_only.py" in e for e in tmp_errors), tmp_errors)

    def test_presubmit_selftest_runs_in_the_sqlite_gate_job_with_pyyaml(self) -> None:
        steps = self.workflows["ci.yml"]["jobs"]["sqlite"]["steps"]
        runs = [str(st.get("run", "")) for st in steps]
        install = next(i for i, r in enumerate(runs) if "pip install -r requirements.txt" in r)
        self.assertIn("pip install -r requirements-dev.txt", runs[install])
        self.assertIn("python3 scripts/test_presubmit.py", runs)
        self.assertRegex((REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8"), r"(?mi)^pyyaml>=")

    def test_deferred_workflow_contract(self) -> None:
        self.assertEqual(deferred_workflow_errors(self.deferred, self.requirements), [])

    def test_deferred_validator_detects_regressions(self) -> None:
        import copy
        wf = copy.deepcopy(self.deferred)
        _workflow_triggers(wf)["schedule"] = [{"cron": "17 3 * * 1"}]
        wf["jobs"]["fuzz-stress"]["strategy"]["matrix"]["include"][1]["replay-env"] = ""
        for st in wf["jobs"]["dependency-minimum"]["steps"]:
            if "min-requirements.txt" in str(st.get("run", "")):
                st["run"] = 'pip install "sqlglot==25.0.0"'
        wf["jobs"]["compatibility-sweep"]["strategy"]["matrix"]["include"][-1]["allow-failure"] = True
        wf["jobs"]["compatibility-sweep"]["continue-on-error"] = "${{ matrix.allow-failure == true }}"
        wf["jobs"]["compatibility-sweep"]["strategy"]["matrix"]["include"][0]["image"] = "mariadb:13.1-rc"
        del wf["jobs"]["report-failure"]
        errors = deferred_workflow_errors(wf, self.requirements + "\nnewdep\n")
        for needle in ("schedules", "APSW SQLite", "derive its exact pins", "hand-copy", "must not be allow-failure", "must not set continue-on-error",
                       "pre-release", "issue-reporting", "'newdep'"):
            self.assertTrue(any(needle in e for e in errors), (needle, errors))

    def test_deferred_success_skips_fuzz_only_on_the_nightly_schedule(self) -> None:
        import subprocess
        script = _deferred_success_script(self.deferred)

        def run(event: str, schedule: str, results: Dict[str, str]) -> int:
            needs = {k: {"result": v} for k, v in results.items()}
            env = dict(os.environ, NEEDS_JSON=__import__("json").dumps(needs), EVENT_NAME=event, EVENT_SCHEDULE=schedule)
            return subprocess.run([sys.executable, "-c", script], env=env, capture_output=True).returncode

        ok = {"python-deferred": "success", "compatibility-sweep": "success", "dependency-minimum": "success"}
        self.assertEqual(run("schedule", CI_DEFERRED_NIGHTLY_CRON, dict(ok, **{"fuzz-stress": "skipped"})), 0)
        self.assertEqual(run("schedule", CI_DEFERRED_WEEKLY_CRON, dict(ok, **{"fuzz-stress": "skipped"})), 1)
        self.assertEqual(run("workflow_dispatch", "", dict(ok, **{"fuzz-stress": "skipped"})), 1)
        self.assertEqual(run("schedule", CI_DEFERRED_NIGHTLY_CRON, dict(ok, **{"python-deferred": "skipped", "fuzz-stress": "skipped"})), 1)
        self.assertEqual(run("schedule", CI_DEFERRED_WEEKLY_CRON, dict(ok, **{"fuzz-stress": "success"})), 0)
        self.assertEqual(run("schedule", CI_DEFERRED_WEEKLY_CRON, dict(ok, **{"fuzz-stress": "failure"})), 1)


@unittest.skipUnless(__import__("shutil").which("node"), "node is required to execute scripts/ci_failure_issue.js")
class TestCiFailureIssueScript(unittest.TestCase):
    """Executes scripts/ci_failure_issue.js against a fake GitHub client (dedupe, labels, reports)."""

    DRIVER = r"""
const report = require(process.argv[1]);
const state = JSON.parse(process.argv[2]);
const calls = [];
const github = {
  paginate: async (fn, args) => { calls.push(["listForRepo", args.labels, args.state]); return state.open; },
  rest: { issues: {
    listForRepo: "listForRepo",
    getLabel: async a => { calls.push(["getLabel", a.name]); if (!state.labelExists) { const e = new Error("nf"); e.status = 404; throw e; } },
    createLabel: async a => { calls.push(["createLabel", a.name]); },
    createComment: async a => { calls.push(["createComment", a.issue_number, a.body]); },
    create: async a => { calls.push(["create", a.title, a.labels, a.body]);
      if (state.issuesGone) { const e = new Error("Issues are disabled"); e.status = 410; throw e; }
      return { data: { number: 99 } }; },
  } },
};
const context = { repo: { owner: "o", repo: "r" }, serverUrl: "https://gh", runId: 7, eventName: "schedule", sha: "abc",
  payload: state.hasIssues === undefined ? {} : { repository: { has_issues: state.hasIssues } } };
const summary = { text: "", addHeading(t) { this.text += t + "\n"; return this; }, addRaw(t) { this.text += t; return this; },
  async write() { calls.push(["summary", this.text]); } };
const core = { info: () => {}, error: m => calls.push(["error", m]), setFailed: m => calls.push(["setFailed", m]), summary };
report({ github, context, core }, { title: "T", label: "L", labelDescription: "d", intro: "INTRO",
  reportsDir: state.dir, reportName: "fuzz-failures.md" })
  .then(n => { console.log(JSON.stringify({ n, calls })); })
  .catch(e => { console.error(e); process.exit(1); });
"""

    def _run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import json
        import subprocess
        out = subprocess.run(["node", "-e", self.DRIVER, str(CI_FAILURE_ISSUE_JS), json.dumps(state)],
                             env=dict(os.environ, NEEDS_JSON=json.dumps({"fuzz-stress": {"result": "failure"}})),
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout)

    def test_opens_one_labelled_issue_with_run_link_results_and_reports(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a").mkdir()
            (Path(tmp) / "a" / "fuzz-failures.md").write_text("SEED 222 replay")
            res = self._run({"open": [], "labelExists": False, "dir": tmp})
        kinds = [c[0] for c in res["calls"]]
        self.assertEqual(kinds, ["getLabel", "createLabel", "listForRepo", "create"])
        _, title, labels, body = res["calls"][-1]
        self.assertEqual((title, labels, res["n"]), ("T", ["L"], 99))
        for needle in ("INTRO", "https://gh/o/r/actions/runs/7", "fuzz-stress: failure", "SEED 222 replay"):
            self.assertIn(needle, body)

    def test_comments_on_the_open_issue_instead_of_opening_another(self) -> None:
        res = self._run({"open": [{"number": 6, "title": "T", "pull_request": {}}, {"number": 5, "title": "T"}],
                         "labelExists": True, "dir": "/nonexistent"})
        kinds = [c[0] for c in res["calls"]]
        self.assertEqual(kinds, ["getLabel", "listForRepo", "createComment"])
        self.assertEqual(res["calls"][-1][1], 5)
        self.assertIn("No report artifact was found", res["calls"][-1][2])

    def test_disabled_issues_annotate_and_summarize_without_failing_the_reporter(self) -> None:
        # Payload says so (no API write attempted) ...
        res = self._run({"open": [], "labelExists": True, "dir": "/nonexistent", "hasIssues": False})
        kinds = [c[0] for c in res["calls"]]
        self.assertEqual(kinds, ["error", "summary"])
        self.assertIn("Issues are disabled", res["calls"][0][1])
        self.assertIn("https://gh/o/r/actions/runs/7", res["calls"][1][1])
        self.assertIsNone(res["n"])
        # ... or the API answers 410 Gone.
        res = self._run({"open": [], "labelExists": True, "dir": "/nonexistent", "issuesGone": True})
        self.assertEqual([c[0] for c in res["calls"]][-2:], ["error", "summary"])
        self.assertNotIn("setFailed", [c[0] for c in res["calls"]])

    def test_other_titles_do_not_deduplicate(self) -> None:
        res = self._run({"open": [{"number": 5, "title": "other"}], "labelExists": True, "dir": "/nonexistent"})
        self.assertEqual(res["calls"][-1][0], "create")



class TestEmulatedBackendSkip(unittest.TestCase):
    """MATVIEW_EMULATED_BACKENDS=skip (live CI jobs): suites skip emulated batches only."""

    def test_skip_applies_to_the_suite_runner_only_and_needs_a_live_backend(self) -> None:
        from test_matview_sync import Dialect, MVTestRunner

        live = {"MATVIEW_EMULATED_BACKENDS": "skip", "SQLITE_COMPAT_ENGINE": "sqlite3"}
        with patch.dict(os.environ, live, clear=False):
            os.environ.pop("POSTGRES_DSN", None)
            os.environ.pop("MYSQL_DSN", None)
            # The unittest classes build runners that must keep running every dialect.
            self.assertFalse(MVTestRunner().skip_emulated_backends)
            suite_runner = MVTestRunner(honor_emulated_skip=True)
            with patch("sys.stdout", io.StringIO()):
                self.assertFalse(suite_runner.skips_emulated(Dialect.SQLITE, "s"))
                self.assertTrue(suite_runner.skips_emulated(Dialect.POSTGRES, "s"))
                self.assertTrue(suite_runner.skips_emulated(Dialect.MYSQL, "s"))
            self.assertEqual(suite_runner.skipped_emulated_batches, 2)
        with patch.dict(os.environ, {"MATVIEW_EMULATED_BACKENDS": "skip"}, clear=False):
            for k in ("POSTGRES_DSN", "MYSQL_DSN", "SQLITE_COMPAT_ENGINE"):
                os.environ.pop(k, None)
            with self.assertRaises(ValueError):
                MVTestRunner(honor_emulated_skip=True)
        with patch.dict(os.environ, {"MATVIEW_EMULATED_BACKENDS": "sometimes"}, clear=False):
            with self.assertRaises(ValueError):
                MVTestRunner()
_SUITE_KEYS = [str(i) for i in range(1, 14)] + ["unit"]


def matview_sync_partition_errors(workflow: Dict[str, Any]) -> List[str]:
    """Every gate job that runs test_matview_sync.py must run every suite exactly once.

    A job either runs it unsharded (no ``--suites``) or as ``--suites`` shards that
    partition suites 1-13 + unit; Suite 12 may be split further by ``--seed-shard i/n``
    shards that together cover i = 0..n-1 exactly once. ``--fuzz`` runs (the
    extended/deferred campaigns) are not shards and are ignored here.
    """
    from test_matview_sync import parse_suite_selection

    errors: List[str] = []
    for jid, job in (workflow.get("jobs") or {}).items():
        calls: List[str] = []
        for step in job.get("steps") or []:
            calls += re.findall(r"test_matview_sync\.py((?: --[\w-]+(?: [^\s\"\\]+)?)*)", str(step.get("run") or ""))
        calls = [c for c in calls if "--fuzz" not in c]
        if not calls:
            continue
        counts = {k: 0 for k in _SUITE_KEYS}
        seed_shards: Dict[int, List[int]] = {}
        for call in calls:
            m = re.search(r"--suites (\S+)", call)
            keys = parse_suite_selection(m.group(1)) if m else list(_SUITE_KEYS)
            sm = re.search(r"--seed-shard (\d+)/(\d+)", call)
            if sm:
                if keys != ["12"]:
                    errors.append(f"{jid}: --seed-shard must select only Suite 12: {call!r}")
                seed_shards.setdefault(int(sm.group(2)), []).append(int(sm.group(1)))
                continue
            for k in keys:
                counts[k] += 1
        for n, idx in seed_shards.items():
            if sorted(idx) != list(range(n)):
                errors.append(f"{jid}: Suite 12 seed shards of /{n} are {sorted(idx)}, not 0..{n - 1}")
            counts["12"] += 1
        if len(seed_shards) > 1:
            errors.append(f"{jid}: Suite 12 split with several shard counts {sorted(seed_shards)}")
        for k, c in counts.items():
            if c != 1:
                errors.append(f"{jid}: suite {k} runs {c} times (must be exactly once)")
    return errors


class TestMatviewSyncSharding(unittest.TestCase):
    """CI shards test_matview_sync.py across parallel processes: no suite lost or doubled."""

    def test_ci_shards_partition_every_suite(self) -> None:
        self.assertEqual(matview_sync_partition_errors(_all_workflows()["ci.yml"]), [])

    def test_partition_check_detects_missing_doubled_and_bad_seed_shards(self) -> None:
        def wf(*runs: str) -> Dict[str, Any]:
            return {"jobs": {"j": {"steps": [{"run": r} for r in runs]}}}

        ok = wf('x --task "python3 test_matview_sync.py --suites 1-11,13,unit" --task '
                '"python3 test_matview_sync.py --suites 12 --seed-shard 0/2" '
                '--task "python3 test_matview_sync.py --suites 12 --seed-shard 1/2"')
        self.assertEqual(matview_sync_partition_errors(ok), [])
        self.assertEqual(matview_sync_partition_errors(wf("python3 test_matview_sync.py")), [])
        self.assertTrue(any("suite 5 runs 0" in e for e in matview_sync_partition_errors(
            wf("python3 test_matview_sync.py --suites 1-4,6-13,unit"))))
        self.assertTrue(any("suite 3 runs 2" in e for e in matview_sync_partition_errors(
            wf("python3 test_matview_sync.py --suites 1-13,unit", "python3 test_matview_sync.py --suites 3"))))
        self.assertTrue(any("not 0..1" in e for e in matview_sync_partition_errors(
            wf("python3 test_matview_sync.py --suites 1-11,13,unit",
               "python3 test_matview_sync.py --suites 12 --seed-shard 0/2"))))

    def test_seed_shards_partition_the_seed_list(self) -> None:
        from test_matview_sync import select_seed_shard

        seeds = [20261106, 1337, 424242, 992138513, 7]
        self.assertEqual(select_seed_shard(seeds, ""), seeds)
        for n in range(1, 7):
            parts = [select_seed_shard(seeds, f"{i}/{n}") for i in range(n)]
            self.assertEqual(sorted(x for part in parts for x in part), sorted(seeds))
        for bad in ("2/2", "1", "a/b", "-1/2"):
            with self.assertRaises(ValueError):
                select_seed_shard(seeds, bad)

    def test_suite_selection_rejects_typos(self) -> None:
        from test_matview_sync import parse_suite_selection

        self.assertEqual(parse_suite_selection("unit,13,2-3"), ["2", "3", "13", "unit"])
        for bad in ("", "0", "14", "3-1", "units", "1,,2"):
            with self.assertRaises(ValueError):
                parse_suite_selection(bad)


class TestCiParallelRunner(unittest.TestCase):
    """scripts/ci_parallel.py: failures propagate, tasks are never retried, setup retries are bounded."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import ci_parallel

        cls.cp = ci_parallel

    def _main(self, *args: str) -> Tuple[int, str]:
        buf = io.StringIO()
        env = {k: v for k, v in os.environ.items() if k not in ("POSTGRES_DSN", "MYSQL_DSN", "GITHUB_ACTIONS")}
        with patch.dict(os.environ, env, clear=True), patch("sys.stdout", buf):
            rc = self.cp.main(list(args))
        return rc, buf.getvalue()

    def test_all_tasks_run_and_a_failure_fails_the_run_without_retry(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            counter = Path(tmp) / "count"
            py = sys.executable
            rc, out = self._main(
                "--jobs", "2",
                "--task", f"{py} -c \"print('alpha-out')\"",
                "--task", f"{py} -c \"open('{counter}', 'a').write('x'); raise SystemExit(3)\"",
                "--serial", f"{py} -c \"import os; print('tmp-is-private', os.environ['TMPDIR'] != '{tmp}')\"",
            )
            self.assertEqual(rc, 1)
            self.assertEqual(counter.read_text(), "x", "a failing task must run exactly once (never retried)")
        self.assertIn("alpha-out", out)
        self.assertIn("tmp-is-private True", out)
        self.assertIn("FAILED (exit 3)", out)
        rc, _ = self._main("--jobs", "3", "--task", f"{sys.executable} -c pass")
        self.assertEqual(rc, 0)

    def test_each_task_gets_its_own_fuzz_manifest(self) -> None:
        py = sys.executable
        rc, out = self._main("--jobs", "2",
                             "--task", f"{py} -c \"import os; print('M=' + os.environ['MATVIEW_FUZZ_MANIFEST'])\"",
                             "--task", f"{py} -c \"import os; print('M=' + os.environ['MATVIEW_FUZZ_MANIFEST'])\"")
        self.assertEqual(rc, 0)
        manifests = re.findall(r"M=(fuzz-campaign\S+\.json)", out)
        self.assertEqual(len(set(manifests)), 2, manifests)

    def test_setup_retries_only_transient_errors_and_are_bounded(self) -> None:
        calls: List[int] = []

        def deadlock() -> None:
            calls.append(1)
            raise RuntimeError("(1213, 'Deadlock found when trying to get lock')")

        with patch.object(self.cp.time, "sleep"), patch("sys.stdout", io.StringIO()) as out:
            with self.assertRaises(RuntimeError):
                self.cp.with_setup_retries("CREATE DATABASE x", deadlock)
        self.assertEqual(len(calls), self.cp.SETUP_ATTEMPTS)
        self.assertEqual(out.getvalue().count("RETRY CREATE DATABASE x"), self.cp.SETUP_ATTEMPTS - 1)

        calls.clear()

        def syntax_error() -> None:
            calls.append(1)
            raise RuntimeError("(1064, 'You have an error in your SQL syntax')")

        with self.assertRaises(RuntimeError):
            self.cp.with_setup_retries("CREATE DATABASE x", syntax_error)
        self.assertEqual(len(calls), 1, "a non-transient error must not be retried")

    def test_one_at_a_time_tasks_never_overlap_while_others_fill_slots(self) -> None:
        import threading
        import time

        tasks = [self.cp.Task(i, ("heavy" if i < 3 else "light") + str(i), False) for i in range(9)]
        active: List[str] = []
        peak = {"heavy": 0, "all": 0}
        guard = threading.Lock()

        def run(task: Any) -> None:
            with guard:
                active.append(task.command)
                peak["heavy"] = max(peak["heavy"], sum(a.startswith("heavy") for a in active))
                peak["all"] = max(peak["all"], len(active))
            time.sleep(0.02)
            with guard:
                active.remove(task.command)
            task.rc = 0

        self.cp.run_parallel_phase(tasks, 3, lambda t: t.command.startswith("heavy"), run)
        self.assertEqual(peak["heavy"], 1)
        self.assertEqual(peak["all"], 3)
        self.assertTrue(all(t.rc == 0 for t in tasks), "every task must run")

    def test_a_hung_task_is_killed_and_its_partial_output_kept(self) -> None:
        py = sys.executable
        rc, out = self._main("--jobs", "1", "--task-timeout", "1",
                             "--task", f"{py} -c \"import time; print('partial-out', flush=True); time.sleep(30)\"")
        self.assertEqual(rc, 1)
        self.assertIn("partial-out", out)
        self.assertIn("TIMEOUT: killed after 1s", out)

    def test_every_workflow_invocation_parses(self) -> None:
        """Each `scripts/ci_parallel.py` step, read as bash reads it, is a valid runner command line."""
        import shlex

        seen = 0
        for fname, wf in _all_workflows().items():
            for jid, job in (wf.get("jobs") or {}).items():
                for step in job.get("steps") or []:
                    run = str(step.get("run") or "")
                    if "ci_parallel.py" not in run:
                        continue
                    seen += 1
                    words = shlex.split(run.replace("\\\n", " "), comments=True)
                    args = words[words.index("scripts/ci_parallel.py") + 1:]
                    with patch("sys.stderr", io.StringIO()) as err:
                        try:
                            parsed = self.cp.build_parser().parse_args(args)
                        except SystemExit:
                            self.fail(f"{fname} {jid}: ci_parallel arguments do not parse: {err.getvalue()}")
                    for cmd in parsed.task + parsed.serial:
                        self.assertRegex(cmd, r"^(python3|pytest) ", f"{fname} {jid}: odd task {cmd!r}")
        self.assertGreater(seen, 0)

    def test_task_dsn_names_the_task_database_only(self) -> None:
        self.assertEqual(self.cp._replace_db("postgresql://u:p@127.0.0.1:5432/matview_test", "mvw_task03"),
                         "postgresql://u:p@127.0.0.1:5432/mvw_task03")
        self.assertEqual(self.cp._replace_db("mysql://root:pw@127.0.0.1:3306/matview_test", "mvw_task00"),
                         "mysql://root:pw@127.0.0.1:3306/mvw_task00")


if __name__ == "__main__":
    unittest.main(verbosity=2)
