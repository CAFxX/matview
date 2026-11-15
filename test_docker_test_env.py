#!/usr/bin/env python3
"""Unit tests verifying Dockerfile, docker-compose.yml, Makefile, and test runner configurations."""

from __future__ import annotations

import ast
import os
import re
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent


def parse_compose_yaml(text: str) -> Dict[str, Any]:
    """Pure-Python standard-library YAML parser specifically for docker-compose files.

    Avoids external dependencies like PyYAML which are not declared in requirements.txt.
    """
    def _parse_val(v: str) -> Any:
        v = v.strip()
        if not v:
            return None
        # Handle flow sequences e.g. ["CMD-SHELL", "..."]
        if v.startswith("[") and v.endswith("]"):
            try:
                return ast.literal_eval(v)
            except Exception:
                pass
        # Handle quoted strings
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        # Handle integers
        if v.isdigit():
            return int(v)
        return v

    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    i = 0
    while i < len(lines):
        raw = lines[i]
        # Remove comments outside quotes
        stripped = raw.split("#")[0].rstrip()
        if not stripped.strip():
            i += 1
            continue

        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.strip()

        # Pop stack levels deeper than current indentation
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]

        # List item: '- value' or '- key: value'
        if content.startswith("- "):
            item_val = content[2:].strip()
            if isinstance(parent, list):
                is_quoted = (item_val.startswith('"') and item_val.endswith('"')) or (
                    item_val.startswith("'") and item_val.endswith("'")
                )
                if ":" in item_val and not item_val.startswith("--") and not is_quoted:
                    k, v = item_val.split(":", 1)
                    parsed_v = _parse_val(v)
                    elem = {k.strip(): parsed_v} if parsed_v is not None else {k.strip(): {}}
                    parent.append(elem)
                    stack.append((indent, elem))
                else:
                    parent.append(_parse_val(item_val))
            i += 1
            continue

        # Key-value or container header: 'key: value' or 'key:'
        if ":" in content:
            key, val = content.split(":", 1)
            key = key.strip()
            val = val.strip()

            # Look ahead to see if child is a list or dictionary
            next_is_list = False
            next_indent = indent
            for next_line in lines[i + 1:]:
                ns = next_line.split("#")[0].rstrip()
                if ns.strip():
                    next_indent = len(ns) - len(ns.lstrip(" "))
                    next_is_list = ns.strip().startswith("- ")
                    break

            if not val:
                new_container: Any = [] if (next_is_list and next_indent > indent) else {}
                if isinstance(parent, dict):
                    parent[key] = new_container
                elif isinstance(parent, list):
                    parent.append({key: new_container})
                stack.append((indent, new_container))
            else:
                parsed_v = _parse_val(val)
                if isinstance(parent, dict):
                    parent[key] = parsed_v
                elif isinstance(parent, list):
                    parent.append({key: parsed_v})
        i += 1

    return root


def load_compose(text: str) -> Dict[str, Any]:
    """Parse docker-compose.yml using yaml.safe_load if available, or fallback to stdlib."""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except (ImportError, Exception):
        return parse_compose_yaml(text)


class DockerfileValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dockerfile_path = REPO_ROOT / "Dockerfile"
        self.assertTrue(self.dockerfile_path.is_file(), "Dockerfile must exist at repository root")
        self.content = self.dockerfile_path.read_text(encoding="utf-8")

    def test_base_image_is_python_312(self) -> None:
        match = re.search(r"^FROM\s+python:(3\.12(?:-\w+)?)\b", self.content, re.MULTILINE)
        self.assertIsNotNone(
            match,
            f"Dockerfile must use Python 3.12 base image, found content: {self.content[:200]!r}",
        )

    def test_installs_system_build_dependencies_and_drivers(self) -> None:
        required_pkgs = ["build-essential", "libsqlcipher-dev", "sqlcipher", "libpq-dev"]
        for pkg in required_pkgs:
            self.assertIn(
                pkg,
                self.content,
                f"Dockerfile must install system package {pkg!r} for database/encryption driver support",
            )

    def test_installs_requirements_txt(self) -> None:
        self.assertIn("requirements.txt", self.content, "Dockerfile must copy and install requirements.txt")
        self.assertIn("-r requirements.txt", self.content, "Dockerfile must pip install -r requirements.txt")

    def test_installs_optional_sqlite_drivers(self) -> None:
        drivers = ["apsw", "sqlcipher", "libsql"]
        for driver in drivers:
            self.assertIn(
                driver,
                self.content,
                f"Dockerfile must install optional SQLite driver {driver!r} to match CI matrix",
            )


class DockerComposeValidationTests(unittest.TestCase):
    REQUIRED_SERVICES = {
        "postgres15": "postgres:15",
        "postgres18": "postgres:18",
        "mysql84": "mysql:8.4",
        "mysql97": "mysql:9.7",
        "mariadb1011": "mariadb:10.11",
        "mariadb118": "mariadb:11.8",
        "mariadb123": "mariadb:12.3",
    }

    def setUp(self) -> None:
        self.compose_path = REPO_ROOT / "docker-compose.yml"
        self.assertTrue(self.compose_path.is_file(), "docker-compose.yml must exist at repository root")
        self.raw_content = self.compose_path.read_text(encoding="utf-8")
        self.compose_data = parse_compose_yaml(self.raw_content)
        self.assertIsInstance(self.compose_data, dict, "docker-compose.yml must define a top-level mapping")
        self.assertIn("services", self.compose_data, "docker-compose.yml must contain a 'services' section")
        self.services: Dict[str, Any] = self.compose_data["services"]

    def test_stdlib_parser_works_independently_of_pyyaml(self) -> None:
        parsed = parse_compose_yaml(self.raw_content)
        self.assertIn("services", parsed)
        self.assertEqual(len(parsed["services"]), 8)
        self.assertEqual(parsed["services"]["postgres15"]["image"], "postgres:15")
        self.assertEqual(parsed["services"]["mysql84"]["environment"]["MYSQL_ROOT_PASSWORD"], "password")

    def test_required_services_and_exact_images_present(self) -> None:
        for srv_name, expected_image in self.REQUIRED_SERVICES.items():
            self.assertIn(srv_name, self.services, f"Service {srv_name!r} must be defined in docker-compose.yml")
            service = self.services[srv_name]
            self.assertEqual(
                service.get("image"),
                expected_image,
                f"Service {srv_name} must use image {expected_image!r}",
            )

    def test_distinct_host_ports_and_default_primary_ports(self) -> None:
        host_ports: List[int] = []
        port_mappings: Dict[str, int] = {}

        for srv_name in self.REQUIRED_SERVICES:
            service = self.services[srv_name]
            ports = service.get("ports", [])
            self.assertTrue(ports, f"Service {srv_name} must expose port mapping")

            port_entry = str(ports[0])
            match = re.search(r":(\d+)$", port_entry)
            self.assertIsNotNone(match, f"Invalid port entry {port_entry} for {srv_name}")

            host_match = re.search(r"(\d+)(?::\d+)?$", port_entry.split(":")[0]) or re.search(r":-?(\d+)", port_entry)
            self.assertIsNotNone(host_match, f"Cannot parse host port in {port_entry} for {srv_name}")
            default_host_port = int(host_match.group(1))

            host_ports.append(default_host_port)
            port_mappings[srv_name] = default_host_port

        self.assertEqual(
            len(host_ports),
            len(set(host_ports)),
            f"All database services must have distinct host ports: {port_mappings}",
        )
        self.assertEqual(port_mappings["postgres15"], 5432, "postgres15 must map default primary port 5432")
        self.assertEqual(port_mappings["mysql84"], 3306, "mysql84 must map default primary port 3306")

    def test_postgresql_environment_and_healthchecks(self) -> None:
        for srv in ("postgres15", "postgres18"):
            service = self.services[srv]
            env = service.get("environment", {})
            self.assertEqual(env.get("POSTGRES_USER"), "postgres")
            self.assertEqual(env.get("POSTGRES_PASSWORD"), "password")
            self.assertEqual(env.get("POSTGRES_DB"), "matview_test")

            healthcheck = service.get("healthcheck", {})
            test_cmd = str(healthcheck.get("test", ""))
            self.assertIn("pg_isready", test_cmd, f"{srv} healthcheck must use pg_isready")
            self.assertIn("-U postgres", test_cmd)
            self.assertIn("-d matview_test", test_cmd)
            self.assertIn("interval", healthcheck)
            self.assertIn("retries", healthcheck)

    def test_mysql_and_mariadb_utf8_collations_and_performance_flags(self) -> None:
        required_flags = [
            "--character-set-server=utf8mb4",
            "--collation-server=utf8mb4_unicode_ci",
            "--innodb-flush-log-at-trx-commit=0",
            "--sync-binlog=0",
            "--skip-log-bin",
        ]
        for srv in ("mysql84", "mysql97", "mariadb1011", "mariadb118", "mariadb123"):
            service = self.services[srv]
            cmd = service.get("command", [])
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            for flag in required_flags:
                self.assertIn(
                    flag,
                    cmd_str,
                    f"Service {srv} command must include required flag {flag!r}",
                )

    def test_mysql_and_mariadb_healthchecks_and_credentials(self) -> None:
        for srv in ("mysql84", "mysql97"):
            service = self.services[srv]
            env = service.get("environment", {})
            self.assertEqual(env.get("MYSQL_ROOT_PASSWORD"), "password")
            self.assertEqual(env.get("MYSQL_DATABASE"), "matview_test")

            healthcheck = service.get("healthcheck", {})
            test_cmd = str(healthcheck.get("test", ""))
            self.assertIn("mysqladmin ping", test_cmd)
            self.assertIn("-uroot", test_cmd)
            self.assertIn("-ppassword", test_cmd)

        for srv in ("mariadb1011", "mariadb118", "mariadb123"):
            service = self.services[srv]
            env = service.get("environment", {})
            self.assertEqual(env.get("MYSQL_ROOT_PASSWORD"), "password")
            self.assertEqual(env.get("MARIADB_ROOT_PASSWORD"), "password")
            self.assertEqual(env.get("MYSQL_DATABASE"), "matview_test")
            self.assertEqual(env.get("MARIADB_DATABASE"), "matview_test")

            healthcheck = service.get("healthcheck", {})
            test_cmd = str(healthcheck.get("test", ""))
            self.assertTrue(
                "mariadb-admin ping" in test_cmd or "mysqladmin ping" in test_cmd,
                f"MariaDB service {srv} healthcheck must execute ping test",
            )

    def test_test_runner_service_dependencies_and_dsns(self) -> None:
        self.assertIn("test-runner", self.services, "docker-compose.yml must define test-runner service")
        runner = self.services["test-runner"]

        depends_on = runner.get("depends_on", {})
        for srv in self.REQUIRED_SERVICES:
            self.assertIn(srv, depends_on, f"test-runner must depend on {srv}")
            condition = depends_on[srv].get("condition") if isinstance(depends_on[srv], dict) else None
            self.assertEqual(
                condition,
                "service_healthy",
                f"test-runner dependency {srv} must require condition: service_healthy",
            )

        env = runner.get("environment", {})
        expected_dsn_keys = [
            "POSTGRES15_DSN",
            "POSTGRES18_DSN",
            "MYSQL84_DSN",
            "MYSQL97_DSN",
            "MARIADB1011_DSN",
            "MARIADB118_DSN",
            "MARIADB123_DSN",
        ]
        for key in expected_dsn_keys:
            self.assertIn(key, env, f"test-runner must provide {key} environment variable")


class MakefileValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.makefile_path = REPO_ROOT / "Makefile"
        self.assertTrue(self.makefile_path.is_file(), "Makefile must exist at repository root")
        self.content = self.makefile_path.read_text(encoding="utf-8")

    def test_makefile_has_required_targets(self) -> None:
        required_targets = ["test", "docker-up", "docker-down", "docker-test"]
        for target in required_targets:
            pattern = rf"^{re.escape(target)}\s*:"
            self.assertIsNotNone(
                re.search(pattern, self.content, re.MULTILINE),
                f"Makefile must define target {target!r}",
            )

    def test_make_test_runs_local_sqlite_suites(self) -> None:
        required_suites = [
            "test_matview_sync.py",
            "test_concurrent_backfill.py",
            "test_query_ast_fuzzer.py",
            "test_query_ast_materialization.py",
            "test_docker_test_env.py",
        ]
        for suite in required_suites:
            self.assertIn(
                suite,
                self.content,
                f"Makefile must reference and execute local suite {suite!r}",
            )

    def test_docker_up_waits_for_services(self) -> None:
        self.assertIn("--wait", self.content, "make docker-up must use --wait to ensure service readiness")

    def test_docker_down_cleans_up_volumes(self) -> None:
        self.assertIn("down", self.content, "make docker-down must run compose down")
        self.assertIn("-v", self.content, "make docker-down must remove test volumes")

    def test_docker_test_wires_engine_verification_and_suites(self) -> None:
        self.assertIn(
            "verify_engine_matrix.py",
            self.content,
            "Makefile must wire verify_engine_matrix.py for native engine verification",
        )
        self.assertIn("run_docker_tests.py", self.content, "make docker-test must invoke test runner script")
        self.assertIn("trap", self.content, "make docker-test must use trap to guarantee teardown on exit")


class DockerRunnerScriptValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner_path = REPO_ROOT / "run_docker_tests.py"
        self.assertTrue(self.runner_path.is_file(), "run_docker_tests.py must exist at repository root")
        self.content = self.runner_path.read_text(encoding="utf-8")

    def test_syntax_is_valid_python(self) -> None:
        try:
            ast.parse(self.content, filename="run_docker_tests.py")
        except SyntaxError as e:
            self.fail(f"run_docker_tests.py has syntax error: {e}")

    def test_runner_defines_all_matrix_engines(self) -> None:
        expected_engines = {
            "postgres15",
            "postgres18",
            "mysql84",
            "mysql97",
            "mariadb1011",
            "mariadb118",
            "mariadb123",
        }
        for eng in expected_engines:
            self.assertIn(eng, self.content, f"run_docker_tests.py must configure engine {eng!r}")

    def test_runner_executes_verification_and_suites(self) -> None:
        suites = [
            "verify_engine_matrix.py",
            "test_matview_sync.py",
            "test_concurrent_backfill.py",
            "test_query_ast_fuzzer.py",
            "test_query_ast_materialization.py",
        ]
        for s in suites:
            self.assertIn(s, self.content, f"run_docker_tests.py must invoke suite {s!r}")


class TodoCompletionValidationTests(unittest.TestCase):
    def test_docker_compose_todo_marked_complete(self) -> None:
        todo_path = REPO_ROOT / "TODO.md"
        self.assertTrue(todo_path.is_file(), "TODO.md must exist")
        content = todo_path.read_text(encoding="utf-8")
        match = re.search(
            r"-\s*\[x\]\s*\*\*Docker\s*/\s*Docker Compose Integration Test Environment",
            content,
            re.IGNORECASE,
        )
        self.assertIsNotNone(
            match,
            "TODO.md must mark 'Docker / Docker Compose Integration Test Environment (make docker-test)' complete with '[x]'",
        )


class CIWorkflowValidationTests(unittest.TestCase):
    def test_ci_workflow_wires_docker_test_env(self) -> None:
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        self.assertTrue(ci_path.is_file(), ".github/workflows/ci.yml must exist")
        content = ci_path.read_text(encoding="utf-8")
        self.assertIn("test_docker_test_env.py", content, "ci.yml must execute test_docker_test_env.py in sqlite job")


if __name__ == "__main__":
    unittest.main(verbosity=2)
