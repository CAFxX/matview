#!/usr/bin/env python3
"""Execute full materialized-view test matrix against Docker / Docker Compose engine containers."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


ENGINES: List[Dict[str, Any]] = [
    {
        "id": "postgres15",
        "name": "PostgreSQL 15",
        "image": "postgres:15",
        "type": "postgres",
        "host_port": 5432,
        "container_port": 5432,
        "default_host": "postgres15",
        "dsn_env_key": "POSTGRES15_DSN",
        "verify_env": {
            "EXPECTED_PG_MAJOR": "15",
        },
        "target_label": "PostgreSQL 15",
    },
    {
        "id": "postgres18",
        "name": "PostgreSQL 18",
        "image": "postgres:18",
        "type": "postgres",
        "host_port": 5433,
        "container_port": 5432,
        "default_host": "postgres18",
        "dsn_env_key": "POSTGRES18_DSN",
        "verify_env": {
            "EXPECTED_PG_MAJOR": "18",
        },
        "target_label": "PostgreSQL 18",
    },
    {
        "id": "mysql84",
        "name": "MySQL 8.4",
        "image": "mysql:8.4",
        "type": "mysql",
        "host_port": 3306,
        "container_port": 3306,
        "default_host": "mysql84",
        "dsn_env_key": "MYSQL84_DSN",
        "verify_env": {
            "EXPECTED_MYSQL_PRODUCT": "mysql",
            "EXPECTED_MYSQL_VERSION": "8.4",
            "MATVIEW_CI_REQUESTED_IMAGE": "mysql:8.4",
            "MATVIEW_CI_ACTUAL_IMAGE": "mysql:8.4",
        },
        "target_label": "MySQL 8.4",
    },
    {
        "id": "mysql97",
        "name": "MySQL 9.7",
        "image": "mysql:9.7",
        "type": "mysql",
        "host_port": 3307,
        "container_port": 3306,
        "default_host": "mysql97",
        "dsn_env_key": "MYSQL97_DSN",
        "verify_env": {
            "EXPECTED_MYSQL_PRODUCT": "mysql",
            "EXPECTED_MYSQL_VERSION": "9.7",
            "MATVIEW_CI_REQUESTED_IMAGE": "mysql:9.7",
            "MATVIEW_CI_ACTUAL_IMAGE": "mysql:9.7",
        },
        "target_label": "MySQL 9.7",
    },
    {
        "id": "mariadb1011",
        "name": "MariaDB 10.11",
        "image": "mariadb:10.11",
        "type": "mariadb",
        "host_port": 3308,
        "container_port": 3306,
        "default_host": "mariadb1011",
        "dsn_env_key": "MARIADB1011_DSN",
        "verify_env": {
            "EXPECTED_MYSQL_PRODUCT": "mariadb",
            "EXPECTED_MYSQL_VERSION": "10.11",
            "MATVIEW_CI_REQUESTED_IMAGE": "mariadb:10.11",
            "MATVIEW_CI_ACTUAL_IMAGE": "mariadb:10.11",
        },
        "target_label": "MariaDB 10.11 LTS",
    },
    {
        "id": "mariadb118",
        "name": "MariaDB 11.8",
        "image": "mariadb:11.8",
        "type": "mariadb",
        "host_port": 3309,
        "container_port": 3306,
        "default_host": "mariadb118",
        "dsn_env_key": "MARIADB118_DSN",
        "verify_env": {
            "EXPECTED_MYSQL_PRODUCT": "mariadb",
            "EXPECTED_MYSQL_VERSION": "11.8",
            "MATVIEW_CI_REQUESTED_IMAGE": "mariadb:11.8",
            "MATVIEW_CI_ACTUAL_IMAGE": "mariadb:11.8",
        },
        "target_label": "MariaDB 11.8 LTS",
    },
    {
        "id": "mariadb123",
        "name": "MariaDB 12.3",
        "image": "mariadb:12.3",
        "type": "mariadb",
        "host_port": 3310,
        "container_port": 3306,
        "default_host": "mariadb123",
        "dsn_env_key": "MARIADB123_DSN",
        "verify_env": {
            "EXPECTED_MYSQL_PRODUCT": "mariadb",
            "EXPECTED_MYSQL_VERSION": "12.3",
            "MATVIEW_CI_REQUESTED_IMAGE": "mariadb:12.3",
            "MATVIEW_CI_ACTUAL_IMAGE": "mariadb:12.3",
        },
        "target_label": "MariaDB 12.3 LTS",
    },
]


def resolve_dsn(engine: Dict[str, Any], use_host: bool) -> str:
    key = engine["dsn_env_key"]
    if key in os.environ and os.environ[key]:
        return os.environ[key]

    if engine["type"] == "postgres":
        if use_host:
            return f"postgresql://postgres:password@127.0.0.1:{engine['host_port']}/matview_test"
        return f"postgresql://postgres:password@{engine['default_host']}:{engine['container_port']}/matview_test"
    else:
        if use_host:
            return f"mysql://root:password@127.0.0.1:{engine['host_port']}/matview_test"
        return f"mysql://root:password@{engine['default_host']}:{engine['container_port']}/matview_test"


def run_step(description: str, cmd: List[str], env_overrides: Optional[Dict[str, str]] = None) -> None:
    print(f"\n>>> [RUN] {description}")
    print(f"    Command: {' '.join(cmd)}")
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
        for k, v in env_overrides.items():
            print(f"    Env: {k}={v}")

    start = time.time()
    res = subprocess.run(cmd, env=env)
    elapsed = time.time() - start
    if res.returncode != 0:
        print(f"!!! [FAIL] {description} exited with code {res.returncode} ({elapsed:.2f}s)")
        sys.exit(res.returncode)
    print(f"*** [PASS] {description} ({elapsed:.2f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run test matrix against Docker Compose database containers.")
    parser.add_argument(
        "--host",
        action="store_true",
        help="Connect to database containers via host-mapped ports on 127.0.0.1 instead of container hostnames",
    )
    parser.add_argument(
        "--engine",
        default="all",
        help="Engine identifier to test (e.g. postgres15, postgres18, mysql84, mysql97, mariadb1011, mariadb118, mariadb123, or 'all', 'sqlite')",
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="Skip local SQLite and AST fuzzer suites",
    )
    args = parser.parse_args()

    python_bin = sys.executable

    # 1. Run local SQLite and AST test suites unless explicitly filtered out
    if args.engine in ("all", "sqlite") and not args.skip_sqlite:
        print("\n============================================================")
        print("Running Base SQLite and Query AST Test Suites")
        print("============================================================")
        run_step("Query AST Recursive Generator Fuzzer", [python_bin, "test_query_ast_fuzzer.py"])
        run_step("Query AST Materialized View Differential Tests", [python_bin, "test_query_ast_materialization.py"])
        run_step("Core Materialized View Sync Tests (SQLite)", [python_bin, "test_matview_sync.py"])
        run_step("Incremental Backfill Concurrency Tests (SQLite WAL)", [python_bin, "test_concurrent_backfill.py"])

    if args.engine == "sqlite":
        print("\n[SUCCESS] SQLite test suites completed successfully.")
        return

    # 2. Filter engines
    selected_engines = [e for e in ENGINES if args.engine == "all" or args.engine == e["id"]]
    if not selected_engines:
        print(f"Error: No engines matching filter {args.engine!r}. Available: {[e['id'] for e in ENGINES]}")
        sys.exit(1)

    print("\n============================================================")
    print(f"Running Live Database Integration Test Matrix ({len(selected_engines)} engines)")
    print("============================================================")

    for eng in selected_engines:
        dsn = resolve_dsn(eng, args.host)
        print(f"\n------------------------------------------------------------")
        print(f"Engine: {eng['name']} (image: {eng['image']})")
        print(f"DSN: {dsn}")
        print(f"------------------------------------------------------------")

        # Environment variables for this engine
        engine_env: Dict[str, str] = {
            "MATVIEW_FUZZ_TARGET": eng["target_label"],
        }
        if eng["type"] == "postgres":
            engine_env["POSTGRES_DSN"] = dsn
        else:
            engine_env["MYSQL_DSN"] = dsn

        # Merge verification environment variables
        verify_env = engine_env.copy()
        verify_env.update(eng["verify_env"])

        # Step A: Verify actual database engine and version
        run_step(f"Verify Native Engine [{eng['name']}]", [python_bin, "verify_engine_matrix.py"], verify_env)

        # Step B: Run full test suite with live integration tests
        run_step(f"Run test_matview_sync.py [{eng['name']}]", [python_bin, "test_matview_sync.py"], engine_env)

        # Step C: Run concurrent backfill tests
        run_step(f"Run test_concurrent_backfill.py [{eng['name']}]", [python_bin, "test_concurrent_backfill.py"], engine_env)

    print("\n============================================================")
    print("ALL DOCKER INTEGRATION TEST SUITES COMPLETED SUCCESSFULLY!")
    print("============================================================")


if __name__ == "__main__":
    main()
