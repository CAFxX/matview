# Makefile for materialized view tests and Docker integration test environment

.PHONY: all test docker-up docker-down docker-test presubmit presubmit-full presubmit-check install-hooks test-presubmit help

PYTHON ?= python3
COMPOSE ?= docker compose

# Database services matching docker-compose.yml
DB_SERVICES = postgres15 postgres18 mysql84 mysql97 mariadb1011 mariadb118 mariadb123

# Default target runs local test suites
all: test

# Run local SQLite test suites: core matview sync, backfill concurrency, query AST fuzzer, AST materialization, and Docker env verification
test:
	$(PYTHON) test_matview_sync.py
	$(PYTHON) test_concurrent_backfill.py
	$(PYTHON) test_query_ast_fuzzer.py
	$(PYTHON) test_query_ast_materialization.py
	$(PYTHON) test_docker_test_env.py

# Spin up all integration test database containers and wait for health readiness
docker-up:
	$(COMPOSE) up -d --wait $(DB_SERVICES)

# Stop and clean up all integration test database containers, networks, and volumes
docker-down:
	$(COMPOSE) down -v --remove-orphans

# Spin up containers, wait for readiness, execute verify_engine_matrix.py and all test suites
# (test_matview_sync.py, test_concurrent_backfill.py, test_query_ast_fuzzer.py, test_query_ast_materialization.py)
# across PostgreSQL, MySQL, and MariaDB, then guarantee teardown cleanup
docker-test:
	@status=0; \
	trap '$(COMPOSE) down -v --remove-orphans' EXIT INT TERM; \
	$(COMPOSE) up -d --wait $(DB_SERVICES) && \
	$(COMPOSE) run --rm test-runner $(PYTHON) run_docker_tests.py || status=$$?; \
	exit $$status

# Local pre-push gate derived from .github/workflows/ci.yml (see scripts/presubmit.py --help).
# Fast tier: changed-path classification plus affected SQLite-job steps on the working tree.
presubmit:
	$(PYTHON) scripts/presubmit.py --tier fast

# Full tier: every PR-gate job and matrix row of HEAD; writes an attestation stamp only when all are green.
presubmit-full:
	$(PYTHON) scripts/presubmit.py --tier full

presubmit-check:
	$(PYTHON) scripts/presubmit.py --check-stamp

install-hooks:
	git config core.hooksPath .githooks

test-presubmit:
	$(PYTHON) scripts/test_presubmit.py

help:
	@echo "Available make targets:"
	@echo "  test         Run local test suite (test_matview_sync, test_concurrent_backfill, test_query_ast_fuzzer, test_query_ast_materialization, test_docker_test_env)"
	@echo "  docker-up    Start all database containers (postgres15/18, mysql84/97, mariadb1011/118/123) and wait for health readiness"
	@echo "  docker-down  Stop and remove all containers, volumes, and networks"
	@echo "  presubmit    Fast local pre-push checks (not a CI-equivalence claim)"
	@echo "  presubmit-full  Run every PR-gate CI job locally; stamps HEAD tree when fully green"
	@echo "  install-hooks   Use .githooks/pre-push (requires a valid stamp before pushing)"
	@echo "  docker-test  Start containers, wait for readiness, run verify_engine_matrix.py and test suites across all engines, and tear down cleanly"
