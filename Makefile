# Makefile for materialized view tests and Docker integration test environment

.PHONY: all test test-regressions docker-up docker-down docker-test presubmit presubmit-full presubmit-check install-hooks test-presubmit help

PYTHON ?= python3
COMPOSE ?= docker compose

# Database services matching docker-compose.yml
DB_SERVICES = postgres15 postgres18 mysql84 mysql97 mariadb1011 mariadb118 mariadb123

# Default target runs the complete local SQLite-oriented suite.
all: test

# Standalone regression programs that are not part of the core suites below:
# every root test module must be listed here or in `test` (enforced by
# test_readme_examples.py::MakefileCoverage).
# These use SQLite by default; native-engine portions skip unless their DSNs are set.
REGRESSION_TESTS = \
	test_sqlite_runtime_boundaries.py \
	test_native_result_types.py \
	test_native_statistical_aggregates.py \
	test_avg_concurrent_maintenance.py \
	test_native_bulk_statements.py \
	test_native_dml_interactions.py \
	test_oracle_mutation_sensitivity.py \
	test_sqlite_writer_connections.py \
	test_registration_replacement.py \
	test_registration_lifecycle.py \
	test_bulk_changes_safety.py \
	test_oracle_adversarial_boundaries.py \
	test_sqlite_exact_float_aggregates.py \
	test_minmax_group_arg_index.py \
	test_p0_bulk_changes_isolation.py \
	test_p0_pk_swap_and_collation.py \
	test_p0_backfill_watermark_safety.py \
	test_p0_compiler_alias_aux_pk_safety.py \
	test_p1_dialect_parsing.py \
	test_cli_diff.py \
	test_p1_packaging_version.py \
	test_p1_manager_safety.py \
	test_p1_engine_codegen.py \
	test_p1_hypothesis_stateful.py \
	test_p1_quote_identifiers.py \
	test_p1_pg_transition_vectorization.py \
	test_p1_expr_storage_types.py \
	test_p1_sqlglot_builders.py \
	test_p1_jinja2_templates.py \
	test_p1_sqlalchemy_integration.py \
	test_p1_modular_package.py \
	test_p1_user_view_indexes.py \
	test_p1_static_typing_gate.py \
	test_p2_ci_matrix_verification.py \
	test_p2_index_intelligence.py \
	test_p2_window_partition_scoping.py \
	test_pgmy_exact_float_aggregates.py \
	test_pg_statement_maintenance.py \
	test_pg_logical_subscription.py \
	test_readme_examples.py \
	test_schema_graph_minimizer.py \
	test_unrecognized_aggregates.py \
	test_sqlite_strict_sources.py \
	test_sqlite_strict_diagnostics.py \
	test_native_function_classification.py \
	test_group_key_collation.py \
	test_sqlite_table_shapes.py \
	test_sqlite_distinct_filtered_aggregates.py

test:
	$(PYTHON) test_matview_sync.py
	$(PYTHON) test_concurrent_backfill.py
	$(PYTHON) test_query_ast_fuzzer.py
	$(PYTHON) test_query_ast_materialization.py
	$(PYTHON) test_docker_test_env.py
	$(MAKE) test-regressions

test-regressions:
	@set -e; for test in $(REGRESSION_TESTS); do \
		case "$$test" in \
			test_native_bulk_statements.py|test_native_dml_interactions.py|test_p1_hypothesis_stateful.py|test_pg_statement_maintenance.py|test_pg_logical_subscription.py) \
				$(PYTHON) -m pytest -v "$$test" ;; \
			*) $(PYTHON) "$$test" ;; \
		esac; \
	done
	$(PYTHON) test_strength_reporting.py

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
	@echo "  test         Run core SQLite suites plus all standalone regression programs and strength reporting"
	@echo "  test-regressions  Run standalone regression programs and strength reporting"
	@echo "  docker-up    Start all database containers (postgres15/18, mysql84/97, mariadb1011/118/123) and wait for health readiness"
	@echo "  docker-down  Stop and remove all containers, volumes, and networks"
	@echo "  presubmit    Fast local pre-push checks (not a CI-equivalence claim)"
	@echo "  presubmit-full  Run every PR-gate CI job locally; stamps HEAD tree when fully green"
	@echo "  install-hooks   Use .githooks/pre-push (requires a valid stamp before pushing)"
	@echo "  docker-test  Start containers, wait for readiness, run verify_engine_matrix.py and test suites across all engines, and tear down cleanly"
