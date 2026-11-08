#!/usr/bin/env python3
"""Differentially exercise generated SELECTs as maintained materialized views."""

from __future__ import annotations

import sqlite3
import unittest

from matview_sync import EngineLimitationError, MaterializedViewManager
from test_matview_sync import (
    _register_sqlite_compat_udfs,
    _rows_equal,
    VariableSchemaDifferentialFuzzer,
)
from test_query_ast_fuzzer import SchemaAwareSelectFuzzer


class QueryASTMaterializedViewTests(unittest.TestCase):
    def test_generated_selects_match_materialized_views_through_dml(self) -> None:
        supported = 0
        explicitly_limited = 0
        observed_features: set[str] = set()

        # Keep this focused and deterministic. The separate query-AST test
        # explores 64 schemas x 20 queries; this test takes a smaller sample
        # through the compiler, trigger installation, and DML maintenance path.
        for schema_seed in range(8):
            schema, _, seed_sql, mutations, _ = (
                VariableSchemaDifferentialFuzzer(schema_seed).generate_scenario(1)
            )
            for query_index in range(4):
                conn = sqlite3.connect(":memory:")
                try:
                    conn.execute("PRAGMA foreign_keys = ON;")
                    _register_sqlite_compat_udfs(conn)
                    conn.executescript(schema)
                    for statement in seed_sql:
                        conn.execute(statement)
                    conn.commit()

                    generated = SchemaAwareSelectFuzzer(
                        schema_seed * 1009 + query_index
                    ).generate(conn)
                    observed_features.update(generated.features)
                    view_name = f"ast_mv_{schema_seed}_{query_index}"
                    view_sql = f"CREATE VIEW {view_name} AS {generated.sql};"
                    manager = MaterializedViewManager(conn)

                    try:
                        plan = manager.register_view(view_sql)
                    except EngineLimitationError as exc:
                        codes = {lim.code for lim in exc.limitations}
                        self.assertTrue(
                            codes and all(codes),
                            f"unsupported generated plan lacked stable limitation codes: {exc}",
                        )
                        explicitly_limited += 1
                        continue

                    supported += 1

                    def assert_matches(stage: str) -> None:
                        actual = conn.execute(
                            f"SELECT * FROM {plan.db_view_name}"
                        ).fetchall()
                        expected = conn.execute(generated.sql).fetchall()
                        self.assertTrue(
                            _rows_equal(actual, expected),
                            f"materialized SELECT mismatch at {stage}; "
                            f"seed={schema_seed}, query={query_index}, SQL={generated.sql!r}, "
                            f"actual={actual!r}, expected={expected!r}",
                        )

                    assert_matches("initial population")
                    for mutation_index, (_, statement) in enumerate(mutations):
                        conn.execute(statement)
                        assert_matches(f"mutation {mutation_index}: {statement}")
                finally:
                    conn.close()

        self.assertGreater(supported, 0, "no generated SELECT was tested as a materialized view")
        self.assertEqual(supported + explicitly_limited, 8 * 4)
        self.assertTrue(observed_features)
        print(
            "[QUERY AST MV DIFFERENTIAL] "
            f"supported={supported}, explicitly_limited={explicitly_limited}, "
            f"features={','.join(sorted(observed_features))}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
