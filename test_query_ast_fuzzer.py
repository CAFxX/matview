#!/usr/bin/env python3
"""Bounded recursive, schema-aware SELECT AST generator and execution tests."""

from __future__ import annotations

from dataclasses import dataclass
import random
import sqlite3
import unittest
from typing import Any

from test_matview_sync import VariableSchemaDifferentialFuzzer


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _expr_shape(node: tuple[Any, ...]) -> tuple[Any, ...]:
    kind = node[0]
    if kind == "column":
        return ("column",)
    if kind == "literal":
        return ("literal", type(node[1]).__name__)
    return (kind,) + tuple(
        _expr_shape(part) if isinstance(part, tuple) else part
        for part in node[1:]
    )


def _render_expr(node: tuple[Any, ...]) -> str:
    kind = node[0]
    if kind == "column":
        return f"{_ident(node[1])}.{_ident(node[2])}"
    if kind == "literal":
        return _literal(node[1])
    if kind == "coalesce":
        return f"COALESCE({_render_expr(node[1])}, {_render_expr(node[2])})"
    if kind in ("add", "multiply"):
        op = "+" if kind == "add" else "*"
        return f"({_render_expr(node[1])} {op} {_render_expr(node[2])})"
    if kind == "case":
        return (
            f"(CASE WHEN {_render_predicate(node[1])} "
            f"THEN {_render_expr(node[2])} ELSE {_render_expr(node[3])} END)"
        )
    raise ValueError(f"unknown expression node: {kind}")


def _predicate_shape(node: tuple[Any, ...]) -> tuple[Any, ...]:
    kind = node[0]
    if kind == "is_null":
        return ("is_null", _expr_shape(node[1]))
    if kind == "compare":
        return ("compare", node[2], _expr_shape(node[1]), _expr_shape(node[3]))
    if kind == "between":
        return ("between", _expr_shape(node[1]))
    if kind == "not":
        return ("not", _predicate_shape(node[1]))
    if kind in ("and", "or"):
        return (kind, _predicate_shape(node[1]), _predicate_shape(node[2]))
    raise ValueError(f"unknown predicate node: {kind}")


def _render_predicate(node: tuple[Any, ...]) -> str:
    kind = node[0]
    if kind == "is_null":
        return f"{_render_expr(node[1])} IS NULL"
    if kind == "compare":
        return f"{_render_expr(node[1])} {node[2]} {_render_expr(node[3])}"
    if kind == "between":
        return f"{_render_expr(node[1])} BETWEEN {_literal(node[2])} AND {_literal(node[3])}"
    if kind == "not":
        return f"(NOT {_render_predicate(node[1])})"
    if kind in ("and", "or"):
        op = "AND" if kind == "and" else "OR"
        return f"({_render_predicate(node[1])} {op} {_render_predicate(node[2])})"
    raise ValueError(f"unknown predicate node: {kind}")


def _predicate(
    rng: random.Random, alias: str, depth: int
) -> tuple[tuple[Any, ...], int]:
    """Build a recursive predicate AST using columns present in every generated table."""
    if depth <= 0 or rng.random() < 0.38:
        atom = rng.choice(("id_eq", "id_between", "qty_null", "qty_ge", "amount_le", "label_null", "label_eq"))
        if atom == "id_eq":
            node = ("compare", ("column", alias, "id"), "=", ("literal", rng.randint(1, 8)))
        elif atom == "id_between":
            lo = rng.randint(1, 4)
            node = ("between", ("column", alias, "id"), lo, lo + rng.randint(0, 4))
        elif atom == "qty_null":
            node = ("is_null", ("column", alias, "qty"))
        elif atom == "qty_ge":
            node = (
                "compare",
                ("coalesce", ("column", alias, "qty"), ("literal", 0)),
                ">=",
                ("literal", rng.randint(0, 60)),
            )
        elif atom == "amount_le":
            node = ("compare", ("column", alias, "amount"), "<=", ("literal", rng.randint(0, 100)))
        elif atom == "label_null":
            node = ("is_null", ("column", alias, "label"))
        else:
            node = (
                "compare",
                ("column", alias, "label"),
                "=",
                ("literal", f"vs1_{rng.randint(0, 5)}_{rng.randint(1, 4)}"),
            )
        return node, 1

    operator = rng.choice(("and", "or", "not"))
    left, left_depth = _predicate(rng, alias, depth - 1)
    if operator == "not":
        return ("not", left), left_depth + 1
    right, right_depth = _predicate(rng, alias, depth - 1)
    return (operator, left, right), max(left_depth, right_depth) + 1


@dataclass(frozen=True)
class GeneratedQuery:
    sql: str
    kind: str
    features: frozenset[str]
    shape: tuple[Any, ...]
    max_predicate_depth: int
    max_expression_depth: int


class SchemaAwareSelectFuzzer:
    """Generate a bounded recursive SELECT grammar from live SQLite schema metadata."""

    KINDS = (
        "basic", "join", "group_having", "distinct", "cte",
        "derived_table", "set_operation", "window", "exists", "nested_subquery",
    )

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    @staticmethod
    def _schema(conn: sqlite3.Connection) -> tuple[list[str], dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'vs*_t*' ORDER BY name"
            )
        ]
        columns: dict[str, list[tuple[str, str]]] = {}
        foreign_keys: list[tuple[str, str, str]] = []
        for table in tables:
            columns[table] = [
                (row[1], (row[2] or "").upper())
                for row in conn.execute(f"PRAGMA table_info({_literal(table)})")
            ]
            for row in conn.execute(f"PRAGMA foreign_key_list({_literal(table)})"):
                foreign_keys.append((table, row[3], row[2]))
        return tables, columns, foreign_keys

    def generate(self, conn: sqlite3.Connection, query_index: int) -> GeneratedQuery:
        tables, columns, foreign_keys = self._schema(conn)
        if len(tables) < 2:
            raise ValueError("query generation requires at least two generated tables")
        kind = self.KINDS[query_index % len(self.KINDS)]
        rng = self.rng
        table = rng.choice(tables)
        other = rng.choice([name for name in tables if name != table])
        alias = "t"
        predicate, predicate_depth = _predicate(rng, alias, 3)
        features = {"projection"}
        expr_depth = 0

        if kind == "basic":
            value_expr = (
                "add",
                ("coalesce", ("column", alias, "qty"), ("literal", 0)),
                ("literal", rng.randint(1, 9)),
            )
            sql = (
                f"SELECT t.id, t.label, {_render_expr(value_expr)} AS adjusted "
                f"FROM {_ident(table)} t WHERE {_render_predicate(predicate)} "
                f"ORDER BY t.id LIMIT {rng.randint(2, 8)}"
            )
            features.update(("filter", "recursive_predicate", "expression", "order", "limit"))
            expr_depth = 2
            shape = ("select", "basic", _predicate_shape(predicate), _expr_shape(value_expr))

        elif kind == "join":
            fk = next((item for item in foreign_keys if item[0] == table), None)
            if fk:
                child, child_col, parent = fk
                sql = (
                    f"SELECT p.id, c.id, c.label FROM {_ident(parent)} p "
                    f"INNER JOIN {_ident(child)} c ON p.id = c.{_ident(child_col)} "
                    f"WHERE {_render_predicate(_predicate(rng, 'c', 2)[0])} ORDER BY p.id, c.id"
                )
            else:
                sql = (
                    f"SELECT a.id, b.id, b.label FROM {_ident(table)} a "
                    f"INNER JOIN {_ident(other)} b ON a.id = b.id "
                    f"WHERE a.id >= 1 ORDER BY a.id, b.id"
                )
            features.update(("join", "filter", "order"))
            shape = ("select", "join", "foreign_key" if fk else "key_equality")

        elif kind == "group_having":
            sql = (
                f"SELECT t.label, COUNT(*) AS n, SUM(t.qty) AS total_qty, AVG(t.amount) AS mean_amount "
                f"FROM {_ident(table)} t GROUP BY t.label HAVING COUNT(*) >= {rng.randint(1, 2)} "
                f"ORDER BY t.label"
            )
            features.update(("group_by", "aggregate", "having", "order"))
            shape = ("select", "group_having", "count_sum_avg")

        elif kind == "distinct":
            sql = (
                f"SELECT DISTINCT t.label FROM {_ident(table)} t "
                f"WHERE {_render_predicate(predicate)} ORDER BY t.label LIMIT {rng.randint(2, 8)}"
            )
            features.update(("distinct", "filter", "order", "limit"))
            shape = ("select", "distinct", _predicate_shape(predicate))

        elif kind == "cte":
            sql = (
                f"WITH q AS (SELECT id, label, qty FROM {_ident(table)} "
                f"WHERE {_render_predicate(predicate)}) "
                f"SELECT q.id, q.label FROM q WHERE q.qty IS NULL OR q.qty >= 0 "
                f"ORDER BY q.id LIMIT {rng.randint(2, 8)}"
            )
            features.update(("cte", "nested_select", "filter", "order", "limit"))
            shape = ("select", "cte", _predicate_shape(predicate))

        elif kind == "derived_table":
            sql = (
                f"SELECT d.label, d.total FROM "
                f"(SELECT label, SUM(qty) AS total FROM {_ident(table)} GROUP BY label) d "
                f"WHERE d.total IS NOT NULL ORDER BY d.label"
            )
            features.update(("derived_table", "nested_select", "group_by", "aggregate", "filter"))
            shape = ("select", "derived_table", "group_sum")

        elif kind == "set_operation":
            op = rng.choice(("UNION ALL", "UNION", "INTERSECT", "EXCEPT"))
            sql = (
                f"SELECT id FROM {_ident(table)} WHERE id <= {rng.randint(1, 5)} "
                f"{op} SELECT id FROM {_ident(other)} WHERE id >= {rng.randint(1, 3)} "
                f"ORDER BY id LIMIT {rng.randint(2, 8)}"
            )
            features.update(("set_operation", op.lower(), "filter", "order", "limit"))
            shape = ("select", "set_operation", op)

        elif kind == "window":
            sql = (
                f"SELECT t.id, t.label, ROW_NUMBER() OVER "
                f"(PARTITION BY t.label ORDER BY t.id) AS rn "
                f"FROM {_ident(table)} t ORDER BY t.id"
            )
            features.update(("window", "partition_by", "order"))
            shape = ("select", "window", "row_number_partition_order")

        elif kind == "exists":
            fk = next((item for item in foreign_keys if item[0] == table), None)
            if fk:
                child, child_col, parent = fk
                sql = (
                    f"SELECT c.id FROM {_ident(child)} c WHERE EXISTS "
                    f"(SELECT 1 FROM {_ident(parent)} p WHERE p.id = c.{_ident(child_col)}) "
                    f"ORDER BY c.id"
                )
                shape = ("select", "exists", "correlated_fk")
            else:
                sql = (
                    f"SELECT t.id FROM {_ident(table)} t WHERE t.id IN "
                    f"(SELECT x.id FROM {_ident(other)} x WHERE x.qty IS NULL OR x.qty >= 0) "
                    f"ORDER BY t.id"
                )
                shape = ("select", "in_subquery", "uncorrelated")
            features.update(("subquery", "exists_or_in", "correlation"))
            
        else:
            sql = (
                f"SELECT t.id, (SELECT COUNT(*) FROM {_ident(other)} x WHERE x.id = t.id) AS matches "
                f"FROM {_ident(table)} t WHERE t.id IN "
                f"(SELECT y.id FROM {_ident(other)} y WHERE {_render_predicate(_predicate(rng, 'y', 2)[0])}) "
                f"ORDER BY t.id"
            )
            features.update(("scalar_subquery", "in_subquery", "nested_select", "correlation", "filter"))
            shape = ("select", "nested_subqueries", "scalar_count_and_in")

        if kind not in ("join", "group_having", "derived_table", "set_operation", "window", "exists", "nested_subquery"):
            features.add("recursive_predicate")
        return GeneratedQuery(
            sql=sql,
            kind=kind,
            features=frozenset(features),
            shape=shape,
            max_predicate_depth=predicate_depth,
            max_expression_depth=expr_depth,
        )


class RecursiveQueryASTTests(unittest.TestCase):
    def test_generated_select_grammar_executes_over_variable_schemas(self) -> None:
        features: set[str] = set()
        shapes: set[tuple[Any, ...]] = set()
        executed = 0

        for seed in range(32):
            schema, _, seed_sql, mutations, _ = VariableSchemaDifferentialFuzzer(seed).generate_scenario(1)
            conn = sqlite3.connect(":memory:")
            try:
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.executescript(schema)
                for statement in seed_sql:
                    conn.execute(statement)
                conn.commit()
                for _, statement in mutations:
                    conn.execute(statement)
                conn.commit()

                fuzzer = SchemaAwareSelectFuzzer(seed ^ 0x5A17)
                for query_index in range(len(SchemaAwareSelectFuzzer.KINDS) * 2):
                    query = fuzzer.generate(conn, query_index)
                    # Execute the rendered AST against the exact schema and data
                    # from which its identifiers, joins, and predicates were drawn.
                    rows = conn.execute(query.sql).fetchall()
                    self.assertIsInstance(rows, list)
                    features.update(query.features)
                    shapes.add(query.shape)
                    self.assertLessEqual(query.max_predicate_depth, 4)
                    self.assertLessEqual(query.max_expression_depth, 3)
                    executed += 1
            finally:
                conn.close()

        required = {
            "projection", "filter", "recursive_predicate", "expression", "order", "limit",
            "join", "group_by", "aggregate", "having", "distinct", "cte",
            "derived_table", "set_operation", "window", "partition_by",
            "subquery", "exists_or_in", "scalar_subquery", "correlation",
        }
        self.assertTrue(required <= features, f"missing query grammar features: {sorted(required - features)}")
        self.assertGreaterEqual(len(shapes), 10)
        self.assertEqual(executed, 32 * len(SchemaAwareSelectFuzzer.KINDS) * 2)
        print(
            f"[QUERY AST] executed={executed}, structural_shapes={len(shapes)}, "
            f"features={','.join(sorted(features))}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)