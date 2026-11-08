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


@dataclass(frozen=True)
class SelectAST:
    source: tuple[Any, ...]
    projections: tuple[tuple[Any, ...], ...]
    distinct: bool = False
    joins: tuple[tuple[Any, ...], ...] = ()
    where: tuple[Any, ...] | None = None
    group_by: tuple[tuple[Any, ...], ...] = ()
    having: tuple[Any, ...] | None = None
    cte: tuple[str, "SelectAST"] | None = None
    set_operation: tuple[str, "SelectAST"] | None = None
    order_by: tuple[tuple[Any, ...], ...] = ()
    limit: int | None = None


@dataclass(frozen=True)
class GeneratedQuery:
    sql: str
    features: frozenset[str]
    shape: tuple[Any, ...]
    max_predicate_depth: int
    max_expression_depth: int


def _render_expr(node: tuple[Any, ...]) -> str:
    kind = node[0]
    if kind == "column":
        return f"{_ident(node[1])}.{_ident(node[2])}"
    if kind == "literal":
        return _literal(node[1])
    if kind == "star":
        return "*"
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
    if kind == "aggregate":
        return f"{node[1]}({_render_expr(node[2])})"
    if kind == "window":
        return (
            f"{node[1]}() OVER (PARTITION BY {_render_expr(node[2])} "
            f"ORDER BY {_render_expr(node[3])})"
        )
    if kind == "scalar_count":
        return (
            f"(SELECT COUNT(*) FROM {_ident(node[1])} {_ident(node[2])} "
            f"WHERE {_ident(node[2])}.id = {_render_expr(node[3])})"
        )
    raise ValueError(f"unknown expression node: {kind}")


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
    if kind == "exists":
        return (
            f"EXISTS (SELECT 1 FROM {_ident(node[1])} {_ident(node[2])} "
            f"WHERE {_ident(node[2])}.id = {_render_expr(node[3])})"
        )
    if kind == "in_subquery":
        return (
            f"{_render_expr(node[1])} IN (SELECT {_ident(node[3])}.id "
            f"FROM {_ident(node[2])} {_ident(node[3])} "
            f"WHERE {_ident(node[3])}.qty IS NULL OR {_ident(node[3])}.qty >= 0)"
        )
    raise ValueError(f"unknown predicate node: {kind}")


def _render_select(query: SelectAST, *, include_order: bool = True) -> str:
    parts: list[str] = []
    if query.cte is not None:
        cte_name, cte_query = query.cte
        parts.append(f"WITH {_ident(cte_name)} AS ({_render_select(cte_query)}) ")

    projections = ", ".join(_render_expr(expr) for expr in query.projections)
    parts.append("SELECT " + ("DISTINCT " if query.distinct else "") + projections)

    source = query.source
    if source[0] == "table":
        parts.append(f"FROM {_ident(source[1])} {_ident(source[2])}")
    elif source[0] == "derived":
        parts.append(f"FROM ({_render_select(source[1])}) {_ident(source[2])}")
    elif source[0] == "cte":
        parts.append(f"FROM {_ident(source[1])} {_ident(source[2])}")
    else:
        raise ValueError(f"unknown source node: {source[0]}")

    for table, alias, predicate in query.joins:
        parts.append(
            f"JOIN {_ident(table)} {_ident(alias)} ON {_render_predicate(predicate)}"
        )
    if query.where is not None:
        parts.append("WHERE " + _render_predicate(query.where))
    if query.group_by:
        parts.append("GROUP BY " + ", ".join(_render_expr(expr) for expr in query.group_by))
    if query.having is not None:
        parts.append("HAVING " + _render_predicate(query.having))

    sql = " ".join(parts)
    if query.set_operation is not None:
        operator, right = query.set_operation
        sql += f" {operator} {_render_select(right, include_order=False)}"
    if include_order and query.order_by:
        sql += " ORDER BY " + ", ".join(_render_expr(expr) for expr in query.order_by)
    if include_order and query.limit is not None:
        sql += f" LIMIT {query.limit}"
    return sql


def _shape_expr(node: tuple[Any, ...]) -> tuple[Any, ...]:
    kind = node[0]
    if kind == "column":
        return ("column",)
    if kind == "literal":
        return ("literal", type(node[1]).__name__)
    return (kind,) + tuple(
        _shape_predicate(part) if isinstance(part, tuple) and part and part[0] in {
            "is_null", "compare", "between", "not", "and", "or", "exists", "in_subquery"
        } else _shape_expr(part) if isinstance(part, tuple) else part
        for part in node[1:]
    )


def _shape_predicate(node: tuple[Any, ...]) -> tuple[Any, ...]:
    kind = node[0]
    if kind == "is_null":
        return (kind, _shape_expr(node[1]))
    if kind == "compare":
        return (kind, node[2], _shape_expr(node[1]), _shape_expr(node[3]))
    if kind == "between":
        return (kind, _shape_expr(node[1]))
    if kind == "not":
        return (kind, _shape_predicate(node[1]))
    if kind in ("and", "or"):
        return (kind, _shape_predicate(node[1]), _shape_predicate(node[2]))
    if kind in ("exists", "in_subquery"):
        return (kind,)
    raise ValueError(f"unknown predicate node: {kind}")


class SchemaAwareSelectFuzzer:
    """Compose SELECT clauses and nested query nodes from schema-derived identifiers."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.features: set[str] = set()
        self.max_predicate_depth = 0
        self.max_expression_depth = 0

    @staticmethod
    def _schema(conn: sqlite3.Connection) -> tuple[list[str], list[tuple[str, str, str]]]:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'vs*_t*' ORDER BY name"
            )
        ]
        foreign_keys: list[tuple[str, str, str]] = []
        for table in tables:
            for row in conn.execute(f"PRAGMA foreign_key_list({_literal(table)})"):
                foreign_keys.append((table, row[3], row[2]))
        return tables, foreign_keys

    def _predicate(
        self, alias: str, depth: int, tables: list[str]
    ) -> tuple[Any, ...]:
        rng = self.rng
        if depth <= 0 or rng.random() < 0.36:
            atom = rng.choice(("id_eq", "id_between", "qty_null", "qty_ge", "amount_le", "label_null"))
            self.features.add("predicate_atom")
            if atom == "id_eq":
                return ("compare", ("column", alias, "id"), "=", ("literal", rng.randint(1, 8)))
            if atom == "id_between":
                lo = rng.randint(1, 4)
                return ("between", ("column", alias, "id"), lo, lo + rng.randint(0, 4))
            if atom == "qty_null":
                return ("is_null", ("column", alias, "qty"))
            if atom == "qty_ge":
                return (
                    "compare",
                    ("coalesce", ("column", alias, "qty"), ("literal", 0)),
                    ">=",
                    ("literal", rng.randint(0, 60)),
                )
            if atom == "amount_le":
                return ("compare", ("column", alias, "amount"), "<=", ("literal", rng.randint(0, 100)))
            return ("is_null", ("column", alias, "label"))

        operator = rng.choice(("and", "or", "not", "exists", "in_subquery"))
        self.features.add(operator)
        if operator == "not":
            return ("not", self._predicate(alias, depth - 1, tables))
        if operator in ("and", "or"):
            return (
                operator,
                self._predicate(alias, depth - 1, tables),
                self._predicate(alias, depth - 1, tables),
            )
        other = rng.choice(tables)
        sub_alias = "sq"
        if operator == "exists":
            return ("exists", other, sub_alias, ("column", alias, "id"))
        return ("in_subquery", ("column", alias, "id"), other, sub_alias)

    def _expression(self, alias: str, depth: int, tables: list[str]) -> tuple[Any, ...]:
        rng = self.rng
        if depth <= 0:
            return ("column", alias, rng.choice(("qty", "amount")))
        kind = rng.choice(("coalesce", "add", "multiply", "case", "scalar_count"))
        self.features.add("expression_" + kind)
        self.max_expression_depth = max(self.max_expression_depth, depth)
        if kind == "coalesce":
            return (
                "coalesce",
                ("column", alias, rng.choice(("qty", "amount"))),
                ("literal", rng.randint(0, 5)),
            )
        if kind in ("add", "multiply"):
            return (
                kind,
                self._expression(alias, depth - 1, tables),
                ("literal", rng.randint(1, 5)),
            )
        if kind == "case":
            return (
                "case",
                ("is_null", ("column", alias, "qty")),
                ("literal", rng.randint(1, 5)),
                self._expression(alias, depth - 1, tables),
            )
        return ("scalar_count", rng.choice(tables), "sq", ("column", alias, "id"))

    def _build_select(
        self,
        tables: list[str],
        foreign_keys: list[tuple[str, str, str]],
        *,
        depth: int,
        simple: bool = False,
    ) -> SelectAST:
        rng = self.rng
        table = rng.choice(tables)
        alias = "t"
        cte: tuple[str, SelectAST] | None = None
        source: tuple[Any, ...] = ("table", table, alias)
        available_alias = alias

        if not simple and depth > 0 and rng.random() < 0.14:
            cte_query = self._build_select(tables, foreign_keys, depth=depth - 1, simple=True)
            cte = ("q", cte_query)
            source = ("cte", "q", "c")
            available_alias = "c"
            self.features.add("cte")
            self.features.add("nested_select")
        elif not simple and depth > 0 and rng.random() < 0.14:
            derived = self._build_select(tables, foreign_keys, depth=depth - 1, simple=True)
            source = ("derived", derived, "d")
            available_alias = "d"
            self.features.add("derived_table")
            self.features.add("nested_select")

        joins: list[tuple[Any, ...]] = []
        if source[0] == "table" and not simple and rng.random() < 0.48:
            relation = next((fk for fk in foreign_keys if fk[0] == table), None)
            if relation is not None:
                child, child_col, parent = relation
                join_table = parent
                on = ("compare", ("column", alias, child_col), "=", ("column", "j", "id"))
                joins.append((join_table, "j", on))
                self.features.add("foreign_key_join")
            else:
                child_relation = next((fk for fk in foreign_keys if fk[2] == table), None)
                if child_relation is not None:
                    child, child_col, _ = child_relation
                    on = ("compare", ("column", "j", child_col), "=", ("column", alias, "id"))
                    joins.append((child, "j", on))
                    self.features.add("foreign_key_join")
                else:
                    other = rng.choice([name for name in tables if name != table])
                    on = ("compare", ("column", alias, "id"), "=", ("column", "j", "id"))
                    joins.append((other, "j", on))
                    self.features.add("key_join")
            self.features.add("join")

        where = None
        if not simple and rng.random() < 0.78:
            where = self._predicate(available_alias, 3, tables)
            self.features.add("where")
            self.max_predicate_depth = max(
                self.max_predicate_depth, self._depth_predicate(where)
            )

        group = not simple and rng.random() < 0.30
        distinct = not simple and not group and rng.random() < 0.26
        group_by: tuple[tuple[Any, ...], ...] = ()
        having = None
        if simple:
            # Nested sources expose a stable, schema-known projection to their parent.
            projections = tuple(
                ("column", available_alias, name)
                for name in ("id", "label", "qty", "amount")
            )
        elif group:
            group_by = (("column", available_alias, "label"),)
            projections = (
                ("column", available_alias, "label"),
                ("aggregate", "COUNT", ("star",)),
                ("aggregate", "SUM", ("column", available_alias, "qty")),
                ("aggregate", "AVG", ("column", available_alias, "amount")),
            )
            having = (
                "compare",
                ("aggregate", "COUNT", ("star",)),
                ">=",
                ("literal", rng.randint(1, 2)),
            )
            self.features.update(("group_by", "aggregate", "having"))
        else:
            projections_list: list[tuple[Any, ...]] = [
                ("column", available_alias, "id"),
                ("column", available_alias, "label"),
                self._expression(available_alias, 2, tables),
            ]
            if depth > 0 and rng.random() < 0.24:
                projections_list.append(
                    ("window", "ROW_NUMBER", ("column", available_alias, "label"),
                     ("column", available_alias, "id"))
                )
                self.features.update(("window", "partition_by"))
            projections = tuple(projections_list)

        if distinct:
            self.features.add("distinct")

        set_operation = None
        if not simple and depth > 0 and not group and rng.random() < 0.18:
            # Set-operation branches have matching one-column projections.
            projections = (("column", available_alias, "id"),)
            right_table = rng.choice([name for name in tables if name != table])
            right = SelectAST(
                source=("table", right_table, "r"),
                projections=(("column", "r", "id"),),
                where=("compare", ("column", "r", "id"), ">=", ("literal", rng.randint(1, 3))),
            )
            set_operation = (rng.choice(("UNION", "UNION ALL", "INTERSECT", "EXCEPT")), right)
            self.features.update(("set_operation", set_operation[0].lower()))

        order_by = () if simple or rng.random() < 0.22 else (("column", available_alias, "id"),)
        if order_by:
            self.features.add("order_by")
        limit = None if simple or rng.random() < 0.30 else rng.randint(2, 8)
        if limit is not None:
            self.features.add("limit")

        return SelectAST(
            source=source,
            projections=projections,
            distinct=distinct,
            joins=tuple(joins),
            where=where,
            group_by=group_by,
            having=having,
            cte=cte,
            set_operation=set_operation,
            order_by=order_by,
            limit=limit,
        )

    @staticmethod
    def _depth_predicate(node: tuple[Any, ...]) -> int:
        if node[0] in ("not",):
            return 1 + SchemaAwareSelectFuzzer._depth_predicate(node[1])
        if node[0] in ("and", "or"):
            return 1 + max(
                SchemaAwareSelectFuzzer._depth_predicate(node[1]),
                SchemaAwareSelectFuzzer._depth_predicate(node[2]),
            )
        return 1

    def _shape(self, query: SelectAST) -> tuple[Any, ...]:
        source = query.source[0]
        source_shape = (source,)
        if source == "derived":
            source_shape += (self._shape(query.source[1]),)
        return (
            "select",
            source_shape,
            tuple(_shape_expr(expr) for expr in query.projections),
            query.distinct,
            bool(query.joins),
            _shape_predicate(query.where) if query.where else None,
            bool(query.group_by),
            _shape_predicate(query.having) if query.having else None,
            bool(query.cte),
            query.set_operation[0] if query.set_operation else None,
            bool(query.order_by),
            query.limit is not None,
        )

    def generate(self, conn: sqlite3.Connection) -> GeneratedQuery:
        tables, foreign_keys = self._schema(conn)
        if len(tables) < 2:
            raise ValueError("query generation requires at least two generated tables")
        query = self._build_select(tables, foreign_keys, depth=2)
        return GeneratedQuery(
            sql=_render_select(query),
            features=frozenset(self.features),
            shape=self._shape(query),
            max_predicate_depth=self.max_predicate_depth,
            max_expression_depth=self.max_expression_depth,
        )


class RecursiveQueryASTTests(unittest.TestCase):
    def test_generated_select_grammar_executes_over_variable_schemas(self) -> None:
        features: set[str] = set()
        shapes: set[tuple[Any, ...]] = set()
        executed = 0

        for seed in range(64):
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

                for query_index in range(20):
                    fuzzer = SchemaAwareSelectFuzzer(seed * 1009 + query_index)
                    query = fuzzer.generate(conn)
                    rows = conn.execute(query.sql).fetchall()
                    self.assertIsInstance(rows, list)
                    features.update(query.features)
                    shapes.add(query.shape)
                    self.assertLessEqual(query.max_predicate_depth, 4)
                    self.assertLessEqual(query.max_expression_depth, 2)
                    executed += 1
            finally:
                conn.close()

        required = {
            "where", "join", "group_by", "aggregate", "having", "distinct",
            "cte", "derived_table", "set_operation", "window", "partition_by",
            "nested_select", "foreign_key_join", "expression_coalesce",
            "expression_add", "expression_multiply", "expression_case",
            "expression_scalar_count", "exists", "in_subquery", "and", "or", "not",
        }
        missing = required - features
        self.assertFalse(missing, f"missing generated query features: {sorted(missing)}")
        self.assertGreaterEqual(len(shapes), 50)
        self.assertEqual(executed, 64 * 20)
        print(
            f"[QUERY AST] executed={executed}, structural_shapes={len(shapes)}, "
            f"features={','.join(sorted(features))}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)