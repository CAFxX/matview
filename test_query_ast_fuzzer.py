#!/usr/bin/env python3
"""Bounded recursive, schema-aware SELECT AST generator and execution tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
import re
import sqlite3
import unittest
from typing import Any, Callable, Sequence

from test_matview_sync import VariableSchemaDifferentialFuzzer


def _ident(name: str, dialect: Any = None) -> str:
    from matview_sync import Dialect
    if dialect == Dialect.MYSQL:
        return '`' + name.replace('`', '``') + '`'
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
class ColumnInfo:
    name: str
    data_type: str = "TEXT"
    is_pk: bool = False
    not_null: bool = False


@dataclass(frozen=True)
class GeneratedQuery:
    sql: str
    features: frozenset[str]
    shape: tuple[Any, ...]
    max_predicate_depth: int
    max_expression_depth: int


def _render_expr(node: tuple[Any, ...], dialect: Any = None) -> str:
    kind = node[0]
    if kind == "column":
        return f"{_ident(node[1], dialect)}.{_ident(node[2], dialect)}"
    if kind == "bare_column":
        return _ident(node[1], dialect)
    if kind == "literal":
        return _literal(node[1])
    if kind == "star":
        return "*"
    if kind == "coalesce":
        return f"COALESCE({_render_expr(node[1], dialect)}, {_render_expr(node[2], dialect)})"
    if kind in ("add", "multiply"):
        op = "+" if kind == "add" else "*"
        return f"({_render_expr(node[1], dialect)} {op} {_render_expr(node[2], dialect)})"
    if kind == "case":
        return (
            f"(CASE WHEN {_render_predicate(node[1], dialect)} "
            f"THEN {_render_expr(node[2], dialect)} ELSE {_render_expr(node[3], dialect)} END)"
        )
    if kind == "aggregate":
        return f"{node[1]}({_render_expr(node[2], dialect)})"
    if kind == "window":
        return (
            f"{node[1]}() OVER (PARTITION BY {_render_expr(node[2], dialect)} "
            f"ORDER BY {_render_expr(node[3], dialect)})"
        )
    if kind == "scalar_count":
        target_col = node[3] if len(node) > 4 else "id"
        expr = node[4] if len(node) > 4 else node[3]
        return (
            f"(SELECT COUNT(*) FROM {_ident(node[1], dialect)} {_ident(node[2], dialect)} "
            f"WHERE {_ident(node[2], dialect)}.{_ident(target_col, dialect)} = {_render_expr(expr, dialect)})"
        )
    raise ValueError(f"unknown expression node: {kind}")


def _render_predicate(node: tuple[Any, ...], dialect: Any = None) -> str:
    kind = node[0]
    if kind == "is_null":
        return f"{_render_expr(node[1], dialect)} IS NULL"
    if kind == "compare":
        return f"{_render_expr(node[1], dialect)} {node[2]} {_render_expr(node[3], dialect)}"
    if kind == "between":
        return f"{_render_expr(node[1], dialect)} BETWEEN {_literal(node[2])} AND {_literal(node[3])}"
    if kind == "not":
        return f"(NOT {_render_predicate(node[1], dialect)})"
    if kind in ("and", "or"):
        op = "AND" if kind == "and" else "OR"
        return f"({_render_predicate(node[1], dialect)} {op} {_render_predicate(node[2], dialect)})"
    if kind == "exists":
        target_col = node[3] if len(node) > 4 else "id"
        expr = node[4] if len(node) > 4 else node[3]
        return (
            f"EXISTS (SELECT 1 FROM {_ident(node[1], dialect)} {_ident(node[2], dialect)} "
            f"WHERE {_ident(node[2], dialect)}.{_ident(target_col, dialect)} = {_render_expr(expr, dialect)})"
        )
    if kind == "in_subquery":
        sub_col = node[4] if len(node) > 4 else "id"
        filter_col = node[5] if len(node) > 5 else "qty"
        where_clause = (
            f" WHERE {_ident(node[3], dialect)}.{_ident(filter_col, dialect)} IS NULL OR "
            f"{_ident(node[3], dialect)}.{_ident(filter_col, dialect)} >= 0"
            if filter_col is not None
            else ""
        )
        return (
            f"{_render_expr(node[1], dialect)} IN (SELECT {_ident(node[3], dialect)}.{_ident(sub_col, dialect)} "
            f"FROM {_ident(node[2], dialect)} {_ident(node[3], dialect)}{where_clause})"
        )
    raise ValueError(f"unknown predicate node: {kind}")


def _render_select(query: SelectAST, *, include_order: bool = True, dialect: Any = None) -> str:
    parts: list[str] = []
    if query.cte is not None:
        cte_name, cte_query = query.cte
        parts.append(f"WITH {_ident(cte_name, dialect)} AS ({_render_select(cte_query, dialect=dialect)}) ")

    projections = ", ".join(_render_expr(expr, dialect) for expr in query.projections)
    parts.append("SELECT " + ("DISTINCT " if query.distinct else "") + projections)

    source = query.source
    if source[0] == "table":
        parts.append(f"FROM {_ident(source[1], dialect)} {_ident(source[2], dialect)}")
    elif source[0] == "derived":
        parts.append(f"FROM ({_render_select(source[1], dialect=dialect)}) {_ident(source[2], dialect)}")
    elif source[0] == "cte":
        parts.append(f"FROM {_ident(source[1], dialect)} {_ident(source[2], dialect)}")
    else:
        raise ValueError(f"unknown source node: {source[0]}")

    for table, alias, predicate in query.joins:
        parts.append(
            f"JOIN {_ident(table, dialect)} {_ident(alias, dialect)} ON {_render_predicate(predicate, dialect)}"
        )
    if query.where is not None:
        parts.append("WHERE " + _render_predicate(query.where, dialect))
    if query.group_by:
        parts.append("GROUP BY " + ", ".join(_render_expr(expr, dialect) for expr in query.group_by))
    if query.having is not None:
        parts.append("HAVING " + _render_predicate(query.having, dialect))

    sql = " ".join(parts)
    if query.set_operation is not None:
        operator, right = query.set_operation
        sql += f" {operator} {_render_select(right, include_order=False, dialect=dialect)}"
    if include_order and query.order_by:
        sql += " ORDER BY " + ", ".join(_render_expr(expr, dialect) for expr in query.order_by)
    if include_order and query.limit is not None:
        sql += f" LIMIT {query.limit}"
    return sql


def _shape_expr(node: tuple[Any, ...]) -> tuple[Any, ...]:
    kind = node[0]
    if kind == "column":
        return ("column",)
    if kind == "bare_column":
        return ("bare_column",)
    if kind == "literal":
        return ("literal", type(node[1]).__name__)
    if kind == "scalar_count":
        expr = node[4] if len(node) > 4 else node[3]
        return ("scalar_count", node[1], node[2], _shape_expr(expr))
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
        self._table_columns: dict[str, list[ColumnInfo]] = {}
        self._current_scope_columns: dict[str, tuple[str, str, str, str]] = {}

    @staticmethod
    def _schema(conn: Any) -> tuple[list[str], list[tuple[str, str, str]]]:
        if isinstance(conn, sqlite3.Connection) or type(conn).__module__.startswith("sqlite3"):
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name GLOB 'vs*_t*' ORDER BY name"
                )
            ]
            if not tables:
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '__matview_%' "
                        "ORDER BY name"
                    )
                ]
            foreign_keys: list[tuple[str, str, str]] = []
            for table in tables:
                for row in conn.execute(f"PRAGMA foreign_key_list({_literal(table)})"):
                    foreign_keys.append((table, row[3], row[2]))
            return tables, foreign_keys

        cur = conn.cursor() if hasattr(conn, "cursor") else conn
        try:
            from matview_sync import detect_engine_version, Dialect
            dialect = detect_engine_version(conn).dialect
        except Exception:
            dialect = None

        from matview_sync import Dialect
        if dialect == Dialect.POSTGRES:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "AND table_type = 'BASE TABLE' "
                "AND table_name LIKE 'vs%#_t%' ESCAPE '#' "
                "ORDER BY table_name"
            )
            tables = [row[0] for row in (cur.fetchall() or [])]
            if not tables:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = current_schema() "
                    "AND table_type = 'BASE TABLE' "
                    "AND table_name NOT LIKE '__matview_%' "
                    "ORDER BY table_name"
                )
                tables = [row[0] for row in (cur.fetchall() or [])]
            cur.execute(
                """
                SELECT
                    tc.table_name,
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = current_schema()
                """
            )
            foreign_keys = [(row[0], row[1], row[2]) for row in (cur.fetchall() or [])]
            return tables, foreign_keys

        if dialect == Dialect.MYSQL:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = DATABASE() "
                "AND table_type = 'BASE TABLE' "
                "AND table_name LIKE 'vs%#_t%' ESCAPE '#' "
                "ORDER BY table_name"
            )
            tables = [row[0] for row in (cur.fetchall() or [])]
            if not tables:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = DATABASE() "
                    "AND table_type = 'BASE TABLE' "
                    "AND table_name NOT LIKE '__matview_%' "
                    "ORDER BY table_name"
                )
                tables = [row[0] for row in (cur.fetchall() or [])]
            cur.execute(
                """
                SELECT
                    table_name,
                    column_name,
                    referenced_table_name
                FROM information_schema.key_column_usage
                WHERE table_schema = DATABASE()
                  AND referenced_table_name IS NOT NULL
                  AND table_name LIKE 'vs%#_t%' ESCAPE '#'
                """
            )
            foreign_keys = [(row[0], row[1], row[2]) for row in (cur.fetchall() or [])]
            return tables, foreign_keys

        return [], []

    @staticmethod
    def _introspect_columns(
        conn: Any, tables: list[str], dialect: Any = None
    ) -> dict[str, list[ColumnInfo]]:
        if not conn:
            return {}
        result: dict[str, list[ColumnInfo]] = {}
        if isinstance(conn, sqlite3.Connection) or type(conn).__module__.startswith("sqlite3"):
            for table in tables:
                cols: list[ColumnInfo] = []
                try:
                    for row in conn.execute(f"PRAGMA table_info({_literal(table)})"):
                        cols.append(
                            ColumnInfo(
                                name=str(row[1]),
                                data_type=str(row[2] or "TEXT").upper(),
                                is_pk=bool(row[5]),
                                not_null=bool(row[3] or row[5]),
                            )
                        )
                except Exception:
                    pass
                if cols:
                    result[table] = cols
            return result

        cur = conn.cursor() if hasattr(conn, "cursor") else conn
        if dialect is None:
            try:
                from matview_sync import detect_engine_version
                dialect = detect_engine_version(conn).dialect
            except Exception:
                dialect = None

        from matview_sync import Dialect
        if dialect == Dialect.POSTGRES:
            for table in tables:
                try:
                    cur.execute(
                        """
                        SELECT
                            c.column_name,
                            c.data_type,
                            c.is_nullable,
                            CASE WHEN pk.column_name IS NOT NULL THEN 1 ELSE 0 END AS is_pk
                        FROM information_schema.columns c
                        LEFT JOIN (
                            SELECT kcu.table_name, kcu.column_name
                            FROM information_schema.table_constraints tc
                            JOIN information_schema.key_column_usage kcu
                              ON tc.constraint_name = kcu.constraint_name
                              AND tc.table_schema = kcu.table_schema
                            WHERE tc.constraint_type = 'PRIMARY KEY'
                              AND tc.table_schema = current_schema()
                        ) pk ON c.table_name = pk.table_name AND c.column_name = pk.column_name
                        WHERE c.table_schema = current_schema() AND c.table_name = %s
                        ORDER BY c.ordinal_position
                        """,
                        (table,),
                    )
                    cols = [
                        ColumnInfo(
                            name=str(row[0]),
                            data_type=str(row[1] or "TEXT").upper(),
                            is_pk=bool(row[3]),
                            not_null=(str(row[2]).upper() == "NO" or bool(row[3])),
                        )
                        for row in (cur.fetchall() or [])
                    ]
                    if cols:
                        result[table] = cols
                except Exception:
                    pass
            return result

        if dialect == Dialect.MYSQL:
            for table in tables:
                try:
                    cur.execute(
                        """
                        SELECT
                            column_name,
                            data_type,
                            is_nullable,
                            CASE WHEN column_key = 'PRI' THEN 1 ELSE 0 END AS is_pk
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE() AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (table,),
                    )
                    cols = [
                        ColumnInfo(
                            name=str(row[0]),
                            data_type=str(row[1] or "TEXT").upper(),
                            is_pk=bool(row[3]),
                            not_null=(str(row[2]).upper() == "NO" or bool(row[3])),
                        )
                        for row in (cur.fetchall() or [])
                    ]
                    if cols:
                        result[table] = cols
                except Exception:
                    pass
            return result

        return result

    def _get_table_canonical_columns(self, table: str) -> tuple[str, str, str, str]:
        cols = self._table_columns.get(table)
        if not cols:
            return ("id", "label", "qty", "amount")

        # 1. key_col (priority: "id", is_pk, first col)
        key_col = "id" if any(c.name == "id" for c in cols) else None
        if key_col is None:
            pk_col = next((c.name for c in cols if c.is_pk), None)
            key_col = pk_col if pk_col is not None else cols[0].name

        # 2. text_col (priority: "label", char/text type, non-key col, first col)
        text_col = "label" if any(c.name == "label" for c in cols) else None
        if text_col is None:
            txt_cand = next(
                (
                    c.name
                    for c in cols
                    if any(t in c.data_type for t in ("CHAR", "TEXT", "CLOB", "STRING"))
                    and c.name != key_col
                ),
                None,
            )
            if txt_cand is not None:
                text_col = txt_cand
            else:
                other_col = next((c.name for c in cols if c.name != key_col), None)
                text_col = other_col if other_col is not None else cols[0].name

        # 3. num_col_1 (priority: "qty", numeric type, non-key/text col, any numeric, first col)
        num_col_1 = "qty" if any(c.name == "qty" for c in cols) else None
        if num_col_1 is None:
            num_cand = next(
                (
                    c.name
                    for c in cols
                    if any(
                        t in c.data_type
                        for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")
                    )
                    and c.name not in (key_col, text_col)
                ),
                None,
            )
            if num_cand is not None:
                num_col_1 = num_cand
            else:
                num_cand_any = next(
                    (
                        c.name
                        for c in cols
                        if any(
                            t in c.data_type
                            for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")
                        )
                    ),
                    None,
                )
                if num_cand_any is not None:
                    num_col_1 = num_cand_any
                else:
                    other_col = next((c.name for c in cols if c.name not in (key_col, text_col)), None)
                    num_col_1 = other_col if other_col is not None else cols[0].name

        # 4. num_col_2 (priority: "amount", distinct numeric, same numeric, non-key col, second col)
        num_col_2 = "amount" if any(c.name == "amount" for c in cols) else None
        if num_col_2 is None:
            num_cand = next(
                (
                    c.name
                    for c in cols
                    if any(
                        t in c.data_type
                        for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")
                    )
                    and c.name not in (key_col, text_col, num_col_1)
                ),
                None,
            )
            if num_cand is not None:
                num_col_2 = num_cand
            elif any(
                t in c.data_type
                for c in cols
                if c.name == num_col_1
                for t in ("INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")
            ):
                num_col_2 = num_col_1
            else:
                other_col = next((c.name for c in cols if c.name != key_col), None)
                num_col_2 = (
                    other_col
                    if other_col is not None
                    else (cols[1].name if len(cols) > 1 else cols[0].name)
                )

        return (key_col, text_col, num_col_1, num_col_2)

    def _predicate(
        self, alias: str, depth: int, tables: list[str]
    ) -> tuple[Any, ...]:
        rng = self.rng
        cols = self._current_scope_columns.get(alias, ("id", "label", "qty", "amount"))
        if depth <= 0 or rng.random() < 0.36:
            atom = rng.choice(("id_eq", "id_between", "qty_null", "qty_ge", "amount_le", "label_null"))
            self.features.add("predicate_atom")
            if atom == "id_eq":
                return ("compare", ("column", alias, cols[0]), "=", ("literal", rng.randint(1, 8)))
            if atom == "id_between":
                lo = rng.randint(1, 4)
                return ("between", ("column", alias, cols[0]), lo, lo + rng.randint(0, 4))
            if atom == "qty_null":
                return ("is_null", ("column", alias, cols[2]))
            if atom == "qty_ge":
                return (
                    "compare",
                    ("coalesce", ("column", alias, cols[2]), ("literal", 0)),
                    ">=",
                    ("literal", rng.randint(0, 60)),
                )
            if atom == "amount_le":
                return ("compare", ("column", alias, cols[3]), "<=", ("literal", rng.randint(0, 100)))
            return ("is_null", ("column", alias, cols[1]))

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
        other_cols = self._get_table_canonical_columns(other)
        if operator == "exists":
            return ("exists", other, sub_alias, other_cols[0], ("column", alias, cols[0]))
        return ("in_subquery", ("column", alias, cols[0]), other, sub_alias, other_cols[0], other_cols[2])

    def _expression(self, alias: str, depth: int, tables: list[str]) -> tuple[Any, ...]:
        rng = self.rng
        cols = self._current_scope_columns.get(alias, ("id", "label", "qty", "amount"))
        if depth <= 0:
            return ("column", alias, rng.choice((cols[2], cols[3])))
        kind = rng.choice(("coalesce", "add", "multiply", "case", "scalar_count"))
        self.features.add("expression_" + kind)
        self.max_expression_depth = max(self.max_expression_depth, depth)
        if kind == "coalesce":
            return (
                "coalesce",
                ("column", alias, rng.choice((cols[2], cols[3]))),
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
                ("is_null", ("column", alias, cols[2])),
                ("literal", rng.randint(1, 5)),
                self._expression(alias, depth - 1, tables),
            )
        other = rng.choice(tables)
        other_cols = self._get_table_canonical_columns(other)
        return ("scalar_count", other, "sq", other_cols[0], ("column", alias, cols[0]))

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
            self._current_scope_columns["c"] = tuple(p[2] for p in cte_query.projections)
            self.features.add("cte")
            self.features.add("nested_select")
        elif not simple and depth > 0 and rng.random() < 0.14:
            derived = self._build_select(tables, foreign_keys, depth=depth - 1, simple=True)
            source = ("derived", derived, "d")
            available_alias = "d"
            self._current_scope_columns["d"] = tuple(p[2] for p in derived.projections)
            self.features.add("derived_table")
            self.features.add("nested_select")
        else:
            self._current_scope_columns[available_alias] = self._get_table_canonical_columns(table)

        joins: list[tuple[Any, ...]] = []
        if source[0] == "table" and not simple and rng.random() < 0.48:
            relation = next((fk for fk in foreign_keys if fk[0] == table), None)
            if relation is not None:
                child, child_col, parent = relation[:3]
                parent_key = relation[3] if len(relation) > 3 else self._get_table_canonical_columns(parent)[0]
                join_table = parent
                on = ("compare", ("column", alias, child_col), "=", ("column", "j", parent_key))
                joins.append((join_table, "j", on))
                self.features.add("foreign_key_join")
            else:
                child_relation = next((fk for fk in foreign_keys if fk[2] == table), None)
                if child_relation is not None:
                    child, child_col, _ = child_relation[:3]
                    table_key = self._get_table_canonical_columns(table)[0]
                    on = ("compare", ("column", "j", child_col), "=", ("column", alias, table_key))
                    joins.append((child, "j", on))
                    self.features.add("foreign_key_join")
                else:
                    other = rng.choice([name for name in tables if name != table])
                    table_key = self._get_table_canonical_columns(table)[0]
                    other_key = self._get_table_canonical_columns(other)[0]
                    on = ("compare", ("column", alias, table_key), "=", ("column", "j", other_key))
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
        cols = self._current_scope_columns.get(available_alias, ("id", "label", "qty", "amount"))
        if simple:
            # Nested sources expose a stable, schema-known projection to their parent.
            projections = tuple(
                ("column", available_alias, name)
                for name in cols
            )
        elif group:
            group_by = (("column", available_alias, cols[1]),)
            projections = (
                ("column", available_alias, cols[1]),
                ("aggregate", "COUNT", ("star",)),
                ("aggregate", "SUM", ("column", available_alias, cols[2])),
                ("aggregate", "AVG", ("column", available_alias, cols[3])),
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
                ("column", available_alias, cols[0]),
                ("column", available_alias, cols[1]),
                self._expression(available_alias, 2, tables),
            ]
            if depth > 0 and rng.random() < 0.24:
                projections_list.append(
                    ("window", "ROW_NUMBER", ("column", available_alias, cols[1]),
                     ("column", available_alias, cols[0]))
                )
                self.features.update(("window", "partition_by"))
            projections = tuple(projections_list)

        if distinct:
            self.features.add("distinct")

        set_operation = None
        if not simple and depth > 0 and not group and rng.random() < 0.18:
            # Set-operation branches have matching one-column projections.
            projections = (("column", available_alias, cols[0]),)
            right_table = rng.choice([name for name in tables if name != table])
            right_cols = self._get_table_canonical_columns(right_table)
            right = SelectAST(
                source=("table", right_table, "r"),
                projections=(("column", "r", right_cols[0]),),
                where=("compare", ("column", "r", right_cols[0]), ">=", ("literal", rng.randint(1, 3))),
            )
            set_operation = (rng.choice(("UNION", "UNION ALL", "INTERSECT", "EXCEPT")), right)
            self.features.update(("set_operation", set_operation[0].lower()))

        if set_operation is not None:
            # Compound SELECT ordering is expressed against the result column,
            # not a table-qualified name from either branch.
            order_by = (("bare_column", cols[0]),)
        elif group:
            # Keep ORDER BY compliant with ONLY_FULL_GROUP_BY / PostgreSQL grouping rules.
            order_by = () if simple or rng.random() < 0.22 else (("column", available_alias, cols[1]),)
        else:
            order_by = () if simple or rng.random() < 0.22 else (("column", available_alias, cols[0]),)
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

    def generate(
        self,
        conn: Any = None,
        *,
        tables: list[str] | None = None,
        foreign_keys: list[tuple[str, ...]] | None = None,
        columns: dict[str, Sequence[Any]] | None = None,
        dialect: Any = None,
    ) -> GeneratedQuery:
        if tables is not None:
            t = list(tables)
            fk = list(foreign_keys or [])
        elif conn is not None:
            t, fk = self._schema(conn)
        else:
            raise ValueError("query generation requires either conn or tables")

        if len(t) < 2:
            raise ValueError("query generation requires at least two generated tables")

        eff_dialect = dialect
        if eff_dialect is None and conn is not None:
            try:
                from matview_sync import detect_engine_version
                eff_dialect = detect_engine_version(conn).dialect
            except Exception:
                eff_dialect = None

        if columns is not None:
            parsed_cols: dict[str, list[ColumnInfo]] = {}
            for tbl, col_seq in columns.items():
                col_infos = []
                for idx, c in enumerate(col_seq):
                    if isinstance(c, ColumnInfo):
                        col_infos.append(c)
                    elif isinstance(c, str):
                        col_infos.append(ColumnInfo(name=c, is_pk=(idx == 0)))
                    elif isinstance(c, (tuple, list)) and len(c) >= 2:
                        col_infos.append(
                            ColumnInfo(
                                name=str(c[0]),
                                data_type=str(c[1]).upper(),
                                is_pk=bool(c[2]) if len(c) > 2 else (idx == 0),
                            )
                        )
                    else:
                        col_infos.append(ColumnInfo(name=str(c), is_pk=(idx == 0)))
                parsed_cols[tbl] = col_infos
            self._table_columns = parsed_cols
        elif conn is not None:
            self._table_columns = self._introspect_columns(conn, t, dialect=eff_dialect)
        else:
            self._table_columns = {}

        for table in t:
            if table not in self._table_columns:
                self._table_columns[table] = [
                    ColumnInfo("id", "INTEGER", is_pk=True),
                    ColumnInfo("label", "VARCHAR"),
                    ColumnInfo("qty", "INTEGER"),
                    ColumnInfo("amount", "REAL"),
                ]

        query = self._build_select(t, fk, depth=2)
        return GeneratedQuery(
            sql=_render_select(query, dialect=eff_dialect),
            features=frozenset(self.features),
            shape=self._shape(query),
            max_predicate_depth=self.max_predicate_depth,
            max_expression_depth=self.max_expression_depth,
        )


def _query_coverage_score(
    query: GeneratedQuery,
    feature_counts: dict[str, int],
    seen_shapes: set[tuple[Any, ...]],
) -> tuple[int, int, float, int]:
    """Prefer candidates that expand structural coverage, then rare features."""
    unseen_features = sum(feature_counts.get(feature, 0) == 0 for feature in query.features)
    rarity = sum(1.0 / (1 + feature_counts.get(feature, 0)) for feature in query.features)
    return (
        unseen_features,
        int(query.shape not in seen_shapes),
        rarity,
        len(query.features),
    )


class RecursiveQueryASTTests(unittest.TestCase):
    def test_coverage_score_prioritizes_uncovered_structure(self) -> None:
        already_seen = GeneratedQuery(
            sql="SELECT 1",
            features=frozenset({"where"}),
            shape=("seen",),
            max_predicate_depth=0,
            max_expression_depth=0,
        )
        new_structure = GeneratedQuery(
            sql="SELECT 1",
            features=frozenset({"join", "group_by", "having"}),
            shape=("new",),
            max_predicate_depth=1,
            max_expression_depth=0,
        )
        counts = {"where": 4}
        shapes = {("seen",)}
        self.assertGreater(
            _query_coverage_score(new_structure, counts, shapes),
            _query_coverage_score(already_seen, counts, shapes),
        )

    def test_generated_select_grammar_executes_with_coverage_guided_selection(self) -> None:
        features: set[str] = set()
        feature_counts: dict[str, int] = {}
        shapes: set[tuple[Any, ...]] = set()
        executed = 0
        candidate_evaluations = 0
        candidate_count = 8

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
                    candidates = [
                        SchemaAwareSelectFuzzer(
                            seed * 1009 + query_index + candidate_index * 100_003
                        ).generate(conn)
                        for candidate_index in range(candidate_count)
                    ]
                    candidate_evaluations += len(candidates)
                    selected_index = max(
                        range(len(candidates)),
                        key=lambda index: (
                            _query_coverage_score(candidates[index], feature_counts, shapes),
                            -index,
                        ),
                    )
                    query = candidates[selected_index]
                    rows = conn.execute(query.sql).fetchall()
                    self.assertIsInstance(rows, list)
                    features.update(query.features)
                    shapes.add(query.shape)
                    for feature in query.features:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
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
        self.assertEqual(candidate_evaluations, executed * candidate_count)
        self.assertTrue(all(count > 0 for count in feature_counts.values()))
        print(
            f"[QUERY AST COVERAGE-GUIDED] executed={executed}, "
            f"candidates={candidate_evaluations}, structural_shapes={len(shapes)}, "
            f"features={','.join(sorted(features))}"
        )

    def test_generated_select_grammar_executes_over_arbitrary_non_id_non_qty_schemas(self) -> None:
        """Verify SchemaAwareSelectFuzzer generates and executes valid SQL on schemas without id/qty."""
        schemas = [
            (
                # Schema 1: E-commerce with custom PK and column naming
                """
                CREATE TABLE cust_orders (
                    order_code INTEGER PRIMARY KEY,
                    customer_ref VARCHAR(40) NOT NULL,
                    total_price REAL NOT NULL DEFAULT 0,
                    item_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE order_items (
                    item_uid INTEGER PRIMARY KEY,
                    order_fk INTEGER REFERENCES cust_orders(order_code),
                    product_name VARCHAR(40),
                    unit_price REAL DEFAULT 0,
                    units_shipped INTEGER DEFAULT 0
                );
                """,
                [
                    "INSERT INTO cust_orders VALUES (1, 'cust_alpha', 150.50, 3);",
                    "INSERT INTO cust_orders VALUES (2, 'cust_beta', 75.00, 1);",
                    "INSERT INTO cust_orders VALUES (3, 'cust_gamma', 320.25, 5);",
                    "INSERT INTO cust_orders VALUES (4, 'cust_delta', 0.0, 0);",
                    "INSERT INTO order_items VALUES (101, 1, 'Widget A', 50.0, 2);",
                    "INSERT INTO order_items VALUES (102, 1, 'Gadget B', 50.5, 1);",
                    "INSERT INTO order_items VALUES (103, 2, 'Part C', 75.0, 1);",
                    "INSERT INTO order_items VALUES (104, 3, 'Widget A', 50.0, 4);",
                    "INSERT INTO order_items VALUES (105, 3, 'Special D', 120.25, 1);",
                    "INSERT INTO order_items VALUES (106, NULL, 'Unassigned E', 19.99, 1);",
                ],
            ),
            (
                # Schema 2: Telemetry / metrics with custom PK and column naming
                """
                CREATE TABLE telemetry_nodes (
                    node_pk INTEGER PRIMARY KEY,
                    region_tag VARCHAR(32) NOT NULL,
                    utilization_rate REAL NOT NULL,
                    core_count INTEGER NOT NULL
                );
                CREATE TABLE telemetry_events (
                    event_seq INTEGER PRIMARY KEY,
                    target_node_pk INTEGER REFERENCES telemetry_nodes(node_pk),
                    event_name VARCHAR(32),
                    metric_val REAL,
                    retry_count INTEGER
                );
                """,
                [
                    "INSERT INTO telemetry_nodes VALUES (10, 'us-east', 0.75, 16);",
                    "INSERT INTO telemetry_nodes VALUES (20, 'us-west', 0.42, 32);",
                    "INSERT INTO telemetry_nodes VALUES (30, 'eu-central', 0.88, 8);",
                    "INSERT INTO telemetry_events VALUES (1001, 10, 'heartbeat', 99.8, 0);",
                    "INSERT INTO telemetry_events VALUES (1002, 10, 'packet_drop', 0.05, 2);",
                    "INSERT INTO telemetry_events VALUES (1003, 20, 'heartbeat', 99.9, 0);",
                    "INSERT INTO telemetry_events VALUES (1004, 30, 'failover', 1.0, 5);",
                    "INSERT INTO telemetry_events VALUES (1005, NULL, 'orphan_ping', 50.0, 1);",
                ],
            ),
        ]

        total_executed = 0
        all_features: set[str] = set()
        all_shapes: set[tuple[Any, ...]] = set()

        for schema_idx, (ddl, seed_data) in enumerate(schemas):
            conn = sqlite3.connect(":memory:")
            try:
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.executescript(ddl)
                for stmt in seed_data:
                    conn.execute(stmt)
                conn.commit()

                tables, fks = SchemaAwareSelectFuzzer._schema(conn)
                self.assertGreaterEqual(len(tables), 2)
                self.assertGreaterEqual(len(fks), 1)

                for seed in range(32):
                    for q_idx in range(10):
                        fuzzer = SchemaAwareSelectFuzzer((schema_idx + 1) * 10007 + seed * 31 + q_idx)
                        query = fuzzer.generate(conn)
                        # Assert no hardcoded 'id' or 'qty' column name appeared in the query
                        self.assertIsNone(
                            re.search(r'\b(id|qty)\b', query.sql, re.IGNORECASE),
                            f"query contained unexpected hardcoded column: {query.sql}",
                        )
                        # Execute query against SQLite runtime
                        rows = conn.execute(query.sql).fetchall()
                        self.assertIsInstance(rows, list)
                        all_features.update(query.features)
                        all_shapes.add(query.shape)
                        self.assertLessEqual(query.max_predicate_depth, 4)
                        self.assertLessEqual(query.max_expression_depth, 2)
                        total_executed += 1
            finally:
                conn.close()

        required = {
            "where", "join", "group_by", "aggregate", "having", "distinct",
            "cte", "derived_table", "set_operation", "window", "partition_by",
            "nested_select", "foreign_key_join", "expression_coalesce",
            "expression_add", "expression_multiply", "expression_case",
            "expression_scalar_count", "exists", "in_subquery", "and", "or", "not",
        }
        missing = required - all_features
        self.assertFalse(missing, f"missing generated query features on non-id schemas: {sorted(missing)}")
        self.assertGreaterEqual(len(all_shapes), 40)
        self.assertEqual(total_executed, len(schemas) * 32 * 10)

    def test_standalone_generation_with_custom_columns_and_dialects(self) -> None:
        """Verify SchemaAwareSelectFuzzer generation with explicit column mapping across dialects."""
        from matview_sync import Dialect

        tables = ["custom_alpha", "custom_beta"]
        fks = [("custom_beta", "alpha_ref", "custom_alpha", "alpha_pk")]
        cols = {
            "custom_alpha": [
                ColumnInfo("alpha_pk", "INTEGER", is_pk=True),
                ColumnInfo("alpha_title", "VARCHAR(50)"),
                ColumnInfo("alpha_score", "REAL"),
                ColumnInfo("alpha_count", "INTEGER"),
            ],
            "custom_beta": [
                ColumnInfo("beta_uid", "INTEGER", is_pk=True),
                ColumnInfo("alpha_ref", "INTEGER"),
                ColumnInfo("beta_label", "VARCHAR(50)"),
                ColumnInfo("beta_val", "REAL"),
                ColumnInfo("beta_weight", "INTEGER"),
            ],
        }

        for seed in range(16):
            fuzzer = SchemaAwareSelectFuzzer(seed * 77)
            for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
                q = fuzzer.generate(tables=tables, foreign_keys=fks, columns=cols, dialect=dialect)
                self.assertTrue(q.sql)
                self.assertIsNone(
                    re.search(r'\b(id|qty)\b', q.sql, re.IGNORECASE),
                    f"generated SQL should not contain id/qty: {q.sql}",
                )
                if dialect == Dialect.MYSQL:
                    self.assertTrue(
                        "`custom_alpha`" in q.sql or "`custom_beta`" in q.sql,
                        f"MySQL identifiers must be backtick quoted: {q.sql}",
                    )
                else:
                    self.assertTrue(
                        '"custom_alpha"' in q.sql or '"custom_beta"' in q.sql,
                        f"Standard identifiers must be double quoted: {q.sql}",
                    )

    def test_differential_execution_over_mutations_on_arbitrary_schema(self) -> None:
        """Verify deterministic differential execution across data mutations on arbitrary schemas."""
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE cust_orders (
                    order_code INTEGER PRIMARY KEY,
                    customer_ref VARCHAR(40) NOT NULL,
                    total_price REAL NOT NULL DEFAULT 0,
                    item_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE order_items (
                    item_uid INTEGER PRIMARY KEY,
                    order_fk INTEGER REFERENCES cust_orders(order_code),
                    product_name VARCHAR(40),
                    unit_price REAL DEFAULT 0,
                    units_shipped INTEGER DEFAULT 0
                );
                """
            )
            conn.execute("INSERT INTO cust_orders VALUES (1, 'cust_alpha', 100.0, 2);")
            conn.execute("INSERT INTO cust_orders VALUES (2, 'cust_beta', 200.0, 4);")
            conn.execute("INSERT INTO order_items VALUES (10, 1, 'Part X', 50.0, 2);")
            conn.execute("INSERT INTO order_items VALUES (20, 2, 'Part Y', 50.0, 4);")
            conn.commit()

            for seed in range(20):
                fuzzer = SchemaAwareSelectFuzzer(seed * 101)
                generated = fuzzer.generate(conn)
                # Query must produce identical results on repeated execution (determinism)
                r1 = conn.execute(generated.sql).fetchall()
                r2 = conn.execute(generated.sql).fetchall()
                self.assertEqual(r1, r2)

            # Apply DML mutation
            conn.execute("INSERT INTO cust_orders VALUES (3, 'cust_gamma', 300.0, 6);")
            conn.execute("INSERT INTO order_items VALUES (30, 3, 'Part Z', 75.0, 4);")
            conn.commit()

            # Execute after mutation to verify execution stability
            for seed in range(20):
                fuzzer = SchemaAwareSelectFuzzer(seed * 101)
                generated = fuzzer.generate(conn)
                rows = conn.execute(generated.sql).fetchall()
                self.assertIsInstance(rows, list)
        finally:
            conn.close()


def _predicate_shrinks(node: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """Return strictly simpler, local predicate alternatives in deterministic order."""
    kind = node[0]
    candidates: list[tuple[Any, ...]] = []
    if kind == "not":
        candidates.append(node[1])
    elif kind in ("and", "or"):
        candidates.extend((node[1], node[2]))
        for index in (1, 2):
            child = node[index]
            if child[0] in ("and", "or", "not"):
                for simpler in _predicate_shrinks(child):
                    parts = list(node)
                    parts[index] = simpler
                    candidates.append(tuple(parts))
    # Stable de-duplication also prevents a malformed/custom AST from looping.
    return list(dict.fromkeys(candidate for candidate in candidates if candidate != node))


def _query_shrink_candidates(query: SelectAST) -> list[SelectAST]:
    """Generate conservative AST reductions; validity is decided by the caller."""
    candidates: list[SelectAST] = []
    if query.order_by:
        candidates.append(replace(query, order_by=()))
    if query.limit is not None:
        candidates.append(replace(query, limit=None))
    if query.distinct:
        candidates.append(replace(query, distinct=False))
    if len(query.projections) > 1:
        candidates.append(replace(query, projections=query.projections[:-1]))
    if query.where is not None:
        candidates.append(replace(query, where=None))
        candidates.extend(
            replace(query, where=simpler)
            for simpler in _predicate_shrinks(query.where)
        )
    if query.having is not None:
        candidates.append(replace(query, having=None))
        candidates.extend(
            replace(query, having=simpler)
            for simpler in _predicate_shrinks(query.having)
        )
    if query.joins:
        candidates.append(replace(query, joins=query.joins[:-1]))
    if query.set_operation is not None:
        candidates.append(replace(query, set_operation=None))
    if query.group_by:
        # Removing grouping is only accepted if the caller confirms the resulting
        # query is valid and still exhibits the same failure.
        candidates.append(replace(query, group_by=(), having=None))
    return list(dict.fromkeys(candidates))


def minimize_select_ast(
    query: SelectAST,
    still_fails: Callable[[str], bool],
) -> SelectAST:
    """Greedily minimize a failing SELECT while preserving the caller's failure predicate.

    The callback should execute the query against the same schema/data and return
    true only for the failure being minimized (ideally matching its failure class,
    not merely any exception). Invalid candidate SQL should return false.
    """
    if not still_fails(_render_select(query)):
        raise ValueError("initial SELECT does not reproduce the requested failure")

    current = query
    changed = True
    while changed:
        changed = False
        for candidate in _query_shrink_candidates(current):
            if candidate == current:
                continue
            try:
                reproduces = still_fails(_render_select(candidate))
            except Exception:
                # A shrinker must not replace the original failure with an
                # unrelated parser/renderer/connection exception.
                reproduces = False
            if reproduces:
                current = candidate
                changed = True
                break
    return current


class QueryASTShrinkingTests(unittest.TestCase):
    def test_minimizer_removes_irrelevant_select_structure_but_keeps_failure(self) -> None:
        query = SelectAST(
            source=("table", "vs0_t0", "t"),
            projections=(
                ("column", "t", "id"),
                ("column", "t", "label"),
                ("add", ("column", "t", "qty"), ("literal", 1)),
            ),
            distinct=True,
            where=(
                "and",
                ("compare", ("column", "t", "id"), "=", ("literal", 1)),
                ("is_null", ("column", "t", "qty")),
            ),
            order_by=(("column", "t", "id"),),
            limit=7,
        )

        def reproduces(sql: str) -> bool:
            # Model a stable, query-specific failure: it requires the NULL
            # predicate, but is independent of projection, sort, limit and
            # the unrelated conjunct.
            return "WHERE" in sql and '"qty" IS NULL' in sql

        minimized = minimize_select_ast(query, reproduces)
        sql = _render_select(minimized)
        self.assertTrue(reproduces(sql), sql)
        self.assertEqual(len(minimized.projections), 1)
        self.assertIsNone(minimized.limit)
        self.assertFalse(minimized.order_by)
        self.assertFalse(minimized.distinct)
        self.assertEqual(minimized.where, ("is_null", ("column", "t", "qty")))
        self.assertLess(len(sql), len(_render_select(query)))

    def test_minimizer_rejects_non_reproducing_input(self) -> None:
        query = SelectAST(
            source=("table", "vs0_t0", "t"),
            projections=(("column", "t", "id"),),
        )
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            minimize_select_ast(query, lambda sql: False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
