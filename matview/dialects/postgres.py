"""PostgreSQL-specific codegen helpers (CASE result type inference)."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp as sqlglot_exp

from .._host import MixinHost
from ..engines import (
    AggFunc,
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    MaintenanceStrategy,
    UnsupportedMaterializedViewError,
    _SIMPLE_IDENT_RE,
    _SQL_RESERVED_KEYWORDS,
    _sqlglot_dialect,
)
from ..logical_ir import ColumnDef, JoinSpec, SelectItem, TableSchema, ViewSpec
from ..sql_frontend import (
    _has_subquery_ast,
    _split_top_level,
    _unquote_ident,
    _unwrap_outer_parens,
    build_create_view_sql,
    build_insert_select_sql,
    build_select_sql,
    quote_sql_identifier,
    tokenize_sql,
)
from .templates import render_procedural_sql


class PostgresDialectMixin(MixinHost):
    def _postgres_case_result_storage_type(self, expr: str) -> Optional[str]:
        """Infer PostgreSQL CASE common types, including nested column branches."""
        if self.dialect != Dialect.POSTGRES:
            return None
        return self._infer_ast_expr_storage_type(expr)

    def _enforce_pg_no_pseudotype_outputs(self) -> None:
        """Refuse PostgreSQL views whose output select items evaluate to a
        pseudo-type (such as anonymous ``record`` from ``ROW(v, g)`` or
        ``(v, g)``, ``record[]`` from ``ARRAY[ROW(...)]``, or ``void``).

        PostgreSQL forbids pseudo-type columns in tables and views; silently
        falling back to a ``TEXT`` backing column would change the result type
        from ``record`` (OID 2249) to ``text`` (OID 25).
        """
        if self.dialect != Dialect.POSTGRES:
            return

        def _raise_for(alias: str, raw_expr: str, dt: str) -> None:
            raise UnsupportedMaterializedViewError(
                f"PostgreSQL materialized view column {alias!r} ({raw_expr!r}) has pseudo-type "
                f"{dt.lower()}; PostgreSQL does not allow pseudo-type columns in tables or views. "
                f"Cast the expression to a named composite type (for example CAST(... AS my_type)) "
                f"or to TEXT."
            )

        for item in self.view.select_items:
            inferred = self._infer_expr_storage_type_or_none(item)
            if self._is_pg_pseudotype(inferred):
                assert inferred is not None
                _raise_for(item.alias, item.raw_expr, inferred)

        if self.view.has_compound_set_op:
            branches = self._parse_setop_branches()
            if branches:
                for _op, br_spec in branches:
                    br_ctx = self.with_ir(
                        primary_schema=self._schemas_lower.get(br_spec.base_table.lower()),
                        view=br_spec,
                    )
                    for idx, br_item in enumerate(br_spec.select_items):
                        br_dt = br_ctx._infer_expr_storage_type_or_none(br_item)
                        if self._is_pg_pseudotype(br_dt):
                            assert br_dt is not None
                            out_alias = (
                                self.view.select_items[idx].alias
                                if idx < len(self.view.select_items)
                                else br_item.alias
                            )
                            _raise_for(out_alias, br_item.raw_expr, br_dt)



