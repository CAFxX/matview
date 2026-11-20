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



