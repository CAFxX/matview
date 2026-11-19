#!/usr/bin/env python3
"""
matview_sync.py — Incremental Materialized View Compiler & Lifecycle Manager
============================================================================

Backwards-compatible public facade re-exporting all compiler, frontend,
IR, connectivity, manager, and CLI symbols from the modular `matview` package.
"""

from __future__ import annotations

import sqlite3
import sys

from packaging.version import InvalidVersion, Version
import sqlglot
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from matview import *  # noqa: F401, F403
from matview.cli import main
from matview.compiler import IncrementalViewCompiler
from matview.connectivity import (
    _NonClosingConnectionProxy,
    _can_use_sqlalchemy_inspector,
    _format_sa_column_type,
    connect_database_via_sqlalchemy,
    detect_engine_version,
    introspect_database_schemas,
    introspect_schemas_with_sqlalchemy,
    introspect_sqlite_schemas,
    normalize_sqlalchemy_url,
    render_parameterized_sql,
    unwrap_dbapi_connection,
)
from matview.dialects import (
    PROCEDURAL_SQL_TEMPLATES,
    _PROCEDURAL_TEMPLATE_ENV,
    render_procedural_sql,
)
from matview.engines import (
    _IDENT_TOKEN_PATTERN,
    _SIMPLE_IDENT_RE,
    _SQL_RESERVED_KEYWORDS,
    _extract_packaging_and_tuple,
    _extract_version_tuple,
    _read_mariadb_div_precision_increment,
    _sqlglot_dialect,
    _to_packaging_version,
)
from matview.logical_ir import CompilationContext, LogicalIR
from matview.manager import (
    MaterializedViewManager,
    _resolve_dialect_and_engine_version,
    generate_materialized_view,
    generate_materialized_views,
    install_materialized_view_sqlite,
)
from matview.sql_frontend import (
    _DOLLAR_QUOTE_RE,
    _FN_TO_AGG_ENUM,
    _SQL_EXPR_FUNCTIONS,
    _ast_ident_expr,
    _ast_needs_quotes,
    _ast_table_expr,
    _contains_aggregate_call,
    _extract_all_referenced_tables,
    _extract_created_object,
    _extract_cte_names_and_strip_with,
    _extract_set_op_branches_from_ast,
    _extract_sqlite_raw_declared_types,
    _extract_verbatim_cte_or_subquery_sql,
    _find_top_level_token,
    _has_subquery_ast,
    _is_simple_column_expr,
    _is_word_or_ident_token,
    _parse_from_and_joins,
    _parse_index_col_ast,
    _parse_select_item,
    _parse_single_agg_expr,
    _parse_single_create_view_stmt,
    _sanitize_ident,
    _split_top_level,
    _strip_sql_comments,
    _tokenize_sql,
    _unquote_ident,
    _unwrap_outer_parens,
)

if __name__ == "__main__":
    sys.exit(main())
