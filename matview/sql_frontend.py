"""SQLGlot-based SQL tokenization, parsing, normalization, AST validation, and builders."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import sqlglot
import sqlglot.errors
from sqlglot import exp as sqlglot_exp
from sqlglot import TokenType as SQLGlotTokenType

from .engines import (
    AggFunc,
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    UnsupportedMaterializedViewError,
    _IDENT_TOKEN_PATTERN,
    _SIMPLE_IDENT_RE,
    _SQL_RESERVED_KEYWORDS,
    _sqlglot_dialect,
    parse_engine_version,
)
from .logical_ir import (
    ColumnDef,
    JoinSpec,
    SelectItem,
    TableIndexDef,
    TableSchema,
    ViewIndexSpec,
    ViewSpec,
)

_DOLLAR_QUOTE_RE = re.compile(r"^\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")



def _is_word_or_ident_token(tk: Any) -> bool:
    if tk.token_type == SQLGlotTokenType.IDENTIFIER:
        return True
    if tk.token_type in (
        SQLGlotTokenType.STRING,
        SQLGlotTokenType.NUMBER,
        SQLGlotTokenType.RAW_STRING,
        SQLGlotTokenType.HEX_STRING,
        SQLGlotTokenType.BYTE_STRING,
        SQLGlotTokenType.BIT_STRING,
        SQLGlotTokenType.NATIONAL_STRING,
        SQLGlotTokenType.NATIONAL_RAW_STRING,
        SQLGlotTokenType.UNICODE_STRING,
        SQLGlotTokenType.HEREDOC_STRING,
    ):
        return False
    return bool(tk.text and (tk.text[0].isalpha() or tk.text[0] == "_"))


try:
    from sqlglot.dialects.mysql import MySQL as _SQLGlotMySQL
    from sqlglot.dialects.dialect import Dialect as _SQLGlotDialectBase

    class _MySQLDialect(_SQLGlotMySQL):
        class Tokenizer(_SQLGlotMySQL.Tokenizer):
            IDENTIFIERS = ["`", '"']
            QUOTES = ["'"]

    _SQLGlotDialectBase.classes["mysql"] = _MySQLDialect
    _SQLGlotDialectBase.classes["mariadb"] = _MySQLDialect
except Exception:
    pass


def _tokenize_sql(
    sql: str,
    read: Union[str, Dialect, EngineSubdialect, EngineVersion, None] = "sqlite",
) -> List[Any]:
    """Tokenize SQL while preserving dollar-quoted strings and dialect quoting rules.

    SQLGlot's PostgreSQL lexer understands dollar-quoted strings but treats
    backtick-quoted identifiers differently from SQLite/MySQL. Tokenize ordinary
    SQL segments with the requested dialect and represent each dollar-quoted
    literal as one synthetic token, preserving source offsets for the scanner.
    """
    sg_read = _sqlglot_dialect(read)

    class _TokenSpan:
        __slots__ = ("token_type", "text", "start", "end")

        def __init__(self, token_type: Any, text: str, start: int, end: int) -> None:
            self.token_type = token_type
            self.text = text
            self.start = start
            self.end = end

    def _dollar_spans() -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        i = 0
        n = len(sql)
        state = ""
        block_depth = 0
        while i < n:
            ch = sql[i]
            if state == "single":
                if sg_read == "postgres" and ch == "\\":
                    i += 2
                    continue
                if ch == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    state = ""
                i += 1
                continue
            if state == "double":
                if ch == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        i += 2
                        continue
                    state = ""
                i += 1
                continue
            if state == "backtick":
                if ch == chr(96):
                    if i + 1 < n and sql[i + 1] == chr(96):
                        i += 2
                        continue
                    state = ""
                i += 1
                continue
            if state == "bracket":
                if ch == "]":
                    state = ""
                i += 1
                continue
            if state == "line_comment":
                if ch == "\n":
                    state = ""
                i += 1
                continue
            if state == "block_comment":
                if sql.startswith("/*", i):
                    block_depth += 1
                    i += 2
                    continue
                if sql.startswith("*/", i):
                    block_depth -= 1
                    i += 2
                    if block_depth == 0:
                        state = ""
                    continue
                i += 1
                continue

            if sql.startswith("--", i):
                state = "line_comment"
                i += 2
                continue
            if sql.startswith("/*", i):
                state = "block_comment"
                block_depth = 1
                i += 2
                continue
            if ch == "'":
                state = "single"
                i += 1
                continue
            if ch == '"':
                state = "double"
                i += 1
                continue
            if ch == chr(96):
                state = "backtick"
                i += 1
                continue
            if ch == "[":
                state = "bracket"
                i += 1
                continue
            if ch == "$":
                delimiter_match = re.match(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$", sql[i:])
                if delimiter_match:
                    delimiter = delimiter_match.group(0)
                    content_start = i + len(delimiter)
                    close_at = sql.find(delimiter, content_start)
                    if close_at >= 0:
                        end_at = close_at + len(delimiter)
                        spans.append((i, end_at))
                        i = end_at
                        continue
            i += 1
        return spans

    spans = _dollar_spans()
    if not spans:
        return sqlglot.tokenize(sql, read=sg_read)

    tokens: List[Any] = []

    def _append_segment(start: int, end: int) -> None:
        if start >= end:
            return
        for token in sqlglot.tokenize(sql[start:end], read=sg_read):
            tokens.append(
                _TokenSpan(
                    token.token_type,
                    token.text,
                    token.start + start,
                    token.end + start,
                )
            )

    cursor = 0
    for start, end in spans:
        _append_segment(cursor, start)
        tokens.append(
            _TokenSpan(
                SQLGlotTokenType.HEREDOC_STRING,
                sql[start:end],
                start,
                end - 1,
            )
        )
        cursor = end
    _append_segment(cursor, len(sql))
    return tokens

class SQLTokenType(str, Enum):
    WHITESPACE = "WHITESPACE"
    STRING = "STRING"
    DOLLAR_STRING = "DOLLAR_STRING"
    QUOTED_IDENT = "QUOTED_IDENT"
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    OP = "OP"
    PUNCT = "PUNCT"


@dataclass(frozen=True)
class SQLToken:
    kind: SQLTokenType
    text: str
    start: int
    end: int

    @property
    def unquoted(self) -> str:
        if self.kind == SQLTokenType.QUOTED_IDENT:
            return _unquote_ident(self.text)
        return self.text


@dataclass(frozen=True)
class SQLSelectAST:
    raw_sql: str
    cte_names: Tuple[str, ...]
    main_select_sql: str
    has_cte: bool
    has_distinct: bool
    select_items: Tuple[SelectItem, ...]
    from_clause: Optional[str]
    joins: Tuple[JoinSpec, ...]
    where_clause: Optional[str]
    group_by_exprs: Tuple[str, ...]
    having_clause: Optional[str]
    has_window_func: bool
    has_compound_set_op: bool
    has_limit_or_order: bool
    has_subquery_in_from: bool
    sqlglot_ast: Optional[sqlglot_exp.Expr] = None


def parse_sql_ast(sql: str, dialect: Dialect = Dialect.SQLITE) -> sqlglot_exp.Expr:
    """Parses a SQL expression or statement into a `sqlglot.exp.Expr` AST."""
    return sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))


def tokenize_sql(
    sql: str,
    dialect: Dialect = Dialect.SQLITE,
) -> List[SQLToken]:
    """
    Tokenizes a SQL fragment into lossless typed `SQLToken` nodes backed by `sqlglot.tokenize`.
    """
    if not sql:
        return []
    sg_tokens = _tokenize_sql(sql, read=_sqlglot_dialect(dialect))
    tokens: List[SQLToken] = []
    cursor = 0
    n = len(sql)

    for sg_tok in sg_tokens:
        t_start = sg_tok.start
        t_end = sg_tok.end + 1
        if t_start > cursor:
            tokens.append(SQLToken(SQLTokenType.WHITESPACE, sql[cursor:t_start], cursor, t_start))
        raw_slice = sql[t_start:t_end]
        tt = sg_tok.token_type
        if tt in (
            SQLGlotTokenType.STRING,
            SQLGlotTokenType.RAW_STRING,
            SQLGlotTokenType.NATIONAL_STRING,
            SQLGlotTokenType.NATIONAL_RAW_STRING,
            SQLGlotTokenType.BYTE_STRING,
            SQLGlotTokenType.UNICODE_STRING,
        ):
            tokens.append(SQLToken(SQLTokenType.STRING, raw_slice, t_start, t_end))
        elif tt == SQLGlotTokenType.HEREDOC_STRING or raw_slice.startswith("$"):
            tokens.append(SQLToken(SQLTokenType.DOLLAR_STRING, raw_slice, t_start, t_end))
        elif tt == SQLGlotTokenType.IDENTIFIER or raw_slice[:1] in ('"', "`", "["):
            tokens.append(SQLToken(SQLTokenType.QUOTED_IDENT, raw_slice, t_start, t_end))
        elif tt in (SQLGlotTokenType.NUMBER, SQLGlotTokenType.BIT_STRING, SQLGlotTokenType.HEX_STRING):
            tokens.append(SQLToken(SQLTokenType.NUMBER, raw_slice, t_start, t_end))
        elif raw_slice in ("(", ")", ",", ".", ";"):
            tokens.append(SQLToken(SQLTokenType.PUNCT, raw_slice, t_start, t_end))
        elif raw_slice[:1].isalpha() or raw_slice[:1] == "_":
            # Split multi-word sqlglot keywords (e.g. "GROUP BY", "ORDER BY", "PRIMARY KEY")
            if any(ch.isspace() for ch in raw_slice):
                for m_part in re.finditer(r"\S+|\s+", raw_slice):
                    p_str = m_part.group(0)
                    p_s = t_start + m_part.start()
                    p_e = t_start + m_part.end()
                    if p_str.isspace():
                        tokens.append(SQLToken(SQLTokenType.WHITESPACE, p_str, p_s, p_e))
                    else:
                        tokens.append(SQLToken(SQLTokenType.IDENT, p_str, p_s, p_e))
            else:
                tokens.append(SQLToken(SQLTokenType.IDENT, raw_slice, t_start, t_end))
        else:
            tokens.append(SQLToken(SQLTokenType.OP, raw_slice, t_start, t_end))
        cursor = t_end

    if cursor < n:
        tokens.append(SQLToken(SQLTokenType.WHITESPACE, sql[cursor:n], cursor, n))
    return tokens


def _strip_outer_ident_quotes(s: str) -> str:
    """Strips a single pair of outer SQL identifier quotes without splitting on dots."""
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1].replace('""', '"')
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        return s[1:-1].replace("``", "`")
    if s.startswith("[") and s.endswith("]") and len(s) >= 2:
        return s[1:-1]
    return s


def quote_sql_identifier(
    ident: str,
    dialect: Dialect = Dialect.SQLITE,
    force: bool = False,
    allow_qualified: bool = True,
) -> str:
    """
    Quotes a SQL identifier via `sqlglot.exp.to_identifier` using dialect-appropriate
    delimiters (`"ident"` on SQLite/PostgreSQL, `` `ident` `` on MySQL/MariaDB) when
    `force=True`, when the identifier is a SQL reserved keyword, or when it contains
    spaces or special characters.
    """
    s = ident.strip()
    if not s:
        return s
    raw = _unquote_ident(s, dialect=dialect) if allow_qualified else _strip_outer_ident_quotes(s)
    needs_q = (
        force
        or raw.upper() in _SQL_RESERVED_KEYWORDS
        or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw)
    )
    if not needs_q:
        return raw
    sg_id = sqlglot_exp.to_identifier(raw, quoted=True)
    return sg_id.sql(dialect=_sqlglot_dialect(dialect))


def format_sql(
    sql: str,
    dialect: Dialect = Dialect.SQLITE,
    pretty: bool = False,
) -> str:
    """
    Formats and normalizes a SQL statement or script using `sqlglot.transpile`.
    """
    sg_dialect = _sqlglot_dialect(dialect)
    try:
        transpiled = sqlglot.transpile(sql, read=sg_dialect, write=sg_dialect, pretty=pretty)
        if transpiled:
            sep = ";\n" if pretty else "; "
            return sep.join(t.rstrip(";") for t in transpiled) + (";" if sql.strip().endswith(";") else "")
    except sqlglot.errors.SqlglotError:
        pass
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in sql.strip().splitlines()]
    return "\n".join(ln for ln in lines if ln or lines)


def _ast_needs_quotes(raw_ident: str, original_ident: str) -> bool:
    """Returns True if an identifier was already quoted or requires dialect quoting."""
    orig = original_ident.strip()
    if (orig.startswith('"') and orig.endswith('"')) or (orig.startswith("`") and orig.endswith("`")):
        return True
    return raw_ident.upper() in _SQL_RESERVED_KEYWORDS or not re.match(
        r"^[A-Za-z_][A-Za-z0-9_]*$", raw_ident
    )


def _ast_ident_expr(
    ident: str,
    dialect: Dialect = Dialect.SQLITE,
    *,
    allow_qualified: bool = True,
) -> sqlglot_exp.Identifier:
    """Builds a `sqlglot.exp.Identifier` preserving dialect quoting requirements."""
    s = ident.strip()
    raw = _unquote_ident(s, dialect=dialect) if allow_qualified else _strip_outer_ident_quotes(s)
    return sqlglot_exp.to_identifier(raw, quoted=_ast_needs_quotes(raw, s))


def _ast_table_expr(
    name: str,
    dialect: Dialect = Dialect.SQLITE,
    *,
    allow_qualified: bool = True,
) -> sqlglot_exp.Table:
    """Builds a `sqlglot.exp.Table`, handling optional `schema.table` qualification."""
    s = name.strip()
    if allow_qualified and "." in s:
        try:
            parts = _split_top_level(s, ".", dialect=dialect)
        except sqlglot.errors.SqlglotError:
            parts = [s]
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            db_ident = _ast_ident_expr(parts[0], dialect)
            tbl_ident = _ast_ident_expr(parts[1], dialect)
            return sqlglot_exp.Table(this=tbl_ident, db=db_ident)
    return sqlglot_exp.Table(this=_ast_ident_expr(s, dialect, allow_qualified=allow_qualified))


def build_drop_sql(
    kind: str,
    names: Union[str, Sequence[str]],
    dialect: Dialect = Dialect.SQLITE,
    if_exists: bool = True,
    cascade: bool = False,
    on_table: Optional[str] = None,
    allow_qualified: bool = True,
) -> str:
    """Constructs a dialect-appropriate DROP statement via `sqlglot.exp.Drop`."""
    name_seq = [names] if isinstance(names, str) else list(names)
    tables = [_ast_table_expr(n, dialect, allow_qualified=allow_qualified) for n in name_seq]
    cluster = (
        sqlglot_exp.OnProperty(this=_ast_table_expr(on_table, dialect))
        if on_table
        else None
    )
    node = sqlglot_exp.Drop(
        tables=tables,
        kind=kind.strip().upper(),
        exists=if_exists,
        cascade=cascade,
        cluster=cluster,
    )
    return node.sql(dialect=_sqlglot_dialect(dialect)) + ";"


def build_alter_rename_table_sql(
    old_table: str,
    new_table: str,
    dialect: Dialect = Dialect.SQLITE,
) -> str:
    """Constructs an `ALTER TABLE ... RENAME ...` statement via `sqlglot.exp.Alter`."""
    node = sqlglot_exp.Alter(
        this=_ast_table_expr(old_table, dialect),
        kind="TABLE",
        actions=[sqlglot_exp.AlterRename(this=_ast_table_expr(new_table, dialect))],
    )
    return node.sql(dialect=_sqlglot_dialect(dialect)) + ";"


def build_select_sql(
    select_exprs: Sequence[str],
    from_table: str,
    where_sql: Optional[str] = None,
    group_by_exprs: Sequence[str] = (),
    having_sql: Optional[str] = None,
    order_by_exprs: Sequence[str] = (),
    limit: Optional[int] = None,
    dialect: Dialect = Dialect.SQLITE,
) -> str:
    """Builds a SELECT statement via `sqlglot.exp.Select` and serializes for `dialect`."""
    sg_dialect = _sqlglot_dialect(dialect)
    proj_nodes = [parse_sql_ast(expr, dialect=dialect) for expr in select_exprs]
    query = sqlglot_exp.select(*proj_nodes).from_(_ast_table_expr(from_table, dialect))
    if where_sql:
        query = query.where(parse_sql_ast(where_sql, dialect=dialect))
    if group_by_exprs:
        query = query.group_by(*(parse_sql_ast(g, dialect=dialect) for g in group_by_exprs))
    if having_sql:
        query = query.having(parse_sql_ast(having_sql, dialect=dialect))
    if order_by_exprs:
        order_nodes = [
            sqlglot.parse_one(f"SELECT 1 ORDER BY {o}", read=sg_dialect).args["order"].expressions[0]
            for o in order_by_exprs
        ]
        query = query.order_by(*order_nodes)
    if limit is not None:
        query = query.limit(int(limit))
    return query.sql(dialect=sg_dialect) + ";"


def build_insert_select_sql(
    target_table: str,
    columns: Sequence[str],
    select_sql_or_ast: Union[str, sqlglot_exp.Query],
    dialect: Dialect = Dialect.SQLITE,
) -> str:
    """Builds an `INSERT INTO tbl (cols) SELECT ...` statement via `sqlglot.exp.Insert`."""
    select_ast = (
        parse_sql_ast(select_sql_or_ast, dialect=dialect)
        if isinstance(select_sql_or_ast, str)
        else select_sql_or_ast
    )
    target = (
        sqlglot_exp.Schema(
            this=_ast_table_expr(target_table, dialect),
            expressions=[_ast_ident_expr(c, dialect) for c in columns],
        )
        if columns
        else _ast_table_expr(target_table, dialect)
    )
    node = sqlglot_exp.Insert(this=target, expression=select_ast)
    return node.sql(dialect=_sqlglot_dialect(dialect)) + ";"


def build_create_view_sql(
    view_name: str,
    from_table: str,
    select_exprs: Sequence[str],
    where_sql: Optional[str] = None,
    dialect: Dialect = Dialect.SQLITE,
    replace: bool = False,
    algorithm: Optional[str] = None,
    allow_qualified_view: bool = True,
) -> str:
    """Builds a `CREATE [OR REPLACE] [ALGORITHM=...] VIEW` statement via `sqlglot.exp.Create`."""
    sg_dialect = _sqlglot_dialect(dialect)
    proj_nodes = [parse_sql_ast(expr, dialect=dialect) for expr in select_exprs]
    query = sqlglot_exp.select(*proj_nodes).from_(_ast_table_expr(from_table, dialect))
    if where_sql:
        query = query.where(parse_sql_ast(where_sql, dialect=dialect))
    props = None
    if algorithm:
        props = sqlglot_exp.Properties(
            expressions=[sqlglot_exp.AlgorithmProperty(this=sqlglot_exp.Var(this=algorithm.strip().upper()))]
        )
    node = sqlglot_exp.Create(
        this=_ast_table_expr(view_name, dialect, allow_qualified=allow_qualified_view),
        kind="VIEW",
        replace=replace,
        expression=query,
        properties=props,
    )
    sql = node.sql(dialect=sg_dialect)
    if algorithm:
        sql = re.sub(r"\bALGORITHM\s*=\s*", "ALGORITHM = ", sql, count=1, flags=re.IGNORECASE)
    return sql + ";"


def _parse_index_col_ast(col_spec: str, dialect: Dialect) -> sqlglot_exp.Expr:
    """Parses an index key specification into a sqlglot AST expression."""
    cleaned = col_spec.strip()
    if re.match(r"^(?:\"[^\"]+\"|`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)$", cleaned):
        return sqlglot_exp.Column(this=_ast_ident_expr(cleaned, dialect))
    # MySQL functional index keys wrap expressions in outer `((expr))`; strip one layer
    # before feeding sqlglot so serialization does not emit redundant triple parens.
    probe_spec = cleaned
    if dialect == Dialect.MYSQL and cleaned.startswith("(("):
        m_tail = re.search(r"\)\s*(?:ASC|DESC)?\s*$", cleaned, flags=re.IGNORECASE)
        if m_tail and ")" in cleaned:
            close_paren = cleaned.rfind(")")
            tail = cleaned[close_paren + 1 :]
            inner = cleaned[1:close_paren]
            if inner.startswith("(") and inner.endswith(")"):
                probe_spec = f"{inner}{tail}"
    sg_dialect = _sqlglot_dialect(dialect)
    probe_ast = sqlglot.parse_one(f"CREATE INDEX __mv_probe_idx ON __mv_probe_tbl ({probe_spec})", read=sg_dialect)
    if isinstance(probe_ast, sqlglot_exp.Create) and isinstance(probe_ast.this, sqlglot_exp.Index):
        params = probe_ast.this.args.get("params")
        if params and params.args.get("columns"):
            return params.args["columns"][0]
    return parse_sql_ast(cleaned, dialect=dialect)


def build_create_index_sql(
    index_name: str,
    table_name: str,
    col_specs: Sequence[str],
    dialect: Dialect = Dialect.SQLITE,
    unique: bool = False,
    using_method: Optional[str] = None,
    where_clause: Optional[str] = None,
    nulls_not_distinct: bool = False,
) -> str:
    """Builds a `CREATE [UNIQUE] INDEX` statement via `sqlglot.exp.Create` and `sqlglot.exp.Index`."""
    sg_dialect = _sqlglot_dialect(dialect)
    cols = [_parse_index_col_ast(c, dialect) for c in col_specs]
    where_node = None
    if where_clause:
        where_node = sqlglot_exp.Where(this=parse_sql_ast(where_clause, dialect=dialect))
    params = sqlglot_exp.IndexParameters(
        using=sqlglot_exp.Var(this=using_method.strip()) if using_method else None,
        columns=cols,
        where=where_node,
    )
    tbl_sql = _ast_table_expr(table_name, dialect).sql(dialect=sg_dialect)
    node = sqlglot_exp.Create(
        this=sqlglot_exp.Index(
            this=_ast_ident_expr(index_name, dialect),
            table=_ast_table_expr(table_name, dialect),
            params=params,
        ),
        kind="INDEX",
        unique=unique,
    )
    sql = node.sql(dialect=sg_dialect)
    # Standardize `ON <table>(` emitted by sqlglot to `ON <table> (` for readability and planner regexes.
    sql = sql.replace(f"ON {tbl_sql}(", f"ON {tbl_sql} (", 1)
    if nulls_not_distinct:
        sql = f"{sql} NULLS NOT DISTINCT"
    return sql + ";"


def _strip_sql_comments(
    sql: str,
    dialect: Dialect = Dialect.SQLITE,
) -> str:
    """Strips `-- ...` line comments and `/* ... */` block comments via `sqlglot.tokenize`."""
    if not sql:
        return ""
    toks = _tokenize_sql(sql, read=_sqlglot_dialect(dialect))
    if not toks:
        return ""
    out: List[str] = []
    cursor = 0
    for tok in toks:
        if tok.start > cursor:
            gap = sql[cursor:tok.start]
            if "--" in gap or "/*" in gap:
                out.append(" ")
            else:
                out.append(gap)
        out.append(sql[tok.start : tok.end + 1])
        cursor = tok.end + 1
    return "".join(out).strip()


def _unquote_ident(
    ident: str,
    dialect: Dialect = Dialect.SQLITE,
) -> str:
    """Unquotes a SQL identifier or extracts the last component of a qualified name via `sqlglot.tokenize`."""
    s = ident.strip()
    if not s:
        return s
    if "." in s and not (
        (s.startswith('"') and s.endswith('"') and '".' not in s)
        or (s.startswith("`") and s.endswith("`") and "`." not in s)
        or (s.startswith("[") and s.endswith("]") and "]." not in s)
    ):
        try:
            parts = _split_top_level(s, ".", dialect=dialect)
            if len(parts) > 1:
                s = parts[-1].strip()
        except sqlglot.errors.SqlglotError:
            pass
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        return s[1:-1].replace('""', '"')
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        return s[1:-1].replace("``", "`")
    if s.startswith("[") and s.endswith("]") and len(s) >= 2:
        return s[1:-1]
    return s


def _sanitize_ident(raw: str) -> str:
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in raw.strip())
    return cleaned.strip("_") or "col"


def _split_top_level(
    text: str,
    sep: str = ",",
    dialect: Dialect = Dialect.SQLITE,
) -> List[str]:
    """Splits a SQL fragment at top-level separators outside parentheses and literals via `sqlglot.tokenize`."""
    if not text:
        return []
    toks = _tokenize_sql(text, read=_sqlglot_dialect(dialect))
    if not toks:
        return [text.strip()] if text.strip() else []
    parts: List[str] = []
    depth = 0
    bracket_depth = 0
    start = 0
    for tk in toks:
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
            depth -= 1
        elif tk.token_type == SQLGlotTokenType.L_BRACKET:
            bracket_depth += 1
        elif tk.token_type == SQLGlotTokenType.R_BRACKET and bracket_depth > 0:
            bracket_depth -= 1
        elif depth == 0 and bracket_depth == 0:
            is_match = False
            if sep == "," and tk.token_type == SQLGlotTokenType.COMMA:
                is_match = True
            elif sep == ";" and tk.token_type == SQLGlotTokenType.SEMICOLON:
                is_match = True
            elif sep == "." and tk.token_type == SQLGlotTokenType.DOT:
                is_match = True
            elif tk.text == sep:
                is_match = True
            if is_match:
                part = text[start:tk.start].strip()
                if part or sep == ";":
                    parts.append(part)
                start = tk.end + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _find_top_level_token(
    sql: str,
    target_types: Sequence[SQLGlotTokenType],
    dialect: Dialect = Dialect.SQLITE,
) -> Optional[Any]:
    """Finds the first top-level (depth == 0) `sqlglot` token matching `target_types`."""
    toks = _tokenize_sql(sql, read=_sqlglot_dialect(dialect))
    depth = 0
    bracket_depth = 0
    target_set = set(target_types)
    for tk in toks:
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
            depth -= 1
        elif tk.token_type == SQLGlotTokenType.L_BRACKET:
            bracket_depth += 1
        elif tk.token_type == SQLGlotTokenType.R_BRACKET and bracket_depth > 0:
            bracket_depth -= 1
        elif depth == 0 and bracket_depth == 0 and tk.token_type in target_set:
            return tk
    return None


def _extract_set_op_branches_from_ast(
    query_ast: sqlglot_exp.Expr,
    raw_sql: Optional[str] = None,
    dialect: Dialect = Dialect.SQLITE,
) -> Tuple[List[str], List[str]]:
    """
    Traverses a `sqlglot.exp.SetOperation` AST (`Union`, `Intersect`, `Except`) in order,
    returning `(operators, branch_sqls)`. When `raw_sql` is provided, slices the verbatim
    branch substrings using `sqlglot.tokenize` so original function names and syntax are preserved.
    """
    sg_d = _sqlglot_dialect(dialect)
    if raw_sql:
        s = raw_sql.strip()
        toks = _tokenize_sql(s, read=sg_d)
        if toks:
            ops_t: List[str] = []
            branches_t: List[str] = []
            depth = 0
            last_start = 0
            i = 0
            n = len(toks)
            while i < n:
                tk = toks[i]
                if tk.token_type == SQLGlotTokenType.L_PAREN:
                    depth += 1
                elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
                    depth -= 1
                elif depth == 0:
                    w = tk.text.upper()
                    if w in ("UNION", "INTERSECT", "EXCEPT"):
                        branches_t.append(s[last_start : tk.start].strip())
                        if w == "UNION" and i + 1 < n and toks[i + 1].token_type == SQLGlotTokenType.ALL:
                            ops_t.append("UNION ALL")
                            last_start = toks[i + 1].end + 1
                            i += 2
                            continue
                        ops_t.append(w)
                        last_start = tk.end + 1
                i += 1
            if ops_t:
                branches_t.append(s[last_start:].strip())
                return ops_t, branches_t

    ops: List[str] = []
    branches: List[str] = []

    def _walk(node: sqlglot_exp.Expr) -> None:
        if isinstance(node, sqlglot_exp.SetOperation):
            _walk(node.this)
            if isinstance(node, sqlglot_exp.Union):
                is_dist = node.args.get("distinct", True)
                ops.append("UNION" if is_dist else "UNION ALL")
            elif isinstance(node, sqlglot_exp.Intersect):
                ops.append("INTERSECT")
            elif isinstance(node, sqlglot_exp.Except):
                ops.append("EXCEPT")
            else:
                ops.append(node.key.upper())
            _walk(node.expression)
        else:
            branches.append(node.sql(sg_d))

    _walk(query_ast)
    return ops, branches


def _has_subquery_ast(
    expr_or_sql: Optional[str | sqlglot_exp.Expr],
    dialect: Dialect = Dialect.SQLITE,
) -> bool:
    """Returns True if the SQL expression or AST contains a `SELECT` or `EXISTS` subquery."""
    if not expr_or_sql:
        return False
    node = parse_sql_ast(expr_or_sql, dialect=dialect) if isinstance(expr_or_sql, str) else expr_or_sql
    return bool(node.find(sqlglot_exp.Subquery, sqlglot_exp.Select, sqlglot_exp.Exists))


def _is_simple_column_expr(
    expr_sql: str,
    dialect: Dialect = Dialect.SQLITE,
) -> bool:
    """Returns True if `expr_sql` parses as a single `sqlglot.exp.Column` (`col` or `tbl.col`)."""
    s = expr_sql.strip()
    if not s:
        return False
    try:
        node = parse_sql_ast(s, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return False
    return isinstance(node, sqlglot_exp.Column) and not isinstance(node.this, sqlglot_exp.Star)


def _sqlite_type_affinity(declared_type: str) -> str:
    """SQLite column affinity of a declared type (datatype3.html section 3.1).

    The rules apply in order: "INT" -> INTEGER; "CHAR", "CLOB" or "TEXT" ->
    TEXT; "BLOB", "ANY" (STRICT table), or no type -> BLOB; "REAL", "FLOA" or
    "DOUB" -> REAL; otherwise NUMERIC. For example "POINT REAL" has INTEGER
    affinity ("INT" in "POINT"), and "FLOATING POINT" has INTEGER affinity too.
    """
    t = (declared_type or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or not t.strip() or t.strip() == "ANY":
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


def _extract_sqlite_raw_types_and_sanitized_ddl(stmt_sql: str) -> Tuple[Dict[str, str], str]:
    """Extract raw SQLite declared types per column and a sqlglot-parseable DDL.

    SQLite accepts arbitrary multi-word column type names (e.g. ``POINT REAL``,
    ``FLOATING POINT``, ``UNSIGNED BIG INT``, ``ANY``) that sqlglot's parser
    rejects. Returns ``(raw_types_by_col_lower, sanitized_ddl)`` where
    ``sanitized_ddl`` replaces each column's declared type span with ``BLOB``
    so sqlglot can parse constraints/PKs/FKs while callers restore the raw type.
    """
    toks = _tokenize_sql(stmt_sql, read="sqlite")
    depth = 0
    lp_idx: Optional[int] = None
    rp_idx: Optional[int] = None
    for idx, tk in enumerate(toks):
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            if depth == 0:
                lp_idx = idx
            depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
            depth -= 1
            if depth == 0:
                rp_idx = idx
                break
    if lp_idx is None or rp_idx is None or rp_idx <= lp_idx + 1:
        return {}, stmt_sql
    constraint_starts = {
        "PRIMARY", "NOT", "NULL", "UNIQUE", "CHECK", "DEFAULT",
        "COLLATE", "REFERENCES", "GENERATED", "AS", "CONSTRAINT",
    }
    table_constraint_starts = {"CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN"}
    raw_types: Dict[str, str] = {}
    type_spans: List[Tuple[int, int]] = []
    items: List[List[Any]] = []
    cur: List[Any] = []
    inner_depth = 0
    for tk in toks[lp_idx + 1 : rp_idx]:
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            inner_depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and inner_depth > 0:
            inner_depth -= 1
        elif inner_depth == 0 and tk.token_type == SQLGlotTokenType.COMMA:
            if cur:
                items.append(cur)
                cur = []
            continue
        cur.append(tk)
    if cur:
        items.append(cur)
    for item_toks in items:
        if not item_toks:
            continue
        first_tok = item_toks[0]
        first_up = first_tok.text.upper()
        if first_tok.token_type != SQLGlotTokenType.IDENTIFIER and first_up in table_constraint_starts:
            continue
        col_low = _unquote_ident(first_tok.text).lower()
        t_toks: List[Any] = []
        idx_t = 1
        n_item = len(item_toks)
        while idx_t < n_item:
            tk = item_toks[idx_t]
            if tk.token_type == SQLGlotTokenType.L_PAREN:
                # Include optional type parameter list like DECIMAL(12, 4) or VARCHAR(32)
                if not t_toks:
                    break
                p_depth = 0
                while idx_t < n_item:
                    ptk = item_toks[idx_t]
                    t_toks.append(ptk)
                    if ptk.token_type == SQLGlotTokenType.L_PAREN:
                        p_depth += 1
                    elif ptk.token_type == SQLGlotTokenType.R_PAREN:
                        p_depth -= 1
                        if p_depth == 0:
                            idx_t += 1
                            break
                    idx_t += 1
                break
            if tk.token_type != SQLGlotTokenType.IDENTIFIER and tk.text.upper().split()[0] in constraint_starts:
                break
            t_toks.append(tk)
            idx_t += 1
        if t_toks:
            span_start = t_toks[0].start
            span_end = t_toks[-1].end + 1
            raw_types[col_low] = re.sub(r"\s+", " ", stmt_sql[span_start:span_end].strip())
            type_spans.append((span_start, span_end))
        else:
            raw_types[col_low] = ""

    if not type_spans:
        return raw_types, stmt_sql
    out_parts: List[str] = []
    cursor = 0
    for s_start, s_end in sorted(type_spans):
        if s_start >= cursor:
            out_parts.append(stmt_sql[cursor:s_start])
            out_parts.append("BLOB")
            cursor = s_end
    out_parts.append(stmt_sql[cursor:])
    return raw_types, "".join(out_parts)


def _extract_sqlite_raw_declared_types(stmt_sql: str) -> Dict[str, str]:
    """Extract raw SQLite declared types per column from a CREATE TABLE statement."""
    return _extract_sqlite_raw_types_and_sanitized_ddl(stmt_sql)[0]


def parse_table_index_def(
    index_sql: str,
    dialect: Dialect = Dialect.SQLITE,
) -> Optional[TableIndexDef]:
    """Parse a single `CREATE [UNIQUE] INDEX` SQL statement into a `TableIndexDef`."""
    eff_dialect = dialect if dialect is not None else Dialect.SQLITE
    sg_d = _sqlglot_dialect(eff_dialect)
    try:
        stmt_ast = parse_sql_ast(index_sql, dialect=eff_dialect)
    except sqlglot.errors.SqlglotError:
        return None
    if not isinstance(stmt_ast, sqlglot_exp.Create) or (stmt_ast.kind or "").upper() != "INDEX":
        return None
    idx_node = stmt_ast.this
    if not isinstance(idx_node, sqlglot_exp.Index):
        return None
    tbl_expr = idx_node.args.get("table")
    idx_tbl = tbl_expr.name if isinstance(tbl_expr, sqlglot_exp.Table) else ""
    idx_name = idx_node.name or ""
    params = idx_node.args.get("params")
    col_exprs = (
        list(params.args.get("columns") or params.expressions or ())
        if params is not None
        else list(idx_node.expressions or ())
    )
    cols_out: List[str] = []
    for c in col_exprs:
        if isinstance(c, sqlglot_exp.Ordered):
            inner = c.this
            if isinstance(inner, (sqlglot_exp.Identifier, sqlglot_exp.Column)) and inner.name:
                cols_out.append(inner.name)
            elif inner is not None:
                cols_out.append(inner.sql(dialect=sg_d))
        elif isinstance(c, (sqlglot_exp.Identifier, sqlglot_exp.Column)) and c.name:
            cols_out.append(c.name)
        else:
            cols_out.append(c.sql(dialect=sg_d))
    where_n = params.args.get("where") if params is not None else None
    where_s = (
        where_n.this.sql(dialect=sg_d)
        if isinstance(where_n, sqlglot_exp.Where) and where_n.this is not None
        else None
    )
    if not idx_tbl or not cols_out:
        return None
    return TableIndexDef(
        name=idx_name or f"idx_{idx_tbl}_{'_'.join(cols_out)}",
        table_name=idx_tbl,
        columns=tuple(cols_out),
        unique=bool(stmt_ast.args.get("unique")),
        where_clause=where_s,
    )


def _find_pg_balanced_paren_span(sql: str, start_idx: int) -> Tuple[int, int]:
    """Return ``(open_idx, close_idx)`` of the first balanced ``(...)`` block
    starting at or after ``start_idx``, skipping single/double-quoted strings
    (including ``E'...'`` escapes) and ``$tag$...$tag$`` dollar-quoted strings."""
    n = len(sql)
    i = start_idx
    open_idx = -1
    depth = 0
    in_sq = False
    sq_escape = False
    in_dq = False
    dollar_tag: Optional[str] = None
    while i < n:
        ch = sql[i]
        if dollar_tag is not None:
            if sql.startswith(dollar_tag, i):
                i += len(dollar_tag)
                dollar_tag = None
                continue
            i += 1
            continue
        if in_sq:
            if sq_escape and ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                if i + 1 < n and sql[i + 1] == "'":
                    i += 2
                    continue
                in_sq = False
                sq_escape = False
            i += 1
            continue
        if in_dq:
            if ch == '"':
                if i + 1 < n and sql[i + 1] == '"':
                    i += 2
                    continue
                in_dq = False
            i += 1
            continue
        if ch == "$":
            m_tag = re.match(r"\$[A-Za-z_0-9]*\$", sql[i:])
            if m_tag:
                dollar_tag = m_tag.group(0)
                i += len(dollar_tag)
                continue
        if ch == "'":
            in_sq = True
            sq_escape = i > 0 and sql[i - 1] in ("e", "E") and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_"))
            i += 1
            continue
        if ch == '"':
            in_dq = True
            i += 1
            continue
        if ch == "(":
            if depth == 0:
                open_idx = i
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                return open_idx, i
        i += 1
    return -1, -1


def _normalize_pg_virtual_generated_ddl(raw_stmt: str) -> Tuple[str, Set[str]]:
    """Rewrite PostgreSQL 18 ``GENERATED ALWAYS AS (<expr>) [VIRTUAL]`` column
    clauses to ``STORED`` for sqlglot parsing while returning the lowercased
    names of columns that are ``VIRTUAL`` (explicitly or by PG 18 default)."""
    lp, rp = _find_pg_balanced_paren_span(raw_stmt, 0)
    if lp < 0 or rp <= lp:
        return raw_stmt, set()
    body = raw_stmt[lp + 1 : rp]
    parts = _split_top_level(body, ",", dialect=Dialect.POSTGRES)
    virtual_cols: Set[str] = set()
    new_parts: List[str] = []
    gen_re = re.compile(r"\bGENERATED\s+ALWAYS\s+AS\s*\(", re.IGNORECASE)
    for part in parts:
        m = gen_re.search(part)
        if not m:
            new_parts.append(part)
            continue
        _, close_idx = _find_pg_balanced_paren_span(part, m.end() - 1)
        if close_idx < 0:
            new_parts.append(part)
            continue
        tail = part[close_idx + 1 :]
        if re.match(r"^\s*STORED\b", tail, re.IGNORECASE):
            new_parts.append(part)
            continue
        # Extract leading column name (quoted or bare) to mark as virtual.
        p_strip = part.lstrip()
        m_col = re.match(r'^"((?:[^"]|"")+)"|^([A-Za-z_][A-Za-z0-9_$]*)', p_strip)
        if m_col:
            c_name = m_col.group(1).replace('""', '"') if m_col.group(1) is not None else m_col.group(2)
            virtual_cols.add(c_name.lower())
        m_virt = re.match(r"^(\s*)VIRTUAL\b", tail, re.IGNORECASE)
        if m_virt:
            new_tail = m_virt.group(1) + "STORED" + tail[m_virt.end() :]
        else:
            new_tail = " STORED" + tail
        new_parts.append(part[: close_idx + 1] + new_tail)
    return raw_stmt[: lp + 1] + ",".join(new_parts) + raw_stmt[rp:], virtual_cols


def parse_table_schema(
    schema_sql: str,
    dialect: Dialect = Dialect.SQLITE,
) -> Dict[str, TableSchema]:
    """Parses CREATE TABLE and CREATE [UNIQUE] INDEX statements into TableSchema objects via `sqlglot.parse`."""
    eff_dialect = dialect if dialect is not None else Dialect.SQLITE
    clean = _strip_sql_comments(schema_sql, dialect=eff_dialect)
    sg_d = _sqlglot_dialect(eff_dialect)
    schemas: Dict[str, TableSchema] = {}
    pending_indexes: Dict[str, List[TableIndexDef]] = {}
    for raw_stmt in _split_top_level(clean, ";", dialect=eff_dialect):
        raw_stmt_s = raw_stmt.strip()
        if not raw_stmt_s:
            continue
        parsed_idx = parse_table_index_def(raw_stmt_s, dialect=eff_dialect)
        if parsed_idx is not None:
            if not parsed_idx.table_name.startswith(("__matview_", "__mv_")):
                pending_indexes.setdefault(parsed_idx.table_name.lower(), []).append(parsed_idx)
            continue
        raw_col_types: Dict[str, str] = {}
        used_sqlite_type_fallback = False
        pg_virtual_cols_lower: Set[str] = set()
        parse_input_s = raw_stmt_s
        if eff_dialect == Dialect.POSTGRES and re.search(
            r"\bGENERATED\s+ALWAYS\s+AS\s*\(", raw_stmt_s, re.IGNORECASE
        ):
            parse_input_s, pg_virtual_cols_lower = _normalize_pg_virtual_generated_ddl(raw_stmt_s)
        try:
            stmt_ast = parse_sql_ast(parse_input_s, dialect=eff_dialect)
        except sqlglot.errors.SqlglotError:
            if eff_dialect == Dialect.POSTGRES and re.search(r"\bBIT\s+VARYING\b", parse_input_s, re.IGNORECASE):
                pg_norm_sql = re.sub(r"\bBIT\s+VARYING\b", "VARBIT", parse_input_s, flags=re.IGNORECASE)
                try:
                    stmt_ast = parse_sql_ast(pg_norm_sql, dialect=eff_dialect)
                except sqlglot.errors.SqlglotError:
                    continue
            elif eff_dialect == Dialect.SQLITE and re.match(
                r"^\s*CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\b", raw_stmt_s, re.IGNORECASE
            ):
                raw_col_types, sanitized_sql = _extract_sqlite_raw_types_and_sanitized_ddl(raw_stmt_s)
                try:
                    stmt_ast = parse_sql_ast(sanitized_sql, dialect=eff_dialect)
                    used_sqlite_type_fallback = True
                except sqlglot.errors.SqlglotError:
                    continue
            else:
                continue
        if not isinstance(stmt_ast, sqlglot_exp.Create) or (stmt_ast.kind or "").upper() != "TABLE":
            continue
        if eff_dialect == Dialect.SQLITE and not raw_col_types:
            raw_col_types, _ = _extract_sqlite_raw_types_and_sanitized_ddl(raw_stmt_s)
        schema_node = stmt_ast.this
        if not isinstance(schema_node, sqlglot_exp.Schema) or not isinstance(schema_node.this, sqlglot_exp.Table):
            continue
        table_name = schema_node.this.name
        if not table_name or table_name.startswith(("__matview_", "__mv_")):
            continue
        cols: Dict[str, ColumnDef] = {}
        table_pk_cols_order: List[str] = []
        table_pk_cols: Set[str] = set()
        inline_indexes: List[TableIndexDef] = []
        fk_cascades: List[str] = []
        has_on_conflict_replace = False

        st_sql = stmt_ast.sql(sg_d)
        st_toks = _tokenize_sql(st_sql, read=sg_d)
        for k_i in range(len(st_toks)):
            if (
                k_i + 2 < len(st_toks)
                and st_toks[k_i].token_type == SQLGlotTokenType.ON
                and st_toks[k_i + 1].text.upper() == "CONFLICT"
                and st_toks[k_i + 2].token_type == SQLGlotTokenType.REPLACE
            ):
                has_on_conflict_replace = True
            if (
                k_i + 2 < len(st_toks)
                and st_toks[k_i].token_type == SQLGlotTokenType.ON
                and st_toks[k_i + 1].token_type in (SQLGlotTokenType.DELETE, SQLGlotTokenType.UPDATE)
            ):
                ev_type = st_toks[k_i + 1].text.upper()
                act_w1 = st_toks[k_i + 2].text.upper()
                act_str = act_w1
                if act_w1 == "SET" and k_i + 3 < len(st_toks):
                    act_str = f"SET {st_toks[k_i + 3].text.upper()}"
                if act_str in ("CASCADE", "SET NULL", "SET DEFAULT"):
                    act_full = f"ON {ev_type} {act_str}"
                    if act_full not in fk_cascades:
                        fk_cascades.append(act_full)

        for ref_node in stmt_ast.find_all(sqlglot_exp.Reference):
            for opt in (ref_node.args.get("options") or []):
                opt_s = str(opt).strip().upper()
                if opt_s in (
                    "ON DELETE CASCADE", "ON DELETE SET NULL", "ON DELETE SET DEFAULT",
                    "ON UPDATE CASCADE", "ON UPDATE SET NULL", "ON UPDATE SET DEFAULT",
                ) and opt_s not in fk_cascades:
                    fk_cascades.append(opt_s)

        for pk_node in schema_node.find_all(sqlglot_exp.PrimaryKey):
            for pk_id in pk_node.expressions:
                pk_name = pk_id.name or (pk_id.this.name if isinstance(pk_id.this, sqlglot_exp.Expr) else "")
                if pk_name and pk_name.lower() not in table_pk_cols:
                    table_pk_cols.add(pk_name.lower())
                    table_pk_cols_order.append(pk_name)

        for uq_node in schema_node.find_all(sqlglot_exp.UniqueColumnConstraint):
            # Table-level UNIQUE(col1, col2, ...) inside Schema.expressions or Constraint.expressions
            uq_parent = uq_node.parent
            if isinstance(uq_parent, sqlglot_exp.ColumnConstraint):
                continue
            uq_schema = getattr(uq_node, "this", None)
            uq_exprs = (
                list(uq_schema.expressions)
                if isinstance(uq_schema, sqlglot_exp.Schema)
                else list(uq_node.expressions or ())
            )
            uq_cols = tuple(
                c.name or (c.this.name if isinstance(c.this, sqlglot_exp.Expr) else "")
                for c in uq_exprs
            )
            uq_cols = tuple(c for c in uq_cols if c)
            if uq_cols:
                inline_indexes.append(
                    TableIndexDef(
                        name=f"uq_{table_name}_{'_'.join(uq_cols)}",
                        table_name=table_name,
                        columns=uq_cols,
                        unique=True,
                    )
                )

        without_rowid = False
        strict = False
        if eff_dialect == Dialect.SQLITE:
            stripped_stmt = _strip_sql_comments(raw_stmt_s)
            rparen_idx = stripped_stmt.rfind(")")
            tail_opts = stripped_stmt[rparen_idx + 1 :] if rparen_idx >= 0 else ""
            without_rowid = bool(re.search(r"\bWITHOUT\s+ROWID\b", tail_opts, re.IGNORECASE))
            strict = bool(re.search(r"\bSTRICT\b", tail_opts, re.IGNORECASE))
        col_items: List[Tuple[str, str, bool, bool, bool]] = []
        generated_cols_lower: Set[str] = set()
        generation_exprs_lower: Dict[str, str] = {}
        all_pk_cols_lower: Set[str] = set(table_pk_cols)
        ordered_pk_cols: List[str] = list(table_pk_cols_order)
        for expr_item in schema_node.expressions:
            if eff_dialect == Dialect.SQLITE and isinstance(expr_item, sqlglot_exp.Identifier) and expr_item.name:
                # SQLite column without a declared type or constraints
                # ("CREATE TABLE t (id INTEGER PRIMARY KEY, x)"): sqlglot yields
                # a bare identifier. It is a real column with BLOB affinity
                # (values keep their storage class).
                col_items.append((expr_item.name, "BLOB", False, False, False))
                continue
            if isinstance(expr_item, sqlglot_exp.ColumnDef):
                col_name = expr_item.name
                if not col_name:
                    continue
                if eff_dialect == Dialect.SQLITE:
                    raw_decl = raw_col_types.get(col_name.lower(), "").strip()
                    parsed_dt = expr_item.kind.sql() if expr_item.kind is not None else ""
                    if used_sqlite_type_fallback:
                        data_type = raw_decl if raw_decl else "BLOB"
                    elif raw_decl and (
                        not parsed_dt
                        or _sqlite_type_affinity(raw_decl) != _sqlite_type_affinity(parsed_dt)
                    ):
                        # Preserve the raw SQLite declared type whenever sqlglot's
                        # canonicalization changes SQLite column affinity (e.g.
                        # BLOB -> VARBINARY (NUMERIC) or STRING -> TEXT (TEXT)).
                        data_type = raw_decl
                    elif parsed_dt:
                        data_type = parsed_dt
                    else:
                        # No declared type: BLOB affinity in SQLite. Declaring the
                        # derived backing columns TEXT would convert numbers to text.
                        data_type = "BLOB"
                elif expr_item.kind is not None:
                    data_type = expr_item.kind.sql(sg_d)
                else:
                    data_type = "TEXT"
                is_pk = any(
                    isinstance(c.kind, sqlglot_exp.PrimaryKeyColumnConstraint)
                    for c in expr_item.constraints
                )
                pk_desc = any(
                    isinstance(c.kind, sqlglot_exp.PrimaryKeyColumnConstraint)
                    and bool(c.kind.args.get("desc"))
                    for c in expr_item.constraints
                )
                if any(
                    isinstance(c.kind, sqlglot_exp.UniqueColumnConstraint)
                    for c in expr_item.constraints
                ):
                    inline_indexes.append(
                        TableIndexDef(
                            name=f"uq_{table_name}_{col_name}",
                            table_name=table_name,
                            columns=(col_name,),
                            unique=True,
                        )
                    )
                explicit_not_null = any(
                    isinstance(c.kind, sqlglot_exp.NotNullColumnConstraint)
                    and not c.kind.args.get("allow_null")
                    for c in expr_item.constraints
                )
                for c in expr_item.constraints:
                    if isinstance(c.kind, sqlglot_exp.ComputedColumnConstraint):
                        generated_cols_lower.add(col_name.lower())
                        if c.kind.this is not None:
                            generation_exprs_lower[col_name.lower()] = c.kind.this.sql(dialect=sg_d)
                if is_pk and col_name.lower() not in all_pk_cols_lower:
                    all_pk_cols_lower.add(col_name.lower())
                    ordered_pk_cols.append(col_name)
                col_items.append((col_name, data_type, is_pk, explicit_not_null, pk_desc))

        single_pk = len(all_pk_cols_lower) == 1
        for col_name, data_type, is_pk, explicit_not_null, pk_desc in col_items:
            eff_pk = is_pk or (col_name.lower() in all_pk_cols_lower)
            is_rowid_alias = False
            if eff_dialect == Dialect.SQLITE:
                # On SQLite, only explicit NOT NULL, WITHOUT ROWID PKs, or a
                # single-column INTEGER PRIMARY KEY (not DESC, not WITHOUT ROWID)
                # disallow NULL and enforce 64-bit signed integer storage.
                raw_declared = re.sub(
                    r"\s+",
                    " ",
                    raw_col_types.get(col_name.lower(), data_type).strip().upper(),
                )
                is_rowid_alias = (
                    eff_pk
                    and single_pk
                    and not without_rowid
                    and not pk_desc
                    and raw_declared == "INTEGER"
                )
                not_null = explicit_not_null or (eff_pk and (without_rowid or is_rowid_alias))
            else:
                not_null = explicit_not_null or eff_pk
            cols[col_name] = ColumnDef(
                name=col_name,
                data_type=data_type,
                not_null=not_null,
                primary_key=eff_pk,
                generated=col_name.lower() in generated_cols_lower,
                generated_virtual=col_name.lower() in pg_virtual_cols_lower,
                generation_expr=generation_exprs_lower.get(col_name.lower()),
                is_rowid_alias=is_rowid_alias,
            )

        schemas[table_name] = TableSchema(
            name=table_name,
            columns=cols,
            fk_cascade_actions=tuple(fk_cascades),
            has_on_conflict_replace=has_on_conflict_replace,
            indexes=tuple(inline_indexes),
            pk_columns=tuple(ordered_pk_cols),
            without_rowid=without_rowid,
            strict=strict,
        )

    if pending_indexes:
        low_to_key = {k.lower(): k for k in schemas}
        for tbl_low, idx_list in pending_indexes.items():
            tbl_key = low_to_key.get(tbl_low)
            if tbl_key is None:
                continue
            existing = schemas[tbl_key]
            existing_names = {i.name.lower() for i in existing.indexes}
            merged = list(existing.indexes) + [
                idx for idx in idx_list if idx.name.lower() not in existing_names
            ]
            schemas[tbl_key] = TableSchema(
                name=existing.name,
                columns=existing.columns,
                fk_cascade_actions=existing.fk_cascade_actions,
                has_on_conflict_replace=existing.has_on_conflict_replace,
                indexes=tuple(merged),
                pk_columns=existing.pk_columns,
                without_rowid=existing.without_rowid,
                strict=existing.strict,
            )
    return schemas


_SQL_EXPR_FUNCTIONS = {
    "COUNT", "SUM", "TOTAL", "AVG", "MIN", "MAX",
    "GROUP_CONCAT", "STRING_AGG", "JSON_GROUP_ARRAY", "JSON_GROUP_OBJECT", "ARRAY_AGG",
    "VAR_POP", "VARIANCE", "VAR_SAMP", "STDDEV_POP", "STDDEV", "STD", "STDDEV_SAMP",
    "BIT_AND", "BIT_OR", "BIT_XOR", "BOOL_AND", "EVERY", "BOOL_OR",
    "COALESCE", "IFNULL", "NULLIF", "IIF", "IF",
    "LOWER", "UPPER", "TRIM", "LTRIM", "RTRIM", "LENGTH", "CHAR_LENGTH",
    "SUBSTR", "SUBSTRING", "REPLACE", "INSTR", "PRINTF", "FORMAT", "CONCAT", "CONCAT_WS",
    "ABS", "ROUND", "CEIL", "CEILING", "FLOOR", "MOD", "POWER", "SQRT", "SIGN",
    "CAST", "TYPEOF", "QUOTE", "HEX", "ZEROBLOB",
    "DATE", "TIME", "DATETIME", "JULIANDAY", "STRFTIME", "UNIXEPOCH",
    "ROW_NUMBER", "RANK", "DENSE_RANK", "NTILE", "LAG", "LEAD",
    "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE", "PERCENT_RANK", "CUME_DIST",
    "JSON", "JSON_ARRAY", "JSON_OBJECT", "JSON_EXTRACT", "JSON_TYPE", "JSON_VALID",
}


_FN_TO_AGG_ENUM: Dict[str, AggFunc] = {
    "SUM": AggFunc.SUM,
    "TOTAL": AggFunc.TOTAL,
    "AVG": AggFunc.AVG,
    "MIN": AggFunc.MIN,
    "MAX": AggFunc.MAX,
    "GROUP_CONCAT": AggFunc.GROUP_CONCAT,
    "STRING_AGG": AggFunc.STRING_AGG,
    "JSON_GROUP_ARRAY": AggFunc.JSON_GROUP_ARRAY,
    "JSON_GROUP_OBJECT": AggFunc.JSON_GROUP_OBJECT,
    "ARRAY_AGG": AggFunc.ARRAY_AGG,
    "VAR_POP": AggFunc.VAR_POP,
    "VAR_SAMP": AggFunc.VAR_SAMP,
    "STDDEV_POP": AggFunc.STDDEV_POP,
    "STDDEV_SAMP": AggFunc.STDDEV_SAMP,
    "BIT_AND": AggFunc.BIT_AND,
    "BIT_OR": AggFunc.BIT_OR,
    "BIT_XOR": AggFunc.BIT_XOR,
    "BOOL_AND": AggFunc.BOOL_AND,
    "EVERY": AggFunc.BOOL_AND,
    "BOOL_OR": AggFunc.BOOL_OR,
}


# Non-standard statistical aggregate spellings whose meaning is defined by
# each engine, not by SQL: PostgreSQL's VARIANCE/STDDEV are the *sample*
# statistics, MySQL's and MariaDB's VARIANCE/STDDEV/STD the *population*
# ones. PostgreSQL has no STD() aggregate and SQLite has none of them (any
# function with such a name is an application-defined UDF of unknown
# semantics). A spelling missing from a dialect's map is rejected.
STATISTICAL_AGGREGATE_ALIASES: Dict[Dialect, Dict[str, AggFunc]] = {
    Dialect.POSTGRES: {
        "VARIANCE": AggFunc.VAR_SAMP,
        "STDDEV": AggFunc.STDDEV_SAMP,
    },
    Dialect.MYSQL: {
        "VARIANCE": AggFunc.VAR_POP,
        "STDDEV": AggFunc.STDDEV_POP,
        "STD": AggFunc.STDDEV_POP,
    },
    Dialect.SQLITE: {},
}
_STATISTICAL_ALIAS_NAMES = frozenset({"VARIANCE", "STDDEV", "STD"})


def _aggregate_function_for_dialect(fn_name: str, dialect: Dialect) -> Optional[AggFunc]:
    """Resolve an aggregate function name against the target dialect.

    Returns None for names that are not maintained aggregates, including a
    statistical alias the dialect does not define (callers reject those via
    `undefined_statistical_aggregate_aliases`).
    """
    fn_up = fn_name.upper()
    if fn_up in _STATISTICAL_ALIAS_NAMES:
        return STATISTICAL_AGGREGATE_ALIASES.get(Dialect(dialect), {}).get(fn_up)
    return _FN_TO_AGG_ENUM.get(fn_up)


def undefined_statistical_aggregate_aliases(sql: str, dialect: Dialect) -> List[str]:
    """Return the statistical alias calls in `sql` that `dialect` does not define."""
    defined = STATISTICAL_AGGREGATE_ALIASES.get(Dialect(dialect), {})
    try:
        toks = _tokenize_sql(sql, read=_sqlglot_dialect(dialect))
    except sqlglot.errors.SqlglotError:
        return []
    found: List[str] = []
    for idx in range(len(toks) - 1):
        name = toks[idx].text.upper()
        if (
            name in _STATISTICAL_ALIAS_NAMES
            and name not in defined
            and toks[idx + 1].token_type == SQLGlotTokenType.L_PAREN
            and toks[idx].token_type != SQLGlotTokenType.STRING
            and (idx == 0 or toks[idx - 1].token_type != SQLGlotTokenType.DOT)
            and name not in found
        ):
            found.append(name)
    return found


def _leading_function_name(expr: str, dialect: Dialect) -> Optional[str]:
    s = _unwrap_outer_parens(expr, dialect=dialect)
    try:
        toks = _tokenize_sql(s, read=_sqlglot_dialect(dialect))
    except sqlglot.errors.SqlglotError:
        return None
    if len(toks) >= 2 and toks[1].token_type == SQLGlotTokenType.L_PAREN:
        return toks[0].text.upper()
    return None


def resolve_statistical_aggregate_aliases(view_spec: ViewSpec, dialect: Dialect) -> ViewSpec:
    """Bind statistical alias spellings to the target engine's semantics.

    A `ViewSpec` may have been parsed for a different dialect than the one it
    is compiled for (the parser defaults to SQLite), so every top-level
    aggregate spelled as an alias is re-parsed for `dialect`. Spellings the
    engine does not define are rejected rather than guessed.
    """
    dialect = Dialect(dialect)
    undefined = undefined_statistical_aggregate_aliases(view_spec.raw_select_sql, dialect)
    if undefined:
        native = ", ".join(f"{name}()" for name in undefined)
        raise UnsupportedMaterializedViewError(
            f"View '{view_spec.view_name}' uses {native}, which is not a native {dialect.value} "
            "aggregate; statistical alias semantics (sample vs population) are engine-specific, "
            "so it is not materialized. Use VAR_POP/VAR_SAMP/STDDEV_POP/STDDEV_SAMP explicitly."
        )
    changed = False
    items: List[SelectItem] = []
    for it in view_spec.select_items:
        fn = _leading_function_name(it.raw_expr, dialect) if it.is_aggregate else None
        if fn in _STATISTICAL_ALIAS_NAMES:
            reparsed = _parse_single_agg_expr(it.raw_expr, it.alias, dialect=dialect)
            if reparsed is not None and reparsed != it:
                items.append(reparsed)
                changed = True
                continue
        items.append(it)
    additional = tuple(
        resolve_statistical_aggregate_aliases(extra, dialect) for extra in view_spec.additional_views
    )
    if additional != view_spec.additional_views:
        changed = True
    if not changed:
        return view_spec
    return replace(view_spec, select_items=tuple(items), additional_views=additional)


def _unwrap_outer_parens(
    sql: str,
    dialect: Dialect = Dialect.SQLITE,
) -> str:
    s = sql.strip()
    sg_d = _sqlglot_dialect(dialect)
    while s.startswith("(") and s.endswith(")"):
        toks = _tokenize_sql(s, read=sg_d)
        if not toks or toks[0].token_type != SQLGlotTokenType.L_PAREN:
            break
        depth = 0
        closes_at_end = False
        for idx, tk in enumerate(toks):
            if tk.token_type == SQLGlotTokenType.L_PAREN:
                depth += 1
            elif tk.token_type == SQLGlotTokenType.R_PAREN:
                depth -= 1
                if depth == 0:
                    closes_at_end = (idx == len(toks) - 1)
                    break
        if closes_at_end:
            s = s[1:-1].strip()
        else:
            break
    return s


def _parse_single_agg_expr(
    expr: str,
    alias: str,
    dialect: Dialect = Dialect.SQLITE,
) -> Optional[SelectItem]:
    """Parses a single top-level aggregate function call using `sqlglot.parse_one` and `sqlglot.tokenize`."""
    s = _unwrap_outer_parens(expr, dialect=dialect)
    if not s:
        return None
    try:
        ast_node = parse_sql_ast(s, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return None

    if isinstance(ast_node, sqlglot_exp.Window) or ast_node.find(sqlglot_exp.Window):
        return None

    agg_node: sqlglot_exp.Expr = ast_node.this if isinstance(ast_node, sqlglot_exp.Filter) else ast_node
    if not isinstance(agg_node, (sqlglot_exp.AggFunc, sqlglot_exp.Func)):
        return None

    sg_d = _sqlglot_dialect(dialect)
    toks = _tokenize_sql(s, read=sg_d)
    if len(toks) < 3 or not toks[0].text or not (toks[0].text[0].isalpha() or toks[0].text[0] == "_") or toks[1].token_type != SQLGlotTokenType.L_PAREN:
        return None

    depth = 0
    close_idx = -1
    for idx in range(1, len(toks)):
        if toks[idx].token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif toks[idx].token_type == SQLGlotTokenType.R_PAREN:
            depth -= 1
            if depth == 0:
                close_idx = idx
                break
    if close_idx == -1:
        return None

    filter_where: Optional[str] = None
    if close_idx < len(toks) - 1:
        # Must be `FILTER ( WHERE <filter_where> )` spanning to the end of `s`
        if (
            close_idx + 4 >= len(toks)
            or toks[close_idx + 1].token_type != SQLGlotTokenType.FILTER
            or toks[close_idx + 2].token_type != SQLGlotTokenType.L_PAREN
            or toks[close_idx + 3].token_type != SQLGlotTokenType.WHERE
            or toks[-1].token_type != SQLGlotTokenType.R_PAREN
        ):
            return None
        f_depth = 0
        f_close = -1
        for idx in range(close_idx + 2, len(toks)):
            if toks[idx].token_type == SQLGlotTokenType.L_PAREN:
                f_depth += 1
            elif toks[idx].token_type == SQLGlotTokenType.R_PAREN:
                f_depth -= 1
                if f_depth == 0:
                    f_close = idx
                    break
        if f_close != len(toks) - 1:
            return None
        filter_where = s[toks[close_idx + 3].end + 1 : toks[-1].start].strip()
        if not filter_where:
            return None
        fw_toks = _tokenize_sql(filter_where, read=sg_d)
        fw_d = 0
        for ftk in fw_toks:
            if ftk.token_type == SQLGlotTokenType.L_PAREN:
                fw_d += 1
            elif ftk.token_type == SQLGlotTokenType.R_PAREN:
                fw_d -= 1
                if fw_d < 0:
                    return None
        if fw_d != 0:
            return None

    fn_raw = toks[0].text.upper()
    inner_args_sql = s[toks[1].end + 1 : toks[close_idx].start].strip()
    in_toks = _tokenize_sql(inner_args_sql, read=sg_d)
    is_dist = bool(agg_node.args.get("distinct")) or isinstance(agg_node.this, sqlglot_exp.Distinct)
    if in_toks and in_toks[0].token_type == SQLGlotTokenType.DISTINCT:
        is_dist = True
        inner_args_sql = inner_args_sql[in_toks[0].end + 1 :].strip()
    elif in_toks and in_toks[0].token_type == SQLGlotTokenType.ALL:
        is_dist = False
        inner_args_sql = inner_args_sql[in_toks[0].end + 1 :].strip()

    if fn_raw == "COUNT" or (
        fn_raw in _FN_TO_AGG_ENUM
        and _FN_TO_AGG_ENUM[fn_raw] not in (
            AggFunc.GROUP_CONCAT, AggFunc.STRING_AGG, AggFunc.JSON_GROUP_ARRAY,
            AggFunc.JSON_GROUP_OBJECT, AggFunc.ARRAY_AGG,
        )
    ):
        depth = 0
        for itk in _tokenize_sql(inner_args_sql, read=sg_d):
            if itk.token_type == SQLGlotTokenType.L_PAREN:
                depth += 1
            elif itk.token_type == SQLGlotTokenType.R_PAREN:
                depth -= 1
            elif depth == 0 and itk.token_type == SQLGlotTokenType.ORDER_BY:
                # PostgreSQL accepts SUM(x ORDER BY y); the maintenance SQL
                # would carry the clause into positions where it is a syntax
                # error (native: "syntax error at or near ORDER").
                raise UnsupportedMaterializedViewError(
                    f"ORDER BY inside {fn_raw}(...) is not supported in materialized views: {expr.strip()!r}"
                )
    arg_parts = [p.strip() for p in _split_top_level(inner_args_sql, ",", dialect=dialect) if p.strip()] if inner_args_sql else []
    first_arg = arg_parts[0] if arg_parts else ""
    extra_args = ", ".join(arg_parts[1:]) if len(arg_parts) > 1 else None
    src_col: Optional[str] = None
    if first_arg and first_arg != "*":
        try:
            first_node = parse_sql_ast(first_arg, dialect=dialect)
            if isinstance(first_node, sqlglot_exp.Column) and not isinstance(first_node.this, sqlglot_exp.Star):
                src_col = first_node.name
        except sqlglot.errors.SqlglotError:
            pass

    if fn_raw == "COUNT":
        if first_arg in ("*", ""):
            return SelectItem(
                raw_expr=expr.strip(),
                alias=alias,
                is_aggregate=True,
                agg_func=AggFunc.COUNT_STAR,
                arg_expr="*",
                is_distinct=False,
                filter_where=filter_where,
            )
        return SelectItem(
            raw_expr=expr.strip(),
            alias=alias,
            is_aggregate=True,
            agg_func=AggFunc.COUNT_COL,
            arg_expr=first_arg,
            is_distinct=is_dist,
            filter_where=filter_where,
            source_col=src_col,
        )

    agg_func = _aggregate_function_for_dialect(fn_raw, dialect)
    if agg_func is None:
        return None

    return SelectItem(
        raw_expr=expr.strip(),
        alias=alias,
        is_aggregate=True,
        agg_func=agg_func,
        arg_expr=first_arg,
        extra_args=extra_args,
        is_distinct=is_dist,
        filter_where=filter_where,
        source_col=src_col,
    )


def _contains_aggregate_call(
    expr: str,
    dialect: Dialect = Dialect.SQLITE,
) -> bool:
    """Checks via `sqlglot.parse_one` whether `expr` contains an aggregate function call (excluding window functions and subqueries)."""
    try:
        ast_node = parse_sql_ast(expr, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return False
    if ast_node.find(sqlglot_exp.Window) or ast_node.find(sqlglot_exp.Subquery, sqlglot_exp.Select):
        return False
    if ast_node.find(sqlglot_exp.AggFunc):
        return True
    for anon_node in ast_node.find_all(sqlglot_exp.Anonymous):
        name_up = anon_node.name.upper()
        if name_up in _FN_TO_AGG_ENUM or name_up in _STATISTICAL_ALIAS_NAMES or name_up == "COUNT":
            return True
    return False


def _parse_select_item(
    raw_item: str,
    idx: int,
    dialect: Dialect = Dialect.SQLITE,
) -> SelectItem:
    """Parses a single `SELECT` projection item via `sqlglot.parse_one`."""
    s = raw_item.strip()
    sg_d = _sqlglot_dialect(dialect)
    try:
        sel_ast = sqlglot.parse_one(f"SELECT {s}", read=sg_d)
        proj_ast = sel_ast.expressions[0] if isinstance(sel_ast, sqlglot_exp.Select) and sel_ast.expressions else parse_sql_ast(s, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        proj_ast = sqlglot_exp.Expr()

    if isinstance(proj_ast, sqlglot_exp.Alias):
        alias = proj_ast.alias
        inner_node = proj_ast.this
        toks = _tokenize_sql(s, read=sg_d)
        if len(toks) >= 2 and toks[-2].token_type == SQLGlotTokenType.ALIAS:
            expr = s[: toks[-2].start].strip()
        elif len(toks) >= 2:
            expr = s[: toks[-1].start].strip()
        else:
            expr = inner_node.sql(sg_d).strip() if isinstance(inner_node, sqlglot_exp.Expr) else str(inner_node or "").strip()
    else:
        inner_node = proj_ast
        expr = s
        if isinstance(inner_node, sqlglot_exp.Column) and not isinstance(inner_node.this, sqlglot_exp.Star):
            alias = inner_node.name
        else:
            alias = f"col_{idx}"

    parsed_agg = _parse_single_agg_expr(expr, alias, dialect=dialect)
    if parsed_agg is not None:
        return parsed_agg

    if _contains_aggregate_call(expr, dialect=dialect):
        return SelectItem(
            raw_expr=expr,
            alias=alias,
            is_aggregate=True,
            agg_func=None,
            arg_expr=expr,
        )

    src_col = (
        inner_node.name
        if isinstance(inner_node, sqlglot_exp.Column) and not isinstance(inner_node.this, sqlglot_exp.Star)
        else None
    )
    return SelectItem(
        raw_expr=expr,
        alias=alias,
        is_aggregate=False,
        source_col=src_col,
    )


def _extract_verbatim_cte_or_subquery_sql(
    raw_select_sql: str,
    has_cte: bool,
    dialect: Dialect = Dialect.SQLITE,
) -> Optional[str]:
    """Slices the verbatim inner query of a single CTE (`WITH x AS (...)`) or `FROM (...)` subquery via `sqlglot.tokenize`."""
    s = raw_select_sql.strip()
    toks = _tokenize_sql(s, read=_sqlglot_dialect(dialect))
    if not toks:
        return None
    if has_cte:
        depth = 0
        for i, tk in enumerate(toks):
            if depth == 0 and tk.token_type == SQLGlotTokenType.L_PAREN and i > 0 and toks[i - 1].token_type == SQLGlotTokenType.ALIAS:
                start_char = tk.end + 1
                p_depth = 1
                for j in range(i + 1, len(toks)):
                    if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                        p_depth += 1
                    elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                        p_depth -= 1
                        if p_depth == 0:
                            return s[start_char : toks[j].start].strip()
            elif tk.token_type == SQLGlotTokenType.L_PAREN:
                depth += 1
            elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
                depth -= 1
    else:
        depth = 0
        seen_from = False
        for i, tk in enumerate(toks):
            if depth == 0 and tk.token_type == SQLGlotTokenType.FROM:
                seen_from = True
            elif seen_from and depth == 0 and tk.token_type == SQLGlotTokenType.L_PAREN:
                start_char = tk.end + 1
                p_depth = 1
                for j in range(i + 1, len(toks)):
                    if toks[j].token_type == SQLGlotTokenType.L_PAREN:
                        p_depth += 1
                    elif toks[j].token_type == SQLGlotTokenType.R_PAREN:
                        p_depth -= 1
                        if p_depth == 0:
                            return s[start_char : toks[j].start].strip()
            elif tk.token_type == SQLGlotTokenType.L_PAREN:
                depth += 1
            elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
                depth -= 1
    return None


def _extract_cte_names_and_strip_with(
    select_sql: str,
    dialect: Dialect = Dialect.SQLITE,
) -> Tuple[Set[str], str, bool]:
    """Extracts CTE names and the main SELECT body from a query using `sqlglot.parse_one` AST and `sqlglot.tokenize`."""
    s = select_sql.strip()
    sg_d = _sqlglot_dialect(dialect)
    try:
        q_ast = parse_sql_ast(s, dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return set(), s, False
    with_node = q_ast.args.get("with_") or q_ast.args.get("with")
    if not isinstance(with_node, sqlglot_exp.With):
        return set(), s, False
    cte_names = {cte.alias for cte in with_node.expressions if isinstance(cte, sqlglot_exp.CTE) and cte.alias}
    toks = _tokenize_sql(s, read=sg_d)
    depth = 0
    for tk in toks:
        if tk.token_type == SQLGlotTokenType.L_PAREN:
            depth += 1
        elif tk.token_type == SQLGlotTokenType.R_PAREN and depth > 0:
            depth -= 1
        elif depth == 0 and tk.token_type == SQLGlotTokenType.SELECT:
            main_select = s[tk.start:].strip()
            return cte_names, (main_select if main_select else s), True
    no_with = q_ast.copy()
    no_with.set("with_", None)
    no_with.set("with", None)
    no_with.args.pop("_raw_sql", None)
    main_select = no_with.sql(sg_d).strip()
    return cte_names, (main_select if main_select else s), True


def _parse_from_and_joins(
    from_clause: str,
    cte_names: Set[str],
    dialect: Dialect = Dialect.SQLITE,
) -> Tuple[List[JoinSpec], bool]:
    """Parses a `FROM ... [JOIN ...]` clause into `JoinSpec` objects using `sqlglot.parse_one`."""
    s = from_clause.strip()
    if not s:
        return [], False
    sg_d = _sqlglot_dialect(dialect)
    toks = _tokenize_sql(s, read=sg_d)
    if not toks:
        return [], False
    if toks[0].token_type == SQLGlotTokenType.FROM:
        s_body = s[toks[0].end + 1 :].strip()
    else:
        s_body = s
    if not s_body:
        return [], False
    try:
        sel_ast = sqlglot.parse_one(f"SELECT * FROM {s_body}", read=sg_d)
    except sqlglot.errors.SqlglotError:
        return [], False
    if not isinstance(sel_ast, sqlglot_exp.Select):
        return [], False

    from_node = sel_ast.args.get("from_") or sel_ast.args.get("from")
    join_nodes = sel_ast.args.get("joins") or []
    if not isinstance(from_node, sqlglot_exp.From):
        return [], False

    # Also extract raw ON condition slices from `s_body` via `_tokenize_sql`
    body_toks = _tokenize_sql(s_body, read=sg_d)
    raw_on_slices: List[Optional[str]] = []
    b_depth = 0
    join_kw_words = {"INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL", "JOIN"}
    on_start_char: Optional[int] = None
    in_join = False
    b_idx = 0
    while b_idx < len(body_toks):
        btk = body_toks[b_idx]
        if btk.token_type == SQLGlotTokenType.L_PAREN:
            b_depth += 1
        elif btk.token_type == SQLGlotTokenType.R_PAREN and b_depth > 0:
            b_depth -= 1
        elif b_depth == 0:
            if btk.token_type == SQLGlotTokenType.COMMA or btk.text.upper() in join_kw_words:
                if in_join:
                    raw_on_slices.append(
                        s_body[on_start_char : btk.start].strip() if on_start_char is not None else None
                    )
                    on_start_char = None
                in_join = True
                if btk.token_type != SQLGlotTokenType.COMMA:
                    while b_idx + 1 < len(body_toks) and body_toks[b_idx + 1].text.upper() in (join_kw_words | {"OUTER"}):
                        b_idx += 1
            elif in_join and btk.token_type == SQLGlotTokenType.ON and on_start_char is None:
                on_start_char = btk.end + 1
        b_idx += 1
    if in_join:
        raw_on_slices.append(s_body[on_start_char:].strip() if on_start_char is not None else None)

    joins: List[JoinSpec] = []
    has_subquery = False

    first_target = from_node.this
    if isinstance(first_target, sqlglot_exp.Table):
        t_name = first_target.name or "t"
        al = first_target.alias or t_name
        joins.append(JoinSpec(join_type="FROM", table_name=t_name, alias=al))
    else:
        has_subquery = True
        al = (first_target.alias if isinstance(first_target, sqlglot_exp.Expr) else "") or "__mv_sub"
        joins.append(JoinSpec(join_type="FROM", table_name=al, alias=al))

    for idx, j_node in enumerate(join_nodes):
        side = j_node.side.upper()
        kind = j_node.kind.upper()
        method = j_node.method.upper()
        if method == "NATURAL":
            j_type = "NATURAL JOIN"
        elif side == "LEFT":
            j_type = "LEFT JOIN"
        elif side == "RIGHT":
            j_type = "RIGHT JOIN"
        elif side == "FULL":
            j_type = "FULL JOIN"
        elif kind == "CROSS" or (
            not side and not kind and not method and not j_node.args.get("on") and not j_node.args.get("using")
        ):
            j_type = "CROSS JOIN"
        else:
            j_type = "INNER JOIN"

        if j_node.args.get("using"):
            has_subquery = True

        j_target = j_node.this
        if isinstance(j_target, sqlglot_exp.Table):
            t_name = j_target.name or f"t_{idx}"
            al = j_target.alias or t_name
        else:
            has_subquery = True
            t_name = (j_target.alias if isinstance(j_target, sqlglot_exp.Expr) else "") or f"__mv_sub_{idx}"
            al = t_name

        on_expr = j_node.args.get("on")
        on_cond: Optional[str] = None
        if idx < len(raw_on_slices) and raw_on_slices[idx]:
            on_cond = raw_on_slices[idx]
        elif isinstance(on_expr, sqlglot_exp.Expr):
            on_cond = on_expr.sql(sg_d).strip()
        joins.append(
            JoinSpec(
                join_type=j_type,
                table_name=t_name,
                alias=al,
                on_condition=on_cond,
            )
        )

    return joins, has_subquery


def _extract_all_referenced_tables(
    select_sql: str,
    cte_names: Set[str],
    known_tables: Optional[Set[str]] = None,
    dialect: Dialect = Dialect.SQLITE,
) -> List[str]:
    """Extracts all referenced base tables across CTEs, FROM/JOIN clauses, set operations, and subqueries via `sqlglot` AST."""
    cte_lower = {c.lower() for c in cte_names}
    found: List[str] = []
    seen_lower: Set[str] = set()
    sql_reserved = {
        "select", "from", "where", "group", "by", "having", "order", "limit", "offset",
        "union", "all", "intersect", "except", "join", "inner", "left", "right", "full",
        "outer", "cross", "natural", "on", "using", "as", "distinct", "case", "when",
        "then", "else", "end", "and", "or", "not", "in", "exists", "is", "null", "true",
        "false", "dual", "lateral", "values",
    }

    def _add_candidate(cand: str, was_quoted: bool = False) -> None:
        if not cand:
            return
        c_low = cand.lower()
        if c_low in cte_lower or c_low.startswith(("__matview_", "__mv_")):
            return
        if not was_quoted and c_low in sql_reserved and (not known_tables or c_low not in known_tables):
            return
        if known_tables is not None and known_tables and c_low not in known_tables:
            return
        if c_low not in seen_lower:
            seen_lower.add(c_low)
            found.append(cand)

    sg_d = _sqlglot_dialect(dialect)
    try:
        q_ast = parse_sql_ast(select_sql, dialect=dialect)
        for tbl_node in q_ast.find_all(sqlglot_exp.Table):
            id_node = tbl_node.this
            was_q = isinstance(id_node, sqlglot_exp.Identifier) and id_node.quoted
            _add_candidate(tbl_node.name, was_quoted=was_q)
    except sqlglot.errors.SqlglotError:
        pass

    # Also scan token stream for any `FROM <ident>` / `JOIN <ident>` inside nested expressions
    toks = _tokenize_sql(select_sql, read=sg_d)
    for idx, tk in enumerate(toks):
        if tk.token_type in (SQLGlotTokenType.FROM, SQLGlotTokenType.JOIN) and idx + 1 < len(toks):
            nxt = toks[idx + 1]
            if _is_word_or_ident_token(nxt):
                _add_candidate(nxt.text, was_quoted=(nxt.token_type == SQLGlotTokenType.IDENTIFIER))

    return found


def parse_view_index_spec(
    spec: ViewIndexSpec | str | Sequence[str] | Dict[str, Any],
    default_view_name: Optional[str] = None,
    dialect: Dialect = Dialect.SQLITE,
) -> ViewIndexSpec:
    """Parses a user-defined index specification (`CREATE [UNIQUE] INDEX ...`, tuple/list, dict, or ViewIndexSpec) via `sqlglot` AST."""
    if isinstance(spec, ViewIndexSpec):
        if spec.target_view is None and default_view_name:
            return ViewIndexSpec(
                columns=spec.columns,
                name=spec.name,
                unique=spec.unique,
                where_clause=spec.where_clause,
                using_method=spec.using_method,
                target_view=default_view_name,
            )
        return spec

    if isinstance(spec, dict):
        cols_raw = spec.get("columns") or spec.get("cols")
        if isinstance(cols_raw, str):
            cols = tuple(c.strip() for c in cols_raw.split(",") if c.strip())
        elif isinstance(cols_raw, Sequence):
            cols = tuple(str(c).strip() for c in cols_raw if str(c).strip())
        else:
            raise ValueError(f"Index dict specification requires 'columns': {spec!r}")
        if not cols:
            raise ValueError(f"Index specification must contain at least one column: {spec!r}")
        return ViewIndexSpec(
            columns=cols,
            name=spec.get("name"),
            unique=bool(spec.get("unique", False)),
            where_clause=spec.get("where") or spec.get("where_clause"),
            using_method=spec.get("using") or spec.get("using_method"),
            target_view=spec.get("target_view") or spec.get("view") or default_view_name,
        )

    if isinstance(spec, (tuple, list)):
        cols = tuple(str(c).strip() for c in spec if str(c).strip())
        if not cols:
            raise ValueError(f"Index column list cannot be empty: {spec!r}")
        return ViewIndexSpec(columns=cols, target_view=default_view_name)

    if not isinstance(spec, str):
        raise ValueError(f"Unsupported index specification type: {type(spec).__name__}")

    s = _strip_sql_comments(spec, dialect=dialect).strip().rstrip(";").strip()
    if not s:
        raise ValueError(f"Empty index specification: {spec!r}")

    sg_d = _sqlglot_dialect(dialect)
    toks = _tokenize_sql(s, read=sg_d)
    if toks and toks[0].token_type == SQLGlotTokenType.CREATE:
        idx_create = parse_sql_ast(s, dialect=dialect)
        if isinstance(idx_create, sqlglot_exp.Create) and (idx_create.kind or "").upper() == "INDEX":
            idx_node = idx_create.this
            params_node = idx_node.args.get("params") if isinstance(idx_node, sqlglot_exp.Index) else None
            if isinstance(idx_node, sqlglot_exp.Index) and isinstance(params_node, sqlglot_exp.IndexParameters):
                cols = tuple(c.sql(sg_d).strip() for c in (params_node.args.get("columns") or []) if c.sql(sg_d).strip())
                if not cols:
                    raise ValueError(f"CREATE INDEX statement has no columns: {spec!r}")
                tbl_node = idx_node.args.get("table")
                target_v = tbl_node.name if isinstance(tbl_node, sqlglot_exp.Table) else default_view_name
                where_node = params_node.args.get("where")
                where_c: Optional[str] = None
                where_tok_ci = _find_top_level_token(s, (SQLGlotTokenType.WHERE,), dialect=dialect)
                if where_tok_ci is not None:
                    where_c = s[where_tok_ci.end + 1 :].strip()
                elif isinstance(where_node, sqlglot_exp.Where) and where_node.this is not None:
                    where_c = where_node.this.sql(sg_d).strip()
                using_node = params_node.args.get("using")
                using_m = (
                    (using_node.name or using_node.sql(sg_d).strip())
                    if isinstance(using_node, sqlglot_exp.Expr)
                    else (str(using_node) if using_node else None)
                )
                return ViewIndexSpec(
                    columns=cols,
                    name=idx_node.name or None,
                    unique=bool(idx_create.args.get("unique")),
                    where_clause=where_c,
                    using_method=using_m,
                    target_view=target_v,
                )

    # Shorthand string format: `[UNIQUE:]col1, col2 [WHERE ...]`
    is_unique_sh = False
    if len(toks) >= 2 and toks[0].token_type == SQLGlotTokenType.UNIQUE and toks[1].token_type == SQLGlotTokenType.COLON:
        is_unique_sh = True
        s = s[toks[1].end + 1 :].strip()

    where_sh: Optional[str] = None
    where_tok = _find_top_level_token(s, (SQLGlotTokenType.WHERE,), dialect=dialect)
    if where_tok is not None:
        where_sh = s[where_tok.end + 1 :].strip()
        s = s[: where_tok.start].strip()

    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    cols_sh = tuple(c.strip() for c in _split_top_level(s, ",", dialect=dialect) if c.strip())
    if not cols_sh:
        raise ValueError(f"Invalid index specification: {spec!r}")
    return ViewIndexSpec(
        columns=cols_sh,
        unique=is_unique_sh,
        where_clause=where_sh,
        target_view=default_view_name,
    )


def parse_view_ast(
    view_or_select_sql: str,
    dialect: Dialect = Dialect.SQLITE,
    known_tables: Optional[Set[str]] = None,
) -> SQLSelectAST:
    """
    Parses a `CREATE VIEW ... AS SELECT ...` or standalone `SELECT ...` query into
    a `SQLSelectAST` backed by `sqlglot.parse_one`.
    """
    clean = _strip_sql_comments(view_or_select_sql, dialect=dialect).strip().rstrip(";").strip()
    sg_ast = parse_sql_ast(clean, dialect=dialect)
    toks = _tokenize_sql(clean, read=_sqlglot_dialect(dialect))
    if toks and toks[0].token_type == SQLGlotTokenType.CREATE:
        vs = _parse_single_create_view_stmt(clean, known_tables=known_tables, dialect=dialect)
    else:
        vs = _parse_single_create_view_stmt(f"CREATE VIEW __mv_ast_tmp AS {clean}", known_tables=known_tables, dialect=dialect)
    cte_names, main_sel, has_cte = _extract_cte_names_and_strip_with(vs.raw_select_sql, dialect=dialect)
    return SQLSelectAST(
        raw_sql=vs.raw_select_sql,
        cte_names=tuple(sorted(cte_names)),
        main_select_sql=main_sel,
        has_cte=has_cte,
        has_distinct=vs.has_top_level_distinct,
        select_items=vs.select_items,
        from_clause=vs.base_table,
        joins=vs.joins,
        where_clause=vs.where_clause,
        group_by_exprs=vs.group_by_cols,
        having_clause=vs.having_clause,
        has_window_func=vs.has_window_func,
        has_compound_set_op=vs.has_compound_set_op,
        has_limit_or_order=vs.has_limit_or_order,
        has_subquery_in_from=vs.has_subquery_in_from,
        sqlglot_ast=sg_ast,
    )


def _parse_single_create_view_stmt(
    stmt_sql: str,
    known_tables: Optional[Set[str]] = None,
    extra_indexes: Sequence[ViewIndexSpec] = (),
    dialect: Dialect = Dialect.SQLITE,
) -> ViewSpec:
    """Parses a single `CREATE VIEW ... AS SELECT ...` statement into a `ViewSpec` using `sqlglot.parse_one`."""
    clean = _strip_sql_comments(stmt_sql, dialect=dialect).strip().rstrip(";").strip()
    try:
        create_ast = parse_sql_ast(clean, dialect=dialect)
    except sqlglot.errors.SqlglotError as exc:
        raise ValueError(f"Invalid CREATE VIEW syntax: {stmt_sql!r}") from exc

    if not isinstance(create_ast, sqlglot_exp.Create) or (create_ast.kind or "").upper() != "VIEW":
        raise ValueError(f"Invalid CREATE VIEW syntax: {stmt_sql!r}")

    view_tbl = create_ast.this
    view_name = view_tbl.name if isinstance(view_tbl, sqlglot_exp.Table) else ""
    if not view_name:
        raise ValueError(f"Invalid CREATE VIEW syntax (missing view name): {stmt_sql!r}")

    query_ast = create_ast.expression
    if not isinstance(query_ast, (sqlglot_exp.Select, sqlglot_exp.SetOperation)):
        raise ValueError(f"View body must start with SELECT or WITH: {stmt_sql!r}")

    sg_d = _sqlglot_dialect(dialect)
    as_tok = _find_top_level_token(clean, (SQLGlotTokenType.ALIAS,), dialect=dialect)
    raw_select = clean[as_tok.end + 1 :].strip() if as_tok is not None else query_ast.sql(sg_d).strip()
    cte_names, main_select_body, has_cte = _extract_cte_names_and_strip_with(raw_select, dialect=dialect)

    has_window = bool(query_ast.find(sqlglot_exp.Window))
    has_compound = isinstance(query_ast, sqlglot_exp.SetOperation)

    first_select: sqlglot_exp.Expr = query_ast
    while isinstance(first_select, sqlglot_exp.SetOperation):
        first_select = first_select.this
    if not isinstance(first_select, sqlglot_exp.Select):
        raise ValueError(f"Unable to parse SELECT clause in view {view_name!r}: {raw_select!r}")

    has_limit_or_order = bool(
        query_ast.find(sqlglot_exp.Order, sqlglot_exp.Limit, sqlglot_exp.Offset)
        if has_compound
        else (
            first_select.args.get("order")
            or first_select.args.get("limit")
            or first_select.args.get("offset")
        )
    )

    has_top_distinct = bool(first_select.args.get("distinct"))

    if has_compound:
        _, setop_branches = _extract_set_op_branches_from_ast(query_ast, main_select_body, dialect=dialect)
        first_branch_sql = setop_branches[0] if setop_branches else main_select_body
    else:
        first_branch_sql = _unwrap_outer_parens(main_select_body, dialect=dialect)

    fb_toks = _tokenize_sql(first_branch_sql, read=sg_d)
    fb_depth = 0
    sel_tok_i = -1
    from_tok_i = -1
    where_tok_i = -1
    group_tok_i = -1
    having_tok_i = -1
    trail_tok_i = -1
    for t_i, ftk in enumerate(fb_toks):
        if ftk.token_type == SQLGlotTokenType.L_PAREN:
            fb_depth += 1
        elif ftk.token_type == SQLGlotTokenType.R_PAREN and fb_depth > 0:
            fb_depth -= 1
        elif fb_depth == 0:
            tt = ftk.token_type
            if tt == SQLGlotTokenType.SELECT and sel_tok_i == -1:
                sel_tok_i = t_i
            elif sel_tok_i != -1:
                if tt == SQLGlotTokenType.FROM and from_tok_i == -1:
                    from_tok_i = t_i
                elif tt == SQLGlotTokenType.WHERE and where_tok_i == -1:
                    where_tok_i = t_i
                elif tt == SQLGlotTokenType.GROUP_BY and group_tok_i == -1:
                    group_tok_i = t_i
                elif tt == SQLGlotTokenType.HAVING and having_tok_i == -1:
                    having_tok_i = t_i
                elif tt in (
                    SQLGlotTokenType.WINDOW,
                    SQLGlotTokenType.ORDER_BY,
                    SQLGlotTokenType.LIMIT,
                    SQLGlotTokenType.OFFSET,
                    SQLGlotTokenType.FETCH,
                ) and trail_tok_i == -1:
                    trail_tok_i = t_i

    clause_tok_positions = [
        p for p in (from_tok_i, where_tok_i, group_tok_i, having_tok_i, trail_tok_i) if p != -1
    ]

    def _slice_fb_clause(tok_idx: int) -> str:
        if tok_idx == -1:
            return ""
        c_start = fb_toks[tok_idx].end + 1
        later_pos = [p for p in clause_tok_positions if p > tok_idx]
        c_end = fb_toks[min(later_pos)].start if later_pos else len(first_branch_sql)
        return first_branch_sql[c_start:c_end].strip()

    if sel_tok_i != -1:
        p_start_i = sel_tok_i + 1
        if p_start_i < len(fb_toks) and fb_toks[p_start_i].token_type in (SQLGlotTokenType.DISTINCT, SQLGlotTokenType.ALL):
            p_start_i += 1
        p_start_char = fb_toks[p_start_i].start if p_start_i < len(fb_toks) else len(first_branch_sql)
        p_end_char = fb_toks[min(clause_tok_positions)].start if clause_tok_positions else len(first_branch_sql)
        proj_list_sql = first_branch_sql[p_start_char:p_end_char].strip()
        raw_proj_items = [p.strip() for p in _split_top_level(proj_list_sql, ",", dialect=dialect) if p.strip()]
    else:
        raw_proj_items = [proj.sql(sg_d) for proj in first_select.expressions]

    select_items = [
        _parse_select_item(raw_p, idx, dialect=dialect)
        for idx, raw_p in enumerate(raw_proj_items)
    ]

    from_clause = _slice_fb_clause(from_tok_i)
    where_raw = _slice_fb_clause(where_tok_i)
    where_clause = where_raw if where_raw else None
    group_raw = _slice_fb_clause(group_tok_i)
    group_by_cols = [g.strip() for g in _split_top_level(group_raw, ",", dialect=dialect) if g.strip()] if group_raw else []
    having_raw = _slice_fb_clause(having_tok_i)
    having_clause = having_raw if having_raw else None

    joins, has_subquery_in_from = _parse_from_and_joins(from_clause, cte_names, dialect=dialect)
    ref_tables = _extract_all_referenced_tables(raw_select, cte_names, known_tables=known_tables, dialect=dialect)
    cte_lower = {c.lower() for c in cte_names}
    if (
        not has_subquery_in_from
        and joins
        and joins[0].table_name.lower() not in cte_lower
        and not joins[0].table_name.startswith("__mv_")
        and (not known_tables or joins[0].table_name.lower() in known_tables)
    ):
        primary_t = joins[0].table_name
        ref_tables = [primary_t] + [t for t in ref_tables if t.lower() != primary_t.lower()]
        base_table = primary_t
    else:
        base_table = ref_tables[0] if ref_tables else (joins[0].table_name if joins else "unknown_table")

    return ViewSpec(
        view_name=view_name,
        raw_select_sql=raw_select,
        base_table=base_table,
        joins=tuple(joins),
        referenced_tables=tuple(ref_tables),
        select_items=tuple(select_items),
        where_clause=where_clause,
        group_by_cols=tuple(group_by_cols),
        having_clause=having_clause,
        has_cte=has_cte,
        has_window_func=has_window,
        has_compound_set_op=has_compound,
        has_top_level_distinct=has_top_distinct,
        has_limit_or_order=has_limit_or_order,
        has_subquery_in_from=has_subquery_in_from,
        user_indexes=tuple(extra_indexes),
    )


def parse_view_definitions(
    view_sql_or_list: Sequence[ViewSpec | str] | ViewSpec | str,
    known_tables: Optional[Set[str]] = None,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    dialect: Dialect = Dialect.SQLITE,
) -> List[ViewSpec]:
    """
    Parses one or more CREATE VIEW statements (and optional inline `CREATE [UNIQUE] INDEX`
    statements or programmatic `indexes`) into a list of `ViewSpec` objects.
    """
    raw_inputs: List[ViewSpec | str]
    if isinstance(view_sql_or_list, (str, ViewSpec)):
        raw_inputs = [view_sql_or_list]
    else:
        raw_inputs = list(view_sql_or_list)

    initial_specs: List[ViewSpec] = []
    pending_index_specs: List[ViewIndexSpec] = []

    for item in raw_inputs:
        if isinstance(item, ViewSpec):
            initial_specs.append(item)
            for av in item.additional_views:
                initial_specs.append(av)
            continue
        clean = _strip_sql_comments(item, dialect=dialect).strip()
        stmts = [st.strip() for st in _split_top_level(clean, ";", dialect=dialect) if st.strip()]
        if not stmts:
            raise ValueError(f"Invalid CREATE VIEW syntax: {item!r}")
        for st in stmts:
            st_ast = parse_sql_ast(st, dialect=dialect)
            if isinstance(st_ast, sqlglot_exp.Create) and (st_ast.kind or "").upper() == "VIEW":
                initial_specs.append(_parse_single_create_view_stmt(st, known_tables=known_tables, dialect=dialect))
            elif isinstance(st_ast, sqlglot_exp.Create) and (st_ast.kind or "").upper() == "INDEX":
                pending_index_specs.append(parse_view_index_spec(st, dialect=dialect))
            else:
                raise ValueError(f"Unsupported statement in view definition script: {st!r}")

    if not initial_specs:
        raise ValueError(f"No CREATE VIEW statement found in: {view_sql_or_list!r}")

    if indexes:
        default_v = initial_specs[0].view_name if len(initial_specs) == 1 else None
        for raw_idx in indexes:
            pending_index_specs.append(parse_view_index_spec(raw_idx, default_view_name=default_v, dialect=dialect))

    if not pending_index_specs:
        return initial_specs

    by_view_idx: Dict[int, List[ViewIndexSpec]] = {i: list(vs.user_indexes) for i, vs in enumerate(initial_specs)}
    for idx_spec in pending_index_specs:
        if idx_spec.target_view is None and len(initial_specs) == 1:
            by_view_idx[0].append(idx_spec)
            continue
        matched = False
        if idx_spec.target_view is not None:
            t_low = idx_spec.target_view.lower()
            for i, vs in enumerate(initial_specs):
                v_low = vs.view_name.lower()
                v_unpref = v_low[len(DEFAULT_OBJECT_PREFIX) :] if v_low.startswith(DEFAULT_OBJECT_PREFIX) else v_low
                if t_low in {
                    v_low,
                    v_unpref,
                    f"{DEFAULT_OBJECT_PREFIX}{v_unpref}",
                    f"{DEFAULT_OBJECT_PREFIX}store_{v_unpref}",
                    f"__mv_store_{v_unpref}",
                }:
                    by_view_idx[i].append(idx_spec)
                    matched = True
                    break
            if not matched:
                by_view_idx[0].append(idx_spec)
        else:
            for i, vs in enumerate(initial_specs):
                exposed_low = {it.alias.lower() for it in vs.select_items}
                cols_low = {
                    _unquote_ident(re.sub(r"\b(?:ASC|DESC|NULLS\s+FIRST|NULLS\s+LAST)\b", "", c, flags=re.IGNORECASE).strip(), dialect=dialect).lower()
                    for c in idx_spec.columns
                }
                if cols_low and cols_low.issubset(exposed_low):
                    by_view_idx[i].append(idx_spec)
                    matched = True
            if not matched:
                by_view_idx[0].append(idx_spec)

    out_specs: List[ViewSpec] = []
    for i, vs in enumerate(initial_specs):
        out_specs.append(
            ViewSpec(
                view_name=vs.view_name,
                raw_select_sql=vs.raw_select_sql,
                base_table=vs.base_table,
                joins=vs.joins,
                referenced_tables=vs.referenced_tables,
                select_items=vs.select_items,
                where_clause=vs.where_clause,
                group_by_cols=vs.group_by_cols,
                having_clause=vs.having_clause,
                has_cte=vs.has_cte,
                has_window_func=vs.has_window_func,
                has_compound_set_op=vs.has_compound_set_op,
                has_top_level_distinct=vs.has_top_level_distinct,
                has_limit_or_order=vs.has_limit_or_order,
                has_subquery_in_from=vs.has_subquery_in_from,
                user_indexes=tuple(by_view_idx[i]),
            )
        )
    return out_specs


def parse_view_definition(
    view_sql: str | ViewSpec,
    known_tables: Optional[Set[str]] = None,
    indexes: Optional[Sequence[ViewIndexSpec | str | Sequence[str] | Dict[str, Any]]] = None,
    dialect: Dialect = Dialect.SQLITE,
) -> ViewSpec:
    """
    Parses a single-view or multi-view SQL script into a `ViewSpec`.
    If the script contains multiple `CREATE VIEW` statements, the first `ViewSpec`
    carries the remaining views in `additional_views` so both `parse_view_definition`
    and `generate_materialized_view` handle multi-view `.sql` files seamlessly.
    """
    specs = parse_view_definitions(view_sql, known_tables=known_tables, indexes=indexes, dialect=dialect)
    if len(specs) == 1:
        return specs[0]
    first = specs[0]
    return ViewSpec(
        view_name=first.view_name,
        raw_select_sql=first.raw_select_sql,
        base_table=first.base_table,
        joins=first.joins,
        referenced_tables=first.referenced_tables,
        select_items=first.select_items,
        where_clause=first.where_clause,
        group_by_cols=first.group_by_cols,
        having_clause=first.having_clause,
        has_cte=first.has_cte,
        has_window_func=first.has_window_func,
        has_compound_set_op=first.has_compound_set_op,
        has_top_level_distinct=first.has_top_level_distinct,
        has_limit_or_order=first.has_limit_or_order,
        has_subquery_in_from=first.has_subquery_in_from,
        user_indexes=first.user_indexes,
        additional_views=tuple(specs[1:]),
    )


def _extract_created_object(stmt: str) -> Optional[Tuple[str, str]]:
    """Extracts object type ('trigger', 'view', 'table') and name created by a DDL statement."""
    cleaned = _strip_sql_comments(stmt).strip()
    ident_pat = r'(?:"(?:""|[^"])+"|`(?:``|[^`])+`|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_$]*)'
    qual_pat = rf"({ident_pat}(?:\.{ident_pat})?)"

    def _extract_leaf(raw_qual: str) -> str:
        # Take the unqualified object name when schema-qualified, preserving dots
        # inside a quoted identifier such as `"my.view"`.
        parts = re.findall(ident_pat, raw_qual)
        leaf = parts[-1] if parts else raw_qual
        if leaf.startswith('"') and leaf.endswith('"') and len(leaf) >= 2:
            return leaf[1:-1].replace('""', '"')
        if leaf.startswith("`") and leaf.endswith("`") and len(leaf) >= 2:
            return leaf[1:-1].replace("``", "`")
        if leaf.startswith("[") and leaf.endswith("]") and len(leaf) >= 2:
            return leaf[1:-1]
        return leaf

    m_trig = re.match(
        rf"^CREATE(?:\s+OR\s+REPLACE)?(?:\s+DEFINER\s*=\s*\S+)?\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?{qual_pat}",
        cleaned,
        re.IGNORECASE,
    )
    if m_trig:
        return ("trigger", _extract_leaf(m_trig.group(1)))

    m_view = re.match(
        rf"^CREATE(?:\s+OR\s+REPLACE)?(?:\s+ALGORITHM\s*=\s*\w+)?(?:\s+DEFINER\s*=\s*\S+)?\s+VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?{qual_pat}",
        cleaned,
        re.IGNORECASE,
    )
    if m_view:
        return ("view", _extract_leaf(m_view.group(1)))

    m_tbl = re.match(
        rf"^CREATE(?:\s+TEMPORARY)?\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{qual_pat}",
        cleaned,
        re.IGNORECASE,
    )
    if m_tbl:
        return ("table", _extract_leaf(m_tbl.group(1)))

    return None

