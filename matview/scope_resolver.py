"""One scope resolver for the exact floating-point refresh default.

Under the exact default (see planner._float_refresh_required) the view is
maintained by a full refresh of its ORIGINAL query, so the stored column types
must equal the types the engine gives the original query. They are derived
from declared base-table column types; that is only sound when every column
an aggregate (or window function) reads is PROVABLY a declared base-table
column of a known type. This module is the single place that decides it, for
every nesting level (top-level projection, windows, derived tables, CTEs,
LATERAL, scalar/correlated/EXISTS/IN subqueries, set operations) and every
join form (ON, USING, NATURAL):

* ``BASE``: the reference resolves, through its own scope chain, to exactly one
  declared base-table column (also through a derived table or CTE column that
  is a bare pass-through of such a column, and through a column that several
  USING/NATURAL sources define with the IDENTICAL type).
* ``CONFLICT``: the reference can denote columns of different declared types
  (a USING/NATURAL-merged column whose sources differ, an ambiguous column).
* ``UNKNOWN``: anything else (expression-valued derived/CTE column, recursive
  CTE, set operation, ``SELECT *``, table function, window result, unresolved
  name). The caller fails closed.

No type is ever guessed from an inner query's expressions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from sqlglot import exp as sqlglot_exp

from .logical_ir import ColumnDef

BASE = "base"
UNKNOWN = "unknown"
CONFLICT = "conflict"


@dataclass(frozen=True)
class Resolution:
    status: str
    column: Optional[ColumnDef] = None
    table: Optional[str] = None
    reason: str = ""
    # The defining expression of an expression-valued derived-table/CTE column
    # (UNKNOWN: its type is not derived here, but its leaf columns can be).
    expr: Any = None

    @property
    def ok(self) -> bool:
        return self.status == BASE


@dataclass
class FloatAuditNode:
    """One aggregate/window function of the query with the resolver's verdict."""

    site: Any  # the AST node: the call, or its Window
    fn_node: Any  # the call itself
    sql: str  # the call (with its OVER clause for a window)
    fn_sql: str  # the function call alone
    is_window: bool
    reaches: bool  # its value can flow into a stored column
    final: bool  # directly in the projection of the outermost SELECT / a top-level set-operation branch
    status: str  # BASE | UNKNOWN | CONFLICT for the argument columns
    per: List[Tuple[Any, Resolution]]
    reason: str
    order_sensitive: bool  # SUM/AVG/stat whose result depends on the row order
    in_derived: bool = False  # inside the body of a derived table, CTE or LATERAL
    derived_name: Optional[str] = None  # output column of the derived table/CTE holding the call
    compound_expr: Any = None  # the whole projection item when this call is part of a larger expression


def norm_type(data_type: str) -> str:
    return re.sub(r"\s+", " ", data_type.strip().upper())


@dataclass
class _Source:
    names: List[str]
    kind: str  # "base" | "cte" | "derived" | "other"
    table: Optional[str] = None
    # Output column names when known (derived table / CTE / aliased function).
    cols: Optional[List[str]] = None
    body: Any = None  # the Select a pass-through column is resolved in
    branches: Optional[List[Any]] = None  # the Selects of a set-operation derived table


def _src_alias(node: Any) -> str:
    alias = node.args.get("alias") if hasattr(node, "args") else None
    return (getattr(alias, "name", "") or "").lower() if alias is not None else ""


class ScopeResolver:
    def __init__(
        self,
        root: Any,
        schemas_lower: Mapping[str, Any],
        common_type: Optional[Callable[[List[str], bool], Optional[str]]] = None,
        literal_type: Optional[Callable[[Any], str]] = None,
        expr_type: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> None:
        self.root = root
        self.schemas = schemas_lower
        # The engine's UNION result-type rule over declared column types
        # (None: a set-operation column is never provable).
        self.common_type = common_type
        self.literal_type = literal_type  # the engine type of a numeric literal
        # The provable type of a branch expression (None: not provable).
        self.expr_type = expr_type
        self._ctes: Dict[str, Tuple[Any, bool]] = {}
        for cte in root.find_all(sqlglot_exp.CTE):
            if cte.alias:
                with_node = cte.parent
                recursive = isinstance(with_node, sqlglot_exp.With) and bool(with_node.args.get("recursive"))
                self._ctes[cte.alias.lower()] = (cte, recursive)
        self._sources_cache: Dict[int, List[_Source]] = {}
        self._merged_cache: Dict[int, Set[str]] = {}
        self._active: Set[int] = set()

    # ---- scope construction -------------------------------------------------

    @staticmethod
    def _projection_names(select: Any) -> Optional[List[str]]:
        names: List[str] = []
        for proj in select.expressions:
            if isinstance(proj, sqlglot_exp.Star) or (
                isinstance(proj, sqlglot_exp.Column) and isinstance(proj.this, sqlglot_exp.Star)
            ):
                return None
            name = proj.alias_or_name
            if not name:
                return None
            names.append(name.lower())
        return names

    def _body_source(self, body: Any, names: List[str], kind: str) -> _Source:
        while isinstance(body, sqlglot_exp.Subquery):
            body = body.this
        if isinstance(body, sqlglot_exp.Select):
            return _Source(names=names, kind=kind, cols=self._projection_names(body), body=body)
        # A set operation: its column names are those of the first branch and
        # its types are the branches' common type. Provable only when every
        # branch projects a bare column at that position (see _setop_column).
        branches: List[Any] = []

        def flatten(n: Any) -> bool:
            if isinstance(n, sqlglot_exp.SetOperation):
                return flatten(n.this) and flatten(n.expression)
            while isinstance(n, sqlglot_exp.Subquery):
                n = n.this
            if isinstance(n, sqlglot_exp.Select):
                branches.append(n)
                return True
            if isinstance(n, sqlglot_exp.SetOperation):
                return flatten(n)
            return False

        ok = flatten(body)
        cols = self._projection_names(branches[0]) if ok and branches else None
        return _Source(names=names, kind=kind, cols=cols, body=None, branches=branches if ok else None)

    def _make_source(self, src: Any) -> Optional[_Source]:
        if isinstance(src, sqlglot_exp.Table):
            if not src.name:
                return None
            tname = src.name.lower()
            alias = (src.alias or "").lower()
            names = [alias] if alias else [tname]
            if not src.args.get("db") and tname in self._ctes:
                cte, recursive = self._ctes[tname]
                tbl_alias = cte.args.get("alias")
                renamed = isinstance(tbl_alias, sqlglot_exp.TableAlias) and bool(tbl_alias.columns)
                if recursive or renamed:
                    cols = [c.name.lower() for c in tbl_alias.columns] if renamed else None
                    return _Source(names=names, kind="cte", cols=cols, body=None)
                return self._body_source(cte.this, names, "cte")
            if tname in self.schemas:
                return _Source(names=names, kind="base", table=tname)
            return _Source(names=names, kind="other")
        if isinstance(src, sqlglot_exp.Subquery):
            alias = _src_alias(src)
            return self._body_source(src.this, [alias] if alias else [], "derived")
        if isinstance(src, sqlglot_exp.Lateral) and isinstance(src.this, sqlglot_exp.Subquery):
            alias = _src_alias(src)
            return self._body_source(src.this.this, [alias] if alias else [], "derived")
        alias = _src_alias(src)
        return _Source(names=[alias] if alias else [], kind="other")

    def _sources(self, select: Any) -> List[_Source]:
        key = id(select)
        cached = self._sources_cache.get(key)
        if cached is not None:
            return cached
        out: List[_Source] = []
        from_clause = select.args.get("from_") or select.args.get("from")
        srcs: List[Any] = [getattr(from_clause, "this", None)] if from_clause is not None else []
        srcs += [getattr(j, "this", None) for j in (select.args.get("joins") or ())]
        for src in srcs:
            if src is None:
                continue
            made = self._make_source(src)
            if made is not None:
                out.append(made)
        self._sources_cache[key] = out
        return out

    # ---- resolution ---------------------------------------------------------

    def _base_column(self, table: str, name: str) -> Optional[ColumnDef]:
        ts = self.schemas.get(table)
        if ts is None:
            return None
        for c in ts.columns.values():
            if c.name.lower() == name:
                return c
        return None

    def _source_resolution(self, src: _Source, name: str, ref: Any) -> Resolution:
        if src.kind == "base":
            col = self._base_column(src.table or "", name)
            if col is None:
                return Resolution(UNKNOWN, reason=f"column {name} is not a declared column of {src.table}")
            return Resolution(BASE, col, src.table)
        if src.kind in ("derived", "cte") and src.branches:
            return self._setop_column(src, name)
        if src.kind in ("cte", "derived") and src.body is not None:
            return self._passthrough(src, name)
        if src.kind == "cte":
            return Resolution(UNKNOWN, reason=f"column {name} of a recursive CTE (or a CTE with a column list)")
        if src.kind in ("derived", "cte") and src.branches:
            return self._setop_column(src, name)
        if src.kind == "derived":
            return Resolution(UNKNOWN, reason=f"column {name} of a set-operation derived table")
        return Resolution(UNKNOWN, reason=f"column {name} of a table function or unsupported source")

    def _setop_column(self, src: _Source, name: str) -> Resolution:
        """Column of a UNION/UNION ALL/... derived table or CTE: BASE with the
        engine's common type when every branch projects a bare declared base
        column at that position, otherwise UNKNOWN."""
        if self.common_type is None or src.cols is None or name not in src.cols:
            return Resolution(UNKNOWN, reason=f"column {name} of a set-operation derived table")
        pos = src.cols.index(name)
        types: List[str] = []
        has_null = False
        for br in src.branches or []:
            if pos >= len(br.expressions):
                return Resolution(UNKNOWN, reason=f"column {name} of a set-operation derived table (arity)")
            proj = br.expressions[pos]
            inner = proj.this if isinstance(proj, sqlglot_exp.Alias) else proj
            while isinstance(inner, sqlglot_exp.Paren):
                inner = inner.this
            if isinstance(inner, sqlglot_exp.Null):
                has_null = True  # an untyped NULL takes the type of the other branches
                continue
            if isinstance(inner, sqlglot_exp.Literal) and not inner.args.get("is_string") and self.literal_type is not None:
                types.append(self.literal_type(inner))  # a numeric literal has its own engine type
                continue
            if not isinstance(inner, sqlglot_exp.Column) or isinstance(inner.this, sqlglot_exp.Star):
                et = self.expr_type(inner) if self.expr_type is not None else None
                if et:
                    types.append(et)
                    continue
                return Resolution(UNKNOWN, reason=f"column {name} of a set-operation derived table is an expression in a branch")
            r = self.resolve(inner)
            if not r.ok or r.column is None:
                return Resolution(UNKNOWN if r.status != CONFLICT else CONFLICT, reason=r.reason or f"a branch of column {name} is not a base column")
            types.append(r.column.data_type)
        common = self.common_type(types, has_null) if types else None
        if not common:
            return Resolution(UNKNOWN, reason=f"column {name} of a set-operation derived table has no provable common type")
        return Resolution(BASE, ColumnDef(name=name, data_type=common), None)

    def _passthrough(self, src: _Source, name: str) -> Resolution:
        body = src.body
        matches = [p for p in body.expressions if (p.alias_or_name or "").lower() == name]
        if len(matches) != 1:
            return Resolution(UNKNOWN, reason=f"derived column {name} is ambiguous or not projected")
        proj = matches[0]
        inner = proj.this if isinstance(proj, sqlglot_exp.Alias) else proj
        while isinstance(inner, sqlglot_exp.Paren):
            inner = inner.this
        if not isinstance(inner, sqlglot_exp.Column) or isinstance(inner.this, sqlglot_exp.Star):
            return Resolution(
                UNKNOWN,
                reason=f"derived table or CTE column {name} is an expression ({inner.sql()[:60]}), not a base column",
                expr=inner,
            )
        if id(inner) in self._active:
            return Resolution(UNKNOWN, reason=f"derived column {name} is defined recursively")
        self._active.add(id(inner))
        try:
            return self.resolve(inner)
        finally:
            self._active.discard(id(inner))

    def resolve(self, col: Any) -> Resolution:
        name = (col.name or "").lower()
        qual = (col.table or "").lower()
        if isinstance(col.this, sqlglot_exp.Star) or not name:
            return Resolution(UNKNOWN, reason="star reference")
        node = col.parent
        while node is not None:
            if isinstance(node, sqlglot_exp.Select):
                sources = self._sources(node)
                if qual:
                    hits = [s for s in sources if qual in s.names]
                    if len(hits) == 1:
                        return self._source_resolution(hits[0], name, col)
                    if len(hits) > 1:
                        return Resolution(UNKNOWN, reason=f"alias {qual} is used by several sources")
                else:
                    owners: List[_Source] = []
                    for s in sources:
                        if s.kind == "base":
                            if self._base_column(s.table or "", name) is not None:
                                owners.append(s)
                        elif s.cols is None or name in s.cols:
                            owners.append(s)
                    if len(owners) == 1:
                        return self._source_resolution(owners[0], name, col)
                    if len(owners) > 1:
                        return self._merged(owners, name, col)
            node = node.parent
        return Resolution(UNKNOWN, reason=f"column {name} is not defined by any enclosing scope")

    def _merged(self, owners: List[_Source], name: str, col: Any) -> Resolution:
        """An unqualified column defined by several sources of one scope
        (USING / NATURAL merged column): its type is provable only when every
        source defines it with the identical declared type."""
        res = [self._source_resolution(s, name, col) for s in owners]
        if all(r.ok for r in res):
            types = {norm_type(r.column.data_type) for r in res if r.column is not None}
            if len(types) == 1:
                return res[0]
            return Resolution(
                CONFLICT,
                reason=(
                    f"column {name} is defined by several joined sources with different types "
                    f"({', '.join(sorted(types))}): a USING/NATURAL-merged column gets the engine's "
                    f"common type, which is not derived here"
                ),
            )
        return Resolution(UNKNOWN, reason=f"column {name} is defined by several sources of unknown type")

    def leaf_columns(self, expr: Any, depth: int = 0) -> Optional[List[ColumnDef]]:
        """The declared base columns an expression ultimately reads, looking
        through expression-valued derived-table/CTE columns; None when any
        column cannot be traced to a declared base column (unprovable)."""
        if depth > 8 or expr is None:
            return None
        out: List[ColumnDef] = []
        for col in expr.find_all(sqlglot_exp.Column):
            r = self.resolve(col)
            if r.ok and r.column is not None:
                out.append(r.column)
            elif r.expr is not None:
                inner = self.leaf_columns(r.expr, depth + 1)
                if inner is None:
                    return None
                out += inner
            else:
                return None
        return out

    def resolve_expr(self, expr: Any) -> Tuple[str, List[Tuple[Any, Resolution]], str]:
        """Resolution of every column of ``expr``: (status, per-column list,
        reason). CONFLICT wins over UNKNOWN wins over BASE; an argument that
        contains a subquery is UNKNOWN."""
        if expr is None:
            return BASE, [], ""
        if expr.find(sqlglot_exp.Select) is not None or expr.find(sqlglot_exp.Subquery) is not None:
            return UNKNOWN, [], "a subquery inside the aggregate argument"
        per: List[Tuple[Any, Resolution]] = []
        status, reason = BASE, ""
        for col in expr.find_all(sqlglot_exp.Column):
            r = self.resolve(col)
            per.append((col, r))
            if r.status == CONFLICT:
                status, reason = CONFLICT, r.reason
            elif r.status == UNKNOWN and status != CONFLICT:
                status, reason = UNKNOWN, r.reason
        return status, per, reason
