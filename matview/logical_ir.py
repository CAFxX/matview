"""Engine-independent immutable schemas, plans, diffs, and typed CompilationContext/LogicalIR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .engines import (
    AggFunc,
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    MaintenanceStrategy,
    TransparencyLimitation,
    parse_engine_version,
)

@dataclass(frozen=True)
class ColumnDef:
    name: str
    data_type: str
    not_null: bool = False
    primary_key: bool = False
    # A generated (computed) column changes without being named in an
    # UPDATE's SET list. Set by DDL parsing and PostgreSQL catalog introspection.
    generated: bool = False
    # PostgreSQL 18+ VIRTUAL generated column: cannot be read directly from
    # statement-trigger transition tables, so triggers compute it from
    # ``generation_expr`` over the non-virtual transition columns.
    generated_virtual: bool = False
    generation_expr: Optional[str] = None
    # True only for a SQLite single-column INTEGER PRIMARY KEY (without DESC or
    # WITHOUT ROWID), which is SQLite's guaranteed 64-bit signed-integer rowid alias.
    is_rowid_alias: bool = False


def _strip_single_quoted_ident(s: str) -> Optional[str]:
    """If ``s`` is a single quoted SQL identifier, return its unescaped text; else None."""
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        if '"' not in inner.replace('""', ""):
            return inner.replace('""', '"')
    elif len(s) >= 2 and s[0] == "`" and s[-1] == "`":
        inner = s[1:-1]
        if "`" not in inner.replace("``", ""):
            return inner.replace("``", "`")
    elif len(s) >= 2 and s[0] == "[" and s[-1] == "]" and "]" not in s[1:-1]:
        return s[1:-1]
    return None


def _split_top_level_dots_simple(s: str) -> List[str]:
    """Split ``s`` on '.' outside quotes and parentheses."""
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    quote: Optional[str] = None
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if quote is not None:
            cur.append(ch)
            if quote == "[" and ch == "]":
                quote = None
            elif ch == quote:
                if i + 1 < n and s[i + 1] == quote:
                    cur.append(s[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue
        if ch in ('"', "`", "["):
            quote = ch
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")" and depth > 0:
            depth -= 1
            cur.append(ch)
        elif ch == "." and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur).strip())
    return parts


def _normalize_index_column_key(
    col_spec: str,
    known_columns_lower: Optional[frozenset[str]] = None,
) -> str:
    """Normalize a column or key specification for prefix-coverage comparison."""
    s = str(col_spec or "").strip()
    if known_columns_lower and s.lower() in known_columns_lower:
        return s.lower()
    single = _strip_single_quoted_ident(s)
    if single is not None:
        return single.strip().lower()
    s = re.sub(r"\s+NULLS\s+(?:FIRST|LAST)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(?:ASC|DESC)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(
        r'\s+COLLATE\s+(?:"(?:""|[^"])+"|`(?:``|[^`])+`|\S+)\s*$',
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = s.strip()
    if known_columns_lower and s.lower() in known_columns_lower:
        return s.lower()
    single2 = _strip_single_quoted_ident(s)
    if single2 is not None:
        return single2.strip().lower()
    if "." in s and not s.startswith("("):
        parts = _split_top_level_dots_simple(s)
        if len(parts) > 1 and parts[-1]:
            s = parts[-1]
            single3 = _strip_single_quoted_ident(s)
            if single3 is not None:
                return single3.strip().lower()
    return s.strip().lower()


@dataclass(frozen=True)
class TableIndexDef:
    name: str
    table_name: str
    columns: Tuple[str, ...]
    unique: bool = False
    where_clause: Optional[str] = None
    is_primary: bool = False


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: Dict[str, ColumnDef]
    fk_cascade_actions: Tuple[str, ...] = ()
    has_on_conflict_replace: bool = False
    indexes: Tuple[TableIndexDef, ...] = ()
    pk_columns: Tuple[str, ...] = ()
    without_rowid: bool = False
    strict: bool = False

    def effective_pk_columns(self) -> Tuple[str, ...]:
        if self.pk_columns:
            return self.pk_columns
        pks = tuple(c.name for c in self.columns.values() if c.primary_key)
        return pks if len(pks) == 1 else ()

    def covering_indexes_for(
        self,
        cols: Sequence[str],
        *,
        partial_ok: bool = False,
    ) -> Tuple[str, ...]:
        """Return existing index / PRIMARY KEY names whose leading columns cover `cols`."""
        known_cols = frozenset(c.lower() for c in self.columns)
        req = [_normalize_index_column_key(c, known_cols) for c in cols if str(c).strip()]
        if not req:
            return ()
        matches: List[str] = []
        pk_cols = [_normalize_index_column_key(c, known_cols) for c in self.effective_pk_columns()]
        if pk_cols and len(pk_cols) >= len(req) and pk_cols[: len(req)] == req:
            matches.append("PRIMARY KEY")
        for idx in self.indexes:
            if idx.where_clause and not partial_ok:
                continue
            idx_cols = [_normalize_index_column_key(c, known_cols) for c in idx.columns]
            if len(idx_cols) >= len(req) and idx_cols[: len(req)] == req:
                if idx.name not in matches:
                    matches.append(idx.name)
        return tuple(matches)

    def has_index_covering(
        self,
        cols: Sequence[str],
        *,
        partial_ok: bool = False,
    ) -> bool:
        return bool(self.covering_indexes_for(cols, partial_ok=partial_ok))


@dataclass(frozen=True)
class SelectItem:
    raw_expr: str
    alias: str
    is_aggregate: bool
    agg_func: Optional[AggFunc] = None
    arg_expr: Optional[str] = None
    extra_args: Optional[str] = None
    is_distinct: bool = False
    filter_where: Optional[str] = None
    source_col: Optional[str] = None


@dataclass(frozen=True)
class JoinSpec:
    join_type: str  # "FROM", "INNER JOIN", "LEFT JOIN", "CROSS JOIN"
    table_name: str
    alias: str
    on_condition: Optional[str] = None


@dataclass(frozen=True)
class ViewIndexSpec:
    columns: Tuple[str, ...]
    name: Optional[str] = None
    unique: bool = False
    where_clause: Optional[str] = None
    using_method: Optional[str] = None
    target_view: Optional[str] = None


@dataclass(frozen=True)
class ViewSpec:
    view_name: str
    raw_select_sql: str
    base_table: str
    joins: Tuple[JoinSpec, ...]
    referenced_tables: Tuple[str, ...]
    select_items: Tuple[SelectItem, ...]
    where_clause: Optional[str] = None
    group_by_cols: Tuple[str, ...] = ()
    having_clause: Optional[str] = None
    has_cte: bool = False
    has_window_func: bool = False
    has_compound_set_op: bool = False
    has_top_level_distinct: bool = False
    has_limit_or_order: bool = False
    has_subquery_in_from: bool = False
    user_indexes: Tuple[ViewIndexSpec, ...] = ()
    additional_views: Tuple["ViewSpec", ...] = ()
    # Original user SELECT SQL before plain-view inlining (None when no plain
    # SQL views were inlined).
    original_select_sql: Optional[str] = None
    # Sorted (plain_view_name, normalized_view_def_sql) pairs transitively
    # inlined into raw_select_sql, included in drift signatures.
    referenced_plain_views: Tuple[Tuple[str, str], ...] = ()
    # (UPPER function name, native result type or None) of user-defined
    # aggregates classified through the engine catalog (matview.native_functions).
    native_aggregates: Tuple[Tuple[str, Optional[str]], ...] = ()
    # ("UPPER_NAME/arity", native result type) of scalar functions sqlglot
    # does not model, from the same catalog lookup.
    native_scalar_types: Tuple[Tuple[str, str], ...] = ()
    # (lower-case output column, native result type) reported by the server
    # for select items that call functions sqlglot does not model.
    native_result_types: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class IncrementalBackfillSpec:
    mode: str  # "pk_range_fold", "group_key_batch", "offset_batch"
    pk_column: Optional[str] = None
    chunk_sql_templates: List[str] = field(default_factory=list)
    distinct_group_query: Optional[str] = None
    group_insert_template: Optional[str] = None
    offset_insert_template: Optional[str] = None


@dataclass(frozen=True)
class PlanDiagnostics:
    strategy: str
    tier: int
    is_incremental: bool
    selection_reason: str
    fallback_reason: Optional[str] = None
    algebraic_mode: Optional[str] = None
    engine_limitations: List[Dict[str, Any]] = field(default_factory=list)
    backing_table: str = ""
    auxiliary_tables: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    reader_view: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "tier": self.tier,
            "is_incremental": self.is_incremental,
            "selection_reason": self.selection_reason,
            "fallback_reason": self.fallback_reason,
            "algebraic_mode": self.algebraic_mode,
            "engine_limitations": list(self.engine_limitations),
            "backing_table": self.backing_table,
            "auxiliary_tables": list(self.auxiliary_tables),
            "triggers": list(self.triggers),
            "indexes": list(self.indexes),
            "reader_view": self.reader_view,
            "details": dict(self.details),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())


@dataclass(frozen=True)
class MaterializedViewPlan:
    dialect: Dialect
    strategy: MaintenanceStrategy
    view_spec: ViewSpec
    backing_table_name: str
    backing_table_ddl: str
    auxiliary_table_ddls: List[str]
    index_ddls: List[str]
    initial_populate_sqls: List[str]
    trigger_ddls: List[str]
    readonly_trigger_ddls: List[str] = field(default_factory=list)
    user_columns: List[str] = field(default_factory=list)
    reader_view_ddl: Optional[str] = None
    incremental_backfill_spec: Optional[IncrementalBackfillSpec] = None
    auxiliary_table_names: List[str] = field(default_factory=list)
    trigger_names: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    limitations: Sequence[TransparencyLimitation] = field(default_factory=list)
    object_prefix: str = DEFAULT_OBJECT_PREFIX
    db_view_name: str = ""
    index_names: List[str] = field(default_factory=list)
    user_indexes: List[ViewIndexSpec] = field(default_factory=list)
    sub_plans: List["MaterializedViewPlan"] = field(default_factory=list)
    incremental_backfill: bool = False
    batch_size: int = 1000
    full_populate_sqls: List[str] = field(default_factory=list)
    incremental_populate_sqls: List[str] = field(default_factory=list)
    engine_version: Optional[EngineVersion] = None
    diagnostics: Optional[PlanDiagnostics] = None
    recommended_base_indexes: List[str] = field(default_factory=list)
    base_index_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    # matview.diagnostics.Diagnostic objects: the maintenance strategy chosen
    # where the fast exact algebraic maintenance does not apply.
    strategy_diagnostics: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.diagnostics is None:
            object.__setattr__(self, "diagnostics", self._build_default_diagnostics())

    def _build_default_diagnostics(self) -> PlanDiagnostics:
        tier = (
            1 if self.strategy == MaintenanceStrategy.ALGEBRAIC_DELTA
            else 2 if self.strategy == MaintenanceStrategy.GROUP_SCOPED_DELTA
            else 3
        )
        is_inc = tier in (1, 2)
        sel_reason = f"Tier {tier} {self.strategy.value} maintenance."
        fb_reason = None
        if tier == 2:
            fb_reason = "Holistic aggregate, group structure, or dialect constraints require Tier 2 group-scoped recomputation."
        elif tier == 3:
            fb_reason = "Query complexity (e.g. window function, CTE, or subqueries) requires Tier 3 atomic refresh."
        return PlanDiagnostics(
            strategy=self.strategy.value,
            tier=tier,
            is_incremental=is_inc,
            selection_reason=sel_reason,
            fallback_reason=fb_reason,
            algebraic_mode=None,
            engine_limitations=[
                {
                    "code": lim.code,
                    "description": lim.description,
                    "severity": lim.severity.value,
                }
                for lim in self.limitations
            ],
            backing_table=self.backing_table_name,
            auxiliary_tables=list(self.auxiliary_table_names),
            triggers=list(self.trigger_names),
            indexes=list(self.index_names),
            reader_view=self.logical_view_name,
            details={
                "dialect": self.dialect.value,
                "engine_version": self.engine_version.label if self.engine_version else None,
            },
        )

    @property
    def subdialect(self) -> EngineSubdialect:
        if self.engine_version is not None:
            return self.engine_version.subdialect
        return EngineSubdialect(self.dialect.value)

    @property
    def view_name(self) -> str:
        # The SQL view name is user-facing; object_prefix is reserved for
        # catalog, backing, trigger, and auxiliary implementation objects.
        return self.logical_view_name

    @property
    def logical_view_name(self) -> str:
        raw = self.view_spec.view_name
        if self.object_prefix and raw.startswith(self.object_prefix):
            return raw[len(self.object_prefix) :]
        return raw

    @property
    def initial_populate_sql(self) -> str:
        return "\n\n".join(self.initial_populate_sqls)

    @property
    def has_watermark_triggers(self) -> bool:
        if self.sub_plans:
            return all(sp.has_watermark_triggers for sp in self.sub_plans)
        return any("backfill_active" in trig for trig in self.trigger_ddls)

    def get_populate_statements(
        self,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        use_inc = self.incremental_backfill if incremental_backfill is None else incremental_backfill
        eff_batch = self.batch_size if batch_size is None else batch_size
        if use_inc and self.incremental_populate_sqls:
            if eff_batch != self.batch_size and self.batch_size > 0:
                old_b = re.escape(str(self.batch_size))
                new_b = str(eff_batch)
                out_inc: List[str] = []
                for s in self.incremental_populate_sqls:
                    s_mod = re.sub(
                        rf"(__mv_batch_size\s+BIGINT\s*(?:DEFAULT|:=)\s*){old_b}\b",
                        rf"\g<1>{new_b}",
                        s,
                        flags=re.IGNORECASE,
                    )
                    s_mod = re.sub(
                        rf"(\(\(ROW_NUMBER\(\)\s+OVER\s*\(\)\s*-\s*1\)\s*/\s*){old_b}(\)\s+AS\s+__mv_batch_id)",
                        rf"\g<1>{new_b}\g<2>",
                        s_mod,
                        flags=re.IGNORECASE,
                    )
                    out_inc.append(s_mod)
                return out_inc
            return list(self.incremental_populate_sqls)
        if not use_inc and self.full_populate_sqls:
            return list(self.full_populate_sqls)
        return list(self.initial_populate_sqls)

    def create_structure_statements(self) -> List[str]:
        """Returns DDLs to create auxiliary tables, backing table, indexes, and read-only view."""
        if self.sub_plans:
            stmts: List[str] = []
            for sp in self.sub_plans:
                stmts.extend(sp.create_structure_statements())
            return stmts
        stmts = []
        stmts.extend(self.auxiliary_table_ddls)
        stmts.append(self.backing_table_ddl)
        stmts.extend(self.index_ddls)
        if self.reader_view_ddl:
            stmts.append(self.reader_view_ddl)
        stmts.extend(self.readonly_trigger_ddls)
        return stmts

    def all_statements(
        self,
        include_populate: bool = True,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Returns all setup statements in concurrency-safe order:
          1. Backing & auxiliary tables, indexes, read-only user view & write-protection triggers
          2. Atomic maintenance triggers on base tables (installed BEFORE backfill so no concurrent
             base-table DML can be missed while backfill is running)
          3. Initial backfill (single-shot or chunked incremental backfill)
        """
        if self.sub_plans:
            stmts: List[str] = []
            for sp in self.sub_plans:
                stmts.extend(
                    sp.all_statements(
                        include_populate=include_populate,
                        incremental_backfill=incremental_backfill,
                        batch_size=batch_size,
                    )
                )
            return stmts
        stmts = self.create_structure_statements()
        stmts.extend(self.trigger_ddls)
        if include_populate:
            stmts.extend(
                self.get_populate_statements(
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                )
            )
        return stmts

    def drop_statements(self) -> List[str]:
        """Returns idempotent DROP statements for triggers, functions, view, and backing tables."""
        from .sql_frontend import build_drop_sql

        if self.sub_plans:
            stmts: List[str] = []
            for sp in self.sub_plans:
                stmts.extend(sp.drop_statements())
            return stmts
        stmts = []
        if self.dialect != Dialect.POSTGRES:
            for t_name in self.trigger_names:
                stmts.append(build_drop_sql("TRIGGER", t_name, dialect=self.dialect, allow_qualified=False))
        for fn_name in self.function_names:
            stmts.append(f"DROP FUNCTION IF EXISTS {fn_name if '(' in fn_name else fn_name + '()'} CASCADE;")
        stmts.append(build_drop_sql("VIEW", self.view_name, dialect=self.dialect, allow_qualified=False))
        stmts.append(build_drop_sql("TABLE", self.backing_table_name, dialect=self.dialect, allow_qualified=False))
        for aux_t in self.auxiliary_table_names:
            stmts.append(build_drop_sql("TABLE", aux_t, dialect=self.dialect, allow_qualified=False))
        return stmts

    def render_sql_script(
        self,
        include_populate: bool = True,
        incremental_backfill: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> str:
        if self.sub_plans:
            return "\n".join(
                sp.render_sql_script(
                    include_populate=include_populate,
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                )
                for sp in self.sub_plans
            )
        use_inc = self.incremental_backfill if incremental_backfill is None else incremental_backfill
        eff_batch = self.batch_size if batch_size is None else batch_size
        pop_stmts = self.get_populate_statements(
            incremental_backfill=use_inc,
            batch_size=eff_batch,
        )
        bf_label = f"incremental (batch_size={eff_batch})" if use_inc else "full"
        eng_label = f" | Engine: {self.engine_version.label}" if self.engine_version else ""
        lines: List[str] = [
            "-- =============================================================================",
            f"-- Incremental Materialized View: {self.view_name} (logical: {self.logical_view_name})",
            f"-- Backing Table: {self.backing_table_name}",
            f"-- Base Tables: {', '.join(self.view_spec.referenced_tables)} | Dialect: {self.dialect.value}{eng_label} | Strategy: {self.strategy.value} | Backfill: {bf_label}",
            "-- =============================================================================",
        ]
        if self.limitations:
            lines.append("-- WARNING (Ignored Engine/View Limitations):")
            for lim in self.limitations:
                lines.append(f"--   * [{lim.code}] {lim.description}")
        lines.append("")
        lines.append("-- 1. Internal Backing Table")
        for aux in self.auxiliary_table_ddls:
            lines.append(aux)
        lines.append(self.backing_table_ddl)
        lines.append("")
        if self.index_ddls:
            lines.append("-- 2. Indexes")
            for idx_ddl in self.index_ddls:
                lines.append(idx_ddl)
            lines.append("")
        if self.reader_view_ddl or self.readonly_trigger_ddls:
            lines.append("-- 3. Read-Only User View & Write-Protection Triggers")
            if self.reader_view_ddl:
                lines.append(self.reader_view_ddl)
            for rt in self.readonly_trigger_ddls:
                lines.append(rt)
            lines.append("")
        lines.append("-- 4. Atomic Maintenance Triggers (Installed Before Backfill)")
        for trig in self.trigger_ddls:
            lines.append(trig)
            lines.append("")
        if include_populate and pop_stmts:
            lines.append(
                f"-- 5. Initial Backfill ({'Incremental Batched' if use_inc else 'Full'})"
            )
            for pop in pop_stmts:
                lines.append(pop)
            lines.append("")
        return "\n".join(lines)


@dataclass
class ViewDiff:
    view_name: str
    status: str  # "CREATE", "REPLACE", "ALTER", "DROP", "NOOP"
    has_changes: bool
    strategy: Optional[str]
    current_strategy: Optional[str]
    signature_hash: Optional[str]
    current_signature_hash: Optional[str]
    diagnostics: Optional[PlanDiagnostics]
    pending_ddls: List[str]
    pending_changes: Dict[str, Any]
    summary: str
    plan: Optional[MaterializedViewPlan] = None
    action: str = ""

    def __post_init__(self) -> None:
        if not self.action:
            self.action = self.status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view_name": self.view_name,
            "status": self.status,
            "action": self.action,
            "has_changes": self.has_changes,
            "strategy": self.strategy,
            "current_strategy": self.current_strategy,
            "signature_hash": self.signature_hash,
            "current_signature_hash": self.current_signature_hash,
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
            "pending_changes": self.pending_changes,
            "pending_ddls": list(self.pending_ddls),
            "summary": self.summary,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())

    def render_text(self) -> str:
        lines: List[str] = [f"[{self.status}] {self.view_name}"]
        if self.diagnostics:
            tier_str = f"Tier {self.diagnostics.tier}"
            mode_str = f" ({self.diagnostics.algebraic_mode})" if self.diagnostics.algebraic_mode else ""
            lines.append(f"  Strategy: {self.diagnostics.strategy} ({tier_str}{mode_str})")
            if self.diagnostics.selection_reason:
                lines.append(f"  Selection Reason: {self.diagnostics.selection_reason}")
            if self.diagnostics.fallback_reason:
                lines.append(f"  Fallback Reason:  {self.diagnostics.fallback_reason}")
            if self.diagnostics.engine_limitations:
                lines.append(f"  Limitations ({len(self.diagnostics.engine_limitations)}):")
                for lim in self.diagnostics.engine_limitations:
                    lines.append(f"    * [{lim['code']}] {lim['description']}")
            perf_notes = list((self.diagnostics.details or {}).get("performance_notes") or ())
            if perf_notes:
                lines.append(f"  Performance Notes ({len(perf_notes)}):")
                for note in perf_notes:
                    lines.append(f"    * {note}")

        if self.status == "NOOP":
            lines.append("  Status: Up to date; no pending changes.")
            return "\n".join(lines)

        lines.append(f"  Summary: {self.summary}")
        lines.append("  Pending Changes:")
        bt = self.pending_changes.get("backing_table", {})
        if bt:
            act = bt.get("action", "")
            lines.append(f"    Backing Table: {bt.get('name')} [{act}]")
        rv = self.pending_changes.get("reader_view", {})
        if rv:
            act = rv.get("action", "")
            lines.append(f"    Reader View:   {rv.get('name')} [{act}]")
        trigs = self.pending_changes.get("triggers", {})
        if trigs.get("to_create"):
            lines.append(f"    Triggers to Create ({len(trigs['to_create'])}):")
            for t in trigs["to_create"]:
                lines.append(f"      + {t}")
        if trigs.get("to_drop"):
            lines.append(f"    Triggers to Drop ({len(trigs['to_drop'])}):")
            for t in trigs["to_drop"]:
                lines.append(f"      - {t}")
        idxs = self.pending_changes.get("indexes", {})
        if idxs.get("to_create"):
            lines.append(f"    Indexes to Create ({len(idxs['to_create'])}):")
            for idx in idxs["to_create"]:
                lines.append(f"      + {idx}")
        if idxs.get("to_drop"):
            lines.append(f"    Indexes to Drop ({len(idxs['to_drop'])}):")
            for idx in idxs["to_drop"]:
                lines.append(f"      - {idx}")
        if self.pending_ddls:
            lines.append(f"  Pending DDL Statements: {len(self.pending_ddls)} statement(s)")
        return "\n".join(lines)


class ViewDiffReport(dict):
    """
    Mapping of view_name -> ViewDiff, with helper methods for inspection,
    JSON serialization, and text rendering.
    """

    def __init__(self, diffs: Dict[str, ViewDiff]) -> None:
        super().__init__(diffs)

    @property
    def has_changes(self) -> bool:
        return any(d.has_changes for d in self.values())

    @property
    def counts(self) -> Dict[str, int]:
        c = {"CREATE": 0, "REPLACE": 0, "ALTER": 0, "DROP": 0, "NOOP": 0}
        for d in self.values():
            st = d.status.upper()
            c[st] = c.get(st, 0) + 1
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {
            "views": {name: diff.to_dict() for name, diff in self.items()},
            "summary": {
                "total_views": len(self),
                "has_changes": self.has_changes,
                "counts": self.counts,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render_text(self) -> str:
        lines: List[str] = [
            f"=== Matview Diff: {len(self)} view(s) inspected ===",
            "",
        ]
        for name, diff in self.items():
            lines.append(diff.render_text())
            lines.append("")
        lines.append("-" * 78)
        c = self.counts
        changed = sum(1 for d in self.values() if d.has_changes)
        lines.append(
            f"Summary: {changed} view(s) with pending changes "
            f"({c['CREATE']} CREATE, {c['REPLACE']} REPLACE, {c['ALTER']} ALTER, {c['DROP']} DROP, {c['NOOP']} NOOP)"
        )
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Typed Immutable Compilation Intermediate Representation
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalIR:
    """Engine-independent, immutable relational & aggregate representation."""

    raw_view: ViewSpec
    view: ViewSpec
    logical_view_name: str
    db_view_name: str
    primary_schema: Optional[TableSchema]
    working_select_items: Tuple[SelectItem, ...]
    compound_reader_exprs: Tuple[Tuple[str, str], ...]
    extra_having_select_items: Tuple[SelectItem, ...]
    rewritten_having_clause: Optional[str]
    decomposed_ok: bool
    effective_group_by_exprs: Tuple[str, ...]
    user_indexes: Tuple[ViewIndexSpec, ...] = ()
    setop_pk_cols: Tuple[str, ...] = ()
    union_all_inner_sql: Optional[str] = None
    union_all_sub_alias: str = "__mv_u"
    union_all_branch_specs: Tuple[ViewSpec, ...] = ()
    uall_has_nullable_grp: Optional[bool] = None
    uall_orig_grp_exprs: Optional[Tuple[str, ...]] = None
    uall_grp_alias_map: Optional[Mapping[str, str]] = None
    uall_orig_item_map: Optional[Mapping[int, SelectItem]] = None

    @property
    def compound_reader_exprs_dict(self) -> Dict[str, str]:
        return dict(self.compound_reader_exprs)

    def with_updates(self, **changes: Any) -> "LogicalIR":
        return replace(self, **changes)


@dataclass(frozen=True)
class CompilationContext:
    """Immutable compilation context threaded between planner, rules, and codegen."""

    schemas: Mapping[str, TableSchema]
    schemas_lower: Mapping[str, TableSchema]
    engine_version: EngineVersion
    dialect: Dialect
    subdialect: EngineSubdialect
    ignore_limitations: bool
    object_prefix: str
    incremental_backfill: bool
    batch_size: int
    strict: bool
    watermark_catalog_table: Optional[str]
    pg_fillfactor: Optional[int]
    limitations: Tuple[TransparencyLimitation, ...]
    strategy: MaintenanceStrategy
    alg_mode: str
    ir: LogicalIR
    # Limitation codes explicitly accepted by the caller (accept_limitations).
    accepted_limitations: Tuple[str, ...] = ()
    # PostgreSQL catalog table used to fence pre-repair REPEATABLE READ /
    # SERIALIZABLE snapshots inside statement-level maintenance triggers.
    repair_fence_catalog_table: Optional[str] = None

    def with_ir(self, **ir_changes: Any) -> "CompilationContext":
        return replace(self, ir=self.ir.with_updates(**ir_changes))

    def with_updates(self, **changes: Any) -> "CompilationContext":
        return replace(self, **changes)
