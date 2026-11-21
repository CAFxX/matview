"""IncrementalViewCompiler wiring immutable CompilationContext/LogicalIR, planner, maintenance rules, and dialect codegen."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .dialects import DialectMixin
from .engines import (
    DEFAULT_OBJECT_PREFIX,
    Dialect,
    EngineSubdialect,
    EngineVersion,
    MaintenanceStrategy,
    StrictCompilationError,
    TransparencyLimitation,
    parse_engine_version,
)
from .logical_ir import (
    CompilationContext,
    LogicalIR,
    SelectItem,
    TableSchema,
    ViewIndexSpec,
    ViewSpec,
)
from .maintenance import MaintenanceMixin
from .native_functions import mark_native_aggregates
from .planner import PlannerMixin
from .planner_exact_routing import ExactAggregateRoutingMixin
from .sql_frontend import resolve_statistical_aggregate_aliases


class IncrementalViewCompiler(PlannerMixin, ExactAggregateRoutingMixin, MaintenanceMixin, DialectMixin):
    """
    Compiles a `ViewSpec` + table schemas + target `EngineVersion` into a
    `MaterializedViewPlan` via an immutable `CompilationContext` / `LogicalIR`.
    """

    @classmethod
    def _from_context(cls, context: CompilationContext) -> "IncrementalViewCompiler":
        obj = cls.__new__(cls)
        obj._context = context
        return obj

    def _with_context(self, context: CompilationContext) -> "IncrementalViewCompiler":
        return self._from_context(context)

    def with_ir(self, **ir_changes: Any) -> "IncrementalViewCompiler":
        """Return a derived compiler bound to an updated immutable `LogicalIR`."""
        return self._from_context(self._context.with_ir(**ir_changes))

    def __init__(
        self,
        schemas: Mapping[str, TableSchema],
        view_spec: ViewSpec,
        dialect: Dialect | str | EngineVersion = Dialect.SQLITE,
        ignore_limitations: bool = False,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        incremental_backfill: bool = False,
        batch_size: int = 1000,
        strict: bool = False,
        engine_version: Optional[EngineVersion | str] = None,
        watermark_catalog_table: Optional[str] = None,
        pg_fillfactor: Optional[int] = 80,
        accept_limitations: Sequence[str] = (),
    ) -> None:
        self._context: CompilationContext = self._build_compilation_context(
            schemas=schemas,
            view_spec=view_spec,
            dialect=dialect,
            ignore_limitations=ignore_limitations,
            object_prefix=object_prefix,
            incremental_backfill=incremental_backfill,
            batch_size=batch_size,
            strict=strict,
            engine_version=engine_version,
            watermark_catalog_table=watermark_catalog_table,
            pg_fillfactor=pg_fillfactor,
            accept_limitations=accept_limitations,
        )

    @classmethod
    def _build_compilation_context(
        cls,
        schemas: Mapping[str, TableSchema],
        view_spec: ViewSpec,
        dialect: Dialect | str | EngineVersion = Dialect.SQLITE,
        ignore_limitations: bool = False,
        object_prefix: str = DEFAULT_OBJECT_PREFIX,
        incremental_backfill: bool = False,
        batch_size: int = 1000,
        strict: bool = False,
        engine_version: Optional[EngineVersion | str] = None,
        watermark_catalog_table: Optional[str] = None,
        pg_fillfactor: Optional[int] = 80,
        accept_limitations: Sequence[str] = (),
    ) -> CompilationContext:
        schemas_ro = MappingProxyType(dict(schemas))
        schemas_lower_ro = MappingProxyType({k.lower(): v for k, v in schemas_ro.items()})

        if engine_version is not None:
            parsed_ev = parse_engine_version(
                engine_version,
                default_dialect=dialect if not isinstance(dialect, EngineVersion) else dialect.dialect,
            )
        else:
            parsed_ev = parse_engine_version(dialect)

        resolved_dialect: Dialect = parsed_ev.dialect
        resolved_subdialect: EngineSubdialect = parsed_ev.subdialect
        eff_batch = max(1, int(batch_size)) if batch_size else 1000

        view_spec = resolve_statistical_aggregate_aliases(view_spec, resolved_dialect)

        raw_vname = view_spec.view_name
        logical_view_name = (
            raw_vname[len(object_prefix) :]
            if object_prefix and raw_vname.startswith(object_prefix)
            else raw_vname
        )
        if not logical_view_name:
            raise ValueError("A materialized view must have a non-empty user-facing name.")
        db_view_name = logical_view_name

        base_ir = LogicalIR(
            raw_view=view_spec,
            view=view_spec,
            logical_view_name=logical_view_name,
            db_view_name=db_view_name,
            primary_schema=None,
            working_select_items=(),
            compound_reader_exprs=(),
            extra_having_select_items=(),
            rewritten_having_clause=None,
            decomposed_ok=False,
            effective_group_by_exprs=(),
            user_indexes=tuple(view_spec.user_indexes),
        )
        stage_ctx = CompilationContext(
            schemas=schemas_ro,
            schemas_lower=schemas_lower_ro,
            engine_version=parsed_ev,
            dialect=resolved_dialect,
            subdialect=resolved_subdialect,
            ignore_limitations=ignore_limitations,
            object_prefix=object_prefix,
            incremental_backfill=incremental_backfill,
            batch_size=eff_batch,
            strict=strict,
            watermark_catalog_table=watermark_catalog_table,
            pg_fillfactor=pg_fillfactor,
            limitations=(),
            strategy=MaintenanceStrategy.ATOMIC_QUERY_SYNC,
            alg_mode="single",
            ir=base_ir,
            accepted_limitations=tuple(sorted({str(c).strip() for c in (accept_limitations or ()) if str(c).strip()})),
        )
        stage1 = cls._from_context(stage_ctx)

        unwrapped_view = stage1._try_unwrap_simple_cte_or_derived_view(view_spec)
        primary_schema: Optional[TableSchema] = schemas_lower_ro.get(
            unwrapped_view.base_table.lower()
        )
        if logical_view_name.lower() in schemas_lower_ro:
            raise ValueError(
                f"User-facing materialized view name '{logical_view_name}' "
                "conflicts with an existing base table. Choose a distinct view name."
            )

        stage2 = stage1.with_ir(primary_schema=primary_schema)
        expanded_view = mark_native_aggregates(
            replace(stage2._try_expand_select_star(unwrapped_view), view_name=logical_view_name),
            stage2.dialect,
        )
        stage3 = stage2.with_ir(view=expanded_view)

        (
            work_items,
            comp_reader_exprs,
            extra_having_items,
            rewritten_having,
            decomposed_ok,
        ) = stage3._decompose_compound_aggregates_and_having()
        eff_group_by = tuple(stage3._resolve_effective_group_by_exprs())

        stage4 = stage3.with_ir(
            working_select_items=tuple(work_items),
            compound_reader_exprs=tuple(comp_reader_exprs.items()),
            extra_having_select_items=tuple(extra_having_items),
            rewritten_having_clause=rewritten_having,
            decomposed_ok=decomposed_ok,
            effective_group_by_exprs=eff_group_by,
        )

        limitations = tuple(stage4._detect_limitations())
        stage5 = stage4._with_context(stage4.context.with_updates(limitations=limitations))
        # Limitations that depend on the maintenance mode (order-sensitive
        # aggregates over joins/UNION ALL) are reported together with the
        # others, so a rejection lists every code and an opt-in warns once.
        try:
            pre_strategy, pre_mode = stage5._select_strategy_and_mode()
        except Exception:
            pre_strategy = pre_mode = None  # surfaced by validation below
        if pre_strategy is not None and pre_mode is not None:
            mode_limitations = tuple(stage5._order_sensitive_aggregate_limitations(pre_strategy, pre_mode))
            if mode_limitations:
                limitations = limitations + mode_limitations
                stage5 = stage5._with_context(stage5.context.with_updates(limitations=limitations))
        stage5._enforce_limitations()
        stage5._validate_schema_and_view()

        strategy, alg_mode = stage5._select_strategy_and_mode()
        if strict and strategy != MaintenanceStrategy.ALGEBRAIC_DELTA:
            raise StrictCompilationError(
                f"View '{logical_view_name}' cannot be compiled with Tier 1 ALGEBRAIC_DELTA "
                f"maintenance (selected fallback strategy: {strategy.value}) under strict=True."
            )

        ir_extra: Dict[str, Any] = {}
        if strategy == MaintenanceStrategy.ALGEBRAIC_DELTA:
            if alg_mode == "union_all_agg":
                meta = stage5._extract_union_all_agg_metadata()
                if meta is not None:
                    ir_extra["union_all_inner_sql"] = meta[0]
                    ir_extra["union_all_sub_alias"] = meta[1]
                    ir_extra["union_all_branch_specs"] = meta[2]
            elif alg_mode == "setop":
                _, setop_pks, _, _ = stage5._resolve_setop_pk_info()
                ir_extra["setop_pk_cols"] = setop_pks

        final_ir = stage5.ir.with_updates(**ir_extra) if ir_extra else stage5.ir
        return stage5.context.with_updates(strategy=strategy, alg_mode=alg_mode, ir=final_ir)

    # Read-only accessors backed by immutable CompilationContext / LogicalIR
    @property
    def context(self) -> CompilationContext:
        return self._context

    @property
    def ir(self) -> LogicalIR:
        return self._context.ir

    @property
    def schemas(self) -> Mapping[str, TableSchema]:
        return self._context.schemas

    @property
    def _schemas_lower(self) -> Mapping[str, TableSchema]:
        return self._context.schemas_lower

    @property
    def raw_view(self) -> ViewSpec:
        return self._context.ir.raw_view

    @property
    def engine_version(self) -> EngineVersion:
        return self._context.engine_version

    @property
    def dialect(self) -> Dialect:
        return self._context.dialect

    @property
    def subdialect(self) -> EngineSubdialect:
        return self._context.subdialect

    @property
    def ignore_limitations(self) -> bool:
        return self._context.ignore_limitations

    @property
    def accepted_limitations(self) -> Tuple[str, ...]:
        return self._context.accepted_limitations

    @property
    def object_prefix(self) -> str:
        return self._context.object_prefix

    @property
    def incremental_backfill(self) -> bool:
        return self._context.incremental_backfill

    @property
    def batch_size(self) -> int:
        return self._context.batch_size

    @property
    def strict(self) -> bool:
        return self._context.strict

    @property
    def watermark_catalog_table(self) -> Optional[str]:
        return self._context.watermark_catalog_table

    @property
    def pg_fillfactor(self) -> Optional[int]:
        return self._context.pg_fillfactor

    @property
    def logical_view_name(self) -> str:
        return self._context.ir.logical_view_name

    @property
    def db_view_name(self) -> str:
        return self._context.ir.db_view_name

    @property
    def user_indexes(self) -> Tuple[ViewIndexSpec, ...]:
        return self._context.ir.user_indexes

    @property
    def primary_schema(self) -> Optional[TableSchema]:
        return self._context.ir.primary_schema

    @property
    def view(self) -> ViewSpec:
        return self._context.ir.view

    @property
    def _working_select_items(self) -> Tuple[SelectItem, ...]:
        return self._context.ir.working_select_items

    @property
    def _compound_reader_exprs(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._context.ir.compound_reader_exprs))

    @property
    def _extra_having_select_items(self) -> Tuple[SelectItem, ...]:
        return self._context.ir.extra_having_select_items

    @property
    def _rewritten_having_clause(self) -> Optional[str]:
        return self._context.ir.rewritten_having_clause

    @property
    def _decomposed_ok(self) -> bool:
        return self._context.ir.decomposed_ok

    @property
    def _effective_group_by_exprs(self) -> Tuple[str, ...]:
        return self._context.ir.effective_group_by_exprs

    @property
    def limitations(self) -> Tuple[TransparencyLimitation, ...]:
        return self._context.limitations

    @property
    def _alg_mode(self) -> str:
        return self._context.alg_mode

    @property
    def strategy(self) -> MaintenanceStrategy:
        return self._context.strategy

    @property
    def _setop_pk_cols(self) -> Tuple[str, ...]:
        return self._context.ir.setop_pk_cols

    @property
    def _union_all_inner_sql(self) -> Optional[str]:
        return self._context.ir.union_all_inner_sql

    @property
    def _union_all_sub_alias(self) -> str:
        return self._context.ir.union_all_sub_alias

    @property
    def _union_all_branch_specs(self) -> Tuple[ViewSpec, ...]:
        return self._context.ir.union_all_branch_specs

    @property
    def _uall_has_nullable_grp(self) -> Optional[bool]:
        return self._context.ir.uall_has_nullable_grp

    @property
    def _uall_orig_grp_exprs(self) -> Tuple[str, ...]:
        if self._context.ir.uall_orig_grp_exprs is not None:
            return self._context.ir.uall_orig_grp_exprs
        return self._context.ir.effective_group_by_exprs

    @property
    def _uall_grp_alias_map(self) -> Optional[Mapping[str, str]]:
        return self._context.ir.uall_grp_alias_map

    @property
    def _uall_orig_item_map(self) -> Mapping[int, SelectItem]:
        if self._context.ir.uall_orig_item_map is not None:
            return self._context.ir.uall_orig_item_map
        return MappingProxyType({})
