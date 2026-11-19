"""PostgreSQL statement-level maintenance: the only PostgreSQL mechanism.

Every maintained source table gets one ``AFTER INSERT``, one ``AFTER UPDATE``
and one ``AFTER DELETE`` statement-level trigger with transition tables, plus
an ``AFTER TRUNCATE`` trigger, on the table itself and on every partition
(docs/design-bulk-maintenance.md, "PostgreSQL: statement-level
transition-table triggers, the only mechanism"). The trigger functions are
set-based; nothing iterates over transition rows.

The rules every body follows:

1. One consolidated signed delta per statement. The transition rows become
   ``__mv_s = +1`` (new) and ``__mv_s = -1`` (old) rows; an ``UPDATE`` is the
   ``UNION ALL`` of both, so a row that stays in its group nets out and a
   primary-key or group-key change is a retraction plus an insertion. A
   statement with an empty transition table returns before any work.
2. Deterministic lock order. Shared backing rows (aggregate groups, counted
   rows) are upserted with one ``INSERT ... SELECT ... GROUP BY key ORDER BY
   key ON CONFLICT`` statement, so every writer locks them in the same key
   order. Rows keyed by a source primary key are already serialized by the
   source rows' own locks. Group-scoped recomputes lock one lock-table row
   per affected group key hash, in hash order (and the whole-view row in
   SHARE mode); full recomputes lock the whole-view row exclusively.
3. Lock, then recompute in a later statement. Native recomputations (MIN/MAX
   after a retraction of the extreme, BIT_AND/BIT_OR after a retraction,
   approximate or scale-dependent AVG, statistics, group-scoped and full
   recomputes) run in statements after the lock statement. Under READ
   COMMITTED every PL/pgSQL statement takes a fresh snapshot, so the
   recompute sees every committed writer of the locked groups; under
   REPEATABLE READ/SERIALIZABLE a concurrent change of a locked row makes the
   lock statement fail with a serialization error.
4. Empty-group deletes are restricted to the keys the statement retracted
   from.
5. Partitioned sources: the triggers go on the parent and on every partition
   (direct partition writes fire only the partition's statement triggers). A
   source that is itself a partition or an inheritance child, or that has
   non-partition inheritance children, is refused at installation.
6. The functions run with the search_path of their creation (``SET
   search_path FROM CURRENT``): a writer's search_path cannot redirect the
   recomputes to other tables.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .._host import MixinHost
from ..engines import AggFunc
from ..logical_ir import SelectItem
from .templates import render_procedural_sql

SIGN = "__mv_s"
"""Column carrying +1 (new row) or -1 (old row) in the signed delta."""

PG_NEW_TABLE = "__mv_new_rows"
PG_OLD_TABLE = "__mv_old_rows"

EVENTS: Tuple[str, ...] = ("INSERT", "UPDATE", "DELETE")
"""Statement events with transition tables (TRUNCATE has none)."""


def event_has_new(event: str) -> bool:
    return event in ("INSERT", "UPDATE")


def event_has_old(event: str) -> bool:
    return event in ("DELETE", "UPDATE")


def signed_rows_sql(event: str) -> str:
    """The signed delta of one statement: every transition row with ``__mv_s``.

    The columns are ``__mv_s`` followed by the source table's columns, so view
    expressions rewritten against the derived table's alias stay valid.
    """
    new = f"SELECT 1 AS {SIGN}, __mv_n.* FROM {PG_NEW_TABLE} AS __mv_n"
    old = f"SELECT -1 AS {SIGN}, __mv_o.* FROM {PG_OLD_TABLE} AS __mv_o"
    if event == "INSERT":
        return new
    if event == "DELETE":
        return old
    if event == "UPDATE":
        return f"{old}\n        UNION ALL\n        {new}"
    raise ValueError(f"no transition tables for event {event!r}")


@dataclass(frozen=True)
class PgDeltaSource:
    """One relation of signed rows that feeds a view's maintenance.

    ``from_sql`` receives the signed-rows SELECT of the firing statement and
    returns a FROM-clause body (without the ``FROM`` keyword) that exposes it,
    for example ``(<signed rows>) AS __mv_d`` for a single-table view or the
    view's join with the source alias replaced by it. Every other field is SQL
    valid in that FROM clause.
    """

    from_sql: Callable[[str], str]
    sign: str
    where: Optional[str]
    keys: Sequence[str] = ()
    values: Mapping[str, str] = field(default_factory=dict)
    extras: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PgAggregateSpec:
    """Backing layout of a grouped (or global) aggregate view.

    ``items`` are the working select items of the view whose layout names the
    backing columns; ``grp_exprs`` its effective GROUP BY expressions (the
    sources provide one key per expression, in order); ``value_key`` maps an
    aggregate item to the key of its value slot in ``PgDeltaSource.values``.
    ``recompute`` returns the native per-group recomputation of an item,
    correlated with the backing table's row; ``native_recompute`` says which
    items are never folded.
    """

    mv_table: str
    row_count_col: str
    items: Sequence[SelectItem]
    grp_exprs: Sequence[str]
    has_nullable_grp: bool
    pk_cols: Sequence[str]
    nonnull_map: Mapping[str, str]
    sum_map: Mapping[str, str]
    sum2_map: Mapping[str, str]
    tcnt_map: Mapping[str, str]
    distinct_aux: Mapping[str, Tuple[str, SelectItem, str]]
    value_key: Callable[[SelectItem], str]
    recompute: Callable[[SelectItem], str]
    native_recompute: Callable[[SelectItem], bool]


_FOLDED_AGGS = (
    AggFunc.COUNT_STAR,
    AggFunc.COUNT_COL,
    AggFunc.SUM,
    AggFunc.TOTAL,
    AggFunc.AVG,
    AggFunc.MIN,
    AggFunc.MAX,
    AggFunc.BIT_AND,
    AggFunc.BIT_OR,
    AggFunc.BOOL_AND,
    AggFunc.BOOL_OR,
)


class PgStatementMaintenanceMixin(MixinHost):
    """Builds the PostgreSQL statement-level trigger functions and triggers."""

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------
    def _pg_emit_statement_triggers(
        self,
        base_t: str,
        bodies: Mapping[str, str],
        trunc_body: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Functions and statement triggers for one (view, source table) pair.

        Returns ``(ddls, trigger_names, function_names)``. The last DDL is a
        ``DO`` block that refuses unsupported inheritance shapes and installs
        the same triggers on every partition of ``base_t``.
        """
        missing = [e for e in EVENTS if not bodies.get(e, "").strip()]
        if missing:
            raise AssertionError(
                f"PostgreSQL statement-level maintenance of {base_t!r} has no set-based body for {missing}"
            )
        p = self.object_prefix if self.object_prefix else "__matview_"
        stem = f"{self.logical_view_name}__{base_t}"
        q_base_t = self._q(base_t)
        suffix = {"INSERT": "_ai", "UPDATE": "_au", "DELETE": "_ad"}
        referencing = {
            "INSERT": f"NEW TABLE AS {PG_NEW_TABLE}",
            "UPDATE": f"OLD TABLE AS {PG_OLD_TABLE} NEW TABLE AS {PG_NEW_TABLE}",
            "DELETE": f"OLD TABLE AS {PG_OLD_TABLE}",
        }
        guard_table = {"INSERT": PG_NEW_TABLE, "UPDATE": PG_NEW_TABLE, "DELETE": PG_OLD_TABLE}

        ddls: List[str] = []
        trig_names: List[str] = []
        fn_names: List[str] = []
        # (trigger name, event clause, referencing clause or None, function)
        installs: List[Tuple[str, str, Optional[str], str]] = []
        for event in EVENTS:
            fn = self._bounded_ident(f"{p}fn_sync_", stem, suffix[event])
            trg = self._bounded_ident(f"{p}trg_", stem, suffix[event])
            ddls.append(
                render_procedural_sql(
                    "pg_statement_function",
                    fn_name=fn,
                    guard_table=guard_table[event],
                    body_sql=bodies[event].strip(),
                )
            )
            ddls.append(
                render_procedural_sql(
                    "pg_statement_trigger",
                    trg_name=trg,
                    event=event,
                    q_base_t=q_base_t,
                    referencing=referencing[event],
                    fn_name=fn,
                )
            )
            installs.append((trg, event, referencing[event], fn))
            trig_names.append(trg)
            fn_names.append(fn)

        fn_trunc = self._bounded_ident(f"{p}fn_sync_", stem, "_trunc")
        trg_trunc = self._bounded_ident(f"{p}trg_", stem, "_trunc")
        ddls.append(render_procedural_sql("pg_truncate_function", fn_name=fn_trunc, trunc_block=trunc_body.strip()))
        ddls.append(render_procedural_sql("pg_truncate_trigger", trg_name=trg_trunc, q_base_t=q_base_t, fn_name=fn_trunc))
        installs.append((trg_trunc, "TRUNCATE", None, fn_trunc))
        trig_names.append(trg_trunc)
        fn_names.append(fn_trunc)

        ddls.append(self._pg_partition_install_block(base_t, installs))
        return ddls, trig_names, fn_names

    def _pg_partition_install_block(
        self, base_t: str, installs: Sequence[Tuple[str, str, Optional[str], str]]
    ) -> str:
        """``DO`` block: refuse inheritance shapes statement triggers cannot see,
        then create every trigger on each partition (all levels) of ``base_t``."""
        reg = "'" + self._q(base_t).replace("'", "''") + "'"
        creates: List[str] = []
        for trg, event, referencing, fn in installs:
            ref_sql = f" REFERENCING {referencing}" if referencing else ""
            stmt = (
                f"CREATE TRIGGER {trg} AFTER {event} ON %s{ref_sql} "
                f"FOR EACH STATEMENT EXECUTE FUNCTION {fn}()"
            )
            creates.append(f"        EXECUTE format('{stmt}', __mv_child);")
        return render_procedural_sql(
            "pg_partition_trigger_install",
            reg=reg,
            table_label=base_t.replace("'", "''"),
            creates="\n".join(creates),
        )

    # ------------------------------------------------------------------
    # Shared SQL helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pg_projection_sql(event: str, sources: Sequence[PgDeltaSource], columns: Callable[[PgDeltaSource], List[str]]) -> str:
        """``(<branch> UNION ALL ...) AS __mv_p``: each source's signed rows,
        filtered by its predicate and projected to ``columns(source)``."""
        signed = signed_rows_sql(event)
        parts: List[str] = []
        for src in sources:
            cols = [f"{src.sign} AS {SIGN}", *columns(src)]
            where = f"\n        WHERE {src.where}" if src.where else ""
            parts.append(f"SELECT {', '.join(cols)}\n        FROM {src.from_sql(signed)}{where}")
        return "(\n        " + "\n        UNION ALL\n        ".join(parts) + "\n    ) AS __mv_p"

    def _pg_watermark_predicate(self, record: str, pk_col: Optional[str] = None) -> Optional[str]:
        """Skip source rows inside an active pk-range backfill window.

        The backfill folds those rows itself (``backfill_incrementally``);
        ``pk_col`` defaults to the first integer primary-key column of the
        view's primary table. ``None`` when no watermark applies.
        """
        if not self.watermark_catalog_table:
            return None
        if pk_col is None:
            if not self.primary_schema:
                return None
            for c_def in self.primary_schema.columns.values():
                if c_def.primary_key and "INT" in c_def.data_type.upper():
                    pk_col = c_def.name
                    break
        if not pk_col:
            return None
        q_pk = self._q(pk_col)
        view_lit = self.db_view_name.replace("'", "''")
        return (
            f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
            f"WHERE view_name = '{view_lit}' AND backfill_active = 1 "
            f"AND {record}.{q_pk} > backfill_cursor_pk AND {record}.{q_pk} <= backfill_max_pk)"
        )

    @staticmethod
    def _pg_null_split_subquery(
        build: Callable[[List[str]], str], nullable_keys: Sequence[Tuple[str, str]]
    ) -> str:
        """A correlated per-group subquery whose nullable key equalities stay
        index-friendly.

        ``nullable_keys`` are ``(source expression, correlated value)``
        pairs that must match NULL-safely; ``build(conditions)`` returns the
        subquery for extra WHERE conditions. ``IS NOT DISTINCT FROM`` cannot
        use an index (one recompute would scan the whole source), so the
        subquery is split on which correlated values are NULL: ``= value``
        or ``IS NULL`` per key, only one branch runs. Beyond three nullable
        keys (eight branches) it falls back to ``(a = b OR (a IS NULL AND b
        IS NULL))``, which still uses a bitmap index scan.
        """
        if not nullable_keys:
            return build([])
        if len(nullable_keys) > 3:
            return build([f"({l} = {r} OR ({l} IS NULL AND {r} IS NULL))" for l, r in nullable_keys])
        branches: List[Tuple[str, str]] = []
        for mask in itertools.product((False, True), repeat=len(nullable_keys)):
            when = " AND ".join(f"{r} IS NULL" if is_null else f"{r} IS NOT NULL" for (_l, r), is_null in zip(nullable_keys, mask))
            conds = [f"{l} IS NULL" if is_null else f"{l} = {r}" for (l, r), is_null in zip(nullable_keys, mask)]
            branches.append((when, build(conds)))
        last = branches.pop()[1]
        return "(CASE " + " ".join(f"WHEN {w} THEN {b}" for w, b in branches) + f" ELSE {last} END)"

    @staticmethod
    def _pg_key_match(lhs_cols: Sequence[str], rhs_sql: str) -> str:
        """``(a, b) IN (<rhs_sql>)`` (row-value form only for several columns)."""
        if len(lhs_cols) == 1:
            return f"{lhs_cols[0]} IN ({rhs_sql})"
        return f"({', '.join(lhs_cols)}) IN ({rhs_sql})"

    # ------------------------------------------------------------------
    # Grouped / global aggregates (single table, join, UNION ALL branches)
    # ------------------------------------------------------------------
    def _pg_aggregate_bodies(
        self, spec: PgAggregateSpec, sources: Sequence[PgDeltaSource]
    ) -> Dict[str, str]:
        """Set-based bodies for a backing table with one row per group."""
        return {event: self._pg_aggregate_body(spec, sources, event) for event in EVENTS}

    def _pg_aggregate_body(
        self, spec: PgAggregateSpec, sources: Sequence[PgDeltaSource], event: str
    ) -> str:
        q = self._q
        mv = spec.mv_table
        grp_exprs = list(spec.grp_exprs)
        n_keys = len(grp_exprs)
        has_old = event_has_old(event)
        items = list(spec.items)
        s = f"__mv_p.{SIGN}"

        # Value slots: one projected column per distinct value key.
        slot_of: Dict[str, str] = {}
        for it in items:
            if it.is_aggregate:
                vk = spec.value_key(it)
                if vk not in slot_of:
                    slot_of[vk] = f"__mv_v{len(slot_of)}"
        for vk in list(spec.nonnull_map) + list(spec.sum_map) + list(spec.sum2_map) + list(spec.tcnt_map):
            if vk not in slot_of:
                slot_of[vk] = f"__mv_v{len(slot_of)}"
        for vk in spec.distinct_aux:
            if vk not in slot_of:
                slot_of[vk] = f"__mv_v{len(slot_of)}"

        # Non-aggregate items: a group column (reuse its key) or an extra
        # grouped expression that is functionally dependent on the keys.
        key_of_alias: Dict[str, int] = {}
        for i, g in enumerate(grp_exprs):
            key_of_alias.setdefault(self._group_col_alias(g).lower(), i)
        extra_of_alias: Dict[str, str] = {}
        for it in items:
            if not it.is_aggregate and it.alias.lower() not in key_of_alias:
                extra_of_alias[it.alias.lower()] = f"__mv_x{len(extra_of_alias)}"

        def _columns(src: PgDeltaSource) -> List[str]:
            cols = [f"{k} AS __mv_k{i}" for i, k in enumerate(src.keys)]
            for vk, slot in slot_of.items():
                cols.append(f"{src.values[vk]} AS {slot}")
            for al, xcol in extra_of_alias.items():
                cols.append(f"{src.extras[al]} AS {xcol}")
            return cols

        for src in sources:
            if len(src.keys) != n_keys:
                raise AssertionError("every delta source must provide one key per GROUP BY expression")
        proj = self._pg_projection_sql(event, sources, _columns)
        keys = [f"__mv_p.__mv_k{i}" for i in range(n_keys)]

        # The backing primary key, as SQL over the projection.
        if not grp_exprs:
            pk_exprs = ["1"]
        elif spec.has_nullable_grp:
            pk_exprs = [self._canonical_group_pk_expr(keys)]
        else:
            pk_by_alias = {self._group_col_alias(g).lower(): keys[i] for i, g in enumerate(grp_exprs)}
            pk_exprs = [pk_by_alias[c.lower()] for c in spec.pk_cols]
        q_pk_cols = [q(c) for c in spec.pk_cols]
        group_by = keys + [f"__mv_p.{x}" for x in extra_of_alias.values()]
        group_by_sql = f"\n    GROUP BY {', '.join(group_by)}" if group_by else "\n    GROUP BY 1"
        order_by_sql = f"\n    ORDER BY {', '.join(pk_exprs)}" if grp_exprs else ""

        def v(vk: str) -> str:
            return f"__mv_p.{slot_of[vk]}"

        def cnt_delta(val: str) -> str:
            return f"SUM(CASE WHEN ({val}) IS NOT NULL THEN {s} ELSE 0 END)"

        def sum_delta(val: str) -> str:
            # Positive and negative parts are summed separately: multiplying
            # by the sign could overflow (-1 * INT_MIN) or change the type.
            return (
                f"(CASE WHEN COUNT({val}) = 0 THEN NULL ELSE "
                f"COALESCE(SUM(CASE WHEN {s} > 0 THEN {val} END), 0) - "
                f"COALESCE(SUM(CASE WHEN {s} < 0 THEN {val} END), 0) END)"
            )

        def merged_sum(col: str, nn_col: str) -> str:
            return (
                f"CASE WHEN ({mv}.{nn_col} + EXCLUDED.{nn_col}) = 0 THEN NULL "
                f"ELSE COALESCE({mv}.{col}, 0) + COALESCE(EXCLUDED.{col}, 0) END"
            )

        ins_cols: List[str] = []
        ins_vals: List[str] = []
        sets: List[str] = []
        done: set = set()
        always_recompute: List[SelectItem] = []
        minmax_recompute: List[SelectItem] = []
        bitwise_recompute: List[SelectItem] = []
        distinct_items: List[SelectItem] = []

        def add(col: str, ins: str, merge: Optional[str]) -> None:
            qcol = q(col)
            if qcol in done:
                return
            done.add(qcol)
            ins_cols.append(qcol)
            ins_vals.append(ins)
            if merge is not None:
                sets.append(f"{qcol} = {merge}")

        if not grp_exprs:
            add("__mv_global_id", "1", None)
        elif spec.has_nullable_grp:
            add("__mv_pk", pk_exprs[0], None)
        nonagg_aliases = {it.alias.lower() for it in items if not it.is_aggregate}
        for i, g in enumerate(grp_exprs):
            mc = self._group_col_alias(g)
            if mc.lower() not in nonagg_aliases:
                add(mc, keys[i], None)

        for it in items:
            if not it.is_aggregate:
                al = it.alias.lower()
                if al in key_of_alias:
                    add(it.alias, keys[key_of_alias[al]], None)
                else:
                    add(it.alias, f"__mv_p.{extra_of_alias[al]}", None)
                continue
            col = q(it.alias)
            vk = spec.value_key(it)
            val = v(vk)
            fn = it.agg_func
            if it.is_distinct and vk in spec.distinct_aux:
                distinct_items.append(it)
                ins = {AggFunc.COUNT_COL: "0", AggFunc.TOTAL: "0.0"}.get(fn, "NULL") if fn else "NULL"
                add(it.alias, ins, f"{mv}.{col}")
            elif it.is_distinct or spec.native_recompute(it) or fn not in _FOLDED_AGGS:
                always_recompute.append(it)
                # Placeholder until the recompute statement; TOTAL is NOT NULL.
                add(it.alias, "0.0" if fn == AggFunc.TOTAL else "NULL", f"{mv}.{col}")
            elif fn == AggFunc.COUNT_STAR and not it.filter_where:
                add(it.alias, f"SUM({s})", f"{mv}.{col} + EXCLUDED.{col}")
            elif fn in (AggFunc.COUNT_STAR, AggFunc.COUNT_COL):
                add(it.alias, cnt_delta(val), f"{mv}.{col} + EXCLUDED.{col}")
            elif fn == AggFunc.SUM:
                nn = spec.nonnull_map.get(vk)
                if nn is None:
                    always_recompute.append(it)
                    add(it.alias, "NULL", f"{mv}.{col}")
                else:
                    add(it.alias, sum_delta(val), merged_sum(col, q(nn)))
            elif fn == AggFunc.TOTAL:
                # TOTAL is never NULL: 0.0 for an empty group.
                d_total = (
                    f"(COALESCE(SUM(CASE WHEN {s} > 0 THEN 1.0 * ({val}) END), 0.0) - "
                    f"COALESCE(SUM(CASE WHEN {s} < 0 THEN 1.0 * ({val}) END), 0.0))"
                )
                add(it.alias, d_total, f"COALESCE({mv}.{col}, 0.0) + COALESCE(EXCLUDED.{col}, 0.0)")
            elif fn == AggFunc.AVG:
                nn = spec.nonnull_map.get(vk)
                sm = spec.sum_map.get(vk)
                if nn is None or sm is None:
                    always_recompute.append(it)
                    add(it.alias, "NULL", f"{mv}.{col}")
                else:
                    qn, qs = q(nn), q(sm)
                    d_cnt, d_sum = cnt_delta(val), sum_delta(val)
                    new_cnt = f"({mv}.{qn} + EXCLUDED.{qn})"
                    new_sum = f"(COALESCE({mv}.{qs}, 0) + COALESCE(EXCLUDED.{qs}, 0))"
                    add(
                        it.alias,
                        f"(CASE WHEN {d_cnt} = 0 THEN NULL ELSE {self._avg_from_sum_count_sql(it, d_sum, d_cnt)} END)",
                        f"CASE WHEN {new_cnt} = 0 THEN NULL ELSE {self._avg_from_sum_count_sql(it, new_sum, new_cnt)} END",
                    )
            elif fn in (AggFunc.MIN, AggFunc.MAX):
                agg, op = ("MIN", "<") if fn == AggFunc.MIN else ("MAX", ">")
                add(
                    it.alias,
                    f"{agg}(CASE WHEN {s} > 0 THEN {val} END)",
                    f"CASE WHEN EXCLUDED.{col} IS NULL THEN {mv}.{col} "
                    f"WHEN {mv}.{col} IS NULL OR EXCLUDED.{col} {op} {mv}.{col} THEN EXCLUDED.{col} "
                    f"ELSE {mv}.{col} END",
                )
                minmax_recompute.append(it)
            elif fn in (AggFunc.BIT_AND, AggFunc.BIT_OR):
                agg, op = ("BIT_AND", "&") if fn == AggFunc.BIT_AND else ("BIT_OR", "|")
                add(
                    it.alias,
                    f"{agg}(CASE WHEN {s} > 0 THEN {val} END)",
                    f"CASE WHEN EXCLUDED.{col} IS NULL THEN {mv}.{col} "
                    f"WHEN {mv}.{col} IS NULL THEN EXCLUDED.{col} "
                    f"ELSE ({mv}.{col} {op} EXCLUDED.{col}) END",
                )
                bitwise_recompute.append(it)
            else:  # BOOL_AND / BOOL_OR
                nn = spec.nonnull_map.get(vk)
                tc = spec.tcnt_map.get(vk)
                if nn is None or tc is None:
                    always_recompute.append(it)
                    add(it.alias, "NULL", f"{mv}.{col}")
                    continue
                qn, qt = q(nn), q(tc)
                d_cnt = cnt_delta(val)
                d_true = f"SUM(CASE WHEN ({val}) IS NOT NULL AND ({val}) THEN {s} ELSE 0 END)"
                new_cnt = f"({mv}.{qn} + EXCLUDED.{qn})"
                new_true = f"({mv}.{qt} + EXCLUDED.{qt})"
                if fn == AggFunc.BOOL_AND:
                    ins = f"(CASE WHEN {d_cnt} = 0 THEN NULL WHEN {d_cnt} = {d_true} THEN TRUE ELSE FALSE END)"
                    merge = f"CASE WHEN {new_cnt} = 0 THEN NULL WHEN {new_cnt} = {new_true} THEN TRUE ELSE FALSE END"
                else:
                    ins = f"(CASE WHEN {d_cnt} = 0 THEN NULL WHEN {d_true} > 0 THEN TRUE ELSE FALSE END)"
                    merge = f"CASE WHEN {new_cnt} = 0 THEN NULL WHEN {new_true} > 0 THEN TRUE ELSE FALSE END"
                add(it.alias, ins, merge)

        # Hidden state columns. A distinct value key's helpers hold distinct
        # counts/sums and are recomputed from the auxiliary table instead.
        distinct_helper_cols: List[Tuple[str, str]] = []  # (quoted col, "count"|"sum")
        for vk, nn in spec.nonnull_map.items():
            if vk in spec.distinct_aux:
                if q(nn) not in done:
                    distinct_helper_cols.append((q(nn), "count"))
                    add(nn, "0", f"{mv}.{q(nn)}")
                continue
            add(nn, cnt_delta(v(vk)), f"{mv}.{q(nn)} + EXCLUDED.{q(nn)}")
        for vk, sm in spec.sum_map.items():
            if vk in spec.distinct_aux:
                if q(sm) not in done:
                    distinct_helper_cols.append((q(sm), "sum"))
                    add(sm, "NULL", f"{mv}.{q(sm)}")
                continue
            nn = spec.nonnull_map.get(vk)
            if nn is None:
                raise AssertionError(f"hidden sum {sm} has no non-NULL count")
            add(sm, sum_delta(v(vk)), merged_sum(q(sm), q(nn)))
        for vk, sm2 in spec.sum2_map.items():
            nn = spec.nonnull_map.get(vk)
            if nn is None:
                raise AssertionError(f"hidden sum of squares {sm2} has no non-NULL count")
            add(sm2, sum_delta(self._var_sq_sql(v(vk))), merged_sum(q(sm2), q(nn)))
        for vk, tc in spec.tcnt_map.items():
            val = v(vk)
            add(tc, f"SUM(CASE WHEN ({val}) IS NOT NULL AND ({val}) THEN {s} ELSE 0 END)", f"{mv}.{q(tc)} + EXCLUDED.{q(tc)}")
        q_rc = q(spec.row_count_col)
        add(spec.row_count_col, f"SUM({s})", f"{mv}.{q_rc} + EXCLUDED.{q_rc}")

        stmts: List[str] = []
        sep = ",\n        "
        stmts.append(
            f"INSERT INTO {mv} ({', '.join(ins_cols)})\n"
            f"    SELECT {', '.join(ins_vals)}\n"
            f"    FROM {proj}{group_by_sql}{order_by_sql}\n"
            f"    ON CONFLICT ({', '.join(q_pk_cols)}) DO UPDATE SET\n"
            f"        {sep.join(sets)};"
        )

        mv_pk_refs = [f"{mv}.{c}" for c in q_pk_cols]
        pk_select = ", ".join(pk_exprs)

        # Distinct auxiliary tables: maintained while the group rows are
        # locked by the upsert above, then the distinct aggregates of the
        # touched groups are recomputed from them.
        if spec.distinct_aux:
            if not grp_exprs:
                aux_key_cols = ["__mv_global_id"]
            elif spec.has_nullable_grp:
                aux_key_cols = ["__mv_pk"]
            else:
                aux_key_cols = [self._group_col_alias(g) for g in grp_exprs]
            q_aux_keys = [q(c) for c in aux_key_cols]
            touched: List[str] = []
            for vk, (aux_tbl, _rep, _dt) in spec.distinct_aux.items():
                val = v(vk)
                aux_pk_exprs = pk_exprs if (not grp_exprs or spec.has_nullable_grp) else keys
                stmts.append(
                    f"INSERT INTO {aux_tbl} ({', '.join(q_aux_keys)}, __val, __ref_cnt)\n"
                    f"    SELECT {', '.join(aux_pk_exprs)}, {val}, SUM({s})\n"
                    f"    FROM {proj}\n"
                    f"    WHERE {val} IS NOT NULL\n"
                    f"    GROUP BY {', '.join(aux_pk_exprs)}, {val}\n"
                    f"    ORDER BY {', '.join(aux_pk_exprs)}, {val}\n"
                    f"    ON CONFLICT ({', '.join(q_aux_keys)}, __val) DO UPDATE SET\n"
                    f"        __ref_cnt = {aux_tbl}.__ref_cnt + EXCLUDED.__ref_cnt;"
                )
                if has_old:
                    aux_refs = [f"{aux_tbl}.{c}" for c in q_aux_keys] + [f"{aux_tbl}.__val"]
                    stmts.append(
                        f"DELETE FROM {aux_tbl}\n"
                        f"    WHERE {self._pg_key_match(aux_refs, f'SELECT {chr(44).join(aux_pk_exprs)}, {val} FROM {proj} WHERE {s} < 0 AND {val} IS NOT NULL')}\n"
                        f"    AND {aux_tbl}.__ref_cnt <= 0;"
                    )
                touched.append(f"{val} IS NOT NULL")
            match_aux = " AND ".join(f"__mv_a.{c} = {mv}.{c}" for c in q_aux_keys)
            assigns: List[str] = []
            for it in distinct_items:
                aux_tbl = spec.distinct_aux[spec.value_key(it)][0]
                src = f"(SELECT {{agg}} FROM {aux_tbl} AS __mv_a WHERE {match_aux})"
                # A DISTINCT aggregate accumulates its distinct values in sort
                # order; accumulate the auxiliary values in the same order (a
                # float sum depends on it).
                if it.agg_func == AggFunc.SUM:
                    rhs = src.format(agg="SUM(__mv_a.__val ORDER BY __mv_a.__val)")
                elif it.agg_func == AggFunc.TOTAL:
                    rhs = f"COALESCE({src.format(agg='1.0 * SUM(__mv_a.__val ORDER BY __mv_a.__val)')}, 0.0)"
                elif it.agg_func == AggFunc.AVG:
                    rhs = src.format(agg="AVG(__mv_a.__val ORDER BY __mv_a.__val)")
                else:
                    rhs = f"COALESCE({src.format(agg='COUNT(*)')}, 0)"
                assigns.append(f"{q(it.alias)} = {rhs}")
            for qcol, kind in distinct_helper_cols:
                vk_for = next(k for k, c in [*spec.nonnull_map.items(), *spec.sum_map.items()] if q(c) == qcol)
                aux_tbl = spec.distinct_aux[vk_for][0]
                src = f"(SELECT {{agg}} FROM {aux_tbl} AS __mv_a WHERE {match_aux})"
                rhs = (
                    f"COALESCE({src.format(agg='COUNT(*)')}, 0)" if kind == "count" else src.format(agg="SUM(__mv_a.__val)")
                )
                assigns.append(f"{qcol} = {rhs}")
            if assigns:
                stmts.append(
                    f"UPDATE {mv} SET {', '.join(assigns)}\n"
                    f"    WHERE {self._pg_key_match(mv_pk_refs, f'SELECT {pk_select} FROM {proj} WHERE ' + ' OR '.join(touched))};"
                )

        if grp_exprs and has_old:
            stmts.append(
                f"DELETE FROM {mv}\n"
                f"    WHERE {self._pg_key_match(mv_pk_refs, f'SELECT {pk_select} FROM {proj} WHERE {s} < 0')}\n"
                f"    AND {mv}.{q_rc} <= 0;"
            )

        # Native recomputations, after the lock statement.
        rec_assigns: List[str] = []
        rec_conds: List[str] = []
        rec_cols: List[str] = []
        for it in always_recompute:
            rec_assigns.append(f"{q(it.alias)} = {spec.recompute(it)}")
        if always_recompute:
            rec_conds.append("TRUE")
        if has_old:
            for i, it in enumerate(minmax_recompute):
                col = q(it.alias)
                val = v(spec.value_key(it))
                agg, op = ("MIN", "<=") if it.agg_func == AggFunc.MIN else ("MAX", ">=")
                rcol = f"__mv_r{i}"
                rec_cols.append(f"{agg}(CASE WHEN {s} < 0 THEN {val} END) AS {rcol}")
                cond = f"(__mv_r.{rcol} IS NOT NULL AND ({mv}.{col} IS NULL OR __mv_r.{rcol} {op} {mv}.{col}))"
                rec_assigns.append(f"{col} = CASE WHEN {cond} THEN {spec.recompute(it)} ELSE {mv}.{col} END")
                rec_conds.append(cond)
            for i, it in enumerate(bitwise_recompute):
                col = q(it.alias)
                val = v(spec.value_key(it))
                rcol = f"__mv_b{i}"
                rec_cols.append(f"SUM(CASE WHEN {s} < 0 AND ({val}) IS NOT NULL THEN 1 ELSE 0 END) AS {rcol}")
                cond = f"(__mv_r.{rcol} > 0)"
                rec_assigns.append(f"{col} = CASE WHEN {cond} THEN {spec.recompute(it)} ELSE {mv}.{col} END")
                rec_conds.append(cond)
        if rec_assigns:
            key_cols = [f"{e} AS __mv_key{i}" for i, e in enumerate(pk_exprs)]
            rec_group = f"\n        GROUP BY {', '.join(group_by)}" if group_by else "\n        GROUP BY 1"
            r_sel = ", ".join(key_cols + rec_cols)
            match = " AND ".join(f"{mv}.{c} = __mv_r.__mv_key{i}" for i, c in enumerate(q_pk_cols))
            any_cond = "TRUE" if "TRUE" in rec_conds else " OR ".join(rec_conds)
            stmts.append(
                f"UPDATE {mv} SET\n        {sep.join(rec_assigns)}\n"
                f"    FROM (\n        SELECT {r_sel}\n        FROM {proj}{rec_group}\n    ) AS __mv_r\n"
                f"    WHERE {match} AND ({any_cond});"
            )
        return "\n".join(stmts)

    # ------------------------------------------------------------------
    # Rows keyed by a source primary key (projection, UNION ALL)
    # ------------------------------------------------------------------
    def _pg_keyed_rows_bodies(
        self,
        mv_table: str,
        key_cols: Sequence[str],
        value_cols: Sequence[str],
        sources: Sequence[PgDeltaSource],
        count_col: Optional[str] = None,
    ) -> Dict[str, str]:
        """Bodies for a backing table with exactly one row per source row.

        ``PgDeltaSource.keys`` give the backing key and ``values`` (keyed by
        backing column) the other columns. Each key belongs to one source row,
        whose row lock serializes its writers, so no extra lock order applies.
        A retracted key that the statement does not re-insert is deleted; every
        inserted key is upserted.
        """
        q = self._q
        q_keys = [q(c) for c in key_cols]
        q_vals = [q(c) for c in value_cols]

        def _key_columns(src: PgDeltaSource) -> List[str]:
            return [f"{k} AS __mv_k{i}" for i, k in enumerate(src.keys)]

        def _row_columns(src: PgDeltaSource) -> List[str]:
            return _key_columns(src) + [f"{src.values[c]} AS __mv_c{j}" for j, c in enumerate(value_cols)]

        k_sel = ", ".join(f"__mv_p.__mv_k{i}" for i in range(len(key_cols)))
        mv_refs = [f"{mv_table}.{c}" for c in q_keys]
        bodies: Dict[str, str] = {}
        for event in EVENTS:
            stmts: List[str] = []
            if event_has_old(event):
                old_keys = f"SELECT {k_sel} FROM {self._pg_projection_sql('DELETE', sources, _key_columns)}"
                cond = self._pg_key_match(mv_refs, old_keys)
                if event == "UPDATE":
                    new_keys = f"SELECT {k_sel} FROM {self._pg_projection_sql('INSERT', sources, _key_columns)}"
                    cond = f"{cond}\n    AND NOT {self._pg_key_match(mv_refs, new_keys)}"
                stmts.append(f"DELETE FROM {mv_table}\n    WHERE {cond};")
            if event_has_new(event):
                proj = self._pg_projection_sql("INSERT", sources, _row_columns)
                cols = q_keys + q_vals + ([q(count_col)] if count_col else [])
                vals = [f"__mv_p.__mv_k{i}" for i in range(len(key_cols))] + [
                    f"__mv_p.__mv_c{j}" for j in range(len(value_cols))
                ] + (["1"] if count_col else [])
                sets = [f"{c} = EXCLUDED.{c}" for c in q_vals]
                if count_col:
                    sets.append(f"{q(count_col)} = 1")
                conflict = f"DO UPDATE SET {', '.join(sets)}" if sets else "DO NOTHING"
                stmts.append(
                    f"INSERT INTO {mv_table} ({', '.join(cols)})\n"
                    f"    SELECT {', '.join(vals)}\n"
                    f"    FROM {proj}\n"
                    f"    ORDER BY {k_sel}\n"
                    f"    ON CONFLICT ({', '.join(q_keys)}) {conflict};"
                )
            bodies[event] = "\n".join(stmts)
        return bodies

    # ------------------------------------------------------------------
    # Counted rows (DISTINCT-like Z-sets: UNION, INTERSECT, EXCEPT, bags)
    # ------------------------------------------------------------------
    def _pg_counted_rows_bodies(
        self,
        mv_table: str,
        key_col: str,
        value_cols: Sequence[str],
        count_cols: Sequence[str],
        sources: Sequence[Tuple[PgDeltaSource, Mapping[str, str]]],
        empty_condition: str,
    ) -> Dict[str, str]:
        """Bodies for a backing table with one row per distinct value tuple.

        ``key_col`` is the canonical key of the value tuple (``keys[0]`` of
        each source); ``values`` give the value columns. Each source also
        maps every count column to the multiplier of its contribution
        (``"1"`` or ``"0"``). Rows are upserted in key order with the signed
        counts added; rows whose counts satisfy ``empty_condition`` (over the
        backing table) are deleted, restricted to retracted keys.
        """
        q = self._q
        q_key = q(key_col)
        q_vals = [q(c) for c in value_cols]
        q_cnts = [q(c) for c in count_cols]
        contrib_of = {id(src): contrib for src, contrib in sources}
        srcs = [src for src, _ in sources]

        def _columns(src: PgDeltaSource) -> List[str]:
            contrib = contrib_of[id(src)]
            cols = [f"{src.keys[0]} AS __mv_k0"]
            cols += [f"{src.values[c]} AS __mv_c{j}" for j, c in enumerate(value_cols)]
            cols += [f"{contrib[c]} AS __mv_n{j}" for j, c in enumerate(count_cols)]
            return cols

        bodies: Dict[str, str] = {}
        s = f"__mv_p.{SIGN}"
        for event in EVENTS:
            proj = self._pg_projection_sql(event, srcs, _columns)
            group_cols = ["__mv_p.__mv_k0"] + [f"__mv_p.__mv_c{j}" for j in range(len(value_cols))]
            cnt_vals = [f"SUM({s} * __mv_p.__mv_n{j})" for j in range(len(count_cols))]
            sets = [f"{c} = {mv_table}.{c} + EXCLUDED.{c}" for c in q_cnts]
            stmts = [
                f"INSERT INTO {mv_table} ({', '.join([q_key, *q_vals, *q_cnts])})\n"
                f"    SELECT {', '.join(group_cols + cnt_vals)}\n"
                f"    FROM {proj}\n"
                f"    GROUP BY {', '.join(group_cols)}\n"
                f"    ORDER BY __mv_p.__mv_k0\n"
                f"    ON CONFLICT ({q_key}) DO UPDATE SET {', '.join(sets)};"
            ]
            if event_has_old(event):
                stmts.append(
                    f"DELETE FROM {mv_table}\n"
                    f"    WHERE {mv_table}.{q_key} IN (SELECT __mv_p.__mv_k0 FROM {proj} WHERE {s} < 0)\n"
                    f"    AND {empty_condition};"
                )
            bodies[event] = "\n".join(stmts)
        return bodies

    # ------------------------------------------------------------------
    # Recompute strategies (group-scoped and full)
    # ------------------------------------------------------------------
    def _pg_view_lock_table(self) -> str:
        p = self.object_prefix if self.object_prefix else "__matview_"
        return self._bounded_ident(f"{p}lock_", self.logical_view_name)

    def _pg_view_lock_ddl(self) -> str:
        return self._create_table_ddl(
            self._pg_view_lock_table(),
            ["__mv_id BIGINT NOT NULL PRIMARY KEY", "__mv_n BIGINT NOT NULL DEFAULT 0"],
            is_backing_store=False,
        )

    def _pg_view_lock_stmt(self) -> str:
        """Exclusive whole-view lock (row id 0) for full recomputes.

        READ COMMITTED: later statements of the function take a snapshot that
        includes every writer that held the row before. REPEATABLE READ and
        SERIALIZABLE: if another transaction changed the row after this
        transaction's snapshot, the upsert fails with a serialization error.
        Group-scoped writers hold the same row in SHARE mode
        (``_pg_view_shared_lock_stmts``), so they exclude full recomputes but
        not each other.
        """
        lock_t = self._pg_view_lock_table()
        return (
            f"INSERT INTO {lock_t} (__mv_id, __mv_n) VALUES (0, 1)\n"
            f"    ON CONFLICT (__mv_id) DO UPDATE SET __mv_n = {lock_t}.__mv_n + 1;"
        )

    def _pg_view_shared_lock_stmts(self) -> List[str]:
        """SHARE lock on the whole-view lock row (created if missing)."""
        lock_t = self._pg_view_lock_table()
        # The row normally exists (created with the table, see
        # _pg_view_lock_ddls); a repopulation that emptied the table makes the
        # first writer recreate it.
        return [
            f"PERFORM 1 FROM {lock_t} WHERE __mv_id = 0 FOR SHARE;\n"
            f"    IF NOT FOUND THEN\n"
            f"        INSERT INTO {lock_t} (__mv_id, __mv_n) VALUES (0, 0) ON CONFLICT (__mv_id) DO NOTHING;\n"
            f"        PERFORM 1 FROM {lock_t} WHERE __mv_id = 0 FOR SHARE;\n"
            f"    END IF;",
        ]

    def _pg_view_lock_ddls(self) -> List[str]:
        """Lock table DDL plus its whole-view row (id 0)."""
        return [
            self._pg_view_lock_ddl(),
            f"INSERT INTO {self._pg_view_lock_table()} (__mv_id, __mv_n) VALUES (0, 0) ON CONFLICT (__mv_id) DO NOTHING;",
        ]

    def _pg_group_lock_id_sql(self, canonical_key: str) -> str:
        """Lock-table row id of a group (never 0, the whole-view row).

        Equal groups get equal ids because ``canonical_key`` is the canonical
        text (``_pg_key_text_fn``); unequal groups whose hashes collide share a
        row, which only makes their writers wait for each other.
        """
        return f"hashtextextended({canonical_key}, 0) | 1"

    # ------------------------------------------------------------------
    # Canonical group identity
    # ------------------------------------------------------------------
    def _pg_key_text_fn(self) -> str:
        """Name of the view's canonical-key-text function (see ``_pg_key_text_fn_ddl``)."""
        p = self.object_prefix if self.object_prefix else "__matview_"
        return self._bounded_ident(f"{p}key_", self.logical_view_name)

    def _pg_key_text_fn_ddl(self) -> str:
        """``<fn>(anyelement) RETURNS text``: equal values give equal text.

        The canonical text identifies a group in a backing primary key
        (``__mv_pk``) and in a lock id, so two values that the type's equality
        (the one ``GROUP BY``, ``DISTINCT`` and set operations use) treats as
        equal must map to the same text, and the text must not depend on
        session settings. ``ROW(v)::text`` meets neither: ``NUMERIC`` 1.0 and
        1.00, ``FLOAT8`` 0 and -0, ``INTERVAL`` '1 day' and '24 hours' are equal
        but print differently, and timestamps, ``bytea``, ``money`` and floats
        print according to ``TimeZone``/``DateStyle``, ``bytea_output``,
        ``lc_monetary`` and ``extra_float_digits``.

        Normalized: numeric and money (trailing fractional zeros), float4/8
        (both zeros, every NaN, otherwise the exact binary value), date,
        timestamp, timestamptz (UTC), interval (its comparison span in
        microseconds), bytea (hex), citext (lowercase), jsonb (numbers
        normalized, keys sorted), arrays and ranges (recursively). Types whose
        text output is canonical are returned as text. Composite, record and
        multirange values raise ``feature_not_supported``. Text under a
        nondeterministic collation is refused at registration
        (``_pg_collation_guard_ddl``).

        PL/pgSQL plans each statement only when it first runs, per argument
        type, so the casts in branches for other types are never planned.
        Built-ins are schema-qualified: the function does not depend on the
        caller's ``search_path``.
        """
        fn = self._pg_key_text_fn()
        text_types = (
            "int4", "int8", "int2", "text", "varchar", "bpchar", "bool", "uuid", "time", "timetz",
            "name", "char", "oid", "inet", "cidr", "macaddr",
        )
        text_list = ",\n           ".join(f"'pg_catalog.{t}'::pg_catalog.regtype" for t in text_types)
        num = "pg_catalog.numeric"
        ivl = "v::pg_catalog.interval"
        return (
            f"CREATE OR REPLACE FUNCTION {fn}(v anyelement) RETURNS text\n"
            "LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE AS $mv_key$\n"
            "DECLARE\n"
            "  t pg_catalog.regtype := pg_catalog.pg_typeof(v);\n"
            "  s pg_catalog.text;\n"
            "  f pg_catalog.float8;\n"
            "  ty record;\n"
            "BEGIN\n"
            f"  IF t IN ({text_list}) THEN\n"
            "    RETURN v::pg_catalog.text;\n"
            "  ELSIF t IN ('pg_catalog.numeric'::pg_catalog.regtype, 'pg_catalog.money'::pg_catalog.regtype) THEN\n"
            f"    s := v::{num}::pg_catalog.text;\n"
            "    IF pg_catalog.strpos(s, '.') > 0 THEN\n"
            "      s := pg_catalog.rtrim(pg_catalog.rtrim(s, '0'), '.');\n"
            "    END IF;\n"
            "    RETURN s;\n"
            "  ELSIF t IN ('pg_catalog.float8'::pg_catalog.regtype, 'pg_catalog.float4'::pg_catalog.regtype) THEN\n"
            "    f := v::pg_catalog.float8;\n"
            "    IF f = 0 THEN\n"
            "      RETURN '0';\n"
            "    ELSIF f = 'NaN'::pg_catalog.float8 THEN\n"
            "      RETURN 'NaN';\n"
            "    END IF;\n"
            "    RETURN pg_catalog.encode(pg_catalog.float8send(f), 'hex');\n"
            "  ELSIF t = 'pg_catalog.timestamptz'::pg_catalog.regtype THEN\n"
            "    IF pg_catalog.isfinite(v::pg_catalog.timestamptz) THEN\n"
            "      RETURN pg_catalog.to_char(v::pg_catalog.timestamptz AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS.US BC');\n"
            "    END IF;\n"
            "    RETURN v::pg_catalog.text;\n"
            "  ELSIF t = 'pg_catalog.timestamp'::pg_catalog.regtype THEN\n"
            "    IF pg_catalog.isfinite(v::pg_catalog.timestamp) THEN\n"
            "      RETURN pg_catalog.to_char(v::pg_catalog.timestamp, 'YYYY-MM-DD HH24:MI:SS.US BC');\n"
            "    END IF;\n"
            "    RETURN v::pg_catalog.text;\n"
            "  ELSIF t = 'pg_catalog.date'::pg_catalog.regtype THEN\n"
            "    IF pg_catalog.isfinite(v::pg_catalog.date) THEN\n"
            "      RETURN (v::pg_catalog.date - '2000-01-01'::pg_catalog.date)::pg_catalog.text;\n"
            "    END IF;\n"
            "    RETURN v::pg_catalog.text;\n"
            "  ELSIF t = 'pg_catalog.interval'::pg_catalog.regtype THEN\n"
            "    s := v::pg_catalog.text;\n"
            "    IF s IN ('infinity', '-infinity') THEN\n"
            "      RETURN s;\n"
            "    END IF;\n"
            "    -- interval equality compares months * 30 days + days + time.\n"
            f"    RETURN (((pg_catalog.date_part('year', {ivl})::{num} * 12\n"
            f"              + pg_catalog.date_part('month', {ivl})::{num}) * 30\n"
            f"             + pg_catalog.date_part('day', {ivl})::{num}) * 86400000000\n"
            f"            + pg_catalog.date_part('hour', {ivl})::{num} * 3600000000\n"
            f"            + pg_catalog.date_part('minute', {ivl})::{num} * 60000000\n"
            f"            + pg_catalog.date_part('microseconds', {ivl})::{num})::pg_catalog.text;\n"
            "  ELSIF t = 'pg_catalog.bytea'::pg_catalog.regtype THEN\n"
            "    RETURN pg_catalog.encode(v::pg_catalog.bytea, 'hex');\n"
            "  ELSIF t = 'pg_catalog.jsonb'::pg_catalog.regtype THEN\n"
            "    s := pg_catalog.jsonb_typeof(v::pg_catalog.jsonb);\n"
            "    IF s = 'number' THEN\n"
            f"      RETURN 'n' || {fn}((v::pg_catalog.jsonb #>> '{{}}')::{num});\n"
            "    ELSIF s = 'object' THEN\n"
            "      SELECT '{' || pg_catalog.string_agg(pg_catalog.length(e.key) || ':' || e.key || '=' || "
            f"{fn}(e.value), ',' ORDER BY e.key COLLATE \"C\") || '}}'\n"
            "        INTO s FROM pg_catalog.jsonb_each(v::pg_catalog.jsonb) AS e;\n"
            "      RETURN COALESCE(s, '{}');\n"
            "    ELSIF s = 'array' THEN\n"
            f"      SELECT '[' || pg_catalog.string_agg({fn}(e.value), ',' ORDER BY e.o) || ']'\n"
            "        INTO s FROM pg_catalog.jsonb_array_elements(v::pg_catalog.jsonb) WITH ORDINALITY AS e(value, o);\n"
            "      RETURN COALESCE(s, '[]');\n"
            "    END IF;\n"
            "    RETURN v::pg_catalog.text;\n"
            "  END IF;\n"
            "  SELECT typ.typname, typ.typtype, typ.typcategory INTO ty FROM pg_catalog.pg_type AS typ WHERE typ.oid = t;\n"
            "  IF ty.typcategory = 'A' THEN\n"
            "    SELECT COALESCE(pg_catalog.array_dims(v), '') || '{' || COALESCE(pg_catalog.string_agg(\n"
            "             CASE WHEN u.c IS NULL THEN 'N' ELSE pg_catalog.length(u.c) || ':' || u.c END, ',' ORDER BY u.o), '') || '}'\n"
            f"      INTO s FROM (SELECT {fn}(x.e) AS c, x.o FROM pg_catalog.unnest(v) WITH ORDINALITY AS x(e, o)) AS u;\n"
            "    RETURN s;\n"
            "  ELSIF ty.typtype = 'r' THEN\n"
            "    IF pg_catalog.isempty(v) THEN\n"
            "      RETURN 'empty';\n"
            "    END IF;\n"
            f"    s := {fn}(pg_catalog.lower(v));\n"
            "    s := (CASE WHEN pg_catalog.lower_inc(v) THEN '[' ELSE '(' END) || COALESCE(pg_catalog.length(s) || ':' || s, '') || ',';\n"
            f"    RETURN s || COALESCE(pg_catalog.length({fn}(pg_catalog.upper(v))) || ':' || {fn}(pg_catalog.upper(v)), '')\n"
            "           || (CASE WHEN pg_catalog.upper_inc(v) THEN ']' ELSE ')' END);\n"
            "  ELSIF ty.typname = 'citext' THEN\n"
            "    RETURN pg_catalog.lower(v::pg_catalog.text);\n"
            "  ELSIF ty.typtype IN ('b', 'e') THEN\n"
            "    RETURN v::pg_catalog.text;\n"
            "  END IF;\n"
            "  RAISE EXCEPTION 'matview: a grouping key of type % is not supported', pg_catalog.format_type(t, NULL)\n"
            "    USING ERRCODE = 'feature_not_supported';\n"
            "END\n"
            "$mv_key$;"
        )

    def _pg_collation_guard_ddl(self) -> Optional[str]:
        """Refuse (0A000) a view whose groups could compare under a
        nondeterministic collation.

        Under such a collation (PostgreSQL 12+) distinct strings compare equal
        ('A' = 'a' under a case-insensitive ICU collation): ``GROUP BY`` merges
        them, so no canonical text or index on the stored value identifies the
        group. Returns None when the view neither groups, deduplicates nor
        takes MIN/MAX. Otherwise the block checks, at registration, every
        column of a source table that the view text mentions and every
        collation it names after ``COLLATE``; it can refuse a view whose
        mention is not a key (a conservative refusal, never a silent one).
        """
        sql = self.view.raw_select_sql
        if not re.search(r"\b(?:GROUP|DISTINCT|UNION|INTERSECT|EXCEPT|PARTITION|MIN|MAX)\b", sql, re.IGNORECASE):
            return None
        tables = sorted({t for t in self.view.referenced_tables})
        if not tables:
            return None
        tag = "$mv_sql$"
        while tag in sql:
            tag = tag[:-1] + "x$"
        table_list = ", ".join("'" + self._q(t).replace("'", "''") + "'" for t in tables)
        return (
            "DO $mv_coll$\n"
            "DECLARE\n"
            f"  q pg_catalog.text := {tag}{sql}{tag};\n"
            "  r record;\n"
            "BEGIN\n"
            "  IF pg_catalog.current_setting('server_version_num')::pg_catalog.int4 < 120000 THEN\n"
            "    RETURN;\n"
            "  END IF;\n"
            "  FOR r IN EXECUTE\n"
            "      'SELECT a.attrelid::pg_catalog.regclass::pg_catalog.text AS rel, a.attname::pg_catalog.text AS what, c.collname::pg_catalog.text AS coll'\n"
            "      ' FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_collation c ON c.oid = a.attcollation'\n"
            "      ' WHERE a.attrelid = ANY ($1) AND a.attnum > 0 AND NOT a.attisdropped AND NOT c.collisdeterministic'\n"
            "      ' UNION ALL SELECT NULL, NULL, c.collname::pg_catalog.text FROM pg_catalog.pg_collation c'\n"
            "      ' WHERE NOT c.collisdeterministic'\n"
            f"      USING ARRAY[{table_list}]::pg_catalog.regclass[]\n"
            "  LOOP\n"
            "    IF (r.what IS NOT NULL AND q ~* ('(^|[^A-Za-z0-9_$])\"?' || pg_catalog.regexp_replace(r.what, '([^A-Za-z0-9_])', '\\\\\\1', 'g') || '\"?($|[^A-Za-z0-9_$])'))\n"
            "       OR (r.what IS NULL AND q ~* ('COLLATE\\s+(\"?[A-Za-z0-9_]+\"?\\.)?\"?' || pg_catalog.regexp_replace(r.coll, '([^A-Za-z0-9_])', '\\\\\\1', 'g') || '\"?($|[^A-Za-z0-9_$])')) THEN\n"
            "      RAISE EXCEPTION 'matview: % uses the nondeterministic collation %, which is not supported for grouping, DISTINCT, set operations, window partitions or MIN/MAX',\n"
            "        COALESCE(r.rel || '.' || pg_catalog.quote_ident(r.what), 'the view'), pg_catalog.quote_ident(r.coll)\n"
            "        USING ERRCODE = 'feature_not_supported';\n"
            "    END IF;\n"
            "  END LOOP;\n"
            "END\n"
            "$mv_coll$;"
        )
