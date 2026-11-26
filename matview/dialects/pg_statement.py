"""PostgreSQL statement-level maintenance: the only PostgreSQL mechanism.

Every maintained source table gets one ``AFTER INSERT``, one ``AFTER UPDATE``
and one ``AFTER DELETE`` statement-level trigger with transition tables, plus
an ``AFTER TRUNCATE`` trigger, on the table itself and on every partition
(docs/design-bulk-maintenance.md, "PostgreSQL: statement-level
transition-table triggers, the only mechanism"). All maintenance triggers are
``ENABLE ALWAYS``, so they also run under
``session_replication_role = replica``. Logical-replication publications must
exclude matview backing and auxiliary tables; otherwise both source changes and
their derived writes would be applied on the subscriber. The trigger functions
are set-based; nothing iterates over transition rows.

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
   source rows' own locks. Single-source group-scoped recomputes lock bounded
   hash-bucket rows (ids ``1..PG_GROUP_LOCK_BUCKETS``) in bucket-id order, with
   no shared lock on row 0 (avoiding MultiXactId churn); whole-view recomputes
   on such views lock all rows ``0..PG_GROUP_LOCK_BUCKETS`` in ascending id
   order so both directions exclude each other symmetrically without deadlocks.
   Full recomputes and multi-source join views (algebraic and group-scoped)
   lock row 0 exclusively before joining their delta against other sources, so
   every later statement takes a fresh READ COMMITTED snapshot (and REPEATABLE
   READ / SERIALIZABLE fail closed with 40001 on any concurrent write to
   another source).
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

import hashlib
import itertools
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .._host import MixinHost
from ..engines import AggFunc, UnsupportedMaterializedViewError
from ..logical_ir import SelectItem, TableSchema
from .templates import render_procedural_sql

SIGN = "__mv_s"
"""Column carrying +1 (new row) or -1 (old row) in the signed delta."""

PG_NEW_TABLE = "__mv_new_rows"
PG_OLD_TABLE = "__mv_old_rows"

EVENTS: Tuple[str, ...] = ("INSERT", "UPDATE", "DELETE")
"""Statement events with transition tables (TRUNCATE has none)."""

PG_GROUP_LOCK_BUCKETS = 1024
"""Fixed number of hash buckets (ids ``1..PG_GROUP_LOCK_BUCKETS``) in a
group-scoped view's PostgreSQL lock table. Must be a power of two so bucket
ids are computed with a single bitmask."""

PG_LOCK_TABLE_FILLFACTOR = 50
"""Heap fillfactor for PostgreSQL lock tables so repeated counter increments
stay HOT updates with opportunistic in-page pruning."""


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
    recompute_group_keys: Optional[Callable[[Sequence[str]], str]] = None
    has_variable_numeric_scale: Optional[Callable[[str], bool]] = None


@dataclass(frozen=True)
class PgStatementBracket:
    """Multi-source statement-overlap bracket for PostgreSQL join views (Stage 3).

    Pairs one ``BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE`` statement trigger
    with the ``AFTER`` statement triggers on every source table of a multi-source
    view. Single-source statements observe a single source bit in the transaction-
    local bracket GUC and run the unchanged fast path with zero staging overhead.
    When one query, writable CTE, FK cascade, or user trigger mutates two or more
    sources inside one open bracket, intermediate firings stage their signed
    deltas (or defer) and the closing trigger reconciles once against the final
    state of all sources.
    """

    lock_table: str
    source_bit: int
    reconcile_fn: Optional[str] = None
    stage_sqls: Mapping[str, str] = field(default_factory=dict)
    overlap_refresh_sql: str = ""
    trunc_cleanup_sql: str = ""
    overlap_lock_sql: str = ""


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
        bracket: Optional[PgStatementBracket] = None,
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
        # PostgreSQL fires AFTER triggers in alphabetical tgname order. Prefix
        # trigger names with a fixed-width lowercase hash of the view name alone
        # so any two views fire in the exact same order on every source table
        # (even for long shared prefixes or underscore/case boundary names),
        # preventing multi-view AB-BA lock-order deadlocks across sources.
        view_order_key = hashlib.sha1(self.logical_view_name.encode("utf-8")).hexdigest()[:12]
        trg_prefix = f"{p}trg_{view_order_key}_"
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
        # (trigger name, timing, event clause, referencing clause or None, function)
        installs: List[Tuple[str, str, str, Optional[str], str]] = []

        lock_reg: Optional[str] = None
        repair_fence_sql = self._pg_repair_snapshot_fence_sql()
        trunc_refresh_sql = trunc_body.strip()
        overlap_close_sql = ""
        if bracket is not None:
            lock_reg = "'" + bracket.lock_table.replace("'", "''") + "'"
            if bracket.trunc_cleanup_sql.strip():
                trunc_refresh_sql = f"{bracket.trunc_cleanup_sql.strip()}\n        {trunc_body.strip()}"
            overlap_close_sql = (
                f"PERFORM {bracket.reconcile_fn}();"
                if bracket.reconcile_fn
                else (bracket.overlap_refresh_sql.strip() or trunc_body.strip())
            )
            fn_before = self._bounded_ident(f"{p}fn_sync_", stem, "_bs")
            trg_before = self._bounded_ident(trg_prefix, stem, "_bs")
            ddls.append(
                render_procedural_sql(
                    "pg_before_bracket_function",
                    fn_name=fn_before,
                    lock_reg=lock_reg,
                    src_bit=bracket.source_bit,
                )
            )
            ddls.append(
                render_procedural_sql(
                    "pg_before_bracket_trigger",
                    trg_name=trg_before,
                    q_base_t=q_base_t,
                    fn_name=fn_before,
                )
            )
            ddls.append(f"ALTER TABLE {q_base_t} ENABLE ALWAYS TRIGGER {trg_before};")
            installs.append((trg_before, "BEFORE", "INSERT OR UPDATE OR DELETE OR TRUNCATE", None, fn_before))
            trig_names.append(trg_before)
            fn_names.append(fn_before)

        for event in EVENTS:
            fn = self._bounded_ident(f"{p}fn_sync_", stem, suffix[event])
            trg = self._bounded_ident(trg_prefix, stem, suffix[event])
            if bracket is not None:
                assert lock_reg is not None
                ddls.append(
                    render_procedural_sql(
                        "pg_bracketed_statement_function",
                        fn_name=fn,
                        lock_reg=lock_reg,
                        guard_table=guard_table[event],
                        overlap_lock_sql=bracket.overlap_lock_sql.strip(),
                        stage_sql=bracket.stage_sqls.get(event, "").strip(),
                        trunc_refresh_sql=trunc_refresh_sql,
                        overlap_close_sql=overlap_close_sql,
                        body_sql=bodies[event].strip(),
                        repair_fence_sql=repair_fence_sql,
                    )
                )
            else:
                ddls.append(
                    render_procedural_sql(
                        "pg_statement_function",
                        fn_name=fn,
                        guard_table=guard_table[event],
                        body_sql=bodies[event].strip(),
                        repair_fence_sql=repair_fence_sql,
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
            ddls.append(f"ALTER TABLE {q_base_t} ENABLE ALWAYS TRIGGER {trg};")
            installs.append((trg, "AFTER", event, referencing[event], fn))
            trig_names.append(trg)
            fn_names.append(fn)

        fn_trunc = self._bounded_ident(f"{p}fn_sync_", stem, "_trunc")
        trg_trunc = self._bounded_ident(trg_prefix, stem, "_trunc")
        if bracket is not None:
            assert lock_reg is not None
            ddls.append(
                render_procedural_sql(
                    "pg_bracketed_truncate_function",
                    fn_name=fn_trunc,
                    lock_reg=lock_reg,
                    overlap_lock_sql=bracket.overlap_lock_sql.strip(),
                    trunc_refresh_sql=trunc_refresh_sql,
                    repair_fence_sql=repair_fence_sql,
                )
            )
        else:
            ddls.append(
                render_procedural_sql(
                    "pg_truncate_function",
                    fn_name=fn_trunc,
                    trunc_block=trunc_body.strip(),
                    repair_fence_sql=repair_fence_sql,
                )
            )
        ddls.append(render_procedural_sql("pg_truncate_trigger", trg_name=trg_trunc, q_base_t=q_base_t, fn_name=fn_trunc))
        ddls.append(f"ALTER TABLE {q_base_t} ENABLE ALWAYS TRIGGER {trg_trunc};")
        installs.append((trg_trunc, "AFTER", "TRUNCATE", None, fn_trunc))
        trig_names.append(trg_trunc)
        fn_names.append(fn_trunc)

        ddls.append(self._pg_partition_install_block(base_t, installs))
        return ddls, trig_names, fn_names

    def _pg_partition_install_block(
        self, base_t: str, installs: Sequence[Tuple[str, str, str, Optional[str], str]]
    ) -> str:
        """``DO`` block: refuse inheritance shapes statement triggers cannot see,
        then create every trigger on each partition (all levels) of ``base_t``."""
        reg = "'" + self._q(base_t).replace("'", "''") + "'"
        creates: List[str] = []
        for trg, timing, event, referencing, fn in installs:
            ref_sql = f" REFERENCING {referencing}" if referencing else ""
            stmt = (
                f"CREATE TRIGGER {trg} {timing} {event} ON %s{ref_sql} "
                f"FOR EACH STATEMENT EXECUTE FUNCTION {fn}()"
            )
            creates.append(f"        EXECUTE format('{stmt}', __mv_child);")
            creates.append(
                f"        EXECUTE format('ALTER TABLE %s ENABLE ALWAYS TRIGGER {trg}', __mv_child);"
            )
        return render_procedural_sql(
            "pg_partition_trigger_install",
            reg=reg,
            table_label=base_t.replace("'", "''"),
            creates="\n".join(creates),
        )

    def _pg_join_stage_table(self, base_t: str) -> str:
        """Per-(view, source) staging table name for Stage 3 multi-source overlap deltas."""
        p = self.object_prefix if self.object_prefix else "__matview_"
        return self._bounded_ident(f"{p}stg_", f"{self.logical_view_name}__{base_t}")

    @staticmethod
    def _pg_quote_raw_col(name: str) -> str:
        """Quote a raw PostgreSQL catalog column name verbatim without trimming
        whitespace or stripping outer quotes, preserving exact case and spaces."""
        return '"' + name.replace('"', '""') + '"'

    def _pg_join_stage_table_ddls(
        self, base_t: str, col_names: Sequence[str]
    ) -> Tuple[str, str, str, str]:
        """Return ``(stage_table, create_table_ddl, index_name, create_index_ddl)``."""
        p = self.object_prefix if self.object_prefix else "__matview_"
        stg_tbl = self._pg_join_stage_table(base_t)
        idx_name = self._bounded_ident(f"{p}idx_stg_", f"{self.logical_view_name}__{base_t}")
        ts = self._schemas_lower.get(base_t.lower())
        col_defs = [
            "__mv_tx BIGINT NOT NULL",
            "__mv_seq INTEGER NOT NULL",
            f"{SIGN} SMALLINT NOT NULL",
        ]
        reserved_meta = {"__mv_tx", "__mv_seq", SIGN.lower()}
        if ts is not None:
            cols_by_lower = {c.lower(): c_def for c, c_def in ts.columns.items()}
            seen_cols: Set[str] = set()
            for cn in col_names:
                c_def = ts.columns.get(cn) or cols_by_lower.get(cn.lower())
                if c_def is None or c_def.name in seen_cols:
                    continue
                if c_def.name.lower() in reserved_meta:
                    raise UnsupportedMaterializedViewError(
                        f"Source table '{base_t}' column '{c_def.name}' collides with internal "
                        f"PostgreSQL staging metadata column name; rename the column or exclude it from the view."
                    )
                seen_cols.add(c_def.name)
                raw_dt = (c_def.data_type or "TEXT").strip()
                m_coll = re.search(
                    r"(\s+COLLATE\s+(?:(?:\"[^\"]+\"|\S+)\.)?(?:\"[^\"]+\"|\S+))\s*$",
                    raw_dt,
                    re.IGNORECASE,
                )
                if m_coll:
                    base_dt = raw_dt[: m_coll.start()].strip()
                    coll_tail = m_coll.group(1)
                else:
                    base_dt = raw_dt
                    coll_tail = ""
                dt_up = base_dt.upper()
                if dt_up in ("CHARACTER VARYING", "VARCHAR"):
                    # Keep unbounded VARCHAR (OID 1043) instead of TEXT (OID 25)
                    # so type-observing expressions (e.g. pg_typeof(v)) on staged
                    # rows match the native source table.
                    ddl_dt = f"VARCHAR{coll_tail}"
                elif dt_up in ("CHAR", "CHARACTER"):
                    # Unbounded CHAR means CHAR(1) in PostgreSQL DDL; BPCHAR keeps full length.
                    ddl_dt = f"BPCHAR{coll_tail}"
                else:
                    ddl_dt = f"{self._normalize_dialect_type(base_dt)}{coll_tail}"
                col_defs.append(f"{self._pg_quote_raw_col(c_def.name)} {ddl_dt}")
        create_tbl = (
            f"CREATE TABLE {stg_tbl} (\n    "
            + ",\n    ".join(col_defs)
            + "\n);"
        )
        create_idx = f"CREATE INDEX {idx_name} ON {stg_tbl} (__mv_tx, __mv_seq);"
        return stg_tbl, create_tbl, idx_name, create_idx

    def _pg_reconcile_fn_name(self) -> str:
        p = self.object_prefix if self.object_prefix else "__matview_"
        return self._bounded_ident(f"{p}fn_s3_", self.logical_view_name, "_reconcile")

    @staticmethod
    def _pg_staged_rows_select(stg_tbl: str) -> str:
        return f"SELECT * FROM {stg_tbl} WHERE __mv_tx = pg_catalog.txid_current() AND __mv_seq = __mv_blk.__mv_bseq"

    def _pg_stage_cols_for_table(
        self, base_t: str, extra_cols: Sequence[str] = ()
    ) -> List[str]:
        from sqlglot import exp as sqlglot_exp
        from ..sql_frontend import parse_sql_ast

        ts = self._schemas_lower.get(base_t.lower())
        if ts is None:
            return []
        ref_cols: List[str]
        try:
            q_ast = parse_sql_ast(self.view.raw_select_sql, dialect=self.dialect)
            has_non_count_star = any(
                not isinstance(st.parent, sqlglot_exp.Count)
                for st in q_ast.find_all(sqlglot_exp.Star)
            )
            if has_non_count_star:
                ref_cols = list(ts.columns.keys())
            else:
                cols_in_ast = {c.name.lower() for c in q_ast.find_all(sqlglot_exp.Column)}
                ref_cols = [c for c in ts.columns if c.lower() in cols_in_ast]
        except Exception:
            ref_cols = list(ts.columns.keys())

        cols_by_low = {c.lower(): (c, c_def) for c, c_def in ts.columns.items()}
        out: List[str] = []
        seen: Set[str] = set()
        for cn in [*ref_cols, *extra_cols]:
            if not cn:
                continue
            exact_def = ts.columns.get(cn)
            if exact_def is not None:
                actual_c, c_def = cn, exact_def
            else:
                entry = cols_by_low.get(cn.lower())
                if entry is None:
                    continue
                actual_c, c_def = entry
            if actual_c in seen:
                continue
            if c_def.generated_virtual and not (c_def.generation_expr and c_def.generation_expr.strip()):
                continue
            seen.add(actual_c)
            out.append(actual_c)
        return out

    def _pg_stage_sqls_for_table(
        self, base_t: str, stg_tbl: str, col_names: Sequence[str]
    ) -> Dict[str, str]:
        ts = self._schemas_lower.get(base_t.lower())
        cols_by_lower = {c.lower(): c_def for c, c_def in ts.columns.items()} if ts is not None else {}
        q_cols = [self._pg_quote_raw_col(c) for c in col_names]
        ins_cols = ["__mv_tx", "__mv_seq", SIGN, *q_cols]

        def _branch_select(sign: int, rec: str, tbl: str) -> str:
            exprs = [
                "pg_catalog.txid_current()",
                "__mv_blk.__mv_bseq",
                f"{sign}::smallint",
            ]
            for cn, q_c in zip(col_names, q_cols):
                c_def = (ts.columns.get(cn) or cols_by_lower.get(cn.lower())) if ts is not None else None
                if c_def is not None and c_def.generated_virtual and c_def.generation_expr and ts is not None:
                    rewritten = self._pg_rewrite_vgen_expr(c_def.generation_expr, ts, rec)
                    dt_raw = (c_def.data_type or "").strip()
                    m_coll = re.search(
                        r"(\s+COLLATE\s+(?:(?:\"[^\"]+\"|\S+)\.)?(?:\"[^\"]+\"|\S+))\s*$",
                        dt_raw,
                        re.IGNORECASE,
                    )
                    if m_coll:
                        base_dt = dt_raw[: m_coll.start()].strip()
                        coll_tail = m_coll.group(1)
                    else:
                        base_dt = dt_raw
                        coll_tail = ""
                    if base_dt and base_dt.upper() not in ("NULL", "USER-DEFINED", "ARRAY"):
                        exprs.append(f"CAST(({rewritten}) AS {base_dt}){coll_tail}")
                    else:
                        exprs.append(f"({rewritten}){coll_tail}")
                else:
                    exprs.append(f"{rec}.{q_c}")
            return f"SELECT {', '.join(exprs)} FROM {tbl} AS {rec}"

        out: Dict[str, str] = {}
        for ev in EVENTS:
            if ev == "INSERT":
                body = _branch_select(1, "__mv_n", PG_NEW_TABLE)
            elif ev == "DELETE":
                body = _branch_select(-1, "__mv_o", PG_OLD_TABLE)
            else:
                body = (
                    f"{_branch_select(-1, '__mv_o', PG_OLD_TABLE)}\n"
                    f"            UNION ALL\n"
                    f"            {_branch_select(1, '__mv_n', PG_NEW_TABLE)}"
                )
            out[ev] = f"INSERT INTO {stg_tbl} ({', '.join(ins_cols)})\n            {body};"
        return out

    def _pg_reconcile_fn_ddl(
        self,
        fn_name: str,
        lock_table: str,
        stage_tables: Sequence[str],
        reconcile_body_sql: str,
    ) -> str:
        lock_reg = "'" + lock_table.replace("'", "''") + "'"
        any_staged = " OR ".join(
            f"EXISTS (SELECT 1 FROM {st} WHERE __mv_tx = pg_catalog.txid_current() AND __mv_seq = __mv_blk.__mv_bseq)"
            for st in stage_tables
        )
        cleanup = "\n    ".join(
            f"DELETE FROM {st} WHERE __mv_tx = pg_catalog.txid_current() AND __mv_seq = __mv_blk.__mv_bseq;"
            for st in stage_tables
        )
        return render_procedural_sql(
            "pg_reconcile_function",
            fn_name=fn_name,
            lock_reg=lock_reg,
            any_staged_cond=any_staged,
            reconcile_body_sql=reconcile_body_sql.strip(),
            cleanup_sql=cleanup,
        )

    # ------------------------------------------------------------------
    # Shared SQL helpers
    # ------------------------------------------------------------------
    def _pg_rewrite_vgen_expr(self, expr: str, ts: TableSchema, record_prefix: str) -> str:
        """Qualify bare/quoted non-virtual column references in a PostgreSQL
        virtual generated column's expression with ``record_prefix``."""
        import sqlglot.errors
        from sqlglot import TokenType as SQLGlotTokenType, exp as sqlglot_exp
        from ..engines import Dialect
        from ..sql_frontend import _tokenize_sql, parse_sql_ast, quote_sql_identifier

        non_vgen_by_exact = {c: c for c, cd in ts.columns.items() if not cd.generated_virtual}
        non_vgen_by_lower = {c.lower(): c for c, cd in ts.columns.items() if not cd.generated_virtual}

        # Primary path: sqlglot AST rewrite of unqualified Column nodes only.
        # Avoids touching SQL keywords (CASE/END/FROM/FOR/NULL/TRUE/ARRAY),
        # multi-word types (double precision, character varying, timestamp with
        # time zone), EXTRACT/SUBSTRING/TRIM clauses, typed literals, and
        # named function parameters. Fall back to the token scanner only if
        # PG-specific syntax like OPERATOR(...) cannot round-trip cleanly.
        if not re.search(r"\bOPERATOR\s*\(", expr, re.IGNORECASE):
            try:
                ast = parse_sql_ast(expr, dialect=Dialect.POSTGRES)
                changed = False
                for col_node in list(ast.find_all(sqlglot_exp.Column)):
                    if col_node.table:
                        continue
                    ident = col_node.this
                    if not isinstance(ident, sqlglot_exp.Identifier) or not ident.name:
                        continue
                    if ident.quoted:
                        actual_c = non_vgen_by_exact.get(ident.name)
                    else:
                        actual_c = non_vgen_by_lower.get(ident.name.lower())
                    if actual_c is None:
                        continue
                    col_node.set("table", sqlglot_exp.Identifier(this=record_prefix, quoted=False))
                    col_node.set("this", sqlglot_exp.Identifier(this=actual_c, quoted=True))
                    changed = True
                if changed:
                    return ast.sql(dialect="postgres")
                return expr
            except sqlglot.errors.SqlglotError:
                pass

        # Fallback token scanner (e.g. for PG OPERATOR(pg_catalog.+) syntax).
        try:
            toks = _tokenize_sql(expr, read="postgres")
        except Exception:
            return expr
        _PG_TYPE_CONTINUATION = frozenset({
            "precision", "varying", "with", "without", "time", "zone",
            "year", "month", "day", "hour", "minute", "second", "to",
            "character", "double", "bit", "timestamp", "interval",
        })
        out: List[str] = []
        cursor = 0
        n = len(toks)
        in_dcolon_type = False
        dcolon_paren_depth = 0
        cast_as_depths: List[int] = []
        paren_depth = 0
        for i, tk in enumerate(toks):
            tt = tk.token_type
            if tt == SQLGlotTokenType.L_PAREN:
                paren_depth += 1
                if in_dcolon_type:
                    dcolon_paren_depth += 1
                continue
            if tt == SQLGlotTokenType.R_PAREN:
                if in_dcolon_type and dcolon_paren_depth > 0:
                    dcolon_paren_depth -= 1
                if cast_as_depths and paren_depth == cast_as_depths[-1]:
                    cast_as_depths.pop()
                paren_depth = max(0, paren_depth - 1)
                continue
            if tt == SQLGlotTokenType.DCOLON:
                in_dcolon_type = True
                dcolon_paren_depth = 0
                continue
            if tt == SQLGlotTokenType.ALIAS and i >= 2:
                # Inside CAST(expr AS <type>): skip type tokens until matching ')'.
                cast_as_depths.append(paren_depth)
                continue
            if in_dcolon_type:
                if dcolon_paren_depth > 0 or tt in (SQLGlotTokenType.DOT, SQLGlotTokenType.L_BRACKET, SQLGlotTokenType.R_BRACKET):
                    continue
                if i > 0 and toks[i - 1].token_type in (SQLGlotTokenType.DCOLON, SQLGlotTokenType.DOT):
                    continue
                if tk.text and tk.text.lower() in _PG_TYPE_CONTINUATION:
                    continue
                in_dcolon_type = False
            if cast_as_depths:
                continue

            is_quoted = tk.start < len(expr) and expr[tk.start] == '"'
            if not is_quoted and tt != SQLGlotTokenType.VAR:
                continue
            if is_quoted and tt not in (SQLGlotTokenType.IDENTIFIER, SQLGlotTokenType.VAR):
                continue
            prev_tt = toks[i - 1].token_type if i > 0 else None
            next_tt = toks[i + 1].token_type if i + 1 < n else None
            if prev_tt in (SQLGlotTokenType.DOT, SQLGlotTokenType.COLLATE):
                continue
            if next_tt in (
                SQLGlotTokenType.DOT,
                SQLGlotTokenType.L_PAREN,
                SQLGlotTokenType.STRING,
                SQLGlotTokenType.FARROW,
                SQLGlotTokenType.COLON_EQ,
            ):
                continue
            if not is_quoted and next_tt == SQLGlotTokenType.L_BRACKET and tk.text.lower() == "array":
                continue
            if not is_quoted and prev_tt == SQLGlotTokenType.L_PAREN and next_tt == SQLGlotTokenType.FROM:
                continue
            actual_c = non_vgen_by_exact.get(tk.text) if is_quoted else non_vgen_by_lower.get(tk.text.lower())
            if actual_c is None:
                continue
            if tk.start > cursor:
                out.append(expr[cursor : tk.start])
            out.append(
                f"{record_prefix}.{quote_sql_identifier(actual_c, dialect=Dialect.POSTGRES, force=True, allow_qualified=False)}"
            )
            cursor = tk.end + 1
        if cursor < len(expr):
            out.append(expr[cursor:])
        return "".join(out)

    def _pg_signed_rows_sql(self, event: str, base_t: Optional[str] = None) -> str:
        """Signed delta of one statement over ``base_t``'s transition tables.

        For ordinary tables (no PostgreSQL 18 VIRTUAL generated columns) this
        emits ``SELECT ±1 AS __mv_s, __mv_n.*``. When ``base_t`` has any
        VIRTUAL generated column, PostgreSQL 18 forbids reading it from a
        transition table (``unexpected virtual generated column reference``),
        so we project every non-virtual column explicitly and compute each
        virtual column from its generation expression over the transition row.
        """
        from sqlglot import exp as sqlglot_exp

        from ..engines import Dialect
        from ..sql_frontend import parse_sql_ast, quote_sql_identifier

        ts = self._schemas_lower.get(base_t.lower()) if base_t else None
        if ts is None or not any(c.generated_virtual for c in ts.columns.values()):
            return signed_rows_sql(event)

        raw_sql = self.view.raw_select_sql or ""
        raw_sql_lower = raw_sql.lower()
        uses_star = False
        if raw_sql:
            try:
                parsed_view = parse_sql_ast(raw_sql, dialect=Dialect.POSTGRES)
                uses_star = any(
                    isinstance(star, sqlglot_exp.Star) and not isinstance(star.parent, sqlglot_exp.Count)
                    for star in parsed_view.find_all(sqlglot_exp.Star)
                )
            except Exception:
                uses_star = bool(
                    re.search(
                        r"\bselect\s+(?:distinct\s+)?(?:\*|\S+\.\*)|,\s*(?:\*|\S+\.\*)",
                        raw_sql_lower,
                    )
                )

        def _select_list(rec: str) -> str:
            items: List[str] = []
            for c_def in ts.columns.values():
                q_c = quote_sql_identifier(c_def.name, dialect=Dialect.POSTGRES, force=True, allow_qualified=False)
                if not c_def.generated_virtual:
                    items.append(f"{rec}.{q_c} AS {q_c}")
                    continue
                if c_def.generation_expr:
                    rewritten = self._pg_rewrite_vgen_expr(c_def.generation_expr, ts, rec)
                    dt_raw = (c_def.data_type or "").strip()
                    m_coll = re.search(
                        r"(\s+COLLATE\s+(?:(?:\"[^\"]+\"|\S+)\.)?(?:\"[^\"]+\"|\S+))\s*$",
                        dt_raw,
                        re.IGNORECASE,
                    )
                    if m_coll:
                        base_dt = dt_raw[: m_coll.start()].strip()
                        coll_tail = m_coll.group(1)
                    else:
                        base_dt = dt_raw
                        coll_tail = ""
                    if base_dt and base_dt.upper() not in ("NULL", "USER-DEFINED", "ARRAY"):
                        items.append(f"CAST(({rewritten}) AS {base_dt}){coll_tail} AS {q_c}")
                    else:
                        items.append(f"({rewritten}){coll_tail} AS {q_c}")
                else:
                    # Check whether the view actually mentions this virtual column
                    # (or uses SELECT * / tbl.*). Ignore COUNT(*) which does not read columns.
                    col_re = re.compile(
                        rf'(?<![A-Za-z0-9_$])"?{re.escape(c_def.name.lower())}"?(?![A-Za-z0-9_$])'
                    )
                    if uses_star or col_re.search(raw_sql_lower):
                        raise UnsupportedMaterializedViewError(
                            f"PostgreSQL virtual generated column {ts.name}.{c_def.name} cannot be read from "
                            f"statement-trigger transition tables because its generation expression is unavailable."
                        )
            return ", ".join(items)

        def _signed(sign: int, rec: str, tbl: str) -> str:
            sel = _select_list(rec)
            cols_sql = f"{sign} AS {SIGN}, {sel}" if sel else f"{sign} AS {SIGN}"
            return f"SELECT {cols_sql} FROM {tbl} AS {rec}"

        new = _signed(1, "__mv_n", PG_NEW_TABLE)
        old = _signed(-1, "__mv_o", PG_OLD_TABLE)
        if event == "INSERT":
            return new
        if event == "DELETE":
            return old
        if event == "UPDATE":
            return f"{old}\n        UNION ALL\n        {new}"
        raise ValueError(f"no transition tables for event {event!r}")

    def _pg_projection_sql(
        self,
        event: str,
        sources: Sequence[PgDeltaSource],
        columns: Callable[[PgDeltaSource], List[str]],
        base_t: Optional[str] = None,
    ) -> str:
        """``(<branch> UNION ALL ...) AS __mv_p``: each source's signed rows,
        filtered by its predicate and projected to ``columns(source)``."""
        signed = self._pg_signed_rows_sql(event, base_t)
        parts: List[str] = []
        for src in sources:
            cols = [f"{src.sign} AS {SIGN}", *columns(src)]
            where = f"\n        WHERE {src.where}" if src.where else ""
            parts.append(f"SELECT {', '.join(cols)}\n        FROM {src.from_sql(signed)}{where}")
        return "(\n        " + "\n        UNION ALL\n        ".join(parts) + "\n    ) AS __mv_p"

    def _pg_repair_snapshot_fence_sql(self) -> str:
        """Fence pre-repair REPEATABLE READ / SERIALIZABLE snapshots against stale maintenance.

        Under READ COMMITTED this is a single GUC check (no locks). Under
        REPEATABLE READ or SERIALIZABLE, ``SELECT ... FOR SHARE`` on the
        view's catalog row succeeds concurrently for writers (``ROW SHARE`` is
        compatible with ``FOR SHARE`` and with ``repair_view()``'s catalog
        ``SHARE ROW EXCLUSIVE`` table lock), fails closed with SQLSTATE
        ``40001`` if ``repair_view()``, rebuild, or backfill updated the
        catalog row after the transaction's snapshot started, and raises
        ``40001`` (``NOT FOUND``) if the view was created or replaced after
        the snapshot started.

        Note on chunked ``backfill_incrementally(snapshot_isolation=False)``:
        a chunk locks backing rows before updating the catalog watermark,
        whereas a concurrent ``REPEATABLE READ`` / ``SERIALIZABLE`` writer
        acquires ``FOR SHARE`` on the catalog row before locking backing rows,
        so a concurrent overlap can resolve as ``40P01`` (deadlock) instead of
        ``40001``; ``_run_pk_range_batch`` retries ``40P01`` and application
        transactions must retry on both ``40001`` and ``40P01`` anyway.
        ``READ COMMITTED`` writers never enter this fence.
        """
        if not self.repair_fence_catalog_table:
            return ""
        view_lit = self.db_view_name.replace("'", "''")
        return (
            "/* __MV_REPAIR_FENCE_START__ */\n"
            "    IF pg_catalog.current_setting('transaction_isolation') IN ('repeatable read', 'serializable') THEN\n"
            f"        PERFORM 1 FROM {self.repair_fence_catalog_table} WHERE view_name = '{view_lit}' FOR SHARE;\n"
            "        IF NOT FOUND THEN\n"
            "            RAISE EXCEPTION 'matview: materialized view % was created or rebuilt after this % transaction started; retry the transaction',\n"
            f"                '{view_lit}', pg_catalog.current_setting('transaction_isolation')\n"
            "                USING ERRCODE = 'serialization_failure';\n"
            "        END IF;\n"
            "    END IF;\n"
            "    /* __MV_REPAIR_FENCE_END__ */"
        )

    def _pg_watermark_predicate(self, record: str, pk_col: Optional[str] = None) -> Optional[str]:
        """Skip source rows inside an active pk-range backfill window.

        The backfill folds those rows itself (``backfill_incrementally``);
        ``pk_col`` defaults to the first integer primary-key column of the
        view's primary table. ``None`` when no watermark applies.
        """
        if not self.watermark_catalog_table:
            return None
        if pk_col is None:
            pk_col = self._resolve_pk_range_column(allow_sqlite_rowid=False)
        if not pk_col:
            return None
        q_pk = self._quote_pk_range_col(pk_col)
        view_lit = self.db_view_name.replace("'", "''")
        return (
            f"NOT EXISTS (SELECT 1 FROM {self.watermark_catalog_table} "
            f"WHERE view_name = '{view_lit}' AND backfill_active = 1 "
            f"AND (backfill_cursor_pk IS NULL OR ({record}.{q_pk} > backfill_cursor_pk AND {record}.{q_pk} <= backfill_max_pk)) "
            f"FOR SHARE)"
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
            return build([
                f"({l} = {r} OR ({l} IS NOT DISTINCT FROM NULL AND {r} IS NOT DISTINCT FROM NULL))"
                for l, r in nullable_keys
            ])
        branches: List[Tuple[str, str]] = []
        for mask in itertools.product((False, True), repeat=len(nullable_keys)):
            when = " AND ".join(
                f"{r} IS NOT DISTINCT FROM NULL" if is_null else f"{r} IS DISTINCT FROM NULL"
                for (_l, r), is_null in zip(nullable_keys, mask)
            )
            conds = [
                f"{l} IS NOT DISTINCT FROM NULL" if is_null else f"{l} = {r}"
                for (l, r), is_null in zip(nullable_keys, mask)
            ]
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
        self,
        spec: PgAggregateSpec,
        sources: Sequence[PgDeltaSource],
        pre_stmts: Sequence[str] = (),
        base_t: Optional[str] = None,
        force_recompute_extremes: bool = False,
    ) -> Dict[str, str]:
        """Set-based bodies for a backing table with one row per group."""
        return {
            event: self._pg_aggregate_body(
                spec,
                sources,
                event,
                pre_stmts=pre_stmts,
                base_t=base_t,
                force_recompute_extremes=force_recompute_extremes,
            )
            for event in EVENTS
        }

    def _pg_aggregate_body(
        self,
        spec: PgAggregateSpec,
        sources: Sequence[PgDeltaSource],
        event: str,
        pre_stmts: Sequence[str] = (),
        base_t: Optional[str] = None,
        force_recompute_extremes: bool = False,
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
        proj = self._pg_projection_sql(event, sources, _columns, base_t=base_t)
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
            return f"SUM(CASE WHEN ({val}) IS DISTINCT FROM NULL THEN {s} ELSE 0 END)"

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
            elif (
                it.is_distinct
                or spec.native_recompute(it)
                or fn not in _FOLDED_AGGS
                or (force_recompute_extremes and fn in (AggFunc.MIN, AggFunc.MAX, AggFunc.BIT_AND, AggFunc.BIT_OR))
            ):
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
                    f"CASE WHEN EXCLUDED.{col} IS NOT DISTINCT FROM NULL THEN {mv}.{col} "
                    f"WHEN {mv}.{col} IS NOT DISTINCT FROM NULL OR EXCLUDED.{col} {op} {mv}.{col} THEN EXCLUDED.{col} "
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

        stmts: List[str] = list(pre_stmts)
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
                    f"    WHERE {val} IS DISTINCT FROM NULL\n"
                    f"    GROUP BY {', '.join(aux_pk_exprs)}, {val}\n"
                    f"    ORDER BY {', '.join(aux_pk_exprs)}, {val}\n"
                    f"    ON CONFLICT ({', '.join(q_aux_keys)}, __val) DO UPDATE SET\n"
                    f"        __ref_cnt = {aux_tbl}.__ref_cnt + EXCLUDED.__ref_cnt;"
                )
                if has_old:
                    aux_refs = [f"{aux_tbl}.{c}" for c in q_aux_keys] + [f"{aux_tbl}.__val"]
                    stmts.append(
                        f"DELETE FROM {aux_tbl}\n"
                        f"    WHERE {self._pg_key_match(aux_refs, f'SELECT {chr(44).join(aux_pk_exprs)}, {val} FROM {proj} WHERE {s} < 0 AND {val} IS DISTINCT FROM NULL')}\n"
                        f"    AND {aux_tbl}.__ref_cnt <= 0;"
                    )
                touched.append(f"{val} IS DISTINCT FROM NULL")
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

        if grp_exprs and (has_old or force_recompute_extremes):
            del_filter = "" if force_recompute_extremes else f" WHERE {s} < 0"
            stmts.append(
                f"DELETE FROM {mv}\n"
                f"    WHERE {self._pg_key_match(mv_pk_refs, f'SELECT {pk_select} FROM {proj}{del_filter}')}\n"
                f"    AND {mv}.{q_rc} <= 0;"
            )

        if force_recompute_extremes:
            # In multi-source inclusion-exclusion, a newly inserted group whose
            # non-NULL values net out to count 0 (e.g., an orphan row whose value
            # changed to NULL in the same statement its parent was inserted) has
            # COUNT(val) > 0 in the projection, so INSERT's sum_delta evaluates
            # to 0 instead of NULL. Reset any surviving SUM / hidden-sum column
            # whose non-NULL count is 0 back to NULL.
            zero_sum_pairs: List[Tuple[str, str]] = []
            seen_zsum: set = set()
            for it in items:
                if (
                    it.is_aggregate
                    and it.agg_func == AggFunc.SUM
                    and not it.is_distinct
                    and not spec.native_recompute(it)
                ):
                    nn = spec.nonnull_map.get(spec.value_key(it))
                    if nn is not None and q(it.alias) not in seen_zsum:
                        seen_zsum.add(q(it.alias))
                        zero_sum_pairs.append((q(it.alias), q(nn)))
            for vk, sm in spec.sum_map.items():
                if vk not in spec.distinct_aux and q(sm) not in seen_zsum:
                    nn = spec.nonnull_map.get(vk)
                    if nn is not None:
                        seen_zsum.add(q(sm))
                        zero_sum_pairs.append((q(sm), q(nn)))
            if zero_sum_pairs:
                z_sets = [
                    f"{sc} = CASE WHEN {mv}.{nc} = 0 THEN NULL ELSE {mv}.{sc} END"
                    for sc, nc in zero_sum_pairs
                ]
                z_cond = " OR ".join(
                    f"({mv}.{nc} = 0 AND {mv}.{sc} IS NOT NULL)" for sc, nc in zero_sum_pairs
                )
                stmts.append(
                    f"UPDATE {mv} SET\n        {sep.join(z_sets)}\n"
                    f"    WHERE {self._pg_key_match(mv_pk_refs, f'SELECT {pk_select} FROM {proj}')}\n"
                    f"    AND ({z_cond});"
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
                cond = (
                    f"(__mv_r.{rcol} IS DISTINCT FROM NULL AND "
                    f"({mv}.{col} IS NOT DISTINCT FROM NULL OR __mv_r.{rcol} {op} {mv}.{col}))"
                )
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
        if (has_old or force_recompute_extremes) and spec.recompute_group_keys is not None and grp_exprs:
            var_scale_check = spec.has_variable_numeric_scale or self._pg_expr_has_variable_numeric_scale
            rep_targets: List[Tuple[str, str, str]] = []
            seen_rep_cols: set = set()
            for g_idx, g_expr in enumerate(grp_exprs):
                if var_scale_check(g_expr):
                    mc = self._group_col_alias(g_expr)
                    if mc.lower() not in seen_rep_cols:
                        seen_rep_cols.add(mc.lower())
                        rep_targets.append((g_expr, q(mc), keys[g_idx]))
            for it in items:
                if not it.is_aggregate and it.alias.lower() in extra_of_alias:
                    if it.alias.lower() not in seen_rep_cols and var_scale_check(it.raw_expr):
                        seen_rep_cols.add(it.alias.lower())
                        rep_targets.append((it.raw_expr, q(it.alias), f"__mv_p.{extra_of_alias[it.alias.lower()]}"))
            if rep_targets:
                rep_conds: List[str] = []
                if force_recompute_extremes:
                    rep_conds.append("TRUE")
                else:
                    for r_idx, (_g_expr, col, k_expr) in enumerate(rep_targets):
                        dcol = f"__mv_gd{r_idx}"
                        icol = f"__mv_gi{r_idx}"
                        rec_cols.append(
                            f"pg_catalog.array_agg(DISTINCT ROW({k_expr})::pg_catalog.text) "
                            f"FILTER (WHERE {s} < 0 AND ({k_expr}) IS DISTINCT FROM NULL) AS {dcol}"
                        )
                        rec_cols.append(
                            f"pg_catalog.array_agg(DISTINCT ROW({k_expr})::pg_catalog.text) "
                            f"FILTER (WHERE {s} > 0 AND ({k_expr}) IS DISTINCT FROM NULL) AS {icol}"
                        )
                        # Refresh whenever a row with the stored representation was
                        # deleted OR updated (an UPDATE writes a new tuple version
                        # at a new ctid, changing scan-order tie winners even when
                        # the key's scale did not change), OR when an inserted/new
                        # row carries a different representation for the same group.
                        rep_conds.append(
                            f"({mv}.{col} IS DISTINCT FROM NULL AND ("
                            f"ROW({mv}.{col})::pg_catalog.text = ANY(__mv_r.{dcol}) OR "
                            f"(__mv_r.{icol} IS NOT NULL AND "
                            f"NOT (ARRAY[ROW({mv}.{col})::pg_catalog.text] @> __mv_r.{icol}))))"
                        )
                any_rep_cond = "TRUE" if force_recompute_extremes else " OR ".join(rep_conds)
                inner_lat = spec.recompute_group_keys([expr_sql for expr_sql, _, _ in rep_targets])
                if len(rep_targets) == 1:
                    col = rep_targets[0][1]
                    if force_recompute_extremes:
                        rec_assigns.append(f"{col} = COALESCE(({inner_lat}), {mv}.{col})")
                    else:
                        rec_assigns.append(
                            f"{col} = CASE WHEN {any_rep_cond} THEN COALESCE(({inner_lat}), {mv}.{col}) ELSE {mv}.{col} END"
                        )
                else:
                    lhs_cols = ", ".join(col for _, col, _ in rep_targets)
                    sel_cols = ", ".join(
                        f"COALESCE(__mv_l.__mv_l{r_idx}, {mv}.{col})"
                        for r_idx, (_, col, _) in enumerate(rep_targets)
                    )
                    rec_assigns.append(
                        f"({lhs_cols}) = (SELECT {sel_cols} FROM (SELECT 1) AS __mv_1 "
                        f"LEFT JOIN LATERAL ({inner_lat}) AS __mv_l ON ({any_rep_cond}))"
                    )
                rec_conds.append(any_rep_cond)
        elif not has_old and not force_recompute_extremes and spec.recompute_group_keys is not None and grp_exprs:
            # INSERT: refresh only when an inserted row introduces a different
            # representation for an already-existing group (e.g. INSERT into a
            # VACUUM-freed earlier heap slot or branch-0 of a UNION ALL GROUP BY).
            var_scale_check = spec.has_variable_numeric_scale or self._pg_expr_has_variable_numeric_scale
            rep_targets_ins: List[Tuple[str, str, str]] = []
            seen_rep_ins: set = set()
            for g_idx, g_expr in enumerate(grp_exprs):
                if var_scale_check(g_expr):
                    mc = self._group_col_alias(g_expr)
                    if mc.lower() not in seen_rep_ins:
                        seen_rep_ins.add(mc.lower())
                        rep_targets_ins.append((g_expr, q(mc), keys[g_idx]))
            for it in items:
                if not it.is_aggregate and it.alias.lower() in extra_of_alias:
                    if it.alias.lower() not in seen_rep_ins and var_scale_check(it.raw_expr):
                        seen_rep_ins.add(it.alias.lower())
                        rep_targets_ins.append((it.raw_expr, q(it.alias), f"__mv_p.{extra_of_alias[it.alias.lower()]}"))
            if rep_targets_ins:
                ins_rep_conds: List[str] = []
                for r_idx, (_g_expr, col, k_expr) in enumerate(rep_targets_ins):
                    icol = f"__mv_gi{r_idx}"
                    rec_cols.append(
                        f"pg_catalog.array_agg(DISTINCT ROW({k_expr})::pg_catalog.text) "
                        f"FILTER (WHERE {s} > 0 AND ({k_expr}) IS DISTINCT FROM NULL) AS {icol}"
                    )
                    ins_rep_conds.append(
                        f"({mv}.{col} IS DISTINCT FROM NULL AND __mv_r.{icol} IS NOT NULL AND "
                        f"NOT (ARRAY[ROW({mv}.{col})::pg_catalog.text] @> __mv_r.{icol}))"
                    )
                any_ins_rep_cond = " OR ".join(ins_rep_conds)
                inner_lat = spec.recompute_group_keys([expr_sql for expr_sql, _, _ in rep_targets_ins])
                if len(rep_targets_ins) == 1:
                    col = rep_targets_ins[0][1]
                    rec_assigns.append(
                        f"{col} = CASE WHEN {any_ins_rep_cond} THEN COALESCE(({inner_lat}), {mv}.{col}) ELSE {mv}.{col} END"
                    )
                else:
                    lhs_cols = ", ".join(col for _, col, _ in rep_targets_ins)
                    sel_cols = ", ".join(
                        f"COALESCE(__mv_l.__mv_l{r_idx}, {mv}.{col})"
                        for r_idx, (_, col, _) in enumerate(rep_targets_ins)
                    )
                    rec_assigns.append(
                        f"({lhs_cols}) = (SELECT {sel_cols} FROM (SELECT 1) AS __mv_1 "
                        f"LEFT JOIN LATERAL ({inner_lat}) AS __mv_l ON ({any_ins_rep_cond}))"
                    )
                rec_conds.append(any_ins_rep_cond)
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
        base_t: Optional[str] = None,
        refresh_key_cols: Sequence[str] = (),
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
        q_refresh_keys = [q(c) for c in refresh_key_cols]

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
                old_keys = f"SELECT {k_sel} FROM {self._pg_projection_sql('DELETE', sources, _key_columns, base_t=base_t)}"
                cond = self._pg_key_match(mv_refs, old_keys)
                if event == "UPDATE":
                    new_keys = f"SELECT {k_sel} FROM {self._pg_projection_sql('INSERT', sources, _key_columns, base_t=base_t)}"
                    cond = f"{cond}\n    AND NOT {self._pg_key_match(mv_refs, new_keys)}"
                stmts.append(f"DELETE FROM {mv_table}\n    WHERE {cond};")
            if event_has_new(event):
                proj = self._pg_projection_sql("INSERT", sources, _row_columns, base_t=base_t)
                cols = q_keys + q_vals + ([q(count_col)] if count_col else [])
                vals = [f"__mv_p.__mv_k{i}" for i in range(len(key_cols))] + [
                    f"__mv_p.__mv_c{j}" for j in range(len(value_cols))
                ] + (["1"] if count_col else [])
                sets = [f"{c} = EXCLUDED.{c}" for c in q_refresh_keys] + [f"{c} = EXCLUDED.{c}" for c in q_vals]
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

    def _pg_keyed_rows_reconcile_body(
        self,
        mv_table: str,
        branches: Sequence[Tuple[str, str, Sequence[str], Sequence[str], Optional[Tuple[str, str]], Sequence[str], Mapping[str, str], Optional[str], Optional[str]]],
        *,
        count_col: Optional[str] = None,
        refresh_key_cols: Sequence[str] = (),
    ) -> str:
        """Reconcile native-PK keyed rows against live base tables for touched PKs in staging tables.

        Each branch tuple is ``(base_t, stg_tbl, src_pk_cols, mv_pk_cols,
        const_prefix_or_none, value_cols, values_map, where_sql, del_wm_sql)``
        where ``values_map`` and ``where_sql`` reference record alias ``__mv_d``
        and ``const_prefix_or_none`` is ``(mv_col, sql_literal)`` for UNION ALL
        ``__mv_src_id`` branches.
        """
        q = self._q
        q_refresh_keys = [q(c) for c in refresh_key_cols]
        stmts: List[str] = []
        for (
            base_t,
            stg_tbl,
            src_pk_cols,
            mv_pk_cols,
            const_prefix,
            value_cols,
            values_map,
            where_sql,
            del_wm_sql,
        ) in branches:
            q_src_pks = [self._pg_quote_raw_col(c) for c in src_pk_cols]
            q_mv_pks = [q(c) for c in mv_pk_cols]
            q_vals = [q(c) for c in value_cols]
            stg_wm = f" AND ({del_wm_sql})" if del_wm_sql else ""
            stg_pks_subq = (
                f"SELECT {', '.join(f'__mv_d.{c}' for c in q_src_pks)} "
                f"FROM {stg_tbl} AS __mv_d "
                f"WHERE __mv_d.__mv_tx = pg_catalog.txid_current() AND __mv_d.__mv_seq = __mv_blk.__mv_bseq{stg_wm}"
            )
            if const_prefix is not None:
                p_col, p_lit = const_prefix
                del_mv_refs = [f"{mv_table}.{c}" for c in q_mv_pks[1:]]
                del_cond = f"{mv_table}.{q(p_col)} = {p_lit} AND {self._pg_key_match(del_mv_refs, stg_pks_subq)}"
            else:
                del_mv_refs = [f"{mv_table}.{c}" for c in q_mv_pks]
                del_cond = self._pg_key_match(del_mv_refs, stg_pks_subq)
            stmts.append(f"DELETE FROM {mv_table}\n    WHERE {del_cond};")

            src_refs = [f"__mv_d.{c}" for c in q_src_pks]
            ins_where = self._pg_key_match(src_refs, stg_pks_subq)
            if where_sql:
                ins_where = f"({ins_where}) AND ({where_sql})"
            cols = q_mv_pks + q_vals + ([q(count_col)] if count_col else [])
            key_exprs = ([const_prefix[1]] if const_prefix is not None else []) + src_refs
            val_exprs = key_exprs + [values_map[c] for c in value_cols] + (["1"] if count_col else [])
            sets = [f"{c} = EXCLUDED.{c}" for c in q_refresh_keys] + [f"{c} = EXCLUDED.{c}" for c in q_vals]
            if count_col:
                sets.append(f"{q(count_col)} = 1")
            conflict = f"DO UPDATE SET {', '.join(sets)}" if sets else "DO NOTHING"
            ord_cols = ", ".join(src_refs)
            stmts.append(
                f"INSERT INTO {mv_table} ({', '.join(cols)})\n"
                f"    SELECT {', '.join(val_exprs)}\n"
                f"    FROM {q(base_t)} AS __mv_d\n"
                f"    WHERE {ins_where}\n"
                f"    ORDER BY {ord_cols}\n"
                f"    ON CONFLICT ({', '.join(q_mv_pks)}) {conflict};"
            )
        return "\n".join(stmts)

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
        base_t: Optional[str] = None,
        recompute_val_col_names: Sequence[str] = (),
        recompute_val_row_sql: Optional[str] = None,
        surviving_cond: Optional[str] = None,
        adopt_on_cnt0_cols: Sequence[str] = (),
    ) -> Dict[str, str]:
        return {
            event: self._pg_counted_rows_body(
                mv_table,
                key_col,
                value_cols,
                count_cols,
                sources,
                empty_condition,
                event,
                base_t=base_t,
                recompute_val_col_names=recompute_val_col_names,
                recompute_val_row_sql=recompute_val_row_sql,
                surviving_cond=surviving_cond,
                adopt_on_cnt0_cols=adopt_on_cnt0_cols,
            )
            for event in EVENTS
        }

    def _pg_counted_rows_body(
        self,
        mv_table: str,
        key_col: str,
        value_cols: Sequence[str],
        count_cols: Sequence[str],
        sources: Sequence[Tuple[PgDeltaSource, Mapping[str, str]]],
        empty_condition: str,
        event: str,
        base_t: Optional[str] = None,
        recompute_val_col_names: Sequence[str] = (),
        recompute_val_row_sql: Optional[str] = None,
        surviving_cond: Optional[str] = None,
        adopt_on_cnt0_cols: Sequence[str] = (),
        force_recompute_vals: bool = False,
    ) -> str:
        """Body for a backing table with one row per distinct value tuple.

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
        adopt_set = {c.lower() for c in adopt_on_cnt0_cols}
        has_cnt0 = any(c == "__mv_cnt_0" for c in count_cols)
        recompute_set = {c.lower() for c in recompute_val_col_names}

        def _columns(src: PgDeltaSource) -> List[str]:
            contrib = contrib_of[id(src)]
            cols = [f"{src.keys[0]} AS __mv_k0"]
            cols += [f"{src.values[c]} AS __mv_c{j}" for j, c in enumerate(value_cols)]
            cols += [f"{contrib[c]} AS __mv_n{j}" for j, c in enumerate(count_cols)]
            return cols

        s = f"__mv_p.{SIGN}"
        proj = self._pg_projection_sql(event, srcs, _columns, base_t=base_t)
        sets = [f"{c} = {mv_table}.{c} + EXCLUDED.{c}" for c in q_cnts]
        if adopt_set and has_cnt0:
            adopt_sets = [
                f"{qc} = CASE WHEN {mv_table}.__mv_cnt_0 <= 0 AND EXCLUDED.__mv_cnt_0 > 0 "
                f"THEN EXCLUDED.{qc} ELSE {mv_table}.{qc} END"
                for c, qc in zip(value_cols, q_vals)
                if c.lower() in adopt_set
            ]
            sets = adopt_sets + sets
            # For INTERSECT / EXCEPT with variable-representation columns
            # (unconstrained NUMERIC/NUMERIC[]/BPCHAR/BPCHAR[]/NUMRANGE),
            # use DISTINCT ON over a CTE instead of array_agg(...)[1] so
            # array-typed columns keep their exact type and dimensionality
            # while preferring branch-0 positive rows over branch-1 rows.
            proj_cte_body = proj[: -len(" AS __mv_p")] if proj.endswith(" AS __mv_p") else f"({proj})"
            cnt_aggs = [f"SUM({SIGN} * __mv_n{j}) AS __mv_sum_{j}" for j in range(len(count_cols))]
            rep_cols = [f"__mv_c{j}" for j in range(len(value_cols))]
            sel_cols = (
                ["__mv_g.__mv_k0"]
                + [f"__mv_rep.__mv_c{j}" for j in range(len(value_cols))]
                + [f"__mv_g.__mv_sum_{j}" for j in range(len(count_cols))]
            )
            stmts = [
                f"WITH __mv_p AS {proj_cte_body}\n"
                f"INSERT INTO {mv_table} ({', '.join([q_key, *q_vals, *q_cnts])})\n"
                f"    SELECT {', '.join(sel_cols)}\n"
                f"    FROM (\n"
                f"        SELECT __mv_k0, {', '.join(cnt_aggs)}\n"
                f"        FROM __mv_p\n"
                f"        GROUP BY __mv_k0\n"
                f"    ) AS __mv_g\n"
                f"    JOIN (\n"
                f"        SELECT DISTINCT ON (__mv_k0) __mv_k0, {', '.join(rep_cols)}\n"
                f"        FROM __mv_p\n"
                f"        ORDER BY __mv_k0, ({SIGN} > 0 AND __mv_n0 > 0) DESC, (__mv_n0 > 0) DESC\n"
                f"    ) AS __mv_rep ON __mv_rep.__mv_k0 = __mv_g.__mv_k0\n"
                f"    ORDER BY __mv_g.__mv_k0\n"
                f"    ON CONFLICT ({q_key}) DO UPDATE SET {', '.join(sets)};"
            ]
        else:
            group_cols = ["__mv_p.__mv_k0"] + [f"__mv_p.__mv_c{j}" for j in range(len(value_cols))]
            sel_val_cols = list(group_cols)
            cnt_vals = [f"SUM({s} * __mv_p.__mv_n{j})" for j in range(len(count_cols))]
            stmts = [
                f"INSERT INTO {mv_table} ({', '.join([q_key, *q_vals, *q_cnts])})\n"
                f"    SELECT {', '.join(sel_val_cols + cnt_vals)}\n"
                f"    FROM {proj}\n"
                f"    GROUP BY {', '.join(group_cols)}\n"
                f"    ORDER BY __mv_p.__mv_k0\n"
                f"    ON CONFLICT ({q_key}) DO UPDATE SET {', '.join(sets)};"
            ]
        if event_has_old(event) or force_recompute_vals:
            del_filter = "" if force_recompute_vals else f" WHERE {s} < 0"
            stmts.append(
                f"DELETE FROM {mv_table}\n"
                f"    WHERE {mv_table}.{q_key} IN (SELECT __mv_p.__mv_k0 FROM {proj}{del_filter})\n"
                f"    AND {empty_condition};"
            )
        if recompute_set and recompute_val_row_sql:
            branch0_filter = " AND __mv_p.__mv_n0 > 0" if has_cnt0 else ""
            ref_targets: List[Tuple[int, str]] = [
                (j, q_vals[j]) for j, c in enumerate(value_cols) if c.lower() in recompute_set
            ]
            if ref_targets:
                if len(ref_targets) == 1:
                    qc = ref_targets[0][1]
                    set_clause = f"{qc} = COALESCE(({recompute_val_row_sql}), {mv_table}.{qc})"
                else:
                    lhs_cols = ", ".join(qc for _, qc in ref_targets)
                    sel_cols = ", ".join(
                        f"COALESCE(__mv_l.__mv_l{r_idx}, {mv_table}.{qc})"
                        for r_idx, (_, qc) in enumerate(ref_targets)
                    )
                    set_clause = (
                        f"({lhs_cols}) = (SELECT {sel_cols} FROM (SELECT 1) AS __mv_1 "
                        f"LEFT JOIN LATERAL ({recompute_val_row_sql}) AS __mv_l ON TRUE)"
                    )
                surv = f" AND ({surviving_cond})" if surviving_cond else ""
                if force_recompute_vals:
                    stmts.append(
                        f"UPDATE {mv_table} SET {set_clause}\n"
                        f"    WHERE {mv_table}.{q_key} IN (SELECT __mv_p.__mv_k0 FROM {proj}){surv};"
                    )
                elif event_has_old(event):
                    rec_cols: List[str] = []
                    rec_conds: List[str] = []
                    for r_idx, (j, qc) in enumerate(ref_targets):
                        dcol = f"__mv_vd{r_idx}"
                        icol = f"__mv_vi{r_idx}"
                        rec_cols.append(
                            f"pg_catalog.array_agg(DISTINCT ROW(__mv_p.__mv_c{j})::pg_catalog.text) "
                            f"FILTER (WHERE {s} < 0{branch0_filter} AND (__mv_p.__mv_c{j}) IS DISTINCT FROM NULL) AS {dcol}"
                        )
                        rec_cols.append(
                            f"pg_catalog.array_agg(DISTINCT ROW(__mv_p.__mv_c{j})::pg_catalog.text) "
                            f"FILTER (WHERE {s} > 0{branch0_filter} AND (__mv_p.__mv_c{j}) IS DISTINCT FROM NULL) AS {icol}"
                        )
                        # Refresh whenever a row with the stored representation was
                        # deleted OR updated (an UPDATE moves the tuple to a new
                        # physical ctid, changing scan-order tie winners even when
                        # its scale stayed the same), OR when an inserted/updated
                        # row carries a different representation for the same key.
                        rec_conds.append(
                            f"({mv_table}.{qc} IS DISTINCT FROM NULL AND ("
                            f"ROW({mv_table}.{qc})::pg_catalog.text = ANY(__mv_r.{dcol}) OR "
                            f"(__mv_r.{icol} IS NOT NULL AND "
                            f"NOT (ARRAY[ROW({mv_table}.{qc})::pg_catalog.text] @> __mv_r.{icol}))))"
                        )
                    any_cond = " OR ".join(rec_conds)
                    stmts.append(
                        f"UPDATE {mv_table} SET {set_clause}\n"
                        f"    FROM (\n"
                        f"        SELECT __mv_p.__mv_k0 AS __mv_key0, {', '.join(rec_cols)}\n"
                        f"        FROM {proj}\n"
                        f"        GROUP BY __mv_p.__mv_k0\n"
                        f"    ) AS __mv_r\n"
                        f"    WHERE {mv_table}.{q_key} = __mv_r.__mv_key0{surv} AND ({any_cond});"
                    )
                else:
                    # INSERT: refresh only when an inserted row introduces a
                    # different representation for an already-existing key (for
                    # example, INSERT into branch 0 of a UNION when branch 1
                    # previously supplied the representative, or into a freed
                    # earlier heap slot).
                    ins_cols: List[str] = []
                    ins_conds: List[str] = []
                    for r_idx, (j, qc) in enumerate(ref_targets):
                        icol = f"__mv_vi{r_idx}"
                        ins_cols.append(
                            f"pg_catalog.array_agg(DISTINCT ROW(__mv_p.__mv_c{j})::pg_catalog.text) "
                            f"FILTER (WHERE {s} > 0{branch0_filter} AND (__mv_p.__mv_c{j}) IS DISTINCT FROM NULL) AS {icol}"
                        )
                        ins_conds.append(
                            f"({mv_table}.{qc} IS DISTINCT FROM NULL AND __mv_r.{icol} IS NOT NULL AND "
                            f"NOT (ARRAY[ROW({mv_table}.{qc})::pg_catalog.text] @> __mv_r.{icol}))"
                        )
                    any_ins_cond = " OR ".join(ins_conds)
                    stmts.append(
                        f"UPDATE {mv_table} SET {set_clause}\n"
                        f"    FROM (\n"
                        f"        SELECT __mv_p.__mv_k0 AS __mv_key0, {', '.join(ins_cols)}\n"
                        f"        FROM {proj}\n"
                        f"        GROUP BY __mv_p.__mv_k0\n"
                        f"    ) AS __mv_r\n"
                        f"    WHERE {mv_table}.{q_key} = __mv_r.__mv_key0{surv} AND ({any_ins_cond});"
                    )
        return "\n".join(stmts)

    # ------------------------------------------------------------------
    # Recompute strategies (group-scoped and full)
    # ------------------------------------------------------------------
    def _pg_view_lock_table(self) -> str:
        p = self.object_prefix if self.object_prefix else "__matview_"
        return self._bounded_ident(f"{p}lock_", self.logical_view_name)

    def _pg_view_lock_ddl(self) -> str:
        return (
            f"CREATE TABLE {self._pg_view_lock_table()} (\n"
            f"    __mv_id BIGINT NOT NULL PRIMARY KEY,\n"
            f"    __mv_n BIGINT NOT NULL DEFAULT 0\n"
            f") WITH (fillfactor = {PG_LOCK_TABLE_FILLFACTOR});"
        )

    def _pg_view_lock_stmt(self, all_buckets: bool = False) -> str:
        """Exclusive whole-view lock for full recomputes and multi-source joins.

        READ COMMITTED: later statements of the function take a snapshot that
        includes every writer that held the lock before. REPEATABLE READ and
        SERIALIZABLE: if another transaction updated a locked row after this
        transaction's snapshot, the upsert fails with a serialization error.

        When ``all_buckets`` is True (whole-view recomputes on a group-scoped
        view, such as ``TRUNCATE`` or writes to a subquery-referenced table),
        upserts every row ``0..PG_GROUP_LOCK_BUCKETS`` in strictly ascending
        ``__mv_id`` order. Because group-scoped writers lock ascending subsets
        of ``1..PG_GROUP_LOCK_BUCKETS`` without taking ``FOR SHARE`` on row 0,
        whole-view recomputes and group-scoped writers exclude each other
        symmetrically (under READ COMMITTED, REPEATABLE READ, and SERIALIZABLE)
        with zero MultiXactId allocations.
        """
        lock_t = self._pg_view_lock_table()
        if all_buckets:
            return (
                f"INSERT INTO {lock_t} (__mv_id, __mv_n)\n"
                f"    SELECT __mv_g, 1 FROM generate_series(0, {PG_GROUP_LOCK_BUCKETS}) AS __mv_g\n"
                f"    ORDER BY 1\n"
                f"    ON CONFLICT (__mv_id) DO UPDATE SET __mv_n = {lock_t}.__mv_n + 1;"
            )
        return (
            f"INSERT INTO {lock_t} (__mv_id, __mv_n) VALUES (0, 1)\n"
            f"    ON CONFLICT (__mv_id) DO UPDATE SET __mv_n = {lock_t}.__mv_n + 1;"
        )

    def _pg_view_lock_ddls(self, with_group_buckets: bool = False) -> List[str]:
        """Lock table DDL plus its pre-seeded row(s) (id 0, and 1..B for group-scoped views)."""
        lock_t = self._pg_view_lock_table()
        if with_group_buckets:
            seed = (
                f"INSERT INTO {lock_t} (__mv_id, __mv_n) "
                f"SELECT __mv_g, 0 FROM generate_series(0, {PG_GROUP_LOCK_BUCKETS}) AS __mv_g "
                f"ON CONFLICT (__mv_id) DO NOTHING;"
            )
        else:
            seed = f"INSERT INTO {lock_t} (__mv_id, __mv_n) VALUES (0, 0) ON CONFLICT (__mv_id) DO NOTHING;"
        return [self._pg_view_lock_ddl(), seed]

    def _pg_group_lock_id_sql(self, canonical_key: str) -> str:
        """Bounded lock-table bucket id of a group (``1..PG_GROUP_LOCK_BUCKETS``, never 0).

        Equal groups get equal bucket ids because ``canonical_key`` is the
        canonical text (``_pg_key_text_fn``); unequal groups whose hashes map
        to the same bucket share a row, which only makes their writers wait
        for each other while bounding the lock table at
        ``<= PG_GROUP_LOCK_BUCKETS + 1`` rows forever.
        """
        mask = PG_GROUP_LOCK_BUCKETS - 1
        return f"((hashtextextended({canonical_key}, 0) & {mask}) + 1)"

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
        normalized, keys sorted), arrays, ranges, multiranges, and named
        composites (recursively). Types whose text output is canonical are
        returned as text. Anonymous ``record`` and other unsupported pseudo-
        types raise ``feature_not_supported`` (and are refused at registration
        by ``_pg_collation_guard_ddl``). A domain value is canonicalized as its
        base type (dynamic SQL, only on that path). Text under a
        nondeterministic collation (including inside domains, ranges,
        multiranges, arrays, or composite fields) is refused at registration
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
            "LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE SET search_path FROM CURRENT AS $mv_key$\n"
            "DECLARE\n"
            "  t pg_catalog.regtype := pg_catalog.pg_typeof(v);\n"
            "  s pg_catalog.text;\n"
            "  f pg_catalog.float8;\n"
            "  ty record;\n"
            "  comp_vals pg_catalog.text;\n"
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
            "  SELECT typ.typname, typ.typtype, typ.typcategory, typ.typbasetype, typ.typtypmod, typ.typrelid INTO ty\n"
            "    FROM pg_catalog.pg_type AS typ WHERE typ.oid = t;\n"
            "  IF ty.typtype = 'd' THEN\n"
            "    -- A domain value groups like its base type (domains share its equality).\n"
            "    EXECUTE pg_catalog.format('SELECT %s($1::%s)', "
            f"'{fn}', pg_catalog.format_type(ty.typbasetype, ty.typtypmod))\n"
            "      INTO s USING v;\n"
            "    RETURN s;\n"
            "  END IF;\n"
            "  IF ty.typcategory = 'A' THEN\n"
            "    -- Unnest in the SELECT list (not FROM) so composite array elements\n"
            "    -- stay intact instead of being expanded into individual fields.\n"
            "    SELECT COALESCE(pg_catalog.array_dims(v), '') || '{' || COALESCE(pg_catalog.string_agg(\n"
            "             CASE WHEN u.c IS NULL THEN 'N' ELSE pg_catalog.length(u.c) || ':' || u.c END, ',' ORDER BY u.o), '') || '}'\n"
            f"      INTO s FROM (SELECT {fn}(x.e) AS c, x.o FROM (\n"
            "        SELECT pg_catalog.unnest(v) AS e, pg_catalog.generate_series(1, pg_catalog.cardinality(v)) AS o\n"
            "      ) AS x) AS u;\n"
            "    RETURN s;\n"
            "  ELSIF ty.typtype = 'r' THEN\n"
            "    IF pg_catalog.isempty(v) THEN\n"
            "      RETURN 'empty';\n"
            "    END IF;\n"
            f"    s := {fn}(pg_catalog.lower(v));\n"
            "    s := (CASE WHEN pg_catalog.lower_inc(v) THEN '[' ELSE '(' END) || COALESCE(pg_catalog.length(s) || ':' || s, '') || ',';\n"
            f"    RETURN s || COALESCE(pg_catalog.length({fn}(pg_catalog.upper(v))) || ':' || {fn}(pg_catalog.upper(v)), '')\n"
            "           || (CASE WHEN pg_catalog.upper_inc(v) THEN ']' ELSE ')' END);\n"
            "  ELSIF ty.typtype = 'm' THEN\n"
            "    SELECT '{' || COALESCE(pg_catalog.string_agg(\n"
            "             pg_catalog.length(u.c) || ':' || u.c, ',' ORDER BY u.o), '') || '}'\n"
            f"      INTO s FROM (SELECT {fn}(x.e) AS c, x.o FROM pg_catalog.unnest(v) WITH ORDINALITY AS x(e, o)) AS u;\n"
            "    RETURN s;\n"
            "  ELSIF ty.typtype = 'c' AND ty.typrelid <> 0 THEN\n"
            "    SELECT pg_catalog.string_agg(\n"
            f"             pg_catalog.format('(%s, %s(($1).%I))', a.attnum, '{fn}', a.attname),\n"
            "             ', ' ORDER BY a.attnum)\n"
            "      INTO comp_vals\n"
            "      FROM pg_catalog.pg_attribute AS a\n"
            "      WHERE a.attrelid = ty.typrelid AND a.attnum > 0 AND NOT a.attisdropped;\n"
            "    IF comp_vals IS NULL THEN\n"
            "      RETURN '()';\n"
            "    END IF;\n"
            "    EXECUTE 'SELECT ''('' || pg_catalog.string_agg('\n"
            "         || 'CASE WHEN f.c IS NULL THEN ''N'' ELSE pg_catalog.length(f.c) || '':'' || f.c END, '','' ORDER BY f.o'\n"
            "         || ') || '')'' FROM (VALUES ' || comp_vals || ') AS f(o, c)'\n"
            "      INTO s USING v;\n"
            "    RETURN s;\n"
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

    def _pg_probe_queries(self) -> List[Tuple[str, str, Optional[str], bool]]:
        """Return ``[(label, probe_sql, fallback_sql, check_key_type), ...]``
        for registration-time probing of grouping, deduplication, set-operation,
        window-partition, aggregate-DISTINCT, and MIN/MAX expression types.

        Supports plain queries, CTEs (including ``WITH RECURSIVE``, ``SEARCH``,
        ``CYCLE``, and branch-local ``WITH``), compound set operations
        (``UNION``/``INTERSECT``/``EXCEPT``/``UNION ALL``, including ``*``,
        ``VALUES``, ``TABLE``, and parenthesized branches with trailing
        ``ORDER BY``/``LIMIT``), ``FROM`` subqueries, ``LATERAL`` joins,
        ``JOIN ... USING``, and ``NATURAL JOIN`` by preserving ``WITH``
        prefixes and slicing each branch's ``FROM`` clause verbatim from
        ``raw_select_sql``. Each probe query returns a single ``pg_catalog.oid``
        row at zero-row cost (``WHERE false``).
        """
        from typing import Any
        import sqlglot.errors
        from sqlglot import exp as sqlglot_exp
        from sqlglot import TokenType as SQLGlotTokenType
        from ..engines import Dialect
        from ..sql_frontend import _tokenize_sql, parse_sql_ast

        raw_sql = self.view.raw_select_sql.strip().rstrip(";").strip()
        if not raw_sql:
            return []
        try:
            toks = _tokenize_sql(raw_sql, read="postgres")
        except sqlglot.errors.SqlglotError:
            return []
        if not toks:
            return []

        def _extract_with_prefix(lo: int, hi: int) -> Tuple[int, str, bool]:
            """If ``toks[lo:hi]`` begins with ``WITH [RECURSIVE] ...``, return
            ``(body_start_idx, with_prefix_sql, is_recursive)``; otherwise
            ``(lo, "", False)``."""
            if lo >= hi or toks[lo].token_type != SQLGlotTokenType.WITH:
                return lo, "", False
            i = lo + 1
            is_rec = False
            if i < hi and toks[i].token_type == SQLGlotTokenType.RECURSIVE:
                is_rec = True
                i += 1
            while i < hi:
                # CTE name
                i += 1
                # Optional column list: cte_name(c1, c2)
                if i < hi and toks[i].token_type == SQLGlotTokenType.L_PAREN:
                    d = 0
                    while i < hi:
                        if toks[i].token_type == SQLGlotTokenType.L_PAREN:
                            d += 1
                        elif toks[i].token_type == SQLGlotTokenType.R_PAREN:
                            d -= 1
                            if d == 0:
                                i += 1
                                break
                        i += 1
                # AS [NOT] [MATERIALIZED] ( ... )
                while i < hi and toks[i].token_type != SQLGlotTokenType.L_PAREN:
                    i += 1
                if i >= hi:
                    return hi, "", is_rec
                d = 0
                while i < hi:
                    if toks[i].token_type == SQLGlotTokenType.L_PAREN:
                        d += 1
                    elif toks[i].token_type == SQLGlotTokenType.R_PAREN:
                        d -= 1
                        if d == 0:
                            i += 1
                            break
                    i += 1
                # Optional PG 14+ SEARCH {BREADTH|DEPTH} FIRST BY ... SET <col>
                # and/or CYCLE ... SET <col> [TO ... DEFAULT ...] USING <col>.
                while i < hi and toks[i].token_type == SQLGlotTokenType.VAR and toks[i].text.upper() in ("SEARCH", "CYCLE"):
                    kw = toks[i].text.upper()
                    i += 1
                    if kw == "SEARCH":
                        while i < hi and toks[i].token_type != SQLGlotTokenType.SET:
                            i += 1
                        if i < hi:
                            i += 2  # SET <seq_col>
                    else:
                        while i < hi and toks[i].token_type != SQLGlotTokenType.USING:
                            i += 1
                        if i < hi:
                            i += 2  # USING <path_col>
                if i < hi and toks[i].token_type == SQLGlotTokenType.COMMA:
                    i += 1
                    continue
                break
            if i >= hi:
                return hi, "", is_rec
            prefix = raw_sql[toks[lo].start : toks[i].start].strip() + "\n"
            return i, prefix, is_rec

        def _combine_with_prefixes(
            outer_prefix: str, outer_rec: bool, inner_prefix: str, inner_rec: bool
        ) -> str:
            if not outer_prefix:
                return inner_prefix
            if not inner_prefix:
                return outer_prefix
            # Merge `WITH [RECURSIVE] a AS (...)` and `WITH [RECURSIVE] b AS (...)`
            # into a single valid PostgreSQL `WITH [RECURSIVE] a AS (...), b AS (...)`.
            merged_outer = outer_prefix.strip()
            if inner_rec and not outer_rec:
                merged_outer = re.sub(r"^WITH\b", "WITH RECURSIVE", merged_outer, count=1, flags=re.IGNORECASE)
            inner_rest = re.sub(
                r"^WITH\s+(?:RECURSIVE\s+)?", "", inner_prefix.strip(), count=1, flags=re.IGNORECASE
            )
            return f"{merged_outer}, {inner_rest}\n"

        # 1. Extract leading WITH [RECURSIVE] ... prefix verbatim.
        main_start, with_prefix, with_is_rec = _extract_with_prefix(0, len(toks))
        if main_start >= len(toks):
            return []

        # Collect CTE projected column counts/names from a query AST (for `*` expansion).
        def _extract_cte_cols_from_sql(sql_text: str) -> Dict[str, Optional[List[str]]]:
            out_map: Dict[str, Optional[List[str]]] = {}
            try:
                ast_obj: Any = parse_sql_ast(sql_text, dialect=Dialect.POSTGRES)
            except sqlglot.errors.SqlglotError:
                return out_map
            with_node = ast_obj.args.get("with_") if hasattr(ast_obj, "args") else None
            if with_node is None:
                return out_map
            for cte in list(with_node.expressions or ()):
                cte_alias = cte.alias
                if not cte_alias:
                    continue
                t_al: Any = cte.args.get("alias")
                if isinstance(t_al, sqlglot_exp.TableAlias) and t_al.columns:
                    out_map[cte_alias.lower()] = [c.name for c in list(t_al.columns) if c.name]
                    continue
                body = cte.this
                while isinstance(body, sqlglot_exp.SetOperation):
                    body = body.this
                if isinstance(body, sqlglot_exp.Select):
                    exprs_c = list(body.expressions or ())
                    if any(
                        isinstance(ce, sqlglot_exp.Star)
                        or (isinstance(ce, sqlglot_exp.Column) and isinstance(ce.this, sqlglot_exp.Star))
                        for ce in exprs_c
                    ):
                        out_map[cte_alias.lower()] = None
                    else:
                        out_map[cte_alias.lower()] = [
                            ce.alias or (ce.this.name if isinstance(ce, sqlglot_exp.Alias) else ce.name) or f"col_{idx_c}"
                            for idx_c, ce in enumerate(exprs_c)
                        ]
                else:
                    out_map[cte_alias.lower()] = None
            return out_map

        cte_cols_map = _extract_cte_cols_from_sql(raw_sql)

        # 2. Split top-level set operations within [main_start, len(toks)) into
        #    individual branches, handling parenthesized branches and trailing
        #    ORDER BY / LIMIT / OFFSET / FETCH / FOR modifiers.
        setop_types = (
            SQLGlotTokenType.UNION,
            SQLGlotTokenType.INTERSECT,
            SQLGlotTokenType.EXCEPT,
        )
        trailing_mod_types = {
            SQLGlotTokenType.ORDER_BY,
            SQLGlotTokenType.LIMIT,
            SQLGlotTokenType.OFFSET,
            SQLGlotTokenType.FETCH,
            SQLGlotTokenType.FOR,
            SQLGlotTokenType.SEMICOLON,
        }
        has_dedup_setop = False
        branch_ranges: List[Tuple[int, int]] = []

        def _trim_outer_parens(lo: int, hi: int) -> Tuple[int, int]:
            while (
                lo < hi
                and toks[lo].token_type == SQLGlotTokenType.L_PAREN
                and toks[hi - 1].token_type == SQLGlotTokenType.R_PAREN
            ):
                d = 0
                wraps_all = True
                for k in range(lo, hi):
                    if toks[k].token_type == SQLGlotTokenType.L_PAREN:
                        d += 1
                    elif toks[k].token_type == SQLGlotTokenType.R_PAREN:
                        d -= 1
                        if d == 0 and k != hi - 1:
                            wraps_all = False
                            break
                if wraps_all and d == 0:
                    lo += 1
                    hi -= 1
                else:
                    break
            return lo, hi

        def _collect_branches(lo: int, hi: int) -> None:
            nonlocal has_dedup_setop
            lo, hi = _trim_outer_parens(lo, hi)
            if lo >= hi:
                return
            splits: List[Tuple[int, int]] = []
            cur_lo = lo
            d = 0
            k = lo
            while k < hi:
                tt = toks[k].token_type
                if tt == SQLGlotTokenType.L_PAREN:
                    d += 1
                elif tt == SQLGlotTokenType.R_PAREN:
                    d = max(0, d - 1)
                elif d == 0 and tt in setop_types:
                    splits.append((cur_lo, k))
                    nxt = k + 1
                    is_all = nxt < hi and toks[nxt].token_type == SQLGlotTokenType.ALL
                    if tt != SQLGlotTokenType.UNION or not is_all:
                        has_dedup_setop = True
                    if nxt < hi and toks[nxt].token_type in (
                        SQLGlotTokenType.ALL,
                        SQLGlotTokenType.DISTINCT,
                    ):
                        nxt += 1
                    cur_lo = nxt
                    k = nxt
                    continue
                k += 1
            if splits:
                splits.append((cur_lo, hi))
                for s_lo, s_hi in splits:
                    _collect_branches(s_lo, s_hi)
                return

            # No top-level set op: strip trailing depth-0 ORDER BY/LIMIT/OFFSET/
            # FETCH/FOR so a parenthesized branch like `(SELECT ... UNION ...) ORDER BY 1`
            # or `(SELECT ... LIMIT 5) ORDER BY n` unwraps cleanly.
            d = 0
            trimmed_hi = hi
            for k in range(lo, hi):
                tt = toks[k].token_type
                if tt == SQLGlotTokenType.L_PAREN:
                    d += 1
                elif tt == SQLGlotTokenType.R_PAREN:
                    d = max(0, d - 1)
                elif d == 0 and tt in trailing_mod_types:
                    trimmed_hi = k
                    break
            if trimmed_hi < hi and trimmed_hi > lo:
                new_lo, new_hi = _trim_outer_parens(lo, trimmed_hi)
                if new_lo != lo or new_hi != trimmed_hi:
                    _collect_branches(new_lo, new_hi)
                    return
            branch_ranges.append((lo, hi))

        _collect_branches(main_start, len(toks))

        probes_by_key: Dict[Tuple[str, Optional[str]], Tuple[str, bool]] = {}

        def _add_probe(
            label: str,
            probe_sql: str,
            check_key_type: bool,
            fallback_sql: Optional[str] = None,
        ) -> None:
            key = (probe_sql, fallback_sql)
            prev = probes_by_key.get(key)
            if prev is None:
                probes_by_key[key] = (label, check_key_type)
            elif check_key_type and not prev[1]:
                probes_by_key[key] = (prev[0], True)

        def _expand_source_col_count(
            src: Any, local_ctes: Dict[str, Optional[List[str]]]
        ) -> Optional[List[str]]:
            if src is None:
                return None
            if isinstance(src, sqlglot_exp.Table):
                t_low = src.name.lower()
                if t_low in local_ctes:
                    return local_ctes[t_low]
                ts = self._schemas_lower.get(t_low)
                if ts is None and src.db:
                    ts = self._schemas_lower.get(f"{src.db.lower()}.{t_low}")
                if ts is not None:
                    return list(ts.columns.keys())
                return None
            if isinstance(src, sqlglot_exp.Subquery):
                t_al: Any = src.args.get("alias")
                if isinstance(t_al, sqlglot_exp.TableAlias) and t_al.columns:
                    return [c.name for c in list(t_al.columns) if c.name]
                sub_q = src.this
                while isinstance(sub_q, sqlglot_exp.SetOperation):
                    sub_q = sub_q.this
                if isinstance(sub_q, sqlglot_exp.Select):
                    return _select_output_cols(sub_q, local_ctes)
                return None
            if isinstance(src, (sqlglot_exp.Join, sqlglot_exp.Lateral)):
                return _expand_source_col_count(src.this, local_ctes)
            return None

        def _select_output_cols(
            sel_ast: sqlglot_exp.Select,
            local_ctes: Dict[str, Optional[List[str]]],
        ) -> Optional[List[str]]:
            exprs_s = list(sel_ast.expressions or ())
            if not exprs_s:
                return None
            sources_by_alias: Dict[str, Optional[List[str]]] = {}
            all_source_cols: List[Optional[List[str]]] = []

            def _reg_src(src: Any) -> None:
                if src is None:
                    return
                if isinstance(src, sqlglot_exp.Join):
                    _reg_src(src.this)
                    return
                if isinstance(src, sqlglot_exp.Paren):
                    _reg_src(src.this)
                    for sj in list(src.args.get("joins") or ()):
                        _reg_src(sj)
                    return
                cols = _expand_source_col_count(src, local_ctes)
                all_source_cols.append(cols)
                if isinstance(src, sqlglot_exp.Table):
                    al = (src.alias or src.name or "").lower()
                    if al:
                        sources_by_alias[al] = cols
                elif isinstance(src, sqlglot_exp.Subquery):
                    al = (src.alias or "").lower()
                    if al:
                        sources_by_alias[al] = cols

            f_node = sel_ast.args.get("from_") or sel_ast.args.get("from")
            if f_node is not None:
                _reg_src(f_node.this)
            for jn in list(sel_ast.args.get("joins") or ()):
                _reg_src(jn)

            out_cols: List[str] = []
            for idx_e, e in enumerate(exprs_s):
                if isinstance(e, sqlglot_exp.Star):
                    if not all_source_cols or any(c is None for c in all_source_cols):
                        return None
                    for cl in all_source_cols:
                        assert cl is not None
                        out_cols.extend(cl)
                elif isinstance(e, sqlglot_exp.Column) and isinstance(e.this, sqlglot_exp.Star):
                    tbl_k = (e.table or "").lower()
                    cl = sources_by_alias.get(tbl_k)
                    if cl is None:
                        return None
                    out_cols.extend(cl)
                else:
                    out_cols.append(e.alias or e.sql(dialect="postgres").strip() or f"col_{idx_e + 1}")
            return out_cols

        def _branch_output_cols(
            b_lo: int, b_hi: int, local_ctes: Dict[str, Optional[List[str]]]
        ) -> Optional[List[str]]:
            if b_lo >= b_hi:
                return None
            # Handle branch-local WITH
            body_lo, _, _ = _extract_with_prefix(b_lo, b_hi)
            if body_lo >= b_hi:
                return None
            b_txt = raw_sql[toks[b_lo].start : toks[b_hi - 1].end + 1].strip()
            eff_ctes = dict(local_ctes)
            if body_lo > b_lo:
                eff_ctes.update(_extract_cte_cols_from_sql(b_txt))
            tt0 = toks[body_lo].token_type
            # TABLE <name>
            if tt0 == SQLGlotTokenType.TABLE and body_lo + 1 < b_hi:
                t_name = toks[body_lo + 1].text.strip('"').lower()
                if t_name in eff_ctes:
                    return eff_ctes[t_name]
                ts = self._schemas_lower.get(t_name)
                return list(ts.columns.keys()) if ts is not None else None
            # VALUES (...)
            if tt0 == SQLGlotTokenType.VALUES:
                try:
                    v_ast = parse_sql_ast(b_txt, dialect=Dialect.POSTGRES)
                    if isinstance(v_ast, sqlglot_exp.Values) and v_ast.expressions:
                        first_row = v_ast.expressions[0]
                        if isinstance(first_row, sqlglot_exp.Tuple):
                            return [f"column{i + 1}" for i in range(len(first_row.expressions or ()))]
                except sqlglot.errors.SqlglotError:
                    return None
                return None
            if tt0 == SQLGlotTokenType.SELECT:
                body_txt = raw_sql[toks[body_lo].start : toks[b_hi - 1].end + 1].strip()
                try:
                    s_ast = parse_sql_ast(body_txt, dialect=Dialect.POSTGRES)
                    if isinstance(s_ast, sqlglot_exp.Select):
                        return _select_output_cols(s_ast, eff_ctes)
                except sqlglot.errors.SqlglotError:
                    return None
            return None

        def _probe_query_output_columns(
            query_sql: str,
            prefix_sql: str,
            col_labels: Optional[List[str]],
            fallback_label: str,
        ) -> None:
            if col_labels:
                ncols = len(col_labels)
                col_list = ", ".join(f"__mv_c{ci}" for ci in range(ncols))
                for ci in range(ncols):
                    q_sql = (
                        f"{prefix_sql}"
                        f"SELECT pg_catalog.pg_typeof(q.__mv_c{ci})::pg_catalog.oid "
                        f"FROM (SELECT 1) AS __mv_one "
                        f"LEFT JOIN (SELECT * FROM ({query_sql}\n) AS __mv_v({col_list}) WHERE false) AS q ON true"
                    )
                    _add_probe(col_labels[ci] or f"{fallback_label} {ci + 1}", q_sql, True)
            else:
                for it in self.view.select_items:
                    if not it.alias or it.raw_expr.strip() == "*" or it.raw_expr.strip().endswith(".*"):
                        continue
                    q_col = self._q(it.alias)
                    q_sql = (
                        f"{prefix_sql}"
                        f"SELECT pg_catalog.pg_typeof(q.{q_col})::pg_catalog.oid "
                        f"FROM (SELECT 1) AS __mv_one "
                        f"LEFT JOIN (SELECT * FROM ({query_sql}\n) AS __mv_v WHERE false) AS q ON true"
                    )
                    _add_probe(it.alias, q_sql, True)

        # 3. For deduplicating set operations (UNION / INTERSECT [ALL] / EXCEPT [ALL]),
        #    resolve output column count across ALL branches and probe every output column.
        if has_dedup_setop:
            resolved_cols: Optional[List[str]] = None
            for b_lo, b_hi in branch_ranges:
                resolved_cols = _branch_output_cols(b_lo, b_hi, cte_cols_map)
                if resolved_cols:
                    break
            _probe_query_output_columns(raw_sql, "", resolved_cols, "set-operation column")

        # 4. For each SELECT branch, slice FROM verbatim and probe GROUP BY,
        #    SELECT DISTINCT / DISTINCT ON, window PARTITION BY, aggregate
        #    DISTINCT args, and MIN/MAX args (including in HAVING).
        clause_stop_types = {
            SQLGlotTokenType.WHERE,
            SQLGlotTokenType.GROUP_BY,
            SQLGlotTokenType.HAVING,
            SQLGlotTokenType.WINDOW,
            SQLGlotTokenType.ORDER_BY,
            SQLGlotTokenType.LIMIT,
            SQLGlotTokenType.OFFSET,
            SQLGlotTokenType.FETCH,
            SQLGlotTokenType.FOR,
            SQLGlotTokenType.UNION,
            SQLGlotTokenType.INTERSECT,
            SQLGlotTokenType.EXCEPT,
            SQLGlotTokenType.SEMICOLON,
        }

        def _unwrap_group_elements(node: Any) -> List[Any]:
            if node is None:
                return []
            if isinstance(node, (sqlglot_exp.GroupingSets, sqlglot_exp.Cube, sqlglot_exp.Rollup)):
                out: List[Any] = []
                for child in list(node.expressions or ()):
                    out.extend(_unwrap_group_elements(child))
                return out
            if isinstance(node, sqlglot_exp.Tuple):
                out = []
                for child in list(node.expressions or ()):
                    out.extend(_unwrap_group_elements(child))
                return out
            if isinstance(node, sqlglot_exp.Paren):
                return _unwrap_group_elements(node.this)
            return [node]

        def _find_top_level_minmax(node: Any, out: List[Tuple[str, bool]]) -> None:
            if node is None or isinstance(node, (sqlglot_exp.Subquery, sqlglot_exp.Select)):
                return
            if isinstance(node, (sqlglot_exp.Min, sqlglot_exp.Max)) and node.this is not None:
                arg = node.this
                if isinstance(arg, sqlglot_exp.Distinct):
                    exprs_d = list(arg.expressions or ())
                    arg = exprs_d[0] if exprs_d else None
                if arg is not None:
                    s_arg = arg.sql(dialect="postgres").strip()
                    if s_arg:
                        has_inner_agg = isinstance(arg, sqlglot_exp.AggFunc) or bool(
                            arg.find(sqlglot_exp.AggFunc)
                        )
                        out.append((s_arg, has_inner_agg))
                return
            if not hasattr(node, "args"):
                return
            for arg_v in node.args.values():
                if hasattr(arg_v, "args"):
                    _find_top_level_minmax(arg_v, out)
                elif isinstance(arg_v, list):
                    for item_v in arg_v:
                        if hasattr(item_v, "args"):
                            _find_top_level_minmax(item_v, out)

        def _find_top_level_distinct_agg_args(node: Any, out: List[Tuple[str, bool]]) -> None:
            if node is None or isinstance(node, (sqlglot_exp.Subquery, sqlglot_exp.Select)):
                return
            if isinstance(node, sqlglot_exp.AggFunc) and isinstance(node.this, sqlglot_exp.Distinct):
                for arg in list(node.this.expressions or ()):
                    s_arg = arg.sql(dialect="postgres").strip()
                    if s_arg:
                        has_inner_agg = isinstance(arg, sqlglot_exp.AggFunc) or bool(
                            arg.find(sqlglot_exp.AggFunc)
                        )
                        out.append((s_arg, has_inner_agg))
                return
            if not hasattr(node, "args"):
                return
            for arg_v in node.args.values():
                if hasattr(arg_v, "args"):
                    _find_top_level_distinct_agg_args(arg_v, out)
                elif isinstance(arg_v, list):
                    for item_v in arg_v:
                        if hasattr(item_v, "args"):
                            _find_top_level_distinct_agg_args(item_v, out)

        for orig_b_lo, orig_b_hi in branch_ranges:
            if orig_b_lo >= orig_b_hi:
                continue
            b_lo, b_with_prefix, b_with_rec = _extract_with_prefix(orig_b_lo, orig_b_hi)
            b_hi = orig_b_hi
            if b_lo >= b_hi or toks[b_lo].token_type != SQLGlotTokenType.SELECT:
                continue
            eff_with_prefix = _combine_with_prefixes(
                with_prefix, with_is_rec, b_with_prefix, b_with_rec
            )
            eff_ctes = dict(cte_cols_map)
            if b_with_prefix:
                eff_ctes.update(
                    _extract_cte_cols_from_sql(
                        raw_sql[toks[orig_b_lo].start : toks[orig_b_hi - 1].end + 1].strip()
                    )
                )

            branch_end_char = toks[b_hi - 1].end + 1
            branch_sql = raw_sql[toks[b_lo].start : branch_end_char].strip()
            try:
                b_ast = parse_sql_ast(branch_sql, dialect=Dialect.POSTGRES)
            except sqlglot.errors.SqlglotError:
                continue
            if not isinstance(b_ast, sqlglot_exp.Select):
                continue

            # Slice FROM ... JOIN ... verbatim at depth 0 within [b_lo, b_hi),
            # ignoring `IS [NOT] DISTINCT FROM` in the SELECT list and stopping
            # strictly on keyword token types (never quoted identifiers/strings).
            from_idx: Optional[int] = None
            from_end_idx: Optional[int] = None
            c_depth = 0
            for k in range(b_lo, b_hi):
                tt = toks[k].token_type
                if tt == SQLGlotTokenType.L_PAREN:
                    c_depth += 1
                    continue
                if tt == SQLGlotTokenType.R_PAREN:
                    c_depth = max(0, c_depth - 1)
                    continue
                if c_depth != 0:
                    continue
                if from_idx is None:
                    if tt == SQLGlotTokenType.FROM:
                        is_distinct_from = (
                            k >= b_lo + 2
                            and toks[k - 1].token_type == SQLGlotTokenType.DISTINCT
                            and (
                                toks[k - 2].token_type == SQLGlotTokenType.IS
                                or (
                                    k >= b_lo + 3
                                    and toks[k - 2].token_type == SQLGlotTokenType.NOT
                                    and toks[k - 3].token_type == SQLGlotTokenType.IS
                                )
                            )
                        )
                        if not is_distinct_from:
                            from_idx = k
                else:
                    if tt in clause_stop_types:
                        from_end_idx = k
                        break

            branch_from_sql: Optional[str] = None
            if from_idx is not None:
                end_char = toks[from_end_idx].start if from_end_idx is not None else branch_end_char
                sliced = raw_sql[toks[from_idx].end + 1 : end_char].strip()
                if sliced:
                    branch_from_sql = sliced

            sel_exprs = list(b_ast.expressions or ())
            select_pos_list: List[Optional[str]] = []
            select_pos_all: List[str] = []
            alias_to_expr: Dict[str, str] = {}
            all_alias_to_expr: Dict[str, str] = {}
            for se in sel_exprs:
                inner_se = se.this if isinstance(se, sqlglot_exp.Alias) else se
                is_agg = isinstance(inner_se, sqlglot_exp.AggFunc) or bool(
                    inner_se.find(sqlglot_exp.AggFunc)
                )
                raw_se = inner_se.sql(dialect="postgres").strip()
                select_pos_all.append(raw_se)
                select_pos_list.append(None if is_agg else raw_se)
                al_name = se.alias if isinstance(se, sqlglot_exp.Alias) else ""
                if al_name and raw_se:
                    all_alias_to_expr[al_name.lower()] = raw_se
                    if not is_agg:
                        alias_to_expr[al_name.lower()] = raw_se

            def _make_branch_probe_sql(expr_sql: str, group_by_sql: Optional[str] = None) -> str:
                from_part = f" FROM {branch_from_sql}\n" if branch_from_sql else " "
                gb_part = f" GROUP BY {group_by_sql}\n" if group_by_sql else ""
                return (
                    f"{eff_with_prefix}"
                    f"SELECT pg_catalog.pg_typeof(q.__mv_k)::pg_catalog.oid "
                    f"FROM (SELECT 1) AS __mv_one "
                    f"LEFT JOIN (SELECT ({expr_sql}) AS __mv_k{from_part}WHERE false{gb_part}) AS q ON true"
                )

            def _emit_branch_expr(
                expr_sql: str,
                check_key_type: bool,
                *,
                fallback_expr_sql: Optional[str] = None,
                group_by_sql: Optional[str] = None,
            ) -> None:
                es = expr_sql.strip()
                if not es:
                    return
                q_sql = _make_branch_probe_sql(es, group_by_sql=group_by_sql)
                fb_sql = (
                    _make_branch_probe_sql(fallback_expr_sql.strip(), group_by_sql=group_by_sql)
                    if fallback_expr_sql and fallback_expr_sql.strip() and fallback_expr_sql.strip() != es
                    else None
                )
                _add_probe(es, q_sql, check_key_type, fallback_sql=fb_sql)

            # 4a. GROUP BY expressions (including GROUPING SETS / ROLLUP / CUBE).
            #     For bare identifiers matching a SELECT output alias, probe the
            #     input column FIRST and only fall back to the output alias on
            #     undefined_column — exact PostgreSQL binding even across SELECT *.
            group_node = b_ast.args.get("group")
            resolved_group_exprs: List[str] = []
            if group_node is not None:
                raw_g_nodes: List[Any] = list(group_node.expressions or ())
                for extra_k in ("grouping_sets", "cube", "rollup"):
                    raw_g_nodes.extend(list(group_node.args.get(extra_k) or ()))
                for raw_ge in raw_g_nodes:
                    for ge in _unwrap_group_elements(raw_ge):
                        coll_suffix = ""
                        inner_ge = ge
                        if (
                            isinstance(ge, sqlglot_exp.Collate)
                            and ge.this is not None
                            and ge.expression is not None
                        ):
                            coll_suffix = f" COLLATE {ge.expression.sql(dialect='postgres')}"
                            inner_ge = ge.this
                        if (
                            isinstance(inner_ge, sqlglot_exp.Literal)
                            and not inner_ge.args.get("is_string")
                            and str(inner_ge.this).isdigit()
                        ):
                            pos_i = int(str(inner_ge.this))
                            if 1 <= pos_i <= len(select_pos_list):
                                pos_e = select_pos_list[pos_i - 1]
                                if pos_e:
                                    e_sql = f"({pos_e}){coll_suffix}" if coll_suffix else pos_e
                                    resolved_group_exprs.append(e_sql)
                                    _emit_branch_expr(e_sql, True)
                            continue
                        direct_sql = ge.sql(dialect="postgres")
                        fb_sql_expr: Optional[str] = None
                        if (
                            isinstance(inner_ge, sqlglot_exp.Column)
                            and not inner_ge.table
                            and inner_ge.name.lower() in alias_to_expr
                        ):
                            al_e = alias_to_expr[inner_ge.name.lower()]
                            fb_sql_expr = f"({al_e}){coll_suffix}" if coll_suffix else al_e
                        resolved_group_exprs.append(direct_sql)
                        _emit_branch_expr(direct_sql, True, fallback_expr_sql=fb_sql_expr)

            probe_group_sql = ", ".join(resolved_group_exprs) if resolved_group_exprs else None

            # 4b. SELECT DISTINCT / DISTINCT ON (...)
            distinct_node = b_ast.args.get("distinct")
            if distinct_node is not None:
                on_node = distinct_node.args.get("on")
                if on_node is not None:
                    on_exprs = (
                        list(on_node.expressions or ())
                        if isinstance(on_node, sqlglot_exp.Tuple)
                        else [on_node]
                    )
                    for oe in on_exprs:
                        coll_suffix = ""
                        inner_oe = oe
                        if (
                            isinstance(oe, sqlglot_exp.Collate)
                            and oe.this is not None
                            and oe.expression is not None
                        ):
                            coll_suffix = f" COLLATE {oe.expression.sql(dialect='postgres')}"
                            inner_oe = oe.this
                        if (
                            isinstance(inner_oe, sqlglot_exp.Literal)
                            and not inner_oe.args.get("is_string")
                            and str(inner_oe.this).isdigit()
                        ):
                            pos_i = int(str(inner_oe.this))
                            if 1 <= pos_i <= len(select_pos_all):
                                pos_e = select_pos_all[pos_i - 1]
                                if pos_e:
                                    e_sql = f"({pos_e}){coll_suffix}" if coll_suffix else pos_e
                                    has_a = select_pos_list[pos_i - 1] is None
                                    _emit_branch_expr(
                                        e_sql,
                                        True,
                                        group_by_sql=probe_group_sql if has_a else None,
                                    )
                            continue
                        oe_sql = oe.sql(dialect="postgres")
                        has_a = isinstance(oe, sqlglot_exp.AggFunc) or bool(oe.find(sqlglot_exp.AggFunc))
                        if (
                            isinstance(inner_oe, sqlglot_exp.Column)
                            and not inner_oe.table
                            and inner_oe.name.lower() in all_alias_to_expr
                        ):
                            al_e = all_alias_to_expr[inner_oe.name.lower()]
                            al_sql = f"({al_e}){coll_suffix}" if coll_suffix else al_e
                            al_has_a = inner_oe.name.lower() not in alias_to_expr
                            # DISTINCT ON follows ORDER BY precedence: output alias first.
                            _emit_branch_expr(
                                al_sql,
                                True,
                                fallback_expr_sql=oe_sql,
                                group_by_sql=probe_group_sql if (al_has_a or has_a) else None,
                            )
                        else:
                            _emit_branch_expr(
                                oe_sql,
                                True,
                                group_by_sql=probe_group_sql if has_a else None,
                            )
                elif not has_dedup_setop:
                    # Probe every output column of the SELECT DISTINCT branch directly
                    # (handles SELECT DISTINCT *, named WINDOW, GROUP BY + aggregate,
                    # and window functions like LAG without tearing them out of scope).
                    dist_cols = _select_output_cols(b_ast, eff_ctes)
                    _probe_query_output_columns(branch_sql, eff_with_prefix, dist_cols, "DISTINCT column")

            # 4c. Window PARTITION BY (inline OVER (...) and named WINDOW clauses)
            for se in sel_exprs:
                for win in se.find_all(sqlglot_exp.Window):
                    for pe in list(win.args.get("partition_by") or ()):
                        pe_has_agg = isinstance(pe, sqlglot_exp.AggFunc) or bool(
                            pe.find(sqlglot_exp.AggFunc)
                        )
                        _emit_branch_expr(
                            pe.sql(dialect="postgres"),
                            True,
                            group_by_sql=probe_group_sql if pe_has_agg else None,
                        )
            for win_def in list(b_ast.args.get("windows") or ()):
                for pe in list(win_def.args.get("partition_by") or ()):
                    pe_has_agg = isinstance(pe, sqlglot_exp.AggFunc) or bool(
                        pe.find(sqlglot_exp.AggFunc)
                    )
                    _emit_branch_expr(
                        pe.sql(dialect="postgres"),
                        True,
                        group_by_sql=probe_group_sql if pe_has_agg else None,
                    )

            # 4d. Aggregate DISTINCT arguments (e.g. COUNT(DISTINCT ...)) in SELECT and HAVING
            dist_agg_args: List[Tuple[str, bool]] = []
            for se in sel_exprs:
                _find_top_level_distinct_agg_args(se, dist_agg_args)
            having_node = b_ast.args.get("having")
            if having_node is not None:
                _find_top_level_distinct_agg_args(having_node, dist_agg_args)
            for da_sql, da_has_agg in dist_agg_args:
                _emit_branch_expr(
                    da_sql,
                    True,
                    group_by_sql=probe_group_sql if da_has_agg else None,
                )

            # 4e. MIN/MAX arguments (nondeterministic collation check only) in SELECT and HAVING
            minmax_args: List[Tuple[str, bool]] = []
            for se in sel_exprs:
                _find_top_level_minmax(se, minmax_args)
            if having_node is not None:
                _find_top_level_minmax(having_node, minmax_args)
            for ma_sql, ma_has_agg in minmax_args:
                _emit_branch_expr(
                    ma_sql,
                    False,
                    group_by_sql=probe_group_sql if ma_has_agg else None,
                )

        return [
            (lbl, q_sql, fb_sql, chk)
            for (q_sql, fb_sql), (lbl, chk) in probes_by_key.items()
        ]

    def _pg_collation_guard_ddl(self) -> Optional[str]:
        """Refuse (0A000) at registration any view whose keys use a
        nondeterministic collation (including inside domains, ranges,
        multiranges, arrays, or composite fields) or an unsupported key type
        (such as anonymous ``record``).

        Under a nondeterministic collation (PostgreSQL 12+) distinct strings
        compare equal ('A' = 'a' under a case-insensitive ICU collation):
        ``GROUP BY`` merges them, so no canonical text or index on the stored
        value identifies the group. Returns None when the view neither groups,
        deduplicates nor takes MIN/MAX.
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

        probe_queries = self._pg_probe_queries()
        probe_block = ""
        if probe_queries:
            def _lit(val: str) -> str:
                return "'" + val.replace("'", "''") + "'"

            values_rows = ",\n      ".join(
                f"({_lit(lbl)}::pg_catalog.text, {_lit(q_sql)}::pg_catalog.text, "
                f"{_lit(fb_sql) + '::pg_catalog.text' if fb_sql is not None else 'NULL::pg_catalog.text'}, "
                f"{'true' if chk else 'false'})"
                for lbl, q_sql, fb_sql, chk in probe_queries
            )
            probe_block = (
                "  FOR probe_rec IN\n"
                "    SELECT * FROM (VALUES\n"
                f"      {values_rows}\n"
                "    ) AS __mv_p(lbl, q_sql, fb_sql, chk_type)\n"
                "  LOOP\n"
                "    k_oid := NULL;\n"
                "    BEGIN\n"
                "      EXECUTE probe_rec.q_sql INTO k_oid;\n"
                "    EXCEPTION WHEN undefined_column OR undefined_table OR undefined_function THEN\n"
                "      k_oid := NULL;\n"
                "      IF probe_rec.fb_sql IS NOT NULL THEN\n"
                "        BEGIN\n"
                "          EXECUTE probe_rec.fb_sql INTO k_oid;\n"
                "        EXCEPTION WHEN undefined_column OR undefined_table OR undefined_function THEN\n"
                "          k_oid := NULL;\n"
                "        END;\n"
                "      END IF;\n"
                "    END;\n"
                "    IF k_oid IS NOT NULL THEN\n"
                "      type_walk_sql :=\n"
                "        'WITH RECURSIVE kt(type_oid, coll_oid, depth) AS ('\n"
                "        ' SELECT $1, 0::pg_catalog.oid, 0'\n"
                "        ' UNION ALL'\n"
                "        ' SELECT nxt.type_oid, nxt.coll_oid, kt.depth + 1'\n"
                "        ' FROM kt JOIN pg_catalog.pg_type t ON t.oid = kt.type_oid'\n"
                "        ' CROSS JOIN LATERAL ('\n"
                "        '   SELECT t.typbasetype AS type_oid, CASE WHEN t.typcollation <> 0 THEN t.typcollation ELSE kt.coll_oid END AS coll_oid WHERE t.typtype = ''d'''\n"
                "        '   UNION ALL SELECT t.typelem, 0::pg_catalog.oid WHERE t.typcategory = ''A'' AND t.typelem <> 0'\n"
                "        '   UNION ALL SELECT rg.rngsubtype, rg.rngcollation FROM pg_catalog.pg_range rg WHERE t.typtype = ''r'' AND rg.rngtypid = t.oid'\n"
                "        || CASE WHEN ver_num >= 140000 THEN\n"
                "           '   UNION ALL SELECT rg.rngtypid, 0::pg_catalog.oid FROM pg_catalog.pg_range rg WHERE t.typtype = ''m'' AND rg.rngmultitypid = t.oid'\n"
                "           ELSE '' END\n"
                "        || '   UNION ALL SELECT fa.atttypid, fa.attcollation FROM pg_catalog.pg_attribute fa WHERE t.typtype = ''c'' AND t.typrelid <> 0 AND fa.attrelid = t.typrelid AND fa.attnum > 0 AND NOT fa.attisdropped'\n"
                "        ' ) AS nxt WHERE kt.depth < 16'\n"
                "        ') ';\n"
                "      IF probe_rec.chk_type THEN\n"
                "        EXECUTE type_walk_sql\n"
                "          || 'SELECT t.oid FROM kt JOIN pg_catalog.pg_type t ON t.oid = kt.type_oid '\n"
                "          || 'WHERE t.typtype NOT IN (''b'', ''e'', ''d'', ''r'', ''m'', ''c'') '\n"
                "          || '   OR (t.typtype = ''c'' AND t.typrelid = 0) LIMIT 1'\n"
                "          INTO bad_oid USING k_oid;\n"
                "        IF bad_oid IS NOT NULL THEN\n"
                "          RAISE EXCEPTION 'matview: a grouping key of type % is not supported', pg_catalog.format_type(bad_oid, NULL)\n"
                "            USING ERRCODE = 'feature_not_supported';\n"
                "        END IF;\n"
                "      END IF;\n"
                "      IF ver_num >= 120000 THEN\n"
                "        EXECUTE type_walk_sql\n"
                "          || 'SELECT c.collname::pg_catalog.text FROM kt JOIN pg_catalog.pg_collation c ON c.oid = kt.coll_oid '\n"
                "          || 'WHERE NOT c.collisdeterministic LIMIT 1'\n"
                "          INTO bad_coll USING k_oid;\n"
                "        IF bad_coll IS NOT NULL THEN\n"
                "          RAISE EXCEPTION 'matview: % uses the nondeterministic collation %, which is not supported for grouping, DISTINCT, set operations, window partitions or MIN/MAX',\n"
                "            probe_rec.lbl, pg_catalog.quote_ident(bad_coll)\n"
                "            USING ERRCODE = 'feature_not_supported';\n"
                "        END IF;\n"
                "      END IF;\n"
                "    END IF;\n"
                "  END LOOP;\n"
            )

        body = (
            "DECLARE\n"
            f"  q pg_catalog.text := {tag}{sql}{tag};\n"
            "  ver_num pg_catalog.int4 := pg_catalog.current_setting('server_version_num')::pg_catalog.int4;\n"
            "  r record;\n"
            "  probe_rec record;\n"
            "  type_walk_sql pg_catalog.text;\n"
            "  k_oid pg_catalog.oid;\n"
            "  bad_oid pg_catalog.oid;\n"
            "  bad_coll pg_catalog.text;\n"
            "BEGIN\n"
            f"{probe_block}"
            "  IF ver_num < 120000 THEN\n"
            "    RETURN;\n"
            "  END IF;\n"
            "  FOR r IN EXECUTE\n"
            "      'WITH RECURSIVE col_types(rel, what, type_oid, coll_oid, depth) AS ('\n"
            "      ' SELECT a.attrelid::pg_catalog.regclass::pg_catalog.text, a.attname::pg_catalog.text, a.atttypid, a.attcollation, 0'\n"
            "      ' FROM pg_catalog.pg_attribute a'\n"
            "      ' WHERE a.attrelid = ANY ($1) AND a.attnum > 0 AND NOT a.attisdropped'\n"
            "      ' UNION ALL'\n"
            "      ' SELECT ct.rel, ct.what, nxt.type_oid, nxt.coll_oid, ct.depth + 1'\n"
            "      ' FROM col_types ct JOIN pg_catalog.pg_type t ON t.oid = ct.type_oid'\n"
            "      ' CROSS JOIN LATERAL ('\n"
            "      '   SELECT t.typbasetype AS type_oid, CASE WHEN t.typcollation <> 0 THEN t.typcollation ELSE ct.coll_oid END AS coll_oid WHERE t.typtype = ''d'''\n"
            "      '   UNION ALL SELECT t.typelem, 0::pg_catalog.oid WHERE t.typcategory = ''A'' AND t.typelem <> 0'\n"
            "      '   UNION ALL SELECT rg.rngsubtype, rg.rngcollation FROM pg_catalog.pg_range rg WHERE t.typtype = ''r'' AND rg.rngtypid = t.oid'\n"
            "      || CASE WHEN ver_num >= 140000 THEN\n"
            "         '   UNION ALL SELECT rg.rngtypid, 0::pg_catalog.oid FROM pg_catalog.pg_range rg WHERE t.typtype = ''m'' AND rg.rngmultitypid = t.oid'\n"
            "         ELSE '' END\n"
            "      || '   UNION ALL SELECT fa.atttypid, fa.attcollation FROM pg_catalog.pg_attribute fa WHERE t.typtype = ''c'' AND t.typrelid <> 0 AND fa.attrelid = t.typrelid AND fa.attnum > 0 AND NOT fa.attisdropped'\n"
            "      ' ) AS nxt WHERE ct.depth < 16'\n"
            "      ') '\n"
            "      'SELECT DISTINCT ct.rel, ct.what, c.collname::pg_catalog.text AS coll'\n"
            "      ' FROM col_types ct JOIN pg_catalog.pg_collation c ON c.oid = ct.coll_oid'\n"
            "      ' WHERE NOT c.collisdeterministic'\n"
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
        )
        do_tag = "$mv_coll$"
        while do_tag in body:
            do_tag = do_tag[:-1] + "x$"
        return f"DO {do_tag}\n{body}{do_tag};"
