#!/usr/bin/env python3
"""Run curated, test-specific source mutants to verify regression sensitivity.

Manual invocation:
    python mutation_testing.py              # every curated mutant
    python mutation_testing.py NAME [...]   # selected mutants by name
    python mutation_testing.py --list       # list mutants and their targets

CI runs this diagnostic in every PR-gate SQLite job (Python 3.11 and 3.15).

Production code lives in the modular ``matview`` package; ``matview_sync.py``
is only a re-exporting facade. Each mutant therefore names the package module
that contains its mutation site. Before any test runs, every mutant's source
file must exist and its original text must occur exactly once in that file;
otherwise the run fails loudly instead of silently testing nothing.

Each mutant is applied to an isolated temporary copy of the ``matview``
package, the facade, and the test module. The run verifies that the copy (not
an installed or repository package) is what gets imported. The target
regression must pass unchanged and fail by assertion for the mutant. Syntax,
import, and setup failures are not counted as killed mutants.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = "matview"
COPIED_FILES: Tuple[str, ...] = (
    "matview_sync.py",
    "test_matview_sync.py",
    "test_native_statistical_aggregates.py",
)
TARGET_TEST = (
    "test_matview_sync.TestGeneratorAndManagerCodePaths."
    "test_25_join_aggregate_decrement_and_group_key_invariants"
)


@dataclass(frozen=True)
class Mutant:
    name: str
    source_path: str
    original: str
    replacement: str
    target_test: str = TARGET_TEST


MUTANTS = (
    Mutant(
        name="filtered-projection-uses-new-predicate-for-old-row-retraction",
        source_path="matview/maintenance/zset.py",
        original="            upd_old_stmts=upd_old_list,\n            upd_new_stmts=[plus_stmt],\n            upd_cleanup_old_stmts=upd_clean_list,\n            trunc_stmts=trunc_stmts,\n            where_new=where_new,\n            where_old=where_old,",
        replacement="            upd_old_stmts=upd_old_list,\n            upd_new_stmts=[plus_stmt],\n            upd_cleanup_old_stmts=upd_clean_list,\n            trunc_stmts=trunc_stmts,\n            where_new=where_new,\n            where_old=where_new,",
        target_test=(
            "test_matview_sync.TestGeneratorAndManagerCodePaths."
            "test_filtered_projection_retracts_old_membership_on_update_and_delete"
        ),
    ),
    # SQLite maintains SUM/TOTAL/AVG over joins by recomputing the affected
    # group natively (exact storage-class semantics), so the retraction site
    # exercised by the SQLite target test is the native recompute assignment.
    # The PostgreSQL/MySQL delta fold (_delta_sum_minus) is not reachable from
    # this SQLite-only runner; see TODO.md (mutation testing) for that gap.
    Mutant(
        name="sqlite-join-sum-retraction-keeps-stale-value",
        source_path="matview/maintenance/join.py",
        original='minus_assignments.append(f"{c} = {_build_join_recompute_subquery(item)}")',
        replacement='minus_assignments.append(f"{c} = {mv_table}.{c}")',
    ),
    Mutant(
        name="sqlite-join-sum-retraction-coalesces-all-null-group-to-zero",
        source_path="matview/maintenance/join.py",
        original='minus_assignments.append(f"{c} = {_build_join_recompute_subquery(item)}")',
        replacement='minus_assignments.append(f"{c} = COALESCE({_build_join_recompute_subquery(item)}, 0)")',
    ),
    Mutant(
        name="distinct-auxiliary-retraction-increments-reference-count",
        source_path="matview/maintenance/single.py",
        original="f\"    SET {self._q('__ref_cnt')} = {aux_tbl}.{self._q('__ref_cnt')} - 1\\n\"",
        replacement="f\"    SET {self._q('__ref_cnt')} = {aux_tbl}.{self._q('__ref_cnt')} + 1\\n\"",
        target_test=(
            "test_matview_sync.TestGeneratorAndManagerCodePaths."
            "test_24_metamorphic_mv_invariants_and_rebuild_consistency"
        ),
    ),
    Mutant(
        name="join-count-retraction-adds-instead-of-subtracting",
        source_path="matview/maintenance/join.py",
        original='minus_assignments.append(f"{c} = {mv_table}.{c} - __mv_delta.{c}")',
        replacement='minus_assignments.append(f"{c} = {mv_table}.{c} + __mv_delta.{c}")',
    ),
    Mutant(
        name="join-update-omits-old-row-retraction",
        source_path="matview/maintenance/join.py",
        original="upd_old_stmts=[minus_stmt],",
        replacement="upd_old_stmts=[],",
    ),
    Mutant(
        name="backfill-watermark-skips-update-at-inclusive-cursor",
        source_path="matview/maintenance/single.py",
        original="NEW.{q_pk} > backfill_cursor_pk",
        replacement="NEW.{q_pk} >= backfill_cursor_pk",
        target_test=(
            "test_matview_sync.TestGeneratorAndManagerCodePaths."
            "test_backfill_watermark_includes_cursor_row_update"
        ),
    ),
    Mutant(
        name="join-retraction-reads-new-row-instead-of-old-row",
        source_path="matview/maintenance/join.py",
        original="upd_old_stmts=[minus_stmt],",
        replacement='upd_old_stmts=[minus_stmt.replace("OLD.", "NEW.")],',
    ),
    Mutant(
        name="pk-swap-native-projection-drops-not-exists-guard",
        source_path="matview/maintenance/zset.py",
        original='f"    AND NOT EXISTS (SELECT 1 FROM {q_base_t} WHERE {base_pk_eq_old}{and_base_where});"',
        replacement='";"',
        target_test=(
            "test_p0_pk_swap_and_collation.TestFinding6PkSwapAndShift."
            "test_pk_swap_execution_data_loss_prevention"
        ),
    ),
    Mutant(
        name="pk-swap-native-projection-drops-where-from-not-exists-guard",
        source_path="matview/maintenance/zset.py",
        original='f"    AND NOT EXISTS (SELECT 1 FROM {q_base_t} WHERE {base_pk_eq_old}{and_base_where});"',
        replacement='f"    AND NOT EXISTS (SELECT 1 FROM {q_base_t} WHERE {base_pk_eq_old});"',
        target_test=(
            "test_p0_pk_swap_and_collation.TestFinding6PkSwapAndShift."
            "test_filtered_native_pk_projection_where_exit_and_swap_live_equivalence"
        ),
    ),
    Mutant(
        name="pk-swap-refcount-projection-drops-row-count-guard",
        source_path="matview/maintenance/zset.py",
        original='f"    WHERE ({mv_table}.__mv_pk = ({pk_old})) AND {mv_table}.__mv_row_count <= 0;"',
        replacement='f"    WHERE ({mv_table}.__mv_pk = ({pk_old}));"',
        target_test=(
            "test_p0_pk_swap_and_collation.TestFinding6PkSwapAndShift."
            "test_refcount_projection_pk_swap_live_equivalence"
        ),
    ),
    Mutant(
        name="pk-swap-refcount-projection-update-omits-old-decrement",
        source_path="matview/maintenance/zset.py",
        original=(
            "            del_stmts_list = [minus_stmt, clean_stmt]\n"
            "            upd_old_list = [minus_stmt]\n"
            "            upd_clean_list = [clean_stmt]"
        ),
        replacement=(
            "            del_stmts_list = [minus_stmt, clean_stmt]\n"
            "            upd_old_list = []\n"
            "            upd_clean_list = [clean_stmt]"
        ),
        target_test=(
            "test_p0_pk_swap_and_collation.TestFinding6PkSwapAndShift."
            "test_refcount_projection_pk_swap_live_equivalence"
        ),
    ),
    Mutant(
        name="zset-watermark-new-skips-update-at-inclusive-cursor",
        source_path="matview/maintenance/zset.py",
        original='                f"AND (backfill_cursor_pk IS NULL OR (NEW.{q_pk} > backfill_cursor_pk AND NEW.{q_pk} <= backfill_max_pk)))"',
        replacement='                f"AND (backfill_cursor_pk IS NULL OR (NEW.{q_pk} >= backfill_cursor_pk AND NEW.{q_pk} <= backfill_max_pk)))"',
        target_test=(
            "test_p0_backfill_watermark_safety.BackfillWatermarkSafetyTests."
            "test_projection_watermark_updates_exact_inclusive_cursor_row"
        ),
    ),
    Mutant(
        name="zset-watermark-old-skips-retraction-at-inclusive-cursor",
        source_path="matview/maintenance/zset.py",
        original='                f"AND (backfill_cursor_pk IS NULL OR (OLD.{q_pk} > backfill_cursor_pk AND OLD.{q_pk} <= backfill_max_pk)))"',
        replacement='                f"AND (backfill_cursor_pk IS NULL OR (OLD.{q_pk} >= backfill_cursor_pk AND OLD.{q_pk} <= backfill_max_pk)))"',
        target_test=(
            "test_p0_backfill_watermark_safety.BackfillWatermarkSafetyTests."
            "test_projection_watermark_deletes_exact_inclusive_cursor_row"
        ),
    ),
    Mutant(
        name="single-watermark-drops-unclaimed-null-cursor-gate",
        source_path="matview/maintenance/single.py",
        original='                f"AND (backfill_cursor_pk IS NULL OR (NEW.{q_pk} > backfill_cursor_pk AND NEW.{q_pk} <= backfill_max_pk)))"',
        replacement='                f"AND NEW.{q_pk} > backfill_cursor_pk AND NEW.{q_pk} <= backfill_max_pk)"',
        target_test=(
            "test_p0_backfill_watermark_safety.BackfillWatermarkSafetyTests."
            "test_populate_false_gates_writes_until_backfill_claims_and_reports_incomplete"
        ),
    ),
    Mutant(
        name="single-backfill-trigger-omits-unreferenced-pk-update-key",
        source_path="matview/maintenance/single.py",
        original="            update_key_columns=[wm_pk_col] if wm_pk_col else None,",
        replacement="            update_key_columns=None,",
        target_test=(
            "test_p0_backfill_watermark_safety.BackfillWatermarkSafetyTests."
            "test_backfill_unreferenced_pk_update_across_watermark_boundaries"
        ),
    ),
    Mutant(
        name="zset-backfill-trigger-omits-watermark-rowid-key",
        source_path="matview/maintenance/zset.py",
        original=(
            "        if (\n"
            "            self.watermark_catalog_table\n"
            "            and wm_pk_col\n"
            "            and wm_pk_col.lower() not in {c.lower() for c in upd_key_cols}\n"
            "        ):\n"
            "            upd_key_cols.append(wm_pk_col)"
        ),
        replacement="        pass",
        target_test=(
            "test_p0_backfill_watermark_safety.BackfillWatermarkSafetyTests."
            "test_backfill_unreferenced_rowid_update_on_text_pk_table"
        ),
    ),
    Mutant(
        name="emitters-sqlite-implicit-rowid-omits-unchanged-guard",
        source_path="matview/dialects/emitters.py",
        original=(
            "            for rowid_alias in implicit_rowid_aliases:\n"
            "                unchanged_conds.append(f\"OLD.{rowid_alias} = NEW.{rowid_alias}\")"
        ),
        replacement="            pass",
        target_test=(
            "test_p0_backfill_watermark_safety.BackfillWatermarkSafetyTests."
            "test_backfill_unreferenced_rowid_update_on_text_pk_table"
        ),
    ),
    Mutant(
        name="pg-stage3-join-omits-inclusion-exclusion-correction",
        source_path="matview/maintenance/join.py",
        original="                for subset_size in range(1, n_joins + 1):",
        replacement="                for subset_size in range(1, 2):",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_stage3_multi_source_inclusion_exclusion_codegen"
        ),
    ),
    Mutant(
        name="stage4-join-empty-group-delete-falls-back-to-full-table-scan",
        source_path="matview/maintenance/join.py",
        original=(
            "                    del_zero_stmt = (\n"
            "                        f\"DELETE FROM {mv_table}\\n\"\n"
            "                        f\"    WHERE {pk_in_lhs} IN (\\n\"\n"
            "                        f\"        SELECT {', '.join(pk_sel_old)}\\n\"\n"
            "                        f\"        FROM {from_old_eff}{where_old_eff}\\n\"\n"
            "                        f\"    ) AND {mv_table}.{self._q(row_count_col)} <= 0;\"\n"
            "                    )"
        ),
        replacement=(
            "                    del_zero_stmt = (\n"
            "                        f\"DELETE FROM {mv_table} WHERE {mv_table}.{self._q(row_count_col)} <= 0;\"\n"
            "                    )"
        ),
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_stage4_lean_row_triggers_codegen_sqlite_and_mysql"
        ),
    ),
    Mutant(
        name="stage4-one-row-sqlite-sum-drops-scalar-subquery-scope",
        source_path="matview/maintenance/join.py",
        original="f\"THEN ({eff_s}) ELSE (SELECT SUM(__v) FROM (SELECT ({eff_s}) AS __v)) END)\"",
        replacement="f\"THEN ({eff_s}) ELSE (SELECT SUM({eff_s})) END)\"",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_stage4_lean_row_triggers_codegen_sqlite_and_mysql"
        ),
    ),
    Mutant(
        name="pg-repair-fence-omits-repeatable-read-check",
        source_path="matview/dialects/pg_statement.py",
        original="\"    IF pg_catalog.current_setting('transaction_isolation') IN ('repeatable read', 'serializable') THEN\\n\"",
        replacement="\"    IF pg_catalog.current_setting('transaction_isolation') IN ('serializable') THEN\\n\"",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_repair_snapshot_fence_codegen"
        ),
    ),
    Mutant(
        name="pg-repair-fence-weakens-for-share-to-for-key-share",
        source_path="matview/dialects/pg_statement.py",
        original="f\"        PERFORM 1 FROM {self.repair_fence_catalog_table} WHERE view_name = '{view_lit}' FOR SHARE;\\n\"",
        replacement="f\"        PERFORM 1 FROM {self.repair_fence_catalog_table} WHERE view_name = '{view_lit}' FOR KEY SHARE;\\n\"",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_repair_snapshot_fence_codegen"
        ),
    ),
    Mutant(
        name="plain-view-inliner-skips-cycle-check",
        source_path="matview/sql_frontend.py",
        original="                if stack_key in active_stack or (not tbl_db and t_low in active_stack):\n                    chain = \" -> \".join([*active_stack, canon_name])\n                    raise ValueError(f\"Cyclic plain SQL view reference detected: {chain}\")",
        replacement="                if False:\n                    chain = \" -> \".join([*active_stack, canon_name])\n                    raise ValueError(f\"Cyclic plain SQL view reference detected: {chain}\")",
        target_test=(
            "test_p1_manager_safety.TestPlainViewAsSource."
            "test_cte_shadowing_cycle_depth_and_registered_mv_refusal"
        ),
    ),
    Mutant(
        name="pg-minmax-drops-collation-on-varchar",
        source_path="matview/dialects/base.py",
        original="        m_coll = self._PG_COLLATE_TAIL_RE.search(s_raw)",
        replacement="        m_coll = None",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_minmax_varchar_collation_name_and_numeric_group_rep_codegen"
        ),
    ),
    Mutant(
        name="pg-numeric-group-key-skips-rep-refresh",
        source_path="matview/dialects/pg_statement.py",
        original="        if (has_old or force_recompute_extremes) and spec.recompute_group_keys is not None and grp_exprs:",
        replacement="        if False and (has_old or force_recompute_extremes) and spec.recompute_group_keys is not None and grp_exprs:",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_minmax_varchar_collation_name_and_numeric_group_rep_codegen"
        ),
    ),
    Mutant(
        name="pg-agg-group-update-skips-same-scale-ctid-move",
        source_path="matview/dialects/pg_statement.py",
        original=(
            "                        rep_conds.append(\n"
            "                            f\"({mv}.{col} IS DISTINCT FROM NULL AND (\"\n"
            "                            f\"ROW({mv}.{col})::pg_catalog.text = ANY(__mv_r.{dcol}) OR \"\n"
            "                            f\"(__mv_r.{icol} IS NOT NULL AND \"\n"
            "                            f\"NOT (ARRAY[ROW({mv}.{col})::pg_catalog.text] @> __mv_r.{icol}))))\"\n"
            "                        )"
        ),
        replacement=(
            "                        rep_conds.append(\n"
            "                            f\"({mv}.{col} IS DISTINCT FROM NULL AND \"\n"
            "                            f\"ROW({mv}.{col})::pg_catalog.text = ANY(__mv_r.{dcol}) AND \"\n"
            "                            f\"NOT (ROW({mv}.{col})::pg_catalog.text = ANY(COALESCE(__mv_r.{icol}, ARRAY[]::pg_catalog.text[]))))\"\n"
            "                        )"
        ),
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_minmax_varchar_collation_name_and_numeric_group_rep_codegen"
        ),
    ),
    Mutant(
        name="pg-agg-group-insert-skips-diff-scale-rep-refresh",
        source_path="matview/dialects/pg_statement.py",
        original="        elif not has_old and not force_recompute_extremes and spec.recompute_group_keys is not None and grp_exprs:",
        replacement="        elif False and not has_old and not force_recompute_extremes and spec.recompute_group_keys is not None and grp_exprs:",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_minmax_varchar_collation_name_and_numeric_group_rep_codegen"
        ),
    ),
    Mutant(
        name="pg-counted-rows-update-skips-same-scale-ctid-move",
        source_path="matview/dialects/pg_statement.py",
        original=(
            "                        rec_conds.append(\n"
            "                            f\"({mv_table}.{qc} IS DISTINCT FROM NULL AND (\"\n"
            "                            f\"ROW({mv_table}.{qc})::pg_catalog.text = ANY(__mv_r.{dcol}) OR \"\n"
            "                            f\"(__mv_r.{icol} IS NOT NULL AND \"\n"
            "                            f\"NOT (ARRAY[ROW({mv_table}.{qc})::pg_catalog.text] @> __mv_r.{icol}))))\"\n"
            "                        )"
        ),
        replacement=(
            "                        rec_conds.append(\n"
            "                            f\"({mv_table}.{qc} IS DISTINCT FROM NULL AND \"\n"
            "                            f\"ROW({mv_table}.{qc})::pg_catalog.text = ANY(__mv_r.{dcol}) AND \"\n"
            "                            f\"NOT (ROW({mv_table}.{qc})::pg_catalog.text = ANY(COALESCE(__mv_r.{icol}, ARRAY[]::pg_catalog.text[]))))\"\n"
            "                        )"
        ),
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_minmax_varchar_collation_name_and_numeric_group_rep_codegen"
        ),
    ),
    Mutant(
        name="pg-counted-rows-insert-skips-diff-scale-rep-refresh",
        source_path="matview/dialects/pg_statement.py",
        original=(
            "                        ins_conds.append(\n"
            "                            f\"({mv_table}.{qc} IS DISTINCT FROM NULL AND __mv_r.{icol} IS NOT NULL AND \"\n"
            "                            f\"NOT (ARRAY[ROW({mv_table}.{qc})::pg_catalog.text] @> __mv_r.{icol}))\"\n"
            "                        )"
        ),
        replacement=(
            "                        ins_conds.append(\"FALSE\")"
        ),
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_minmax_varchar_collation_name_and_numeric_group_rep_codegen"
        ),
    ),
    Mutant(
        name="pg-drop-partition-drift-ignored",
        source_path="matview/manager.py",
        original="                                if not tbl_exists or live_parts != recorded_parts:\n                                    status = _merge(status, \"partitions_changed\")",
        replacement="                                if False:\n                                    status = _merge(status, \"partitions_changed\")",
        target_test=(
            "test_p1_manager_safety.TestPgPartitionDropAndTruncateTriggerDrift."
            "test_pg_partition_drop_and_truncate_trigger_drift_detection"
        ),
    ),
    Mutant(
        name="pg-live-trigger-names-uses-information-schema",
        source_path="matview/manager.py",
        original="                \"SELECT t.tgname FROM pg_catalog.pg_trigger t \"",
        replacement="                \"SELECT trigger_name FROM information_schema.triggers WHERE trigger_schema = current_schema(); -- \"",
        target_test=(
            "test_p1_manager_safety.TestPgPartitionDropAndTruncateTriggerDrift."
            "test_pg_partition_drop_and_truncate_trigger_drift_detection"
        ),
    ),
    Mutant(
        name="pg-expr-probe-skips-dedup-setop-output-cols",
        source_path="matview/dialects/pg_statement.py",
        original="        if has_dedup_setop:\n            resolved_cols: Optional[List[str]] = None",
        replacement="        if False and has_dedup_setop:\n            resolved_cols: Optional[List[str]] = None",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_registration_expr_type_probing_complex_shapes_codegen"
        ),
    ),
    Mutant(
        name="pg-expr-probe-drops-with-prefix",
        source_path="matview/dialects/pg_statement.py",
        original="            prefix = raw_sql[toks[lo].start : toks[i].start].strip() + \"\\n\"",
        replacement="            prefix = \"\"",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_registration_expr_type_probing_complex_shapes_codegen"
        ),
    ),
    Mutant(
        name="dotted-base-table-q-splits-known-ident",
        source_path="matview/dialects/base.py",
        original="        if \".\" in s and self._is_known_single_ident(s):",
        replacement="        if False and \".\" in s and self._is_known_single_ident(s):",
        target_test=(
            "test_p1_sqlglot_builders.TestSqlglotExpBuilders."
            "test_dotted_base_tables_and_special_char_user_indexes"
        ),
    ),
    Mutant(
        name="backfill-dotted-table-splits-qualified",
        source_path="matview/manager.py",
        original="            base_t = quote_sql_identifier(plan.view_spec.base_table, self.dialect, allow_qualified=False)",
        replacement="            base_t = quote_sql_identifier(plan.view_spec.base_table, self.dialect)",
        target_test=(
            "test_p1_sqlglot_builders.TestSqlglotExpBuilders."
            "test_dotted_base_tables_and_special_char_user_indexes"
        ),
    ),
    Mutant(
        name="user-index-backing-col-map-drops-special-alias",
        source_path="matview/maintenance/single.py",
        original="                r'\\s+AS\\s+(?:\"(?:\"\"|[^\"])+\"|`(?:``|[^`])+`|\\[[^\\]]+\\]|[A-Za-z0-9_]+)\\s*$',",
        replacement="                r'\\s+AS\\s+[A-Za-z0-9_\"`\\[\\]]+\\s*$',",
        target_test=(
            "test_p1_sqlglot_builders.TestSqlglotExpBuilders."
            "test_dotted_base_tables_and_special_char_user_indexes"
        ),
    ),
    Mutant(
        name="user-index-bare-compound-root-replace",
        source_path="matview/planner.py",
        original="            if isinstance(copied, sqlglot_exp.Column) and copied.name and not copied.table:",
        replacement="            if False and isinstance(copied, sqlglot_exp.Column) and copied.name and not copied.table:",
        target_test=(
            "test_p1_sqlglot_builders.TestSqlglotExpBuilders."
            "test_dotted_base_tables_and_special_char_user_indexes"
        ),
    ),
    Mutant(
        name="zset-union-populate-unquoted-cols",
        source_path="matview/maintenance/zset.py",
        original="            pk_u = self._canonical_group_pk_expr(q_user_columns)\n            grp_u = \", \".join(self._group_by_exprs_for_query(q_user_columns, include_canonical_pk=True))\n            initial_populate_sqls.append(\n                f\"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\\n\"",
        replacement="            pk_u = self._canonical_group_pk_expr(user_columns)\n            grp_u = \", \".join(self._group_by_exprs_for_query(user_columns, include_canonical_pk=True))\n            initial_populate_sqls.append(\n                f\"INSERT INTO {mv_table} (__mv_pk, {', '.join(q_user_columns)}, __mv_row_count)\\n\"",
        target_test=(
            "test_p1_sqlglot_builders.TestSqlglotExpBuilders."
            "test_dotted_base_tables_and_special_char_user_indexes"
        ),
    ),
    Mutant(
        name="mysql-repair-skips-repeatable-read-base-locks",
        source_path="matview/manager.py",
        original="                    if self.dialect == Dialect.MYSQL and mysql_lock_stmts:",
        replacement="                    if False and self.dialect == Dialect.MYSQL and mysql_lock_stmts:",
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_repair_view_contents_sets_repeatable_read_locks_base_tables_and_retries_on_deadlock"
        ),
    ),
    Mutant(
        name="mysql-lock-planner-skips-looser-cte-name-with-physical-table",
        source_path="matview/manager.py",
        original="                    if _is_postgres_cte_in_scope(tbl_node, mysql_strict=True):\n                        continue",
        replacement="                    if _is_postgres_cte_in_scope(tbl_node, case_insensitive=True):\n                        continue",
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_lock_planner_locks_a_table_that_is_only_a_cte_under_looser_server_rules"
        ),
    ),
    Mutant(
        name="mysql-lock-planner-dedup-drops-db-qualifier",
        source_path="matview/manager.py",
        original='                    key = (db_key, this_part.name)',
        replacement='                    key = ("", this_part.name)',
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_lock_planner_dedups_by_exact_db_and_name_and_skips_internal_objects"
        ),
    ),
    Mutant(
        name="mysql-lock-planner-dedup-casefolds-table-name",
        source_path="matview/manager.py",
        original='                    key = (db_key, this_part.name)',
        replacement='                    key = (db_key, this_part.name.lower())',
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_lock_planner_dedups_by_exact_db_and_name_and_skips_internal_objects"
        ),
    ),
    Mutant(
        name="mysql-lock-planner-locks-internal-objects",
        source_path="matview/manager.py",
        original='                    if leaf_low.startswith(internal_prefixes):\n                        continue',
        replacement='                    if False and leaf_low.startswith(internal_prefixes):\n                        continue',
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_lock_planner_dedups_by_exact_db_and_name_and_skips_internal_objects"
        ),
    ),
    Mutant(
        name="mysql-lock-planner-self-qualified-table-not-merged",
        source_path="matview/manager.py",
        original='                    if current_db is not None and db_key == current_db:',
        replacement='                    if False and current_db is not None and db_key == current_db:',
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_lock_planner_dedups_by_exact_db_and_name_and_skips_internal_objects"
        ),
    ),
    Mutant(
        name="mysql-lock-planner-ignores-foreign-db-schema-lookup",
        source_path="matview/manager.py",
        original='            ts = foreign_schema(db_key, leaf_name) if foreign_schema is not None else None',
        replacement='            ts = None',
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_lock_planner_dedups_by_exact_db_and_name_and_skips_internal_objects"
        ),
    ),
    Mutant(
        name="mysql-recompute-subquery-skips-lock-in-share-mode",
        source_path="matview/maintenance/single.py",
        original="            lock_part = \" LOCK IN SHARE MODE\" if self.dialect == Dialect.MYSQL else \"\"",
        replacement="            lock_part = \"\"",
        target_test=(
            "test_concurrent_backfill.TestMySqlRepairViewLockingOffline."
            "test_mysql_minmax_and_bitwise_recompute_subqueries_emit_lock_in_share_mode"
        ),
    ),
    Mutant(
        name="sqlite-var-distinct-drops-deduplication",
        source_path="matview/dialects/sqlite.py",
        original="                if is_distinct and fn_up in (",
        replacement="                if False and is_distinct and fn_up in (",
        target_test=(
            "test_sqlite_distinct_filtered_aggregates.TestDistinctAndFilteredStatBitBool."
            "test_exact_differential_deterministic_scenarios"
        ),
    ),
    Mutant(
        name="pg-union-array-skips-mixed-dimensions",
        source_path="matview/dialects/base.py",
        original="            if all(m is not None for m in arr_matches):\n                elems = [m.group(1).strip() for m in arr_matches if m is not None]",
        replacement="            if all(m is not None for m in arr_matches) and len({m.group(2) for m in arr_matches if m is not None}) == 1:\n                elems = [m.group(1).strip() for m in arr_matches if m is not None]",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_cross_branch_union_array_scalar_and_collation_widening_offline"
        ),
    ),
    Mutant(
        name="pg-union-drops-nondefault-collation-with-default-branch",
        source_path="matview/dialects/base.py",
        original="                nonempty_imp = [c for c in implicit_colls if c]\n                if nonempty_imp:",
        replacement="                nonempty_imp = [c for c in implicit_colls if c]\n                if nonempty_imp and all(implicit_colls):",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_cross_branch_union_array_scalar_and_collation_widening_offline"
        ),
    ),
    Mutant(
        name="pg-tier3-setop-quantifier-left-on-arm",
        source_path="matview/maintenance/tier3.py",
        original="                if bi > 0:\n                    # The token splitter leaves",
        replacement="                if False and bi > 0:\n                    # The token splitter leaves",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_set_operation_quantifiers_nested_collation_and_arm_chain_offline"
        ),
    ),
    Mutant(
        name="pg-union-nested-explicit-collation-ignored",
        source_path="matview/dialects/base.py",
        original="        coll, explicit = self._infer_pg_expr_collation_and_strength(expr)\n        return coll if explicit else None",
        replacement="        coll, explicit = self._infer_pg_expr_collation_and_strength(expr)\n        return None",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_set_operation_quantifiers_nested_collation_and_arm_chain_offline"
        ),
    ),
    Mutant(
        name="pg-union-arm-chain-through-real-not-refused",
        source_path="matview/dialects/base.py",
        original="                if rank == 5 and wide and not is_root:",
        replacement="                if False and rank == 5 and wide and not is_root:",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_set_operation_pairwise_tree_type_resolution_offline"
        ),
    ),
    Mutant(
        name="pg-union-arm-chain-ignores-set-operation-tree",
        source_path="matview/dialects/base.py",
        original="        tree = arm_tree\n        if tree is None:",
        replacement="        tree = None\n        if tree is None:",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_set_operation_pairwise_tree_type_resolution_offline"
        ),
    ),
    Mutant(
        name="pg-union-nested-arms-typed-by-first-select",
        source_path="matview/maintenance/tier3.py",
        original="            if len(leaf_nodes) != len(b_sqls):",
        replacement="            if False and len(leaf_nodes) != len(b_sqls):",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_set_operation_pairwise_tree_type_resolution_offline"
        ),
    ),
    Mutant(
        name="pg-union-trigger-qualifies-collation-name",
        source_path="matview/maintenance/zset.py",
        original="not is_after_dot and not is_before_dot and not is_fn_call and not is_collation_name:",
        replacement="not is_after_dot and not is_before_dot and not is_fn_call:",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_array_scalar_inference_collation_and_unmodelled_arms_offline"
        ),
    ),
    Mutant(
        name="pg-union-arm-typed-from-base-table-through-cte-output-name",
        source_path="matview/dialects/base.py",
        original="                    and not (bs.has_cte or bs.has_subquery_in_from)\n",
        replacement="",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_set_operation_pairwise_tree_type_resolution_offline"
        ),
    ),
    Mutant(
        name="pg-star-over-setop-relation-accepted",
        source_path="matview/planner.py",
        original="            if isinstance(star_ast, sqlglot_exp.Select) and any(\n                n is not None",
        replacement="            if False and isinstance(star_ast, sqlglot_exp.Select) and any(\n                n is not None",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_array_scalar_inference_collation_and_unmodelled_arms_offline"
        ),
    ),
    Mutant(
        name="pg-citext-mixed-with-char-varchar-widens-to-text",
        source_path="matview/dialects/base.py",
        original='                if any(b != "CITEXT" and b != "TEXT" for b in bases):',
        replacement='                if False and any(b != "CITEXT" and b != "TEXT" for b in bases):',
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_array_scalar_inference_collation_and_unmodelled_arms_offline"
        ),
    ),
    Mutant(
        name="pg-union-zset-skips-branch-cast",
        source_path="matview/maintenance/zset.py",
        original="                if col_i < len(col_types) and self._pg_branch_expr_needs_cast(it, b_ctx, col_types[col_i]):",
        replacement="                if False and col_i < len(col_types) and self._pg_branch_expr_needs_cast(it, b_ctx, col_types[col_i]):",
        target_test=(
            "test_native_result_types.SetopCommonTypesPgCatalogAndNameTests."
            "test_postgres_cross_branch_union_array_scalar_and_collation_widening_offline"
        ),
    ),
    Mutant(
        name="window-cte-truncates-to-int",
        source_path="matview/dialects/base.py",
        original=(
            "            if isinstance(inner, sqlglot_exp.Window):\n"
            "                win_sql = inner.sql(_sqlglot_dialect(self.dialect))"
        ),
        replacement=(
            "            if isinstance(inner, sqlglot_exp.Window):\n"
            "                return self._int_type()\n"
            "                win_sql = inner.sql(_sqlglot_dialect(self.dialect))"
        ),
        target_test=(
            "test_native_result_types.ExactWindowExpressionTypesTests."
            "test_exact_window_expression_types_and_fail_closed_offline"
        ),
    ),
    Mutant(
        name="window-filter-skips-float-refusal",
        source_path="matview/planner.py",
        original="            if isinstance(fn, sqlglot_exp.Filter) and fn.this is not None:\n                fn = fn.this",
        replacement="            if False and isinstance(fn, sqlglot_exp.Filter) and fn.this is not None:\n                fn = fn.this",
        target_test=(
            "test_native_result_types.ExactWindowExpressionTypesTests."
            "test_exact_window_expression_types_and_fail_closed_offline"
        ),
    ),
    Mutant(
        name="pg-bracket-omits-reentrant-flag",
        source_path="matview/dialects/templates.py",
        original="    __mv_overlap BOOLEAN := ((__mv_m & (__mv_m - 1)) <> 0) OR (__mv_re >= 2) OR (__mv_re <> 0 AND __mv_nd > 0);",
        replacement="    __mv_overlap BOOLEAN := (__mv_m & (__mv_m - 1)) <> 0;",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_reentrant_nested_trigger_bracket_codegen"
        ),
    ),
    Mutant(
        name="pg-single-table-omits-reentrant-bracket",
        source_path="matview/maintenance/single.py",
        original="            pg_bracket = PgStatementBracket(\n                lock_table=mv_table,\n                source_bit=1,",
        replacement="            pg_bracket = None\n            if False:\n                pg_bracket = PgStatementBracket(\n                lock_table=mv_table,\n                source_bit=1,",
        target_test=(
            "test_p1_engine_codegen.TestNativeEngineCodegenExecution."
            "test_postgres_reentrant_nested_trigger_bracket_codegen"
        ),
    ),
    Mutant(
        name="sqlite-var-pop-misclassified-as-unrecognized",
        source_path="matview/planner_exact_routing.py",
        original='        "VAR_POP", "VARIANCE", "VARIANCE_POP", "VAR_SAMP", "VARIANCE_SAMP",',
        replacement='        "VAR_POP", "VARIANCE", "VAR_SAMP",',
        target_test=(
            "test_sqlite_distinct_filtered_aggregates.TestDistinctAndFilteredStatBitBool."
            "test_distinct_group_recompute_diagnostics_consistent_across_aggregates_and_dialects"
        ),
    ),
    Mutant(
        name="distinct-group-recompute-fallback-omitted",
        source_path="matview/planner_exact_routing.py",
        original="            return (\n                self._unrecognized_aggregate_group_fallback()\n                or self._distinct_aggregate_group_fallback()\n            )",
        replacement="            return self._unrecognized_aggregate_group_fallback()",
        target_test=(
            "test_sqlite_distinct_filtered_aggregates.TestDistinctAndFilteredStatBitBool."
            "test_distinct_group_recompute_diagnostics_consistent_across_aggregates_and_dialects"
        ),
    ),
    Mutant(
        name="mysql-sa-format-drops-temporal-fsp",
        source_path="matview/connectivity.py",
        original='    elif dialect == Dialect.MYSQL and raw_str in ("TIME", "DATETIME", "TIMESTAMP"):\n        # SQLAlchemy prints MySQL TIME(fsp=n)/DATETIME(fsp=n)/TIMESTAMP(fsp=n)\n        # without their fractional-seconds precision, silently truncating or\n        # rounding sub-second values in backing columns and MIN/MAX deltas.\n        fsp = getattr(col_type, "fsp", None)\n        if fsp is not None and int(fsp) > 0:\n            raw_str = f"{raw_str}({int(fsp)})"',
        replacement='    elif False and dialect == Dialect.MYSQL and raw_str in ("TIME", "DATETIME", "TIMESTAMP"):\n        fsp = getattr(col_type, "fsp", None)\n        if fsp is not None and int(fsp) > 0:\n            raw_str = f"{raw_str}({int(fsp)})"',
        target_test=(
            "test_native_result_types.MySQLPlannedResultTypeTests."
            "test_temporal_precision_introspection_and_expression_types"
        ),
    ),
    Mutant(
        name="mysql-dbapi-introspect-drops-temporal-fsp",
        source_path="matview/connectivity.py",
        original='                "INT", "INTEGER", "SMALLINT", "MEDIUMINT", "BIGINT", "FLOAT", "DOUBLE", "REAL",\n                "TIME", "DATETIME", "TIMESTAMP", "BIT", "BINARY", "VARBINARY",',
        replacement='                "INT", "INTEGER", "SMALLINT", "MEDIUMINT", "BIGINT", "FLOAT", "DOUBLE", "REAL",',
        target_test=(
            "test_native_result_types.MySQLPlannedResultTypeTests."
            "test_temporal_precision_introspection_and_expression_types"
        ),
    ),
    Mutant(
        name="mysql-ast-expr-drops-temporal-fsp",
        source_path="matview/dialects/base.py",
        original='            if self.dialect == Dialect.MYSQL:\n                m_my_temp = re.fullmatch(\n                    r"(TIME|DATETIME|TIMESTAMP)(?:\\s*\\(\\s*(\\d+)\\s*\\))?",\n                    bare.strip(),\n                )',
        replacement='            if False and self.dialect == Dialect.MYSQL:\n                m_my_temp = re.fullmatch(\n                    r"(TIME|DATETIME|TIMESTAMP)(?:\\s*\\(\\s*(\\d+)\\s*\\))?",\n                    bare.strip(),\n                )',
        target_test=(
            "test_native_result_types.MySQLPlannedResultTypeTests."
            "test_temporal_precision_introspection_and_expression_types"
        ),
    ),
    Mutant(
        name="mysql-backing-timestamp-omits-explicit-null",
        source_path="matview/dialects/base.py",
        original='                        elif not re.search(r"\\bNULL\\b", tail_up):\n                            line = f"{head} NULL DEFAULT NULL{tail}"',
        replacement='                        elif False and not re.search(r"\\bNULL\\b", tail_up):\n                            line = f"{head} NULL DEFAULT NULL{tail}"',
        target_test=(
            "test_native_result_types.MySQLPlannedResultTypeTests."
            "test_temporal_precision_introspection_and_expression_types"
        ),
    ),
    Mutant(
        name="pg-row-pseudotype-output-not-refused",
        source_path="matview/planner.py",
        original="        if self.dialect == Dialect.POSTGRES:\n            self._enforce_pg_no_pseudotype_outputs()",
        replacement="        if False and self.dialect == Dialect.POSTGRES:\n            self._enforce_pg_no_pseudotype_outputs()",
        target_test=(
            "test_native_result_types.PostgresPseudotypeOutputRefusalTests."
            "test_postgres_row_and_pseudotype_outputs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-tuple-row-constructor-not-record",
        source_path="matview/dialects/base.py",
        original='                if kind == "Tuple":\n                    return "RECORD"',
        replacement='                if False and kind == "Tuple":\n                    return "RECORD"',
        target_test=(
            "test_native_result_types.PostgresPseudotypeOutputRefusalTests."
            "test_postgres_row_and_pseudotype_outputs_refused_offline"
        ),
    ),
    Mutant(
        name="sql-frontend-unwrap-parens-strips-tuple",
        source_path="matview/sql_frontend.py",
        original="        if closes_at_end and not has_top_level_comma:",
        replacement="        if closes_at_end:",
        target_test=(
            "test_native_result_types.PostgresPseudotypeOutputRefusalTests."
            "test_postgres_row_and_pseudotype_outputs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-setop-pseudotype-output-not-refused",
        source_path="matview/dialects/postgres.py",
        original="            if isinstance(q_node, sqlglot_exp.SetOperation):",
        replacement="            if False and isinstance(q_node, sqlglot_exp.SetOperation):",
        target_test=(
            "test_native_result_types.PostgresPseudotypeOutputRefusalTests."
            "test_postgres_row_and_pseudotype_outputs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-whole-row-and-syscol-guard-omitted",
        source_path="matview/planner.py",
        original="        if self.dialect == Dialect.POSTGRES:\n            self._enforce_pg_no_whole_row_or_system_columns()",
        replacement="        if False and self.dialect == Dialect.POSTGRES:\n            self._enforce_pg_no_whole_row_or_system_columns()",
        target_test=(
            "test_native_result_types.PostgresWholeRowAndSystemColumnRefusalTests."
            "test_postgres_whole_row_and_system_columns_refused_offline"
        ),
    ),
    Mutant(
        name="pg-whole-row-star-arg-not-refused",
        source_path="matview/dialects/postgres.py",
        original="                        if any(tbl_key in rels for rels, _ in scope_stack):\n                            raise UnsupportedMaterializedViewError(\n                                f\"PostgreSQL materialized view references whole-row value \"",
        replacement="                        if False and any(tbl_key in rels for rels, _ in scope_stack):\n                            raise UnsupportedMaterializedViewError(\n                                f\"PostgreSQL materialized view references whole-row value \"",
        target_test=(
            "test_native_result_types.PostgresWholeRowAndSystemColumnRefusalTests."
            "test_postgres_whole_row_and_system_columns_refused_offline"
        ),
    ),
    Mutant(
        name="pg-qualified-system-col-not-refused",
        source_path="matview/dialects/postgres.py",
        original="                    if _is_sys_col_ident(col_ident):\n                        tbl_key = _ident_key(tbl_id)",
        replacement="                    if False and _is_sys_col_ident(col_ident):\n                        tbl_key = _ident_key(tbl_id)",
        target_test=(
            "test_native_result_types.PostgresWholeRowAndSystemColumnRefusalTests."
            "test_postgres_whole_row_and_system_columns_refused_offline"
        ),
    ),
    Mutant(
        name="pg-lateral-source-skips-whole-row-walk",
        source_path="matview/dialects/postgres.py",
        original="                # Bare table-valued expression (e.g. `UNNEST(...)`, `LATERAL jsonb_each(...)`).\n                _walk_expr(src, query_node, cur_stack, local_ctes, set())",
        replacement="                # Bare table-valued expression (e.g. `UNNEST(...)`, `LATERAL jsonb_each(...)`).\n                pass",
        target_test=(
            "test_native_result_types.PostgresWholeRowAndSystemColumnRefusalTests."
            "test_postgres_whole_row_and_system_columns_refused_offline"
        ),
    ),
    Mutant(
        name="pg-inner-scope-syscol-masked-by-outer-cte",
        source_path="matview/dialects/postgres.py",
        original="                        for rels, bases in reversed(scope_stack):\n                            if any(_has_user_col(cs, col_ident) for cs in rels.values()):",
        replacement="                        for rels, bases in scope_stack:\n                            if any(_has_user_col(cs, col_ident) for cs in rels.values()):",
        target_test=(
            "test_native_result_types.PostgresWholeRowAndSystemColumnRefusalTests."
            "test_postgres_whole_row_and_system_columns_refused_offline"
        ),
    ),
    Mutant(
        name="pg-stable-expr-guard-omitted",
        source_path="matview/planner.py",
        original="            self._enforce_pg_no_stable_or_session_dependent_exprs()",
        replacement="            pass",
        target_test=(
            "test_native_result_types.PostgresStableAndSessionDependentExprRefusalTests."
            "test_postgres_stable_and_session_dependent_exprs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-timestamptz-to-date-cast-not-refused",
        source_path="matview/dialects/postgres.py",
        original='            if src_type == "TIMESTAMPTZ" and dst_type in ("DATE", "TIMESTAMP", "TIME", "TIMETZ"):',
        replacement='            if src_type == "TIMESTAMPTZ" and dst_type in ("TIMESTAMP", "TIME", "TIMETZ"):',
        target_test=(
            "test_native_result_types.PostgresStableAndSessionDependentExprRefusalTests."
            "test_postgres_stable_and_session_dependent_exprs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-timestamptz-to-text-cast-not-refused",
        source_path="matview/dialects/postgres.py",
        original='            if dst_type == "STRING" and _pg_type_has_session_output(src_type, for_json=False):',
        replacement='            if False and dst_type == "STRING" and _pg_type_has_session_output(src_type, for_json=False):',
        target_test=(
            "test_native_result_types.PostgresStableAndSessionDependentExprRefusalTests."
            "test_postgres_stable_and_session_dependent_exprs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-date-trunc-2arg-timestamptz-not-refused",
        source_path="matview/dialects/postgres.py",
        original='                if x_t == "TIMESTAMPTZ" and zone_arg is None:',
        replacement='                if False and x_t == "TIMESTAMPTZ" and zone_arg is None:',
        target_test=(
            "test_native_result_types.PostgresStableAndSessionDependentExprRefusalTests."
            "test_postgres_stable_and_session_dependent_exprs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-mixed-tz-coercion-not-refused",
        source_path="matview/dialects/postgres.py",
        original='            if "TIMESTAMPTZ" in present and (present & {"TIMESTAMP", "DATE"}):',
        replacement='            if False and "TIMESTAMPTZ" in present and (present & {"TIMESTAMP", "DATE"}):',
        target_test=(
            "test_native_result_types.PostgresStableAndSessionDependentExprRefusalTests."
            "test_postgres_stable_and_session_dependent_exprs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-array-cast-unwrap-omitted",
        source_path="matview/dialects/postgres.py",
        original='            while src_type.startswith("ARRAY:") and dst_type.startswith("ARRAY:"):',
        replacement='            while False and src_type.startswith("ARRAY:") and dst_type.startswith("ARRAY:"):',
        target_test=(
            "test_native_result_types.PostgresStableAndSessionDependentExprRefusalTests."
            "test_postgres_stable_and_session_dependent_exprs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-date-trunc-on-date-not-refused",
        source_path="matview/dialects/postgres.py",
        original='                if x_t == "DATE":\n                    _refuse(\n                        node,\n                        "DATE_TRUNC on DATE implicitly casts DATE to TIMESTAMPTZ using session TimeZone",',
        replacement='                if False and x_t == "DATE":\n                    _refuse(\n                        node,\n                        "DATE_TRUNC on DATE implicitly casts DATE to TIMESTAMPTZ using session TimeZone",',
        target_test=(
            "test_native_result_types.PostgresStableAndSessionDependentExprRefusalTests."
            "test_postgres_stable_and_session_dependent_exprs_refused_offline"
        ),
    ),
    Mutant(
        name="pg-unaliased-fn-colname-disabled",
        source_path="matview/sql_frontend.py",
        original="                _pg_default_function_col_name(inner_node, expr)\n                if dialect == Dialect.POSTGRES and isinstance(inner_node, sqlglot_exp.Expr)\n                else None",
        replacement="                None",
        target_test=(
            "test_unrecognized_aggregates.OfflineRoutingTests."
            "test_pg_json_and_array_agg_types_aliases_windows_and_backfill"
        ),
    ),
    Mutant(
        name="pg-window-json-agg-falls-back-to-text",
        source_path="matview/dialects/base.py",
        original="            if self.dialect == Dialect.POSTGRES and isinstance(inner_node, sqlglot_exp.Expr):\n                return self._aggregate_call_storage_type(inner_node)",
        replacement="            if False and self.dialect == Dialect.POSTGRES and isinstance(inner_node, sqlglot_exp.Expr):\n                return self._aggregate_call_storage_type(inner_node)",
        target_test=(
            "test_unrecognized_aggregates.OfflineRoutingTests."
            "test_pg_json_and_array_agg_types_aliases_windows_and_backfill"
        ),
    ),
    Mutant(
        name="pg-array-length-and-to-string-type-omitted",
        source_path="matview/dialects/base.py",
        original='                if kind in ("ArraySize", "ArrayPosition"):\n                    return "INTEGER"\n                if kind == "ArrayToString":\n                    return "TEXT"',
        replacement='                if kind in ("ArraySize", "ArrayPosition", "ArrayToString"):\n                    return "DOUBLE PRECISION"',
        target_test=(
            "test_unrecognized_aggregates.OfflineRoutingTests."
            "test_pg_json_and_array_agg_types_aliases_windows_and_backfill"
        ),
    ),
    Mutant(
        name="pg-array-agg-unmodelled-compound-guesses-float",
        source_path="matview/dialects/base.py",
        original="                if has_array_agg_call:",
        replacement="                if False and has_array_agg_call:",
        target_test=(
            "test_unrecognized_aggregates.OfflineRoutingTests."
            "test_pg_json_and_array_agg_types_aliases_windows_and_backfill"
        ),
    ),
    Mutant(
        name="pg-array-agg-collation-dropped",
        source_path="matview/planner_exact_routing.py",
        original="            if first is not None and self._pg_type_supports_collation(arr_t):",
        replacement="            if False and first is not None and self._pg_type_supports_collation(arr_t):",
        target_test=(
            "test_unrecognized_aggregates.OfflineRoutingTests."
            "test_pg_json_and_array_agg_types_aliases_windows_and_backfill"
        ),
    ),
    Mutant(
        name="pg-json-offset-backfill-order-by-uncast",
        source_path="matview/dialects/base.py",
        original='        if base in (\n            "JSON",\n            "XML",\n            "POINT",',
        replacement='        if False and base in (\n            "JSON",\n            "XML",\n            "POINT",',
        target_test=(
            "test_unrecognized_aggregates.OfflineRoutingTests."
            "test_pg_json_and_array_agg_types_aliases_windows_and_backfill"
        ),
    ),
    Mutant(
        name="mariadb-nullable-unique-skips-before-count",
        source_path="matview/dialects/emitters.py",
        original="            if not mysql_prelock and not bu_count_stmt:\n                return [trig_ins, trig_del, trig_upd], t_names, []",
        replacement="            if not mysql_prelock:\n                return [trig_ins, trig_del, trig_upd], t_names, []",
        target_test=(
            "test_p1_engine_codegen.TestMariaDBUpdateIgnoreGuard."
            "test_mariadb_update_ignore_guard_codegen_shapes"
        ),
    ),
    Mutant(
        name="mariadb-nullable-unique-applies-null-unconditionally",
        source_path="matview/dialects/emitters.py",
        original="                            else_expr = f\"(({new_all_have_null}) OR {after_count_lt})\"",
        replacement="                            else_expr = \"TRUE\"",
        target_test=(
            "test_p1_engine_codegen.TestMariaDBUpdateIgnoreGuard."
            "test_mariadb_update_ignore_guard_codegen_shapes"
        ),
    ),
    Mutant(
        name="mariadb-second-nullable-unique-ignored",
        source_path="matview/dialects/emitters.py",
        original="                        for uq in nullable_uniques:",
        replacement="                        for uq in nullable_uniques[:1]:",
        target_test=(
            "test_p1_engine_codegen.TestMariaDBUpdateIgnoreGuard."
            "test_mariadb_update_ignore_guard_codegen_shapes"
        ),
    ),
    Mutant(
        name="mariadb-keyless-fk-guard-omitted",
        source_path="matview/dialects/emitters.py",
        original="                elif nullable_uniques or has_nonkey_skip:",
        replacement="                elif nullable_uniques:",
        target_test=(
            "test_p1_engine_codegen.TestMariaDBUpdateIgnoreGuard."
            "test_mariadb_update_ignore_guard_codegen_shapes"
        ),
    ),
    Mutant(
        name="mysql-pk-range-batch-omits-repeatable-read",
        source_path="matview/manager.py",
        original='                self._rollback()\n                cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;")\n                cur.execute("START TRANSACTION;")',
        replacement='                cur.execute("START TRANSACTION;")',
        target_test=(
            "test_concurrent_backfill.TestMySqlChunkedBackfillConcurrency."
            "test_mysql_pk_range_batch_sets_repeatable_read_and_fails_closed_on_for_update_error"
        ),
    ),
    Mutant(
        name="mysql-pk-range-batch-swallows-for-update-error",
        source_path="matview/manager.py",
        original='                self._execute_param(\n                    cur,\n                    f"SELECT {pk_col} FROM {base_t} WHERE {pk_col} > ? AND {pk_col} <= ? FOR UPDATE",\n                    (cursor_pk, upper_pk),\n                )\n                cur.fetchall()',
        replacement='                try:\n                    self._execute_param(\n                        cur,\n                        f"SELECT {pk_col} FROM {base_t} WHERE {pk_col} > ? AND {pk_col} <= ? FOR UPDATE",\n                        (cursor_pk, upper_pk),\n                    )\n                    cur.fetchall()\n                except Exception as exc:\n                    if self._is_transient_lock_error(exc):\n                        raise',
        target_test=(
            "test_concurrent_backfill.TestMySqlChunkedBackfillConcurrency."
            "test_mysql_pk_range_batch_sets_repeatable_read_and_fails_closed_on_for_update_error"
        ),
    ),
    Mutant(
        name="mysql-pk-range-batch-omits-1020-retry",
        source_path="matview/manager.py",
        original="                                or self._is_snapshot_row_changed_error(exc)",
        replacement="                                or False",
        target_test=(
            "test_concurrent_backfill.TestMySqlChunkedBackfillConcurrency."
            "test_mysql_pk_range_batch_retries_1020_on_chunk_but_fails_closed_on_catalog_takeover"
        ),
    ),
    Mutant(
        name="mysql-single-shot-claim-drops-tx-on-retry",
        source_path="matview/manager.py",
        original="                if on_retry_begin is not None:\n                    on_retry_begin()",
        replacement="                if False and on_retry_begin is not None:\n                    on_retry_begin()",
        target_test=(
            "test_concurrent_backfill.TestMySqlChunkedBackfillConcurrency."
            "test_mysql_pk_range_batch_sets_repeatable_read_and_fails_closed_on_for_update_error"
        ),
    ),
    Mutant(
        name="mariadb-partitioned-guard-omitted",
        source_path="matview/dialects/emitters.py",
        original='            or getattr(t_schema, "is_partitioned", False)',
        replacement="            or False",
        target_test=(
            "test_p1_engine_codegen.TestMariaDBUpdateIgnoreGuard."
            "test_mariadb_update_ignore_guard_codegen_shapes"
        ),
    ),
    Mutant(
        name="mariadb-nullable-unique-nonkey-skip-shortcircuit-not-disabled",
        source_path="matview/dialects/emitters.py",
        original="                        if not has_nonkey_skip:",
        replacement="                        if True:",
        target_test=(
            "test_p1_engine_codegen.TestMariaDBUpdateIgnoreGuard."
            "test_mariadb_update_ignore_guard_codegen_shapes"
        ),
    ),
    Mutant(
        name="mariadb-timestamp-binary-cast-bug-reintroduced",
        source_path="matview/dialects/emitters.py",
        original='                if dt_head in ("TIMESTAMP", "DATETIME", "TIME", "DATE", "YEAR"):',
        replacement='                if dt_head in ("DATETIME", "TIME", "DATE", "YEAR"):',
        target_test=(
            "test_p1_engine_codegen.TestMariaDBUpdateIgnoreGuard."
            "test_mariadb_update_ignore_guard_codegen_shapes"
        ),
    ),
)


def _target_test_module(target_test: str) -> str:
    parts = [p.strip() for p in target_test.split(".")]
    if len(parts) < 3 or any(not p or not p.isidentifier() for p in parts):
        raise RuntimeError(
            f"invalid target_test {target_test!r}: expected 'module.TestClass.test_method'"
        )
    return parts[0]


def _resolve_source(root: Path, mutant: Mutant) -> Path:
    relative = Path(mutant.source_path)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{mutant.name}: source path must be a non-empty repository-relative path")
    if relative.parts[0] != PACKAGE_DIR and relative.as_posix() not in COPIED_FILES:
        raise RuntimeError(
            f"{mutant.name}: {mutant.source_path} is not part of the copied mutation tree"
        )
    source = root / relative
    if not source.is_file():
        raise RuntimeError(f"{mutant.name}: mutation source {mutant.source_path} does not exist")
    return source


def _require_single_site(root: Path, mutant: Mutant) -> str:
    source = _resolve_source(root, mutant)
    contents = source.read_text(encoding="utf-8")
    occurrences = contents.count(mutant.original)
    if occurrences != 1:
        raise RuntimeError(
            f"{mutant.name}: expected exactly one mutation site in "
            f"{mutant.source_path}, found {occurrences}"
        )
    if mutant.original == mutant.replacement:
        raise RuntimeError(f"{mutant.name}: replacement is identical to the original")
    return contents


def validate_catalog(mutants: Sequence[Mutant]) -> None:
    names = [m.name for m in mutants]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate mutant names: {', '.join(duplicates)}")
    for mutant in mutants:
        _require_single_site(ROOT, mutant)
        target_mod = _target_test_module(mutant.target_test)
        if not (ROOT / f"{target_mod}.py").is_file():
            raise RuntimeError(
                f"{mutant.name}: target test module {target_mod}.py does not exist"
            )


def _copy_tree(worktree: Path, target_module: str) -> None:
    shutil.copytree(
        ROOT / PACKAGE_DIR,
        worktree / PACKAGE_DIR,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    files_to_copy: List[str] = list(COPIED_FILES)
    target_file = f"{target_module}.py"
    if target_file not in files_to_copy:
        files_to_copy.append(target_file)
    for filename in files_to_copy:
        shutil.copy2(ROOT / filename, worktree / filename)


def _run(root: Path, args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _require_isolated_import(worktree: Path, label: str, target_module: str) -> None:
    expected = worktree.resolve()
    repo_root = ROOT.resolve()
    probe = _run(
        worktree,
        [
            "-c",
            "import importlib, pathlib, sys; "
            f"mods = [importlib.import_module(m) for m in ('matview', 'matview_sync', {target_module!r})]; "
            f"wt = pathlib.Path({str(expected)!r}).resolve(); "
            f"repo = pathlib.Path({str(repo_root)!r}).resolve(); "
            "leaked = ["
            "  (name, str(pathlib.Path(getattr(mod, '__file__', '')).resolve())) "
            "  for name, mod in list(sys.modules.items()) "
            "  if getattr(mod, '__file__', None) "
            "  and repo in pathlib.Path(mod.__file__).resolve().parents "
            "  and wt not in pathlib.Path(mod.__file__).resolve().parents"
            "]; "
            "assert not leaked, f'leaked repo imports: {leaked}'; "
            "print('\\n'.join(str(pathlib.Path(m.__file__).resolve()) for m in mods))",
        ],
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"{label}: isolated import probe failed.\n{probe.stdout}"
        )
    lines = probe.stdout.strip().splitlines()
    paths = [Path(line).resolve() for line in lines[-3:]] if len(lines) >= 3 else []
    if len(paths) != 3 or any(expected not in p.parents for p in paths):
        raise RuntimeError(
            f"{label}: mutation copy is not the imported implementation "
            f"(expected modules under {expected}).\n{probe.stdout}"
        )


def _run_test(root: Path, target: str) -> subprocess.CompletedProcess[str]:
    return _run(root, ["-m", "unittest", target, "-v"])


def _require_baseline_pass(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0 or "Ran 1 test" not in result.stdout or "\nOK" not in result.stdout:
        raise RuntimeError(
            f"{label}: baseline regression did not pass cleanly; invalid mutation run.\n"
            f"{result.stdout}"
        )


def _is_assertion_kill(result: subprocess.CompletedProcess[str]) -> bool:
    output = result.stdout
    return (
        result.returncode != 0
        and "Ran 1 test" in output
        and "FAILED (failures=" in output
        and "Traceback (most recent call last)" in output
        and "AssertionError" in output
    )


def run_mutation(mutant: Mutant) -> None:
    target_module = _target_test_module(mutant.target_test)
    with tempfile.TemporaryDirectory(prefix="matview-mutant-") as temp:
        worktree = Path(temp)
        _copy_tree(worktree, target_module)
        _require_isolated_import(worktree, mutant.name, target_module)

        contents = _require_single_site(worktree, mutant)
        source = _resolve_source(worktree, mutant)

        baseline = _run_test(worktree, mutant.target_test)
        _require_baseline_pass(baseline, f"{mutant.name} baseline")

        source.write_text(
            contents.replace(mutant.original, mutant.replacement, 1),
            encoding="utf-8",
        )
        mutated = _run_test(worktree, mutant.target_test)
        if not _is_assertion_kill(mutated):
            raise RuntimeError(
                f"{mutant.name}: mutant survived or failed for a non-assertion reason.\n"
                f"{mutated.stdout}"
            )
        print(f"KILLED {mutant.name} [{mutant.source_path}] by {mutant.target_test}")


def _select(names: Iterable[str]) -> List[Mutant]:
    wanted = list(names)
    if not wanted:
        return list(MUTANTS)
    by_name = {m.name: m for m in MUTANTS}
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        raise RuntimeError(f"unknown mutant(s): {', '.join(unknown)}")
    return [by_name[n] for n in wanted]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--list"]:
        for mutant in MUTANTS:
            print(f"{mutant.name}\t{mutant.source_path}\t{mutant.target_test}")
        return 0
    try:
        validate_catalog(MUTANTS)
        selected = _select(args)
    except RuntimeError as exc:
        print(f"MUTATION TESTING FAILED: {exc}", file=sys.stderr)
        return 1

    failures: List[str] = []
    for mutant in selected:
        try:
            run_mutation(mutant)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            print(f"NOT KILLED {mutant.name}: {exc}", file=sys.stderr)
            failures.append(mutant.name)
    killed = len(selected) - len(failures)
    if failures:
        print(
            f"MUTATION TESTING FAILED: {killed}/{len(selected)} curated mutants killed; "
            f"not killed: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    print(f"Mutation testing passed: {killed}/{len(selected)} curated mutants killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
