#!/usr/bin/env python3
"""Run curated, test-specific source mutants to verify regression sensitivity.

Manual invocation:
    python mutation_testing.py

CI also runs this diagnostic once in the SQLite / Python 3.11 matrix leg.

Each mutant is applied to an isolated temporary source copy. The target
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

ROOT = Path(__file__).resolve().parent
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
        source_path="matview_sync.py",
        original="            upd_old_stmts=upd_old_list,\n            upd_new_stmts=[plus_stmt],\n            upd_cleanup_old_stmts=upd_clean_list,\n            trunc_stmts=[f\"DELETE FROM {mv_table};\"],\n            where_new=where_new,\n            where_old=where_old,",
        replacement="            upd_old_stmts=upd_old_list,\n            upd_new_stmts=[plus_stmt],\n            upd_cleanup_old_stmts=upd_clean_list,\n            trunc_stmts=[f\"DELETE FROM {mv_table};\"],\n            where_new=where_new,\n            where_old=where_new,",
        target_test="test_matview_sync.TestGeneratorAndManagerCodePaths.test_filtered_projection_retracts_old_membership_on_update_and_delete",
    ),
    Mutant(
        name="join-sum-retraction-adds-instead-of-subtracting",
        source_path="matview_sync.py",
        original='f"ELSE {mv_table}.{col} - __mv_delta.{col} END)"',
        replacement='f"ELSE {mv_table}.{col} + __mv_delta.{col} END)"',
    ),
    Mutant(
        name="join-sum-retraction-keeps-value-when-nonnull-count-reaches-zero",
        source_path="matview_sync.py",
        original='                f"(CASE WHEN __mv_delta.{col} IS NULL THEN {mv_table}.{col} "'
        + '\n                f"WHEN {new_c} <= 0 THEN NULL "',
        replacement='                f"(CASE WHEN __mv_delta.{col} IS NULL THEN {mv_table}.{col} "'
        + '\n                f"WHEN {new_c} < 0 THEN NULL "',
    ),
    Mutant(
        name="distinct-auxiliary-retraction-increments-reference-count",
        source_path="matview_sync.py",
        original="SET __ref_cnt = {aux_tbl}.__ref_cnt - 1",
        replacement="SET __ref_cnt = {aux_tbl}.__ref_cnt + 1",
        target_test=(
            "test_matview_sync.TestGeneratorAndManagerCodePaths."
            "test_24_metamorphic_mv_invariants_and_rebuild_consistency"
        ),
    ),
    Mutant(
        name="join-count-retraction-adds-instead-of-subtracting",
        source_path="matview_sync.py",
        original='minus_assignments.append(f"{c} = {mv_table}.{c} - __mv_delta.{c}")',
        replacement='minus_assignments.append(f"{c} = {mv_table}.{c} + __mv_delta.{c}")',
    ),
    Mutant(
        name="join-retraction-reads-new-row-instead-of-old-row",
        source_path="matview_sync.py",
        original='upd_old_stmts=[minus_stmt]',
        replacement='upd_old_stmts=[minus_stmt.replace("OLD.", "NEW.")]',
    ),
)


def _run_test(root: Path, target: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "unittest", target, "-v"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


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
    with tempfile.TemporaryDirectory(prefix="matview-mutant-") as temp:
        worktree = Path(temp)
        for filename in ("matview_sync.py", "test_matview_sync.py"):
            shutil.copy2(ROOT / filename, worktree / filename)

        source = worktree / mutant.source_path
        contents = source.read_text(encoding="utf-8")
        occurrences = contents.count(mutant.original)
        if occurrences != 1:
            raise RuntimeError(
                f"{mutant.name}: expected exactly one mutation site, found {occurrences}"
            )

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
        print(f"KILLED {mutant.name} by {mutant.target_test}")


def main() -> int:
    killed = 0
    try:
        for mutant in MUTANTS:
            run_mutation(mutant)
            killed += 1
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"MUTATION TESTING FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Mutation testing passed: {killed}/{len(MUTANTS)} curated mutants killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
