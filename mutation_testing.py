#!/usr/bin/env python3
"""Run curated, test-specific source mutants to verify regression sensitivity.

Manual invocation:
    python mutation_testing.py              # every curated mutant
    python mutation_testing.py NAME [...]   # selected mutants by name
    python mutation_testing.py --list       # list mutants and their targets

CI also runs this diagnostic once in the SQLite / Python 3.11 matrix leg.

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
from typing import Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = "matview"
COPIED_FILES = ("matview_sync.py", "test_matview_sync.py")
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
        original="            upd_old_stmts=upd_old_list,\n            upd_new_stmts=[plus_stmt],\n            upd_cleanup_old_stmts=upd_clean_list,\n            trunc_stmts=[f\"DELETE FROM {mv_table};\"],\n            where_new=where_new,\n            where_old=where_old,",
        replacement="            upd_old_stmts=upd_old_list,\n            upd_new_stmts=[plus_stmt],\n            upd_cleanup_old_stmts=upd_clean_list,\n            trunc_stmts=[f\"DELETE FROM {mv_table};\"],\n            where_new=where_new,\n            where_old=where_new,",
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
)


def _resolve_source(root: Path, mutant: Mutant) -> Path:
    relative = Path(mutant.source_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{mutant.name}: source path must be repository-relative")
    if relative.parts[0] != PACKAGE_DIR and relative.name not in COPIED_FILES:
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


def _copy_tree(worktree: Path) -> None:
    shutil.copytree(
        ROOT / PACKAGE_DIR,
        worktree / PACKAGE_DIR,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in COPIED_FILES:
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


def _require_isolated_import(worktree: Path, label: str) -> None:
    probe = _run(
        worktree,
        [
            "-c",
            "import matview, matview_sync, test_matview_sync; "
            "print(matview.__file__); print(matview_sync.__file__)",
        ],
    )
    expected = worktree.resolve()
    paths = [Path(line).resolve() for line in probe.stdout.strip().splitlines()[-2:]]
    if probe.returncode != 0 or len(paths) != 2 or any(
        expected not in p.parents for p in paths
    ):
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
    with tempfile.TemporaryDirectory(prefix="matview-mutant-") as temp:
        worktree = Path(temp)
        _copy_tree(worktree)
        _require_isolated_import(worktree, mutant.name)

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
