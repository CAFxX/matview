#!/usr/bin/env python3
"""Prove the replacement rollback regression detects a broken DDL commit boundary.

The mutant commits the old-view drop before replacement population. The semantic
regression must pass on the real implementation and fail because the old reader
view is missing on the mutant. Setup/import/runtime failures are not counted as
mutation kills.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = "matview_sync.py"
REGRESSION = "test_registration_replacement_rollback.py"
ORIGINAL = "self.drop_view(plan.view_name, _commit_transaction=False)"
MUTANT = "self.drop_view(plan.view_name)"
BASELINE_MARKER = "PASS: replacement rollback and fail-closed unsupported replacement checks"
EXPECTED_FAILURE = "no such table: replacement_reader"


def run_case(mutated: bool) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="matview-replacement-mutant-") as tmp:
        work = Path(tmp)
        source = (ROOT / SOURCE).read_text(encoding="utf-8")
        if mutated:
            occurrences = source.count(ORIGINAL)
            if occurrences != 1:
                raise RuntimeError(
                    f"expected exactly one transactional drop call to mutate, found {occurrences}"
                )
            source = source.replace(ORIGINAL, MUTANT, 1)
        (work / SOURCE).write_text(source, encoding="utf-8")
        shutil.copy2(ROOT / REGRESSION, work / REGRESSION)
        return subprocess.run(
            [sys.executable, REGRESSION],
            cwd=work,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )


def main() -> None:
    baseline = run_case(mutated=False)
    if baseline.returncode != 0 or BASELINE_MARKER not in baseline.stdout:
        raise RuntimeError(
            "clean replacement rollback regression did not pass; "
            "this is a baseline/setup failure, not a mutation kill:\n" + baseline.stdout
        )

    mutant = run_case(mutated=True)
    if mutant.returncode == 0:
        raise AssertionError("transaction-boundary mutant survived the replacement regression")
    if EXPECTED_FAILURE not in mutant.stdout:
        raise RuntimeError(
            "mutant failed for an unexpected reason; not counted as a kill:\n" + mutant.stdout
        )
    print("PASS: transactional-drop commit mutant was killed by the missing prior reader view")


if __name__ == "__main__":
    main()
