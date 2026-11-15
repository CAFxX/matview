#!/usr/bin/env python3
"""Prove that the differential-oracle regression tests kill representative mutants.

This is deliberately a small, deterministic first slice of the P0 mutation-testing
plan: it mutates only the row comparator used by the exact/type-aware oracle tests.
A mutant is considered killed only when the existing focused tests fail against it.
"""
from __future__ import annotations

import unittest
from typing import Any, Callable

import test_matview_sync as oracle


ORACLE_TESTS = (
    "test_14_differential_oracle_preserves_exact_numeric_values",
    "test_15_differential_oracle_handles_float_and_special_values",
    "test_16_differential_oracle_preserves_types_and_row_multiplicity",
    "test_17_differential_oracle_supports_explicit_numeric_error_budgets",
    "test_18_differential_oracle_uses_independent_column_error_budgets",
    "test_19_differential_oracle_numeric_adversarial_boundaries",
    "test_20_differential_oracle_derives_tolerances_from_select_operations",
    "test_21_differential_oracle_decimal_float_comparison_uses_full_precision",
)


def _map_rows(rows: Any, transform: Callable[[Any], Any]) -> Any:
    return [tuple(transform(value) for value in row) for row in rows]


def _mutant_always_equal(original: Callable[..., bool]) -> Callable[..., bool]:
    def mutant(rows_a: Any, rows_b: Any, *args: Any, **kwargs: Any) -> bool:
        return True
    return mutant


def _mutant_coerce_integers_to_float(original: Callable[..., bool]) -> Callable[..., bool]:
    def coerce(value: Any) -> Any:
        return float(value) if type(value) is int else value

    def mutant(rows_a: Any, rows_b: Any, *args: Any, **kwargs: Any) -> bool:
        return original(_map_rows(rows_a, coerce), _map_rows(rows_b, coerce), *args, **kwargs)
    return mutant


def _mutant_drop_duplicate_rows(original: Callable[..., bool]) -> Callable[..., bool]:
    def dedupe(rows: Any) -> list[Any]:
        unique: list[Any] = []
        for row in rows:
            if row not in unique:
                unique.append(row)
        return unique

    def mutant(rows_a: Any, rows_b: Any, *args: Any, **kwargs: Any) -> bool:
        return original(dedupe(rows_a), dedupe(rows_b), *args, **kwargs)
    return mutant


def _mutant_casefold_text(original: Callable[..., bool]) -> Callable[..., bool]:
    def normalize(value: Any) -> Any:
        return value.casefold() if isinstance(value, str) else value

    def mutant(rows_a: Any, rows_b: Any, *args: Any, **kwargs: Any) -> bool:
        return original(_map_rows(rows_a, normalize), _map_rows(rows_b, normalize), *args, **kwargs)
    return mutant


MUTANTS = (
    ("accept_every_result", _mutant_always_equal),
    ("coerce_exact_integers_to_binary64", _mutant_coerce_integers_to_float),
    ("discard_duplicate_rows", _mutant_drop_duplicate_rows),
    ("casefold_exact_text", _mutant_casefold_text),
)


def run_mutant(name: str, factory: Callable[[Callable[..., bool]], Callable[..., bool]]) -> int:
    original = oracle._rows_equal
    oracle._rows_equal = factory(original)
    try:
        loader = unittest.defaultTestLoader
        suite = unittest.TestSuite(
            loader.loadTestsFromName(
                f"TestGeneratorAndManagerCodePaths.{test_name}", oracle
            )
            for test_name in ORACLE_TESTS
        )
        result = unittest.TestResult()
        suite.run(result)
        killed = len(result.failures) + len(result.errors)
        if result.testsRun == 0 or killed == 0:
            raise AssertionError(
                f"mutant {name!r} survived {result.testsRun} focused oracle tests"
            )
        print(
            f"KILLED {name}: {killed}/{result.testsRun} tests failed "
            f"({len(result.failures)} failures, {len(result.errors)} errors)"
        )
        return killed
    finally:
        oracle._rows_equal = original


def main() -> None:
    total = 0
    for name, factory in MUTANTS:
        total += run_mutant(name, factory)
    print(f"Oracle mutation sensitivity passed: {len(MUTANTS)} mutants killed ({total} failing test executions).")


if __name__ == "__main__":
    main()
