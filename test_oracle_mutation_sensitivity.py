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
    "test_21b_differential_oracle_compares_native_result_metadata",
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
    ("accept_every_result", _mutant_always_equal, "test_14_differential_oracle_preserves_exact_numeric_values"),
    ("coerce_exact_integers_to_binary64", _mutant_coerce_integers_to_float, "test_14_differential_oracle_preserves_exact_numeric_values"),
    ("discard_duplicate_rows", _mutant_drop_duplicate_rows, "test_16_differential_oracle_preserves_types_and_row_multiplicity"),
    ("casefold_exact_text", _mutant_casefold_text, "test_16_differential_oracle_preserves_types_and_row_multiplicity"),
)


# Mutants of the native result-metadata comparators (module attributes other
# than _rows_equal): (name, attribute, factory, designated killer).
def _mutant_no_description_check(original: Callable[..., Any]) -> Callable[..., Any]:
    def mutant(*args: Any, **kwargs: Any) -> list[str]:
        return []
    return mutant


def _mutant_ignore_type_code(original: Callable[..., Any]) -> Callable[..., Any]:
    def strip(desc: Any) -> Any:
        return None if desc is None else [(c[0], None) + tuple(c[2:]) for c in desc]

    def mutant(family: str, mv: Any, live: Any) -> list[str]:
        return original(family, strip(mv), strip(live))
    return mutant


def _mutant_decimal_numeric_only(original: Callable[..., Any]) -> Callable[..., Any]:
    def normalize(rows: Any) -> Any:
        from decimal import Decimal
        return [tuple(v.normalize() if isinstance(v, Decimal) else v for v in r) for r in rows]

    def mutant(mv_rows: Any, live_rows: Any) -> list[str]:
        return original(normalize(mv_rows), normalize(live_rows))
    return mutant


METADATA_MUTANTS = (
    ("skip_result_description", "_result_description_mismatches", _mutant_no_description_check,
     "test_21b_differential_oracle_compares_native_result_metadata"),
    ("ignore_driver_type_code", "_result_description_mismatches", _mutant_ignore_type_code,
     "test_21b_differential_oracle_compares_native_result_metadata"),
    ("skip_value_representation", "_result_representation_mismatches", _mutant_no_description_check,
     "test_21b_differential_oracle_compares_native_result_metadata"),
    ("decimal_scale_insensitive", "_result_representation_mismatches", _mutant_decimal_numeric_only,
     "test_21b_differential_oracle_compares_native_result_metadata"),
)


def _run_focused_oracle_tests() -> unittest.TestResult:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite(
        loader.loadTestsFromName(
            f"TestGeneratorAndManagerCodePaths.{test_name}", oracle
        )
        for test_name in ORACLE_TESTS
    )
    result = unittest.TestResult()
    suite.run(result)
    if result.testsRun != len(ORACLE_TESTS):
        raise AssertionError(
            f"expected {len(ORACLE_TESTS)} focused tests, ran {result.testsRun}"
        )
    return result


def run_mutant(
    name: str,
    factory: Callable[[Callable[..., bool]], Callable[..., bool]],
    expected_killer: str,
) -> int:
    return run_attribute_mutant(name, "_rows_equal", factory, expected_killer)


def run_attribute_mutant(
    name: str,
    attribute: str,
    factory: Callable[[Callable[..., Any]], Callable[..., Any]],
    expected_killer: str,
) -> int:
    original = getattr(oracle, attribute)
    baseline = _run_focused_oracle_tests()
    if not baseline.wasSuccessful():
        raise AssertionError(
            f"baseline failed before applying mutant {name!r}: "
            f"{len(baseline.failures)} failures, {len(baseline.errors)} errors"
        )

    setattr(oracle, attribute, factory(original))
    try:
        result = _run_focused_oracle_tests()
        # Count only assertion failures as kills. Import/setup/runtime errors
        # are invalid mutation runs, not evidence that a test detected a bug.
        failed_test_ids = {test.id() for test, _ in result.failures}
        expected_suffix = f"TestGeneratorAndManagerCodePaths.{expected_killer}"
        if result.errors or not any(
            test_id.endswith(expected_suffix) or f"{expected_suffix} (" in test_id
            for test_id in failed_test_ids
        ):
            raise AssertionError(
                f"mutant {name!r} was not killed by its designated regression "
                f"{expected_suffix!r}; failures={sorted(failed_test_ids)}, "
                f"errors={len(result.errors)}"
            )
        print(
            f"KILLED {name}: {len(result.failures)}/{result.testsRun} tests failed "
            f"with assertions (no test errors)"
        )
        return len(result.failures)
    finally:
        setattr(oracle, attribute, original)


def main() -> None:
    total = 0
    for name, factory, expected_killer in MUTANTS:
        total += run_mutant(name, factory, expected_killer)
    for name, attribute, factory, expected_killer in METADATA_MUTANTS:
        total += run_attribute_mutant(name, attribute, factory, expected_killer)
    count = len(MUTANTS) + len(METADATA_MUTANTS)
    print(f"Oracle mutation sensitivity passed: {count} mutants killed ({total} failing test executions).")


if __name__ == "__main__":
    main()
