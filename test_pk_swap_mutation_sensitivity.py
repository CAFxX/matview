#!/usr/bin/env python3
"""Mutation-sensitivity regression for row-level primary-key swap protection.

The correct trigger must preserve equivalence with the original view. Removing
the OLD-key NOT EXISTS guard must make that same semantic comparison fail.
"""
from __future__ import annotations

import re
import sqlite3

from matview_sync import Dialect, generate_materialized_view


SCHEMA = "CREATE TABLE nodes (id INT PRIMARY KEY, label TEXT);"
VIEW = "CREATE VIEW nodes_mv AS SELECT id, label FROM nodes;"


def _run_swap(mutant: bool) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    conn = sqlite3.connect(":memory:")
    try:
        # Deliberately omit the physical PK here so SQLite can execute the
        # multi-row key permutation as one statement, matching the compiler's
        # declared schema and the regression in test_p0_pk_swap_and_collation.py.
        conn.execute("CREATE TABLE nodes (id INT, label TEXT);")
        conn.executemany(
            "INSERT INTO nodes (id, label) VALUES (?, ?);",
            [(1, "Alpha"), (2, "Beta")],
        )
        plan = generate_materialized_view(SCHEMA, VIEW, dialect=Dialect.SQLITE)
        for statement in plan.create_structure_statements():
            conn.execute(statement)
        for statement in plan.initial_populate_sqls:
            conn.execute(statement)

        guard_re = re.compile(
            r"\\s+AND NOT EXISTS\\s*\\(\\s*SELECT 1 FROM\\s+.*?\\s+WHERE\\s+.*?\\)",
            re.IGNORECASE | re.DOTALL,
        )
        changed = 0
        for statement in plan.trigger_ddls:
            if mutant:
                statement, count = guard_re.subn("", statement, count=1)
                changed += count
            conn.execute(statement)
        if mutant and changed != 1:
            raise AssertionError(
                f"expected to remove exactly one OLD-key guard, removed {changed}"
            )

        conn.execute("UPDATE nodes SET id = 3 - id;")
        base_rows = conn.execute(
            "SELECT id, label FROM nodes ORDER BY id, label;"
        ).fetchall()
        reader_rows = conn.execute(
            "SELECT id, label FROM nodes_mv ORDER BY id, label;"
        ).fetchall()
        return base_rows, reader_rows
    finally:
        conn.close()


def main() -> None:
    base, correct_reader = _run_swap(mutant=False)
    if correct_reader != base:
        raise AssertionError(
            "correct trigger failed the live-view equivalence baseline: "
            f"base={base!r}, reader={correct_reader!r}"
        )
    print("BASELINE PASSED: materialized reader equals the original view")

    mutant_base, mutant_reader = _run_swap(mutant=True)
    if mutant_reader == mutant_base:
        raise AssertionError(
            "PK-swap trigger mutant survived: removing the OLD-key guard "
            "did not produce an observable divergence"
        )
    print(
        "KILLED row_level_pk_swap_old_key_guard: "
        f"base={mutant_base!r}, materialized={mutant_reader!r}"
    )


if __name__ == "__main__":
    main()
