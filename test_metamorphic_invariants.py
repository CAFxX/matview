#!/usr/bin/env python3
"""Metamorphic invariants for independently ordered, commuting base-table DML."""

from __future__ import annotations

import random
import sqlite3
import unittest
from typing import Any

from matview_sync import MaterializedViewManager
from test_matview_sync import _register_sqlite_compat_udfs


class CommutingDMLInvariantTests(unittest.TestCase):
    """Check incremental state against SQL semantics under equivalent DML orders."""

    @staticmethod
    def _make_operations(seed: int) -> list[tuple[str, tuple[Any, ...]]]:
        rng = random.Random(seed)
        target_ids = rng.sample(range(1, 31), 18)
        operations: list[tuple[str, tuple[Any, ...]]] = []
        next_insert_id = 1000

        for target_id in target_ids:
            kind = rng.choices(
                ("update", "delete", "insert", "noop"),
                weights=(4, 2, 3, 1),
                k=1,
            )[0]
            if kind == "update":
                operations.append(
                    (
                        "UPDATE items SET bucket = ?, amount = ? WHERE id = ?;",
                        (f"b{rng.randrange(5)}", rng.randrange(-500, 501), target_id),
                    )
                )
            elif kind == "delete":
                operations.append(("DELETE FROM items WHERE id = ?;", (target_id,)))
            elif kind == "noop":
                operations.append(
                    ("UPDATE items SET amount = amount WHERE id = ?;", (target_id,))
                )
            else:
                operations.append(
                    (
                        "INSERT INTO items (id, bucket, amount) VALUES (?, ?, ?);",
                        (
                            next_insert_id,
                            f"b{rng.randrange(5)}",
                            rng.randrange(-500, 501),
                        ),
                    )
                )
                next_insert_id += 1

        return operations

    @staticmethod
    def _new_database() -> tuple[sqlite3.Connection, MaterializedViewManager]:
        conn = sqlite3.connect(":memory:")
        _register_sqlite_compat_udfs(conn)
        conn.execute(
            "CREATE TABLE items ("
            "id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amount INTEGER NOT NULL"
            ");"
        )
        conn.executemany(
            "INSERT INTO items (id, bucket, amount) VALUES (?, ?, ?);",
            [(i, f"b{i % 5}", (i * 37) % 211 - 100) for i in range(1, 31)],
        )
        manager = MaterializedViewManager(conn)
        manager.register_view(
            "CREATE VIEW commuting_totals AS "
            "SELECT bucket, COUNT(*) AS row_count, SUM(amount) AS total, "
            "MIN(amount) AS minimum, MAX(amount) AS maximum "
            "FROM items GROUP BY bucket;"
        )
        return conn, manager

    @staticmethod
    def _assert_live_equivalence(
        test: unittest.TestCase,
        conn: sqlite3.Connection,
        seed: int,
        step: int,
        sql: str,
    ) -> list[tuple[Any, ...]]:
        live = conn.execute(
            "SELECT bucket, COUNT(*) AS row_count, SUM(amount) AS total, "
            "MIN(amount) AS minimum, MAX(amount) AS maximum "
            "FROM items GROUP BY bucket ORDER BY bucket;"
        ).fetchall()
        materialized = conn.execute(
            "SELECT bucket, row_count, total, minimum, maximum "
            "FROM commuting_totals ORDER BY bucket;"
        ).fetchall()
        test.assertEqual(
            materialized,
            live,
            msg=(
                f"seed={seed}, step={step}, statement={sql!r}; "
                f"materialized={materialized!r}, live={live!r}"
            ),
        )
        return materialized

    def test_disjoint_row_mutations_commute_and_preserve_live_query_equivalence(self) -> None:
        # Every generated operation targets a distinct pre-existing row or inserts
        # a fresh key. Thus the operations commute by relational semantics; their
        # order must not affect the final aggregate, and every intermediate
        # materialized result must equal an independently executed live SELECT.
        for seed in range(8):
            operations = self._make_operations(seed)
            rng = random.Random(seed ^ 0x5A17)
            orders = [
                list(range(len(operations))),
                list(reversed(range(len(operations)))),
                rng.sample(range(len(operations)), len(operations)),
            ]
            final_results: list[list[tuple[Any, ...]]] = []

            for order in orders:
                conn, manager = self._new_database()
                try:
                    self._assert_live_equivalence(self, conn, seed, -1, "<initial>")
                    for step, operation_index in enumerate(order):
                        sql, params = operations[operation_index]
                        conn.execute(sql, params)
                        conn.commit()
                        self._assert_live_equivalence(self, conn, seed, step, sql)
                    final_results.append(
                        conn.execute(
                            "SELECT bucket, row_count, total, minimum, maximum "
                            "FROM commuting_totals ORDER BY bucket;"
                        ).fetchall()
                    )
                finally:
                    try:
                        manager.drop_view("commuting_totals")
                    finally:
                        conn.close()

            self.assertEqual(
                final_results[0],
                final_results[1],
                msg=f"reverse operation order changed the result for seed={seed}",
            )
            self.assertEqual(
                final_results[0],
                final_results[2],
                msg=f"permuted operation order changed the result for seed={seed}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
