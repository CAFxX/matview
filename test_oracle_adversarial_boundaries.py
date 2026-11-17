#!/usr/bin/env python3
"""Exhaustive Adversarial Boundary and Precision Verification for the Differential Oracle.

Covers all remaining P0 differential oracle contracts:
1. Severe floating-point cancellation (1e15 + delta - 1e15, catastrophic subtraction in SUM/AVG deltas).
2. High offsets with tiny increments, signed zero (-0.0 vs +0.0), NaN/Inf handling, overflow/underflow boundaries.
3. Subnormal / tiny magnitudes (1e-308, 1e-12, underflow to zero, ULP tracking across zero).
4. Large aggregates (10^9-10^15 SUM/AVG bounds, exact integer vs float vs Decimal precision boundaries across
   PostgreSQL NUMERIC, MySQL DECIMAL, SQLite REAL).
5. Values differing only beyond current rounding precision vs values differing strictly within tolerance
   (proving the oracle rejects meaningful mismatches while handling approximate-float tolerances intentionally
   without silent false positives).
"""

from __future__ import annotations

from decimal import Decimal, localcontext
import math
import sqlite3
import struct
import sys
import unittest
from typing import Any, List, Optional, Sequence, Tuple

from matview_sync import (
    Dialect,
    generate_materialized_view,
    install_materialized_view_sqlite,
)
from test_matview_sync import (
    NumericTolerance,
    _float_ulp_distance,
    _numeric_values_equal,
    _values_equal,
    _rows_equal,
    _query_column_tolerances,
    _AGGREGATE_NUMERIC_TOLERANCES,
)


class FloatingPointCancellationTests(unittest.TestCase):
    """Verifies floating-point cancellation behaviors and delta trigger resilience."""

    def test_01_catastrophic_cancellation_subtraction_at_1e15_boundary(self) -> None:
        # At 1e15, 1 ULP is exactly 0.125 in IEEE-754 binary64.
        self.assertEqual(math.ulp(1e15), 0.125)

        # Delta at or above 1 ULP is representable:
        for delta in (0.125, 0.25, 0.5, 1.0, 2.0):
            with self.subTest(delta=delta):
                cancelled = (1e15 + delta) - 1e15
                self.assertEqual(cancelled, delta)
                # Matches exact Decimal expectation under float tolerance:
                self.assertTrue(
                    _rows_equal(
                        [(cancelled,)],
                        [(Decimal(str(delta)),)],
                        tol=_AGGREGATE_NUMERIC_TOLERANCES["SUM"],
                    )
                )

        # Delta below 1 ULP (e.g. 0.0625 = 0.5 ULP) is rounded away completely in binary64:
        vanished = (1e15 + 0.0625) - 1e15
        self.assertEqual(vanished, 0.0)

        # The oracle MUST reject 0.0 against the mathematical Decimal expectation Decimal("0.0625"),
        # because 0.0625 exceeds the SUM tolerance floor (1e-12).
        self.assertFalse(
            _rows_equal(
                [(vanished,)],
                [(Decimal("0.0625"),)],
                tol=_AGGREGATE_NUMERIC_TOLERANCES["SUM"],
            )
        )

        # However, tiny differences below the documented 1e-12 noise floor must be accepted:
        near_floor_within = 0.9e-12
        near_floor_exceed = 1.1e-12
        sum_tol = _AGGREGATE_NUMERIC_TOLERANCES["SUM"]
        self.assertTrue(_rows_equal([(0.0,)], [(near_floor_within,)], tol=sum_tol))
        self.assertFalse(_rows_equal([(0.0,)], [(near_floor_exceed,)], tol=sum_tol))

    def test_02_opposite_sign_cancellation_in_aggregates(self) -> None:
        # Sum of massive opposite terms leaving a small non-zero remainder:
        val_pos = 1e15 + 0.5
        val_neg = -1e15
        accumulated = val_pos + val_neg  # exactly 0.5
        self.assertEqual(accumulated, 0.5)

        # When comparing against live aggregate result, exact tolerance verifies exact match:
        self.assertTrue(_rows_equal([(accumulated,)], [(0.5,)], tol=NumericTolerance()))

        # If catastrophic cancellation in an unsafe calculation corrupted 0.5 to 0.0,
        # the oracle detects and rejects the false equality:
        self.assertFalse(
            _rows_equal(
                [(0.0,)],
                [(0.5,)],
                tol=_AGGREGATE_NUMERIC_TOLERANCES["SUM"],
            )
        )

        # Catastrophic cancellation where a small increment is wiped out by 1e16 offset:
        # At 1e16, math.ulp(1e16) == 2.0. Adding 1.0 collapses:
        self.assertEqual(math.ulp(1e16), 2.0)
        collapsed = (1e16 + 1.0) - 1e16
        self.assertEqual(collapsed, 0.0)
        self.assertFalse(
            _rows_equal(
                [(collapsed,)],
                [(1.0,)],
                tol=_AGGREGATE_NUMERIC_TOLERANCES["SUM"],
            )
        )

    def test_03_sqlite_materialized_view_sum_cancellation_deltas(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE TABLE cancel_src (id INTEGER PRIMARY KEY, grp TEXT, amount REAL);"
            )
            view_sql = (
                "CREATE VIEW cancel_mv AS "
                "SELECT grp, COUNT(*) AS cnt, SUM(amount) AS total, AVG(amount) AS avg_amt "
                "FROM cancel_src GROUP BY grp;"
            )
            plan = install_materialized_view_sqlite(conn, view_sql)
            self.assertIsNotNone(plan)

            # Insert huge offsetting numbers plus a modest offset:
            conn.execute("INSERT INTO cancel_src VALUES (1, 'G1', 1e14);")
            conn.execute("INSERT INTO cancel_src VALUES (2, 'G1', -1e14);")
            conn.execute("INSERT INTO cancel_src VALUES (3, 'G1', 3.75);")

            live_sql = (
                "SELECT grp, COUNT(*) AS cnt, SUM(amount) AS total, AVG(amount) AS avg_amt "
                "FROM cancel_src GROUP BY grp;"
            )
            policies = _query_column_tolerances(live_sql)
            self.assertIsNotNone(policies)

            mv_rows = conn.execute("SELECT grp, cnt, total, avg_amt FROM cancel_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))

            # Incremental UPDATE causing delta trigger calculation:
            conn.execute("UPDATE cancel_src SET amount = 1e14 + 1.25 WHERE id = 1;")
            mv_rows = conn.execute("SELECT grp, cnt, total, avg_amt FROM cancel_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))
            self.assertAlmostEqual(mv_rows[0][2], 5.0, places=9)

            # Incremental DELETE and INSERT:
            conn.execute("DELETE FROM cancel_src WHERE id = 2;")
            conn.execute("INSERT INTO cancel_src VALUES (4, 'G1', -1e14);")
            mv_rows = conn.execute("SELECT grp, cnt, total, avg_amt FROM cancel_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))
        finally:
            conn.close()

    def test_04_sqlite_materialized_view_avg_cancellation_under_reweighting(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE TABLE avg_cancel_src (id INTEGER PRIMARY KEY, grp TEXT, val REAL);"
            )
            view_sql = (
                "CREATE VIEW avg_cancel_mv AS "
                "SELECT grp, COUNT(*) AS n, AVG(val) AS mean_val "
                "FROM avg_cancel_src GROUP BY grp;"
            )
            install_materialized_view_sqlite(conn, view_sql)

            # Insert alternating positive and negative large values:
            rows = [
                (1, "A", 1e12),
                (2, "A", -1e12 + 0.5),
                (3, "A", 1e12),
                (4, "A", -1e12 + 0.5),
                (5, "A", 2.0),
            ]
            conn.executemany("INSERT INTO avg_cancel_src VALUES (?, ?, ?);", rows)

            live_sql = (
                "SELECT grp, COUNT(*) AS n, AVG(val) AS mean_val "
                "FROM avg_cancel_src GROUP BY grp;"
            )
            policies = _query_column_tolerances(live_sql)
            mv_rows = conn.execute("SELECT grp, n, mean_val FROM avg_cancel_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))

            # Delete a large term:
            conn.execute("DELETE FROM avg_cancel_src WHERE id = 1;")
            mv_rows = conn.execute("SELECT grp, n, mean_val FROM avg_cancel_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))

            # Update another large term:
            conn.execute("UPDATE avg_cancel_src SET val = -1e12 + 1.0 WHERE id = 2;")
            mv_rows = conn.execute("SELECT grp, n, mean_val FROM avg_cancel_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))
        finally:
            conn.close()

    def test_05_cancellation_in_scalar_floating_arithmetic(self) -> None:
        # Scalar arithmetic whitelist derives a tight budget (max_ulps=4, relative=4.5e-16, absolute=1e-12):
        policies = _query_column_tolerances("SELECT amount - (amount - 0.125) AS delta FROM items;")
        self.assertIsNotNone(policies)
        assert policies is not None
        scalar_policy = policies[0]
        self.assertEqual(scalar_policy.max_ulps, 4)
        self.assertEqual(scalar_policy.absolute, 1e-12)

        # At scale 1e15 (where absolute=1e-12 and relative=4.5e-16*1e15=0.45 are binding),
        # 1 ULP is 0.125. Exactly 4 ULPs away is 0.5 (which exceeds relative 0.45, but is allowed by max_ulps=4):
        val = 1e15
        step_4 = val
        for _ in range(4):
            step_4 = math.nextafter(step_4, math.inf)
        self.assertEqual(step_4 - val, 0.5)
        self.assertTrue(_rows_equal([(val,)], [(step_4,)], tol=scalar_policy))

        # 5 ULPs away (0.625) exceeds both relative bound (0.45) and max_ulps=4, and is rejected:
        step_5 = math.nextafter(step_4, math.inf)
        self.assertEqual(step_5 - val, 0.625)
        self.assertFalse(_rows_equal([(val,)], [(step_5,)], tol=scalar_policy))

    def test_06_variance_catastrophic_cancellation_in_naive_vs_two_pass_or_welford(self) -> None:
        # Textbook catastrophic cancellation in sample variance:
        # For [1e9 + 1, 1e9 + 2, 1e9 + 3], population variance is exactly 2/3 (~0.6666666666666666).
        true_variance = 2.0 / 3.0
        # In single-pass naive formula (E[X^2] - (E[X])^2) using float64:
        # (1e9)^2 = 1e18, whose ULP is 128.0. The small variance (2/3) is completely lost in cancellation!
        naive_variance = (
            ((1e9 + 1.0) ** 2 + (1e9 + 2.0) ** 2 + (1e9 + 3.0) ** 2) / 3.0
            - ((1e9 + 2.0) ** 2)
        )
        self.assertEqual(naive_variance, 0.0)

        # The oracle's VAR_POP tolerance (relative=1e-13, absolute=1e-12, max_ulps=64) MUST reject
        # the corrupted naive variance against the true variance:
        var_tol = _AGGREGATE_NUMERIC_TOLERANCES["VAR_POP"]
        self.assertFalse(
            _rows_equal([(naive_variance,)], [(true_variance,)], tol=var_tol)
        )

        # A numerically stable computation (e.g. Welford or two-pass) matches true variance:
        shifts = [-1.0, 0.0, 1.0]
        stable_variance = sum(s**2 for s in shifts) / 3.0
        self.assertTrue(
            _rows_equal([(stable_variance,)], [(true_variance,)], tol=var_tol)
        )


class HighOffsetsAndSpecialValuesTests(unittest.TestCase):
    """Verifies high offsets, signed zero, NaNs, infinities, and overflow/underflow boundaries."""

    def test_01_high_offsets_with_tiny_increments(self) -> None:
        # Offset 1e12 with delta 0.5: relative error = 0.5 / 1e12 = 5e-13.
        base_1e12 = 1e12
        inc_1e12 = base_1e12 + 0.5
        self.assertTrue(
            _rows_equal([(base_1e12,)], [(inc_1e12,)], tol=NumericTolerance(relative=1e-12))
        )
        self.assertFalse(
            _rows_equal([(base_1e12,)], [(inc_1e12,)], tol=NumericTolerance(relative=1e-13))
        )

        # Offset 1e14 with delta 1.0: relative error = 1.0 / 1e14 = 1e-14.
        base_1e14 = 1e14
        inc_1e14 = base_1e14 + 1.0
        self.assertTrue(
            _rows_equal([(base_1e14,)], [(inc_1e14,)], tol=NumericTolerance(relative=1e-13))
        )
        self.assertFalse(
            _rows_equal([(base_1e14,)], [(inc_1e14,)], tol=NumericTolerance(relative=1e-15))
        )

        # Decimal exact comparison at high offsets must NEVER round away tiny differences:
        d_base = Decimal("1000000000000000.0000000000000001")
        d_diff = Decimal("1000000000000000.0000000000000002")
        self.assertFalse(_rows_equal([(d_base,)], [(d_diff,)], tol=NumericTolerance()))

    def test_02_signed_zero_representation_and_bipartite_matching(self) -> None:
        neg_zero = -0.0
        pos_zero = 0.0

        # Bit patterns differ, but value comparison equates them:
        self.assertNotEqual(struct.pack(">d", neg_zero), struct.pack(">d", pos_zero))
        self.assertTrue(_numeric_values_equal(neg_zero, pos_zero, tol=NumericTolerance()))
        self.assertTrue(_values_equal(neg_zero, pos_zero, tol=NumericTolerance()))
        self.assertTrue(_rows_equal([(neg_zero,)], [(pos_zero,)]))

        # Duplicate multisets with mixed signed zero:
        left = [(neg_zero,), (pos_zero,)]
        right = [(pos_zero,), (neg_zero,)]
        self.assertTrue(_rows_equal(left, right))
        self.assertTrue(_rows_equal([(neg_zero,), (neg_zero,)], [(pos_zero,), (pos_zero,)]))

        # Decimal signed zero:
        d_neg_zero = Decimal("-0.0")
        d_pos_zero = Decimal("0.0")
        self.assertTrue(_rows_equal([(d_neg_zero,)], [(d_pos_zero,)]))
        self.assertTrue(_rows_equal([(Decimal("-0"),)], [(Decimal("0"),)]))

        # Cross-type signed zero:
        self.assertTrue(_values_equal(d_neg_zero, pos_zero, tol=NumericTolerance()))
        self.assertTrue(_values_equal(d_pos_zero, neg_zero, tol=NumericTolerance()))
        self.assertTrue(_rows_equal([(d_neg_zero,)], [(pos_zero,)]))
        self.assertTrue(_rows_equal([(neg_zero,)], [(d_pos_zero,)]))

        # SQLite execution with signed zero:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE TABLE sz_table (id INT PRIMARY KEY, val REAL);")
            conn.execute("INSERT INTO sz_table VALUES (1, -0.0), (2, 0.0);")
            conn.commit()
            view_sql = "CREATE VIEW sz_mv AS SELECT id, val FROM sz_table;"
            install_materialized_view_sqlite(conn, view_sql)
            mv_rows = conn.execute("SELECT id, val FROM sz_mv ORDER BY id;").fetchall()
            live_rows = conn.execute("SELECT id, val FROM sz_table ORDER BY id;").fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows))
        finally:
            conn.close()

    def test_03_nan_and_infinity_deliberate_handling(self) -> None:
        nan_float = float("nan")
        nan_dec = Decimal("NaN")
        inf_pos = float("inf")
        inf_neg = float("-inf")
        inf_dec_pos = Decimal("Infinity")
        inf_dec_neg = Decimal("-Infinity")

        # NaN equality across representations:
        self.assertTrue(_values_equal(nan_float, nan_float, tol=NumericTolerance()))
        self.assertTrue(_values_equal(nan_dec, nan_dec, tol=NumericTolerance()))
        self.assertTrue(_values_equal(nan_float, nan_dec, tol=NumericTolerance()))
        self.assertTrue(_values_equal(nan_dec, nan_float, tol=NumericTolerance()))

        # NaN is never equal to finite numbers or NULL:
        self.assertFalse(_values_equal(nan_float, 0.0, tol=NumericTolerance(absolute=1e10)))
        self.assertFalse(_values_equal(nan_float, None, tol=NumericTolerance()))
        self.assertFalse(_values_equal(nan_dec, Decimal("0"), tol=NumericTolerance()))
        self.assertFalse(_values_equal(nan_float, inf_pos, tol=NumericTolerance()))

        # Infinities:
        self.assertTrue(_values_equal(inf_pos, inf_pos, tol=NumericTolerance()))
        self.assertTrue(_values_equal(inf_neg, inf_neg, tol=NumericTolerance()))
        self.assertFalse(_values_equal(inf_pos, inf_neg, tol=NumericTolerance()))
        self.assertTrue(_values_equal(inf_pos, inf_dec_pos, tol=NumericTolerance()))
        self.assertTrue(_values_equal(inf_neg, inf_dec_neg, tol=NumericTolerance()))
        self.assertFalse(_values_equal(inf_pos, inf_dec_neg, tol=NumericTolerance()))

        # Infinity is NEVER collapsed into finite maximum float, even with huge tolerance:
        self.assertFalse(
            _values_equal(
                inf_pos,
                sys.float_info.max,
                tol=NumericTolerance(absolute=sys.float_info.max, relative=1.0),
            )
        )
        self.assertFalse(_values_equal(inf_pos, None, tol=NumericTolerance()))

    def test_04_overflow_and_underflow_boundary_limits(self) -> None:
        max_f = sys.float_info.max  # ~ 1.7976931348623157e+308
        min_f = sys.float_info.min  # ~ 2.2250738585072014e-308 (smallest normal)
        subnormal_smallest = 5e-324  # math.ulp(0.0)

        # Overflow:
        self.assertEqual(math.nextafter(max_f, math.inf), math.inf)
        self.assertFalse(_rows_equal([(max_f,)], [(math.inf,)]))

        # 1 ULP below maximum finite float:
        one_ulp_below_max = math.nextafter(max_f, 0.0)
        self.assertTrue(
            _rows_equal([(max_f,)], [(one_ulp_below_max,)], tol=NumericTolerance(max_ulps=1))
        )
        self.assertFalse(
            _rows_equal([(max_f,)], [(one_ulp_below_max,)], tol=NumericTolerance(max_ulps=0))
        )

        # Normal to subnormal transition:
        subnormal_largest = math.nextafter(min_f, 0.0)
        self.assertTrue(
            _rows_equal([(min_f,)], [(subnormal_largest,)], tol=NumericTolerance(max_ulps=1))
        )
        self.assertFalse(
            _rows_equal([(min_f,)], [(subnormal_largest,)], tol=NumericTolerance(max_ulps=0))
        )

        # Underflow to zero:
        self.assertFalse(_rows_equal([(subnormal_smallest,)], [(0.0,)], tol=NumericTolerance()))
        self.assertTrue(
            _rows_equal([(subnormal_smallest,)], [(0.0,)], tol=NumericTolerance(max_ulps=1))
        )
        self.assertTrue(
            _rows_equal([(subnormal_smallest,)], [(0.0,)], tol=NumericTolerance(absolute=1e-300))
        )
        # Relative-only tolerance with zero absolute floor cannot match subnormal to zero:
        self.assertFalse(
            _rows_equal([(subnormal_smallest,)], [(0.0,)], tol=NumericTolerance(relative=1e-5))
        )

    def test_05_null_and_boolean_cross_dialect_oracle_contracts(self) -> None:
        # NULL semantics: NULL equals NULL, but never equals 0, empty string, or NaN:
        self.assertTrue(_rows_equal([(None,)], [(None,)]))
        self.assertFalse(_rows_equal([(None,)], [(0,)]))
        self.assertFalse(_rows_equal([(None,)], [(0.0,)]))
        self.assertFalse(_rows_equal([(None,)], [("",)]))
        self.assertFalse(_rows_equal([(None,)], [(float("nan"),)]))
        self.assertFalse(_rows_equal([(None,)], [(Decimal("NaN"),)]))

        # Multiset containing NULL and valid values:
        self.assertTrue(
            _rows_equal([(1, None, 1.25), (2, "x", None)], [(2, "x", None), (1, None, 1.25)])
        )

        # Boolean bridge: PostgreSQL returns bool True/False, SQLite returns 1/0:
        self.assertTrue(_rows_equal([(True,)], [(1,)]))
        self.assertTrue(_rows_equal([(False,)], [(0,)]))
        self.assertTrue(_rows_equal([(True,)], [(Decimal("1"),)]))
        self.assertTrue(_rows_equal([(False,)], [(Decimal("0"),)]))
        self.assertFalse(_rows_equal([(True,)], [(False,)]))
        self.assertFalse(_rows_equal([(True,)], [(2,)]))
        self.assertFalse(_rows_equal([(False,)], [(2,)]))

        # Boolean must NOT be equated with floating point 1.0 or 0.0:
        self.assertFalse(_rows_equal([(True,)], [(1.0,)]))
        self.assertFalse(_rows_equal([(False,)], [(0.0,)]))


class SubnormalAndTinyMagnitudeTests(unittest.TestCase):
    """Verifies subnormal floats, ULP distances across zero, and noise floors."""

    def test_01_subnormal_numbers_and_ulp_distances_across_zero(self) -> None:
        smallest = 5e-324
        step2 = 1.0e-323
        step3 = 1.5e-323

        # ULP distances around 0:
        self.assertEqual(_float_ulp_distance(0.0, smallest), 1)
        self.assertEqual(_float_ulp_distance(-0.0, smallest), 2)  # via +0.0
        self.assertEqual(_float_ulp_distance(-smallest, 0.0), 2)
        self.assertEqual(_float_ulp_distance(-smallest, smallest), 3)

        # ULP distances between positive subnormals:
        self.assertEqual(_float_ulp_distance(smallest, step2), 1)
        self.assertEqual(_float_ulp_distance(smallest, step3), 2)

        # Exact ULP tolerance matching:
        self.assertTrue(
            _rows_equal([(smallest,)], [(step3,)], tol=NumericTolerance(max_ulps=2))
        )
        self.assertFalse(
            _rows_equal([(smallest,)], [(step3,)], tol=NumericTolerance(max_ulps=1))
        )

    def test_02_tiny_magnitude_noise_floor_at_1e12(self) -> None:
        # Arithmetic aggregates require an absolute noise floor (1e-12) to absorb cancelation.
        # Within the noise floor:
        val1 = 5e-13
        val2 = -4e-13  # diff = 9e-13 <= 1e-12
        self.assertTrue(
            _rows_equal([(val1,)], [(val2,)], tol=_AGGREGATE_NUMERIC_TOLERANCES["SUM"])
        )

        # Exceeding the noise floor:
        val3 = 7e-13
        val4 = -6e-13  # diff = 1.3e-12 > 1e-12
        self.assertFalse(
            _rows_equal([(val3,)], [(val4,)], tol=_AGGREGATE_NUMERIC_TOLERANCES["SUM"])
        )

        # Relative tolerance alone breaks down on tiny values:
        t1 = 1e-15
        t2 = 2e-15  # 50% relative difference!
        self.assertFalse(
            _rows_equal([(t1,)], [(t2,)], tol=NumericTolerance(relative=1e-9))
        )
        # But succeeds when combined with absolute noise floor:
        self.assertTrue(
            _rows_equal([(t1,)], [(t2,)], tol=NumericTolerance(absolute=1e-12, relative=1e-9))
        )

    def test_03_sqlite_materialized_view_with_subnormal_and_tiny_values(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE TABLE tiny_src (id INTEGER PRIMARY KEY, tag TEXT, val REAL);"
            )
            view_sql = (
                "CREATE VIEW tiny_mv AS "
                "SELECT tag, COUNT(*) AS cnt, SUM(val) AS sum_val, AVG(val) AS avg_val "
                "FROM tiny_src GROUP BY tag;"
            )
            install_materialized_view_sqlite(conn, view_sql)

            conn.execute("INSERT INTO tiny_src VALUES (1, 'T', 1e-300);")
            conn.execute("INSERT INTO tiny_src VALUES (2, 'T', 2e-300);")
            conn.execute("INSERT INTO tiny_src VALUES (3, 'T', 3e-300);")

            live_sql = (
                "SELECT tag, COUNT(*) AS cnt, SUM(val) AS sum_val, AVG(val) AS avg_val "
                "FROM tiny_src GROUP BY tag;"
            )
            policies = _query_column_tolerances(live_sql)
            mv_rows = conn.execute("SELECT tag, cnt, sum_val, avg_val FROM tiny_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))

            # Incremental DELETE:
            conn.execute("DELETE FROM tiny_src WHERE id = 2;")
            mv_rows = conn.execute("SELECT tag, cnt, sum_val, avg_val FROM tiny_mv;").fetchall()
            live_rows = conn.execute(live_sql).fetchall()
            self.assertTrue(_rows_equal(mv_rows, live_rows, column_tolerances=policies))
        finally:
            conn.close()

    def test_04_gradual_underflow_decay_and_negative_zero_underflow(self) -> None:
        # Gradual underflow starting at smallest normal float (sys.float_info.min):
        current = sys.float_info.min
        steps = []
        for _ in range(52):  # exactly 52 subnormal steps down to 5e-324
            current = current / 2.0
            steps.append(current)

        # The last non-zero subnormal is 5e-324:
        self.assertEqual(steps[-1], 5e-324)
        # Next halving underflows to +0.0:
        underflow_zero = steps[-1] / 2.0
        self.assertEqual(underflow_zero, 0.0)

        # Negative subnormal underflow to -0.0:
        neg_subnormal = -5e-324
        neg_zero = neg_subnormal / 2.0
        self.assertEqual(neg_zero, -0.0)
        self.assertEqual(math.copysign(1.0, neg_zero), -1.0)

        # Oracle equates underflowed signed zeros:
        self.assertTrue(_rows_equal([(underflow_zero,)], [(neg_zero,)]))
        # But distinguishes non-zero subnormal underflow from zero under exact policy:
        self.assertFalse(_rows_equal([(steps[-1],)], [(underflow_zero,)], tol=NumericTolerance()))


class LargeAggregatesAndCrossDialectPrecisionTests(unittest.TestCase):
    """Verifies large aggregates (10^9 to 10^15), exact int vs float vs Decimal, and DDL types."""

    def test_01_exact_integer_aggregates_at_and_above_2_pow_53(self) -> None:
        two_53 = 2**53  # 9007199254740992
        two_53_plus_1 = two_53 + 1
        two_53_plus_2 = two_53 + 2

        # In float, two_53_plus_1 is coerced to two_53:
        self.assertEqual(float(two_53_plus_1), float(two_53))

        # The oracle MUST preserve integer distinctness:
        self.assertFalse(_rows_equal([(two_53_plus_1,)], [(two_53,)]))
        self.assertFalse(_rows_equal([(two_53_plus_2,)], [(two_53_plus_1,)]))
        self.assertTrue(_rows_equal([(two_53_plus_1,)], [(two_53_plus_1,)]))

        # Integer vs float comparison MUST NEVER round exact integers:
        self.assertFalse(_rows_equal([(two_53_plus_1,)], [(float(two_53_plus_1),)]))
        # Even with generous floating tolerance, exact integer semantics remain strict:
        self.assertFalse(
            _rows_equal(
                [(two_53_plus_1,)],
                [(float(two_53_plus_1),)],
                tol=NumericTolerance(relative=1.0),
            )
        )

        # 64-bit integer signed and unsigned extremes:
        min_i64 = -(2**63)
        max_i64 = 2**63 - 1
        max_u64 = 2**64 - 1
        self.assertTrue(
            _rows_equal(
                [(min_i64,), (max_i64,), (max_u64,)],
                [(max_u64,), (min_i64,), (max_i64,)],
            )
        )
        self.assertFalse(_rows_equal([(max_i64,)], [(max_i64 - 1,)]))

    def test_02_large_aggregates_bounds_1e9_to_1e15(self) -> None:
        # Sum of 1,000 items of 10^12 gives 10^15:
        sum_1e15 = 1000 * (10**12)
        d_1e15 = Decimal("1000000000000000")
        f_1e15 = 1e15

        # Exact int vs Decimal:
        self.assertTrue(_rows_equal([(sum_1e15,)], [(d_1e15,)]))
        self.assertFalse(_rows_equal([(sum_1e15 + 1,)], [(d_1e15,)]))

        # Float vs exact Decimal under SUM aggregate budget:
        sum_tol = _AGGREGATE_NUMERIC_TOLERANCES["SUM"]
        self.assertTrue(_rows_equal([(f_1e15,)], [(d_1e15,)], tol=sum_tol))

        # Modifying float by 8 ULPs (8 * 0.125 = 1.0) is allowed under max_ulps=8:
        f_1e15_within_ulps = f_1e15 + 1.0
        self.assertTrue(_rows_equal([(f_1e15_within_ulps,)], [(d_1e15,)], tol=sum_tol))

        # Modifying float by 9 ULPs (9 * 0.125 = 1.125) exceeds max_ulps=8 and relative 4.5e-16 (0.45):
        f_1e15_exceed = f_1e15 + 1.125
        self.assertFalse(_rows_equal([(f_1e15_exceed,)], [(d_1e15,)], tol=sum_tol))

    def test_03_high_precision_decimal_to_float_cross_type_evaluations(self) -> None:
        # When comparing Decimal (from PostgreSQL NUMERIC / MySQL DECIMAL) to float (from SQLite REAL):
        exact_dec = Decimal("1000000000000.0000000000")
        f_val = 1e12

        # Matches within tight relative tolerance:
        self.assertTrue(
            _rows_equal([(f_val,)], [(exact_dec,)], tol=NumericTolerance(relative=1e-9))
        )

        # Exceeds relative tolerance:
        exact_dec_off = Decimal("1000000002000.0000000000")  # relative diff = 2e-9 > 1e-9
        self.assertFalse(
            _rows_equal([(f_val,)], [(exact_dec_off,)], tol=NumericTolerance(relative=1e-9))
        )

        # 65-digit exact Decimals differing only in trailing digits:
        d1 = Decimal("9" * 60 + ".12345")
        d2 = Decimal("9" * 60 + ".12346")
        self.assertFalse(_rows_equal([(d1,)], [(d2,)], tol=NumericTolerance()))

    def test_04_compiler_preserves_native_types_for_large_aggregates(self) -> None:
        schema = """
        CREATE TABLE large_src (
            id INTEGER PRIMARY KEY,
            grp TEXT,
            int_val INTEGER,
            big_val BIGINT,
            dec_val DECIMAL(38, 10),
            real_val DOUBLE PRECISION
        );
        """
        view = """
        CREATE VIEW large_mv AS
        SELECT grp,
               COUNT(*) AS cnt,
               SUM(int_val) AS sum_int,
               SUM(big_val) AS sum_big,
               AVG(big_val) AS avg_big,
               SUM(dec_val) AS sum_dec,
               AVG(dec_val) AS avg_dec,
               SUM(real_val) AS sum_real,
               AVG(real_val) AS avg_real
        FROM large_src GROUP BY grp;
        """

        # PostgreSQL DDL preserves native exact types (SUM(int) -> BIGINT, SUM(bigint) -> NUMERIC):
        pg_plan = generate_materialized_view(schema, view, dialect="postgres")
        self.assertIn("sum_int BIGINT", pg_plan.backing_table_ddl)
        self.assertIn("sum_big NUMERIC", pg_plan.backing_table_ddl)
        self.assertIn("avg_big NUMERIC", pg_plan.backing_table_ddl)
        self.assertIn("sum_dec NUMERIC", pg_plan.backing_table_ddl)
        self.assertIn("avg_dec NUMERIC", pg_plan.backing_table_ddl)
        self.assertIn("sum_real DOUBLE PRECISION", pg_plan.backing_table_ddl)
        self.assertIn("avg_real DOUBLE PRECISION", pg_plan.backing_table_ddl)

        # MySQL DDL preserves native exact types:
        mysql_plan = generate_materialized_view(
            schema, view, dialect="mysql", ignore_limitations=True
        )
        self.assertIn("sum_int DECIMAL(65, 0)", mysql_plan.backing_table_ddl)
        self.assertIn("sum_big DECIMAL(65, 0)", mysql_plan.backing_table_ddl)
        self.assertIn("sum_dec DECIMAL(65, 10)", mysql_plan.backing_table_ddl)
        self.assertIn("avg_big DECIMAL(23, 4)", mysql_plan.backing_table_ddl)
        self.assertIn("avg_dec DECIMAL(42, 14)", mysql_plan.backing_table_ddl)
        self.assertIn("SUM_REAL DOUBLE", mysql_plan.backing_table_ddl.upper())
        self.assertIn("AVG_REAL DOUBLE", mysql_plan.backing_table_ddl.upper())

        # SQLite DDL uses affinities. SUM over non-REAL inputs is stored
        # without affinity (BLOB): its result is INTEGER or REAL depending on
        # the stored values, and an affinity would convert one into the other.
        sqlite_plan = generate_materialized_view(schema, view, dialect="sqlite")
        self.assertIn("sum_int BLOB", sqlite_plan.backing_table_ddl)
        self.assertIn("sum_big BLOB", sqlite_plan.backing_table_ddl)
        self.assertIn("avg_big REAL", sqlite_plan.backing_table_ddl)
        self.assertIn("sum_dec BLOB", sqlite_plan.backing_table_ddl)
        self.assertIn("avg_dec REAL", sqlite_plan.backing_table_ddl)
        self.assertIn("sum_real REAL", sqlite_plan.backing_table_ddl)
        self.assertIn("avg_real REAL", sqlite_plan.backing_table_ddl)

        # Inferred column tolerances assign exact policy to grouping and count, approximate to sums/avgs:
        policies = _query_column_tolerances(view)
        self.assertIsNotNone(policies)
        assert policies is not None
        self.assertEqual(len(policies), 9)
        self.assertEqual(policies[0], NumericTolerance())  # grp exact
        self.assertEqual(policies[1], NumericTolerance())  # cnt exact
        self.assertEqual(policies[2], _AGGREGATE_NUMERIC_TOLERANCES["SUM"])
        self.assertEqual(policies[3], _AGGREGATE_NUMERIC_TOLERANCES["SUM"])
        self.assertEqual(policies[4], _AGGREGATE_NUMERIC_TOLERANCES["AVG"])
        self.assertEqual(policies[5], _AGGREGATE_NUMERIC_TOLERANCES["SUM"])
        self.assertEqual(policies[6], _AGGREGATE_NUMERIC_TOLERANCES["AVG"])
        self.assertEqual(policies[7], _AGGREGATE_NUMERIC_TOLERANCES["SUM"])
        self.assertEqual(policies[8], _AGGREGATE_NUMERIC_TOLERANCES["AVG"])

    def test_05_negative_large_aggregates_and_64bit_overflow_boundaries(self) -> None:
        # Large negative sums: -10^15
        neg_1e15 = -1000 * (10**12)
        d_neg_1e15 = Decimal("-1000000000000000")
        f_neg_1e15 = -1e15

        # Exact equality for ints and Decimals:
        self.assertTrue(_rows_equal([(neg_1e15,)], [(d_neg_1e15,)]))
        self.assertFalse(_rows_equal([(neg_1e15 - 1,)], [(d_neg_1e15,)]))

        # Float negative match under SUM tolerance:
        sum_tol = _AGGREGATE_NUMERIC_TOLERANCES["SUM"]
        self.assertTrue(_rows_equal([(f_neg_1e15,)], [(d_neg_1e15,)], tol=sum_tol))

        # Rejection when negative float diverges beyond 8 ULPs:
        # At -1e15, 9 ULPs is 1.125
        f_neg_diverged = f_neg_1e15 - 1.125
        self.assertFalse(_rows_equal([(f_neg_diverged,)], [(d_neg_1e15,)], tol=sum_tol))

        # 64-bit integer overflow boundary:
        # In SQLite, integer SUM beyond 2^63 - 1 overflows to REAL (float).
        # In PostgreSQL/MySQL, SUM(bigint) produces NUMERIC / DECIMAL(65, 0).
        max_int64 = 2**63 - 1  # 9223372036854775807
        overflowed_sum_dec = Decimal(max_int64) + Decimal(100)  # exact Decimal above 2^63 - 1
        float_approx = float(overflowed_sum_dec)

        # Oracle cross-type comparison evaluates Decimal vs float:
        # With SUM tolerance (which allows bounded float approximation):
        self.assertTrue(
            _rows_equal([(float_approx,)], [(overflowed_sum_dec,)], tol=sum_tol)
        )
        # But under exact policy (which applies when comparing exact columns), it fails:
        self.assertFalse(
            _rows_equal([(float_approx,)], [(overflowed_sum_dec,)], tol=NumericTolerance())
        )


class ToleranceBoundarySensitivityAndRejectionTests(unittest.TestCase):
    """Verifies strict boundary sensitivity and intentional rejection without silent false positives."""

    def test_01_absolute_tolerance_strict_boundary(self) -> None:
        tol = NumericTolerance(absolute=1e-7)
        val_base = 5.0
        val_within = val_base + 0.999999e-7
        val_exceed = val_base + 1.000001e-7

        self.assertTrue(_rows_equal([(val_base,)], [(val_within,)], tol=tol))
        self.assertFalse(_rows_equal([(val_base,)], [(val_exceed,)], tol=tol))

    def test_02_relative_tolerance_strict_boundary(self) -> None:
        tol = NumericTolerance(relative=1e-8)
        val_base = 2e8  # bound = 1e-8 * 2e8 = 2.0
        val_within = val_base + 1.99999
        val_exceed = val_base + 2.00001

        self.assertTrue(_rows_equal([(val_base,)], [(val_within,)], tol=tol))
        self.assertFalse(_rows_equal([(val_base,)], [(val_exceed,)], tol=tol))

    def test_03_max_ulps_strict_boundary(self) -> None:
        tol = NumericTolerance(max_ulps=5)
        val = 2.0
        step = val
        for _ in range(5):
            step = math.nextafter(step, math.inf)
        self.assertTrue(_rows_equal([(val,)], [(step,)], tol=tol))

        step_6 = math.nextafter(step, math.inf)
        self.assertFalse(_rows_equal([(val,)], [(step_6,)], tol=tol))

    def test_04_multiset_duplicate_matching_and_bipartite_resolution(self) -> None:
        # Row count mismatch:
        self.assertFalse(_rows_equal([(1.0,), (1.0,)], [(1.0,)]))
        self.assertFalse(_rows_equal([(1.0,)], [(1.0,), (1.0,)]))

        # Collation and string mismatch:
        self.assertFalse(_rows_equal([("CASE",)], [("case",)]))
        self.assertFalse(_rows_equal([("1",)], [(1,)]))

        # Hopcroft-Karp bipartite matching must resolve ambiguous candidate trade-offs:
        # Left: Row A = (10.0,), Row B = (10.5,)
        # Right: Row C = (10.3,), Row D = (9.8,)
        # With tol absolute = 0.6:
        # Row A matches C (diff 0.3) and D (diff 0.2).
        # Row B matches C (diff 0.2) but NOT D (diff 0.7).
        # A greedy matcher assigning A -> C would fail B -> D.
        # Hopcroft-Karp pairs A -> D and B -> C successfully.
        left = [(10.0,), (10.5,)]
        right = [(10.3,), (9.8,)]
        self.assertTrue(_rows_equal(left, right, tol=NumericTolerance(absolute=0.6)))

        # When two left rows compete for a single right candidate within tolerance:
        right_incompatible = [(10.3,), (9.7,)]
        # Row A matches C (0.3) and D (0.3).
        # With tol = 0.25, neither matches D (diffs 0.3 and 0.8), both require C.
        self.assertFalse(
            _rows_equal(left, right_incompatible, tol=NumericTolerance(absolute=0.25))
        )

    def test_05_column_tolerances_reject_grouping_or_exact_mismatches(self) -> None:
        select_sql = (
            "SELECT dept, COUNT(*) AS c, SUM(salary) AS s "
            "FROM emps GROUP BY dept;"
        )
        policies = _query_column_tolerances(select_sql)
        self.assertIsNotNone(policies)
        assert policies is not None

        row_ref = [("Engineering", 10, 500000.0)]

        # Permitted float variation in SUM column:
        row_float_delta = [("Engineering", 10, 500000.0000000001)]
        self.assertTrue(_rows_equal(row_ref, row_float_delta, column_tolerances=policies))

        # Excessive float variation in SUM column rejected:
        row_float_bad = [("Engineering", 10, 500000.1)]
        self.assertFalse(_rows_equal(row_ref, row_float_bad, column_tolerances=policies))

        # Grouping key mismatch rejected:
        row_group_bad = [("engineering", 10, 500000.0)]
        self.assertFalse(_rows_equal(row_ref, row_group_bad, column_tolerances=policies))

        # COUNT mismatch by 1 rejected:
        row_cnt_bad = [("Engineering", 11, 500000.0)]
        self.assertFalse(_rows_equal(row_ref, row_cnt_bad, column_tolerances=policies))

        # COUNT mismatch by tiny float rejected (COUNT is exact):
        row_cnt_float = [("Engineering", 10.00001, 500000.0)]
        self.assertFalse(_rows_equal(row_ref, row_cnt_float, column_tolerances=policies))

    def test_06_metamorphic_properties_symmetry_identity_and_dimension_checks(self) -> None:
        # Reflexive identity:
        rows = [("a", 1, 10.5, Decimal("100.5")), ("b", 2, 20.25, Decimal("200.75"))]
        self.assertTrue(_rows_equal(rows, rows))

        # Empty relations:
        self.assertTrue(_rows_equal([], []))
        self.assertTrue(_rows_equal([()], [()]))
        self.assertFalse(_rows_equal([], [()]))
        self.assertFalse(_rows_equal([()], []))

        # Symmetric equivalence:
        rows_permuted = [("b", 2, 20.25, Decimal("200.75")), ("a", 1, 10.5, Decimal("100.5"))]
        self.assertTrue(_rows_equal(rows, rows_permuted))
        self.assertTrue(_rows_equal(rows_permuted, rows))

        # Approximate symmetric equivalence under tolerance:
        tol = NumericTolerance(absolute=0.1)
        rows_approx = [("a", 1, 10.55, Decimal("100.5")), ("b", 2, 20.21, Decimal("200.75"))]
        self.assertEqual(
            _rows_equal(rows, rows_approx, tol=tol),
            _rows_equal(rows_approx, rows, tol=tol),
        )

        # Dimension and shape mismatches:
        self.assertFalse(_rows_equal([(1, 2)], [(1,)]))
        self.assertFalse(_rows_equal([(1, 2)], [(1, 2, 3)]))
        self.assertFalse(_rows_equal([(1,)], [(1, 2)]))

        # Column tolerances length mismatch with row width:
        with self.assertRaises(TypeError):
            _rows_equal([(1, 2)], [(1, 2)], column_tolerances=(1e-5, 1e-5))  # type: ignore[arg-type]
        self.assertFalse(
            _rows_equal(
                [(1, 2)],
                [(1, 2)],
                column_tolerances=(NumericTolerance(),),  # length 1 vs row width 2
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
