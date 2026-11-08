#!/usr/bin/env python3
"""
test_matview_sync.py — Exhaustive Verification Suite for matview_sync.py
========================================================================

Executes 10 comprehensive test suites (6,200 materialized views, 52,633 live query
assertions) plus live database integration suites when `POSTGRES_DSN`, `MYSQL_DSN`,
or `SQLITE_COMPAT_ENGINE` is set:
  - Suite 1: Core Single-Table Algebraic, Distinct, Holistic & Window Matrix (192 views)
  - Suite 2 & 3: Joins, Set Operations, CTEs, Subqueries, Manager Lifecycle & Edge Cases (134 views)
  - Suite 4: Cross-Dialect PostgreSQL (Row & Statement Transition Tables) & MySQL Trigger Execution (132 views)
  - Suite 5: Multi-View SQL Bundles & Declarative Index Attachment (160 views)
  - Suite 6: Exhaustive ALGEBRAIC_DELTA Feature Permutations (1,552 views)
  - Suite 7: Adversarial & Multi-Dialect Algebraic Stress Matrix (1,560 views)
  - Suite 8: Massive Algebraic Permutations across Aggregates, Filters & Groupings (1,392 views)
  - Suite 9: Exhaustive Code-Path & Feature Combinations (808 views)
  - Suite 10: Complex DML & Transactional Intermediate States (SAVEPOINT / ROLLBACK / COMMIT) (250 views)
  - Unit & CLI tests (20 views)
"""

from __future__ import annotations

from decimal import Decimal
import json
import math
import os
import re
import sqlite3
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings

from matview_sync import (
    DEFAULT_OBJECT_PREFIX,
    AggFunc,
    Dialect,
    EngineLimitationError,
    EngineSubdialect,
    EngineVersion,
    detect_engine_version,
    parse_engine_version,
    EngineLimitationWarning,
    IncrementalViewCompiler,
    LimitationSeverity,
    MaintenanceStrategy,
    MaterializedViewManager,
    MaterializedViewPlan,
    StrictCompilationError,
    TableSchema,
    ViewIndexSpec,
    ViewSpec,
    _split_top_level,
    _strip_sql_comments,
    generate_materialized_view,
    generate_materialized_views,
    install_materialized_view_sqlite,
    introspect_database_schemas,
    introspect_sqlite_schemas,
    main as cli_main,
    parse_table_schema,
    parse_view_definition,
    parse_view_definitions,
    parse_view_index_spec,
)


# -----------------------------------------------------------------------------
# Core Verification Helpers & Cross-Dialect SQLite Execution Transpilers
# -----------------------------------------------------------------------------


def _numeric_sort_key(value: Any) -> Tuple[Any, ...]:
    """Return a deterministic, precision-preserving sort key for numeric values."""
    if isinstance(value, float):
        if math.isnan(value):
            return (2, 0, 0)
        if math.isinf(value):
            return (0, -1 if value < 0 else 1, 0)
        return (1, Decimal.from_float(value))
    if isinstance(value, Decimal):
        if value.is_nan():
            return (2, 0, 0)
        if value.is_infinite():
            return (0, -1 if value.is_signed() else 1, 0)
        return (1, value)
    return (1, Decimal(value))


def _row_sort_key(row: Sequence[Any]) -> Tuple[Any, ...]:
    """Sort rows without rounding or converting exact numeric values to float."""
    keys = []
    for value in row:
        if value is None:
            keys.append((0,))
        elif isinstance(value, bool):
            keys.append((1, int(value)))
        elif isinstance(value, (int, float, Decimal)):
            keys.append((2,) + _numeric_sort_key(value))
        elif isinstance(value, (bytes, bytearray, memoryview)):
            keys.append((3, bytes(value)))
        elif isinstance(value, str):
            keys.append((4, value))
        else:
            value_type = type(value)
            keys.append((5, value_type.__module__, value_type.__qualname__, repr(value)))
    return tuple(keys)


def _numeric_values_equal(a: Any, b: Any, tol: float) -> bool:
    """Compare numerics exactly unless a floating-point result requires tolerance."""
    if isinstance(a, bool) or isinstance(b, bool):
        if isinstance(a, bool) and isinstance(b, bool):
            return a is b
        boolean = a if isinstance(a, bool) else b
        other = b if isinstance(a, bool) else a
        # PostgreSQL BOOLEAN is commonly returned as bool while SQLite-compatible
        # execution returns its 0/1 representation. Keep that one explicit bridge.
        return isinstance(other, (int, Decimal)) and not isinstance(other, bool) and other == int(boolean)

    if isinstance(a, (int, Decimal)) and isinstance(b, (int, Decimal)):
        da, db = Decimal(a), Decimal(b)
        if da.is_nan() or db.is_nan():
            return da.is_nan() and db.is_nan()
        return da == db

    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= tol

    floating = a if isinstance(a, float) else b
    exact = b if isinstance(a, float) else a
    if isinstance(exact, int):
        # Never round an exact integer through binary64. In particular, this
        # distinguishes adjacent integers above 2**53.
        return (
            math.isfinite(floating)
            and floating.is_integer()
            and Decimal.from_float(floating) == Decimal(exact)
        )

    # A float can only approximate a Decimal result. Compare its exact binary
    # value to the Decimal with an explicit absolute tolerance; do not round the
    # Decimal or scale the tolerance with the magnitude of the result.
    if math.isnan(floating):
        return exact.is_nan()
    if math.isinf(floating):
        return exact.is_infinite() and exact.is_signed() == (floating < 0)
    if exact.is_nan() or exact.is_infinite():
        return False
    return abs(Decimal.from_float(floating) - exact) <= Decimal(str(tol))


def _values_equal(a: Any, b: Any, tol: float) -> bool:
    if a is None or b is None:
        return a is b

    if isinstance(a, (bool, int, float, Decimal)) and isinstance(b, (bool, int, float, Decimal)):
        return _numeric_values_equal(a, b, tol)

    if isinstance(a, (bytes, bytearray, memoryview)) and isinstance(b, (bytes, bytearray, memoryview)):
        return bytes(a) == bytes(b)

    # Do not let Python's cross-type equality (for example, True == 1) make
    # textual or engine-specific values compare equal accidentally.
    return type(a) is type(b) and a == b


def _rows_equal(rows_a: Sequence[Sequence[Any]], rows_b: Sequence[Sequence[Any]], tol: float = 1.5e-3) -> bool:
    """Compare unordered row multisets, preserving duplicates and exact numerics.

    Integers and Decimal values are compared exactly. The absolute tolerance is
    used only when at least one compared numeric value is a float.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if len(rows_a) != len(rows_b):
        return False

    sorted_a = sorted((tuple(row) for row in rows_a), key=_row_sort_key)
    sorted_b = sorted((tuple(row) for row in rows_b), key=_row_sort_key)
    for row_a, row_b in zip(sorted_a, sorted_b):
        if len(row_a) != len(row_b):
            return False
        if not all(_values_equal(a, b, tol) for a, b in zip(row_a, row_b)):
            return False
    return True


def _register_sqlite_compat_udfs(conn: sqlite3.Connection, dialect: Dialect = Dialect.SQLITE) -> None:
    """Registers PostgreSQL/MySQL compatibility functions & aggregates on a SQLite connection for cross-dialect testing."""
    import math

    conn.create_function("SQRT", 1, lambda x: math.sqrt(float(x)) if x is not None and float(x) >= 0 else None)
    conn.create_function("LEAST", 2, lambda a, b: min(a, b) if (a is not None and b is not None) else None)
    conn.create_function("GREATEST", 2, lambda a, b: max(a, b) if (a is not None and b is not None) else None)
    conn.create_function("IF", 3, lambda cond, a, b: a if cond else b)

    def _concat(*args: Any) -> Optional[str]:
        if any(a is None for a in args):
            return None
        return "".join(str(a) for a in args)

    def _concat_ws(sep: Any, *args: Any) -> Optional[str]:
        if sep is None:
            return None
        return str(sep).join(str(a) for a in args if a is not None)

    conn.create_function("CONCAT", -1, _concat)
    conn.create_function("CONCAT_WS", -1, _concat_ws)
    conn.create_function("PG_ROW_KEY", -1, lambda *args: "(" + ",".join("" if a is None else str(a) for a in args) + ")")

    class _VarPopAgg:
        def __init__(self) -> None:
            self.vals: List[float] = []
        def step(self, x: Any) -> None:
            if x is not None:
                self.vals.append(float(x))
        def finalize(self) -> Optional[float]:
            n = len(self.vals)
            if n < 1:
                return None
            mean = sum(self.vals) / n
            return sum((v - mean) ** 2 for v in self.vals) / n

    class _VarSampAgg:
        def __init__(self) -> None:
            self.vals: List[float] = []
        def step(self, x: Any) -> None:
            if x is not None:
                self.vals.append(float(x))
        def finalize(self) -> Optional[float]:
            n = len(self.vals)
            if n < 2:
                return None
            mean = sum(self.vals) / n
            return sum((v - mean) ** 2 for v in self.vals) / (n - 1)

    class _StdPopAgg(_VarPopAgg):
        def finalize(self) -> Optional[float]:
            v = super().finalize()
            return math.sqrt(v) if v is not None else None

    class _StdSampAgg(_VarSampAgg):
        def finalize(self) -> Optional[float]:
            v = super().finalize()
            return math.sqrt(v) if v is not None else None

    class _BitAndAgg:
        def __init__(self) -> None:
            self.acc: Optional[int] = None
        def step(self, x: Any) -> None:
            if x is not None:
                self.acc = int(x) if self.acc is None else (self.acc & int(x))
        def finalize(self) -> Optional[int]:
            if self.acc is None and dialect == Dialect.MYSQL:
                return -1
            return self.acc

    class _BitOrAgg:
        def __init__(self) -> None:
            self.acc: Optional[int] = None
        def step(self, x: Any) -> None:
            if x is not None:
                self.acc = int(x) if self.acc is None else (self.acc | int(x))
        def finalize(self) -> Optional[int]:
            if self.acc is None and dialect == Dialect.MYSQL:
                return 0
            return self.acc

    class _BitXorAgg:
        def __init__(self) -> None:
            self.acc: Optional[int] = None
        def step(self, x: Any) -> None:
            if x is not None:
                self.acc = int(x) if self.acc is None else (self.acc ^ int(x))
        def finalize(self) -> Optional[int]:
            if self.acc is None and dialect == Dialect.MYSQL:
                return 0
            return self.acc

    class _BoolAndAgg:
        def __init__(self) -> None:
            self.seen = False
            self.all_true = True
        def step(self, x: Any) -> None:
            if x is not None:
                self.seen = True
                if not bool(x):
                    self.all_true = False
        def finalize(self) -> Optional[int]:
            return (1 if self.all_true else 0) if self.seen else None

    class _BoolOrAgg:
        def __init__(self) -> None:
            self.seen = False
            self.any_true = False
        def step(self, x: Any) -> None:
            if x is not None:
                self.seen = True
                if bool(x):
                    self.any_true = True
        def finalize(self) -> Optional[int]:
            return (1 if self.any_true else 0) if self.seen else None

    conn.create_aggregate("VAR_POP", 1, _VarPopAgg)
    conn.create_aggregate("VARIANCE", 1, _VarPopAgg)
    conn.create_aggregate("VAR_SAMP", 1, _VarSampAgg)
    conn.create_aggregate("STDDEV_POP", 1, _StdPopAgg)
    conn.create_aggregate("STDDEV", 1, _StdPopAgg)
    conn.create_aggregate("STD", 1, _StdPopAgg)
    conn.create_aggregate("STDDEV_SAMP", 1, _StdSampAgg)
    conn.create_aggregate("BIT_AND", 1, _BitAndAgg)
    conn.create_aggregate("BIT_OR", 1, _BitOrAgg)
    conn.create_aggregate("BIT_XOR", 1, _BitXorAgg)
    conn.create_aggregate("BOOL_AND", 1, _BoolAndAgg)
    conn.create_aggregate("EVERY", 1, _BoolAndAgg)
    conn.create_aggregate("BOOL_OR", 1, _BoolOrAgg)


def _transpile_pg_expr_to_sqlite(sql: str) -> str:
    s = sql
    s = re.sub(r"\bROW\s*\(((?:[^()]|\((?:[^()]|\([^()]*\))*\))+)\)\s*::\s*TEXT\b", r"PG_ROW_KEY(\1)", s, flags=re.IGNORECASE)
    s = re.sub(r"::\s*(?:TEXT|BIGINT|DOUBLE\s+PRECISION|INTEGER|BOOLEAN|NUMERIC)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bIS\s+NOT\s+DISTINCT\s+FROM\b", "IS", s, flags=re.IGNORECASE)
    s = re.sub(r"\bDOUBLE\s+PRECISION\b", "REAL", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBOOLEAN\b", "INTEGER", s, flags=re.IGNORECASE)
    s = re.sub(r"\bTRUE\b", "1", s, flags=re.IGNORECASE)
    s = re.sub(r"\bFALSE\b", "0", s, flags=re.IGNORECASE)
    # Replace PostgreSQL bitwise XOR `#` with SQLite equivalent `((a | b) - (a & b))`
    for _ in range(3):
        s = re.sub(
            r"\(([^()#]+?)\s*#\s*(\((?:[^()]|\([^()]*\))*\)|[A-Za-z0-9_.]+)\)",
            r"((\1 | \2) - (\1 & \2))",
            s,
        )
    return s


def _transpile_mysql_expr_to_sqlite(sql: str, pk_cols: Sequence[str] = ("__mv_pk",)) -> str:
    s = sql
    s = re.sub(r"\bCREATE\s+ALGORITHM\s*=\s*\w+\s+VIEW\b", "CREATE VIEW", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBIGINT\s+UNSIGNED\b", "INTEGER", s, flags=re.IGNORECASE)
    s = s.replace("18446744073709551615", "-1")
    s = re.sub(r"(?<!ELSE )\bBIT_XOR\s*\(\s*([^()]+?)\s*\)", r"COALESCE(BIT_XOR(\1), 0)", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<!ELSE )\bBIT_OR\s*\(\s*([^()]+?)\s*\)", r"COALESCE(BIT_OR(\1), 0)", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<!ELSE )\bBIT_AND\s*\(\s*([^()]+?)\s*\)", r"COALESCE(BIT_AND(\1), -1)", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+FROM\s+DUAL\b", "", s, flags=re.IGNORECASE)
    s = s.replace("<=>", " IS ")
    s = re.sub(r"\bCAST\s*\(([^)]+?)\s+AS\s+CHAR\s*\)", r"CAST(\1 AS TEXT)", s, flags=re.IGNORECASE)
    # Replace MySQL bitwise XOR `^`
    for _ in range(3):
        s = re.sub(
            r"\(([^()^]+?)\s*\^\s*(\((?:[^()]|\([^()]*\))*\)|VALUES\([^)]+\)|[A-Za-z0-9_.]+)\)",
            r"((\1 | \2) - (\1 & \2))",
            s,
        )
    # Replace `VALUES(col)` with `excluded.col`
    s = re.sub(r"\bVALUES\s*\(\s*([A-Za-z0-9_]+)\s*\)", r"excluded.\1", s, flags=re.IGNORECASE)
    return s


# -----------------------------------------------------------------------------
# Live External Database Verification & 50-View x 33-Step Battery
# -----------------------------------------------------------------------------


def _verify_external_db_mv(cur, mv_name: str, select_sql: str, context: str = "") -> None:
    cur.execute(f"SELECT * FROM {mv_name}")
    mv_rows = [tuple(row) for row in cur.fetchall()]
    cur.execute(select_sql)
    live_rows = [tuple(row) for row in cur.fetchall()]
    if not _rows_equal(mv_rows, live_rows):
        raise AssertionError(
            f"[{context}] External DB MV mismatch on {mv_name}!\n"
            f"  Query: {select_sql}\n"
            f"  MV ({len(mv_rows)} rows):   {mv_rows[:10]}\n"
            f"  Live ({len(live_rows)} rows): {live_rows[:10]}"
        )


def _get_live_engine_50_view_battery(dialect: Dialect) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Returns a 4-table schema (`warehouses`, `customers`, `orders`, `order_tags` with composite PK)
    and 50 comprehensive view definitions exercising every generator strategy, aggregate function,
    join topology, set operation, CTE pattern, window function, and edge case for `dialect`.
    """
    if dialect == Dialect.POSTGRES:
        schema = """
        CREATE TABLE warehouses (
            id INT PRIMARY KEY,
            w_code VARCHAR(32) NOT NULL,
            country VARCHAR(32)
        );
        CREATE TABLE customers (
            id INT PRIMARY KEY,
            w_id INT,
            name VARCHAR(100) NOT NULL,
            tier VARCHAR(32) NOT NULL
        );
        CREATE TABLE orders (
            id INT PRIMARY KEY,
            customer_id INT NOT NULL,
            region VARCHAR(32),
            category VARCHAR(32) NOT NULL,
            amount DOUBLE PRECISION,
            qty INT NOT NULL,
            flags INT NOT NULL,
            is_paid BOOLEAN NOT NULL,
            status VARCHAR(32) NOT NULL
        );
        CREATE TABLE order_tags (
            order_id INT NOT NULL,
            tag VARCHAR(32) NOT NULL,
            weight INT NOT NULL,
            PRIMARY KEY (order_id, tag)
        );
        """
    elif dialect == Dialect.MYSQL:
        schema = """
        CREATE TABLE warehouses (
            id INT PRIMARY KEY,
            w_code VARCHAR(32) NOT NULL,
            country VARCHAR(32)
        );
        CREATE TABLE customers (
            id INT PRIMARY KEY,
            w_id INT,
            name VARCHAR(100) NOT NULL,
            tier VARCHAR(32) NOT NULL
        );
        CREATE TABLE orders (
            id INT PRIMARY KEY,
            customer_id INT NOT NULL,
            region VARCHAR(32),
            category VARCHAR(32) NOT NULL,
            amount DOUBLE,
            qty INT NOT NULL,
            flags INT NOT NULL,
            is_paid INT NOT NULL,
            status VARCHAR(32) NOT NULL
        );
        CREATE TABLE order_tags (
            order_id INT NOT NULL,
            tag VARCHAR(32) NOT NULL,
            weight INT NOT NULL,
            PRIMARY KEY (order_id, tag)
        );
        """
    else:
        schema = """
        CREATE TABLE warehouses (
            id INTEGER PRIMARY KEY,
            w_code TEXT NOT NULL,
            country TEXT
        );
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            w_id INTEGER,
            name TEXT NOT NULL,
            tier TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            region TEXT,
            category TEXT NOT NULL,
            amount REAL,
            qty INTEGER NOT NULL,
            flags INTEGER NOT NULL,
            is_paid INTEGER NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE order_tags (
            order_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            weight INTEGER NOT NULL,
            PRIMARY KEY (order_id, tag)
        );
        """

    if dialect == Dialect.POSTGRES:
        v01 = "SELECT region, COUNT(*) AS cnt, COUNT(amount) AS ca, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, VAR_SAMP(amount) AS vs, STDDEV_POP(amount) AS sdp, STDDEV_SAMP(amount) AS sds, BIT_AND(flags) AS ba, BIT_OR(flags) AS bo, BIT_XOR(flags) AS bx, BOOL_AND(is_paid) AS all_p, BOOL_OR(is_paid) AS any_p FROM orders WHERE status = 'settled' GROUP BY region HAVING COUNT(*) >= 1"
        v02 = "SELECT category, COUNT(*) AS cnt, COUNT(*) FILTER (WHERE is_paid) AS paid_cnt, SUM(amount) FILTER (WHERE status = 'settled') AS settled_sum, AVG(amount) FILTER (WHERE qty >= 2) AS bulk_avg FROM orders GROUP BY category"
        v10 = "SELECT region, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, BIT_XOR(flags) AS bx, BOOL_AND(is_paid) AS all_p, BOOL_OR(is_paid) AS any_p FROM orders WHERE status = 'settled' GROUP BY region"
        v12 = "SELECT COUNT(*) AS cnt, COUNT(amount) AS ca, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, STDDEV_SAMP(amount) AS sds, BOOL_AND(is_paid) AS all_p, BOOL_OR(is_paid) AS any_p FROM orders"
        v15 = "SELECT c.tier, o.region, COUNT(*) AS cnt, COUNT(o.amount) AS ca, SUM(o.amount * o.qty) AS rev, AVG(o.amount) AS avg_amt, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq, VAR_POP(o.amount) AS vp, STDDEV_SAMP(o.amount) AS sds, BIT_XOR(o.flags) AS bx, BOOL_AND(o.is_paid) AS all_p, BOOL_OR(o.is_paid) AS any_p FROM orders o INNER JOIN customers c ON o.customer_id = c.id WHERE o.status = 'settled' GROUP BY c.tier, o.region"
        v36 = "WITH u AS (SELECT region AS k, amount AS v, flags AS f FROM orders WHERE status = 'settled' UNION ALL SELECT tier AS k, 10.0 AS v, 3 AS f FROM customers) SELECT k, COUNT(*) AS cnt, COUNT(v) AS cv, SUM(v) AS sv, AVG(v) AS av, MIN(v) AS mnv, MAX(v) AS mxv, VAR_POP(v) AS vp, STDDEV_SAMP(v) AS sds, BIT_XOR(f) AS bx FROM u GROUP BY k"
    elif dialect == Dialect.MYSQL:
        v01 = "SELECT region, COUNT(*) AS cnt, COUNT(amount) AS ca, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, VAR_SAMP(amount) AS vs, STDDEV_POP(amount) AS sdp, STDDEV_SAMP(amount) AS sds, BIT_AND(flags) AS ba, BIT_OR(flags) AS bo, BIT_XOR(flags) AS bx FROM orders WHERE status = 'settled' GROUP BY region HAVING COUNT(*) >= 1"
        v02 = "SELECT category, COUNT(*) AS cnt, SUM(CASE WHEN is_paid = 1 THEN 1 ELSE 0 END) AS paid_cnt, SUM(CASE WHEN status = 'settled' THEN amount ELSE 0.0 END) AS settled_sum, AVG(amount) AS avg_amt FROM orders GROUP BY category"
        v10 = "SELECT region, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, BIT_XOR(flags) AS bx FROM orders WHERE status = 'settled' GROUP BY region"
        v12 = "SELECT COUNT(*) AS cnt, COUNT(amount) AS ca, SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q, VAR_POP(amount) AS vp, STDDEV_SAMP(amount) AS sds FROM orders"
        v15 = "SELECT c.tier, o.region, COUNT(*) AS cnt, COUNT(o.amount) AS ca, SUM(o.amount * o.qty) AS rev, AVG(o.amount) AS avg_amt, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq, VAR_POP(o.amount) AS vp, STDDEV_SAMP(o.amount) AS sds, BIT_XOR(o.flags) AS bx FROM orders o INNER JOIN customers c ON o.customer_id = c.id WHERE o.status = 'settled' GROUP BY c.tier, o.region"
        v36 = "WITH u AS (SELECT region AS k, amount AS v, flags AS f FROM orders WHERE status = 'settled' UNION ALL SELECT tier AS k, 10.0 AS v, 3 AS f FROM customers) SELECT k, COUNT(*) AS cnt, COUNT(v) AS cv, SUM(v) AS sv, AVG(v) AS av, MIN(v) AS mnv, MAX(v) AS mxv, VAR_POP(v) AS vp, STDDEV_SAMP(v) AS sds, BIT_XOR(f) AS bx FROM u GROUP BY k"
    else:
        v01 = "SELECT region, COUNT(*) AS cnt, COUNT(amount) AS ca, SUM(amount) AS s_amt, AVG(amount) AS a_amt, TOTAL(amount) AS tot_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q FROM orders WHERE status = 'settled' GROUP BY region HAVING COUNT(*) >= 1"
        v02 = "SELECT category, COUNT(*) AS cnt, COUNT(*) FILTER (WHERE is_paid = 1) AS paid_cnt, SUM(amount) FILTER (WHERE status = 'settled') AS settled_sum, AVG(amount) FILTER (WHERE qty >= 2) AS bulk_avg, TOTAL(amount) AS tot_a FROM orders GROUP BY category"
        v10 = "SELECT region, SUM(amount) AS s_amt, AVG(amount) AS a_amt, TOTAL(amount) AS tot_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q FROM orders WHERE status = 'settled' GROUP BY region"
        v12 = "SELECT COUNT(*) AS cnt, COUNT(amount) AS ca, SUM(amount) AS s_amt, AVG(amount) AS a_amt, TOTAL(amount) AS tot_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q FROM orders"
        v15 = "SELECT c.tier, o.region, COUNT(*) AS cnt, COUNT(o.amount) AS ca, SUM(o.amount * o.qty) AS rev, AVG(o.amount) AS avg_amt, TOTAL(o.amount) AS tot_a, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq FROM orders o INNER JOIN customers c ON o.customer_id = c.id WHERE o.status = 'settled' GROUP BY c.tier, o.region"
        v36 = "WITH u AS (SELECT region AS k, amount AS v, flags AS f FROM orders WHERE status = 'settled' UNION ALL SELECT tier AS k, 10.0 AS v, 3 AS f FROM customers) SELECT k, COUNT(*) AS cnt, COUNT(v) AS cv, SUM(v) AS sv, AVG(v) AS av, TOTAL(v) AS tv, MIN(v) AS mnv, MAX(v) AS mxv FROM u GROUP BY k"

    view_specs: List[Tuple[str, str]] = [
        # --- Family 1: Tier 1 Single-Table Algebraic (14 views) ---
        ("v01_alg_full", v01),
        ("v02_alg_filter", v02),
        ("v03_alg_compound", "SELECT region, category, COUNT(*) AS cnt, COALESCE(SUM(amount * qty), 0.0) - COALESCE(SUM(amount), 0.0) AS net_rev, COALESCE(MAX(qty) - MIN(qty), 0) AS q_span FROM orders GROUP BY region, category"),
        ("v04_alg_expr_coal", "SELECT COALESCE(region, 'UNK') AS reg_norm, COUNT(*) AS cnt, SUM(amount) AS s_amt, AVG(amount) AS a_amt FROM orders GROUP BY COALESCE(region, 'UNK')"),
        ("v05_alg_expr_lower", "SELECT LOWER(category) AS l_cat, COUNT(*) AS cnt, SUM(qty) AS sq, MIN(qty) AS mnq, MAX(qty) AS mxq FROM orders GROUP BY LOWER(category)"),
        ("v06_alg_expr_case", "SELECT CASE WHEN qty >= 3 THEN 'bulk' ELSE 'small' END AS sz, COUNT(*) AS cnt, SUM(amount) AS s_amt FROM orders GROUP BY CASE WHEN qty >= 3 THEN 'bulk' ELSE 'small' END"),
        ("v07_alg_having_comp", "SELECT region, COUNT(*) AS cnt, SUM(qty) AS sq, SUM(amount) AS sa FROM orders GROUP BY region HAVING COUNT(*) >= 2 OR COALESCE(SUM(qty), 0) >= 4"),
        ("v08_alg_hidden_grp", "SELECT COUNT(*) AS cnt, SUM(amount) AS s_amt, MIN(qty) AS mnq, MAX(qty) AS mxq FROM orders WHERE status = 'settled' GROUP BY region"),
        ("v09_alg_hidden_expr_grp", "SELECT COUNT(*) AS cnt, SUM(qty) AS sq FROM orders GROUP BY COALESCE(region, 'NONE'), category"),
        ("v10_alg_no_cnt_star", v10),
        ("v11_alg_hidden_and_nocnt", "SELECT SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mnq, MAX(qty) AS mxq FROM orders GROUP BY category"),
        ("v12_alg_global_full", v12),
        ("v13_alg_global_nocnt", "SELECT SUM(amount) AS s_amt, AVG(amount) AS a_amt, MIN(qty) AS mn_q, MAX(qty) AS mx_q FROM orders WHERE status = 'settled'"),
        ("v14_alg_comp_pk_tbl", "SELECT tag, COUNT(*) AS cnt, SUM(weight) AS sw, AVG(weight) AS aw, MIN(weight) AS mnw, MAX(weight) AS mxw FROM order_tags GROUP BY tag"),

        # --- Family 2: Tier 1 Multi-Table INNER & CROSS JOIN Algebraic (9 views) ---
        ("v15_join_2t_full", v15),
        ("v16_join_3t_chain", "SELECT w.country, c.tier, o.category, COUNT(*) AS cnt, SUM(o.qty) AS sq, SUM(o.amount) AS sa, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq FROM orders o JOIN customers c ON o.customer_id = c.id JOIN warehouses w ON c.w_id = w.id GROUP BY w.country, c.tier, o.category"),
        ("v17_join_cross", "SELECT w.w_code, c.tier, COUNT(*) AS cnt, SUM(c.id) AS sid FROM warehouses w CROSS JOIN customers c GROUP BY w.w_code, c.tier"),
        ("v18_join_expr_having", "SELECT LOWER(c.tier) AS l_tier, COALESCE(o.region, 'NONE') AS r_norm, COUNT(*) AS cnt, COALESCE(SUM(o.amount * o.qty), 0.0) AS rev FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY LOWER(c.tier), COALESCE(o.region, 'NONE') HAVING COUNT(*) >= 1"),
        ("v19_join_hidden_grp", "SELECT c.tier, COUNT(*) AS cnt, SUM(o.amount) AS s_amt FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.tier, o.category"),
        ("v20_join_no_cnt", "SELECT c.tier, SUM(o.amount) AS s_amt, AVG(o.amount) AS a_amt, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq FROM orders o JOIN customers c ON o.customer_id = c.id GROUP BY c.tier"),
        ("v21_join_global", "SELECT COUNT(*) AS cnt, SUM(o.amount) AS s_amt, AVG(o.amount) AS a_amt, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq FROM orders o JOIN customers c ON o.customer_id = c.id WHERE o.status = 'settled'"),
        ("v22_join_global_nocnt", "SELECT SUM(o.qty) AS sq, AVG(o.amount) AS a_amt, MIN(o.qty) AS mnq, MAX(o.qty) AS mxq FROM orders o JOIN customers c ON o.customer_id = c.id"),
        ("v23_join_comp_pk", "SELECT ot.tag, o.status, COUNT(*) AS cnt, SUM(ot.weight * o.qty) AS w_qty FROM order_tags ot JOIN orders o ON ot.order_id = o.id GROUP BY ot.tag, o.status"),

        # --- Family 3: Tier 1 Z-Set Set Operations, DISTINCT & PK-Lineage Projections (12 views) ---
        ("v24_set_uall_2t", "SELECT id, region AS loc FROM orders WHERE qty > 1 UNION ALL SELECT id, tier AS loc FROM customers"),
        ("v25_set_uall_same_3b", "SELECT id, category AS label FROM orders WHERE status = 'settled' UNION ALL SELECT id, status AS label FROM orders WHERE qty >= 3 UNION ALL SELECT id, COALESCE(region, 'NA') AS label FROM orders WHERE is_paid = TRUE" if dialect == Dialect.POSTGRES else "SELECT id, category AS label FROM orders WHERE status = 'settled' UNION ALL SELECT id, status AS label FROM orders WHERE qty >= 3 UNION ALL SELECT id, COALESCE(region, 'NA') AS label FROM orders WHERE is_paid = 1"),
        ("v26_set_union_2t", "SELECT region AS code FROM orders WHERE region IS NOT NULL UNION SELECT tier AS code FROM customers"),
        ("v27_set_union_same", "SELECT region AS r FROM orders WHERE status = 'settled' AND region IS NOT NULL UNION SELECT category AS r FROM orders WHERE qty >= 2"),
        ("v28_set_inter_2t", "SELECT region AS code FROM orders WHERE region IS NOT NULL INTERSECT SELECT tier AS code FROM customers"),
        ("v29_set_inter_same", "SELECT category FROM orders WHERE status = 'settled' INTERSECT SELECT category FROM orders WHERE qty >= 2"),
        ("v30_set_except_2t", "SELECT region AS code FROM orders WHERE region IS NOT NULL EXCEPT SELECT tier AS code FROM customers"),
        ("v31_set_except_same", "SELECT category FROM orders WHERE status = 'settled' EXCEPT SELECT category FROM orders WHERE qty >= 4"),
        ("v32_dist_orders", "SELECT DISTINCT region, category, status FROM orders WHERE region IS NOT NULL"),
        ("v33_dist_comp_pk", "SELECT DISTINCT tag, weight FROM order_tags WHERE weight > 0"),
        ("v34_proj_orders", "SELECT id, customer_id, COALESCE(region, 'GLOBAL') AS eff_reg, amount * qty AS line_total FROM orders WHERE qty > 0"),
        ("v35_proj_comp_pk", "SELECT order_id, tag, weight * 10 AS scaled_w FROM order_tags WHERE weight >= 1"),

        # --- Family 4: Tier 1 UNION ALL Aggregations (6 views) ---
        ("v36_uagg_cte_full", v36),
        ("v37_uagg_subq", "SELECT k, COUNT(*) AS cnt, SUM(v) AS sv, AVG(v) AS av FROM (SELECT category AS k, qty AS v FROM orders UNION ALL SELECT tag AS k, weight AS v FROM order_tags) AS sub GROUP BY k"),
        ("v38_uagg_hidden_grp", "WITH u AS (SELECT region AS k, qty AS q FROM orders UNION ALL SELECT tier AS k, id AS q FROM customers) SELECT COUNT(*) AS cnt, SUM(q) AS sq FROM u GROUP BY k"),
        ("v39_uagg_no_cnt", "WITH u AS (SELECT region AS k, amount AS v FROM orders WHERE status = 'settled' UNION ALL SELECT tier AS k, 20.0 AS v FROM customers) SELECT k, SUM(v) AS sv, AVG(v) AS av, MIN(v) AS mnv, MAX(v) AS mxv FROM u GROUP BY k"),
        ("v40_uagg_global", "WITH u AS (SELECT qty AS q FROM orders UNION ALL SELECT weight AS q FROM order_tags) SELECT COUNT(*) AS cnt, SUM(q) AS sq, AVG(q) AS aq, MIN(q) AS mnq, MAX(q) AS mxq FROM u"),
        ("v41_uagg_same_tbl", "WITH u AS (SELECT category AS c, amount AS a FROM orders WHERE status = 'settled' UNION ALL SELECT category AS c, amount * 2.0 AS a FROM orders WHERE qty >= 3) SELECT c, COUNT(*) AS cnt, SUM(a) AS sa FROM u WHERE c IS NOT NULL GROUP BY c HAVING COUNT(*) >= 1"),

        # --- Family 5: Simple CTE Unwrapping & Tier 2 Holistic Group-Scoped (5 views) ---
        ("v42_cte_unwrap_alias", "WITH settled_o AS (SELECT customer_id AS cid, region AS reg, amount * qty AS gross FROM orders WHERE status = 'settled') SELECT reg AS region_key, COUNT(*) AS cnt, SUM(gross) AS total_gross, AVG(gross) AS avg_gross FROM settled_o WHERE cid >= 1 GROUP BY reg"),
        ("v43_t2_distinct_aggs", "SELECT region, COUNT(*) AS cnt, COUNT(DISTINCT customer_id) AS dc, SUM(DISTINCT qty) AS sdq, AVG(DISTINCT qty) AS adq, MIN(amount) AS mna, MAX(amount) AS mxa FROM orders GROUP BY region"),
        ("v44_t2_left_join_cust", "SELECT c.tier, COUNT(o.id) AS ord_cnt, COALESCE(SUM(o.qty), 0) AS tot_q, AVG(o.amount) AS avg_a FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.tier"),
        ("v45_t2_left_join_hidden", "SELECT COUNT(c.id) AS cust_cnt, COALESCE(SUM(o.qty), 0) AS tot_q FROM warehouses w LEFT JOIN customers c ON w.id = c.w_id LEFT JOIN orders o ON c.id = o.customer_id GROUP BY w.w_code"),
        ("v46_t2_where_exists", "SELECT tier, COUNT(*) AS cnt FROM customers c WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id AND o.status = 'settled') GROUP BY tier"),

        # --- Family 6: Tier 3 Universal Atomic Sync (4 views) ---
        ("v47_t3_window_rank", "SELECT id, region, category, amount, RANK() OVER (PARTITION BY category ORDER BY amount DESC, id ASC) AS rnk, DENSE_RANK() OVER (ORDER BY qty DESC, id ASC) AS drnk FROM orders"),
        ("v48_t3_chained_cte", "WITH c_totals AS (SELECT customer_id, SUM(qty) AS tq FROM orders GROUP BY customer_id), ranked AS (SELECT customer_id, tq, ROW_NUMBER() OVER (ORDER BY tq DESC, customer_id ASC) AS rn FROM c_totals) SELECT c.name, r.tq, r.rn FROM ranked r JOIN customers c ON r.customer_id = c.id"),
        ("v49_t3_scalar_subq", "SELECT c.id AS cid, c.name, (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS ord_cnt FROM customers c"),
        ("v50_t3_top_n_limit", "SELECT id, category, qty, amount FROM orders ORDER BY qty DESC, id ASC LIMIT 4"),
    ]
    return schema, view_specs


def _get_complex_dml_and_tx_steps(dialect: Dialect) -> List[Tuple[str, str]]:
    """
    Returns a 34-step complex DML and transactional intermediate-state verification sequence
    for `dialect`. Used identically by both in-memory transpiled backends and live external
    database servers.
    """
    t_val = "TRUE" if dialect == Dialect.POSTGRES else "1"
    f_val = "FALSE" if dialect == Dialect.POSTGRES else "0"
    if dialect == Dialect.MYSQL:
        upsert_ord = (
            "INSERT INTO orders VALUES (1, 1, 'EU', 'sw', 35.0, 7, 9, 1, 'settled'), "
            "(9, 4, 'LATAM', 'hw', 60.0, 3, 5, 1, 'settled') "
            "ON DUPLICATE KEY UPDATE region = VALUES(region), category = VALUES(category), "
            "amount = VALUES(amount), qty = VALUES(qty), flags = VALUES(flags), is_paid = VALUES(is_paid), status = VALUES(status);"
        )
        upsert_tag = "INSERT IGNORE INTO order_tags VALUES (1, 'urgent', 999), (6, 'fresh_tag', 11);"
        replace_tag = "REPLACE INTO order_tags VALUES (1, 'p0_urgent', 25), (4, 'replaced_bulk', 19);"
        final_drain = "DELETE FROM orders;"
    elif dialect == Dialect.POSTGRES:
        upsert_ord = (
            "INSERT INTO orders VALUES (1, 1, 'EU', 'sw', 35.0, 7, 9, TRUE, 'settled'), "
            "(9, 4, 'LATAM', 'hw', 60.0, 3, 5, TRUE, 'settled') "
            "ON CONFLICT (id) DO UPDATE SET region = excluded.region, category = excluded.category, "
            "amount = excluded.amount, qty = excluded.qty, flags = excluded.flags, is_paid = excluded.is_paid, status = excluded.status;"
        )
        upsert_tag = "INSERT INTO order_tags VALUES (1, 'urgent', 999), (6, 'fresh_tag', 11) ON CONFLICT (order_id, tag) DO NOTHING;"
        replace_tag = (
            "INSERT INTO order_tags VALUES (1, 'p0_urgent', 25), (4, 'replaced_bulk', 19) "
            "ON CONFLICT (order_id, tag) DO UPDATE SET weight = excluded.weight;"
        )
        final_drain = "TRUNCATE TABLE orders;"
    else:
        upsert_ord = (
            "INSERT INTO orders VALUES (1, 1, 'EU', 'sw', 35.0, 7, 9, 1, 'settled'), "
            "(9, 4, 'LATAM', 'hw', 60.0, 3, 5, 1, 'settled') "
            "ON CONFLICT (id) DO UPDATE SET region = excluded.region, category = excluded.category, "
            "amount = excluded.amount, qty = excluded.qty, flags = excluded.flags, is_paid = excluded.is_paid, status = excluded.status;"
        )
        upsert_tag = "INSERT INTO order_tags VALUES (1, 'urgent', 999), (6, 'fresh_tag', 11) ON CONFLICT (order_id, tag) DO NOTHING;"
        replace_tag = "REPLACE INTO order_tags VALUES (1, 'p0_urgent', 25), (4, 'replaced_bulk', 19);"
        final_drain = "DELETE FROM orders;"

    return [
        # Phase 1: Complex Single-Statement DMLs (multi-row, INSERT..SELECT, Upsert, CASE swap, correlated subquery, PK mutation)
        ("ins_multi", f"INSERT INTO orders VALUES (6, 2, 'NA', 'hw', NULL, 2, 4, {t_val}, 'settled'), (7, 3, 'APAC', 'cloud', 40.0, 2, 2, {t_val}, 'settled'), (8, 1, 'NA', 'hw', 15.0, 6, 3, {f_val}, 'pending');"),
        ("ins_select", "INSERT INTO order_tags SELECT id, 'auto_tag', qty * 2 FROM orders WHERE status = 'settled' AND id >= 4;"),
        ("upsert_ord", upsert_ord),
        ("upsert_tag", upsert_tag),
        ("upd_case_swap", "UPDATE orders SET region = CASE WHEN region = 'NA' THEN 'EU' WHEN region = 'EU' THEN 'NA' ELSE region END, amount = CASE WHEN amount IS NULL THEN 12.5 ELSE amount * 1.5 END, qty = 20 - qty, status = CASE WHEN status = 'pending' THEN 'settled' ELSE status END WHERE id IN (1, 2, 3, 4, 6);"),
        ("upd_corr_subq", "UPDATE orders SET amount = COALESCE(amount, 0.0) + (SELECT COALESCE(SUM(ot.weight), 0) FROM order_tags ot WHERE ot.order_id = orders.id), qty = qty + 1 WHERE EXISTS (SELECT 1 FROM customers c WHERE c.id = orders.customer_id AND c.tier IN ('VIP', 'NA'));"),
        ("upd_comp_pk", "UPDATE order_tags SET tag = CASE WHEN tag = 'urgent' THEN 'p0_urgent' WHEN tag = 'fragile' THEN 'p1_fragile' ELSE tag END, weight = weight + 3 WHERE order_id IN (1, 2);"),
        ("replace_tag", replace_tag),
        ("upd_pk_shift", "UPDATE orders SET id = CASE WHEN id = 1 THEN 101 WHEN id = 2 THEN 102 ELSE id END WHERE id IN (1, 2);"),
        ("upd_pk_restore", "UPDATE orders SET id = CASE WHEN id = 101 THEN 1 WHEN id = 102 THEN 2 ELSE id END WHERE id IN (101, 102);"),
        ("upd_cust_multi", "UPDATE customers SET tier = CASE WHEN tier = 'NA' THEN 'VIP' WHEN tier = 'VIP' THEN 'EU' ELSE 'GOLD' END, w_id = CASE WHEN w_id = 1 THEN 2 ELSE 1 END WHERE id IN (1, 2, 3);"),
        ("upd_wh_multi", "UPDATE warehouses SET country = CASE WHEN country = 'US' THEN 'CA' WHEN country IS NULL THEN 'US' ELSE country END WHERE id IN (1, 2, 3);"),

        # Phase 2: Multi-Statement Transaction with Intermediate Verification, Nested Savepoints & Rollbacks
        ("tx1_begin", "START TRANSACTION;" if dialect == Dialect.MYSQL else "BEGIN;"),
        ("tx1_ins_in_tx", f"INSERT INTO orders VALUES (10, 2, 'NA', 'ai', 120.0, 9, 15, {t_val}, 'settled'), (11, 3, 'EU', 'ai', 80.0, 4, 6, {f_val}, 'settled');"),
        ("tx1_sp_alpha", "SAVEPOINT sp_alpha;"),
        ("tx1_sp_alpha_upd1", "UPDATE orders SET region = 'APAC', amount = 250.0, qty = 15 WHERE id IN (4, 10);"),
        ("tx1_sp_alpha_upd2", "UPDATE customers SET tier = 'PLATINUM' WHERE id = 2;"),
        ("tx1_sp_beta", "SAVEPOINT sp_beta;"),
        ("tx1_sp_beta_del1", "DELETE FROM order_tags WHERE order_id IN (1, 4);"),
        ("tx1_sp_beta_del2", "DELETE FROM orders WHERE id IN (1, 3, 7);"),
        ("tx1_rb_sp_beta", "ROLLBACK TO SAVEPOINT sp_beta;"),
        ("tx1_rb_sp_alpha", "ROLLBACK TO SAVEPOINT sp_alpha;"),
        ("tx1_post_rb_upd", "UPDATE order_tags SET weight = weight + 5 WHERE order_id = 4;"),
        ("tx1_release_sp", "RELEASE SAVEPOINT sp_alpha;"),
        ("tx1_full_rb", "ROLLBACK;"),

        # Phase 3: Multi-Statement Transaction with Intermediate Verification Before & After COMMIT
        ("tx2_begin", "START TRANSACTION;" if dialect == Dialect.MYSQL else "BEGIN;"),
        ("tx2_ins_wh", "INSERT INTO warehouses VALUES (4, 'WH_APAC', 'JP');"),
        ("tx2_ins_cust", "INSERT INTO customers VALUES (5, 4, 'Elena', 'VIP');"),
        ("tx2_ins_ord", f"INSERT INTO orders VALUES (12, 5, 'APAC', 'hw', 300.0, 10, 7, {t_val}, 'settled');"),
        ("tx2_ins_tag", "INSERT INTO order_tags VALUES (12, 'vip_order', 20);"),
        ("tx2_del_corr", "DELETE FROM orders WHERE id IN (2, 5) OR customer_id IN (SELECT id FROM customers WHERE name = 'Cara');"),
        ("tx2_del_cust", "DELETE FROM customers WHERE id = 3;"),
        ("tx2_commit", "COMMIT;"),
        ("final_drain", final_drain),
    ]


def _run_live_manager_lifecycle_check(conn, cur, dialect: Dialect) -> int:
    """
    Exercises MaterializedViewManager lifecycle methods (`fetch_schema`, `register_view`,
    `sync_views` create/unchanged/updated/dropped, `apply_schema_ddl` including `ALTER TABLE`,
    `backfill_incrementally` across `pk_range_fold`, `group_key_batch`, and `offset_batch`
    with `on_progress`, read-only MV enforcement, `RIGHT JOIN` / `FULL OUTER JOIN` live
    execution, `bulk_changes`, and `drop_view`) against a live database connection.
    """
    checks = 0
    drop_suffix = " CASCADE" if dialect == Dialect.POSTGRES else ""
    for v_n in ("v_mgr_alg", "v_mgr_dist", "v_mgr_win", "v_mgr_off", "v_mgr_right", "v_mgr_full"):
        cur.execute(f"DROP VIEW IF EXISTS __matview_{v_n}{drop_suffix};")
        cur.execute(f"DROP TABLE IF EXISTS __matview_store_{v_n}{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS __matview_dist_v_mgr_dist__dc{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS __matview_aux_v_mgr_dist_dc_1{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS __matview_catalog{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_items{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_comp_items{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_join_a{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_join_b{drop_suffix};")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineLimitationWarning)
        mgr = MaterializedViewManager(conn, dialect=dialect, ignore_limitations=True)
        mgr.apply_schema_ddl(
            """
            CREATE TABLE mgr_items (
                id INT PRIMARY KEY,
                grp VARCHAR(32) NOT NULL,
                tag VARCHAR(32),
                val INT NOT NULL
            );
            CREATE TABLE mgr_comp_items (
                a INT NOT NULL,
                b INT NOT NULL,
                label VARCHAR(32) NOT NULL,
                PRIMARY KEY (a, b)
            );
            """
        )
        cur.execute(
            "INSERT INTO mgr_items VALUES "
            "(1, 'A', 'x', 10), (2, 'A', 'y', 20), (3, 'B', 'x', 30), (4, 'B', 'x', 40), (5, 'C', NULL, 50);"
        )
        cur.execute(
            "INSERT INTO mgr_comp_items VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z');"
        )
        if hasattr(conn, "commit"):
            conn.commit()

        q_alg = "SELECT grp, COUNT(*) AS cnt, SUM(val) AS sv, MIN(val) AS mnv, MAX(val) AS mxv FROM mgr_items GROUP BY grp"
        q_dist = "SELECT grp, COUNT(DISTINCT tag) AS dc, SUM(val) AS sv FROM mgr_items GROUP BY grp"
        q_win = "SELECT id, grp, val, ROW_NUMBER() OVER (PARTITION BY grp ORDER BY val DESC, id ASC) AS rn FROM mgr_items"
        q_off = "SELECT a, b, label FROM mgr_comp_items"

        # 1. pk_range_fold strategy
        progress_events: List[Tuple[int, int]] = []
        p_alg = mgr.register_view(
            f"CREATE VIEW v_mgr_alg AS {q_alg};",
            incremental_backfill=True,
            batch_size=2,
            on_progress=lambda b, d: progress_events.append((b, d)),
        )
        assert p_alg.incremental_backfill_spec.mode == "pk_range_fold"
        assert len(progress_events) >= 1
        _verify_external_db_mv(cur, p_alg.view_name, q_alg, f"{dialect.value}:mgr:alg_init")
        checks += 1

        # Read-only MV enforcement across INSERT / UPDATE / DELETE
        for bad_dml in (
            f"INSERT INTO {p_alg.view_name} VALUES ('Z', 1, 10, 10, 10);",
            f"UPDATE {p_alg.view_name} SET sv = 999 WHERE grp = 'A';",
            f"DELETE FROM {p_alg.view_name} WHERE grp = 'A';",
        ):
            ro_blocked = False
            try:
                cur.execute(bad_dml)
            except Exception:
                ro_blocked = True
            assert ro_blocked, f"Expected read-only MV {p_alg.view_name} on {dialect.value} to reject: {bad_dml}"
            checks += 1

        # 2. group_key_batch strategy
        p_dist = mgr.register_view(
            f"CREATE VIEW v_mgr_dist AS {q_dist};",
            incremental_backfill=True,
            batch_size=2,
            on_progress=lambda b, d: progress_events.append((b, d)),
        )
        assert p_dist.incremental_backfill_spec.mode == "group_key_batch"
        _verify_external_db_mv(cur, p_dist.view_name, q_dist, f"{dialect.value}:mgr:dist_init")
        checks += 1

        # 3. Tier 3 refresh + progress callback
        p_win = mgr.register_view(
            f"CREATE VIEW v_mgr_win AS {q_win};",
            incremental_backfill=True,
            batch_size=2,
            on_progress=lambda b, d: progress_events.append((b, d)),
        )
        _verify_external_db_mv(cur, p_win.view_name, q_win, f"{dialect.value}:mgr:win_init")
        checks += 1

        # 4. offset_batch strategy (composite PK base table)
        p_off = mgr.register_view(
            f"CREATE VIEW v_mgr_off AS {q_off};",
            incremental_backfill=True,
            batch_size=1,
            on_progress=lambda b, d: progress_events.append((b, d)),
        )
        assert p_off.incremental_backfill_spec.mode == "offset_batch"
        _verify_external_db_mv(cur, p_off.view_name, q_off, f"{dialect.value}:mgr:off_init")
        checks += 1

        # 5. sync_views: unchanged, updated, dropped
        q_alg_v2 = "SELECT grp, COUNT(*) AS cnt, SUM(val) AS sv FROM mgr_items WHERE val >= 15 GROUP BY grp"
        st = mgr.sync_views(
            [
                f"CREATE VIEW v_mgr_alg AS {q_alg_v2};",
                f"CREATE VIEW v_mgr_dist AS {q_dist};",
            ],
            prune_removed=True,
        )
        assert st["__matview_v_mgr_alg"] == "updated"
        assert st["__matview_v_mgr_dist"] == "unchanged"
        assert st["__matview_v_mgr_win"] == "dropped"
        assert st["__matview_v_mgr_off"] == "dropped"
        _verify_external_db_mv(cur, "__matview_v_mgr_alg", q_alg_v2, f"{dialect.value}:mgr:alg_updated")
        checks += 1

        # 6. bulk_changes context manager
        with mgr.bulk_changes():
            cur.execute("INSERT INTO mgr_items VALUES (6, 'A', 'z', 100), (7, 'D', 'w', 25);")
            cur.execute("UPDATE mgr_items SET val = 35 WHERE id = 2;")
            cur.execute("DELETE FROM mgr_items WHERE id = 3;")
        _verify_external_db_mv(cur, "__matview_v_mgr_alg", q_alg_v2, f"{dialect.value}:mgr:bulk_alg")
        _verify_external_db_mv(cur, "__matview_v_mgr_dist", q_dist, f"{dialect.value}:mgr:bulk_dist")
        checks += 2

        # 7. ALTER TABLE schema migration via apply_schema_ddl + trigger rebuild
        mgr.apply_schema_ddl("ALTER TABLE mgr_items ADD COLUMN note VARCHAR(32) DEFAULT 'ok';")
        cur.execute("INSERT INTO mgr_items (id, grp, tag, val, note) VALUES (8, 'D', 'q', 45, 'migrated');")
        if hasattr(conn, "commit"):
            conn.commit()
        _verify_external_db_mv(cur, "__matview_v_mgr_alg", q_alg_v2, f"{dialect.value}:mgr:alter_alg")
        _verify_external_db_mv(cur, "__matview_v_mgr_dist", q_dist, f"{dialect.value}:mgr:alter_dist")
        checks += 2

        # 8. RIGHT JOIN and FULL OUTER JOIN live execution & DML verification
        mgr.apply_schema_ddl(
            """
            CREATE TABLE mgr_join_a (id INT PRIMARY KEY, k INT, aval VARCHAR(32));
            CREATE TABLE mgr_join_b (id INT PRIMARY KEY, k INT, bval VARCHAR(32));
            """
        )
        cur.execute("INSERT INTO mgr_join_a VALUES (1, 10, 'a1'), (2, 20, 'a2');")
        cur.execute("INSERT INTO mgr_join_b VALUES (100, 20, 'b2'), (200, 30, 'b3');")
        if hasattr(conn, "commit"):
            conn.commit()

        p_right = mgr.register_view(
            "CREATE VIEW v_mgr_right AS SELECT a.id AS aid, a.aval, b.id AS bid, b.bval FROM mgr_join_a a RIGHT JOIN mgr_join_b b ON a.k = b.k;"
        )
        cur.execute(f"SELECT aid, aval, bid, bval FROM {p_right.view_name} ORDER BY bid;")
        assert list(cur.fetchall()) == [(2, "a2", 100, "b2"), (None, None, 200, "b3")]
        checks += 1

        p_full = mgr.register_view(
            "CREATE VIEW v_mgr_full AS SELECT a.k AS ak, a.aval, b.k AS bk, b.bval FROM mgr_join_a a FULL OUTER JOIN mgr_join_b b ON a.k = b.k;"
        )
        cur.execute("INSERT INTO mgr_join_a VALUES (3, 30, 'a3');")
        cur.execute("DELETE FROM mgr_join_b WHERE id = 100;")
        if hasattr(conn, "commit"):
            conn.commit()
        cur.execute(f"SELECT ak, aval, bk, bval FROM {p_full.view_name};")
        full_rows = sorted([tuple(r) for r in cur.fetchall()], key=lambda x: str(x))
        assert full_rows == sorted([(10, "a1", None, None), (20, "a2", None, None), (30, "a3", 30, "b3")], key=lambda x: str(x))
        checks += 1

        # Clean up manager views
        assert mgr.drop_view("v_mgr_alg") is True
        assert mgr.drop_view("v_mgr_dist") is True
        assert mgr.drop_view("v_mgr_right") is True
        assert mgr.drop_view("v_mgr_full") is True
        assert mgr.drop_view("v_mgr_nonexistent") is False

    cur.execute(f"DROP TABLE IF EXISTS __matview_catalog{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_items{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_comp_items{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_join_a{drop_suffix};")
    cur.execute(f"DROP TABLE IF EXISTS mgr_join_b{drop_suffix};")
    return checks


def _open_live_postgres_connection(dsn: str):
    try:
        import psycopg2 as pg_driver  # type: ignore
    except ImportError:
        try:
            import psycopg as pg_driver  # type: ignore
        except ImportError:
            if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
                raise RuntimeError("psycopg2 or psycopg must be installed in CI when POSTGRES_DSN is set")
            return None

    from urllib.parse import urlparse, urlunparse
    try:
        conn = pg_driver.connect(dsn)
    except Exception:
        parsed_pg = urlparse(dsn)
        target_db = (parsed_pg.path or "/matview_test").lstrip("/") or "matview_test"
        admin_dsn = urlunparse(parsed_pg._replace(path="/postgres"))
        admin_conn = pg_driver.connect(admin_dsn)
        admin_conn.autocommit = True
        admin_cur = admin_conn.cursor()
        admin_cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{target_db}'")
        if not admin_cur.fetchone():
            admin_cur.execute(f"CREATE DATABASE {target_db}")
        admin_conn.close()
        conn = pg_driver.connect(dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET synchronous_commit = OFF;")
    return conn


def _open_live_mysql_connection(dsn: str):
    try:
        import pymysql  # type: ignore
    except ImportError:
        if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            raise RuntimeError("pymysql must be installed in CI when MYSQL_DSN is set")
        return None, ""

    from urllib.parse import urlparse
    parsed = urlparse(dsn if "://" in dsn else f"mysql://{dsn}")
    target_db = (parsed.path or "/matview_test").lstrip("/") or "matview_test"
    conn = pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        autocommit=True,
    )
    with conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{target_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cur.execute(f"USE `{target_db}`;")
        cur.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cur.execute("SET SESSION collation_connection = @@collation_database;")
        cur.execute("SET SESSION sql_mode = CONCAT_WS(',', REPLACE(@@SESSION.sql_mode, 'ONLY_FULL_GROUP_BY', ''), 'PIPES_AS_CONCAT');")
    return conn, target_db


class _CompatCursorAdapter:
    def __init__(self, exec_fn, query_fn):
        self._exec = exec_fn
        self._query = query_fn
        self._last_rows = []
        self.rowcount = -1

    def execute(self, sql: str, params: Any = None) -> "_CompatCursorAdapter":
        s = sql.strip()
        if not s:
            self._last_rows = []
            return self
        s_up = s.upper()
        s_up_clean = s_up.rstrip(";").strip()
        if s_up_clean.startswith("DROP SCHEMA PUBLIC") or s_up_clean == "CREATE SCHEMA PUBLIC":
            if s_up_clean.startswith("DROP SCHEMA PUBLIC"):
                objs = self._query(
                    "SELECT type, name FROM sqlite_master WHERE type IN ('view', 'table') AND name NOT LIKE 'sqlite_%'",
                    None,
                )
                for o_type, o_name in (objs or []):
                    self._exec(f"DROP {str(o_type).upper()} IF EXISTS {o_name};", None)
            self._last_rows = []
            return self
        is_query = (
            s_up.startswith("SELECT")
            or s_up.startswith("PRAGMA")
            or (
                s_up.startswith("WITH")
                and not re.search(r"\)\s*(?:INSERT|UPDATE|DELETE|REPLACE)\b", s_up)
            )
        )
        if is_query:
            res = self._query(sql, params)
            self._last_rows = list(res) if res is not None else []
        else:
            self._exec(sql, params)
            self._last_rows = []
        return self

    def fetchall(self):
        return self._last_rows

    def fetchone(self):
        return self._last_rows[0] if self._last_rows else None


class _CompatConnectionAdapter:
    def __init__(self, cursor_factory, close_fn, commit_fn=None, rollback_fn=None):
        self._cursor_factory = cursor_factory
        self._close_fn = close_fn
        self._commit_fn = commit_fn
        self._rollback_fn = rollback_fn

    def cursor(self) -> _CompatCursorAdapter:
        return self._cursor_factory()

    def execute(self, sql: str, params: Any = None) -> _CompatCursorAdapter:
        c = self.cursor()
        c.execute(sql, params)
        return c

    def commit(self) -> None:
        if self._commit_fn:
            try:
                self._commit_fn()
            except Exception:
                pass

    def rollback(self) -> None:
        if self._rollback_fn:
            try:
                self._rollback_fn()
            except Exception:
                pass

    def close(self) -> None:
        if self._close_fn:
            self._close_fn()


def _open_sqlite_compat_connection(engine_spec: str, db_path: str) -> _CompatConnectionAdapter:
    eng = engine_spec.strip().lower()
    if eng in ("apsw", "sqlite3mc"):
        import apsw  # type: ignore
        raw_conn = apsw.Connection(db_path)
        init_cur = raw_conn.cursor()
        if eng == "sqlite3mc":
            try:
                init_cur.execute("PRAGMA cipher = 'chacha20';")
                init_cur.execute("PRAGMA key = 'matview-sqlite3mc-secret';")
            except Exception:
                pass
        init_cur.execute("PRAGMA synchronous = OFF;")
        init_cur.execute("PRAGMA journal_mode = MEMORY;")
        init_cur.execute("PRAGMA temp_store = MEMORY;")
        init_cur.execute("PRAGMA recursive_triggers = ON;")

        def _make_apsw_cursor() -> _CompatCursorAdapter:
            raw_cur = raw_conn.cursor()
            def _exec(s: str, p: Any = None) -> None:
                if p is not None:
                    raw_cur.execute(s, tuple(p))
                else:
                    raw_cur.execute(s)
            def _query(q: str, p: Any = None):
                if p is not None:
                    return [tuple(r) for r in raw_cur.execute(q, tuple(p))]
                return [tuple(r) for r in raw_cur.execute(q)]
            return _CompatCursorAdapter(_exec, _query)

        return _CompatConnectionAdapter(_make_apsw_cursor, raw_conn.close)

    elif eng == "sqlcipher":
        try:
            from sqlcipher3 import dbapi2 as sqlcipher_mod  # type: ignore
        except ImportError:
            import sqlcipher3 as sqlcipher_mod  # type: ignore
        raw_conn = sqlcipher_mod.connect(db_path, isolation_level=None)
        raw_conn.execute("PRAGMA key = 'matview-sqlcipher-secret';")
        raw_conn.execute("PRAGMA synchronous = OFF;")
        raw_conn.execute("PRAGMA journal_mode = MEMORY;")
        raw_conn.execute("PRAGMA temp_store = MEMORY;")
        raw_conn.execute("PRAGMA recursive_triggers = ON;")

        def _make_cipher_cursor() -> _CompatCursorAdapter:
            def _exec(s: str, p: Any = None) -> None:
                if not s.strip():
                    return
                if p is not None:
                    raw_conn.execute(s.strip(), tuple(p))
                else:
                    raw_conn.execute(s.strip())
            def _query(q: str, p: Any = None):
                if p is not None:
                    c = raw_conn.execute(q, tuple(p))
                else:
                    c = raw_conn.execute(q)
                return [tuple(r) for r in (c.fetchall() or [])]
            return _CompatCursorAdapter(_exec, _query)

        return _CompatConnectionAdapter(_make_cipher_cursor, raw_conn.close)

    elif eng == "sqlean":
        import sqlean  # type: ignore
        sqlean.extensions.enable_all()
        raw_conn = sqlean.connect(db_path, isolation_level=None)
        raw_conn.execute("PRAGMA synchronous = OFF;")
        raw_conn.execute("PRAGMA journal_mode = MEMORY;")
        raw_conn.execute("PRAGMA temp_store = MEMORY;")
        raw_conn.execute("PRAGMA recursive_triggers = ON;")

        def _make_sqlean_cursor() -> _CompatCursorAdapter:
            def _exec(s: str, p: Any = None) -> None:
                if not s.strip():
                    return
                if p is not None:
                    raw_conn.execute(s.strip(), tuple(p))
                else:
                    raw_conn.execute(s.strip())
            def _query(q: str, p: Any = None):
                if p is not None:
                    c = raw_conn.execute(q, tuple(p))
                else:
                    c = raw_conn.execute(q)
                return [tuple(r) for r in (c.fetchall() or [])]
            return _CompatCursorAdapter(_exec, _query)

        return _CompatConnectionAdapter(_make_sqlean_cursor, raw_conn.close)

    elif eng == "libsql":
        try:
            import libsql_experimental as libsql_mod  # type: ignore
        except ImportError:
            import libsql as libsql_mod  # type: ignore
        try:
            raw_conn = libsql_mod.connect(db_path, isolation_level=None)
        except TypeError:
            raw_conn = libsql_mod.connect(db_path)
        for pragma_sql in (
            "PRAGMA synchronous = OFF;",
            "PRAGMA journal_mode = MEMORY;",
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA recursive_triggers = ON;",
        ):
            try:
                raw_conn.execute(pragma_sql)
            except Exception:
                pass
        tx_state = {"in_tx": False}

        def _libsql_exec(s: str, p: Any = None) -> None:
            if not s.strip():
                return
            s_up = s.strip().upper().rstrip(";").strip()
            if s_up == "BEGIN" or s_up.startswith("BEGIN "):
                if hasattr(raw_conn, "commit"):
                    try:
                        raw_conn.commit()
                    except Exception:
                        pass
                tx_state["in_tx"] = True
                try:
                    raw_conn.execute(s.strip())
                except Exception as e:
                    if "within a transaction" not in str(e).lower():
                        raise
                return
            if s_up == "COMMIT":
                tx_state["in_tx"] = False
                try:
                    raw_conn.execute(s.strip())
                except Exception:
                    if hasattr(raw_conn, "commit"):
                        raw_conn.commit()
                return
            if s_up.startswith("ROLLBACK") and "TO" not in s_up:
                tx_state["in_tx"] = False
                try:
                    raw_conn.execute(s.strip())
                except Exception:
                    if hasattr(raw_conn, "rollback"):
                        raw_conn.rollback()
                return
            try:
                if p is not None:
                    raw_conn.execute(s.strip(), tuple(p))
                else:
                    raw_conn.execute(s.strip())
            except Exception:
                if not tx_state["in_tx"] and hasattr(raw_conn, "rollback"):
                    try:
                        raw_conn.rollback()
                    except Exception:
                        pass
                raise
            if not tx_state["in_tx"] and hasattr(raw_conn, "commit"):
                try:
                    raw_conn.commit()
                except Exception:
                    pass

        def _libsql_query(q: str, p: Any = None):
            if p is not None:
                c = raw_conn.execute(q, tuple(p))
            else:
                c = raw_conn.execute(q)
            return [tuple(r) for r in (c.fetchall() or [])]

        return _CompatConnectionAdapter(
            lambda: _CompatCursorAdapter(_libsql_exec, _libsql_query),
            lambda: raw_conn.close() if hasattr(raw_conn, "close") else None,
        )

    elif eng == "sqlite3":
        raw_conn = sqlite3.connect(db_path, isolation_level=None)
        raw_conn.execute("PRAGMA synchronous = OFF;")
        raw_conn.execute("PRAGMA journal_mode = MEMORY;")
        raw_conn.execute("PRAGMA temp_store = MEMORY;")
        raw_conn.execute("PRAGMA recursive_triggers = ON;")
        _register_sqlite_compat_udfs(raw_conn)

        def _make_sq3_cursor() -> _CompatCursorAdapter:
            def _exec(s: str, p: Any = None) -> None:
                st = s.strip()
                if not st:
                    return
                st_up = st.upper().rstrip(";").strip()
                if st_up.startswith("DROP SCHEMA PUBLIC") or st_up == "CREATE SCHEMA PUBLIC":
                    if st_up.startswith("DROP SCHEMA PUBLIC"):
                        objs = raw_conn.execute(
                            "SELECT type, name FROM sqlite_master WHERE type IN ('view', 'table') AND name NOT LIKE 'sqlite_%'"
                        ).fetchall()
                        for o_type, o_name in objs:
                            raw_conn.execute(f"DROP {o_type.upper()} IF EXISTS {o_name};")
                    return
                if p is not None:
                    raw_conn.execute(st, tuple(p))
                else:
                    raw_conn.execute(st)
            def _query(q: str, p: Any = None):
                c = raw_conn.execute(q, tuple(p)) if p is not None else raw_conn.execute(q)
                return [tuple(r) for r in (c.fetchall() or [])]
            return _CompatCursorAdapter(_exec, _query)

        return _CompatConnectionAdapter(_make_sq3_cursor, raw_conn.close)

    else:
        raise ValueError(f"Unknown SQLITE_COMPAT_ENGINE: {engine_spec}")


class TranspiledCursorAdapter:
    def __init__(self, conn_adapter: "TranspiledConnectionAdapter") -> None:
        self._ca = conn_adapter
        self._last_rows: List[Tuple[Any, ...]] = []
        self.rowcount: int = -1

    def execute(self, sql: str, params: Any = None) -> "TranspiledCursorAdapter":
        rows, rc = self._ca._execute_sql(sql, params)
        self._last_rows = rows
        self.rowcount = rc
        return self

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return self._last_rows

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        return self._last_rows[0] if self._last_rows else None


class TranspiledConnectionAdapter:
    """
    A full PEP-249 DB-API 2.0 connection adapter backed by in-memory SQLite that
    accepts native PostgreSQL or MySQL DDL, triggers, PL/pgSQL DO blocks, MySQL
    stored procedures, information_schema queries, and DML so that all test
    suites and MaterializedViewManager methods run through the exact same code
    path whether connected to a live external server or in-memory SQLite.
    """

    def __init__(
        self,
        dialect: Dialect = Dialect.POSTGRES,
        engine_version: Optional[EngineVersion | str] = None,
    ) -> None:
        if isinstance(engine_version, (EngineVersion, str)) and engine_version:
            self._sim_engine_version = parse_engine_version(engine_version, default_dialect=dialect)
            dialect = self._sim_engine_version.dialect
        else:
            self._sim_engine_version = parse_engine_version(dialect)
        assert dialect in (Dialect.POSTGRES, Dialect.MYSQL)
        self.dialect = dialect
        self.autocommit = True
        self._sq_conn: Optional[sqlite3.Connection] = None
        self._pg_functions: Dict[str, str] = {}
        self._pg_fn_triggers: Dict[str, List[str]] = {}
        self._mysql_procs: Dict[str, str] = {}
        self._table_pks: Dict[str, List[str]] = {}
        self._skip_triggers: int = 0
        self._reset_sqlite()

    def _reset_sqlite(self) -> None:
        if self._sq_conn is not None:
            try:
                self._sq_conn.close()
            except Exception:
                pass
        self._sq_conn = sqlite3.connect(":memory:", isolation_level=None)
        self._sq_conn.execute("PRAGMA recursive_triggers = ON;")
        _register_sqlite_compat_udfs(self._sq_conn, dialect=self.dialect)
        self._sq_conn.create_function("__mv_skip_triggers", 0, lambda: self._skip_triggers)
        self._pg_functions.clear()
        self._pg_fn_triggers.clear()
        self._mysql_procs.clear()
        self._table_pks.clear()
        self._skip_triggers = 0

    def cursor(self) -> TranspiledCursorAdapter:
        return TranspiledCursorAdapter(self)

    def execute(self, sql: str, params: Any = None) -> TranspiledCursorAdapter:
        c = self.cursor()
        c.execute(sql, params)
        return c

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        if self._sq_conn is not None:
            try:
                self._sq_conn.execute("ROLLBACK;")
            except Exception:
                pass

    def close(self) -> None:
        if self._sq_conn is not None:
            self._sq_conn.close()
            self._sq_conn = None

    def _record_table_pks(self, ddl: str) -> None:
        m_t = re.match(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_]+)\s*\((.*)\)\s*;?$",
            ddl.strip(),
            flags=re.I | re.S,
        )
        if not m_t:
            return
        t_name = m_t.group(1)
        body = m_t.group(2)
        m_pk = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", body, flags=re.I)
        if m_pk:
            self._table_pks[t_name.lower()] = [c.strip() for c in m_pk.group(1).split(",")]
            return
        for part in _split_top_level(body, ","):
            if re.search(r"\bPRIMARY\s+KEY\b", part, flags=re.I):
                col_n = part.strip().split()[0].strip('"`[]')
                self._table_pks[t_name.lower()] = [col_n]
                return

    def _convert_mysql_dml_to_sqlite(self, sql: str, default_pks: Optional[Sequence[str]] = None) -> str:
        s = _transpile_mysql_expr_to_sqlite(sql, default_pks or ("__mv_pk",))
        s = re.sub(r"\bCAST\s*\(([^)]+?)\s+AS\s+CHAR\s*\(\d+\)\)", r"CAST(\1 AS TEXT)", s, flags=re.I)
        s = re.sub(r"\bINSERT\s+IGNORE\s+INTO\b", "INSERT OR IGNORE INTO", s, flags=re.I)

        def _repl_odku(m_ins: re.Match[str]) -> str:
            tbl = m_ins.group(1)
            prefix_sql = re.split(r"\bON\s+DUPLICATE\s+KEY\s+UPDATE\b", m_ins.group(0), flags=re.I)[0].rstrip()
            assigns = m_ins.group(2)
            pks = self._table_pks.get(tbl.lower())
            if not pks:
                if "__matview_dist_" in tbl:
                    pks = [*(default_pks or ["__mv_pk"]), "__val"]
                elif "__mv_global_id" in prefix_sql:
                    pks = ["__mv_global_id"]
                else:
                    pks = list(default_pks or ["id"])
            return f"{prefix_sql}\n    ON CONFLICT ({', '.join(pks)}) DO UPDATE SET {assigns}"

        s = re.sub(
            r"INSERT\s+INTO\s+([A-Za-z0-9_]+)\s*[^;]+?\bON\s+DUPLICATE\s+KEY\s+UPDATE\s+([^;]+)",
            _repl_odku,
            s,
            flags=re.I | re.S,
        )

        def _repl_multi_upd(m_u: re.Match[str]) -> str:
            tbl, subq, on_c, set_cl = m_u.group(1), m_u.group(2), m_u.group(3), m_u.group(4)
            cleaned_assigns = [
                re.sub(rf"^\s*{re.escape(tbl)}\.", "", a_part.strip())
                for a_part in _split_top_level(set_cl, ",")
            ]
            return f"UPDATE {tbl}\n    SET {', '.join(cleaned_assigns)}\n    FROM ({subq}) AS __mv_delta\n    WHERE {on_c};"

        s = re.sub(
            r"UPDATE\s+([A-Za-z0-9_]+)\s+INNER\s+JOIN\s*\((.*?)\)\s*AS\s+__mv_delta\s+ON\s+([^\n]+?)\s+SET\s+([^;]+);?",
            _repl_multi_upd,
            s,
            flags=re.I | re.S,
        )
        return s

    def _exec_loop_backfill_in_sqlite(self, body: str) -> None:
        assert self._sq_conn is not None
        m_minmax = re.search(
            r"SELECT\s+MIN\(([A-Za-z0-9_]+)\),\s*MAX\([A-Za-z0-9_]+\)\s+INTO\s+__mv_min_pk,\s*__mv_max_pk\s+FROM\s+([A-Za-z0-9_]+);",
            body,
            flags=re.I,
        )
        m_bsize = re.search(r"__mv_batch_size\s+BIGINT\s*(?:DEFAULT|:=)\s*(\d+)\s*;", body, flags=re.I)
        batch_size = int(m_bsize.group(1)) if m_bsize else 2

        if m_minmax:
            pk_col, base_t = m_minmax.group(1), m_minmax.group(2)
            row = self._sq_conn.execute(f"SELECT MIN({pk_col}), MAX({pk_col}) FROM {base_t}").fetchone()
            if row and row[0] is not None and row[1] is not None:
                min_pk, max_pk = int(row[0]), int(row[1])
                cur_pk = min_pk - 1
                m_loop_ins = re.search(
                    r"(?:LOOP|DO)\s*SET\s+__mv_top_pk[^;]+;\s*(INSERT\s+INTO\s+.*?)\s*(?:SET\s+__mv_cur_pk|__mv_cur_pk\s*:=)",
                    body,
                    flags=re.I | re.S,
                )
                if not m_loop_ins:
                    m_loop_ins = re.search(
                        r"__mv_top_pk\s*:=\s*LEAST[^;]+;\s*(INSERT\s+INTO\s+.*?)\s*__mv_cur_pk\s*:=",
                        body,
                        flags=re.I | re.S,
                    )
                assert m_loop_ins is not None
                ins_tmpl = m_loop_ins.group(1).strip()
                while cur_pk < max_pk:
                    top_pk = min(max_pk, cur_pk + batch_size)
                    stmt = ins_tmpl.replace("__mv_cur_pk", str(cur_pk)).replace("__mv_top_pk", str(top_pk))
                    self._execute_sql(stmt)
                    cur_pk = top_pk
            m_tail = re.search(
                r"IF\s+NOT\s+EXISTS\s*\(\s*SELECT\s+1\s+FROM\s+([A-Za-z0-9_]+)\s*\)\s*THEN\s*(INSERT\s+INTO\s+.*?)\s*END\s+IF;",
                body,
                flags=re.I | re.S,
            )
            if m_tail:
                mv_t, tail_ins = m_tail.group(1), m_tail.group(2).strip()
                has_r = self._sq_conn.execute(f"SELECT 1 FROM {mv_t} LIMIT 1").fetchone()
                if not has_r:
                    self._execute_sql(tail_ins)
            return

        m_off_ins = re.search(
            r"(?:LOOP|REPEAT)\s*(INSERT\s+INTO\s+.*?)\s*(?:GET\s+DIAGNOSTICS|SET\s+__mv_rows)",
            body,
            flags=re.I | re.S,
        )
        assert m_off_ins is not None
        ins_tmpl = m_off_ins.group(1).strip()
        assert "ORDER BY" in ins_tmpl.upper(), f"Offset-paginated backfill query must include deterministic ORDER BY: {ins_tmpl}"
        offset = 0
        while True:
            stmt = ins_tmpl.replace("__mv_batch_size", str(batch_size)).replace("__mv_offset", str(offset))
            _, rc = self._execute_sql(stmt)
            if rc <= 0:
                break
            offset += batch_size if self.dialect == Dialect.MYSQL else rc
            if self.dialect == Dialect.POSTGRES and rc < batch_size:
                break

    def _execute_sql(self, sql: str, params: Any = None) -> Tuple[List[Tuple[Any, ...]], int]:
        assert self._sq_conn is not None
        s = sql.strip()
        if not s:
            return [], 0
        s_up = s.upper().rstrip(";").strip()

        if s_up == "SHOW SERVER_VERSION_NUM":
            if self.dialect == Dialect.POSTGRES:
                return [(str(self._sim_engine_version.pg_version_num),)], 1
            raise sqlite3.OperationalError("unrecognized configuration parameter server_version_num")
        if s_up == "SELECT VERSION()":
            if self.dialect == Dialect.POSTGRES:
                raw_v = self._sim_engine_version.raw_version or f"PostgreSQL {self._sim_engine_version.version_str}"
                if "postgres" not in raw_v.lower():
                    raw_v = f"PostgreSQL {self._sim_engine_version.version_str} ({raw_v})"
                return [(raw_v,)], 1
            else:
                raw_v = self._sim_engine_version.raw_version
                if self._sim_engine_version.is_mariadb and "mariadb" not in raw_v.lower():
                    raw_v = f"{self._sim_engine_version.version_str}-MariaDB"
                elif not raw_v:
                    raw_v = self._sim_engine_version.version_str
                return [(raw_v,)], 1
        if s_up == "SELECT SQLITE_VERSION()":
            raise sqlite3.OperationalError("function sqlite_version() does not exist")

        if s_up.startswith("DROP SCHEMA PUBLIC") or s_up.startswith("DROP DATABASE"):
            self._reset_sqlite()
            return [], 0
        if (
            s_up.startswith("CREATE DATABASE")
            or s_up.startswith("USE ")
            or s_up.startswith("SET ")
            or s_up.startswith("START TRANSACTION WITH CONSISTENT SNAPSHOT")
            or s_up == "CREATE SCHEMA PUBLIC"
        ):
            if "MATVIEW.SKIP_TRIGGERS" in s_up or "@__MATVIEW_SKIP_TRIGGERS" in s_up:
                self._skip_triggers = 1 if ("'ON'" in s_up or "= 1" in s_up or "=1" in s_up) else 0
            return [], 0

        if "INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS" in s_up:
            schemas = introspect_sqlite_schemas(self._sq_conn)
            rows = []
            for t_name, ts in schemas.items():
                del_rule = "NO ACTION"
                upd_rule = "NO ACTION"
                for act in ts.fk_cascade_actions:
                    if act.startswith("ON DELETE "):
                        del_rule = act[len("ON DELETE ") :]
                    elif act.startswith("ON UPDATE "):
                        upd_rule = act[len("ON UPDATE ") :]
                if del_rule != "NO ACTION" or upd_rule != "NO ACTION":
                    rows.append((t_name, del_rule, upd_rule))
            return rows, len(rows)
        if "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in s_up and "KEY_COLUMN_USAGE" in s_up:
            schemas = introspect_sqlite_schemas(self._sq_conn)
            rows = [
                (t_name, c_name)
                for t_name, ts in schemas.items()
                for c_name, cd in ts.columns.items()
                if cd.primary_key
            ]
            return rows, len(rows)
        if "INFORMATION_SCHEMA.COLUMNS" in s_up and "INFORMATION_SCHEMA.TABLES" in s_up:
            schemas = introspect_sqlite_schemas(self._sq_conn)
            rows = []
            for t_name, ts in schemas.items():
                if t_name.startswith("__matview_") or t_name.startswith("__mv_"):
                    continue
                for c_name, cd in ts.columns.items():
                    if self.dialect == Dialect.POSTGRES:
                        rows.append((t_name, c_name, cd.data_type, "NO" if cd.not_null else "YES", None))
                    else:
                        base_dt = cd.data_type.split("(")[0].strip()
                        rows.append(
                            (
                                t_name,
                                c_name,
                                base_dt,
                                "NO" if cd.not_null else "YES",
                                "PRI" if cd.primary_key else "",
                                cd.data_type,
                            )
                        )
            return rows, len(rows)

        if s_up.startswith("DO $$"):
            self._exec_loop_backfill_in_sqlite(s)
            return [], 0

        if s_up.startswith("DROP PROCEDURE"):
            return [], 0
        m_proc = re.match(
            r"CREATE\s+PROCEDURE\s+([A-Za-z0-9_]+)\s*\(\)\s*BEGIN\s*(.*)\s*END;?",
            s,
            flags=re.I | re.S,
        )
        if m_proc:
            proc_ident = m_proc.group(1)
            assert len(proc_ident) <= 64, f"MySQL procedure identifier exceeds 64-char limit ({len(proc_ident)} chars): {proc_ident}"
            self._mysql_procs[proc_ident] = m_proc.group(2).strip()
            return [], 0
        m_call = re.match(r"CALL\s+([A-Za-z0-9_]+)\s*\(\)\s*;?", s, flags=re.I)
        if m_call:
            proc_body = self._mysql_procs.get(m_call.group(1), "")
            if proc_body:
                self._exec_loop_backfill_in_sqlite(proc_body)
            return [], 0

        m_pg_fn = re.match(
            r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+([A-Za-z0-9_]+)\(\)\s+RETURNS\s+TRIGGER\s+AS\s+\$\$(.*)\$\$\s+LANGUAGE\s+plpgsql;?",
            s,
            flags=re.I | re.S,
        )
        if m_pg_fn:
            fn_ident = m_pg_fn.group(1)
            assert len(fn_ident) <= 63, f"PostgreSQL function identifier exceeds 63-char NAMEDATALEN limit ({len(fn_ident)} chars): {fn_ident}"
            self._pg_functions[fn_ident[:63]] = m_pg_fn.group(2).strip()
            return [], 0

        m_drop_fn = re.match(
            r"DROP\s+FUNCTION\s+IF\s+EXISTS\s+([A-Za-z0-9_]+)\s*(?:\(\))?\s*(?:CASCADE)?\s*;?",
            s,
            flags=re.I,
        )
        if m_drop_fn:
            fn_n = m_drop_fn.group(1)
            for tr_n in self._pg_fn_triggers.pop(fn_n, []):
                self._sq_conn.execute(f"DROP TRIGGER IF EXISTS {tr_n};")
            return [], 0

        if self.dialect == Dialect.POSTGRES and s_up.startswith("CREATE TRIGGER"):
            if "AFTER TRUNCATE ON" in s_up:
                return [], 0
            m_ro = re.match(
                r"CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\s+INSTEAD\s+OF\s+INSERT\s+OR\s+UPDATE\s+OR\s+DELETE\s+ON\s+([A-Za-z0-9_]+)\s+FOR\s+EACH\s+ROW\s+EXECUTE\s+FUNCTION\s+([A-Za-z0-9_]+)\(\);?",
                s,
                flags=re.I | re.S,
            )
            if m_ro:
                t_name, v_name, fn_name = m_ro.group(1), m_ro.group(2), m_ro.group(3)
                created = []
                for op, suf in (("INSERT", "ins"), ("UPDATE", "upd"), ("DELETE", "del")):
                    tn = f"{t_name}_{suf}"
                    self._sq_conn.execute(
                        f"CREATE TRIGGER {tn} INSTEAD OF {op} ON {v_name} BEGIN SELECT RAISE(ABORT, 'read-only'); END;"
                    )
                    created.append(tn)
                self._pg_fn_triggers.setdefault(fn_name, []).extend(created)
                return [], 0

            m_tr = re.match(
                r'CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\s+AFTER\s+INSERT\s+OR\s+UPDATE(?:\s+OF\s+[^;]+?)?\s+OR\s+DELETE\s+ON\s+([A-Za-z0-9_]+|"[^"]+"|`[^`]+`)\s+FOR\s+EACH\s+ROW\s+EXECUTE\s+FUNCTION\s+([A-Za-z0-9_]+)\(\);?',
                s,
                flags=re.I | re.S,
            )
            if m_tr:
                t_name, base_t, fn_name = m_tr.group(1), m_tr.group(2), m_tr.group(3)
                body = self._pg_functions.get(fn_name, "")
                m_parts = re.search(
                    r"IF\s*\(TG_OP\s*=\s*'DELETE'\)\s*THEN\s*(.*?)\s*ELSIF\s*\(TG_OP\s*=\s*'INSERT'\)\s*THEN\s*(.*?)\s*ELSIF\s*\(TG_OP\s*=\s*'UPDATE'\)\s*THEN\s*(.*?)\s*END\s+IF;\s*RETURN\s+NULL;",
                    body,
                    flags=re.I | re.S,
                )
                if m_parts:
                    del_b, ins_b, upd_b = (
                        m_parts.group(1).strip(),
                        m_parts.group(2).strip(),
                        m_parts.group(3).strip(),
                    )
                    created = []
                    for op, b_raw, suf in (("DELETE", del_b, "ad"), ("INSERT", ins_b, "ai"), ("UPDATE", upd_b, "au")):
                        b_clean = _transpile_pg_expr_to_sqlite(b_raw)
                        when_cl = ""
                        m_if = re.match(
                            r"^IF\s*\(([^\n]+?)\)\s*THEN\s*(.*)\s*END\s+IF;$",
                            b_clean.strip(),
                            flags=re.I | re.S,
                        )
                        if m_if:
                            inner_s = m_if.group(2).strip()
                            if m_if.group(1).strip().upper().startswith("NOT ") or "END IF;" not in inner_s.upper():
                                when_cl = f" WHEN ({m_if.group(1)})"
                                b_clean = inner_s

                        def _unwrap_if(m_inner: re.Match[str]) -> str:
                            c_in, st_in = m_inner.group(1).strip(), m_inner.group(2).strip()
                            out_sts = []
                            for st_p in _split_top_level(st_in, ";"):
                                sp = st_p.strip()
                                if not sp:
                                    continue
                                if re.match(r"^INSERT\s+INTO\b", sp, flags=re.I):
                                    if re.search(r"\bWHERE\s+(?:TRUE|1)\b", sp, flags=re.I):
                                        sp = re.sub(
                                            r"\bWHERE\s+(?:TRUE|1)\b",
                                            f"WHERE ({c_in})",
                                            sp,
                                            count=1,
                                            flags=re.I,
                                        )
                                    else:
                                        sp = re.sub(
                                            r"\n\s*WHERE\s+",
                                            f"\n    WHERE ({c_in}) AND ",
                                            sp,
                                            count=1,
                                            flags=re.I,
                                        )
                                elif re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", sp, flags=re.I):
                                    sp = re.sub(
                                        r"\n\s*WHERE\s+",
                                        f"\n    WHERE ({c_in}) AND ",
                                        sp,
                                        count=1,
                                        flags=re.I,
                                    )
                                out_sts.append(sp + ";")
                            return "\n".join(out_sts)

                        b_clean = re.sub(
                            r"IF\s*\(([^\n]+?)\)\s*THEN\s*(.*?)\s*END\s+IF;",
                            _unwrap_if,
                            b_clean,
                            flags=re.I | re.S,
                        )
                        m_outer_if = re.match(
                            r"^IF\s*\(([^\n]+?)\)\s*THEN\s*(.*)\s*END\s+IF;$",
                            b_clean.strip(),
                            flags=re.I | re.S,
                        )
                        if m_outer_if:
                            when_cl = f" WHEN ({m_outer_if.group(1)})"
                            b_clean = m_outer_if.group(2).strip()
                        if when_cl:
                            when_cl = f" WHEN ({when_cl[len(' WHEN '):]}) AND (__mv_skip_triggers() = 0)"
                        else:
                            when_cl = " WHEN (__mv_skip_triggers() = 0)"
                        tn = f"{t_name}_{suf}"
                        self._sq_conn.execute(
                            f"CREATE TRIGGER {tn} AFTER {op} ON {base_t} FOR EACH ROW{when_cl}\nBEGIN\n{b_clean}\nEND;"
                        )
                        created.append(tn)
                    self._pg_fn_triggers.setdefault(fn_name, []).extend(created)
                return [], 0

            m_stmt_tr = re.match(
                r"CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\s+AFTER\s+(INSERT|DELETE|UPDATE)\s+ON\s+([A-Za-z0-9_]+)\s+REFERENCING\s+[^\n]+\s+FOR\s+EACH\s+STATEMENT\s+EXECUTE\s+FUNCTION\s+([A-Za-z0-9_]+)\(\);?",
                s,
                flags=re.I | re.S,
            )
            if m_stmt_tr:
                t_name, op, base_t, fn_name = (
                    m_stmt_tr.group(1),
                    m_stmt_tr.group(2).upper(),
                    m_stmt_tr.group(3),
                    m_stmt_tr.group(4),
                )
                body = self._pg_functions.get(fn_name, "")
                loop_bodies = re.findall(
                    r"FOR\s+(?:NEW|OLD)\s+IN\s+SELECT\s+\*\s+FROM\s+__mv_(?:new|old)_rows\s+LOOP\s*(.*?)\s*END\s+LOOP;",
                    body,
                    flags=re.I | re.S,
                )
                b_clean = _transpile_pg_expr_to_sqlite("\n".join(loop_bodies))

                def _unwrap_if_tr(m_inner: re.Match[str]) -> str:
                    c_in, st_in = m_inner.group(1).strip(), m_inner.group(2).strip()
                    out_sts = []
                    for st_p in _split_top_level(st_in, ";"):
                        sp = st_p.strip()
                        if not sp:
                            continue
                        if re.match(r"^INSERT\s+INTO\b", sp, flags=re.I):
                            if re.search(r"\bWHERE\s+(?:TRUE|1)\b", sp, flags=re.I):
                                sp = re.sub(
                                    r"\bWHERE\s+(?:TRUE|1)\b",
                                    f"WHERE ({c_in})",
                                    sp,
                                    count=1,
                                    flags=re.I,
                                )
                            else:
                                sp = re.sub(
                                    r"\n\s*WHERE\s+",
                                    f"\n    WHERE ({c_in}) AND ",
                                    sp,
                                    count=1,
                                    flags=re.I,
                                )
                        elif re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", sp, flags=re.I):
                            sp = re.sub(
                                r"\n\s*WHERE\s+",
                                f"\n    WHERE ({c_in}) AND ",
                                sp,
                                count=1,
                                flags=re.I,
                            )
                        out_sts.append(sp + ";")
                    return "\n".join(out_sts)

                b_clean = re.sub(
                    r"IF\s*\(([^\n]+?)\)\s*THEN\s*(.*?)\s*END\s+IF;",
                    _unwrap_if_tr,
                    b_clean,
                    flags=re.I | re.S,
                )
                self._sq_conn.execute(
                    f"CREATE TRIGGER {t_name} AFTER {op} ON {base_t} FOR EACH ROW\nBEGIN\n{b_clean}\nEND;"
                )
                self._pg_fn_triggers.setdefault(fn_name, []).append(t_name)
                return [], 0

        if self.dialect == Dialect.MYSQL and s_up.startswith("CREATE TRIGGER"):
            m_tr = re.match(
                r'CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\s+AFTER\s+(INSERT|DELETE|UPDATE)\s+ON\s+([A-Za-z0-9_]+|"[^"]+"|`[^`]+`)\s+FOR\s+EACH\s+ROW\s+BEGIN\s*(.*)\s*END;?',
                s,
                flags=re.I | re.S,
            )
            if m_tr:
                t_name, op, base_t, body = (
                    m_tr.group(1),
                    m_tr.group(2).upper(),
                    m_tr.group(3),
                    m_tr.group(4).strip(),
                )
                m_skip_wrap = re.match(
                    r"^IF\s+COALESCE\(@__matview_skip_triggers,\s*0\)\s*=\s*0\s+THEN\s*(.*)\s*END\s+IF;$",
                    body,
                    flags=re.I | re.S,
                )
                if m_skip_wrap:
                    body = m_skip_wrap.group(1).strip()
                m_target = re.search(
                    r"(?:INSERT\s+INTO|DELETE\s+FROM|UPDATE)\s+(__matview_store_[A-Za-z0-9_]+)",
                    body,
                    flags=re.I,
                )
                main_pks = self._table_pks.get(m_target.group(1).lower(), ["__mv_pk"]) if m_target else ["__mv_pk"]
                b_clean = self._convert_mysql_dml_to_sqlite(body, main_pks)
                when_cl = ""
                m_outer = re.match(
                    r"^IF\s*\(([^\n]+?)\)\s*THEN\s*(.*)\s*END\s+IF;$",
                    b_clean.strip(),
                    flags=re.I | re.S,
                )
                if m_outer:
                    inner_my = m_outer.group(2).strip()
                    if m_outer.group(1).strip().upper().startswith("NOT ") or "END IF;" not in inner_my.upper():
                        when_cl = f" WHEN ({m_outer.group(1)})"
                        b_clean = inner_my

                def _unwrap_my_if(m_inner: re.Match[str]) -> str:
                    c_in, st_in = m_inner.group(1).strip(), m_inner.group(2).strip()
                    out_sts = []
                    for st_p in _split_top_level(st_in, ";"):
                        sp = st_p.strip()
                        if not sp:
                            continue
                        if re.match(r"^INSERT\s+INTO\b", sp, flags=re.I):
                            if re.search(r"\bWHERE\s+(?:TRUE|1)\b", sp, flags=re.I):
                                sp = re.sub(
                                    r"\bWHERE\s+(?:TRUE|1)\b",
                                    f"WHERE ({c_in})",
                                    sp,
                                    count=1,
                                    flags=re.I,
                                )
                            else:
                                sp = re.sub(
                                    r"\n\s*WHERE\s+",
                                    f"\n    WHERE ({c_in}) AND ",
                                    sp,
                                    count=1,
                                    flags=re.I,
                                )
                        elif re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", sp, flags=re.I):
                            sp = re.sub(
                                r"\n\s*WHERE\s+",
                                f"\n    WHERE ({c_in}) AND ",
                                sp,
                                count=1,
                                flags=re.I,
                            )
                        out_sts.append(sp + ";")
                    return "\n".join(out_sts)

                b_clean = re.sub(
                    r"IF\s*\(([^\n]+?)\)\s*THEN\s*(.*?)\s*END\s+IF;",
                    _unwrap_my_if,
                    b_clean,
                    flags=re.I | re.S,
                )
                if when_cl:
                    when_cl = f" WHEN ({when_cl[len(' WHEN '):]}) AND (__mv_skip_triggers() = 0)"
                else:
                    when_cl = " WHEN (__mv_skip_triggers() = 0)"
                q_base_t = base_t
                if q_base_t.startswith("`") and q_base_t.endswith("`"):
                    q_base_t = '"' + q_base_t[1:-1] + '"'
                self._sq_conn.execute(
                    f"CREATE TRIGGER {t_name} AFTER {op} ON {q_base_t} FOR EACH ROW{when_cl}\nBEGIN\n{b_clean}\nEND;"
                )
                return [], 0

        if self.dialect == Dialect.MYSQL and re.match(
            r"^CREATE\s+ALGORITHM\s*=\s*TEMPTABLE\s+VIEW\s+([A-Za-z0-9_]+)", s, flags=re.I
        ):
            m_v = re.match(r"^CREATE\s+ALGORITHM\s*=\s*TEMPTABLE\s+VIEW\s+([A-Za-z0-9_]+)", s, flags=re.I)
            assert m_v is not None
            v_name = m_v.group(1)
            s_v = self._convert_mysql_dml_to_sqlite(s)
            self._sq_conn.execute(s_v)
            for op, suf in (("INSERT", "ins"), ("UPDATE", "upd"), ("DELETE", "del")):
                self._sq_conn.execute(
                    f"CREATE TRIGGER __ro_my_{v_name}_{suf} INSTEAD OF {op} ON {v_name} BEGIN SELECT RAISE(ABORT, 'read-only'); END;"
                )
            return [], 0

        if s_up.startswith("CREATE TABLE"):
            self._record_table_pks(s)

        if self.dialect == Dialect.POSTGRES:
            m_trunc = re.match(r"^TRUNCATE\s+TABLE\s+([A-Za-z0-9_]+)\s*;?$", s, flags=re.I)
            if m_trunc:
                s_exec = f"DELETE FROM {m_trunc.group(1)};"
            else:
                s_exec = _transpile_pg_expr_to_sqlite(s)
                if "NULLS NOT DISTINCT" in s_exec.upper():
                    s_exec = re.sub(r"\s+NULLS\s+NOT\s+DISTINCT\b", "", s_exec, flags=re.I)
                    s_exec = re.sub(r"^CREATE\s+UNIQUE\s+INDEX\b", "CREATE INDEX", s_exec, flags=re.I)
                s_exec = re.sub(
                    r"\bDROP\s+(VIEW|TABLE)\s+IF\s+EXISTS\s+([A-Za-z0-9_]+)\s+CASCADE\s*;?",
                    r"DROP \1 IF EXISTS \2;",
                    s_exec,
                    flags=re.I,
                )
        else:
            if s_up == "START TRANSACTION":
                s_exec = "BEGIN;"
            else:
                s_exec = self._convert_mysql_dml_to_sqlite(s)
                if s_up.startswith("CREATE INDEX") or s_up.startswith("CREATE UNIQUE INDEX"):
                    s_exec = re.sub(r"\(\(\((.*?)\)\)\)", r"((\1))", s_exec)

        if params is not None:
            s_exec = s_exec.replace("%s", "?")
            cur = self._sq_conn.execute(s_exec, tuple(params))
        else:
            cur = self._sq_conn.execute(s_exec)
        rows = cur.fetchall() or []
        return rows, cur.rowcount


def _rewrite_live_reference_select_for_dialect(raw_select_sql: str, dialect: Dialect) -> str:
    if dialect == Dialect.SQLITE:
        if os.environ.get("SQLITE_COMPAT_ENGINE", "").strip().lower() not in ("", "sqlite3"):
            s_sq = raw_select_sql
            s_sq = re.sub(
                r"\bVAR_POP\s*\(\s*([^()]+?)\s*\)",
                lambda m: (
                    f"(CASE WHEN COUNT({m.group(1)}) < 1 THEN NULL "
                    f"WHEN ((1.0 * SUM(1.0 * ({m.group(1)}) * ({m.group(1)})) - ((1.0 * SUM(1.0 * ({m.group(1)}))) * (1.0 * SUM(1.0 * ({m.group(1)})))) / COUNT({m.group(1)})) / COUNT({m.group(1)})) < 0.0 THEN 0.0 "
                    f"ELSE ((1.0 * SUM(1.0 * ({m.group(1)}) * ({m.group(1)})) - ((1.0 * SUM(1.0 * ({m.group(1)}))) * (1.0 * SUM(1.0 * ({m.group(1)})))) / COUNT({m.group(1)})) / COUNT({m.group(1)})) END)"
                ),
                s_sq,
                flags=re.IGNORECASE,
            )
            s_sq = re.sub(
                r"\bBIT_AND\s*\(\s*([^()]+?)\s*\)",
                lambda m: "(" + " | ".join(f"MIN(({m.group(1)}) & {1 << b})" for b in range(32)) + ")",
                s_sq,
                flags=re.IGNORECASE,
            )
            s_sq = re.sub(
                r"\bBIT_OR\s*\(\s*([^()]+?)\s*\)",
                lambda m: "(" + " | ".join(f"MAX(({m.group(1)}) & {1 << b})" for b in range(32)) + ")",
                s_sq,
                flags=re.IGNORECASE,
            )
            s_sq = re.sub(
                r"\bBIT_XOR\s*\(\s*([^()]+?)\s*\)",
                lambda m: "(" + " | ".join(f"((SUM((({m.group(1)}) >> {b}) & 1) & 1) << {b})" for b in range(32)) + ")",
                s_sq,
                flags=re.IGNORECASE,
            )
            return s_sq
        return raw_select_sql
    s = re.sub(
        r"\bTOTAL\s*\(\s*DISTINCT\s+((?:[^()]|\([^()]*\))+)\)",
        r"COALESCE(1.0 * SUM(DISTINCT \1), 0.0)",
        raw_select_sql,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"\bTOTAL\s*\(\s*((?:[^()]|\([^()]*\))+)\)",
        r"COALESCE(1.0 * SUM(\1), 0.0)",
        s,
        flags=re.IGNORECASE,
    )
    if dialect == Dialect.MYSQL:
        s = re.sub(
            r"\b(LAG|LEAD)\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)\s+OVER\s*\(([^()]*)\)",
            r"COALESCE(\1(\2, \3) OVER (\5), \4)",
            s,
            flags=re.IGNORECASE,
        )
        m_foj = re.match(
            r'^SELECT\s+(.*?)\s+FROM\s+([A-Za-z0-9_."`\[\]]+(?:\s+(?:AS\s+)?[A-Za-z0-9_"`\[\]]+)?)\s+FULL\s+(?:OUTER\s+)?JOIN\s+([A-Za-z0-9_."`\[\]]+(?:\s+(?:AS\s+)?[A-Za-z0-9_"`\[\]]+)?)\s+ON\s+(([A-Za-z0-9_."`\[\]]+)\s*=\s*[A-Za-z0-9_."`\[\]]+)$',
            s.strip().rstrip(";"),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_foj:
            sel_p, t1_p, t2_p, on_p, lhs_col = (
                m_foj.group(1),
                m_foj.group(2),
                m_foj.group(3),
                m_foj.group(4),
                m_foj.group(5),
            )
            s = (
                f"SELECT {sel_p} FROM {t1_p} LEFT JOIN {t2_p} ON {on_p} "
                f"UNION ALL "
                f"SELECT {sel_p} FROM {t1_p} RIGHT JOIN {t2_p} ON {on_p} WHERE {lhs_col} IS NULL"
            )
    return s


def _adapt_suite_schema_for_dialect(schema_sql: str, dialect: Dialect) -> str:
    if dialect == Dialect.POSTGRES:
        return re.sub(r"\bREAL\b", "DOUBLE PRECISION", schema_sql)
    if dialect == Dialect.MYSQL:
        s = re.sub(r"\bTEXT\b", "VARCHAR(191)", schema_sql)
        s = re.sub(r"\bREAL\b", "DOUBLE PRECISION", s)
        return s
    return schema_sql


def _adapt_view_ddl_for_dialect(v_ddl: str, dialect: Dialect) -> str:
    if "WITH RECURSIVE tree(id, path, team, depth) AS (SELECT id, label, team, 1" in v_ddl:
        if dialect == Dialect.MYSQL:
            return v_ddl.replace(
                "SELECT id, label, team, 1",
                "SELECT id, CAST(label AS CHAR(191)), team, 1",
            )
        if dialect == Dialect.POSTGRES:
            return v_ddl.replace(
                "SELECT id, label, team, 1",
                "SELECT id, CAST(label AS TEXT), team, 1",
            )
    return v_ddl


def _adapt_mutation_for_dialect(m_sql: str, dialect: Dialect) -> str:
    if dialect == Dialect.MYSQL:
        s_up = m_sql.strip().upper().rstrip(";").strip()
        if s_up == "BEGIN":
            return "START TRANSACTION;"
        if "ON CONFLICT (id) DO UPDATE SET" in m_sql:
            if "INSERT INTO sales" in m_sql:
                return (
                    "INSERT INTO sales VALUES (1, 'NA', 'hw', 150.0, 4, 3.0, 1) "
                    "ON DUPLICATE KEY UPDATE region = VALUES(region), amount = sales.amount + VALUES(amount), qty = VALUES(qty);"
                )
            if "INSERT INTO invoices" in m_sql:
                return (
                    "INSERT INTO invoices VALUES (102, 20, 'cloud', 450.0, 6, 15.0, 1), (109, 30, 'sw', 220.0, 2, 0.0, 1) "
                    "ON DUPLICATE KEY UPDATE subtotal = invoices.subtotal + VALUES(subtotal), qty = VALUES(qty), discount = VALUES(discount), settled = VALUES(settled);"
                )
            if "INSERT INTO merchants" in m_sql:
                return (
                    "INSERT INTO merchants VALUES (20, 2, 'BetaLLC_EU', 'platinum', 1), (50, 3, 'DeltaJP', 'gold', 1) "
                    "ON DUPLICATE KEY UPDATE region_id = VALUES(region_id), mname = VALUES(mname), tier = VALUES(tier);"
                )
    return m_sql


def _run_unified_50_view_battery(
    conn: Any,
    cur: Any,
    dialect: Dialect,
    *,
    inc_mode: bool = False,
    use_trans: bool = False,
    label: str = "",
    reset_sql: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Single unified runner for the 50-view x 35-state verification battery across
    SQLite, PostgreSQL, and MySQL (used identically for both in-memory transpiled
    runs and live external database servers).
    """
    schema, view_specs = _get_live_engine_50_view_battery(dialect)
    if reset_sql:
        for r_st in [st.strip() for st in reset_sql.split(";") if st.strip()]:
            cur.execute(r_st + ";")
    for s in [st.strip() for st in schema.split(";") if st.strip()]:
        cur.execute(s + ";")

    t_val = "TRUE" if dialect == Dialect.POSTGRES else "1"
    f_val = "FALSE" if dialect == Dialect.POSTGRES else "0"
    cur.execute("INSERT INTO warehouses VALUES (1, 'WH_US', 'US'), (2, 'WH_EU', 'EU'), (3, 'WH_GLOBAL', NULL);")
    cur.execute("INSERT INTO customers VALUES (1, 1, 'Alice', 'NA'), (2, 2, 'Bob', 'VIP'), (3, 2, 'Cara', 'EU'), (4, NULL, 'Dan', 'APAC');")
    cur.execute(
        "INSERT INTO orders VALUES "
        f"(1, 1, 'NA', 'hw', 10.0, 2, 3, {t_val}, 'settled'),"
        f"(2, 1, 'NA', 'sw', 20.0, 4, 5, {f_val}, 'settled'),"
        f"(3, 2, 'EU', 'hw', 15.0, 1, 7, {t_val}, 'pending'),"
        f"(4, 2, 'EU', 'hw', 30.0, 3, 6, {t_val}, 'settled'),"
        f"(5, 3, NULL, 'sw', 50.0, 5, 1, {f_val}, 'settled');"
    )
    cur.execute("INSERT INTO order_tags VALUES (1, 'urgent', 2), (1, 'fragile', 5), (2, 'urgent', 3), (4, 'bulk', 10);")

    plans = []
    views_installed = 0
    checks_run = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EngineLimitationWarning)
        for v_idx, (v_name, v_sel) in enumerate(view_specs):
            schema_source: Any = conn if (v_idx % 2 == 1) else schema
            p = generate_materialized_view(
                schema_source,
                f"CREATE VIEW {v_name} AS {v_sel};",
                dialect=dialect,
                pg_use_transition_tables=use_trans,
                incremental_backfill=inc_mode,
                batch_size=2,
                ignore_limitations=True,
            )
            for stmt in p.all_statements():
                cur.execute(stmt)
            live_sel = _rewrite_live_reference_select_for_dialect(v_sel, dialect)
            plans.append((p, live_sel))
            views_installed += 1
            _verify_external_db_mv(cur, p.view_name, live_sel, f"{label}:inc={inc_mode}:tr={use_trans}:{v_name}:init")
            checks_run += 1

    ro_raised = False
    try:
        cur.execute(f"DELETE FROM {plans[0][0].view_name};")
    except Exception:
        ro_raised = True
    assert ro_raised, f"Expected {label} read-only view to block direct DELETE on MV"

    steps = _get_complex_dml_and_tx_steps(dialect)
    for step_lbl, m_sql in steps:
        cur.execute(m_sql)
        for p, live_sel in plans:
            _verify_external_db_mv(cur, p.view_name, live_sel, f"{label}:inc={inc_mode}:tr={use_trans}:{p.logical_view_name}:{step_lbl}")
            checks_run += 1

    for p, _ in reversed(plans):
        for drop_st in p.drop_statements():
            cur.execute(drop_st)

    return views_installed, checks_run


def run_live_postgres_tests_if_available(dsn: str) -> None:
    """Runs comprehensive 50-view x 3-mode (full, incremental, transition-tables) + manager tests against a real PostgreSQL instance."""
    conn = _open_live_postgres_connection(dsn)
    if conn is None:
        print("Skipping live PostgreSQL test: psycopg2/psycopg not installed.")
        return
    cur = conn.cursor()
    total_views = 0
    total_checks = 0
    reset_sql = "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
    for inc_mode, use_trans in ((False, False), (True, False), (False, True)):
        v_cnt, c_cnt = _run_unified_50_view_battery(
            conn, cur, Dialect.POSTGRES, inc_mode=inc_mode, use_trans=use_trans, label="live_pg", reset_sql=reset_sql
        )
        total_views += v_cnt
        total_checks += c_cnt
    total_checks += _run_live_manager_lifecycle_check(conn, cur, Dialect.POSTGRES)
    conn.close()
    print(f"LIVE PostgreSQL server test PASSED ({total_views} views installed, {total_checks} live assertions)!")


def run_live_mysql_tests_if_available(dsn: str) -> None:
    """Runs comprehensive 50-view x 2-mode (full, incremental) + manager tests against a real MySQL/MariaDB/Percona instance."""
    conn, target_db = _open_live_mysql_connection(dsn)
    if conn is None:
        print("Skipping live MySQL test: pymysql not installed.")
        return
    cur = conn.cursor()
    total_views = 0
    total_checks = 0
    reset_sql = (
        f"DROP DATABASE IF EXISTS `{target_db}`; "
        f"CREATE DATABASE `{target_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci; "
        f"USE `{target_db}`; "
        "SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci; "
        "SET SESSION collation_connection = @@collation_database;"
    )
    for inc_mode in (False, True):
        v_cnt, c_cnt = _run_unified_50_view_battery(
            conn, cur, Dialect.MYSQL, inc_mode=inc_mode, use_trans=False, label="live_mysql", reset_sql=reset_sql
        )
        total_views += v_cnt
        total_checks += c_cnt
    total_checks += _run_live_manager_lifecycle_check(conn, cur, Dialect.MYSQL)
    conn.close()
    print(f"LIVE MySQL server test PASSED ({total_views} views installed, {total_checks} live assertions)!")


def run_live_sqlite_compat_tests_if_available(engine_spec: str) -> None:
    """
    Runs comprehensive 50-view x 2-mode (full, incremental) + manager lifecycle live integration
    tests against SQLite-compatible database engines (`apsw`, `sqlite3mc`, `sqlcipher`, `sqlean`, `libsql`).
    """
    eng = engine_spec.strip().lower()
    total_views = 0
    total_checks = 0
    for inc_mode in (False, True):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, f"compat_{eng}_{int(inc_mode)}.db")
            conn = _open_sqlite_compat_connection(eng, db_file)
            cur = conn.cursor()
            try:
                v_cnt, c_cnt = _run_unified_50_view_battery(
                    conn, cur, Dialect.SQLITE, inc_mode=inc_mode, use_trans=False, label=f"live_{eng}"
                )
                total_views += v_cnt
                total_checks += c_cnt
                if inc_mode:
                    total_checks += _run_live_manager_lifecycle_check(conn, cur, Dialect.SQLITE)
            finally:
                conn.close()
    print(f"LIVE SQLite-compatible engine ({eng}) test PASSED ({total_views} views installed, {total_checks} live assertions)!")


import sys
import time
import unittest


class MVTestRunner:
    def __init__(self) -> None:
        self.total_views: int = 0
        self.total_assertions: int = 0
        self.pg_dsn: Optional[str] = os.environ.get("POSTGRES_DSN")
        self.mysql_dsn: Optional[str] = os.environ.get("MYSQL_DSN")
        self.sqlite_compat_engine: Optional[str] = os.environ.get("SQLITE_COMPAT_ENGINE")
        if self.pg_dsn:
            self.default_dialect = Dialect.POSTGRES
        elif self.mysql_dsn:
            self.default_dialect = Dialect.MYSQL
        else:
            self.default_dialect = Dialect.SQLITE

    def _open_backend_session(
        self, dialect: Dialect, suite_tag: str = "batch"
    ) -> Tuple[Any, Any, Optional[str], int, str, Callable[[], None]]:
        """
        Opens a unified `(conn, cur, reset_sql, chunk_size, engine_label, close_fn)` session
        for `dialect`. Every engine (live external PostgreSQL, live external MySQL/MariaDB,
        live SQLite-compatible engines, in-memory SQLite, and in-memory transpiled PostgreSQL/MySQL)
        exposes the exact same DB-API 2.0 connection/cursor interface so `run_batch` executes
        a single unified code path with zero dialect branching.
        """
        if dialect == Dialect.POSTGRES and self.pg_dsn:
            pg_conn = _open_live_postgres_connection(self.pg_dsn)
            if pg_conn is not None:
                return (
                    pg_conn,
                    pg_conn.cursor(),
                    "DROP SCHEMA public CASCADE; CREATE SCHEMA public;",
                    100,
                    "LIVE POSTGRES",
                    pg_conn.close,
                )
        if dialect == Dialect.MYSQL and self.mysql_dsn:
            my_conn, target_db = _open_live_mysql_connection(self.mysql_dsn)
            if my_conn is not None:
                reset_sql = (
                    f"DROP DATABASE IF EXISTS `{target_db}`; "
                    f"CREATE DATABASE `{target_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci; "
                    f"USE `{target_db}`; "
                    "SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci; "
                    "SET SESSION collation_connection = @@collation_database;"
                )
                return (
                    my_conn,
                    my_conn.cursor(),
                    reset_sql,
                    50,
                    "LIVE MYSQL",
                    my_conn.close,
                )
        if dialect == Dialect.SQLITE and self.sqlite_compat_engine:
            eng = self.sqlite_compat_engine.strip().lower()
            tmpdir = tempfile.TemporaryDirectory()
            db_file = os.path.join(tmpdir.name, f"{suite_tag}.db")
            compat_conn = _open_sqlite_compat_connection(eng, db_file)

            def _close_compat() -> None:
                try:
                    compat_conn.close()
                finally:
                    tmpdir.cleanup()

            return compat_conn, compat_conn.cursor(), None, 500, f"LIVE {eng.upper()}", _close_compat

        if dialect == Dialect.SQLITE:
            sq_conn = _open_sqlite_compat_connection("sqlite3", ":memory:")
            return sq_conn, sq_conn.cursor(), None, 500, "SQLITE", sq_conn.close
        else:
            tr_conn = TranspiledConnectionAdapter(dialect)
            return (
                tr_conn,
                tr_conn.cursor(),
                "DROP SCHEMA public CASCADE;",
                500,
                f"TRANSPILED {dialect.value.upper()}",
                tr_conn.close,
            )

    def run_batch(
        self,
        suite_name: str,
        schema_sql: str,
        views: Sequence[Tuple[str, str]],
        seed_sqls: Sequence[str],
        mutations: Sequence[Tuple[str, str]],
        *,
        dialect: Optional[Dialect] = None,
        pg_use_transition_tables: bool = False,
        incremental_backfill: bool = False,
        batch_size: int = 1000,
    ) -> None:
        eff_dialect = dialect if dialect is not None else self.default_dialect
        eff_schema = _adapt_suite_schema_for_dialect(schema_sql, eff_dialect)
        eff_muts = [
            (lbl, _adapt_mutation_for_dialect(msql, eff_dialect))
            for lbl, msql in mutations
        ]
        conn, cur, reset_sql, chunk_size, engine_label, close_fn = self._open_backend_session(
            eff_dialect, suite_tag=f"suite_{abs(hash(suite_name))}"
        )
        try:
            for i in range(0, len(views), chunk_size):
                chunk = views[i : i + chunk_size]
                if reset_sql:
                    for r_st in [s.strip() for s in reset_sql.split(";") if s.strip()]:
                        cur.execute(r_st + ";")
                for st in [s.strip() for s in eff_schema.split(";") if s.strip()]:
                    cur.execute(st + ";")
                for s in seed_sqls:
                    cur.execute(s)
                plans_and_selects: List[Tuple[MaterializedViewPlan, str]] = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    for v_idx, (v_name, v_ddl) in enumerate(chunk):
                        eff_v_ddl = _adapt_view_ddl_for_dialect(v_ddl, eff_dialect)
                        schema_source: Any = conn if (v_idx % 2 == 1) else eff_schema
                        plan = generate_materialized_view(
                            schema_source,
                            eff_v_ddl,
                            dialect=eff_dialect,
                            pg_use_transition_tables=pg_use_transition_tables,
                            incremental_backfill=incremental_backfill,
                            batch_size=batch_size,
                            ignore_limitations=True,
                        )
                        for stmt in plan.all_statements():
                            cur.execute(stmt)
                        v_spec = parse_view_definition(eff_v_ddl)
                        live_sel = _rewrite_live_reference_select_for_dialect(
                            v_spec.raw_select_sql, eff_dialect
                        )
                        plans_and_selects.append((plan, live_sel))
                        self.total_views += 1
                        _verify_external_db_mv(
                            cur, plan.view_name, live_sel, f"{suite_name}:{engine_label}:{v_name}:init"
                        )
                        self.total_assertions += 1
                for step_lbl, m_sql in eff_muts:
                    cur.execute(m_sql)
                    for plan, live_sel in plans_and_selects:
                        _verify_external_db_mv(
                            cur,
                            plan.view_name,
                            live_sel,
                            f"{suite_name}:{engine_label}:{plan.logical_view_name}:{step_lbl}",
                        )
                        self.total_assertions += 1
        finally:
            close_fn()
        print(
            f"  [PASS] {suite_name} [{engine_label}]: {len(views)} views x {len(mutations) + 1} states = {len(views) * (len(mutations) + 1)} assertions"
        )


# ==============================================================================
# SUITE 1: 500 Single-Table Algebraic Aggregate Combinations
# ==============================================================================
def run_suite_1_single_table_algebraic(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY,
        region TEXT,
        category TEXT,
        amount REAL,
        qty INTEGER,
        discount REAL,
        active INTEGER
    );
    """
    seed = [
        "INSERT INTO sales VALUES (1, 'NA', 'hw', 100.0, 2, 5.0, 1);",
        "INSERT INTO sales VALUES (2, 'NA', 'sw', 200.0, 1, NULL, 1);",
        "INSERT INTO sales VALUES (3, 'EU', 'hw', 50.0, 4, 2.5, 0);",
        "INSERT INTO sales VALUES (4, NULL, 'hw', 75.0, 3, 0.0, 1);",
        "INSERT INTO sales VALUES (5, 'EU', NULL, 120.0, 2, 10.0, 1);",
    ]
    mutations = [
        ("ins_batch", "INSERT INTO sales VALUES (6, 'NA', 'hw', 300.0, 5, 15.0, 1), (7, NULL, NULL, 40.0, 1, NULL, 0);"),
        ("upd_val", "UPDATE sales SET amount = amount + 25.0, qty = qty + 1 WHERE region = 'NA';"),
        ("upd_key", "UPDATE sales SET region = 'APAC', category = 'sw' WHERE id = 3;"),
        ("upd_null_to_val", "UPDATE sales SET region = 'NA', discount = 7.5 WHERE id = 4;"),
        ("del_subset", "DELETE FROM sales WHERE id IN (1, 5);"),
        ("del_group", "DELETE FROM sales WHERE region = 'APAC';"),
        ("reinsert", "INSERT INTO sales VALUES (1, 'EU', 'sw', 90.0, 3, 1.0, 1);"),
        ("tx_begin", "BEGIN;"),
        ("tx_upsert", "INSERT INTO sales VALUES (1, 'NA', 'hw', 150.0, 4, 3.0, 1) ON CONFLICT (id) DO UPDATE SET region = excluded.region, amount = sales.amount + excluded.amount, qty = excluded.qty;"),
        ("tx_sp", "SAVEPOINT sp_s1;"),
        ("tx_corr_upd", "UPDATE sales SET amount = amount * 1.5, discount = COALESCE(discount, 0.0) + 2.0 WHERE qty >= (SELECT AVG(qty) FROM (SELECT qty FROM sales WHERE active = 1) AS __sq_avg);"),
        ("tx_rb_sp", "ROLLBACK TO SAVEPOINT sp_s1;"),
        ("tx_del_subq", "DELETE FROM sales WHERE id IN (SELECT id FROM (SELECT id FROM sales WHERE active = 0) AS __sq_del);"),
        ("tx_commit", "COMMIT;"),
    ]

    group_patterns = [
        ("region", "region"),
        ("category", "category"),
        ("region, category", "region, category"),
        ("active", "active"),
        ("COALESCE(region, 'UNK') AS reg_clean", "COALESCE(region, 'UNK')"),
        ("CASE WHEN amount >= 100 THEN 'HIGH' ELSE 'LOW' END AS tier", "CASE WHEN amount >= 100 THEN 'HIGH' ELSE 'LOW' END"),
        ("region, active", "region, active"),
        ("category, CASE WHEN discount IS NULL THEN 0 ELSE 1 END AS has_disc", "category, CASE WHEN discount IS NULL THEN 0 ELSE 1 END"),
        ("qty % 2 AS parity", "qty % 2"),
        ("COALESCE(category, 'NONE') AS cat_clean, active", "COALESCE(category, 'NONE'), active"),
    ]

    where_patterns = [
        "",
        "WHERE active = 1",
        "WHERE amount > 60.0",
        "WHERE region IS NOT NULL",
        "WHERE discount IS NULL OR discount > 2.0",
        "WHERE qty >= 2 AND active = 1",
        "WHERE category IN ('hw', 'sw')",
        "WHERE amount BETWEEN 50.0 AND 250.0",
        "WHERE COALESCE(discount, 0.0) <= 10.0",
        "WHERE (region = 'NA' OR region = 'EU') AND qty > 0",
    ]

    agg_bundles = [
        "COUNT(*) AS c_all, SUM(amount) AS s_amt, AVG(amount) AS a_amt",
        "COUNT(discount) AS c_disc, TOTAL(amount) AS t_amt, SUM(qty) AS s_qty",
        "COUNT(*) AS c_all, SUM(amount * qty) AS gross, AVG(qty) AS a_qty",
        "SUM(CASE WHEN active = 1 THEN amount ELSE 0 END) AS active_amt, COUNT(region) AS c_reg",
        "TOTAL(discount) AS t_disc, AVG(discount) AS a_disc, COUNT(*) AS n",
    ]

    views: List[Tuple[str, str]] = []
    idx = 0
    for g_proj, g_by in group_patterns:
        for w_clause in where_patterns:
            for a_bundle in agg_bundles:
                idx += 1
                v_name = f"s1_alg_{idx:03d}"
                w_sql = f" {w_clause}" if w_clause else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT {g_proj}, {a_bundle} FROM sales{w_sql} GROUP BY {g_by};"
                views.append((v_name, v_sql))

    runner.run_batch("Suite 1 (Single-Table Algebraic Aggregates)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 2: 200 Holistic Aggregate & HAVING Combinations (MIN, MAX, HAVING)
# ==============================================================================
def run_suite_2_holistic_and_having(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE bids (
        id INTEGER PRIMARY KEY,
        auction_id INTEGER,
        bidder TEXT,
        price REAL,
        score INTEGER,
        valid INTEGER
    );
    """
    seed = [
        "INSERT INTO bids VALUES (1, 10, 'alice', 50.0, 8, 1);",
        "INSERT INTO bids VALUES (2, 10, 'bob', 75.0, 9, 1);",
        "INSERT INTO bids VALUES (3, 10, 'cara', 60.0, 7, 0);",
        "INSERT INTO bids VALUES (4, 20, 'alice', 120.0, 10, 1);",
        "INSERT INTO bids VALUES (5, 20, 'dan', 120.0, 6, 1);",
        "INSERT INTO bids VALUES (6, NULL, 'eve', 30.0, 5, 1);",
    ]
    mutations = [
        # Delete the current MAX in auction 10 (bob=75.0) -> MAX must fall back to 60.0 or 50.0!
        ("del_max_holder", "DELETE FROM bids WHERE id = 2;"),
        # Delete one of tied MAX holders in auction 20 (id=4, price=120) -> MAX stays 120.0!
        ("del_tied_max", "DELETE FROM bids WHERE id = 4;"),
        # Delete remaining MAX holder in auction 20 -> group disappears or drops!
        ("upd_lower_max", "UPDATE bids SET price = 40.0, score = 3 WHERE id = 5;"),
        ("ins_new_extrema", "INSERT INTO bids VALUES (7, 10, 'frank', 200.0, 1, 1), (8, 30, 'grace', 15.0, 4, 1);"),
        ("upd_group_move", "UPDATE bids SET auction_id = 30, price = 10.0 WHERE id = 1;"),
        ("del_all_a10", "DELETE FROM bids WHERE auction_id = 10;"),
        ("tx_begin", "BEGIN;"),
        ("tx_ins_extrema", "INSERT INTO bids VALUES (9, 20, 'heidi', 500.0, 10, 1), (10, 20, 'ivan', 5.0, 1, 1);"),
        ("tx_sp", "SAVEPOINT sp_bids;"),
        ("tx_del_extrema", "DELETE FROM bids WHERE id IN (9, 10);"),
        ("tx_rb_sp", "ROLLBACK TO SAVEPOINT sp_bids;"),
        ("tx_upd_extrema", "UPDATE bids SET price = 250.0 WHERE id = 9;"),
        ("tx_commit", "COMMIT;"),
    ]

    group_opts = [
        ("auction_id", "auction_id"),
        ("bidder", "bidder"),
        ("valid", "valid"),
        ("auction_id, valid", "auction_id, valid"),
        ("COALESCE(auction_id, -1) AS aid", "COALESCE(auction_id, -1)"),
    ]
    where_opts = [
        "",
        "WHERE valid = 1",
        "WHERE price >= 30.0",
        "WHERE auction_id IS NOT NULL",
        "WHERE score >= 5",
    ]
    agg_having_opts = [
        ("COUNT(*) AS cnt, MIN(price) AS min_p, MAX(price) AS max_p, SUM(price) AS sum_p", ""),
        ("COUNT(*) AS cnt, MIN(score) AS min_s, MAX(score) AS max_s, AVG(price) AS avg_p", ""),
        ("COUNT(*) AS cnt, MIN(price) AS min_p, MAX(price) AS max_p", "HAVING COUNT(*) >= 2"),
        ("COUNT(*) AS cnt, SUM(price) AS sum_p, MAX(price) AS max_p", "HAVING SUM(price) > 60.0"),
        ("COUNT(*) AS cnt, MIN(price) AS min_p, MAX(score) AS max_s", "HAVING MAX(score) >= 7"),
        ("SUM(score) AS sum_s, AVG(score) AS avg_s, COUNT(*) AS cnt", "HAVING COUNT(*) >= 1 AND SUM(score) >= 6"),
        ("MIN(price) AS min_p, MAX(price) AS max_p, TOTAL(price) AS tot_p", "HAVING MIN(price) <= 100.0"),
        ("COUNT(auction_id) AS c_aid, MIN(score) AS min_s, MAX(price) AS max_p", ""),
    ]

    views: List[Tuple[str, str]] = []
    idx = 0
    for g_proj, g_by in group_opts:
        for w_clause in where_opts:
            for a_proj, h_clause in agg_having_opts:
                idx += 1
                v_name = f"s2_hol_{idx:03d}"
                w_sql = f" {w_clause}" if w_clause else ""
                h_sql = f" {h_clause}" if h_clause else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT {g_proj}, {a_proj} FROM bids{w_sql} GROUP BY {g_by}{h_sql};"
                views.append((v_name, v_sql))

    runner.run_batch("Suite 2 (Holistic MIN/MAX & HAVING Clauses)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 3: 150 DISTINCT Aggregate Combinations (COUNT/SUM/AVG/TOTAL DISTINCT)
# ==============================================================================
def run_suite_3_distinct_aggregates(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE events (
        id INTEGER PRIMARY KEY,
        tenant TEXT,
        user_id INTEGER,
        event_type TEXT,
        latency INTEGER,
        ok INTEGER
    );
    """
    seed = [
        "INSERT INTO events VALUES (1, 't1', 101, 'click', 10, 1);",
        "INSERT INTO events VALUES (2, 't1', 101, 'click', 10, 1);",
        "INSERT INTO events VALUES (3, 't1', 102, 'view', 20, 1);",
        "INSERT INTO events VALUES (4, 't2', 201, 'click', 15, 0);",
        "INSERT INTO events VALUES (5, 't2', NULL, 'purchase', 30, 1);",
        "INSERT INTO events VALUES (6, NULL, 301, 'click', 10, 1);",
    ]
    mutations = [
        # Delete one duplicate of (t1, 101, click, 10) -> distinct counts/sums must NOT change!
        ("del_dup_1", "DELETE FROM events WHERE id = 1;"),
        # Delete the second copy of (t1, 101, click, 10) -> distinct counts/sums MUST decrement!
        ("del_dup_2", "DELETE FROM events WHERE id = 2;"),
        ("ins_dups", "INSERT INTO events VALUES (7, 't1', 102, 'view', 20, 1), (8, 't1', 103, 'checkout', 50, 1);"),
        ("upd_val", "UPDATE events SET user_id = 102, latency = 20 WHERE id = 8;"),
        ("upd_tenant", "UPDATE events SET tenant = 't1' WHERE id = 4;"),
        ("del_tenant", "DELETE FROM events WHERE tenant = 't2';"),
        ("tx_begin", "BEGIN;"),
        ("tx_ins_dup", "INSERT INTO events VALUES (9, 't1', 102, 'view', 20, 1), (10, 't1', 999, 'new_ev', 77, 1);"),
        ("tx_sp", "SAVEPOINT sp_dist;"),
        ("tx_del_all_t1", "DELETE FROM events WHERE tenant = 't1';"),
        ("tx_rb_sp", "ROLLBACK TO SAVEPOINT sp_dist;"),
        ("tx_del_single_dup", "DELETE FROM events WHERE id = 9;"),
        ("tx_commit", "COMMIT;"),
    ]

    group_opts = [
        ("tenant", "tenant"),
        ("event_type", "event_type"),
        ("ok", "ok"),
        ("tenant, ok", "tenant, ok"),
        ("COALESCE(tenant, 'anon') AS t_clean", "COALESCE(tenant, 'anon')"),
    ]
    where_opts = [
        "",
        "WHERE ok = 1",
        "WHERE latency >= 15",
        "WHERE tenant IS NOT NULL",
        "WHERE user_id IS NOT NULL",
    ]
    distinct_bundles = [
        "COUNT(DISTINCT user_id) AS du, COUNT(*) AS total_ev",
        "COUNT(DISTINCT event_type) AS det, SUM(DISTINCT latency) AS sdl",
        "AVG(DISTINCT latency) AS adl, TOTAL(DISTINCT latency) AS tdl, COUNT(*) AS cnt",
        "COUNT(DISTINCT user_id) AS du, COUNT(DISTINCT event_type) AS det, SUM(latency) AS sl",
        "SUM(DISTINCT user_id) AS sdu, AVG(latency) AS al, COUNT(user_id) AS cu",
        "COUNT(DISTINCT latency) AS dl, MIN(latency) AS min_l, MAX(latency) AS max_l",
    ]

    views: List[Tuple[str, str]] = []
    idx = 0
    for g_proj, g_by in group_opts:
        for w_clause in where_opts:
            for d_bundle in distinct_bundles:
                idx += 1
                v_name = f"s3_dist_{idx:03d}"
                w_sql = f" {w_clause}" if w_clause else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT {g_proj}, {d_bundle} FROM events{w_sql} GROUP BY {g_by};"
                views.append((v_name, v_sql))

    runner.run_batch("Suite 3 (DISTINCT Aggregates & Reference Counting)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 4: 50 Ungrouped Global Aggregate Combinations (Zero-Row vs Non-Empty)
# ==============================================================================
def run_suite_4_ungrouped_aggregates(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE metrics (
        id INTEGER PRIMARY KEY,
        host TEXT,
        cpu REAL,
        mem INTEGER,
        err INTEGER
    );
    """
    seed = [
        "INSERT INTO metrics VALUES (1, 'h1', 25.0, 100, 0);",
        "INSERT INTO metrics VALUES (2, 'h1', 75.0, 200, 1);",
        "INSERT INTO metrics VALUES (3, 'h2', 50.0, NULL, 0);",
    ]
    mutations = [
        ("ins_row", "INSERT INTO metrics VALUES (4, 'h3', 90.0, 400, 2);"),
        ("upd_row", "UPDATE metrics SET cpu = 10.0, err = 0 WHERE id = 2;"),
        # Delete ALL rows -> ungrouped aggregate must still return exactly 1 row (with COUNT=0, SUM=NULL, TOTAL=0.0)!
        ("del_all", "DELETE FROM metrics;"),
        # Re-insert after empty table
        ("ins_after_empty", "INSERT INTO metrics VALUES (10, 'h9', 42.5, 512, 3);"),
        ("del_again", "DELETE FROM metrics WHERE id = 10;"),
    ]

    where_opts = [
        "",
        "WHERE err > 0",
        "WHERE cpu >= 50.0",
        "WHERE mem IS NOT NULL",
        "WHERE host = 'h1'",
    ]
    agg_opts = [
        "COUNT(*) AS c_all, SUM(cpu) AS s_cpu, AVG(cpu) AS a_cpu, TOTAL(cpu) AS t_cpu",
        "COUNT(mem) AS c_mem, SUM(mem) AS s_mem, MIN(cpu) AS min_cpu, MAX(cpu) AS max_cpu",
        "COUNT(DISTINCT host) AS d_host, SUM(DISTINCT err) AS sd_err, COUNT(*) AS n",
        "AVG(DISTINCT cpu) AS ad_cpu, TOTAL(err) AS t_err, MAX(mem) AS max_mem",
        "COUNT(*) AS c_all, SUM(cpu * err) AS weighted_err",
        "MIN(err) AS min_err, MAX(err) AS max_err, COUNT(DISTINCT mem) AS d_mem",
        "SUM(CASE WHEN err > 0 THEN 1 ELSE 0 END) AS err_rows, COUNT(*) AS total_rows",
        "TOTAL(DISTINCT mem) AS td_mem, AVG(mem) AS a_mem",
        "COUNT(*) AS c, SUM(cpu) AS s, MIN(id) AS min_id, MAX(id) AS max_id",
        "COUNT(host) AS ch, SUM(err) AS se, AVG(err) AS ae",
    ]

    views: List[Tuple[str, str]] = []
    idx = 0
    for w_clause in where_opts:
        for a_proj in agg_opts:
            idx += 1
            v_name = f"s4_uagg_{idx:03d}"
            w_sql = f" {w_clause}" if w_clause else ""
            v_sql = f"CREATE VIEW {v_name} AS SELECT {a_proj} FROM metrics{w_sql};"
            views.append((v_name, v_sql))

    runner.run_batch("Suite 4 (Ungrouped Global Aggregates & Empty Table Transitions)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 5: 350 Multi-Table Join Combinations (INNER, LEFT, 3-Table, Self-Join)
# ==============================================================================
def run_suite_5_joins(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE depts (
        id INTEGER PRIMARY KEY,
        dname TEXT,
        budget REAL
    );
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        dept_id INTEGER,
        mgr_id INTEGER,
        uname TEXT,
        tier TEXT,
        active INTEGER
    );
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        item TEXT,
        total REAL,
        status TEXT
    );
    """
    seed = [
        "INSERT INTO depts VALUES (10, 'Eng', 1000.0), (20, 'Sales', 500.0), (30, 'EmptyDept', 200.0);",
        "INSERT INTO users VALUES (1, 10, NULL, 'Alice', 'gold', 1), (2, 10, 1, 'Bob', 'silver', 1), (3, 20, 1, 'Cara', 'gold', 0), (4, 99, 2, 'OrphanDeptUser', 'bronze', 1), (5, NULL, NULL, 'NoDeptUser', 'silver', 1);",
        "INSERT INTO orders VALUES (100, 1, 'laptop', 1200.0, 'paid'), (101, 1, 'mouse', 50.0, 'paid'), (102, 2, 'monitor', 300.0, 'pending'), (103, 4, 'keyboard', 80.0, 'paid'), (104, 999, 'ghost', 999.0, 'paid');",
    ]
    mutations = [
        # Insert department 99 so user 4 is no longer an orphan in depts!
        ("ins_missing_dept", "INSERT INTO depts VALUES (99, 'Ops', 750.0);"),
        # Insert order for user 3 (who previously had 0 orders -> tests LEFT JOIN NULL-to-matched transition!)
        ("ins_order_for_u3", "INSERT INTO orders VALUES (105, 3, 'desk', 450.0, 'paid');"),
        # Delete all orders for user 2 -> tests LEFT JOIN matched-to-NULL transition!
        ("del_orders_for_u2", "DELETE FROM orders WHERE user_id = 2;"),
        # Update user's dept_id and tier
        ("upd_user_fk", "UPDATE users SET dept_id = 20, tier = 'platinum', active = 1 WHERE id = 2;"),
        # Update dept name
        ("upd_dept", "UPDATE depts SET dname = 'Engineering_Core', budget = 1500.0 WHERE id = 10;"),
        # Delete department 20 -> users in dept 20 lose their right-side match in LEFT JOIN!
        ("del_dept_20", "DELETE FROM depts WHERE id = 20;"),
        # Self-join manager mutation: update mgr_id
        ("upd_mgr", "UPDATE users SET mgr_id = 3 WHERE id = 5;"),
        # Delete user 1 (who is a manager for user 2 and 3, and has orders 100, 101)
        ("del_user_1", "DELETE FROM users WHERE id = 1;"),
        ("tx_begin", "BEGIN;"),
        ("tx_ins_chain", "INSERT INTO depts VALUES (40, 'R&D', 2000.0);"),
        ("tx_ins_u", "INSERT INTO users VALUES (6, 40, 2, 'Frank', 'gold', 1);"),
        ("tx_ins_o", "INSERT INTO orders VALUES (106, 6, 'server', 5000.0, 'paid');"),
        ("tx_sp", "SAVEPOINT sp_join;"),
        ("tx_upd_fk_cascade_sim", "UPDATE users SET dept_id = 99 WHERE dept_id = 40;"),
        ("tx_del_dept_40", "DELETE FROM depts WHERE id = 40;"),
        ("tx_rb_sp", "ROLLBACK TO SAVEPOINT sp_join;"),
        ("tx_commit", "COMMIT;"),
    ]

    views: List[Tuple[str, str]] = []
    idx = 0

    # 1. Two-table INNER & LEFT joins (users x orders)
    u_o_projs = [
        "u.id AS uid, u.uname, o.id AS oid, o.item, o.total",
        "u.id AS uid, u.tier, o.id AS oid, COALESCE(o.total, 0.0) AS tot",
        "u.uname, u.active, o.id AS oid, o.status",
        "u.id AS uid, o.id AS oid, u.uname || ':' || COALESCE(o.item, 'none') AS tag",
        "u.dept_id, u.uname, o.id AS oid, o.total * 1.1 AS taxed",
    ]
    u_o_wheres = [
        "",
        "WHERE u.active = 1",
        "WHERE u.tier IN ('gold', 'silver')",
        "WHERE o.status = 'paid'",
        "WHERE COALESCE(o.total, 0.0) >= 50.0",
    ]
    for j_type in ["INNER JOIN", "LEFT JOIN"]:
        for proj in u_o_projs:
            for w in u_o_wheres:
                for dist in ["", "DISTINCT "]:
                    idx += 1
                    v_name = f"s5_join_{idx:03d}"
                    w_sql = f" {w}" if w else ""
                    v_sql = f"CREATE VIEW {v_name} AS SELECT {dist}{proj} FROM users u {j_type} orders o ON u.id = o.user_id{w_sql};"
                    views.append((v_name, v_sql))

    # 2. Three-table joins (users x depts x orders) across INNER/LEFT combinations
    three_way_joins = [
        "FROM users u INNER JOIN depts d ON u.dept_id = d.id INNER JOIN orders o ON u.id = o.user_id",
        "FROM users u LEFT JOIN depts d ON u.dept_id = d.id INNER JOIN orders o ON u.id = o.user_id",
        "FROM users u INNER JOIN depts d ON u.dept_id = d.id LEFT JOIN orders o ON u.id = o.user_id",
        "FROM users u LEFT JOIN depts d ON u.dept_id = d.id LEFT JOIN orders o ON u.id = o.user_id",
        "FROM orders o INNER JOIN users u ON o.user_id = u.id LEFT JOIN depts d ON u.dept_id = d.id",
    ]
    three_way_projs = [
        "u.id AS uid, u.uname, d.dname, o.id AS oid, o.total",
        "u.uname, COALESCE(d.dname, 'NoDept') AS dept, COALESCE(o.item, 'NoItem') AS item, o.total",
        "u.id AS uid, d.id AS did, o.id AS oid, d.budget, o.status",
        "u.tier, d.dname, o.status, COALESCE(o.total, 0.0) AS tot",
        "u.id AS uid, u.active, d.dname, o.id AS oid",
    ]
    three_way_wheres = [
        "",
        "WHERE u.active = 1",
        "WHERE COALESCE(d.budget, 500.0) >= 500.0",
        "WHERE u.tier = 'gold' OR o.status = 'paid'",
    ]
    for j_from in three_way_joins:
        for proj in three_way_projs:
            for w in three_way_wheres:
                idx += 1
                v_name = f"s5_join_{idx:03d}"
                w_sql = f" {w}" if w else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT {proj} {j_from}{w_sql};"
                views.append((v_name, v_sql))

    # 3. Join + GROUP BY Aggregates (INNER & LEFT JOIN with COUNT, SUM, AVG, MIN, MAX, DISTINCT)
    join_agg_queries = [
        "SELECT u.tier, COUNT(*) AS r_cnt, COUNT(o.id) AS o_cnt, SUM(o.total) AS s_tot, AVG(o.total) AS a_tot FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.tier",
        "SELECT u.id AS uid, u.uname, COUNT(o.id) AS o_cnt, TOTAL(o.total) AS t_tot, MIN(o.total) AS min_t, MAX(o.total) AS max_t FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.uname",
        "SELECT d.dname, COUNT(u.id) AS u_cnt, COUNT(DISTINCT u.tier) AS d_tiers FROM depts d LEFT JOIN users u ON d.id = u.dept_id GROUP BY d.dname",
        "SELECT d.id AS did, d.dname, COUNT(o.id) AS o_cnt, SUM(o.total) AS rev FROM depts d LEFT JOIN users u ON d.id = u.dept_id LEFT JOIN orders o ON u.id = o.user_id GROUP BY d.id, d.dname",
        "SELECT u.tier, o.status, COUNT(*) AS cnt, SUM(o.total) AS sum_t FROM users u INNER JOIN orders o ON u.id = o.user_id GROUP BY u.tier, o.status",
    ]
    for base_q in join_agg_queries:
        for w in ["", " WHERE u.active = 1", " WHERE u.id >= 1"]:
            for h in ["", " HAVING COUNT(*) >= 1", " HAVING TOTAL(o.total) >= 0.0"]:
                if "o.total" in h and "o.total" not in base_q:
                    continue
                idx += 1
                v_name = f"s5_join_{idx:03d}"
                q = base_q.replace(" GROUP BY", f"{w} GROUP BY") + h
                views.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    # 4. Self-Join (Employee -> Manager) across INNER and LEFT JOIN
    self_join_projs = [
        "e.id AS emp_id, e.uname AS emp_name, m.id AS mgr_id, m.uname AS mgr_name",
        "e.uname AS emp, COALESCE(m.uname, 'CEO') AS boss, e.tier AS emp_tier, m.tier AS mgr_tier",
        "e.id AS eid, m.id AS mid, e.active AS e_act, m.active AS m_act",
    ]
    for j_type in ["INNER JOIN", "LEFT JOIN"]:
        for proj in self_join_projs:
            for w in ["", "WHERE e.active = 1", "WHERE e.id <> COALESCE(m.id, -1)"]:
                idx += 1
                v_name = f"s5_join_{idx:03d}"
                w_sql = f" {w}" if w else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT {proj} FROM users e {j_type} users m ON e.mgr_id = m.id{w_sql};"
                views.append((v_name, v_sql))

    runner.run_batch("Suite 5 (Multi-Table INNER, LEFT, 3-Way, Self-Join & Join-Aggregates)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 6: 200 Subquery Filter Combinations (EXISTS, NOT EXISTS, IN, NOT IN)
# ==============================================================================
def run_suite_6_subqueries(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        country TEXT,
        vip INTEGER
    );
    CREATE TABLE invoices (
        id INTEGER PRIMARY KEY,
        cust_id INTEGER,
        amount REAL,
        status TEXT
    );
    CREATE TABLE blacklist (
        id INTEGER PRIMARY KEY,
        cust_id INTEGER,
        reason TEXT
    );
    """
    seed = [
        "INSERT INTO customers VALUES (1, 'Alice', 'US', 1), (2, 'Bob', 'UK', 0), (3, 'Cara', 'US', 1), (4, 'Dan', 'CA', 0);",
        "INSERT INTO invoices VALUES (10, 1, 500.0, 'paid'), (11, 1, 150.0, 'open'), (12, 2, 80.0, 'paid');",
        "INSERT INTO blacklist VALUES (100, 3, 'fraud');",
    ]
    mutations = [
        # Add invoice for Cara (id=3) -> changes EXISTS / IN results for Cara
        ("ins_inv_cara", "INSERT INTO invoices VALUES (13, 3, 900.0, 'paid');"),
        # Remove Cara from blacklist -> Cara now passes NOT EXISTS / NOT IN (blacklist)!
        ("del_bl_cara", "DELETE FROM blacklist WHERE cust_id = 3;"),
        # Add Alice to blacklist -> Alice now fails NOT EXISTS / NOT IN (blacklist)!
        ("ins_bl_alice", "INSERT INTO blacklist VALUES (101, 1, 'chargeback');"),
        # Delete Bob's only invoice -> Bob now fails EXISTS / IN (invoices)!
        ("del_inv_bob", "DELETE FROM invoices WHERE cust_id = 2;"),
        # Insert a NULL cust_id into blacklist -> critical three-valued logic test for NOT IN!
        ("ins_null_bl", "INSERT INTO blacklist VALUES (102, NULL, 'unknown_target');"),
        # Delete the NULL cust_id from blacklist -> restores NOT IN rows!
        ("del_null_bl", "DELETE FROM blacklist WHERE id = 102;"),
        # Update customer country & vip
        ("upd_cust", "UPDATE customers SET country = 'UK', vip = 1 WHERE id = 4;"),
    ]

    subq_predicates = [
        "EXISTS (SELECT 1 FROM invoices i WHERE i.cust_id = c.id)",
        "EXISTS (SELECT 1 FROM invoices i WHERE i.cust_id = c.id AND i.status = 'paid')",
        "EXISTS (SELECT 1 FROM invoices i WHERE i.cust_id = c.id AND i.amount >= 200.0)",
        "NOT EXISTS (SELECT 1 FROM blacklist b WHERE b.cust_id = c.id)",
        "NOT EXISTS (SELECT 1 FROM invoices i WHERE i.cust_id = c.id AND i.status = 'open')",
        "c.id IN (SELECT i.cust_id FROM invoices i WHERE i.status = 'paid')",
        "c.id IN (SELECT i.cust_id FROM invoices i WHERE i.amount > 100.0)",
        "c.id NOT IN (SELECT b.cust_id FROM blacklist b)",
        "c.id NOT IN (SELECT b.cust_id FROM blacklist b WHERE b.cust_id IS NOT NULL)",
        "EXISTS (SELECT 1 FROM invoices i WHERE i.cust_id = c.id) AND NOT EXISTS (SELECT 1 FROM blacklist b WHERE b.cust_id = c.id)",
    ]
    base_filters = [
        "",
        "AND c.vip = 1",
        "AND c.country = 'US'",
        "AND c.country IN ('US', 'UK')",
        "AND c.id >= 2",
    ]
    shapes = [
        ("c.id, c.name, c.country, c.vip", ""),
        ("DISTINCT c.country, c.vip", ""),
        ("c.country, COUNT(*) AS cust_cnt, SUM(c.vip) AS vip_cnt", "GROUP BY c.country"),
        ("c.vip, COUNT(*) AS cnt, MIN(c.id) AS min_id, MAX(c.id) AS max_id", "GROUP BY c.vip"),
    ]

    views: List[Tuple[str, str]] = []
    idx = 0
    for pred in subq_predicates:
        for bf in base_filters:
            for proj, grp in shapes:
                idx += 1
                v_name = f"s6_subq_{idx:03d}"
                g_sql = f" {grp}" if grp else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT {proj} FROM customers c WHERE {pred} {bf}{g_sql};".replace("  ", " ")
                views.append((v_name, v_sql))

    runner.run_batch("Suite 6 (EXISTS, NOT EXISTS, IN, NOT IN & Three-Valued Logic)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 7: 150 Set Operation Combinations (UNION ALL, UNION, INTERSECT, EXCEPT)
# ==============================================================================
def run_suite_7_set_operations(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE store_orders (
        id INTEGER PRIMARY KEY,
        sku TEXT,
        region TEXT,
        qty INTEGER
    );
    CREATE TABLE web_orders (
        id INTEGER PRIMARY KEY,
        sku TEXT,
        region TEXT,
        qty INTEGER
    );
    CREATE TABLE returns_log (
        id INTEGER PRIMARY KEY,
        sku TEXT,
        region TEXT,
        qty INTEGER
    );
    """
    seed = [
        "INSERT INTO store_orders VALUES (1, 'A', 'NA', 2), (2, 'B', 'EU', 1), (3, 'A', 'NA', 2), (4, NULL, 'NA', 5);",
        "INSERT INTO web_orders VALUES (10, 'B', 'EU', 1), (11, 'C', 'APAC', 3), (12, NULL, 'NA', 5);",
        "INSERT INTO returns_log VALUES (100, 'A', 'NA', 2);",
    ]
    mutations = [
        ("ins_web", "INSERT INTO web_orders VALUES (13, 'A', 'NA', 2), (14, 'D', 'NA', 4);"),
        # Delete one duplicate of ('A', 'NA', 2) from store_orders -> still present in store_orders & web_orders!
        ("del_store_dup1", "DELETE FROM store_orders WHERE id = 1;"),
        # Delete second copy from store_orders -> ('A', 'NA', 2) no longer in store_orders!
        ("del_store_dup2", "DELETE FROM store_orders WHERE id = 3;"),
        ("upd_web", "UPDATE web_orders SET sku = 'Z', qty = 9 WHERE id = 10;"),
        ("del_returns", "DELETE FROM returns_log WHERE id = 100;"),
        ("ins_returns", "INSERT INTO returns_log VALUES (101, 'C', 'APAC', 3);"),
        ("del_null_row", "DELETE FROM store_orders WHERE id = 4;"),
    ]

    projs = [
        "sku, region",
        "sku, qty",
        "region, qty",
        "sku, region, qty",
        "COALESCE(sku, 'UNK') AS s_clean, region",
    ]
    s_wheres = ["", "WHERE qty >= 2", "WHERE region = 'NA'", "WHERE sku IS NOT NULL", "WHERE qty <= 5"]
    w_wheres = ["", "WHERE qty >= 1", "WHERE region IN ('NA', 'EU')", "WHERE sku IS NOT NULL"]
    ops = ["UNION ALL", "UNION", "INTERSECT", "EXCEPT"]

    views: List[Tuple[str, str]] = []
    idx = 0
    for op in ops:
        for proj in projs:
            for sw in s_wheres:
                for ww in w_wheres[:2]:
                    idx += 1
                    if idx > 130:
                        break
                    v_name = f"s7_set_{idx:03d}"
                    sw_sql = f" {sw}" if sw else ""
                    ww_sql = f" {ww}" if ww else ""
                    v_sql = (
                        f"CREATE VIEW {v_name} AS "
                        f"SELECT {proj} FROM store_orders{sw_sql} "
                        f"{op} "
                        f"SELECT {proj} FROM web_orders{ww_sql};"
                    )
                    views.append((v_name, v_sql))

    # Add 20 three-branch compound set operations
    three_branch_ops = [
        ("UNION ALL", "UNION ALL"),
        ("UNION", "UNION"),
        ("UNION", "EXCEPT"),
        ("UNION ALL", "EXCEPT"),
        ("INTERSECT", "UNION"),
    ]
    for op1, op2 in three_branch_ops:
        for proj in projs[:4]:
            idx += 1
            v_name = f"s7_set_{idx:03d}"
            v_sql = (
                f"CREATE VIEW {v_name} AS "
                f"SELECT {proj} FROM store_orders "
                f"{op1} "
                f"SELECT {proj} FROM web_orders "
                f"{op2} "
                f"SELECT {proj} FROM returns_log;"
            )
            views.append((v_name, v_sql))

    runner.run_batch("Suite 7 (UNION ALL, UNION, INTERSECT, EXCEPT & Multi-Branch Sets)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 8: 250 CTE, Recursive CTE, Window Function & Top-N Combinations
# ==============================================================================
def run_suite_8_ctes_windows_topn(runner: MVTestRunner) -> None:
    schema = """
    CREATE TABLE nodes (
        id INTEGER PRIMARY KEY,
        parent_id INTEGER,
        label TEXT,
        team TEXT,
        score INTEGER
    );
    """
    seed = [
        "INSERT INTO nodes VALUES (1, NULL, 'root_a', 'alpha', 50);",
        "INSERT INTO nodes VALUES (2, 1, 'child_a1', 'alpha', 80);",
        "INSERT INTO nodes VALUES (3, 1, 'child_a2', 'beta', 80);",
        "INSERT INTO nodes VALUES (4, 2, 'leaf_a1_1', 'alpha', 30);",
        "INSERT INTO nodes VALUES (5, NULL, 'root_b', 'beta', 95);",
        "INSERT INTO nodes VALUES (6, 5, 'child_b1', 'beta', 60);",
    ]
    mutations = [
        # Insert a new child under leaf 4 -> extends recursive hierarchy to depth 3!
        ("ins_deep", "INSERT INTO nodes VALUES (7, 4, 'deep_leaf', 'alpha', 110);"),
        # Re-parent node 2 from root 1 to root 5 -> shifts entire subtree (nodes 2, 4, 7)!
        ("reparent_subtree", "UPDATE nodes SET parent_id = 5, score = 85 WHERE id = 2;"),
        # Delete node 4 -> prunes nodes 4 and 7 from recursive tree and shifts window ranks/frames!
        ("del_mid_node", "DELETE FROM nodes WHERE id = 4;"),
        # Update scores to create and break ties in RANK / DENSE_RANK / Top-N
        ("upd_scores", "UPDATE nodes SET score = 95 WHERE team = 'alpha';"),
        ("del_root", "DELETE FROM nodes WHERE id = 1;"),
        ("ins_new_root", "INSERT INTO nodes VALUES (8, NULL, 'root_c', 'gamma', 70);"),
    ]

    views: List[Tuple[str, str]] = []
    idx = 0

    # 1. Non-recursive CTEs (single & chained CTEs, with joins & aggregates) - 60 views
    cte_queries = [
        "WITH active_nodes AS (SELECT id, parent_id, label, team, score FROM nodes WHERE score >= 40) SELECT id, label, team, score FROM active_nodes",
        "WITH active_nodes AS (SELECT id, parent_id, label, team, score FROM nodes WHERE score >= 40) SELECT team, COUNT(*) AS cnt, SUM(score) AS total_s, MAX(score) AS max_s FROM active_nodes GROUP BY team",
        "WITH roots AS (SELECT id, label, team FROM nodes WHERE parent_id IS NULL), children AS (SELECT id, parent_id, label, score FROM nodes WHERE parent_id IS NOT NULL) SELECT r.label AS root_lbl, c.label AS child_lbl, c.score FROM roots r INNER JOIN children c ON r.id = c.parent_id",
        "WITH roots AS (SELECT id, label, team FROM nodes WHERE parent_id IS NULL), children AS (SELECT id, parent_id, label, score FROM nodes WHERE parent_id IS NOT NULL) SELECT r.label AS root_lbl, COUNT(c.id) AS num_children, COALESCE(SUM(c.score), 0) AS child_score FROM roots r LEFT JOIN children c ON r.id = c.parent_id GROUP BY r.label",
        "WITH step1 AS (SELECT id, team, score, score * 2 AS s2 FROM nodes), step2 AS (SELECT id, team, s2 FROM step1 WHERE s2 >= 100) SELECT team, COUNT(*) AS cnt, AVG(s2) AS avg_s2 FROM step2 GROUP BY team",
    ]
    for base_cte in cte_queries:
        for thresh in [30, 40, 50, 60, 70, 80]:
            for extra in ["", " WHERE score >= 50" if "GROUP BY" not in base_cte and "roots" not in base_cte else ""]:
                idx += 1
                v_name = f"s8_cwt_{idx:03d}"
                q = base_cte.replace("40", str(thresh)) + extra
                views.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    # 2. Recursive CTEs (hierarchy traversal, depth tracking, path building, subtree rollups) - 30 views
    rec_queries = [
        "WITH RECURSIVE tree(id, parent_id, label, depth) AS (SELECT id, parent_id, label, 0 FROM nodes WHERE parent_id IS NULL UNION ALL SELECT n.id, n.parent_id, n.label, t.depth + 1 FROM nodes n INNER JOIN tree t ON n.parent_id = t.id WHERE t.depth < 10) SELECT id, parent_id, label, depth FROM tree",
        "WITH RECURSIVE tree(id, root_id, score, depth) AS (SELECT id, id, score, 0 FROM nodes WHERE parent_id IS NULL UNION ALL SELECT n.id, t.root_id, n.score, t.depth + 1 FROM nodes n INNER JOIN tree t ON n.parent_id = t.id WHERE t.depth < 10) SELECT root_id, COUNT(*) AS subtree_size, SUM(score) AS subtree_score, MAX(depth) AS max_depth FROM tree GROUP BY root_id",
        "WITH RECURSIVE tree(id, path, team, depth) AS (SELECT id, label, team, 1 FROM nodes WHERE parent_id IS NULL UNION ALL SELECT n.id, t.path || '/' || n.label, n.team, t.depth + 1 FROM nodes n INNER JOIN tree t ON n.parent_id = t.id WHERE t.depth < 10) SELECT id, path, team, depth FROM tree",
    ]
    for rq in rec_queries:
        for root_filter in [
            "parent_id IS NULL",
            "parent_id IS NULL AND score >= 50",
            "parent_id IS NULL AND team = 'alpha'",
            "parent_id IS NULL AND team IN ('alpha', 'beta')",
            "id IN (1, 5)",
        ]:
            for outer_filter in ["", " WHERE depth >= 1" if "GROUP BY" not in rq else " HAVING COUNT(*) >= 1"]:
                idx += 1
                v_name = f"s8_cwt_{idx:03d}"
                q = rq.replace("parent_id IS NULL", root_filter) + outer_filter
                views.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    # 3. Window Functions (ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, FIRST_VALUE, running SUM/AVG) - 100 views
    win_exprs = [
        "ROW_NUMBER() OVER (PARTITION BY team ORDER BY score DESC, id ASC) AS rn",
        "RANK() OVER (PARTITION BY team ORDER BY score DESC) AS rnk",
        "DENSE_RANK() OVER (PARTITION BY team ORDER BY score DESC) AS drnk",
        "SUM(score) OVER (PARTITION BY team ORDER BY id ASC) AS running_sum",
        "AVG(score) OVER (PARTITION BY team ORDER BY id ASC) AS running_avg",
        "LAG(score, 1, -1) OVER (PARTITION BY team ORDER BY id ASC) AS prev_score",
        "LEAD(score, 1, -1) OVER (PARTITION BY team ORDER BY id ASC) AS next_score",
        "FIRST_VALUE(label) OVER (PARTITION BY team ORDER BY score DESC, id ASC) AS top_label",
        "COUNT(*) OVER (PARTITION BY team) AS team_total",
        "ROW_NUMBER() OVER (ORDER BY score DESC, id ASC) AS global_rn",
    ]
    win_wheres = [
        "",
        "WHERE score >= 40",
        "WHERE parent_id IS NOT NULL",
        "WHERE team IN ('alpha', 'beta')",
        "WHERE id <= 10",
    ]
    for w_expr in win_exprs:
        for w_clause in win_wheres:
            for extra_col in ["id, label, team, score", "id, team, score, COALESCE(parent_id, 0) AS pid"]:
                idx += 1
                v_name = f"s8_cwt_{idx:03d}"
                w_sql = f" {w_clause}" if w_clause else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT {extra_col}, {w_expr} FROM nodes{w_sql};"
                views.append((v_name, v_sql))

    # 4. Top-N Queries (ORDER BY ... LIMIT ... OFFSET ...) - 60 views
    order_clauses = [
        "ORDER BY score DESC, id ASC",
        "ORDER BY score ASC, id DESC",
        "ORDER BY team ASC, score DESC, id ASC",
        "ORDER BY COALESCE(parent_id, 0) ASC, id ASC",
    ]
    limit_clauses = [
        "LIMIT 1",
        "LIMIT 2",
        "LIMIT 3",
        "LIMIT 5",
        "LIMIT 2 OFFSET 1",
        "LIMIT 3 OFFSET 2",
    ]
    for ord_c in order_clauses:
        for lim_c in limit_clauses:
            for w_c in ["", "WHERE score >= 50", "WHERE team = 'alpha'"]:
                idx += 1
                if idx > 250:
                    break
                v_name = f"s8_cwt_{idx:03d}"
                w_sql = f" {w_c}" if w_c else ""
                v_sql = f"CREATE VIEW {v_name} AS SELECT id, label, team, score FROM nodes{w_sql} {ord_c} {lim_c};"
                views.append((v_name, v_sql))

    runner.run_batch("Suite 8 (CTEs, Recursive CTEs, Window Functions & Top-N)", schema, views, seed, mutations)


# ==============================================================================
# SUITE 9: 200 Cross-Dialect Equivalence Tests (PostgreSQL & MySQL -> SQLite Execution)
# ==============================================================================
def run_suite_9_cross_dialect_execution(runner: MVTestRunner) -> None:
    """
    Compiles 100 views with Dialect.POSTGRES and 100 views with Dialect.MYSQL
    (covering projections, DISTINCT, algebraic aggregates, holistic MIN/MAX, HAVING,
    COUNT/SUM DISTINCT, ungrouped aggregates, INNER/LEFT joins, subqueries, UNION/EXCEPT,
    window functions, and Top-N), transpiles the generated PostgreSQL/MySQL trigger DDL
    to SQLite via `_transpile_pg_stmt_to_sqlite` / `_transpile_mysql_stmt_to_sqlite`,
    executes them against real SQLite connections across full DML sequences, and asserts
    exact row-multiset parity!
    """
    schema = """
    CREATE TABLE accounts (
        id INTEGER PRIMARY KEY,
        org TEXT,
        tier TEXT,
        balance REAL,
        active INTEGER
    );
    CREATE TABLE txns (
        id INTEGER PRIMARY KEY,
        acct_id INTEGER,
        kind TEXT,
        amount REAL
    );
    """
    seed = [
        "INSERT INTO accounts VALUES (1, 'acme', 'pro', 1000.0, 1), (2, 'acme', 'free', 200.0, 1), (3, 'globex', 'pro', 500.0, 0), (4, NULL, 'ent', 1500.0, 1);",
        "INSERT INTO txns VALUES (10, 1, 'credit', 300.0), (11, 1, 'debit', 100.0), (12, 2, 'credit', 50.0), (13, 3, 'credit', 500.0);",
    ]
    mutations = [
        ("ins_acct_txn", "INSERT INTO accounts VALUES (5, 'initech', 'pro', 800.0, 1);"),
        ("ins_txn", "INSERT INTO txns VALUES (14, 5, 'credit', 250.0), (15, 2, 'credit', 50.0);"),
        ("upd_acct", "UPDATE accounts SET org = 'globex', balance = 1200.0 WHERE id = 2;"),
        ("upd_txn", "UPDATE txns SET amount = 400.0, kind = 'debit' WHERE id = 12;"),
        ("del_txn", "DELETE FROM txns WHERE id = 10;"),
        ("del_acct", "DELETE FROM accounts WHERE id = 3;"),
        ("tx_begin", "BEGIN;"),
        ("tx_ins_acct", "INSERT INTO accounts VALUES (6, 'acme', 'ent', 3000.0, 1);"),
        ("tx_ins_txn", "INSERT INTO txns VALUES (16, 6, 'credit', 1000.0);"),
        ("tx_sp", "SAVEPOINT sp_cd;"),
        ("tx_upd_all", "UPDATE accounts SET balance = balance + 100.0 WHERE org = 'acme';"),
        ("tx_del_txns", "DELETE FROM txns WHERE acct_id = 6;"),
        ("tx_rb_sp", "ROLLBACK TO SAVEPOINT sp_cd;"),
        ("tx_commit", "COMMIT;"),
    ]

    # Build 100 diverse view definitions
    view_defs: List[Tuple[str, str]] = []
    idx = 0

    # 20 projection & DISTINCT views
    for dist in ["", "DISTINCT "]:
        for proj in [
            "id, org, tier, balance",
            "org, tier, balance * 1.05 AS adj_bal",
            "COALESCE(org, 'unassigned') AS org_clean, tier, active",
            "id, CASE WHEN balance >= 1000 THEN 'whale' ELSE 'std' END AS seg",
            "org, active",
        ]:
            for w in ["", "WHERE active = 1"]:
                idx += 1
                v_name = f"s9_cd_{idx:03d}"
                w_sql = f" {w}" if w else ""
                view_defs.append((v_name, f"CREATE VIEW {v_name} AS SELECT {dist}{proj} FROM accounts{w_sql};"))

    # 30 aggregate views (algebraic, holistic MIN/MAX, HAVING, DISTINCT, ungrouped)
    agg_templates = [
        "SELECT org, COUNT(*) AS cnt, SUM(balance) AS sb, AVG(balance) AS ab FROM accounts GROUP BY org",
        "SELECT tier, COUNT(*) AS cnt, TOTAL(balance) AS tb, MIN(balance) AS minb, MAX(balance) AS maxb FROM accounts GROUP BY tier",
        "SELECT org, tier, COUNT(*) AS cnt, SUM(balance) AS sb FROM accounts WHERE active = 1 GROUP BY org, tier",
        "SELECT org, COUNT(*) AS cnt, MAX(balance) AS maxb FROM accounts GROUP BY org HAVING COUNT(*) >= 1",
        "SELECT tier, COUNT(DISTINCT org) AS d_org, SUM(DISTINCT balance) AS sdb, COUNT(*) AS cnt FROM accounts GROUP BY tier",
        "SELECT COUNT(*) AS total_accts, SUM(balance) AS global_bal, MIN(balance) AS min_b, MAX(balance) AS max_b FROM accounts",
    ]
    for tmpl in agg_templates:
        for mod in [
            "",
            " WHERE balance >= 100.0",
            " WHERE org IS NOT NULL",
            " WHERE active = 1",
            " WHERE id >= 1",
        ]:
            idx += 1
            v_name = f"s9_cd_{idx:03d}"
            if "WHERE" in tmpl and mod:
                q = tmpl.replace("WHERE active = 1", f"WHERE active = 1 AND {mod.replace(' WHERE ', '')}")
            elif "GROUP BY" in tmpl and mod:
                q = tmpl.replace(" GROUP BY", f"{mod} GROUP BY")
            else:
                q = tmpl + mod
            view_defs.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    # 25 Join & Subquery views
    join_subq_templates = [
        "SELECT a.id AS aid, a.org, t.id AS tid, t.kind, t.amount FROM accounts a INNER JOIN txns t ON a.id = t.acct_id",
        "SELECT a.id AS aid, a.org, t.id AS tid, COALESCE(t.amount, 0.0) AS amt FROM accounts a LEFT JOIN txns t ON a.id = t.acct_id",
        "SELECT a.org, COUNT(t.id) AS t_cnt, SUM(t.amount) AS t_sum FROM accounts a LEFT JOIN txns t ON a.id = t.acct_id GROUP BY a.org",
        "SELECT a.id, a.org, a.balance FROM accounts a WHERE EXISTS (SELECT 1 FROM txns t WHERE t.acct_id = a.id AND t.kind = 'credit')",
        "SELECT a.id, a.org, a.balance FROM accounts a WHERE NOT EXISTS (SELECT 1 FROM txns t WHERE t.acct_id = a.id AND t.kind = 'debit')",
    ]
    for tmpl in join_subq_templates:
        for extra in ["", " AND a.active = 1", " AND a.balance >= 200.0", " AND a.id <= 10", " AND a.tier IN ('pro', 'ent')"]:
            idx += 1
            v_name = f"s9_cd_{idx:03d}"
            if "WHERE" in tmpl:
                q = tmpl + extra
            elif "GROUP BY" in tmpl and extra:
                q = tmpl.replace(" GROUP BY", f" WHERE {extra.replace(' AND ', '')} GROUP BY")
            elif extra:
                q = tmpl + f" WHERE {extra.replace(' AND ', '')}"
            else:
                q = tmpl
            view_defs.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    # 25 Set Ops, CTEs, Windows, Top-N views
    misc_templates = [
        "SELECT acct_id AS id, kind AS label FROM txns UNION ALL SELECT id, tier AS label FROM accounts",
        "SELECT acct_id AS id FROM txns UNION SELECT id FROM accounts WHERE active = 1",
        "SELECT id FROM accounts EXCEPT SELECT acct_id AS id FROM txns WHERE kind = 'debit'",
        "SELECT id, org, balance, ROW_NUMBER() OVER (PARTITION BY org ORDER BY balance DESC, id ASC) AS rn FROM accounts",
        "SELECT id, org, balance FROM accounts ORDER BY balance DESC, id ASC LIMIT 3",
    ]
    for tmpl in misc_templates:
        for k in range(5):
            idx += 1
            v_name = f"s9_cd_{idx:03d}"
            q = tmpl.replace("LIMIT 3", f"LIMIT {k + 1}") if "LIMIT" in tmpl else tmpl
            view_defs.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    assert len(view_defs) == 100

    # 9A: PostgreSQL Row-Level Triggers
    pg_views = [(f"pg_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW pg_{n}")) for n, sql in view_defs]
    runner.run_batch(
        "Suite 9A (100 PostgreSQL Row-Level Trigger Views)",
        schema,
        pg_views,
        seed,
        mutations,
        dialect=Dialect.POSTGRES,
        pg_use_transition_tables=False,
        incremental_backfill=False,
    )

    # 9B: PostgreSQL Statement-Level Transition Tables & Incremental DO-Block Backfill
    pg_tr_views = [(f"pgtr_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW pgtr_{n}")) for n, sql in view_defs]
    runner.run_batch(
        "Suite 9B (100 PostgreSQL Transition-Table & Incremental Backfill Views)",
        schema,
        pg_tr_views,
        seed,
        mutations,
        dialect=Dialect.POSTGRES,
        pg_use_transition_tables=True,
        incremental_backfill=True,
        batch_size=2,
    )

    # 9C: MySQL Standard Backfill
    my_views = [(f"my_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW my_{n}")) for n, sql in view_defs]
    runner.run_batch(
        "Suite 9C (100 MySQL Standard Trigger Views)",
        schema,
        my_views,
        seed,
        mutations,
        dialect=Dialect.MYSQL,
        incremental_backfill=False,
    )

    # 9D: MySQL Stored-Procedure Incremental Backfill
    my_inc_views = [(f"myinc_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW myinc_{n}")) for n, sql in view_defs]
    runner.run_batch(
        "Suite 9D (100 MySQL Stored-Procedure Incremental Backfill Views)",
        schema,
        my_inc_views,
        seed,
        mutations,
        dialect=Dialect.MYSQL,
        incremental_backfill=True,
        batch_size=2,
    )


# ==============================================================================
# SUITE 10: High-Concurrency & Torture-DML Combinatorial Matrix across ALL 3 Dialects
# ==============================================================================
def run_suite_10_torture_dml_and_transactions(runner: MVTestRunner) -> None:
    """
    Tests a rich 4-table relational schema across complex DML and multi-step transactions
    (SAVEPOINT, ROLLBACK TO SAVEPOINT, full ROLLBACK, UPSERT ON CONFLICT DO UPDATE,
    INSERT INTO ... SELECT, correlated UPDATE, self-referencing swaps, NULL-to-non-NULL
    group migrations, and multi-statement transactions verified at EVERY intermediate
    point inside the open transaction as well as after COMMIT/ROLLBACK) across:
      - SQLite (native + incremental backfill)
      - PostgreSQL (row-level + statement-level transition tables + incremental backfill)
      - MySQL (native + incremental backfill)
    """
    schema = """
    CREATE TABLE regions (
        id INTEGER PRIMARY KEY,
        rname TEXT NOT NULL,
        tax_rate REAL NOT NULL
    );
    CREATE TABLE merchants (
        id INTEGER PRIMARY KEY,
        region_id INTEGER,
        mname TEXT NOT NULL,
        tier TEXT,
        active INTEGER NOT NULL
    );
    CREATE TABLE invoices (
        id INTEGER PRIMARY KEY,
        merchant_id INTEGER,
        category TEXT,
        subtotal REAL,
        qty INTEGER,
        discount REAL,
        settled INTEGER NOT NULL
    );
    CREATE TABLE refunds (
        id INTEGER PRIMARY KEY,
        invoice_id INTEGER,
        reason TEXT,
        refund_amt REAL NOT NULL
    );
    """

    seed = [
        "INSERT INTO regions VALUES (1, 'AMER', 0.08), (2, 'EMEA', 0.20), (3, 'APAC', 0.10);",
        "INSERT INTO merchants VALUES (10, 1, 'AlphaCorp', 'gold', 1), (20, 1, 'BetaLLC', 'silver', 1), (30, 2, 'GammaGmbH', 'gold', 0), (40, NULL, 'GlobalNomad', 'bronze', 1);",
        "INSERT INTO invoices VALUES (100, 10, 'cloud', 500.0, 5, 25.0, 1), (101, 10, 'support', 150.0, 1, NULL, 1), (102, 20, 'cloud', 300.0, 3, 0.0, 0), (103, 30, 'hw', 800.0, 2, 50.0, 1), (104, 40, NULL, 120.0, 4, 10.0, 1), (105, 99, 'orphan_cat', 60.0, 1, 0.0, 0);",
        "INSERT INTO refunds VALUES (1000, 100, 'overcharge', 50.0), (1001, 100, 'promo', 25.0), (1002, 103, 'damaged', 200.0);",
    ]

    # Comprehensive torture DML sequence with intermediate transaction assertions at every step!
    torture_mutations = [
        # 1. Multi-row INSERT with NULLs and duplicate group/distinct values
        ("dml_multi_ins", "INSERT INTO invoices VALUES (106, 10, 'cloud', 500.0, 5, 25.0, 1), (107, 20, 'ai', 1000.0, 10, NULL, 1), (108, NULL, 'cloud', 250.0, 2, 5.0, 0);"),
        # 2. UPSERT (INSERT ... ON CONFLICT DO UPDATE) updating existing + inserting new
        ("dml_upsert_inv", "INSERT INTO invoices VALUES (102, 20, 'cloud', 450.0, 6, 15.0, 1), (109, 30, 'sw', 220.0, 2, 0.0, 1) ON CONFLICT (id) DO UPDATE SET subtotal = invoices.subtotal + excluded.subtotal, qty = excluded.qty, discount = excluded.discount, settled = excluded.settled;"),
        # 3. Self-referencing arithmetic & swapping columns across rows in a single UPDATE
        ("dml_swap_cols", "UPDATE invoices SET subtotal = COALESCE(discount, 10.0) * 10.0, discount = subtotal * 0.05, category = COALESCE(category, 'misc') WHERE settled = 1 AND qty >= 3;"),
        # 4. Correlated subquery UPDATE referencing joined parent tables
        ("dml_corr_upd", "UPDATE invoices SET subtotal = subtotal * (1.0 + COALESCE((SELECT r.tax_rate FROM merchants m INNER JOIN regions r ON m.region_id = r.id WHERE m.id = invoices.merchant_id), 0.05)) WHERE merchant_id IS NOT NULL;"),
        # 5. Foreign-key migration across groups (NULL -> valid FK, valid FK -> another FK)
        ("dml_fk_migrate", "UPDATE merchants SET region_id = 3, tier = 'gold' WHERE id = 40;"),
        ("dml_fk_orphan", "UPDATE merchants SET region_id = NULL, active = 1 WHERE id = 30;"),
        # 6. INSERT INTO ... SELECT derived from aggregate on another table
        ("dml_ins_select", "INSERT INTO refunds SELECT 2000 + id, id, 'auto_rebate', subtotal * 0.1 FROM invoices WHERE subtotal >= 600.0 AND id NOT IN (SELECT invoice_id FROM (SELECT invoice_id FROM refunds WHERE invoice_id IS NOT NULL) AS __sq_ref);"),
        # 7. Transaction 1: Multi-step DML + Nested SAVEPOINTs + Partial Rollbacks + Full Rollback
        ("tx1_begin", "BEGIN;"),
        ("tx1_ins_reg", "INSERT INTO regions VALUES (4, 'LATAM', 0.15);"),
        ("tx1_upd_merch", "UPDATE merchants SET region_id = 4, active = 0 WHERE tier = 'gold';"),
        ("tx1_sp_a", "SAVEPOINT sp_a;"),
        ("tx1_del_refunds", "DELETE FROM refunds WHERE refund_amt < 100.0;"),
        ("tx1_upd_inv_cat", "UPDATE invoices SET category = 'archived', settled = 0 WHERE merchant_id IN (10, 20);"),
        ("tx1_sp_b", "SAVEPOINT sp_b;"),
        ("tx1_del_all_inv", "DELETE FROM invoices;"),
        ("tx1_del_all_merch", "DELETE FROM merchants;"),
        # Intermediate state right here has 0 invoices and 0 merchants — all MVs must reflect empty state inside the open tx!
        ("tx1_rb_sp_b", "ROLLBACK TO SAVEPOINT sp_b;"),
        # Immediately after ROLLBACK TO SAVEPOINT sp_b, all MVs must match tx1_upd_inv_cat state!
        ("tx1_ins_after_sp_b", "INSERT INTO refunds VALUES (3001, 101, 'late_credit', 45.0);"),
        ("tx1_rb_sp_a", "ROLLBACK TO SAVEPOINT sp_a;"),
        # Immediately after ROLLBACK TO SAVEPOINT sp_a, all MVs must match tx1_upd_merch state!
        ("tx1_full_rollback", "ROLLBACK;"),
        # Immediately after full ROLLBACK, all MVs must match pre-tx1 (dml_ins_select) state!
        # 8. Transaction 2: Complex multi-table mutations committed atomically, verified at every intermediate step AND after COMMIT
        ("tx2_begin", "BEGIN;"),
        ("tx2_del_subq", "DELETE FROM invoices WHERE EXISTS (SELECT 1 FROM refunds rf WHERE rf.invoice_id = invoices.id AND rf.refund_amt >= 150.0);"),
        ("tx2_upd_region", "UPDATE regions SET rname = rname || '_V2', tax_rate = tax_rate + 0.02 WHERE id IN (1, 2);"),
        ("tx2_sp_commit", "SAVEPOINT sp_c;"),
        ("tx2_upsert_merch", "INSERT INTO merchants VALUES (20, 2, 'BetaLLC_EU', 'platinum', 1), (50, 3, 'DeltaJP', 'gold', 1) ON CONFLICT (id) DO UPDATE SET region_id = excluded.region_id, mname = excluded.mname, tier = excluded.tier;"),
        ("tx2_release_sp", "RELEASE SAVEPOINT sp_c;"),
        ("tx2_commit", "COMMIT;"),
        # 9. Complete table wipe and rebuild to verify zero-row transition and recovery
        ("dml_wipe_refunds", "DELETE FROM refunds;"),
        ("dml_wipe_invoices", "DELETE FROM invoices;"),
        ("dml_rebuild_inv", "INSERT INTO invoices VALUES (500, 10, 'enterprise', 2500.0, 25, 100.0, 1), (501, 50, 'cloud', 750.0, 8, NULL, 1);"),
        ("dml_rebuild_ref", "INSERT INTO refunds VALUES (5000, 500, 'loyalty', 125.0);"),
    ]

    views: List[Tuple[str, str]] = []
    idx = 0

    # Build 125 comprehensive views spanning all SQL features & combinations on the 4-table schema
    # A. Projections & Complex Expressions (20 views)
    proj_templates = [
        "SELECT id, category, subtotal, qty, COALESCE(discount, 0.0) AS disc, subtotal * qty - COALESCE(discount, 0.0) AS net_rev FROM invoices",
        "SELECT DISTINCT category, settled, CASE WHEN subtotal >= 500.0 THEN 'large' WHEN subtotal >= 200.0 THEN 'mid' ELSE 'small' END AS deal_size FROM invoices",
        "SELECT id, mname, COALESCE(tier, 'std') AS tier_clean, active FROM merchants",
        "SELECT DISTINCT region_id, tier, active FROM merchants",
    ]
    for pt in proj_templates:
        for wf in ["", " WHERE settled = 1", " WHERE subtotal > 150.0", " WHERE category IS NOT NULL", " WHERE qty >= 2"]:
            if "merchants" in pt:
                wf = wf.replace("settled = 1", "active = 1").replace("subtotal > 150.0", "id >= 20").replace("category IS NOT NULL", "region_id IS NOT NULL").replace("qty >= 2", "tier IN ('gold', 'platinum')")
            idx += 1
            v_name = f"s10_torture_{idx:03d}"
            views.append((v_name, f"CREATE VIEW {v_name} AS {pt}{wf};"))

    # B. Algebraic, Holistic (MIN/MAX), HAVING, DISTINCT & Ungrouped Aggregates (35 views)
    agg_templates = [
        "SELECT category, COUNT(*) AS cnt, SUM(subtotal) AS s_sub, AVG(subtotal) AS a_sub, TOTAL(discount) AS t_disc FROM invoices GROUP BY category",
        "SELECT merchant_id, settled, COUNT(*) AS cnt, SUM(subtotal * qty) AS gross, COUNT(discount) AS c_disc FROM invoices GROUP BY merchant_id, settled",
        "SELECT category, COUNT(*) AS cnt, MIN(subtotal) AS min_s, MAX(subtotal) AS max_s, MIN(qty) AS min_q, MAX(qty) AS max_q FROM invoices GROUP BY category",
        "SELECT category, COUNT(*) AS cnt, SUM(subtotal) AS s_sub, MAX(subtotal) AS max_s FROM invoices GROUP BY category HAVING COUNT(*) >= 2 OR SUM(subtotal) >= 600.0",
        "SELECT merchant_id, COUNT(DISTINCT category) AS d_cat, SUM(DISTINCT qty) AS sd_qty, AVG(DISTINCT subtotal) AS ad_sub, COUNT(*) AS n FROM invoices GROUP BY merchant_id",
        "SELECT COUNT(*) AS g_cnt, SUM(subtotal) AS g_sum, AVG(subtotal) AS g_avg, MIN(subtotal) AS g_min, MAX(subtotal) AS g_max, COUNT(DISTINCT category) AS g_dcat FROM invoices",
        "SELECT reason, COUNT(*) AS r_cnt, SUM(refund_amt) AS r_sum, MIN(refund_amt) AS r_min, MAX(refund_amt) AS r_max FROM refunds GROUP BY reason",
    ]
    for at in agg_templates:
        for wf in ["", " WHERE settled = 1", " WHERE subtotal >= 200.0", " WHERE discount IS NOT NULL", " WHERE id >= 100"]:
            if "refunds" in at:
                wf = wf.replace("settled = 1", "refund_amt >= 50.0").replace("subtotal >= 200.0", "invoice_id IS NOT NULL").replace("discount IS NOT NULL", "id >= 1000").replace("id >= 100", "refund_amt <= 500.0")
            idx += 1
            v_name = f"s10_torture_{idx:03d}"
            q = at.replace(" GROUP BY", f"{wf} GROUP BY") if " GROUP BY" in at else at + wf
            views.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    # C. Multi-Table Joins (2-table, 3-table, 4-table INNER & LEFT joins + Join Aggregates) (35 views)
    join_templates = [
        "SELECT m.id AS mid, m.mname, r.rname, r.tax_rate FROM merchants m INNER JOIN regions r ON m.region_id = r.id",
        "SELECT m.id AS mid, m.mname, COALESCE(r.rname, 'UNASSIGNED') AS reg, COALESCE(r.tax_rate, 0.0) AS tax FROM merchants m LEFT JOIN regions r ON m.region_id = r.id",
        "SELECT i.id AS iid, m.mname, r.rname, i.category, i.subtotal FROM invoices i INNER JOIN merchants m ON i.merchant_id = m.id LEFT JOIN regions r ON m.region_id = r.id",
        "SELECT m.id AS mid, m.mname, i.id AS iid, COALESCE(i.subtotal, 0.0) AS sub, rf.id AS rfid, COALESCE(rf.refund_amt, 0.0) AS ref_amt FROM merchants m LEFT JOIN invoices i ON m.id = i.merchant_id LEFT JOIN refunds rf ON i.id = rf.invoice_id",
        "SELECT r.rname, COUNT(m.id) AS m_cnt, COUNT(i.id) AS i_cnt, TOTAL(i.subtotal) AS rev, MIN(i.subtotal) AS min_inv, MAX(i.subtotal) AS max_inv FROM regions r LEFT JOIN merchants m ON r.id = m.region_id LEFT JOIN invoices i ON m.id = i.merchant_id GROUP BY r.rname",
        "SELECT m.tier, COUNT(DISTINCT i.category) AS d_cats, SUM(i.subtotal) AS sum_sub, TOTAL(rf.refund_amt) AS tot_ref FROM merchants m LEFT JOIN invoices i ON m.id = i.merchant_id LEFT JOIN refunds rf ON i.id = rf.invoice_id GROUP BY m.tier",
        "SELECT i.category, COUNT(*) AS cnt, SUM(i.subtotal * (1.0 + COALESCE(r.tax_rate, 0.0))) AS taxed_total FROM invoices i LEFT JOIN merchants m ON i.merchant_id = m.id LEFT JOIN regions r ON m.region_id = r.id GROUP BY i.category",
    ]
    for jt in join_templates:
        for wf in ["", " WHERE m.active = 1", " WHERE m.tier IN ('gold', 'platinum')", " WHERE COALESCE(i.settled, 1) = 1", " WHERE m.id >= 10"]:
            if "invoices i" not in jt:
                wf = wf.replace("COALESCE(i.settled, 1) = 1", "COALESCE(r.id, 1) >= 1")
            if "r.rname, COUNT" in jt:
                wf = wf.replace("m.active = 1", "r.tax_rate >= 0.05").replace("m.tier IN ('gold', 'platinum')", "r.id >= 1").replace("COALESCE(i.settled, 1) = 1", "r.rname IS NOT NULL").replace("m.id >= 10", "r.id <= 10")
            idx += 1
            v_name = f"s10_torture_{idx:03d}"
            q = jt.replace(" GROUP BY", f"{wf} GROUP BY") if " GROUP BY" in jt else jt + wf
            views.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    # D. Subqueries, Set Operations, CTEs, Window Functions & Top-N (35 views)
    complex_templates = [
        "SELECT m.id, m.mname, m.tier FROM merchants m WHERE EXISTS (SELECT 1 FROM invoices i WHERE i.merchant_id = m.id AND i.settled = 1)",
        "SELECT m.id, m.mname, m.tier FROM merchants m WHERE NOT EXISTS (SELECT 1 FROM invoices i WHERE i.merchant_id = m.id AND i.subtotal >= 600.0)",
        "SELECT i.id, i.category, i.subtotal FROM invoices i WHERE i.id NOT IN (SELECT rf.invoice_id FROM refunds rf WHERE rf.invoice_id IS NOT NULL)",
        "SELECT m.tier, COUNT(*) AS active_with_inv FROM merchants m WHERE m.active = 1 AND EXISTS (SELECT 1 FROM invoices i WHERE i.merchant_id = m.id) GROUP BY m.tier",
        "SELECT merchant_id AS entity_id, category AS tag FROM invoices WHERE merchant_id IS NOT NULL UNION SELECT id AS entity_id, tier AS tag FROM merchants",
        "SELECT id AS inv_id FROM invoices WHERE settled = 1 EXCEPT SELECT invoice_id AS inv_id FROM refunds WHERE invoice_id IS NOT NULL",
        "SELECT invoice_id AS inv_id FROM refunds INTERSECT SELECT id AS inv_id FROM invoices WHERE settled = 1",
        "WITH net_inv AS (SELECT i.id, i.merchant_id, i.category, i.subtotal - COALESCE(i.discount, 0.0) AS net FROM invoices i WHERE i.settled = 1) SELECT merchant_id, COUNT(*) AS cnt, SUM(net) AS total_net, MAX(net) AS max_net FROM net_inv GROUP BY merchant_id",
        "SELECT id, merchant_id, category, subtotal, ROW_NUMBER() OVER (PARTITION BY category ORDER BY subtotal DESC, id ASC) AS cat_rn, SUM(subtotal) OVER (PARTITION BY merchant_id ORDER BY id ASC) AS running_merch_rev FROM invoices",
        "SELECT id, category, subtotal, RANK() OVER (ORDER BY subtotal DESC) AS global_rnk, LAG(subtotal, 1) OVER (ORDER BY id ASC) AS prev_sub FROM invoices WHERE settled = 1",
        "SELECT id, merchant_id, category, subtotal FROM invoices ORDER BY subtotal DESC, id ASC LIMIT 4",
        "SELECT id, mname, tier, region_id FROM merchants ORDER BY mname ASC, id ASC LIMIT 3 OFFSET 1",
    ]
    for ct in complex_templates:
        for rep in range(3):
            idx += 1
            if idx > 125:
                break
            v_name = f"s10_torture_{idx:03d}"
            q = ct
            if rep == 1 and "LIMIT 4" in q:
                q = q.replace("LIMIT 4", "LIMIT 2")
            elif rep == 2 and "LIMIT 4" in q:
                q = q.replace("LIMIT 4", "LIMIT 6 OFFSET 1")
            views.append((v_name, f"CREATE VIEW {v_name} AS {q};"))

    assert len(views) == 125

    # 10A: Execute on SQLite (Standard Backfill)
    runner.run_batch(
        "Suite 10A (125 Torture-DML & Intermediate-Tx Views on SQLite Standard)",
        schema,
        views,
        seed,
        torture_mutations,
        dialect=Dialect.SQLITE,
    )

    # 10B: Execute on SQLite (Incremental Keyset/Batch Backfill)
    inc_views = [(f"inc_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW inc_{n}")) for n, sql in views]
    runner.run_batch(
        "Suite 10B (125 Torture-DML & Intermediate-Tx Views on SQLite Incremental Backfill)",
        schema,
        inc_views,
        seed,
        torture_mutations,
        dialect=Dialect.SQLITE,
        incremental_backfill=True,
        batch_size=2,
    )

    # 10C: Execute on PostgreSQL Row-Level Triggers (Transpiled to SQLite)
    pg_row_views = [(f"pgr_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW pgr_{n}")) for n, sql in views]
    runner.run_batch(
        "Suite 10C (125 Torture-DML & Intermediate-Tx Views on PostgreSQL Row-Level)",
        schema,
        pg_row_views,
        seed,
        torture_mutations,
        dialect=Dialect.POSTGRES,
        pg_use_transition_tables=False,
    )

    # 10D: Execute on PostgreSQL Statement-Level Transition Tables (Transpiled to SQLite)
    pg_stmt_views = [(f"pgs_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW pgs_{n}")) for n, sql in views]
    runner.run_batch(
        "Suite 10D (125 Torture-DML & Intermediate-Tx Views on PostgreSQL Transition Tables)",
        schema,
        pg_stmt_views,
        seed,
        torture_mutations,
        dialect=Dialect.POSTGRES,
        pg_use_transition_tables=True,
        incremental_backfill=True,
        batch_size=2,
    )

    # 10E: Execute on MySQL Triggers (Transpiled to SQLite)
    my_views = [(f"mys_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW mys_{n}")) for n, sql in views]
    runner.run_batch(
        "Suite 10E (125 Torture-DML & Intermediate-Tx Views on MySQL Dialect)",
        schema,
        my_views,
        seed,
        torture_mutations,
        dialect=Dialect.MYSQL,
        incremental_backfill=True,
        batch_size=2,
    )


# ==============================================================================
# SUITE 11: Comprehensive Unit Tests for All Generator & Manager Code Paths
# ==============================================================================

def run_suite_11_live_engine_50_view_battery_across_all_dialects(runner: MVTestRunner) -> None:
    """
    Executes the 50-view x 35-state verification battery AND `_run_live_manager_lifecycle_check`
    through `runner._open_backend_session(dialect)` and `_run_unified_50_view_battery` across all
    three dialects (`Dialect.SQLITE`, `Dialect.POSTGRES`, `Dialect.MYSQL`).
    When `POSTGRES_DSN`, `MYSQL_DSN`, or `SQLITE_COMPAT_ENGINE` is set, the corresponding dialect
    executes directly against that live database engine while the remaining dialects execute
    through the unified in-memory transpiled adapters.
    """
    dialect_configs: List[Tuple[Dialect, List[Tuple[bool, bool]]]] = [
        (Dialect.SQLITE, [(False, False), (True, False)]),
        (Dialect.POSTGRES, [(False, False), (True, False), (False, True)]),
        (Dialect.MYSQL, [(False, False), (True, False)]),
    ]
    for dialect, mode_list in dialect_configs:
        conn, cur, reset_sql, _, engine_label, close_fn = runner._open_backend_session(
            dialect, suite_tag=f"s11_{dialect.value}"
        )
        eff_reset = reset_sql or "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
        try:
            for inc_mode, use_trans in mode_list:
                v_cnt, c_cnt = _run_unified_50_view_battery(
                    conn,
                    cur,
                    dialect,
                    inc_mode=inc_mode,
                    use_trans=use_trans,
                    label=f"s11_{dialect.value}",
                    reset_sql=eff_reset,
                )
                runner.total_views += v_cnt
                runner.total_assertions += c_cnt
                print(
                    f"  [PASS] Suite 11 (Unified 50-View Battery {dialect.value.upper()} inc={inc_mode} tr={use_trans}) [{engine_label}]: {v_cnt} views, {c_cnt} assertions"
                )
            mgr_checks = _run_live_manager_lifecycle_check(conn, cur, dialect)
            runner.total_assertions += mgr_checks
            print(
                f"  [PASS] Suite 11 (Unified Manager Lifecycle {dialect.value.upper()}) [{engine_label}]: {mgr_checks} assertions"
            )
        finally:
            close_fn()


class TestGeneratorAndManagerCodePaths(unittest.TestCase):
    def test_01_read_only_mv_enforcement_and_manager_lifecycle(self) -> None:
        conn = sqlite3.connect(":memory:")
        mgr = MaterializedViewManager(conn, dialect="sqlite")
        mgr.apply_schema_ddl(
            """
            CREATE TABLE items (id INTEGER PRIMARY KEY, category TEXT, price REAL);
            INSERT INTO items VALUES (1, 'book', 15.0), (2, 'book', 25.0), (3, 'game', 60.0);
            """
        )
        plan = mgr.register_view(
            "CREATE VIEW item_summary AS SELECT category, COUNT(*) AS cnt, SUM(price) AS total FROM items GROUP BY category;"
        )
        self.assertEqual(plan.view_name, "__matview_item_summary")
        self.assertEqual(plan.backing_table_name, "__matview_store_item_summary")

        # Reading from the read-only view works
        rows = conn.execute("SELECT category, cnt, total FROM __matview_item_summary ORDER BY category;").fetchall()
        self.assertEqual(rows, [("book", 2, 40.0), ("game", 1, 60.0)])

        # Direct INSERT / UPDATE / DELETE on the read-only view is blocked!
        for bad_dml in [
            "INSERT INTO __matview_item_summary VALUES ('music', 1, 10.0);",
            "UPDATE __matview_item_summary SET total = 999.0 WHERE category = 'book';",
            "DELETE FROM __matview_item_summary WHERE category = 'game';",
        ]:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(bad_dml)
        conn.close()

    def test_02_incremental_backfill_strategies_and_progress_callback(self) -> None:
        conn = sqlite3.connect(":memory:")
        mgr = MaterializedViewManager(conn, dialect="sqlite")
        mgr.apply_schema_ddl(
            """
            CREATE TABLE logs (id INTEGER PRIMARY KEY, svc TEXT, latency INTEGER);
            INSERT INTO logs VALUES (1, 'auth', 10), (2, 'auth', 20), (3, 'pay', 30), (4, 'pay', 40), (5, 'web', 50);
            """
        )
        # 1. pk_range_fold strategy (algebraic aggregate with single integer PK base table)
        progress_calls: List[Tuple[int, int]] = []
        p1 = mgr.register_view(
            "CREATE VIEW v_pk_fold AS SELECT svc, COUNT(*) AS cnt, SUM(latency) AS s_lat FROM logs GROUP BY svc;",
            incremental_backfill=True,
            batch_size=2,
            on_progress=lambda b, d: progress_calls.append((b, d)),
        )
        self.assertEqual(p1.incremental_backfill_spec.mode, "pk_range_fold")
        self.assertGreaterEqual(len(progress_calls), 2)
        self.assertEqual(
            conn.execute("SELECT svc, cnt, s_lat FROM __matview_v_pk_fold ORDER BY svc;").fetchall(),
            [("auth", 2, 30), ("pay", 2, 70), ("web", 1, 50)],
        )

        # Verify that incremental_backfill=True produces explicit cursor/loop SQL in generated statements
        p_full = generate_materialized_view(
            "CREATE TABLE logs (id INTEGER PRIMARY KEY, svc TEXT, latency INTEGER);",
            "CREATE VIEW v_pk_fold AS SELECT svc, COUNT(*) AS cnt, SUM(latency) AS s_lat FROM logs GROUP BY svc;",
            dialect="sqlite",
            incremental_backfill=False,
        )
        p_inc = generate_materialized_view(
            "CREATE TABLE logs (id INTEGER PRIMARY KEY, svc TEXT, latency INTEGER);",
            "CREATE VIEW v_pk_fold AS SELECT svc, COUNT(*) AS cnt, SUM(latency) AS s_lat FROM logs GROUP BY svc;",
            dialect="sqlite",
            incremental_backfill=True,
            batch_size=2,
        )
        self.assertNotEqual(p_full.all_statements(), p_inc.all_statements())
        self.assertTrue(any("__mv_batches" in s for s in p_inc.get_populate_statements()))

        # 2. group_key_batch strategy (DISTINCT aggregate in Tier 1)
        p2 = mgr.register_view(
            "CREATE VIEW v_grp_batch AS SELECT svc, COUNT(DISTINCT latency) AS dl, SUM(latency) AS s_lat FROM logs GROUP BY svc;",
            incremental_backfill=True,
            batch_size=1,
        )
        self.assertEqual(p2.incremental_backfill_spec.mode, "group_key_batch")
        self.assertEqual(
            conn.execute("SELECT svc, dl, s_lat FROM __matview_v_grp_batch ORDER BY svc;").fetchall(),
            [("auth", 2, 30), ("pay", 2, 70), ("web", 1, 50)],
        )

        # 3. offset_batch strategy (composite PK base table projection)
        mgr.apply_schema_ddl(
            """
            CREATE TABLE comp_items (a INTEGER, b INTEGER, val TEXT, PRIMARY KEY (a, b));
            INSERT INTO comp_items VALUES (1, 1, 'x'), (1, 2, 'y'), (2, 1, 'z');
            """
        )
        p3 = mgr.register_view(
            "CREATE VIEW v_off_batch AS SELECT a, b, val FROM comp_items;",
            incremental_backfill=True,
            batch_size=1,
        )
        self.assertEqual(p3.incremental_backfill_spec.mode, "offset_batch")
        self.assertEqual(
            conn.execute("SELECT a, b, val FROM __matview_v_off_batch ORDER BY a, b;").fetchall(),
            [(1, 1, "x"), (1, 2, "y"), (2, 1, "z")],
        )
        conn.close()

    def test_03_manager_sync_views_bulk_changes_and_schema_migration(self) -> None:
        conn = sqlite3.connect(":memory:")
        mgr = MaterializedViewManager(conn, dialect="sqlite")
        mgr.apply_schema_ddl(
            """
            CREATE TABLE products (id INTEGER PRIMARY KEY, dept TEXT, price REAL);
            INSERT INTO products VALUES (1, 'elec', 100.0), (2, 'elec', 200.0), (3, 'home', 50.0);
            """
        )
        status1 = mgr.sync_views(
            [
                "CREATE VIEW v_dept AS SELECT dept, COUNT(*) AS cnt, SUM(price) AS sum_p FROM products GROUP BY dept;",
                "CREATE VIEW v_expensive AS SELECT id, dept, price FROM products WHERE price >= 100.0;",
            ]
        )
        self.assertEqual(status1, {"__matview_v_dept": "created", "__matview_v_expensive": "created"})

        status2 = mgr.sync_views(
            [
                "CREATE VIEW v_dept AS SELECT dept, COUNT(*) AS cnt, SUM(price) AS sum_p FROM products GROUP BY dept;",
                "CREATE VIEW v_expensive AS SELECT id, dept, price FROM products WHERE price >= 150.0;",
            ],
            prune_removed=False,
        )
        self.assertEqual(status2["__matview_v_dept"], "unchanged")
        self.assertEqual(status2["__matview_v_expensive"], "updated")

        status3 = mgr.sync_views(
            ["CREATE VIEW v_dept AS SELECT dept, COUNT(*) AS cnt, SUM(price) AS sum_p FROM products GROUP BY dept;"],
            prune_removed=True,
        )
        self.assertEqual(status3["__matview_v_expensive"], "dropped")

        with mgr.bulk_changes():
            conn.execute("INSERT INTO products VALUES (4, 'elec', 300.0), (5, 'toys', 25.0);")
            conn.execute("DELETE FROM products WHERE id = 1;")
        self.assertEqual(
            conn.execute("SELECT dept, cnt, sum_p FROM __matview_v_dept ORDER BY dept;").fetchall(),
            [("elec", 2, 500.0), ("home", 1, 50.0), ("toys", 1, 25.0)],
        )

        mgr.apply_schema_ddl("ALTER TABLE products ADD COLUMN note TEXT DEFAULT 'ok';")
        conn.execute("INSERT INTO products (id, dept, price, note) VALUES (6, 'toys', 75.0, 'new');")
        self.assertEqual(
            conn.execute("SELECT dept, cnt, sum_p FROM __matview_v_dept ORDER BY dept;").fetchall(),
            [("elec", 2, 500.0), ("home", 1, 50.0), ("toys", 2, 100.0)],
        )
        conn.close()

    def test_04_engine_limitations_warnings_and_errors(self) -> None:
        schema = "CREATE TABLE t (id INT PRIMARY KEY, grp TEXT, val INT);"
        view = "CREATE VIEW v AS SELECT grp, COUNT(*) AS c FROM t GROUP BY grp;"

        # By default (ignore_limitations=False), MySQL raises EngineLimitationError due to TRUNCATE bypassing triggers
        with self.assertRaises(EngineLimitationError):
            generate_materialized_view(schema, view, dialect="mysql", ignore_limitations=False)

        # With ignore_limitations=True, MySQL emits EngineLimitationWarning and succeeds
        with warnings.catch_warnings(record=True) as w_rec:
            warnings.simplefilter("always", EngineLimitationWarning)
            plan = generate_materialized_view(schema, view, dialect="mysql", ignore_limitations=True)
            self.assertGreaterEqual(len(plan.limitations), 1)
            self.assertGreaterEqual(len(w_rec), 1)
            self.assertIn("CREATE ALGORITHM = TEMPTABLE VIEW", plan.reader_view_ddl)

    def test_05_non_deterministic_functions_and_right_full_join_lowering(self) -> None:
        schema = """
        CREATE TABLE a (id INTEGER PRIMARY KEY, k INTEGER, aval TEXT);
        CREATE TABLE b (id INTEGER PRIMARY KEY, k INTEGER, bval TEXT);
        """
        # Non-deterministic function rejection (raises EngineLimitationError by default; emits warning with ignore_limitations=True)
        with self.assertRaises(EngineLimitationError):
            generate_materialized_view(schema, "CREATE VIEW v_bad AS SELECT id, RANDOM() AS r FROM a;", dialect="sqlite")
        with warnings.catch_warnings(record=True) as w_nd:
            warnings.simplefilter("always", EngineLimitationWarning)
            p_nd = generate_materialized_view(
                schema,
                "CREATE VIEW v_bad AS SELECT id, RANDOM() AS r FROM a;",
                dialect="sqlite",
                ignore_limitations=True,
            )
            self.assertTrue(any(lim.code == "NONDETERMINISTIC_EXPR" for lim in p_nd.limitations))
            self.assertGreaterEqual(len(w_nd), 1)

        # RIGHT JOIN and FULL OUTER JOIN lowering + live execution verification
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        conn.executescript(
            """
            INSERT INTO a VALUES (1, 10, 'a1'), (2, 20, 'a2');
            INSERT INTO b VALUES (100, 20, 'b2'), (200, 30, 'b3');
            """
        )
        p_right = generate_materialized_view(
            schema,
            "CREATE VIEW v_right AS SELECT a.id AS aid, a.aval, b.id AS bid, b.bval FROM a RIGHT JOIN b ON a.k = b.k;",
            dialect="sqlite",
        )
        for st in p_right.all_statements():
            conn.execute(st)
        self.assertEqual(
            conn.execute("SELECT aid, aval, bid, bval FROM __matview_v_right ORDER BY bid;").fetchall(),
            [(2, "a2", 100, "b2"), (None, None, 200, "b3")],
        )

        p_full = generate_materialized_view(
            schema,
            "CREATE VIEW v_full AS SELECT a.k AS ak, a.aval, b.k AS bk, b.bval FROM a FULL OUTER JOIN b ON a.k = b.k;",
            dialect="sqlite",
        )
        for st in p_full.all_statements():
            conn.execute(st)
        conn.execute("INSERT INTO a VALUES (3, 30, 'a3');")
        conn.execute("DELETE FROM b WHERE id = 100;")
        self.assertEqual(
            sorted(conn.execute("SELECT ak, aval, bk, bval FROM __matview_v_full;").fetchall(), key=lambda x: str(x)),
            sorted([(10, "a1", None, None), (20, "a2", None, None), (30, "a3", 30, "b3")], key=lambda x: str(x)),
        )
        conn.close()

    def test_06_cli_interface_end_to_end(self) -> None:
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = os.path.join(tmpdir, "schema.sql")
            view_path = os.path.join(tmpdir, "view.sql")
            with open(schema_path, "w", encoding="utf-8") as f:
                f.write("CREATE TABLE t (id INTEGER PRIMARY KEY, grp TEXT, val INTEGER);\n")
            with open(view_path, "w", encoding="utf-8") as f:
                f.write("CREATE VIEW v_cli AS SELECT grp, COUNT(*) AS cnt, SUM(val) AS sv FROM t GROUP BY grp;\n")

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(
                    [
                        "--schema", schema_path,
                        "--view", view_path,
                        "--dialect", "postgres",
                        "--incremental-backfill",
                        "--batch-size", "500",
                    ]
                )
            self.assertEqual(rc, 0)
            generated = buf.getvalue()
            self.assertIn("__matview_store_v_cli", generated)
            self.assertIn("DO $$", generated)

    def test_07_unified_multi_dialect_test_infrastructure_parity(self) -> None:
        """
        Verifies that the unified test runner (`MVTestRunner._open_backend_session`,
        `MVTestRunner.run_batch`, and `_run_live_manager_lifecycle_check`) executes the
        exact same code path across SQLite, PostgreSQL (row-level & transition tables),
        and MySQL even when running purely in-memory without external servers.
        """
        runner = MVTestRunner()
        schema = "CREATE TABLE u_items (id INTEGER PRIMARY KEY, grp TEXT, val REAL, qty INTEGER);"
        seed = ["INSERT INTO u_items VALUES (1, 'a', 10.0, 2), (2, 'b', 20.0, 4);"]
        views = [
            (
                "v_u_alg",
                "CREATE VIEW v_u_alg AS SELECT grp, COUNT(*) AS c, SUM(val) AS sv, TOTAL(val) AS tv FROM u_items GROUP BY grp;",
            ),
            (
                "v_u_win",
                "CREATE VIEW v_u_win AS SELECT id, grp, val, LAG(qty, 1, -1) OVER (PARTITION BY grp ORDER BY id ASC) AS prev_q FROM u_items;",
            ),
        ]
        mutations = [
            ("ins", "INSERT INTO u_items VALUES (3, 'a', 30.0, 6);"),
            ("tx_begin", "BEGIN;"),
            ("upd", "UPDATE u_items SET val = val + 5.0 WHERE grp = 'a';"),
            ("tx_commit", "COMMIT;"),
        ]
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            for inc_mode in (False, True):
                runner.run_batch(
                    f"UnifiedParityCheck({dialect.value},inc={inc_mode})",
                    schema,
                    views,
                    seed,
                    mutations,
                    dialect=dialect,
                    pg_use_transition_tables=(dialect == Dialect.POSTGRES and inc_mode),
                    incremental_backfill=inc_mode,
                    batch_size=1,
                )
            conn, cur, _, _, _, close_fn = runner._open_backend_session(dialect, suite_tag=f"u_mgr_{dialect.value}")
            try:
                checks = _run_live_manager_lifecycle_check(conn, cur, dialect)
                self.assertGreaterEqual(checks, 14)
            finally:
                close_fn()

    def test_08_pull_schema_from_existing_database_across_all_dialects(self) -> None:
        """
        Exercises pulling base table schemas directly from existing databases across
        SQLite, PostgreSQL, and MySQL via `introspect_database_schemas`,
        `generate_materialized_view(conn, ...)`, `generate_materialized_views(conn, ...)`,
        `MaterializedViewManager.fetch_schema()`, and CLI `--db` without `--schema`.
        """
        runner = MVTestRunner()
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            conn, cur, reset_sql, _, _, close_fn = runner._open_backend_session(
                dialect, suite_tag=f"pull_schema_{dialect.value}"
            )
            try:
                if reset_sql:
                    for r_st in [s.strip() for s in reset_sql.split(";") if s.strip()]:
                        cur.execute(r_st + ";")
                str_t = "VARCHAR(64)" if dialect == Dialect.MYSQL else "TEXT"
                flt_t = "DOUBLE PRECISION" if dialect in (Dialect.POSTGRES, Dialect.MYSQL) else "REAL"
                cur.execute(
                    f"CREATE TABLE parent_orgs (id INTEGER PRIMARY KEY, name {str_t} NOT NULL);"
                )
                cur.execute(
                    f"CREATE TABLE child_members ("
                    f"org_id INTEGER NOT NULL, "
                    f"member_id INTEGER NOT NULL, "
                    f"role {str_t} NOT NULL, "
                    f"dues {flt_t} NOT NULL, "
                    f"PRIMARY KEY (org_id, member_id), "
                    f"FOREIGN KEY (org_id) REFERENCES parent_orgs(id) ON DELETE CASCADE"
                    f");"
                )
                cur.execute("INSERT INTO parent_orgs VALUES (1, 'Alpha'), (2, 'Beta');")
                cur.execute(
                    "INSERT INTO child_members VALUES (1, 10, 'admin', 50.0), (1, 11, 'dev', 30.0), (2, 20, 'dev', 40.0);"
                )

                # 1. Direct schema introspection via introspect_database_schemas
                pulled = introspect_database_schemas(conn, dialect=dialect)
                self.assertIn("parent_orgs", {k.lower() for k in pulled})
                self.assertIn("child_members", {k.lower() for k in pulled})
                cm_schema = next(v for k, v in pulled.items() if k.lower() == "child_members")
                self.assertEqual({c.name.lower() for c in cm_schema.columns.values() if c.primary_key}, {"org_id", "member_id"})
                self.assertIn("ON DELETE CASCADE", cm_schema.fk_cascade_actions)

                # 2. On MySQL, foreign key ON DELETE CASCADE pulled from the live database triggers EngineLimitationError
                if dialect == Dialect.MYSQL:
                    with self.assertRaises(EngineLimitationError):
                        generate_materialized_view(
                            conn,
                            "CREATE VIEW v_pulled_fk AS SELECT role, COUNT(*) AS cnt FROM child_members GROUP BY role;",
                            dialect=dialect,
                            ignore_limitations=False,
                        )

                # 3. Compile & execute single view and multi-view bundle by pulling schema directly from `conn`
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    plans = generate_materialized_views(
                        conn,
                        [
                            "CREATE VIEW v_pulled_role AS SELECT role, COUNT(*) AS cnt, SUM(dues) AS total_dues FROM child_members GROUP BY role;",
                            "CREATE VIEW v_pulled_join AS SELECT p.name AS org_name, COUNT(c.member_id) AS m_cnt, SUM(c.dues) AS org_dues FROM parent_orgs p LEFT JOIN child_members c ON p.id = c.org_id GROUP BY p.name;",
                        ],
                        dialect=dialect,
                        ignore_limitations=True,
                    )
                for p in plans:
                    for st in p.all_statements():
                        cur.execute(st)
                _verify_external_db_mv(
                    cur,
                    "__matview_v_pulled_role",
                    "SELECT role, COUNT(*) AS cnt, SUM(dues) AS total_dues FROM child_members GROUP BY role",
                    f"pull_schema:{dialect.value}:role_init",
                )
                _verify_external_db_mv(
                    cur,
                    "__matview_v_pulled_join",
                    "SELECT p.name AS org_name, COUNT(c.member_id) AS m_cnt, SUM(c.dues) AS org_dues FROM parent_orgs p LEFT JOIN child_members c ON p.id = c.org_id GROUP BY p.name",
                    f"pull_schema:{dialect.value}:join_init",
                )

                # 4. Alter table in database, pull updated schema from `conn`, and compile a view using the new column
                cur.execute("ALTER TABLE parent_orgs ADD COLUMN bonus INTEGER DEFAULT 5;")
                cur.execute("INSERT INTO parent_orgs (id, name, bonus) VALUES (3, 'Gamma', 15);")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", EngineLimitationWarning)
                    p_after_alter = generate_materialized_view(
                        conn,
                        "CREATE VIEW v_pulled_bonus AS SELECT name, bonus FROM parent_orgs WHERE bonus >= 10;",
                        dialect=dialect,
                        ignore_limitations=True,
                    )
                for st in p_after_alter.all_statements():
                    cur.execute(st)
                _verify_external_db_mv(
                    cur,
                    "__matview_v_pulled_bonus",
                    "SELECT name, bonus FROM parent_orgs WHERE bonus >= 10",
                    f"pull_schema:{dialect.value}:after_alter",
                )
            finally:
                close_fn()

        # 5. CLI `--db <sqlite_file>` without `--schema` (pulls schema from existing SQLite file)
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "existing.db")
            sq_c = sqlite3.connect(db_path)
            sq_c.execute("CREATE TABLE cli_sensors (id INTEGER PRIMARY KEY, zone TEXT NOT NULL, temp REAL NOT NULL);")
            sq_c.execute("INSERT INTO cli_sensors VALUES (1, 'north', 21.5), (2, 'north', 23.5);")
            sq_c.commit()
            sq_c.close()

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(
                    [
                        "--db", db_path,
                        "--view", "CREATE VIEW v_zone_temp AS SELECT zone, COUNT(*) AS n, AVG(temp) AS avg_t FROM cli_sensors GROUP BY zone;",
                        "--apply",
                    ]
                )
            self.assertEqual(rc, 0)
            sq_c2 = sqlite3.connect(db_path)
            rows = sq_c2.execute("SELECT zone, n, avg_t FROM __matview_v_zone_temp;").fetchall()
            sq_c2.close()
            self.assertEqual(rows, [("north", 2, 22.5)])

    def test_09_differential_fuzzer_multi_seed_and_reproducibility(self) -> None:
        """
        Tests that DifferentialSQLFuzzer is deterministically reproducible for any given seed
        and verifies additional multi-seed random schemas, views, and DML sequences across
        SQLite, PostgreSQL, and MySQL.
        """
        f1 = DifferentialSQLFuzzer(seed=987654321)
        f2 = DifferentialSQLFuzzer(seed=987654321)
        for d in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            self.assertEqual(f1.generate_scenario(1, d), f2.generate_scenario(1, d))

        local_runner = MVTestRunner()
        for extra_seed in (1337, 424242):
            fz = DifferentialSQLFuzzer(seed=extra_seed)
            for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
                schema, views, seed_sqls, muts = fz.generate_scenario(1, dialect)
                local_runner.run_batch(
                    f"UnitFuzz(seed={extra_seed},{dialect.value})",
                    schema,
                    views[:12],
                    seed_sqls,
                    muts[:12],
                    dialect=dialect,
                )
        self.assertGreaterEqual(local_runner.total_views, 72)

    def test_10_fuzz_regression_corpus_invariants(self) -> None:
        """
        Verifies the structural invariants of every entry in the fuzzing regression corpus:
        - Corpus #1 compiles as ALGEBRAIC_DELTA with pure-SQLite bitwise expressions.
        - Corpus #2 includes deterministic ORDER BY in every offset_batch incremental backfill loop.
        - Corpus #3 wraps MySQL BIT_OR/BIT_XOR in COALESCE(..., 0) and BIT_AND in COALESCE(..., 18446744073709551615).
        - Corpus #4 parses FILTER (WHERE ...) with balanced parentheses and compiles as ALGEBRAIC_DELTA.
        """
        corpus = {c["id"]: c for c in get_fuzz_regression_corpus()}
        self.assertGreaterEqual(len(corpus), 4)

        # Corpus 1: pure-SQLite bitwise expressions in group_key_batch
        c1 = corpus["corpus_01_sqlite_bitwise_distinct_and_delete_recompute"]
        p1 = generate_materialized_view(
            c1["schema"], c1["views"][0][1], dialect=Dialect.SQLITE, incremental_backfill=True, batch_size=2
        )
        self.assertEqual(p1.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assertIn(">> 0) & 1", "\n".join(p1.initial_populate_sqls))

        # Corpus 2: deterministic ORDER BY in PostgreSQL and MySQL offset_batch loops
        c2 = corpus["corpus_02_pg_mysql_offset_join_and_group_order" if "corpus_02_pg_mysql_offset_join_and_group_order" in corpus else "corpus_02_pg_mysql_offset_batch_left_join_and_group_order"]
        for d in (Dialect.POSTGRES, Dialect.MYSQL):
            for _, vddl in c2["views"]:
                p2 = generate_materialized_view(
                    c2["schema"], vddl, dialect=d, incremental_backfill=True, batch_size=2, ignore_limitations=True
                )
                for pop_sql in p2.initial_populate_sqls:
                    if "OFFSET" in pop_sql.upper():
                        self.assertIn("ORDER BY 1, 2", pop_sql.upper())

        # Corpus 3: MySQL empty-set bitwise identity in reader view
        c3 = corpus["corpus_03_mysql_empty_set_and_all_null_bitwise_identity"]
        p3 = generate_materialized_view(
            c3["schema"], c3["views"][0][1], dialect=Dialect.MYSQL, ignore_limitations=True
        )
        self.assertIn("COALESCE(bx, 0) AS bx", p3.reader_view_ddl)
        self.assertIn("COALESCE(bo, 0) AS bo", p3.reader_view_ddl)
        self.assertIn("COALESCE(ba, 18446744073709551615) AS ba", p3.reader_view_ddl)

        # Corpus 4: FILTER (WHERE ...) compiles as ALGEBRAIC_DELTA
        c4 = corpus["corpus_04_filter_where_greedy_regex_algebraic_delta"]
        for d in (Dialect.SQLITE, Dialect.POSTGRES):
            p4 = generate_materialized_view(c4["schema"], c4["views"][0][1], dialect=d)
            self.assertEqual(p4.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)

        # Corpus 5: PostgreSQL transition-table function & trigger names strictly <= 62 chars and distinct
        c5 = corpus["corpus_05_pg_63char_identifier_truncation_collision"]
        p5 = generate_materialized_view(
            c5["schema"], c5["views"][0][1], dialect=Dialect.POSTGRES, pg_use_transition_tables=True
        )
        self.assertEqual(len(set(p5.function_names)), len(p5.function_names))
        self.assertEqual(len(set(p5.trigger_names)), len(p5.trigger_names))
        for fn_n in p5.function_names:
            self.assertLessEqual(len(fn_n), 62, f"Function name too long: {fn_n}")
        for tr_n in p5.trigger_names:
            self.assertLessEqual(len(tr_n), 62, f"Trigger name too long: {tr_n}")

        # Corpus 6: MariaDB NOT IN (SELECT ...) lowered to NOT EXISTS (...) in MySQL trigger SQL
        c6 = corpus["corpus_06_mariadb_not_in_subquery_empty_transition_with_null_lhs"]
        p6 = generate_materialized_view(
            c6["schema"], c6["views"][0][1], dialect=Dialect.MYSQL, ignore_limitations=True
        )
        for tr_ddl in p6.trigger_ddls:
            self.assertNotIn("NOT IN", tr_ddl.upper())
            self.assertIn("NOT EXISTS", tr_ddl.upper())

    def test_11_independent_code_reviews_urgent_fixes(self) -> None:
        """
        Verifies all 11 urgent correctness, precision, concurrency, and robustness fixes
        identified across the 3 independent code reviews (reviews/1.md, reviews/2.md, reviews/3.md).
        """
        schema = """
            CREATE TABLE rev_items (
                tenant_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                mask_64 INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, item_id)
            );
        """

        # 1. Batch-size substitution in get_populate_statements must not corrupt user literals matching batch_size
        p_batch = generate_materialized_view(
            schema,
            "CREATE VIEW v_lit500 AS SELECT tenant_id, SUM(price) AS sp FROM rev_items WHERE price > 500 GROUP BY tenant_id;",
            dialect=Dialect.POSTGRES,
            incremental_backfill=True,
            batch_size=500,
        )
        stmts_100 = p_batch.get_populate_statements(incremental_backfill=True, batch_size=100)
        joined_100 = "\n".join(stmts_100)
        self.assertIn("price > 500", joined_100)
        self.assertIn("__mv_batch_size BIGINT := 100", joined_100)

        # 2. _strip_sql_comments & _split_top_level handle PostgreSQL $$...$$ dollar-quoting and backtick identifiers
        raw_dq = "SELECT $$keep -- not a comment; /* also not */$$ AS a, `col--2,x` AS b -- strip this\nFROM t;"
        stripped_dq = _strip_sql_comments(raw_dq)
        self.assertIn("$$keep -- not a comment; /* also not */$$", stripped_dq)
        self.assertIn("`col--2,x`", stripped_dq)
        self.assertNotIn("strip this", stripped_dq)
        parts_dq = _split_top_level("$$a,b$$, `c,d`, 'e,f'", ",")
        self.assertEqual(len(parts_dq), 3)

        # 3. SELECT * and SELECT DISTINCT * on single-table views expand into Tier 1 ALGEBRAIC_DELTA
        p_star = generate_materialized_view(
            schema,
            "CREATE VIEW v_star AS SELECT * FROM rev_items WHERE price >= 10.0;",
            dialect=Dialect.SQLITE,
        )
        self.assertEqual(p_star.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)
        self.assertEqual(p_star.user_columns, ["tenant_id", "item_id", "category", "price", "mask_64"])
        # And uses native composite primary key (tenant_id, item_id) without __mv_pk string concatenation
        self.assertIn("PRIMARY KEY (tenant_id, item_id)", p_star.backing_table_ddl)
        self.assertNotIn("__mv_pk ", p_star.backing_table_ddl)

        p_dist_star = generate_materialized_view(
            schema,
            "CREATE VIEW v_dist_star AS SELECT DISTINCT * FROM rev_items WHERE price >= 10.0;",
            dialect=Dialect.SQLITE,
        )
        self.assertEqual(p_dist_star.strategy, MaintenanceStrategy.ALGEBRAIC_DELTA)

        # 4. Native composite PKs on MySQL NOT NULL group columns and partial-PK projection views
        p_my_grp = generate_materialized_view(
            schema,
            "CREATE VIEW v_my_grp AS SELECT tenant_id, category, SUM(price) AS sp FROM rev_items GROUP BY tenant_id, category;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertIn("PRIMARY KEY (tenant_id, category)", p_my_grp.backing_table_ddl)
        self.assertNotIn("__mv_pk ", p_my_grp.backing_table_ddl)

        p_proj_partial = generate_materialized_view(
            schema,
            "CREATE VIEW v_proj_partial AS SELECT tenant_id, price FROM rev_items WHERE price > 0;",
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("PRIMARY KEY (tenant_id, __mv_pk_item_id)", p_proj_partial.backing_table_ddl)

        # 5. 63-bit SQLite BIT_AND, BIT_OR, BIT_XOR and nested parentheses in expressions
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema)
        v_bit64 = (
            "CREATE VIEW v_bit64 AS SELECT tenant_id, "
            "BIT_AND(COALESCE(mask_64, 0)) AS ba, "
            "BIT_OR(COALESCE(mask_64, 0)) AS bo, "
            "BIT_XOR(COALESCE(mask_64, 0)) AS bx "
            "FROM rev_items GROUP BY tenant_id;"
        )
        install_materialized_view_sqlite(conn, v_bit64)
        v1 = (1 << 52) | (1 << 40) | 7
        v2 = (1 << 60) | (1 << 40) | 3
        conn.execute("INSERT INTO rev_items VALUES (1, 10, 'A', 20.0, ?);", (v1,))
        conn.execute("INSERT INTO rev_items VALUES (1, 11, 'A', 30.0, ?);", (v2,))
        row_mv = conn.execute("SELECT ba, bo, bx FROM __matview_v_bit64 WHERE tenant_id = 1;").fetchone()
        self.assertEqual(row_mv, (v1 & v2, v1 | v2, v1 ^ v2))
        conn.execute("DELETE FROM rev_items WHERE tenant_id = 1 AND item_id = 10;")
        row_mv_after = conn.execute("SELECT ba, bo, bx FROM __matview_v_bit64 WHERE tenant_id = 1;").fetchone()
        self.assertEqual(row_mv_after, (v2, v2, v2))

        # 6. High-precision NUMERIC / DECIMAL accumulation for variance & stddev on Postgres & MySQL
        p_pg_var = generate_materialized_view(
            schema,
            "CREATE VIEW v_pg_var AS SELECT tenant_id, SUM(price) AS s, VAR_POP(price) AS vp, STDDEV_SAMP(price) AS ss FROM rev_items GROUP BY tenant_id;",
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("NUMERIC", p_pg_var.backing_table_ddl)
        self.assertIn("::NUMERIC", "\n".join(p_pg_var.trigger_ddls))

        p_my_var = generate_materialized_view(
            schema,
            "CREATE VIEW v_my_var AS SELECT tenant_id, SUM(price) AS s, VAR_POP(price) AS vp FROM rev_items GROUP BY tenant_id;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertIn("DECIMAL(65, 12)", p_my_var.backing_table_ddl)

        # 7. strict=True raises StrictCompilationError on fallback to Tier 2 or Tier 3
        with self.assertRaises(StrictCompilationError):
            generate_materialized_view(
                schema,
                "CREATE VIEW v_t3 AS SELECT tenant_id, item_id, ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY price) AS rn FROM rev_items;",
                dialect=Dialect.SQLITE,
                strict=True,
            )
        with self.assertRaises(StrictCompilationError):
            generate_materialized_view(
                schema,
                "CREATE VIEW v_t2 AS SELECT tenant_id, GROUP_CONCAT(category) AS gc FROM rev_items GROUP BY tenant_id;",
                dialect=Dialect.SQLITE,
                strict=True,
            )

        # 8, 9, 10, 11. Manager exception-safe bulk_changes(), check_schema_drift(), repair_view(), and CLI flags
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            mgr = MaterializedViewManager(f"sqlite:///{tmp.name}")
            mgr.apply_schema_ddl(schema, auto_sync_views=False)
            mgr.conn.execute("INSERT INTO rev_items VALUES (1, 1, 'X', 100.0, 15);")
            mgr.conn.commit()
            mgr.register_view("CREATE VIEW v_star AS SELECT * FROM rev_items WHERE price >= 50.0;")
            self.assertEqual(mgr.check_schema_drift(), {"__matview_v_star": "ok"})

            # Exception inside bulk_changes() rolls back dirty DML and restores triggers cleanly
            with self.assertRaises(RuntimeError):
                with mgr.bulk_changes():
                    mgr.conn.execute("INSERT INTO rev_items VALUES (1, 2, 'Y', 200.0, 31);")
                    raise RuntimeError("simulated bulk failure")
            rows_after_abort = mgr.conn.execute("SELECT item_id, price FROM __matview_v_star;").fetchall()
            self.assertEqual(rows_after_abort, [(1, 100.0)])
            # And triggers are restored so subsequent normal DML works
            mgr.conn.execute("INSERT INTO rev_items VALUES (1, 3, 'Z', 300.0, 63);")
            mgr.conn.commit()
            rows_after_ins = mgr.conn.execute("SELECT item_id, price FROM __matview_v_star ORDER BY item_id;").fetchall()
            self.assertEqual(rows_after_ins, [(1, 100.0), (3, 300.0)])

            # Out-of-band schema change is detected by check_schema_drift() and repaired by repair_view()
            mgr.conn.execute("ALTER TABLE rev_items ADD COLUMN extra_note TEXT;")
            mgr.conn.commit()
            self.assertEqual(mgr.check_schema_drift(), {"__matview_v_star": "drifted"})
            rc_drift = cli_main(["--db", tmp.name, "--check-drift"])
            self.assertEqual(rc_drift, 1)
            rc_rep = cli_main(["--db", tmp.name, "--repair"])
            self.assertEqual(rc_rep, 0)
            self.assertEqual(mgr.check_schema_drift(), {"__matview_v_star": "ok"})
            mgr.conn.close()

    def test_12_engine_and_version_autodetection_and_subdialect_codegen(self) -> None:
        """
        Tests engine & version parsing (`parse_engine_version`), live database connection
        autodetection (`detect_engine_version`), subdialect differentiation (`sqlite`,
        `postgres`, `mysql`, `mariadb`), and version-specific optimal code generation.
        """
        # 1. parse_engine_version across diverse strings & subdialects
        ev_pg17 = parse_engine_version("postgres:17.2")
        self.assertEqual(ev_pg17.dialect, Dialect.POSTGRES)
        self.assertEqual(ev_pg17.subdialect, EngineSubdialect.POSTGRES)
        self.assertEqual(ev_pg17.version, (17, 2, 0))
        self.assertTrue(ev_pg17.supports_pg_transition_tables)
        self.assertTrue(ev_pg17.supports_pg_nulls_not_distinct)

        ev_pg14 = parse_engine_version("postgresql-14.11")
        self.assertEqual(ev_pg14.version, (14, 11, 0))
        self.assertTrue(ev_pg14.supports_pg_transition_tables)
        self.assertFalse(ev_pg14.supports_pg_nulls_not_distinct)

        ev_pg96 = parse_engine_version("pg9.6.24")
        self.assertEqual(ev_pg96.version, (9, 6, 24))
        self.assertFalse(ev_pg96.supports_pg_transition_tables)

        ev_maria = parse_engine_version("5.5.5-10.11.8-MariaDB-1:10.11.8+maria~ubu2204")
        self.assertEqual(ev_maria.dialect, Dialect.MYSQL)
        self.assertEqual(ev_maria.subdialect, EngineSubdialect.MARIADB)
        self.assertTrue(ev_maria.is_mariadb)
        self.assertEqual(ev_maria.version, (10, 11, 8))
        self.assertTrue(ev_maria.supports_mysql_window_functions)

        ev_my84 = parse_engine_version("mysql:8.4.2")
        self.assertEqual(ev_my84.dialect, Dialect.MYSQL)
        self.assertEqual(ev_my84.subdialect, EngineSubdialect.MYSQL)
        self.assertTrue(ev_my84.is_mysql)
        self.assertEqual(ev_my84.version, (8, 4, 2))
        self.assertTrue(ev_my84.supports_mysql_row_alias_upsert)

        ev_my57 = parse_engine_version("mysql:5.7.44")
        self.assertFalse(ev_my57.supports_mysql_window_functions)
        self.assertFalse(ev_my57.supports_mysql_row_alias_upsert)

        ev_sq320 = parse_engine_version("sqlite:3.20.0")
        self.assertFalse(ev_sq320.supports_sqlite_upsert)
        self.assertFalse(ev_sq320.supports_sqlite_window_functions)

        # 2. Live connection autodetection via detect_engine_version() without hint_dialect
        sq_conn = sqlite3.connect(":memory:")
        det_sq = detect_engine_version(sq_conn)
        self.assertEqual(det_sq.dialect, Dialect.SQLITE)
        self.assertEqual(det_sq.subdialect, EngineSubdialect.SQLITE)
        self.assertEqual(det_sq.version, sqlite3.sqlite_version_info[:3])
        sq_conn.close()

        tr_pg17 = TranspiledConnectionAdapter(Dialect.POSTGRES, engine_version="postgres:17.2")
        det_pg17 = detect_engine_version(tr_pg17)
        self.assertEqual(det_pg17.dialect, Dialect.POSTGRES)
        self.assertEqual(det_pg17.subdialect, EngineSubdialect.POSTGRES)
        self.assertEqual(det_pg17.version, (17, 2, 0))

        tr_maria11 = TranspiledConnectionAdapter(Dialect.MYSQL, engine_version="11.4.3-MariaDB-ubu2404")
        det_maria11 = detect_engine_version(tr_maria11)
        self.assertEqual(det_maria11.dialect, Dialect.MYSQL)
        self.assertEqual(det_maria11.subdialect, EngineSubdialect.MARIADB)
        self.assertEqual(det_maria11.version, (11, 4, 3))

        # Also test a generic DB-API 2.0 connection wrapper with NO .dialect attribute (probing via SQL)
        class _OpaqueDBWrapper:
            def __init__(self, inner: Any) -> None:
                self._inner = inner
            def cursor(self) -> Any:
                return self._inner.cursor()
            def commit(self) -> None:
                self._inner.commit()
            def rollback(self) -> None:
                self._inner.rollback()

        opaque_maria = _OpaqueDBWrapper(tr_maria11)
        det_opaque = detect_engine_version(opaque_maria)
        self.assertEqual(det_opaque.dialect, Dialect.MYSQL)
        self.assertEqual(det_opaque.subdialect, EngineSubdialect.MARIADB)
        self.assertEqual(det_opaque.version, (11, 4, 3))

        # 3. Version-specific code generation differences
        schema = """
            CREATE TABLE ev_sales (
                id INTEGER PRIMARY KEY,
                region VARCHAR(64),
                amount DOUBLE PRECISION NOT NULL
            );
        """
        view_sql = "CREATE VIEW v_ev_reg AS SELECT region, COUNT(*) AS c, SUM(amount) AS s FROM ev_sales GROUP BY region;"

        # PostgreSQL 14 vs PostgreSQL 17 on nullable GROUP BY column (`UNIQUE ... NULLS NOT DISTINCT`)
        p_pg14 = generate_materialized_view(schema, view_sql, engine_version="postgres:14.5")
        self.assertEqual(p_pg14.engine_version.version, (14, 5, 0))
        self.assertTrue(any("CREATE INDEX __matview_idx_v_ev_reg__grp ON" in d for d in p_pg14.index_ddls))
        self.assertFalse(any("NULLS NOT DISTINCT" in d for d in p_pg14.index_ddls))

        p_pg17 = generate_materialized_view(schema, view_sql, engine_version="postgres:17.1")
        self.assertEqual(p_pg17.engine_version.version, (17, 1, 0))
        self.assertTrue(
            any("CREATE UNIQUE INDEX __matview_idx_v_ev_reg__grp ON __matview_store_v_ev_reg (region) NULLS NOT DISTINCT;" in d for d in p_pg17.index_ddls)
        )
        self.assertIn("Engine: postgres 17.1.0", p_pg17.render_sql_script())

        # PostgreSQL 9.6 automatically disables transition tables and falls back to FOR EACH ROW triggers
        p_pg96 = generate_materialized_view(
            schema, view_sql, engine_version="postgres:9.6", pg_use_transition_tables=True
        )
        self.assertNotIn("REFERENCING NEW TABLE", "\n".join(p_pg96.trigger_ddls))
        self.assertIn("FOR EACH ROW EXECUTE FUNCTION", "\n".join(p_pg96.trigger_ddls))

        # SQLite 3.20 (before 3.24 UPSERT) automatically selects GROUP_SCOPED_DELTA instead of ALGEBRAIC_DELTA
        p_sq320 = generate_materialized_view(schema, view_sql, engine_version="sqlite:3.20.0")
        self.assertEqual(p_sq320.strategy, MaintenanceStrategy.GROUP_SCOPED_DELTA)
        self.assertNotIn("ON CONFLICT", "\n".join(p_sq320.trigger_ddls))

        # Window functions on MySQL 5.7 or SQLite 3.24 raise EngineLimitationError
        win_view = "CREATE VIEW v_win AS SELECT id, region, ROW_NUMBER() OVER (PARTITION BY region ORDER BY id) AS rn FROM ev_sales;"
        with self.assertRaises(EngineLimitationError) as ctx_my57:
            generate_materialized_view(schema, win_view, engine_version="mysql:5.7")
        self.assertIn("MYSQL_VERSION_NO_WINDOW_FUNC", str(ctx_my57.exception))

        with self.assertRaises(EngineLimitationError) as ctx_sq324:
            generate_materialized_view(schema, win_view, engine_version="sqlite:3.24.0")
        self.assertIn("SQLITE_VERSION_NO_WINDOW_FUNC", str(ctx_sq324.exception))

        # 4. MaterializedViewManager autodetects dialect & version when no dialect is passed
        mgr_pg17 = MaterializedViewManager(tr_pg17)
        self.assertEqual(mgr_pg17.dialect, Dialect.POSTGRES)
        self.assertEqual(mgr_pg17.subdialect, EngineSubdialect.POSTGRES)
        self.assertEqual(mgr_pg17.engine_version.version, (17, 2, 0))
        mgr_pg17.apply_schema_ddl(schema, auto_sync_views=False)
        tr_pg17.execute("INSERT INTO ev_sales VALUES (1, 'EU', 10.0), (2, NULL, 25.0);")
        plan_reg = mgr_pg17.register_view(view_sql)
        self.assertEqual(plan_reg.engine_version.version, (17, 2, 0))
        self.assertTrue(any("NULLS NOT DISTINCT" in d for d in plan_reg.index_ddls))
        tr_pg17.execute("INSERT INTO ev_sales VALUES (3, NULL, 15.0);")
        rows_pg17 = tr_pg17.execute("SELECT region, c, s FROM __matview_v_ev_reg ORDER BY c;").fetchall()
        self.assertEqual(rows_pg17, [("EU", 1, 10.0), (None, 2, 40.0)])
        tr_pg17.close()

        mgr_maria = MaterializedViewManager(tr_maria11, ignore_limitations=True)
        self.assertEqual(mgr_maria.dialect, Dialect.MYSQL)
        self.assertEqual(mgr_maria.subdialect, EngineSubdialect.MARIADB)
        self.assertEqual(mgr_maria.engine_version.version, (11, 4, 3))
        mgr_maria.apply_schema_ddl(schema, auto_sync_views=False)
        plan_maria = mgr_maria.register_view(view_sql)
        self.assertEqual(plan_maria.subdialect, EngineSubdialect.MARIADB)
        self.assertIn("Engine: mariadb 11.4.3", plan_maria.render_sql_script())
        tr_maria11.close()

        # 5. CLI --engine-version and versioned --dialect flags
        rc_ev = cli_main([
            "--schema", schema,
            "--view", view_sql,
            "--engine-version", "postgres:17.2",
        ])
        self.assertEqual(rc_ev, 0)
        rc_maria = cli_main([
            "--schema", schema,
            "--view", view_sql,
            "--dialect", "mariadb:11.4",
            "--ignore-limitations",
        ])
        self.assertEqual(rc_maria, 0)

    def test_13_p0_roadmap_features_complete(self) -> None:
        """
        Verifies all P0 roadmap features and their combinations across SQLite, PostgreSQL,
        MySQL, and MariaDB:
          1. AST-Backed SQL Parsing, Formatting & Dialect Identifier Quoting (`tokenize_sql`,
             `quote_sql_identifier`, `format_sql`, reserved keywords `"order"`, `"group"`,
             `"select"`, `"where"`, and identifiers with spaces `"unit price"`, `"unit avg"`).
          2. Welford Running-Variance (`__mv_m2_*`) & Chan Parallel Merge for IEEE-754 Float
             Engines (`VAR_POP`, `VAR_SAMP`, `STDDEV_POP`, `STDDEV_SAMP` on `1e9` and `1e12`
             offset values under single-row DML, chunked backfills, and Joins).
          3. Snapshot-Isolated & Watermark-Gated Chunked Incremental Backfills
             (`incremental_backfill=True`, `backfill_cursor_pk`, `backfill_max_pk`,
             `backfill_active`, concurrent DML between batches, and `snapshot_isolation=True`).
          4. Native Composite Keys & Leading/Secondary Group-Column Indexes for Joins,
             `UNION ALL` Z-set views, `UNION ALL` aggregate views, and PostgreSQL 15+
             `UNIQUE ... NULLS NOT DISTINCT`.
          5. Session-Scoped & Shadow-Table Swap `bulk_changes()` Isolation
             (`SET LOCAL matview.skip_triggers = 'on'`, `SET @__matview_skip_triggers = 1`,
             transactional `SAVEPOINT` rollback, and `use_shadow_swap=True`).
        """
        import pathlib as _pathlib
        import sqlglot as _sg
        import matview_sync as _mv_mod
        from matview_sync import (
            SQLTokenType,
            format_sql,
            parse_sql_ast,
            parse_view_ast,
            quote_sql_identifier,
            tokenize_sql,
        )

        # --- 1. Required `sqlglot` dependency, `requirements.txt`, AST, Formatter & Quoting ---
        self.assertFalse(hasattr(_mv_mod, "HAS_SQLGLOT"))
        self.assertIs(_mv_mod.sqlglot, _sg)
        req_file = _pathlib.Path(__file__).resolve().parent / "requirements.txt"
        self.assertTrue(req_file.is_file(), "requirements.txt must exist in repository root")
        req_contents = req_file.read_text()
        for expected_dep in ("sqlglot", "psycopg2-binary", "pymysql", "cryptography"):
            self.assertIn(expected_dep, req_contents)
        ci_file = _pathlib.Path(__file__).resolve().parent / ".github" / "workflows" / "ci.yml"
        if ci_file.is_file():
            self.assertGreaterEqual(ci_file.read_text().count("-r requirements.txt"), 7)

        sg_expr = parse_sql_ast('SELECT "group", SUM("select") AS "unit avg" FROM "order" WHERE "select" > 10 GROUP BY "group"')
        self.assertIsInstance(sg_expr, _sg.exp.Select)
        self.assertEqual([e.output_name for e in sg_expr.expressions], ["group", "unit avg"])
        v_ast = parse_view_ast('CREATE VIEW v_ast_test AS SELECT "group", SUM("select") AS "unit avg" FROM "order" GROUP BY "group"')
        self.assertIsInstance(v_ast.sqlglot_ast, _sg.exp.Create)
        self.assertEqual(v_ast.group_by_exprs, ('"group"',))

        self.assertEqual(quote_sql_identifier("orders", Dialect.SQLITE), "orders")
        self.assertEqual(quote_sql_identifier("order", Dialect.SQLITE), '"order"')
        self.assertEqual(quote_sql_identifier("group", Dialect.POSTGRES), '"group"')
        self.assertEqual(quote_sql_identifier("select", Dialect.MYSQL), "`select`")
        self.assertEqual(quote_sql_identifier("unit price", Dialect.SQLITE), '"unit price"')
        self.assertEqual(quote_sql_identifier("unit price", Dialect.MYSQL), "`unit price`")

        toks = tokenize_sql('SELECT "group", SUM("select") AS "unit avg" FROM "order" WHERE "select" > 10;')
        self.assertTrue(any(t.kind == SQLTokenType.QUOTED_IDENT and t.unquoted == "unit avg" for t in toks))
        formatted = format_sql("SELECT   a,   b   FROM   t;", Dialect.POSTGRES)
        self.assertIn("SELECT a, b FROM t;", formatted)

        for d_mode in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            if d_mode == Dialect.SQLITE:
                conn_q: Any = sqlite3.connect(":memory:")
                _register_sqlite_compat_udfs(conn_q)
            else:
                conn_q = TranspiledConnectionAdapter(d_mode)
            mgr_q = MaterializedViewManager(conn_q, dialect=d_mode, ignore_limitations=True)
            mgr_q.apply_schema_ddl(
                '''
                CREATE TABLE "order" (
                    "id" INTEGER PRIMARY KEY,
                    "group" VARCHAR(32) NOT NULL,
                    "select" INTEGER NOT NULL,
                    "unit price" DOUBLE PRECISION NOT NULL
                );
                ''',
                auto_sync_views=False,
            )
            conn_q.execute("""INSERT INTO "order" VALUES (1, 'A', 10, 1.5), (2, 'A', 20, 2.5), (3, 'B', 5, 4.0);""")
            plan_q = mgr_q.register_view(
                '''
                CREATE VIEW "order_summary" AS
                SELECT "group", COUNT(*) AS "select", SUM("select") AS "where", AVG("unit price") AS "unit avg"
                FROM "order"
                WHERE "select" > 0
                GROUP BY "group";
                '''
            )
            if d_mode == Dialect.MYSQL:
                self.assertIn("`group`", plan_q.backing_table_ddl)
                self.assertIn("`select`", plan_q.backing_table_ddl)
            else:
                self.assertIn('"group"', plan_q.backing_table_ddl)
                self.assertIn('"select"', plan_q.backing_table_ddl)

            conn_q.execute("""INSERT INTO "order" VALUES (4, 'A', 30, 5.0);""")
            conn_q.execute('UPDATE "order" SET "select" = 15, "unit price" = 3.5 WHERE "id" = 1;')
            conn_q.execute('DELETE FROM "order" WHERE "id" = 3;')
            mv_q_rows = conn_q.execute('SELECT * FROM __matview_order_summary ORDER BY "group";').fetchall()
            self.assertEqual(len(mv_q_rows), 1)
            self.assertEqual(mv_q_rows[0][0], "A")
            self.assertEqual(mv_q_rows[0][1], 3)
            self.assertEqual(mv_q_rows[0][2], 65)
            self.assertAlmostEqual(float(mv_q_rows[0][3]), 11.0 / 3.0, places=5)
            if hasattr(conn_q, "close"):
                conn_q.close()

        # --- 2. Welford Running-Variance (M2) & Chan Parallel Merge on High-Offset Floats ---
        conn_w = sqlite3.connect(":memory:")
        _register_sqlite_compat_udfs(conn_w)
        conn_w.execute(
            "CREATE TABLE sensors (id INTEGER PRIMARY KEY, grp TEXT NOT NULL, reading REAL);"
        )
        conn_w.executemany(
            "INSERT INTO sensors VALUES (?, ?, ?);",
            [
                (1, "g1", 1e9 + 1.0),
                (2, "g1", 1e9 + 2.0),
                (3, "g1", 1e9 + 3.0),
                (4, "g1", 1e9 + 4.0),
                (5, "g2", 1e12 + 10.0),
                (6, "g2", 1e12 + 20.0),
                (7, "g2", 1e12 + 30.0),
            ],
        )
        mgr_w = MaterializedViewManager(conn_w)
        # Test chunked incremental backfill (Chan parallel M2 merge across chunks of size 2)
        plan_w = mgr_w.register_view(
            """
            CREATE VIEW sensor_stats AS
            SELECT grp,
                   COUNT(reading) AS n,
                   VAR_POP(reading) AS vp,
                   VAR_SAMP(reading) AS vs,
                   STDDEV_POP(reading) AS sp,
                   STDDEV_SAMP(reading) AS ss
            FROM sensors
            GROUP BY grp;
            """,
            incremental_backfill=True,
            batch_size=2,
        )
        self.assertIn("__mv_m2_", plan_w.backing_table_ddl)
        r_g1_init = conn_w.execute("SELECT n, vp, vs, sp, ss FROM __matview_sensor_stats WHERE grp = 'g1'").fetchone()
        self.assertEqual(r_g1_init[0], 4)
        self.assertAlmostEqual(r_g1_init[1], 1.25, places=5)
        self.assertAlmostEqual(r_g1_init[2], 5.0 / 3.0, places=5)

        # Single-row DELETE, INSERT, UPDATE transitions on high-offset values
        conn_w.execute("DELETE FROM sensors WHERE id = 4;")
        conn_w.execute("INSERT INTO sensors VALUES (8, 'g2', 1e12 + 40.0);")
        conn_w.execute("DELETE FROM sensors WHERE id = 8;")
        conn_w.execute("UPDATE sensors SET reading = 1e9 + 10.0 WHERE id = 3;")
        conn_w.execute("UPDATE sensors SET reading = 1e9 + 3.0 WHERE id = 3;")

        r_g1 = conn_w.execute("SELECT n, vp, vs, sp, ss FROM __matview_sensor_stats WHERE grp = 'g1'").fetchone()
        r_g2 = conn_w.execute("SELECT n, vp, vs, sp, ss FROM __matview_sensor_stats WHERE grp = 'g2'").fetchone()
        self.assertEqual(r_g1[0], 3)
        self.assertAlmostEqual(r_g1[1], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(r_g1[2], 1.0, places=5)
        self.assertEqual(r_g2[0], 3)
        self.assertAlmostEqual(r_g2[1], 200.0 / 3.0, places=4)
        self.assertAlmostEqual(r_g2[2], 100.0, places=4)
        conn_w.close()

        # --- 3. Watermark-Gated & Snapshot-Isolated Chunked Incremental Backfills ---
        conn_wm = sqlite3.connect(":memory:")
        _register_sqlite_compat_udfs(conn_wm)
        conn_wm.execute(
            "CREATE TABLE ledger (id INTEGER PRIMARY KEY, bucket TEXT NOT NULL, amt INTEGER NOT NULL);"
        )
        conn_wm.executemany(
            "INSERT INTO ledger VALUES (?, ?, ?);",
            [(i, "B1" if i % 2 == 1 else "B2", i * 10) for i in range(1, 11)],
        )
        mgr_wm = MaterializedViewManager(conn_wm)

        def _concurrent_dml_during_backfill(batch_idx: int, rows_done: int) -> None:
            if batch_idx == 1:
                # At batch_idx == 1, watermark cursor is at id = 3, max_pk = 10:
                # (a) Insert above max_pk (id = 11 > 10) -> trigger must apply immediately
                conn_wm.execute("INSERT INTO ledger VALUES (11, 'B1', 110);")
                # (b) Delete below cursor_pk (id = 2 <= 3, already folded) -> trigger must subtract immediately
                conn_wm.execute("DELETE FROM ledger WHERE id = 2;")
                # (c) Delete above cursor_pk (id = 8 in (3, 10], not yet folded) -> trigger must skip; batch 3 will not see it
                conn_wm.execute("DELETE FROM ledger WHERE id = 8;")
                # (d) Update below cursor_pk (id = 1 <= 3, already folded) -> trigger must apply delta immediately
                conn_wm.execute("UPDATE ledger SET amt = 999 WHERE id = 1;")
                # (e) Update above cursor_pk (id = 9 in (3, 10], not yet folded) -> trigger must skip; batch 3 will fold 888
                conn_wm.execute("UPDATE ledger SET amt = 888 WHERE id = 9;")

        plan_wm = mgr_wm.register_view(
            "CREATE VIEW ledger_totals AS SELECT bucket, COUNT(*) AS c, SUM(amt) AS s FROM ledger GROUP BY bucket;",
            incremental_backfill=True,
            batch_size=3,
            on_progress=_concurrent_dml_during_backfill,
        )
        cat_wm = mgr_wm.get_catalog()[plan_wm.view_name]
        self.assertEqual(cat_wm["backfill_active"], 0)
        self.assertEqual(cat_wm["backfill_cursor_pk"], 10)

        mv_wm_rows = conn_wm.execute("SELECT bucket, c, s FROM __matview_ledger_totals ORDER BY bucket;").fetchall()
        raw_wm_rows = conn_wm.execute("SELECT bucket, COUNT(*) AS c, SUM(amt) AS s FROM ledger GROUP BY bucket ORDER BY bucket;").fetchall()
        self.assertEqual(mv_wm_rows, raw_wm_rows)

        # Also verify snapshot_isolation=True on backfill_incrementally across dialects
        conn_wm.execute("DELETE FROM __matview_store_ledger_totals;")
        batches_snap = mgr_wm.backfill_incrementally(plan_wm, batch_size=4, snapshot_isolation=True)
        self.assertGreaterEqual(batches_snap, 2)
        self.assertEqual(
            conn_wm.execute("SELECT bucket, c, s FROM __matview_ledger_totals ORDER BY bucket;").fetchall(),
            raw_wm_rows,
        )
        conn_wm.close()

        # --- 4. Native Composite Keys for Joins, UNION ALL & Secondary Group Indexes ---
        join_schema = """
        CREATE TABLE depts (
            id INTEGER PRIMARY KEY,
            d_code VARCHAR(32) NOT NULL,
            region VARCHAR(32) NOT NULL
        );
        CREATE TABLE emps (
            id INTEGER PRIMARY KEY,
            dept_id INTEGER NOT NULL,
            role VARCHAR(32) NOT NULL,
            salary INTEGER NOT NULL
        );
        CREATE TABLE contractors (
            id INTEGER PRIMARY KEY,
            dept_id INTEGER NOT NULL,
            role VARCHAR(32) NOT NULL,
            fee INTEGER NOT NULL
        );
        """
        # 4a. INNER JOIN with NOT NULL composite group key (d.d_code, e.role) -> native PRIMARY KEY (d_code, role), no __mv_pk
        p_join_native = generate_materialized_view(
            join_schema,
            "CREATE VIEW v_dept_role AS SELECT d.d_code, e.role, COUNT(*) AS hc, SUM(e.salary) AS payroll FROM depts d INNER JOIN emps e ON d.id = e.dept_id GROUP BY d.d_code, e.role;",
            dialect=Dialect.POSTGRES,
        )
        self.assertIn("PRIMARY KEY (d_code, role)", p_join_native.backing_table_ddl)
        self.assertNotIn("__mv_pk", p_join_native.backing_table_ddl)
        # Also creates secondary index on non-leading group column `role`
        self.assertTrue(any("ON __matview_store_v_dept_role (role);" in d for d in p_join_native.index_ddls))

        # 4b. UNION ALL Z-set view -> native composite PRIMARY KEY (__mv_branch_id, __mv_pk_0), no __mv_pk
        p_uall_native = generate_materialized_view(
            join_schema,
            "CREATE VIEW v_all_workers AS SELECT id, dept_id, role, salary AS comp FROM emps UNION ALL SELECT id, dept_id, role, fee AS comp FROM contractors;",
            dialect=Dialect.SQLITE,
        )
        self.assertIn("PRIMARY KEY (__mv_branch_id, __mv_pk_0)", p_uall_native.backing_table_ddl)
        self.assertNotIn("__mv_pk ", p_uall_native.backing_table_ddl)

        # 4c. UNION ALL aggregate view with NOT NULL group key -> native PRIMARY KEY (role), no __mv_pk
        p_uall_agg_native = generate_materialized_view(
            join_schema,
            "CREATE VIEW v_role_comp AS SELECT role, COUNT(*) AS n, SUM(comp) AS total_comp FROM (SELECT role, salary AS comp FROM emps UNION ALL SELECT role, fee AS comp FROM contractors) u GROUP BY role;",
            dialect=Dialect.MYSQL,
            ignore_limitations=True,
        )
        self.assertIn("PRIMARY KEY (role)", p_uall_agg_native.backing_table_ddl)
        self.assertNotIn("__mv_pk", p_uall_agg_native.backing_table_ddl)

        # Execute live DML on Join + UNION ALL native composite PK views across SQLite, PG, MySQL
        for d_mode in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            conn_j: Any = (
                sqlite3.connect(":memory:")
                if d_mode == Dialect.SQLITE
                else TranspiledConnectionAdapter(d_mode)
            )
            if d_mode == Dialect.SQLITE:
                _register_sqlite_compat_udfs(conn_j)
            mgr_j = MaterializedViewManager(conn_j, dialect=d_mode, ignore_limitations=True)
            mgr_j.apply_schema_ddl(join_schema, auto_sync_views=False)
            conn_j.execute("INSERT INTO depts VALUES (1, 'ENG', 'US'), (2, 'OPS', 'EU');")
            conn_j.execute("INSERT INTO emps VALUES (1, 1, 'SWE', 100), (2, 1, 'SWE', 150), (3, 2, 'SRE', 120);")
            conn_j.execute("INSERT INTO contractors VALUES (1, 1, 'SWE', 80), (2, 2, 'QA', 90);")
            mgr_j.register_view(
                """
                CREATE VIEW v_dept_role AS
                SELECT d.d_code, e.role, COUNT(*) AS hc, SUM(e.salary) AS payroll
                FROM depts d INNER JOIN emps e ON d.id = e.dept_id
                GROUP BY d.d_code, e.role;
                CREATE VIEW v_all_workers AS
                SELECT id, dept_id, role, salary AS comp FROM emps
                UNION ALL
                SELECT id, dept_id, role, fee AS comp FROM contractors;
                CREATE VIEW v_role_comp AS
                SELECT role, COUNT(*) AS n, SUM(comp) AS total_comp
                FROM (SELECT role, salary AS comp FROM emps UNION ALL SELECT role, fee AS comp FROM contractors) u
                GROUP BY role;
                """
            )
            conn_j.execute("INSERT INTO emps VALUES (4, 2, 'SRE', 130);")
            conn_j.execute("UPDATE emps SET salary = 200 WHERE id = 1;")
            conn_j.execute("DELETE FROM contractors WHERE id = 1;")
            self.assertEqual(
                conn_j.execute("SELECT * FROM __matview_v_dept_role ORDER BY d_code, role;").fetchall(),
                conn_j.execute("SELECT d.d_code, e.role, COUNT(*) AS hc, SUM(e.salary) AS payroll FROM depts d INNER JOIN emps e ON d.id = e.dept_id GROUP BY d.d_code, e.role ORDER BY d.d_code, e.role;").fetchall(),
            )
            self.assertEqual(
                sorted(conn_j.execute("SELECT * FROM __matview_v_all_workers;").fetchall()),
                sorted(conn_j.execute("SELECT id, dept_id, role, salary AS comp FROM emps UNION ALL SELECT id, dept_id, role, fee AS comp FROM contractors;").fetchall()),
            )
            self.assertEqual(
                conn_j.execute("SELECT * FROM __matview_v_role_comp ORDER BY role;").fetchall(),
                conn_j.execute("SELECT role, COUNT(*) AS n, SUM(comp) AS total_comp FROM (SELECT role, salary AS comp FROM emps UNION ALL SELECT role, fee AS comp FROM contractors) u GROUP BY role ORDER BY role;").fetchall(),
            )

            # --- 5. Session-Scoped & Shadow-Table Swap bulk_changes() Isolation ---
            with mgr_j.bulk_changes(use_shadow_swap=True):
                conn_j.execute("INSERT INTO emps VALUES (5, 1, 'PM', 140), (6, 2, 'SRE', 110);")
                conn_j.execute("INSERT INTO contractors VALUES (3, 1, 'PM', 70);")

            self.assertEqual(
                conn_j.execute("SELECT * FROM __matview_v_dept_role ORDER BY d_code, role;").fetchall(),
                conn_j.execute("SELECT d.d_code, e.role, COUNT(*) AS hc, SUM(e.salary) AS payroll FROM depts d INNER JOIN emps e ON d.id = e.dept_id GROUP BY d.d_code, e.role ORDER BY d.d_code, e.role;").fetchall(),
            )
            self.assertEqual(
                conn_j.execute("SELECT * FROM __matview_v_role_comp ORDER BY role;").fetchall(),
                conn_j.execute("SELECT role, COUNT(*) AS n, SUM(comp) AS total_comp FROM (SELECT role, salary AS comp FROM emps UNION ALL SELECT role, fee AS comp FROM contractors) u GROUP BY role ORDER BY role;").fetchall(),
            )
            if hasattr(conn_j, "close"):
                conn_j.close()


    def test_14_differential_oracle_preserves_exact_numeric_values(self) -> None:
        # Adjacent integers beyond binary64's exact integer range must not collapse.
        self.assertFalse(_rows_equal([(2**53 + 1,)], [(float(2**53 + 1),)]))
        self.assertFalse(_rows_equal([(2**53 + 1,)], [(2**53,)]))
        self.assertTrue(_rows_equal([(2**53,)], [(float(2**53),)]))
        self.assertTrue(_rows_equal([(-(2**63),), (2**64 - 1,)], [(2**64 - 1,), (-(2**63),)]))

        # Decimal-to-Decimal comparisons retain every digit, including trailing precision.
        self.assertFalse(_rows_equal([(Decimal("1.00001"),)], [(Decimal("1.00002"),)]))
        self.assertFalse(
            _rows_equal(
                [(Decimal("12345678901234567890.123456"),)],
                [(Decimal("12345678901234567890.123457"),)],
            )
        )
        self.assertTrue(_rows_equal([(Decimal("1.25"),)], [(Decimal("1.25"),)]))

    def test_15_differential_oracle_handles_float_and_special_values(self) -> None:
        self.assertTrue(_rows_equal([(1.0,)], [(1.001,)]))
        self.assertFalse(_rows_equal([(1,)], [(1.001,)]))
        self.assertTrue(_rows_equal([(Decimal("0.1"),)], [(0.1,)]))
        self.assertFalse(_rows_equal([(float("nan"),)], [(1.0,)]))
        self.assertTrue(_rows_equal([(float("nan"),)], [(float("nan"),)]))
        self.assertTrue(_rows_equal([(float("inf"),)], [(float("inf"),)]))
        self.assertFalse(_rows_equal([(float("inf"),)], [(float("-inf"),)]))
        self.assertTrue(_rows_equal([(-0.0,)], [(0.0,)]))
        self.assertTrue(_rows_equal([(Decimal("NaN"),)], [(Decimal("NaN"),)]))
        self.assertFalse(_rows_equal([(Decimal("1.0"),)], [(Decimal("1.01"),)]))

    def test_16_differential_oracle_preserves_types_and_row_multiplicity(self) -> None:
        self.assertFalse(_rows_equal([("1",)], [(1,)]))
        self.assertFalse(_rows_equal([(True,)], [(False,)]))
        self.assertTrue(_rows_equal([(True,)], [(1,)]))
        self.assertTrue(_rows_equal([(b"payload",)], [(bytearray(b"payload"),)]))
        self.assertTrue(_rows_equal([("a",), ("b",), ("a",)], [("a",), ("a",), ("b",)]))
        self.assertFalse(_rows_equal([("a",), ("a",)], [("a",), ("b",)]))
        self.assertFalse(_rows_equal([("a",), ("a",)], [("a",)]))
        with self.assertRaises(ValueError):
            _rows_equal([(1.0,)], [(1.0,)], tol=-1.0)


    def test_variable_schema_fuzzer_produces_diverse_reproducible_structures(self) -> None:
        self.assertEqual(
            VariableSchemaDifferentialFuzzer(123).generate_scenario(4),
            VariableSchemaDifferentialFuzzer(123).generate_scenario(4),
        )
        shapes = []
        topologies, table_counts = set(), set()
        for seed in range(64):
            schema, views, seeds, mutations, shape = VariableSchemaDifferentialFuzzer(seed).generate_scenario(1)
            shapes.append((
                shape["topology"], shape["table_count"], shape["column_counts"],
                shape["amount_types"], shape["foreign_key_count"],
                shape["composite_primary_key_tables"], shape["nullable_columns"],
            ))
            topologies.add(shape["topology"])
            table_counts.add(shape["table_count"])
            self.assertGreaterEqual(len(views), shape["table_count"])
            self.assertGreater(len(seeds), shape["table_count"])
            self.assertGreaterEqual(len(mutations), 3)

            # Validate generated DDL and seed rows with FK enforcement enabled.
            conn = sqlite3.connect(":memory:")
            try:
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.executescript(schema)
                for seed_sql in seeds:
                    conn.execute(seed_sql)
                conn.commit()
            finally:
                conn.close()
        self.assertGreaterEqual(len(topologies), 4)
        self.assertGreaterEqual(len(table_counts), 4)
        self.assertGreaterEqual(len(set(shapes)), 48)

    def test_variable_schema_fuzzer_executes_across_dialects(self) -> None:
        schema, views, seeds, mutations, shape = VariableSchemaDifferentialFuzzer(314159).generate_scenario(2)
        runner = MVTestRunner()
        for dialect in (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL):
            runner.run_batch(
                f"VariableSchema(seed=314159,{dialect.value},{shape['topology']})",
                schema, views[:8], seeds, mutations[:9], dialect=dialect,
            )
        view_count = min(8, len(views))
        mutation_count = min(9, len(mutations))
        self.assertEqual(runner.total_views, view_count * 3)
        self.assertEqual(runner.total_assertions, view_count * (mutation_count + 1) * 3)


import random

import secrets

class DifferentialSQLFuzzer:
    def __init__(self, seed: int = 20261106) -> None:
        self.seed = seed

    def generate_scenario(
        self,
        scenario_idx: int,
        dialect: Dialect = Dialect.SQLITE,
        views_count: int = 80,
        mutations_count: int = 60,
    ) -> Tuple[str, List[Tuple[str, str]], List[str], List[Tuple[str, str]]]:
        rng = random.Random((self.seed * 1000003) ^ (scenario_idx * 104729))
        p = f"fz{scenario_idx}"

        root_names = ["sectors", "regions", "hubs", "zones", "divisions"]
        dim_names = ["orgs", "customers", "accounts", "merchants", "tenants"]
        fact_names = ["txns", "orders", "events", "records", "entries"]
        c1_names = ["tags", "badges", "labels", "flags_map"]
        c2_names = ["allocs", "splits", "tranches", "slices"]

        t_root = f"{p}_{rng.choice(root_names)}"
        t_dim = f"{p}_{rng.choice(dim_names)}"
        t_fact = f"{p}_{rng.choice(fact_names)}"
        t_comp1 = f"{p}_{rng.choice(c1_names)}"
        t_comp2 = f"{p}_{rng.choice(c2_names)}"

        cats_a = rng.sample(["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "zeta"], 6)
        cats_b = rng.sample(["hw", "sw", "cloud", "ai", "sec", "infra", "mobile", "data"], 6)
        tiers = rng.sample(["gold", "silver", "vip", "std", "plat", "core"], 5)
        codes = rng.sample(["NA", "EU", "APAC", "LATAM", "EMEA", "ANZ"], 5)
        tag_keys = rng.sample(["urgent", "bulk", "promo", "audit", "p0", "review"], 5)
        alloc_keys = rng.sample(["A100", "B200", "C300", "D400", "E500", "F600"], 5)

        schema_sql = f"""
        CREATE TABLE {t_root} (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            country TEXT,
            active INTEGER NOT NULL
        );
        CREATE TABLE {t_dim} (
            id INTEGER PRIMARY KEY,
            root_id INTEGER,
            parent_id INTEGER,
            pname TEXT NOT NULL,
            tier TEXT,
            quota INTEGER NOT NULL,
            score REAL
        );
        CREATE TABLE {t_fact} (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER,
            grp_str TEXT NOT NULL,
            sub_cat TEXT,
            val_int INTEGER,
            val_flt REAL,
            weight_cnt INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            flag INTEGER NOT NULL
        );
        CREATE TABLE {t_comp1} (
            ref_id INTEGER NOT NULL,
            tag_key TEXT NOT NULL,
            pts INTEGER NOT NULL,
            bonus INTEGER,
            PRIMARY KEY (ref_id, tag_key)
        );
        CREATE TABLE {t_comp2} (
            fact_id INTEGER NOT NULL,
            alloc_code TEXT NOT NULL,
            slot_no INTEGER NOT NULL,
            amount REAL NOT NULL,
            active_flag INTEGER NOT NULL,
            PRIMARY KEY (fact_id, alloc_code, slot_no)
        );
        """

        seed_sqls: List[str] = []
        num_roots = rng.randint(4, 6)
        for rid in range(1, num_roots + 1):
            c_code = codes[(rid - 1) % len(codes)]
            c_country = "NULL" if rid == num_roots else f"'{c_code}_HQ'"
            act = 1 if rid % 2 == 1 else 0
            seed_sqls.append(f"INSERT INTO {t_root} VALUES ({rid}, '{c_code}', {c_country}, {act});")

        num_dims = rng.randint(6, 9)
        for pid in range(1, num_dims + 1):
            r_ref = "NULL" if pid == num_dims else str(((pid - 1) % num_roots) + 1)
            p_parent = "NULL" if pid <= 2 else str(((pid - 2) % 2) + 1)
            t_val = "NULL" if pid % 4 == 0 else f"'{tiers[(pid - 1) % len(tiers)]}'"
            quota = rng.randint(10, 120)
            score = "NULL" if pid % 5 == 0 else f"{rng.randint(5, 95) * 1.5:.1f}"
            seed_sqls.append(
                f"INSERT INTO {t_dim} VALUES ({pid}, {r_ref}, {p_parent}, 'ORG_{pid}', {t_val}, {quota}, {score});"
            )

        next_fact_id = 1
        num_initial_facts = rng.randint(14, 20)
        for _ in range(num_initial_facts):
            fid = next_fact_id
            next_fact_id += 1
            p_ref = "NULL" if rng.random() < 0.2 else str(rng.randint(1, num_dims))
            g_str = rng.choice(cats_a)
            s_cat = "NULL" if rng.random() < 0.25 else f"'{rng.choice(cats_b)}'"
            v_int = "NULL" if rng.random() < 0.25 else str(rng.randint(1, 60))
            v_flt = "NULL" if rng.random() < 0.2 else f"{rng.randint(1, 40) * 2.5:.1f}"
            w_cnt = rng.randint(1, 20)
            prio = rng.randint(1, 5)
            flg = rng.choice([0, 1])
            seed_sqls.append(
                f"INSERT INTO {t_fact} VALUES ({fid}, {p_ref}, '{g_str}', {s_cat}, {v_int}, {v_flt}, {w_cnt}, {prio}, {flg});"
            )
            if fid in (1, 3, 6) and next_fact_id <= num_initial_facts + 5:
                dup_id = next_fact_id
                next_fact_id += 1
                seed_sqls.append(
                    f"INSERT INTO {t_fact} VALUES ({dup_id}, {p_ref}, '{g_str}', {s_cat}, {v_int}, {v_flt}, {w_cnt}, {prio}, {flg});"
                )

        used_c1 = set()
        for _ in range(12):
            rf = rng.randint(1, 10)
            tk = rng.choice(tag_keys)
            if (rf, tk) not in used_c1:
                used_c1.add((rf, tk))
                pts = rng.randint(1, 30)
                bon = "NULL" if rng.random() < 0.3 else str(rng.randint(1, 15))
                seed_sqls.append(f"INSERT INTO {t_comp1} VALUES ({rf}, '{tk}', {pts}, {bon});")

        used_c2 = set()
        for _ in range(12):
            fid = rng.randint(1, 12)
            ac = rng.choice(alloc_keys)
            sl = rng.randint(1, 3)
            if (fid, ac, sl) not in used_c2:
                used_c2.add((fid, ac, sl))
                amt = f"{rng.randint(5, 60) * 1.5:.1f}"
                af = rng.choice([0, 1])
                seed_sqls.append(f"INSERT INTO {t_comp2} VALUES ({fid}, '{ac}', {sl}, {amt}, {af});")

        views: List[Tuple[str, str]] = []
        seen_sqls = set()
        v_idx_holder = [0]

        def add_v(tag: str, select_sql: str) -> None:
            if len(views) >= views_count:
                return
            norm_s = select_sql.strip().lower()
            if norm_s in seen_sqls:
                return
            seen_sqls.add(norm_s)
            v_idx_holder[0] += 1
            vn = f"{p}_v{v_idx_holder[0]:03d}_{tag}"
            views.append((vn, f"CREATE VIEW {vn} AS {select_sql};"))

        w_filters = [
            "",
            "WHERE flag = 1",
            "WHERE flag = 0",
            f"WHERE weight_cnt >= {rng.randint(3, 8)}",
            f"WHERE priority >= {rng.randint(2, 4)}",
            f"WHERE grp_str IN ('{cats_a[0]}', '{cats_a[1]}', '{cats_a[2]}')",
            "WHERE sub_cat IS NOT NULL AND val_int IS NOT NULL",
            "WHERE sub_cat IS NULL OR flag = 1",
            f"WHERE (flag = 1 OR weight_cnt >= {rng.randint(4, 10)}) AND weight_cnt <= 22",
            f"WHERE priority BETWEEN 1 AND {rng.randint(3, 5)} AND weight_cnt >= 2",
        ]

        grp_candidates = [
            ("grp_str", "grp_str"),
            ("sub_cat", "sub_cat"),
            ("flag", "flag"),
            ("priority", "priority"),
            ("grp_str, sub_cat", "grp_str, sub_cat"),
            ("grp_str, flag", "grp_str, flag"),
            ("sub_cat, flag", "sub_cat, flag"),
            ("grp_str, priority", "grp_str, priority"),
            ("grp_str, sub_cat, flag", "grp_str, sub_cat, flag"),
            ("COALESCE(sub_cat, 'UNK') AS sc_norm", "COALESCE(sub_cat, 'UNK')"),
            ("LOWER(grp_str) AS l_grp", "LOWER(grp_str)"),
            ("UPPER(COALESCE(sub_cat, grp_str)) AS u_cat", "UPPER(COALESCE(sub_cat, grp_str))"),
            (f"CASE WHEN weight_cnt >= 8 THEN 'heavy' ELSE 'light' END AS w_bucket", f"CASE WHEN weight_cnt >= 8 THEN 'heavy' ELSE 'light' END"),
            (f"CASE WHEN priority >= 3 THEN 'high' WHEN priority = 2 THEN 'med' ELSE 'low' END AS p_tier", f"CASE WHEN priority >= 3 THEN 'high' WHEN priority = 2 THEN 'med' ELSE 'low' END"),
        ]

        # Family 1: Projections & SELECT DISTINCT (up to 10 views)
        for p_i in range(5):
            wf = rng.choice(w_filters[1:])
            c_mult = rng.randint(2, 5)
            add_v(
                f"proj_{p_i}",
                f"SELECT id, grp_str, COALESCE(sub_cat, 'none') AS sc, weight_cnt * {c_mult} + priority AS metric_w, COALESCE(val_int, 0) AS vi, flag FROM {t_fact} {wf}",
            )
        dist_Patterns = [
            "grp_str, sub_cat, flag",
            "COALESCE(sub_cat, grp_str) AS merged_cat, flag",
            "grp_str, priority",
            "sub_cat, priority, flag",
            "parent_id, grp_str",
        ]
        for d_i, dp in enumerate(dist_Patterns):
            wf = rng.choice(w_filters)
            w_sp = f" {wf}" if wf else ""
            add_v(f"distinct_{d_i}", f"SELECT DISTINCT {dp} FROM {t_fact}{w_sp}")

        # Family 2: Single-table Grouped Algebraic, Statistical, Bitwise, Boolean & Compound Aggregates (up to 25 views)
        base_aggs = [
            "COUNT(*) AS c_all",
            "COUNT(val_int) AS c_int",
            "COUNT(sub_cat) AS c_sc",
            "SUM(val_int) AS s_int",
            "SUM(val_flt) AS s_flt",
            "SUM(weight_cnt) AS s_w",
            "AVG(val_flt) AS a_flt",
            "AVG(val_int) AS a_int",
            "MIN(weight_cnt) AS mn_w",
            "MAX(weight_cnt) AS mx_w",
            "MIN(val_int) AS mn_vi",
            "MAX(val_int) AS mx_vi",
            "SUM(val_int * weight_cnt) AS w_sum",
            "SUM(weight_cnt * priority) AS wp_sum",
            "VAR_POP(val_flt) AS vp_flt",
            "VAR_POP(weight_cnt) AS vp_w",
            "BIT_AND(weight_cnt) AS ba_w",
            "BIT_OR(weight_cnt) AS bo_w",
            "BIT_XOR(weight_cnt) AS bx_w",
            "BIT_XOR(priority) AS bx_p",
        ]
        if dialect == Dialect.SQLITE:
            base_aggs.extend([
                "TOTAL(val_flt) AS tot_flt",
                "SUM(weight_cnt) FILTER (WHERE flag = 1) AS filt_w",
                "COUNT(*) FILTER (WHERE priority >= 3) AS filt_p",
            ])
        elif dialect == Dialect.POSTGRES:
            base_aggs.extend([
                "STDDEV_POP(val_flt) AS sdp_flt",
                "SUM(weight_cnt) FILTER (WHERE flag = 1) AS filt_w",
                "COUNT(*) FILTER (WHERE priority >= 3) AS filt_p",
            ])
        else:
            base_aggs.append("STDDEV_POP(val_flt) AS sdp_flt")

        for g_i in range(18):
            g_sel, g_by = grp_candidates[g_i % len(grp_candidates)]
            w_cl = rng.choice(w_filters)
            w_sp = f" {w_cl}" if w_cl else ""
            c_aggs = rng.sample(base_aggs, rng.randint(3, min(7, len(base_aggs))))
            hav = ""
            if g_i % 4 == 0:
                hav = f" HAVING COUNT(*) >= {rng.randint(1, 3)}"
            elif g_i % 4 == 1:
                hav = f" HAVING COALESCE(SUM(weight_cnt), 0) >= {rng.randint(5, 18)}"
            elif g_i % 4 == 2:
                hav = f" HAVING COUNT(*) >= 1 AND COALESCE(MAX(weight_cnt), 0) >= {rng.randint(4, 10)}"
            add_v(f"alg_grp_{g_i}", f"SELECT {g_sel}, {', '.join(c_aggs)} FROM {t_fact}{w_sp} GROUP BY {g_by}{hav}")

        # Hidden GROUP BY, No COUNT(*), Compound Arithmetic, Dimension & Composite PK table aggs
        add_v("alg_hidden_1", f"SELECT COUNT(*) AS cnt, SUM(val_int) AS si, MIN(weight_cnt) AS mnw, MAX(weight_cnt) AS mxw FROM {t_fact} WHERE flag = 1 GROUP BY grp_str")
        add_v("alg_hidden_2", f"SELECT SUM(weight_cnt) AS sw, AVG(val_flt) AS af FROM {t_fact} GROUP BY grp_str, sub_cat")
        add_v("alg_no_cnt_1", f"SELECT grp_str, SUM(val_flt) AS sf, AVG(val_int) AS ai, MIN(weight_cnt) AS mnw, MAX(weight_cnt) AS mxw, BIT_XOR(weight_cnt) AS bxw FROM {t_fact} GROUP BY grp_str")
        add_v("alg_no_cnt_2", f"SELECT sub_cat, SUM(val_int) AS si, VAR_POP(val_flt) AS vp, BIT_OR(weight_cnt) AS bow FROM {t_fact} GROUP BY sub_cat")
        add_v("alg_compound_1", f"SELECT grp_str, COUNT(*) AS cnt, COALESCE(SUM(val_int * weight_cnt), 0) - COALESCE(SUM(val_int), 0) AS net_v, COALESCE(MAX(weight_cnt) - MIN(weight_cnt), 0) AS span_w FROM {t_fact} GROUP BY grp_str")
        add_v("alg_compound_2", f"SELECT grp_str, flag, COUNT(*) - COUNT(val_int) AS null_vi_cnt, COALESCE(SUM(weight_cnt * priority), 0) + COALESCE(SUM(weight_cnt), 0) AS comb_w FROM {t_fact} GROUP BY grp_str, flag")
        add_v("alg_dim_tier", f"SELECT tier, COUNT(*) AS cnt, SUM(quota) AS sq, AVG(score) AS a_sc, MIN(quota) AS mnq, MAX(quota) AS mxq FROM {t_dim} GROUP BY tier")
        add_v("alg_comp_pk1", f"SELECT tag_key, COUNT(*) AS cnt, SUM(pts) AS s_pts, SUM(bonus) AS s_bon, AVG(pts) AS a_pts, MIN(pts) AS mn_p, MAX(pts) AS mx_p FROM {t_comp1} GROUP BY tag_key")
        add_v("alg_comp_pk2", f"SELECT alloc_code, active_flag, COUNT(*) AS cnt, SUM(amount) AS tot_amt, AVG(amount) AS avg_amt, MIN(amount) AS mn_a, MAX(amount) AS mx_a FROM {t_comp2} GROUP BY alloc_code, active_flag")

        # Family 3: Reference-Counted DISTINCT Aggregates (8 views)
        dist_agg_pool = [
            "COUNT(DISTINCT sub_cat) AS cd_sc",
            "COUNT(DISTINCT val_int) AS cd_vi",
            "COUNT(DISTINCT priority) AS cd_pr",
            "SUM(DISTINCT weight_cnt) AS sd_w",
            "SUM(DISTINCT val_int) AS sd_vi",
            "AVG(DISTINCT val_flt) AS ad_vf",
            "SUM(weight_cnt) AS s_w",
            "BIT_XOR(weight_cnt) AS bx_w",
            "VAR_POP(val_flt) AS vp_vf",
            "COUNT(*) AS cnt",
        ]
        for d_i in range(8):
            g_sel, g_by = grp_candidates[d_i % 8]
            w_cl = rng.choice(w_filters[:6])
            w_sp = f" {w_cl}" if w_cl else ""
            d_aggs = rng.sample(dist_agg_pool, 4)
            if not any("DISTINCT" in a for a in d_aggs):
                d_aggs[0] = "COUNT(DISTINCT sub_cat) AS cd_sc"
            add_v(f"dist_agg_{d_i}", f"SELECT {g_sel}, {', '.join(d_aggs)} FROM {t_fact}{w_sp} GROUP BY {g_by}")

        # Family 4: Ungrouped Global Aggregates (6 views)
        add_v("global_full", f"SELECT COUNT(*) AS cnt, COUNT(val_int) AS ci, SUM(val_int) AS si, AVG(val_flt) AS af, MIN(weight_cnt) AS mnw, MAX(weight_cnt) AS mxw, VAR_POP(val_flt) AS vp, BIT_XOR(weight_cnt) AS bx FROM {t_fact}")
        add_v("global_filtered_nocnt", f"SELECT SUM(val_int) AS si, AVG(val_flt) AS af, MIN(weight_cnt) AS mnw, MAX(weight_cnt) AS mxw, COUNT(DISTINCT grp_str) AS cd_g FROM {t_fact} WHERE flag = 1 AND weight_cnt >= 10")
        add_v("global_dim", f"SELECT COUNT(*) AS cnt, SUM(quota) AS sq, AVG(score) AS a_sc, MIN(quota) AS mnq, MAX(quota) AS mxq FROM {t_dim} WHERE quota >= 20")
        add_v("global_comp1", f"SELECT COUNT(*) AS cnt, SUM(pts) AS sp, COUNT(DISTINCT tag_key) AS cd_tk, MIN(pts) AS mnp, MAX(pts) AS mxp FROM {t_comp1}")
        add_v("global_comp2", f"SELECT COUNT(*) AS cnt, SUM(amount) AS sa, AVG(amount) AS aa, MIN(amount) AS mna, MAX(amount) AS mxa FROM {t_comp2} WHERE active_flag = 1")
        add_v("global_bit_only", f"SELECT BIT_OR(weight_cnt) AS bo, BIT_XOR(weight_cnt) AS bx, BIT_AND(weight_cnt) AS ba, SUM(weight_cnt) AS sw FROM {t_fact} WHERE priority >= 4")

        # Family 5: Multi-Table Joins (14 views)
        add_v("join_2t_agg_1", f"SELECT p.tier, m.grp_str, COUNT(*) AS cnt, SUM(m.weight_cnt * p.quota) AS w_quota, AVG(m.val_flt) AS af, MIN(m.weight_cnt) AS mnw, MAX(m.weight_cnt) AS mxw, BIT_XOR(m.weight_cnt) AS bx FROM {t_fact} m INNER JOIN {t_dim} p ON m.parent_id = p.id WHERE m.flag = 1 GROUP BY p.tier, m.grp_str")
        add_v("join_2t_agg_2", f"SELECT COALESCE(p.tier, 'NONE') AS t_norm, LOWER(m.grp_str) AS l_grp, COUNT(*) AS cnt, SUM(m.val_int) AS si, VAR_POP(m.val_flt) AS vp FROM {t_fact} m JOIN {t_dim} p ON m.parent_id = p.id GROUP BY COALESCE(p.tier, 'NONE'), LOWER(m.grp_str) HAVING COUNT(*) >= 1")
        add_v("join_2t_global", f"SELECT COUNT(*) AS cnt, SUM(m.weight_cnt) AS sw, AVG(p.quota) AS aq, MIN(m.weight_cnt) AS mnw, MAX(m.weight_cnt) AS mxw FROM {t_fact} m JOIN {t_dim} p ON m.parent_id = p.id WHERE m.flag = 1")
        add_v("join_3t_chain_1", f"SELECT r.code, p.tier, COUNT(*) AS cnt, SUM(m.weight_cnt) AS sw, SUM(m.val_int) AS si, MIN(m.weight_cnt) AS mnw, MAX(m.weight_cnt) AS mxw FROM {t_fact} m JOIN {t_dim} p ON m.parent_id = p.id JOIN {t_root} r ON p.root_id = r.id GROUP BY r.code, p.tier")
        add_v("join_3t_chain_2", f"SELECT r.code, m.grp_str, COUNT(*) AS cnt, SUM(c.pts * m.weight_cnt) AS w_pts FROM {t_comp1} c JOIN {t_fact} m ON c.ref_id = m.id JOIN {t_dim} p ON m.parent_id = p.id JOIN {t_root} r ON p.root_id = r.id GROUP BY r.code, m.grp_str")
        add_v("join_cross_1", f"SELECT r.code, p.tier, COUNT(*) AS cnt, SUM(p.quota) AS sq FROM {t_root} r CROSS JOIN {t_dim} p WHERE r.active = 1 GROUP BY r.code, p.tier")
        add_v("join_comp_pk1", f"SELECT c.tag_key, m.grp_str, COUNT(*) AS cnt, SUM(c.pts * m.weight_cnt) AS total_pts, MIN(c.pts) AS mnp, MAX(c.pts) AS mxp FROM {t_comp1} c JOIN {t_fact} m ON c.ref_id = m.id GROUP BY c.tag_key, m.grp_str")
        add_v("join_comp_pk2", f"SELECT a.alloc_code, m.grp_str, COUNT(*) AS cnt, SUM(a.amount) AS tot_alloc, AVG(a.amount) AS avg_alloc FROM {t_comp2} a JOIN {t_fact} m ON a.fact_id = m.id WHERE a.active_flag = 1 GROUP BY a.alloc_code, m.grp_str")
        add_v("left_join_proj", f"SELECT p.id AS pid, p.pname, m.id AS mid, m.grp_str, m.weight_cnt FROM {t_dim} p LEFT JOIN {t_fact} m ON p.id = m.parent_id")
        add_v("left_join_agg_1", f"SELECT p.pname, p.tier, COUNT(*) AS row_cnt, COUNT(m.id) AS match_cnt, SUM(m.weight_cnt) AS sum_w, MIN(m.weight_cnt) AS min_w, MAX(m.weight_cnt) AS max_w FROM {t_dim} p LEFT JOIN {t_fact} m ON p.id = m.parent_id GROUP BY p.pname, p.tier")
        add_v("left_join_agg_2", f"SELECT r.code, COUNT(*) AS row_cnt, COUNT(p.id) AS org_cnt, SUM(p.quota) AS sum_q FROM {t_root} r LEFT JOIN {t_dim} p ON r.id = p.root_id GROUP BY r.code")
        add_v("right_join_proj", f"SELECT m.id AS mid, m.grp_str, p.id AS pid, p.pname, p.tier FROM {t_fact} m RIGHT JOIN {t_dim} p ON m.parent_id = p.id")
        add_v("full_outer_join_proj", f"SELECT p.id AS pid, p.pname, r.id AS rid, r.code FROM {t_dim} p FULL OUTER JOIN {t_root} r ON p.root_id = r.id")
        add_v("self_join_tree", f"SELECT c.id AS cid, c.pname AS child_name, p.pname AS parent_name, c.quota AS child_quota, p.quota AS parent_quota FROM {t_dim} c LEFT JOIN {t_dim} p ON c.parent_id = p.id")

        # Family 6: Subqueries (EXISTS, NOT EXISTS, IN, NOT IN — 6 views)
        add_v("subq_exists_1", f"SELECT id, grp_str, weight_cnt, priority FROM {t_fact} WHERE EXISTS (SELECT 1 FROM {t_dim} p WHERE p.id = {t_fact}.parent_id AND p.quota >= 30)")
        add_v("subq_exists_2", f"SELECT id, grp_str, weight_cnt FROM {t_fact} WHERE EXISTS (SELECT 1 FROM {t_comp1} c WHERE c.ref_id = {t_fact}.id AND c.pts >= 10)")
        add_v("subq_not_exists_1", f"SELECT id, pname, tier, quota FROM {t_dim} WHERE NOT EXISTS (SELECT 1 FROM {t_fact} m WHERE m.parent_id = {t_dim}.id AND m.flag = 1)")
        add_v("subq_not_exists_2", f"SELECT id, code, country FROM {t_root} WHERE NOT EXISTS (SELECT 1 FROM {t_dim} p WHERE p.root_id = {t_root}.id AND p.quota >= 50)")
        add_v("subq_in", f"SELECT id, grp_str, weight_cnt FROM {t_fact} WHERE parent_id IN (SELECT id FROM {t_dim} WHERE quota >= 40)")
        add_v("subq_not_in", f"SELECT id, grp_str, weight_cnt FROM {t_fact} WHERE parent_id NOT IN (SELECT id FROM {t_dim} WHERE quota < 25 AND id IS NOT NULL)")

        # Family 7: Set Operations, CTEs, Window Functions & Top-N (11 views)
        add_v("set_union_all_2b", f"SELECT grp_str AS label, weight_cnt AS metric FROM {t_fact} WHERE flag = 1 UNION ALL SELECT pname AS label, quota AS metric FROM {t_dim}")
        add_v("set_union_all_3b", f"SELECT grp_str AS label, weight_cnt AS metric FROM {t_fact} WHERE flag = 1 UNION ALL SELECT pname AS label, quota AS metric FROM {t_dim} UNION ALL SELECT tag_key AS label, pts AS metric FROM {t_comp1}")
        add_v("set_union_dedup", f"SELECT grp_str AS code_lbl FROM {t_fact} UNION SELECT tier AS code_lbl FROM {t_dim} WHERE tier IS NOT NULL")
        add_v("set_intersect", f"SELECT parent_id AS ref_k FROM {t_fact} WHERE parent_id IS NOT NULL INTERSECT SELECT id AS ref_k FROM {t_dim} WHERE quota >= 20")
        add_v("set_except", f"SELECT id AS ref_key FROM {t_dim} EXCEPT SELECT parent_id AS ref_key FROM {t_fact} WHERE parent_id IS NOT NULL")
        add_v("cte_union_agg_1", f"WITH u AS (SELECT grp_str AS k, weight_cnt AS v FROM {t_fact} WHERE flag = 1 UNION ALL SELECT COALESCE(tier, 'NONE') AS k, quota AS v FROM {t_dim}) SELECT k, COUNT(*) AS cnt, SUM(v) AS sv, AVG(v) AS av, MIN(v) AS mnv, MAX(v) AS mxv FROM u GROUP BY k")
        add_v("cte_union_agg_2", f"WITH u AS (SELECT tag_key AS k, pts AS v FROM {t_comp1} UNION ALL SELECT alloc_code AS k, slot_no * 5 AS v FROM {t_comp2}) SELECT k, COUNT(*) AS cnt, SUM(v) AS sv, MIN(v) AS mnv, MAX(v) AS mxv FROM u GROUP BY k")
        add_v("window_rank_1", f"SELECT id, grp_str, weight_cnt, ROW_NUMBER() OVER (PARTITION BY grp_str ORDER BY weight_cnt DESC, id ASC) AS rn, SUM(weight_cnt) OVER (PARTITION BY grp_str ORDER BY id ASC) AS run_w, LAG(weight_cnt, 1, 0) OVER (PARTITION BY grp_str ORDER BY id ASC) AS prev_w FROM {t_fact}")
        add_v("window_rank_2", f"SELECT id, pname, tier, quota, DENSE_RANK() OVER (PARTITION BY tier ORDER BY quota DESC, id ASC) AS drk, LEAD(quota, 1, -1) OVER (PARTITION BY tier ORDER BY id ASC) AS next_q FROM {t_dim}")
        add_v("topn_limit_1", f"SELECT id, grp_str, weight_cnt, val_int FROM {t_fact} WHERE flag = 1 ORDER BY weight_cnt DESC, id ASC LIMIT 5 OFFSET 1")
        add_v("topn_limit_2", f"SELECT id, pname, quota, score FROM {t_dim} ORDER BY quota DESC, id ASC LIMIT 4 OFFSET 0")

        views = views[:views_count]

        # Extensive Random DML + Transaction Sequence
        mutations: List[Tuple[str, str]] = []
        op_types = [
            "ins_fact_single",
            "ins_fact_multi",
            "upd_fact_val",
            "upd_fact_grp",
            "upd_fact_parent",
            "upd_fact_case_swap",
            "del_fact_row",
            "upsert_fact",
            "upsert_c1",
            "upsert_c2",
            "upd_dim_quota",
            "upd_root_active",
            "ins_select_fact",
            "del_comp_row",
        ]

        pre_tx_steps = max(10, mutations_count - 20)
        for s_idx in range(1, pre_tx_steps + 1):
            op = rng.choice(op_types)
            if op == "ins_fact_single":
                fid = next_fact_id
                next_fact_id += 1
                p_ref = "NULL" if rng.random() < 0.25 else str(rng.randint(1, num_dims))
                g_str = rng.choice(cats_a)
                s_cat = "NULL" if rng.random() < 0.3 else f"'{rng.choice(cats_b)}'"
                v_int = "NULL" if rng.random() < 0.25 else str(rng.randint(1, 70))
                v_flt = "NULL" if rng.random() < 0.2 else f"{rng.randint(1, 50) * 1.5:.1f}"
                w_cnt = rng.randint(1, 25)
                prio = rng.randint(1, 5)
                flg = rng.choice([0, 1])
                mutations.append(
                    (f"m{s_idx}_ins1", f"INSERT INTO {t_fact} VALUES ({fid}, {p_ref}, '{g_str}', {s_cat}, {v_int}, {v_flt}, {w_cnt}, {prio}, {flg});")
                )
            elif op == "ins_fact_multi":
                f1, f2 = next_fact_id, next_fact_id + 1
                next_fact_id += 2
                g1, g2 = rng.choice(cats_a), rng.choice(cats_a)
                sc1 = "NULL" if rng.random() < 0.3 else f"'{rng.choice(cats_b)}'"
                sc2 = "NULL" if rng.random() < 0.3 else f"'{rng.choice(cats_b)}'"
                w1, w2 = rng.randint(1, 20), rng.randint(1, 20)
                p1, p2 = rng.randint(1, 5), rng.randint(1, 5)
                mutations.append(
                    (
                        f"m{s_idx}_ins2",
                        f"INSERT INTO {t_fact} VALUES "
                        f"({f1}, {rng.randint(1, num_dims)}, '{g1}', {sc1}, {rng.randint(1, 40)}, 12.5, {w1}, {p1}, 1), "
                        f"({f2}, NULL, '{g2}', {sc2}, NULL, 27.5, {w2}, {p2}, 0);",
                    )
                )
            elif op == "upd_fact_val":
                lo_id = rng.randint(1, max(1, next_fact_id - 5))
                hi_id = lo_id + rng.randint(1, 4)
                delta_w = rng.randint(1, 6)
                mutations.append(
                    (
                        f"m{s_idx}_upd_val",
                        f"UPDATE {t_fact} SET val_int = CASE WHEN val_int IS NULL THEN {rng.randint(5, 35)} ELSE NULL END, "
                        f"val_flt = COALESCE(val_flt, 10.0) + 2.5, weight_cnt = weight_cnt + {delta_w}, flag = 1 - flag "
                        f"WHERE id BETWEEN {lo_id} AND {hi_id};",
                    )
                )
            elif op == "upd_fact_grp":
                target_id = rng.randint(1, next_fact_id - 1)
                new_g = rng.choice(cats_a)
                new_sc = "NULL" if rng.random() < 0.35 else f"'{rng.choice(cats_b)}'"
                new_p = "NULL" if rng.random() < 0.25 else str(rng.randint(1, num_dims))
                new_prio = rng.randint(1, 5)
                mutations.append(
                    (
                        f"m{s_idx}_upd_grp",
                        f"UPDATE {t_fact} SET grp_str = '{new_g}', sub_cat = {new_sc}, parent_id = {new_p}, priority = {new_prio} WHERE id = {target_id};",
                    )
                )
            elif op == "upd_fact_parent":
                target_id = rng.randint(1, next_fact_id - 1)
                new_p = "NULL" if rng.random() < 0.25 else str(rng.randint(1, num_dims))
                mutations.append(
                    (f"m{s_idx}_upd_par", f"UPDATE {t_fact} SET parent_id = {new_p} WHERE id = {target_id};")
                )
            elif op == "upd_fact_case_swap":
                c0, c1 = cats_a[0], cats_a[1]
                mutations.append(
                    (
                        f"m{s_idx}_swap",
                        f"UPDATE {t_fact} SET grp_str = CASE WHEN grp_str = '{c0}' THEN '{c1}' WHEN grp_str = '{c1}' THEN '{c0}' ELSE grp_str END, "
                        f"priority = CASE WHEN priority >= 5 THEN 1 ELSE priority + 1 END WHERE flag = 1;",
                    )
                )
            elif op == "del_fact_row":
                del_id = rng.randint(1, next_fact_id - 1)
                mutations.append((f"m{s_idx}_del_f", f"DELETE FROM {t_fact} WHERE id = {del_id};"))
            elif op == "upsert_fact":
                uf_id = rng.randint(1, next_fact_id)
                if uf_id == next_fact_id:
                    next_fact_id += 1
                g_s = rng.choice(cats_a)
                w_c = rng.randint(2, 18)
                pr = rng.randint(1, 5)
                if dialect == Dialect.MYSQL:
                    mutations.append(
                        (
                            f"m{s_idx}_ups_f",
                            f"INSERT INTO {t_fact} VALUES ({uf_id}, 1, '{g_s}', '{cats_b[0]}', 15, 22.0, {w_c}, {pr}, 1) "
                            f"ON DUPLICATE KEY UPDATE grp_str = VALUES(grp_str), weight_cnt = VALUES(weight_cnt), priority = VALUES(priority), flag = VALUES(flag);",
                        )
                    )
                else:
                    mutations.append(
                        (
                            f"m{s_idx}_ups_f",
                            f"INSERT INTO {t_fact} VALUES ({uf_id}, 1, '{g_s}', '{cats_b[0]}', 15, 22.0, {w_c}, {pr}, 1) "
                            f"ON CONFLICT (id) DO UPDATE SET grp_str = excluded.grp_str, weight_cnt = excluded.weight_cnt, priority = excluded.priority, flag = excluded.flag;",
                        )
                    )
            elif op == "upsert_c1":
                rf = rng.randint(1, 10)
                tk = rng.choice(tag_keys)
                pts = rng.randint(5, 35)
                bon = "NULL" if rng.random() < 0.3 else str(rng.randint(2, 12))
                if dialect == Dialect.MYSQL:
                    mutations.append(
                        (f"m{s_idx}_ups_c1", f"INSERT INTO {t_comp1} VALUES ({rf}, '{tk}', {pts}, {bon}) ON DUPLICATE KEY UPDATE pts = VALUES(pts), bonus = VALUES(bonus);")
                    )
                else:
                    mutations.append(
                        (f"m{s_idx}_ups_c1", f"INSERT INTO {t_comp1} VALUES ({rf}, '{tk}', {pts}, {bon}) ON CONFLICT (ref_id, tag_key) DO UPDATE SET pts = excluded.pts, bonus = excluded.bonus;")
                    )
            elif op == "upsert_c2":
                fid = rng.randint(1, 12)
                ac = rng.choice(alloc_keys)
                sl = rng.randint(1, 3)
                amt = f"{rng.randint(10, 80) * 1.5:.1f}"
                af = rng.choice([0, 1])
                if dialect == Dialect.MYSQL:
                    mutations.append(
                        (f"m{s_idx}_ups_c2", f"INSERT INTO {t_comp2} VALUES ({fid}, '{ac}', {sl}, {amt}, {af}) ON DUPLICATE KEY UPDATE amount = VALUES(amount), active_flag = VALUES(active_flag);")
                    )
                else:
                    mutations.append(
                        (f"m{s_idx}_ups_c2", f"INSERT INTO {t_comp2} VALUES ({fid}, '{ac}', {sl}, {amt}, {af}) ON CONFLICT (fact_id, alloc_code, slot_no) DO UPDATE SET amount = excluded.amount, active_flag = excluded.active_flag;")
                    )
            elif op == "upd_dim_quota":
                pid = rng.randint(1, num_dims)
                new_t = "NULL" if rng.random() < 0.25 else f"'{rng.choice(tiers)}'"
                new_r = "NULL" if rng.random() < 0.2 else str(rng.randint(1, num_roots))
                mutations.append(
                    (f"m{s_idx}_upd_dim", f"UPDATE {t_dim} SET tier = {new_t}, root_id = {new_r}, quota = quota + {rng.randint(5, 25)} WHERE id = {pid};")
                )
            elif op == "upd_root_active":
                rid = rng.randint(1, num_roots)
                mutations.append(
                    (f"m{s_idx}_upd_root", f"UPDATE {t_root} SET active = 1 - active WHERE id = {rid};")
                )
            elif op == "ins_select_fact":
                fid = next_fact_id
                next_fact_id += 1
                mutations.append(
                    (
                        f"m{s_idx}_ins_sel",
                        f"INSERT INTO {t_fact} SELECT {fid}, id, code, 'auto', 10, 20.0, 5, 2, 1 FROM {t_root} WHERE id = 1;",
                    )
                )
            elif op == "del_comp_row":
                rf = rng.randint(1, 10)
                tk = rng.choice(tag_keys)
                mutations.append((f"m{s_idx}_del_c1", f"DELETE FROM {t_comp1} WHERE ref_id = {rf} AND tag_key = '{tk}';"))

        # Multi-level nested savepoint trees + rollback + commit + drain + reseed
        begin_sql = "START TRANSACTION;" if dialect == Dialect.MYSQL else "BEGIN;"
        tx_fid1 = next_fact_id
        tx_fid2 = next_fact_id + 1
        tx_fid3 = next_fact_id + 2
        next_fact_id += 3

        mutations.extend([
            ("tx1_begin", begin_sql),
            ("tx1_ins", f"INSERT INTO {t_fact} VALUES ({tx_fid1}, 1, '{cats_a[0]}', '{cats_b[0]}', 42, 18.5, 9, 3, 1);"),
            ("tx1_sp1", "SAVEPOINT sp_fuzz_1;"),
            ("tx1_upd_in_sp1", f"UPDATE {t_fact} SET grp_str = '{cats_a[-1]}', weight_cnt = weight_cnt + 11 WHERE flag = 1;"),
            ("tx1_sp2", "SAVEPOINT sp_fuzz_2;"),
            ("tx1_del_in_sp2", f"DELETE FROM {t_fact} WHERE grp_str = '{cats_a[0]}';"),
            ("tx1_sp3", "SAVEPOINT sp_fuzz_3;"),
            ("tx1_drain_in_sp3", f"DELETE FROM {t_fact};"),
            ("tx1_rb_sp3", "ROLLBACK TO SAVEPOINT sp_fuzz_3;"),
            ("tx1_rb_sp2", "ROLLBACK TO SAVEPOINT sp_fuzz_2;"),
            ("tx1_rb_sp1", "ROLLBACK TO SAVEPOINT sp_fuzz_1;"),
            ("tx1_upd_after_rb", f"UPDATE {t_dim} SET quota = quota + 7 WHERE id IN (1, 2);"),
            ("tx1_rel_sp1", "RELEASE SAVEPOINT sp_fuzz_1;"),
            ("tx1_rollback", "ROLLBACK;"),
            ("tx2_begin", begin_sql),
            ("tx2_ins", f"INSERT INTO {t_fact} VALUES ({tx_fid2}, 2, '{cats_a[1]}', NULL, 19, 7.5, 14, 4, 1);"),
            ("tx2_del_c1", f"DELETE FROM {t_comp1} WHERE ref_id = 1;"),
            ("tx2_commit", "COMMIT;"),
            ("drain_fact", f"DELETE FROM {t_fact};"),
            (
                "reseed_after_drain",
                f"INSERT INTO {t_fact} VALUES "
                f"({tx_fid3}, 1, '{cats_a[0]}', '{cats_b[0]}', 100, 50.0, 12, 2, 1), "
                f"({tx_fid3 + 1}, 2, '{cats_a[1]}', NULL, NULL, 15.0, 6, 4, 0);",
            ),
        ])

        return schema_sql, views, seed_sqls, mutations




class VariableSchemaDifferentialFuzzer:
    """Deterministic schema-graph generator that supplements the legacy templates."""

    def __init__(self, seed: int = 20261107) -> None:
        self.seed = seed

    def generate_scenario(
        self, scenario_idx: int, *, views_count: int = 24, mutations_count: int = 24
    ) -> Tuple[str, List[Tuple[str, str]], List[str], List[Tuple[str, str]], Dict[str, Any]]:
        rng = random.Random((self.seed * 1000003) ^ (scenario_idx * 104729))
        prefix = f"vs{scenario_idx}"
        count = rng.randint(2, 6)
        topology = rng.choice(["chain", "star", "disconnected", "self", "bridge"])
        if topology == "bridge" and count < 3:
            topology = "chain"
        names = [f"{prefix}_t{i}" for i in range(count)]

        edges: List[Tuple[int, int, str]] = []
        if topology == "chain":
            edges = [(i, i - 1, f"parent_{i - 1}_id") for i in range(1, count)]
        elif topology == "star":
            edges = [(i, 0, "parent_0_id") for i in range(1, count)]
        elif topology == "disconnected":
            # Leave table 0 isolated; later tables may form separate components.
            edges = [(i, i - 1, f"parent_{i - 1}_id") for i in range(2, count) if i % 3 == 2]
        elif topology == "self":
            edges = [(0, 0, "parent_0_id")] + [
                (i, i - 1, f"parent_{i - 1}_id") for i in range(2, count) if i % 2 == 0
            ]
        else:
            edges = [(count - 1, 0, "ref_0_id"), (count - 1, 1, "ref_1_id")]

        fks: Dict[int, List[Tuple[int, str]]] = {i: [] for i in range(count)}
        for child, parent, col in edges:
            fks[child].append((parent, col))

        metas: List[Dict[str, Any]] = []
        ddls: List[str] = []
        for i, name in enumerate(names):
            label_nn, qty_nn, amount_nn = (rng.choice([True, False]) for _ in range(3))
            amount_type = rng.choice(["REAL", "NUMERIC(18, 4)", "INTEGER"])
            composite = i == count - 1 and count > 2 and rng.random() < 0.5
            unique_label, check_qty, check_amount = (
                rng.choice([True, False]) for _ in range(3)
            )
            cols = [
                "id INTEGER NOT NULL",
                f"label VARCHAR(40){' NOT NULL' if label_nn else ''}",
                f"qty INTEGER{' NOT NULL DEFAULT 0' if qty_nn else ''}",
                f"amount {amount_type}{' NOT NULL DEFAULT 0' if amount_nn else ''}",
            ]
            extras = []
            for extra_name, extra_type in rng.sample(
                [("note", "VARCHAR(48)"), ("rank", "INTEGER"), ("ratio", "REAL"), ("tag", "VARCHAR(24)")],
                rng.randint(0, 2),
            ):
                nullable = rng.choice([True, False])
                default = ""
                if not nullable and rng.random() < 0.5:
                    default = " DEFAULT 0" if extra_type in ("INTEGER", "REAL") else " DEFAULT 'default'"
                cols.append(f"{extra_name} {extra_type}{' NOT NULL' if not nullable else ''}{default}")
                extras.append((extra_name, extra_type, nullable))

            constraints: List[str] = []
            if composite:
                cols.append("slot INTEGER NOT NULL")
                constraints.append("PRIMARY KEY (id, slot)")
            else:
                constraints.append("PRIMARY KEY (id)")
            for parent, fk_col in fks[i]:
                # Keep all column definitions before table-level constraints.
                cols.append(f"{fk_col} INTEGER")
                constraints.append(
                    f"FOREIGN KEY ({fk_col}) REFERENCES {names[parent]} (id) ON DELETE CASCADE"
                )
            if unique_label:
                constraints.append("UNIQUE (label)")
            if check_qty:
                constraints.append("CHECK (qty IS NULL OR qty >= 0)")
            if check_amount:
                constraints.append("CHECK (amount IS NULL OR amount >= 0)")
            ddls.append(
                "CREATE TABLE " + name + " (" + chr(10) + "    "
                + ("," + chr(10) + "    ").join(cols + constraints)
                + chr(10) + ");"
            )
            metas.append({
                "name": name, "label_nn": label_nn, "qty_nn": qty_nn,
                "amount_nn": amount_nn, "amount_type": amount_type,
                "composite": composite, "extras": extras,
                "fk_cols": [col for _, col in fks[i]],
                "unique_label": unique_label, "check_qty": check_qty,
                "check_amount": check_amount,
            })

        seed_sql: List[str] = []
        row_counts: List[int] = []
        for i, meta in enumerate(metas):
            nrows = rng.randint(1, 4)
            row_counts.append(nrows)
            fields = ["id"] + (["slot"] if meta["composite"] else [])
            fields += ["label", "qty", "amount"] + [x[0] for x in meta["extras"]] + meta["fk_cols"]
            for rid in range(1, nrows + 1):
                vals = [str(rid)] + ([str(rid)] if meta["composite"] else [])
                vals.append("NULL" if not meta["label_nn"] and rng.random() < 0.2 else f"'{prefix}_{i}_{rid}'")
                vals.append("NULL" if not meta["qty_nn"] and rng.random() < 0.25 else str(rng.randint(0, 100)))
                if not meta["amount_nn"] and rng.random() < 0.2:
                    vals.append("NULL")
                elif meta["amount_type"] == "INTEGER":
                    vals.append(str(rng.randint(0, 100)))
                else:
                    vals.append(f"{rng.randint(0, 100) + 0.125:.3f}")
                for extra_name, extra_type, nullable in meta["extras"]:
                    null_value = nullable and rng.random() < 0.2
                    if null_value:
                        vals.append("NULL")
                    elif extra_type in ("INTEGER", "REAL"):
                        vals.append(str(rng.randint(0, 20)))
                    else:
                        vals.append(f"'{prefix}_{extra_name}_{rid}'")
                for parent, _ in fks[i]:
                    vals.append("NULL" if parent == i else str(rng.randint(1, row_counts[parent])))
                seed_sql.append(
                    f"INSERT INTO {meta['name']} ({', '.join(fields)}) VALUES ({', '.join(vals)});"
                )

        views: List[Tuple[str, str]] = []
        for i, meta in enumerate(metas):
            for suffix, select in [
                ("projection", f"SELECT * FROM {meta['name']}"),
                ("group", f"SELECT label, COUNT(*) AS row_count, SUM(qty) AS total_qty, "
                          f"AVG(amount) AS average_amount FROM {meta['name']} GROUP BY label"),
            ]:
                if len(views) < views_count:
                    view_name = f"{prefix}_v{i}_{suffix}"
                    views.append((view_name, f"CREATE VIEW {view_name} AS {select};"))
        for edge_idx, (child, parent, fk_col) in enumerate(edges):
            if len(views) >= views_count:
                break
            view_name = f"{prefix}_v_join_{edge_idx}"
            select = (
                f"SELECT p.id AS parent_id, c.id AS child_id, c.label, c.qty "
                f"FROM {names[parent]} p LEFT JOIN {names[child]} c ON p.id = c.{fk_col}"
            )
            views.append((view_name, f"CREATE VIEW {view_name} AS {select};"))

        mutations: List[Tuple[str, str]] = []
        # Delete child tables before their parents. MySQL/MariaDB do not fire
        # child-table triggers for FK-driven ON DELETE CASCADE actions, so a
        # parent-first delete would test a documented engine limitation rather
        # than the row-trigger DML path this differential scenario is intended
        # to exercise. Inserts and updates remain schema-aware and use the
        # same deterministic table order as before.
        for i in reversed(range(len(metas))):
            meta = metas[i]
            if len(mutations) + 3 > mutations_count:
                break
            new_id = row_counts[i] + 1
            fields = ["id"] + (["slot"] if meta["composite"] else [])
            fields += ["label", "qty", "amount"] + [x[0] for x in meta["extras"]] + meta["fk_cols"]
            vals = [str(new_id)] + ([str(new_id)] if meta["composite"] else [])
            vals += [f"'{prefix}_{i}_inserted'", str(rng.randint(1, 100))]
            vals.append(str(rng.randint(1, 100)) if meta["amount_type"] == "INTEGER" else f"{rng.randint(1, 100) + 0.375:.3f}")
            for extra_name, extra_type, _ in meta["extras"]:
                vals.append(str(rng.randint(1, 20)) if extra_type in ("INTEGER", "REAL") else f"'{prefix}_{extra_name}_inserted'")
            # Parent rows may have been removed by earlier generated DELETEs.
            # FK columns are nullable, so keep these later INSERTs valid while
            # the initial seed data continues to exercise non-NULL relationships.
            vals += ["NULL" for _ in fks[i]]
            # A direct parent delete can cascade into child tables without
            # firing their maintenance triggers on MySQL/MariaDB. Keep parent
            # and self-referencing deletes out of this row-trigger differential
            # scenario; leaf-table deletes still exercise DELETE maintenance.
            has_referencing_children = any(
                parent_idx == i
                for child_refs in fks.values()
                for parent_idx, _ in child_refs
            )
            delete_guard = " AND 1 = 0" if has_referencing_children else ""
            mutations.extend([
                (f"{prefix}_t{i}_insert", f"INSERT INTO {meta['name']} ({', '.join(fields)}) VALUES ({', '.join(vals)});"),
                (f"{prefix}_t{i}_update", f"UPDATE {meta['name']} SET qty = COALESCE(qty, 0) + {rng.randint(1, 9)}, amount = COALESCE(amount, 0) + 1 WHERE id = 1;"),
                (f"{prefix}_t{i}_delete", f"DELETE FROM {meta['name']} WHERE id = {row_counts[i]}{delete_guard};"),
            ])

        shape = {
            "topology": topology,
            "table_count": count,
            "column_counts": tuple(
                4 + len(m["extras"]) + len(m["fk_cols"]) + int(m["composite"])
                for m in metas
            ),
            "amount_types": tuple(sorted({m["amount_type"] for m in metas})),
            "nullable_columns": sum(
                int(not m["label_nn"]) + int(not m["qty_nn"]) + int(not m["amount_nn"])
                for m in metas
            ),
            "foreign_key_count": len(edges),
            "composite_primary_key_tables": sum(int(m["composite"]) for m in metas),
            "unique_label_constraints": sum(int(m["unique_label"]) for m in metas),
            "check_constraints": sum(
                int(m["check_qty"]) + int(m["check_amount"]) for m in metas
            ),
        }
        return "\n".join(ddls), views, seed_sql, mutations, shape


def run_suite_12_differential_random_fuzzer(runner: MVTestRunner) -> None:
    """
    Suite 12: Extensive Multi-Order-of-Magnitude Differential Random SQL Fuzzer.
    Generates valid random 5-table schemas (single PK, 2-column & 3-column composite PKs,
    self-referential hierarchy trees, and multi-hop foreign keys), up to 80 random views
    across all 7 SQL families per schema, random seed datasets, and 45-80+ random DML
    and 3-level nested SAVEPOINT/ROLLBACK/COMMIT steps per scenario.
    Runs both deterministic baseline seeds and fresh cryptographic random 32-bit seeds
    (logged upfront for deterministic reproduction via MATVIEW_FUZZ_SEEDS=<seed>).
    """
    is_ci = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
    scale = max(1, int(os.environ.get("MATVIEW_FUZZ_SCALE", "2" if is_ci else "1")))
    seeds_env = os.environ.get("MATVIEW_FUZZ_SEEDS")
    if seeds_env:
        seeds = [int(s.strip()) for s in seeds_env.split(",") if s.strip()]
    else:
        det_seeds = [20261106, 1337]
        if scale > 1:
            extra_det = [424242, 992138513, 987654321, 104729, 314159]
            det_seeds.extend(extra_det[: min(len(extra_det), scale)])
        num_rand = int(os.environ.get("MATVIEW_FUZZ_RANDOM_SEEDS", str(2 * scale)))
        rand_seeds = [secrets.randbits(32) for _ in range(num_rand)]
        seeds = det_seeds + rand_seeds

    num_scenarios = max(1, int(os.environ.get("MATVIEW_FUZZ_SCENARIOS", str(2 if scale == 1 else min(4, 1 + scale)))))
    views_count = int(os.environ.get("MATVIEW_FUZZ_VIEWS_PER_SCHEMA", str(60 if scale == 1 else min(80, 50 + scale * 10))))
    mutations_count = int(os.environ.get("MATVIEW_FUZZ_MUTATIONS", str(45 if scale == 1 else min(80, 40 + scale * 10))))

    manifest_path = os.environ.get("MATVIEW_FUZZ_MANIFEST", "fuzz-campaign.json")
    manifest = {
        "schema_version": 1,
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "commit": os.environ.get("GITHUB_SHA"),
        "job": os.environ.get("GITHUB_JOB"),
        "target": os.environ.get("MATVIEW_FUZZ_TARGET") or os.environ.get("GITHUB_JOB"),
        "python": __import__("sys").version.split()[0],
        "default_dialect": runner.default_dialect.value,
        "scale": scale,
        "seeds": seeds,
        "scenarios_per_seed": num_scenarios,
        "views_per_schema": views_count,
        "mutations_per_scenario": mutations_count,
        "runs": [],
    }

    def write_manifest() -> None:
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
        os.replace(tmp_path, manifest_path)

    def run_fuzz_batch(suite_name: str, schema_sql: str, views: Sequence[Tuple[str, str]],
                       seed_sqls: Sequence[str], mutations: Sequence[Tuple[str, str]], **kwargs: Any) -> None:
        dialect = kwargs.get("dialect", runner.default_dialect)
        record = {
            "suite": suite_name,
            "seed": current_seed,
            "scenario": current_scenario,
            "dialect": dialect.value,
            "transition_tables": bool(kwargs.get("pg_use_transition_tables", False)),
            "incremental_backfill": bool(kwargs.get("incremental_backfill", False)),
            "status": "running",
        }
        manifest["runs"].append(record)
        write_manifest()
        try:
            runner.run_batch(suite_name, schema_sql, views, seed_sqls, mutations, **kwargs)
        except Exception as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            write_manifest()
            raise
        else:
            record["status"] = "passed"
            write_manifest()

    print(
        f"  [FUZZ CONFIG] scale={scale}, seeds={seeds} ({len(seeds)} seeds x {num_scenarios} schemas x {views_count} views x {mutations_count} mutations)"
    )

    current_seed = None
    current_scenario = None
    write_manifest()

    for s_val in seeds:
        current_seed = s_val
        print(f"  [FUZZ SEED] seed={s_val} (reproduce with: MATVIEW_FUZZ_SEEDS={s_val} python3 test_matview_sync.py --fuzz)")
        fuzzer = DifferentialSQLFuzzer(seed=s_val)
        for sc_idx in range(1, num_scenarios + 1):
            current_scenario = sc_idx
            # 1. Default runner dialect (SQLite locally, or live Postgres/MySQL/SQLite-compat in CI)
            s_def, v_def, seed_def, m_def = fuzzer.generate_scenario(
                sc_idx, runner.default_dialect, views_count=views_count, mutations_count=mutations_count
            )
            run_fuzz_batch(
                f"Suite 12a (Differential Fuzzer seed={s_val} sc={sc_idx} - Default Engine)",
                s_def,
                v_def,
                seed_def,
                m_def,
                dialect=runner.default_dialect,
            )

            # 2. PostgreSQL dialect (row-level triggers & statement-level transition tables + incremental backfill)
            s_pg, v_pg, seed_pg, m_pg = fuzzer.generate_scenario(
                sc_idx, Dialect.POSTGRES, views_count=views_count, mutations_count=mutations_count
            )
            run_fuzz_batch(
                f"Suite 12b (Differential Fuzzer seed={s_val} sc={sc_idx} - PostgreSQL Row Triggers)",
                s_pg,
                v_pg,
                seed_pg,
                m_pg,
                dialect=Dialect.POSTGRES,
                pg_use_transition_tables=False,
            )
            run_fuzz_batch(
                f"Suite 12c (Differential Fuzzer seed={s_val} sc={sc_idx} - PostgreSQL Transition Tables + Incremental)",
                s_pg,
                v_pg,
                seed_pg,
                m_pg,
                dialect=Dialect.POSTGRES,
                pg_use_transition_tables=True,
                incremental_backfill=True,
                batch_size=2,
            )

            # 3. MySQL dialect (incremental backfill + stored procedures + triggers)
            s_my, v_my, seed_my, m_my = fuzzer.generate_scenario(
                sc_idx, Dialect.MYSQL, views_count=views_count, mutations_count=mutations_count
            )
            run_fuzz_batch(
                f"Suite 12d (Differential Fuzzer seed={s_val} sc={sc_idx} - MySQL Incremental & Triggers)",
                s_my,
                v_my,
                seed_my,
                m_my,
                dialect=Dialect.MYSQL,
                incremental_backfill=True,
                batch_size=2,
            )


            # 4. Run a separately generated schema graph to exercise variable
            # table/column/key/constraint/FK structures beyond the legacy templates.
            s_var, v_var, seed_var, m_var, shape_var = VariableSchemaDifferentialFuzzer(
                seed=s_val
            ).generate_scenario(sc_idx)
            print(
                f"  [SCHEMA SHAPE] seed={s_val} scenario={sc_idx}: "
                f"tables={shape_var['table_count']} topology={shape_var['topology']} "
                f"columns={shape_var['column_counts']} fks={shape_var['foreign_key_count']} "
                f"composite_pks={shape_var['composite_primary_key_tables']}"
            )
            run_fuzz_batch(
                f"Suite 12e (Variable Schema seed={s_val} sc={sc_idx})",
                s_var, v_var, seed_var, m_var, dialect=Dialect.SQLITE,
            )


# ==============================================================================
# SUITE 13: Permanent Fuzzing-Discovered Regression Corpus
# ==============================================================================
def get_fuzz_regression_corpus() -> List[Dict[str, Any]]:
    """
    Returns the permanent corpus of minimal, self-contained regression scenarios
    for every failure uncovered by differential fuzzing across SQLite, PostgreSQL,
    and MySQL/MariaDB. Every entry in this corpus is executed across all applicable
    dialects and modes on every test and CI run to guarantee zero regressions.
    """
    return [
        {
            "id": "corpus_01_sqlite_bitwise_distinct_and_delete_recompute",
            "description": (
                "Discovered by DifferentialSQLFuzzer on SQLite: views combining BIT_XOR/BIT_AND/BIT_OR "
                "with SUM(DISTINCT ...)/COUNT(DISTINCT ...) (group_key_batch backfill) or DELETE recomputation "
                "previously fell through to SUM(...) in _sql_backfill_agg_expr on Dialect.SQLITE."
            ),
            "dialects": (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL),
            "schema": """
                CREATE TABLE fc1_t1 (
                    id INTEGER PRIMARY KEY,
                    grp_str TEXT NOT NULL,
                    sub_cat TEXT,
                    val_flt REAL,
                    weight_cnt INTEGER NOT NULL,
                    flag INTEGER NOT NULL
                );
            """,
            "views": [
                (
                    "fc1_v_bit_dist",
                    "CREATE VIEW fc1_v_bit_dist AS SELECT grp_str, SUM(val_flt) AS s_flt, BIT_XOR(weight_cnt) AS bx_qty, BIT_AND(weight_cnt) AS ba_qty, BIT_OR(weight_cnt) AS bo_qty, SUM(DISTINCT weight_cnt) AS sd_qty, COUNT(DISTINCT sub_cat) AS cd_sc, SUM(weight_cnt) AS s_qty FROM fc1_t1 WHERE (flag = 1 OR weight_cnt >= 4) AND weight_cnt <= 15 GROUP BY grp_str;",
                ),
            ],
            "seed": [
                "INSERT INTO fc1_t1 VALUES (1, 'alpha', 'hw', 100.0, 7, 1);",
                "INSERT INTO fc1_t1 VALUES (2, 'alpha', 'hw', 100.0, 7, 1);",
                "INSERT INTO fc1_t1 VALUES (3, 'alpha', 'sw', 50.0, 8, 1);",
                "INSERT INTO fc1_t1 VALUES (4, 'beta', NULL, 25.0, 12, 1);",
                "INSERT INTO fc1_t1 VALUES (5, 'beta', 'ai', 75.0, 5, 0);",
            ],
            "mutations": [
                ("del_dup_1", "DELETE FROM fc1_t1 WHERE id = 1;"),
                ("upd_bit", "UPDATE fc1_t1 SET weight_cnt = 11 WHERE id = 3;"),
                ("ins_new", "INSERT INTO fc1_t1 VALUES (6, 'alpha', 'cloud', 30.0, 3, 1);"),
                ("del_extrema", "DELETE FROM fc1_t1 WHERE id = 2;"),
                ("tx_begin", "BEGIN;"),
                ("tx_sp", "SAVEPOINT sp_fc1;"),
                ("tx_del_beta", "DELETE FROM fc1_t1 WHERE grp_str = 'beta';"),
                ("tx_rb", "ROLLBACK TO SAVEPOINT sp_fc1;"),
                ("tx_commit", "COMMIT;"),
            ],
        },
        {
            "id": "corpus_02_pg_mysql_offset_batch_left_join_and_group_order",
            "description": (
                "Discovered by DifferentialSQLFuzzer in CI Run #15 on PostgreSQL 14..18 (fz2_v22_left_join_proj:init): "
                "offset_batch incremental backfill loops (LIMIT batch_size OFFSET offset) over LEFT JOIN and GROUP BY "
                "queries without an explicit ORDER BY caused PostgreSQL to switch between Nested Loop and Hash Right Join "
                "mid-loop as OFFSET grew, duplicating (2, 'P_2', None, None, None) and skipping (4, 'P_4', 5, 'beta', 5)."
            ),
            "dialects": (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL),
            "schema": """
                CREATE TABLE fc2_orgs (
                    id INTEGER PRIMARY KEY,
                    pname TEXT NOT NULL,
                    tier TEXT
                );
                CREATE TABLE fc2_events (
                    id INTEGER PRIMARY KEY,
                    parent_id INTEGER,
                    grp_str TEXT NOT NULL,
                    weight_cnt INTEGER NOT NULL
                );
            """,
            "views": [
                (
                    "fc2_v_left_join_proj",
                    "CREATE VIEW fc2_v_left_join_proj AS SELECT p.id AS pid, p.pname, m.id AS mid, m.grp_str, m.weight_cnt FROM fc2_orgs p LEFT JOIN fc2_events m ON p.id = m.parent_id;",
                ),
                (
                    "fc2_v_left_join_agg",
                    "CREATE VIEW fc2_v_left_join_agg AS SELECT p.pname, p.tier, COUNT(*) AS row_cnt, COUNT(m.id) AS match_cnt, SUM(m.weight_cnt) AS sum_w, MIN(m.weight_cnt) AS min_w, MAX(m.weight_cnt) AS max_w FROM fc2_orgs p LEFT JOIN fc2_events m ON p.id = m.parent_id GROUP BY p.pname, p.tier;",
                ),
                (
                    "fc2_v_right_join_proj",
                    "CREATE VIEW fc2_v_right_join_proj AS SELECT m.id AS mid, m.grp_str, p.id AS pid, p.pname, p.tier FROM fc2_events m RIGHT JOIN fc2_orgs p ON m.parent_id = p.id;",
                ),
            ],
            "seed": [
                "INSERT INTO fc2_orgs VALUES (1, 'P_1', 'gold'), (2, 'P_2', 'silver'), (3, 'P_3', 'vip'), (4, 'P_4', NULL), (5, 'P_5', 'std');",
                "INSERT INTO fc2_events VALUES (1, 3, 'omega', 11), (2, 3, 'omega', 11), (3, 3, 'gamma', 10), (4, 3, 'gamma', 10), (5, 4, 'beta', 5), (6, 4, 'alpha', 8), (7, NULL, 'alpha', 14), (8, 1, 'beta', 13), (9, 3, 'gamma', 13), (10, 3, 'omega', 1);",
            ],
            "mutations": [
                ("ins_unmatched_org", "INSERT INTO fc2_orgs VALUES (6, 'P_6', 'gold');"),
                ("attach_to_p2", "INSERT INTO fc2_events VALUES (11, 2, 'omega', 9);"),
                ("detach_from_p4", "UPDATE fc2_events SET parent_id = NULL WHERE id = 5;"),
                ("del_p3_events", "DELETE FROM fc2_events WHERE parent_id = 3 AND weight_cnt = 10;"),
            ],
        },
        {
            "id": "corpus_03_mysql_empty_set_and_all_null_bitwise_identity",
            "description": (
                "Discovered by DifferentialSQLFuzzer in CI Run #15 on MySQL 8.4..26.10 & MariaDB 10.11..13.0 "
                "(fz1_v16_global_full:drain_main): when an ungrouped global aggregate or all-NULL group evaluates "
                "BIT_XOR/BIT_OR/BIT_AND over 0 non-null rows, MySQL/MariaDB return 0 / 0 / 18446744073709551615 "
                "whereas PostgreSQL/SQLite return NULL."
            ),
            "dialects": (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL),
            "schema": """
                CREATE TABLE fc3_txns (
                    id INTEGER PRIMARY KEY,
                    grp_str TEXT NOT NULL,
                    val_int INTEGER,
                    val_flt REAL,
                    weight_cnt INTEGER NOT NULL,
                    flag INTEGER NOT NULL
                );
            """,
            "views": [
                (
                    "fc3_v_global_full",
                    "CREATE VIEW fc3_v_global_full AS SELECT COUNT(*) AS cnt, COUNT(val_int) AS ci, SUM(val_int) AS si, AVG(val_flt) AS af, MIN(weight_cnt) AS mnw, MAX(weight_cnt) AS mxw, VAR_POP(val_flt) AS vp, BIT_XOR(weight_cnt) AS bx, BIT_OR(weight_cnt) AS bo, BIT_AND(weight_cnt) AS ba FROM fc3_txns;",
                ),
                (
                    "fc3_v_grp_nullable_bit",
                    "CREATE VIEW fc3_v_grp_nullable_bit AS SELECT grp_str, COUNT(*) AS cnt, COUNT(val_int) AS ci, BIT_XOR(val_int) AS bx_vi, BIT_OR(val_int) AS bo_vi, BIT_AND(val_int) AS ba_vi FROM fc3_txns GROUP BY grp_str;",
                ),
            ],
            "seed": [
                "INSERT INTO fc3_txns VALUES (1, 'alpha', NULL, 10.0, 6, 1);",
                "INSERT INTO fc3_txns VALUES (2, 'alpha', NULL, 20.0, 3, 1);",
                "INSERT INTO fc3_txns VALUES (3, 'beta', 12, 15.0, 5, 0);",
            ],
            "mutations": [
                ("set_alpha_nonnull", "UPDATE fc3_txns SET val_int = 9 WHERE id = 1;"),
                ("set_alpha_back_to_null", "UPDATE fc3_txns SET val_int = NULL WHERE id = 1;"),
                ("drain_all_rows", "DELETE FROM fc3_txns;"),
                ("reseed_after_drain", "INSERT INTO fc3_txns VALUES (4, 'gamma', NULL, 50.0, 12, 1);"),
                ("drain_again", "DELETE FROM fc3_txns;"),
            ],
        },
        {
            "id": "corpus_04_filter_where_greedy_regex_algebraic_delta",
            "description": (
                "Discovered by DifferentialSQLFuzzer on SQLite & PostgreSQL (fz1_v05_alg_grp_1): "
                "greedy (.*) in _parse_single_agg_expr swallowed the closing paren of fn(...) when "
                "FILTER (WHERE ...) followed, demoting the view from ALGEBRAIC_DELTA to GROUP_SCOPED_DELTA."
            ),
            "dialects": (Dialect.SQLITE, Dialect.POSTGRES),
            "schema": """
                CREATE TABLE fc4_txns (
                    id INTEGER PRIMARY KEY,
                    grp_str TEXT NOT NULL,
                    val_int INTEGER,
                    val_flt REAL,
                    weight_cnt INTEGER NOT NULL,
                    flag INTEGER NOT NULL
                );
            """,
            "views": [
                (
                    "fc4_v_filter_where_alg",
                    "CREATE VIEW fc4_v_filter_where_alg AS SELECT LOWER(grp_str) AS l_grp, COUNT(*) AS c_all, MAX(weight_cnt) AS mx_w, VAR_POP(val_flt) AS vp_flt, SUM(val_int) AS s_int, SUM(weight_cnt) FILTER (WHERE flag = 1) AS filt_w, COUNT(*) FILTER (WHERE flag = 0) AS filt_c0, BIT_XOR(weight_cnt) AS bx_w, SUM(val_int * weight_cnt) AS w_sum FROM fc4_txns GROUP BY LOWER(grp_str) HAVING COALESCE(SUM(weight_cnt), 0) >= 5;",
                ),
            ],
            "seed": [
                "INSERT INTO fc4_txns VALUES (1, 'Alpha', 10, 5.0, 4, 1);",
                "INSERT INTO fc4_txns VALUES (2, 'alpha', 20, 15.0, 6, 0);",
                "INSERT INTO fc4_txns VALUES (3, 'Beta', 5, 8.0, 3, 1);",
            ],
            "mutations": [
                ("upd_flag_flip", "UPDATE fc4_txns SET flag = 1 - flag, weight_cnt = weight_cnt + 3 WHERE id IN (1, 3);"),
                ("ins_beta", "INSERT INTO fc4_txns VALUES (4, 'BETA', 15, 12.0, 8, 1);"),
                ("del_alpha_1", "DELETE FROM fc4_txns WHERE id = 1;"),
                ("del_alpha_2", "DELETE FROM fc4_txns WHERE id = 2;"),
            ],
        },
        {
            "id": "corpus_05_pg_63char_identifier_truncation_collision",
            "description": (
                "Discovered by DifferentialSQLFuzzer in CI Run #18 on PostgreSQL 14..18 "
                "(fz3_v046_global_filtered_nocnt on fz3_records with transition tables): "
                "__matview_fn_sync_fz3_v046_global_filtered_nocnt__fz3_records_ai/_ad/_au were 64 characters long, "
                "causing PostgreSQL to truncate all three to the same 63-char identifier (..._a) and overwrite "
                "the INSERT trigger function with the UPDATE trigger function referencing __mv_old_rows."
            ),
            "dialects": (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL),
            "schema": """
                CREATE TABLE fz3_records (
                    id INTEGER PRIMARY KEY,
                    grp_str TEXT NOT NULL,
                    val_int INTEGER,
                    val_flt REAL,
                    weight_cnt INTEGER NOT NULL,
                    flag INTEGER NOT NULL
                );
            """,
            "views": [
                (
                    "fz3_v046_global_filtered_nocnt",
                    "CREATE VIEW fz3_v046_global_filtered_nocnt AS SELECT SUM(val_int) AS si, AVG(val_flt) AS af, MIN(weight_cnt) AS mnw, MAX(weight_cnt) AS mxw, COUNT(DISTINCT grp_str) AS cd_g FROM fz3_records WHERE flag = 1 AND weight_cnt >= 10;",
                ),
            ],
            "seed": [
                "INSERT INTO fz3_records VALUES (1, 'alpha', 10, 20.0, 12, 1);",
                "INSERT INTO fz3_records VALUES (2, 'beta', 30, 40.0, 5, 1);",
            ],
            "mutations": [
                ("ins_rec", "INSERT INTO fz3_records VALUES (3, 'gamma', 25, 15.0, 14, 1);"),
                ("upd_rec", "UPDATE fz3_records SET weight_cnt = 18, val_int = 50 WHERE id = 2;"),
                ("del_rec", "DELETE FROM fz3_records WHERE id = 1;"),
            ],
        },
        {
            "id": "corpus_06_mariadb_not_in_subquery_empty_transition_with_null_lhs",
            "description": (
                "Discovered by DifferentialSQLFuzzer in CI Run #18 on MariaDB 10.11 (seed=992138513, fz3_v069_subq_not_in:tx1_upd_after_rb) "
                "and MariaDB 11.4 (seed=424242, fz3_v071_subq_not_in:m23_upd_dim): when a view with `parent_id NOT IN (SELECT id FROM merchants WHERE quota < 25 AND id IS NOT NULL)` "
                "has NULL values in `parent_id` and an UPDATE on `merchants` causes the subquery to transition from 1 row to 0 rows (empty), "
                "MariaDB's cached trigger Item_in_optimizer mis-evaluates `NULL NOT IN (empty)` as FALSE instead of TRUE unless lowered to NOT EXISTS."
            ),
            "dialects": (Dialect.SQLITE, Dialect.POSTGRES, Dialect.MYSQL),
            "schema": """
                CREATE TABLE fc6_merchants (
                    id INTEGER PRIMARY KEY,
                    pname TEXT NOT NULL,
                    tier TEXT,
                    quota INTEGER NOT NULL
                );
                CREATE TABLE fc6_events (
                    id INTEGER PRIMARY KEY,
                    parent_id INTEGER,
                    grp_str TEXT NOT NULL,
                    weight_cnt INTEGER NOT NULL
                );
            """,
            "views": [
                (
                    "fc6_v_subq_not_in",
                    "CREATE VIEW fc6_v_subq_not_in AS SELECT id, grp_str, weight_cnt FROM fc6_events WHERE parent_id NOT IN (SELECT id FROM fc6_merchants WHERE quota < 25 AND id IS NOT NULL);",
                ),
            ],
            "seed": [
                "INSERT INTO fc6_merchants VALUES (1, 'ORG_1', 'std', 63), (2, 'ORG_2', 'gold', 96), (3, 'ORG_3', 'core', 22), (4, 'ORG_4', NULL, 28);",
                "INSERT INTO fc6_events VALUES (1, 1, 'alpha', 14), (2, NULL, 'gamma', 17), (3, 3, 'zeta', 3), (4, NULL, 'sigma', 6);",
            ],
            "mutations": [
                # Updating id=3 from quota=22 to quota=38 empties the subquery (quota < 25) -> rows with parent_id IS NULL must now be INCLUDED!
                ("empty_subq_upd", "UPDATE fc6_merchants SET tier = NULL, quota = quota + 16 WHERE id = 3;"),
                # Updating id=3 back to quota=18 makes subquery non-empty again -> rows with parent_id IS NULL must now be EXCLUDED!
                ("refill_subq_upd", "UPDATE fc6_merchants SET quota = 18 WHERE id = 3;"),
                # Deleting id=3 empties the subquery again -> rows with parent_id IS NULL must be INCLUDED!
                ("del_subq_row", "DELETE FROM fc6_merchants WHERE id = 3;"),
                # Re-inserting id=3 with quota=20 makes subquery non-empty again!
                ("ins_subq_row", "INSERT INTO fc6_merchants VALUES (3, 'ORG_3', 'core', 20);"),
            ],
        },
    ]


def run_suite_13_fuzz_regression_corpus(runner: MVTestRunner) -> None:
    """
    Suite 13: Executes every scenario in the permanent fuzzing-discovered regression
    corpus across all applicable dialects and execution modes (standard, incremental
    backfill, and PostgreSQL transition tables).
    """
    corpus = get_fuzz_regression_corpus()
    for entry in corpus:
        cid = entry["id"]
        schema = entry["schema"]
        views = entry["views"]
        seed = entry["seed"]
        muts = entry["mutations"]
        for dialect in entry["dialects"]:
            runner.run_batch(
                f"Suite 13 (Fuzz Corpus: {cid} - {dialect.value} std)",
                schema,
                views,
                seed,
                muts,
                dialect=dialect,
                incremental_backfill=False,
            )
            runner.run_batch(
                f"Suite 13 (Fuzz Corpus: {cid} - {dialect.value} inc)",
                schema,
                views,
                seed,
                muts,
                dialect=dialect,
                pg_use_transition_tables=(dialect == Dialect.POSTGRES),
                incremental_backfill=True,
                batch_size=2,
            )




def _run_all_tests() -> None:
    t0 = time.time()
    runner = MVTestRunner()

    fuzz_only = False
    args = list(sys.argv[1:])
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--fuzz":
            fuzz_only = True
        elif arg == "--fuzz-scale" and idx + 1 < len(args):
            os.environ["MATVIEW_FUZZ_SCALE"] = args[idx + 1]
            idx += 1
        elif arg == "--fuzz-seeds" and idx + 1 < len(args):
            os.environ["MATVIEW_FUZZ_SEEDS"] = args[idx + 1]
            idx += 1
        elif arg == "--fuzz-scenarios" and idx + 1 < len(args):
            os.environ["MATVIEW_FUZZ_SCENARIOS"] = args[idx + 1]
            idx += 1
        elif arg == "--fuzz-views" and idx + 1 < len(args):
            os.environ["MATVIEW_FUZZ_VIEWS_PER_SCHEMA"] = args[idx + 1]
            idx += 1
        elif arg == "--fuzz-mutations" and idx + 1 < len(args):
            os.environ["MATVIEW_FUZZ_MUTATIONS"] = args[idx + 1]
            idx += 1
        idx += 1

    if fuzz_only:
        print("=" * 84)
        print("RUNNING STANDALONE DIFFERENTIAL RANDOM SQL FUZZER (SUITES 12 & 13)")
        print("=" * 84)
        run_suite_12_differential_random_fuzzer(runner)
        run_suite_13_fuzz_regression_corpus(runner)
        elapsed = time.time() - t0
        print("=" * 84)
        print(f"FUZZER PASSED SUCCESSFULLY in {elapsed:.2f}s ({runner.total_views} views, {runner.total_assertions} assertions)!")
        print("=" * 84)
        return

    print("=" * 84)
    print("RUNNING EXHAUSTIVE COMBINATORIAL & TRANSACTIONAL MATERIALIZED VIEW TEST SUITE")
    print("=" * 84)

    run_suite_1_single_table_algebraic(runner)
    run_suite_2_holistic_and_having(runner)
    run_suite_3_distinct_aggregates(runner)
    run_suite_4_ungrouped_aggregates(runner)
    run_suite_5_joins(runner)
    run_suite_6_subqueries(runner)
    run_suite_7_set_operations(runner)
    run_suite_8_ctes_windows_topn(runner)
    run_suite_9_cross_dialect_execution(runner)
    run_suite_10_torture_dml_and_transactions(runner)
    run_suite_11_live_engine_50_view_battery_across_all_dialects(runner)
    run_suite_12_differential_random_fuzzer(runner)
    run_suite_13_fuzz_regression_corpus(runner)

    print("-" * 84)
    print(f"COMBINATORIAL & TRANSACTIONAL SUITES PASSED: {runner.total_views} views, {runner.total_assertions} live SQL assertions!")
    print("-" * 84)

    # Run unit tests for generator & manager API
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGeneratorAndManagerCodePaths)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    if not res.wasSuccessful():
        sys.exit(1)

    elapsed = time.time() - t0
    print("=" * 84)
    print(f"ALL TESTS PASSED SUCCESSFULLY in {elapsed:.2f}s ({runner.total_views} views, {runner.total_assertions} assertions)!")
    print("=" * 84)


if __name__ == "__main__":
    _run_all_tests()
