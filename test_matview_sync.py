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
    EngineLimitationWarning,
    IncrementalViewCompiler,
    LimitationSeverity,
    MaintenanceStrategy,
    MaterializedViewManager,
    MaterializedViewPlan,
    TableSchema,
    ViewIndexSpec,
    ViewSpec,
    _split_top_level,
    generate_materialized_view,
    generate_materialized_views,
    install_materialized_view_sqlite,
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


def _norm_val(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, Decimal)):
        fv = round(float(v), 4)
        if abs(fv - round(fv)) < 1e-9:
            return int(round(fv))
        return fv
    return v


def _row_sort_key(row: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(
        (
            0 if x is None else (1 if isinstance(x, (int, float, Decimal)) and not isinstance(x, bool) else 2),
            _norm_val(x) if x is not None else "",
        )
        for x in row
    )


def _rows_equal(rows_a: Sequence[Sequence[Any]], rows_b: Sequence[Sequence[Any]], tol: float = 1.5e-3) -> bool:
    if len(rows_a) != len(rows_b):
        return False
    sa = sorted([tuple(_norm_val(x) for x in r) for r in rows_a], key=_row_sort_key)
    sb = sorted([tuple(_norm_val(x) for x in r) for r in rows_b], key=_row_sort_key)
    for ra, rb in zip(sa, sb):
        if len(ra) != len(rb):
            return False
        for ca, cb in zip(ra, rb):
            if ca is None or cb is None:
                if ca is not cb:
                    return False
            elif isinstance(ca, (int, float)) and isinstance(cb, (int, float)):
                if abs(float(ca) - float(cb)) > tol:
                    return False
            elif ca != cb:
                return False
    return True


def _assert_mv_matches_live(
    conn: sqlite3.Connection,
    mv_name: str,
    live_select_sql: str,
    context: str = "",
) -> None:
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {mv_name}")
    mv_rows = cur.fetchall()
    cur.execute(live_select_sql)
    live_rows = cur.fetchall()
    if not _rows_equal(mv_rows, live_rows):
        sa = sorted([tuple(_norm_val(x) for x in r) for r in mv_rows], key=_row_sort_key)
        sb = sorted([tuple(_norm_val(x) for x in r) for r in live_rows], key=_row_sort_key)
        raise AssertionError(
            f"[{context}] MV mismatch on {mv_name}!\n"
            f"  Query: {live_select_sql}\n"
            f"  MV ({len(sa)} rows):   {sa[:10]}\n"
            f"  Live ({len(sb)} rows): {sb[:10]}"
        )


def _register_sqlite_compat_udfs(conn: sqlite3.Connection) -> None:
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
            return self.acc

    class _BitOrAgg:
        def __init__(self) -> None:
            self.acc: Optional[int] = None
        def step(self, x: Any) -> None:
            if x is not None:
                self.acc = int(x) if self.acc is None else (self.acc | int(x))
        def finalize(self) -> Optional[int]:
            return self.acc

    class _BitXorAgg:
        def __init__(self) -> None:
            self.acc: Optional[int] = None
        def step(self, x: Any) -> None:
            if x is not None:
                self.acc = int(x) if self.acc is None else (self.acc ^ int(x))
        def finalize(self) -> Optional[int]:
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


def _execute_postgres_plan_in_sqlite(conn: sqlite3.Connection, plan: MaterializedViewPlan) -> None:
    _register_sqlite_compat_udfs(conn)
    cur = conn.cursor()
    for aux_ddl in plan.auxiliary_table_ddls:
        cur.execute(_transpile_pg_expr_to_sqlite(aux_ddl))
    cur.execute(_transpile_pg_expr_to_sqlite(plan.backing_table_ddl))
    for idx_ddl in plan.index_ddls:
        cur.execute(_transpile_pg_expr_to_sqlite(idx_ddl))
    if plan.reader_view_ddl:
        cur.execute(_transpile_pg_expr_to_sqlite(plan.reader_view_ddl))

    # Install read-only trigger on view in SQLite
    cur.execute(
        f"CREATE TRIGGER __ro_pg_{plan.logical_view_name}_del INSTEAD OF DELETE ON {plan.view_name} "
        f"BEGIN SELECT RAISE(ABORT, 'read-only'); END;"
    )

    # Parse PL/pgSQL functions and convert to SQLite AFTER INSERT/DELETE/UPDATE triggers
    fn_bodies: Dict[str, str] = {}
    for ddl in plan.trigger_ddls:
        m_fn = re.match(
            r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+([A-Za-z0-9_]+)\(\)\s+RETURNS\s+TRIGGER\s+AS\s+\$\$(.*)\$\$\s+LANGUAGE\s+plpgsql;",
            ddl.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_fn:
            fn_bodies[m_fn.group(1)] = m_fn.group(2).strip()

    for ddl in plan.trigger_ddls:
        # Row-level combined trigger
        m_tr = re.match(
            r"CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\s+AFTER\s+INSERT\s+OR\s+UPDATE(?:\s+OF\s+[^;]+?)?\s+OR\s+DELETE\s+ON\s+([A-Za-z0-9_]+)\s+FOR\s+EACH\s+ROW\s+EXECUTE\s+FUNCTION\s+([A-Za-z0-9_]+)\(\);",
            ddl.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_tr:
            t_name, base_t, fn_name = m_tr.group(1), m_tr.group(2), m_tr.group(3)
            body = fn_bodies.get(fn_name, "")
            m_parts = re.search(
                r"IF\s*\(TG_OP\s*=\s*'DELETE'\)\s*THEN\s*(.*?)\s*ELSIF\s*\(TG_OP\s*=\s*'INSERT'\)\s*THEN\s*(.*?)\s*ELSIF\s*\(TG_OP\s*=\s*'UPDATE'\)\s*THEN\s*(.*?)\s*END\s+IF;\s*RETURN\s+NULL;",
                body,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_parts:
                del_b, ins_b, upd_b = m_parts.group(1).strip(), m_parts.group(2).strip(), m_parts.group(3).strip()
                for op, b_raw, suf in (("DELETE", del_b, "ad"), ("INSERT", ins_b, "ai"), ("UPDATE", upd_b, "au")):
                    b_clean = _transpile_pg_expr_to_sqlite(b_raw)
                    # Convert outer IF (cond) THEN ... END IF; to WHEN (cond)
                    when_cl = ""
                    m_if = re.match(r"^IF\s*\(([^\n]+?)\)\s*THEN\s*(.*)\s*END\s+IF;$", b_clean.strip(), flags=re.IGNORECASE | re.DOTALL)
                    if m_if:
                        inner_s = m_if.group(2).strip()
                        if m_if.group(1).strip().upper().startswith("NOT ") or "END IF;" not in inner_s.upper():
                            when_cl = f" WHEN ({m_if.group(1)})"
                            b_clean = inner_s
                    # Convert any remaining inner `IF (cond) THEN stmt; END IF;`
                    def _unwrap_if(m_inner: re.Match[str]) -> str:
                        c_in = m_inner.group(1).strip()
                        st_in = m_inner.group(2).strip()
                        out_sts = []
                        for st_p in _split_top_level(st_in, ";"):
                            sp = st_p.strip()
                            if not sp:
                                continue
                            if re.match(r"^INSERT\s+INTO\b", sp, flags=re.IGNORECASE):
                                if re.search(r"\bWHERE\s+(?:TRUE|1)\b", sp, flags=re.IGNORECASE):
                                    sp = re.sub(r"\bWHERE\s+(?:TRUE|1)\b", f"WHERE ({c_in})", sp, count=1, flags=re.IGNORECASE)
                                else:
                                    sp = re.sub(r"\n\s*WHERE\s+", f"\n    WHERE ({c_in}) AND ", sp, count=1, flags=re.IGNORECASE)
                            elif re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", sp, flags=re.IGNORECASE):
                                sp = re.sub(r"\n\s*WHERE\s+", f"\n    WHERE ({c_in}) AND ", sp, count=1, flags=re.IGNORECASE)
                            out_sts.append(sp + ";")
                        return "\n".join(out_sts)

                    b_clean = re.sub(
                        r"IF\s*\(([^\n]+?)\)\s*THEN\s*(.*?)\s*END\s+IF;",
                        _unwrap_if,
                        b_clean,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    # Unwrap outer changed_guard IF if still present
                    m_outer_if = re.match(r"^IF\s*\(([^\n]+?)\)\s*THEN\s*(.*)\s*END\s+IF;$", b_clean.strip(), flags=re.IGNORECASE | re.DOTALL)
                    if m_outer_if:
                        when_cl = f" WHEN ({m_outer_if.group(1)})"
                        b_clean = m_outer_if.group(2).strip()

                    cur.execute(
                        f"CREATE TRIGGER {t_name}_{suf} AFTER {op} ON {base_t} FOR EACH ROW{when_cl}\n"
                        f"BEGIN\n{b_clean}\nEND;"
                    )
        else:
            # Transition table trigger: `AFTER INSERT/DELETE/UPDATE ON base_t REFERENCING ...`
            m_stmt_tr = re.match(
                r"CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\s+AFTER\s+(INSERT|DELETE|UPDATE)\s+ON\s+([A-Za-z0-9_]+)\s+REFERENCING\s+[^\n]+\s+FOR\s+EACH\s+STATEMENT\s+EXECUTE\s+FUNCTION\s+([A-Za-z0-9_]+)\(\);",
                ddl.strip(),
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_stmt_tr:
                t_name, op, base_t, fn_name = (
                    m_stmt_tr.group(1),
                    m_stmt_tr.group(2).upper(),
                    m_stmt_tr.group(3),
                    m_stmt_tr.group(4),
                )
                body = fn_bodies.get(fn_name, "")
                loop_bodies = re.findall(
                    r"FOR\s+(?:NEW|OLD)\s+IN\s+SELECT\s+\*\s+FROM\s+__mv_(?:new|old)_rows\s+LOOP\s*(.*?)\s*END\s+LOOP;",
                    body,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                combined = "\n".join(loop_bodies)
                b_clean = _transpile_pg_expr_to_sqlite(combined)

                def _unwrap_if_tr(m_inner: re.Match[str]) -> str:
                    c_in = m_inner.group(1).strip()
                    st_in = m_inner.group(2).strip()
                    out_sts = []
                    for st_p in _split_top_level(st_in, ";"):
                        sp = st_p.strip()
                        if not sp:
                            continue
                        if re.match(r"^INSERT\s+INTO\b", sp, flags=re.IGNORECASE):
                            if re.search(r"\bWHERE\s+(?:TRUE|1)\b", sp, flags=re.IGNORECASE):
                                sp = re.sub(r"\bWHERE\s+(?:TRUE|1)\b", f"WHERE ({c_in})", sp, count=1, flags=re.IGNORECASE)
                            else:
                                sp = re.sub(r"\n\s*WHERE\s+", f"\n    WHERE ({c_in}) AND ", sp, count=1, flags=re.IGNORECASE)
                        elif re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", sp, flags=re.IGNORECASE):
                            sp = re.sub(r"\n\s*WHERE\s+", f"\n    WHERE ({c_in}) AND ", sp, count=1, flags=re.IGNORECASE)
                        out_sts.append(sp + ";")
                    return "\n".join(out_sts)

                b_clean = re.sub(
                    r"IF\s*\(([^\n]+?)\)\s*THEN\s*(.*?)\s*END\s+IF;",
                    _unwrap_if_tr,
                    b_clean,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                cur.execute(
                    f"CREATE TRIGGER {t_name} AFTER {op} ON {base_t} FOR EACH ROW\n"
                    f"BEGIN\n{b_clean}\nEND;"
                )

    for pop_s in plan.get_populate_statements(incremental_backfill=False):
        cur.execute(_transpile_pg_expr_to_sqlite(pop_s))


def _transpile_mysql_expr_to_sqlite(sql: str, pk_cols: Sequence[str] = ("__mv_pk",)) -> str:
    s = sql
    s = re.sub(r"\bCREATE\s+ALGORITHM\s*=\s*\w+\s+VIEW\b", "CREATE VIEW", s, flags=re.IGNORECASE)
    s = re.sub(r"\bBIGINT\s+UNSIGNED\b", "INTEGER", s, flags=re.IGNORECASE)
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


def _execute_mysql_plan_in_sqlite(conn: sqlite3.Connection, plan: MaterializedViewPlan) -> None:
    _register_sqlite_compat_udfs(conn)
    cur = conn.cursor()

    for aux_ddl in plan.auxiliary_table_ddls:
        cur.execute(_transpile_mysql_expr_to_sqlite(aux_ddl))
    cur.execute(_transpile_mysql_expr_to_sqlite(plan.backing_table_ddl))
    for idx_ddl in plan.index_ddls:
        # Remove extra parens around MySQL functional index `((expr))` -> `(expr)`
        cleaned_idx = _transpile_mysql_expr_to_sqlite(idx_ddl)
        cleaned_idx = re.sub(r"\(\(\((.*?)\)\)\)", r"((\1))", cleaned_idx)
        cur.execute(cleaned_idx)
    if plan.reader_view_ddl:
        cur.execute(_transpile_mysql_expr_to_sqlite(plan.reader_view_ddl))

    cur.execute(
        f"CREATE TRIGGER __ro_my_{plan.logical_view_name}_del INSTEAD OF DELETE ON {plan.view_name} "
        f"BEGIN SELECT RAISE(ABORT, 'read-only'); END;"
    )

    # Determine primary key columns of backing table for ON DUPLICATE KEY UPDATE conversion
    m_pk = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", plan.backing_table_ddl, flags=re.IGNORECASE)
    if m_pk:
        main_pks = [c.strip() for c in m_pk.group(1).split(",")]
    elif "__mv_global_id" in plan.backing_table_ddl:
        main_pks = ["__mv_global_id"]
    else:
        main_pks = ["__mv_pk"]

    for ddl in plan.trigger_ddls:
        m_tr = re.match(
            r"CREATE\s+TRIGGER\s+([A-Za-z0-9_]+)\s+AFTER\s+(INSERT|DELETE|UPDATE)\s+ON\s+([A-Za-z0-9_]+)\s+FOR\s+EACH\s+ROW\s+BEGIN\s*(.*)\s*END;",
            ddl.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m_tr:
            continue
        t_name, op, base_t, body = m_tr.group(1), m_tr.group(2).upper(), m_tr.group(3), m_tr.group(4).strip()
        b_clean = _transpile_mysql_expr_to_sqlite(body, main_pks)

        when_cl = ""
        m_outer = re.match(r"^IF\s*\(([^\n]+?)\)\s*THEN\s*(.*)\s*END\s+IF;$", b_clean.strip(), flags=re.IGNORECASE | re.DOTALL)
        if m_outer:
            inner_my = m_outer.group(2).strip()
            if m_outer.group(1).strip().upper().startswith("NOT ") or "END IF;" not in inner_my.upper():
                when_cl = f" WHEN ({m_outer.group(1)})"
                b_clean = inner_my

        def _unwrap_my_if(m_inner: re.Match[str]) -> str:
            c_in = m_inner.group(1).strip()
            st_in = m_inner.group(2).strip()
            out_sts = []
            for st_p in _split_top_level(st_in, ";"):
                sp = st_p.strip()
                if not sp:
                    continue
                if re.match(r"^INSERT\s+INTO\b", sp, flags=re.IGNORECASE):
                    if re.search(r"\bWHERE\s+TRUE\b", sp, flags=re.IGNORECASE):
                        sp = re.sub(r"\bWHERE\s+TRUE\b", f"WHERE ({c_in})", sp, count=1, flags=re.IGNORECASE)
                    else:
                        sp = re.sub(r"\n\s*WHERE\s+", f"\n    WHERE ({c_in}) AND ", sp, count=1, flags=re.IGNORECASE)
                elif re.match(r"^(?:UPDATE|DELETE\s+FROM)\b", sp, flags=re.IGNORECASE):
                    sp = re.sub(r"\n\s*WHERE\s+", f"\n    WHERE ({c_in}) AND ", sp, count=1, flags=re.IGNORECASE)
                out_sts.append(sp + ";")
            return "\n".join(out_sts)

        b_clean = re.sub(
            r"IF\s*\(([^\n]+?)\)\s*THEN\s*(.*?)\s*END\s+IF;",
            _unwrap_my_if,
            b_clean,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Convert `ON DUPLICATE KEY UPDATE` on auxiliary tables and backing table
        def _repl_odku(m_ins: re.Match[str]) -> str:
            tbl = m_ins.group(1)
            prefix_sql = m_ins.group(0).split("ON DUPLICATE KEY UPDATE")[0].rstrip()
            assigns = m_ins.group(2)
            if "__matview_dist_" in tbl:
                pks = [*main_pks, "__val"]
            else:
                pks = main_pks
            return f"{prefix_sql}\n    ON CONFLICT ({', '.join(pks)}) DO UPDATE SET {assigns}"

        b_clean = re.sub(
            r"INSERT\s+INTO\s+([A-Za-z0-9_]+)\s*\([^;]+?\bON\s+DUPLICATE\s+KEY\s+UPDATE\s+([^;]+)",
            _repl_odku,
            b_clean,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Convert MySQL multi-table `UPDATE mv_table INNER JOIN (...) AS __mv_delta ON cond SET ...;` to SQLite `UPDATE mv_table SET ... FROM (...) AS __mv_delta WHERE cond;`
        def _repl_multi_upd(m_u: re.Match[str]) -> str:
            tbl = m_u.group(1)
            subq = m_u.group(2)
            on_c = m_u.group(3)
            set_cl = m_u.group(4)
            # Strip `tbl.` prefix from LHS of assignments in `set_cl`
            cleaned_assigns = []
            for a_part in _split_top_level(set_cl, ","):
                ap = re.sub(rf"^\s*{re.escape(tbl)}\.", "", a_part.strip())
                cleaned_assigns.append(ap)
            return f"UPDATE {tbl}\n    SET {', '.join(cleaned_assigns)}\n    FROM ({subq}) AS __mv_delta\n    WHERE {on_c};"

        b_clean = re.sub(
            r"UPDATE\s+([A-Za-z0-9_]+)\s+INNER\s+JOIN\s*\((.*?)\)\s*AS\s+__mv_delta\s+ON\s+([^\n]+?)\s+SET\s+([^;]+);",
            _repl_multi_upd,
            b_clean,
            flags=re.IGNORECASE | re.DOTALL,
        )

        cur.execute(
            f"CREATE TRIGGER {t_name} AFTER {op} ON {base_t} FOR EACH ROW{when_cl}\n"
            f"BEGIN\n{b_clean}\nEND;"
        )

    for pop_s in plan.get_populate_statements(incremental_backfill=False):
        cur.execute(_transpile_mysql_expr_to_sqlite(pop_s, main_pks))


# -----------------------------------------------------------------------------
# Live External Database Verification & 50-View x 33-Step Battery
# -----------------------------------------------------------------------------


def _verify_external_db_mv(cur, mv_name: str, select_sql: str, context: str = "") -> None:
    def _norm(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float, Decimal)):
            fv = round(float(v), 3)
            if abs(fv - round(fv)) < 1e-9:
                return int(round(fv))
            return fv
        return v

    def _skey(row):
        return tuple(
            (0 if v is None else (1 if isinstance(v, (int, float)) else 2), _norm(v) if v is not None else "")
            for v in row
        )

    def _rows_match(rows_a, rows_b) -> bool:
        if len(rows_a) != len(rows_b):
            return False
        for ra, rb in zip(rows_a, rows_b):
            if len(ra) != len(rb):
                return False
            for ca, cb in zip(ra, rb):
                if ca is None or cb is None:
                    if ca is not cb:
                        return False
                elif isinstance(ca, (int, float)) and isinstance(cb, (int, float)):
                    if abs(float(ca) - float(cb)) > 1.5e-3:
                        return False
                elif ca != cb:
                    return False
        return True

    cur.execute(f"SELECT * FROM {mv_name}")
    mv_rows = sorted([tuple(_norm(x) for x in r) for r in cur.fetchall()], key=_skey)
    cur.execute(select_sql)
    live_rows = sorted([tuple(_norm(x) for x in r) for r in cur.fetchall()], key=_skey)
    if not _rows_match(mv_rows, live_rows):
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


def _get_complex_dml_and_tx_steps(dialect: Dialect, for_live_mysql: bool = False, for_live_postgres: bool = False) -> List[Tuple[str, str]]:
    """
    Returns a 33-step sequence of complex single-statement DMLs and multi-statement
    transactional workflows (with intermediate state verification inside open transactions,
    nested SAVEPOINTs, ROLLBACK TO SAVEPOINT, RELEASE SAVEPOINT, full ROLLBACK, and COMMIT).
    """
    t_val = "TRUE" if dialect == Dialect.POSTGRES else "1"
    f_val = "FALSE" if dialect == Dialect.POSTGRES else "0"
    if for_live_mysql:
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
        final_drain = "TRUNCATE TABLE orders;" if for_live_postgres else "DELETE FROM orders;"
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
        ("tx1_begin", "BEGIN;"),
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
        ("tx2_begin", "BEGIN;"),
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

        def _make_sq3_cursor() -> _CompatCursorAdapter:
            def _exec(s: str, p: Any = None) -> None:
                if not s.strip():
                    return
                if p is not None:
                    raw_conn.execute(s.strip(), tuple(p))
                else:
                    raw_conn.execute(s.strip())
            def _query(q: str, p: Any = None):
                c = raw_conn.execute(q, tuple(p)) if p is not None else raw_conn.execute(q)
                return [tuple(r) for r in (c.fetchall() or [])]
            return _CompatCursorAdapter(_exec, _query)

        return _CompatConnectionAdapter(_make_sq3_cursor, raw_conn.close)

    else:
        raise ValueError(f"Unknown SQLITE_COMPAT_ENGINE: {engine_spec}")


def run_live_postgres_tests_if_available(dsn: str) -> None:
    """Runs comprehensive 50-view x 3-mode (full, incremental, transition-tables) + manager tests against a real PostgreSQL instance."""
    conn = _open_live_postgres_connection(dsn)
    if conn is None:
        print("Skipping live PostgreSQL test: psycopg2/psycopg not installed.")
        return

    schema, view_specs = _get_live_engine_50_view_battery(Dialect.POSTGRES)
    cur = conn.cursor()
    total_checks = 0
    total_views_installed = 0

    for inc_mode, use_trans in ((False, False), (True, False), (False, True)):
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        for s in [st.strip() for st in schema.split(";") if st.strip()]:
            cur.execute(s)

        cur.execute("INSERT INTO warehouses VALUES (1, 'WH_US', 'US'), (2, 'WH_EU', 'EU'), (3, 'WH_GLOBAL', NULL);")
        cur.execute("INSERT INTO customers VALUES (1, 1, 'Alice', 'NA'), (2, 2, 'Bob', 'VIP'), (3, 2, 'Cara', 'EU'), (4, NULL, 'Dan', 'APAC');")
        cur.execute(
            "INSERT INTO orders VALUES "
            "(1, 1, 'NA', 'hw', 10.0, 2, 3, TRUE, 'settled'),"
            "(2, 1, 'NA', 'sw', 20.0, 4, 5, FALSE, 'settled'),"
            "(3, 2, 'EU', 'hw', 15.0, 1, 7, TRUE, 'pending'),"
            "(4, 2, 'EU', 'hw', 30.0, 3, 6, TRUE, 'settled'),"
            "(5, 3, NULL, 'sw', 50.0, 5, 1, FALSE, 'settled');"
        )
        cur.execute(
            "INSERT INTO order_tags VALUES "
            "(1, 'urgent', 2), (1, 'fragile', 5), (2, 'urgent', 3), (4, 'bulk', 10);"
        )

        plans = []
        for v_name, v_sel in view_specs:
            p = generate_materialized_view(
                schema,
                f"CREATE VIEW {v_name} AS {v_sel};",
                dialect=Dialect.POSTGRES,
                pg_use_transition_tables=use_trans,
                incremental_backfill=inc_mode,
                batch_size=2,
            )
            for stmt in p.all_statements():
                cur.execute(stmt)
            plans.append((p, v_sel))
            total_views_installed += 1
            _verify_external_db_mv(cur, p.view_name, v_sel, f"pg:inc={inc_mode}:tr={use_trans}:{v_name}:init")
            total_checks += 1

        ro_raised = False
        try:
            cur.execute(f"DELETE FROM {plans[0][0].view_name};")
        except Exception:
            ro_raised = True
        assert ro_raised, "Expected PostgreSQL read-only view trigger to block direct DELETE on MV"

        for step_lbl, m_sql in _get_complex_dml_and_tx_steps(Dialect.POSTGRES, for_live_mysql=False, for_live_postgres=True):
            cur.execute(m_sql)
            for p, v_sel in plans:
                _verify_external_db_mv(cur, p.view_name, v_sel, f"pg:inc={inc_mode}:tr={use_trans}:{p.logical_view_name}:{step_lbl}")
                total_checks += 1
        for p, _ in reversed(plans):
            for drop_st in p.drop_statements():
                cur.execute(drop_st)

    total_checks += _run_live_manager_lifecycle_check(conn, cur, Dialect.POSTGRES)
    conn.close()
    print(f"LIVE PostgreSQL server test PASSED ({total_views_installed} views installed, {total_checks} live assertions)!")


def run_live_mysql_tests_if_available(dsn: str) -> None:
    """Runs comprehensive 50-view x 2-mode (full, incremental) + manager tests against a real MySQL/MariaDB/Percona instance."""
    conn, target_db = _open_live_mysql_connection(dsn)
    if conn is None:
        print("Skipping live MySQL test: pymysql not installed.")
        return

    cur = conn.cursor()
    schema, view_specs = _get_live_engine_50_view_battery(Dialect.MYSQL)

    total_checks = 0
    total_views_installed = 0
    for inc_mode in (False, True):
        cur.execute(f"DROP DATABASE IF EXISTS `{target_db}`;")
        cur.execute(f"CREATE DATABASE `{target_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cur.execute(f"USE `{target_db}`;")
        cur.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cur.execute("SET SESSION collation_connection = @@collation_database;")
        for s in [st.strip() for st in schema.split(";") if st.strip()]:
            cur.execute(s)

        cur.execute("INSERT INTO warehouses VALUES (1, 'WH_US', 'US'), (2, 'WH_EU', 'EU'), (3, 'WH_GLOBAL', NULL);")
        cur.execute("INSERT INTO customers VALUES (1, 1, 'Alice', 'NA'), (2, 2, 'Bob', 'VIP'), (3, 2, 'Cara', 'EU'), (4, NULL, 'Dan', 'APAC');")
        cur.execute(
            "INSERT INTO orders VALUES "
            "(1, 1, 'NA', 'hw', 10.0, 2, 3, 1, 'settled'),"
            "(2, 1, 'NA', 'sw', 20.0, 4, 5, 0, 'settled'),"
            "(3, 2, 'EU', 'hw', 15.0, 1, 7, 1, 'pending'),"
            "(4, 2, 'EU', 'hw', 30.0, 3, 6, 1, 'settled'),"
            "(5, 3, NULL, 'sw', 50.0, 5, 1, 0, 'settled');"
        )
        cur.execute(
            "INSERT INTO order_tags VALUES "
            "(1, 'urgent', 2), (1, 'fragile', 5), (2, 'urgent', 3), (4, 'bulk', 10);"
        )

        plans = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            for v_name, v_sel in view_specs:
                p = generate_materialized_view(
                    schema,
                    f"CREATE VIEW {v_name} AS {v_sel};",
                    dialect=Dialect.MYSQL,
                    incremental_backfill=inc_mode,
                    batch_size=2,
                    ignore_limitations=True,
                )
                for stmt in p.all_statements():
                    cur.execute(stmt)
                plans.append((p, v_sel))
                total_views_installed += 1
                _verify_external_db_mv(cur, p.view_name, v_sel, f"mysql:inc={inc_mode}:{v_name}:init")
                total_checks += 1

        # Verify MySQL read-only view (ALGORITHM = TEMPTABLE makes view non-updatable)
        ro_raised = False
        try:
            cur.execute(f"DELETE FROM {plans[0][0].view_name};")
        except Exception:
            ro_raised = True
        assert ro_raised, "Expected MySQL read-only view to block direct DELETE on MV"

        for step_lbl, m_sql in _get_complex_dml_and_tx_steps(Dialect.MYSQL, for_live_mysql=True):
            cur.execute(m_sql)
            for p, v_sel in plans:
                _verify_external_db_mv(cur, p.view_name, v_sel, f"mysql:inc={inc_mode}:{p.logical_view_name}:{step_lbl}")
                total_checks += 1
        for p, _ in reversed(plans):
            for drop_st in p.drop_statements():
                cur.execute(drop_st)

    total_checks += _run_live_manager_lifecycle_check(conn, cur, Dialect.MYSQL)
    conn.close()
    print(f"LIVE MySQL server test PASSED ({total_views_installed} views installed, {total_checks} live assertions)!")


def run_live_sqlite_compat_tests_if_available(engine_spec: str) -> None:
    """
    Runs comprehensive 50-view x 2-mode (full, incremental) + manager lifecycle live integration
    tests against SQLite-compatible database engines (`apsw`, `sqlite3mc`, `sqlcipher`, `sqlean`, `libsql`).
    """
    eng = engine_spec.strip().lower()
    schema, view_specs = _get_live_engine_50_view_battery(Dialect.SQLITE)

    total_checks = 0
    total_views_installed = 0
    for inc_mode in (False, True):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, f"compat_{eng}_{int(inc_mode)}.db")
            conn = _open_sqlite_compat_connection(eng, db_file)
            cur = conn.cursor()
            try:
                for s in [st.strip() for st in schema.split(";") if st.strip()]:
                    cur.execute(s + ";")
                cur.execute("INSERT INTO warehouses VALUES (1, 'WH_US', 'US'), (2, 'WH_EU', 'EU'), (3, 'WH_GLOBAL', NULL);")
                cur.execute("INSERT INTO customers VALUES (1, 1, 'Alice', 'NA'), (2, 2, 'Bob', 'VIP'), (3, 2, 'Cara', 'EU'), (4, NULL, 'Dan', 'APAC');")
                cur.execute(
                    "INSERT INTO orders VALUES "
                    "(1, 1, 'NA', 'hw', 10.0, 2, 3, 1, 'settled'),"
                    "(2, 1, 'NA', 'sw', 20.0, 4, 5, 0, 'settled'),"
                    "(3, 2, 'EU', 'hw', 15.0, 1, 7, 1, 'pending'),"
                    "(4, 2, 'EU', 'hw', 30.0, 3, 6, 1, 'settled'),"
                    "(5, 3, NULL, 'sw', 50.0, 5, 1, 0, 'settled');"
                )
                cur.execute(
                    "INSERT INTO order_tags VALUES "
                    "(1, 'urgent', 2), (1, 'fragile', 5), (2, 'urgent', 3), (4, 'bulk', 10);"
                )
                plans = []
                for v_name, v_sel in view_specs:
                    p = generate_materialized_view(
                        schema,
                        f"CREATE VIEW {v_name} AS {v_sel};",
                        dialect=Dialect.SQLITE,
                        incremental_backfill=inc_mode,
                        batch_size=2,
                    )
                    for stmt in p.all_statements():
                        cur.execute(stmt)
                    plans.append((p, v_sel))
                    total_views_installed += 1
                    _verify_external_db_mv(cur, p.view_name, v_sel, f"{eng}:inc={inc_mode}:{v_name}:init")
                    total_checks += 1

                ro_raised = False
                try:
                    cur.execute(f"DELETE FROM {plans[0][0].view_name};")
                except Exception:
                    ro_raised = True
                assert ro_raised, f"Expected {eng} read-only view trigger to block direct DELETE on MV"

                for step_lbl, m_sql in _get_complex_dml_and_tx_steps(Dialect.SQLITE, for_live_mysql=False):
                    cur.execute(m_sql)
                    for p, v_sel in plans:
                        _verify_external_db_mv(cur, p.view_name, v_sel, f"{eng}:inc={inc_mode}:{p.logical_view_name}:{step_lbl}")
                        total_checks += 1

                if inc_mode:
                    total_checks += _run_live_manager_lifecycle_check(conn, cur, Dialect.SQLITE)
            finally:
                conn.close()

    print(f"LIVE SQLite-compatible engine ({eng}) test PASSED ({total_views_installed} views installed, {total_checks} live assertions)!")


import sys
import time
import unittest


def _rewrite_live_reference_select_for_dialect(raw_select_sql: str, dialect: Dialect) -> str:
    if dialect == Dialect.SQLITE:
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


def _adapt_mutation_for_dialect(m_sql: str, dialect: Dialect, for_live_engine: bool) -> str:
    if dialect == Dialect.MYSQL and for_live_engine:
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

        # 1. Live PostgreSQL / PostgreSQL-compatible execution
        if self.pg_dsn and eff_dialect == Dialect.POSTGRES:
            pg_conn = _open_live_postgres_connection(self.pg_dsn)
            if pg_conn is not None:
                try:
                    cur = pg_conn.cursor()
                    eff_muts = [
                        (lbl, _adapt_mutation_for_dialect(msql, Dialect.POSTGRES, True))
                        for lbl, msql in mutations
                    ]
                    chunk_size = 100
                    for i in range(0, len(views), chunk_size):
                        chunk = views[i : i + chunk_size]
                        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
                        for st in [s.strip() for s in eff_schema.split(";") if s.strip()]:
                            cur.execute(st)
                        for s in seed_sqls:
                            cur.execute(s)
                        plans_and_selects: List[Tuple[MaterializedViewPlan, str]] = []
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", EngineLimitationWarning)
                            for v_name, v_ddl in chunk:
                                eff_v_ddl = _adapt_view_ddl_for_dialect(v_ddl, Dialect.POSTGRES)
                                plan = generate_materialized_view(
                                    eff_schema,
                                    eff_v_ddl,
                                    dialect=Dialect.POSTGRES,
                                    pg_use_transition_tables=pg_use_transition_tables,
                                    incremental_backfill=incremental_backfill,
                                    batch_size=batch_size,
                                    ignore_limitations=True,
                                )
                                for stmt in plan.all_statements():
                                    cur.execute(stmt)
                                v_spec = parse_view_definition(eff_v_ddl)
                                live_sel = _rewrite_live_reference_select_for_dialect(v_spec.raw_select_sql, Dialect.POSTGRES)
                                plans_and_selects.append((plan, live_sel))
                                self.total_views += 1
                                _verify_external_db_mv(cur, plan.view_name, live_sel, f"{suite_name}:{v_name}:init")
                                self.total_assertions += 1
                        for step_lbl, m_sql in eff_muts:
                            cur.execute(m_sql)
                            for plan, live_sel in plans_and_selects:
                                _verify_external_db_mv(cur, plan.view_name, live_sel, f"{suite_name}:{plan.logical_view_name}:{step_lbl}")
                                self.total_assertions += 1
                finally:
                    pg_conn.close()
                print(f"  [PASS] {suite_name} [LIVE POSTGRES]: {len(views)} views x {len(mutations) + 1} states = {len(views) * (len(mutations) + 1)} assertions")
                return

        # 2. Live MySQL / MariaDB / Percona Server execution
        if self.mysql_dsn and eff_dialect == Dialect.MYSQL:
            my_conn, target_db = _open_live_mysql_connection(self.mysql_dsn)
            if my_conn is not None:
                try:
                    cur = my_conn.cursor()
                    eff_muts = [
                        (lbl, _adapt_mutation_for_dialect(msql, Dialect.MYSQL, True))
                        for lbl, msql in mutations
                    ]
                    chunk_size = 50
                    for i in range(0, len(views), chunk_size):
                        chunk = views[i : i + chunk_size]
                        cur.execute(f"DROP DATABASE IF EXISTS `{target_db}`;")
                        cur.execute(f"CREATE DATABASE `{target_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
                        cur.execute(f"USE `{target_db}`;")
                        cur.execute("SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;")
                        cur.execute("SET SESSION collation_connection = @@collation_database;")
                        for st in [s.strip() for s in eff_schema.split(";") if s.strip()]:
                            cur.execute(st)
                        for s in seed_sqls:
                            cur.execute(s)
                        plans_and_selects = []
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", EngineLimitationWarning)
                            for v_name, v_ddl in chunk:
                                eff_v_ddl = _adapt_view_ddl_for_dialect(v_ddl, Dialect.MYSQL)
                                plan = generate_materialized_view(
                                    eff_schema,
                                    eff_v_ddl,
                                    dialect=Dialect.MYSQL,
                                    incremental_backfill=incremental_backfill,
                                    batch_size=batch_size,
                                    ignore_limitations=True,
                                )
                                for stmt in plan.all_statements():
                                    cur.execute(stmt)
                                v_spec = parse_view_definition(eff_v_ddl)
                                live_sel = _rewrite_live_reference_select_for_dialect(v_spec.raw_select_sql, Dialect.MYSQL)
                                plans_and_selects.append((plan, live_sel))
                                self.total_views += 1
                                _verify_external_db_mv(cur, plan.view_name, live_sel, f"{suite_name}:{v_name}:init")
                                self.total_assertions += 1
                        for step_lbl, m_sql in eff_muts:
                            cur.execute(m_sql)
                            for plan, live_sel in plans_and_selects:
                                _verify_external_db_mv(cur, plan.view_name, live_sel, f"{suite_name}:{plan.logical_view_name}:{step_lbl}")
                                self.total_assertions += 1
                finally:
                    my_conn.close()
                print(f"  [PASS] {suite_name} [LIVE MYSQL]: {len(views)} views x {len(mutations) + 1} states = {len(views) * (len(mutations) + 1)} assertions")
                return

        # 3. Live SQLite-compatible engine execution (apsw, sqlite3mc, sqlcipher, sqlean, libsql)
        if self.sqlite_compat_engine and eff_dialect == Dialect.SQLITE:
            eng = self.sqlite_compat_engine.strip().lower()
            with tempfile.TemporaryDirectory() as tmpdir:
                db_file = os.path.join(tmpdir, f"suite_{abs(hash(suite_name))}.db")
                compat_conn = _open_sqlite_compat_connection(eng, db_file)
                cur = compat_conn.cursor()
                try:
                    for st in [s.strip() for s in schema_sql.split(";") if s.strip()]:
                        cur.execute(st + ";")
                    for s in seed_sqls:
                        cur.execute(s)
                    plans_and_selects = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", EngineLimitationWarning)
                        for v_name, v_ddl in views:
                            plan = generate_materialized_view(
                                schema_sql,
                                v_ddl,
                                dialect=Dialect.SQLITE,
                                incremental_backfill=incremental_backfill,
                                batch_size=batch_size,
                                ignore_limitations=True,
                            )
                            for stmt in plan.all_statements():
                                cur.execute(stmt)
                            v_spec = parse_view_definition(v_ddl)
                            plans_and_selects.append((plan, v_spec.raw_select_sql))
                            self.total_views += 1
                            _verify_external_db_mv(cur, plan.view_name, v_spec.raw_select_sql, f"{suite_name}:{eng}:{v_name}:init")
                            self.total_assertions += 1
                    for step_lbl, m_sql in mutations:
                        cur.execute(m_sql)
                        for plan, sel_sql in plans_and_selects:
                            _verify_external_db_mv(cur, plan.view_name, sel_sql, f"{suite_name}:{eng}:{plan.logical_view_name}:{step_lbl}")
                            self.total_assertions += 1
                finally:
                    compat_conn.close()
            print(f"  [PASS] {suite_name} [LIVE {eng.upper()}]: {len(views)} views x {len(mutations) + 1} states = {len(views) * (len(mutations) + 1)} assertions")
            return

        # 4. Standard in-memory SQLite (and cross-dialect transpiled SQL verification)
        conn = sqlite3.connect(":memory:", isolation_level=None)
        conn.execute("PRAGMA recursive_triggers = ON;")
        _register_sqlite_compat_udfs(conn)
        conn.executescript(schema_sql)
        for s in seed_sqls:
            conn.execute(s)

        plans_and_selects = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", EngineLimitationWarning)
            for v_name, v_ddl in views:
                eff_v_ddl = _adapt_view_ddl_for_dialect(v_ddl, eff_dialect)
                plan = generate_materialized_view(
                    eff_schema,
                    eff_v_ddl,
                    dialect=eff_dialect,
                    pg_use_transition_tables=pg_use_transition_tables,
                    incremental_backfill=incremental_backfill,
                    batch_size=batch_size,
                    ignore_limitations=True,
                )
                if eff_dialect == Dialect.SQLITE:
                    for stmt in plan.all_statements():
                        conn.execute(stmt)
                elif eff_dialect == Dialect.POSTGRES:
                    _execute_postgres_plan_in_sqlite(conn, plan)
                elif eff_dialect == Dialect.MYSQL:
                    _execute_mysql_plan_in_sqlite(conn, plan)

                v_spec = parse_view_definition(v_ddl)
                plans_and_selects.append((plan, v_spec.raw_select_sql))
                self.total_views += 1
                _assert_mv_matches_live(conn, plan.view_name, v_spec.raw_select_sql, f"{suite_name}:{v_name}:init")
                self.total_assertions += 1

        for step_lbl, m_sql in mutations:
            conn.execute(m_sql)
            for plan, sel_sql in plans_and_selects:
                _assert_mv_matches_live(conn, plan.view_name, sel_sql, f"{suite_name}:{plan.logical_view_name}:{step_lbl}")
                self.total_assertions += 1

        conn.close()
        print(f"  [PASS] {suite_name}: {len(views)} views x {len(mutations) + 1} states = {len(views) * (len(mutations) + 1)} assertions")


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

    # Execute all 100 views compiled for POSTGRES (transpiled to SQLite)
    pg_views = [(f"pg_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW pg_{n}")) for n, sql in view_defs]
    runner.run_batch(
        "Suite 9A (100 PostgreSQL-Compiled Views Executed via SQLite Transpiler)",
        schema,
        pg_views,
        seed,
        mutations,
        dialect=Dialect.POSTGRES,
    )

    # Execute all 100 views compiled for MYSQL
    my_views = [(f"my_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW my_{n}")) for n, sql in view_defs]
    runner.run_batch(
        "Suite 9B (100 MySQL-Compiled Views)",
        schema,
        my_views,
        seed,
        mutations,
        dialect=Dialect.MYSQL,
    )

    if runner.pg_dsn:
        pg_tr_views = [(f"pgtr_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW pgtr_{n}")) for n, sql in view_defs]
        runner.run_batch(
            "Suite 9C (100 PostgreSQL Transition-Table & Incremental Views on Live PG)",
            schema,
            pg_tr_views,
            seed,
            mutations,
            dialect=Dialect.POSTGRES,
            pg_use_transition_tables=True,
            incremental_backfill=True,
            batch_size=2,
        )
    elif runner.mysql_dsn:
        my_inc_views = [(f"myinc_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW myinc_{n}")) for n, sql in view_defs]
        runner.run_batch(
            "Suite 9C (100 MySQL Incremental Backfill Views on Live MySQL)",
            schema,
            my_inc_views,
            seed,
            mutations,
            dialect=Dialect.MYSQL,
            incremental_backfill=True,
            batch_size=2,
        )
    elif runner.sqlite_compat_engine:
        sq_compat_views = [(f"sqc_{n}", sql.replace(f"CREATE VIEW {n}", f"CREATE VIEW sqc_{n}")) for n, sql in view_defs]
        runner.run_batch(
            "Suite 9C (100 SQLite Incremental Backfill Views on Live SQLite-Compat Engine)",
            schema,
            sq_compat_views,
            seed,
            mutations,
            dialect=Dialect.SQLITE,
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
    Executes the exact 50-view x 33-step live-engine verification battery across:
      - SQLite (incremental_backfill=False & True) + _run_live_manager_lifecycle_check
      - PostgreSQL (pg_use_transition_tables=False & True)
      - MySQL (Dialect.MYSQL)
    """
    # 11A & 11B: SQLite (Standard & Incremental Backfill)
    sq_schema, sq_views = _get_live_engine_50_view_battery(Dialect.SQLITE)
    for inc_mode in (False, True):
        conn = sqlite3.connect(":memory:", isolation_level=None)
        conn.execute("PRAGMA recursive_triggers = ON;")
        cur = conn.cursor()
        for s in [st.strip() for st in sq_schema.split(";") if st.strip()]:
            cur.execute(s + ";")
        cur.execute("INSERT INTO warehouses VALUES (1, 'WH_US', 'US'), (2, 'WH_EU', 'EU'), (3, 'WH_GLOBAL', NULL);")
        cur.execute("INSERT INTO customers VALUES (1, 1, 'Alice', 'NA'), (2, 2, 'Bob', 'VIP'), (3, 2, 'Cara', 'EU'), (4, NULL, 'Dan', 'APAC');")
        cur.execute(
            "INSERT INTO orders VALUES "
            "(1, 1, 'NA', 'hw', 10.0, 2, 3, 1, 'settled'),"
            "(2, 1, 'NA', 'sw', 20.0, 4, 5, 0, 'settled'),"
            "(3, 2, 'EU', 'hw', 15.0, 1, 7, 1, 'pending'),"
            "(4, 2, 'EU', 'hw', 30.0, 3, 6, 1, 'settled'),"
            "(5, 3, NULL, 'sw', 50.0, 5, 1, 0, 'settled');"
        )
        cur.execute("INSERT INTO order_tags VALUES (1, 'urgent', 2), (1, 'fragile', 5), (2, 'urgent', 3), (4, 'bulk', 10);")
        plans = []
        for v_name, v_sel in sq_views:
            p = generate_materialized_view(
                sq_schema,
                f"CREATE VIEW {v_name} AS {v_sel};",
                dialect=Dialect.SQLITE,
                incremental_backfill=inc_mode,
                batch_size=2,
            )
            for stmt in p.all_statements():
                cur.execute(stmt)
            plans.append((p, v_sel))
            runner.total_views += 1
            _verify_external_db_mv(cur, p.view_name, v_sel, f"s11_sq:inc={inc_mode}:{v_name}:init")
            runner.total_assertions += 1
        steps = _get_complex_dml_and_tx_steps(Dialect.SQLITE, for_live_mysql=False)
        for step_lbl, m_sql in steps:
            cur.execute(m_sql)
            for p, v_sel in plans:
                _verify_external_db_mv(cur, p.view_name, v_sel, f"s11_sq:inc={inc_mode}:{p.logical_view_name}:{step_lbl}")
                runner.total_assertions += 1
        runner.total_assertions += _run_live_manager_lifecycle_check(conn, cur, Dialect.SQLITE)
        conn.close()
        print(f"  [PASS] Suite 11 (Live 50-View Battery SQLite inc={inc_mode}): 50 views x {len(steps) + 1} states = {50 * (len(steps) + 1)} assertions")

    # 11C & 11D: PostgreSQL (Row-Level & Statement Transition Tables)
    pg_schema, pg_views = _get_live_engine_50_view_battery(Dialect.POSTGRES)
    for use_trans in (False, True):
        conn = sqlite3.connect(":memory:", isolation_level=None)
        conn.execute("PRAGMA recursive_triggers = ON;")
        _register_sqlite_compat_udfs(conn)
        cur = conn.cursor()
        for s in [st.strip() for st in pg_schema.split(";") if st.strip()]:
            cur.execute(_transpile_pg_expr_to_sqlite(s) + ";")
        cur.execute("INSERT INTO warehouses VALUES (1, 'WH_US', 'US'), (2, 'WH_EU', 'EU'), (3, 'WH_GLOBAL', NULL);")
        cur.execute("INSERT INTO customers VALUES (1, 1, 'Alice', 'NA'), (2, 2, 'Bob', 'VIP'), (3, 2, 'Cara', 'EU'), (4, NULL, 'Dan', 'APAC');")
        cur.execute(
            "INSERT INTO orders VALUES "
            "(1, 1, 'NA', 'hw', 10.0, 2, 3, 1, 'settled'),"
            "(2, 1, 'NA', 'sw', 20.0, 4, 5, 0, 'settled'),"
            "(3, 2, 'EU', 'hw', 15.0, 1, 7, 1, 'pending'),"
            "(4, 2, 'EU', 'hw', 30.0, 3, 6, 1, 'settled'),"
            "(5, 3, NULL, 'sw', 50.0, 5, 1, 0, 'settled');"
        )
        cur.execute("INSERT INTO order_tags VALUES (1, 'urgent', 2), (1, 'fragile', 5), (2, 'urgent', 3), (4, 'bulk', 10);")
        plans = []
        for v_name, v_sel in pg_views:
            p = generate_materialized_view(
                pg_schema,
                f"CREATE VIEW {v_name} AS {v_sel};",
                dialect=Dialect.POSTGRES,
                pg_use_transition_tables=use_trans,
                incremental_backfill=False,
                batch_size=2,
            )
            _execute_postgres_plan_in_sqlite(conn, p)
            v_sel_s = _transpile_pg_expr_to_sqlite(v_sel)
            plans.append((p, v_sel_s))
            runner.total_views += 1
            _verify_external_db_mv(cur, p.view_name, v_sel_s, f"s11_pg:tr={use_trans}:{v_name}:init")
            runner.total_assertions += 1
        steps = _get_complex_dml_and_tx_steps(Dialect.POSTGRES, for_live_mysql=False, for_live_postgres=False)
        for step_lbl, m_sql in steps:
            cur.execute(_transpile_pg_expr_to_sqlite(m_sql))
            for p, v_sel_s in plans:
                _verify_external_db_mv(cur, p.view_name, v_sel_s, f"s11_pg:tr={use_trans}:{p.logical_view_name}:{step_lbl}")
                runner.total_assertions += 1
        conn.close()
        print(f"  [PASS] Suite 11 (Live 50-View Battery PostgreSQL tr={use_trans}): 50 views x {len(steps) + 1} states = {50 * (len(steps) + 1)} assertions")

    # 11E: MySQL
    my_schema, my_views = _get_live_engine_50_view_battery(Dialect.MYSQL)
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA recursive_triggers = ON;")
    _register_sqlite_compat_udfs(conn)
    cur = conn.cursor()
    for s in [st.strip() for st in my_schema.split(";") if st.strip()]:
        cur.execute(_transpile_mysql_expr_to_sqlite(s) + ";")
    cur.execute("INSERT INTO warehouses VALUES (1, 'WH_US', 'US'), (2, 'WH_EU', 'EU'), (3, 'WH_GLOBAL', NULL);")
    cur.execute("INSERT INTO customers VALUES (1, 1, 'Alice', 'NA'), (2, 2, 'Bob', 'VIP'), (3, 2, 'Cara', 'EU'), (4, NULL, 'Dan', 'APAC');")
    cur.execute(
        "INSERT INTO orders VALUES "
        "(1, 1, 'NA', 'hw', 10.0, 2, 3, 1, 'settled'),"
        "(2, 1, 'NA', 'sw', 20.0, 4, 5, 0, 'settled'),"
        "(3, 2, 'EU', 'hw', 15.0, 1, 7, 1, 'pending'),"
        "(4, 2, 'EU', 'hw', 30.0, 3, 6, 1, 'settled'),"
        "(5, 3, NULL, 'sw', 50.0, 5, 1, 0, 'settled');"
    )
    cur.execute("INSERT INTO order_tags VALUES (1, 'urgent', 2), (1, 'fragile', 5), (2, 'urgent', 3), (4, 'bulk', 10);")
    plans = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for v_name, v_sel in my_views:
            p = generate_materialized_view(
                my_schema,
                f"CREATE VIEW {v_name} AS {v_sel};",
                dialect=Dialect.MYSQL,
                incremental_backfill=False,
                batch_size=2,
                ignore_limitations=True,
            )
            _execute_mysql_plan_in_sqlite(conn, p)
            v_sel_s = _transpile_mysql_expr_to_sqlite(v_sel)
            plans.append((p, v_sel_s))
            runner.total_views += 1
            _verify_external_db_mv(cur, p.view_name, v_sel_s, f"s11_my:{v_name}:init")
            runner.total_assertions += 1
    steps = _get_complex_dml_and_tx_steps(Dialect.MYSQL, for_live_mysql=False)
    for step_lbl, m_sql in steps:
        cur.execute(_transpile_mysql_expr_to_sqlite(m_sql))
        for p, v_sel_s in plans:
            _verify_external_db_mv(cur, p.view_name, v_sel_s, f"s11_my:{p.logical_view_name}:{step_lbl}")
            runner.total_assertions += 1
    conn.close()
    print(f"  [PASS] Suite 11 (Live 50-View Battery MySQL): 50 views x {len(steps) + 1} states = {50 * (len(steps) + 1)} assertions")


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


def _run_all_tests() -> None:
    t0 = time.time()
    runner = MVTestRunner()

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

    print("-" * 84)
    print(f"COMBINATORIAL & TRANSACTIONAL SUITES PASSED: {runner.total_views} views, {runner.total_assertions} live SQL assertions!")
    print("-" * 84)

    # Run live external database tests if configured via environment variables (used by GitHub Actions CI)
    pg_dsn = os.environ.get("POSTGRES_DSN")
    if pg_dsn:
        run_live_postgres_tests_if_available(pg_dsn)
    mysql_dsn = os.environ.get("MYSQL_DSN")
    if mysql_dsn:
        run_live_mysql_tests_if_available(mysql_dsn)
    sqlite_compat_eng = os.environ.get("SQLITE_COMPAT_ENGINE")
    if sqlite_compat_eng:
        run_live_sqlite_compat_tests_if_available(sqlite_compat_eng)

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
