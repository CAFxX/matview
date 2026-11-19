"""Schema-graph-aware counterexample reduction for generated differential scenarios.

The reducer removes only leaf tables (tables not referenced by another remaining
table's foreign key), together with dependent views, indexes, seed rows, and DML.
A caller-supplied failure-signature oracle decides whether each candidate still
reproduces the same defect. Generated SQL identifiers are intentionally simple.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Scenario = Tuple[
    str,
    List[Tuple[str, str]],
    List[str],
    List[Tuple[str, str]],
    Dict[str, Any],
]
FailureSignature = Callable[[Scenario], Any]

_TABLE_RE = re.compile(
    r'^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?'
    r'[`"]?([A-Za-z_][A-Za-z0-9_]*)[`"]?\b',
    re.IGNORECASE,
)
_INDEX_RE = re.compile(
    r'^\s*CREATE\s+(?:UNIQUE\s+)?INDEX\b.*?\bON\s+[`"]?'
    r'([A-Za-z_][A-Za-z0-9_]*)[`"]?\b',
    re.IGNORECASE | re.DOTALL,
)


def _split_statements(sql: str) -> List[str]:
    """Split generated SQL at semicolons outside quoted strings/identifiers."""
    statements: List[str] = []
    start = 0
    quote: Optional[str] = None
    i = 0
    while i < len(sql):
        char = sql[i]
        if quote:
            if char == quote:
                if i + 1 < len(sql) and sql[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif char in ("'", '"', "`"):
            quote = char
        elif char == ";":
            statement = sql[start:i].strip()
            if statement:
                statements.append(statement)
            start = i + 1
        i += 1
    tail = sql[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _split_top_level(text: str) -> List[str]:
    """Split a table-definition body on commas outside nested expressions."""
    parts: List[str] = []
    start = 0
    depth = 0
    quote: Optional[str] = None
    i = 0
    while i < len(text):
        char = text[i]
        if quote:
            if char == quote:
                if i + 1 < len(text) and text[i + 1] == quote:
                    i += 2
                    continue
                quote = None
        elif char in ("'", '"', "`"):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _identifier_pattern(name: str) -> str:
    return rf'[`"]?{re.escape(name)}[`"]?\b'


def _uses_table(sql: str, table: str, *, mutation: bool = False) -> bool:
    identifier = _identifier_pattern(table)
    if mutation:
        pattern = (
            rf'\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|FROM|JOIN)\s+'
            rf'{identifier}'
        )
    else:
        pattern = rf'\b(?:FROM|JOIN)\s+{identifier}'
    return re.search(pattern, sql, re.IGNORECASE) is not None


def _table_names(statements: Sequence[str]) -> List[str]:
    names = []
    for statement in statements:
        match = _TABLE_RE.match(statement)
        if match:
            names.append(match.group(1))
    return names


def _table_column_count(statement: str) -> int:
    opening = statement.find("(")
    closing = statement.rfind(")")
    if opening < 0 or closing <= opening:
        return 0
    definitions = _split_top_level(statement[opening + 1 : closing])
    constraints = re.compile(
        r'^(?:CONSTRAINT\b|PRIMARY\s+KEY\b|FOREIGN\s+KEY\b|'
        r'UNIQUE\b|CHECK\s*\()',
        re.IGNORECASE,
    )
    return sum(not constraints.match(part) for part in definitions)


def _refresh_shape(
    original: Dict[str, Any],
    table_statements: Sequence[str],
    index_statements: Sequence[str],
    seed_sql: Sequence[str],
) -> Dict[str, Any]:
    names = _table_names(table_statements)
    counts = [
        sum(
            re.search(
                rf'\bINSERT\s+INTO\s+{_identifier_pattern(name)}',
                statement,
                re.IGNORECASE,
            )
            is not None
            for statement in seed_sql
        )
        for name in names
    ]
    delete_actions = set()
    update_actions = set()
    for statement in table_statements:
        delete_actions.update(
            re.findall(r'\bON\s+DELETE\s+(CASCADE|RESTRICT|SET\s+NULL)', statement, re.I)
        )
        update_actions.update(
            re.findall(r'\bON\s+UPDATE\s+(CASCADE|RESTRICT|SET\s+NULL)', statement, re.I)
        )

    shape = dict(original)
    shape.update(
        {
            "topology": "shrunk",
            "table_count": len(names),
            "column_counts": tuple(_table_column_count(stmt) for stmt in table_statements),
            "row_counts": tuple(counts),
            "empty_tables": sum(count == 0 for count in counts),
            "sparse_tables": sum(0 < count <= 3 for count in counts),
            "dense_tables": sum(count >= 8 for count in counts),
            "foreign_key_count": sum(
                len(re.findall(r'\bREFERENCES\s+', statement, re.I))
                for statement in table_statements
            ),
            "index_count": len(index_statements),
            "generated_columns": sum(
                len(re.findall(r'\bGENERATED\s+ALWAYS\b', stmt, re.I))
                for stmt in table_statements
            ),
            "composite_primary_key_tables": sum(
                bool(re.search(r'\bPRIMARY\s+KEY\s*\([^)]*,', stmt, re.I))
                for stmt in table_statements
            ),
            "check_constraints": sum(
                len(re.findall(r'\bCHECK\s*\(', stmt, re.I))
                for stmt in table_statements
            ),
            "fk_delete_actions": tuple(sorted({value.upper() for value in delete_actions})),
            "fk_update_actions": tuple(sorted({value.upper() for value in update_actions})),
            "minimized": True,
            "coverage_metadata_valid": False,
        }
    )
    return shape


def _drop_leaf_table(scenario: Scenario, table: str) -> Optional[Scenario]:
    schema_sql, views, seed_sql, mutations, shape = scenario
    statements = _split_statements(schema_sql)
    table_statements = [stmt for stmt in statements if _TABLE_RE.match(stmt)]
    target_statement = next(
        (stmt for stmt in table_statements
         if (m := _TABLE_RE.match(stmt)) and m.group(1).lower() == table.lower()),
        None,
    )
    if target_statement is None:
        return None

    # A referenced parent cannot be removed while any other table still has an
    # FK to it. Removing child/leaf tables first makes the graph reduction safe.
    for statement in table_statements:
        match = _TABLE_RE.match(statement)
        if match and match.group(1).lower() != table.lower():
            if re.search(
                rf'\bREFERENCES\s+{_identifier_pattern(table)}',
                statement,
                re.IGNORECASE,
            ):
                return None

    remaining_statements = []
    remaining_tables = []
    remaining_indexes = []
    for statement in statements:
        table_match = _TABLE_RE.match(statement)
        index_match = _INDEX_RE.match(statement)
        if table_match and table_match.group(1).lower() == table.lower():
            continue
        if index_match and index_match.group(1).lower() == table.lower():
            continue
        remaining_statements.append(statement)
        if table_match:
            remaining_tables.append(statement)
        if index_match:
            remaining_indexes.append(statement)

    if not remaining_tables:
        return None

    remaining_views = [
        view for view in views if not _uses_table(view[1], table)
    ]
    if not remaining_views:
        return None
    remaining_seed = [
        statement for statement in seed_sql
        if not re.search(
            rf'\bINSERT\s+INTO\s+{_identifier_pattern(table)}',
            statement,
            re.IGNORECASE,
        )
    ]
    remaining_mutations = [
        mutation for mutation in mutations
        if not _uses_table(mutation[1], table, mutation=True)
    ]

    new_shape = _refresh_shape(shape, remaining_tables, remaining_indexes, remaining_seed)
    return (
        ";\n".join(remaining_statements) + ";",
        remaining_views,
        remaining_seed,
        remaining_mutations,
        new_shape,
    )


def minimize_schema_graph(
    scenario: Scenario,
    failure_signature: FailureSignature,
    *,
    max_evaluations: int = 128,
) -> Scenario:
    """Greedily remove leaf tables while preserving the exact failure signature.

    The callback must return a stable, non-None signature for the target failure
    and None for invalid/non-reproducing candidates. Candidate evaluation is
    bounded and deterministic. This first graph reducer removes whole leaf
    tables and dependent artifacts; it intentionally does not yet minimize
    columns, constraints, or individual foreign-key edges.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    original_signature = failure_signature(scenario)
    if original_signature is None:
        return scenario

    current = scenario
    evaluations = 0
    while evaluations < max_evaluations:
        statements = _split_statements(current[0])
        names = _table_names(statements)
        changed = False
        for table in reversed(names):
            candidate = _drop_leaf_table(current, table)
            if candidate is None:
                continue
            evaluations += 1
            if failure_signature(candidate) == original_signature:
                current = candidate
                changed = True
                break
            if evaluations >= max_evaluations:
                break
        if not changed:
            break
    return current
