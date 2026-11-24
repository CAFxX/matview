"""Which collation does a group / DISTINCT / set-operation key expression compare under?

SQLite, MySQL and MariaDB group (and de-duplicate) values by the collation of the
grouped *expression*, not of the stored bytes: ``'a'`` and ``'A'`` are one group
under ``COLLATE NOCASE``, ``'a'`` and ``'a '`` one group under any PAD SPACE
collation. The canonical key text matview stores (``__mv_pk``) must therefore
be equal exactly when the engine's own comparison says "equal". This module
derives, from the expression's AST and the declared column collations, how that
key has to be computed, or says that it cannot be derived exactly:

* ``classify_sqlite_key``: SQLite derives an expression's collation as
  documented ("Collating Sequence Selection"): a column has its declared
  collation, ``x COLLATE c`` has ``c``, ``CAST``/unary ``+``/parentheses pass
  the operand's collation through, and an operator or function takes the
  collation of its first operand that carries an *explicit* ``COLLATE`` (a
  column inside ``lower(g)``, ``g || 'x'``, ``coalesce(g, '')`` or ``CASE`` does
  not pass its collation on: the result is BINARY).
* ``classify_mysql_key``: MySQL/MariaDB decide the comparison collation at run
  time from the argument coercibility rules, so the key is the server's own
  ``WEIGHT_STRING`` of the expression (see ``dialects.base``) for any
  *character-valued* expression. What matters here is only whether the
  expression is character-valued (key from weights), not character-valued
  (byte-exact key) or not known (refused: the caller falls back).

``UNKNOWN`` always means "cannot be derived exactly"; callers refuse or fall
back to a full refresh of the original query with an announced diagnostic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import sqlglot
import sqlglot.errors
from sqlglot import exp as E

from .engines import Dialect, _sqlglot_dialect
from .logical_ir import ColumnDef

# Resolves a column reference to its declared column, or None when it is not a
# base-table column (derived-table/CTE column, unknown name, ambiguous name).
ColumnResolver = Callable[[E.Column], Optional[ColumnDef]]


class KeyKind(str, Enum):
    PLAIN = "plain"  # no character collation involved: the byte-exact key is exact
    TEXT = "text"  # character-valued: the key must follow the comparison collation
    UNKNOWN = "unknown"  # not derivable exactly


@dataclass(frozen=True)
class KeyClass:
    kind: KeyKind
    # SQLite: the derived collation (upper-case; BINARY when none applies).
    # MySQL/MariaDB: the declared collation of the first character column the
    # value derives from (a hint for multi-level checks), or None.
    collation: Optional[str] = None
    reason: str = ""
    # Set when the expression references a column that is not a declared base
    # table column (hidden/internal column or derived-table column).
    missing_column: bool = False

    @property
    def known(self) -> bool:
        return self.kind != KeyKind.UNKNOWN


def _unknown(reason: str, missing: bool = False) -> KeyClass:
    return KeyClass(KeyKind.UNKNOWN, None, reason, missing)


_SQLITE_SUPPORTED_COLLATIONS = ("BINARY", "NOCASE", "RTRIM")


def _collate_name(node: E.Collate) -> Optional[str]:
    target = node.expression
    if target is None:
        return None
    if isinstance(target, (E.Var, E.Identifier, E.Literal)):
        return str(target.name or target.this).strip().strip('"`\'').upper()
    return target.sql().strip().strip('"`\'').upper()


def _has_collate(node: E.Expr) -> bool:
    return any(True for _ in node.find_all(E.Collate))


def _has_subquery(node: E.Expr) -> bool:
    return any(True for _ in node.find_all(E.Subquery, E.Select, E.Exists, E.Window))


def _child_expressions(node: E.Expr):
    for child in node.iter_expressions():
        yield child


def classify_sqlite_key(sql: str, resolve: ColumnResolver) -> KeyClass:
    """Collation a SQLite key expression compares under (see module docstring)."""
    try:
        root = sqlglot.parse_one(sql, read=_sqlglot_dialect(Dialect.SQLITE))
    except sqlglot.errors.SqlglotError as exc:
        return _unknown(f"the expression {sql.strip()!r} cannot be parsed ({exc.__class__.__name__})")
    if root is None:
        return _unknown(f"the expression {sql.strip()!r} cannot be parsed")
    return _sqlite_derive(root, resolve)


def _sqlite_derive(node: E.Expr, resolve: ColumnResolver) -> KeyClass:
    if _has_subquery(node):
        return _unknown("it contains a subquery or window function")
    if isinstance(node, E.Collate):
        name = _collate_name(node)
        if not name:
            return _unknown("an unparseable COLLATE clause")
        return KeyClass(KeyKind.TEXT, name)
    if isinstance(node, (E.Paren, E.Cast, E.Pos, E.Neg, E.TryCast)):
        inner = node.this
        return _sqlite_derive(inner, resolve) if inner is not None else KeyClass(KeyKind.TEXT, "BINARY")
    if isinstance(node, E.Column):
        col = resolve(node)
        if col is None:
            return _unknown(f"column {node.sql()} is not a declared base-table column", missing=True)
        return KeyClass(KeyKind.TEXT, col.effective_collation or "BINARY")
    if isinstance(node, (E.Literal, E.Null, E.Boolean, E.Star, E.Placeholder, E.Parameter)):
        return KeyClass(KeyKind.TEXT, "BINARY")
    if not _has_collate(node):
        return KeyClass(KeyKind.TEXT, "BINARY")
    # An operator or function over an explicit COLLATE: the first operand (in
    # evaluation order) that carries one decides.
    for child in _child_expressions(node):
        if _has_collate(child):
            return _sqlite_derive(child, resolve)
    return _unknown("an unsupported expression around a COLLATE clause")


# --- MySQL / MariaDB ---------------------------------------------------------

_MYSQL_CHAR_TYPE_RE = re.compile(
    r"^\s*(?:NATIONAL\s+)?(?:N?CHAR|N?VARCHAR|TINYTEXT|TEXT|MEDIUMTEXT|LONGTEXT|CHARACTER)\b", re.IGNORECASE
)
_MYSQL_BINARY_CHARSET_RE = re.compile(r"CHARACTER\s+SET\s+BINARY|\bBINARY\b", re.IGNORECASE)

# Function nodes whose result is character-valued exactly when an operand is.
_TEXT_PROPAGATING = (
    E.Lower, E.Upper, E.Trim, E.Substring, E.Left, E.Right, E.Concat, E.ConcatWs, E.Coalesce, E.Nullif,
    E.Greatest, E.Least, E.Reverse, E.Repeat, E.Replace, E.Pad, E.Initcap, E.Paren,
)
# Anonymous (unmodelled) function names, MySQL/MariaDB.
_TEXT_PROPAGATING_NAMES = frozenset(
    "LCASE UCASE LTRIM RTRIM MID SUBSTR SUBSTRING_INDEX LPAD RPAD INSERT ELT IFNULL TRIM_ORACLE REPLACE REVERSE "
    "REPEAT SPACE CONCAT CONCAT_WS LEFT RIGHT SUBSTRING LOWER UPPER COALESCE NULLIF GREATEST LEAST".split()
)
# Result is always character-valued (the comparison collation is the server's).
_TEXT_ALWAYS_NAMES = frozenset(
    "TIME_TO_STR DATE_FORMAT TIME_FORMAT MONTHNAME DAYNAME HEX MD5 SHA SHA1 SHA2 FORMAT UUID BIN OCT CONV CHAR DATABASE "
    "USER VERSION CURRENT_USER SOUNDEX QUOTE TO_BASE64 FROM_BASE64 UNHEX MD5 SHA1 SHA2 TO_BASE FROM_BASE".split()
)
# Result is never character-valued, whatever the operands.
_PLAIN_NAMES = frozenset(
    "LENGTH CHAR_LENGTH CHARACTER_LENGTH OCTET_LENGTH BIT_LENGTH ASCII ORD LOCATE INSTR POSITION STRCMP FIELD "
    "FIND_IN_SET CRC32 ABS ROUND FLOOR CEIL CEILING SIGN SQRT POW POWER MOD EXP LN LOG LOG2 LOG10 TRUNCATE "
    "YEAR MONTH DAY DAYOFMONTH DAYOFWEEK DAYOFYEAR WEEK WEEKDAY WEEKOFYEAR QUARTER HOUR MINUTE SECOND "
    "TO_DAYS FROM_DAYS UNIX_TIMESTAMP DATEDIFF TIMESTAMPDIFF DATE TIME TIMESTAMP LAST_DAY NOW CURDATE "
    "CURRENT_DATE CURRENT_TIMESTAMP DATE_ADD DATE_SUB ADDDATE SUBDATE STR_TO_DATE MAKEDATE EXTRACT "
    "ISNULL BIT_COUNT RAND PI TS_OR_DS_TO_DATE TS_OR_DS_TO_TIME TS_OR_DS_TO_TIMESTAMP TS_OR_DS_TO_DATETIME "
    "UNIX_TO_TIME TIME_TO_UNIX LEVENSHTEIN CURRENT_TIME LOCALTIME LOCALTIMESTAMP".split()
)
_PLAIN_NODES = (
    E.Add, E.Sub, E.Mul, E.Div, E.Mod, E.IntDiv, E.Neg, E.Pos, E.Abs, E.Round, E.Floor, E.Ceil, E.Length,
    E.EQ, E.NEQ, E.GT, E.GTE, E.LT, E.LTE, E.NullSafeEQ, E.NullSafeNEQ, E.And, E.Or, E.Not, E.Xor, E.Is,
    E.Between, E.In, E.Like, E.ILike, E.RegexpLike, E.BitwiseAnd, E.BitwiseOr, E.BitwiseXor, E.BitwiseNot,
    E.BitwiseLeftShift, E.BitwiseRightShift, E.Extract, E.Year, E.Month, E.Day, E.DateAdd, E.DateSub,
    E.DateDiff, E.CurrentDate, E.CurrentTimestamp, E.Boolean, E.Null, E.Exp, E.Ln, E.Sqrt, E.Pow,
)
_TEXT_ALWAYS_NODES = (E.TimeToStr,)
_MYSQL_TEXT_CAST = frozenset({"CHAR", "VARCHAR", "NCHAR", "NVARCHAR", "TEXT", "TINYTEXT", "MEDIUMTEXT", "LONGTEXT"})
_MYSQL_CAST_PLAIN = frozenset(
    "BINARY VARBINARY BLOB SIGNED UNSIGNED INT BIGINT DECIMAL DOUBLE FLOAT DATE DATETIME TIME YEAR JSON BOOLEAN "
    "TINYINT SMALLINT MEDIUMINT REAL BIT TIMESTAMP".split()
)


def mysql_column_is_character(col: ColumnDef) -> bool:
    """True when the column holds character data compared under a collation."""
    dt = col.data_type or ""
    if not _MYSQL_CHAR_TYPE_RE.match(dt):
        return False
    return not _MYSQL_BINARY_CHARSET_RE.search(dt.upper().split("COLLATE")[0])


def classify_mysql_key(sql: str, resolve: ColumnResolver, dialect: Dialect = Dialect.MYSQL) -> KeyClass:
    """Whether a MySQL/MariaDB key expression is character-valued (see module docstring)."""
    try:
        root = sqlglot.parse_one(sql, read=_sqlglot_dialect(dialect))
    except sqlglot.errors.SqlglotError as exc:
        return _unknown(f"the expression {sql.strip()!r} cannot be parsed ({exc.__class__.__name__})")
    if root is None:
        return _unknown(f"the expression {sql.strip()!r} cannot be parsed")
    return _mysql_derive(root, resolve)


def _combine(parts: "list[KeyClass]") -> KeyClass:
    for p in parts:
        if p.kind == KeyKind.UNKNOWN:
            return p
    texts = [p for p in parts if p.kind == KeyKind.TEXT]
    if not texts:
        return KeyClass(KeyKind.PLAIN)
    named = {(p.collation or "").upper() for p in texts if p.collation}
    if len(named) > 1:
        # The server picks the comparison collation by coercibility at run time.
        return _unknown(f"its operands have different collations ({', '.join(sorted(named))})")
    return texts[0] if texts[0].collation or not named else KeyClass(KeyKind.TEXT, next(iter(named)))


def _mysql_derive(node: E.Expr, resolve: ColumnResolver) -> KeyClass:
    if _has_subquery(node):
        return _unknown("it contains a subquery or window function")
    if isinstance(node, E.Collate):
        return KeyClass(KeyKind.TEXT, _collate_name(node))
    if isinstance(node, E.Paren):
        return _mysql_derive(node.this, resolve)
    if isinstance(node, E.Column):
        col = resolve(node)
        if col is None:
            return _unknown(f"column {node.sql()} is not a declared base-table column", missing=True)
        if mysql_column_is_character(col):
            return KeyClass(KeyKind.TEXT, col.effective_collation)
        return KeyClass(KeyKind.PLAIN)
    if isinstance(node, (E.Literal, E.Boolean, E.Null)):
        return KeyClass(KeyKind.PLAIN)
    if isinstance(node, (E.Cast, E.TryCast)):
        to = node.args.get("to")
        tname = ""
        if isinstance(to, E.DataType):
            tname = (to.this.name if hasattr(to.this, "name") else str(to.this)).upper()
        if tname in _MYSQL_TEXT_CAST or isinstance(node.args.get("format"), E.Expr):
            return KeyClass(KeyKind.TEXT, None)
        if tname in _MYSQL_CAST_PLAIN:
            return KeyClass(KeyKind.PLAIN)
        return _unknown(f"a cast to {tname or 'an unknown type'}")
    if isinstance(node, _TEXT_ALWAYS_NODES):
        return KeyClass(KeyKind.TEXT, None)
    if isinstance(node, _PLAIN_NODES):
        return KeyClass(KeyKind.PLAIN)
    if isinstance(node, E.If):
        return _combine([_mysql_derive(a, resolve) for a in (node.args.get("true"), node.args.get("false")) if a is not None])
    if isinstance(node, E.Case):
        parts = [_mysql_derive(i.args["true"], resolve) for i in node.args.get("ifs") or [] if i.args.get("true") is not None]
        default = node.args.get("default")
        if default is not None:
            parts.append(_mysql_derive(default, resolve))
        return _combine(parts)
    if isinstance(node, _TEXT_PROPAGATING):
        return _combine([_mysql_derive(c, resolve) for c in _child_expressions(node)])
    if isinstance(node, E.Func):
        name = (str(node.name or "") if isinstance(node, E.Anonymous) else node.sql_name()).upper()
        if name in _TEXT_PROPAGATING_NAMES:
            return _combine([_mysql_derive(c, resolve) for c in node.expressions])
        if name in _TEXT_ALWAYS_NAMES:
            return KeyClass(KeyKind.TEXT, None)
        if name in _PLAIN_NAMES:
            return KeyClass(KeyKind.PLAIN)
    return _unknown(f"an expression ({type(node).__name__}) whose character collation is not known")


_MARIADB_AI_CS_RE = re.compile(r"uca1400_(?:[a-z0-9]+_)*ai_cs$")
_MARIADB_PAD_MULTILEVEL_RE = re.compile(r"uca1400_(?:[a-z0-9]+_)*(?:as_ci|as_cs)$")


def mysql_collation_has_unsound_weights(name: str) -> bool:
    """True for MariaDB collations whose equality is not the equality of the
    (pad-normalised) weight string: ``uca1400_*_ai_cs`` (``'a' = 'á'`` while the
    weight strings differ), the PAD SPACE multi-level ``uca1400_*_as_ci/_as_cs``
    (pad weights are spread over the levels) and ``*_thai_520_w2``. Measured natively
    on MariaDB 11.8 over every collation of the utf8mb4, utf8mb3, latin1, ucs2, utf16
    and utf32 character sets: all other collations' equality is exactly the equality
    of their pad-normalised weight strings (see test_group_key_collation.py)."""
    n = (name or "").lower()
    return (
        "thai_520_w2" in n
        or bool(_MARIADB_AI_CS_RE.search(n))
        or ("nopad" not in n and bool(_MARIADB_PAD_MULTILEVEL_RE.search(n)))
    )


def mysql_collation_has_spelling_variants(name: Optional[str]) -> bool:
    """False only for collations under which equal values are byte-identical (NO PAD
    binary collations); True for every other, and for an unknown, collation."""
    n = (name or "").lower()
    return not (n == "binary" or (n.endswith("_bin") and ("0900" in n or "nopad" in n)))


# MySQL (not MariaDB) PAD SPACE collations whose GROUP BY / DISTINCT equality is the
# same in every execution plan. Measured natively on MySQL 8.4 and 9.7 over every
# collation of the utf8mb4, utf8mb3, latin1, ascii, ucs2, utf16 and utf32 character
# sets (plus the single-byte character sets, see below): ``'a' = 'a' || c`` for a
# character c other than U+0020 that weighs like a space (NBSP and the other
# Unicode spaces under the UCA collations ``unicode_ci``, ``unicode_520_ci`` and
# the language collations) is TRUE, but GROUP BY / DISTINCT pad only U+0020 when
# they use a temporary table and pad by weight when they walk an index (or run
# COUNT(DISTINCT)). The original view is then plan-dependent: no key text can
# reproduce it. Every other measured PAD SPACE collation (``*_bin``,
# ``*_general_ci``, ``*_general_mysql500_ci``, ``*_tolower_ci``, ``latin1_*``,
# ``ascii_*``) has no such character; every NO PAD collation (``utf8mb4_0900_*``)
# does not pad at all.
_MYSQL_PAD_STABLE_CHARSETS = frozenset({"utf8mb4", "utf8mb3", "utf8", "ucs2", "utf16", "utf32", "latin1", "ascii"})
_MYSQL_PAD_STABLE_SUFFIXES = ("_bin", "_general_ci", "_general_mysql500_ci", "_tolower_ci")


def mysql_collation_is_plan_dependent(name: Optional[str]) -> bool:
    """True when a MySQL PAD SPACE collation may group (and de-duplicate) the same
    values differently depending on the execution plan, so that the original view
    itself has no single result (see above). False for NO PAD collations and for the
    measured PAD SPACE collations whose only pad character is U+0020. A collation
    that was not measured is plan-dependent (refused, never guessed); ``None`` (no
    collation known, an offline plan) is not."""
    if not name:
        return False
    n = name.strip().lower()
    if n == "binary" or "0900" in n:
        return False
    charset = n.split("_", 1)[0]
    if charset in _MYSQL_PAD_STABLE_CHARSETS and (
        n.endswith(_MYSQL_PAD_STABLE_SUFFIXES) or charset in ("latin1", "ascii")
    ):
        return False
    return True
