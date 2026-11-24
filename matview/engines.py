"""Engine/dialect enumerations, version parsing (`packaging`), and error types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import dataclasses
import functools
import math
import re
import sqlite3
from typing import Any, Optional, Sequence, Tuple, Union

from packaging.version import InvalidVersion, Version

DEFAULT_OBJECT_PREFIX = "__matview_"


class Dialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"


class EngineSubdialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MARIADB = "mariadb"


@functools.total_ordering
@dataclass(frozen=True)
class EngineVersion:
    """
    Represents a specific database engine, subdialect (`sqlite`, `postgres`, `mysql`, `mariadb`),
    and semantic version for version-optimal SQL generation.
    Delegates semantic version parsing, normalization, and rich comparison to `packaging.version.Version`.
    """
    dialect: Dialect
    subdialect: EngineSubdialect
    version: Tuple[int, int, int]
    raw_version: str = field(default="", compare=False)
    # MariaDB fixes VAR/STDDEV result scale using this session setting at
    # parse/prepare time. Capture it as a literal: MariaDB forbids system-variable
    # references in CREATE VIEW definitions.
    mariadb_div_precision_increment: int = field(default=4, compare=False)
    # Names (lower case) of VAR_*/STDDEV_* aggregates the SQLite connection
    # defines (sqlean's `stats` extension, a loaded extension, an
    # application-defined function) whose results do NOT match the
    # engine-neutral exact definition matview maintains. Found by probing the
    # connection (`probe_sqlite_nonstandard_statistics`), never by guessing from
    # the driver, and independently of any pinned engine version.
    sqlite_nonstandard_statistics: frozenset[str] = field(default_factory=frozenset, compare=False)
    packaging_version: Optional[Version] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.packaging_version is None:
            pkg_v: Optional[Version] = None
            if self.raw_version:
                try:
                    pkg_v, _ = _extract_packaging_and_tuple(
                        self.raw_version, self.version, require_valid=False
                    )
                except Exception:
                    pkg_v = None
            if pkg_v is None:
                pkg_v = Version(f"{self.version[0]}.{self.version[1]}.{self.version[2]}")
            object.__setattr__(self, "packaging_version", pkg_v)

    @property
    def major(self) -> int:
        return self.version[0]

    @property
    def minor(self) -> int:
        return self.version[1]

    @property
    def patch(self) -> int:
        return self.version[2]

    @property
    def version_str(self) -> str:
        if self.packaging_version is not None:
            return str(self.packaging_version)
        return f"{self.version[0]}.{self.version[1]}.{self.version[2]}"

    @property
    def pg_version_num(self) -> int:
        return self.version[0] * 10000 + self.version[1] * 100 + self.version[2]

    @property
    def label(self) -> str:
        return f"{self.subdialect.value} {self.version_str}"

    @property
    def is_sqlite(self) -> bool:
        return self.dialect == Dialect.SQLITE

    @property
    def is_postgres(self) -> bool:
        return self.dialect == Dialect.POSTGRES

    @property
    def is_mysql(self) -> bool:
        return self.subdialect == EngineSubdialect.MYSQL

    @property
    def is_mariadb(self) -> bool:
        return self.subdialect == EngineSubdialect.MARIADB

    @property
    def is_prerelease(self) -> bool:
        return bool(self.packaging_version and self.packaging_version.is_prerelease)

    @property
    def is_postrelease(self) -> bool:
        return bool(self.packaging_version and self.packaging_version.is_postrelease)

    @property
    def is_devrelease(self) -> bool:
        return bool(self.packaging_version and self.packaging_version.is_devrelease)

    @property
    def base_version(self) -> str:
        return self.packaging_version.base_version if self.packaging_version else self.version_str

    def at_least(self, target: Version | str | Tuple[int, ...] | EngineVersion) -> bool:
        """
        Returns True if this EngineVersion satisfies the specified target version floor.
        Accepts packaging.version.Version, version string (e.g. '15.0', '8.4.0rc1'),
        version tuple (e.g. (15, 0, 0)), or another EngineVersion.
        """
        if isinstance(target, EngineVersion):
            assert target.packaging_version is not None
            target_v = target.packaging_version
        elif isinstance(target, Version):
            target_v = target
        elif isinstance(target, tuple):
            target_v = Version(".".join(str(x) for x in target))
        elif isinstance(target, str):
            target_v = _to_packaging_version(target)
        else:
            raise TypeError(f"Unsupported target type for at_least: {type(target)!r}")
        assert self.packaging_version is not None
        return self.packaging_version >= target_v

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EngineVersion):
            return NotImplemented
        return (
            self.dialect == other.dialect
            and self.subdialect == other.subdialect
            and self.packaging_version == other.packaging_version
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, EngineVersion):
            return NotImplemented
        if (self.dialect, self.subdialect) != (other.dialect, other.subdialect):
            return (self.dialect.value, self.subdialect.value) < (
                other.dialect.value,
                other.subdialect.value,
            )
        assert self.packaging_version is not None
        assert other.packaging_version is not None
        return self.packaging_version < other.packaging_version

    def __hash__(self) -> int:
        return hash((self.dialect, self.subdialect, self.packaging_version))

    @property
    def supports_pg_statement_maintenance(self) -> bool:
        """Statement-level triggers with transition tables (PostgreSQL 10)
        created with ``EXECUTE FUNCTION`` (PostgreSQL 11): the only
        PostgreSQL maintenance mechanism."""
        return self.is_postgres and self.at_least((11, 0, 0))

    @property
    def supports_pg_nulls_not_distinct(self) -> bool:
        return self.is_postgres and self.at_least((15, 0, 0))

    @property
    def supports_pg_merge(self) -> bool:
        return self.is_postgres and self.at_least((15, 0, 0))

    @property
    def supports_sqlite_upsert(self) -> bool:
        return self.is_sqlite and self.at_least((3, 24, 0))

    @property
    def supports_sqlite_window_functions(self) -> bool:
        return self.is_sqlite and self.at_least((3, 25, 0))

    @property
    def supports_sqlite_returning(self) -> bool:
        return self.is_sqlite and self.at_least((3, 35, 0))

    @property
    def supports_mysql_window_functions(self) -> bool:
        return (self.is_mysql and self.at_least((8, 0, 0))) or (
            self.is_mariadb and self.at_least((10, 2, 0))
        )

    @property
    def supports_mysql_row_alias_upsert(self) -> bool:
        return self.is_mysql and self.at_least((8, 0, 19))


BARE_DIALECT_NAMES = {
    "postgres",
    "postgresql",
    "pg",
    "cockroach",
    "yugabyte",
    "mysql",
    "percona",
    "tidb",
    "mariadb",
    "sqlite",
    "sqlite3",
    "apsw",
    "sqlcipher",
    "sqlean",
    "libsql",
}


def _extract_packaging_and_tuple(
    text_s: str,
    default_ver: Tuple[int, int, int],
    require_valid: bool = False,
) -> Tuple[Version, Tuple[int, int, int]]:
    """
    Extracts a standardized packaging.version.Version and backward-compatible
    (major, minor, patch) tuple from arbitrary database engine banners or version strings.
    """
    s = (text_s or "").strip()
    if not s:
        if require_valid:
            raise ValueError(f"Empty version string in specification {text_s!r}")
        v_str = f"{default_ver[0]}.{default_ver[1]}.{default_ver[2]}"
        return Version(v_str), default_ver

    # 1. PostgreSQL 5- or 6-digit server_version_num like 170002 or 90624
    m_pgnum = re.search(r"\b(\d{5,6})\b", s)
    if m_pgnum and "." not in s:
        num = int(m_pgnum.group(1))
        maj = num // 10000
        min_v = (num % 10000) // 100
        pat = num % 100
        ver_tuple = (maj, min_v, pat)
        return Version(f"{maj}.{min_v}.{pat}"), ver_tuple

    # 2. MariaDB banner (e.g. '5.5.5-10.11.8-MariaDB-1:10.11.8+maria~ubu2204' or '10.11.8-MariaDB')
    cleaned = re.sub(r"^5\.5\.5-", "", s, flags=re.IGNORECASE)
    m_maria = re.search(r"(\d+\.\d+(?:\.\d+)?(?:[-._]?[a-zA-Z0-9]+)?)(?=-MariaDB)", cleaned, flags=re.IGNORECASE)
    if m_maria:
        cand = m_maria.group(1)
        cand_norm = re.sub(r"-(rc|beta|alpha|b|a)", r"\1", cand, flags=re.IGNORECASE)
        try:
            pv = Version(cand_norm)
            rel = pv.release
            maj = rel[0] if len(rel) > 0 else default_ver[0]
            min_v = rel[1] if len(rel) > 1 else 0
            pat = rel[2] if len(rel) > 2 else 0
            if not (pv.is_prerelease or pv.is_postrelease or pv.is_devrelease) and len(rel) < 3:
                pv = Version(f"{maj}.{min_v}.{pat}")
            return pv, (maj, min_v, pat)
        except InvalidVersion:
            pass

    # 3. Strip engine prefixes
    cleaned = re.sub(r"\bsqlite3\b", "sqlite", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:postgres(?:ql)?|pg|mysql|mariadb|sqlite)[-:_/\s]*", "", cleaned, flags=re.IGNORECASE)

    # 4. Strip distribution build suffixes
    cleaned = re.sub(r"-(?:0ubuntu|\d+:|\d+ubuntu|\d+build|\d+deb).*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[+~].*$", "", cleaned)

    # 5. Check if anything is left
    cand_first = cleaned.split()[0] if cleaned.split() else cleaned
    cand_first = cand_first.rstrip(";,.)]")

    # Normalize pre-releases
    cand_norm = re.sub(r"-(rc|beta|alpha|b|a|dev|post)", r"\1", cand_first, flags=re.IGNORECASE)
    cand_norm = re.sub(r"\bbeta\b", "b", cand_norm, flags=re.IGNORECASE)
    cand_norm = re.sub(r"\balpha\b", "a", cand_norm, flags=re.IGNORECASE)

    if cand_norm:
        try:
            pv = Version(cand_norm)
            rel = pv.release
            maj = rel[0] if len(rel) > 0 else default_ver[0]
            min_v = rel[1] if len(rel) > 1 else 0
            pat = rel[2] if len(rel) > 2 else 0
            if not (pv.is_prerelease or pv.is_postrelease or pv.is_devrelease) and len(rel) < 3:
                pv = Version(f"{maj}.{min_v}.{pat}")
            return pv, (maj, min_v, pat)
        except InvalidVersion:
            pass

    # 6. Fallback regex search for digits
    m_ver = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?((?:a|b|rc|dev|post|\.post)\d+)?", cleaned, flags=re.IGNORECASE)
    if m_ver:
        maj = int(m_ver.group(1))
        min_v = int(m_ver.group(2)) if m_ver.group(2) is not None else 0
        pat = int(m_ver.group(3)) if m_ver.group(3) is not None else 0
        suffix = m_ver.group(4) or ""
        v_str = f"{maj}.{min_v}.{pat}{suffix}"
        try:
            pv = Version(v_str)
            return pv, (maj, min_v, pat)
        except InvalidVersion:
            return Version(f"{maj}.{min_v}.{pat}"), (maj, min_v, pat)

    if require_valid:
        raise ValueError(f"Invalid engine version specification {text_s!r}: cannot extract valid semantic version.")

    v_str = f"{default_ver[0]}.{default_ver[1]}.{default_ver[2]}"
    return Version(v_str), default_ver


def _extract_version_tuple(
    text_s: str, default_ver: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    return _extract_packaging_and_tuple(text_s, default_ver, require_valid=False)[1]


def _to_packaging_version(
    spec: Version | str | Tuple[int, ...] | EngineVersion,
    default_ver: Tuple[int, int, int] = (0, 0, 0),
) -> Version:
    if isinstance(spec, EngineVersion):
        assert spec.packaging_version is not None
        return spec.packaging_version
    if isinstance(spec, Version):
        return spec
    if isinstance(spec, tuple):
        return Version(".".join(str(x) for x in spec))
    if isinstance(spec, str):
        try:
            return Version(spec)
        except InvalidVersion:
            pv, _ = _extract_packaging_and_tuple(spec, default_ver, require_valid=True)
            return pv
    raise TypeError(f"Cannot convert {spec!r} of type {type(spec)!r} to packaging.version.Version")


def parse_engine_version(
    spec: Optional[EngineVersion | str | Dialect] = None,
    default_dialect: Optional[Dialect | str] = None,
) -> EngineVersion:
    """
    Parses an engine/version specification (e.g. `'postgres:17'`, `'postgresql-16.4'`,
    `'mariadb:11.4.2'`, `'mysql:8.4'`, `'sqlite:3.45.1'`, `'10.11.8-MariaDB'`, or a bare
    version `'17.2'` with `default_dialect`) into a normalized `EngineVersion`.
    """
    if isinstance(spec, EngineVersion):
        return spec

    def_d: Optional[Dialect] = None
    def_sub: Optional[EngineSubdialect] = None
    if isinstance(default_dialect, Dialect):
        def_d = default_dialect
        def_sub = EngineSubdialect(default_dialect.value)
    elif isinstance(default_dialect, str) and default_dialect.strip():
        dl_s = default_dialect.strip().lower()
        if "mariadb" in dl_s:
            def_d = Dialect.MYSQL
            def_sub = EngineSubdialect.MARIADB
        elif any(k in dl_s for k in ("postgres", "pg")):
            def_d = Dialect.POSTGRES
            def_sub = EngineSubdialect.POSTGRES
        elif "mysql" in dl_s:
            def_d = Dialect.MYSQL
            def_sub = EngineSubdialect.MYSQL
        elif "sqlite" in dl_s:
            def_d = Dialect.SQLITE
            def_sub = EngineSubdialect.SQLITE

    if isinstance(spec, Dialect):
        if spec == Dialect.POSTGRES:
            return EngineVersion(
                Dialect.POSTGRES,
                EngineSubdialect.POSTGRES,
                (14, 0, 0),
                "14.0.0",
                packaging_version=Version("14.0.0"),
            )
        if spec == Dialect.MYSQL:
            sub = EngineSubdialect.MARIADB if def_sub == EngineSubdialect.MARIADB else EngineSubdialect.MYSQL
            ver = (11, 4, 0) if sub == EngineSubdialect.MARIADB else (8, 0, 0)
            v_str = f"{ver[0]}.{ver[1]}.{ver[2]}"
            return EngineVersion(
                Dialect.MYSQL,
                sub,
                ver,
                v_str,
                packaging_version=Version(v_str),
            )
        sq_v = sqlite3.sqlite_version_info[:3]
        ver = (sq_v[0], sq_v[1], sq_v[2])
        v_str = f"{ver[0]}.{ver[1]}.{ver[2]}"
        return EngineVersion(
            Dialect.SQLITE,
            EngineSubdialect.SQLITE,
            ver,
            sqlite3.sqlite_version,
            packaging_version=Version(v_str),
        )

    s = (spec or "").strip()
    if not s:
        target_d = def_d or Dialect.SQLITE
        return parse_engine_version(target_d, default_dialect=default_dialect)

    s_low = s.lower()
    target_d: Optional[Dialect] = None
    target_sub: Optional[EngineSubdialect] = None
    def_ver: Tuple[int, int, int] = (0, 0, 0)

    if "mariadb" in s_low:
        target_d = Dialect.MYSQL
        target_sub = EngineSubdialect.MARIADB
        def_ver = (11, 4, 0)
    elif any(k in s_low for k in ("postgres", "postgresql", "pg", "cockroach", "yugabyte")):
        target_d = Dialect.POSTGRES
        target_sub = EngineSubdialect.POSTGRES
        def_ver = (14, 0, 0)
    elif any(k in s_low for k in ("mysql", "percona", "tidb")):
        target_d = Dialect.MYSQL
        target_sub = EngineSubdialect.MYSQL
        def_ver = (8, 0, 0)
    elif any(k in s_low for k in ("sqlite", "apsw", "sqlcipher", "sqlean", "libsql")):
        target_d = Dialect.SQLITE
        target_sub = EngineSubdialect.SQLITE
        sq_def = sqlite3.sqlite_version_info[:3]
        def_ver = (sq_def[0], sq_def[1], sq_def[2])
    elif def_d is not None:
        target_d = def_d
        if def_d == Dialect.POSTGRES:
            target_sub = EngineSubdialect.POSTGRES
            def_ver = (14, 0, 0)
        elif def_d == Dialect.MYSQL:
            target_sub = EngineSubdialect.MARIADB if def_sub == EngineSubdialect.MARIADB else EngineSubdialect.MYSQL
            def_ver = (11, 4, 0) if target_sub == EngineSubdialect.MARIADB else (8, 0, 0)
        else:
            target_sub = EngineSubdialect.SQLITE
            sq_def = sqlite3.sqlite_version_info[:3]
            def_ver = (sq_def[0], sq_def[1], sq_def[2])
    else:
        raise ValueError(
            f"Cannot determine database dialect from engine/version specification {spec!r}; "
            f"use '<engine>:<version>' (e.g. 'postgres:17', 'mysql:8.4', 'mariadb:11.4', 'sqlite:3.45')."
        )

    if s_low in BARE_DIALECT_NAMES:
        v_str = f"{def_ver[0]}.{def_ver[1]}.{def_ver[2]}"
        pv = Version(v_str)
        ver = def_ver
    else:
        pv, ver = _extract_packaging_and_tuple(s, def_ver, require_valid=True)

    return EngineVersion(target_d, target_sub, ver, s, packaging_version=pv)


SQLITE_STATISTICAL_FUNCTIONS: Tuple[str, ...] = ("var_pop", "var_samp", "stddev_pop", "stddev_samp")

# name -> (aggregate over 1..5, over one row, over no row): the exact definition
# (population variance of 1..5 is 2, sample variance 2.5, both exactly
# representable; a sample statistic of one value is NULL, a population one is 0;
# nothing aggregated is NULL). sqlean's stats, for one, return 0.0 for the
# statistic of one value (sample) and of no value.
_SQLITE_STATISTICS_EXPECTED = {
    "var_pop": (2.0, 0.0),
    "var_samp": (2.5, None),
    "stddev_pop": (math.sqrt(2.0), 0.0),
    "stddev_samp": (math.sqrt(2.5), None),
}


def probe_sqlite_nonstandard_statistics(conn: Any) -> frozenset[str]:
    """Statistical aggregates the SQLite connection defines that do not match
    the exact definition (see ``_SQLITE_STATISTICS_EXPECTED``).

    Behavioural: each of ``var_pop``/``var_samp``/``stddev_pop``/``stddev_samp``
    is evaluated on known values through the connection. SQLite has none built
    in, so a function that exists was supplied by the connection (sqlean's
    stats, a loaded extension, an application-defined function) and is what the
    ORIGINAL view would call; if it differs from the definition matview
    maintains the materialized view could not equal the view. A function that
    errors is treated as non-standard. An unconditional probe: it does not depend
    on the driver class, proxies, or on a pinned engine version.
    """
    try:
        cur = conn.cursor()
    except Exception:
        return frozenset()
    found = set()
    try:
        for name, (expected_many, expected_one) in _SQLITE_STATISTICS_EXPECTED.items():
            try:
                cur.execute(f"SELECT {name}(x) FROM (SELECT 1 AS x UNION ALL SELECT 2 UNION ALL SELECT 3 "
                            "UNION ALL SELECT 4 UNION ALL SELECT 5)")
                many = cur.fetchone()
                cur.execute(f"SELECT {name}(x) FROM (SELECT 1 AS x)")
                one = cur.fetchone()
                cur.execute(f"SELECT {name}(x) FROM (SELECT 1 AS x) WHERE 0")
                none = cur.fetchone()
            except Exception as exc:
                if "no such function" in str(exc).lower():
                    continue
                found.add(name)
                continue
            if not all(isinstance(r, (tuple, list)) and len(r) == 1 for r in (many, one)):
                # Not a SQL result (a stub/mock connection): nothing to probe.
                return frozenset()
            # An aggregate over no row still yields one row; no row at all means
            # a scalar function of that name.
            got = (many[0], one[0], none[0] if isinstance(none, (tuple, list)) and len(none) == 1 else "<no row>")
            if got != (expected_many, expected_one, None) or not all(
                isinstance(v, float) or v is None for v in got
            ):
                found.add(name)
    finally:
        close = getattr(cur, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return frozenset(found)


def with_probed_sqlite_statistics(conn: Any, version: EngineVersion) -> EngineVersion:
    """``version`` with ``sqlite_nonstandard_statistics`` filled in from a probe
    of ``conn`` (SQLite only; other dialects are returned unchanged)."""
    if version.dialect != Dialect.SQLITE:
        return version
    probed = probe_sqlite_nonstandard_statistics(conn) | version.sqlite_nonstandard_statistics
    if probed == version.sqlite_nonstandard_statistics:
        return version
    return dataclasses.replace(version, sqlite_nonstandard_statistics=probed)


def _read_mariadb_div_precision_increment(cur: Any) -> int:
    """Read MariaDB session precision increment, falling back to its default."""
    try:
        cur.execute("SELECT @@SESSION.div_precision_increment;")
        row = cur.fetchone()
        if row and row[0] is not None:
            value = int(row[0])
            if 0 <= value <= 30:
                return value
    except Exception:
        pass
    return 4



class MaintenanceStrategy(str, Enum):
    ALGEBRAIC_DELTA = "algebraic_delta"
    GROUP_SCOPED_DELTA = "group_scoped_delta"
    ATOMIC_QUERY_SYNC = "atomic_query_sync"


class AggFunc(str, Enum):
    COUNT_STAR = "COUNT(*)"
    COUNT_COL = "COUNT"
    SUM = "SUM"
    TOTAL = "TOTAL"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    GROUP_CONCAT = "GROUP_CONCAT"
    STRING_AGG = "STRING_AGG"
    JSON_GROUP_ARRAY = "JSON_GROUP_ARRAY"
    JSON_GROUP_OBJECT = "JSON_GROUP_OBJECT"
    ARRAY_AGG = "ARRAY_AGG"
    VAR_POP = "VAR_POP"
    VAR_SAMP = "VAR_SAMP"
    STDDEV_POP = "STDDEV_POP"
    STDDEV_SAMP = "STDDEV_SAMP"
    BIT_AND = "BIT_AND"
    BIT_OR = "BIT_OR"
    BIT_XOR = "BIT_XOR"
    BOOL_AND = "BOOL_AND"
    BOOL_OR = "BOOL_OR"


class LimitationSeverity(str, Enum):
    WARNING = "warning"
    BLOCKER = "blocker"


@dataclass(frozen=True)
class TransparencyLimitation:
    code: str
    severity: LimitationSeverity
    dialect: Dialect
    affected_object: str
    description: str
    mitigation: str


# Limitations whose override changes the result contract (best effort instead
# of exact). Only an explicit per-code acceptance (``accept_limitations`` /
# ``--accept-limitation CODE``) overrides them; the global
# ``ignore_limitations`` does not.
PER_CODE_OPT_IN_LIMITATIONS = frozenset({
    "ORDER_SENSITIVE_AGGREGATE_NOT_REPRODUCIBLE",
    # Element order of string/array/JSON aggregates without a total ORDER BY
    # (accepting it keeps the group-scoped recompute, announced as relaxed).
    "AGGREGATE_ORDER_NOT_REPRODUCIBLE",
    "ORDER_SENSITIVE_AGGREGATE_NO_RACE_FREE_RECOMPUTE",
    "MYSQL_GROUP_RECOMPUTE_REQUIRES_GROUP_INDEX",
})


class EngineLimitationError(RuntimeError):
    """Raised when a materialized view encounters engine transparency limitations and ignore_limitations=False."""

    def __init__(self, view_name: str, dialect: Dialect, limitations: Sequence[TransparencyLimitation]) -> None:
        self.view_name = view_name
        self.dialect = dialect
        self.limitations = list(limitations)
        bullet_lines = "\n".join(
            f"  - [{lim.code}] {lim.description} Mitigation: {lim.mitigation}"
            for lim in self.limitations
        )
        msg = (
            f"Refusing to create materialized view '{view_name}' on engine '{dialect.value}' "
            f"because full transparency cannot be guaranteed for all cases:\n{bullet_lines}\n"
            + self._override_hint()
        )
        super().__init__(msg)

    def _override_hint(self) -> str:
        per_code = sorted({l.code for l in self.limitations if l.code in PER_CODE_OPT_IN_LIMITATIONS})
        other = sorted({l.code for l in self.limitations if l.code not in PER_CODE_OPT_IN_LIMITATIONS})
        parts = []
        if per_code:
            parts.append(
                "Accept " + ", ".join(per_code) + " explicitly with `--accept-limitation CODE` (CLI) or "
                "`accept_limitations=[...]` (API) to install the view with BEST-EFFORT results "
                "(`ignore_limitations` does not cover these codes)."
            )
        if other:
            parts.append(
                "Pass `--ignore-limitations` (CLI) or `ignore_limitations=True` (API) to override "
                + ", ".join(other) + " and proceed anyway."
            )
        return " ".join(parts)


class EngineLimitationWarning(UserWarning):
    """Emitted when proceeding with ignore_limitations=True despite engine transparency limitations."""


class StrictCompilationError(ValueError):
    """Raised when strict=True is enabled and a view cannot be compiled with Tier 1 ALGEBRAIC_DELTA."""


class UnsupportedMaterializedViewError(ValueError):
    """Raised when an unsupported materialized view operation or configuration is attempted."""


_SQL_RESERVED_KEYWORDS = {
    "ABORT", "ADD", "ALL", "ALTER", "ANALYSE", "ANALYZE", "AND", "ANY", "AS", "ASC",
    "ATTACH", "BEGIN", "BETWEEN", "BIGINT", "BINARY", "BIT", "BLOB", "BOOL", "BOOLEAN",
    "BY", "CASE", "CAST", "CHAR", "CHARACTER", "CHECK", "CLOB", "COLLATE", "COLUMN",
    "COMMIT", "CONDITION", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_DATE", "CURRENT_TIME",
    "CURRENT_TIMESTAMP", "CURRENT_USER", "DATABASE", "DATE", "DATETIME", "DECIMAL",
    "DEFAULT", "DELETE", "DESC", "DESCRIBE", "DETACH", "DISTINCT", "DIV", "DO", "DOUBLE",
    "DROP", "ELSE", "ELSEIF", "END", "EXCEPT", "EXISTS", "EXPLAIN", "FALSE", "FETCH",
    "FILTER", "FLOAT", "FOR", "FOREIGN", "FROM", "FULL", "FUNCTION", "GRANT", "GROUP",
    "HAVING", "IF", "IN", "INDEX", "INNER", "INOUT", "INSERT", "INT", "INTEGER",
    "INTERSECT", "INTERVAL", "INTO", "IS", "JOIN", "KEY", "KILL", "LEADING", "LEFT",
    "LIKE", "LIMIT", "LOAD", "LOCALTIME", "LOCALTIMESTAMP", "LOCK", "MATCH", "MEDIUMINT",
    "MOD", "NATURAL", "NOT", "NULL", "NUMERIC", "OFFSET", "ON", "ONLY", "OR", "ORDER",
    "OUT", "OUTER", "OVER", "PARTITION", "PRAGMA", "PRECISION", "PRIMARY", "PROCEDURE",
    "RANGE", "REAL", "RECURSIVE", "REFERENCES", "REGEXP", "RELEASE", "RENAME", "REPLACE",
    "RETURNING", "REVOKE", "RIGHT", "RLIKE", "ROLLBACK", "ROW", "ROWS", "SAVEPOINT",
    "SCHEMA", "SELECT", "SEPARATOR", "SESSION_USER", "SET", "SHOW", "SMALLINT", "SOME",
    "STATUS", "SYMMETRIC", "TABLE", "TEXT", "THEN", "TIME", "TIMESTAMP", "TINYINT", "TO",
    "TRAILING", "TRIGGER", "TRUE", "UNION", "UNIQUE", "UNLOCK", "UPDATE", "USAGE", "USE",
    "USER", "USING", "VACUUM", "VALUES", "VARCHAR", "VARIADIC", "VIEW", "WHEN", "WHERE",
    "WINDOW", "WITH", "XOR", "YEAR",
}

_IDENT_TOKEN_PATTERN = r'(?:"[^"]*(?:""[^"]*)*"|`[^`]*(?:``[^`]*)*`|\[[^\]]*\]|[A-Za-z_][A-Za-z0-9_]*)'
_SIMPLE_IDENT_RE = re.compile(
    rf"^(?:{_IDENT_TOKEN_PATTERN}\.)?{_IDENT_TOKEN_PATTERN}$"
)

def _sqlglot_dialect(
    dialect: Union[Dialect, EngineSubdialect, EngineVersion, str, None] = Dialect.SQLITE,
) -> str:
    if dialect is None:
        return "sqlite"
    if isinstance(dialect, EngineVersion):
        dialect = dialect.subdialect
    if isinstance(dialect, (Dialect, EngineSubdialect)):
        dialect_val = dialect.value
    elif isinstance(dialect, str):
        dialect_val = dialect.lower().strip()
    else:
        dialect_val = str(dialect).lower().strip()

    if dialect_val in ("postgres", "postgresql", "pgsql"):
        return "postgres"
    if dialect_val in ("mysql", "mariadb"):
        # Note: sqlglot parses MariaDB using its 'mysql' dialect.
        try:
            from sqlglot import Dialects
            if "mariadb" in Dialects.__members__:
                return "mariadb"
        except Exception:
            pass
        return "mysql"
    if dialect_val == "sqlite":
        return "sqlite"
    return dialect_val
