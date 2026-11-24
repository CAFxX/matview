# SQLite STRICT tables, rowids and minimal columns (design note)

Directive: support and recommend STRICT source tables, make diagnostics and code
generation STRICT-aware, drop rowids and columns matview does not need.
The project is unreleased: no migration or compatibility work anywhere below.

## Decisions

1. **Source tables (PR 1, this note).** STRICT sources are supported and
   recommended (README "SQLite STRICT tables"). Exactness never depends on
   STRICT: plain tables stay supported with the existing affinity-safe
   strategies. STRICT only lets the planner *drop* guards. Implemented now:
   NOT NULL integer PK on a STRICT table is a trusted watermark column (already
   in `_resolve_pk_range_column`). Everything else that is provable but not
   trivial is listed in `TODO.md` (hidden SUM/AVG non-integer counter for STRICT
   INTEGER arguments; an audit of unchanged-row and comparison guards).
2. **Diagnostics (PR 2).** STRICT-awareness is a *note* (INFO) in
   `plan.strategy_diagnostics` / CLI, never a warning, because a recommendation
   does not change a guarantee. A note names the table, the columns whose type
   is only an affinity, the concrete benefit, the `CREATE TABLE ... STRICT` hint
   and the 12-step rebuild requirement. `check_schema_drift()` reports
   STRICT -> plain as `invalid`. STRICT-ness comes from `PRAGMA table_list`
   (authoritative, 3.37+) with the `CREATE TABLE` text as the offline source.
3. **Generated tables (PR 3).** Per table, never per view guess:
   - `WITHOUT ROWID` by default for every matview-owned table with a NOT NULL
     primary key that is not already the rowid, no AUTOINCREMENT and narrow rows
     (estimate <= 128 bytes). Audit: no generated SQL reads a matview-owned rowid
     (the `rowid` references in `maintenance/` are about *base* tables). Measured:
     -9% to -26% statement time on aggregate/MIN-MAX/DISTINCT views, noise on
     join/projection views, break-even near 150 payload bytes (README).
   - `STRICT` opt-in (`sqlite_strict_backing`): only when every column type is
     *provably fixed* by the plan (BLOB -> ANY, counters/COUNT -> INTEGER;
     never accumulators: a REAL/INTEGER source column may hold TEXT that SUM/AVG folds in). A wrong-typed value raises inside a maintenance trigger
     and aborts the user's DML, so a table with a copied base column stays plain
     (announced). Off by default: guard rail only, +0..5% write time on top of
     WITHOUT ROWID (measured), no result change; needs SQLite 3.37 (announced fallback).
     WITHOUT ROWID can be disabled (`sqlite_without_rowid=False`) because the width
     estimate cannot see value sizes.
4. **Minimal columns (PR 4).** Every matview-owned column/index/implicit key on
   every engine is justified (needed for exactness or O(1) maintenance) or removed.

## Exclusions

- STRICT on the *materialized* result is never required for correctness.
- No ALTER-to-STRICT: SQLite cannot do it; matview does not rebuild user tables.
- libSQL/sqlean/SQLCipher/APSW: STRICT is used only if the library reports
  >= 3.37; otherwise silently-unavailable features are announced, not assumed.
