# Design: server-derived result types

Status: proposal (no code yet). Lane: "Server-derived result types" in `TODO.md`.

## Problem

Every backing column needs a declared type that makes the reader view return
exactly what the live view returns (value, Python type, protocol type code).
Today the planner derives types statically from the SQL text
(`dialects/base.py`: `_infer_expr_storage_type` and friends). Static
inference is right for modelled shapes and guesses elsewhere: user functions
(stored as TEXT until #261), recursive CTEs and set operations (#255 float
follow-up), MariaDB `UNSIGNED` results of `MAX`/`MIN` (the "UNSIGNED MAX
gap"), compound expressions mixing aggregates and casts. A wrong guess is a
silent wrong result (`'10'` vs `10`, `DOUBLE` vs `DECIMAL`), the worst
failure class under AGENTS.md.

Two server-side mechanisms already exist, each for one purpose:

* `matview.native_functions.probe_result_types` (#261): at registration the
  manager asks the server for the original query's output types (PostgreSQL:
  `format_type(atttypid, atttypmod)` of a temporary view inside a savepoint;
  MySQL/MariaDB: result-set metadata of `SELECT * FROM (<q>) LIMIT 0`, because
  `CREATE TABLE ... AS SELECT` widens `INT` to `BIGINT`). Used only for items
  that call functions sqlglot does not model, and their CTE columns.
* PostgreSQL install-time key probes (`pg_statement._pg_probe_queries`,
  20e7151): a `DO` block checks grouping/MIN/MAX key types and refuses
  unsupported ones. It validates; it does not type columns.

## Proposal

One **type oracle** per registration, owned by the manager, consumed by the
planner through the `ViewSpec` (already `native_result_types`):

1. **When.** At registration and on every stored-plan recompile (sync,
   repair, drift check): the same probe, so signatures stay stable. Offline
   generation has no server: static inference, as today, plus the existing
   refusal (`CAST(... AS <type>)` remedy) where static inference cannot be
   exact.
2. **What.** Output column types of the ORIGINAL query (the reader's
   contract), plus on demand the types of named sub-expressions the
   maintenance SQL stores separately (group keys, MIN/MAX arguments,
   aggregate state columns: `SUM` count/sum pairs, CTE columns). Each request
   is a `SELECT <exprs> FROM (<q>) ... LIMIT 0` / `WHERE false` probe; one
   round trip per registration where possible.
3. **How, per engine.**
   * PostgreSQL: temporary view + `pg_attribute` (keeps typmods: `numeric(10,2)`,
     `varchar(8)`); savepoint when inside the manager's transaction.
   * MySQL/MariaDB: result-set metadata (type code, length, decimals,
     `UNSIGNED`/binary flags) mapped to DDL (`_mysql_type_from_metadata`);
     never `CREATE TABLE ... AS SELECT`.
   * SQLite: no probe; columns without affinity (`BLOB`) keep the storage
     class, as today.
4. **Precedence.** Server type > static inference for any column whose static
   type is a heuristic (an explicit allow-list of "exact" static rules stays
   authoritative to keep plans reproducible offline). A disagreement between
   an exact static rule and the server is a bug: surface it as a
   registration error in tests (assert mode), never silently.
5. **Failure.** A failed probe (permissions, syntax the engine rejects only
   in a subquery) falls back to static inference and, where that is a guess,
   to the CAST refusal. Never a silent guess.
6. **Install-time key probes** stay (they guard keys the planner must hash or
   compare), but read their expected types from the oracle instead of
   re-deriving them.

## Consumers (first)

* #261 unmodelled functions (already on it; generalise the call site).
* #255 float follow-up: recursive CTE and set-operation column types.
* MariaDB `UNSIGNED` `MAX`/`MIN` results (protocol type and range).
* MySQL `GROUP_CONCAT` / `JSON_ARRAYAGG` protocol metadata (TODO under L1).

## Tests

A table-driven test per engine: each shape registered natively, reader vs
live compared on values, Python types and type codes/OIDs (the existing
oracle), plus a mutation that disables the probe and must fail. Cost: one
extra round trip per registration (no data read).
