# Research: Asynchronous, Database-Resident Materialized-View Maintenance

**Status:** exploratory research and prototype findings; no production implementation or API has been committed.  
**Primary requirement:** after setup, capture, scheduling, maintenance, recovery and publication must be self-contained in the database engine. No Python runtime, application-managed worker, external broker, CDC connector, cron job or external database service may be needed to keep a view maintained. Installation-time tooling may generate SQL and install native database modules.  
**Relationship to existing behavior:** synchronous maintenance remains the default. Async maintenance is an explicit opt-in with an explicit staleness/consistency contract. The long-term architecture must not depend on Python-controlled \`bulk_changes()\`; that mechanism is not an async foundation and is expected to be removed as the database-resident paths mature.

## 1. Central architectural principle

**The maintenance routine is the portable unit of execution. Scheduling and change capture are replaceable adapters around it.**

A single logical maintenance operation accepts a bounded work budget, claims pending work, executes the installed per-view maintenance program, publishes the resulting state, acknowledges the work and returns structured progress. The scheduler does not understand SQL view shapes or maintenance algorithms; it only invokes the common coordinator.

Illustrative logical interface (names and signatures are not API decisions):

\`\`\`text
matview_maintain(view_name, work_budget) -> result
matview_maintain_all(work_budget)       -> result
matview_maintenance_status(view_name)   -> status
\`\`\`

Possible result fields include \`work_units_processed\`, \`work_remaining\`, \`oldest_pending_age\`, \`last_success_at\`, \`last_error\`, \`published_generation\` and the strongest freshness frontier currently available.

This separates four dimensions that must not be conflated:

1. **Capture:** no capture, dirty scopes, row changes, compressed deltas, native engine change streams, or caller-supplied changes.
2. **Fold:** algebraic batch delta, affected-scope recomputation, full original-query refresh, or an adaptive combination.
3. **Invocation:** explicit SQL call, cooperative invocation by an application issuing SQL, database event scheduler, or in-server background worker.
4. **Publication:** one transactional batch, a generation switch, or another explicitly specified atomic publication unit.

The SQL program is generated at installation from the compiler's plan. It must be versioned and bound to the registered view definition. The early prototype stores ordered SQL steps in a table; that is a proof of executor/program separation, not yet a recommended production bytecode or metadata format. A production representation needs validation, bounded parameters, stable program identity, cleanup on replacement/drop, and clear privileges.

## 2. SQLite: native UDF execution is feasible

A minimal native C extension was compiled and exercised against SQLite **3.46.1**. Its UDF obtains the invoking connection with \`sqlite3_context_db_handle()\`, runs several SQL statements through that same connection, and is registered with \`SQLITE_DIRECTONLY\`.

A second prototype exposes a generic runner:

\`\`\`sql
SELECT matview_maintain('sales', 100);
\`\`\`

The UDF reads an ordered per-view SQL program from a prototype table, obtains a SQLite write reservation before reading pending work, and runs a bounded number of steps on the originating connection.

### 2.1 Observed prototype results

- Nested SQL execution against the invoking connection works.
- DML issued by the UDF can update backing tables and fire ordinary SQL triggers.
- A savepoint created inside the UDF can be released without committing an existing caller transaction.
- An injected uniqueness violation after an earlier successful insert is rolled back in its entirety using \`ROLLBACK TO\` plus \`RELEASE\`, both in autocommit mode and inside a caller-owned transaction.
- The caller's transaction remains open after success or after the UDF's internal failure path.
- Successful maintenance inside a caller-owned transaction is itself rolled back if the caller later rolls back the outer transaction.
- \`SQLITE_DIRECTONLY\` rejects invocation through a stored view.
- A competing connection holding a write transaction causes the UDF call to fail safely with \`database is locked\`.
- The dirty-group test program correctly handles repeated changes to one group, OLD+NEW group moves, bounded batches, deleting the last row in a group, retry after injected failure, nested savepoints and malformed/missing programs.

**A savepoint is essential.** Merely executing a list of nested SQL statements from a UDF is not enough: in autocommit mode, earlier nested statements can persist if a later statement fails. The runner must put the complete bounded operation inside one savepoint and roll it back on any error.

### 2.2 SQLite execution constraints

1. SQLite does not persist UDF registrations in the database. The native extension must be loaded/registered on each connection that invokes it, or statically registered by the host. This is a connection integration requirement, not a separate maintenance worker.
2. SQLite has no autonomous server scheduler. The UDF supplies **cooperative/manual in-database execution** when a host issues SQL. It cannot continue making progress with no process opening the database. Do not claim autonomous scheduling for SQLite.
3. Invoke the side-effecting UDF as a standalone top-level command. Do not embed it in a view, trigger, source-table scan or arbitrary query expression. \`SQLITE_DIRECTONLY\` blocks schema-object invocation but does not prevent every unsafe top-level SQL construction.
4. \`sqlite3_stmt_readonly()\` reports the outer \`SELECT\` as read-only even when the UDF indirectly modifies the database. Callers and adapters must treat this as a write operation.
5. \`SQLITE_DIRECTONLY\` was added in SQLite 3.30.0. The async UDF feature should require SQLite >= 3.30 without necessarily raising the minimum SQLite version for synchronous maintenance.
6. The UDF must be transaction-neutral: create a uniquely named savepoint, execute one bounded unit, release it on success, and roll back to it on failure. It must never issue top-level \`COMMIT\` or \`ROLLBACK\`.

The regular SQLite commit hook is not a safe place to execute SQL on the connection whose commit invoked it: it is non-reentrant. The WAL hook runs after commit and after the write lock is released, but still runs inline in the source call and therefore adds maintenance work to source latency. Spawning a thread from the extension would reintroduce a host-process worker and is outside the design.

References: [sqlite3_context_db_handle](https://www.sqlite.org/c3ref/context_db_handle.html), [sqlite3_exec](https://www.sqlite.org/c3ref/exec.html), [function flags and SQLITE_DIRECTONLY](https://www.sqlite.org/c3ref/c_deterministic.html), [sqlite3_stmt_readonly](https://www.sqlite.org/c3ref/stmt_readonly.html), [SQLite 3.30.0 release notes](https://www.sqlite.org/releaselog/3_30_0.html), [commit hook](https://www.sqlite.org/c3ref/commit_hook.html), [WAL hook](https://www.sqlite.org/c3ref/wal_hook.html).

## 3. Change-capture alternatives

Append-only row-delta logs are **one candidate**, not the assumed design. Capture and fold strategy should be selected independently for each view/engine/workload.

### A. No capture: scheduled or explicit full refresh

Source writes incur no view-specific capture work. The database routine executes the original query and replaces the stored result, publishing it atomically (or by a generation switch).

- Strengths: minimum source-write overhead; simple correctness model; universal fallback for complicated Tier 3 query shapes.
- Weaknesses: O(source data) per refresh; potentially large refresh transactions; repeated work if refreshes are too frequent.
- Best candidates: complex views, low change rates, or views where incremental capture/folding costs exceed the cost of recomputing.

### B. Whole-view dirty flag

A source trigger inserts one unique marker for the view. Repeated changes coalesce until the marker is consumed; the worker runs the original query.

- Strengths: minimal capture payload and simple correctness reasoning.
- Weaknesses: all writers for a view contend on one hot marker; still one trigger invocation per source row on row-trigger engines.
- Useful as a cheap fallback when narrower affected-scope analysis is impossible.

### C. Dirty-scope markers

Capture affected source keys, output keys, groups or window partitions, with duplicate markers collapsed by a unique key. The worker recomputes the affected scopes from the original query and replaces their backing rows.

- Best initial target: single-table GROUP BY, projections, DISTINCT, and windows with a clear PARTITION BY key.
- An UPDATE that moves a row from scope A to B must dirty both OLD and NEW scopes.
- A delete that removes the last contributing row must remove the output group.
- Joins and changes that can affect many output keys should conservatively broaden to a larger scope or a whole-view dirty flag until precise scope propagation is proven.
- Worker/trigger races must be designed so a source write during a refresh cannot have its dirty marker accidentally consumed by the worker.

This is attractive when many source changes collapse to a small number of affected groups. It is not automatically cheap: row triggers still execute once per changed source row on SQLite/MySQL/MariaDB, and a high-cardinality dirty-key table can approach row-log volume.

### D. Append-only row-change log

Triggers append OLD/NEW row images or signed changes. The worker coalesces them and performs a set-based delta fold.

- Strengths: retains enough information for algebraic updates and many joins; avoids one hot dirty-marker row; can collapse repeated changes during batch processing.
- Weaknesses: row-image/log volume, index and cleanup costs, more source-write I/O, and a more complex batch algebra.
- Important correctness trap: never use an ordinary auto-increment/event sequence as a global committed-progress watermark on multi-writer engines. Transaction A can allocate a lower ID and commit after transaction B allocates a higher ID. A consumer that advances to the higher ID can permanently skip the late commit. Claim visible committed records and acknowledge them in the same transaction as the backing fold, or use a true engine commit-order position.

### E. Pending algebraic accumulators

Instead of storing every event, source triggers fold changes into compact pending aggregate state keyed by output group. The worker applies one net delta per group.

- Strengths: very cheap worker; small pending state; avoids storing every OLD/NEW image.
- Weaknesses: source transactions still update shared group state; hot groups can become lock-contention points; exact floating/order-sensitive behavior must retain existing guarantee-aware fallbacks.
- Best candidates: simple exactly foldable aggregates (for example integer/decimal SUM and COUNT), subject to native proof of exact equivalence.

### F. Database-native change streams

An external CDC connector or consumer is excluded. A consumer that actually runs inside the database server can be evaluated as an optional engine-specific backend.

- **PostgreSQL:** an in-server background worker could consume logical decoding using the server APIs or slot functions, then apply changes in database transactions. Logical decoding provides commit-ordered changes and omits rolled-back transactions, but slots can replay recent changes after a crash because durable position is checkpointed. The consumer needs idempotency/deduplication, WAL-retention monitoring, slot lifecycle and privileges; \`wal_level=logical\`, server extension installation and a per-database worker/supervisor add operational cost. Treat it as an advanced alternative, not the baseline.
- **MySQL/MariaDB:** the SQL Event Scheduler can invoke stored procedures but does not expose a general SQL interface for consuming the server binlog as a local change stream. A server plugin/native extension might be possible, but the required maintenance APIs and deployment burden need independent research. Do not assume external binlog consumers are acceptable.
- **SQLite:** triggers and connection-local native hooks are the practical self-contained capture choices. The session extension/pre-update hook are possible investigation items, but are not a database-wide autonomous scheduler.

PostgreSQL references: [logical decoding concepts](https://www.postgresql.org/docs/current/logicaldecoding-explanation.html), [logical decoding output plugins](https://www.postgresql.org/docs/current/logicaldecoding-output-plugin.html), [replication-slot functions](https://www.postgresql.org/docs/current/functions-admin.html).

## 4. Maintenance algorithms are independent from capture

The common maintenance routine should select or execute a strategy that matches the installed plan and current batch:

1. **Algebraic batch fold:** aggregate signed deltas by output key, then perform set-based writes to backing/auxiliary structures.
2. **Scoped recomputation:** recompute dirty rows/groups/partitions from the original source query and replace those output scopes.
3. **Full original-query refresh:** recompute the original query unmodified and publish a new generation/result.
4. **Adaptive choice:** when the capture representation supports alternatives, choose based on batch volume, distinct affected scopes, source cardinality and measured cost.

For joins, the batch compiler must account for changes on both sides in the same batch. For example, a bilinear join requires the cross term:
\`Δ(R ⋈ S) = ΔR ⋈ S₀ + R₀ ⋈ ΔS + ΔR ⋈ ΔS\`.
A per-row trigger loop is not an adequate substitute for a set-based batch compiler.

The compiler should describe: accepted view shape/engine, capture requirements, affected work unit, maintenance algorithm, generated SQL/program steps, exactness guarantee, cost class, concurrency assumptions, and publication contract. Scheduler implementations should not contain query-specific logic.

Research background: [DBSP: Automatic Incremental View Maintenance for Rich Query Languages](https://arxiv.org/abs/2203.16684) (VLDB 2023); [Differential dataflow](https://www.cidrdb.org/cidr2013/Papers/CIDR13_Paper111.pdf); [Incremental maintenance of views with duplicates](https://doi.org/10.1145/223784.223785).

## 5. Database-resident execution by engine

| Engine | Common routine | Autonomous invocation after setup |
|---|---|---|
| SQLite | Native C UDF, called as a top-level SQL command | Not available without a host issuing SQL; cooperative/manual execution is the honest capability |
| PostgreSQL | Transaction-neutral SQL-callable function/program, optionally exposed through a C extension | In-server background worker invokes it through SPI; a supervisor/worker lifecycle must account for every database containing async views |
| MySQL | Stored procedure | Event Scheduler invokes a coordinator procedure |
| MariaDB | Stored procedure | Event Scheduler invokes a coordinator procedure |

### PostgreSQL

A background worker is an in-server process, not an external application worker. It can connect to a database and execute SQL through SPI. The worker should own the outer transaction for each bounded batch; the SQL-callable maintenance function remains transaction-neutral. Startup registration commonly requires a server extension and \`shared_preload_libraries\`. A worker's database connection is database-specific, so multi-database installations need a supervisor or worker-per-database arrangement.

### MySQL and MariaDB

Use **stored procedures**, not stored functions, for the generic executor. MySQL stored functions/triggers have material restrictions on dynamic SQL and side effects; stored procedures permit PREPARE/EXECUTE. But transaction ownership is delicate:

- The core procedure must not issue \`START TRANSACTION\`, \`COMMIT\` or \`ROLLBACK\` when called by arbitrary application SQL.
- \`START TRANSACTION\` inside a procedure can implicitly commit a transaction already owned by the caller.
- The procedure should operate within the supplied transaction and use a savepoint for local error handling.
- A scheduler event executes on its own connection; its **wrapper** can own the outer bounded transaction, call the neutral core procedure, then commit or roll back.
- Manual callers explicitly begin/commit their transaction.
- MySQL and MariaDB differ in transaction-state introspection; do not make the shared procedure depend on a non-portable \`in_transaction\` variable.
- Prepared statements cannot directly reference stored-program local variables/parameters. Prototype carefully scoped session variables versus installation-time generated static per-view procedures; test statement cleanup and session-state leakage.
- Use a single database-resident coordinator event where practical, not one scheduler event per view. Prevent overlapping invocations with durable claims/locking.

References: [MySQL stored-program restrictions](https://dev.mysql.com/doc/refman/8.4/en/stored-program-restrictions.html), [MySQL Event Scheduler](https://dev.mysql.com/doc/refman/8.4/en/events-overview.html), [MySQL transactions](https://dev.mysql.com/doc/refman/8.4/en/innodb-autocommit-commit-rollback.html), [PostgreSQL SPI](https://www.postgresql.org/docs/current/spi.html), [PostgreSQL background workers](https://www.postgresql.org/docs/current/bgworker.html).

## 6. Atomicity, freshness and progress are separate guarantees

An atomic batch does not automatically make the complete async view correspond to one source snapshot. If group A is recomputed at T1 and group B at T2, the backing table can contain individually correct groups evaluated at different source states.

Define these as distinct contracts:

- **Eventual scope consistency:** every completed batch is atomic; every processed scope is exact for the source state it reads; unprocessed scopes may remain stale; repeated processing converges. This is the natural dirty-scope mode.
- **Committed-prefix consistency:** the complete published view equals the original query at a well-defined committed source frontier. This needs commit-ordered capture/application or a consistent snapshot plus a replayable log, or a full generation rebuild. A dirty-scope queue by itself does not provide it.

A freshness barrier must specify what it waits for: draining all currently known dirty scopes is not the same as publishing through a requested commit frontier. Expose staleness in metadata/diagnostics; never silently make the existing synchronous mode stale.

For large full refreshes, generation-based publication is worth comparing: build a new backing generation while readers continue using the old one, then atomically switch an active-generation pointer. Costs include temporary storage, generation cleanup and reader indirection. For narrower batches, transactionally update backing state and acknowledge work in the same transaction.

## 7. Shared transaction and work-claim contract

One invocation should process one bounded, independently atomic unit:

1. Begin/participate in a transaction and establish a savepoint as appropriate.
2. Claim pending work without losing it to concurrent workers/writers.
3. Execute the generated maintenance program.
4. Publish backing and auxiliary changes.
5. Acknowledge the claimed work and advance durable progress.
6. Commit only if the outer execution wrapper owns the transaction.

Capture must commit/roll back with the source write. Applying backing changes and acknowledging work must commit together. A crash before commit leaves work pending; a crash after commit leaves the result published and the work acknowledged. Ownership tokens/generation IDs may be needed to prevent an old worker from mutating a replaced/re-registered view.

SQLite's single-writer serialization can simplify a claim-and-delete queue: obtain a write reservation *before* reading pending work, then claim/process/acknowledge selected work in the same transaction. Do not generalize this to PostgreSQL/MySQL multi-writer concurrency. For those engines, prove claim semantics using independent connections and barrier-synchronized races.

## 8. Preliminary benchmark evidence (SQLite 3.46.1 only)

Two local microbenchmarks compare alternative capture/fold choices. They are directional results, not a cross-engine cost model. Setup: file-backed SQLite in WAL mode, \`synchronous=NORMAL\`; insert workload is one \`INSERT INTO ... SELECT\` statement; medians over five trials.

### 8.1 Capture-only: 50,000 inserted source rows

| Groups in source | Capture strategy | Median source insert | Extra capture-table pages | Captured records |
|---:|---|---:|---:|---:|
| 100 | None | 13.85 ms | 0 | 0 |
| 100 | Whole-view dirty flag | 41.12 ms | 4 KiB | 1 |
| 100 | Dirty-group markers | 34.69 ms | 4 KiB | 100 |
| 100 | Append-only row log | 39.33 ms | 880 KiB | 50,000 |
| 100 | Pending SUM/COUNT | 42.89 ms | 4 KiB | 100 |
| 25,000 | None | 14.86 ms | 0 | 0 |
| 25,000 | Whole-view dirty flag | 41.76 ms | 4 KiB | 1 |
| 25,000 | Dirty-group markers | 41.66 ms | 200 KiB | 25,000 |
| 25,000 | Append-only row log | 41.25 ms | 932 KiB | 50,000 |
| 25,000 | Pending SUM/COUNT | 45.42 ms | 336 KiB | 25,000 |

The results show that capture is not free even when markers coalesce: row-trigger work still runs per source row. Dirty markers save space when group cardinality is low, but high-cardinality dirty keys can approach row-log scale. Pending accumulators were most expensive on this simple source-write benchmark because every row updated shared group state.

### 8.2 Capture plus maintenance: insert 10,000 rows affecting 1% of output groups

Starting state: 50,000 base rows and a populated grouped SUM/COUNT view.

| Groups | Strategy | Source insert incl. capture | Worker batch | Sum of phases |
|---:|---|---:|---:|---:|
| 100 (1 group affected) | Full refresh | 8.61 ms | 20.93 ms | 29.54 ms |
| 100 (1 group affected) | Dirty-group recompute | 12.76 ms | 6.86 ms | 19.62 ms |
| 100 (1 group affected) | Row-log fold | 18.14 ms | 2.83 ms | 20.97 ms |
| 100 (1 group affected) | Pending SUM/COUNT | 31.99 ms | 0.12 ms | 32.11 ms |
| 10,000 (100 groups affected) | Full refresh | 11.02 ms | 27.02 ms | 38.04 ms |
| 10,000 (100 groups affected) | Dirty-group recompute | 13.63 ms | 9.09 ms | 22.72 ms |
| 10,000 (100 groups affected) | Row-log fold | 14.28 ms | 2.86 ms | 17.14 ms |
| 10,000 (100 groups affected) | Pending SUM/COUNT | 15.88 ms | 0.17 ms | 16.05 ms |

In this insert-only SUM/COUNT workload, dirty recomputation was best when very few groups were affected, while row-log folding and pending accumulators were competitive/better when more groups were affected. Pending accumulators made the worker extremely cheap but shifted more work into source transactions. This does **not** establish that one strategy is universally fastest.

Limitations: SQLite 3.46.1 only; integer SUM/COUNT; insert-only; one bulk insert statement; no updates/deletes, OLD/NEW row-image cost, floating-point exactness, joins, concurrent sessions, process-kill recovery, or cross-engine tests in these benchmark numbers. Use these as hypotheses for broader native benchmarks, not production planner thresholds. Reproduction scripts/results were kept in a separate prototype package during this study; the Linux-built shared objects are not proposed for committing.

## 9. Recommended research/implementation sequence

1. **Specify the contract first:** async mode options, eventual-scope versus committed-prefix consistency, freshness barriers, error visibility, retention/cleanup, and interaction with replacement/drop/backfill.
2. **Stabilize the maintenance-program abstraction:** compiler output declares work unit, capture needs, fold strategy, cost class, correctness guarantee, transaction assumptions, and publication mode. Version it and bind it to the view signature.
3. **SQLite reference prototype:** retain the top-level native UDF runner and dirty-scope grouped aggregate. Extend tests to WAL and rollback-journal modes, multiple independent writers, worker/write races, busy retry policy, process termination/crash recovery, composite/nullable/collated keys, UPDATE/DELETE and repeated invocation. Decide how extension loading is integrated with supported host APIs.
4. **Benchmark capture independently:** whole-view flag, dirty row/group/partition, append-only OLD/NEW log, pending exact algebraic accumulators, and no-capture full refresh. Measure source p50/p95/p99, storage/WAL per source change, worker throughput, affected-scope ratio, backing writes, read latency and freshness lag.
5. **PostgreSQL:** first prototype trigger capture plus SQL-callable function and in-server background worker. In parallel, test an in-server logical-decoding consumer as an optional no-trigger capture backend; include duplicate replay, restart, slot invalidation and retained-WAL monitoring.
6. **MySQL/MariaDB:** prototype stored procedure plus one Event Scheduler coordinator, caller transaction preservation, savepoint/error behavior, non-overlap claims, scheduler-disabled behavior, server restart and session prepared-statement cleanup. Investigate whether native plugin work is justified for no-trigger capture.
7. **Only then broaden shapes:** projections, DISTINCT, joins, windows/partitions, holistic aggregates, UNION ALL and Tier 3 full refresh. Every strategy needs native differential tests and barrier-synchronized concurrency/recovery tests.

## 10. Open decisions

- Is eventual-scope consistency sufficient for the first public async mode, or must the initial release also offer committed-prefix consistency?
- What exact freshness barrier is exposed: drain current work, wait up to a timeout, or wait through a specified source frontier?
- Is PostgreSQL in-server logical decoding a supported advanced backend or an experimental alternative?
- Is trigger-based capture enough for MySQL/MariaDB, or does an in-server native plugin justify its installation and operational complexity?
- What is the correct SQLite extension distribution/loading contract across Python, C/C++, Rust, Go and other hosts while keeping maintenance execution in the database connection?
- Should database-resident scheduling be one coordinator per database, one per schema, or another shared topology?
- What is the migration/drop protocol for a view with pending work or a worker currently executing?

No design choice is final beyond the requirements stated at the top and the demonstrated SQLite UDF feasibility. Capture algorithms, public API, consistency modes and scheduler topology remain research questions.

## References

- SQLite UDF connection handle: https://www.sqlite.org/c3ref/context_db_handle.html
- SQLite nested SQL execution: https://www.sqlite.org/c3ref/exec.html
- SQLite function flags / DIRECTONLY: https://www.sqlite.org/c3ref/c_deterministic.html
- SQLite statement readonly caveat: https://www.sqlite.org/c3ref/stmt_readonly.html
- SQLite commit hook: https://www.sqlite.org/c3ref/commit_hook.html
- SQLite WAL hook: https://www.sqlite.org/c3ref/wal_hook.html
- PostgreSQL logical decoding: https://www.postgresql.org/docs/current/logicaldecoding-explanation.html
- PostgreSQL background workers: https://www.postgresql.org/docs/current/bgworker.html
- PostgreSQL SPI: https://www.postgresql.org/docs/current/spi.html
- MySQL stored-program restrictions: https://dev.mysql.com/doc/refman/8.4/en/stored-program-restrictions.html
- MySQL Event Scheduler: https://dev.mysql.com/doc/refman/8.4/en/events-overview.html
- DBSP: https://arxiv.org/abs/2203.16684
