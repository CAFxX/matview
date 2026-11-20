# Research: More Efficient Maintenance Strategies for More SQL Constructs

Status: research input for the "Strategy ladder: replace refusals with
fallbacks" initiative in [TODO.md](../TODO.md). Nothing here is implemented.
The ranked lanes at the end are tracked as TODO items under that initiative.
The inventory describes master `b5220f6`, after bulk-maintenance stage 2,
when statement-level triggers became the only PostgreSQL mechanism.

The goal is to make more SQL constructs cheap to maintain on SQLite,
PostgreSQL, MySQL and MariaDB: incrementally, or at least more cheaply than
the group-scoped or full recompute they get today. The non-negotiable
contract in [AGENTS.md](../AGENTS.md) is unchanged. Every strategy below is
either exact (the MV equals the live view, bit for bit), or it is a named
**relaxed** class that the user enables per code. Relaxed classes are never
on by default.

## 1. Method

- **Inventory.** The inventory comes from reading `matview/planner.py`,
  `matview/maintenance/*`, `matview/dialects/*` and the tests. It was then
  spot-checked by registering views on native PostgreSQL 18 and on SQLite
  (section 2.3).
- **Prototypes.** Throwaway SQL-level prototypes live outside the repository,
  in `~/.cache/matview-work/research/proto/` (`ptrig.py`, `ptrig_protos.py`
  and `pfloat.py`). Each prototype installs the **row-trigger body** a
  strategy would generate, runs one seeded literal DML batch inside one
  transaction and one client round trip, and reports:

  ```text
  per-row trigger cost = (T_strategy − T_no_triggers) / N
  ```

  After every run it compares the MV with the live query exactly, as a row
  multiset. Every number in section 5 passed that differential. Two caveats:
  - **Shape.** The prototypes use row triggers on every engine. The PostgreSQL
    statement-level path amortises the per-statement part, but the per-row
    asymptotics are the same.
  - **Engines.** The engines were PostgreSQL 18.7, MySQL 8.4.13, MariaDB
    11.8.9 and SQLite 3.46.1. The server engines ran in containers under
    Lima/QEMU with `fsync` left at its default, except PostgreSQL, which ran
    with `fsync=off`. Absolute numbers are therefore indicative. The ratios
    between strategies are the result.

## 2. Inventory: what each construct gets today

Legend:

- **T1**: algebraic delta (fold).
- **T2**: group- or partition-scoped recompute with the engine's own
  aggregate.
- **T3**: full recompute of the original query (`ATOMIC_QUERY_SYNC`).
- **S/P/M**: SQLite, PostgreSQL, and MySQL or MariaDB.
- **Cost**: per changed source row. On P the same work runs once per
  statement over the transition tables.
- **"exact"**: the design target, verified by the existing differential
  tests. Known divergences are listed in the README.

### 2.1 Matrix

| Construct | S | P | M | Cost class | Guarantee |
|---|---|---|---|---|---|
| COUNT, integer/DECIMAL SUM, AVG(int) | T1 | T1 | T1 | O(1) | exact |
| float SUM/TOTAL | T1 while the group holds only integers, else T2 | T1 fold | T1 fold | S O(1) or O(group). P/M O(1) | S exact. **P/M drift** (README, TODO P0) |
| float AVG, NUMERIC AVG (P) | T2 recompute | T2 | T2 | O(group) | exact |
| MIN/MAX | insert: compare. Deleting the current extreme recomputes the group | same | same | insert O(1). Extreme delete O(group). Recommended index is `(group)`, not `(group, arg)` | exact |
| BIT_AND/BIT_OR | fold; recompute on delete | same | same | delete O(group) | exact |
| BIT_XOR | T1 XOR fold | **recompute** | T1 | P O(group), elsewhere O(1) | exact |
| BOOL_AND/OR | counters | counters | counters | O(1) | exact |
| VAR/STDDEV | T2 | T2 | T2 | O(group) | P maps `VARIANCE`/`STDDEV` to POP (TODO) |
| GROUP_CONCAT/STRING_AGG/ARRAY_AGG/JSON_GROUP_* | T2 | T2 | T2 | O(group) | exact |
| JSON_AGG, JSON_OBJECT_AGG, percentile_cont/disc, mode(), ANY_VALUE, regr_*, other unrecognised aggregates | — | **registration fails** (section 2.3) | not verified | — | fails closed, but there is no fallback |
| Any aggregate without GROUP BY that needs a recompute | T3 | T3 | T3 | O(table) | exact |
| COUNT(DISTINCT), single table | refcount aux table | refcount, but the count is **recounted** from the aux table | refcount | S/M O(1). P O(distinct values in group) | exact |
| DISTINCT aggregates over joins, MIN/MAX(DISTINCT) | T2 | T2 | T2 | O(group) | exact |
| FILTER, HAVING, compound aggregate expressions | T1 rewrite, or residual in the reader view | same | same | as the elementary aggregates | exact |
| GROUP BY expression | T1 | T1 + canonical key | T1 | O(1) | exact (S/M `1.0` vs `1.00` key split, TODO) |
| ROLLUP/CUBE/GROUPING SETS | `ValueError` (incidental) | same | same | — | refused |
| SELECT DISTINCT, single table | T1 counted | T1 | T1 | O(1) | exact |
| Windows with a common PARTITION BY | T2 partition recompute | T2 | T2 | O(partition) | exact |
| Windows without PARTITION BY, named WINDOW, windows + GROUP BY, windows + outer join | T3 | T3 | T3 | O(table) | exact |
| INNER/CROSS join + aggregate | T1 delta join | T1 | T1 | O(fan-out); recompute paths O(group over the join) | exact (P/M float SUM drift) |
| Join **projection** without aggregate (inner, outer, many-to-many) | **T3** | T3 | T3 | **O(table)** | exact |
| LEFT/RIGHT/FULL join + aggregate | T2 via an EXISTS probe over the join graph | T2 | T2 | O(group) + probe | exact (a known S LEFT-join group-move divergence is in TODO) |
| Self-join | T2 if grouped, else T3 | same | same | O(group) / O(table) | exact |
| LATERAL, USING, subquery in FROM (unless it can be unwrapped) | T3 | T3 | T3 | O(table) | exact |
| IN/EXISTS/NOT IN/NOT EXISTS, scalar subqueries | grouped: T2 for changes to the outer table, but a change to the **subquery table refreshes everything**. Otherwise T3 | same | same | O(group) / O(table) | exact |
| UNION/INTERSECT/EXCEPT (set semantics), UNION ALL with PKs | T1 Z-set counters | T1 | T1 | O(1) | exact |
| INTERSECT ALL / EXCEPT ALL, mixed operators | T3 | T3 | T3 | O(table) | exact |
| ORDER BY (even **without** LIMIT), LIMIT/OFFSET, top-k | T3 | T3 | T3 | O(table) | exact |
| Recursive CTE, several CTEs, nested aggregates | T3 | T3 | T3 | O(table) | exact |
| CASE/COALESCE/deterministic scalar UDFs | inline in any tier | same | same | as the tier | exact if the UDF is deterministic (not checked) |
| now()/random()/uuid() | refused `NONDETERMINISTIC_EXPR` (regex over a fixed list) | same | same | — | refused |

### 2.2 Structural observations

- **No registry yet.** The tier choice is a first-match `if/elif` chain in
  `planner.py` (`_select_strategy_and_mode`). Every proposal below assumes the
  registry from the first step of the strategy-ladder initiative: a strategy
  declares its shapes and engines, its guarantee, its cost class per engine
  and its concurrency properties.
- **T3 is the silent catch-all.** Anything that parses but matches no T1/T2
  rule becomes O(table) **per row** on SQLite/MySQL/MariaDB, which is O(N ×
  table) for an N-row statement. Most of the wins below are T3 → T1 or T3 →
  T2 moves, not T2 → T1 moves.
- **No statement-level path outside PostgreSQL.** SQLite, MySQL and MariaDB
  have only row triggers, so any per-statement amortisation on those engines
  needs a deferred change log, which is a staleness relaxation (section 3.9).

### 2.3 Native spot-checks done for this research

- **PostgreSQL 18.7.** Registering a grouped view with each of these fails at
  install time with `UndefinedColumn`: `JSON_AGG`, `JSON_OBJECT_AGG`,
  `percentile_cont`, `percentile_disc`, `mode() WITHIN GROUP`, `ANY_VALUE`,
  `regr_slope`. The cause is that the reader view references base columns on
  the backing table, because `_parse_single_agg_expr` does not recognise
  these aggregates. This fails closed, so no wrong data is produced, but these
  constructs can be maintained exactly by T2 and today get neither T2 nor a
  clear refusal.
- **SQLite.** A user-defined Python aggregate falls through to T3 and stays
  exact.
- **SQLite.** `GROUP BY ROLLUP(g)` raises `ValueError` ("Non-aggregated
  column").

## 3. Technique survey

Each technique below gives: where it comes from, what it needs from the
engine, its guarantee and its cost class.

### 3.1 Counting and delta rules (the foundation)

- **The counting algorithm.** Gupta, Mumick and Subrahmanian
  ([SIGMOD 1993](https://doi.org/10.1145/170035.170066)) keep a derivation
  count per view tuple, so deletions are exact without rederivation.
- **Bag-algebra delta rules.** Griffin and Libkin, *Incremental maintenance of
  views with duplicates*, SIGMOD 1995.
- **Summary-delta tables** for aggregates with COUNT companions. Mumick,
  Quass and Mumick, *Maintenance of data cubes and summary tables in a
  warehouse*, SIGMOD 1997. Oracle's fast refresh has the same requirement:
  `COUNT(*)` and `COUNT(x)` columns must accompany `SUM`/`AVG`.
- **Where the repo is.** It already has all three: `__mv_row_count`,
  `__mv_cnt_*`, Z-set counters, and DISTINCT refcount tables. The open
  question is not the theory. It is the **coverage** of constructs that
  today fall to T3.

### 3.2 DBSP, Z-sets, differential dataflow and higher-order IVM

- **DBSP.** Budiu, McSherry, Ryzhyk and Tannen,
  [*DBSP: Automatic Incremental View Maintenance for Rich Query
  Languages*](https://arxiv.org/abs/2203.16684), VLDB 2023. It shows that
  every relational operator, plus recursion, has an incremental form over
  Z-sets (multisets with integer weights). Linear operators are incremental
  for free. Bilinear ones (join) are `Δa ⋈ b + a ⋈ Δb + Δa ⋈ Δb`.
  Non-linear ones (DISTINCT, aggregates) need **integrated state**.
- **Differential dataflow.** McSherry, Murray, Isaacs and Isard,
  [CIDR 2013](https://www.cidrdb.org/cidr2013/Papers/CIDR13_Paper111.pdf).
  Feldera (DBSP), Materialize (differential dataflow) and RisingWave all keep
  that integrated state as indexed "arrangements" or state tables. In
  particular, MIN/MAX and top-N over retractable inputs keep the **full
  ordered input** per group: the multiset counter table of section 3.3.
- **Higher-order IVM.** DBToaster (Koch et al.,
  [VLDB 2012](https://arxiv.org/abs/1207.0137)) materialises deltas of
  deltas, which makes them cheap. F-IVM (Nikolic and Olteanu,
  [SIGMOD 2018](https://arxiv.org/abs/1703.07484)) factorises the
  maintenance of aggregates over joins with ring payloads.
- **What is applicable here.**
  - The Z-set view is already the repo's model.
  - The integrated state of non-linear operators maps to **auxiliary tables
    maintained by triggers** (sections 3.3 to 3.5).
  - Higher-order delta views are useful for aggregates over **multi-way
    joins**, where `Δa ⋈ b ⋈ c` would otherwise rescan `b ⋈ c`. Materialising
    `b ⋈ c` partial aggregates (F-IVM's view tree) is a later lane. It
    multiplies storage and triggers, so it should be an opt-in cost class.

### 3.3 MIN/MAX and other non-invertible aggregates

- **Literature.**
  - Palpanas, Sidle, Cochrane and Pirahesh, *Incremental Maintenance for
    Non-Distributive Aggregate Functions*, VLDB 2002: recompute only the
    affected groups. This is today's T2.
  - The stream literature keeps an ordered multiset (balanced tree or heap
    with lazy deletion).
  - In a database, the natural ordered multiset is a **B-tree**: either an
    index on `(group, arg)` over the base table, or an auxiliary table
    `(group, value) → count` with that primary key. In both cases the
    recompute after deleting the extreme becomes one O(log n) index probe.
  - pg_ivm documents the same O(group) recompute
    ([README](https://github.com/sraoss/pg_ivm)). BigQuery and Snowflake
    simply restrict the MV shapes.
- **Variants.**
  - **(a)** Recommend, or require, the `(group, arg)` index for MIN/MAX. The
    repo's `recommended_base_indexes` names only `(group)` today, and the
    measurement shows that is not enough.
  - **(b)** A multiset auxiliary table, for arguments that are expressions or
    come from joins, where no base index can serve.
  - **(c)** BIT_AND/BIT_OR via per-bit counters: 64 counters per group,
    O(1).
  - **(d)** BIT_XOR on PostgreSQL via fold, since XOR is self-inverse.
- **Guarantee.** All exact.

### 3.4 DISTINCT and COUNT(DISTINCT) via reference counts

- **Technique.** Reference counts per `(group, value)`, the counting
  algorithm. The repo already does this. On PostgreSQL, however, the per-group
  count is recounted from the aux table: O(distinct values in the group). The
  O(1) transition form is `cd += [refcount 0→1] − [refcount 1→0]`, and it
  works over transition tables too: aggregate the signed per-value deltas,
  then join to the aux table's old counts.
- **Extensions.**
  - DISTINCT aggregates over joins: refcount the join output values.
  - `SUM(DISTINCT)`: fold over 0↔1 transitions.
- **Sketches** (HyperLogLog, Flajolet et al. 2007) would be a relaxed class
  for COUNT(DISTINCT). They are **dominated**: the exact refcount is already
  O(1) per row, and the only saving would be storage. Not proposed.

### 3.5 Outer joins, semi-joins and anti-joins

- **Outer joins.**
  - Larson and Zhou, *Efficient Maintenance of Materialized Outer-Join Views*,
    ICDE 2007.
  - Griffin and Kumar, *Algebraic change propagation for semijoin and
    outerjoin queries*, SIGMOD Record 1998.
  - The null-padded row of a preserved-side tuple exists **iff its match
    count is 0**. A delta on the null-supplying side that moves a key's
    match count across 0 retracts or emits the padded rows of that key.
  - **Per-row cost:** O(fan-out of the key). No table scan is needed.
  - **FULL** applies the same rule on both sides.
  - **RIGHT** is LEFT mirrored.
- **Semi/anti joins** (`EXISTS`/`IN`/`NOT EXISTS`/`NOT IN`).
  - Keep a per-key match counter on the subquery side.
  - An outer row qualifies iff `cnt > 0` (semi) or `cnt = 0` (anti).
  - **`NOT IN` caveat.** Its NULL semantics need an extra "subquery side
    contains NULL" counter, because a NULL makes `NOT IN` UNKNOWN for every
    row.
- **Precedent elsewhere.** pg_ivm supports outer joins (without aggregates)
  and EXISTS by exactly this kind of rule. BigQuery supports LEFT JOIN only
  for append-only changes to the leftmost table.
- **Today in the repo.** Join projections are T3. A change to a subquery
  table refreshes the whole view.

### 3.6 Top-k, ORDER BY and LIMIT

- **Literature.** Yi, Yu, Yang, Xia and Chen, *Efficient Maintenance of
  Materialized Top-k Views*, ICDE 2003. Keep `k' ≥ k` rows, absorb inserts
  below the threshold for free, and refill from the base table, using an
  index, only when deletions drop below `k`. Streaming systems (RisingWave
  "managed top-N", Materialize `TopK`) keep the ordered group state.
- **Two cheaper observations for this repo.**
  1. **ORDER BY without LIMIT is not a maintenance concern.** The rows are
     the inner query's rows. The reader view can re-apply the ORDER BY over
     the backing table, the same way it already applies HAVING. Today such a
     view falls to T3. With this change it gets the inner shape's tier.
  2. **ORDER BY ... LIMIT k over a maintainable inner query.** Maintain the
     un-limited inner result, which is usually a grouped aggregate, and let
     the reader view apply `ORDER BY ... LIMIT k` over the backing table,
     with an index on the order key. Writes then cost the inner tier: O(1)
     for SUM/COUNT. Reads cost O(log n + k). This is exact by construction
     and covers the common leaderboard shape, "top customers by SUM". The
     cost moves to readers, so it must be declared as a cost class in the
     registry: *per read O(k log n)*.
- **Threshold maintenance of a k-row table.** This is only worth it when the
  inner query is not maintainable or not stored. Measured below, it gives
  **no win** over an index-backed full recompute at k = 50. Low priority.

### 3.7 Window functions

- **Sliding-window aggregation (streams).** For a ROWS frame of fixed width
  over an append stream:
  - SUBTRACT-ON-EVICT works for invertible aggregates.
  - Two-stacks, DABA (Tangwongsan, Hirzel and Schneider, *Low-Latency
    Sliding-Window Aggregation in Worst-Case Constant Time*, DEBS 2017) and
    FlatFIT (Shein, Chrysanthis and Labrinidis, SSDBM 2017) handle
    non-invertible aggregates in O(1) amortised or worst-case time.
  - These assume FIFO arrival. A materialised window view can see arbitrary
    inserts and deletes, so the useful database form is different.
- **Prefix-scoped (suffix) recompute.** For every window function whose
  frame looks only **backwards**, the output of rows *before* the changed
  position cannot change. That covers ROW_NUMBER, RANK, DENSE_RANK, LAG,
  running aggregates with `UNBOUNDED PRECEDING .. CURRENT ROW`, and `ROWS n
  PRECEDING`. Recompute only the suffix from the changed position, plus the
  frame look-back. This is generic, exact and cheap for **appends**, the
  dominant time-series pattern, where the suffix is one row.
- **Invertible suffix delta.** For running SUM/COUNT (integers and decimals)
  and for ROW_NUMBER/RANK: `UPDATE … SET rs = rs ± v, rn = rn ± 1 WHERE
  (ts, id) > changed`, plus one point insert or delete.
  - **Cost:** O(changed outputs). That is optimal, because those outputs
    really change.
  - **RANGE frames.** Peers (ties on the ORDER BY key) are inside the frame,
    so the update predicate becomes `ts ≥ changed.ts` and the new row's value
    comes from the last peer.
  - **Floats** are not exact under this rule (section 3.8), so floats stay on
    the suffix recompute.
- **Not helped.** Functions that depend on partition **size** or on rows
  *after* the current one: NTILE, PERCENT_RANK, CUME_DIST, LEAD, `FOLLOWING`
  frames. They keep the partition recompute. LEAD's change is still local
  (one predecessor row), but it needs its own rule.

### 3.8 Exact floating-point summation (a relaxed class)

- **Why this is relaxed.** The live `SUM(float)` is the engine's
  order-dependent left fold. The MV can match it bit for bit only by
  recomputing in the same order, which is what the TODO P0 float item does
  with O(group) recomputes.
- **The alternative.** Maintain the **exact** sum with a long accumulator
  (Kulisch), a superaccumulator (Neal,
  [*Fast exact summation using small and large
  superaccumulators*](https://arxiv.org/abs/1505.05571), 2015), or a
  reproducible accumulator (Demmel and Nguyen, *Parallel Reproducible
  Summation*, IEEE TC 2015). Then round once, correctly, at read time.
- **Properties of that result.**
  - It is deterministic, order-independent and history-independent.
  - It is *more* accurate than the live value.
  - It is **not** equal to the live value in general. The measurement found
    the live `SUM(float8)` differing from the correctly rounded sum in 4 of 5
    adversarial groups.
  - So it is the relaxed class `float_sum_correctly_rounded`, opt-in per
    code.
- **Engine applicability.**
  - **PostgreSQL:** an exact `numeric` integer accumulator. Every finite
    double is an integer multiple of 2⁻¹⁰⁷⁴. Prototyped and exact against
    `math.fsum`. `float8::numeric` rounds to 15 significant digits, so the
    conversion must decompose the mantissa and exponent.
  - **MySQL/MariaDB:** `DECIMAL(65)` holds about 2²¹⁵, not the full double
    range (about 2²⁰⁹⁸). This needs fixed limbs: about 66 BIGINT columns of
    32 bits each, lazily normalised, or a per-view exponent-window opt-in.
  - **SQLite:** limbs in INTEGER columns, using the math functions
    (`log2`/`pow`) or an application-defined function. Writers outside
    Python would not have that function, so the math-functions build is
    required.
  - **±Inf and NaN:** separate counters, mirroring IEEE: any NaN, or both
    infinities, gives NaN.

### 3.9 Deferred, lazy and batched maintenance (staleness opt-in)

- **Literature.** Colby, Griffin, Libkin, Mumick and Trickey, *Algorithms for
  Deferred View Maintenance*, SIGMOD 1996. Zhou, Larson and Elmongui, *Lazy
  Maintenance of Materialized Views*, VLDB 2007. Oracle's `FAST` / `FAST_PCT`
  / `FORCE` refresh (FORCE means fast, falling back to complete) is the
  commercial precedent for a strategy ladder with fallback, and
  `DBMS_MVIEW.EXPLAIN_MVIEW` is its diagnostics.
- **What it would be here.** A change log per source table, written by cheap
  O(1) row triggers, plus a set-based merge. The merge would run on read
  (lazy), on commit, or periodically. Batching and coalescing the deltas lets
  SQLite, MySQL and MariaDB get statement-level amortisation, which they have
  no trigger for.
- **Guarantee.** This is exact only if the reader merges before reading.
  Otherwise it is the relaxed class `bounded_staleness(t)`. It is already
  sketched as P3 "Asynchronous, Bulk-Based … Materialized Views". This
  research adds only the classification.

### 3.10 Recursive queries

- **Insertions:** semi-naive evaluation (Bancilhon and Ramakrishnan, 1986).
- **Deletions:** DRed, "delete and rederive" (Gupta, Mumick and Subrahmanian,
  1993), or the backward/forward algorithm (Motik, Nenov, Piro and Horrocks,
  *Maintenance of Datalog Materialisations Revisited*, AIJ 2019). For
  transitive-closure-like views, counting does not terminate on cycles.
- DBSP handles recursion generically, but inside a dataflow runtime. Done
  with triggers, it is a substantial project.
- pg_ivm, BigQuery and Snowflake all refuse recursive CTEs.
- Recommendation: keep T3. Research only.

### 3.11 Engineering improvements to scoped recompute

These keep the strategy but cut the constant or the scope:

- **(a)** Index the recompute's *full* access path: `(group, arg)` for
  MIN/MAX, `(partition, order)` for windows. Measured gains are 25x to 940x,
  section 5.
- **(b)** Skip the recompute when the change cannot affect the result:
  - MIN delete of a non-extreme. The repo already does this.
  - UPDATEs that touch no column the view reads. On PostgreSQL this is open
    because transition tables cannot be combined with `UPDATE OF`. A
    column-diff predicate over `old ⋈ new` does it.
- **(c)** Suffix-scoped window recompute (section 3.7).
- **(d)** Key-ordered locks and once-per-statement recompute on PostgreSQL:
  done in stage 2.

## 4. Applicability per engine

| Technique | SQLite | PostgreSQL | MySQL / MariaDB | Guarantee | Cost class | Extra storage | Concurrency notes |
|---|---|---|---|---|---|---|---|
| `(group,arg)` index for MIN/MAX recompute | yes | yes | yes | exact | O(log n) per extreme delete | base index | same as T2 today |
| Multiset aux table (MIN/MAX of expressions or joins) | UPSERT ≥ 3.24 | `ON CONFLICT` | `ON DUPLICATE KEY` | exact | O(log n) per row | one row per distinct (group, value) | aux rows are locked in key order (PG); InnoDB row locks on the aux PK |
| Per-bit counters BIT_AND/OR | yes | yes | yes | exact | O(64) per row | 64 ints per group | as T1 |
| O(1) COUNT(DISTINCT) on PG | — | transition tables: aggregate Δ per value, join old counts | — | exact | O(Δ values) per statement | existing aux | aux upserts in key order |
| Outer-join delta rules | yes | statement form: Δ ⋈ other + pad rule; padding must be idempotent | yes | exact | O(fan-out) per row | index on MV `(a_id, b_id)` and partial `(b_id)` | the pad rule reads the other side; under PG READ COMMITTED it needs the same lock-then-recheck discipline as stage 2 |
| Semi/anti join counters | yes | yes | yes | exact | O(fan-out) per row | counter per key | as above |
| Reader-side ORDER BY / LIMIT | yes | yes | yes (a LIMIT inside a view is allowed) | exact | writes: inner tier. Reads: O(log n + k) | index on the order key | none beyond the inner tier |
| Suffix-scoped window recompute | yes | yes | yes | exact | O(suffix + frame) | MV index `(partition, order, pk)` | per-partition lock as T2 |
| Invertible suffix delta (running SUM/COUNT int/decimal, ROW_NUMBER/RANK) | yes | yes | yes | exact | O(changed outputs) | same | same |
| INTERSECT ALL / EXCEPT ALL via branch counters | yes | yes | yes | exact | O(1) | existing `__mv_cnt_0/1` | as Z-set |
| GROUPING SETS → UNION ALL of GROUP BYs | SQLite has no GROUPING SETS (live view impossible) | yes | ROLLUP only (`WITH ROLLUP`) | exact | sum of the branches' tiers | one store per branch, or a shared store with a grouping id | as T1 |
| Exact float accumulator | limbs + math functions | `numeric` | limbs | **relaxed** `float_sum_correctly_rounded` | O(1) per row | ~66 ints (limbs) or one numeric | as T1 |
| Deferred change log + merge | yes | yes | yes | exact if merged on read, else **relaxed** `bounded_staleness` | O(1) per row + O(Δ) per merge | log tables | the merge must serialise with readers |
| Top-k threshold table | yes | yes | yes | exact | O(1) typical; refill O(log n) | k rows | refill reads the base under the trigger's isolation |
| Median/percentile cursor | no native aggregate | yes (`percentile_disc`) | MariaDB window form only | exact for `_disc`; `_cont` adds interpolation | O(log n) per row (one cursor step) | cursor row per group | the cursor needs a stable total order `(v, pk)` |
| Semi-naive / DRed recursive CTE | yes | yes | yes | exact | output-sensitive; DRed can be O(table) | derivation state | complex; research only |

## 5. Measurements

Each prototype first loads 5 groups/partitions of 20 000 rows (100 000 rows;
the left-join prototype also loads a 50 000-row right side). The hot group
or partition is the target of 400 single-row DML operations (left join: 60),
50% inserts and 50% deletes unless noted. Cost is **per changed row**, in µs.
Every run passed the exact differential against the live query.

| Prototype (data) | Strategy | SQLite 3.46 | PostgreSQL 18 | MySQL 8.4 | MariaDB 11.8 |
|---|---|---|---|---|---|
| **MIN after deleting the current minimum** (5 × 20k rows, about 5k distinct values per group) | recompute, no index | 1734 | 2664 | 13380 | 12186 |
| | recompute, index `(g)` (today's recommendation) | 2066 | 999 | 14109 | 12336 |
| | recompute, index `(g, v)` | **2.2** | **41** | **165** | **136** |
| | multiset aux table, no base index | 15 | 48 | ≈ 0 ¹ | 120 |
| **COUNT(DISTINCT v)** (same data, random inserts and deletes) | recompute with `(g, v)` index | 758 | 3738 | 23489 | 21767 |
| | refcount, count recounted from aux (PG today) | 126 | 500 | 1405 | 1297 |
| | refcount, O(1) 0↔1 transitions (S/M today) | **14** | **46** | **92** | ≈ 0 ¹ |
| **Window, append** (running SUM + ROW_NUMBER, 5k-row partition ²) | partition recompute (today) | 9298 | 189840 ³ | 74353 | 48185 |
| | suffix delta | **41** | **99** | **255** | **189** |
| **Window, random position** (same) | partition recompute (today) | 9267 | 183828 ³ | **69000** | 48782 |
| | suffix delta | **826** | **91629** | 137725 | **9567** |
| **Top-k, k = 50** (100k rows, index `(v, id)`, 25% of the changes delete a member) | full recompute (= T3 today) | **25** | 1208 | 734 | **372** |
| | threshold + single-row refill | 28 | **36** | **275** | 846 |
| **LEFT JOIN projection** (100k left × 50k right, 1% NULL keys, changes on both sides, key updates) | full recompute (T3 today) | 273414 | 806090 | 2458496 ⁵ | 50734590 ⁵ |
| | null-padding delta rules | **27** | **772** | **6477** ⁵ | **1003** ⁵ |
| **Float SUM** (5 × 20k float8 rows with adversarial magnitudes and subnormals) | group recompute (exact vs live) | — | 2508 | — | — |
| | numeric superaccumulator (relaxed: correctly rounded) | — | **42** ⁴ | — | — |

¹ Below the timing noise. On these server engines a bare DML statement
costs 45–260 µs in this setup, and the difference fell inside its jitter.
² SQLite at 20k rows per partition: append 45314 → 44 µs, random 46103 →
4175 µs.
³ PostgreSQL costs here include MVCC dead-tuple growth: all changes run in
one transaction, and each recompute rewrites the partition. At 20k rows per
partition the run did not finish 400 changes in 9 minutes (more than
1.4 s per row).
⁴ The accumulator equalled the `math.fsum` oracle in all groups. The live
`SUM(float8)` differed from the correctly rounded sum in 4 of 5 groups, so
this strategy is not equal to the live view. It is a relaxed class.
⁵ Six changes only (the T3 path costs seconds per row). The bare-DML
baseline is noisier at this sample size.

Reading the numbers:

- **MIN/MAX.** The cheapest fix is an index, not a new strategy. With
  `(g, v)`, the recompute after deleting the extreme becomes one index probe
  on every engine: 25x (PG) to 940x (SQLite) faster than with today's
  recommended `(g)` index. On SQLite, `(g)` alone is *slower* than no index
  at all. The multiset aux table reaches the same cost class without any
  base index. It is the tool for `MIN(expr)` and for MIN over joins.
- **COUNT(DISTINCT).** The O(1) transition form is 11–15x faster than
  PostgreSQL's current recount of the aux table, and the recount's cost grows
  with the number of distinct values per group.
- **Windows.**
  - *Appends* are where suffix maintenance pays: 190x to 1900x on every
    engine.
  - *At random positions* the in-place suffix UPDATE touches about half of
    the partition. It wins on SQLite (11x), MariaDB (5x) and PostgreSQL (2x)
    but **loses on MySQL** (0.5x): updating many rows in place is costlier on
    InnoDB than rebuilding the partition with `INSERT … SELECT`.
  - So L5 must choose by the cost model: suffix delta near the tail, a
    suffix or partition *recompute* otherwise. The generic suffix-scoped
    recompute was not prototyped.
- **Top-k threshold maintenance is engine-dependent.** It wins on PG (33x,
  where the plpgsql rewrite of k rows is expensive) and MySQL (2.7x), ties on
  SQLite and loses on MariaDB (0.4x). L3, the reader-side `LIMIT`, avoids
  the question for maintainable inner queries. L11 stays low priority.
- **LEFT JOIN.** Moving join projections off T3 is the largest single win
  measured: 380x (MySQL) to 50 000x (MariaDB).
- **Exact float accumulator.** O(1) instead of O(group), about 60x at 20k
  rows per group. It is opt-in only, because it is deliberately not equal to
  the live, order-dependent result.

## 6. Ranked candidates

Value is the number of realistic views that become cheap. Cost and risk
cover the implementation size, how many engines need special forms, and how
much concurrency is affected. Each candidate maps to a lane in section 7.

| Rank | Candidate (lane) | Views that become cheap | Measured gain (section 5) | Cost / risk |
|---|---|---|---|---|
| 1 | Unrecognised aggregates → T2 (L1) | every PG view with `JSON_AGG`, percentiles, `mode()`, `ANY_VALUE`, `regr_*`; UDF aggregates | from "registration fails" to exact O(group) | tiny / low: routing plus a frontend fix |
| 2 | Reader-side ORDER BY / LIMIT (L3) | every view with ORDER BY (T3 today); top-N leaderboards over aggregates | from O(table) per row to the inner tier (O(1) for SUM/COUNT); reads O(log n + k) | small / low: reader-view SQL, registry per-read cost |
| 3 | `(group,arg)` index, multiset aux, bit counters (L2) | MIN/MAX/BIT_AND/OR on delete-heavy tables | MIN extreme delete, `(g)` → `(g,v)` index: SQLite 2066 → 2.2, PG 999 → 41, MySQL 14109 → 165, MariaDB 12336 → 136 µs | small / low |
| 4 | Outer-join delta rules (L4) | LEFT/RIGHT/FULL join projections (T3 today), later outer-join aggregates | SQLite 273 ms → 27 µs, PG 806 → 0.8 ms, MySQL 2.5 s → 6.5 ms, MariaDB 51 s → 1.0 ms per row | medium / medium: PG statement form and idempotent padding |
| 5 | Window suffix maintenance (L5) | time-series windows (appends), ranking, running totals | append: 190x–1900x on all engines; random position 0.5x (MySQL) to 11x (SQLite), so it needs a cost-model choice | medium / medium |
| 6 | O(1) COUNT(DISTINCT) on PG (L6) | PG COUNT/SUM(DISTINCT) with many distinct values | PG 500 → 46 µs per row (5k distinct values per group) | small / low |
| 7 | Semi/anti-join counters (L7) | EXISTS/IN/NOT EXISTS filters (full refresh today on subquery-side changes) | O(table) → O(fan-out), by analogy with L4 (not prototyped) | medium / medium: `NOT IN` NULL semantics |
| 8 | INTERSECT/EXCEPT ALL via counters (L8) | bag set operations (T3 today) | O(table) → O(1) (not prototyped; uses the existing counters) | small / low |
| 9 | GROUPING SETS/ROLLUP/CUBE rewrite (L9) | reporting views (fail with `ValueError` today) | refused → T1 per branch | medium / low |
| 10 | Exact float accumulator, relaxed (L10) | float SUM/AVG where the user accepts the correctly rounded result | PG 2508 → 42 µs per row at 20k rows/group | medium / medium: opt-in only, limbs on M/S |
| 11 | Top-k threshold (L11) | `LIMIT` over non-storable inner queries | engine-dependent at k = 50: PG 33x, MySQL 2.7x, SQLite even, MariaDB 0.4x | small / low value |
| 12 | Median/percentile_disc cursor (L12) | PG percentile views | O(group log group) → O(log n) (design only) | medium / low value |
| 13 | Recursive CTE semi-naive/DRed (L13) | graph and hierarchy views | — | high / high |

## 7. Roadmap: implementation lanes

### 7.1 Ground rules for every lane

- **One PR per lane**, registering one strategy (or a variant) in the
  strategy registry with:
  - its shapes and engines;
  - its guarantee;
  - its cost class per engine;
  - its concurrency properties.
- **Lanes that depend on the registry.** L1 to L9 can start before the
  registry exists by plugging into the current planner chain, but each PR
  must land its registry entry as soon as the registry exists. L10 and L13
  need the registry and the user constraint knobs (`exact|relaxed`, staleness)
  first, because they introduce relaxed classes.
- **Test plan for every lane** (AGENTS.md):
  - **Native differential.** Native, exact differential against the live view
    after initial population and after INSERT, UPDATE (including group and
    key moves), DELETE, upsert, multi-row statements, rollback, and NULL and
    duplicate edge cases, on every engine/version in the CI matrix.
  - **Concurrency.** Independent-connection interleavings where the lane takes
    new locks.
  - **Mutation checks.** Targeted mutations that the differential must catch,
    for example "drop the pad rule", "skip the refill", "off-by-one in the
    suffix predicate".
  - **Cost assertions.** Assert the strategy's cost class through invocation
    or row-touch counts, not wall-clock time.

### 7.2 Test-duration budget

The per-change gate must not get slower in a way anyone notices:

- **Fast cases only.** Every lane adds only fast, deterministic, table-driven
  cases: at most a few hundred rows, shared fixtures per engine, no sleeps.
- **Budget:** at most +3 s per engine job per lane.
- **Out of the gate:** stress, large-partition and benchmark runs, such as the
  prototypes here, go to the opt-in `scripts/bench_bulk.py` style runners
  (`MATVIEW_*` env gates).
- **Expected net effect:** about zero or negative, because tests that
  currently install T3 views on 1k-row tables get cheaper per DML once these
  views move to T1/T2.

### 7.3 Lane order

Lanes run in rank order. L1–L3 are the proposed **first three**: each is
small, exact and low-risk, and together they remove the most refusals and
the most O(table) paths.

- **L1. Unrecognised aggregates fall back to group-scoped recompute.**
  - **Strategy:** T2 when there is a GROUP BY, otherwise T3.
  - **Tests:** a table of aggregates per engine (PG: `JSON_AGG`,
    `JSON_OBJECT_AGG`, percentiles, `mode()`, `ANY_VALUE`, `regr_*`, a UDF
    aggregate; M/S: the engine's own equivalents), each with insert, delete,
    group move and empty group, compared with the live view.
  - **Gate impact:** about +1 s per engine.
- **L2. MIN/MAX/BIT_AND/BIT_OR deletes.**
  - **Changes:**
    - `recommended_base_indexes` names `(group…, arg)`.
    - The recompute is written so that it uses that index.
    - Expression or join arguments use the multiset aux table `(group…,
      value) → count`. This turns the P2 aux-table item into a registry
      strategy.
    - Per-bit counters for BIT_AND/OR.
    - BIT_XOR fold on PG.
  - **Registry:** cost class O(log n) when the index is present, O(group)
    otherwise. The planner prints which one applies.
  - **Tests:**
    - extreme-delete sequences, ties, NULLs, all-deleted groups, multi-row
      deletes of several extremes;
    - concurrent deletes of the same extreme from two connections;
    - a mutation check: dropping the "recompute when the extreme is deleted"
      branch must fail.
  - **Gate impact:** about +2 s per engine.
- **L3. Reader-side ORDER BY / LIMIT.**
  - **Changes:**
    - Split the top-level `ORDER BY [LIMIT/OFFSET]` off the query.
    - Plan the inner query with the normal ladder.
    - Re-apply the clause in the reader view, and index the order key on the
      backing table.
    - Ties: the reader's `LIMIT` must pick the same rows as the live query
      only where the SQL contract defines the order. With ties at the cut,
      both results are valid. The oracle compares the multiset of order keys
      at the cut, which is the documented, deliberate rule.
  - **Registry:** a new per-read cost dimension.
  - **Tests:**
    - ORDER BY without LIMIT;
    - LIMIT/OFFSET, including ties at the cut, NULL ordering per engine,
      `LIMIT 0`, and OFFSET past the end.
  - **Gate impact:** about +1 s per engine.
- **L4. Outer-join delta rules.**
  - **Strategy:** row form on S/M. On PG, a statement form over the
    transition tables, with idempotent padding.
  - **Tests:**
    - keys whose match count goes 0↔1↔n;
    - NULL join keys;
    - updates that change keys on either side;
    - multi-row statements that delete every match of a key;
    - PG READ COMMITTED with two writers on the two sides.
  - **Gate impact:** about +3 s per engine.
- **L5. Window suffix maintenance.**
  - **Strategy:** suffix recompute generically; invertible deltas for
    integer and decimal running SUM/COUNT and for ROW_NUMBER/RANK.
  - **Tests:**
    - append, middle insert, delete of the first/last/peer row, ties;
    - RANGE vs ROWS frames;
    - partition moves;
    - a mutation check on the suffix predicate.
  - **Gate impact:** about +2 s per engine.
- **L6. O(1) COUNT(DISTINCT) on PG, then refcounted DISTINCT aggregates over
  joins.**
  - **Gate impact:** about +1 s.
- **L7. Semi/anti-join counters.**
  - **Gate impact:** about +2 s per engine.
- **L8. INTERSECT/EXCEPT ALL.**
  - **Gate impact:** about +1 s per engine.
- **L9. GROUPING SETS/ROLLUP/CUBE.**
  - **Engines:** PG; MySQL/MariaDB `WITH ROLLUP`. SQLite has no grouping
    sets, so there is no live view to compare with.
  - **Gate impact:** about +1 s per engine.
- **L10. Relaxed `float_sum_correctly_rounded`.**
  - **Prerequisites:** the registry, the `exact|relaxed` knob and the
    diagnostics framework.
  - **Oracle:** an independent one, the correctly rounded sum (`math.fsum`
    over the current rows), *not* the live view. Separate tests assert that
    the default stays exact and that the relaxed diagnostic is printed.
- **L11, L12, L13.** Low priority, or research only (section 6).

## 8. Risks and explicit exclusions

- **Reader-side residuals move cost to reads.** The registry's cost model
  must express per-read cost, so that a budget can forbid it.
- **Deferred row triggers on PostgreSQL.** AFTER row triggers there fire after
  the whole statement, and statement triggers see the final state. Any rule
  that reads "the other side's current state" must therefore be idempotent:
  the pad rule must check that no padded row exists yet. Otherwise multi-row
  statements double-apply. The prototype's pad rule is idempotent for this
  reason.
- **Aux tables add write amplification and lock footprint.** Their key order
  must follow the global lock-ordering item in TODO.
- **The relaxed float class must never be the default.** It must be
  announced with `relaxed result allowed: using strategy
  float_sum_correctly_rounded`.
- **Not proposed:**
  - sketches for exact aggregates (dominated by refcounts);
  - generic higher-order IVM / F-IVM view trees (a later opt-in, once
    multi-way join aggregates are measured as a bottleneck);
  - recursive-CTE IVM (research only; it stays T3).
- **Scope of the prototypes.** They used row triggers, single-row
  statements, one isolation level and no concurrent writers. The production
  lanes must re-establish exactness under the full matrix.

## 9. Prototype artifacts (outside the repository)

All in `~/.cache/matview-work/research/proto/`:

| File | Contents |
|---|---|
| `ptrig.py` | Harness: engines, triggers, batch timing, exact differential |
| `ptrig_protos.py` | MIN/MAX, COUNT(DISTINCT), windows, top-k and LEFT JOIN strategies |
| `pfloat.py` | PostgreSQL exact float accumulator and its `fsum` oracle |
| `ptrig.out`, `ptrig2.out`, `pfloat.out` | Raw results (`ptrig.out` holds the first run, including superseded SQLite lines from before the top-k, left-join and COUNT(DISTINCT) harness fixes) |
