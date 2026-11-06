[ ] I would like to avoid relying on regexes and other brittle manual parsing of the SQL, and instead use something proven like sqlglot
[ ] I would like the output SQL to be properly formatted (possibly by sqlglot itself) and possibly even optimized
[ ] I would like, when multiple views use the same base tables, to avoid redundant computations (e.g. in case two views both use the same aggregation over the same data)
[ ] I would like materialized view definitions to also be able to reference other materialized views, and their updates to still happen as efficiently as possible
[ ] I would like to be able to define, in addition to the default synchronously and atomically-updated MVs, also asynchronously, bulk-based, and atomically-updated materialized views
[ ] I would like the generated code to be optimal and as efficient as if it had been written as efficiently as possible for that specific view and that specific engine/version by a senior DBA
[ ] I would like the generator to be fuzzed by generating valid random schemas and views, as well as random DMLs, and then comparing the materialized view after each DML to the result of running the view directly
[ ] I would like the tests to exercise also pulling the schema from an existing database.
[ ] I would like the generator to autodetect (when connecting to the database) or accept as arg the engine and version, so that it can autoselect the correct and most efficient dialect/subdialect
[ ] I would like the generator to consider existing indexes on the base tables when deciding how to most efficiently perform the materialization
[ ] I would like the generator to propose additional indexes on the base tables, when adding them would yield most efficient materialization
[ ] I would like the generator to automatically create indexes on the aux or materialized tables when creating them helps materialization efficiency
[ ] I would like the generator to support as efficiently as possible every engine and engine version (possibly considering performance features specific to each)
[ ] I would like the README (or other doc linked from the README) to showcase how each SQL construct/aggregation and their combinations is efficiently rendered as an incremental materialization
[ ] I would like the repo to contain a Dockerfile/Docker compose that spins up all servers needed for integration testing (make docker-test)
[ ] I would like the generator to support also modern newsql engines (cockroach, tidb, yugabyte, spanner, etc.) where possible

