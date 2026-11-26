# Contributing

This file contains maintainer and contributor information. User-facing installation and usage documentation lives in [README.md](README.md).

## Validation

Install the development dependencies and run the local suite:

```sh
python3 -m pip install -r requirements-dev.txt
make test
```

The README examples and links are also executable documentation:

```sh
python3 test_readme_examples.py
```

The test suite includes fast offline tests and native database tests. The CI workflows define the authoritative engine/version matrix; a local SQLite-only run does not cover the other engines.

## CI coverage

The tables below are generated from the workflow definitions. To refresh them after changing a workflow, run `python3 test_readme_examples.py --print-matrix` and replace the generated section.

<!-- BEGIN GENERATED: ci-matrix -->
PR gate ([`ci.yml`](.github/workflows/ci.yml); required, every pull request; fixed fuzz seeds only):

| CI job | Python | Database |
| --- | --- | --- |
| SQLite 3 (Python 3.11) | 3.11 | SQLite bundled with CPython (`sqlite3`) |
| SQLite 3 (Python 3.15) | 3.15 | SQLite bundled with CPython (`sqlite3`) |
| APSW SQLite | 3.12 | SQLite via `apsw` |
| SQLCipher 4 | 3.12 | SQLite via `sqlcipher3-binary` |
| Turso libSQL | 3.12 | SQLite via `libsql-experimental` |
| PostgreSQL 15 | 3.12 | `postgres:15` |
| PostgreSQL 18 | 3.12 | `postgres:18` |
| MySQL 8.4 | 3.12 | `mysql:8.4` |
| MySQL 9.7 | 3.12 | `mysql:9.7` |
| MariaDB 10.11 LTS | 3.12 | `mariadb:10.11` |
| MariaDB 11.8 LTS | 3.12 | `mariadb:11.8` |
| MariaDB 12.3 LTS | 3.12 | `mariadb:12.3` |

Extended random-seed fuzz ([`ci-extended.yml`](.github/workflows/ci-extended.yml); every push to master, nightly and on demand; not a merge gate, failures open an issue):

| CI job | Python | Database |
| --- | --- | --- |
| Random-seed fuzz / SQLite 3 (Python 3.12) | 3.12 | SQLite bundled with CPython (`sqlite3`) |
| Random-seed fuzz / PostgreSQL 18 | 3.12 | `postgres:18` |
| Random-seed fuzz / MySQL 8.4 | 3.12 | `mysql:8.4` |
| Random-seed fuzz / MariaDB 12.3 LTS | 3.12 | `mariadb:12.3` |

Deferred compatibility sweep ([`ci-deferred.yml`](.github/workflows/ci-deferred.yml); nightly, scale-3 fuzz weekly; not a merge gate, scheduled failures open an issue):

| CI job | Python | Database |
| --- | --- | --- |
| CPython 3.13 / SQLite | 3.13 | SQLite bundled with CPython (`sqlite3`) |
| CPython 3.14 / SQLite | 3.14 | SQLite bundled with CPython (`sqlite3`) |
| SQLite3 Multiple Ciphers | 3.12 | SQLite via `sqlite3mc` |
| sqlean SQLite | 3.12 | SQLite via `sqlean` |
| PostgreSQL 16 | 3.12 | `postgres:16` |
| PostgreSQL 17 | 3.12 | `postgres:17` |
| Percona PostgreSQL 18 | 3.12 | `percona/percona-distribution-postgresql:18` |
| MySQL 26.10 Innovation | 3.12 | `mysql:26.10` |
| MariaDB 11.4 LTS | 3.12 | `mariadb:11.4` |
| MariaDB 13.0 Rolling | 3.12 | `mariadb:13.0` |
| Percona Server for MySQL 8.4 | 3.12 | `percona/percona-server:8.4` |
| Percona Server for MySQL 9.7 | 3.12 | `percona/percona-server:9.7` |
| Minimum declared dependencies / PostgreSQL 18 | 3.11 | `postgres:18` |
| Minimum declared dependencies / MySQL 9.7 | 3.11 | `mysql:9.7` |
| Scale-3 differential fuzz / SQLite 3 + transpiled PostgreSQL/MySQL | 3.12 | SQLite bundled with CPython (`sqlite3`) |
| Scale-3 differential fuzz / APSW SQLite | 3.12 | SQLite via `apsw` |
| Scale-3 differential fuzz / PostgreSQL 17 | 3.12 | `postgres:17` |
| Scale-3 differential fuzz / PostgreSQL 18 | 3.12 | `postgres:18` |
| Scale-3 differential fuzz / MySQL 8.4 | 3.12 | `mysql:8.4` |
| Scale-3 differential fuzz / MariaDB 11.4 | 3.12 | `mariadb:11.4` |
<!-- END GENERATED: ci-matrix -->
