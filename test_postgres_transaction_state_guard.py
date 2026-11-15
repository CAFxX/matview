#!/usr/bin/env python3
"""Unit tests for PostgreSQL transaction ownership across wrapped DB-API drivers."""
from __future__ import annotations

import unittest

from matview_sync import Dialect, MaterializedViewManager


class PostgresTransactionStateGuardTests(unittest.TestCase):
    def test_wrapped_driver_active_transaction_overrides_idle_facade(self):
        class Driver:
            autocommit = False
            in_transaction = True

        class PoolProxy:
            def __init__(self):
                self.driver_connection = Driver()

        class SQLAlchemyConnection:
            def __init__(self):
                self.connection = PoolProxy()

            def in_transaction(self):
                return False

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_postgres_connection(SQLAlchemyConnection())

    def test_idle_facade_with_unknown_driver_state_fails_closed(self):
        class UnknownDriver:
            autocommit = False

        class PoolProxy:
            driver_connection = UnknownDriver()

        class SQLAlchemyConnection:
            autocommit = False
            connection = PoolProxy()

            def in_transaction(self):
                return False

        with self.assertRaisesRegex(RuntimeError, "every physical DB-API connection"):
            MaterializedViewManager._require_idle_postgres_connection(SQLAlchemyConnection())

    def test_idle_facade_with_unknown_driver_autocommit_fails_closed(self):
        class Driver:
            in_transaction = False

        class PoolProxy:
            driver_connection = Driver()

        class SQLAlchemyConnection:
            autocommit = False
            connection = PoolProxy()

            def in_transaction(self):
                return False

        with self.assertRaisesRegex(RuntimeError, "autocommit=False"):
            MaterializedViewManager._require_idle_postgres_connection(SQLAlchemyConnection())

    def test_wrapped_idle_driver_is_accepted(self):
        class Driver:
            autocommit = False
            in_transaction = False

        class PoolProxy:
            driver_connection = Driver()

        class SQLAlchemyConnection:
            connection = PoolProxy()

            def in_transaction(self):
                return False

        MaterializedViewManager._require_idle_postgres_connection(SQLAlchemyConnection())

    def test_unknown_transaction_state_fails_closed(self):
        class Connection:
            autocommit = False

        with self.assertRaisesRegex(RuntimeError, "transaction state on every physical DB-API connection"):
            MaterializedViewManager._require_idle_postgres_connection(Connection())

    def test_driver_autocommit_is_checked_even_when_facade_is_idle(self):
        class Driver:
            autocommit = True
            in_transaction = False

        class PoolProxy:
            driver_connection = Driver()

        class SQLAlchemyConnection:
            connection = PoolProxy()

            def in_transaction(self):
                return False

        with self.assertRaisesRegex(RuntimeError, "autocommit=False"):
            MaterializedViewManager._require_idle_postgres_connection(SQLAlchemyConnection())

    def test_psycopg2_transaction_status_fallback(self):
        class Connection:
            autocommit = False

            def get_transaction_status(self):
                return 2

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_postgres_connection(Connection())

    def test_explicit_postgres_engine_version_guards_generic_wrapper(self):
        class GenericConnectionWrapper:
            autocommit = False
            in_transaction = True

            def cursor(self):
                raise AssertionError("constructor must reject before SQL access")

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager(
                GenericConnectionWrapper(), engine_version="postgres:16"
            )

    def test_explicit_engine_version_overrides_conflicting_dialect_hint(self):
        class GenericConnectionWrapper:
            autocommit = False
            in_transaction = True

            def cursor(self):
                raise AssertionError("constructor must reject before SQL access")

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager(
                GenericConnectionWrapper(),
                dialect=Dialect.SQLITE,
                engine_version="postgres:16",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
