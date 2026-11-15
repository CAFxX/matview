#!/usr/bin/env python3
"""Unit tests for PostgreSQL transaction ownership across wrapped DB-API drivers."""
from __future__ import annotations

import unittest

from matview_sync import MaterializedViewManager


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
            MaterializedViewManager._require_idle_postgres_connection(
                SQLAlchemyConnection()
            )

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

        MaterializedViewManager._require_idle_postgres_connection(
            SQLAlchemyConnection()
        )

    def test_unknown_transaction_state_fails_closed(self):
        class Connection:
            autocommit = False

        with self.assertRaisesRegex(RuntimeError, "cannot safely determine"):
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
            MaterializedViewManager._require_idle_postgres_connection(
                SQLAlchemyConnection()
            )

    def test_psycopg2_transaction_status_fallback(self):
        class Connection:
            autocommit = False

            def get_transaction_status(self):
                return 2

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_postgres_connection(Connection())


if __name__ == "__main__":
    unittest.main(verbosity=2)
