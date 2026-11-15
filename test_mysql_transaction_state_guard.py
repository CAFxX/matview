#!/usr/bin/env python3
"""Unit tests for fail-closed MySQL/MariaDB constructor transaction detection."""
from __future__ import annotations

import unittest

from matview_sync import MaterializedViewManager


class MySQLTransactionStateGuardTests(unittest.TestCase):
    def test_explicit_idle_state_is_accepted(self):
        class Connection:
            in_transaction = False

        MaterializedViewManager._require_idle_mysql_connection(Connection())

    def test_explicit_active_state_is_rejected(self):
        class Connection:
            in_transaction = True

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_mysql_connection(Connection())

    def test_native_server_status_fallback_detects_active_transaction(self):
        class Connection:
            server_status = 0x0001

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_mysql_connection(Connection())

    def test_native_server_status_fallback_accepts_idle_connection(self):
        class Connection:
            server_status = 0x0002

        MaterializedViewManager._require_idle_mysql_connection(Connection())

    def test_unknown_state_fails_closed(self):
        class Connection:
            pass

        with self.assertRaisesRegex(RuntimeError, "cannot safely determine"):
            MaterializedViewManager._require_idle_mysql_connection(Connection())

    def test_callable_transaction_state_is_supported(self):
        class Connection:
            def in_transaction(self):
                return True

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_mysql_connection(Connection())

    def test_wrapped_driver_transaction_is_not_hidden_by_idle_sqlalchemy_state(self):
        class DriverConnection:
            server_status = 0x0001

        class PoolProxy:
            driver_connection = DriverConnection()

        class SQLAlchemyConnection:
            in_transaction = False
            connection = PoolProxy()

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_mysql_connection(SQLAlchemyConnection())

    def test_wrapped_idle_driver_state_is_accepted(self):
        class DriverConnection:
            server_status = 0x0002

        class PoolProxy:
            driver_connection = DriverConnection()

        class SQLAlchemyConnection:
            in_transaction = False
            connection = PoolProxy()

        MaterializedViewManager._require_idle_mysql_connection(SQLAlchemyConnection())

    def test_wrapped_callable_driver_transaction_is_rejected(self):
        class DriverConnection:
            def in_transaction(self):
                return True

        class PoolProxy:
            driver_connection = DriverConnection()

        class SQLAlchemyConnection:
            in_transaction = False
            connection = PoolProxy()

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager._require_idle_mysql_connection(SQLAlchemyConnection())


    def test_idle_facade_cannot_hide_unknown_driver_transaction_state(self):
        class DriverConnection:
            pass

        class PoolProxy:
            driver_connection = DriverConnection()

        class SQLAlchemyConnection:
            in_transaction = False
            connection = PoolProxy()

        with self.assertRaisesRegex(RuntimeError, "every underlying driver"):
            MaterializedViewManager._require_idle_mysql_connection(
                SQLAlchemyConnection()
            )

    def test_explicit_mysql_engine_version_guards_generic_connection_wrapper(self):
        class GenericConnectionWrapper:
            in_transaction = True

            def __init__(self):
                self.cursor_calls = 0

            def cursor(self):
                self.cursor_calls += 1
                raise AssertionError("constructor must reject before SQL access")

        connection = GenericConnectionWrapper()
        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager(connection, engine_version="mysql:8.4")
        self.assertEqual(connection.cursor_calls, 0)

    def test_explicit_engine_version_overrides_conflicting_dialect_hint(self):
        from matview_sync import Dialect

        class GenericConnectionWrapper:
            in_transaction = True

            def cursor(self):
                raise AssertionError("constructor must reject before SQL access")

        with self.assertRaisesRegex(RuntimeError, "active caller transaction"):
            MaterializedViewManager(
                GenericConnectionWrapper(),
                dialect=Dialect.SQLITE,
                engine_version="mysql:8.4",
            )

    def test_explicit_mysql_engine_version_accepts_idle_generic_wrapper(self):
        from matview_sync import Dialect

        class GenericConnectionWrapper:
            in_transaction = False

            def __init__(self):
                self.statements = []

            def cursor(self):
                return Cursor(self)

            def commit(self):
                pass

            def rollback(self):
                pass

        class Cursor:
            def __init__(self, connection):
                self.connection = connection

            def execute(self, statement, params=None):
                self.connection.statements.append(statement)
                return self

            def fetchone(self):
                return (1,)

            def fetchall(self):
                return []

            def close(self):
                pass

        connection = GenericConnectionWrapper()
        manager = MaterializedViewManager(connection, engine_version="mysql:8.4")
        self.assertEqual(manager.dialect, Dialect.MYSQL)
        self.assertTrue(connection.statements)


if __name__ == "__main__":
    unittest.main(verbosity=2)