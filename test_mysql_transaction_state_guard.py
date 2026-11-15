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


if __name__ == "__main__":
    unittest.main(verbosity=2)