#!/usr/bin/env python3
"""Native MySQL/MariaDB regression for manager-constructor transaction ownership."""
from __future__ import annotations

import os
import uuid

import pymysql

from matview_sync import Dialect, MaterializedViewManager


def main() -> None:
    dsn = os.environ.get("MYSQL_DSN")
    if not dsn:
        raise RuntimeError("MYSQL_DSN is required; this test must run against native MySQL/MariaDB")

    from urllib.parse import urlparse

    parsed = urlparse(dsn)
    options = {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": parsed.username or "root",
        "password": parsed.password or "",
        "database": parsed.path.lstrip("/"),
        "charset": "utf8mb4",
        "autocommit": False,
    }
    conn = pymysql.connect(**options)
    observer = pymysql.connect(**options)
    suffix = uuid.uuid4().hex[:16]
    table = f"mv_constructor_owner_{suffix}"
    prefix = f"__constructor_guard_{suffix}_"
    catalog = prefix + "catalog"
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE TABLE `{table}` (id INTEGER PRIMARY KEY)")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(f"INSERT INTO `{table}` VALUES (1)")
        assert bool(conn.server_status & 0x0001), "test setup did not start a transaction"

        try:
            MaterializedViewManager(conn, dialect=Dialect.MYSQL, object_prefix=prefix)
        except RuntimeError as exc:
            assert "active caller transaction" in str(exc), str(exc)
        else:
            raise AssertionError("manager construction accepted a caller-owned MySQL transaction")

        assert bool(conn.server_status & 0x0001), "manager changed the caller transaction state"
        with observer.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{table}`")
            assert cur.fetchone()[0] == 0, "manager construction committed caller-owned DML"
            cur.execute("SHOW TABLES LIKE %s", (catalog,))
            assert cur.fetchone() is None, "manager touched catalog before rejecting caller transaction"

        conn.rollback()
        with observer.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM `{table}`")
            assert cur.fetchone()[0] == 0, "caller rollback did not undo its own DML"

        print(
            "PASS: native MySQL/MariaDB manager initialization rejects active caller "
            "transactions without executing catalog DDL or committing caller work"
        )
    finally:
        try:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS `{table}`")
            conn.commit()
        finally:
            observer.close()
            conn.close()


if __name__ == "__main__":
    main()