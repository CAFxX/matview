#!/usr/bin/env python3
"""Native PostgreSQL regression for failed single-shot MV replacement rollback."""
from __future__ import annotations

import os
import uuid

import psycopg2

from matview_sync import Dialect, MaterializedViewManager


def main() -> None:
    dsn = os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is required; this test must run against native PostgreSQL")

    conn = psycopg2.connect(dsn)
    schema = "mv_replace_" + uuid.uuid4().hex[:16]
    try:
        with conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
            cur.execute(
                "CREATE TABLE replacement_rows (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)"
            )
            cur.execute(
                "CREATE TABLE replacement_control (should_fail BOOLEAN NOT NULL)"
            )
            cur.execute("INSERT INTO replacement_control VALUES (FALSE)")
            cur.execute(
                """
                CREATE FUNCTION maybe_fail(input_value INTEGER) RETURNS INTEGER
                LANGUAGE plpgsql AS $$
                BEGIN
                    IF (SELECT should_fail FROM replacement_control LIMIT 1) THEN
                        RAISE EXCEPTION 'injected replacement population failure';
                    END IF;
                    RETURN input_value;
                END;
                $$
                """
            )
            cur.execute(
                "INSERT INTO replacement_rows VALUES (1, 10), (2, 20), (3, 30)"
            )
        conn.commit()

        manager = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
        view_sql = (
            "CREATE VIEW replacement_reader AS "
            "SELECT id, maybe_fail(value) AS value FROM replacement_rows;"
        )
        plan = manager.register_view(view_sql)
        original_catalog = manager.get_catalog()[plan.view_name]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, value FROM replacement_reader ORDER BY id"
            )
            original_rows = cur.fetchall()
        assert original_rows == [(1, 10), (2, 20), (3, 30)], original_rows

        with conn.cursor() as cur:
            cur.execute("UPDATE replacement_control SET should_fail = TRUE")
        conn.commit()

        try:
            manager.register_view(view_sql, force_rebuild=True)
        except Exception as exc:
            if "injected replacement population failure" not in str(exc):
                raise AssertionError(
                    "replacement failed for an unexpected reason"
                ) from exc
            conn.rollback()
        else:
            raise AssertionError("replacement unexpectedly succeeded")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, value FROM replacement_reader ORDER BY id"
            )
            restored_rows = cur.fetchall()
        assert restored_rows == original_rows, (original_rows, restored_rows)

        restored_catalog = manager.get_catalog()[plan.view_name]
        assert (
            restored_catalog["signature_hash"]
            == original_catalog["signature_hash"]
        )

        with conn.cursor() as cur:
            for trigger_name in original_catalog["trigger_names"]:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM pg_trigger t
                    JOIN pg_class c ON c.oid = t.tgrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND t.tgname = %s AND NOT t.tgisinternal
                    """,
                    (schema, trigger_name),
                )
                assert cur.fetchone()[0] == 1, (
                    f"old maintenance trigger {trigger_name} was not restored"
                )
        conn.commit()
        # Inject a failure after the old reader has been dropped but before
        # catalog cleanup. Transactional PostgreSQL DDL must restore it.
        original_drop = manager.drop_view

        def fail_mid_drop(view_name, _commit_transaction=True):
            with conn.cursor() as cur:
                cur.execute(f'DROP VIEW IF EXISTS "{view_name}"')
            raise RuntimeError("injected mid-drop failure")

        manager.drop_view = fail_mid_drop
        try:
            try:
                manager.register_view(view_sql, force_rebuild=True)
            except RuntimeError as exc:
                assert "injected mid-drop failure" in str(exc)
                conn.rollback()
            else:
                raise AssertionError("mid-drop failure was not injected")
        finally:
            manager.drop_view = original_drop

        with conn.cursor() as cur:
            cur.execute("SELECT id, value FROM replacement_reader ORDER BY id")
            assert cur.fetchall() == original_rows
        assert manager.get_catalog()[plan.view_name]["signature_hash"] == original_catalog["signature_hash"]
        with conn.cursor() as cur:
            for trigger_name in original_catalog["trigger_names"]:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM pg_trigger t
                    JOIN pg_class c ON c.oid = t.tgrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND t.tgname = %s AND NOT t.tgisinternal
                    """,
                    (schema, trigger_name),
                )
                assert cur.fetchone()[0] == 1, (
                    f"old maintenance trigger {trigger_name} was not restored after mid-drop"
                )
        conn.commit()

        # An unrelated caller transaction must not be committed by registration.
        with conn.cursor() as cur:
            cur.execute("UPDATE replacement_control SET should_fail = FALSE")
        try:
            manager.register_view(view_sql, force_rebuild=True)
        except RuntimeError as exc:
            assert "idle PostgreSQL connection" in str(exc)
        else:
            raise AssertionError("caller-owned PostgreSQL transaction was accepted")
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT should_fail FROM replacement_control")
            assert cur.fetchone()[0] is True

        # sync_views shares the same pre-read transaction ownership guard.
        with conn.cursor() as cur:
            cur.execute("UPDATE replacement_control SET should_fail = FALSE")
        try:
            manager.sync_views([view_sql])
        except RuntimeError as exc:
            assert "idle PostgreSQL connection" in str(exc)
        else:
            raise AssertionError("sync_views accepted a caller-owned PostgreSQL transaction")
        conn.rollback()

        # PostgreSQL autocommit makes DDL non-transactional; reject before any
        # schema/catalog reads or destructive replacement statements.
        conn.autocommit = True
        try:
            try:
                manager.register_view(view_sql, force_rebuild=True)
            except RuntimeError as exc:
                assert "autocommit=False" in str(exc)
            else:
                raise AssertionError("PostgreSQL autocommit replacement was accepted")
        finally:
            conn.autocommit = False

        with conn.cursor() as cur:
            cur.execute("UPDATE replacement_control SET should_fail = FALSE")
            cur.execute("INSERT INTO replacement_rows VALUES (4, 40)")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, value FROM replacement_reader ORDER BY id"
            )
            after_write = cur.fetchall()
        assert after_write == [
            (1, 10), (2, 20), (3, 30), (4, 40)
        ], after_write
        print("PASS: PostgreSQL replacement rollback, mid-drop rollback, and transaction guards")
    finally:
        try:
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    main()