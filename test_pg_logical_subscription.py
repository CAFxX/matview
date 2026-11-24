"""PostgreSQL logical-replication subscribers are an unsupported configuration.

The subscription apply worker fires only ROW-level triggers; matview's PostgreSQL
maintenance triggers are statement-level, so replicated INSERT/UPDATE/DELETE are
not maintained (ENABLE ALWAYS does not change that). A view over a subscription
target must therefore be refused at registration and reported ``invalid`` by
``check_schema_drift()`` (no relaxed mode exists).

Native test: needs POSTGRES_DSN (the subscriber) plus a publisher with
``wal_level = logical``: PG_LOGICAL_PUBLISHER_DSN (client connection to the
publisher) and PG_LOGICAL_PUBLISHER_CONNINFO (libpq connection string of the
publisher as seen from the subscriber server). Skipped otherwise.
"""

from __future__ import annotations

import os
import time
import uuid
import warnings

import psycopg2
import pytest

from matview import Dialect, MaterializedViewManager
from matview.engines import UnsupportedMaterializedViewError

SUB_DSN = os.environ.get("POSTGRES_DSN")
PUB_DSN = os.environ.get("PG_LOGICAL_PUBLISHER_DSN")
PUB_CONNINFO = os.environ.get("PG_LOGICAL_PUBLISHER_CONNINFO")

needs_publisher = pytest.mark.skipif(
    not (SUB_DSN and PUB_DSN and PUB_CONNINFO),
    reason="set POSTGRES_DSN, PG_LOGICAL_PUBLISHER_DSN and PG_LOGICAL_PUBLISHER_CONNINFO",
)
needs_postgres = pytest.mark.skipif(not SUB_DSN, reason="set POSTGRES_DSN")

VIEW = "SELECT g, COUNT(*) AS c, SUM(v) AS s FROM {t} GROUP BY g"


def _rows(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
        out = cur.fetchall()
    conn.commit()
    return sorted(out)


@needs_publisher
def test_subscription_target_is_refused_and_reported_invalid() -> None:
    tag = uuid.uuid4().hex[:8]
    ns, t, ctl, pub, sub = f"lr_{tag}", "tt", "ctl", f"pub_{tag}", f"sub_{tag}"
    pconn = psycopg2.connect(PUB_DSN)
    pconn.autocommit = True
    sconn = psycopg2.connect(SUB_DSN)
    sconn.autocommit = True
    subscribed = False
    slot_created = False
    try:
        with pconn.cursor() as c:
            c.execute(f"CREATE SCHEMA {ns}")
            c.execute(f"CREATE TABLE {ns}.{t} (id INT PRIMARY KEY, g INT, v INT)")
            c.execute(f"CREATE PUBLICATION {pub} FOR TABLE {ns}.{t}")
        with sconn.cursor() as c:
            c.execute(f"CREATE SCHEMA {ns}")
            c.execute(f"SET search_path = {ns}")
            c.execute(f"CREATE TABLE {t} (id INT PRIMARY KEY, g INT, v INT)")
            c.execute(f"CREATE TABLE {ctl} (id INT PRIMARY KEY, g INT, v INT)")
        sconn.autocommit = False
        manager = MaterializedViewManager(sconn, dialect=Dialect.POSTGRES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manager.register_view(f"CREATE VIEW mv_{t} AS {VIEW.format(t=t)};")
            manager.register_view(f"CREATE VIEW mv_{ctl} AS {VIEW.format(t=ctl)};")
        sconn.commit()
        assert manager.check_schema_drift() == {f"mv_{t}": "ok", f"mv_{ctl}": "ok"}
        sconn.commit()

        # The slot is created up front (create_slot = false): CREATE SUBSCRIPTION
        # would otherwise wait for it on the same connection when publisher and
        # subscriber are databases of one server (as in CI).
        with pconn.cursor() as c:
            c.execute("SELECT pg_create_logical_replication_slot(%s, 'pgoutput')", (sub,))
        slot_created = True
        sconn.autocommit = True
        with sconn.cursor() as c:
            c.execute(
                f"CREATE SUBSCRIPTION {sub} CONNECTION %s PUBLICATION {pub} "
                f"WITH (copy_data = false, create_slot = false, slot_name = '{sub}')",
                (PUB_CONNINFO,),
            )
        subscribed = True
        sconn.autocommit = False

        # Why this is refused: the replicated INSERT reaches the table but not the view.
        with pconn.cursor() as c:
            c.execute(f"INSERT INTO {ns}.{t} VALUES (1, 1, 10)")
        deadline = time.monotonic() + 30
        while not _rows(sconn, f"SELECT id FROM {t}"):
            assert time.monotonic() < deadline, "replicated row never arrived"
            time.sleep(0.05)
        assert _rows(sconn, f"SELECT g, c, s FROM mv_{t}") == []
        assert _rows(sconn, f"SELECT g, COUNT(*), SUM(v) FROM {t} GROUP BY g") == [(1, 1, 10)]  # the live view

        # The stale view is reported, never "ok"; the control view is unaffected.
        assert manager.check_schema_drift() == {f"mv_{t}": "invalid", f"mv_{ctl}": "ok"}
        sconn.commit()
        with pytest.raises(UnsupportedMaterializedViewError, match="logical-replication"):
            manager.repair_view(f"mv_{t}")
        sconn.rollback()
        # A new view over the subscription target is refused; other tables register.
        with pytest.raises(UnsupportedMaterializedViewError, match="logical-replication"):
            manager.register_view(f"CREATE VIEW mv2_{t} AS SELECT id, v FROM {t};")
        sconn.rollback()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manager.register_view(f"CREATE VIEW mv2_{ctl} AS SELECT id, v FROM {ctl};")
        sconn.commit()
    finally:
        sconn.rollback()
        sconn.autocommit = True
        with sconn.cursor() as c:
            if subscribed:
                c.execute(f"ALTER SUBSCRIPTION {sub} DISABLE")
                c.execute(f"ALTER SUBSCRIPTION {sub} SET (slot_name = NONE)")
                c.execute(f"DROP SUBSCRIPTION {sub}")
            c.execute(f"DROP SCHEMA IF EXISTS {ns} CASCADE")
        with pconn.cursor() as c:
            if slot_created:
                c.execute("SELECT pg_drop_replication_slot(slot_name) FROM pg_replication_slots WHERE slot_name = %s", (sub,))
            c.execute(f"DROP PUBLICATION IF EXISTS {pub}")
            c.execute(f"DROP SCHEMA IF EXISTS {ns} CASCADE")
        sconn.close()
        pconn.close()


@needs_postgres
def test_subscription_refusal_wiring_without_a_publisher(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI-runnable: a catalog hit for the source (faked here; a real subscription
    needs wal_level = logical, see the native test above) refuses registration
    and repair and makes check_schema_drift() report "invalid", while a clean
    catalog changes nothing."""
    ns = f"lw_{uuid.uuid4().hex[:8]}"
    conn = psycopg2.connect(SUB_DSN)
    conn.autocommit = True
    try:
        with conn.cursor() as c:
            c.execute(f"CREATE SCHEMA {ns}")
            c.execute(f"SET search_path = {ns}")
            c.execute("CREATE TABLE tt (id INT PRIMARY KEY, g INT, v INT)")
        conn.autocommit = False
        manager = MaterializedViewManager(conn, dialect=Dialect.POSTGRES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manager.register_view(f"CREATE VIEW mv AS {VIEW.format(t='tt')};")
        conn.commit()
        assert manager.check_schema_drift() == {"mv": "ok"}
        conn.commit()
        assert manager._pg_subscription_targets(["tt", "missing_table"]) == []
        conn.commit()

        monkeypatch.setattr(MaterializedViewManager, "_pg_subscription_targets", lambda self, tables: list(tables))
        assert manager.check_schema_drift() == {"mv": "invalid"}
        conn.commit()
        with pytest.raises(UnsupportedMaterializedViewError, match="logical-replication"):
            manager.repair_view("mv")
        conn.rollback()
        with pytest.raises(UnsupportedMaterializedViewError, match="logical-replication"):
            manager.register_view("CREATE VIEW mv2 AS SELECT id FROM tt;")
        conn.rollback()
    finally:
        conn.rollback()
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"DROP SCHEMA IF EXISTS {ns} CASCADE")
        conn.close()
