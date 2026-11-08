#!/usr/bin/env python3
"""Fail CI when a database service does not match its requested engine matrix target."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse


def verify_postgres() -> None:
    import psycopg2

    dsn = os.environ["POSTGRES_DSN"]
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version(), current_setting('server_version_num')::integer")
            version_text, version_num = cur.fetchone()
            major = version_num // 10000

            expected = os.environ.get("EXPECTED_PG_MAJOR")
            if expected is None:
                label = os.environ.get("EXPECTED_PG_LABEL", "")
                matches = re.findall(r"\b(1[4-8])\b", label)
                if not matches:
                    raise RuntimeError(f"cannot derive expected PostgreSQL major from matrix label {label!r}")
                expected = matches[-1]
            if major != int(expected):
                raise RuntimeError(
                    f"requested PostgreSQL major {expected}, but server reports major {major}: {version_text}"
                )

            label = os.environ.get("EXPECTED_PG_LABEL", "").lower()
            extension = (
                "timescaledb" if "timescaledb" in label else
                "postgis" if "postgis" in label else
                "vector" if "pgvector" in label else None
            )
            if extension:
                cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = %s)",
                    (extension,),
                )
                if not cur.fetchone()[0]:
                    raise RuntimeError(
                        f"image was expected to provide PostgreSQL extension {extension!r}, "
                        f"but it is not available: {version_text}"
                    )

    print(f"[NATIVE ENGINE VERIFIED] PostgreSQL major={major}; server={version_text}")


def _mysql_version_pair(version: str, product: str) -> tuple[int, int]:
    combined = f"{version} {product}".lower()
    if "mariadb" in combined:
        match = re.search(r"(\d+)\.(\d+)(?:\.\d+)?-MariaDB", version, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+)\.(\d+)", version)
    else:
        match = re.match(r"(\d+)\.(\d+)", version)
    if not match:
        raise RuntimeError(f"cannot parse database server version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def verify_mysql() -> None:
    import pymysql

    dsn = urlparse(os.environ["MYSQL_DSN"])
    with pymysql.connect(
        host=dsn.hostname or "127.0.0.1",
        port=dsn.port or 3306,
        user=dsn.username or "root",
        password=dsn.password or "",
        database=dsn.path.lstrip("/") or None,
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION(), @@version_comment")
            version, comment = cur.fetchone()

    actual_text = f"{version} {comment}".lower()
    expected_product = os.environ["EXPECTED_MYSQL_PRODUCT"].lower()
    if expected_product == "mariadb":
        product_matches = "mariadb" in actual_text
    elif expected_product == "percona":
        product_matches = "percona" in actual_text
    elif expected_product == "mysql":
        product_matches = "mysql" in actual_text and "mariadb" not in actual_text and "percona" not in actual_text
    else:
        raise RuntimeError(f"unsupported expected MySQL-family product: {expected_product!r}")
    if not product_matches:
        raise RuntimeError(
            f"requested {expected_product}, but server identifies as {version!r} / {comment!r}"
        )

    actual_pair = _mysql_version_pair(version, comment)
    expected_pair = tuple(int(part) for part in os.environ["EXPECTED_MYSQL_VERSION"].split(".")[:2])
    if actual_pair != expected_pair:
        raise RuntimeError(
            f"requested {expected_product} {expected_pair[0]}.{expected_pair[1]}, "
            f"but server reports {version!r} / {comment!r}"
        )

    fallback = os.environ.get("MATVIEW_CI_FALLBACK", "false").lower() == "true"
    requested_image = os.environ.get("MATVIEW_CI_REQUESTED_IMAGE", "unspecified")
    actual_image = os.environ.get("MATVIEW_CI_ACTUAL_IMAGE", "unspecified")
    if fallback:
        print(
            f"::notice::Fallback image used: requested={requested_image}, actual={actual_image}; "
            "the running product/version was checked against the matrix target."
        )
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(
                f"**Fallback image used.** Requested: {requested_image}; actual: {actual_image}. "
                f"Verified runtime: {expected_product} {actual_pair[0]}.{actual_pair[1]}.\n"
            )

    print(
        f"[NATIVE ENGINE VERIFIED] product={expected_product}; "
        f"version={version}; comment={comment}; image={actual_image}"
    )


if __name__ == "__main__":
    if os.environ.get("POSTGRES_DSN"):
        verify_postgres()
    elif os.environ.get("MYSQL_DSN"):
        verify_mysql()
    else:
        raise SystemExit("Set POSTGRES_DSN or MYSQL_DSN to verify a native database service.")