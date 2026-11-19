#!/usr/bin/env python3
"""Fail CI when a database service does not match its requested engine matrix target."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse


def check_postgres_version(
    version_text: str,
    version_num: int,
    expected_major: int | str | None = None,
    label: str = "",
    requested_image: str = "",
    available_extensions: set[str] | list[str] | None = None,
) -> int:
    """Verify PostgreSQL server version number and optional required extensions."""
    major = version_num // 10000

    expected = str(expected_major).strip() if expected_major is not None and str(expected_major).strip() else None
    if expected is None:
        context = f"{label} {requested_image}".strip()
        matches = re.findall(r"\b(1[4-9]|[2-9]\d)\b", context)
        if not matches:
            matches = re.findall(r":(\d+)", requested_image)
        if not matches:
            raise RuntimeError(f"cannot derive expected PostgreSQL major from matrix label {label!r} or image {requested_image!r}")
        expected = matches[-1]

    if major != int(expected):
        raise RuntimeError(
            f"requested PostgreSQL major {expected}, but server reports major {major}: {version_text}"
        )

    check_label = f"{label} {requested_image}".lower()
    extension = (
        "timescaledb" if "timescaledb" in check_label else
        "postgis" if "postgis" in check_label else
        "vector" if "pgvector" in check_label else None
    )
    if extension and available_extensions is not None:
        if extension not in available_extensions:
            raise RuntimeError(
                f"image was expected to provide PostgreSQL extension {extension!r}, "
                f"but it is not available: {version_text}"
            )

    return major


def verify_postgres() -> None:
    import psycopg2

    dsn = os.environ["POSTGRES_DSN"]
    label = os.environ.get("EXPECTED_PG_LABEL", "")
    requested_image = os.environ.get("MATVIEW_CI_REQUESTED_IMAGE", "")
    actual_image = os.environ.get("MATVIEW_CI_ACTUAL_IMAGE", requested_image or "unspecified")
    expected_major = os.environ.get("EXPECTED_PG_MAJOR")

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version(), current_setting('server_version_num')::integer")
            row = cur.fetchone()
            assert row is not None
            version_text, version_num = row

            check_label = f"{label} {requested_image}".lower()
            extension = (
                "timescaledb" if "timescaledb" in check_label else
                "postgis" if "postgis" in check_label else
                "vector" if "pgvector" in check_label else None
            )
            available_exts = None
            if extension:
                cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = %s)",
                    (extension,),
                )
                ext_row = cur.fetchone()
                assert ext_row is not None
                available_exts = {extension} if ext_row[0] else set()

            major = check_postgres_version(
                version_text=version_text,
                version_num=version_num,
                expected_major=expected_major,
                label=label,
                requested_image=requested_image,
                available_extensions=available_exts,
            )

    fallback = os.environ.get("MATVIEW_CI_FALLBACK", "false").lower() == "true"
    if fallback:
        print(
            f"::notice::Fallback image used: requested={requested_image}, actual={actual_image}; "
            "the running product/version was checked against the matrix target."
        )
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            with open(step_summary, "a", encoding="utf-8") as summary:
                summary.write(
                    f"**Fallback image used.** Requested: {requested_image}; actual: {actual_image}. "
                    f"Verified runtime: PostgreSQL {major}.\n"
                )

    print(f"[NATIVE ENGINE VERIFIED] PostgreSQL major={major}; server={version_text}")


def _mysql_version_pair(version: str, product: str) -> tuple[int, int]:
    combined = f"{version} {product}".lower()
    if "mariadb" in combined:
        match = re.search(r"(\d+)\.(\d+)(?:\.\d+)?-MariaDB", version, re.IGNORECASE)
        if not match:
            match = re.search(r"(\d+)\.(\d+)", version)
    else:
        match = re.search(r"(\d+)\.(\d+)", version)
    if not match:
        raise RuntimeError(f"cannot parse database server version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def check_mysql_version(
    version: str,
    comment: str,
    expected_product: str | None = None,
    expected_version: str | tuple[int, int] | None = None,
    requested_image: str = "",
    label: str = "",
) -> tuple[str, tuple[int, int]]:
    """Verify MySQL/MariaDB/Percona product identity and major.minor version pair."""
    actual_text = f"{version} {comment}".lower()

    prod = (expected_product or "").strip().lower()
    ver = expected_version

    hint = f"{label} {requested_image}".lower()
    if not prod:
        if "mariadb" in hint:
            prod = "mariadb"
        elif "percona" in hint:
            prod = "percona"
        elif "mysql" in hint:
            prod = "mysql"
        else:
            raise RuntimeError(f"unsupported or unspecified expected MySQL-family product: {expected_product!r}")

    if prod == "mariadb":
        product_matches = "mariadb" in actual_text
    elif prod == "percona":
        product_matches = "percona" in actual_text
    elif prod == "mysql":
        product_matches = "mysql" in actual_text and "mariadb" not in actual_text and "percona" not in actual_text
    else:
        raise RuntimeError(f"unsupported expected MySQL-family product: {prod!r}")

    if not product_matches:
        raise RuntimeError(
            f"requested {prod}, but server identifies as {version!r} / {comment!r}"
        )

    actual_pair = _mysql_version_pair(version, comment)

    if ver is None or ver == "":
        m = re.search(r":(\d+\.\d+)", requested_image) or re.search(r"\b(\d+\.\d+)\b", label)
        if m:
            ver = m.group(1)
        else:
            raise RuntimeError(
                f"unspecified expected MySQL version and cannot derive from image {requested_image!r} or label {label!r}"
            )

    if isinstance(ver, tuple):
        expected_pair = ver
    else:
        expected_pair = tuple(int(part) for part in str(ver).split(".")[:2])

    if actual_pair != expected_pair:
        raise RuntimeError(
            f"requested {prod} {expected_pair[0]}.{expected_pair[1]}, "
            f"but server reports {version!r} / {comment!r}"
        )

    return prod, actual_pair


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
            row = cur.fetchone()
            assert row is not None
            version, comment = row

    expected_product = os.environ.get("EXPECTED_MYSQL_PRODUCT")
    expected_version = os.environ.get("EXPECTED_MYSQL_VERSION")
    requested_image = os.environ.get("MATVIEW_CI_REQUESTED_IMAGE", "")
    actual_image = os.environ.get("MATVIEW_CI_ACTUAL_IMAGE", requested_image or "unspecified")
    label = os.environ.get("EXPECTED_MYSQL_LABEL", "")

    verified_prod, actual_pair = check_mysql_version(
        version=version,
        comment=comment,
        expected_product=expected_product,
        expected_version=expected_version,
        requested_image=requested_image,
        label=label,
    )

    fallback = os.environ.get("MATVIEW_CI_FALLBACK", "false").lower() == "true"
    if fallback:
        print(
            f"::notice::Fallback image used: requested={requested_image}, actual={actual_image}; "
            "the running product/version was checked against the matrix target."
        )
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            with open(step_summary, "a", encoding="utf-8") as summary:
                summary.write(
                    f"**Fallback image used.** Requested: {requested_image}; actual: {actual_image}. "
                    f"Verified runtime: {verified_prod} {actual_pair[0]}.{actual_pair[1]}.\n"
                )

    print(
        f"[NATIVE ENGINE VERIFIED] product={verified_prod}; "
        f"version={version}; comment={comment}; image={actual_image}"
    )


if __name__ == "__main__":
    if os.environ.get("POSTGRES_DSN"):
        verify_postgres()
    elif os.environ.get("MYSQL_DSN"):
        verify_mysql()
    else:
        raise SystemExit("Set POSTGRES_DSN or MYSQL_DSN to verify a native database service.")