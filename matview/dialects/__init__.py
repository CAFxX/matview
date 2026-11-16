"""Dialect code generation mixins and Jinja2 procedural templates."""

from __future__ import annotations

from .base import DialectBaseMixin
from .emitters import DialectEmitMixin
from .mysql import MySQLDialectMixin
from .postgres import PostgresDialectMixin
from .sqlite import SQLiteDialectMixin
from .templates import PROCEDURAL_SQL_TEMPLATES, _PROCEDURAL_TEMPLATE_ENV, render_procedural_sql


class DialectMixin(
    DialectBaseMixin,
    SQLiteDialectMixin,
    PostgresDialectMixin,
    MySQLDialectMixin,
    DialectEmitMixin,
):
    """Combined dialect code generation capabilities."""


__all__ = [
    "DialectBaseMixin",
    "SQLiteDialectMixin",
    "PostgresDialectMixin",
    "MySQLDialectMixin",
    "DialectEmitMixin",
    "DialectMixin",
    "PROCEDURAL_SQL_TEMPLATES",
    "_PROCEDURAL_TEMPLATE_ENV",
    "render_procedural_sql",
]
