"""Tier 1, Tier 2, and Tier 3 incremental maintenance rule generators."""

from __future__ import annotations

from .join import Tier1JoinMixin
from .single import Tier1SingleMixin
from .tier2 import Tier2Mixin
from .tier3 import Tier3Mixin
from .union_all_agg import Tier1UnionAllAggMixin
from .zset import Tier1ZSetMixin


class MaintenanceMixin(
    Tier1SingleMixin,
    Tier1JoinMixin,
    Tier1ZSetMixin,
    Tier1UnionAllAggMixin,
    Tier2Mixin,
    Tier3Mixin,
):
    """Combined Tier 1, Tier 2, and Tier 3 rule emitters."""


__all__ = [
    "Tier1SingleMixin",
    "Tier1JoinMixin",
    "Tier1ZSetMixin",
    "Tier1UnionAllAggMixin",
    "Tier2Mixin",
    "Tier3Mixin",
    "MaintenanceMixin",
]
