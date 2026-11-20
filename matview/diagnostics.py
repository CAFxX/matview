"""Typed diagnostics for maintenance fallbacks and relaxed (opt-in) modes.

Every fallback from the preferred exact maintenance strategy, and every
opt-in mode that relaxes exactness, is announced with a fixed lead phrase so
that users and tests can find it:

* ``exact result required: falling back to strategy <strategy>: ...`` for a
  fallback that keeps results exact at a higher maintenance cost;
* ``relaxed result allowed: using strategy <strategy>: ...`` for a mode the
  caller enabled that gives up exactness.

A diagnostic is emitted as a :class:`MaintenanceFallbackWarning` when a view
is compiled and is kept in ``plan.diagnostics.fallbacks``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict

FALLBACK_LEAD = "exact result required: falling back to strategy"
RELAXED_LEAD = "relaxed result allowed: using strategy"


class MaintenanceFallbackWarning(UserWarning):
    """Emitted when a view is maintained by a fallback or relaxed mode."""


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str  # "info" (exact fallback) or "warning" (relaxed exactness)
    view: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "severity": self.severity, "view": self.view, "message": self.message}


def fallback_message(strategy: str, reason: str, cost: str, how_to_change: str) -> str:
    return f"{FALLBACK_LEAD} {strategy}: {reason}. Cost: {cost}. To change: {how_to_change}."


def relaxed_message(strategy: str, reason: str, consequence: str, how_to_change: str) -> str:
    return f"{RELAXED_LEAD} {strategy}: {reason}. Consequence: {consequence}. To change: {how_to_change}."


def emit(diagnostic: Diagnostic) -> None:
    warnings.warn(f"[{diagnostic.code}] view '{diagnostic.view}': {diagnostic.message}", MaintenanceFallbackWarning, stacklevel=3)
