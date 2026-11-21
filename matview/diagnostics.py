"""Structured diagnostics for maintenance-strategy decisions.

A diagnostic tells the user which maintenance strategy a view got and why,
whenever matview cannot use the fast exact algebraic maintenance:

* exact result required (the default): ``exact result required: falling back
  to strategy <X>`` -- matview keeps the result equal to the original view by
  a more expensive strategy;
* relaxed result allowed (an explicit per-code opt-in): ``relaxed result
  allowed: using strategy <Y>`` -- matview uses a faster strategy whose result
  may differ from the original view as documented.

Strategy names are defined once here (``Strategy``) and used by the planner,
the diagnostics, the CLI output, the README and the tests. The registry is
open: new strategies and new diagnostic codes are added here.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Sequence, Tuple

_LOG = logging.getLogger("matview")

EXACT_LEAD = "exact result required: falling back to strategy"
RELAXED_LEAD = "relaxed result allowed: using strategy"


class Strategy(str, Enum):
    """Human-readable maintenance strategy names (shared vocabulary)."""

    FULL_REFRESH = "full refresh of the original query"
    SCOPED_REFRESH = "scoped refresh of the affected groups"
    ALGEBRAIC_DELTA = "algebraic delta (add/subtract)"
    GROUP_SCOPED_RECOMPUTE = "group-scoped recompute"
    # Relaxed single-table float aggregates: plan.strategy algebraic_delta,
    # the order-sensitive columns natively recomputed per group after the lock.
    ALGEBRAIC_DELTA_NATIVE_RECOMPUTE = "algebraic delta with post-lock native group recompute"


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"


# Diagnostic codes (open registry).
FLOAT_EXACT_REFRESH = "FLOAT_EXACT_REFRESH"
FLOAT_RELAXED_MAINTENANCE = "FLOAT_RELAXED_MAINTENANCE"


@dataclass(frozen=True)
class Diagnostic:
    """One diagnostic for one view.

    ``columns`` and ``expressions`` name the output columns and aggregate
    expressions that triggered it; ``strategy`` is the strategy in effect;
    ``message`` starts with ``EXACT_LEAD`` or ``RELAXED_LEAD``."""

    code: str
    severity: DiagnosticSeverity
    view: str
    strategy: Strategy
    columns: Tuple[str, ...]
    expressions: Tuple[str, ...]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["severity"] = self.severity.value
        out["strategy"] = self.strategy.value
        out["columns"] = list(self.columns)
        out["expressions"] = list(self.expressions)
        return out

    def __str__(self) -> str:
        # The lead phrase comes first; code and view follow.
        return f"{self.message} [{self.code}; view {self.view}]"


class MatviewDiagnosticWarning(UserWarning):
    """Python warning carrying a ``Diagnostic`` (``warning.diagnostic``)."""

    def __init__(self, diagnostic: Diagnostic) -> None:
        super().__init__(str(diagnostic))
        self.diagnostic = diagnostic


def exact_fallback_message(strategy: Strategy, reason: str, cost: str, remedy: str) -> str:
    return f"{EXACT_LEAD} {strategy.value}. Why: {reason} Cost: {cost} To change it: {remedy}"


def relaxed_message(strategy: Strategy, contract: str, remedy: str) -> str:
    return f"{RELAXED_LEAD} {strategy.value}. Contract: {contract} To return to exact: {remedy}"


def emit(diagnostics: Iterable[Diagnostic], silence: Sequence[str] = (), stacklevel: int = 3) -> None:
    """Emit each diagnostic as a MatviewDiagnosticWarning (and a WARNING
    record on the ``matview`` logger) unless its code is in
    ``silence`` (the per-code suppression for applications that already
    decided; the global ``ignore_limitations`` never silences diagnostics)."""
    silenced = set(silence or ())
    for d in diagnostics:
        if d.code not in silenced:
            if _LOG.hasHandlers():  # no duplicate via logging's last-resort stderr handler
                _LOG.warning("%s", d)
            warnings.warn(MatviewDiagnosticWarning(d), stacklevel=stacklevel)
