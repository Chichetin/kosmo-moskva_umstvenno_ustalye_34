"""Kopeck arithmetic of asset_loss.csv: one implementation for procurement, assets and validate.

EL in asset_loss.csv are rounded to the kopeck (#127). Sums of them are exact decimal sums, so
files carry no float tail; the expected cost of a wrong decision R = V·q·min(p, 1−p) (D-049) is
recovered from the written EL and p, so it cannot drift from asset_loss.csv by the rounding.
"""

from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any


def ok_rows(loss_rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rows with status=ok: only they enter coverage, its denominator and R."""
    return [r for r in loss_rows if r["status"] == "ok"]


def kopeck_sum(values: Iterable[float | str | None]) -> float:
    """Exact decimal sum of kopeck amounts (floats or CSV strings); None is skipped."""
    return float(sum((Decimal(v if isinstance(v, str) else repr(v)) for v in values
                      if v is not None), Decimal(0)))


def decision_risk_from_el(el: float | str, p: float | str) -> float:
    """R from asset_loss.csv fields: V·q = EL / p, so R = EL for p ≤ 0,5 and EL·(1−p)/p for
    p > 0,5 (p = 0 has EL = 0 and R = 0)."""
    el, p = float(el), float(p)
    if p <= 0.5:
        return el
    return el * (1.0 - p) / p
