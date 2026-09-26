"""floodvalue.procurement — see docs/project-structure.md.

Public API for the operator interface (D-013): :func:`plan` recomputes A/B/C for a new
budget from the files of a results directory, without touching p, mask, threshold or portfolio.
"""
from pathlib import Path
from typing import Any


def plan(run_dir: Path, budget_rub: float, social: bool | None = None) -> dict[str, Any]:
    """Recompute strategies A/B/C for ``budget_rub``; keys: plans, plan_rows, comparison,
    reasons, social_constraint. ``social`` switches the C social-coverage constraint
    (None — as in configs/run.yaml)."""
    from floodvalue.procurement.run import plan as _plan
    return _plan(run_dir, budget_rub, social)


def priority_plan(run_dir: Path, budget_rub: float, asset_id: str) -> dict[str, Any]:
    """Preview a C plan that covers one assessed asset before optimizing the remaining budget."""
    from floodvalue.procurement.run import priority_plan as _priority_plan
    return _priority_plan(run_dir, budget_rub, asset_id)
