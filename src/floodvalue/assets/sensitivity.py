"""Sensitivity of EL to p, V, q separately (task.md "Чувствительность", criteria 14, D-028).

Pure function on an already computed portfolio: no raster I/O, no procurement/budget logic
(that is issue P18b in `floodvalue.procurement`, writing `procurement_sensitivity.csv`; export
merges both into `sensitivity.csv`, contract in docs/contracts.md and
`floodvalue.contracts.schemas.SensitivityRow`). `el_sensitivity` takes the placed `assets` (for
V, q, asset_class) and the baseline `loss_rows` produced by `assess_assets` (for p, status, rank)
and returns rows in the `sensitivity.csv` field order.

`changed_inputs_json` uses the literal keys required by `floodvalue.contracts.files
.SENSITIVITY_REQUIRED_KEYS` (`p_flood`, `asset_value_rub`, `vulnerability_coef` are this
module's share; `budget_rub`, `base_rate_rub_km2` belong to procurement/P18b) so that
`floodvalue.export.validate` finds, per required key, at least one row that changes only it
(docs/contracts.md §7a, D-014, D-028).
"""

import json

from floodvalue.assets.portfolio import Asset, expected_loss, rank_assets

SENSITIVITY_FIELDS = [
    "scenario_id",
    "strategy",
    "changed_inputs_json",
    "decision_cost_rub",
    "covered_expected_loss_rub",
    "result_status",
    "interpretation",
]

# strategy is Literal["A", "B", "C"] in the contract (floodvalue.contracts.schemas.SensitivityRow);
# these rows probe the loss model itself, not any procurement basket, so "A" (no paid data) is the
# closest fit -- D-028.
STRATEGY = "A"

# The key each Python-side parameter maps to in changed_inputs_json (contract-mandated names).
REQUIRED_KEY = {"p": "p_flood", "V": "asset_value_rub", "q": "vulnerability_coef"}

# Factors applied to the whole portfolio at once (>= 3 required per parameter; task.md example).
PORTFOLIO_FACTORS = (0.5, 0.8, 1.2, 1.5)
# Factors applied to exactly one asset, to show rank movement that a uniform factor cannot
# produce (a single k applied to every ok asset scales every EL by the same k and never
# reorders them -- reviewer skeptic 2026-09-25, D-028).
SINGLE_ASSET_DOWN_FACTOR = 0.2
SINGLE_ASSET_UP_FACTOR = 3.0

P_RULE = (
    "p' = clip(p * k, 0, 1) applied to the p already extracted from flood_probability.tif "
    "(loss_rows['p_flood']); no retraining, no new raster read. Multiplicative, not additive: "
    "it keeps the p=0 boundary fixed (a pixel with p=0 stays dry), unlike a p +/- 0.1 shift, "
    "which would invent risk at p=0. Portfolio scenarios use k in "
    f"{PORTFOLIO_FACTORS} on every ok asset; single-asset scenarios use k="
    f"{SINGLE_ASSET_DOWN_FACTOR} (leader) and k={SINGLE_ASSET_UP_FACTOR} (challenger)."
)
V_RULE = "V' = V * k (k > 0, no clip needed); same k values and scopes as p."
Q_RULE = "q' = clip(q * k, 0, 1) (q stays a damage share in [0, 1]); same k values and scopes as p."


def _baseline(assets: list[Asset], loss_rows: list[dict]) -> dict[str, dict]:
    by_id = {a.asset_id: a for a in assets}
    base = {}
    for row in loss_rows:
        aid = row["asset_id"]
        a = by_id[aid]
        base[aid] = {
            "p": row["p_flood"],
            "status": row["status"],
            "rank": row["rank"],
            "V": a.asset_value_rub,
            "q": a.vulnerability_coef,
            "asset_class": a.asset_class,
        }
    return base


def _clip01(x: float) -> float:
    return max(0.0, min(x, 1.0))


def _recompute(base: dict[str, dict], overrides: dict[str, dict]) -> dict[str, float | None]:
    els: dict[str, float | None] = {}
    for aid, b in base.items():
        if b["status"] != "ok":
            els[aid] = None
            continue
        o = overrides.get(aid, {})
        p = o.get("p", b["p"])
        v = o.get("V", b["V"])
        q = o.get("q", b["q"])
        els[aid] = expected_loss(p, v, q)
    return els


def _overrides_for(parameter: str, k: float, base: dict[str, dict], targets: list[str]) -> tuple[dict, list[str]]:
    overrides: dict[str, dict] = {}
    clipped: list[str] = []
    for aid in targets:
        b = base[aid]
        if b["status"] != "ok":
            continue
        if parameter == "p":
            raw = b["p"] * k
            new = _clip01(raw)
            if new != raw:
                clipped.append(aid)
            overrides[aid] = {"p": new}
        elif parameter == "V":
            overrides[aid] = {"V": max(0.0, b["V"] * k)}
        else:  # "q"
            raw = b["q"] * k
            new = _clip01(raw)
            if new != raw:
                clipped.append(aid)
            overrides[aid] = {"q": new}
    return overrides, clipped


def _interpretation(
    base: dict[str, dict],
    new_ranks: dict[str, int | None],
    base_total: float,
    new_total: float,
    clipped: list[str],
    highlight: str | None,
) -> str:
    parts = []
    if base_total:
        pct = (new_total - base_total) / base_total * 100
        parts.append(
            f"суммарный EL портфеля: {base_total:,.0f} -> {new_total:,.0f} руб "
            f"({pct:+.0f}%)".replace(",", " ")
        )
    else:
        parts.append("суммарный EL портфеля не определён: нет оценённых (ok) объектов")

    ok_ids = [aid for aid, b in base.items() if b["status"] == "ok"]
    base_top3 = {aid for aid in ok_ids if (base[aid]["rank"] or 99) <= 3}
    new_top3 = {aid for aid in ok_ids if (new_ranks.get(aid) or 99) <= 3}
    moves = [(aid, base[aid]["rank"], new_ranks.get(aid)) for aid in ok_ids
             if base[aid]["rank"] != new_ranks.get(aid)]

    if not moves:
        parts.append("ранжирование не изменилось")
    else:
        parts.append("топ-3 не изменился" if base_top3 == new_top3 else "топ-3 изменился")
        # On a tie (e.g. a challenger overtakes a neighbour, both move by the same margin),
        # name the asset this scenario actually changed (`highlight`, single-asset scope),
        # not whichever tied asset happens first in loss_rows order (reviewer PR #55, D-028).
        aid, old_r, new_r = max(
            moves,
            key=lambda m: (abs((m[1] or 99) - (m[2] or 99)), m[0] == highlight),
        )
        cls = base[aid]["asset_class"]
        direction = "поднялся" if (new_r or 99) < (old_r or 99) else "опустился"
        parts.append(f"{cls} ({aid}) {direction} с {old_r} на {new_r}")

    if clipped:
        parts.append(f"отсечение [0,1] сработало у: {', '.join(sorted(clipped))}")

    no_data_ids = sorted(aid for aid, b in base.items() if b["status"] != "ok")
    if no_data_ids:
        parts.append(f"no_data не участвуют: {', '.join(no_data_ids)}")

    parts.append("uncertainty и статус ok/no_data в этом сценарии не пересчитываются")
    return "; ".join(parts)


def _scenario_row(
    scenario_id: str,
    parameter: str,
    scope: str,
    k: float,
    target: str | None,
    base: dict[str, dict],
) -> dict:
    targets = [target] if scope == "single_asset" else [aid for aid, b in base.items() if b["status"] == "ok"]
    overrides, clipped = _overrides_for(parameter, k, base, targets)
    new_els = _recompute(base, overrides)
    new_ranks = rank_assets(new_els)

    base_els = {aid: (expected_loss(b["p"], b["V"], b["q"]) if b["status"] == "ok" else None)
                for aid, b in base.items()}
    base_total = sum(v for v in base_els.values() if v is not None)
    new_total = sum(v for v in new_els.values() if v is not None)

    # Exactly one of SENSITIVITY_REQUIRED_KEYS per row (docs/contracts.md §7a): the changed
    # parameter is nested under its contract key; scope/asset_id/totals/clips are sibling keys,
    # which the validator does not count against the "changed only one of the five" rule.
    changed: dict = {REQUIRED_KEY[parameter]: {"factor": k}, "scope": scope,
                     "baseline_expected_loss_rub": base_total, "portfolio_expected_loss_rub": new_total}
    if scope == "single_asset":
        changed["asset_id"] = target
    if clipped:
        changed["clipped_asset_ids"] = sorted(clipped)

    return {
        "scenario_id": scenario_id,
        "strategy": STRATEGY,
        "changed_inputs_json": json.dumps(changed, ensure_ascii=False, sort_keys=True),
        # The procurement plan is not recomputed by this function -- empty, not 0.
        "decision_cost_rub": None,
        # "covered" means EL in event_observation zones available by the deadline (contracts.md);
        # there is no zone/plan here, so this field does not apply and stays empty. The
        # recomputed portfolio EL lives in changed_inputs_json instead (D-028).
        "covered_expected_loss_rub": None,
        "result_status": "ok" if any(v is not None for v in new_els.values()) else "no_data",
        "interpretation": _interpretation(base, new_ranks, base_total, new_total, clipped, target),
    }


def el_sensitivity(assets: list[Asset], loss_rows: list[dict]) -> list[dict]:
    """Rows of `sensitivity.csv` for p, V, q varied separately (>= 3 scenarios each).

    For each parameter: PORTFOLIO_FACTORS applied to every `ok` asset at once (shows the
    portfolio-total effect; ranks are unchanged by construction whenever no value clips, since
    scaling every EL by the same k preserves order -- this is stated in `interpretation`, not
    hidden), plus two single-asset scenarios that can move ranks: the current rank-1 asset
    scaled down (SINGLE_ASSET_DOWN_FACTOR) and the highest-EL asset outside the top 3 scaled up
    (SINGLE_ASSET_UP_FACTOR), fixing the other two inputs. `no_data` assets are skipped (no p to
    scale) and listed in `interpretation`.
    """
    base = _baseline(assets, loss_rows)
    ok_by_rank = sorted((aid for aid, b in base.items() if b["status"] == "ok"),
                        key=lambda aid: base[aid]["rank"])
    leader = ok_by_rank[0] if ok_by_rank else None
    challenger = ok_by_rank[3] if len(ok_by_rank) > 3 else None

    rows = []
    for parameter in ("p", "V", "q"):
        for k in PORTFOLIO_FACTORS:
            rows.append(_scenario_row(f"{parameter}_portfolio_x{k}", parameter, "portfolio", k, None, base))
        if leader is not None:
            rows.append(_scenario_row(
                f"{parameter}_leader_x{SINGLE_ASSET_DOWN_FACTOR}", parameter, "single_asset",
                SINGLE_ASSET_DOWN_FACTOR, leader, base,
            ))
        if challenger is not None:
            rows.append(_scenario_row(
                f"{parameter}_challenger_x{SINGLE_ASSET_UP_FACTOR}", parameter, "single_asset",
                SINGLE_ASSET_UP_FACTOR, challenger, base,
            ))
    return rows
