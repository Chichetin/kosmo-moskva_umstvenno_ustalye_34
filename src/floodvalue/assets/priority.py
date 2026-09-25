"""Inspection order for the operator (criterion 14, D-035).

A separate, declared rule next to `rank` in asset_loss.csv (which stays "1 = max EL" by
contract). Pure functions on asset_loss rows, no I/O. Rubles are never rescaled: uncertainty
and missing estimates move assets between groups, the social rule is a position constraint.
"""

from floodvalue.assets.portfolio import binary_entropy

# Operator policy, not fitted on data: with p calibrated on validation (D-029), p < 0.1 reads
# as "water (the map shows all water, not only the flood) at the asset pixel in about <= 10 % of
# such cases" -> likely dry. The mask threshold is a pixel-map choice (IoU), not this one.
P_DRY = 0.1
# High uncertainty for the current measure (binary entropy of p in bits, D-018): H(P_DRY), so
# "the model is not sure" <=> p in [0.1, 0.9] -- the same boundary, no second tuned number.
# Revisit when #21 replaces the uncertainty measure.
UNCERTAINTY_HIGH = binary_entropy(P_DRY)
# Social assets of the fixed portfolio (task.md: "социальная значимость ... отдельное
# ограничение"): people on site. Critical infrastructure (substation, water_intake, ...) carries
# its weight in V*q already; widening the list is the operator's choice.
SOCIAL_CLASSES = ("clinic", "school")
# First inspection wave: a third of the 10-asset portfolio (scenario parameter).
SOCIAL_MAX_POSITION = 3

GROUPS = ("at_risk", "not_assessed", "likely_dry")

RULE = (
    "Очерёдность осмотра — отдельный показатель; rank в asset_loss.csv (1 = max EL, тай по "
    "asset_id) не меняется. Группы по порядку: (1) at_risk — оценённые объекты с "
    f"p ≥ {P_DRY} или с неопределённостью ≥ uncertainty_high (вода или сомнение модели); "
    "(2) not_assessed — no_data/partial: p и EL нет, рублями не ранжируются, по asset_id; "
    "осмотр на месте или снимок; стоят после at_risk (там есть признаки воды) и перед "
    "likely_dry (неизвестное состояние хуже уверенного «сухо»); (3) likely_dry — оценённые с "
    f"p < {P_DRY} и неопределённостью ниже порога (или не оценённой). Внутри at_risk и "
    "likely_dry — по убыванию EL, при равном EL — большая неопределённость раньше, затем "
    "asset_id (поэтому при равном EL порядок может отличаться от rank). Неопределённость, кроме "
    "группы, задаёт действие (action): at_risk и модель не уверена — подтвердить осмотром или "
    "снимком (кандидат в план C); at_risk и модель уверена — выезд сразу. Социальное "
    f"ограничение: объекты классов {', '.join(SOCIAL_CLASSES)} из at_risk и not_assessed стоят "
    f"не ниже места {SOCIAL_MAX_POSITION} (первая волна осмотра, треть портфеля); из likely_dry "
    "не поднимаются; EL, V, q и rank при этом не меняются, рубли на коэффициент важности не "
    f"умножаются. p = {P_DRY} — граница политики оператора, не подобрана на данных: по "
    "калибровке на validation (D-029) в пикселе объекта вода (карта воды, не только затопление) "
    "примерно в ≤ 10 % таких случаев; порог маски карты для осмотра не используется. Порог "
    f"неопределённости для бинарной энтропии p (D-018) = H({P_DRY}) ≈ {UNCERTAINTY_HIGH:.3f} "
    "бит, т. е. та же граница по p; пересмотреть после замены показателя неопределённости "
    "(#21). Если порог для показателя не задан — неопределённость только тай-брейк."
)


def _num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def _rub(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def _group(r: dict, uncertainty_high: float | None) -> str:
    if r["status"] != "ok":
        return "not_assessed"
    if r["p_flood"] >= P_DRY or _uncertain(r, uncertainty_high):
        return "at_risk"
    return "likely_dry"


def _uncertain(r: dict, uncertainty_high: float | None) -> bool:
    u = r["uncertainty"]
    return uncertainty_high is not None and u is not None and u >= uncertainty_high


def _action(r: dict, group: str, uncertainty_high: float | None) -> str:
    if group == "not_assessed":
        return "нет оценки — осмотр на месте или дополнительный снимок"
    if group == "likely_dry":
        return "проверить последним"
    if _uncertain(r, uncertainty_high):
        return ("модель не уверена — подтвердить осмотром или дополнительным снимком "
                "(кандидат в план съёмки C)")
    if r["uncertainty"] is not None and uncertainty_high is not None:
        return "вода в точке почти наверняка — выезд и меры сразу"
    return "признаки воды, неопределённость не оценена — осмотр"


def _reason(r: dict, group: str, uncertainty_high: float | None,
            mask_threshold: float | None) -> str:
    if group == "not_assessed":
        missing = "нет данных в точке объекта" if r["status"] == "no_data" else "данные неполные"
        return (f"нет оценки p (статус {r['status']}: {missing}) — проверить на месте или "
                "снимком; рублями не ранжируется")
    p, u = r["p_flood"], r["uncertainty"]
    el = f"EL {_rub(r['expected_loss_rub'])} руб. (место {r['rank']} по ущербу)"
    if u is None:
        u_text = "неопределённость не оценена"
    elif uncertainty_high is None:
        u_text = f"неопределённость {_num(u)} (порог не задан, только тай-брейк)"
    elif u >= uncertainty_high:
        u_text = f"неопределённость {_num(u)} ≥ {_num(uncertainty_high)}: модель не уверена"
    else:
        u_text = f"неопределённость {_num(u)} < {_num(uncertainty_high)}"
    if group == "at_risk":
        p_text = (f"p = {_num(p)} ≥ {_num(P_DRY)}: признаки воды" if p >= P_DRY
                  else f"p = {_num(p)} < {_num(P_DRY)}")
        if mask_threshold is not None and p < mask_threshold:
            p_text += (f" (ниже порога карты {_num(mask_threshold)}: на flood_mask.tif сухо, "
                       f"но выше границы осмотра {_num(P_DRY)})")
        return f"{p_text}; {u_text}; {el}"
    return f"вероятно сухо: p = {_num(p)} < {_num(P_DRY)}, {u_text}; {el}; проверить после остальных"


def inspection_order(
    loss_rows: list[dict],
    asset_classes: dict[str, str],
    uncertainty_high: float | None = UNCERTAINTY_HIGH,
    social_max_position: int = SOCIAL_MAX_POSITION,
    mask_threshold: float | None = None,
) -> list[dict]:
    """All assets in inspection order: {asset_id, asset_class, inspection_order, group, el_rank,
    action, reason}. `loss_rows` are asset_loss rows (p_flood, expected_loss_rub, uncertainty,
    rank, status); `uncertainty_high=None` means no threshold for the uncertainty measure;
    `mask_threshold` (flood_mask.tif) is used only to explain the reason text."""
    groups = {r["asset_id"]: _group(r, uncertainty_high) for r in loss_rows}

    def key(r: dict) -> tuple:
        g = GROUPS.index(groups[r["asset_id"]])
        if r["status"] != "ok":
            return (g, 0.0, 0.0, r["asset_id"])
        u = r["uncertainty"]
        return (g, -r["expected_loss_rub"], -u if u is not None else float("inf"), r["asset_id"])

    base = sorted(loss_rows, key=key)
    base_pos = {r["asset_id"]: i for i, r in enumerate(base, start=1)}
    # Social constraint: fill positions in base order, but once the free places within the
    # first `social_max_position` equal the social assets still waiting, place the next of them.
    social = [r for r in base if asset_classes.get(r["asset_id"]) in SOCIAL_CLASSES
              and groups[r["asset_id"]] != "likely_dry"]
    social_ids = {r["asset_id"] for r in social}
    others = [r for r in base if r["asset_id"] not in social_ids]
    final: list[dict] = []
    for pos in range(1, len(base) + 1):
        slots_left = max(0, social_max_position - pos + 1)
        if social and (not others or len(social) >= slots_left
                       or base_pos[social[0]["asset_id"]] < base_pos[others[0]["asset_id"]]):
            final.append(social.pop(0))
        else:
            final.append(others.pop(0))

    out = []
    for pos, r in enumerate(final, start=1):
        aid, group = r["asset_id"], groups[r["asset_id"]]
        reason = _reason(r, group, uncertainty_high, mask_threshold)
        if aid in social_ids and pos < base_pos[aid]:
            reason = (f"социальный объект ({asset_classes[aid]}): поднят с места {base_pos[aid]} "
                      f"на {pos} — социальное ограничение (не ниже места {social_max_position}), "
                      f"рубли EL не менялись; {reason}")
        out.append({"asset_id": aid, "asset_class": asset_classes.get(aid),
                    "inspection_order": pos, "group": group, "el_rank": r["rank"],
                    "action": _action(r, group, uncertainty_high), "reason": reason})
    return out


def order_moves(before: list[dict], after: list[dict]) -> list[str]:
    """'<asset_id> <old>→<new>[ (<old group>→<new group>)]' for assets whose inspection_order
    or group changed, in the new order."""
    old = {o["asset_id"]: o for o in before}
    moves = []
    for o in sorted(after, key=lambda o: o["inspection_order"]):
        b = old[o["asset_id"]]
        if (b["inspection_order"], b["group"]) == (o["inspection_order"], o["group"]):
            continue
        text = f"{o['asset_id']} {b['inspection_order']}→{o['inspection_order']}"
        if b["group"] != o["group"]:
            text += f" ({b['group']}→{o['group']})"
        moves.append(text)
    return moves
