"""Цена заказа ДЗЗ по учебному сценарию ПП РФ № 840: РП = Б × К × О × П × Т × Р.

Чистые функции; коэффициенты и правила округления — только из ``configs/pricing.yaml``
(ввод-вывод — лишь в :func:`load_pricing`). Деньги и коэффициенты — ``Decimal`` с
``ROUND_HALF_UP``. Порядок округлений (task.md, «Что является правилом учебного расчёта»):
Р до 9 знаков → unit_price = round(Б × О × П × Т × Р, 0,01) → cost = round(unit × К, 0,01);
итог корзины — сумма cost строк. Трактовки — D-017 в docs/decisions.md.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Any

import yaml

from floodvalue.paths import CONFIGS_DIR

DEFAULT_CONFIG = CONFIGS_DIR / "pricing.yaml"
ACQUISITION_TYPES = ("new", "operational", "archive")


def load_pricing(path: str | Path | None = None) -> dict[str, Any]:
    """Прочитать конфиг цены (единственная функция модуля с вводом-выводом)."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dec(value: Any) -> Decimal:
    """Число → Decimal через str, чтобы 1.2 из YAML не стало 1.19999…"""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError(f"ожидалось число, получено {value!r}")
    return Decimal(str(value))


def _quant(value: Decimal, step: str) -> Decimal:
    return value.quantize(Decimal(step), rounding=ROUND_HALF_UP)


def _rounding_step(cfg: Mapping[str, Any], key: str) -> str:
    rounding = cfg["rounding"]
    if rounding.get("mode") != "ROUND_HALF_UP":
        raise ValueError(f"поддерживается только ROUND_HALF_UP, в конфиге {rounding.get('mode')!r}")
    return str(rounding[key])


def round_money(value: Decimal, step: str = "0.01") -> Decimal:
    """Округление до копеек ROUND_HALF_UP."""
    return _quant(_dec(value), step)


def _lookup(table: Mapping[str, Any], key: str, name: str) -> Decimal:
    if key not in table:
        raise ValueError(f"{name}={key!r} не из {sorted(table)}")
    return _dec(table[key])


def processing_coef(level: str, cfg: Mapping[str, Any]) -> Decimal:
    """О — коэффициент уровня обработки (L0/L1/L2)."""
    return _lookup(cfg["processing_coef"], level, "processing_level")


def usage_coef(usage: str, cfg: Mapping[str, Any]) -> Decimal:
    """П — коэффициент условий использования (internal/limited/unrestricted)."""
    return _lookup(cfg["usage_coef"], usage, "usage_type")


def _to_date(value: date | datetime | str) -> date:
    """Календарная дата в UTC: aware-datetime приводится к UTC, naive считается UTC (D-017)."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value) if len(value) > 10 else date.fromisoformat(value)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"ожидалась дата, получено {value!r}")


def existing_data_category(observed_at: date | datetime | str, as_of: date | datetime | str,
                           cfg: Mapping[str, Any]) -> str:
    """operational/archive по давности снимка: дата получения → дата расчёта, календарные дни.

    Ровно ``operational_max_age_days`` (90) — ещё operational (граница включительно).
    """
    age = (_to_date(as_of) - _to_date(observed_at)).days
    if age < 0:
        raise ValueError(f"дата снимка {observed_at} позже даты расчёта {as_of}")
    return "operational" if age <= int(cfg["freshness"]["operational_max_age_days"]) else "archive"


def freshness_coef(acquisition_type: str, guaranteed_purchase: bool,
                   observed_at: date | datetime | str | None, as_of: date | datetime | str,
                   cfg: Mapping[str, Any]) -> Decimal:
    """Т — актуальность / новая съёмка.

    ``new`` допустим только с ``guaranteed_purchase=True`` (иначе ошибка — правило не задано
    постановкой); гарантия у имеющихся данных — ошибка. Для имеющихся данных заявленная
    категория обязана совпасть с давностью по ``observed_at`` на ``as_of``.
    """
    fr = cfg["freshness"]
    if acquisition_type not in ACQUISITION_TYPES:
        raise ValueError(f"acquisition_type={acquisition_type!r} не из {ACQUISITION_TYPES}")
    if not isinstance(guaranteed_purchase, bool):
        raise TypeError(f"guaranteed_purchase должен быть bool, получено {guaranteed_purchase!r}")
    if acquisition_type == "new":
        if not guaranteed_purchase:
            raise ValueError("acquisition_type=new без guaranteed_purchase=true — "
                             "правило Т не задано постановкой")
        return _dec(fr["new_guaranteed"])
    if guaranteed_purchase:
        raise ValueError(f"guaranteed_purchase=true допустим только при acquisition_type=new, "
                         f"получено {acquisition_type}")
    if observed_at is None:
        raise ValueError(f"для {acquisition_type} нужна дата получения снимка observation_at")
    actual = existing_data_category(observed_at, as_of, cfg)
    if actual != acquisition_type:
        raise ValueError(f"заявлено {acquisition_type}, а по давности снимка {observed_at} "
                         f"на {as_of} — {actual}")
    return _dec(fr[actual])


def _sensor_family(sensor_type: str, cfg: Mapping[str, Any]) -> str:
    for family, members in cfg["discount"]["sensor_family"].items():
        if sensor_type in members:
            return family
    raise ValueError(f"sensor_type={sensor_type!r} не относится ни к одному семейству скидки")


def discount_coef(sensor_type: str, group_area_km2: Any, resolution_m: Any,
                  cfg: Mapping[str, Any]) -> Decimal:
    """Р = min(1; max(floor; slope × ln(S / r²) + intercept)), S — км², r — м; 9 знаков."""
    params = cfg["discount"][_sensor_family(sensor_type, cfg)]
    s, r = _dec(group_area_km2), _dec(resolution_m)
    if s <= 0 or r <= 0:
        raise ValueError(f"S и r должны быть положительны: S={s} км², r={r} м")
    with localcontext() as ctx:
        ctx.prec = 40
        raw = _dec(params["slope"]) * (s / (r * r)).ln() + _dec(params["intercept"])
        value = min(Decimal(1), max(_dec(params["floor"]), raw))
    return _quant(value, _rounding_step(cfg, "discount_coef"))


def discount_group_id(sensor_type: str, resolution_m: Any, year: int,
                      cfg: Mapping[str, Any]) -> str:
    """Группа скидки: семейство сенсора (optical/sar) × resolution_m × календарный год заказа.

    Ключ — семейство из ``discount.sensor_family``, а не сырое ``sensor_type``: два оптических
    сенсора одного разрешения — один «тип данных» и одна группа (D-017).
    """
    family = _sensor_family(sensor_type, cfg)
    return f"{family}_{_dec(resolution_m).normalize():f}m_{year}"


def price_basket(items: Iterable[Mapping[str, Any]], cfg: Mapping[str, Any], *,
                 as_of: date | datetime | str,
                 previous_area_km2: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Цена всей корзины одной стратегии со скидкой по группе целиком.

    ``items`` — позиции каталога с полями candidate_id, sensor_type, resolution_m,
    acquisition_type, guaranteed_purchase, observation_at, processing_level, usage_type,
    area_km2 (уже округлённая до 0,001 площадь из :mod:`geometry`). ``as_of`` — дата расчёта:
    от неё считаются давность снимков и календарный год группы. ``previous_area_km2`` —
    ранее учтённая площадь по ``discount_group_id`` (по умолчанию
    ``discount.previous_area_km2`` конфига). Возвращает строки в порядке ``items`` с
    промежуточными множителями под поля ``procurement_plan.csv`` (без ``strategy``).
    """
    items = list(items)
    ids = [it["candidate_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id в корзине повторяется")
    area_step = _rounding_step(cfg, "area_km2")
    money_step = _rounding_step(cfg, "unit_price_rub_km2")
    cost_step = _rounding_step(cfg, "cost_rub")
    base_rate = _dec(cfg["base_rate"]["rub_per_km2"])
    min_area = _dec(cfg["min_area_km2"])
    default_prev = _dec(cfg["discount"].get("previous_area_km2", 0))
    previous = previous_area_km2 or {}
    year = _to_date(as_of).year

    prepared = []
    for it in items:
        area = _quant(_dec(it["area_km2"]), area_step)
        if area < min_area:
            raise ValueError(f"{it['candidate_id']}: area_km2={area} < {min_area} км²")
        if _dec(it["resolution_m"]) <= 0:
            raise ValueError(f"{it['candidate_id']}: resolution_m должно быть > 0")
        gid = discount_group_id(it["sensor_type"], it["resolution_m"], year, cfg)
        prepared.append((it, area, gid))

    basket_area: dict[str, Decimal] = {}
    for _, area, gid in prepared:
        basket_area[gid] = basket_area.get(gid, Decimal(0)) + area
    group_area: dict[str, Decimal] = {}
    for gid, area_sum in basket_area.items():
        prev = _dec(previous.get(gid, default_prev))
        if prev < 0:
            raise ValueError(f"previous_area_km2 для {gid} отрицательна")
        group_area[gid] = _quant(area_sum + prev, area_step)

    rows = []
    for it, area, gid in prepared:
        o = processing_coef(it["processing_level"], cfg)
        p = usage_coef(it["usage_type"], cfg)
        t = freshness_coef(it["acquisition_type"], it["guaranteed_purchase"],
                           it.get("observation_at"), as_of, cfg)
        r = discount_coef(it["sensor_type"], group_area[gid], it["resolution_m"], cfg)
        unit = _quant(base_rate * o * p * t * r, money_step)
        cost = _quant(unit * area, cost_step)
        rows.append({
            "candidate_id": it["candidate_id"],
            "area_km2": area,
            "base_rate_rub_km2": base_rate,
            "base_rate_status": cfg["base_rate"]["status"],
            "processing_level": it["processing_level"],
            "usage_type": it["usage_type"],
            "guaranteed_purchase": it["guaranteed_purchase"],
            "processing_coef": o,
            "usage_coef": p,
            "freshness_coef": t,
            "discount_coef": r,
            "discount_group_id": gid,
            "group_area_km2": group_area[gid],
            "unit_price_rub_km2": unit,
            "cost_rub": cost,
            "formula": cfg["formula"],
            "legal_edition": cfg["legal_edition"],
            "sensor_type": it["sensor_type"],
            "resolution_m": it["resolution_m"],
            "acquisition_type": it["acquisition_type"],
        })
    return rows


def basket_total(rows: Iterable[Mapping[str, Any]]) -> Decimal:
    """Итог стратегии — сумма уже округлённых cost_rub строк."""
    return sum((_dec(r["cost_rub"]) for r in rows), Decimal("0.00"))
