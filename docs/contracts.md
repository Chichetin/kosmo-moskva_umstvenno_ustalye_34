# Контракты (черновик v0)

Единственная точка стыковки модулей. Код схем — `src/floodvalue/contracts/` (pydantic v2,
D-004). Изменение — отдельный PR с перечнем потребителей (`docs/project-structure.md`),
merge после вердикта `reviewer`, проверившего каждого. Черновик фиксируется issue
«contracts v1» к сквозному пути; до этого поля ниже — ориентир.

## 1. Каталог результатов одного запуска

`$FLOODVALUE_OUTPUT_DIR/<run_id>/` (по умолчанию `outputs/<run_id>/`), `run_id` =
`<chip_id>_<budget>_<короткий commit>`. Один комплект = один чип + один портфель +
один бюджет. Новый бюджет — новый комплект; растры копируются без пересчёта.

## 2. Растры (сетка = S1-чип: CRS, transform, размер)

| Файл | Тип | Значения | nodata |
|---|---|---|---|
| `flood_probability.tif` | Float32, 1 канал | p ∈ [0, 1] основного метода | −9999 |
| `flood_mask.tif` | UInt8, 1 канал | 1 вода, 0 не вода; 1 ⇔ p ≥ `threshold` из `run_metadata.json` | 255 |
| `uncertainty.tif` (доп.) | Float32 | показатель неопределённости, смысл — `run_metadata.json` | −9999 |
| `baseline_mask.tif` (доп.) | UInt8 | маска baseline | 255 |

Невалидный пиксель: S1 не конечен (NaN) в любом канале. Метка −1 на растры не влияет.

## 3. Интерфейс «метод → p-растр»

```python
def predict_proba(s1_db: np.ndarray) -> np.ndarray:
    """s1_db: float32 (2, H, W), каналы [VV, VH] в dB, NaN = нет данных.
    Возвращает float32 (H, W): p ∈ [0, 1], NaN там, где нет данных."""
```

- Реализуют: `baseline` (временно как основной — p = сигмоида от расстояния до порога,
  D-009), `model.pixel`, `model.unet`. Калибратор `calibration` — функция `p_raw → p` на том же интерфейсе.
- Нормализация, веса, порог — из `models/<name>/` (`config.json` с sha256 весов).
- Работает на CPU, только S1 (режим S1-only обязателен).

## 4. Split — `splits/split_v1.json`

```json
{"version": 1, "criterion": "event", "seed": 20260925,
 "source": "Sen1Floods11 v1.1 HandLabeled",
 "train": ["Ghana_103272", "..."], "validation": ["..."], "test": ["..."],
 "events": {"train": ["Ghana", "..."], "validation": ["..."], "test": ["..."]},
 "excluded": {"<chip_id>": "причина"},
 "sha256": "<sha256 от канонического JSON списков train/validation/test>"}
```

Хеш — в `docs/decisions.md`. Test-чипы читает только `eval-test`.

## 5. Конфиги

- `configs/pricing.yaml` — формула ПП 840: редакция, дата проверки, Б и её статус,
  таблицы О/П/Т, параметры скидки для optical/sar, правила округления.
- `configs/assets.yaml` — 10 типов: `asset_class`, `asset_value_rub`, `vulnerability_coef`.
- `configs/run.yaml` — демо-чип, seed портфеля, бюджет, `decision_at`, `decision_deadline`.

## 6. Обязательные файлы (поля — дословно `task.md`, раздел III)

Общее: CSV — UTF-8, `,`, десятичная точка, пусто вместо NaN/Inf; JSON/GeoJSON без
NaN/Inf; GeoJSON — WGS 84, [долгота, широта]. Дополнительные поля разрешены.

| Файл | Обязательные поля / структура | Ключевые проверки валидатора |
|---|---|---|
| `assets.geojson` | FeatureCollection из 10 Point; `asset_id`, `chip_id`, `asset_class`, `asset_value_rub`, `vulnerability_coef` | ровно 10, по одному типу, V/q = `configs/assets.yaml`, точка внутри чипа |
| `assets.csv` | `asset_id`, `chip_id`, `asset_class`, `asset_value_rub`, `vulnerability_coef`, `longitude`, `latitude` | совпадает с geojson |
| `asset_loss.csv` | `asset_id`, `chip_id`, `p_flood`, `expected_loss_rub`, `uncertainty`, `rank`, `status` (ok/partial/no_data) | EL = p·V·q; p = значение растра в пикселе объекта; `rank` 1 = max EL, тай — по `asset_id`; `partial`/`no_data` → пустые p/EL/rank |
| `candidate_orders.geojson` | Polygon; `candidate_id`, `sensor_type`, `resolution_m`, `acquisition_type` (new/operational/archive), `data_role` (event_observation/context), `observation_at`, `available_at`, `processing_level` (L0/L1/L2), `usage_type` (internal/limited/unrestricted), `guaranteed_purchase`, `area_km2`, `base_rate_rub_km2`, `base_rate_status`, `base_rate_source` | valid, без дыр, площадь EPSG:6933 ≥ 1 км² до округления, `area_km2` = ROUND_HALF_UP(0,001), уникальные id, существующие снимки → `guaranteed_purchase=false`; `guaranteed_purchase=true` только при `acquisition_type=new`; `new` без гарантии — ошибка |
| `procurement_plan.csv` | `strategy`, `candidate_id`, `area_km2`, `base_rate_rub_km2`, `base_rate_status`, `processing_level`, `usage_type`, `guaranteed_purchase`, `processing_coef`, `usage_coef`, `freshness_coef`, `discount_coef`, `discount_group_id`, `group_area_km2`, `unit_price_rub_km2`, `cost_rub`, `formula`, `legal_edition`; доп.: `sensor_type`, `resolution_m`, `acquisition_type` (критерий 17) | уникальна пара (strategy, candidate_id); нет строк A; категории = каталог; цена пересчитывается; скидка по группе целиком |
| `strategy_plans.json` | `{"A": [], "B": [...], "C": [...]}` | = строкам плана; id из каталога |
| `strategy_comparison.csv` | `strategy`, `data_cost_rub`, `other_cost_rub`, `decision_cost_rub`, `budget_rub`, `budget_feasible`, `covered_expected_loss_rub`, `coverage_share`, `residual_uncertainty`, `uncertainty_status`, `uncertainty_basis` | data_cost = сумма плана; decision = data + other; C в бюджете; B сверх — `budget_feasible=false` = контрфактическая; охват уникален по `asset_id` через `covers` и только по зонам `data_role=event_observation` с `available_at ≤ decision_deadline`; `coverage_share` пуст при нулевом знаменателе (причина в `run_metadata`); `uncertainty_status` ∈ baseline/scenario/measured/not_estimated; A — `baseline`; `not_estimated` → пустое число |
| `sensitivity.csv` | `scenario_id`, `strategy`, `changed_inputs_json`, `decision_cost_rub`, `covered_expected_loss_rub`, `result_status`, `interpretation` | p, V, q по отдельности; бюджет; ставка; JSON валиден |
| `run_metadata.json` | `run_id`, `chip_id`, `event_id`, источники/S1-файлы, split, seed и правило размещения, commit, версии/веса и пороги обоих методов, модель EL, бюджет и deadline, тариф и скидка, построение и доступность зон, неопределённость, прокси-проверка ущерба, ссылки на метрики, команды воспроизведения | ключи есть; `threshold` согласован с маской |
| `source_manifest.json` | по источнику: id/файл, URL, версия, дата наблюдения, дата публикации/доступности, лицензия, назначение, ограничения | S1 демо-чипа есть; нет источников позже даты события без пометки |

## 7. Метрики методов (для критериев 2, 4, 5, 12)

`metrics.csv` (доп. файл комплекта и `docs/verification/`): `method` (baseline/main),
`split` (dev/validation/test), `event` (или `ALL`), `iou`, `f1`, `precision`, `recall`,
`tp`, `fp`, `fn`, `tn`, `n_pixels`, `threshold`, `valid_mask_sha256`. Оба метода — на
одной маске валидных пикселей (`valid_mask_sha256` совпадает). `threshold_curve.csv` —
«порог → метрики» на validation.

## Открытые вопросы v1

- Точная структура `run_metadata.json` (вложенные объекты `methods`, `portfolio`,
  `pricing`, `uncertainty`, `proxy_check`) — фиксирует issue contracts v1.
- Смысл `uncertainty` в `asset_loss.csv` (например, стандартное отклонение EL или
  энтропия p) — фиксирует issue calibration, до того поле обязательно, смысл в метаданных.
