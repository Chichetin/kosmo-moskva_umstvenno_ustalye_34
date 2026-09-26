# Контракты v1

Единственная точка стыковки модулей. Код схем — `src/floodvalue/contracts/` (pydantic v2,
D-004): `files.py` (имена, обязательные поля, спецификация растров), `schemas.py` (строки
CSV, свойства GeoJSON, `run_metadata.json`, `source_manifest.json`, `configs/run.yaml`),
`geo.py` (ROUND_HALF_UP, площадь EPSG:6933, произведение цены в Decimal), `io.py` (чтение
CSV/JSON с отказом на NaN/Inf). Валидатор — `floodvalue.export.validate` (§11, D-014).
Изменение — отдельный PR с перечнем потребителей (`docs/project-structure.md`), merge
после вердикта `reviewer`, проверившего каждого. Эталонный комплект —
`tests/fixtures/good/` (синтетический чип, собирается `tests/contracts/build_fixture.py`).

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
 "train": ["Ghana_103272", "..."], "dev": ["USA_..."], "validation": ["..."], "test": ["..."],
 "events": {"train": ["Ghana", "..."], "dev": ["USA"], "validation": ["..."], "test": ["..."]},
 "excluded": {"<chip_id>": "причина"},
 "protocol": {"train": "…", "dev": "…", "validation": "…", "test": "…", "weak": "…"},
 "sha256": "<sha256 канонического JSON четырёх списков train/dev/validation/test>"}
```

Хеш — не sha256 файла, а sha256 от `json.dumps({train, dev, validation, test}, sort_keys=True,
separators=(",", ":"), ensure_ascii=True)`; записан в `docs/decisions.md` D-016, сверяется
`make split`. Роли частей — D-012/D-016: dev — выбор модели и эксперимент, validation —
калибровка и порог. Загрузчик — `floodvalue.data.split.chip_ids(part)`; test отдаётся только с
`allow_test=True`, и это делает только `eval-test` (D-010).

## 5. Конфиги

- `configs/pricing.yaml` — формула ПП 840: редакция, дата проверки, Б и её статус,
  таблицы О/П/Т, параметры скидки для optical/sar, правила округления.
- `configs/assets.yaml` — 10 типов: `asset_class`, `asset_value_rub`, `vulnerability_coef`.
- `configs/run.yaml` (схема `RunConfig`, `floodvalue.contracts.load_run_config()`):
  `chip_id`, `event_id`, `chip_split`, `portfolio_seed`, `budget_rub`, `decision_at`,
  `decision_deadline` (ISO 8601 с поясом). До заморозки — `Sri-Lanka_713926` (validation,
  D-010); после — `Spain_7370579` (test) только замороженной конфигурацией.

## 6. Обязательные файлы (поля — дословно `task.md`, раздел III)

Общее: CSV — UTF-8, `,`, десятичная точка, пусто вместо NaN/Inf; JSON/GeoJSON без
NaN/Inf; GeoJSON — WGS 84, [долгота, широта]. Дополнительные поля разрешены.

| Файл | Обязательные поля / структура | Ключевые проверки валидатора |
|---|---|---|
| `assets.geojson` | FeatureCollection из 10 Point; `asset_id`, `chip_id`, `asset_class`, `asset_value_rub`, `vulnerability_coef` | ровно 10, по одному типу, V/q = `configs/assets.yaml`, точка внутри чипа |
| `assets.csv` | `asset_id`, `chip_id`, `asset_class`, `asset_value_rub`, `vulnerability_coef`, `longitude`, `latitude` | совпадает с geojson |
| `asset_loss.csv` | `asset_id`, `chip_id`, `p_flood`, `expected_loss_rub`, `uncertainty`, `rank`, `status` (ok/partial/no_data) | EL = p·V·q; p = значение растра в пикселе объекта; `rank` 1 = max EL, тай — по `asset_id`; `partial`/`no_data` → пустые p/EL/rank |
| `candidate_orders.geojson` | Polygon; `candidate_id`, `sensor_type`, `resolution_m`, `acquisition_type` (new/operational/archive), `data_role` (event_observation/context), `observation_at`, `available_at`, `processing_level` (L0/L1/L2), `usage_type` (internal/limited/unrestricted), `guaranteed_purchase`, `area_km2`, `base_rate_rub_km2`, `base_rate_status`, `base_rate_source` | кольца замкнуты и ≥ 4 точек, valid, без дыр, координаты в диапазоне [lon, lat], зона пересекает чип, площадь EPSG:6933 ≥ 1 км² до округления, `area_km2` = ROUND_HALF_UP(0,001), уникальные id, `base_rate_rub_km2`/`base_rate_status` = `configs/pricing.yaml`, существующие снимки → `guaranteed_purchase=false`; `guaranteed_purchase=true` только при `acquisition_type=new`; `new` без гарантии — ошибка (подробно §7a) |
| `procurement_plan.csv` | `strategy`, `candidate_id`, `area_km2`, `base_rate_rub_km2`, `base_rate_status`, `processing_level`, `usage_type`, `guaranteed_purchase`, `processing_coef`, `usage_coef`, `freshness_coef`, `discount_coef`, `discount_group_id`, `group_area_km2`, `unit_price_rub_km2`, `cost_rub`, `formula`, `legal_edition`; доп.: `sensor_type`, `resolution_m`, `acquisition_type` (критерий 17) | уникальна пара (strategy, candidate_id); нет строк A; категории = каталог; цена пересчитывается; скидка по группе целиком |
| `strategy_plans.json` | `{"A": [], "B": [...], "C": [...]}` | состав B и C = множеству `candidate_id` строк плана своей стратегии; id из каталога |
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

- **Строки `split=train`** пишет только baseline в `docs/verification/baseline-metrics.csv`
  (порог подобран на train — это качество подгонки, не проверка). В `metrics.csv` комплекта
  экспорт их не переносит (`export.assemble.METRICS_SPLITS` = dev/validation/test).
- **Единица `threshold`** — своя у каждого метода: у `baseline` — dB канала правила
  (`VH ≤ −23,0` → `-23.0`), у `main` — вероятность p (mask = 1 ⇔ p ≥ threshold). В
  `metrics.csv` комплекта единицу называет доп. столбец `threshold_units` (`dB VH` / `p`).
- **Один алгоритм на всех** (D-032): маска валидности — `floodvalue.data.io.valid_mask`
  (S1 конечен в обоих каналах **и** метка ≠ −1, D-015); счёт TP/FP/FN/TN, IoU/F1/P/R и
  `valid_mask_sha256` (упакованные маски в порядке `chip_id`) —
  `floodvalue.model.metrics`. Изменение его формул или хеша — изменение контракта §7.

## 7a. Уточнения к §6, v1 (проверяет валидатор)

- Пустая ячейка CSV = нет значения; литералы `nan`/`inf`/`Infinity` — ошибка. Булевы в
  CSV — строго `true`/`false`. Даты — ISO 8601; время без пояса считается UTC.
- `asset_loss.csv`: `p_flood` — значение `flood_probability.tif` в пикселе объекта
  (допуск 1e-6 к float32), без округления сверх кратчайшей записи; `expected_loss_rub` =
  записанный `p_flood` × V × q с допуском 0,01 руб.; `ok` ⇒ пиксель валиден, `no_data` ⇒
  пиксель nodata (`partial` — на усмотрение производителя, с причиной в метаданных);
  `partial`/`no_data` → пустые `p_flood`, `expected_loss_rub`, `rank`; `rank` 1…n только у
  `ok` по (−EL, `asset_id`). `uncertainty` может быть пустым; смысл — `run_metadata.uncertainty`.
- `candidate_orders.geojson`: только `Polygon` (не MultiPolygon); `observation_at`,
  `available_at` — ключи обязательны, значение может быть `null` (неизвестно);
  `base_rate_status` ∈ {scenario, official}. Геометрия проверяется до разбора shapely
  (он молча замыкает кольца): каждое кольцо — ≥ 4 позиций, первая = последней (RFC 7946
  §3.1.6), иначе ошибка; при такой ошибке или ошибке схемы любой зоны каталог дальше (в т. ч.
  план ↔ каталог) не проверяется. Затем: все вершины внешнего кольца в
  диапазоне |lon| ≤ 180, |lat| ≤ 90; зона **пересекает** охват чипа (прямоугольник сетки
  `flood_probability.tif` в WGS 84, только если растр прочитан; целиком внутри быть не
  обязана — буфер у края чипа допустим); valid, без внутренних колец. `base_rate_rub_km2` и `base_rate_status` каждой
  зоны равны `configs/pricing.yaml` (`base_rate.rub_per_km2`, `base_rate.status`).
  **Ограничение:** порядок осей проверяется только косвенно — через диапазон и пересечение с
  чипом. Перепутанные [широта, долгота], которые случайно тоже пересекают чип (чип около
  диагонали lon = lat), валидатор не ловит; гарантия порядка — производитель
  (`shapely.mapping` от геометрии в [lon, lat], `always_xy=True`) и round-trip тест.
- `procurement_plan.csv` ↔ каталог: каждый `candidate_id` есть в каталоге; `area_km2`,
  `base_rate_rub_km2`, `base_rate_status`, `processing_level`, `usage_type`,
  `guaranteed_purchase` строки = полям зоны каталога; доп. поля `sensor_type`,
  `acquisition_type`, `resolution_m`, если заполнены, тоже = каталогу.
- `strategy_plans.json` ↔ план ↔ каталог: все id B и C есть в каталоге; множество id
  стратегии = множеству `candidate_id` её строк в `procurement_plan.csv`.
- `run_metadata.pricing`: `legal_edition`, `base_rate_rub_km2`, `base_rate_status` =
  `configs/pricing.yaml` (ошибка приписывается `run_metadata.json`).
- `procurement_plan.csv`: группа скидки — строки одной стратегии с одним
  `discount_group_id` (D-017: `<семейство>_<r>m_<год>`), одно семейство сенсора и одно
  `resolution_m`; `group_area_km2` одинакова в группе и ≥ суммы `area_km2` корзины группы
  (разница — `previous_area_km2`); Р = min(1; max(floor; slope·ln(S/r²)+intercept)) до
  9 знаков (допуск 1,5e-9); О, П — по `configs/pricing.yaml`; Т — по `acquisition_type`
  каталога (new → 1,8; operational → 1,0; archive → 0,6); `unit_price_rub_km2` =
  ROUND_HALF_UP(Б·О·П·Т·Р, 0,01) и `cost_rub` = ROUND_HALF_UP(unit·area, 0,01) в Decimal от
  записанных чисел — точно до копейки; `legal_edition` = `configs/pricing.yaml`.
- `strategy_comparison.csv`: ровно по одной строке A, B, C; один `budget_rub` =
  `run_metadata.budget.budget_rub`; `budget_feasible` ⇔ `data_cost_rub ≤ budget_rub`; C
  обязана быть в бюджете; `covered_expected_loss_rub` (B, C) = Σ EL `ok`-объектов, которые
  `covers` хотя бы одна зона стратегии с `data_role=event_observation` и `available_at ≤
  decision_deadline` (допуск 0,01); `coverage_share` = covered / Σ EL `ok`-объектов (допуск
  1e-4), пусто ⇔ Σ EL = 0; `residual_uncertainty` пусто ⇔ `not_estimated`; A — `baseline`;
  у A платных зон нет, поэтому её `covered_expected_loss_rub` = 0, иное — только с
  объяснением в `run_metadata.strategies.coverage_A`.
- `metrics.csv` комплекта (§7): каждая строка `method=main` (любой `split` —
  dev/validation/test — и любой `event`) имеет `threshold`, равный `run_metadata.threshold`
  (единицы p, допуск 1e-9), — метрики main посчитаны тем же порогом, что и `flood_mask.tif`;
  пустой или нечисловой `threshold` у main — ошибка. Строки `baseline` (порог в dB канала)
  с `run_metadata.threshold` не сравниваются. Ошибка приписывается `run_metadata.json` (#75).
  Производитель — `export/run.py::_metrics` (строки main — из
  `docs/verification/calibration/<method>-metrics.csv` при калибраторе, #62). Если
  калибратор применён, а калиброванных метрик нет, export пишет предупреждение и берёт
  сырые строки с другим порогом — `validate` на таком комплекте даёт FAIL (так задумано:
  метрики не описывают маску; лечится `make calibrate`).
- `sensitivity.csv`: ключи `changed_inputs_json` для обязательных вариаций —
  `p_flood`, `asset_value_rub`, `vulnerability_coef`, `budget_rub`, `base_rate_rub_km2`;
  на каждый — хотя бы одна строка, где из этих пяти изменён только он (другие ключи,
  например `asset_id`, допустимы). Ставка как «экономический параметр» и раздельные p, V, q —
  решение команды (D-014): task.md требует бюджет и хотя бы один экономический параметр.

## 8. `run_metadata.json` (схема `RunMetadata`)

Обязательные ключи верхнего уровня; вложенные объекты допускают дополнительные поля.

```json
{"contract_version": "1", "run_id": "…", "chip_id": "Sri-Lanka_713926", "event_id": "Sri-Lanka",
 "event_date": "2017-05-30", "created_at": "<ISO>", "commit": "<sha>",
 "threshold": 0.5,
 "source": {"dataset": "Sen1Floods11 v1.1 HandLabeled", "dataset_url": "…",
            "s1_files": ["Sri-Lanka_713926_S1Hand.tif"], "source_manifest": "source_manifest.json"},
 "split": {"file": "splits/split_v1.json", "sha256": "…|null", "chip_split": "validation"},
 "portfolio": {"seed": 20260925, "placement_rule": "…", "assets_config": "configs/assets.yaml"},
 "methods": {"baseline": {"name", "version", "weights": [{"path", "sha256"}], "threshold",
                          "threshold_units": "dB VH", "threshold_source"},
             "main": {… "threshold" = threshold, "threshold_units": "p"},
             "evaluation_order": "…"},
 "expected_loss": {"formula": "EL = p_flood × V × q", "p_source", "status_rules"},
 "budget": {"budget_rub", "decision_at", "decision_deadline"},
 "pricing": {"config", "legal_edition", "base_rate_rub_km2", "base_rate_status",
             "discount_rule", "grouping"},
 "zones": {"construction", "availability_rule", "area_method", "unknown_dates_rule"},
 "strategies": {"rule_B", "rule_C", "other_costs", "coverage_aggregation",
                "coverage_share_note": null},
 "uncertainty": {"meaning", "units", "method", "raster": null},
 "proxy_check": {"status": "done|planned|not_done", "test_chip", "control_points",
                 "fixed_v_q", "n_points", "error_metric", "result"},
 "metrics": {"files": ["metrics.csv", "…"]},   # ≥ 1 файл; путь от комплекта или репо
 "reproduce": ["make all CHIP=… BUDGET=…"]}
```

`threshold` — порог основного метода (mask = 1 ⇔ p ≥ threshold, сравнение в float64
значения float32), равен `methods.main.threshold`. `source.s1_files` содержит S1 чипа.
Каждый файл `metrics.files` существует; `metrics.csv` (§7) содержит строки `baseline` и
`main`, и в каждой паре (`split`, `event`) у них один `valid_mask_sha256` (критерий 12:
«метрики двух методов»; ошибки приписываются `run_metadata.json`).
`assets_metadata.json` модуля `assets` (D-018) экспорт переносит так: `seed`,
`placement_rule` → `portfolio`; `el_model`, `p_extraction_rule`, `status_rule` →
`expected_loss`; `uncertainty` → `uncertainty` (пока `binary_entropy_bits`, биты, временно).
Необязательный `inspection_priority` (D-035: очерёдность осмотра рядом с `rank`, который
не меняется) → `expected_loss.inspection_priority`; в прогоне до D-035 — `null`.

## 9. `source_manifest.json` (схема `SourceManifest`)

`{"sources": [{"id" | "source_id", "file", "url", "version", "observation_date",
"available_date" | "published_at", "license", "purpose", "limitations", "applies_to",
"after_event_note"}]}`; `observation_date`, `available_date` — `YYYY-MM-DD` или `null`,
`published_at` — ISO 8601; нужна хотя бы одна из `available_date`/`published_at`. Записи
`$FLOODVALUE_DATA_DIR/interim/source_manifest_s1.json` (`floodvalue.data`, D-015:
`source_id`, `published_at`, вложенный `label`) копируются экспортом как есть. Есть источник
с `<chip_id>_S1Hand` в `id` или `file`; `id` уникальны; `observation_date` позже
`run_metadata.event_date` — только с непустым `after_event_note`.

## 10. Пересчёт бюджета для интерфейса (D-005, D-013, D-038)

```python
# floodvalue.procurement.plan — реализует area:procurement
def plan(run_dir: Path, budget_rub: float, social: bool | None = None) -> dict:
    """Читает из run_dir assets.csv (ровно один chip_id), asset_loss.csv и растры комплекта
    (flood_probability.tif, uncertainty.tif — если есть); конфиги configs/run.yaml и
    configs/pricing.yaml; $FLOODVALUE_DATA_DIR/interim/source_manifest_s1.json (пишет
    `make data`) — observation_date чипа, от которой считаются decision_at и
    decision_deadline (D-043). Каталог зон строится заново из текущих входов (устаревший
    candidate_orders.geojson не читается). Ничего не пишет; растры, p, маску, порог и
    портфель не меняет; детерминирована при тех же входах.

    Исключения (плана нет): FileNotFoundError — нет манифеста S1 или assets.csv/asset_loss.csv;
    ValueError — в assets.csv не один chip_id, в манифесте нет observation_date чипа, или
    дата decision_at в run.yaml ≠ дате S1 чипа из run.yaml.

    social — социальное ограничение стратегии C (D-038), необязательный аргумент:
      None  — как в configs/run.yaml → social_constraint.enabled (по умолчанию false);
      True  — C сначала покрывает объекты классов social_constraint.asset_classes;
      False — ограничение выключено независимо от конфига.

    Возвращает dict ровно с ключами верхнего уровня:
      "plans":       {"A": [], "B": [candidate_id…], "C": [candidate_id…]}  # = strategy_plans.json
      "plan_rows":   [dict]  # строки procurement_plan.csv (поля §6) БЕЗ "reason":
                             #   reason подставляет procure из reasons["B"|"C"]
      "comparison":  [dict]  # строки strategy_comparison.csv для A, B, C (поля §6);
                             #   + addressed_decision_risk_rub, decision_risk_share (D-049: Σ R = V·q·min(p, 1−p) покрытых ok; доля None при Σ R = 0);
                             #   включая covered_asset_ids (asset_id через пробел);
                             #   значения — Python-типы (str/Decimal/float/bool/None), не строки CSV;
                             #   budget_rub = аргумент, округлённый до 0,01
                             # Decimal в plan_rows/comparison json.dumps не сериализует —
                             #   приводит к строке потребитель (как _cell в procure)
      "reasons": {
          "B":              {candidate_id: правило B},
          "C":              {candidate_id: "шаг k: новый охват …, цена корзины …"},
          "not_selected_C": {candidate_id: почему не выбрана в C},  # все прочие позиции каталога
          "chosen_any":     [candidate_id…],  # B ∪ C, отсортировано
          "B_empty":        str,              # только если B пуста
      }
      "social_constraint": {"required_asset_ids": []}               # выключено или нет
                                                                    #   объектов asset_classes
                         | {"required_asset_ids": [asset_id…], "satisfied": bool,
                            "uncovered_required": [asset_id…], "note": str}  # список не пуст
      "c_selection": {"variant": "greedy" | "start:<candidate_id>",
                      "alternatives": {variant: {"chosen": […], "data_cost_rub": str,
                                                 "covered_expected_loss_rub": число}}}
                     # alternatives — только "greedy" и выбранный variant, не все проходы
    """
```

`app` вызывает её при смене бюджета (и переключателе социального ограничения);
`procure` (`procurement/run.py: write_outputs`) пишет тот же пересчёт при `social=None` в
файлы комплекта, а `social_constraint` и `c_selection` — в `procurement_metadata.json`;
`export` читает только файлы и `procurement_metadata.json`, не вызывает `plan()`. Результат,
записанный в файлы, обязан проходить валидатор. Ключи верхнего уровня и необязательность
`social` зафиксированы тестом `tests/procurement/test_plan_contract.py`.

## 11. Валидатор (критерий 12, D-014)

`python -m floodvalue validate --out <каталог>` = `make validate OUT=<каталог>`. Без
`--out` — самый свежий комплект в `FLOODVALUE_OUTPUT_DIR` (подкаталог с
`run_metadata.json`, по времени изменения этого файла), если его нет — эталон
`tests/fixtures/good`; путь печатается первой строкой, эталон помечен «ЭТАЛОН …, не
результат запуска». Реальный комплект проверяется явно: `make validate OUT=outputs/<run_id>`. Вывод — `OK`/`FAIL` по каждому из
12 файлов с причинами и итог `N/12`; код возврата 0 только при 12/12. Каждая ошибка
приписана одному файлу; перекрёстные проверки (маска ↔ порог, EL ↔ растр, план ↔
сравнение) приписаны файлу, который им противоречит. Если найден S1 чипа в
`FLOODVALUE_DATA_DIR`, сетка `flood_probability.tif` сверяется и с ним.

## Открытые вопросы

- Смысл `uncertainty` в `asset_loss.csv` — временно бинарная энтропия p в битах (D-018),
  окончательно фиксирует issue calibration/uncertainty (записывается в
  `run_metadata.uncertainty`).
