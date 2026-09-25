# Соответствие условиям

Строка обновляется в том же PR, что и артефакт. Статус без команды — не статус.
Статусы: ⬜ нет · 🟡 частично · ✅ проверено командой.

## 20 критериев (`criteria.md`)

| № | Критерий | Артефакт | Проверка (команда) | Статус |
|---|---|---|---|---|
| 1 | Подготовка исходных геоданных | `src/floodvalue/data/`, `docs/data.md` §11, `interim/qc.csv`, `interim/source_manifest_s1.json` → `source_manifest.json` | `make data` → 446/446 QC формата (md5, VV/VH float32, int16, EPSG:4326, сетка S1 = метка через allclose, dB, метка ⊂ {−1,0,1}), 6 чипов без валидных пикселей; `uv run --frozen pytest tests/data` | ✅ QC и манифест S1; 🟡 перенос в `source_manifest.json` — `export` |
| 2 | Пороговый baseline | `src/floodvalue/baseline/`, `models/baseline/config.json`, `docs/verification/baseline-val.md`, `baseline-metrics.csv`, `baseline-threshold-curve.csv` | `make baseline` → VH ≤ −23,0 dB (порог на train, канал на dev, D-021), IoU/F1/P/R/пиксели по событиям train/dev/validation; `uv run --frozen python -m floodvalue.baseline.predict --chip Sri-Lanka_713926` → `baseline_mask.tif` и временные p/маска на сетке S1; `uv run --frozen pytest tests/baseline` | ✅ validation IoU 0,580; на test — при `eval-test` (критерий 4) |
| 3 | Основной метод | `src/floodvalue/model/`, `models/<main>/`, `notebooks/` | `make train`, `make predict CHIP=…`; описание в README | ⬜ |
| 4 | Сравнение методов | `docs/verification/test-comparison.md` | `make eval-test` → оба метода, идентичные валидные пиксели, по событиям, FP/FN | ⬜ |
| 5 | Итоговая карта и порог | `flood_probability.tif`, `flood_mask.tif`, кривая порога на val | `make validate` (mask = p ≥ threshold); `docs/verification/threshold.md` | ⬜ |
| 6 | Split без утечки | `splits/split_v1.json`, `splits/README.md`, `docs/decisions.md` D-006/D-012/D-016, `test-log.md` | `make split` → хеш = decisions (D-016); `uv run --frozen pytest tests/split` (события не пересекаются, 446 чипов учтены, test запечатан) | ✅ split заморожен; журнал test — при `eval-test` |
| 7 | Перенос и неполные данные | test на новых событиях, S1-only, другой чип | `make predict CHIP=<другой>`; `docs/verification/transfer.md` | ⬜ |
| 8 | Контролируемый эксперимент | `docs/verification/experiment-*.md` | таблица на validation (признаки / вариант / постобработка) | ⬜ |
| 9 | Надёжность вероятностей | калибратор, reliability на test | `make calibrate`; ECE/Brier до и после; параметры опубликованы | ⬜ |
| 10 | Неопределённость | `uncertainty.tif`, `asset_loss.csv:uncertainty` | неопределённость выше на ошибках (val/test); влияние на выбор зон | ⬜ |
| 11 | Повторный запуск | README, `uv.lock`, веса в GitVerse LFS | чистый клон GitVerse → `make setup && make all` | ⬜ |
| 12 | Обязательные файлы и проверки | `src/floodvalue/contracts/` (схемы v1), `src/floodvalue/export/validate.py`, `tests/fixtures/good/` | `make validate OUT=<комплект>` → N/12 с причинами; `uv run --frozen pytest tests/contracts` (эталон 12/12 и сломанные: NaN, дыра, 0,9996 км², дубль `asset_id`, маска ≠ порогу, C сверх бюджета; совместимость с `assets` и `procurement.pricing`) | 🟡 валидатор и эталон `make validate OUT=tests/fixtures/good` → 12/12; реального комплекта ещё нет (export) |
| 13 | Ожидаемый прямой ущерб | `src/floodvalue/assets/`, `assets.*`, `asset_loss.csv`, seed и правила в `assets_metadata.json` → `run_metadata.json` (D-018) | `uv run --frozen pytest tests/assets` (EL = 1 000 000 при p=0,4, V=10 млн, q=0,25; p из растра, не обнуляется ниже порога; no_data пусто; 10 по одному типу; размещение не зависит от p и меток); `make assets CHIP=…` | 🟡 код и тесты; seed — константа до `configs/run.yaml`, в `run_metadata.json` — после export |
| 14 | Приоритеты и интерпретация | `asset_loss.csv:rank`, `src/floodvalue/assets/sensitivity.py` → `assets_sensitivity.csv` (D-024), прокси-проверка | `uv run --frozen pytest tests/assets` (rank: 1 = max EL, тай по `asset_id`, пусто у no_data; чувствительность p, V, q по отдельности, ≥ 3 сценария на каждый, смена порога бинаризации не меняет EL); `make assets CHIP=…` | 🟡 rank, временная `uncertainty` (энтропия p) и чувствительность p/V/q — код и тесты; прокси-проверка — нет |
| 15 | План съёмки в бюджете | `candidate_orders.geojson`, `procurement_plan.csv` | `make procure BUDGET=…`; валидатор геометрии; C ≤ бюджета | ⬜ |
| 16 | Сравнение A/B/C | `strategy_plans.json`, `strategy_comparison.csv` | `make validate`; B сверх бюджета помечена | ⬜ |
| 17 | Стоимость платных данных | `procurement_plan.csv`, `configs/pricing.yaml`, `src/floodvalue/procurement/pricing.py`, `geometry.py` (D-017) | `uv run --frozen pytest tests/procurement` → 1 703,46 / 946,37 / 567,82; 90 дн. → Т = 1, 91 → 0,6; `new` без гарантии — ошибка; скидка по всей группе корзины; S в км²; 0,9996 км² отклонён | 🟡 функции цены и площади проверены тестами; `procurement_plan.csv` — после #8 |
| 18 | ПП 840 и допущения | `configs/pricing.yaml` (редакция, дата проверки), README | `legal_checked_at` заполнен; `base_rate_status=scenario` везде | 🟡 редакция записана, дата проверки — нет |
| 19 | Интерфейс оператора | `src/floodvalue/app/` | `make app`; смена бюджета: хеши растров не меняются | ⬜ |
| 20 | Границы прототипа | README «Ограничения и развитие» | `scribe` + `jury` | ⬜ |

## 12 обязательных файлов (`task.md`, раздел III)

| Файл | Производит (`area:`) | Проверка | Статус |
|---|---|---|---|
| `flood_probability.tif` | model / export | `make validate`: Float32, [0,1], nodata −9999, сетка = S1 | ⬜ |
| `flood_mask.tif` | calibration / export | UInt8, 0/1/255, mask ⇔ p ≥ threshold | ⬜ |
| `assets.geojson` | assets | 10 Point, [lon, lat], типы и V/q | ⬜ |
| `assets.csv` | assets | = geojson | ⬜ |
| `asset_loss.csv` | assets | EL = p·V·q, rank, пустые у no_data | ⬜ |
| `candidate_orders.geojson` | procurement | Polygon, без дыр, ≥ 1 км², площадь EPSG:6933 | ⬜ |
| `procurement_plan.csv` | procurement | цена пересчитывается, скидка по корзине | ⬜ |
| `strategy_plans.json` | procurement | = строкам плана | ⬜ |
| `strategy_comparison.csv` | procurement | суммы, бюджет, охват по `asset_id` | ⬜ |
| `sensitivity.csv` | assets / procurement | p, V, q (`assets_sensitivity.csv`, D-024), бюджет, ставка (`procurement_sensitivity.csv`, export #10 сливает) | 🟡 p/V/q готовы, бюджет/ставка и слияние — нет |
| `run_metadata.json` | export | ключи, threshold = маска | ⬜ |
| `source_manifest.json` | data / export | S1 демо-чипа, даты, лицензии | ⬜ |
