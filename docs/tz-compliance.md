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
| 11 | Повторный запуск | README, `uv.lock`, веса в GitVerse LFS, `scripts/setup_gpu_env.sh`; `make all` (D-027), `run_metadata.reproduce`/`versions`/`commit` | `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → комплект `outputs/Sri-Lanka_713926_5000_<commit>/` и validate; чистый клон GitVerse → `make setup && make all` | 🟡 `make check` зелёный (в чистом клоне — эталон, после `make all` — реальный комплект 12/12); GPU-окружение обучения готово (D-024); `make all` на реальном чипе проходит все этапы из кода (predict пока baseline D-009 до #16/#17), validate 12/12; чистый клон — не проверен |
| 12 | Обязательные файлы и проверки | `src/floodvalue/contracts/` (схемы v1), `src/floodvalue/export/validate.py`, `src/floodvalue/export/run.py` (D-027), `tests/fixtures/good/`, `tests/export/` | `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → `make validate OUT=outputs/Sri-Lanka_713926_5000_<commit>`; `uv run --frozen pytest tests/contracts tests/export` (эталон 12/12, сломанные комплекты; e2e на синтетическом чипе: predict → assets → procure → export → validate 12/12) | ✅ реальный комплект 12/12 (`make all CHIP=Sri-Lanka_713926 BUDGET=5000`); 🟡 метрики main = baseline (D-009) до #16/#17, proxy_check — P24a |
| 13 | Ожидаемый прямой ущерб | `src/floodvalue/assets/`, `assets.*`, `asset_loss.csv`, seed и правила в `assets_metadata.json` → `run_metadata.json` (D-018) | `uv run --frozen pytest tests/assets` (EL = 1 000 000 при p=0,4, V=10 млн, q=0,25; p из растра, не обнуляется ниже порога; no_data пусто; 10 по одному типу; размещение не зависит от p и меток); `make assets CHIP=…` | 🟡 код и тесты; seed — константа до `configs/run.yaml`, в `run_metadata.json` — после export |
| 14 | Приоритеты и интерпретация | `asset_loss.csv:rank`, `src/floodvalue/assets/sensitivity.py` → `assets_sensitivity.csv` (D-028), прокси-проверка | `uv run --frozen pytest tests/assets` (rank: 1 = max EL, тай по `asset_id`, пусто у no_data; чувствительность p, V, q по отдельности, ≥ 3 сценария на каждый, смена порога бинаризации не меняет EL); `make assets CHIP=…` | 🟡 rank, временная `uncertainty` (энтропия p) и чувствительность p/V/q — код и тесты; прокси-проверка — нет |
| 15 | План съёмки в бюджете | `candidate_orders.geojson`, `procurement_plan.csv`, `procurement_metadata.json`; D-023 | `make assets CHIP=Sri-Lanka_713926 && make procure BUDGET=5000` (C: Z-A02, Z-A10 — 4 905,96 ≤ 5 000); `uv run --frozen pytest tests/procurement` (≥ 1 км² до округления, covers, срок, C ≤ бюджета, пересчёт корзины) | 🟡 зоны v0 |
| 16 | Сравнение A/B/C | `strategy_plans.json`, `strategy_comparison.csv`, `procurement_sensitivity.csv`; D-023 | `make procure BUDGET=5000` → B 42 465,93 > 5 000, `budget_feasible=false`; `uv run --frozen pytest tests/procurement/test_strategies.py tests/procurement/test_plan.py` (бюджет меняется — хеши p/маски/asset_loss те же); `make validate` | 🟡 A/B/C v0 |
| 17 | Стоимость платных данных | `procurement_plan.csv`, `configs/pricing.yaml`, `src/floodvalue/procurement/pricing.py`, `geometry.py` (D-017) | `uv run --frozen pytest tests/procurement` → 1 703,46 / 946,37 / 567,82; 90 дн. → Т = 1, 91 → 0,6; `new` без гарантии — ошибка; скидка по всей группе корзины; S в км²; 0,9996 км² отклонён | 🟡 функции цены и площади проверены тестами; `procurement_plan.csv` — после #8 |
| 18 | ПП 840 и допущения | `configs/pricing.yaml` (редакция, дата проверки), README | `legal_checked_at` заполнен; `base_rate_status=scenario` везде | 🟡 редакция записана, дата проверки — нет |
| 19 | Интерфейс оператора | `src/floodvalue/app/` | `make app`; смена бюджета: хеши растров не меняются | ⬜ |
| 20 | Границы прототипа | README «Ограничения и развитие» | `scribe` + `jury` | ⬜ |

## 12 обязательных файлов (`task.md`, раздел III)

| Файл | Производит (`area:`) | Проверка | Статус |
|---|---|---|---|
| `flood_probability.tif` | model / export | `make validate`: Float32, [0,1], nodata −9999, сетка = S1 | ✅ `make all` (baseline D-009) |
| `flood_mask.tif` | calibration / export | UInt8, 0/1/255, mask ⇔ p ≥ threshold | ✅ `make all` (порог 0,5, D-009) |
| `assets.geojson` | assets | 10 Point, [lon, lat], типы и V/q | ✅ `make all` |
| `assets.csv` | assets | = geojson | ✅ `make all` |
| `asset_loss.csv` | assets | EL = p·V·q, rank, пустые у no_data | ✅ `make all` |
| `candidate_orders.geojson` | procurement | Polygon, без дыр, ≥ 1 км², площадь EPSG:6933 | ✅ `make all` |
| `procurement_plan.csv` | procurement | цена пересчитывается, скидка по корзине | ✅ `make all` |
| `strategy_plans.json` | procurement | = строкам плана | ✅ `make all` |
| `strategy_comparison.csv` | procurement | суммы, бюджет, охват по `asset_id` | ✅ `make all` |
| `sensitivity.csv` | assets / procurement | p, V, q (`assets_sensitivity.csv`, D-028), бюджет, ставка (`procurement_sensitivity.csv`, D-023; export #10 сливает) | ✅ `make all`: бюджет, ставка, p, V, q по отдельности + строки фиксированного плана C (D-027) |
| `run_metadata.json` | export | ключи, threshold = маска | ✅ `make all` (proxy_check=planned, P24a) |
| `source_manifest.json` | data / export | S1 демо-чипа, даты, лицензии | ✅ `make all` |
