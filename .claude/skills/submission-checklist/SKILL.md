---
name: submission-checklist
description: Проверка комплекта сдачи FloodValue перед стоп-кодом — 12 обязательных файлов через валидатор, GitVerse (commit, веса в LFS/релизе, README), запуск из чистого клона, inference на другом официальном чипе, режим S1-only, журнал test. Использовать перед 27.09 00:00 и после любой правки комплекта.
---

# Проверка сдачи

Каждый пункт — команда и её вывод. Нет вывода — пункт не выполнен.

## 1. Чистый клон GitVerse

```bash
D=$(mktemp -d) && git clone <gitverse-url> "$D/r" && cd "$D/r"
git lfs pull && git lfs ls-files            # веса скачались, а не указатели
git rev-parse HEAD                          # = commit в README
uv sync --frozen && uv run pytest
```

- [ ] commit в README = `git rev-parse HEAD` клона и = `origin/main` GitHub.
- [ ] Веса в `models/` — реальные файлы (размер ≠ ~130 байт указателя), sha256 = `run_metadata.json`.
- [ ] Sen1Floods11 в репо нет: `git ls-files | grep -E '\.(tif|tiff)$'` → только fixtures.

## 2. Цепочка по README на демо-чипе

```bash
make data && make predict CHIP=<демо> && make assets && make procure BUDGET=<б> && make export && make validate
```

- [ ] `validate`: 12/12 — `flood_probability.tif`, `flood_mask.tif`, `assets.geojson`,
      `assets.csv`, `asset_loss.csv`, `candidate_orders.geojson`, `procurement_plan.csv`,
      `strategy_plans.json`, `strategy_comparison.csv`, `sensitivity.csv`,
      `run_metadata.json`, `source_manifest.json`.
- [ ] Маска = p ≥ threshold на всех валидных пикселях; сетка = S1-чип.
- [ ] 10 объектов, по одному типу, V/q из таблицы; seed и правило в `run_metadata.json`.
- [ ] Нет NaN/Inf в CSV/JSON/GeoJSON; пустые поля у `no_data`.
- [ ] Зоны: Polygon без дыр, ≥ 1 км² до округления, площадь EPSG:6933 совпадает с `area_km2`.
- [ ] Цены: контрольные 1 703,46 / 946,37 / 567,82 руб. в тестах; `base_rate_status=scenario`.
- [ ] C ≤ бюджета; B сверх бюджета — `budget_feasible=false` и пометка контрфактической.

## 3. Бюджет без переобучения

```bash
sha256sum outputs/flood_probability.tif outputs/flood_mask.tif
make procure BUDGET=<другой> && make export && make validate
sha256sum outputs/flood_probability.tif outputs/flood_mask.tif   # те же хеши
```

- [ ] Хеши растров и порог не изменились, план C изменился (или объяснено, почему нет).

## 4. Другой официальный чип и S1-only

- [ ] `make predict CHIP=<другой чип, не демо, не из train>` → растры и `validate` без подгонки.
- [ ] Режим S1-only: команда из README работает без S2/JRC/дополнительных данных.

## 5. Test и split

- [ ] `splits/` = хеш из `docs/decisions.md`; события не пересекаются или обосновано.
- [ ] `docs/verification/test-log.md`: каждый `eval-test` — commit и хеш конфигурации;
      ни одного подбора после просмотра test.
- [ ] Метрики baseline и основного метода на идентичных валидных тестовых пикселях.

## 6. README и документы

- [ ] README: ссылка на Sen1Floods11, commit, версии (`uv.lock`), получение данных,
      команды всех этапов, обучение (Colab), интерфейс, экспорт, веса и sha256.
- [ ] Ограничения и развитие (критерий 20); ПП 840: редакция и дата проверки (18).
- [ ] `docs/tz-compliance.md`: все 20 критериев и 12 файлов — статус с командой.

## 7. Итог

Таблица «пункт → статус → доказательство», список дыр с владельцем-issue. Прогон
`jury` по итогам. Commit сдачи — в `HANDOFF.md`.
