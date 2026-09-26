# Структура проекта и владение каталогами

Владельца-человека у каталога нет: каталогом «владеет» issue с его `area:` и assignee,
пока она в работе. Две issue одной `area:` одновременно — только у одного человека
(его параллельные `builder` на непересекающихся файлах).

| `area:` | Каталоги | Производит (контракт) | Потребляет |
|---|---|---|---|
| `data` | `src/floodvalue/data/` (fetch, QC, манифест), `tests/data/` | `data/` (вне git), `source_manifest` часть S1 | официальный бакет |
| `split` | `splits/`, `src/floodvalue/data/split.py` | `splits/split_v1.json` + хеш | манифест `data` |
| `baseline` | `src/floodvalue/baseline/` | порог baseline, маска, метрики val/test | S1, метки, split |
| `model-pixel` | `src/floodvalue/model/pixel/`, `models/pixel/` | метод A: `predict_proba`, веса, эксперимент | S1, split |
| `model-unet` | `src/floodvalue/model/unet/`, `notebooks/`, `models/unet/` | метод B: `predict_proba`, веса | S1, split |
| `eval` | `src/floodvalue/model/eval_test.py`, `src/floodvalue/model/metrics.py`, `tests/eval/` | `eval-test`, метрики на идентичных пикселях, перенос | p-растры обоих методов, split |
| `calibration` | `src/floodvalue/calibration/` | калибратор, порог основного метода, неопределённость | p на validation |
| `assets` | `src/floodvalue/assets/` | `assets.*`, `asset_loss.csv`, чувствительность p/V/q | p-растр, `configs/assets.yaml` |
| `procurement` | `src/floodvalue/procurement/`, `configs/pricing.yaml` | зоны, цена, A/B/C, `procurement_plan.csv`, `strategy_*`, чувствительность бюджета/ставки | `asset_loss.csv`, p, неопределённость |
| `export` | `src/floodvalue/export/` | сборка 12 файлов, `run_metadata.json`, `validate` | всё выше |
| `app` | `src/floodvalue/app/` (FastAPI + собранный React), `web/` (React/MapLibre, D-050) | интерфейс оператора | файлы в `outputs/`, `procurement` как функция пересчёта бюджета |
| `media` | `src/floodvalue/media/` (со своим `sources.yaml` и `static/`), `tests/media/` | реестр сообщений СМИ, сводок и очевидцев; `GET /api/media/events` (GeoJSON) для будущего слоя на глобусе (D-051) | `splits/split_v1.json` и `$FLOODVALUE_DATA_DIR/interim/source_manifest_s1.json` как файлы; комплекты `outputs/*/run_metadata.json` — только чтение `chip_id` |
| `contracts` | `src/floodvalue/contracts/`, `docs/contracts.md` | схемы 12 файлов, форматы растров, интерфейс метода | все |
| `infra` | `Makefile`, `pyproject.toml`, `uv.lock`, `pyrightconfig.json`, `.claude/`, `.githooks/`, `scripts/` | окружение, хуки, синхронизация GitVerse | — |
| `docs` | `README.md`, `docs/`, `HANDOFF.md` | README, ограничения | — |

Общие файлы, которые правят все в своих PR (конфликт — сохранить обе стороны):
`docs/decisions.md`, `docs/tz-compliance.md`, `docs/questions-for-organizers.md`,
`docs/verification/`.

## Правила импортов

- `floodvalue.<модуль>` импортирует только `floodvalue.contracts`, `floodvalue.paths`
  и сторонние библиотеки. Соседний модуль — только через файл по контракту (D-005).
- Все исключения — одна таблица `ALLOWED` (и `FILES` для имён, разрешённых только
  отдельным файлам) в `tests/test_import_rules.py`: AST всех `.py` пакета, включая
  ленивые и относительные импорты, строки-пути модулей в реестрах и атрибуты модулей,
  импортированных целиком. Лишняя (неиспользуемая) строка таблицы — тоже ошибка. Новый
  межмодульный импорт = строка таблицы со ссылкой на `D-…` в том же PR.
- Записанные исключения (только чтение; каждому модулю — только то, что он использует):
  - чтение чипа `floodvalue.data.io`: `load_s1`, `load_label`, `valid_mask` (D-015) —
    baseline, model, calibration. Сверх D-015 (D-057): baseline — `layer_path`,
    `S1_LAYER`; calibration — `layer_path`, `parse_chip_id`; model — `layer_path`,
    `raw_dir`, `parse_chip_id`, `S1_LAYER`, `JRC_LAYER`; context — `layer_path`,
    `raw_dir`, `parse_chip_id`. `layer_path`, `raw_dir`, `JRC_LAYER` — только
    перечисленным в `FILES` файлам (путь к слою мимо замка `load_label`);
  - split `floodvalue.data.split`: `chip_ids` (§4 контрактов) — baseline, model,
    calibration. Сверх §4 (D-057): `load_split` — baseline, model, calibration;
    `SealedTestError` — model, context; `SPLIT_PATH`, `event_of` — только
    `model/eval_test.py`; `EVENT_PARTS` — app (D-050) и context (D-057);
  - общий алгоритм метрик §7 `floodvalue.model.metrics` — baseline, calibration (D-032);
  - `calibration` → `model.predict.METHODS` (D-041); `model` → `calibration.calibrator`,
    `calibration.uncertainty` ленивым импортом (D-039, D-041);
  - `model` → `baseline.predict` — только `model/predict.py` (реестр методов, §3) и
    `model/unet/report.py` (сравнение с baseline) (D-057, пересмотреть после `eval-test`);
  - `model/eval_test.py` → `cli.main` (стадия assets через CLI, D-030);
  - `app` → `procurement.plan` (§10 контрактов, D-050) и `data.split.EVENT_PARTS` (D-050);
  - `media` — только `floodvalue.paths` (D-051); `context` никто из этапов не импортирует
    (D-042, `tests/context/test_firewall.py`);
  - `cli.py` импортирует модули этапов лениво по имени этапа (`STAGES`), `__main__` —
    `cli.main`.
- Что тест не видит (D-057): что вызывающий делает с модулем, полученным из реестра через
  `importlib` (calibration → модули методов по `METHODS` — граница D-041; `eval_test` →
  `baseline.predict`), имена модулей, собранные f-строкой, и реэкспорт имени из `FILES`
  через другой файл своего модуля (импорт внутри своего модуля разрешён всегда). Узкие
  тесты `tests/baseline/test_baseline_imports.py`,
  `tests/calibration/test_calibration_imports.py`, `tests/context/test_firewall.py` (D-042),
  `tests/media/test_media_firewall.py` (D-051) остаются.

## Дерево

```
src/floodvalue/{data,baseline,model,calibration,assets,procurement,export,app,contracts}/
configs/     pricing.yaml, assets.yaml, run.yaml (демо-чип, бюджет, seed)
splits/      split_v1.json (заморожен), README
models/      веса и калибраторы (LFS)
notebooks/   Colab-обёртка обучения
tests/       по модулям + tests/fixtures (маленькие синтетические растры)
docs/        data, contracts, decisions, tz-compliance, verification/, research/, plans/
web/         React + TypeScript + MapLibre (D-050), сборка → src/floodvalue/app/static/
scripts/     sync-gitverse.sh и прочее
```
