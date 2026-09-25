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
| `app` | `src/floodvalue/app/` (API, собранный бандл в `static/`), `web/` (React) | интерфейс оператора | файлы в `outputs/`, `procurement` как функция пересчёта бюджета |
| `contracts` | `src/floodvalue/contracts/`, `docs/contracts.md` | схемы 12 файлов, форматы растров, интерфейс метода | все |
| `infra` | `Makefile`, `pyproject.toml`, `uv.lock`, `.claude/`, `.githooks/`, `scripts/` | окружение, хуки, синхронизация GitVerse | — |
| `docs` | `README.md`, `docs/`, `HANDOFF.md` | README, ограничения | — |

Общие файлы, которые правят все в своих PR (конфликт — сохранить обе стороны):
`docs/decisions.md`, `docs/tz-compliance.md`, `docs/questions-for-organizers.md`,
`docs/verification/`.

## Правила импортов

- `floodvalue.<модуль>` импортирует только `floodvalue.contracts`, `floodvalue.paths`
  и сторонние библиотеки. Соседний модуль — только через файл по контракту.
- Исключения, записанные в контракте: `app` вызывает `procurement.plan(...)` как
  чистую функцию пересчёта бюджета; `export` вызывает валидаторы из `contracts`.
- `cli.py` импортирует модули этапов лениво (по имени этапа).

## Дерево

```
src/floodvalue/{data,baseline,model,calibration,assets,procurement,export,app,contracts}/
configs/     pricing.yaml, assets.yaml, run.yaml (демо-чип, бюджет, seed)
splits/      split_v1.json (заморожен), README
models/      веса и калибраторы (LFS)
notebooks/   Colab-обёртка обучения
tests/       по модулям + tests/fixtures (маленькие синтетические растры)
docs/        data, contracts, decisions, tz-compliance, verification/, research/, plans/
web/         исходники интерфейса на React (D-013), сборка → src/floodvalue/app/static/
scripts/     sync-gitverse.sh и прочее
```
