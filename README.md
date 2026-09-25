# FloodValue — карта воды по Sentinel-1 → ожидаемый ущерб → план заказа ДЗЗ

Космохакатон 2026. Прототип для **инфраструктурного оператора**: по официальному
Sentinel-1 чипу Sen1Floods11 строит вероятностную карту воды, оценивает ожидаемый
прямой ущерб 10 учебных объектов и выбирает дополнительную съёмку в пределах
бюджета (стратегии A/B/C).

> Заготовка. Каждый раздел заполняется командой и её фактическим выводом; пока
> команды нет — стоит «не реализовано». Прототип не является официальной оценкой ущерба;
> ставка 788,64 руб./км² — сценарная.

- Данные: официальный Sen1Floods11 — https://github.com/cloudtostreet/Sen1Floods11
  (v1.1, `gs://sen1floods11`), в репозиторий не кладётся. Подробно — `docs/data.md`.
- Commit сдачи: `TBD` · версии зависимостей — `uv.lock` · Python 3.12.

Новый участник команды — `docs/onboarding-prompt.md`.

## Установка

```bash
pipx install uv            # или https://docs.astral.sh/uv/
git lfs install             # веса моделей — через Git LFS (models/README.md, D-011)
make setup                  # uv sync --frozen + git-хук стража данных
cp .env.example .env        # FLOODVALUE_DATA_DIR, FLOODVALUE_OUTPUT_DIR
```

`git lfs install` включает автоскачивание содержимого LFS при `git clone`/`git pull`;
вручную — `git lfs pull`. Зависимости `uv.lock` — **CPU-сборка torch**, inference
одного чипа работает без GPU (D-003). Обучение на GPU — раздел «Обучение» ниже.

Сен1Floods11 (не в git) `make data` качает сам — только слои, нужные конвейеру
(S1Hand, LabelHand, JRCWaterHand, метаданные), с QC и манифестом; повторный запуск
только сверяет хеши. Полный набор (все 5 слоёв HandLabeled + perm_water + сплиты,
~3,8 ГБ) — отдельно, если нужен целиком:

```bash
python3 scripts/download_sen1floods11.py            # HandLabeled + perm_water + сплиты, ~3,8 ГБ
```

Подробности и режимы (`--no-s2`, `--with-weak`) — `docs/data.md` §3а.

## Обучение: GPU локально или Colab

Одна и та же команда `python -m floodvalue train` (`docs/decisions.md` D-009);
различается только окружение — обучение вне `uv.lock`, инференс строго по нему:

- **GPU локально** (например RTX 4080): `make gpu-env` — `scripts/setup_gpu_env.sh`
  создаёт отдельный venv **вне репозитория** (по умолчанию
  `$HOME/.cache/floodvalue-gpu`, переопределяется `FLOODVALUE_GPU_VENV`), ставит
  GPU-сборку torch (индекс `cu128`) и `floodvalue -e .` поверх; `uv.lock` не трогает.
  Идемпотентно — повторный запуск на готовом окружении ничего не ломает и не качает
  заново. Затем обучение — `make train GPU=1` (через python из этого venv) вместо
  `make train` (CPU, `uv run --frozen`, для проверки на машине без GPU).
- **Colab**: `notebooks/train_colab.ipynb` — обёртка над `python -m floodvalue train`
  (issue P10b, ставит свой GPU-torch внутри ноутбука, `uv.lock` не касается).

Веса — `models/<method>/`, через Git LFS, sha256 в `run_metadata.json` — `models/README.md`.

## Воспроизведение

Статусы: **не реализовано** — команда падает с `stage '<этап>' not implemented yet`;
**в работе** — модуль есть, часть цепочки уже проходит, полный прогон ещё не собран;
**готово** — код завершён и проверен (`docs/decisions.md`, тесты модуля).

| Этап | Команда | Статус |
|---|---|---|
| Получение данных и QC | `make data` | готово (QC 446/446 чипов, `pytest tests/data`; сырые файлы — `scripts/download_sen1floods11.py`, см. выше) |
| Split (заморожен) | `make split` | готово (D-016, `pytest tests/split`) |
| Baseline: порог на train/val | `make baseline` | не реализовано |
| Обучение основного метода | `make train` (GPU: `make train GPU=1`; Colab: `notebooks/train_colab.ipynb`) | не реализовано |
| Калибровка и порог на validation | `make calibrate` | не реализовано |
| Inference на чипе (S1-only) | `make predict CHIP=<chip_id>` | не реализовано |
| Финальная оценка на test | `make eval-test` | не реализовано |
| 10 объектов и ущерб | `make assets CHIP=<chip_id>` | в работе (портфель и EL готовы, `pytest tests/assets`; ждёт p-растра от `predict`) |
| Зоны, цена, A/B/C | `make procure BUDGET=<руб>` | не реализовано (геометрия и цена по ПП 840 — чистые функции готовы, `pytest tests/procurement`; CLI-этап собирается) |
| 12 обязательных файлов | `make export && make validate` | не реализовано |
| Интерфейс оператора | `make app` | не реализовано |
| Всё на готовых весах | `make all CHIP=<chip_id> BUDGET=<руб>` | не реализовано (цепочка ждёт `predict`/`export`; `data` уже готов) |

`make check` (`test` + `hooks-test` + `validate`) — обязателен перед каждым push; пока
`floodvalue.export.validate` не реализован, `check` печатает причину и не роняет
остальные проверки (Makefile, issue area:export/contracts).

## Результаты

Метрики — `docs/verification/`. Решения — `docs/decisions.md`. Соответствие критериям —
`docs/tz-compliance.md`.
Оригиналы условий организатора (PDF, текст, выжимка) — `docs/task/`.

## Ограничения и развитие

TBD (критерий 20): split, один демо-чип, учебные V/q, доступность дополнительных
данных, отсутствие фактической покупки, перенос.
