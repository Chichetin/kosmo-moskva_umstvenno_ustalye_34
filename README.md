# FloodValue — карта воды по Sentinel-1 → ожидаемый ущерб → план заказа ДЗЗ

Космохакатон 2026. Прототип для **инфраструктурного оператора**: по официальному
Sentinel-1 чипу Sen1Floods11 строит вероятностную карту воды, оценивает ожидаемый
прямой ущерб 10 учебных объектов и выбирает дополнительную съёмку в пределах
бюджета (стратегии A/B/C).

> Каждый раздел ниже подтверждён командой и её фактическим выводом; где команды ещё
> нет — стоит «не реализовано» (таблица «Воспроизведение»). Прототип не является
> официальной оценкой ущерба; ставка 788,64 руб./км² — сценарная. Ограничения — раздел
> «Ограничения и развитие» и `docs/limitations.md`.

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
| Получение данных и QC | `make data` | готово — `data: 446 чипов, скачано файлов: 0`, `QC формата 446/446` (`docs/data.md` §11; `pytest tests/data`; сырые файлы — `scripts/download_sen1floods11.py`, см. выше) |
| Split (заморожен) | `make split` | готово — `split v1: train 237, dev 69, validation 60, test 75, excluded 5`, `OK: rebuilt split == splits/split_v1.json == docs/decisions.md` (D-016; `pytest tests/split`) |
| Baseline: порог на train/val | `make baseline` | готово — `baseline: VH <= -23.0 dB, s = 2.1315`, validation IoU 0,580 (D-021, `docs/verification/baseline-val.md`; `pytest tests/baseline`) |
| Обучение основного метода (U-Net, ансамбль 5 seed) | `make train` (GPU: `make train GPU=1`; Colab: `notebooks/train_colab.ipynb`) | готово — веса 5 членов в `models/unet/` через LFS, dev IoU 0,584 против baseline 0,391 (D-019, `docs/verification/unet-metrics.md`); команда обучения проверена смоук-прогоном (`--max-chips`, CPU и GPU, код 0), полный переобучающий прогон (~75 с/член на GPU) после обучения весов сдачи не повторялся (время и устройство — `docs/verification/unet-metrics.md` §«Воспроизведение») |
| Калибровка и порог на validation | `make calibrate` | готово — Platt-калибровка, порог 0,37 по IoU на validation (D-029, `docs/verification/calibration-val.md`); пишет `models/unet/calibration.json`, `predict` подхватывает его автоматически без правок вызова |
| Inference на чипе (S1-only, оба метода) | `FLOODVALUE_METHOD=unet\|baseline make predict CHIP=<chip_id>` | готово — на `Sri-Lanka_713926` (unet: `threshold 0.37, threshold_status "calibrated (platt) and chosen on validation, D-029"`, `valid_pixels 262144`; baseline: `seconds 0.0`) и на `Nigeria_1095404` (другой официальный чип, unet: `valid_pixels 262144`) |
| Финальная оценка на part (validation/dev/test) | `make eval-test` (сырое, по умолчанию `--part test` — запечатан); `uv run --frozen python -m floodvalue eval-test --part validation` | готово, **test честно запечатан**: `make eval-test` без подтверждения → `eval-test: отказ — test is sealed (D-010): set FLOODVALUE_EVAL_TEST_CONFIRM=<sha256 of the frozen configuration>; ...`; `--part validation` реально считает 60 чипов → `docs/verification/eval-validation.md` (калиброванная модель: IoU 0,685 против baseline 0,580, ECE 0,007, Brier 0,041, AUROC неопределённости 0,902); `docs/verification/test-log.md` по test пуст — test ещё не открывался (`eval-test` замороженной конфигурации — не позже 26.09 20:00, D-033) |
| 10 объектов и ущерб | `make assets CHIP=<chip_id>` | готово — `assets: 10 объектов, оценено 10` (D-018; `pytest tests/assets`) |
| Зоны, цена, A/B/C | `make procure BUDGET=<руб>` | готово — `procure: A/B/C` с ценой, охватом EL, ценой ошибки решения и `budget_feasible` (D-023; `pytest tests/procurement`). C покупает снимки там, где они могут изменить решение: max прироста R = V·q·min(p, 1−p) (сценарная «цена ошибки решения», не снижение потерь) на рубль корзины; объекты с почти известным исходом (p ≈ 0 или 1) не покупаются; прежнее правило «EL на рубль» — строка `C_by_EL` в `sensitivity.csv` (D-049). Поэтому доля покрытого EL у C может быть малой: объекты с исходом, почти известным по карте (p ≈ 1), идут на осмотр (D-035), а не на снимок |
| 12 обязательных файлов | `make export && make validate` | готово — реальный комплект `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → `12/12 файлов прошли проверку`, `flood_mask.tif`/`flood_probability.tif` — с калиброванным порогом 0,37, `metrics.csv` — с калиброванными метриками main (D-027, D-029; `pytest tests/contracts tests/export`); `run_metadata.warnings` пуст на чистом дереве — `make all CHIP=Sri-Lanka_713926 BUDGET=5000` на commit `89c8e4f` (26.09) → `warnings: []`, validate 12/12 (ранее было предупреждение про рассинхрон порога метрик, снято подключением калибратора в `export`) |
| Интерфейс оператора | `make app` | готово — React + MapLibre, карта Земли, слои и пересчёт бюджета (`pytest tests/app`) |
| Отчёт QC данных | `make data-qc` | готово — `docs/verification/data-qc.md` из `interim/qc.csv` после `make data` |
| Комплект и веса для релиза | `make release OUT=<комплект> [TAG=<тег>]` | готово — `run_<run_id>.zip`, `models_<commit>.zip`, `SHA256SUMS` (D-037, `pytest tests/test_release_bundle.py`); загрузка финального комплекта — после `eval-test` |
| Перенос рецепта U-Net между событиями (LOEO) | `python -m floodvalue.model.unet.loeo` через `notebooks/loeo_colab.ipynb` (D-036) | прогнан `MODE=full` на commit `1eaf2cc9e572` (локально на RTX 4080 вместо Colab, 6 фолдов): IoU U-Net на невиданном событии 0,206–0,532, ΔIoU к порогу фолда значимо > 0 в 4 из 6 фолдов — `docs/verification/transfer-loeo.md` (таблица из отчёта прогона в #86, CSV/JSON фолдов не приложены); проводка — `pytest tests/model/test_loeo.py tests/model/test_loeo_notebook.py` |
| Всё на готовых весах | `make all CHIP=<chip_id> BUDGET=<руб>` | готово — `make all CHIP=Sri-Lanka_713926 BUDGET=5000` собирает `outputs/Sri-Lanka_713926_5000_<commit>/` (predict уже калиброван) и проходит `validate` 12/12 (`app` в цепочку не входит; `calibrate` — отдельный шаг, записывает веса в `models/unet/calibration.json` заранее) |

Статусы получены реальными запусками перечисленных команд в PR #60 (`.env` из
корня, `set -a; . ./.env; set +a`; полный вывод — в описании PR); при вливании PR
сверено, что каждая команда таблицы существует в `Makefile` и в
`uv run --frozen python -m floodvalue --help` текущего `main`, числа метрик — с
`docs/verification/` после калибровки (D-029).

`make check` (`test` + `hooks-test` + `validate`) — обязателен перед каждым push; без
`OUT=` `validate` проверяет самый свежий комплект в `outputs/`, если он есть, иначе
эталон `tests/fixtures/good` (12/12); `make validate OUT=<комплект>` — конкретный
собранный комплект. Чистый клон GitVerse `6fe66f0`: `uv sync --frozen && uv run --frozen
pytest` → 473 passed, `make all` → validate 12/12 (`docs/verification/clean-clone-gitverse.md`).
На commit `89c8e4f` (26.09) `make check` → 646 passed, 4 skipped, `test-hooks.sh` OK,
`validate` 12/12. Число тестов растёт с каждым PR.

## Интерфейс оператора

Запустите `make app` и откройте `http://127.0.0.1:8000`. Интерфейс
показывает исходный S1, вероятность воды и маску, объекты с ожидаемым ущербом,
кандидатные и выбранные зоны, сравнение A/B/C и сценарии чувствительности. Бюджет
можно менять числом или ползунком: пересчитывается только план закупки, без запуска
модели и изменения растров. Кнопка «Скачать комплект» выгружает архив результатов.
Начальный экран — интерактивная Земля MapLibre. Маркеры и поиск показывают реальные
локальные наблюдения Sen1Floods11; прямой выбор точки открывает её территорию на той
же карте. Каталог читает даты из `data/interim/source_manifest_s1.json`, а границы и
центр — из геопривязки каждого GeoTIFF. Сейчас локально доступны 446 исходных S1,
но только один комплект с вероятностью воды, объектами и планом. Для остальных
снимков приложение не подставляет чужие расчёты. Сцены без геопривязки остаются в
поиске, но не появляются как точка на Земле. При отсутствии готовых результатов и
сырого каталога приложение показывает пустой каталог; тестовый эталон доступен
только при явном указании `FLOODVALUE_APP_RUN_DIR=tests/fixtures/good`.

В локальном режиме доступны географическая карта OpenFreeMap / OpenStreetMap и
оптическая мозаика EOX Sentinel-2 cloudless за 2019 год, а отдельно — исходный S1,
вероятность воды, маска, объекты и зоны. Внешняя оптика служит только контекстом:
она не является снимком паводка за 2017 год и не входит в модель. Мозаика 2019 года
распространяется на условиях CC BY-NC-SA 4.0; её использование ограничено
некоммерческими сценариями ([условия EOX](https://cloudless.eox.at/documentation/license),
[схема WMTS](https://cloudless.eox.at/documentation/usage)). Для обычной карты действуют
[условия сервиса OpenFreeMap](https://openfreemap.org/tos/) и атрибуция OpenStreetMap. Внешние тайлы требуют сети; локальные результаты и
аналитика доступны при их отказе. Смена бюджета пересчитывает только план закупки.

Раздел «Анализ» показывает полный портфель, каталог зон, сравнение A/B/C и
чувствительность. Прямая ссылка `#/scene/<chip_id>` открывает сцену, а
`#/analysis/<chip_id>` — её анализ; кнопки браузера сохраняют этот переход.

Для открытия конкретного комплекта: `make app OUT=outputs/<run_id>`. Порт задаётся
`FLOODVALUE_APP_PORT`. Собранный React-бандл хранится в репозитории, поэтому для
`make app` нужен только Python из `uv.lock`; после правки `web/` пересоберите его
командой `make web` (Node.js нужен только для сборки).
Для просмотра изменений во время разработки запустите `make app` в одном терминале
и `cd web && npm install && npm run dev` в другом, затем откройте
`http://127.0.0.1:5173`. Скриншоты: [Земля](docs/verification/ui-global.png),
[оптическая подложка](docs/verification/ui-territory-optical.png),
[вероятность воды](docs/verification/ui-territory-probability.png),
[анализ](docs/verification/ui-analysis.png).

## S1-only и выбор метода

Основной метод обязан работать только по Sentinel-1 (без S2/DEM/JRC); переменная
`FLOODVALUE_METHOD` выбирает метод и для обучения, и для inference (по умолчанию
`unet`, альтернатива — временный `baseline`, D-009/D-019):

```bash
uv run --frozen python -m floodvalue predict --chip Sri-Lanka_713926
# {"method": "unet", ..., "threshold": 0.37, "threshold_status": "calibrated (platt) and chosen on validation, D-029", "valid_pixels": 262144}
FLOODVALUE_METHOD=baseline uv run --frozen python -m floodvalue predict --chip Sri-Lanka_713926
# {"method": "baseline", ..., "valid_pixels": 262144, "seconds": 0.0}
```

Оба варианта равнозначно работают через `make`: `FLOODVALUE_METHOD=baseline make
predict CHIP=Sri-Lanka_713926` и `FLOODVALUE_METHOD=unet make predict
CHIP=Sri-Lanka_713926` — оба дают код 0 на commit `89c8e4f` (26.09; baseline — `threshold` 0.5, unet — 0.37). Inference на другом официальном чипе
(не только на демонстрационном) проверен на `Nigeria_1095404`
(`uv run --frozen python -m floodvalue predict --chip Nigeria_1095404`, вход только
`S1Hand`, `valid_pixels: 262144`) — до заморозки функций чипы `test`-части (Spain,
Mekong, Bolivia) не используются нигде, кроме будущего `make eval-test` (D-010,
раздел «Ограничения и развитие»). Полный сквозной путь до 12 файлов на другом чипе —
подраздел ниже.

### Любой S1-файл вне HandLabeled (например S1Weak)

- любой из 446 HandLabeled: `make all CHIP=<chip_id> BUDGET=<руб>` (подраздел ниже);
- любой S1-файл: `python -m floodvalue predict --s1 <path>` (или `make predict S1=<path>`).

`--s1` принимает любой 2-канальный S1 GeoTIFF (VV, VH в dB, NaN = нет данных), например
официальный `WeaklyLabeled/S1Weak/<chip>_S1Weak.tif`, и пишет `flood_probability.tif`,
`flood_mask.tif`, `uncertainty.tif`, `flood_probability_range.tif`, `prediction.json` на
сетке входного файла в `$FLOODVALUE_OUTPUT_DIR/<chip_id>` (`chip_id` — имя файла до
`_S1`; `--out` переопределяет каталог). Метод, калибратор и порог — те же, что у
`--chip`; метка не читается, метрики не считаются, в `prediction.json` —
`"label_status": "no_label"`. Проверено на `India_1029501_S1Weak.tif` (train-событие) —
`docs/verification/predict-s1.md`:

```bash
uv run --frozen python -m floodvalue predict --s1 data/raw/WeaklyLabeled/S1Weak/India_1029501_S1Weak.tif
# {"chip_id": "India_1029501", "method": "unet", "s1_file": "India_1029501_S1Weak.tif", ..., "threshold": 0.37, ..., "valid_pixels": 262144, ..., "label_status": "no_label"}
```

Граница: `assets`/`procure`/`export` (и `make all`) пока читают S1 только из
`HandLabeled/S1Hand` — сквозной путь до 12 файлов на файле вне HandLabeled не
поддержан, только карта `predict`.

### Полный запуск на другом официальном чипе

Жюри может самостоятельно повторить inference на любом другом официальном S1-чипе
Sen1Floods11 (task.md, «Проверка на защите») — без подгонки: метод, калибратор и порог
зафиксированы в `models/unet/` и не зависят от `--chip`.

```bash
set -a; . ./.env; set +a
make all CHIP=<chip_id> BUDGET=<руб>     # например CHIP=Nigeria_81933 BUDGET=5000
make validate OUT=outputs/<chip_id>_<руб>_<commit>
```

Основной метод и baseline читают **только** `S1Hand` (VV/VH) — ни `S2Hand`, ни
`JRCWaterHand`, ни метка не нужны и не читаются на этапах `predict`/`assets`/`procure`
(проверено запуском на изолированном каталоге данных без единого файла S2/JRC —
`docs/verification/transfer-val.md` §4). Перенос на 4 официальных чипах из
`dev`/`validation` (USA, Nigeria, Sri-Lanka, включая чип с 66,6 % nodata в S1) — метрики,
причины ошибок и картинки в `docs/verification/transfer-val.md`; `12/12` на каждом.

`decision_at`/`decision_deadline` (сроки для A/B/C) берутся от даты S1 обрабатываемого
чипа (D-043, #84): например, `USA_198411` → `2019-05-22T23:59:59+00:00` /
`2019-05-23T11:59:59+00:00` — правка `configs/run.yaml` при смене чипа не нужна
(`docs/verification/transfer-val.md` §7).

## Результаты

Метрики — `docs/verification/`. Решения — `docs/decisions.md`. Соответствие критериям —
`docs/tz-compliance.md`.
Оригиналы условий организатора (PDF, текст, выжимка) — `docs/task/`.

### A/B/C при бюджете 5 000 руб.: что это значит для оператора

Демо-чип validation `Sri-Lanka_713926`, `make all CHIP=Sri-Lanka_713926 BUDGET=5000` на
`9040ece` (26.09) → `strategy_comparison.csv`; те же числа —
`docs/verification/procurement-catalog.md` §5, строка «5 000 / стало». Все рубли — учебные
V·q и сценарная ставка Б (`base_rate_status=scenario`), не фактические потери.

- **C** (в бюджете, 4 207,54 руб.: E-2, Z-A03) покрывает лишь 8,7 % ожидаемого ущерба
  (1 462 348,49 из 16 842 207,70 руб.), но 70,6 % ожидаемой цены ошибки решения
  R = V·q·min(p, 1−p) (1 462 348,49 из 2 070 290,02 руб.). Это сценарная ценность информации
  (D-049), а не снижение потерь: снимок нужен там, где карта не знает ответа (A08, p ≈ 0,38).
- **Объекты с p > 0,97** (A10 — 0,997, A02 — 0,978, A01 — 0,970; `asset_loss.csv`) —
  основная часть EL, но исход по карте почти известен: их ставят первыми **на осмотр**
  (D-035), а снимок их решения не изменит. Прежнее правило «EL на рубль» купило бы Z-A02,
  Z-A10 (EL 12 610 565,84 руб., R лишь 189 434,16) — строка `C_by_EL` в `sensitivity.csv`.
- **B** (весь чип, W-CHIP) покрывает все 10 объектов и 100 % R, но стоит 42 465,93 руб. —
  сверх бюджета (`budget_feasible=false`), это **контрфактическая** стратегия для сравнения,
  а не рекомендация. **A** — ничего не покупать (0 руб. данных).

### Комплект и веса в релизе

`outputs/` и `*.tif` в git не входят, поэтому 12 файлов демо-запуска и веса `models/`
лежат ассетами релиза GitVerse: `run_<run_id>.zip`, `models_<commit>.zip` и `SHA256SUMS`
(D-037). Проверка после скачивания всех трёх файлов в один каталог:

```bash
sha256sum -c SHA256SUMS      # обе строки — OK
```

**Плейсхолдер до финального запуска** — значения появятся после `eval-test` и `make all`
на демо-чипе:

```text
<sha256 — ещё не собрано>  run_<run_id>.zip
<sha256 — ещё не собрано>  models_<commit>.zip
```

Сборка у себя: `make release OUT=outputs/<run_id>` → `outputs/release/` (отказ, если
`validate` не 12/12 или в `models/` LFS-указатель вместо весов; повторная сборка даёт тот же
sha256). Загрузка в релиз: `make release OUT=outputs/<run_id> TAG=<тег>` (токен — `.env`).

## Стоимость данных и нормативные допущения

Цена зон дополнительной съёмки считается по учебному сценарию ПП РФ № 840 от
29.06.2019 в ред. ПП РФ № 1297 от 27.08.2025 (`configs/pricing.yaml`,
`legal_checked_at` — дата самостоятельной проверки редакции). Ставка
Б = 788,64 руб./км² — **сценарная** (`base_rate_status=scenario`), не официальный
тариф: методика Роскосмоса (Приказ № 116 от 21.05.2026) определяет расчёт стоимости
базовой расчётной единицы, а не публикует конкретное число рублей. Сценарий принят для
частного коммерческого пользователя (`usage_type=internal`, D-001) — особый порядок и
льготы для органов государственной власти в расчёте не применены. Полный разбор
источников (URL, дата проверки, что подтверждено первоисточником и что остаётся
сценарным допущением) — `docs/pp840.md`; трактовки применения формулы — `docs/decisions.md`
D-017.

Момент решения — конец суток съёмки S1 обрабатываемого чипа (UTC; дата — из метаданных
события Sen1Floods11), срок доступности данных — через 12 ч; давность снимков и год группы
скидки считаются от этой даты (D-043, `run_metadata.json` → `budget`, `pricing.as_of_rule`).
Формула, коэффициенты и ставка Б при этом — по проверенной редакции 2025 года, которой на
дату события (2016–2019) могло не быть: это гибридный учебный сценарий, не историческая цена.

## Ограничения и развитие

Прототип демонстрирует цепочку «S1 → вероятность → ущерб портфеля → план съёмки» на
**одном** S1-чипе с учебными данными и не является официальной оценкой ущерба, тарифом
или готовым к покупке планом. Полный разбор с командами и ссылками — `docs/limitations.md`.

- **Развитие — запланировано, не реализовано** (после сдачи, вне 20 критериев): мониторинг СМИ
  и проверка сообщений о паводках — спецификация `docs/features/media-monitoring.md`, план
  `docs/plans/media-monitoring-mvp.md`.

- **Split** — по событиям, не по чипам (D-016). Dev — единственное событие USA (4,6 %
  воды): на нём одном выбраны вариант метода и порог baseline, поэтому оценка
  устойчивости даётся бутстрэпом по чипам, не по независимым событиям. Validation
  используется дважды — для отчёта (`python -m floodvalue eval-test --part validation`, commit `af52cc72a2d6` —
  `docs/verification/eval-validation.md`) и как единственная выборка для калибровки и выбора порога
  (`make calibrate` — Platt, порог 0,37, D-029, `docs/verification/calibration-val.md`). **Test
  запечатан корректно**: `make eval-test` без
  `FLOODVALUE_EVAL_TEST_CONFIRM` реально отказывает («test is sealed»),
  `docs/verification/test-log.md` пока пуст — test ещё не открывался. Сводные доли
  меток (включая test) смотрели до разбиения и они повлияли на выбор test-событий и
  демо-чипа (`docs/data.md` §10, D-016) — не на модель и пороги.
- **Один демонстрационный чип.** До `eval-test` замороженной конфигурации (не позже
  26.09 20:00 МСК, D-033) — validation `Sri-Lanka_713926`; после — test `Spain_7370579`,
  только замороженной конфигурацией с записью в журнал (D-010).
- **«Карта воды», не «временное затопление»**: постоянная вода не исключена — слой
  JRCWaterHand набора, вероятно, включает наблюдения после дат событий 2016–2018
  (`docs/data.md` §7, D-007).
- **Учебные V/q.** EL = p·V·q — прямой ущерб одного события на одном учебном
  портфеле, не годовой ожидаемый ущерб, не страховое возмещение и не официальная
  оценка (task.md, «Границы применения»). Прокси-проверка сверяет модель с ручными
  метками воды, а не с фактическими потерями объектов; на validation она уже
  выполнена и показывает MAE прогноза против прокси-ущерба
  (`docs/verification/eval-validation.md` §5), но это выборка для настройки, а не
  независимая. `export` берёт `run_metadata.proxy_check` из сводки `eval-test` той части
  split, где лежит чип (`docs/verification/eval-<part>-summary.json`, для test —
  `test-eval-summary.json`), только если в сводке тот же чип, метод, порог и веса `models/`
  (#107). Для validation-чипа числа переносятся, но статус остаётся `planned`: выборка
  настроечная (`make all CHIP=Sri-Lanka_713926 BUDGET=5000` на `9040ece`, 26.09 →
  `proxy_check.status=planned`, `result.estimated=true`, `independent=false`, MAE 274 285,66
  руб. на 10 точках демо-чипа, источник `docs/verification/eval-validation-summary.json`).
  `status=done` бывает только у test-чипа после `eval-test` замороженной конфигурации; до
  него на test проверка не выполнялась вовсе, и у test-чипа статус `planned`.
- **Стоимость данных — сценарная.** Ставка Б = 788,64 руб./км² (`base_rate_status=scenario`)
  и сроки `available_at` не подтверждены поставщиком (D-026, `docs/pp840.md`). Покупка
  ни разу не совершалась: A/B/C — расчётные стратегии на одной карте, и ни одна не
  предотвращает реальные потери (task.md); B сверх бюджета — контрфактическая
  (`budget_feasible=false`). Без новых наблюдений неопределённость B/C не «снижается»
  автоматически: у B/C статус `scenario` — сумма неопределённости непокрытых объектов,
  оптимистичная граница «снимок снял бы неопределённость покрытых», а не измеренное
  снижение (D-038 п. 5).
- **Перенос.** Выдаваемая модель — U-Net с калибровкой Platt и порогом 0,37,
  подобранными на validation (D-029): на validation IoU 0,685 против baseline 0,580, на
  Sri-Lanka 0,621 против 0,585 (разница не значима, CI [−0,010; 0,079]) —
  `docs/verification/eval-validation.md` §1; числа validation оптимистичны (она же
  настроечная). На событии с малой долей воды калибровка **ухудшает** маску: dev (USA,
  4,6 % воды) IoU 0,584 → 0,467 (`docs/verification/calibration-val.md`), p и EL
  завышены; на чипах validation с долей воды < 10 % IoU около 0,27
  (`eval-validation.md` §7). Известные условия снижения качества по одному S1: городская застройка и яркие
  рассеиватели (VV до +24,7 dB в train, на других событиях набора выше — `docs/limitations.md` §1, §8), вода под пологом
  растительности или при ветровой ряби, невидимая в SAR (`docs/data.md` §8 п. 13),
  тени рельефа (layover/shadow — общее свойство SAR, отдельно на этом наборе не
  измерялось), и распределения dB/доли воды, сильно отличные от train. `make
  eval-test` реализован и запечатан на test правильно (`make eval-test` без
  `FLOODVALUE_EVAL_TEST_CONFIRM` на `89c8e4f`, 26.09 → «отказ — test is sealed (D-010)», код 2;
  `--part validation` считает 60 чипов — `docs/verification/eval-validation.md`);
  перенос на test (новые события Spain/Mekong/Bolivia) сам по себе ещё не проверен —
  это следующий шаг после заморозки функций, а не нереализованный код.
- **Доступность дополнительных данных.** Лицензия официального Sen1Floods11 явно не
  указана (в STAC-коллекциях `"license": "proprietary"`, LICENSE в репозитории нет,
  `docs/data.md` §1); данные набора опубликованы на 1–4 года позже дат событий
  (`published_after_event: true`, `docs/data.md` §11).
- **Не используем** (подробно — `docs/limitations.md` §11): weak-метки набора
  (`S1OtsuLabelWeak`, `S2IndexLabelWeak`; время, weak test-событий исключены D-006);
  Prithvi — оптический (S2/HLS), дообучен на авторском split, чей train содержит чипы наших test-событий;
  JRCWaterHand — вероятно, информация из будущего. Кривые «глубина → ущерб» (JRC 2017)
  — развитие: нужна глубина, которой нет в карте S1. ПП 840 регулирует только данные
  федерального фонда ДЗЗ. IoU несравним со статьёй: у нас pooled по пикселям на split
  по событиям и «вся вода», в литературе — часто среднее по чипам на авторском split.
- **Режим S1-only обязателен и поддержан**: `FLOODVALUE_METHOD=unet|baseline` — см.
  «S1-only и выбор метода» выше; `predict` на `Sri-Lanka_713926` и `Nigeria_1095404` на
  `89c8e4f` (26.09) — код 0, вход только `S1Hand`.

План проверки: `make eval-test --part test` для замороженной конфигурации (D-033) (код и запечатывание
уже готовы: отказ без confirm-хеша проверен `make eval-test` на `89c8e4f` — следующий шаг чисто процедурный, не разработка),
затем перенос на события вне Sen1Floods11 и на реальный реестр объектов инфраструктуры
вместо учебных V/q; пилот с инфраструктурным оператором (D-001) с метриками из task.md
— время от данных до решения, доля пропущенных значимых объектов, доля ручной
проверки; реальные котировки поставщика ДЗЗ вместо сценарной ставки Б. Подробности и
источники — `docs/limitations.md`.
