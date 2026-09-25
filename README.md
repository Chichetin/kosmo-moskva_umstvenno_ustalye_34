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
| Обучение основного метода (U-Net, ансамбль 5 seed) | `make train` (GPU: `make train GPU=1`; Colab: `notebooks/train_colab.ipynb`) | готово — веса 5 членов в `models/unet/` через LFS, dev IoU 0,584 против baseline 0,391 (D-019, `docs/verification/unet-metrics.md`); команда обучения проверена смоук-прогоном (`--max-chips`, CPU и GPU, код 0), полный переобучающий прогон (~75 с/член на GPU) в этом PR не повторялся — `models/` вне разрешённых путей issue |
| Калибровка и порог на validation | `make calibrate` | готово — Platt-калибровка, порог 0,37 по IoU на validation (D-029, `docs/verification/calibration-val.md`); пишет `models/unet/calibration.json`, `predict` подхватывает его автоматически без правок вызова |
| Inference на чипе (S1-only, оба метода) | `FLOODVALUE_METHOD=unet\|baseline make predict CHIP=<chip_id>` | готово — на `Sri-Lanka_713926` (unet: `threshold 0.37, threshold_status "calibrated (platt) and chosen on validation, D-029"`, `valid_pixels 262144`; baseline: `seconds 0.0`) и на `Nigeria_1095404` (другой официальный чип, unet: `valid_pixels 262144`) |
| Финальная оценка на part (validation/dev/test) | `make eval-test` (сырое, по умолчанию `--part test` — запечатан); `uv run --frozen python -m floodvalue eval-test --part validation` | готово, **test честно запечатан**: `make eval-test` без подтверждения → `eval-test: отказ — test is sealed (D-010): set FLOODVALUE_EVAL_TEST_CONFIRM=<sha256 of the frozen configuration>; ...`; `--part validation` реально считает 60 чипов → `docs/verification/eval-validation.md` (ECE 0,035, Brier 0,048, AUROC неопределённости 0,890); `docs/verification/test-log.md` по test пуст — test ещё не открывался (заморозка не наступила, D-022) |
| 10 объектов и ущерб | `make assets CHIP=<chip_id>` | готово — `assets: 10 объектов, оценено 10` (D-018; `pytest tests/assets`) |
| Зоны, цена, A/B/C | `make procure BUDGET=<руб>` | готово — `procure: A/B/C` с ценой, охватом EL и `budget_feasible` (D-023; `pytest tests/procurement`) |
| 12 обязательных файлов | `make export && make validate` | готово — реальный комплект `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → `12/12 файлов прошли проверку`, `flood_mask.tif`/`flood_probability.tif` — с калиброванным порогом 0,37, `metrics.csv` — с калиброванными метриками main (D-027, D-029; `pytest tests/contracts tests/export`); `run_metadata.warnings` пуст на чистом дереве — проверено в этом PR (ранее было предупреждение про рассинхрон порога метрик, снято подключением калибратора в `export`) |
| Интерфейс оператора | `make app` | не реализовано — `stage 'app' not implemented yet (floodvalue.app.run); issue area:app` |
| Всё на готовых весах | `make all CHIP=<chip_id> BUDGET=<руб>` | готово — `make all CHIP=Sri-Lanka_713926 BUDGET=5000` собирает `outputs/Sri-Lanka_713926_5000_<commit>/` (predict уже калиброван) и проходит `validate` 12/12 (`app` в цепочку не входит; `calibrate` — отдельный шаг, записывает веса в `models/unet/calibration.json` заранее) |

Статусы получены реальными запусками перечисленных команд в этом PR (`.env` из
корня, `set -a; . ./.env; set +a`); полный вывод каждой — в описании PR.

`make check` (`test` + `hooks-test` + `validate`) — обязателен перед каждым push; без
`OUT=` `validate` проверяет самый свежий комплект в `outputs/`, если он есть, иначе
эталон `tests/fixtures/good` (12/12); `make validate OUT=<комплект>` — конкретный
собранный комплект. Проверено в этом PR: `make check` зелёный после слияния #58
«eval-test», #59 «calibration» и #61 «validator» (точное число тестов растёт с
каждым PR — см. вывод `pytest` в актуальном CI/PR), `test-hooks.sh` 40/40, `validate`
12/12.

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
CHIP=Sri-Lanka_713926` — проверено в этом PR. Inference на другом официальном чипе
(не только на демонстрационном) проверен на `Nigeria_1095404`
(`uv run --frozen python -m floodvalue predict --chip Nigeria_1095404`, вход только
`S1Hand`, `valid_pixels: 262144`) — до заморозки функций чипы `test`-части (Spain,
Mekong, Bolivia) не используются нигде, кроме будущего `make eval-test` (D-010,
раздел «Ограничения и развитие»).

## Запуск на другом официальном чипе / S1-only

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

**Важно:** `decision_at`/`decision_deadline` (сроки для A/B/C) берутся из
`configs/run.yaml` и не подстраиваются под `--chip` автоматически — для реалистичного
сценария A/B/C на чипе другого события отредактируйте их в `configs/run.yaml` перед
`make procure`/`make all` (подробности — `docs/verification/transfer-val.md` §7; привязка к дате S1 чипа — #84).
`flood_probability.tif`/`flood_mask.tif`/`asset_loss.csv` от этого не зависят.

## Результаты

Метрики — `docs/verification/`. Решения — `docs/decisions.md`. Соответствие критериям —
`docs/tz-compliance.md`.
Оригиналы условий организатора (PDF, текст, выжимка) — `docs/task/`.

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

## Ограничения и развитие

Прототип демонстрирует цепочку «S1 → вероятность → ущерб портфеля → план съёмки» на
**одном** S1-чипе с учебными данными и не является официальной оценкой ущерба, тарифом
или готовым к покупке планом. Полный разбор с командами и ссылками — `docs/limitations.md`.

- **Split** — по событиям, не по чипам (D-016). Dev — единственное событие USA (4,6 %
  воды): на нём одном выбраны вариант метода и порог baseline, поэтому оценка
  устойчивости даётся бутстрэпом по чипам, не по независимым событиям. Validation
  используется дважды — для отчёта (`make eval-test --part validation`, реально
  выполнено в этом PR) и как единственная выборка для калибровки и выбора порога
  (`make calibrate` — Platt, порог 0,37, D-029, реально выполнено в этом PR). **Test
  запечатан корректно**: `make eval-test` без
  `FLOODVALUE_EVAL_TEST_CONFIRM` реально отказывает («test is sealed»),
  `docs/verification/test-log.md` пока пуст — test ещё не открывался. Сводные доли
  меток (включая test) смотрели до разбиения и они повлияли на выбор test-событий и
  демо-чипа (`docs/data.md` §10, D-016) — не на модель и пороги.
- **Один демонстрационный чип.** До заморозки функций (план — 26.09 20:30 МСК, D-022)
  — validation `Sri-Lanka_713926`; после — test `Spain_7370579`, только замороженной
  конфигурацией с записью в журнал (D-010).
- **«Карта воды», не «временное затопление»**: постоянная вода не исключена — слой
  JRCWaterHand набора, вероятно, включает наблюдения после дат событий 2016–2018
  (`docs/data.md` §7, D-007).
- **Учебные V/q.** EL = p·V·q — прямой ущерб одного события на одном учебном
  портфеле, не годовой ожидаемый ущерб, не страховое возмещение и не официальная
  оценка (task.md, «Границы применения»). Прокси-проверка сверяет модель с ручными
  метками воды, а не с фактическими потерями объектов; на validation она уже
  выполнена и показывает MAE прогноза против прокси-ущерба
  (`docs/verification/eval-validation.md` §5), но это выборка для настройки, а не
  независимая. В `run_metadata.json` собранного комплекта поле `proxy_check` пока
  `"status": "planned"` в любом случае (проверено на калиброванном прогоне в этом PR)
  — `export` ещё не сшит с протоколом `eval-test`, а на замороженном test проверка
  не выполнялась вовсе.
- **Стоимость данных — сценарная.** Ставка Б = 788,64 руб./км² (`base_rate_status=scenario`)
  и сроки `available_at` не подтверждены поставщиком (D-026, `docs/pp840.md`). Покупка
  ни разу не совершалась: A/B/C — расчётные стратегии на одной карте, и ни одна не
  предотвращает реальные потери (task.md); B сверх бюджета — контрфактическая
  (`budget_feasible=false`). Без новых наблюдений неопределённость B/C не «снижается»
  автоматически — она помечена `not_estimated`, а не пересчитанным меньшим числом
  (D-023).
- **Перенос.** На validation U-Net в среднем не хуже baseline (0,600 против 0,580,
  разница не значима), но **на Sri-Lanka хуже** (0,526 против 0,585,
  `docs/verification/unet-metrics.md`) — сеть занижает p на событии с большей долей
  воды. Известные условия снижения качества по одному S1: городская застройка и яркие
  рассеиватели (VV до +36,8 dB, `docs/data.md` §8, §11), вода под пологом
  растительности или при ветровой ряби, невидимая в SAR (`docs/data.md` §8 п. 13),
  тени рельефа (layover/shadow — общее свойство SAR, отдельно на этом наборе не
  измерялось), и распределения dB/доли воды, сильно отличные от train. `make
  eval-test` реализован и запечатан на test правильно (проверено в этом PR: `--part
  test` без confirm-хеша — отказ; `--part validation` реально считает 60 чипов);
  перенос на test (новые события Spain/Mekong/Bolivia) сам по себе ещё не проверен —
  это следующий шаг после заморозки функций, а не нереализованный код.
- **Доступность дополнительных данных.** Лицензия официального Sen1Floods11 явно не
  указана (в STAC-коллекциях `"license": "proprietary"`, LICENSE в репозитории нет,
  `docs/data.md` §1); данные набора опубликованы на 1–4 года позже дат событий
  (`published_after_event: true`, `docs/data.md` §11).
- **Режим S1-only обязателен и поддержан**: `FLOODVALUE_METHOD=unet|baseline` — см.
  «S1-only и выбор метода» выше; проверено на двух официальных чипах в этом PR.

План проверки: `make eval-test --part test` после заморозки функций (код и запечатывание
уже готовы и проверены в этом PR — следующий шаг чисто процедурный, не разработка),
затем перенос на события вне Sen1Floods11 и на реальный реестр объектов инфраструктуры
вместо учебных V/q; пилот с инфраструктурным оператором (D-001) с метриками из task.md
— время от данных до решения, доля пропущенных значимых объектов, доля ручной
проверки; реальные котировки поставщика ДЗЗ вместо сценарной ставки Б. Подробности и
источники — `docs/limitations.md`.
