# Baseline (issue #5, D-021): независимая сверка «было → стало»

Проверка выполнена независимым запуском `make baseline` в worktree `5-baseline`, commit
`0689648` (полный `068964818dba...`), 2026-09-25. Каталог данных —
`FLOODVALUE_DATA_DIR=<repo>/data`,
каталог выходов — временный scratch-каталог сессии (для `predict`, в git не попадает).
Test не читался, `make eval-test` не запускался.

## Команды

```bash
export FLOODVALUE_DATA_DIR=<repo>/data
export FLOODVALUE_OUTPUT_DIR=<scratch>/eval-out
uv sync --frozen
make baseline
uv run --frozen python -m floodvalue.baseline.predict --chip Sri-Lanka_713926
```

## Что сравнивалось

«Было» — состояние `main` до появления baseline-порога (issue #5): baseline не
рассчитывался, отчёта `docs/verification/baseline-val.md` и
`docs/verification/baseline-metrics.csv` не существовало — метрик нет, в таблице
прочерк.

«Стало» — вывод независимого запуска `make baseline` на этом коммите, сверенный с
зафиксированными в репозитории `docs/verification/baseline-val.md` и
`docs/verification/baseline-metrics.csv` (сгенерированы на commit `cbf7637832d6`).

## Таблица «было → стало» (ALL, по частям сплита)

Правило: вода ⇔ VH ≤ −23,0 dB (порог на train, канал выбран на dev, D-021).

| split | метрика | было | стало (мой запуск) | стало (в репо, baseline-val.md) | Δ / совпадение |
|---|---|---|---|---|---|
| train | IoU | — | 0.378 | 0.377983 | совпадает |
| train | F1 | — | 0.549 | 0.548603 | совпадает |
| train | Precision | — | 0.536 | 0.535804 | совпадает |
| train | Recall | — | 0.562 | 0.562028 | совпадает |
| train | валидных пикселей | — | 51517039 | 51517039 | совпадает |
| dev | IoU | — | 0.391 | 0.390798 | совпадает |
| dev | F1 | — | 0.562 | 0.561977 | совпадает |
| dev | Precision | — | 0.460 | 0.460026 | совпадает |
| dev | Recall | — | 0.722 | 0.721984 | совпадает |
| dev | валидных пикселей | — | 17638830 | 17638830 | совпадает |
| validation | IoU | — | 0.580 | 0.579962 | совпадает |
| validation | F1 | — | 0.734 | 0.734147 | совпадает |
| validation | Precision | — | 0.838 | 0.838217 | совпадает |
| validation | Recall | — | 0.653 | 0.653065 | совпадает |
| validation | валидных пикселей | — | 13983779 | 13983779 | совпадает |

Консольный вывод моего запуска (`make baseline`):

```
train      IoU 0.378 F1 0.549 P 0.536 R 0.562 n 51517039
dev        IoU 0.391 F1 0.562 P 0.460 R 0.722 n 17638830
validation IoU 0.580 F1 0.734 P 0.838 R 0.653 n 13983779
baseline: VH <= -23.0 dB, s = 2.1315 -> models/baseline/config.json
```

Полный лог: `<scratch>/baseline-run.log`. Полные значения по событиям (Ghana, India,
Pakistan, Paraguay, Somalia — train; USA — dev; Nigeria, Sri-Lanka — validation) —
сверены по `docs/verification/baseline-metrics.csv`: после `make baseline` файл
байт-в-байт не изменился (`git diff --stat` — пусто), т.е. числа по событиям в моём
запуске идентичны зафиксированным в репозитории.

Файлы `docs/verification/baseline-metrics.csv`, `baseline-variants.csv`,
`baseline-threshold-curve.csv`, `baseline-threshold-curve.png` после `make baseline` не
изменились (`git diff --stat` — пусто). Изменился только текст commit-хеша в
`docs/verification/baseline-val.md` (`cbf7637832d6` → `068964818dba`, дата коммита
запуска отличается от даты, зафиксированной в отчёте) и в `models/baseline/config.json`
(`"commit"` поле). Оба изменения откачены командой:

```bash
git checkout -- docs/verification/baseline-val.md models/baseline/config.json
```

`git status --porcelain` после отката — пусто, worktree чист.

## `predict --chip Sri-Lanka_713926`: проверка контракта p-растра

Команда:

```bash
uv run --frozen python -m floodvalue.baseline.predict --chip Sri-Lanka_713926
```

Вывод: `Sri-Lanka_713926: flood_probability.tif, flood_mask.tif, baseline_mask.tif -> <out>/Sri-Lanka_713926`.

Проверено скриптом на растрах результата (rasterio):

| проверка | результат |
|---|---|
| dtype `flood_probability.tif` | `float32` |
| nodata `flood_probability.tif` | `-9999.0` |
| dtype `flood_mask.tif` | `uint8` |
| значения `flood_mask.tif` | `{0, 1}` |
| nodata `flood_mask.tif` | `255.0` |
| сетка (transform) prob/mask vs S1-чип (`allclose`) | совпадает (равны с точностью 1e-12) |
| CRS prob/mask vs S1-чип | совпадает |
| shape | prob (512, 512), mask (512, 512), S1 (512, 512) — совпадает |
| диапазон p (валидные пиксели) | [5.87e-07; 0.99999964] ⊂ [0, 1] |
| NaN в S1-чипе Sri-Lanka_713926 | 0 (оба канала) — валидных пикселей 262144 из 262144 |
| `mask == (p ≥ 0.5)` на валидных пикселях | совпадает на 262144 / 262144 (100 %), несовпадений 0 |

Для этого конкретного чипа NaN в S1 отсутствует, поэтому nodata-ветка p (NaN → -9999 в
файле) не была упражнена данными этого чипа; сама схема файла (dtype/nodata/сетка)
соответствует контракту `docs/contracts.md`.

`prediction.json` для чипа:

```json
{
 "chip_id": "Sri-Lanka_713926",
 "method": "baseline_sigmoid (temporary main method, D-009)",
 "threshold": 0.5,
 "threshold_status": "fixed 0.5 (= baseline rule), not chosen on validation; temporary main method D-009, calibration replaces it",
 "baseline_threshold_db": -23.0,
 "channel": "VH",
 "scale_db": 2.1315,
 "config_sha256": "7f3ae5005f13091ed92af73d68b35527ba5d23ef3e65af56b697b570874b1e0a",
 "s1_file": "Sri-Lanka_713926_S1Hand.tif"
}
```

## Вывод

Независимый прогон `make baseline` на commit `068964818dba` воспроизводит числа
`docs/verification/baseline-val.md` и `baseline-metrics.csv` точно (совпадение по всем
15 метрикам ALL-строк train/dev/validation и по числу валидных пикселей); единственное
расхождение в tracked-файлах — строка commit-хеша, откачена. `predict --chip
Sri-Lanka_713926` отдаёт растры по контракту (dtype, nodata, сетка = S1, mask =
(p ≥ 0.5) на всех валидных пикселях чипа).
