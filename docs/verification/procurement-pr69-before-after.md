# Procurement PR #69 (zones-full, issue #22): «было → стало» на validation-чипе Sri-Lanka_713926

Проверка независимым запуском `make all` в двух worktree, 2026-09-26. Test-чипы не
читались, `make eval-test` не запускался.

## Важная поправка к брифу: `origin/main` уже содержит PR #69

На момент запуска `origin/main` (`b8903a0`) — это уже **слитый** PR #69
(`git merge-base --is-ancestor 4377c33 origin/main` → true; `b8903a0^2` =
`f76741d`, один коммит после `4377c33`, только текст примечания в метаданных, код не
меняет: `git diff --stat 4377c33 f76741d` → `src/floodvalue/procurement/run.py | 7
+++++--`, поле `note`). PR #89 (issue #25, `62afcca`) влит **раньше** PR #69:
`b8903a0^1 = 62afcca`, и `62afcca` — предок `4377c33`. Разница кода между сторонами
сравнения — только `src/floodvalue/procurement/` и `configs/run.yaml`.

Сравнение «worktree `origin/main` vs worktree PR #69» в буквальном виде дало бы
`before == after` (обе стороны содержат код PR #69). Чтобы таблица была осмысленной,
«было» взято как **`b8903a0^1` = `62afcca`** (main непосредственно перед мерджем PR
#69, но после ранее слитого PR #89) — это последний коммит в истории `origin/main`,
не содержащий изменений PR #69 в `src/floodvalue/procurement/`. «Стало» — worktree PR
#69, `HEAD 4377c33` (как указано в задании; отличие от финального `f76741d` —
только текстовое примечание, числа не меняет).

## Команды

```bash
# «было»
git -C .claude/worktrees/land-69 worktree add --detach <scratch>/main-wt 62afcca
cd <scratch>/main-wt && uv sync --frozen
FLOODVALUE_DATA_DIR=<repo>/data FLOODVALUE_OUTPUT_DIR=<scratch>/out-main \
  make all CHIP=Sri-Lanka_713926 BUDGET=2500  OUT=<scratch>/out-main/sl_2500
  make all CHIP=Sri-Lanka_713926 BUDGET=5000  OUT=<scratch>/out-main/sl_5000
  make all CHIP=Sri-Lanka_713926 BUDGET=20000 OUT=<scratch>/out-main/sl_20000

# «стало» (worktree PR #69, HEAD 4377c33)
cd .claude/worktrees/land-69
# outputs-land69 — временный каталог вне git (после прогона перенесён в scratchpad сессии)
FLOODVALUE_DATA_DIR=<repo>/data FLOODVALUE_OUTPUT_DIR=.claude/worktrees/land-69/outputs-land69 \
  make all CHIP=Sri-Lanka_713926 BUDGET=2500  OUT=outputs-land69/sl_2500
  make all CHIP=Sri-Lanka_713926 BUDGET=20000 OUT=outputs-land69/sl_20000
  # BUDGET=5000 уже был в комплекте PR (outputs-land69/sl_5000, не пересчитывался)
```

`make all` = `data` → `predict`(unet) → `assets` → `procure` → `export` → `validate`.
Код выхода всех запусков (`echo $?` после каждого `make all`) — `0`.

## Что прогнано: коды выхода

| ветка | commit | BUDGET | код выхода `make all` | `validate` |
|---|---|---|---|---|
| было | `62afcca` | 2500 | 0 | 12/12 |
| было | `62afcca` | 5000 | 0 | 12/12 |
| было | `62afcca` | 20000 | 0 | 12/12 |
| стало | `4377c33` | 2500 | 0 | 12/12 |
| стало | `4377c33` | 5000 | 0 (комплект PR, не пересчитывался) | 12/12 |
| стало | `4377c33` | 20000 | 0 | 12/12 |

Все 6 комплектов: `validate` → `12/12 файлов прошли проверку`, ошибок нет.

## p-растр, маска, порог: идентичны между бюджетами и между ветками

```bash
sha256sum <out>/sl_<budget>/flood_probability.tif <out>/sl_<budget>/flood_mask.tif
```

| файл | sha256 |
|---|---|
| `flood_probability.tif` (все 6 комплектов) | `b767ad16cf4c3c883726c8cb3c5f885f0891618ae997d3c7d74f882386947bcc` |
| `flood_mask.tif` (все 6 комплектов) | `f38ec062c673607488e2293126f43ef7be6785dc88b06de7810abf7e367ff393` |

`threshold` в `run_metadata.json`/`prediction.json` — `0.37` во всех 6 комплектах
(порог из `models/unet/calibration.json`, `models/unet/` побайтово идентичны между
worktree — `sha256sum models/unet/*.pt models/baseline/*.json` совпадает по всем
файлам). Бюджет не меняет p, маску, порог ни на «было», ни на «стало» — контрактное
требование выполнено по обе стороны.

## Таблица «было → стало»: `strategy_comparison.csv` по стратегиям A/B/C

Пиксели: чип 512×512, валидных S1-пикселей 262144 — идентично во всех запусках
(`valid_pixels` в консольном выводе `predict`).

Бюджет ограничивает только стоимость данных (`data_cost_rub`, docs/contracts.md);
`other_cost_rub` (сценарные C_compute + C_review, configs/run.yaml) в бюджет не входит,
поэтому `decision_cost_rub` у C может превышать бюджет при `budget_feasible=true`
(«стало»: 2500 → 3 952,98; 5000 → 6 905,96; 20000 → 19 939,51 руб.). При 20000 у C
`other_cost_rub` 5 000 → 4 000 и `decision_cost_rub` 24 272,96 → 19 939,51 (−4 333,45)
за счёт меньшего числа позиций.

`residual_uncertainty` у B/C — сценарная оптимистичная граница (`status=scenario`):
новые снимки не получены, фактическое снижение неопределённости не заявляется.

### BUDGET = 2 500 руб.

| метрика | стратегия | было | стало | Δ |
|---|---|---|---|---|
| data_cost_rub | A | 0.00 | 0.00 | 0 |
| data_cost_rub | B | 42 465.93 | 42 465.93 | 0 |
| data_cost_rub | C | 2 452.98 | 2 452.98 | 0 |
| budget_feasible | A | true | true | — |
| budget_feasible | B | false | false | — |
| budget_feasible | C | true | true | — |
| covered_expected_loss_rub | A | 0 | 0 | 0 |
| covered_expected_loss_rub | B | 16 842 207.70 | 16 842 207.70 | 0 |
| covered_expected_loss_rub | C | 7 825 102.40 | 7 825 102.40 | 0 |
| coverage_share | C | 0.4646126 | 0.4646126 | 0 |
| covered_asset_ids | C | A02 | A02 | — |
| residual_uncertainty (B/C) | B/C | not_estimated (пусто) | B=0, C=2.634922 (status=scenario) | новая оценка появилась |

### BUDGET = 5 000 руб.

| метрика | стратегия | было | стало | Δ |
|---|---|---|---|---|
| data_cost_rub | A | 0.00 | 0.00 | 0 |
| data_cost_rub | B | 42 465.93 | 42 465.93 | 0 |
| data_cost_rub | C | 4 905.96 | 4 905.96 | 0 |
| budget_feasible | A/B/C | true/false/true | true/false/true | — |
| covered_expected_loss_rub | C | 12 610 565.84 | 12 610 565.84 | 0 |
| coverage_share | C | 0.7487478 | 0.7487478 | 0 |
| covered_asset_ids | C | A02 A10 | A02 A10 | — |
| residual_uncertainty (C) | C | not_estimated (пусто) | 2.605220 (status=scenario) | новая оценка появилась |

### BUDGET = 20 000 руб.

| метрика | стратегия | было | стало | Δ |
|---|---|---|---|---|
| data_cost_rub | A | 0.00 | 0.00 | 0 |
| data_cost_rub | B | 42 465.93 | 42 465.93 | 0 |
| **data_cost_rub** | **C** | **19 272.96** | **15 939.51** | **−3 333.45 (дешевле на 17,3 %)** |
| budget_feasible | A/B/C | true/false/true | true/false/true | — |
| covered_expected_loss_rub | C | 16 842 207.70 | 16 842 207.70 | 0 (то же полное покрытие) |
| coverage_share | C | 1.0 | 1.0 | 0 |
| covered_asset_ids | C | A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 | A01 A02 A03 A04 A05 A06 A07 A08 A09 A10 | совпадает |
| **зоны C** | | Z-A02, Z-A10, Z-A01, Z-A08, Z-A03, Z-A06, Z-A07, Z-A04 (8 буферов объектов) | E-1, E-2, Z-A03, Z-A06, Z-A07, U-3 (кластеры EL + буферы + неопределённость) | другой набор, тот же охват |
| residual_uncertainty (C) | C | not_estimated (пусто) | 0 (status=scenario) | новая оценка появилась |

Источники чисел: `<out>/sl_<budget>/strategy_comparison.csv` в каждом из 6 каталогов
(scratch `out-main/sl_2500|5000|20000` и `outputs-land69/sl_2500|5000|20000`).

## `candidate_orders.geojson` и каталог зон (`procurement_metadata.json`)

| | было (`62afcca`) | стало (`4377c33`) |
|---|---|---|
| `n_candidates` (procurement_metadata.json→zones) | нет поля `n_candidates` (старая версия метаданных) | 73 |
| позиций в `candidate_orders.geojson` | 14 (одинаково при BUDGET=2500/5000/20000) | 73 (одинаково при BUDGET=2500/5000/20000) |
| состав каталога (было) | 10× `Z-<asset_id>` + `W-CHIP` + `N-OPT-LATE` + `K-ARCH` + `K-OPER` = 14 | — |
| состав каталога (стало) | — | `asset_buffer`=30, `el_cluster`=9, `uncertainty_cluster`=6, `grid`=21, `wide`=7 → 73 |

PR #69 расширяет каталог кандидатов с 14 до 73 (добавлены `el_cluster`, сетки 2×2/3×3,
`uncertainty_cluster` и варианты предложений/офферов на геометрию), что и даёт более
дешёвый вариант C при BUDGET=20000 без потери охвата.

## Вывод

- Охват EL не изменился ни при одном бюджете; при 20000 тот же охват дешевле. На BUDGET=2500 и 5000 охват EL и состав покрытых
  объектов у C идентичны «было → стало» (расширенный каталог зон не даёт более дешёвого
  варианта при этих бюджетах — тот же единственный выбор Z-A02 / Z-A02+Z-A10). На
  BUDGET=20000 — регрессии по охвату нет (100 % → 100 %, тот же набор объектов), при
  этом стоимость покупки того же охвата ниже на 3 333,45 руб. (19 272,96 → 15 939,51,
  −17,3 %) за счёт более широкого каталога кандидатов (кластеры EL/неопределённости).
- Соблюдение бюджета C (`budget_feasible=true`) выполняется на всех 3 бюджетах по обеим
  сторонам; B остаётся `false` (сверх бюджета, контрфактическая) на всех 6 запусках —
  без изменений.
- p, маска, порог не зависят от бюджета и не изменились между ветками (общий sha256,
  общий threshold=0.37) — контрактное требование «бюджет меняется — p/маска/порог не
  меняются» выполнено.
- Новое: PR #69 добавляет `residual_uncertainty` (числовую сценарную оценку,
  `uncertainty_status=scenario`) для стратегий B и C вместо пустого `not_estimated` —
  расширение отчётности, не пересчёт EL/охвата.

## Аномалии

1. **Бриф устарел**: к моменту запуска PR #69 уже слит в `origin/main`
   (`b8903a0`), причём ещё раньше в `origin/main` влит PR #89 (issue #25,
   `62afcca`, правка порога карты в `assets`), не относящийся к PR #69. Буквальное
   `origin/main` как «было» дало бы `before == after`. Взято `b8903a0^1 = 62afcca` —
   последний коммит истории без изменений PR #69 в `procurement/`.
2. Комплект PR (`outputs-land69/sl_5000`) собран на `4377c33`, а фактический коммит,
   слитый в `main` (`b8903a0^2`), — `f76741d` (на один коммит дальше, меняет только
   текст поля `note` в `procurement/run.py`, без влияния на числа/зоны — проверено
   `git diff --stat 4377c33 f76741d`). Числа BUDGET=5000 из `sl_5000` (собран на
   `4377c33`) использованы как есть по указанию задания.
3. Старая версия `procurement_metadata.json` («было») не содержит поля
   `zones.n_candidates` — посчитано по факту (`len(features)` в
   `candidate_orders.geojson`).

Сырые каталоги (не в git, временные):
`<scratch>/main-wt` и `<scratch>/out-main/sl_{2500,5000,20000}` удалены после
завершения проверки; `outputs-land69/sl_{2500,5000,20000}` перенесены из worktree в scratchpad сессии
(временно, вне репозитория).
