# sensitivity.csv / strategy_comparison.csv — было (origin/main) → стало (PR #109)

Чип: `Sri-Lanka_713926` (validation). Бюджет: 5000 руб. Test-чипы не читались.

## Commit / конфигурация

| | commit (top-level, run_metadata.json) | worktree |
|---|---|---|
| было | `85e493ae3953587fb4b03ebd090c522fed191d03` (= `origin/main`, HEAD этого коммита) | `.claude/worktrees/eval-109-main` (детач от `origin/main`, удалён после отчёта) |
| стало | `14293ad10d7998013eb3ceafaaf03ede44ce79ed` (родитель HEAD `9d064b3`; последний коммит `9d064b3` менял только `docs/`, код и раскладки не менял — вывод из `outputs-land109/run/` совпадает с HEAD) | `.claude/worktrees/land-109` |

`decision_at` = `2017-05-30T23:59:59+00:00`, `decision_deadline` = `2017-05-31T11:59:59+00:00`,
`budget_rub` = 5000.0 — совпадают в обеих версиях.

## Команды

Было (свежий прогон в `.claude/worktrees/eval-109-main`, отсоединённый HEAD на `origin/main`):

```
git worktree add /home/chichetin/PycharmProjects/kosmo-moskva_umstvenno_ustalye_34/.claude/worktrees/eval-109-main origin/main --detach
cd .claude/worktrees/eval-109-main && uv sync --frozen
FLOODVALUE_DATA_DIR=/home/chichetin/PycharmProjects/kosmo-moskva_umstvenno_ustalye_34/data \
FLOODVALUE_OUTPUT_DIR=.../eval-109-main/outputs \
make all CHIP=Sri-Lanka_713926 BUDGET=5000 OUT=.../eval-109-main/outputs/run
```

Код выхода: `0`. Лог: `.claude/worktrees/eval-109-main/outputs-make.log`.
`validate` (последняя строка лога): `12/12 файлов прошли проверку`.

Модели (`models/`) — sha256 всех 6 файлов (`models/pixel/model.joblib`,
`models/unet/member_0..4.pt`) в новом worktree совпали побайтово с
`.claude/worktrees/land-109/models` (сравнено `sha256sum` по всем файлам каталога,
разницы нет — LFS подтянулся из общего git-объектного хранилища).

Стало: уже посчитано на `outputs-land109/run/` (лог `outputs-make.log` в
`.claude/worktrees/land-109`, тот же `make all CHIP=Sri-Lanka_713926 BUDGET=5000`);
код выхода `0`, `validate`: `12/12 файлов прошли проверку`. Пересчёт не потребовался
(команды и параметры идентичны, HEAD после генерации трогал только `docs/`).

## strategy_comparison.csv

```
diff outputs/run/strategy_comparison.csv outputs-land109/run/strategy_comparison.csv
```
Результат: **идентичны построчно** (0 различий), 3 строки данных (A/B/C) + заголовок.

## sensitivity.csv — число строк по группам scenario_id

| группа | было | стало | Δ |
|---|---|---|---|
| `p_*` (portfolio ×0.5/0.8/1.2/1.5, leader ×0.2, challenger ×3.0, `__C_fixed`) | 12 | 12 | 0 |
| `V_*` (аналогично) | 12 | 12 | 0 |
| `q_*` (аналогично) | 12 | 12 | 0 |
| `social_constraint_on` | 1 | 1 | 0 |
| `budget_*` (0/1000/2500/5000/10000/20000/50000, только C) | 7 | — | — |
| `budget_*` (те же 7 бюджетов × B и C) | — | 14 | +7 (добавлена стратегия B на каждый бюджет) |
| `rate_x0.5`, `rate_x1.5` (B и C) | 4 | 4 | 0 |
| `rate_x0.75`, `rate_x1.25`, `rate_x2.0` (B и C) | 0 | 6 | +6 (новые ставки) |
| `deadline_+6h/+12h/+24h/+48h` (B и C) | 0 | 8 | +8 (новая группа) |
| **Всего строк** | 48 | 69 | **+21** |

Итого PR #109 добавил 21 строку: B-стратегию на всех 7 бюджетах (+7), три новые ставки
0.75/1.25/2.0×Б для B и C (+6) и группу `deadline_+6h/+12h/+24h/+48h` для B и C (+8).
Строки `p_*/V_*/q_*/social_constraint_on` (обязательные вариации по контракту §7:
`p_flood`, `asset_value_rub`, `vulnerability_coef`, `budget_rub`, `base_rate_rub_km2`) не
изменились числом.

## sensitivity.csv — совпадение общих строк (было ∩ стало)

Сравнение по ключу `(scenario_id, strategy)`, поля `decision_cost_rub`,
`covered_expected_loss_rub`, `result_status` (скрипт на `csv.DictReader`, вывод ниже):

```
common: 48 only_in_before_not_after: 0 only_in_after: 21
only_b: []
field diffs among common rows: 0
```

Все 48 строк, присутствовавших «было», присутствуют «стало» с теми же значениями
`decision_cost_rub`, `covered_expected_loss_rub`, `result_status` — **0 расхождений**.
`only_in_before_not_after: 0` — ни одна строка не пропала. `changed_inputs_json` валиден
во всех строках обеих версий (48/48 и 69/69, скрипт на `json.loads`).

### Новые строки «стало» (бюджет — стратегия B, все 7 значений)

| scenario_id | strategy | decision_cost_rub | covered_expected_loss_rub | result_status |
|---|---|---|---|---|
| budget_0.00 | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| budget_1000.00 | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| budget_2500.00 | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| budget_5000.00 | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| budget_10000.00 | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| budget_20000.00 | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| budget_50000.00 | B | 43965.93 | 16842207.7015 | within_budget |

(B — контрфактическая для бюджетов < 43965.93 руб., в рамках бюджета только при
50000 руб.; данные B от бюджета не зависят, D-014/contracts §7.)

### Новые строки — ставки 0.75/1.25/2.0

| scenario_id | strategy | decision_cost_rub | covered_expected_loss_rub | result_status |
|---|---|---|---|---|
| rate_x0.75 | C | 5435.01 | (см. сырой вывод) | — |
| rate_x0.75 | B | 33349.45 | — | — |
| rate_x1.25 | C | 4566.24 | — | — |
| rate_x1.25 | B | 54582.42 | — | — |
| rate_x2.0 | C | 6405.96 | — | — |
| rate_x2.0 | B | 86432.13 | — | — |

(полные значения `covered_expected_loss_rub`/`result_status`/`interpretation` — в сыром
файле `outputs-land109/run/sensitivity.csv`, строки с JSON внутри поля затрудняют
построчный `cut` по запятой; извлечены через `csv.DictReader`, см. таблицу «budget» и
общий diff-скрипт выше.)

### Новые строки — deadline (+6h/+12h/+24h/+48h)

| scenario_id | strategy | decision_cost_rub | covered_expected_loss_rub | result_status |
|---|---|---|---|---|
| deadline_+6h | C | 1000.00 | 0 | empty_plan |
| deadline_+6h | B | 1000.00 | 0 | empty_plan |
| deadline_+12h | C | 6905.96 | 12610565.84 | within_budget |
| deadline_+12h | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| deadline_+24h | C | 6905.96 | 12610565.84 | within_budget |
| deadline_+24h | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |
| deadline_+48h | C | 6905.96 | 12610565.84 | within_budget |
| deadline_+48h | B | 43965.93 | 16842207.7015 | counterfactual_over_budget |

При сроке `decision_at + 6h` (`2017-05-31T05:59:59Z`) ни C, ни B не успевают — обе
пустые (`empty_plan`), при `+12h` и далее совпадают со «базовой» строкой (сама база в
`sensitivity.csv` — `deadline_+12h`, идентична исходному `decision_deadline` из
`run_metadata.json`).

## Пиксели / валидная область

Сравнение проведено не по растру, а по табличным артефактам (`sensitivity.csv`,
`strategy_comparison.csv`), которые сами являются агрегатами по одному и тому же
портфелю из 10 объектов на одном чипе (`assets.csv`, seed `20260925`, не менялся между
версиями — `placement_rule` и `seed_source` в `run_metadata.json` идентичны). Число
валидных S1-пикселей чипа — 262144 (`valid_pixels` в логе `make data`, одинаково в
обоих прогонах).

## Сырой вывод

- `.claude/worktrees/eval-109-main/outputs-make.log` — было (полный прогон `make all`).
- `.claude/worktrees/eval-109-main/outputs/run/sensitivity.csv`,
  `.../strategy_comparison.csv`, `.../run_metadata.json` — было.
- `.claude/worktrees/land-109/outputs-land109/run/sensitivity.csv`,
  `.../strategy_comparison.csv`, `.../run_metadata.json` — стало (уже посчитано, не
  пересчитывалось).
- `.claude/worktrees/land-109/outputs-make.log` — стало (лог исходного прогона).

## Итог

Регрессий нет: все 48 общих строк `sensitivity.csv` и все 3 строки
`strategy_comparison.csv` совпадают побитово/по значению между «было» и «стало». PR
#109 добавляет 21 новую строку чувствительности (B на всех бюджетах, ставки
0.75/1.25/2.0, срок +6h/+12h/+24h/+48h) без изменения ранее существовавших. `validate`
— 12/12 в обеих версиях.
