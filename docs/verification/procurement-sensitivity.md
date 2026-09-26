# Чувствительность плана съёмки к бюджету, ставке Б и сроку решения (#24, критерии 15–17)

Чип **Sri-Lanka_713926** (validation; test-чипы не читались), основной метод `unet`, порог 0,37,
базовый бюджет 5 000 руб., ставка Б = 788,64 руб./км² (`base_rate_status=scenario`), срок
решения по D-043: `decision_at` = дата S1 чипа (2017-05-30) + сценарное время из
`configs/run.yaml` = 2017-05-30T23:59:59+00:00, окно 12 ч (база — срок 2017-05-31T11:59:59+00:00). Код — commit `14293ad` (дальнейшие коммиты PR меняют только документы и округление `with_base_rate` до ROUND_HALF_UP — при множителях ×0,5…×2,0 ставка точная, числа те же) (ветка PR #109 после merge `origin/main` и правок скептика D-047).
Решение — D-047.

> **С D-049 (#114, PR #141) правило C другое.** Таблица и выводы ниже получены прежним правилом
> C («EL на рубль», D-038). После #114 оно осталось только в строке `C_by_EL` файла
> `sensitivity.csv` (Z-A02, Z-A10: EL 12 610 565,84, Σ R 189 434,16 руб.). Строки B не
> изменились. Строки C при текущем правиле (max прироста R = V·q·min(p, 1−p) на рубль; R —
> сценарная цена ошибки решения, а не снижение потерь) получены командой
> `make all CHIP=Sri-Lanka_713926 BUDGET=5000` на `dbee3cd` → validate 12/12 и взяты из
> `procurement_sensitivity.csv`:
>
> | scenario_id | C | decision_cost, руб. | охват EL, руб. | Σ R, руб. | статус |
> |---|---|---:|---:|---:|---|
> | budget_0 / 1 000 | пусто | 1 000,00 | 0 | 0 | `empty_plan` |
> | budget_2 500 | E-2 | 3 254,56 | 1 013 716,73 | 1 013 716,73 | `within_budget` |
> | budget_5 000 (база) | E-2, Z-A03 | 6 207,54 | 1 462 348,49 | 1 462 348,49 | `within_budget` |
> | budget_10 000 | E-2, Z-A03, Z-A02, Z-A06 | 12 090,35 | 9 455 260,64 | 1 805 055,84 | `within_budget` |
> | budget_20 000 / 50 000 | E-1, E-2, Z-A03, Z-A06, Z-A07, U-3 | 19 939,51 | 16 842 207,70 | 2 070 290,02 | `within_budget` |
> | rate_x0.5 | E-2, Z-A03, Z-A02, Z-A06 | 7 545,20 | 9 455 260,64 | 1 805 055,84 | `within_budget` |
> | rate_x0.75 | E-2, Z-A03, Z-A02 | 7 495,41 | 9 287 450,89 | 1 637 246,09 | `within_budget` |
> | rate_x1.25 | E-2, U-3 | 6 450,30 | 1 042 195,32 | 1 042 195,32 | `within_budget` |
> | rate_x1.5 / x2.0 | E-2 | 4 131,85 / 5 009,13 | 1 013 716,73 | 1 013 716,73 | `within_budget` |
> | deadline_+6h | пусто | 1 000,00 | 0 | 0 | `empty_plan` |
> | deadline_+12h / +24h / +48h | E-2, Z-A03 | 6 207,54 | 1 462 348,49 | 1 462 348,49 | `within_budget` |
> | social_constraint_on | G3-11 | 6 383,82 | 293 703,75 | 293 703,75 | `within_budget` |
>
> По бюджету Σ R не убывает. Доля EL у C при 5 000 руб. мала (0,087): C покупает проверку
> там, где исход неясен. Объекты с почти известным исходом (A02, A10, p > 0,97) проверяет
> осмотр (D-035). Было → стало на 2 500/5 000/20 000 — `procurement-catalog.md` §5.

## Воспроизведение

```bash
export FLOODVALUE_DATA_DIR=<каталог Sen1Floods11>
make all CHIP=Sri-Lanka_713926 BUDGET=5000 OUT=outputs-land109/run
#   procure: A: данные 0.00 руб., бюджет 5000.00, в бюджете True, охват EL 0.00 руб., зоны []
#   procure: B: данные 42465.93 руб., бюджет 5000.00, в бюджете False, охват EL 16842207.70 руб., зоны ['W-CHIP']
#   procure: C: данные 4905.96 руб., бюджет 5000.00, в бюджете True, охват EL 12610565.84 руб., зоны ['Z-A02', 'Z-A10']
#   export: комплект Sri-Lanka_713926_5000_14293ad
#   12/12 файлов прошли проверку
```

Таблица ниже — строки `sensitivity.csv` этого комплекта со стратегиями B и C (строки p/V/q
портфеля — из `assets`, P18a/D-028, здесь не повторяются). «Корзина» и «данные» взяты из поля
`interpretation` той же строки; `decision_cost` = данные + прочие сценарные издержки.

## Сценарии

| scenario_id | стратегия | изменённый вход | корзина | данные, руб. | decision_cost, руб. | охват EL, руб. | result_status |
|---|---|---|---|---:|---:|---:|---|
| `budget_0.00` | C | `{"budget_rub": 0.0}` | пусто | 0.00 | 1000.00 | 0.00 | `empty_plan` |
| `budget_0.00` | B | `{"budget_rub": 0.0}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `budget_1000.00` | C | `{"budget_rub": 1000.0}` | пусто | 0.00 | 1000.00 | 0.00 | `empty_plan` |
| `budget_1000.00` | B | `{"budget_rub": 1000.0}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `budget_2500.00` | C | `{"budget_rub": 2500.0}` | Z-A02 | 2452.98 | 3952.98 | 7825102.40 | `within_budget` |
| `budget_2500.00` | B | `{"budget_rub": 2500.0}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `budget_5000.00` | C | `{"budget_rub": 5000.0}` | Z-A02, Z-A10 | 4905.96 | 6905.96 | 12610565.84 | `within_budget` |
| `budget_5000.00` | B | `{"budget_rub": 5000.0}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `budget_10000.00` | C | `{"budget_rub": 10000.0}` | E-1, E-2, Z-A03 | 9423.24 | 11923.24 | 16498307.33 | `within_budget` |
| `budget_10000.00` | B | `{"budget_rub": 10000.0}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `budget_20000.00` | C | `{"budget_rub": 20000.0}` | E-1, E-2, Z-A03, Z-A06, Z-A07, U-3 | 15939.51 | 19939.51 | 16842207.70 | `within_budget` |
| `budget_20000.00` | B | `{"budget_rub": 20000.0}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `budget_50000.00` | C | `{"budget_rub": 50000.0}` | E-1, E-2, Z-A03, Z-A06, Z-A07, U-3 | 15939.51 | 19939.51 | 16842207.70 | `within_budget` |
| `budget_50000.00` | B | `{"budget_rub": 50000.0}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `within_budget` |
| `rate_x0.5` | C | `{"base_rate_factor": 0.5, "base_rate_rub_km2": 394.32, "base_rate_status": "scenario"}` | E-1, E-2, Z-A03 | 4711.62 | 7211.62 | 16498307.33 | `within_budget` |
| `rate_x0.5` | B | `{"base_rate_factor": 0.5, "base_rate_rub_km2": 394.32, "base_rate_status": "scenario"}` | W-CHIP | 21232.97 | 22732.97 | 16842207.70 | `counterfactual_over_budget` |
| `rate_x0.75` | C | `{"base_rate_factor": 0.75, "base_rate_rub_km2": 591.48, "base_rate_status": "scenario"}` | E-1 | 3935.01 | 5435.01 | 15035958.84 | `within_budget` |
| `rate_x0.75` | B | `{"base_rate_factor": 0.75, "base_rate_rub_km2": 591.48, "base_rate_status": "scenario"}` | W-CHIP | 31849.45 | 33349.45 | 16842207.70 | `counterfactual_over_budget` |
| `rate_x1.25` | C | `{"base_rate_factor": 1.25, "base_rate_rub_km2": 985.8, "base_rate_status": "scenario"}` | Z-A02 | 3066.24 | 4566.24 | 7825102.40 | `within_budget` |
| `rate_x1.25` | B | `{"base_rate_factor": 1.25, "base_rate_rub_km2": 985.8, "base_rate_status": "scenario"}` | W-CHIP | 53082.42 | 54582.42 | 16842207.70 | `counterfactual_over_budget` |
| `rate_x1.5` | C | `{"base_rate_factor": 1.5, "base_rate_rub_km2": 1182.96, "base_rate_status": "scenario"}` | Z-A02 | 3679.47 | 5179.47 | 7825102.40 | `within_budget` |
| `rate_x1.5` | B | `{"base_rate_factor": 1.5, "base_rate_rub_km2": 1182.96, "base_rate_status": "scenario"}` | W-CHIP | 63698.90 | 65198.90 | 16842207.70 | `counterfactual_over_budget` |
| `rate_x2.0` | C | `{"base_rate_factor": 2.0, "base_rate_rub_km2": 1577.28, "base_rate_status": "scenario"}` | Z-A02 | 4905.96 | 6405.96 | 7825102.40 | `within_budget` |
| `rate_x2.0` | B | `{"base_rate_factor": 2.0, "base_rate_rub_km2": 1577.28, "base_rate_status": "scenario"}` | W-CHIP | 84932.13 | 86432.13 | 16842207.70 | `counterfactual_over_budget` |
| `deadline_+6h` | C | `{"deadline_hours_after_decision": 6, "decision_deadline": "2017-05-31T05:59:59+00:00"}` | пусто | 0.00 | 1000.00 | 0.00 | `empty_plan` |
| `deadline_+6h` | B | `{"deadline_hours_after_decision": 6, "decision_deadline": "2017-05-31T05:59:59+00:00"}` | пусто | 0.00 | 1000.00 | 0.00 | `empty_plan` |
| `deadline_+12h` | C | `{"deadline_hours_after_decision": 12, "decision_deadline": "2017-05-31T11:59:59+00:00"}` | Z-A02, Z-A10 | 4905.96 | 6905.96 | 12610565.84 | `within_budget` |
| `deadline_+12h` | B | `{"deadline_hours_after_decision": 12, "decision_deadline": "2017-05-31T11:59:59+00:00"}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `deadline_+24h` | C | `{"deadline_hours_after_decision": 24, "decision_deadline": "2017-05-31T23:59:59+00:00"}` | Z-A02, Z-A10 | 4905.96 | 6905.96 | 12610565.84 | `within_budget` |
| `deadline_+24h` | B | `{"deadline_hours_after_decision": 24, "decision_deadline": "2017-05-31T23:59:59+00:00"}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `deadline_+48h` | C | `{"deadline_hours_after_decision": 48, "decision_deadline": "2017-06-01T23:59:59+00:00"}` | Z-A02, Z-A10 | 4905.96 | 6905.96 | 12610565.84 | `within_budget` |
| `deadline_+48h` | B | `{"deadline_hours_after_decision": 48, "decision_deadline": "2017-06-01T23:59:59+00:00"}` | W-CHIP | 42465.93 | 43965.93 | 16842207.70 | `counterfactual_over_budget` |
| `social_constraint_on` | C | `{"social_asset_ids": ["A06", "A07"], "social_constraint": true}` | G3-11 | 4883.82 | 6383.82 | 293703.75 | `within_budget` |

## Что из этого следует (только по числам таблицы)

- **Бюджет.** C пуста при 0 и 1 000 руб.; при 2 500 — одна зона Z-A02 (охват EL 7 825 102,40 руб.);
  при 5 000 — Z-A02 + Z-A10 (12 610 565,84); при 10 000 — E-1, E-2, Z-A03 (16 498 307,33); с
  20 000 — весь EL портфеля 16 842 207,70 за 15 939,51 руб. данных, дальше рост бюджета ничего не
  добавляет. Охват C по бюджету не убывает.
- **B** — одна позиция W-CHIP за 42 465,93 руб. данных при любом бюджете; до 20 000 включительно
  `counterfactual_over_budget` (сверх бюджета не закупается, показана для сравнения), при
  50 000 — `within_budget`. B покрывает тот же EL, что C при 20 000, но в 2,7 раза дороже по данным.
- **Ставка Б (сценарная).** При бюджете 5 000 удешевление до ×0,5 меняет корзину C на E-1, E-2,
  Z-A03 (16 498 307,33 — как при бюджете 10 000), ×0,75 — на E-1 (15 035 958,84); подорожание
  ×1,25…×2,0 оставляет одну Z-A02 (7 825 102,40). Охват C по ставке монотонно не растёт с её
  ростом. B при всех ставках остаётся сверх бюджета (от 21 232,97 до 84 932,13 руб. данных).
- **Срок решения** (сроки доступности предложений — сценарные, D-038; вывод о сроке — следствие
  этого допущения). При `decision_at + 6 ч` ни одна позиция каталога ещё не доступна (0 пригодных) —
  C и B пусты, охват 0; в `interpretation` B: «правило B (W-CHIP) к сроку не исполнимо — W-CHIP
  доступна 2017-05-31T08:59:59+00:00». Окно +12 ч (база), +24 ч и +48 ч дают 51 пригодную позицию и
  тот же план; позже добавляется только оптика с неподтверждённой облачностью (14 позиций к +12 и
  +24 ч, 15 к +48 ч), она охвата не даёт.
- **Социальное ограничение** (сначала покрыть A06, A07) при 5 000 руб. выполняется одной ячейкой
  G3-11, охват EL падает до 293 703,75 руб. — цена социального приоритета в покрытом EL.
- Остаточная неопределённость непустых планов B/C — `scenario` (сумма по непокрытым объектам);
  у пустого плана — исходная, как у A (`baseline`, 2,787). Новых снимков нет, фактическое снижение
  неопределённости не заявляется.

## Смена бюджета не меняет карту

```bash
cp -r outputs-land109/run outputs-land109/b20k
uv run --frozen python -m floodvalue procure --chip Sri-Lanka_713926 --budget 20000 --out outputs-land109/b20k
#   procure: C: данные 15939.51 руб., бюджет 20000.00, в бюджете True, охват EL 16842207.70 руб., зоны ['E-1', 'E-2', 'Z-A03', 'Z-A06', 'Z-A07', 'U-3']
sha256sum outputs-land109/{run,b20k}/flood_{probability,mask}.tif   # b767ad16… и f38ec062… в обоих
# threshold в run_metadata.json: 0.37 и 0.37
```
