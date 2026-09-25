# Выбор итогового основного метода: baseline, кандидат A и кандидат B (D-040, #19)

Протокол D-012/D-016. Метод выбирается **только по dev** (USA, 69 чипов, отложенное событие
train) при сыром p ≥ 0,5 на одних и тех же валидных пикселях. Validation (Sri-Lanka, Nigeria)
здесь только подтверждение. **Test не читался**: ни этот документ, ни скрипты ниже test-чипы
не запрашивают (`chip_ids("dev")`, метки test запечатаны, D-010).

Валидный пиксель: S1 конечен в обоих каналах **и** метка ∈ {0, 1}.
IoU везде пулированный: TP / (TP + FP + FN) по всем валидным пикселям части.

**Итог: основной метод — B, U-Net-ансамбль из 5 seed (D-019).**
- Оба кандидата уверенно лучше baseline на dev. Оптимизм dev есть у всех трёх методов: канал
  baseline (VH) тоже выбран на dev (D-021). Но запас +0,194 и +0,139 он не съедает.
- Между собой на dev B выше A на +0,055 [+0,018; +0,102]. Но эта разница по величине равна
  оптимизму B от выбора эпохи на dev (около 0,06, §3). С поправкой на него на dev **ничья**.
- При ничьей выбор решают доводы §4. Они не опираются на сравнение метрик кандидатов на
  validation или test:
  - B уже откалиброван (D-029);
  - весь сквозной путь уже идёт на B;
  - 5 членов ансамбля позволяют оценить разброс для неопределённости. Это возможность, которой
    у A нет; сама оценка ещё не реализована (план D-034, #21).

  У A калибровку и порог пришлось бы подбирать заново на той же validation.
- Сравнение с baseline на независимых данных — только `eval-test` для замороженной
  конфигурации (#32). Этот документ его не заменяет.

## 1. Таблица выбора: dev (USA), p ≥ 0,5

| Метод | IoU | F1 | Precision | Recall | Пикселей | `valid_mask_sha256` | 95 % CI IoU (бутстрэп по чипам) |
|---|---|---|---|---|---|---|---|
| baseline, VH ≤ −23 dB (D-021) | 0,391 | 0,562 | 0,460 | 0,722 | 17 638 830 | `cf89ac0b…` | [0,295; 0,479] ¹ |
| A — пиксельный HGB (PR #76, D-025 в #76) | 0,530 | 0,693 | 0,731 | 0,658 | 17 638 830 | `cf89ac0b…` | [0,409; 0,635] ² |
| **B — U-Net, ансамбль 5 seed (D-019)** | **0,584** | **0,738** | 0,745 | 0,731 | 17 638 830 | `cf89ac0b…` | [0,479; 0,681] ³ |

Источники чисел по строкам:
- baseline: `docs/verification/baseline-metrics.csv`, строка `baseline,dev,USA`;
- A: `docs/verification/pixel-metrics.csv` из **ещё не влитого PR #76** на коммите
  `80817c907c95` (`git show 80817c9:docs/verification/pixel-metrics.csv`), строка `pixel,dev,USA`;
- B: `docs/verification/unet-metrics.csv`, строка `unet,dev,USA`.

Все три строки дают один и тот же хеш
`cf89ac0b49f7819ea5581c1a8e19aed53fe9f4451a11d1472361f62e5a465bbc`. Значит, методы
сравниваются на идентичных валидных пикселях. Пересчёт в §2 перепроверил это: хеш масок
69 dev-чипов тот же.

CI:
- ¹ `unet-metrics.md` и `pixel-val.md` (@80817c9);
- ² `pixel-val.md` (@80817c9), 1000 повторов, seed 20260925;
- ³ `models/unet/summary.json` → `iou_ci95.dev`.

Интервалы отдельных методов перекрываются, поэтому ниже приведены парные разности.

### Парный бутстрэп по чипам dev (одни выборки чипов для обоих методов, 2000 повторов, seed 0)

| Разность IoU | Δ | 95 % CI | доля выборок с Δ ≤ 0 | Источник |
|---|---|---|---|---|
| B − baseline | +0,194 | [+0,141; +0,250] | 0,000 | `unet-vs-baseline-bootstrap.json` → `dev`; повторён в §2 |
| A − baseline | +0,139 | [+0,055; +0,204] | 0,000 | §2; в `pixel-val.md` @80817c9 (1000 повторов, seed 20260925): +0,139 [+0,061; +0,205] |
| B − A | +0,055 | [+0,018; +0,102] | 0,001 | §2 — посчитан для этого решения, раньше не считался |

Бутстрэп по чипам считает модели фиксированными. Дисперсию, которая появляется из-за выбора
эпохи и гиперпараметров на том же dev, он не учитывает (§3).

## 2. Пересчёт для B − A: команда, скрипты, вывод

Парного бутстрэпа A против B в файлах не было. Его посчитали без переобучения, по сохранённым
весам и только на dev.

- Код и веса — checkout `80817c9` (PR #76) с `git lfs pull` для `models/pixel/*` и
  `models/unet/*`. sha256 пяти `member_*.pt` в этом checkout совпадают с `origin/main`.
- Данные — только 69 чипов USA: `S1Hand` и `LabelHand`, md5 проверяется в
  `floodvalue.data.fetch.fetch_missing`.
- Метрики и бутстрэп — функции репозитория: `floodvalue.model.metrics` и
  `floodvalue.model.unet.report.paired_bootstrap`.

```bash
git worktree add --detach <tmp>/pr76 80817c9 && cd <tmp>/pr76
git lfs pull --include="models/pixel/*,models/unet/*"
FLOODVALUE_DATA_DIR=<tmp>/data uv run --frozen python <tmp>/fetch_dev.py   # 69 dev-чипов, 138 файлов
FLOODVALUE_DATA_DIR=<tmp>/data uv run --frozen python <tmp>/dev_paired_bootstrap.py
```

`fetch_dev.py`:

```python
"""Download only dev (USA) S1Hand + LabelHand into $FLOODVALUE_DATA_DIR/raw. No test files."""
import sys

sys.path.insert(0, "src")
from floodvalue.data import fetch as F  # noqa: E402
from floodvalue.data.io import layer_relpath, raw_dir  # noqa: E402
from floodvalue.data.split import chip_ids  # noqa: E402

ids = chip_ids("dev")
assert all(c.startswith("USA_") for c in ids), ids
rels = [layer_relpath(c, l) for c in ids for l in ("S1Hand", "LabelHand")]
prefixes = [f"{F.VERSION_PREFIX}data/flood_events/HandLabeled/{l}/USA_" for l in ("S1Hand", "LabelHand")]
print(len(ids), "dev chips; downloaded", F.fetch_missing(raw_dir(), rels, prefixes))
```

`dev_paired_bootstrap.py`:

```python
"""Dev (USA) only: A (pixel, PR #76) vs B (U-Net ensemble, raw p) vs baseline (VH <= -23 dB).
p >= 0.5; one valid mask per chip (S1 finite in both channels AND label in {0, 1}).
Reads split part "dev" only; test is never requested. Run from a checkout of 80817c9."""
import json
import sys
import time
from functools import reduce

import numpy as np

sys.path.insert(0, "src")
from floodvalue.baseline.predict import predict_proba as baseline_p  # noqa: E402
from floodvalue.model import metrics as M  # noqa: E402
from floodvalue.model.pixel.predict import predict_proba as pixel_p  # noqa: E402
from floodvalue.model.unet import data as D  # noqa: E402
from floodvalue.model.unet.features import loss_mask  # noqa: E402
from floodvalue.model.unet.predict import predict_proba as unet_p  # noqa: E402
from floodvalue.model.unet.report import paired_bootstrap  # noqa: E402

methods = {"pixel": pixel_p, "unet": unet_p, "baseline": baseline_p}
per, secs, masks = {k: [] for k in methods}, dict.fromkeys(methods, 0.0), []
chips = sorted(D.split_chip_ids("dev"))
for chip in chips:
    s1, label = D.read_s1(chip), D.read_label(chip)
    m = loss_mask(s1, label)
    masks.append(m)
    for k, f in methods.items():
        t = time.perf_counter()
        p = f(s1)
        secs[k] += time.perf_counter() - t
        per[k].append(M.confusion(np.nan_to_num(p) >= 0.5, label == 1, m))
print(json.dumps({
    "chips": len(chips), "valid_mask_sha256": M.mask_sha256(masks),
    "pooled": {k: M.scores(reduce(M.add, v)) for k, v in per.items()},
    "cpu_seconds_per_chip": {k: v / len(chips) for k, v in secs.items()},
    "unet_minus_pixel": paired_bootstrap(per["unet"], per["pixel"]),
    "unet_minus_baseline": paired_bootstrap(per["unet"], per["baseline"]),
    "pixel_minus_baseline": paired_bootstrap(per["pixel"], per["baseline"]),
}, indent=1))
```

Вывод (сокращён):

```
chips 69, valid_mask_sha256 cf89ac0b…
pooled IoU: pixel 0.52996, unet 0.58447, baseline 0.39080
cpu_seconds_per_chip (только predict_proba): pixel 0.442, unet 0.572, baseline 0.002
unet_minus_pixel      diff 0.0545  ci95 [0.0180, 0.1016]  p_diff_le_0 0.001
unet_minus_baseline   diff 0.1937  ci95 [0.1413, 0.2501]  p_diff_le_0 0.0
pixel_minus_baseline  diff 0.1392  ci95 [0.0554, 0.2040]  p_diff_le_0 0.0
```

Сверка с файлами:
- IoU A = 0,529964 — совпадает с `pixel-metrics.csv`;
- IoU baseline = 0,390798 — совпадает с `baseline-metrics.csv`;
- IoU B на CPU (fp32) = 0,58447 против 0,58448 в `unet-metrics.csv` (GPU, bf16); расхождение
  описано в `unet-metrics.md`;
- B − baseline совпадает с `unet-vs-baseline-bootstrap.json` до 4-го знака.

Замер времени — AMD Ryzen 7 H 255, 16 потоков.

## 3. Оптимизм dev: количественно

Dev был у обоих кандидатов и частью выбора, и частью отчёта.

**B.** На dev для каждого члена выбрана лучшая эпоха. Сколько это даёт, видно по
`models/unet/history_member_{0..4}.csv` (колонка `dev_iou`):

| Член | Выбранная эпоха (max dev IoU) | Среднее по эпохам 15–60 | Медиана по эпохам 15–60 | Эпоха 60 |
|---|---|---|---|---|
| 0 | 0,571 | 0,501 | 0,513 | 0,503 |
| 1 | 0,556 | 0,505 | 0,514 | 0,494 |
| 2 | 0,583 | 0,495 | 0,513 | 0,517 |
| 3 | 0,577 | 0,512 | 0,520 | 0,526 |
| 4 | 0,555 | 0,488 | 0,497 | 0,499 |

- Выбор эпохи прибавляет одиночному члену около **0,06**: 0,568 в среднем против 0,49–0,52.
- Ансамбль к одиночной модели прибавляет +0,016 (`unet-metrics.md`).
- Ансамбль без выбора эпохи не сохранялся. Грубая оценка: 0,51 + 0,016 ≈ 0,52–0,53.

**A.** На dev выбран лучший из 18 наборов гиперпараметров. Соседние наборы дают 0,528,
0,510, 0,507 (`pixel-val.md` @80817c9, «Выбор гиперпараметров»), поэтому смещение порядка
0,00–0,02.

**Вывод.** B − A = +0,055 по величине равна поправке на выбор эпохи у B. Нижняя граница CI
(+0,018) от этого смещения не защищает: бутстрэп не учитывает дисперсию выбора. **С учётом
оптимизма B и A на dev не различимы.** «B лучше A по IoU» не утверждается.

Ограничения dev:
- dev — одно событие, 4,6 % воды (D-016);
- LOEO по рецепту U-Net (D-036, `transfer-loeo.md`) ещё не прогнан;
- независимая оценка — только test (`eval-test`, #32).

Оба кандидата лучше baseline с запасом, который оптимизм не съедает: +0,194 и +0,139.

## 4. При ничьей на dev выбор решают эти критерии

| Критерий | baseline | A — пиксельный HGB | B — U-Net-ансамбль | Источник |
|---|---|---|---|---|
| **Калибровка и порог (критерии 5, 9)** | не требуется | не сделана; калибровку и порог пришлось бы подбирать заново на той же validation | сделана: `models/unet/calibration.json`, D-029 | D-029, `calibration-val.md` |
| **Неопределённость (критерий 10, #21)** | — | только функция от p (1−\|2p−1\|, энтропия) сейчас функция от p; 5 членов позволяют добавить разброс ансамбля — **не реализовано**, план D-034, #21 (`unet/predict.py` отдаёт только среднее) | D-034, #21 |
| **Сквозной путь и 12 файлов** | — | не подключён (`FLOODVALUE_METHOD=pixel` в #76) | метод по умолчанию в `predict`, `eval-test`, `make all` | D-019, D-030 |
| **Устойчивость к seed на dev** | — | один seed | ансамбль 5 seed; одиночные модели dev 0,568 ± 0,013 (на выбранных по dev эпохах, оптимистично) | `unet-metrics.md`, эксперимент |
| Только S1 (критерий 3) | да (VH) | да: VV, VH, VV−VH, окна 3×3/7×7 | да: VV, VH, VV−VH, канал валидности | `pixel-val.md` @80817c9; `unet-metrics.md` «Метод» |
| Самостоятельный метод, не порог baseline | — | да | да | те же |
| CPU `predict_proba`, с/чип (dev, §2) | 0,002 | 0,44 | 0,57 | §2 |
| CPU, полный predict одного чипа по отчёту метода | — | 0,3 с | 0,46–1,0 с (ансамбль), около 2 с с I/O | `pixel-val.md` @80817c9; `unet-metrics.md` |
| Размер весов | конфиг | 487 653 байт (`models/pixel/model.joblib`, LFS) | 5 × 7 810 843 байт = 39,1 МБ = 37,2 МиБ (7,45 МиБ на член; «7,4 МБ» в `unet-metrics.md` — МиБ) (`models/unet/member_*.pt`, LFS) | `stat` по файлам |
| Контролируемый эксперимент (критерий 8) | — | сетка 18 точек и абляция входов на dev (`pixel-ablation.md` @80817c9) | 5 вариантов × 3 seed на dev (`unet-variants.csv`) | файлы |

Первые четыре строки (выделены) и есть **основание выбора B**. Ни одна из них не опирается на
сравнение метрик кандидатов на validation или test.
- Калибровка B подобрана на validation по протоколу D-012. Это штатный этап основного метода,
  а не довод «B лучше A на validation». Её числа (0,685) в сравнение не входят.
- У A калибровку и порог пришлось бы подбирать заново на той же validation.
- Разброс ансамбля для неопределённости — возможность, а не готовый артефакт (#21).

Время inference и размер весов на выбор не влияют: оба кандидата работают на CPU без GPU и
быстрее 1 минуты на чип.

## 5. Validation — только справочно (не довод выбора)

Сырая p ≥ 0,5, одни пиксели (`eab3f6c6…` для ALL у всех трёх методов):

| Часть / событие | baseline | A | B | Источник |
|---|---|---|---|---|
| validation ALL | 0,580 | 0,445 | 0,600 | `baseline-metrics.csv`; `pixel-metrics.csv` @80817c9; `unet-metrics.csv` |
| Nigeria | 0,572 | 0,598 | 0,691 | те же |
| Sri-Lanka | 0,585 | 0,321 | 0,526 | те же |

Как это читать:
- С dev совпадает только направление B ≥ A.
- **Преимущество над baseline validation не подтверждает**:
  - B − baseline = +0,020 [−0,070; +0,078], не значимо (`unet-vs-baseline-bootstrap.json`);
  - A − baseline = −0,135 [−0,260; −0,021] (`pixel-val.md` @80817c9).
- Оба кандидата занижают p на событии с большей долей воды. Сильнее всего это видно на
  Sri-Lanka.
- **Эти числа — не довод выбора и не оценка качества.** Сюда же относятся числа, для которых
  калибровка или порог подбирались на той же validation:
  - B после калибровки Platt и порога 0,37 даёт IoU 0,685 (D-029, `calibration-val.md`,
    `eval-validation.md`);
  - у A оракульный порог 0,20 даёт 0,636 (`pixel-val.md` @80817c9).

  Это оценки на данных, на которых подбирали калибровку и порог, поэтому они завышены
  относительно будущих данных.

## 6. Ограничения сравнения

1. Числа A взяты из **непротянутого PR #76** (@80817c9). Если в #76 после этого коммита
   изменятся модель или отчёт, таблицу нужно пересчитать командой из §2.
2. Dev — одно событие (USA, 4,6 % воды). Бутстрэп по чипам внутри одного события не отражает
   разброс между событиями и не учитывает дисперсию выбора эпохи и гиперпараметров.
3. Оптимизм dev у B (около 0,06) больше, чем у A (0,00–0,02). После поправки на dev ничья (§3).
4. Validation-метрики B после калибровки получены на тех же данных, на которых калибровка
   подбиралась (D-029). Для выбора они не использовались.
5. LOEO рецепта U-Net (D-036) не прогнан. Независимая оценка — только `eval-test` (#32).
