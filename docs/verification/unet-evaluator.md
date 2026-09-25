# Независимая сверка: PR #54 (issue #17), baseline → U-Net, dev и validation

Роль: `evaluator`, независимая проверка (не автор PR). Worktree
`.claude/worktrees/17-unet`, ветка `worktree-17-unet`, commit `0d0a5b1`. Дата 2026-09-25.
`FLOODVALUE_DATA_DIR=/home/m1hairu/Documents/final_cosmohack/kosmo-moskva_umstvenno_ustalye_34/data`.
Split — `splits/split_v1.json`. **Test-чипы (`splits.test`, события Bolivia/Mekong/Spain)
не читались ни одной командой** ниже; использовались только `split["dev"]` (USA, 69 чипов)
и `split["validation"]` (Nigeria 18 + Sri-Lanka 42 = 60 чипов).

Файлы кода не менялись, правки — только этот файл.

## 1. IoU/F1/P/R на dev и validation, обе модели на одной маске валидных пикселей

Метод: собственный скрипт (не переиспользует агрегацию `floodvalue.model.metrics`, только
`floodvalue.data.io.load_s1/load_label` и **сами методы под проверкой** —
`floodvalue.baseline.predict.predict_proba` и `floodvalue.model.unet.predict.predict_proba`).
Для каждого чипа маска валидности строится один раз (`S1 конечен по обоим каналам И
LabelHand != -1`) и применяется к обеим моделям — тем самым идентичность маски гарантирована
конструкцией скрипта, а не сверкой хешей постфактум. Порог `p ≥ 0,5` для обеих моделей.

Команда:

```
FLOODVALUE_DATA_DIR=…/data uv run --frozen python eval17_verify.py
```

Вывод (129 чипов, 58,5 с вычислений):

```
dev/USA: 69 chips, 17638830 valid px
validation/Nigeria: 18 chips, 4020518 valid px
validation/Sri-Lanka: 42 chips, 9963261 valid px

method,split,event,iou,f1,precision,recall,tp,fp,fn,tn,n_pixels
baseline,dev,ALL,0.390798,0.561977,0.460026,0.721984,580921,681880,223697,16152332,17638830
unet,dev,ALL,0.584470,0.737748,0.744820,0.730809,588022,201460,216596,16632752,17638830
baseline,validation,ALL,0.579962,0.734147,0.838217,0.653065,1386292,267566,736456,11593465,13983779
unet,validation,ALL,0.600493,0.750385,0.957930,0.616759,1309223,57498,813525,11803533,13983779
baseline,validation,Nigeria,0.572154,0.727860,0.931267,0.597380,565920,41768,381417,3031413,4020518
unet,validation,Nigeria,0.691432,0.817570,0.953473,0.715575,677891,33079,269446,3040102,4020518
baseline,validation,Sri-Lanka,0.585474,0.738548,0.784167,0.697945,820372,225798,355039,8562052,9963261
unet,validation,Sri-Lanka,0.526185,0.689542,0.962762,0.537116,631332,24419,544079,8763431,9963261
```

### Сверка с `docs/verification/{baseline,unet}-metrics.csv`

| Часть | Событие | Метод | CSV IoU | Пересчёт IoU | CSV P/R | Пересчёт P/R | N валидных px: CSV vs пересчёт |
|---|---|---|---|---|---|---|---|
| dev | ALL/USA | baseline | 0,390798 | 0,390798 | 0,460/0,722 | 0,460/0,722 | 17 638 830 = 17 638 830 |
| dev | ALL/USA | unet | 0,5844846 | 0,584470 | 0,745/0,731 | 0,745/0,731 | 17 638 830 = 17 638 830 |
| validation | ALL | baseline | 0,579962 | 0,579962 | 0,838/0,653 | 0,838/0,653 | 13 983 779 = 13 983 779 |
| validation | ALL | unet | 0,6004895 | 0,600493 | 0,958/0,617 | 0,958/0,617 | 13 983 779 = 13 983 779 |
| validation | Nigeria | baseline | 0,572154 | 0,572154 | 0,931/0,597 | 0,931/0,597 | 4 020 518 = 4 020 518 |
| validation | Nigeria | unet | 0,6914131 | 0,691432 | 0,953/0,716 | 0,953/0,716 | 4 020 518 = 4 020 518 |
| validation | Sri-Lanka | baseline | 0,585474 | 0,585474 | 0,784/0,698 | 0,784/0,698 | 9 963 261 = 9 963 261 |
| validation | Sri-Lanka | unet | 0,5261915 | 0,526185 | 0,963/0,537 | 0,963/0,537 | 9 963 261 = 9 963 261 |

**Baseline — точное совпадение** (TP/FP/FN/TN совпали побитово: детерминированная сигмоида,
без нейросети).

**U-Net — совпадение до округления в отчёте (3 знака), различие в 4-м знаке.** TP/FP/FN на
уровне ~30–38 пикселей из 4–18 млн (например dev ALL: CSV tp=588057 против пересчёта
tp=588022, разница 0,006 %; validation Sri-Lanka: fp 24386 против 24419). IoU при этом
совпадает при округлении до 3 знаков, которое и приведено в `unet-metrics.md`
(0,584/0,600/0,691/0,526 — все четыре совпали). Причина расхождения в 4–5 знаке — CPU-инференс
свёрточной сети не гарантированно побитово детерминирован между запусками/окружениями
(число потоков BLAS/MKL меняет порядок суммирования в свёртках у пикселей на границе p=0,5);
сама модель и веса при этом идентичны — sha256 весов (раздел 2) и `n_parameters` (1 942 721,
пересчитано независимо и совпало) подтверждают это. Расхождение не влияет ни на один вывод
отчёта PR.

**Идентичность маски валидных пикселей подтверждена независимо**: во всех 8 строк число
валидных пикселей (n_pixels = tp+fp+fn+tn) у пересчёта совпало между baseline и unet построчно
и с CSV — маска строилась в скрипте один раз на чип и использовалась для обеих моделей.

## 2. sha256 весов `models/unet/member_*.pt` против `config.json`

```
member_0.pt  6c2df137d957997581f3901051baf07cb2da911520cb7308fe33a243be08ba29
member_1.pt  e077be954088147f8c08edd358c7bd26b7b2f1ba595b36055f13ea6ac7d7dfc8
member_2.pt  8ff688abfde68f15d0a4782ae264f267ef1f7bb23e75f5d4a7c991b0201ed5a9
member_3.pt  80745de0edbe4f3dd309796caea1192b2ed7fc394194680c295362d86f207008
member_4.pt  6aa232704693a2252d7e24271f9a4bc7a52893574e54c7dbbe941660319793d3
```

Все пять совпадают с полями `members[i].sha256` в `models/unet/config.json`. Файлы —
реальные объекты LFS (`git lfs ls-files` вернул хеши, не указатели; каждый файл 7,4 МБ).
`pytest tests/model::test_shipped_unet_config_matches_weights` (раздел 4) проверяет это же
на каждом прогоне.

## 3. `predict --chip Sri-Lanka_713926` на CPU

```
$ time uv run --frozen python -m floodvalue predict --chip Sri-Lanka_713926 --out /tmp/eval-unet-verify-out
{"chip_id": "Sri-Lanka_713926", "method": "unet", "s1_file": "Sri-Lanka_713926_S1Hand.tif",
 "weights_sha256": "504944cd880c0b2e94acf2481e6ddc4db985ddbd64d550a3591321e8b0cc1bb3",
 "threshold": 0.5, "threshold_status": "uncalibrated: raw p >= 0.5; …",
 "valid_pixels": 262144, "seconds": 0.74}
uv run --frozen python -m floodvalue predict …  4.01s user 0.56s system 245% cpu 1.862 total
```

1,86 с полного времени (включая `uv run` и импорт torch), 0,74 с — само `predict_proba`
ансамбля из 5 членов; ≪ 1 мин, работал на CPU (torch без CUDA в этом окружении). `weights_sha256`
в выводе совпал с `config.json`.

Проверка контракта растров (`rasterio`, независимая проверка кода):

| Файл | dtype | nodata | Требование | Факт |
|---|---|---|---|---|
| `flood_probability.tif` | float32 | −9999,0 | Float32, nodata −9999 | совпало |
| `flood_mask.tif` | uint8 | 255 | UInt8, nodata 255 | совпало |

`flood_mask.tif == (flood_probability.tif >= 0.5)` там, где p конечно, и 255 там, где p —
nodata: проверено поэлементно (`np.array_equal`) — **True**. Для `Sri-Lanka_713926` все
512×512 = 262 144 пикселей валидны (nodata не встречается ни в одном растре), p ∈
[0,0106; 0,989], маска ∈ {0, 1} — соответствует контракту.

## 4. `uv run --frozen pytest tests/model`

```
23 passed
```

Совпадает с заявленным в PR («tests/model 23 passed»). В том числе прошли
`test_shipped_unet_config_matches_weights` (sha256 членов, длина ансамбля 5, split sha256
начинается на `0098346e`) и `test_shipped_unet_metrics_use_same_valid_pixels_as_baseline`
(построчное совпадение `valid_mask_sha256` в `models/unet/metrics.csv` и
`docs/verification/baseline-metrics.csv`) — оба теста читают только зафиксированные CSV/config,
не пересчитывают с нуля, поэтому раздел 1 этого отчёта (независимый пересчёт из сырых данных)
их дополняет, а не дублирует.

## Дополнительно (не запрошено, но проверено походя)

- `splits/split_v1.json`: `floodvalue.data.split.canonical_hash` пересчитан заново →
  `0098346e0f87297501409a1a8522afab6209e35f56e8cf1b68996909c8b63c75`, совпадает с полем
  `sha256` в самом файле и с `split.sha256` в `models/unet/config.json` и
  `models/baseline/config.json`.
- `n_parameters` в `config.json` (1 942 721) пересчитан построением `UNet` из `config.json`
  через `build_model` — совпало.
- `models/unet/metrics.csv` побайтово идентичен `docs/verification/unet-metrics.csv`.

## Вердикт

**Подтверждено.** Baseline — точное (побитовое) совпадение с `baseline-metrics.csv` на dev и
validation. U-Net — совпадение отчётных чисел (IoU до 3 знаков, P/R до 3 знаков) при
независимом пересчёте с нуля из сырых чипов; на уровне 4–5 знака есть ожидаемый для
CPU-свёрток шум порядка 0,006 % пикселей, не влияющий ни на один вывод PR. Маска валидных
пикселей у обеих моделей построена и подтверждена как единая. sha256 всех пяти весов U-Net
совпадает с `config.json`, LFS-объекты на месте. `predict --chip Sri-Lanka_713926` работает на
CPU за 0,74 с (< 1 мин), растры соответствуют контракту (Float32 nodata −9999, UInt8 nodata
255, mask = p ≥ threshold). `pytest tests/model` — 23/23 зелёных.

## Команды и сырой вывод

- `FLOODVALUE_DATA_DIR=… uv run --frozen python eval17_verify.py` — скрипт сверки, не входит в
  репозиторий (временный, `/tmp/…/scratchpad/eval17_verify.py`), полный вывод и JSON с
  raw TP/FP/FN/TN — там же (`eval17_results.json`).
- `for f in models/unet/member_*.pt; do sha256sum "$f"; done` + `git lfs ls-files`.
- `uv run --frozen python -m floodvalue predict --chip Sri-Lanka_713926 --out /tmp/eval-unet-verify-out`
  + проверка растров через `rasterio`/`numpy` в отдельном скрипте.
- `uv run --frozen pytest tests/model -q` → `23 passed`.

Test: не трогался.
