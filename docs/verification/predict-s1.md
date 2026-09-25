# `predict --s1` на S1-файле вне HandLabeled (#87)

Commit кода: ветка `worktree-87-predict-s1` (PR закрывает #87), sha256 «до» сняты на `main` 628f269; после merge `origin/main` (450af86) `--chip` повторён — те же sha256.
Веса `models/unet` sha256 `504944cd…0cc1bb3`, калибратор `models/unet/calibration.json`
sha256 `3e5bad6c…a255ee60f` (Platt), порог 0,37 — ничего не подбиралось.

## 1. Вход вне HandLabeled

Файл `gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak/India_1029501_S1Weak.tif`
(событие India — train-часть split, D-010/D-016; test-события Spain/Mekong/Bolivia не брались),
1 682 476 байт, md5 бакета `n/RNTytilnOmzcnozP1Hrw==` совпал. 2 канала float32, 512×512,
EPSG:4326, nodata NaN (0 пикселей NaN), значения −41,9…+22,5 dB. Лежит в `data/` вне git.

```bash
FLOODVALUE_DATA_DIR=<пустой каталог> uv run --frozen python -m floodvalue predict \
  --s1 data/raw/WeaklyLabeled/S1Weak/India_1029501_S1Weak.tif
```

Вывод (сокращён): `"chip_id": "India_1029501"`, `"s1_file": "India_1029501_S1Weak.tif"`,
`"threshold": 0.37`, `"threshold_status": "calibrated (platt) and chosen on validation, D-029"`,
`"uncertainty": {"measure": "entropy_p", "n_members": 5, …}`, `"valid_pixels": 262144`,
`"seconds": 1.91`, `"label_status": "no_label"`. Каталог данных пустой — метка и HandLabeled
не читались.

Проверка растров (rasterio): у всех четырёх растров crs/transform/shape = входному файлу;
`flood_probability` float32 nodata −9999, `flood_mask` uint8 nodata 255; p ∈ [0,0090; 0,9921];
доля воды по маске 1,26 %; `mask == (p ≥ 0,37)` на всех валидных пикселях — True.
`make predict S1=<path> OUT=…` даёт те же байты (sha256 `flood_probability.tif`
`5bd49a6d…b66fa2`, `flood_mask.tif` `bf32424b…93c29f`).

Качество на этом файле не оценивается: метки нет (S1Weak — без ручной разметки), `no_label`.

## 2. `--chip` не изменился

`predict --chip Sri-Lanka_713926` до (main 628f269) и после правки — байт в байт:

| файл | sha256 до = после |
|---|---|
| flood_probability.tif | `b767ad16cf4c3c883726c8cb3c5f885f0891618ae997d3c7d74f882386947bcc` |
| flood_mask.tif | `f38ec062c673607488e2293126f43ef7be6785dc88b06de7810abf7e367ff393` |
| uncertainty.tif | `5af38420585d0791e8c0c23a03486e11bea8dc7329f8c2005aa3f51623e1a0e6` |
| flood_probability_range.tif | `a8dc1784cec02eda0f49136d796db43ca367db18770cc34a51e268000e39de93` |

`prediction.json` без поля `seconds` — идентичен (поля `label_status` в режиме `--chip` нет).

## 3. Тесты

`uv run --frozen pytest tests/model/test_predict_s1.py` — синтетический S1 без метки:
растры на сетке входа, NaN → −9999/255, `mask = p ≥ порог`, `label_status: no_label`,
`chip_id` из имени, отказы (нет файла, `--chip` и `--s1` вместе, не 2 канала, ни одного флага).
Мутации `predict.py` (каждая — замена в коде и `uv run --frozen pytest tests/model/test_predict_s1.py tests/model/test_dispatch.py`): `chip_id` = весь stem; нет `label_status`; проверка каналов `!= 2` → `> 2`; `--chip` и `--s1` вместе разрешены; каталог выхода по `args.chip`; `--s1` читается из HandLabeled — 6 из 6 дали красные тесты.

## Граница

`assets`/`procure`/`export` читают S1 только из `HandLabeled/S1Hand`
(`assets --chip India_1029501` → `RasterioIOError … S1Hand/India_1029501_S1Hand.tif`),
поэтому 12 файлов на файле вне HandLabeled не собираются — только карта `predict`.
