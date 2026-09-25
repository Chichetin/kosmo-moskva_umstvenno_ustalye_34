# Маска валидности calibration (#73, D-032)

Дата: 2026-09-25. Числа `make calibrate` до/после не сравнивались: validation-чипов Sri-Lanka и
Nigeria в этой копии нет (`data/raw/HandLabeled/S1Hand` пуст), данные фикстурами не заменялись.
Эквивалентность доказывается кодом и тестом.

## Почему поведение не меняется

Прежняя маска: `isfinite(p_raw) & label != -1`. Новая: `data.io.valid_mask(s1, label)`
(S1 конечен в обоих каналах и метка ≠ −1). Оба метода из `METHODS` ставят NaN в p ровно там,
где `io.valid_mask` ложна по S1, а на остальных пикселях p конечна:

- `src/floodvalue/model/unet/predict.py:92`: `p[~valid_mask(s1_db)] = np.nan`;
- `src/floodvalue/baseline/core.py:74`: NaN по той же маске S1.

Поэтому маски совпадают, и `valid_mask_sha256` в `docs/verification/calibration-val.md` не
изменится. При перезапуске `make calibrate` на данных сверить этот хеш.

## Что проверяют тесты

- `tests/calibration/test_calibrate_stage.py::test_load_chips_uses_shared_s1_and_label_valid_mask`:
  NaN в одном канале S1 при метке 1 под ним: старый код включил бы пиксель, новый исключает.
- `::test_load_chips_rejects_nan_p_on_valid_pixels`: нарушение contracts §3 (NaN в p при
  конечном S1) даёт явную ошибку, а не молчаливый NaN в log-loss (хеш этого не ловит, он теперь
  зависит только от S1 и метки).
- `tests/calibration/test_calibration_imports.py`: узость импортов calibration (D-041).
