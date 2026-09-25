# Проверка маски калибровки (#73)

Дата: 2026-09-25. Сравнивались `origin/main` (`d9bcf5b174764ed36eecba1e6760f8faac6f4de2`) и PR-коммит (`825f42aed3240d84bec53b5364b3c6d01104e334`). Запускался только `make calibrate`; калибровка использует validation для подбора, а dev — для отчётной проверки. Команды `eval-test` не запускались, test-чипы не открывались.

## Команды и результат

На PR-коммите, из worktree `.claude/worktrees/73-calibration-valid-mask`:

Команда завершилась с кодом 2. Ключевые строки вывода:

```text
$ make calibrate
uv run --frozen python -m floodvalue calibrate
rasterio._err.CPLE_OpenFailedError: /home/ir6/my/kosmo-moskva_umstvenno_ustalye_34/.claude/worktrees/73-calibration-valid-mask/data/raw/HandLabeled/S1Hand/Nigeria_1095404_S1Hand.tif: No such file or directory
rasterio.errors.RasterioIOError: /home/ir6/my/kosmo-moskva_umstvenno_ustalye_34/.claude/worktrees/73-calibration-valid-mask/data/raw/HandLabeled/S1Hand/Nigeria_1095404_S1Hand.tif: No such file or directory
make: *** [Makefile:36: calibrate] Error 1
```

На `origin/main`, worktree `/tmp/issue73-origin`, использован уже созданный `.venv`; `PYTHONPATH` указывает на исходники именно `origin/main`:

Команда завершилась с кодом 2. Ключевые строки вывода:

```text
$ PYTHONPATH=/tmp/issue73-origin/src UV_PROJECT_ENVIRONMENT=/home/ir6/my/kosmo-moskva_umstvenno_ustalye_34/.claude/worktrees/73-calibration-valid-mask/.venv UV_NO_SYNC=1 make calibrate
uv run --frozen python -m floodvalue calibrate
rasterio._err.CPLE_OpenFailedError: /tmp/issue73-origin/data/raw/HandLabeled/S1Hand/Nigeria_1095404_S1Hand.tif: No such file or directory
rasterio.errors.RasterioIOError: /tmp/issue73-origin/data/raw/HandLabeled/S1Hand/Nigeria_1095404_S1Hand.tif: No such file or directory
make: *** [Makefile:36: calibrate] Error 1
```

Перед повтором обычный `make calibrate` на `origin/main` завершился при создании окружения. Команда вернула код 2; существенные строки вывода:

```text
error: Failed to install: torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl (torch==2.14.0+cpu)
Caused by: failed to copy file from /home/ir6/.cache/uv/archive-v0/EfN-Sd7tEd-wRRmlj6PFl/torch/lib/libtorch_python.so to /tmp/issue73-origin/.venv/lib/python3.12/site-packages/torch/lib/.tmpUlwCUk/libtorch_python.so: Disk quota exceeded (os error 122)
make: *** [Makefile:36: calibrate] Error 2
```

Повторный запуск выше использовал доступное окружение и дошёл до отсутствующего validation-файла.

## Сравнение validation: было → стало

| Артефакт или метрика | Было: `origin/main` | Стало: `825f42a` |
|---|---|---|
| Калибровка и порог | Не измерено: отсутствует `Nigeria_1095404_S1Hand.tif` | Не измерено: отсутствует `Nigeria_1095404_S1Hand.tif` |
| Validation IoU, F1 и прочие метрики | Не рассчитаны | Не рассчитаны |
| `valid_mask_sha256` | Не рассчитан | Не рассчитан |
| Новые calibration validation-артефакты | Не созданы | Не созданы |

Сопоставление чисел и хеша valid pixels невозможно: оба запуска остановились до чтения validation-чипа. Данные не заменялись фикстурами; по этой проверке вывод об изменении метрик или маски не делается.
