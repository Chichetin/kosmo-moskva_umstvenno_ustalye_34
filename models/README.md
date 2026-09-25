# `models/` — веса и калибраторы (Git LFS)

Веса, нормализация по train и калибраторы — здесь, через **Git LFS** (`.gitattributes`
в корне репозитория), каждый файл < 100 МБ (лимит LFS GitVerse по HTTP, `GIT.md`,
`docs/decisions.md` D-011).

## Структура

```
models/
  baseline/
    config.json        # канал, порог (VH ≤ −23,0 dB, D-021), метрики dev/val — без бинарных весов
  pixel/
    config.json         # признаки, гиперпараметры, метрики dev/val
    weights.joblib        # или .pkl — через LFS
  unet/
    config.json
    weights.pt             # через LFS
  calibration/
    config.json              # метод калибровки (Platt/isotonic/…), параметры, порог
    calibrator.joblib          # через LFS, если бинарный
```

У `baseline` бинарного файла весов нет — параметры (канал, порог, масштаб сигмоиды)
целиком в `config.json`, а `weights_sha256` в нём — sha256 канонического JSON этих
параметров, не файла (см. `models/baseline/config.json`). У методов с обученными
весами (`pixel`, `unet`, калибратор) — обычный файл через LFS, как показано выше.

`models/<method>/` — по одному каталогу на метод, который производит `predict_proba`
или калибратор (`area:baseline`, `area:model-pixel`, `area:model-unet`,
`area:calibration` — см. `docs/project-structure.md`). Имя `<method>` и точный состав
`config.json`/файлов весов определяет модуль, который их пишет; здесь — только общий
контракт каталога, LFS и правило sha256.

## sha256 и `run_metadata.json`

Каждый файл весов, использованный в запуске, называется вместе с sha256 в
`run_metadata.json` (12-й обязательный файл, `docs/contracts.md`) — так проверяется,
что commit, конфигурация и веса согласованы:

```bash
sha256sum models/<method>/weights.*
```

## Получение весов на новой машине

```bash
git lfs install       # один раз на машине
git lfs pull          # после clone/pull — скачивает содержимое LFS-объектов в models/
```

Без `git lfs pull` файлы в `models/` — LFS-указатели (текстовые заглушки с hash и
размером), не рабочие веса; inference (`make predict`, `make calibrate`) на них упадёт
с ошибкой чтения формата.

## LFS

`.gitattributes` в корне репозитория:
`models/**/*.{pt,pth,onnx,joblib,pkl,safetensors,ckpt,npz}` →
`filter=lfs diff=lfs merge=lfs -text`. `config.json` — обычный текстовый файл git, не
LFS (версионируется как код, виден в diff PR).

Страж коммита (`check-data-files.sh`) не пропускает файл ≥ 5 МБ вне LFS-атрибута —
если появляется новое расширение весов, его сначала добавляют в `.gitattributes`.
