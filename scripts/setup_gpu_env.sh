#!/usr/bin/env bash
# GPU-окружение для обучения основного метода на локальной видеокарте (например RTX 4080).
#
# Полностью вне репозитория и вне uv.lock: uv.lock фиксирует CPU-сборку torch (D-003) —
# это гарантирует, что CPU-инференс одного чипа у жюри ставится и работает без GPU.
# Это окружение uv.lock не трогает и не меняет; создаётся отдельным venv через uv,
# в который отдельно ставится GPU-сборка torch (индекс cu128), поверх — editable
# `floodvalue` (та же команда `python -m floodvalue train`, что и в Makefile/README).
#
# Использование:
#   bash scripts/setup_gpu_env.sh
#   FLOODVALUE_GPU_VENV=/свой/путь bash scripts/setup_gpu_env.sh
#
# Идемпотентно: повторный запуск на уже готовом окружении не пересобирает venv и не
# перекачивает то, что уже установлено и подходит (uv сверяет и печатает "no changes").
#
# Один venv — один чекаут: если на машине несколько worktree (несколько параллельных
# задач), у общего дефолтного пути $HOME/.cache/floodvalue-gpu editable-`floodvalue`
# молча переключится на тот чекаут, откуда скрипт запущен последним (D-024, замечание
# скептика). Скрипт это обнаруживает и громко предупреждает, но не запрещает — если
# нужно держать несколько GPU-venv параллельно, у каждого worktree свой
# FLOODVALUE_GPU_VENV.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${FLOODVALUE_GPU_VENV:-$HOME/.cache/floodvalue-gpu}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
PYTHON_VERSION="3.12"

if ! command -v uv >/dev/null 2>&1; then
  echo "setup_gpu_env: нужен uv (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

echo "setup_gpu_env: venv -> $VENV_DIR"
if [ -x "$VENV_DIR/bin/python" ]; then
  echo "setup_gpu_env: venv уже существует, переиспользую"
else
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

GPU_PYTHON="$VENV_DIR/bin/python"

echo "setup_gpu_env: torch (cu128) — окружение обучения, не влияет на uv.lock/CPU-инференс"
uv pip install --python "$GPU_PYTHON" --index-url "$TORCH_INDEX" torch

CURRENT_TARGET="$("$GPU_PYTHON" -c 'import os, floodvalue; print(os.path.dirname(os.path.dirname(os.path.dirname(floodvalue.__file__))))' 2>/dev/null || true)"
if [ -n "$CURRENT_TARGET" ] && [ "$CURRENT_TARGET" != "$REPO_ROOT" ]; then
  echo "setup_gpu_env: ВНИМАНИЕ — venv сейчас смотрит на floodvalue из $CURRENT_TARGET, переключаю на $REPO_ROOT" >&2
fi

echo "setup_gpu_env: floodvalue -e . (torch уже поставлен выше — pip его не понижает до CPU-сборки)"
# --no-sources: игнорировать [tool.uv.sources] из pyproject.toml (там torch привязан к
# CPU-индексу для uv.lock, D-003) — иначе uv может учесть его при разрешении версий
# и здесь, хотя сейчас cu128-сборка и так не переустанавливается (удовлетворяет torch>=2.4).
uv pip install --python "$GPU_PYTHON" -e "$REPO_ROOT" --no-sources

echo "setup_gpu_env: проверка CUDA"
"$GPU_PYTHON" - <<'PY'
import sys

import torch

print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    print("предупреждение: CUDA недоступна в этом окружении — обучение пойдёт на CPU")

if "+cu" not in torch.__version__:
    print(
        "setup_gpu_env: ОШИБКА — torch в этом venv не cu128-сборка "
        f"({torch.__version__}); GPU-torch подменился CPU-сборкой, обучение на GPU не пойдёт",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo "setup_gpu_env: готово. Обучение: $GPU_PYTHON -m floodvalue train ...  (или make train GPU=1)"
