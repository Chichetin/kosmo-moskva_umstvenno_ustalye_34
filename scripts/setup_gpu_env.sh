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

echo "setup_gpu_env: floodvalue -e . (torch уже поставлен выше — pip его не понижает до CPU-сборки)"
uv pip install --python "$GPU_PYTHON" -e "$REPO_ROOT"

echo "setup_gpu_env: проверка CUDA"
"$GPU_PYTHON" - <<'PY'
import torch

print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
else:
    print("предупреждение: CUDA недоступна в этом окружении — обучение пойдёт на CPU")
PY

echo "setup_gpu_env: готово. Обучение: $GPU_PYTHON -m floodvalue train ...  (или make train GPU=1)"
