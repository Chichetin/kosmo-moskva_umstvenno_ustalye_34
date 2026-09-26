#!/usr/bin/env bash
# LOEO U-Net на локальной видеокарте — замена notebooks/loeo_colab.ipynb для своей машины.
#
# Та же единственная команда, что в блокноте (`python -m floodvalue.model.unet.loeo`), но без
# Colab: код — этот клон (без git clone и токена), данные — уже скачанные в FLOODVALUE_DATA_DIR,
# GPU-torch — venv вне репозитория (scripts/setup_gpu_env.sh, D-024), результаты — локально.
#
#   bash scripts/loeo_local.sh smoke   # фолд USA, 2 эпохи, 4 чипа на событие: проверка проводки
#   bash scripts/loeo_local.sh full    # 6 фолдов x 60 эпох, один seed (D-019, D-036)
#
# Переменные: FLOODVALUE_DATA_DIR, FLOODVALUE_OUTPUT_DIR (из .env), FLOODVALUE_GPU_VENV,
# LOEO_DEVICE (по умолчанию cuda). Validation и test модуль не читает; models/ не меняется.
set -euo pipefail

MODE="${1:-full}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export FLOODVALUE_DATA_DIR="${FLOODVALUE_DATA_DIR:-$REPO/data}"
export FLOODVALUE_OUTPUT_DIR="${FLOODVALUE_OUTPUT_DIR:-$REPO/outputs}"
GPU_VENV="${FLOODVALUE_GPU_VENV:-$HOME/.cache/floodvalue-gpu}"
PY="$GPU_VENV/bin/python"
DEVICE="${LOEO_DEVICE:-cuda}"

case "$MODE" in
  full)  OUT="$FLOODVALUE_OUTPUT_DIR/loeo";       ARGS=() ;;
  smoke) OUT="$FLOODVALUE_OUTPUT_DIR/loeo-smoke"; ARGS=(--events USA --max-chips 4 --epochs 2 --eval-epochs 1,2) ;;
  *) echo "usage: $0 full|smoke" >&2; exit 2 ;;
esac

[ -x "$PY" ] || bash scripts/setup_gpu_env.sh
# Код этого клона, а не путь, с которого venv когда-то ставил пакет в режиме -e.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export FLOODVALUE_COMMIT="$(git rev-parse HEAD)"
dirty="$(git status --porcelain --untracked-files=no | wc -l)"
[ "$dirty" -eq 0 ] || echo "ВНИМАНИЕ: в клоне $dirty изменённых файлов — числа не привязаны к commit $FLOODVALUE_COMMIT" >&2

"$PY" - "$DEVICE" <<'EOF'
import sys, torch, floodvalue
dev = sys.argv[1]
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", ok, "|", torch.cuda.get_device_name(0) if ok else "-")
print("floodvalue из", floodvalue.__file__)
if dev.startswith("cuda") and not ok:
    sys.exit("CUDA недоступна: запустите `make gpu-env` или LOEO_DEVICE=cpu (часы)")
EOF

# Данные уже скачаны: команда ничего не качает, только проверяет наличие и QC.
"$PY" -m floodvalue data

mkdir -p "$OUT"
echo "LOEO $MODE -> $OUT (commit ${FLOODVALUE_COMMIT:0:12}, device $DEVICE)"
start=$(date +%s)
"$PY" -m floodvalue.model.unet.loeo --out-dir "$OUT" --device "$DEVICE" "${ARGS[@]}" 2>&1 | tee "$OUT/loeo.log"
"$PY" -m floodvalue.model.unet.loeo --table-from "$OUT" | tee "$OUT/loeo-table.md"
echo "$FLOODVALUE_COMMIT" > "$OUT/COMMIT"
echo "LOEO: $(( $(date +%s) - start )) с"

"$PY" - "$OUT" "$MODE" "$FLOODVALUE_COMMIT" <<'EOF'
import os, shutil, sys
out, mode, commit = sys.argv[1:]
files = sorted(f for f in os.listdir(out) if f.startswith("loeo"))
assert "loeo-metrics.csv" in files and "loeo-summary.json" in files, files
archive = shutil.make_archive(os.path.join(os.path.dirname(out), f"loeo_{mode}_{commit[:12]}"), "zip", out)
print("архив:", archive, os.path.getsize(archive), "байт:", ", ".join(files))
EOF
