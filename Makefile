# Одна команда на этап и одна на всё. Параметры: CHIP=<chip_id> BUDGET=<руб> OUT=<каталог комплекта>.
PY ?= uv run --frozen python -m floodvalue
CHIP ?=
BUDGET ?=
OUT ?=
GPU ?=
STAGE_ARGS = $(if $(CHIP),--chip $(CHIP)) $(if $(BUDGET),--budget $(BUDGET)) $(if $(OUT),--out $(OUT))

# GPU-окружение для обучения (scripts/setup_gpu_env.sh, D-024): отдельный venv вне
# репозитория, вне uv.lock (тот фиксирует CPU-torch для инференса у жюри, D-003).
ifdef FLOODVALUE_GPU_VENV
GPU_VENV := $(FLOODVALUE_GPU_VENV)
else
GPU_VENV := $(HOME)/.cache/floodvalue-gpu
endif
GPU_PY := $(GPU_VENV)/bin/python -m floodvalue

.PHONY: setup test hooks-test check data split baseline train calibrate predict assets procure export validate app eval-test all gpu-env

setup:            ## окружение строго по lockfile + git-хук стража данных
	uv sync --frozen
	git config core.hooksPath .githooks

gpu-env:           ## venv вне репозитория с GPU-torch (cu128) для make train GPU=1; uv.lock не меняет
	bash scripts/setup_gpu_env.sh

test:
	uv run --frozen pytest

hooks-test:
	bash .claude/hooks/test-hooks.sh

check: test hooks-test validate   ## полный прогон перед push

data split baseline calibrate predict assets procure export validate app eval-test:
	$(PY) $@ $(STAGE_ARGS)

train:              ## GPU=1 — обучение через python из GPU-venv (make gpu-env), иначе CPU по uv.lock
ifeq ($(filter 1,$(GPU)),1)
	@[ -x "$(GPU_VENV)/bin/python" ] || { echo "train: нет GPU-venv в $(GPU_VENV) — сначала make gpu-env"; exit 1; }
	$(GPU_PY) train $(STAGE_ARGS)
else
	$(PY) train $(STAGE_ARGS)
endif

# Сквозной путь без обучения: веса и порог уже в репо (models/), бюджет меняется без переобучения.
all: data predict assets procure export validate
