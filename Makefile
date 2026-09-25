# Одна команда на этап и одна на всё. Параметры: CHIP=<chip_id> BUDGET=<руб>.
PY ?= uv run --frozen python -m floodvalue
CHIP ?=
BUDGET ?=
GPU ?=
STAGE_ARGS = $(if $(CHIP),--chip $(CHIP)) $(if $(BUDGET),--budget $(BUDGET))

# GPU-окружение для обучения (scripts/setup_gpu_env.sh, D-023): отдельный venv вне
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

# validate — критерий 12 (issue area:contracts/export, ещё не влит). Пока
# floodvalue.export.validate не реализован, check не роняет остальные зелёные
# проверки — печатает явную причину; настоящий отказ validate всё равно роняет check.
check: test hooks-test   ## полный прогон перед push
	@out="$$($(PY) validate $(STAGE_ARGS) 2>&1)"; code=$$?; echo "$$out"; if [ $$code -ne 0 ]; then if echo "$$out" | grep -q "^stage 'validate' not implemented yet"; then echo "check: validate пропущен — floodvalue.export.validate ещё не реализован (issue area:contracts/export, критерий 12); test и hooks-test зелёные"; else exit $$code; fi; fi

data split baseline calibrate predict assets procure export validate app eval-test:
	$(PY) $@ $(STAGE_ARGS)

train:              ## GPU=1 — обучение через python из GPU-venv (make gpu-env), иначе CPU по uv.lock
	$(if $(filter 1,$(GPU)),$(GPU_PY) train $(STAGE_ARGS),$(PY) train $(STAGE_ARGS))

# Сквозной путь без обучения: веса и порог уже в репо (models/), бюджет меняется без переобучения.
all: data predict assets procure export validate
