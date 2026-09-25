# Одна команда на этап и одна на всё. Параметры: CHIP=<chip_id> BUDGET=<руб> OUT=<каталог комплекта>.
PY ?= uv run --frozen python -m floodvalue
CHIP ?=
BUDGET ?=
OUT ?=
GPU ?=
TAG ?=
STAGE_ARGS = $(if $(CHIP),--chip $(CHIP)) $(if $(S1),--s1 $(S1)) $(if $(BUDGET),--budget $(BUDGET)) $(if $(OUT),--out $(OUT))

# GPU-окружение для обучения (scripts/setup_gpu_env.sh, D-024): отдельный venv вне
# репозитория, вне uv.lock (тот фиксирует CPU-torch для инференса у жюри, D-003).
ifdef FLOODVALUE_GPU_VENV
GPU_VENV := $(FLOODVALUE_GPU_VENV)
else
GPU_VENV := $(HOME)/.cache/floodvalue-gpu
endif
GPU_PY := $(GPU_VENV)/bin/python -m floodvalue

.PHONY: setup test hooks-test check data data-qc split baseline train calibrate predict assets procure export validate app eval-test all gpu-env release

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

release:            ## архивы комплекта OUT и models/ + SHA256SUMS для релиза GitVerse (#85); TAG=<тег> — загрузить
	uv run --frozen python scripts/release_bundle.py $(if $(OUT),--out $(OUT)) $(if $(TAG),--upload $(TAG))

data-qc:            ## docs/verification/data-qc.md из interim/qc.csv (после make data); test — без пикселей
	uv run --frozen python -m floodvalue.data.qc_report

train:              ## GPU=1 — обучение через python из GPU-venv (make gpu-env), иначе CPU по uv.lock
ifeq ($(filter 1,$(GPU)),1)
	@[ -x "$(GPU_VENV)/bin/python" ] || { echo "train: нет GPU-venv в $(GPU_VENV) — сначала make gpu-env"; exit 1; }
	$(GPU_PY) train $(STAGE_ARGS)
else
	$(PY) train $(STAGE_ARGS)
endif

# Сквозной путь без обучения: веса и порог уже в репо (models/), бюджет меняется без переобучения.
# CHIP по умолчанию — configs/run.yaml; комплект -> $$FLOODVALUE_OUTPUT_DIR/<chip>_<budget>_<commit>
# (или OUT), validate проверяет именно его (самый свежий run_metadata.json, contracts §11).
ALL_CHIP = $(or $(CHIP),$(shell sed -n 's/^chip_id: *//p' configs/run.yaml))
ALL_ARGS = --chip $(ALL_CHIP) $(if $(BUDGET),--budget $(BUDGET)) $(if $(OUT),--out $(OUT))
HAS_MODEL_PREDICT = uv run --frozen python -c "import importlib.util as u, sys; sys.exit(u.find_spec('floodvalue.model.predict') is None)"

all:
	$(PY) data
	@# TODO(#16/#17): убрать запасной путь, когда floodvalue.model.predict будет в main.
	@if $(HAS_MODEL_PREDICT); then $(PY) predict $(ALL_ARGS); else \
	  echo "predict: floodvalue.model.predict ещё нет — временный основной метод baseline (D-009)"; \
	  uv run --frozen python -m floodvalue.baseline.predict --chip $(ALL_CHIP) $(if $(OUT),--out $(OUT)); fi
	$(PY) assets $(ALL_ARGS)
	$(PY) procure $(ALL_ARGS)
	$(PY) export $(ALL_ARGS)
	$(PY) validate $(if $(OUT),--out $(OUT))
