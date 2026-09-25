# Одна команда на этап и одна на всё. Параметры: CHIP=<chip_id> BUDGET=<руб> OUT=<каталог комплекта>.
PY ?= uv run --frozen python -m floodvalue
CHIP ?=
BUDGET ?=
OUT ?=
STAGE_ARGS = $(if $(CHIP),--chip $(CHIP)) $(if $(BUDGET),--budget $(BUDGET)) $(if $(OUT),--out $(OUT))

.PHONY: setup test hooks-test check data split baseline train calibrate predict assets procure export validate app eval-test all

setup:            ## окружение строго по lockfile + git-хук стража данных
	uv sync --frozen
	git config core.hooksPath .githooks

test:
	uv run --frozen pytest

hooks-test:
	bash .claude/hooks/test-hooks.sh

check: test hooks-test validate   ## полный прогон перед push

data split baseline train calibrate predict assets procure export validate app eval-test:
	$(PY) $@ $(STAGE_ARGS)

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
