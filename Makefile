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
all: data predict assets procure export validate
