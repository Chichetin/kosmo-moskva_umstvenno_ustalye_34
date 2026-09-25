#!/usr/bin/env bash
# Stop hook: ход не считается завершённым, если pytest красный.
# Нет pyproject.toml или tests/, или pytest не нашёл ни одного теста (код 5) — молчит.
# Прогон пропускается, если ничего не менялось с последнего зелёного.
set -uo pipefail

MARKER=".claude/.last-green-test-run"
[ -f pyproject.toml ] && [ -d tests ] || exit 0

if [ -f "$MARKER" ]; then
  CHANGED="$(find . -type f -newer "$MARKER" \
    -not -path './.git/*' -not -path './.claude/*' -not -path './data/*' -not -path './outputs/*' \
    -not -path '*/.venv/*' -not -path '*/__pycache__/*' -not -path '*/.pytest_cache/*' 2>/dev/null | head -1)"
  [ -z "$CHANGED" ] && exit 0
fi

if [ -x .venv/bin/python ]; then PYTEST=(.venv/bin/python -m pytest)
elif [ -x .venv/Scripts/python.exe ]; then PYTEST=(.venv/Scripts/python.exe -m pytest)  # Windows venv
elif command -v uv >/dev/null 2>&1; then PYTEST=(uv run --frozen pytest)
elif command -v python3 >/dev/null 2>&1; then PYTEST=(python3 -m pytest)
else PYTEST=(python -m pytest)
fi

OUT="$("${PYTEST[@]}" -q -x 2>&1)"; STATUS=$?
if [ "$STATUS" -eq 0 ] || [ "$STATUS" -eq 5 ]; then
  mkdir -p .claude && touch "$MARKER"
  exit 0
fi

TAIL="$(printf '%s' "$OUT" | tail -n 40)"
jq -n --arg out "$TAIL" '{decision: "block", reason: ("pytest красный перед завершением хода:\n" + $out)}'
exit 0
