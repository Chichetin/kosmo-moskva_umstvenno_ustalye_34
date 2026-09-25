#!/usr/bin/env bash
# Страж коммита: в git не попадают растры Sen1Floods11 и крупные файлы вне Git LFS.
# Проверяет индекс (staged). Режимы:
#   pre-tool — PreToolUse на Bash: срабатывает только на `git commit`, отказывает JSON-ом;
#   (без аргумента) — git pre-commit (.githooks/pre-commit) и ручной запуск: код 1 при нарушении.
# Лимит — FLOODVALUE_MAX_FILE_MB (по умолчанию 5). Веса моделей — через LFS (.gitattributes).
set -uo pipefail

mode="${1:-git}"
if [ "$mode" = pre-tool ]; then
  cmd="$(jq -r '.tool_input.command // ""' 2>/dev/null)"
  printf '%s' "$cmd" | grep -qE '(^|[;&|[:space:]])git([[:space:]]+-C[[:space:]]+[^[:space:]]+)?[[:space:]]+commit' || exit 0
  dir="$(printf '%s' "$cmd" | sed -nE 's/.*git[[:space:]]+-C[[:space:]]+([^[:space:]]+).*/\1/p')"
  [ -z "$dir" ] && dir="$(printf '%s' "$cmd" | sed -nE 's/^[[:space:]]*cd[[:space:]]+([^[:space:];&]+).*/\1/p')"
  [ -n "$dir" ] && { cd "$dir" 2>/dev/null || exit 0; }
fi
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

MAX_MB="${FLOODVALUE_MAX_FILE_MB:-5}"
# Слои Sen1Floods11 (HandLabeled и WeakLabeled) и любые растры из каталогов данных.
DATA_RE='(S1Hand|S2Hand|LabelHand|JRCWaterHand|S1OtsuLabelHand|S1Weak|S2IndexLabelWeak|S1OtsuLabelWeak|S2Weak)[^/]*\.tiff?$|^(data|outputs)/.*\.(tiff?|npy|npz|h5|zarr)$'

FOUND=""
while IFS= read -r -d '' f; do
  [ -f "$f" ] || continue
  if printf '%s' "$f" | grep -qE "$DATA_RE"; then
    FOUND+="$f: растр/массив данных Sen1Floods11 — в git не кладём (data/ вне git, см. docs/data.md)"$'\n'
    continue
  fi
  size=$(( $(stat -c %s "$f") / 1024 / 1024 ))
  if [ "$size" -ge "$MAX_MB" ] && [ "$(git check-attr filter -- "$f" | awk '{print $NF}')" != lfs ]; then
    FOUND+="$f: ${size} МБ ≥ ${MAX_MB} МБ вне Git LFS (git lfs track + .gitattributes)"$'\n'
  fi
done < <(git diff --cached --name-only -z --diff-filter=AM)

[ -z "$FOUND" ] && exit 0

MSG="Коммит остановлен стражем данных:"$'\n'"$FOUND""Убрать из индекса: git restore --staged <путь>."
if [ "$mode" = pre-tool ]; then
  jq -n --arg r "$MSG" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
  exit 0
fi
printf '%s\n' "$MSG" >&2
exit 1
