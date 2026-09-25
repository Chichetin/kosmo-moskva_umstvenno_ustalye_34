#!/usr/bin/env bash
# SessionStart: stdout попадает в контекст сессии. Показывает, где агент стоит в Git,
# предупреждает о работе в основной копии на main, выводит мои открытые issues и
# занятые area: (открытые issues с assignee). Ничего не блокирует. Без gh или без
# авторизации — одно предупреждение вместо списка.
set -uo pipefail

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

branch="$(git branch --show-current 2>/dev/null)"
dirty="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
echo "[git] ветка: ${branch:-detached}; незакоммиченных путей: $dirty; $(git rev-parse --show-toplevel)"

# В связанном worktree git-dir и git-common-dir различаются.
if [ "$(git rev-parse --git-dir)" = "$(git rev-parse --git-common-dir)" ] && [ "$branch" = main ]; then
  echo "[git] ВНИМАНИЕ: ты в основной копии на main. Задачу, меняющую файлы, делай в своём worktree:"
  echo "      git worktree add .claude/worktrees/<issue>-<тема> -b worktree-<issue>-<тема> origin/main (GIT.md)"
  [ "$dirty" != 0 ] && echo "      Незакоммиченные изменения здесь могут быть чужими — не коммить и не трогай их."
fi

[ "${FLOODVALUE_SKIP_ISSUES:-0}" = 1 ] && exit 0
if ! command -v gh >/dev/null 2>&1 || ! timeout 5 gh auth status >/dev/null 2>&1; then
  echo "[issues] gh недоступен или не авторизован — список задач не показан (gh auth login)."
  exit 0
fi
issues="$(timeout 8 gh issue list --state open --limit 200 \
  --json number,title,labels,assignees 2>/dev/null)" || {
  echo "[issues] не удалось получить issues (нет remote или сети) — проверь вручную: gh issue list"; exit 0; }

me="$(timeout 5 gh api user --jq .login 2>/dev/null)"
mine="$(printf '%s' "$issues" | jq -r --arg me "$me" \
  '.[] | select(any(.assignees[]; .login == $me)) | "  #\(.number) \(.title) [\([.labels[].name] | join(", "))]"')"
busy="$(printf '%s' "$issues" | jq -r \
  '[.[] | select(.assignees | length > 0) | . as $i | .labels[].name | select(startswith("area:"))
    | "\(.) ← #\($i.number) \($i.assignees | map(.login) | join(","))"] | unique | .[] | "  " + .')"
echo "[issues] мои открытые (${me:-?}):"; [ -n "$mine" ] && printf '%s\n' "$mine" || echo "  нет"
echo "[issues] занятые area: (issue той же area: не брать, пока она в работе):"
[ -n "$busy" ] && printf '%s\n' "$busy" || echo "  нет"
