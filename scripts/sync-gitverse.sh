#!/usr/bin/env bash
# Зеркало сдачи в GitVerse (D-002, GIT.md «GitVerse»): push origin/main в main, master и
# ветку по умолчанию GitVerse, LFS-объекты (веса) — явно, теги origin; затем сверка по
# чистому клону ветки по умолчанию (её клонирует жюри).
#   bash scripts/sync-gitverse.sh           # синхронизация + сверка
#   bash scripts/sync-gitverse.sh --verify  # только сверка
# FLOODVALUE_REQUIRE_WEIGHTS=1 — сверка падает, если LFS-файлов нет (перед сдачей).
# Токен — GITVERSE_TOKEN из .env (в .gitignore); в URL, remote и git config не пишется.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${GITVERSE_TOKEN:?GITVERSE_TOKEN не задан (.env)}"
: "${GITVERSE_USER:?GITVERSE_USER не задан (.env) — логин GitVerse}"
GITVERSE_URL="${GITVERSE_URL:-https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34.git}"
GITVERSE_API="${GITVERSE_API:-https://api.gitverse.ru/repos/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34}"
[ "$GITVERSE_TOKEN" = your_token_here ] && { echo "GITVERSE_TOKEN — плейсхолдер" >&2; exit 1; }

# Учётные данные только на время команды: helper читает токен из окружения.
export GV_USER="$GITVERSE_USER"
export GIT_TERMINAL_PROMPT=0
gv() { git -c credential.helper= \
  -c 'credential.helper=!f() { echo "username=$GV_USER"; echo "password=$GITVERSE_TOKEN"; }; f' "$@"; }

# remote без токена и ровно на наш репозиторий
if git remote get-url gitverse >/dev/null 2>&1; then git remote set-url gitverse "$GITVERSE_URL"
else git remote add gitverse "$GITVERSE_URL"; fi

default_branch="$(curl -sf -m 30 -H "Authorization: Bearer $GITVERSE_TOKEN" \
  -H 'Accept: application/vnd.gitverse.object+json;version=1' "$GITVERSE_API" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["default_branch"])')"
branches=(main master)
[[ " ${branches[*]} " == *" $default_branch "* ]] || branches+=("$default_branch")

git fetch -q --prune --tags origin
want="$(git rev-parse origin/main)"

if [ "${1:-}" != --verify ]; then
  git lfs fetch --all origin               # веса нужны локально, чтобы их отправить
  gv lfs push gitverse "$want"
  gv push gitverse $(printf "${want}:refs/heads/%s " "${branches[@]}")
  mapfile -t tags < <(git ls-remote --tags --refs origin | cut -f2)
  [ "${#tags[@]}" -gt 0 ] && gv push gitverse "${tags[@]}"
  echo "[sync] отправлен origin/main $want → ${branches[*]}; тегов: ${#tags[@]}"
fi

# Сверка: все ветки = origin/main, LFS-файлы в клоне ветки по умолчанию — содержимое.
for b in "${branches[@]}"; do
  have="$(gv ls-remote gitverse "refs/heads/$b" | cut -f1)"
  [ "$want" = "$have" ] || { echo "[verify] FAIL: GitVerse $b=${have:-нет}, origin/main=$want" >&2; exit 1; }
done
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
gv clone -q "$GITVERSE_URL" "$tmp/c"
[ "$(git -C "$tmp/c" rev-parse HEAD)" = "$want" ] || { echo "[verify] FAIL: клон по умолчанию ≠ origin/main" >&2; exit 1; }
(cd "$tmp/c" && gv lfs pull)
mapfile -t lfs_files < <(cd "$tmp/c" && git lfs ls-files -n)
bad=0
for f in "${lfs_files[@]}"; do
  if [[ "$(head -c 100 "$tmp/c/$f" 2>/dev/null | tr -d '\0')" == "version https://git-lfs"* ]]; then
    echo "[verify] FAIL: $f в GitVerse — указатель, объекта нет" >&2; bad=1
  else
    echo "[verify] ok: $f $(sha256sum "$tmp/c/$f" | cut -c1-16)… $(stat -c %s "$tmp/c/$f") байт"
  fi
done
[ "$bad" = 0 ] || exit 1
if [ "${#lfs_files[@]}" -eq 0 ]; then
  [ "${FLOODVALUE_REQUIRE_WEIGHTS:-0}" = 1 ] && { echo "[verify] FAIL: LFS-файлов (весов) нет" >&2; exit 1; }
  echo "[verify] WARN: LFS-файлов (весов) пока нет"
fi
echo "[verify] ok: GitVerse ${branches[*]} = $want (по умолчанию: $default_branch), LFS-файлов: ${#lfs_files[@]}"
