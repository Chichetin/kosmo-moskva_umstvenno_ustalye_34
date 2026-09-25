#!/usr/bin/env bash
# Зеркало сдачи в GitVerse (D-002, GIT.md «GitVerse»): push origin/main и тегов,
# LFS-объекты (веса) — явно, затем сверка по чистому клону GitVerse.
#   bash scripts/sync-gitverse.sh           # синхронизация + сверка
#   bash scripts/sync-gitverse.sh --verify  # только сверка
# Токен — GITVERSE_TOKEN из .env (в .gitignore); в URL, remote и git config не пишется.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${GITVERSE_TOKEN:?GITVERSE_TOKEN не задан (.env)}"
GITVERSE_USER="${GITVERSE_USER:-chichetin}"
GITVERSE_URL="${GITVERSE_URL:-https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34.git}"
[ "$GITVERSE_TOKEN" = your_token_here ] && { echo "GITVERSE_TOKEN — плейсхолдер" >&2; exit 1; }

# Учётные данные только на время команды: helper читает токен из окружения.
export GV_USER="$GITVERSE_USER"
export GIT_TERMINAL_PROMPT=0
gv() { git -c credential.helper= \
  -c 'credential.helper=!f() { echo "username=$GV_USER"; echo "password=$GITVERSE_TOKEN"; }; f' "$@"; }

git remote get-url gitverse >/dev/null 2>&1 || git remote add gitverse "$GITVERSE_URL"

if [ "${1:-}" != --verify ]; then
  git fetch -q --prune origin
  git lfs fetch -q --all origin 2>/dev/null || true   # веса нужны локально, чтобы их отправить
  sha="$(git rev-parse origin/main)"
  # main — рабочая ветка; master — ветка по умолчанию GitVerse, держим её равной main.
  gv lfs push gitverse "$sha"
  gv push gitverse "$sha:refs/heads/main" "$sha:refs/heads/master"
  gv push gitverse --tags 2>/dev/null || true
  echo "[sync] отправлен origin/main $sha"
fi

# Сверка: commit совпадает, LFS-файлы в чистом клоне GitVerse — содержимое, а не указатели.
want="$(git rev-parse origin/main)"
have="$(gv ls-remote gitverse refs/heads/main | cut -f1)"
[ "$want" = "$have" ] || { echo "[verify] FAIL: GitVerse main=$have, origin/main=$want" >&2; exit 1; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
gv clone -q --branch main "$GITVERSE_URL" "$tmp/c"
(cd "$tmp/c" && gv lfs pull >/dev/null)
bad=0; n=0
while IFS= read -r f; do
  n=$((n + 1))
  if head -c 100 "$tmp/c/$f" | grep -q '^version https://git-lfs'; then
    echo "[verify] FAIL: $f в GitVerse — указатель, объекта нет" >&2; bad=1
  else
    echo "[verify] ok: $f $(sha256sum "$tmp/c/$f" | cut -c1-16)… $(stat -c %s "$tmp/c/$f") байт"
  fi
done < <(cd "$tmp/c" && git lfs ls-files -n)
[ "$bad" = 0 ] || exit 1
echo "[verify] ok: GitVerse main = $want, LFS-файлов: $n"
