#!/usr/bin/env bash
# Зеркало сдачи в GitVerse (D-002, GIT.md «GitVerse»): push origin/main в main, master и
# ветку по умолчанию GitVerse, LFS-объекты (веса) — явно и первыми, теги origin; затем
# сверка по чистому клону ветки по умолчанию (её клонирует жюри).
#   bash scripts/sync-gitverse.sh              # LFS, затем код + сверка; LFS упал — выход 1
#   bash scripts/sync-gitverse.sh --code-only  # LFS упал — всё равно отправить код (веса — указатели)
#   bash scripts/sync-gitverse.sh --verify     # только сверка
# FLOODVALUE_REQUIRE_WEIGHTS=1 — сверка падает, если LFS-файлов нет (перед сдачей).
# GITVERSE_LFS_TIMEOUT — предел LFS-шага в секундах (по умолчанию 900).
# Токен — GITVERSE_TOKEN из .env (в .gitignore); в URL, remote, git config и файлы не
# пишется. Трассировку git (GIT_TRACE*, GIT_CURL_VERBOSE) не включать: она печатает Authorization.
set -euo pipefail

mode=sync
case "${1:-}" in
  "") ;;
  --verify) mode=verify ;;
  --code-only) mode=code-only ;;
  *) echo "неизвестный аргумент: $1 (есть --verify, --code-only)" >&2; exit 2 ;;
esac

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${GITVERSE_TOKEN:?GITVERSE_TOKEN не задан (.env)}"
: "${GITVERSE_USER:?GITVERSE_USER не задан (.env) — логин GitVerse}"
GITVERSE_URL="${GITVERSE_URL:-https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34.git}"
GITVERSE_API="${GITVERSE_API:-https://api.gitverse.ru/repos/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34}"
LFS_TIMEOUT="${GITVERSE_LFS_TIMEOUT:-900}"
[ "$GITVERSE_TOKEN" = your_token_here ] && { echo "GITVERSE_TOKEN — плейсхолдер" >&2; exit 1; }
unset GIT_TRACE GIT_TRACE_CURL GIT_TRACE_PACKET GIT_CURL_VERBOSE GIT_TRACE2 GIT_TRACE2_EVENT

# Учётные данные только на время команды: helper читает токен из окружения.
# HTTP/1.1: по HTTP/2 GitVerse рвёт LFS-поток (PROTOCOL_ERROR); повторы LFS ограничены,
# иначе при 403 git-lfs повторяет без конца.
export GV_USER="$GITVERSE_USER"
export GIT_TERMINAL_PROMPT=0
GV=(git -c credential.helper=
  -c 'credential.helper=!f() { echo "username=$GV_USER"; echo "password=$GITVERSE_TOKEN"; }; f'
  -c http.version=HTTP/1.1
  -c lfs.concurrenttransfers=1
  -c lfs.transfer.maxretries=2
  -c lfs.transfer.maxretrydelay=10)
gv() { "${GV[@]}" "$@"; }
# git-lfs (Go): HTTP/2 выключен и через окружение — на случай, если http.version не подхвачен.
gv_lfs() { GODEBUG=http2client=0 timeout "$LFS_TIMEOUT" "${GV[@]}" lfs "$@"; }

# remote без токена и ровно на наш репозиторий
if git remote get-url gitverse >/dev/null 2>&1; then git remote set-url gitverse "$GITVERSE_URL"
else git remote add gitverse "$GITVERSE_URL"; fi

# заголовок с токеном — через stdin, не в аргументах curl (их видно в ps)
default_branch="$(printf 'Authorization: Bearer %s\n' "$GITVERSE_TOKEN" |
  curl -sf -m 30 -H @- -H 'Accept: application/vnd.gitverse.object+json;version=1' "$GITVERSE_API" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["default_branch"])')"
branches=(main master)
[[ " ${branches[*]} " == *" $default_branch "* ]] || branches+=("$default_branch")

git fetch -q --prune --tags origin
want="$(git rev-parse origin/main)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

if [ "$mode" != verify ]; then
  git lfs fetch --all origin               # веса нужны локально, чтобы их отправить
  lfs_ok=1
  echo "[sync] LFS → GitVerse (HTTP/1.1, предел ${LFS_TIMEOUT} с)"
  if gv_lfs push gitverse "$want" 2>&1 | tee "$tmp/lfs.log"; then :; else
    rc=$?; lfs_ok=0
    if [ "$rc" = 124 ]; then
      echo "[sync] FAIL: LFS не уложился в ${LFS_TIMEOUT} с (GITVERSE_LFS_TIMEOUT)" >&2
    elif grep -qiE '403|Authorization error|Forbidden' "$tmp/lfs.log"; then
      echo "[sync] FAIL: LFS PUT → 403: у токена нет записи в LFS GitVerse — нужен токен с правом записи/LFS (или права от admin hackrus.experts)" >&2
    else
      echo "[sync] FAIL: git lfs push завершился с кодом $rc" >&2
    fi
  fi
  rm -f "$tmp/lfs.log"
  push_opts=()
  if [ "$lfs_ok" = 0 ]; then
    if [ "$mode" != code-only ]; then
      echo "[sync] код в GitVerse не отправлен. Чтобы GitVerse не отставал: bash scripts/sync-gitverse.sh --code-only (веса останутся указателями)" >&2
      exit 1
    fi
    echo "[sync] WARN: --code-only — код отправляется без LFS (--no-verify), веса в GitVerse — указатели; сверка ниже даст FAIL" >&2
    push_opts=(--no-verify)
  fi
  refspecs=()
  for b in "${branches[@]}"; do refspecs+=("${want}:refs/heads/${b}"); done
  gv push "${push_opts[@]}" gitverse "${refspecs[@]}"
  mapfile -t tags < <(git ls-remote --tags --refs origin | cut -f2)
  [ "${#tags[@]}" -gt 0 ] && gv push "${push_opts[@]}" gitverse "${tags[@]}"
  echo "[sync] отправлен origin/main ${want} → ${branches[*]}; тегов: ${#tags[@]}"
fi

# Сверка: все ветки = origin/main, LFS-файлы в клоне ветки по умолчанию — содержимое.
for b in "${branches[@]}"; do
  have="$(gv ls-remote gitverse "refs/heads/$b" | cut -f1)"
  [ "$want" = "$have" ] || { echo "[verify] FAIL: GitVerse $b=${have:-нет}, origin/main=$want" >&2; exit 1; }
done
# клон без smudge: при отсутствии объектов clone не падает, а указатели ловятся ниже
GIT_LFS_SKIP_SMUDGE=1 gv clone -q "$GITVERSE_URL" "$tmp/c"
[ "$(git -C "$tmp/c" rev-parse HEAD)" = "$want" ] || { echo "[verify] FAIL: клон по умолчанию ≠ origin/main" >&2; exit 1; }
(cd "$tmp/c" && gv_lfs pull) || echo "[verify] WARN: git lfs pull из GitVerse завершился с ошибкой" >&2
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
