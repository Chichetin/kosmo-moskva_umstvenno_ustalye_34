#!/usr/bin/env bash
# Проверка хуков на временном git-репозитории: bash .claude/hooks/test-hooks.sh
set -uo pipefail

HOOKS="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP" && git init -q -b main . && mkdir -p .claude
git config user.name t && git config user.email t@t
FAIL=0

expect() { # expect <block|pass> <описание> <хук>
  local out
  out="$(bash "$HOOKS/$3" 2>&1)"
  if [ "$1" = block ] && ! printf '%s' "$out" | grep -q '"decision": *"block"'; then
    echo "FAIL: $2 — ожидалась блокировка, вывод: $out"; FAIL=1
  elif [ "$1" = pass ] && [ -n "$out" ]; then
    echo "FAIL: $2 — ожидался пустой вывод, вывод: $out"; FAIL=1
  else
    echo "ok:   $2"
  fi
}

# --- check-secrets.sh
expect pass  "чистый репозиторий"                   check-secrets.sh
echo 'GITVERSE_TOKEN=your_token_here' > .env.example
expect pass  ".env.example с плейсхолдером"         check-secrets.sh
echo 'token = "f9a8Xc72kLmQ01zYvB3n4dHs"' > sync.py
expect block "токен в коде"                         check-secrets.sh
echo 'token = "f9a8Xc72kLmQ01zYvB3n4dHs"  # secret-scan: allow' > sync.py
expect pass  "строка с явным secret-scan: allow"    check-secrets.sh
echo 'GH=ghp_abcdefghijklmnopqrstuvwxyz0123456789' > sync.py
expect block "токен GitHub"                         check-secrets.sh
rm sync.py
echo 'GITVERSE_TOKEN=abc' > .env
expect block ".env не в .gitignore"                 check-secrets.sh
echo '.env' > .gitignore
expect pass  ".env в .gitignore"                    check-secrets.sh
printf -- '-----BEGIN RSA PRIVATE KEY-----\n' > key.pem
expect block "приватный ключ"                       check-secrets.sh
rm key.pem

# --- check-tests.sh (pytest подменяется .venv/bin/python с заданным кодом выхода)
expect pass  "нет pyproject/tests — хук молчит"     check-tests.sh
touch pyproject.toml && mkdir -p tests .venv/bin
fake_pytest() { printf '#!/usr/bin/env bash\necho "fake pytest"\nexit %s\n' "$1" > .venv/bin/python; chmod +x .venv/bin/python; }
fake_pytest 5
expect pass  "pytest: тестов не найдено (код 5)"     check-tests.sh
fake_pytest 1; touch tests/test_x.py
expect block "pytest красный"                        check-tests.sh
fake_pytest 0
expect pass  "pytest зелёный"                        check-tests.sh
fake_pytest 1; touch -d '-1 min' .venv/bin/python tests/test_x.py
expect pass  "ничего не менялось с зелёного прогона" check-tests.sh
rm -rf .venv tests pyproject.toml .claude/.last-green-test-run

# --- check-data-files.sh (страж коммита)
expect_data() { # expect_data <block|pass> <описание>
  local rc out; out="$(bash "$HOOKS/check-data-files.sh" 2>&1)"; rc=$?
  if [ "$1" = block ] && [ "$rc" -eq 0 ]; then echo "FAIL: $2 — ожидался отказ, вывод: $out"; FAIL=1
  elif [ "$1" = pass ] && [ "$rc" -ne 0 ]; then echo "FAIL: $2 — ожидался пропуск, вывод: $out"; FAIL=1
  else echo "ok:   $2"; fi
}
echo 'x = 1' > small.py && git add small.py
expect_data pass  "данные: мелкий код проходит"
mkdir -p v && echo tif > v/Ghana_103272_S1Hand.tif && git add v/Ghana_103272_S1Hand.tif
expect_data block "данные: слой S1Hand вне data/ — отказ"
git rm -q --cached v/Ghana_103272_S1Hand.tif
mkdir -p data && echo x > data/chip.tif && git add -f data/chip.tif
expect_data block "данные: растр в data/ — отказ"
git rm -q --cached data/chip.tif
head -c 6000000 /dev/zero > weights.pt && git add weights.pt
expect_data block "данные: 6 МБ вне LFS — отказ"
echo 'weights.pt filter=lfs diff=lfs merge=lfs -text' > .gitattributes
expect_data pass  "данные: тот же файл под LFS-атрибутом проходит"
git rm -q --cached weights.pt small.py && rm -f weights.pt .gitattributes
git add v/Ghana_103272_S1Hand.tif
out="$(printf '{"tool_input":{"command":"git commit -m x"}}' | bash "$HOOKS/check-data-files.sh" pre-tool)"
if printf '%s' "$out" | jq -e '.hookSpecificOutput.permissionDecision == "deny"' >/dev/null 2>&1; then
  echo "ok:   данные pre-tool: git commit запрещён JSON-ом"
else echo "FAIL: данные pre-tool: нет deny, вывод: $out"; FAIL=1; fi
out="$(printf '{"tool_input":{"command":"git status"}}' | bash "$HOOKS/check-data-files.sh" pre-tool)"
if [ -z "$out" ]; then echo "ok:   данные pre-tool: не commit — молчит"
else echo "FAIL: данные pre-tool: сработал не на commit, вывод: $out"; FAIL=1; fi
git rm -q --cached v/Ghana_103272_S1Hand.tif && rm -rf v data

# --- session-context.sh (SessionStart: stdout уходит в контекст)
expect_out() { # expect_out <has|lacks> <описание> <шаблон>
  local out
  out="$(bash "$HOOKS/session-context.sh" 2>&1)"
  if [ "$1" = has ] && ! printf '%s' "$out" | grep -q -- "$3"; then
    echo "FAIL: $2 — нет «$3», вывод: $out"; FAIL=1
  elif [ "$1" = lacks ] && printf '%s' "$out" | grep -q -- "$3"; then
    echo "FAIL: $2 — лишнее «$3», вывод: $out"; FAIL=1
  else
    echo "ok:   $2"
  fi
}
echo init > README.md && git add README.md .gitignore .env.example && git commit -q -m init
mkdir -p "$TMP/bin"
printf '#!/usr/bin/env bash\n[ "$1" = auth ] && exit 1\nexit 1\n' > "$TMP/bin/gh"; chmod +x "$TMP/bin/gh"
PATH="$TMP/bin:$PATH" expect_out has "gh без авторизации — предупреждение, не падение" "не авторизован"
cat > "$TMP/bin/gh" <<'GH'
#!/usr/bin/env bash
case "$1 $2" in
  "auth status") exit 0 ;;
  "api user") echo me ;;
  "issue list") echo '[{"number":7,"title":"baseline порог","labels":[{"name":"area:baseline"}],"assignees":[{"login":"me"}]},
    {"number":9,"title":"цена ПП 840","labels":[{"name":"area:procurement"}],"assignees":[{"login":"mate"}]},
    {"number":11,"title":"свободная","labels":[{"name":"area:app"}],"assignees":[]}]' ;;
esac
GH
PATH="$TMP/bin:$PATH" expect_out has   "issues: моя задача"                    "#7 baseline порог"
PATH="$TMP/bin:$PATH" expect_out has   "issues: занятая area: соседа"          "area:procurement ← #9 mate"
PATH="$TMP/bin:$PATH" expect_out lacks "issues: area: без assignee не занята"   "area:app"
FLOODVALUE_SKIP_ISSUES=1 expect_out has "основная копия на main — предупреждение" "основной копии"
git worktree add -q .claude/worktrees/wt-test -b worktree-wt-test 2>/dev/null
(cd .claude/worktrees/wt-test && FLOODVALUE_SKIP_ISSUES=1 expect_out lacks "worktree — без предупреждения" "основной копии")
(cd .claude/worktrees/wt-test && FLOODVALUE_SKIP_ISSUES=1 expect_out has   "worktree — видна ветка"        "worktree-wt-test")

# --- check-worktrees.sh (SubagentStop: секреты и данные во всех worktree)
expect pass  "worktrees: чисто везде"                   check-worktrees.sh
echo 'token = "f9a8Xc72kLmQ01zYvB3n4dHs"' > .claude/worktrees/wt-test/bot.py
expect block "worktrees: секрет в чужом worktree"       check-worktrees.sh
rm .claude/worktrees/wt-test/bot.py
(cd .claude/worktrees/wt-test && echo t > Spain_1_LabelHand.tif && git add Spain_1_LabelHand.tif)
expect block "worktrees: растр в индексе worktree"      check-worktrees.sh
(cd .claude/worktrees/wt-test && git restore --staged Spain_1_LabelHand.tif && rm Spain_1_LabelHand.tif)
expect pass  "worktrees: после уборки чисто"            check-worktrees.sh

# --- check-upstream.sh: чужие коммиты в origin/main против моей работы
cd "$TMP" && git init -q --bare -b main remote.git
git clone -q remote.git mate 2>/dev/null
mate_push() { # mate_push <файл> <сообщение>
  (cd "$TMP/mate" && mkdir -p "$(dirname "$1")" && echo "$2" >> "$1" && git add "$1" &&
   git -c user.name=mate -c user.email=m@m commit -q -m "$2" && git push -q origin HEAD:main)
}
mate_push README.md "init"
git clone -q remote.git me && cd me && git checkout -q -b worktree-me origin/main
expect_up() { # expect_up <block|pass> <описание> [шаблон]
  local out
  out="$(echo '{}' | bash "$HOOKS/check-upstream.sh" stop 2>&1)"
  if [ "$1" = block ] && ! { printf '%s' "$out" | grep -q '"decision": *"block"' &&
                            printf '%s' "$out" | grep -q -- "$3"; }; then
    echo "FAIL: $2 — ожидалась блокировка с «$3», вывод: $out"; FAIL=1
  elif [ "$1" = pass ] && [ -n "$out" ]; then
    echo "FAIL: $2 — ожидался пустой вывод, вывод: $out"; FAIL=1
  else
    echo "ok:   $2"
  fi
}
expect_up pass  "upstream: новых коммитов нет"
mate_push other.txt "docs: чужой файл"
expect_up pass  "upstream: чужой коммит не пересекается с моими файлами"
echo mine > shared.txt
mate_push shared.txt "api: тот же файл"
expect_up block "upstream: пересечение с незакоммиченным файлом" "shared.txt"
expect_up pass  "upstream: то же состояние origin/main второй раз не блокирует"
echo mine > new.txt && git add new.txt && git -c user.name=me -c user.email=me@me commit -q -m wip
mate_push new.txt "assets: тот же новый файл"
expect_up block "upstream: пересечение с закоммиченным в ветке файлом" "new.txt"
mate_push docs/contracts.md "contracts: схема"
expect_up block "upstream: изменился контракт — показать всегда" "docs/contracts.md"
mate_push shared.txt "api: ещё правка"
out="$(echo '{}' | bash "$HOOKS/check-upstream.sh" post 2>&1)"
if printf '%s' "$out" | jq -e '.hookSpecificOutput.additionalContext | contains("shared.txt")' >/dev/null 2>&1; then
  echo "ok:   upstream post: факты в additionalContext без блокировки"
else echo "FAIL: upstream post: нет additionalContext, вывод: $out"; FAIL=1; fi
mate_push shared.txt "api: третья правка"
out="$(echo '{}' | bash "$HOOKS/check-upstream.sh" post 2>&1)"
if [ -z "$out" ]; then echo "ok:   upstream post: не чаще раза в 5 минут"
else echo "FAIL: upstream post: сработал раньше 5 минут, вывод: $out"; FAIL=1; fi
git remote set-url origin "$TMP/nope.git"
expect_up pass  "upstream: сеть недоступна — молчит"

exit $FAIL
