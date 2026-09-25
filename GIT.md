# GIT.md — правила ведения Git

Цель: `main` в любой момент — рабочий кандидат на сдачу (`make check` зелёный,
12 файлов проходят валидатор), а сданную версию можно однозначно восстановить по commit.

Правило: **закончил одно проверенное действие — закоммить и запушь ветку.**

## Хостинг

- Рабочий репозиторий — **GitHub, приватный** (D-002): issues, PR, `gh`. `origin` —
  GitHub.
- **`main` защищена на сервере** (включено 25.09 через API, `protected: true`): только
  через PR (0 обязательных апрувов — аппрув даёт `reviewer`), правило действует и на
  админа, force-push и удаление `main` запрещены. Дублируется `permissions.deny` в
  `.claude/settings.json`. Проверка: `gh api repos/Chichetin/kosmo-moskva_umstvenno_ustalye_34/branches/main --jq .protected`.
- **Сдача — GitVerse**, зеркало (`gitverse` remote) заводится позже; см. «GitVerse».

## Автономность агента

Агент коммитит и пушит **свою ветку сам, без запроса подтверждения**, создаёт PR, по
вердикту `reviewer` «merge» и зелёному прогону сам вливает его (`gh pr merge --merge`).
Разрешения — `permissions.allow`, запреты — `permissions.deny` в `.claude/settings.json`.

- перед коммитом и push — `git fetch --prune` и просмотр обеих сторон расхождения;
- в коммит — только текущий шаг: `git add <явные пути>`. Никогда `git add -A`,
  `git add .`, `git commit -a`: в дереве может лежать чужая работа;
- `--force`, `rebase`, `amend`, `reset --hard`, `clean` — только по явной просьбе человека;
- секреты, `.env`, `data/`, `outputs/`, растры не коммитятся (Stop-хук
  `check-secrets.sh`, PreToolUse- и pre-commit-хук `check-data-files.sh`).
- Трейлеры соавторства ИИ (`Co-Authored-By`, «Generated with Claude Code») в коммиты и
  PR не добавляются.

## Worktree на задачу

Основная копия — не рабочее место: в ней сидят человек и другие сессии. **Любая
задача, меняющая файлы, делается в своём worktree на своей ветке.**

```bash
git fetch --prune
git worktree add .claude/worktrees/<issue>-<тема> -b worktree-<issue>-<тема> origin/main
cp .env .claude/worktrees/<issue>-<тема>/ 2>/dev/null
cd .claude/worktrees/<issue>-<тема> && uv sync --frozen
# или сразу сессией: claude -w <issue>-<тема>
```

`<issue>` — номер issue (`12-baseline-otsu`). `.claude/worktrees/` в `.gitignore`.

## Задача: issue → ветка → PR → merge

```bash
gh issue edit <N> --add-assignee @me                 # взять задачу (атомарно, видно всем)
# ... тест → код → проверки
git fetch --prune && git merge origin/main           # затем проверки заново
make check                                           # pytest + test-hooks + validate
git add <пути> && git commit -m "<area>: <что закончено>"
git push -u origin HEAD
gh pr create --base main --title "<area>: <что>" --body "Closes #<N>
Критерий: <№ и название>. Проверка: <команда> → <результат>.
Потребители контракта: <список или «не затронуты»>"
```

Затем субагент `reviewer`. Вердикт `merge` →

```bash
git fetch --prune && git merge origin/main && make check && git push   # строго по одному
gh pr merge <PR> --merge --delete-branch                               # merge-коммит, без squash/rebase
git -C <основная копия> pull --ff-only                                 # если там чисто
```

Merge строго по одному: перед каждым — `merge origin/main` и полный прогон. Два
вливания одновременно не делать; если `gh pr merge` отказал из-за нового `main` —
повторить блок.

## Уборка

Ветку или worktree можно удалить, только если **оба** условия выполнены:

1. ветка влита: `git fetch --prune && git rev-list --count origin/main..<ветка>` → `0`;
2. ветка не открыта ни в одном worktree, кроме твоего, а в твоём пусто:
   `git -C <путь> status --porcelain` → пусто. Чужой worktree не трогать никогда.

```bash
git worktree remove .claude/worktrees/<issue>-<тема>   # без --force, из основной копии
git branch -D worktree-<issue>-<тема>
git worktree prune
```

## Перед началом работы

```bash
git fetch --prune && git status -sb
git log --oneline HEAD..origin/main   # что пришло от команды
gh issue list --assignee @me          # мои задачи
```

## Коммиты

`<area>: <что закончено>`, по-русски. `area` = метка issue: `data`, `split`,
`baseline`, `model`, `calibration`, `assets`, `procurement`, `export`, `app`,
`contracts`, `docs`, `infra`. Один коммит — один осмысленный шаг.

## Данные, веса, LFS

- Sen1Floods11 в git не кладётся никогда: только манифест и команды получения.
- Веса и калибраторы — `models/`, через **Git LFS** (`.gitattributes`). На каждой машине
  один раз: `git lfs install`. Файл > 5 МБ вне LFS страж коммита не пропустит.
- Лимиты GitVerse (docs/research/hosting.md): LFS — 100 МБ/файл по HTTP; релизы — 100 МБ
  на файл, 4 ГБ суммарно. Веса держать < 100 МБ.

## GitVerse (сдача)

Репозиторий: https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34
(приватный, у нас push без admin). GitHub — рабочий, GitVerse — зеркало `origin/main`.

```bash
# в .env: GITVERSE_TOKEN=<токен>, GITVERSE_USER=<логин GitVerse>
bash scripts/sync-gitverse.sh            # push origin/main → main и master, LFS-объекты, теги + сверка
bash scripts/sync-gitverse.sh --verify   # только сверка: commit совпадает, LFS в клоне — не указатели
```

Скрипт пушит только `origin/main` (не локальную ветку), токен в URL и config не пишет,
remote `gitverse` добавляет сам. `master` — ветка по умолчанию GitVerse, держится равной
`main`. Пробный LFS-объект (2 МБ) 25.09 прошёл туда и обратно с тем же sha256.

Синхронизация — после каждой вехи и не реже раза в 3 часа с 26.09; последняя — до
27.09 00:00, затем `--verify` и контрольный прогон из чистого клона GitVerse.
**После 01:00 27.09 push нет.**

## Сдача и теги

После `submission-checklist` на commit из `origin/main`:

```bash
git tag -a submit-final origin/main -m "Сдача: <commit>"
git push origin submit-final && bash scripts/sync-gitverse.sh
```

Commit сдачи — в README и `HANDOFF.md`.

## Безопасный откат

| Ситуация | Действие |
|---|---|
| Вернуть свой файл | `git restore <path>` после просмотра diff |
| Отменить влитый коммит | `git revert <hash>` в новой ветке → PR |
| Посмотреть старое состояние | `git switch --detach <tag-or-hash>` |
