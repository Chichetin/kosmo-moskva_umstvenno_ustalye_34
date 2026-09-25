---
name: orchestrate
description: Личный протокол сессии участника — взять свои issues, завести worktree, запустить параллельных builder, провести каждый PR через reviewer → evaluator → merge, держать main зелёным. Общего оркестратора нет. Использовать командой /orchestrate в начале рабочего окна и когда просят «возьми задачи», «доведи до main».
---

# Протокол сессии участника

Ты — сессия одного из пяти равноправных участников. Координатора нет: задачи —
GitHub Issues, занятость — assignee, стыковка — контракты. Основная копия на `main` —
только чтение, `pull --ff-only` и запуск агентов; правки — в worktree.

Инварианты:

1. `main` зелёный (`make check`); сломал — чинишь раньше всего.
2. Одна `area:` — одна issue в работе на команду: не бери issue с `area:`, на которой
   есть assignee в работе (у другого человека).
3. Контракт, split, `configs/`, зависимости — отдельный PR с перечнем потребителей.
4. Merge строго по одному, после `merge origin/main` и полного прогона.
5. Нет команды и вывода — нет статуса.

## 1. Синхронизация

```bash
git fetch --prune && git status -sb && git pull --ff-only
gh issue list --assignee @me
gh issue list --search "no:assignee" --label "<интересная area>"
gh pr list
```

Вехи и зависимости — `docs/plans/` (последний план). Блокеры — `HANDOFF.md`.

## 2. Выбор issues

- Ближайшая веха таймлайна и критический путь раньше ширины.
- Берёшь, если «Зависит от #…» уже в `main` (иначе — на fixtures с пометкой в PR).
- Параллельные `builder` — на независимые issues **одной своей** `area:`-группы или на
  свободные `area:`; каталоги не пересекаются.
- Взял — `gh issue edit <N> --add-assignee @me` сразу, до worktree.

## 3. Worktree (делаешь ты, не builder)

```bash
git worktree add .claude/worktrees/<N>-<тема> -b worktree-<N>-<тема> origin/main
cp .env .claude/worktrees/<N>-<тема>/ 2>/dev/null
(cd .claude/worktrees/<N>-<тема> && uv sync --frozen)
```

`isolation: worktree` у Agent не использовать: свои имена, без `.env`.

## 4. Бриф builder

```
Issue: #<N> — <название>; area:<…>; коммит `<area>: …`
Worktree: <абсолютный путь>; ветка: worktree-<N>-<тема>
Разрешённые пути: <каталоги area из docs/project-structure.md + tests/ + docs-строки>
Запрещённые: src/floodvalue/contracts/, configs/, splits/, pyproject.toml, uv.lock (если issue не про них)
Контракты: <какие файлы/функции потребляет и производит>
Критерий приёмки: <№ критерия, команда проверки, ожидаемый вывод>
Документы: D-<следующий номер> (tail docs/decisions.md в origin/main; параллельным — разные номера)
```

`Agent(subagent_type: "builder", prompt: <бриф>, run_in_background: true)`. Пока он
работает — не трогаешь его файлы.

## 5. Ревью и измерение

- Builder сам вызывает `reviewer`; если отчёт пришёл без вердикта — вызови сам.
- Изменены метод, порог, калибровка, экономика → `evaluator` (validation «было → стало»)
  до merge. Таблицу — в PR.
- `доработать` → `SendMessage` тому же builder; `отклонить` дважды → `jury` (арбитр).

## 6. Merge

Builder вливает сам после `merge`. Проверь: `gh pr view <PR> --json state,mergeCommit`,
затем `git pull --ff-only` в основной копии и `make check`. Красный `main` — сразу
issue/фикс.

## 7. Вехи

Перед каждой вехой таймлайна (25.09 21:00, 26.09 16:30, 26.09 20:30, 26.09 22:30; стоп-код 26.09 23:30) один
из участников запускает `jury`; его «топ-5» — новые issues. Перед 26.09 22:30 —
скилл `submission-checklist`. Итоги вехи — 3–5 строк в `HANDOFF.md` через PR.

## К человеку — только

Доступы (GitHub, GitVerse, Colab), физические действия, сокращение обязательного
объёма. Вопросы организаторам — строкой в `docs/questions-for-organizers.md`, работа
идёт дальше по рекомендуемой трактовке.

## Отчёт в конце окна

Влито (PR → merge-commit, числа проверок); в работе (ветка, стадия); застряло и
почему; что брать следующим; что ждёт человека.
