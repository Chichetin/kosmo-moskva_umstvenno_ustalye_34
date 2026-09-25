# HANDOFF — блокеры и итоги вех

Задачи и занятость — только GitHub Issues (assignee + `area:`), не здесь. Сюда —
блокеры, которые нельзя закрыть внутри репозитория, и итоги по вехам таймлайна.

## Блокеры (ждут человека)

| Дата | Что | Кто нужен |
|---|---|---|
| 2026-09-25 | Приглашения в репозиторий не приняты: IR630, fafakaj | эти участники |
| 2026-09-25 | GitVerse: код = `6fe66f0` (25.09 ~23:30, `--no-verify`), веса не залиты: LFS PUT → 403 на токен chichetin (push:true по API). Дальше — `bash scripts/sync-gitverse.sh --code-only` до получения прав (GIT.md «GitVerse») | токен с записью в LFS / проверка прав у admin hackrus.experts |

## Итоги вех

| Веха | Итог | Commit |
|---|---|---|
| 25.09 GitVerse | `hackrus.experts/kosmo-moskva_umstvenno_ustalye_34`: `scripts/sync-gitverse.sh` (#31), LFS-проба 2 МБ прошла; ветка по умолчанию `lfs-probe` = main (сменить может только admin) | 0ff5f08 |
| 25.09 каркас | каркас, план v2, 35 issues (#2–#36); collaborators добавлены, git-lfs стоит, кейс № 3 | 20ad6e3 |
