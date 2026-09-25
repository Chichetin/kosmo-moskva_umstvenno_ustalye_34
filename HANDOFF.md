# HANDOFF — блокеры и итоги вех

Задачи и занятость — только GitHub Issues (assignee + `area:`), не здесь. Сюда —
блокеры, которые нельзя закрыть внутри репозитория, и итоги по вехам таймлайна.

## Блокеры (ждут человека)

| Дата | Что | Кто нужен |
|---|---|---|
| 2026-09-25 | Приглашения в репозиторий не приняты: IR630, fafakaj | эти участники |

## Итоги вех

| Веха | Итог | Commit |
|---|---|---|
| 25.09 GitVerse LFS | Снят блокер «LFS PUT → 403»: причина не права токена — git-lfs шлёт тело chunked (так велит `upload.header` GitVerse), GitVerse отвечает 403. Веса (5 × U-Net) залиты в LFS GitVerse 25.09 ~23:50 PUT с Content-Length; sync использует `scripts/gitverse_lfs_push.py` (GIT.md «GitVerse»). Код GitVerse = `6fe66f0`; сверка чистого клона (`bash scripts/sync-gitverse.sh --verify`) — после восстановления сети | 6fe66f0 |
| 25.09 GitVerse | `hackrus.experts/kosmo-moskva_umstvenno_ustalye_34`: `scripts/sync-gitverse.sh` (#31), LFS-проба 2 МБ прошла; ветка по умолчанию `lfs-probe` = main (сменить может только admin) | 0ff5f08 |
| 25.09 каркас | каркас, план v2, 35 issues (#2–#36); collaborators добавлены, git-lfs стоит, кейс № 3 | 20ad6e3 |
