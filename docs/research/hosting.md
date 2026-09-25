# Хостинг: GitVerse и GitHub (разведка 2026-09-25)

Источник — документация GitVerse и GitHub (субагент, WebFetch). Решение — D-002.

| Возможность | GitVerse | GitHub Free |
|---|---|---|
| Issues, исполнитель, метки | да — https://gitverse.ru/docs/collaborative/issues_2/quick-start | да |
| «Closes #N» закрывает issue при merge | не подтверждено документацией (есть только связывание ветки/PR с задачей) — https://gitverse.ru/docs/collaborative/issues_2/manuals/task-development | да |
| PR/MR и ревью | да — https://gitverse.ru/docs/collaborative/repositories/guides/pull-requests | да |
| Защита веток | да — https://gitverse.ru/docs/collaborative/repositories/guides/pull-requests/branch-protection ; тарифные ограничения не указаны | **только публичные репо**; приватные — Pro/Team |
| Git LFS | да; 100 МБ/файл по HTTP, до 1 ГБ по SSH; 2 ГБ push/мес vs 5 ГБ хранилища (источники расходятся) — https://gitverse.ru/docs/limits-lfs-cicd/limits-lfs | да |
| Релизы | да; 100 МБ/файл, 5 файлов и 500 МБ за загрузку, 4 ГБ суммарно — https://gitverse.ru/docs/limits-lfs-cicd/limits-releases | да |
| API | REST `https://api.gitverse.ru`, Bearer-токен, свой MIME `application/vnd.gitverse.object+json;version=1`; issues/PR создаются — https://gitverse.ru/docs/developers/public-api | REST + `gh` |
| Зеркало | pull-зеркало из GitHub нативно (раз в 8 ч); push-зеркало — только своим CI-workflow — https://gitverse.ru/docs/collaborative/repositories/guides/repository-workflow/mirroring-uc | — |
| CI | есть, синтаксис как у Actions — https://gitverse.ru/docs/cicd/introduction | Actions |

Выводы для нас: (1) pull-зеркало GitVerse раз в 8 часов слишком редкое для стоп-кода —
синхронизируем push-скриптом со своей машины; (2) LFS-объекты надо пушить в GitVerse
явно (`git lfs push --all gitverse`) и сверять, иначе в GitVerse останутся указатели;
(3) веса держать < 100 МБ на файл.
