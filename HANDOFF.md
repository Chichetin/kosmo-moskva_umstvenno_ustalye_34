# HANDOFF — блокеры и итоги вех

Задачи и занятость — только GitHub Issues (assignee + `area:`), не здесь. Сюда —
блокеры, которые нельзя закрыть внутри репозитория, и итоги по вехам таймлайна.

## Блокеры (ждут человека)

Нет открытых блокеров на 2026-09-25 19:50: приглашения IR630 и fafakaj приняты —
оба в `gh api repos/Chichetin/kosmo-moskva_umstvenno_ustalye_34/collaborators` рядом с
Chichetin, Ivana-banana, M1hairu (проверено в этом PR).

## Итоги вех

| Веха | Итог | Commit |
|---|---|---|
| 25.09 GitVerse | `hackrus.experts/kosmo-moskva_umstvenno_ustalye_34`: `scripts/sync-gitverse.sh` (#31), LFS-проба 2 МБ прошла; ветка по умолчанию `lfs-probe` = main (сменить может только admin) | 0ff5f08 |
| 25.09 каркас | каркас, план v2, 35 issues (#2–#36); collaborators добавлены, git-lfs стоит, кейс № 3 | 20ad6e3 |
| 25.09 сквозной путь | 16 PR (#41–#57, ветка `worktree-XX-…` → `main`), см. список ниже: split заморожен, официальный baseline (VH ≤ −23,0 dB, validation IoU 0,580), основной метод U-Net (ансамбль 5 seed, dev IoU 0,584 против baseline 0,391), портфель 10 объектов и чувствительность p/V/q, зоны+A/B/C+ПП 840 (редакция проверена), контракты v1 и валидатор, экспорт: `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → 12/12 файлов. Блокеры этой вехи: `calibrate`, `eval-test`, `app` ещё не реализованы (README, `docs/tz-compliance.md`) | 56288a4 |

### PR #41–#57 (25.09, сквозной путь): что влито

Числа — дословно из описаний PR (`gh pr view <N> --json body`), не пересчитаны заново.
№57 идёт после №56 по времени слияния (16:29 и 16:31), порядок в списке — по номеру.

| PR | Тема (Closes) | Ключевые числа из описания |
|---|---|---|
| #41 | split: split v1 по событиям (D-016) · Closes #4 | train 237 / dev 69 / validation 60 / test 75 / excluded 5; sha256 `0098346e0f87…`; `pytest` 17 passed (tests/split 15); мутации 6/6 |
| #42 | procurement: цена ПП 840 и площадь контура · Closes #6 | `pytest tests/procurement` 46 passed: 1703.46 / 946.37 / 567.82 руб.; 0,9996 км² отклонён; 90 дн. → Т=1, 91 → 0,6; мутаций 20, выжило 0 |
| #43 | assets: портфель 10 объектов, p из растра, EL, rank · Closes #7 | `pytest tests/assets` 17 passed (весь `pytest` 19); мутаций 12, убито 12 |
| #44 | data: make data — QC 446 чипов, sha256, S1-часть source_manifest · Closes #3 | 446 чипов, скачано 0 (уже на диске); QC формата 446/446, без валидных пикселей 6; `pytest` 32 passed (tests/data 30); мутаций 11 (1 эквивалентная) |
| #45 | docs: D-022 — стоп-код 26.09 23:30 | перенос стоп-кода с 27.09 01:00 на 26.09 23:30; заморозка функций 26.09 20:30 (было 22:00); прогон из чистого клона 26.09 22:30 |
| #46 | docs: промпт для новых участников + правка sync-gitverse | `docs/onboarding-prompt.md`; починка проверки LFS-указателя (pipefail); `bash -n scripts/sync-gitverse.sh`, `pytest` — без счётчика в теле PR |
| #47 | docs: вопросы к трекерам 25.09 (Q-09…Q-20) · Closes #13 | 12 новых вопросов Q-09…Q-20 в `docs/questions-for-organizers.md`; `make check`: `pytest` 77 passed, 4 skipped, hooks 40/40 (validate — ещё не реализован на момент PR, не относится к нему) |
| #48 | contracts: контракты v1 — схемы 12 файлов, run_metadata, run.yaml, валидатор · Closes #2 | `make validate OUT=tests/fixtures/good` → 12/12; `pytest tests/contracts` 75 passed; весь `pytest` 152 passed, 4 skipped; мутаций валидатора 11, все убиты после 2 новых тестов |
| #49 | baseline: порог VH по train/dev, метрики по событиям, p-растр на сетке S1 · Closes #5 | train IoU 0,378 / dev 0,391 / validation 0,580; VH ≤ −23,0 dB, s = 2,1315; `pytest` tests/baseline 25 passed |
| #50 | procurement: зоны v0 и стратегии A/B/C · Closes #8 | пример на бюджете 5000: A 0 руб., B 42 465,93 руб. (вне бюджета), C 4 905,96 руб. (Z-A02, Z-A10); чувствительность C к бюджету 0→0→1→2→4→8 зон; мутаций 8, убито 8 |
| #51 | infra: GPU-окружение обучения, models/+LFS, README-каркас, check без validate · Closes #12 | `make check` → 0 (pytest 77 passed / 4 skipped, hooks 40/40); `setup_gpu_env.sh` дважды подряд → 0 ошибок, `torch 2.11.0+cu128`, `cuda available: True` |
| #52 | baseline: замок test в read_chip до открытия файлов (follow-up к #49) | до правки без данных: 1 failed, 214 passed, 4 skipped; после: `make check` → 215 passed, 4 skipped, validate 12/12 (эталон) |
| #54 | model: основной метод B — компактный U-Net по S1, ансамбль 5 seed · Closes #17 | `pytest tests/model` 23 passed; CPU predict `Sri-Lanka_713926` → 0,46 с, 262 144 валидных пикселей; dev IoU 0,584 vs baseline 0,391; validation ALL 0,600 vs 0,580 (CI разницы [−0,070; +0,078]); Sri-Lanka 0,526 vs 0,585 (U-Net хуже); мутаций 11, выжила 1 → закрыта |
| #55 | assets: чувствительность EL к p, V, q по отдельности · Closes #9 | `pytest tests/assets` 39 passed; `make check` (pytest 264 passed, 4 skipped) зелёный; мутаций 7, все в итоге убиты |
| #56 | export: комплект outputs/<run_id>, run_metadata, source_manifest, metrics, make all · Closes #10 | `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → validate 11/12 на момент PR (без `assets_sensitivity.csv`, ожидаемо до #55); `pytest` 259 passed, в т.ч. tests/export 12; мутаций 6, выжила 1 → закрыта |
| #57 | procurement: ПП 840 — редакция проверена, источники и допущения зафиксированы · Closes #23 | `pytest` 247 passed; `validate` 12/12; редакция ПП № 840 от 29.06.2019 в ред. № 1297 от 27.08.2025, `legal_checked_at=2026-09-25`, источники government.ru + base.garant.ru |

Сквозной путь после слияния всех 16 PR проверен заново в PR #30 (`scribe`, этот PR):
`make all CHIP=Sri-Lanka_713926 BUDGET=5000` → `outputs/Sri-Lanka_713926_5000_56288a4/`,
`validate` 12/12; чистый клон ветки → `git lfs pull; make setup && make check` →
303 passed, 4 skipped, hooks 40/40, validate 12/12 на эталоне, без обращения к
Sen1Floods11. Ещё не реализовано после этой вехи: `calibrate` (#20), `eval-test`
(журнал `docs/verification/test-log.md` пуст), `app` (#11 интерфейс).
