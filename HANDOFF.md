# HANDOFF — блокеры и итоги вех

Задачи и занятость — только GitHub Issues (assignee + `area:`), не здесь. Сюда —
блокеры, которые нельзя закрыть внутри репозитория, и итоги по вехам таймлайна.

## Блокеры (ждут человека)

| Дата | Что | Кто нужен |
|---|---|---|
| 2026-09-25 | GitVerse: код = `6fe66f0` (25.09 ~23:30, `--no-verify`), веса не залиты: LFS PUT → 403 на токен chichetin (push:true по API). Дальше — `bash scripts/sync-gitverse.sh --code-only` до получения прав (GIT.md «GitVerse») | токен с записью в LFS / проверка прав у admin hackrus.experts |

Снят: приглашения IR630 и fafakaj приняты — оба в `gh api repos/Chichetin/kosmo-moskva_umstvenno_ustalye_34/collaborators` рядом с Chichetin, Ivana-banana, M1hairu, открытых приглашений нет (`gh api …/invitations` пуст; проверено при вливании PR #60).

## Итоги вех

| Веха | Итог | Commit |
|---|---|---|
| 25.09 GitVerse | `hackrus.experts/kosmo-moskva_umstvenno_ustalye_34`: `scripts/sync-gitverse.sh` (#31), LFS-проба 2 МБ прошла; ветка по умолчанию `lfs-probe` = main (сменить может только admin) | 0ff5f08 |
| 25.09 каркас | каркас, план v2, 35 issues (#2–#36); collaborators добавлены, git-lfs стоит, кейс № 3 | 20ad6e3 |
| 25.09 сквозной путь | 16 PR (#41–#57, ветка `worktree-XX-…` → `main`), см. список ниже: split заморожен, официальный baseline (VH ≤ −23,0 dB, validation IoU 0,580), основной метод U-Net (ансамбль 5 seed, dev IoU 0,584 против baseline 0,391), портфель 10 объектов и чувствительность p/V/q, зоны+A/B/C+ПП 840 (редакция проверена), контракты v1 и валидатор, экспорт: `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → 12/12 файлов. Ещё не реализованы на этой вехе: `calibrate`, `eval-test`, `app` | 56288a4 |
| 25.09 eval-test | PR #58 (Closes #29, D-030): `make eval-test --part <dev\|validation\|test>` — метрики на идентичных валидных пикселях, парный бутстрэп ΔIoU, reliability/ECE/Brier, неопределённость на ошибках (AUROC), JRC-диагностика, перенос по группам чипов, прокси-проверка ущерба. **Test запечатан честно**: `--part test` без `FLOODVALUE_EVAL_TEST_CONFIRM=<sha256>` и при незакоммиченном дереве — отказ (код 3); запуск на test — issue #32, после заморозки. На **validation** (60 чипов, не test): ALL IoU основной 0,600 против baseline 0,580 (ΔIoU CI [−0,066; +0,079]); ECE 0,035, Brier 0,048; неопределённость AUROC 0,890; перенос по группам чипов — основной метод хуже baseline на доле воды 1–30 % (0,121–0,173 против 0,265–0,379), лучше на ≥ 30 % воды (0,767 против 0,710); прокси-проверка EL vs EL_ref — MAE 350 006 руб. на 536 из 600 точек. `pytest tests/eval` 27 passed; `make check` → 334 passed. Перепроверено в PR #30 (`scribe`): `make eval-test` (bare) → отказ с sha256; `--part validation` → 60 чипов, отчёт | 09c52b4 |
| 25.09 calibration | PR #59 (Closes #20, D-029): `make calibrate` — калибратор `p_raw → p` (Platt, выбран по log-loss вне события), порог 0,37 по max IoU на validation; `predict` подключает `models/unet/calibration.json` автоматически. Validation было → стало (сырая p≥0,5 → Platt p≥0,37): IoU 0,600 → **0,685**, F1 0,750 → 0,813, ECE 0,0398 → **0,0100**, Brier 0,0477 → 0,0406, log-loss 0,1693 → 0,1462; парный бутстрэп ΔIoU +0,084 [+0,021; +0,155]. Перенос порога между событиями: Sri-Lanka 0,593 (было 0,526), Nigeria 0,773 (было 0,691). Раскрытое ограничение: на dev (USA, не участвовал в подборе) ECE ухудшается 0,032 → 0,051. Follow-up вне PR: `export` пока берёт метрики main из сырого `unet-metrics.csv` (порог 0,5), калиброванные строки уже есть в `docs/verification/calibration/unet-metrics.csv`. Перепроверено в PR #30 (`scribe`): `make calibrate` реально выполнен (Platt a=1,4647, b=1,5210, порог 0,37); `predict` подхватывает калибровку без правок вызова | 09c52b4 |
| 25.09 validator | PR #61 (Closes #14): валидатор доведён до полного объёма — кольца зон замкнуты и ≥ 4 точек, зона обязана пересекать экстент чипа (ловит перепутанный [lat, lon]), каталог/`run_metadata.pricing` сверяются с `configs/pricing.yaml`; 6 сломанных fixtures дают каждый свой текст ошибки (`test_broken_fixture_gives_its_own_error`). `make check` → 353 passed, 4 skipped; мутаций 7, все убиты | 09c52b4 |

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

Сквозной путь после слияния всех 16 PR (#41–#57) проверен заново в PR #30 (`scribe`,
этот PR): `make all CHIP=Sri-Lanka_713926 BUDGET=5000` → `outputs/Sri-Lanka_713926_5000_56288a4/`,
`validate` 12/12; чистый клон ветки → `git lfs pull; make setup && make check` →
303 passed, 4 skipped, hooks 40/40, validate 12/12 на эталоне, без обращения к
Sen1Floods11. Пока этот PR готовился, в `main` последовательно влились PR #58
(`eval-test`), #59 (`calibration`) и #61 (`validator`) — worktree трижды смёржен с
`origin/main` (`56288a4` → `09c52b4` → `0c06c33`), README и `docs/tz-compliance.md`
пересверены под каждое новое состояние (все команды в README перепроверены реальными
запусками после финального слияния). Ещё не реализован **только** `app` (#11
интерфейс). `eval-test` реализован и запечатан правильно, `calibrate` реально
выполнен и подключён в `predict`; сам прогон `eval-test` на **test** — ещё нет, ждёт
заморозки функций (#32, D-022).
