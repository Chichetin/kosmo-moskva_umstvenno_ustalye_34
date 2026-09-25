# Контрольный прогон из чистого клона (#34, критерий 11)

Как жюри: пустой каталог, клон, `uv sync --frozen`, `pytest`, `make data` с нуля,
`make all`. Демо-чип до `eval-test` — `Sri-Lanka_713926` (validation, D-010); другой чип —
`Nigeria_1095404` (validation, не train, не демо). Test-чипы не читались.

## Прогон 3 — 26.09 00:30, GitVerse @ `6d8e99d`, путь жюри без обходов — ✅ (действующий статус)

GitVerse `main`, `master` и `lfs-probe` = `6d8e99d` (синхронизация с загрузкой LFS через PUT с
Content-Length, #98; `--verify` прошёл). Обычный клон с включённым git lfs, данные из `make data` с нуля (прогон 1b).

```bash
git clone https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34.git j2 && cd j2
# HEAD=6d8e99d91b75…, ветка по умолчанию lfs-probe, 265 файлов
git lfs ls-files -s   # 6 файлов, все «*»: models/pixel/model.joblib (488 KB), member_0..4.pt (7.8 MB)
sha256sum models/unet/member_*.pt models/pixel/model.joblib   # 6c2df137…, e077be95…, 8ff688ab…, 80745de0…, 6aa23270…, 6672fa09… = OID LFS
uv sync --frozen && uv run --frozen pytest -o addopts="" -q   # 612 passed in 176.74s
```

| Шаг | Команда | Результат |
|---|---|---|
| Сквозной путь | `make all` | A 0,00; B 42 465,93 руб. (`в бюджете False`); C 4 905,96 руб., `Z-A02`, `Z-A10`; `validate` 12/12 |
| Смена бюджета | `make all BUDGET=20000` | C 15 939,51 руб., 6 зон (`E-1`, `E-2`, `Z-A03`, `Z-A06`, `Z-A07`, `U-3`); `validate` 12/12 |
| p/маска/порог не меняются | `sha256sum …/flood_{probability,mask}.tif`, `threshold` в `run_metadata.json` (5000 и 20000) | `b767ad16cf4c3c88…`, `f38ec062c6736074…`, 0,37 в обоих — те же, что в прогонах 0–2 |
| Другой чип | `make all CHIP=Nigeria_1095404` | `validate` 12/12 |
| Интерфейс | `make app` | ❌ `stage 'app' not implemented yet` — #83 |

План C при 20 000 руб. отличается от прогона 2 (19 272,96 руб., 8 зон): между `6fe66f0` и `6d8e99d`
менялся `src/floodvalue/procurement/` (#99); p, маска и порог те же. На момент записи GitVerse
`6d8e99d` отстаёт от `origin/main` (в том числе `predict --s1` #113, `src/floodvalue/app/`) — после
следующей синхронизации прогон повторяется.

## Прогон 2 — 25.09 23:46, GitVerse @ `6fe66f0`, путь жюри без обходов — ✅

Веса дозалиты в LFS GitVerse (PUT с Content-Length вместо chunked). Обычный клон, git lfs
включён, без `GIT_LFS_SKIP_SMUDGE` и без remote GitHub.

```bash
git clone https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34.git j && cd j
# HEAD=6fe66f023e9b…, ветка по умолчанию lfs-probe, 228 файлов, origin = gitverse.ru
git lfs ls-files -s        # member_0..4.pt, все «*» (содержимое), 7.8 MB
sha256sum models/unet/member_*.pt   # 6c2df137…, e077be95…, 8ff688ab…, 80745de0…, 6aa23270… = OID LFS
uv sync --frozen && uv run --frozen pytest -o addopts="" -q   # 473 passed in 88.02s
export FLOODVALUE_DATA_DIR=<каталог, заполненный make data с нуля, см. прогон 1b>
```

| Шаг | Команда | Результат |
|---|---|---|
| Сквозной путь | `make all` | A 0,00; B 42 465,93 руб. (`в бюджете False`); C 4 905,96 руб., `Z-A02`, `Z-A10`; `validate` 12/12 |
| Смена бюджета | `make all BUDGET=20000` | C 19 272,96 руб., 8 зон; `validate` 12/12 |
| p/маска/порог не меняются | `sha256sum …/flood_{probability,mask}.tif` и `threshold` в `run_metadata.json` (5000 и 20000) | `b767ad16cf4c3c88…`, `f38ec062c6736074…` и 0,37 в обоих — те же, что в прогонах 0 и 1b |
| Другой чип | `make all CHIP=Nigeria_1095404` | `validate` 12/12 |
| Интерфейс | `make app` | ❌ `stage 'app' not implemented yet` — #83 |

На момент записи GitVerse отстаёт от `origin/main` на #94 (`scripts/`, `GIT.md`, `HANDOFF.md`)
и #89 (очерёдность осмотра D-035: `src/floodvalue/assets/`, `export/assemble.py`) — проверенный
клон их не содержит; после синхронизации прогон повторяется.

## Прогон 1 — 25.09 ~23:40, GitVerse @ `6fe66f0`

### 1a. `git clone` GitVerse с установленным git lfs (путь жюри по README) — ❌

```bash
git clone https://gitverse.ru/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34.git r3
# exit 128
# Downloading models/unet/member_0.pt (7.8 MB)
# Error downloading object: models/unet/member_0.pt (6c2df13): Smudge error: ... [404] Not Found
# fatal: models/unet/member_0.pt: smudge filter lfs failed
# warning: Клонирование прошло успешно, но во время переключения на состояние произошла ошибка.
```

Ветка по умолчанию — `lfs-probe` = `6fe66f0`. Checkout прерывается: `git ls-files` → 0,
рабочее дерево пустое. В LFS GitVerse есть только `member_4.pt`, у `member_0..3.pt`
ответ 404. Причина: LFS PUT получает 403 на текущий токен, поэтому код ушёл с
`--no-verify` без весов (`HANDOFF.md`, `GIT.md` «GitVerse»). Пока веса не были залиты,
клон GitVerse у жюри не работал. Снято в прогоне 2.

### 1b. Паритет кода: клон GitVerse без smudge, веса из GitHub LFS — ✅

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <gitverse> p && cd p       # HEAD=6fe66f0…, 228 файлов
git remote add github https://github.com/Chichetin/kosmo-moskva_umstvenno_ustalye_34.git
git lfs pull github && git lfs ls-files -s                  # 5 × member_*.pt 7.8 MB, «*»
sha256sum models/unet/member_*.pt                           # 6c2df137…, e077be95…, 8ff688ab…, 80745de0…, 6aa23270… = OID LFS
uv sync --frozen && uv run --frozen pytest -o addopts="" -q # 473 passed in 103.31s
export FLOODVALUE_DATA_DIR=<пустой каталог, заполнен make data ниже>
```

| Шаг | Команда | Результат |
|---|---|---|
| Данные с нуля | `make data` в пустой `FLOODVALUE_DATA_DIR` (код `origin/main` @ `d9bcf5b`, модуль `data` в `6fe66f0` не менялся) | скачано 1343 файла, 704 МБ, 31 мин 55 с; QC 446/446; без валидных пикселей 6 (5 Ghana + `Paraguay_34417`); test — только заголовки и хеши |
| Сквозной путь | `make all` (чип из `configs/run.yaml`, бюджет 5000) | A 0,00 руб.; B 42 465,93 руб., `в бюджете False`; C 4 905,96 руб., зоны `Z-A02`, `Z-A10`; `validate` 12/12 |
| Смена бюджета | `make all BUDGET=20000` | C 19 272,96 руб., 8 зон; `validate` 12/12 |
| p/маска/порог не меняются | `sha256sum …/flood_{probability,mask}.tif` у комплектов 5000 и 20000 | `b767ad16cf4c3c88…` и `f38ec062c6736074…` в обоих; `threshold` 0,37 в обоих `run_metadata.json` |
| Другой чип | `make all CHIP=Nigeria_1095404` | `validate` 12/12 |
| Интерфейс | `make app` | ❌ `stage 'app' not implemented yet` — #83 в работе |

## Прогон 0 — 25.09 ~23:00, GitHub @ `d9bcf5b` (до синхронизации)

Клон GitHub, `git lfs pull` (5 весов по 7,8 МБ), `uv sync --frozen`, `pytest` → 453 passed,
4 skipped. `make all`, данные из основной копии: `validate` 12/12, те же A/B/C.
`BUDGET=20000`: те же хеши p и маски, порог 0,37. `Nigeria_1095404` → 12/12. Хеши
растров Sri-Lanka совпадают с прогоном 1b: на данных, скачанных с нуля, результат тот же.

## Что осталось

- После каждой синхронизации — `FLOODVALUE_REQUIRE_WEIGHTS=1 bash scripts/sync-gitverse.sh --verify`
  и `make all` в клоне.
- Ветка по умолчанию GitVerse — `lfs-probe` (сменить может только admin): жюри без `-b`
  клонирует её, поэтому перед сдачей `lfs-probe` = commit сдачи (скрипт держит её равной `main`).
- `make app` — после #83.
- Финальный прогон — на commit сдачи, после `eval-test` на `Spain_7370579`, до 27.09 00:00.
