# Один домен: интерфейс оператора, форма очевидца, сообщения СМИ (#174)

Все части работают **на одном домене одним процессом**: `scripts/serve_site.py` направляет
запрос по пути в одно из двух FastAPI-приложений. Сбор источников идёт отдельным фоновым
процессом. Решение — D-054.

| Путь | Что | Кто видит | Приложение |
|---|---|---|---|
| `/` | интерфейс оператора (React), `/api/scenes`, `/api/session`, `/api/plan`, `/api/layers/*`, `/api/download` | пока публично | `floodvalue.app.run` |
| `/report`, `/report/status` | форма очевидца и статус обращения по токену | публично | `floodvalue.media.web` |
| `/media/analyst` | очередь модерации, события, источники | вход по токену аналитика | `floodvalue.media.web` |
| `/api/media/map` | слой «Сообщения» для карты оператора | см. ниже | `floodvalue.media.web` |
| `/api/media/*` (остальное) | API аналитика | вход по токену аналитика | `floodvalue.media.web` |

Почему диспетчер лежит в `scripts/`, а не в пакете: ни одна стадия FloodValue не импортирует
`floodvalue.media` (`tests/media/test_media_firewall.py`). Сообщения не могут повлиять на p, маску,
порог, EL и A/B/C. Приложения соединены только путём URL.

Один адрес вместо поддоменов:

- не нужен CORS;
- cookie входа аналитика общая с интерфейсом оператора;
- один TLS-сертификат и одна DNS-запись.

## Локально

```bash
make site                               # http://127.0.0.1:8000  (OUT=outputs/<run_id> — конкретный комплект)
uv run --frozen python -m floodvalue.media demo      # помеченные ДЕМО-сообщения, по желанию
uv run --frozen python -m floodvalue.media worker    # фоновый сбор, по желанию
```

Без токена аналитика всё открыто только с этой машины (loopback). Разработка фронта:
`cd web && npm run dev`, Vite проксирует `/api`, `/report` и `/media` на `:8000`.

`make app` работает как прежде: только интерфейс оператора, без СМИ.

## Сервер

1. Клон в `/opt/floodvalue`, пользователь `floodvalue`, `make setup`, `git lfs pull`
   (веса модели) и готовый комплект в `outputs/`.
2. `deploy/floodvalue.env.example` скопировать в `/etc/floodvalue.env` (права 600) и задать
   `FLOODVALUE_MEDIA_ANALYST_TOKEN`.
3. `deploy/*.service` скопировать в `/etc/systemd/system/` и выполнить
   `systemctl enable --now floodvalue-site floodvalue-media-worker`.
4. `deploy/Caddyfile`: заменить `flood.example.org` на домен (A-запись должна указывать на
   сервер) и выполнить `caddy reload`. Сертификат Caddy получает сам.

Юниты systemd разрешают запись только в `/opt/floodvalue/outputs` (`ReadWritePaths`). Если
`FLOODVALUE_OUTPUT_DIR` или `FLOODVALUE_MEDIA_DIR` указывают в другое место, добавьте этот путь
в `ReadWritePaths` обоих юнитов.

Процесс слушает только `127.0.0.1:8000`. Заголовкам `X-Forwarded-*` он доверяет только от
`127.0.0.1`, то есть от Caddy. Так ограничение частоты видит реальный адрес клиента, а cookie
получает `Secure`.

**Защита от ошибки конфигурации.** С `FLOODVALUE_SITE_PUBLIC=1` или при `--host`, отличном от
loopback, сайт без `FLOODVALUE_MEDIA_ANALYST_TOKEN` не запускается: иначе модерацию мог бы
выполнять кто угодно.

## Отдельный домен для формы очевидца (D-062)

Второй блок `deploy/Caddyfile` (`report.example.org`) — короткий адрес только для формы: тот же
процесс, но через этот домен открыты лишь `GET /report`, `/report/status`, `/media/static/*`,
`/api/media/report-config` и `POST /api/media/reports`, `/api/media/reports/status`; `/`
перенаправляет на форму, всё остальное — 404. Интерфейс оператора, страница и API аналитика и
слой `/api/media/map` через него недоступны. Первый домен работает как прежде. Второй домен не
нужен — блок удалить.

## Сервер без root (D-062)

Если на сервере нет root, а Caddy уже работает от обычного пользователя:

1. Код — в `~/apps/floodvalue`. Приватный репозиторий без учётных данных на сервере: локально
   `git bundle create floodvalue.bundle <ветка>`, на сервер — rsync, там
   `GIT_LFS_SKIP_SMUDGE=1 git clone -b <ветка> floodvalue.bundle floodvalue`; объекты LFS из
   `models/` — rsync в `.git/lfs/objects/`, затем `git lfs install --local && git lfs checkout`
   (`git status` должен быть чистым: иначе `run_metadata.json` получит `commit_dirty`).
   Обновление — `git bundle create upd.bundle <старый commit>..<ветка>`, `git fetch upd.bundle
   <ветка> && git merge --ff-only FETCH_HEAD`. Если менялись `models/`: сначала rsync новых
   объектов LFS, `GIT_LFS_SKIP_SMUDGE=1` перед `merge`, затем `git lfs checkout`.
2. `uv sync --frozen`. Окружение — `~/.config/floodvalue/floodvalue.env` (права 600) из
   `deploy/floodvalue.env.example`; `FLOODVALUE_OUTPUT_DIR` и `FLOODVALUE_DATA_DIR` — абсолютные
   пути в домашнем каталоге (`EnvironmentFile` не раскрывает `~` и `%h`).
3. Юниты `deploy/user/*.service` → `~/.config/systemd/user/`, `systemctl --user enable --now
   floodvalue-site floodvalue-media-worker`; нужен `loginctl enable-linger`.
4. Блоки из `deploy/Caddyfile` — в Caddyfile пользователя (или отдельным файлом через `import`),
   `caddy validate`, затем `caddy reload`. Порт в `reverse_proxy` = `FLOODVALUE_APP_PORT`.

## Данные на публичном сервере

**`make data` и `make all` на публичном сервере не запускать**: `data` докачивает недостающие
слои всего набора, включая `LabelHand` test. Нужны только:

- `raw/HandLabeled/S1Hand/*.tif` сцен каталога — train, dev, validation; из test — только
  демо-чип после `eval-test` (D-033);
- `raw/Sen1Floods11_Metadata.geojson` — слой «вода до события» (запасная копия есть в
  `docs/sen1floods11/`);
- `interim/source_manifest_s1.json` — целиком или без строк test, **кроме демо-чипа**: из него
  берутся дата снимка для закупки и экспорта и превью S1; строки — имена файлов, хеши и даты,
  разметки в них нет, события test каталог сам пропускает;
- `interim/qc.csv` — без строк test (в нём доли воды по разметке).

`LabelHand`, `S2Hand`, `JRCWaterHand` интерфейсу и этапам анализа не нужны (`metrics.csv` берётся
из `docs/verification`). Готовый комплект сцены — те же этапы, что `make all`, без `data`:

```bash
P() { .venv/bin/python -m "$@"; }; C=<chip_id>; B=5000
P floodvalue predict --chip $C --budget $B
P floodvalue.baseline.predict --chip $C --mask-only
P floodvalue prewater --chip $C
P floodvalue assets --chip $C --budget $B
P floodvalue procure --chip $C --budget $B
P floodvalue export --chip $C --budget $B
P floodvalue validate
```

`FLOODVALUE_APP_ALLOW_TEST` на публичном сервере не выставлять.

## Слой «Сообщения» для карты оператора: `GET /api/media/map`

```json
{
  "events":  {"type": "FeatureCollection", "features": [ ... ]},
  "reports": {"type": "FeatureCollection", "features": [ ... ]},
  "exact_locations": true
}
```

- **`events`**: события из СМИ и принятые обращения очевидцев. Поля те же, что у
  `/api/media/events`: `event_id`, `title`, `report_type`, `place_label`, `geo_precision`,
  `event_time_start`/`end`, `review_state`, `verification_result`,
  `independent_origin_count`, `source_kinds`, `is_demo`, `is_replay`. Исторические события
  скрыты. Карточка события — `GET /api/media/events/{id}` (нужен вход аналитика).
- **`reports`**: обращения очевидцев, ещё не прошедшие модерацию, у которых есть координаты.
  Point, `report_id`, `category`/`category_label`, `location_precision`, `observed_at`,
  `submitted_at`, `is_demo`. **Текста обращения, фото и токенов в ответе нет.**
  Принятое обращение переходит в `events`, отклонённые и спам не показываются.
- `?include_demo=false` скрывает ДЕМО.

Доступ:

- по умолчанию только аналитик: `Authorization: Bearer <токен>`, cookie после `/media/login`
  или loopback без токена;
- при `FLOODVALUE_MEDIA_OPERATOR_MAP_PUBLIC=1` (интерфейс оператора пока публичный) слой
  открыт всем. Тогда точки обращений округляются до 0,01° (около 1 км,
  `location_precision=coarsened`, `exact_locations=false`), а время — до часа: публичная карта
  показывает, где есть заявка, но не дверь очевидца. У событий в этом режиме тоже округлены
  точка и время, а `title` и `place_label` не отдаются: заголовок заменён на «Сообщение СМИ»,
  «Сообщение очевидца» или «Сообщения СМИ и очевидцев», `geo_precision=coarsened`. Текст и
  место события могли прийти от очевидца даже тогда, когда сейчас в событии остались только
  новости (обращение принято, потом разделено). Аналитик и в этом режиме видит всё точно.

Слой — повод проверить территорию. На p, маску, ущерб и план съёмки он не влияет
(`docs/features/media-monitoring.md`, раздел И).
