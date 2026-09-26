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

Процесс слушает только `127.0.0.1:8000`. Заголовкам `X-Forwarded-*` он доверяет только от
`127.0.0.1`, то есть от Caddy. Так ограничение частоты видит реальный адрес клиента, а cookie
получает `Secure`.

**Защита от ошибки конфигурации.** С `FLOODVALUE_SITE_PUBLIC=1` или при `--host`, отличном от
loopback, сайт без `FLOODVALUE_MEDIA_ANALYST_TOKEN` не запускается: иначе модерацию мог бы
выполнять кто угодно.

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
  показывает, где есть заявка, но не дверь очевидца. События, в которые вошло обращение
  очевидца, отдаются так же: точка округлена, `title` — «Сообщение очевидца», `place_label` пуст,
  `geo_precision=coarsened`. Аналитик и в этом режиме видит всё точно.

Слой — повод проверить территорию. На p, маску, ущерб и план съёмки он не влияет
(`docs/features/media-monitoring.md`, раздел И).
