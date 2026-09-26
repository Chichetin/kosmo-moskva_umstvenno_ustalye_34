# Мониторинг сообщений и форма очевидцев — проверка 26.09.2026 (#168)

Живые запросы к внешним источникам выполнены вручную с машины разработчика 26.09.2026
(08:00–08:40 UTC). Тесты в CI сеть не используют. Всё ниже — вывод команд. Данные
лежали во временном `FLOODVALUE_MEDIA_DIR` вне репозитория.

## 1. Автотесты

```bash
uv run pytest tests/media -q -p no:warnings     # 107 passed (после замечаний reviewer #171)
uv run ruff check src/floodvalue/media tests/media   # All checks passed!
```

Покрыто (по разделу 15 промпта): отправка без фото и с фото; защита от повторной отправки
(тот же ключ → то же обращение); невалидные координаты, время, вложения (подделка под JPEG,
огромные размеры); место без координат; доступ к чужому обращению (номер вместо токена →
404); закрытые вложения (без авторизации 401); операторские действия анонимно (403 не с
loopback, 401 при заданном токене, 403 без заголовка `X-FloodValue-Media`); модерация →
событие в `GET /api/media/events`; повторный импорт без дублей; перепечатки → одно
событие, один первоисточник; одна статья про две территории → два события; прогноз,
отрицание, исторический материал; ошибка источника (503 → `failed`, 429 →
`skipped_rate_limit`, курсор и «последний успех» не сдвигаются, другие источники
работают); сохранение после перезапуска; поиск снимков по месту и времени (локальный
каталог без test-событий, сохранённый ответ Copernicus); нет подходящего наблюдения →
нельзя поставить «согласуется»/«расхождение»; файрвол: ни один этап FloodValue не
импортирует `floodvalue.media`, модуль не пишет в `FLOODVALUE_OUTPUT_DIR`.

## 2. Живые источники

```text
$ uv run python -m floodvalue.media init
$ uv run python -m floodvalue.media ingest --source gdacs_floods
{"run_id": 1, "source_id": "gdacs_floods", "status": "ok", "error": null, "requests_used": 1, "items_seen": 3, "items_new": 3, "items_updated": 0, "items_relevant": 3, "llm_calls": 0}
$ uv run python -m floodvalue.media ingest --source gdacs_floods      # повтор сразу
{"run_id": 2, "source_id": "gdacs_floods", "status": "ok", "error": null, "requests_used": 1, "items_seen": 3, "items_new": 0, "items_updated": 0, "items_relevant": 0, "llm_calls": 0}
$ uv run python -m floodvalue.media ingest --source gdelt_floods
{"run_id": 3, "source_id": "gdelt_floods", "status": "failed", "error": "сеть: ReadTimeout: HTTPSConnectionPool(host='api.gdeltproject.org', port=443): Read timed out. (read timeout=10.0)", "requests_used": 3, ...}
$ uv run python -m floodvalue.media sources
{"source_id": "gdacs_floods", "kind": "bulletin", "enabled": 1, "terms_status": "checked", "last_success_at": "2026-09-26T08:30:46Z", "last_error_at": null, "last_error": null}
{"source_id": "gdelt_floods", "kind": "news", "enabled": 1, "terms_status": "checked", "last_success_at": null, "last_error_at": "2026-09-26T08:31:23Z", "last_error": "сеть: ReadTimeout: ..."}
```

| Источник | Результат | Вывод |
|---|---|---|
| GDACS event list API (сводка) | **работает**: 3 события FL (India, China, Nepal), повтор — 0 новых | реальная сводка проходит сбор → событие с точностью «регион» (центроид GDACS) |
| GDACS, пустое окно | `HTTP 204` без тела (проверено `curl`, окно 24–27.09) | обрабатывается как «в окне событий нет», не как ошибка |
| GDELT DOC 2.0 (новости) | **не проверен**: в 08:03, 08:12, 08:28 — `HTTP 429` («one request every 5 seconds») на единичные запросы, в 08:31 — таймаут 10 с | внешний блокер (ограничение по нашему адресу); состояние источника честное: `error`, «последний успех» пуст |
| Copernicus Data Space OData (поиск S1 GRD) | **работает** анонимно: для сводки India — 8 записей за окно события | записи `found`, пригодность `partial_cover` (место известно только до региона), подготовка сцены не реализована |

Условия источников проверены 26.09.2026: GDACS — https://www.gdacs.org/About/termofuse.aspx
(данные «as is», ориентировочные, не заменяют официальные оповещения); GDELT —
https://www.gdeltproject.org/about.html#termsofuse (свободное использование с обязательной
ссылкой на GDELT; тексты статей принадлежат изданиям → храним только заголовок и URL).

## 3. Наблюдения для события

```text
$ curl -X POST -H "X-FloodValue-Media: 1" -d '{"external": true}' .../api/media/events/1/observations
{"found":8,"errors":[]}
cdse S1D_IW_GRDH_1SDV_20260810T004021_… 2026-08-10 partial_cover found | место известно только до района/региона — снимок покрывает точку-заместитель, а не всю область
$ uv run python -m floodvalue.media demo
{"demo_events": 2, "demo_report_pending": "R-260926-A9E506"}
$ curl -X POST ... -d '{"external": false}' .../api/media/events/4/observations
{"found":13,"errors":[]}
ДЕМО: Floods hit Batticaloa, hundreds evacuated
sen1floods11_local Sri-Lanka_713926 2017-05-30 suitable prepared
```

ДЕМО-replay привязан к validation-чипу `Sri-Lanka_713926` (событие 2017 года). Test-события
(Bolivia, Mekong, Spain) локальный каталог не читает.

## 4. Браузер (Chromium через Playwright, вне lockfile: `uv run --no-project --with playwright`)

Телефон — эмуляция 390×844, touch, DPR 2. Реального устройства не было.

```text
empty submit errors: Выберите, что произошло. | Опишите подробнее — хотя бы 10 символов. | Отметьте место на карте или напишите его название.
point: 55.693068 37.792969                       # точка поставлена нажатием на карту
network error msg: Не удалось отправить — нет связи. Введённый текст сохранён, попробуйте ещё раз: повторная отправка не создаст дубль.
text kept: Вода подошла к домам у дороги, двор залит
done number: R-260926-A84F29
status page: Обращение | R-260926-A84F29 | Отправлено | 26.09.2026, 11:26:31 | Статус | На проверке у аналитика
queue count: 2
events on map list: 6 no-geo: 1                  # после «Принять → событие»
review options disabled: ['not_checked:false', 'consistent:true', 'discrepancy:true', 'insufficient_data:false']
toast: Найдено записей: 8                        # «Найти снимки + Copernicus» для сводки India
no console errors                                # кроме намеренно оборванного запроса
```

Скриншоты: `media/report_phone_top.jpg`, `media/report_phone_done.jpg`,
`media/analyst_card_obs.jpg`, `media/analyst_gdacs_india.jpg`.

## 5. Worker

```text
$ uv run python -m floodvalue.media worker      # SIGTERM через ~14 с
worker: интервал — poll_interval_s каждого источника (sources.yaml); проверка расписания каждые 30 с
worker остановлен, записанные данные сохранены
```

Параллельный цикл при удерживаемой блокировке отклоняется — тест
`tests/media/test_worker.py::test_second_cycle_is_refused_while_first_holds_lock`.

## Что не проверено

- Новостной источник живьём (GDELT недоступен, см. выше).
- Реальный телефон; геолокация браузера (в эмуляции не вызывалась).
- Геокодер Nominatim живьём (выключен по умолчанию, чтобы не нагружать публичный сервис).
- Языковая модель: провайдер не подключён.
