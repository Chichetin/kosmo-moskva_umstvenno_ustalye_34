# Журнал запусков на test

Test запечатан (D-010): читается только `make eval-test` для замороженной конфигурации.
Каждый запуск — строка ниже. Повторная настройка после просмотра test — новый
эксперимент с новым независимым test; эту конфигурацию больше не оценивать.

Как запускается (D-030, после заморозки 26.09 20:30, issue #32):

```bash
set -a; . ./.env; set +a
make eval-test        # без подтверждения — отказ; печатает sha256 текущей конфигурации
FLOODVALUE_EVAL_TEST_CONFIRM=<sha256 из отказа или из eval-validation.md> make eval-test
```

sha256 конфигурации — от имени основного метода и байтов `models/<метод>/config.json`
(в нём sha256 весов), `models/baseline/config.json`, `splits/split_v1.json` и файлов
`models/calibration/` (`floodvalue.model.eval_test.frozen_config_hash`). Команда
отказывает, если в `src/`, `models/`, `configs/`, `splits/`, `pyproject.toml`, `uv.lock`
есть незакоммиченные изменения. Строку ниже пишет сама команда: «начат» — до чтения
первого test-чипа; по завершении заменяется результатом, при сбое — «прерван: <ошибка>».
Отчёт — `docs/verification/test-eval.md` и `test-eval*.csv/json/png`.

| Дата, время | Commit | sha256 конфигурации | Метод(ы) | Кто | Результат (файл) |
|---|---|---|---|---|---|
