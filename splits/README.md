# Split v1: разбиение по событиям (заморожен)

`split_v1.json` описывает, как 446 ручных чипов Sen1Floods11 v1.1 разложены по частям.
Код: `src/floodvalue/data/split.py`. Решения: D-006, D-012, D-016 в `docs/decisions.md`.
Хеш sha256: `0098346e0f87297501409a1a8522afab6209e35f56e8cf1b68996909c8b63c75`.

| Часть | События | Чипов | Вода среди валидных пикселей | Для чего |
|---|---|---|---|---|
| train | Ghana, India, Pakistan, Paraguay, Somalia | 237 | 8,9 % | обучение, нормализация, порог baseline |
| dev | USA | 69 | 4,6 % | выбор модели и чекпойнта, контролируемый эксперимент |
| validation | Sri-Lanka, Nigeria | 60 | 15,2 % | калибровка и порог основного метода |
| test | Spain, Mekong, Bolivia | 75 | не считалась (метки запечатаны) | только `eval-test` замороженной конфигурации |
| excluded | Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483 | 5 | все пиксели −1 | |

Итого 237 + 69 + 60 + 75 + 5 = 446.

- Части назначаются по событию. Все окна одной сцены S1 попадают в одну часть, поэтому
  соседние чипы не оказываются по разные стороны границы. Контуры событий разных частей
  из `Sen1Floods11_Metadata.geojson` не пересекаются; это проверяется при каждой сборке.
- Метки test при построении не читаются. Исключения по меткам есть только у событий
  train/dev/validation. Test-чип исключается только если нет его файла.
- Seed `20260925` в состав частей не входит: разбиение задано правилом. Это общий seed
  проекта для последующих случайных шагов.
- Хеш считается от канонического JSON четырёх списков (`sort_keys`, разделители `,`/`:`, ASCII).
- Weak-данные (S1Weak) событий dev, validation, test и Colombia не используются.

## Команды

```bash
make split                        # пересобрать из data/raw и сверить с файлом и с D-016
uv run --frozen pytest tests/split
```

`make split` читает `$FLOODVALUE_DATA_DIR/raw`: авторские CSV `splits/flood_handlabeled/`,
`Sen1Floods11_Metadata.geojson`, `HandLabeled/S1Hand` и `HandLabeled/LabelHand`.
Код выходит с 1, если пересобранный файл или хеш в `docs/decisions.md` не совпадают.

## В коде

```python
from floodvalue.data.split import chip_ids
chip_ids("train"); chip_ids("dev"); chip_ids("validation")
chip_ids("test")                   # SealedTestError
chip_ids("test", allow_test=True)  # только eval-test (D-010)
```
