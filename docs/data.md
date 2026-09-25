# Данные: официальный Sen1Floods11

Проверено 2026-09-25. Всё ниже получено командами из этого файла; выводы сокращены. Где факт взят из статьи или README, а не из файлов, это сказано прямо.

## 1. Источник, версия, лицензия

| Что | Значение | Откуда |
|---|---|---|
| Репозиторий | https://github.com/cloudtostreet/Sen1Floods11, ветка `master`, последний push 2021-04-09 | `gh api repos/cloudtostreet/Sen1Floods11` |
| Бакет | `gs://sen1floods11`, публичный HTTP `https://storage.googleapis.com/sen1floods11/`. В README опечатка: `gs://senfloods11/` | README + листинг ниже |
| Версия | **v1.1**, всё лежит под префиксом `v1.1/` (другого верхнего префикса нет). Объекты датированы 2020-10-01…2020-10-15. v1.0 по README лежит в старом бакете `cnn_chips` | листинг |
| Лицензия | **Явно не указана.** В репозитории нет LICENSE (`raw.../LICENSE` → 404, GitHub API `license: null`), в STAC-коллекциях `"license": "proprietary"`. README просит цитировать статью Bonafilia et al. 2020 (CVPRW) | см. команды |
| Цитирование | Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. 2020. Sen1Floods11… CVPR Workshops, pp. 210–211 | README |

```bash
curl -sL https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/README.md
curl -sL https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/master/docs/README.md
gh api repos/cloudtostreet/Sen1Floods11 --jq '{license:.license, pushed:.pushed_at}'
# {"license":null,"pushed":"2021-04-09T06:33:39Z"}
curl -s https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_label/collection.json \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['license'], d['extent'])"
# proprietary {'spatial': {...}, 'temporal': {'interval': [['2016-08-12T00:00:00Z', '2019-09-17T00:00:00Z']]}}
```

## 2. Структура бакета и объём

gsutil/gcloud не нужны: листинг идёт через публичный JSON API GCS с пагинацией.

```bash
cat > list_bucket.py <<'EOF'
import json, urllib.request, urllib.parse, sys
prefix, out = sys.argv[1], sys.argv[2]; items, tok = [], None
while True:
    q = {"prefix": prefix, "maxResults": "1000", "fields": "items(name,size,updated),nextPageToken"}
    if tok: q["pageToken"] = tok
    d = json.load(urllib.request.urlopen("https://storage.googleapis.com/storage/v1/b/sen1floods11/o?" + urllib.parse.urlencode(q)))
    items += d.get("items", []); tok = d.get("nextPageToken")
    if not tok: break
json.dump(items, open(out, "w")); print(len(items), "objects", sum(int(i["size"]) for i in items)/1e9, "GB")
EOF
python3 list_bucket.py v1.1/ listing_v1.1.json
# 31074 objects 36.864294387 GB
```

Агрегация по каталогам (catalog/* = 9664 STAC-json, не показаны):

```
v1.1  (Sen1Floods11_Metadata.geojson 14 KB, catalog.zip 10.5 MB)
v1.1/checkpoints                                          4     527.2 MB
v1.1/data/flood_events/HandLabeled/JRCWaterHand         446       0.5 MB
v1.1/data/flood_events/HandLabeled/LabelHand            446       2.6 MB
v1.1/data/flood_events/HandLabeled/S1Hand               446     729.5 MB
v1.1/data/flood_events/HandLabeled/S1OtsuLabelHand      446       1.7 MB
v1.1/data/flood_events/HandLabeled/S2Hand               446    1018.6 MB
v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak   4384      18.2 MB
v1.1/data/flood_events/WeaklyLabeled/S1Weak            4384    7322.7 MB
v1.1/data/flood_events/WeaklyLabeled/S2IndexLabelWeak  4385      33.0 MB
v1.1/data/flood_events/WeaklyLabeled/S2Weak            4384   25691.9 MB
v1.1/data/perm_water/JRCPerm                            814       6.0 MB
v1.1/data/perm_water/S1Perm                             814    1478.8 MB
v1.1/splits/flood_handlabeled                             4       0.0 MB
v1.1/splits/perm_water                                    4       0.2 MB
```

- **HandLabeled целиком: 2230 файлов, 1752.9 MB.** Только S1Hand + LabelHand: 892 файла, 732.0 MB.
- README обещает «~14 GB» на весь набор, реально v1.1 весит 36.9 GB (из них S2Weak 25.7 GB).
- docs/README пишет 4 385 weak-чипов и 815 JRCPerm. В бакете S1Weak/S2Weak/S1OtsuLabelWeak по 4384, S2IndexLabelWeak 4385, JRCPerm/S1Perm по 814.
- Имена файлов: `EVENT_CHIPID_LAYER.tif`, например `Bolivia_129334_S1Hand.tif`. Имя события может содержать дефис (`Sri-Lanka`), поэтому парсить надо через `rsplit("_", 2)`.

## 3. Что скачано в `data/raw/` (6.8 MB)

```bash
B=https://storage.googleapis.com/sen1floods11/v1.1
mkdir -p data/raw/splits/flood_handlabeled data/raw/catalog
for f in flood_train_data.csv flood_valid_data.csv flood_test_data.csv flood_bolivia_data.csv; do
  curl -sSf -o data/raw/splits/flood_handlabeled/$f $B/splits/flood_handlabeled/$f; done
curl -sSf -o data/raw/Sen1Floods11_Metadata.geojson $B/Sen1Floods11_Metadata.geojson
curl -sSf -o data/raw/catalog/catalog.json $B/catalog/catalog.json
for c in Spain_7370579 Mekong_922373 Sri-Lanka_713926 Bolivia_129334; do
  for l in S1Hand LabelHand JRCWaterHand S1OtsuLabelHand; do
    mkdir -p data/raw/HandLabeled/$l
    curl -sSf -o data/raw/HandLabeled/$l/${c}_$l.tif $B/data/flood_events/HandLabeled/$l/${c}_$l.tif; done; done
```

Структура `data/raw/` повторяет бакет (`HandLabeled/<LAYER>/<EVENT>_<ID>_<LAYER>.tif`), так что полную загрузку можно положить туда же без переименований. Кроме того, в `data/interim/hand_label_stats.csv` лежат доли −1/0/1 и JRC по всем 446 ручным меткам (см. §5). Их прочитали по HTTP через `/vsicurl/` и на диск не сохраняли.

> В репозитории **нет `.gitignore`**, и `data/` сейчас отображается в `git status` как untracked. Перед коммитом нужно добавить `data/` в `.gitignore`.

### 3а. Полная загрузка (без WeaklyLabeled): 13 537 файлов, 3,8 ГБ

Скрипт `scripts/download_sen1floods11.py` качает по HTTPS без gsutil, проверяет md5 каждого объекта и раскладывает в ту же структуру `data/raw/`. В неё входят HandLabeled (все 5 слоёв), perm_water, splits, catalog, checkpoints и метаданные. Повторный запуск ничего не качает заново, а только сверяет md5.

```bash
python3 scripts/download_sen1floods11.py            # в $FLOODVALUE_DATA_DIR/raw, по умолчанию data/raw
python3 scripts/download_sen1floods11.py --no-s2    # без S2Hand (-1 ГБ)
python3 scripts/download_sen1floods11.py --with-weak  # + WeaklyLabeled (~33 ГБ), кейсу не нужен
# 25.09: 13537 файлов, done {'ok': 13537, 'skip': 0, 'fail': 0}; повтор — {'skip': 13537}
```

Манифест бакета (имя, размер, md5, crc32c, generation) пишется в `data/raw/bucket_manifest.json`, копия на 25.09 лежит в `docs/sen1floods11/bucket_manifest_v1.1.json`. Там же лежат авторские сплиты, `Sen1Floods11_Metadata.geojson` и README авторов. Оригиналы условий кейса лежат в `docs/task/`: PDF, их текст и выжимка `ЗАДАНИЕ.md`.

Окружение для проверки: `python3.12 -m venv /tmp/claude-1000/sfvenv && /tmp/claude-1000/sfvenv/bin/pip install rasterio numpy`. Результат: rasterio 1.5.1, GDAL 3.12.4, numpy 2.5.3.

## 4. Формат чипов (проверено rasterio на 4 скачанных чипах)

```python
# /tmp/claude-1000/sfvenv/bin/python, cwd = корень репо
import rasterio, numpy as np
with rasterio.open("data/raw/HandLabeled/S1Hand/Bolivia_129334_S1Hand.tif") as r:
    print(r.count, r.dtypes, r.width, r.height, r.crs, r.nodata, r.transform, r.descriptions)
    a = r.read(); print(np.isnan(a).mean(axis=(1, 2)), np.nanmin(a, (1, 2)), np.nanmax(a, (1, 2)))
# аналогично LabelHand / JRCWaterHand / S1OtsuLabelHand + np.unique(..., return_counts=True)
```

| Слой | Каналы | dtype | nodata в теге | Значения (факт) | Описание канала |
|---|---|---|---|---|---|
| S1Hand | 2 (VV, VH) | float32 | `nan` | dB: VV от −48 до +26, VH от −54 до +8; медиана VV ≈ −10…−15, VH ≈ −17…−24 | `('VV','VH')` |
| LabelHand | 1 | int16 | **не задан** | −1 / 0 / 1 | нет |
| JRCWaterHand | 1 | uint8 | не задан | **только 0 / 1** | `'occurrence'` |
| S1OtsuLabelHand | 1 | int16 | −1 | 0 / 1 (в 4 чипах −1 не встретился) | `'s1flood'` |

Общее для всех слоёв: 512×512, `EPSG:4326`, шаг **8.983152841195e-05° (в градусах)**, DEFLATE, не тайлованы (`tiled=False`), overview по NEAREST.

Выход по чипам (сокращён):

```
Spain_7370579    lat 38.1  px ≈ 7.87 × 9.93 м  чип ≈ 4.03 × 5.09 км = 20.49 км²
  S1 NaN 0.0000 | VV min −39.24 p1 −25.35 med −10.48 p99 2.46 max 25.65 | VH min −47.06 med −17.73 max 4.43
  Label {−1: 0.0004, 0: 0.7325, 1: 0.2672}  JRC {0: 1.0}  Otsu {0: .867, 1: .133}  Otsu=hand на валидных 0.848
Mekong_922373    lat 12.2  px ≈ 9.77 × 9.93 м  25.45 км²
  S1 NaN 0.0000 | VV med −10.13 max 9.00 | VH med −16.67 max −5.97
  Label {−1: 0.0000, 0: 0.5431, 1: 0.4569}  JRC {0: .9997, 1: .0003}  Otsu=hand 0.783
Sri-Lanka_713926 lat 7.75  px ≈ 9.91 × 9.93 м  25.80 км²
  Label {−1: 0.0002, 0: 0.6927, 1: 0.3071}  JRC {0: .6983, 1: .3017}  метка=1 внутри JRC=1: 0.985  Otsu=hand 0.948
Bolivia_129334   lat −14.4 px ≈ 9.69 × 9.93 м  25.22 км²
  S1 NaN 0.0029 (оба канала) | VV max 0.47 | VH max −6.32
  Label {−1: 0.0928, 0: 0.3096, 1: 0.5975}  JRC {0: .9842, 1: .0158}  Otsu=hand 0.927
  в пикселях, где S1 = NaN, метка: 0 → 173 px, 1 → 576 px  (не −1!)
```

Сетки. Origin, размер и CRS у всех четырёх слоёв совпадают, но шаг у S1Hand `8.983152841195215e-05`, а у меток `8.983152841195857e-05`. Разница ~6e-18°, это шум float. Строгое `transform ==` возвращает False, поэтому сравнивать надо через `np.allclose`. Пиксели совпадают один к одному.

## 5. События, число ручных чипов, авторский split

```python
import csv, collections
S = "data/raw/splits/flood_handlabeled/"; ev = collections.defaultdict(collections.Counter)
for s in ["train", "valid", "test", "bolivia"]:
    for a, b in (r for r in csv.reader(open(S + f"flood_{s}_data.csv")) if r):
        e, cid, _ = a.rsplit("_", 2); assert b == f"{e}_{cid}_LabelHand.tif"; ev[e][s] += 1
```

CSV без заголовка, две колонки `EVENT_ID_S1Hand.tif,EVENT_ID_LabelHand.tif`. Строк: train 252, valid 89, test 90, bolivia 15, всего 446 уникальных чипов.

| Событие | train | valid | test | bolivia | всего | S1 дата | ср. доля воды | ср. доля −1 | ср. доля JRC=1 |
|---|---|---|---|---|---|---|---|---|---|
| Bolivia | 0 | 0 | 0 | 15 | 15 | 2018-02-15 | 0.116 | 0.271 | 0.015 |
| Ghana | 31 | 11 | 11 | 0 | 53 | 2018-09-18 | 0.042 | 0.338 | 0.013 |
| India | 40 | 14 | 14 | 0 | 68 | 2016-08-12 | 0.104 | 0.160 | 0.023 |
| Mekong (в метаданных «Cambodia») | 18 | 6 | 6 | 0 | 30 | 2018-08-05 | 0.219 | 0.093 | 0.076 |
| Nigeria | 10 | 4 | 4 | 0 | 18 | 2018-09-21 | 0.201 | 0.143 | 0.019 |
| Pakistan | 16 | 6 | 6 | 0 | 28 | 2017-06-28 | 0.055 | 0.267 | 0.027 |
| Paraguay | 39 | 14 | 14 | 0 | 67 | 2018-10-31 | 0.080 | 0.050 | 0.028 |
| Somalia | 15 | 5 | 6 | 0 | 26 | 2018-05-07 | 0.051 | 0.211 | 0.000 |
| Spain | 18 | 6 | 6 | 0 | 30 | 2019-09-17 | 0.129 | 0.017 | 0.052 |
| Sri-Lanka | 24 | 9 | 9 | 0 | 42 | 2017-05-30 | 0.107 | 0.095 | 0.079 |
| USA | 41 | 14 | 14 | 0 | 69 | 2019-05-22 | 0.044 | 0.024 | 0.016 |
| **Итого** | 252 | 89 | 90 | 15 | **446** | | | | |

- **Пары полные.** По листингу множества чипов в S1Hand, LabelHand, S2Hand, JRCWaterHand и S1OtsuLabelHand совпадают друг с другом и со списками split (446 = 446). У каждого чипа есть S1Hand + LabelHand.
- **В авторском split события перемешаны.** Все 10 событий, кроме Bolivia, есть одновременно в train, valid и test. В статье (§2.2) это прямо сказано: «random 60-20-20 split» по чипам, «All hand-labeled data from Bolivia is held out». Test авторов поэтому не проверяет перенос на новое событие, это проверяет только Bolivia.
- Доли по меткам (`data/interim/hand_label_stats.csv`, 446 чипов): наборы значений `{−1,0,1}` у 326 чипов, `{0,1}` у 67, `{−1,0}` у 31, `{0}` у 16, `{−1}` у 5, `{−1,1}` у 1. У 52 чипов нет ни одного пикселя воды, у 38 больше 50% пикселей −1. **Пять чипов полностью −1:** Ghana_26376 (train), Ghana_5079, Ghana_234935, Ghana_277 (valid), Ghana_83483 (test). JRC-слой: `{0,1}` у 233 чипов, `{0}` у 213.

Как получены доли (читается по HTTP, на диск не пишется):

```python
B = "/vsicurl/https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/"
l = rasterio.open(B + f"LabelHand/{chip}_LabelHand.tif").read(1)
j = rasterio.open(B + f"JRCWaterHand/{chip}_JRCWaterHand.tif").read(1)
# invalid=(l==-1).mean(), water=(l==1).mean(), jrc_frac=(j==1).mean(); 16 потоков, ~1 мин на все 446
```

## 6. Даты событий: `Sen1Floods11_Metadata.geojson`

```python
import json; d = json.load(open("data/raw/Sen1Floods11_Metadata.geojson"))
for f in d["features"]: print(f["properties"])
```

- FeatureCollection из **12 полигонов** (по одному на событие), `crs` не задан. Поля: `ID, location, ISO_CC, s1_date, s2_date, orbit, rel_orbit_num, coincident_size, VH_thresh, train_chip, val_chip`.
- **Формат дат `YYYY/MM/DD`** (например `'2018/02/15'`), а не `YYYY-MM-dd`, как написано в README. Парсить через `datetime.strptime(s, "%Y/%m/%d")`. S1 и S2 сняты в пределах 0–2 дней друг от друга.
- `location` не совпадает с префиксом в именах файлов: в метаданных **`Cambodia`**, в файлах `Mekong` (30 ручных чипов, дата 2018/08/05). Сопоставлять лучше по числу чипов или ISO, а не по строке.
- В метаданных есть **Colombia** (2018/08/22, `val_chip: 0`). Ручных чипов у неё нет, только weak.
- `val_chip` равно числу ручных чипов события (Ghana 53, India 68, … совпадает с таблицей §5). Сумма `train_chip` = 4384 = числу weak-чипов.
- Дата есть и на уровне чипа: STAC-item `.../catalog/sen1floods11_hand_labeled_label/<EVENT>_<ID>_label/<EVENT>_<ID>_label.json` → `properties.datetime`. Для Mekong_382276 это `2018-08-05T00:00:00Z`.
- `VH_thresh` — порог VH (dB) авторского Otsu для события, от −25.13 (Spain) до −19.56 (Pakistan).

## 7. Постоянная вода: JRCWaterHand и риск «информации из будущего»

Что известно:
- Слой бинарный 0/1 (uint8), описание канала `'occurrence'`, разрешение пересчитано на 10-м сетку чипа (исходник Landsat 30 м).
- Версию JRC GSW авторы нигде не указывают: её нет в README, docs/README, STAC и тегах GeoTIFF (теги: только `OVR_RESAMPLING_ALG`, `AREA_OR_POINT`). В статье (§2.1, про JRCPerm) сказано: «water samples were derived from the transition layer, which identifies "permanent" water as pixels … observed to have detected water presence at both the beginning (1984) and the end (2018) of the dataset». Период **1984–2018** соответствует выпуску GSW v1.1. То, что тем же способом построен и JRCWaterHand, **предположение**: описание канала `occurrence` с «transition layer» не совпадает.
- На 4 чипах метка = 1 в 96–100% пикселей JRC = 1 (Sri-Lanka 0.985, Bolivia 0.962, Mekong 1.000). Значит, **LabelHand = вся вода (паводковая + постоянная)**, а JRC — её подмножество.

Где риск утечки. Для событий 2016–2017 (India 2016-08, Sri-Lanka 2017-05, Pakistan 2017-06) и для большинства событий 2018 года слой, построенный по данным до конца 2018, использует наблюдения **после** даты события. Для Spain и USA (2019) данные заканчиваются раньше события. Для имитации оперативного решения (task.md: «запрещено использовать информацию, возникшую после даты анализируемого события») JRCWaterHand как маску постоянной воды без оговорок использовать **нельзя**.

Корректные варианты:
1. **Называть результат «картой воды»** и постоянную воду не вычитать. Это самый безопасный вариант, и task.md его допускает.
2. Если заявлять «временное затопление», брать маску, построенную **только по годам до события**. Например, JRC GSW YearlyHistory (класс «permanent» за год Y−1, в GEE `JRC/GSW1_4/YearlyHistory`) или MonthlyHistory до месяца события. В `docs/` записать URL, версию, дату публикации и то, что использованы только годы < года события. *Версию и доступность этого источника мы здесь не проверяли.*
3. JRCWaterHand можно оставить как **диагностику**: IoU отдельно на пикселях JRC = 1 и вне их, как в статье. Но не как вход модели и не для вычитания в демо.

## 8. Ловушки

1. **S1 уже в dB.** Повторно логарифмировать нельзя. Бывают положительные значения (VV до +25.65 dB у Spain_7370579: яркие рассеиватели, городская застройка), поэтому клиппинг обязателен. Авторский Train.ipynb делает `np.nan_to_num(x)`, затем `np.clip(x, -50, 1)`, затем `(x + 50) / 51`, затем `Normalize([0.6851, 0.5235], [0.0820, 0.1102])`.
2. **NaN в S1 — это nodata** (тег `nodata=nan`). Метка в этих пикселях **не обязательно −1**: у Bolivia_129334 под NaN лежат 0 и 1 (749 px). Маска валидности = `(label != -1) & isfinite(VV) & isfinite(VH)`. Если сделать `nan_to_num` → 0 dB, NaN превращаются в «яркий сухой» пиксель.
3. **−1 в LabelHand не объявлен как nodata** (тег `nodata=None`), а в S1OtsuLabelHand объявлен (`-1`). Маскировать вручную. У авторов CrossEntropy с `ignore_index=255`, то есть −1 перемаплено.
4. **Пустые и почти пустые чипы.** 5 чипов Ghana целиком −1, 38 чипов больше чем на 50% −1, 52 чипа без воды. Для IoU по чипу это деление на 0: метрики надо агрегировать по пикселям или явно исключать такие чипы.
5. **EPSG:4326, шаг в градусах.** 8.983e-05° ≈ 10 м только по широте. По долготе 10 м·cos(lat): у Spain 7.87 м, и чип там 20.5 км², а у экватора ~25.8 км². Площади воды и объектов (кандидатные зоны ≥ 1 км², цена по контуру) считать в UTM или геодезически (`pyproj.Geod`), не как `n_px * 100 м²`.
6. **Строгое сравнение transform ломается** из-за разницы шага ~1e-17° между S1 и метками. Использовать `np.allclose` или `rasterio.transform` с допуском.
7. **Соседние чипы одного события** не перекрываются (статья: «non-overlapping»), но лежат рядом, в одной сцене S1 и с тем же порогом `VH_thresh`. Пространственная автокорреляция делает random split оптимистичным. Split нужен по событиям.
8. **Авторский test не hold-out по событиям.** Честный перенос на новое событие проверяет только Bolivia (15 чипов, у них в среднем 27% пикселей −1). В статье чекпоинт выбирался по valid, где есть все 11 событий.
9. **Авторские чекпоинты (`v1.1/checkpoints`) обучены на авторском split.** Их test-чипы соседствуют с train-чипами, поэтому оценивать их на нашем event-split как «чистые» нельзя.
10. **Названия событий не совпадают**: Mekong ↔ Cambodia, Sri-Lanka с дефисом, Colombia без ручных чипов. Формат дат `YYYY/MM/DD`.
11. **JRC-слой = «будущее»** для событий 2016–2018 (§7). К тому же он бинарный, происходит от Landsat 30 м и назван `occurrence`, хотя по статье это transition.
12. **S1OtsuLabelHand — это не ground truth**, а авторский baseline (Otsu по VH с фокальным сглаживанием, порог на событие из `VH_thresh`). Согласие с ручной меткой на 4 чипах 0.78–0.95. Для сравнения его можно взять, но свой пороговый baseline всё равно нужен, и порог должен подбираться только на train.
13. **Ручная метка сделана по оптике.** Разметчики правили S2-классификацию (статья §2.2). Вода, невидимая в SAR (под пологом леса, при ветровой ряби) и видимая в S2, может считаться ошибкой модели. Кроме того, выборка стратифицирована: 75% чипов с водой больше 0.02 км² по S2.
14. **Опечатки и расхождения в README**: `gs://senfloods11/`, «~14 GB» (реально 36.9 GB), 4385/815 против 4384/814 в бакете.

## 9. Рекомендация (предложение, не решение)

*Предложение:* отказаться от авторского split и разделить 446 ручных чипов **по событиям**. **Train** — Ghana, India, Paraguay, Pakistan, Somalia, USA (311 чипов). **Valid** для выбора порога, эпохи и калибровки — Sri-Lanka + Nigeria (60 чипов). **Test** — Spain + Mekong + Bolivia (75 чипов), не трогать до финала. В test попадают три разных региона (Европа, Юго-Восточная Азия, Южная Америка), авторский hold-out Bolivia и события с наибольшей долей воды (Mekong 0.22). Числа чипов проверены по таблице §5: 53+68+67+28+26+69 = 311, 42+18 = 60, 30+30+15 = 75. Weak-данные (S1Weak) из событий test/valid в обучение не брать. Кандидаты на демо-чип (все в test при этом разбиении, с ручной меткой, почти без −1 и NaN):
- **Spain_7370579**: вода 26.7%, −1 0.04%, NaN в S1 0%, JRC 0%. Это пример из самого README; в авторском split он тоже в test. Основной кандидат: почти вся вода здесь паводковая, так что вопрос «карта воды или затопление» стоит мягче.
- **Mekong_922373**: вода 45.7%, −1 0.002%, NaN 0%, JRC 0.03%.
- **Bolivia_129334**: вода 59.8%, −1 9.3%, NaN 0.29%, JRC 1.6%. Это авторский hold-out, хорош для рассказа «новое событие», но надо показать работу с −1 и NaN.

Запасной вариант с большой долей воды — Spain_7387658 (вода 58.4%, −1 0%, JRC 6.8%, авторский test). S1 для него не скачан и не проверен.
