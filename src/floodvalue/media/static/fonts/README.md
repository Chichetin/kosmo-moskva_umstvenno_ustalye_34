# Шрифты публичных страниц (/report, /report/status)

Раздаются как `/media/static/fonts/<файл>.woff2` (`floodvalue.media.web.font_file`): только
woff2 из этого каталога, CSP `font-src 'self'` — внешние шрифтовые CDN запрещены.
Рядом с каждым шрифтом лежит его лицензия (`<Семейство>-OFL.txt`); подключать только шрифты
со свободной лицензией (SIL OFL 1.1 и т. п.), источник и версию — в таблице ниже.

Гарнитура та же, что у интерфейса оператора (`web/public/fonts/`, #247): одна Science Gothic,
текст — ширина 85, логотип — ширина 200.

| Файл | Семейство | Лицензия | Источник |
|---|---|---|---|
| `ScienceGothic-Variable.woff2` | Science Gothic (`wdth` 85–200, `wght` 300–800) | SIL OFL 1.1 без зарезервированного имени, `ScienceGothic-OFL.txt` | `web/public/fonts/ScienceGothic-Variable.woff2` (коммит `29c9226`), субсет латиница + кириллица + ₽ № стрелки (`pyftsubset`), оси `slnt` = 0 и `CTRS` = 0 зафиксированы (`fonttools varLib.instancer`): 127 КБ вместо 544 КБ |
