# Шрифты публичных страниц (/report, /report/status)

Раздаются как `/media/static/fonts/<файл>.woff2` (`floodvalue.media.web.font_file`): только
woff2 из этого каталога, CSP `font-src 'self'` — внешние шрифтовые CDN запрещены.
Рядом с каждым шрифтом лежит его лицензия (`<Семейство>-OFL.txt`); подключать только шрифты
со свободной лицензией (SIL OFL 1.1 и т. п.), источник и версию — в таблице ниже.

| Файл | Семейство | Лицензия | Источник |
|---|---|---|---|
| `Unbounded-Variable.woff2` | Unbounded (`wght` 200–900) — заголовки формы | SIL OFL 1.1, `Unbounded-OFL.txt` | [google/fonts `ofl/unbounded/Unbounded[wght].ttf` @ 8b0a1d0](https://github.com/google/fonts/blob/8b0a1d0f5983c89bc2b93f1b5fb55f9e252744b5/ofl/unbounded/Unbounded%5Bwght%5D.ttf), субсет латиница + кириллица + ₽ № |
| `Onest-Variable.woff2` | Onest (`wght` 100–900) — текст формы | SIL OFL 1.1, `Onest-OFL.txt` | [google/fonts `ofl/onest/Onest[wght].ttf` @ 4c1db3a](https://github.com/google/fonts/blob/4c1db3aec83c67dd223dd82a68c039cab30917a9/ofl/onest/Onest%5Bwght%5D.ttf), тот же субсет |
