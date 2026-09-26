import { useEffect, useMemo, useState } from "react";
import { ExternalLink, X } from "lucide-react";
import { EXTRACTION, ORIGIN, REPORT_TYPE } from "./mediaLabels";
import "./newsPanel.css";

// Everything the media monitor has read, so the operator (and the jury) can see which news and
// bulletins the "Сообщения" layer comes from. Read-only: /api/media/publications of the media
// module (make site). Titles and excerpts are third-party text, shown as text only.

type Publication = {
  publication_id: number;
  source_id: string;
  source_name?: string;
  source_kind?: string;
  attribution?: string | null;
  url: string;
  title: string;
  text_excerpt?: string | null;
  published_at?: string | null;
  fetched_at: string;
  report_type?: string | null;
  extraction_method?: string | null;
  extraction_status: string;
  is_demo: number | boolean;
  event_id?: number | null;
};

type Filter = "all" | "real" | "demo";

const STATUS: Record<string, string> = {
  pending: "ждёт разбора",
  ok: "разобрано",
  needs_review: "на ручной проверке",
  invalid: "не удалось разобрать",
  irrelevant: "не о паводке",
};

const safeUrl = (value: string) => /^https?:\/\//i.test(value) ? value : null;

const day = (value?: string | null) => {
  if (!value) return "дата неизвестна";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? "дата неизвестна"
    : new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" })
      .format(parsed);
};

export default function NewsPanel({ onClose }: { onClose: () => void }) {
  const [items, setItems] = useState<Publication[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");

  useEffect(() => {
    let alive = true;
    fetch("/api/media/publications?limit=200")
      .then(async (response) => {
        if (response.status === 404) {
          throw new Error("Модуль сообщений СМИ не запущен. Запустите интерфейс командой make site "
            + "(оператор, форма очевидца и сообщения на одном адресе).");
        }
        if (response.status === 401 || response.status === 403) {
          throw new Error("Список новостей доступен аналитику: откройте интерфейс локально "
            + "или войдите на странице /media/analyst.");
        }
        if (!response.ok) throw new Error(`Не удалось загрузить новости (${response.status}).`);
        return response.json() as Promise<{ publications?: Publication[] }>;
      })
      .then((data) => {
        if (alive) setItems(data.publications ?? []);
      })
      .catch((reason: unknown) => {
        if (alive) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  const demoCount = items?.filter((item) => Boolean(item.is_demo)).length ?? 0;
  const realCount = (items?.length ?? 0) - demoCount;
  const shown = useMemo(() => (items ?? []).filter((item) =>
    filter === "all" ? true : filter === "demo" ? Boolean(item.is_demo) : !item.is_demo), [items, filter]);

  return (
    <aside className="news-panel" role="dialog" aria-label="Новости и сводки о паводках">
      <div className="news-head">
        <div>
          <small>МОНИТОРИНГ СООБЩЕНИЙ</small>
          <h2>Новости и сводки о паводках</h2>
          {items && (
            <p>
              {items.length} сообщений · реальных {realCount} · демо {demoCount}
            </p>
          )}
        </div>
        <button onClick={onClose} aria-label="Закрыть новости"><X size={17} /></button>
      </div>

      {demoCount > 0 && (
        <p className="news-demo-banner">
          <strong>ДЕМО</strong> — {demoCount} тестовых сообщений: имитация паводка для показа
          (помечены ниже). Это не реальные публикации.
        </p>
      )}

      {items && items.length > 0 && (
        <div className="news-filter" role="group" aria-label="Какие сообщения показать">
          {([["all", "Все"], ["real", "Реальные"], ["demo", "Демо"]] as const).map(([key, label]) => (
            <button key={key} className={filter === key ? "active" : ""} onClick={() => setFilter(key)}>
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="news-list">
        {error && <p className="news-note">{error}</p>}
        {!error && !items && <p className="news-note">Загружаем новости…</p>}
        {items && !items.length && (
          <p className="news-note">
            Сообщений пока нет: источники ещё не опрашивались. Сбор — python -m floodvalue.media ingest,
            демо-набор — python -m floodvalue.media demo.
          </p>
        )}
        {items && items.length > 0 && !shown.length && <p className="news-note">В этом фильтре сообщений нет.</p>}
        {shown.map((item) => {
          const url = safeUrl(item.url);
          return (
            <article key={item.publication_id} className="news-item">
              <div className="news-meta">
                {Boolean(item.is_demo) && <span className="news-badge demo">ДЕМО</span>}
                <span className="news-badge">{ORIGIN[item.source_kind ?? ""] ?? "источник"}</span>
                <span>{item.source_name ?? item.source_id}</span>
                <span>· {day(item.published_at ?? item.fetched_at)}</span>
              </div>
              <h3>
                {url
                  ? <a href={url} target="_blank" rel="noopener noreferrer">{item.title} <ExternalLink size={12} /></a>
                  : item.title}
              </h3>
              {item.text_excerpt && <p className="news-excerpt">{item.text_excerpt}</p>}
              <p className="news-reading">
                Разбор: {REPORT_TYPE[item.report_type ?? ""] ?? "тип не определён"} ·{" "}
                {STATUS[item.extraction_status] ?? item.extraction_status}
                {item.extraction_method && <> · {EXTRACTION[item.extraction_method] ?? item.extraction_method}</>}
                {item.event_id != null && <> · событие № {item.event_id}</>}
              </p>
              {item.attribution && <p className="news-attribution">{item.attribution}</p>}
            </article>
          );
        })}
      </div>

      <p className="news-foot">
        Сообщения — повод проверить территорию. В вероятность воды, ущерб и план съёмки не входят.{" "}
        <a href="/media/pipeline" target="_blank" rel="noopener noreferrer">Как модуль читает СМИ</a>
      </p>
    </aside>
  );
}
