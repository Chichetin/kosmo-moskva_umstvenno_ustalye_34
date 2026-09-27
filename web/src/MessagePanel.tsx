import { useEffect, useState, type ReactNode } from "react";
import { ExternalLink, LocateFixed, X } from "lucide-react";
import "./messagePanel.css";
import {
  EXTRACTION, MODERATION, ORIGIN, PRECISION, REPORT_TYPE, REVIEW_STATE, SUITABILITY, VERIFICATION,
  label, when,
} from "./mediaLabels";

/** A point or ring of the «Сообщения» layer the operator clicked. */
export type SelectedMessage = {
  kind: "event" | "report";
  id: string;
  lngLat: [number, number];
  props: Record<string, unknown>;
};

type Report = {
  report_id: string;
  category_label?: string;
  description?: string;
  place_text?: string | null;
  location_precision?: string;
  observed_at?: string | null;
  observed_precision?: string;
  submitted_at?: string;
  moderation_status?: string;
  is_demo?: number;
  lat?: number | null;
  lon?: number | null;
  photos?: string[];
};
type Evidence = {
  evidence_id: number;
  origin: string;
  is_original?: number;
  report_type?: string | null;
  title?: string | null;
  url?: string | null;
  text_excerpt?: string | null;
  source_name?: string | null;
  attribution?: string | null;
  published_at?: string | null;
  extraction_method?: string | null;
  facts?: Record<string, unknown> | null;
  time_hint?: Record<string, unknown> | null;
  report?: Report | null;
  photos?: string[];
};
type Observation = {
  match_id: number;
  catalog: string;
  observation_ref: string;
  product_type?: string | null;
  observation_start?: string | null;
  suitability: string;
  reason: string;
};
type EventCard = {
  exact_locations?: boolean; // false: the public card without login (sources only, coarse place)
  event: {
    event_id: number;
    title: string;
    report_type: string;
    place_label?: string | null;
    geo_precision: string;
    event_time_start?: string | null;
    event_time_end?: string | null;
    time_precision: string;
    first_detected_at: string;
    review_state: string;
    verification_result: string;
    independent_origin_count: number;
    source_kinds?: string | null;
    is_demo: number;
    is_replay: number;
  };
  evidence: Evidence[];
  observations: Observation[];
  missing?: string[];
};
type Loaded =
  | { status: "loading" }
  | { status: "locked" }
  | { status: "error"; text: string }
  | { status: "event"; card: EventCard }
  | { status: "report"; report: Report };

const ZOOM: Record<string, number> = {
  point: 12, street: 12, coarsened: 11, locality: 10, district: 9, region: 6, country: 5,
};
const NOT_IN_MODEL = "Сообщение не проверено по снимкам и не входит в p, ущерб и план.";
const FACT_LABEL: Record<string, string> = {
  alert_level: "уровень тревоги", glide: "GLIDE", provider: "поставщик данных",
  severity: "серьёзность", country: "страна",
};
const SHOWN_FACTS = new Set(["method", "place_mention", "quotes", "reasons", "reprint"]);

async function load(message: SelectedMessage, signal: AbortSignal): Promise<Loaded> {
  const get = (url: string) => fetch(url, { signal, credentials: "same-origin" });
  const locked = (r: Response) => r.status === 401 || r.status === 403;
  if (message.kind === "event") {
    const r = await get(`/api/media/events/${encodeURIComponent(message.id)}`);
    if (locked(r)) return { status: "locked" };
    if (!r.ok) return { status: "error", text: `Не удалось загрузить сообщение (${r.status}).` };
    return { status: "event", card: (await r.json()) as EventCard };
  }
  const r = await get(`/api/media/reports/${encodeURIComponent(message.id)}`);
  if (locked(r)) return { status: "locked" };
  if (r.ok) {
    const body = (await r.json()) as { report?: Report; photos?: string[] } & Report;
    const report = body.report ?? body;
    return { status: "report", report: { ...report, photos: report.photos ?? body.photos ?? [] } };
  }
  // Older backend without the single-report route: look the report up in the moderation queues.
  for (const status of ["pending", "accepted", "rejected"]) {
    const q = await get(`/api/media/reports?status=${status}`);
    if (locked(q)) return { status: "locked" };
    if (!q.ok) continue;
    const found = ((await q.json()) as { reports: Report[] }).reports.find((x) => x.report_id === message.id);
    if (found) return { status: "report", report: found };
  }
  return { status: "error", text: "Обращение не найдено: его могли удалить по сроку хранения." };
}

export default function MessagePanel({
  message,
  onClose,
  onZoom,
}: {
  message: SelectedMessage;
  onClose: () => void;
  onZoom: (lngLat: [number, number], zoom: number) => void;
}) {
  const [loaded, setLoaded] = useState<Loaded>({ status: "loading" });
  const [photo, setPhoto] = useState<string | null>(null);

  useEffect(() => {
    const abort = new AbortController();
    setLoaded({ status: "loading" });
    setPhoto(null);
    load(message, abort.signal)
      .then(setLoaded)
      .catch((error: unknown) => {
        if (!abort.signal.aborted) setLoaded({ status: "error", text: `Нет связи с сервером: ${String(error)}` });
      });
    return () => abort.abort();
  }, [message.kind, message.id]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (photo) setPhoto(null);
      else onClose();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [photo, onClose]);

  const p = message.props;
  const card = loaded.status === "event" ? loaded.card : null;
  const report = loaded.status === "report" ? loaded.report : null;
  const demo = card ? card.event.is_demo : Number(p.is_demo ?? 0);
  const replay = card ? card.event.is_replay : Number(p.is_replay ?? 0);
  const precision = String(card?.event.geo_precision ?? report?.location_precision ?? p.geo_precision ?? p.location_precision ?? "");
  const title = message.kind === "report"
    ? report?.category_label ?? String(p.category_label ?? "Обращение очевидца")
    : card?.event.title ?? String(p.title ?? "Сообщение");

  return (
    <aside className="message-panel" aria-label="Содержание сообщения">
      <header className="mp-head">
        <div>
          {!!demo && <p className="mp-demo">{replay ? "ДЕМО · REPLAY" : "ДЕМО"}</p>}
          <small className="mp-kicker">
            {message.kind === "report" ? "Обращение очевидца" : `Событие · ${originText(card?.event.source_kinds ?? p.source_kinds)}`}
          </small>
          <h1>{title}</h1>
        </div>
        <button className="mp-close" onClick={onClose} aria-label="Закрыть">
          <X size={17} />
        </button>
      </header>

      <div className="mp-actions">
        <button onClick={() => onZoom(message.lngLat, ZOOM[precision] ?? 9)}>
          <LocateFixed size={16} />
          Приблизить к месту
        </button>
      </div>

      <p className="mp-note">{NOT_IN_MODEL}</p>

      {loaded.status === "loading" && <p className="mp-state">Загружаем содержание…</p>}
      {loaded.status === "error" && <p className="mp-state mp-error">{loaded.text}</p>}
      {loaded.status === "locked" && (
        <p className="mp-state">
          Сообщение известно только от очевидцев: их текст и фото — персональные данные,
          они видны после входа ·{" "}
          <a href="/media/analyst" target="_blank" rel="noopener noreferrer">Войти</a>
        </p>
      )}
      {card?.exact_locations === false && (
        <p className="mp-state">
          Показано то, что опубликовали источники; место — с точностью ≈ 1 км, время — до часа.
          Точные данные и сообщения очевидцев — после{" "}
          <a href="/media/analyst" target="_blank" rel="noopener noreferrer">входа</a>.
        </p>
      )}
      {card && <EventBody card={card} onPhoto={setPhoto} />}
      {report && <ReportBody report={report} onPhoto={setPhoto} />}

      {photo && (
        <div className="mp-lightbox" role="dialog" aria-label="Фото очевидца" onClick={() => setPhoto(null)}>
          <img src={`/api/media/attachments/${encodeURIComponent(photo)}`} alt="Фото очевидца" />
          <button aria-label="Закрыть фото">
            <X size={20} />
          </button>
        </div>
      )}
    </aside>
  );
}

// «1 независимый», «2 / 5 / 11 независимых»
function independent(n: number) {
  return `${n} ${n % 10 === 1 && n % 100 !== 11 ? "независимый" : "независимых"}`;
}

function originText(kinds: unknown) {
  return String(kinds ?? "").split(",").filter(Boolean).map((k) => ORIGIN[k] ?? k).join(", ") || "источник неизвестен";
}

function Row({ name, children }: { name: string; children: ReactNode }) {
  return (
    <>
      <dt>{name}</dt>
      <dd>{children}</dd>
    </>
  );
}

function EventBody({ card, onPhoto }: { card: EventCard; onPhoto: (id: string) => void }) {
  const e = card.event;
  const period = e.event_time_end && e.event_time_end !== e.event_time_start
    ? `${when(e.event_time_start, e.time_precision, false)} — ${when(e.event_time_end, e.time_precision, false)}`
    : when(e.event_time_start ?? e.first_detected_at, e.event_time_start ? e.time_precision : "hour", false);
  return (
    <>
      <section className="mp-section">
        <dl className="mp-facts">
          <Row name="Что">{label(REPORT_TYPE, e.report_type)}</Row>
          <Row name="Где">{e.place_label ?? "место не определено"} · точность: {label(PRECISION, e.geo_precision)}</Row>
          <Row name="Когда">{period}</Row>
          <Row name="Проверка">{label(REVIEW_STATE, e.review_state)} · {label(VERIFICATION, e.verification_result)}</Row>
          <Row name="Источники">{independent(e.independent_origin_count)}</Row>
        </dl>
        {!!card.missing?.length && (
          <ul className="mp-missing">
            {card.missing.map((text) => <li key={text}>{text}</li>)}
          </ul>
        )}
      </section>

      <section className="mp-section">
        <h2>Источники · {card.evidence.length}</h2>
        {card.evidence.map((ev) => <Source key={ev.evidence_id} ev={ev} place={e.place_label} onPhoto={onPhoto} />)}
      </section>

      <section className="mp-section">
        <h2>Подходящие снимки</h2>
        {card.observations.length ? (
          <ul className="mp-list">
            {card.observations.map((o) => (
              <li key={o.match_id}>
                <strong>{o.catalog} · {o.product_type ?? o.observation_ref}</strong>
                <span>{when(o.observation_start, "hour", false)} · {label(SUITABILITY, o.suitability)}</span>
                <small>{o.reason}</small>
              </li>
            ))}
          </ul>
        ) : (
          <p className="mp-muted">Снимки для этого сообщения не искали или не нашли — это не опровержение.</p>
        )}
      </section>
    </>
  );
}

function Source({ ev, place, onPhoto }: { ev: Evidence; place?: string | null; onPhoto: (id: string) => void }) {
  const facts = ev.facts ?? {};
  const quotes = Array.isArray(facts.quotes) ? facts.quotes.map(String) : [];
  const reasons = Array.isArray(facts.reasons) ? facts.reasons.map(String) : [];
  const other = Object.entries(facts).filter(([k, v]) => !SHOWN_FACTS.has(k) && v !== null && typeof v !== "object");
  const hint = ev.time_hint ?? {};
  const timeText = typeof hint.text === "string" ? hint.text : typeof hint.time_basis === "string" ? hint.time_basis : null;
  // A structured feed (GDACS) gives place and period as fields, not as words in a text.
  const structured = ev.extraction_method === "source_structured";
  return (
    <article className="mp-source">
      <small className="mp-kicker">
        {label(ORIGIN, ev.origin)}
        {ev.source_name ? ` · ${ev.source_name}` : ""}
        {ev.published_at ? ` · ${when(ev.published_at, "hour", false)}` : ""}
      </small>
      {ev.report ? (
        <ReportBody report={{ ...ev.report, photos: ev.photos }} onPhoto={onPhoto} nested />
      ) : (
        <>
          {ev.url ? (
            <a className="mp-title" href={ev.url} target="_blank" rel="noopener noreferrer">
              {ev.title ?? ev.url}
              <ExternalLink size={13} />
            </a>
          ) : (
            <strong className="mp-title">{ev.title ?? "без заголовка"}</strong>
          )}
          {ev.text_excerpt && <blockquote>{ev.text_excerpt}</blockquote>}
        </>
      )}
      <div className="mp-understood">
        <b>Как система поняла</b>
        <dl className="mp-facts">
          <Row name="Тип">{label(REPORT_TYPE, ev.report_type)}</Row>
          <Row name="Место">
            {typeof facts.place_mention === "string" ? facts.place_mention
              : structured ? `из полей источника${place ? ` — ${place}` : ""}` : "не названо"}
          </Row>
          <Row name="Время">{timeText ?? (structured ? "из полей источника — период выше" : "в тексте не указано")}</Row>
          <Row name="Как">
            {label(EXTRACTION, ev.extraction_method, ev.origin === "citizen" ? "форма очевидца" : "—")}
            {reasons.length ? ` · ${reasons.join("; ")}` : ""}
          </Row>
          {other.map(([k, v]) => <Row key={k} name={FACT_LABEL[k] ?? k}>{String(v)}</Row>)}
        </dl>
        {quotes.map((q) => <q key={q}>{q}</q>)}
        {facts.reprint === true && <small>Перепечатка: как отдельный источник не считается.</small>}
      </div>
      {ev.attribution && <small className="mp-attribution">{ev.attribution}</small>}
    </article>
  );
}

function ReportBody({ report, onPhoto, nested = false }: { report: Report; onPhoto: (id: string) => void; nested?: boolean }) {
  const coarse = report.location_precision === "coarsened";
  return (
    <section className={nested ? "mp-report" : "mp-section"}>
      {report.description ? <blockquote className="mp-full">{report.description}</blockquote> : <p className="mp-muted">Текст обращения недоступен.</p>}
      <dl className="mp-facts">
        {nested && report.category_label && <Row name="Что">{report.category_label}</Row>}
        <Row name="Где">
          {report.place_text || "место не названо"} · точность: {label(PRECISION, report.location_precision)}
          {typeof report.lat === "number" && typeof report.lon === "number" && !coarse
            ? ` · ${report.lat.toFixed(4)}, ${report.lon.toFixed(4)}`
            : ""}
        </Row>
        <Row name="Когда видели">{when(report.observed_at, report.observed_precision, coarse)}</Row>
        <Row name="Отправлено">{when(report.submitted_at, "hour", false)}</Row>
        {report.moderation_status && <Row name="Статус">{label(MODERATION, report.moderation_status)}</Row>}
      </dl>
      {!!report.photos?.length && (
        <div className="mp-photos">
          {report.photos.map((id) => (
            <button key={id} onClick={() => onPhoto(id)} aria-label="Открыть фото крупно">
              <img src={`/api/media/attachments/${encodeURIComponent(id)}`} alt="Фото очевидца" loading="lazy" />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
