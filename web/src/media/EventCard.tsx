import { useRef, useState } from "react";
import type { FormEvent } from "react";
import { Crosshair, ExternalLink, Satellite, ShieldCheck, X } from "lucide-react";
import { label, opsPatch, opsPost, when } from "./api";
import type { EventCard as Card, Evidence, MediaEvent, ObservationMatch } from "./types";

type Props = {
  card: Card | null;
  error: string | null;
  onClose: () => void;
  onChanged: (card: Card | null) => void;
  onOpenTerritory: (event: MediaEvent) => void;
  onOpenScene: (chipId: string) => void;
  onSelectEvent: (eventId: number) => void;
};

export default function EventCard({
  card,
  error,
  onClose,
  onChanged,
  onOpenTerritory,
  onOpenScene,
  onSelectEvent,
}: Props) {
  const sources = useRef<HTMLElement>(null);
  const [mode, setMode] = useState<"none" | "review" | "place">("none");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  if (error)
    return (
      <aside className="context-panel media-card">
        <CardHead title="Сообщение" onClose={onClose} />
        <p className="inline-error media-pad">{error}</p>
      </aside>
    );
  if (!card)
    return (
      <aside className="context-panel media-card">
        <CardHead title="Сообщение" onClose={onClose} />
        <p className="media-muted media-pad">Загружаем карточку…</p>
      </aside>
    );
  const { event } = card;
  const alert = event.report_type === "forecast" || event.report_type === "warning";
  const run = async (task: () => Promise<Card | null>) => {
    setBusy(true);
    setActionError(null);
    try {
      onChanged(await task());
      setMode("none");
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "Действие не выполнено");
    } finally {
      setBusy(false);
    }
  };
  const findObservations = () =>
    run(() => opsPost<Card>(`/api/ops/events/${event.event_id}/observations/search`));
  return (
    <aside className="context-panel media-card" aria-label="Карточка сообщения">
      <CardHead
        title={event.title}
        caption={[
          ...(["media", "bulletin", "citizen"] as const)
            .filter((origin) => event.evidence_counts[origin])
            .map((origin) => label("origin", origin)),
          ...(event.is_demo ? ["ДЕМО"] : []),
          ...(event.is_replay ? ["REPLAY"] : []),
        ].join(" · ")}
        onClose={onClose}
      />
      <div className="media-status-line">
        <span className="media-type">{label("reportType", event.report_type)}</span>
        <span className={`media-state state-${event.review_state}`}>{label("reviewState", event.review_state)}</span>
        <span className={`media-verdict verdict-${event.verification_result}`}>
          {label("verification", event.verification_result)}
        </span>
      </div>
      {alert && (
        <p className="media-callout">
          Это {event.report_type === "forecast" ? "прогноз" : "предупреждение"}, а не сообщение о
          произошедшем затоплении.
        </p>
      )}
      {event.review_state === "merged" && event.merged_into && (
        <p className="media-callout">
          Объединено с{" "}
          <button className="media-link" onClick={() => onSelectEvent(event.merged_into!)}>
            событием #{event.merged_into}
          </button>
          .
        </p>
      )}
      <div className="media-actions">
        <button onClick={() => sources.current?.scrollIntoView({ behavior: "smooth", block: "start" })}>
          Источники · {card.evidence.length}
        </button>
        <button disabled={event.geo_status !== "resolved"} onClick={() => onOpenTerritory(event)}>
          <Crosshair size={14} aria-hidden="true" /> Открыть территорию
        </button>
        <button disabled={busy || event.geo_status !== "resolved"} onClick={findObservations}>
          <Satellite size={14} aria-hidden="true" /> Найти снимки
        </button>
        <button className={mode === "review" ? "active" : ""} onClick={() => setMode(mode === "review" ? "none" : "review")}>
          <ShieldCheck size={14} aria-hidden="true" /> Проверить сообщение
        </button>
      </div>
      {actionError && <p className="inline-error media-pad">{actionError}</p>}
      {mode === "review" && <ReviewForm card={card} busy={busy} run={run} />}

      <section className="context-section">
        <h2>Где</h2>
        <p className="media-fact">
          {event.place_label ?? (event.lat != null ? `${event.lat.toFixed(4)}, ${event.lon?.toFixed(4)}` : "Место не указано")}
        </p>
        <p className="media-sub">
          Точность: {label("precision", event.geo_precision)}
          {event.geo_status === "ambiguous" && " · неоднозначно"}
          {event.geo_status === "missing" && " · координат нет"}
          {event.lat != null && ` · ${event.lat.toFixed(4)}, ${event.lon?.toFixed(4)}`}
        </p>
        {event.geo_candidates && event.geo_status !== "resolved" && (
          <div className="media-candidates">
            <small>Варианты геокодера — выберите, если уверены:</small>
            {event.geo_candidates.map((item) => (
              <button
                key={`${item.lat},${item.lon}`}
                disabled={busy}
                onClick={() =>
                  run(() =>
                    opsPatch<Card>(`/api/ops/events/${event.event_id}`, {
                      lat: item.lat,
                      lon: item.lon,
                      geo_precision: item.precision,
                      place_label: item.label.split(",").slice(0, 3).join(","),
                      note: "выбран вариант геокодера",
                    }),
                  )
                }
              >
                {item.label} <small>{label("precision", item.precision)}</small>
              </button>
            ))}
          </div>
        )}
        <button className="media-link" onClick={() => setMode(mode === "place" ? "none" : "place")}>
          {event.geo_status === "resolved" ? "Исправить место" : "Уточнить место"}
        </button>
        {mode === "place" && <PlaceForm event={event} busy={busy} run={run} />}
      </section>

      <section className="context-section">
        <h2>Когда</h2>
        <dl className="media-times">
          <dt>Событие</dt>
          <dd>
            {event.event_time_start
              ? `${when(event.event_time_start)}${event.event_time_end && event.event_time_end !== event.event_time_start ? ` — ${when(event.event_time_end)}` : ""}`
              : "неизвестно"}{" "}
            <small>точность: {label("timePrecision", event.time_precision)}</small>
          </dd>
          <dt>Первая публикация</dt>
          <dd>{when(firstPublished(card.evidence))}</dd>
          <dt>Узнала система</dt>
          <dd>{when(event.first_detected_at)}</dd>
        </dl>
      </section>

      <section className="context-section" ref={sources}>
        <h2>Источники</h2>
        <p className="media-sub">
          Независимых первоисточников: <b>{event.independent_origin_count}</b>. Перепечатки не
          считаются отдельными подтверждениями.
        </p>
        <ul className="media-evidence">
          {card.evidence.map((item) => (
            <EvidenceItem key={item.evidence_id} item={item} />
          ))}
        </ul>
      </section>

      <section className="context-section">
        <h2>Спутниковые наблюдения</h2>
        <SearchStatus card={card} />
        <ul className="media-observations">
          {card.observations.map((item) => (
            <ObservationItem key={item.match_id} item={item} onOpenScene={onOpenScene} />
          ))}
        </ul>
      </section>

      {card.merge_candidates.length > 0 && event.review_state !== "merged" && (
        <section className="context-section">
          <h2>Возможно, то же событие</h2>
          {card.merge_candidates.slice(0, 3).map((item) => (
            <div className="media-merge" key={item.event_id}>
              <button className="media-link" onClick={() => onSelectEvent(item.event_id)}>
                {item.title}
              </button>
              <small>
                {item.distance_km} км · {item.confidence === "high" ? "близко по месту и времени" : "совпадение неполное"}
              </small>
              <button
                disabled={busy}
                onClick={() =>
                  run(() =>
                    opsPost<Card>(`/api/ops/events/${event.event_id}/merge`, {
                      into_event_id: item.event_id,
                      note: "объединено аналитиком из карточки",
                    }),
                  )
                }
              >
                Объединить
              </button>
            </div>
          ))}
        </section>
      )}

      <section className="context-section">
        <h2>Чего не хватает</h2>
        <ul className="media-gaps">
          {card.gaps.map((gap) => (
            <li key={gap}>{gap}</li>
          ))}
        </ul>
      </section>

      <details className="context-section media-history">
        <summary>История · {card.history.length}</summary>
        <ol>
          {card.history.map((item, index) => (
            <li key={index}>
              <time>{when(item.at)}</time> {item.actor}: {historyLabel(item.action)}
              {item.note && <small> — {item.note}</small>}
            </li>
          ))}
        </ol>
      </details>
      <p className="context-foot">
        Сообщения — повод проверить место. Они не меняют карту вероятности воды, порог,
        объекты, ожидаемый ущерб и план A/B/C.
      </p>
    </aside>
  );
}

function SearchStatus({ card }: { card: Card }) {
  const last = [...card.history].reverse().find((item) => item.action === "observations_searched");
  if (!last)
    return <p className="media-sub">Наблюдения ещё не искали.</p>;
  const result = JSON.parse(last.after_json ?? "{}") as { found?: number; suitable?: number };
  return (
    <p className="media-sub">
      Искали {when(last.at)} в каталоге проекта (чипы Sen1Floods11 2016–2019, без test):
      найдено {result.found ?? 0}, подходящих по месту и времени {result.suitable ?? 0}.
      {!result.found && " Снимков этого места в каталоге нет."} Отсутствие снимка не
      опровергает сообщение.
    </p>
  );
}

function CardHead({ title, caption, onClose }: { title: string; caption?: string; onClose: () => void }) {
  return (
    <div className="context-heading">
      <div>
        <small>СООБЩЕНИЕ{caption ? ` · ${caption.toUpperCase()}` : ""}</small>
        <h1>{title}</h1>
      </div>
      <button onClick={onClose} aria-label="Закрыть карточку">
        <X size={17} />
      </button>
    </div>
  );
}

const firstPublished = (evidence: Evidence[]) =>
  evidence
    .map((item) => item.publication?.published_at ?? item.report?.submitted_at ?? null)
    .filter((value): value is string => !!value)
    .sort()[0] ?? null;

const historyLabel = (action: string) =>
  ({
    created: "создано",
    edited: "изменено",
    merged_into: "объединено в другое событие",
    merged_from: "присоединено другое событие",
    split: "разделено",
    split_from: "выделено из события",
    report_deleted: "обращение удалено",
    review_start: "взято на проверку",
    review_result: "записан результат проверки",
    review_dismiss: "отклонено",
    review_reopen: "возвращено в работу",
    observations_searched: "искали наблюдения",
  })[action] ?? action;

function EvidenceItem({ item }: { item: Evidence }) {
  if (item.publication) {
    const pub = item.publication;
    return (
      <li className={item.is_original ? "" : "reprint"}>
        <span className="media-origin">
          {label("origin", item.origin)} · {pub.source_name}
          {!item.is_original && " · перепечатка"}
        </span>
        <a href={pub.url} target="_blank" rel="noopener noreferrer">
          {pub.title} <ExternalLink size={12} aria-hidden="true" />
        </a>
        {pub.text_excerpt && <q>{pub.text_excerpt}</q>}
        <small>
          {pub.domain} · опубликовано {when(pub.published_at)} · получено {when(pub.fetched_at)}
          {pub.content_level === "title_only" && " · доступен только заголовок"}
          {" · "}извлечение: {pub.extraction_method}
        </small>
        {pub.attribution && <small className="media-attribution">{pub.attribution}</small>}
      </li>
    );
  }
  const report = item.report!;
  return (
    <li>
      <span className="media-origin">
        Очевидец · {report.public_id} · {report.kind_label}
      </span>
      <p className="queue-text">{report.description}</p>
      {report.photos.length > 0 && (
        <div className="queue-photos">
          {report.photos.map((photo) => (
            <a key={photo.url} href={photo.url} target="_blank" rel="noopener noreferrer">
              <img src={photo.url} alt="Фото из обращения" loading="lazy" />
            </a>
          ))}
        </div>
      )}
      <small>
        Заметили {when(report.observed_at)} ({report.observed_tz}) · отправлено {when(report.submitted_at)} · модерация: принято
      </small>
    </li>
  );
}

function ObservationItem({ item, onOpenScene }: { item: ObservationMatch; onOpenScene: (chipId: string) => void }) {
  const offset =
    item.time_offset_days == null
      ? "время неизвестно"
      : item.time_offset_days < 0
        ? `за ${Math.abs(item.time_offset_days).toFixed(1)} сут. до события`
        : `через ${item.time_offset_days.toFixed(1)} сут. после начала`;
  return (
    <li className={`suit-${item.suitability}`}>
      <span className="media-origin">
        {item.catalog === "local" ? "Каталог проекта · Sen1Floods11" : "Copernicus · каталог"} ·{" "}
        {label("suitability", item.suitability)}
      </span>
      <strong>{item.observation_ref}</strong>
      <small>
        {when(item.observation_at)} · {offset} · {item.spatial_relation === "covers" ? "покрывает место" : item.spatial_relation === "intersects" ? "частично" : `рядом, ${item.distance_km ?? "?"} км`}
      </small>
      <small>
        Этап: {label("stage", item.stage)}. {item.reason}
      </small>
      {item.catalog === "local" && (
        <button className="media-link" onClick={() => onOpenScene(item.observation_ref)}>
          Открыть сцену на карте
        </button>
      )}
    </li>
  );
}

function ReviewForm({ card, busy, run }: { card: Card; busy: boolean; run: (task: () => Promise<Card | null>) => Promise<void> }) {
  const { event } = card;
  const suitable = card.observations.some((item) => item.suitability === "suitable");
  const [result, setResult] = useState<string>(suitable ? "consistent" : "insufficient_data");
  const [basis, setBasis] = useState("");
  const [comment, setComment] = useState("");
  const [needsImagery, setNeedsImagery] = useState(event.follow_up === "needs_imagery");
  const post = (action: string, extra: Record<string, unknown> = {}) =>
    run(() =>
      opsPost<Card>(`/api/ops/events/${event.event_id}/review`, {
        action,
        basis: basis.trim() || null,
        comment: comment.trim() || null,
        follow_up: needsImagery ? "needs_imagery" : "none",
        ...extra,
      }),
    );
  const submit = (formEvent: FormEvent) => {
    formEvent.preventDefault();
    void post("result", { verification_result: result });
  };
  return (
    <form className="media-review" onSubmit={submit}>
      {event.review_state !== "under_review" && event.review_state !== "reviewed" && (
        <button type="button" disabled={busy} onClick={() => post("start")}>
          Взять на проверку
        </button>
      )}
      <fieldset>
        <legend>Результат сопоставления</legend>
        {(["not_checked", "consistent", "discrepancy", "insufficient_data"] as const).map((value) => {
          const blocked = (value === "consistent" || value === "discrepancy") && !suitable;
          return (
            <label key={value} className={blocked ? "disabled" : ""}>
              <input type="radio" name="verification" value={value} disabled={blocked} checked={result === value} onChange={() => setResult(value)} />
              {label("verification", value)}
              {blocked && <small>нужно подходящее наблюдение</small>}
            </label>
          );
        })}
      </fieldset>
      <label>
        Основание
        <input value={basis} maxLength={1000} placeholder="Какие источники и снимки смотрели" onChange={(e) => setBasis(e.target.value)} />
      </label>
      <label>
        Комментарий
        <input value={comment} maxLength={1000} onChange={(e) => setComment(e.target.value)} />
      </label>
      <label className="media-check">
        <input type="checkbox" checked={needsImagery} onChange={() => setNeedsImagery(!needsImagery)} />
        Нужна дополнительная съёмка (отметка для плана, заказ не создаётся)
      </label>
      <div className="media-review-actions">
        <button type="submit" className="media-primary" disabled={busy}>
          Сохранить результат
        </button>
        <button
          type="button"
          className="media-danger"
          disabled={busy || !comment.trim()}
          title="Укажите причину в комментарии"
          onClick={() => post("dismiss")}
        >
          Отклонить сообщение
        </button>
        {(event.review_state === "dismissed" || event.review_state === "reviewed") && (
          <button type="button" disabled={busy} onClick={() => post("reopen")}>
            Вернуть в работу
          </button>
        )}
      </div>
    </form>
  );
}

function PlaceForm({ event, busy, run }: { event: MediaEvent; busy: boolean; run: (task: () => Promise<Card | null>) => Promise<void> }) {
  const [query, setQuery] = useState(event.place_label ?? "");
  const [results, setResults] = useState<Array<{ label: string; lat: number; lon: number; precision: string }>>([]);
  const [note, setNote] = useState<string | null>(null);
  const [lat, setLat] = useState(event.lat?.toString() ?? "");
  const [lon, setLon] = useState(event.lon?.toString() ?? "");
  const search = async () => {
    setNote(null);
    const response = await fetch(`/api/public/geocode?q=${encodeURIComponent(query.trim())}`);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      setNote(data.detail ?? "Поиск недоступен");
      return;
    }
    setResults(data.results);
    if (!data.results.length) setNote("Ничего не найдено");
  };
  const save = (point: { lat: number; lon: number; precision: string; label?: string }) =>
    run(() =>
      opsPatch<Card>(`/api/ops/events/${event.event_id}`, {
        lat: point.lat,
        lon: point.lon,
        geo_precision: point.precision,
        ...(point.label ? { place_label: point.label.split(",").slice(0, 3).join(",") } : {}),
        note: "место уточнено аналитиком",
      }),
    );
  return (
    <div className="media-place-form">
      <div className="media-inline">
        <input value={query} placeholder="Населённый пункт, район" aria-label="Поиск места" onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void search(); } }} />
        <button type="button" disabled={query.trim().length < 2} onClick={search}>Найти</button>
      </div>
      {results.map((item) => (
        <button key={`${item.lat},${item.lon}`} className="media-candidate" disabled={busy} onClick={() => save(item)}>
          {item.label} <small>{label("precision", item.precision)}</small>
        </button>
      ))}
      {note && <small>{note}</small>}
      <div className="media-inline">
        <input value={lat} inputMode="decimal" placeholder="широта" aria-label="Широта" onChange={(e) => setLat(e.target.value)} />
        <input value={lon} inputMode="decimal" placeholder="долгота" aria-label="Долгота" onChange={(e) => setLon(e.target.value)} />
        <button
          type="button"
          disabled={busy}
          onClick={() => {
            const point = { lat: Number(lat.replace(",", ".")), lon: Number(lon.replace(",", ".")), precision: "point" };
            if (!Number.isFinite(point.lat) || !Number.isFinite(point.lon)) setNote("Нужны числа");
            else void save(point);
          }}
        >
          Сохранить
        </button>
      </div>
      <small className="media-attribution">© OpenStreetMap contributors · Nominatim</small>
    </div>
  );
}
