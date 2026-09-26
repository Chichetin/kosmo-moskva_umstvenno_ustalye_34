import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { ChevronLeft, LogOut, MapPinOff } from "lucide-react";
import { label, opsGet, opsPost, when } from "./api";
import type { Media } from "./useMedia";
import type { CitizenReport, MediaEvent, Origin } from "./types";

type Props = {
  media: Media;
  selectedEventId: number | null;
  onSelectEvent: (event: MediaEvent) => void;
};

const ORIGINS: Origin[] = ["media", "bulletin", "citizen"];

export default function MessagesPanel({ media, selectedEventId, onSelectEvent }: Props) {
  const [view, setView] = useState<"events" | "queue">("events");
  if (media.operator === undefined) return <p className="media-muted">Проверяем вход…</p>;
  if (media.operator === null) return <LoginForm media={media} />;
  if (view === "queue")
    return <ReportQueue media={media} onBack={() => setView("events")} onSelectEvent={onSelectEvent} />;
  const located = media.events.filter((item) => item.geo_status === "resolved");
  const unplaced = media.events.filter((item) => item.geo_status !== "resolved");
  const { filters, setFilters } = media;
  return (
    <div className="media-panel">
      <div className="media-toolbar">
        <button
          className={`media-queue-button ${media.summary?.pending_reports ? "has-items" : ""}`}
          onClick={() => setView("queue")}
        >
          Новые обращения
          <b>{media.summary?.pending_reports ?? "…"}</b>
        </button>
        <label className="media-switch">
          <input
            type="checkbox"
            checked={media.showOnMap}
            onChange={() => media.setShowOnMap(!media.showOnMap)}
          />
          На карте
        </label>
      </div>
      <div className="media-filters" aria-label="Фильтры сообщений">
        <div className="media-chips">
          {ORIGINS.map((origin) => (
            <button
              key={origin}
              aria-pressed={filters.origins[origin]}
              className={filters.origins[origin] ? "active" : ""}
              onClick={() =>
                setFilters({
                  ...filters,
                  origins: { ...filters.origins, [origin]: !filters.origins[origin] },
                })
              }
            >
              {label("origin", origin)}
            </button>
          ))}
        </div>
        <div className="media-selects">
          <select
            aria-label="Период"
            value={filters.days}
            onChange={(event) => setFilters({ ...filters, days: Number(event.target.value) })}
          >
            <option value={1}>Сутки</option>
            <option value={7}>7 дней</option>
            <option value={30}>30 дней</option>
            <option value={365}>Год</option>
          </select>
          <select
            aria-label="Тип сообщения"
            value={filters.types}
            onChange={(event) =>
              setFilters({ ...filters, types: event.target.value as typeof filters.types })
            }
          >
            <option value="situation">Ситуации</option>
            <option value="alerts">Прогнозы и предупреждения</option>
            <option value="all">Все, включая исторические</option>
          </select>
          <select
            aria-label="Состояние проверки"
            value={filters.state}
            onChange={(event) =>
              setFilters({ ...filters, state: event.target.value as typeof filters.state })
            }
          >
            <option value="open">В работе</option>
            <option value="needs_geo">Нужно место</option>
            <option value="reviewed">Проверенные</option>
            <option value="all">Все состояния</option>
          </select>
        </div>
      </div>
      <div className="media-list" role="listbox" aria-label="Сообщения">
        {media.eventsError && <p className="media-error">{media.eventsError}</p>}
        {!media.eventsError && !media.events.length && (
          <p className="media-muted">
            {media.loading
              ? "Загружаем сообщения…"
              : "По этим фильтрам сообщений нет. Это не значит, что паводков нет: смотрите состояние источников."}
          </p>
        )}
        {located.map((item) => (
          <EventRow key={item.event_id} item={item} selected={item.event_id === selectedEventId} onSelect={onSelectEvent} />
        ))}
        {unplaced.length > 0 && (
          <>
            <h3 className="media-subhead">
              <MapPinOff size={14} aria-hidden="true" /> Без точного места · {unplaced.length}
            </h3>
            {unplaced.map((item) => (
              <EventRow key={item.event_id} item={item} selected={item.event_id === selectedEventId} onSelect={onSelectEvent} />
            ))}
          </>
        )}
      </div>
      <div className="media-foot">
        <span>
          Аналитик: <b>{media.operator}</b>
        </span>
        <button onClick={media.logout} aria-label="Выйти">
          <LogOut size={14} /> Выйти
        </button>
      </div>
    </div>
  );
}

function EventRow({
  item,
  selected,
  onSelect,
}: {
  item: MediaEvent;
  selected: boolean;
  onSelect: (event: MediaEvent) => void;
}) {
  const origins = ORIGINS.filter((origin) => item.evidence_counts[origin]);
  return (
    <button
      role="option"
      aria-selected={selected}
      className={`media-row type-${item.report_type}`}
      onClick={() => onSelect(item)}
    >
      <span className="media-row-copy">
        <strong>{item.title}</strong>
        <small>
          {item.place_label ?? label("precision", item.geo_precision)} ·{" "}
          {when(item.event_time_start ?? item.first_detected_at, false)}
        </small>
        <span className="media-tags">
          {item.is_demo && <em className="tag-demo">ДЕМО</em>}
          {item.report_type !== "occurred" && <em>{label("reportType", item.report_type)}</em>}
          {origins.map((origin) => (
            <em key={origin}>
              {label("origin", origin)} {item.evidence_counts[origin]}
            </em>
          ))}
        </span>
      </span>
      <span className={`media-state state-${item.review_state}`}>{label("reviewState", item.review_state)}</span>
    </button>
  );
}

function LoginForm({ media }: { media: Media }) {
  const [token, setToken] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await media.login(token.trim(), name.trim() || "аналитик");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Вход не выполнен");
    } finally {
      setBusy(false);
    }
  };
  return (
    <form className="media-login" onSubmit={submit}>
      <p>
        Сообщения СМИ, сводки и обращения очевидцев видит только аналитик. Токен задаётся
        переменной <code>FLOODVALUE_OPERATOR_TOKEN</code> или лежит на сервере: <code>python -m floodvalue.media token</code>.
      </p>
      <label>
        Токен аналитика
        <input type="password" autoComplete="current-password" value={token} onChange={(event) => setToken(event.target.value)} />
      </label>
      <label>
        Имя в журнале проверок
        <input value={name} maxLength={60} placeholder="аналитик" onChange={(event) => setName(event.target.value)} />
      </label>
      {error && <p className="media-error">{error}</p>}
      <button className="media-primary" disabled={busy || token.trim().length < 8}>
        {busy ? "Входим…" : "Войти"}
      </button>
    </form>
  );
}

function ReportQueue({
  media,
  onBack,
  onSelectEvent,
}: {
  media: Media;
  onBack: () => void;
  onSelectEvent: (event: MediaEvent) => void;
}) {
  const [reports, setReports] = useState<CitizenReport[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    opsGet<CitizenReport[]>("/api/ops/reports?status=pending", controller.signal)
      .then((list) => {
        setReports(list);
        setError(null);
      })
      .catch((cause) => {
        if (cause.name !== "AbortError") setError(String(cause.message));
      });
    return () => controller.abort();
  }, [revision]);
  return (
    <div className="media-panel">
      <div className="media-toolbar">
        <button className="media-back" onClick={onBack}>
          <ChevronLeft size={16} /> Сообщения
        </button>
        <span className="media-muted">Очередь модерации</span>
      </div>
      <div className="media-list">
        {error && <p className="media-error">{error}</p>}
        {reports && !reports.length && <p className="media-muted">Новых обращений нет.</p>}
        {reports?.map((report) => (
          <QueueItem
            key={report.report_id}
            report={report}
            onDone={(eventId) => {
              setRevision((value) => value + 1);
              media.refresh();
              if (eventId != null)
                onSelectEvent({ event_id: eventId } as MediaEvent);
            }}
          />
        ))}
      </div>
      <p className="media-note">
        Принять — значит допустить сообщение к работе. Это не подтверждение затопления.
      </p>
    </div>
  );
}

function QueueItem({ report, onDone }: { report: CitizenReport; onDone: (eventId: number | null) => void }) {
  const [candidates, setCandidates] = useState<Array<{ event_id: number; title: string; distance_km: number; confidence: string }>>([]);
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    opsGet<typeof candidates>(`/api/ops/reports/${report.report_id}/candidates`)
      .then(setCandidates)
      .catch(() => setCandidates([]));
  }, [report.report_id]);
  const decide = async (decision: "approve" | "reject", eventId: number | null = null) => {
    if (decision === "reject" && !comment.trim()) {
      setError("Укажите причину отклонения.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await opsPost<{ event_id: number | null }>(`/api/ops/reports/${report.report_id}/review`, {
        decision,
        comment: comment.trim() || null,
        event_id: eventId,
      });
      onDone(result.event_id);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Не удалось сохранить решение");
      setBusy(false);
    }
  };
  return (
    <article className="queue-item">
      <header>
        <strong>{report.kind_label}</strong>
        <code>{report.public_id}</code>
        {report.is_demo && <em className="tag-demo">ДЕМО</em>}
      </header>
      <p className="queue-text">{report.description}</p>
      <dl>
        <dt>Где</dt>
        <dd>
          {report.lat != null
            ? `${report.lat.toFixed(4)}, ${report.lon?.toFixed(4)} · ${label("precision", report.location_precision)}`
            : "координат нет"}
          {report.place_text && <span className="queue-place">«{report.place_text}»</span>}
        </dd>
        <dt>Заметили</dt>
        <dd>
          {when(report.observed_at)} · {report.observed_time_precision === "exact" ? "точно" : report.observed_time_precision === "approx_hour" ? "примерно" : "только день"} ({report.observed_tz})
        </dd>
        <dt>Получено</dt>
        <dd>{when(report.submitted_at)}</dd>
      </dl>
      {report.photos.length > 0 && (
        <div className="queue-photos">
          {report.photos.map((photo) => (
            <a key={photo.url} href={photo.url} target="_blank" rel="noopener noreferrer">
              <img src={photo.url} alt="Фото из обращения" loading="lazy" />
            </a>
          ))}
        </div>
      )}
      <label className="queue-comment">
        Комментарий
        <input value={comment} maxLength={1000} placeholder="Основание решения" onChange={(event) => setComment(event.target.value)} />
      </label>
      {error && <p className="media-error">{error}</p>}
      <div className="queue-actions">
        <button className="media-primary" disabled={busy} onClick={() => decide("approve")}>
          Принять — новое событие
        </button>
        {candidates.slice(0, 2).map((item) => (
          <button key={item.event_id} disabled={busy} onClick={() => decide("approve", item.event_id)}>
            Привязать к «{item.title}» · {item.distance_km} км
          </button>
        ))}
        <button className="media-danger" disabled={busy} onClick={() => decide("reject")}>
          Отклонить
        </button>
      </div>
    </article>
  );
}
