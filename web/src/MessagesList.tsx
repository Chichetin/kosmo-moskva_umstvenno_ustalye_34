import { useEffect, useMemo } from "react";
import { X } from "lucide-react";
import type { Feature, MediaMap } from "./types";
import type { SelectedMessage } from "./MessagePanel";
import { ORIGIN, REPORT_TYPE, when } from "./mediaLabels";
import "./messagesList.css";

function location(feature: Feature): [number, number] | null {
  if (feature.geometry.type === "Point") {
    const [lon, lat] = feature.geometry.coordinates as number[];
    return Number.isFinite(lon) && Number.isFinite(lat) ? [lon, lat] : null;
  }
  if (feature.geometry.type === "Polygon") {
    const ring = (feature.geometry.coordinates as number[][][])[0];
    if (!ring?.length) return null;
    const points = ring.filter(([lon, lat]) => Number.isFinite(lon) && Number.isFinite(lat));
    if (!points.length) return null;
    return [
      (Math.min(...points.map(([lon]) => lon)) + Math.max(...points.map(([lon]) => lon))) / 2,
      (Math.min(...points.map(([, lat]) => lat)) + Math.max(...points.map(([, lat]) => lat))) / 2,
    ];
  }
  return null;
}

export default function MessagesList({ messages, onClose, onSelect }: {
  messages: MediaMap | null;
  onClose: () => void;
  onSelect: (message: SelectedMessage) => void;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [onClose]);

  const items = useMemo(() => [
    ...(messages?.events.features ?? []).map((feature) => ({ kind: "event" as const, feature })),
    ...(messages?.reports.features ?? []).map((feature) => ({ kind: "report" as const, feature })),
  ].sort((a, b) => {
    const time = (item: typeof a) => String(item.feature.properties.event_time_start
      ?? item.feature.properties.observed_at ?? item.feature.properties.first_detected_at
      ?? item.feature.properties.submitted_at ?? "");
    return time(b).localeCompare(time(a));
  }), [messages]);

  return (
    <aside className="messages-list" role="dialog" aria-label="Сообщения на карте">
      <div className="messages-list-head">
        <div>
          <small>МОНИТОРИНГ ТЕРРИТОРИЙ</small>
          <h2>Сообщения</h2>
          {messages && <p>{items.length} на карте · выберите место</p>}
        </div>
        <button onClick={onClose} aria-label="Закрыть сообщения"><X size={17} /></button>
      </div>
      <div className="messages-list-body">
        {!messages && <p className="messages-list-empty">Сообщения пока недоступны.</p>}
        {messages && !items.length && <p className="messages-list-empty">Сообщений с местом на карте пока нет.</p>}
        {items.map(({ kind, feature }) => {
          const props = feature.properties;
          const lngLat = location(feature);
          const id = kind === "event" ? props.event_id : props.report_id;
          if (!lngLat || id == null) return null;
          const demo = Boolean(props.is_demo);
          const title = kind === "event" ? String(props.title ?? "Сообщение СМИ") : String(props.category_label ?? "Обращение очевидца");
          const place = kind === "event" ? props.place_label : null;
          const time = kind === "event" ? props.event_time_start ?? props.first_detected_at : props.observed_at ?? props.submitted_at;
          const precision = kind === "event" ? props.time_precision : props.observed_precision;
          return (
            <button
              key={`${kind}-${id}`}
              className={`messages-list-item ${kind === "event" ? "news" : "report"}`}
              onClick={() => onSelect({ kind, id: String(id), lngLat, props })}
            >
              <span className="messages-list-meta">
                <span>{kind === "event" ? (ORIGIN[String(props.source_kinds ?? "")] ?? "Сообщение СМИ") : "Очевидец"}</span>
                {demo && <em>ДЕМО</em>}
              </span>
              <strong>{title}</strong>
              <span className="messages-list-summary">
                {kind === "event" ? (REPORT_TYPE[String(props.report_type ?? "")] ?? "Событие") : "Ожидает проверки"}
                {place ? ` · ${place}` : ""}
              </span>
              <small>{when(time, precision, props.geo_precision === "coarsened" || props.location_precision === "coarsened")}</small>
            </button>
          );
        })}
      </div>
      <p className="messages-list-foot">Сообщения — повод проверить территорию; они не входят в расчёт ущерба.</p>
    </aside>
  );
}
