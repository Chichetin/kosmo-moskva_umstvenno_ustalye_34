import { useEffect, useMemo, useRef, useState } from "react";
import { ExternalLink, LocateFixed, Satellite, X } from "lucide-react";
import { EXTRACTION, ORIGIN, PRECISION, REPORT_TYPE, SUITABILITY } from "./mediaLabels";
import type { SceneSummary } from "./types";
import "./newsPanel.css";

// Everything the media monitor has read, so the operator (and the jury) can see which news and
// bulletins the «Сообщения» layer comes from, jump to the place on the map and see whether a
// satellite scene of that territory is in the catalogue. Read-only: /api/media/news of the media
// module (make site), refreshed every minute. Titles and excerpts are third-party text, shown as
// text only.

type Place = {
  label?: string | null;
  center?: [number, number] | null;
  bbox?: [number, number, number, number] | null;
  precision?: string | null;
};
type CatalogObservation = {
  catalog: string;
  observation_ref: string;
  product_type?: string | null;
  observation_start?: string | null;
  suitability: string;
};
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
  event_time_start?: string | null;
  event_time_end?: string | null;
  place?: Place | null;
  observations?: CatalogObservation[];
};
type Feed = { publications?: Publication[]; sources_polled_at?: string | null };
type Filter = "all" | "real" | "demo";

const REFRESH_MS = 60_000;
const NEAR_KM = 60;  // a scene this close to the place is shown as «снимок рядом»
const SAME_EVENT_DAYS = 15;  // scene date this close to the news → a scene of the same event

const STATUS: Record<string, string> = {
  pending: "ждёт разбора",
  ok: "разобрано",
  needs_review: "на ручной проверке",
  invalid: "не удалось разобрать",
  irrelevant: "не о паводке",
};
const ZOOM: Record<string, number> = {
  point: 11, street: 11, coarsened: 10, locality: 10, district: 9, region: 6, country: 5,
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
const clock = (value?: string | number | Date | null) => {
  if (value == null || value === "") return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? null
    : new Intl.DateTimeFormat("ru-RU", { hour: "2-digit", minute: "2-digit" }).format(parsed);
};

const km = (a: [number, number], b: [number, number]) => {
  const rad = Math.PI / 180;
  const dLat = (b[1] - a[1]) * rad;
  const dLon = (b[0] - a[0]) * rad;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * rad) * Math.cos(b[1] * rad) * Math.sin(dLon / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(Math.min(1, h)));
};
const sceneCenter = (scene: SceneSummary): [number, number] | null =>
  scene.bounds
    ? [(scene.bounds[0] + scene.bounds[2]) / 2, (scene.bounds[1] + scene.bounds[3]) / 2]
    : scene.position ? [scene.position.lon, scene.position.lat] : null;
const inside = (scene: SceneSummary, [lon, lat]: [number, number]) =>
  !!scene.bounds && lon >= scene.bounds[0] && lon <= scene.bounds[2] && lat >= scene.bounds[1] && lat <= scene.bounds[3];

type NearScene = { scene: SceneSummary; distance: number; sameEvent: boolean };

function nearScenes(item: Publication, scenes: SceneSummary[]): NearScene[] {
  const center = item.place?.center;
  if (!center) return [];
  const when = new Date(item.event_time_start ?? item.published_at ?? item.fetched_at).getTime();
  const found: NearScene[] = [];
  for (const scene of scenes) {
    const c = sceneCenter(scene);
    if (!c) continue;
    const distance = inside(scene, center) ? 0 : km(center, c);
    if (distance > NEAR_KM) continue;
    const observed = scene.observation_date ? new Date(scene.observation_date).getTime() : NaN;
    const sameEvent = Number.isFinite(when) && Number.isFinite(observed)
      && Math.abs(observed - when) <= SAME_EVENT_DAYS * 86_400_000;
    found.push({ scene, distance, sameEvent });
  }
  // analysed scenes of the same event first, then the closest
  return found.sort((a, b) =>
    Number(b.sameEvent) - Number(a.sameEvent)
    || Number(b.scene.status === "analyzed") - Number(a.scene.status === "analyzed")
    || a.distance - b.distance).slice(0, 2);
}

export default function NewsPanel({
  onClose, scenes, onShowOnMap, onOpenScene,
}: {
  onClose: () => void;
  scenes: SceneSummary[];
  onShowOnMap: (lngLat: [number, number], zoom: number) => void;
  onOpenScene: (scene: SceneSummary) => void;
}) {
  const [items, setItems] = useState<Publication[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [floodOnly, setFloodOnly] = useState(true);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const [polledAt, setPolledAt] = useState<string | null>(null);
  const [fresh, setFresh] = useState<Set<number>>(new Set());
  const seen = useRef<Set<number> | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch("/api/media/news?limit=200", { credentials: "same-origin" })
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
          return response.json() as Promise<Feed>;
        })
        .then((data) => {
          if (!alive) return;
          const list = data.publications ?? [];
          const ids = new Set(list.map((item) => item.publication_id));
          if (seen.current) {
            const added = list.filter((item) => !seen.current!.has(item.publication_id)).map((item) => item.publication_id);
            if (added.length) setFresh((value) => new Set([...value, ...added]));
          }
          seen.current = ids;
          setItems(list);
          setPolledAt(data.sources_polled_at ?? null);
          setCheckedAt(new Date());
          setError(null);
        })
        .catch((reason: unknown) => {
          // keep the list we already have; only a first failure replaces it with the message
          if (alive && !seen.current) setError(reason instanceof Error ? reason.message : String(reason));
        });
    load();
    const timer = window.setInterval(load, REFRESH_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);

  const aboutFloods = (item: Publication) => item.extraction_status !== "irrelevant" && item.extraction_status !== "invalid";
  const demoCount = items?.filter((item) => Boolean(item.is_demo)).length ?? 0;
  const realCount = (items?.length ?? 0) - demoCount;
  const offTopic = items?.filter((item) => !aboutFloods(item)).length ?? 0;
  const shown = useMemo(() => (items ?? []).filter((item) =>
    (filter === "all" ? true : filter === "demo" ? Boolean(item.is_demo) : !item.is_demo)
    && (!floodOnly || aboutFloods(item))), [items, filter, floodOnly]);
  const polled = clock(polledAt);
  const checked = clock(checkedAt);

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
          {checked && (
            <p className="news-live" aria-live="polite">
              <span className="news-live-dot" aria-hidden="true" />
              обновляется каждую минуту · проверено в {checked}
              {polled && <> · источники опрошены в {polled}</>}
            </p>
          )}
        </div>
        <button onClick={onClose} aria-label="Закрыть новости"><X size={17} /></button>
      </div>

      {demoCount > 0 && (
        <p className="news-demo-banner">
          <strong>ДЕМО</strong> — {demoCount} тестовых сообщений: имитация прошлых паводков на
          территориях наших снимков (Шри-Ланка 2017, Ассам 2016, Нигерия 2018), чтобы показать путь
          «новость → место на карте → снимок». Это не реальные публикации.
        </p>
      )}

      {items && items.length > 0 && (
        <div className="news-filter" role="group" aria-label="Какие сообщения показать">
          {([["all", "Все"], ["real", "Реальные"], ["demo", "Демо"]] as const).map(([key, label]) => (
            <button key={key} className={filter === key ? "active" : ""} onClick={() => setFilter(key)}>
              {label}
            </button>
          ))}
          {offTopic > 0 && (
            <button className={floodOnly ? "" : "active"} onClick={() => setFloodOnly(!floodOnly)}
              aria-pressed={!floodOnly} title="Публикации источников, которые разбор счёл не относящимися к паводку">
              не о паводке · {offTopic}
            </button>
          )}
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
          const url = item.is_demo ? null : safeUrl(item.url);  // demo links point to demo.invalid
          const center = item.place?.center ?? null;
          const zoom = ZOOM[item.place?.precision ?? ""] ?? 8;
          const near = nearScenes(item, scenes);
          return (
            <article key={item.publication_id} className={`news-item ${fresh.has(item.publication_id) ? "fresh" : ""}`}>
              <div className="news-meta">
                {fresh.has(item.publication_id) && <span className="news-badge fresh">НОВОЕ</span>}
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
              {(center || item.place?.label) && (
                <div className="news-place">
                  <span>
                    {item.place?.label ?? "место из сообщения"}
                    {item.place?.precision && item.place.precision !== "unknown" && (
                      <> · {PRECISION[item.place.precision] ?? item.place.precision}</>
                    )}
                  </span>
                  {center && (
                    <button type="button" onClick={() => onShowOnMap(center, near.length ? Math.min(zoom, 9) : zoom)}
                      title="Приблизить карту к месту из сообщения">
                      <LocateFixed size={14} /> На карте
                    </button>
                  )}
                </div>
              )}
              {center && (
                near.length ? near.map(({ scene, distance, sameEvent }) => (
                  <div key={scene.chip_id} className={`news-scene ${sameEvent ? "same" : ""}`}>
                    <Satellite size={14} aria-hidden="true" />
                    <span>
                      <strong>Снимок Sentinel-1 · {scene.territory} · {day(scene.observation_date)}</strong>
                      <small>
                        {sameEvent ? "снят в дни этого паводка" : "архивный снимок территории — не этого паводка"}
                        {" · "}{distance < 1 ? "место внутри снимка" : `≈ ${Math.round(distance)} км`}
                        {" · "}{scene.status === "analyzed" ? "анализ готов" : "только снимок"}
                      </small>
                    </span>
                    <button type="button" onClick={() => onOpenScene(scene)}>Открыть</button>
                  </div>
                )) : (
                  <p className="news-scene none">Снимка этой территории в каталоге сцен стенда нет — повод заказать съёмку.</p>
                )
              )}
              {item.observations && item.observations.length > 0 && (
                <ul className="news-catalog">
                  {item.observations.map((o) => (
                    <li key={`${o.catalog}-${o.observation_ref}`}>
                      {o.catalog}{o.product_type ? ` · ${o.product_type}` : ""} · {day(o.observation_start)} ·{" "}
                      {SUITABILITY[o.suitability] ?? o.suitability}
                    </li>
                  ))}
                </ul>
              )}
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
