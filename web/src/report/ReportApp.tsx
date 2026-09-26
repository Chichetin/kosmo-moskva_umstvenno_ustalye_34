import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import { Camera, Crosshair, MapPin, Search, X } from "lucide-react";
import PlacePicker, { type Point } from "./PlacePicker";

const KINDS: Array<[string, string]> = [
  ["water_on_road", "Вода на дороге"],
  ["flooded_building", "Подтоплен дом или двор"],
  ["rising_water", "Быстро поднимается вода"],
  ["infrastructure", "Пострадала инфраструктура"],
  ["other", "Другое"],
];
const PRECISION: Record<string, string> = {
  point: "точка",
  street: "улица",
  locality: "населённый пункт",
  district: "район",
  region: "регион",
  country: "страна",
  unknown: "неизвестно",
};
const MAX_PHOTOS = 3;
const MAX_PHOTO_BYTES = 8 * 1024 * 1024;
const PHOTO_TYPES = ["image/jpeg", "image/png", "image/webp"];
const DRAFT_KEY = "fv-report-draft";
const RECEIPTS_KEY = "fv-report-receipts";

type Place = Point & { precision: string; source: "map" | "search" | "gps" | "manual" };
type Draft = {
  clientKey: string;
  kind: string;
  description: string;
  place: Place | null;
  placeText: string;
  observedLocal: string;
  timePrecision: "exact" | "approx_hour" | "approx_day";
};
type Receipt = {
  public_id: string;
  status_token: string;
  submitted_at: string;
  retention_until: string;
  duplicate: boolean;
};
type Status = {
  public_id: string;
  submitted_at: string;
  observed_at: string;
  kind: string;
  status: string;
  status_text: string;
  photos: number;
  retention_until: string;
};
type GeoResult = {
  label: string;
  lat: number;
  lon: number;
  precision: string;
  bbox: [number, number, number, number];
};

const zone = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
const pad = (value: number) => String(value).padStart(2, "0");
const localNow = () => {
  const now = new Date();
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}`;
};
const offsetLabel = () => {
  const minutes = -new Date().getTimezoneOffset();
  const sign = minutes >= 0 ? "+" : "−";
  const abs = Math.abs(minutes);
  return `UTC${sign}${Math.floor(abs / 60)}${abs % 60 ? `:${pad(abs % 60)}` : ""}`;
};
const newKey = () =>
  typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) =>
        b.toString(16).padStart(2, "0"),
      ).join("");
const emptyDraft = (): Draft => ({
  clientKey: newKey(),
  kind: "",
  description: "",
  place: null,
  placeText: "",
  observedLocal: localNow(),
  timePrecision: "approx_hour",
});
// Storage can be unavailable (private mode); the form keeps working without it.
const readStore = <T,>(key: string, fallback: T): T => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
};
const writeStore = (key: string, value: unknown) => {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
};
const formatDate = (value: string) =>
  new Intl.DateTimeFormat("ru-RU", {
    day: "numeric",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
const statusToken = () => /#status=([0-9a-f]{40})/.exec(location.hash)?.[1] ?? null;

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail ?? `Ошибка ${response.status}`);
  return data as T;
}

export default function ReportApp() {
  const [token, setToken] = useState(statusToken);
  useEffect(() => {
    const onHash = () => setToken(statusToken());
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);
  return (
    <div className="report-page">
      <header className="report-head">
        <a className="report-brand" href="/report/" aria-label="Меридиан — сообщить о подтоплении">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M3.5 18.5V5.5L12 14l8.5-8.5v13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Меридиан
        </a>
      </header>
      <main className="report-main">
        {token ? <StatusView token={token} /> : <ReportForm />}
      </main>
    </div>
  );
}

function ReportForm() {
  const [draft, setDraft] = useState<Draft>(() => {
    const saved = readStore<Draft | null>(DRAFT_KEY, null);
    return saved?.clientKey ? { ...emptyDraft(), ...saved } : emptyDraft();
  });
  const [photos, setPhotos] = useState<File[]>([]);
  const [photoError, setPhotoError] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [attempted, setAttempted] = useState(false);
  const [sending, setSending] = useState(false);
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<GeoResult[] | null>(null);
  const [searchNote, setSearchNote] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ point: Point; zoom: number } | null>(null);
  const [manualOpen, setManualOpen] = useState(false);
  const [manual, setManual] = useState({ lat: "", lon: "" });
  const [locating, setLocating] = useState(false);
  const errorBox = useRef<HTMLDivElement>(null);
  const previews = useMemo(() => photos.map((file) => URL.createObjectURL(file)), [photos]);
  useEffect(() => () => previews.forEach((url) => URL.revokeObjectURL(url)), [previews]);
  useEffect(() => writeStore(DRAFT_KEY, draft), [draft]);
  const update = (patch: Partial<Draft>) => setDraft((value) => ({ ...value, ...patch }));
  const setPlace = (place: Place | null) => update({ place });

  const search = async () => {
    if (query.trim().length < 2) return;
    setSearching(true);
    setSearchNote(null);
    try {
      const response = await fetch(`/api/public/geocode?q=${encodeURIComponent(query.trim())}`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail ?? "Поиск недоступен");
      setResults(data.results);
      if (!data.results.length) setSearchNote("Ничего не нашлось. Уточните название или отметьте точку на карте.");
      else if (data.results.length > 1) setSearchNote("Нашлось несколько мест — выберите нужное.");
    } catch (error) {
      setResults(null);
      setSearchNote(error instanceof Error ? error.message : "Поиск недоступен");
    } finally {
      setSearching(false);
    }
  };
  const locate = () => {
    if (!navigator.geolocation) {
      setSearchNote("Браузер не даёт определить местоположение.");
      return;
    }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const point = {
          lat: Number(position.coords.latitude.toFixed(5)),
          lon: Number(position.coords.longitude.toFixed(5)),
        };
        setPlace({ ...point, precision: "point", source: "gps" });
        setFocus({ point, zoom: 14 });
        setSearchNote("Отмечено ваше текущее положение. Если вода в другом месте — передвиньте точку.");
        setLocating(false);
      },
      () => {
        setSearchNote("Местоположение не получено. Отметьте место на карте или найдите его поиском.");
        setLocating(false);
      },
      { enableHighAccuracy: false, timeout: 10000, maximumAge: 60000 },
    );
  };
  const applyManual = () => {
    const lat = Number(manual.lat.replace(",", "."));
    const lon = Number(manual.lon.replace(",", "."));
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) {
      setSearchNote("Широта от −90 до 90, долгота от −180 до 180.");
      return;
    }
    if (lat === 0 && lon === 0) {
      setSearchNote("Координаты 0, 0 похожи на пустое значение.");
      return;
    }
    setPlace({ lat, lon, precision: "point", source: "manual" });
    setFocus({ point: { lat, lon }, zoom: 13 });
    setSearchNote(null);
  };
  const addPhotos = (files: FileList | null) => {
    setPhotoError(null);
    const incoming = Array.from(files ?? []);
    const next = [...photos];
    for (const file of incoming) {
      if (next.length >= MAX_PHOTOS) {
        setPhotoError(`Не больше ${MAX_PHOTOS} фото.`);
        break;
      }
      if (!PHOTO_TYPES.includes(file.type)) {
        setPhotoError(`«${file.name}»: подходят только JPEG, PNG или WebP.`);
        continue;
      }
      if (file.size > MAX_PHOTO_BYTES) {
        setPhotoError(`«${file.name}» больше 8 МБ.`);
        continue;
      }
      next.push(file);
    }
    setPhotos(next);
  };

  const validate = () => {
    const found: string[] = [];
    if (!draft.kind) found.push("Выберите, что произошло.");
    if (draft.description.trim().length < 10) found.push("Опишите ситуацию хотя бы парой слов (от 10 символов).");
    if (!draft.place && draft.placeText.trim().length < 3)
      found.push("Отметьте место на карте, найдите его или опишите словами.");
    const observed = new Date(draft.observedLocal);
    if (Number.isNaN(observed.getTime())) found.push("Укажите, когда вы это заметили.");
    else if (observed.getTime() > Date.now() + 10 * 60 * 1000) found.push("Время наблюдения не может быть в будущем.");
    else if (observed.getTime() < Date.now() - 30 * 24 * 3600 * 1000)
      found.push("Форма принимает наблюдения не старше 30 дней.");
    return found;
  };

  // After the first attempt the error list follows the fields as they are corrected.
  useEffect(() => {
    if (attempted) setErrors(validate());
  }, [draft]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (sending) return;
    setAttempted(true);
    const found = validate();
    setErrors(found);
    if (found.length) {
      requestAnimationFrame(() => errorBox.current?.focus());
      return;
    }
    const payload = {
      client_key: draft.clientKey,
      kind: draft.kind,
      description: draft.description.trim(),
      lat: draft.place?.lat ?? null,
      lon: draft.place?.lon ?? null,
      place_text: draft.placeText.trim() || null,
      location_precision: draft.place?.precision ?? "unknown",
      observed_at: new Date(draft.observedLocal).toISOString(),
      observed_time_precision: draft.timePrecision,
      observed_tz: zone,
    };
    const body = new FormData();
    body.append("report", JSON.stringify(payload));
    photos.forEach((file) => body.append("photos", file, file.name));
    setSending(true);
    try {
      const response = await fetch("/api/public/reports", { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setErrors([data.detail ?? `Сервер ответил ошибкой ${response.status}. Текст сохранён — попробуйте ещё раз.`]);
        requestAnimationFrame(() => errorBox.current?.focus());
        return;
      }
      const saved = readStore<Array<Pick<Receipt, "public_id" | "status_token" | "submitted_at">>>(RECEIPTS_KEY, []);
      writeStore(RECEIPTS_KEY, [
        { public_id: data.public_id, status_token: data.status_token, submitted_at: data.submitted_at },
        ...saved.filter((item) => item.public_id !== data.public_id),
      ].slice(0, 10));
      writeStore(DRAFT_KEY, null);
      setReceipt(data);
      window.scrollTo({ top: 0 });
    } catch {
      setErrors(["Нет связи с сервером. Ничего не потеряно: текст сохранён на этом устройстве, фото остаются выбранными. Попробуйте отправить ещё раз."]);
      requestAnimationFrame(() => errorBox.current?.focus());
    } finally {
      setSending(false);
    }
  };

  if (receipt)
    return (
      <Confirmation
        receipt={receipt}
        onAnother={() => {
          setReceipt(null);
          setAttempted(false);
          setErrors([]);
          setPhotos([]);
          setDraft(emptyDraft());
          setResults(null);
          setQuery("");
        }}
      />
    );

  return (
    <form className="report-form" onSubmit={submit} noValidate>
      <div className="report-intro">
        <h1>Сообщить о подтоплении</h1>
        <p>
          Сообщение получит аналитик и проверит его по другим источникам и спутниковым данным.
        </p>
        <p className="report-warning">
          Это не вызов экстренных служб. Если есть угроза жизни — звоните <a href="tel:112">112</a>.
          Не подходите к воде ради фотографии.
        </p>
      </div>

      {errors.length > 0 && (
        <div className="report-errors" role="alert" tabIndex={-1} ref={errorBox}>
          {errors.map((item) => (
            <p key={item}>{item}</p>
          ))}
        </div>
      )}

      <fieldset className="report-section">
        <legend>
          <span>1</span> Что произошло?
        </legend>
        <div className="kind-grid" role="radiogroup" aria-label="Тип сообщения">
          {KINDS.map(([value, label]) => (
            <label key={value} className={`kind-option ${draft.kind === value ? "active" : ""}`}>
              <input
                type="radio"
                name="kind"
                value={value}
                checked={draft.kind === value}
                onChange={() => update({ kind: value })}
              />
              {label}
            </label>
          ))}
        </div>
        <label className="report-field">
          Что вы видите
          <textarea
            value={draft.description}
            maxLength={2000}
            rows={4}
            placeholder="Например: вода перелилась через дорогу у моста, глубина примерно по колено"
            onChange={(event) => update({ description: event.target.value })}
          />
          <small>{draft.description.length} / 2000</small>
        </label>
      </fieldset>

      <fieldset className="report-section">
        <legend>
          <span>2</span> Где?
        </legend>
        <p className="report-hint">
          Отметьте место, где вода, — оно может не совпадать с тем, где вы сейчас.
        </p>
        <div className="search-row">
          <label className="report-input-icon">
            <Search size={16} aria-hidden="true" />
            <input
              value={query}
              placeholder="Город, улица или ориентир"
              aria-label="Поиск места"
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void search();
                }
              }}
            />
          </label>
          <button type="button" className="ghost-button" onClick={search} disabled={searching || query.trim().length < 2}>
            {searching ? "Ищем…" : "Найти"}
          </button>
        </div>
        {results && results.length > 0 && (
          <ul className="geo-results">
            {results.map((item) => (
              <li key={`${item.lat},${item.lon}`}>
                <button
                  type="button"
                  onClick={() => {
                    setPlace({ lat: item.lat, lon: item.lon, precision: item.precision, source: "search" });
                    setFocus({ point: item, zoom: item.precision === "point" || item.precision === "street" ? 15 : item.precision === "locality" ? 12 : 8 });
                    setResults(null);
                    setSearchNote(null);
                    if (!draft.placeText) update({ placeText: item.label.split(",").slice(0, 2).join(",") });
                  }}
                >
                  <span>{item.label}</span>
                  <small>{PRECISION[item.precision] ?? item.precision}</small>
                </button>
              </li>
            ))}
            <li className="geo-attribution">© OpenStreetMap contributors · Nominatim</li>
          </ul>
        )}
        {searchNote && <p className="report-note" role="status">{searchNote}</p>}
        <PlacePicker
          point={draft.place}
          focus={focus}
          onPick={(point) => {
            setPlace({ ...point, precision: "point", source: "map" });
            setSearchNote(null);
          }}
        />
        <div className="place-summary">
          {draft.place ? (
            <>
              <MapPin size={15} aria-hidden="true" />
              <span>
                {draft.place.lat.toFixed(5)}, {draft.place.lon.toFixed(5)} · точность: {PRECISION[draft.place.precision]}
              </span>
              <button type="button" onClick={() => setPlace(null)} aria-label="Убрать отметку">
                <X size={15} />
              </button>
            </>
          ) : (
            <span>Место не отмечено</span>
          )}
        </div>
        <div className="place-actions">
          <button type="button" className="ghost-button" onClick={locate} disabled={locating}>
            <Crosshair size={15} aria-hidden="true" /> {locating ? "Определяем…" : "Моё местоположение"}
          </button>
          <button type="button" className="ghost-button" aria-expanded={manualOpen} onClick={() => setManualOpen(!manualOpen)}>
            Ввести координаты
          </button>
        </div>
        {manualOpen && (
          <div className="manual-coords">
            <label className="report-field">
              Широта
              <input inputMode="decimal" value={manual.lat} placeholder="55.75580" onChange={(event) => setManual({ ...manual, lat: event.target.value })} />
            </label>
            <label className="report-field">
              Долгота
              <input inputMode="decimal" value={manual.lon} placeholder="37.61730" onChange={(event) => setManual({ ...manual, lon: event.target.value })} />
            </label>
            <button type="button" className="ghost-button" onClick={applyManual}>
              Отметить
            </button>
          </div>
        )}
        <label className="report-field">
          Ориентир или адрес словами {draft.place ? <em>необязательно</em> : null}
          <input
            value={draft.placeText}
            maxLength={200}
            placeholder="Например: Орск, ул. Набережная, у школы № 5"
            onChange={(event) => update({ placeText: event.target.value })}
          />
        </label>
      </fieldset>

      <fieldset className="report-section">
        <legend>
          <span>3</span> Когда заметили?
        </legend>
        <label className="report-field">
          Дата и время
          <input
            type="datetime-local"
            value={draft.observedLocal}
            max={localNow()}
            onChange={(event) => update({ observedLocal: event.target.value })}
          />
          <small>
            Часовой пояс устройства: {zone} ({offsetLabel()})
          </small>
        </label>
        <div className="segmented" role="radiogroup" aria-label="Насколько точно время">
          {(
            [
              ["exact", "Точно"],
              ["approx_hour", "Примерно"],
              ["approx_day", "Только день"],
            ] as const
          ).map(([value, label]) => (
            <label key={value} className={draft.timePrecision === value ? "active" : ""}>
              <input
                type="radio"
                name="time-precision"
                checked={draft.timePrecision === value}
                onChange={() => update({ timePrecision: value })}
              />
              {label}
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="report-section">
        <legend>
          <span>4</span> Фото <em>необязательно</em>
        </legend>
        <p className="report-hint">
          До 3 снимков JPEG, PNG или WebP, до 8 МБ каждый. Метаданные файла, включая координаты
          камеры, удаляются при сохранении. Фото видит только аналитик.
        </p>
        {photos.length > 0 && (
          <ul className="photo-list">
            {photos.map((file, index) => (
              <li key={`${file.name}-${index}`}>
                <img src={previews[index]} alt={`Фото ${index + 1}`} />
                <button
                  type="button"
                  aria-label={`Убрать фото ${index + 1}`}
                  onClick={() => setPhotos(photos.filter((_, other) => other !== index))}
                >
                  <X size={14} />
                </button>
              </li>
            ))}
          </ul>
        )}
        {photos.length < MAX_PHOTOS && (
          <label className="ghost-button photo-add">
            <Camera size={16} aria-hidden="true" /> Добавить фото
            <input
              type="file"
              accept={PHOTO_TYPES.join(",")}
              multiple
              onChange={(event) => {
                addPhotos(event.target.files);
                event.target.value = "";
              }}
            />
          </label>
        )}
        {photoError && <p className="report-note" role="status">{photoError}</p>}
      </fieldset>

      <div className="report-submit">
        <p>
          Сообщение попадёт на проверку аналитику. Отправка не подтверждает затопление и не
          заказывает спутниковую съёмку автоматически. Не указывайте имя, телефон и другие
          личные данные — они не нужны.
        </p>
        <button type="submit" className="primary-button" disabled={sending}>
          {sending ? "Отправляем…" : "Отправить сообщение"}
        </button>
      </div>
    </form>
  );
}

function Confirmation({ receipt, onAnother }: { receipt: Receipt; onAnother: () => void }) {
  const link = `${location.origin}/report/#status=${receipt.status_token}`;
  const [copied, setCopied] = useState(false);
  return (
    <section className="report-done" aria-live="polite">
      <h1>Сообщение принято</h1>
      <p className="report-number">
        Номер обращения <strong>{receipt.public_id}</strong>
      </p>
      <p>
        Его проверит аналитик. Отправка не означает, что затопление подтверждено, и не заказывает
        спутниковую съёмку автоматически.
      </p>
      <div className="status-link">
        <span>Ссылка на статус — сохраните её, по ней можно проверить статус или удалить сообщение:</span>
        <code>{link}</code>
        <div className="place-actions">
          <button
            type="button"
            className="ghost-button"
            onClick={() => {
              navigator.clipboard?.writeText(link).then(() => setCopied(true), () => setCopied(false));
            }}
          >
            {copied ? "Скопировано" : "Скопировать ссылку"}
          </button>
          <a className="ghost-button" href={`#status=${receipt.status_token}`}>
            Открыть статус
          </a>
        </div>
      </div>
      <p className="report-hint">
        Сообщение хранится до {formatDate(receipt.retention_until)} и затем удаляется вместе с фото.
      </p>
      <button type="button" className="primary-button" onClick={onAnother}>
        Отправить ещё одно
      </button>
    </section>
  );
}

function StatusView({ token }: { token: string }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [deleted, setDeleted] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    setStatus(null);
    setError(null);
    postJson<Status>("/api/public/reports/status", { token })
      .then(setStatus)
      .catch((cause) => setError(cause instanceof Error ? cause.message : "Статус недоступен"));
  }, [token]);
  const remove = async () => {
    try {
      const result = await postJson<{ deleted: string }>("/api/public/reports/delete", { token });
      setDeleted(result.deleted);
      const saved = readStore<Array<{ public_id: string }>>(RECEIPTS_KEY, []);
      writeStore(RECEIPTS_KEY, saved.filter((item) => item.public_id !== result.deleted));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Не удалось удалить");
    }
  };
  if (deleted)
    return (
      <section className="report-done">
        <h1>Сообщение удалено</h1>
        <p>Обращение {deleted} и его фото удалены.</p>
        <a className="primary-button" href="/report/">Новое сообщение</a>
      </section>
    );
  return (
    <section className="report-done" aria-live="polite">
      <h1>Статус сообщения</h1>
      {error && <p className="report-errors">{error}</p>}
      {!status && !error && <p>Загружаем…</p>}
      {status && (
        <>
          <p className="report-number">
            Номер обращения <strong>{status.public_id}</strong>
          </p>
          <dl className="status-grid">
            <dt>Состояние</dt>
            <dd>{status.status_text}</dd>
            <dt>Что сообщили</dt>
            <dd>{status.kind}</dd>
            <dt>Заметили</dt>
            <dd>{formatDate(status.observed_at)}</dd>
            <dt>Отправлено</dt>
            <dd>{formatDate(status.submitted_at)}</dd>
            <dt>Фото</dt>
            <dd>{status.photos || "нет"}</dd>
            <dt>Хранится до</dt>
            <dd>{formatDate(status.retention_until)}</dd>
          </dl>
          <div className="place-actions">
            {confirming ? (
              <>
                <button type="button" className="danger-button" onClick={remove}>
                  Да, удалить сообщение и фото
                </button>
                <button type="button" className="ghost-button" onClick={() => setConfirming(false)}>
                  Отмена
                </button>
              </>
            ) : (
              <button type="button" className="ghost-button" onClick={() => setConfirming(true)}>
                Удалить сообщение
              </button>
            )}
          </div>
        </>
      )}
      <a className="report-back" href="/report/">
        ← Новое сообщение
      </a>
    </section>
  );
}
