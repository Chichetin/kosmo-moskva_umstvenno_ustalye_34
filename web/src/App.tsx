import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowLeft,
  BarChart3,
  ChevronDown,
  Layers3,
  Search,
  X,
} from "lucide-react";
import Backdrop from "./Backdrop";
import MapPanel from "./MapPanel";
import {
  formatExactProbability,
  formatNumber as money,
  formatProbability,
  formatRub,
  formatScenarioAmount,
  formatScenarioLabel,
  formatShare as pct,
} from "./format";
import type { Feature, MediaMap, Plan, SceneSummary, Session } from "./types";

const dateTimeUtc = (value: string | null | undefined) =>
  value
    ? `${new Intl.DateTimeFormat("ru-RU", {
        day: "numeric",
        month: "long",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "UTC",
      }).format(new Date(value))} UTC`
    : "—";
function useCountUp(value: number | null | undefined, enabled: boolean) {
  const [shown, setShown] = useState<number | null>(value ?? null);
  const lastShown = useRef<number | null>(value ?? null);
  const hasAnimated = useRef(false);
  useEffect(() => {
    if (value == null) {
      setShown(null);
      lastShown.current = null;
      hasAnimated.current = false;
      return;
    }
    if (!enabled || window.matchMedia("(prefers-reduced-motion: reduce)").matches || value === 0) {
      setShown(value);
      lastShown.current = value;
      if (enabled && value !== 0) hasAnimated.current = true;
      return;
    }
    const from = hasAnimated.current ? (lastShown.current ?? 0) : 0;
    hasAnimated.current = true;
    if (from === value) return;
    let frame = 0;
    const start = performance.now();
    setShown(from);
    const tick = (now: number) => {
      const progress = Math.min((now - start) / 850, 1);
      const next = from + (value - from) * (1 - Math.pow(1 - progress, 3));
      setShown(next);
      lastShown.current = next;
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, enabled]);
  return shown;
}
const date = (value: string | null | undefined) =>
  value
    ? new Intl.DateTimeFormat("ru-RU", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      }).format(new Date(value))
    : "Дата неизвестна";
const labels: Record<string, string> = {
  warehouse: "Склад",
  substation: "Подстанция",
  road_unit: "Дорожный узел",
  facility: "Объект инфраструктуры",
  pumping_station: "Насосная станция",
  hospital: "Медицинский объект",
  school: "Образовательный объект",
  bridge: "Мост",
  water_treatment: "Очистные сооружения",
  telecom: "Узел связи",
  telecom_node: "Узел связи",
  water_intake: "Водозабор",
  workshop: "Мастерская",
  clinic: "Медпункт",
};
const assetName = (feature: Feature) =>
  labels[String(feature.properties.asset_class)] ??
  String(feature.properties.asset_class).replaceAll("_", " ");
const pointInZone = (asset: Feature, zone: Feature) => {
  if (asset.geometry.type !== "Point" || zone.geometry.type !== "Polygon")
    return false;
  const [x, y] = asset.geometry.coordinates as number[];
  const rings = zone.geometry.coordinates as number[][][];
  const inside = (ring: number[][]) => {
    let hit = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i],
        [xj, yj] = ring[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
        hit = !hit;
    }
    return hit;
  };
  return !!rings[0] && inside(rings[0]) && !rings.slice(1).some(inside);
};
const parseRoute = () => {
  const match = /^#\/(earth|scene|analysis)(?:\/([^/]+))?/.exec(location.hash);
  return {
    view: match?.[1] === "analysis" ? "analysis" : "map",
    chipId:
      match?.[1] === "earth"
        ? null
        : match?.[2]
          ? decodeURIComponent(match[2])
          : null,
  };
};
const navigate = (mode: "earth" | "scene" | "analysis", id?: string) => {
  location.hash = `#/${mode}${id ? `/${encodeURIComponent(id)}` : ""}`;
};
async function json<T>(url: string, signal?: AbortSignal, options?: RequestInit): Promise<T> {
  const response = await fetch(url, { ...options, signal });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail ?? `Ошибка ${response.status}`, response.status);
  }
  return response.json() as Promise<T>;
}
class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}
type LayerFlags = {
  s1: boolean;
  probability: boolean;
  mask: boolean;
  uncertainty: boolean;
  assets: boolean;
  zones: boolean;
};
type AnalysisJob = {
  job_id: string | null;
  chip_id: string;
  status: "queued" | "running" | "done" | "failed";
  step: number;
  steps: number;
  step_label: string | null;
  run_id: string | null;
  error: string | null;
  detail: string | null;
};
const defaults: LayerFlags = {
  s1: true,
  probability: false,
  mask: false,
  uncertainty: false,
  assets: true,
  zones: true,
};

export default function App() {
  const [scenes, setScenes] = useState<SceneSummary[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [route, setRoute] = useState(parseRoute);
  const [session, setSession] = useState<Session | null>(null);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const cache = useRef<Record<string, Session>>({});
  const budgets = useRef<Record<string, number>>({});
  const [plan, setPlan] = useState<Plan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [budget, setBudget] = useState<number | null>(null);
  const [baseMap, setBaseMap] = useState<"map" | "optical">("map");
  const [layers, setLayers] = useState<LayerFlags>(defaults);
  const [opacity, setOpacity] = useState(0.76);
  const [assetId, setAssetId] = useState<string | null>(null);
  const [zoneId, setZoneId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [analysisQuery, setAnalysisQuery] = useState("");
  const [expandedTerritory, setExpandedTerritory] = useState<string | null>(null);
  const [sourceViewerOpen, setSourceViewerOpen] = useState(false);
  const [analysisJob, setAnalysisJob] = useState<AnalysisJob | null>(null);
  const [analysisStarting, setAnalysisStarting] = useState<string | null>(null);
  const [analysisJobError, setAnalysisJobError] = useState<string | null>(null);
  const [readyAnalysisNotice, setReadyAnalysisNotice] = useState<{ chipId: string; territory: string } | null>(null);
  const [searchOpen, setSearchOpen] = useState(() => window.innerWidth >= 700);
  const [panelOpen, setPanelOpen] = useState(true);
  const [overviewRequest, setOverviewRequest] = useState(0);
  const [mapIssue, setMapIssue] = useState<string | null>(null);
  // «Сообщения» (/api/media/map): null — the layer is unavailable here (media not served,
  // or closed to this viewer); the toggle is then hidden rather than shown broken.
  const [messages, setMessages] = useState<MediaMap | null>(null);
  const [showMessages, setShowMessages] = useState(true);
  useEffect(() => {
    let stopped = false;
    const load = () =>
      fetch("/api/media/map", { credentials: "same-origin" })
        .then((response) => (response.ok ? (response.json() as Promise<MediaMap>) : null))
        .catch(() => null)
        .then((data) => {
          if (!stopped) setMessages(data);
        });
    load();
    const timer = window.setInterval(load, 60_000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, []);
  const messageCount =
    (messages?.events.features.length ?? 0) + (messages?.reports.features.length ?? 0);
  const [zoneView, setZoneView] = useState<"plan" | "all">("plan");
  const scene = scenes.find((item) => item.chip_id === route.chipId) ?? null;
  const activeSession = scene?.run_id === session?.run_id ? session : null;
  const mapActive = route.view === "map";
  const mapVisible = mapActive || sourceViewerOpen;
  useEffect(() => {
    const onHash = () => setRoute(parseRoute());
    addEventListener("hashchange", onHash);
    if (!location.hash) navigate("earth");
    return () => removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    if (route.chipId) setSearchOpen(false);
  }, [route.chipId]);
  useEffect(() => {
    if (mapActive) setSourceViewerOpen(false);
  }, [mapActive]);
  useEffect(() => {
    const controller = new AbortController();
    json<SceneSummary[]>("/api/scenes", controller.signal)
      .then(setScenes)
      .catch((error) => {
        if (error.name !== "AbortError") setCatalogError(String(error.message));
      })
      .finally(() => {
        if (!controller.signal.aborted) setCatalogLoading(false);
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!scene?.run_id) {
      setSession(null);
      setPlan(null);
      setSessionLoading(false);
      setSessionError(null);
      return;
    }
    const runId = scene.run_id;
    const cached = cache.current[runId];
    if (cached) {
      setSession(cached);
      setBudget(
        budgets.current[runId] ?? Number(cached.metadata.budget.budget_rub),
      );
      setSessionLoading(false);
      return;
    }
    const controller = new AbortController();
    setSession(null);
    setPlan(null);
    setSessionLoading(true);
    setSessionError(null);
    json<Session>(
      `/api/session?run_id=${encodeURIComponent(runId)}`,
      controller.signal,
    )
      .then((result) => {
        if (controller.signal.aborted) return;
        cache.current[runId] = result;
        setSession(result);
        setBudget(
          budgets.current[runId] ?? Number(result.metadata.budget.budget_rub),
        );
      })
      .catch((error) => {
        if (error.name !== "AbortError") setSessionError(String(error.message));
      })
      .finally(() => {
        if (!controller.signal.aborted) setSessionLoading(false);
      });
    return () => controller.abort();
  }, [scene?.run_id]);
  useEffect(() => {
    if (!activeSession || budget === null) return;
    const runId = activeSession.run_id;
    budgets.current[runId] = budget;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      json<Plan>(
        `/api/plan?budget=${budget}&run_id=${encodeURIComponent(runId)}`,
        controller.signal,
      )
        .then((result) => {
          if (!controller.signal.aborted) {
            setPlan(result);
            setPlanError(null);
          }
        })
        .catch((error) => {
          if (error.name !== "AbortError") setPlanError(String(error.message));
        });
    }, 140);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [activeSession?.run_id, budget]);
  const openFinishedAnalysis = useCallback(async (finished: AnalysisJob) => {
    const updated = await json<SceneSummary[]>("/api/scenes");
    setScenes(updated);
    const target = updated.find((item) => item.chip_id === finished.chip_id && item.run_id);
    if (!target) {
      throw new Error("Расчёт завершён, но результат ещё не появился в каталоге. Обновите его повторно.");
    }
    if (parseRoute().chipId === finished.chip_id) {
      setReadyAnalysisNotice(null);
      navigate("analysis", finished.chip_id);
    } else {
      setReadyAnalysisNotice({ chipId: finished.chip_id, territory: target.territory.replaceAll("-", " ") });
    }
  }, []);
  const startAnalysis = useCallback(async (target: SceneSummary) => {
    setAnalysisStarting(target.chip_id);
    setAnalysisJobError(null);
    try {
      const result = await json<AnalysisJob>(
        `/api/analyses?chip_id=${encodeURIComponent(target.chip_id)}`,
        undefined,
        { method: "POST" },
      );
      setAnalysisJob(result);
      if (result.status === "done") await openFinishedAnalysis(result);
    } catch (error) {
      setAnalysisJobError(error instanceof Error ? error.message : String(error));
    } finally {
      setAnalysisStarting(null);
    }
  }, [openFinishedAnalysis]);
  useEffect(() => {
    if (!analysisJob?.job_id || !["queued", "running"].includes(analysisJob.status)) return;
    let stopped = false;
    let timer: number;
    const poll = async () => {
      try {
        const result = await json<AnalysisJob>(`/api/analyses/${encodeURIComponent(analysisJob.job_id!)}`);
        if (stopped) return;
        setAnalysisJob(result);
        setAnalysisJobError(null);
        if (result.status === "done") {
          try {
            await openFinishedAnalysis(result);
          } catch (error) {
            if (!stopped) setAnalysisJobError(error instanceof Error ? error.message : String(error));
          }
        } else if (result.status !== "failed") {
          timer = window.setTimeout(poll, 1000);
        }
      } catch (error) {
        if (stopped) return;
        if (error instanceof ApiError && error.status === 404) {
          setAnalysisJob((current) => current?.job_id === analysisJob.job_id
            ? { ...current, status: "failed", error: "Задача потеряна после перезапуска сервера" }
            : current);
          setAnalysisJobError(null);
          return;
        }
        setAnalysisJobError(error instanceof Error ? error.message : String(error));
        timer = window.setTimeout(poll, 3000);
      }
    };
    timer = window.setTimeout(poll, 1000);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [analysisJob?.job_id, openFinishedAnalysis]);
  useEffect(() => setAnalysisJobError(null), [route.chipId]);
  useEffect(() => {
    if (scene?.territory) setExpandedTerritory(scene.territory);
  }, [scene?.territory]);
  const selectScene = useCallback((selected: SceneSummary) => {
    setPanelOpen(true);
    setSearchOpen(false);
    setAssetId(null);
    setZoneId(null);
    setPlan(null);
    navigate("scene", selected.chip_id);
  }, []);
  const overview = () => {
    navigate("earth");
    setSearchOpen(true);
    setOverviewRequest((value) => value + 1);
  };
  const toggle = (key: keyof LayerFlags) =>
    setLayers((value) => ({ ...value, [key]: !value[key] }));
  const assets = useMemo(
    () =>
      [...(activeSession?.assets.features ?? [])].sort(
        (a, b) =>
          Number(a.properties.rank || 999) - Number(b.properties.rank || 999),
      ),
    [activeSession],
  );
  const zones = activeSession?.candidates.features ?? [];
  const chosen = plan?.comparison.find((row) => row.strategy === "C");
  const chosenZoneIds = new Set(plan?.plans.C ?? []);
  const matchingScenes = scenes.filter((item) =>
    `${item.territory} ${item.chip_id} ${item.observation_date}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const seenTerritories = new Set<string>();
  const firstByTerritory: SceneSummary[] = [];
  const remainingScenes: SceneSummary[] = [];
  for (const item of matchingScenes) {
    if (!seenTerritories.has(item.territory)) {
      seenTerritories.add(item.territory);
      firstByTerritory.push(item);
    } else remainingScenes.push(item);
  }
  const filtered = (query ? matchingScenes : [...firstByTerritory, ...remainingScenes])
    .slice(0, query ? 30 : 10);
  const analysisLocations = scenes.filter((item) =>
    `${item.territory} ${item.chip_id} ${item.observation_date ?? ""}`
      .toLocaleLowerCase("ru")
      .includes(analysisQuery.trim().toLocaleLowerCase("ru")),
  );
  const locationGroups = [...new Set(analysisLocations.map((item) => item.territory))]
    .sort((left, right) => left.localeCompare(right, "ru"))
    .map((territory) => ({
      territory,
      items: analysisLocations.filter((item) => item.territory === territory),
    }));
  const renderAnalysisAction = (target: SceneSummary) => {
    if (target.analyzable === false) {
      return <p className="analysis-unavailable">{target.not_analyzable_reason ?? "В исходном снимке нет данных для расчёта."}</p>;
    }
    const current = analysisJob?.chip_id === target.chip_id ? analysisJob : null;
    const running = current?.status === "queued" || current?.status === "running";
    const errorText = analysisJobError && scene?.chip_id === target.chip_id
      ? analysisJobError : current?.status === "failed" ? current.error : null;
    const statusText = errorText ?? (current?.status === "queued" ? "В очереди · ожидает запуска"
      : current?.status === "running" ? `Шаг ${current.step} из ${current.steps} · ${current.step_label}`
      : null);
    return (
      <div className="analysis-job">
        <button
          className="analysis-start"
          disabled={running || analysisStarting === target.chip_id}
          onClick={() => void startAnalysis(target)}
        >
          {running ? "Анализ выполняется" : analysisStarting === target.chip_id
            ? "Отправляем запрос…" : current?.status === "failed" ? "Повторить анализ" : "Запустить анализ"}
        </button>
        {statusText && <p className={errorText ? "inline-error" : undefined} role="status" aria-live="polite">{statusText}</p>}
        {current?.status === "failed" && current.detail && (
          <details><summary>Подробности ошибки</summary><pre>{current.detail}</pre></details>
        )}
        {current?.status === "done" && !target.run_id && (
          <button className="analysis-refresh" onClick={() => void openFinishedAnalysis(current).catch((error) => setAnalysisJobError(String(error)))}>
            Обновить результат
          </button>
        )}
      </div>
    );
  };
  const activeAsset = assets.find(
    (item) => item.properties.asset_id === assetId,
  );
  const activeZone = zones.find(
    (item) => item.properties.candidate_id === zoneId,
  );
  const zoneAssets = activeZone
    ? assets.filter((item) => pointInZone(item, activeZone))
    : [];
  const totalLoss = assets.reduce(
    (sum, item) => sum + Number(item.properties.expected_loss_rub || 0),
    0,
  );
  const shownLoss = useCountUp(activeSession ? totalLoss : null, !mapVisible && !!activeSession);
  const shownCost = useCountUp(chosen?.data_cost_rub, !mapVisible && !!activeSession);
  const shownCoverage = useCountUp(chosen?.coverage_share, !mapVisible && !!activeSession);
  const exportUrl = activeSession
    ? `/api/download?run_id=${encodeURIComponent(activeSession.run_id)}`
    : null;
  return (
    <div className="app">
      <Backdrop paused={!mapActive || !!scene} ocean={baseMap === "optical"} />
      <header className={`topbar ${mapVisible ? "topbar-map" : "topbar-analysis"}`}>
        <button className="brand" onClick={overview} aria-label="Митрандир — обзор Земли">
          <svg className="brand-mark" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M3.5 18.5V5.5L12 14l8.5-8.5v13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Митрандир
        </button>
        <span className="top-divider" />
        <nav className="top-nav">
          <button
            className={mapActive ? "active" : ""}
            onClick={() => {
              setSourceViewerOpen(false);
              scene ? navigate("scene", scene.chip_id) : overview();
            }}
          >
            Земля
          </button>
          <button
            className={!mapActive ? "active" : ""}
            onClick={() => {
              setSourceViewerOpen(false);
              navigate("analysis", scene?.chip_id);
            }}
          >
            Анализ
          </button>
        </nav>
        <span className="top-spacer" />
        {exportUrl && (
          <a
            className="top-export"
            href={exportUrl}
            title="Выгрузить файлы текущего анализа"
          >
            <ArrowDownToLine size={16} />
            <span>Выгрузить комплект</span>
            {activeSession && (
              <small className="export-note">
                при бюджете {money(activeSession.metadata.budget.budget_rub)} ₽
              </small>
            )}
          </a>
        )}
      </header>
      <main className="workspace">
        <div
          className={`map-screen ${mapVisible ? "" : "map-hidden"} ${scene ? "scene-mode" : ""} ${baseMap === "optical" ? "optical-mode" : ""}`}
          aria-hidden={!mapVisible}
          inert={!mapVisible}
        >
          <MapPanel
            scenes={scenes}
            scene={scene}
            session={activeSession}
            plan={plan}
            baseMap={baseMap}
            visibleLayers={layers}
            messages={messages}
            showMessages={showMessages}
            overlayOpacity={opacity}
            selectedAsset={assetId}
            selectedZone={zoneId}
            active={mapVisible}
            overviewRequest={overviewRequest}
            onSelectScene={selectScene}
            onSelectAsset={(id) => {
              setAssetId(id);
              setZoneId(null);
              setPanelOpen(true);
            }}
            onSelectZone={(id) => {
              setZoneId(id);
              setAssetId(null);
              setPanelOpen(true);
            }}
            onIssue={setMapIssue}
          />
          <div className="map-top-left">
            <div className="search-panel">
              <div className="scene-library-head">
                <h1>Наблюдения</h1>
              </div>
              <label className="search-box">
                <Search size={17} />
                <input
                  value={query}
                  onFocus={() => setSearchOpen(true)}
                  onChange={(event) => {
                    setQuery(event.target.value);
                    setSearchOpen(true);
                  }}
                  placeholder="Страна или ID сцены"
                  aria-label="Поиск наблюдений"
                />
                {query && (
                  <button onClick={() => setQuery("")} aria-label="Очистить">
                    <X size={15} />
                  </button>
                )}
              </label>
              <button
                className="search-toggle"
                onClick={() => setSearchOpen(!searchOpen)}
                aria-expanded={searchOpen}
              >
                <span>{query ? "Результаты поиска" : "Все сцены"}</span>
                <span className="search-count">
                  {query
                    ? `${filtered.length}${filtered.length === 30 ? "+" : ""}`
                    : scenes.length}
                </span>
                <ChevronDown size={16} />
              </button>
              {searchOpen && (
                <div
                  className="scene-results"
                  role="listbox"
                  aria-label="Доступные сцены"
                >
                  {catalogError && <p>{catalogError}</p>}
                  {!catalogError && !scenes.length && (
                    <p>
                      {catalogLoading
                        ? "Загружаем каталог…"
                        : "Нет доступных наблюдений. Добавьте исходные данные или готовый комплект."}
                    </p>
                  )}
                  {filtered.map((item, index) => (
                    <button
                      key={item.chip_id}
                      style={{ animationDelay: `${Math.min(index, 8) * 42 + 80}ms` }}
                      role="option"
                      aria-selected={scene?.chip_id === item.chip_id}
                      onClick={() => selectScene(item)}
                    >
                      <span className="scene-copy">
                        <strong>{item.territory.replaceAll("-", " ")}</strong>
                        <small>
                          {date(item.observation_date)}
                          <code>{item.chip_id}</code>
                        </small>
                      </span>
                      {(item.fixture || item.status === "analyzed") && (
                        <em>{item.fixture ? "Тест" : "Анализ"}</em>
                      )}
                    </button>
                  ))}
                  {!filtered.length && scenes.length > 0 && (
                    <p>Сцен не найдено</p>
                  )}
                  {scenes.length > 10 && !query && (
                    <small className="search-hint">
                      Другие снимки доступны через поиск или точки на Земле.
                    </small>
                  )}
                </div>
              )}
            </div>
          </div>
          <div className="map-top-center">
            <div className="basemap-switch">
              <button
                className={baseMap === "map" ? "active" : ""}
                onClick={() => setBaseMap("map")}
              >
                Карта
              </button>
              <button
                className={baseMap === "optical" ? "active" : ""}
                onClick={() => setBaseMap("optical")}
                title="Мозаика EOX 2019 — снята после события; только ориентир, в p, ущерб и план не входит"
              >
                Спутник
              </button>
            </div>
            {messages && (
              <div className="basemap-switch media-switch">
                <button
                  className={showMessages ? "active" : ""}
                  onClick={() => setShowMessages(!showMessages)}
                  aria-pressed={showMessages}
                  title={`События из СМИ и обращения очевидцев: ${messages.events.features.length} событий, ${messages.reports.features.length} обращений ждут модерации. Повод проверить территорию — не входит в p, ущерб и план.`}
                >
                  Сообщения · {messageCount}
                </button>
              </div>
            )}
          </div>
          {scene && panelOpen && (
            <aside className="context-panel">
              <div className="context-heading">
                <div>
                  <small>НАБЛЮДЕНИЕ · SENTINEL-1</small>
                  <h1>{scene.territory.replaceAll("-", " ")}</h1>
                  <p>
                    {date(scene.observation_date)} ·{" "}
                    <code>{scene.chip_id}</code>
                  </p>
                </div>
                <button
                  onClick={() => setPanelOpen(false)}
                  aria-label="Свернуть панель"
                >
                  <X size={17} />
                </button>
              </div>
              {scene.preview_url && (
                <img
                  className="scene-preview"
                  src={scene.preview_url}
                  alt={`Радарный снимок ${scene.chip_id}`}
                />
              )}
              <div className="scene-message">
                {scene.fixture
                  ? "Синтетический тестовый эталон"
                  : scene.status === "analyzed"
                    ? "Результаты модели доступны"
                    : "Исходный снимок есть; анализ не выполнен"}
              </div>
              {scene.status === "observation_only" && renderAnalysisAction(scene)}
              {!scene.bounds && (
                <p className="inline-error">
                  Геопривязка отсутствует. Снимок нельзя разместить на карте.
                </p>
              )}
              {sessionLoading && <p>Загружаем расчёт…</p>}
              {sessionError && <p className="inline-error">{sessionError}</p>}
              <section className="context-section">
                <h2>
                  <Layers3 size={17} /> Данные наблюдения
                </h2>
                <label className="layer-row">
                  <input
                    type="checkbox"
                    checked={layers.s1}
                    disabled={
                      scene.status === "analyzed" && !activeSession?.layers.s1
                    }
                    onChange={() => toggle("s1")}
                  />
                  <span>
                    Радарный снимок S1
                    <small>{date(scene.observation_date)} · VV/VH</small>
                  </span>
                </label>
                {activeSession && (
                  <>
                    <label className="layer-row">
                      <input
                        type="checkbox"
                        checked={layers.probability}
                        onChange={() => toggle("probability")}
                      />
                      <span>
                        Вероятность воды<small>Оценка модели · 0–100%</small>
                      </span>
                    </label>
                    <label className="layer-row">
                      <input
                        type="checkbox"
                        checked={layers.mask}
                        onChange={() => toggle("mask")}
                      />
                      <span>
                        Бинарная маска
                        <small>
                          Порог ≥ {activeSession.metadata.threshold}
                        </small>
                      </span>
                    </label>
                    <label className="layer-row">
                      <input
                        type="checkbox"
                        checked={layers.uncertainty}
                        disabled={!activeSession.layers.uncertainty}
                        onChange={() => toggle("uncertainty")}
                      />
                      <span>
                        Неопределённость
                        <small>Энтропия p · 0–1 бит</small>
                      </span>
                    </label>
                    <p className="hint-note">
                      Подложка «Карта / Спутник» (OpenStreetMap, EOX 2019) снята
                      после события — только для ориентира, в p, ущерб и план не
                      входит.
                    </p>
                    {(layers.probability || layers.mask || layers.uncertainty) && (
                      <label className="opacity-control">
                        Прозрачность наложения{" "}
                        <b>{Math.round(opacity * 100)}%</b>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={Math.round(opacity * 100)}
                          onChange={(event) =>
                            setOpacity(Number(event.target.value) / 100)
                          }
                        />
                      </label>
                    )}
                    {layers.probability && (
                      <div className="prob-legend">
                        <span>0% · низкая</span>
                        <i />
                        <span>100% · высокая</span>
                      </div>
                    )}
                    <label className="layer-row">
                      <input
                        type="checkbox"
                        checked={layers.assets}
                        onChange={() => toggle("assets")}
                      />
                      <span>
                        Объекты<small>{assets.length} точек</small>
                      </span>
                    </label>
                    <label className="layer-row">
                      <input
                        type="checkbox"
                        checked={layers.zones}
                        onChange={() => toggle("zones")}
                      />
                      <span>
                        Зоны съёмки
                        <small>
                          {zones.length} кандидатов · план C выделен синим
                        </small>
                      </span>
                    </label>
                  </>
                )}
              </section>
              {activeSession && (
                <>
                  <section className="context-section">
                    <h2>План дополнительной съёмки</h2>
                    <label className="budget-field">
                      Бюджет данных, ₽
                      <input
                        type="number"
                        min="0"
                        max="1000000000"
                        step="100"
                        value={budget ?? 0}
                        onChange={(event) =>
                          setBudget(
                            Math.max(0, Number(event.target.value) || 0),
                          )
                        }
                      />
                    </label>
                    <input
                      className="budget-slider"
                      type="range"
                      min="0"
                      max="50000"
                      step="100"
                      value={Math.min(budget ?? 0, 50000)}
                      onChange={(event) =>
                        setBudget(Number(event.target.value))
                      }
                      aria-label="Изменить бюджет"
                    />
                    <div className="plan-summary">
                      <span>План C · {plan?.plans.C.length ?? "—"} зон</span>
                      <strong>{formatRub(chosen?.data_cost_rub)}</strong>
                    </div>
                    <div className="plan-summary">
                      <span>Охват ожидаемого ущерба</span>
                      <strong>{pct(chosen?.coverage_share)}</strong>
                    </div>
                    <div className="plan-summary">
                      <span>Доля адресуемой цены ошибки · сценарий</span>
                      <strong>{pct(chosen?.decision_risk_share)}</strong>
                    </div>
                    <p className="hint-note">
                      Цены по сценарной ставке Б ={" "}
                      {activeSession.metadata.pricing?.base_rate_rub_km2 ?? "—"}{" "}
                      ₽/км², не тариф. Срок решения —{" "}
                      {dateTimeUtc(activeSession.metadata.budget.decision_deadline)}.
                    </p>
                    {planError && <p className="inline-error">{planError}</p>}
                    <button
                      className="analysis-link"
                      onClick={() => navigate("analysis", scene.chip_id)}
                    >
                      Подробный анализ <BarChart3 size={17} />
                    </button>
                  </section>
                  {(activeAsset || activeZone) && (
                    <section className="context-section focus-section">
                      <h2>
                        {activeAsset ? "Выбранный объект" : "Выбранная зона"}
                      </h2>
                      {activeAsset && (
                        <>
                          <strong>{assetName(activeAsset)}</strong>
                          <p>
                            {String(activeAsset.properties.asset_id)} ·
                            вероятность воды{" "}
                            {formatProbability(activeAsset.properties.p_flood)}
                          </p>
                          <p>
                            Ожидаемый ущерб{" "}
                            {formatRub(activeAsset.properties.expected_loss_rub)}
                          </p>
                        </>
                      )}
                      {activeZone && (
                        <>
                          <strong>
                            {String(activeZone.properties.candidate_id)}
                          </strong>
                          <p>
                            {activeZone.properties.area_km2} км² · отдельно{" "}
                            {money(activeZone.properties.solo_cost_rub)} ₽
                          </p>
                          <p>
                            Доступна{" "}
                            {date(
                              String(activeZone.properties.available_at ?? ""),
                            )}
                          </p>
                          <p>
                            Объекты в зоне:{" "}
                            {zoneAssets.length
                              ? zoneAssets.map(assetName).join(", ")
                              : "нет объектов портфеля"}
                          </p>
                        </>
                      )}
                    </section>
                  )}
                </>
              )}
              <p className="context-foot">
                Маркеры — места снимков, не подтверждённые зоны затопления.
                Оптическая мозаика 2019 года — географический контекст; дата
                отдельного пикселя неизвестна.
              </p>
            </aside>
          )}
          {scene && !panelOpen && (
            <button className="reopen-panel" onClick={() => setPanelOpen(true)}>
              <Layers3 size={17} /> Слои и сцена
            </button>
          )}
          {mapIssue && (
            <div className="map-issue" role="status">
              {mapIssue}
            </div>
          )}
          {sourceViewerOpen && scene && (
            <button className="source-viewer-return" onClick={() => setSourceViewerOpen(false)}>
              <ArrowLeft size={16} /> К результату анализа
            </button>
          )}
        </div>
          <div className={`analysis-screen ${mapVisible ? "analysis-hidden" : ""}`} aria-hidden={mapVisible} inert={mapVisible}>
            <div className="analysis-shell">
              <aside className="analysis-locations" aria-label="Локации для анализа">
                <div className="analysis-locations-head">
                  <h2>Локации</h2>
                  <span>{locationGroups.length} · {scenes.length} снимков</span>
                </div>
                <label className="analysis-location-search">
                  <Search size={16} />
                  <input
                    value={analysisQuery}
                    onChange={(event) => setAnalysisQuery(event.target.value)}
                    placeholder="Страна или ID снимка"
                    aria-label="Поиск локации для анализа"
                  />
                </label>
                <div className="analysis-location-list">
                  {catalogError && <p className="inline-error">{catalogError}</p>}
                  {catalogLoading && !scenes.length && <p>Загружаем локации…</p>}
                  {locationGroups.map(({ territory, items }) => {
                    const expanded = !!analysisQuery || expandedTerritory === territory;
                    return (
                      <div className="analysis-location-group" key={territory}>
                        <button
                          className="analysis-location-group-toggle"
                          aria-expanded={expanded}
                          onClick={() => setExpandedTerritory(expanded ? null : territory)}
                        >
                          <strong>{territory.replaceAll("-", " ")}</strong>
                          <span>{items.length} <ChevronDown size={14} /></span>
                        </button>
                        {expanded && items.map((item) => (
                          <button
                            key={item.chip_id}
                            className={`analysis-location-item ${scene?.chip_id === item.chip_id ? "active" : ""}`}
                            aria-current={scene?.chip_id === item.chip_id ? "true" : undefined}
                            onClick={() => navigate("analysis", item.chip_id)}
                          >
                            <strong>{item.chip_id}</strong>
                            <small>{date(item.observation_date)}</small>
                            <em>{item.status === "analyzed" ? "Готов" : item.analyzable === false ? "Без данных для расчёта" : "Исходный снимок"}</em>
                          </button>
                        ))}
                      </div>
                    );
                  })}
                  {!catalogLoading && !catalogError && !analysisLocations.length && (
                    <p>{analysisQuery ? "Локации не найдены" : "Нет доступных локаций"}</p>
                  )}
                </div>
              </aside>
              <div className="analysis-content">
            <div className="analysis-head">
              <button
                onClick={() =>
                  scene ? navigate("scene", scene.chip_id) : overview()
                }
              >
                <ArrowLeft size={17} /> К карте
              </button>
              <div>
                <span>{scene ? "АНАЛИЗ СЦЕНЫ" : "АНАЛИЗ НАБЛЮДЕНИЙ"}</span>
                <h1>
                  {scene?.territory.replaceAll("-", " ") ?? "Выберите локацию"}
                </h1>
                <p>
                  {scene
                    ? `${date(scene.observation_date)} · ${scene.chip_id}`
                    : "Выберите локацию в списке слева"}
                </p>
              </div>
              {scene && (
                <button
                  className="analysis-source-link"
                  disabled={!scene.bounds}
                  title={!scene.bounds ? "У снимка нет геопривязки для карты" : undefined}
                  onClick={() => {
                    setLayers((current) => ({ ...current, s1: true }));
                    setPanelOpen(true);
                    setSourceViewerOpen(true);
                  }}
                >
                  <Layers3 size={16} /> Исходный снимок и слои
                </button>
              )}
            </div>
            {!activeSession && (
              <div className="analysis-empty">
                {sessionLoading ? "Загружаем анализ…" : sessionError ?? (
                  scene ? scene.analyzable === false
                    ? "Результат анализа для этого снимка недоступен."
                    : "Для этой сцены есть исходный S1, но расчёт объектов и плана не выполнен."
                    : "Выберите локацию слева, чтобы открыть результат или запустить расчёт."
                )}
                {scene?.status === "observation_only" && renderAnalysisAction(scene)}
              </div>
            )}
            {activeSession && (
              <div className="analysis-body">
                <div className="analysis-metrics">
                  <div>
                    <span>Объектов</span>
                    <strong>{assets.length}</strong>
                  </div>
                  <div>
                    <span>Ожидаемый ущерб</span>
                    <strong>{formatRub(shownLoss)}</strong>
                  </div>
                  <div>
                    <span>План C · данные</span>
                    <strong>{formatRub(shownCost)}</strong>
                  </div>
                  <div>
                    <span>Охват ущерба</span>
                    <strong>{pct(shownCoverage)}</strong>
                  </div>
                  <div>
                    <span>Доля адресуемой цены ошибки · сценарий</span>
                    <strong>{pct(chosen?.decision_risk_share)}</strong>
                  </div>
                </div>
                <div className="analysis-columns">
                  <section className="analysis-card">
                    <div className="card-title">
                      <h2>Приоритет осмотра</h2>
                      <span>Выбор выделит объект на карте</span>
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Объект</th>
                            <th>Вероятность воды</th>
                            <th>Ожидаемый ущерб</th>
                            <th>Неопределённость карты</th>
                          </tr>
                        </thead>
                        <tbody>
                          {assets.map((item) => (
                            <tr
                              key={String(item.properties.asset_id)}
                              tabIndex={0}
                              className={
                                assetId === item.properties.asset_id
                                  ? "selected"
                                  : ""
                              }
                              onClick={() => {
                                setAssetId(String(item.properties.asset_id));
                                setZoneId(null);
                              }}
                              onKeyDown={(event) => {
                                if (
                                  event.key === "Enter" ||
                                  event.key === " "
                                ) {
                                  event.preventDefault();
                                  setAssetId(String(item.properties.asset_id));
                                  setZoneId(null);
                                }
                              }}
                            >
                              <td>
                                <strong>{assetName(item)}</strong>
                                <small>
                                  {String(item.properties.asset_id)} ·{" "}
                                  {item.properties.status === "no_data"
                                    ? "нет данных"
                                    : "оценка доступна"}
                                </small>
                              </td>
                              <td>
                                {item.properties.status === "no_data"
                                  ? "—"
                                  : <span title={`Точное p: ${formatExactProbability(item.properties.p_flood)}`}>
                                      {formatProbability(item.properties.p_flood)}
                                    </span>}
                              </td>
                              <td>
                                {formatRub(item.properties.expected_loss_rub)}
                              </td>
                              <td>
                                {item.properties.uncertainty == null ||
                                item.properties.uncertainty === ""
                                  ? "—"
                                  : Number(item.properties.uncertainty).toFixed(
                                      2,
                                    )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {activeAsset && (
                      <div className="card-foot">
                        {assetName(activeAsset)} · стоимость{" "}
                        {money(activeAsset.properties.asset_value_rub)} ₽ ·
                        уязвимость{" "}
                        {Number(
                          activeAsset.properties.vulnerability_coef,
                        ).toFixed(2)} · точное p: {formatExactProbability(activeAsset.properties.p_flood)}
                      </div>
                    )}
                  </section>
                  <div className="analysis-stack">
                    <section className="analysis-card">
                      <div className="card-title">
                        <h2>Стратегии A / B / C</h2>
                        <span>Стоимость данных и охват</span>
                      </div>
                      {(plan?.comparison ?? []).map((row) => (
                        <div
                          key={row.strategy}
                          className={`strategy ${row.strategy === "C" ? "chosen" : ""}`}
                        >
                          <b>{row.strategy}</b>
                          <div>
                            <strong>
                              {row.strategy === "A"
                                ? "Текущие данные"
                                : row.strategy === "B"
                                  ? "Сплошная съёмка"
                                  : "Приоритетный заказ"}
                            </strong>
                            <small>
                              Данные {formatRub(row.data_cost_rub)} · охват
                              ущерба {pct(row.coverage_share)} · доля адресуемой
                              цены ошибки {pct(row.decision_risk_share)}
                            </small>
                            {row.strategy === "B" && !row.budget_feasible && (
                              <small>
                                Выше бюджета — контрфактическая, не
                                рекомендация
                              </small>
                            )}
                            {row.strategy === "C" && (
                              <small>
                                Полная стоимость решения{" "}
                                {money(row.decision_cost_rub)} ₽
                              </small>
                            )}
                          </div>
                        </div>
                      ))}
                      <div className="card-foot">
                        <p>Охват — доля ожидаемого ущерба объектов в зонах, а не предотвращённый ущерб.</p>
                        <p>Цены рассчитаны по сценарной ставке {money(activeSession.metadata.pricing?.base_rate_rub_km2)} ₽/км²; это не тариф.</p>
                        <details>
                          <summary>Методика расчёта</summary>
                          <p>Сценарная цена ошибки решения R = V·q·min(p, 1−p). Показана доля R объектов в зонах от R всех оценённых объектов; снижение фактических потерь не заявляется.</p>
                          <p>План C не покупает новые данные ради объектов, для которых исход по карте почти определён: их осмотр планируется по исходным данным.</p>
                        </details>
                        <details>
                          <summary>Воспроизводимость</summary>
                          <p>Исходная выгрузка соответствует бюджету {formatRub(activeSession.metadata.budget.budget_rub)}. Метод и входные данные записаны в комплекте файлов.</p>
                        </details>
                      </div>
                    </section>
                    <section className="analysis-card">
                      <div className="card-title">
                        <h2>Дополнительная съёмка</h2>
                        <span>{zones.length} кандидатных зон</span>
                      </div>
                      <div className="zone-tabs">
                        <button
                          className={zoneView === "plan" ? "active" : ""}
                          onClick={() => setZoneView("plan")}
                        >
                          План C · {chosenZoneIds.size}
                        </button>
                        <button
                          className={zoneView === "all" ? "active" : ""}
                          onClick={() => setZoneView("all")}
                        >
                          Все зоны
                        </button>
                      </div>
                      <div className="zone-list">
                        {zones
                          .filter(
                            (item) =>
                              zoneView === "all" ||
                              chosenZoneIds.has(
                                String(item.properties.candidate_id),
                              ),
                          )
                          .map((item) => (
                            <button
                              key={String(item.properties.candidate_id)}
                              className={
                                zoneId === item.properties.candidate_id
                                  ? "active"
                                  : ""
                              }
                              onClick={() => {
                                setZoneId(String(item.properties.candidate_id));
                                setAssetId(null);
                              }}
                            >
                              <span>
                                <b>{String(item.properties.candidate_id)}</b>
                                <small>
                                  {item.properties.area_km2} км² ·{" "}
                                  {String(
                                    item.properties.sensor_type,
                                  ).toUpperCase()}{" "}
                                  · доступна{" "}
                                  {date(
                                    String(item.properties.available_at ?? ""),
                                  )}
                                </small>
                                {item.properties.exclusion_reason ? (
                                  <small className="zone-excluded">
                                    исключена:{" "}
                                    {String(item.properties.exclusion_reason)}
                                  </small>
                                ) : null}
                              </span>
                              <strong>
                                {money(item.properties.solo_cost_rub)} ₽
                              </strong>
                            </button>
                          ))}
                        {zoneView === "plan" && !chosenZoneIds.size && (
                          <p>При текущем бюджете зон в плане C нет.</p>
                        )}
                      </div>
                      {activeZone && (
                        <div className="card-foot">
                          {plan?.reasons.C[
                            String(activeZone.properties.candidate_id)
                          ] ??
                            plan?.reasons.not_selected_C[
                              String(activeZone.properties.candidate_id)
                            ] ??
                            "Причина отсутствует"}
                        </div>
                      )}
                    </section>
                  </div>
                </div>
                <section className="analysis-card sensitivity">
                  <div className="card-title">
                    <h2>Проверка устойчивости</h2>
                    <span>Сценарии из комплекта</span>
                  </div>
                  <div className="sensitivity-grid">
                    <div>
                      <h3>Бюджет и ставка</h3>
                      {activeSession.sensitivity
                        .filter(
                          (item) =>
                            item.strategy === "C" &&
                            /^(budget|rate)_/.test(item.scenario_id),
                        )
                        .map((item) => (
                          <div className="scenario" key={item.scenario_id}>
                            <span>{formatScenarioLabel(item)}</span>
                            <b>{formatScenarioAmount(item)} охват</b>
                          </div>
                        ))}
                    </div>
                    <div>
                      <h3>Вероятность и ущерб</h3>
                      {activeSession.sensitivity
                        .filter(
                          (item) =>
                            item.strategy === "A" &&
                            /^[pVq]_portfolio_x[0-9.]+$/.test(item.scenario_id),
                        )
                        .map((item) => (
                          <div className="scenario" key={item.scenario_id}>
                            <span>{formatScenarioLabel(item)}</span>
                            <b>{formatScenarioAmount(item)}</b>
                          </div>
                        ))}
                    </div>
                  </div>
                  <div className="card-foot">
                    Чувствительность не пересчитывает карту вероятности воды и
                    не меняет портфель.
                  </div>
                </section>
              </div>
            )}
              </div>
            </div>
          </div>
      </main>
      {readyAnalysisNotice && (
        <div className="analysis-ready-notice">
          <span role="status" aria-live="polite">Анализ {readyAnalysisNotice.territory} готов</span>
          <button onClick={() => {
            navigate("analysis", readyAnalysisNotice.chipId);
            setReadyAnalysisNotice(null);
          }}>Открыть</button>
          <button
            className="analysis-ready-dismiss"
            aria-label="Закрыть уведомление"
            onClick={() => setReadyAnalysisNotice(null)}
          ><X size={15} /></button>
        </div>
      )}
    </div>
  );
}
