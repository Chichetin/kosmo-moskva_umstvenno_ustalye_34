import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  ArrowDownToLine,
  ArrowLeft,
  BarChart3,
  ChevronDown,
  ChevronUp,
  Globe2,
  Layers3,
  Search,
  X,
} from "lucide-react";
import Backdrop from "./Backdrop";
import MapPanel from "./MapPanel";
import { WorldLocator } from "./WorldLocator";
import MessagePanel, { type SelectedMessage } from "./MessagePanel";
import MessagesList from "./MessagesList";
import SensitivityPanel from "./SensitivityPanel";
import RiskMatrix from "./RiskMatrix";
import { finiteNumber, riskAsset } from "./riskAssets";
import NewsPanel from "./NewsPanel";
import PlanZones from "./PlanZones";
import { pointInZone } from "./geometry";
import {
  formatExactProbability,
  formatNumber as money,
  formatProbability,
  formatRub,
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
const date = (value: string | null | undefined) =>
  value
    ? new Intl.DateTimeFormat("ru-RU", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      }).format(new Date(value))
    : "Дата неизвестна";
// Budget scale before a run: plan C's saturation is not known yet; 50 000 ₽ is above full
// coverage B of the known scenes (37–42 thousand).
const PRE_RUN_BUDGET_MAX = 50_000;
const ruPlural = new Intl.PluralRules("ru");
const countLabel = (count: number, one: string, few: string, many: string) =>
  `${count} ${ruPlural.select(count) === "one" ? one : ruPlural.select(count) === "few" ? few : many}`;
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
// Russian names of the Sen1Floods11 event territories: the operator searches by country.
const TERRITORY_RU: Record<string, string> = {
  Bolivia: "Боливия",
  Ghana: "Гана",
  India: "Индия",
  Mekong: "Меконг Камбоджа",
  Nigeria: "Нигерия",
  Pakistan: "Пакистан",
  Paraguay: "Парагвай",
  Somalia: "Сомали",
  Spain: "Испания",
  "Sri-Lanka": "Шри-Ланка Шри Ланка",
  USA: "США Соединённые Штаты",
};
const countryMatches = (territory: string, query: string) => {
  const needle = query.trim().toLocaleLowerCase("ru").replaceAll("ё", "е");
  if (!needle) return true;
  return `${territory} ${territory.replaceAll("-", " ")} ${TERRITORY_RU[territory] ?? ""}`
    .toLocaleLowerCase("ru")
    .replaceAll("ё", "е")
    .includes(needle);
};
const assetName = (feature: Feature) =>
  labels[String(feature.properties.asset_class)] ??
  String(feature.properties.asset_class).replaceAll("_", " ");
type AssetSort = "inspection" | "name" | "probability" | "loss" | "uncertainty";
const assetSortValue = (feature: Feature, key: AssetSort): number | string | null => {
  if (key === "name") return assetName(feature);
  if (key === "inspection") return finiteNumber(feature.properties.rank);
  const raw = feature.properties[key === "probability" ? "p_flood" : key === "loss" ? "expected_loss_rub" : "uncertainty"];
  return feature.properties.status !== "ok" ? null : finiteNumber(raw);
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
  const [planFor, setPlanFor] = useState<{ runId: string; budget: number } | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [budget, setBudget] = useState<number | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
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
  // Data budget the operator sets before the run: plan A/B/C of the bundle is built for it.
  const [analysisBudget, setAnalysisBudget] = useState(5_000);
  const [readyAnalysisNotice, setReadyAnalysisNotice] = useState<{ chipId: string; territory: string } | null>(null);
  const [searchOpen, setSearchOpen] = useState(() => window.innerWidth >= 700);
  const [panelOpen, setPanelOpen] = useState(true);
  const [newsOpen, setNewsOpen] = useState(false);
  const [messagesOpen, setMessagesOpen] = useState(false);
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [overviewRequest, setOverviewRequest] = useState(0);
  const [mapIssue, setMapIssue] = useState<string | null>(null);
  // «Сообщения» (/api/media/map): null — the layer is unavailable here (media not served,
  // or closed to this viewer); the toggle is then hidden rather than shown broken.
  const [messages, setMessages] = useState<MediaMap | null>(null);
  const [message, setMessage] = useState<SelectedMessage | null>(null);
  const [messageFocus, setMessageFocus] = useState<{ lngLat: [number, number]; zoom: number; request: number } | null>(null);
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
  const [assetSort, setAssetSort] = useState<{ key: AssetSort; descending: boolean }>({ key: "inspection", descending: false });
  const scene = scenes.find((item) => item.chip_id === route.chipId) ?? null;
  const locatorPosition = scene?.bounds
    ? { lon: (scene.bounds[0] + scene.bounds[2]) / 2, lat: (scene.bounds[1] + scene.bounds[3]) / 2 }
    : scene?.position ?? null;
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
      setPlanFor(null);
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
    setPlanFor(null);
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
    setPlanFor(null);
    setPlanError(null);
    const controller = new AbortController();
    const timer = setTimeout(() => {
      json<Plan>(
        `/api/plan?budget=${budget}&run_id=${encodeURIComponent(runId)}`,
        controller.signal,
      )
        .then((result) => {
          if (!controller.signal.aborted) {
            setPlan(result);
            setPlanFor({ runId, budget });
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
  useEffect(() => setExportError(null), [activeSession?.run_id, budget]);
  const openFinishedAnalysis = useCallback(async (finished: AnalysisJob) => {
    const updated = await json<SceneSummary[]>("/api/scenes");
    setScenes(updated);
    const target = updated.find((item) => item.chip_id === finished.chip_id && item.run_id);
    if (!target) {
      throw new Error("Расчёт завершён, но результат ещё не появился в каталоге. Обновите его повторно.");
    }
    const here = parseRoute();
    if (here.chipId === finished.chip_id) {
      // On the map the results appear in the side panel («Подробный анализ» is the operator's own
      // step); on the analysis page they fill the page in place — no jump to the map.
      setReadyAnalysisNotice(null);
      if (here.view !== "analysis") {
        setPanelOpen(true);
        setPanelCollapsed(false);
        navigate("scene", finished.chip_id);
      }
    } else {
      setReadyAnalysisNotice({ chipId: finished.chip_id, territory: target.territory.replaceAll("-", " ") });
    }
  }, []);
  const startAnalysis = useCallback(async (target: SceneSummary) => {
    setAnalysisStarting(target.chip_id);
    setAnalysisJobError(null);
    try {
      const result = await json<AnalysisJob>(
        `/api/analyses?chip_id=${encodeURIComponent(target.chip_id)}&budget=${analysisBudget}`,
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
  }, [openFinishedAnalysis, analysisBudget]);
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
    setPanelCollapsed(false);
    setSearchOpen(false);
    setAssetId(null);
    setZoneId(null);
    setPlan(null);
    navigate("scene", selected.chip_id);
  }, []);
  // Leave the observation: the scene overlay folds back into its point, all points
  // return; the camera keeps its place and zoom (no overviewRequest).
  const leaveScene = () => {
    setAssetId(null);
    setZoneId(null);
    setPlan(null);
    setPanelCollapsed(false);
    setSearchOpen(window.innerWidth >= 700);
    navigate("earth");
  };
  const overview = () => {
    setMessageFocus(null);
    navigate("earth");
    setSearchOpen(true);
    setOverviewRequest((value) => value + 1);
  };
  const selectMessageFromList = (selected: SelectedMessage) => {
    setNewsOpen(false);
    setMessagesOpen(false);
    setMessage(selected);
    setMessageFocus((previous) => ({
      lngLat: selected.lngLat,
      zoom: selected.kind === "report" ? 10 : 9,
      request: (previous?.request ?? 0) + 1,
    }));
    if (!mapActive || scene) navigate("earth");
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
  const inspectionPriority = activeSession?.metadata.expected_loss?.inspection_priority;
  const inspection = useMemo(() => new Map((inspectionPriority?.order ?? []).map(row => [row.asset_id, row])), [inspectionPriority]);
  const unassessed = assets.filter(item => !riskAsset(item));
  const entropy = /^(entropy_p|binary_entropy_bits)(?:$|[\s(])/.test(String(activeSession?.metadata.uncertainty?.method ?? ""));
  const selectRiskAsset = (id: string) => { setAssetId(id); setZoneId(null); };
  const sortedAssets = useMemo(() => [...assets].sort((left, right) => {
    const value = (item: Feature) => assetSort.key === "inspection" && inspection.size
      ? inspection.get(String(item.properties.asset_id))?.inspection_order ?? null : assetSortValue(item, assetSort.key);
    const a = value(left);
    const b = value(right);
    if (a == null || (typeof a === "number" && !Number.isFinite(a))) return b == null ? 0 : 1;
    if (b == null || (typeof b === "number" && !Number.isFinite(b))) return -1;
    const comparison = typeof a === "string" && typeof b === "string"
      ? a.localeCompare(b, "ru") : Number(a) - Number(b);
    return (assetSort.descending ? -comparison : comparison) ||
      String(left.properties.asset_id).localeCompare(String(right.properties.asset_id), "ru");
  }), [assets, assetSort, inspection]);
  const toggleAssetSort = (key: AssetSort) => setAssetSort((current) => ({
    key, descending: current.key === key ? !current.descending : key !== "name",
  }));
  const zones = activeSession?.candidates.features ?? [];
  const currentPlan = planFor && planFor.runId === activeSession?.run_id && planFor.budget === budget ? plan : null;
  const chosen = currentPlan?.comparison.find((row) => row.strategy === "C");
  const chosenZoneIds = new Set(currentPlan?.plans.C ?? []);
  // Slider scale: fixed for the scene (does not move while typing), ending at the point where
  // plan C stops changing, rounded up to a thousand (15 940 → 16 000).
  const sliderMax = Math.min(1_000_000_000, Math.ceil(Math.max(
    activeSession?.plan_c_saturation_rub ?? 0, activeSession?.metadata.budget.budget_rub ?? 0,
    1_000) / 1_000) * 1_000);
  // The operator knows the country, not a Sen1Floods11 chip id: search by country only,
  // in English (as in the catalog) or Russian.
  const matchingScenes = scenes.filter((item) => countryMatches(item.territory, query));
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
    .slice(0, query ? undefined : 10);
  const analysisLocations = scenes.filter((item) => countryMatches(item.territory, analysisQuery));
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
    const asking = !running && current?.status !== "done";
    return (
      <div className="analysis-job">
        {asking && (
          <label className="analysis-budget-ask">
            <span>Бюджет на дополнительную съёмку, ₽</span>
            <input
              type="number"
              min="0"
              max="1000000000"
              step="100"
              value={analysisBudget}
              disabled={analysisStarting === target.chip_id}
              onChange={(event) =>
                setAnalysisBudget(Math.min(1_000_000_000, Math.max(0, Number(event.target.value) || 0)))
              }
            />
            <input
              type="range"
              min="0"
              max={PRE_RUN_BUDGET_MAX}
              step="100"
              value={Math.min(analysisBudget, PRE_RUN_BUDGET_MAX)}
              disabled={analysisStarting === target.chip_id}
              onChange={(event) => setAnalysisBudget(Number(event.target.value))}
              aria-label="Бюджет на дополнительную съёмку"
            />
            <small>
              Карта воды и ущерб считаются по уже полученному снимку S1 — за него не платим.
              Бюджет ограничивает только дозаказ снимков: под него строится план C. После
              расчёта бюджет можно изменить.
            </small>
          </label>
        )}
        <button
          className="analysis-start"
          disabled={running || analysisStarting === target.chip_id}
          onClick={() => void startAnalysis(target)}
        >
          {running ? "Анализ выполняется" : analysisStarting === target.chip_id
            ? "Отправляем запрос…" : current?.status === "failed" ? "Повторить анализ"
            : `Запустить анализ · ${formatRub(analysisBudget)}`}
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
  const totalLoss = assets.some(item => riskAsset(item)) ? assets.reduce(
    (sum, item) => sum + (riskAsset(item) ? finiteNumber(item.properties.expected_loss_rub) ?? 0 : 0),
    0,
  ) : null;
  const planRowsC = new Map((currentPlan?.plan_rows ?? [])
    .filter((row) => row.strategy === "C")
    .map((row) => [String(row.candidate_id), row]));
  const downloadCurrentPlan = async () => {
    if (!activeSession || budget === null || !currentPlan || exporting) return;
    setExporting(true);
    setExportError(null);
    try {
      const query = new URLSearchParams({ run_id: activeSession.run_id, budget: String(budget) });
      const response = await fetch(`/api/download?${query}`);
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ?? `Ошибка выгрузки ${response.status}`);
      }
      const match = /filename="?([^";]+)"?/i.exec(response.headers.get("Content-Disposition") ?? "");
      const filename = match?.[1].split(/[\\/]/).pop() || `${activeSession.metadata.chip_id}_${budget}.zip`;
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (error) {
      setExportError(error instanceof Error ? error.message : String(error));
    } finally {
      setExporting(false);
    }
  };
  return (
    <div className="app">
      <Backdrop paused={!mapActive || !!scene} ocean={baseMap === "optical"} />
      <header className={`topbar ${mapVisible ? "topbar-map" : "topbar-analysis"}${scene ? " topbar-scene" : ""}`}>
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
          <button
            className={`nav-news${newsOpen ? " active" : ""}`}
            onClick={() => { setMessagesOpen(false); setMessage(null); setNewsOpen(!newsOpen); }}
            aria-expanded={newsOpen}
            title="Все новости и сводки, из которых берётся слой «Сообщения»"
          >
            Новости
          </button>
          <button
            className={`nav-messages${messagesOpen ? " active" : ""}`}
            onClick={() => { setNewsOpen(false); setMessage(null); setMessagesOpen(!messagesOpen); }}
            aria-expanded={messagesOpen}
            title={`Сообщения на карте: ${messageCount}`}
          >
            Сообщения{messageCount > 0 && <span className="nav-count"> · {messageCount}</span>}
          </button>
        </nav>
        <span className="top-spacer" />
        {activeSession && (
          <button
            className="top-export"
            type="button"
            disabled={!currentPlan || exporting}
            aria-busy={exporting}
            onClick={() => void downloadCurrentPlan()}
            title="Выгрузить файлы этой сцены и выбранного бюджета"
          >
            <ArrowDownToLine size={16} />
            <span>{exporting ? "Готовим комплект…" : "Выгрузить комплект"}</span>
            <small className="export-note">при бюджете {formatRub(budget)}</small>
          </button>
        )}
      </header>
      {exportError && <div className="export-error" role="alert">
        {exportError}
        <button onClick={() => setExportError(null)} aria-label="Скрыть ошибку выгрузки"><X size={14} /></button>
      </div>}
      {newsOpen && (
        <NewsPanel
          onClose={() => setNewsOpen(false)}
          scenes={scenes}
          onShowOnMap={(lngLat, zoom) => {
            // leave a scene or the analysis first, then fly (the scene exit moves the camera too)
            const leaving = route.view !== "map" || !!route.chipId || sourceViewerOpen;
            setSourceViewerOpen(false);
            if (leaving) navigate("earth");
            if (window.innerWidth < 700) setNewsOpen(false);
            window.setTimeout(
              () => setMessageFocus((f) => ({ lngLat, zoom, request: (f?.request ?? 0) + 1 })),
              leaving ? 700 : 0,
            );
          }}
          onOpenScene={(selected) => {
            setNewsOpen(false);
            setSourceViewerOpen(false);
            selectScene(selected);
          }}
        />
      )}
      {messagesOpen && <MessagesList messages={messages} onClose={() => setMessagesOpen(false)} onSelect={selectMessageFromList} />}
      <main className="workspace">
        <div
          className={`map-screen ${mapVisible ? "" : "map-hidden"} ${scene ? "scene-mode" : ""} ${baseMap === "optical" ? "optical-mode" : ""} ${scene && panelOpen && (message || newsOpen || messagesOpen) ? "tiled" : ""}`}
          style={{ "--tile-right": newsOpen ? "30.25rem" : "26.5rem" } as CSSProperties}
          aria-hidden={!mapVisible}
          inert={!mapVisible}
        >
          <MapPanel
            scenes={scenes}
            scene={scene}
            session={activeSession}
            plan={currentPlan}
            baseMap={baseMap}
            visibleLayers={layers}
            messages={messages}
            showMessages={true}
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
              setPanelCollapsed(false);
            }}
            onSelectZone={(id) => {
              setZoneId(id);
              setAssetId(null);
              setPanelOpen(true);
              setPanelCollapsed(false);
            }}
            onIssue={setMapIssue}
            onSelectMessage={(selected) => {
              setNewsOpen(false);
              setMessagesOpen(false);
              setMessage(selected);
            }}
            focus={messageFocus}
            newsOpen={newsOpen}
          />
          {message && mapVisible && (
            <MessagePanel
              message={message}
              onClose={() => setMessage(null)}
              onZoom={(lngLat, zoom) => setMessageFocus((f) => ({ lngLat, zoom, request: (f?.request ?? 0) + 1 }))}
            />
          )}
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
                  placeholder="Страна"
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
                    ? `${filtered.length}`
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
          </div>
          {scene && panelOpen && (
            <aside className={`context-panel${panelCollapsed ? " collapsed" : ""}`}>
              <div className="context-heading">
                <div>
                  <small>НАБЛЮДЕНИЕ · SENTINEL-1</small>
                  <h1>{scene.territory.replaceAll("-", " ")}</h1>
                  <p>
                    {date(scene.observation_date)} ·{" "}
                    <code>{scene.chip_id}</code>
                  </p>
                </div>
                <div className="context-heading-actions">
                  <button
                    onClick={() => setPanelCollapsed(!panelCollapsed)}
                    aria-expanded={!panelCollapsed}
                    aria-label={panelCollapsed ? "Развернуть панель" : "Свернуть панель"}
                    title={panelCollapsed ? "Развернуть" : "Свернуть"}
                  >
                    {panelCollapsed ? <ChevronDown size={17} /> : <ChevronUp size={17} />}
                  </button>
                  <button
                    onClick={leaveScene}
                    aria-label="Закрыть наблюдение"
                    title="Закрыть наблюдение"
                  >
                    <X size={17} />
                  </button>
                </div>
              </div>
              <button className="back-to-globe" onClick={overview}>
                <Globe2 size={15} /> Все наблюдения
              </button>
              {!panelCollapsed && (<>
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
                  </>
                )}
              </section>
              {activeSession && (
                <>
                  <section className="context-section">
                    <h2>План дополнительной съёмки</h2>
                    <p className="zone-map-note">Зелёным — зоны плана C. Контуры расчётные, не форма снимка.</p>
                    <label className="budget-field">
                      Бюджет данных, ₽
                      <input
                        type="number"
                        min="0"
                        max="1000000000"
                        step="100"
                        disabled={exporting}
                        value={budget ?? 0}
                        onChange={(event) =>
                          setBudget(
                            Math.min(1_000_000_000, Math.max(0, Number(event.target.value) || 0)),
                          )
                        }
                      />
                    </label>
                    <input
                      className="budget-slider"
                      type="range"
                      min="0"
                      max={sliderMax}
                      step="100"
                      disabled={exporting}
                      value={Math.min(budget ?? 0, sliderMax)}
                      onChange={(event) =>
                        setBudget(Number(event.target.value))
                      }
                      aria-label="Изменить бюджет"
                    />
                    {activeSession.plan_c_saturation_rub != null && (
                      <p className="budget-saturation">
                        Свыше ≈ {formatRub(activeSession.plan_c_saturation_rub)} состав плана C не меняется.
                      </p>
                    )}
                    <div className="plan-summary">
                      <span>План C · {currentPlan ? countLabel(currentPlan.plans.C.length, "зона", "зоны", "зон") : "—"}</span>
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
                    {planError ? <p className="inline-error">{planError}</p>
                      : !currentPlan && <p role="status">Пересчитываем план…</p>}
                    <button
                      className="analysis-link"
                      onClick={() => {
                        setSourceViewerOpen(false);
                        navigate("analysis", scene.chip_id);
                      }}
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
              <details className="context-foot">
                <summary>Пояснения к карте</summary>
                Маркеры — места снимков, не подтверждённые зоны затопления.
                Оптическая мозаика 2019 года — географический контекст; дата
                отдельного пикселя неизвестна.
              </details>
              </>)}
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
        </div>
          <div
            className={`analysis-screen ${mapVisible ? "analysis-hidden" : ""} ${newsOpen || messagesOpen ? "with-side-window" : ""}`}
            style={{ "--tile-right": newsOpen ? "30.25rem" : "26.5rem" } as CSSProperties}
            aria-hidden={mapVisible}
            inert={mapVisible}
          >
            <div className="analysis-shell">
              <div className="analysis-side">
              {/* Screen-wide actions of the analysis page: above the locations, not in a scene's head. */}
              <nav className="analysis-rail" aria-label="Переход с экрана анализа">
                <button onClick={() => (scene ? navigate("scene", scene.chip_id) : overview())}>
                  <ArrowLeft size={15} /> К карте
                </button>
                <button onClick={overview}>
                  <Globe2 size={15} /> Все наблюдения
                </button>
              </nav>
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
                    placeholder="Страна"
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
              </div>
              <div className="analysis-content">
            <div className="analysis-head">
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
              {scene && locatorPosition && (
                <WorldLocator
                  lon={locatorPosition.lon}
                  lat={locatorPosition.lat}
                  label={scene.territory.replaceAll("-", " ")}
                />
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
                <div className="analysis-metrics" aria-label="Краткое резюме плана C">
                  <div><span>Зон в плане C</span><strong>{currentPlan ? currentPlan.plans.C.length : "—"}</strong></div>
                  <div><span>Стоимость данных</span><strong>{formatRub(chosen?.data_cost_rub)}</strong></div>
                  <div><span>Охват ожидаемого ущерба</span><strong>{pct(chosen?.coverage_share)}</strong></div>
                </div>
                <PlanZones
                  session={activeSession}
                  plan={currentPlan}
                  budget={budget}
                  assets={assets}
                  selectedZone={zoneId}
                  onSelectZone={(id) => {
                    setZoneId(id);
                    setAssetId(null);
                  }}
                  assetName={assetName}
                  observationDate={date(scene?.observation_date ?? activeSession.observation_date)}
                />
                <div className="operator-context">
                  <strong>Учебный портфель · {assets.length} объектов</strong>
                  <p>Стоимость и уязвимость заданы сценарием. Карта показывает воду, включая постоянную; ожидаемый ущерб не является подтверждёнными потерями.</p>
                  <p>Срок решения: <strong>{dateTimeUtc(activeSession.metadata.budget.decision_deadline)}</strong>. Цены и сроки получения дополнительных снимков сценарные.</p>
                </div>
                {unassessed.length > 0 && <div className="risk-missing" role="status">
                  <strong>Требуют проверки: недостаточно данных · {unassessed.length}</strong>
                  <p>Состояние неизвестно. Эти объекты не входят в оценённый ущерб и сценарные суммы; проверьте их осмотром или дополнительным снимком.</p>
                  <div>{unassessed.map(item => <button key={String(item.properties.asset_id)} onClick={() => selectRiskAsset(String(item.properties.asset_id))}>{assetName(item)} · {String(item.properties.asset_id)}{item.properties.status === "partial" ? " · неполные данные" : " · нет оценки"}</button>)}</div>
                </div>}
                <div className="analysis-columns">
                  <section className="analysis-card">
                    <div className="card-title">
                      <h2>Приоритет осмотра</h2>
                      <span>{assets.length - unassessed.length} из {assets.length} оценено · ущерб {formatRub(totalLoss)}</span>
                    </div>
                    <details className="analysis-explain">
                      <summary>Что означает неопределённость карты?</summary>
                      {entropy ? (<>
                      <p>
                        Насколько модель сомневается, есть ли вода в точке объекта. Считается по
                        вероятности воды p (энтропия, в битах от 0 до 1): 0 — модель уверена
                        (p около 0 или 1), 1 — не знает (p = 50 %).
                      </p>
                      <p>
                        Высокая неопределённость — повод подтвердить объект осмотром или
                        дополнительным снимком; снимок там может изменить решение «ехать / не ехать», поэтому такие объекты чаще попадают в план съёмки C (он выбирает по цене ошибки решения).
                      </p>
                      <p>
                        Диапазон вероятности по пяти моделям ансамбля (от минимума до максимума) показывает, как
                        сильно они расходятся; если порог маски попадает внутрь диапазона, маска в
                        точке неустойчива.
                      </p>
                      <p>Это свойство карты воды в точке объекта: оно не подтверждает затопление и не показывает ущерб.</p>
                      </>) : (<>
                        <p>{activeSession.metadata.uncertainty?.meaning ?? "Методика неопределённости не указана в комплекте."}</p>
                        {activeSession.metadata.uncertainty?.units && <p>Единицы: {activeSession.metadata.uncertainty.units}.</p>}
                      </>)}
                    </details>
                    <div className="inspection-controls">
                      {inspection.size > 0 ? <><button className={assetSort.key === "inspection" ? "active" : ""} onClick={() => setAssetSort({ key: "inspection", descending: false })}>По очередности осмотра</button><details><summary>Правило осмотра</summary><p>{inspectionPriority?.rule}</p></details></> : <p>Рекомендации осмотра не сохранены в этом комплекте. Для сравнения доступны сортировки по ущербу и вероятности; это не готовое решение о выезде.</p>}
                    </div>
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            {([
                              ["name", "Объект"],
                              ["probability", "Вероятность воды"],
                              ["loss", "Ожидаемый ущерб"],
                              ["uncertainty", "Неопределённость карты"],
                            ] as const).map(([key, label]) => (
                              <th key={key} aria-sort={assetSort.key === key ? assetSort.descending ? "descending" : "ascending" : "none"}>
                                <button className="asset-sort" onClick={() => toggleAssetSort(key)}>
                                  {label}<span aria-hidden="true">{assetSort.key === key ? assetSort.descending ? "↓" : "↑" : "↕"}</span>
                                </button>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {sortedAssets.map((item) => (
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
                                  {String(item.properties.asset_id)}
                                  {!riskAsset(item) && " · нет оценки"}
                                  {inspection.get(String(item.properties.asset_id)) && ` · осмотр № ${inspection.get(String(item.properties.asset_id))!.inspection_order}`}
                                </small>
                                <small className="asset-action">{!riskAsset(item) ? "Проверить на месте или дополнительным снимком" : inspection.get(String(item.properties.asset_id))?.action}</small>
                              </td>
                              <td>
                                {!riskAsset(item)
                                  ? "—"
                                  : <span title={`Точное p: ${formatExactProbability(item.properties.p_flood)}`}>
                                      {formatProbability(item.properties.p_flood)}
                                    </span>}
                              </td>
                              <td>
                                {riskAsset(item) ? formatRub(item.properties.expected_loss_rub) : "—"}
                              </td>
                              <td>
                                {!riskAsset(item) || item.properties.uncertainty == null ||
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
                      <div className="card-foot asset-detail">
                        <div><strong>{assetName(activeAsset)} · {String(activeAsset.properties.asset_id)}</strong>
                          <span>Стоимость {formatRub(activeAsset.properties.asset_value_rub)} · уязвимость {money(activeAsset.properties.vulnerability_coef, 2)} · точное p {riskAsset(activeAsset) ? formatExactProbability(activeAsset.properties.p_flood) : "нет оценки"}</span>
                          <span>{!riskAsset(activeAsset) ? "Недостаточно данных — нужен осмотр или дополнительный снимок." : inspection.get(String(activeAsset.properties.asset_id))?.action ?? "Рекомендация отсутствует в комплекте."}</span>
                          {riskAsset(activeAsset) && inspection.get(String(activeAsset.properties.asset_id))?.reason && <details><summary>Почему такой приоритет</summary><p>{inspection.get(String(activeAsset.properties.asset_id))?.reason}</p></details>}
                        </div>
                        <button onClick={() => {
                          setLayers((current) => ({ ...current, assets: true }));
                          setPanelOpen(true);
                          setSourceViewerOpen(true);
                        }}>Показать на карте</button>
                      </div>
                    )}
                    <RiskMatrix assets={assets} selected={assetId} onSelect={selectRiskAsset} name={assetName} entropy={entropy} />
                  </section>
                  <div className="analysis-stack">
                    <section className="analysis-card">
                      <div className="card-title">
                        <h2>План дополнительной съёмки</h2>
                        <span>Выбран вариант C</span>
                      </div>
                      <div className="analysis-budget">
                        <label>
                          Бюджет данных
                          <span><input
                            type="number"
                            min="0"
                            max="1000000000"
                            step="100"
                            disabled={exporting}
                            value={budget ?? 0}
                            onChange={(event) => setBudget(Math.min(1_000_000_000, Math.max(0, Number(event.target.value) || 0)))}
                          /> ₽</span>
                        </label>
                        <input
                          type="range"
                          min="0"
                          max={sliderMax}
                          step="100"
                          disabled={exporting}
                          value={Math.min(budget ?? 0, sliderMax)}
                          onChange={(event) => setBudget(Number(event.target.value))}
                          aria-label="Бюджет данных плана"
                        />
                        {activeSession.plan_c_saturation_rub != null && (
                          <p>Свыше ≈ {formatRub(activeSession.plan_c_saturation_rub)} состав плана C не меняется.</p>
                        )}
                      </div>
                      {planError ? <p className="inline-error">{planError}</p>
                        : !currentPlan && <p className="analysis-plan-loading" role="status">Пересчитываем план…</p>}
                      {chosen && <div className="analysis-plan-totals">
                        <div><span>Расчёт и проверка зон</span><strong>{formatRub(chosen.other_cost_rub)}</strong></div>
                        <div><span>Полная стоимость решения</span><strong>{formatRub(chosen.decision_cost_rub)}</strong></div>
                      </div>}
                      <h3 className="analysis-compare-title">Сравнение A / B / C</h3>
                      {(currentPlan?.comparison ?? []).map((row) => (
                        <div
                          key={row.strategy}
                          className={`strategy ${row.strategy === "C" ? "chosen" : ""} ${!row.budget_feasible ? "overbudget" : ""}`}
                        >
                          <b>{row.strategy}</b>
                          <div>
                            <strong className="strategy-name">
                              {row.strategy === "A"
                                ? "Текущие данные"
                                : row.strategy === "B"
                                  ? "Сплошная съёмка"
                                  : "Приоритетный заказ"}
                            </strong>
                            <div className="strategy-values">
                              <span>Данные <strong>{formatRub(row.data_cost_rub)}</strong></span>
                              <span>Полная стоимость <strong>{formatRub(row.decision_cost_rub)}</strong></span>
                              <span>Охват <strong>{pct(row.coverage_share)}</strong></span>
                              <span>Доля R <strong>{pct(row.decision_risk_share)}</strong></span>
                            </div>
                            <small>{row.budget_feasible ? "В бюджете данных" : "Превышен бюджет данных"}</small>
                            <div className="strategy-uncertainty">
                              <span>Остаточная неопределённость <strong>{row.residual_uncertainty == null ? "—" : `${money(row.residual_uncertainty, 3)} ${entropy ? "бит · сумма по объектам" : activeSession.metadata.uncertainty?.units ?? "ед. показателя"}`}</strong></span>
                              <small>{({ baseline: "Исходная оценка", scenario: "Сценарий · не измеренное улучшение", measured: "Измерено", not_estimated: "Не оценена" } as Record<string, string>)[row.uncertainty_status] ?? "Статус не указан"}</small>
                              {row.uncertainty_basis && <details><summary>Основание оценки</summary><p>{row.uncertainty_basis}</p></details>}
                            </div>
                            {row.strategy === "B" && !row.budget_feasible && (
                              <small>
                                Выше бюджета · сравнение, не рекомендация
                              </small>
                            )}
                          </div>
                        </div>
                      ))}
                      <div className="card-foot">
                        <p>Охват — доля ожидаемого ущерба объектов в зонах, а не предотвращённый ущерб.</p>
                        <p>Цены рассчитаны по сценарной ставке {money(activeSession.metadata.pricing?.base_rate_rub_km2, 2)} ₽/км²; это не тариф.</p>
                        <details>
                          <summary>Методика расчёта</summary>
                          <p>Сценарная цена ошибки решения R = V·q·min(p, 1−p). Показана доля R объектов в зонах от R всех оценённых объектов; снижение фактических потерь не заявляется.</p>
                          <p>Допущения R: одинаковая цена двух типов ошибок и идеальное уточнение снимком. Реальные стоимость выезда, цена пропуска и качество нового снимка здесь не моделируются; это не подтверждённая экономия.</p>
                          <p>План C не покупает новые данные ради объектов, для которых исход по карте почти определён: их осмотр планируется по исходным данным.</p>
                        </details>
                        <details>
                          <summary>Воспроизводимость</summary>
                          <p>Выгрузка пересчитывает план для выбранного бюджета {formatRub(budget)}. Вероятность воды, маска и порог остаются прежними; метод и входные данные записаны в комплекте.</p>
                        </details>
                      </div>
                    </section>
                    <section className="analysis-card">
                      <div className="card-title">
                        <h2>Зоны заказа</h2>
                        <span>{countLabel(zones.length, "зона", "зоны", "зон")}</span>
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
                      <div className="zone-list" aria-label={zoneView === "plan" ? "Зоны плана C" : "Все зоны заказа"}>
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
                                  {money(item.properties.area_km2, 3)} км² ·{" "}
                                  {String(
                                    item.properties.sensor_type,
                                  ).toUpperCase()}{" "}
                                  · доступна{" "}
                                  {dateTimeUtc(
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
                              <span className="zone-cost">
                                <strong>{formatRub(planRowsC.get(String(item.properties.candidate_id))?.cost_rub ?? item.properties.solo_cost_rub)}</strong>
                                <small>{planRowsC.has(String(item.properties.candidate_id)) ? "в корзине C" : "отдельный заказ"}</small>
                              </span>
                            </button>
                          ))}
                        {zoneView === "plan" && currentPlan && !chosenZoneIds.size && (
                          <p>При текущем бюджете зон в плане C нет.</p>
                        )}
                      </div>
                      {activeZone && (
                        <div className="card-foot zone-detail">
                          <span>{currentPlan?.reasons.C[
                            String(activeZone.properties.candidate_id)
                          ] ??
                            currentPlan?.reasons.not_selected_C[
                              String(activeZone.properties.candidate_id)
                            ] ??
                            "Причина выбора недоступна"}</span>
                          <button onClick={() => {
                            setPanelOpen(true);
                            setSourceViewerOpen(true);
                          }}>Показать на карте</button>
                        </div>
                      )}
                    </section>
                  </div>
                </div>
                <SensitivityPanel
                  rows={activeSession.sensitivity}
                  baseBudget={activeSession.metadata.budget.budget_rub}
                  baseLoss={totalLoss}
                />
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
            setPanelOpen(true);
            setPanelCollapsed(false);
            navigate(parseRoute().view === "analysis" ? "analysis" : "scene", readyAnalysisNotice.chipId);
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
