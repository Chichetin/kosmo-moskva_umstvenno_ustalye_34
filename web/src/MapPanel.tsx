import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, Map } from "maplibre-gl";
import type { FeatureCollection, Geometry, Point, Polygon } from "geojson";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import { mapProviders } from "./mapProviders";
import { POLAR_ICE } from "./polarIce";
import type { Feature, MediaMap, Plan, SceneSummary, Session } from "./types";
import type { SelectedMessage } from "./MessagePanel";
import { ORIGIN, REPORT_TYPE, REVIEW_STATE, when } from "./mediaLabels";

maplibregl.setWorkerUrl(workerUrl);
type LayerFlags = {
  s1: boolean;
  probability: boolean;
  mask: boolean;
  uncertainty: boolean;
  assets: boolean;
  zones: boolean;
};
type Props = {
  scenes: SceneSummary[];
  scene: SceneSummary | null;
  session: Session | null;
  plan: Plan | null;
  baseMap: "map" | "optical";
  visibleLayers: LayerFlags;
  messages: MediaMap | null;
  showMessages: boolean;
  overlayOpacity: number;
  selectedAsset: string | null;
  selectedZone: string | null;
  active: boolean;
  overviewRequest: number;
  onSelectScene: (scene: SceneSummary) => void;
  onSelectAsset: (id: string) => void;
  onSelectZone: (id: string) => void;
  onIssue: (message: string | null) => void;
  onSelectMessage?: (message: SelectedMessage) => void;
  focus?: { lngLat: [number, number]; zoom: number; request: number } | null;
};
const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
const features = (items: Feature[]): FeatureCollection => ({
  type: "FeatureCollection",
  features: items.flatMap((item) => {
    let geometry: Geometry;
    if (item.geometry.type === "Point") {
      geometry = {
        type: "Point",
        coordinates: item.geometry.coordinates as number[],
      } as Point;
    } else if (item.geometry.type === "Polygon") {
      geometry = {
        type: "Polygon",
        coordinates: item.geometry.coordinates as number[][][],
      } as Polygon;
    } else return [];
    return [
      { type: "Feature" as const, geometry, properties: item.properties },
    ];
  }),
});
const markers = (scenes: SceneSummary[]): FeatureCollection<Point> => ({
  type: "FeatureCollection",
  features: scenes.flatMap((scene) =>
    scene.position
      ? [
          {
            type: "Feature" as const,
            geometry: {
              type: "Point" as const,
              coordinates: [scene.position.lon, scene.position.lat],
            },
            properties: { chip_id: scene.chip_id, status: scene.status },
          },
        ]
      : [],
  ),
});
// «Сообщения» layer (/api/media/map): a reason to look at a territory, never an input to p or EL.
const mediaFeatures = (items: Feature[]): FeatureCollection => ({
  type: "FeatureCollection",
  features: items.flatMap((item) =>
    item.geometry
      ? [{ type: "Feature" as const, geometry: item.geometry as Geometry, properties: item.properties }]
      : [],
  ),
});
// Text nodes only: titles come from news and eyewitnesses and are never parsed as HTML.
const messageCard = (props: Record<string, unknown>, kind: "event" | "report") => {
  const node = document.createElement("div");
  node.className = "map-preview media-preview";
  const line = (text: string, tag: "strong" | "span" | "small" = "span") => {
    const el = document.createElement(tag);
    el.textContent = text;
    node.append(el);
  };
  const demo = props.is_demo ? (props.is_replay ? "ДЕМО · REPLAY · " : "ДЕМО · ") : "";
  if (kind === "report") {
    line(`${demo}Обращение очевидца`, "strong");
    line(String(props.category_label ?? "без категории"));
    const coarse = props.location_precision === "coarsened";
    line(`${when(props.observed_at ?? props.submitted_at, props.observed_at ? props.observed_precision : "hour", coarse)} · ждёт модерации`);
    if (coarse) line("место показано с точностью ≈ 1 км, время — до часа", "small");
  } else {
    line(`${demo}${String(props.title ?? "Сообщение")}`, "strong");
    const origins = String(props.source_kinds ?? "")
      .split(",").filter(Boolean).map((k) => ORIGIN[k] ?? k).join(", ");
    line(`${REPORT_TYPE[String(props.report_type)] ?? "тип неизвестен"} · ${origins || "источник неизвестен"}`);
    if (props.place_label) line(String(props.place_label));
    const coarse = props.geo_precision === "coarsened";
    line(`${when(props.event_time_start ?? props.first_detected_at, props.event_time_start ? props.time_precision : "hour", coarse)} · ${REVIEW_STATE[String(props.review_state)] ?? ""}`);
    if (coarse) line("место показано с точностью ≈ 1 км, время — до часа; подробности — у аналитика", "small");
  }
  line("Сообщение не проверено по снимкам и не входит в p, ущерб и план.", "small");
  return node;
};
const reducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const minimumZoom = () => (window.innerWidth < 700 ? 0.9 : 1.15);
const overviewZoom = () => (window.innerWidth < 700 ? 1.15 : 2.05);
// The globe sits on the layout grid axis (minimal.css «Layout grid»): the centre of the area
// right of the search panel. The offset is the panel's rendered edge, so the CSS zoom step
// and gutter are accounted for exactly as the navigation above the globe sees them.
const overviewPadding = () => {
  const rail = document.querySelector(".map-top-left")?.getBoundingClientRect().right ?? 0;
  return {
    top: 0,
    right: 0,
    bottom: 0,
    left: window.innerWidth <= 700 ? 0 : Math.max(0, Math.round(rail)),
  };
};
const noPadding = () => ({ top: 0, right: 0, bottom: 0, left: 0 });

export default function MapPanel(props: Props) {
  const host = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const current = useRef(props);
  current.current = props;
  const popup = useRef<maplibregl.Popup | null>(null);
  const lastScene = useRef<string | null>(null);
  const lastOverview = useRef(0);
  const [ready, setReady] = useState(0);
  const [unavailable, setUnavailable] = useState<string | null>(null);

  const updateVisibility = () => {
    const map = mapRef.current;
    if (!map || !map.getLayer("boundary")) return;
    const {
      scene,
      baseMap,
      visibleLayers,
      overlayOpacity,
      selectedAsset,
      selectedZone,
      messages,
      showMessages,
    } = current.current;
    const show: Record<string, boolean> = {
      optical: baseMap === "optical",
      cluster: !scene,
      "cluster-count": !scene,
      observation: !scene,
      analyzed: !scene,
      s1: !!scene && visibleLayers.s1,
      probability: !!scene && visibleLayers.probability,
      mask: !!scene && visibleLayers.mask,
      uncertainty: !!scene && visibleLayers.uncertainty,
      boundary: !!scene,
      assets: !!scene && visibleLayers.assets,
      "asset-focus": !!scene && visibleLayers.assets,
      candidates: !!scene && visibleLayers.zones,
      "candidate-hit": !!scene && visibleLayers.zones,
      "plan-fill": !!scene && visibleLayers.zones,
      "plan-line": !!scene && visibleLayers.zones,
      "zone-focus": !!scene && visibleLayers.zones,
      "media-area": showMessages && !!messages,
      "media-event": showMessages && !!messages,
      "media-report": showMessages && !!messages,
    };
    for (const [id, visible] of Object.entries(show)) {
      if (map.getLayer(id))
        map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
    }
    // Global view needs only the largest country names. Detailed map labels
    // return when an observation is selected.
    for (const id of [
      "water_name", "road_oneway", "road_oneway_opposite",
      "highway_name_other", "highway_name_motorway", "place_other",
      "place_suburb", "place_village", "place_town", "place_city",
      "place_city_large", "place_state", "place_country_other",
      "place_country_minor",
    ]) {
      if (map.getLayer(id))
        map.setLayoutProperty(id, "visibility", scene ? "visible" : "none");
    }
    for (const id of [
      "waterway", "road_pier", "highway_path", "highway_minor",
      "highway_major_casing", "highway_major_inner", "highway_major_subtle",
      "highway_motorway_casing", "highway_motorway_inner", "highway_motorway_subtle",
      "railway_transit", "railway_transit_dashline", "railway_minor",
      "railway_minor_dashline", "railway", "railway_dashline", "boundary_state",
    ]) {
      if (map.getLayer(id))
        map.setLayoutProperty(id, "visibility", scene ? "visible" : "none");
    }
    for (const id of ["probability", "mask", "uncertainty"])
      if (map.getLayer(id))
        map.setPaintProperty(id, "raster-opacity", overlayOpacity);
    if (map.getLayer("asset-focus"))
      map.setFilter("asset-focus", [
        "==",
        ["get", "asset_id"],
        selectedAsset ?? "",
      ]);
    if (map.getLayer("zone-focus"))
      map.setFilter("zone-focus", [
        "==",
        ["get", "candidate_id"],
        selectedZone ?? "",
      ]);
  };

  const updateScene = () => {
    const map = mapRef.current;
    if (!map || !map.getLayer("boundary")) return;
    const { scene, session, plan } = current.current;
    for (const id of ["s1", "probability", "mask", "uncertainty"] as const) {
      if (map.getLayer(id)) map.removeLayer(id);
      if (map.getSource(id)) map.removeSource(id);
      const available =
        id === "s1"
          ? (session?.layers.s1 ?? scene?.status === "observation_only")
          : !!session?.layers[id];
      const coordinates = session?.layer_rasters[id]?.corners ?? scene?.corners;
      if (!scene || !coordinates || !available) continue;
      const url = session
        ? `/api/layers/${id}.png?run_id=${encodeURIComponent(session.run_id)}`
        : `/api/observations/${encodeURIComponent(scene.chip_id)}/s1.png`;
      map.addSource(id, { type: "image", url, coordinates });
      map.addLayer(
        {
          id,
          type: "raster",
          source: id,
          paint: {
            "raster-fade-duration": 0,
            "raster-resampling": id === "mask" ? "nearest" : "linear",
          },
        },
        "boundary",
      );
    }
    const bounds = scene?.bounds;
    const outline: Feature[] = bounds
      ? [
          {
            type: "Feature",
            geometry: {
              type: "Polygon",
              coordinates: [
                [
                  [bounds[0], bounds[1]],
                  [bounds[2], bounds[1]],
                  [bounds[2], bounds[3]],
                  [bounds[0], bounds[3]],
                  [bounds[0], bounds[1]],
                ],
              ],
            },
            properties: {},
          },
        ]
      : [];
    (map.getSource("scene-extent") as GeoJSONSource | undefined)?.setData(
      features(outline),
    );
    (map.getSource("assets-data") as GeoJSONSource | undefined)?.setData(
      features(session?.assets.features ?? []),
    );
    (map.getSource("candidate-data") as GeoJSONSource | undefined)?.setData(
      features(session?.candidates.features ?? []),
    );
    const selected = new Set(plan?.plans.C ?? []);
    (map.getSource("plan-data") as GeoJSONSource | undefined)?.setData(
      features(
        session?.candidates.features.filter((item) =>
          selected.has(String(item.properties.candidate_id)),
        ) ?? [],
      ),
    );
    updateVisibility();
  };

  useEffect(() => {
    if (!host.current) return;
    const probe = document.createElement("canvas");
    if (!probe.getContext("webgl2") && !probe.getContext("webgl")) {
      setUnavailable("WebGL не поддерживается этим браузером");
      return;
    }
    let map: Map;
    let styleReady = false;
    let fallbackUsed = false;
    try {
      map = new maplibregl.Map({
        container: host.current,
        style: mapProviders.map.style,
        center: [39, 13],
        zoom: overviewZoom(),
        minZoom: minimumZoom(),
        attributionControl: false,
        cooperativeGestures: false,
        dragRotate: false,
        pitchWithRotate: false,
        touchPitch: false,
        maxPitch: 0,
      });
    } catch (cause) {
      setUnavailable(
        cause instanceof Error ? cause.message : "WebGL недоступен",
      );
      return;
    }
    mapRef.current = map;
    // Trackpad: two-finger swipe pans, pinch (wheel with ctrlKey) zooms.
    map.scrollZoom.disable();
    map.getCanvasContainer().addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        if (event.ctrlKey) {
          const rect = map.getCanvasContainer().getBoundingClientRect();
          const around = map.unproject([event.clientX - rect.left, event.clientY - rect.top]);
          const step = event.deltaY * (event.deltaMode === 1 ? 0.16 : 0.01);
          map.zoomTo(map.getZoom() - step, { around, duration: 0 });
        } else {
          // Line-mode deltas (Firefox mouse wheel) are ~3 per notch.
          const scale = event.deltaMode === 1 ? 16 : 1;
          map.panBy([event.deltaX * scale, event.deltaY * scale], { duration: 0 });
        }
      },
      { passive: false },
    );
    map.setPadding(props.scene ? noPadding() : overviewPadding());
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    const attribution = host.current.querySelector<HTMLDetailsElement>(
      ".maplibregl-ctrl-attrib",
    );
    // MapLibre opens compact attribution on first load. Start with its info
    // button collapsed, then leave user toggling entirely to MapLibre.
    const attributionObserver = new MutationObserver(() => {
      if (!attribution?.classList.contains("maplibregl-compact")) return;
      attributionObserver.disconnect();
      attribution.removeAttribute("open");
      attribution.classList.remove("maplibregl-compact-show");
    });
    if (attribution) {
      attributionObserver.observe(attribution, {
        attributes: true,
        attributeFilter: ["class"],
      });
      if (attribution.classList.contains("maplibregl-compact")) {
        attributionObserver.disconnect();
        attribution.removeAttribute("open");
        attribution.classList.remove("maplibregl-compact-show");
      }
    }
    map.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      "bottom-right",
    );
    const onStyle = () => {
      styleReady = true;
      map.setProjection({ type: "globe" });
      map.setSky({
        "sky-color": "#06080a",
        "horizon-color": "#0a0e10",
        "atmosphere-blend": ["interpolate", ["linear"], ["zoom"], 0, 0.25, 5, 0],
      });
      if (map.getLayer("background"))
        map.setPaintProperty("background", "background-color", "#1c2124");
      if (map.getLayer("water"))
        map.setPaintProperty("water", "fill-color", "#0a0f12");
      // Only polar ice is white (Arctic and Antarctic); mountain glaciers blend into land — the
      // satellite view shows them anyway. Tile glaciers mix both, so polar ice comes from its own layer.
      if (map.getLayer("landcover_glacier")) map.setPaintProperty("landcover_glacier", "fill-opacity", 0);
      if (map.getLayer("landcover_ice_shelf")) {
        map.setPaintProperty("landcover_ice_shelf", "fill-color", "#e2eaee");
        map.setPaintProperty("landcover_ice_shelf", "fill-opacity", 0.95);
      }
      const styleLayers = map.getStyle().layers;
      const glacierAt = styleLayers.findIndex((l) => l.id === "landcover_glacier");
      map.addSource("polar-ice", { type: "geojson", data: POLAR_ICE });
      map.addLayer(
        { id: "polar-ice", type: "fill", source: "polar-ice", paint: { "fill-color": "#eef3f5", "fill-opacity": 0.95 } },
        glacierAt >= 0 ? styleLayers[glacierAt + 1]?.id : undefined, // under labels and the satellite layer
      );
      for (const id of [
        "place_country_major",
        "place_country_minor",
        "place_city_large",
        "place_city",
      ]) {
        if (map.getLayer(id)) map.setPaintProperty(id, "text-color", "#a3aab4");
      }
      map.addSource("optical-data", {
        type: "raster",
        tiles: [...mapProviders.optical.tiles],
        tileSize: 256,
        maxzoom: mapProviders.optical.maxzoom,
        attribution: mapProviders.optical.attribution,
      });
      map.addLayer({
        id: "optical",
        type: "raster",
        source: "optical-data",
        layout: { visibility: "none" },
        paint: { "raster-opacity": 1, "raster-fade-duration": 0 },
      });
      // All scenes cluster together: with hundreds of analyzed scenes an unclustered layer
      // piles markers on top of each other. The cluster counts its analyzed scenes.
      map.addSource("observations", {
        type: "geojson",
        data: markers(current.current.scenes),
        cluster: true,
        clusterRadius: 38,
        clusterMaxZoom: 9,
        clusterProperties: {
          analyzed: ["+", ["case", ["==", ["get", "status"], "analyzed"], 1, 0]],
        },
      });
      map.addLayer({
        id: "cluster",
        type: "circle",
        source: "observations",
        filter: ["has", "point_count"],
        paint: {
          "circle-color": "#e3eaed",
          "circle-opacity": 0.92,
          "circle-radius": ["step", ["get", "point_count"], 14, 50, 17, 150, 20],
          "circle-stroke-color": [
            "case", [">", ["get", "analyzed"], 0], "#8bc2ff", "#080b0d",
          ],
          "circle-stroke-opacity": 0.9,
          "circle-stroke-width": 2,
        },
      });
      map.addLayer({
        id: "cluster-count",
        type: "symbol",
        source: "observations",
        filter: ["has", "point_count"],
        layout: {
          "text-field": ["get", "point_count_abbreviated"],
          "text-size": 12,
          "text-font": ["Noto Sans Regular"],
        },
        paint: { "text-color": "#11171a" },
      });
      map.addLayer({
        id: "observation",
        type: "circle",
        source: "observations",
        filter: [
          "all", ["!", ["has", "point_count"]], ["!=", ["get", "status"], "analyzed"],
        ],
        paint: {
          "circle-color": "#f1f5f6",
          "circle-opacity": 1,
          "circle-radius": 6,
          "circle-stroke-color": "#111719",
          "circle-stroke-width": 2,
        },
      });
      map.addLayer({
        id: "analyzed",
        type: "circle",
        source: "observations",
        filter: [
          "all", ["!", ["has", "point_count"]], ["==", ["get", "status"], "analyzed"],
        ],
        paint: {
          "circle-color": "#f3f6f7",
          "circle-opacity": 1,
          "circle-radius": 8,
          "circle-stroke-color": "#131a1d",
          "circle-stroke-width": 2.5,
        },
      });
      map.addSource("scene-extent", { type: "geojson", data: empty });
      map.addLayer({
        id: "boundary",
        type: "line",
        source: "scene-extent",
        paint: {
          "line-color": "#d3e2f3",
          "line-width": 1.6,
          "line-dasharray": [3, 2],
        },
      });
      map.addSource("candidate-data", { type: "geojson", data: empty });
      map.addLayer({
        id: "candidate-hit",
        type: "fill",
        source: "candidate-data",
        paint: { "fill-color": "#4D9FFF", "fill-opacity": 0.01 },
      });
      map.addLayer({
        id: "candidates",
        type: "line",
        source: "candidate-data",
        paint: {
          "line-color": "#d9e3ed",
          "line-opacity": 0.7,
          "line-width": 1.4,
        },
      });
      map.addSource("plan-data", { type: "geojson", data: empty });
      map.addLayer({
        id: "plan-fill",
        type: "fill",
        source: "plan-data",
        paint: { "fill-color": "#4D9FFF", "fill-opacity": 0.19 },
      });
      map.addLayer({
        id: "plan-line",
        type: "line",
        source: "plan-data",
        paint: { "line-color": "#8bc2ff", "line-width": 2.4 },
      });
      map.addLayer({
        id: "zone-focus",
        type: "line",
        source: "candidate-data",
        filter: ["==", ["get", "candidate_id"], ""],
        paint: { "line-color": "#fff", "line-width": 3.5 },
      });
      map.addSource("assets-data", { type: "geojson", data: empty });
      map.addLayer({
        id: "assets",
        type: "circle",
        source: "assets-data",
        paint: {
          "circle-radius": 6,
          "circle-color": "#f5f5f7",
          "circle-stroke-color": "#142439",
          "circle-stroke-width": 2,
        },
      });
      map.addLayer({
        id: "asset-focus",
        type: "circle",
        source: "assets-data",
        filter: ["==", ["get", "asset_id"], ""],
        paint: {
          "circle-radius": 12,
          "circle-color": "#4D9FFF",
          "circle-opacity": 0.25,
          "circle-stroke-color": "#4D9FFF",
          "circle-stroke-width": 2,
        },
      });
      const media = current.current.messages;
      map.addSource("media-events", {
        type: "geojson",
        data: mediaFeatures(media?.events.features ?? []),
      });
      map.addSource("media-reports", {
        type: "geojson",
        data: mediaFeatures(media?.reports.features ?? []),
      });
      // Calm markers on top of rasters and zones: an outline for areas, no fill, no pulsing.
      map.addLayer({
        id: "media-area",
        type: "line",
        source: "media-events",
        filter: ["!=", ["geometry-type"], "Point"],
        paint: { "line-color": "#f0b85a", "line-width": 1.4, "line-opacity": 0.75, "line-dasharray": [2, 2] },
      });
      map.addLayer({
        id: "media-event",
        type: "circle",
        source: "media-events",
        filter: ["==", ["geometry-type"], "Point"],
        paint: {
          "circle-radius": 6,
          "circle-color": "#f0b85a",
          "circle-opacity": ["case", ["==", ["get", "review_state"], "reviewed"], 0.95, 0.7],
          "circle-stroke-color": "#1a1408",
          "circle-stroke-width": 1.5,
        },
      });
      map.addLayer({
        id: "media-report",
        type: "circle",
        source: "media-reports",
        paint: {
          "circle-radius": 6,
          "circle-color": "#f0b85a",
          "circle-opacity": 0.12,
          "circle-stroke-color": "#f0b85a",
          "circle-stroke-width": 2,
        },
      });
      updateScene();
      setReady((value) => value + 1);
      current.current.onIssue(
        fallbackUsed
          ? "Внешняя карта недоступна. Показаны локальные данные на нейтральном фоне."
          : null,
      );
    };
    map.on("style.load", onStyle);
    map.on("error", () =>
      current.current.onIssue(
        map.isStyleLoaded()
          ? "Часть картографических данных недоступна. Локальные данные и анализ сохранены."
          : "Не удалось загрузить географическую подложку. Локальные данные и анализ доступны.",
      ),
    );
    const timeout = window.setTimeout(() => {
      if (!styleReady) {
        fallbackUsed = true;
        map.setStyle({
          version: 8,
          sources: {},
          layers: [
            {
              id: "fallback",
              type: "background",
              paint: { "background-color": "#111318" },
            },
          ],
        });
        current.current.onIssue(
          "Внешняя карта недоступна. Показаны локальные данные на нейтральном фоне.",
        );
      }
    }, 8000);
    const onMessage = (point: maplibregl.PointLike) =>
      map.queryRenderedFeatures(point, {
        layers: ["media-event", "media-report", "media-area"].filter((id) => map.getLayer(id)),
      }).length > 0;
    const selectMarker = (event: {
      point: maplibregl.Point;
      features?: Array<{ properties?: Record<string, unknown> }>;
    }) => {
      if (onMessage(event.point)) return;
      const id = event.features?.[0]?.properties?.chip_id;
      const scene = current.current.scenes.find((item) => item.chip_id === id);
      if (scene) current.current.onSelectScene(scene);
    };
    map.on("click", "observation", selectMarker);
    map.on("click", "analyzed", selectMarker);
    map.on("click", "cluster", (event) => {
      if (
        onMessage(event.point) ||
        map.queryRenderedFeatures(event.point, { layers: ["analyzed"] }).length
      )
        return;
      const feature = event.features?.[0];
      const source = map.getSource("observations") as GeoJSONSource | undefined;
      if (source && feature?.geometry.type === "Point") {
        const center = feature.geometry.coordinates as [number, number];
        source
          .getClusterExpansionZoom(Number(feature.properties?.cluster_id))
          .then((zoom) =>
            map.easeTo({
              center,
              zoom,
              duration: reducedMotion() ? 0 : 400,
            }),
          );
      }
    });
    const showPreview = (event: {
      features?: Array<{ properties?: Record<string, unknown> }>;
    }) => {
      map.getCanvas().style.cursor = "pointer";
      const scene = current.current.scenes.find(
        (item) => item.chip_id === event.features?.[0]?.properties?.chip_id,
      );
      if (!scene?.position) return;
      popup.current?.remove();
      const node = document.createElement("div");
      node.className = "map-preview";
      const title = document.createElement("strong");
      title.textContent = scene.territory.replaceAll("-", " ");
      const detail = document.createElement("span");
      detail.textContent = scene.observation_date
        ? new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(scene.observation_date))
        : "Дата неизвестна";
      node.append(title, detail);
      popup.current = new maplibregl.Popup({
        closeButton: false,
        closeOnClick: false,
        offset: 18,
      })
        .setLngLat([scene.position.lon, scene.position.lat])
        .setDOMContent(node)
        .addTo(map);
    };
    map.on("mouseenter", "observation", showPreview);
    map.on("mouseenter", "analyzed", showPreview);
    map.on("mouseleave", "observation", () => {
      map.getCanvas().style.cursor = "";
      popup.current?.remove();
    });
    map.on("mouseleave", "analyzed", () => {
      map.getCanvas().style.cursor = "";
      popup.current?.remove();
    });
    const previewMessage = (kind: "event" | "report") => (event: maplibregl.MapLayerMouseEvent) => {
      const feature = event.features?.[0];
      if (!feature) return;
      map.getCanvas().style.cursor = "pointer";
      popup.current?.remove();
      popup.current = new maplibregl.Popup({ closeButton: false, closeOnClick: true, offset: 12, maxWidth: "280px" })
        .setLngLat(event.lngLat)
        .setDOMContent(messageCard(feature.properties ?? {}, kind))
        .addTo(map);
    };
    const openMessage = (kind: "event" | "report") => (event: maplibregl.MapLayerMouseEvent) => {
      const feature = event.features?.[0];
      const props = (feature?.properties ?? {}) as Record<string, unknown>;
      const id = kind === "event" ? props.event_id : props.report_id;
      if (id === undefined || id === null) return;
      popup.current?.remove();
      const lngLat: [number, number] = feature?.geometry.type === "Point"
        ? [feature.geometry.coordinates[0], feature.geometry.coordinates[1]]
        : [event.lngLat.lng, event.lngLat.lat];
      current.current.onSelectMessage?.({ kind, id: String(id), lngLat, props });
    };
    map.on("click", "media-event", openMessage("event"));
    map.on("click", "media-area", openMessage("event"));
    map.on("click", "media-report", openMessage("report"));
    map.on("mouseenter", "media-event", previewMessage("event"));
    map.on("mouseenter", "media-report", previewMessage("report"));
    map.on("mouseenter", "media-area", () => (map.getCanvas().style.cursor = "pointer"));
    for (const id of ["media-event", "media-report", "media-area"]) {
      map.on("mouseleave", id, () => {
        map.getCanvas().style.cursor = "";
        popup.current?.remove();
      });
    }
    map.on("click", "assets", (event) => {
      if (onMessage(event.point)) return;
      const id = event.features?.[0]?.properties?.asset_id;
      if (typeof id === "string") current.current.onSelectAsset(id);
    });
    map.on("click", "candidate-hit", (event) => {
      if (onMessage(event.point) ||
          map.queryRenderedFeatures(event.point, { layers: ["assets"] }).length)
        return;
      const id = event.features?.[0]?.properties?.candidate_id;
      if (typeof id === "string") current.current.onSelectZone(id);
    });
    const observer = new ResizeObserver(() => {
      map.setMinZoom(minimumZoom());
      map.resize();
      map.setPadding(current.current.scene ? noPadding() : overviewPadding());
    });
    observer.observe(host.current);
    return () => {
      window.clearTimeout(timeout);
      attributionObserver.disconnect();
      observer.disconnect();
      popup.current?.remove();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    (
      mapRef.current?.getSource("observations") as GeoJSONSource | undefined
    )?.setData(markers(props.scenes));
  }, [props.scenes, ready]);
  // «Приблизить к месту» from the message panel.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !props.focus) return;
    map.easeTo({ center: props.focus.lngLat, zoom: props.focus.zoom, duration: reducedMotion() ? 0 : 900 });
  }, [props.focus?.request]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !map.getLayer("boundary")) return;
    updateScene();
    // A hidden map (the analysis page) has no size: fit when it is shown, not before.
    if (!props.active || !props.scene?.bounds || lastScene.current === props.scene.chip_id)
      return;
    lastScene.current = props.scene.chip_id;
    popup.current?.remove();
    map.stop();
    map.resize();
    map.setPadding(noPadding());
    const [west, south, east, north] = props.scene.bounds;
    map.fitBounds(
      [
        [west, south],
        [east, north],
      ],
      {
        padding:
          window.innerWidth < 700
            ? { top: 70, right: 22, bottom: 255, left: 22 }
            : { top: 80, right: 420, bottom: 90, left: 340 },
        maxZoom: 12.5,
        duration: reducedMotion() ? 0 : 850,
      },
    );
  }, [props.scene?.chip_id, props.session?.run_id, props.active, ready]);
  useEffect(() => {
    const map = mapRef.current;
    (map?.getSource("media-events") as GeoJSONSource | undefined)?.setData(
      mediaFeatures(props.messages?.events.features ?? []),
    );
    (map?.getSource("media-reports") as GeoJSONSource | undefined)?.setData(
      mediaFeatures(props.messages?.reports.features ?? []),
    );
  }, [props.messages, ready]);
  useEffect(() => {
    updateVisibility();
  }, [
    props.messages,
    props.showMessages,
    props.baseMap,
    props.visibleLayers,
    props.overlayOpacity,
    props.selectedAsset,
    props.selectedZone,
    props.scene?.chip_id,
    ready,
  ]);
  useEffect(() => {
    const selected = new Set(props.plan?.plans.C ?? []);
    (
      mapRef.current?.getSource("plan-data") as GeoJSONSource | undefined
    )?.setData(
      features(
        props.session?.candidates.features.filter((item) =>
          selected.has(String(item.properties.candidate_id)),
        ) ?? [],
      ),
    );
  }, [props.plan, props.session, ready]);
  useEffect(() => {
    const map = mapRef.current;
    const feature = props.session?.assets.features.find(
      (item) => item.properties.asset_id === props.selectedAsset,
    );
    if (map && feature?.geometry.type === "Point")
      map.easeTo({
        center: feature.geometry.coordinates as [number, number],
        duration: reducedMotion() ? 0 : 350,
      });
  }, [props.selectedAsset]);
  useEffect(() => {
    if (props.active) mapRef.current?.resize();
  }, [props.active]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !props.active || props.scene) return;
    // Leaving a scene without an overview request keeps the camera where it is.
    const requested = props.overviewRequest !== lastOverview.current;
    lastOverview.current = props.overviewRequest;
    lastScene.current = null;
    if (!requested) return;
    map.stop();
    map.setPadding(overviewPadding());
    map.easeTo({
      center: [39, 13],
      zoom: overviewZoom(),
      duration: reducedMotion() ? 0 : 600,
    });
  }, [props.overviewRequest, props.scene?.chip_id, props.active, ready]);
  return (
    <div className="map-host">
      <div
        ref={host}
        className="map-canvas"
        aria-label="Интерактивная Земля и наблюдения"
      />
      {unavailable && (
        <div className="map-unavailable" role="alert">
          Карта недоступна: {unavailable}. Выберите сцену в списке и откройте
          анализ.
        </div>
      )}
    </div>
  );
}
