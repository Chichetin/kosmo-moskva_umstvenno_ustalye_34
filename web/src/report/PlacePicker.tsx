import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { Map, Marker, StyleSpecification } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import { mapProviders } from "../mapProviders";

maplibregl.setWorkerUrl(workerUrl);

export type Point = { lat: number; lon: number };
type Props = {
  point: Point | null;
  focus: { point: Point; zoom: number } | null;
  onPick: (point: Point) => void;
};

// Plain OSM raster tiles if the vector style cannot be reached from this network.
const fallbackStyle: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: 19,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a>',
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};
const reducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export default function PlacePicker({ point, focus, onPick }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const marker = useRef<Marker | null>(null);
  const pick = useRef(onPick);
  pick.current = onPick;
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    if (!host.current) return;
    const probe = document.createElement("canvas");
    if (!probe.getContext("webgl2") && !probe.getContext("webgl")) {
      setProblem("Карта не поддерживается этим браузером. Найдите место поиском или введите координаты.");
      return;
    }
    let loaded = false;
    let fallback = false;
    const map = new maplibregl.Map({
      container: host.current,
      style: mapProviders.map.style,
      center: [60, 55],
      zoom: 2.2,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false,
      touchPitch: false,
      maxPitch: 0,
      cooperativeGestures: true,
      locale: {
        "CooperativeGesturesHandler.WindowsHelpText": "Ctrl + колесо — масштаб карты",
        "CooperativeGesturesHandler.MacHelpText": "⌘ + колесо — масштаб карты",
        "CooperativeGesturesHandler.MobileHelpText": "Двигайте карту двумя пальцами",
      },
    });
    mapRef.current = map;
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-left");
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      loaded = true;
    });
    map.on("error", () => {
      if (!loaded && !fallback) {
        fallback = true;
        map.setStyle(fallbackStyle);
      } else if (fallback) {
        setProblem(
          "Подложка карты не загружается. Точку можно поставить и так, но надёжнее найти место поиском или ввести координаты.",
        );
      }
    });
    map.on("click", (event) =>
      pick.current({
        lat: Number(event.lngLat.lat.toFixed(5)),
        lon: Number(event.lngLat.lng.toFixed(5)),
      }),
    );
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (!point) {
      marker.current?.remove();
      marker.current = null;
      return;
    }
    if (!marker.current) {
      const node = document.createElement("div");
      node.className = "report-pin";
      marker.current = new maplibregl.Marker({ element: node });
    }
    marker.current.setLngLat([point.lon, point.lat]).addTo(map);
  }, [point]);

  useEffect(() => {
    if (!focus || !mapRef.current) return;
    mapRef.current.easeTo({
      center: [focus.point.lon, focus.point.lat],
      zoom: focus.zoom,
      duration: reducedMotion() ? 0 : 500,
    });
  }, [focus]);

  return (
    <div className="picker">
      <div
        ref={host}
        className="picker-map"
        aria-label="Карта: нажмите, чтобы отметить место события"
      />
      {problem && <p className="picker-problem">{problem}</p>}
    </div>
  );
}
