import type { FeatureCollection, Point, Polygon } from "geojson";
import { label, when } from "./api";
import type { MediaEvent } from "./types";

const PIN_PRECISION = new Set(["point", "street", "locality"]);
const family = (event: MediaEvent) =>
  event.report_type === "forecast" || event.report_type === "warning"
    ? "alert"
    : event.report_type === "negation"
      ? "negation"
      : "situation";

/** Pins only where the place is known to a settlement or better (spec §Е). */
export const eventPoints = (events: MediaEvent[]): FeatureCollection<Point> => ({
  type: "FeatureCollection",
  features: events.flatMap((event) =>
    event.lat != null && event.lon != null && PIN_PRECISION.has(event.geo_precision)
      ? [
          {
            type: "Feature" as const,
            geometry: { type: "Point" as const, coordinates: [event.lon, event.lat] },
            properties: { event_id: event.event_id, family: family(event) },
          },
        ]
      : [],
  ),
});

const circle = (lon: number, lat: number, radiusKm: number): number[][] => {
  const ring: number[][] = [];
  const dLat = radiusKm / 110.574;
  const dLon = radiusKm / (111.32 * Math.max(Math.cos((lat * Math.PI) / 180), 0.05));
  for (let step = 0; step <= 64; step++) {
    const angle = (step / 64) * 2 * Math.PI;
    ring.push([lon + dLon * Math.cos(angle), lat + dLat * Math.sin(angle)]);
  }
  return ring;
};

/** District/region/country: the known extent (or an approximate disc), never a pin. */
export const eventAreas = (events: MediaEvent[]): FeatureCollection<Polygon> => ({
  type: "FeatureCollection",
  features: events.flatMap((event) => {
    if (event.lat == null || event.lon == null || PIN_PRECISION.has(event.geo_precision))
      return [];
    let ring: number[][] | null = null;
    if (event.bbox) {
      const [west, south, east, north] = event.bbox;
      ring = [[west, south], [east, south], [east, north], [west, north], [west, south]];
    } else if (event.area_radius_km) {
      ring = circle(event.lon, event.lat, event.area_radius_km);
    }
    return ring
      ? [
          {
            type: "Feature" as const,
            geometry: { type: "Polygon" as const, coordinates: [ring] },
            properties: { event_id: event.event_id, family: family(event) },
          },
        ]
      : [];
  }),
});

/** Hover card built with textContent: source text never becomes HTML. */
export const eventPopup = (event: MediaEvent) => {
  const node = document.createElement("div");
  node.className = "map-preview media-preview";
  const kind = document.createElement("small");
  const origins = (["media", "bulletin", "citizen"] as const)
    .filter((origin) => event.evidence_counts[origin])
    .map((origin) => label("origin", origin))
    .join(", ");
  kind.textContent = `Сообщение · ${origins || "без источников"}${event.is_demo ? " · ДЕМО" : ""}`;
  const title = document.createElement("strong");
  title.textContent = event.title;
  const place = document.createElement("span");
  place.textContent = `${event.place_label ?? "место без названия"} · точность: ${label("precision", event.geo_precision)}`;
  const time = document.createElement("span");
  time.textContent = `${when(event.event_time_start ?? event.first_detected_at)} · ${label("reviewState", event.review_state)}`;
  node.append(kind, title, place, time);
  return node;
};
