import { useState } from "react";
import { pointInZone } from "./geometry";
import { formatNumber, formatProbability, formatRub } from "./format";
import type { Feature, Plan, Session } from "./types";
import "./planZones.css";

// Plan C of the current budget drawn on the observation itself: the whole scene with the chosen
// zones, then one crop per zone with the objects it covers. Only zones of plan C are drawn —
// the other candidates are in «Все зоны» of the list. The picture is the existing S1 scene
// inside the zone; the new acquisition is not ordered yet.

type Base = "s1" | "probability" | "mask";
type Box = { x: number; y: number; side: number };

const BASES: Array<[Base, string]> = [
  ["s1", "Снимок S1"],
  ["probability", "S1 + вероятность воды"],
  ["mask", "S1 + маска"],
];

const rings = (zone: Feature): number[][][] =>
  zone.geometry.type === "Polygon"
    ? (zone.geometry.coordinates as number[][][])
    : zone.geometry.type === "MultiPolygon"
      ? (zone.geometry.coordinates as unknown as number[][][][]).flat()
      : [];

type Props = {
  session: Session;
  plan: Plan | null;
  budget: number | null;
  assets: Feature[];
  selectedZone: string | null;
  onSelectZone: (id: string) => void;
  assetName: (feature: Feature) => string;
  observationDate: string;
};

export default function PlanZones(props: Props) {
  const { session, plan } = props;
  const [base, setBase] = useState<Base>(session.layers.s1 ? "s1" : "probability");
  const frame = session.layer_rasters.s1 ?? session.layer_rasters.probability;
  const bounds = frame?.bounds ?? session.raster.bounds;
  if (!bounds) return null;
  const width = frame?.width || session.raster.width || 512;
  const height = frame?.height || session.raster.height || 512;
  const [west, south, east, north] = bounds;
  const px = ([lon, lat]: number[]) => [
    ((lon - west) / (east - west)) * width,
    ((north - lat) / (north - south)) * height,
  ];
  const layerUrl = (id: Base) =>
    `/api/layers/${id}.png?run_id=${encodeURIComponent(session.run_id)}`;
  const shown: Base = base === "s1" && !session.layers.s1 ? "probability" : base;
  const overlay = shown !== "s1" && session.layers[shown] ? shown : null;

  const chosen = (plan?.plans.C ?? [])
    .map((id) => session.candidates.features.find((zone) => String(zone.properties.candidate_id) === id))
    .filter((zone): zone is Feature => !!zone);
  const rows = new Map((plan?.plan_rows ?? [])
    .filter((row) => row.strategy === "C")
    .map((row) => [String(row.candidate_id), row]));
  const covered = new Set(chosen.flatMap((zone) =>
    props.assets.filter((asset) => pointInZone(asset, zone)).map((asset) => String(asset.properties.asset_id))));

  const zonePath = (zone: Feature) =>
    rings(zone)
      .map((ring) => `M${ring.map((point) => px(point).map((v) => v.toFixed(2)).join(",")).join("L")}Z`)
      .join("");
  const cropOf = (zone: Feature): Box => {
    const points = rings(zone).flat().map(px);
    const xs = points.map(([x]) => x);
    const ys = points.map(([, y]) => y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const side = Math.min(Math.max(width, height), Math.max(x1 - x0, y1 - y0) * 1.5 + 16);
    const clamp = (value: number, size: number) => Math.min(Math.max(value, 0), Math.max(0, size - side));
    return { x: clamp((x0 + x1 - side) / 2, width), y: clamp((y0 + y1 - side) / 2, height), side };
  };

  const picture = (box: Box, zones: Feature[], label: (zone: Feature) => string, detail: boolean) => {
    const unit = box.side / 100; // sizes in % of the visible side: same look in the scene and in a crop
    const visible = props.assets.filter((asset) => {
      if (asset.geometry.type !== "Point") return false;
      const [x, y] = px(asset.geometry.coordinates as number[]);
      return x >= box.x && x <= box.x + box.side && y >= box.y && y <= box.y + box.side;
    });
    return (
      <svg
        viewBox={`${box.x} ${box.y} ${box.side} ${box.side}`}
        role="img"
        aria-label={detail ? `Зона ${label(zones[0])} на снимке` : "Сцена с зонами плана C"}
      >
        <rect x={0} y={0} width={width} height={height} className="pz-ground" />
        {session.layers.s1 && <image href={layerUrl("s1")} x={0} y={0} width={width} height={height} preserveAspectRatio="none" />}
        {overlay && <image href={layerUrl(overlay)} x={0} y={0} width={width} height={height} preserveAspectRatio="none" opacity={0.78} />}
        {zones.map((zone) => {
          const id = String(zone.properties.candidate_id);
          const corner = rings(zone).flat().map(px);
          const lx = Math.min(...corner.map(([x]) => x));
          const ly = Math.min(...corner.map(([, y]) => y));
          return (
            <g key={id} className={`pz-zone ${props.selectedZone === id ? "active" : ""}`}>
              <path d={zonePath(zone)} vectorEffect="non-scaling-stroke" />
              {!detail && (
                <text x={lx + unit} y={ly + unit * 4.2} fontSize={unit * 3.6}>{label(zone)}</text>
              )}
            </g>
          );
        })}
        {visible.map((asset) => {
          const id = String(asset.properties.asset_id);
          const [x, y] = px(asset.geometry.coordinates as number[]);
          return (
            <g key={id} className={`pz-asset ${covered.has(id) ? "covered" : ""}`}>
              <circle cx={x} cy={y} r={unit * (detail ? 2.2 : 1.3)} vectorEffect="non-scaling-stroke" />
              <text x={x + unit * (detail ? 3.2 : 2)} y={y + unit * (detail ? 1.6 : 1)} fontSize={unit * (detail ? 5.2 : 2.9)}>{id}</text>
            </g>
          );
        })}
      </svg>
    );
  };

  const order = (zone: Feature) => `${chosen.indexOf(zone) + 1}`;
  return (
    <section className="analysis-card plan-zones" aria-label="Зоны плана C на снимке">
      <div className="plan-zones-head">
        <div>
          <h2>Зоны плана C на снимке</h2>
          <p>
            {plan
              ? chosen.length
                ? `${chosen.length} ${chosen.length === 1 ? "зона" : chosen.length < 5 ? "зоны" : "зон"} при бюджете ${formatRub(props.budget)} · номер — очерёдность выбора`
                : `При бюджете ${formatRub(props.budget)} в план C не попала ни одна зона`
              : "Пересчитываем план…"}
          </p>
        </div>
        <div className="zone-tabs" role="group" aria-label="Подложка снимков зон">
          {BASES.filter(([id]) => id === "s1" ? session.layers.s1 : session.layers[id]).map(([id, text]) => (
            <button key={id} className={shown === id ? "active" : ""} aria-pressed={shown === id} onClick={() => setBase(id)}>
              {text}
            </button>
          ))}
        </div>
      </div>
      <div className="plan-zones-body">
        <figure className="plan-zones-scene">
          {picture({ x: 0, y: 0, side: Math.max(width, height) }, chosen, order, false)}
          <figcaption>
            <span><i className="pz-key zone" /> зона плана C</span>
            <span><i className="pz-key covered" /> объект в зоне</span>
            <span><i className="pz-key asset" /> объект вне плана</span>
          </figcaption>
        </figure>
        {chosen.length > 0 && (
          <div className="plan-zones-grid">
            {chosen.map((zone, index) => {
              const id = String(zone.properties.candidate_id);
              const inside = props.assets
                .filter((asset) => pointInZone(asset, zone))
                .sort((a, b) => Number(b.properties.expected_loss_rub || 0) - Number(a.properties.expected_loss_rub || 0));
              const loss = inside.reduce((sum, asset) => sum + Number(asset.properties.expected_loss_rub || 0), 0);
              const earlier = new Set(chosen.slice(0, index).flatMap((previous) =>
                props.assets.filter((asset) => pointInZone(asset, previous)).map((asset) => String(asset.properties.asset_id))));
              const added = inside.filter((asset) => !earlier.has(String(asset.properties.asset_id)));
              const addedLoss = added.reduce((sum, asset) => sum + Number(asset.properties.expected_loss_rub || 0), 0);
              return (
                <article key={id} className={`plan-zone ${props.selectedZone === id ? "active" : ""}`}>
                  <button className="plan-zone-picture" onClick={() => props.onSelectZone(id)} aria-label={`Выбрать зону ${id}`}>
                    {picture(cropOf(zone), [zone], () => id, true)}
                    <b>{order(zone)}</b>
                  </button>
                  <div className="plan-zone-meta">
                    <div className="plan-zone-title">
                      <strong>{id}</strong>
                      <strong>{formatRub(rows.get(id)?.cost_rub)}</strong>
                    </div>
                    <small>
                      {formatNumber(zone.properties.area_km2, 2)} км² · {String(zone.properties.sensor_type ?? "").toUpperCase()}
                      {zone.properties.resolution_m != null && ` ${formatNumber(zone.properties.resolution_m, 1)} м`}
                    </small>
                    {inside.length ? (
                      <ul className="plan-zone-assets">
                        {inside.map((asset) => (
                          <li key={String(asset.properties.asset_id)}>
                            <strong>{props.assetName(asset)} <em>{String(asset.properties.asset_id)}</em></strong>
                            <div className="plan-zone-asset-facts">
                              <span><small>Вероятность воды</small><b>{formatProbability(asset.properties.p_flood)}</b></span>
                              <span><small>Ожидаемый ущерб</small><b>{formatRub(asset.properties.expected_loss_rub)}</b></span>
                            </div>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <small>Объектов портфеля в зоне нет</small>
                    )}
                    {inside.length > 1 && (
                      <div className="plan-zone-sum"><span>Ожидаемый ущерб в зоне</span><strong>{formatRub(loss)}</strong></div>
                    )}
                    {added.length > 0 && (
                      <details className="plan-zone-details">
                        <summary>Почему в плане</summary>
                        <div className="plan-zone-reason">
                          <p>На шаге {index + 1} зона добавляет к плану {added.map((asset) => String(asset.properties.asset_id)).join(", ")}.</p>
                          <p>Ожидаемый ущерб этих объектов — {formatRub(addedLoss)}. Зона укладывается в бюджет и доступна к сроку решения.</p>
                        </div>
                      </details>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </div>
      <p className="plan-zones-note">
        На снимках — исходное наблюдение S1 от {props.observationDate} в границах зон; новые снимки
        ещё не заказаны: это участки, которые будут досняты по плану C. Подложка не входит в расчёт p,
        ущерба и плана.
      </p>
    </section>
  );
}
