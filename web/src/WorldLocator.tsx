import "./worldLocator.css";
import { WORLD_LAND_PATH } from "./worldLand";

// Flat world map with one point: where the current scene is. Orientation only, no interaction.
export function WorldLocator({
  lon,
  lat,
  label,
}: {
  lon: number;
  lat: number;
  label?: string;
}) {
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  const x = Math.min(360, Math.max(0, lon + 180));
  const y = Math.min(180, Math.max(0, 90 - lat));
  const coords = `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "с. ш." : "ю. ш."}, ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "в. д." : "з. д."}`;
  return (
    <figure className="world-locator" aria-label={`Место на карте мира: ${label ?? coords}`}>
      <svg viewBox="0 18 360 144" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <path className="world-locator-land" d={WORLD_LAND_PATH} />
        <line className="world-locator-cross" x1={x} y1="18" x2={x} y2="162" />
        <line className="world-locator-cross" x1="0" y1={y} x2="360" y2={y} />
        <circle className="world-locator-pulse" cx={x} cy={y} r="4" />
        <circle className="world-locator-dot" cx={x} cy={y} r="3" />
      </svg>
      <figcaption>
        {label && <strong>{label}</strong>}
        <span>{coords}</span>
      </figcaption>
    </figure>
  );
}
