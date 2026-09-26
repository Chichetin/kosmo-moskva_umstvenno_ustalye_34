import type { Feature } from "./types";

export type RiskAsset = { id: string; probability: number; value: number; vulnerability: number };

export function finiteNumber(value: unknown): number | null {
  if ((typeof value !== "number" && typeof value !== "string") || (typeof value === "string" && !value.trim())) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

// Unknown or invalid estimates remain unknown everywhere: chart and totals.
export function riskAsset(feature: Feature): RiskAsset | null {
  const p = feature.properties;
  const probability = finiteNumber(p.p_flood);
  const value = finiteNumber(p.asset_value_rub);
  const vulnerability = finiteNumber(p.vulnerability_coef);
  if (p.status !== "ok" || probability == null || probability < 0 || probability > 1 ||
      value == null || value < 0 || vulnerability == null || vulnerability < 0 || vulnerability > 1) return null;
  return { id: String(p.asset_id), probability, value, vulnerability };
}
