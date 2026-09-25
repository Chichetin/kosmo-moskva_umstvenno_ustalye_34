export type Num = number | null;

export interface Meta {
  run_id: string;
  demo?: boolean;
  demo_note?: string;
  chip_id: string;
  event_id: string;
  s1_date?: string;
  threshold: number;
  budget_rub: number;
  decision: { decision_at: string; decision_deadline: string; user?: string };
  costs?: { other_cost_rub: number; other_cost_basis?: string };
  uncertainty?: { asset_field?: string; raster?: string };
  strategy_rules?: Record<string, string>;
  pricing?: { legal_edition?: string; base_rate_status?: string };
  el_model?: string;
  portfolio?: { seed: number; rule: string };
  methods?: Record<string, Record<string, unknown>>;
  proxy_check?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface LossRow {
  asset_id: string;
  chip_id: string;
  p_flood: Num;
  expected_loss_rub: Num;
  uncertainty: Num;
  rank: Num;
  status: "ok" | "partial" | "no_data";
}

export interface PlanRow {
  strategy: "B" | "C";
  candidate_id: string;
  area_km2: number;
  cost_rub: number;
  unit_price_rub_km2: number;
  discount_coef: number;
  processing_coef: number;
  usage_coef: number;
  freshness_coef: number;
  discount_group_id: string;
  group_area_km2: number;
  [k: string]: unknown;
}

export interface CompRow {
  strategy: "A" | "B" | "C";
  data_cost_rub: number;
  other_cost_rub: number;
  decision_cost_rub: number;
  budget_rub: number;
  budget_feasible: boolean;
  covered_expected_loss_rub: number;
  coverage_share: Num;
  residual_uncertainty: Num;
  uncertainty_status: string;
  uncertainty_basis: string;
}

export interface SensRow {
  scenario_id: string;
  strategy: string;
  changed_inputs_json: string;
  decision_cost_rub: Num;
  covered_expected_loss_rub: Num;
  result_status: string;
  interpretation: string;
}

export interface Feature<P> {
  type: "Feature";
  geometry: { type: string; coordinates: any };
  properties: P;
}
export interface FC<P> {
  type: "FeatureCollection";
  features: Feature<P>[];
}

export interface AssetProps {
  asset_id: string;
  asset_class: string;
  asset_value_rub: number;
  vulnerability_coef: number;
}

export interface ZoneProps {
  candidate_id: string;
  sensor_type: string;
  resolution_m: number;
  acquisition_type: string;
  data_role: string;
  observation_at: string | null;
  available_at: string | null;
  processing_level: string;
  usage_type: string;
  guaranteed_purchase: boolean;
  area_km2: number;
  base_rate_status: string;
}

export interface Plan {
  engine?: string;
  plans: { A: string[]; B: string[]; C: string[] };
  plan_rows: PlanRow[];
  comparison: CompRow[];
  reasons?: Record<string, string>;
  budget_rub?: number;
}

export interface Kit extends Plan {
  meta: Meta;
  bounds: [number, number, number, number];
  assets: FC<AssetProps>;
  zones: FC<ZoneProps>;
  loss: LossRow[];
  sensitivity: SensRow[];
  sources: Record<string, string>[];
  files: string[];
  layers: string[];
  replan_engine: string;
}

export interface RunInfo {
  run_id: string;
  chip_id: string;
  event_id: string;
  s1_date: string;
  demo: boolean;
  budget_rub: number;
}
