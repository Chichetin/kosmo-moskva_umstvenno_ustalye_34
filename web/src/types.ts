export type Feature = {
  type: "Feature";
  geometry: { type: string; coordinates: number[] | number[][][] };
  properties: Record<string, string | number | boolean | null>;
};

export type FeatureCollection = {
  type: "FeatureCollection";
  features: Feature[];
};

export type Session = {
  run_id: string;
  fixture: boolean;
  observation_date: string | null;
  metadata: {
    chip_id: string;
    event_date: string;
    threshold: number;
    budget: {
      budget_rub: number;
      decision_at: string;
      decision_deadline: string;
    };
    pricing?: { base_rate_status?: string };
    uncertainty?: { meaning?: string };
    [key: string]: unknown;
  };
  raster: {
    bounds: [number, number, number, number] | null;
    corners:
      | [[number, number], [number, number], [number, number], [number, number]]
      | null;
    width: number;
    height: number;
    crs: string | null;
  };
  layer_rasters: Record<
    "s1" | "probability" | "mask",
    {
      bounds: [number, number, number, number] | null;
      corners:
        | [
            [number, number],
            [number, number],
            [number, number],
            [number, number],
          ]
        | null;
      width: number;
      height: number;
      crs: string | null;
    } | null
  >;
  layers: {
    s1: boolean;
    probability: boolean;
    mask: boolean;
    uncertainty: boolean;
  };
  assets: FeatureCollection;
  candidates: FeatureCollection;
  sensitivity: Array<Record<string, string>>;
};

export type SceneSummary = {
  run_id: string | null;
  fixture: boolean;
  chip_id: string;
  territory: string;
  observation_date: string | null;
  position: { lat: number; lon: number } | null;
  bounds: [number, number, number, number] | null;
  corners:
    | [[number, number], [number, number], [number, number], [number, number]]
    | null;
  status: "analyzed" | "observation_only";
  preview_url: string | null;
};

export type Comparison = {
  strategy: "A" | "B" | "C";
  data_cost_rub: number;
  other_cost_rub: number;
  decision_cost_rub: number;
  budget_rub: number;
  budget_feasible: boolean;
  covered_expected_loss_rub: number;
  coverage_share: number | null;
  residual_uncertainty: number | null;
  uncertainty_status: string;
  covered_asset_ids: string;
};

export type Plan = {
  plans: Record<"A" | "B" | "C", string[]>;
  plan_rows: Array<Record<string, string | number>>;
  comparison: Comparison[];
  reasons: {
    B: Record<string, string>;
    C: Record<string, string>;
    not_selected_C: Record<string, string>;
  };
};
