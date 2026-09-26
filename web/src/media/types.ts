export type Origin = "media" | "bulletin" | "citizen";
export type ReportType =
  | "occurred"
  | "forecast"
  | "warning"
  | "negation"
  | "historical"
  | "unclear";
export type ReviewState =
  | "new"
  | "needs_geo"
  | "ready"
  | "under_review"
  | "reviewed"
  | "merged"
  | "split"
  | "dismissed";
export type VerificationResult =
  | "not_checked"
  | "consistent"
  | "discrepancy"
  | "insufficient_data";

export type MediaEvent = {
  event_id: number;
  title: string;
  report_type: ReportType;
  place_label: string | null;
  country: string | null;
  lat: number | null;
  lon: number | null;
  bbox: [number, number, number, number] | null;
  geo_precision: string;
  geo_status: "resolved" | "ambiguous" | "missing";
  geo_candidates: Array<{
    label: string;
    lat: number;
    lon: number;
    precision: string;
    bbox: [number, number, number, number];
  }> | null;
  event_time_start: string | null;
  event_time_end: string | null;
  time_precision: string;
  first_detected_at: string;
  updated_at: string;
  review_state: ReviewState;
  verification_result: VerificationResult;
  follow_up: string;
  possible_duplicate_of: number | null;
  merged_into: number | null;
  is_demo: boolean;
  is_replay: boolean;
  evidence_counts: Record<Origin, number>;
  independent_origin_count: number;
  area_radius_km: number | null;
};

export type CitizenReport = {
  report_id: number;
  public_id: string;
  kind: string;
  kind_label: string;
  description: string;
  lat: number | null;
  lon: number | null;
  place_text: string | null;
  location_precision: string;
  observed_at: string;
  observed_time_precision: string;
  observed_tz: string;
  submitted_at: string;
  moderation_status: "pending" | "approved" | "rejected";
  is_demo: boolean;
  retention_until: string;
  photos: Array<{ url: string; width: number; height: number }>;
  event_ids: number[];
};

export type Publication = {
  publication_id: number;
  url: string;
  domain: string | null;
  title: string;
  text_excerpt: string | null;
  content_level: "title_only" | "excerpt" | "structured";
  language: string | null;
  published_at: string | null;
  fetched_at: string;
  report_type: ReportType | null;
  extraction_method: string;
  extraction_confidence: string | null;
  original_source_hint: string | null;
  source_name: string;
  source_kind: string;
  attribution: string | null;
};

export type Evidence = {
  evidence_id: number;
  origin: Origin;
  link_kind: string;
  linked_at: string;
  linked_by: string;
  is_original: boolean;
  facts: Record<string, unknown> | null;
  publication?: Publication;
  report?: CitizenReport;
};

export type ObservationMatch = {
  match_id: number;
  catalog: "local" | "copernicus";
  observation_ref: string;
  product_type: string | null;
  observation_at: string | null;
  footprint_json: string | null;
  spatial_relation: string;
  distance_km: number | null;
  time_offset_days: number | null;
  suitability: "suitable" | "too_early" | "too_late" | "partial_cover" | "unusable";
  reason: string;
  stage: "found" | "retrieved" | "prepared" | "analyzed";
  analysis_run_id: string | null;
  searched_at: string;
};

export type EventCard = {
  event: MediaEvent;
  evidence: Evidence[];
  history: Array<{
    at: string;
    actor: string;
    action: string;
    before_json: string | null;
    after_json: string | null;
    note: string | null;
  }>;
  reviews: Array<Record<string, string | number | null>>;
  observations: ObservationMatch[];
  merge_candidates: Array<{
    event_id: number;
    title: string;
    distance_km: number;
    confidence: "high" | "medium";
  }>;
  gaps: string[];
};

export type SourceStatus = {
  source_id: string;
  name: string;
  kind: string;
  enabled: boolean;
  terms_status: string;
  attribution: string | null;
  state: "ok" | "delayed" | "error" | "disabled" | "never";
  last_attempt_at: string | null;
  last_success_at: string | null;
  last_new_item_at: string | null;
  last_error_at: string | null;
  last_error: string | null;
};

export type Summary = {
  pending_reports: number;
  needs_geo: number;
  worker: { alive: boolean; heartbeat_at: string | null };
  monitoring?: { enabled: boolean; interval_s: number; run_requested: boolean };
  sources?: SourceStatus[];
};

export type MessageFilters = {
  origins: Record<Origin, boolean>;
  days: number;
  types: "situation" | "alerts" | "all";
  state: "open" | "needs_geo" | "reviewed" | "all";
};
