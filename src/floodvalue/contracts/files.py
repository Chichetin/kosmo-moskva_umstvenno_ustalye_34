"""Names, fields and raster specs of the 12 required files (task.md, section III).

Field lists are the mandatory minimum, copied verbatim from task.md; extra fields are allowed.
"""

CONTRACT_VERSION = "1"

FLOOD_PROBABILITY = "flood_probability.tif"
FLOOD_MASK = "flood_mask.tif"
ASSETS_GEOJSON = "assets.geojson"
ASSETS_CSV = "assets.csv"
ASSET_LOSS = "asset_loss.csv"
CANDIDATE_ORDERS = "candidate_orders.geojson"
PROCUREMENT_PLAN = "procurement_plan.csv"
STRATEGY_PLANS = "strategy_plans.json"
STRATEGY_COMPARISON = "strategy_comparison.csv"
SENSITIVITY = "sensitivity.csv"
RUN_METADATA = "run_metadata.json"
SOURCE_MANIFEST = "source_manifest.json"

REQUIRED_FILES = (
    FLOOD_PROBABILITY, FLOOD_MASK, ASSETS_GEOJSON, ASSETS_CSV, ASSET_LOSS, CANDIDATE_ORDERS,
    PROCUREMENT_PLAN, STRATEGY_PLANS, STRATEGY_COMPARISON, SENSITIVITY, RUN_METADATA,
    SOURCE_MANIFEST,
)

# Rasters: same grid (CRS, transform, size) as the source S1 chip.
PROBABILITY_DTYPE = "float32"
PROBABILITY_NODATA = -9999.0
MASK_DTYPE = "uint8"
MASK_NODATA = 255

ASSET_FIELDS = ("asset_id", "chip_id", "asset_class", "asset_value_rub", "vulnerability_coef")
ASSETS_CSV_FIELDS = ASSET_FIELDS + ("longitude", "latitude")
ASSET_LOSS_FIELDS = ("asset_id", "chip_id", "p_flood", "expected_loss_rub", "uncertainty",
                     "rank", "status")
CANDIDATE_FIELDS = (
    "candidate_id", "sensor_type", "resolution_m", "acquisition_type", "data_role",
    "observation_at", "available_at", "processing_level", "usage_type", "guaranteed_purchase",
    "area_km2", "base_rate_rub_km2", "base_rate_status", "base_rate_source",
)
PROCUREMENT_PLAN_FIELDS = (
    "strategy", "candidate_id", "area_km2", "base_rate_rub_km2", "base_rate_status",
    "processing_level", "usage_type", "guaranteed_purchase", "processing_coef", "usage_coef",
    "freshness_coef", "discount_coef", "discount_group_id", "group_area_km2",
    "unit_price_rub_km2", "cost_rub", "formula", "legal_edition",
)
STRATEGY_COMPARISON_FIELDS = (
    "strategy", "data_cost_rub", "other_cost_rub", "decision_cost_rub", "budget_rub",
    "budget_feasible", "covered_expected_loss_rub", "coverage_share", "residual_uncertainty",
    "uncertainty_status", "uncertainty_basis",
)
SENSITIVITY_FIELDS = ("scenario_id", "strategy", "changed_inputs_json", "decision_cost_rub",
                      "covered_expected_loss_rub", "result_status", "interpretation")

# Keys of changed_inputs_json that sensitivity.csv must vary one at a time (task.md, II).
SENSITIVITY_REQUIRED_KEYS = ("p_flood", "asset_value_rub", "vulnerability_coef", "budget_rub",
                             "base_rate_rub_km2")

STRATEGIES = ("A", "B", "C")
UNCERTAINTY_STATUSES = ("baseline", "scenario", "measured", "not_estimated")
ASSET_STATUSES = ("ok", "partial", "no_data")
