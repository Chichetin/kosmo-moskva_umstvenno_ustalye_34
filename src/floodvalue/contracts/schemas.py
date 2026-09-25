"""Pydantic v2 schemas of the 12 required files and configs/run.yaml (D-004, docs/contracts.md).

Row models validate one CSV row / one GeoJSON feature's properties / one JSON document.
Cross-file checks (EL = p·V·q, mask vs threshold, geometry, sums) live in
floodvalue.export.validate. All models allow extra fields and reject NaN/Inf.
"""

import json
import math
from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]
Positive = Annotated[float, Field(gt=0.0)]


def _csv_bool(value: Any) -> bool:
    """CSV booleans are the literal strings true/false (task.md, procurement_plan.csv)."""
    if isinstance(value, bool):
        return value
    if value in ("true", "false"):
        return value == "true"
    raise ValueError(f"ожидается true/false, получено {value!r}")


CsvBool = Annotated[bool, BeforeValidator(_csv_bool)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class _CsvRow(_Model):
    """CSV cells arrive as strings; an empty cell means 'no value' (never NaN, never 0)."""

    @field_validator("*", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        return None if value == "" else value


# --- assets.geojson / assets.csv -------------------------------------------------------------

class AssetProperties(_Model):
    asset_id: NonEmptyStr
    chip_id: NonEmptyStr
    asset_class: NonEmptyStr
    asset_value_rub: Positive
    vulnerability_coef: Probability


class AssetCsvRow(_CsvRow, AssetProperties):
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]
    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]


# --- asset_loss.csv ----------------------------------------------------------------------------

class AssetLossRow(_CsvRow):
    asset_id: NonEmptyStr
    chip_id: NonEmptyStr
    p_flood: Probability | None
    expected_loss_rub: NonNegative | None
    uncertainty: float | None
    rank: Annotated[int, Field(ge=1)] | None
    status: Literal["ok", "partial", "no_data"]

    @model_validator(mode="after")
    def _status_rules(self):
        values = (self.p_flood, self.expected_loss_rub, self.rank)
        if self.status == "ok" and any(v is None for v in values):
            raise ValueError("status=ok требует p_flood, expected_loss_rub и rank")
        if self.status != "ok" and any(v is not None for v in values):
            raise ValueError(f"status={self.status}: p_flood/expected_loss_rub/rank должны быть пустыми")
        return self


# --- candidate_orders.geojson ------------------------------------------------------------------

class CandidateProperties(_Model):
    candidate_id: NonEmptyStr
    sensor_type: NonEmptyStr
    resolution_m: Positive
    acquisition_type: Literal["new", "operational", "archive"]
    data_role: Literal["event_observation", "context"]
    observation_at: datetime | None = Field(...)
    available_at: datetime | None = Field(...)
    processing_level: Literal["L0", "L1", "L2"]
    usage_type: Literal["internal", "limited", "unrestricted"]
    guaranteed_purchase: StrictBool
    area_km2: Positive
    base_rate_rub_km2: Positive
    base_rate_status: Literal["scenario", "official"]
    base_rate_source: NonEmptyStr

    @model_validator(mode="after")
    def _guarantee_rules(self):
        if self.acquisition_type == "new" and not self.guaranteed_purchase:
            raise ValueError("acquisition_type=new без guaranteed_purchase=true — правило цены не задано")
        if self.acquisition_type != "new" and self.guaranteed_purchase:
            raise ValueError("guaranteed_purchase=true допустимо только для acquisition_type=new")
        return self


# --- procurement_plan.csv ----------------------------------------------------------------------

class ProcurementPlanRow(_CsvRow):
    strategy: Literal["B", "C"]  # A has no paid rows
    candidate_id: NonEmptyStr
    area_km2: Positive
    base_rate_rub_km2: Positive
    base_rate_status: Literal["scenario", "official"]
    processing_level: Literal["L0", "L1", "L2"]
    usage_type: Literal["internal", "limited", "unrestricted"]
    guaranteed_purchase: CsvBool
    processing_coef: Positive
    usage_coef: Positive
    freshness_coef: Positive
    discount_coef: Annotated[float, Field(gt=0.0, le=1.0)]
    discount_group_id: NonEmptyStr
    group_area_km2: Positive
    unit_price_rub_km2: NonNegative
    cost_rub: NonNegative
    formula: NonEmptyStr
    legal_edition: NonEmptyStr


# --- strategy_plans.json -----------------------------------------------------------------------

class StrategyPlans(BaseModel):
    model_config = ConfigDict(extra="forbid")
    A: list[NonEmptyStr]
    B: list[NonEmptyStr]
    C: list[NonEmptyStr]

    @model_validator(mode="after")
    def _rules(self):
        if self.A:
            raise ValueError("A — без платных зон: список A должен быть пустым")
        for name in ("B", "C"):
            ids = getattr(self, name)
            if len(ids) != len(set(ids)):
                raise ValueError(f"повтор candidate_id в стратегии {name}")
        return self


# --- strategy_comparison.csv -------------------------------------------------------------------

class StrategyComparisonRow(_CsvRow):
    strategy: Literal["A", "B", "C"]
    data_cost_rub: NonNegative
    other_cost_rub: NonNegative
    decision_cost_rub: NonNegative
    budget_rub: NonNegative
    budget_feasible: CsvBool
    covered_expected_loss_rub: NonNegative
    coverage_share: Probability | None
    residual_uncertainty: float | None
    uncertainty_status: Literal["baseline", "scenario", "measured", "not_estimated"]
    uncertainty_basis: NonEmptyStr

    @model_validator(mode="after")
    def _rules(self):
        if self.uncertainty_status == "not_estimated" and self.residual_uncertainty is not None:
            raise ValueError("uncertainty_status=not_estimated: residual_uncertainty должно быть пустым")
        if self.uncertainty_status != "not_estimated" and self.residual_uncertainty is None:
            raise ValueError(f"uncertainty_status={self.uncertainty_status} требует residual_uncertainty")
        if self.strategy == "A" and self.uncertainty_status != "baseline":
            raise ValueError("для A uncertainty_status должен быть baseline")
        return self


# --- sensitivity.csv ---------------------------------------------------------------------------

def _reject_constant(name: str) -> Any:
    raise ValueError(f"недопустимое значение {name} в JSON")


class SensitivityRow(_CsvRow):
    scenario_id: NonEmptyStr
    strategy: Literal["A", "B", "C"]
    changed_inputs_json: NonEmptyStr
    decision_cost_rub: NonNegative | None
    covered_expected_loss_rub: NonNegative | None
    result_status: NonEmptyStr
    interpretation: NonEmptyStr

    @field_validator("changed_inputs_json")
    @classmethod
    def _json_object(cls, value: str) -> str:
        parsed = json.loads(value, parse_constant=_reject_constant)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("changed_inputs_json — непустой JSON-объект")
        return value

    def changed_keys(self) -> set[str]:
        return set(json.loads(self.changed_inputs_json))


# --- run_metadata.json -------------------------------------------------------------------------

class WeightFile(_Model):
    path: NonEmptyStr
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class MethodInfo(_Model):
    name: NonEmptyStr
    version: NonEmptyStr
    weights: list[WeightFile]
    threshold: float
    threshold_units: NonEmptyStr  # "p" for the main method, "dB VH" etc. for baseline
    threshold_source: NonEmptyStr  # which split/procedure chose it


class MethodsInfo(_Model):
    baseline: MethodInfo
    main: MethodInfo
    evaluation_order: NonEmptyStr


class SourceInfo(_Model):
    dataset: NonEmptyStr
    dataset_url: NonEmptyStr
    s1_files: Annotated[list[NonEmptyStr], Field(min_length=1)]
    source_manifest: NonEmptyStr


class SplitInfo(_Model):
    file: NonEmptyStr
    sha256: str | None  # None until the split is frozen
    chip_split: Literal["train", "dev", "validation", "test", "none"]


class PortfolioInfo(_Model):
    seed: int
    placement_rule: NonEmptyStr
    assets_config: NonEmptyStr


class ExpectedLossInfo(_Model):
    formula: NonEmptyStr
    p_source: NonEmptyStr
    status_rules: NonEmptyStr


class BudgetInfo(_Model):
    budget_rub: NonNegative
    decision_at: datetime
    decision_deadline: datetime

    @model_validator(mode="after")
    def _order(self):
        if self.decision_deadline < self.decision_at:
            raise ValueError("decision_deadline раньше decision_at")
        return self


class PricingInfo(_Model):
    config: NonEmptyStr
    legal_edition: NonEmptyStr
    base_rate_rub_km2: Positive
    base_rate_status: Literal["scenario", "official"]
    discount_rule: NonEmptyStr
    grouping: NonEmptyStr


class ZonesInfo(_Model):
    construction: NonEmptyStr
    availability_rule: NonEmptyStr
    area_method: NonEmptyStr
    unknown_dates_rule: NonEmptyStr


class StrategiesInfo(_Model):
    rule_B: NonEmptyStr
    rule_C: NonEmptyStr
    other_costs: NonEmptyStr
    coverage_aggregation: NonEmptyStr
    coverage_share_note: str | None = None  # why coverage_share is empty, if it is


class UncertaintyInfo(_Model):
    meaning: NonEmptyStr
    units: NonEmptyStr
    method: NonEmptyStr
    raster: str | None = None


class ProxyCheckInfo(_Model):
    status: Literal["done", "planned", "not_done"]
    test_chip: str | None
    control_points: str | None
    fixed_v_q: bool | None
    n_points: int | None
    error_metric: str | None
    result: Any = None


class MetricsInfo(_Model):
    files: Annotated[list[NonEmptyStr], Field(min_length=1)]  # metrics.csv of both methods


class RunMetadata(_Model):
    contract_version: Literal["1"]
    run_id: NonEmptyStr
    chip_id: NonEmptyStr
    event_id: NonEmptyStr
    event_date: date
    created_at: datetime
    commit: NonEmptyStr
    threshold: Probability  # main method: mask = 1 <=> p >= threshold
    source: SourceInfo
    split: SplitInfo
    portfolio: PortfolioInfo
    methods: MethodsInfo
    expected_loss: ExpectedLossInfo
    budget: BudgetInfo
    pricing: PricingInfo
    zones: ZonesInfo
    strategies: StrategiesInfo
    uncertainty: UncertaintyInfo
    proxy_check: ProxyCheckInfo
    metrics: MetricsInfo
    reproduce: Annotated[list[NonEmptyStr], Field(min_length=1)]

    @model_validator(mode="after")
    def _threshold(self):
        if not math.isclose(self.methods.main.threshold, self.threshold, rel_tol=0, abs_tol=1e-12):
            raise ValueError("methods.main.threshold не совпадает с threshold")
        return self


# --- source_manifest.json ----------------------------------------------------------------------

class SourceEntry(_Model):
    id: NonEmptyStr
    file: str | None
    url: NonEmptyStr
    version: NonEmptyStr
    observation_date: date | None
    available_date: date | None
    license: NonEmptyStr
    purpose: NonEmptyStr
    limitations: NonEmptyStr
    applies_to: str | None = None
    after_event_note: str | None = None  # mandatory if observation_date > event_date


class SourceManifest(_Model):
    sources: Annotated[list[SourceEntry], Field(min_length=1)]


# --- configs/run.yaml --------------------------------------------------------------------------

class RunConfig(_Model):
    chip_id: NonEmptyStr
    event_id: NonEmptyStr
    chip_split: Literal["train", "dev", "validation", "test"]
    portfolio_seed: int
    budget_rub: NonNegative
    decision_at: datetime
    decision_deadline: datetime

    @model_validator(mode="after")
    def _order(self):
        if self.decision_deadline < self.decision_at:
            raise ValueError("decision_deadline раньше decision_at")
        return self
