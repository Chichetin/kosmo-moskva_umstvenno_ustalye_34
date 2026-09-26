"""Settings from the server environment and configs/media_sources.yaml. No secrets in code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from floodvalue.paths import REPO_ROOT, output_dir

SOURCES_YAML = Path(__file__).with_name("sources.yaml")


def media_dir() -> Path:
    """Database, photos and caches live outside git (outputs/ is ignored)."""
    return Path(os.environ.get("FLOODVALUE_MEDIA_DIR", output_dir() / "media"))


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    db_path: Path
    photo_dir: Path
    analyst_token: str | None
    public_reports: bool
    poll_interval_s: int | None  # overrides every source's poll_interval_s when set
    max_items_per_cycle: int
    max_requests_per_cycle: int
    llm_max_calls_per_cycle: int
    retention_days: int
    reports_per_hour: int
    geocoder: str
    user_agent: str
    sources_yaml: Path = field(default=SOURCES_YAML)


def load_settings() -> Settings:
    root = media_dir()
    return Settings(
        db_path=Path(os.environ.get("FLOODVALUE_MEDIA_DB", root / "media.sqlite")),
        photo_dir=root / "photos",
        analyst_token=os.environ.get("FLOODVALUE_MEDIA_ANALYST_TOKEN") or None,
        public_reports=os.environ.get("FLOODVALUE_MEDIA_PUBLIC_REPORTS") == "1",
        poll_interval_s=_int("FLOODVALUE_MEDIA_POLL_INTERVAL_S", 0) or None,
        max_items_per_cycle=_int("FLOODVALUE_MEDIA_MAX_ITEMS_PER_CYCLE", 200),
        max_requests_per_cycle=_int("FLOODVALUE_MEDIA_MAX_REQUESTS_PER_CYCLE", 20),
        llm_max_calls_per_cycle=_int("FLOODVALUE_MEDIA_LLM_MAX_CALLS", 0),
        retention_days=_int("FLOODVALUE_MEDIA_RETENTION_DAYS", 90),
        reports_per_hour=_int("FLOODVALUE_MEDIA_REPORTS_PER_HOUR", 5),
        geocoder=os.environ.get("FLOODVALUE_MEDIA_GEOCODER", "none"),
        user_agent=os.environ.get("FLOODVALUE_MEDIA_USER_AGENT",
                                  "FloodValue-media/0.1 (hackathon prototype; no bulk use)"),
    )


def load_sources(path: Path = SOURCES_YAML) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def split_parts() -> dict[str, list[str]]:
    """Events per split part, read as a file (contracts §4), not through floodvalue.data."""
    import json
    with open(REPO_ROOT / "splits" / "split_v1.json", encoding="utf-8") as f:
        return json.load(f)["events"]
