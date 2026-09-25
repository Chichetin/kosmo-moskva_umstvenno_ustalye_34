"""Frozen event split of the 446 Sen1Floods11 hand-labelled chips (D-006, D-012, D-016).

Parts are assigned by *event*, never by chip, so neighbouring windows of one S1 scene
always land in the same part:

    train       Ghana, India, Paraguay, Pakistan, Somalia   fit models and thresholds
    dev         USA                                         model/checkpoint choice, experiment
    validation  Sri-Lanka, Nigeria                          calibration, final threshold
    test        Spain, Mekong, Bolivia                      sealed: only `eval-test` (D-010)

Test chips are sealed at build time too: their labels are never opened. Chips are excluded
only by facts read from non-test labels (all pixels −1) or by missing files.

`make split` rebuilds the split from the raw data and checks it against
`splits/split_v1.json` and the sha256 recorded in `docs/decisions.md`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from floodvalue.paths import REPO_ROOT, SPLITS_DIR, data_dir

VERSION = 1
SEED = 20260925
SOURCE = "Sen1Floods11 v1.1 HandLabeled"
SPLIT_PATH = SPLITS_DIR / "split_v1.json"
DECISIONS_PATH = REPO_ROOT / "docs" / "decisions.md"

PARTS = ("train", "dev", "validation", "test")
OPEN_PARTS = ("train", "dev", "validation")
EVENT_PARTS: dict[str, tuple[str, ...]] = {
    "train": ("Ghana", "India", "Pakistan", "Paraguay", "Somalia"),
    "dev": ("USA",),
    "validation": ("Nigeria", "Sri-Lanka"),
    "test": ("Bolivia", "Mekong", "Spain"),
}
# file prefix -> `location` in Sen1Floods11_Metadata.geojson (docs/data.md §6)
METADATA_NAME = {"Mekong": "Cambodia"}
AUTHOR_LISTS = ("train", "valid", "test", "bolivia")
PROTOCOL = {
    "train": "обучение, нормализация, порог baseline",
    "dev": "выбор модели и чекпойнта, контролируемый эксперимент (D-012)",
    "validation": "калибровка и порог основного метода (D-012)",
    "test": "только eval-test замороженной конфигурации (D-010)",
    "weak": "S1Weak событий dev/validation/test и Colombia не используются",
}


class SealedTestError(PermissionError):
    """Raised when test chips are requested without the explicit eval-test flag."""


def event_of(chip_id: str) -> str:
    event, _ = chip_id.rsplit("_", 1)
    return event


def part_of_event(event: str) -> str:
    for part, events in EVENT_PARTS.items():
        if event in events:
            return part
    raise KeyError(f"event {event!r} is not assigned to any split part")


def author_chips(raw: Path) -> list[str]:
    """All hand-labelled chip ids listed in the authors' split CSVs (446 in v1.1)."""
    chips: set[str] = set()
    folder = raw / "splits" / "flood_handlabeled"
    for name in AUTHOR_LISTS:
        with open(folder / f"flood_{name}_data.csv", newline="") as fh:
            for row in csv.reader(fh):
                if row:
                    chips.add(row[0].removesuffix("_S1Hand.tif"))
    return sorted(chips)


def _label_path(raw: Path, chip_id: str) -> Path:
    return raw / "HandLabeled" / "LabelHand" / f"{chip_id}_LabelHand.tif"


def _s1_path(raw: Path, chip_id: str) -> Path:
    return raw / "HandLabeled" / "S1Hand" / f"{chip_id}_S1Hand.tif"


def all_invalid(raw: Path, chip_id: str) -> bool:
    """True if every label pixel is −1. Refuses test chips: their labels stay sealed."""
    if part_of_event(event_of(chip_id)) == "test":
        raise SealedTestError(f"label of test chip {chip_id} must not be read by split")
    import rasterio

    with rasterio.open(_label_path(raw, chip_id)) as src:
        return bool((src.read(1) == -1).all())


def exclusion_reason(raw: Path, chip_id: str) -> str | None:
    if not _s1_path(raw, chip_id).exists():
        return "нет S1Hand"
    if not _label_path(raw, chip_id).exists():
        return "нет LabelHand"
    if part_of_event(event_of(chip_id)) == "test":
        return None  # test labels are sealed; invalid pixels are masked at eval time
    if all_invalid(raw, chip_id):
        return "все пиксели LabelHand = −1 (нет валидной разметки)"
    return None


def check_events_disjoint(raw: Path) -> None:
    """Event footprints (metadata polygons) of different parts must not touch."""
    from shapely.geometry import shape

    meta = json.loads((raw / "Sen1Floods11_Metadata.geojson").read_text())
    geom = {f["properties"]["location"]: shape(f["geometry"]) for f in meta["features"]}
    for i, a in enumerate(PARTS):
        for b in PARTS[i + 1:]:
            for ea in EVENT_PARTS[a]:
                for eb in EVENT_PARTS[b]:
                    ga = geom[METADATA_NAME.get(ea, ea)]
                    gb = geom[METADATA_NAME.get(eb, eb)]
                    if ga.intersects(gb):
                        raise ValueError(f"event footprints overlap: {ea} ({a}) / {eb} ({b})")


def canonical_hash(split: dict) -> str:
    """sha256 of the canonical JSON of the chip lists (all four parts)."""
    lists = {part: sorted(split[part]) for part in PARTS}
    blob = json.dumps(lists, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def build_split(raw: Path | None = None, seed: int = SEED) -> dict:
    """Rule-based: membership depends only on the event table, not on the seed.

    The seed is recorded for downstream randomness (sampling, object placement).
    """
    raw = raw or data_dir() / "raw"
    check_events_disjoint(raw)
    parts: dict[str, list[str]] = {p: [] for p in PARTS}
    excluded: dict[str, str] = {}
    for chip in author_chips(raw):
        part = part_of_event(event_of(chip))
        reason = exclusion_reason(raw, chip)
        if reason:
            excluded[chip] = reason
        else:
            parts[part].append(chip)
    split = {
        "version": VERSION,
        "criterion": "event",
        "seed": seed,
        "source": SOURCE,
        **{p: sorted(parts[p]) for p in PARTS},
        "events": {p: list(EVENT_PARTS[p]) for p in PARTS},
        "excluded": dict(sorted(excluded.items())),
        "protocol": PROTOCOL,
    }
    split["sha256"] = canonical_hash(split)
    return split


def dumps(split: dict) -> str:
    return json.dumps(split, ensure_ascii=False, indent=1) + "\n"


def load_split(path: Path | None = None) -> dict:
    split = json.loads((path or SPLIT_PATH).read_text())
    if canonical_hash(split) != split["sha256"]:
        raise ValueError("split file lists do not match its sha256")
    return split


def chip_ids(part: str, *, allow_test: bool = False, path: Path | None = None) -> list[str]:
    """Chip ids of one part. `test` only with allow_test=True (eval-test, D-010)."""
    if part not in PARTS:
        raise KeyError(f"unknown part {part!r}; expected one of {PARTS}")
    if part == "test" and not allow_test:
        raise SealedTestError("test chips are sealed: only eval-test passes allow_test=True")
    return list(load_split(path)[part])


def decisions_hash(path: Path = DECISIONS_PATH) -> str | None:
    """sha256 recorded in the decisions log row that mentions split_v1.json."""
    for line in path.read_text().splitlines():
        if "split_v1.json" in line:
            m = re.search(r"sha256[^0-9a-f]*([0-9a-f]{64})", line)
            if m:
                return m.group(1)
    return None


def run(args) -> int:
    built = build_split()
    counts = ", ".join(f"{p} {len(built[p])}" for p in PARTS)
    print(f"split v{VERSION}: {counts}, excluded {len(built['excluded'])}")
    print(f"sha256 {built['sha256']}")
    if not SPLIT_PATH.exists():
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_PATH.write_text(dumps(built))
        print(f"written {SPLIT_PATH}; record the sha256 in docs/decisions.md")
        return 0
    ok = True
    if SPLIT_PATH.read_text() != dumps(built):
        print(f"MISMATCH: rebuilt split differs from {SPLIT_PATH}")
        ok = False
    recorded = decisions_hash()
    if recorded != built["sha256"]:
        print(f"MISMATCH: docs/decisions.md records {recorded}")
        ok = False
    if ok:
        print("OK: rebuilt split == splits/split_v1.json == docs/decisions.md")
    return 0 if ok else 1
