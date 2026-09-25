"""Context imagery is for people, never for the model or the economics (D-042).

No pipeline module that produces p, the mask, thresholds, EL or the procurement plan may
reference optical/basemap context. A new reference here must be a deliberate, reviewed
decision (and then S1-only vs S1+S2 has to be compared, criterion 8), not an accident."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "floodvalue"
GUARDED = ["model", "baseline", "calibration", "assets", "procurement"]
FORBIDDEN = re.compile(r"S2Hand|floodvalue\.context|context_s2|s2cloudless|World_Imagery|tile\.openstreetmap")


def test_pipeline_modules_never_touch_context_imagery():
    hits = []
    for pkg in GUARDED:
        for f in sorted((SRC / pkg).rglob("*.py")):
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if FORBIDDEN.search(line):
                    hits.append(f"{f.relative_to(SRC)}:{i}: {line.strip()}")
    assert not hits, "контекстные снимки попали в конвейер модели/экономики:\n" + "\n".join(hits)


def test_guard_actually_scans_code():
    # A guard over empty directories would pass forever.
    assert all(any((SRC / pkg).rglob("*.py")) for pkg in GUARDED)
    assert (SRC / "context" / "s2.py").read_text(encoding="utf-8").count("S2Hand") > 0
