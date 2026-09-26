"""Every result_status a bundle can carry has a label in the «Анализ» table (#289).

Without a case the operator reads «Статус не указан» for a row the panel itself describes.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "web" / "src" / "SensitivityPanel.tsx"
LITERAL = re.compile(r'"result_status":\s*"(\w+)"')
# procurement writes the plan status of each sensitivity scenario (strategies.sensitivity)
PLAN_STATUSES = {"within_budget", "counterfactual_over_budget", "empty_plan"}


def produced_statuses() -> set[str]:
    found = set(PLAN_STATUSES) | {"no_data"}  # assets/sensitivity.py: "ok" if … else "no_data"
    for p in (ROOT / "src" / "floodvalue").rglob("*.py"):
        found |= set(LITERAL.findall(p.read_text(encoding="utf-8")))
    with open(ROOT / "tests" / "fixtures" / "good" / "sensitivity.csv", encoding="utf-8") as f:
        found |= {r["result_status"] for r in csv.DictReader(f)}
    return found


def test_every_produced_status_has_a_label():
    labelled = set(re.findall(r'case "(\w+)"', PANEL.read_text(encoding="utf-8")))
    produced = produced_statuses()
    assert {"fixed_plan", "ok", "recomputed"} <= produced  # the scan still sees the producers
    assert produced - labelled == set()
