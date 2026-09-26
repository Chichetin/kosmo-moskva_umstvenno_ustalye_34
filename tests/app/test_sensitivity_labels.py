"""Every result_status a bundle can carry has a label in the «Анализ» table (#289).

Without a case the operator reads «Статус не указан» for a row the panel itself describes.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "web" / "src" / "SensitivityPanel.tsx"
LITERAL = re.compile(r'"result_status":\s*"(\w+)"')
STRATEGIES = ROOT / "src" / "floodvalue" / "procurement" / "strategies.py"
# procurement writes the plan status of each sensitivity scenario: `status = ("…" if … else "…")`
PLAN_STATUS = re.compile(r'status = \(([^)]*)\)')


def produced_statuses() -> set[str]:
    found = {"no_data"}  # assets/sensitivity.py: "ok" if … else "no_data"
    for expr in PLAN_STATUS.findall(STRATEGIES.read_text(encoding="utf-8")):
        found |= set(re.findall(r'"(\w+)"', expr))
    for p in (ROOT / "src" / "floodvalue").rglob("*.py"):
        found |= set(LITERAL.findall(p.read_text(encoding="utf-8")))
    with open(ROOT / "tests" / "fixtures" / "good" / "sensitivity.csv", encoding="utf-8") as f:
        found |= {r["result_status"] for r in csv.DictReader(f)}
    return found


def test_every_produced_status_has_a_label():
    body = re.search(r"function statusLabel\(.*?\n}\n", PANEL.read_text(encoding="utf-8"),
                     re.DOTALL)
    assert body, "statusLabel not found in SensitivityPanel.tsx"
    labelled = set(re.findall(r'case "(\w+)"', body.group(0)))
    produced = produced_statuses()
    # the scan still sees every producer
    assert {"fixed_plan", "ok", "recomputed", "within_budget", "empty_plan",
            "counterfactual_over_budget"} <= produced
    assert produced - labelled == set()
