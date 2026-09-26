"""Простой детерминированный график для слайдов презентации (issue #36).

Не пересчитывает метрики — только читает уже сгенерированные CSV из
docs/verification/ и рисует сравнение IoU (ALL) baseline / pixel / U-Net (сырой)
/ U-Net (калиброванный) на dev и validation. Числа не выдумываются: если файла
или строки нет, столбец не рисуется, а не подставляется значение.

Запуск:
    uv run --frozen python scripts/presentation_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VERIFICATION = ROOT / "docs" / "verification"
OUT_DIR = VERIFICATION / "presentation"


def _read_iou(csv_path: Path, method: str, split: str, event: str = "ALL") -> float | None:
    if not csv_path.exists():
        return None
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == method and row["split"] == split and row["event"] == event:
                return float(row["iou"])
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    series = {
        "baseline": [
            _read_iou(VERIFICATION / "baseline-metrics.csv", "baseline", "dev", "USA"),
            _read_iou(VERIFICATION / "baseline-metrics.csv", "baseline", "validation"),
        ],
        "pixel (кандидат A)": [
            _read_iou(VERIFICATION / "pixel-metrics.csv", "pixel", "dev", "USA"),
            _read_iou(VERIFICATION / "pixel-metrics.csv", "pixel", "validation"),
        ],
        "U-Net, сырой p≥0.5": [
            _read_iou(VERIFICATION / "unet-metrics.csv", "unet", "dev"),
            _read_iou(VERIFICATION / "unet-metrics.csv", "unet", "validation"),
        ],
        "U-Net, калиброванный p≥0.37": [
            _read_iou(VERIFICATION / "calibration" / "unet-metrics.csv", "unet", "dev"),
            _read_iou(VERIFICATION / "calibration" / "unet-metrics.csv", "unet", "validation"),
        ],
    }

    labels = ["dev (USA, выбор)", "validation (справочно)"]
    x = range(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (name, values) in enumerate(series.items()):
        offsets = [xi + (i - 1.5) * width for xi in x]
        plotted = [v if v is not None else 0.0 for v in values]
        bars = ax.bar(offsets, plotted, width=width, label=name)
        for bar, v in zip(bars, values):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("IoU (пулинг, ALL событий части)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline vs основной метод: IoU на dev и validation")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out_path = OUT_DIR / "method-comparison-iou.png"
    fig.savefig(out_path, dpi=150)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
