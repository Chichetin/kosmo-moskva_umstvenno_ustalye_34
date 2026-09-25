"""Strict readers for contract files: no NaN/Inf literals in CSV, JSON or GeoJSON."""

import csv
import json
from pathlib import Path
from typing import Any

NON_FINITE_LITERALS = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity",
                       "-infinity"}


def _reject_constant(name: str) -> Any:
    raise ValueError(f"недопустимое значение {name} (NaN/Inf) в JSON")


def read_json(path: Path) -> Any:
    """json.load that fails on NaN/Infinity instead of silently accepting them."""
    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=_reject_constant)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Return (header, rows, errors). Errors list NaN/Inf literals cell by cell."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    errors = []
    for i, row in enumerate(rows, start=2):
        if None in row:
            errors.append(f"строка {i}: лишние ячейки без заголовка")
        for key, value in row.items():
            if isinstance(value, str) and value.strip().lower() in NON_FINITE_LITERALS:
                errors.append(f"строка {i}, {key}: {value!r} — вместо NaN/Inf нужна пустая ячейка")
    return header, rows, errors
