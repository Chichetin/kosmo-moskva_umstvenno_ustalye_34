"""`python -m floodvalue export [--chip <id>] [--budget <руб>] [--out <dir>]`: one result set.

Reads the files the earlier stages wrote into the working directory
(`$FLOODVALUE_OUTPUT_DIR/<chip>/`, or `--out` itself when the stages were run with the same
`--out`) and assembles `$FLOODVALUE_OUTPUT_DIR/<run_id>/` (contracts §1, D-027):

- copies the rasters, assets, zones, plan and A/B/C files without recomputation;
- sensitivity.csv = procurement_sensitivity.csv ∪ assets_sensitivity.csv;
- metrics.csv (§7) from docs/verification: baseline and main on the same valid-pixel mask;
- source_manifest.json (§9) from $FLOODVALUE_DATA_DIR/interim/source_manifest_s1.json;
- run_metadata.json (§8) from prediction.json, assets_metadata.json, procurement_metadata.json,
  the split, configs, commit and package versions.
Nothing is recomputed here: p, mask, threshold, portfolio and plan come from the stage files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import yaml

from floodvalue.contracts import files as F
from floodvalue.contracts import load_run_config
from floodvalue.export import assemble
from floodvalue.paths import CONFIGS_DIR, MODELS_DIR, REPO_ROOT, SPLITS_DIR, data_dir, output_dir

COPIED = (F.FLOOD_PROBABILITY, F.FLOOD_MASK, F.ASSETS_GEOJSON, F.ASSETS_CSV, F.ASSET_LOSS,
          F.CANDIDATE_ORDERS, F.PROCUREMENT_PLAN, F.STRATEGY_PLANS, F.STRATEGY_COMPARISON)
OPTIONAL = ("baseline_mask.tif", "uncertainty.tif", "prediction.json", "assets_metadata.json",
            "procurement_metadata.json", "procurement_sensitivity.csv", "assets_sensitivity.csv")
STAGE_META = ("prediction.json", "assets_metadata.json", "procurement_metadata.json",
              "procurement_sensitivity.csv")
BASELINE_METRICS = "docs/verification/baseline-metrics.csv"
MAIN_METRICS = "docs/verification/main-metrics.csv"  # written by the main method when it lands
BASELINE_CONFIG = MODELS_DIR / "baseline" / "config.json"
SPLIT_FILE = SPLITS_DIR / "split_v1.json"
PACKAGES = ("numpy", "rasterio", "shapely", "pyproj", "pydantic", "pyyaml", "scikit-learn",
            "scipy", "torch", "pandas")


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow(["" if r.get(k) is None else r.get(k) for k in fields])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
                    encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> tuple[str, str, bool]:
    """(full sha, short sha, working tree dirty); ('unknown', 'nogit', True) outside git."""
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], check=True,
                              capture_output=True, text=True).stdout.strip()
    try:
        full, short = git("rev-parse", "HEAD"), git("rev-parse", "--short=7", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "nogit", True
    return full, short, dirty


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    lock = REPO_ROOT / "uv.lock"
    if lock.is_file():
        out["uv.lock_sha256"] = _sha256(lock)
    return out


def _work_dir(args, chip_id: str) -> Path:
    """Where predict/assets/procure wrote: --out if the stages used it, else outputs/<chip>."""
    if getattr(args, "out", None) and (Path(args.out) / "assets_metadata.json").is_file():
        return Path(args.out)
    return output_dir() / chip_id


def _metrics(work_prediction: dict, dest: Path, warnings: list[str]) -> list[str]:
    base_path = REPO_ROOT / BASELINE_METRICS
    method = str(work_prediction.get("method", "main"))
    main_rel = next((rel for rel in (f"docs/verification/{method}-metrics.csv", MAIN_METRICS)
                     if (REPO_ROOT / rel).is_file()), MAIN_METRICS)
    main_path = REPO_ROOT / main_rel
    if not base_path.is_file():
        warnings.append(f"нет {BASELINE_METRICS}: metrics.csv без baseline (make baseline)")
        base_rows = []
    else:
        base_rows = _read_csv(base_path)
    main_rows = _read_csv(main_path) if main_path.is_file() else None
    if main_rows and assemble.is_baseline_prediction(work_prediction):
        warnings.append(f"{main_rel} есть, но растр комплекта — baseline (D-009): строки "
                        "main взяты из baseline, чтобы метрики соответствовали растру")
        main_rows = None
    if main_rows is None and not assemble.is_baseline_prediction(work_prediction):
        warnings.append(f"основной метод {method!r} без {main_rel}: "
                        "в metrics.csv нет строк main")
    stale = sorted({r.get("threshold") for r in main_rows or []
                    if r.get("threshold") not in (None, "")
                    and abs(float(r["threshold"]) - float(work_prediction["threshold"])) > 1e-9})
    if stale:
        warnings.append(f"{main_rel}: порог строк main {stale} ≠ порогу комплекта "
                        f"{work_prediction['threshold']} — метрики не соответствуют маске")
    rows = assemble.metrics_rows(base_rows, main_rows, work_prediction, BASELINE_METRICS,
                                 main_rel if main_rows else None)
    _write_csv(dest / "metrics.csv", assemble.METRICS_FIELDS + assemble.METRICS_EXTRA, rows)
    files = ["metrics.csv", BASELINE_METRICS]
    return files + ([main_rel] if main_rows else [])


def export(chip_id: str, work: Path, dest: Path | None = None,
           budget_rub: float | None = None) -> Path:
    """Assemble the result set; returns its directory. Raises FileNotFoundError/ValueError."""
    missing = [n for n in (*COPIED, *STAGE_META) if not (work / n).is_file()]
    if missing:
        raise FileNotFoundError(f"в {work} нет файлов этапов: {missing} — сначала "
                                "make predict/assets/procure с тем же CHIP")
    prediction = _read_json(work / "prediction.json")
    assets_meta = _read_json(work / "assets_metadata.json")
    proc_meta = _read_json(work / "procurement_metadata.json")
    for name, meta in (("prediction.json", prediction), ("assets_metadata.json", assets_meta)):
        if meta.get("chip_id") not in (None, chip_id):
            raise ValueError(f"{work / name}: chip_id {meta.get('chip_id')} ≠ {chip_id}")
    budget = float(proc_meta["budget_rub"])
    if budget_rub is not None and abs(float(budget_rub) - budget) > 1e-9:
        raise ValueError(f"procure считал бюджет {budget}, а export просит {budget_rub} — "
                         f"перезапустить make procure BUDGET={budget_rub}")

    commit, short, dirty = git_commit()
    run_id = assemble.make_run_id(chip_id, budget, short)
    dest = dest or output_dir() / run_id
    dest.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if dirty:
        warnings.append("рабочее дерево git с незакоммиченными изменениями: commit не "
                        "воспроизводит запуск полностью")
    if str(prediction.get("threshold_status", "")).startswith("uncalibrated"):
        warnings.append("вероятности основного метода не калиброваны, порог 0,5 не выбран на "
                        "validation (калибровка — #20): " + str(prediction["threshold_status"]))
    if assemble.is_baseline_prediction(prediction):
        warnings.append("основной метод — временно baseline-сигмоида (D-009): "
                        "floodvalue.model.predict ещё не влит (#16/#17)")

    if dest.resolve() != work.resolve():
        for name in COPIED:
            shutil.copy2(work / name, dest / name)
        for name in OPTIONAL:
            if (work / name).is_file():
                shutil.copy2(work / name, dest / name)
            elif (dest / name).exists():
                (dest / name).unlink()  # stale file of an earlier export into the same dir

    # sensitivity.csv -------------------------------------------------------------------------
    proc_sens = _read_csv(work / "procurement_sensitivity.csv")
    assets_sens_path = work / "assets_sensitivity.csv"
    assets_sens = _read_csv(assets_sens_path) if assets_sens_path.is_file() else None
    if assets_sens is None:
        warnings.append("нет assets_sensitivity.csv (#9): sensitivity.csv только из procurement "
                        "(бюджет, ставка) — сценарии p, V, q отсутствуют")
    if assets_sens:
        c_rows, c_warn = assemble.fixed_plan_rows(
            assets_sens, _read_csv(work / F.ASSETS_CSV), _read_csv(work / F.ASSET_LOSS),
            _read_csv(work / F.STRATEGY_COMPARISON),
            _read_json(work / F.STRATEGY_PLANS).get("C", []))
        warnings += c_warn
        assets_sens = assemble.with_model_note(assets_sens) + c_rows
    sens = assemble.merge_sensitivity(proc_sens, assets_sens)
    lacking = assemble.missing_sensitivity_keys(sens)
    if lacking:
        warnings.append(f"sensitivity.csv: нет раздельных сценариев {lacking}")
    _write_csv(dest / F.SENSITIVITY, assemble.sensitivity_fields(sens), sens)

    # source_manifest.json ----------------------------------------------------------------------
    pricing = yaml.safe_load((CONFIGS_DIR / "pricing.yaml").read_text(encoding="utf-8"))
    s1_manifest = _read_json(data_dir() / "interim" / "source_manifest_s1.json")
    manifest = assemble.source_manifest(s1_manifest, chip_id, label_read=False,
                                        extra_sources=[])
    s1_entry = next(e for e in manifest["sources"] if e.get("chip_id") == chip_id)
    event_date = s1_entry.get("observation_date")
    if not event_date:
        raise ValueError(f"в source_manifest_s1.json у {chip_id} нет observation_date (дата события)")
    manifest["sources"] += assemble.config_sources(pricing, str(event_date))
    _write_json(dest / F.SOURCE_MANIFEST, manifest)

    # metrics.csv -------------------------------------------------------------------------------
    metrics_files = _metrics(prediction, dest, warnings)

    # run_metadata.json -------------------------------------------------------------------------
    rc = load_run_config()
    run_config = rc.model_dump(mode="json") if rc.chip_id == chip_id else {}
    event_id = s1_entry.get("event_id") or (rc.event_id if rc.chip_id == chip_id
                                            else chip_id.rsplit("_", 1)[0])
    split = _read_json(SPLIT_FILE)
    baseline_config = _read_json(BASELINE_CONFIG)
    main_cfg_path = MODELS_DIR / str(prediction.get("method", "")) / "config.json"
    main_config = (_read_json(main_cfg_path) if main_cfg_path.is_file()
                   and not assemble.is_baseline_prediction(prediction) else None)
    budget_arg = assemble.budget_label(budget)
    reproduce = [
        f"git checkout {commit}",
        "make setup",
        "make data",
        f"make all CHIP={chip_id} BUDGET={budget_arg}",
        f"make validate OUT={dest}",
        ("# по этапам: make predict CHIP=… (пока нет floodvalue.model.predict — "
         "uv run --frozen python -m floodvalue.baseline.predict --chip …), make assets CHIP=…, "
         "make procure CHIP=… BUDGET=…, make export CHIP=… BUDGET=…"),
    ]
    meta = assemble.run_metadata(
        run_id=run_id, chip_id=chip_id, event_id=event_id, event_date=str(event_date),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"), commit=commit,
        commit_dirty=dirty, prediction=prediction, baseline_config=baseline_config,
        baseline_config_path=str(BASELINE_CONFIG.relative_to(REPO_ROOT)),
        baseline_config_sha256=_sha256(BASELINE_CONFIG), main_config=main_config,
        assets_meta=assets_meta,
        procurement_meta=proc_meta, pricing=pricing, split=split,
        split_file=str(SPLIT_FILE.relative_to(REPO_ROOT)), run_config=run_config,
        uncertainty_raster="uncertainty.tif" if (dest / "uncertainty.tif").is_file() else None,
        metrics_files=metrics_files, reproduce=reproduce, versions=package_versions(),
        warnings=warnings)
    _write_json(dest / F.RUN_METADATA, meta)
    return dest


def run(args) -> int:
    rc = load_run_config()
    chip_id = getattr(args, "chip", None) or rc.chip_id
    work = _work_dir(args, chip_id)
    dest = Path(args.out) if getattr(args, "out", None) else None
    try:
        out = export(chip_id, work, dest, getattr(args, "budget", None))
    except (FileNotFoundError, ValueError) as exc:
        print(f"export: {exc}")
        return 1
    meta = _read_json(out / F.RUN_METADATA)
    for w in meta["warnings"]:
        print(f"export: ВНИМАНИЕ: {w}")
    print(f"export: комплект {meta['run_id']} -> {out}")
    return 0
