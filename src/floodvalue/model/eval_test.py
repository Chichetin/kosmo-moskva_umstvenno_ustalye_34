"""`python -m floodvalue eval-test --part <dev|validation|test>`: final comparison report (#29).

One command per split part. For every chip of the part:
  main method   `python -m floodvalue predict` in-process -> flood_probability.tif, flood_mask.tif,
                prediction.json (exactly the delivered rasters, incl. calibration once #20
                wires it into predict);
  baseline      its contract §3 `predict_proba` + the threshold rule (`baseline_mask`);
  labels        LabelHand (+ JRCWaterHand for the diagnostic), test only after the gate;
  portfolio     `python -m floodvalue assets` on the same rasters (placement rule and p
                extraction of the delivered portfolio, D-005: through the CLI and files).
Both methods are scored on the identical valid pixels (S1 finite and label != -1): the
valid_mask_sha256 of each method and of the S1/label reference must coincide, otherwise the
run fails.

Test is sealed (D-010): `--part test` (the default of `make eval-test`) reads test chips only
when FLOODVALUE_EVAL_TEST_CONFIRM equals the sha256 of the frozen configuration and the
tracked code/configs are committed, and only for the final method (predict.DEFAULT_METHOD;
FLOODVALUE_METHOD naming a candidate is refused, D-025); every such run writes a row to
docs/verification/test-log.md. Outputs: docs/verification/eval-<part>*.{md,csv,json,png}, for
test — test-eval*.
"""

from __future__ import annotations

import csv
import datetime as dt
import importlib
import json
import os
import subprocess
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import rasterio

from floodvalue.data import io
from floodvalue.data.split import SPLIT_PATH, chip_ids, event_of
from floodvalue.model import metrics as M
from floodvalue.model import predict as main_predict
from floodvalue.model.eval import core as C
from floodvalue.paths import MODELS_DIR, REPO_ROOT

PARTS = ("dev", "validation", "test")
CONFIRM_ENV = "FLOODVALUE_EVAL_TEST_CONFIRM"
VERIFICATION = REPO_ROOT / "docs" / "verification"
TEST_LOG = VERIFICATION / "test-log.md"
# Demo chip of the part: the proxy check runs its 10 portfolio points there (task.md).
PROXY_CHIPS = {"validation": "Sri-Lanka_713926", "test": "Spain_7370579"}
# Tracked paths that must be committed before test is read (the frozen configuration).
FROZEN_PATHS = ("src", "models", "configs", "splits", "pyproject.toml", "uv.lock")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260925
MAX_ERROR_MAPS = 4
WATER_SHARE_GROUPS = ((0.0, 0.01, "< 1 %"), (0.01, 0.10, "1–10 %"), (0.10, 0.30, "10–30 %"),
                      (0.30, 1.01, "≥ 30 %"))
PROXY_NOTE = "прокси по ручной метке воды, не фактические потери"


def main_method() -> str:
    return os.environ.get("FLOODVALUE_METHOD", main_predict.DEFAULT_METHOD)


def config_files(method: str, root: Path = REPO_ROOT) -> list[str]:
    """Files whose bytes define the evaluated configuration: every file of `models/<method>/`
    and `models/baseline/` (config, weights, `calibration.json` of #20 and anything added
    later), `models/calibration/` if it appears, and the split file."""
    root = Path(root)
    files = [SPLIT_PATH.relative_to(REPO_ROOT).as_posix(), f"models/{method}/config.json",
             "models/baseline/config.json"]
    for d in {f"models/{method}", "models/baseline", "models/calibration"}:
        folder = root / d
        if folder.is_dir():
            files += [q.relative_to(root).as_posix() for q in sorted(folder.rglob("*")) if q.is_file()]
    return sorted(set(files))


def frozen_config_hash(root: Path = REPO_ROOT, method: str | None = None) -> str:
    method = method or main_method()
    return C.config_sha256(root, config_files(method, root), method)


def git(*args: str, root: Path = REPO_ROOT) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                           check=False)
    except OSError as exc:
        return 1, str(exc)
    return r.returncode, (r.stdout.strip() if r.returncode == 0 else r.stderr.strip())


def uncommitted_frozen_paths(root: Path = REPO_ROOT, method: str | None = None) -> str:
    """Why the configuration is not reproducible from the commit ("" = it is): changed or
    untracked (not ignored) files under FROZEN_PATHS, configuration files git does not track,
    or git itself failing — the gate refuses rather than passes on a failure."""
    rc, status = git("status", "--porcelain", "--", *FROZEN_PATHS, root=root)
    if rc:
        return f"git status failed: {status}"
    if status:
        return status
    problems = []
    for rel in config_files(method or main_method(), root):
        if not (Path(root) / rel).is_file():
            continue
        rc, _ = git("ls-files", "--error-unmatch", "--", rel, root=root)
        if rc:
            problems.append(f"?? {rel} (not tracked by git)")
    return "\n".join(problems)


def who() -> str:
    rc, name = git("config", "user.name")
    return os.environ.get("FLOODVALUE_EVAL_WHO") or (name if not rc else "") or \
        os.environ.get("USER", "unknown")


def open_test_gate() -> tuple[str, str]:
    """Raise EvalGateError unless test may be read now; returns (config hash, commit).
    The method check comes first, before the hash: test opens only for the final method."""
    method = main_method()
    if method != main_predict.DEFAULT_METHOD:
        raise C.EvalGateError(
            f"test открывается только для итогового метода {main_predict.DEFAULT_METHOD!r}; "
            f"FLOODVALUE_METHOD={method!r} — кандидат, его сравнение только на dev/validation "
            "(D-025, D-040). Уберите FLOODVALUE_METHOD или задайте итоговый метод.")
    expected = frozen_config_hash()
    C.check_test_confirmation(os.environ.get(CONFIRM_ENV), expected)
    dirty = uncommitted_frozen_paths()
    if dirty:
        raise C.EvalGateError("test needs a committed configuration; uncommitted changes:\n"
                              + dirty)
    rc, commit = git("rev-parse", "HEAD")
    if rc or not commit:
        raise C.EvalGateError(f"no git commit to record in test-log: {commit}")
    return expected, commit


def append_test_log(row: str, path: Path | None = None) -> None:
    path = path or TEST_LOG
    text = path.read_text(encoding="utf-8")
    path.write_text(text.rstrip("\n") + "\n" + row + "\n", encoding="utf-8")


def replace_test_log(old: str, new: str, path: Path | None = None) -> None:
    path = path or TEST_LOG
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# --- per-chip inputs ------------------------------------------------------------------------

def read_layer(chip: str, layer: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(io.layer_path(chip, layer)) as src:
        return src.read(), dict(src.profile)


def predict_main(chip: str, out: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the delivered `predict` stage and read its files back: (p with NaN, mask bool, meta)."""
    rc = main_predict.run(Namespace(chip=chip, out=str(out)))
    if rc:
        raise RuntimeError(f"predict failed for {chip} (exit {rc})")
    with rasterio.open(out / "flood_probability.tif") as src:
        p = src.read(1).astype(np.float32)
        nodata = src.nodata
    p[(p == nodata) | ~np.isfinite(p)] = np.nan
    with rasterio.open(out / "flood_mask.tif") as src:
        mask = src.read(1)
    meta = json.loads((out / "prediction.json").read_text(encoding="utf-8"))
    finite = np.isfinite(p)
    expect = np.where(finite, np.nan_to_num(p) >= float(meta["threshold"]), False)
    if not np.array_equal(mask == 1, expect):
        raise RuntimeError(f"{chip}: flood_mask.tif != (p >= threshold)")
    return p, mask == 1, meta


def predict_baseline(s1: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Baseline threshold rule through its public module: (finite bool, mask bool, config)."""
    module = importlib.import_module(main_predict.METHODS["baseline"])
    config = module.load_config()
    r = module.to_rasters(s1, config)
    finite = r["flood_probability"] != module.P_NODATA
    return finite, (r["baseline_mask"] == 1) & finite, config


def proxy_points(chip: str, out: Path, label: np.ndarray, transform) -> list[dict]:
    """10 portfolio points by the delivered `assets` stage on the main p of this chip."""
    from floodvalue import cli

    rc = cli.main(["assets", "--chip", chip, "--out", str(out)])
    if rc:
        raise RuntimeError(f"assets failed for {chip} (exit {rc})")
    with open(out / "assets.csv", encoding="utf-8", newline="") as f:
        assets = {r["asset_id"]: r for r in csv.DictReader(f)}
    with open(out / "asset_loss.csv", encoding="utf-8", newline="") as f:
        losses = list(csv.DictReader(f))
    points = []
    for r in losses:
        a = assets[r["asset_id"]]
        row, col = C.lonlat_to_rowcol(transform, float(a["longitude"]), float(a["latitude"]))
        points.append({"chip_id": chip, "asset_id": r["asset_id"],
                       "asset_class": a["asset_class"], "V": float(a["asset_value_rub"]),
                       "q": float(a["vulnerability_coef"]), "row": row, "col": col,
                       "p": float(r["p_flood"]) if r["p_flood"] else None,
                       "el": float(r["expected_loss_rub"]) if r["expected_loss_rub"] else None,
                       "y": int(label[row, col])})
    return points


# --- evaluation -----------------------------------------------------------------------------

def _zero() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def evaluate(part: str, chips: list[str], workdir: Path, *, allow_test: bool = False,
             proxy_chip: str | None = None) -> dict:
    """All numbers of the report for one part; chips in chip_id order."""
    chips = sorted(chips)
    proxy_chip = proxy_chip if proxy_chip in chips else chips[0]
    per_chip, masks, rel, unc_hist = [], {}, {}, {}
    jrc = {m: {"jrc_1": _zero(), "jrc_0": _zero()} for m in ("main", "baseline")}
    proxy_all, proxy_demo, main_meta, base_cfg = [], [], None, None
    edges = np.linspace(0.0, 1.0, 100_001)
    for chip in chips:
        event = event_of(chip)
        s1, s1_profile = read_layer(chip, io.S1_LAYER)
        s1 = s1.astype(np.float32)
        label, _ = io.load_label(chip, allow_test=allow_test)
        jrc_layer, _ = read_layer(chip, io.JRC_LAYER)
        out = workdir / chip
        p, mask_main, main_meta = predict_main(chip, out)
        base_finite, mask_base, base_cfg = predict_baseline(s1)
        ref_valid = io.valid_mask(s1, label)
        valid_main = np.isfinite(p) & (label != -1)
        valid_base = base_finite & (label != -1)
        for key, v in (("main", valid_main), ("baseline", valid_base), ("reference", ref_valid)):
            masks.setdefault(key, {})[chip] = v
        valid = ref_valid
        target = label == 1
        c_main = M.confusion(mask_main, target, valid)
        c_base = M.confusion(mask_base, target, valid)
        for m, pred in (("main", mask_main), ("baseline", mask_base)):
            for k, c in C.jrc_confusions(pred, target, valid, jrc_layer[0]).items():
                jrc[m][k] = M.add(jrc[m][k], c)
        pv, yv = p[valid], target[valid]
        counts = C.reliability_counts(pv, yv)
        u = C.binary_entropy(pv)
        err = mask_main[valid] != yv
        h_err = np.histogram(u[err], bins=edges)[0]
        h_ok = np.histogram(u[~err], bins=edges)[0]
        for scope in (event, "ALL"):
            rel[scope] = C.add_reliability(rel.get(scope), counts)
            acc = unc_hist.setdefault(scope, {"err": 0, "ok": 0, "sum_err": 0.0, "sum_ok": 0.0})
            acc["err"] = acc["err"] + h_err
            acc["ok"] = acc["ok"] + h_ok
            acc["sum_err"] += float(u[err].sum())
            acc["sum_ok"] += float(u[~err].sum())
        n_valid = int(valid.sum())
        per_chip.append({
            "chip_id": chip, "event": event, "n_pixels": n_valid,
            "water_share": float(yv.mean()) if n_valid else float("nan"),
            "label_invalid_share": float((label == -1).mean()),
            "s1_nodata_share": float((~np.isfinite(s1).all(axis=0)).mean()),
            "main": c_main, "baseline": c_base})
        points = proxy_points(chip, out, label, s1_profile["transform"])
        proxy_all += points
        if chip == proxy_chip:
            proxy_demo = points
    # identical valid pixels, per event and for the whole part
    hashes: dict[str, str] = {}
    events = sorted({c["event"] for c in per_chip})
    for scope in [*events, "ALL"]:
        scope_chips = [c for c in chips if scope == "ALL" or event_of(c) == scope]
        h = {k: M.mask_sha256([masks[k][c] for c in scope_chips]) for k in masks}
        hashes[scope] = C.require_same_valid_pixels(h)
    return {"part": part, "chips": chips, "per_chip": per_chip, "events": events,
            "hashes": hashes, "reliability": rel, "uncertainty": unc_hist, "edges": edges,
            "jrc": jrc, "proxy_chip": proxy_chip,
            "proxy_demo": C.proxy_rows(proxy_demo), "proxy_all": C.proxy_rows(proxy_all),
            "main_meta": main_meta, "baseline_config": base_cfg}


def metric_rows(res: dict) -> list[dict]:
    """§7 rows: baseline and main per event and ALL, same valid_mask_sha256 per pair."""
    rows = []
    thresholds = {"main": float(res["main_meta"]["threshold"]),
                  "baseline": float(res["baseline_config"]["threshold_db"])}
    for scope in [*res["events"], "ALL"]:
        for method in ("baseline", "main"):
            total = _zero()
            for c in res["per_chip"]:
                if scope in ("ALL", c["event"]):
                    total = M.add(total, c[method])
            s = M.scores(total)
            rows.append({"method": method, "split": res["part"], "event": scope,
                         "iou": s["iou"], "f1": s["f1"], "precision": s["precision"],
                         "recall": s["recall"], **total, "n_pixels": s["n_pixels"],
                         "threshold": thresholds[method],
                         "valid_mask_sha256": res["hashes"][scope]})
    return rows


def bootstraps(res: dict) -> dict:
    out = {}
    for scope in [*res["events"], "ALL"]:
        chips = [c for c in res["per_chip"] if scope in ("ALL", c["event"])]
        out[scope] = C.paired_bootstrap_delta_iou([c["main"] for c in chips],
                                                  [c["baseline"] for c in chips],
                                                  n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    return out


def uncertainty_summary(res: dict) -> dict:
    out = {}
    for scope, acc in res["uncertainty"].items():
        n_err, n_ok = int(np.sum(acc["err"])), int(np.sum(acc["ok"]))
        out[scope] = {"n_error": n_err, "n_correct": n_ok,
                      "mean_u_error": acc["sum_err"] / n_err if n_err else float("nan"),
                      "mean_u_correct": acc["sum_ok"] / n_ok if n_ok else float("nan"),
                      "auroc": C.auroc_from_hist(acc["err"], acc["ok"])}
    return out


def transfer_groups(res: dict) -> list[dict]:
    """Pooled IoU of both methods by water share of the chip (conditions of degradation)."""
    rows = []
    for lo, hi, name in WATER_SHARE_GROUPS:
        chips = [c for c in res["per_chip"] if c["n_pixels"] and lo <= c["water_share"] < hi]
        rows.append({"group": f"доля воды {name}", "n_chips": len(chips),
                     "iou_main": C.pooled_iou(c["main"] for c in chips),
                     "iou_baseline": C.pooled_iou(c["baseline"] for c in chips),
                     "fp_main": sum(c["main"]["fp"] for c in chips),
                     "fn_main": sum(c["main"]["fn"] for c in chips)})
    for name, cond in (("невалидных пикселей > 10 %",
                        lambda c: c["label_invalid_share"] + c["s1_nodata_share"] > 0.10),
                       ("невалидных пикселей ≤ 10 %",
                        lambda c: c["label_invalid_share"] + c["s1_nodata_share"] <= 0.10)):
        chips = [c for c in res["per_chip"] if cond(c)]
        rows.append({"group": name, "n_chips": len(chips),
                     "iou_main": C.pooled_iou(c["main"] for c in chips),
                     "iou_baseline": C.pooled_iou(c["baseline"] for c in chips),
                     "fp_main": sum(c["main"]["fp"] for c in chips),
                     "fn_main": sum(c["main"]["fn"] for c in chips)})
    return rows


# --- outputs --------------------------------------------------------------------------------

def prefix_of(part: str) -> str:
    return "test-eval" if part == "test" else f"eval-{part}"


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float | np.floating):
        return "" if not np.isfinite(v) else repr(round(float(v), 6))
    return str(v)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([_cell(r.get(k)) for k in fields])


def f3(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.3f}".replace(".", ",").replace("-", "−")


def rub(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:,.0f}".replace(",", " ")


def error_map_chips(res: dict) -> list[str]:
    """Demo chip + the chip with most main-method errors of each event, at most 4."""
    picked = [res["proxy_chip"]]
    for event in res["events"]:
        chips = [c for c in res["per_chip"] if c["event"] == event]
        worst = max(chips, key=lambda c: c["main"]["fp"] + c["main"]["fn"])["chip_id"]
        if worst not in picked:
            picked.append(worst)
    return picked[:MAX_ERROR_MAPS]


def draw_error_maps(res: dict, workdir: Path, path: Path, allow_test: bool) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    chips = error_map_chips(res)
    # 0 invalid, 1 TN, 2 TP, 3 FP, 4 FN
    cmap = ListedColormap(["#9e9e9e", "#ffffff", "#1f77b4", "#d62728", "#ff9f1c"])
    fig, axes = plt.subplots(len(chips), 3, figsize=(10.5, 3.5 * len(chips)), squeeze=False)
    for ax_row, chip in zip(axes, chips):
        s1, _ = read_layer(chip, io.S1_LAYER)
        label, _ = io.load_label(chip, allow_test=allow_test)
        _, mask_main, _ = predict_main(chip, workdir / f"map_{chip}")
        _, mask_base, _ = predict_baseline(s1.astype(np.float32))
        valid = io.valid_mask(s1, label)
        t = label == 1
        ax_row[0].imshow(s1[1], cmap="gray", vmin=-30, vmax=0)
        ax_row[0].set_title(f"{chip}: VH, dB", fontsize=9)
        for ax, pred, name in ((ax_row[1], mask_main, "основной"),
                               (ax_row[2], mask_base, "baseline")):
            code = np.where(~valid, 0, np.select([pred & t, pred & ~t, ~pred & t], [2, 3, 4], 1))
            ax.imshow(code, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
            c = M.confusion(pred, t, valid)
            ax.set_title(f"{name}: FP {c['fp']}, FN {c['fn']}", fontsize=9)
        for ax in ax_row:
            ax.set_axis_off()
    fig.suptitle("синий — TP, красный — FP, оранжевый — FN, белый — TN, серый — невалидно",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return chips


def baseline_crosscheck(rows: list[dict], part: str) -> str:
    """Baseline rows vs docs/verification/baseline-metrics.csv (written by `make baseline`)."""
    ref_path = VERIFICATION / "baseline-metrics.csv"
    if part == "test" or not ref_path.exists():
        return "не применимо"
    with open(ref_path, encoding="utf-8", newline="") as f:
        ref = {(r["split"], r["event"]): r for r in csv.DictReader(f) if r["method"] == "baseline"}
    bad = []
    for r in rows:
        if r["method"] != "baseline":
            continue
        e = ref.get((part, r["event"]))
        if e is None or any(int(e[k]) != r[k] for k in ("tp", "fp", "fn", "tn")) \
                or e["valid_mask_sha256"] != r["valid_mask_sha256"]:
            bad.append(r["event"])
    return "совпадает (TP/FP/FN/TN и valid_mask_sha256)" if not bad else \
        f"НЕ совпадает: {', '.join(bad)}"


def write_report(res: dict, out_dir: Path, workdir: Path, allow_test: bool, commit: str,
                 config_hash: str) -> dict[str, Path]:
    part, prefix = res["part"], prefix_of(res["part"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{prefix}{suffix}" for k, suffix in (
        ("md", ".md"), ("csv", ".csv"), ("chips", "-chips.csv"),
        ("reliability", "-reliability.csv"), ("proxy", "-proxy.csv"), ("json", "-summary.json"),
        ("png", "-errors.png"))}
    rows = metric_rows(res)
    write_csv(paths["csv"], M.METRIC_COLUMNS, rows)
    chip_rows = []
    for c in res["per_chip"]:
        r = {k: c[k] for k in ("chip_id", "event", "n_pixels", "water_share",
                               "label_invalid_share", "s1_nodata_share")}
        for m in ("main", "baseline"):
            r[f"iou_{m}"] = M.scores(c[m])["iou"]
            r.update({f"{k}_{m}": c[m][k] for k in ("tp", "fp", "fn", "tn")})
        chip_rows.append(r)
    write_csv(paths["chips"], list(chip_rows[0]), chip_rows)
    rel_rows = []
    for scope in [*res["events"], "ALL"]:
        for r in C.reliability_table(res["reliability"][scope]):
            rel_rows.append({"scope": scope, **r})
    write_csv(paths["reliability"], ["scope", "bin", "lo", "hi", "n", "mean_p", "observed"],
              rel_rows)
    proxy_fields = ["chip_id", "asset_id", "asset_class", "V", "q", "row", "col", "p", "y",
                    "el", "el_ref", "abs_error", "included", "reason"]
    write_csv(paths["proxy"], proxy_fields, res["proxy_all"])
    maps = draw_error_maps(res, workdir, paths["png"], allow_test)

    boot = bootstraps(res)
    unc = uncertainty_summary(res)
    cal = {s: {"ece": C.ece(res["reliability"][s]), "brier": C.brier(res["reliability"][s])}
           for s in [*res["events"], "ALL"]}
    demo, allp = C.proxy_summary(res["proxy_demo"]), C.proxy_summary(res["proxy_all"])
    jrc = {m: {k: M.scores(c)["iou"] for k, c in v.items()} for m, v in res["jrc"].items()}
    jrc_counts = res["jrc"]
    groups = transfer_groups(res)
    meta = res["main_meta"]
    calibrated = not str(meta.get("threshold_status", "")).startswith("uncalibrated")
    proxy_check = {
        "status": "done" if part == "test" else "planned",
        "test_chip": res["proxy_chip"],
        "control_points": "10 точек портфеля тем же правилом `python -m floodvalue assets` "
                          "(seed 20260925, только валидность S1; прогноз и метки не читаются); "
                          "оцениваются точки со статусом ok и меткой 0/1",
        "fixed_v_q": "configs/assets.yaml (учебные V и q, те же, что в asset_loss.csv)",
        "n_points": demo["n_points"],
        "error_metric": "MAE и сумма |EL − EL_ref|, руб.; EL = p·V·q, EL_ref = y·V·q",
        "result": {"demo_chip": demo, "all_chips_of_part": allp, "note": PROXY_NOTE,
                   "independent": part == "test"}}
    summary = {"part": part, "commit": commit, "config_sha256": config_hash,
               "main_method": meta.get("method"), "threshold": meta.get("threshold"),
               "threshold_status": meta.get("threshold_status"), "n_chips": len(res["chips"]),
               "valid_mask_sha256": res["hashes"], "bootstrap_delta_iou": boot,
               "calibration": cal, "uncertainty": unc, "jrc_iou": jrc,
               "proxy_check": proxy_check, "error_maps": maps,
               "baseline_crosscheck": baseline_crosscheck(rows, part)}
    paths["json"].write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float)
                             + "\n", encoding="utf-8")
    paths["md"].write_text(render_md(res, rows, summary, groups, jrc_counts, calibrated, paths),
                           encoding="utf-8")
    return paths


def render_md(res, rows, s, groups, jrc_counts, calibrated, paths) -> str:
    part, meta = res["part"], res["main_meta"]
    by = {(r["method"], r["event"]): r for r in rows}
    L = [f"# Итоговая оценка на части `{part}`: baseline против основного метода", "",
         f"Сгенерировано `python -m floodvalue eval-test --part {part}` "
         f"(commit `{s['commit'][:12]}`, {dt.date.today().isoformat()}), sha256 конфигурации "
         f"`{s['config_sha256']}`. Числа — из `{paths['csv'].name}`, `{paths['chips'].name}`, "
         f"`{paths['reliability'].name}`, `{paths['proxy'].name}`, `{paths['json'].name}`.", ""]
    if part == "test":
        L += ["Test прочитан один раз замороженной конфигурацией; запуск записан в "
              "`docs/verification/test-log.md`.", ""]
    else:
        L += [f"**Это не итоговая независимая проверка.** Часть `{part}` использовалась для "
              "настройки (D-012/D-016: dev — выбор модели, validation — калибровка и порог), "
              "поэтому прогон проверяет проводку отчёта и даёт ориентир, но не оценку переноса. "
              "Test не читался (D-010).", ""]
    L += [f"- Основной метод: `{meta.get('method')}`, порог p ≥ {meta.get('threshold')} "
          f"(`{meta.get('threshold_status')}`). Растры — те же, что пишет `make predict`.",
          f"- Калибровка: {'применена в predict' if calibrated else 'не применена — оценивается сырая p (калибратор #20 ещё не влит)'}.",
          f"- Baseline: VH ≤ {res['baseline_config']['threshold_db']} dB (`models/baseline/config.json`).",
          f"- Чипов: {len(res['chips'])}; события: {', '.join(res['events'])}.",
          "- Валидный пиксель: VV и VH конечны **и** LabelHand ≠ −1. Маска валидности "
          "считается отдельно по растру каждого метода и по S1/метке; `valid_mask_sha256` "
          "всех трёх совпал по каждому событию и ALL (иначе запуск падает).",
          f"- Сверка baseline с `baseline-metrics.csv`: {s['baseline_crosscheck']}.", "",
          "## 1. Метрики на идентичных валидных пикселях (критерий 4)", "",
          "| событие | метод | IoU | F1 | Precision | Recall | TP | FP | FN | TN | пикселей |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for scope in [*res["events"], "ALL"]:
        for m in ("baseline", "main"):
            r = by[(m, scope)]
            L.append(f"| {'**ALL**' if scope == 'ALL' else scope} | "
                     f"{'основной' if m == 'main' else 'baseline'} | {f3(r['iou'])} | "
                     f"{f3(r['f1'])} | {f3(r['precision'])} | {f3(r['recall'])} | {r['tp']} | "
                     f"{r['fp']} | {r['fn']} | {r['tn']} | {r['n_pixels']} |")
    L += ["", f"Парный бутстрэп по чипам ({BOOTSTRAP_N} повторов, seed {BOOTSTRAP_SEED}; "
          "одни и те же выборки чипов для обоих методов), ΔIoU = основной − baseline:", "",
          "| событие | ΔIoU | 95 % CI | доля Δ > 0 |", "|---|---|---|---|"]
    note_boot = ("Интервал считается здесь заново (свои выборки чипов), поэтому может отличаться "
                 "в третьем знаке от других отчётов; для этой части основным считается этот.")
    for scope, b in s["bootstrap_delta_iou"].items():
        L.append(f"| {scope} | {f3(b['delta'])} | [{f3(b['ci95'][0])}; {f3(b['ci95'][1])}] | "
                 f"{f3(b['share_gt_0'])} |")
    L += ["", note_boot, "", "## 2. Ложные срабатывания и пропуски по событиям", "",
          "| событие | метод | FP | FN | FP / (TP+FP) | FN / (TP+FN) |", "|---|---|---|---|---|---|"]
    for scope in [*res["events"], "ALL"]:
        for m in ("baseline", "main"):
            r = by[(m, scope)]
            fdr = r["fp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else float("nan")
            fnr = r["fn"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else float("nan")
            L.append(f"| {scope} | {'основной' if m == 'main' else 'baseline'} | {r['fp']} | "
                     f"{r['fn']} | {f3(fdr)} | {f3(fnr)} |")
    L += ["", f"Карты ошибок (`{paths['png'].name}`): {', '.join(s['error_maps'])} — "
          "демо-чип части и чип с наибольшим числом ошибок основного метода в каждом событии.",
          "", f"![карты ошибок]({paths['png'].name})", "",
          "## 3. Надёжность вероятностей основного метода (критерий 9)", "",
          "10 равных корзин p на [0; 1], по всем валидным пикселям. ECE = Σ n_b/N · "
          "|среднее p − доля воды|; Brier = среднее (p − y)².", "",
          "| событие | ECE | Brier |", "|---|---|---|"]
    for scope, c in s["calibration"].items():
        L.append(f"| {scope} | {f3(c['ece'])} | {f3(c['brier'])} |")
    L += ["", "| корзина p | пикселей | среднее p | доля воды |", "|---|---|---|---|"]
    for r in C.reliability_table(res["reliability"]["ALL"]):
        close = "]" if r["hi"] >= 1 else ")"
        lo, hi = (f"{x:.1f}".replace(".", ",") for x in (r["lo"], r["hi"]))
        L.append(f"| [{lo}; {hi}{close} | {r['n']} | {f3(r['mean_p'])} | "
                 f"{f3(r['observed'])} |")
    L += ["", "## 4. Неопределённость на ошибках (критерий 10)", "",
          "Показатель (временный, до #21) — бинарная энтропия p основного метода в битах: "
          "нерешительность модели в пикселе, не проверенная неопределённость. Ошибка — маска "
          "основного метода ≠ метке. AUROC — вероятность, что у случайного ошибочного пикселя "
          "энтропия выше, чем у случайного верного (0,5 — не различает; по гистограмме с шагом "
          "1e-5 бита).", "",
          "| событие | ошибочных | верных | средняя H на ошибках | на верных | AUROC H → ошибка |",
          "|---|---|---|---|---|---|"]
    for scope in [*res["events"], "ALL"]:
        u = s["uncertainty"][scope]
        L.append(f"| {scope} | {u['n_error']} | {u['n_correct']} | {f3(u['mean_u_error'])} | "
                 f"{f3(u['mean_u_correct'])} | {f3(u['auroc'])} |")
    demo = s["proxy_check"]["result"]["demo_chip"]
    allp = s["proxy_check"]["result"]["all_chips_of_part"]
    L += ["", "## 5. Прокси-проверка ущерба (критерий 14) — прокси, не фактические потери", "",
          "Правило точек: `python -m floodvalue assets` на растре p основного метода этого "
          "чипа — то же правило размещения, что у сдаваемого портфеля (seed 20260925, 10 типов "
          "из `configs/assets.yaml`, только валидность S1; прогноз, ущерб и метки не "
          "читаются). EL = p·V·q из `asset_loss.csv`; EL_ref = y·V·q, y — ручная метка "
          "воды в пикселе точки при тех же учебных V и q. Оцениваются только точки со "
          "статусом ok и меткой 0/1.", "",
          "| выборка | точек поставлено | оценено | исключено: метка −1 | исключено: нет p | "
          "MAE, руб. | Σ \\|EL − EL_ref\\|, руб. | Σ EL, руб. | Σ EL_ref, руб. |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| демо-чип `{res['proxy_chip']}` | {demo['n_placed']} | {demo['n_points']} | "
          f"{demo['n_excluded_label_invalid']} | {demo['n_excluded_no_data']} | "
          f"{rub(demo['mae_rub'])} | "
          f"{rub(demo['sum_abs_error_rub'])} | {rub(demo['sum_el_rub'])} | "
          f"{rub(demo['sum_el_ref_rub'])} |",
          f"| все чипы части, по 10 точек | {allp['n_placed']} | {allp['n_points']} | "
          f"{allp['n_excluded_label_invalid']} | {allp['n_excluded_no_data']} | "
          f"{rub(allp['mae_rub'])} | "
          f"{rub(allp['sum_abs_error_rub'])} | {rub(allp['sum_el_rub'])} | "
          f"{rub(allp['sum_el_ref_rub'])} |", "",
          f"Ошибка — {PROXY_NOTE}: EL_ref сам зависит от учебных V, q и не наблюдает ущерб. "
          f"По точкам — `{paths['proxy'].name}`.", "",
          "## 6. JRC-диагностика: IoU внутри и вне JRCWaterHand = 1", "",
          "JRCWaterHand — слой воды набора; вероятно, собран по данным после события, поэтому "
          "только диагностика (в метод не подаётся). Внутри JRC = 1 — в основном постоянная "
          "вода, вне — то, что ближе к паводку.", "",
          "| метод | IoU внутри JRC=1 | IoU вне JRC | пикселей внутри | пикселей вне |",
          "|---|---|---|---|---|"]
    for m in ("baseline", "main"):
        c1, c0 = jrc_counts[m]["jrc_1"], jrc_counts[m]["jrc_0"]
        L.append(f"| {'основной' if m == 'main' else 'baseline'} | {f3(s['jrc_iou'][m]['jrc_1'])}"
                 f" | {f3(s['jrc_iou'][m]['jrc_0'])} | {sum(c1.values())} | "
                 f"{sum(c0.values())} |")
    L += ["", "## 7. Перенос: каждое событие части — новое событие (критерий 7)", "",
          "Ни одно событие части не участвовало в обучении (split по событиям, D-016). "
          "Метрики по событиям — в разделе 1; ниже — условия, при которых качество падает "
          f"(пулинг по чипам, построчно — `{paths['chips'].name}`). Метод работает только по "
          "S1: S2, DEM, JRC не подаются.", "",
          "| группа чипов | чипов | IoU основной | IoU baseline | FP основной | FN основной |",
          "|---|---|---|---|---|---|"]
    for g in groups:
        L.append(f"| {g['group']} | {g['n_chips']} | {f3(g['iou_main'])} | "
                 f"{f3(g['iou_baseline'])} | {g['fp_main']} | {g['fn_main']} |")
    L += ["", "## Воспроизведение", "", "```bash", "set -a; . ./.env; set +a",
          f"uv run --frozen python -m floodvalue eval-test --part {part}",
          "uv run --frozen pytest tests/eval", "```", ""]
    return "\n".join(L)


def run(args) -> int:
    part = getattr(args, "part", None) or "test"
    if part not in PARTS:
        print(f"eval-test: unknown part {part!r}; expected one of {PARTS}")
        return 2
    out_dir = Path(args.out) if getattr(args, "out", None) else VERIFICATION
    rc, commit = git("rev-parse", "HEAD")
    commit = commit if not rc and commit else "unknown"
    log_row = None
    if part == "test":
        try:
            config_hash, commit = open_test_gate()
        except C.EvalGateError as exc:
            print(f"eval-test: отказ — {exc}")
            return 3
        when = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        methods = f"baseline, {main_method()}"
        log_row = C.log_row(when, commit, config_hash, methods, who(), "начат")
        append_test_log(log_row)
        chips = chip_ids("test", allow_test=True)
    else:
        config_hash = frozen_config_hash()
        chips = chip_ids(part)
    allow_test = part == "test"
    with tempfile.TemporaryDirectory(prefix="floodvalue-eval-") as tmp:
        try:
            res = evaluate(part, chips, Path(tmp), allow_test=allow_test,
                           proxy_chip=PROXY_CHIPS.get(part))
            paths = write_report(res, out_dir, Path(tmp), allow_test, commit, config_hash)
        except Exception as exc:
            if log_row:
                replace_test_log(log_row, log_row.replace(
                    "| начат |", f"| прерван: {type(exc).__name__} |"))
            raise
    if log_row:
        all_row = next(r for r in metric_rows(res) if r["method"] == "main"
                       and r["event"] == "ALL")
        result = f"IoU основного (ALL) {all_row['iou']:.3f} → `{paths['md'].name}`"
        replace_test_log(log_row, log_row.replace("| начат |", f"| {result} |"))
    print(f"eval-test {part}: {len(chips)} чипов -> {paths['md']}")
    return 0
