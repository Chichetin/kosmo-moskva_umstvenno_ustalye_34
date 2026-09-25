"""`python -m floodvalue.calibration.uncertainty_val [--assets-run <dir>]`: choose and verify the
per-pixel uncertainty measure (issue #21, criterion 10, D-033). Test chips are never read.

Protocol (fixed before the numbers, D-033):
  * candidates: `floodvalue.calibration.uncertainty.MEASURES` on the delivered pipeline —
    members of `predict_members` (or the single p of `predict_proba`), calibrator and threshold
    of `models/<method>/calibration.json`;
  * error = (p >= threshold) != LabelHand on valid pixels (S1 finite and label != -1);
  * selection on validation (Sri-Lanka, Nigeria): max worst-event AUROC(u -> error); a simpler
    measure wins when the chip-bootstrap 95 % interval of the difference contains 0;
  * dev (USA, not used for the selection and not for calibration) — transfer check only.
Writes:
  models/<method>/uncertainty.json            chosen measure + provenance (read by predict)
  docs/verification/uncertainty-val.md        report
  docs/verification/uncertainty/*.csv|png     per-scope table, risk by decile, 2 figures
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import os
from pathlib import Path

import numpy as np

from floodvalue.calibration import calibrator as C
from floodvalue.calibration import uncertainty as U
from floodvalue.calibration.run import git_commit, write_csv
from floodvalue.data import io
from floodvalue.data.split import chip_ids, load_split
from floodvalue.model.predict import METHODS as METHOD_MODULES
from floodvalue.paths import MODELS_DIR, REPO_ROOT

DEFAULT_METHOD = "unet"
DOC = REPO_ROOT / "docs" / "verification" / "uncertainty-val.md"
DOC_DIR = REPO_ROOT / "docs" / "verification" / "uncertainty"
DEMO_CHIP = "Sri-Lanka_713926"
N_EDGES = 10_000       # quantile bins of each measure for the chip-bootstrap histograms
TOP_SHARE = 0.10       # "flagged" = top 10 % of u on validation


def members_and_p(module, s1: np.ndarray, cal: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """(raw members (K, H, W), delivered calibrated p (H, W)) — the same calls as `predict`."""
    p_raw = module.predict_proba(s1)
    members = module.predict_members(s1) if hasattr(module, "predict_members") else p_raw[None]
    return members, (C.apply(cal, p_raw) if cal else p_raw)


def collect(method: str, part: str, cal: dict | None, threshold: float) -> dict:
    """Valid pixels of every chip of the part: measures, error flags, member range, chip index."""
    if part not in ("validation", "dev"):
        raise ValueError(f"uncertainty is studied on validation and dev only, not {part!r}")
    module = importlib.import_module(METHOD_MODULES[method])
    cols = {k: [] for k in (*U.MEASURES, "err", "fp", "fn", "unstable", "chip")}
    chips, events, maps = [], [], {}
    for i, cid in enumerate(chip_ids(part)):
        s1, _ = io.load_s1(cid)
        label, _ = io.load_label(cid)
        members, p = members_and_p(module, s1, cal)
        u = U.measures(members, cal, threshold, p)
        lo, hi = U.p_range(members, cal)
        valid = np.isfinite(p) & (label != -1)
        y = label[valid] == 1
        pred = p[valid] >= np.float32(threshold)
        for k in U.MEASURES:
            cols[k].append(u[k][valid])
        cols["err"].append(pred != y)
        cols["fp"].append(pred & ~y)
        cols["fn"].append(~pred & y)
        cols["unstable"].append((lo[valid] < threshold) & (hi[valid] >= threshold))
        cols["chip"].append(np.full(int(valid.sum()), i, np.int32))
        chips.append(cid)
        events.append(io.parse_chip_id(cid)[0])
        if cid == DEMO_CHIP:
            maps = {"p": p, "valid": valid, "label": label, "u": u, "lo": lo, "hi": hi}
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data.update(chips=chips, events=events, n_members=int(members.shape[0]), maps=maps)
    return data


def quantile_edges(values: np.ndarray, n: int = N_EDGES) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0, 1, n + 1)))


def chip_hists(u: np.ndarray, err: np.ndarray, chip: np.ndarray, n_chips: int,
               edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(errors, correct) histograms per chip, shape (n_chips, n_bins)."""
    b = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    key = chip.astype(np.int64) * nb + b
    h_err = np.bincount(key[err], minlength=n_chips * nb).reshape(n_chips, nb)
    h_ok = np.bincount(key[~err], minlength=n_chips * nb).reshape(n_chips, nb)
    return h_err, h_ok


def scope_stats(d: dict, sel: np.ndarray, flag_cut: dict[str, float]) -> dict:
    """Numbers of one scope (event or ALL) of one part for every measure."""
    err = d["err"][sel]
    out = {"n_pixels": int(sel.sum()), "n_error": int(err.sum()),
           "error_rate": float(err.mean()) if sel.any() else float("nan"),
           "unstable_share": float(d["unstable"][sel].mean()) if sel.any() else float("nan"),
           "error_rate_unstable": _rate(err, d["unstable"][sel]),
           "error_rate_stable": _rate(err, ~d["unstable"][sel])}
    for m in U.MEASURES:
        u = d[m][sel]
        edges = quantile_edges(u)
        h = lambda mask: np.histogram(u[mask], bins=edges)[0]  # noqa: E731
        flagged = u >= flag_cut[m]
        out[m] = {"mean_u_error": _mean(u, err), "mean_u_correct": _mean(u, ~err),
                  "mean_u_fp": _mean(u, d["fp"][sel]), "mean_u_fn": _mean(u, d["fn"][sel]),
                  "auroc": float(U.auroc_hist(h(err), h(~err))),
                  "auroc_unstable": float(U.auroc_hist(h(d["unstable"][sel]),
                                                       h(~d["unstable"][sel]))),
                  "flagged_share": float(flagged.mean()),
                  "errors_flagged_share": _rate(flagged, err),
                  "error_rate_flagged": _rate(err, flagged),
                  "fp_flagged_share": _rate(flagged, d["fp"][sel]),
                  "fn_flagged_share": _rate(flagged, d["fn"][sel])}
    return out


def _mean(u: np.ndarray, mask: np.ndarray) -> float:
    return float(u[mask].mean()) if mask.any() else float("nan")


def _rate(x: np.ndarray, mask: np.ndarray) -> float:
    return float(x[mask].mean()) if mask.any() else float("nan")


def deciles(u: np.ndarray, err: np.ndarray) -> list[dict]:
    """Error rate by decile of u (rank-based, ties broken by pixel order)."""
    order = np.argsort(u, kind="stable")
    rows = []
    for k, idx in enumerate(np.array_split(order, 10), start=1):
        rows.append({"decile": k, "u_min": float(u[idx].min()), "u_max": float(u[idx].max()),
                     "error_rate": float(err[idx].mean()), "errors_share": float(err[idx].sum() / max(err.sum(), 1))})
    return rows


def chip_spearman(d: dict, m: str) -> float:
    """Spearman correlation over chips of mean u and error rate (chip-level instability)."""
    n = len(d["chips"])
    cnt = np.bincount(d["chip"], minlength=n)
    ok = cnt > 0
    mu = np.bincount(d["chip"], weights=d[m], minlength=n)[ok] / cnt[ok]
    er = np.bincount(d["chip"], weights=d["err"], minlength=n)[ok] / cnt[ok]
    ra, rb = np.argsort(np.argsort(mu)), np.argsort(np.argsort(er))
    return float(np.corrcoef(ra, rb)[0, 1])


def study(val: dict, dev: dict) -> dict:
    """All numbers of the report; pure given the collected pixels."""
    n_chips = len(val["chips"])
    per_chip, edges = {}, {}
    for m in U.MEASURES:
        edges[m] = quantile_edges(val[m])
        he, ho = chip_hists(val[m], val["err"], val["chip"], n_chips, edges[m])
        per_chip[m] = [(val["events"][i], he[i], ho[i]) for i in range(n_chips)]
    sel = U.select(per_chip)
    flag_cut = {m: float(np.quantile(val[m], 1 - TOP_SHARE)) for m in U.MEASURES}
    parts = {}
    for name, d in (("validation", val), ("dev", dev)):
        ev = np.array(d["events"])[d["chip"]]
        scopes = {e: ev == e for e in sorted(set(d["events"]))}
        scopes["ALL"] = np.ones(d["err"].size, bool)
        parts[name] = {s: scope_stats(d, mask, flag_cut) for s, mask in scopes.items()}
        parts[name]["_spearman"] = {m: chip_spearman(d, m) for m in U.MEASURES}
        parts[name]["_deciles"] = {m: deciles(d[m], d["err"]) for m in U.MEASURES}
    return {"selection": sel, "flag_cut": flag_cut, "parts": parts}


def write_outputs(res: dict, method: str, cal_path: Path, cal: dict, config: dict,
                  val: dict, assets_run: Path | None) -> None:
    from floodvalue.calibration import uncertainty_report as R
    chosen = res["selection"]["chosen"]
    va = res["parts"]["validation"]
    info = {
        "method": method, "measure": chosen, "units": U.UNITS[chosen],
        "selection_rule": "max worst-event AUROC(u -> error) on validation (Sri-Lanka, Nigeria); "
                          "a simpler measure (order entropy_p, threshold_margin, member_std, "
                          "mutual_info) wins when the chip-bootstrap 95 % interval of the "
                          "difference contains 0; error = (p >= threshold) != LabelHand",
        "status": "связь с ошибками проверена на validation (часть подбора калибровки и порога) "
                  "и на dev (USA, не участвовал в выборе); на test — через eval-test",
        "validation": {s: {k: va[s][chosen][k] for k in ("auroc", "mean_u_error", "mean_u_correct")}
                       for s in va if not s.startswith("_")},
        "dev": {s: {k: res["parts"]["dev"][s][chosen][k] for k in ("auroc", "mean_u_error",
                                                                      "mean_u_correct")}
                for s in res["parts"]["dev"] if not s.startswith("_")},
        "flag_cut_top10_validation": res["flag_cut"][chosen],
        "p_range": "p_low/p_high = min/max over members of calibrated member p (seed spread, "
                   "not a confidence interval)",
        "input_model_sha256": config["weights_sha256"],
        "calibration_sha256": C.file_sha256(cal_path) if cal_path.exists() else None,
        "split_sha256": load_split()["sha256"],
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "commit": git_commit(), "report": "docs/verification/uncertainty-val.md",
    }
    (MODELS_DIR / method / "uncertainty.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for part, scopes in res["parts"].items():
        for s, st in scopes.items():
            if s.startswith("_"):
                continue
            for m in U.MEASURES:
                rows.append({"part": part, "scope": s, "measure": m, "n_pixels": st["n_pixels"],
                             "n_error": st["n_error"], **st[m]})
    write_csv(DOC_DIR / "uncertainty-scopes.csv", rows)
    write_csv(DOC_DIR / "uncertainty-deciles.csv",
              [{"part": part, "measure": m, **r} for part, scopes in res["parts"].items()
               for m in U.MEASURES for r in scopes["_deciles"][m]])
    assets = read_assets(assets_run) if assets_run else None
    R.figures(res, val, DOC_DIR, DEMO_CHIP, cal["threshold"] if cal else None)
    DOC.write_text(R.markdown(res, info, cal, assets), encoding="utf-8")


def read_assets(run: Path) -> dict:
    """asset_loss.csv + assets.csv of a run of `make all` (for the priority table)."""
    def rows(name):
        with open(run / name, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    meta_path = run / "run_metadata.json"
    return {"run": str(run), "loss": rows("asset_loss.csv"), "assets": rows("assets.csv"),
            "run_id": json.loads(meta_path.read_text(encoding="utf-8"))["run_id"]
            if meta_path.exists() else run.name}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="floodvalue.calibration.uncertainty_val")
    ap.add_argument("--assets-run", help="results dir of `make all` for the priority table")
    args = ap.parse_args(argv)
    method = os.environ.get("FLOODVALUE_METHOD", DEFAULT_METHOD)
    config = json.loads((MODELS_DIR / method / "config.json").read_text(encoding="utf-8"))
    cal_path = MODELS_DIR / method / "calibration.json"
    cal_file = C.load(cal_path, config["weights_sha256"]) if cal_path.exists() else None
    cal = cal_file["calibrator"] if cal_file else None
    threshold = float(cal_file["threshold"]) if cal_file else float(config["threshold"])
    val = collect(method, "validation", cal, threshold)
    dev = collect(method, "dev", cal, threshold)
    res = study(val, dev)
    res.update(method=method, threshold=threshold, n_members=val["n_members"],
               n_chips={"validation": len(val["chips"]), "dev": len(dev["chips"])})
    write_outputs(res, method, cal_path, {"calibrator": cal, "threshold": threshold}, config, val,
                  Path(args.assets_run) if args.assets_run else None)
    print(json.dumps({"chosen": res["selection"]["chosen"], "scores": res["selection"]["scores"]},
                     ensure_ascii=False))
    print(f"report: {DOC.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
