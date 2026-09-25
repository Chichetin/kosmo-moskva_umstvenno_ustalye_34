import json
import math
import sys
import types

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from floodvalue.calibration import calibrator as C
from floodvalue.calibration import uncertainty as U
from floodvalue.calibration import uncertainty_val as V
from floodvalue.model import predict as P
from floodvalue.model.eval.core import auroc as exact_auroc
from floodvalue.paths import MODELS_DIR

PLATT = {"kind": "platt", "a": 1.4647, "b": 1.521}


def h(p):
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# --- measures -----------------------------------------------------------------------------

def test_entropy_bits_hand_values_and_nan():
    e = U.entropy_bits(np.array([0.0, 0.5, 1.0, 0.1, np.nan]))
    assert e[:3].tolist() == [0.0, 1.0, 0.0]
    assert e[3] == pytest.approx(h(0.1))
    assert np.isnan(e[4])


def test_threshold_margin_is_one_at_threshold_and_odds_ratio_elsewhere():
    t = 0.37
    m = U.threshold_margin(np.array([t, 0.5, 0.2]), t)
    odds = lambda p: p / (1 - p)  # noqa: E731
    assert m[0] == pytest.approx(1.0)
    assert m[1] == pytest.approx(odds(t) / odds(0.5))
    assert m[2] == pytest.approx(odds(0.2) / odds(t))
    # symmetric in log-odds around t, not around 0.5
    assert U.threshold_margin(np.array([0.5]), 0.5)[0] == pytest.approx(1.0)


def test_measures_identity_hand_values():
    # two members that disagree completely: q = 0 and 1 -> mean 0.5
    members = np.array([[[0.0, 0.3]], [[1.0, 0.3]]], dtype=np.float32)
    u = U.measures(members, None, 0.5)
    assert u["entropy_p"][0, 0] == pytest.approx(1.0)
    assert u["mutual_info"][0, 0] == pytest.approx(1.0)       # all of it is disagreement
    assert u["member_std"][0, 0] == pytest.approx(0.5)
    # members agree -> no epistemic part, entropy of p stays
    assert u["mutual_info"][0, 1] == pytest.approx(0.0, abs=1e-7)
    assert u["member_std"][0, 1] == pytest.approx(0.0, abs=1e-7)
    assert u["entropy_p"][0, 1] == pytest.approx(h(0.3), rel=1e-6)
    assert all(v.dtype == np.float32 and v.shape == (1, 2) for v in u.values())


def test_measures_use_calibrated_members_and_delivered_p():
    members = np.array([[[0.1]], [[0.3]], [[0.2]]], dtype=np.float32)
    q = C.apply(PLATT, members).astype(np.float64)
    p = C.apply(PLATT, members.mean(0))
    u = U.measures(members, PLATT, 0.37, p)
    assert u["member_std"][0, 0] == pytest.approx(q.std(), rel=1e-5)
    assert u["entropy_p"][0, 0] == pytest.approx(h(float(p[0, 0])), rel=1e-5)
    mi = h(q.mean()) - np.mean([h(x) for x in q.ravel()])
    assert u["mutual_info"][0, 0] == pytest.approx(mi, rel=1e-4)
    # an explicitly passed delivered p is the one whose entropy is reported
    other = np.array([[0.5]], dtype=np.float32)
    assert U.measures(members, PLATT, 0.37, other)["entropy_p"][0, 0] == pytest.approx(1.0)


def test_measures_nodata_propagates():
    members = np.full((3, 2, 2), 0.4, np.float32)
    members[1, 0, 0] = np.nan
    for v in U.measures(members, PLATT, 0.37).values():
        assert np.isnan(v[0, 0]) and np.isfinite(v[1, 1])
    lo, hi = U.p_range(members, PLATT)
    assert np.isnan(lo[0, 0]) and np.isnan(hi[0, 0])


def test_p_range_contains_delivered_p():
    rng = np.random.default_rng(0)
    members = rng.uniform(0, 1, (5, 64, 64)).astype(np.float32)
    p = C.apply(PLATT, members.mean(0))
    lo, hi = U.p_range(members, PLATT)
    assert np.all(lo <= p) and np.all(p <= hi) and np.all(lo < hi)
    assert np.allclose(lo, C.apply(PLATT, members.min(0)))


def test_single_member_method_falls_back_to_entropy():
    assert U.resolve("member_std", 1) == "entropy_p"
    assert U.resolve("mutual_info", 1) == "entropy_p"
    assert U.resolve("threshold_margin", 1) == "threshold_margin"
    assert U.resolve("member_std", 5) == "member_std"
    with pytest.raises(ValueError):
        U.resolve("total_entropy", 5)
    lo, hi = U.p_range(np.array([[0.2, 0.7]], np.float32), None)   # (H, W) = one member
    assert np.array_equal(lo, hi) and lo.shape == (1, 2)
    assert lo[0].tolist() == [pytest.approx(0.2), pytest.approx(0.7)]


# --- AUROC and the selection rule ---------------------------------------------------------

def test_auroc_hist_hand_values():
    assert U.auroc_hist([0, 0, 5], [5, 0, 0]) == 1.0      # errors always higher
    assert U.auroc_hist([5, 0, 0], [0, 0, 5]) == 0.0
    assert U.auroc_hist([2, 0], [2, 0]) == 0.5            # ties count one half
    assert np.isnan(U.auroc_hist([0, 0], [1, 1]))


def test_auroc_hist_on_quantile_bins_matches_exact():
    rng = np.random.default_rng(3)
    u = rng.normal(size=20000)
    err = rng.uniform(size=u.size) < 1 / (1 + np.exp(-2 * u))
    edges = V.quantile_edges(u, 5000)
    he, ho = V.chip_hists(u, err, np.zeros(u.size, np.int32), 1, edges)
    assert U.auroc_hist(he[0], ho[0]) == pytest.approx(exact_auroc(u, err), abs=2e-3)


def _chips(sep_by_event, n_chips=6, n=400, seed=0):
    """Per-chip histograms of a measure whose error/correct separation depends on the event."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(-6, 8, 200)
    rows = []
    for ev, sep in sep_by_event.items():
        for _ in range(n_chips):
            e = np.histogram(rng.normal(sep, 1, n), edges)[0]
            c = np.histogram(rng.normal(0, 1, 4 * n), edges)[0]
            rows.append((ev, e, c))
    return rows


def test_select_uses_worst_event_and_prefers_simpler_on_near_tie():
    per_chip = {
        # good on A, useless on B -> worst event decides against it
        "entropy_p": _chips({"A": 3.0, "B": 0.0}, seed=1),
        # clearly best in the worse event
        "member_std": _chips({"A": 1.5, "B": 1.5}, seed=2),
        # indistinguishable from member_std but later in the simplicity order
        "mutual_info": _chips({"A": 1.5, "B": 1.5}, seed=3),
    }
    res = U.select(per_chip, n_boot=200)
    assert res["scores"]["entropy_p"] < res["scores"]["member_std"]
    assert res["chosen"] == "member_std"          # simpler of the two tied, not entropy_p
    assert res["vs_best"]["entropy_p"]["ci_lo"] > 0


def test_select_keeps_clearly_better_complex_measure():
    per_chip = {"entropy_p": _chips({"A": 0.5, "B": 0.5}, seed=1),
                "mutual_info": _chips({"A": 2.5, "B": 2.5}, seed=2)}
    res = U.select(per_chip, n_boot=200)
    assert res["best"] == res["chosen"] == "mutual_info"


def test_select_picks_simpler_measure_within_noise():
    # same separation, independent noise: whichever scores higher, entropy_p (simpler) is kept
    per_chip = {"entropy_p": _chips({"A": 1.0, "B": 1.0}, n=60, seed=5),
                "threshold_margin": _chips({"A": 1.0, "B": 1.0}, n=60, seed=6)}
    res = U.select(per_chip, n_boot=200)
    assert res["chosen"] == "entropy_p"
    assert res["vs_best"]["entropy_p"]["ci_lo"] <= 0 <= res["vs_best"]["entropy_p"]["ci_hi"]


# --- predict writes uncertainty.tif and the member range (contracts §2) -----------------------

H = W = 16


def _fake_members(s1):
    base = C.sigmoid(-(s1[1] + 20.0)).astype(np.float32)
    m = np.stack([np.clip(base + d, 0, 1) for d in (-0.1, 0.0, 0.1)]).astype(np.float32)
    m[:, ~np.isfinite(s1).all(0)] = np.nan
    return m


@pytest.fixture
def ensemble_method(tmp_path, monkeypatch):
    mod = types.ModuleType("fake_ens_mod")
    mod.predict_members = _fake_members
    mod.predict_proba = lambda s1: _fake_members(s1).mean(0)
    monkeypatch.setitem(sys.modules, "fake_ens_mod", mod)
    monkeypatch.setitem(P.METHODS, "fake", "fake_ens_mod")
    models = tmp_path / "models"
    (models / "fake").mkdir(parents=True)
    (models / "fake" / "config.json").write_text(json.dumps(
        {"threshold": 0.5, "threshold_status": "raw", "weights_sha256": "w1"}))
    (models / "fake" / "calibration.json").write_text(json.dumps(
        {"calibrator": PLATT, "threshold": 0.37, "input_model_sha256": "w1"}))
    monkeypatch.setattr(P, "MODELS_DIR", models)
    monkeypatch.setenv("FLOODVALUE_METHOD", "fake")
    rng = np.random.default_rng(1)
    s1 = rng.normal(-20, 3, (2, H, W)).astype(np.float32)
    s1[:, :2, :3] = np.nan
    d = tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand"
    d.mkdir(parents=True)
    with rasterio.open(d / "X_1_S1Hand.tif", "w", driver="GTiff", width=W, height=H, count=2,
                       dtype="float32", crs="EPSG:4326", nodata=np.nan,
                       transform=from_origin(80, 7, 9e-5, 9e-5)) as dst:
        dst.write(s1)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    return models / "fake", s1


def _predict(tmp_path):
    out = tmp_path / "out"
    assert P.run(types.SimpleNamespace(chip="X_1", out=str(out))) == 0
    with rasterio.open(out / "flood_probability.tif") as a:
        p, grid = a.read(1), (a.transform, a.crs, a.shape)
    with rasterio.open(out / "uncertainty.tif") as a:
        u, u_grid, u_nd = a.read(1), (a.transform, a.crs, a.shape), a.nodata
        assert a.count == 1 and a.dtypes[0] == "float32"
    with rasterio.open(out / "flood_probability_range.tif") as a:
        rng_, r_grid = a.read(), (a.transform, a.crs, a.shape)
        assert a.descriptions == ("p_low", "p_high")
    assert grid == u_grid == r_grid
    return p, u, u_nd, rng_, json.loads((out / "prediction.json").read_text())


def test_predict_writes_chosen_measure_and_range(ensemble_method, tmp_path):
    model_dir, s1 = ensemble_method
    (model_dir / "uncertainty.json").write_text(json.dumps(
        {"measure": "member_std", "status": "checked", "input_model_sha256": "w1",
         "validation": {"ALL": {"auroc": 0.9}}}))
    p, u, u_nd, (lo, hi), meta = _predict(tmp_path)
    ok = np.isfinite(s1).all(0)
    want = U.measures(_fake_members(s1), PLATT, 0.37)["member_std"]
    assert np.allclose(u[ok], want[ok], atol=1e-6)
    assert u_nd == -9999 and np.all(u[~ok] == -9999)
    assert np.all(lo[ok] <= p[ok]) and np.all(p[ok] <= hi[ok]) and np.all(lo[~ok] == -9999)
    unc = meta["uncertainty"]
    assert unc["measure"] == "member_std" and unc["status"] == "checked"
    assert unc["n_members"] == 3 and unc["validation"] == {"ALL": {"auroc": 0.9}}


def test_predict_without_choice_writes_unverified_entropy_of_delivered_p(ensemble_method, tmp_path):
    _, s1 = ensemble_method
    p, u, _, _, meta = _predict(tmp_path)
    ok = np.isfinite(s1).all(0)
    # the same measure eval-test reports: entropy of flood_probability.tif
    assert np.allclose(u[ok], U.entropy_bits(p[ok]), atol=1e-6)
    assert meta["uncertainty"]["measure"] == "entropy_p"
    assert meta["uncertainty"]["status"].startswith("не проверено")
    assert meta["uncertainty"]["validation"] is None


def test_predict_refuses_uncertainty_choice_of_other_weights(ensemble_method, tmp_path):
    model_dir, _ = ensemble_method
    (model_dir / "uncertainty.json").write_text(json.dumps(
        {"measure": "entropy_p", "status": "x", "input_model_sha256": "other"}))
    with pytest.raises(RuntimeError, match="input_model_sha256"):
        P.run(types.SimpleNamespace(chip="X_1", out=str(tmp_path / "out")))


def test_single_model_method_gets_entropy_fallback_even_if_ensemble_measure_chosen(
        ensemble_method, tmp_path, monkeypatch):
    model_dir, s1 = ensemble_method
    mod = sys.modules["fake_ens_mod"]
    monkeypatch.delattr(mod, "predict_members")
    (model_dir / "uncertainty.json").write_text(json.dumps(
        {"measure": "mutual_info", "status": "checked", "input_model_sha256": "w1"}))
    p, u, _, (lo, hi), meta = _predict(tmp_path)
    ok = np.isfinite(s1).all(0)
    assert meta["uncertainty"]["measure"] == "entropy_p"
    assert meta["uncertainty"]["status"].startswith("не проверено")
    assert np.allclose(u[ok], U.entropy_bits(p[ok]), atol=1e-6)
    assert np.allclose(lo[ok], p[ok], atol=1e-6) and np.allclose(hi[ok], p[ok], atol=1e-6)


def test_shipped_uncertainty_choice_matches_shipped_model():
    path = MODELS_DIR / "unet" / "uncertainty.json"
    if not path.exists():
        pytest.skip("no shipped uncertainty choice")
    info = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads((MODELS_DIR / "unet" / "config.json").read_text(encoding="utf-8"))
    assert info["input_model_sha256"] == config["weights_sha256"]
    assert info["measure"] in U.MEASURES
    assert info["calibration_sha256"] == C.file_sha256(MODELS_DIR / "unet" / "calibration.json")


def test_study_reads_only_validation_and_dev():
    for part in ("test", "train"):
        with pytest.raises(ValueError):
            V.collect("unet", part, None, 0.5)


def _collected(events, n_per_chip=3000, seed=0):
    """Synthetic output of `collect`: entropy separates errors, the other measures are noise."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in (*U.MEASURES, "err", "fp", "fn", "unstable", "chip")}
    chips, evs = [], []
    for i, ev in enumerate(events):
        err = rng.uniform(size=n_per_chip) < 0.1
        cols["entropy_p"].append(np.clip(rng.normal(0.2 + 0.5 * err, 0.15), 0, 1))
        for m in ("threshold_margin", "member_std", "mutual_info"):
            cols[m].append(rng.uniform(size=n_per_chip))
        fp = err & (rng.uniform(size=n_per_chip) < 0.5)
        cols["err"].append(err)
        cols["fp"].append(fp)
        cols["fn"].append(err & ~fp)
        cols["unstable"].append(rng.uniform(size=n_per_chip) < 0.07)
        cols["chip"].append(np.full(n_per_chip, i, np.int32))
        chips.append(f"{ev}_{i}")
        evs.append(ev)
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d.update(chips=chips, events=evs, n_members=5, maps={})
    return d


def test_study_selects_informative_measure_and_report_renders():
    from floodvalue.calibration import uncertainty_report as R
    val = _collected(["Nigeria"] * 4 + ["Sri-Lanka"] * 4)
    dev = _collected(["USA"] * 4, seed=1)
    res = V.study(val, dev)
    assert res["selection"]["chosen"] == "entropy_p"
    st = res["parts"]["validation"]["ALL"]["entropy_p"]
    assert st["mean_u_error"] > st["mean_u_correct"] and st["auroc"] > 0.9
    assert res["parts"]["dev"]["USA"]["mutual_info"]["auroc"] == pytest.approx(0.5, abs=0.05)
    # the top-10 % flag is cut on validation and reused on dev
    assert res["parts"]["validation"]["ALL"]["entropy_p"]["flagged_share"] == pytest.approx(0.1, abs=0.01)
    res.update(method="unet", threshold=0.37, n_members=5, n_chips={"validation": 8, "dev": 4})
    info = {"commit": "abc", "created_at": "2026-09-25T00:00:00"}
    md = R.markdown(res, info, {}, None)
    assert "**выбран**" in md and "| dev | USA |" in md and "| dev | ALL |" not in md
    assert "не** доверительный" in md
