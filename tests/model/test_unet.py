import json
import shutil

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from floodvalue.model.unet import data as D
from floodvalue.model.unet import predict as P
from floodvalue.model.unet.features import loss_mask, make_features, valid_mask
from floodvalue.model.unet.net import UNet
from floodvalue.model.unet.train import augment, jitter_std, masked_loss

CH = ["vv", "vh", "vv_minus_vh"]
MEAN, STD = [-10.0, -17.0, 7.0], [3.0, 3.0, 3.0]


def s1_with_nan(h=32, w=48):
    rng = np.random.default_rng(0)
    s1 = np.stack([rng.normal(-10, 3, (h, w)), rng.normal(-17, 3, (h, w))]).astype(np.float32)
    s1[0, 0, 0] = np.nan  # only VV missing -> invalid
    s1[1, 5, 7] = np.nan  # only VH missing -> invalid
    return s1


def test_features_nan_become_zero_and_validity_channel():
    s1 = s1_with_nan()
    x = make_features(s1, CH, MEAN, STD)
    assert x.shape == (4, 32, 48) and x.dtype == np.float32
    assert np.isfinite(x).all()
    assert x[:, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert x[3, 5, 7] == 0 and x[3, 1, 1] == 1
    assert x[0, 1, 1] == pytest.approx((s1[0, 1, 1] - MEAN[0]) / STD[0])


def test_features_clip_bright_vv():
    s1 = np.full((2, 16, 16), -15.0, np.float32)
    s1[0, 0, 0] = 25.0
    x = make_features(s1, ["vv"], [0.0], [1.0])
    assert x[0, 0, 0] == 10.0


def test_loss_mask_excludes_minus_one_and_nan_under_label():
    s1 = s1_with_nan()
    label = np.zeros((32, 48), np.int16)
    label[0, 0] = 1       # label present under NaN S1 -> excluded
    label[2, 2] = -1      # unlabeled -> excluded
    m = loss_mask(s1, label)
    assert not m[0, 0] and not m[2, 2] and not m[5, 7] and m[1, 1]
    assert m.sum() == 32 * 48 - 3


def test_masked_loss_ignores_masked_pixels():
    logits = torch.zeros(1, 4, 4)
    target = torch.zeros(1, 4, 4)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    mask[0, 0, 0] = False
    base = masked_loss(logits, target, mask, 0.5)
    target2 = target.clone()
    target2[0, 0, 0] = 1  # change only a masked pixel
    logits2 = logits.clone()
    logits2[0, 0, 0] = 50.0
    assert masked_loss(logits2, target2, mask, 0.5) == pytest.approx(base.item())
    target3 = target.clone()
    target3[0, 1, 1] = 1  # change a counted pixel
    neg = logits - 2.0
    assert masked_loss(neg, target3, mask, 0.0) > masked_loss(neg, target, mask, 0.0)
    assert masked_loss(logits, target3, mask, 0.5).item() != pytest.approx(base.item())


def test_augment_keeps_label_and_mask_aligned_with_validity():
    gen = torch.Generator().manual_seed(0)
    x = torch.zeros(2, 4, 64, 64)
    y = torch.zeros(2, 64, 64, dtype=torch.uint8)
    m = torch.ones(2, 64, 64, dtype=torch.uint8)
    x[:, 3] = 1.0
    x[:, 3, :10, :] = 0.0   # invalid stripe
    m[:, :10, :] = 0
    y[:, :10, :] = 1
    xb, yb, mb = augment(x, y, m, 32, 2.0, jitter_std(CH, STD), 3, gen)
    assert xb.shape == (2, 4, 32, 32)
    assert torch.equal(xb[:, 3].bool(), mb.bool())
    assert torch.equal(yb.bool(), ~mb.bool())
    assert (xb[:, :3][~xb[:, 3:4].bool().expand(-1, 3, -1, -1)] == 0).all()  # no jitter on NaN
    assert (xb[:, 2] == 0).all()  # VV−VH not shifted by a common dB offset


def test_unet_shapes():
    net = UNet(in_channels=4, base=4).eval()
    assert net(torch.zeros(1, 4, 64, 32)).shape == (1, 64, 32)


@pytest.fixture
def tiny_model_dir(tmp_path):
    torch.manual_seed(0)
    config = {"method": "unet", "arch": {"base": 4, "dropout": 0.0},
              "features": {"channels": CH, "mean": MEAN, "std": STD}, "threshold": 0.5}
    model = P.build_model(config)
    torch.save(model.state_dict(), tmp_path / P.WEIGHTS_FILE)
    config["weights_sha256"] = P.file_sha256(tmp_path / P.WEIGHTS_FILE)
    (tmp_path / P.CONFIG_FILE).write_text(json.dumps(config))
    P.load_model.cache_clear()
    return tmp_path


def test_predict_proba_contract(tiny_model_dir):
    s1 = s1_with_nan(40, 50)  # not a multiple of 16 -> padded internally
    p = P.predict_proba(s1, model_dir=tiny_model_dir)
    assert p.shape == (40, 50) and p.dtype == np.float32
    assert np.isnan(p[0, 0]) and np.isnan(p[5, 7])
    ok = valid_mask(s1)
    assert np.isfinite(p[ok]).all() and (p[ok] >= 0).all() and (p[ok] <= 1).all()


def test_predict_proba_rejects_weights_with_wrong_sha(tiny_model_dir):
    (tiny_model_dir / P.WEIGHTS_FILE).write_bytes(b"version https://git-lfs.github.com/spec/v1")
    P.load_model.cache_clear()
    with pytest.raises(RuntimeError, match="sha256"):
        P.predict_proba(s1_with_nan(), model_dir=tiny_model_dir)
    P.load_model.cache_clear()


def test_read_label_refuses_test_event():
    with pytest.raises(D.SealedTestError):
        D.read_label("Spain_7370579")


def test_split_chip_ids_refuse_test():
    with pytest.raises(D.SealedTestError):
        D.split_chip_ids("test")


def test_predict_cli_writes_contract_rasters(tiny_model_dir, tmp_path, monkeypatch):
    from floodvalue.model import predict as cli_predict

    chip = "Sri-Lanka_000001"
    raw = tmp_path / "data" / "raw" / "HandLabeled" / "S1Hand"
    raw.mkdir(parents=True)
    s1 = s1_with_nan(32, 32)
    with rasterio.open(raw / f"{chip}_S1Hand.tif", "w", driver="GTiff", width=32, height=32,
                       count=2, dtype="float32", crs="EPSG:4326", nodata=float("nan"),
                       transform=from_origin(80.0, 7.0, 8.983e-05, 8.983e-05)) as dst:
        dst.write(s1)
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(cli_predict, "MODELS_DIR", tiny_model_dir.parent)
    monkeypatch.setattr(P, "DEFAULT_MODEL_DIR", tiny_model_dir)
    (tiny_model_dir.parent / "unet").exists() or (tiny_model_dir.parent / "unet").symlink_to(
        tiny_model_dir)

    args = type("Args", (), {"chip": chip, "out": str(tmp_path / "out")})
    assert cli_predict.run(args) == 0
    with rasterio.open(tmp_path / "out" / "flood_probability.tif") as src:
        p = src.read(1)
        assert src.dtypes[0] == "float32" and src.nodata == -9999
        assert src.width == 32 and src.crs.to_epsg() == 4326
    with rasterio.open(tmp_path / "out" / "flood_mask.tif") as src:
        m = src.read(1)
        assert src.dtypes[0] == "uint8" and src.nodata == 255
    assert p[0, 0] == -9999 and m[0, 0] == 255
    valid = p != -9999
    assert np.array_equal(m[valid], (p[valid] >= 0.5).astype(np.uint8))


def test_ensemble_is_mean_of_member_probabilities(tmp_path):
    torch.manual_seed(1)
    config = {"method": "unet", "arch": {"base": 4, "dropout": 0.0},
              "features": {"channels": CH, "mean": MEAN, "std": STD}, "threshold": 0.5,
              "members": []}
    singles = []
    for i in range(2):
        d = tmp_path / f"single{i}"
        d.mkdir()
        model = P.build_model(config)
        torch.save(model.state_dict(), d / P.WEIGHTS_FILE)
        shutil.copyfile(d / P.WEIGHTS_FILE, tmp_path / f"member_{i}.pt")
        sha = P.file_sha256(d / P.WEIGHTS_FILE)
        config["members"].append({"file": f"member_{i}.pt", "sha256": sha})
        (d / P.CONFIG_FILE).write_text(json.dumps(
            {k: v for k, v in config.items() if k != "members"} | {"weights_sha256": sha}))
        singles.append(d)
    config["weights_sha256"] = P.members_sha256(config["members"])
    (tmp_path / P.CONFIG_FILE).write_text(json.dumps(config))
    P.load_model.cache_clear()
    s1 = s1_with_nan(32, 32)
    p = P.predict_proba(s1, model_dir=tmp_path)
    p0, p1 = (P.predict_proba(s1, model_dir=d) for d in singles)
    ok = valid_mask(s1)
    assert np.allclose(p[ok], (p0[ok] + p1[ok]) / 2, atol=1e-6)
    assert not np.allclose(p0[ok], p1[ok])
    P.load_model.cache_clear()
