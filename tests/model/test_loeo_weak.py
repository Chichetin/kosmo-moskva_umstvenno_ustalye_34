"""LOEO weak-label mixing, fine-tune stage and replay command (#103, critic 26.09)."""

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine

from floodvalue.model.unet import loeo as L
from floodvalue.model.unet.train import augment, masked_loss


def _old_loss(logits, target, mask, dice_weight):
    """D-019 formula before the weighted-Dice fix (boolean masks only)."""
    import torch.nn.functional as F
    logits, target, m = logits.float(), target.float(), mask.float()
    n = m.sum().clamp_min(1.0)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * m).sum() / n
    p, t = torch.sigmoid(logits) * m, target * m
    return bce + dice_weight * (1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1))


def _case(seed=0, shape=(2, 16, 16)):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g), (torch.rand(shape, generator=g) > 0.7).float(),
            torch.rand(shape, generator=g) > 0.2)


def test_boolean_mask_loss_is_unchanged_d019():
    logits, target, mask = _case()
    assert torch.equal(masked_loss(logits, target, mask, 0.5), _old_loss(logits, target, mask, 0.5))


def test_fractional_weight_enters_every_dice_sum_once():
    logits, target, mask = _case(1)
    w = torch.where(mask, 0.3, 0.0)
    s = torch.sigmoid(logits)
    inter, den = (s * target * w).sum(), (s * w).sum() + (target * w).sum()
    import torch.nn.functional as F
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum()
    want = bce + 0.5 * (1 - (2 * inter + 1) / (den + 1))
    assert torch.allclose(masked_loss(logits, target, w, 0.5), want)


def test_zero_weight_pixels_do_not_matter():
    logits, target, mask = _case(2)
    w = torch.where(mask, 1.0, 0.0)
    noisy = torch.where(mask, logits, logits + 100.0)
    assert torch.allclose(masked_loss(logits, target, w, 0.5), masked_loss(noisy, target, w, 0.5))


def test_mask_codes_survive_augmentation():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 4, 32, 32)
    y = torch.zeros(4, 32, 32, dtype=torch.uint8)
    m = torch.randint(0, 3, (4, 32, 32), generator=g).to(torch.uint8)
    _, _, mb = augment(x, y, m, 16, 0.0, torch.ones(3), 3, g)
    assert set(mb.unique().tolist()) <= {0, 1, 2}


def test_sample_batch_hand_fraction_and_uniform_default():
    g = torch.Generator().manual_seed(0)
    hand, weak = torch.arange(0, 10), torch.arange(10, 60)
    idx = L.sample_batch(g, 16, hand, weak, 0.5)
    assert (idx < 10).sum() == 8 and (idx >= 10).sum() == 8
    idx = L.sample_batch(g, 16, hand, weak, None)
    assert idx.min() >= 0 and idx.max() < 60


def _tif(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=arr.dtype, crs="EPSG:4326",
                       transform=Affine(1e-4, 0, 0, 0, -1e-4, 0)) as f:
        f.write(arr)


def test_weak_chip_ids_only_allowed_train_events(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))
    root = tmp_path / "raw" / "WeaklyLabeled" / L.WEAK_LABEL
    for c in ("Ghana_1", "India_2", "USA_3", "Sri-Lanka_4", "Colombia_5", "Spain_6"):
        _tif(root / f"{c}_{L.WEAK_LABEL}.tif", np.zeros((1, 4, 4), np.int16))
    events = {"Ghana", "India", "USA", "Sri-Lanka", "Colombia", "Spain"}
    assert L.weak_chip_ids(events) == ["Ghana_1", "India_2"]
    assert L.weak_chip_ids({"India"}) == ["India_2"]


def test_read_weak_chip_maps_labels_and_checks_bands(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOODVALUE_DATA_DIR", str(tmp_path))
    root = tmp_path / "raw" / "WeaklyLabeled"
    _tif(root / "S1Weak" / "Ghana_1_S1Weak.tif", np.full((2, 4, 4), -15.0, np.float32))
    _tif(root / L.WEAK_LABEL / f"Ghana_1_{L.WEAK_LABEL}.tif",
         np.array([[[-1, 0, 1, 7]] * 4], np.int16))
    s1, lab = L.read_weak_chip("Ghana_1")
    assert s1.shape == (2, 4, 4) and lab[0].tolist() == [-1, 0, 1, -1]
    _tif(root / "S1Weak" / "Ghana_2_S1Weak.tif", np.zeros((3, 4, 4), np.float32))
    _tif(root / L.WEAK_LABEL / f"Ghana_2_{L.WEAK_LABEL}.tif", np.zeros((1, 4, 4), np.int16))
    with pytest.raises(ValueError, match="2 bands"):
        L.read_weak_chip("Ghana_2")


def test_replay_command_round_trips_weak_zero_and_flags():
    args = L.parse_args(["--weak-weight", "0.0", "--tta", "--finetune-epochs", "5",
                         "--hand-frac", "0.5", "--out-dir", "x"])
    cmd = L.replay_command(args)
    assert "--weak-weight 0.0" in cmd and "--tta" in cmd and "--tta True" not in cmd
    again = L.parse_args(cmd.split()[3:] + ["--out-dir", "x"])
    assert vars(again) == vars(args)


def test_chip_median_norm_shifts_each_band_and_keeps_nan_and_label():
    s1 = np.array([[[-10.0, -12.0], [np.nan, -14.0]], [[-20.0, -22.0], [-24.0, np.nan]]], np.float32)
    lab = np.array([[1, 0], [-1, 0]], np.int16)
    out, lab2 = L.chip_median_norm(lambda c: (s1, lab))("X_1")
    assert np.nanmedian(out[0]) == 0 and np.nanmedian(out[1]) == 0
    assert np.isnan(out[0, 1, 0]) and np.isnan(out[1, 1, 1]) and lab2 is lab
    assert np.allclose(out[0] - np.nanmedian(s1[0]) * 0, s1[0] - np.nanmedian(s1[0]), equal_nan=True)
    assert not np.shares_memory(out, s1)  # the raw reader's array (baseline) is not modified


def test_build_net_default_is_d019_unet_and_rejects_unknown_arch():
    from floodvalue.model.unet.net import UNet
    config = {"arch": {"base": 16, "dropout": 0.1}, "features": {"channels": ["vv", "vh", "vv_minus_vh"]}}
    assert isinstance(L.build_net({}, config), UNet)
    assert isinstance(L.build_net({"arch": "unet"}, config), UNet)
    with pytest.raises(ValueError, match="unknown arch"):
        L.build_net({"arch": "resnet"}, config)


def test_smp_net_has_the_unet_logits_shape():
    pytest.importorskip("segmentation_models_pytorch")
    config = {"arch": {"base": 16}, "features": {"channels": ["vv", "vh", "vv_minus_vh"]}}
    net = L.build_net({"arch": "smp_unet:resnet18", "pretrained": False}, config)
    assert net(torch.randn(2, 4, 64, 64)).shape == (2, 64, 64)
