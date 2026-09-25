"""Main method B: compact U-Net on Sentinel-1 only (VV, VH in dB). See docs/contracts.md §3.

`predict_proba` is re-exported lazily so importing the package does not load torch.
"""


def predict_proba(s1_db, model_dir=None):
    from floodvalue.model.unet.predict import predict_proba as _predict_proba

    return _predict_proba(s1_db, model_dir=model_dir)
