"""Main method candidate A (issue #16): per-pixel classifier on S1 features.

features.py  pure feature maps from S1 dB (clipping, NaN-aware local statistics)
train.py     sample train pixels, fit, choose on dev, report dev/validation (D-012)
predict.py   contract §3 predict_proba and `--chip` rasters (S1-only, CPU)
ablation.py  controlled experiment on dev: which inputs (VV / VH / VV+VH / +texture)
"""
