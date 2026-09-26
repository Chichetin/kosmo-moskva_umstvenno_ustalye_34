"""Media monitoring and eyewitness reports (#156, #168): a standalone module.

It feeds a register of *reports about* floods (news, bulletins, eyewitnesses) that tells the
operator where to look. It never touches p, the mask, the threshold, the portfolio, EL or
A/B/C, and no other stage imports it (tests/media/test_media_firewall.py). Imports: only
floodvalue.paths and third-party libraries; the split and the S1 manifest are read as files.
Run: python -m floodvalue.media --help (docs/features/media-monitoring-run.md).
"""
