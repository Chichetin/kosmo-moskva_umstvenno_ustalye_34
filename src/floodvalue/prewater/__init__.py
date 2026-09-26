"""Pre-event water from the Sentinel-1 archive before the flood: the "new water" layer (#149).

For a chip, Sentinel-1 RTC scenes of a fixed window that ends well before the S1 date of the
event are read from the Planetary Computer STAC archive; a pixel is pre-event water when
VH <= -23 dB (D-021) in at least half of its observations. The layer

  pre_event_water.tif         UInt8 1/0, nodata 255, S1 grid of the chip
  pre_event_water.json        scenes, parameters, status ok/insufficient/unavailable, sources
  new_water_probability.tif   Float32 p x (1 - pre_event_water), nodata -9999

is written next to the run's rasters and only carried into the set by export as optional
files. It is never an input of p, the mask, thresholds, EL or the procurement plan
(tests/prewater/test_prewater_firewall.py): the product stays a water map (D-007).
"""
