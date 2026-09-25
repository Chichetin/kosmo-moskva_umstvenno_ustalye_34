"""floodvalue.context — visual context layers for the operator (D-042).

Context is for a human looking at the map, never an input: no model, calibration, EL or
procurement code reads it (tests/context/test_firewall.py enforces that). Two kinds:

* offline — Sentinel-2 true colour of the same chip (Sen1Floods11 S2Hand: same grid as S1,
  acquired days before/after S1), rendered to PNG with date and disclaimer;
* online — basemap tile services (Esri World Imagery, OSM, EOX s2cloudless) listed with
  attribution for the interface to load live; their imagery date differs from the event.

    python -m floodvalue.context --chip Sri-Lanka_713926 --out outputs/<run_id>
"""
