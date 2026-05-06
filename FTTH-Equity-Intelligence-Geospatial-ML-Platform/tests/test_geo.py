import pandas as pd

from ftth_equity import geo


def test_detect_crs_wgs84_when_degrees():
    df = pd.DataFrame({"x": [4.0, 4.1], "y": [48.0, 48.5]})
    assert geo.detect_crs(df) == "EPSG:4326"


def test_detect_crs_lambert_when_metres():
    df = pd.DataFrame({"x": [700000, 720000], "y": [6800000, 6810000]})
    assert geo.detect_crs(df) == "EPSG:2154"


def test_to_wgs84_passthrough_for_degree_input():
    df = pd.DataFrame({"x": [4.087, 4.063], "y": [48.254, 48.291]})
    out = geo.to_wgs84(df)
    assert (out["lon"] == out["x"]).all()
    assert (out["lat"] == out["y"]).all()


def test_haversine_zero_for_same_point():
    d = geo.haversine_km(48.3, 4.0, 48.3, 4.0)
    assert abs(float(d)) < 1e-9


def test_haversine_known_distance():
    # Troyes ~ 4.07 E, 48.30 N to Paris ~ 2.35 E, 48.86 N is ~ 140 km
    d = float(geo.haversine_km(48.30, 4.07, 48.86, 2.35))
    assert 130 < d < 160
