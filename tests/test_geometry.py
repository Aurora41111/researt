"""Geometry / LoS-probability unit tests."""

import numpy as np

from muav_isac import geometry as geom
from muav_isac.config import default_params


def test_distance3d_is_pythagorean():
    uav = np.array([0.0, 0.0])
    grd = np.array([3.0, 4.0])
    H = 12.0
    assert np.isclose(geom.distance3d(uav, grd, H), np.sqrt(5.0**2 + 12.0**2))


def test_elevation_zero_and_overhead():
    # UAV on the ground -> 0 deg; UAV directly above -> 90 deg
    assert np.isclose(geom.elevation_deg(H=0.0, d=10.0), 0.0)
    assert np.isclose(geom.elevation_deg(H=10.0, d=10.0), 90.0)


def test_elevation_monotonic_in_altitude():
    d = 200.0
    el = [geom.elevation_deg(H, d) for H in (50.0, 100.0, 150.0, 190.0)]
    assert all(el[i] < el[i + 1] for i in range(len(el) - 1))


def test_plos_in_range_and_monotonic():
    p = default_params()
    elevs = np.linspace(0.0, 90.0, 20)
    plos = geom.p_los(p.C, p.D, elevs)
    assert np.all((plos >= 0.0) & (plos <= 1.0))
    # higher elevation -> higher LoS probability
    assert np.all(np.diff(plos) > 0.0)
