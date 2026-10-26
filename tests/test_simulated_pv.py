"""Simulated-PV physical model (offline: no network)."""

import numpy as np

from src.phase1.regions import REGIONS, get_region
from src.phase1.simulated_pv import pv_power_kw, soiling_factor


def test_no_output_without_sun_and_never_negative():
    p = pv_power_kw(np.array([0.0, -5.0, 0.0]), np.array([30.0, 30.0, 10.0]), 1000.0)
    assert (p == 0).all()


def test_output_rises_with_irradiance_and_is_clipped_by_inverter():
    ghi = np.array([100.0, 400.0, 800.0, 1500.0])
    p = pv_power_kw(ghi, np.full(4, 25.0), 1000.0)
    assert np.all(np.diff(p) >= 0)
    assert p.max() <= 1000.0 / 1.25 + 1e-9


def test_hot_panels_produce_less():
    cool = pv_power_kw(np.array([800.0]), np.array([15.0]), 1000.0)
    hot = pv_power_kw(np.array([800.0]), np.array([40.0]), 1000.0)
    assert hot[0] < cool[0]


def test_soiling_is_bounded_and_resets():
    s = soiling_factor(48 * 120, seed=1)
    assert s.min() >= 0.90 and s.max() <= 1.0
    assert (np.diff(s) > 0).any()  # cleaning events restore output


def test_vellore_region_is_flagged_simulated():
    r = get_region(900)
    assert r.simulated and r.capacity_kw > 0 and "SIMULATED" in r.name
    assert not any(v.simulated for k, v in REGIONS.items() if k != 900)
