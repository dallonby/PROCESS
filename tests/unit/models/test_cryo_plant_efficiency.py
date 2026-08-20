"""Unit tests for the Strobridge-survey cryoplant second-law efficiency model."""

import pytest

from process.models.power import Power


@pytest.fixture
def power():
    return Power()


def test_strobridge_small_cooler_anchor(power):
    """~2% of Carnot at 1 W (Kittel 2007: the 1974 survey mean at 1 W)."""
    eta = power.cryo_plant_second_law_efficiency(1.0, 0.30)
    assert eta == pytest.approx(0.0184, rel=0.02)


def test_strobridge_lhc_scale(power):
    """LHC-class plants (18 kW at 4.5 K): curve gives ~0.24; the built plants
    achieved 0.28 (230 W/W), so the correlation is conservative there."""
    eta = power.cryo_plant_second_law_efficiency(18.0e3, 0.30)
    assert eta == pytest.approx(0.2406, rel=0.01)
    assert eta < 0.28


def test_cap_binds_at_plant_scale(power):
    """Above ~130 kW the eta_cryo_plant_max cap binds."""
    assert power.cryo_plant_second_law_efficiency(2.0e5, 0.30) == 0.30
    assert power.cryo_plant_second_law_efficiency(3.675e6, 0.30) == 0.30


def test_capacity_clamped_to_validity_range(power):
    """Outside 0.2 W - 1 MW the capacity is clamped, not extrapolated."""
    lo = power.cryo_plant_second_law_efficiency(1.0e-3, 0.99)
    assert lo == power.cryo_plant_second_law_efficiency(0.2, 0.99)
    hi = power.cryo_plant_second_law_efficiency(1.0e9, 0.99)
    assert hi == power.cryo_plant_second_law_efficiency(1.0e6, 0.99)
    assert hi == pytest.approx(1.047 / 3.0, rel=0.01)


def test_monotone_in_capacity(power):
    """Efficiency must not decrease with plant size within the validity range."""
    caps = [1.0, 10.0, 100.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
    etas = [power.cryo_plant_second_law_efficiency(q, 0.99) for q in caps]
    assert all(b >= a for a, b in zip(etas, etas[1:]))


def test_electric_power_ratio_at_20k(power):
    """At 20 K with the 0.30 cap the specific power is 45.5 W_e/W, vs 105.1
    W_e/W for the legacy eff_tf_cryo = 0.13 (ITER 4.5 K figure)."""
    t_room, t_cold = 293.15, 20.0
    eta = power.cryo_plant_second_law_efficiency(2.0e5, 0.30)
    assert (t_room - t_cold) / (eta * t_cold) == pytest.approx(45.5, rel=0.01)
    assert (t_room - t_cold) / (0.13 * t_cold) == pytest.approx(105.06, rel=0.01)
