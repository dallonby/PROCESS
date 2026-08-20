"""Unit tests for the ST centre-post fast-fluence wiring (constraint 53)."""

import pytest

from process.models.availability import YEAR_SECONDS
from process.models.blankets.hcpb import CCFE_HCPB


def test_st_cp_fluence_arithmetic():
    """Fluence = flux x availability x plant life (365.25-day years).

    Round-trip against the availability model's centre-post lifetime
    definition: cplife = flu_max / (flux * yr), so a plant living exactly
    cplife at 100% availability accumulates exactly flu_max.
    """
    flux = 6.417e16  # shipped ST baseline value [m^-2 s^-1]
    flu_max = 1.0e23

    cplife_fpy = flu_max / (flux * YEAR_SECONDS)
    fluence = CCFE_HCPB.st_cp_fluence(flux, 1.0, cplife_fpy)
    assert fluence == pytest.approx(flu_max, rel=1e-12)


def test_st_cp_fluence_scales_with_availability():
    full = CCFE_HCPB.st_cp_fluence(1.0e16, 1.0, 40.0)
    derated = CCFE_HCPB.st_cp_fluence(1.0e16, 0.75, 40.0)
    assert derated == pytest.approx(0.75 * full)
    assert full == pytest.approx(1.0e16 * 40.0 * YEAR_SECONDS)
    assert YEAR_SECONDS == pytest.approx(365.25 * 24.0 * 3600.0)


def test_st_cp_fluence_zero_flux():
    assert CCFE_HCPB.st_cp_fluence(0.0, 0.75, 40.0) == 0.0
