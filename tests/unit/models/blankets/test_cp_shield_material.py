"""Unit tests for the ST centre-post shield-material option (W2B5)."""

import pytest

from process.core.exceptions import ProcessValueError
from process.models.blankets.hcpb import CCFE_HCPB


def test_default_heat_factor_is_identity():
    assert CCFE_HCPB.cp_shield_material_heat_factor(0, -1.0) == 1.0


def test_w2b5_heat_defaults():
    """Windsor et al. (2021) Table 2 geometric-mean ratios vs WC+H2O."""
    assert CCFE_HCPB.cp_shield_material_heat_factor(1, -1.0) == 0.37
    assert CCFE_HCPB.cp_shield_material_heat_factor(2, -1.0) == 0.24


def test_user_override_beats_heat_default():
    assert CCFE_HCPB.cp_shield_material_heat_factor(2, 0.3) == 0.3


def test_unknown_material_raises():
    with pytest.raises(ProcessValueError):
        CCFE_HCPB.cp_shield_material_heat_factor(7, -1.0)


def test_flux_fit_scaled_by_material(process_models, monkeypatch):
    ccfe_hcpb = process_models.ccfe_hcpb
    monkeypatch.setattr(ccfe_hcpb.data.tfcoil, "i_tf_sup", 1)

    reference = ccfe_hcpb.st_tf_centrepost_fast_neut_flux(
        p_neutron_total_mw=400.0, sh_width=0.6, rmajor=3.0
    )
    monkeypatch.setattr(ccfe_hcpb.data.fwbs, "i_cp_shield_material", 2)
    w2b5 = ccfe_hcpb.st_tf_centrepost_fast_neut_flux(
        p_neutron_total_mw=400.0, sh_width=0.6, rmajor=3.0
    )
    # default is the thickness-dependent fit, evaluated at the steel-derated
    # path length (0.54 m, inside the measured domain) for this 0.6 m shield
    expected = CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, 0.6)
    assert w2b5 == pytest.approx(expected * reference)

    # a constant user override still applies verbatim
    monkeypatch.setattr(ccfe_hcpb.data.fwbs, "f_cp_shield_material_flux", 0.24)
    w2b5_const = ccfe_hcpb.st_tf_centrepost_fast_neut_flux(
        p_neutron_total_mw=400.0, sh_width=0.6, rmajor=3.0
    )
    assert w2b5_const == pytest.approx(0.24 * reference)


def test_heating_fit_scaled_by_material_sc_branch(process_models, monkeypatch):
    ccfe_hcpb = process_models.ccfe_hcpb
    monkeypatch.setattr(ccfe_hcpb.data.physics, "rmajor", 3.0)
    monkeypatch.setattr(ccfe_hcpb.data.tfcoil, "i_tf_sup", 1)

    tf_ref, shield_ref, _total_ref = ccfe_hcpb.st_centrepost_nuclear_heating(
        pneut=400.0, sh_width=0.6
    )
    monkeypatch.setattr(ccfe_hcpb.data.fwbs, "i_cp_shield_material", 1)
    tf_w2b5, shield_w2b5, total_w2b5 = ccfe_hcpb.st_centrepost_nuclear_heating(
        pneut=400.0, sh_width=0.6
    )

    assert tf_w2b5 == pytest.approx(0.37 * tf_ref)
    # the method's fitted shield term is unscaled; note that in the full
    # run() the CP shield heating is overwritten with the residual
    # f_geom_cp * p_neutron - pnuc_cp_tf, so at production level the heat
    # removed from the TF reappears in the shield coolant
    assert shield_w2b5 == pytest.approx(shield_ref)
    assert total_w2b5 == pytest.approx(0.37 * tf_ref + shield_ref)


def test_unknown_material_type_raises():
    """Defence in depth: non-hashable/wrong-type switches raise the
    documented ProcessValueError, not a bare TypeError."""
    with pytest.raises(ProcessValueError):
        CCFE_HCPB.cp_shield_material_heat_factor([2], -1.0)


MEASURED_FLUX_RATIOS = {
    # clean-material depth [m] -> (monolithic W2B5, W2B5+H2O): E > 0.1 MeV
    # neutron-flux ratios vs WC+13% H2O from the neutron-only OpenMC slab
    # campaign of 2026-08-21 (1 cm bins, 4e8 analog histories/material)
    0.255: (0.5435, 0.5227),
    0.305: (0.4697, 0.4820),
    0.355: (0.4041, 0.4446),
    0.465: (0.2842, 0.3708),
    0.545: (0.2275, 0.3292),
}


def test_flux_fit_reproduces_measured_ratios():
    """The fit, evaluated at the clean-material path length the slabs
    measured, must stay within 3% of the neutron-only OpenMC measurements
    across the measured domain (fit residuals are <= 1.7%). The method
    takes the PHYSICAL width and derates by the steel fraction internally,
    so pass depth / (1 - f_steel)."""
    f_steel = CCFE_HCPB.CP_SHIELD_F_STEEL_STRUCT
    for depth, (mono, layered) in MEASURED_FLUX_RATIOS.items():
        f2 = CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, depth / (1 - f_steel))
        f1 = CCFE_HCPB.cp_shield_material_flux_factor(1, -1.0, depth / (1 - f_steel))
        assert f2 == pytest.approx(mono, rel=0.03)
        assert f1 == pytest.approx(layered, rel=0.03)


def test_flux_fit_clamps_outside_validated_domain():
    t_lo, t_hi = CCFE_HCPB.CP_SHIELD_FLUX_FIT_DOMAIN
    f_steel = CCFE_HCPB.CP_SHIELD_F_STEEL_STRUCT
    lo2 = CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, 0.05)
    assert lo2 == CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, t_lo / (1 - f_steel))
    hi2 = CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, 0.90)
    assert hi2 == CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, t_hi / (1 - f_steel))


def test_flux_constant_override_wins_at_any_thickness():
    for depth in (0.10, 0.30, 0.55, 0.90):
        assert CCFE_HCPB.cp_shield_material_flux_factor(2, 0.2, depth) == 0.2


def test_flux_fit_identity_for_wc_baseline():
    for depth in (0.10, 0.30, 0.55):
        assert CCFE_HCPB.cp_shield_material_flux_factor(0, -1.0, depth) == 1.0


def test_flux_fit_unknown_material_raises():
    with pytest.raises(ProcessValueError):
        CCFE_HCPB.cp_shield_material_flux_factor(9, -1.0, 0.4)
