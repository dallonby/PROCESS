"""Unit tests for the ST centre-post shield-material option (W2B5)."""

import pytest

from process.core.exceptions import ProcessValueError
from process.models.blankets.hcpb import CCFE_HCPB


def test_default_material_factors_are_identity():
    assert CCFE_HCPB.cp_shield_material_factors(0, -1.0, -1.0) == (1.0, 1.0)


def test_w2b5_defaults():
    """Windsor et al. (2021) Table 2 geometric-mean ratios vs WC+H2O."""
    assert CCFE_HCPB.cp_shield_material_factors(1, -1.0, -1.0) == (0.37, 0.37)
    assert CCFE_HCPB.cp_shield_material_factors(2, -1.0, -1.0) == (0.24, 0.24)


def test_user_override_beats_material_default():
    assert CCFE_HCPB.cp_shield_material_factors(2, 0.3, -1.0) == (0.3, 0.24)
    assert CCFE_HCPB.cp_shield_material_factors(2, -1.0, 0.15) == (0.24, 0.15)


def test_unknown_material_raises():
    with pytest.raises(ProcessValueError):
        CCFE_HCPB.cp_shield_material_factors(7, -1.0, -1.0)


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
    # default is the thickness-dependent fit, evaluated at the clamped
    # top of the validated domain (0.52 m) for this 0.6 m shield
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
        CCFE_HCPB.cp_shield_material_factors([2], -1.0, -1.0)


MEASURED_FLUX_RATIOS = {
    # depth [m] -> (monolithic W2B5, W2B5+H2O), OpenMC verification 2026-08
    0.25: (0.4105, 0.4187),
    0.35: (0.2989, 0.3434),
    0.46: (0.2081, 0.2898),
    0.55: (0.1692, 0.2570),
}


def test_flux_fit_reproduces_measured_ratios():
    """The thickness-dependent fit must stay within 4% of the OpenMC
    measurements across the validated domain (largest residual: layered
    -3.2% at 0.46 m; the 0.55 m points sit beyond the clamp, where the
    held endpoint values land at +2.0% / -2.6%)."""
    for depth, (mono, layered) in MEASURED_FLUX_RATIOS.items():
        f2 = CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, depth)
        f1 = CCFE_HCPB.cp_shield_material_flux_factor(1, -1.0, depth)
        assert f2 == pytest.approx(mono, rel=0.04)
        assert f1 == pytest.approx(layered, rel=0.04)


def test_flux_fit_clamps_outside_validated_domain():
    lo2 = CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, 0.10)
    assert lo2 == CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, 0.25)
    hi2 = CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, 0.90)
    assert hi2 == CCFE_HCPB.cp_shield_material_flux_factor(2, -1.0, 0.52)


def test_flux_constant_override_wins_at_any_thickness():
    for depth in (0.10, 0.30, 0.55, 0.90):
        assert CCFE_HCPB.cp_shield_material_flux_factor(2, 0.2, depth) == 0.2


def test_flux_fit_identity_for_wc_baseline():
    for depth in (0.10, 0.30, 0.55):
        assert CCFE_HCPB.cp_shield_material_flux_factor(0, -1.0, depth) == 1.0


def test_flux_fit_unknown_material_raises():
    with pytest.raises(ProcessValueError):
        CCFE_HCPB.cp_shield_material_flux_factor(9, -1.0, 0.4)
