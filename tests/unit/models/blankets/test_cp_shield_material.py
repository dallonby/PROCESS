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
    assert w2b5 == pytest.approx(0.24 * reference)


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
