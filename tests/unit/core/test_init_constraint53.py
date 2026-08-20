"""Validation tests: constraint 53 must be rejected when no model calculates
flu_tf_neutron_fast_peak (it stays 0.0 and the constraint is silently
satisfied)."""

import numpy as np
import pytest

from process.core.exceptions import ProcessValidationError
from process.core.init import check_process
from process.core.model import DataStructure
from process.models.tfcoil.base import TFConductorModel


def _base_data():
    data = DataStructure()
    # minimal consistent numerics so earlier checks pass
    data.numerics.icc = np.array([53] + [0] * 9)
    data.numerics.n_equality_constraints = 0
    data.numerics.n_inequality_constraints = 1
    # a set thickness so unrelated TF validation does not fire first
    data.build.dr_tf_inboard = 1.0
    return data


def test_constraint_53_rejected_for_conventional_tokamak():
    data = _base_data()
    data.physics.itart = 0
    data.stellarator.istell = 0
    data.tfcoil.i_tf_sup = TFConductorModel.SUPERCONDUCTING
    with pytest.raises(ProcessValidationError, match="flu_tf_neutron_fast_peak"):
        check_process(None, data)


def test_constraint_53_rejected_for_resistive_st():
    data = _base_data()
    data.physics.itart = 1
    data.stellarator.istell = 0
    data.tfcoil.i_tf_sup = TFConductorModel.WATER_COOLED_COPPER
    with pytest.raises(ProcessValidationError, match="flu_tf_neutron_fast_peak"):
        check_process(None, data)


def test_constraint_53_rejected_for_aluminium_st():
    data = _base_data()
    data.physics.itart = 1
    data.stellarator.istell = 0
    data.tfcoil.i_tf_sup = TFConductorModel.HELIUM_COOLED_ALUMINIUM
    with pytest.raises(ProcessValidationError, match="flu_tf_neutron_fast_peak"):
        check_process(None, data)


def test_constraint_53_rejected_for_dcll_superconducting_st():
    """The fluence writer is the CCFE HCPB centre-post path; a DCLL blanket
    (i_blanket_type=5) never calculates it, so constraint 53 must be
    rejected even for a superconducting ST."""
    data = _base_data()
    data.physics.itart = 1
    data.stellarator.istell = 0
    data.tfcoil.i_tf_sup = TFConductorModel.SUPERCONDUCTING
    data.fwbs.i_blanket_type = 5
    with pytest.raises(ProcessValidationError, match="flu_tf_neutron_fast_peak"):
        check_process(None, data)


def _message_or_empty(data):
    try:
        check_process(None, data)
    except ProcessValidationError as error:
        return str(error)
    return ""


def test_constraint_53_accepted_for_superconducting_hcpb_st():
    data = _base_data()
    data.physics.itart = 1
    data.stellarator.istell = 0
    data.tfcoil.i_tf_sup = TFConductorModel.SUPERCONDUCTING
    data.fwbs.i_blanket_type = 1
    assert "flu_tf_neutron_fast_peak" not in _message_or_empty(data)


def test_constraint_53_accepted_for_stellarator():
    data = _base_data()
    data.physics.itart = 0
    data.stellarator.istell = 1
    assert "flu_tf_neutron_fast_peak" not in _message_or_empty(data)
