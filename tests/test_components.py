"""Tests for components.py dataclasses validation and derived properties."""

import pytest

from rolland.components import ContPad, DiscrPad, _damping_mode


def test_damping_mode_hysteretic():
    """
    Ensure the damping mode logic correctly identifies hysteretic damping.

    This is important because hysteretic damping requires purely loss factors (eta)
    and any missing or viscous values should invalidate this mode.
    """
    # Only loss factors provided
    mode = _damping_mode(
        eta_values=[0.1, 0.1, 0.1, 0.1],
        viscous_values=[None, None, None, None],
        eta_name='Loss factors',
        viscous_name='viscous damping coefficients',
    )
    assert mode == 'hysteretic'


def test_damping_mode_viscous():
    """
    Ensure the damping mode logic correctly identifies viscous damping.

    Viscous modes calculate energy loss differently, relying purely on viscous
    coefficients instead of loss factors.
    """
    # Only viscous coefficients provided
    mode = _damping_mode(
        eta_values=[None, None, None, None],
        viscous_values=[100, 100, 100, 100],
        eta_name='Loss factors',
        viscous_name='viscous damping coefficients',
    )
    assert mode == 'viscous'


def test_damping_mode_missing_both():
    """
    Verify that an exception is raised when neither damping parameter set is fully provided.

    This prevents running a simulation with an undefined energy dissipation mechanism.
    """
    with pytest.raises(ValueError, match='Both Loss factors and viscous damping coefficients are missing'):
        _damping_mode([None, None], [None, None], 'Loss factors', 'viscous damping coefficients')


def test_damping_mode_both_provided():
    """Verify that an exception is raised when both damping parameters are provided."""
    with pytest.raises(ValueError, match='Both Loss factors and viscous damping coefficients are provided'):
        _damping_mode([0.1, 0.1], [100, 100], 'Loss factors', 'viscous damping coefficients')


def test_damping_mode_partially_missing_eta():
    """Verify that an exception is raised when loss factors are partially missing."""
    with pytest.raises(ValueError, match='Loss factors values are missing'):
        _damping_mode([0.1, None], [None, None], 'Loss factors', 'viscous damping coefficients')


def test_discr_pad_post_init():
    """
    Check that a DiscrPad instance correctly derives rotational stiffness (sp_xr, sp_yr, sp_zr).

    Based on the base stiffness values and the pad width. This validation ensures
    subsequent track physics calculations are based on accurate geometry.
    """
    pad = DiscrPad(sp_z=300e6, sp_y=100e6, sp_x=50e6, etap_z=0.25, etap_y=0.25, etap_x=0.25, etap_r=0.25, wdthp=0.5)
    # Check derived attributes
    assert pad.sp_xr == 300e6 * (0.5**2) / 12.0
    assert pad.sp_yr == 300e6 * (0.5**2) / 12.0
    assert pad.sp_zr == (100e6 + 50e6) * (0.5**2) / 12.0
    assert pad.damping_mode == 'hysteretic'


def test_cont_pad_post_init():
    """
    Check that a ContPad calculates derived attributes accurately.

    And also identifies its correct damping mode ("viscous") automatically based on inputs.
    """
    pad = ContPad(sp_z=300e6, sp_y=100e6, sp_x=50e6, dp_z=3000, dp_y=1000, dp_x=500, dp_xr=100, wdthp=0.5)
    # Check derived attributes
    assert pad.sp_xr == 300e6 * (0.5**2) / 12.0
    assert pad.sp_yr == 300e6 * (0.5**2) / 12.0
    assert pad.sp_zr == (100e6 + 50e6) * (0.5**2) / 12.0
    assert pad.damping_mode == 'viscous'
