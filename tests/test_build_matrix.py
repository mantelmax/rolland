# ruff: noqa: N806
"""Tests for build_matrix.py matrix generation logic."""

import numpy as np

from rolland.database.rail.db_rail import UIC60
from rolland.helper.build_matrix import build_rail_matrices


def test_build_rail_matrices_symmetry_and_shape():
    """
    Verify the fundamental properties of the manually constructed stiffness (K).

    And mass (Mr) matrices. Mass matrices must be symmetric and positive definite,
    and K2 (the structural stiffness matrix) must also be symmetric.
    """
    K0, K1, K2, Mr = build_rail_matrices(UIC60, damp_type='viscous')

    # Check shapes
    assert K0.shape == (7, 7)
    assert K1.shape == (7, 7)
    assert K2.shape == (7, 7)
    assert Mr.shape == (7, 7)

    # Mr (Mass Matrix) should be symmetric
    np.testing.assert_allclose(Mr, Mr.T, err_msg='Mass matrix Mr is not symmetric')

    # Mr should be positive definite (diagonals must be strictly positive)
    assert np.all(np.diag(Mr) > 0), 'Mass matrix Mr has non-positive diagonal entries'

    # K2 (Stiffness Matrix) should be symmetric
    np.testing.assert_allclose(K2, K2.T, err_msg='Stiffness matrix K2 is not symmetric')
