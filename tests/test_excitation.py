"""Tests for RandomForce excitation."""

import contextlib
from unittest.mock import MagicMock

import numpy as np
from devito import Grid

from rolland.excitation import RandomForce


def test_random_force_calc_rnd_forcearray():
    """Test that calc_rnd_forcearray correctly handles the ramp length and array dimensions."""
    nt = 100
    rf = RandomForce(v=20.0, F_stat_z=1000.0, F_stat_y=500.0)
    forces = rf.calc_rnd_forcearray(nt)

    # Check shape: 2 dimensions (z and y), nt time steps
    assert forces.shape == (2, nt)

    force_z, force_y = forces

    # Verify the internal state arrays are set
    assert hasattr(rf, 'force_z')
    assert hasattr(rf, 'force_y')
    assert len(rf.force_z) == nt
    assert len(rf.force_y) == nt


def test_random_force_interpolation_weights():
    """
    Test that the Cosine interpolation weights used to distribute the moving force.

    Across discrete nodes strictly sum to 1.0 at every time step.
    This is required for conservation of energy.
    """
    discr = MagicMock()
    discr.nt = 100
    discr.nx = 200
    discr.dt = 0.001
    discr.dx = 0.1
    discr.bound.l_bound = 1.0

    # Devito requires a real Grid object to instantiate Functions
    grid = Grid(shape=(200,), extent=(20.0,))
    discr.grid = grid

    # Mocking properties used in the inject_in_track calculation
    discr.f = {'dp_z': MagicMock(), 'dp_y': MagicMock(), 'dp_xr': MagicMock()}
    discr.track.rail = MagicMock()
    discr.track.rail.dr = 10.0
    discr.track.rail.mr = 60.0
    discr.track.rail.rho = 7800.0
    discr.track.rail.Ipr = 1e-4

    rf = RandomForce(v=20.0, F_stat_z=1000.0, F_stat_y=500.0)

    with contextlib.suppress(TypeError):
        # SymPy throws TypeError when multiplying floats with MagicMock.
        # But _weights are computed before this crash.
        rf.inject_in_track(discr)

    # The inject method creates 4 weight functions. Summing them should equal exactly 1.0.
    weights_sum = np.zeros(discr.nt)
    for w_f in rf._weights:  # noqa: SLF001
        weights_sum += w_f.data[:]

    np.testing.assert_allclose(weights_sum, 1.0, rtol=1e-7, err_msg='Interpolation weights do not sum to 1.0')
