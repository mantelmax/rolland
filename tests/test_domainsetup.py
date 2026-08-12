"""Tests for domainsetup.py specifically checking grid dimension logic."""

from unittest.mock import MagicMock

from rolland.boundary import CFSPML
from rolland.domainsetup import DomSetup
from rolland.track import ContSlabSingleRailTrack


def test_domsetup_build_grid():
    """
    Verify that DomSetup accurately translates continuous spatial and temporal.

    Simulation requests into discrete Grid node dimensions (nx) and time steps (nt).
    This ensures that the devito numerical solvers run over the exact intended physical span.
    """
    # Mocking a Track object
    mock_track = MagicMock(spec=ContSlabSingleRailTrack)
    mock_track.l_track = 100.0

    mock_pad = MagicMock()
    mock_pad.sp_z = 300e6
    mock_pad.sp_y = 0
    mock_pad.sp_x = 0
    mock_pad.sp_w = 0
    mock_pad.sp_zr = 0
    mock_pad.sp_yr = 0
    mock_pad.sp_xr = 0
    mock_pad.dp_z = 3000
    mock_pad.dp_y = 0
    mock_pad.dp_x = 0
    mock_pad.dp_xr = 0
    mock_track.pad = mock_pad

    mock_slab = MagicMock()
    mock_slab.ms = 250
    mock_slab.Is_x = 1.0
    mock_slab.Is_y = 1.0
    mock_slab.Is_z = 1.0
    mock_slab.rhos = 2500
    mock_slab.z_st = 0.25
    mock_slab.z_sb = 0.25
    mock_track.slab = mock_slab

    mock_rail = MagicMock()
    mock_rail.G = 1.0
    mock_rail.E = 1.0
    mock_rail.Ar = 1.0
    mock_rail.Iyr = 1.0
    mock_rail.Izr = 1.0
    mock_rail.Iwz = 1.0
    mock_rail.Iwy = 1.0
    mock_rail.Iyz = 1.0
    mock_rail.rho = 1.0
    mock_rail.mr = 1.0
    mock_rail.kapy = 1.0
    mock_rail.kapz = 1.0
    mock_rail.ey = 1.0
    mock_rail.ez = 1.0
    mock_rail.Iw = 1.0
    mock_rail.J = 1.0
    mock_rail.J_t = 1.0
    mock_rail.Ipr = 1.0
    mock_rail.dr = 1.0
    mock_track.rail = mock_rail
    mock_track.z_f = 0.0
    mock_track.y_f = 0.0

    # Mocking a CFSPML boundary object
    mock_bound = MagicMock(spec=CFSPML)
    mock_bound.l_bound = 10.0
    mock_bound.initialize_on_grid.return_value = (None, None)
    mock_bound.apply_pml.return_value = (1, 1, [])

    # Setup DomSetup
    ds = DomSetup(
        track=mock_track,
        bound=mock_bound,
        dt=0.5e-5,
        req_simt=0.1,
        dx=0.05,
    )

    # Check if grid dimensions are calculated properly
    # For time: nt = int(req_simt / dt) = int(0.1 / 0.5e-5) = 20000
    # For space: nx = int(l_track / dx) = int(100.0 / 0.05) = 2000
    # A mismatch here will cause numerical blowups or incorrect durations.
    assert ds.nt == 20000
    assert ds.nx == 2000

    # Verify the grids are instantiated with the correct extents.
    # `extent` should match the physical physical track length (100.0)
    # and `shape` must be exactly the calculated number of spatial nodes.
    assert ds.grid.extent == (100.0,)
    assert ds.grid.shape == (2000,)

    assert ds.bd_grid.extent == (100.0,)
    assert ds.bd_grid.shape == (2000,)
