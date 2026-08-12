# ruff: noqa: N806

"""Defines Domain setup classes for FDM simulation.

.. autosummary::
    :toctree: domainsetup

    DomSetup
"""

from dataclasses import dataclass

from devito import Border, Constant, Eq, Function, Grid, TimeFunction, solve
from numpy import float64, linspace

from .boundary import CFSPML
from .track import (
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    DiscrSlabSingleRailTrack,
    Track,
)

RIGID_PENALTY = 1e20
EPSILON = 1e-20

@dataclass(kw_only=True)
class DomSetup:
    r"""Set up the computational domain.

    This class manages the translation of physical track properties into a finite-difference
    computational grid. It builds the DEVITO grids, applies Perfectly Matched Layer (PML)
    boundaries, and constructs the explicit update equations (operators) for the simulation.

    Attributes
    ----------
    track : Track
        The track instance containing physical parameters.
    bound : CFSPML
        The boundary damping instance handling wave absorption at the edges.
    dt : float, default=0.5e-5
        Step size in time :math:`[s]`.
    req_simt : float, default=0.1
        Requested simulation time :math:`[s]`.
    dx : float, default=0.05
        Spatial step size :math:`[m]`.
    nt : int
        Calculated number of time steps :math:`[-]`.
    nx : int
        Calculated number of spatial steps :math:`[-]`.
    grid : devito.Grid
        The main DEVITO computational grid.
    bd_grid : devito.Grid
        Secondary DEVITO grid dedicated to the boundary domains.
    bound_dom : Border
        The domain definition used for applying boundary updates.
    op_track : list of devito.Eq
        The list of DEVITO update equations forming the core simulation operator.
    u_z : devito.TimeFunction
        The continuous time function representing vertical displacement.
    u_y : devito.TimeFunction
        The continuous time function representing lateral displacement.
    phi_x : devito.TimeFunction
        The continuous time function representing torsional rotation.
    f : dict[str, devito.Function]
        Dictionary mapping property names to their corresponding DEVITO spatial functions.
    """

    track: Track
    bound: CFSPML
    dt: float = 0.5e-5
    req_simt: float = 0.1
    dx: float = 0.05

    def __post_init__(self, *args, **kwargs):
        """Initialize discretization internals by building the grid and operator."""
        self.build_grid()
        self.build_operator()

    def build_grid(self):
        """Build the Devito computational grid.

        Initializes `nt`, `nx`, `grid`, `bd_grid`, and `bound_dom` attributes.
        """
        self.nt = int(self.req_simt / self.dt)
        self.nx = int(self.track.l_track / self.dx)

        self.grid = Grid(
            shape=(self.nx,),
            extent=(self.track.l_track,),
            dtype=float64,
            origin=(0.0,),
        )

        self.bd_grid = Grid(
            shape=(self.nx,),
            extent=(self.track.l_track,),
            dtype=float64,
            origin=(0.0,),
        )

        # Define Boundary Domain
        _nx_bound = int(self.bound.l_bound / self.dx)
        (x,) = self.bd_grid.dimensions
        # Assuming Border is imported or available in scope
        self.bound_dom = Border(grid=self.bd_grid, border=_nx_bound, dims=x)

    def _setup_track_interpolation(self, f, track):
        """Set up spatial functions with track properties and handle interpolation.

        Parameters
        ----------
        f : dict[str, devito.Function]
            Dictionary of spatial property functions.
        track : Track
            The track model instance.

        Returns
        -------
        tuple
            A tuple containing (is_slab, rho_s_val, z_st_val, z_sb_val, Ex, Ez).
        """
        is_cont_slab_or_ballast = isinstance(track, (ContSlabSingleRailTrack, ContBallastedSingleRailTrack))
        is_slab = isinstance(track, (ContSlabSingleRailTrack, DiscrSlabSingleRailTrack))

        if is_cont_slab_or_ballast:
            seclay = track.slab
        else:
            mount_pos = list(track.mount_prop.keys())
            patterns = track.get_mount_patterns(linspace(0, track.l_track, self.nx), self.dx, mount_pos)

        # Vectorized property assignment
        pad_vars = ['sp_z', 'sp_y', 'sp_x', 'sp_w', 'sp_zr', 'sp_yr', 'sp_xr', 'dp_z', 'dp_y', 'dp_x', 'dp_xr']
        for var in pad_vars:
            if is_cont_slab_or_ballast:
                f[var].data[:] = getattr(track.pad, var)
            else:
                f[var].data[:] = sum(getattr(track.mount_prop[pos][0], var) * patterns[pos] for pos in mount_pos)

        ballast_vars = [
            'sb_x', 'sb_z', 'sb_y', 'sb_xr', 'sb_zr', 'sb_yr',
            'db_x', 'db_z', 'db_y', 'db_xr', 'db_zr', 'db_yr',
        ]

        if is_slab:
            for var in ballast_vars:
                f[var].data[:] = 0

            f['ms'].data[:] = RIGID_PENALTY
            f['Is_x'].data[:] = RIGID_PENALTY
            f['Is_y'].data[:] = RIGID_PENALTY
            f['Is_z'].data[:] = RIGID_PENALTY

            rho_s_val, z_st_val, z_sb_val = RIGID_PENALTY, 0, 0
            Ex = Constant(name='Ex', value=1)
            Ez = Constant(name='Ez', value=1)
        else:
            for var in ballast_vars:
                if is_cont_slab_or_ballast:
                    f[var].data[:] = getattr(track.ballast, var)
                else:
                    f[var].data[:] = sum(getattr(track.mount_prop[pos][2], var) * patterns[pos] for pos in mount_pos)

            if is_cont_slab_or_ballast:
                f['ms'].data[:] = seclay.ms + EPSILON
                f['Is_x'].data[:] = seclay.Is_x + EPSILON
                f['Is_y'].data[:] = seclay.Is_y + EPSILON
                f['Is_z'].data[:] = seclay.Is_z + EPSILON
                rho_s_val, z_st_val, z_sb_val = seclay.rhos, seclay.z_st, seclay.z_sb
            else:
                f['ms'].data[:] = sum(track.mount_prop[pos][1].ms * patterns[pos] for pos in mount_pos) + EPSILON
                f['Is_x'].data[:] = sum(track.mount_prop[pos][1].Is_x * patterns[pos] for pos in mount_pos) + EPSILON
                f['Is_y'].data[:] = sum(track.mount_prop[pos][1].Is_y * patterns[pos] for pos in mount_pos) + EPSILON
                f['Is_z'].data[:] = sum(track.mount_prop[pos][1].Is_z * patterns[pos] for pos in mount_pos) + EPSILON
                # Take these scalar values from the first sleeper since they are constants per
                # equation for now or assume they are uniform
                first_sleeper = track.mount_prop[mount_pos[0]][1]
                rho_s_val, z_st_val, z_sb_val = first_sleeper.rhos, first_sleeper.z_st, first_sleeper.z_sb

            Ex = Constant(name='Ex', value=track.E[0])
            Ez = Constant(name='Ez', value=track.E[1])

        return is_slab, rho_s_val, z_st_val, z_sb_val, Ex, Ez

    def build_operator(self):
        """Build the track operator for the simulation.

        Initializes dynamic spatial functions (`f`), sets up wave equations and ADE-PML
        updates, and populates `op_track`, `u_z`, `u_y`, and `phi_x`.
        """
        # --- 1. Dynamic Spatial Function Initialization ---
        func_names = [
            'sp_z', 'sp_y', 'sp_x', 'sp_w', 'sp_zr', 'sp_yr', 'sp_xr',
            'sb_x', 'sb_z', 'sb_y', 'sb_xr', 'sb_zr', 'sb_yr',
            'dp_z', 'dp_y', 'dp_x', 'dp_xr',
            'db_x', 'db_z', 'db_y', 'db_xr', 'db_zr', 'db_yr',
            'ms', 'Is_x', 'Is_y', 'Is_z',
        ]
        f = {name: Function(name=name, grid=self.grid, dtype=float64) for name in func_names}
        self.f = f

        # Damping initialization
        sigm, alph = self.bound.initialize_on_grid(self.grid, self.dx, self.nx)

        # --- 2. Track & Interpolation Logic ---
        track = self.track
        is_slab, rho_s_val, z_st_val, z_sb_val, Ex, Ez = self._setup_track_interpolation(f, track)

        # --- 3. Primary State Time Functions ---
        u_x = TimeFunction(name='u_x', grid=self.grid, time_order=2, space_order=6, dtype=float64, save=None)
        u_y = TimeFunction(name='u_y', grid=self.grid, time_order=2, space_order=6, dtype=float64, save=None)
        u_z = TimeFunction(name='u_z', grid=self.grid, time_order=2, space_order=6, dtype=float64, save=None)
        phi_x = TimeFunction(name='phi_x', grid=self.grid, time_order=2, space_order=6, dtype=float64, save=None)
        phi_y = TimeFunction(name='phi_y', grid=self.grid, time_order=2, space_order=6, dtype=float64, save=None)
        phi_z = TimeFunction(name='phi_z', grid=self.grid, time_order=2, space_order=6, dtype=float64, save=None)
        u_w = TimeFunction(name='u_w', grid=self.grid, time_order=2, space_order=6, dtype=float64, save=None)

        # Sleeper variables
        u_sx = TimeFunction(name='u_sx', grid=self.grid, time_order=2, dtype=float64, save=None)
        u_sz = TimeFunction(name='u_sz', grid=self.grid, time_order=2, dtype=float64, save=None)
        u_sy = TimeFunction(name='u_sy', grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sx = TimeFunction(name='phi_sx', grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sz = TimeFunction(name='phi_sz', grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sy = TimeFunction(name='phi_sy', grid=self.grid, time_order=2, dtype=float64, save=None)

        # --- 4. ADE-PML Boundary Calculations ---
        pml_vars = [
            (u_x, 'ux'),
            (u_z, 'uz'),
            (phi_y, 'phiy'),
            (u_y, 'uy'),
            (phi_z, 'phiz'),
            (phi_x, 'phix'),
            (u_w, 'uw'),
        ]

        pml_dx, pml_dx2 = {}, {}
        pml_updates = []

        for var, suffix in pml_vars:
            # Call the new method on self.bound
            dx, dx2, updates = self.bound.apply_pml(var, suffix, self.grid,
                                                    self.bound_dom, sigm, alph)
            pml_dx[suffix] = dx
            pml_dx2[suffix] = dx2
            pml_updates.extend(updates)

        # --- 5. Define Explicit Constants ---
        rail = self.track.rail
        G = Constant(name='G', value=rail.G)
        E = Constant(name='E', value=rail.E)
        A = Constant(name='A', value=rail.Ar)
        Iy = Constant(name='I_y', value=rail.Iyr)
        Iz = Constant(name='I_z', value=rail.Izr)
        Iwz = Constant(name='I_wz', value=rail.Iwz)
        Iwy = Constant(name='I_wy', value=rail.Iwy)
        Iyz = Constant(name='I_yz', value=rail.Iyz)
        rho = Constant(name='rho', value=rail.rho)
        mr = Constant(name='m_r', value=rail.mr)
        kap_y = Constant(name='kappa_y', value=rail.kapy)
        kap_z = Constant(name='kappa_z', value=rail.kapz)
        e_y = Constant(name='e_y', value=rail.ey)
        e_z = Constant(name='e_z', value=rail.ez)
        chi_f = Constant(name='chi_f', value=0)
        Iw = Constant(name='I_w', value=rail.Iw)
        J = Constant(name='J', value=rail.J)
        Jt = Constant(name='J_t', value=rail.J_t)
        Ip = Constant(name='I_p', value=rail.Ipr)
        z_f = Constant(name='z_f', value=track.z_f)
        y_f = Constant(name='y_f', value=track.y_f)
        z_st = Constant(name='z_st', value=z_st_val)
        z_sb = Constant(name='z_sb', value=z_sb_val)
        rho_s = Constant(name='rho_s', value=rho_s_val)

        # --- 6. Physics Equations ---

        # Longitudinal Wave
        longw = Eq(
            -A * E * pml_dx2['ux']
            + chi_f * f['dp_x'] * u_w.backward.dt
            + chi_f * f['sp_x'] * u_w
            + f['dp_x'] * y_f * phi_z.backward.dt
            - f['dp_x'] * z_f * phi_y.backward.dt
            - f['dp_x'] * u_sx.backward.dt
            + (f['dp_x'] + rail.dr) * u_x.dt
            + mr * u_x.dt2
            + f['sp_x'] * y_f * phi_z
            - f['sp_x'] * z_f * phi_y
            - f['sp_x'] * u_sx
            + f['sp_x'] * u_x,
        )

        # Timoshenko Beam Wave - Sum of Vertical Forces
        tbw_vert_trans = Eq(
            A * G * e_y * kap_z * pml_dx2['phix']
            # + A * G * e_y * kap_z * pml_dx['uw'] --> neglected
            + A * G * kap_z * pml_dx['phiy']
            - A * G * kap_z * pml_dx2['uz']
            + mr * u_z.dt2
            - f['sp_z'] * y_f * phi_x
            - f['sp_z'] * u_sz
            + f['sp_z'] * u_z
            - y_f * f['dp_z'] * phi_x.backward.dt
            + (-f['dp_z']) * u_sz.backward.dt
            + (f['dp_z'] + rail.dr) * u_z.dt,
        )

        # Timoshenko Beam Wave - Sum of Lateral Forces
        tbw_lat_trans = Eq(
            -A * G * e_z * kap_y * pml_dx2['phix']
            # - A * G * e_z * kap_y * pml_dx['uw'] --> neglected
            - A * G * kap_y * pml_dx['phiz']
            - A * G * kap_y * pml_dx2['uy']
            + mr * u_y.dt2
            + f['sp_y'] * z_f * phi_x
            - f['sp_y'] * z_st * phi_sx
            - f['sp_y'] * u_sy
            + f['sp_y'] * u_y
            + z_f * f['dp_y'] * phi_x.backward.dt
            + z_st * (-f['dp_y']) * phi_sx.backward.dt
            + (-f['dp_y']) * u_sy.backward.dt
            + (f['dp_y'] + rail.dr) * u_y.dt,
        )

        # Torsional Beam Wave (Rotation about x-axis)
        torsw = Eq(
            -A * G * e_y * kap_z * pml_dx['phiy']
            + A * G * e_y * kap_z * pml_dx2['uz']
            - A * G * e_z * kap_y * pml_dx['phiz']
            - A * G * e_z * kap_y * pml_dx2['uy']
            - G * Jt * pml_dx['uw']
            - G * (J + Jt) * pml_dx2['phix']
            + Ip * rho * phi_x.dt2
            - f['sp_y'] * z_f * u_sy
            + f['sp_y'] * z_f * u_y
            + f['sp_z'] * y_f * u_sz
            - f['sp_z'] * y_f * u_z
            + y_f * f['dp_z'] * u_sz.backward.dt
            - y_f * f['dp_z'] * u_z.dt
            - z_f * f['dp_y'] * u_sy.backward.dt
            + z_f * f['dp_y'] * u_y.dt
            + (-f['dp_xr'] - z_f * z_st * f['dp_y']) * phi_sx.backward.dt
            + (-f['sp_xr'] - f['sp_y'] * z_f * z_st) * phi_sx
            + (f['dp_xr'] + y_f**2 * f['dp_z'] + z_f**2 * f['dp_y']) * phi_x.dt
            + (f['sp_xr'] + f['sp_y'] * z_f**2 + f['sp_z'] * y_f**2) * phi_x,
        )

        # Timoshenko Beam Wave - Sum of Lateral Moments
        tbw_lat_rot = Eq(
            +A * G * e_z * kap_y * pml_dx['phix']
            + A * G * kap_y * pml_dx['uy']
            # - E * Iwz * pml_dx2['uw'] --> neglected
            + E * Iyz * pml_dx2['phiy']
            - E * Iz * pml_dx2['phiz']
            # + Iwz * rho * u_w.backward.dt2 --> neglected
            - Iyz * rho * phi_y.backward.dt2
            # + chi_f * f['dp_x'] * y_f * u_w.backward.dt --> neglected
            + f['dp_x'] * y_f**2 * phi_z.dt
            - f['dp_x'] * y_f * z_f * phi_y.backward.dt
            - f['dp_x'] * y_f * u_sx.backward.dt
            + f['dp_x'] * y_f * u_x.dt
            + rho * (-Iwz + Iz) * phi_z.dt2
            - f['sp_x'] * y_f * z_f * phi_y
            - f['sp_x'] * y_f * u_sx
            + f['sp_x'] * y_f * u_x
            - f['sp_zr'] * phi_sz
            # (A * G * e_z * kap_y + chi_f * f['sp_x'] * y_f * u_w) * u_w --> neglected
            + (A * G * kap_y + f['sp_x'] * y_f**2 + f['sp_zr']) * phi_z,
        )

        # Timoshenko Beam Wave - Sum of Vertical Moments
        tbw_vert_rot = Eq(
            A * G * e_y * kap_z * pml_dx['phix']
            - A * G * kap_z * pml_dx['uz']
            # + E * Iwy * pml_dx2['uw'] --> neglected
            - E * Iy * pml_dx2['phiy']
            + E * Iyz * pml_dx2['phiz']
            # - Iwy * rho * u_w.backward.dt2 --> neglected
            - Iyz * rho * phi_z.dt2
            # - chi_f * f['dp_x'] * z_f * u_w.backward.dt --> neglected
            - f['dp_x'] * y_f * z_f * phi_z.dt
            + f['dp_x'] * z_f**2 * phi_y.dt
            + f['dp_x'] * z_f * u_sx.backward.dt
            - f['dp_x'] * z_f * u_x.dt
            + rho * (Iwy + Iy) * phi_y.dt2
            - f['sp_x'] * y_f * z_f * phi_z
            + f['sp_x'] * z_f * u_sx
            - f['sp_x'] * z_f * u_x
            - f['sp_yr'] * phi_sy
            # + (A * G * e_y * kap_z - chi_f * f['sp_x'] * z_f) * u_w --> neglected
            + (A * G * kap_z + f['sp_x'] * z_f**2 + f['sp_yr']) * phi_y,
        )

        # Warping Equation
        warp = Eq(
            -A * G * e_y * kap_z * pml_dx['uz']
            + A * G * e_z * kap_y * pml_dx['uy']
            - E * Iw * pml_dx2['uw']
            + E * Iwy * pml_dx2['phiy']
            - E * Iwz * pml_dx2['phiz']
            + G * Jt * pml_dx['phix']
            + Iw * rho * u_w.dt2
            - Iwy * rho * phi_y.dt2
            + Iwz * rho * phi_z.dt2
            + chi_f**2 * f['dp_x'] * u_w.dt
            + chi_f * f['dp_x'] * y_f * phi_z.dt
            - chi_f * f['dp_x'] * z_f * phi_y.dt
            - chi_f * f['dp_x'] * u_sx.backward.dt
            + chi_f * f['dp_x'] * u_x.dt
            - chi_f * f['sp_x'] * u_sx
            + chi_f * f['sp_x'] * u_x
            + (A * G * e_y * kap_z - chi_f * f['sp_x'] * z_f) * phi_y
            + (A * G * e_z * kap_y + chi_f * f['sp_x'] * y_f) * phi_z
            + (G * Jt + chi_f**2 * f['sp_x'] + f['sp_w']) * u_w,
        )

        upd_longw = Eq(u_x.forward, solve(longw, u_x.forward))
        upd_tbw_vert_trans = Eq(u_z.forward, solve(tbw_vert_trans, u_z.forward))
        upd_tbw_lat_trans = Eq(u_y.forward, solve(tbw_lat_trans, u_y.forward))
        upd_torsw = Eq(phi_x.forward, solve(torsw, phi_x.forward))
        upd_tbw_vert_rot = Eq(phi_y.forward, solve(tbw_vert_rot, phi_y.forward))
        upd_tbw_lat_rot = Eq(phi_z.forward, solve(tbw_lat_rot, phi_z.forward))
        upd_warp = Eq(u_w.forward, solve(warp, u_w.forward))

        op_track = [
            upd_longw,
            upd_tbw_vert_trans,
            upd_tbw_lat_trans,
            upd_torsw,
            upd_tbw_lat_rot,
            upd_tbw_vert_rot,
            upd_warp,
        ] + pml_updates

        if not is_slab:
            # Translational Sleeper Equation (x-direction)
            slep_trans_x = Eq(
                -chi_f * f['dp_x'] * u_w.dt
                - chi_f * f['sp_x'] * u_w
                - f['dp_x'] * y_f * phi_z.dt
                + f['dp_x'] * z_f * phi_y.dt
                - f['dp_x'] * u_x.dt
                - f['sp_x'] * y_f * phi_z
                + f['sp_x'] * z_f * phi_y
                - f['sp_x'] * u_x
                + (f['sp_x'] + f['sb_x'] * Ex) * u_sx
                + (f['db_x'] * Ex + f['dp_x']) * u_sx.dt
                + f['ms'] * u_sx.dt2 * Ex,
            )

            # Translational Sleeper Equation (z-direction)
            slep_trans_z = Eq(
                f['sp_z'] * y_f * phi_x
                - f['sp_z'] * u_z
                - y_f * (-f['dp_z']) * phi_x.dt
                + (-f['dp_z']) * u_z.dt
                + (f['sp_z'] + f['sb_z'] * Ez) * u_sz
                + (f['db_z'] * Ez + f['dp_z']) * u_sz.dt
                + f['ms'] * u_sz.dt2 * Ez,
            )

            # Translational Sleeper Equation (y-direction)
            slep_trans_y = Eq(
                f['ms'] * u_sy.dt2
                - f['sp_y'] * z_f * phi_x
                - f['sp_y'] * u_y
                + z_f * (-f['dp_y']) * phi_x.dt
                + (-f['dp_y']) * u_y.dt
                + (f['sb_y'] + f['sp_y']) * u_sy
                + (f['db_y'] * z_sb + z_st * f['dp_y']) * phi_sx.dt
                + (f['sb_y'] * z_sb + f['sp_y'] * z_st) * phi_sx
                + (f['db_y'] + f['dp_y']) * u_sy.dt,
            )

            # Rotational Sleeper Equation (x-axis)
            slep_rot_x = Eq(
                f['Is_x'] * rho_s * phi_sx.dt2
                - f['sp_y'] * z_st * u_y
                - z_st * f['dp_y'] * u_y.dt
                + (-f['dp_xr'] - z_f * z_st * f['dp_y']) * phi_x.dt
                + (-f['sp_xr'] - f['sp_y'] * z_f * z_st) * phi_x
                + (f['db_y'] * z_sb + z_st * f['dp_y']) * u_sy.dt
                + (f['sb_y'] * z_sb + f['sp_y'] * z_st) * u_sy
                + (f['db_xr'] + f['db_y'] * z_sb**2 + f['dp_xr'] + z_st**2 * f['dp_y']) * phi_sx.dt
                + (f['sb_xr'] + f['sb_y'] * z_sb**2 + f['sp_xr'] + f['sp_y'] * z_st**2) * phi_sx,
            )

            # Rotational Sleeper Equation (z-axis)
            slep_rot_z = Eq(
                f['Is_z'] * rho_s * phi_sz.dt2 + f['db_zr'] * phi_sz.dt - f['sp_zr'] * phi_z
                + (f['sb_zr'] + f['sp_zr']) * phi_sz,
            )

            # Rotational Sleeper Equation (y-axis)
            slep_rot_y = Eq(
                f['Is_y'] * rho_s * phi_sy.dt2 + f['db_yr'] * phi_sy.dt - f['sp_yr'] * phi_y
                + (f['sb_yr'] + f['sp_yr']) * phi_sy,
            )

            upd_u_sx = Eq(u_sx.forward, solve(slep_trans_x, u_sx.forward))
            upd_u_sz = Eq(u_sz.forward, solve(slep_trans_z, u_sz.forward))
            upd_u_sy = Eq(u_sy.forward, solve(slep_trans_y, u_sy.forward))
            upd_phi_sx = Eq(phi_sx.forward, solve(slep_rot_x, phi_sx.forward))
            upd_phi_sz = Eq(phi_sz.forward, solve(slep_rot_z, phi_sz.forward))
            upd_phi_sy = Eq(phi_sy.forward, solve(slep_rot_y, phi_sy.forward))

            sleeper_updates = [
                upd_u_sx,
                upd_u_sz,
                upd_u_sy,
                upd_phi_sx,
                upd_phi_sz,
                upd_phi_sy,
            ]
            op_track.extend(sleeper_updates)

        self.op_track = op_track
        self.u_z = u_z
        self.u_y = u_y
        self.phi_x = phi_x
