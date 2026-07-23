"""Defines discretization classes for FDM simulation.

.. autosummary::
    :toctree: discretization

    Discretization
    DiscretizationEBBVertic
    DiscretizationEBBVerticConst
    DiscretizationEBBVerticTimeDepend
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

from devito import Border, Constant, Eq, Function, Grid, TimeFunction, solve
from numpy import float64, linspace, ones, sqrt, zeros
from scipy.sparse import SparseEfficiencyWarning, csc_matrix, diags, eye

from .boundary import DevitoPMLDamp
from .excitation import Excitation
from .track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    DiscrSlabSingleRailTrack,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
    Track,
)


class Discretization(ABC):
    r"""Abstract base class for discretization classes."""

    @abstractmethod
    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""


@dataclass(kw_only=True)
class DiscretizeDEVITO(Discretization):
    r"""Descritize DEVITO.

    Attributes
    ----------
    track : Track
        Track instance.
    dt : float, default=0.5e-5
        Step size in time :math:`[s]`.
    req_simt : float, default=0.1
        Requested simulation time :math:`[s]`.
    nt : int
        Number of time steps :math:`[-]`.
    sim_t : float
        Actual simulation time :math:`[s]`.
    nx : int
        Number of spatial steps :math:`[-]`.
    dx : float, default=0.05
        Spatial step size :math:`[m]`.
    bound : PMLRailDampVertic
        Boundary instance.
    grid : devito.Grid
        Devito grid instance.
    z_f : float, default=0.0
        Excitation excentricity in z-direction :math:`[m]`.
    y_f : float, default=0.0
        Excitation excentricity in y-direction :math:`[m]`.
    store : str, default="point"
        Decides whether to store deflection only at the excitation position ('point') or not.
    equi_sm : bool, default=True
        If True, equivalent sleeper model is applied
    y_sc : float, default=0.7175
        Distance from excitation point to sleeper center :math:`[m]`.
    """

    track: Track
    bound: DevitoPMLDamp
    dt: float = 0.5e-5
    req_simt: float = 0.1
    dx: float = 0.05
    z_f: float = 0.0
    y_f: float = 0.0
    store: str = "point"
    equi_sm: bool = True
    y_sc: float = 0.7175

    def __post_init__(self, *args, **kwargs):
        """Post-initialization method to build the Devito grid."""
        self.build_grid()

    def build_grid(self):
        """Build the Devito computational grid."""
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
        self.bound_dom = Border(grid=self.bd_grid, border=_nx_bound, dims=x)


    def build_operator(self):
        """Build the Devito operator for the simulation."""
        sp_z = Function(name="sp_z", grid=self.grid, dtype=float64)
        sp_y = Function(name="sp_y", grid=self.grid, dtype=float64)
        sp_x = Function(name="sp_x", grid=self.grid, dtype=float64)
        sp_w = Function(name="sp_w", grid=self.grid, dtype=float64)
        sp_zr = Function(name="sp_zr", grid=self.grid, dtype=float64)
        sp_yr = Function(name="sp_yr", grid=self.grid, dtype=float64)
        sp_xr = Function(name="sp_xr", grid=self.grid, dtype=float64)
        sb_x = Function(name="sb_x", grid=self.grid, dtype=float64)
        sb_z = Function(name="sb_z", grid=self.grid, dtype=float64)
        sb_y = Function(name="sb_y", grid=self.grid, dtype=float64)
        sb_xr = Function(name="sb_xr", grid=self.grid, dtype=float64)
        sb_zr = Function(name="sb_zr", grid=self.grid, dtype=float64)
        sb_yr = Function(name="sb_yr", grid=self.grid, dtype=float64)
        dp_z = Function(name="dp_z", grid=self.grid, dtype=float64)
        dp_y = Function(name="dp_y", grid=self.grid, dtype=float64)
        dp_x = Function(name="dp_x", grid=self.grid, dtype=float64)
        dp_xr = Function(name="dp_xr", grid=self.grid, dtype=float64)
        db_x = Function(name="db_x", grid=self.grid, dtype=float64)
        db_z = Function(name="db_z", grid=self.grid, dtype=float64)
        db_y = Function(name="db_y", grid=self.grid, dtype=float64)
        db_xr = Function(name="db_xr", grid=self.grid, dtype=float64)
        db_zr = Function(name="db_zr", grid=self.grid, dtype=float64)
        db_yr = Function(name="db_yr", grid=self.grid, dtype=float64)
        ms = Function(name="ms", grid=self.grid, dtype=float64)
        moinnert_s_x = Function(name="Is_x", grid=self.grid, dtype=float64)
        moinnert_s_y = Function(name="Is_y", grid=self.grid, dtype=float64)
        moinnert_s_z = Function(name="Is_z", grid=self.grid, dtype=float64)


        track = self.track
        if isinstance(track, (ContSlabSingleRailTrack, ContBallastedSingleRailTrack)):
            mount_pos_interp = 1
            seclay = track.slab

        else:
            mount_pos = list(track.mount_prop.keys())
            mount_pos_interp = track.interpol_pad_width(
                linspace(0, track.l_track, self.nx), self.dx, mount_pos)

            seclay = track.slab if isinstance(track, SimplePeriodicSlabSingleRailTrack) else track.sleeper


        if isinstance(track, (ContSlabSingleRailTrack, DiscrSlabSingleRailTrack)):
            sb_x.data[:] = 0
            sb_z.data[:] = 0
            sb_y.data[:] = 0
            sb_xr.data[:] = 0
            sb_zr.data[:] = 0
            sb_yr.data[:] = 0
            db_x.data[:] = 0
            db_z.data[:] = 0
            db_y.data[:] = 0
            db_xr.data[:] = 0
            db_yr.data[:] = 0
            db_zr.data[:] = 0

            # Assigns the sleeper mass to a very large number
            ms.data[:] = 1e20
            moinnert_s_x.data[:] = 1e20
            moinnert_s_y.data[:] = 1e20
            moinnert_s_z.data[:] = 1e20
            rho_s = 1e20
            z_st = 0
            z_sb = 0
            Ex = Constant(name="Ex", value=1)
            Ez = Constant(name="Ez", value=1)


        else:
            ballast = track.ballast
            sb_x.data[:] = ballast.sb_x * mount_pos_interp
            sb_z.data[:] = ballast.sb_z * mount_pos_interp
            sb_y.data[:] = ballast.sb_y * mount_pos_interp
            sb_xr.data[:] = ballast.sb_xr * mount_pos_interp
            sb_zr.data[:] = ballast.sb_zr * mount_pos_interp
            sb_yr.data[:] = ballast.sb_yr * mount_pos_interp
            db_x.data[:] = ballast.db_x * mount_pos_interp
            db_z.data[:] = ballast.db_z * mount_pos_interp
            db_y.data[:] = ballast.db_y * mount_pos_interp
            db_xr.data[:] = ballast.db_xr * mount_pos_interp
            db_zr.data[:] = ballast.db_zr * mount_pos_interp
            db_yr.data[:] = ballast.db_yr * mount_pos_interp

            ms.data[:] = (seclay.ms * mount_pos_interp) + 1e-20
            moinnert_s_x.data[:] = seclay.Is_x * mount_pos_interp + 1e-20
            moinnert_s_y.data[:] = seclay.Is_y * mount_pos_interp + 1e-20
            moinnert_s_z.data[:] = seclay.Is_z * mount_pos_interp + 1e-20
            rho_s = seclay.rhos
            z_st = seclay.z_st
            z_sb = seclay.z_sb
            Ex = Constant(name="Ex", value=track.E[0])
            Ez = Constant(name="Ez", value=track.E[1])

        sp_z.data[:] = track.pad.sp_z * mount_pos_interp
        sp_y.data[:] = track.pad.sp_y * mount_pos_interp
        sp_x.data[:] = track.pad.sp_x * mount_pos_interp
        sp_w.data[:] = track.pad.sp_w * mount_pos_interp
        sp_zr.data[:] = track.pad.sp_zr * mount_pos_interp
        sp_yr.data[:] = track.pad.sp_yr * mount_pos_interp
        sp_xr.data[:] = track.pad.sp_xr * mount_pos_interp
        dp_z.data[:] = track.pad.dp_z * mount_pos_interp
        dp_y.data[:] = track.pad.dp_y * mount_pos_interp
        dp_x.data[:] = track.pad.dp_x * mount_pos_interp
        dp_xr.data[:] = track.pad.dp_xr * mount_pos_interp


        # Define variables
        if self.store == "point":
            save = None
        elif self.store == "full":
            save = self.nt

        so1 = 6
        u_x = TimeFunction(name="u_x", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)
        u_y = TimeFunction(name="u_y",grid=self.grid,time_order=2,space_order=so1, dtype=float64, save=save)
        u_z = TimeFunction(name="u_z",grid=self.grid,time_order=2,space_order=so1, dtype=float64, save=save)
        phi_y = TimeFunction(name="phi_y", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)
        phi_z = TimeFunction(name="phi_z", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)
        phi_x = TimeFunction(name="phi_x",grid=self.grid,time_order=2,space_order=so1, dtype=float64, save=save)
        u_w = TimeFunction(name="u_w", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)

        u_sx = TimeFunction(name="u_sx", grid=self.grid, time_order=2, dtype=float64, save=None)
        u_sz = TimeFunction(name="u_sz", grid=self.grid, time_order=2, dtype=float64, save=None)
        u_sy = TimeFunction(name="u_sy", grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sx = TimeFunction(name="phi_sx", grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sz = TimeFunction(name="phi_sz", grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sy = TimeFunction(name="phi_sy", grid=self.grid, time_order=2, dtype=float64, save=None)

        ##### Damping
        sigm = Function(name="sigma", grid=self.grid)
        alph = Function(name="alpha", grid=self.grid)
        self.bound.calc_damping_profile(dx=self.dx, nx=self.nx)
        sigm.data[:] = self.bound.sigma
        alph.data[:] = self.bound.alpha

        # Auxiliary variables
        so2 = 3
        to2 = 2
        psi_ux = TimeFunction(name="psi_ux", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_ux = TimeFunction(name="theta_ux", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_x_dx_pml = u_x.dx - psi_ux * sigm
        u_x_dx2_pml = (u_x.dx2 - (sigm.dx * psi_ux + sigm * psi_ux.dx) - sigm * theta_ux)

        psi_uz = TimeFunction(name="psi_uz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_uz = TimeFunction( name="theta_uz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_z_dx_pml = u_z.dx - psi_uz * sigm
        u_z_dx2_pml = (u_z.dx2 - (sigm.dx * psi_uz + sigm * psi_uz.dx) - sigm * theta_uz)

        psi_phiy = TimeFunction(name="psi_phiy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_phiy = TimeFunction(name="theta_phiy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        phi_y_dx_pml = phi_y.dx - psi_phiy * sigm
        phi_y_dx2_pml = (phi_y.dx2 - (sigm.dx * psi_phiy + sigm * psi_phiy.dx) - sigm * theta_phiy)

        psi_uy = TimeFunction(name="psi_uy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_uy = TimeFunction(name="theta_uy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_y_dx_pml = u_y.dx - psi_uy * sigm
        u_y_dx2_pml = (u_y.dx2 - (sigm.dx * psi_uy + sigm * psi_uy.dx)- sigm * theta_uy)

        psi_phiz = TimeFunction(name="psi_phiz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_phiz = TimeFunction(name="theta_phiz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        phi_z_dx_pml = phi_z.dx - psi_phiz * sigm
        phi_z_dx2_pml = (phi_z.dx2 - (sigm.dx * psi_phiz + sigm * psi_phiz.dx) - sigm * theta_phiz)

        psi_phix = TimeFunction(name="psi_phix", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_phix = TimeFunction(name="theta_phix", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        phi_x_dx_pml = phi_x.dx - psi_phix * sigm
        phi_x_dx2_pml = (phi_x.dx2 - (sigm.dx * psi_phix + sigm * psi_phix.dx) - sigm * theta_phix)

        psi_uw = TimeFunction(name="psi_uw", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_uw = TimeFunction(name="theta_uw", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_w_dx_pml = u_w.dx - psi_uw * sigm
        u_w_dx2_pml = (u_w.dx2 - (sigm.dx * psi_uw + sigm * psi_uw.dx) - sigm * theta_uw)


        # Auxiliary equations for ADE-PML
        ade_psi_ux = Eq(psi_ux.dt, u_x_dx_pml - psi_ux * alph)
        ade_theta_ux = Eq(theta_ux.dt, u_x_dx2_pml - theta_ux * alph)

        ade_psi_uz = Eq(psi_uz.dt, u_z_dx_pml - psi_uz * alph)
        ade_theta_uz = Eq(theta_uz.dt, u_z_dx2_pml - theta_uz * alph)

        ade_psi_phiy = Eq(psi_phiy.dt, phi_y_dx_pml - psi_phiy * alph)
        ade_theta_phiy = Eq(theta_phiy.dt, phi_y_dx2_pml - theta_phiy * alph)

        ade_psi_uy = Eq(psi_uy.dt, u_y_dx_pml - psi_uy * alph)
        ade_theta_uy = Eq(theta_uy.dt, u_y_dx2_pml - theta_uy * alph)

        ade_psi_phiz = Eq(psi_phiz.dt, phi_z_dx_pml - psi_phiz * alph)
        ade_theta_phiz = Eq(theta_phiz.dt, phi_z_dx2_pml - theta_phiz * alph)

        ade_psi_phix = Eq(psi_phix.dt, phi_x_dx_pml - psi_phix * alph)
        ade_theta_phix = Eq(theta_phix.dt, phi_x_dx2_pml - theta_phix * alph)

        ade_psi_uw = Eq(psi_uw.dt, u_w_dx_pml - psi_uw * alph)
        ade_theta_uw = Eq(theta_uw.dt, u_w_dx2_pml - theta_uw * alph)

        # Define Timoshenko beam equation

        rail = self.track.rail
        G = Constant(name="G", value=rail.G)
        E = Constant(name="E", value=rail.E)
        A = Constant(name="A", value=rail.Ar)
        Iy = Constant(name="I_y", value=rail.Iyr)
        Iz = Constant(name="I_z", value=rail.Izr)
        Iwz = Constant(name="I_wz", value=rail.Iwz)
        Iwy = Constant(name="I_wy", value=rail.Iwy)
        Iyz = Constant(name="I_yz", value=rail.Iyz)
        rho = Constant(name="rho", value=rail.rho)
        mr = Constant(name="m_r", value=rail.mr)
        kap_y = Constant(name="kappa_y", value=rail.kapy)
        kap_z = Constant(name="kappa_z", value=rail.kapz)
        e_y = Constant(name="e_y", value=rail.ey)
        e_z = Constant(name="e_z", value=rail.ez)
        chi_f = Constant(name="chi_f", value=0)
        Iw = Constant(name="I_w", value=rail.Iw)
        J = Constant(name="J", value=rail.J)
        Jt = Constant(name="J_t", value=rail.J_t)
        Ip = Constant(name="I_p", value=rail.Ipr)
        z_f = Constant(name="z_f", value=self.z_f)
        y_f = Constant(name="y_f", value=self.y_f)
        z_st = Constant(name="z_st", value=z_st)
        z_sb = Constant(name="z_sb", value=z_sb)
        rho_s = Constant(name="rho_s", value=rho_s)

        # Longitudinal Wave
        longw = Eq(
            -A * E * u_x_dx2_pml
            + chi_f * dp_x * u_w.backward.dt
            + chi_f * sp_x * u_w
            + dp_x * y_f * phi_z.backward.dt
            - dp_x * z_f * phi_y.backward.dt
            - dp_x * u_sx.backward.dt
            + (dp_x + rail.dr) * u_x.dt
            + mr * u_x.dt2
            + sp_x * y_f * phi_z
            - sp_x * z_f * phi_y
            - sp_x * u_sx
            + sp_x * u_x,
        )

        # Timoshenko Beam Wave - Sum of Vertical Forces
        tbw_vert_trans = Eq(
            A * G * e_y * kap_z * phi_x_dx2_pml
            # + A * G * e_y * kap_z * u_w_dx_pml --> neglected
            + A * G * kap_z * phi_y_dx_pml
            - A * G * kap_z * u_z_dx2_pml
            + mr * u_z.dt2
            - sp_z * y_f * phi_x
            - sp_z * u_sz
            + sp_z * u_z
            - y_f * dp_z * phi_x.backward.dt
            + (-dp_z) * u_sz.backward.dt
            + (dp_z + rail.dr) * u_z.dt,
        )

        # Timoshenko Beam Wave - Sum of Lateral Forces
        tbw_lat_trans = Eq(
            - A * G * e_z * kap_y * phi_x_dx2_pml
            # - A * G * e_z * kap_y * u_w_dx_pml --> neglected
            - A * G * kap_y * phi_z_dx_pml
            - A * G * kap_y * u_y_dx2_pml
            + mr * u_y.dt2
            + sp_y * z_f * phi_x
            - sp_y * z_st * phi_sx
            - sp_y * u_sy
            + sp_y * u_y
            + z_f * dp_y * phi_x.backward.dt
            + z_st * (-dp_y) * phi_sx.backward.dt
            + (-dp_y) * u_sy.backward.dt
            + (dp_y + rail.dr) * u_y.dt,
        )

        # Torsional Beam Wave (Rotation about x-axis)
        torsw = Eq(
            -A * G * e_y * kap_z * phi_y_dx_pml
            + A * G * e_y * kap_z * u_z_dx2_pml
            - A * G * e_z * kap_y * phi_z_dx_pml
            - A * G * e_z * kap_y * u_y_dx2_pml
            - G * Jt * u_w_dx_pml
            - G * (J + Jt) * phi_x_dx2_pml
            + Ip * rho * phi_x.dt2
            - sp_y * z_f * u_sy
            + sp_y * z_f * u_y
            + sp_z * y_f * u_sz
            - sp_z * y_f * u_z
            + y_f * dp_z * u_sz.backward.dt
            - y_f * dp_z * u_z.dt
            - z_f * dp_y * u_sy.backward.dt
            + z_f * dp_y * u_y.dt
            + (-dp_xr - z_f * z_st * dp_y) * phi_sx.backward.dt
            + (-sp_xr - sp_y * z_f * z_st) * phi_sx
            + (dp_xr + y_f**2 * dp_z + z_f**2 * dp_y) * phi_x.dt
            + (sp_xr + sp_y * z_f**2 + sp_z * y_f**2) * phi_x,
        )

        # Timoshenko Beam Wave - Sum of Lateral Moments
        tbw_lat_rot = Eq(
            + A * G * e_z * kap_y * phi_x_dx_pml
            + A * G * kap_y * u_y_dx_pml
            # - E * Iwz * u_w_dx2_pml --> neglected
            + E * Iyz * phi_y_dx2_pml
            - E * Iz * phi_z_dx2_pml
            # + Iwz * rho * u_w.backward.dt2 --> neglected
            - Iyz * rho * phi_y.backward.dt2
            # + chi_f * dp_x * y_f * u_w.backward.dt --> neglected
            + dp_x * y_f**2 * phi_z.dt
            - dp_x * y_f * z_f * phi_y.backward.dt
            - dp_x * y_f * u_sx.backward.dt
            + dp_x * y_f * u_x.dt
            + rho * (-Iwz + Iz) * phi_z.dt2
            - sp_x * y_f * z_f * phi_y
            - sp_x * y_f * u_sx
            + sp_x * y_f * u_x
            - sp_zr * phi_sz
            # (A * G * e_z * kap_y + chi_f * sp_x * y_f * u_w) * u_w --> neglected
            + (A * G * kap_y + sp_x * y_f**2 + sp_zr) * phi_z,
        )

        # Timoshenko Beam Wave - Sum of Vertical Moments
        tbw_vert_rot = Eq(
            A * G * e_y * kap_z * phi_x_dx_pml
            - A * G * kap_z * u_z_dx_pml
            # + E * Iwy * u_w_dx2_pml --> neglected
            - E * Iy * phi_y_dx2_pml
            + E * Iyz * phi_z_dx2_pml
            # - Iwy * rho * u_w.backward.dt2 --> neglected
            - Iyz * rho * phi_z.dt2
            # - chi_f * dp_x * z_f * u_w.backward.dt --> neglected
            - dp_x * y_f * z_f * phi_z.dt
            + dp_x * z_f**2 * phi_y.dt
            + dp_x * z_f * u_sx.backward.dt
            - dp_x * z_f * u_x.dt
            + rho * (Iwy + Iy) * phi_y.dt2
            - sp_x * y_f * z_f * phi_z
            + sp_x * z_f * u_sx
            - sp_x * z_f * u_x
            - sp_yr * phi_sy
            # + (A * G * e_y * kap_z - chi_f * sp_x * z_f) * u_w --> neglected
            + (A * G * kap_z + sp_x * z_f**2 + sp_yr) * phi_y,
        )

        # Warping Equation
        warp = Eq(
            - A * G * e_y * kap_z * u_z_dx_pml
            + A * G * e_z * kap_y * u_y_dx_pml
            - E * Iw * u_w_dx2_pml
            + E * Iwy * phi_y_dx2_pml
            - E * Iwz * phi_z_dx2_pml
            + G * Jt * phi_x_dx_pml
            + Iw * rho * u_w.dt2
            - Iwy * rho * phi_y.dt2
            + Iwz * rho * phi_z.dt2
            + chi_f**2 * dp_x * u_w.dt
            + chi_f * dp_x * y_f * phi_z.dt
            - chi_f * dp_x * z_f * phi_y.dt
            - chi_f * dp_x * u_sx.backward.dt
            + chi_f * dp_x * u_x.dt
            - chi_f * sp_x * u_sx
            + chi_f * sp_x * u_x
            + (A * G * e_y * kap_z - chi_f * sp_x * z_f) * phi_y
            + (A * G * e_z * kap_y + chi_f * sp_x * y_f) * phi_z
            + (G * Jt + chi_f**2 * sp_x + sp_w) * u_w,
        )

        # Translational Sleeper Equation (x-direction)
        slep_trans_x = Eq(
            - chi_f * dp_x * u_w.dt
            - chi_f * sp_x * u_w
            - dp_x * y_f * phi_z.dt
            + dp_x * z_f * phi_y.dt
            - dp_x * u_x.dt
            - sp_x * y_f * phi_z
            + sp_x * z_f * phi_y
            - sp_x * u_x
            + (sp_x + sb_x * Ex) * u_sx
            + (db_x * Ex + dp_x) * u_sx.dt
            + ms * u_sx.dt2 * Ex,
        )

        # Translational Sleeper Equation (z-direction)
        slep_trans_z = Eq(
            sp_z * y_f * phi_x
            - sp_z * u_z
            - y_f * (-dp_z) * phi_x.dt
            + (-dp_z) * u_z.dt
            + (sp_z + sb_z * Ez) * u_sz
            + (db_z * Ez + dp_z) * u_sz.dt
            + ms * u_sz.dt2 * Ez,
        )

        # Translational Sleeper Equation (y-direction)
        slep_trans_y = Eq(
            ms * u_sy.dt2
            - sp_y * z_f * phi_x
            - sp_y * u_y
            + z_f * (-dp_y) * phi_x.dt
            + (-dp_y) * u_y.dt
            + (sb_y + sp_y) * u_sy
            + (db_y * z_sb + z_st * dp_y) * phi_sx.dt
            + (sb_y * z_sb + sp_y * z_st) * phi_sx
            + (db_y + dp_y) * u_sy.dt,
        )

        # Rotational Sleeper Equation (x-axis)
        slep_rot_x = Eq(
            moinnert_s_x * rho_s * phi_sx.dt2
            - sp_y * z_st * u_y
            - z_st * dp_y * u_y.dt
            + (-dp_xr - z_f * z_st * dp_y) * phi_x.dt
            + (-sp_xr - sp_y * z_f * z_st) * phi_x
            + (db_y * z_sb + z_st * dp_y) * u_sy.dt
            + (sb_y * z_sb + sp_y * z_st) * u_sy
            + (db_xr + db_y * z_sb**2 + dp_xr + z_st**2 * dp_y) * phi_sx.dt
            + (sb_xr + sb_y * z_sb**2 + sp_xr + sp_y * z_st**2) * phi_sx,
        )

        # Rotational Sleeper Equation (z-axis)
        slep_rot_z = Eq(
            moinnert_s_z * rho_s * phi_sz.dt2
            + db_zr * phi_sz.dt
            - sp_zr * phi_z
            + (sb_zr + sp_zr) * phi_sz,
        )

        # Rotational Sleeper Equation (y-axis)
        slep_rot_y = Eq(
            moinnert_s_y * rho_s * phi_sy.dt2
            + db_yr * phi_sy.dt
            - sp_yr * phi_y
            + (sb_yr + sp_yr) * phi_sy,
        )

        upd_longw = Eq(u_x.forward, solve(longw, u_x.forward))
        upd_tbw_vert_trans = Eq(u_z.forward, solve(tbw_vert_trans, u_z.forward))
        upd_tbw_lat_trans = Eq(u_y.forward, solve(tbw_lat_trans, u_y.forward))
        upd_torsw = Eq(phi_x.forward, solve(torsw, phi_x.forward))
        upd_tbw_vert_rot = Eq(phi_y.forward, solve(tbw_vert_rot, phi_y.forward))
        upd_tbw_lat_rot = Eq(phi_z.forward, solve(tbw_lat_rot, phi_z.forward))
        upd_warp = Eq(u_w.forward, solve(warp, u_w.forward))

        upd_psi_ux = Eq(psi_ux.forward, solve(ade_psi_ux, psi_ux.forward), subdomain=self.bound_dom)
        upd_theta_ux = Eq(theta_ux.forward, solve(ade_theta_ux, theta_ux.forward), subdomain=self.bound_dom)
        upd_psi_uz = Eq(psi_uz.forward, solve(ade_psi_uz, psi_uz.forward), subdomain=self.bound_dom)
        upd_theta_uz = Eq(theta_uz.forward, solve(ade_theta_uz, theta_uz.forward), subdomain=self.bound_dom)
        upd_psi_phiy = Eq(psi_phiy.forward, solve(ade_psi_phiy, psi_phiy.forward), subdomain=self.bound_dom)
        upd_theta_phiy = Eq(theta_phiy.forward, solve(ade_theta_phiy, theta_phiy.forward), subdomain=self.bound_dom)
        upd_psi_uy = Eq(psi_uy.forward, solve(ade_psi_uy, psi_uy.forward), subdomain=self.bound_dom)
        upd_theta_uy = Eq(theta_uy.forward, solve(ade_theta_uy, theta_uy.forward), subdomain=self.bound_dom)
        upd_psi_phiz = Eq(psi_phiz.forward, solve(ade_psi_phiz, psi_phiz.forward), subdomain=self.bound_dom)
        upd_theta_phiz = Eq(theta_phiz.forward, solve(ade_theta_phiz, theta_phiz.forward), subdomain=self.bound_dom)
        upd_psi_phix = Eq(psi_phix.forward, solve(ade_psi_phix, psi_phix.forward), subdomain=self.bound_dom)
        upd_theta_phix = Eq(theta_phix.forward, solve(ade_theta_phix, theta_phix.forward), subdomain=self.bound_dom)
        upd_psi_uw = Eq(psi_uw.forward, solve(ade_psi_uw, psi_uw.forward), subdomain=self.bound_dom)
        upd_theta_uw = Eq(theta_uw.forward, solve(ade_theta_uw, theta_uw.forward), subdomain=self.bound_dom)

        if (isinstance(self.track, (ContSlabSingleRailTrack, DiscrSlabSingleRailTrack))):

            op = [
                upd_longw,
                upd_tbw_vert_trans,
                upd_tbw_lat_trans,
                upd_torsw,
                upd_tbw_lat_rot,
                upd_tbw_vert_rot,
                upd_warp,
                # upd_u_sx,
                # upd_u_sz,
                # upd_u_sy,
                # upd_phi_sx,
                # upd_phi_sz,
                # upd_phi_sy,
                upd_psi_ux,
                upd_theta_ux,
                upd_psi_uz,
                upd_theta_uz,
                upd_psi_phiy,
                upd_theta_phiy,
                upd_psi_uy,
                upd_theta_uy,
                upd_psi_phiz,
                upd_theta_phiz,
                upd_psi_phix,
                upd_theta_phix,
                upd_psi_uw,
                upd_theta_uw,
            ]

        else:
            upd_u_sx = Eq(u_sx.forward, solve(slep_trans_x, u_sx.forward))
            upd_u_sz = Eq(u_sz.forward, solve(slep_trans_z, u_sz.forward))
            upd_u_sy = Eq(u_sy.forward, solve(slep_trans_y, u_sy.forward))
            upd_phi_sx = Eq(phi_sx.forward, solve(slep_rot_x, phi_sx.forward))
            upd_phi_sz = Eq(phi_sz.forward, solve(slep_rot_z, phi_sz.forward))
            upd_phi_sy = Eq(phi_sy.forward, solve(slep_rot_y, phi_sy.forward))

            op = [
                upd_longw,
                upd_tbw_vert_trans,
                upd_tbw_lat_trans,
                upd_torsw,
                upd_tbw_lat_rot,
                upd_tbw_vert_rot,
                upd_warp,
                upd_u_sx,
                upd_u_sz,
                upd_u_sy,
                upd_phi_sx,
                upd_phi_sz,
                upd_phi_sy,
                upd_psi_ux,
                upd_theta_ux,
                upd_psi_uz,
                upd_theta_uz,
                upd_psi_phiy,
                upd_theta_phiy,
                upd_psi_uy,
                upd_theta_uy,
                upd_psi_phiz,
                upd_theta_phiz,
                upd_psi_phix,
                upd_theta_phix,
                upd_psi_uw,
                upd_theta_uw,
            ]

        return op, u_z, u_y, phi_x

    def check_stability(self):
        """Check stability."""
        # Courant number
        c_ql = sqrt(self.track.rail.E / self.track.rail.rho)
        C = c_ql * self.dt / self.dx
        cfl_status = "stable" if C < 1 else "unstable"
        print(f"Courant number: {C:.2f} ({cfl_status})")

    def _abstract(self) -> None:
        pass

