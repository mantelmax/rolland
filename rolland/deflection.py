"""Defines deflection classes for FDM simulation.

.. autosummary::
    :toctree: deflection

    Deflection
    DeflectionEBBVertic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from devito import (
    Operator,
    SparseTimeFunction,
)
from examples.seismic.source import TimeAxis

from .discretization import Discretization, DiscretizeDEVITO
from .excitation import Excitation, GaussPulse_DEVITO
from .track import (
    Track,
)


@dataclass(kw_only=True)
class Deflection(ABC):
    r"""Abstract base class for deflection classes.

    Attributes
    ----------
    excit : Excitation
        Excitation instance.
    discr : Discretization
        Discretization instance.
    """

    discr: Discretization
    excit: Excitation
    track: Track = field(init=False)

    def __post_init__(self, *args, **kwargs):
        """post_init method to set track attribute after initialization."""
        self.track = self.discr.track

    @abstractmethod
    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""


class RunDEVITO(Deflection):
    r"""Run DEVITO.

    Attributes
    ----------
    x_obs : float
        Response positions.
    """

    x_obs: float

    def __init__(self, *args, **kwargs):
        x_obs = kwargs.pop('x_obs', None)
        super().__init__(*args, **kwargs)
        self.x_obs = self.excit.x_excit if x_obs is None else x_obs
        self.run_devito()

    def run_devito(self):
        """Run DEVITO simulation."""
        # Calculate derived parameters

        op, u_z, u_y, phi_x = self.discr.build_operator()

        # Excitation
        time_range = TimeAxis(start=0, num=self.discr.nt, step=self.discr.dt)
        F = GaussPulse_DEVITO(
            name='F', grid=self.discr.grid, time_range=time_range, a=self.excit.a, t0=self.excit.sigma, npoint=1
        )
        F.coordinates.data[:] = self.excit.x_excit

        rail = self.discr.track.rail
        track = self.discr.track
        dx = self.discr.dx
        dt = self.discr.dt

        rhs_bw1 = 1 / (rail.dr / dt + track.pad.dp_z / dt + rail.mr / dt**2)
        rhs_bw2 = 1 / (rail.dr / dt + track.pad.dp_y / dt + rail.mr / dt**2)
        rhs_tw = 1 / (track.pad.dp_xr / dt + rail.rho * track.rail.Ipr / dt**2)

        # The excitation injection term
        if self.excit.force_dir == 'vertical':
            exc_z = F.inject(field=u_z.forward, expr=F * rhs_bw1 * 1 / dx)
            exc_y = F.inject(field=u_y.forward, expr=0)
            exc_x = F.inject(field=phi_x.forward, expr=(-F * self.excit.y_e) * rhs_tw * 1 / dx)

        elif self.excit.force_dir == 'lateral':
            exc_z = F.inject(field=u_z.forward, expr=0)
            exc_y = F.inject(field=u_y.forward, expr=F * rhs_bw2 * 1 / dx)
            exc_x = F.inject(field=phi_x.forward, expr=(F * self.excit.z_e) * rhs_tw * 1 / dx)

        # TODO: Add longitudinal excitation

        # Define observation points at x_excit
        u_z_obs = SparseTimeFunction(
            name='u_z_obs', grid=self.discr.grid, npoint=1, nt=self.discr.nt, coordinates=[[self.x_obs]]
        )
        u_y_obs = SparseTimeFunction(
            name='u_y_obs', grid=self.discr.grid, npoint=1, nt=self.discr.nt, coordinates=[[self.x_obs]]
        )
        phi_x_obs = SparseTimeFunction(
            name='phi_x_obs', grid=self.discr.grid, npoint=1, nt=self.discr.nt, coordinates=[[self.x_obs]]
        )
        obs_term_uz = u_z_obs.interpolate(expr=u_z)
        obs_term_uy = u_y_obs.interpolate(expr=u_y)
        obs_term_phix = phi_x_obs.interpolate(expr=phi_x)

        op_exc = Operator(op + exc_z + exc_y + exc_x + obs_term_uz + obs_term_uy + obs_term_phix)

        op_exc.apply(dt=self.discr.dt)

        self.u_z = u_z
        self.u_y = u_y
        self.phi_x = phi_x
        self.u_z_obs = u_z_obs
        self.u_y_obs = u_y_obs
        self.phi_x_obs = phi_x_obs
        self.F = F

    def _abstract(self) -> None:
        pass
