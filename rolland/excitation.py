"""Defines excitation classes for FDM simulation.

.. autosummary::
    :toctree: excitation

    Excitation
    StationaryExcitation
    GaussianImpulse
    MovingExcitation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from devito import SparseTimeFunction
from examples.seismic import TimeAxis
from numpy import exp, arange


class Excitation(ABC):
    """Abstract base class for excitation."""

    @abstractmethod
    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""


class StationaryExcitation(Excitation):
    """Abstract base class for stationary excitation."""


@dataclass(kw_only=True)
class GaussianImpulse(StationaryExcitation):
    """Gaussian impulse excitation class.

    Gaussian impulse according to :cite:t:`stampka2022a`. This excitation type is used for
    non-moving sources.

    Attributes
    ----------
    sigma : float, default=0.7e-4
        Pulse parameter (regulates pulse-time) :math:`[-]`.
    a : float, default=0.5e2
        Pulse parameter (regulates amplitude) :math:`[s]`.
    x_excit : list | float, default=50.0
        Excitation position :math:`[m]`.
    force_dir : str, default='vertical'
        Force direction ('vertical' or 'lateral').
    z_e : float, default=0.0
        Excitation z-coordinate :math:`[m]`.
    y_e : float, default=0.0
        Excitation y-coordinate :math:`[m]`.
    force : devito.SparseTimeFunction
        The computed force array containing force over time.
    """

    sigma: float = 0.7e-4
    a: float = 0.5e2
    x_excit: list | float = 50.0
    force_dir: str = 'vertical'
    z_e: float
    y_e: float

    def force(self, t):
        """Compute force array (contains force over time)."""
        tg = t - 4 * self.sigma
        return self.a * tg / self.sigma**2 * exp(-(tg**2) / self.sigma**2)

    def build_force_array(self, nt, dt, grid):
        """Build force array (contains force over time)."""
        time = dt * arange(nt)

        force = SparseTimeFunction(
            name='F',
            grid=grid,
            nt=nt,
            npoint=1,
        )

        force.coordinates.data[:] = self.x_excit

        shifted_time = time - 4 * self.sigma
        wavelet = self.a * shifted_time / self.sigma**2 * exp(
            -(shifted_time**2) / self.sigma**2
        )

        force.data[:, 0] = wavelet
        self.force = force
        return force

    def inject_in_track(self, discr):
        """Build injection expressions for the excitation.

        Parameters
        ----------
        discr : object
            Discretization object.

        Returns
        -------
        tuple
            Injection expressions for vertical displacement, lateral
            displacement, and rotation about the x-axis.
        """
        force = self.build_force_array(nt=discr.nt, dt=discr.dt, grid=discr.grid)
        rail = discr.track.rail
        track = discr.track
        dx = discr.dx
        dt = discr.dt

        rhs_bw1 = 1 / (rail.dr / dt + track.pad.dp_z / dt + rail.mr / dt**2)
        rhs_bw2 = 1 / (rail.dr / dt + track.pad.dp_y / dt + rail.mr / dt**2)
        rhs_tw = 1 / (track.pad.dp_xr / dt + rail.rho * track.rail.Ipr / dt**2)

        # The excitation injection term
        if self.force_dir == 'vertical':
            exc_z = force.inject(field=discr.u_z.forward, expr=force * rhs_bw1 * 1 / dx)
            exc_y = force.inject(field=discr.u_y.forward, expr=0)
            exc_rotx = force.inject(field=discr.phi_x.forward, expr=(-force * self.y_e) * rhs_tw * 1 / dx)

        elif self.force_dir == 'lateral':
            exc_z = force.inject(field=discr.u_z.forward, expr=0)
            exc_y = force.inject(field=discr.u_y.forward, expr=force * rhs_bw2 * 1 / dx)
            exc_rotx = force.inject(field=discr.phi_x.forward, expr=(force * self.z_e) * rhs_tw * 1 / dx)
        # TODO: Add longitudinal excitation
        return exc_z, exc_y, exc_rotx


    def _abstract(self) -> None:
        pass

class MovingExcitation(Excitation):
    """Moving excitation class."""