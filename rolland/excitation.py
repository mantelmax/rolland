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

import numpy as np
from devito import SparseTimeFunction


class Excitation(ABC):
    """Abstract base class for excitation."""

    @abstractmethod
    def inject_in_track(self, discr) -> tuple:
        """Build injection expressions for the excitation."""

@dataclass(kw_only=True)
class StationaryExcitation(Excitation, ABC):
    """Abstract base class for stationary excitation.

    Attributes
    ----------
    z_e : float, default=0.0
        Excitation z-coordinate :math:`[m]`.
    y_e : float, default=0.0
        Excitation y-coordinate :math:`[m]`.
    """

    z_e: float = 0.0
    y_e: float = 0.0



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
    x_excit : float, default=50.0
        Excitation position :math:`[m]`.
    force_dir : str, default='vertical'
        Force direction ('vertical' or 'lateral').
    z_e : float, default=0.0
        Excitation z-coordinate :math:`[m]`.
    y_e : float, default=0.0
        Excitation y-coordinate :math:`[m]`.
    force: SparseTimeFunction
        The Devito SparseTimeFunction representing the excitation force.
    """

    sigma: float = 0.7e-4
    a: float = 0.5e2
    x_excit: float = 50.0
    force_dir: str = 'vertical'

    def evaluate_wavelet(self, time_array: np.ndarray) -> np.ndarray:
        """Compute the force magnitude over time."""
        shifted_time = time_array - 4 * self.sigma
        return self.a * shifted_time / self.sigma**2 * np.exp(-(shifted_time**2) / self.sigma**2)

    def build_force_array(self, nt: int, dt: float, grid) -> SparseTimeFunction:
        """Build the Devito SparseTimeFunction for the excitation."""
        time = dt * np.arange(nt)

        force_func = SparseTimeFunction(
            name='F',
            grid=grid,
            nt=nt,
            npoint=1,
        )

        force_func.coordinates.data[0, 0] = self.x_excit
        force_func.data[:, 0] = self.evaluate_wavelet(time)

        self.force = force_func
        return force_func

    def inject_in_track(self, discr) -> tuple:
        """Build injection expressions for the excitation."""
        force = self.build_force_array(nt=discr.nt, dt=discr.dt, grid=discr.grid)

        rail = discr.track.rail
        track = discr.track
        dx = discr.dx
        dt = discr.dt

        rhs_tw = 1 / (track.pad.dp_xr / dt + rail.rho * rail.Ipr / dt**2)
        injections = []

        # The excitation injection terms (omitting zero-value injections)
        if self.force_dir == 'vertical':
            rhs_bw1 = 1 / (rail.dr / dt + track.pad.dp_z / dt + rail.mr / dt**2)
            injections.append(force.inject(field=discr.u_z.forward, expr=force * rhs_bw1 / dx))

            if self.y_e != 0.0:
                injections.append(force.inject(field=discr.phi_x.forward, expr=(-force * self.y_e) * rhs_tw / dx))

        elif self.force_dir == 'lateral':
            rhs_bw2 = 1 / (rail.dr / dt + track.pad.dp_y / dt + rail.mr / dt**2)
            injections.append(force.inject(field=discr.u_y.forward, expr=force * rhs_bw2 / dx))

            if self.z_e != 0.0:
                injections.append(force.inject(field=discr.phi_x.forward, expr=(force * self.z_e) * rhs_tw / dx))

        return tuple(injections)

    def observe_excitation(self, discr):
        """Observes deflections exactly at the excitation position.

        Returns
        -------
        tuple
            A tuple containing (obs_funcs, obs_terms), where obs_funcs is a dictionary
            of SparseTimeFunctions and obs_terms is a list of interpolation equations.
        """
        obs_funcs = {}
        obs_terms = []

        npoint = 1
        coordinates = [[self.x_excit]]

        fields_map = {
            'u_z_obs': discr.u_z,
            'u_y_obs': discr.u_y,
            'phi_x_obs': discr.phi_x,
        }

        for name, field_expr in fields_map.items():
            obs = SparseTimeFunction(
                name=name,
                grid=discr.grid,
                npoint=npoint,
                nt=discr.nt,
                coordinates=coordinates,
            )
            obs_funcs[name] = obs
            obs_terms.append(obs.interpolate(expr=field_expr))

        return obs_funcs, obs_terms


class MovingExcitation(Excitation):
    """Moving excitation class."""

