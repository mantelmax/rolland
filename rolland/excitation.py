"""Defines excitation classes for FDM simulation.

.. autosummary::
    :toctree: excitation

    Excitation
    StationaryExcitation
    GaussianImpulse
    MovingExcitation
    RandomForce
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import devito
import numpy as np
from devito import Eq, Function, SparseTimeFunction

if TYPE_CHECKING:
    from rolland.domainsetup import DomSetup


class Excitation(ABC):
    """Abstract base class for excitation."""

    @abstractmethod
    def inject_in_track(self, discr: 'DomSetup') -> list[Eq]:
        """Build injection expressions for the excitation.

        Parameters
        ----------
        discr : DomSetup
            The domain setup instance.

        Returns
        -------
        list[devito.Eq]
            List of injection equations.
        """

    @abstractmethod
    def observe_excitation(self, discr: 'DomSetup') -> tuple[dict, list[Eq]]:
        """Build expressions to observe deflections at the excitation position.

        Parameters
        ----------
        discr : DomSetup
            The domain setup instance.

        Returns
        -------
        tuple[dict, list[devito.Eq]]
            Observation functions and their corresponding equations.
        """


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

    Gaussian impulse according to :cite:p:`stampka_time-domain_2022`. This excitation type is used for
    non-moving sources.

    Attributes
    ----------
    sigma : float, default=0.7e-4
        Pulse parameter (regulates pulse-time) :math:`[s]`.
    a : float, default=0.5e2
        Pulse parameter (regulates amplitude) :math:`[N]`.
    x_excit : float, default=50.0
        Excitation position :math:`[m]`.
    force_dir : str, default='vertical'
        Force direction ('vertical' or 'lateral').
    force : SparseTimeFunction
        The Devito SparseTimeFunction representing the excitation force.
    """

    sigma: float = 0.7e-4
    a: float = 0.5e2
    x_excit: float = 50.0
    force_dir: str = 'vertical'

    def evaluate_wavelet(self, time_array: np.ndarray) -> np.ndarray:
        """Compute the force magnitude over time.

        Parameters
        ----------
        time_array : numpy.ndarray
            Array of time values.

        Returns
        -------
        numpy.ndarray
            The computed force magnitude array.
        """
        shifted_time = time_array - 4 * self.sigma
        return self.a * shifted_time / self.sigma**2 * np.exp(-(shifted_time**2) / self.sigma**2)

    def build_force_array(self, nt: int, dt: float, grid: 'devito.Grid') -> SparseTimeFunction:
        """Build the Devito SparseTimeFunction for the excitation.

        Parameters
        ----------
        nt : int
            Number of time steps.
        dt : float
            Time step size.
        grid : devito.Grid
            The computational grid.

        Returns
        -------
        devito.SparseTimeFunction
            The generated force function.
        """
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

    def inject_in_track(self, discr: 'DomSetup') -> list[Eq]:
        """Build injection expressions for the excitation.

        Parameters
        ----------
        discr : DomSetup
            The domain setup instance.

        Returns
        -------
        list[devito.Eq]
            List of injection equations.
        """
        force = self.build_force_array(nt=discr.nt, dt=discr.dt, grid=discr.grid)

        rail = discr.track.rail
        dx = discr.dx
        dt = discr.dt

        rhs_tw = 1 / (discr.f['dp_xr'] / dt + rail.rho * rail.Ipr / dt**2)
        injections = []

        if self.force_dir == 'vertical':
            rhs_bw1 = 1 / (rail.dr / dt + discr.f['dp_z'] / dt + rail.mr / dt**2)
            injections.append(force.inject(field=discr.u_z.forward, expr=force * rhs_bw1 / dx))

            if self.y_e != 0.0:
                injections.append(force.inject(field=discr.phi_x.forward, expr=(-force * self.y_e) * rhs_tw / dx))

        elif self.force_dir == 'lateral':
            rhs_bw2 = 1 / (rail.dr / dt + discr.f['dp_y'] / dt + rail.mr / dt**2)
            injections.append(force.inject(field=discr.u_y.forward, expr=force * rhs_bw2 / dx))

            if self.z_e != 0.0:
                injections.append(force.inject(field=discr.phi_x.forward, expr=(force * self.z_e) * rhs_tw / dx))

        return injections

    def observe_excitation(self, discr: 'DomSetup') -> tuple[dict, list[Eq]]:
        """Observes deflections exactly at the stationary excitation position.

        Parameters
        ----------
        discr : DomSetup
            The domain setup instance.

        Returns
        -------
        tuple[dict, list[devito.Eq]]
            Observation functions and their corresponding equations.
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


@dataclass(kw_only=True)
class MovingExcitation(Excitation, ABC):
    """Abstract base class for moving excitation.

    Attributes
    ----------
    z_e : float, default=0.0
        Contact point z-coordinate :math:`[m]`.
    y_e : float, default=0.0
        Contact point y-coordinate :math:`[m]`.
    """

    z_e: float = 0.0
    y_e: float = 0.0


@dataclass(kw_only=True)
class RandomForce(MovingExcitation):
    """A moving random excitation source for finite-difference track simulations.
    Based on the moving source model according to :cite:p:`nordborg_wheelrail_2002`.

    Applies a moving force along the track with a random dynamic component
    superimposed on a static load. The force ramps up smoothly over the first
    40% of the simulation to prevent shock artifacts.

    Attributes
    ----------
    v : float
        Velocity of the moving excitation :math:`[m/s]`.
    F_stat_z : float
        Static force in the vertical (z) direction :math:`[N]`.
    F_stat_y : float
        Static force in the lateral (y) direction :math:`[N]`.
    force_z : numpy.ndarray
        Calculated vertical random force time series (automatically calculated).
    force_y : numpy.ndarray
        Calculated lateral random force time series (automatically calculated).
    """

    v: float
    F_stat_z: float
    F_stat_y: float

    def calc_rnd_forcearray(self, nt: int) -> np.ndarray:
        """Calculate the time-series arrays for the randomized forces.

        Parameters
        ----------
        nt : int
            Number of time steps.

        Returns
        -------
        numpy.ndarray
            Array of shape (2, nt) containing vertical and lateral random forces.
        """
        ramp_len = int(0.4 * nt)

        # Create a linear ramp from ~0 to 1, padded with ones for the remaining time
        ramp = np.linspace(1 / ramp_len, 1.0, ramp_len)
        ramp_factor = np.concatenate([ramp, np.ones(nt - ramp_len)])

        # Generate Gaussian noise (mean=0, std=1)
        rnd = np.random.normal(loc=0.0, scale=1.0, size=nt)

        # Superimpose noise on the static force and apply the ramp
        force_z = self.F_stat_z * (1 + rnd) * ramp_factor
        force_y = self.F_stat_y * (1 + rnd) * ramp_factor
        self.force_z = force_z
        self.force_y = force_y

        return np.stack([force_z, force_y], axis=0)

    def inject_in_track(self, discr: 'DomSetup') -> list[Eq]:
        """Build Devito equations to inject the moving random force into the track.

        Parameters
        ----------
        discr : DomSetup
            The domain setup instance.

        Returns
        -------
        list[devito.Eq]
            List of injection equations.
        """
        grid = discr.grid
        t = grid.time_dim
        x = grid.dimensions[0]
        nt, nx = discr.nt, discr.nx
        dt, dx = discr.dt, discr.dx

        traj_pos = np.clip((discr.bound.l_bound + self.v * np.arange(nt) * dt) / dx, 2, nx - 3)
        traj_i = traj_pos.astype(np.int32)
        alpha = traj_pos - traj_i

        k_offsets = np.array([-1, 0, 1, 2])
        dists = k_offsets[:, np.newaxis] - alpha[np.newaxis, :]
        raw_weights = 0.5 + 0.5 * np.cos(np.pi * dists / 2.0)
        weights_norm = raw_weights / np.sum(raw_weights, axis=0)

        self._indices, self._weights = [], []
        for k in range(4):
            idx_f = Function(name=f'idx{k}', grid=grid, shape=(nt,), dimensions=(t,), dtype=np.int32)
            w_f = Function(name=f'w{k}', grid=grid, shape=(nt,), dimensions=(t,), dtype=np.float64)

            idx_f.data[:] = traj_i + k_offsets[k]
            w_f.data[:] = weights_norm[k]

            self._indices.append(idx_f)
            self._weights.append(w_f)

        self._F_z = Function(name='Fz', grid=grid, shape=(nt,), dimensions=(t,), dtype=np.float64)
        self._F_y = Function(name='Fy', grid=grid, shape=(nt,), dimensions=(t,), dtype=np.float64)

        force_z, force_y = self.calc_rnd_forcearray(nt)
        self._F_z.data[:] = force_z
        self._F_y.data[:] = force_y

        rail = discr.track.rail
        rhs_bw1 = 1.0 / (rail.dr / dt + discr.f['dp_z'] / dt + rail.mr / dt**2)
        rhs_bw2 = 1.0 / (rail.dr / dt + discr.f['dp_y'] / dt + rail.mr / dt**2)
        rhs_tw = 1.0 / (discr.f['dp_xr'] / dt + rail.rho * rail.Ipr / dt**2)
        inv_dx = 1.0 / dx

        # Uses the inherited self.y_e and self.z_e
        torque = -self._F_y * self.z_e + self._F_z * self.y_e

        injections = []
        for k in range(4):
            idx, w = self._indices[k], self._weights[k]

            injections.extend(
                [
                    Eq(
                        discr.u_z.forward.subs(x, idx),
                        discr.u_z.forward.subs(x, idx) - w * self._F_z * rhs_bw1 * inv_dx,
                    ),
                    Eq(
                        discr.u_y.forward.subs(x, idx),
                        discr.u_y.forward.subs(x, idx) - w * self._F_y * rhs_bw2 * inv_dx,
                    ),
                    Eq(
                        discr.phi_x.forward.subs(x, idx),
                        discr.phi_x.forward.subs(x, idx) + w * torque * rhs_tw * inv_dx,
                    ),
                ],
            )

        return injections

    def observe_excitation(self, discr: 'DomSetup') -> tuple[dict, list[Eq]]:
        """Observes deflections dynamically tracking the moving excitation position.

        Parameters
        ----------
        discr : DomSetup
            The domain setup instance.

        Returns
        -------
        tuple[dict, list[devito.Eq]]
            Observation functions and their corresponding equations.
        """
        if not hasattr(self, '_indices'):
            msg = 'inject_in_track must be called before observe_excitation.'
            raise RuntimeError(msg)

        grid = discr.grid
        t = grid.time_dim
        x = grid.dimensions[0]
        nt = discr.nt

        u_z_obs = Function(name='u_z_obs_mov', grid=grid, shape=(nt,), dimensions=(t,), dtype=np.float64)
        u_y_obs = Function(name='u_y_obs_mov', grid=grid, shape=(nt,), dimensions=(t,), dtype=np.float64)
        phi_x_obs = Function(name='phi_x_obs_mov', grid=grid, shape=(nt,), dimensions=(t,), dtype=np.float64)

        uz_sum, uy_sum, phix_sum = 0, 0, 0

        for k in range(4):
            idx, w = self._indices[k], self._weights[k]
            uz_sum += w * discr.u_z.forward.subs(x, idx)
            uy_sum += w * discr.u_y.forward.subs(x, idx)
            phix_sum += w * discr.phi_x.forward.subs(x, idx)

        obs_eqs = [Eq(u_z_obs, uz_sum), Eq(u_y_obs, uy_sum), Eq(phi_x_obs, phix_sum)]

        obs_funcs = {'u_z_obs': u_z_obs, 'u_y_obs': u_y_obs, 'phi_x_obs': phi_x_obs}

        return obs_funcs, obs_eqs
