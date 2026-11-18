"""Defines boundary classes for FDM simulation.

.. autosummary::
    :toctree: boundary

    CFSPML
"""

from dataclasses import dataclass

import numpy as np
from devito import Eq, Function, TimeFunction, solve
from devito.types.equation import Eq as DevitoEq
from devito.types.grid import Grid
from numpy import float64, linspace, zeros


@dataclass
class CFSPML:
    r"""Complex Frequency Shifted Perfectly Matched Layer (CFS-PML).

    Generates spatial damping profiles to absorb outgoing waves at computational
    grid boundaries, minimizing artificial reflections.

    Attributes
    ----------
    a : float, default=1e5
        Maximum damping coefficient :math:`[s^{-1}]`.
    alpha : float, default=10000
        Complex Frequency Shift (CFS) coefficient :math:`[-]`.
    m : float, default=7
        Polynomial damping exponent :math:`[-]`.
    l_bound : float, default=10
        Length of the boundary domain (single-sided) :math:`[m]`.
    """

    a: float = 1e5
    alpha: float = 10000.0
    m: float = 7.0
    l_bound: float = 10.0

    def _generate_damping_profile(self, dx: float, nx: int) -> np.ndarray:
        """Calculate the 1D spatial damping array across the entire grid.

        Parameters
        ----------
        dx : float
            Spatial grid spacing.
        nx : int
            Number of grid points in the domain.

        Returns
        -------
        numpy.ndarray
            The 1D spatial damping profile array.
        """
        n_pml = int(self.l_bound / dx)
        x_pml = linspace(0, self.l_bound, n_pml)

        # Polynomial damping curve
        pml_curve = self.a * ((x_pml / self.l_bound) ** self.m)

        # Apply to both boundaries
        sigma = zeros(nx)
        sigma[:n_pml] = pml_curve[::-1]  # Left boundary (reversed)
        sigma[-n_pml:] += pml_curve      # Right boundary

        return sigma

    def initialize_on_grid(self, grid: Grid, dx: float, nx: int) -> tuple[Function, Function]:
        """Create Devito Functions for damping and populate them with data.

        Parameters
        ----------
        grid : Grid
            The Devito computational grid.
        dx : float
            Spatial grid spacing.
        nx : int
            Number of grid points in the domain.

        Returns
        -------
        tuple[Function, Function]
            The Devito Functions (`sigma`, `alpha`) initialized with damping profiles.
        """
        sigm = Function(name='sigma', grid=grid)
        alph = Function(name='alpha', grid=grid)

        sigm.data[:] = self._generate_damping_profile(dx, nx)
        alph.data[:] = self.alpha

        return sigm, alph

    def apply_pml(self, base_var: TimeFunction, name_suffix: str, grid: Grid, bound_dom, sigm: Function,
                  alph: Function) -> tuple[DevitoEq, DevitoEq, list[DevitoEq]]:
        """Generate ADE-PML boundary auxiliary variables and DEVITO equations.

        Parameters
        ----------
        base_var : TimeFunction
            The primary wavefield variable to apply PML to.
        name_suffix : str
            Suffix for naming the auxiliary variables (`psi`, `theta`).
        grid : Grid
            The Devito computational grid.
        bound_dom : SubDomain
            The subdomain representing the boundary regions where PML applies.
        sigm : Function
            The spatial damping profile `sigma`.
        alph : Function
            The spatial CFS profile `alpha`.

        Returns
        -------
        tuple[DevitoEq, DevitoEq, list[DevitoEq]]
            The modified spatial derivatives (`dx_pml`, `dx2_pml`) and a list
            of forward update equations for the auxiliary variables.
        """
        # Auxiliary fields
        psi = TimeFunction(
            name=f'psi_{name_suffix}', grid=grid, time_order=2, space_order=3, dtype=float64, save=None,
        )
        theta = TimeFunction(
            name=f'theta_{name_suffix}', grid=grid, time_order=2, space_order=3, dtype=float64, save=None,
        )

        # Calculate modified spatial derivatives
        dx_pml = base_var.dx - psi * sigm
        dx2_pml = base_var.dx2 - (sigm.dx * psi + sigm * psi.dx) - sigm * theta

        # Auxiliary equations
        ade_psi = Eq(psi.dt, dx_pml - psi * alph)
        ade_theta = Eq(theta.dt, dx2_pml - theta * alph)

        # Forward update stencils
        upd_psi = Eq(psi.forward, solve(ade_psi, psi.forward), subdomain=bound_dom)
        upd_theta = Eq(theta.forward, solve(ade_theta, theta.forward), subdomain=bound_dom)

        return dx_pml, dx2_pml, [upd_psi, upd_theta]
