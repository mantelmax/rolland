"""Defines boundary classes for FDM simulation.

.. autosummary::
    :toctree: boundary

    PMLRailDampVertic
"""

from dataclasses import dataclass

from numpy import linspace, ndarray, ones, zeros


@dataclass
class PMLRailDampVertic:
    r"""Calculate the boundary domain properties according to :cite:t:`stampka2022a`.

    A perfectly matched layer (PML) method is used which increases the rail damping
    coefficient in the boundary domain for the vertical rail deflection.

    Attributes
    ----------
    alpha : float, default=7.0
        Damping exponent :math:`[-]`.
    l_bound : float, default=33.0
        Length of the boundary domain (single sided) :math:`[m]`.
    """

    alpha: float = 7.0
    l_bound: float = 33.0

    def pml(self, drbc, xbc):
        """Exponential increasing rail damping, added to dr."""
        return drbc * xbc ** self.alpha / self.l_bound ** self.alpha

@dataclass
class DevitoPMLDamp:
    r"""Devito PML.

    Attributes
    ----------
    a : float
        Damping coefficient :math:`[s^{-1}]`.
    alpha : float
        CFS - coefficient :math:`[-]`.
    m : float
        Damping exponent :math:`[-]`.
    l_bound : float
        Length of the boundary domain (single sided) :math:`[m]`.
    sigma : ndarray
        Spatial damping profile :math:`[s^{-1}]`.
    """

    a: float = 1e5
    alpha: float = 10000
    m: float = 7
    l_bound: float = 10

    def calc_damping_function(self, dx):
        """Calculate the damping profile for the PML boundary domain."""
        x_pml = linspace(0, self.l_bound, int(self.l_bound / dx))
        return self.a * ((x_pml / self.l_bound) ** self.m)

    def calc_damping_profile(self, dx, nx):
        """Apply the damping profile to the grid."""
        pml = self.calc_damping_function(dx)
        self.sigma = zeros(nx)
        self.sigma[: pml.size] = pml[::-1]
        self.sigma[-pml.size :] += pml

        alpha_pml = ones(len(pml)) * self.alpha
        self.alpha_tot = zeros(nx)
        self.alpha_tot[: pml.size] = alpha_pml[::-1]
        self.alpha_tot[-pml.size :] += alpha_pml
