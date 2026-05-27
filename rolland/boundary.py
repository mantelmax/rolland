"""Defines boundary classes for FDM simulation.

.. autosummary::
    :toctree: boundary

    PMLRailDampVertic
"""

from dataclasses import dataclass


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
