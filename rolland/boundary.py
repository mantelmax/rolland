"""Defines boundary classes for FDM simulation.

.. autosummary::
    :toctree: boundary

    PMLRailDampVertic
"""

from traitlets import Float, HasTraits


class PMLRailDampVertic(HasTraits):
    r"""Calculate the boundary domain properties according to :cite:t:`stampka2022a`.

    A perfectly matched layer (PML) method is used which increases the rail damping
    coefficient in the boundary domain for the vertical rail deflection.

    Attributes
    ----------
    alpha : float
        Damping exponent :math:`[-]`.
    l_bound : float
        Length of the boundary domain (single sided) :math:`[m]`.
    """

    alpha = Float(default_value=7)
    l_bound = Float(default_value=33.0)

    def pml(self, drbc, xbc):
        """Exponential increasing rail damping, added to dr."""
        return drbc * xbc ** self.alpha / self.l_bound ** self.alpha
