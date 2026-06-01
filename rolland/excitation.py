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

from numpy import exp


class Excitation(ABC):
    """Abstract base class for excitation."""

    @abstractmethod
    def validate_excitation(self):
        """Validate excitation parameters."""


class StationaryExcitation(Excitation):
    """Abstract base class for stationary excitation."""

    @abstractmethod
    def validate_stationary_excitation(self):
        """Validate stationary excitation parameters."""


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
    """

    sigma: float = 0.7e-4
    a: float = 0.5e2
    x_excit: list | float = 50.0

    def validate_excitation(self):
        """Validate excitation parameters."""

    def validate_stationary_excitation(self):
        """Validate stationary excitation parameters."""

    def force(self, t):
        """Compute force array (contains force over time)."""
        tg = t - 4 * self.sigma
        return self.a * tg / self.sigma ** 2 * exp(-tg ** 2 / self.sigma ** 2)


class MovingExcitation(Excitation):
    """Moving excitation class."""

