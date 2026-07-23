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
    force_dir : str
        Force direction ('vertical' or 'lateral').
    z_e : float
        Excitation z-coordinate :math:`[m]`.
    y_e : float
        Excitation y-coordinate :math:`[m]`.
    """

    sigma: float = 0.7e-4
    a: float = 0.5e2
    x_excit: list | float = 50.0
    force_dir: str = "vertical"
    z_e: float = 0.0
    y_e: float = 0.0

    def validate_excitation(self):
        """Validate excitation parameters."""

    def validate_stationary_excitation(self):
        """Validate stationary excitation parameters."""

    def force(self, t):
        """Compute force array (contains force over time)."""
        tg = t - 4 * self.sigma
        return self.a * tg / self.sigma ** 2 * exp(-tg ** 2 / self.sigma ** 2)

class GaussPulse_DEVITO(SparseTimeFunction):
    """Simplified Gauss pulse source for Devito."""

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        # nt aus time_range extrahieren
        time_range = kwargs.get("time_range")
        if time_range is not None:
            kwargs["nt"] = time_range.num

        # npoint sicherstellen
        if "npoint" not in kwargs:
            kwargs["npoint"] = 1

        return args, kwargs

    def __init__(self, *args, **kwargs):
        # Parameter vor super().__init__() extrahieren
        time_range = kwargs.pop("time_range", None)
        a = kwargs.pop("a", 1.0)
        t0 = kwargs.pop("t0", None)

        # SparseTimeFunction initialisieren
        super().__init__(*args, **kwargs)

        # Parameter speichern
        self._time_range = time_range
        self.a = a
        self.t0 = t0

        # Gauss-Puls berechnen
        if time_range is not None and t0 is not None:
            sigma = self.t0
            time = time_range.time_values - 4 * sigma
            wavelet = self.a * time / sigma**2 * exp(-(time**2) / sigma**2)

            # Wavelet für alle Punkte setzen
            for p in range(self.npoint):
                self.data[:, p] = wavelet

    @property
    def time_values(self):
        return self._time_range.time_values

    @property
    def time_range(self):
        return self._time_range

class MovingExcitation(Excitation):
    """Moving excitation class."""

