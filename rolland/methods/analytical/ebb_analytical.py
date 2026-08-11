"""Analytical models for continuous tracks using Euler-Bernoulli Beam (EBB) theory.

This module provides analytical solutions for the dynamic response of railway tracks modeled
as continuously supported Euler-Bernoulli beams. These continuous models are particularly
useful for understanding low to mid-frequency track dynamics where the discrete nature of
sleeper spacing is less influential. The module includes single-layer (slab) and double-layer
(ballasted) support variations, calculating track responses such as mobility, receptance,
and accelerance.

.. autosummary::
    :toctree: analytical/

    EBBContBase
    EBBCont1L
    EBBCont2L
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numpy import array, exp, ndarray, pi, sqrt, squeeze

from rolland.track import (
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
)


@dataclass(kw_only=True)
class EBBCont(ABC):
    r"""Abstract base class for continuous analytical methods.

    Attributes
    ----------
    force : numpy.ndarray
        Force amplitude corresponding to the excitation frequencies :math:`[N]`.
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    x_excit : float
        Excitation point :math:`[m]`.
    x : float | list[float] | numpy.ndarray
        Distances to the excitation point :math:`[m]`.
    mobility : numpy.ndarray
        Calculated mobility of the track :math:`[m/N]`.
    """

    f: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    force: ndarray = field(default_factory=lambda: array(1.0), metadata={"default_repr": "numpy.array([1.0])"})
    x_excit: float = 0.0
    x: float | list[float] | ndarray = 0.0
    mobility: ndarray = field(init=False, default_factory=lambda: array([]),
                              metadata={"default_repr": "numpy.array([])"})

    def __post_init__(self) -> None:
        """Post-initialization to set defaults, validate track and compute mobility."""
        self._validate_track()
        self._set_default_x()
        self.compute_mobility()

    def _set_default_x(self) -> None:
        """Set default value for x if it is 0.0."""
        if isinstance(self.x, float) and self.x == 0.0:
            self.x = self.x_excit

    @property
    def omega(self) -> ndarray:
        """Calculate the angular frequency."""
        return 2 * pi * self.f

    @property
    def receptance(self) -> ndarray:
        """Calculate the receptance of the track :math:`[m/N]`."""
        return self.mobility / (1j * self.omega)

    @property
    def accelerance(self) -> ndarray:
        """Calculate the accelerance of the track :math:`[m/(s^2 N)]`."""
        return self.mobility * (1j * self.omega)

    @abstractmethod
    def _validate_track(self) -> None:
        """Validate track type."""

    @abstractmethod
    def compute_mobility(self) -> None:
        """
        Compute the mobility of the track.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        msg = "Subclasses must implement compute_mobility."
        raise NotImplementedError(msg)


@dataclass(kw_only=True)
class EBBCont1L(EBBCont):
    r"""Method for continuous slab single rail track according to :cite:t:`thompson2024j`.

    Utilizes a single-layer support with continuous track properties, applying Euler-Bernoulli beam
    theory. The excitation is stationary, and the corresponding method calculates the track's
    mobility for the positions specified.

    Attributes
    ----------
    track : ContSlabSingleRailTrack
        Track instance.
    omega_0 : float
        Resonance frequency rail <--> foundation :math:`[Hz]`.
    """

    track: ContSlabSingleRailTrack
    omega_0: float = field(init=False, default=0.0)

    def _validate_track(self) -> None:
        """Validate track type."""
        if not isinstance(self.track, ContSlabSingleRailTrack):
            msg = f"Expected ContSlabSingleRailTrack, got {type(self.track).__name__}"
            raise TypeError(msg)

    def compute_mobility(self) -> None:
        r"""
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a continuous slab single rail track.
        """
        mr = self.track.rail.mr
        sp = self.track.pad.sp_z
        dp = self.track.pad.dp_z

        # Eq. 3.5
        self.omega_0 = float(sqrt(sp / mr))

        # Eq. 3.6
        k_p = ((self.omega ** 2 * mr - sp - 1j * self.omega * dp) /
               (self.track.rail.E * self.track.rail.Iyr)) ** (1/4)

        abs_x = abs(array(self.x, ndmin=1)[:, None] - self.x_excit)  # Broadcast x over omega
        term1 = exp(-1j * k_p * abs_x)
        term2 = -1j * exp(-k_p * abs_x)

        # Eq. 3.17 / 3.18
        self.mobility = (self.omega / (4 * (self.track.rail.E * self.track.rail.Iyr) * k_p ** 3)
                         * (term1 + term2))
        self.mobility = squeeze(self.mobility) # Remove axes of length one


@dataclass(kw_only=True)
class EBBCont2L(EBBCont):
    r"""Method for continuous ballasted single rail track according to :cite:t:`thompson2024j`.

    Utilizes a double-layer support with continuous track properties, applying Euler-Bernoulli beam
    theory. The excitation is stationary, and the corresponding method calculates the track's
    mobility for the positions specified.

    Attributes
    ----------
    track : ContBallastedSingleRailTrack
        Track instance.
    omega_0 : float
        Resonance frequency rail <--> foundation :math:`[Hz]`.
    omega_1 : float
        Resonance frequency ballast <--> slab :math:`[Hz]`.
    omega_2 : float
        Resonance frequency rail <--> slab :math:`[Hz]`.
    """

    track: ContBallastedSingleRailTrack
    omega_0: float = field(init=False, default=0.0)
    omega_1: float = field(init=False, default=0.0)
    omega_2: float = field(init=False, default=0.0)

    def _validate_track(self) -> None:
        """Validate track type."""
        if not isinstance(self.track, ContBallastedSingleRailTrack):
            msg = f"Expected ContBallastedSingleRailTrack, got {type(self.track).__name__}"
            raise TypeError(msg)

    def compute_mobility(self) -> None:
        r"""
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a continuous ballasted single rail track.
        """
        mr = self.track.rail.mr
        sp = self.track.pad.sp_z
        sb = self.track.ballast.sb_z
        dp = self.track.pad.dp_z
        db = self.track.ballast.db_z
        ms = self.track.slab.ms

        self.omega_0 = float(sqrt(sp / mr))            # Eq. 3.47
        self.omega_1 = float(sqrt(sb / ms))            # Eq. 3.44
        self.omega_2 = float(sqrt((sp + sb) / ms))     # Eq. 3.44

        # Eq. 3.40
        sp_tot = sp + 1j * self.omega * dp
        sb_tot = sb + 1j * self.omega * db
        s_tot = (sp_tot * (sb_tot - ms * self.omega ** 2)) / (sp_tot + sb_tot - ms * self.omega ** 2)

        # Eq. 3.6
        k_p = ((self.omega ** 2 * mr - s_tot) /
                (self.track.rail.E * self.track.rail.Iyr)) ** (1/4)

        abs_x = abs(array(self.x, ndmin=1)[:, None] - self.x_excit)  # Broadcast x over omega
        term1 = exp(-1j * k_p * abs_x)
        term2 = -1j * exp(-k_p * abs_x)

        # Eq. 3.17 / 3.18
        self.mobility = (self.omega / (4 * (self.track.rail.E * self.track.rail.Iyr) * k_p ** 3)
                         * (term1 + term2))
        self.mobility = squeeze(self.mobility) # Remove axes of length one
