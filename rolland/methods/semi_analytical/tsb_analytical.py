"""Analytical models for discrete tracks using Timoshenko Beam (TB) theory.

This module provides analytical solutions for the dynamic response of railway tracks modeled
as discretely supported Timoshenko beams. These models account for shear deformation and
rotational inertia, making them accurate for higher frequencies, and incorporate the discrete
nature of sleepers/supports (pinned-pinned resonance). The module calculates structural
responses such as mobility, receptance, and accelerance for both slab and ballasted tracks.

.. autosummary::
    :toctree: analytical/

    TSBDiscr
    TSBDiscr1L
    TSBDiscr2L
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from numpy import array, exp, eye, lib, linalg, ndarray, newaxis, pi, real, sqrt, squeeze
from numpy import errstate as np_errstate

from rolland.track import (
    DiscrBallastedSingleRailTrack,
    DiscrSlabSingleRailTrack,
)


@dataclass(kw_only=True)
class TSBDiscr(ABC):
    r"""Abstract base class for discrete analytical methods using Timoshenko beam theory.

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
    damp_type : Literal['viscous', 'hysteretic']
        Damping type, either 'viscous' or 'hysteretic'. The viscous damping
        approach is an adopted version of :cite:t:`heckl1995`.
    mobility : numpy.ndarray
        Calculated mobility of the track :math:`[m/(s \cdot N)]`.
    """

    f: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    force: ndarray = field(default_factory=lambda: array(1.0), metadata={"default_repr": "numpy.array([1.0])"})
    x_excit: float = 0.0
    x: float | list[float] | ndarray = 0.0
    damp_type: Literal["viscous", "hysteretic"]
    mobility: ndarray = field(init=False, default_factory=lambda: array([]),
                              metadata={"default_repr": "numpy.array([])"})

    def __post_init__(self):
        """Post-initialization to set defaults, validate track and compute mobility."""
        if self.damp_type not in ("viscous", "hysteretic"):
            msg = f"Invalid damp_type: {self.damp_type}. Must be 'viscous' or 'hysteretic'."
            raise ValueError(msg)
        self._validate_track()
        self._set_default_x()
        self.compute_mobility()

    def _set_default_x(self):
        """Set default value for x if it is 0.0, and convert x to array."""
        if isinstance(self.x, float) and self.x == 0.0:
            self.x = array([self.x_excit])
        else:
            self.x = array(self.x, ndmin=1)

    @property
    def omega(self):
        """Calculate the angular frequency."""
        return 2 * pi * self.f

    @property
    def receptance(self):
        """Calculate the receptance of the track :math:`[m/N]`."""
        with np_errstate(divide='ignore', invalid='ignore'):
            return self.mobility / (1j * self.omega)

    @property
    def accelerance(self):
        """Calculate the accelerance of the track :math:`[m/(s^2 N)]`."""
        return self.mobility * (1j * self.omega)

    def calc_greens_func(self, dist, k_p, k_d, f_p, f_d):
        """
        Calculate Greens function of free Timoshenko Beam (eq. 3.69).

        Parameters
        ----------
        dist : numpy.ndarray
            Absolute distances between points :math:`[m]`.
        k_p : numpy.ndarray
            Wave number of propagating wave :math:`[1/m]`.
        k_d : numpy.ndarray
            Wave number of decaying wave :math:`[1/m]`.
        f_p : numpy.ndarray
            Factor for propagating wave :math:`[-]`.
        f_d : numpy.ndarray
            Factor for decaying wave :math:`[-]`.

        Returns
        -------
        numpy.ndarray
            Calculated Greens function.
        """
        term1 = exp(-1j * k_p * dist)
        term2 = exp(-1j * k_d * dist)
        return f_p * term1 + f_d * term2

    def compute_mobility_common(self, track, ms, sb, etab, db=0.0):
        """
        Compute common mobility for 1-layer and 2-layer support.

        Parameters
        ----------
        track : object
            Track instance containing rail and pad properties.
        ms : float
            Mass per unit length of the sleeper or slab [kg/m].
        sb : float
            Stiffness of the ballast [N/m^2].
        etab : float
            Damping coefficient of the ballast.
        db : float, optional
            Viscous damping coefficient of the ballast [Ns/m^2].
        """
        if self.f.size == 0:
            self.mobility = array([])
            return

        e_z = track.E[1]
        ms = ms * e_z
        sb = sb * e_z
        db = db * e_z

        mr = track.rail.mr
        rho = track.rail.rho
        kap = track.rail.kapz
        youm = track.rail.E * (1 + 1j * track.rail.etar)
        shearm = track.rail.G * (1 + 1j * track.rail.etar)
        ar = track.rail.Ar
        aream = track.rail.Iyr
        sp = track.pad.sp_z
        sb = sb

        etap = getattr(track.pad, "etap_z", 0.0)
        dp = getattr(track.pad, "dp_z", 0.0)

        # Positions of point forces [m]
        x_n = array(list(track.mount_prop.keys()))

        # Resonance frequencies
        self.f_0 = real(sqrt(sp / mr)) / (2 * pi)
        self.f_1 = real(sqrt(sb / ms)) / (2 * pi)
        self.f_2 = real(sqrt(sp + sb)) / (2 * pi)

        # Dynamic stiffness (eq. 3.68)
        if self.damp_type == "viscous":
            k_pad = sp + 1j * self.omega * dp
            k_bal = sb + 1j * self.omega * db
        else:
            k_pad = sp * (1 + 1j * etap)
            k_bal = sb * (1 + 1j * etab)

        impend = (k_pad * (k_bal - (ms * (self.omega ** 2)))) / (k_pad + k_bal - (ms * (self.omega ** 2)))

        # Eq. 3.72
        c1 = ((shearm * kap * ar) / (youm * aream)) - ((rho * self.omega ** 2 * aream) / (youm * aream))

        # Eq. 3.73
        c2 = - ((mr * self.omega ** 2) / (shearm * kap * ar)) - ((rho * self.omega ** 2 * aream) / (youm * aream))

        # Eq. 3.74
        c3 = ((mr * self.omega ** 2) / (youm * aream)) * ((rho * self.omega ** 2 * aream) / (shearm * kap * ar) - 1)

        # Eq. 3.71
        k__p = -1 / 2 * c2 + 1 / 2 * sqrt(c2 ** 2 - 4 * c3)
        k__d = -1 / 2 * c2 - 1 / 2 * sqrt(c2 ** 2 - 4 * c3)

        k_p_raw = lib.scimath.sqrt(k__p)
        k_d_raw = -1 * lib.scimath.sqrt(k__d)

        # Ensure correct branch cuts for spatial decay
        k_p = array([ -k if k.imag > 0 else k for k in k_p_raw.flat ]).reshape(k_p_raw.shape)
        # For k_d which is typically imaginary, ensure positive real and
        # negative imaginary parts for decay
        k_d = array([ -k if k.imag > 0 else k for k in k_d_raw.flat ]).reshape(k_d_raw.shape)


        # Eq. 3.70
        f_p = (-1j / (shearm * ar * kap)) * ((k_p ** 2 + c1) / (4 * k_p ** 3 + 2 * k_p * c2))
        f_d = (-1j / (shearm * ar * kap)) * ((k_d ** 2 + c1) / (4 * k_d ** 3 + 2 * k_d * c2))

        # Precompute distance matrices for spatial distances (independent of frequency)
        dist_mn = abs(x_n[:, newaxis] - x_n[newaxis, :])  # (N, N)
        dist_exc = abs(x_n - self.x_excit)                # (N,)
        dist_xn = abs(self.x[:, newaxis] - x_n[newaxis, :]) # (P, N)
        dist_xf = abs(self.x - self.x_excit)              # (P,)

        n_points = x_n.size

        # Reshape factors for broadcasting over F frequencies (F, 1, 1) or (F, 1)
        k_p_f = k_p[:, newaxis, newaxis]
        k_d_f = k_d[:, newaxis, newaxis]
        f_p_f = f_p[:, newaxis, newaxis]
        f_d_f = f_d[:, newaxis, newaxis]
        k_p_1 = k_p[:, newaxis]
        k_d_1 = k_d[:, newaxis]
        f_p_1 = f_p[:, newaxis]
        f_d_1 = f_d[:, newaxis]

        # Greens function matrix reaction points <--> reaction points
        greensm_mn = self.calc_greens_func(dist_mn[newaxis, :, :], k_p_f, k_d_f, f_p_f, f_d_f)

        # Greens function matrix reaction points <--> excitation point
        greensm_exc = self.calc_greens_func(dist_exc[newaxis, :], k_p_1, k_d_1, f_p_1, f_d_1)

        # m = I + impend * greensm_mn
        m = eye(n_points)[newaxis, :, :] + impend[:, newaxis, newaxis] * greensm_mn

        # Calculate displacements at reaction points (eq. 3.76)
        # linalg.solve treats 2D `b` arrays as a single matrix.
        # We need to reshape greensm_exc to (F, N, 1) so it's treated as a batch of column vectors.
        uxn = linalg.solve(m, greensm_exc[:, :, newaxis])  # Output shape is (F, N, 1)

        # Greens function matrix requested points <--> reaction points
        greensm_xn = self.calc_greens_func(dist_xn[newaxis, :, :], k_p_f, k_d_f, f_p_f, f_d_f)

        # Greens function matrix requested points <--> excitation point
        greensm_xf = self.calc_greens_func(dist_xf[newaxis, :], k_p_1, k_d_1, f_p_1, f_d_1)

        # Batch multiply: (F, P, N) @ (F, N, 1) -> (F, P, 1)
        ux_reaction = (greensm_xn @ uxn)[..., 0]

        # Total displacement, shape (F, P)
        ux_total = -impend[:, newaxis] * ux_reaction + greensm_xf

        # Original expects self.ux of shape (P, F)
        self.ux = ux_total.T

        with np_errstate(divide='ignore', invalid='ignore'):
            self.mobility = (self.ux * self.omega * 1j) / self.force
        self.mobility = squeeze(self.mobility)  # Remove axes of length one

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
class TSBDiscr1L(TSBDiscr):
    r"""Method for discrete slab track according to :cite:t:`thompson2024j` and :cite:t:`heckl1995`.

    Utilizes a single-layer support with discrete track properties, applying Timoshenko beam
    theory. The excitation is a non-moving sound source. The corresponding method calculates the
    track's mobility for the positions specified.

    .. caution::
        This method is an implementation of :cite:t:`thompson2024j` which is a modified version of
        :cite:t:`heckl1995`. Theoretically, the results should be identical, but Heckl's work
        contains the following mistakes, which lead to incorrect results:

        1. Missing negativ sign in the definition of the decaying wave number (Eq. 2b).
        2. Shear modulus :math:`G` needs to be substituted by :math:`G * \kappa`

    Attributes
    ----------
    track : DiscrSlabSingleRailTrack
        Track instance.
    damp_type : Literal['viscous', 'hysteretic']
        Damping type, either 'viscous' or 'hysteretic'. The viscous damping
        approach is an adopted version of :cite:t:`heckl1995`.
    f_0 : float
        Resonance frequency rail <--> foundation :math:`[Hz]`.
    force : numpy.ndarray
        Excitation force array.
    f : numpy.ndarray
        Frequency array.
    x_excit : float
        Excitation position.
    x : float | list[float] | numpy.ndarray
        Response positions.
    mobility : numpy.ndarray
        Calculated mobility.
    """

    track: DiscrSlabSingleRailTrack
    f_0: float = field(init=False, default=0.0)

    def _validate_track(self):
        """Validate method."""
        if not isinstance(self.track, DiscrSlabSingleRailTrack):
            msg = f"Expected DiscrSlabSingleRailTrack, got {type(self.track).__name__}"
            raise TypeError(msg)

    def compute_mobility(self):
        """
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a discrete slab track.
        """
        self.compute_mobility_common(self.track, self.track.slab.ms, 1e20, 0, 0.0)


@dataclass(kw_only=True)
class TSBDiscr2L(TSBDiscr):
    r"""Method for discrete ballasted track according to :cite:t:`thompson2024j`.

    Utilizes a double-layer support with discrete track properties, applying Timoshenko beam
    theory. The excitation is a non-moving sound source. The corresponding method calculates the
    track's mobility for the positions specified.

    .. caution::
        This method is an implementation of :cite:t:`thompson2024j` which is a modified version of
        :cite:t:`heckl1995`. Theoretically, the results should be identical, but Heckl's work
        contains the following mistakes, which lead to incorrect results:

        1. Missing negativ sign in the definition of the decaying wave number (Eq. 2b).
        2. Shear modulus :math:`G` needs to be substituted by :math:`G * \kappa`

    Attributes
    ----------
    track : DiscrBallastedSingleRailTrack
        Track instance.
    damp_type : Literal['viscous', 'hysteretic']
        Damping type, either 'viscous' or 'hysteretic'. The viscous damping
        approach is an adopted version of :cite:t:`heckl1995`.
    f_0 : float
        Resonance frequency rail <--> foundation :math:`[Hz]`.
    f_1 : float
        Resonance frequency ballast <--> slab :math:`[Hz]`.
    f_2 : float
        Resonance frequency rail <--> slab :math:`[Hz]`.
    force : numpy.ndarray
        Excitation force array.
    f : numpy.ndarray
        Frequency array.
    x_excit : float
        Excitation position.
    x : float | list[float] | numpy.ndarray
        Response positions.
    mobility : numpy.ndarray
        Calculated mobility.
    """

    track: DiscrBallastedSingleRailTrack
    f_0: float = field(init=False, default=0.0)
    f_1: float = field(init=False, default=0.0)
    f_2: float = field(init=False, default=0.0)

    def _validate_track(self):
        """Validate method."""
        if not isinstance(self.track, DiscrBallastedSingleRailTrack):
            msg = f"Expected DiscrBallastedSingleRailTrack, got {type(self.track).__name__}"
            raise TypeError(msg)

    def compute_mobility(self):
        """
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a discrete ballasted track.
        """
        self.compute_mobility_common(self.track, self.track.sleeper.ms,
                                     self.track.ballast.sb_z, getattr(self.track.ballast, "etab_z", 0.0),
                                     getattr(self.track.ballast, "db_z", 0.0))

