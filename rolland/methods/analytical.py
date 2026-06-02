"""Contains analytical methods that allow comparison with FDM results.

.. autosummary::
    :toctree: analytical/

    AnalyticalMethods
    EBBCont1LSupp
    EBBCont2LSupp
    TBDiscr
    TSDiscr1LSupp
    TSDiscr2LSupp
"""
import abc

from numpy import array, exp, eye, lib, linalg, newaxis, pi, real, sqrt, squeeze, zeros
from traitlets import Float, Instance, List, Union, observe
from traittypes import Array

from rolland.track import (
    ABCHasTraits,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    DiscrBallastedSingleRailTrack,
    DiscrSlabSingleRailTrack,
)


class AnalyticalMethods(ABCHasTraits):
    r"""Abstract base class for analytical methods.

    Attributes
    ----------
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    force : numpy.ndarray
        Force amplitude corresponding to the excitation frequencies :math:`[N]`.
    x : float or list
        Distances to the excitation point :math:`[m]`.
    x_excit : numpy.ndarray
        Excitation point :math:`[m]`.
    mobility : numpy.ndarray
        Calculated mobility of the track :math:`[m/N]`.
    """

    f = Array()
    force = Array(default_value=1.0)
    x_excit = Float(default_value=0.0)
    x = Union([Float(default_value=0.0), List()])
    mobility = Array()

    def __init__(self, **kwargs):
        """
        Initialize the AnalyticalMethods class.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to initialize the class attributes.
        """
        super().__init__(**kwargs)
        self.compute_mobility()

    @observe('f', 'force', 'x_excit', 'x')
    def _update_on_change(self, change):
        self.compute_mobility()

    @observe('x_excit')
    def _set_default_x(self, change):
        if self.x == 0.0:
            self.x = self.x_excit

    @property
    def omega(self):
        """Calculate the angular frequency."""
        return 2 * pi * self.f

    @abc.abstractmethod
    def compute_mobility(self):
        """
        Compute the mobility of the track.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        message = "Subclasses must implement compute_mobility."
        raise NotImplementedError(message)


class EBBCont1LSupp(AnalyticalMethods):
    r"""Method for continuous slab single rail track according to :cite:t:`thompson2024j`.

    Utilizes a single-layer support with continuous track properties, applying Euler-Bernoulli beam
    theory. The excitation is stationary, and the corresponding method calculates the track's
    mobility for the postions specified.

    Attributes
    ----------
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    force : numpy.ndarray
        Force amplitude corresponding to the excitation frequencies :math:`[N]`.
    x : float or list
        Distances to the excitation point :math:`[m]`.
    x_excit : numpy.ndarray
        Excitation point :math:`[m]`.
    track : ContSlabSingleRailTrack
        Track instance.
    """

    track = Instance(ContSlabSingleRailTrack)

    def compute_mobility(self):
        r"""
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a continuous slab single rail track.

        Attributes
        ----------
        mr : float
            Mass per unit length of the rail :math:`[kg/m]`.
        sp : float
            Stiffness of continuous pad :math:`[N/m^2]`.
        dp : float
            Viscous damping coefficient of continuous pad :math:`[Ns/m^2]`.
        omega_0 : float
            Resonance frequency rail <--> foundation :math:`[Hz]`.
        k_p : numpy.ndarray
            Wave number of propagating wave :math:`[1/m]`.
        abs_x : numpy.ndarray
            Absolute distance between given positions and the excitation point :math:`[m]`.
        mobility : numpy.ndarray
            Calculated mobility of the track :math:`[m/N]`.

        Returns
        -------
        mobility : numpy.ndarray
            The mobility of the track :math:`[m/N]`.
        omega_0 : float
            The resonance frequency rail <--> foundation :math:`[Hz]`.
        """
        mr = self.track.rail.mr
        sp = self.track.pad.sp[0]
        dp = self.track.pad.dp[0]

        # Eq. 3.5
        self.omega_0 = sqrt(sp / mr)

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


class EBBCont2LSupp(AnalyticalMethods):
    r"""Method for continuous ballasted single rail track according to :cite:t:`thompson2024j`.

    Utilizes a double-layer support with continuous track properties, applying Euler-Bernoulli beam
    theory. The excitation is stationary, and the corresponding method calculates the track's
    mobility for the postions specified.

    Attributes
    ----------
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    force : numpy.ndarray
        Force amplitude corresponding to the excitation frequencies :math:`[N]`.
    x : float or list
        Distances to the excitation point :math:`[m]`.
    x_excit : numpy.ndarray
        Excitation point :math:`[m]`.
    track : ContBallastedSingleRailTrack
        Track instance.
    omega_0 : float
        Resonance frequency rail <--> foundation :math:`[Hz]`.
    mobility : numpy.ndarray
        Mobility matrix :math:`[m/N]`.
    """

    track = Instance(ContBallastedSingleRailTrack)

    def compute_mobility(self):
        r"""
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a continuous slab single rail track.

        Attributes
        ----------
        mr : float
            Mass per unit length of the rail :math:`[kg/m]`.
        sp : float
            Stiffness of continuous pad :math:`[N/m^2]`.
        sb : float
            Stiffness of continuous ballast :math:`[N/m^2]`.
        dp : float
            Viscous damping coefficient of continuous pad :math:`[Ns/m^2]`.
        db : float
            Viscous damping coefficient of continuous ballast :math:`[Ns/m^2]`.
        ms : float
            Mass per unit length of the sleeper :math:`[kg/m]`.

        Returns
        -------
        mobility : numpy.ndarray
            The mobility of the track :math:`[m/N]`.
        omega_0 : float
            The resonance frequency rail <--> foundation :math:`[Hz]`.
        omega_1 : float
            The resonance frequency ballast <--> slab :math:`[Hz]`.
        omega_2 : float
            The resonance frequency rail <--> slab :math:`[Hz]`.
        """
        mr = self.track.rail.mr
        sp = self.track.pad.sp[0]
        sb = self.track.ballast.sb[0]
        dp = self.track.pad.dp[0]
        db = self.track.ballast.db[0]
        ms = self.track.slab.ms

        self.omega_0 = sqrt(sp / mr)            # Eq. 3.47
        self.omega_1 = sqrt(sb / ms)            # Eq. 3.44
        self.omega_2 = sqrt((sp + sb) / ms)     # Eq. 3.44

        # Eq. 3.40
        sp_tot = sp + 1j * self.omega * dp
        sb_tot = sb + 1j * self.omega * db
        s_tot = (sp_tot * (sb_tot - ms * self.omega ** 2)) / (sp_tot + sb_tot - ms * self.omega ** 2)

        # Eq. 3.6
        k_p = ((self.omega ** 2 * mr - s_tot) /
                (self.track.rail.E * self.track.rail.Iyr)) ** (1/4)

        abs_x = abs(array(self.x, ndmin=1)[:, None] - self.x_excit)  # Broadcast x over omega  # Broadcast x over omega
        term1 = exp(-1j * k_p * abs_x)
        term2 = -1j * exp(-k_p * abs_x)

        # Eq. 3.17 / 3.18
        self.mobility = (self.omega / (4 * (self.track.rail.E * self.track.rail.Iyr) * k_p ** 3)
                         * (term1 + term2))
        self.mobility = squeeze(self.mobility) # Remove axes of length one


class TBDiscr(AnalyticalMethods):
    r"""Base class for analytical solutions of a discrete single rail track.

    Attributes
    ----------
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    force : numpy.ndarray
        Force amplitude corresponding to the excitation frequencies :math:`[N]`.
    x : float or list
        Distances to the excitation point :math:`[m]`.
    x_excit : numpy.ndarray
        Excitation point :math:`[m]`.
    """

    @abc.abstractmethod
    def validate_method(self):
        """Validate method."""

    def calc_greens_func(self, xm, xn, k_p, k_d, f_p, f_d):
        """
        Calculate Greens function of free Timoshenko Beam (eq. 3.69).

        Parameters
        ----------
        xm : numpy.ndarray
            Positions of the first set of points :math:`[m]`.
        xn : numpy.ndarray
            Positions of the second set of points :math:`[m]`.
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
        dist = abs(xm - xn)
        term1 = exp(-1j * k_p * dist)
        term2 = exp(-1j * k_d * dist)
        term3 = 1
        return f_p * term1 + (f_d * term2) * term3

    def compute_mobility_common(self, track, ms, sb, etab):
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

        Attributes
        ----------
        f_0 : float
            Resonance frequency rail <--> pads [Hz].
        f_1 : float
            Resonance frequency ballast <--> sleepers [Hz].
        f_2 : float
            Resonance frequency rail <--> sleepers [Hz].
        mobility : numpy.ndarray
            Calculated mobility of the track [m/N].
        """
        mr = track.rail.mr
        rho = track.rail.rho
        etap = track.pad.etap
        kap = track.rail.kap[0]
        youm = track.rail.E
        shearm = track.rail.G
        ar = track.rail.Ar
        aream = track.rail.Iyr
        sp = track.pad.sp[0]
        sb = sb

        # Positions of point forces [m]
        x_n = array(list(track.mount_prop.keys()))

        # Resonance frequencies
        self.f_0 = real(sqrt(sp / mr)) / (2 * pi)
        self.f_1 = real(sqrt(sb / ms)) / (2 * pi)
        self.f_2 = real(sqrt(sp + sb)) / (2 * pi)

        # Dynamic stiffness (eq. 3.68)
        impend = ((sp * (1 + (1j * etap)) * ((sb * (1 + (1j * etab))) - (ms * (self.omega ** 2)))) /
                  ((sp * (1 + (1j * etap))) + (sb * (1 + (1j * etab))) - (ms * (self.omega ** 2))))

        # Eq. 3.72
        c1 = ((shearm * kap * ar) / (youm * aream)) - ((rho * self.omega ** 2 * aream) / (youm * aream))

        # Eq. 3.73
        c2 = - ((mr * self.omega ** 2) / (shearm * kap * ar)) - ((rho * self.omega ** 2 * aream) / (youm * aream))

        # Eq. 3.74
        c3 = ((mr * self.omega ** 2) / (youm * aream)) * ((rho * self.omega ** 2 * aream) / (shearm * kap * ar) - 1)

        # Eq. 3.71
        k__p = -1 / 2 * c2 + 1 / 2 * sqrt(c2 ** 2 - 4 * c3)
        k__d = -1 / 2 * c2 - 1 / 2 * sqrt(c2 ** 2 - 4 * c3)

        k_p = lib.scimath.sqrt(k__p)
        k_d = -1 * lib.scimath.sqrt(k__d)

        # Eq. 3.70
        f_p = (-1j / (shearm * ar * kap)) * ((k_p ** 2 + c1) / (4 * k_p ** 3 + 2 * k_p * c2))
        f_d = (-1j / (shearm * ar * kap)) * ((k_d ** 2 + c1) / (4 * k_d ** 3 + 2 * k_d * c2))

        # Displacements at reaction points
        uxn = zeros((self.f.size, x_n.size), dtype=complex)

        # Displacements at requested points
        self.ux = zeros((array(self.x).size, self.f.size), dtype=complex)

        for f in range(self.f.size):

            # Greens function matrix reaction points <--> reaction points
            greensm_mn = self.calc_greens_func(x_n[:, newaxis], x_n[newaxis, :], k_p[f], k_d[f], f_p[f], f_d[f])

            # Greens function matrix reaction points <--> excitation point
            greensm_exc = self.calc_greens_func(x_n, self.x_excit, k_p[f], k_d[f], f_p[f], f_d[f])

            # m = I + impend * greensm_mn
            m = eye(x_n.size) + impend[f] * greensm_mn

            # Calculate displacements at reaction points (eq. 3.76)
            uxn[f, :] = linalg.solve(m, greensm_exc)

            # Calculate displacements at requested points (eq. 3.77)
            for p in range(array(self.x, ndmin=1).size):
                # Greens function matrix requested points <--> reaction points
                greensm_xn = self.calc_greens_func(array(self.x, ndmin=1)[p], x_n, k_p[f], k_d[f], f_p[f], f_d[f])

                # Greens function matrix requested points <--> excitation point
                greensm_xf = self.calc_greens_func(array(self.x, ndmin=1)[p],
                                                   self.x_excit, k_p[f], k_d[f], f_p[f], f_d[f])
                self.ux[p, f] = - impend[f] * greensm_xn.dot(uxn[f, :]) + greensm_xf

        self.mobility = (self.ux * self.omega * 1j) / self.force
        self.mobility = squeeze(self.mobility)  # Remove axes of length one


class TSDiscr1LSupp(TBDiscr):
    r"""Method for discrete slab track according to :cite:t:`thompson2024j` and :cite:t:`heckl1995`.

    Utilizes a single-layer support with discrete track properties, applying Timoshenko beam
    theory. The excitation is a non-moving sound source. The corresponding method calculates the
    track's mobility for the postions specified.

    .. caution::
        This method is an implementation of :cite:t:`thompson2024j` which is a modified version of
        :cite:t:`heckl1995`. Theoretically, the results should be identical, but Heckl's work
        contains the following mistakes, which lead to incorrect results:

        1. Missing negativ sign in the definition of the decaying wave number (Eq. 2b).
        2. Shear modulus :math:`G` needs to be substituted by :math:`G * \kappa`

    Attributes
    ----------
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    force : numpy.ndarray
        Force amplitude corresponding to the excitation frequencies :math:`[N]`.
    x : float or list
        Distances to the excitation point :math:`[m]`.
    x_excit : numpy.ndarray
        Excitation point :math:`[m]`.
    track : DiscrSlabSingleRailTrack
        Track instance.
    omega_0 : float
        Resonance frequency rail <--> foundation :math:`[Hz]`.
    """

    track = Instance(DiscrSlabSingleRailTrack)

    def validate_method(self):
        """Validate method."""

    def compute_mobility(self):
        """
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a discrete slab track.

        Attributes
        ----------
        mobility : numpy.ndarray
            Calculated mobility of the track :math:`[m/N]`.
        """
        self.compute_mobility_common(self.track, self.track.slab.ms, 1e20, 0)


class TSDiscr2LSupp(TBDiscr):
    r"""Method for discrete ballasted track according to :cite:t:`thompson2024j`.

    Utilizes a double-layer support with discrete track properties, applying Timoshenko beam
    theory. The excitation is a non-moving sound source. The corresponding method calculates the
    track's mobility for the postions specified.

    .. caution::
        This method is an implementation of :cite:t:`thompson2024j` which is a modified version of
        :cite:t:`heckl1995`. Theoretically, the results should be identical, but Heckl's work
        contains the following mistakes, which lead to incorrect results:

        1. Missing negativ sign in the definition of the decaying wave number (Eq. 2b).
        2. Shear modulus :math:`G` needs to be substituted by :math:`G * \kappa`

    Attributes
    ----------
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    force : numpy.ndarray
        Force amplitude corresponding to the excitation frequencies :math:`[N]`.
    x : float or list
        Distances to the excitation point :math:`[m]`.
    x_excit : numpy.ndarray
        Excitation point :math:`[m]`.
    track : DiscrSlabSingleRailTrack
        Track instance.
    omega_0 : float
        Resonance frequency rail <--> foundation :math:`[Hz]`.
    omega_1 : float
        Resonance frequency ballast <--> slab :math:`[Hz]`.
    omega_2 : float
        Resonance frequency rail <--> slab :math:`[Hz]`.
    """

    track = Instance(DiscrBallastedSingleRailTrack)

    def validate_method(self):
        """Validate method."""

    def compute_mobility(self):
        """
        Compute the mobility of the track.

        This method calculates the mobility of the track using the given parameters
        and the analytical solution for a discrete ballasted track.

        Attributes
        ----------
        mobility : numpy.ndarray
            Calculated mobility of the track :math:`[m/N]`.
        """
        self.compute_mobility_common(self.track, self.track.sleeper.ms,
                                     self.track.ballast.sb[0], self.track.ballast.etab)
