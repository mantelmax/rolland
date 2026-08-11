"""Contains numerical methods as developed by K.Stampka and E.Sarradj.

These methods utilize the Euler-Bernoulli Beam Theory and only calculate
the vertical beam dynamics (1 DOF per node). Eccentricities cannot be
represented, meaning that the mechanical properties of the foundation
and the excitation act directly at the rail centroid.

Katja Stampka and Ennes Sarradj.
A Time-Domain Finite-Difference Method for Bending Waves
on Infinite Beams on an Elastic Foundation.
Acoustics, January 2022. Num Pages: 18. doi:10.3390/acoustics4040052.

.. autosummary::
    :toctree: numerical/

    PMLStampka
    GaussianImpulseStampka
    DiscretizationStampka
    DeflectionStampka
    PostProcessing
    AnalyticPP
    RollandPP
    Response
    TDR
"""

# ruff: noqa: N806

import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from numpy import (  # noqa: A004
    array,
    convolve,
    empty,
    exp,
    linspace,
    ndarray,
    ones,
    pi,
    rint,
    squeeze,
    where,
    zeros,
)
from numpy.fft import fft, fftfreq
from scipy.sparse import SparseEfficiencyWarning, bmat, diags, eye
from scipy.sparse.linalg import splu

from rolland.methods import AnalyticalMethods
from rolland.track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
    Track,
)


# ---boundary.py---
@dataclass
class PMLStampka:
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
        """Calculate exponentially increasing rail damping for boundary domain."""
        # Exponential increasing rail damping, added to dr.
        return drbc * xbc ** self.alpha / self.l_bound ** self.alpha

@dataclass(kw_only=True)
class GaussianImpulseStampka:
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

    def force(self, t):
        """Compute force array (contains force over time)."""
        tg = t - 4 * self.sigma
        return self.a * tg / self.sigma ** 2 * exp(-tg ** 2 / self.sigma ** 2)


@dataclass(kw_only=True)
class DiscretizationStampka:
    r"""Discretization with non-time-dependent parameters according to :cite:t:`stampka2022a`.

    The parameters are constant over time. Only applicable for non-moving sound sources
    and linear superstructure properties.

    Attributes
    ----------
    track : Track
        Track instance.
    dt : float, default=2e-5
        Step size in time :math:`[s]`.
    req_simt : float, default=0.4
        Requested simulation time :math:`[s]`.
    bx : float, default=1.0
        Stability coefficient for dx calculation (must be :math:`b_x \geq 1`) :math:`[-]`.
    nt : int
        Number of time steps :math:`[-]`.
    sim_t : float
        Actual simulation time :math:`[s]`.
    dx : float
        Step size in space :math:`[m]`.
    bx_upd : float
        Updated stability coefficient :math:`[-]`.
    nx : int
        Number of spatial steps :math:`[-]`.
    bound : PMLStampka
        Boundary instance.
    n_bound : int
        Number of spatial steps in single sided boundary domain :math:`[-]`.
    pml : numpy.ndarray
        Damping array for boundary domain.
    A : scipy.sparse.csc_matrix
        Coefficient matrix A.
    B : scipy.sparse.csc_matrix
        Coefficient matrix B.
    C : scipy.sparse.csc_matrix
        Coefficient matrix C.
    vec_dr : numpy.ndarray
        Rail damping vector.
    vec_sp : numpy.ndarray
        Pad stiffness vector.
    vec_dp : numpy.ndarray
        Pad damping vector.
    vec_ms : numpy.ndarray
        Sleeper/Slab mass vector.
    vec_sb : numpy.ndarray
        Ballast stiffness vector.
    vec_db : numpy.ndarray
        Ballast damping vector.
    """

    track: Track
    bound: PMLStampka
    dt: float = 2e-5
    req_simt: float = 0.4
    bx: float = 1.0

    def calc_grid(self):
        """Calculate grid parameters."""
        self.nt = int(self.req_simt / self.dt)
        self.sim_t = self.nt * self.dt
        dx_min = (self.bx * ((self.track.rail.E * self.track.rail.Iyr) /
                             (6 * self.track.rail.mr)) ** (1 / 4) * self.dt ** (1 / 2))

        # 0.6 refers to the theoretical sleeper distance
        min_sleeper_dist = 0.6
        self.dx = min_sleeper_dist / (min_sleeper_dist // dx_min)
        self.bx_upd = self.dx / (((self.track.rail.E * self.track.rail.Iyr) /
                                  (6 * self.track.rail.mr)) ** (1 / 4) * self.dt ** (1 / 2))
        self.nx = int(self.track.l_track / self.dx) + 1

    def calc_bound(self):
        """Calculate boundary properties."""
        youm = self.track.rail.E
        shearm = self.track.rail.Iyr
        mr = self.track.rail.mr

        # Characteristic numerical value
        r = (youm * shearm) / mr *  self.dt ** 2 / self.dx ** 4

        # Maximum value of damping coefficient in boundary domain
        drbc = r / 2 * mr / self.dt

        # Lenght of single sided boundary domain
        self.n_bound = int(self.bound.l_bound / self.dx)

        # Grid points in boundary domain
        xbc = linspace(0, self.bound.l_bound, self.n_bound)

        # Calculate damping array for boundary domain
        self.pml = self.bound.pml(drbc, xbc)


    def build_matrix(
        self,
        vec_dr: ndarray,
        vec_sp: ndarray,
        vec_dp: ndarray,
        vec_ms: ndarray,
        vec_sb: ndarray,
        vec_db: ndarray,
    ):
        """Build matrices A, B, and C according to :cite:t:`stampka2022a`.

        Parameters
        ----------
        vec_dr : numpy.ndarray
            Rail damping vector.
        vec_sp : numpy.ndarray
            Pad stiffness vector.
        vec_dp : numpy.ndarray
            Pad damping vector.
        vec_ms : numpy.ndarray
            Sleeper/Slab mass vector.
        vec_sb : numpy.ndarray
            Ballast stiffness vector.
        vec_db : numpy.ndarray
            Ballast damping vector.

        Returns
        -------
        None
        """
        #   Suppressing warning
        warnings.simplefilter("ignore", category=SparseEfficiencyWarning)

        # simplification factor
        r = ((self.track.rail.E * self.track.rail.Iyr) * self.dt ** 2 /
             (2 * self.track.rail.mr * self.dx ** 4))

        # Coefficient matrix for x'''' (4th derivative)
        D_diagonals = [ones(self.nx - 2),
                       (-4) * ones(self.nx - 1),
                       6 * ones(self.nx),
                       (-4) * ones(self.nx - 1),
                       ones(self.nx - 2)]

        D = diags(D_diagonals, [-2, -1, 0, 1, 2])
        Eye = eye(self.nx)

        A11_1_diagonals = self.dt / self.track.rail.mr * (vec_dr + vec_dp)
        A11_1_diagonals += self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp
        A11_1 = diags([A11_1_diagonals], [0])
        A11 = (r * D + Eye + A11_1).tocsc()

        B11_1_diagonals = self.dt / self.track.rail.mr * (vec_dr + vec_dp)
        B11_1 = diags([B11_1_diagonals], [0])
        B11 = (2 * Eye + B11_1).tocsc()

        C11_1_diagonals = self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp
        C11_1 = diags([C11_1_diagonals], [0])
        C11 = (-(Eye + C11_1 + r * D)).tocsc()

        A12_diagonals = -self.dt / self.track.rail.mr * vec_dp
        A12_diagonals += -self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp
        A12 = diags([A12_diagonals], [0]).tocsc()

        A21_diagonals = -self.dt * vec_dp / vec_ms
        A21_diagonals += -self.dt ** 2 / (2 * vec_ms) * vec_sp
        A21 = diags([A21_diagonals], [0]).tocsc()

        A22_1_diagonals = self.dt * ((vec_dp + vec_db) / vec_ms)
        A22_1_diagonals += self.dt ** 2 / (2 * vec_ms) * (vec_sp + vec_sb)
        A22_1 = diags([A22_1_diagonals], [0])
        A22 = (Eye + A22_1).tocsc()

        B12_diagonals = -self.dt / self.track.rail.mr * vec_dp
        B12 = diags([B12_diagonals], [0]).tocsc()

        B21_diagonals = -self.dt * vec_dp / vec_ms
        B21 = diags([B21_diagonals], [0]).tocsc()

        B22_1_diagonals = self.dt * (vec_db + vec_dp) / vec_ms
        B22_1 = diags([B22_1_diagonals], [0])
        B22 = (2 * Eye + B22_1).tocsc()

        C12_diagonals = self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp
        C12 = diags([C12_diagonals], [0]).tocsc()

        C21_diagonals = self.dt ** 2 / (2 * vec_ms) * vec_sp
        C21 = diags([C21_diagonals], [0]).tocsc()

        C22_1_diagonals = self.dt ** 2 * (vec_sp + vec_sb) / (2 * vec_ms)
        C22_1 = diags([C22_1_diagonals], [0])
        C22 = (-(Eye + C22_1)).tocsc()

        self.A = bmat([[A11, A12], [A21, A22]], format='csc')
        self.B = bmat([[B11, B12], [B21, B22]], format='csc')
        self.C = bmat([[C11, C12], [C21, C22]], format='csc')




    def __post_init__(self):
        """Calculate superstructure property vectors."""
        self.calc_grid()
        self.calc_bound()
        self.initialize_vectors()
        self.add_boundary_conditions()
        self.build_superstructure_vectors()
        self.build_matrix(self.vec_dr, self.vec_sp, self.vec_dp, self.vec_ms, self.vec_sb, self.vec_db)

    def initialize_vectors(self):
        """Initialize the vectors."""
        self.vec_dr = self.track.rail.dr * ones(self.nx)
        self.vec_sp = zeros(self.nx)
        self.vec_dp = zeros(self.nx)
        self.vec_ms = ones(self.nx)    # ones instead of zeros to avoid division by zero
        self.vec_sb = zeros(self.nx)
        self.vec_db = zeros(self.nx)

    def add_boundary_conditions(self):
        """
        Add boundary conditions to the rail damping vector.

        This method modifies the rail damping vector (`vec_dr`) by adding the boundary conditions
        from the Perfectly Matched Layer (PML) at both ends of the grid.
        """
        # Boundary condition left side
        self.vec_dr[:self.pml.size] += self.pml[::-1]
        # Boundary condition right side
        self.vec_dr[-self.pml.size:] += self.pml

    def build_superstructure_vectors(self):
        """Handle track-specific logic.

        This method initializes the superstructure property vectors based on the type of track.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if isinstance(self.track, ContSlabSingleRailTrack):
            # Properties are assigned to each grid point
            self.vec_sp += self.track.pad.sp_z
            self.vec_dp += self.track.pad.dp_z
            self.vec_ms += self.track.slab.ms

        elif isinstance(self.track, SimplePeriodicSlabSingleRailTrack | ArrangedSlabSingleRailTrack):
            self.build_discrete_slab_track()

        elif isinstance(self.track, ContBallastedSingleRailTrack):
            # Properties are assigned to each grid point
            self.vec_sp += self.track.pad.sp_z
            self.vec_dp += self.track.pad.dp_z
            self.vec_ms += self.track.slab.ms
            self.vec_sb += self.track.ballast.sb_z
            self.vec_db += self.track.ballast.db_z

        elif isinstance(self.track, SimplePeriodicBallastedSingleRailTrack | ArrangedBallastedSingleRailTrack):
            self.build_discrete_ballasted_track()

        else:
            msg = "Track type not recognized!"
            raise ValueError(msg)

    def build_discrete_slab_track(self):
        """
        Build discrete slab track.

        Properties are assigned to the corresponding mounting positions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Get mounting positions as list
        mount_pos = list(self.track.mount_prop.keys())
        for i in mount_pos:
            x_ind = int(i / self.dx)
            self.vec_sp[x_ind] = self.track.mount_prop[i][0].sp_z / self.dx
            self.vec_dp[x_ind] = self.track.mount_prop[i][0].dp_z / self.dx
            self.vec_ms[x_ind] = self.track.slab.ms / self.dx

    def build_discrete_ballasted_track(self):
        """Build discrete ballasted track.

        Properties are assigned to the corresponding mounting positions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Get mounting positions as list
        mount_pos = list(self.track.mount_prop.keys())
        for i in mount_pos:
            x_ind = int(i / self.dx)
            self.vec_sp[x_ind] = self.track.mount_prop[i][0].sp_z / self.dx
            self.vec_dp[x_ind] = self.track.mount_prop[i][0].dp_z / self.dx
            self.vec_ms[x_ind] = self.track.mount_prop[i][1].ms / self.dx
            self.vec_sb[x_ind] = self.track.mount_prop[i][2].sb_z / self.dx
            self.vec_db[x_ind] = self.track.mount_prop[i][2].db_z / self.dx

#---deflection.py---
@dataclass(kw_only=True)
class DeflectionStampka:
    r"""Calculate deflection according to :cite:t:`stampka2022a`.

    Attributes
    ----------
    track : Track
        Track instance.
    excit : GaussianImpulseStampka
        Excitation instance.
    discr : Discretization
        Discretization instance.
    deflection : numpy.ndarray
        Deflection array :math:`[m]`.
    ind_excit : int
        Index of excitation point :math:`[-]`.
    """

    discr: DiscretizationStampka
    excit: GaussianImpulseStampka
    track: Track = field(init=False)

    def __post_init__(self, *args, **kwargs):
        """post_init method to set track attribute after initialization."""
        self.track = self.discr.track

    deflection: ndarray | None = field(default=None, init=False)

    def solve(self):
        """
        Solve the deflection equation.

        Returns
        -------
        None
        """
        # Initialize starting values
        self.calc_force()
        defl = self.initialize_start_values()
        # Calculate deflection
        self.deflection = self.calc_deflection(defl)

    def initialize_start_values(self):
        """Set starting values of deflections to zero.

        Returns
        -------
        defl : numpy.ndarray
            Array of deflections initialized to zero with shape (2 * nx, nt + 1).
        """
        defl = empty((2 * self.discr.nx, self.discr.nt + 1))

        # Set starting values to zero for two time steps
        defl[:, 0:2] = zeros((2 * self.discr.nx, 2))
        return defl

    def calc_force(self):
        """Calculate force array."""
        t = linspace(0, self.discr.sim_t, self.discr.nt)
        self.force = self.excit.force(t)

    def calc_rightside_crank_nicolson(self, u1, u0, ind_excit, t):
        """Calculate the right-hand side of the equation according to :cite:t:`stampka2022a`.

        Parameters
        ----------
        u1 : numpy.ndarray
            Deflection array at the current time step.
        u0 : numpy.ndarray
            Deflection array at the previous time step.
        ind_excit : int
            Index of the excitation point.
        t : int
            Current time step.

        Returns
        -------
        numpy.ndarray
            Right-hand side of the equation.
        """
        # Write excitation force for time step t into force array
        f = zeros(2 * self.discr.nx)

        f[ind_excit] = self.force[t]

        return self.discr.B.dot(u1) + self.discr.C.dot(u0) + self.discr.dt**2 / (self.track.rail.mr * self.discr.dx) * f

    def calc_deflection(self, defl):
        """
        Calculate deflection.

        Parameters
        ----------
        defl : numpy.ndarray
            Array of deflections initialized to zero with shape (2 * nx, nt + 1).

        Returns
        -------
        defl : numpy.ndarray
            Array of calculated deflections with shape (2 * nx, nt + 1).
        """
        # Index of excitation point/points
        if isinstance(self.excit.x_excit, list):
            self.ind_excit = [int(x / self.discr.dx) for x in self.excit.x_excit]
        else:
            self.ind_excit = int(self.excit.x_excit / self.discr.dx)

        # Factorization of matrix A (LU decomposition)
        factoriz = splu(self.discr.A)

        for t in range(1, self.discr.nt):
            # Calculate right hand side of equation
            b = self.calc_rightside_crank_nicolson(u1=defl[:, t], u0=defl[:, t - 1], ind_excit=self.ind_excit, t=t)

            # Calculate deflection for time step t
            u = factoriz.solve(b)

            defl[:, t + 1] = u[0 : 2 * self.discr.nx]
        return defl

#---postprocessing.py---
class PostProcessing:
    r"""Abstract base class for postprocessing classes."""

    @staticmethod
    def fast_fourier_tr(tsignal, dt):
        """Calculate the Fast Fourier Transform (FFT) of a time signal.

        Parameters
        ----------
        tsignal : numpy.ndarray
            Time signal to transform.
        dt : float
            Time step between samples.

        Returns
        -------
        tuple
            Frequencies and FFT of the signal.
        """
        samples = len(tsignal)
        window = ones(samples)
        fftrans = 2.0 / samples * fft(tsignal[:samples] * window)
        fftfre = fftfreq(samples, dt)
        return fftfre[0 : samples // 2], fftrans[0 : samples // 2]

    @staticmethod
    def plot(
        arrays, labels, title='Universal Plot', x_label='X-axis', y_label='Y-axis', colors=None, plot_type='loglog',
    ):
        """Universal plot function for multiple data sets.

        Parameters
        ----------
        arrays : list of tuple
            List of tuples, where each tuple contains two numpy.ndarray (x and y data).
        labels : list of str
            List of labels for each array.
        title : str, optional
            Title of the plot. Default is 'Universal Plot'.
        x_label : str, optional
            Label for the x-axis. Default is 'X-axis'.
        y_label : str, optional
            Label for the y-axis. Default is 'Y-axis'.
        colors : list of str, optional
            List of colors for each array. Default is None.
        plot_type : str, optional
            Type of plot (e.g., 'loglog', 'plot'). Default is 'loglog'.
        """
        plt.figure(figsize=(10, 6))
        if colors is None:
            colors = ['k', 'r', 'b', 'g', 'c', 'm', 'y']

        for (x, y), label, color in zip(arrays, labels, colors, strict=False):
            if plot_type == 'loglog':
                plt.loglog(x, y, label=label, color=color)
            else:
                plt.plot(x, y, label=label, color=color)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


class AnalyticPP(PostProcessing):
    r"""Analytic postprocessing class.

    This class is used to perform postprocessing on analytical methods.

    Attributes
    ----------
    results : AnalyticalMethods
        Instance of the AnalyticalMethods class containing the results.
    """

    def __init__(self, results: AnalyticalMethods):
        """Initialize AnalyticPP.

        Parameters
        ----------
        results : AnalyticalMethods
            Instance of the AnalyticalMethods class containing the results.
        """
        self.results = results

    @property
    def f(self):
        """Frequency vector."""
        return self.results.f

    @property
    def vb(self):
        """Velocity vector."""
        return self.results.mobility * self.results.force

    @property
    def ub(self):
        """Displacement vector."""
        return self.vb / (self.results.omega * 1j)


@dataclass(kw_only=True)
class RollandPP(PostProcessing):
    r"""Rolland postprocessing base class.

    This class is used to perform postprocessing on Rolland methods.

    Attributes
    ----------
    results : DeflectionStampka
        Instance of the Deflection class containing the results.
    f_min : float
        Minimum frequency for response calculation :math:`[Hz]`.
    f_max : float
        Maximum frequency for response calculation :math:`[Hz]`.
    """

    results: DeflectionStampka
    f_min: float = 100.0
    f_max: float = 3000.0


@dataclass(kw_only=True)
class Response(RollandPP):
    r"""Postprocessing class for Rolland response quantities.

    This class calculates and stores response quantities such as receptance,
    mobility, and accelerance based on the results of the Deflection class.

    Attributes
    ----------
    results : DeflectionStampka
        Instance of the Deflection class containing the results.
    x_resp : list[float] | None
        List of response points in meters :math:`[m]` (default value is x_excit).
    ind_resp : list[int] | None
        List of response indices (None if x_resp is provided).
    freq : numpy.ndarray
        Frequency vector :math:`[Hz]`.
    rez : numpy.ndarray
        Receptance vector :math:`[m/N]`.
    mob : numpy.ndarray
        Mobility vector :math:`[m/Ns]`.
    accel : numpy.ndarray
        Accelerance vector :math:`[m/Ns^2]`.
    """

    x_resp: list[float] | None = None
    ind_resp: list[int] | None = None
    freq: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    rez: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    mob: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    accel: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})

    def __post_init__(self):
        """Initialize Response and calculate response quantities."""
        self.calculate_response()

    def calculate_response(self):
        """Calculate and store response quantities (Receptance, Mobility, Accelerance)."""
        if self.x_resp is None and self.ind_resp is None:
            self.x_resp = [self.results.discr.dx * self.results.ind_excit]
            self.ind_resp = [int(x / self.results.discr.dx) for x in self.x_resp]

        elif self.x_resp is None and self.ind_resp is not None:
            self.x_resp = [(x * self.results.discr.dx) for x in self.ind_resp]

        else:
            self.ind_resp = [int(x / self.results.discr.dx) for x in self.x_resp]

        # Compute force FFT once
        fftfre, ffft = self.fast_fourier_tr(self.results.force, self.results.discr.dt)

        # Initialize arrays for results
        n_points = len(self.ind_resp)
        n_freq = len(fftfre)
        ufft = zeros((n_points, n_freq), dtype=complex)

        # Compute deflection FFTs separately for each point
        for i, ind in enumerate(self.ind_resp):
            defl = self.results.deflection[ind, : self.results.discr.nt]
            _, ufft[i] = self.fast_fourier_tr(defl, self.results.discr.dt)

        # Calculate quantities for all points
        rez = ufft / ffft  # Receptance
        mob = 1j * fftfre * 2 * pi * rez  # Mobility
        accel = -((fftfre * 2 * pi) ** 2) * rez  # Accelerance

        # Frequency range
        mask = (fftfre > self.f_min) & (fftfre <= self.f_max)

        # Store results as attributes
        self.freq = fftfre[mask]
        self.rez = squeeze(rez[:, mask])
        self.mob = squeeze(mob[:, mask])
        self.accel = squeeze(accel[:, mask])


@dataclass(kw_only=True)
class TDR(RollandPP):
    r"""Postprocessing class for TDR (Track-Decay-Rate).

    This class calculates and stores the Track-Decay-Rate (TDR) based on :cite:`EN15461:2008`.

    Attributes
    ----------
    results : Deflection
        Instance of the Deflection class containing the results.
    tdr : numpy.ndarray
        Track-Decay-Rate vector :math:`[dB/m]`.
    ind_tdr : list[int]
        Indices of the TDR points.
    x_tdr : numpy.ndarray
        Distances of the TDR points from the excitation point :math:`[m]`.
    filter : str | None
        Filter type (default is '1/3 Octave').
    freq : numpy.ndarray
        Frequency vector :math:`[Hz]`.
    """

    tdr: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    filter: str | None = None
    freq: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})

    def __post_init__(self):
        """Initialize TDR and calculate TDR values."""
        self.find_tdr_points()
        self.calculate_tdr()

    def find_tdr_points(self):
        """Find the corresponding measurement points depending on track type."""
        if isinstance(self.results.track, ArrangedSlabSingleRailTrack | ArrangedBallastedSingleRailTrack):
            # TDR for non-uniform mounting positions
            #   Identification of TDR positions

            # Determination of mounting positions
            x_mp = array(list(self.results.track.mount_prop.keys()))    # Position
            ind_mp = (x_mp / self.results.discr.dx).astype(int)         # Index
            # Left sleeper Index
            idx_s = int(where(ind_mp < self.results.ind_excit)[0][-1])
            # Calculate distance from excitation point
            x_s = x_mp[idx_s:] - x_mp[idx_s]  # Sleeper distances from excitation point.
            x_sc = convolve(x_s, ones(2) / 2, mode='valid')  # Sleeper centers from excitation point.

            def tdr_points_betw1(idx):
                """Calculate of theoretical measurement points (1st part)."""
                return ((x_s[idx + 1] - x_sc[idx]) / 2) + x_sc[idx]

            def tdr_points_betw2(dx):
                """Calculate of theoretical measurement points (2nd part)."""
                return ((x_sc[dx] - x_s[dx]) / 2) + x_s[dx]

            # Theoretical measurement points
            self.x_tdr = array([x_sc[0], tdr_points_betw1(0), x_s[1], tdr_points_betw2(1), x_sc[1], tdr_points_betw1(1),
                         x_s[2], tdr_points_betw2(2), x_sc[2], tdr_points_betw1(2), x_s[3], x_sc[3], x_s[4], x_sc[4],
                         x_sc[5], x_sc[6], x_sc[7], x_sc[8], x_sc[10], x_sc[12], x_sc[16], x_sc[20], x_sc[24], x_sc[30],
                         x_sc[36], x_sc[42], x_sc[48], x_sc[54], x_sc[66]]) - x_sc[0]

            # Determination of measurement position indices
            ind_tdr = rint(round(self.x_tdr, 5) / self.results.discr.dx) + self.results.ind_excit
            self.ind_tdr = list(ind_tdr.astype(int))

        else:
            # TDR for continuous slab and ballasted tracks
            # Identification of TDR positions
            ind_excit = self.results.ind_excit              # Start index.
            l_s = 0.6                                       # Theoretical Sleeper distance.
            x_tdr = array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5.5, 6.5, 7.5, 8.5,
                         10.5, 12.5, 16.5, 20.5, 24.5, 30.5, 36.5, 42.5, 48.5, 54.5, 66.5]) * l_s

            self.x_tdr = x_tdr - l_s / 2
            ind_tdr = rint(self.x_tdr / self.results.discr.dx) + ind_excit
            self.ind_tdr = list(ind_tdr.astype(int))

    def calculate_tdr(self):
        """Calculate the Track-Decay-Rate (TDR) based on the results."""
        # Calculation of mobilities
        resp = Response(results=self.results, ind_resp=self.ind_tdr)
        mob = resp.mob

        # Calculation of TDR (according to DIN)
        sum_tdr = abs(mob[1, 1:]) ** 2 / abs(mob[0, 1:]) ** 2 * (self.x_tdr[1])
        for n in range(2, len(self.ind_tdr)):
            sum_tdr = sum_tdr + abs(mob[n, 1:]) ** 2 / abs(mob[0, 1:]) ** 2 * (self.x_tdr[n])

        self.tdr = 4.343 / sum_tdr
        self.freq = resp.freq[1:]
