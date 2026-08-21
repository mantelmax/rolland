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
"""

# ruff: noqa: N806

import warnings
from dataclasses import dataclass, field

from numpy import (  # noqa: A004
    empty,
    exp,
    linspace,
    ndarray,
    ones,
    zeros,
)
from scipy.sparse import SparseEfficiencyWarning, bmat, diags, eye
from scipy.sparse.linalg import splu

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

    Gaussian impulse according to :cite:p:`stampka_time-domain_2022`. This
    excitation type is used for non-moving sources.

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
    req_simt: float = 0.7
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
        e_z = self.track.E[1]

        if isinstance(self.track, ContSlabSingleRailTrack):
            # Properties are assigned to each grid point
            self.vec_sp += self.track.pad.sp_z
            self.vec_dp += self.track.pad.dp_z
            self.vec_ms += self.track.slab.ms * e_z

        elif isinstance(self.track, SimplePeriodicSlabSingleRailTrack | ArrangedSlabSingleRailTrack):
            self.build_discrete_slab_track(e_z)

        elif isinstance(self.track, ContBallastedSingleRailTrack):
            # Properties are assigned to each grid point
            self.vec_sp += self.track.pad.sp_z
            self.vec_dp += self.track.pad.dp_z
            self.vec_ms += self.track.slab.ms * e_z
            self.vec_sb += self.track.ballast.sb_z * e_z
            self.vec_db += self.track.ballast.db_z * e_z

        elif isinstance(self.track, SimplePeriodicBallastedSingleRailTrack | ArrangedBallastedSingleRailTrack):
            self.build_discrete_ballasted_track(e_z)

        else:
            msg = "Track type not recognized!"
            raise ValueError(msg)

    def build_discrete_slab_track(self, e_z):
        """
        Build discrete slab track.

        Properties are assigned to the corresponding mounting positions.

        Parameters
        ----------
        e_z : float
            Equivalent track foundation stiffness parameter.

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
            self.vec_ms[x_ind] = (self.track.slab.ms * e_z) / self.dx

    def build_discrete_ballasted_track(self, e_z):
        """Build discrete ballasted track.

        Properties are assigned to the corresponding mounting positions.

        Parameters
        ----------
        e_z : float
            Equivalent track foundation stiffness parameter.

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
            self.vec_ms[x_ind] = (self.track.mount_prop[i][1].ms * e_z) / self.dx
            self.vec_sb[x_ind] = (self.track.mount_prop[i][2].sb_z * e_z) / self.dx
            self.vec_db[x_ind] = (self.track.mount_prop[i][2].db_z * e_z) / self.dx

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


