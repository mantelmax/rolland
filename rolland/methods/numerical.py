"""Contains numerical methods as developed by K.Stampka and E.Sarradj.

Katja Stampka and Ennes Sarradj.
A Time-Domain Finite-Difference Method for Bending Waves
on Infinite Beams on an Elastic Foundation.
Acoustics, January 2022. Num Pages: 18. doi:10.3390/acoustics4040052.
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numpy import (
    empty,
    exp,
    linspace,
    ones,
    zeros,
)
from scipy.sparse import SparseEfficiencyWarning, csc_matrix, diags, eye
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

#---domainsetup.py---
class Discretization(ABC):
    r"""Abstract base class for discretization classes."""

    @abstractmethod
    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""


@dataclass(kw_only=True)
class DiscretizationEBBVertic(Discretization):
    r"""Abstract base class for FDM discretization according to :cite:t:`stampka2022a`.

    Discretizes the differential equation and can be applied either with constant or time-dependent
    parameters, which is the case, for example, with a moving sound source.

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
    bound : PMLRailDampVertic
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
    """

    track: Track
    bound: PMLRailDampVertic
    dt: float = 2e-5
    req_simt: float = 0.4
    bx: float = 1.0

    def calc_grid(self):
        """Calculate grid parameters."""
        self.nt = int(self.req_simt / self.dt)
        self.sim_t = self.nt * self.dt
        dx_min = (self.bx * ((self.track.rail.E * self.track.rail.Iyr) /
                             (6 * self.track.rail.mr)) ** (1 / 4) * self.dt ** (1 / 2))
        self.dx = 0.6 / (0.6 // dx_min)
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


    def build_matrix(self, vec_dr, vec_sp, vec_dp, vec_ms, vec_sb, vec_db):
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
        D_diagonals = [ones(self.nx - 2),  # noqa: N806
                       (-4) * ones(self.nx - 1),
                       6 * ones(self.nx),
                       (-4) * ones(self.nx - 1),
                       ones(self.nx - 2)]

        D = diags(D_diagonals, [-2, -1, 0, 1, 2])  # noqa: N806
        Eye = eye(self.nx)  # noqa: N806

        A11_1_diagonals = self.dt / self.track.rail.mr * (vec_dr + vec_dp)  # noqa: N806
        A11_1_diagonals += self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp  # noqa: N806
        A11_1 = diags([A11_1_diagonals], [0])  # noqa: N806
        A11 = (r * D + Eye + A11_1).tocsc()  # noqa: N806

        B11_1_diagonals = self.dt / self.track.rail.mr * (vec_dr + vec_dp)  # noqa: N806
        B11_1 = diags([B11_1_diagonals], [0])  # noqa: N806
        B11 = (2 * Eye + B11_1).tocsc()  # noqa: N806

        C11_1_diagonals = self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp  # noqa: N806
        C11_1 = diags([C11_1_diagonals], [0])  # noqa: N806
        C11 = (-(Eye + C11_1 + r * D)).tocsc()  # noqa: N806

        A12_diagonals = -self.dt / self.track.rail.mr * vec_dp  # noqa: N806
        A12_diagonals += -self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp  # noqa: N806
        A12 = diags([A12_diagonals], [0]).tocsc()  # noqa: N806

        A21_diagonals = -self.dt * vec_dp / vec_ms  # noqa: N806
        A21_diagonals += -self.dt ** 2 / (2 * vec_ms) * vec_sp  # noqa: N806
        A21 = diags([A21_diagonals], [0]).tocsc()  # noqa: N806

        A22_1_diagonals = self.dt * ((vec_dp + vec_db) / vec_ms)  # noqa: N806
        A22_1_diagonals += self.dt ** 2 / (2 * vec_ms) * (vec_sp + vec_sb)  # noqa: N806
        A22_1 = diags([A22_1_diagonals], [0])  # noqa: N806
        A22 = (Eye + A22_1).tocsc()  # noqa: N806

        B12_diagonals = -self.dt / self.track.rail.mr * vec_dp  # noqa: N806
        B12 = diags([B12_diagonals], [0]).tocsc()  # noqa: N806

        B21_diagonals = -self.dt * vec_dp / vec_ms  # noqa: N806
        B21 = diags([B21_diagonals], [0]).tocsc()  # noqa: N806

        B22_1_diagonals = self.dt * (vec_db + vec_dp) / vec_ms  # noqa: N806
        B22_1 = diags([B22_1_diagonals], [0])  # noqa: N806
        B22 = (2 * Eye + B22_1).tocsc()  # noqa: N806

        C12_diagonals = self.dt ** 2 / (2 * self.track.rail.mr) * vec_sp  # noqa: N806
        C12 = diags([C12_diagonals], [0]).tocsc()  # noqa: N806

        C21_diagonals = self.dt ** 2 / (2 * vec_ms) * vec_sp  # noqa: N806
        C21 = diags([C21_diagonals], [0]).tocsc()  # noqa: N806

        C22_1_diagonals = self.dt ** 2 * (vec_sp + vec_sb) / (2 * vec_ms)  # noqa: N806
        C22_1 = diags([C22_1_diagonals], [0])  # noqa: N806
        C22 = (-(Eye + C22_1)).tocsc()  # noqa: N806

        self.A = csc_matrix((2 * self.nx, 2 * self.nx))  # noqa: N806
        self.A[0:self.nx, 0:self.nx] = A11
        self.A[0:self.nx, self.nx:2 * self.nx] = A12
        self.A[self.nx:2 * self.nx, 0:self.nx] = A21
        self.A[self.nx:2 * self.nx, self.nx:2 * self.nx] = A22

        self.B = csc_matrix((2 * self.nx, 2 * self.nx))  # noqa: N806
        self.B[0:self.nx, 0:self.nx] = B11
        self.B[0:self.nx, self.nx:2 * self.nx] = B12
        self.B[self.nx:2 * self.nx, 0:self.nx] = B21
        self.B[self.nx:2 * self.nx, self.nx:2 * self.nx] = B22

        self.C = csc_matrix((2 * self.nx, 2 * self.nx))  # noqa: N806
        self.C[0:self.nx, 0:self.nx] = C11
        self.C[0:self.nx, self.nx:2 * self.nx] = C12
        self.C[self.nx:2 * self.nx, 0:self.nx] = C21
        self.C[self.nx:2 * self.nx, self.nx:2 * self.nx] = C22

    @abstractmethod
    def _abstract(self) -> None:
        """Validate the discretization according to Stampka."""


class DiscretizationEBBVerticConst(DiscretizationEBBVertic):
    r"""Discretization with non-time-dependent parameters according to :cite:t:`stampka2022a`.

    The parameters are constant over time. Only applicable for non-moving sound sources
    and linear superstructure properties.

    Attributes
    ----------
    track : Track
        Track instance.
    dt : float
        Step size in time :math:`[s]`.
    req_simt : float
        Requested simulation time :math:`[s]`.
    bx : float
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
    bound : PMLRailDampVertic
        Boundary instance.
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

    def validate_discretization(self):
        """Validate the discretization."""

    def validate_discretization_stampka(self):
        """Validate the discretization according to Stampka."""

    def __init__(self, *args, **kwargs):
        """Calculate superstructure property vectors."""
        super().__init__(*args, **kwargs)
        self.calc_grid()
        self.calc_bound()
        self.initialize_vectors()
        self.add_boundary_conditions()
        self.build_superstructure_vectors()
        self.build_matrix(self.vec_dr, self.vec_sp, self.vec_dp, self.vec_ms, self.vec_sb, self.vec_db)

    # take signature from parent class for better documentation
    __init__.__signature__ = inspect.signature(DiscretizationEBBVertic.__init__)

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

    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""

class DiscretizationEBBVerticTimeDepend(DiscretizationEBBVerticConst):
    """
    Discretization with time-dependent parameters based on :cite:t:`stampka2022a`.

    This class extends :class:`DiscretizationFDMStampkaConst` to handle cases where the parameters
    vary over time, such as with a moving sound source or non-linear superstructure properties.
    This approach is a extended version of the discretization described in :cite:t:`stampka2022a`.

    .. note:: This class is not implemented yet.

    """

    def validate_discretization(self):
        """Validate the discretization."""

    def validate_discretization_stampka(self):
        """Validate the discretization according to Stampka."""

#---excitation.py---
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

#---deflection.py---
@dataclass(kw_only=True)
class Deflection(ABC):
    r"""Abstract base class for deflection classes.

    Attributes
    ----------
    excit : Excitation
        Excitation instance.
    discr : Discretization
        Discretization instance.
    """

    discr: Discretization
    excit: Excitation
    track: Track = field(init=False)

    def __post_init__(self, *args, **kwargs):
        """post_init method to set track attribute after initialization."""
        self.track = self.discr.track

    @abstractmethod
    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""


class DeflectionEBBVertic(Deflection):
    r"""Calculate deflection according to :cite:t:`stampka2022a`.

    Attributes
    ----------
    track : Track
        Track instance.
    excit : Excitation
        Excitation instance.
    discr : Discretization
        Discretization instance.
    deflection : numpy.ndarray
        Deflection array :math:`[m]`.
    rotation : numpy.ndarray
        Rotation array :math:`[rad]`.
    ind_excit : int
        Index of excitation point :math:`[-]`.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the DeflectionFDMStampka class.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Attributes
        ----------
        deflection : numpy.ndarray
            Array of calculated deflections with shape (nx, nt + 1).
        rotation : numpy.ndarray
            Array of calculated rotations with shape (nx, nt + 1).
        """
        super().__init__(*args, **kwargs)
        # Initialize starting values
        self.calc_force()
        state = self.initialize_start_values()
        # Calculate deflection
        self.state = self.calc_deflection(state)
        self.deflection = self.state[:self.discr.nx, :].T
        self.rotation = self.state[self.discr.nx:2 * self.discr.nx, :].T

    __init__.__signature__ = inspect.signature(Deflection.__init__)

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

        if isinstance(ind_excit, list):
            for idx in ind_excit:
                f[idx] = self.force[t]
        else:
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

    def _abstract(self) -> None:
        pass
