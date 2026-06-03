"""Defines discretization classes for FDM simulation.

.. autosummary::
    :toctree: discretization

    Discretization
    DiscretizationEBBVertic
    DiscretizationEBBVerticConst
    DiscretizationEBBVerticTimeDepend
"""

import abc
import warnings

from numpy import linspace, ones, zeros
from scipy.sparse import SparseEfficiencyWarning, csc_matrix, diags, eye
from traitlets import Float, Instance

from .abstract_traits import ABCHasTraits
from .boundary import PMLRailDampVertic
from .track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
    Track,
)


class Discretization(ABCHasTraits):
    r"""Abstract base class for discretization classes."""

    @abc.abstractmethod
    def validate_discretization(self):
        """Validate the discretization."""


class DiscretizationEBBVertic(Discretization):
    r"""Abstract base class for FDM discretization according to :cite:t:`stampka2022a`.

    Discretizes the differential equation and can be applied either with constant or time-dependent
    parameters, which is the case, for example, with a moving sound source.

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

    track = Instance(Track)
    bound = Instance(PMLRailDampVertic)
    dt = Float(default_value=2e-5)
    req_simt = Float(default_value=0.4)
    bx = Float(default_value=1.0)

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

    @abc.abstractmethod
    def validate_discretization_stampka(self):
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
            self.vec_sp += self.track.pad.sp[0]
            self.vec_dp += self.track.pad.dp[0]
            self.vec_ms += self.track.slab.ms

        elif isinstance(self.track, SimplePeriodicSlabSingleRailTrack | ArrangedSlabSingleRailTrack):
            self.build_discrete_slab_track()

        elif isinstance(self.track, ContBallastedSingleRailTrack):
            # Properties are assigned to each grid point
            self.vec_sp += self.track.pad.sp[0]
            self.vec_dp += self.track.pad.dp[0]
            self.vec_ms += self.track.slab.ms
            self.vec_sb += self.track.ballast.sb[0]
            self.vec_db += self.track.ballast.db[0]

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
            self.vec_sp[x_ind] = self.track.mount_prop[i][0].sp[0] / self.dx
            self.vec_dp[x_ind] = self.track.mount_prop[i][0].dp[0] / self.dx
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
            self.vec_sp[x_ind] = self.track.mount_prop[i][0].sp[0] / self.dx
            self.vec_dp[x_ind] = self.track.mount_prop[i][0].dp[0] / self.dx
            self.vec_ms[x_ind] = self.track.mount_prop[i][1].ms / self.dx
            self.vec_sb[x_ind] = self.track.mount_prop[i][2].sb[0] / self.dx
            self.vec_db[x_ind] = self.track.mount_prop[i][2].db[0] / self.dx


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
