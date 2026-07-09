"""Defines discretization classes for FDM simulation.

.. autosummary::
    :toctree: discretization

    Discretization
    DiscretizationEBBVertic
    DiscretizationEBBVerticConst
    DiscretizationEBBVerticTimeDepend
"""

import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

from devito import Border, Constant, Eq, Function, Grid, TimeFunction, solve
from numpy import float64, linspace, ones, sqrt, zeros
from scipy.sparse import SparseEfficiencyWarning, csc_matrix, diags, eye

from .boundary import DevitoPMLDamp, PMLRailDampVertic
from .excitation import Excitation
from .track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    DiscrSlabSingleRailTrack,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
    Track,
)


class Discretization(ABC):
    r"""Abstract base class for discretization classes."""

    @abstractmethod
    def validate_discretization(self):
        """Validate the discretization."""


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

@dataclass(kw_only=True)
class DiscretizeDEVITO(Discretization):
    r"""Descritize DEVITO.

    Attributes
    ----------
    track : Track
        Track instance.
    dt : float, default=0.5e-5
        Step size in time :math:`[s]`.
    req_simt : float, default=0.1
        Requested simulation time :math:`[s]`.
    nt : int
        Number of time steps :math:`[-]`.
    sim_t : float
        Actual simulation time :math:`[s]`.
    nx : int
        Number of spatial steps :math:`[-]`.
    dx : float, default=0.05
        Spatial step size :math:`[m]`.
    bound : PMLRailDampVertic
        Boundary instance.
    grid : devito.Grid
        Devito grid instance.
    z_f : float, default=0.0
        Excitation excentricity in z-direction :math:`[m]`.
    y_f : float, default=0.0
        Excitation excentricity in y-direction :math:`[m]`.
    store : str, default="point"
        Decides whether to store deflection only at the excitation position ('point') or not.
    equi_sm : bool, default=True
        If True, equivalent sleeper model is applied
    y_sc : float, default=0.7175
        Distance from excitation point to sleeper center :math:`[m]`.
    """

    track: Track
    bound: DevitoPMLDamp
    dt: float = 0.5e-5
    req_simt: float = 0.1
    nt: int
    dx: float = 0.05
    grid: Grid
    z_f: float = 0.0
    y_f: float = 0.0
    store: str = "point"
    exc: Excitation
    equi_sm: bool = True
    y_sc: float = 0.7175

    def __post_init__(self, *args, **kwargs):
        """Post-initialization method to build the Devito grid."""
        self.build_grid()

    def build_grid(self):
        """Build the Devito computational grid."""
        self.nt = int(self.req_simt / self.dt)
        self.nx = int(self.track.l_track / self.dx)

        self.grid = Grid(
            shape=(self.nx,),
            extent=(self.track.l_track,),
            dtype=float64,
            origin=(0.0,),
        )

        self.bd_grid = Grid(
            shape=(self.nx,),
            extent=(self.track.l_track,),
            dtype=float64,
            origin=(0.0,),
        )

        # Define Boundary Domain
        _nx_bound = int(self.bound.l_bound / self.dx)
        (x,) = self.bd_grid.dimensions
        self.bound_dom = Border(grid=self.bd_grid, border=_nx_bound, dims=x)


    def build_operator(self):
        """Build the Devito operator for the simulation."""
        sp_z = Function(name="sp_z", grid=self.grid, dtype=float64)
        sp_y = Function(name="sp_y", grid=self.grid, dtype=float64)
        sp_x = Function(name="sp_x", grid=self.grid, dtype=float64)
        sp_w = Function(name="sp_w", grid=self.grid, dtype=float64)
        sp_zr = Function(name="sp_zr", grid=self.grid, dtype=float64)
        sp_yr = Function(name="sp_yr", grid=self.grid, dtype=float64)
        sp_xr = Function(name="sp_xr", grid=self.grid, dtype=float64)
        sb_x = Function(name="sb_x", grid=self.grid, dtype=float64)
        sb_z = Function(name="sb_z", grid=self.grid, dtype=float64)
        sb_y = Function(name="sb_y", grid=self.grid, dtype=float64)
        sb_xr = Function(name="sb_xr", grid=self.grid, dtype=float64)
        sb_zr = Function(name="sb_zr", grid=self.grid, dtype=float64)
        sb_yr = Function(name="sb_yr", grid=self.grid, dtype=float64)
        dp_z = Function(name="dp_z", grid=self.grid, dtype=float64)
        dp_y = Function(name="dp_y", grid=self.grid, dtype=float64)
        dp_x = Function(name="dp_x", grid=self.grid, dtype=float64)
        dp_xr = Function(name="dp_xr", grid=self.grid, dtype=float64)
        db_x = Function(name="db_x", grid=self.grid, dtype=float64)
        db_z = Function(name="db_z", grid=self.grid, dtype=float64)
        db_y = Function(name="db_y", grid=self.grid, dtype=float64)
        db_xr = Function(name="db_xr", grid=self.grid, dtype=float64)
        db_zr = Function(name="db_zr", grid=self.grid, dtype=float64)
        db_yr = Function(name="db_yr", grid=self.grid, dtype=float64)
        ms = Function(name="ms", grid=self.grid, dtype=float64)
        moinnert_s_x = Function(name="Is_x", grid=self.grid, dtype=float64)
        moinnert_s_y = Function(name="Is_y", grid=self.grid, dtype=float64)
        moinnert_s_z = Function(name="Is_z", grid=self.grid, dtype=float64)


        track = self.track
        if isinstance(track, (ContSlabSingleRailTrack, ContBallastedSingleRailTrack)):
            mount_pos_interp = 1
            seclay = track.slab

        else:
            mount_pos = list(track.mount_prop.keys())
            mount_pos_interp = track.pad.interpol_pad_width(
                linspace(0, track.l_track, self.nx), self.dx, mount_pos,)

            seclay = track.slab if isinstance(track, SimplePeriodicSlabSingleRailTrack) else track.sleeper


        if isinstance(track, (ContSlabSingleRailTrack, DiscrSlabSingleRailTrack)):
            K0r, K1r, K2r, Mr = build_rail_matrices(track.rail, 'viscous')  # noqa: N806
            Tf, _, _ = build_transfm_matrices(self.z_f, self.y_f, 0, 0, 0)
            Kp, _ = build_pad_ballast_stiff_matrices(track, self.z_f, 'viscous')
            K_fnd = build_fnd_stiff_matrix(Kp, Tf)
            cof = calc_cut_on_frequ(K0r, K_fnd, Mr)
            Dp, _ = build_pad_ballast_damp_matrices(track, cof)

            sb_x.data[:] = 0
            sb_z.data[:] = 0
            sb_y.data[:] = 0
            sb_xr.data[:] = 0
            sb_zr.data[:] = 0
            sb_yr.data[:] = 0
            db_x.data[:] = 0
            db_z.data[:] = 0
            db_y.data[:] = 0
            db_xr.data[:] = 0
            db_yr.data[:] = 0
            db_zr.data[:] = 0

            # Assigns the sleeper mass to a very large number
            ms.data[:] = 1e20
            moinnert_s_x.data[:] = 1e20
            moinnert_s_y.data[:] = 1e20
            moinnert_s_z.data[:] = 1e20
            rho_s = 1e20
            z_st = 0
            z_sb = 0
            Ex = Constant(name="Ex", value=1)
            Ez = Constant(name="Ez", value=1)


        else:
            K0r, K1r, K2r, Mr = build_rail_matrices(track.rail, 'viscous')
            Tf, Tst, Tsb = build_transfm_matrices(self.z_f, self.y_f, seclay.z_st, seclay.z_sb, 0)
            E = build_equ_sleeper_matrix(track, self.y_sc, self.equi_sm)
            Kp, Kb = build_pad_ballast_stiff_matrices(track, self.z_f, 'viscous', E)
            Ms = build_sleep_mass_matrix(track, E)
            K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb)
            cof = calc_cut_on_frequ(K0r, K_fnd, Mr, Ms)
            Dp, Db = build_pad_ballast_damp_matrices(track, cof, E)
            # D_fnd = build_fnd_damp_matrix(Dp, Tf, Db, Tst, Tsb)

            ballast = track.ballast
            sb_x.data[:] = ballast.sb_x * mount_pos_interp
            sb_z.data[:] = ballast.sb_z * mount_pos_interp
            sb_y.data[:] = ballast.sb_y * mount_pos_interp
            sb_xr.data[:] = ballast.sb_xr * mount_pos_interp
            sb_zr.data[:] = ballast.sb_zr * mount_pos_interp
            sb_yr.data[:] = ballast.sb_yr * mount_pos_interp
            db_x.data[:] = ballast.db_x * mount_pos_interp
            db_z.data[:] = ballast.db_z * mount_pos_interp
            db_y.data[:] = ballast.db_y * mount_pos_interp
            db_xr.data[:] = ballast.db_xr * mount_pos_interp
            db_zr.data[:] = ballast.db_zr * mount_pos_interp
            db_yr.data[:] = ballast.db_yr * mount_pos_interp

            ms.data[:] = (seclay.ms * mount_pos_interp) + 1e-20
            moinnert_s_x.data[:] = seclay.is_x * mount_pos_interp + 1e-20
            moinnert_s_y.data[:] = seclay.is_y * mount_pos_interp + 1e-20
            moinnert_s_z.data[:] = seclay.is_z * mount_pos_interp + 1e-20
            rho_s = seclay.rhos
            z_st = seclay.z_st
            z_sb = seclay.z_sb
            Ex = Constant(name="Ex", value=E[0])
            Ez = Constant(name="Ez", value=E[1])

        sp_z.data[:] = track.pad.sp_z * mount_pos_interp
        sp_y.data[:] = track.pad.sp_y * mount_pos_interp
        sp_x.data[:] = track.pad.sp_x * mount_pos_interp
        sp_w.data[:] = track.pad.sp_w * mount_pos_interp
        sp_zr.data[:] = track.pad.sp_zr * mount_pos_interp
        sp_yr.data[:] = track.pad.sp_yr * mount_pos_interp
        sp_xr.data[:] = track.pad.sp_xr * mount_pos_interp
        dp_z.data[:] = track.pad.dp_z * mount_pos_interp
        dp_y.data[:] = track.pad.dp_y * mount_pos_interp
        dp_x.data[:] = track.pad.dp_x * mount_pos_interp
        dp_xr.data[:] = track.pad.dp_xr * mount_pos_interp


        # Define variables
        if self.store == "point":
            save = None
        elif self.store == "full":
            save = self.nt

        so1 = 6
        u_x = TimeFunction(name="u_x", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)
        u_y = TimeFunction(name="u_y",grid=self.grid,time_order=2,space_order=so1, dtype=float64, save=save)
        u_z = TimeFunction(name="u_z",grid=self.grid,time_order=2,space_order=so1, dtype=float64, save=save)
        phi_y = TimeFunction(name="phi_y", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)
        phi_z = TimeFunction(name="phi_z", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)
        phi_x = TimeFunction(name="phi_x",grid=self.grid,time_order=2,space_order=so1, dtype=float64, save=save)
        u_w = TimeFunction(name="u_w", grid=self.grid, time_order=2, space_order=so1, dtype=float64, save=None)

        u_sx = TimeFunction(name="u_sx", grid=self.grid, time_order=2, dtype=float64, save=None)
        u_sz = TimeFunction(name="u_sz", grid=self.grid, time_order=2, dtype=float64, save=None)
        u_sy = TimeFunction(name="u_sy", grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sx = TimeFunction(name="phi_sx", grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sz = TimeFunction(name="phi_sz", grid=self.grid, time_order=2, dtype=float64, save=None)
        phi_sy = TimeFunction(name="phi_sy", grid=self.grid, time_order=2, dtype=float64, save=None)

        ##### Damping
        sigm = Function(name="sigma", grid=self.grid)
        alph = Function(name="alpha", grid=self.grid)
        self.bound.calc_damping_profile(dx=self.dx, nx=self.nx)
        sigm.data[:] = self.bound.sigma
        alph.data[:] = self.bound.alpha

        # Auxiliary variables
        so2 = 3
        to2 = 2
        psi_ux = TimeFunction(name="psi_ux", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_ux = TimeFunction(name="theta_ux", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_x_dx_pml = u_x.dx - psi_ux * sigm
        u_x_dx2_pml = (u_x.dx2 - (sigm.dx * psi_ux + sigm * psi_ux.dx) - sigm * theta_ux)

        psi_uz = TimeFunction(name="psi_uz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_uz = TimeFunction( name="theta_uz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_z_dx_pml = u_z.dx - psi_uz * sigm
        u_z_dx2_pml = (u_z.dx2 - (sigm.dx * psi_uz + sigm * psi_uz.dx) - sigm * theta_uz)

        psi_phiy = TimeFunction(name="psi_phiy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_phiy = TimeFunction(name="theta_phiy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        phi_y_dx_pml = phi_y.dx - psi_phiy * sigm
        phi_y_dx2_pml = (phi_y.dx2 - (sigm.dx * psi_phiy + sigm * psi_phiy.dx) - sigm * theta_phiy)

        psi_uy = TimeFunction(name="psi_uy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_uy = TimeFunction(name="theta_uy", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_y_dx_pml = u_y.dx - psi_uy * sigm
        u_y_dx2_pml = (u_y.dx2 - (sigm.dx * psi_uy + sigm * psi_uy.dx)- sigm * theta_uy)

        psi_phiz = TimeFunction(name="psi_phiz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_phiz = TimeFunction(name="theta_phiz", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        phi_z_dx_pml = phi_z.dx - psi_phiz * sigm
        phi_z_dx2_pml = (phi_z.dx2 - (sigm.dx * psi_phiz + sigm * psi_phiz.dx) - sigm * theta_phiz)

        psi_phix = TimeFunction(name="psi_phix", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_phix = TimeFunction(name="theta_phix", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        phi_x_dx_pml = phi_x.dx - psi_phix * sigm
        phi_x_dx2_pml = (phi_x.dx2 - (sigm.dx * psi_phix + sigm * psi_phix.dx) - sigm * theta_phix)

        psi_uw = TimeFunction(name="psi_uw", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        theta_uw = TimeFunction(name="theta_uw", grid=self.grid, time_order=to2, space_order=so2, dtype=float64, save=None)
        u_w_dx_pml = u_w.dx - psi_uw * sigm
        u_w_dx2_pml = (u_w.dx2 - (sigm.dx * psi_uw + sigm * psi_uw.dx) - sigm * theta_uw)


        # Auxiliary equations for ADE-PML
        ade_psi_ux = Eq(psi_ux.dt, u_x_dx_pml - psi_ux * alph)
        ade_theta_ux = Eq(theta_ux.dt, u_x_dx2_pml - theta_ux * alph)

        ade_psi_uz = Eq(psi_uz.dt, u_z_dx_pml - psi_uz * alph)
        ade_theta_uz = Eq(theta_uz.dt, u_z_dx2_pml - theta_uz * alph)

        ade_psi_phiy = Eq(psi_phiy.dt, phi_y_dx_pml - psi_phiy * alph)
        ade_theta_phiy = Eq(theta_phiy.dt, phi_y_dx2_pml - theta_phiy * alph)

        ade_psi_uy = Eq(psi_uy.dt, u_y_dx_pml - psi_uy * alph)
        ade_theta_uy = Eq(theta_uy.dt, u_y_dx2_pml - theta_uy * alph)

        ade_psi_phiz = Eq(psi_phiz.dt, phi_z_dx_pml - psi_phiz * alph)
        ade_theta_phiz = Eq(theta_phiz.dt, phi_z_dx2_pml - theta_phiz * alph)

        ade_psi_phix = Eq(psi_phix.dt, phi_x_dx_pml - psi_phix * alph)
        ade_theta_phix = Eq(theta_phix.dt, phi_x_dx2_pml - theta_phix * alph)

        ade_psi_uw = Eq(psi_uw.dt, u_w_dx_pml - psi_uw * alph)
        ade_theta_uw = Eq(theta_uw.dt, u_w_dx2_pml - theta_uw * alph)

        # Define Timoshenko beam equation

        rail = self.track.rail
        G = Constant(name="G", value=rail.G)
        E = Constant(name="E", value=rail.E)
        A = Constant(name="A", value=rail.Ar)
        Iy = Constant(name="I_y", value=rail.Iyr)
        Iz = Constant(name="I_z", value=rail.Izr)
        Iwz = Constant(name="I_wz", value=rail.Iwz)
        Iwy = Constant(name="I_wy", value=rail.Iwy)
        Iyz = Constant(name="I_yz", value=rail.Iyz)
        rho = Constant(name="rho", value=rail.rho)
        mr = Constant(name="m_r", value=rail.mr)
        kap_y = Constant(name="kappa_y", value=rail.kapy)
        kap_z = Constant(name="kappa_z", value=rail.kapz)
        e_y = Constant(name="e_y", value=rail.ey)
        e_z = Constant(name="e_z", value=rail.ez)
        chi_f = Constant(name="chi_f", value=0)
        Iw = Constant(name="I_w", value=rail.Iw)
        J = Constant(name="J", value=rail.J)
        Jt = Constant(name="J_t", value=rail.J_t)
        Ip = Constant(name="I_p", value=rail.Ipr)
        z_f = Constant(name="z_f", value=self.z_f)
        y_f = Constant(name="y_f", value=self.y_f)
        z_st = Constant(name="z_st", value=z_st)
        z_sb = Constant(name="z_sb", value=z_sb)
        rho_s = Constant(name="rho_s", value=rho_s)

        # Longitudinal Wave
        longw = Eq(
            -A * E * u_x_dx2_pml
            + chi_f * dp_x * u_w.backward.dt
            + chi_f * sp_x * u_w
            + dp_x * y_f * phi_z.backward.dt
            - dp_x * z_f * phi_y.backward.dt
            - dp_x * u_sx.backward.dt
            + (dp_x + rail.dr) * u_x.dt
            + mr * u_x.dt2
            + sp_x * y_f * phi_z
            - sp_x * z_f * phi_y
            - sp_x * u_sx
            + sp_x * u_x,
        )

        # Timoshenko Beam Wave - Sum of Vertical Forces
        tbw_vert_trans = Eq(
            A * G * e_y * kap_z * phi_x_dx2_pml
            # + A * G * e_y * kap_z * u_w_dx_pml --> neglected
            + A * G * kap_z * phi_y_dx_pml
            - A * G * kap_z * u_z_dx2_pml
            + mr * u_z.dt2
            - sp_z * y_f * phi_x
            - sp_z * u_sz
            + sp_z * u_z
            - y_f * dp_z * phi_x.backward.dt
            + (-dp_z) * u_sz.backward.dt
            + (dp_z + rail.dr) * u_z.dt
        )

        # Timoshenko Beam Wave - Sum of Lateral Forces
        tbw_lat_trans = Eq(
            - A * G * e_z * kap_y * phi_x_dx2_pml
            # - A * G * e_z * kap_y * u_w_dx_pml --> neglected
            - A * G * kap_y * phi_z_dx_pml
            - A * G * kap_y * u_y_dx2_pml
            + mr * u_y.dt2
            + sp_y * z_f * phi_x
            - sp_y * z_st * phi_sx
            - sp_y * u_sy
            + sp_y * u_y
            + z_f * dp_y * phi_x.backward.dt
            + z_st * (-dp_y) * phi_sx.backward.dt
            + (-dp_y) * u_sy.backward.dt
            + (dp_y + rail.dr) * u_y.dt
        )

        # Torsional Beam Wave (Rotation about x-axis)
        torsw = Eq(
            -A * G * e_y * kap_z * phi_y_dx_pml
            + A * G * e_y * kap_z * u_z_dx2_pml
            - A * G * e_z * kap_y * phi_z_dx_pml
            - A * G * e_z * kap_y * u_y_dx2_pml
            - G * Jt * u_w_dx_pml
            - G * (J + Jt) * phi_x_dx2_pml
            + Ip * rho * phi_x.dt2
            - sp_y * z_f * u_sy
            + sp_y * z_f * u_y
            + sp_z * y_f * u_sz
            - sp_z * y_f * u_z
            + y_f * dp_z * u_sz.backward.dt
            - y_f * dp_z * u_z.dt
            - z_f * dp_y * u_sy.backward.dt
            + z_f * dp_y * u_y.dt
            + (-dp_xr - z_f * z_st * dp_y) * phi_sx.backward.dt
            + (-sp_xr - sp_y * z_f * z_st) * phi_sx
            + (dp_xr + y_f**2 * dp_z + z_f**2 * dp_y) * phi_x.dt
            + (sp_xr + sp_y * z_f**2 + sp_z * y_f**2) * phi_x
        )

        # Timoshenko Beam Wave - Sum of Lateral Moments
        tbw_lat_rot = Eq(
            + A * G * e_z * kap_y * phi_x_dx_pml
            + A * G * kap_y * u_y_dx_pml
            # - E * Iwz * u_w_dx2_pml --> neglected
            + E * Iyz * phi_y_dx2_pml
            - E * Iz * phi_z_dx2_pml
            # + Iwz * rho * u_w.backward.dt2 --> neglected
            - Iyz * rho * phi_y.backward.dt2
            # + chi_f * dp_x * y_f * u_w.backward.dt --> neglected
            + dp_x * y_f**2 * phi_z.dt
            - dp_x * y_f * z_f * phi_y.backward.dt
            - dp_x * y_f * u_sx.backward.dt
            + dp_x * y_f * u_x.dt
            + rho * (-Iwz + Iz) * phi_z.dt2
            - sp_x * y_f * z_f * phi_y
            - sp_x * y_f * u_sx
            + sp_x * y_f * u_x
            - sp_zr * phi_sz
            # (A * G * e_z * kap_y + chi_f * sp_x * y_f * u_w) * u_w --> neglected
            + (A * G * kap_y + sp_x * y_f**2 + sp_zr) * phi_z,
        )

        # Timoshenko Beam Wave - Sum of Vertical Moments
        tbw_vert_rot = Eq(
            A * G * e_y * kap_z * phi_x_dx_pml
            - A * G * kap_z * u_z_dx_pml
            # + E * Iwy * u_w_dx2_pml --> neglected
            - E * Iy * phi_y_dx2_pml
            + E * Iyz * phi_z_dx2_pml
            # - Iwy * rho * u_w.backward.dt2 --> neglected
            - Iyz * rho * phi_z.dt2
            # - chi_f * dp_x * z_f * u_w.backward.dt --> neglected
            - dp_x * y_f * z_f * phi_z.dt
            + dp_x * z_f**2 * phi_y.dt
            + dp_x * z_f * u_sx.backward.dt
            - dp_x * z_f * u_x.dt
            + rho * (Iwy + Iy) * phi_y.dt2
            - sp_x * y_f * z_f * phi_z
            + sp_x * z_f * u_sx
            - sp_x * z_f * u_x
            - sp_yr * phi_sy
            # + (A * G * e_y * kap_z - chi_f * sp_x * z_f) * u_w --> neglected
            + (A * G * kap_z + sp_x * z_f**2 + sp_yr) * phi_y,
        )

        # Warping Equation
        warp = Eq(
            - A * G * e_y * kap_z * u_z_dx_pml
            + A * G * e_z * kap_y * u_y_dx_pml
            - E * Iw * u_w_dx2_pml
            + E * Iwy * phi_y_dx2_pml
            - E * Iwz * phi_z_dx2_pml
            + G * Jt * phi_x_dx_pml
            + Iw * rho * u_w.dt2
            - Iwy * rho * phi_y.dt2
            + Iwz * rho * phi_z.dt2
            + chi_f**2 * dp_x * u_w.dt
            + chi_f * dp_x * y_f * phi_z.dt
            - chi_f * dp_x * z_f * phi_y.dt
            - chi_f * dp_x * u_sx.backward.dt
            + chi_f * dp_x * u_x.dt
            - chi_f * sp_x * u_sx
            + chi_f * sp_x * u_x
            + (A * G * e_y * kap_z - chi_f * sp_x * z_f) * phi_y
            + (A * G * e_z * kap_y + chi_f * sp_x * y_f) * phi_z
            + (G * Jt + chi_f**2 * sp_x + sp_w) * u_w
        )

        # Translational Sleeper Equation (x-direction)
        slep_trans_x = Eq(
            - chi_f * dp_x * u_w.dt
            - chi_f * sp_x * u_w
            - dp_x * y_f * phi_z.dt
            + dp_x * z_f * phi_y.dt
            - dp_x * u_x.dt
            - sp_x * y_f * phi_z
            + sp_x * z_f * phi_y
            - sp_x * u_x
            + (sp_x + sb_x * Ex) * u_sx
            + (db_x * Ex + dp_x) * u_sx.dt
            + ms * u_sx.dt2 * Ex
        )

        # Translational Sleeper Equation (z-direction)
        slep_trans_z = Eq(
            sp_z * y_f * phi_x
            - sp_z * u_z
            - y_f * (-dp_z) * phi_x.dt
            + (-dp_z) * u_z.dt
            + (sp_z + sb_z * Ez) * u_sz
            + (db_z * Ez + dp_z) * u_sz.dt
            + ms * u_sz.dt2 * Ez
        )

        # Translational Sleeper Equation (y-direction)
        slep_trans_y = Eq(
            ms * u_sy.dt2
            - sp_y * z_f * phi_x
            - sp_y * u_y
            + z_f * (-dp_y) * phi_x.dt
            + (-dp_y) * u_y.dt
            + (sb_y + sp_y) * u_sy
            + (db_y * z_sb + z_st * dp_y) * phi_sx.dt
            + (sb_y * z_sb + sp_y * z_st) * phi_sx
            + (db_y + dp_y) * u_sy.dt
        )

        # Rotational Sleeper Equation (x-axis)
        slep_rot_x = Eq(
            moinnert_s_x * rho_s * phi_sx.dt2
            - sp_y * z_st * u_y
            - z_st * dp_y * u_y.dt
            + (-dp_xr - z_f * z_st * dp_y) * phi_x.dt
            + (-sp_xr - sp_y * z_f * z_st) * phi_x
            + (db_y * z_sb + z_st * dp_y) * u_sy.dt
            + (sb_y * z_sb + sp_y * z_st) * u_sy
            + (db_xr + db_y * z_sb**2 + dp_xr + z_st**2 * dp_y) * phi_sx.dt
            + (sb_xr + sb_y * z_sb**2 + sp_xr + sp_y * z_st**2) * phi_sx
        )

        # Rotational Sleeper Equation (z-axis)
        slep_rot_z = Eq(
            moinnert_s_z * rho_s * phi_sz.dt2
            + db_zr * phi_sz.dt
            - sp_zr * phi_z
            + (sb_zr + sp_zr) * phi_sz,
        )

        # Rotational Sleeper Equation (y-axis)
        slep_rot_y = Eq(
            moinnert_s_y * rho_s * phi_sy.dt2
            + db_yr * phi_sy.dt
            - sp_yr * phi_y
            + (sb_yr + sp_yr) * phi_sy,
        )

        upd_longw = Eq(u_x.forward, solve(longw, u_x.forward))
        upd_tbw_vert_trans = Eq(u_z.forward, solve(tbw_vert_trans, u_z.forward))
        upd_tbw_lat_trans = Eq(u_y.forward, solve(tbw_lat_trans, u_y.forward))
        upd_torsw = Eq(phi_x.forward, solve(torsw, phi_x.forward))
        upd_tbw_vert_rot = Eq(phi_y.forward, solve(tbw_vert_rot, phi_y.forward))
        upd_tbw_lat_rot = Eq(phi_z.forward, solve(tbw_lat_rot, phi_z.forward))
        upd_warp = Eq(u_w.forward, solve(warp, u_w.forward))

        upd_psi_ux = Eq(psi_ux.forward, solve(ade_psi_ux, psi_ux.forward), subdomain=self.bound_dom)
        upd_theta_ux = Eq(theta_ux.forward, solve(ade_theta_ux, theta_ux.forward), subdomain=self.bound_dom)
        upd_psi_uz = Eq(psi_uz.forward, solve(ade_psi_uz, psi_uz.forward), subdomain=self.bound_dom)
        upd_theta_uz = Eq(theta_uz.forward, solve(ade_theta_uz, theta_uz.forward), subdomain=self.bound_dom)
        upd_psi_phiy = Eq(psi_phiy.forward, solve(ade_psi_phiy, psi_phiy.forward), subdomain=self.bound_dom)
        upd_theta_phiy = Eq(theta_phiy.forward, solve(ade_theta_phiy, theta_phiy.forward), subdomain=self.bound_dom)
        upd_psi_uy = Eq(psi_uy.forward, solve(ade_psi_uy, psi_uy.forward), subdomain=self.bound_dom)
        upd_theta_uy = Eq(theta_uy.forward, solve(ade_theta_uy, theta_uy.forward), subdomain=self.bound_dom)
        upd_psi_phiz = Eq(psi_phiz.forward, solve(ade_psi_phiz, psi_phiz.forward), subdomain=self.bound_dom)
        upd_theta_phiz = Eq(theta_phiz.forward, solve(ade_theta_phiz, theta_phiz.forward), subdomain=self.bound_dom)
        upd_psi_phix = Eq(psi_phix.forward, solve(ade_psi_phix, psi_phix.forward), subdomain=self.bound_dom)
        upd_theta_phix = Eq(theta_phix.forward, solve(ade_theta_phix, theta_phix.forward), subdomain=self.bound_dom)
        upd_psi_uw = Eq(psi_uw.forward, solve(ade_psi_uw, psi_uw.forward), subdomain=self.bound_dom)
        upd_theta_uw = Eq(theta_uw.forward, solve(ade_theta_uw, theta_uw.forward), subdomain=self.bound_dom)

        if (isinstance(self.track, (ContSlabSingleRailTrack, DiscrSlabSingleRailTrack))):

            op = [
                upd_longw,
                upd_tbw_vert_trans,
                upd_tbw_lat_trans,
                upd_torsw,
                upd_tbw_lat_rot,
                upd_tbw_vert_rot,
                upd_warp,
                # upd_u_sx,
                # upd_u_sz,
                # upd_u_sy,
                # upd_phi_sx,
                # upd_phi_sz,
                # upd_phi_sy,
                upd_psi_ux,
                upd_theta_ux,
                upd_psi_uz,
                upd_theta_uz,
                upd_psi_phiy,
                upd_theta_phiy,
                upd_psi_uy,
                upd_theta_uy,
                upd_psi_phiz,
                upd_theta_phiz,
                upd_psi_phix,
                upd_theta_phix,
                upd_psi_uw,
                upd_theta_uw,
            ]

        else:
            upd_u_sx = Eq(u_sx.forward, solve(slep_trans_x, u_sx.forward))
            upd_u_sz = Eq(u_sz.forward, solve(slep_trans_z, u_sz.forward))
            upd_u_sy = Eq(u_sy.forward, solve(slep_trans_y, u_sy.forward))
            upd_phi_sx = Eq(phi_sx.forward, solve(slep_rot_x, phi_sx.forward))
            upd_phi_sz = Eq(phi_sz.forward, solve(slep_rot_z, phi_sz.forward))
            upd_phi_sy = Eq(phi_sy.forward, solve(slep_rot_y, phi_sy.forward))

            op = [
                upd_longw,
                upd_tbw_vert_trans,
                upd_tbw_lat_trans,
                upd_torsw,
                upd_tbw_lat_rot,
                upd_tbw_vert_rot,
                upd_warp,
                upd_u_sx,
                upd_u_sz,
                upd_u_sy,
                upd_phi_sx,
                upd_phi_sz,
                upd_phi_sy,
                upd_psi_ux,
                upd_theta_ux,
                upd_psi_uz,
                upd_theta_uz,
                upd_psi_phiy,
                upd_theta_phiy,
                upd_psi_uy,
                upd_theta_uy,
                upd_psi_phiz,
                upd_theta_phiz,
                upd_psi_phix,
                upd_theta_phix,
                upd_psi_uw,
                upd_theta_uw,
            ]

        return op, u_z, u_y, phi_x


    def check_stability(self):
        """Check stability."""
        # Courant number
        c_ql = sqrt(self.track.rail.E / self.track.rail.rho)
        C = c_ql * self.dt / self.dx
        cfl_status = "stable" if C < 1 else "unstable"
        print(f"Courant number: {C:.2f} ({cfl_status})")

