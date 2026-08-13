# ruff: noqa: N806, N803
"""Analytical models for continuous tracks using Kostovasilis approach.

This module provides analytical solutions for the dynamic response of railway tracks modeled
as continuously supported beams based on the Kostovasilis approach.

.. autosummary::
    :toctree: analytical/

    TBCont1LKosto
    TBCont2LKosto
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import scipy.linalg as sp
from numpy import array, block, exp, eye, imag, isfinite, ndarray, newaxis, pi, zeros, zeros_like
from numpy import errstate as np_errstate

from rolland.helper.build_matrix import (
    build_equ_sleeper_matrix,
    build_fnd_damp_matrix,
    build_fnd_stiff_matrix,
    build_pad_ballast_damp_matrices,
    build_pad_ballast_stiff_matrices,
    build_rail_matrices,
    build_sleep_mass_matrix,
    build_transfm_matrices,
    calc_cut_on_frequ,
)
from rolland.track import (
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
)


@dataclass(kw_only=True)
class AnalyticalMethodsKosto(ABC):
    r"""Abstract base class for analytical methods based on Kostovasilis.

    Implements the core solver using the Residue Theorem to calculate
    track mobility. Solves the generalized eigenvalue problem in the frequency domain.

    Attributes
    ----------
    f : numpy.ndarray
        Excitation frequencies :math:`[Hz]`.
    force_dir : str
        Direction of the applied force, either 'vertical' or 'lateral'.
    x_excit : float
        Longitudinal coordinate of the excitation point :math:`[m]`.
    x_resp : float | list[float] | numpy.ndarray
        Longitudinal coordinate(s) of the response point(s) :math:`[m]`.
    z_e : float
        Vertical eccentricity of excitation (rail head <--> centroid) :math:`[m]`.
    y_e : float
        Lateral eccentricity of excitation (rail head <--> centroid) :math:`[m]`.
    z_f : float
        Vertical eccentricity of reaction (rail foot <--> centroid) :math:`[m]`.
    y_f : float
        Lateral eccentricity of reaction (rail foot <--> centroid) :math:`[m]`.
    mobility : numpy.ndarray
        Calculated mobility of the track :math:`[m/s/N]`.
    """

    f: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    force_dir: str = "vertical"
    x_excit: float = 0.0
    x_resp: float | list[float] | ndarray | None = None
    z_e: float = 0.0
    y_e: float = 0.0
    z_f: float = 0.0
    y_f: float = 0.0
    mobility: ndarray = field(init=False, default_factory=lambda: array([]),
                              metadata={"default_repr": "numpy.array([])"})

    def __post_init__(self):
        """Post-initialization to set defaults, validate parameters and compute mobility."""
        self._set_default_x_resp()
        if self.force_dir not in ["vertical", "lateral"]:
            msg = "force_dir must be 'vertical' or 'lateral'"
            raise ValueError(msg)
        self.compute_mobility()

    def _set_default_x_resp(self):
        """Set default value for x_resp if it is None, and convert x_resp to array."""
        if self.x_resp is None:
            self.x_resp = array([self.x_excit])
        else:
            self.x_resp = array(self.x_resp, ndmin=1)

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

    @abstractmethod
    def compute_mobility(self):
        """Must be implemented by subclasses to build specific matrices."""
        raise NotImplementedError

    def _solve_and_compute_mobility(self, K0, K1, K2, M, K_fnd, D_fnd, damp_type):
        """
        Solves the Generalized Eigenvalue Problem and computes response via Residue Theorem.

        Corresponds to Kostovasilis Paper Eq. (30) and (31).

        Parameters
        ----------
        K0 : numpy.ndarray
            Stiffness matrix of the rail.
        K1 : numpy.ndarray
            Stiffness matrix of the rail.
        K2 : numpy.ndarray
            Stiffness matrix of the rail.
        M : numpy.ndarray
            Mass matrix of the system (Rail + Foundation contributions).
        K_fnd : numpy.ndarray
            Foundation stiffness matrix.
        D_fnd : numpy.ndarray
            Foundation damping matrix.
        damp_type : str
            Damping type, either 'viscous' or 'hysteretic'.
        """
        if self.f.size == 0:
            self.mobility = array([])
            return

        num_freqs = len(self.f)
        size = K0.shape[0]  # Detects 7 (1-layer) or 14 (2-layer)
        mobility = zeros((num_freqs, len(self.x_resp)), dtype=complex)

        # Prepare identity and zero blocks
        I_mat = eye(size)
        Z = zeros((size, size))

        # Build Force Vector F (Point excitation)
        F = zeros(size, dtype=complex)

        # Indices: 1=Vertical, 2=Lateral, 3=Torsional (Relative to Rail Head)
        if self.force_dir == "lateral":
            F[2] = 1.0
            F[3] = self.z_e * F[2]
        else:  # vertical
            F[1] = 1.0
            F[3] = self.y_e * F[1]

        # Pre-assemble constant blocks for A2
        A2 = block([[Z, K2], [-I_mat, Z]])

        for i, f_hz in enumerate(self.f):
            w = 2 * pi * f_hz

            # Construct A1 matrix depending on damping model
            # A1 * v = lambda * A2 * v
            K_dyn = (
                K0 + K_fnd + 1j * w * D_fnd - w**2 * M
                if damp_type == "viscous"
                else K0 + K_fnd - w**2 * M
            )

            A1 = block([[K_dyn, -K1], [Z, I_mat]])

            # Solve GEP: A1 * v = lambda * A2 * v
            vals, V_l, V_r = sp.eig(A1, A2, left=True, right=True)

            # Clean infinite values and transform to wavenumber
            vals[~isfinite(vals)] = 0
            xi = 1j * vals

            # Filter propagating waves (Im(xi) < 0) and numerical stability
            mask = (imag(xi) < 0) & isfinite(xi) & (abs(xi) < 1e6)

            xi_n = xi[mask]
            Un_R = V_r[:size, mask]  # Extract displacement part of eigenvector
            Un_L = V_l[:size, mask]

            # Residue Summation (Eq. 30)
            U_total = zeros((size, len(self.x_resp)), dtype=complex)
            factor = -1j

            for n in range(len(xi_n)):
                # Eq. (31): A'(xi) = -i*K1 - 2*xi*K2
                A_prime = -1j * K1 - 2 * xi_n[n] * K2

                UnR_n = Un_R[:, n]
                UnL_n = Un_L[:, n]

                denominator = UnL_n.conj().T @ A_prime @ UnR_n

                # Check against numerical zero (epsilon) to avoid div/0
                if abs(denominator) < 1e-12:
                    continue

                numerator = UnL_n.conj().T @ F
                prop_term = exp(-1j * xi_n[n] * abs(self.x_resp - self.x_excit))

                U_total += factor * (numerator / denominator) * UnR_n[:, newaxis] * prop_term[newaxis, :]

            # Convert Displacement to Velocity: v = i * w * u
            velocity = 1j * w * U_total

            # Calculate Mobility at excitation point (accounting for eccentricity)
            if self.force_dir == "lateral":
                mobility[i, :] = velocity[2, :] + self.z_e * velocity[3, :]
            else:
                mobility[i, :] = velocity[1, :] + self.y_e * velocity[3, :]

        self.mobility = mobility.squeeze()


@dataclass(kw_only=True)
class TBCont1LKosto(AnalyticalMethodsKosto):
    r"""Timoshenko beam on continuous 1-layer foundation according to :cite:p:`kostovasilis_semi-analytical_2017`.

    Utilizes a single-layer support with continuous track properties, applying Timoshenko
    beam theory. The excitation is stationary. This method calculates the track's mobility
    considering 7 degrees of freedom (rail only), neglecting sleeper mass effects.

    Attributes
    ----------
    track : ContSlabSingleRailTrack
        Track instance containing rail and pad properties.
    damp_type : str
        Damping type, either 'viscous' or 'hysteretic'.
    """

    track: ContSlabSingleRailTrack
    damp_type: str = "hysteretic"

    def compute_mobility(self):
        """Compute track mobility for single-layer continuous support."""
        if self.f.size == 0:
            self.mobility = array([])
            return

        track = self.track

        # 1. Build Matrices (External functions assumed)
        K0r, K1r, K2r, Mr = build_rail_matrices(track.rail, self.damp_type)
        Tf, _, _ = build_transfm_matrices(self.z_f, self.y_f, 0, 0, 0)
        Kp, _ = build_pad_ballast_stiff_matrices(track, self.damp_type)

        # Foundation Stiffness
        K_fnd = build_fnd_stiff_matrix(Kp, Tf)

        # Foundation Damping (only needed for viscous)
        D_fnd = zeros_like(K_fnd)
        if self.damp_type == "viscous":
            cof = calc_cut_on_frequ(K0r, K_fnd, Mr)
            Dp, _ = build_pad_ballast_damp_matrices(track, cof)
            D_fnd = build_fnd_damp_matrix(Dp, Tf)
            # Add rail damping
            D_fnd[1, 1] += track.rail.dr
            D_fnd[2, 2] += track.rail.dr

        # 2. Solve using base class method (Matrices are 7x7)
        # Note: Slicing [:7,:7] ensures compatibility with the 1-layer model
        self._solve_and_compute_mobility(
            K0r, K1r, K2r, Mr, K_fnd[:7, :7], D_fnd[:7, :7], self.damp_type,
        )


@dataclass(kw_only=True)
class TBCont2LKosto(AnalyticalMethodsKosto):
    r"""Timoshenko beam on continuous 2-layer foundation according to :cite:p:`kostovasilis_analytical_2017`.

    Utilizes a double-layer support (rail and sleeper/slab masses) with continuous track
    properties. Considers 14 degrees of freedom (7 for rail, 7 for sleeper) to account for
    vertical, lateral, and torsional deformations including sleeper dynamics.

    Attributes
    ----------
    track : ContBallastedSingleRailTrack
        Track instance containing rail, pad, sleeper, and ballast properties.
    damp_type : str
        Damping type, either 'viscous' or 'hysteretic'.
    equi_sm : bool
        If True, an equivalent sleeper model is applied to distribute discrete support.
    y_sc : float
        Lateral distance from excitation point to sleeper center :math:`[m]`.
    """

    track: ContBallastedSingleRailTrack
    damp_type: str = "hysteretic"
    equi_sm: bool = True
    y_sc: float = 0.7175

    def compute_mobility(self):
        """Compute track mobility for double-layer continuous support."""
        if self.f.size == 0:
            self.mobility = array([])
            return

        track = self.track

        # 1. Build Sub-Matrices
        K0r, K1r, K2r, Mr = build_rail_matrices(track.rail, self.damp_type)
        Tf, Tst, Tsb = build_transfm_matrices(
            self.z_f, self.y_f, track.slab.z_st, track.slab.z_sb, 0,
        )
        E = build_equ_sleeper_matrix(track)
        Kp, Kb = build_pad_ballast_stiff_matrices(track, self.damp_type, E)
        Ms = build_sleep_mass_matrix(track, E)

        # Assemble 14x14 Block Matrices
        # Structure: [[Rail (7x7), 0], [0, Sleeper (7x7)]]
        Z7 = zeros((7, 7))
        Z14 = zeros((7, 14))

        K0 = block([[K0r, Z7], [Z14]])
        K1 = block([[K1r, Z7], [Z14]])
        K2 = block([[K2r, Z7], [Z14]])
        M = block([[Mr, Z7], [Z7, Ms]])

        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb)

        # Damping
        D_fnd = zeros_like(K0)
        if self.damp_type == "viscous":
            cof = calc_cut_on_frequ(K0r, K_fnd, Mr, Ms)
            Dp, Db = build_pad_ballast_damp_matrices(track, cof, E)
            D_fnd = build_fnd_damp_matrix(Dp, Tf, Db, Tst, Tsb)
            D_fnd[1, 1] += track.rail.dr
            D_fnd[2, 2] += track.rail.dr

        # 2. Solve using base class method (Matrices are 14x14)
        self._solve_and_compute_mobility(
            K0, K1, K2, M, K_fnd, D_fnd, self.damp_type,
        )
