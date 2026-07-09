"""Defines required superstructure components.

.. autosummary::
    :toctree: components

    Rail
    DiscrPad
    ContPad
    Sleeper
    Slab
    Ballast
    Wheel
"""

from dataclasses import dataclass, field

from numpy import ndarray, pi, piecewise, real, sqrt
from scipy import linalg
from scipy.interpolate import interp1d


@dataclass(kw_only=True)
class Rail:
    r"""Represents a rail with specific physical and geometric properties.

    UIC 60 rail is typically used as the default rail profile.

    .. hint::
        A set of predefined rail instances is available in the :mod:`database` module.

    Attributes
    ----------
    rl_geo : list[tuple[float, float]]
        Rail outline coordinates in :math:`[\mathrm{m}]`.
    E : float
        Young's modulus of rail in :math:`[\mathrm{Pa}]`.
    G : float
        Shear modulus of rail in :math:`[\mathrm{Pa}]`.
    nu : float
        Poisson's ratio of rail :math:`[-]`.
    kapz : float
        Timoshenko shear correction factor in z direction (vertical) :math:`[-]`.
    kapy : float
        Timoshenko shear correction factor in y direction (lateral) :math:`[-]`.
    mr : float
        Rail mass per unit length in :math:`[\mathrm{kg/m}]`.
    rho : float
        Density of rail in :math:`[\mathrm{kg/m^3}]`.
    etar : float
        Rail loss factor :math:`[-]`.
    dr : float
        Rail damping coefficient (viscous) in :math:`[\mathrm{Ns/m}]`.
    shearc : list[float]
        Coordinates of rail shear center :math:`[m]`.
    centr : list[float]
        Coordinates of centroid :math:`[m]`.
    ez : float
        Vertical shear center eccentricity :math:`[m]`.
    ey : float
        Lateral shear center eccentricity :math:`[m]`.
    Iyr : float
        Second moment of area around y-axis :math:`[m^4]`.
    Izr : float
        Second moment of area around z-axis :math:`[m^4]`.
    Iyz : float
        Product moment of area :math:`[m^4]`.
    Ipr : float
        Polar moment of area :math:`[m^4]`.
    Ar : float
        Cross-section area :math:`[m^2]`.
    Asr : float
        Surface area per unit length of rail :math:`[m^2/m]`.
    Vr : float
        Volume per unit length of rail :math:`[m^3/m]`.
    kapp_s : float
        Correction factor for the effective shear due to restrained torsional warping :math:`[-]
    Iw : float
        Warping constant :math:`[m^6]`.
    Iwz : float
        Warping product moment of area :math:`[m^5]`.
    Iwy : float
        Warping product moment of area :math:`[m^5]`.
    k_w : float
        Warping factor for rail foot :math:`[-]`.
    J : float
        Torsional constant :math:`[m^4]`.
    J_t : float
        Secondary torsional constant :math:`[m^4]`.
    J_rs : float
        Effective shear area :math:`[m^4]`.
    chi : ndarray
        Warping function of the cross-section :math:`[m^2]`.

    Examples
    --------
    Create a custom rail profile:

    >>> custom_rail = Rail(
    ...     rl_geo=[(0.0, 0.0), (0.075, 0.0)],
    ...     E=2.1e11,
    ...     G=8.1e10,
    ...     nu=0.3,
    ...     kapz=0.4,
    ...     kapy=0.4,
    ...     mr=60.0,
    ...     # ... specify remaining parameters ...
    ... )
    """

    rl_geo: list[tuple[float, float]]
    E: float
    G: float
    nu: float | None=None
    kapz: float
    kapy: float
    mr: float
    rho: float
    etar: float
    dr: float
    shearc: list[float]
    centr: list[float]
    ez: float = field(init=False)
    ey: float = field(init=False)
    Iyr: float
    Izr: float
    Iyz: float
    Itr: float
    Ipr: float
    Ar: float
    Asr: float | None=None
    Vr: float | None=None
    kapp_s: float
    Iw: float
    Iwz: float
    Iwy: float
    k_w: float
    J: float
    J_t: float = field(init=False)
    J_rs: float = field(init=False)
    chi: float | None=None

    def __post_init__(self):
        """Post-initialization to calculate derived attributes."""
        self.ez = self.shearc[1] - self.centr[1]
        self.ey = self.shearc[0] - self.centr[0]
        self.J_rs = self.kapp_s * (self.Ipr - self.J)
        self.J_t = self.J_rs + self.Ar * self.kapy * self.ez**2 + self.Ar * self.kapz * self.ey**2

@dataclass(kw_only=True)
class DiscrPad:
    r"""Discrete Pad Class.

    Contains the properties of a discrete pad.

    Attributes
    ----------
    sp_z : float
        Vertical pad stiffness (total value) :math:`[N/m]`.
    sp_y : float
        Lateral pad stiffness (total value) :math:`[N/m]`.
    sp_x : float
        Longitudinal pad stiffness (total value) :math:`[N/m]`.
    sp_w : float
        Warping pad stiffness (total value) :math:`[Nm/m]`.
    sp_xr : float
        Longitudinal rotational pad stiffness (total value) :math:`[Nm/rad]`.
    sp_yr : float
        Lateral rotational pad stiffness (total value) :math:`[Nm/rad]`.
    sp_zr : float
        Vertical rotational pad stiffness (total value) :math:`[Nm/rad]`.
    etap_z : float
        Vertical pad loss factor :math:`[-]`.
    etap_y : float
        Lateral pad loss factor :math:`[-]`.
    etap_x : float
        Longitudinal pad loss factor :math:`[-]`.
    etap_r : float
        Rotational pad loss factor :math:`[-]`.
    fresp_x : float
        Vertical resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    fresp_y : float
        Lateral resonance frequency [Hz]. This frequency is needed for calculating the corresponding
        viscous damping coefficient if not provided.
    fresp_z : float
        Longitudinal resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    fresp_r : float
        Rotational resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    dp_z : float
        Vertical pad damping coefficient (viscous) :math:`[Ns/m]`.
    dp_y : float
        Lateral pad damping coefficient (viscous) :math:`[Ns/m]`.
    dp_x : float
        Longitudinal pad damping coefficient (viscous) :math:`[Ns/m].
    dp_xr : float
        Rotational pad damping coefficient (viscous) :math:`[Nms/rad]`.
    wdthp : float
        Pad width in x-direction :math:`[m]`.
    """

    sp_z: float
    sp_y: float
    sp_x: float
    sp_w: float
    sp_xr: float = field(init=False)
    sp_yr: float = field(init=False)
    sp_zr: float = field(init=False)
    etap_z: float
    etap_y: float
    etap_x: float
    etap_r: float
    fresp_z: float
    fresp_y: float
    fresp_x: float
    fresp_r: float
    dp_z: float
    dp_y: float
    dp_x: float
    dp_xr: float
    wdthp: float

    def __post_init__(self):
        """Post-initialization to calculate derived attributes."""
        self.sp_xr = self.sp_z * (self.wdthp**2) / 12.0
        self.sp_yr = self.sp_z * (self.wdthp**2) / 12.0
        self.sp_zr = (self.sp_y + self.sp_x) * (self.wdthp ** 2) / 12

    def calc_warping_stiffn(self, rail, z_f):
        """Calculate warping stiffness."""
        e_s = rail.shearc[1] - z_f
        self.sp_w = (rail.k_w * e_s) ** 2 * self.wdthp**2 / 12 * self.sp_y

    def calc_viscous_damp(self, rail):
        """Calculate viscous damping coefficients from loss factors."""
        self.fresp_z = sqrt(self.sp_z / rail.mr) / (2 * pi)
        self.dp_z = self.etap_z * self.sp_z / (self.fresp_z * (2 * pi))

        self.fresp_y = sqrt(self.sp_y / rail.mr) / (2 * pi)
        self.dp_y = self.etap_y * self.sp_y / (self.fresp_y * 2 * pi)

        self.fresp_x = sqrt(self.sp_x / rail.mr) / (2 * pi)
        self.dp_x = self.etap_x * self.sp_x / (self.fresp_x * 2 * pi)

        self.fresp_r = sqrt(self.sp_xr / (rail.rho * rail.Ipr)) / (2 * pi)
        self.dp_xr = self.etap_r * self.sp_xr / (self.fresp_r * 2 * pi)

    def calc_viscous_damp_coupled(self, rail, k_mat, m_mat):
        """Calculate coupled viscous damping coefficients."""
        eigval_y_xr, eigvec_y_xr = linalg.eigh(k_mat, m_mat)
        # tudo: coupled longitudinal and vertical damping is not considered yet!!!!

        self.fresp_z = sqrt(self.sp_z / rail.mr) / (2 * pi)
        self.dp_z = self.etap_z * self.sp_z / (self.fresp_z * (2 * pi))

        self.fresp_y = sqrt(real(eigval_y_xr[0])) / (2 * pi)
        self.dp_y = self.etap_y * self.fresp_y * 2 * pi * rail.mr

        self.fresp_x = sqrt(self.sp_x / rail.mr) / (2 * pi)
        self.dp_x = self.etap_x * self.sp_x / (self.fresp_x * 2 * pi)

        self.fresp_r = sqrt(real(eigval_y_xr[1])) / (2 * pi)
        self.dp_xr = self.etap_r * self.fresp_r * 2 * pi * rail.rho * rail.Ipr

    def calc_viscous_damp_cuton(self, cof):
        """Calculate coupled viscous damping coefficients based on cut on frequencies."""
        self.dp_x = self.etap_x * self.sp_x / (cof[0] * (2 * pi))
        self.dp_z = self.etap_z * self.sp_z / (cof[1] * (2 * pi))
        self.dp_y = self.etap_y * self.sp_y / (cof[3] * (2 * pi))
        self.dp_xr = self.etap_r * self.sp_xr / (cof[2] * (2 * pi))

    def interpol_pad_width(self, x, dx, mp):
        """Interpolated pad width distribution along track."""
        def single_mount_pattern(pos):
            start, end = pos - self.wdthp / 2, pos + self.wdthp / 2
            f_left = interp1d([start - dx, start - dx / 2, start], [0, 0.25, 1], "quadratic")
            f_right = interp1d([end, end + dx / 2, end + dx], [1, 0.25, 0], "quadratic")

            pattern = piecewise(
                x,
                [
                    x < start - dx,
                    (x >= start - dx) & (x < start),
                    (x >= start) & (x <= end),
                    (x > end) & (x <= end + dx),
                    x > end + dx,
                ],
                [0, f_left, 1, f_right, 0],
            )
            # Normalize using rectangular integration (sum of values * dx)
            integral = sum(pattern) * dx
            return pattern / integral  # Normalize single pattern

        # Sum normalized contributions from all mounting positions
        return sum(single_mount_pattern(pos) for pos in mp)


@dataclass(kw_only=True)
class ContPad:
    r"""Cont Pad Class.

    Contains the properties of a continuous pad.

    Attributes
    ----------
    sp_z : float
        Vertical pad stiffness (per meter) :math:`[N/m^2]`.
    sp_y : float
        Lateral pad stiffness (per meter) :math:`[N/m^2]`.
    sp_x : float
        Longitudinal pad stiffness (per meter) :math:`[N/m^2]`.
    sp_w : float
        Warping pad stiffness (per meter) :math:`[N/m^2]`.
    sp_zr : float
        Vertical rotational pad stiffness (per meter) :math:`[Nm/rad/m]`.
    sp_yr : float
        Lateral rotational pad stiffness (per meter) :math:`[Nm/rad/m]`.
    sp_xr : float
        Longitudinal rotational pad stiffness (per meter) :math:`[Nm/rad/m]`.
    etap_z : float
        Vertical pad loss factor :math:`[-]`.
    etap_y : float
        Lateral pad loss factor :math:`[-]`.
    etap_x : float
        Longitudinal pad loss factor :math:`[-]`.
    etap_r : float
        Rotational pad loss factor :math:`[-]`.
    fresp_z : float
        Longitudinal resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    fresp_y : float
        Lateral resonance frequency [Hz]. This frequency is needed for calculating the corresponding
        viscous damping coefficient if not provided.
    fresp_x : float
        Vertical resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    fresp_r : float
        Rotational resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    dp_z : float
        Vertical pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    dp_y : float
        Lateral pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    dp_x : float
        Longitudinal pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    dp_xr : float
        Rotational pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    equ_wdthp : float
        Equivalent pad width in x- and y-direction :math:`[m]`.
    """

    sp_z: float
    sp_y: float
    sp_x: float
    sp_w: float
    sp_zr: float = field(init=False)
    sp_yr: float = field(init=False)
    sp_xr: float = field(init=False)
    etap_z: float
    etap_y: float
    etap_x: float
    etap_r: float
    fresp_z: float
    fresp_y: float
    fresp_x: float
    fresp_r: float
    dp_z: float
    dp_y: float
    dp_x: float
    dp_xr: float
    equ_wdthp: float

    def __post_init__(self):
        """Post-initialization to calculate derived attributes."""
        self.sp_xr = self.sp_z * (self.equ_wdthp**2) / 12.0
        self.sp_yr = self.sp_z * (self.equ_wdthp**2) / 12.0
        self.sp_zr = (self.sp_y + self.sp_x) * (self.equ_wdthp ** 2) / 12

    def calc_warping_stiffn(self, rail, z_f):
        """Calculate warping stiffness."""
        e_s = rail.shearc[1] - z_f
        self.sp_w = (rail.k_w * e_s) ** 2 * self.equ_wdthp**2 / 12 * self.sp_y

    def calc_viscous_damp(self, rail):
        """Calculate viscous damping coefficients from loss factors."""
        self.fresp_z = sqrt(self.sp_z / rail.mr) / (2 * pi)
        self.dp_z = self.etap_z * self.sp_z / (self.fresp_z * (2 * pi))

        self.fresp_y = sqrt(self.sp_y / rail.mr) / (2 * pi)
        self.dp_y = self.etap_y * self.sp_y / (self.fresp_y * 2 * pi)

        self.fresp_x = sqrt(self.sp_x / rail.mr) / (2 * pi)
        self.dp_x = self.etap_x * self.sp_x / (self.fresp_x * 2 * pi)

        self.fresp_r = sqrt(self.sp_xr / (rail.rho * rail.Ipr)) / (2 * pi)
        self.dp_xr = self.etap_r * self.sp_xr / (self.fresp_r * 2 * pi)

    def calc_viscous_damp_coupled(self, rail, k_mat, m_mat):
        """Calculate coupled viscous damping coefficients."""
        eigval_y_xr, eigvec_y_xr = linalg.eigh(k_mat, m_mat)
        # tudo: coupled longitudinal and vertical damping is not considered yet!!!!

        self.fresp_z = sqrt(self.sp_z / rail.mr) / (2 * pi)
        self.dp_z = self.etap_z * self.sp_z / (self.fresp_z * (2 * pi))

        self.fresp_y = sqrt(real(eigval_y_xr[0])) / (2 * pi)
        self.dp_y = self.etap_y * self.fresp_y * 2 * pi * rail.mr

        self.fresp_x = sqrt(self.sp_x / rail.mr) / (2 * pi)
        self.dp_x = self.etap_x * self.sp_x / (self.fresp_x * 2 * pi)

        self.fresp_r = sqrt(real(eigval_y_xr[1])) / (2 * pi)
        self.dp_xr = self.etap_r * self.fresp_r * 2 * pi * rail.rho * rail.Ipr

    def calc_viscous_damp_cuton(self, cof):
        """Calculate coupled viscous damping coefficients based on cut on frequencies."""
        self.dp_x = self.etap_x * self.sp_x / (cof[0] * (2 * pi))
        self.dp_z = self.etap_z * self.sp_z / (cof[1] * (2 * pi))
        self.dp_y = self.etap_y * self.sp_y / (cof[3] * (2 * pi))
        self.dp_xr = self.etap_r * self.sp_xr / (cof[2] * (2 * pi))


@dataclass(kw_only=True)
class Sleeper:
    r"""Sleeper Class.

    Contains the properties of a sleeper.

    Attributes
    ----------
    ms : float
        Sleeper mass :math:`[kg]`.
    Is_x : float
        Sleeper moment of inertia around x-axis :math:`[m^4]`.
    Is_y : float
        Sleeper moment of inertia around y-axis :math:`[m^4]`.
    Is_z : float
        Sleeper moment of inertia around z-axis :math:`[m^4]`.
    rhos : float
        Density of sleeper :math:`[kg/m^3]`.
    Bs : float
        Sleeper bending stiffness :math:`[Nm^2]`.
    lengs : float
        Sleeper length in y-direction :math:`[m]`.
    wdths : float
        Sleeper width in x-direction :math:`[m]`.
    hights : float
        Sleeper hight in z-direction :math:`[m]`.
    z_st : float
        Vertical distance from sleeper centroid to top of sleeper :math:`[m]`.
    z_sb : float
        Vertical distance from sleeper centroid to bottom of sleeper :math:`[m]`.
    f_x : float
        Equivalent sleeper factor (x-direction) :math:`[-]`.
    f_z : float
        Equivalent sleeper factor (z-direction) :math:`[-]`.
    """

    ms: float
    Is_x: float
    Is_y: float
    Is_z: float
    rhos: float
    Bs: float
    lengs: float
    wdths: float
    hights: float
    z_st: float
    z_sb: float
    f_z: float = 1.0
    f_x: float = 1.0

    def calc_equiv_sleeper_factors(self, y_sc, equi_sim):
        """Calculate equivalent sleeper factors according to Kostovasilis."""
        if not equi_sim:
            pass
        else:
            self.f_z = 1 + 12 * y_sc**2 /(self.lengs**2 + self.hights**2)
            self.f_x = 1 + 12 * y_sc**2 / (self.lengs**2 + self.wdths**2)


@dataclass(kw_only=True)
class Slab:
    r"""Slab class.

    Contains the properties of the slab.

    Attributes
    ----------
    ms : float
        Slab mass per meter :math:`[kg/m]`.
    Is_z : float
        Slab moment of inertia around z-axis :math:`[m^4/m]`.
    Is_y : float
        Slab moment of inertia around y-axis :math:`[m^4/m]`.
    Is_x : float
        Slab moment of inertia around x-axis :math:`[m^4/m]`.
    rhos : float
        Density of slab :math:`[kg/m^3]`.
    Bs : float
        Slab bending stiffness :math:`[Nm^2]`.
    lengs : float
        Slab length in y-direction :math:`[m]`.
    equ_wdths : float
        Equivalent slab width in x-direction :math:`[m]`.
    heights : float
        Slab hight in z-direction :math:`[m]`.
    z_st : float
        Vertical distance from slab centroid to top of slab :math:`[m]`.
    z_sb : float
        Vertical distance from slab centroid to bottom of slab :math:`[m]`.
    f_x : float
        Equivalent slab factor (x-direction) :math:`[-]`.
    f_z : float
        Equivalent slab factor (z-direction) :math:`[-]`.
    """

    ms: float
    Is_z: float
    Is_y: float
    Is_x: float
    rhos: float
    Bs: float
    lengs: float
    equ_wdths: float
    heights: float
    z_st: float
    z_sb: float
    f_z: float = 1.0
    f_x: float = 1.0

    def calc_equiv_slab_factors(self, y_sc, equi_sim):
        """Calculate equivalent slab factors according to Kostovasilis."""
        if not equi_sim:
            pass

        else:
            self.f_z = 1 + 12 * y_sc ** 2 / (self.lengs ** 2 + self.heights ** 2)
            self.f_x = 1 + 12 * y_sc ** 2 / (self.lengs ** 2 + self.equ_wdths ** 2)


@dataclass(kw_only=True)
class Ballast:
    r"""Ballast Class.

    Contains the properties of the ballast.

    .. caution::
        Properties of the ballast can either be defined as discrete values acting at the
        mounting positions or as continuous values acting per meter depending. The values need to be
        chosen accordingly to the track type.

    Attributes
    ----------
    sb_z : float
        Vertical ballast stiffness (total value :math:`[N/m]` or per meter :math:`[N/m^2]`).
    sb_y : float
        Lateral ballast stiffness (total value :math:`[N/m]` or per meter :math:`[N/m^2]`).
    sb_x : float
        Longitudinal ballast stiffness (total value :math:`[N/m]` or per meter :math:`[N/m^2]`).
    sb_zr : float
        Vertical rotational ballast stiffness (total value :math:`[Nm/rad]` or per meter
        :math:`[Nm/rad/m]`).
    sb_yr : float
        Lateral rotational ballast stiffness (total value :math:`[Nm/rad]` or per meter
        :math:`[Nm/rad/m]`).
    sb_xr : float
        Longitudinal rotational ballast stiffness (total value :math:`[Nm/rad]` or per meter
        :math:`[Nm/rad/m]`).
    etab_z : float
        Vertical ballast loss factor :math:`[-]`.
    etab_y : float
        Lateral ballast loss factor :math:`[-]`.
    etab_x : float
        Longitudinal ballast loss factor :math:`[-]`.
    etab_r : float
        Rotational ballast loss factor :math:`[-]`.
    fresb_z : float
        Longitudinal ballast resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    fresb_y : float
        Lateral ballast resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    fresb_x : float
        Vertical ballast resonance frequency [Hz]. This frequency is needed for calculating the
        corresponding viscous damping coefficient if not provided.
    fresb_zr : float
        Rotational ballast resonance frequency for z-axis [Hz]. This frequency is needed for
        calculating the corresponding viscous damping coefficient if not provided.
    fresb_yr : float
        Rotational ballast resonance frequency for y-axis [Hz]. This frequency is needed for
        calculating the corresponding viscous damping coefficient if not provided.
    fresb_xr : float
        Rotational ballast resonance frequency for x-axis [Hz]. This frequency is needed for
        calculating the corresponding viscous damping coefficient if not provided.
    db_z : float
        Vertical ballast damping coefficient (viscous) :math:`[Ns/m^2]`.
    db_y : float
        Lateral ballast damping coefficient (viscous) :math:`[Ns/m^2]`.
    db_x : float
        Longitudinal ballast damping coefficient (viscous) :math:`[Ns/m^2]`.
    db_zr : float
        Rotational ballast damping coefficient for z-axis (viscous) :math:`[Nms/rad/m]`.
    db_yr : float
        Rotational ballast damping coefficient for y-axis (viscous) :math:`[Nms/rad/m]`.
    db_xr : float
        Rotational ballast damping coefficient for x-axis (viscous) :math:`[Nms/rad/m]`.
    """

    sb_z: float
    sb_y: float
    sb_x: float
    sb_xr: float
    sb_yr: float
    sb_zr: float
    etab_z: float
    etab_y: float
    etab_x: float
    etab_r: float
    fresb_x: float
    fresb_y: float
    fresb_z: float
    fresb_xr: float
    fresb_zr: float
    fresb_yr: float
    db_z: float
    db_y: float
    db_x: float
    db_xr: float
    db_yr: float
    db_zr: float

    def calc_rotational_stiffn(self, seclay):
        """Calculate rotational stiffnesses from ballast stiffnesses and sleeper/slab dimensions."""
        s_l = seclay.lengs

        if isinstance(seclay, Sleeper):
            s_w = seclay.wdths
        if isinstance(seclay, Slab):
            s_w = seclay.equ_wdths

        self.sb_xr = s_l**2 / 12 * self.sb_z
        self.sb_yr = s_w**2 / 12 * self.sb_z
        self.sb_zr = s_l**2 / 12 * self.sb_x + s_w**2 / 12 * self.sb_y

    def calc_viscous_damp(self, seclay):
        """Calculate viscous damping coefficients from loss factors."""
        def _f_db(stiff, mass, eta):
            if mass <= 0 or stiff <= 0:
                return 0.0, 0.0
            fres = sqrt(stiff / mass) / (2 * pi)
            db = eta * stiff / (fres * 2 * pi)
            return fres, db

        self.fresb_z, self.db_z = _f_db(self.sb_z, seclay.ms, self.etab_z)
        self.fresb_y, self.db_y = _f_db(self.sb_y, seclay.ms, self.etab_y)
        self.fresb_x, self.db_x = _f_db(self.sb_x, seclay.ms, self.etab_x)
        self.fresb_xr, self.db_xr = _f_db(self.sb_xr, seclay.rhos * seclay.Is_x, self.etab_r)
        self.fresb_zr, self.db_zr = _f_db(self.sb_zr, seclay.rhos * seclay.Is_z, self.etab_r)
        self.fresb_yr, self.db_yr = _f_db(self.sb_yr, seclay.rhos * seclay.Is_y, self.etab_r)

    def calc_viscous_damp_cuton(self, cof):
        """Calculate coupled viscous damping coefficients based on cut on frequencies."""
        self.db_x = self.etab_x * self.sb_x / (cof[7] * (2 * pi))
        self.db_z = self.etab_z * self.sb_z / (cof[8] * (2 * pi))
        self.db_y = self.etab_y * self.sb_y / (cof[9] * (2 * pi))
        self.db_xr = self.etab_r * self.sb_xr / (cof[10] * (2 * pi))
        self.db_yr = self.etab_r * self.sb_yr / (cof[11] * (2 * pi))
        self.db_zr = self.etab_r * self.sb_zr / (cof[12] * (2 * pi))


@dataclass(kw_only=True)
class Wheel:
    r"""Wheel Class.

    Contains the properties of a wheel.

    Attributes
    ----------
    w_geo_cross_sec : list[tuple[float, float]]
        Wheel cross-sectional geometry coordinates (y-z plane) :math:`[m]`.
        TODO: Define coordinate system
    w_prof : str
        Wheel running surface profile.
    w_geo : list[tuple[float, float]]
        Wheel geometry coordinates (x-y plane) :math:`[m]`.
        TODO: Define coordinate system
    mw : float
        Wheel mass :math:`[kg]`.
    mw_red : float
        Reduced wheel mass :math:`[kg]`. Needed in order to calculate the lateral dynamics
        according to :cite:t:`wu2004a`.
    rw : float
        Wheel radius from the axis of rotation to the contact point :math:`[m]`.
    """

    w_geo_cross_sec: list[tuple[float, float]]
    w_prof: str
    w_geo: list[tuple[float, float]]
    mw: float
    mw_red: float
    rw: float


@dataclass(kw_only=True)
class WheelRoughness:
    r"""Wheel Roughness Class.

    Contains a wheel roughness spectrum in frequency domain.

    Attributes
    ----------
    w_rough : tuple[list[float], list[float]]
        Wheel roughness spectrum :math:`[f, m]`.
    """

    w_rough: tuple[list[float], list[float]]


@dataclass(kw_only=True)
class WheelGreensfunc:
    r"""Wheel Greens Function Class.

    Contains the Green's function of a wheel.

    Attributes
    ----------
    w_gf : ndarray
        Green's function data. Contains the response of the wheel to a unit impulse at
        multiple points :math:`[m/N]`.
    w_gf_freq : ndarray
        Frequency values of the Green's function :math:`[Hz]`.
    """

    w_gf: ndarray
    w_gf_freq: ndarray
