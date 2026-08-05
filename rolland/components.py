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
from typing import Literal

import numpy as np
from numpy import array, ndarray


def _damping_mode(eta_values, viscous_values, eta_name, viscous_name):
    missing_eta = np.sum(array([value is None for value in eta_values]))
    missing_d = np.sum(array([value is None for value in viscous_values]))

    if 0 < missing_eta < len(eta_values):
        msg = f"{eta_name} values are missing ({missing_eta} of {len(eta_values)} values are missing)."
        raise ValueError(msg)

    if 0 < missing_d < len(viscous_values):
        msg = f"{viscous_name} are missing ({missing_d} of {len(viscous_values)} values are missing)."
        raise ValueError(msg)

    if missing_eta == len(eta_values) and missing_d == len(viscous_values):
        msg = f"Both {eta_name} and {viscous_name} are missing. Please provide one set of values."
        raise ValueError(msg)

    if missing_eta == 0 and missing_d == 0:
        msg = f"Both {eta_name} and {viscous_name} are provided. Please provide one set of values."
        raise ValueError(msg)

    if missing_eta == 0:
        return "hysteretic"

    return "viscous"


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
    nu : float, optional, default: None
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
    ez : float, internal (automatically calculated)
        Vertical shear center eccentricity :math:`[m]`.
    ey : float, internal (automatically calculated)
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
    Asr : float, optional, default: None
        Surface area per unit length of rail :math:`[m^2/m]`.
    Vr : float, optional, default: None
        Volume per unit length of rail :math:`[m^3/m]`.
    kapp_s : float
        Correction factor for the effective shear due to restrained torsional warping :math:`[-]`.
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
    J_t : float, internal (automatically calculated)
        Secondary torsional constant :math:`[m^4]`.
    J_rs : float, internal (automatically calculated)
        Effective shear area :math:`[m^4]`.
    chi : float, optional, default: None
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
    sp_w : float, internal (automatically calculated)
        Warping pad stiffness (total value) :math:`[Nm/m]`.
    sp_xr : float, internal (automatically calculated)
        Longitudinal rotational pad stiffness (total value) :math:`[Nm/rad]`.
    sp_yr : float, internal (automatically calculated)
        Lateral rotational pad stiffness (total value) :math:`[Nm/rad]`.
    sp_zr : float, internal (automatically calculated)
        Vertical rotational pad stiffness (total value) :math:`[Nm/rad]`.
    etap_z : float, optional, default: None
        Vertical pad loss factor :math:`[-]`.
    etap_y : float, optional, default: None
        Lateral pad loss factor :math:`[-]`.
    etap_x : float, optional, default: None
        Longitudinal pad loss factor :math:`[-]`.
    etap_r : float, optional, default: None
        Rotational pad loss factor :math:`[-]`.
    dp_z : float, optional, default: None
        Vertical pad damping coefficient (viscous) :math:`[Ns/m]`.
    dp_y : float, optional, default: None
        Lateral pad damping coefficient (viscous) :math:`[Ns/m]`.
    dp_x : float, optional, default: None
        Longitudinal pad damping coefficient (viscous) :math:`[Ns/m]`.
    dp_xr : float, optional, default: None
        Rotational pad damping coefficient (viscous) :math:`[Nms/rad]`.
    wdthp : float
        Pad width in x-direction :math:`[m]`.
    """

    sp_z: float
    sp_y: float
    sp_x: float
    sp_w: float = field(init=False)
    sp_xr: float = field(init=False)
    sp_yr: float = field(init=False)
    sp_zr: float = field(init=False)
    etap_z: float | None = None
    etap_y: float | None = None
    etap_x: float | None = None
    etap_r: float | None = None
    damping_mode: Literal["viscous", "hysteretic"] = field(init=False)
    dp_z: float | None = None
    dp_y: float | None = None
    dp_x: float | None = None
    dp_xr: float | None = None
    wdthp: float

    def __post_init__(self):
        """Post-initialization to calculate derived attributes."""
        self.sp_xr = self.sp_z * (self.wdthp**2) / 12.0
        self.sp_yr = self.sp_z * (self.wdthp**2) / 12.0
        self.sp_zr = (self.sp_y + self.sp_x) * (self.wdthp ** 2) / 12

        self.damping_mode = _damping_mode(
            (self.etap_z, self.etap_y, self.etap_x, self.etap_r),
            (self.dp_z, self.dp_y, self.dp_x, self.dp_xr),
            "Loss factors",
            "viscous damping coefficients",
        )


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
    sp_w : float, internal (automatically calculated)
        Warping pad stiffness (per meter) :math:`[N/m^2]`.
    sp_zr : float, internal (automatically calculated)
        Vertical rotational pad stiffness (per meter) :math:`[Nm/rad/m]`.
    sp_yr : float, internal (automatically calculated)
        Lateral rotational pad stiffness (per meter) :math:`[Nm/rad/m]`.
    sp_xr : float, internal (automatically calculated)
        Longitudinal rotational pad stiffness (per meter) :math:`[Nm/rad/m]`.
    etap_z : float, optional, default: None
        Vertical pad loss factor :math:`[-]`.
    etap_y : float, optional, default: None
        Lateral pad loss factor :math:`[-]`.
    etap_x : float, optional, default: None
        Longitudinal pad loss factor :math:`[-]`.
    etap_r : float, optional, default: None
        Rotational pad loss factor :math:`[-]`.
    dp_z : float, optional, default: None
        Vertical pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    dp_y : float, optional, default: None
        Lateral pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    dp_x : float, optional, default: None
        Longitudinal pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    dp_xr : float, optional, default: None
        Rotational pad damping coefficient (viscous) :math:`[Ns/m^2]`.
    wdthp : float
        Equivalent pad width in x- and y-direction :math:`[m]`.
    """

    sp_z: float
    sp_y: float
    sp_x: float
    sp_w: float = field(init=False)
    sp_zr: float = field(init=False)
    sp_yr: float = field(init=False)
    sp_xr: float = field(init=False)
    etap_z: float | None = None
    etap_y: float | None = None
    etap_x: float | None = None
    etap_r: float | None = None
    damping_mode: Literal["viscous", "hysteretic"] = field(init=False)
    dp_z: float | None = None
    dp_y: float | None = None
    dp_x: float | None = None
    dp_xr: float | None = None
    wdthp: float

    def __post_init__(self):
        """Post-initialization to calculate derived attributes."""
        self.sp_xr = self.sp_z * (self.wdthp**2) / 12.0
        self.sp_yr = self.sp_z * (self.wdthp**2) / 12.0
        self.sp_zr = (self.sp_y + self.sp_x) * (self.wdthp ** 2) / 12
        self.damping_mode = _damping_mode(
            (self.etap_z, self.etap_y, self.etap_x, self.etap_r),
            (self.dp_z, self.dp_y, self.dp_x, self.dp_xr),
            "Loss factors",
            "viscous damping coefficients",
        )


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
    Bs : float, optional, default: None
        Sleeper bending stiffness :math:`[Nm^2]`.
    lengs : float
        Sleeper length in y-direction :math:`[m]`.
    wdths : float
        Sleeper width in x-direction :math:`[m]`.
    heights : float
        Sleeper hight in z-direction :math:`[m]`.
    z_st : float
        Vertical distance from sleeper centroid to top of sleeper :math:`[m]`.
    z_sb : float
        Vertical distance from sleeper centroid to bottom of sleeper :math:`[m]`.
    y_sc : float, optional, default: 0.7175
        Lateral sleeper eccentricity :math:`[m]`.
        It is half of the track gauge (0.7175 m for standard gauge).
    f_x : float, optional, default: 1.0
        Equivalent sleeper factor (x-direction) :math:`[-]`.
    f_z : float, optional, default: 1.0
        Equivalent sleeper factor (z-direction) :math:`[-]`.
    equi_sm : bool, optional, default: True
        If True the model uses the equivalent sleeper model, otherwise the real sleeper model is
        used.
    """

    ms: float
    Is_x: float
    Is_y: float
    Is_z: float
    rhos: float
    Bs: float | None = None
    lengs: float
    wdths: float
    heights: float
    z_st: float
    z_sb: float
    y_sc: float = 0.7175
    f_z: float = 1.0
    f_x: float = 1.0
    equi_sm: bool = True


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
    Bs : float, optional, default: None
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
    y_sc : float, optional, default: 0.7175
        Lateral slab eccentricity :math:`[m]`.
        It is half of the track gauge (0.7175 m for standard gauge).
    f_x : float, optional, default: 1.0
        Equivalent slab factor (x-direction) :math:`[-]`.
    f_z : float, optional, default: 1.0
        Equivalent slab factor (z-direction) :math:`[-]`.
    equi_sm : bool, optional, default: True
        If True the model uses the equivalent sleeper model, otherwise the real sleeper model is
        used.
    """

    ms: float
    Is_z: float
    Is_y: float
    Is_x: float
    rhos: float
    Bs: float | None = None
    lengs: float
    equ_wdths: float
    heights: float
    z_st: float
    z_sb: float
    y_sc: float = 0.7175
    f_z: float = 1.0
    f_x: float = 1.0
    equi_sm: bool = True


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
    sb_zr : float, internal (automatically calculated)
        Vertical rotational ballast stiffness.
    sb_yr : float, internal (automatically calculated)
        Lateral rotational ballast stiffness.
    sb_xr : float, internal (automatically calculated)
        Longitudinal rotational ballast stiffness.
    etab_z : float, optional, default: None
        Vertical ballast loss factor :math:`[-]`.
    etab_y : float, optional, default: None
        Lateral ballast loss factor :math:`[-]`.
    etab_x : float, optional, default: None
        Longitudinal ballast loss factor :math:`[-]`.
    etab_r : float, optional, default: None
        Rotational ballast loss factor :math:`[-]`.
    db_z : float, optional, default: None
        Vertical ballast damping coefficient (viscous) :math:`[Ns/m^2]`.
    db_y : float, optional, default: None
        Lateral ballast damping coefficient (viscous) :math:`[Ns/m^2]`.
    db_x : float, optional, default: None
        Longitudinal ballast damping coefficient (viscous) :math:`[Ns/m^2]`.
    db_zr : float, optional, default: None
        Rotational ballast damping coefficient for z-axis (viscous).
    db_yr : float, optional, default: None
        Rotational ballast damping coefficient for y-axis (viscous).
    db_xr : float, optional, default: None
        Rotational ballast damping coefficient for x-axis (viscous).
    """

    sb_z: float
    sb_y: float
    sb_x: float
    sb_xr: float = field(init=False)
    sb_yr: float = field(init=False)
    sb_zr: float = field(init=False)
    etab_z: float | None = None
    etab_y: float | None = None
    etab_x: float | None = None
    etab_r: float | None = None
    damping_mode: Literal["viscous", "hysteretic"] = field(init=False)
    db_z: float | None = None
    db_y: float | None = None
    db_x: float | None = None
    db_xr: float | None = None
    db_yr: float | None = None
    db_zr: float | None = None

    def __post_init__(self):
        """Validate that one complete damping representation is provided."""
        self.damping_mode = _damping_mode(
            (self.etab_z, self.etab_y, self.etab_x, self.etab_r),
            (self.db_z, self.db_y, self.db_x, self.db_xr, self.db_yr, self.db_zr),
            "Loss factors",
            "viscous damping coefficients",
        )


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
