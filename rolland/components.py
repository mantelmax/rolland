"""Defines required superstructure components.

.. autosummary::
    :toctree: components

    Rail
    RailRoughness
    DiscrPad
    ContPad
    Sleeper
    Slab
    Ballast
    Wheel
    WheelRoughness
    WheelGreensfunc
"""

from dataclasses import dataclass, field

from numpy import ndarray


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
    kap : list[float]
        Timoshenko shear correction factor (vertical, lateral) :math:`[-]`.
    mr : float
        Rail mass per unit length in :math:`[\mathrm{kg/m}]`.
    rho : float
        Density of rail in :math:`[\mathrm{kg/m^3}]`.
    etar : float
        Rail loss factor :math:`[-]`.
    fresr : float
        Rail resonance frequency in :math:`[\mathrm{Hz}]`.
    dr : float
        Rail damping coefficient (viscous) in :math:`[\mathrm{Ns/m}]`.
    gamr : list[float]
        Coordinates of rail shear center in :math:`[\mathrm{m}]`.
    epsr : list[float]
        Coordinates of center of gravity in :math:`[\mathrm{m}]`.
    Iyr : float
        Area moment of inertia of rail around y-axis in :math:`[\mathrm{m^4}]`.
    Izr : float
        Area moment of inertia of rail around z-axis in :math:`[\mathrm{m^4}]`.
    Itr : float
        Torsional constant of rail in :math:`[\mathrm{m^4}]`.
    Ipr : float
        Polar moment of inertia of rail in :math:`[\mathrm{m^4}]`.
    Ar : float
        Cross-sectional area of rail in :math:`[\mathrm{m^2}]`.
    Asr : float
        Surface area per unit length of rail in :math:`[\mathrm{m^2/m}]`.
    Vr : float
        Volume per unit length of rail in :math:`[\mathrm{m^3/m}]`.

    Examples
    --------
    Create a custom rail profile:

    >>> custom_rail = Rail(
    ...     rl_geo=[(0.0, 0.0), (0.075, 0.0)],
    ...     E=2.1e11,
    ...     G=8.1e10,
    ...     nu=0.3,
    ...     kap=[0.4, 0.4],
    ...     mr=60.0,
    ...     # ... specify remaining parameters ...
    ... )
    """

    rl_geo: list[tuple[float, float]]
    E: float
    G: float
    nu: float
    kap: list[float]
    mr: float
    rho: float
    etar: float
    fresr: float
    dr: float
    gamr: list[float]
    epsr: list[float]
    Iyr: float
    Izr: float
    Itr: float
    Ipr: float
    Ar: float
    Asr: float
    Vr: float

@dataclass(kw_only=True)
class RailRoughness:
    r"""Rail Roughness Class.

    Contains a rail roughness spectrum in frequency domain, which can later be used to calculate the
    rail roughness along the track.

    Attributes
    ----------
    r_rough : tuple[list[float], list[float]]
        Rail roughness spectrum :math:`[f, m]`.
    """

    r_rough: tuple[list[float], list[float]]

@dataclass(kw_only=True)
class DiscrPad:
    r"""Discrete Pad Class.

    Contains the properties of a discrete pad.

    Attributes
    ----------
    sp : list[float], default=[0.0, 0.0]
        Vertical/lateral pad stiffness (total value) :math:`[N/m]`. Lateral value can be set to zero
        when lateral rail deflections are omitted.
    wdthp : float
        Pad width in x-direction :math:`[m]`.
    etap : float
        Pad loss factor :math:`[-]`.
    fresp : list[float], default=[0.0, 0.0]
        Vertical/lateral pad resonance frequencies [Hz]. This frequency is needed for calculating
        the viscous damping coefficient if it is not provided. Lateral value can be set to zero when
        lateral rail deflections are omitted.
    dp : list[float], default=[0.0, 0.0]
        Vertical/lateral pad damping coefficient (viscous) :math:`[Ns/m]`. Lateral value can be set
        to zero when lateral rail deflections are omitted.
    """

    sp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    wdthp: float = 0.0
    etap: float
    fresp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    dp: list[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass(kw_only=True)
class ContPad:
    r"""Cont Pad Class.

    Contains the properties of a continuous pad.

    Attributes
    ----------
    sp : list[float], default=[0.0, 0.0]
        Vertical/lateral pad stiffness (per meter) :math:`[N/m^2]`. Lateral value can be set to zero
        when lateral rail deflections are omitted.
    etap : float
        Pad loss factor :math:`[-]`.
    fresp : list[float], default=[0.0, 0.0]
        Vertical/lateral pad resonance frequencies :math:`[Hz]`. These frequencies are needed for
        calculating the viscous damping coefficients if they are not provided. Lateral value can be
        set to zero when lateral rail deflections are omitted.
    dp : list[float], default=[0.0, 0.0]
        Vertical/lateral viscous damping coefficient (per meter) :math:`[Ns/m^2]`. Lateral value can
        be set to zero when lateral rail deflections are omitted.
    """

    sp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    etap: float
    fresp: list[float] = field(default_factory=lambda: [0.0, 0.0])
    dp: list[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass(kw_only=True)
class Sleeper:
    r"""Sleeper Class.

    Contains the properties of a sleeper.

    Attributes
    ----------
    ms : float
        Sleeper mass :math:`[kg]`.
    Bs : float
        Sleeper bending stiffness :math:`[Nm^2]`.
    ls : float
        Sleeper length in y-direction :math:`[m]`.
    wdths : float
        Sleeper width in x-direction :math:`[m]`.
    """

    ms: float = 0.0
    Bs: float = 0.0
    ls: float = 0.0
    wdths: float = 0.0


@dataclass(kw_only=True)
class Slab:
    r"""Slab class.

    Contains the properties of the slab.

    Attributes
    ----------
    ms : float
        Slab mass per unit length :math:`[kg/m]`.
    ls : float
        Slab depth :math:`[m]`.
    """

    ms: float
    ls: float = 0.0


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
    sb : list[float], default=[0.0, 0.0]
        Vertical/lateral ballast stiffness (total value :math:`[N/m]` or per meter :math:`[N/m^2]`).
        Lateral value can be set to zero when lateral rail deflections are omitted.

    etab : float
        Ballast loss factor :math:`[-]`.
    fresb : list[float], default=[0.0, 0.0]
        Vertical/lateral ballast resonance frequencies :math:`[Hz]`. These frequencies are needed
        for calculating the viscous damping coefficients if they are not provided. Lateral value can
        be set to zero when lateral rail deflections are omitted.
    db : list[float], default=[0.0, 0.0]
        Vertical/lateral viscous damping coefficient (per meter) :math:`[Ns/m]`. Lateral value can
        be set to zero when lateral rail deflections are omitted.
    """

    sb: list[float] = field(default_factory=lambda: [0.0, 0.0])
    etab: float = 0.0
    fresb: list[float] = field(default_factory=lambda: [0.0, 0.0])
    db: list[float] = field(default_factory=lambda: [0.0, 0.0])


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
