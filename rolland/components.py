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

from traitlets import Float, HasTraits, List, Tuple, Unicode
from traittypes import Array


class Rail(HasTraits):
    r"""Rail Class.

    Defines a rail with certain properties. UIC 60 rail is usually used as default rail.

    .. hint:: A set of predefined rail instances is available in the `database` module.

    Attributes
    ----------
    rl_geo : list of tuple of float
        Rail outline coordinates :math:`[m]`. TODO: Define coordinate system!
    E : float
        Young's modulus of rail :math:`[Pa]`.
    G : float
        Shear modulus of rail :math:`[Pa]`.
    nu : float
        Poisson's ratio of rail :math:`[-]`.
    kap : list of float
        Timoshenko shear correction factor (vertical, lateral) :math:`[-]`.
    mr : float
        Rail mass per unit length :math:`[kg/m]`.
    rho : float
        Density of rail :math:`[kg/m^3]`.
    etar : float
        Rail loss factor :math:`[-]`.
    fresr : float
        Rail resonance frequency :math:`[Hz]`.
    dr : float
        Rail damping coefficient (viscous) :math:`[Ns/m]`.
    gamr : list of float
        Coordinates of rail shear center :math:`[m]`.
    epsr : list of float
        Coordinates of center of gravity :math:`[m]`.
    Iyr : float
        Area moment of inertia of rail around y-axis :math:`[m^4]`.
    Izr : float
        Area moment of inertia of rail around z-axis :math:`[m^4]`.
    Itr : float
        Torsional constant of rail :math:`[m^4]`.
    Ipr : float
        Polar moment of inertia of rail :math:`[m^4]`.
    Ar : float
        Cross-sectional area of rail :math:`[m^2]`.
    Asr : float
        Surface area per unit length of rail :math:`[m^2/m]`.
    Vr : float
        Volume per unit length of rail :math:`[m^3/m]`.
    """

    rl_geo = List(Tuple(Float(), Float()))
    E = Float()
    G = Float()
    nu = Float()
    kap = List(Float())
    mr = Float()
    rho = Float()
    etar = Float()
    fresr = Float()
    dr = Float()
    gamr = List(Float())
    epsr = List(Float())
    Iyr = Float()
    Izr = Float()
    Itr = Float()
    Ipr = Float()
    Ar = Float()
    Asr = Float()
    Vr = Float()


class RailRoughness(HasTraits):
    r"""Rail Roughness Class.

    Contains a rail roughness spectrum in frequency domain, which can later be used to calculate the
    rail roughness along the track.

    Attributes
    ----------
    r_rough : Tuple containing two lists of float
        Rail roughness spectrum :math:`[f, m]`.
    """

    r_rough = Tuple(List(Float()), List(Float()))


class DiscrPad(HasTraits):
    r"""Discrete Pad Class.

    Contains the properties of a discrete pad.

    Attributes
    ----------
    sp : list of float
        Vertical/lateral pad stiffness (total value) :math:`[N/m]`. Lateral value can be set to zero
        when lateral rail deflections are omitted.
    wdthp : float
        Pad width in x-direction :math:`[m]`.
    etap : float
        Pad loss factor :math:`[-]`.
    fresp : list of float
        Vertical/lateral pad resonance frequencies [Hz]. This frequency is needed for calculating
        the viscous damping coefficient if it is not provided. Lateral value can be set to zero when
        lateral rail deflections are omitted.
    dp : list of float
        Vertical/lateral pad damping coefficient (viscous) :math:`[Ns/m]`. Lateral value can be set
        to zero when lateral rail deflections are omitted.
    """

    sp = List(Float(), default_value=[0.0, 0.0], maxlen=2)
    wdthp = Float()
    etap = Float()
    fresp = List(Float(), default_value=[0.0, 0.0], maxlen=2)
    dp = List(Float(), default_value=[0.0, 0.0], maxlen=2)


class ContPad(HasTraits):
    r"""Cont Pad Class.

    Contains the properties of a continuous pad.

    Attributes
    ----------
    sp : list of float
        Vertical/lateral pad stiffness (per meter) :math:`[N/m^2]`. Lateral value can be set to zero
        when lateral rail deflections are omitted.
    etap : float
        Pad loss factor :math:`[-]`.
    fresp : list of float
        Vertical/lateral pad resonance frequencies :math:`[Hz]`. These frequencies are needed for
        calculating the viscous damping coefficients if they are not provided. Lateral value can be
        set to zero when lateral rail deflections are omitted.
    dp : list of float
        Vertical/lateral viscous damping coefficient (per meter) :math:`[Ns/m^2]`. Lateral value can
        be set to zero when lateral rail deflections are omitted.
    """

    sp = List(Float(), default_value=[0.0, 0.0], maxlen=2)
    etap = Float()
    fresp = List(Float(), default_value=[0.0, 0.0], maxlen=2)
    dp = List(Float(), default_value=[0.0, 0.0], maxlen=2)


class Sleeper(HasTraits):
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

    ms = Float()
    Bs = Float()
    ls = Float()
    wdths = Float()


class Slab(HasTraits):
    r"""Slab class."""

    # Slab mass per unit length [kg/m]
    ms = Float()

    # Slab depth [m]
    ls = Float()


class Ballast(HasTraits):
    r"""Ballast Class.

    Contains the properties of the ballast.

    .. caution::
        Properties of the ballast can either be defined as discrete values acting at the
        mounting positions or as continuous values acting per meter depending. The values need to be
        chosen accordingly to the track type.

    Attributes
    ----------
    sb : list of float
        Vertical/lateral ballast stiffness (total value :math:`[N/m]` or per meter :math:`[N/m^2]`).
        Lateral value can be set to zero when lateral rail deflections are omitted.

    etab : float
        Ballast loss factor :math:`[-]`.
    fresb : list of float
        Vertical/lateral ballast resonance frequencies :math:`[Hz]`. These frequencies are needed
        for calculating the viscous damping coefficients if they are not provided. Lateral value can
        be set to zero when lateral rail deflections are omitted.
    db : list of float
        Vertical/lateral viscous damping coefficient (per meter) :math:`[Ns/m]`. Lateral value can
        be set to zero when lateral rail deflections are omitted.
    """

    sb = List(Float(), default_value=[0.0, 0.0], maxlen=2)
    etab = Float()
    fresb = List(Float(), default_value=[0.0, 0.0], maxlen=2)
    db = List(Float(), default_value=[0.0, 0.0], maxlen=2)


class Wheel(HasTraits):
    r"""Wheel Class.

    Contains the properties of a wheel.

    Attributes
    ----------
    w_geo_cross_sec : list of tuple of float
        Wheel cross-sectional geometry coordinates (y-z plane) :math:`[m]`.
        TODO: Define coordinate system
    w_prof : str
        Wheel running surface profile.
    w_geo : list of tuple of float
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

    w_geo_cross_sec = List(Tuple(Float(), Float()))
    w_prof = Unicode()
    w_geo = List(Tuple(Float(), Float()))
    mw = Float()
    mw_red = Float()
    rw = Float()


class WheelRoughness(HasTraits):
    r"""Wheel Roughness Class.

    Contains a wheel roughness spectrum in frequency domain.

    Attributes
    ----------
    w_rough : tuple of two lists of float
        Wheel roughness spectrum :math:`[f, m]`.
    """

    w_rough = Tuple(List(Float()), List(Float()))


class WheelGreensfunc(HasTraits):
    r"""Wheel Greens Function Class.

    Contains the Green's function of a wheel.

    Attributes
    ----------
    w_gf : numpy.ndarray
        Green's function data. Contains the response of the wheel to a unit impulse at
        multiple points :math:`[m/N]`.
    w_gf_freq : numpy.ndarray
        Frequency values of the Green's function :math:`[Hz]`.
    """

    w_gf = Array()
    w_gf_freq = Array()
