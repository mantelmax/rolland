"""Defines track structure and arrangement.

.. autosummary::
    :toctree: track

    Track
    SingleRailTrack
    SlabSingleRailTrack
    ContSlabSingleRailTrack
    DiscrSlabSingleRailTrack
    SimplePeriodicSlabSingleRailTrack
    ArrangedSlabSingleRailTrack
    BallastedSingleRailTrack
    ContBallastedSingleRailTrack
    DiscrBallastedSingleRailTrack
    SimplePeriodicBallastedSingleRailTrack
    ArrangedBallastedSingleRailTrack
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal

from numpy import ndarray, pi, piecewise
from scipy.interpolate import interp1d

from rolland.helper.build_matrix import (
    build_equ_sleeper_matrix,
    build_fnd_stiff_matrix,
    build_pad_ballast_stiff_matrices,
    build_rail_matrices,
    build_sleep_mass_matrix,
    build_transfm_matrices,
    calc_cut_on_frequ,
)

from .arrangement import Arrangement
from .components import Ballast, ContPad, DiscrPad, Rail, Slab, Sleeper


class Track(ABC):
    r"""Abstract base class for track classes."""

    @abstractmethod
    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""

@dataclass(kw_only=True)
class SingleRailTrack(Track):
    r"""Abstract base class for single rail track classes.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    """

    rail: Rail

    def _needs_viscous_damping(self, pad=None, ballast=None) -> bool:
        """Return True when any damping component still needs eta-to-viscous conversion."""
        p = pad if pad is not None else getattr(self, "pad", None)
        b = ballast if ballast is not None else getattr(self, "ballast", None)

        components = [c for c in (p, b) if c is not None]
        return any(c.damping_mode == "hysteretic" for c in components)

    def calc_pad_warping_stiffn(self, pad=None):
        """Calculate the pad's foundation resistance to rail warping.

        Based on the vertical eccentricity from the rail shear center.
        """
        p = pad if pad is not None else self.pad
        e_s = self.z_f - self.rail.shearc[1]
        p.sp_w = (self.rail.k_w * e_s) ** 2 * p.wdthp**2 / 12 * p.sp_y

    def calc_pad_viscous_damp_cuton(self, pad=None, cof=None):
        """Calculate coupled viscous damping coefficients from cut-on frequencies when needed."""
        p = pad if pad is not None else self.pad
        c = cof if cof is not None else self.cof
        if p.damping_mode != "hysteretic":
            return

        if c is None:
            msg = "Cut-on frequencies are required to derive viscous pad damping from loss factors."
            raise ValueError(msg)

        p.dp_x = p.etap_x * p.sp_x / (c[0] * (2 * pi))
        p.dp_z = p.etap_z * p.sp_z / (c[1] * (2 * pi))
        p.dp_y = p.etap_y * p.sp_y / (c[3] * (2 * pi))
        p.dp_xr = p.etap_r * p.sp_xr / (c[2] * (2 * pi))

    def get_mount_patterns(self, x, dx, mp):
        """Return the mounting patterns for each position in a dictionary."""
        patterns = {}
        for pos in mp:
            pad = self.mount_prop[pos][0] if hasattr(self, 'mount_prop') and pos in self.mount_prop else self.pad
            start, end = pos - pad.wdthp / 2, pos + pad.wdthp / 2
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
            integral = sum(pattern) * dx
            patterns[pos] = pattern / integral
        return patterns

    def interpol_pad_width(self, x, dx, mp):
        """Interpolated pad width distribution along track."""
        patterns = self.get_mount_patterns(x, dx, mp)
        return sum(patterns.values())

    def calc_equiv_sleeper_factors(self, sleeper=None):
        """Calculate equivalent sleeper factors.

        According to :cite:p:`kostovasilis_analytical_2017`.
        """
        s = sleeper if sleeper is not None else self.sleeper
        if not s.equi_sm:
            pass
        else:
            s.f_z = 1 + 12 * s.y_sc**2 /(s.lengs**2 + s.heights**2)
            s.f_x = 1 + 12 * s.y_sc**2 / (s.lengs**2 + s.wdths**2)

    def calc_equiv_slab_factors(self, slab=None):
        """Calculate equivalent slab factors according to :cite:p:`kostovasilis_analytical_2017`."""
        s = slab if slab is not None else self.slab
        if not s.equi_sm:
            pass

        else:
            s.f_z = 1 + 12 * s.y_sc ** 2 / (s.lengs ** 2 + s.heights ** 2)
            s.f_x = 1 + 12 * s.y_sc ** 2 / (s.lengs ** 2 + s.equ_wdths ** 2)

    def calc_ballast_rotational_stiffn(self, sleeper=None, slab=None, ballast=None):
        """Calculate rotational stiffnesses from ballast and sleeper dimensions.

        Calculated according to :cite:p:`thompson_developments_1998`.
        """
        b = ballast if ballast is not None else self.ballast
        if sleeper is not None or hasattr(self, 'sleeper'):
            s = sleeper if sleeper is not None else self.sleeper
            s_l = s.lengs
            s_w = s.wdths
        if slab is not None or hasattr(self, 'slab'):
            s = slab if slab is not None else self.slab
            s_l = s.lengs
            s_w = s.equ_wdths

        b.sb_xr = s_l**2 / 12 * b.sb_z
        b.sb_yr = s_w**2 / 12 * b.sb_z
        b.sb_zr = s_l**2 / 12 * b.sb_x + s_w**2 / 12 * b.sb_y

    def calc_ballast_viscous_damp_cuton(self, ballast=None, cof=None):
        """Calculate coupled viscous damping coefficients from cut-on frequencies when needed."""
        b = ballast if ballast is not None else self.ballast
        c = cof if cof is not None else self.cof
        if b.damping_mode != "hysteretic":
            return

        if c is None:
            msg = "Cut-on frequencies are required to derive viscous ballast damping from loss factors."
            raise ValueError(msg)

        b.db_x = b.etab_x * b.sb_x / (c[7] * (2 * pi))
        b.db_z = b.etab_z * b.sb_z / (c[8] * (2 * pi))
        b.db_y = b.etab_y * b.sb_y / (c[9] * (2 * pi))
        b.db_xr = b.etab_r * b.sb_xr / (c[10] * (2 * pi))
        b.db_yr = b.etab_r * b.sb_yr / (c[11] * (2 * pi))
        b.db_zr = b.etab_r * b.sb_zr / (c[12] * (2 * pi))


@dataclass(kw_only=True)
class SlabSingleRailTrack(SingleRailTrack):
    r"""Abstract base class for slab single rail track classes.

    Slab mass is set to a very large number to avoid displacement and simulate a rigid slab.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab, default=Slab(ms=1e20)
        Slab placeholder instance.
    """

    slab: Slab = field(
        default_factory=lambda: Slab(
            ms=1e20,
            Is_z=1e20,
            Is_y=1e20,
            Is_x=1e20,
            rhos=1e20,
            lengs=1e20,
            equ_wdths=1e20,
            heights=1e20,
            z_st=0,
            z_sb=0,
        ),
        metadata={"default_repr": "Slab(ms=1e20)"},
    )

@dataclass(kw_only=True)
class ContSlabSingleRailTrack(SlabSingleRailTrack):
    r"""Single rail slab track with continuous support.

    All superstructure properties are continuous along the track. The slab is assumed to be rigid.
    Assigning slab properties is not required.

    +------------------+-----------+--------------------+-------------+
    | Layer of Support | Component | Condition          | Variability |
    +==================+===========+====================+=============+
    | /                | rail      | continuous         | no          |
    +------------------+-----------+--------------------+-------------+
    | 1st              | pads      | continuous         | no          |
    +------------------+-----------+--------------------+-------------+
    | 1st/2nd          | slab      | continuous (rigid) | no          |
    +------------------+-----------+--------------------+-------------+
    | 2nd              | /         | /                  | /           |
    +------------------+-----------+--------------------+-------------+


    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab placeholder instance.
    pad : ContPad
        Continuous pad instance.
    l_track : float, default=100.0
        Track length :math:`[m]`. (May change slightly after discretization.
        The inclusion of boundary and calculation domain is required).
    cof : ndarray
        Cut-on frequencies corresponding to the DOFs :math:`[Hz]`.
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.
    chi_yz: tuple[float, float], optional
        Coordinates :math:`(y, z)` at which the warping function of the rail is evaluated
        :math:`[m]`. Defaults to :math:`(y_f, z_f)`.
    chi_f: float
        Warping function of the rail at ``chi_yz`` :math:`[m^2]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import ContPad
    >>> from rolland.track import ContSlabSingleRailTrack

    >>> thepad = ContPad(
    ...        sp_z=200e6,
    ...        sp_y=66.7e6,
    ...        sp_x=66.7e6,
    ...        etap_z=0.2,
    ...        etap_y=0.2,
    ...        etap_x=0.2,
    ...        etap_r=0.2,
    ...        wdthp=0.15
    ...    )
    >>> track = ContSlabSingleRailTrack(rail=rail_60E1, pad=thepad, l_track=100, z_f=0.081, y_f=0.0)
    """

    pad: ContPad
    l_track: float = 100.0
    cof: ndarray | None = field(init=False, default=None)
    z_f: float
    y_f: float
    chi_yz: tuple[float, float] | None = None
    chi_f: float = field(init=False, default=0.0)

    def __post_init__(self):
        """post_init method to calculate derived properties after initialization."""
        self.calc_pad_warping_stiffn()
        self.calc_equiv_slab_factors()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        self.chi_f = self.rail.chi_at(*(self.chi_yz or (self.y_f, self.z_f)))
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.z_f,
            self.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.chi_f,
            )
        self.E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, self.E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", self.E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        self.cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806
        if self._needs_viscous_damping():
            self.calc_pad_viscous_damp_cuton()

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class DiscrSlabSingleRailTrack(SlabSingleRailTrack):
    r"""Abstract base class for discrete slab single rail track classes.

    The pad and sleeper properties are discrete. The slab is assumed to be rigid.
    Assigning slab properties is not required.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab placeholder instance.
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, None, None)).
    """

    def __repr__(self):
        """Represent mounting properties as string."""
        st = ""
        for x in sorted(self.mount_prop.keys()):
            p, s, b= self.mount_prop[x]
            st += f'{x}, {p.sp}, {s.ms}, {b.sb} \n'
        return st


@dataclass(kw_only=True)
class SimplePeriodicSlabSingleRailTrack(DiscrSlabSingleRailTrack):
    r"""Single rail slab track with simple periodic support.

    All mounting properties are uniform and no variation is allowed.
    The slab is assumed to be rigid. Assigning slab properties is not required.

    +---------+-----------+------------------+-------------+
    | Layer   | Component | Condition        | Variability |
    +=========+===========+==================+=============+
    | /       | rail      | continuous       | no          |
    +---------+-----------+------------------+-------------+
    | 1st     | pads      | discrete         | no          |
    +---------+-----------+------------------+-------------+
    | 1st/2nd | slab      | discrete (rigid) | no          |
    +---------+-----------+------------------+-------------+
    | 2nd     | /         | /                | /           |
    +---------+-----------+------------------+-------------+

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab placeholder instance.
    pad : DiscrPad
        Discrete pad instance.
    distance : float, default=0.6
        Distance between mounting positions.
    num_mount : int, default=100
        Number of mounting positions.
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, None, None)).
    l_track : float
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    cof : ndarray
        Cut-on frequencies corresponding to the DOFs :math:`[Hz]`.
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.
    chi_yz: tuple[float, float], optional
        Coordinates :math:`(y, z)` at which the warping function of the rail is evaluated
        :math:`[m]`. Defaults to :math:`(y_f, z_f)`.
    chi_f: float
        Warping function of the rail at ``chi_yz`` :math:`[m^2]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import DiscrPad
    >>> from rolland.track import SimplePeriodicSlabSingleRailTrack

    >>> thepad = DiscrPad(
    ...     sp_z=120e6,
    ...     sp_y=40e6,
    ...     sp_x=40e6,
    ...     etap_z=0.2,
    ...     etap_y=0.2,
    ...     etap_x=0.2,
    ...     etap_r=0.2,
    ...     wdthp=0.15
    ...     )
    >>> track = SimplePeriodicSlabSingleRailTrack(rail=rail_60E1,
    ...     pad=thepad,
    ...     distance=0.6,
    ...     num_mount=167,
    ...     z_f=0.081,
    ...     y_f=0.0
    ...     )
    """

    pad: DiscrPad
    distance: float = 0.6
    num_mount: int = 100
    cof: ndarray | None = field(init=False, default=None)
    z_f: float
    y_f: float
    chi_yz: tuple[float, float] | None = None
    chi_f: float = field(init=False, default=0.0)

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

        self.calc_pad_warping_stiffn()
        self.calc_equiv_slab_factors()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        self.chi_f = self.rail.chi_at(*(self.chi_yz or (self.y_f, self.z_f)))
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.z_f,
            self.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.chi_f,
            )
        self.E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, self.E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", self.E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        self.cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806
        if self._needs_viscous_damping():
            self.calc_pad_viscous_damp_cuton()

    def calc_mount_prop(self, change=None):
        """Calculate the mounting properties."""
        self.mount_prop = {}
        for _i in range(self.num_mount):
            x = float(Decimal(str(_i)) * Decimal(str(self.distance)))
            self.mount_prop[x] = (self.pad, None, None)
        self.l_track = max(self.mount_prop.keys())

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class ArrangedSlabSingleRailTrack(DiscrSlabSingleRailTrack):
    """Single rail slab track with varying support properties.

    Variations in the form of periodically or stochastically varying mounting properties are
    allowed. Slab is assumed to be rigid.

    +---------+-----------+------------------+---------------------+
    | Layer   | Component | Condition        | Variability         |
    +=========+===========+==================+=====================+
    | /       | rail      | continuous       | no                  |
    +---------+-----------+------------------+---------------------+
    | 1st     | pads      | discrete         | periodic/stochastic |
    +---------+-----------+------------------+---------------------+
    | 1st/2nd | slab      | discrete (rigid) | periodic/stochastic |
    +---------+-----------+------------------+---------------------+
    | 2nd     | /         | /                | /                   |
    +---------+-----------+------------------+---------------------+

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab placeholder instance.
    pad : Arrangement
        Arrangement instance containing multiple pads.
    distance : Arrangement
        Arrangement instance containing multiple distances.
    num_mount : int, default=100
        Number of mounting positions.
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, None, None)).
    l_track : float
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    cof : dict[float, ndarray]
        Cut-on frequencies corresponding to the DOFs :math:`[Hz]`.
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.
    chi_yz: tuple[float, float], optional
        Coordinates :math:`(y, z)` at which the warping function of the rail is evaluated
        :math:`[m]`. Defaults to :math:`(y_f, z_f)`.
    chi_f: float
        Warping function of the rail at ``chi_yz`` :math:`[m^2]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import DiscrPad
    >>> from rolland.arrangement import PeriodicArrangement
    >>> from rolland.track import ArrangedSlabSingleRailTrack

    >>> thepadA = DiscrPad(
    ...     sp_z=120e6,     # Low vertical stiffness
    ...     sp_y=40e6,
    ...     sp_x=40e6,
    ...     etap_z=0.2,
    ...     etap_y=0.2,
    ...     etap_x=0.2,
    ...     etap_r=0.2,
    ...     wdthp=0.15
    ...     )
    >>> thepadB = DiscrPad(
    ...     sp_z=700e6,     # High vertical stiffness
    ...     sp_y=40e6,
    ...     sp_x=40e6,
    ...     etap_z=0.2,
    ...     etap_y=0.2,
    ...     etap_x=0.2,
    ...     etap_r=0.2,
    ...     wdthp=0.15
    ...     )
    >>> pads = PeriodicArrangement(item=[thepadA, thepadB])
    >>> distances = PeriodicArrangement(item=[0.65, 0.5]) # Varying sleeper spacings
    >>> track = ArrangedSlabSingleRailTrack(
    ...     rail=rail_60E1,
    ...     pad=pads,
    ...     distance=distances,
    ...     num_mount=167,
    ...     z_f=0.081,
    ...     y_f=0.0
    ...     )
    """

    pad: Arrangement
    distance: Arrangement
    num_mount: int = 100
    cof: ndarray | None = field(init=False, default=None)
    z_f: float
    y_f: float
    chi_yz: tuple[float, float] | None = None
    chi_f: float = field(init=False, default=0.0)

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()
        self.cof = {}

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        self.chi_f = self.rail.chi_at(*(self.chi_yz or (self.y_f, self.z_f)))

        for x, (pad, _slab_unused, _ballast_unused) in self.mount_prop.items():
            self.calc_pad_warping_stiffn(pad=pad)
            self.calc_equiv_slab_factors(slab=self.slab)

            Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
                self.z_f,
                self.y_f,
                self.slab.z_st,
                self.slab.z_sb,
                self.chi_f,
                )
            self.E = build_equ_sleeper_matrix(self, seclay=self.slab) # noqa: N806
            Ms = build_sleep_mass_matrix(self, self.E, seclay=self.slab) # noqa: N806
            Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", self.E, pad=pad) # noqa: N806
            K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
            cof_x = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806
            self.cof[x] = cof_x
            if self._needs_viscous_damping(pad=pad):
                self.calc_pad_viscous_damp_cuton(pad=pad, cof=cof_x)

    def calc_mount_prop(self, change=None):
        """Calculate the mounting properties."""
        x = Decimal(str(0))
        self.mount_prop = {}
        for p, d in zip(self.pad.generate(self.num_mount),
                        self.distance.generate(self.num_mount), strict=False):
            self.mount_prop[float(Decimal(str(x)))] = (p, None, None)
            x += Decimal(str(d))
        self.l_track = max(self.mount_prop.keys())

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class BallastedSingleRailTrack(SingleRailTrack):
    r"""Abstract base class for ballasted single rail track classes.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    ballast : Ballast
        Ballast instance.
    """

    ballast: Ballast


@dataclass(kw_only=True)
class ContBallastedSingleRailTrack(BallastedSingleRailTrack):
    r"""Single rail slab track with ballasted support.

    All superstructure properties are continuous along the track.

    .. note:: Properties of ballast need to be defined as continuous values (per meter).

    +---------+-----------+------------+-------------+
    | Layer   | Component | Condition  | Variability |
    +=========+===========+============+=============+
    | /       | rail      | continuous | no          |
    +---------+-----------+------------+-------------+
    | 1st     | pads      | continuous | no          |
    +---------+-----------+------------+-------------+
    | 1st/2nd | slab      | continuous | no          |
    +---------+-----------+------------+-------------+
    | 2nd     | ballast   | continuous | no          |
    +---------+-----------+------------+-------------+

    Attributes
    ----------
    rail : Rail
        Rail instance.
    pad : ContPad
        Continuous pad instance.
    slab : Slab
        Slab instance.
    ballast : Ballast
        Ballast instance.
    l_track : float, default=100.0
        Track length :math:`[m]`. (May change slightly after discretization.
        The inclusion of boundary and interior domain is required).
    cof : ndarray
        Cut-on frequencies corresponding to the DOFs :math:`[Hz]`.
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.
    chi_yz: tuple[float, float], optional
        Coordinates :math:`(y, z)` at which the warping function of the rail is evaluated
        :math:`[m]`. Defaults to :math:`(y_f, z_f)`.
    chi_f: float
        Warping function of the rail at ``chi_yz`` :math:`[m^2]`.

    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import ContPad, Slab
    >>> from rolland.track import ContBallastedSingleRailTrack

    >>> thepad = ContPad(
    ...        sp_z=200e6,
    ...        sp_y=66.7e6,
    ...        sp_x=66.7e6,
    ...        etap_z=0.2,
    ...        etap_y=0.2,
    ...        etap_x=0.2,
    ...        etap_r=0.2,
    ...        wdthp=0.15
    ...    )
    >>> theslab = Slab(
    ...        ms=300/0.6,
    ...        Is_x=0.05934/0.6,
    ...        Is_y=0.00089/0.6,
    ...        Is_z=0.05934/0.6,
    ...        lengs=2.5,
    ...        rhos=2648,
    ...        equ_wdths=0.245,
    ...        heights=0.185,
    ...        z_st=-0.185/2,
    ...        z_sb=0.185/2,
    ...    )
    >>> theballast = Ballast(
    ...        sb_z=200e6,
    ...        sb_y=200e6,
    ...        sb_x=200e6,
    ...        etab_z=1,
    ...        etab_y=2,
    ...        etab_x=2,
    ...        etab_r=2,
    ...        )
    >>> track = ContBallastedSingleRailTrack(
    ...     rail=rail_60E1, pad=thepad, slab=theslab, ballast=theballast, z_f=0.081, y_f=0.0)
    """

    pad: ContPad
    slab: Slab
    l_track: float = 100.0
    cof: ndarray | None = field(init=False, default=None)
    z_f: float
    y_f: float
    chi_yz: tuple[float, float] | None = None
    chi_f: float = field(init=False, default=0.0)

    def __post_init__(self):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_pad_warping_stiffn()
        self.calc_equiv_slab_factors()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        self.chi_f = self.rail.chi_at(*(self.chi_yz or (self.y_f, self.z_f)))
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.z_f,
            self.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.chi_f,
            )
        self.E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, self.E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", self.E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        self.cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806
        if self._needs_viscous_damping():
            self.calc_pad_viscous_damp_cuton()
            self.calc_ballast_viscous_damp_cuton()

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class DiscrBallastedSingleRailTrack(BallastedSingleRailTrack):
    """Abstract base class for discrete ballasted single rail track classes.

    .. note:: The pad, sleeper and ballast properties are discrete.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    ballast : Ballast
        Ballast instance.
    mount_prop : dict[float, tuple[DiscrPad, Sleeper, Ballast]]
        Dictionary for discrete mounting positions (x-> (Pad, Sleeper, Ballast)).
    """

    def __repr__(self):
        """Represent mounting properties as string."""
        st = ""
        for x in sorted(self.mount_prop.keys()):
            p, s, b = self.mount_prop[x]
            st += f'{x}, {p.sp}, {s.ms}, {b.sb} \n'
        return st


@dataclass(kw_only=True)
class SimplePeriodicBallastedSingleRailTrack(DiscrBallastedSingleRailTrack):
    """Single rail ballasted track with simple periodic support.

    All mounting properties are uniform and no variation is allowed.

    .. note:: Properties of ballast need to be defined as discrete values.


    +---------+-----------+------------+-------------+
    | Layer   | Component | Condition  | Variability |
    +=========+===========+============+=============+
    | /       | rail      | continuous | no          |
    +---------+-----------+------------+-------------+
    | 1st     | pads      | discrete   | no          |
    +---------+-----------+------------+-------------+
    | 1st/2nd | sleeper   | discrete   | no          |
    +---------+-----------+------------+-------------+
    | 2nd     | ballast   | discrete   | no          |
    +---------+-----------+------------+-------------+

    Attributes
    ----------
    rail : Rail
        Rail instance.
    ballast : Ballast
        Ballast instance.
    pad : DiscrPad
        Discrete pad instance.
    sleeper : Instance of :class:`~rolland.components.sleeper` class
        Sleeper instance.
    distance : float, default=0.6
        Distance between mounting positions.
    num_mount : int, default=100
        Number of mounting positions.
    mount_prop : dict[float, tuple[DiscrPad, Sleeper, Ballast]]
        Dictionary for discrete mounting positions (x-> (Pad, Sleeper, Ballast)).
    l_track : float
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    cof : ndarray
        Cut-on frequencies corresponding to the DOFs :math:`[Hz]`.
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.
    chi_yz: tuple[float, float], optional
        Coordinates :math:`(y, z)` at which the warping function of the rail is evaluated
        :math:`[m]`. Defaults to :math:`(y_f, z_f)`.
    chi_f: float
        Warping function of the rail at ``chi_yz`` :math:`[m^2]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import DiscrPad, Sleeper
    >>> from rolland.track import SimplePeriodicBallastedSingleRailTrack

    >>> thepad = DiscrPad(
    ...        sp_z=120e6,
    ...        sp_y=40e6,
    ...        sp_x=40e6,
    ...        etap_z=0.2,
    ...        etap_y=0.2,
    ...        etap_x=0.2,
    ...        etap_r=0.2,
    ...        wdthp=0.15
    ...    )
    >>> thesleeper = Sleeper(
    ...        ms=300,
    ...        Is_x=0.05934,
    ...        Is_y=0.00089,
    ...        Is_z=0.05934,
    ...        lengs=2.5,
    ...        rhos=2648,
    ...        equ_wdths=0.245,
    ...        heights=0.185,
    ...        z_st=-0.185/2,
    ...        z_sb=0.185/2,
    ...    )
    >>> theballast = Ballast(
    ...        sb_z=120e6,
    ...        sb_y=120e6,
    ...        sb_x=120e6,
    ...        etab_z=1,
    ...        etab_y=2,
    ...        etab_x=2,
    ...        etab_r=2,
    ...        )
    >>> tr = SimplePeriodicBallastedSingleRailTrack(
    ...     rail=rail_60E1,
    ...     pad=thepad,
    ...     sleeper=thesleeper,
    ...     ballast=theballast,
    ...     distance=0.6,
    ...     z_f=0.081,
    ...     y_f=0.0)
    """

    sleeper: Sleeper
    pad: DiscrPad
    distance: float = 0.6
    num_mount: int = 100
    cof: ndarray | None = field(init=False, default=None)
    z_f: float
    y_f: float
    chi_yz: tuple[float, float] | None = None
    chi_f: float = field(init=False, default=0.0)

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

        self.calc_pad_warping_stiffn()
        self.calc_equiv_sleeper_factors()
        self.calc_ballast_rotational_stiffn()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        self.chi_f = self.rail.chi_at(*(self.chi_yz or (self.y_f, self.z_f)))
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.z_f,
            self.y_f,
            self.sleeper.z_st,
            self.sleeper.z_sb,
            self.chi_f,
            )
        self.E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, self.E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", self.E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        self.cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806
        if self._needs_viscous_damping():
            self.calc_pad_viscous_damp_cuton()
            self.calc_ballast_viscous_damp_cuton()

    def calc_mount_prop(self, change=None):
        """Calculate the mounting properties."""
        self.mount_prop = {}
        for _i in range(self.num_mount):
            # Calculate the mounting position
            # Use Decimal to avoid floating-point representation errors
            x = float(Decimal(str(_i)) * Decimal(str(self.distance)))
            self.mount_prop[x] = (self.pad, self.sleeper, self.ballast)
        self.l_track = max(self.mount_prop.keys())

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class ArrangedBallastedSingleRailTrack(DiscrBallastedSingleRailTrack):
    """Single rail ballasted track with varying support properties.

    Variations in the form of periodically or stochastically varying mounting properties are
    allowed.

    .. note:: Properties of ballast need to be defined as discrete values.

    +---------+-----------+------------+---------------------+
    | Layer   | Component | Condition  | Variability         |
    +=========+===========+============+=====================+
    | /       | rail      | continuous | no                  |
    +---------+-----------+------------+---------------------+
    | 1st     | pads      | discrete   | periodic/stochastic |
    +---------+-----------+------------+---------------------+
    | 1st/2nd | sleepers  | discrete   | periodic/stochastic |
    +---------+-----------+------------+---------------------+
    | 2nd     | ballast   | discrete   | no                  |
    +---------+-----------+------------+---------------------+

    Attributes
    ----------
    rail : Rail
        Rail instance.
    ballast : Arrangement
        Ballast instance.
    pad : Arrangement
        Arrangement instance containing multiple pads.
    sleeper : Arrangement
        Arrangement instance containing multiple sleepers.
    distance : Arrangement
        Arrangement instance containing multiple distances.
    num_mount : int, default=100
        Number of mounting positions.
    mount_prop : dict[float, tuple[DiscrPad, Sleeper, Ballast]]
        Dictionary for discrete mounting positions (x-> (Pad, Sleeper, Ballast)).
    l_track : float
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    cof : dict[float, ndarray]
        Cut-on frequencies corresponding to the DOFs :math:`[Hz]`.
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.
    chi_yz: tuple[float, float], optional
        Coordinates :math:`(y, z)` at which the warping function of the rail is evaluated
        :math:`[m]`. Defaults to :math:`(y_f, z_f)`.
    chi_f: float
        Warping function of the rail at ``chi_yz`` :math:`[m^2]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import rail_60E1
    >>> from rolland.components import DiscrPad, Sleeper
    >>> from rolland.arrangement import PeriodicArrangement
    >>> from rolland.track import ArrangedBallastedSingleRailTrack

    >>> thepadA = DiscrPad(
    ...        sp_z=120e6,      # Low vertical stiffness
    ...        sp_y=40e6,
    ...        sp_x=40e6,
    ...        etap_z=0.2,
    ...        etap_y=0.2,
    ...        etap_x=0.2,
    ...        etap_r=0.2,
    ...        wdthp=0.15
    ...    )
    >>> thepadB = DiscrPad(
    ...        sp_z=700e6,      # High vertical stiffness
    ...        sp_y=40e6,
    ...        sp_x=40e6,
    ...        etap_z=0.2,
    ...        etap_y=0.2,
    ...        etap_x=0.2,
    ...        etap_r=0.2,
    ...        wdthp=0.15
    ...    )
    >>> thesleeperA = Sleeper(
    ...        ms=300/2,        # Low mass
    ...        Is_x=0.05934/2,
    ...        Is_y=0.00089/2,
    ...        Is_z=0.05934/2,
    ...        lengs=2.5,
    ...        rhos=2648,
    ...        equ_wdths=0.245,
    ...        heights=0.185,
    ...        z_st=-0.185/2,
    ...        z_sb=0.185/2,
    ...    )
    >>> thesleeperB = Sleeper(
    ...        ms=300*2,        # High mass
    ...        Is_x=0.05934*2,
    ...        Is_y=0.00089*2,
    ...        Is_z=0.05934*2,
    ...        lengs=2.5,
    ...        rhos=2648,
    ...        equ_wdths=0.245,
    ...        heights=0.185,
    ...        z_st=-0.185/2,
    ...        z_sb=0.185/2,
    ...    )
    >>> theballast = Ballast(
    ...        sb_z=120e6,
    ...        sb_y=120e6,
    ...        sb_x=120e6,
    ...        etab_z=1,
    ...        etab_y=2,
    ...        etab_x=2,
    ...        etab_r=2,
    ...        )
    >>> pads = PeriodicArrangement(item=[thepadA, thepadB])
    >>> distances = PeriodicArrangement(item=[0.65, 0.5])    #Varying sleeper spacings
    >>> sleepers = PeriodicArrangement(item=[thesleeperA, thesleeperB])
    >>> track = ArrangedBallastedSingleRailTrack(
    ...     rail=rail_60E1,
    ...     pad=pads,
    ...     sleeper=sleepers,
    ...     ballast=theballast,
    ...     distance=distances,
    ...     num_mount=167,
    ...     z_f=0.081,
    ...     y_f=0.0)
    """

    sleeper: Arrangement
    pad: Arrangement
    ballast: Arrangement
    distance: Arrangement
    num_mount: int = 100
    cof: ndarray | None = field(init=False, default=None)
    z_f: float
    y_f: float
    chi_yz: tuple[float, float] | None = None
    chi_f: float = field(init=False, default=0.0)

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()
        self.cof = {}

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        self.chi_f = self.rail.chi_at(*(self.chi_yz or (self.y_f, self.z_f)))

        for x, (pad, sleeper, ballast) in self.mount_prop.items():
            self.calc_pad_warping_stiffn(pad=pad)
            self.calc_equiv_sleeper_factors(sleeper=sleeper)
            self.calc_ballast_rotational_stiffn(sleeper=sleeper, ballast=ballast)

            Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
                self.z_f,
                self.y_f,
                sleeper.z_st,
                sleeper.z_sb,
                self.chi_f,
                )
            self.E = build_equ_sleeper_matrix(self, seclay=sleeper) # noqa: N806
            Ms = build_sleep_mass_matrix(self, self.E, seclay=sleeper) # noqa: N806
            Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", self.E, pad=pad, ballast=ballast) # noqa: N806
            K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
            cof_x = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806
            self.cof[x] = cof_x

            if self._needs_viscous_damping(pad=pad, ballast=ballast):
                self.calc_pad_viscous_damp_cuton(pad=pad, cof=cof_x)
                self.calc_ballast_viscous_damp_cuton(ballast=ballast, cof=cof_x)

    def calc_mount_prop(self, change=None):
        """Calculate the mounting properties."""
        x = Decimal(str(0))
        self.mount_prop = {}
        for s, p, b, d in zip(self.sleeper.generate(self.num_mount),
                           self.pad.generate(self.num_mount),
                           self.ballast.generate(self.num_mount),
                           self.distance.generate(self.num_mount), strict=False):
            self.mount_prop[float(Decimal(str(x)))] = (p, s, b)
            x += Decimal(str(d))
        self.l_track = max(self.mount_prop.keys())

    def _abstract(self) -> None:
        pass
