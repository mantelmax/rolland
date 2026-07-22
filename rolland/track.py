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

from numpy import pi, piecewise
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

    def calc_pad_warping_stiffn(self):
        """Calculate warping stiffness."""
        e_s = self.rail.shearc[1] - self.rail.z_f
        self.pad.sp_w = (self.rail.k_w * e_s) ** 2 * self.pad.wdthp**2 / 12 * self.pad.sp_y

    def calc_pad_viscous_damp_cuton(self):
        """Calculate coupled viscous damping coefficients based on cut on frequencies."""
        self.pad.dp_x = self.pad.etap_x * self.pad.sp_x / (self.pad.cof[0] * (2 * pi))
        self.pad.dp_z = self.pad.etap_z * self.pad.sp_z / (self.pad.cof[1] * (2 * pi))
        self.pad.dp_y = self.pad.etap_y * self.pad.sp_y / (self.pad.cof[3] * (2 * pi))
        self.pad.dp_xr = self.pad.etap_r * self.pad.sp_xr / (self.pad.cof[2] * (2 * pi))

    def interpol_pad_width(self, x, dx, mp):
        """Interpolated pad width distribution along track."""
        def single_mount_pattern(pos):
            start, end = pos - self.pad.wdthp / 2, pos + self.pad.wdthp / 2
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

    def calc_equiv_sleeper_factors(self):
        """Calculate equivalent sleeper factors according to Kostovasilis."""
        if not self.sleeper.equi_sm:
            pass
        else:
            self.sleeper.f_z = 1 + 12 * self.sleeper.y_sc**2 /(self.sleeper.lengs**2 + self.sleeper.hights**2)
            self.sleeper.f_x = 1 + 12 * self.sleeper.y_sc**2 / (self.sleeper.lengs**2 + self.sleeper.wdths**2)

    def calc_equiv_slab_factors(self):
        """Calculate equivalent slab factors according to Kostovasilis."""
        if not self.slab.equi_sm:
            pass

        else:
            self.slab.f_z = 1 + 12 * self.slab.y_sc ** 2 / (self.slab.lengs ** 2 + self.slab.heights ** 2)
            self.slab.f_x = 1 + 12 * self.slab.y_sc ** 2 / (self.slab.lengs ** 2 + self.slab.equ_wdths ** 2)

    def calc_ballast_rotational_stiffn(self):
        """Calculate rotational stiffnesses from ballast stiffnesses and sleeper/slab dimensions."""
        if hasattr(self, 'sleeper'):
            s_l = self.sleeper.lengs
            s_w = self.sleeper.wdths
        if hasattr(self, 'slab'):
            s_l = self.slab.lengs
            s_w = self.slab.equ_wdths

        self.ballast.sb_xr = s_l**2 / 12 * self.ballast.sb_z
        self.ballast.sb_yr = s_w**2 / 12 * self.ballast.sb_z
        self.ballast.sb_zr = s_l**2 / 12 * self.ballast.sb_x + s_w**2 / 12 * self.ballast.sb_y

    def calc_ballast_viscous_damp_cuton(self):
        """Calculate coupled viscous damping coefficients based on cut on frequencies."""
        self.ballast.db_x = self.ballast.etab_x * self.ballast.sb_x / (self.ballast.cof[7] * (2 * pi))
        self.ballast.db_z = self.ballast.etab_z * self.ballast.sb_z / (self.ballast.cof[8] * (2 * pi))
        self.ballast.db_y = self.ballast.etab_y * self.ballast.sb_y / (self.ballast.cof[9] * (2 * pi))
        self.ballast.db_xr = self.ballast.etab_r * self.ballast.sb_xr / (self.ballast.cof[10] * (2 * pi))
        self.ballast.db_yr = self.ballast.etab_r * self.ballast.sb_yr / (self.ballast.cof[11] * (2 * pi))
        self.ballast.db_zr = self.ballast.etab_r * self.ballast.sb_zr / (self.ballast.cof[12] * (2 * pi))


@dataclass(kw_only=True)
class SlabSingleRailTrack(SingleRailTrack):
    r"""Abstract base class for slab single rail track classes.

    Slab mass is set to a very large number to avoid displacement and simulate a rigid slab.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab, default=Slab(ms=1e20)
        Slab instance.
    """

    slab: Slab = field(default_factory=lambda: Slab(ms=1e20), metadata={'default_repr': 'Slab(ms=1e20)'})


@dataclass(kw_only=True)
class ContSlabSingleRailTrack(SlabSingleRailTrack):
    r"""Single rail slab track with continuous support.

    All superstructure properties are continuous along the track. The slab is assumed to be rigid.

    +------------------+-----------+--------------------+-------------+
    | Layer of Support | Component | Condition          | Variability |
    +==================+===========+====================+=============+
    | /                | rail      | continuous         | no          |
    +------------------+-----------+--------------------+-------------+
    | 1st              | pads      | continuous         | no          |
    +------------------+-----------+--------------------+-------------+
    | 1st/2nd          | slab      | continuous (rigid) | no          |
    +------------------+-----------+--------------------+-------------+
    | 2nd              | ballast   | /                  | /           |
    +------------------+-----------+--------------------+-------------+


    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab instance.
    pad : ContPad
        Continuous pad instance.
    l_track : float, default=100.0
        Track length :math:`[m]`. (May change slightly after discretization.
        The inclusion of boundary and calculation domain is required).
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import UIC60
    >>> from rolland.components import ContPad, Slab
    >>> from rolland.track import ContSlabSingleRailTrack

    >>> thepad = ContPad(sp = [300*10**6, 0], dp = [30000, 0])
    >>> theslab = Slab(ms = 250)
    >>> track = ContSlabSingleRailTrack(rail = UIC60, pad = thepad, slab = theslab, l_track = 145)
    ...
    """

    pad: ContPad
    l_track: float = 100.0
    z_f: float
    y_f: float

    def __post_init__(self):
        """post_init method to calculate derived properties after initialization."""
        self.calc_pad_warping_stiffn()
        self.calc_equiv_slab_factors()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.rail.z_f,
            self.rail.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.rail.chi,
            )
        E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806

        self.pad.cof = cof
        self.calc_pad_viscous_damp_cuton()

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class DiscrSlabSingleRailTrack(SlabSingleRailTrack):
    r"""Abstract base class for discrete slab single rail track classes.

    The pad and sleeper properties are discrete and the slab is assumed to be rigid.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab instance.
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, None)).
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

    All mounting properties are uniform and no variation is allowed. Slab is assumed to be rigid.

    +---------+-----------+------------------+-------------+
    | Layer   | Component | Condition        | Variability |
    +=========+===========+==================+=============+
    | /       | rail      | continuous       | no          |
    +---------+-----------+------------------+-------------+
    | 1st     | pads      | discrete         | no          |
    +---------+-----------+------------------+-------------+
    | 1st/2nd | slab      | discrete (rigid) | no          |
    +---------+-----------+------------------+-------------+
    | 2nd     | ballast   | /                | /           |
    +---------+-----------+------------------+-------------+

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab instance.
    pad : DiscrPad
        Discrete pad instance.
    distance : float, default=0.6
        Distance between mounting positions.
    num_mount : int, default=100
        Number of mounting positions.
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, None)).
    l_track : float, default=100.0
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import UIC60
    >>> from rolland.components import DiscrPad, Slab
    >>> from rolland.track import SimplePeriodicSlabSingleRailTrack

    >>> thepad = DiscrPad(sp = [300*10**6, 0], dp = [30000, 0])
    >>> theslab = Slab(ms = 250)
    >>> track = SimplePeriodicSlabSingleRailTrack(
    ...     rail=UIC60,
    ...     pad=thepad,
    ...     slab=theslab,
    ...     distance=0.6,
    ...     num_mount=100)
    ...
    """

    pad: DiscrPad
    distance: float = 0.6
    num_mount: int = 100
    z_f: float
    y_f: float

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

        self.calc_pad_warping_stiffn()
        self.calc_equiv_slab_factors()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.rail.z_f,
            self.rail.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.rail.chi,
            )
        E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806

        self.pad.cof = cof
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
    """Single rail slab track with varying periodic support.

    Variations in the form of periodicaly or stochasticaly varying mounting properties are allowed.
    Slab is assumed to be rigid.

    +---------+-----------+------------------+---------------------+
    | Layer   | Component | Condition        | Variability         |
    +=========+===========+==================+=====================+
    | /       | rail      | continuous       | no                  |
    +---------+-----------+------------------+---------------------+
    | 1st     | pads      | discrete         | periodic/stochastic |
    +---------+-----------+------------------+---------------------+
    | 1st/2nd | slab      | discrete (rigid) | periodic/stochastic |
    +---------+-----------+------------------+---------------------+
    | 2nd     | ballast   | /                | /                   |
    +---------+-----------+------------------+---------------------+

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab instance.
    pad : Arrangement
        Arrangement instance containing multiple pads.
    distance : Arrangement
        Arrangement instance containing multiple distances.
    num_mount : int, default=100
        Number of mounting positions.
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, None)).
    l_track : float, default=100.0
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import UIC60
    >>> from rolland.components import DiscrPad, Slab
    >>> from rolland.arrangement import PeriodicArrangement
    >>> from rolland.track import ArrangedSlabSingleRailTrack

    >>> thepadA = DiscrPad(sp = [300*10**6, 0], dp = [30000, 0])
    >>> thepadB = DiscrPad(sp = [400*10**6, 0], dp = [40000, 0])
    >>> theslab = Slab(ms = 250)
    >>> pad = PeriodicArrangement(item=[thepadA, thepadB])
    >>> distance = PeriodicArrangement(item=[0.65, 0.5])
    >>> track = ArrangedSlabSingleRailTrack(
    ...     rail=UIC60,
    ...     pad=pad,
    ...     slab=theslab,
    ...     distance=distance,
    ...     num_mount=100)
    ...
    """

    pad: Arrangement
    distance: Arrangement
    num_mount: int = 100
    z_f: float
    y_f: float

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

        self.calc_pad_warping_stiffn()
        self.calc_equiv_slab_factors()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.rail.z_f,
            self.rail.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.rail.chi,
            )
        E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806

        self.pad.cof = cof
        self.calc_pad_viscous_damp_cuton()

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

    .. note:: Properties of ballast need to be defined as continious values (per meter).

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
        The inclusion of boundary and calculation domain is required).
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.

    Example
    --------
    >>> from rolland.database.rail.db_rail import UIC60
    >>> from rolland.components import ContPad, Slab
    >>> from rolland.track import ContBallastedSingleRailTrack

    >>> thepad = ContPad(sp = [300*10**6, 0], dp = [30000, 0])
    >>> theslab = Slab(ms = 250)
    >>> track = ContBallastedSingleRailTrack(rail = UIC60, pad = thepad, slab = theslab)
    ...
    """

    pad: ContPad
    slab: Slab
    l_track: float = 100.0
    z_f: float
    y_f: float

    def __post_init__(self):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_pad_warping_stiffn()
        self.calc_equiv_slab_factors()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.rail.z_f,
            self.rail.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.rail.chi,
            )
        E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806

        self.pad.cof = cof
        self.ballast.cof = cof
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
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, Sleeper)).
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
    pad : ContPad
        Continuous pad instance.
    sleeper : Instance of :class:`~rolland.components.sleeper` class
        Sleeper instance.
    distance : float, default=0.6
        Distance between mounting positions.
    num_mount : int, default=100
        Number of mounting positions.
    mount_prop : dict
        Dictionary for discrete mounting positions (x-> (Pad, Sleeper)).
    l_track : float
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import UIC60
    >>> from rolland.components import DiscrPad, Sleeper
    >>> from rolland.track import SimplePeriodicBallastedSingleRailTrack

    >>> thepad = DiscrPad(sp = [300*10**6, 0], dp = [30000, 0])
    >>> thesleeper = Sleeper(ms = 150)
    >>> distance = 0.6
    >>> tr = SimplePeriodicBallastedSingleRailTrack(
    ...     rail=UIC60,
    ...     pad=thepad,
    ...     sleeper=thesleeper,
    ...     distance=distance)
    """

    sleeper: Sleeper
    pad: DiscrPad
    distance: float = 0.6
    num_mount: int = 100
    z_f: float
    y_f: float

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

        self.calc_pad_warping_stiffn()
        self.calc_equiv_sleeper_factors()
        self.calc_ballast_rotational_stiffn()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.rail.z_f,
            self.rail.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.rail.chi,
            )
        E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806

        self.pad.cof = cof
        self.ballast.cof = cof
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
    """Single rail ballasted track with varying periodic support.

    Variations in the form of periodicaly or stochasticaly varying mounting properties are allowed.

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
    mount_prop : dict
        Dictionary for discrete mounting positions (x-> (Pad, Sleeper)).
    l_track : float
        Track length :math:`[m]`. (May change slightly after discretization.
        Results from the number of mounting positions and the mounting distances).
    z_f: float
        Vertical distance from rail foot to centroid :math:`[m]`.
    y_f: float
        Lateral distance from rail foot to centroid :math:`[m]`.


    Example
    --------
    >>> from rolland.database.rail.db_rail import UIC60
    >>> from rolland.components import DiscrPad, Sleeper
    >>> from rolland.arrangement import PeriodicArrangement
    >>> from rolland.track import ArrangedBallastedSingleRailTrack

    >>> thepadA = DiscrPad(sp = [300*10**6, 0], dp = [30000, 0])
    >>> thepadB = DiscrPad(sp = [400*10**6, 0], dp = [40000, 0])
    >>> thesleeperA = Sleeper(ms = 150)
    >>> thesleeperB = Sleeper(ms = 200)
    >>> pad = PeriodicArrangement(item=[thepadA, thepadB])
    >>> distance = PeriodicArrangement(item=[0.65, 0.5])
    >>> sleeper = PeriodicArrangement(item=[thesleeperA, thesleeperB])
    >>> track = ArrangedBallastedSingleRailTrack(
    ...     rail=UIC60,
    ...     pad=pad,
    ...     sleeper=sleeper,
    ...     distance=distance)
    """

    sleeper: Arrangement
    pad: Arrangement
    ballast: Arrangement
    distance: Arrangement
    num_mount: int = 100
    z_f: float
    y_f: float

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

        self.calc_pad_warping_stiffn()
        self.calc_equiv_sleeper_factors()
        self.calc_ballast_rotational_stiffn()

        K0, K1, K2, Mr = build_rail_matrices(self.rail, "viscous")  # noqa: N806
        Tf, Tst, Tsb = build_transfm_matrices( # noqa: N806
            self.rail.z_f,
            self.rail.y_f,
            self.slab.z_st,
            self.slab.z_sb,
            self.rail.chi,
            )
        E = build_equ_sleeper_matrix(self) # noqa: N806
        Ms = build_sleep_mass_matrix(self, E) # noqa: N806
        Kp, Kb = build_pad_ballast_stiff_matrices(self, "viscous", E) # noqa: N806
        K_fnd = build_fnd_stiff_matrix(Kp, Tf, Kb, Tst, Tsb) # noqa: N806
        cof = calc_cut_on_frequ(K0, K_fnd, Mr, Ms) # noqa: N806

        self.pad.cof = cof
        self.ballast.cof = cof
        self.calc_pad_viscous_damp_cuton()
        self.calc_ballast_viscous_damp_cuton()

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
