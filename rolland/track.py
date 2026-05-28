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

from .arrangement import Arrangement
from .components import Ballast, ContPad, DiscrPad, Rail, Slab, Sleeper
from .observing import observable, observe


class Track(ABC):
    r"""Abstract base class for track classes."""

    @abstractmethod
    def validate_track(self):
        """Validate the track configuration."""

@dataclass(kw_only=True)
class SingleRailTrack(Track):
    r"""Abstract base class for single rail track classes.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    """

    rail: Rail

    @abstractmethod
    def validate_single_rail_track(self):
        """Validate the single rail configuration."""


@dataclass(kw_only=True)
class SlabSingleRailTrack(SingleRailTrack):
    r"""Abstract base class for slab single rail track classes.

    Slab mass is set to a very large number to avoid displacement and simulate a rigid slab.

    Attributes
    ----------
    rail : Rail
        Rail instance.
    slab : Slab
        Slab instance.
    """

    slab: Slab = field(default_factory=lambda: Slab(ms=1e20))

    @abstractmethod
    def validate_slab_single_rail_track(self):
        """Validate the slab single rail configuration."""


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
    l_track: float = field(default=100.0)

    def validate_track(self):
        """Validate the track configuration."""

    def validate_single_rail_track(self):
        """Validate the single rail configuration."""

    def validate_slab_single_rail_track(self):
        """Validate the slab single rail configuration."""


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
    pad : DiscrPad
        Discrete pad instance.
    mount_prop : dict[float, tuple[DiscrPad, None, None]]
        Dictionary for discrete mounting positions (x-> (Pad, None)).
    """

    pad: DiscrPad

    # Dictionary for discrete mounting positions (x-> (Pad)).
    # May have nonuniform properties.
    mount_prop: dict[float, tuple[DiscrPad, None, None]] = field(default_factory=dict) #

    def __repr__(self):
        """Represent mounting properties as string."""
        st = ""
        for x in sorted(self.mount_prop.keys()):
            p, s, b= self.mount_prop[x]
            st += f'{x}, {p.sp}, {s.ms}, {b.sb} \n'
        return st

    @abstractmethod
    def validate_discr_slab_single_rail_track(self):
        """Validate the discrete slab single rail configuration."""


@observable
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

    distance: float = 0.6
    num_mount: int = 100

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

    @observe('num_mount', 'distance', 'pad')
    def calc_mount_prop(self, change=None):
        """"Calculate the mounting properties."""
        self.mount_prop = {}
        for _i in range(self.num_mount):
            x = float(Decimal(str(_i)) * Decimal(str(self.distance)))
            self.mount_prop[x] = (self.pad, None, None)
        self.l_track = max(self.mount_prop.keys())

    def validate_track(self):
        """Validate the track configuration."""

    def validate_single_rail_track(self):
        """Validate the single rail configuration."""

    def validate_slab_single_rail_track(self):
        """Validate the slab single rail configuration."""

    def validate_discr_slab_single_rail_track(self):
        """Validate the discrete slab single rail configuration."""


@observable
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

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

    @observe('num_mount', 'distance', 'pad')
    def calc_mount_prop(self, change=None):
        """Calculate the mounting properties."""
        x = Decimal(str(0))
        for p, d in zip(self.pad.generate(self.num_mount),
                        self.distance.generate(self.num_mount), strict=False):
            self.mount_prop[float(Decimal(str(x)))] = (p, None, None)
            x += Decimal(str(d))
        self.l_track = max(self.mount_prop.keys())

    def validate_track(self):
        """Validate the track configuration."""

    def validate_single_rail_track(self):
        """Validate the single rail configuration."""

    def validate_slab_single_rail_track(self):
        """Validate the slab single rail configuration."""

    def validate_discr_slab_single_rail_track(self):
        """Validate the discrete slab single rail configuration."""


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

    @abstractmethod
    def validate_ballasted_single_rail_track(self):
        """Validate the ballasted single rail configuration."""


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

    def validate_track(self):
        """Validate the track configuration."""

    def validate_single_rail_track(self):
        """Validate the single rail configuration."""

    def validate_ballasted_single_rail_track(self):
        """Validate the ballasted single rail configuration."""


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

    # Pads and sleepers may have nonuniform properties Dictionary (x-> (Pad, Sleeper))
    mount_prop: dict[float, tuple[DiscrPad, None, None]] = field(default_factory=dict)

    def __repr__(self):
        """Represent mounting properties as string."""
        st = ""
        for x in sorted(self.mount_prop.keys()):
            p, s, b = self.mount_prop[x]
            st += f'{x}, {p.sp}, {s.ms}, {b.sb} \n'
        return st

    @abstractmethod
    def validate_discr_ballasted_single_rail_track(self):
        """Validate the discrete ballasted single rail configuration."""


@observable
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
    ballast: Ballast
    distance: float = field(default=0.6)
    num_mount: int = field(default=100)

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

    @observe('num_mount', 'distance', 'pad', 'sleeper')
    def calc_mount_prop(self, change=None):
        """Calculate the mounting properties."""
        for _i in range(self.num_mount):
            # Calculate the mounting position
            # Use Decimal to avoid floating-point representation errors
            x = float(Decimal(str(_i)) * Decimal(str(self.distance)))
            self.mount_prop[x] = (self.pad, self.sleeper, self.ballast)
        self.l_track = max(self.mount_prop.keys())

    def validate_track(self):
        """Validate the track configuration."""

    def validate_single_rail_track(self):
        """Validate the single rail configuration."""

    def validate_ballasted_single_rail_track(self):
        """Validate the ballasted single rail configuration."""

    def validate_discr_ballasted_single_rail_track(self):
        """Validate the discrete ballasted single rail configuration."""


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
    ballast : Ballast
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
    num_mount: int = field(default=100)

    def __post_init__(self, *args, **kwargs):
        """post_init method to calculate mounting properties after initialization."""
        self.calc_mount_prop()

    #@observe('num_mount', 'distance', 'pad', 'sleeper', 'ballast')
    def calc_mount_prop(self, change=None):
        """Calculate the mounting properties."""
        x = Decimal(str(0))
        for s, p, b, d in zip(self.sleeper.generate(self.num_mount),
                           self.pad.generate(self.num_mount),
                           self.ballast.generate(self.num_mount),
                           self.distance.generate(self.num_mount), strict=False):
            self.mount_prop[float(Decimal(str(x)))] = (p, s, b)
            x += Decimal(str(d))
        self.l_track = max(self.mount_prop.keys())

    def validate_track(self):
        """Validate the track configuration."""

    def validate_single_rail_track(self):
        """Validate the single rail configuration."""

    def validate_ballasted_single_rail_track(self):
        """Validate the ballasted single rail configuration."""

    def validate_discr_ballasted_single_rail_track(self):
        """Validate the discrete ballasted single rail configuration."""
