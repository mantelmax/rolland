# ------------------------------------------------------------------------------
# Rolland
# ------------------------------------------------------------------------------

"""The Rolland library: several classes for the implementation of rolling noise calculation."""

__version__ = '26.11'

from .arrangement import Arrangement, PeriodicArrangement, RandomArrangement
from .boundary import CFSPML
from .components import Ballast, ContPad, DiscrPad, Rail, Slab, Sleeper, Wheel, WheelGreensfunc
from .deflection import Deflection
from .domainsetup import DomSetup
from .excitation import Excitation, GaussianImpulse
from .postprocessing import PostProcessing, TrackDecayRate, TrackResponse, VehicleResponse
from .radiation import Radiation, SimplifiedMonopoleRadiation, a_weighting
from .track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    ContBallastedSingleRailTrack,
    ContSlabSingleRailTrack,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
)
from .visualization import TrackSchematic

__all__ = [
    'Arrangement',
    'PeriodicArrangement',
    'RandomArrangement',
    'Ballast',
    'ContPad',
    'DiscrPad',
    'Rail',
    'Slab',
    'Sleeper',
    'Wheel',
    'WheelGreensfunc',
    'Excitation',
    'ArrangedBallastedSingleRailTrack',
    'ArrangedSlabSingleRailTrack',
    'ContBallastedSingleRailTrack',
    'ContSlabSingleRailTrack',
    'SimplePeriodicBallastedSingleRailTrack',
    'SimplePeriodicSlabSingleRailTrack',
    'Deflection',
    'CFSPML',
    'DomSetup',
    'GaussianImpulse',
    'PostProcessing',
    'TrackResponse',
    'TrackDecayRate',
    'VehicleResponse',
    'Radiation',
    'SimplifiedMonopoleRadiation',
    'a_weighting',
    'TrackSchematic',
]
