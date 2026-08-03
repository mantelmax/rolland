.. _track:

Track
=====

The ``track`` module provides classes for defining railway track superstructures.

Abstract Base Classes
---------------------
Classes used for subclassing and type hints.

.. autosummary::
   :toctree: track

   rolland.track.Track
   rolland.track.SingleRailTrack
   rolland.track.SlabSingleRailTrack
   rolland.track.BallastedSingleRailTrack

Slab Track Models
-----------------
Concrete models for slab track structures.

.. autosummary::
   :toctree: track

   rolland.track.ContSlabSingleRailTrack
   rolland.track.DiscrSlabSingleRailTrack
   rolland.track.SimplePeriodicSlabSingleRailTrack
   rolland.track.ArrangedSlabSingleRailTrack

Ballasted Track Models
----------------------
Concrete models for ballasted track structures.

.. autosummary::
   :toctree: track

   rolland.track.ContBallastedSingleRailTrack
   rolland.track.DiscrBallastedSingleRailTrack
   rolland.track.SimplePeriodicBallastedSingleRailTrack
   rolland.track.ArrangedBallastedSingleRailTrack

