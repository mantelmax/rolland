.. _track:

Track
=====

The ``track`` module provides classes for defining railway track superstructures.

.. currentmodule:: rolland.track

Abstract Base Classes
---------------------
Classes used for subclassing and type hints.

.. autosummary::
   :toctree: track

   Track
   SingleRailTrack
   SlabSingleRailTrack
   DiscrSlabSingleRailTrack
   BallastedSingleRailTrack
   DiscrBallastedSingleRailTrack

Slab Track Models
-----------------
Concrete models for slab track structures.

.. autosummary::
   :toctree: track

   ContSlabSingleRailTrack
   SimplePeriodicSlabSingleRailTrack
   ArrangedSlabSingleRailTrack

Ballasted Track Models
----------------------
Concrete models for ballasted track structures.

.. autosummary::
   :toctree: track

   ContBallastedSingleRailTrack
   SimplePeriodicBallastedSingleRailTrack
   ArrangedBallastedSingleRailTrack


