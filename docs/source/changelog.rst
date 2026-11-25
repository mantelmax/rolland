Changelog
=========

Unreleased
----------

**New Features**

* **Sound radiation (experimental)**: New module :mod:`rolland.radiation` with :class:`~rolland.radiation.SimplifiedMonopoleRadiation`. The radiated sound pressure is computed from a full ``Deflection`` wavefield using a monopole model on a GMSH surface mesh of the rail, with optional ground reflection (image source method). Results are available as amplitude spectrum, spectral density and time signal, optionally A-weighted (IEC 61672-1, default), and can be exported as calibrated WAV file (0 dBFS = 20 Pa). Adds ``gmsh``, ``librosa`` and ``soundfile`` as dependencies.
* New example: :doc:`Sound Radiation of a Vertically Excited Rail </examples/radiation/vertical_impulse_radiation>`.

v26.11
------

**Breaking Changes**

* The ``UIC60`` rail profile instance has been removed. Use ``rail_60E1`` or ``load_rail('rail_60E1')`` from :mod:`rolland.database.rail.db_rail` instead.

**New Features**

* **File-based rail profile database**: Rail profiles are now stored as TOML/CSV/NPY files and loaded via ``load_rail()`` / ``available_rails()``. Seven profiles are bundled: 49E1, 49E5, 54E1, 54E3, 54E4, 60E1, 60E2.
* **Spatially resolved warping function**: ``Rail.chi`` is now an array of shape ``(n, 3)`` with a new ``Rail.chi_at(y, z)`` interpolation method. All track classes evaluate the warping at the rail foot position ``(y_f, z_f)``.
* **TDR for analytical methods**: :class:`~rolland.postprocessing.TrackDecayRate` now supports analytical/semi-analytical results (Kosto, EBB, TSB) in addition to Rolland and Stampka simulations.
* **User guide notebooks**: Three new Jupyter notebooks (create track, excite track, evaluate results).

**Bug Fixes**

* Fixed sign of ``e_s`` in pad warping stiffness calculation.
* ``chi_f`` in :class:`~rolland.domainsetup.DomSetup` is now correctly taken from the track instead of being hardcoded to 0.

v26.08
------

The initial release of **Rolland**, providing a highly optimized, Devito-based time-domain simulation framework designed for coupled wave propagation and dynamics in railway tracks. 

This release introduces the foundational mechanics validated in our publication: `Time-domain modeling of coupled wave propagation in discretely supported railway tracks <https://doi.org/10.1016/j.compstruc.2026.108432>`_ :cite:`mantel2026timedomain`.

**Core Features**

* **Time-Domain Solver**: Implements an explicit Finite Difference Method (FDM) capable of efficiently solving 13 differential equations of motion alongside 14 auxiliary differential equations.
* **Track Dynamics**: Full structural representation capturing vertical and lateral bending waves, longitudinal waves, axial torsion, cross-sectional warping, and sleeper movement under eccentric excitation and support conditions.
* **Track Structures**: Built-in support for modeling flexible track configurations, including ballasted and slab tracks with either continuous or discrete supports. Supports spatial variations (periodic and stochastic) of track properties.
* **Infinite Track Boundaries**: First-of-its-kind time-domain implementation of Complex Frequency-Shifted Perfectly Matched Layers (CFS-PML) to reliably eliminate artificial boundary reflections in the waveguide.
* **Dynamic Excitations**: Flexible configuration for both stationary sources (e.g., Gaussian impulses) and moving sources (e.g., moving random forces) along the track.
* **Post-Processing & Validation**: Integrated utilities for calculating point and transfer mobilities, as well as Track Decay Rates (TDR). Includes five built-in analytical frequency-domain reference models for rapid validation: Euler-Bernoulli and Timoshenko beam models on continuous and discrete one- or two-layer supports.
