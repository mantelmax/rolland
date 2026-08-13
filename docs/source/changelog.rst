Changelog
=========

v26.08
------

The initial release of **Rolland**, providing a highly optimized, Devito-based time-domain simulation framework designed for coupled wave propagation and dynamics in railway tracks. 

This release introduces the foundational mechanics validated in our publication: *Time-domain modeling of coupled wave propagation in discretely supported railway tracks*.

**Core Features**

* **Time-Domain Solver**: Implements an explicit Finite Difference Method (FDM) capable of efficiently solving 13 differential equations of motion alongside 14 auxiliary differential equations.
* **Track Dynamics**: Full structural representation capturing vertical and lateral bending waves, longitudinal waves, axial torsion, cross-sectional warping, and sleeper movement under eccentric excitation and support conditions.
* **Track Structures**: Built-in support for modeling flexible track configurations, including ballasted and slab tracks with either continuous or discrete supports. Supports spatial variations (periodic and stochastic) of track properties.
* **Infinite Track Boundaries**: First-of-its-kind time-domain implementation of Complex Frequency-Shifted Perfectly Matched Layers (CFS-PML) to reliably eliminate artificial boundary reflections in the waveguide.
* **Dynamic Excitations**: Flexible configuration for both stationary sources (e.g., Gaussian impulses) and moving sources (e.g., moving random forces) along the track.
* **Post-Processing & Validation**: Integrated utilities for calculating point and transfer mobilities, as well as Track Decay Rates (TDR). Includes five built-in analytical frequency-domain reference models for rapid validation: Euler-Bernoulli and Timoshenko beam models on continuous and discrete one- or two-layer supports.
