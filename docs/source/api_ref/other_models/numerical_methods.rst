.. _nummod:

Numerical Models
================

.. note:: 
   The ``methods.numerical`` package is provided solely for reference, benchmarking, and validation purposes against the core Rolland model.

The package contains numerical methods using 1 DOF Euler-Bernoulli beam finite-difference formulations (as developed by K. Stampka and E. Sarradj).

.. currentmodule:: rolland.methods.numerical

Finite Difference Time-Domain (Stampka)
---------------------------------------

.. autosummary::
   :toctree: numerical/

   PMLStampka
   GaussianImpulseStampka
   DiscretizationStampka
   DeflectionStampka
