Rolland
========================

.. note:: This repository is still under development!

Rolling Noise and Dynamics (**Rolland**) is an advanced simulation and calculation model designed to analyze, predict,
and optimize the acoustic properties of railway tracks, with a focus on realistic, efficient, and fast computations.


Features
--------

**Current Features**
    #. Calculates the track response for a non-moving gaussian impulse at a certain position
    #. Applies Finite Difference Method in time domain
    #. Allows the definition of arbitrary track structures

       * Enables periodic or stochastic variations of the track properties (e.g. stochastically varying sleeper distances)
       * Enables the representation of track property deviations that occur in practise
    #. Includes several analytical models for comparison and validation

**Planned Features**
    #. Full rail dynamics
    #. Consideration of rail radiation
    #. Consideration of non-linear effects
    #. Excitation by multiple moving wheels


.. image:: images/mwi_light.png
   :alt: MWI Image
   :width: 700px
   :class: light-mode-image

.. image:: images/mwi_dark.png
   :alt: MWI Image
   :width: 700px
   :class: dark-mode-image


.. toctree::
   :hidden:
   
   Installation <install/index>
   User Guide <user_guide/index>
   API Reference <api_ref/index>
   Examples <examples/index>
   What's New <whats_new/index>
   License <license/index>
   Literature <literature/index>