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


Installation
------------
To install Rolland, you can use pip. It is recommended to create a virtual environment first.

.. code-block:: bash

   pip install git+https://github.com/mantelmax/rolland.git


Explore
-------

.. tab-set::

   .. tab-item:: How to

      In the **How-to** section, you can learn more about the basic usage of Rolland.

      .. toctree::
         :caption: How to
         :maxdepth: 1
         :titlesonly:

         how_to/quick_start
         how_to/different_tracks
         how_to/variation

   .. tab-item:: Build Track

      The **Build Track** section contains all classes and functions to create a track model.

      .. toctree::
         :caption: Build Track
         :maxdepth: 1
         :titlesonly:

         buildtrack/components
         buildtrack/track
         buildtrack/arrangement

   .. tab-item:: Rolland

      The **Rolland** section contains all classes and functions corresponding to the Rolland model.

      .. toctree::
         :caption: Rolland
         :maxdepth: 1
         :titlesonly:

         rolland/boundary
         rolland/excitation
         rolland/discretization
         rolland/deflection



   .. tab-item:: Other Models

      This section contains several analytical and numerical models in order to compare the results of the
      Rolland model.

      .. toctree::
         :caption: Other Models
         :maxdepth: 1
         :titlesonly:

         other_models/analyticalmethods



   .. tab-item:: Postprocessing

      The **Postprocessing** section contains all classes and functions to postprocess the results of
      the Rolland model and the analytical methods.

      .. toctree::
         :caption: Postprocessing
         :maxdepth: 1
         :titlesonly:

         postprocessing




.. toctree::
   :hidden:
   :caption: Literature

   literature/index