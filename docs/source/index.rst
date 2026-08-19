Rolland
========================

.. note::
   This repository is in an early stage of development and may still contain bugs or errors. Please be aware of this when using the software. Any feedback, bug reports, or suggestions are highly appreciated and will help us improve **Rolland**!


.. image:: images/mwi_light.png
   :alt: MWI Image
   :width: 700px
   :class: light-mode-image

.. image:: images/mwi_animated_sphinx.gif
   :alt: MWI Image
   :width: 700px
   :class: dark-mode-image


Rolling Noise and Dynamics (**Rolland**) is an open-source, high-performance time-domain simulation framework designed to analyze, predict, and optimize the dynamic and acoustic properties of railway tracks.

By employing an explicit Finite Difference Method (FDM) scheme, **Rolland** solves 13 differential equations of motion alongside 14 additional auxiliary differential equations corresponding to the boundary domain. This captures full track dynamics—including coupled vertical and lateral bending, longitudinal waves, axial torsion, cross-sectional warping, and sleeper movement under eccentric excitation and support conditions. The framework incorporates Complex Frequency-Shifted Perfectly Matched Layers (CFS-PML) for infinite track modeling and supports spatially varying track properties as well as moving excitation sources.



Key Features
------------

Current Features
~~~~~~~~~~~~~~~~

* **Full Track Dynamics:** Solves 13 differential equations of motion alongside 14 additional auxiliary differential equations to capture vertical and lateral bending waves, longitudinal waves, torsional waves, and warping effects together with sleeper movement and eccentric support reactions.
* **Fast Time-Domain Solver:** Uses explicit Finite Difference Method (FDM) schemes with high-order stencils for fast simulations.
* **Infinite Track Boundary (CFS-PML):** Uses absorbing boundary layers to eliminate artificial wave reflections.
* **Flexible Track Structures:** Supports ballasted and slab tracks with continuous or discrete supports, including spatial (periodic or stochastic) track property variations.
* **Excitation:** Includes stationary excitation (Gaussian impulse) and moving sources (e.g. moving random force).
* **Post-Processing & Validation:** Computes point/transfer mobility, Track Decay Rate (TDR), and provides built-in reference models for easy validation.

Planned Features
~~~~~~~~~~~~~~~~

* Non-linear Hertzian contact dynamics for wheel-rail interaction.
* Multi-wheel vehicle pass-by excitation models.
* Rail acoustic radiation modeling.


Citation
--------
If you use **Rolland** for academic work, please consider citing both our publication:

   Mantel, M., & Sarradj, E. (in press). Time-domain modeling of coupled wave propagation in discretely supported railway tracks. Computers & Structures.

.. code-block:: bibtex

    @article{mantel2026timedomain,
      author   = {Mantel, Maximilian and Sarradj, Ennes},
      title    = {Time-domain modeling of coupled wave propagation in discretely supported railway tracks},
      journal  = {Computers \& Structures},
      pubstate = {inpress}
    }


and our software:

   Mantel, M., Wagner, B., & Sarradj, E. (2026). Rolland: A time-domain simulation framework for railway track dynamics and rolling noise (Version 26.08a0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21920225

.. code-block:: bibtex

    @software{mantel2026rolland_api,
      author  = {Mantel, Maximilian and Wagner, Benjamin and Sarradj, Ennes},
      title   = {Rolland: A Time-Domain Simulation Framework for Railway Track Dynamics and Rolling Noise},
      year    = {2026},
      version = {v26.08a0},
      doi     = {10.5281/zenodo.21920225},
      url     = {https://github.com/mantelmax/rolland}
    }


.. toctree::
   :hidden:
   :maxdepth: 2
   
   Installation <install/index>
   User Guide <user_guide/index>
   API Reference <api_ref/index>
   Examples <examples/index>
   Literature <literature/index>