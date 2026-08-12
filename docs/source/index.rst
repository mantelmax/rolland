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

By employing an explicit Finite Difference Method (FDM) scheme, **Rolland** solves 27 coupled differential equations to capture full 3D track dynamics—including coupled vertical bending, lateral bending, axial torsion, cross-sectional warping, and sleeper movement under eccentric excitation and support conditions. The framework incorporates Complex Frequency-Shifted Perfectly Matched Layers (CFS-PML) for infinite track modeling and supports spatially varying track properties as well as moving excitation sources.



Key Features
------------

Current Features
~~~~~~~~~~~~~~~~

* **Full Track Dynamics:** Solves 27 coupled differential equations to capture vertical and lateral bending waves, torsional waves, and warping effects together with sleeper movement and eccentric support reactions.
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

   Mantel, M., & Sarradj, E. (2026). Time-domain modeling of coupled wave propagation in discretely supported railway tracks. *Computers & Structures* (Accepted for publication on August 12, 2026).

.. code-block:: bibtex

   @article{mantel2026timedomain,
     title   = {Time-domain modeling of coupled wave propagation in discretely supported railway tracks},
     author  = {Mantel, Maximilian and Sarradj, Ennes},
     journal = {Computers \& Structures},
     year    = {2026},
     note    = {Accepted for publication on August 12, 2026},
     doi     = {}
   }

and our software:

   Mantel, M., Wagner, B., & Sarradj, E. (2026). Rolland: A Time-Domain Simulation Framework for Railway Track Dynamics and Rolling Noise. https://github.com/mantelmax/rolland

.. code-block:: bibtex

   @software{mantel2026rolland_api,
     title  = {Rolland: A Time-Domain Simulation Framework for Railway Track Dynamics and Rolling Noise},
     author = {Mantel, Maximilian and Wagner, Benjamin and Sarradj, Ennes},
     year   = {2026},
     url    = {https://github.com/mantelmax/rolland},
     doi    = {}
   }


.. toctree::
   :hidden:
   :maxdepth: 2
   
   Installation <install/index>
   User Guide <user_guide/index>
   API Reference <api_ref/index>
   Examples <examples/index>
   What's New <whats_new/index>
   Funding <funding/index>
   License <license/index>
   Literature <literature/index>