API Reference
=============

.. toctree::
   :hidden:
   :caption: Build Track
   :maxdepth: 2
   :titlesonly:

   buildtrack/components
   buildtrack/track
   buildtrack/arrangement

.. toctree::
   :hidden:
   :caption: Rolland
   :maxdepth: 2
   :titlesonly:

   rolland/domainsetup
   rolland/boundary
   rolland/excitation
   rolland/deflection

.. toctree::
   :hidden:
   :caption: Other Models
   :maxdepth: 2
   :titlesonly:

   other_models/analyticalmethods

.. toctree::
   :hidden:
   :caption: Postprocessing
   :maxdepth: 2
   :titlesonly:

   postprocessing





.. tab-set::

   .. tab-item:: Build Track

      The **Build Track** section contains all classes and functions to create a track model.

      .. grid:: 1 2 3 3
         :gutter: 3

         .. grid-item-card:: Components
            :link: buildtrack/components
            :link-type: doc

            Rail profiles, pads, sleepers, slabs, and ballast component definitions.

         .. grid-item-card:: Track
            :link: buildtrack/track
            :link-type: doc

            Track assembly classes for continuous and discrete single rail track structures.

         .. grid-item-card:: Arrangement
            :link: buildtrack/arrangement
            :link-type: doc

            Periodic and random support property and spacing arrangements.

   .. tab-item:: Rolland

      The **Rolland** section contains all classes and functions corresponding to the Rolland model.

      .. grid:: 1 2 2 2
         :gutter: 3

         .. grid-item-card:: Domain Setup
            :link: rolland/domainsetup
            :link-type: doc

            Discretization, grid generation, and temporal/spatial numerical domain setup.

         .. grid-item-card:: Boundary
            :link: rolland/boundary
            :link-type: doc

            Complex Frequency-Shifted Perfectly Matched Layer (CFS-PML) absorbing boundaries.

         .. grid-item-card:: Excitation
            :link: rolland/excitation
            :link-type: doc

            Stationary impulse sources and moving force excitation models.

         .. grid-item-card:: Deflection
            :link: rolland/deflection
            :link-type: doc

            Time-domain finite difference wavefield deflection solvers and record stores.

   .. tab-item:: Other Models

      This section contains analytical and numerical models for benchmark comparisons.

      .. grid:: 1 2 2 2
         :gutter: 3

         .. grid-item-card:: Analytical Methods
            :link: other_models/analyticalmethods
            :link-type: doc

            Reference analytical solutions and benchmark track models.

   .. tab-item:: Postprocessing

      The **Postprocessing** section contains classes and functions to evaluate simulation outputs.

      .. grid:: 1 2 2 2
         :gutter: 3

         .. grid-item-card:: Postprocessing
            :link: postprocessing
            :link-type: doc

            FFT spectral analyses, mobility calculation, moving receptance, and plot utilities.