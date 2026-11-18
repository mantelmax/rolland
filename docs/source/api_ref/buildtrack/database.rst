.. _rail_database:

Rail Database
=============

**Rolland** ships a set of standard rail profiles according to EN 13674-1. Each profile
is a ready-to-use :class:`~rolland.components.Rail` instance that can be passed directly to
a track.

.. currentmodule:: rolland.database.rail.db_rail

Using a bundled profile
-----------------------

A bundled profile can either be imported directly or loaded by its name. Both ways
return a :class:`~rolland.components.Rail` instance with the same properties. The
imported profile is shared by all imports, whereas :func:`load_rail` creates a new
instance on every call.

.. tab-set::

   .. tab-item:: Import

      .. code-block:: python

         from rolland.database.rail.db_rail import rail_60E1

   .. tab-item:: Load by name

      .. code-block:: python

         from rolland.database.rail.db_rail import load_rail

         rail = load_rail('rail_60E1')

:func:`available_rails` returns the names of all bundled profiles.

Available profiles
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 30 15 40

   * - Name
     - Standard
     - Mass [kg/m]
     - Description
   * - ``rail_49E1``
     - EN 13674-1 (49E1, S49)
     - 49.67
     - 49 kg/m rail for regional and secondary lines.
   * - ``rail_49E5``
     - EN 13674-1 (49E5)
     - 49.19
     - 49 kg/m rail, variant for lightly loaded track.
   * - ``rail_54E1``
     - EN 13674-1 (54E1, UIC54)
     - 54.81
     - 54 kg/m mainline rail, the second most common European profile.
   * - ``rail_54E3``
     - EN 13674-1 (54E3, S54)
     - 54.64
     - 54 kg/m rail, German S54 variant.
   * - ``rail_54E4``
     - EN 13674-1 (54E4)
     - 54.38
     - 54 kg/m rail, variant of the 54 kg/m family.
   * - ``rail_60E1``
     - EN 13674-1 (60E1, UIC60)
     - 60.20
     - Standard 60 kg/m mainline rail (formerly UIC60), used as the default profile in **Rolland**.
   * - ``rail_60E2``
     - EN 13674-1 (60E2)
     - 60.12
     - 60 kg/m mainline rail for heavily loaded track.

Using an own profile
--------------------

An own profile is loaded by passing the path to its TOML file instead of a name:

.. code-block:: python

   from rolland.database.rail.db_rail import load_rail

   rail = load_rail('path/to/my_rail.toml')

A profile consists of the following files located in the same directory:

* a TOML file holding the scalar rail parameters,
* a CSV file holding the rail outline coordinates with the columns ``Y,Z`` in metres and
  one header line, and
* optionally an NPY file holding the warping function as an array of shape ``(n, 3)``
  with the columns ``Y``, ``Z`` in metres, in the same coordinate system as the outline,
  and the warping value with respect to the shear center in :math:`\mathrm{m^2}`.
  Without this file, the warping is neglected.

The TOML file groups its values into tables. The ``[meta]`` table describes the profile,
and its ``name`` must match the file name. All other tables pass their keys directly as
arguments to :class:`~rolland.components.Rail`, so their names only serve to structure
the file. The key ``outline`` names the CSV file and the key ``warping`` names the NPY file.

.. code-block:: toml

   # my_rail.toml
   [meta]
   name = "my_rail"                # must match the file name
   standard = "EN 13674-1 (60E1)"
   description = "Short description of the profile."

   [material]
   E = 210e9                       # Young's modulus [Pa]
   G = 80.769e9                    # shear modulus [Pa]
   rho = 7860.0                    # density [kg/m^3]
   # ...

   [geometry]
   outline = "my_rail.csv"         # rail outline coordinates [m]
   warping = "my_rail_warping.npy" # warping function (optional)
   mr = 60.2                       # mass per unit length [kg/m]
   # ...

   [cross_section]
   Iyr = 3.03813e-05               # second moment of area about y [m^4]
   # ...

All attributes of :class:`~rolland.components.Rail` are listed in its documentation. The
bundled file ``rolland/database/rail/profiles/rail_60E1.toml`` serves as a complete
example.

Functions
---------

.. autosummary::
   :toctree: database

   available_rails
   load_rail
   load_rail_geo
   load_rail_warping
