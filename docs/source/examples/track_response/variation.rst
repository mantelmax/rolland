.. _variation:

Structural Irregularity
=======================
Structural irregularity in railway tracks (such as non-uniform sleeper spacing or varying rail pad properties)
can significantly affect track vibration characteristics (see :cite:t:`mantel2024`).
This example demonstrates how to set up and run a simulation using the Rolland library to calculate the vertical
and lateral frequency responses of a track model with structural irregularity.


.. note:: This example calculates the vertical and lateral track response at a single excitation position and at 10 m distance.
            For tracks with structural irregularity, it is recommended to evaluate responses at multiple positions for a comprehensive representation.

.. code-block:: python
  :caption: Python Code
  :linenos:

    """
    Track Vibration Analysis with Custom Support Arrangement using Rolland API

    This example demonstrates how to set up and analyze a ballasted track with
    periodic support arrangements (alternating sleeper distances and pad stiffness).
    """

    from matplotlib import pyplot as plt
    from rolland import DiscrPad, Sleeper, Ballast
    from rolland.database.rail.db_rail import UIC60
    from rolland import ArrangedBallastedSingleRailTrack
    from rolland import PeriodicArrangement
    from rolland import CFSPML
    from rolland import GaussianImpulse
    from rolland import Deflection
    from rolland import DomSetup
    from rolland.postprocessing import TrackResponse

    # 1. TRACK & ARRANGEMENT DEFINITION -------------------------------------------
    rail = UIC60

    pad_A = DiscrPad(
        # Stiffness [N/m]
        sp_z=120e6, sp_y=40e6, sp_x=40e6,
        # Damping loss factors [-]
        etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2,
        # Geometry [m]
        wdthp=0.15
    )

    pad_B = DiscrPad(
        # Stiffness [N/m]
        sp_z=60e6, sp_y=20e6, sp_x=20e6,
        # Damping loss factors [-]
        etap_z=0.2, etap_y=0.2, etap_x=0.2, etap_r=0.2,
        # Geometry [m]
        wdthp=0.15
    )

    sleeper = Sleeper(
        # Mass [kg] and Density [kg/m^3]
        ms=300.05, rhos=2648,
        # Inertia [kg*m^2]
        Is_x=0.0593, Is_y=0.00089, Is_z=0.0596,
        # Geometry [m]
        lengs=2.5, wdths=0.245, heights=0.185, z_st=-0.0925, z_sb=0.0925
    )

    ballast = Ballast(
        # Stiffness [N/m]
        sb_z=120e6, sb_y=120e6, sb_x=120e6,
        # Damping loss factors [-]
        etab_z=1.0, etab_y=2.0, etab_x=2.0, etab_r=2.0
    )

    track = ArrangedBallastedSingleRailTrack(
        rail=rail,
        pad=PeriodicArrangement(item=[pad_A, pad_B]),
        sleeper=PeriodicArrangement(item=[sleeper]),
        ballast=PeriodicArrangement(item=[ballast]),
        distance=PeriodicArrangement(item=[0.7, 0.5]),
        z_f=81e-3,
        y_f=0,
        num_mount=100,
    )

    # 2. BOUNDARY & EXCITATIONS ---------------------------------------------------
    bound = CFSPML()
    exc_vert = GaussianImpulse(x_excit=30.3)
    exc_lat = GaussianImpulse(x_excit=30.3, force_dir="lateral", z_e=-71e-3)

    # 3. DISCRETIZATION & SIMULATION ----------------------------------------------
    discr = DomSetup(
        track=track,
        bound=bound,
        req_simt=0.5,
    )

    # 3.1 Run vertical deflection simulations
    defl_vert_excit = Deflection(discr=discr, excit=exc_vert, store="excit")
    defl_vert_dist = Deflection(discr=discr, excit=exc_vert, store="observe", obs_pos=40.3)

    # 3.2 Run lateral deflection simulations
    defl_lat_excit = Deflection(discr=discr, excit=exc_lat, store="excit")
    defl_lat_dist = Deflection(discr=discr, excit=exc_lat, store="observe", obs_pos=40.3)

    # 4. POSTPROCESSING & VISUALIZATION -------------------------------------------
    plt.figure(figsize=(12, 10))

    # 4.1 Process & Plot Vertical Mobility at Excitation Point
    ax1 = plt.subplot(2, 2, 1)
    response_vert_excit = TrackResponse(result=defl_vert_excit)
    response_vert_excit.show(ax=ax1, label="Vertical (x = 30.3 m)")
    ax1.set_title("Vertical Mobility (Excitation Point)")

    # 4.2 Process & Plot Vertical Mobility at 10 m Distance
    ax2 = plt.subplot(2, 2, 2)
    response_vert_dist = TrackResponse(result=defl_vert_dist)
    response_vert_dist.show(ax=ax2, label="Vertical (x = 40.3 m)")
    ax2.set_title("Vertical Mobility (10 m Distance)")

    # 4.3 Process & Plot Lateral Mobility at Excitation Point
    ax3 = plt.subplot(2, 2, 3)
    response_lat_excit = TrackResponse(result=defl_lat_excit)
    response_lat_excit.show(ax=ax3, label="Lateral (x = 30.3 m)")
    ax3.set_title("Lateral Mobility at Rail Head (Excitation Point)")

    # 4.4 Process & Plot Lateral Mobility at 10 m Distance
    ax4 = plt.subplot(2, 2, 4)
    response_lat_dist = TrackResponse(result=defl_lat_dist)
    response_lat_dist.show(ax=ax4, label="Lateral (x = 40.3 m)")
    ax4.set_title("Lateral Mobility at Rail Head (10 m Distance)")

    plt.tight_layout()
    plt.show()



.. image:: ../../images/example_variation.png
   :width: 700px
   :align: center


