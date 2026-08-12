.. _quick_start:

First Simulation
================

This example determines the track response of a double layer track with discrete mounting positions.
The track is excited between two sleepers by a Gaussian impulse.


.. code-block:: python
  :caption: Python Code
  :linenos:

    """
    Example: Track Vibration Analysis using Rolland API

    This example demonstrates how to:
        1. Create a railway track model
        2. Apply excitation and boundary conditions
        3. Run a vibration simulation
        4. Analyze and plot mobility results
    """

    from matplotlib import pyplot as plt
    from rolland import DiscrPad, Sleeper, Ballast
    from rolland.database.rail.db_rail import UIC60
    from rolland.track import SimplePeriodicBallastedSingleRailTrack
    from rolland.boundary import CFSPML
    from rolland.excitation import GaussianImpulse
    from rolland.deflection import Deflection
    from rolland.domainsetup import DomSetup
    from rolland.postprocessing import DEVITO_PP

    # 1. TRACK DEFINITION ----------------------------------------------------------
    track = SimplePeriodicBallastedSingleRailTrack(
        rail=UIC60,
        pad=DiscrPad(
            sp_z=120e6,
            sp_y=40e6,
            sp_x=40e6,
            etap_z=0.2,
            etap_y=0.2,
            etap_x=0.2,
            etap_r=0.2,
            wdthp=0.15,
        ),
        sleeper=Sleeper(
            ms=300.05,
            rhos=2648,
            Is_x=0.0593,
            Is_y=0.00089,
            Is_z=0.0596,
            lengs=2.5,
            wdths=0.245,
            heights=0.185,
            z_st=-0.0925,
            z_sb=0.0925,
        ),
        ballast=Ballast(
            sb_z=120e6,
            sb_y=120e6,
            sb_x=120e6,
            etab_z=1.0,
            etab_y=2.0,
            etab_x=2.0,
            etab_r=2.0,
        ),
        z_f=81e-3,
        y_f=0,
        num_mount=100,
    )

    # 2. BOUNDARY & EXCITATION ----------------------------------------------------
    bound = CFSPML()
    exc_vert = GaussianImpulse(x_excit=30.3)

    # 3. DISCRETIZATION & SIMULATION ----------------------------------------------
    discr = DomSetup(
        track=track,
        bound=bound,
        req_simt=0.5,
    )

    # Deflection at excitation point
    defl_excit = Deflection(
        discr=discr,
        excit=exc_vert,
        store="excit",
    )

    # Deflection at 10 m distance (30.3 + 10 = 40.3 m)
    defl_dist = Deflection(
        discr=discr,
        excit=exc_vert,
        store="observe",
        obs_pos=40.3,
    )

    # 4. POSTPROCESSING & VISUALIZATION -------------------------------------------
    freq, mob_excit = DEVITO_PP.calculate_mobility(defl_excit.u_z_obs, exc_vert.force, discr)
    _, mob_dist = DEVITO_PP.calculate_mobility(defl_dist.u_z_obs, exc_vert.force, discr)

    plt.figure(figsize=(10, 8))

    # Subplot 1: Response at excitation point
    plt.subplot(2, 1, 1)
    plt.loglog(freq[:discr.nt // 2], abs(mob_excit[:discr.nt // 2]), label="At Excitation Point (x = 30.3 m)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Mobility [m/sN]")
    plt.title("Vertical Frequency Response (Excitation Point)")
    plt.xlim(50, 6000)
    plt.grid(True, which="both")
    plt.legend()

    # Subplot 2: Response at 10 m distance
    plt.subplot(2, 1, 2)
    plt.loglog(freq[:discr.nt // 2], abs(mob_dist[:discr.nt // 2]), "r", label="At 10 m Distance (x = 40.3 m)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Mobility [m/sN]")
    plt.title("Vertical Frequency Response (10 m Distance)")
    plt.xlim(50, 6000)
    plt.grid(True, which="both")
    plt.legend()

    plt.tight_layout()
    plt.show()


