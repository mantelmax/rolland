.. _different_tracks:

Compare Different Tracks
=========================
This example demonstrates how to set up and run a basic simulation using the Rolland library to calculate the
frequency response of different railway tracks. The example includes continuous and discrete tracks with different
with a single or double layer. See :cite:`thompson2024j` for more information.


.. code-block:: python
  :caption: Python Code
  :linenos:

    """
    Comparative Track Vibration Analysis using Rolland API

    This example demonstrates a comparison of vibration characteristics for:
        1. Continuous slab track
        2. Continuous ballasted track
        3. Discrete slab track
        4. Discrete ballasted track

    All tracks are excited with the same Gaussian impulse at 71.7m position.
    """

    # Import required components from Rolland library
    from rolland.postprocessing import Response as resp
    from rolland.postprocessing import TDR
    from rolland import (
        ContSlabSingleRailTrack,
        ContBallastedSingleRailTrack,
        SimplePeriodicSlabSingleRailTrack,
        SimplePeriodicBallastedSingleRailTrack
    )
    from rolland import DiscrPad, Sleeper, Ballast, ContPad, Slab
    from rolland import PMLRailDampVertic, DiscretizationEBBVerticConst
    from rolland import DeflectionEBBVertic, GaussianImpulse
    from rolland.database.rail.db_rail import UIC60  # Standard rail profile

    # 1. TRACK DEFINITIONS ---------------------------------------------------------

    # 1.1 Continuous slab track
    cont_slab_track = ContSlabSingleRailTrack(
        rail=UIC60,                                     # Standard UIC60 rail profile
        pad=ContPad(sp=[300e6, 0], dp=[30000, 0]),      # Stiffness [N/m], damping [Ns/m]
        l_track=145.2                                   # Track length [m]
    )

    # 1.2 Continuous ballasted track
    cont_ballasted_track = ContBallastedSingleRailTrack(
        rail=UIC60,
        pad=ContPad(sp=[300e6, 0], dp=[30000, 0]),
        slab=Slab(ms=250),                              # Slab mass [kg]
        ballast=Ballast(sb=[100e6, 0], db=[80000, 0]),  # Ballast properties
        l_track=145.2
    )

    # 1.3 Discrete slab track (equally spaced mounting positions)
    discr_slab_track = SimplePeriodicSlabSingleRailTrack(
        rail=UIC60,
        pad=DiscrPad(sp=[180e6, 0], dp=[30000, 0]),
        num_mount=243,                                  # Number of mounting positions
        distance=0.6                                    # sleeper spacing [m]
    )

    # 1.4 Discrete ballasted track (equally spaced mounting positions)
    discr_ballasted_track = SimplePeriodicBallastedSingleRailTrack(
        rail=UIC60,
        pad=DiscrPad(sp=[180e6, 0], dp=[18000, 0]),
        sleeper=Sleeper(ms=150),                        # Sleeper mass [kg]
        ballast=Ballast(sb=[105e6, 0], db=[48000, 0]),
        num_mount=243,
        distance=0.6
    )

    # 2. BOUNDARY CONDITIONS ------------------------------------------------------
    # Perfectly Matched Layer absorbing boundaries
    bound = PMLRailDampVertic(l_bound=33.0)             # 33.0 m boundary domain


    # 3. EXCITATION SETUP ---------------------------------------------------------
    # Gaussian impulse at the same position for all tracks
    x_excit = 71.7                                      # Excitation position [m]
    excit = GaussianImpulse(x_excit=x_excit)

    # 4. SIMULATION SETUP & EXECUTION ----------------------------------------------
    # Discretize and simulate each track type
    discr1 = DiscretizationEBBVerticConst(track=cont_slab_track, bound=bound)
    discr2 = DiscretizationEBBVerticConst(track=cont_ballasted_track, bound=bound)
    discr3 = DiscretizationEBBVerticConst(track=discr_slab_track, bound=bound)
    discr4 = DiscretizationEBBVerticConst(track=discr_ballasted_track, bound=bound)

    defl1 = DeflectionEBBVertic(discr=discr1, excit=excit)
    defl2 = DeflectionEBBVertic(discr=discr2, excit=excit)
    defl3 = DeflectionEBBVertic(discr=discr3, excit=excit)
    defl4 = DeflectionEBBVertic(discr=discr4, excit=excit)

    # 5. POSTPROCESSING & COMPARISON ----------------------------------------------
    # 5.1 Calculate frequency responses for each track at the excitation point
    pp1 = resp(results=defl1)  # Continuous slab
    pp2 = resp(results=defl2)  # Continuous ballasted
    pp3 = resp(results=defl3)  # Discrete slab
    pp4 = resp(results=defl4)  # Discrete ballasted

    resp.plot(
        [(pp1.freq, abs(pp1.mob)),
         (pp2.freq, abs(pp2.mob)),
         (pp3.freq, abs(pp3.mob)),
         (pp4.freq, abs(pp4.mob))],

        ['ContSlabSingleRailTrack',
         'ContBallastedSingleRailTrack',
         'SimplePeriodicSlabSingleRailTrack',
         'SimplePeriodicBallastedSingleRailTrack'],

        title='Frequency Response',
        x_label='Frequency [Hz]',
        y_label='Mobility [m/Ns]',
    )

    # 5.2 Calculate Track Decay Rate (TDR) for each track
    tdr1 = TDR(results=defl1)
    tdr2 = TDR(results=defl2)
    tdr3 = TDR(results=defl3)
    tdr4 = TDR(results=defl4)

    TDR.plot([(tdr1.freq, tdr1.tdr), (tdr2.freq, tdr2.tdr), (tdr3.freq, tdr3.tdr), (tdr4.freq, tdr4.tdr)],
         ['ContSlabSingleRailTrack',
          'ContBallastedSingleRailTrack',
          'SimplePeriodicSlabSingleRailTrack',
          'SimplePeriodicBallastedSingleRailTrack'],
          'Track-Decay-Rate', 'f [Hz]', 'TDR [dB/m]', plot_type='loglog')



.. image:: ../images/example_different_tracks.png
   :width: 700px
   :align: center

.. image:: ../images/example_different_tracks_tdr.png
   :width: 700px
   :align: center
