User Guide
==========

.. note::
   A detailed User Guide featuring interactive Jupyter Notebooks and the comprehensive mathematical and physical theory behind **Rolland** will be added shortly.


Getting Started
---------------

This guide walks through the basic structure of writing a Rolland simulation
based on the example scripts that are provided in the "examples" folder.

Prerequisites
~~~~~~~~~~~~~

This tutorial assumes that Rolland is :doc:`installed </install/index>`
together with its dependencies.

The full runnable examples can be found in:

* :doc:`examples </examples/index>`

Simulation Structure
~~~~~~~~~~~~~~~~~~~~
Creating a simulation in Rolland is structured in the following steps:

1. **Define the components**
2. **Define the track model**
3. **Define the boundary conditions, Domain Setup and Excitations**
4. **Run the simulation**
5. **Post-process the results**

Simulation Example Step-by-Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A common application of Rolland is to determine a vertical frequency response of a railway track.
The example
:doc:`quick_start </examples/track_response/quick_start>`
contains the full runnable script examining a ballasted track with periodic sleepers.

First, import the required packages:

.. literalinclude:: ../examples/track_response/quick_start.rst
   :language: python
   :dedent: 4
   :start-at: from matplotlib import pyplot as plt
   :end-before: # 1. PARAMETERS DEFINITION -----------------------------------------------------

The fundamental step is to define the components of the track, which are the rail, the sleepers, the pads and the ballast.
A variety of different typical rail types are available in the database module.
The following code snippet shows how to define a discrete pad, sleeper and ballast:

.. literalinclude:: ../examples/track_response/quick_start.rst
   :language: python
   :dedent: 4
   :start-after: # 1. PARAMETERS DEFINITION -----------------------------------------------------
   :end-before: # 2. TRACK DEFINITIONS ---------------------------------------------------------

Using the defined components, a track model can be created. The track classes serve as an assembly of the components.
Note that some parameters of the components, such as viscous damping coefficients, are calculated automatically on track instantiation
and are not available before the track is instantiated.

.. literalinclude:: ../examples/track_response/quick_start.rst
   :language: python
   :dedent: 4
   :start-after: # 2. TRACK DEFINITIONS ---------------------------------------------------------
   :end-before: # 3. BOUNDARY & EXCITATION ----------------------------------------------------

The next step is the definition of the boundary conditions, domain setup and excitations, which provide all the missing information required for the simulation.
The domain setup combines the track model and boundary condition into a calculation domain for the FDM simulation. The excitation is applied to the track when the simulation is run.

.. literalinclude:: ../examples/track_response/quick_start.rst
   :language: python
   :dedent: 4
   :start-after: # 3. BOUNDARY & EXCITATION ----------------------------------------------------
   :end-before: # 4.1 Deflection at excitation point

The simulation is run by using the ``Deflection`` class. It can be run in several different ways, depending on the desired results.
In the following example, the deflection is calculated at the excitation point using the ``store="excit"`` option and it is also calculated at
10 m distance using the ``store="observe"`` option.

.. literalinclude:: ../examples/track_response/quick_start.rst
   :language: python
   :dedent: 4
   :start-after: # 4.1 Deflection at excitation point
   :end-before: # 4.3 Compute frequency responses and plot

In the last step, the results are post-processed and plotted. The ``TrackResponse`` class is used to compute the frequency response
of the track at the excitation point and at the observation point.

.. literalinclude:: ../examples/track_response/quick_start.rst
   :language: python
   :dedent: 4
   :start-after: # 4.3 Compute frequency responses and plot
   :end-at: plt.show()

The explained skript leads to the results shown in the following figure, which shows the frequency response of a ballasted track with periodic sleepers at the excitation point and at 10 m distance:

.. image:: ../images/example_quick_start.png
   :width: 700px
   :align: center
