"""Wheel type and Green's function instances.

This module contains instances of the Wheel class representing different wheel types.

Wheel types:
    - w_type_a
    - w_type_b

Green's functions:
    - gf_w_type_a
    - gf_w_type_b


Attributes (Wheel)
----------
    w_geo (list): Wheel geometry coordinates [m].
    w_prof (str): Wheel running surface profile.
    w_geo_cross_sec (list): Wheel cross-sectional geometry coordinates [m].
    mw (float): Wheel mass [kg].
    mw_red (float): Reduced wheel mass (lateral dynamics) [kg].
    rw (float): Wheel radius [m].

Attributes (WheelGreensfunc)
----------
    w_gf (array): Green's function
    w_gf_freq (array): Green's function frequency values [Hz].
"""

from numpy import array

from rolland.components import Wheel, WheelGreensfunc

w_type_a = Wheel(
    w_geo=[(0.0, 0.0)],
    w_prof="S1002",
    w_geo_cross_sec=[(0.0, 0.0)],
    mw=400.0,
    mw_red=5.0,
    rw=0.46,
)

w_type_b = Wheel(
    w_geo=[(0.0, 0.0)],
    w_prof="S1002",
    w_geo_cross_sec=[(0.0, 0.0)],
    mw=450.0,
    mw_red=5.5,
    rw=0.48,
)

gf_w_type_a = WheelGreensfunc(
    w_gf=array(([0.0, 0.0], [0.0, 0.0])),
    w_gf_freq=array([0.0, 0.0]),
)

gf_w_type_b = WheelGreensfunc(
    w_gf=array(([0.0, 0.0], [0.0, 0.0])),
    w_gf_freq=array([0.0, 0.0]),
)
