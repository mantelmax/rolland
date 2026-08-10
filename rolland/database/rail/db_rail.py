"""Rail type instances.

This module contains instances of the Rail class representing different rail types.

Rail types:
    - UIC60
    - UIC54

Attributes
----------
    rl_geo (list): Rail outline coordinates [m].
    E (float): Young's modulus of rail [Pa].
    G (float): Shear modulus of rail [Pa].
    nu (float): Poisson's ratio of rail [-].
    kap (float): Timoshenko shear correction factor [-].
    mr (float): Rail mass per unit length [kg/m].
    gamr (list): Rail shear center [m].
    epsr (list): Center of gravity [m].
    Iyr (float): Area moment of inertia of rail around y-axis [m^4].
    Izr (float): Area moment of inertia of rail around z-axis [m^4].
    Itr (float): Torsional constant of rail [m^4].
    Ar (float): Cross-sectional area of rail [m^2].
    Asr (float): Surface area per unit length of rail [m^2/m].
    Vr (float): Volume per unit length of rail [m^3/m].
"""
import csv
import os

from rolland.components import Rail


def load_rail_geo(file_path):
    """Load rail geometry from pts file."""
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        return [(float(row[0]), float(row[1])) for row in csvreader]


UIC60 = Rail(
    rl_geo=load_rail_geo(os.path.join(os.path.dirname(__file__), 'UIC60.csv')),
    E=210e9,
    G=80.769e9,
    nu=0.3,
    kapz=0.393,
    kapy=0.538,
    mr=60.2,
    rho=7860,
    etar=0.02,
    dr=1000,
    shearc=[0.0, 33e-3],
    centr=[0.0, 0],
    Iyr=3.037e-05,
    Izr=5.127e-06,
    Iyz=0.0,
    Ipr=3.55e-05,
    Itr=0.0, # arbitrary value, check
    Ar=76.70e-4,
    Asr=0.688,
    Vr=7670.00e-6,
    kapp_s = 1,
    Iw = 2.161e-8,
    Iwz = 1.6971e-7,
    Iwy = 0.0,
    k_w = -0.6016,
    J = 2.212e-6,
    chi = 0.0, # TODO: Add full function
)

# UIC54 = Rail(
#     rl_geo=[ # arbitrary values!
#         (0, 4.6), (1, 4.5), (2, 5.3), (3, 4.8), (4, 9.8), (5, 3.7), (6, 7.9), (7, 2.1), (8, 0.5),
#         (9, 1.7),(10, 2.0), (11, 2.6), (12, 8.0), (13, 8.2), (14, 1.8), (15, 8.4), (16, 5.3),
#         (17, 7.9), (18, 5.5),(19, 7.0), (20, 0.6), (21, 3.6), (22, 6.2), (23, 4.3), (24, 1.1),
#         (25, 5.2), (26, 0.0), (27, 3.6),(28, 1.1), (29, 5.3), (30, 0.9), (31, 7.0), (32, 5.9),
#         (33, 8.8), (34, 7.4), (35, 4.4), (36, 2.3),(37, 7.5), (38, 6.9), (39, 2.4), (40, 0.9),
#         (41, 7.8), (42, 8.4), (43, 5.7), (44, 8.3), (45, 5.5), (46, 7.8), (47, 2.0), (48, 4.4),
#         (49, 2.7),
#     ],
#     E=210e9,
#     G=81e9,
#     nu=0.3,
#     kap=[0.4, 0.54],
#     mr=54.0,
#     rho=7850,
#     etar=0.01,
#     fresr=1000,
#     dr=1000,
#     gamr=[0.0, 0.0],
#     epsr=[0.0, 0.0],
#     Iyr=3038.30e-8,
#     Izr=512.30e-8,
#     Itr=209.20e-8,
#     Ipr=3550.60e-8,
#     Ar=76.70e-4,
#     Asr=0.688,
#     Vr=7670.00e-6,
# )

