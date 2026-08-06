"""Tests for FDM Stampka methods."""

import csv
import os

import numpy as np
import pytest

from rolland import (
    Ballast,
    ContBallastedSingleRailTrack,
    ContPad,
    ContSlabSingleRailTrack,
    DiscrPad,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
    Slab,
    Sleeper,
)
from rolland.database.rail.db_rail import UIC60
from rolland.methods.numerical import (
    DeflectionEBBVertic,
    DiscretizationEBBVerticConst,
    GaussianImpulse,
    PMLRailDampVertic,
)
from rolland.postprocessing import PointResponse

RELATIVE_TOLERANCE = 1e-9

# atol must be 0. numpy's default atol=1e-8 is of the same order as the mobilities
# themselves, so it would make the comparison pass regardless of the values.
ABSOLUTE_TOLERANCE = 0.0

# Frequency band of the reference data.
F_MIN = 100.0
F_MAX = 3000.0

# Mapping between CSV keys and test keys
CSV_KEY_MAPPING = {
    'ContSlabSingleRailTrack': 'mob_cont_slab',
    'ContBallastedSingleRailTrack': 'mob_cont_ball',
    'SimplePeriodicSlabSingleRailTrack': 'mob_discr_slab',
    'SimplePeriodicBallastedSingleRailTrack': 'mob_discr_ball',
}

contpad = ContPad(
    sp_z=300 * 10**6,
    sp_y=0.0,
    sp_x=0.0,
    dp_z=30000,
    dp_y=0.0,
    dp_x=0.0,
    dp_xr=0.0,
    wdthp=0.0,
)

discrpad = DiscrPad(
    sp_z=180 * 10**6,
    sp_y=0.0,
    sp_x=0.0,
    dp_z=30000,
    dp_y=0.0,
    dp_x=0.0,
    dp_xr=0.0,
    wdthp=0.0,
)

discrpad_ballasted = DiscrPad(
    sp_z=180 * 10**6,
    sp_y=0.0,
    sp_x=0.0,
    dp_z=18000,
    dp_y=0.0,
    dp_x=0.0,
    dp_xr=0.0,
    wdthp=0.0,
)

slep_dist = 0.6
sl_len = 2.5
sl_width = 0.245
sl_height = 0.185

rhos = 2648
m = rhos * sl_len * sl_width * sl_height

I_sz = (sl_len**2 + sl_width**2) * m / 12 * 1 / rhos
I_sy = (sl_height**2 + sl_width**2) * m / 12 * 1 / rhos
I_sx = (sl_len**2 + sl_height**2) * m / 12 * 1 / rhos

sleeper = Sleeper(
    ms=150,
    rhos=rhos,
    Is_x=I_sx,
    Is_y=I_sy,
    Is_z=I_sz,
    lengs=sl_len,
    wdths=sl_width,
    heights=sl_height,
    z_st=-sl_height / 2,
    z_sb=sl_height / 2,
    equi_sm=False,
)

slab = Slab(
    ms=250,
    Is_x=I_sx / slep_dist,
    Is_y=I_sy / slep_dist,
    Is_z=I_sz / slep_dist,
    lengs=sl_len,
    rhos=rhos,
    equ_wdths=sl_width,
    heights=sl_height,
    z_st=-sl_height / 2,
    z_sb=sl_height / 2,
    equi_sm=False,
)

contballast = Ballast(
    sb_z=100 * 10**6,
    sb_y=0.0,
    sb_x=0.0,
    db_z=80000,
    db_y=0.0,
    db_x=0.0,
    db_xr=0.0,
    db_yr=0.0,
    db_zr=0.0,
)

discrballast = Ballast(
    sb_z=105 * 10**6,
    sb_y=0.0,
    sb_x=0.0,
    db_z=48000,
    db_y=0.0,
    db_x=0.0,
    db_xr=0.0,
    db_yr=0.0,
    db_zr=0.0,
)


@pytest.fixture(scope='module')
def tracks():
    """Create track instances for testing."""
    return {
        'track_cont_slab': ContSlabSingleRailTrack(
            rail=UIC60,
            pad=contpad,
            l_track=90,
            z_f=81 * 10**-3,
            y_f=0,
        ),
        'track_cont_ball': ContBallastedSingleRailTrack(
            rail=UIC60,
            pad=contpad,
            slab=slab,
            ballast=contballast,
            l_track=90,
            z_f=81 * 10**-3,
            y_f=0,
        ),
        'track_discr_slab': SimplePeriodicSlabSingleRailTrack(
            rail=UIC60,
            pad=discrpad,
            num_mount=int(90 / 0.6),
            distance=0.6,
            z_f=81 * 10**-3,
            y_f=0,
        ),
        'track_discr_ball': SimplePeriodicBallastedSingleRailTrack(
            rail=UIC60,
            pad=discrpad_ballasted,
            sleeper=sleeper,
            ballast=discrballast,
            num_mount=int(90 / 0.6),
            distance=0.6,
            z_f=81 * 10**-3,
            y_f=0,
        ),
    }


@pytest.fixture(scope='module')
def deflections(tracks):
    """Create deflection instances for testing."""
    bounds = {
        'bound1': PMLRailDampVertic(l_bound=32.73),
        'bound2': PMLRailDampVertic(l_bound=32.73),
        'bound3': PMLRailDampVertic(l_bound=32.73),
        'bound4': PMLRailDampVertic(l_bound=32.73),
    }

    forces = {
        'force1': GaussianImpulse(x_excit=45.3),
        'force2': GaussianImpulse(x_excit=45.3),
        'force3': GaussianImpulse(x_excit=45.3),
        'force4': GaussianImpulse(x_excit=45.3),
    }

    discretizations = {
        'discr1': DiscretizationEBBVerticConst(
            track=tracks['track_cont_slab'],
            bound=bounds['bound1'],
            dt=2e-5,
            req_simt=0.4,
            bx=1,
        ),
        'discr2': DiscretizationEBBVerticConst(
            track=tracks['track_cont_ball'],
            bound=bounds['bound2'],
            dt=2e-5,
            req_simt=0.4,
            bx=1,
        ),
        'discr3': DiscretizationEBBVerticConst(
            track=tracks['track_discr_slab'],
            bound=bounds['bound3'],
            dt=2e-5,
            req_simt=0.4,
            bx=1,
        ),
        'discr4': DiscretizationEBBVerticConst(
            track=tracks['track_discr_ball'],
            bound=bounds['bound4'],
            dt=2e-5,
            req_simt=0.4,
            bx=1,
        ),
    }

    return {
        'mob_cont_slab': DeflectionEBBVertic(
            discr=discretizations['discr1'],
            excit=forces['force1'],
        ),
        'mob_cont_ball': DeflectionEBBVertic(
            discr=discretizations['discr2'],
            excit=forces['force2'],
        ),
        'mob_discr_slab': DeflectionEBBVertic(
            discr=discretizations['discr3'],
            excit=forces['force3'],
        ),
        'mob_discr_ball': DeflectionEBBVertic(
            discr=discretizations['discr4'],
            excit=forces['force4'],
        ),
    }


@pytest.fixture(scope='module')
def mobility_results(deflections):
    """Compute mobility results for testing."""
    results = {}
    for key, deflection in deflections.items():
        # Point mobility: response taken at the excitation point itself. The
        # deflection is stored as (n_time, n_positions), so the position is the
        # second index.
        signal = deflection.deflection[:, deflection.ind_excit]
        fftfre, mob = PointResponse.calculate_mobility_1d(
            signal, deflection.force, deflection.discr.dt,
        )
        band = (fftfre > F_MIN) & (fftfre <= F_MAX)
        results[key] = (fftfre[band], mob[band])
    return results


@pytest.fixture(scope='module')
def csv_data():
    """Load precomputed mobility data from CSV."""
    data = {}
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'data_fdm_stampka.csv')
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            freq = float(row['Frequency'])
            # Map CSV keys to test keys
            mapped_row = {CSV_KEY_MAPPING[key]: float(value) for key, value in row.items() if key != 'Frequency'}
            data[freq] = mapped_row
    return data


@pytest.mark.parametrize(
    'mobility_name',
    [
        'mob_cont_slab',
        'mob_cont_ball',
        'mob_discr_slab',
        'mob_discr_ball',
    ],
)
def test_fdm_stampka_methods(mobility_name, mobility_results, csv_data):
    """Test FDM Stampka methods against precomputed mobility data."""
    fftfre, mob = mobility_results[mobility_name]

    for i, freq in enumerate(fftfre):
        expected_value = csv_data[freq][mobility_name]
        actual_value = abs(mob[i])
        assert np.isclose(
            actual_value,
            expected_value,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        ), f'Mismatch in {mobility_name} at frequency {freq}: expected {expected_value}, got {actual_value}'
