"""Tests for analytical methods in rolland package."""

import csv
import dataclasses

import numpy as np
import pytest
from numpy import linspace

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
from rolland.methods import (
    EBBCont1L,
    EBBCont2L,
    TSBDiscr1L,
    TSBDiscr2L,
)

# Constants
CSV_FILE_PATH = 'tests/data/data_analytical_methods.csv'
FREQUENCY_RANGE = linspace(20, 3000, 1500)
FORCE = 1
X_POSITION = 0
X_EXCIT = 240 * 0.3

REFERENCE_RAIL = dataclasses.replace(
    UIC60, G=81e9, kapz=0.4, kapy=0.54, rho=7850, Iyr=3038.30e-8, Ipr=3550.60e-8,
)


RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 0.0

# Sleeper/slab geometry.
SL_LEN, SL_WIDTH, SL_HEIGHT, RHOS = 2.5, 0.245, 0.185, 2648
_M = RHOS * SL_LEN * SL_WIDTH * SL_HEIGHT
_GEOMETRY = {
    'Is_x': (SL_LEN**2 + SL_HEIGHT**2) * _M / 12 / RHOS,
    'Is_y': (SL_HEIGHT**2 + SL_WIDTH**2) * _M / 12 / RHOS,
    'Is_z': (SL_LEN**2 + SL_WIDTH**2) * _M / 12 / RHOS,
    'rhos': RHOS,
    'lengs': SL_LEN,
    'heights': SL_HEIGHT,
    'z_st': -SL_HEIGHT / 2,
    'z_sb': SL_HEIGHT / 2,
    'equi_sm': False,
}


def _viscous_pad():
    """Pad for the Euler-Bernoulli methods, which use viscous damping."""
    return ContPad(sp_z=300e6, sp_y=0.0, sp_x=0.0,
                   dp_z=30000, dp_y=0.0, dp_x=0.0, dp_xr=0.0, wdthp=0.0)


def _hysteretic_pad():
    """Pad for the Timoshenko methods, which use loss factors."""
    return DiscrPad(sp_z=300e6, sp_y=0.0, sp_x=0.0,
                    etap_z=0.25, etap_y=0.0, etap_x=0.0, etap_r=0.0, wdthp=0.0)


@pytest.fixture(scope="module")
def load_csv_data():
    """Load test data from a CSV file."""
    data = {}
    try:
        with open(CSV_FILE_PATH) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                freq = float(row['Frequency'])
                data[freq] = {key: float(row[key]) for key in reader.fieldnames if key != 'Frequency'}
    except FileNotFoundError:
        pytest.fail(f"Test data file not found: {CSV_FILE_PATH}")
    except csv.Error as e:
        pytest.fail(f"CSV parsing error: {e}")
    except ValueError as e:
        pytest.fail(f"Data format error in CSV: {e}")
    return data


@pytest.fixture(scope="module")
def tracks():
    """Create different types of tracks for testing."""
    return {
        'track_cont_slab': ContSlabSingleRailTrack(
            rail=REFERENCE_RAIL,
            pad=_viscous_pad(),
            slab=Slab(ms=250, equ_wdths=SL_WIDTH, **_GEOMETRY),
            z_f=81e-3,
            y_f=0.0,
        ),
        'track_cont_ball': ContBallastedSingleRailTrack(
            rail=REFERENCE_RAIL,
            pad=_viscous_pad(),
            slab=Slab(ms=250, equ_wdths=SL_WIDTH, **_GEOMETRY),
            ballast=Ballast(sb_z=100e6, sb_y=0.0, sb_x=0.0,
                            db_z=80000, db_y=0.0, db_x=0.0,
                            db_xr=0.0, db_yr=0.0, db_zr=0.0),
            z_f=81e-3,
            y_f=0.0,
        ),
        'track_discr_slab': SimplePeriodicSlabSingleRailTrack(
            rail=REFERENCE_RAIL,
            pad=_hysteretic_pad(),
            slab=Slab(ms=162, equ_wdths=SL_WIDTH, **_GEOMETRY),
            num_mount=241,
            distance=0.6,
            z_f=81e-3,
            y_f=0.0,
        ),
        'track_discr_ball': SimplePeriodicBallastedSingleRailTrack(
            rail=REFERENCE_RAIL,
            pad=_hysteretic_pad(),
            sleeper=Sleeper(ms=162, wdths=SL_WIDTH, **_GEOMETRY),
            ballast=Ballast(sb_z=50e6, sb_y=0.0, sb_x=0.0,
                            etab_z=1.0, etab_y=0.0, etab_x=0.0, etab_r=0.0),
            num_mount=241,
            distance=0.6,
            z_f=81e-3,
            y_f=0.0,
        ),
    }


@pytest.fixture(scope="module")
def methods(tracks):
    """Create analytical methods for testing."""
    return [
        EBBCont1L(track=tracks['track_cont_slab'], f=FREQUENCY_RANGE, force=FORCE, x=X_POSITION),
        EBBCont2L(track=tracks['track_cont_ball'], f=FREQUENCY_RANGE, force=FORCE, x=X_POSITION),
        TSBDiscr2L(track=tracks['track_discr_ball'], f=FREQUENCY_RANGE, force=FORCE, x=X_EXCIT,
                      x_excit=X_EXCIT),
        TSBDiscr1L(track=tracks['track_discr_slab'], f=FREQUENCY_RANGE, force=FORCE, x=X_EXCIT,
                      x_excit=X_EXCIT),
    ]


@pytest.mark.parametrize("method_name", [
    'EBBCont1L',
    'EBBCont2L',
    'TSBDiscr2L',
    'TSBDiscr1L',
])
def test_analytical_methods(method_name, methods, load_csv_data):
    """Test analytical methods against precomputed data."""
    method = next((m for m in methods if m.__class__.__name__ == method_name), None)

    if method is None:
        pytest.fail(f"Method {method_name} not found in the created methods.")

    method.compute_mobility()
    for i, freq in enumerate(method.f):
        expected_value = load_csv_data[freq][method_name]
        actual_value = abs(method.mobility[i])
        assert np.isclose(actual_value, expected_value,
                          rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE), \
            f"Mismatch in {method_name} at frequency {freq}: expected {expected_value}, got {actual_value}"


def test_damping_mode_mismatch_is_rejected(tracks):
    """A track built for one damping formulation must not silently feed the other."""
    with pytest.raises(ValueError, match="requires viscous damping"):
        EBBCont1L(track=tracks['track_discr_slab'], f=FREQUENCY_RANGE, force=FORCE, x=X_POSITION)

    with pytest.raises(ValueError, match="requires hysteretic damping"):
        TSBDiscr1L(track=tracks['track_cont_slab'], f=FREQUENCY_RANGE, force=FORCE, x=X_POSITION)
