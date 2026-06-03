"""Tests for analytical methods in rolland package."""

import csv

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
    EBBCont1LSupp,
    EBBCont2LSupp,
    TSDiscr1LSupp,
    TSDiscr2LSupp,
)

# Constants
CSV_FILE_PATH = 'tests/data/data_analytical_methods.csv'
FREQUENCY_RANGE = linspace(20, 3000, 1500)
FORCE = 1
X_POSITION = 0
X_EXCIT = 240 * 0.3
RELATIVE_TOLERANCE = 1e-5

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
            rail=UIC60,
            pad=ContPad(sp=[300e6, 0], dp=[30000, 0]),
        ),
        'track_cont_ball': ContBallastedSingleRailTrack(
            rail=UIC60,
            pad=ContPad(sp=[300e6, 0], dp=[30000, 0]),
            slab=Slab(ms=250),
            ballast=Ballast(sb=[100e6, 0], db=[80000, 0]),
        ),
        'track_discr_slab': SimplePeriodicSlabSingleRailTrack(
            rail=UIC60,
            pad=DiscrPad(sp=[300e6, 0], etap=0.25),
            slab=Slab(ms=162),
            num_mount=241,
            distance=0.6,
        ),
        'track_discr_ball': SimplePeriodicBallastedSingleRailTrack(
            rail=UIC60,
            pad=DiscrPad(sp=[300e6, 0], etap=0.25),
            sleeper=Sleeper(ms=162),
            ballast=Ballast(sb=[50e6, 0], etab=1),
            num_mount=241,
            distance=0.6,
        ),
    }

@pytest.fixture(scope="module")
def methods(tracks):
    """Create analytical methods for testing."""
    return [
        EBBCont1LSupp(track=tracks['track_cont_slab'], f=FREQUENCY_RANGE, force=FORCE, x=X_POSITION),
        EBBCont2LSupp(track=tracks['track_cont_ball'], f=FREQUENCY_RANGE, force=FORCE, x=X_POSITION),
        TSDiscr2LSupp(track=tracks['track_discr_ball'], f=FREQUENCY_RANGE, force=FORCE, x=X_EXCIT,
                      x_excit=X_EXCIT),
        TSDiscr1LSupp(track=tracks['track_discr_slab'], f=FREQUENCY_RANGE, force=FORCE, x=X_EXCIT,
                      x_excit=X_EXCIT),
    ]

@pytest.mark.parametrize("method_name", [
    'EBBCont1LSupp',
    'EBBCont2LSupp',
    'TSDiscr2LSupp',
    'TSDiscr1LSupp',
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
        assert np.isclose(actual_value, expected_value, rtol=RELATIVE_TOLERANCE), \
            f"Mismatch in {method_name} at frequency {freq}: expected {expected_value}, got {actual_value}"
