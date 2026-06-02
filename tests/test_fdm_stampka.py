"""Tests for FDM Stampka methods."""

import csv

import numpy as np
import pytest

from rolland import (
    Ballast,
    ContBallastedSingleRailTrack,
    ContPad,
    ContSlabSingleRailTrack,
    DeflectionEBBVertic,
    DiscretizationEBBVerticConst,
    DiscrPad,
    GaussianImpulse,
    PMLRailDampVertic,
    SimplePeriodicBallastedSingleRailTrack,
    SimplePeriodicSlabSingleRailTrack,
    Slab,
    Sleeper,
)
from rolland.database.rail.db_rail import UIC60
from rolland.postprocessing import Response

RELATIVE_TOLERANCE = 1e-5

# Mapping between CSV keys and test keys
CSV_KEY_MAPPING = {
    'ContSlabSingleRailTrack': 'mob_cont_slab',
    'ContBallastedSingleRailTrack': 'mob_cont_ball',
    'SimplePeriodicSlabSingleRailTrack': 'mob_discr_slab',
    'SimplePeriodicBallastedSingleRailTrack': 'mob_discr_ball',
}

@pytest.fixture(scope="module")
def tracks():
    """Create track instances for testing."""
    return {
        'track_cont_slab': ContSlabSingleRailTrack(
            rail=UIC60,
            pad=ContPad(sp=[300 * 10**6, 0], dp=[30000, 0]),
            l_track=90,
        ),
        'track_cont_ball': ContBallastedSingleRailTrack(
            rail=UIC60,
            pad=ContPad(sp=[300 * 10**6, 0], dp=[30000, 0]),
            slab=Slab(ms=250),
            ballast=Ballast(sb=[100 * 10**6, 0], db=[80000, 0]),
            l_track=90,
        ),
        'track_discr_slab': SimplePeriodicSlabSingleRailTrack(
            rail=UIC60,
            pad=DiscrPad(sp=[180 * 10**6, 0], dp=[30000, 0]),
            num_mount=int(90/0.6),
            distance=0.6,
        ),
        'track_discr_ball': SimplePeriodicBallastedSingleRailTrack(
            rail=UIC60,
            pad=DiscrPad(sp=[180 * 10**6, 0], dp=[18000, 0]),
            sleeper=Sleeper(ms=150),
            ballast=Ballast(sb=[105 * 10**6, 0], db=[48000, 0]),
            num_mount=int(90/0.6),
            distance=0.6,
        ),
    }

@pytest.fixture(scope="module")
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

@pytest.fixture(scope="module")
def mobility_results(deflections):
    """Compute mobility results for testing."""
    results = {}
    for key, deflection in deflections.items():
        # Compute mobility using the Response class from PostProcessing
        response = Response(results=deflection)
        fftfre, mob = response.freq, response.mob
        results[key] = (fftfre, mob)
    return results

@pytest.fixture(scope="module")
def csv_data():
    """Load precomputed mobility data from CSV."""
    data = {}
    with open('tests/data/data_fdm_stampka.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            freq = float(row['Frequency'])
            # Map CSV keys to test keys
            mapped_row = {
                CSV_KEY_MAPPING[key]: float(value)
                for key, value in row.items()
                if key != 'Frequency'
            }
            data[freq] = mapped_row
    return data

@pytest.mark.parametrize("mobility_name", [
    'mob_cont_slab',
    'mob_cont_ball',
    'mob_discr_slab',
    'mob_discr_ball',
])
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
        ), (
            f"Mismatch in {mobility_name} at frequency {freq}: "
            f"expected {expected_value}, got {actual_value}"
        )
