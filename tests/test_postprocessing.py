"""Tests for postprocessing module."""

import dataclasses

import numpy as np
import pytest

from rolland import ContPad, ContSlabSingleRailTrack, DiscrPad, SimplePeriodicSlabSingleRailTrack, Slab
from rolland.database.rail.db_rail import UIC60
from rolland.methods import EBBCont1L, TSBDiscr1L
from rolland.methods.semi_analytical import TBCont1LKosto
from rolland.postprocessing import DB_CONVERSION_FACTOR, TrackDecayRate, TrackResponse, compute_frf

# TDR measurement positions relative to the excitation [m]
# (continuous tracks and discrete tracks with a regular sleeper spacing of 0.6 m)
X_TDR = np.array(
    [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5.5,
     6.5, 7.5, 8.5, 10.5, 12.5, 16.5, 20.5, 24.5, 30.5, 36.5, 42.5, 48.5, 54.5, 66.5],
) * 0.6 - 0.3


class DummyExcit:
    """Dummy Excitation."""

    def __init__(self):
        self.x_excit = 5.0
        self.force_dir = 'vertical'
        self.force = type('Force', (), {'data': np.random.rand(100)})()


class DummyDiscr:
    """Dummy Discretization."""

    def __init__(self):
        self.dt = 0.01
        self.dx = 1.0


class DummyAnalyticalResult:
    """Dummy Analytical Result."""

    def __init__(self):
        self.f = np.array([10.0, 20.0, 30.0])
        self.mobility = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])


class DummyStampkaResult:
    """Dummy Stampka Result."""

    def __init__(self):
        self.deflection = np.random.rand(100, 100)
        self.force = np.random.rand(100)
        self.discr = DummyDiscr()
        self.discr.dx = 0.6
        self.ind_excit = 5
        self.track = None


class DummyRollandResult:
    """Dummy Rolland Result."""

    def __init__(self):
        self.u_z_obs = np.random.rand(100, 100)
        self.store = 'full'
        self.excit = DummyExcit()
        self.discr = DummyDiscr()
        self.skip = 1
        self.track = None


REFERENCE_RAIL = dataclasses.replace(UIC60, G=81e9, kapz=0.4, kapy=0.54, rho=7850, Iyr=3038.30e-8, Ipr=3550.60e-8)
PAD_PROPS = {'sp_z': 300e6, 'sp_y': 0.0, 'sp_x': 0.0, 'etap_z': 0.25, 'etap_y': 0.0, 'etap_x': 0.0,
             'etap_r': 0.0, 'wdthp': 0.0}


def _slab():
    """Slab for the analytical methods."""
    sl_len, sl_width, sl_height, rhos = 2.5, 0.245, 0.185, 2648
    mass = rhos * sl_len * sl_width * sl_height
    return Slab(
        ms=250, equ_wdths=sl_width, rhos=rhos, lengs=sl_len, heights=sl_height,
        Is_x=(sl_len**2 + sl_height**2) * mass / 12 / rhos,
        Is_y=(sl_height**2 + sl_width**2) * mass / 12 / rhos,
        Is_z=(sl_len**2 + sl_width**2) * mass / 12 / rhos,
        z_st=-sl_height / 2, z_sb=sl_height / 2, equi_sm=False,
    )


def _cont_slab_track():
    """Continuous slab track for the Kostovasilis and EBB methods."""
    return ContSlabSingleRailTrack(rail=REFERENCE_RAIL, pad=ContPad(**PAD_PROPS), slab=_slab(), z_f=81e-3, y_f=0.0)


def _discr_slab_track():
    """Discrete slab track with 241 supports every 0.6 m."""
    return SimplePeriodicSlabSingleRailTrack(
        rail=REFERENCE_RAIL, pad=DiscrPad(**PAD_PROPS), slab=_slab(), num_mount=241, distance=0.6, z_f=81e-3, y_f=0.0,
    )


class TestPostprocessing:
    """Test suite for postprocessing operations and FRF calculations."""

    def test_compute_frf(self):
        """Test the computation of Frequency Response Functions (FRF).

        Verifies that given a known signal and excitation in time domain,
        the output spectra lengths are correctly matching half of the
        input sample size (due to real FFT).
        """
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        freq_hz = 10.0
        signal = np.sin(2 * np.pi * freq_hz * t)
        excitation = np.ones_like(t)

        freq, receptance, mobility, accelerance = compute_frf(signal, excitation, dt)

        assert len(freq) == len(t) // 2
        assert len(receptance) == len(t) // 2
        assert len(mobility) == len(t) // 2
        assert len(accelerance) == len(t) // 2

    def test_track_response_analytical(self):
        """Test TrackResponse with an analytical result.

        Ensures that an analytical simulation result is parsed properly
        and the derived spectrum properties (receptance, accelerance)
        are populated without errors.
        """
        result = DummyAnalyticalResult()
        tr = TrackResponse(result=result)

        np.testing.assert_array_equal(tr.freq, result.f)
        np.testing.assert_array_equal(tr.mobility, result.mobility)
        assert tr.receptance is not None
        assert tr.accelerance is not None

    def test_track_response_stampka(self):
        """Test TrackResponse with a Stampka simulation result.

        Verifies that numerical time-domain simulations can be
        processed into frequency response spectra correctly.
        """
        result = DummyStampkaResult()
        tr = TrackResponse(result=result)

        assert tr.freq is not None
        assert tr.receptance is not None

    def test_track_response_rolland(self):
        """Test TrackResponse with a Rolland simulation result.

        Validates extraction of displacement observables and force
        excitation to compute FRF.
        """
        result = DummyRollandResult()
        tr = TrackResponse(result=result)

        assert tr.freq is not None
        assert tr.receptance is not None

    def test_track_decay_rate_stampka(self):
        """Test calculation of narrowband Track Decay Rate.

        Evaluates track decay rate natively over raw frequencies
        from numerical simulation results.
        """
        result = DummyStampkaResult()
        tdr = TrackDecayRate(result=result, octave_fraction=None)

        assert tdr.tdr is not None
        assert tdr.freq is not None

    def test_track_decay_rate_octave_bands(self):
        """Test calculation of Track Decay Rate in 1/3 octave bands.

        Follows the DIN EN 15461 methodology to synthesize
        track decay rate over averaged frequency intervals.
        """
        result = DummyStampkaResult()
        tdr = TrackDecayRate(result=result, octave_fraction=3)

        assert tdr.tdr is not None
        assert tdr.freq is not None

    def test_track_decay_rate_discrete_track_stampka(self):
        """Test the TDR measurement points of a Stampka simulation on a discrete track.

        Starting from the excitation in the sleeper bay centre, the measurement
        positions are mapped onto the spatial grid.
        """
        result = DummyStampkaResult()
        result.track = _discr_slab_track()
        result.discr.dx = 0.05
        result.ind_excit = 606  # x = 30.3 m, centre of the sleeper bay 30.0 m - 30.6 m
        result.deflection = np.random.rand(2881, 100)
        tdr = TrackDecayRate(result=result, octave_fraction=None)

        np.testing.assert_array_equal(tdr.ind_tdr, 606 + np.rint(X_TDR / 0.05))
        assert np.all(np.isfinite(tdr.tdr))

    @pytest.mark.parametrize(
        ('method', 'track', 'positions'),
        [
            (TBCont1LKosto, _cont_slab_track, 'x_resp'),
            (EBBCont1L, _cont_slab_track, 'x'),
            (TSBDiscr1L, _discr_slab_track, 'x'),
        ],
    )
    def test_track_decay_rate_analytical(self, method, track, positions):
        """Test calculation of Track Decay Rate for analytical results.

        The mobility is taken directly from the analytical method at the
        TDR measurement positions and compared to the EN 15461 summation.
        """
        x_excit = 30.3  # centre of a sleeper bay for the discrete track
        kwargs = {} if method is TBCont1LKosto else {'damp_type': 'hysteretic'}
        result = method(
            track=track(), f=np.logspace(2, 3.5, 20), x_excit=x_excit, **{positions: x_excit + X_TDR}, **kwargs,
        )
        tdr_narrow = TrackDecayRate(result=result, octave_fraction=None)
        tdr_octave = TrackDecayRate(result=result, octave_fraction=3)

        # Kostovasilis: mobility with shape (n_freq, n_positions), EBB / TSB: (n_positions, n_freq)
        mob = np.abs(result.mobility) if positions == 'x_resp' else np.abs(result.mobility).T
        dx_n = TrackDecayRate._interval_weights(X_TDR)  # noqa: SLF001
        expected_tdr = DB_CONVERSION_FACTOR / (((mob / mob[:, [0]]) ** 2) * dx_n).sum(axis=1)

        np.testing.assert_allclose(tdr_narrow.freq, result.f)
        np.testing.assert_allclose(tdr_narrow.tdr, expected_tdr)
        assert tdr_octave.tdr.size > 0
        assert np.all(np.isfinite(tdr_octave.tdr))

    def test_track_decay_rate_kosto_missing_positions(self):
        """Test that a Kostovasilis result without the TDR positions is rejected.

        Without matching response positions, the user is asked to provide
        the required x_resp.
        """
        result = TBCont1LKosto(track=_cont_slab_track(), f=np.logspace(2, 3.5, 5), x_excit=10.0)

        with pytest.raises(ValueError, match='x_resp'):
            TrackDecayRate(result=result)

    def test_track_decay_rate_analytical_excitation_outside_bay_centre(self):
        """Test that an analytical excitation outside the sleeper bay centre is rejected.

        Analytical excitation positions are exact, so no grid tolerance is applied.
        """
        x_excit = 30.45  # sleeper bay 30.0 m - 30.6 m, centre at 30.3 m
        result = TSBDiscr1L(
            track=_discr_slab_track(), f=np.logspace(2, 3.5, 5), x_excit=x_excit, x=x_excit + X_TDR,
            damp_type='hysteretic',
        )

        with pytest.raises(ValueError, match='sleeper bay centre'):
            TrackDecayRate(result=result)

    def test_interval_weights(self):
        """Test the computation of spatial summation weights.

        Ensures the delta-x widths correspond accurately to the
        distance between the midpoints of sequential positions.
        """
        x = np.array([0, 1, 3, 6])
        dx = TrackDecayRate._interval_weights(x)  # noqa: SLF001
        expected_dx = np.array([0.5, 1.5, 2.5, 3])
        np.testing.assert_array_equal(dx, expected_dx)
