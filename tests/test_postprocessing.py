"""Tests for postprocessing module."""

import numpy as np

from rolland.postprocessing import TrackDecayRate, TrackResponse, compute_frf


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

    def test_interval_weights(self):
        """Test the computation of spatial summation weights.

        Ensures the delta-x widths correspond accurately to the
        distance between the midpoints of sequential positions.
        """
        x = np.array([0, 1, 3, 6])
        dx = TrackDecayRate._interval_weights(x)  # noqa: SLF001
        expected_dx = np.array([0.5, 1.5, 2.5, 3])
        np.testing.assert_array_equal(dx, expected_dx)
