"""Tests for radiation module."""

import warnings

import matplotlib as mpl
import numpy as np
import pytest
import soundfile as sf

from rolland.radiation import NORMALIZE_PEAK, Radiation, SimplifiedMonopoleRadiation, a_weighting

mpl.use('Agg')

# Synthetic simulation parameters
DT = 1e-4  # Time step [s]
NT = 512  # Number of time steps [-]
DX = 0.05  # Spatial grid spacing [m]
NX = 41  # Number of grid points [-]
X_EXCIT = 1.0  # Excitation position [m]

# Square rail outline (0.1 m x 0.1 m), centred at the origin [m]
SQUARE = [(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)]

# IEC 61672-1 A-weighting table values [Hz] -> [dB]
IEC_A_WEIGHTING = {
    31.5: -39.4,
    63: -26.2,
    125: -16.1,
    250: -8.6,
    500: -3.2,
    1000: 0.0,
    2000: 1.2,
    4000: 1.0,
    8000: -1.1,
    16000: -6.6,
}


class DummyRail:
    """Dummy Rail."""

    def __init__(self):
        self.rl_geo = SQUARE


class DummyExcit:
    """Dummy Stationary Excitation."""

    def __init__(self):
        self.x_excit = X_EXCIT


class DummyMovingExcit:
    """Dummy Excitation without fixed position."""


class DummyDiscr:
    """Dummy Discretization."""

    def __init__(self):
        self.dt = DT
        self.dx = DX


class DummyResult:
    """Dummy Deflection result with a uniform vertical Gaussian pulse."""

    def __init__(self, amplitude=1e-6, store='full', excit=None):
        t = np.arange(NT) * DT
        pulse = amplitude * np.exp(-(((t - 0.01) / 5e-4) ** 2))
        self.u_z_obs = np.tile(pulse[:, np.newaxis], (1, NX))
        self.u_y_obs = np.zeros((NT, NX))
        self.phi_x_obs = np.zeros((NT, NX))
        self.store = store
        self.skip = 1
        self.discr = DummyDiscr()
        self.excit = excit if excit is not None else DummyExcit()


def make_radiation(result=None, **kwargs):
    """Create a small, fast radiation model."""
    params = {
        'result': result if result is not None else DummyResult(),
        'rail': DummyRail(),
        'measurement_point': np.array([X_EXCIT, 3.0, 0.0]),
        'mesh_length': 1.0,
        'mesh_size': 0.05,
        'num_pts': 20,
        'spatial_window': 1.0,
    }
    params.update(kwargs)
    return SimplifiedMonopoleRadiation(**params)


@pytest.fixture(scope='module')
def rad():
    """Radiation model with default settings."""
    return make_radiation()


# --- A-weighting ------------------------------------------------------------------


@pytest.mark.parametrize(('freq', 'level'), IEC_A_WEIGHTING.items())
def test_a_weighting_matches_iec_table(freq, level):
    """A-weighting agrees with the IEC 61672-1 table (rounded to 0.1 dB)."""
    assert 20 * np.log10(a_weighting(np.array([freq]))[0]) == pytest.approx(level, abs=0.15)


def test_a_weighting_zero_at_dc():
    """A-weighting suppresses the DC component."""
    assert a_weighting(np.array([0.0]))[0] == 0.0


# --- Spectral conventions ---------------------------------------------------------


def test_spectral_density_matches_analytical_fourier_transform():
    """Spectral density of a Gaussian equals its continuous Fourier transform."""
    sigma, t = 1e-3, np.arange(10000) * 5e-5
    signal = np.exp(-(((t - 0.1) / sigma) ** 2))

    spectrum = np.fft.rfft(signal, norm='forward')
    spectrum[1:-1] *= 2.0  # One-sided amplitude spectrum
    freq = np.fft.rfftfreq(len(signal), 5e-5)

    density = SimplifiedMonopoleRadiation._spectral_density(spectrum, 5e-5)  # noqa: SLF001
    analytical = sigma * np.sqrt(np.pi) * np.exp(-((np.pi * freq * sigma) ** 2))

    np.testing.assert_allclose(np.abs(density[:200]), analytical[:200], rtol=1e-6, atol=1e-12)


def test_two_sided_scaling_roundtrip():
    """One-sided spectrum reverts to the original time signal (no 1/N or factor 2 error)."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(1000)

    spectrum = np.fft.rfft(signal, norm='forward')
    spectrum[1:-1] *= 2.0

    scaled = SimplifiedMonopoleRadiation._two_sided_scaling(spectrum)  # noqa: SLF001
    restored = np.fft.irfft(scaled, norm='forward')
    np.testing.assert_allclose(restored, signal, atol=1e-10)


def test_velocity_spectrum_of_harmonic_displacement():
    """Harmonic displacement u0 sin(wt) yields a velocity amplitude of w u0."""
    f0, u0 = 25 / (NT * DT), 1e-6  # Exactly on a frequency bin (no leakage)
    t = np.arange(NT) * DT
    displacement = (u0 * np.sin(2 * np.pi * f0 * t))[:, np.newaxis]

    velocity, freq = SimplifiedMonopoleRadiation._velocity_spectrum(displacement, DT)  # noqa: SLF001
    idx = np.argmin(np.abs(freq - f0))

    assert freq[idx] == pytest.approx(f0)
    assert np.abs(velocity[0, idx]) == pytest.approx(2 * np.pi * f0 * u0, rel=1e-4)


# --- Input validation -------------------------------------------------------------


def test_requires_full_store():
    """Radiation requires store='full'."""
    with pytest.raises(ValueError, match="store='full'"):
        make_radiation(result=DummyResult(store='excit'))


def test_requires_x_centre_without_fixed_excitation():
    """x_centre must be given for excitations without position."""
    with pytest.raises(ValueError, match='x_centre'):
        make_radiation(result=DummyResult(excit=DummyMovingExcit()))


def test_x_centre_explicit():
    """Explicit x_centre is used for excitations without position."""
    model = make_radiation(result=DummyResult(excit=DummyMovingExcit()), x_centre=X_EXCIT)
    assert model.x_centre == X_EXCIT


def test_measurement_point_on_surface_raises(rad):
    """Receiver on a source centre raises an error."""
    centres = rad.base_centre.astype(np.float32)
    with pytest.raises(ValueError, match='too close'):
        rad._source_distance(centres, centres[0])  # noqa: SLF001


def test_radiation_is_abstract():
    """The base class cannot be instantiated."""
    with pytest.raises(TypeError):
        Radiation(result=DummyResult(), rail=DummyRail())  # pyright: ignore[reportAbstractUsage, reportArgumentType]


# --- Mesh -------------------------------------------------------------------------


def test_mesh_surface_area(rad):
    """Total mesh area equals the lateral surface plus both end caps."""
    contour = Radiation._interpolate_contour_2d(np.asarray(SQUARE), rad.num_pts)  # noqa: SLF001
    closed = np.vstack([contour, contour[:1]])
    perimeter = np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1))
    y, z = contour[:, 0], contour[:, 1]
    cross_section = 0.5 * abs(np.dot(y, np.roll(z, 1)) - np.dot(z, np.roll(y, 1)))

    expected = perimeter * rad.mesh_length + 2 * cross_section
    assert np.sum(rad.base_area) == pytest.approx(expected, rel=1e-6)


def test_mesh_normals_are_unit_and_outward(rad):
    """Normals have unit length and point away from the cross-section centroid."""
    np.testing.assert_allclose(np.linalg.norm(rad.base_norm, axis=1), 1.0, rtol=1e-6)

    lateral = np.abs(rad.base_norm[:, 0]) < 0.5  # Exclude end caps
    radial = rad.base_centre[lateral, 1:] - np.mean(rad.triangle_coords[:, 1:], axis=0)
    assert np.all(np.sum(rad.base_norm[lateral, 1:] * radial, axis=1) > 0)


def test_interpolate_contour_invalid_input():
    """Invalid contours raise an error."""
    with pytest.raises(ValueError, match='Invalid contour'):
        Radiation._interpolate_contour_2d(np.array([[0.0, 0.0]]), 10)  # noqa: SLF001


# --- Sound pressure ---------------------------------------------------------------


def test_result_shapes(rad):
    """Spectra and time signals have consistent shapes."""
    n_freq = NT // 2 + 1
    for attr in ('sp_freq', 'sp_density', 'sp_freq_a', 'sp_density_a'):
        assert getattr(rad, attr).shape == (n_freq,)
    assert rad.freq.shape == (n_freq,)
    assert rad.sp_time.shape == (NT,)
    assert rad.sp_time_a.shape == (NT,)


def test_zero_deflection_gives_zero_pressure():
    """Without deflection there is no sound pressure."""
    model = make_radiation(result=DummyResult(amplitude=0.0))
    assert np.all(model.sp_freq == 0)
    assert np.all(model.sp_time == 0)


def test_pressure_is_linear_in_deflection(rad):
    """Doubling the deflection doubles the sound pressure."""
    model = make_radiation(result=DummyResult(amplitude=2e-6))
    np.testing.assert_allclose(model.sp_freq, 2 * rad.sp_freq, rtol=1e-4, atol=1e-12)


def test_far_field_decay():
    """Sound pressure decays approximately with 1/r in the far field.

    The receiver is placed above the rail; laterally at z = 0 a vertically
    vibrating symmetric cross-section has a dipole null.
    """
    near = make_radiation(measurement_point=np.array([X_EXCIT, 0.0, 20.0]))
    far = make_radiation(measurement_point=np.array([X_EXCIT, 0.0, 40.0]))

    band = (near.freq > 100) & (near.freq < 2000)
    ratio = np.abs(near.sp_freq[band]) / np.abs(far.sp_freq[band])
    assert np.median(ratio) == pytest.approx(2.0, rel=0.1)


def test_time_signal_matches_spectrum(rad):
    """Time signal is the inverse transform of the amplitude spectrum."""
    spectrum = np.fft.rfft(rad.sp_time, norm='forward')
    spectrum[1:-1] *= 2.0
    np.testing.assert_allclose(spectrum, rad.sp_freq, rtol=1e-4, atol=1e-10)


def test_propagation_delay(rad):
    """The pulse arrives with the acoustic propagation delay r / c."""
    distance = np.linalg.norm(rad.measurement_point - np.array([X_EXCIT, 0.0, 0.0]))
    t_peak = np.argmax(np.abs(rad.sp_time)) * rad.dt
    assert t_peak == pytest.approx(0.01 + distance / rad.speed_of_sound, abs=1e-3)


def test_ground_reflection():
    """Reflection coefficient 0 equals no ground; a rigid ground increases the pressure."""
    free = make_radiation()
    no_reflection = make_radiation(ground_distance=0.5, reflection_coefficient=0.0)
    rigid = make_radiation(ground_distance=0.5, reflection_coefficient=1.0)

    np.testing.assert_allclose(no_reflection.sp_freq, free.sp_freq, rtol=1e-6)
    assert np.sum(np.abs(rigid.sp_freq) ** 2) > np.sum(np.abs(free.sp_freq) ** 2)


def test_workers_do_not_change_result(rad):
    """Parallel processing gives exactly the same result."""
    model = make_radiation(workers=4)
    np.testing.assert_array_equal(model.sp_freq, rad.sp_freq)


def test_chunk_size_does_not_change_result(rad):
    """Chunk size only changes the float32 summation order."""
    model = make_radiation(mesh_chunk_size=50)
    # Cancellation between opposite faces amplifies float32 round-off
    atol = 5e-3 * np.max(np.abs(rad.sp_freq))
    np.testing.assert_allclose(model.sp_freq, rad.sp_freq, rtol=0, atol=atol)


def test_a_weighted_results(rad):
    """A-weighted spectrum is the unweighted spectrum times the A-weighting."""
    expected = rad.sp_freq * a_weighting(rad.freq)
    np.testing.assert_allclose(rad.sp_freq_a, expected, rtol=1e-5, atol=1e-12)
    assert rad.sp_freq_a[0] == 0


# --- Plotting ---------------------------------------------------------------------


@pytest.mark.parametrize('quantity', ['density', 'spl', 'time'])
@pytest.mark.parametrize('weighted', [True, False])
def test_show(rad, quantity, weighted):
    """All quantities can be plotted, weighted and unweighted."""
    rad.a_weighting = weighted
    try:
        ax = rad.show(quantity=quantity, label='test')
        assert len(ax.lines) == 1
        assert ('A-weighted' in ax.get_title()) == weighted
    finally:
        rad.a_weighting = True


def test_show_invalid_quantity(rad):
    """Unknown quantities raise an error."""
    with pytest.raises(ValueError, match='quantity'):
        rad.show(quantity='unknown')


# --- WAV export -------------------------------------------------------------------


@pytest.mark.parametrize('weighted', [True, False])
def test_to_wav_is_calibrated(rad, tmp_path, weighted):
    """WAV samples times full_scale_pa reproduce the sound pressure in Pa."""
    rad.a_weighting = weighted
    try:
        sr = int(round(1 / rad.dt))
        path = rad.to_wav(str(tmp_path / 'p.wav'), target_sr=sr, apply_fade=False, subtype='FLOAT')
    finally:
        rad.a_weighting = True

    samples, sr_read = sf.read(path)
    expected = rad.sp_time_a if weighted else rad.sp_time

    assert sr_read == sr
    np.testing.assert_allclose(samples * 20.0, expected, atol=1e-6)


def test_to_wav_default_format(rad, tmp_path):
    """Default export is 24-bit PCM at 44.1 kHz."""
    path = rad.to_wav(str(tmp_path / 'p.wav'))
    info = sf.info(path)
    assert info.subtype == 'PCM_24'
    assert info.samplerate == 44100


def test_to_wav_normalize(rad, tmp_path):
    """Normalized export scales the peak to NORMALIZE_PEAK."""
    path = rad.to_wav(str(tmp_path / 'p.wav'), normalize=True, subtype='FLOAT')
    samples, _ = sf.read(path)
    assert np.max(np.abs(samples)) == pytest.approx(NORMALIZE_PEAK, rel=1e-6)


def test_to_wav_clipping_warning(rad, tmp_path):
    """A too small full scale clips the signal and warns."""
    peak = np.max(np.abs(rad.sp_time_a))
    with pytest.warns(UserWarning, match='clipped'):
        path = rad.to_wav(str(tmp_path / 'p.wav'), full_scale_pa=peak / 10, subtype='FLOAT')
    samples, _ = sf.read(path)
    assert np.max(np.abs(samples)) <= 1.0


def test_to_wav_no_warning_within_full_scale(rad, tmp_path):
    """No warning if the signal stays below full scale."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rad.to_wav(str(tmp_path / 'p.wav'))
