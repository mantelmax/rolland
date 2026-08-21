"""Postprocessing operations.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy import (
    ndarray,
    pi,
)
from numpy.fft import fft, fftfreq

from .track import (
    DiscrBallastedSingleRailTrack,
    DiscrSlabSingleRailTrack,
)

# Constants
DB_CONVERSION_FACTOR = 4.343
DEFAULT_PLOT_FREQ_RANGE = (100, 5000)
EPSILON_FRF = 1e-12


class SimulationResult(Protocol):
    """Protocol defining a generic simulation result."""


class RollandResult(SimulationResult, Protocol):
    """Protocol defining a Rolland-based simulation result."""

    u_z_obs: ndarray
    store: str
    excit: Any
    discr: Any
    skip: int
    track: Any


class StampkaResult(SimulationResult, Protocol):
    """Protocol defining an Stampka-based simulation result."""

    deflection: ndarray
    force: ndarray
    discr: Any
    ind_excit: int
    track: Any


class AnalyticalResult(SimulationResult, Protocol):
    """Protocol defining an analytical simulation result."""

    mobility: ndarray
    f: ndarray


def compute_frf(signal: ndarray, excitation: ndarray, dt: float) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """Compute Frequency Response Function (FRF) from time-domain signals."""
    signal_arr = np.asarray(signal).flatten()
    excitation_arr = np.asarray(excitation).flatten()

    n_samples = min(signal_arr.shape[0], excitation_arr.shape[0])
    n_freq = n_samples // 2

    resp_fft = fft(signal_arr[:n_samples])
    exc_fft = fft(excitation_arr[:n_samples])
    freq = fftfreq(n_samples, dt)[:n_freq]

    exc_fft_mag = np.abs(exc_fft[:n_freq])
    exc_fft_safe = np.where(exc_fft_mag < EPSILON_FRF, EPSILON_FRF, exc_fft[:n_freq])

    receptance = resp_fft[:n_freq] / exc_fft_safe
    omega = 2 * pi * freq
    mobility = (1j * omega) * receptance
    accelerance = -(omega**2) * receptance

    return freq, receptance, mobility, accelerance


@dataclass(kw_only=True)
class PostProcessing:
    r"""Abstract base class for postprocessing operations.

    Attributes
    ----------
    result : SimulationResult, optional
        The simulation result object (Rolland, Stampka, or Analytical).
    results : SimulationResult, optional
        Alias for `result`, provided for backwards compatibility.
    freq : ndarray
        The frequency array of the response :math:`[Hz]`.
    """

    result: SimulationResult | None = None
    results: SimulationResult | None = None
    freq: ndarray = field(default=None, init=False)

    def __post_init__(self):
        """Initialize base attributes."""
        if self.result is None and self.results is not None:
            self.result = self.results

    def _plot_base(  # noqa: C901

        self,
        x,
        y,
        quantity: str,
        title: str | None = None,
        ax=None,
        label=None,
        plot_type='loglog',
        octave_fraction=None,
        **kwargs,
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            msg = 'matplotlib is required for plotting.'
            raise ImportError(msg) from e

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if octave_fraction is not None:
            kwargs.setdefault('drawstyle', 'steps-mid')

        y_plot = np.abs(y) if np.iscomplexobj(y) else y

        if plot_type == 'loglog':
            mask = x > 0
            x_plot = x[mask]
            y_plot = y_plot[mask]
        else:
            x_plot = x

        if plot_type == 'loglog':
            ax.loglog(x_plot, y_plot, label=label, **kwargs)
        elif plot_type == 'semilogx':
            ax.semilogx(x_plot, y_plot, label=label, **kwargs)
        elif plot_type == 'semilogy':
            ax.semilogy(x_plot, y_plot, label=label, **kwargs)
        else:
            ax.plot(x_plot, y_plot, label=label, **kwargs)

        has_previous_data = len(ax.lines) > 1
        if has_previous_data:
            cur_bottom, cur_top = ax.get_ylim()

        x_min_plot, x_max_plot = DEFAULT_PLOT_FREQ_RANGE
        mask_x = (x_plot >= x_min_plot) & (x_plot <= x_max_plot)
        y_visible = y_plot[mask_x]

        if len(y_visible) > 0:
            min_y, max_y = np.min(y_visible), np.max(y_visible)
            if min_y > 0 and max_y > 0 and plot_type in ['loglog', 'semilogy']:
                target_bottom = min_y / 5
                target_top = max_y * 5
            else:
                margin = (max_y - min_y) * 0.1
                target_bottom = min_y - margin
                target_top = max_y + margin

            if has_previous_data:
                target_bottom = min(target_bottom, cur_bottom)
                target_top = max(target_top, cur_top)

            ax.set_ylim(target_bottom, target_top)

        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(quantity)
        ax.set_title(title or f'Track Response ({quantity})')
        ax.grid(True, which='both', ls='-', alpha=0.5)
        if label:
            ax.legend()

        return ax


@dataclass(kw_only=True)
class TrackResponse(PostProcessing):
    r"""Unified Track Response class supporting Rolland, Stampka, and Analytical models.

    Attributes
    ----------
    result : SimulationResult, optional
        The simulation result object (Rolland, Stampka, or Analytical).
    position_index : int, optional
        The spatial index at which to evaluate the response. Defaults to the excitation position.
    direction : str, optional
        The primary direction to extract. Inferred for Rolland StationaryExcitation.
    coupled_rotation : str, optional
        If provided, coupled rotational mobility is added. Automatically inferred for Rolland.
    offset : float, optional
        The offset :math:`[m]` for computing coupled mobility. Automatically inferred for Rolland.
    results : SimulationResult, optional
        Alias for `result`, provided for backwards compatibility.
    freq : ndarray
        The frequency array of the response :math:`[Hz]`.
    receptance : ndarray
        The receptance (displacement / force) spectrum array :math:`[m/N]`.
    mobility : ndarray
        The mobility (velocity / force) spectrum array :math:`[m/(sN)]`.
    accelerance : ndarray
        The accelerance (acceleration / force) spectrum array :math:`[m/(s^2N)]`.

    Example
    -------
    >>> from rolland.postprocessing import TrackResponse
    >>> response = TrackResponse(result=deflection_results)
    >>> response.show(quantity='mobility')
    """

    position_index: int | None = None
    direction: str | None = None
    coupled_rotation: str | None = None
    offset: float | None = None

    receptance: ndarray = field(default=None, init=False)
    mobility: ndarray = field(default=None, init=False)
    accelerance: ndarray = field(default=None, init=False)

    def __post_init__(self):
        """Initialize TrackResponse."""
        super().__post_init__()
        if self.result is not None:
            self._parse_result(self.result, self.position_index, self.direction, self.coupled_rotation, self.offset)

    def _parse_result(self, result, position_index, direction, coupled_rotation, offset):
        if hasattr(result, 'u_z_obs'):
            self._parse_rolland(result, position_index, direction, coupled_rotation, offset)
        elif hasattr(result, 'deflection') and hasattr(result, 'force'):
            self._parse_stampka(result, position_index, direction, coupled_rotation, offset)
        elif hasattr(result, 'mobility'):
            self._parse_analytical(result)
        else:
            msg = 'Unsupported result type for TrackResponse.'
            raise TypeError(msg)

    def _parse_rolland(self, result, position_index, direction, coupled_rotation, offset):
        from rolland.excitation import StationaryExcitation

        if isinstance(result.excit, StationaryExcitation):
            if getattr(result.excit, 'force_dir', 'vertical') == 'vertical':
                direction = direction or 'z'
                coupled_rotation = coupled_rotation or 'x'
                offset = offset if offset is not None else result.excit.y_e
            elif getattr(result.excit, 'force_dir', 'vertical') == 'lateral':
                direction = direction or 'y'
                coupled_rotation = coupled_rotation or 'x'
                offset = offset if offset is not None else result.excit.z_e
        else:
            direction = direction or 'z'
            offset = offset if offset is not None else 0.0

        signal = getattr(result, f'u_{direction}_obs')
        if signal.ndim == 2:
            idx = (
                position_index
                if position_index is not None
                else (round(result.excit.x_excit / result.discr.dx) if result.store == 'full' else 0)
            )
            signal = signal[:, idx]

        if coupled_rotation:
            phi_signal = getattr(result, f'phi_{coupled_rotation}_obs')
            if phi_signal.ndim == 2:
                idx = (
                    position_index
                    if position_index is not None
                    else (round(result.excit.x_excit / result.discr.dx) if result.store == 'full' else 0)
                )
                phi_signal = phi_signal[:, idx]
            signal = signal + phi_signal * offset

        if hasattr(result.excit, 'force'):
            excitation = result.excit.force.data[:: result.skip]
        else:
            excitation = getattr(result.excit, f'force_{direction}')[:: result.skip]
        dt = result.discr.dt * result.skip

        self.freq, self.receptance, self.mobility, self.accelerance = compute_frf(signal, excitation, dt)

    def _parse_stampka(self, result, position_index, direction, coupled_rotation, offset):
        direction = direction or 'z'
        offset = offset if offset is not None else 0.0
        idx = position_index if position_index is not None else result.ind_excit

        signal = result.deflection[idx]

        if coupled_rotation and hasattr(result, 'rotation'):
            phi_signal = result.rotation[idx]
            signal = signal + phi_signal * offset

        excitation = result.force
        dt = result.discr.dt
        self.freq, self.receptance, self.mobility, self.accelerance = compute_frf(signal, excitation, dt)

    def _parse_analytical(self, result):
        self.freq = result.f
        self.mobility = result.mobility
        omega = 2 * pi * self.freq
        self.receptance = self.mobility / (1j * omega)
        self.accelerance = self.mobility * (1j * omega)

    def show(self, quantity='mobility', ax=None, label=None, plot_type='loglog', octave_fraction=None, **kwargs):
        """Plot the specified response quantity against frequency.

        Parameters
        ----------
        quantity : str, default 'mobility'
            The quantity to plot ('receptance', 'mobility', or 'accelerance').
        ax : matplotlib.axes.Axes, optional
            An existing matplotlib axes to plot on. If None, a new figure is created.
        label : str, optional
            The label for the plotted curve in the legend.
        plot_type : str, default 'loglog'
            The matplotlib plotting method to use ('loglog', 'semilogx', 'semilogy', or 'plot').
        octave_fraction : int, optional
            If provided, smooths the spectrum into fractional octave bands.
        **kwargs : dict
            Additional styling arguments passed to the matplotlib plot function.
        """
        if octave_fraction is not None:
            x, y = self.to_octave_bands(fraction=octave_fraction, quantity=quantity)
            if label:
                label = f'{label} (1/{octave_fraction} Octave)'
        else:
            x = self.freq
            y = getattr(self, quantity)

        if y is None:
            msg = f"Quantity '{quantity}' is not available."
            raise ValueError(msg)

        return self._plot_base(
            x,
            y,
            quantity.capitalize(),
            ax=ax,
            label=label,
            plot_type=plot_type,
            octave_fraction=octave_fraction,
            **kwargs,
        )

    def to_octave_bands(self, fraction=3, quantity='mobility'):
        """Convert narrow-band quantity to fractional octave bands."""
        y = getattr(self, quantity, None)
        if self.freq is None or y is None:
            return None, None

        y = np.abs(y) if np.iscomplexobj(y) else y
        f_min, f_max = self.freq[self.freq > 0].min(), self.freq.max()
        if f_min == f_max:
            return self.freq, y

        bands = []
        band_y = []

        f = f_min
        factor = 2 ** (1.0 / (2.0 * fraction))

        while f < f_max:
            f_lower = f / factor
            f_upper = f * factor
            mask = (self.freq >= f_lower) & (self.freq < f_upper)
            if np.any(mask):
                bands.append(f)
                band_y.append(np.sqrt(np.mean(np.abs(y[mask]) ** 2)))
            f = f * (2 ** (1.0 / fraction))

        return np.array(bands), np.array(band_y)

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class TrackDecayRate(PostProcessing):
    r"""Unified Track-Decay-Rate class supporting Rolland and Numerical Stampka simulations.

    The evaluation method is based on :cite:p:`EN15461:2008`.

    **Measurement Grid and Spatial Summation**

    The TDR is determined by spatially summing the responses at measurement positions :math:`x_n`.
    The intervals :math:`\Delta x_n` (`dx_n`) are centered between adjacent positions
    using midpoints `M_n` as integration boundaries:

    .. code-block:: text

        Track axis (Distance in meters)
        ================================================================================

        Measurement positions (x_n):
               x_0             x_1                   x_2                      x_max
               │               │                     │                        │
               ▼               ▼                     ▼                        ▼
        ───────┼───────────────┼─────────────────────┼───────── ⋯ ────────────┼───────

        Integration boundaries (Midpoints):
               │       M_1             M_2                     M_max-1        │
               │       │               │                       │              │

        Corresponding intervals (\Delta x_n):
               ├─dx_0──┼─────dx_1──────┼────────dx_2───────────┤  ⋯           ├dx_max┤

    Attributes
    ----------
    result : SimulationResult, optional
        The simulation result object (Rolland or Stampka) containing deflection and excitation.
    position_index : int, optional
        The spatial index at which to evaluate the response.
    direction : str, optional
        The primary direction to extract. Inferred for Rolland.
    coupled_rotation : str, optional
        If provided, coupled rotational mobility is added.
    offset : float, optional
        The offset :math:`[m]` for computing coupled mobility.
    results : SimulationResult, optional
        Alias for `result`, provided for backwards compatibility.
    octave_fraction : int or None, default 3
        Fraction for octave band averaging (e.g., 3 for 1/3-octave bands).
        If None, calculates narrowband TDR.
    f_min : float, default 0.0
        Minimum frequency for the resulting TDR spectrum :math:`[Hz]`.
    f_max : float, optional
        Maximum frequency for the resulting TDR spectrum :math:`[Hz]`.
    tol_excit : float, optional
        Tolerance for verifying the excitation point lies in the sleeper bay center :math:`[m]`.
    freq : ndarray
        The frequency array of the response :math:`[Hz]`.
    receptance : ndarray
        The receptance spectrum array :math:`[m/N]`.
    mobility : ndarray
        The mobility spectrum array :math:`[m/(sN)]`.
    accelerance : ndarray
        The accelerance spectrum array :math:`[m/(s^2N)]`.
    tdr : ndarray
        The Track Decay Rate spectrum array :math:`[dB/m]`.
    ind_tdr : list[int]
        List of spatial indices used for TDR calculation.
    x_tdr : ndarray
        Spatial measurement positions for TDR calculation :math:`[m]`.

    Example
    -------
    >>> from rolland.postprocessing import TrackDecayRate
    >>> # 1/3 Octave bands (default per EN 15461):
    >>> tdr_calc = TrackDecayRate(result=deflection_results)
    >>> # Narrowband (raw frequencies):
    >>> tdr_narrow = TrackDecayRate(result=deflection_results, octave_fraction=None)
    """

    position_index: int | None = None
    direction: str | None = None
    coupled_rotation: str | None = None
    offset: float | None = None
    octave_fraction: int | None = 3
    f_min: float = 0.0
    f_max: float | None = None
    tol_excit: float | None = None

    tdr: ndarray = field(default_factory=lambda: np.array([]), init=False)
    ind_tdr: list[int] = field(default_factory=list, init=False)
    x_tdr: ndarray = field(default_factory=lambda: np.array([]), init=False)

    def __post_init__(self):
        """Initialize TrackDecayRate."""
        super().__post_init__()
        if self.result is not None:
            if hasattr(self.result, 'u_z_obs'):
                self._parse_rolland(self.result)
            elif hasattr(self.result, 'deflection'):
                self._parse_stampka(self.result)
            else:
                msg = 'Unsupported result type for TrackDecayRate.'
                raise TypeError(msg)

            self.validate_excitation_position()
            self.find_tdr_points()
            self.validate_tdr_points()
            self.calculate_tdr(fraction=self.octave_fraction)

    def _parse_rolland(self, result):
        if result.store != 'full':
            msg = "Rolland simulation must be run with store='full' for TDR."
            raise ValueError(msg)

        from rolland.excitation import StationaryExcitation

        direction = self.direction
        if isinstance(result.excit, StationaryExcitation):
            if getattr(result.excit, 'force_dir', 'vertical') == 'vertical':
                direction = direction or 'z'
            elif getattr(result.excit, 'force_dir', 'vertical') == 'lateral':
                direction = direction or 'y'
        else:
            direction = direction or 'z'

        self.response_matrix = getattr(result, f'u_{direction}_obs')

        if hasattr(result.excit, 'force'):
            self.excitation = result.excit.force.data[:: result.skip]
        else:
            self.excitation = getattr(result.excit, f'force_{direction}')[:: result.skip]
        self.dt = result.discr.dt * result.skip
        self.dx = result.discr.dx
        self.ind_excit = round(result.excit.x_excit / self.dx)
        self.track = result.track

    def _parse_stampka(self, result):
        self.response_matrix = result.deflection.T
        self.excitation = result.force
        self.dt = result.discr.dt
        self.dx = result.discr.dx
        self.ind_excit = result.ind_excit
        self.track = result.track

    def validate_excitation_position(self):
        r"""Check that the TDR starts in the centre of a sleeper bay."""
        if not isinstance(self.track, (DiscrSlabSingleRailTrack, DiscrBallastedSingleRailTrack)):
            return

        x_mp = np.array(list(self.track.mount_prop.keys()))
        x_excit = self.ind_excit * self.dx

        before = np.where(x_mp <= x_excit)[0]
        after = np.where(x_mp > x_excit)[0]
        if before.size == 0 or after.size == 0:
            msg = f'The excitation at x = {x_excit:.4f} m does not lie between two supports.'
            raise ValueError(msg)

        x_left, x_right = x_mp[before[-1]], x_mp[after[0]]
        x_centre = (x_left + x_right) / 2
        tol = self.dx / 2 if self.tol_excit is None else self.tol_excit

        deviation = abs(x_excit - x_centre)
        if deviation > tol + 1e-9:
            msg = f'Excitation not in sleeper bay centre. Deviation {deviation:.4f} > {tol:.4f}.'
            raise ValueError(msg)

    def find_tdr_points(self):
        r"""Determine the TDR measurement positions x_n and their grid indices."""
        if isinstance(self.track, (DiscrSlabSingleRailTrack, DiscrBallastedSingleRailTrack)):
            x_mp = np.array(list(self.track.mount_prop.keys()))
            ind_mp = (x_mp / self.dx).astype(int)

            before = np.where(ind_mp < self.ind_excit)[0]
            if before.size == 0:
                msg = 'No mounting position found before the excitation index.'
                raise ValueError(msg)
            idx_s = int(before[-1])

            x_s = x_mp[idx_s:] - x_mp[idx_s]
            x_sc = np.convolve(x_s, np.ones(2) / 2, mode='valid')

            n_required = 68
            if x_s.size < n_required:
                msg = f'Only {x_s.size} mounting positions lie at or behind the excitation, required {n_required}.'
                raise ValueError(
                    msg,
                )

            def tdr_points_betw1(idx):
                return ((x_s[idx + 1] - x_sc[idx]) / 2) + x_sc[idx]

            def tdr_points_betw2(idx):
                return ((x_sc[idx] - x_s[idx]) / 2) + x_s[idx]

            self.x_tdr = (
                np.array(
                    [
                        x_sc[0],
                        tdr_points_betw1(0),
                        x_s[1],
                        tdr_points_betw2(1),
                        x_sc[1],
                        tdr_points_betw1(1),
                        x_s[2],
                        tdr_points_betw2(2),
                        x_sc[2],
                        tdr_points_betw1(2),
                        x_s[3],
                        x_sc[3],
                        x_s[4],
                        x_sc[4],
                        x_sc[5],
                        x_sc[6],
                        x_sc[7],
                        x_sc[8],
                        x_sc[10],
                        x_sc[12],
                        x_sc[16],
                        x_sc[20],
                        x_sc[24],
                        x_sc[30],
                        x_sc[36],
                        x_sc[42],
                        x_sc[48],
                        x_sc[54],
                        x_sc[66],
                    ],
                )
                - x_sc[0]
            )

            ind_tdr = np.rint(self.x_tdr.round(5) / self.dx) + self.ind_excit
            self.ind_tdr = list(ind_tdr.astype(int))

        else:
            l_s = 0.6
            x_tdr = (
                np.array(
                    [
                        0.5,
                        0.75,
                        1,
                        1.25,
                        1.5,
                        1.75,
                        2,
                        2.25,
                        2.5,
                        2.75,
                        3,
                        3.5,
                        4,
                        4.5,
                        5.5,
                        6.5,
                        7.5,
                        8.5,
                        10.5,
                        12.5,
                        16.5,
                        20.5,
                        24.5,
                        30.5,
                        36.5,
                        42.5,
                        48.5,
                        54.5,
                        66.5,
                    ],
                )
                * l_s
            )
            self.x_tdr = x_tdr - l_s / 2
            ind_tdr = np.rint(self.x_tdr / self.dx) + self.ind_excit
            self.ind_tdr = list(ind_tdr.astype(int))

    @staticmethod
    def _interval_weights(x):
        r"""Compute the summation weights."""
        x = np.asarray(x, dtype=float)
        dx = np.zeros_like(x)
        dx[0] = (x[1] - x[0]) / 2
        dx[1:-1] = (x[2:] - x[:-2]) / 2
        dx[-1] = x[-1] - x[-2]
        return dx

    def validate_tdr_points(self):
        r"""Check that all TDR measurement points lie inside the simulated domain."""
        response = self.response_matrix
        if getattr(response, 'ndim', 2) != 2:
            msg = 'response_matrix must be two-dimensional with shape (n_time, n_positions).'
            raise ValueError(msg)

        n_positions = response.shape[1]
        ind_min, ind_max = min(self.ind_tdr), max(self.ind_tdr)
        if ind_min < 0 or ind_max >= n_positions:
            msg = 'TDR measurement points lie outside the simulated domain.'
            raise ValueError(msg)

    def _calculate_mobility_spectra(self):
        r"""Calculate the mobility spectrum at every TDR measurement point."""
        mobility_rows = []
        frequency = None
        for ind in self.ind_tdr:
            defl = self.response_matrix[:, ind]
            frequency, _, mobility, _ = compute_frf(defl, self.excitation, self.dt)
            mobility_rows.append(mobility)

        mask = frequency > self.f_min
        if self.f_max is not None:
            mask &= frequency <= self.f_max

        return frequency[mask], np.array(mobility_rows)[:, mask]

    def calculate_tdr(self, fraction=3):
        r"""Calculate the Track-Decay-Rate (TDR) per DIN EN 15461.

        Calculates TDR in fractional octave bands or narrowband.
        """
        freq, mob = self._calculate_mobility_spectra()
        dx_n = self._interval_weights(self.x_tdr)

        if fraction is not None:
            f_min, f_max = freq[freq > 0].min(), freq.max()
            bands = []
            band_mob_list = []

            f = f_min
            factor = 2 ** (1.0 / (2.0 * fraction))

            while f < f_max:
                f_lower = f / factor
                f_upper = f * factor
                mask = (freq >= f_lower) & (freq < f_upper)
                if np.any(mask):
                    bands.append(f)
                    band_rms = np.sqrt(np.mean(np.abs(mob[:, mask]) ** 2, axis=1))
                    band_mob_list.append(band_rms)
                f = f * (2 ** (1.0 / fraction))

            band_freq = np.array(bands)
            band_mob = np.column_stack(band_mob_list) if band_mob_list else np.empty((len(self.x_tdr), 0))

            if band_mob.shape[1] > 0:
                ratio_sq = (band_mob / band_mob[0, :]) ** 2
                sum_tdr = (ratio_sq * dx_n[:, None]).sum(axis=0)
                self.tdr = DB_CONVERSION_FACTOR / sum_tdr
            else:
                self.tdr = np.array([])

            self.freq = band_freq
        else:
            mob_mag = np.abs(mob)
            ratio_sq = (mob_mag / mob_mag[0, :]) ** 2
            sum_tdr = (ratio_sq * dx_n[:, None]).sum(axis=0)
            self.tdr = DB_CONVERSION_FACTOR / sum_tdr
            self.freq = freq

    def show(self, ax=None, label=None, plot_type='loglog', **kwargs):
        """Plot the Track Decay Rate against frequency.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            An existing matplotlib axes to plot on. If None, a new figure is created.
        label : str, optional
            The label for the plotted curve in the legend.
        plot_type : str, default 'loglog'
            The matplotlib plotting method to use ('loglog', 'semilogx', 'semilogy', or 'plot').
        **kwargs : dict
            Additional styling arguments passed to the matplotlib plot function.
        """
        x = self.freq
        y = self.tdr

        if self.octave_fraction is not None:
            kwargs.setdefault('drawstyle', 'steps-mid')

        return self._plot_base(
            x,
            y,
            'TDR [dB/m]',
            title='Track Decay Rate',
            ax=ax,
            label=label,
            plot_type=plot_type,
            octave_fraction=self.octave_fraction,
            **kwargs,
        )

    def dr_min(self):
        """Calculate the minimum theoretical track decay rate."""
        return DB_CONVERSION_FACTOR / self.x_tdr[-1]

    def _abstract(self) -> None:
        pass


@dataclass(kw_only=True)
class VehicleResponse:
    r"""Postprocessing class for vehicle response results. Placeholder for future implementation.

    Example
    -------
    >>> from rolland.postprocessing import VehicleResponse
    >>> vr = VehicleResponse()
    """

    def _abstract(self) -> None:
        pass
