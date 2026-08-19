"""Postprocessing classes.

.. autosummary::
    :toctree: postprocessing

    PostProcessing
    TrackResponse
    PointResponse
    TransferResponse
    TrackDecayRate
    VehicleResponse
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import warnings
from numpy import (
    array,
    asarray,
    convolve,
    iscomplexobj,
    ndarray,
    ones,
    pi,
    rint,
    where,
    zeros_like,
)
from numpy.fft import fft, fftfreq

from .track import (
    ArrangedBallastedSingleRailTrack,
    ArrangedSlabSingleRailTrack,
    DiscrBallastedSingleRailTrack,
    DiscrSlabSingleRailTrack,
)


class PostProcessing(ABC):
    r"""Abstract base class for postprocessing classes."""

    @abstractmethod
    def _abstract(self) -> None:
        """Prevents instantiation of abstract classes."""

    @staticmethod
    def as_array(value):
        """Return raw array data from ndarray-like objects or objects with .data.

        Attributes
        ----------
        value : object
            Input object that may be an ndarray or have a .data attribute.

        Returns
        -------
        array: ndarray
            Raw array data extracted from the input object.
        """
        if hasattr(value, "data"):
            return asarray(value.data)
        return asarray(value)

    @staticmethod
    def as_signal_1d(value, index=0):
        """Extract a 1D signal from 1D or 2D input.

        Parameters
        ----------
        value : ndarray | object
            Input signal (1D or 2D).
        index : int, optional
            Column index for 2D input. Default is 0.

        Returns
        -------
        ndarray
            1D signal.
        """
        arr = PostProcessing.as_array(value)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2:
            return arr[:, index]
        msg = f"Signal must be 1D or 2D, got ndim={arr.ndim}."
        raise ValueError(msg)

    @staticmethod
    def frequency_response(signal, excitation, dt):
        """Calculate the frequency response function (FRF) of a system.

        Both signals are truncated to their common length, transformed, and
        divided. ``numpy.fft.fft`` returns a two-sided spectrum whose upper
        half is the conjugate mirror of the lower one; only the positive
        half is physically meaningful for a response function and is
        returned here. The DC line is kept as element 0 -- note that the
        excitation spectrum is close to zero there, so that line is
        numerically meaningless and callers should discard it.

        Parameters
        ----------
        signal : ndarray | object
            Time-domain response of the system, shape ``(n_time,)``.
        excitation : ndarray | object
            Time-domain excitation of the system, shape ``(n_time,)`` or
            ``(n_time, 1)``.
        dt : float
            Time step between samples :math:`[s]`.

        Returns
        -------
        frequ : ndarray
            Positive frequencies corresponding to the FRF :math:`[Hz]`.
        frf : ndarray
            Complex frequency response function of the system.
        """
        signal = PostProcessing.as_array(signal)
        excitation = PostProcessing.as_signal_1d(excitation)

        n_samples = min(signal.shape[0], excitation.shape[0])
        n_freq = n_samples // 2

        response_spectrum = fft(signal[:n_samples])
        excitation_spectrum = fft(excitation[:n_samples])
        frequ = fftfreq(n_samples, dt)

        frf = response_spectrum[:n_freq] / excitation_spectrum[:n_freq]
        return frequ[:n_freq], frf

    @staticmethod
    def plot(
        arrays, labels, title='Universal Plot', x_label='X-axis', y_label='Y-axis', colors=None, plot_type='loglog',
    ):
        """Universal plot function for multiple data sets.

        Response functions such as mobility and accelerance are complex. Their
        magnitude is what gets plotted over frequency, so complex y data is
        reduced with ``abs`` here -- matplotlib would otherwise silently drop the
        imaginary part and draw the real part instead. Pass ``numpy.angle(y)``
        explicitly to plot a phase.

        Parameters
        ----------
        arrays : list of tuple
            List of tuples, where each tuple contains two numpy.ndarray (x and y data).
        labels : list of str
            List of labels for each array.
        title : str, optional
            Title of the plot. Default is 'Universal Plot'.
        x_label : str, optional
            Label for the x-axis. Default is 'X-axis'.
        y_label : str, optional
            Label for the y-axis. Default is 'Y-axis'.
        colors : list of str, optional
            List of colors for each array. Default is None.
        plot_type : str, optional
            Type of plot (e.g., 'loglog', 'plot'). Default is 'loglog'.
        """
        plt.figure(figsize=(10, 6))
        if colors is None:
            colors = ['k', 'r', 'b', 'g', 'c', 'm', 'y']

        for (x, y), label, color in zip(arrays, labels, colors, strict=False):
            y = abs(y) if iscomplexobj(y) else y
            if plot_type == 'loglog':
                plt.loglog(x, y, label=label, color=color)
            else:
                plt.plot(x, y, label=label, color=color)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


class TrackResponse(PostProcessing):
    """Unified Track Response class supporting Rolland (Devito), Stampka (FDM), and Analytical models."""
    
    def __init__(self, result=None, position_index=None, direction='z', coupled_rotation=None, offset=0.0, results=None):
        """Initialize the TrackResponse from a simulation result.

        Parameters
        ----------
        result : object
            The simulation result object (Devito `Deflection`, FDM `DeflectionStampka`, or Analytical).
        position_index : int, optional
            The spatial index at which to evaluate the response. If None, it defaults to the excitation position.
        direction : str, default 'z'
            The primary direction of deflection to extract ('z' or 'y').
        coupled_rotation : str, optional
            If provided (e.g., 'x'), coupled rotational mobility is added to the response.
        offset : float, default 0.0
            The lateral or vertical offset [m] for computing coupled mobility (used with `coupled_rotation`).
        results : object, optional
            Alias for `result`, provided for backwards compatibility.
        """
        if result is None and results is not None:
            result = results
            
        self.freq = None
        self.receptance = None
        self.mobility = None
        self.accelerance = None
        
        if result is not None:
            self._parse_result(result, position_index, direction, coupled_rotation, offset)
        
    def _parse_result(self, result, position_index, direction, coupled_rotation, offset):
        # 1. Rolland Model (Devito)
        if hasattr(result, "u_z_obs"):
            signal = getattr(result, f"u_{direction}_obs")
            if signal.ndim == 2:
                idx = position_index if position_index is not None else (round(result.excit.x_excit / result.discr.dx) if result.store == 'full' else 0)
                signal = signal[:, idx]
            
            if coupled_rotation:
                phi_signal = getattr(result, f"phi_{coupled_rotation}_obs")
                if phi_signal.ndim == 2:
                    idx = position_index if position_index is not None else (round(result.excit.x_excit / result.discr.dx) if result.store == 'full' else 0)
                    phi_signal = phi_signal[:, idx]
                signal = signal + phi_signal * offset
                
            excitation = result.excit.force.data[::result.skip]
            dt = result.discr.dt * result.skip
            self._compute_frf(signal, excitation, dt)
            
        # 2. Numerical FDM (Stampka)
        elif hasattr(result, "deflection") and hasattr(result, "force"):
            idx = position_index if position_index is not None else result.ind_excit
            signal = result.deflection[idx]
            # Stampka doesn't have coupled rotation implemented yet, but we allow future proofing
            if coupled_rotation and hasattr(result, "rotation"):
                phi_signal = result.rotation[idx]
                signal = signal + phi_signal * offset
                
            excitation = result.force
            dt = result.discr.dt
            self._compute_frf(signal, excitation, dt)
            
        # 3. Analytical Methods
        elif hasattr(result, "mobility"):
            self.freq = result.f
            self.mobility = result.mobility
            # Receptance and accelerance can be derived from mobility
            omega = 2 * pi * self.freq
            self.receptance = self.mobility / (1j * omega)
            self.accelerance = self.mobility * (1j * omega)
            
        else:
            raise TypeError("Unsupported result type for TrackResponse.")

    def _compute_frf(self, signal, excitation, dt):
        """Internal method to compute FRF from time-domain signals."""
        import numpy as np
        signal = np.asarray(signal).flatten()
        excitation = np.asarray(excitation).flatten()
        
        n_samples = min(signal.shape[0], excitation.shape[0])
        n_freq = n_samples // 2
        
        resp_fft = fft(signal[:n_samples])
        exc_fft = fft(excitation[:n_samples])
        freq = fftfreq(n_samples, dt)
        
        self.freq = freq[:n_freq]
        
        # Robust Error Handling (Division by Zero)
        exc_fft_mag = np.abs(exc_fft[:n_freq])
        epsilon = 1e-12
        exc_fft_safe = np.where(exc_fft_mag < epsilon, epsilon, exc_fft[:n_freq])
        
        self.receptance = resp_fft[:n_freq] / exc_fft_safe
        
        omega = 2 * pi * self.freq
        self.mobility = (1j * omega) * self.receptance
        self.accelerance = -(omega**2) * self.receptance
        
    def to_octave_bands(self, fraction=3):
        """Convert narrow-band mobility to fractional octave bands."""
        import numpy as np
        if self.freq is None or self.mobility is None:
            return None, None
            
        f_min, f_max = self.freq[self.freq > 0].min(), self.freq.max()
        if f_min == f_max: return self.freq, self.mobility
        
        bands = []
        band_mobs = []
        
        f = f_min
        factor = 2 ** (1.0 / (2.0 * fraction))
        
        while f < f_max:
            f_lower = f / factor
            f_upper = f * factor
            mask = (self.freq >= f_lower) & (self.freq < f_upper)
            if np.any(mask):
                bands.append(f)
                band_mobs.append(np.mean(np.abs(self.mobility[mask])))
            f = f * (2 ** (1.0 / fraction))
            
        return np.array(bands), np.array(band_mobs)

    def _abstract(self) -> None:
        pass


class TrackDecayRate(PostProcessing):
    """Unified Track-Decay-Rate class supporting Devito and Numerical FDM simulations."""
    
    def __init__(self, result=None, f_min=0.0, f_max=None, tol_excit=None, results=None):
        """Initialize TrackDecayRate and calculate the TDR per DIN EN 15461.

        Parameters
        ----------
        result : object
            The simulation result object (Devito or FDM) containing deflection and excitation.
        f_min : float, default 0.0
            Minimum frequency [Hz] for the resulting TDR spectrum.
        f_max : float, optional
            Maximum frequency [Hz] for the resulting TDR spectrum.
        tol_excit : float, optional
            Tolerance [m] for verifying the excitation point lies in the sleeper bay center.
        results : object, optional
            Alias for `result`, provided for backwards compatibility.
        """
        if result is None and results is not None:
            result = results
            
        self.f_min = f_min
        self.f_max = f_max
        self.tol_excit = tol_excit
        self.tdr = array([])
        self.freq = array([])
        self.ind_tdr = []
        self.x_tdr = array([])
        
        if result is not None:
            if hasattr(result, "u_z_obs"):
                # Devito Extraction
                if result.store != 'full':
                    raise ValueError("Devito simulation must be run with store='full' for TDR.")
                self.response_matrix = result.u_z_obs
                self.excitation = result.excit.force.data[::result.skip]
                self.dt = result.discr.dt * result.skip
                self.dx = result.discr.dx
                self.ind_excit = round(result.excit.x_excit / self.dx)
                self.track = result.track
                
            elif hasattr(result, "deflection"):
                # Stampka Extraction
                self.response_matrix = result.deflection.T  # Transpose to match (time, space)
                self.excitation = result.force
                self.dt = result.discr.dt
                self.dx = result.discr.dx
                self.ind_excit = result.ind_excit
                self.track = result.track
                
            else:
                raise TypeError("Unsupported result type for TrackDecayRate.")

            self.validate_excitation_position()
            self.find_tdr_points()
            self.validate_tdr_points()
            self.calculate_tdr()

    def validate_excitation_position(self):
        r"""Check that the TDR starts in the centre of a sleeper bay."""
        if not isinstance(self.track, (DiscrSlabSingleRailTrack, DiscrBallastedSingleRailTrack)):
            return

        x_mp = array(list(self.track.mount_prop.keys()))
        x_excit = self.ind_excit * self.dx

        before = where(x_mp <= x_excit)[0]
        after = where(x_mp > x_excit)[0]
        if before.size == 0 or after.size == 0:
            raise ValueError(f"The excitation at x = {x_excit:.4f} m does not lie between two supports.")

        x_left, x_right = x_mp[before[-1]], x_mp[after[0]]
        x_centre = (x_left + x_right) / 2
        tol = self.dx / 2 if self.tol_excit is None else self.tol_excit

        deviation = abs(x_excit - x_centre)
        if deviation > tol + 1e-9:
            raise ValueError(f"Excitation not in sleeper bay centre. Deviation {deviation:.4f} > {tol:.4f}.")

    def find_tdr_points(self):
        r"""Determine the TDR measurement positions x_n and their grid indices."""
        if isinstance(self.track, (ArrangedSlabSingleRailTrack, ArrangedBallastedSingleRailTrack)):
            x_mp = array(list(self.track.mount_prop.keys()))
            ind_mp = (x_mp / self.dx).astype(int)

            before = where(ind_mp < self.ind_excit)[0]
            if before.size == 0:
                raise ValueError("No mounting position found before the excitation index.")
            idx_s = int(before[-1])

            x_s = x_mp[idx_s:] - x_mp[idx_s]
            x_sc = convolve(x_s, ones(2) / 2, mode='valid')

            n_required = 68
            if x_s.size < n_required:
                raise ValueError(f"Only {x_s.size} mounting positions lie at or behind the excitation, required {n_required}.")

            def tdr_points_betw1(idx):
                return ((x_s[idx + 1] - x_sc[idx]) / 2) + x_sc[idx]

            def tdr_points_betw2(idx):
                return ((x_sc[idx] - x_s[idx]) / 2) + x_s[idx]

            self.x_tdr = array([
                x_sc[0], tdr_points_betw1(0), x_s[1], tdr_points_betw2(1), x_sc[1], tdr_points_betw1(1),
                x_s[2], tdr_points_betw2(2), x_sc[2], tdr_points_betw1(2), x_s[3], x_sc[3], x_s[4], x_sc[4],
                x_sc[5], x_sc[6], x_sc[7], x_sc[8], x_sc[10], x_sc[12], x_sc[16], x_sc[20], x_sc[24], x_sc[30],
                x_sc[36], x_sc[42], x_sc[48], x_sc[54], x_sc[66],
            ]) - x_sc[0]

            ind_tdr = rint(self.x_tdr.round(5) / self.dx) + self.ind_excit
            self.ind_tdr = list(ind_tdr.astype(int))

        else:
            l_s = 0.6
            x_tdr = array([
                0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5.5, 6.5, 7.5, 8.5,
                10.5, 12.5, 16.5, 20.5, 24.5, 30.5, 36.5, 42.5, 48.5, 54.5, 66.5,
            ]) * l_s
            self.x_tdr = x_tdr - l_s / 2
            ind_tdr = rint(self.x_tdr / self.dx) + self.ind_excit
            self.ind_tdr = list(ind_tdr.astype(int))

    @staticmethod
    def _interval_weights(x):
        r"""Compute the summation weights."""
        x = asarray(x, dtype=float)
        dx = zeros_like(x)
        dx[0] = (x[1] - x[0]) / 2
        dx[1:-1] = (x[2:] - x[:-2]) / 2
        dx[-1] = x[-1] - x[-2]
        return dx

    def validate_tdr_points(self):
        r"""Check that all TDR measurement points lie inside the simulated domain."""
        response = self.as_array(self.response_matrix)
        if response.ndim != 2:
            raise ValueError("response_matrix must be two-dimensional with shape (n_time, n_positions).")

        n_positions = response.shape[1]
        ind_min, ind_max = min(self.ind_tdr), max(self.ind_tdr)
        if ind_min < 0 or ind_max >= n_positions:
            raise ValueError(f"TDR measurement points lie outside the simulated domain.")

    def _calculate_mobility_spectra(self):
        r"""Calculate the mobility spectrum at every TDR measurement point."""
        mobility_rows = []
        frequency = None
        for ind in self.ind_tdr:
            defl = self.response_matrix[:, ind]
            
            # Compute FRF directly without dummy result objects
            tr = TrackResponse.__new__(TrackResponse)
            tr._compute_frf(defl, self.excitation, self.dt)
            frequency, mobility = tr.freq, tr.mobility
            mobility_rows.append(mobility)

        mask = frequency > self.f_min
        if self.f_max is not None:
            mask &= frequency <= self.f_max

        return frequency[mask], array(mobility_rows)[:, mask]

    def calculate_tdr(self):
        r"""Calculate the Track-Decay-Rate (TDR) per DIN EN 15461."""
        freq, mob = self._calculate_mobility_spectra()
        dx_n = self._interval_weights(self.x_tdr)

        ratio_sq = abs(mob) ** 2 / abs(mob[0]) ** 2
        sum_tdr = (ratio_sq * dx_n[:, None]).sum(axis=0)

        self.tdr = 4.343 / sum_tdr
        self.freq = freq

    def dr_min(self):
        return 4.343 / self.x_tdr[-1]

    def _abstract(self) -> None:
        pass


class VehicleResponse(PostProcessing):
    r"""Postprocessing class for vehicle response results. Placeholder for future implementation."""

    def _abstract(self) -> None:
        pass
