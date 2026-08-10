"""Postprocessing classes.

.. autosummary::
    :toctree: postprocessing

    PostProcessing
"""
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from numpy import ones, pi  # noqa: A004
from numpy.fft import fft, fftfreq


class PostProcessing(ABC):
    r"""Abstract base class for postprocessing classes."""

    @abstractmethod
    def validate_postprocessing(self):
        """Validate the postprocessing methods."""

    @staticmethod
    def fast_fourier_tr(tsignal, dt):
        """Calculate the Fast Fourier Transform (FFT) of a time signal.

        Parameters
        ----------
        tsignal : numpy.ndarray
            Time signal to transform.
        dt : float
            Time step between samples.

        Returns
        -------
        tuple
            Frequencies and FFT of the signal.
        """
        samples = len(tsignal)
        window = ones(samples)
        fftrans = 2.0 / samples * fft(tsignal[:samples] * window)
        fftfre = fftfreq(samples, dt)
        return fftfre[0 : samples // 2], fftrans[0 : samples // 2]

    @staticmethod
    def plot(
        arrays, labels, title='Universal Plot', x_label='X-axis', y_label='Y-axis', colors=None, plot_type='loglog',
    ):
        """Universal plot function for multiple data sets.

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

class DEVITO_PP(PostProcessing):  # noqa: N801
    r"""Postprocessing class for DEVITO results."""

    def validate_postprocessing(self):
        """Validate the postprocessing methods."""

    @staticmethod
    def calculate_mobility(u, excit, pd):
        """Calculate mobility from displacement and excitation force.

        Parameters
        ----------
        u : numpy.ndarray
            Displacement time series.
        excit : SparseTimeFunction
            Excitation force function.
        pd : DomSetup
            Domain setup instance.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequency array and mobility spectrum.
        """
        displ_exc = u[:,0]

        # Extract excitation force data
        exc = excit.data[:,0]

        # Perform FFT on displacement and excitation force
        displ_fft = fft(displ_exc)
        exc_fft = fft(exc)
        frequ = fftfreq(pd.nt, pd.dt)

        # Calculate mobility (velocity/displacement)
        recep = displ_fft / exc_fft
        mob = abs((1j * 2 * pi * frequ) * recep)

        return frequ, mob

    @staticmethod
    def calculate_recep(u, excit, pd):
        """Calculate receptance from displacement and excitation force.

        Parameters
        ----------
        u : Function
            Displacement function.
        excit : SparseTimeFunction
            Excitation force function.
        pd : DomSetup
            Domain setup instance.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequency array and receptance spectrum.
        """
        displ_exc = u.data[:, 0]

        # Extract excitation force data
        exc = excit.data[:, 0]

        # Perform FFT on displacement and excitation force
        displ_fft = fft(displ_exc)
        exc_fft = fft(exc)
        frequ = fftfreq(pd.nt, pd.dt)

        # Calculate mobility (velocity/displacement)
        recep = displ_fft / exc_fft

        return frequ, recep

    @staticmethod
    def calc_coupled_mobility(u, phi, offset, excit, pd):
        """Calculate coupled mobility taking rotational offset into account.

        Parameters
        ----------
        u : numpy.ndarray
            Translational displacement time series.
        phi : numpy.ndarray
            Rotational displacement time series.
        offset : float
            Distance offset.
        excit : SparseTimeFunction
            Excitation force function.
        pd : DomSetup
            Domain setup instance.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequency array and coupled mobility spectrum.
        """
        displ_exc = u[:, 0] + phi[:, 0] * offset
        # TODO(mantelmax): Implement transformation matrix for lateral eccentricity  # noqa: TD003, FIX002

        exc = excit.data[:, 0]
        displ_fft = fft(displ_exc)
        exc_fft = fft(exc)
        frequ = fftfreq(pd.nt, pd.dt)

        # Calculate mobility (velocity/displacement)
        recep = displ_fft / exc_fft
        mob = abs((1j * 2 * pi * frequ) * recep)
        return frequ, mob

    @staticmethod
    def calc_coupled_recep(u, phi, offset, excit, pd):
        """Calculate coupled receptance taking rotational offset into account.

        Parameters
        ----------
        u : numpy.ndarray
            Translational displacement time series.
        phi : numpy.ndarray
            Rotational displacement time series.
        offset : float
            Distance offset.
        excit : SparseTimeFunction
            Excitation force function.
        pd : DomSetup
            Domain setup instance.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequency array and coupled receptance spectrum.
        """
        displ_exc = u[:, 0] + phi[:, 0] * offset

        exc = excit.data[:, 0]
        displ_fft = fft(displ_exc)
        exc_fft = fft(exc)
        frequ = fftfreq(displ_fft.shape[0], pd.dt)

        recep = displ_fft / exc_fft
        return frequ, recep

    @staticmethod
    def calculate_mov_recep(u, excit, pd, skip):
        """Calculate moving receptance skipping initial transient time steps.

        Parameters
        ----------
        u : numpy.ndarray
            Displacement time series.
        excit : Function
            Excitation force function.
        pd : DomSetup
            Domain setup instance.
        skip : int
            Number of initial time steps to skip.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequency array and moving receptance spectrum.
        """
        # Perform FFT on displacement and excitation force
        displ_fft = fft(u[skip:])
        exc_fft = fft(excit.data[skip:])
        frequ = fftfreq(displ_fft.shape[0], pd.dt)

        # Calculate mobility (velocity/displacement)
        recep = displ_fft / exc_fft
        return frequ, recep

    @staticmethod
    def calc_coupled_mov_recep(u, phi, offset, excit, pd, skip):
        """Calculate coupled moving receptance skipping initial transient time steps.

        Parameters
        ----------
        u : Function
            Translational displacement function.
        phi : Function
            Rotational displacement function.
        offset : float
            Distance offset.
        excit : Function
            Excitation force function.
        pd : DomSetup
            Domain setup instance.
        skip : int
            Number of initial time steps to skip.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequency array and coupled moving receptance spectrum.
        """
        displ_exc = u.data[skip:] + phi.data[skip:] * offset

        exc = excit.data[skip:]
        displ_fft = fft(displ_exc)
        exc_fft = fft(exc)
        frequ = fftfreq(displ_fft.shape[0], pd.dt)

        recep = displ_fft / exc_fft
        return frequ, recep
