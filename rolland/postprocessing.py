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
    r"""Postprocessing class for track response results."""

    @staticmethod
    def calculate_recep(signal, excitation, dt):
        r"""Calculate the receptance of the system.

        Attributes
        ----------
        signal : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the receptance.
        receptance : ndarray
            Receptance of the system.
        """
        signal = PostProcessing.as_array(signal)
        excitation = PostProcessing.as_array(excitation)
        return PostProcessing.frequency_response(signal, excitation, dt)

    @staticmethod
    def calculate_mobility(signal, excitation, dt):
        r"""Calculate the mobility of the system.

        Attributes
        ----------
        signal : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the mobility.
        mobility : ndarray
            Complex mobility of the system. The phase is retained so that
            results stay comparable to the complex mobility of
            :class:`~rolland.methods.analytical.AnalyticalMethods` and can be
            superposed with other mobilities; take ``abs`` at the point of use.
        """
        frequency, receptance = TrackResponse.calculate_recep(signal, excitation, dt)
        mobility = (1j * 2 * pi * frequency) * receptance
        return frequency, mobility

    @staticmethod
    def calculate_accelerance(signal, excitation, dt):
        r"""Calculate the accelerance of the system.

        Attributes
        ----------
        signal : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the accelerance.
        accelerance : ndarray
            Complex accelerance of the system; take ``abs`` at the point of use.
        """
        frequency, receptance = TrackResponse.calculate_recep(signal, excitation, dt)
        omega = 2 * pi * frequency
        accelerance = -(omega**2) * receptance
        return frequency, accelerance

    @staticmethod
    def calc_coupled_mobility(u, phi, offset, excit, pd):
        # Tudo: make more general, [:, 0] is not always clear
        r"""Calculate the coupled mobility of the system.

        Attributes
        ----------
        u : ndarray | object
            Deflection signal of the system.
        phi : ndarray | object
            Torsional rotation signal of the system.
        offset : float
            Offset distance for the coupled mobility calculation.
        excit : ndarray | object
            Time-domain signal of the system's excitation.
        pd : object
            Object containing the time step (dt) attribute.

        Returns
        -------
        frequ : ndarray
            Frequencies corresponding to the coupled mobility.
        mob : ndarray
            Complex coupled mobility of the system; take ``abs`` at the point of use.
        """
        # TUDO: Implement tranformation Matrix, for lateral excentricity
        displ_exc = PostProcessing.as_signal_1d(u) + PostProcessing.as_signal_1d(phi) * offset
        exc = PostProcessing.as_signal_1d(excit)
        frequ, recep = PostProcessing.frequency_response(displ_exc, exc, pd.dt)
        mob = (1j * 2 * pi * frequ) * recep
        return frequ, mob

    @staticmethod
    def calc_coupled_recep(u, phi, offset, excit, pd):
        # Tudo: make more general, [:, 0] is not always clear
        r"""Calculate the coupled receptance of the system.

        Attributes
        ----------
        u : ndarray | object
            Deflection signal of the system.
        phi : ndarray | object
            Torsional rotation signal of the system.
        offset : float
            Offset distance for the coupled receptance calculation.
        excit : ndarray | object
            Time-domain signal of the system's excitation.
        pd : object
            Object containing the time step (dt) attribute.

        Returns
        -------
        frequ : ndarray
            Frequencies corresponding to the coupled receptance.
        recep : ndarray
            Coupled receptance of the system.
        """
        displ_exc = PostProcessing.as_signal_1d(u) + PostProcessing.as_signal_1d(phi) * offset
        exc = PostProcessing.as_signal_1d(excit)
        return PostProcessing.frequency_response(displ_exc, exc, pd.dt)

    @staticmethod
    def calculate_mov_recep(u, excit, pd, skip):
        # Tudo: make more general, [:, 0] is not always clear
        r"""Calculate the mobility and receptance of the system.

        Attributes
        ----------
        u : ndarray | object
            Deflection signal of the system.
        excit : ndarray | object
            Time-domain signal of the system's excitation.
        pd : object
            Object containing the time step (dt) attribute.
        skip : int
            Number of samples to skip.

        Returns
        -------
        frequ : ndarray
            Frequencies corresponding to the mobility/receptance.
        recep : ndarray
            Receptance of the system.
        """
        signal = PostProcessing.as_signal_1d(u)[skip:]
        exc = PostProcessing.as_signal_1d(excit)[skip:]
        return PostProcessing.frequency_response(signal, exc, pd.dt)

    @staticmethod
    def calc_coupled_mov_recep(u, phi, offset, excit, pd, skip):
        # Tudo: make more general, [:, 0] is not always clear
        r"""Calculate the coupled mobility and receptance of the system.

        Attributes
        ----------
        u : ndarray | object
            Deflection signal of the system.
        phi : ndarray | object
            Torsional rotation signal of the system.
        offset : float
            Offset distance for the coupled receptance calculation.
        excit : ndarray | object
            Time-domain signal of the system's excitation.
        pd : object
            Object containing the time step (dt) attribute.
        skip : int
            Number of samples to skip.

        Returns
        -------
        frequ : ndarray
            Frequencies corresponding to the coupled receptance.
        recep : ndarray
            Coupled receptance of the system.
        """
        signal = PostProcessing.as_signal_1d(u)[skip:] + PostProcessing.as_signal_1d(phi)[skip:] * offset
        exc = PostProcessing.as_signal_1d(excit)[skip:]
        return PostProcessing.frequency_response(signal, exc, pd.dt)


class PointResponse(TrackResponse):
    r"""Postprocessing class for point response results."""

    def _abstract(self) -> None:
        pass

    @classmethod
    def calculate_recep_1d(cls, response, excitation, dt):
        r"""
        Calculate the Point response receptance.

        The point response receptance is calculated as the receptance between the excitation and
        response at the same position.

        Attributes
        ----------
        response : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the receptance.
        receptance : ndarray
            Point ResponseReceptance of the system.
        """
        #Tudo: Difference in data structure of time between methods
        response = PostProcessing.as_signal_1d(response)
        excitation = PostProcessing.as_signal_1d(excitation)
        return cls.calculate_recep(response, excitation, dt)

    @classmethod
    def calculate_mobility_1d(cls, response, excitation, dt):
        r"""
        Calculate the Point response mobility.

        The point response mobility is calculated as the mobility between the excitation and
        response at the same position.

        Attributes
        ----------
        response : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the mobility.
        mobility : ndarray
            Point Response Mobility of the system.
        """
        response = PostProcessing.as_signal_1d(response)
        excitation = PostProcessing.as_signal_1d(excitation)
        return cls.calculate_mobility(response, excitation, dt)

    @classmethod
    def calculate_accelerance_1d(cls, response, excitation, dt):
        r"""Calculate the Point response accelerance.

        The point response accelerance is calculated as the accelerance between the excitation and
        response at the same position.

        Attributes
        ----------
        response : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the accelerance.
        accelerance : ndarray
            Point ResponseAccelerance of the system.
        """
        response = PostProcessing.as_signal_1d(response)
        excitation = PostProcessing.as_signal_1d(excitation)
        return cls.calculate_accelerance(response, excitation, dt)

class TransferResponse(TrackResponse):
    r"""Postprocessing class for transfer response results."""

    def _abstract(self) -> None:
        pass

    @classmethod
    def calculate_recep_transfer(cls, response, excitation, dt, position=0):
        r"""
        Calculate the Transfer receptance.

        The transfer response receptance is calculated as the receptance between the excitation and
        response at a defined position.

        Attributes
        ----------
        response : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.
        position : int, optional, default=0
            Index of the position in the response signal to calculate the transfer receptance.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the receptance.
        receptance : ndarray
            Point Response Receptance of the system.
        """
        excitation = PostProcessing.as_signal_1d(excitation)
        response = cls.as_signal_1d(response, index=position)
        return cls.calculate_recep(response, excitation, dt)

    @classmethod
    def calculate_mobility_transfer(cls, response, excitation, dt, position=0):
        r"""Calculate the transfer mobility.

        The transfer mobility is calculated as the mobility between the excitation and
        response at a defined position.

        Attributes
        ----------
        response : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.
        position : int, optional, default=0
            Index of the position in the response signal to calculate the transfer mobility.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the mobility.
        mobility : ndarray
            Point Response Mobility of the system.
        """
        excitation = PostProcessing.as_signal_1d(excitation)
        response = cls.as_signal_1d(response, index=position)
        return cls.calculate_mobility(response, excitation, dt)

    @classmethod
    def calculate_accelerance_transfer(cls, response, excitation, dt, position=0):
        r"""Calculate the transfer accelerance.

        The transfer accelerance is calculated as the accelerance between the excitation and
        response at a defined position.

        Attributes
        ----------
        response : ndarray | object
            Time-domain signal of the system's response.
        excitation : ndarray | object
            Time-domain signal of the system's excitation.
        dt : float
            Time step between samples.
        position : int, optional, default=0
            Index of the position in the response signal to calculate the transfer accelerance.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the accelerance.
        accelerance : ndarray
            Point Response Accelerance of the system.
        """
        excitation = PostProcessing.as_signal_1d(excitation)
        response = cls.as_signal_1d(response, index=position)
        return cls.calculate_accelerance(response, excitation, dt)

@dataclass(kw_only=True)
class TrackDecayRate(TrackResponse):
    r"""Postprocessing class for TDR (Track-Decay-Rate).

    This class calculates and stores the Track-Decay-Rate (TDR) based on :cite:`EN15461:2008`.

    The decay rate is derived from a ratio of mobilities, so the shape of the
    excitation cancels out and any excitation signal may be used -- as long as it
    is stationary, since the TDR is defined relative to a fixed reference point.

    All inputs are plain arrays and scalars, taken from the simulation objects at
    the call site. For the numerical FDM methods::

        tdr = TrackDecayRate(
            response_matrix=defl.deflection,
            excitation=defl.force,
            dt=defl.discr.dt,
            dx=defl.discr.dx,
            ind_excit=defl.ind_excit,   # see note on the reference index below
            track=defl.track,
        )

    For a :class:`~rolland.deflection.Deflection` simulation, note that it stores
    only every ``skip``-th time step, so both the time step and the excitation
    have to account for it::

        tdr = TrackDecayRate(
            response_matrix=defl.u_z_obs,
            excitation=defl.excit.force.data[:: defl.skip],
            dt=defl.discr.dt * defl.skip,
            dx=defl.discr.dx,
            ind_excit=round(defl.excit.x_excit / defl.discr.dx),
            track=defl.track,
        )

    The simulation must have been run with ``store='full'``: the TDR points are
    addressed by spatial grid index, which the single observation points of
    ``store='observe'`` or ``store='excit'`` do not provide.

    On a discretely supported track the excitation must sit in the centre of a
    sleeper bay, since the measurement grid is built relative to that point --
    see :meth:`validate_excitation_position`.

    .. note:: **On the reference index.**

        ``ind_excit`` selects the reference point :math:`x_0` against which all
        mobilities are normalised. That is the point mobility and hence the
        largest value of the whole series, so an index off by one grid cell
        distorts the entire TDR curve rather than a single summand -- which is
        why the two examples above derive it differently.

        A devito simulation injects its force through
        ``force.inject(coordinates=x_excit)``, i.e. interpolated onto the exact
        physical position, so no single grid point is the "true" one and the
        nearest one is the best approximation: ``round(x_excit / dx)``. Grid point
        *i* sits at :math:`i \cdot dx` because :class:`~rolland.domainsetup.DomSetup`
        fixes the grid origin at 0.0; for a shifted origin the offset would have to
        be subtracted first.

        The numerical FDM methods instead compute their own index by truncation,
        ``int(x_excit / dx)``, and write the force into exactly that cell. Their
        ``ind_excit`` must therefore be reused as-is -- rounding it independently
        could point at a neighbouring cell of the actual force application.

    Attributes
    ----------
    response_matrix : numpy.ndarray
        Deflection over time and the full spatial grid, shape (n_time, n_positions).
        Time is axis 0, matching the output of both :class:`~rolland.deflection.Deflection`
        and the numerical methods. The spatial axis must span the whole grid, since the
        TDR points are addressed by grid index.
    excitation : numpy.ndarray
        Time-domain excitation signal at the excitation location ("direct FRF", x_0 = 0).
    dt : float
        Time step of ``response_matrix`` [s].
    dx : float
        Spatial grid spacing [m].
    ind_excit : int
        Spatial grid index of the excitation. Must be supplied explicitly by the caller.
    track : object
        Track object (only used to select the point layout scheme per clause 6.7 of the standard).
    f_min : float, default=0.0
        Lower band limit, exclusive [Hz]. The default of 0.0 discards the DC line,
        where the excitation spectrum vanishes and the mobility is meaningless.
    f_max : float | None, default=None
        Upper band limit, inclusive [Hz]. ``None`` keeps everything up to Nyquist.
    tol_excit : float | None, default=None
        Tolerance for the sleeper bay centre check of :meth:`validate_excitation_position` [m].
        ``None`` uses half a grid spacing, i.e. the excitation has to be the grid point
        closest to the centre. Only relevant for discretely supported tracks.
    tdr : numpy.ndarray
        Track-Decay-Rate vector [dB/m].
    ind_tdr : list[int]
        Spatial indices of the TDR measurement points x_n.
    x_tdr : numpy.ndarray
        Distances x_n of the TDR points from the excitation point [m] (x_0 = 0).
    filter : str | None
        Filter type.
    freq : numpy.ndarray
        Frequency vector [Hz].
    """

    response_matrix: ndarray
    excitation: ndarray
    dt: float
    dx: float
    ind_excit: int
    track: object
    f_min: float = 0.0
    f_max: float | None = None
    tol_excit: float | None = None
    tdr: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    filter: str | None = None
    freq: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})
    ind_tdr: list[int] = field(default_factory=list)
    x_tdr: ndarray = field(default_factory=lambda: array([]), metadata={"default_repr": "numpy.array([])"})

    def __post_init__(self):
        """Post-initialization to check the excitation, find TDR points and calculate TDR."""
        self.validate_excitation_position()
        self.find_tdr_points()
        self.validate_tdr_points()
        self.calculate_tdr()

    def validate_excitation_position(self):
        r"""Check that the TDR starts in the centre of a sleeper bay.

        The excitation is accepted when its grid position lies within
        ``tol_excit`` of the bay centre; the default of half a grid spacing
        means it has to be the grid point closest to that centre. Continuously
        supported tracks are skipped: their support layer is smeared out over
        the length of the track, so there are no sleeper bays to be centred in.

        Raises
        ------
        ValueError
            If the excitation does not lie between two supports, or if it is
            further than ``tol_excit`` from the centre of its sleeper bay.
        """
        if not isinstance(self.track, (DiscrSlabSingleRailTrack, DiscrBallastedSingleRailTrack)):
            return

        x_mp = array(list(self.track.mount_prop.keys()))
        x_excit = self.ind_excit * self.dx

        before = where(x_mp <= x_excit)[0]
        after = where(x_mp > x_excit)[0]
        if before.size == 0 or after.size == 0:
            msg = (
                f"The excitation at x = {x_excit:.4f} m (grid index {self.ind_excit}) does not lie "
                f"between two supports, which span {x_mp[0]:.4f} m to {x_mp[-1]:.4f} m. The TDR "
                f"starts in a sleeper bay, so the excitation must sit inside the supported track."
            )
            raise ValueError(msg)

        x_left, x_right = x_mp[before[-1]], x_mp[after[0]]
        x_centre = (x_left + x_right) / 2
        tol = self.dx / 2 if self.tol_excit is None else self.tol_excit

        # The epsilon keeps a centre falling exactly between two grid points from being
        # rejected for both of its two equally close neighbours.
        deviation = abs(x_excit - x_centre)
        if deviation > tol + 1e-9:
            msg = (
                f"The excitation at x = {x_excit:.4f} m (grid index {self.ind_excit}) is not in a "
                f"sleeper bay centre: it lies between the supports at {x_left:.4f} m and "
                f"{x_right:.4f} m, whose centre is {x_centre:.4f} m -- a deviation of "
                f"{deviation:.4f} m > {tol:.4f} m. The TDR is defined for an excitation in the bay "
                f"centre, so move the excitation to {x_centre:.4f} m (grid index "
                f"{round(x_centre / self.dx)}), refine dx so that the centre is met more closely, "
                f"or widen tol_excit if the remaining deviation is acceptable."
            )
            raise ValueError(msg)

    def find_tdr_points(self):
        r"""Determine the TDR measurement positions x_n and their grid indices.

        Implements the near-field/far-field measurement grid described in
        clause 6.7 of the standard: for an arranged (discretely supported)
        track, positions are derived from the actual fastener spacing
        (``track.mount_prop``); otherwise a fixed 29-point grid is used,
        scaled by an assumed sleeper spacing of 0.6 m.

        Sets
        ----
        x_tdr : numpy.ndarray
            Distances of the 29 TDR points from the excitation point [m].
        ind_tdr : list[int]
            Corresponding spatial grid indices, offset by ``ind_excit``.
        """
        if isinstance(self.track, (ArrangedSlabSingleRailTrack, ArrangedBallastedSingleRailTrack)):
            x_mp = array(list(self.track.mount_prop.keys()))
            ind_mp = (x_mp / self.dx).astype(int)

            before = where(ind_mp < self.ind_excit)[0]
            if before.size == 0:
                msg = (
                    f"No mounting position found before the excitation index {self.ind_excit}. "
                    f"The excitation must lie behind the first fastener of the track."
                )
                raise ValueError(msg)
            idx_s = int(before[-1])

            x_s = x_mp[idx_s:] - x_mp[idx_s]
            x_sc = convolve(x_s, ones(2) / 2, mode='valid')

            # The farthest point of the standard grid is x_sc[66], i.e. the centre
            # between the 67th and 68th fastener behind the excitation.
            n_required = 68
            if x_s.size < n_required:
                msg = (
                    f"Only {x_s.size} mounting positions lie at or behind the excitation, "
                    f"but {n_required} are required to place all 29 TDR measurement points. "
                    f"Extend the track or move the excitation closer to its start."
                )
                raise ValueError(msg)

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
        r"""Compute the summation weights :math:`\Delta x_n` per Annex A.

        :math:`\Delta x_n` is the local length of track that point n
        represents in the summation, not its distance from the excitation:
        for interior points it is the symmetric midpoint distance
        :math:`(x_{n+1} - x_{n-1}) / 2`; for the first point (:math:`x_0 = 0`,
        at the excitation) it only extends forward to the midpoint with its
        neighbour, since distance cannot be negative; for the last point it
        is mirrored symmetrically outward, i.e. equal to its distance from
        the previous point.

        Parameters
        ----------
        x : ndarray
            Monotonically increasing measurement-point distances from the
            excitation, x[0] == 0.

        Returns
        -------
        ndarray
            Interval weights :math:`\Delta x_n`, same shape as ``x``.
        """
        x = asarray(x, dtype=float)
        dx = zeros_like(x)
        dx[0] = (x[1] - x[0]) / 2
        dx[1:-1] = (x[2:] - x[:-2]) / 2
        dx[-1] = x[-1] - x[-2]
        return dx

    def validate_tdr_points(self):
        r"""Check that all TDR measurement points lie inside the simulated domain.

        The points are addressed by spatial grid index, so ``response_matrix``
        must cover the whole grid and must reach at least ``x_tdr[-1]`` beyond
        the excitation. Without this check an out-of-range index would either
        raise an opaque ``IndexError`` or, for negative indices, silently wrap
        around and produce a plausible but wrong decay rate.

        Raises
        ------
        ValueError
            If ``response_matrix`` is not two-dimensional or does not contain
            all TDR measurement points.
        """
        response = self.as_array(self.response_matrix)
        if response.ndim != 2:
            msg = (
                f"response_matrix must be two-dimensional with shape (n_time, n_positions), "
                f"got ndim={response.ndim}. The TDR needs the deflection over the full "
                f"spatial grid, so a simulation storing only single observation points "
                f"cannot be used."
            )
            raise ValueError(msg)

        n_positions = response.shape[1]
        ind_min, ind_max = min(self.ind_tdr), max(self.ind_tdr)
        if ind_min < 0 or ind_max >= n_positions:
            # Far too few positions means the response is a handful of observation
            # points rather than the spatial grid the indices refer to.
            cause = (
                "The response does not cover the spatial grid at all -- run the simulation "
                "with store='full' instead of single observation points."
                if n_positions < len(self.ind_tdr) else
                f"The farthest point is {self.x_tdr[-1]:.2f} m behind the excitation, so the "
                f"track must extend at least that far beyond it."
            )
            msg = (
                f"TDR measurement points lie outside the simulated domain: required grid "
                f"index range [{ind_min}, {ind_max}], available [0, {n_positions - 1}]. {cause}"
            )
            raise ValueError(msg)

    def _calculate_mobility_spectra(self):
        r"""Calculate the mobility spectrum at every TDR measurement point.

        Restricts the spectra to the band ``(f_min, f_max]``, which also
        removes the DC line where the excitation spectrum vanishes.

        Returns
        -------
        frequency : ndarray
            Frequencies corresponding to the mobility spectra.
        mobility : ndarray
            Complex mobility at each TDR point, shape (n_points, n_freq), with
            row 0 corresponding to the reference point at the excitation (x_0 = 0).
        """
        mobility_rows = []
        frequency = None
        for ind in self.ind_tdr:
            # Time is axis 0 of the response matrix, position is axis 1.
            defl = self.response_matrix[:, ind]
            frequency, mobility = TrackResponse.calculate_mobility(defl, self.excitation, self.dt)
            mobility_rows.append(mobility)

        mask = frequency > self.f_min
        if self.f_max is not None:
            mask &= frequency <= self.f_max

        return frequency[mask], array(mobility_rows)[:, mask]

    def calculate_tdr(self):
        r"""Calculate the Track-Decay-Rate (TDR) per DIN EN 15461, Annex A, Eq. (A.3).

        .. math::
            DR \approx \dfrac{4.343}
            {\sum_{n=0}^{n_{max}} \dfrac{|A(x_n)|^2}{|A(x_0)|^2}\,\Delta x_n}

        The summation runs over all measurement points including the
        reference point itself (n = 0, where the ratio is 1), each weighted
        by its local interval :math:`\Delta x_n` from :meth:`_interval_weights`.

        Sets
        ----
        tdr : numpy.ndarray
            Track-Decay-Rate [dB/m] for every frequency line in the band.
        freq : numpy.ndarray
            Corresponding frequency vector [Hz].
        """
        freq, mob = self._calculate_mobility_spectra()
        dx_n = self._interval_weights(self.x_tdr)

        ratio_sq = abs(mob) ** 2 / abs(mob[0]) ** 2      # |A(xn)/A(x0)|^2, all n
        sum_tdr = (ratio_sq * dx_n[:, None]).sum(axis=0)  # Σ_{n=0}^{nmax} ... Δx_n

        self.tdr = 4.343 / sum_tdr
        self.freq = freq

    def dr_min(self):
        r"""Calculate the minimum measurable decay rate per Eq. (2) of the standard.

        Used as an acceptance check (clause 7/8): a value from
        :meth:`calculate_tdr` in a given band is considered unreliable if it
        is less than twice this minimum, since the true decay could not have
        been fully resolved within the measured distance ``x_max``.

        Returns
        -------
        float
            Minimum decay rate :math:`DR_{min} = 4.343 / x_{max}` [dB/m],
            where :math:`x_{max}` is the distance of the farthest TDR point.
        """
        return 4.343 / self.x_tdr[-1]

    def _abstract(self) -> None:
        pass


class VehicleResponse(PostProcessing):
    r"""Postprocessing class for vehicle response results. Placeholder for future implementation."""

    def _abstract(self) -> None:
        pass
