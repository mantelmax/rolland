"""Defines sound radiation classes for rail noise calculation.

.. autosummary::
    :toctree: radiation

    Radiation
    SimplifiedMonopoleRadiation
"""

import concurrent.futures
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gmsh
import librosa
import numpy as np
import soundfile as sf
from numpy import ndarray, pi
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from rolland.components import Rail
    from rolland.deflection import Deflection

# Constants
P_REF = 20e-6  # Reference sound pressure [Pa]
FADE_DURATION = 5e-3  # Fade-in/out duration of WAV export [s]
NORMALIZE_PEAK = 0.944  # Peak value for normalized WAV export [-]
MIN_SOURCE_DISTANCE = 1e-6  # Minimum distance between source and receiver [m]
GMSH_MESH_OPTIONS = (('Mesh.Algorithm', 6), ('Mesh.Optimize', 1), ('Mesh.OptimizeNetgen', 1))
A_WEIGHTING_POLES = (20.598997, 107.65265, 737.86223, 12194.217)  # IEC 61672-1 pole frequencies [Hz]
A_WEIGHTING_GAIN_1KHZ = 2.0  # Normalization to 0 dB at 1 kHz [dB]


def a_weighting(freq: ndarray) -> ndarray:
    r"""Compute the linear A-weighting factors according to IEC 61672-1.

    .. math::

        R_A(f) = \frac{f_4^2 f^4}{(f^2 + f_1^2) \sqrt{(f^2 + f_2^2)(f^2 + f_3^2)} (f^2 + f_4^2)},
        \qquad A(f) = 20 \log_{10} R_A(f) + 2.00\,dB

    Parameters
    ----------
    freq : numpy.ndarray
        Frequency array :math:`[Hz]`.

    Returns
    -------
    numpy.ndarray
        Real-valued (zero-phase) amplitude weighting factors :math:`[-]`.
    """
    f1, f2, f3, f4 = A_WEIGHTING_POLES
    f_sq = np.asarray(freq, dtype=float) ** 2

    r_a = (f4**2 * f_sq**2) / ((f_sq + f1**2) * np.sqrt((f_sq + f2**2) * (f_sq + f3**2)) * (f_sq + f4**2))
    return r_a * 10 ** (A_WEIGHTING_GAIN_1KHZ / 20)


@dataclass(kw_only=True)
class Radiation(ABC):
    r"""Abstract base class for sound radiation models.

    Builds a 3D surface mesh of the rail by extruding the 2D rail outline
    along the track axis (GMSH) and evaluates triangle centres, areas and
    outward normals. The mesh is centred around ``x_centre``. Deflections are
    taken from a :class:`~rolland.deflection.Deflection` result, which must be
    computed with ``store='full'``.

    Attributes
    ----------
    result : Deflection
        Simulation result providing the deflection wavefields
        ``u_y_obs``, ``u_z_obs`` and ``phi_x_obs``.
    rail : Rail
        Rail whose outline ``rl_geo`` defines the radiating surface.
    x_centre : float, optional
        Centre position of the radiating rail section :math:`[m]`.
        Defaults to the excitation position ``result.excit.x_excit``.
    mesh_length : float, default=10.0
        Length of the radiating rail section :math:`[m]`.
    mesh_size : float, default=0.05
        Target element size of the surface mesh :math:`[m]`.
    num_pts : int, default=30
        Number of points for the rail outline interpolation :math:`[-]`.
    dt : float
        Effective time step of the deflection data :math:`[s]`.
    dx : float
        Spatial grid spacing of the deflection data :math:`[m]`.
    triangle_coords : ndarray
        Mesh node coordinates :math:`[m]`, shape ``(n_nodes, 3)``.
    triangle_index : ndarray
        Node indices of each triangle, shape ``(n_triangles, 3)``.
    base_area : ndarray
        Triangle areas :math:`[m^2]`.
    base_centre : ndarray
        Triangle centres in mesh coordinates :math:`[m]`.
    base_norm : ndarray
        Outward unit normals of the triangles :math:`[-]`.
    """

    result: 'Deflection'
    rail: 'Rail'
    x_centre: float | None = None
    mesh_length: float = 10.0
    mesh_size: float = 0.05
    num_pts: int = 30

    dt: float = field(init=False)
    dx: float = field(init=False)
    triangle_coords: ndarray = field(init=False)
    triangle_index: ndarray = field(init=False)
    base_area: ndarray = field(init=False)
    base_centre: ndarray = field(init=False)
    base_norm: ndarray = field(init=False)
    _x_mid: float = field(init=False, repr=False)

    def __post_init__(self):
        """Validate the result and build the rail surface mesh."""
        if getattr(self.result, 'store', None) != 'full':
            msg = "Radiation requires a Deflection result computed with store='full'."
            raise ValueError(msg)

        if self.x_centre is None:
            x_excit = getattr(self.result.excit, 'x_excit', None)
            if x_excit is None:
                msg = 'x_centre must be provided for excitations without a fixed position.'
                raise ValueError(msg)
            self.x_centre = float(x_excit)
        self._x_mid = float(self.x_centre)

        self.dt = self.result.discr.dt * self.result.skip
        self.dx = self.result.discr.dx

        self._setup_mesh()

    @abstractmethod
    def _calculate(self) -> None:
        """Calculate the sound pressure. Must be implemented by subclasses."""

    def _setup_mesh(self) -> None:
        """Interpolate the rail outline, create the 3D mesh and evaluate triangle properties."""
        contour = self._interpolate_contour_2d(np.asarray(self.rail.rl_geo), self.num_pts)
        self.triangle_coords, self.triangle_index = self._create_rail_mesh(
            contour,
            self.mesh_size,
            self.mesh_length,
        )
        self.base_area, self.base_centre, self.base_norm = self._calculate_centre_and_area_triangles(
            self.triangle_coords,
            self.triangle_index,
        )

    def view_mesh(self, measurement_point: ndarray | None = None) -> None:
        """Open the rail mesh in the GMSH GUI.

        Parameters
        ----------
        measurement_point : numpy.ndarray, optional
            Coordinates :math:`[x, y, z]` of a point to display :math:`[m]`.
            Defaults to the model's measurement point, if available.
        """
        if measurement_point is None:
            measurement_point = getattr(self, 'measurement_point', None)

        if not gmsh.isInitialized():
            gmsh.initialize()

        try:
            gmsh.model.add('rail_mesh')

            contour = self._interpolate_contour_2d(np.asarray(self.rail.rl_geo), self.num_pts)
            x_start = self._x_mid - self.mesh_length / 2
            self._build_gmsh_geometry(
                np.c_[np.full(contour.shape[0], x_start), contour],
                self.mesh_size,
                self.mesh_length,
            )

            if measurement_point is not None:
                mp = np.asarray(measurement_point, dtype=float)
                view_tag = gmsh.view.add('measurement_point_view')
                v_idx = gmsh.view.getIndex(view_tag)

                view_opts = {
                    'PointSize': 20,
                    'ShowScale': 0,
                    'RangeType': 2,
                    'CustomMin': 0.0,
                    'CustomMax': 1.0,
                    'ColormapNumber': 1,
                }
                for key, val in view_opts.items():
                    gmsh.option.setNumber(f'View[{v_idx}].{key}', val)

                gmsh.view.addListData(view_tag, 'SP', 1, [mp[0], mp[1], mp[2], 1.0])

                point_opts = {'PointSize': 25, 'PointType': 2, 'Visible': 1, 'ShowScale': 0}
                for key, val in point_opts.items():
                    gmsh.option.setNumber(f'View[{view_tag - 1}].{key}', val)

            gmsh.model.geo.synchronize()
            gmsh.option.setNumber('General.Trackball', 0)
            gmsh.fltk.run()

        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    @staticmethod
    def _interpolate_contour_2d(array: ndarray, n: int) -> ndarray:
        """Resample a 2D contour equidistantly along its arc length.

        Parameters
        ----------
        array : numpy.ndarray
            Contour points, shape ``(m, 2)``.
        n : int
            Number of output points.

        Returns
        -------
        numpy.ndarray
            Resampled contour, shape ``(n, 2)``.
        """
        array = np.atleast_2d(array)
        if array.shape[0] < 2 or array.shape[1] != 2 or n < 2:
            msg = 'Invalid contour shape or point count.'
            raise ValueError(msg)

        # Arc length parametrisation
        contour_s = np.zeros(array.shape[0], dtype=float)
        contour_s[1:] = np.cumsum(np.linalg.norm(np.diff(array, axis=0), axis=1))

        new_contour_s = np.linspace(0.0, contour_s[-1], n)
        x_new = np.interp(new_contour_s, contour_s, array[:, 0])
        y_new = np.interp(new_contour_s, contour_s, array[:, 1])

        return np.c_[x_new, y_new]

    @staticmethod
    def _build_gmsh_geometry(rail_geometry_3d: ndarray, mesh_size: float, length: float) -> None:
        """Create the extruded rail geometry in the active GMSH model and mesh its surface."""
        point_tags = [gmsh.model.geo.addPoint(p[0], p[1], p[2], mesh_size) for p in rail_geometry_3d]
        n_pts = len(point_tags)
        line_tags = [gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n_pts]) for i in range(n_pts)]

        curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        gmsh.model.geo.synchronize()

        gmsh.model.geo.extrude(dimTags=[(2, surface)], dx=length, dy=0, dz=0)
        gmsh.model.geo.synchronize()

        for opt, val in GMSH_MESH_OPTIONS:
            gmsh.option.setNumber(opt, val)

        gmsh.model.mesh.generate(2)

    @classmethod
    def _create_rail_mesh(cls, rail_contour: ndarray, mesh_size: float, length: float) -> tuple[ndarray, ndarray]:
        """Create a 3D surface mesh of the rail starting at :math:`x = 0`.

        Parameters
        ----------
        rail_contour : numpy.ndarray
            2D rail outline :math:`[y, z]`, shape ``(n, 2)`` :math:`[m]`.
        mesh_size : float
            Target element size :math:`[m]`.
        length : float
            Extrusion length along the track axis :math:`[m]`.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Node coordinates and triangle node indices.
        """
        rail_geometry_3d = np.c_[np.zeros(rail_contour.shape[0]), rail_contour]

        if not gmsh.isInitialized():
            gmsh.initialize()

        try:
            gmsh.model.add('rail')
            cls._build_gmsh_geometry(rail_geometry_3d, mesh_size, length)

            _, _, element_node_tags = gmsh.model.mesh.getElements(2)
            triangle_tags = np.asarray(element_node_tags[0]).reshape(-1, 3)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            triangle_coords = np.asarray(node_coords).reshape(-1, 3)

            tag_to_index = {tag: i for i, tag in enumerate(node_tags)}
            triangle_index = np.vectorize(tag_to_index.get)(triangle_tags)

            return triangle_coords, triangle_index

        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    @staticmethod
    def _calculate_centre_and_area_triangles(
        triangle_coords: ndarray,
        triangle_index: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Calculate centre, area and outward unit normal of each triangle.

        Parameters
        ----------
        triangle_coords : numpy.ndarray
            Mesh node coordinates :math:`[m]`.
        triangle_index : numpy.ndarray
            Node indices of each triangle.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Triangle areas, centres and outward unit normals.
        """
        p1, p2, p3 = (triangle_coords[triangle_index[:, i]] for i in range(3))
        cross_product = np.cross(p1 - p2, p1 - p3)

        area = np.linalg.norm(cross_product, axis=1) * 0.5
        centre = (p1 + p2 + p3) / 3.0
        norm = cross_product / np.linalg.norm(cross_product, axis=1, keepdims=True)

        # Flip normals pointing towards the centroid of the cross-section
        outward_vector = np.zeros_like(centre)
        outward_vector[:, 1] = centre[:, 1] - np.mean(triangle_coords[:, 1])
        outward_vector[:, 2] = centre[:, 2] - np.mean(triangle_coords[:, 2])

        flip_mask = np.sum(norm * outward_vector, axis=1) < 0
        norm[flip_mask] *= -1.0

        return area, centre, norm


@dataclass(kw_only=True)
class SimplifiedMonopoleRadiation(Radiation):
    r"""Simplified sound radiation of the rail using monopole sources.

    Each surface triangle of the rail mesh is treated as a monopole source with
    volume velocity :math:`Q = A \, v_n`, where :math:`v_n` is the normal
    surface velocity derived from :math:`u_y`, :math:`u_z` and
    :math:`\varphi_x`. The sound pressure at the measurement point is

    .. math::

        p(\omega) = j \omega \rho \sum_i \frac{A_i \, v_{n,i}(\omega) \, e^{-jkr_i}}{4 \pi r_i}

    An optional reflecting ground is included with the image source method.
    The calculation runs on initialization. Unweighted and A-weighted results
    are both stored; ``a_weighting`` selects which ones are used by
    :meth:`show` and :meth:`to_wav`.

    .. note:: This is a simplified approach that has not yet been validated
       against measurements or reference models (e.g. BEM). Near-field terms
       are neglected, i.e. the model assumes :math:`kr \gg 1` and receiver
       distances that are large compared to the rail cross-section. It requires
       the optional dependencies ``pip install rolland[radiation]``.

    Attributes
    ----------
    measurement_point : ndarray
        Coordinates :math:`[x, y, z]` of the receiver :math:`[m]`.
    speed_of_sound : float, default=343.0
        Speed of sound in air :math:`[m/s]`.
    air_density : float, default=1.2041
        Density of air :math:`[kg/m^3]`.
    spatial_window : float, default=5.0
        Half-width of the evaluated track section around ``x_centre`` :math:`[m]`.
    ground_distance : float, optional
        Distance between the lowest rail point and a reflecting ground
        :math:`[m]`. No ground reflection if None.
    reflection_coefficient : float, default=1.0
        Ground reflection coefficient (1.0 = acoustically rigid) :math:`[-]`.
    mesh_chunk_size : int, default=1000
        Number of triangles processed per chunk (memory management) :math:`[-]`.
    workers : int, default=1
        Number of worker threads :math:`[-]`.
    a_weighting : bool, default=True
        Use A-weighted results (IEC 61672-1) in :meth:`show` and :meth:`to_wav`.
    freq : ndarray
        Frequency array :math:`[Hz]`.
    sp_freq : ndarray
        One-sided complex sound pressure amplitude spectrum :math:`[Pa]`.
    sp_density : ndarray
        Complex sound pressure spectral density :math:`[Pa/Hz]`.
    sp_time : ndarray
        Sound pressure time signal at the measurement point :math:`[Pa]`.
    sp_freq_a : ndarray
        A-weighted one-sided complex sound pressure amplitude spectrum :math:`[Pa]`.
    sp_density_a : ndarray
        A-weighted complex sound pressure spectral density :math:`[Pa/Hz]`.
    sp_time_a : ndarray
        A-weighted sound pressure time signal :math:`[Pa]`.

    Example
    -------
    >>> from rolland import SimplifiedMonopoleRadiation
    >>> rad = SimplifiedMonopoleRadiation(
    ...     result=deflection_results,
    ...     rail=rail,
    ...     measurement_point=np.array([30.3, 7.5, 1.2]),
    ... )
    >>> rad.show()
    >>> rad.to_wav('sound_pressure.wav')
    """

    measurement_point: ndarray
    speed_of_sound: float = 343.0
    air_density: float = 1.2041
    spatial_window: float = 5.0
    ground_distance: float | None = None
    reflection_coefficient: float = 1.0
    mesh_chunk_size: int = 1000
    workers: int = 1
    a_weighting: bool = True

    freq: ndarray = field(init=False)
    sp_freq: ndarray = field(init=False)
    sp_density: ndarray = field(init=False)
    sp_time: ndarray = field(init=False)
    sp_freq_a: ndarray = field(init=False)
    sp_density_a: ndarray = field(init=False)
    sp_time_a: ndarray = field(init=False)

    def __post_init__(self):
        """Build the mesh and calculate the sound pressure."""
        super().__post_init__()
        self.measurement_point = np.asarray(self.measurement_point, dtype=float)
        self._calculate()

    def _calculate(self) -> None:
        """Calculate the sound pressure spectrum, spectral density and time signal."""
        u_y, u_z, phi_x, x_axis = self._preprocess_data()

        # Displacement -> velocity in the frequency domain
        v_y, self.freq = self._velocity_spectrum(u_y, self.dt)
        v_z, _ = self._velocity_spectrum(u_z, self.dt)
        v_phi, _ = self._velocity_spectrum(phi_x, self.dt)

        interp_kw = {'axis': 0, 'kind': 'linear', 'fill_value': 'extrapolate', 'assume_sorted': True}
        interp_vy = interp1d(x_axis, v_y, **interp_kw)
        interp_vz = interp1d(x_axis, v_z, **interp_kw)
        interp_vphi = interp1d(x_axis, v_phi, **interp_kw)

        # Mesh coordinates -> global track coordinates
        shifted_centre = self.base_centre.astype(np.float32, copy=True)
        shifted_centre[:, 0] += self._x_mid - self.mesh_length / 2

        z_ground = 0.0
        if self.ground_distance is not None:
            z_ground = np.min(self.triangle_coords[:, 2]) - self.ground_distance

        n_mesh = len(shifted_centre)
        mp = self.measurement_point.astype(np.float32)

        def process_chunk(chunk_start: int) -> ndarray:
            """Calculate the sound pressure contribution of one mesh chunk."""
            chunk = slice(chunk_start, min(chunk_start + self.mesh_chunk_size, n_mesh))
            centres = shifted_centre[chunk]
            areas = self.base_area[chunk].astype(np.float32)

            v_norm = self._map_velocity_to_mesh(
                centres,
                self.base_norm[chunk].astype(np.float32),
                interp_vy,
                interp_vz,
                interp_vphi,
            )

            p_chunk = self._calc_pressure_freq(v_norm, self._source_distance(centres, mp), areas)

            # Image source for the ground reflection
            if self.ground_distance is not None:
                centres_mirrored = centres.copy()
                centres_mirrored[:, 2] = 2.0 * z_ground - centres[:, 2]
                p_chunk += self.reflection_coefficient * self._calc_pressure_freq(
                    v_norm,
                    self._source_distance(centres_mirrored, mp),
                    areas,
                )

            return p_chunk

        self.sp_freq = np.zeros(len(self.freq), dtype=np.complex64)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            for p_chunk in executor.map(process_chunk, range(0, n_mesh, self.mesh_chunk_size)):
                self.sp_freq += p_chunk

        self.sp_density = self._spectral_density(self.sp_freq, self.dt)
        self.sp_time = np.fft.irfft(self._two_sided_scaling(self.sp_freq), norm='forward')

        # A-weighting (zero-phase filter in the frequency domain)
        self.sp_freq_a = (self.sp_freq * a_weighting(self.freq)).astype(np.complex64)
        self.sp_density_a = self._spectral_density(self.sp_freq_a, self.dt)
        self.sp_time_a = np.fft.irfft(self._two_sided_scaling(self.sp_freq_a), norm='forward')

    def show(self, quantity: str = 'density', ax=None, label=None, **kwargs):
        """Plot the sound pressure at the measurement point.

        Parameters
        ----------
        quantity : str, default='density'
            Quantity to plot: 'density' (spectral density :math:`[Pa/Hz]`),
            'spl' (one-sided amplitude in :math:`[dB]` re 20 µPa) or 'time'
            (time signal :math:`[Pa]`). A-weighted if ``a_weighting=True``.
        ax : matplotlib.axes.Axes, optional
            Axes to plot into. A new figure is created if None.
        label : str, optional
            Line label for the legend.
        **kwargs
            Additional keyword arguments passed to ``matplotlib``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            msg = 'matplotlib is required for plotting.'
            raise ImportError(msg) from e

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if self.a_weighting:
            sp_freq, sp_density, sp_time = self.sp_freq_a, self.sp_density_a, self.sp_time_a
            tag, db_unit = ' (A-weighted)', 'dB(A)'
        else:
            sp_freq, sp_density, sp_time = self.sp_freq, self.sp_density, self.sp_time
            tag, db_unit = '', 'dB'

        if quantity == 'density':
            ax.semilogy(self.freq[1:], np.abs(sp_density[1:]), label=label, **kwargs)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('|P(f)| [Pa/Hz]')
            ax.set_title(f'Sound Pressure Spectral Density{tag}')
        elif quantity == 'spl':
            spl = 20 * np.log10(np.maximum(np.abs(sp_freq[1:]) / np.sqrt(2), 1e-12) / P_REF)
            ax.plot(self.freq[1:], spl, label=label, **kwargs)
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(f'SPL [{db_unit} re 20 µPa]')
            ax.set_title(f'Sound Pressure Level Spectrum{tag}')
        elif quantity == 'time':
            ax.plot(np.arange(len(sp_time)) * self.dt, sp_time, label=label, **kwargs)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Sound pressure [Pa]')
            ax.set_title(f'Sound Pressure Time Signal{tag}')
        else:
            msg = "quantity must be one of 'density', 'spl' or 'time'."
            raise ValueError(msg)

        ax.grid(True, which='both', ls='-', alpha=0.5)
        if label:
            ax.legend()

        return ax

    def to_wav(
        self,
        filename: str = 'sound_pressure.wav',
        target_sr: int = 44100,
        apply_fade: bool = True,
        normalize: bool = False,
        full_scale_pa: float = 20.0,
        subtype: str = 'PCM_24',
    ) -> str:
        r"""Export the sound pressure time signal as WAV file.

        The WAV is physically calibrated: a sample value of 1.0 (0 dBFS)
        corresponds to ``full_scale_pa``. With the default of 20 Pa,
        0 dBFS equals 120 dB SPL, i.e. :math:`L_{SPL} = L_{dBFS} + 120\,dB`.
        If ``a_weighting=True``, the A-weighted signal is exported (levels in dB(A)).
        Note that the ear applies its own frequency weighting during playback;
        use ``a_weighting=False`` for a natural-sounding signal.

        Parameters
        ----------
        filename : str, default='sound_pressure.wav'
            Output file path.
        target_sr : int, default=44100
            Sample rate of the WAV file :math:`[Hz]`.
        apply_fade : bool, default=True
            Apply a short fade-in/out to avoid clicks at the signal boundaries.
        normalize : bool, default=False
            Scale the peak to full scale (removes the physical calibration).
        full_scale_pa : float, default=20.0
            Sound pressure mapped to digital full scale :math:`[Pa]`.
            Ignored if ``normalize=True``. Samples beyond are clipped.
        subtype : str, default='PCM_24'
            soundfile subtype, e.g. 'PCM_16', 'PCM_24' or 'FLOAT'.

        Returns
        -------
        str
            Path to the WAV file.
        """
        signal = (self.sp_time_a if self.a_weighting else self.sp_time).copy()

        orig_sr = 1.0 / self.dt
        if abs(orig_sr - target_sr) > 1.0:
            signal = librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)

        fade_samples = int(FADE_DURATION * target_sr)
        if apply_fade and 0 < 2 * fade_samples < len(signal):
            fade_curve = np.linspace(0, 1, fade_samples)
            signal[:fade_samples] *= fade_curve
            signal[-fade_samples:] *= fade_curve[::-1]

        if normalize:
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val * NORMALIZE_PEAK
        else:
            signal = signal / full_scale_pa
            if (peak := np.max(np.abs(signal))) > 1.0:
                warnings.warn(
                    f'Peak pressure {peak * full_scale_pa:.3g} Pa exceeds full scale ({full_scale_pa} Pa); '
                    'signal is clipped. Increase full_scale_pa.',
                    stacklevel=2,
                )
                signal = np.clip(signal, -1.0, 1.0)

        sf.write(filename, signal, target_sr, subtype=subtype)
        return filename

    def _preprocess_data(self) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        r"""Extract the deflections within the spatial window around ``x_centre``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Windowed :math:`u_y`, :math:`u_z`, :math:`\varphi_x` (shape ``(nt, nx)``)
            and the corresponding x-axis :math:`[m]`.
        """
        u_z_raw = np.asarray(self.result.u_z_obs)
        x_axis = np.arange(u_z_raw.shape[1]) * self.dx
        mask = (x_axis >= self._x_mid - self.spatial_window) & (x_axis <= self._x_mid + self.spatial_window)

        u_y, u_z, phi_x = (
            np.asarray(getattr(self.result, name))[:, mask].astype(np.float32)
            for name in ('u_y_obs', 'u_z_obs', 'phi_x_obs')
        )
        return u_y, u_z, phi_x, x_axis[mask]

    def _source_distance(self, centres: ndarray, mp: ndarray) -> ndarray:
        """Distance between source centres and measurement point :math:`[m]`."""
        dist = np.linalg.norm(centres - mp, axis=1)
        if np.any(dist < MIN_SOURCE_DISTANCE):
            msg = 'Measurement point is too close to the mesh surface.'
            raise ValueError(msg)
        return dist

    @staticmethod
    def _velocity_spectrum(displacement: ndarray, dt: float) -> tuple[ndarray, ndarray]:
        """Transform displacements to a one-sided velocity amplitude spectrum.

        Parameters
        ----------
        displacement : numpy.ndarray
            Displacement time series, shape ``(nt, nx)``.
        dt : float
            Time step :math:`[s]`.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Velocity spectrum (shape ``(nx, nf)``) and frequency array :math:`[Hz]`.
        """
        n_samples = displacement.shape[0]
        disp_freq = np.fft.rfft(displacement.T, norm='forward', axis=1).astype(np.complex64)
        freq = np.fft.rfftfreq(n_samples, dt)

        # One-sided amplitude spectrum (DC and Nyquist bin not doubled)
        if disp_freq.shape[1] > 1:
            disp_freq[:, 1:] *= 2.0
            if n_samples % 2 == 0:
                disp_freq[:, -1] /= 2.0

        omega = (2 * pi * freq).astype(np.float32)
        return (1j * omega[np.newaxis, :] * disp_freq).astype(np.complex64), freq

    @staticmethod
    def _map_velocity_to_mesh(
        centres: ndarray,
        normals: ndarray,
        interp_vy: interp1d,
        interp_vz: interp1d,
        interp_vphi: interp1d,
    ) -> ndarray:
        """Map the beam velocities to the normal velocity of the triangle centres.

        Parameters
        ----------
        centres : numpy.ndarray
            Triangle centres in global coordinates :math:`[m]`.
        normals : numpy.ndarray
            Outward unit normals of the triangles :math:`[-]`.
        interp_vy, interp_vz, interp_vphi : scipy.interpolate.interp1d
            Interpolators of the lateral, vertical and torsional velocity spectra along x.

        Returns
        -------
        numpy.ndarray
            Normal surface velocity spectra, shape ``(n_triangles, nf)``.
        """
        x_c, y_c, z_c = centres[:, 0], centres[:, 1], centres[:, 2]
        n_y, n_z = normals[:, 1], normals[:, 2]

        vy_c = interp_vy(x_c).astype(np.complex64)
        vz_c = interp_vz(x_c).astype(np.complex64)
        vphi_c = interp_vphi(x_c).astype(np.complex64)

        # Rigid cross-section: torsion adds a rotational velocity component
        vy_p = vy_c + vphi_c * z_c[:, np.newaxis]
        vz_p = vz_c - vphi_c * y_c[:, np.newaxis]

        return (vy_p * n_y[:, np.newaxis] + vz_p * n_z[:, np.newaxis]).astype(np.complex64, copy=False)

    def _calc_pressure_freq(self, velocity: ndarray, distances: ndarray, areas: ndarray) -> ndarray:
        """Sum the monopole contributions of a mesh chunk at the measurement point.

        Parameters
        ----------
        velocity : numpy.ndarray
            Normal surface velocity spectra, shape ``(n_triangles, nf)``.
        distances : numpy.ndarray
            Source-receiver distances :math:`[m]`.
        areas : numpy.ndarray
            Triangle areas :math:`[m^2]`.

        Returns
        -------
        numpy.ndarray
            Complex sound pressure spectrum :math:`[Pa]`.
        """
        omega = 2 * pi * self.freq
        k = omega / self.speed_of_sound
        dist, area = distances[:, np.newaxis], areas[:, np.newaxis]

        source_term = np.exp(-1j * k[np.newaxis, :] * dist) / (4 * pi * dist) * area * velocity
        return (np.sum(source_term, axis=0) * (1j * omega * self.air_density)).astype(np.complex64)

    @staticmethod
    def _two_sided_scaling(sp_freq: ndarray) -> ndarray:
        """Revert the one-sided doubling to the raw rfft convention (even signal length assumed)."""
        spectrum = np.asarray(sp_freq, dtype=np.complex128).copy()
        if spectrum.size > 2:
            spectrum[1:-1] /= 2.0
        return spectrum

    @classmethod
    def _spectral_density(cls, sp_freq: ndarray, dt: float) -> ndarray:
        r"""Convert the one-sided amplitude spectrum to the Fourier spectral density.

        For transient signals the meaningful quantity is the continuous Fourier
        transform :math:`P(f) = \int p(t) e^{-j 2 \pi f t} dt` :math:`[Pa/Hz]`,
        which is independent of the signal length.
        """
        n_samples = 2 * (sp_freq.size - 1)
        return cls._two_sided_scaling(sp_freq) * (n_samples * dt)
