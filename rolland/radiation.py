"""Monopole radiation methods for sound pressure calculation in the frequency domain."""
import concurrent.futures
import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import gmsh
import librosa
import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d


@dataclass(kw_only=True)
class Radiation(ABC):
    r"""Abstract base class for sound radiation models handling GMSH mesh generation.

    This class encapsulates the rail contour interpolation, 3D mesh extrusion,
    and calculation of mesh properties (centres, areas, normals).
    """

    # Mesh properties (internal, excluded from initialization)
    triangle_coords: np.ndarray | None = field(default=None, init=False)
    triangle_index: np.ndarray | None = field(default=None, init=False)
    base_area: np.ndarray | None = field(default=None, init=False)
    base_centre: np.ndarray | None = field(default=None, init=False)
    base_norm: np.ndarray | None = field(default=None, init=False)
    mesh_length: float = field(default=0.0, init=False)

    # Store mesh creation parameters for visualization
    _rail_contour_2d: np.ndarray | None = field(default=None, init=False)
    _mesh_size: float = field(default=0.05, init=False)
    _num_contour_points: int = field(default=30, init=False)

    @abstractmethod
    def calculate_sound_pressure(self, *args: Any, **kwargs: Any) -> Any:
        """Calculate sound pressure. Must be implemented by subclasses."""

    def setup_mesh(
        self,
        rail_contour_2d: np.ndarray,
        mesh_length: float = 10.0,
        mesh_size: float = 0.05,
        num_pts: int = 30,
    ) -> None:
        """Interpolate the rail contour, create the 3D mesh, and calculate properties.

        Parameters
        ----------
        rail_contour_2d : np.ndarray
            2D contour points of the rail cross-section.
        mesh_length : float, default=10.0
            Length of the rail mesh to extrude [m].
        mesh_size : float, default=0.05
            Target element size for the mesh [m].
        num_pts : int, default=30
            Number of points for contour interpolation.
        """
        self.mesh_length = mesh_length
        self._rail_contour_2d = np.array(rail_contour_2d)
        self._mesh_size = mesh_size
        self._num_contour_points = num_pts

        contour = self._interpolate_contour_2d(self._rail_contour_2d, num_pts)
        self.triangle_coords, self.triangle_index = self._create_rail_mesh(contour, mesh_size, mesh_length)

        self.base_area, self.base_centre, self.base_norm = self._calculate_centre_and_area_triangles(
            self.triangle_coords,
            self.triangle_index,
        )

    def view_mesh(self, measurement_point: np.ndarray | None = None, x_exc: float = 0.0) -> None:
        """Open the mesh in GMSH GUI for visualization.

        Parameters
        ----------
        measurement_point : np.ndarray | None, optional
            Coordinate of measurement point [x, y, z] to display.
        x_exc : float, default=0.0
            Global x-coordinate of the excitation point.
        """
        if self._rail_contour_2d is None:
            msg = 'Mesh has not been created yet. Please call setup_mesh() first.'
            raise RuntimeError(msg)

        if not gmsh.isInitialized():
            gmsh.initialize()

        try:
            gmsh.model.add('rail_mesh')

            contour = self._interpolate_contour_2d(self._rail_contour_2d, self._num_contour_points)
            rail_geometry_3d = np.c_[np.full(contour.shape[0], x_exc - (self.mesh_length / 2.0)), contour]

            point_tags = [gmsh.model.geo.addPoint(p[0], p[1], p[2], self._mesh_size) for p in rail_geometry_3d]
            n_pts = len(point_tags)
            line_tags = [gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n_pts]) for i in range(n_pts)]

            curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
            surface = gmsh.model.geo.addPlaneSurface([curve_loop])
            gmsh.model.geo.synchronize()

            gmsh.model.geo.extrude(dimTags=[(2, surface)], dx=self.mesh_length, dy=0, dz=0)
            gmsh.model.geo.synchronize()

            for opt, val in [('Mesh.Algorithm', 6), ('Mesh.Optimize', 1), ('Mesh.OptimizeNetgen', 1)]:
                gmsh.option.setNumber(opt, val)

            gmsh.model.mesh.generate(2)
            gmsh.model.geo.synchronize()

            if measurement_point is not None:
                mp = measurement_point.astype(float)
                view_tag = gmsh.view.add('measurement_point_view')
                v_idx = gmsh.view.getIndex(view_tag)

                opts = {
                    'PointSize': 20,
                    'ShowScale': 0,
                    'RangeType': 2,
                    'CustomMin': 0.0,
                    'CustomMax': 1.0,
                    'ColormapNumber': 1,
                }
                for key, val in opts.items():
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
    def _interpolate_contour_2d(array: np.ndarray, n: int) -> np.ndarray:
        """Interpolate a 2D contour to a specified number of points."""
        array = np.atleast_2d(array)
        if array.shape[0] < 2 or array.shape[1] != 2 or n < 2:
            msg = 'Invalid contour shape or point count.'
            raise ValueError(msg)

        contour_s = np.zeros(array.shape[0], dtype=float)
        contour_s[1:] = np.cumsum(np.linalg.norm(np.diff(array, axis=0), axis=1))

        new_contour_s = np.linspace(0.0, contour_s[-1], n)
        x_new = np.interp(new_contour_s, contour_s, array[:, 0])
        y_new = np.interp(new_contour_s, contour_s, array[:, 1])

        return np.c_[x_new, y_new]

    @staticmethod
    def _create_rail_mesh(
        rail_contour: np.ndarray,
        mesh_size: float | None = None,
        length: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a 3D surface mesh using GMSH based on the 2D rail contour."""
        rail_geometry_3d = np.c_[np.zeros(rail_contour.shape[0]), rail_contour]

        if not gmsh.isInitialized():
            gmsh.initialize()

        try:
            gmsh.model.add('rail')

            if mesh_size is None:
                mesh_size = float(np.linalg.norm(rail_geometry_3d[0] - rail_geometry_3d[1]))

            point_tags = [gmsh.model.geo.addPoint(p[0], p[1], p[2], mesh_size) for p in rail_geometry_3d]
            n_pts = len(point_tags)
            line_tags = [gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n_pts]) for i in range(n_pts)]

            curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
            surface = gmsh.model.geo.addPlaneSurface([curve_loop])
            gmsh.model.geo.synchronize()

            gmsh.model.geo.extrude(dimTags=[(2, surface)], dx=length, dy=0, dz=0)
            gmsh.model.geo.synchronize()

            for opt, val in [('Mesh.Algorithm', 6), ('Mesh.Optimize', 1), ('Mesh.OptimizeNetgen', 1)]:
                gmsh.option.setNumber(opt, val)

            gmsh.model.mesh.generate(2)

            _, _, element_node_tags = gmsh.model.mesh.getElements(2)
            triangle_tags = element_node_tags[0].reshape(-1, 3)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            triangle_coords = node_coords.reshape(-1, 3)

            tag_to_index = {tag: i for i, tag in enumerate(node_tags)}
            triangle_index = np.vectorize(tag_to_index.get)(triangle_tags)

            return triangle_coords, triangle_index

        finally:
            if gmsh.isInitialized():
                gmsh.finalize()

    @staticmethod
    def _calculate_centre_and_area_triangles(
        triangle_coords: np.ndarray,
        triangle_index: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the centre, area, and normal vector for each triangle.

        Ensures normal vectors point outwards.
        """
        idx0, idx1, idx2 = triangle_index[:, 0], triangle_index[:, 1], triangle_index[:, 2]
        p1, p2, p3 = triangle_coords[idx0], triangle_coords[idx1], triangle_coords[idx2]
        cross_product = np.cross(p1 - p2, p1 - p3)

        area = np.linalg.norm(cross_product, axis=1) * 0.5
        centre = (p1 + p2 + p3) / 3.0
        norm = cross_product / np.linalg.norm(cross_product, axis=1, keepdims=True)

        outward_vector = np.zeros_like(centre)
        outward_vector[:, 1] = centre[:, 1] - np.mean(triangle_coords[:, 1])
        outward_vector[:, 2] = centre[:, 2] - np.mean(triangle_coords[:, 2])

        flip_mask = np.sum(norm * outward_vector, axis=1) < 0
        norm[flip_mask] *= -1.0

        return area, centre, norm


@dataclass(kw_only=True)
class SimplifiedMonopoleRadiation(Radiation):
    r"""Simplified calculation of sound radiation from a rail model using monopole sources.

    Attributes
    ----------
    speed_of_sound : float, default=343.0
        Speed of sound in the medium [m/s].
    air_density : float, default=1.2041
        Air density [kg/m^3].
    """

    speed_of_sound: float = 343.0
    air_density: float = 1.2041

    def _prepare_raw_inputs(
        self,
        u_y: np.ndarray | None,
        u_z: np.ndarray | None,
        phi_x: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and provide defaults for raw displacement inputs."""
        provided_arrays = [arr for arr in (u_y, u_z, phi_x) if arr is not None]
        if not provided_arrays:
            msg = 'At least one of u_y_raw, u_z_raw, or phi_x_raw must be provided.'
            raise ValueError(msg)

        ref_array = provided_arrays[0]
        target_shape = ref_array.shape
        target_dtype = ref_array.dtype

        if u_y is None:
            u_y = np.zeros(target_shape, dtype=target_dtype)
        if u_z is None:
            u_z = np.zeros(target_shape, dtype=target_dtype)
        if phi_x is None:
            phi_x = np.zeros(target_shape, dtype=target_dtype)

        return u_y, u_z, phi_x

    def calculate_sound_pressure(
        self,
        u_y_raw: np.ndarray | None = None,
        u_z_raw: np.ndarray | None = None,
        phi_x_raw: np.ndarray | None = None,
        *,
        dt: float,
        dx: float,
        x_exc: float,
        measurement_point: np.ndarray,
        downsample_factor: int = 20,
        spatial_window: float = 5.0,
        mesh_chunk_size: int = 1000,
        ground_distance: float | None = None,
        reflection_coefficient: float = 1.0,
        workers: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform data preprocessing, FFT, and complex sound pressure calculation.

        Includes an optional reflective ground using the Image Source Method.

        Parameters
        ----------
        u_y_raw : np.ndarray | None, optional
            Raw displacement data in y-direction.
        u_z_raw : np.ndarray | None, optional
            Raw displacement data in z-direction.
        phi_x_raw : np.ndarray | None, optional
            Raw rotational displacement data around x-axis.
        dt : float
            Time step [s].
        dx : float
            Spatial step [m].
        x_exc : float
            Excitation position [m].
        measurement_point : np.ndarray
            Measurement point coordinates [x, y, z].
        spatial_window : float, default=5.0
            Spatial window size [m].
        mesh_chunk_size : int, default=1000
            Mesh chunk size for memory management.
        ground_distance : float | None, optional
            Distance from lowest rail point to ground [m]. If None, no reflection.
        reflection_coefficient : float, default=1.0
            Acoustic reflection coefficient of the ground (1.0 = perfectly rigid).
        workers : int, default=1
            Number of worker processes for parallel computation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Frequency axis and complex sound pressure in frequency domain.
        """
        u_y_raw, u_z_raw, phi_x_raw = self._prepare_raw_inputs(u_y_raw, u_z_raw, phi_x_raw)

        if any(v is None for v in (self.base_centre, self.base_area, self.base_norm)):
            msg = 'Mesh has not been created yet. Please call setup_mesh() first.'
            raise ValueError(msg)

        u_y_filt, u_z_filt, phi_x_filt, x_axis_filt = self._preprocess_data(
            u_y_raw,
            u_z_raw,
            phi_x_raw,
            dx,
            x_exc,
            spatial_window,
        )

        del u_y_raw, u_z_raw, phi_x_raw
        gc.collect()

        u_y_freq, freq_axis = self._transform_to_frequency_domain(u_y_filt.T, dt)
        u_z_freq, _ = self._transform_to_frequency_domain(u_z_filt.T, dt)
        phi_x_freq, _ = self._transform_to_frequency_domain(phi_x_filt.T, dt)

        del u_y_filt, u_z_filt, phi_x_filt
        gc.collect()

        v_y_freq = self._calculate_velocity_frequency_domain(u_y_freq, freq_axis)
        v_z_freq = self._calculate_velocity_frequency_domain(u_z_freq, freq_axis)
        v_phi_freq = self._calculate_velocity_frequency_domain(phi_x_freq, freq_axis)

        del u_y_freq, u_z_freq, phi_x_freq
        gc.collect()

        interp_kw = {'axis': 0, 'kind': 'linear', 'fill_value': 'extrapolate', 'assume_sorted': True}
        interp_vy = interp1d(x_axis_filt, v_y_freq, **interp_kw)
        interp_vz = interp1d(x_axis_filt, v_z_freq, **interp_kw)
        interp_vphi = interp1d(x_axis_filt, v_phi_freq, **interp_kw)

        shifted_centre = self.base_centre.astype(np.float32, copy=True)
        shifted_centre[:, 0] += x_exc - (self.mesh_length / 2.0)

        z_ground = 0.0
        if ground_distance is not None:
            z_ground = np.min(self.triangle_coords[:, 2]) - ground_distance

        n_mesh = len(shifted_centre)
        sound_pressure_freq = np.zeros(len(freq_axis), dtype=np.complex64)
        c, rho = self.speed_of_sound, self.air_density
        mp_f32 = measurement_point.astype(np.float32)

        def process_chunk(chunk_start: int) -> np.ndarray:
            """Process a single mesh chunk and return the resulting sound pressure array."""
            chunk_end = min(chunk_start + mesh_chunk_size, n_mesh)

            centres_chunk = shifted_centre[chunk_start:chunk_end]
            normals_chunk = self.base_norm[chunk_start:chunk_end].astype(np.float32)
            areas_chunk = self.base_area[chunk_start:chunk_end].astype(np.float32)

            v_norm = self._map_velocity_to_mesh_chunk(
                centres_chunk,
                normals_chunk,
                interp_vy,
                interp_vz,
                interp_vphi,
            )
            dist_chunk = np.linalg.norm(centres_chunk - mp_f32, axis=1)

            if np.any(dist_chunk < 1e-6):
                msg = 'Measurement point is too close to the direct mesh surface!'
                raise ValueError(msg)

            p_chunk = self._calc_pressure_freq(v_norm, freq_axis, dist_chunk, areas_chunk, c, rho)

            if ground_distance is not None:
                centres_mirrored = centres_chunk.copy()
                centres_mirrored[:, 2] = 2.0 * z_ground - centres_chunk[:, 2]
                dist_mirrored = np.linalg.norm(centres_mirrored - mp_f32, axis=1)

                if np.any(dist_mirrored < 1e-6):
                    msg = 'Measurement point is too close to the mirrored mesh surface!'
                    raise ValueError(msg)

                p_chunk_mirrored = self._calc_pressure_freq(v_norm, freq_axis, dist_mirrored, areas_chunk, c, rho)
                p_chunk += reflection_coefficient * p_chunk_mirrored

            return p_chunk

        chunk_starts = range(0, n_mesh, mesh_chunk_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for p_chunk in executor.map(process_chunk, chunk_starts):
                sound_pressure_freq += p_chunk

        return freq_axis, sound_pressure_freq

    def generate_wav_file(
        self,
        sp_freq: np.ndarray,
        dt: float,
        filename: str = 'sound_pressure.wav',
        target_sr: int = 44100,
    ) -> str:
        """Generate WAV file from freq-domain sound pressure using librosa and soundfile.

        Parameters
        ----------
        sp_freq : np.ndarray
            Complex sound pressure from rfft.
        dt : float
            Effective time step [s].
        filename : str, default='sound_pressure.wav'
            Output filename.
        target_sr : int, default=44100
            Target sample rate for WAV file [Hz].

        Returns
        -------
        str
            Path to WAV file.
        """
        orig_sr = 1.0 / dt
        sp_time = np.fft.irfft(sp_freq)

        if abs(orig_sr - target_sr) > 1.0:
            sp_time = librosa.resample(sp_time, orig_sr=orig_sr, target_sr=target_sr)

        if (fade_samples := int(0.005 * target_sr)) > 0 and 2 * fade_samples < len(sp_time):
            fade_curve = np.linspace(0, 1, fade_samples)
            sp_time[:fade_samples] *= fade_curve
            sp_time[-fade_samples:] *= fade_curve[::-1]

        max_val = np.max(np.abs(sp_time))
        if max_val > 0:
            sp_time = (sp_time / max_val) * 0.944

        sf.write(filename, sp_time, target_sr, subtype='PCM_16')
        return filename

    @staticmethod
    def _preprocess_data(
        u_y_raw: np.ndarray,
        u_z_raw: np.ndarray,
        phi_x_raw: np.ndarray,
        dx: float,
        x_exc: float,
        spatial_window: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """Apply a spatial window filter without additional time downsampling."""
        x_axis = np.arange(u_y_raw.shape[1]) * dx
        spatial_mask = (x_axis >= x_exc - spatial_window) & (x_axis <= x_exc + spatial_window)

        u_y = u_y_raw[:, spatial_mask].astype(np.float32, copy=False)
        u_z = u_z_raw[:, spatial_mask].astype(np.float32, copy=False)
        phi_x = phi_x_raw[:, spatial_mask].astype(np.float32, copy=False)

        return u_y, u_z, phi_x, x_axis[spatial_mask]

    @staticmethod
    def _map_velocity_to_mesh_chunk(
        centres: np.ndarray,
        normals: np.ndarray,
        interp_vy: interp1d,
        interp_vz: interp1d,
        interp_vphi: interp1d,
    ) -> np.ndarray:
        """Map generalized centroid velocities to a chunk of 3D triangle centres."""
        x_c, y_c, z_c = centres[:, 0], centres[:, 1], centres[:, 2]
        n_y, n_z = normals[:, 1], normals[:, 2]

        vy_c = interp_vy(x_c).astype(np.complex64)
        vz_c = interp_vz(x_c).astype(np.complex64)
        vphi_c = interp_vphi(x_c).astype(np.complex64)

        vy_p = vy_c + vphi_c * z_c[:, np.newaxis]
        vz_p = vz_c - vphi_c * y_c[:, np.newaxis]

        return (vy_p * n_y[:, np.newaxis] + vz_p * n_z[:, np.newaxis]).astype(np.complex64, copy=False)

    @staticmethod
    def _transform_to_frequency_domain(displacement_time: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Transform displacement from the time domain to the frequency domain (RFFT)."""
        n_samples = displacement_time.shape[1]
        disp_freq = np.fft.rfft(displacement_time, norm='forward', axis=1).astype(np.complex64)
        freq_axis = np.fft.rfftfreq(n_samples, dt)

        if disp_freq.shape[1] > 1:
            disp_freq[:, 1:] *= 2.0
            if n_samples % 2 == 0:
                disp_freq[:, -1] /= 2.0

        return disp_freq, freq_axis

    @staticmethod
    def _calculate_velocity_frequency_domain(displacement_freq: np.ndarray, frequency_axis: np.ndarray) -> np.ndarray:
        """Compute velocity from displacement in the frequency domain."""
        omega = (2 * np.pi * frequency_axis).astype(np.float32)
        return (1j * omega[np.newaxis, :] * displacement_freq).astype(np.complex64)

    @staticmethod
    def _calc_pressure_freq(
        velocity_freq_chunk: np.ndarray,
        frequency_axis: np.ndarray,
        distances_chunk: np.ndarray,
        areas_chunk: np.ndarray,
        speed_of_sound: float,
        air_density: float,
    ) -> np.ndarray:
        """Calculate sound pressure from monopole sources."""
        dist_2d, areas_2d = distances_chunk[:, np.newaxis], areas_chunk[:, np.newaxis]
        k, omega = 2 * np.pi * frequency_axis / speed_of_sound, 2 * np.pi * frequency_axis

        phase_term = np.exp(-1j * k[np.newaxis, :] * dist_2d)
        amp_term = 1.0 / (4 * np.pi * dist_2d)
        source_term = phase_term * amp_term * areas_2d * velocity_freq_chunk

        return (np.sum(source_term, axis=0) * (1j * omega * air_density)).astype(np.complex64)
