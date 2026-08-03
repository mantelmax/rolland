<picture>
  <source srcset="docs/source/images/logo_rolland_light.svg" media="(prefers-color-scheme: light)">
  <img src="docs/source/images/logo_rolland_dark.svg" alt="Logo" width="160">
</picture>

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mantelmax/rolland)
[![Documentation Status](https://readthedocs.org/projects/rolland-rolling-noise-and-dynamics/badge/?version=latest)](https://rolland-rolling-noise-and-dynamics.readthedocs.io/en/latest/?badge=latest)
[![Python Version](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)

# Rolland

Rolling Noise and Dynamics (**Rolland**) is an open-source, high-performance time-domain simulation framework designed to analyze, predict, and optimize the dynamic and acoustic properties of railway tracks.

By employing an explicit Finite Difference Method (FDM) scheme, **Rolland** solves 27 coupled differential equations to capture full 3D track dynamics—including coupled vertical bending, lateral bending, axial torsion, cross-sectional warping, and sleeper movement under eccentric excitation and support conditions. The framework incorporates Complex Frequency-Shifted Perfectly Matched Layers (CFS-PML) for infinite track modeling and supports spatially varying track properties as well as moving excitation sources.

# Key Features

- **Multi-DOF Coupled Track Wave Modeling:**
  - Full multi-degree-of-freedom formulation solving **27 coupled differential equations**, accounting for 3D rail dynamics (longitudinal, vertical, lateral, torsional, warping) together with sleeper and support movement.
  - Explicit inclusion of mechanical eccentricities: rail head contact excitation eccentricity ($y_{\text{e}}, z_{\text{e}}$), rail foot support eccentricity ($z_{\text{f}}$), and cross-sectional shear center/centroid offsets.
- **High-Performance Time-Domain Solver:**
  - Automated symbolic finite-difference operator compilation.
  - High-order numerical schemes (6th-order central spatial stencils for primary fields, 2nd-order for PML auxiliary fields).
  - Highly computationally efficient: a 0.5 s time-domain simulation completes in ~7 seconds on standard desktop hardware (16 threads).
- **Infinite Track Boundary Modeling (CFS-PML):**
  - Time-domain Auxiliary Differential Equation (ADE) formulation of Complex Frequency-Shifted Perfectly Matched Layers.
  - Eliminates artificial boundary wave reflections across a broad frequency spectrum ($50\text{ Hz} - 6000\text{ Hz}$).
- **Damping Formulations:**
  - Hysteretic-to-viscous damping conversion tuned to wave cut-on frequencies ($\Omega_{\text{d}}$) for explicit time-domain integration.
- **Flexible Track Architectures:**
  - Continuous and discrete single-rail models for both ballasted and slab tracks.
  - Arbitrary spatial track property variations (periodic or stochastic, e.g., sleeper spacing, pad/ballast stiffness).
- **Excitation & Post-Processing:**
  - Stationary (Gaussian impulse) and moving excitation modes (moving random force for wheel-rail interaction).
  - Point and transfer mobility analysis, coupled cross-mobility calculations, and Track Decay Rate (TDR) in dB/m.
- **Reference Models:**
  - Built-in analytical, semi-analytical and numerical reference models for instant validation.

**Planned Features:**

- Non-linear Hertzian contact dynamics for wheel-rail interaction.
- Multi-wheel vehicle pass-by excitation models.
- Rail acoustic radiation modeling.

<picture>
  <source srcset="docs/source/images/mwi_github_dark.png" media="(prefers-color-scheme: dark)">
  <img src="docs/source/images/mwi_light.png">
</picture>

# Installation

To install Rolland, you can use pip. It is recommended to create a virtual environment first:

```bash
pip install git+https://github.com/mantelmax/rolland.git
```

# Documentation

Documentation is available [here](https://rolland-rolling-noise-and-dynamics.readthedocs.io) with how-to guides and detailed API reference.

# Contributing

Contributions to **Rolland** are welcome! Whether you are reporting bugs, improving documentation, or submitting pull requests for new features:

1. Fork the repository on GitHub.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Ensure code formatting meets requirements using `ruff check` and tests pass with `pytest`.
4. Submit a Pull Request with a clear summary of your changes.

# License

This project is licensed under the **BSD 3-Clause License**. See the `pyproject.toml` file for licensing metadata.

# Citation

If you use **Rolland** in your research, please cite the following paper:

```bibtex
@inproceedings{mantel2026rolland,
  title     = {Rolland: A New Framework for Realistic and Computationally Efficient Rolling Noise Modeling in the Time Domain},
  author    = {Mantel, Maximilian and Sarradj, Ennes},
  booktitle = {Proceedings of Forum Acusticum 2026},
  year      = {2026}
}
```
