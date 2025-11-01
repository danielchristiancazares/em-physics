# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2025-10-31
### Added
- **Sparse MNA solver** (`sparse` feature): High-performance sparse Modified Nodal Analysis with automatic symbolic factorization reuse for AC sweeps, reducing computation time by 2-10× for frequency sweeps.
- **Transient circuit simulation**: New `MnaTransientEngine` with multiple implicit integrators (trapezoidal, BDF1, BDF2) for time-domain analysis of dynamic circuits.
- **Field computation modules**:
  - **Boundary Element Method (BEM)**: Single/double layer potential operators for electrostatic problems with Dirichlet boundary conditions.
  - **Electrostatics**: Point charge systems, uniform charge patches, and analytical field computations.
  - **Retarded potentials**: Time-domain electromagnetic field calculations for moving charges and time-varying currents.
- **AC sweep optimization**: `ac_sweep_sparse_mna()` function with symbolic factorization reuse, dramatically improving performance for multi-frequency circuit analysis.
- `SolveReport` diagnostics for MNA, including condition estimates and floating-node warnings, surfaced through `MnaBuilder::solve_with_report` and AC sweep results.
- Closed-form Biot–Savart evaluator for finite wire segments with automatic fallback to numerical quadrature.
- `Switch` component with on/off resistances, and interactive Bevy example now driven by switch states.
- Complex material response support via `MaterialResponse` and `MaterialResponseProvider` for frequency-dependent ε(ω)/μ(ω).
- CSV helpers for `FrequencyPoint` and AC MNA sweeps to streamline plotting and regression capture.
- Transient waveform capture and CSV export functions for time-domain results.

### Removed
- Deprecated `CLAUDE.md` contributor instructions file.

## [0.1.1] - 2025-10-31
### Changed
- Refined `VACUUM_PERMITTIVITY`, `VACUUM_PERMEABILITY`, and `FREE_SPACE_IMPEDANCE` to CODATA 2022 values with higher precision.
- Tightened vacuum impedance regression test tolerance to reflect CODATA 2022 data.
- Updated documentation to reference CODATA 2022 constants and list free-space impedance.

## [0.1.0] - 2025-10-30
### Added
- Initial release with dense Modified Nodal Analysis (MNA) circuit solver.
- AC frequency-domain analysis with complex frequency support.
- Passive component library (resistors, capacitors, inductors, voltage/current sources).
- Controlled sources (VCCS, VCVS, CCCS, CCVS) for dependent elements.
- Material properties and dispersive media support.
- Magnetic field computation using Biot-Savart law.
- Bevy-based interactive visualization examples.
- CSV export utilities for analysis results.
