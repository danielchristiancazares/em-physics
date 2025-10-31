# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2025-10-31
### Changed
- Refined `VACUUM_PERMITTIVITY`, `VACUUM_PERMEABILITY`, and `FREE_SPACE_IMPEDANCE`
  to CODATA 2022 values with higher precision.
- Tightened vacuum impedance regression test tolerance to reflect CODATA 2022 data.
- Updated documentation to reference CODATA 2022 constants and list free-space impedance.

## [0.2.0] - 2025-10-30
### Added
- `SolveReport` diagnostics for MNA, including condition estimates and floating-node warnings, surfaced through `MnaBuilder::solve_with_report` and AC sweep results.
- Closed-form Biot–Savart evaluator for finite wire segments with automatic fallback to numerical quadrature.
- `Switch` component with on/off resistances, and interactive Bevy example now driven by switch states.
- Complex material response support via `MaterialResponse` and `MaterialResponseProvider` for frequency-dependent ε(ω)/μ(ω).
- CSV helpers for `FrequencyPoint` and AC MNA sweeps to streamline plotting and regression capture.

### Removed
- Deprecated `CLAUDE.md` contributor instructions file.
