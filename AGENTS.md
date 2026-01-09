# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/` with the crate root at `src/lib.rs`. Circuit analysis code is under `src/circuits/` (MNA stamping, solvers, SPICE, Touchstone). Field models and sources are under `src/fields/`. Shared pieces include `src/constants.rs` (CODATA 2022), `src/units.rs`, `src/materials.rs`, `src/math.rs`, `src/simulation.rs`, and `src/sweep.rs`. Examples live in `examples/` (`rlc_sweep`, `bevy_circuit`) and benchmarks in `benches/` (Criterion).

## Build, Test, and Development Commands
- `cargo build`: build with default features.
- `cargo build --features sparse`: enable nalgebra-sparse solvers.
- `cargo build --all-features`: build everything (includes serde).
- `cargo test`: run unit tests.
- `cargo test --features sparse`: run tests for sparse solvers.
- `cargo run --example rlc_sweep`: AC sweep example.
- `cargo run --example bevy_circuit`: Bevy visualization demo.
- `cargo bench`: run Criterion benchmarks.

## Coding Style & Naming Conventions
Rust 2024 edition; format with `rustfmt` and check core changes with `cargo clippy`. Use Rust naming (`snake_case`, `UpperCamelCase`, `SCREAMING_SNAKE_CASE`). Prefer `f64` and `num_complex::Complex<f64>` for numeric work. Circuit nodes are `Option<usize>` where `None` is ground (0-based indices). When adding public APIs, update `src/prelude.rs` exports if they should be downstream-friendly.

## Testing Guidelines
Unit tests live next to implementation using `#[cfg(test)]`. For floating-point assertions, use `approx::assert_relative_eq!`. Tests typically validate against analytic solutions or known reference values; add those references in comments when introducing new behavior.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commits (`feat:`, `fix:`); use that style. PRs should include a brief summary, test commands run, and any required feature flags. The API is still stabilizing, so coordinate larger refactors with maintainers.

## Agent-Specific Notes
Start by scanning `README.md` and `CLAUDE.md` for current workflows and module overviews. Feature flags: `std` (default), `serde`, and `sparse`; keep optional code behind `cfg(feature = "sparse")` in `src/circuits/`. MNA stamping lives in `src/circuits/stamp.rs`, solver traits in `src/circuits/solver.rs`, Krylov solvers in `src/circuits/iterative.rs`, and matrix ordering in `src/circuits/ordering.rs`. Touchstone parsing is in `src/circuits/nport.rs`, SPICE parsing in `src/circuits/spice.rs`. Bevy is a dev dependency; the visualization example may need graphics support.
