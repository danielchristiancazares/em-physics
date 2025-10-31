# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`em-physics` is a research-grade Rust library for modeling electromagnetism and circuit behavior. The project is in early scaffolding phase with rapid API evolution expected. Focus is on robust architecture, strong typing, and rigorous validation against reference data.

## Development Commands

### Build and Check
```sh
cargo build              # Standard build
cargo check              # Fast compile check
cargo clippy             # Lint check (strict: all, cargo, nursery warnings)
cargo fmt                # Format code
cargo test               # Run test suite
cargo test <test_name>   # Run single test (matches substring)
cargo doc --open         # Build and open documentation
```

### Examples and Benchmarks
```sh
cargo run --example rlc_sweep          # Run series RLC AC sweep example
cargo run --example <name>             # Run specific example
cargo bench                            # Run criterion benchmarks
cargo bench --bench network_impedance  # Run specific benchmark
```

### Feature Flags
```sh
cargo build --no-default-features  # Build without std
cargo build --features serde       # Enable serde serialization support
cargo build --features sparse      # Enable nalgebra-sparse support
cargo build --all-features         # Build with all features
```

## Architecture

### Module Structure
- **constants**: Fundamental physical constants (e.g., angular_frequency conversion)
- **units**: Strongly typed unit helpers and quantity abstractions
- **math**: Shared mathematical utilities (vectors, matrices, transforms via nalgebra)
- **fields**: EM field representations (electric, magnetic modules)
- **materials**: Material property models (permittivity, permeability, conductivity, dispersion)
- **circuits**: Circuit primitives and solvers spanning lumped and distributed systems
  - `component`: Lumped component definitions (Resistor, Inductor, Capacitor, VoltageSource) with Component trait
  - `network`: Network composition with ConnectionKind (Series/Parallel)
  - `twoport`: ABCD-based two-port network representations with cascading and parameter conversion (Z/Y/S)
  - `transmission`: Transmission line elements with RLGC parameterization
  - `stamp`: Nodal stamping builders (NodalBuilder for DC/AC, MNA for modified nodal analysis)
  - `analysis`: Frequency-domain solvers (sweep_network_impedance, NodalAnalysis, AdmittanceMatrix)
  - `sparse`: Optional sparse matrix helpers (requires `sparse` feature)
- **sweep**: Frequency sweep builders (linear/log) and post-processing helpers (magnitude/phase)
- **simulation**: High-level experiment orchestrators
- **errors**: Shared error types (uses thiserror)
- **prelude**: Common exports for downstream usage

### Key Design Patterns
- **Strong Typing**: Use units module for typed quantities; avoid raw floats where possible
- **Frequency Domain**: Primary focus is AC analysis; angular frequency (ω) is standard
- **Complex Numbers**: Use num_complex::Complex64 for impedance calculations
- **Component Trait**: All circuit components implement Component trait with impedance methods
- **Network Composition**: Networks aggregate components with ConnectionKind for topology
- **Two-Port Networks**: ABCD matrix representation enables cascading of transmission line segments and conversions between parameter sets (Z/Y/S)
- **Nodal Stamping**: Element-by-element matrix assembly via NodalBuilder; supports DC/AC analysis with reactive elements

### Dependencies
- `nalgebra`: Linear algebra (matrices for nodal analysis)
- `nalgebra-sparse`: Optional sparse matrix support (requires `sparse` feature)
- `num-complex`: Complex number arithmetic for AC circuit analysis
- `num-traits`: Numeric trait abstractions
- `thiserror`: Error type derivation
- `approx`: Floating-point comparison in tests
- `criterion`: Benchmark framework (dev dependency)

## Code Quality Requirements

### Strict Linting
The crate enforces `#![warn(clippy::all, clippy::cargo, clippy::nursery, missing_docs)]`. All code must:
- Pass clippy with these lint groups enabled
- Include doc comments for all public items (missing_docs enforced)
- Address warnings; suppressions require justification

### Testing and Validation
- Validation against reference data is critical for research-grade accuracy
- Use approx crate for floating-point assertions with appropriate epsilon
- Examples should be runnable and demonstrate realistic usage patterns

## Rust Edition

Uses **edition = "2024"** (very recent). Ensure compatibility with latest Rust edition features and idioms.
