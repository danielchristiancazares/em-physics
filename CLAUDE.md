# em-physics

Research-grade Rust library for electromagnetism and circuit analysis.

## Build and Test

```bash
# Build (default features)
cargo build

# Build with sparse matrix support (nalgebra-sparse)
cargo build --features sparse

# Build all features
cargo build --all-features

# Run tests
cargo test

# Run tests with sparse solvers
cargo test --features sparse

# Run specific example
cargo run --example rlc_sweep
cargo run --example bevy_circuit

# Run benchmarks
cargo bench
```

## Project Structure

```
src/
  lib.rs          # Crate root, module declarations
  constants.rs    # CODATA 2022 physical constants (c, e, k_B, epsilon_0, mu_0)
  units.rs        # Strongly-typed unit wrappers (Volt, Ampere, Impedance, etc.)
  math.rs         # Scalar alias (f64), R3 vectors, phasor utilities
  materials.rs    # MaterialProperties, DrudeModel dispersive material
  errors.rs       # Shared error types
  prelude.rs      # Common re-exports for downstream use
  sweep.rs        # Frequency sweep builders (linspace, logspace, mag_db, phase_deg)
  simulation.rs   # MnaTransientEngine, time integrators (Trapezoidal, BDF1, BDF2)

  fields/
    mod.rs          # Field module exports
    electric.rs     # ElectricField, ElectricFieldKind enums
    magnetic.rs     # MagneticField descriptor
    sources.rs      # Biot-Savart: LineCurrent, WireSegment3D, vector potential
    bem.rs          # Boundary Element Method: FlatPanel, single/double layer matrices
    electrostatic.rs # PointCharge, uniform patch potential/field
    retarded.rs     # Retarded potentials for time-varying sources

  circuits/
    mod.rs          # Circuit module exports
    component.rs    # Resistor, Inductor, Capacitor, VoltageSource, Switch
    network.rs      # Network aggregation (Series/Parallel)
    analysis.rs     # NodalAnalysis, AdmittanceMatrix, sweep utilities
    twoport.rs      # TwoPort ABCD/Z/Y/S parameter conversions
    transmission.rs # TransmissionLine (RLGC parameterization)
    stamp.rs        # NodalBuilder, MnaBuilder for nodal/MNA stamping
    nport.rs        # NPortNetwork, Touchstone sNp parser (feature: default)
    spice.rs        # SPICE netlist parser (R, L, C, V, I elements)

    # Sparse feature (--features sparse)
    sparse.rs       # Sparse matrix helpers
    solver.rs       # SparseSolver trait, BaselineLuSolver, BiCgStabSolver
    iterative.rs    # BiCGSTAB, GMRES solvers, ILU(0) preconditioner
    ordering.rs     # Matrix ordering algorithms (AMD, RCM)
```

## Key Abstractions

### Circuit Analysis

The library uses Modified Nodal Analysis (MNA) for circuit simulation:

```rust
use em_physics::prelude::*;

// Build MNA system
let mut mna = MnaBuilder::new(node_count);
mna.stamp_resistor(Some(0), Some(1), 1000.0);  // 1k between nodes 0-1
mna.stamp_capacitor(Some(1), None, 1e-9, AcContext { omega });  // 1nF to ground
mna.stamp_voltage_source(Some(0), None, Complex::new(5.0, 0.0));  // 5V source

let solution = mna.solve();
let (voltages, currents) = mna.split_solution(solution.unwrap());
```

### Controlled Sources

MNA supports all four controlled source types:
- VCCS: `stamp_vccs(op, on, cp, cn, g)` - voltage-controlled current source
- VCVS: `stamp_vcvs(op, on, cp, cn, mu)` - voltage-controlled voltage source
- CCCS: `stamp_cccs(op, on, ctrl_k, alpha)` - current-controlled current source
- CCVS: `stamp_ccvs(op, on, ctrl_k, mu)` - current-controlled voltage source

### Transient Simulation

```rust
let mut engine = MnaTransientEngine::new(node_count, TimeIntegrator::Trapezoidal);
engine.add_resistor(Some(0), None, 1000.0);
engine.add_capacitor(Some(0), None, 1e-6, 0.0);  // initial voltage = 0
engine.add_voltage_source(Some(0), None, |t| (2.0 * PI * 1000.0 * t).sin());

let config = SimulationConfig::time("transient", Duration::from_millis(10), Duration::from_micros(10));
engine.run(&config)?;
let waveform = engine.waveform();
```

### Sparse Solvers (feature: sparse)

For large systems (>10k nodes), use iterative Krylov solvers:

```rust
use em_physics::circuits::solver::{SparseSolver, BaselineLuSolver};
use em_physics::circuits::iterative::{BiCGSTAB, GMRES, ConvergenceCriteria};

// Direct solver (small systems)
let mut solver = BaselineLuSolver::new();

// Iterative solvers (large systems)
let mut solver = BiCGSTAB::new(ConvergenceCriteria::default());
let mut solver = GMRES::new(30, ConvergenceCriteria::default());  // restart=30

solver.symbolic(&matrix)?;
solver.numeric(&matrix)?;
let solution = solver.solve(&rhs)?;
```

## Conventions

### Node Representation
- `Node = Option<usize>` where `None` represents ground
- Node indices are 0-based

### Complex Numbers
- AC analysis uses `num_complex::Complex<f64>`
- Phasors: `Complex::from_polar(magnitude, phase_radians)`

### Physical Constants
All constants follow CODATA 2022 values from NIST. Constants exact by SI 2019 definition:
- `SPEED_OF_LIGHT`: 299,792,458 m/s (exact)
- `ELEMENTARY_CHARGE`: 1.602176634e-19 C (exact)
- `BOLTZMANN_CONSTANT`: 1.380649e-23 J/K (exact)

Measured values (~10^-10 relative uncertainty):
- `VACUUM_PERMITTIVITY`: 8.8541878188e-12 F/m
- `VACUUM_PERMEABILITY`: 1.25663706127e-6 H/m

### Testing
- Unit tests use `approx::assert_relative_eq!` for floating-point comparisons
- Tests validate against known analytical solutions or reference implementations

### Error Handling
- Solver failures return `Result<_, SolverError>` with variants:
  - `SingularMatrix`, `ConvergenceFailure`, `InvalidMatrix`, `NumericalInstability`
- Simulation errors use `SimulationError`

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std`   | Standard library (default) |
| `serde` | Serialization support for types |
| `sparse`| Sparse matrix solvers (nalgebra-sparse) |

## Dependencies

- `nalgebra`: Dense linear algebra
- `nalgebra-sparse`: Sparse matrices (optional)
- `num-complex`: Complex number support
- `num-traits`: Numeric traits
- `thiserror`: Error derive macros
- `approx`: Floating-point comparison (dev)
- `criterion`: Benchmarking (dev)
- `bevy`: Visualization example (dev)
