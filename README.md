# em-physics

`em-physics` is a research-grade Rust library under active development for modeling electromagnetism and circuit behavior. The crate aims to provide reusable building blocks—components, material models, field solvers, and simulation utilities—so researchers and engineers can assemble complex experiments without reinventing core EM calculations.

## Project status

The project is in the scaffolding phase. The initial focus is on establishing a robust architecture, shared numerical foundations, and high-quality documentation. Expect rapid iteration and API evolution before any stable release.

## Quick start

Build and run a simple series RLC AC sweep example:

```sh
cargo run --example rlc_sweep
```

Launch the interactive Bevy visualization tying circuit gating to field probes:

```sh
cargo run --example bevy_circuit
```

Programmatic usage (sketch):

```rust,no_run
use em_physics::circuits::network::{ConnectionKind, Network};
use em_physics::circuits::component::{Resistor, Inductor, Capacitor};
use em_physics::circuits::analysis::sweep_network_impedance;
use em_physics::constants::angular_frequency;

let mut net = Network::new("series_rlc", ConnectionKind::Series);
net.add_component(Resistor::new("R1", 50.0));
net.add_component(Inductor::new("L1", 1e-6));
net.add_component(Capacitor::new("C1", 1e-9));

let omegas = [1.0e6, 10.0e6, 100.0e6].into_iter().map(angular_frequency);
let data = sweep_network_impedance(&net, omegas);
```

## Roadmap (high level)

- Fundamental constants, units, and numerical traits for strongly typed EM quantities.
- Models for electric and magnetic fields, including high-frequency behavior.
- Circuit primitives compatible with multi-domain simulations (lumped, distributed, transmission-line).
- Material models capturing permittivity, permeability, conductivity, and dispersion.
- Deterministic and stochastic solvers with rigorous validation against reference data.
- Integration tooling for benchmarks, reproducible experiments, and visualization pipelines.

## Implementation status

### Fully functional (with tests)
- **Two-port networks**: ABCD-based representation with Z/Y/S parameter conversions and cascading
- **Transmission lines**: RLGC parameterization producing ABCD matrices vs. frequency
- **Nodal analysis**: Dense stamping builders supporting:
  - NodalBuilder: DC/AC stamping of R, L, C, current sources
  - MnaBuilder: Modified nodal analysis with voltage sources and controlled sources (VCCS, CCCS, VCVS, CCVS)
- **Circuit components**: Resistor, Inductor, Capacitor, VoltageSource with Component trait
- **Network composition**: Series and parallel aggregation via Network type
- **Frequency sweeps**: Linear/log angular-frequency generators, magnitude/phase/dB helpers
- **AC sweep driver**: Stamp-per-frequency closure over MNA returning node voltages and source currents
- **Field sources**: Biot-Savart calculations for line currents, vector potential, quasi-static E-field
- **Interactive rendering**: `bevy_circuit` example with keyboard-controlled gates, live current solver, and field readouts
- **Material models**: Basic MaterialProperties and Drude dispersion model

### Scaffolding (trait definitions, minimal implementation)
- **Simulation orchestration**: SimulationConfig and SimulationEngine trait defined but no concrete engines implemented
- **Field representations**: ElectricField and MagneticField descriptors exist with limited physics functionality

## Numerical accuracy

### Floating-point precision
All calculations use IEEE 754 double precision (f64), providing ~15-17 decimal digits of precision. This is suitable for engineering and research applications but not arbitrary-precision symbolic computation.

### Physical constants precision

Constants are based on CODATA recommended values:

| Constant | Code value | Significant figures | Status | Relative accuracy |
|----------|-----------|---------------------|---------|-------------------|
| Speed of light *c* | 299,792,458 m/s | exact | SI definition (2019) | exact |
| Elementary charge *e* | 1.602176634×10⁻¹⁹ C | exact | SI definition (2019) | exact |
| Boltzmann constant *k_B* | 1.380649×10⁻²³ J/K | exact | SI definition (2019) | exact |
| Vacuum permittivity ε₀ | 8.8541878128×10⁻¹² F/m | 11 digits | measured (CODATA) | ~10⁻¹⁰ |
| Vacuum permeability μ₀ | 1.25663706212×10⁻⁶ H/m | 12 digits | measured (CODATA) | ~10⁻¹⁰ |

**Note**: The ε₀ and μ₀ values in the code are approximations. Latest NIST CODATA 2022 values differ in the final digits. For applications requiring higher precision, consult <https://physics.nist.gov/cuu/Constants/> for current values and uncertainties.

### Numerical methods accuracy
- **Biot-Savart integration**: Midpoint quadrature with adaptive sampling (20-800 subdivisions). Error scales as 1/N² for smooth integrands. Typical relative error ~10⁻⁶ to 10⁻⁹ for well-behaved geometries.
- **Nodal analysis**: Direct LU solver. Accuracy limited by matrix conditioning (condition number). Well-conditioned circuits achieve relative error ~10⁻¹² to 10⁻¹⁴.
- **Transmission line ABCD**: Closed-form hyperbolic functions (cosh, sinh). Error dominated by f64 precision (~10⁻¹⁵).

## Scientific references

### Physical constants
Physical constants follow CODATA recommended values published by NIST:
- Mohr, P. J., Newell, D. B., Taylor, B. N., & Tiesinga, E. (2019). *CODATA Recommended Values of the Fundamental Physical Constants: 2018*. Available at: <https://physics.nist.gov/cuu/Constants/>
- Latest values: NIST CODATA 2022 (<https://physics.nist.gov/cuu/Constants/>)
- Values reflect the 2019 SI redefinition where certain constants (speed of light, elementary charge, Boltzmann constant) are exact by definition

### Circuit analysis
- Two-port network ABCD parameters and conversions: Pozar, D. M. (2011). *Microwave Engineering* (4th ed.). Wiley. Chapter 4.
- Modified Nodal Analysis (MNA): Ho, C. W., Ruehli, A. E., & Brennan, P. A. (1975). The modified nodal approach to network analysis. *IEEE Transactions on Circuits and Systems*, 22(6), 504-509.
- Transmission line theory (RLGC parameters): Paul, C. R. (2007). *Analysis of Multiconductor Transmission Lines* (2nd ed.). Wiley-IEEE Press.

### Electromagnetic field theory
- Biot-Savart law for line currents: Jackson, J. D. (1998). *Classical Electrodynamics* (3rd ed.). Wiley. Section 5.4.
- Vector potential and quasi-static approximation: Griffiths, D. J. (2017). *Introduction to Electrodynamics* (4th ed.). Cambridge University Press. Chapter 10.

## Contributing

The contribution workflow and coding guidelines will be documented as the API solidifies. For now, contributions are limited to the core team while the architecture stabilizes.
