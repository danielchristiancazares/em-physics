//! Circuit primitives and solvers spanning lumped and distributed systems.

/// Frequency-domain circuit analysis utilities.
pub mod analysis;
/// Lumped component definitions and traits.
pub mod component;
/// Iterative Krylov subspace solvers (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod iterative;
/// Aggregate network composition helpers.
pub mod network;
/// N-port networks and Touchstone import.
pub mod nport;
/// Matrix ordering and reordering algorithms (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod ordering;
/// Sparse linear system solvers (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod solver;
/// Optional sparse helpers (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod sparse;
/// SPICE netlist importer (linear subset).
pub mod spice;
/// Nodal stamping helpers for DC/AC analysis.
pub mod stamp;
/// Transmission line primitives and ABCD parameterization.
pub mod transmission;
/// Two-port network representations and conversions.
pub mod twoport;
/// Vector Fitting algorithm for rational approximation.
pub mod vf;

pub use analysis::{AdmittanceMatrix, NodalAnalysis};
pub use component::{Capacitor, Component, Inductor, Resistor, Switch, VoltageSource};
pub use network::{ConnectionKind, Network};
pub use twoport::TwoPort;
