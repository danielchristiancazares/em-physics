//! Circuit primitives and solvers spanning lumped and distributed systems.

/// Frequency-domain circuit analysis utilities.
pub mod analysis;
/// Lumped component definitions and traits.
pub mod component;
/// Aggregate network composition helpers.
pub mod network;
/// Two-port network representations and conversions.
pub mod twoport;
/// Transmission line primitives and ABCD parameterization.
pub mod transmission;
/// Nodal stamping helpers for DC/AC analysis.
pub mod stamp;
/// Optional sparse helpers (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod sparse;
/// N-port networks and Touchstone import.
pub mod nport;
/// SPICE netlist importer (linear subset).
pub mod spice;
/// Sparse linear system solvers (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod solver;
/// Iterative Krylov subspace solvers (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod iterative;
/// Matrix ordering and reordering algorithms (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod ordering;

pub use analysis::{AdmittanceMatrix, NodalAnalysis};
pub use component::{Capacitor, Component, Inductor, Resistor, Switch, VoltageSource};
pub use network::{ConnectionKind, Network};
pub use twoport::TwoPort;
