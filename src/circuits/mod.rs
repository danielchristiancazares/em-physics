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
/// Sparse linear system solvers (feature = "sparse").
#[cfg(feature = "sparse")]
pub mod solver;

pub use analysis::{AdmittanceMatrix, NodalAnalysis};
pub use component::{Capacitor, Component, Inductor, Resistor, Switch, VoltageSource};
pub use network::{ConnectionKind, Network};
pub use twoport::TwoPort;
