#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![warn(clippy::all, clippy::cargo, clippy::nursery, missing_docs)]
#![doc = include_str!("../README.md")]

/// Fundamental physical constants used throughout the library.
pub mod constants;
/// Strongly typed unit helpers and quantity abstractions.
pub mod units;
/// Shared mathematical utilities (vectors, matrices, transforms).
pub mod math;
/// Electromagnetic field representations and utilities.
pub mod fields;
/// Material property models (permittivity, conductivity, etc.).
pub mod materials;
/// Circuit components, networks, and solvers.
pub mod circuits;
/// Frequency sweep builders and post-processing helpers.
pub mod sweep;
/// High-level simulation orchestrators and experiment builders.
pub mod simulation;
/// Error types shared between sub-crates.
pub mod errors;

/// Common exports for downstream crates.
pub mod prelude;
