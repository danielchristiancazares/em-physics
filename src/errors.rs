//! Shared error types used across submodules.

use thiserror::Error;

use crate::simulation::SimulationError;

/// Top-level error type for the crate.
#[derive(Debug, Error)]
pub enum EmPhysicsError {
    /// Wraps simulation-related errors.
    #[error(transparent)]
    Simulation(#[from] SimulationError),
    /// Raised when a component configuration is invalid.
    #[error("component error: {0}")]
    Component(String),
    /// Raised when numerical procedures fail to converge.
    #[error("solver convergence failure: {0}")]
    Convergence(String),
}
