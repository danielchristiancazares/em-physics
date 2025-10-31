//! High-level orchestration for time and frequency-domain experiments.

use std::time::Duration;

use crate::math::Scalar;

/// Supported simulation domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimulationDomain {
    /// Frequency-domain / phasor analysis.
    Frequency,
    /// Time-domain transient analysis.
    Time,
}

/// Metadata describing a simulation experiment.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Human-readable identifier.
    pub name: String,
    /// Domain of operation.
    pub domain: SimulationDomain,
    /// Target angular frequency for single-tone analysis.
    pub angular_frequency: Option<Scalar>,
    /// Total runtime for time-domain simulations.
    pub duration: Option<Duration>,
    /// Time step for integration (when applicable).
    pub time_step: Option<Duration>,
}

impl SimulationConfig {
    /// Creates a default frequency-domain configuration.
    #[must_use]
    pub fn frequency(name: impl Into<String>, angular_frequency: Scalar) -> Self {
        Self {
            name: name.into(),
            domain: SimulationDomain::Frequency,
            angular_frequency: Some(angular_frequency),
            duration: None,
            time_step: None,
        }
    }

    /// Creates a time-domain configuration.
    #[must_use]
    pub fn time(name: impl Into<String>, duration: Duration, time_step: Duration) -> Self {
        Self {
            name: name.into(),
            domain: SimulationDomain::Time,
            angular_frequency: None,
            duration: Some(duration),
            time_step: Some(time_step),
        }
    }
}

/// Trait for simulation engines.
pub trait SimulationEngine {
    /// Executes the simulation using the provided configuration.
    fn run(&mut self, config: &SimulationConfig) -> Result<(), SimulationError>;
}

/// Errors that can occur while configuring or executing simulations.
#[derive(Debug, thiserror::Error)]
pub enum SimulationError {
    /// Raised when a required parameter is missing.
    #[error("missing parameter: {0}")]
    MissingParameter(&'static str),
    /// Raised when the configuration is internally inconsistent.
    #[error("configuration error: {0}")]
    InvalidConfig(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEngine;

    impl SimulationEngine for DummyEngine {
        fn run(&mut self, config: &SimulationConfig) -> Result<(), SimulationError> {
            if config.domain == SimulationDomain::Frequency && config.angular_frequency.is_none()
            {
                return Err(SimulationError::MissingParameter("angular_frequency"));
            }
            Ok(())
        }
    }

    #[test]
    fn dummy_engine_validates_frequency_configs() {
        let mut engine = DummyEngine;
        let config = SimulationConfig::frequency("freq", 1.0);
        engine.run(&config).expect("valid config");
    }
}
