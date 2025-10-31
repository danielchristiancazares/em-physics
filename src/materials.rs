//! Material property models and abstractions.

use crate::constants::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};
use crate::math::Scalar;
use num_complex::Complex;
use crate::units::Impedance;

/// Fundamental linear isotropic material parameters expressed in SI units.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialProperties {
    /// Electric permittivity ε in F/m.
    pub permittivity: Scalar,
    /// Magnetic permeability μ in H/m.
    pub permeability: Scalar,
    /// Electrical conductivity σ in S/m.
    pub conductivity: Scalar,
}

impl MaterialProperties {
    /// Creates material properties for free space.
    #[must_use]
    pub const fn vacuum() -> Self {
        Self {
            permittivity: VACUUM_PERMITTIVITY,
            permeability: VACUUM_PERMEABILITY,
            conductivity: 0.0,
        }
    }

    /// Computes the intrinsic impedance √(μ / ε).
    #[must_use]
    pub fn intrinsic_impedance(&self) -> Impedance<Scalar> {
        let value = (self.permeability / self.permittivity).sqrt();
        Impedance::new(value)
    }

    /// Complex response using ε_c(ω) = ε - jσ/ω and μ_c = μ (non-magnetic lossless).
    #[must_use]
    pub fn response(&self, omega: Scalar) -> MaterialResponse {
        let j = Complex::new(0.0, 1.0);
        let eps_c = Complex::new(self.permittivity, 0.0) - j * (self.conductivity / omega);
        let mu_c = Complex::new(self.permeability, 0.0);
        MaterialResponse { epsilon: eps_c, mu: mu_c, sigma: self.conductivity }
    }
}

/// Frequency-dependent complex material response.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialResponse {
    /// Complex permittivity ε(ω) in F/m.
    pub epsilon: Complex<Scalar>,
    /// Complex permeability μ(ω) in H/m.
    pub mu: Complex<Scalar>,
    /// Conduction σ in S/m (real), included separately for convenience.
    pub sigma: Scalar,
}

/// Trait for frequency-dependent material models.
pub trait DispersiveMaterial {
    /// Returns effective properties at the angular frequency `omega` (rad/s).
    fn properties(&self, omega: Scalar) -> MaterialProperties;
}

/// Trait for frequency-dependent complex response providers.
pub trait MaterialResponseProvider {
    /// Complex ε(ω), μ(ω), and σ at `omega`.
    fn response(&self, omega: Scalar) -> MaterialResponse;
}

/// Simple Drude dispersive material approximation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
pub struct DrudeModel {
    /// Angular plasma frequency ωₚ (rad/s).
    pub plasma_frequency: Scalar,
    /// Collision frequency γ (rad/s).
    pub collision_frequency: Scalar,
    /// High-frequency permittivity limit ε∞.
    pub epsilon_infinity: Scalar,
}

impl DispersiveMaterial for DrudeModel {
    fn properties(&self, omega: Scalar) -> MaterialProperties {
        if omega.abs() < Scalar::EPSILON {
            return MaterialProperties {
                permittivity: self.epsilon_infinity * VACUUM_PERMITTIVITY,
                permeability: VACUUM_PERMEABILITY,
                conductivity: 0.0,
            };
        }

        let j = Complex::new(0.0, 1.0);
        let omega_c = Complex::new(omega, 0.0);
        let numerator = Complex::new(self.plasma_frequency.powi(2), 0.0);
        let denominator = omega_c * (omega_c + j * self.collision_frequency);
        let epsilon_rel = Complex::new(self.epsilon_infinity, 0.0) - numerator / denominator;

        let epsilon = epsilon_rel.re * VACUUM_PERMITTIVITY;
        let sigma = -omega * epsilon_rel.im * VACUUM_PERMITTIVITY;

        MaterialProperties { permittivity: epsilon, permeability: VACUUM_PERMEABILITY, conductivity: sigma }
    }
}

impl MaterialResponseProvider for MaterialProperties {
    fn response(&self, omega: Scalar) -> MaterialResponse {
        self.response(omega)
    }
}

impl MaterialResponseProvider for DrudeModel {
    fn response(&self, omega: Scalar) -> MaterialResponse {
        if omega.abs() < Scalar::EPSILON {
            return MaterialResponse {
                epsilon: Complex::new(self.epsilon_infinity * VACUUM_PERMITTIVITY, 0.0),
                mu: Complex::new(VACUUM_PERMEABILITY, 0.0),
                sigma: 0.0,
            };
        }
        let j = Complex::new(0.0, 1.0);
        let omega_c = Complex::new(omega, 0.0);
        let numerator = Complex::new(self.plasma_frequency.powi(2), 0.0);
        let denominator = omega_c * (omega_c + j * self.collision_frequency);
        let epsilon_rel = Complex::new(self.epsilon_infinity, 0.0) - numerator / denominator;
        let epsilon = epsilon_rel * Complex::new(VACUUM_PERMITTIVITY, 0.0);
        MaterialResponse { epsilon, mu: Complex::new(VACUUM_PERMEABILITY, 0.0), sigma: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn vacuum_impedance_matches_reference() {
        let z0 = MaterialProperties::vacuum().intrinsic_impedance();
        // Z₀ = √(μ₀/ε₀) using CODATA 2022 values = 376.730313412 Ω
        assert_relative_eq!(z0.value(), 376.730_313_412, epsilon = 1.0e-9);
        let printed = format!("{z0}");
        assert!(
            printed.ends_with('Ω'),
            "expected impedance string to include ohm symbol, got {printed}"
        );
    }
}
