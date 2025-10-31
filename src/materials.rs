//! Material property models and abstractions.

use crate::constants::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};
use crate::math::Scalar;
use crate::units::Impedance;
use num_complex::Complex;

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
}

/// Trait for frequency-dependent material models.
pub trait DispersiveMaterial {
    /// Returns effective properties at the angular frequency `omega` (rad/s).
    fn properties(&self, omega: Scalar) -> MaterialProperties;
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

        MaterialProperties {
            permittivity: epsilon,
            permeability: VACUUM_PERMEABILITY,
            conductivity: sigma,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn vacuum_impedance_matches_reference() {
        let z0 = MaterialProperties::vacuum().intrinsic_impedance();
        assert_relative_eq!(z0.value(), 376.730_313_668, epsilon = 1.0e-6);
        let printed = format!("{z0}");
        assert!(
            printed.ends_with('Ω'),
            "expected impedance string to include ohm symbol, got {printed}"
        );
    }
}
