//! Baseline physical constants and utility functions.
//!
//! ## Accuracy
//!
//! Constants marked "exact" have zero uncertainty by SI definition (2019 revision).
//! Measured constants (ε₀, μ₀) are provided with 11-12 significant figures, suitable
//! for engineering applications. Values are approximations; for higher precision or
//! latest values, consult NIST directly.
//!
//! ## References
//!
//! Physical constants are based on CODATA recommended values:
//! - NIST Reference on Constants, Units, and Uncertainty: <https://physics.nist.gov/cuu/Constants/>
//! - CODATA 2018 values published May 20, 2019 (following 2019 SI redefinition)
//! - Mohr, P. J., Newell, D. B., Taylor, B. N., & Tiesinga, E. (2019). CODATA Recommended Values of the Fundamental Physical Constants: 2018.
//! - Note: Latest CODATA 2022 values differ in final digits for ε₀ and μ₀

use std::f64::consts::PI;

/// Vacuum permittivity ε₀ in farads per meter (F/m).
/// Approximate value: 8.8541878128 × 10⁻¹² F/m (11 significant figures).
/// Note: CODATA 2022 value is 8.8541878188 × 10⁻¹² F/m with relative uncertainty ~10⁻¹⁰.
pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12;
/// Vacuum permeability μ₀ in henries per meter (H/m).
/// Approximate value: 1.25663706212 × 10⁻⁶ H/m (12 significant figures).
/// Note: CODATA 2022 value is 1.25663706127 × 10⁻⁶ H/m with relative uncertainty ~10⁻¹⁰.
pub const VACUUM_PERMEABILITY: f64 = 1.256_637_062_12e-6;
/// Speed of light in vacuum _c_ in meters per second (m/s).
/// Exact value by SI definition (2019): 299,792,458 m/s.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Characteristic impedance of free space Z₀ in ohms (Ω).
/// Derived from Z₀ = √(μ₀/ε₀) ≈ 376.730313668 Ω.
pub const FREE_SPACE_IMPEDANCE: f64 = 376.730_313_668;
/// Elementary charge _e_ in coulombs (C).
/// Exact value by 2019 SI definition: 1.602176634 × 10⁻¹⁹ C.
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// Boltzmann constant _k_B_ in joules per kelvin (J/K).
/// Exact value by 2019 SI definition: 1.380649 × 10⁻²³ J/K.
pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23;

/// Returns the angular frequency corresponding to a linear frequency `hz`.
#[inline]
#[must_use]
pub fn angular_frequency(hz: f64) -> f64 {
    2.0 * PI * hz
}

/// Returns the free-space wavelength in meters for a given frequency in hertz.
#[inline]
#[must_use]
pub fn wavelength_from_frequency(hz: f64) -> f64 {
    SPEED_OF_LIGHT / hz
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn wavelength_matches_reference() {
        let freq = 1.0e9;
        let lambda = wavelength_from_frequency(freq);
        assert_relative_eq!(lambda, 0.299_792_458, max_relative = 1.0e-9);
    }
}
