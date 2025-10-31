//! Shared numerical primitives anchored on `nalgebra`.

use nalgebra::{Matrix3, Vector3};

/// Primary scalar type used across the crate.
pub type Scalar = f64;
/// Convenient alias for three-dimensional real vectors.
pub type R3 = Vector3<Scalar>;
/// Convenient alias for three-by-three real matrices.
pub type R3x3 = Matrix3<Scalar>;
/// Primary complex scalar type used for phasors.
pub type CScalar = num_complex::Complex<Scalar>;
/// Convenient alias for three-dimensional complex vectors.
pub type C3 = Vector3<CScalar>;

/// Returns the complex exponential `e^(j * theta)` using `Scalar` precision.
#[must_use]
pub fn phasor(theta: Scalar) -> num_complex::Complex<Scalar> {
    num_complex::Complex::from_polar(1.0, theta)
}

/// Computes the RMS magnitude of a sinusoidal waveform with peak value `peak`.
#[must_use]
pub fn sinusoid_rms(peak: Scalar) -> Scalar {
    peak / Scalar::sqrt(2.0)
}

/// Calculates the magnitude of a time-harmonic phasor vector.
#[must_use]
pub fn phasor_magnitude(vector: &R3) -> Scalar {
    vector.norm()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn phasor_magnitude_matches_euclidean_norm() {
        let v = R3::new(1.0, 2.0, 2.0);
        assert_relative_eq!(phasor_magnitude(&v), 3.0, epsilon = 1.0e-12);
    }
}
