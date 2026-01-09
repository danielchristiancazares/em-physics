//! Shared numerical primitives anchored on `nalgebra`.

use std::cmp::Ordering;

use nalgebra::{Matrix3, Vector2, Vector3};
use thiserror::Error;

/// Primary scalar type used across the crate.
pub type Scalar = f64;
/// Convenient alias for two-dimensional real vectors.
pub type R2 = Vector2<Scalar>;
/// Convenient alias for three-dimensional real vectors.
pub type R3 = Vector3<Scalar>;
/// Convenient alias for three-by-three real matrices.
pub type R3x3 = Matrix3<Scalar>;
/// Primary complex scalar type used for phasors.
pub type CScalar = num_complex::Complex<Scalar>;
/// Convenient alias for three-dimensional complex vectors.
pub type C3 = Vector3<CScalar>;

/// Errors returned when fitting or evaluating splines.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SplineError {
    /// Raised when there are not enough data points to fit a spline.
    #[error("spline requires at least two points, got {0}")]
    NotEnoughPoints(usize),
    /// Raised when input slices have different lengths.
    #[error("mismatched lengths: {left} vs {right}")]
    MismatchedLengths { left: usize, right: usize },
    /// Raised when x-values are not strictly increasing.
    #[error("x values must be strictly increasing; check index {index}")]
    NonIncreasing { index: usize },
    /// Raised when an x-value is not finite.
    #[error("non-finite x value at index {index}")]
    NonFiniteX { index: usize },
    /// Raised when a y-value is not finite.
    #[error("non-finite y value at index {index}")]
    NonFiniteY { index: usize },
}

/// Natural cubic spline defined by knot points and per-segment coefficients.
#[derive(Debug, Clone)]
pub struct NaturalCubicSpline {
    xs: Vec<Scalar>,
    coeffs: Vec<[Scalar; 4]>,
}

impl NaturalCubicSpline {
    /// Fits a natural cubic spline through `(xs, ys)` pairs.
    pub fn fit(xs: &[Scalar], ys: &[Scalar]) -> Result<Self, SplineError> {
        if xs.len() != ys.len() {
            return Err(SplineError::MismatchedLengths {
                left: xs.len(),
                right: ys.len(),
            });
        }
        if xs.len() < 2 {
            return Err(SplineError::NotEnoughPoints(xs.len()));
        }
        for (index, value) in xs.iter().enumerate() {
            if !value.is_finite() {
                return Err(SplineError::NonFiniteX { index });
            }
        }
        for (index, value) in ys.iter().enumerate() {
            if !value.is_finite() {
                return Err(SplineError::NonFiniteY { index });
            }
        }
        for i in 0..xs.len() - 1 {
            if xs[i + 1] <= xs[i] {
                return Err(SplineError::NonIncreasing { index: i });
            }
        }

        let n = xs.len();
        let mut h = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            h.push(xs[i + 1] - xs[i]);
        }

        let mut alpha = vec![0.0; n];
        for i in 1..n - 1 {
            alpha[i] = (3.0 / h[i]) * (ys[i + 1] - ys[i]) - (3.0 / h[i - 1]) * (ys[i] - ys[i - 1]);
        }

        let mut l = vec![0.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;

        for i in 1..n - 1 {
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n - 1] = 1.0;
        z[n - 1] = 0.0;

        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n - 1];
        let mut d = vec![0.0; n - 1];
        let a_values = ys.to_vec();

        for j in (0..n - 1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a_values[j + 1] - a_values[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        let mut coeffs = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            coeffs.push([a_values[i], b[i], c[i], d[i]]);
        }

        Ok(Self {
            xs: xs.to_vec(),
            coeffs,
        })
    }

    /// Evaluates the spline at `x`, clamping to the spline domain.
    #[must_use]
    pub fn evaluate(&self, x: Scalar) -> Scalar {
        if !x.is_finite() {
            return Scalar::NAN;
        }
        let min = self.xs[0];
        let max = self.xs[self.xs.len() - 1];
        let x = x.clamp(min, max);
        let index = Self::segment_index(&self.xs, x);
        let dx = x - self.xs[index];
        let [a, b, c, d] = self.coeffs[index];
        a + b * dx + c * dx * dx + d * dx * dx * dx
    }

    fn segment_index(xs: &[Scalar], x: Scalar) -> usize {
        match xs.binary_search_by(|value| value.partial_cmp(&x).unwrap_or(Ordering::Less)) {
            Ok(index) => {
                if index == xs.len() - 1 {
                    xs.len() - 2
                } else {
                    index
                }
            }
            Err(index) => index.saturating_sub(1).min(xs.len() - 2),
        }
    }
}

/// Parametric natural cubic spline in 2D using a shared parameter `t`.
#[derive(Debug, Clone)]
pub struct ParametricSpline2 {
    x: NaturalCubicSpline,
    y: NaturalCubicSpline,
}

impl ParametricSpline2 {
    /// Fits a 2D parametric spline through points sampled at `ts`.
    pub fn fit(ts: &[Scalar], points: &[R2]) -> Result<Self, SplineError> {
        if ts.len() != points.len() {
            return Err(SplineError::MismatchedLengths {
                left: ts.len(),
                right: points.len(),
            });
        }
        let xs: Vec<Scalar> = points.iter().map(|p| p.x).collect();
        let ys: Vec<Scalar> = points.iter().map(|p| p.y).collect();
        Ok(Self {
            x: NaturalCubicSpline::fit(ts, &xs)?,
            y: NaturalCubicSpline::fit(ts, &ys)?,
        })
    }

    /// Evaluates the parametric spline at `t`, clamping to the spline domain.
    #[must_use]
    pub fn evaluate(&self, t: Scalar) -> R2 {
        R2::new(self.x.evaluate(t), self.y.evaluate(t))
    }
}

/// Parametric natural cubic spline in 3D using a shared parameter `t`.
#[derive(Debug, Clone)]
pub struct ParametricSpline3 {
    x: NaturalCubicSpline,
    y: NaturalCubicSpline,
    z: NaturalCubicSpline,
}

impl ParametricSpline3 {
    /// Fits a 3D parametric spline through points sampled at `ts`.
    pub fn fit(ts: &[Scalar], points: &[R3]) -> Result<Self, SplineError> {
        if ts.len() != points.len() {
            return Err(SplineError::MismatchedLengths {
                left: ts.len(),
                right: points.len(),
            });
        }
        let xs: Vec<Scalar> = points.iter().map(|p| p.x).collect();
        let ys: Vec<Scalar> = points.iter().map(|p| p.y).collect();
        let zs: Vec<Scalar> = points.iter().map(|p| p.z).collect();
        Ok(Self {
            x: NaturalCubicSpline::fit(ts, &xs)?,
            y: NaturalCubicSpline::fit(ts, &ys)?,
            z: NaturalCubicSpline::fit(ts, &zs)?,
        })
    }

    /// Evaluates the parametric spline at `t`, clamping to the spline domain.
    #[must_use]
    pub fn evaluate(&self, t: Scalar) -> R3 {
        R3::new(self.x.evaluate(t), self.y.evaluate(t), self.z.evaluate(t))
    }
}

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

    #[test]
    fn natural_cubic_matches_linear_function() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys: Vec<Scalar> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let spline = NaturalCubicSpline::fit(&xs, &ys).unwrap();

        assert_relative_eq!(spline.evaluate(0.5), 2.0 * 0.5 + 1.0, epsilon = 1.0e-12);
        assert_relative_eq!(spline.evaluate(2.5), 2.0 * 2.5 + 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn natural_cubic_interpolates_knots() {
        let xs = [0.0, 0.5, 1.5, 3.0];
        let ys = [1.0, -1.0, 2.0, 0.0];
        let spline = NaturalCubicSpline::fit(&xs, &ys).unwrap();

        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_relative_eq!(spline.evaluate(*x), *y, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn natural_cubic_has_zero_boundary_second_derivative() {
        let xs = [0.0, 1.0, 2.0, 4.0];
        let ys = [0.0, 1.0, 0.0, 2.0];
        let spline = NaturalCubicSpline::fit(&xs, &ys).unwrap();

        let first_c = spline.coeffs[0][2];
        let last_index = spline.coeffs.len() - 1;
        let last_c = spline.coeffs[last_index][2];
        let last_d = spline.coeffs[last_index][3];
        let last_h = spline.xs[last_index + 1] - spline.xs[last_index];

        let second_start = 2.0 * first_c;
        let second_end = 2.0 * last_c + 6.0 * last_d * last_h;

        assert_relative_eq!(second_start, 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(second_end, 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn natural_cubic_rejects_non_increasing_x() {
        let xs = [0.0, 0.0, 1.0];
        let ys = [1.0, 2.0, 3.0];
        let err = NaturalCubicSpline::fit(&xs, &ys).unwrap_err();
        assert!(matches!(err, SplineError::NonIncreasing { .. }));
    }

    #[test]
    fn parametric_spline2_interpolates_points() {
        let ts = [0.0, 1.0, 2.0, 3.0];
        let points: Vec<R2> = ts.iter().map(|t| R2::new(*t, t * t)).collect();
        let spline = ParametricSpline2::fit(&ts, &points).unwrap();

        for (t, point) in ts.iter().zip(points.iter()) {
            let value = spline.evaluate(*t);
            assert_relative_eq!(value.x, point.x, epsilon = 1.0e-12);
            assert_relative_eq!(value.y, point.y, epsilon = 1.0e-12);
        }
    }

    #[test]
    fn parametric_spline3_interpolates_points() {
        let ts = [0.0, 0.5, 1.5, 2.5];
        let points: Vec<R3> = ts.iter().map(|t| R3::new(*t, 2.0 * t, 1.0 - t)).collect();
        let spline = ParametricSpline3::fit(&ts, &points).unwrap();

        for (t, point) in ts.iter().zip(points.iter()) {
            let value = spline.evaluate(*t);
            assert_relative_eq!(value.x, point.x, epsilon = 1.0e-12);
            assert_relative_eq!(value.y, point.y, epsilon = 1.0e-12);
            assert_relative_eq!(value.z, point.z, epsilon = 1.0e-12);
        }
    }
}
