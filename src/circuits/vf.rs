//! Vector Fitting algorithm for rational approximation of frequency-domain data.
//!
//! This module implements the Vector Fitting (VF) algorithm, which approximates
//! frequency-domain transfer function data as a sum of partial fractions:
//!
//! ```text
//! H(s) = d + s·e + Σₖ (rₖ / (s - pₖ))
//! ```
//!
//! The algorithm iteratively relocates poles via weighted least-squares until
//! convergence, then solves for the final residues.
//!
//! # References
//!
//! - Gustavsen, B., & Semlyen, A. (1999). "Rational Approximation of Frequency
//!   Domain Responses by Vector Fitting". IEEE Trans. Power Delivery.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::math::Scalar;

/// Complex scalar type alias.
type CScalar = Complex<Scalar>;

/// Error types for Vector Fitting operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VFError {
    /// Input frequency and sample arrays have different lengths.
    LengthMismatch { frequencies: usize, samples: usize },
    /// Not enough frequency samples for the requested number of poles.
    InsufficientSamples { samples: usize, poles: usize },
    /// The linear system became singular during fitting.
    SingularSystem,
    /// Eigenvalue computation failed during pole relocation.
    EigenvalueFailed,
    /// No frequency data provided.
    EmptyInput,
}

impl std::fmt::Display for VFError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch {
                frequencies,
                samples,
            } => {
                write!(
                    f,
                    "length mismatch: {frequencies} frequencies vs {samples} samples"
                )
            }
            Self::InsufficientSamples { samples, poles } => {
                write!(
                    f,
                    "insufficient samples: {samples} samples for {poles} poles"
                )
            }
            Self::SingularSystem => write!(f, "singular system during fitting"),
            Self::EigenvalueFailed => write!(f, "eigenvalue computation failed"),
            Self::EmptyInput => write!(f, "empty input data"),
        }
    }
}

impl std::error::Error for VFError {}

/// A rational function model from Vector Fitting.
///
/// Represents a transfer function in pole-residue form:
/// ```text
/// H(s) = d + s·e + Σₖ (residues[k] / (s - poles[k]))
/// ```
#[derive(Debug, Clone)]
pub struct RationalModel {
    /// Complex poles (typically come in conjugate pairs for real-valued systems).
    pub poles: Vec<CScalar>,
    /// Residues corresponding to each pole.
    pub residues: Vec<CScalar>,
    /// Direct feedthrough term.
    pub d: CScalar,
    /// Proportional term (typically zero).
    pub e: CScalar,
}

impl RationalModel {
    /// Evaluate the rational function at a complex frequency s.
    #[must_use]
    pub fn eval(&self, s: CScalar) -> CScalar {
        let mut sum = self.d + self.e * s;
        for (&p, &r) in self.poles.iter().zip(self.residues.iter()) {
            sum += r / (s - p);
        }
        sum
    }

    /// Evaluate at a frequency in Hz (s = j·2π·f).
    #[must_use]
    pub fn eval_hz(&self, freq_hz: Scalar) -> CScalar {
        self.eval(Complex::new(0.0, 2.0 * PI * freq_hz))
    }

    /// Evaluate at an angular frequency ω (s = jω).
    #[must_use]
    pub fn eval_omega(&self, omega: Scalar) -> CScalar {
        self.eval(Complex::new(0.0, omega))
    }

    /// Returns the number of poles in the model.
    #[must_use]
    pub fn pole_count(&self) -> usize {
        self.poles.len()
    }

    /// Returns true if all poles have negative real parts (stable system).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.poles.iter().all(|p| p.re < 0.0)
    }
}

/// Configuration for the Vector Fitting algorithm.
#[derive(Debug, Clone)]
pub struct VFConfig {
    /// Number of poles to use in the approximation.
    pub n_poles: usize,
    /// Maximum iterations for pole relocation.
    pub max_iter: usize,
    /// Convergence tolerance (relative pole movement).
    pub tol: Scalar,
    /// Whether to enforce stability by flipping unstable poles.
    pub enforce_stable: bool,
    /// Whether to fit the proportional term e (usually false).
    pub fit_e: bool,
}

impl Default for VFConfig {
    fn default() -> Self {
        Self {
            n_poles: 10,
            max_iter: 50,
            tol: 1e-8,
            enforce_stable: true,
            fit_e: false,
        }
    }
}

/// Result of a Vector Fitting run.
#[derive(Debug, Clone)]
pub struct VFResult {
    /// The fitted rational model.
    pub model: RationalModel,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
    /// Root-mean-square error of the final fit.
    pub final_rmse: Scalar,
}

/// Generate initial poles distributed across the frequency range.
///
/// Uses logarithmic spacing with complex conjugate pairs having small negative
/// real parts for initial stability.
fn initial_poles(f_min: Scalar, f_max: Scalar, n_poles: usize) -> Vec<CScalar> {
    if n_poles == 0 {
        return Vec::new();
    }

    let mut poles = Vec::with_capacity(n_poles);

    // Ensure sensible bounds
    let f_min = f_min.max(1e-6);
    let f_max = f_max.max(f_min * 10.0);

    let log_min = f_min.ln();
    let log_max = f_max.ln();

    // Generate conjugate pairs
    let n_pairs = n_poles / 2;
    for i in 0..n_pairs {
        let t = if n_pairs > 1 {
            i as Scalar / (n_pairs - 1) as Scalar
        } else {
            0.5
        };
        let f = (log_min + t * (log_max - log_min)).exp();
        let omega = 2.0 * PI * f;
        // Small negative real part for stability (1% damping)
        let alpha = -omega * 0.01;
        poles.push(Complex::new(alpha, omega));
        poles.push(Complex::new(alpha, -omega));
    }

    // If odd number of poles, add one real pole
    if n_poles % 2 == 1 {
        let f_mid = ((log_min + log_max) / 2.0).exp();
        let omega = 2.0 * PI * f_mid;
        poles.push(Complex::new(-omega * 0.01, 0.0));
    }

    poles
}

/// Compute new poles from the sigma function residues.
///
/// The new poles are eigenvalues of the matrix: diag(old_poles) - ones * sigma^T.
/// Since nalgebra doesn't support eigenvalues of complex matrices directly, we
/// convert to a real 2n×2n block matrix representation.
fn new_poles_from_sigma(
    old_poles: &[CScalar],
    sigma_residues: &[CScalar],
) -> Result<Vec<CScalar>, VFError> {
    let n = old_poles.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build the complex matrix H = diag(poles) - ones * sigma^T
    // Then convert to real 2n×2n representation:
    // For complex a+bi, the real block is [[a, -b], [b, a]]
    let mut h_real = DMatrix::<Scalar>::zeros(2 * n, 2 * n);

    for i in 0..n {
        // Diagonal: old_poles[i]
        let diag_re = old_poles[i].re;
        let diag_im = old_poles[i].im;

        // Place the 2x2 block for diagonal element
        h_real[(2 * i, 2 * i)] = diag_re;
        h_real[(2 * i, 2 * i + 1)] = -diag_im;
        h_real[(2 * i + 1, 2 * i)] = diag_im;
        h_real[(2 * i + 1, 2 * i + 1)] = diag_re;

        // Subtract sigma_residues contribution (rank-1 update)
        for j in 0..n {
            let sr_re = sigma_residues[j].re;
            let sr_im = sigma_residues[j].im;

            h_real[(2 * i, 2 * j)] -= sr_re;
            h_real[(2 * i, 2 * j + 1)] -= -sr_im;
            h_real[(2 * i + 1, 2 * j)] -= sr_im;
            h_real[(2 * i + 1, 2 * j + 1)] -= sr_re;
        }
    }

    // Compute eigenvalues of the real 2n×2n matrix
    let eig = h_real.complex_eigenvalues();

    // Extract complex eigenvalues (they come in conjugate pairs from real matrix)
    // We only need n of them (one from each pair)
    let mut poles: Vec<CScalar> = Vec::with_capacity(n);
    let mut i = 0;
    while i < eig.len() && poles.len() < n {
        let e = eig[i];
        poles.push(e);
        // Skip conjugate if present (eigenvalues with nonzero imag part come in pairs)
        if e.im.abs() > 1e-10 && i + 1 < eig.len() {
            i += 1; // Skip the conjugate
        }
        i += 1;
    }

    // If we still need more poles, we might have real eigenvalues
    // Ensure we have exactly n poles
    while poles.len() < n && poles.len() < eig.len() {
        poles.push(eig[poles.len()]);
    }

    Ok(poles)
}

/// Perform one iteration of the VF algorithm.
///
/// Builds the augmented linear system and solves for pole-relocating weights.
fn vf_iteration(
    s_points: &[CScalar],
    samples: &[CScalar],
    poles: &[CScalar],
    config: &VFConfig,
) -> Result<Vec<CScalar>, VFError> {
    let n_freq = s_points.len();
    let n_poles = poles.len();

    // Number of columns: residues (n_poles) + d + [e] + sigma_residues (n_poles)
    let n_cols = 2 * n_poles + if config.fit_e { 2 } else { 1 };

    // Build the overdetermined system A*x = b
    let mut a = DMatrix::<CScalar>::zeros(n_freq, n_cols);
    let mut b = DVector::<CScalar>::zeros(n_freq);

    for (i, (&s, &h)) in s_points.iter().zip(samples.iter()).enumerate() {
        // Columns for residues: 1/(s - pₖ)
        for (k, &p) in poles.iter().enumerate() {
            a[(i, k)] = CScalar::new(1.0, 0.0) / (s - p);
        }

        // Column for d (direct term)
        a[(i, n_poles)] = CScalar::new(1.0, 0.0);

        // Column for e (proportional term) if fitting
        let col_offset = if config.fit_e {
            a[(i, n_poles + 1)] = s;
            2
        } else {
            1
        };

        // Columns for σ weights: -H/(s - pₖ)
        for (k, &p) in poles.iter().enumerate() {
            a[(i, n_poles + col_offset + k)] = -h / (s - p);
        }

        b[i] = h;
    }

    // Solve via SVD (least squares for overdetermined system)
    let svd = a.svd(true, true);
    let x = svd.solve(&b, 1e-12).map_err(|_| VFError::SingularSystem)?;

    // Extract σ coefficients
    let col_offset = if config.fit_e { 2 } else { 1 };
    let sigma_residues: Vec<CScalar> = (0..n_poles).map(|k| x[n_poles + col_offset + k]).collect();

    // Compute new poles as eigenvalues
    new_poles_from_sigma(poles, &sigma_residues)
}

/// Compute the final residues once poles have converged.
fn compute_residues(
    s_points: &[CScalar],
    samples: &[CScalar],
    poles: &[CScalar],
    config: &VFConfig,
) -> Result<RationalModel, VFError> {
    let n_freq = s_points.len();
    let n_poles = poles.len();

    // Columns: residues (n_poles) + d + [e]
    let n_cols = n_poles + if config.fit_e { 2 } else { 1 };

    let mut a = DMatrix::<CScalar>::zeros(n_freq, n_cols);
    let b = DVector::from_iterator(n_freq, samples.iter().cloned());

    for (i, &s) in s_points.iter().enumerate() {
        for (k, &p) in poles.iter().enumerate() {
            a[(i, k)] = CScalar::new(1.0, 0.0) / (s - p);
        }
        a[(i, n_poles)] = CScalar::new(1.0, 0.0);
        if config.fit_e {
            a[(i, n_poles + 1)] = s;
        }
    }

    let svd = a.svd(true, true);
    let x = svd.solve(&b, 1e-12).map_err(|_| VFError::SingularSystem)?;

    let residues: Vec<CScalar> = (0..n_poles).map(|k| x[k]).collect();
    let d = x[n_poles];
    let e = if config.fit_e {
        x[n_poles + 1]
    } else {
        CScalar::new(0.0, 0.0)
    };

    Ok(RationalModel {
        poles: poles.to_vec(),
        residues,
        d,
        e,
    })
}

/// Compute RMSE between model and samples.
fn compute_rmse(s_points: &[CScalar], samples: &[CScalar], model: &RationalModel) -> Scalar {
    if s_points.is_empty() {
        return 0.0;
    }

    let sum_sq: Scalar = s_points
        .iter()
        .zip(samples.iter())
        .map(|(&s, &h)| {
            let h_fit = model.eval(s);
            (h - h_fit).norm_sqr()
        })
        .sum();

    (sum_sq / s_points.len() as Scalar).sqrt()
}

/// Compute relative change in poles between iterations.
fn pole_change(old: &[CScalar], new: &[CScalar]) -> Scalar {
    if old.is_empty() {
        return 0.0;
    }

    let max_change: Scalar = old
        .iter()
        .zip(new.iter())
        .map(|(o, n)| (o - n).norm() / o.norm().max(1e-10))
        .fold(0.0, Scalar::max);

    max_change
}

/// Fit a rational model to frequency-domain data using Vector Fitting.
///
/// # Arguments
///
/// * `frequencies_hz` - Frequency points in Hz
/// * `samples` - Complex transfer function values at each frequency
/// * `config` - Algorithm configuration
///
/// # Returns
///
/// A `VFResult` containing the fitted model and convergence information.
///
/// # Errors
///
/// Returns `VFError` if input validation fails or the algorithm encounters
/// numerical issues.
///
/// # Example
///
/// ```ignore
/// use em_physics::circuits::vf::{fit, VFConfig};
/// use num_complex::Complex;
///
/// // Fit a simple first-order system
/// let freqs: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
/// let samples: Vec<Complex<f64>> = freqs.iter()
///     .map(|&f| {
///         let s = Complex::new(0.0, 2.0 * std::f64::consts::PI * f);
///         Complex::new(1.0, 0.0) / (s + Complex::new(1.0, 0.0))
///     })
///     .collect();
///
/// let config = VFConfig { n_poles: 2, ..Default::default() };
/// let result = fit(&freqs, &samples, &config).unwrap();
/// assert!(result.converged);
/// ```
pub fn fit(
    frequencies_hz: &[Scalar],
    samples: &[CScalar],
    config: &VFConfig,
) -> Result<VFResult, VFError> {
    // Input validation
    if frequencies_hz.is_empty() {
        return Err(VFError::EmptyInput);
    }
    if frequencies_hz.len() != samples.len() {
        return Err(VFError::LengthMismatch {
            frequencies: frequencies_hz.len(),
            samples: samples.len(),
        });
    }

    let n_freq = frequencies_hz.len();
    let n_poles = config.n_poles;

    // Need at least 2*n_poles + 2 samples for the overdetermined system
    let min_samples = 2 * n_poles + 2;
    if n_freq < min_samples {
        return Err(VFError::InsufficientSamples {
            samples: n_freq,
            poles: n_poles,
        });
    }

    // Convert to s = jω
    let s_points: Vec<CScalar> = frequencies_hz
        .iter()
        .map(|&f| Complex::new(0.0, 2.0 * PI * f))
        .collect();

    // Initial poles distributed across frequency range
    let f_min = frequencies_hz
        .iter()
        .cloned()
        .fold(Scalar::INFINITY, Scalar::min);
    let f_max = frequencies_hz.iter().cloned().fold(0.0, Scalar::max);
    let mut poles = initial_poles(f_min, f_max, n_poles);

    // Iterative pole relocation
    for iter in 0..config.max_iter {
        let poles_prev = poles.clone();

        // Perform one VF iteration
        poles = vf_iteration(&s_points, samples, &poles, config)?;

        // Enforce stability if requested
        if config.enforce_stable {
            for p in &mut poles {
                if p.re > 0.0 {
                    *p = Complex::new(-p.re, p.im);
                }
            }
        }

        // Check convergence
        let rel_change = pole_change(&poles_prev, &poles);
        if rel_change < config.tol {
            let model = compute_residues(&s_points, samples, &poles, config)?;
            let rmse = compute_rmse(&s_points, samples, &model);
            return Ok(VFResult {
                model,
                iterations: iter + 1,
                converged: true,
                final_rmse: rmse,
            });
        }
    }

    // Max iterations reached without convergence
    let model = compute_residues(&s_points, samples, &poles, config)?;
    let rmse = compute_rmse(&s_points, samples, &model);
    Ok(VFResult {
        model,
        iterations: config.max_iter,
        converged: false,
        final_rmse: rmse,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test fitting a simple first-order lowpass: H(s) = 1 / (s + 1)
    #[test]
    fn test_fit_first_order_lowpass() {
        // Generate samples from H(s) = 1 / (s + 1)
        let freqs: Vec<Scalar> = (1..=50).map(|i| i as Scalar * 0.02).collect();
        let samples: Vec<CScalar> = freqs
            .iter()
            .map(|&f| {
                let s = Complex::new(0.0, 2.0 * PI * f);
                CScalar::new(1.0, 0.0) / (s + CScalar::new(1.0, 0.0))
            })
            .collect();

        let config = VFConfig {
            n_poles: 2,
            max_iter: 50,
            tol: 1e-8,
            enforce_stable: true,
            fit_e: false,
        };

        let result = fit(&freqs, &samples, &config).unwrap();

        // Should converge with good accuracy
        assert!(result.converged, "VF should converge");
        assert!(
            result.final_rmse < 1e-6,
            "RMSE should be small: {}",
            result.final_rmse
        );
        assert!(result.model.is_stable(), "Model should be stable");
    }

    /// Test fitting a second-order resonant system (RLC circuit).
    #[test]
    fn test_fit_resonant_system() {
        // H(s) = ω₀² / (s² + 2ζω₀s + ω₀²) with f₀=1kHz, ζ=0.1
        let f0 = 1000.0;
        let omega0 = 2.0 * PI * f0;
        let zeta = 0.1;

        let freqs: Vec<Scalar> = (1..=200).map(|i| i as Scalar * 10.0).collect();
        let samples: Vec<CScalar> = freqs
            .iter()
            .map(|&f| {
                let s = Complex::new(0.0, 2.0 * PI * f);
                let num = CScalar::new(omega0 * omega0, 0.0);
                let den = s * s
                    + s * CScalar::new(2.0 * zeta * omega0, 0.0)
                    + CScalar::new(omega0 * omega0, 0.0);
                num / den
            })
            .collect();

        let config = VFConfig {
            n_poles: 4,
            max_iter: 100,
            tol: 1e-8,
            enforce_stable: true,
            fit_e: false,
        };

        let result = fit(&freqs, &samples, &config).unwrap();

        assert!(result.converged, "VF should converge for resonant system");
        assert!(
            result.final_rmse < 1e-4,
            "RMSE should be small: {}",
            result.final_rmse
        );
        assert!(result.model.is_stable(), "Model should be stable");
    }

    /// Test that model evaluation matches original data.
    #[test]
    fn test_model_evaluation_accuracy() {
        // Simple pole at s = -10
        let freqs: Vec<Scalar> = (1..=100).map(|i| i as Scalar * 0.1).collect();
        let true_pole = CScalar::new(-10.0, 0.0);
        let samples: Vec<CScalar> = freqs
            .iter()
            .map(|&f| {
                let s = Complex::new(0.0, 2.0 * PI * f);
                CScalar::new(10.0, 0.0) / (s - true_pole)
            })
            .collect();

        let config = VFConfig {
            n_poles: 2,
            ..Default::default()
        };

        let result = fit(&freqs, &samples, &config).unwrap();

        // Check that model evaluation is close to samples
        for (&f, &h_true) in freqs.iter().zip(samples.iter()) {
            let h_fit = result.model.eval_hz(f);
            let error = (h_true - h_fit).norm() / h_true.norm().max(1e-10);
            assert!(error < 0.01, "Relative error too large at f={f}: {error}");
        }
    }

    /// Test input validation.
    #[test]
    fn test_input_validation() {
        let config = VFConfig::default();

        // Empty input
        assert!(matches!(fit(&[], &[], &config), Err(VFError::EmptyInput)));

        // Length mismatch
        let freqs = vec![1.0, 2.0];
        let samples = vec![CScalar::new(1.0, 0.0)];
        assert!(matches!(
            fit(&freqs, &samples, &config),
            Err(VFError::LengthMismatch { .. })
        ));

        // Insufficient samples
        let freqs: Vec<Scalar> = (1..=5).map(|i| i as Scalar).collect();
        let samples: Vec<CScalar> = freqs.iter().map(|_| CScalar::new(1.0, 0.0)).collect();
        let config = VFConfig {
            n_poles: 10,
            ..Default::default()
        };
        assert!(matches!(
            fit(&freqs, &samples, &config),
            Err(VFError::InsufficientSamples { .. })
        ));
    }

    /// Test stability enforcement.
    #[test]
    fn test_stability_enforcement() {
        let freqs: Vec<Scalar> = (1..=50).map(|i| i as Scalar * 0.1).collect();
        let samples: Vec<CScalar> = freqs
            .iter()
            .map(|&f| {
                let s = Complex::new(0.0, 2.0 * PI * f);
                CScalar::new(1.0, 0.0) / (s + CScalar::new(1.0, 0.0))
            })
            .collect();

        let config = VFConfig {
            n_poles: 2,
            enforce_stable: true,
            ..Default::default()
        };

        let result = fit(&freqs, &samples, &config).unwrap();
        assert!(
            result.model.is_stable(),
            "Model should be stable when enforce_stable=true"
        );
    }

    /// Test initial pole distribution.
    #[test]
    fn test_initial_poles() {
        let poles = initial_poles(1.0, 1000.0, 4);
        assert_eq!(poles.len(), 4);

        // All should have negative real parts
        for p in &poles {
            assert!(p.re < 0.0, "Initial poles should be stable");
        }

        // Should come in conjugate pairs
        let mut found_conjugate = false;
        for (i, p1) in poles.iter().enumerate() {
            for p2 in poles.iter().skip(i + 1) {
                if (p1.re - p2.re).abs() < 1e-10 && (p1.im + p2.im).abs() < 1e-10 {
                    found_conjugate = true;
                }
            }
        }
        assert!(found_conjugate, "Should have conjugate pairs");
    }
}
