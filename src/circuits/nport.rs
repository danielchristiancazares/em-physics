//! N-port network elements and Touchstone sNp import.
//!
//! Provides utilities to parse Touchstone files, stamp N-port admittance
//! models into MNA systems for AC analysis, and validate passivity.

use nalgebra::DMatrix;
use nalgebra::linalg::SymmetricEigen;
use num_complex::Complex;
use thiserror::Error;

use crate::math::Scalar;

use super::stamp::{AcContext, MnaBuilder, Node};

/// Default tolerance for passivity checks (accounts for numerical precision).
const PASSIVITY_TOLERANCE: Scalar = 1e-12;

/// Errors returned when constructing reference impedances.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ReferenceImpedanceError {
    /// Raised when an empty impedance list is provided.
    #[error("reference impedance list is empty")]
    Empty,
    /// Raised when a reference impedance value is not finite.
    #[error("reference impedance at index {index} is not finite: {value}")]
    NonFinite { index: usize, value: Scalar },
    /// Raised when a reference impedance value is non-positive.
    #[error("reference impedance at index {index} must be positive, got {value}")]
    NonPositive { index: usize, value: Scalar },
}

/// Per-port reference impedances for multiport conversions.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceImpedance {
    values: Vec<Scalar>,
}

impl ReferenceImpedance {
    /// Constructs per-port reference impedances from explicit values.
    pub fn new(values: Vec<Scalar>) -> Result<Self, ReferenceImpedanceError> {
        if values.is_empty() {
            return Err(ReferenceImpedanceError::Empty);
        }
        for (index, value) in values.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ReferenceImpedanceError::NonFinite { index, value });
            }
            if value <= 0.0 {
                return Err(ReferenceImpedanceError::NonPositive { index, value });
            }
        }
        Ok(Self { values })
    }

    /// Constructs a uniform reference impedance applied to all ports.
    pub fn uniform(z0: Scalar, ports: usize) -> Result<Self, ReferenceImpedanceError> {
        Self::new(vec![z0; ports])
    }

    /// Returns the number of ports covered by this reference impedance.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if there are no reference impedances.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the per-port reference impedances.
    #[must_use]
    pub fn values(&self) -> &[Scalar] {
        &self.values
    }

    fn sqrt_values(&self) -> Vec<Scalar> {
        self.values.iter().map(|value| value.sqrt()).collect()
    }

    fn inv_sqrt_values(&self) -> Vec<Scalar> {
        self.values.iter().map(|value| 1.0 / value.sqrt()).collect()
    }
}

/// Errors returned by N-port conversions and operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum NPortError {
    /// Raised when attempting to operate on an empty network.
    #[error("N-port network has no frequency data")]
    EmptyNetwork,
    /// Raised when port counts do not match expectations.
    #[error("port count mismatch: expected {expected}, got {found}")]
    PortCountMismatch { expected: usize, found: usize },
    /// Raised when a matrix is not square.
    #[error("matrix must be square, got {rows}x{cols}")]
    NonSquare { rows: usize, cols: usize },
    /// Raised when a matrix dimension does not match the port count.
    #[error("matrix size {rows}x{cols} does not match port count {expected}")]
    DimensionMismatch {
        expected: usize,
        rows: usize,
        cols: usize,
    },
    /// Raised when a matrix is singular or ill-conditioned.
    #[error("matrix is singular or ill-conditioned")]
    SingularMatrix,
    /// Raised when reference impedances are invalid.
    #[error(transparent)]
    ReferenceImpedance(#[from] ReferenceImpedanceError),
}

/// Result of passivity validation at a single frequency point.
///
/// A network is passive at a frequency if all singular values of its
/// S-parameter matrix are <= 1, meaning it cannot generate energy.
#[derive(Debug, Clone, PartialEq)]
pub struct PassivityResult {
    /// Frequency in Hz where this check was performed.
    pub frequency_hz: Scalar,
    /// True if the network is passive at this frequency.
    pub is_passive: bool,
    /// Maximum singular value of the S-parameter matrix.
    pub max_singular_value: Scalar,
    /// Passivity margin: 1.0 - max_singular_value.
    /// Positive means passive, negative means active (energy-generating).
    pub margin: Scalar,
}

/// Summary of passivity validation across all frequencies.
///
/// # References
///
/// - Gustavsen, B., & Semlyen, A. (1999). "Enforcing Passivity for Admittance
///   Matrices Approximated by Rational Functions". IEEE Trans. Power Systems.
#[derive(Debug, Clone)]
pub struct PassivityReport {
    /// True if the network is passive at all frequencies.
    pub all_passive: bool,
    /// List of frequency points where passivity is violated.
    pub violations: Vec<PassivityResult>,
    /// Worst (most negative) passivity margin across all frequencies.
    pub worst_margin: Scalar,
    /// Frequency (Hz) where the worst margin occurs, if any data exists.
    pub worst_frequency_hz: Option<Scalar>,
}

/// Result of passivity validation on the Hermitian part of Y or Z.
#[derive(Debug, Clone, PartialEq)]
pub struct HermitianPassivityResult {
    /// Frequency in Hz where this check was performed.
    pub frequency_hz: Scalar,
    /// True if the network is passive at this frequency.
    pub is_passive: bool,
    /// Minimum eigenvalue of the Hermitian part.
    pub min_eigenvalue: Scalar,
    /// Passivity margin: min_eigenvalue (>= 0 means passive).
    pub margin: Scalar,
}

/// Summary of Hermitian passivity validation across all frequencies.
#[derive(Debug, Clone)]
pub struct HermitianPassivityReport {
    /// True if the network is passive at all frequencies.
    pub all_passive: bool,
    /// List of frequency points where passivity is violated.
    pub violations: Vec<HermitianPassivityResult>,
    /// Worst (most negative) passivity margin across all frequencies.
    pub worst_margin: Scalar,
    /// Frequency (Hz) where the worst margin occurs, if any data exists.
    pub worst_frequency_hz: Option<Scalar>,
}

/// Representation of an N-port network as frequency-dependent S-parameters.
#[derive(Debug, Clone)]
pub struct NPortNetwork {
    pub port_count: usize,
    /// Reference impedance per port (real, power-wave normalization).
    pub z0: ReferenceImpedance,
    /// Frequencies in Hz.
    pub frequencies: Vec<Scalar>,
    /// For each frequency index f, an (n x n) matrix of S-parameters.
    pub sparams: Vec<DMatrix<Complex<Scalar>>>,
}

impl NPortNetwork {
    /// Returns the S-parameter matrix interpolated (nearest neighbor) at `freq_hz`.
    pub fn s_at(&self, freq_hz: Scalar) -> Result<&DMatrix<Complex<Scalar>>, NPortError> {
        if self.frequencies.is_empty() {
            return Err(NPortError::EmptyNetwork);
        }
        // Nearest neighbor selection
        let mut best_idx = 0usize;
        let mut best_err = (self.frequencies[0] - freq_hz).abs();
        for (i, f) in self.frequencies.iter().enumerate() {
            let err = (f - freq_hz).abs();
            if err < best_err {
                best_err = err;
                best_idx = i;
            }
        }
        Ok(&self.sparams[best_idx])
    }

    /// Converts S-parameters to Y-parameters using power-wave normalization.
    ///
    /// Y = K^{-1} * (I - S) * (I + S)^{-1} * K^{-1}, K = diag(sqrt(z0)).
    pub fn s_to_y(
        &self,
        s: &DMatrix<Complex<Scalar>>,
    ) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        self.ensure_matrix(s)?;
        let n = self.port_count;
        let i = DMatrix::<Complex<Scalar>>::identity(n, n);
        let denom = &i + s;
        let inv = Self::invert(&denom)?;
        let middle = (&i - s) * inv;
        let inv_sqrt = self.z0.inv_sqrt_values();
        Self::scale_rows_cols(&middle, &inv_sqrt, &inv_sqrt)
    }

    /// Converts S-parameters to Z-parameters using power-wave normalization.
    ///
    /// Z = K * (I + S) * (I - S)^{-1} * K, K = diag(sqrt(z0)).
    pub fn s_to_z(
        &self,
        s: &DMatrix<Complex<Scalar>>,
    ) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        self.ensure_matrix(s)?;
        let n = self.port_count;
        let i = DMatrix::<Complex<Scalar>>::identity(n, n);
        let denom = &i - s;
        let inv = Self::invert(&denom)?;
        let middle = (&i + s) * inv;
        let sqrt = self.z0.sqrt_values();
        Self::scale_rows_cols(&middle, &sqrt, &sqrt)
    }

    /// Converts Z-parameters to S-parameters using power-wave normalization.
    pub fn z_to_s(
        &self,
        z: &DMatrix<Complex<Scalar>>,
    ) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        self.ensure_matrix(z)?;
        let n = self.port_count;
        let inv_sqrt = self.z0.inv_sqrt_values();
        let normalized = Self::scale_rows_cols(z, &inv_sqrt, &inv_sqrt)?;
        let i = DMatrix::<Complex<Scalar>>::identity(n, n);
        let denom = &normalized + &i;
        let inv = Self::invert(&denom)?;
        let num = &normalized - &i;
        Ok(num * inv)
    }

    /// Converts Y-parameters to S-parameters using power-wave normalization.
    pub fn y_to_s(
        &self,
        y: &DMatrix<Complex<Scalar>>,
    ) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        self.ensure_matrix(y)?;
        let n = self.port_count;
        let sqrt = self.z0.sqrt_values();
        let normalized = Self::scale_rows_cols(y, &sqrt, &sqrt)?;
        let i = DMatrix::<Complex<Scalar>>::identity(n, n);
        let denom = &normalized + &i;
        let inv = Self::invert(&denom)?;
        let num = &i - &normalized;
        Ok(num * inv)
    }

    /// Converts Z-parameters to Y-parameters.
    pub fn z_to_y(
        &self,
        z: &DMatrix<Complex<Scalar>>,
    ) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        self.ensure_matrix(z)?;
        Self::invert(z)
    }

    /// Converts Y-parameters to Z-parameters.
    pub fn y_to_z(
        &self,
        y: &DMatrix<Complex<Scalar>>,
    ) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        self.ensure_matrix(y)?;
        Self::invert(y)
    }

    /// Returns the Y-parameter matrix interpolated (nearest neighbor) at `freq_hz`.
    pub fn y_at(&self, freq_hz: Scalar) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        let s = self.s_at(freq_hz)?;
        self.s_to_y(s)
    }

    /// Returns the Z-parameter matrix interpolated (nearest neighbor) at `freq_hz`.
    pub fn z_at(&self, freq_hz: Scalar) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        let s = self.s_at(freq_hz)?;
        self.s_to_z(s)
    }

    fn ensure_matrix(&self, matrix: &DMatrix<Complex<Scalar>>) -> Result<(), NPortError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(NPortError::NonSquare {
                rows: matrix.nrows(),
                cols: matrix.ncols(),
            });
        }
        if matrix.nrows() != self.port_count {
            return Err(NPortError::DimensionMismatch {
                expected: self.port_count,
                rows: matrix.nrows(),
                cols: matrix.ncols(),
            });
        }
        if self.z0.len() != self.port_count {
            return Err(NPortError::PortCountMismatch {
                expected: self.port_count,
                found: self.z0.len(),
            });
        }
        Ok(())
    }

    fn scale_rows_cols(
        matrix: &DMatrix<Complex<Scalar>>,
        left: &[Scalar],
        right: &[Scalar],
    ) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        if matrix.nrows() != left.len() || matrix.ncols() != right.len() {
            return Err(NPortError::DimensionMismatch {
                expected: matrix.nrows(),
                rows: left.len(),
                cols: right.len(),
            });
        }
        let mut out = matrix.clone();
        for i in 0..out.nrows() {
            let row_scale = left[i];
            for j in 0..out.ncols() {
                out[(i, j)] *= Complex::new(row_scale * right[j], 0.0);
            }
        }
        Ok(out)
    }

    fn invert(matrix: &DMatrix<Complex<Scalar>>) -> Result<DMatrix<Complex<Scalar>>, NPortError> {
        matrix.clone().try_inverse().ok_or(NPortError::SingularMatrix)
    }

    /// Check passivity at a single frequency index.
    ///
    /// Uses SVD to compute singular values of the S-parameter matrix. A network is
    /// passive if all singular values are <= 1 (within numerical tolerance).
    ///
    /// # Panics
    ///
    /// Panics if `freq_idx` is out of bounds.
    #[must_use]
    pub fn check_passivity_at(&self, freq_idx: usize) -> PassivityResult {
        let s = &self.sparams[freq_idx];
        let freq_hz = self.frequencies[freq_idx];

        // Compute singular values via SVD
        // For complex matrices, singular values are the square roots of eigenvalues of S^H * S
        let svd = s.clone().svd(false, false);
        let max_sv = svd.singular_values.iter().cloned().fold(0.0_f64, f64::max);
        let margin = 1.0 - max_sv;

        PassivityResult {
            frequency_hz: freq_hz,
            is_passive: max_sv <= 1.0 + PASSIVITY_TOLERANCE,
            max_singular_value: max_sv,
            margin,
        }
    }

    /// Check passivity across all frequency points.
    ///
    /// Returns a comprehensive report including all violations and the worst-case margin.
    #[must_use]
    pub fn check_passivity(&self) -> PassivityReport {
        if self.frequencies.is_empty() {
            return PassivityReport {
                all_passive: true,
                violations: Vec::new(),
                worst_margin: Scalar::INFINITY,
                worst_frequency_hz: None,
            };
        }

        let mut violations = Vec::new();
        let mut worst_margin = Scalar::INFINITY;
        let mut worst_freq = None;

        for idx in 0..self.frequencies.len() {
            let result = self.check_passivity_at(idx);

            if result.margin < worst_margin {
                worst_margin = result.margin;
                worst_freq = Some(result.frequency_hz);
            }

            if !result.is_passive {
                violations.push(result);
            }
        }

        PassivityReport {
            all_passive: violations.is_empty(),
            violations,
            worst_margin,
            worst_frequency_hz: worst_freq,
        }
    }

    /// Check passivity from the Hermitian part of Y-parameters at a single frequency index.
    ///
    /// # Panics
    ///
    /// Panics if `freq_idx` is out of bounds.
    pub fn check_passivity_y_at(
        &self,
        freq_idx: usize,
    ) -> Result<HermitianPassivityResult, NPortError> {
        let s = &self.sparams[freq_idx];
        let freq_hz = self.frequencies[freq_idx];
        let y = self.s_to_y(s)?;
        let min_eigenvalue = Self::hermitian_min_eigenvalue(&y);
        let margin = min_eigenvalue;
        Ok(HermitianPassivityResult {
            frequency_hz: freq_hz,
            is_passive: min_eigenvalue >= -PASSIVITY_TOLERANCE,
            min_eigenvalue,
            margin,
        })
    }

    /// Check passivity from the Hermitian part of Z-parameters at a single frequency index.
    ///
    /// # Panics
    ///
    /// Panics if `freq_idx` is out of bounds.
    pub fn check_passivity_z_at(
        &self,
        freq_idx: usize,
    ) -> Result<HermitianPassivityResult, NPortError> {
        let s = &self.sparams[freq_idx];
        let freq_hz = self.frequencies[freq_idx];
        let z = self.s_to_z(s)?;
        let min_eigenvalue = Self::hermitian_min_eigenvalue(&z);
        let margin = min_eigenvalue;
        Ok(HermitianPassivityResult {
            frequency_hz: freq_hz,
            is_passive: min_eigenvalue >= -PASSIVITY_TOLERANCE,
            min_eigenvalue,
            margin,
        })
    }

    /// Check passivity from the Hermitian part of Y-parameters across all frequencies.
    pub fn check_passivity_y(&self) -> Result<HermitianPassivityReport, NPortError> {
        if self.frequencies.is_empty() {
            return Ok(HermitianPassivityReport {
                all_passive: true,
                violations: Vec::new(),
                worst_margin: Scalar::INFINITY,
                worst_frequency_hz: None,
            });
        }

        let mut violations = Vec::new();
        let mut worst_margin = Scalar::INFINITY;
        let mut worst_freq = None;

        for idx in 0..self.frequencies.len() {
            let result = self.check_passivity_y_at(idx)?;

            if result.margin < worst_margin {
                worst_margin = result.margin;
                worst_freq = Some(result.frequency_hz);
            }

            if !result.is_passive {
                violations.push(result);
            }
        }

        Ok(HermitianPassivityReport {
            all_passive: violations.is_empty(),
            violations,
            worst_margin,
            worst_frequency_hz: worst_freq,
        })
    }

    /// Check passivity from the Hermitian part of Z-parameters across all frequencies.
    pub fn check_passivity_z(&self) -> Result<HermitianPassivityReport, NPortError> {
        if self.frequencies.is_empty() {
            return Ok(HermitianPassivityReport {
                all_passive: true,
                violations: Vec::new(),
                worst_margin: Scalar::INFINITY,
                worst_frequency_hz: None,
            });
        }

        let mut violations = Vec::new();
        let mut worst_margin = Scalar::INFINITY;
        let mut worst_freq = None;

        for idx in 0..self.frequencies.len() {
            let result = self.check_passivity_z_at(idx)?;

            if result.margin < worst_margin {
                worst_margin = result.margin;
                worst_freq = Some(result.frequency_hz);
            }

            if !result.is_passive {
                violations.push(result);
            }
        }

        Ok(HermitianPassivityReport {
            all_passive: violations.is_empty(),
            violations,
            worst_margin,
            worst_frequency_hz: worst_freq,
        })
    }

    /// Returns true if the network is passive at all frequencies.
    ///
    /// This is a convenience method equivalent to `check_passivity().all_passive`.
    #[must_use]
    pub fn is_passive(&self) -> bool {
        self.frequencies
            .iter()
            .enumerate()
            .all(|(idx, _)| self.check_passivity_at(idx).is_passive)
    }

    fn hermitian_min_eigenvalue(matrix: &DMatrix<Complex<Scalar>>) -> Scalar {
        let hermitian = (matrix.clone() + matrix.adjoint()) * Complex::new(0.5, 0.0);
        let eigen = SymmetricEigen::new(hermitian);
        let mut min_value = Scalar::INFINITY;
        for value in eigen.eigenvalues.iter() {
            if *value < min_value {
                min_value = *value;
            }
        }
        min_value
    }

    /// Stamps the N-port as a frequency-domain port admittance matrix between `ports`.
    ///
    /// Each port i is described by a pair of nodes (p_i, n_i) for positive and negative terminals.
    ///
    /// For Y-port matrix Y (n x n), contributions are applied for each Y_ij as a 2x2 block
    /// between the node pairs of port i and port j.
    pub fn stamp_into_mna(
        &self,
        ctx: AcContext,
        ports: &[(Node, Node)],
        mna: &mut MnaBuilder,
    ) -> Result<(), NPortError> {
        if ports.len() != self.port_count {
            return Err(NPortError::PortCountMismatch {
                expected: self.port_count,
                found: ports.len(),
            });
        }
        if ctx.omega == 0.0 {
            return Ok(());
        }

        let freq_hz = ctx.omega / (2.0 * std::f64::consts::PI);
        let s = self.s_at(freq_hz)?;
        let y = self.s_to_y(s)?;

        // For each Y_ij, stamp 4 entries coupling port voltages V_i and V_j.
        // V_i = V(p_i) - V(n_i). Current defined into the network.
        for i in 0..self.port_count {
            let (pi, ni) = ports[i];
            let pi_idx = pi;
            let ni_idx = ni;
            for j in 0..self.port_count {
                let (pj, nj) = ports[j];
                let yij = y[(i, j)];
                if yij == Complex::new(0.0, 0.0) {
                    continue;
                }

                // Stamp 2x2 block:
                // [ +yij  -yij; -yij  +yij ] between (pi,ni) rows and (pj,nj) cols
                if let Some(pi_) = pi_idx {
                    if let Some(pj_) = pj {
                        mna.add_matrix_entry(pi_, pj_, yij);
                    }
                }
                if let Some(pi_) = pi_idx {
                    if let Some(nj_) = nj {
                        mna.add_matrix_entry(pi_, nj_, -yij);
                    }
                }
                if let Some(ni_) = ni_idx {
                    if let Some(pj_) = pj {
                        mna.add_matrix_entry(ni_, pj_, -yij);
                    }
                }
                if let Some(ni_) = ni_idx {
                    if let Some(nj_) = nj {
                        mna.add_matrix_entry(ni_, nj_, yij);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Minimal Touchstone sNp parser for common cases: `# Hz S RI R 50` or `# Hz S MA R 50`.
pub fn read_touchstone(contents: &str) -> Result<NPortNetwork, String> {
    let mut z0 = 50.0;
    let mut format = String::from("RI"); // RI, MA, or DB
    let mut freq_unit = String::from("Hz");
    let mut data: Vec<(Scalar, Vec<Complex<Scalar>>)> = Vec::new();
    let mut nports: Option<usize> = None;

    for line in contents.lines() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('!') {
            continue;
        }
        if l.starts_with('#') {
            // Example: # Hz S RI R 50
            let tokens: Vec<_> = l[1..].split_whitespace().collect();
            if tokens.len() >= 2 {
                freq_unit = tokens[0].to_string();
            }
            if tokens.len() >= 2 { /* tokens[1] should be 'S' */ }
            if tokens.len() >= 3 {
                format = tokens[2].to_string();
            }
            if let Some(r_pos) = tokens.iter().position(|t| *t == "R") {
                if r_pos + 1 < tokens.len() {
                    if let Ok(v) = tokens[r_pos + 1].parse::<Scalar>() {
                        z0 = v;
                    }
                }
            }
            continue;
        }

        // Data line: f  S11  S21  S12  S22 ... order is standard Touchstone row-wise by port pairs
        let toks: Vec<&str> = l.split_whitespace().collect();
        if toks.is_empty() {
            continue;
        }
        let f_parsed: Scalar = toks[0].parse().map_err(|_| "invalid frequency")?;
        let f_hz = match freq_unit.as_str() {
            "Hz" => f_parsed,
            "kHz" => f_parsed * 1e3,
            "MHz" => f_parsed * 1e6,
            "GHz" => f_parsed * 1e9,
            _ => f_parsed,
        };

        // Infer n from number of remaining tokens: per S-parameter we have 2 numbers
        let param_tokens = &toks[1..];
        if param_tokens.is_empty() {
            continue;
        }
        let pairs = param_tokens.len() / 2;
        let n = (pairs as f64).sqrt() as usize;
        if n * n * 2 != param_tokens.len() {
            return Err("malformed sNp row".into());
        }
        if nports.is_none() {
            nports = Some(n);
        }
        if Some(n) != nports {
            return Err("inconsistent port count across rows".into());
        }

        let mut vals: Vec<Complex<Scalar>> = Vec::with_capacity(n * n);
        for k in 0..pairs {
            let a: Scalar = param_tokens[2 * k]
                .parse()
                .map_err(|_| "invalid parameter")?;
            let b: Scalar = param_tokens[2 * k + 1]
                .parse()
                .map_err(|_| "invalid parameter")?;
            let c = match format.as_str() {
                "RI" => Complex::new(a, b),
                "MA" => {
                    // magnitude, angle (degrees)
                    let mag = a;
                    let ang = b.to_radians();
                    Complex::from_polar(mag, ang)
                }
                "DB" => {
                    // dB, angle (degrees)
                    let mag = 10f64.powf(a / 20.0);
                    let ang = b.to_radians();
                    Complex::from_polar(mag, ang)
                }
                _ => return Err("unsupported Touchstone format".into()),
            };
            vals.push(c);
        }
        data.push((f_hz, vals));
    }

    let n = nports.ok_or_else(|| "missing data".to_string())?;
    let mut freqs = Vec::with_capacity(data.len());
    let mut mats = Vec::with_capacity(data.len());
    for (f, vals) in data.into_iter() {
        freqs.push(f);
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = vals[i * n + j];
            }
        }
        mats.push(m);
    }

    let z0 = ReferenceImpedance::uniform(z0, n).map_err(|err| err.to_string())?;
    Ok(NPortNetwork {
        port_count: n,
        z0,
        frequencies: freqs,
        sparams: mats,
    })
}

/// Parse a Touchstone file and validate passivity.
///
/// Returns both the parsed network and a passivity report. This is useful when
/// importing S-parameter data that should represent a passive device (e.g., a
/// connector, filter, or transmission line).
///
/// # Errors
///
/// Returns an error if the Touchstone file is malformed.
pub fn read_touchstone_checked(contents: &str) -> Result<(NPortNetwork, PassivityReport), String> {
    let network = read_touchstone(contents)?;
    let report = network.check_passivity();
    Ok((network, report))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn uniform_z0(ports: usize) -> ReferenceImpedance {
        ReferenceImpedance::uniform(50.0, ports).unwrap()
    }

    /// Create a matched load (perfect absorber): S = 0 matrix.
    /// This is maximally passive with margin = 1.0.
    #[test]
    fn test_passivity_matched_load() {
        let network = NPortNetwork {
            port_count: 2,
            z0: uniform_z0(2),
            frequencies: vec![1e9],
            sparams: vec![DMatrix::zeros(2, 2)],
        };

        let result = network.check_passivity_at(0);
        assert!(result.is_passive);
        assert!((result.max_singular_value - 0.0).abs() < 1e-10);
        assert!((result.margin - 1.0).abs() < 1e-10);
        assert!(network.is_passive());
    }

    /// Create a lossless network (unitary S-matrix): S^H * S = I.
    /// Singular values are all 1.0, margin = 0.0.
    #[test]
    fn test_passivity_lossless() {
        // Simple lossless 2-port: ideal coupler-like behavior
        // S = [[0, 1], [1, 0]] is unitary (swap ports)
        let mut s = DMatrix::zeros(2, 2);
        s[(0, 1)] = Complex::new(1.0, 0.0);
        s[(1, 0)] = Complex::new(1.0, 0.0);

        let network = NPortNetwork {
            port_count: 2,
            z0: uniform_z0(2),
            frequencies: vec![1e9],
            sparams: vec![s],
        };

        let result = network.check_passivity_at(0);
        assert!(result.is_passive);
        assert!((result.max_singular_value - 1.0).abs() < 1e-10);
        assert!(result.margin.abs() < 1e-10);
    }

    /// Create an active (non-passive) network with gain.
    /// S has singular value > 1, should fail passivity check.
    #[test]
    fn test_passivity_active_network() {
        // S = [[2, 0], [0, 0]] has singular value 2.0 (amplifier-like)
        let mut s = DMatrix::zeros(2, 2);
        s[(0, 0)] = Complex::new(2.0, 0.0);

        let network = NPortNetwork {
            port_count: 2,
            z0: uniform_z0(2),
            frequencies: vec![1e9],
            sparams: vec![s],
        };

        let result = network.check_passivity_at(0);
        assert!(!result.is_passive);
        assert!((result.max_singular_value - 2.0).abs() < 1e-10);
        assert!((result.margin - (-1.0)).abs() < 1e-10);
        assert!(!network.is_passive());
    }

    /// Test passivity report across multiple frequencies with one violation.
    #[test]
    fn test_passivity_report_mixed() {
        let passive_s = DMatrix::zeros(2, 2);
        let mut active_s = DMatrix::zeros(2, 2);
        active_s[(0, 0)] = Complex::new(1.5, 0.0);

        let network = NPortNetwork {
            port_count: 2,
            z0: uniform_z0(2),
            frequencies: vec![1e9, 2e9, 3e9],
            sparams: vec![passive_s.clone(), active_s, passive_s],
        };

        let report = network.check_passivity();
        assert!(!report.all_passive);
        assert_eq!(report.violations.len(), 1);
        assert!((report.violations[0].frequency_hz - 2e9).abs() < 1e-6);
        assert!((report.worst_margin - (-0.5)).abs() < 1e-10);
        assert!((report.worst_frequency_hz.unwrap() - 2e9).abs() < 1e-6);
    }

    /// Test Touchstone parsing with passivity check.
    #[test]
    fn test_read_touchstone_checked() {
        let touchstone = r#"
! 2-port matched load
# GHz S RI R 50
1.0  0.0 0.0  0.0 0.0  0.0 0.0  0.0 0.0
2.0  0.0 0.0  0.0 0.0  0.0 0.0  0.0 0.0
"#;
        let (network, report) = read_touchstone_checked(touchstone).unwrap();
        assert_eq!(network.port_count, 2);
        assert_eq!(network.frequencies.len(), 2);
        assert!(report.all_passive);
        assert!(report.violations.is_empty());
    }

    #[test]
    fn test_s_to_z_and_y_for_matched_load() {
        let z0 = ReferenceImpedance::new(vec![50.0, 75.0]).unwrap();
        let network = NPortNetwork {
            port_count: 2,
            z0: z0.clone(),
            frequencies: vec![1e9],
            sparams: vec![DMatrix::zeros(2, 2)],
        };
        let s = &network.sparams[0];
        let z = network.s_to_z(s).unwrap();
        assert_relative_eq!(z[(0, 0)].re, 50.0, epsilon = 1e-12);
        assert_relative_eq!(z[(1, 1)].re, 75.0, epsilon = 1e-12);
        assert_relative_eq!(z[(0, 1)].norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(z[(1, 0)].norm(), 0.0, epsilon = 1e-12);

        let y = network.s_to_y(s).unwrap();
        assert_relative_eq!(y[(0, 0)].re, 1.0 / 50.0, epsilon = 1e-12);
        assert_relative_eq!(y[(1, 1)].re, 1.0 / 75.0, epsilon = 1e-12);
        assert_relative_eq!(y[(0, 1)].norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(y[(1, 0)].norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_z_to_s_roundtrip_for_reference_diagonal() {
        let z0 = ReferenceImpedance::new(vec![50.0, 75.0]).unwrap();
        let network = NPortNetwork {
            port_count: 2,
            z0,
            frequencies: vec![1e9],
            sparams: vec![DMatrix::zeros(2, 2)],
        };
        let mut z = DMatrix::zeros(2, 2);
        z[(0, 0)] = Complex::new(50.0, 0.0);
        z[(1, 1)] = Complex::new(75.0, 0.0);
        let s = network.z_to_s(&z).unwrap();
        assert_relative_eq!(s[(0, 0)].norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(s[(1, 1)].norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(s[(0, 1)].norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(s[(1, 0)].norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_passivity_y_for_matched_load() {
        let network = NPortNetwork {
            port_count: 2,
            z0: uniform_z0(2),
            frequencies: vec![1e9],
            sparams: vec![DMatrix::zeros(2, 2)],
        };
        let report = network.check_passivity_y().unwrap();
        assert!(report.all_passive);
        assert!(report.violations.is_empty());
    }

    /// Test empty network edge case.
    #[test]
    fn test_passivity_empty_network() {
        let network = NPortNetwork {
            port_count: 2,
            z0: uniform_z0(2),
            frequencies: vec![],
            sparams: vec![],
        };

        let report = network.check_passivity();
        assert!(report.all_passive);
        assert!(report.violations.is_empty());
        assert!(report.worst_frequency_hz.is_none());
    }
}
