//! Sparse linear system solvers for circuit simulation.
//!
//! This module provides trait-based abstractions for sparse direct and iterative
//! solvers, specifically designed for Modified Nodal Analysis (MNA) matrices
//! arising in circuit simulation.
//!
//! # Solver Types
//!
//! - **Direct Solvers**: LU factorization-based methods (baseline, future KLU/UMFPACK)
//! - **Iterative Solvers**: Krylov subspace methods (BiCGSTAB, GMRES) for very large systems
//!
//! # References
//!
//! - Davis & Palamadai Natarajan (2010). "Algorithm 907: KLU, A Direct Sparse Solver
//!   for Circuit Simulation Problems". ACM TOMS 37(3).
//! - Saad & Schultz (1986). "GMRES: A Generalized Minimal Residual Algorithm for
//!   Solving Nonsymmetric Linear Systems". SIAM J. Sci. Stat. Comput. 7(3), 856-869.
//! - van der Vorst (2003). "Iterative Krylov Methods for Large Linear Systems".
//!   Cambridge University Press.

#![cfg(feature = "sparse")]

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CscMatrix;
use num_complex::Complex;
use std::time::Duration;

use crate::math::Scalar;

/// Error types for sparse solvers.
#[derive(Debug, Clone)]
pub enum SolverError {
    /// Matrix is singular or numerically singular (det ≈ 0).
    SingularMatrix,
    /// Iterative solver failed to converge within maximum iterations.
    ConvergenceFailure {
        /// Number of iterations completed before failure.
        iterations: usize,
        /// Final residual norm at termination.
        residual_norm: Scalar,
    },
    /// Matrix structure is invalid or inconsistent.
    InvalidMatrix(String),
    /// Numerical instability detected during factorization/iteration.
    NumericalInstability(String),
    /// Memory allocation failure or resource exhaustion.
    ResourceExhaustion(String),
    /// Other solver-specific errors.
    Other(String),
}

impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingularMatrix => write!(f, "Matrix is singular"),
            Self::ConvergenceFailure { iterations, residual_norm } => {
                write!(f, "Failed to converge after {} iterations (residual: {:.2e})",
                       iterations, residual_norm)
            }
            Self::InvalidMatrix(msg) => write!(f, "Invalid matrix: {}", msg),
            Self::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
            Self::ResourceExhaustion(msg) => write!(f, "Resource exhaustion: {}", msg),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SolverError {}

/// Statistics and diagnostics from a sparse solver execution.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// True if the solve succeeded.
    pub success: bool,
    /// Number of nonzeros in the system matrix.
    pub nnz_matrix: usize,
    /// Number of nonzeros after factorization (fill-in for direct solvers).
    pub nnz_factor: Option<usize>,
    /// Condition number estimate (if available).
    pub condition_estimate: Option<Scalar>,
    /// Time spent in symbolic analysis/preprocessing.
    pub symbolic_time: Option<Duration>,
    /// Time spent in numerical factorization or iteration.
    pub numeric_time: Option<Duration>,
    /// Time spent in solution phase.
    pub solve_time: Option<Duration>,
    /// Number of iterations (for iterative solvers).
    pub iterations: Option<usize>,
    /// Final residual norm (for iterative solvers).
    pub residual_norm: Option<Scalar>,
    /// Memory used in bytes (estimate).
    pub memory_bytes: Option<usize>,
    /// Human-readable solver-specific notes.
    pub notes: Vec<String>,
}

impl SolverStats {
    /// Creates a failure stats object with an error message.
    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            notes: vec![message.into()],
            ..Default::default()
        }
    }
}

/// Trait for sparse linear system solvers: solve Ax = b for sparse A.
///
/// This trait provides a unified interface for both direct solvers (LU factorization)
/// and iterative solvers (Krylov methods), allowing algorithm selection at runtime
/// based on problem characteristics.
///
/// # Type Parameters
///
/// Solvers operate on complex-valued systems for AC circuit analysis.
///
/// # Solver Pattern
///
/// 1. **Symbolic Phase** (direct solvers): Analyze matrix structure, compute ordering
/// 2. **Numeric Phase**: Factor or iterate to solve the system
/// 3. **Solve Phase**: Apply factors or solution to RHS vector(s)
///
/// For repeated solves with the same structure (e.g., AC sweeps), implementations
/// should cache symbolic factorization data.
pub trait SparseSolver {
    /// Analyzes the matrix structure and prepares for factorization.
    ///
    /// This phase computes fill-reducing orderings (AMD, COLAMD, RCM) and
    /// symbolic factorization. The result can be reused for matrices with
    /// the same sparsity pattern but different values.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The system matrix A (only pattern is analyzed, values ignored)
    ///
    /// # Returns
    ///
    /// Success or error indicating structural issues.
    fn symbolic(&mut self, matrix: &CscMatrix<Complex<Scalar>>) -> Result<(), SolverError>;

    /// Performs numerical factorization or prepares the iterative solver.
    ///
    /// For direct solvers, this computes LU factors using the symbolic analysis
    /// from the previous step. For iterative solvers, this may compute
    /// preconditioners (ILU, Jacobi).
    ///
    /// # Arguments
    ///
    /// * `matrix` - The system matrix A (structure must match symbolic phase)
    ///
    /// # Returns
    ///
    /// Success or error (singularity, numerical breakdown, etc.)
    fn numeric(&mut self, matrix: &CscMatrix<Complex<Scalar>>) -> Result<(), SolverError>;

    /// Solves Ax = b using the factorization or iterative method.
    ///
    /// The matrix must have been previously analyzed (symbolic) and factored/prepared
    /// (numeric) before calling this method.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side vector b
    ///
    /// # Returns
    ///
    /// Solution vector x, or error if solve fails
    fn solve(&self, rhs: &DVector<Complex<Scalar>>) -> Result<DVector<Complex<Scalar>>, SolverError>;

    /// Solves the system and returns detailed statistics.
    ///
    /// This is the recommended interface for production use, providing diagnostics
    /// useful for performance analysis and debugging convergence issues.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Right-hand side vector b
    ///
    /// # Returns
    ///
    /// Tuple of (solution, statistics) or error
    fn solve_with_stats(
        &self,
        rhs: &DVector<Complex<Scalar>>,
    ) -> Result<(DVector<Complex<Scalar>>, SolverStats), SolverError> {
        let start = std::time::Instant::now();
        let solution = self.solve(rhs)?;
        let solve_time = start.elapsed();

        let stats = SolverStats {
            success: true,
            solve_time: Some(solve_time),
            ..Default::default()
        };

        Ok((solution, stats))
    }

    /// Returns solver name for logging/debugging.
    fn name(&self) -> &str;

    /// Returns true if this solver has been factored/prepared and is ready to solve.
    fn is_ready(&self) -> bool;
}

/// Baseline sparse solver using dense conversion and nalgebra LU.
///
/// This is a reference implementation that validates the solver infrastructure.
/// It converts the sparse matrix to dense, uses nalgebra's LU factorization,
/// and solves the system. **Not suitable for large systems** due to O(n²) memory
/// and O(n³) factorization cost.
///
/// # Usage
///
/// ```ignore
/// use em_physics::circuits::solver::{SparseSolver, BaselineLuSolver};
///
/// let mut solver = BaselineLuSolver::new();
/// solver.symbolic(&matrix)?;
/// solver.numeric(&matrix)?;
/// let solution = solver.solve(&rhs)?;
/// ```
///
/// # When to Use
///
/// - Testing and validation against dense solutions
/// - Small systems (<1000 nodes) where sparse overhead isn't justified
/// - Prototyping before implementing optimized solvers
///
/// For large systems (>10k nodes), use iterative solvers (BiCGSTAB, GMRES).
pub struct BaselineLuSolver {
    /// Cached dense LU factorization.
    lu: Option<nalgebra::LU<Complex<Scalar>, nalgebra::Dyn, nalgebra::Dyn>>,
    /// Matrix dimensions.
    dimension: Option<usize>,
    /// Number of nonzeros in original matrix.
    nnz: Option<usize>,
}

impl BaselineLuSolver {
    /// Creates a new baseline LU solver.
    pub fn new() -> Self {
        Self {
            lu: None,
            dimension: None,
            nnz: None,
        }
    }

    /// Converts CSC sparse matrix to dense for factorization.
    fn csc_to_dense(csc: &CscMatrix<Complex<Scalar>>) -> DMatrix<Complex<Scalar>> {
        let n = csc.nrows();
        let m = csc.ncols();
        let mut dense = DMatrix::zeros(n, m);

        for (row, col, &value) in csc.triplet_iter() {
            dense[(row, col)] = value;
        }
        dense
    }
}

impl Default for BaselineLuSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSolver for BaselineLuSolver {
    fn symbolic(&mut self, matrix: &CscMatrix<Complex<Scalar>>) -> Result<(), SolverError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(SolverError::InvalidMatrix(
                format!("Matrix must be square: {}x{}", matrix.nrows(), matrix.ncols())
            ));
        }

        self.dimension = Some(matrix.nrows());
        self.nnz = Some(matrix.triplet_iter().count());
        self.lu = None; // Clear any previous factorization

        Ok(())
    }

    fn numeric(&mut self, matrix: &CscMatrix<Complex<Scalar>>) -> Result<(), SolverError> {
        let dim = self.dimension.ok_or_else(|| {
            SolverError::Other("Must call symbolic() before numeric()".into())
        })?;

        if matrix.nrows() != dim {
            return Err(SolverError::InvalidMatrix(
                "Matrix dimensions changed since symbolic phase".into()
            ));
        }

        // Convert to dense and factor
        let dense = Self::csc_to_dense(matrix);
        let lu = dense.lu();

        // Check for singularity using diagonal of U
        let u = lu.u();
        let min_diag = (0..dim)
            .map(|i| u[(i, i)].norm())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        if min_diag < 1e-14 {
            return Err(SolverError::SingularMatrix);
        }

        self.lu = Some(lu);
        Ok(())
    }

    fn solve(&self, rhs: &DVector<Complex<Scalar>>) -> Result<DVector<Complex<Scalar>>, SolverError> {
        let lu = self.lu.as_ref().ok_or_else(|| {
            SolverError::Other("Must call numeric() before solve()".into())
        })?;

        lu.solve(rhs).ok_or(SolverError::SingularMatrix)
    }

    fn solve_with_stats(
        &self,
        rhs: &DVector<Complex<Scalar>>,
    ) -> Result<(DVector<Complex<Scalar>>, SolverStats), SolverError> {
        let start = std::time::Instant::now();
        let solution = self.solve(rhs)?;
        let solve_time = start.elapsed();

        let lu = self.lu.as_ref().unwrap();
        let u = lu.u();
        let dim = u.nrows().min(u.ncols());

        // Condition estimate from U diagonal ratio
        let mut max_diag = 0.0;
        let mut min_diag = f64::INFINITY;
        for k in 0..dim {
            let d = u[(k, k)].norm();
            if d > max_diag {
                max_diag = d;
            }
            if d < min_diag {
                min_diag = d;
            }
        }

        let condition = if min_diag > 0.0 { max_diag / min_diag } else { f64::INFINITY };

        // Estimate memory: dense matrix is O(n²)
        let memory_estimate = self.dimension.unwrap_or(0).pow(2)
            * std::mem::size_of::<Complex<Scalar>>();

        Ok((solution, SolverStats {
            success: true,
            nnz_matrix: self.nnz.unwrap_or(0),
            nnz_factor: Some(self.dimension.unwrap_or(0).pow(2)), // Dense: all entries
            condition_estimate: Some(condition),
            solve_time: Some(solve_time),
            memory_bytes: Some(memory_estimate),
            notes: vec!["Baseline dense LU solver (use iterative for >10k nodes)".into()],
            ..Default::default()
        }))
    }

    fn name(&self) -> &str {
        "BaselineLU"
    }

    fn is_ready(&self) -> bool {
        self.lu.is_some()
    }
}

/// Automatic solver selection based on problem characteristics.
///
/// Analyzes the system matrix to recommend an appropriate solver type
/// (direct vs iterative) based on size, sparsity, and conditioning.
///
/// # Selection Criteria
///
/// **Direct Solvers (Baseline LU)**:
/// - Systems with <1000 nodes
/// - Dense matrices (density > 10%)
/// - When symbolic factorization can be reused (AC sweeps)
///
/// **Iterative Solvers (BiCGSTAB/GMRES)**:
/// - Systems with >10,000 nodes
/// - Very sparse matrices (density < 1%)
/// - Memory-constrained environments
/// - Systems with good preconditioners available
///
/// **Hybrid Zone (1k-10k nodes)**:
/// - Direct if sufficient memory
/// - Iterative if memory-constrained or very sparse
///
/// # References
///
/// - Davis (2010). "Algorithm 907: KLU". Recommends direct for circuit matrices <100k nonzeros.
/// - Saad (2003). "Iterative Methods". Recommends iterative for large sparse (>10k unknowns).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverRecommendation {
    /// Use direct solver (LU factorization).
    Direct,
    /// Use iterative solver (BiCGSTAB or GMRES).
    Iterative,
    /// Either approach viable; user/application should decide.
    Either,
}

/// Recommends solver type based on matrix characteristics.
///
/// # Arguments
///
/// * `n` - Number of unknowns (matrix dimension)
/// * `nnz` - Number of nonzeros in the matrix
/// * `available_memory_gb` - Available RAM in gigabytes (optional)
///
/// # Returns
///
/// Recommended solver type with rationale
pub fn recommend_solver(
    n: usize,
    nnz: usize,
    available_memory_gb: Option<f64>,
) -> (SolverRecommendation, String) {
    let density = (nnz as f64) / (n * n) as f64;

    // Estimate memory requirements
    let direct_memory_gb = (n * n * std::mem::size_of::<Complex<Scalar>>()) as f64 / 1e9;
    let iterative_memory_gb = (nnz * std::mem::size_of::<Complex<Scalar>>() * 10) as f64 / 1e9; // ~10x nnz for working vectors

    // Very small systems: direct is faster
    if n < 1000 {
        return (
            SolverRecommendation::Direct,
            format!("Small system ({} nodes): direct solver overhead minimal, ~{:.1}MB memory",
                   n, direct_memory_gb * 1000.0)
        );
    }

    // Memory constraint check
    if let Some(avail_gb) = available_memory_gb {
        if direct_memory_gb > avail_gb * 0.5 {
            return (
                SolverRecommendation::Iterative,
                format!("Memory-constrained: direct needs {:.1}GB > {:.1}GB available. Iterative needs ~{:.1}GB",
                       direct_memory_gb, avail_gb, iterative_memory_gb)
            );
        }
    }

    // Large systems: iterative required
    if n > 50_000 {
        return (
            SolverRecommendation::Iterative,
            format!("Large system ({} nodes): direct unfeasible ({:.1}GB), iterative required ({:.1}GB)",
                   n, direct_memory_gb, iterative_memory_gb)
        );
    }

    // Very sparse: iterative favorable
    if density < 0.001 && n > 5000 {
        return (
            SolverRecommendation::Iterative,
            format!("Very sparse ({:.2e} density, {} nonzeros): iterative favorable",
                   density, nnz)
        );
    }

    // Medium range: either works
    if n >= 1000 && n <= 10_000 {
        return (
            SolverRecommendation::Either,
            format!("Medium system ({} nodes, {:.1}GB direct vs {:.1}GB iterative): both viable",
                   n, direct_memory_gb, iterative_memory_gb)
        );
    }

    // Default for medium-large: prefer iterative for safety
    (
        SolverRecommendation::Iterative,
        format!("System size {} nodes, {:.2e} density: iterative recommended",
               n, density)
    )
}

#[cfg(test)]
mod solver_selection_tests {
    use super::*;

    #[test]
    fn recommend_direct_for_small() {
        let (rec, msg) = recommend_solver(500, 2500, None);
        assert_eq!(rec, SolverRecommendation::Direct);
        assert!(msg.contains("Small system"));
    }

    #[test]
    fn recommend_iterative_for_large() {
        let (rec, _msg) = recommend_solver(100_000, 500_000, None);
        assert_eq!(rec, SolverRecommendation::Iterative);
    }

    #[test]
    fn recommend_iterative_for_memory_constrained() {
        // 5k x 5k dense would need ~200MB
        let (rec, msg) = recommend_solver(5000, 25_000_000, Some(0.1)); // Only 100MB available
        assert_eq!(rec, SolverRecommendation::Iterative);
        assert!(msg.contains("Memory-constrained"));
    }

    #[test]
    fn recommend_either_for_medium() {
        let (rec, _msg) = recommend_solver(5000, 25_000, None);
        assert_eq!(rec, SolverRecommendation::Either);
    }
}


/// Iterative BiCGSTAB solver with diagonal (Jacobi) preconditioning.
///
/// Suitable for large, non-symmetric sparse systems typical of MNA.
/// Uses left preconditioning with an easily computed diagonal inverse.
pub struct BiCgStabSolver {
    /// Cached copy of matrix for matvec operations.
    a: Option<CscMatrix<Complex<Scalar>>>,
    /// Inverse of diagonal preconditioner M ≈ diag(A).
    m_inv_diag: Option<DVector<Complex<Scalar>>>,
    /// Matrix dimension.
    n: Option<usize>,
    /// Maximum iterations.
    max_iters: usize,
    /// Relative tolerance on residual norm.
    tol: Scalar,
}

impl BiCgStabSolver {
    /// Creates a new BiCGSTAB solver.
    pub fn new() -> Self {
        Self {
            a: None,
            m_inv_diag: None,
            n: None,
            max_iters: 2000,
            tol: 1e-8,
        }
    }

    /// Sets solver parameters.
    pub fn with_params(mut self, max_iters: usize, tol: Scalar) -> Self {
        self.max_iters = max_iters;
        self.tol = tol;
        self
    }

    fn spmv(matrix: &CscMatrix<Complex<Scalar>>, x: &DVector<Complex<Scalar>>) -> DVector<Complex<Scalar>> {
        let n = matrix.nrows();
        let mut y = DVector::from_element(n, Complex::new(0.0, 0.0));

        let col_offsets = matrix.col_offsets();
        let row_indices = matrix.row_indices();
        let values = matrix.values();

        for j in 0..matrix.ncols() {
            let start = col_offsets[j];
            let end = col_offsets[j + 1];
            let xj = x[j];
            for idx in start..end {
                let i = row_indices[idx];
                y[i] += values[idx] * xj;
            }
        }
        y
    }

    #[inline]
    fn apply_preconditioner(m_inv_diag: &DVector<Complex<Scalar>>, r: &DVector<Complex<Scalar>>) -> DVector<Complex<Scalar>> {
        // Jacobi: z = M^{-1} r, with M = diag(A)
        let mut z = r.clone();
        for i in 0..z.len() {
            z[i] *= m_inv_diag[i];
        }
        z
    }

    #[inline]
    fn dot_conj(a: &DVector<Complex<Scalar>>, b: &DVector<Complex<Scalar>>) -> Complex<Scalar> {
        // Conjugate(a)^T b
        let mut acc = Complex::new(0.0, 0.0);
        for i in 0..a.len() {
            acc += a[i].conj() * b[i];
        }
        acc
    }
}

impl Default for BiCgStabSolver {
    fn default() -> Self { Self::new() }
}

impl SparseSolver for BiCgStabSolver {
    fn symbolic(&mut self, matrix: &CscMatrix<Complex<Scalar>>) -> Result<(), SolverError> {
        if matrix.nrows() != matrix.ncols() {
            return Err(SolverError::InvalidMatrix(
                format!("Matrix must be square: {}x{}", matrix.nrows(), matrix.ncols())
            ));
        }
        self.n = Some(matrix.nrows());
        self.a = None; // reset
        self.m_inv_diag = None;
        Ok(())
    }

    fn numeric(&mut self, matrix: &CscMatrix<Complex<Scalar>>) -> Result<(), SolverError> {
        let n = self.n.ok_or_else(|| SolverError::Other("Must call symbolic() before numeric()".into()))?;
        if matrix.nrows() != n {
            return Err(SolverError::InvalidMatrix("Matrix dimensions changed since symbolic phase".into()));
        }

        // Build diagonal inverse for Jacobi preconditioner
        let mut diag = DVector::from_element(n, Complex::new(0.0, 0.0));
        let col_offsets = matrix.col_offsets();
        let row_indices = matrix.row_indices();
        let values = matrix.values();
        for j in 0..n {
            let start = col_offsets[j];
            let end = col_offsets[j + 1];
            for idx in start..end {
                let i = row_indices[idx];
                if i == j {
                    diag[i] = values[idx];
                }
            }
        }
        let mut inv = diag.clone();
        for i in 0..n {
            let d = diag[i];
            if d.norm() < 1e-18 {
                // Fallback to 1 on zero diagonal to avoid NaNs
                inv[i] = Complex::new(1.0, 0.0);
            } else {
                inv[i] = Complex::new(1.0, 0.0) / d;
            }
        }
        self.m_inv_diag = Some(inv);
        self.a = Some(matrix.clone());
        Ok(())
    }

    fn solve(&self, rhs: &DVector<Complex<Scalar>>) -> Result<DVector<Complex<Scalar>>, SolverError> {
        let a = self.a.as_ref().ok_or_else(|| SolverError::Other("Must call numeric() before solve()".into()))?;
        let m_inv = self.m_inv_diag.as_ref().ok_or_else(|| SolverError::Other("Preconditioner not built".into()))?;
        let n = rhs.len();

        // Initial guess x0 = 0
        let mut x = DVector::from_element(n, Complex::new(0.0, 0.0));
        let mut r = rhs - &Self::spmv(a, &x);
        let r_hat = r.clone();
        let mut v = DVector::from_element(n, Complex::new(0.0, 0.0));
        let mut p = DVector::from_element(n, Complex::new(0.0, 0.0));

        let mut rho_prev = Complex::new(1.0, 0.0);
        let mut alpha = Complex::new(1.0, 0.0);
        let mut omega = Complex::new(1.0, 0.0);

        let rhs_norm = rhs.norm();
        let mut r_norm = r.norm();
        if rhs_norm == 0.0 || r_norm / (rhs_norm + 1e-30) < self.tol {
            return Ok(x);
        }

        for _iter in 0..self.max_iters {
            let rho = Self::dot_conj(&r_hat, &r);
            if rho.norm() < 1e-30 {
                return Err(SolverError::NumericalInstability("Breakdown: rho ~ 0".into()));
            }
            let beta = (rho / rho_prev) * (alpha / omega);
            // p = r + beta*(p - omega*v)
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }

            // z = M^{-1} p
            let z = Self::apply_preconditioner(m_inv, &p);
            v = Self::spmv(a, &z);
            let r_hat_v = Self::dot_conj(&r_hat, &v);
            if r_hat_v.norm() < 1e-30 {
                return Err(SolverError::NumericalInstability("Breakdown: r_hat·v ~ 0".into()));
            }
            alpha = rho / r_hat_v;
            // s = r - alpha*v
            let mut s = r.clone();
            for i in 0..n { s[i] -= alpha * v[i]; }
            let s_norm = s.norm();
            if s_norm / (rhs_norm + 1e-30) < self.tol {
                // x = x + alpha*z
                for i in 0..n { x[i] += alpha * z[i]; }
                return Ok(x);
            }
            // z_s = M^{-1} s
            let z_s = Self::apply_preconditioner(m_inv, &s);
            let t = Self::spmv(a, &z_s);
            let tt = Self::dot_conj(&t, &t);
            if tt.norm() < 1e-30 {
                return Err(SolverError::NumericalInstability("Breakdown: t·t ~ 0".into()));
            }
            let omega_new = Self::dot_conj(&t, &s) / tt;
            // x = x + alpha*z + omega*z_s
            for i in 0..n { x[i] += alpha * z[i] + omega_new * z_s[i]; }
            // r = s - omega*t
            r = s;
            for i in 0..n { r[i] -= omega_new * t[i]; }
            r_norm = r.norm();
            if r_norm / (rhs_norm + 1e-30) < self.tol {
                return Ok(x);
            }
            if omega_new.norm() < 1e-30 {
                return Err(SolverError::NumericalInstability("Breakdown: omega ~ 0".into()));
            }
            rho_prev = rho;
            omega = omega_new;
        }

        Err(SolverError::ConvergenceFailure { iterations: self.max_iters, residual_norm: r_norm })
    }

    fn solve_with_stats(
        &self,
        rhs: &DVector<Complex<Scalar>>,
    ) -> Result<(DVector<Complex<Scalar>>, SolverStats), SolverError> {
        let start = std::time::Instant::now();
        let x = self.solve(rhs)?;
        let solve_time = start.elapsed();
        Ok((x, SolverStats {
            success: true,
            solve_time: Some(solve_time),
            notes: vec!["BiCGSTAB (Jacobi)".into()],
            ..Default::default()
        }))
    }

    fn name(&self) -> &str { "BiCGSTAB(Jacobi)" }

    fn is_ready(&self) -> bool { self.a.is_some() && self.m_inv_diag.is_some() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn baseline_solver_simple_system() {
        // 2x2 system: [2, 1; 1, 2] * [x; y] = [3; 3]
        // Solution: x = 1, y = 1
        let mut coo = nalgebra_sparse::coo::CooMatrix::new(2, 2);
        coo.push(0, 0, Complex::new(2.0, 0.0));
        coo.push(0, 1, Complex::new(1.0, 0.0));
        coo.push(1, 0, Complex::new(1.0, 0.0));
        coo.push(1, 1, Complex::new(2.0, 0.0));
        let matrix = CscMatrix::from(&coo);

        let rhs = DVector::from_vec(vec![Complex::new(3.0, 0.0), Complex::new(3.0, 0.0)]);

        let mut solver = BaselineLuSolver::new();
        solver.symbolic(&matrix).unwrap();
        solver.numeric(&matrix).unwrap();

        let solution = solver.solve(&rhs).unwrap();

        assert_relative_eq!(solution[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(solution[1].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn baseline_solver_complex_system() {
        // Complex 2x2: [(1+j), -j; j, (1+j)] * [x; y] = [(1+j); 1]
        let mut coo = nalgebra_sparse::coo::CooMatrix::new(2, 2);
        coo.push(0, 0, Complex::new(1.0, 1.0));
        coo.push(0, 1, Complex::new(0.0, -1.0));
        coo.push(1, 0, Complex::new(0.0, 1.0));
        coo.push(1, 1, Complex::new(1.0, 1.0));
        let matrix = CscMatrix::from(&coo);

        let rhs = DVector::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(1.0, 0.0)]);

        let mut solver = BaselineLuSolver::new();
        solver.symbolic(&matrix).unwrap();
        solver.numeric(&matrix).unwrap();

        let (solution, stats) = solver.solve_with_stats(&rhs).unwrap();

        // Verify solution by computing residual
        let dense = BaselineLuSolver::csc_to_dense(&matrix);
        let residual = &dense * &solution - &rhs;
        let residual_norm = residual.norm();

        assert!(residual_norm < 1e-10, "Residual too large: {}", residual_norm);
        assert!(stats.success);
        assert!(stats.condition_estimate.is_some());
    }

    #[test]
    fn baseline_solver_singular_matrix() {
        // Singular matrix: [1, 1; 1, 1]
        let mut coo = nalgebra_sparse::coo::CooMatrix::new(2, 2);
        coo.push(0, 0, Complex::new(1.0, 0.0));
        coo.push(0, 1, Complex::new(1.0, 0.0));
        coo.push(1, 0, Complex::new(1.0, 0.0));
        coo.push(1, 1, Complex::new(1.0, 0.0));
        let matrix = CscMatrix::from(&coo);

        let mut solver = BaselineLuSolver::new();
        solver.symbolic(&matrix).unwrap();
        let result = solver.numeric(&matrix);

        assert!(matches!(result, Err(SolverError::SingularMatrix)));
    }
}
