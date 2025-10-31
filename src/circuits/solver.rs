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
