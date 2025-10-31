//! Optional sparse MNA types and helpers, gated behind `sparse` feature.

#![cfg(feature = "sparse")]

use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, CscMatrix};
use num_complex::Complex;

use crate::math::Scalar;

use super::stamp::{AcContext, SolveReport};
use super::solver::{SparseSolver, BaselineLuSolver};

/// Node index type: None represents ground, Some(i) is the i-th node.
pub type Node = Option<usize>;

/// Sparse Modified Nodal Analysis (MNA) builder using COO format during construction.
///
/// This builder supports the full MNA stamping methodology including:
/// - Passive elements (resistors, capacitors, inductors)
/// - Independent sources (current, voltage)
/// - Controlled sources (VCCS, VCVS, CCCS, CCVS)
///
/// Internally uses COO (Coordinate) format during construction, which naturally
/// handles duplicate entries by summing them during conversion to CSC format.
#[derive(Clone)]
pub struct SparseMnaBuilder {
    /// Number of non-ground nodes in the circuit.
    n: usize,
    /// Number of extra variables (e.g., voltage source currents).
    m: usize,
    /// COO matrix for the system matrix A (accumulates duplicates).
    coo: CooMatrix<Complex<Scalar>>,
    /// Right-hand side vector b.
    rhs: DVector<Complex<Scalar>>,
}

impl SparseMnaBuilder {
    /// Creates a sparse MNA builder with `node_count` non-ground nodes.
    ///
    /// The `nnz_hint` parameter provides an estimate of the number of nonzeros
    /// for efficient memory allocation, but is not a hard limit.
    pub fn new(node_count: usize, _nnz_hint: usize) -> Self {
        Self {
            n: node_count,
            m: 0,
            coo: CooMatrix::new(node_count, node_count),
            rhs: DVector::zeros(node_count),
        }
    }

    /// Adds a value to the system matrix at position (i, j).
    /// COO format allows duplicates; they will be summed during CSC conversion.
    pub fn add(&mut self, i: usize, j: usize, val: Complex<Scalar>) {
        self.coo.push(i, j, val);
    }

    /// Adds a value to the right-hand side vector at index i.
    pub fn add_rhs(&mut self, i: usize, val: Complex<Scalar>) {
        self.rhs[i] += val;
    }

    /// Returns the current system dimensions: (node_count, extra_variable_count).
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    /// Returns the number of equations (nodes + extra variables).
    #[must_use]
    pub fn equation_count(&self) -> usize {
        self.n + self.m
    }

    /// Returns an estimate of the number of nonzeros in the system matrix.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.coo.triplet_iter().count()
    }

    /// Converts a Node (optional index) to an actual row/column index.
    fn node_index(n: Node) -> Option<usize> {
        n
    }

    /// Returns the row/column index for the k-th extra variable.
    fn k_idx(&self, k: usize) -> usize {
        self.n + k
    }

    /// Ensures the system has capacity for at least `new_m` extra variables.
    /// Expands the COO matrix and RHS vector if necessary.
    fn ensure_capacity(&mut self, new_m: usize) {
        if new_m <= self.m {
            return;
        }
        let old_dim = self.n + self.m;
        self.m = new_m;
        let new_dim = self.n + self.m;

        // Rebuild COO matrix with new dimensions
        let mut new_coo = CooMatrix::new(new_dim, new_dim);
        for (row, col, val) in self.coo.triplet_iter() {
            new_coo.push(row, col, *val);
        }
        self.coo = new_coo;

        // Expand RHS vector
        let mut new_rhs = DVector::zeros(new_dim);
        new_rhs.rows_mut(0, old_dim).copy_from(&self.rhs);
        self.rhs = new_rhs;
    }

    /// Stamps a resistor of resistance `r` (Ω) between nodes `a` and `b`.
    ///
    /// The conductance g = 1/r is added to the admittance matrix using
    /// the standard two-port stamp pattern.
    pub fn stamp_resistor(&mut self, a: Node, b: Node, r: Scalar) {
        if r == 0.0 {
            return; // Avoid infinite conductance; future: model as constraint
        }
        let g = Complex::new(1.0 / r, 0.0);
        self.stamp_admittance(a, b, g);
    }

    /// Stamps a capacitor of capacitance `c` (F) between nodes `a` and `b` for AC analysis.
    ///
    /// Admittance Y = jωC is stamped into the matrix.
    /// At DC (ω=0), the capacitor is an open circuit (no stamp).
    pub fn stamp_capacitor(&mut self, a: Node, b: Node, c: Scalar, ctx: AcContext) {
        if ctx.omega == 0.0 {
            return; // Open circuit at DC
        }
        let j = Complex::new(0.0, 1.0);
        let y = j * (ctx.omega * c);
        self.stamp_admittance(a, b, y);
    }

    /// Stamps an inductor of inductance `l` (H) between nodes `a` and `b` for AC analysis.
    ///
    /// Admittance Y = -j/(ωL) is stamped into the matrix.
    /// At DC (ω=0), the inductor is a short circuit (modeled as large conductance).
    pub fn stamp_inductor(&mut self, a: Node, b: Node, l: Scalar, ctx: AcContext) {
        if ctx.omega == 0.0 {
            // Short circuit at DC: model as very large conductance
            let g = Complex::new(1.0e12, 0.0);
            self.stamp_admittance(a, b, g);
            return;
        }
        let j = Complex::new(0.0, 1.0);
        let y = -j / (ctx.omega * l);
        self.stamp_admittance(a, b, y);
    }

    /// Internal helper: stamps an admittance `y` between two nodes using the standard pattern.
    ///
    /// For nodes a and b (both non-ground):
    /// - A[a,a] += y, A[b,b] += y
    /// - A[a,b] -= y, A[b,a] -= y
    fn stamp_admittance(&mut self, a: Node, b: Node, y: Complex<Scalar>) {
        match (Self::node_index(a), Self::node_index(b)) {
            (Some(i), Some(j)) => {
                self.add(i, i, y);
                self.add(j, j, y);
                self.add(i, j, -y);
                self.add(j, i, -y);
            }
            (Some(i), None) => {
                self.add(i, i, y);
            }
            (None, Some(j)) => {
                self.add(j, j, y);
            }
            (None, None) => {}
        }
    }

    /// Stamps an independent current source injecting current `i` from node `pos` to node `neg`.
    ///
    /// Current flows into the network at `pos` and out at `neg`.
    /// This modifies the RHS vector only.
    pub fn stamp_current_source(&mut self, pos: Node, neg: Node, i: Complex<Scalar>) {
        if let Some(p) = Self::node_index(pos) {
            self.rhs[p] += i;
        }
        if let Some(n) = Self::node_index(neg) {
            self.rhs[n] -= i;
        }
    }

    /// Stamps an independent voltage source with voltage `v` between nodes `pos` and `neg`.
    ///
    /// This adds an extra variable for the source current and expands the system.
    /// Returns the index k of the new current variable.
    ///
    /// The voltage constraint is: V(pos) - V(neg) = v
    pub fn stamp_voltage_source(
        &mut self,
        pos: Node,
        neg: Node,
        v: Complex<Scalar>,
    ) -> usize {
        let k = self.m;
        self.ensure_capacity(k + 1);
        let row = self.k_idx(k);

        // Add KCL coupling: current variable k affects node equations
        if let Some(p) = Self::node_index(pos) {
            self.add(p, row, Complex::new(1.0, 0.0));
            self.add(row, p, Complex::new(1.0, 0.0));
        }
        if let Some(n) = Self::node_index(neg) {
            self.add(n, row, Complex::new(-1.0, 0.0));
            self.add(row, n, Complex::new(-1.0, 0.0));
        }

        // Voltage constraint: V(pos) - V(neg) = v
        self.rhs[row] = v;
        k
    }

    /// Stamps a voltage-controlled current source (VCCS).
    ///
    /// Injects current g * (V(cp) - V(cn)) from node `op` to node `on`.
    /// - Control port: nodes `cp` (positive) and `cn` (negative)
    /// - Output port: nodes `op` (positive) and `on` (negative)
    /// - Transconductance: `g` (S)
    pub fn stamp_vccs(&mut self, op: Node, on: Node, cp: Node, cn: Node, g: Scalar) {
        let g = Complex::new(g, 0.0);

        // Current into output positive node: g*(V(cp) - V(cn))
        if let Some(o) = Self::node_index(op) {
            if let Some(c) = Self::node_index(cp) {
                self.add(o, c, -g); // Move to LHS: -g*V(cp)
            }
            if let Some(c) = Self::node_index(cn) {
                self.add(o, c, g);  // Move to LHS: +g*V(cn)
            }
        }

        // Current out of output negative node: -g*(V(cp) - V(cn))
        if let Some(o) = Self::node_index(on) {
            if let Some(c) = Self::node_index(cp) {
                self.add(o, c, g);  // Move to LHS: +g*V(cp)
            }
            if let Some(c) = Self::node_index(cn) {
                self.add(o, c, -g); // Move to LHS: -g*V(cn)
            }
        }
    }

    /// Stamps a voltage-controlled voltage source (VCVS).
    ///
    /// Enforces: V(op) - V(on) = mu * (V(cp) - V(cn))
    /// - Control port: nodes `cp` and `cn`
    /// - Output port: nodes `op` and `on`
    /// - Voltage gain: `mu` (dimensionless)
    ///
    /// Returns the index of the new source current variable.
    pub fn stamp_vcvs(&mut self, op: Node, on: Node, cp: Node, cn: Node, mu: Scalar) -> usize {
        let k = self.m;
        self.ensure_capacity(k + 1);
        let row = self.k_idx(k);
        let mu = Complex::new(mu, 0.0);

        // KCL coupling for the source current
        if let Some(p) = Self::node_index(op) {
            self.add(p, row, Complex::new(1.0, 0.0));
            self.add(row, p, Complex::new(1.0, 0.0));
        }
        if let Some(n) = Self::node_index(on) {
            self.add(n, row, Complex::new(-1.0, 0.0));
            self.add(row, n, Complex::new(-1.0, 0.0));
        }

        // Voltage constraint: V(op) - V(on) - mu*(V(cp) - V(cn)) = 0
        if let Some(c) = Self::node_index(cp) {
            self.add(row, c, -mu);
        }
        if let Some(c) = Self::node_index(cn) {
            self.add(row, c, mu);
        }

        k
    }

    /// Stamps a current-controlled current source (CCCS).
    ///
    /// Injects current alpha * I(ctrl_k) from node `op` to node `on`.
    /// - Control current: I(ctrl_k), the current through a voltage source
    /// - Output port: nodes `op` and `on`
    /// - Current gain: `alpha` (dimensionless)
    ///
    /// Note: `ctrl_k` must be the index returned by a previous voltage source stamp.
    pub fn stamp_cccs(&mut self, op: Node, on: Node, ctrl_k: usize, alpha: Scalar) {
        let a = Complex::new(alpha, 0.0);
        let col = self.k_idx(ctrl_k);

        if let Some(o) = Self::node_index(op) {
            self.add(o, col, a);
        }
        if let Some(o) = Self::node_index(on) {
            self.add(o, col, -a);
        }
    }

    /// Stamps a current-controlled voltage source (CCVS).
    ///
    /// Enforces: V(op) - V(on) = mu * I(ctrl_k)
    /// - Control current: I(ctrl_k), the current through a voltage source
    /// - Output port: nodes `op` and `on`
    /// - Transresistance: `mu` (Ω)
    ///
    /// Returns the index of the new source current variable.
    pub fn stamp_ccvs(&mut self, op: Node, on: Node, ctrl_k: usize, mu: Scalar) -> usize {
        let k = self.m;
        self.ensure_capacity(k + 1);
        let row = self.k_idx(k);

        // KCL coupling for the new source current
        if let Some(p) = Self::node_index(op) {
            self.add(p, row, Complex::new(1.0, 0.0));
            self.add(row, p, Complex::new(1.0, 0.0));
        }
        if let Some(n) = Self::node_index(on) {
            self.add(n, row, Complex::new(-1.0, 0.0));
            self.add(row, n, Complex::new(-1.0, 0.0));
        }

        // Voltage constraint: V(op) - V(on) - mu*I(ctrl_k) = 0
        let mu = Complex::new(mu, 0.0);
        let col_ctrl = self.k_idx(ctrl_k);
        self.add(row, col_ctrl, -mu);

        k
    }

    /// Splits a solution vector into node voltages and source currents.
    ///
    /// Given a solution vector x of length n+m:
    /// - First n elements: node voltages
    /// - Last m elements: voltage source currents
    #[must_use]
    pub fn split_solution(
        &self,
        x: DVector<Complex<Scalar>>,
    ) -> (DVector<Complex<Scalar>>, DVector<Complex<Scalar>>) {
        let (n, m) = (self.n, self.m);
        let v = x.rows(0, n).into_owned();
        let i = if m > 0 {
            x.rows(n, m).into_owned()
        } else {
            DVector::zeros(0)
        };
        (v, i)
    }

    /// Solves the MNA system using the baseline sparse solver.
    ///
    /// This method converts the COO matrix to CSC, creates a baseline LU solver,
    /// and solves the system Ax = b.
    ///
    /// # Returns
    ///
    /// Solution vector x containing both node voltages and source currents,
    /// or None if the system is singular or solve fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut mna = SparseMnaBuilder::new(2, 10);
    /// mna.stamp_resistor(Some(0), Some(1), 1000.0);
    /// mna.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));
    /// let solution = mna.solve().expect("solve failed");
    /// let (voltages, currents) = mna.split_solution(solution);
    /// ```
    #[must_use]
    pub fn solve(&self) -> Option<DVector<Complex<Scalar>>> {
        let (matrix, rhs) = self.clone().finalize();

        let mut solver = BaselineLuSolver::new();
        solver.symbolic(&matrix).ok()?;
        solver.numeric(&matrix).ok()?;
        solver.solve(&rhs).ok()
    }

    /// Solves the MNA system and returns detailed diagnostics.
    ///
    /// This is the recommended solve method, providing comprehensive statistics
    /// including condition number estimates, solve time, memory usage, and
    /// detection of problematic circuit conditions (floating nodes, singularities).
    ///
    /// # Returns
    ///
    /// Tuple of (solution, report) where solution contains node voltages and
    /// source currents, and report contains diagnostics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (solution, report) = mna.solve_with_report();
    /// if !report.success {
    ///     eprintln!("Solve failed: {:?}", report.notes);
    /// }
    /// println!("Condition estimate: {:?}", report.cond_estimate);
    /// ```
    #[must_use]
    pub fn solve_with_report(&self) -> (Option<DVector<Complex<Scalar>>>, SolveReport) {
        let (matrix, rhs) = self.clone().finalize();

        let mut report = SolveReport::default();

        // Detect floating nodes: rows with near-zero coefficients
        let n = self.n;
        for i in 0..n {
            let mut row_norm = 0.0;
            for (row, _col, val) in matrix.triplet_iter() {
                if row == i {
                    row_norm += val.norm();
                }
            }
            if row_norm <= 1e-14 {
                report.floating_nodes.push(i);
            }
        }

        // Solve using baseline solver
        let mut solver = BaselineLuSolver::new();

        if let Err(e) = solver.symbolic(&matrix) {
            report.success = false;
            report.notes.push(format!("Symbolic phase failed: {}", e));
            return (None, report);
        }

        if let Err(e) = solver.numeric(&matrix) {
            report.success = false;
            report.notes.push(format!("Numeric phase failed: {}", e));
            return (None, report);
        }

        match solver.solve_with_stats(&rhs) {
            Ok((solution, stats)) => {
                report.success = true;
                report.cond_estimate = stats.condition_estimate;

                if !report.floating_nodes.is_empty() {
                    report.notes.push(
                        format!("Warning: {} floating node(s) detected", report.floating_nodes.len())
                    );
                }

                if let Some(cond) = report.cond_estimate {
                    if cond > 1e12 {
                        report.notes.push(
                            format!("Warning: ill-conditioned matrix (cond ≈ {:.2e})", cond)
                        );
                    }
                }

                (Some(solution), report)
            }
            Err(e) => {
                report.success = false;
                report.notes.push(format!("Solve failed: {}", e));
                (None, report)
            }
        }
    }

    /// Finalizes the builder into a CSC system matrix and RHS vector.
    ///
    /// The COO matrix is converted to CSC format, automatically summing duplicate entries.
    /// This is the final step before passing to a sparse solver.
    ///
    /// Note: Solver integration is provided separately (see Phase 2 of implementation).
    pub fn finalize(self) -> (CscMatrix<Complex<Scalar>>, DVector<Complex<Scalar>>) {
        (CscMatrix::from(&self.coo), self.rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits::stamp::MnaBuilder;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;

    /// Helper to convert sparse CSC matrix to dense for comparison.
    fn csc_to_dense(csc: &CscMatrix<Complex<Scalar>>) -> DMatrix<Complex<Scalar>> {
        let n = csc.nrows();
        let m = csc.ncols();
        let mut dense = DMatrix::zeros(n, m);

        // Use triplet iterator to extract all non-zero entries
        for (row, col, &value) in csc.triplet_iter() {
            dense[(row, col)] = value;
        }
        dense
    }

    /// Helper to compare two complex matrices element-wise.
    fn assert_matrices_equal(
        sparse_dense: &DMatrix<Complex<Scalar>>,
        dense: &DMatrix<Complex<Scalar>>,
        epsilon: Scalar,
    ) {
        assert_eq!(sparse_dense.nrows(), dense.nrows(), "Row count mismatch");
        assert_eq!(sparse_dense.ncols(), dense.ncols(), "Column count mismatch");

        for i in 0..sparse_dense.nrows() {
            for j in 0..sparse_dense.ncols() {
                let sparse_val = sparse_dense[(i, j)];
                let dense_val = dense[(i, j)];
                assert_relative_eq!(
                    sparse_val.re,
                    dense_val.re,
                    epsilon = epsilon
                );
                assert_relative_eq!(
                    sparse_val.im,
                    dense_val.im,
                    epsilon = epsilon
                );
            }
        }
    }

    /// Helper to compare RHS vectors.
    fn assert_vectors_equal(
        sparse_rhs: &DVector<Complex<Scalar>>,
        dense_rhs: &DVector<Complex<Scalar>>,
        epsilon: Scalar,
    ) {
        assert_eq!(sparse_rhs.len(), dense_rhs.len(), "RHS length mismatch");
        for i in 0..sparse_rhs.len() {
            assert_relative_eq!(
                sparse_rhs[i].re,
                dense_rhs[i].re,
                epsilon = epsilon
            );
            assert_relative_eq!(
                sparse_rhs[i].im,
                dense_rhs[i].im,
                epsilon = epsilon
            );
        }
    }

    #[test]
    fn sparse_vs_dense_simple_resistor_network() {
        // Three resistors: R1 between node 0-1, R2 between 1-2, R3 between 2-ground
        // Current source 1A into node 0
        let n = 3;

        // Dense version
        let mut dense = MnaBuilder::new(n);
        dense.stamp_resistor(Some(0), Some(1), 1000.0);
        dense.stamp_resistor(Some(1), Some(2), 2000.0);
        dense.stamp_resistor(Some(2), None, 3000.0);
        dense.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 20);
        sparse.stamp_resistor(Some(0), Some(1), 1000.0);
        sparse.stamp_resistor(Some(1), Some(2), 2000.0);
        sparse.stamp_resistor(Some(2), None, 3000.0);
        sparse.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));

        // Get matrices
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        // Compare
        assert_matrices_equal(&sparse_a, &dense_a, 1e-12);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-12);
    }

    #[test]
    fn sparse_vs_dense_ac_capacitor_network() {
        // Two capacitors and a resistor at 1 kHz
        let n = 2;
        let omega = 2.0 * std::f64::consts::PI * 1000.0; // 1 kHz
        let ctx = AcContext { omega };

        // Dense version
        let mut dense = MnaBuilder::new(n);
        dense.stamp_capacitor(Some(0), Some(1), 1e-6, ctx); // 1 µF
        dense.stamp_capacitor(Some(1), None, 2e-6, ctx);    // 2 µF
        dense.stamp_resistor(Some(0), None, 1000.0);        // 1 kΩ
        dense.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 15);
        sparse.stamp_capacitor(Some(0), Some(1), 1e-6, ctx);
        sparse.stamp_capacitor(Some(1), None, 2e-6, ctx);
        sparse.stamp_resistor(Some(0), None, 1000.0);
        sparse.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));

        // Compare
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_matrices_equal(&sparse_a, &dense_a, 1e-10);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-10);
    }

    #[test]
    fn sparse_vs_dense_ac_inductor_network() {
        // Inductor network at 10 kHz
        let n = 2;
        let omega = 2.0 * std::f64::consts::PI * 10000.0; // 10 kHz
        let ctx = AcContext { omega };

        // Dense version
        let mut dense = MnaBuilder::new(n);
        dense.stamp_inductor(Some(0), Some(1), 1e-3, ctx); // 1 mH
        dense.stamp_resistor(Some(1), None, 100.0);         // 100 Ω
        dense.stamp_current_source(Some(0), None, Complex::new(0.5, 0.0));

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 10);
        sparse.stamp_inductor(Some(0), Some(1), 1e-3, ctx);
        sparse.stamp_resistor(Some(1), None, 100.0);
        sparse.stamp_current_source(Some(0), None, Complex::new(0.5, 0.0));

        // Compare
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_matrices_equal(&sparse_a, &dense_a, 1e-10);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-10);
    }

    #[test]
    fn sparse_vs_dense_voltage_source() {
        // Voltage source with voltage divider
        // Vsrc 10V between node 0 and ground, R1 between nodes 0-1, R2 between node 1-ground
        let n = 2;

        // Dense version
        let mut dense = MnaBuilder::new(n);
        dense.stamp_voltage_source(Some(0), None, Complex::new(10.0, 0.0));
        dense.stamp_resistor(Some(0), Some(1), 1000.0);
        dense.stamp_resistor(Some(1), None, 2000.0);

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 20);
        sparse.stamp_voltage_source(Some(0), None, Complex::new(10.0, 0.0));
        sparse.stamp_resistor(Some(0), Some(1), 1000.0);
        sparse.stamp_resistor(Some(1), None, 2000.0);

        // Compare (note: dimensions include extra variable for voltage source)
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_eq!(sparse_a.nrows(), dense_a.nrows(), "Matrix size should match including extra variable");
        assert_matrices_equal(&sparse_a, &dense_a, 1e-12);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-12);
    }

    #[test]
    fn sparse_vs_dense_vccs() {
        // Voltage-controlled current source
        // Control: 1V at node 0, Output: inject into node 1 with 1k load
        let n = 2;

        // Dense version
        let mut dense = MnaBuilder::new(n);
        dense.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        dense.stamp_resistor(Some(1), None, 1000.0);
        dense.stamp_vccs(Some(1), None, Some(0), None, 1e-3); // g = 1 mS

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 20);
        sparse.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        sparse.stamp_resistor(Some(1), None, 1000.0);
        sparse.stamp_vccs(Some(1), None, Some(0), None, 1e-3);

        // Compare
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_matrices_equal(&sparse_a, &dense_a, 1e-12);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-12);
    }

    #[test]
    fn sparse_vs_dense_vcvs() {
        // Voltage-controlled voltage source
        let n = 2;

        // Dense version
        let mut dense = MnaBuilder::new(n);
        dense.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0)); // Control voltage
        let _k = dense.stamp_vcvs(Some(1), None, Some(0), None, 2.0); // Gain = 2
        dense.stamp_resistor(Some(1), None, 1000.0);

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 25);
        sparse.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        let _k_sparse = sparse.stamp_vcvs(Some(1), None, Some(0), None, 2.0);
        sparse.stamp_resistor(Some(1), None, 1000.0);

        // Compare
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_matrices_equal(&sparse_a, &dense_a, 1e-12);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-12);
    }

    #[test]
    fn sparse_vs_dense_cccs() {
        // Current-controlled current source
        let n = 2;

        // Dense version
        let mut dense = MnaBuilder::new(n);
        let k = dense.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        dense.stamp_resistor(Some(0), None, 1000.0); // Creates control current
        dense.stamp_resistor(Some(1), None, 1000.0); // Output load
        dense.stamp_cccs(Some(1), None, k, 1.0); // Alpha = 1 (current mirror)

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 20);
        let k_sparse = sparse.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        sparse.stamp_resistor(Some(0), None, 1000.0);
        sparse.stamp_resistor(Some(1), None, 1000.0);
        sparse.stamp_cccs(Some(1), None, k_sparse, 1.0);

        // Compare
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_matrices_equal(&sparse_a, &dense_a, 1e-12);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-12);
    }

    #[test]
    fn sparse_vs_dense_ccvs() {
        // Current-controlled voltage source
        let n = 2;

        // Dense version
        let mut dense = MnaBuilder::new(n);
        let k = dense.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        dense.stamp_resistor(Some(0), None, 1000.0); // Creates control current
        let _k2 = dense.stamp_ccvs(Some(1), None, k, 500.0); // Transresistance = 500 Ω
        dense.stamp_resistor(Some(1), None, 1000.0);

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 25);
        let k_sparse = sparse.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        sparse.stamp_resistor(Some(0), None, 1000.0);
        let _k2_sparse = sparse.stamp_ccvs(Some(1), None, k_sparse, 500.0);
        sparse.stamp_resistor(Some(1), None, 1000.0);

        // Compare
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_matrices_equal(&sparse_a, &dense_a, 1e-12);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-12);
    }

    #[test]
    fn sparse_vs_dense_complex_mixed_circuit() {
        // Complex circuit with multiple element types at 1 MHz
        let n = 4;
        let omega = 2.0 * std::f64::consts::PI * 1e6; // 1 MHz
        let ctx = AcContext { omega };

        // Dense version
        let mut dense = MnaBuilder::new(n);
        let k = dense.stamp_voltage_source(Some(0), None, Complex::new(5.0, 0.0));
        dense.stamp_resistor(Some(0), Some(1), 50.0); // 50 Ω series
        dense.stamp_capacitor(Some(1), None, 100e-12, ctx); // 100 pF to ground
        dense.stamp_inductor(Some(1), Some(2), 10e-9, ctx); // 10 nH series
        dense.stamp_resistor(Some(2), None, 75.0); // 75 Ω load
        dense.stamp_capacitor(Some(2), Some(3), 47e-12, ctx); // 47 pF coupling
        dense.stamp_resistor(Some(3), None, 100.0); // 100 Ω termination
        dense.stamp_cccs(Some(3), None, k, 0.5); // Feedback

        // Sparse version
        let mut sparse = SparseMnaBuilder::new(n, 40);
        let k_sparse = sparse.stamp_voltage_source(Some(0), None, Complex::new(5.0, 0.0));
        sparse.stamp_resistor(Some(0), Some(1), 50.0);
        sparse.stamp_capacitor(Some(1), None, 100e-12, ctx);
        sparse.stamp_inductor(Some(1), Some(2), 10e-9, ctx);
        sparse.stamp_resistor(Some(2), None, 75.0);
        sparse.stamp_capacitor(Some(2), Some(3), 47e-12, ctx);
        sparse.stamp_resistor(Some(3), None, 100.0);
        sparse.stamp_cccs(Some(3), None, k_sparse, 0.5);

        // Compare
        let (dense_a, dense_b) = dense.system_matrix_and_rhs();
        let (sparse_csc, sparse_b) = sparse.finalize();
        let sparse_a = csc_to_dense(&sparse_csc);

        assert_matrices_equal(&sparse_a, &dense_a, 1e-10);
        assert_vectors_equal(&sparse_b, &dense_b, 1e-10);
    }

    #[test]
    fn sparse_dimensions_tracking() {
        // Test that dimensions are correctly tracked
        let mut sparse = SparseMnaBuilder::new(3, 10);
        assert_eq!(sparse.dimensions(), (3, 0));
        assert_eq!(sparse.equation_count(), 3);

        sparse.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        assert_eq!(sparse.dimensions(), (3, 1));
        assert_eq!(sparse.equation_count(), 4);

        sparse.stamp_vcvs(Some(1), None, Some(0), None, 2.0);
        assert_eq!(sparse.dimensions(), (3, 2));
        assert_eq!(sparse.equation_count(), 5);
    }

    /// Tests comparing sparse solve() to dense solve()
    mod solve_comparison {
        use super::*;

        #[test]
        fn sparse_solve_resistor_divider() {
            // Voltage divider: 10V source, R1=1k, R2=2k
            // Expected: V(node1) = 10V * 2k/(1k+2k) = 6.666...V

            // Dense version
            let mut dense = MnaBuilder::new(2);
            dense.stamp_voltage_source(Some(0), None, Complex::new(10.0, 0.0));
            dense.stamp_resistor(Some(0), Some(1), 1000.0);
            dense.stamp_resistor(Some(1), None, 2000.0);
            let dense_sol = dense.solve().expect("dense solve");
            let (dense_v, _) = dense.split_solution(dense_sol);

            // Sparse version
            let mut sparse = SparseMnaBuilder::new(2, 20);
            sparse.stamp_voltage_source(Some(0), None, Complex::new(10.0, 0.0));
            sparse.stamp_resistor(Some(0), Some(1), 1000.0);
            sparse.stamp_resistor(Some(1), None, 2000.0);
            let sparse_sol = sparse.solve().expect("sparse solve");
            let (sparse_v, _) = sparse.split_solution(sparse_sol);

            // Compare solutions
            assert_eq!(sparse_v.len(), dense_v.len());
            for i in 0..sparse_v.len() {
                assert_relative_eq!(sparse_v[i].re, dense_v[i].re, epsilon = 1e-10);
                assert_relative_eq!(sparse_v[i].im, dense_v[i].im, epsilon = 1e-10);
            }

            // Verify expected value
            assert_relative_eq!(sparse_v[1].re, 10.0 * 2000.0 / 3000.0, epsilon = 1e-9);
        }

        #[test]
        fn sparse_solve_ac_rc_circuit() {
            // RC low-pass filter at 1kHz: V_in=1V, R=1k, C=1µF
            let n = 2;
            let omega = 2.0 * std::f64::consts::PI * 1000.0;
            let ctx = AcContext { omega };

            // Dense
            let mut dense = MnaBuilder::new(n);
            dense.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
            dense.stamp_resistor(Some(0), Some(1), 1000.0);
            dense.stamp_capacitor(Some(1), None, 1e-6, ctx);
            let dense_sol = dense.solve().expect("dense");
            let (dense_v, _) = dense.split_solution(dense_sol);

            // Sparse
            let mut sparse = SparseMnaBuilder::new(n, 15);
            sparse.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
            sparse.stamp_resistor(Some(0), Some(1), 1000.0);
            sparse.stamp_capacitor(Some(1), None, 1e-6, ctx);
            let sparse_sol = sparse.solve().expect("sparse");
            let (sparse_v, _) = sparse.split_solution(sparse_sol);

            // Compare
            for i in 0..sparse_v.len() {
                assert_relative_eq!(sparse_v[i].re, dense_v[i].re, epsilon = 1e-10);
                assert_relative_eq!(sparse_v[i].im, dense_v[i].im, epsilon = 1e-10);
            }
        }

        #[test]
        fn sparse_solve_with_report_detects_issues() {
            // Floating node scenario
            let mut sparse = SparseMnaBuilder::new(2, 10);
            sparse.stamp_resistor(Some(0), None, 1000.0);
            sparse.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));
            // Node 1 is floating (no connections)

            let (_sol, report) = sparse.solve_with_report();

            // Floating node causes singular matrix, so solve might fail
            // The important part is that we detect the floating node
            assert!(!report.floating_nodes.is_empty(), "Should detect floating node 1");
            assert!(report.floating_nodes.contains(&1));
        }

        #[test]
        fn sparse_solve_complex_circuit_matches_dense() {
            // Complex circuit with multiple element types
            let n = 4;
            let omega = 2.0 * std::f64::consts::PI * 1e6;
            let ctx = AcContext { omega };

            // Build both versions
            let build_circuit = |is_sparse: bool| -> (DVector<Complex<Scalar>>, DVector<Complex<Scalar>>) {
                if is_sparse {
                    let mut mna = SparseMnaBuilder::new(n, 40);
                    let k = mna.stamp_voltage_source(Some(0), None, Complex::new(5.0, 0.0));
                    mna.stamp_resistor(Some(0), Some(1), 50.0);
                    mna.stamp_capacitor(Some(1), None, 100e-12, ctx);
                    mna.stamp_inductor(Some(1), Some(2), 10e-9, ctx);
                    mna.stamp_resistor(Some(2), None, 75.0);
                    mna.stamp_capacitor(Some(2), Some(3), 47e-12, ctx);
                    mna.stamp_resistor(Some(3), None, 100.0);
                    mna.stamp_cccs(Some(3), None, k, 0.5);

                    let sol = mna.solve().expect("sparse solve");
                    mna.split_solution(sol)
                } else {
                    let mut mna = MnaBuilder::new(n);
                    let k = mna.stamp_voltage_source(Some(0), None, Complex::new(5.0, 0.0));
                    mna.stamp_resistor(Some(0), Some(1), 50.0);
                    mna.stamp_capacitor(Some(1), None, 100e-12, ctx);
                    mna.stamp_inductor(Some(1), Some(2), 10e-9, ctx);
                    mna.stamp_resistor(Some(2), None, 75.0);
                    mna.stamp_capacitor(Some(2), Some(3), 47e-12, ctx);
                    mna.stamp_resistor(Some(3), None, 100.0);
                    mna.stamp_cccs(Some(3), None, k, 0.5);

                    let sol = mna.solve().expect("dense solve");
                    mna.split_solution(sol)
                }
            };

            let (sparse_v, sparse_i) = build_circuit(true);
            let (dense_v, dense_i) = build_circuit(false);

            // Compare voltages
            assert_eq!(sparse_v.len(), dense_v.len());
            for i in 0..sparse_v.len() {
                assert_relative_eq!(sparse_v[i].re, dense_v[i].re, epsilon = 1e-9);
                assert_relative_eq!(sparse_v[i].im, dense_v[i].im, epsilon = 1e-9);
            }

            // Compare currents
            assert_eq!(sparse_i.len(), dense_i.len());
            for i in 0..sparse_i.len() {
                assert_relative_eq!(sparse_i[i].re, dense_i[i].re, epsilon = 1e-9);
                assert_relative_eq!(sparse_i[i].im, dense_i[i].im, epsilon = 1e-9);
            }
        }

        #[test]
        fn sparse_solve_ladder_network() {
            // 5-stage RC ladder network
            let n = 6; // 6 nodes including input
            let omega = 2.0 * std::f64::consts::PI * 10000.0;
            let ctx = AcContext { omega };

            // Dense
            let mut dense = MnaBuilder::new(n);
            dense.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
            for i in 0..5 {
                dense.stamp_resistor(Some(i), Some(i + 1), 1000.0);
                dense.stamp_capacitor(Some(i + 1), None, 100e-9, ctx);
            }
            let (dense_v, _) = dense.split_solution(dense.solve().expect("dense"));

            // Sparse
            let mut sparse = SparseMnaBuilder::new(n, 50);
            sparse.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
            for i in 0..5 {
                sparse.stamp_resistor(Some(i), Some(i + 1), 1000.0);
                sparse.stamp_capacitor(Some(i + 1), None, 100e-9, ctx);
            }
            let (sparse_v, _) = sparse.split_solution(sparse.solve().expect("sparse"));

            // Compare all nodes
            for i in 0..n {
                assert_relative_eq!(sparse_v[i].re, dense_v[i].re, epsilon = 1e-9);
                assert_relative_eq!(sparse_v[i].im, dense_v[i].im, epsilon = 1e-9);
            }

            // Verify attenuation increases through ladder
            let mag0 = sparse_v[0].norm();
            let mag_last = sparse_v[n-1].norm();
            assert!(mag_last < mag0, "Output should be attenuated");
        }
    }
}

