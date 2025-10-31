//! Nodal stamping helpers for DC/AC analysis (dense prototype).

use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

use crate::math::Scalar;

use super::analysis::{AdmittanceMatrix, CurrentVector, NodalAnalysis};

/// Node index (0-based). The ground node is represented by `None`.
pub type Node = Option<usize>;

/// AC frequency context for stamping reactive elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcContext {
    /// Angular frequency ω (rad/s).
    pub omega: Scalar,
}

impl AcContext {
    /// DC context (ω = 0).
    #[must_use]
    pub fn dc() -> Self {
        Self { omega: 0.0 }
    }
}

/// Builder for nodal analysis systems using element stamping.
pub struct NodalBuilder {
    y: AdmittanceMatrix,
    i: CurrentVector,
}

impl NodalBuilder {
    /// Creates a stamping context with `node_count` non-ground nodes.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            y: DMatrix::zeros(node_count, node_count),
            i: DVector::zeros(node_count),
        }
    }

    fn idx(n: Node) -> Option<usize> {
        n
    }

    /// Stamps a resistor of resistance `r` between nodes `a` and `b`.
    pub fn stamp_resistor(&mut self, a: Node, b: Node, r: Scalar) {
        if r == 0.0 {
            return; // avoid inf conductance; future: handle as short constraint
        }
        let g = Complex::new(1.0 / r, 0.0);
        self.stamp_admittance(a, b, g);
    }

    /// Stamps a capacitor between nodes `a` and `b` for AC with `omega`.
    pub fn stamp_capacitor(&mut self, a: Node, b: Node, c: Scalar, ctx: AcContext) {
        if ctx.omega == 0.0 {
            // Open circuit at DC: no stamp
            return;
        }
        let j = Complex::new(0.0, 1.0);
        let y = j * (ctx.omega * c);
        self.stamp_admittance(a, b, y);
    }

    /// Stamps an inductor between nodes `a` and `b` for AC with `omega`.
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

    /// Stamps a current source `i` from node `pos` to `neg` (into the network).
    pub fn stamp_current_source(&mut self, pos: Node, neg: Node, i: Complex<Scalar>) {
        if let Some(p) = Self::idx(pos) {
            self.i[p] += i;
        }
        if let Some(n) = Self::idx(neg) {
            self.i[n] -= i;
        }
    }

    /// Internal: stamp an admittance `y` between nodes.
    fn stamp_admittance(&mut self, a: Node, b: Node, y: Complex<Scalar>) {
        match (Self::idx(a), Self::idx(b)) {
            (Some(i), Some(j)) => {
                self.y[(i, i)] += y;
                self.y[(j, j)] += y;
                self.y[(i, j)] -= y;
                self.y[(j, i)] -= y;
            }
            (Some(i), None) => {
                self.y[(i, i)] += y;
            }
            (None, Some(j)) => {
                self.y[(j, j)] += y;
            }
            (None, None) => {}
        }
    }

    /// Finalizes the builder into a `NodalAnalysis` system.
    #[must_use]
    pub fn build(self) -> NodalAnalysis {
        NodalAnalysis {
            admittance: self.y,
            current: self.i,
        }
    }
}

/// Report with diagnostics from solving an MNA system.
#[derive(Debug, Clone, Default)]
pub struct SolveReport {
    /// True if the linear solve succeeded.
    pub success: bool,
    /// Rough condition estimate from LU diagonal ratio (larger is worse).
    pub cond_estimate: Option<Scalar>,
    /// Indices of floating (disconnected) nodes detected before solve.
    pub floating_nodes: Vec<usize>,
    /// Any overload conditions detected (placeholder for now).
    pub overloads: Vec<String>,
    /// Additional human-readable notes.
    pub notes: Vec<String>,
}

/// Modified Nodal Analysis (MNA) builder supporting voltage sources and VCVS/VCCS.
pub struct MnaBuilder {
    n: usize,
    a: DMatrix<Complex<Scalar>>, // system matrix (n+m) x (n+m)
    b: DVector<Complex<Scalar>>, // rhs
    m: usize,                    // number of extra variables (currents)
}

impl MnaBuilder {
    /// Creates an MNA system with `node_count` non-ground nodes.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            n: node_count,
            a: DMatrix::zeros(node_count, node_count),
            b: DVector::zeros(node_count),
            m: 0,
        }
    }

    fn ensure_capacity(&mut self, new_m: usize) {
        if new_m <= self.m {
            return;
        }
        let old_n = self.n + self.m;
        self.m = new_m;
        let new_n = self.n + self.m;
        let mut na = DMatrix::zeros(new_n, new_n);
        let mut nb = DVector::zeros(new_n);
        for r in 0..old_n {
            for c in 0..old_n {
                na[(r, c)] = self.a[(r, c)];
            }
        }
        nb.rows_mut(0, old_n).copy_from(&self.b);
        self.a = na;
        self.b = nb;
    }

    fn node_index(n: Node) -> Option<usize> {
        n
    }

    fn k_idx(&self, k: usize) -> usize {
        self.n + k
    }

    /// Stamps a resistor `r` between nodes `a` and `b`.
    pub fn stamp_resistor(&mut self, a: Node, b: Node, r: Scalar) {
        if r == 0.0 {
            return;
        }
        let g = Complex::new(1.0 / r, 0.0);
        match (Self::node_index(a), Self::node_index(b)) {
            (Some(i), Some(j)) => {
                self.a[(i, i)] += g;
                self.a[(j, j)] += g;
                self.a[(i, j)] -= g;
                self.a[(j, i)] -= g;
            }
            (Some(i), None) => self.a[(i, i)] += g,
            (None, Some(j)) => self.a[(j, j)] += g,
            (None, None) => {}
        }
    }

    /// Stamps a current source `i` from `pos` to `neg`.
    pub fn stamp_current_source(&mut self, pos: Node, neg: Node, i: Complex<Scalar>) {
        if let Some(p) = Self::node_index(pos) {
            self.b[p] += i;
        }
        if let Some(n) = Self::node_index(neg) {
            self.b[n] -= i;
        }
    }

    /// Adds an independent voltage source between `pos` and `neg` with voltage `v` (complex for AC).
    /// Returns the index of the source current variable.
    pub fn stamp_voltage_source(
        &mut self,
        pos: Node,
        neg: Node,
        v: Complex<Scalar>,
    ) -> usize {
        let k = self.m; // new source index
        self.ensure_capacity(k + 1);
        let row = self.k_idx(k);

        // KCL coupling columns
        if let Some(p) = Self::node_index(pos) {
            self.a[(p, row)] += Complex::new(1.0, 0.0);
            self.a[(row, p)] += Complex::new(1.0, 0.0);
        }
        if let Some(n) = Self::node_index(neg) {
            self.a[(n, row)] -= Complex::new(1.0, 0.0);
            self.a[(row, n)] -= Complex::new(1.0, 0.0);
        }

        // Voltage constraint RHS
        self.b[row] += v;
        k
    }

    /// Voltage-controlled current source (VCCS). Injects `g*(v_cp - v_cn)` from `op` to `on`.
    pub fn stamp_vccs(&mut self, op: Node, on: Node, cp: Node, cn: Node, g: Scalar) {
        let g = Complex::new(g, 0.0);
        // Row for op: move injection g*(vcp - vcn) to LHS: -g*vcp + g*vcn
        if let Some(o) = Self::node_index(op) {
            if let Some(c) = Self::node_index(cp) {
                self.a[(o, c)] -= g;
            }
            if let Some(c) = Self::node_index(cn) {
                self.a[(o, c)] += g;
            }
        }
        // Row for on: opposite current, move to LHS: +g*vcp - g*vcn
        if let Some(o) = Self::node_index(on) {
            if let Some(c) = Self::node_index(cp) {
                self.a[(o, c)] += g;
            }
            if let Some(c) = Self::node_index(cn) {
                self.a[(o, c)] -= g;
            }
        }
    }

    /// Voltage-controlled voltage source (VCVS). Enforces `v(op)-v(on) = mu*(v(cp)-v(cn))`.
    /// Adds a source current variable, similar to an independent voltage source.
    pub fn stamp_vcvs(&mut self, op: Node, on: Node, cp: Node, cn: Node, mu: Scalar) -> usize {
        let k = self.m;
        self.ensure_capacity(k + 1);
        let row = self.k_idx(k);
        let mu = Complex::new(mu, 0.0);

        if let Some(p) = Self::node_index(op) {
            self.a[(p, row)] += Complex::new(1.0, 0.0);
            self.a[(row, p)] += Complex::new(1.0, 0.0);
        }
        if let Some(n) = Self::node_index(on) {
            self.a[(n, row)] -= Complex::new(1.0, 0.0);
            self.a[(row, n)] -= Complex::new(1.0, 0.0);
        }
        // Control: v(op)-v(on) - mu*(v(cp)-v(cn)) = 0
        if let Some(c) = Self::node_index(cp) {
            self.a[(row, c)] -= mu;
        }
        if let Some(c) = Self::node_index(cn) {
            self.a[(row, c)] += mu;
        }
        k
    }

    /// Stamps a capacitor between nodes for AC.
    pub fn stamp_capacitor(&mut self, a: Node, b: Node, cval: Scalar, ctx: AcContext) {
        if ctx.omega == 0.0 {
            return;
        }
        let j = Complex::new(0.0, 1.0);
        let y = j * (ctx.omega * cval);
        match (Self::node_index(a), Self::node_index(b)) {
            (Some(i), Some(jn)) => {
                self.a[(i, i)] += y;
                self.a[(jn, jn)] += y;
                self.a[(i, jn)] -= y;
                self.a[(jn, i)] -= y;
            }
            (Some(i), None) => self.a[(i, i)] += y,
            (None, Some(jn)) => self.a[(jn, jn)] += y,
            (None, None) => {}
        }
    }

    /// Stamps an inductor between nodes for AC.
    pub fn stamp_inductor(&mut self, a: Node, b: Node, lval: Scalar, ctx: AcContext) {
        if ctx.omega == 0.0 {
            let g = Complex::new(1.0e12, 0.0);
            match (Self::node_index(a), Self::node_index(b)) {
                (Some(i), Some(j)) => {
                    self.a[(i, i)] += g;
                    self.a[(j, j)] += g;
                    self.a[(i, j)] -= g;
                    self.a[(j, i)] -= g;
                }
                (Some(i), None) => self.a[(i, i)] += g,
                (None, Some(j)) => self.a[(j, j)] += g,
                (None, None) => {}
            }
            return;
        }
        let j = Complex::new(0.0, 1.0);
        let y = -j / (ctx.omega * lval);
        match (Self::node_index(a), Self::node_index(b)) {
            (Some(i), Some(jn)) => {
                self.a[(i, i)] += y;
                self.a[(jn, jn)] += y;
                self.a[(i, jn)] -= y;
                self.a[(jn, i)] -= y;
            }
            (Some(i), None) => self.a[(i, i)] += y,
            (None, Some(jn)) => self.a[(jn, jn)] += y,
            (None, None) => {}
        }
    }

    /// Solves the MNA system and returns diagnostics.
    #[must_use]
    pub fn solve_with_report(&self) -> (Option<DVector<Complex<Scalar>>>, SolveReport) {
        let mut report = SolveReport::default();
        // Detect floating nodes: rows in the node block with near-zero coefficients.
        let n = self.n;
        for i in 0..n {
            let mut row_norm = 0.0;
            for j in 0..(self.n + self.m) {
                let v = self.a[(i, j)];
                row_norm += v.norm();
            }
            if row_norm <= 1e-14 {
                report.floating_nodes.push(i);
            }
        }

        let lu = self.a.clone().lu();
        // Condition estimate from U diagonal (ratio max/min magnitude).
        let mut max_d = 0.0;
        let mut min_d = f64::INFINITY;
        let u = lu.u();
        let dim = u.nrows().min(u.ncols());
        for k in 0..dim {
            let d = u[(k, k)].norm();
            if d > max_d {
                max_d = d;
            }
            if d < min_d {
                min_d = d;
            }
        }
        if min_d.is_finite() && min_d > 0.0 {
            report.cond_estimate = Some(max_d / min_d);
        }

        let sol = lu.solve(&self.b);
        report.success = sol.is_some();
        if !report.floating_nodes.is_empty() {
            report
                .notes
                .push("floating nodes detected; results may be invalid".into());
        }
        (sol, report)
    }

    /// Solves the MNA system.
    #[must_use]
    pub fn solve(&self) -> Option<DVector<Complex<Scalar>>> {
        let (x, _report) = self.solve_with_report();
        x
    }

    /// Returns (node_count, source_count).
    #[must_use]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    /// Splits a raw solution vector into (node voltages, source currents).
    #[must_use]
    pub fn split_solution(
        &self,
        x: DVector<Complex<Scalar>>,
    ) -> (DVector<Complex<Scalar>>, DVector<Complex<Scalar>>) {
        let (n, m) = (self.n, self.m);
        let v = x.rows(0, n).into_owned();
        let i = if m > 0 { x.rows(n, m).into_owned() } else { DVector::zeros(0) };
        (v, i)
    }

    /// Current-controlled current source (CCCS). Output from `op` to `on` equals `alpha * i_k`.
    /// `ctrl_k` is the index returned by stamping a (dependent or independent) voltage source.
    pub fn stamp_cccs(&mut self, op: Node, on: Node, ctrl_k: usize, alpha: Scalar) {
        let a = Complex::new(alpha, 0.0);
        let col = self.k_idx(ctrl_k);
        if let Some(o) = Self::node_index(op) {
            self.a[(o, col)] += a;
        }
        if let Some(o) = Self::node_index(on) {
            self.a[(o, col)] -= a;
        }
    }

    /// Current-controlled voltage source (CCVS). Enforces `v(op)-v(on) = mu * i_k`.
    /// Returns the index of the new source current variable.
    pub fn stamp_ccvs(&mut self, op: Node, on: Node, ctrl_k: usize, mu: Scalar) -> usize {
        let k = self.m;
        self.ensure_capacity(k + 1);
        let row = self.k_idx(k);

        // KCL coupling for the new source current
        if let Some(p) = Self::node_index(op) {
            self.a[(p, row)] += Complex::new(1.0, 0.0);
            self.a[(row, p)] += Complex::new(1.0, 0.0);
        }
        if let Some(n) = Self::node_index(on) {
            self.a[(n, row)] -= Complex::new(1.0, 0.0);
            self.a[(row, n)] -= Complex::new(1.0, 0.0);
        }

        // Control by existing source current i_k
        let mu = Complex::new(mu, 0.0);
        let col_ctrl = self.k_idx(ctrl_k);
        self.a[(row, col_ctrl)] -= mu;
        k
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dc_resistor_with_current_source() {
        // 1Ω from node 0 to ground, 1A source into node 0 => 1V at node 0.
        let mut b = NodalBuilder::new(1);
        b.stamp_resistor(Some(0), None, 1.0);
        b.stamp_current_source(Some(0), None, Complex::new(1.0, 0.0));
        let na = b.build();
        let v = na.solve().unwrap();
        assert_relative_eq!(v[0].re, 1.0, epsilon = 1e-12);
    }
}

#[cfg(test)]
mod tests_mna {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dc_voltage_divider_with_vsource() {
        // Nodes: n1 (source+), n0 (midpoint)
        // Vsrc 10V between n1 and GND; R1 between n1-n0 (1k), R2 n0-GND (2k)
        // Expect V(n0) = 10 * 2k/(1k+2k) = 6.666666...
        let mut mna = MnaBuilder::new(2);
        mna.stamp_voltage_source(Some(0), None, Complex::new(10.0, 0.0)); // n1 = node 0
        mna.stamp_resistor(Some(0), Some(1), 1_000.0);
        mna.stamp_resistor(Some(1), None, 2_000.0);
        let (x, _r) = mna.solve_with_report();
        let x = x.expect("solution");
        let v_n0 = x[1].re;
        assert_relative_eq!(v_n0, 10.0 * 2000.0 / 3000.0, epsilon = 1e-9);
    }

    #[test]
    fn vccs_biases_node_as_expected() {
        // 1V at control port produces g*1 A into output node across 1k => 1V.
        let mut mna = MnaBuilder::new(2);
        // Control: v(cp)=1V relative to ground => use voltage source at node 0
        mna.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        // Output node is node 1 with 1k to ground
        mna.stamp_resistor(Some(1), None, 1_000.0);
        // VCCS from op=node1 to on=GND, controlled by cp=node0 to cn=GND, g=1e-3 S
        mna.stamp_vccs(Some(1), None, Some(0), None, 1e-3);
        let (x, _r) = mna.solve_with_report();
        let x = x.expect("solution");
        assert_relative_eq!(x[1].re, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn cccs_tracks_controlling_source_current() {
        // Sense current through a 1V source feeding 1k => 1mA.
        // CCCS mirrors that current into node1. With 1k to ground => 1V at node1.
        let mut mna = MnaBuilder::new(2);
        let k = mna.stamp_voltage_source(Some(0), None, Complex::new(1.0, 0.0));
        mna.stamp_resistor(Some(0), None, 1_000.0);
        mna.stamp_resistor(Some(1), None, 1_000.0);
        mna.stamp_cccs(Some(1), None, k, 1.0);
        let (x, _r) = mna.solve_with_report();
        let x = x.expect("solution");
        assert_relative_eq!(x[1].re, 1.0, epsilon = 1e-9);
    }
}
