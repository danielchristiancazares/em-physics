//! Optional sparse MNA types and helpers, gated behind `sparse` feature.

#![cfg(feature = "sparse")]

use nalgebra::DVector;
use nalgebra_sparse::{coo::CooMatrix, CscMatrix};
use num_complex::Complex;

use crate::math::Scalar;

/// Minimal sparse MNA container (coo builder -> csc system).
pub struct SparseMnaBuilder {
    n: usize,
    coo: CooMatrix<Complex<Scalar>>,
    rhs: DVector<Complex<Scalar>>,
}

impl SparseMnaBuilder {
    /// Creates a sparse builder with `node_count` nodes and an initial capacity.
    pub fn new(node_count: usize, nnz_hint: usize) -> Self {
        Self {
            n: node_count,
            coo: CooMatrix::new(node_count, node_count),
            rhs: DVector::zeros(node_count),
        }
    }

    /// Adds a value at (i,j).
    pub fn add(&mut self, i: usize, j: usize, val: Complex<Scalar>) {
        self.coo.push(i, j, val);
    }

    /// Adds to RHS at i.
    pub fn add_rhs(&mut self, i: usize, val: Complex<Scalar>) {
        self.rhs[i] += val;
    }

    /// Finalizes into a CSC system. Note: solver not yet provided.
    pub fn finalize(self) -> (CscMatrix<Complex<Scalar>>, DVector<Complex<Scalar>>) {
        (CscMatrix::from(&self.coo), self.rhs)
    }
}

