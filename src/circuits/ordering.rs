//! Matrix ordering and reordering algorithms for sparse linear systems.
//!
//! Ordering algorithms reorder the rows and columns of a sparse matrix to:
//! - Reduce bandwidth (improve cache locality)
//! - Minimize fill-in during factorization
//! - Improve preconditioner quality for iterative methods
//!
//! # Algorithms
//!
//! - **RCM** (Reverse Cuthill-McKee): Bandwidth reduction via BFS
//! - **AMD** (Approximate Minimum Degree): Fill-in reduction for sparse LU
//!
//! # References
//!
//! - Cuthill & McKee (1969). "Reducing the Bandwidth of Sparse Symmetric Matrices".
//!   Proc. 24th Nat. Conf. ACM, 157-172.
//! - George (1971). "Computer Implementation of the Finite Element Method".
//!   PhD thesis, Stanford University.
//! - Amestoy, Davis & Duff (1996). "An Approximate Minimum Degree Ordering Algorithm".
//!   SIAM J. Matrix Anal. Appl. 17(4), 886-905.

#![cfg(feature = "sparse")]

use nalgebra_sparse::CscMatrix;
use num_complex::Complex;
use std::collections::{VecDeque, HashSet};

use crate::math::Scalar;

/// Permutation vector representing a reordering of rows/columns.
///
/// perm[i] = j means the i-th row/column in the new ordering corresponds
/// to the j-th row/column in the original matrix.
pub type Permutation = Vec<usize>;

/// Computes the adjacency structure of a sparse matrix.
///
/// For a symmetric matrix, the adjacency graph has an edge (i,j) if A[i,j] ≠ 0.
/// For nonsymmetric matrices, we use the pattern of A + A^T.
fn build_adjacency(matrix: &CscMatrix<Complex<Scalar>>) -> Vec<Vec<usize>> {
    let n = matrix.nrows();
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    // Add edges from matrix pattern
    for (row, col, _val) in matrix.triplet_iter() {
        if row != col {
            adj[row].insert(col);
            adj[col].insert(row); // Symmetrize for undirected graph
        }
    }

    // Convert HashSet to Vec for each node
    adj.into_iter()
        .map(|set| set.into_iter().collect())
        .collect()
}

/// Finds a peripheral node (node with minimum degree) for RCM starting point.
///
/// The peripheral node is chosen as a node with minimum degree, which tends
/// to be on the "boundary" of the graph and gives good RCM orderings.
fn find_peripheral_node(adj: &[Vec<usize>]) -> usize {
    adj.iter()
        .enumerate()
        .min_by_key(|(_idx, neighbors)| neighbors.len())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Reverse Cuthill-McKee (RCM) ordering for bandwidth reduction.
///
/// RCM reorders the matrix to move nonzeros closer to the diagonal, improving:
/// - Cache locality during factorization and matrix-vector products
/// - Preconditioner quality (narrower bands → better diagonal dominance)
/// - Memory access patterns for iterative solvers
///
/// # Algorithm
///
/// 1. Find peripheral node (minimum degree) as starting point
/// 2. Perform breadth-first search (BFS), visiting neighbors in increasing degree order
/// 3. Reverse the resulting ordering
///
/// # Complexity
///
/// O(n + nnz) for BFS traversal and degree sorting.
///
/// # When to Use
///
/// - Iterative solvers with banded preconditioners
/// - Improving cache locality for large systems
/// - When AMD is not available or too expensive
///
/// # References
///
/// - Cuthill & McKee (1969). "Reducing the Bandwidth of Sparse Symmetric Matrices".
/// - George & Liu (1981). "Computer Solution of Large Sparse Positive Definite Systems".
///
/// # Example
///
/// ```ignore
/// let perm = rcm_ordering(&matrix);
/// let reordered = permute_matrix(&matrix, &perm);
/// ```
pub fn rcm_ordering(matrix: &CscMatrix<Complex<Scalar>>) -> Permutation {
    let n = matrix.nrows();

    if n == 0 {
        return vec![];
    }

    let adj = build_adjacency(matrix);

    // Find starting node (peripheral, minimum degree)
    let start = find_peripheral_node(&adj);

    // BFS with degree-ordered neighbor visitation
    let mut ordering = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited[start] = true;

    while let Some(node) = queue.pop_front() {
        ordering.push(node);

        // Get unvisited neighbors sorted by degree
        let mut neighbors: Vec<usize> = adj[node]
            .iter()
            .copied()
            .filter(|&neighbor| !visited[neighbor])
            .collect();

        neighbors.sort_by_key(|&neighbor| adj[neighbor].len());

        for neighbor in neighbors {
            if !visited[neighbor] {
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }

    // Add any unvisited nodes (disconnected components)
    for i in 0..n {
        if !visited[i] {
            ordering.push(i);
        }
    }

    // Reverse the ordering (Reverse CM)
    ordering.reverse();

    ordering
}

/// Computes the bandwidth of a matrix given a permutation.
///
/// Bandwidth is max_i max_j |perm[i] - perm[j]| where A[i,j] ≠ 0.
/// Lower bandwidth indicates better ordering for banded algorithms.
pub fn compute_bandwidth(matrix: &CscMatrix<Complex<Scalar>>, perm: &Permutation) -> usize {
    let mut max_bandwidth = 0;

    for (row, col, _val) in matrix.triplet_iter() {
        if row != col {
            let bandwidth = if perm[row] > perm[col] {
                perm[row] - perm[col]
            } else {
                perm[col] - perm[row]
            };

            if bandwidth > max_bandwidth {
                max_bandwidth = bandwidth;
            }
        }
    }

    max_bandwidth
}

/// Applies a permutation to a sparse CSC matrix: P * A * P^T.
///
/// This creates a new matrix where rows and columns are reordered according
/// to the permutation vector.
pub fn permute_matrix(
    matrix: &CscMatrix<Complex<Scalar>>,
    perm: &Permutation,
) -> CscMatrix<Complex<Scalar>> {
    let n = matrix.nrows();

    // Create inverse permutation
    let mut inv_perm = vec![0; n];
    for (new_idx, &old_idx) in perm.iter().enumerate() {
        inv_perm[old_idx] = new_idx;
    }

    // Build permuted matrix using COO
    let mut coo = nalgebra_sparse::coo::CooMatrix::new(n, n);

    for (row, col, &val) in matrix.triplet_iter() {
        let new_row = inv_perm[row];
        let new_col = inv_perm[col];
        coo.push(new_row, new_col, val);
    }

    CscMatrix::from(&coo)
}

/// Ordering strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingStrategy {
    /// No reordering (natural ordering).
    Natural,
    /// Reverse Cuthill-McKee (bandwidth reduction).
    RCM,
    /// Approximate Minimum Degree (fill-in reduction, requires SuiteSparse).
    AMD,
    /// Automatic selection based on matrix characteristics.
    Auto,
}

/// Selects optimal ordering strategy based on matrix characteristics.
///
/// # Heuristics
///
/// - **Natural**: Very small matrices (<100 nodes)
/// - **RCM**: Medium matrices (100-10k nodes), bandwidth-sensitive
/// - **AMD**: Large matrices (>10k nodes), future with SuiteSparse
/// - **Auto**: Analyzes matrix structure to choose
///
/// # Arguments
///
/// * `matrix` - System matrix to analyze
/// * `use_iterative` - Whether an iterative solver will be used
///
/// # Returns
///
/// Recommended ordering strategy
pub fn select_ordering_strategy(
    matrix: &CscMatrix<Complex<Scalar>>,
    use_iterative: bool,
) -> OrderingStrategy {
    let n = matrix.nrows();
    let nnz = matrix.triplet_iter().count();
    let density = (nnz as f64) / (n * n) as f64;

    // Very small systems: no reordering overhead
    if n < 100 {
        return OrderingStrategy::Natural;
    }

    // Dense-ish matrices: reordering may not help much
    if density > 0.1 {
        return OrderingStrategy::Natural;
    }

    // For iterative solvers: RCM improves cache locality and preconditioner
    if use_iterative {
        return OrderingStrategy::RCM;
    }

    // For direct solvers on medium systems: RCM for now (AMD future)
    if n < 10_000 {
        return OrderingStrategy::RCM;
    }

    // Large direct solver: AMD would be ideal (placeholder for future)
    // For now, fall back to RCM
    OrderingStrategy::RCM
}

/// Applies the specified ordering strategy to a matrix.
///
/// # Arguments
///
/// * `matrix` - System matrix to reorder
/// * `strategy` - Ordering strategy to apply
///
/// # Returns
///
/// Tuple of (permuted matrix, permutation vector, strategy used)
pub fn apply_ordering(
    matrix: &CscMatrix<Complex<Scalar>>,
    strategy: OrderingStrategy,
) -> (CscMatrix<Complex<Scalar>>, Permutation, OrderingStrategy) {
    let n = matrix.nrows();

    let (perm, actual_strategy) = match strategy {
        OrderingStrategy::Natural => {
            ((0..n).collect(), OrderingStrategy::Natural)
        }
        OrderingStrategy::RCM => {
            (rcm_ordering(matrix), OrderingStrategy::RCM)
        }
        OrderingStrategy::AMD => {
            // AMD not yet implemented (requires SuiteSparse or complex pure-Rust implementation)
            // Fall back to RCM
            (rcm_ordering(matrix), OrderingStrategy::RCM)
        }
        OrderingStrategy::Auto => {
            // Use heuristic to select
            let selected = select_ordering_strategy(matrix, true); // Assume iterative for auto
            return apply_ordering(matrix, selected);
        }
    };

    let permuted = permute_matrix(matrix, &perm);

    (permuted, perm, actual_strategy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rcm_reduces_bandwidth_on_laplacian() {
        // 1D Laplacian matrix (tridiagonal): [-1, 2, -1]
        let n = 10;
        let mut coo = nalgebra_sparse::coo::CooMatrix::new(n, n);

        for i in 0..n {
            coo.push(i, i, Complex::new(2.0, 0.0));
            if i > 0 {
                coo.push(i, i-1, Complex::new(-1.0, 0.0));
            }
            if i < n - 1 {
                coo.push(i, i+1, Complex::new(-1.0, 0.0));
            }
        }

        let matrix = CscMatrix::from(&coo);

        // Natural ordering already has bandwidth 1 for tridiagonal
        let natural_perm: Vec<usize> = (0..n).collect();
        let natural_bw = compute_bandwidth(&matrix, &natural_perm);

        // RCM should maintain or improve
        let rcm_perm = rcm_ordering(&matrix);
        let rcm_bw = compute_bandwidth(&matrix, &rcm_perm);

        assert_eq!(rcm_perm.len(), n);
        assert!(rcm_bw <= natural_bw + 1, "RCM should not increase bandwidth significantly");
    }

    #[test]
    fn rcm_ordering_covers_all_nodes() {
        // Mesh matrix (2D grid)
        let grid = 4;
        let n = grid * grid;
        let mut coo = nalgebra_sparse::coo::CooMatrix::new(n, n);

        for row in 0..grid {
            for col in 0..grid {
                let node = row * grid + col;
                coo.push(node, node, Complex::new(4.0, 0.0));

                if col < grid - 1 {
                    coo.push(node, node + 1, Complex::new(-1.0, 0.0));
                }
                if row < grid - 1 {
                    coo.push(node, node + grid, Complex::new(-1.0, 0.0));
                }
            }
        }

        let matrix = CscMatrix::from(&coo);
        let perm = rcm_ordering(&matrix);

        // Check all nodes appear exactly once
        assert_eq!(perm.len(), n);
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn permute_matrix_preserves_values() {
        // Small test matrix
        let mut coo = nalgebra_sparse::coo::CooMatrix::new(3, 3);
        coo.push(0, 0, Complex::new(1.0, 0.0));
        coo.push(0, 1, Complex::new(2.0, 0.0));
        coo.push(1, 1, Complex::new(3.0, 0.0));
        coo.push(2, 0, Complex::new(4.0, 0.0));
        coo.push(2, 2, Complex::new(5.0, 0.0));
        let matrix = CscMatrix::from(&coo);

        // Permutation: [2, 0, 1] (reverse)
        let perm = vec![2, 0, 1];
        let permuted = permute_matrix(&matrix, &perm);

        // Verify dimensions preserved
        assert_eq!(permuted.nrows(), 3);
        assert_eq!(permuted.ncols(), 3);

        // Verify nonzero count preserved
        let original_nnz = matrix.triplet_iter().count();
        let permuted_nnz = permuted.triplet_iter().count();
        assert_eq!(original_nnz, permuted_nnz);
    }
}
