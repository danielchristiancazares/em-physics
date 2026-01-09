# Multiport Network Synthesis - Roadmap

This document outlines the planned features for multiport network synthesis in em-physics.

## Completed Features

### Passivity Validation (v0.2.x)

SVD-based passivity checking for N-port S-parameter networks. A network is passive if all singular values of its S-parameter matrix are <= 1 at all frequencies.

- `PassivityResult` and `PassivityReport` structs for detailed analysis
- `NPortNetwork::check_passivity()` and `is_passive()` methods
- `read_touchstone_checked()` for Touchstone import with passivity validation
- Tolerance handling for numerical precision (1e-12)

## Planned Features

### 1. Y/Z/ABCD Conversions for N-Port

Generalize the existing TwoPort conversion patterns to arbitrary N-port networks.

**Matrix formulas:**
- Y = (I - S)(I + S)^{-1} / Z0
- Z = Z0(I + S)(I - S)^{-1}
- Handle reference impedance normalization for mixed-impedance ports

**Implementation:**
- Add `NPortNetwork::to_y()`, `to_z()` methods
- Support per-port reference impedances (generalized S-parameters)

### 2. Passivity Enforcement

Perturb non-passive S-parameter data to restore physical realizability.

**Approaches:**
- Eigenvalue clipping: constrain singular values to <= 1
- Perturbation methods: minimize ||S_perturbed - S_original||
- Hamiltonian-based convex optimization (requires SDP solver)

**References:**
- Gustavsen, B., & Semlyen, A. (1999). "Enforcing Passivity for Admittance Matrices Approximated by Rational Functions". IEEE Trans. Power Systems.

### 3. Rational Fitting (Vector Fitting)

Approximate frequency-domain S/Y/Z data with pole-residue rational functions for efficient time-domain simulation.

**Core algorithm:**
- Pole-residue representation: H(s) = sum_k(r_k / (s - p_k)) + d + s*e
- VF iteration: pole relocation via linear least squares
- Residue identification after pole convergence

**Extensions:**
- Passivity-preserving vector fitting
- Time-domain macromodel generation for transient analysis
- Integration with `MnaTransientEngine`

**References:**
- Gustavsen, B., & Semlyen, A. (1999). "Rational Approximation of Frequency Domain Responses by Vector Fitting". IEEE Trans. Power Delivery.

## Future Considerations

- Sparse rational fitting for large port counts
- Frequency-dependent reference impedances
- Mixed S/Y/Z parameter cascading
- IBIS-AMI model integration
