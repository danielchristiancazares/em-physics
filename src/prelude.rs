//! Convenience re-exports for building electromagnetism experiments.

pub use crate::circuits::{
    analysis::{AdmittanceMatrix, NodalAnalysis},
    component::{Capacitor, Component, Inductor, Resistor, VoltageSource},
    network::{ConnectionKind, Network},
    nport::{
        HermitianPassivityReport, HermitianPassivityResult, NPortError, NPortNetwork,
        PassivityReport, PassivityResult, ReferenceImpedance, ReferenceImpedanceError,
        read_touchstone, read_touchstone_checked,
    },
    stamp::{AcContext, MnaBuilder, NodalBuilder},
    transmission::TransmissionLine,
    twoport::TwoPort,
    vf::{RationalModel, VFConfig, VFError, VFResult, fit as vf_fit},
};
pub use crate::constants::*;
pub use crate::errors::EmPhysicsError;
pub use crate::fields::{ElectricField, ElectricFieldKind, MagneticField};
pub use crate::fields::{
    FlatPanel, PointCharge, TimeLineCurrent, build_double_layer_matrix, build_single_layer_matrix,
    electric_field_from_point_charges, electric_field_from_retarded_potential,
    electric_field_from_single_layer, electric_field_from_uniform_patch,
    potential_from_point_charges, potential_from_single_layer, potential_from_uniform_patch,
    solve_dirichlet_single_layer, vector_potential_retarded,
};
pub use crate::fields::{
    LineCurrent, WireSegment3D, electric_field_from_vector_potential, magnetic_field_from_lines,
    vector_potential_from_lines,
};
pub use crate::materials::{DispersiveMaterial, DrudeModel, MaterialProperties};
pub use crate::math::{
    NaturalCubicSpline, ParametricSpline2, ParametricSpline3, R2, R3, R3x3, Scalar, SplineError,
    phasor, phasor_magnitude, sinusoid_rms,
};
pub use crate::simulation::{
    MnaTransientEngine, SimulationConfig, SimulationDomain, SimulationEngine, SimulationError,
    TimeIntegrator, TransientWaveform, write_transient_node_csv,
    write_transient_vsource_current_csv,
};
pub use crate::sweep::{
    angular_freq_linspace, angular_freq_logspace, linspace, logspace_hz, mag, mag_db, phase_deg,
    sweep_map,
};
pub use crate::units::{Ampere, Current, Farad, Henry, Impedance, Quantity, Unit, Volt, Voltage};
