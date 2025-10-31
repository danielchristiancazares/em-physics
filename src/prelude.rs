//! Convenience re-exports for building electromagnetism experiments.

pub use crate::circuits::{
    analysis::{AdmittanceMatrix, NodalAnalysis},
    component::{Capacitor, Component, Inductor, Resistor, VoltageSource},
    network::{ConnectionKind, Network},
    transmission::TransmissionLine,
    twoport::TwoPort,
    stamp::{AcContext, MnaBuilder, NodalBuilder},
};
pub use crate::constants::*;
pub use crate::errors::EmPhysicsError;
pub use crate::fields::{ElectricField, ElectricFieldKind, MagneticField};
pub use crate::fields::{
    electric_field_from_vector_potential, magnetic_field_from_lines, vector_potential_from_lines,
    LineCurrent, WireSegment3D,
};
pub use crate::fields::{
    FlatPanel,
    build_single_layer_matrix,
    build_double_layer_matrix,
    potential_from_single_layer,
    electric_field_from_single_layer,
    solve_dirichlet_single_layer,
    PointCharge,
    potential_from_point_charges,
    electric_field_from_point_charges,
    potential_from_uniform_patch,
    electric_field_from_uniform_patch,
    TimeLineCurrent,
    vector_potential_retarded,
    electric_field_from_retarded_potential,
};
pub use crate::materials::{DispersiveMaterial, DrudeModel, MaterialProperties};
pub use crate::math::{phasor, phasor_magnitude, sinusoid_rms, R3, R3x3, Scalar};
pub use crate::simulation::{
    MnaTransientEngine,
    SimulationConfig,
    SimulationDomain,
    SimulationEngine,
    SimulationError,
    TimeIntegrator,
    TransientWaveform,
    write_transient_node_csv,
    write_transient_vsource_current_csv,
};
pub use crate::units::{
    Ampere, Current, Farad, Henry, Impedance, Quantity, Unit, Volt, Voltage,
};
pub use crate::sweep::{
    angular_freq_linspace, angular_freq_logspace, linspace, logspace_hz, mag, mag_db, phase_deg,
    sweep_map,
};
