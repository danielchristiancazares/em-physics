//! Electromagnetic field representations and helper utilities.

mod electric;
mod magnetic;
mod sources;
mod bem;
mod electrostatic;
mod retarded;

pub use electric::{ElectricField, ElectricFieldKind};
pub use magnetic::MagneticField;
pub use sources::{
    electric_field_from_vector_potential, magnetic_field_from_lines, magnetic_field_segment,
    vector_potential_from_lines, vector_potential_segment, LineCurrent, WireSegment3D,
};
pub use bem::{
    FlatPanel,
    build_single_layer_matrix,
    build_double_layer_matrix,
    potential_from_single_layer,
    electric_field_from_single_layer,
    solve_dirichlet_single_layer,
};
pub use electrostatic::{
    PointCharge,
    potential_from_point_charges,
    electric_field_from_point_charges,
    potential_from_uniform_patch,
    electric_field_from_uniform_patch,
};
pub use retarded::{
    TimeLineCurrent,
    vector_potential_retarded,
    electric_field_from_retarded_potential,
};
