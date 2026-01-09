//! Electromagnetic field representations and helper utilities.

mod bem;
mod electric;
mod electrostatic;
mod magnetic;
mod retarded;
mod sources;

pub use bem::{
    FlatPanel, build_double_layer_matrix, build_single_layer_matrix,
    electric_field_from_single_layer, potential_from_single_layer, solve_dirichlet_single_layer,
};
pub use electric::{ElectricField, ElectricFieldKind};
pub use electrostatic::{
    PointCharge, electric_field_from_point_charges, electric_field_from_uniform_patch,
    potential_from_point_charges, potential_from_uniform_patch,
};
pub use magnetic::MagneticField;
pub use retarded::{
    TimeLineCurrent, electric_field_from_retarded_potential, vector_potential_retarded,
};
pub use sources::{
    LineCurrent, WireSegment3D, electric_field_from_vector_potential, magnetic_field_from_lines,
    magnetic_field_segment, vector_potential_from_lines, vector_potential_segment,
};
