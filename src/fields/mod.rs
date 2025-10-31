//! Electromagnetic field representations and helper utilities.

mod electric;
mod magnetic;
mod sources;

pub use electric::{ElectricField, ElectricFieldKind};
pub use magnetic::MagneticField;
pub use sources::{
    electric_field_from_vector_potential, magnetic_field_from_lines, magnetic_field_segment,
    vector_potential_from_lines, vector_potential_segment, LineCurrent, WireSegment3D,
};
