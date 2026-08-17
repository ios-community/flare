//! Bitwise pointer primitives.
//!
//! This module hosts the 64-bit polymorphic [`TaggedPointer`] encoding that
//! replaces raw virtual pointers across FLARE. Nodes are addressed by a
//! 40-bit relative offset plus a 4-bit arena instance identifier; the
//! remaining bits carry node type, tombstone status, and polymorphic
//! metadata such as child-presence bitmaps.

pub mod tagged;

pub use tagged::{NodeType, TaggedPointer, resolve_child_index};
