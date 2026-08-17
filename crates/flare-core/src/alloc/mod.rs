//! Contiguous memory allocation primitives.
//!
//! This module hosts the [`FlatArena`] bump allocator and the 4 KB
//! size-classified [`SlabPool`]. Per the memory ownership constraint, nodes
//! are addressed with 40-bit relative offsets and raw virtual pointers are
//! never stored inside tree nodes.
//! # Unsafe Confinement
//!
//! This is the primary `unsafe`-confined module of the crate. All `unsafe`
//! here is confined to two shapes, each with a documented invariant proof:
//!
//! - Raw access to the arena backing store, justified by the disjoint
//!   region ownership rule enforced by the bump allocator.
//! - The lock-free slab freelist, justified by exclusive slot ownership
//!   transferred at pop time.
//! - The readonly pinned host blocks of [`pinned`], justified by exclusive
//!   block ownership transferred at allocation time.
#![allow(unsafe_code)]

pub mod arena;
pub mod pinned;
pub mod slab;

pub use arena::{ArenaRef, FlatArena};
pub use pinned::{allocate_pinned_block, deallocate_pinned_block};
pub use slab::{SlabClass, SlabPool, SlabSlot};
