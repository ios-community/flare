//! Adaptive radix tree nodes and the lock-free `FlareArtTree` index.
//!
//! This module implements the four adaptive node families (`Node4`,
//! `Node16`, `Node64`, `Node256`) sized to fit the 4 KB slab slot classes,
//! and the [`FlareArtTree`] index that addresses them through 40-bit arena
//! offsets. Child resolution uses hardware-accelerated popcount via
//! [`resolve_child_index`](crate::ptr::resolve_child_index).
//!
//! # Key Encoding
//!
//! Keys are consumed nibble-by-nibble: the high nibble of byte `k[d >> 1]`
//! at even depths, the low nibble at odd depths. `Node4` and `Node16` store
//! up to 4 and 16 nibble-keyed children with a 16-bit presence bitmap in
//! the parent's polymorphic field; `Node64` keeps a 64-bit internal bitmap
//! (16 bits used by the nibble domain) and `Node256` indexes children
//! directly by nibble position. `Node64`/`Node256` encode an 8-bit active
//! reference count plus an 8-bit generation identifier in the polymorphic
//! field for ABA protection. The full-byte indexing optimisation for
//! `Node256` is documented as a follow-up in the node-resizing milestone.

#[allow(clippy::module_inception)]
pub mod tree;

pub use tree::{FlareArtTree, Node4, Node16, Node64, Node256};

/// Alias of [`NodeType`](crate::ptr::NodeType) re-exported from the tree
/// module for ergonomics.
pub use crate::ptr::NodeType as NodeKind;

/// Marker for an empty child slot or an absent leaf (`u64::MAX`).
///
/// The value decodes as node type `7`, which is reserved and never packed
/// by [`TaggedPointer::pack`](crate::ptr::TaggedPointer::pack).
#[doc(hidden)]
pub const EMPTY_CHILD: u64 = u64::MAX;

/// Converts a byte key into the nibble sequence consumed by the radix tree.
///
/// Each byte contributes its high nibble first, then its low nibble, so the
/// returned sequence has twice the length of the input.
///
/// # Examples
///
/// ```
/// # use flare_core::tree::key_nibbles;
/// assert_eq!(key_nibbles(&[0xAB]), vec![0xA, 0xB]);
/// assert_eq!(key_nibbles(&[]), Vec::<u8>::new());
/// ```
#[must_use]
pub fn key_nibbles(key: &[u8]) -> alloc_crate::vec::Vec<u8> {
    let mut nibbles = alloc_crate::vec::Vec::with_capacity(key.len() * 2);
    for byte in key {
        nibbles.push(byte >> 4);
        nibbles.push(byte & 0x0F);
    }
    nibbles
}

#[cfg(test)]
mod tests {
    use super::key_nibbles;
    use alloc_crate::vec;
    use alloc_crate::vec::Vec;

    /// Verifies nibble splitting covers all byte values.
    #[test]
    fn key_nibbles_split() {
        assert_eq!(key_nibbles(&[0x00]), vec![0, 0]);
        assert_eq!(key_nibbles(&[0xFF]), vec![0xF, 0xF]);
        assert_eq!(key_nibbles(&[0x12, 0x34]), vec![1, 2, 3, 4]);
        assert_eq!(key_nibbles(&[]), Vec::<u8>::new());
    }
}
