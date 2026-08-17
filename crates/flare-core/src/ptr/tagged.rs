//! 64-bit polymorphic tagged pointer encoding.
//!
//! The [`TaggedPointer`] word replaces raw virtual pointers inside FLARE
//! tree nodes. Its bitwise layout is defined by the architecture
//! specification as follows:
//!
//! ```text
//!  63             48 47 46    43 42                                 3 2      0
//! +-----------------+--+--------+------------------------------------+--------+
//! | Polymorphic     | T| Arena  |       Relative Array Offset        |  Node  |
//! | Field (16-bit)  |  | ID(4b) |          (40-bit Index)            |  Type  |
//! +-----------------+--+--------+------------------------------------+--------+
//! ```
//!
//! Bit 47 is the isolated Tombstone flag: it is never overwritten by data
//! payloads, even when the remaining 56 bits carry an inlined leaf value.

/// Identifies the concrete node structure addressed by a [`TaggedPointer`].
///
/// The discriminants map directly onto bits `0..2` of the tagged pointer
/// word. Values `6` and `7` are reserved for future node families and must
/// not be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NodeType {
    /// A leaf whose 56-bit payload is inlined inside the pointer word.
    ///
    /// The value is recovered via [`TaggedPointer::unpack_inline_payload`]
    /// after zero-extending the truncated 8 most significant bits.
    LeafInlined = 0,
    /// A leaf whose 64-bit value resides in a dedicated arena slot.
    ///
    /// The offset field addresses the value slot inside the owning arena.
    LeafOffset = 1,
    /// An adaptive radix node holding up to 4 byte-keyed children.
    Node4 = 2,
    /// An adaptive radix node holding up to 16 byte-keyed children.
    Node16 = 3,
    /// An adaptive radix node holding up to 64 byte-keyed children.
    Node64 = 4,
    /// An adaptive radix node holding up to 256 byte-keyed children.
    Node256 = 5,
}

impl NodeType {
    /// Returns the 3-bit discriminant of this node type.
    ///
    /// The value occupies bits `0..2` of a tagged pointer word.
    #[must_use]
    pub const fn discriminant(self) -> u8 {
        self as u8
    }

    /// Converts a raw 3-bit discriminant into a [`NodeType`].
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidNodeType`](crate::FlareError::InvalidNodeType)
    /// when `raw` is greater than `5`, since values `6` and `7` are reserved.
    pub const fn from_discriminant(raw: u8) -> Result<Self, crate::FlareError> {
        match raw & 0x07 {
            0 => Ok(Self::LeafInlined),
            1 => Ok(Self::LeafOffset),
            2 => Ok(Self::Node4),
            3 => Ok(Self::Node16),
            4 => Ok(Self::Node64),
            5 => Ok(Self::Node256),
            _ => Err(crate::FlareError::InvalidNodeType(raw & 0x07)),
        }
    }
}

/// A 64-bit polymorphic tagged pointer addressing nodes inside a [`FlatArena`](crate::alloc::arena::FlatArena).
///
/// The word packs a 3-bit node type, a 40-bit relative arena offset, a 4-bit
/// arena instance identifier, an isolated 1-bit tombstone flag, and a 16-bit
/// polymorphic metadata field. The polymorphic field is interpreted
/// according to the node type:
///
/// - [`NodeType::Node4`] and [`NodeType::Node16`]: a 16-bit inline child
///   presence bitmap used for popcount child resolution.
/// - [`NodeType::Node64`] and [`NodeType::Node256`]: an 8-bit active
///   reference count plus an 8-bit generation identifier for ABA
///   protection.
/// - KV-cache nodes: a 16-bit token sequence length.
///
/// All packing and unpacking routines are `const` and execute in `O(1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TaggedPointer(pub u64);

impl TaggedPointer {
    /// Bitmask isolating the 3-bit node type field.
    pub const MASK_TYPE: u64 = 0x0000_0000_0000_0007;
    /// Bitmask isolating the 40-bit relative array offset field.
    pub const MASK_OFFSET: u64 = 0x0000_07FF_FFFF_FFF8;
    /// Bitmask isolating the 4-bit arena instance identifier field.
    pub const MASK_ARENA: u64 = 0x0000_7800_0000_0000;
    /// Bitmask isolating the tombstone flag at bit 47.
    pub const MASK_TOMBSTONE: u64 = 0x0000_8000_0000_0000;
    /// Bitmask isolating the 16-bit polymorphic metadata field.
    pub const MASK_POLY: u64 = 0xFFFF_0000_0000_0000;

    /// Packs a tagged pointer word from its component fields.
    ///
    /// `offset` is truncated to its low 40 bits, `arena_id` to its low
    /// 4 bits, and `node_type` to its low 3 bits. The tombstone flag is
    /// placed in its isolated bit 47 position.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::ptr::{NodeType, TaggedPointer};
    /// let ptr = TaggedPointer::pack(NodeType::Node16, 4096, 1, false, 0b1100);
    /// assert_eq!(ptr.node_type(), NodeType::Node16);
    /// assert_eq!(ptr.offset(), 4096);
    /// assert_eq!(ptr.arena_id(), 1);
    /// assert!(!ptr.is_tombstone());
    /// assert_eq!(ptr.polymorphic_field(), 0b1100);
    /// ```
    #[inline]
    #[must_use]
    pub const fn pack(
        node_type: NodeType,
        offset: u64,
        arena_id: u8,
        tombstone: bool,
        poly: u16,
    ) -> Self {
        let t_bit = if tombstone { 1u64 } else { 0u64 };
        let raw = ((poly as u64) << 48)
            | (t_bit << 47)
            | (((arena_id as u64) & 0x0F) << 43)
            | ((offset & 0x00FF_FFFF_FFFF) << 3)
            | ((node_type as u64) & 0x07);
        Self(raw)
    }

    /// Extracts the [`NodeType`] field of this pointer.
    ///
    /// The value is derived from bits `0..2` and never panics; malformed
    /// discriminants must be rejected by the caller through
    /// [`NodeType::from_discriminant`].
    #[inline]
    #[must_use]
    pub const fn node_type(self) -> NodeType {
        // The raw discriminant is guaranteed to decode because the word
        // packs `type & 0x07` and values 6..7 only arise from corrupted
        // memory; decode failure is handled by the caller-facing API.
        match NodeType::from_discriminant((self.0 & Self::MASK_TYPE) as u8) {
            Ok(kind) => kind,
            Err(_) => NodeType::LeafInlined,
        }
    }

    /// Extracts the 40-bit relative arena offset of this pointer.
    ///
    /// Bits `3..42` are shifted down; the result spans `[0, 2^40 - 1]`.
    #[inline]
    #[must_use]
    pub const fn offset(self) -> u64 {
        (self.0 & Self::MASK_OFFSET) >> 3
    }

    /// Extracts the 4-bit arena instance identifier of this pointer.
    #[inline]
    #[must_use]
    pub const fn arena_id(self) -> u8 {
        ((self.0 & Self::MASK_ARENA) >> 43) as u8
    }

    /// Reports whether this pointer carries a logical-deletion tombstone.
    ///
    /// Bit 47 is reserved exclusively for tombstone routing; inlined leaf
    /// payloads bypass it (see [`Self::pack_inline_payload`]).
    #[inline]
    #[must_use]
    pub const fn is_tombstone(self) -> bool {
        (self.0 & Self::MASK_TOMBSTONE) != 0
    }

    /// Extracts the 16-bit polymorphic metadata field of this pointer.
    ///
    /// The interpretation depends on [`Self::node_type`]: child presence
    /// bitmap for `Node4`/`Node16`, reference count plus generation for
    /// `Node64`/`Node256`, and token sequence length for KV-cache nodes.
    #[inline]
    #[must_use]
    pub const fn polymorphic_field(self) -> u16 {
        (self.0 >> 48) as u16
    }

    /// Packs a 56-bit extended leaf payload around the isolated tombstone bit.
    ///
    /// The payload's low 44 bits occupy bits `3..46`, the high 12 bits
    /// occupy bits `48..59`, and bits `60..63` stay reserved. The tombstone
    /// flag at bit 47 is written independently and can never be overwritten
    /// by payload data. Callers packing 64-bit identifiers apply 8-bit MSB
    /// truncation: the low 56 bits of the identifier are stored raw.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::ptr::TaggedPointer;
    /// let ptr = TaggedPointer::pack_inline_payload(u64::from(u32::MAX), false);
    /// assert_eq!(ptr.node_type(), flare_core::ptr::NodeType::LeafInlined);
    /// assert_eq!(ptr.unpack_inline_payload(), u64::from(u32::MAX));
    /// assert!(!ptr.is_tombstone());
    /// ```
    #[inline]
    #[must_use]
    pub const fn pack_inline_payload(payload_56: u64, tombstone: bool) -> Self {
        let t_bit = if tombstone { 1u64 } else { 0u64 };
        let p_low = payload_56 & 0x0FFF_FFFF_FFFF; // 44 bits
        let p_high = (payload_56 >> 44) & 0x0FFF; // 12 bits
        let raw = (p_high << 48) | (t_bit << 47) | (p_low << 3); // Type 0 (LeafInlined)
        Self(raw)
    }

    /// Unpacks the 56-bit leaf payload previously stored by
    /// [`Self::pack_inline_payload`].
    ///
    /// The low 44 bits are recovered from bits `3..46` and the high 12 bits
    /// from bits `48..59`; the tombstone bit is bypassed. The caller is
    /// responsible for zero-extending the result back to 64 bits, which is
    /// a no-op because the truncated MSBs were zero by construction.
    #[inline]
    #[must_use]
    pub const fn unpack_inline_payload(self) -> u64 {
        let p_low = (self.0 & 0x0000_7FFF_FFFF_FFF8) >> 3;
        let p_high = (self.0 >> 48) & 0x0FFF;
        (p_high << 44) | p_low
    }

    /// Marks this pointer as logically deleted by raising the tombstone bit.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::ptr::TaggedPointer;
    /// let ptr = TaggedPointer::pack_inline_payload(7, false);
    /// let deleted = ptr.mark_tombstone();
    /// assert!(deleted.is_tombstone());
    /// // The payload survives the tombstone routing intact.
    /// assert_eq!(deleted.unpack_inline_payload(), 7);
    /// ```
    #[inline]
    #[must_use]
    pub const fn mark_tombstone(self) -> Self {
        Self(self.0 | Self::MASK_TOMBSTONE)
    }

    /// Wraps the raw 64-bit word without re-validating its fields.
    ///
    /// The word must have been produced by one of the pack routines of this
    /// type, otherwise field accessors may decode reserved bit patterns.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw 64-bit word backing this pointer.
    #[inline]
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        self.0
    }
}

impl From<u64> for TaggedPointer {
    /// Wraps a raw 64-bit word as a [`TaggedPointer`].
    fn from(bits: u64) -> Self {
        Self(bits)
    }
}

impl From<TaggedPointer> for u64 {
    /// Unwraps the raw 64-bit word backing a [`TaggedPointer`].
    fn from(ptr: TaggedPointer) -> Self {
        ptr.0
    }
}

/// Resolves the dense child-array index for a key nibble via popcount.
///
/// The child-presence bitmap is read from the polymorphic field of the
/// tagged pointer. A tombstoned pointer reports no child, and a missing
/// bit reports `None`. The dense index is the number of set bits strictly
/// before the requested bit position, computed with the hardware popcount
/// instruction (`u16::count_ones`).
///
/// # Examples
///
/// ```
/// # use flare_core::ptr::{NodeType, TaggedPointer, resolve_child_index};
/// // Children 1 and 3 are present; child 2 is absent.
/// let ptr = TaggedPointer::pack(NodeType::Node16, 0, 0, false, 0b1010);
/// assert_eq!(resolve_child_index(ptr, 3), Some(1));
/// assert_eq!(resolve_child_index(ptr, 2), None);
/// ```
#[inline]
#[must_use]
pub const fn resolve_child_index(tagged_ptr: TaggedPointer, nibble: u8) -> Option<usize> {
    if tagged_ptr.is_tombstone() {
        return None;
    }
    let presence_mask = tagged_ptr.polymorphic_field();
    let bit = 1u16 << (nibble & 0x0F);
    if (presence_mask & bit) == 0 {
        return None;
    }
    let mask_prior = bit - 1;
    let dense_index = (presence_mask & mask_prior).count_ones() as usize;
    Some(dense_index)
}

#[cfg(test)]
mod tests {
    use super::{NodeType, TaggedPointer, resolve_child_index};

    /// Verifies that every node type discriminant round-trips through the
    /// 3-bit type field.
    #[test]
    fn node_type_roundtrip() {
        for kind in 0..=5u8 {
            let decoded = NodeType::from_discriminant(kind).expect("valid discriminant");
            assert_eq!(decoded.discriminant(), kind);
        }
        assert!(NodeType::from_discriminant(6).is_err());
        assert!(NodeType::from_discriminant(7).is_err());
        assert!(NodeType::from_discriminant(255).is_err());
    }

    /// Verifies that full-range offset and arena identifiers survive a pack
    /// and unpack cycle.
    #[test]
    fn full_range_roundtrip() {
        let max_offset = (1u64 << 40) - 1;
        for node_type in [
            NodeType::LeafInlined,
            NodeType::LeafOffset,
            NodeType::Node4,
            NodeType::Node16,
            NodeType::Node64,
            NodeType::Node256,
        ] {
            for arena_id in [0u8, 15u8] {
                for tombstone in [false, true] {
                    let ptr =
                        TaggedPointer::pack(node_type, max_offset, arena_id, tombstone, 0xABCD);
                    assert_eq!(ptr.node_type(), node_type);
                    assert_eq!(ptr.offset(), max_offset);
                    assert_eq!(ptr.arena_id(), arena_id);
                    assert_eq!(ptr.is_tombstone(), tombstone);
                    assert_eq!(ptr.polymorphic_field(), 0xABCD);
                }
            }
        }
    }

    /// Verifies that the tombstone bit is strictly isolated from the 56-bit
    /// inline payload: no payload pattern may overwrite bit 47.
    #[test]
    fn tombstone_bit_is_isolated_from_payload() {
        let max_payload = (1u64 << 56) - 1;
        let ptr = TaggedPointer::pack_inline_payload(max_payload, false);
        assert!(!ptr.is_tombstone());
        assert_eq!(ptr.unpack_inline_payload(), max_payload);
        let tombstoned = TaggedPointer::pack_inline_payload(max_payload, true);
        assert!(tombstoned.is_tombstone());
        assert_eq!(tombstoned.unpack_inline_payload(), max_payload);
        assert_eq!(
            TaggedPointer::pack_inline_payload(max_payload, true).to_bits()
                & TaggedPointer::MASK_TOMBSTONE,
            TaggedPointer::MASK_TOMBSTONE
        );
    }

    /// Verifies 8-bit MSB truncation semantics for 64-bit identifiers.
    #[test]
    fn msb_truncation_zero_extension() {
        let value = 0x00FF_FFFF_FFFF_FFFFu64; // MSB byte is zero.
        let ptr = TaggedPointer::pack_inline_payload(value, false);
        assert_eq!(ptr.unpack_inline_payload(), value);
        assert_eq!(ptr.to_bits() >> 60, 0x0000); // Reserved nibble stays clear.
    }

    /// Verifies popcount dense-index resolution across the full bitmap.
    #[test]
    fn popcount_resolution() {
        let mask = 0b1000_0000_0000_1110u16;
        let ptr = TaggedPointer::pack(NodeType::Node16, 0, 0, false, mask);
        assert_eq!(resolve_child_index(ptr, 1), Some(0));
        assert_eq!(resolve_child_index(ptr, 2), Some(1));
        assert_eq!(resolve_child_index(ptr, 3), Some(2));
        assert_eq!(resolve_child_index(ptr, 15), Some(3));
        assert_eq!(resolve_child_index(ptr, 4), None);
        let tombstoned = ptr.mark_tombstone();
        assert_eq!(resolve_child_index(tombstoned, 1), None);
    }

    /// Verifies that `mark_tombstone` preserves every other field.
    #[test]
    fn tombstone_preserves_fields() {
        let ptr = TaggedPointer::pack(NodeType::Node64, 123_456, 3, false, 0x42);
        let marked = ptr.mark_tombstone();
        assert!(marked.is_tombstone());
        assert_eq!(marked.node_type(), NodeType::Node64);
        assert_eq!(marked.offset(), 123_456);
        assert_eq!(marked.arena_id(), 3);
        assert_eq!(marked.polymorphic_field(), 0x42);
    }

    /// Verifies the bitwise constants cover the exact field boundaries.
    #[test]
    fn mask_boundaries() {
        assert_eq!(TaggedPointer::MASK_TYPE, 0b111);
        assert_eq!(TaggedPointer::MASK_OFFSET, 0x0000_07FF_FFFF_FFF8);
        assert_eq!(TaggedPointer::MASK_ARENA, 0x0000_7800_0000_0000);
        assert_eq!(TaggedPointer::MASK_TOMBSTONE, 0x0000_8000_0000_0000);
        assert_eq!(TaggedPointer::MASK_POLY, 0xFFFF_0000_0000_0000);
        assert_eq!(
            TaggedPointer::MASK_TYPE
                | TaggedPointer::MASK_OFFSET
                | TaggedPointer::MASK_ARENA
                | TaggedPointer::MASK_TOMBSTONE
                | TaggedPointer::MASK_POLY,
            u64::MAX
        );
    }

    /// Verifies the bidirectional `From` conversions with `u64`.
    #[test]
    fn from_bits_conversions() {
        let ptr = TaggedPointer::pack(NodeType::Node4, 7, 1, false, 0);
        let bits: u64 = u64::from(ptr);
        assert_eq!(bits, ptr.to_bits());
        assert_eq!(TaggedPointer::from(bits).to_bits(), bits);
    }

    /// Verifies that a corrupted type discriminant decodes to a leaf
    /// instead of panicking, deferring rejection to the caller.
    #[test]
    fn corrupted_type_discriminant_falls_back() {
        let ptr = TaggedPointer::from_bits(0b110);
        assert_eq!(ptr.node_type(), NodeType::LeafInlined);
    }
}
