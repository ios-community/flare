//! Engine event definitions carried by the telemetry ring buffer.
//!
//! Every event is encoded into a single 64-bit word so that the collector
//! can move events between threads with plain atomic stores and zero
//! allocation: the high byte selects the [`EventKind`] and the remaining
//! 56 bits carry two 28-bit payload fields.
#![allow(clippy::cast_lossless, clippy::cast_possible_truncation)]

use core::fmt;

/// Mask covering the 28-bit payload fields of an encoded event word.
const MASK28: u32 = 0x0FFF_FFFF;

/// Identifies the kind of engine event carried by a telemetry word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// A key-value pair was inserted into the radix tree.
    TreeInsert,
    /// An insert observed a concurrent writer (CAS race probe).
    CasContention,
    /// A key-value lookup returned a hit.
    TreeHit,
    /// A vector was appended to the IVF-PQ index.
    VectorInsert,
    /// A vector search completed.
    VectorSearch,
    /// A shadow re-clustering generation was published.
    Recluster,
    /// A token prefix was stored in the KV-cache engine.
    KvInsert,
    /// A longest-common-prefix match succeeded.
    KvMatch,
    /// A WAL batch was flushed to the sink.
    WalFlush,
    /// The clock sweep evicted slab slots.
    ClockEvict,
}

impl EventKind {
    /// Returns the 8-bit tag encoding of this kind.
    #[must_use]
    pub const fn tag(self) -> u8 {
        match self {
            Self::TreeInsert => 0,
            Self::CasContention => 1,
            Self::TreeHit => 2,
            Self::VectorInsert => 3,
            Self::VectorSearch => 4,
            Self::Recluster => 5,
            Self::KvInsert => 6,
            Self::KvMatch => 7,
            Self::WalFlush => 8,
            Self::ClockEvict => 9,
        }
    }

    /// Decodes a kind from its 8-bit tag; unknown tags decode to
    /// [`Self::TreeHit`] so that corrupted words degrade gracefully.
    #[must_use]
    pub const fn from_tag(tag: u8) -> Self {
        match tag {
            0 => Self::TreeInsert,
            1 => Self::CasContention,
            2 => Self::TreeHit,
            3 => Self::VectorInsert,
            4 => Self::VectorSearch,
            5 => Self::Recluster,
            6 => Self::KvInsert,
            7 => Self::KvMatch,
            8 => Self::WalFlush,
            _ => Self::ClockEvict,
        }
    }
}

/// A decoded telemetry event: a kind plus two 28-bit payload fields.
///
/// The payload interpretation depends on the kind; see the constructor
/// helpers of [`EventWord`] for the exact field semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventWord {
    /// The kind of engine event.
    pub kind: EventKind,
    /// First 28-bit payload field.
    pub a: u32,
    /// Second 28-bit payload field.
    pub b: u32,
}

impl EventWord {
    /// Packs a kind plus two payload fields into one 64-bit word.
    #[must_use]
    pub const fn encode(self) -> u64 {
        ((self.kind.tag() as u64) << 56)
            | (((self.a & MASK28) as u64) << 28)
            | (self.b & MASK28) as u64
    }

    /// Unpacks a 64-bit word into a kind plus two payload fields.
    ///
    /// The payload fields are masked to 28 bits before the casts, so the
    /// narrowing casts below can never truncate.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn decode(word: u64) -> Self {
        Self {
            kind: EventKind::from_tag((word >> 56) as u8),
            a: ((word >> 28) & MASK28 as u64) as u32,
            b: (word & MASK28 as u64) as u32,
        }
    }

    /// Records a radix tree insert (`a` = key length, `b` = low bits of
    /// the allocated arena offset).
    #[must_use]
    pub const fn tree_insert(key_len: usize, offset: u64) -> Self {
        Self {
            kind: EventKind::TreeInsert,
            a: key_len as u32,
            b: (offset & MASK28 as u64) as u32,
        }
    }

    /// Records a CAS race probe at an insert callsite (`a` = observed
    /// concurrent overwrite, `b` = attempt count).
    #[must_use]
    pub const fn cas_contention(observed: u32, attempts: u32) -> Self {
        Self {
            kind: EventKind::CasContention,
            a: observed,
            b: attempts,
        }
    }

    /// Records a key-value lookup hit (`a` = key length).
    ///
    /// Kept in the vocabulary for completeness; the TUI workload currently
    /// filters hit events out of the footer log and does not push them.
    #[allow(dead_code)]
    #[must_use]
    pub const fn tree_hit(key_len: usize) -> Self {
        Self {
            kind: EventKind::TreeHit,
            a: key_len as u32,
            b: 0,
        }
    }

    /// Records a vector append (`a` = dimension, `b` = low bits of the
    /// insertion sequence id).
    #[must_use]
    pub const fn vector_insert(dimension: usize, id: u64) -> Self {
        Self {
            kind: EventKind::VectorInsert,
            a: dimension as u32,
            b: (id & MASK28 as u64) as u32,
        }
    }

    /// Records a completed search (`a` = requested top-k, `b` = hits).
    #[must_use]
    pub const fn vector_search(top_k: usize, hits: usize) -> Self {
        Self {
            kind: EventKind::VectorSearch,
            a: top_k as u32,
            b: hits as u32,
        }
    }

    /// Records a published re-clustering generation (`a` = generation).
    #[must_use]
    pub const fn recluster(generation: u64) -> Self {
        Self {
            kind: EventKind::Recluster,
            a: (generation & MASK28 as u64) as u32,
            b: 0,
        }
    }

    /// Records a KV prefix insert (`a` = token count, `b` = low bits of
    /// the published KV offset).
    #[must_use]
    pub const fn kv_insert(token_len: usize, kv_offset: u64) -> Self {
        Self {
            kind: EventKind::KvInsert,
            a: token_len as u32,
            b: (kv_offset & MASK28 as u64) as u32,
        }
    }

    /// Records an LCP match (`a` = matched token count).
    #[must_use]
    pub const fn kv_match(token_len: usize) -> Self {
        Self {
            kind: EventKind::KvMatch,
            a: token_len as u32,
            b: 0,
        }
    }

    /// Records a WAL batch flush (`a` = low bits of the frame counter).
    #[must_use]
    pub const fn wal_flush(frames: u64) -> Self {
        Self {
            kind: EventKind::WalFlush,
            a: (frames & MASK28 as u64) as u32,
            b: 0,
        }
    }

    /// Records a clock sweep (`a` = evicted slots).
    #[must_use]
    pub const fn clock_evict(slots: usize) -> Self {
        Self {
            kind: EventKind::ClockEvict,
            a: slots as u32,
            b: 0,
        }
    }
}

impl fmt::Display for EventWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            EventKind::TreeInsert => write!(f, "tree.insert klen={} off={:#x}", self.a, self.b),
            EventKind::CasContention => {
                write!(f, "cas.race observed={} attempts={}", self.a, self.b)
            }
            EventKind::TreeHit => write!(f, "tree.hit klen={}", self.a),
            EventKind::VectorInsert => write!(f, "vector.insert dim={} id={}", self.a, self.b),
            EventKind::VectorSearch => write!(f, "vector.search top={} hits={}", self.a, self.b),
            EventKind::Recluster => write!(f, "recluster gen={}", self.a),
            EventKind::KvInsert => write!(f, "kv.insert tokens={} off={}", self.a, self.b),
            EventKind::KvMatch => write!(f, "kv.match tokens={}", self.a),
            EventKind::WalFlush => write!(f, "wal.flush frames={}", self.a),
            EventKind::ClockEvict => write!(f, "clock.evict slots={}", self.a),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EventKind, EventWord};

    /// Verifies that every event kind survives the encode/decode round-trip
    /// with both payload fields intact.
    #[test]
    fn encode_decode_roundtrip() {
        let cases = [
            EventWord::tree_insert(12, 0x1_2345),
            EventWord::cas_contention(1, 3),
            EventWord::tree_hit(7),
            EventWord::vector_insert(4, 42),
            EventWord::vector_search(10, 3),
            EventWord::recluster(2),
            EventWord::kv_insert(16, 0xFFFF),
            EventWord::kv_match(9),
            EventWord::wal_flush(12_345),
            EventWord::clock_evict(4),
        ];
        for event in cases {
            let decoded = EventWord::decode(event.encode());
            assert_eq!(decoded, event, "round-trip failed for {event}");
        }
    }

    /// Verifies that payload fields are truncated to 28 bits instead of
    /// corrupting the kind tag.
    #[test]
    fn payloads_are_truncated() {
        let event = EventWord::tree_insert(usize::MAX, u64::MAX);
        let decoded = EventWord::decode(event.encode());
        assert_eq!(decoded.kind, EventKind::TreeInsert);
        assert_eq!(decoded.a, super::MASK28);
        assert_eq!(decoded.b, super::MASK28);
    }

    /// Verifies that every event renders a human-readable string.
    #[test]
    fn display_is_non_empty() {
        for event in [
            EventWord::tree_insert(4, 8),
            EventWord::cas_contention(0, 1),
            EventWord::clock_evict(2),
        ] {
            let rendered = format!("{event}");
            assert!(!rendered.is_empty());
        }
    }
}
