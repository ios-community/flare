//! Error handling for the FLARE core primitives.
//!
//! The [`FlareError`] enumeration describes every failure mode surfaced by
//! the `flare-core` primitives, from arena capacity exhaustion through
//! malformed WAL frames to unknown pinned-memory pointers.

use core::fmt;

/// Defines every failure mode that the FLARE core primitives can surface.
///
/// All operations that may fail return `Result<_, FlareError>`. The variants
/// are deliberately narrow so that callers can react to each condition
/// precisely: capacity exhaustion is recoverable by growing the backing
/// arena, malformed WAL frames are recoverable by truncating the log, and
/// pointer mismatches indicate a caller-side lifecycle violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlareError {
    /// A read or write touched an offset range outside the arena capacity.
    ///
    /// The fields identify the offending region and the current arena
    /// capacity. This normally indicates a use-after-retire cycle where a
    /// tagged pointer outlived its allocated region.
    ArenaBoundsExceeded {
        /// The region start offset that violated the arena boundary.
        offset: u64,
        /// The number of bytes the caller attempted to touch.
        length: usize,
        /// The total capacity of the arena in bytes.
        capacity: u64,
    },
    /// The bump frontier advanced beyond the arena capacity.
    ///
    /// The requested allocation could not be satisfied because the
    /// contiguous arena is full. The caller may resize the arena or retire
    /// slab-backed nodes to reclaim space.
    ArenaCapacityExceeded {
        /// The number of bytes the caller requested.
        requested: u64,
        /// The number of free bytes remaining at request time.
        available: u64,
    },
    /// The system allocator failed to back the arena or a pinned block.
    ///
    /// This is the only condition that may abort the process on memory
    /// exhaustion; for arena growth it is recoverable as a regular error.
    AllocationFailed,
    /// A tagged pointer carried an encoding that cannot be interpreted.
    ///
    /// The embedded node-type field is compared against the known
    /// [`NodeType`](crate::ptr::NodeType) discriminants, and type values
    /// `6` and `7` are reserved for future extensions.
    InvalidNodeType(u8),
    /// A WAL frame could not be decoded from raw bytes.
    ///
    /// The rejection reason is a `'static` description of the offending
    /// field, for example a frame length larger than the remaining buffer.
    WalFrameMalformed {
        /// Human-readable description of the malformed field.
        reason: &'static str,
    },
    /// A WAL frame declared a payload larger than the configured limit.
    ///
    /// The limit defends replay against corrupted length fields that would
    /// otherwise trigger unbounded `memcpy` operations during recovery.
    WalFrameTooLarge {
        /// The length declared by the frame header.
        declared: u32,
    },
    /// A deallocation referenced a pinned block that was never allocated.
    ///
    /// This indicates a lifecycle violation: the caller passed a pointer
    /// that does not belong to this driver instance.
    UnknownPinnedPointer,
    /// A tree traversal reached a state that the construction invariants
    /// forbid.
    ///
    /// Examples include a leaf word encountered mid-path or an internal
    /// node treated as a leaf; both indicate memory corruption or a stale
    /// snapshot.
    TreeInvariantViolation {
        /// Human-readable description of the violated invariant.
        reason: &'static str,
    },
    /// A vector operation received a slice whose length is not a multiple
    /// of the engine dimension.
    ///
    /// The fields identify the expected row stride and the offending slice
    /// length. This normally indicates a caller-side packing error.
    VectorDimensionMismatch {
        /// The engine dimension each row must satisfy.
        expected: usize,
        /// The length of the offending slice.
        got: usize,
    },
    /// An operation received a parameter combination the engine rejects.
    ///
    /// The reason is a `'static` description of the offending parameter,
    /// for example a training set smaller than the requested cluster count
    /// or a dimension not divisible by the sub-vector count.
    InvalidParameter {
        /// Human-readable description of the offending parameter.
        reason: &'static str,
    },
    /// A KV-cache eviction round could not reclaim the requested capacity.
    ///
    /// This indicates that every slab slot is pinned live (or the hand scan
    /// budget was exhausted) and no further insert is possible without
    /// explicit external eviction.
    CacheCapacityExceeded,
    /// The GPU runtime required by a driver was not found or unusable.
    ///
    /// Optional drivers (for example the CUDA driver in `flare-ffi`) load
    /// the runtime dynamically, so a missing runtime surfaces as this
    /// error instead of a link-time failure. The reason names the missing
    /// runtime or the CUDA error that made it unusable.
    GpuDriverUnavailable {
        /// Human-readable description of the missing runtime.
        reason: &'static str,
    },
}

impl fmt::Display for FlareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArenaBoundsExceeded {
                offset,
                length,
                capacity,
            } => write!(
                f,
                "arena access out of bounds: offset {offset} + {length} bytes exceeds capacity {capacity}"
            ),
            Self::ArenaCapacityExceeded {
                requested,
                available,
            } => write!(
                f,
                "arena capacity exhausted: requested {requested} bytes, only {available} available"
            ),
            Self::AllocationFailed => write!(f, "system allocator returned null"),
            Self::InvalidNodeType(kind) => write!(f, "invalid node type discriminant {kind}"),
            Self::WalFrameMalformed { reason } => write!(f, "malformed WAL frame: {reason}"),
            Self::WalFrameTooLarge { declared } => {
                write!(
                    f,
                    "WAL frame payload length {declared} exceeds replay limit"
                )
            }
            Self::UnknownPinnedPointer => write!(f, "deallocation of an unknown pinned pointer"),
            Self::TreeInvariantViolation { reason } => {
                write!(f, "tree invariant violation: {reason}")
            }
            Self::VectorDimensionMismatch { expected, got } => write!(
                f,
                "vector dimension mismatch: expected {expected} elements per row, got {got}"
            ),
            Self::InvalidParameter { reason } => {
                write!(f, "invalid parameter: {reason}")
            }
            Self::CacheCapacityExceeded => write!(
                f,
                "KV-cache capacity exceeded: eviction could not reclaim any slot"
            ),
            Self::GpuDriverUnavailable { reason } => {
                write!(f, "GPU runtime unavailable: {reason}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FlareError {}

#[cfg(test)]
mod tests {
    use super::FlareError;
    use core::fmt::Write;

    /// Asserts that every variant renders a non-empty human-readable message.
    #[test]
    fn display_messages_are_non_empty() {
        let cases = [
            FlareError::ArenaBoundsExceeded {
                offset: 10,
                length: 4,
                capacity: 1024,
            },
            FlareError::ArenaCapacityExceeded {
                requested: 17,
                available: 3,
            },
            FlareError::AllocationFailed,
            FlareError::InvalidNodeType(6),
            FlareError::WalFrameMalformed {
                reason: "header shorter than 4 bytes",
            },
            FlareError::WalFrameTooLarge { declared: 99_999 },
            FlareError::UnknownPinnedPointer,
            FlareError::TreeInvariantViolation {
                reason: "leaf mid-path",
            },
            FlareError::VectorDimensionMismatch {
                expected: 128,
                got: 96,
            },
            FlareError::InvalidParameter {
                reason: "training set smaller than cluster count",
            },
            FlareError::CacheCapacityExceeded,
            FlareError::GpuDriverUnavailable {
                reason: "nvcuda.dll not found",
            },
        ];
        for case in cases {
            let mut buffer = alloc_crate::string::String::new();
            let _ = write!(buffer, "{case}");
            assert!(!buffer.is_empty(), "display output must not be empty");
        }
    }
}
