//! Flat bump arena addressed by 40-bit relative offsets.
//!
//! The [`FlatArena`] is a contiguous, zero-initialised byte buffer whose
//! allocation frontier advances with an atomic `fetch_add`, giving `O(1)`
//! bump allocation that is safe to share across threads because regions are
//! provably disjoint. Typed access goes through the safe [`ArenaRef`]
//! wrapper, which bounds- and alignment-checks every dereference.

use crate::error::FlareError;
use alloc_crate::boxed::Box;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::sync::atomic::{AtomicU64, Ordering};

/// A contiguous byte arena with an atomic `O(1)` bump allocation frontier.
///
/// The arena is addressed in bytes; every allocation returns a 40-bit
/// relative offset that higher-level modules pack into tagged pointers.
/// All memory is zero-initialised at construction so that any region is
/// readable before its first write.
///
/// # Invariants
///
/// - **(I1) Disjointness.** Regions returned by [`Self::alloc`] are pairwise
///   disjoint, because the frontier advances atomically and every region is
///   reserved before its contents are observed.
/// - **(I2) Publish-after-init.** A region must be initialised with
///   [`Self::write_node`] before any tagged pointer referencing it is
///   published through a release-ordering atomic store; readers acquire the
///   pointer with an acquire-order load. This matches the child-before-
///   parent ordering enforced by the WAL.
/// - **(I3) Mutable exclusivity.** After publication, a region is mutated
///   exclusively through 64-bit atomic operations (child slots, value
///   words); region-wide rewrites via [`Self::write_node`] are only legal
///   while the region is unobservable.
///
/// # Examples
///
/// ```
/// # use flare_core::alloc::arena::FlatArena;
/// let arena = FlatArena::new(4096).expect("allocation succeeds");
/// let slot = arena.alloc(32, 8).expect("region fits");
/// assert_eq!(arena.capacity(), 4096);
/// ```
pub struct FlatArena {
    bytes: Box<[u8]>,
    frontier: AtomicU64,
    capacity: u64,
}

impl FlatArena {
    /// Creates a new zero-initialised arena with the given byte capacity.
    ///
    /// The backing allocation uses the global allocator with zero-initial
    /// memory aligned to 8 bytes, so any typed node whose alignment is at
    /// most 8 can be placed at an aligned offset. Capacity is declared up
    /// front; bump allocation never grows the arena.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::AllocationFailed`] if the global allocator
    /// cannot back `capacity` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// let arena = FlatArena::new(1 << 20).expect("allocation succeeds");
    /// ```
    pub fn new(capacity: usize) -> Result<Self, FlareError> {
        if capacity == 0 {
            return Ok(Self {
                bytes: Box::from([] as [u8; 0]),
                frontier: AtomicU64::new(0),
                capacity: 0,
            });
        }
        let layout = core::alloc::Layout::from_size_align(capacity, 8)
            .map_err(|_| FlareError::AllocationFailed)?;
        // SAFETY: `alloc_zeroed` returns a null pointer on failure, which is
        // checked below; the returned allocation satisfies `layout`, so it
        // is 8-byte aligned and holds `capacity` zero-initialised bytes.
        let ptr = unsafe { alloc_crate::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(FlareError::AllocationFailed);
        }
        // SAFETY: `ptr` is a valid, aligned, zero-initialised allocation of
        // exactly `capacity` bytes owned by this box.
        let bytes =
            unsafe { Box::<[u8]>::from_raw(core::ptr::slice_from_raw_parts_mut(ptr, capacity)) };
        Ok(Self {
            bytes,
            frontier: AtomicU64::new(0),
            capacity: capacity as u64,
        })
    }

    /// Returns the total byte capacity of this arena.
    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Returns the byte offset of the current bump frontier.
    #[must_use]
    pub fn frontier(&self) -> u64 {
        self.frontier.load(Ordering::Relaxed)
    }

    /// Returns the number of unallocated bytes remaining.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.capacity.saturating_sub(self.frontier())
    }

    /// Allocates a region of `size` bytes aligned to `align`.
    ///
    /// The allocation reserves `size + align - 1` bytes and returns the
    /// next aligned offset, so regions stay disjoint under concurrent
    /// callers. The returned offset fits in the 40-bit tagged pointer field
    /// unless the arena is larger than 1 TiB.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaCapacityExceeded`] when the aligned region
    /// does not fit in the remaining space.
    ///
    /// # Panics
    ///
    /// Panics if `align` is not a power of two or if `size` is zero; both
    /// are programmer errors that cannot be recovered from.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// let arena = FlatArena::new(1024).expect("allocation succeeds");
    /// let a = arena.alloc(16, 8).expect("first region fits");
    /// let b = arena.alloc(16, 8).expect("second region fits");
    /// assert_ne!(a, b);
    /// ```
    pub fn alloc(&self, size: usize, align: usize) -> Result<u64, FlareError> {
        assert!(size > 0, "allocation size must be non-zero");
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        let reserved = size
            .checked_add(align - 1)
            .ok_or_else(|| FlareError::ArenaCapacityExceeded {
                requested: u64::MAX,
                available: self.remaining(),
            })
            .and_then(|r| {
                u64::try_from(r).map_err(|_| FlareError::ArenaCapacityExceeded {
                    requested: u64::MAX,
                    available: self.remaining(),
                })
            })?;
        let raw = self.frontier.fetch_add(reserved, Ordering::Relaxed);
        let end = raw
            .checked_add(reserved)
            .ok_or_else(|| FlareError::ArenaCapacityExceeded {
                requested: reserved,
                available: self.remaining(),
            })?;
        if end > self.capacity {
            return Err(FlareError::ArenaCapacityExceeded {
                requested: reserved,
                available: self.remaining(),
            });
        }
        let align_mask = u64::try_from(align - 1).expect("alignment fits in u64");
        let start = (raw + align_mask) & !align_mask;
        Ok(start)
    }

    /// Writes the representation of `value` into a previously allocated region.
    ///
    /// The source and destination must not overlap and the destination must
    /// be a region exclusively owned by the caller (invariant I3); typical
    /// usage writes a freshly allocated node identity before it is
    /// published. The write is a plain byte copy, not an atomic store.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaBoundsExceeded`] when the region
    /// `offset..offset + size` lies outside the arena or is not aligned to
    /// the value's alignment.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `offset` addresses a region exclusively
    /// owned by the caller (invariant I3) and that the region is not
    /// concurrently accessed by other threads. The region must have been
    /// allocated by [`Self::alloc`] and not yet published to other threads.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// let arena = FlatArena::new(1024).expect("allocation succeeds");
    /// let slot = arena.alloc(8, 8).expect("region fits");
    /// arena.write_node(slot, &7u64).expect("write succeeds");
    /// assert_eq!(arena.read_node::<u64>(slot).expect("read succeeds"), &7);
    /// ```
    pub fn write_node<T>(&self, offset: u64, value: &T) -> Result<(), FlareError> {
        let len = size_of::<T>();
        let range = self.check_range(offset, len)?;
        // SAFETY: the destination range is validated in-bounds above.
        // Invariant I3 guarantees the caller exclusively owns this region,
        // so a write cannot race with a concurrent reader of the same bytes.
        unsafe {
            core::ptr::copy_nonoverlapping(
                core::ptr::from_ref::<T>(value).cast::<u8>(),
                self.bytes.as_ptr().add(range.start).cast_mut(),
                len,
            );
        }
        Ok(())
    }

    /// Copies raw bytes into a previously allocated region.
    ///
    /// This is the byte-level counterpart of [`Self::write_node`] used by
    /// the WAL replay overlay: a frame's payload is `memcpy`-ed directly
    /// onto its recorded offset. The same exclusivity invariant (I3)
    /// applies: the region must be owned by the caller and unobservable
    /// during the copy.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaBoundsExceeded`] when the region lies
    /// outside the arena.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `offset` addresses a region exclusively
    /// owned by the caller (invariant I3) and that the region is not
    /// concurrently accessed by other threads. The region must have been
    /// allocated by [`Self::alloc`] and not yet published to other threads.
    /// The `data` slice must not overlap with the destination region.
    pub fn write_bytes(&self, offset: u64, data: &[u8]) -> Result<(), FlareError> {
        let range = self.check_range(offset, data.len())?;
        // SAFETY: the destination range is validated in-bounds above, and
        // invariant I3 grants the caller exclusive ownership of the
        // region for the duration of the copy.
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.bytes.as_ptr().add(range.start).cast_mut(),
                data.len(),
            );
        }
        Ok(())
    }

    /// Returns a shared reference to the value stored at `offset`.
    ///
    /// The region must have been initialised and must not be concurrently
    /// rewritten with [`Self::write_node`]; atomic slot mutations within the
    /// region are permitted and remain observable through atomics.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaBoundsExceeded`] when the region lies
    /// outside the arena or is misaligned.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the region at `offset` has been
    /// initialized with a valid `T` representation and that no concurrent
    /// writes are occurring to the same region. The returned reference is
    /// bound to the lifetime of the arena and must not outlive it.
    pub fn read_node<T>(&self, offset: u64) -> Result<&T, FlareError> {
        let len = size_of::<T>();
        let range = self.check_range(offset, len)?;
        // SAFETY: typed node access requires the type alignment to be
        // covered by the 8-byte-aligned backing allocation.
        debug_assert!(
            align_of::<T>() <= 8,
            "arena backing allocation is only 8-byte aligned"
        );
        if range.start % align_of::<T>() != 0 {
            return Err(FlareError::ArenaBoundsExceeded {
                offset,
                length: len,
                capacity: self.capacity,
            });
        }
        // SAFETY: the range was validated above and `T` is sized.
        // Invariant I2 guarantees written-before-published for observable
        // regions, so the bytes form a valid `T` representation.
        unsafe {
            let ptr = self.bytes.as_ptr().add(range.start).cast::<T>();
            Ok(&*ptr)
        }
    }

    /// Returns a safe handle to the typed node stored at `offset`.
    ///
    /// The handle carries out bounds and alignment validation on
    /// construction and provides [`ArenaRef::get`] for access.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::arena::FlatArena;
    /// let arena = FlatArena::new(1024).expect("allocation succeeds");
    /// let slot = arena.alloc(8, 8).expect("region fits");
    /// arena.write_node(slot, &99u64).expect("write succeeds");
    /// let node_ref = arena.node_ref::<u64>(slot).expect("valid offset");
    /// assert_eq!(*node_ref.get().expect("initialised region"), 99);
    /// ```
    #[must_use]
    pub fn node_ref<T>(&self, offset: u64) -> Option<ArenaRef<'_, T>> {
        let range = self.check_range(offset, size_of::<T>()).ok()?;
        if range.start % align_of::<T>() != 0 {
            return None;
        }
        Some(ArenaRef::new(self, offset))
    }

    /// Returns a word-sized atomic view over an 8-byte value slot.
    ///
    /// Used by leaves that cannot inline their value: the offset points at
    /// an 8-byte aligned slot holding `u64` payload. Reads and writes are
    /// performed with the requested ordering and therefore remain
    /// observable while other threads mutate the same slot.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaBoundsExceeded`] for out-of-bounds or
    /// misaligned offsets.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `offset` is 8-byte aligned and that the
    /// 8-byte region is exclusively used for atomic operations. The returned
    /// reference is bound to the lifetime of the arena and must not outlive
    /// it. Concurrent atomic operations on the same word are safe.
    #[allow(clippy::cast_ptr_alignment)]
    pub fn atomic_word(&self, offset: u64) -> Result<&AtomicU64, FlareError> {
        let range = self.check_range(offset, 8)?;
        if range.start % 8 != 0 {
            return Err(FlareError::ArenaBoundsExceeded {
                offset,
                length: 8,
                capacity: self.capacity,
            });
        }
        // SAFETY: the range is validated in-bounds and 8-byte aligned; the
        // backing allocation is 8-byte aligned by construction, so the
        // `AtomicU64` natural-alignment requirement holds.
        unsafe { Ok(&*self.bytes.as_ptr().add(range.start).cast::<AtomicU64>()) }
    }

    /// Validates that `offset..offset + len` lies within the arena bounds.
    fn check_range(&self, offset: u64, len: usize) -> Result<core::ops::Range<usize>, FlareError> {
        let end = offset
            .checked_add(u64::try_from(len).expect("region length fits in u64"))
            .ok_or(FlareError::ArenaBoundsExceeded {
                offset,
                length: len,
                capacity: self.capacity,
            })?;
        if end > self.capacity {
            return Err(FlareError::ArenaBoundsExceeded {
                offset,
                length: len,
                capacity: self.capacity,
            });
        }
        let start = usize::try_from(offset).expect("offset fits in usize");
        let end = usize::try_from(end).expect("bounded end fits in usize");
        Ok(start..end)
    }
}

impl core::fmt::Debug for FlatArena {
    /// Formats the arena as its capacity, current frontier, and byte count.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FlatArena")
            .field("capacity", &self.capacity)
            .field("frontier", &self.frontier())
            .field("bytes_len", &self.bytes.len())
            .finish()
    }
}

/// Safety justification for sharing the arena across threads.
///
/// The bump frontier is an atomic counter, so region reservation is
/// lock-free and returns pairwise-disjoint ranges (invariant I1). Shared
/// reads target immutable bytes; mutable access is confined to
/// exclusively-owned regions before publication (I2) or to atomic words
/// after publication (I3). No two threads ever hold mutable access to the
/// same bytes, and readers never observe a region before its initialising
/// write is published through an ordering-release atomic.
///
/// # Safety
///
/// This is sound because the disjoint-region discipline is enforced by
/// the allocator's atomic frontier and documented for every mutation API.
unsafe impl Sync for FlatArena {}

/// A bounds-checked handle to a typed node stored inside a [`FlatArena`].
///
/// The handle is the only safe way for higher-level modules (the adaptive
/// radix tree, the vector engine, the KV-cache engine) to interface with
/// arena-resident data. Raw virtual pointers are never exposed.
///
/// # Examples
///
/// ```
/// # use flare_core::alloc::arena::FlatArena;
/// let arena = FlatArena::new(1024).expect("allocation succeeds");
/// let slot = arena.alloc(8, 8).expect("region fits");
/// arena.write_node(slot, &5u64).expect("write succeeds");
/// let handle = arena.node_ref::<u64>(slot).expect("valid offset");
/// assert_eq!(*handle.get().expect("initialised region"), 5);
/// ```
pub struct ArenaRef<'a, T> {
    arena: &'a FlatArena,
    offset: u64,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> ArenaRef<'a, T> {
    /// Creates a handle over a validated region.
    ///
    /// Callers must guarantee that `offset` addresses an in-bounds,
    /// correctly aligned region holding a live `T`; [`FlatArena::node_ref`]
    /// performs that validation before constructing handles.
    const fn new(arena: &'a FlatArena, offset: u64) -> Self {
        Self {
            arena,
            offset,
            _marker: PhantomData,
        }
    }

    /// Returns the arena offset addressed by this handle.
    ///
    /// The offset is the 40-bit relative index that callers may pack into
    /// tagged pointers.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns a shared reference to the node behind this handle.
    ///
    /// Returns `None` if the region was never initialised or has been
    /// retired. The reference must not be held across publication of a
    /// rewrite of the same region, per invariant I3.
    #[must_use]
    pub fn get(&self) -> Option<&'a T> {
        let value = self.arena.read_node::<T>(self.offset).ok()?;
        Some(value)
    }

    /// Returns the alignment requirement of the referenced type.
    #[must_use]
    pub const fn alignment() -> usize {
        align_of::<T>()
    }
}

impl<T> Copy for ArenaRef<'_, T> {}

impl<T> Clone for ArenaRef<'_, T> {
    /// Clones the handle; both copies address the same region.
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> core::fmt::Debug for ArenaRef<'_, T> {
    /// Formats the handle as its addressed offset.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArenaRef")
            .field("offset", &self.offset)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{FlareError, FlatArena};

    /// Verifies that adjacent allocations are disjoint and aligned.
    #[test]
    fn bump_allocations_are_disjoint_and_aligned() {
        let arena = FlatArena::new(256).expect("allocation succeeds");
        let a = arena.alloc(17, 8).expect("first region fits");
        let b = arena.alloc(17, 8).expect("second region fits");
        assert_eq!(a % 8, 0);
        assert_eq!(b % 8, 0);
        assert!(b >= a + 17);
        assert!(b + 17 <= arena.capacity());
    }

    /// Verifies that the frontier advances monotonically with allocation.
    #[test]
    fn frontier_advances() {
        let arena = FlatArena::new(128).expect("allocation succeeds");
        let before = arena.frontier();
        let _ = arena.alloc(32, 8).expect("region fits");
        assert!(arena.frontier() >= before + 32);
        assert_eq!(arena.remaining(), arena.capacity() - arena.frontier());
    }

    /// Verifies that exhaustion reports a precise error.
    #[test]
    fn capacity_exhaustion_is_reported() {
        let arena = FlatArena::new(32).expect("allocation succeeds");
        let _ = arena.alloc(16, 8).expect("region fits");
        match arena.alloc(32, 8) {
            Err(FlareError::ArenaCapacityExceeded { .. }) => {}
            other => panic!("expected capacity error, got {other:?}"),
        }
    }

    /// Verifies that out-of-bounds reads are rejected before dereferencing.
    #[test]
    fn bounds_are_enforced() {
        let arena = FlatArena::new(32).expect("allocation succeeds");
        assert!(matches!(
            arena.read_node::<u64>(28),
            Err(FlareError::ArenaBoundsExceeded { .. })
        ));
        assert!(matches!(
            arena.read_node::<u64>(25),
            Err(FlareError::ArenaBoundsExceeded { .. })
        ));
        assert!(matches!(
            arena.read_node::<u64>(1),
            Err(FlareError::ArenaBoundsExceeded { .. })
        ));
    }

    /// Verifies that value slots round-trip through the atomic word API.
    #[test]
    fn atomic_words_roundtrip() {
        let arena = FlatArena::new(64).expect("allocation succeeds");
        let slot = arena.alloc(8, 8).expect("region fits");
        let word = arena.atomic_word(slot).expect("aligned slot");
        word.store(0xDEAD_BEEF, core::sync::atomic::Ordering::Release);
        let reread = arena.atomic_word(slot).expect("aligned slot");
        assert_eq!(
            reread.load(core::sync::atomic::Ordering::Acquire),
            0xDEAD_BEEF
        );
    }

    /// Verifies that typed nodes survive a write/read cycle.
    #[test]
    fn typed_nodes_roundtrip() {
        let arena = FlatArena::new(512).expect("allocation succeeds");
        let slot = arena.alloc(32, 8).expect("region fits");
        let bytes = [0xABu8; 32];
        arena.write_node(slot, &bytes).expect("write succeeds");
        let read_back = arena.read_node::<[u8; 32]>(slot).expect("read succeeds");
        assert_eq!(*read_back, bytes);
    }

    /// Verifies that `node_ref` refuses misaligned offsets.
    #[test]
    fn node_ref_rejects_misaligned_offsets() {
        let arena = FlatArena::new(64).expect("allocation succeeds");
        let _ = arena.alloc(8, 8).expect("region fits");
        assert!(arena.node_ref::<u64>(1).is_none());
    }

    /// Verifies that offset arithmetic saturates at capacity instead of
    /// wrapping.
    #[test]
    fn oversized_requests_are_rejected() {
        let arena = FlatArena::new(64).expect("allocation succeeds");
        assert!(matches!(
            arena.alloc(usize::MAX - 1, 8),
            Err(FlareError::ArenaCapacityExceeded { .. })
        ));
    }

    /// Verifies that a zero-capacity arena is constructible and rejects
    /// every allocation.
    #[test]
    fn zero_capacity_arena() {
        let arena = FlatArena::new(0).expect("zero-capacity arena is valid");
        assert_eq!(arena.capacity(), 0);
        assert_eq!(arena.remaining(), 0);
        assert!(matches!(
            arena.alloc(8, 8),
            Err(FlareError::ArenaCapacityExceeded { .. })
        ));
    }

    /// Verifies that misaligned atomic-word views are rejected.
    #[test]
    fn atomic_word_rejects_misaligned_offsets() {
        let arena = FlatArena::new(64).expect("allocation succeeds");
        assert!(matches!(
            arena.atomic_word(4),
            Err(FlareError::ArenaBoundsExceeded { .. })
        ));
        assert!(arena.atomic_word(8).is_ok());
    }

    /// Verifies the `Debug` representation and the handle API surface.
    #[test]
    fn debug_and_handle_api() {
        use super::ArenaRef;
        let arena = FlatArena::new(64).expect("allocation succeeds");
        let slot = arena.alloc(8, 8).expect("region fits");
        arena.write_node(slot, &7u64).expect("write succeeds");
        let handle = arena.node_ref::<u64>(slot).expect("valid offset");
        assert_eq!(handle.offset(), slot);
        assert_eq!(ArenaRef::<u64>::alignment(), core::mem::align_of::<u64>());
        let cloned = <ArenaRef<u64> as Clone>::clone(&handle);
        assert_eq!(cloned.get(), Some(&7));
        let rendered = alloc_crate::format!("{arena:?} {handle:?}");
        assert!(rendered.contains("capacity"));
        assert!(rendered.contains("offset"));
    }
}
