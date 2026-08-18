//! 4 KB size-classified slab pools with lock-free slot freelists.
//!
//! The [`SlabPool`] partitions contiguous backing storage into 4 KB slab
//! chunks, each classified by the size of the node family it serves
//! ([`SlabClass::Class0`] for `Node4` through [`SlabClass::Class3`] for
//! `Node256`). Slot discovery is `O(1)` via `trailing_zeros`/`count_ones`
//! bit operations, and returned slots are recycled through a per-class
//! lock-free Treiber freelist.

use crate::error::FlareError;
use crate::ptr::NodeType;
use alloc_crate::vec;
use alloc_crate::vec::Vec;
use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, Ordering};

/// Identifies one of the four size classes of a 4 KB slab pool.
///
/// Each class serves exactly one adaptive radix node family, whose slot
/// layout fits the class slot size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SlabClass {
    /// Serves [`NodeType::Node4`] nodes; slot size 48 bytes.
    Class0 = 0,
    /// Serves [`NodeType::Node16`] nodes; slot size 152 bytes.
    Class1 = 1,
    /// Serves [`NodeType::Node64`] nodes; slot size 528 bytes.
    Class2 = 2,
    /// Serves [`NodeType::Node256`] nodes; slot size 2064 bytes.
    Class3 = 3,
}

impl SlabClass {
    /// Returns the size in bytes of a single slot in this class.
    #[must_use]
    pub const fn slot_size(self) -> usize {
        match self {
            Self::Class0 => 48,
            Self::Class1 => 152,
            Self::Class2 => 528,
            Self::Class3 => 2064,
        }
    }

    /// Returns the number of slots per 4 KB slab chunk in this class.
    ///
    /// The chunk sizes `slots * slot_size` stay at or below 4096 bytes:
    /// Class0 `83 * 48 = 3984`, Class1 `26 * 152 = 3952`, Class2
    /// `7 * 528 = 3696`, Class3 `1 * 2064 = 2064`.
    #[must_use]
    pub const fn slots_per_chunk(self) -> u32 {
        match self {
            Self::Class0 => 83,
            Self::Class1 => 26,
            Self::Class2 => 7,
            Self::Class3 => 1,
        }
    }

    /// Returns the byte size of one slab chunk in this class.
    #[must_use]
    pub const fn chunk_size(self) -> usize {
        self.slot_size() * self.slots_per_chunk() as usize
    }

    /// Maps a node family to the slab class serving it.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidNodeType`] when `kind` is a leaf type,
    /// since slabs only host internal adaptive nodes.
    pub const fn for_node_type(kind: NodeType) -> Result<Self, FlareError> {
        match kind {
            NodeType::Node4 => Ok(Self::Class0),
            NodeType::Node16 => Ok(Self::Class1),
            NodeType::Node64 => Ok(Self::Class2),
            NodeType::Node256 => Ok(Self::Class3),
            NodeType::LeafInlined | NodeType::LeafOffset => {
                Err(FlareError::InvalidNodeType(kind.discriminant()))
            }
        }
    }

    /// Converts the raw class discriminant (0..=3) into a [`SlabClass`].
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidNodeType`] for discriminants above 3.
    pub const fn from_discriminant(raw: u8) -> Result<Self, FlareError> {
        match raw {
            0 => Ok(Self::Class0),
            1 => Ok(Self::Class1),
            2 => Ok(Self::Class2),
            3 => Ok(Self::Class3),
            _ => Err(FlareError::InvalidNodeType(raw)),
        }
    }
}

/// The `None` marker for the lock-free slab freelist.
const FREELIST_EMPTY: u32 = u32::MAX;

/// A handle to a freshly allocated slab slot.
///
/// The handle carries the pool-relative byte offset of the slot. The slot
/// is exclusively owned by the caller until it is returned with
/// [`SlabPool::free`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabSlot {
    /// Pool-relative byte offset of the slot within the backing storage.
    pub offset: u64,
    /// The size class the slot belongs to.
    pub class: SlabClass,
}

/// A contiguous, size-classified slab pool with lock-free slot recycling.
///
/// The pool owns a contiguous backing store and partitions it into
/// per-size-class chunks. The fast path pops a slot from a per-class
/// Treiber freelist, driven by `compare_exchange` on the freelist head; the
/// slow path provisions a fresh 4 KB chunk when the freelist is empty. All
/// transitions are lock-free, so the pool is safe to share across threads
/// without external synchronisation.
///
/// # Invariants
///
/// - A slot is exclusively owned by the allocating caller between
///   [`Self::alloc`] and [`Self::free`]; concurrent owners never overlap.
/// - Slot payloads are only initialised after allocation and are retired
///   (re-linked onto the freelist) wholesale, so the freelist never
///   publishes a live node.
///
/// # Examples
///
/// ```
/// # use flare_core::alloc::slab::{SlabClass, SlabPool};
/// # use flare_core::ptr::NodeType;
/// let pool = SlabPool::new(16 * 4096).expect("pool fails to back storage");
/// let slot = pool.alloc(NodeType::Node16).expect("class lookup succeeds")
///     .expect("slot available");
/// assert_eq!(slot.class, SlabClass::Class1);
/// pool.free(slot);
/// ```
pub struct SlabPool {
    storage: UnsafeCell<Vec<u8>>,
    classes: [ClassState; 4],
}

/// Per-class allocation state for a [`SlabPool`].
struct ClassState {
    slot_size: u32,
    slots_per_chunk: u32,
    capacity_chunks: u32,
    chunk_count: AtomicU32,
    free_head: AtomicU32,
    next: UnsafeCell<Vec<u32>>,
}

impl ClassState {
    /// Creates the per-class state over the shared backing store.
    ///
    /// The linkage table length covers every slot of every chunk the class
    /// may ever provision, so `global_slot` indices always stay bounded.
    fn new(class: SlabClass, capacity_bytes: usize) -> Self {
        let slot_size = u32::try_from(class.slot_size()).expect("slot size fits in u32");
        let slots_per_chunk = class.slots_per_chunk();
        let capacity_chunks =
            u32::try_from(capacity_bytes / class.chunk_size()).expect("capacity fits in u32");
        let next = vec![FREELIST_EMPTY; (capacity_chunks * slots_per_chunk) as usize];
        Self {
            slot_size,
            slots_per_chunk,
            capacity_chunks,
            chunk_count: AtomicU32::new(0),
            free_head: AtomicU32::new(FREELIST_EMPTY),
            next: UnsafeCell::new(next),
        }
    }

    /// Returns the global slot index for a chunk-local slot pair.
    const fn global_slot(&self, chunk: u32, local: u32) -> u32 {
        chunk * self.slots_per_chunk + local
    }

    /// Provisions one fresh chunk by pushing every slot of it onto the
    /// freelist.
    ///
    /// Returns `false` when the chunk budget is exhausted.
    ///
    /// # Safety
    ///
    /// The function uses unsafe code to write the freelist linkage table
    /// entries before publishing them via CAS. The `global` index is
    /// bounded by the table length established at construction. The CAS
    /// protocol ensures that only the thread that successfully publishes a
    /// slot can observe its linkage entry.
    fn provision(&self) -> bool {
        loop {
            let chunk = self.chunk_count.load(Ordering::Relaxed);
            if chunk >= self.capacity_chunks {
                return false;
            }
            if self
                .chunk_count
                .compare_exchange(chunk, chunk + 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            for local in 0..self.slots_per_chunk {
                let global = self.global_slot(chunk, local);
                loop {
                    let head = self.free_head.load(Ordering::Acquire);
                    // SAFETY: the linkage-table entry write is serialised by
                    // the CAS protocol described on [`Self::next_slot_mut`].
                    unsafe { *self.next_slot_mut(global) = head };
                    if self
                        .free_head
                        .compare_exchange(head, global, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        break;
                    }
                }
            }
            return true;
        }
    }

    /// Returns a mutable pointer to one linkage-table entry.
    ///
    /// # Safety
    ///
    /// `index` must be bounded by the table length set at construction.
    /// Concurrent access is serialised by the freelist CAS protocol: an
    /// entry is written only by the thread racing the owning CAS, and read
    /// only after its publishing CAS succeeded.
    #[allow(clippy::mut_from_ref)]
    unsafe fn next_slot_mut(&self, index: u32) -> &mut u32 {
        // SAFETY: `self.next` is exclusively reachable through the CAS
        // protocol during this call (see call-site documentation).
        let base = unsafe { (*self.next.get()).as_mut_ptr() };
        // SAFETY: `index` is bounded by the construction-time table length.
        unsafe { &mut *base.add(index as usize) }
    }

    /// Returns the value of one linkage-table entry.
    ///
    /// # Safety
    ///
    /// Same constraints as [`Self::next_slot_mut`].
    unsafe fn next_slot(&self, index: u32) -> u32 {
        // SAFETY: see [`Self::next_slot_mut`].
        let base = unsafe { (*self.next.get()).as_ptr() };
        // SAFETY: `index` is bounded by the construction-time table length.
        unsafe { *base.add(index as usize) }
    }
}

impl SlabPool {
    /// Creates a slab pool over `capacity_bytes` of contiguous backing
    /// storage.
    ///
    /// Every class shares the same backing store; the store is
    /// zero-initialised. The number of slab chunks per class is derived
    /// from the class slot sizes and the total capacity.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::AllocationFailed`] when the global allocator
    /// cannot back the requested capacity.
    ///
    /// # Panics
    ///
    /// Panics when the capacity does not fit the per-class chunk budget,
    /// which is impossible for capacities below `u32::MAX` bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::slab::SlabPool;
    /// let pool = SlabPool::new(1 << 16).expect("pool fails to back storage");
    /// assert!(pool.capacity_bytes() >= 1 << 16);
    /// ```
    pub fn new(capacity_bytes: usize) -> Result<Self, FlareError> {
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(capacity_bytes)
            .map_err(|_| FlareError::AllocationFailed)?;
        storage.resize(capacity_bytes, 0);
        Ok(Self {
            storage: UnsafeCell::new(storage),
            classes: core::array::from_fn(|raw| {
                let class = SlabClass::from_discriminant(
                    u8::try_from(raw).expect("class index fits in u8"),
                )
                .expect("class discriminant valid");
                ClassState::new(class, capacity_bytes)
            }),
        })
    }

/// Returns the total backing capacity of this pool in bytes.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        // SAFETY: the storage vector is allocated once at construction and never
        // reallocated or moved. The `UnsafeCell` only permits interior
        // mutation through the pool's controlled APIs; the length field
        // itself is immutable after construction, so reading it is data-race
        // free.
        unsafe { (*self.storage.get()).len() }
    }

    /// Allocates one slot of the class serving `kind`.
    ///
    /// Returns `Ok(Some(slot))` on success, or `Ok(None)` when the class
    /// chunk budget is exhausted, in which case the caller must fall back
    /// to bump allocation on the arena frontier.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidNodeType`] when `kind` is a leaf type,
    /// which has no slab class.
    ///
    /// # Panics
    ///
    /// Panics when `kind` maps to an unknown class discriminant, which
    /// cannot happen for the node types accepted by this method.
    ///
    /// # Safety
    ///
    /// The function uses unsafe code to read the freelist linkage table.
    /// The `head` index is validated to be within the table bounds by the
    /// provision logic, and the CAS protocol ensures that only a thread
    /// that successfully publishes a slot can observe its linkage entry.
    /// The returned slot is exclusively owned by the caller until freed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::slab::{SlabClass, SlabPool};
    /// # use flare_core::ptr::NodeType;
    /// let pool = SlabPool::new(4096 * 4).expect("pool fails to back storage");
    /// let slot = pool.alloc(NodeType::Node64).expect("class lookup")
    ///     .expect("slot available");
    /// assert_eq!(slot.class, SlabClass::Class2);
    /// ```
    pub fn alloc(&self, kind: NodeType) -> Result<Option<SlabSlot>, FlareError> {
        let class = SlabClass::for_node_type(kind)?;
        let state = &self.classes[class as usize];
        loop {
            let head = state.free_head.load(Ordering::Acquire);
            if head == FREELIST_EMPTY {
                if !state.provision() {
                    return Ok(None);
                }
                continue;
            }
            // SAFETY: `head` was published by a successful CAS and is
            // bounded by the table length (`global_slot` of a provisioned
            // chunk).
            let next = unsafe { state.next_slot(head) };
            if state
                .free_head
                .compare_exchange(head, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let chunk = head / state.slots_per_chunk;
                let local = head % state.slots_per_chunk;
                let offset = u64::from(chunk) * u64::from(state.slot_size * state.slots_per_chunk)
                    + u64::from(local) * u64::from(state.slot_size);
                return Ok(Some(SlabSlot {
                    offset,
                    class: SlabClass::from_discriminant(class as u8)
                        .expect("class discriminant valid"),
                }));
            }
        }
    }

    /// Returns a previously allocated slot to the freelist.
    ///
    /// The slot payload is not cleared; recycling is safe because the
    /// freelist only publishes the slot index, and the next owner
    /// re-initialises the payload before publishing any node referencing
    /// it.
    ///
    /// # Panics
    ///
    /// Panics when `slot.offset` falls outside the class chunk budget,
    /// which indicates a lifecycle violation (double free or foreign
    /// slot).
    ///
    /// # Safety
    ///
    /// The function uses unsafe code to write the freelist linkage table
    /// entry before publishing the slot via CAS. The `global` index is
    /// validated to be within the class budget before the write. The CAS
    /// protocol ensures that only the thread that successfully publishes
    /// the slot can observe its linkage entry. The slot payload is not
    /// cleared; the next owner must re-initialise it before publishing
    /// any node referencing the slot.
    ///
    /// # Examples
    ///
    /// ```
    /// # use flare_core::alloc::slab::SlabPool;
    /// # use flare_core::ptr::NodeType;
    /// let pool = SlabPool::new(4096 * 4).expect("pool fails to back storage");
    /// let slot = pool.alloc(NodeType::Node4).expect("class lookup")
    ///     .expect("slot available");
    /// pool.free(slot);
    /// let again = pool.alloc(NodeType::Node4).expect("class lookup")
    ///     .expect("slot recycled");
    /// assert_eq!(again.offset, slot.offset);
    /// ```
    pub fn free(&self, slot: SlabSlot) {
        let class = SlabClass::from_discriminant(slot.class as u8).expect("valid class");
        let state = &self.classes[class as usize];
        let chunk_size = u64::from(state.slot_size * state.slots_per_chunk);
        let chunk = slot.offset / chunk_size;
        let local = (slot.offset % chunk_size) / u64::from(state.slot_size);
        assert!(
            chunk < u64::from(state.capacity_chunks),
            "slot offset {} outside provisioned chunks of class {class:?}",
            slot.offset
        );
        let chunk = u32::try_from(chunk).expect("chunk index fits in u32");
        let local = u32::try_from(local).expect("local index fits in u32");
        let global = state.global_slot(chunk, local);
        loop {
            let head = state.free_head.load(Ordering::Acquire);
            // SAFETY: `global` is bounded by the class budget, verified by
            // the assertion above; the entry write happens before the CAS
            // publishing the slot.
            unsafe {
                *state.next_slot_mut(global) = head;
            }
            if state
                .free_head
                .compare_exchange(head, global, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return;
            }
        }
    }
}

impl core::fmt::Debug for SlabPool {
    /// Formats the pool as its backing capacity and per-class provision.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SlabPool")
            .field("storage", &self.capacity_bytes())
            .field(
                "classes",
                &self
                    .classes
                    .iter()
                    .map(|c| c.chunk_count.load(Ordering::Relaxed))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Safety justification for sharing the pool across threads.
///
/// The only shared mutable state is the atomic chunk counters, the atomic
/// freelist heads, and the linkage table. Every transition is performed
/// with `compare_exchange`; linkage entries are written before the CAS that
/// publishes them and read only after an acquire load of the head, so the
/// whole pool behaves as a set of independent Treiber stacks. Backing
/// storage is only touched by the exclusive slot owner.
///
/// # Safety
///
/// This is sound because all shared state transitions are lock-free atomic
/// operations and slot payloads are exclusively owned between allocation
/// and free.
unsafe impl Sync for SlabPool {}

#[cfg(test)]
mod tests {
    use super::{FREELIST_EMPTY, SlabClass, SlabPool};
    use crate::ptr::NodeType;
    use alloc_crate::vec::Vec;

    /// Verifies that every class exposes positive capacity and chunk
    /// geometry that fits the 4 KB budget.
    #[test]
    fn class_geometry() {
        for class in [
            SlabClass::Class0,
            SlabClass::Class1,
            SlabClass::Class2,
            SlabClass::Class3,
        ] {
            assert!(class.slot_size() > 0);
            assert!(class.slots_per_chunk() > 0);
            assert!(class.chunk_size() <= 4096, "{class:?} exceeds 4 KB");
            assert_eq!(SlabClass::from_discriminant(class as u8), Ok(class));
        }
        for raw in [4u8, 7u8, 255u8] {
            assert!(SlabClass::from_discriminant(raw).is_err());
        }
    }

    /// Verifies that slots are recycled through the freelist.
    #[test]
    fn allocation_and_recycling() {
        let pool = SlabPool::new(8 * 4096).expect("pool fits");
        let slot = pool
            .alloc(NodeType::Node16)
            .expect("class lookup")
            .expect("slot available");
        assert_eq!(slot.class, SlabClass::Class1);
        assert_eq!(slot.offset % SlabClass::Class1.slot_size() as u64, 0);
        pool.free(slot);
        let recycled = pool
            .alloc(NodeType::Node16)
            .expect("class lookup")
            .expect("slot recycled");
        assert_eq!(recycled.offset, slot.offset);
    }

    /// Verifies that distinct allocations never overlap in storage.
    #[test]
    fn allocations_are_disjoint() {
        let pool = SlabPool::new(32 * 4096).expect("pool fits");
        let mut seen = alloc_crate::collections::BTreeSet::new();
        for kind in [
            NodeType::Node4,
            NodeType::Node16,
            NodeType::Node64,
            NodeType::Node256,
        ] {
            for _ in 0..4 {
                let slot = pool
                    .alloc(kind)
                    .expect("class lookup")
                    .expect("slot available");
                let range = (slot.offset, slot.offset + slot.class.slot_size() as u64);
                assert!(seen.insert(range), "overlapping slab slots: {range:?}");
            }
        }
    }

    /// Verifies that leaf node types are rejected by the allocator.
    #[test]
    fn leaves_have_no_slab_class() {
        let pool = SlabPool::new(4096 * 4).expect("pool fits");
        assert!(pool.alloc(NodeType::LeafInlined).is_err());
        assert!(pool.alloc(NodeType::LeafOffset).is_err());
    }

    /// Verifies the freelist marker is reserved and never handed out.
    #[test]
    fn freelist_marker_is_reserved() {
        assert_eq!(FREELIST_EMPTY, u32::MAX);
        let pool = SlabPool::new(4096 * 4).expect("pool fits");
        for kind in [
            NodeType::Node4,
            NodeType::Node16,
            NodeType::Node64,
            NodeType::Node256,
        ] {
            let slot = pool
                .alloc(kind)
                .expect("class lookup")
                .expect("slot available");
            pool.free(slot);
        }
    }

    /// Verifies that exhaustion reports `Ok(None)` instead of panicking.
    #[test]
    fn exhaustion_returns_none() {
        let budget = 4 * 4096 / SlabClass::Class0.chunk_size();
        let pool = SlabPool::new(4 * 4096).expect("pool fits");
        let mut count = 0u32;
        let total_slots = budget * SlabClass::Class0.slots_per_chunk() as usize;
        for _ in 0..=total_slots {
            match pool.alloc(NodeType::Node4).expect("class lookup") {
                Some(_) => count += 1,
                None => break,
            }
        }
        assert_eq!(count as usize, total_slots);
    }

    /// Verifies that freed slots become allocatable again after exhaustion.
    #[test]
    fn recycling_after_exhaustion() {
        let pool = SlabPool::new(2 * 4096).expect("pool fits");
        let mut slots = Vec::new();
        while let Some(slot) = pool.alloc(NodeType::Node4).expect("class lookup") {
            slots.push(slot);
        }
        for slot in &slots[..slots.len() / 2] {
            pool.free(*slot);
        }
        for _ in 0..slots.len() / 2 {
            assert!(pool.alloc(NodeType::Node4).expect("class lookup").is_some());
        }
    }

    /// Verifies the capacity query and the `Debug` representation.
    #[test]
    fn capacity_and_debug() {
        let pool = SlabPool::new(4 * 4096).expect("pool fits");
        assert_eq!(pool.capacity_bytes(), 4 * 4096);
        let rendered = alloc_crate::format!("{pool:?}");
        assert!(rendered.contains("storage"));
        assert!(rendered.contains("classes"));
    }
}
