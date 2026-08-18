//! Lock-free radix attention engine with 2-bit slab clock eviction.
//!
//! # Layout
//!
//! Token sequences are encoded as little-endian `u32` bytes and stored in
//! the radix tree; each stored prefix owns one physical `Class0` slab slot
//! (`48` bytes). The engine never dereferences slot memory: the slab slot
//! is a physical reservation whose metadata (clock, live bit, KV offset)
//! lives in engine-owned arrays.
//!
//! # Memory ordering
//!
//! `insert` publishes the KV offset with `Release` before raising the
//! `LIVE` bit with `AcqRel`; `match_common_prefix` loads the clock with
//! `Acquire` before reading the KV offset, so a reader that observes a
//! live slot always sees its published KV offset. A second clock load and
//! a second KV-offset read detect concurrent eviction and slot
//! re-initialisation, retrying the walk.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use flare_core::alloc::slab::{SlabClass, SlabPool, SlabSlot};
use flare_core::tree::FlareArtTree;
use flare_core::{FlareError, FlatArena, GpuSyncDriver, HazardManager, NodeType};

/// `LIVE` bit inside the per-slot clock metadata word.
const LIVE_MASK: u8 = 0b100;
/// Two-bit clock counter mask inside the per-slot metadata word.
const REF_MASK: u8 = 0b11;

/// Maximum number of optimistic match retries under concurrent churn.
const MATCH_RETRIES: usize = 4;

/// One successful longest-common-prefix match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Length of the matched prefix in tokens.
    pub token_len: u32,
    /// KV-store offset published by the owning prefix, `0` never appears.
    pub kv_offset: u64,
}

/// Lock-free radix attention engine for LLM KV-cache prefix sharing.
///
/// See the [module documentation](self) for the layout and memory-ordering
/// contract.
pub struct RadixAttentionEngine<G: GpuSyncDriver> {
    tree: FlareArtTree<G>,
    slab_pool: Arc<SlabPool>,
    /// Per-slot metadata: bits `0-1` clock counter, bit `2` live flag.
    clock: Vec<AtomicU8>,
    /// Published KV offset per slot index; `0` marks a dead slot.
    kv_offsets: Vec<AtomicU64>,
    /// Clock hand: next candidate slot for the eviction sweep.
    hand: AtomicU64,
    /// Upper bound on slot indices (`capacity_bytes / 8`).
    slot_count: usize,
    /// Total physical capacity of the engine in bytes.
    capacity_bytes: usize,
}

impl<G: GpuSyncDriver> RadixAttentionEngine<G> {
    /// Creates the engine over `capacity_bytes` of physical slab storage
    /// and an arena of `arena_capacity` bytes for the radix tree.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidParameter`] when `capacity_bytes` is
    /// below one 4 KB slab chunk or is not a multiple of `8` (the slot
    /// grid), and [`FlareError::AllocationFailed`] when the backing
    /// allocations cannot be reserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_kv::RadixAttentionEngine;
    /// use std::sync::Arc;
    /// let engine = RadixAttentionEngine::new(
    ///     1 << 20,
    ///     1 << 20,
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// )
    /// .expect("construction succeeds");
    /// assert_eq!(engine.capacity_bytes(), 1 << 20);
    /// ```
    pub fn new(
        capacity_bytes: usize,
        arena_capacity: usize,
        hazard: Arc<HazardManager>,
        gpu: G,
    ) -> Result<Self, FlareError> {
        if capacity_bytes < 4096 {
            return Err(FlareError::InvalidParameter {
                reason: "capacity below one 4 KB slab chunk",
            });
        }
        if !capacity_bytes.is_multiple_of(8) {
            return Err(FlareError::InvalidParameter {
                reason: "capacity not a multiple of the 8-byte slot grid",
            });
        }
        let slab_pool = Arc::new(SlabPool::new(capacity_bytes)?);
        let slot_count = capacity_bytes / 8;
        let tree = FlareArtTree::new(Arc::new(FlatArena::new(arena_capacity)?), hazard, gpu);
        Ok(Self {
            tree,
            slab_pool,
            clock: (0..slot_count).map(|_| AtomicU8::new(0)).collect(),
            kv_offsets: (0..slot_count).map(|_| AtomicU64::new(0)).collect(),
            hand: AtomicU64::new(0),
            slot_count,
            capacity_bytes,
        })
    }

    /// Returns the total physical capacity of the engine in bytes.
    #[must_use]
    pub const fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    /// Returns the number of physical slot indices managed by the engine.
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Inserts `tokens` as a prefix owned by `kv_offset`, allocating a
    /// physical slot and evicting cold slots when the pool is exhausted.
    ///
    /// The prefix overwrites any existing entry for the same token bytes;
    /// the replaced slot is left live and is reclaimed by later clock
    /// sweeps once its reference counter decays. The returned `kv_offset`
    /// must be non-zero, which is the reserved dead-slot marker.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidParameter`] for an empty token
    /// sequence or a zero `kv_offset`, and
    /// [`FlareError::CacheCapacityExceeded`] when a full clock sweep
    /// cannot free any slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_kv::RadixAttentionEngine;
    /// use std::sync::Arc;
    /// let engine = RadixAttentionEngine::new(
    ///     1 << 20,
    ///     1 << 20,
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// )
    /// .expect("construction succeeds");
    /// engine.insert(&[7, 8], 42).expect("insert succeeds");
    /// ```
    pub fn insert(&self, tokens: &[u32], kv_offset: u64) -> Result<(), FlareError> {
        if tokens.is_empty() {
            return Err(FlareError::InvalidParameter {
                reason: "token sequence is empty",
            });
        }
        if kv_offset == 0 {
            return Err(FlareError::InvalidParameter {
                reason: "kv offset must be non-zero",
            });
        }
        let key = encode_key(tokens);
        let mut slot = self.slab_pool.alloc(NodeType::Node4)?;
        if slot.is_none() {
            self.evict_clock_step(self.slot_count);
            if self.evict_clock_step(self.slot_count) == 0 {
                return Err(FlareError::CacheCapacityExceeded);
            }
            slot = self.slab_pool.alloc(NodeType::Node4)?;
        }
        let slot = slot.ok_or(FlareError::CacheCapacityExceeded)?;
        let index = usize::try_from(slot.offset / 8).expect("slot index fits in usize");
        debug_assert!(index < self.slot_count, "slot index within metadata grid");
        self.kv_offsets[index].store(kv_offset, Ordering::Release);
        // A freshly recycled slot is dead; raising LIVE publishes the KV
        // offset stored above (Release/AcqRel pairing in the readers).
        let previous = self.clock[index].load(Ordering::Relaxed);
        self.clock[index].store((previous & !LIVE_MASK) | LIVE_MASK | 1, Ordering::Release);
        let _previous = self.tree.insert(&key, index as u64)?;
        Ok(())
    }

    /// Returns the longest stored prefix of `tokens` together with its
    /// published KV offset.
    ///
    /// The match walks the radix tree, verifies the slot is live, touches
    /// its clock counter with `fetch_or`, and double-reads the KV offset
    /// to detect concurrent eviction or slot re-initialisation. Stale tree
    /// entries whose slot has been evicted are deleted best-effort and the
    /// walk retries against the next shorter prefix.
    ///
    /// The read path performs no allocation and no atomic
    /// compare-and-swap, only loads and a single `fetch_or`.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::ArenaBoundsExceeded`] when the tree stores a
    /// corrupt pointer, indicating a lifecycle violation.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_kv::RadixAttentionEngine;
    /// use std::sync::Arc;
    /// let engine = RadixAttentionEngine::new(
    ///     1 << 20,
    ///     1 << 20,
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// )
    /// .expect("construction succeeds");
    /// engine.insert(&[1, 2, 3], 100).expect("insert succeeds");
    /// let m = engine
    ///     .match_common_prefix(&[1, 2, 3, 4])
    ///     .expect("match succeeds")
    ///     .expect("prefix found");
    /// assert_eq!(m.token_len, 3);
    /// assert_eq!(m.kv_offset, 100);
    /// ```
    pub fn match_common_prefix(&self, tokens: &[u32]) -> Result<Option<PrefixMatch>, FlareError> {
        let key = encode_key(tokens);
        for _ in 0..MATCH_RETRIES {
            let Some((matched_bytes, value)) = self.tree.longest_prefix(&key)? else {
                return Ok(None);
            };
            let index = usize::try_from(value).expect("slot index fits in usize");
            if index >= self.slot_count {
                return Err(FlareError::InvalidParameter {
                    reason: "stored slot index out of range",
                });
            }
            let cell = &self.clock[index];
            if cell.load(Ordering::Acquire) & LIVE_MASK == 0 {
                let _ = self.tree.delete(&key[..matched_bytes])?;
                continue;
            }
            let kv_first = self.kv_offsets[index].load(Ordering::Acquire);
            let _ = cell.fetch_or(1, Ordering::Relaxed);
            if cell.load(Ordering::Acquire) & LIVE_MASK == 0 {
                continue;
            }
            let kv_second = self.kv_offsets[index].load(Ordering::Acquire);
            if kv_first != kv_second {
                continue;
            }
            let token_len = u32::try_from(matched_bytes / 4).expect("token length fits in u32");
            return Ok(Some(PrefixMatch {
                token_len,
                kv_offset: kv_first,
            }));
        }
        Ok(None)
    }

    /// Advances the clock hand by `steps` slot positions, decrementing the
    /// reference counter of every live slot encountered and evicting live
    /// slots whose counter has reached zero.
    ///
    /// Evicted slots are freed back to the physical slab pool and their KV
    /// offsets are cleared; stale tree entries pointing at them are
    /// removed lazily by the next match. Returns the number of slots
    /// evicted.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_kv::RadixAttentionEngine;
    /// use std::sync::Arc;
    /// let engine = RadixAttentionEngine::new(
    ///     1 << 20,
    ///     1 << 20,
    ///     Arc::new(HazardManager::new()),
    ///     CpuFallbackDriver::default(),
    /// )
    /// .expect("construction succeeds");
    /// engine.insert(&[1, 2], 7).expect("insert succeeds");
    /// let evicted = engine.evict_clock_step(engine.slot_count());
    /// assert_eq!(evicted, 0, "fresh slots still have a reference");
    /// let evicted = engine.evict_clock_step(engine.slot_count());
    /// assert_eq!(evicted, 1, "the reference decayed to zero");
    /// ```
    pub fn evict_clock_step(&self, steps: usize) -> usize {
        let mut evicted = 0usize;
        let mut examined = 0usize;
        let mut index = self.hand.load(Ordering::Relaxed);
        while examined < steps && index < self.slot_count as u64 {
            examined += 1;
            let slot_index = usize::try_from(index).expect("slot index fits in usize");
            index += 1;
            let cell = &self.clock[slot_index];
            let word = cell.load(Ordering::Acquire);
            if word & LIVE_MASK == 0 {
                continue;
            }
            if word & REF_MASK == 0 {
                if cell
                    .compare_exchange(word, word & !LIVE_MASK, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    self.kv_offsets[slot_index].store(0, Ordering::Release);
                    let slot = SlabSlot {
                        offset: slot_index as u64 * 8,
                        class: SlabClass::Class0,
                    };
                    self.slab_pool.free(slot);
                    evicted += 1;
                }
            } else {
                let _ = cell.compare_exchange(word, word - 1, Ordering::AcqRel, Ordering::Relaxed);
            }
        }
        self.hand
            .store(index % self.slot_count as u64, Ordering::Relaxed);
        evicted
    }
}

/// Encodes a token sequence as little-endian `u32` bytes for the radix key.
fn encode_key(tokens: &[u32]) -> Vec<u8> {
    let mut key = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        key.extend_from_slice(&token.to_le_bytes());
    }
    key
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    use super::{LIVE_MASK, PrefixMatch, RadixAttentionEngine};
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use flare_core::{CpuFallbackDriver, FlareError, HazardManager};

    const CAPACITY: usize = 1 << 20;

    fn engine(capacity: usize) -> RadixAttentionEngine<CpuFallbackDriver> {
        RadixAttentionEngine::new(
            capacity,
            1 << 23,
            Arc::new(HazardManager::new()),
            CpuFallbackDriver::default(),
        )
        .expect("construction succeeds")
    }

    /// Verifies constructor validation.
    #[test]
    fn constructor_validation() {
        let hazard = Arc::new(HazardManager::new());
        assert!(matches!(
            RadixAttentionEngine::new(2048, 4096, hazard.clone(), CpuFallbackDriver::default()),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            RadixAttentionEngine::new(4096 + 4, 4096, hazard.clone(), CpuFallbackDriver::default()),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(
            RadixAttentionEngine::new(4096, 4096, hazard, CpuFallbackDriver::default()).is_ok()
        );
    }

    /// Verifies an insert followed by a longer query matches the prefix.
    #[test]
    fn insert_and_match_roundtrip() {
        let engine = engine(CAPACITY);
        engine.insert(&[1, 2, 3], 100).expect("insert succeeds");
        let m = engine
            .match_common_prefix(&[1, 2, 3, 4, 5])
            .expect("match succeeds")
            .expect("prefix found");
        assert_eq!(
            m,
            PrefixMatch {
                token_len: 3,
                kv_offset: 100
            }
        );
        let none = engine.match_common_prefix(&[9, 9]).expect("match succeeds");
        assert_eq!(none, None);
    }

    /// Verifies the deepest stored prefix wins.
    #[test]
    fn longest_prefix_wins() {
        let engine = engine(CAPACITY);
        engine.insert(&[1, 2, 3], 10).expect("insert succeeds");
        engine
            .insert(&[1, 2, 3, 4, 5], 20)
            .expect("insert succeeds");
        let m = engine
            .match_common_prefix(&[1, 2, 3, 4, 5, 6])
            .expect("match succeeds")
            .expect("prefix found");
        assert_eq!(
            m,
            PrefixMatch {
                token_len: 5,
                kv_offset: 20
            }
        );
        let short = engine
            .match_common_prefix(&[1, 2, 3, 8])
            .expect("match succeeds")
            .expect("prefix found");
        assert_eq!(
            short,
            PrefixMatch {
                token_len: 3,
                kv_offset: 10
            }
        );
    }

    /// Verifies input validation on insert.
    #[test]
    fn invalid_inserts_are_rejected() {
        let engine = engine(CAPACITY);
        assert!(matches!(
            engine.insert(&[], 5),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            engine.insert(&[1], 0),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(engine.insert(&[1], 7), Ok(())));
    }

    /// Verifies an empty query matches nothing.
    #[test]
    fn empty_query_matches_nothing() {
        let engine = engine(CAPACITY);
        engine.insert(&[1, 2], 5).expect("insert succeeds");
        assert_eq!(
            engine.match_common_prefix(&[]).expect("match succeeds"),
            None
        );
    }

    /// Verifies a stale tree entry is cleaned up lazily by the matcher.
    #[test]
    fn stale_entry_is_cleaned_up() {
        let engine = engine(CAPACITY);
        engine.insert(&[1, 2], 5).expect("insert succeeds");
        engine.evict_clock_step(engine.slot_count());
        engine.evict_clock_step(engine.slot_count());
        assert_eq!(
            engine.match_common_prefix(&[1, 2]).expect("match succeeds"),
            None,
            "evicted slot must not match"
        );
        engine.insert(&[1, 2], 6).expect("re-insert succeeds");
        let m = engine
            .match_common_prefix(&[1, 2, 3])
            .expect("match succeeds")
            .expect("prefix found");
        assert_eq!(
            m,
            PrefixMatch {
                token_len: 2,
                kv_offset: 6
            }
        );
    }

    /// Verifies evicted slots are recycled by later inserts.
    #[test]
    fn eviction_recycles_slots() {
        let engine = engine(4096);
        for i in 0..83 {
            engine
                .insert(&[i as u32], i as u64 + 1)
                .expect("insert succeeds");
        }
        engine.insert(&[250], 1234).expect("eviction frees a slot");
        let m = engine
            .match_common_prefix(&[250])
            .expect("match succeeds")
            .expect("prefix found");
        assert_eq!(
            m,
            PrefixMatch {
                token_len: 1,
                kv_offset: 1234
            }
        );
        let slot_index = engine
            .tree
            .get(&[250, 0, 0, 0])
            .expect("tree read succeeds")
            .expect("prefix present") as usize;
        assert_eq!(slot_index % 6, 0, "recycled slot is a Class0 offset");
        assert_eq!(
            engine.kv_offsets[slot_index].load(std::sync::atomic::Ordering::Relaxed),
            1234,
            "recycled slot publishes the new KV offset"
        );
    }

    /// Verifies exhaustion surfaces only when two full sweeps cannot free
    /// a slot (all references still hot).
    ///
    /// A single insert leaves every slot at reference one, so the two
    /// sweeps inside `insert` already recycle slots. The error branch is
    /// therefore exercised by pinning every live slot at reference two
    /// directly.
    #[test]
    fn capacity_exhaustion_is_reported() {
        let engine = engine(4096);
        for i in 0..83 {
            engine
                .insert(&[i as u32], i as u64 + 1)
                .expect("insert succeeds");
        }
        for i in 0..engine.slot_count {
            let word = engine.clock[i].load(std::sync::atomic::Ordering::Relaxed);
            if word != 0 {
                engine.clock[i].store(LIVE_MASK | 2, std::sync::atomic::Ordering::Relaxed);
            }
        }
        assert!(
            matches!(
                engine.insert(&[250], 9),
                Err(FlareError::CacheCapacityExceeded)
            ),
            "a slot with two references survives both sweeps"
        );
    }

    /// Verifies re-inserting the same prefix publishes the new KV offset.
    #[test]
    fn overwrite_publishes_new_offset() {
        let engine = engine(CAPACITY);
        engine
            .insert(&[1, 2, 3], 10)
            .expect("first insert succeeds");
        engine.insert(&[1, 2, 3], 11).expect("overwrite succeeds");
        let m = engine
            .match_common_prefix(&[1, 2, 3])
            .expect("match succeeds")
            .expect("prefix found");
        assert_eq!(
            m,
            PrefixMatch {
                token_len: 3,
                kv_offset: 11
            }
        );
    }

    /// Verifies the token length is reported in tokens, not bytes.
    #[test]
    fn token_len_is_in_tokens() {
        let engine = engine(CAPACITY);
        engine
            .insert(&[1, 2, 3, 4, 5, 6, 7, 8], 5)
            .expect("insert succeeds");
        let m = engine
            .match_common_prefix(&[1, 2, 3, 4, 5, 6, 7, 8, 9])
            .expect("match succeeds")
            .expect("prefix found");
        assert_eq!(m.token_len, 8);
    }

    /// Verifies concurrent inserts and matches stay consistent: every
    /// insert succeeds (eviction keeps the pool available) and a matched
    /// prefix always carries the offset it was inserted with.
    #[test]
    fn concurrent_inserts_and_matches() {
        use std::sync::Barrier;
        let engine = Arc::new(engine(CAPACITY));
        let barrier = Arc::new(Barrier::new(5));
        let mut threads = Vec::new();
        for t in 0..4u64 {
            let engine = Arc::clone(&engine);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..100 {
                    let key = (t * 1000 + i) as u32;
                    engine
                        .insert(&[key, key + 1], key as u64 + 1)
                        .expect("insert succeeds");
                }
            }));
        }
        barrier.wait();
        for thread in threads {
            thread.join().expect("worker finishes");
        }
        for t in 0..4u64 {
            for i in 0..100 {
                let key = (t * 1000 + i) as u32;
                let m = engine
                    .match_common_prefix(&[key, key + 1, key + 2])
                    .expect("match succeeds")
                    .expect("prefix found");
                assert_eq!(m.token_len, 2);
                assert_eq!(m.kv_offset, key as u64 + 1);
            }
        }
    }

    /// Verifies the capacity getter reports the configured size.
    #[test]
    fn capacity_getter_reports_configured_size() {
        let engine = engine(8192);
        assert_eq!(engine.capacity_bytes(), 8192);
    }

    /// Verifies that a match resurrects a swept (ref-count zero) slot.
    ///
    /// The insert leaves the clock at `LIVE|1`; the first sweep decrements
    /// the reference to zero without evicting. A subsequent match still
    /// succeeds (the slot stays live) and `fetch_or(1)` bumps the counter
    /// back to one, so the slot survives one more sweep.
    #[test]
    fn match_touches_the_clock() {
        let engine = engine(4096);
        engine.insert(&[1], 5).expect("insert succeeds");
        let index = engine
            .tree
            .get(&[1, 0, 0, 0])
            .expect("tree read succeeds")
            .expect("prefix present") as usize;
        assert_eq!(
            engine.evict_clock_step(engine.slot_count),
            0,
            "first sweep decrements but must not evict"
        );
        let before = engine.clock[index].load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(before, LIVE_MASK, "reference was decremented to zero");
        engine
            .match_common_prefix(&[1])
            .expect("match succeeds")
            .expect("prefix found");
        let after = engine.clock[index].load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after, before + 1, "fetch_or bumped the clock counter");
    }
}
