//! Arena-resident IVF-PQ index with lock-free snapshot handoff.
//!
//! Every piece of index state — centroids, codebooks, posting lists, the
//! vector journal, and the snapshot header — lives in the engine-owned
//! [`FlatArena`] and is published through a single `AtomicU64` handoff
//! cell. Readers load the handoff with an acquire-load and then walk
//! immutable arena blocks; writers publish a freshly built snapshot with a
//! single compare-and-swap, so [`IvfPqIndex::trigger_shadow_reclustering`]
//! swaps the working snapshot lock-free. Obsolete generations are retained
//! in the append-only arena, mirroring the hazard-era retirement model of
//! `flare-core`.

use crate::codebook::{CODES_PER_SUBVECTOR, PqCodebooks};
use crate::distance::l2_sq_dispatch;
use crate::kmeans::kmeans_l2;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use flare_core::{FlareArtTree, FlareError, FlatArena, GpuSyncDriver, HazardManager};

/// Default number of Lloyd iterations used by training and re-clustering.
const TRAIN_ITERATIONS: usize = 24;

/// Arena offset of the snapshot handoff cell.
const ROOT_CELL_OFFSET: u64 = 0;

/// Bit width of the generation counter inside the handoff cell.
const GENERATION_BITS: u32 = 24;

/// Mask isolating the 40-bit state-block offset inside the handoff cell.
const STATE_OFFSET_MASK: u64 = (1 << 40) - 1;

/// Packs a generation and state offset into one handoff word.
const fn pack_root(generation: u64, state_offset: u64) -> u64 {
    let masked = generation & ((1 << GENERATION_BITS) - 1);
    (masked << 40) | (state_offset & STATE_OFFSET_MASK)
}

/// Unpacks the generation and state offset from a handoff word.
const fn unpack_root(word: u64) -> (u64, u64) {
    (word >> 40, word & STATE_OFFSET_MASK)
}

/// Immutable snapshot header stored in the arena.
///
/// All offsets reference arena blocks that are written exactly once before
/// the snapshot is published and never rewritten afterwards.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StateHeader {
    /// Offset of the flat `f32` centroid block.
    centroids_off: u64,
    /// Number of centroids.
    centroids_count: u64,
    /// Offset of the flat `f32` codebook block.
    codebooks_off: u64,
    /// Offset of the per-centroid posting head cells.
    posting_heads_off: u64,
    /// Number of posting head cells (equals `centroids_count`).
    posting_heads_count: u64,
    /// Offset of the journal head cell.
    journal_off: u64,
    /// Offset of the journal counter cell.
    journal_count: u64,
    /// Offset of the centroid routing-code block.
    centroid_codes_off: u64,
    /// Offset of the flat `f32` routing codebook block (raw domain).
    routing_codebooks_off: u64,
}

/// Fixed 16-byte header of one posting block.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PostingHeader {
    /// Offset of the previous block in the linked list, or `0`.
    next: u64,
    /// Insertion-sequence identifier of the stored vector.
    id: u64,
}

/// Fixed 16-byte header of one journal block.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JournalHeader {
    /// Offset of the previous block in the linked list, or `0`.
    next: u64,
    /// Insertion-sequence identifier of the stored vector.
    id: u64,
}

/// One candidate hit produced by [`IvfPqIndex::search`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// Insertion-sequence identifier of the stored vector.
    pub id: u64,
    /// Asymmetric (ADC) L2-squared distance to the query; lower is closer.
    pub distance: f32,
}

/// An arena-resident IVF-PQ vector index.
///
/// The index routes queries through a radix tree keyed by centroid
/// routing codes (`O(log C)` per spec), scores candidates with an ADC
/// distance table built from the query, and trains its centroids and
/// codebooks with deterministic Lloyd iterations. All mutation is
/// lock-free; a single CAS on the handoff cell publishes new snapshots
/// produced by [`Self::train`] or [`Self::trigger_shadow_reclustering`].
///
/// # Concurrency model
///
/// - Readers load the handoff cell once (acquire) and treat the addressed
///   snapshot as immutable.
/// - [`Self::insert`] appends to the journal and posting lists of the
///   snapshot it read, then verifies the handoff did not change; on a
///   changed handoff the entry is retried against the new snapshot (the
///   orphaned block is retained as harmless garbage, mirroring arena
///   append-only semantics).
/// - [`Self::trigger_shadow_reclustering`] trains on the journal, builds a
///   complete shadow snapshot, publishes it with one CAS, and only then
///   refreshes the routing tree. Stale routing entries remain valid
///   (they address the same posting-head count) and are cleaned on the
///   next rebuild.
///
/// The arena must be sized generously: each rebuild allocates a fresh
/// snapshot while old generations are retained.
pub struct IvfPqIndex<G: GpuSyncDriver> {
    dimension: usize,
    n_centroids: usize,
    sub_vectors: usize,
    seed: u64,
    arena: Arc<FlatArena>,
    router: FlareArtTree<G>,
}

impl<G: GpuSyncDriver> IvfPqIndex<G> {
    /// Creates an empty index over a fresh engine-owned arena.
    ///
    /// `arena_capacity` sizes the backing arena; the handoff cell,
    /// training data, postings, and every shadow generation are allocated
    /// from it.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidParameter`] when `dimension`,
    /// `sub_vectors`, or `n_centroids` is zero, or when `dimension` is
    /// not divisible by `sub_vectors`. Returns
    /// [`FlareError::AllocationFailed`] when the arena backing store
    /// cannot be allocated.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_vector::IvfPqIndex;
    /// use std::sync::Arc;
    /// let index = IvfPqIndex::new(
    ///     4, 2, 2, 7, 1 << 20, Arc::new(HazardManager::new()), CpuFallbackDriver::default(),
    /// ).expect("construction succeeds");
    /// assert!(!index.is_trained());
    /// ```
    pub fn new(
        dimension: usize,
        n_centroids: usize,
        sub_vectors: usize,
        seed: u64,
        arena_capacity: usize,
        hazard: Arc<HazardManager>,
        gpu: G,
    ) -> Result<Self, FlareError> {
        if dimension == 0 {
            return Err(FlareError::InvalidParameter {
                reason: "dimension is zero",
            });
        }
        if sub_vectors == 0 {
            return Err(FlareError::InvalidParameter {
                reason: "sub-vector count is zero",
            });
        }
        if !dimension.is_multiple_of(sub_vectors) {
            return Err(FlareError::InvalidParameter {
                reason: "dimension not divisible by sub-vector count",
            });
        }
        if n_centroids == 0 {
            return Err(FlareError::InvalidParameter {
                reason: "centroid count is zero",
            });
        }
        let arena = Arc::new(FlatArena::new(arena_capacity)?);
        let root_off = arena.alloc(8, 8)?;
        debug_assert_eq!(
            root_off, ROOT_CELL_OFFSET,
            "handoff cell must live at offset 0"
        );
        arena.write_node(root_off, &0u64)?;
        let router = FlareArtTree::new(arena.clone(), hazard, gpu);
        Ok(Self {
            dimension,
            n_centroids,
            sub_vectors,
            seed,
            arena,
            router,
        })
    }

    /// Returns the vector dimension of the index.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of sub-vectors (code bytes per vector).
    #[must_use]
    pub const fn sub_vectors(&self) -> usize {
        self.sub_vectors
    }

    /// Returns the centroid count of the index.
    #[must_use]
    pub const fn n_centroids(&self) -> usize {
        self.n_centroids
    }

    /// Reports whether a snapshot has been published by [`Self::train`].
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.root_cell().load(Ordering::Acquire) != 0
    }

    /// Returns the number of vectors inserted since index creation.
    ///
    /// The count reflects the journal, which is retained across
    /// re-clustering rounds.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::InvalidParameter`] when the index has not
    /// been trained yet.
    pub fn vector_count(&self) -> Result<u64, FlareError> {
        let Some(state) = self.load_state()? else {
            return Err(FlareError::InvalidParameter {
                reason: "index has not been trained",
            });
        };
        Ok(self
            .arena
            .atomic_word(state.journal_count)?
            .load(Ordering::Acquire))
    }

    /// Trains centroids and codebooks over `samples`, publishing the first
    /// snapshot.
    ///
    /// `samples` must hold whole rows of `dimension` values. Training is
    /// deterministic for a fixed `(samples, seed)` pair. Calling this on
    /// an already-trained index rebuilds and republishes a new snapshot
    /// from the same `samples`.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::VectorDimensionMismatch`] when `samples` is
    /// not a multiple of `dimension`, and
    /// [`FlareError::InvalidParameter`] when the sample count is smaller
    /// than the centroid count or than the codebook size. Arena capacity
    /// errors propagate as [`FlareError::ArenaCapacityExceeded`].
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_vector::IvfPqIndex;
    /// use std::sync::Arc;
    /// let index = IvfPqIndex::new(
    ///     4, 2, 2, 7, 1 << 20, Arc::new(HazardManager::new()), CpuFallbackDriver::default(),
    /// ).expect("construction succeeds");
    /// let samples: Vec<f32> = (0..512)
    ///     .flat_map(|i| {
    ///         let base = if i % 2 == 0 { 10.0 } else { -10.0 };
    ///         [base, base, base, base]
    ///     })
    ///     .collect();
    /// index.train(&samples).expect("training succeeds");
    /// assert!(index.is_trained());
    /// ```
    pub fn train(&self, samples: &[f32]) -> Result<(), FlareError> {
        self.rebuild(samples)
    }

    /// Inserts one `dimension`-long vector into the index.
    ///
    /// The vector is appended to the journal (so re-clustering can retrain
    /// on it), quantized with the current codebooks, and linked into the
    /// posting list of its nearest centroid. The operation is lock-free
    /// and retries internally when a concurrent re-clustering swaps the
    /// snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::VectorDimensionMismatch`] when `vector` is
    /// not exactly `dimension` values long,
    /// [`FlareError::InvalidParameter`] when the index has not been
    /// trained, and arena capacity errors when the arena is exhausted.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_vector::IvfPqIndex;
    /// use std::sync::Arc;
    /// let index = IvfPqIndex::new(
    ///     4, 2, 2, 7, 1 << 20, Arc::new(HazardManager::new()), CpuFallbackDriver::default(),
    /// ).expect("construction succeeds");
    /// let samples: Vec<f32> = (0..512)
    ///     .flat_map(|i| {
    ///         let base = if i % 2 == 0 { 10.0 } else { -10.0 };
    ///         [base, base, base, base]
    ///     })
    ///     .collect();
    /// index.train(&samples).expect("training succeeds");
    /// index.insert(&[10.5, 10.5, 10.5, 10.5]).expect("insert succeeds");
    /// assert_eq!(index.vector_count().expect("count succeeds"), 1);
    /// ```
    pub fn insert(&self, vector: &[f32]) -> Result<(), FlareError> {
        if vector.len() != self.dimension {
            return Err(FlareError::VectorDimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        loop {
            let root_word = self.root_cell().load(Ordering::Acquire);
            let (generation, state_off) = unpack_root(root_word);
            if generation == 0 {
                return Err(FlareError::InvalidParameter {
                    reason: "index has not been trained",
                });
            }
            let state = self.read_header(state_off)?;
            let centroid_idx = self.nearest_centroid(&state, vector)?;
            let cb = self.read_codebooks(&state)?;
            let residual = self.residual_of(&state, centroid_idx, vector)?;
            let code = cb.encode(&residual)?;
            let id = self
                .arena
                .atomic_word(state.journal_count)?
                .fetch_add(1, Ordering::Relaxed);
            let j_off = self.arena.alloc(16 + 4 * self.dimension, 8)?;
            self.arena
                .write_node(j_off, &JournalHeader { next: 0, id })?;
            self.write_f32_block(j_off + 16, vector)?;
            let p_off = self.arena.alloc(16 + self.sub_vectors, 8)?;
            self.arena
                .write_node(p_off, &PostingHeader { next: 0, id })?;
            self.arena.write_bytes(p_off + 16, &code)?;
            let head = self
                .arena
                .atomic_word(state.posting_heads_off + 8 * centroid_idx as u64)?;
            self.push_cas(head, p_off)?;
            let journal = self.arena.atomic_word(state.journal_off)?;
            self.push_cas(journal, j_off)?;
            if self.root_cell().load(Ordering::Acquire) == root_word {
                return Ok(());
            }
        }
    }

    /// Searches the index for the `top_k` nearest stored vectors.
    ///
    /// The query is routed through the centroid radix tree by its routing
    /// code (falling back to an exact centroid scan when the tree misses),
    /// scored with an ADC distance table, and returned sorted by distance
    /// ascending.
    ///
    /// # Errors
    ///
    /// Returns [`FlareError::VectorDimensionMismatch`] when `query` is
    /// not exactly `dimension` values long, and
    /// [`FlareError::InvalidParameter`] when the index has not been
    /// trained.
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_vector::IvfPqIndex;
    /// use std::sync::Arc;
    /// let index = IvfPqIndex::new(
    ///     4, 2, 2, 7, 1 << 20, Arc::new(HazardManager::new()), CpuFallbackDriver::default(),
    /// ).expect("construction succeeds");
    /// let samples: Vec<f32> = (0..512)
    ///     .flat_map(|i| {
    ///         let base = if i % 2 == 0 { 10.0 } else { -10.0 };
    ///         [base, base, base, base]
    ///     })
    ///     .collect();
    /// index.train(&samples).expect("training succeeds");
    /// index.insert(&[10.5, 10.5, 10.5, 10.5]).expect("insert succeeds");
    /// let hits = index.search(&[10.4, 10.4, 10.4, 10.4], 1).expect("search succeeds");
    /// assert_eq!(hits.len(), 1);
    /// ```
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>, FlareError> {
        if query.len() != self.dimension {
            return Err(FlareError::VectorDimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        let Some(state) = self.load_state()? else {
            return Err(FlareError::InvalidParameter {
                reason: "index has not been trained",
            });
        };
        let cb = self.read_codebooks(&state)?;
        let cb_raw = self.read_routing_codebooks(&state)?;
        let centroid_idx = self.route(&state, &cb_raw, query)?;
        let residual_query = self.residual_of(&state, centroid_idx, query)?;
        let table = cb.table(&residual_query)?;
        let mut hits = Vec::new();
        let head = self
            .arena
            .atomic_word(state.posting_heads_off + 8 * centroid_idx as u64)?;
        let mut block = head.load(Ordering::Acquire);
        while block != 0 {
            let header = *self.arena.read_node::<PostingHeader>(block)?;
            let mut dist = 0.0f32;
            for s in 0..self.sub_vectors {
                let code = *self.arena.read_node::<u8>(block + 16 + s as u64)?;
                dist += table[s * CODES_PER_SUBVECTOR + code as usize];
            }
            hits.push(SearchResult {
                id: header.id,
                distance: dist,
            });
            block = header.next;
        }
        hits.sort_unstable_by(|x, y| x.distance.total_cmp(&y.distance));
        hits.truncate(top_k);
        Ok(hits)
    }

    /// Re-trains the whole index on every inserted vector and publishes
    /// the result as a shadow snapshot.
    ///
    /// The journal is the single source of truth: the vectors inserted
    /// since index creation are re-clustered from scratch (centroids,
    /// codebooks, postings), and the new snapshot is published with one
    /// CAS. No-op (returns `Ok`) when the index is untrained or the
    /// journal is empty. Because the arena retains obsolete generations,
    /// callers should size the arena for the expected number of rebuilds.
    ///
    /// # Errors
    ///
    /// Propagates training and arena capacity errors; see [`Self::train`].
    ///
    /// # Examples
    ///
    /// ```
    /// use flare_core::sync::gpu::CpuFallbackDriver;
    /// use flare_core::sync::hazard::HazardManager;
    /// use flare_vector::IvfPqIndex;
    /// use std::sync::Arc;
    /// let index = IvfPqIndex::new(
    ///     4, 2, 2, 7, 1 << 20, Arc::new(HazardManager::new()), CpuFallbackDriver::default(),
    /// ).expect("construction succeeds");
    /// let samples: Vec<f32> = (0..512)
    ///     .flat_map(|i| {
    ///         let base = if i % 2 == 0 { 10.0 } else { -10.0 };
    ///         [base, base, base, base]
    ///     })
    ///     .collect();
    /// index.train(&samples).expect("training succeeds");
    /// index.trigger_shadow_reclustering().expect("re-clustering succeeds");
    /// assert_eq!(index.vector_count().expect("count succeeds"), 0);
    /// ```
    pub fn trigger_shadow_reclustering(&self) -> Result<(), FlareError> {
        let Some(state) = self.load_state()? else {
            return Ok(());
        };
        let mut rows = Vec::new();
        let journal = self.arena.atomic_word(state.journal_off)?;
        let mut block = journal.load(Ordering::Acquire);
        while block != 0 {
            let header = *self.arena.read_node::<JournalHeader>(block)?;
            rows.extend(self.read_f32_block(block + 16, self.dimension)?);
            block = header.next;
        }
        if rows.is_empty() {
            return Ok(());
        }
        self.rebuild(&rows)
    }

    // -- internals ---------------------------------------------------------

    /// Returns a handle to the handoff cell.
    fn root_cell(&self) -> &AtomicU64 {
        self.arena
            .atomic_word(ROOT_CELL_OFFSET)
            .expect("handoff cell is always in bounds")
    }

    /// Loads the current snapshot header, or `None` when untrained.
    fn load_state(&self) -> Result<Option<StateHeader>, FlareError> {
        let (generation, state_off) = unpack_root(self.root_cell().load(Ordering::Acquire));
        if generation == 0 {
            return Ok(None);
        }
        Ok(Some(self.read_header(state_off)?))
    }

    /// Reads the snapshot header at `state_off`.
    fn read_header(&self, state_off: u64) -> Result<StateHeader, FlareError> {
        Ok(*self.arena.read_node::<StateHeader>(state_off)?)
    }

    /// Reads `count` `f32` values from a block, one element per call.
    fn read_f32_block(&self, off: u64, count: usize) -> Result<Vec<f32>, FlareError> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(*self.arena.read_node::<f32>(off + 4 * i as u64)?);
        }
        Ok(out)
    }

    /// Writes `values` into a block, one element per call.
    fn write_f32_block(&self, off: u64, values: &[f32]) -> Result<(), FlareError> {
        for (i, value) in values.iter().enumerate() {
            self.arena.write_node(off + 4 * i as u64, value)?;
        }
        Ok(())
    }

    /// Reconstructs the codebooks from the snapshot's arena block.
    fn read_codebooks(&self, state: &StateHeader) -> Result<PqCodebooks, FlareError> {
        let count = self.sub_vectors * CODES_PER_SUBVECTOR * (self.dimension / self.sub_vectors);
        let data = self.read_f32_block(state.codebooks_off, count)?;
        PqCodebooks::from_centroids(self.dimension, self.sub_vectors, data)
    }

    /// Reconstructs the raw-domain routing codebooks from the arena.
    fn read_routing_codebooks(&self, state: &StateHeader) -> Result<PqCodebooks, FlareError> {
        let count = self.sub_vectors * CODES_PER_SUBVECTOR * (self.dimension / self.sub_vectors);
        let data = self.read_f32_block(state.routing_codebooks_off, count)?;
        PqCodebooks::from_centroids(self.dimension, self.sub_vectors, data)
    }

    /// Returns the index of the nearest centroid for `vector`.
    fn nearest_centroid(&self, state: &StateHeader, vector: &[f32]) -> Result<usize, FlareError> {
        let count = usize::try_from(state.centroids_count).expect("centroid count fits in usize");
        let centroids = self.read_f32_block(state.centroids_off, count * self.dimension)?;
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..count {
            let d = l2_sq_dispatch(
                vector,
                &centroids[c * self.dimension..(c + 1) * self.dimension],
            );
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        Ok(best)
    }

    /// Computes the residual `vector - centroid` for `centroid_idx`.
    fn residual_of(
        &self,
        state: &StateHeader,
        centroid_idx: usize,
        vector: &[f32],
    ) -> Result<Vec<f32>, FlareError> {
        let base = state.centroids_off + 4 * (centroid_idx * self.dimension) as u64;
        let centroid = self.read_f32_block(base, self.dimension)?;
        Ok(vector.iter().zip(&centroid).map(|(v, c)| v - c).collect())
    }

    /// Routes `query` to a centroid: radix tree lookup on the routing
    /// code first, exact scan as fallback.
    fn route(
        &self,
        state: &StateHeader,
        cb: &PqCodebooks,
        query: &[f32],
    ) -> Result<usize, FlareError> {
        let code = cb.encode(query)?;
        if let Some(value) = self.router.get(&code)? {
            let idx = usize::try_from(value).expect("router value fits in usize");
            let count =
                usize::try_from(state.centroids_count).expect("centroid count fits in usize");
            if idx < count {
                return Ok(idx);
            }
        }
        self.nearest_centroid(state, query)
    }

    /// Links `block` onto the head cell with a CAS loop, writing the old
    /// head into the block's `next` field before publishing it.
    fn push_cas(&self, head: &AtomicU64, block: u64) -> Result<(), FlareError> {
        loop {
            let old = head.load(Ordering::Relaxed);
            self.arena.write_node(block, &old)?;
            if head
                .compare_exchange(old, block, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Trains over `rows` (whole `dimension`-long rows) and publishes a
    /// new snapshot with a single CAS, retrying on races.
    fn rebuild(&self, rows: &[f32]) -> Result<(), FlareError> {
        let n = rows.len() / self.dimension;
        if !rows.len().is_multiple_of(self.dimension) {
            return Err(FlareError::VectorDimensionMismatch {
                expected: self.dimension,
                got: rows.len(),
            });
        }
        if n == 0 {
            return Err(FlareError::InvalidParameter {
                reason: "training set is empty",
            });
        }
        if n < self.n_centroids {
            return Err(FlareError::InvalidParameter {
                reason: "training set smaller than centroid count",
            });
        }
        let (centroids, cb, cb_raw, codes) = self.build_models(rows)?;
        loop {
            let root_word = self.root_cell().load(Ordering::Acquire);
            let (generation, old_off) = unpack_root(root_word);
            let mut old_codes = Vec::new();
            let (journal_off, journal_count) = if generation == 0 {
                let j = self.arena.alloc(8, 8)?;
                let c = self.arena.alloc(8, 8)?;
                self.arena.write_node(j, &0u64)?;
                self.arena.write_node(c, &0u64)?;
                (j, c)
            } else {
                let old = self.read_header(old_off)?;
                old_codes = self.read_u8_block(
                    old.centroid_codes_off,
                    usize::try_from(old.centroids_count).expect("centroid count fits in usize")
                        * self.sub_vectors,
                )?;
                (old.journal_off, old.journal_count)
            };
            let state_off = self.build_state_blocks(
                &centroids,
                &cb,
                &cb_raw,
                &codes,
                journal_off,
                journal_count,
            )?;
            let new_root = pack_root(generation + 1, state_off);
            if self
                .root_cell()
                .compare_exchange(root_word, new_root, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                continue;
            }
            let fresh = self.read_header(state_off)?;
            let cb = self.read_codebooks(&fresh)?;
            let journal = self.arena.atomic_word(journal_off)?;
            let mut block = journal.load(Ordering::Acquire);
            while block != 0 {
                let entry = *self.arena.read_node::<JournalHeader>(block)?;
                let vector = self.read_f32_block(block + 16, self.dimension)?;
                let idx = self.nearest_centroid(&fresh, &vector)?;
                let residual = self.residual_of(&fresh, idx, &vector)?;
                let code = cb.encode(&residual)?;
                let p_off = self.arena.alloc(16 + self.sub_vectors, 8)?;
                self.arena.write_node(
                    p_off,
                    &PostingHeader {
                        next: 0,
                        id: entry.id,
                    },
                )?;
                self.arena.write_bytes(p_off + 16, &code)?;
                let head = self
                    .arena
                    .atomic_word(fresh.posting_heads_off + 8 * idx as u64)?;
                self.push_cas(head, p_off)?;
                block = entry.next;
            }
            for code in old_codes.chunks(self.sub_vectors) {
                let _ = self.router.delete(code)?;
            }
            for (i, code) in codes.chunks(self.sub_vectors).enumerate() {
                self.router.insert(code, i as u64)?;
            }
            return Ok(());
        }
    }

    /// Trains the centroids, the residual-domain codebook, the raw-domain
    /// routing codebook, and the centroid routing codes over `rows`.
    fn build_models(
        &self,
        rows: &[f32],
    ) -> Result<(Vec<f32>, PqCodebooks, PqCodebooks, Vec<u8>), FlareError> {
        let n = rows.len() / self.dimension;
        let centroids = kmeans_l2(
            self.dimension,
            self.n_centroids,
            rows,
            TRAIN_ITERATIONS,
            self.seed,
        )?;
        let mut residuals = Vec::with_capacity(n * self.dimension);
        for r in 0..n {
            let row = &rows[r * self.dimension..(r + 1) * self.dimension];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.n_centroids {
                let d = l2_sq_dispatch(
                    row,
                    &centroids[c * self.dimension..(c + 1) * self.dimension],
                );
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            for (i, v) in row.iter().enumerate() {
                residuals.push(v - centroids[best * self.dimension + i]);
            }
        }
        let mut cb = PqCodebooks::new(self.dimension, self.sub_vectors)?;
        cb.train(&residuals, TRAIN_ITERATIONS, self.seed)?;
        let mut cb_raw = PqCodebooks::new(self.dimension, self.sub_vectors)?;
        cb_raw.train(rows, TRAIN_ITERATIONS, self.seed)?;
        let mut codes = Vec::with_capacity(self.n_centroids * self.sub_vectors);
        for c in 0..self.n_centroids {
            let code = cb_raw.encode(&centroids[c * self.dimension..(c + 1) * self.dimension])?;
            codes.extend_from_slice(&code);
        }
        Ok((centroids, cb, cb_raw, codes))
    }

    /// Reads a raw byte block.
    fn read_u8_block(&self, off: u64, count: usize) -> Result<Vec<u8>, FlareError> {
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            out.push(*self.arena.read_node::<u8>(off + i as u64)?);
        }
        Ok(out)
    }

    /// Allocates and initialises every block of a fresh snapshot.
    fn build_state_blocks(
        &self,
        centroids: &[f32],
        cb: &PqCodebooks,
        cb_raw: &PqCodebooks,
        codes: &[u8],
        journal_off: u64,
        journal_count: u64,
    ) -> Result<u64, FlareError> {
        let centroids_off = self.arena.alloc(4 * centroids.len(), 8)?;
        self.write_f32_block(centroids_off, centroids)?;
        let codebooks_off = self.arena.alloc(4 * cb.raw_centroids().len(), 8)?;
        self.write_f32_block(codebooks_off, cb.raw_centroids())?;
        let routing_codebooks_off = self.arena.alloc(4 * cb_raw.raw_centroids().len(), 8)?;
        self.write_f32_block(routing_codebooks_off, cb_raw.raw_centroids())?;
        let codes_off = self.arena.alloc(codes.len(), 8)?;
        self.arena.write_bytes(codes_off, codes)?;
        let heads_off = self.arena.alloc(8 * self.n_centroids, 8)?;
        for c in 0..self.n_centroids {
            self.arena.write_node(heads_off + 8 * c as u64, &0u64)?;
        }
        let header = StateHeader {
            centroids_off,
            centroids_count: self.n_centroids as u64,
            codebooks_off,
            posting_heads_off: heads_off,
            posting_heads_count: self.n_centroids as u64,
            journal_off,
            journal_count,
            centroid_codes_off: codes_off,
            routing_codebooks_off,
        };
        let state_off = self.arena.alloc(core::mem::size_of::<StateHeader>(), 8)?;
        self.arena.write_node(state_off, &header)?;
        Ok(state_off)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    use super::{IvfPqIndex, SearchResult};
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use flare_core::{CpuFallbackDriver, FlareError, HazardManager};

    const DIM: usize = 4;
    const SUBS: usize = 2;
    const CENTROIDS: usize = 2;
    const CAPACITY: usize = 1 << 21;

    fn index(seed: u64) -> IvfPqIndex<CpuFallbackDriver> {
        IvfPqIndex::new(
            DIM,
            CENTROIDS,
            SUBS,
            seed,
            CAPACITY,
            Arc::new(HazardManager::new()),
            CpuFallbackDriver::default(),
        )
        .expect("index construction succeeds")
    }

    /// Two well-separated clusters of 4-dimensional rows.
    fn training_samples() -> Vec<f32> {
        (0..512)
            .flat_map(|i| {
                let base = if i % 2 == 0 { 10.0 } else { -10.0 };
                [base, base, base, base]
            })
            .collect()
    }

    /// Verifies that an untrained index rejects every operation.
    #[test]
    fn untrained_index_rejects_operations() {
        let index = index(1);
        assert!(!index.is_trained());
        assert!(matches!(
            index.search(&[0.0; DIM], 1),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            index.insert(&[0.0; DIM]),
            Err(FlareError::InvalidParameter { .. })
        ));
        assert!(matches!(
            index.vector_count(),
            Err(FlareError::InvalidParameter { .. })
        ));
        index
            .trigger_shadow_reclustering()
            .expect("no-op on untrained");
        assert!(!index.is_trained());
    }

    /// Verifies train, insert, and search round-trip on separable data.
    #[test]
    fn train_insert_search_roundtrip() {
        let index = index(7);
        index.train(&training_samples()).expect("training succeeds");
        assert!(index.is_trained());
        index
            .insert(&[10.5, 10.5, 10.5, 10.5])
            .expect("insert succeeds");
        index
            .insert(&[9.6, 9.6, 9.6, 9.6])
            .expect("insert succeeds");
        index
            .insert(&[-10.5, -10.5, -10.5, -10.5])
            .expect("insert succeeds");
        assert_eq!(index.vector_count().expect("count succeeds"), 3);
        let hits = index
            .search(&[10.4, 10.4, 10.4, 10.4], 1)
            .expect("search succeeds");
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].distance < 10.0,
            "close vector must be near: {hits:?}"
        );
        let negative = index
            .search(&[-10.4, -10.4, -10.4, -10.4], 1)
            .expect("search succeeds");
        assert_eq!(negative.len(), 1);
    }

    /// Verifies search ordering and top-k truncation.
    #[test]
    fn search_orders_and_truncates() {
        let index = index(11);
        index.train(&training_samples()).expect("training succeeds");
        for i in 0..40 {
            let v = 10.0 + i as f32 * 0.25;
            index.insert(&[v, v, v, v]).expect("insert succeeds");
        }
        let hits = index
            .search(&[10.0, 10.0, 10.0, 10.0], 5)
            .expect("search succeeds");
        assert_eq!(hits.len(), 5);
        for pair in hits.windows(2) {
            assert!(
                pair[0].distance <= pair[1].distance,
                "hits must be ascending: {hits:?}"
            );
        }
        let all = index
            .search(&[10.0, 10.0, 10.0, 10.0], 1000)
            .expect("search succeeds");
        assert_eq!(all.len(), 40);
        assert!(
            index
                .search(&[10.0; DIM], 0)
                .expect("search succeeds")
                .is_empty()
        );
    }

    /// Verifies a search with no postings returns an empty result.
    #[test]
    fn search_without_inserts_is_empty() {
        let index = index(3);
        index.train(&training_samples()).expect("training succeeds");
        let hits = index.search(&[10.0; DIM], 5).expect("search succeeds");
        assert!(hits.is_empty());
    }

    /// Verifies dimension mismatch validation on insert and search.
    #[test]
    fn rejects_dimension_mismatch() {
        let index = index(5);
        index.train(&training_samples()).expect("training succeeds");
        assert!(matches!(
            index.insert(&[1.0, 2.0, 3.0]),
            Err(FlareError::VectorDimensionMismatch { expected: DIM, .. })
        ));
        assert!(matches!(
            index.search(&[1.0; 9], 1),
            Err(FlareError::VectorDimensionMismatch { expected: DIM, .. })
        ));
    }

    /// Verifies constructor validation.
    #[test]
    fn constructor_validation() {
        let hazard = Arc::new(HazardManager::new());
        assert!(
            IvfPqIndex::new(
                0,
                2,
                2,
                1,
                1024,
                hazard.clone(),
                CpuFallbackDriver::default()
            )
            .is_err()
        );
        assert!(
            IvfPqIndex::new(
                4,
                0,
                2,
                1,
                1024,
                hazard.clone(),
                CpuFallbackDriver::default()
            )
            .is_err()
        );
        assert!(
            IvfPqIndex::new(
                5,
                2,
                2,
                1,
                1024,
                hazard.clone(),
                CpuFallbackDriver::default()
            )
            .is_err()
        );
        assert!(IvfPqIndex::new(4, 2, 0, 1, 1024, hazard, CpuFallbackDriver::default()).is_err());
    }

    /// Verifies that re-clustering rebuilds the snapshot without losing
    /// vectors and that search remains correct afterwards.
    #[test]
    fn reclustering_preserves_vectors() {
        let index = index(17);
        index.train(&training_samples()).expect("training succeeds");
        for i in 0..300 {
            let v = 10.0 + (i % 3) as f32;
            index.insert(&[v, v, v, v]).expect("insert succeeds");
        }
        assert_eq!(index.vector_count().expect("count succeeds"), 300);
        index
            .trigger_shadow_reclustering()
            .expect("re-clustering succeeds");
        assert_eq!(index.vector_count().expect("count succeeds"), 300);
        let hits = index
            .search(&[10.0, 10.0, 10.0, 10.0], 3)
            .expect("search succeeds");
        assert_eq!(hits.len(), 3);
        index
            .trigger_shadow_reclustering()
            .expect("second round succeeds");
        assert_eq!(index.vector_count().expect("count succeeds"), 300);
    }

    /// Verifies re-clustering is a no-op on an empty journal.
    #[test]
    fn reclustering_noop_on_empty_journal() {
        let index = index(19);
        index.train(&training_samples()).expect("training succeeds");
        index.trigger_shadow_reclustering().expect("no-op succeeds");
        assert_eq!(index.vector_count().expect("count succeeds"), 0);
        assert_eq!(
            index
                .search(&[10.0; DIM], 1)
                .expect("search succeeds")
                .len(),
            0
        );
    }

    /// Verifies a second `train()` publishes a fresh snapshot that remains
    /// searchable, and that vectors inserted before the retrain survive.
    #[test]
    fn retrain_publishes_fresh_snapshot() {
        let index = index(23);
        index.train(&training_samples()).expect("training succeeds");
        index
            .insert(&[10.5, 10.5, 10.5, 10.5])
            .expect("insert succeeds");
        index.train(&training_samples()).expect("retrain succeeds");
        assert!(index.is_trained());
        assert_eq!(index.vector_count().expect("count succeeds"), 1);
        let hits = index
            .search(&[10.4, 10.4, 10.4, 10.4], 1)
            .expect("search succeeds");
        assert_eq!(hits.len(), 1);
    }

    /// Verifies a long posting chain is walked completely.
    #[test]
    fn posting_chain_is_walked() {
        let index = index(29);
        index.train(&training_samples()).expect("training succeeds");
        for _ in 0..64 {
            index
                .insert(&[10.0, 10.0, 10.0, 10.0])
                .expect("insert succeeds");
        }
        let hits = index.search(&[10.0; DIM], 200).expect("search succeeds");
        assert_eq!(hits.len(), 64, "all chain entries must be scored");
        let mut ids: Vec<u64> = hits.iter().map(|h: &SearchResult| h.id).collect();
        ids.sort_unstable();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(*id, i as u64, "ids must be insertion order");
        }
    }

    /// Verifies the routing fallback: when the centroid tree misses, the
    /// exact centroid scan must still produce a valid result.
    #[test]
    fn routing_falls_back_on_tree_miss() {
        let index = index(31);
        index.train(&training_samples()).expect("training succeeds");
        index
            .insert(&[10.0, 10.0, 10.0, 10.0])
            .expect("insert succeeds");
        let before = index.search(&[10.0; DIM], 1).expect("search succeeds");
        assert_eq!(before.len(), 1);
        let state = index.load_state().expect("state loads").expect("trained");
        let codes = index
            .read_u8_block(
                state.centroid_codes_off,
                state.centroids_count as usize * SUBS,
            )
            .expect("codes read");
        for code in codes.chunks(SUBS) {
            index.router.delete(code).expect("router cleanup");
        }
        let after = index
            .search(&[10.0; DIM], 1)
            .expect("fallback search succeeds");
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].id, before[0].id);
    }

    /// Verifies arena exhaustion surfaces as an error during training.
    #[test]
    fn arena_exhaustion_is_reported() {
        let index = IvfPqIndex::new(
            DIM,
            2,
            SUBS,
            1,
            4096,
            Arc::new(HazardManager::new()),
            CpuFallbackDriver::default(),
        )
        .expect("construction succeeds");
        assert!(matches!(
            index.train(&training_samples()),
            Err(FlareError::ArenaCapacityExceeded { .. })
        ));
    }

    /// Verifies a training set that cannot satisfy the cluster count or
    /// codebook size is rejected.
    #[test]
    fn undersized_training_set_is_rejected() {
        let index = index(37);
        assert!(matches!(
            index.train(&[1.0, 2.0, 3.0]),
            Err(FlareError::VectorDimensionMismatch { .. })
        ));
        let tiny = (0..10)
            .flat_map(|_| [1.0f32, 1.0, 1.0, 1.0])
            .collect::<Vec<f32>>();
        assert!(matches!(
            index.train(&tiny),
            Err(FlareError::InvalidParameter { .. })
        ));
    }

    /// Verifies concurrent inserts and a concurrent re-clustering round
    /// are mutually consistent: every insert either lands in the journal
    /// or is retried, and the final journal count is exact.
    #[test]
    fn concurrent_inserts_with_reclustering() {
        use std::sync::Barrier;
        let index = Arc::new(index(41));
        index.train(&training_samples()).expect("training succeeds");
        let barrier = Arc::new(Barrier::new(5));
        let mut threads = Vec::new();
        for t in 0..4 {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..75 {
                    let v = 10.0 + ((t * 75 + i) % 7) as f32;
                    index.insert(&[v, v, v, v]).expect("insert succeeds");
                }
            }));
        }
        barrier.wait();
        let mut reclustered = false;
        for _ in 0..128 {
            if index.trigger_shadow_reclustering().is_ok() {
                reclustered = true;
                break;
            }
            std::thread::yield_now();
        }
        assert!(
            reclustered,
            "re-clustering succeeds once the journal is large enough"
        );
        for thread in threads {
            thread.join().expect("worker finishes");
        }
        assert_eq!(index.vector_count().expect("count succeeds"), 300);
        let hits = index.search(&[10.0; DIM], 20).expect("search succeeds");
        assert_eq!(hits.len(), 20);
    }

    /// Verifies the configuration getters report the construction inputs.
    #[test]
    fn configuration_getters() {
        let index = index(5);
        assert_eq!(index.dimension(), DIM);
        assert_eq!(index.sub_vectors(), SUBS);
        assert_eq!(index.n_centroids(), CENTROIDS);
    }

    /// Verifies train rejects an empty set and a set smaller than the
    /// centroid count.
    #[test]
    fn train_rejects_empty_and_undersized_sets() {
        let index = index(5);
        assert!(matches!(
            index.train(&[]),
            Err(FlareError::InvalidParameter {
                reason: "training set is empty"
            })
        ));
        assert!(matches!(
            index.train(&[1.0; DIM]),
            Err(FlareError::InvalidParameter {
                reason: "training set smaller than centroid count"
            })
        ));
        assert!(!index.is_trained());
    }

    /// Verifies that re-clustering retries until the journal grows past
    /// the codebook training minimum.
    #[test]
    fn recluster_retries_until_journal_is_ready() {
        let index = index(9);
        index.train(&training_samples()).expect("training succeeds");
        for i in 0..255 {
            let v = 10.0 + (i % 5) as f32;
            index.insert(&[v, v, v, v]).expect("insert succeeds");
        }
        assert!(
            index.trigger_shadow_reclustering().is_err(),
            "journal below the codebook minimum fails"
        );
        index.insert(&[10.0; DIM]).expect("insert succeeds");
        index
            .trigger_shadow_reclustering()
            .expect("succeeds once the journal holds 256 rows");
        assert_eq!(index.vector_count().expect("count succeeds"), 256);
    }
}
