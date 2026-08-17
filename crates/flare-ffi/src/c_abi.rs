//! C ABI exports over opaque engine handles.
//!
//! Every exported function returns a status code: `0` means success, any
//! other value maps one-to-one onto `flare_core::error::FlareError` as
//! documented in `flare_status`. Handles are opaque pointers that must only
//! be passed back to the same module; ownership transfers are explicit:
//!
//! - `*_create` allocates a handle the caller owns;
//! - `*_destroy` releases it exactly once;
//! - `*_train`, `*_insert`, `*_recluster`, `*_evict` mutate in place;
//! - `*_search`, `*_match`, `*_vector_count`, `*_is_trained` read only.
//!
//! The C header `include/flare.h` is regenerated from this module by the
//! build script (cbindgen) and must stay in sync with it.
//!
//! # Safety
//!
//! All functions dereference caller-provided pointers. Callers must pass
//! pointers produced by this module (or valid arrays of the declared
//! length) and must not pass the same handle to two threads concurrently
//! while a mutation is in flight.
#![allow(
    clippy::not_unsafe_ptr_arg_deref,
    clippy::identity_op,
    clippy::erasing_op
)]

use alloc::boxed::Box;
use alloc::sync::Arc;
use flare_core::error::FlareError;
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_kv::RadixAttentionEngine;
use flare_vector::IvfPqIndex;

/// Opaque handle to an [`IvfPqIndex`] backed by the CPU driver.
#[repr(C)]
pub struct flare_index {
    /// Handles are never created by value; this keeps the type opaque.
    _private: [u8; 0],
}

/// Opaque handle to a [`RadixAttentionEngine`] backed by the CPU driver.
#[repr(C)]
pub struct flare_kv_engine {
    /// Handles are never created by value; this keeps the type opaque.
    _private: [u8; 0],
}

/// One hit returned by [`flare_index_search`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct flare_search_result {
    /// Insertion-sequence identifier of the stored vector.
    pub id: u64,
    /// Asymmetric L2-squared distance to the query; lower is closer.
    pub distance: f32,
}

/// One prefix match returned by [`flare_kv_match`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct flare_prefix_match {
    /// Length of the matched prefix in tokens.
    pub token_len: u32,
    /// KV-store offset published by the owning prefix.
    pub kv_offset: u64,
}

/// Status code reported by every exported function.
///
/// `0` is success; the positive codes map onto [`FlareError`] variants in
/// declaration order, and [`FLARE_STATUS_NULL_ARGUMENT`] is reserved for
/// caller-side null pointers (which never reach the engine).
#[allow(non_camel_case_types)] // C ABI naming convention by design.
pub type flare_status_t = i32;

/// Success.
pub const FLARE_STATUS_OK: i32 = 0;
/// A pointer argument was null.
pub const FLARE_STATUS_NULL_ARGUMENT: i32 = 100;
/// Maps a crate error onto the C status space.
const fn flare_status(error: FlareError) -> i32 {
    match error {
        FlareError::ArenaBoundsExceeded { .. } => 1,
        FlareError::ArenaCapacityExceeded { .. } => 2,
        FlareError::AllocationFailed => 3,
        FlareError::InvalidNodeType(_) => 4,
        FlareError::WalFrameMalformed { .. } => 5,
        FlareError::WalFrameTooLarge { .. } => 6,
        FlareError::UnknownPinnedPointer => 7,
        FlareError::TreeInvariantViolation { .. } => 8,
        FlareError::VectorDimensionMismatch { .. } => 9,
        FlareError::InvalidParameter { .. } => 10,
        FlareError::CacheCapacityExceeded => 11,
        FlareError::GpuDriverUnavailable { .. } => 12,
    }
}

/// Returns the workspace version as `major * 10_000 + minor * 100 + patch`.
///
/// # Examples
///
/// ```
/// assert_eq!(flare_ffi::c_abi::flare_version(), 100);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn flare_version() -> u32 {
    100
}

/// Creates an [`IvfPqIndex`] handle; ownership transfers to the caller.
///
/// `dimension`, `n_centroids`, and `sub_vectors` mirror the Rust
/// constructor; `seed` seeds the deterministic training RNG;
/// `arena_capacity` sizes the backing arena in bytes. On success the new
/// handle is written to `out`.
///
/// # Safety
///
/// `out` must point to a writable `flare_index*` slot.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_create(
    dimension: u32,
    n_centroids: u32,
    sub_vectors: u32,
    seed: u64,
    arena_capacity: u64,
    out: *mut *mut flare_index,
) -> flare_status_t {
    if out.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    let arena = usize::try_from(arena_capacity).unwrap_or(usize::MAX);
    let index = match IvfPqIndex::new(
        usize::try_from(dimension).unwrap_or(0),
        usize::try_from(n_centroids).unwrap_or(0),
        usize::try_from(sub_vectors).unwrap_or(0),
        seed,
        arena,
        Arc::new(HazardManager::new()),
        CpuFallbackDriver::default(),
    ) {
        Ok(index) => index,
        Err(error) => return flare_status(error),
    };
    // SAFETY: `out` is a writable slot for the caller-owned pointer.
    unsafe { out.write(Box::into_raw(Box::new(index)) as *mut flare_index) };
    FLARE_STATUS_OK
}

/// Releases an [`IvfPqIndex`] handle created by [`flare_index_create`].
///
/// # Safety
///
/// `index` must be a handle created by [`flare_index_create`] and must not
/// be used again after this call.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_destroy(index: *mut flare_index) {
    if index.is_null() {
        return;
    }
    // SAFETY: the handle was produced by `Box::into_raw` in `flare_index_create`.
    unsafe { drop(Box::from_raw(index as *mut IvfPqIndex<CpuFallbackDriver>)) };
}

/// Trains the index over `sample_count` rows of `dimension` `f32` values.
///
/// `samples` must hold exactly `sample_count * dimension` values laid out
/// row-major.
///
/// # Safety
///
/// `samples` must point to `sample_count * dimension` readable values.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_train(
    index: *mut flare_index,
    samples: *const f32,
    sample_count: usize,
) -> flare_status_t {
    if index.is_null() || samples.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let index = unsafe { &*(index as *const IvfPqIndex<CpuFallbackDriver>) };
    // SAFETY: `samples` holds `sample_count * dimension` values per the contract.
    let slice = unsafe { core::slice::from_raw_parts(samples, sample_count * index.dimension()) };
    match index.train(slice) {
        Ok(()) => FLARE_STATUS_OK,
        Err(error) => flare_status(error),
    }
}

/// Inserts one `dimension`-long vector into the index.
///
/// # Safety
///
/// `vector` must point to `dimension` readable values.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_insert(
    index: *mut flare_index,
    vector: *const f32,
) -> flare_status_t {
    if index.is_null() || vector.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let index = unsafe { &*(index as *const IvfPqIndex<CpuFallbackDriver>) };
    // SAFETY: `vector` holds `dimension` values per the contract.
    let slice = unsafe { core::slice::from_raw_parts(vector, index.dimension()) };
    match index.insert(slice) {
        Ok(()) => FLARE_STATUS_OK,
        Err(error) => flare_status(error),
    }
}

/// Searches the index and writes up to `top_k` hits into `out`.
///
/// On success `out_count` receives the number of hits written (fewer than
/// `top_k` when fewer vectors are stored).
///
/// # Safety
///
/// `query` must point to `dimension` readable values; `out` must hold at
/// least `top_k` writable results; `out_count` must be writable.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_search(
    index: *const flare_index,
    query: *const f32,
    top_k: usize,
    out: *mut flare_search_result,
    out_count: *mut usize,
) -> flare_status_t {
    if index.is_null() || query.is_null() || out.is_null() || out_count.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let index = unsafe { &*(index as *const IvfPqIndex<CpuFallbackDriver>) };
    // SAFETY: `query` holds `dimension` values per the contract.
    let query = unsafe { core::slice::from_raw_parts(query, index.dimension()) };
    let hits = match index.search(query, top_k) {
        Ok(hits) => hits,
        Err(error) => return flare_status(error),
    };
    let hit_count = hits.len();
    // SAFETY: `out` holds at least `top_k` slots and `hits.len() <= top_k`.
    unsafe {
        for (slot, hit) in core::slice::from_raw_parts_mut(out, hit_count)
            .iter_mut()
            .zip(hits)
        {
            *slot = flare_search_result {
                id: hit.id,
                distance: hit.distance,
            };
        }
        out_count.write(hit_count);
    }
    FLARE_STATUS_OK
}

/// Writes the number of stored vectors into `out_count`.
///
/// # Safety
///
/// `out_count` must be writable.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_vector_count(
    index: *const flare_index,
    out_count: *mut u64,
) -> flare_status_t {
    if index.is_null() || out_count.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let index = unsafe { &*(index as *const IvfPqIndex<CpuFallbackDriver>) };
    match index.vector_count() {
        Ok(count) => {
            // SAFETY: `out_count` is writable per the contract.
            unsafe { out_count.write(count) };
            FLARE_STATUS_OK
        }
        Err(error) => flare_status(error),
    }
}

/// Writes whether the index has been trained into `out_trained`.
///
/// # Safety
///
/// `out_trained` must be writable.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_is_trained(
    index: *const flare_index,
    out_trained: *mut bool,
) -> flare_status_t {
    if index.is_null() || out_trained.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let index = unsafe { &*(index as *const IvfPqIndex<CpuFallbackDriver>) };
    // SAFETY: `out_trained` is writable per the contract.
    unsafe { out_trained.write(index.is_trained()) };
    FLARE_STATUS_OK
}

/// Triggers one shadow re-clustering round over the journal.
#[unsafe(no_mangle)]
pub extern "C" fn flare_index_recluster(index: *mut flare_index) -> flare_status_t {
    if index.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let index = unsafe { &*(index as *const IvfPqIndex<CpuFallbackDriver>) };
    match index.trigger_shadow_reclustering() {
        Ok(()) => FLARE_STATUS_OK,
        Err(error) => flare_status(error),
    }
}

/// Creates a [`RadixAttentionEngine`] handle; ownership transfers to the
/// caller.
///
/// `capacity_bytes` sizes the physical slab storage (at least one 4 KB
/// chunk, multiple of 8); `arena_capacity` sizes the radix tree arena.
///
/// # Safety
///
/// `out` must point to a writable `flare_kv_engine*` slot.
#[unsafe(no_mangle)]
pub extern "C" fn flare_kv_create(
    capacity_bytes: u64,
    arena_capacity: u64,
    out: *mut *mut flare_kv_engine,
) -> flare_status_t {
    if out.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    let engine = match RadixAttentionEngine::new(
        usize::try_from(capacity_bytes).unwrap_or(0),
        usize::try_from(arena_capacity).unwrap_or(0),
        Arc::new(HazardManager::new()),
        CpuFallbackDriver::default(),
    ) {
        Ok(engine) => engine,
        Err(error) => return flare_status(error),
    };
    // SAFETY: `out` is a writable slot for the caller-owned pointer.
    unsafe { out.write(Box::into_raw(Box::new(engine)) as *mut flare_kv_engine) };
    FLARE_STATUS_OK
}

/// Releases a [`RadixAttentionEngine`] handle created by [`flare_kv_create`].
///
/// # Safety
///
/// `engine` must be a handle created by [`flare_kv_create`] and must not
/// be used again after this call.
#[unsafe(no_mangle)]
pub extern "C" fn flare_kv_destroy(engine: *mut flare_kv_engine) {
    if engine.is_null() {
        return;
    }
    // SAFETY: the handle was produced by `Box::into_raw` in `flare_kv_create`.
    unsafe {
        drop(Box::from_raw(
            engine as *mut RadixAttentionEngine<CpuFallbackDriver>,
        ))
    };
}

/// Inserts `token_count` tokens under `kv_offset` into the engine.
///
/// # Safety
///
/// `tokens` must point to `token_count` readable values.
#[unsafe(no_mangle)]
pub extern "C" fn flare_kv_insert(
    engine: *mut flare_kv_engine,
    tokens: *const u32,
    token_count: usize,
    kv_offset: u64,
) -> flare_status_t {
    if engine.is_null() || tokens.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let engine = unsafe { &*(engine as *const RadixAttentionEngine<CpuFallbackDriver>) };
    // SAFETY: `tokens` holds `token_count` values per the contract.
    let tokens = unsafe { core::slice::from_raw_parts(tokens, token_count) };
    match engine.insert(tokens, kv_offset) {
        Ok(()) => FLARE_STATUS_OK,
        Err(error) => flare_status(error),
    }
}

/// Matches the longest inserted prefix of `tokens` and writes it into
/// `out`. A query that matches nothing reports success with
/// `out.token_len == 0`.
///
/// # Safety
///
/// `tokens` must point to `token_count` readable values; `out` must be
/// writable.
#[unsafe(no_mangle)]
pub extern "C" fn flare_kv_match(
    engine: *const flare_kv_engine,
    tokens: *const u32,
    token_count: usize,
    out: *mut flare_prefix_match,
) -> flare_status_t {
    if engine.is_null() || tokens.is_null() || out.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let engine = unsafe { &*(engine as *const RadixAttentionEngine<CpuFallbackDriver>) };
    // SAFETY: `tokens` holds `token_count` values per the contract.
    let tokens = unsafe { core::slice::from_raw_parts(tokens, token_count) };
    let matched = match engine.match_common_prefix(tokens) {
        Ok(matched) => matched,
        Err(error) => return flare_status(error),
    };
    // SAFETY: `out` is writable per the contract.
    unsafe {
        out.write(flare_prefix_match {
            token_len: matched.map_or(0, |m| m.token_len),
            kv_offset: matched.map_or(0, |m| m.kv_offset),
        });
    }
    FLARE_STATUS_OK
}

/// Runs `steps` clock-sweep rounds and writes the number of evicted slots
/// into `out_evicted`.
///
/// # Safety
///
/// `out_evicted` must be writable.
#[unsafe(no_mangle)]
pub extern "C" fn flare_kv_evict(
    engine: *mut flare_kv_engine,
    steps: usize,
    out_evicted: *mut usize,
) -> flare_status_t {
    if engine.is_null() || out_evicted.is_null() {
        return FLARE_STATUS_NULL_ARGUMENT;
    }
    // SAFETY: handles are opaque pointers to live engines.
    let engine = unsafe { &*(engine as *const RadixAttentionEngine<CpuFallbackDriver>) };
    // SAFETY: `out_evicted` is writable per the contract.
    unsafe { out_evicted.write(engine.evict_clock_step(steps)) };
    FLARE_STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Verifies the version constant matches the workspace version.
    #[test]
    fn version_matches_workspace() {
        assert_eq!(flare_version(), 100);
    }

    /// Verifies null pointer arguments are rejected before any engine
    /// access.
    #[test]
    fn null_arguments_are_rejected() {
        assert_eq!(flare_index_destroy(core::ptr::null_mut()), ());
        assert_eq!(
            flare_index_create(4, 2, 2, 7, 1 << 20, core::ptr::null_mut()),
            FLARE_STATUS_NULL_ARGUMENT
        );
        let mut handle: *mut flare_index = core::ptr::null_mut();
        assert_eq!(
            flare_index_create(4, 2, 2, 7, 1 << 20, &mut handle),
            FLARE_STATUS_OK
        );
        assert_eq!(
            flare_index_train(handle, core::ptr::null(), 4),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(
            flare_index_insert(handle, core::ptr::null()),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(
            flare_index_search(
                handle,
                core::ptr::null(),
                1,
                core::ptr::null_mut(),
                core::ptr::null_mut()
            ),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(
            flare_index_vector_count(handle, core::ptr::null_mut()),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(
            flare_index_is_trained(handle, core::ptr::null_mut()),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(flare_index_recluster(handle), FLARE_STATUS_OK);
        assert_eq!(flare_index_destroy(handle), ());
        let mut kv: *mut flare_kv_engine = core::ptr::null_mut();
        assert_eq!(
            flare_kv_create(1 << 20, 1 << 20, core::ptr::null_mut()),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(flare_kv_create(1 << 20, 1 << 20, &mut kv), FLARE_STATUS_OK);
        assert_eq!(
            flare_kv_insert(kv, core::ptr::null(), 1, 5),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(
            flare_kv_match(kv, core::ptr::null(), 1, core::ptr::null_mut()),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(
            flare_kv_evict(kv, 1, core::ptr::null_mut()),
            FLARE_STATUS_NULL_ARGUMENT
        );
        assert_eq!(flare_kv_destroy(kv), ());
    }

    /// Verifies engine parameter errors map onto the documented status
    /// codes instead of panicking across the boundary.
    #[test]
    fn engine_errors_map_to_status_codes() {
        let mut handle: *mut flare_index = core::ptr::null_mut();
        assert_eq!(
            flare_index_create(0, 2, 2, 7, 1 << 20, &mut handle),
            10,
            "zero dimension is invalid"
        );
        assert_eq!(
            flare_index_create(3, 2, 2, 7, 1 << 20, &mut handle),
            10,
            "dimension not divisible by sub_vectors"
        );
        let mut kv: *mut flare_kv_engine = core::ptr::null_mut();
        assert_eq!(
            flare_kv_create(100, 4096, &mut kv),
            10,
            "capacity below one slab chunk"
        );
        assert_eq!(flare_kv_create(4096 + 4, 4096, &mut kv), 10);
    }

    /// Verifies a full train/insert/search round trip through the C ABI.
    #[test]
    fn index_round_trip_through_c_abi() {
        let mut handle: *mut flare_index = core::ptr::null_mut();
        assert_eq!(
            flare_index_create(4, 2, 2, 7, 1 << 20, &mut handle),
            FLARE_STATUS_OK
        );
        let mut trained = false;
        assert_eq!(
            flare_index_is_trained(handle, &mut trained),
            FLARE_STATUS_OK
        );
        assert!(!trained);
        let samples: Vec<f32> = (0..512)
            .flat_map(|i| {
                let base = if i % 2 == 0 { 10.0 } else { -10.0 };
                [base, base, base, base]
            })
            .collect();
        assert_eq!(
            flare_index_train(handle, samples.as_ptr(), 512),
            FLARE_STATUS_OK
        );
        assert_eq!(
            flare_index_is_trained(handle, &mut trained),
            FLARE_STATUS_OK
        );
        assert!(trained);
        let vectors: Vec<f32> = (0..256)
            .flat_map(|i| {
                let base = if i % 2 == 0 { 10.0 } else { -10.0 };
                [base, base, base, base]
            })
            .collect();
        for v in vectors.chunks(4) {
            assert_eq!(flare_index_insert(handle, v.as_ptr()), FLARE_STATUS_OK);
        }
        let mut count = 0;
        assert_eq!(
            flare_index_vector_count(handle, &mut count),
            FLARE_STATUS_OK
        );
        assert_eq!(count, 256);
        let query = [10.5f32, 10.5, 10.5, 10.5];
        let mut hits = [flare_search_result {
            id: 0,
            distance: 0.0,
        }; 1];
        let mut hit_count = 0;
        assert_eq!(
            flare_index_search(handle, query.as_ptr(), 1, hits.as_mut_ptr(), &mut hit_count),
            FLARE_STATUS_OK
        );
        // The search returns up to top_k=1 hit; with 4 vectors (two [10...] and
        // two [-10...]), the query [10.5...] is closest to the first [10...]
        // vector (id 0). Accept 1 or 2 hits (the second [10...] vector has the
        // same distance) but require at least 1 and the first hit to be id 0.
        assert!(hit_count >= 1, "at least one hit");
        // All even-indexed vectors are [10.0, 10.0, 10.0, 10.0] and have identical
        // distance to the query [10.5...]; the top hit can be any even id.
        assert!(
            hits[0].id % 2 == 0,
            "first hit must be a [10.0] vector (even id), got {}",
            hits[0].id
        );
        assert_eq!(
            flare_index_recluster(handle),
            FLARE_STATUS_OK,
            "re-clustering with sufficient journal succeeds"
        );
        assert_eq!(flare_index_destroy(handle), ());
    }

    /// Verifies a full insert/match/evict round trip through the C ABI.
    #[test]
    fn kv_round_trip_through_c_abi() {
        let mut handle: *mut flare_kv_engine = core::ptr::null_mut();
        assert_eq!(flare_kv_create(4096, 1 << 20, &mut handle), FLARE_STATUS_OK);
        let tokens = [1u32, 2, 3, 4];
        assert_eq!(
            flare_kv_insert(handle, tokens.as_ptr(), 1, 100),
            FLARE_STATUS_OK
        );
        let mut matched = flare_prefix_match {
            token_len: 0,
            kv_offset: 0,
        };
        assert_eq!(
            flare_kv_match(handle, tokens.as_ptr(), 4, &mut matched),
            FLARE_STATUS_OK
        );
        assert_eq!(matched.token_len, 1);
        assert_eq!(matched.kv_offset, 100);
        let mut evicted = 0;
        // Two full sweeps (matching the engine test pattern) so the single
        // live slot is decremented then evicted.
        assert_eq!(flare_kv_evict(handle, 512, &mut evicted), FLARE_STATUS_OK);
        assert_eq!(evicted, 0, "first sweep decrements but must not evict");
        assert_eq!(flare_kv_evict(handle, 512, &mut evicted), FLARE_STATUS_OK);
        assert_eq!(evicted, 1, "second sweep evicts the slot");
        assert_eq!(
            flare_kv_match(handle, tokens.as_ptr(), 4, &mut matched),
            FLARE_STATUS_OK
        );
        assert_eq!(matched.token_len, 0, "evicted prefix no longer matches");
        assert_eq!(flare_kv_destroy(handle), ());
    }
}
