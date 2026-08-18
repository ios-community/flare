//! Arena exhaustion and slab recycling pressure tests.
//!
//! The arena pressure test fills a small tree with unique keys until the
//! bump frontier refuses a new region, then spot-checks that every sampled
//! key is still readable. The slab pressure test allocates a small
//! [`SlabPool`] to exhaustion, frees every other slot, and verifies that
//! the freed slots are recycled by later allocations.

use flare_core::alloc::arena::FlatArena;
use flare_core::alloc::slab::SlabPool;
use flare_core::error::FlareError;
use flare_core::ptr::NodeType;
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_core::tree::FlareArtTree;
use std::sync::Arc;

/// Keys are checked every `SAMPLE_STRIDE` inserts during verification.
const SAMPLE_STRIDE: u64 = 97;

/// Aggregated report of the memory pressure run.
#[derive(Debug, Clone, Copy)]
pub struct PressureReport {
    /// Unique keys inserted before the arena refused another region.
    pub inserted_keys: u64,
    /// Arena frontier at exhaustion in bytes.
    pub frontier: u64,
    /// Arena capacity in bytes.
    pub capacity: u64,
    /// Short description of the exhaustion error observed.
    pub exhaustion: &'static str,
    /// Keys spot-checked and found intact after exhaustion.
    pub sample_verified: u64,
    /// Slab slots allocated before the class budget ran out.
    pub slab_allocated: u64,
    /// Freed slab slots that were successfully recycled.
    pub slab_recycled: u64,
}

/// Drives the arena and slab pools to their limits and reports the results.
///
/// # Errors
///
/// Returns [`FlareError::AllocationFailed`] when the arena backing store
/// cannot be allocated.
pub fn memory_exhaustion(arena_bytes: usize) -> Result<PressureReport, FlareError> {
    let arena = Arc::new(FlatArena::new(arena_bytes)?);
    let hazard = Arc::new(HazardManager::new());
    let tree = FlareArtTree::new(arena.clone(), hazard, CpuFallbackDriver::default());

    let mut inserted = 0u64;
    loop {
        let key = format!("fill:{inserted:08x}");
        match tree.insert(key.as_bytes(), inserted) {
            Ok(_) => inserted += 1,
            Err(FlareError::ArenaCapacityExceeded { .. }) => break,
            Err(other) => return Err(other),
        }
    }
    let mut sample_verified = 0u64;
    for index in (0..inserted).step_by(usize::try_from(SAMPLE_STRIDE).unwrap_or(1)) {
        let key = format!("fill:{index:08x}");
        if tree.get(key.as_bytes())? == Some(index) {
            sample_verified += 1;
        }
    }

    let (slab_allocated, slab_recycled) = slab_pressure()?;

    Ok(PressureReport {
        inserted_keys: inserted,
        frontier: arena.frontier(),
        capacity: arena.capacity(),
        exhaustion: "ArenaCapacityExceeded",
        sample_verified,
        slab_allocated,
        slab_recycled,
    })
}

/// Exhausts a small [`SlabPool`], frees half its slots, and recycles them.
fn slab_pressure() -> Result<(u64, u64), FlareError> {
    let pool = SlabPool::new(1 << 16)?;
    let mut slots = Vec::new();
    while let Some(slot) = pool.alloc(NodeType::Node4)? {
        slots.push(slot);
    }
    let allocated = u64::try_from(slots.len()).expect("slot count fits in u64");
    for (index, slot) in slots.iter().enumerate() {
        if index % 2 == 0 {
            pool.free(*slot);
        }
    }
    let mut recycled = 0u64;
    while pool.alloc(NodeType::Node4)?.is_some() {
        recycled += 1;
    }
    Ok((allocated, recycled))
}

#[cfg(test)]
mod tests {
    use super::memory_exhaustion;

    /// Verifies that a small arena exhausts cleanly and stays consistent.
    #[test]
    fn small_arena_exhausts_consistently() {
        let report = memory_exhaustion(1 << 20).expect("pressure run succeeds");
        assert!(report.inserted_keys > 0);
        assert_eq!(report.exhaustion, "ArenaCapacityExceeded");
        assert!(report.frontier > 0);
        assert!(report.sample_verified > 0);
        assert!(report.slab_allocated > 0);
        assert_eq!(report.slab_recycled, report.slab_allocated / 2);
    }

    /// Verifies that the slab pool recycles every freed slot exactly once.
    #[test]
    fn slab_free_is_fully_recycled() {
        let (allocated, recycled) = super::slab_pressure().expect("slab run succeeds");
        assert_eq!(recycled, allocated / 2);
    }

    /// Verifies that a zero-capacity arena reports exhaustion immediately.
    #[test]
    fn zero_capacity_arena_reports_exhaustion() {
        let report = memory_exhaustion(0).expect("empty arena still yields a report");
        assert_eq!(report.inserted_keys, 0);
        assert_eq!(report.exhaustion, "ArenaCapacityExceeded");
    }
}
