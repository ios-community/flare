//! Multi-threaded CAS contention storm.
//!
//! The storm hammers a shared [`FlareArtTree`] from `threads` workers. All
//! workers race on the *same* deterministic key space with a shared
//! common prefix (`storm:`), maximising contention on shared radix path
//! nodes. Every insert is preceded by a lock-free read; when the insert's
//! returned previous value differs from the pre-read, a concurrent writer
//! won the race and the probe is counted as a CAS collision.
//!
//! Because every worker writes the same deterministic value per key, the
//! final tree state is fully known in advance: after the storm, every key
//! must be present with its expected value. Any mismatch is a lost update.

use crate::telemetry::Collector;
use crate::telemetry::events::EventWord;
use flare_core::alloc::arena::FlatArena;
use flare_core::error::FlareError;
use flare_core::sync::gpu::CpuFallbackDriver;
use flare_core::sync::hazard::HazardManager;
use flare_core::tree::FlareArtTree;
use std::sync::Arc;
use std::time::Instant;

/// Key prefix shared by every storm key (5 bytes of common path).
const KEY_PREFIX: &[u8; 6] = b"storm:";

/// Bit mask keeping values inside the 56-bit inline leaf payload domain.
const VALUE_MASK: u64 = 0x00FF_FFFF_FFFF_FFFF;

/// Configuration of a contention storm run.
#[derive(Debug, Clone, Copy)]
pub struct StormConfig {
    /// Number of concurrent worker threads.
    pub threads: usize,
    /// Total insert attempts spread across all workers.
    pub attempts: u64,
    /// Number of distinct keys in the shared contention key space.
    pub keyspace: u64,
    /// Byte capacity of the backing arena.
    pub arena_bytes: usize,
}

impl Default for StormConfig {
    fn default() -> Self {
        Self {
            threads: 16,
            attempts: 1_000_000,
            keyspace: 1 << 16,
            arena_bytes: 1 << 26,
        }
    }
}

/// Aggregated audit report of one storm run.
#[derive(Debug, Clone, Copy)]
pub struct StormReport {
    /// Worker threads that participated.
    pub threads: usize,
    /// Total insert attempts performed.
    pub attempts: u64,
    /// Wall time of the storm in milliseconds.
    pub elapsed_ms: f64,
    /// Aggregate throughput in millions of operations per second.
    pub throughput_mops: f64,
    /// Insert callsites that observed a concurrent overwrite.
    pub contended_cas: u64,
    /// Keys whose final value differs from the expected value.
    pub lost_updates: u64,
    /// Keys verified after the storm.
    pub verified_keys: u64,
    /// Bytes consumed from the arena after the storm.
    pub arena_frontier: u64,
    /// Slab slots reclaimed through the hazard manager.
    pub reclaimed: usize,
}

/// Builds the deterministic storm key for `index` as 12 hex digits.
fn key_for(index: u64) -> [u8; 18] {
    let mut key = [0u8; 18];
    key[..6].copy_from_slice(KEY_PREFIX);
    for pos in 0..12 {
        let shift = 44 - pos * 4;
        let nibble = ((index >> shift) & 0xF) as u8;
        key[6 + pos] = if nibble < 10 {
            b'0' + nibble
        } else {
            b'a' + nibble - 10
        };
    }
    key
}

/// FNV-1a 64-bit hash of the key bytes, masked into the inline domain.
fn value_for(key: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in key {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash & VALUE_MASK
}

/// Runs a multi-threaded contention storm and audits the final state.
///
/// Every worker writes the deterministic key range
/// `0..attempts / threads`, so only that range is audited afterwards.
/// Telemetry events are published into `collector` when provided.
///
/// # Errors
///
/// Returns [`FlareError::AllocationFailed`] when the arena backing store
/// cannot be allocated, or a tree-level error when a worker cannot finish
/// its insert loop.
pub fn contention_storm(
    cfg: StormConfig,
    collector: Option<&Arc<Collector>>,
) -> Result<StormReport, FlareError> {
    let arena = Arc::new(FlatArena::new(cfg.arena_bytes)?);
    let hazard = Arc::new(HazardManager::new());
    let tree = Arc::new(FlareArtTree::new(
        arena.clone(),
        hazard.clone(),
        CpuFallbackDriver::default(),
    ));

    let effective_threads = cfg.threads.max(1);
    let attempts_per_thread = cfg.attempts / u64::try_from(effective_threads).unwrap_or(1);
    let actual_attempts = attempts_per_thread * u64::try_from(effective_threads).unwrap_or(1);
    let started = Instant::now();

    let totals = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(cfg.threads);
        for _ in 0..cfg.threads {
            let tree = Arc::clone(&tree);
            handles.push(scope.spawn(move || {
                let mut contended = 0u64;
                for attempt in 0..attempts_per_thread {
                    let key = key_for(attempt % cfg.keyspace);
                    let value = value_for(&key);
                    let seen = tree.get(&key)?;
                    let old = tree.insert(&key, value)?;
                    if old != seen {
                        contended += 1;
                    }
                    if attempt % 256 == 0
                        && let Some(ring) = collector
                    {
                        ring.try_push(EventWord::cas_contention(
                            u32::from(contended > 0),
                            u32::try_from(attempt % 1_000_000).unwrap_or(u32::MAX),
                        ));
                    }
                }
                Ok::<u64, FlareError>(contended)
            }));
        }
        let mut contended = 0u64;
        for handle in handles {
            contended += handle.join().expect("storm worker panicked")?;
        }
        Ok::<u64, FlareError>(contended)
    })?;

    let elapsed = started.elapsed();
    let mut lost_updates = 0u64;
    let verified = if cfg.threads == 0 {
        0
    } else {
        cfg.keyspace.min(attempts_per_thread)
    };
    for index in 0..verified {
        let key = key_for(index);
        if tree.get(&key)? != Some(value_for(&key)) {
            lost_updates += 1;
        }
    }
    let reclaimed = hazard.try_reclaim();

    let elapsed_secs = elapsed.as_secs_f64();
    let throughput_mops = if elapsed_secs > 0.0 {
        (f64::from(u32::try_from(actual_attempts / 1000).unwrap_or(u32::MAX)) * 1000.0)
            / elapsed_secs
            / 1_000_000.0
    } else {
        0.0
    };

    Ok(StormReport {
        threads: cfg.threads,
        attempts: actual_attempts,
        elapsed_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_mops,
        contended_cas: totals,
        lost_updates,
        verified_keys: verified,
        arena_frontier: arena.frontier(),
        reclaimed,
    })
}

#[cfg(test)]
mod tests {
    use super::{StormConfig, contention_storm, key_for, value_for};
    use crate::telemetry::Collector;
    use std::sync::Arc;

    /// Verifies that a small storm loses no updates and reports plausible
    /// statistics.
    #[test]
    fn small_storm_loses_no_updates() {
        let collector = Arc::new(Collector::default());
        let report = contention_storm(
            StormConfig {
                threads: 4,
                attempts: 40_000,
                keyspace: 1 << 10,
                arena_bytes: 1 << 23,
            },
            Some(&collector),
        )
        .expect("storm succeeds");
        assert_eq!(report.lost_updates, 0, "storm must not lose updates");
        assert_eq!(report.verified_keys, 1 << 10);
        assert!(report.attempts >= report.verified_keys);
        assert!(report.arena_frontier > 0);
    }

    /// Verifies that the audit only covers the range the workers actually
    /// wrote: when the keyspace exceeds the per-thread attempt budget, the
    /// untouched tail must not be reported as lost updates.
    #[test]
    fn audit_covers_only_written_range() {
        let report = contention_storm(
            StormConfig {
                threads: 4,
                attempts: 40_000,
                keyspace: 1 << 16,
                arena_bytes: 1 << 26,
            },
            None,
        )
        .expect("storm succeeds");
        assert_eq!(
            report.verified_keys, 10_000,
            "only the 0..attempts/threads prefix was written"
        );
        assert_eq!(report.lost_updates, 0, "audited range must be intact");
        assert_eq!(report.attempts, 40_000);
    }

    /// Verifies the deterministic key/value mapping stays stable.
    #[test]
    fn key_value_mapping_is_deterministic() {
        for index in [0, 1, 42, 65_535] {
            let key = key_for(index);
            assert_eq!(key[..6], *b"storm:");
            assert_eq!(value_for(&key), value_for(&key));
            assert!(value_for(&key) < (1u64 << 56));
        }
    }
}
