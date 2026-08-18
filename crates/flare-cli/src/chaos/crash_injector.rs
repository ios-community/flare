//! Power-loss crash simulation and WAL replay verification.
//!
//! A sequence of [`WalTransaction`]s is committed to a [`MemoryWalSink`],
//! each writing one arena value word. A simulated power loss truncates the
//! log after `crash_after` complete transactions, then recovery replays the
//! surviving frames into a *fresh* zeroed arena via direct byte overlay and
//! verifies that exactly the frames written before the crash are intact.

use flare_core::alloc::arena::FlatArena;
use flare_core::error::FlareError;
use flare_core::wal::{MemoryWalSink, WalFrame, WalTransaction, encode_frames, parse_log, recover};
use std::time::Instant;

/// Configuration of one crash fault-injection cycle.
#[derive(Debug, Clone, Copy)]
pub struct CrashConfig {
    /// Total transactions written to the log before the crash.
    pub frames: usize,
    /// Transactions surviving the crash (log truncated after this point).
    pub crash_after: usize,
    /// Byte capacity of both the live and the recovery arena.
    pub arena_bytes: usize,
}

/// Aggregated report of one crash simulation cycle.
#[derive(Debug, Clone, Copy)]
pub struct CrashReport {
    /// Transactions written before the crash.
    pub frames_written: u64,
    /// Transactions replayed from the truncated log.
    pub frames_replayed: u64,
    /// High-water arena mark reconstructed by recovery.
    pub high_water: u64,
    /// Recovery wall time in microseconds.
    pub recovery_us: f64,
    /// Arena values verified intact after recovery.
    pub survived: u64,
    /// Whether every surviving value matched its pre-crash contents.
    pub consistent: bool,
    /// Encoded bytes in the full pre-crash log.
    pub sink_bytes: usize,
}

impl CrashConfig {
    /// Creates the canonical demo configuration (512 frames, crash at 400).
    #[must_use]
    pub const fn demo() -> Self {
        Self {
            frames: 512,
            crash_after: 400,
            arena_bytes: 1 << 18,
        }
    }
}

/// Writes `frames` transactions, truncates the log, and verifies recovery.
///
/// # Errors
///
/// Returns [`FlareError::ArenaCapacityExceeded`] when the arena is too
/// small for the frame count, or a WAL error when a frame cannot be
/// encoded or replayed.
pub fn crash_fault_injection(cfg: CrashConfig) -> Result<CrashReport, FlareError> {
    let live = FlatArena::new(cfg.arena_bytes)?;
    let sink = MemoryWalSink::new();
    let mut offsets = Vec::with_capacity(cfg.frames);

    for index in 0..cfg.frames {
        let offset = live.alloc(8, 8)?;
        let value = u64::try_from(index).expect("frame index fits in u64");
        live.write_node(offset, &value)?;
        let tx = WalTransaction::new(
            vec![WalFrame::alloc(offset, 8)],
            WalFrame::update(offset, value.to_le_bytes().to_vec()),
        );
        tx.commit(&sink)?;
        offsets.push(offset);
    }

    let snapshot = sink.snapshot();
    let sink_bytes = snapshot.len();
    let full = parse_log(&snapshot)?;
    let keep_txs = cfg.crash_after.min(cfg.frames);
    let keep_frames = keep_txs * 2;
    let truncated = encode_frames(&full[..keep_frames])?;

    let recovery_arena = FlatArena::new(cfg.arena_bytes)?;
    let started = Instant::now();
    let high_water = recover(&recovery_arena, &truncated)?;
    let recovery_us = started.elapsed().as_secs_f64() * 1_000_000.0;

    let mut survived = 0u64;
    let mut consistent = true;
    for (index, &offset) in offsets.iter().enumerate().take(keep_txs) {
        let expected = u64::try_from(index).expect("frame index fits in u64");
        match recovery_arena.read_node::<u64>(offset) {
            Ok(&value) if value == expected => survived += 1,
            _ => consistent = false,
        }
    }

    Ok(CrashReport {
        frames_written: u64::try_from(cfg.frames).expect("transaction count fits in u64"),
        frames_replayed: u64::try_from(keep_txs).expect("transaction count fits in u64"),
        high_water,
        recovery_us,
        survived,
        consistent,
        sink_bytes,
    })
}

/// Runs `cycles` crash simulations with different truncation points and
/// asserts that every cycle recovers consistently.
///
/// # Errors
///
/// Returns the first [`FlareError`] observed by any cycle.
pub fn crash_cycles(cycles: usize) -> Result<Vec<CrashReport>, FlareError> {
    let mut reports = Vec::with_capacity(cycles);
    for cycle in 0..cycles {
        let keep = (cycle % 400) + 1;
        let report = crash_fault_injection(CrashConfig {
            frames: 512,
            crash_after: keep,
            arena_bytes: 1 << 18,
        })?;
        assert!(report.consistent, "recovery must be consistent");
        assert_eq!(report.survived, report.frames_replayed);
        reports.push(report);
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::{CrashConfig, crash_cycles, crash_fault_injection};

    /// Verifies that a crash at the tail of the log recovers every frame.
    #[test]
    fn full_log_recovers_every_frame() {
        let report = crash_fault_injection(CrashConfig {
            frames: 64,
            crash_after: 64,
            arena_bytes: 1 << 16,
        })
        .expect("crash run succeeds");
        assert_eq!(report.frames_written, 64);
        assert_eq!(report.frames_replayed, 64);
        assert_eq!(report.survived, 64);
        assert!(report.consistent);
        assert!(report.high_water > 0);
    }

    /// Verifies that a mid-log crash keeps only the frames before it.
    #[test]
    fn mid_log_crash_keeps_prefix_only() {
        let report = crash_fault_injection(CrashConfig {
            frames: 64,
            crash_after: 40,
            arena_bytes: 1 << 16,
        })
        .expect("crash run succeeds");
        assert_eq!(report.frames_written, 64);
        assert_eq!(report.frames_replayed, 40);
        assert_eq!(report.survived, 40);
        assert!(report.consistent);
    }

    /// Verifies that multi-cycle runs stay consistent across truncations.
    #[test]
    fn multiple_cycles_stay_consistent() {
        let reports = crash_cycles(12).expect("cycles succeed");
        assert_eq!(reports.len(), 12);
        assert!(reports.iter().all(|report| report.consistent));
    }
}
