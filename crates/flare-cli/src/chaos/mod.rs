//! Headless chaos testing arena: contention storms, memory pressure, and
//! WAL crash fault injection, all runnable from the command line for
//! CI/CD integration.

pub mod crash_injector;
pub mod memory_pressure;
pub mod storm;

use clap::Args;
use std::error::Error;

pub use crash_injector::{CrashReport, crash_cycles};
pub use memory_pressure::{PressureReport, memory_exhaustion};
pub use storm::{StormConfig, StormReport, contention_storm};

/// Command-line arguments of the `flare-cli chaos` subcommand.
#[derive(Debug, Clone, Args)]
pub struct ChaosArgs {
    /// Scenario to run: `contention-storm`, `memory-exhaustion`, or
    /// `crash-fault-injection`.
    #[arg(long, default_value = "contention-storm")]
    pub scenario: String,
    /// Worker threads for the contention storm.
    #[arg(long, default_value_t = 16)]
    pub threads: usize,
    /// Total insert attempts for the contention storm.
    #[arg(long, default_value_t = 1_000_000)]
    pub ops: u64,
    /// Arena byte limit as a plain number or a `KB`/`MB`/`GB` suffix
    /// (for example `4MB`).
    #[arg(long, default_value = "64MB")]
    pub arena_limit: String,
    /// Crash simulation cycles for the WAL fault-injection scenario.
    #[arg(long, default_value_t = 100)]
    pub cycles: usize,
}

/// Dispatches a chaos scenario from the command line.
///
/// # Errors
///
/// Returns an error when the arena limit cannot be parsed or a scenario
/// fails at runtime.
pub fn run(args: &ChaosArgs) -> Result<(), Box<dyn Error>> {
    match args.scenario.as_str() {
        "contention-storm" => {
            let arena_bytes = parse_arena_limit(&args.arena_limit)?;
            let report = contention_storm(
                StormConfig {
                    threads: args.threads,
                    attempts: args.ops,
                    keyspace: (1 << 16).min(args.ops),
                    arena_bytes,
                },
                None,
            )?;
            print_storm_report(&report);
        }
        "memory-exhaustion" => {
            let arena_bytes = parse_arena_limit(&args.arena_limit)?;
            let report = memory_exhaustion(arena_bytes)?;
            print_pressure_report(&report);
        }
        "crash-fault-injection" => {
            let reports = crash_cycles(args.cycles)?;
            print_crash_summary(&reports);
        }
        other => return Err(format!("unknown scenario '{other}'").into()),
    }
    Ok(())
}

/// Parses a size string like `4MB`, `64KB`, `1GB`, or a plain byte count.
fn parse_arena_limit(raw: &str) -> Result<usize, Box<dyn Error>> {
    let trimmed = raw.trim();
    let (number, factor) = [("KB", 10usize), ("MB", 20), ("GB", 30)]
        .into_iter()
        .find_map(|(suffix, shift)| {
            trimmed
                .strip_suffix(suffix)
                .map(|rest| (rest, 1usize << shift))
        })
        .unwrap_or((trimmed, 1));
    let value: usize = number
        .parse()
        .map_err(|_| format!("invalid size '{raw}'"))?;
    value
        .checked_mul(factor)
        .ok_or_else(|| format!("size '{raw}' overflows usize").into())
}

/// Renders the contention storm audit report as a terminal table.
fn print_storm_report(report: &StormReport) {
    let header = "[CHAOS ARENA] Running Scenario: Contention Storm";
    println!(
        "{header}\n[CONFIG] Threads: {} | Attempts: {} | Contention Target: Common Prefix Keys",
        report.threads, report.attempts
    );
    println!("---------------------------------------------------------------------------");
    println!("[AUDIT REPORT & CONSISTENCY VERIFICATION]");
    println!("  • Total Time Elapsed     : {:.1} ms", report.elapsed_ms);
    println!(
        "  • Average Throughput     : {:.2} Million Ops / Second",
        report.throughput_mops
    );
    println!(
        "  • CAS Collisions Seen    : {} ({:.2}% Contention Rate)",
        report.contended_cas,
        contention_rate(report)
    );
    println!(
        "  • Lost Update Errors     : {} ({} keys verified)",
        report.lost_updates, report.verified_keys
    );
    println!(
        "  • Arena Frontier         : {} bytes",
        report.arena_frontier
    );
    println!("  • Hazard Eras Reclaimed  : {} slots", report.reclaimed);
    let status = if report.lost_updates == 0 {
        "SUCCESS (PASSED ALL SAFETY INVARIANTS)"
    } else {
        "FAILED (LOST UPDATES DETECTED)"
    };
    println!("  • Final Status           : {status}");
}

/// Computes the contention rate as a percentage of all attempts.
#[allow(clippy::cast_precision_loss)]
pub fn contention_rate(report: &StormReport) -> f64 {
    if report.attempts == 0 {
        return 0.0;
    }
    let contended = f64::from(u32::try_from(report.contended_cas).unwrap_or(u32::MAX));
    let attempts = f64::from(u32::try_from(report.attempts).unwrap_or(u32::MAX));
    contended * 100.0 / attempts
}

/// Renders the memory pressure report.
fn print_pressure_report(report: &PressureReport) {
    println!("[CHAOS ARENA] Running Scenario: Memory Exhaustion");
    println!("---------------------------------------------------------------------------");
    println!("  • Keys Inserted          : {}", report.inserted_keys);
    println!(
        "  • Arena Frontier         : {} / {} bytes",
        report.frontier, report.capacity
    );
    println!("  • Exhaustion Error       : {}", report.exhaustion);
    println!("  • Sample Keys Verified   : {}", report.sample_verified);
    println!("  • Slab Slots Allocated   : {}", report.slab_allocated);
    println!("  • Slab Slots Recycled    : {}", report.slab_recycled);
    println!("  • Final Status           : SUCCESS (EXHAUSTION HANDLED CLEANLY)");
}

/// Renders the crash fault-injection summary across all cycles.
fn print_crash_summary(reports: &[CrashReport]) {
    println!(
        "[CHAOS ARENA] Running Scenario: Crash Fault Injection ({} cycles)",
        reports.len()
    );
    println!("---------------------------------------------------------------------------");
    let last = reports.last().expect("at least one cycle");
    println!(
        "  • Frames Per Cycle       : {} written",
        last.frames_written
    );
    println!(
        "  • Frames Replayed        : {} (variable truncation point)",
        last.frames_replayed
    );
    println!("  • Recovery High-Water    : {} bytes", last.high_water);
    println!(
        "  • Last Recovery Time     : {:.2} us (direct memcpy overlay)",
        last.recovery_us
    );
    println!("  • Log Sink Bytes         : {}", last.sink_bytes);
    let consistent = reports.iter().all(|report| report.consistent);
    let verified = reports.iter().filter(|report| report.consistent).count();
    println!(
        "  • Consistency            : {verified} / {} cycles verified",
        reports.len()
    );
    let status = if consistent {
        "SUCCESS (WAL REPLAY CONSISTENT ACROSS ALL CYCLES)"
    } else {
        "FAILED (REPLAY INCONSISTENCY DETECTED)"
    };
    println!("  • Final Status           : {status}");
}

#[cfg(test)]
mod tests {
    use super::{contention_rate, parse_arena_limit, print_crash_summary, print_pressure_report};
    use crate::chaos::crash_injector::CrashReport;
    use crate::chaos::memory_pressure::PressureReport;
    use crate::chaos::storm::StormReport;

    /// Verifies the size-suffix parser across all supported units.
    #[test]
    fn arena_limit_parses_suffixes() {
        assert_eq!(parse_arena_limit("4KB").expect("parses"), 4 << 10);
        assert_eq!(parse_arena_limit("64MB").expect("parses"), 64 << 20);
        assert_eq!(parse_arena_limit("1GB").expect("parses"), 1 << 30);
        assert_eq!(parse_arena_limit("1024").expect("parses"), 1024);
        assert!(parse_arena_limit("nope").is_err());
        assert!(parse_arena_limit("1XB").is_err());
    }

    /// Verifies that the contention rate stays bounded.
    #[test]
    fn contention_rate_is_bounded() {
        let report = StormReport {
            threads: 4,
            attempts: 1_000,
            elapsed_ms: 1.0,
            throughput_mops: 1.0,
            contended_cas: 500,
            lost_updates: 0,
            verified_keys: 128,
            arena_frontier: 4096,
            reclaimed: 0,
        };
        assert!((contention_rate(&report) - 50.0).abs() < 1e-6);
    }

    /// Verifies that the report printers accept every report shape.
    #[test]
    fn report_printers_do_not_panic() {
        let pressure = PressureReport {
            inserted_keys: 10,
            frontier: 512,
            capacity: 1024,
            exhaustion: "ArenaCapacityExceeded",
            sample_verified: 3,
            slab_allocated: 8,
            slab_recycled: 4,
        };
        print_pressure_report(&pressure);
        let crash = CrashReport {
            frames_written: 4,
            frames_replayed: 2,
            high_water: 48,
            recovery_us: 5.0,
            survived: 2,
            consistent: true,
            sink_bytes: 96,
        };
        print_crash_summary(&[crash]);
    }
}
