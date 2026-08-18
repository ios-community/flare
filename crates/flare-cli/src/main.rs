//! FLARE interactive workbench: real-time TUI dashboard, engineering REPL
//! shell, and chaos stress-testing arena for the FLARE engine crates.
//!
//! # Commands
//!
//! - `flare-cli tui [--arena 64MB]` — real-time dashboard with five tabs.
//! - `flare-cli repl [--arena 64MB]` — interactive shell over the engines.
//! - `flare-cli chaos --scenario contention-storm` — headless stress test.
#![deny(missing_docs)]

mod chaos;
mod repl;
mod telemetry;
mod tui;

use crate::chaos::ChaosArgs;
use clap::{Parser, Subcommand};

/// Interactive workbench for the FLARE engine stack.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Mode to launch.
    #[command(subcommand)]
    command: Command,
}

/// The three workbench modes.
#[derive(Debug, Subcommand)]
enum Command {
    /// Real-time TUI dashboard (five tabs).
    Tui(TuiArgs),
    /// Interactive REPL shell.
    Repl(ReplArgs),
    /// Headless chaos stress-testing arena.
    Chaos(ChaosArgs),
}

/// Arguments of the `tui` subcommand.
#[derive(Debug, clap::Args)]
struct TuiArgs {
    /// Arena byte limit shared by all engines (`4MB`, `64MB`, `1GB`).
    #[arg(long, default_value = "64MB")]
    arena: String,
}

/// Arguments of the `repl` subcommand.
#[derive(Debug, clap::Args)]
struct ReplArgs {
    /// Arena byte limit shared by all engines (`4MB`, `64MB`, `1GB`).
    #[arg(long, default_value = "64MB")]
    arena: String,
}

fn main() {
    let cli = Cli::parse();
    let exit_code = match cli.command {
        Command::Tui(args) => match parse_arena(&args.arena) {
            Ok(arena) => match tui::run(arena) {
                Ok(()) => 0,
                Err(error) => {
                    eprintln!("tui error: {error}");
                    1
                }
            },
            Err(message) => {
                eprintln!("{message}");
                2
            }
        },
        Command::Repl(args) => match parse_arena(&args.arena) {
            Ok(arena) => {
                let _ = arena;
                repl::run();
                0
            }
            Err(message) => {
                eprintln!("{message}");
                2
            }
        },
        Command::Chaos(args) => match chaos::run(&args) {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("chaos error: {error}");
                1
            }
        },
    };
    std::process::exit(exit_code);
}

/// Parses a size string like `64MB` into bytes.
fn parse_arena(raw: &str) -> Result<usize, String> {
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
        .map_err(|_| format!("invalid arena size '{raw}'"))?;
    value
        .checked_mul(factor)
        .ok_or_else(|| format!("arena size '{raw}' overflows usize"))
}

#[cfg(test)]
mod tests {
    use super::parse_arena;

    /// Verifies the arena size parser across all suffixes.
    #[test]
    fn arena_parser_handles_suffixes() {
        assert_eq!(parse_arena("64MB").expect("parses"), 64 << 20);
        assert_eq!(parse_arena("2KB").expect("parses"), 2 << 10);
        assert_eq!(parse_arena("1GB").expect("parses"), 1 << 30);
        assert_eq!(parse_arena("12345").expect("parses"), 12_345);
        assert!(parse_arena("bogus").is_err());
    }
}
