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
mod config;
mod repl;
mod telemetry;
mod tui;

use crate::chaos::ChaosArgs;
use crate::config::{apply_config_to_tui, load_config};
use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::Shell;
use clap_mangen::Man;

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
    Tui(CliTuiArgs),
    /// Interactive REPL shell.
    Repl(ReplArgs),
    /// Headless chaos stress-testing arena.
    Chaos(ChaosArgs),
    /// Generate shell completions.
    Completion(CompletionArgs),
    /// Generate man pages.
    GenerateMan(ManArgs),
}

/// Arguments for the `completion` subcommand.
#[derive(Debug, clap::Args)]
struct CompletionArgs {
    /// Target shell.
    #[arg(value_enum, default_value = "bash")]
    shell: Shell,
}

/// Arguments for the `generate-man` subcommand.
#[derive(Debug, clap::Args)]
struct ManArgs {
    /// Output directory for generated man pages.
    #[arg(long, default_value = "man")]
    out_dir: String,
}

/// Arguments of the `tui` subcommand.
#[derive(Debug, clap::Args)]
struct CliTuiArgs {
    /// Arena byte limit shared by all engines (`4MB`, `64MB`, `1GB`).
    #[arg(long, default_value = "64MB")]
    arena: String,
    /// Default tab index (0=Dashboard, 1=Memory, 2=Vector, 3=KV-Cache, 4=Chaos).
    #[arg(long, default_value = "0")]
    tab: usize,
    /// Refresh interval in milliseconds.
    #[arg(long, default_value = "30")]
    refresh: u64,
    /// Theme: dark, light, high-contrast, protanopia, deuteranopia, tritanopia.
    #[arg(long, default_value = "dark")]
    theme: String,
    /// Start with workload running (default: paused).
    #[arg(long)]
    run: bool,
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
    let config = load_config();
    let exit_code = match cli.command {
        Command::Tui(args) => {
            let mut tui_args = apply_config_to_tui(&config);
            // CLI overrides config
            let arena = crate::config::parse_arena(&args.arena).unwrap_or(tui_args.arena_capacity);
            tui_args.arena_capacity = arena;
            tui_args.default_tab = args.tab.min(4);
            tui_args.refresh_ms = args.refresh;
            tui_args.theme = args.theme;
            tui_args.start_paused = !args.run;
            match tui::run(&tui_args) {
                Ok(()) => 0,
                Err(error) => {
                    eprintln!("tui error: {error}");
                    1
                }
            }
        }
        Command::Repl(args) => match crate::config::parse_arena(&args.arena) {
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
        Command::Completion(args) => {
            let mut cmd = Cli::command();
            clap_complete::generate(args.shell, &mut cmd, "flare-cli", &mut std::io::stdout());
            0
        }
        Command::GenerateMan(args) => {
            let cmd = Cli::command();
            let man = Man::new(cmd);
            let out_dir = std::path::Path::new(&args.out_dir);
            std::fs::create_dir_all(out_dir).expect("create man dir");
            man.render(&mut std::fs::File::create(out_dir.join("flare-cli.1")).expect("create man file"))
                .expect("render man page");
            0
        }
    };
    std::process::exit(exit_code);
}

#[allow(dead_code)]
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
