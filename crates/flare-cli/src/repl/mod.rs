//! Interactive read-eval-print loop shell backed by reedline.
//!
//! The shell exposes a small command surface over the three FLARE
//! engines — radix tree, IVF-PQ vector index and radix KV-cache — plus
//! arena inspection and WAL flushing. Command execution lives in
//! [`crate::repl::commands`] so it can be unit-tested without a terminal;
//! this module only wires reedline's prompt, completer, highlighter and
//! history around it.

pub mod commands;
pub mod highlighter;

use crate::repl::commands::{ReplAction, ReplEngine};
use crate::repl::highlighter::FlareHighlighter;
use reedline::{
    DefaultCompleter, FileBackedHistory, Prompt, PromptEditMode, PromptHistorySearch, Reedline,
    Signal,
};
use std::borrow::Cow;

/// Maximum history lines kept by the shell.
const HISTORY_LIMIT: usize = 2_000;

/// The shell prompt: `flare> ` without reedline's built-in `> ` indicator.
struct FlarePrompt;

impl Prompt for FlarePrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("flare> ")
    }

    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _prompt_mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("..> ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        _history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        Cow::Borrowed("(search) ")
    }
}

/// Runs the interactive REPL shell.
///
/// Returns normally after the user exits with `exit`/`quit` or Ctrl-D.
/// Prints any engine error as a red one-liner and continues.
pub fn run() {
    let mut engine = match ReplEngine::new(1 << 24) {
        Ok(engine) => engine,
        Err(error) => {
            eprintln!("failed to initialise REPL engines: {error}");
            return;
        }
    };

    let completer = Box::new(DefaultCompleter::new_with_wordlen(
        commands::COMMANDS
            .iter()
            .map(|cmd| (*cmd).to_string())
            .collect(),
        1,
    ));
    let mut line_editor = Reedline::create()
        .with_completer(completer)
        .with_highlighter(Box::new(FlareHighlighter))
        .with_ansi_colors(true);

    if let Ok(history) = FileBackedHistory::with_file(HISTORY_LIMIT, history_path()) {
        line_editor = line_editor.with_history(Box::new(history));
    }

    let prompt = FlarePrompt;

    println!("FLARE interactive shell. Type `help` for commands, `exit` to quit.");
    loop {
        let signal = match line_editor.read_line(&prompt) {
            Ok(signal) => signal,
            Err(error) => {
                eprintln!("reedline error: {error}");
                break;
            }
        };
        match signal {
            Signal::Success(line) => match engine.execute(&line) {
                Ok(ReplAction::Exit) => break,
                Ok(ReplAction::Continue) => {
                    println!("\n{}", engine.take_output());
                }
                Err(message) => {
                    println!("{message}");
                }
            },
            Signal::CtrlC => {
                let _ = line_editor.clear_screen();
                let _ = engine.take_output();
            }
            Signal::CtrlD | Signal::ExternalBreak(..) => break,
            _ => {}
        }
    }
    println!("bye.");
}

/// Returns the per-user history file path.
fn history_path() -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push("flare-repl-history.txt");
    path
}
