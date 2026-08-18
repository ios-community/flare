//! Real-time TUI dashboard.
//!
//! Five tabs (Dashboard, Memory, Vector, KV-Cache, Chaos) render live
//! engine state while a background workload thread — paused until the
//! user presses `SPACE` — drives inserts, lookups, vector appends,
//! KV-cache prefix inserts and WAL flushes.

pub mod app;
pub mod tab_chaos;
pub mod tab_dashboard;
pub mod tab_kvcache;
pub mod tab_memory;
pub mod tab_vector;
pub mod ui;

use crate::config::TuiArgs;
use crate::tui::app::TuiApp;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use std::error::Error;
use std::io::IsTerminal;
use std::time::Duration;

/// Runs the TUI until the user quits, restoring the terminal afterwards.
///
/// # Errors
///
/// Returns an error when stdout is not a terminal, terminal setup fails,
/// or an event poll fails.
pub fn run(args: &TuiArgs) -> Result<(), Box<dyn Error>> {
    if !std::io::stdout().is_terminal() {
        return Err("TUI requires an interactive terminal; use `repl` or `chaos` instead".into());
    }
    let mut terminal = ratatui::try_init().map_err(|error| error.to_string())?;
    let result = run_loop(&mut terminal, args);
    let _ = ratatui::try_restore();
    result
}

/// Drives the event loop: draw, poll keys, pump engine state.
fn run_loop(
    terminal: &mut ratatui::DefaultTerminal,
    args: &TuiArgs,
) -> Result<(), Box<dyn Error>> {
    let config = crate::config::load_config();
    let mut state = crate::config::load_tui_state();
    let mut app = TuiApp::new(
        args.arena_capacity,
        state.last_tab.min(4),
        state.was_paused,
        args.refresh_ms,
        config,
    )?;
    while !app.quit {
        terminal.draw(|frame| app.draw(frame))?;
        if event::poll(Duration::from_millis(args.refresh_ms))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            handle_key(&mut app, key.code);
        }
        app.pump_log();
        app.pump_chaos();
        app.update_rates();
        // Update persisted state on each iteration
        state.last_tab = app.tab;
        state.was_paused = app.paused;
    }
    crate::config::save_tui_state(&state);
    app.shutdown();
    Ok(())
}

/// Dispatches one pressed key.
fn handle_key(app: &mut TuiApp, code: KeyCode) {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.request_quit(),
        KeyCode::Char('1') => app.tab = 0,
        KeyCode::Char('2') => app.tab = 1,
        KeyCode::Char('3') => app.tab = 2,
        KeyCode::Char('4') => app.tab = 3,
        KeyCode::Char('5') => app.tab = 4,
        KeyCode::Char(' ') => app.toggle_pause(),
        KeyCode::Char('i') => {
            let inserted = app.burst_insert(1_000);
            app.log
                .push_back(format!("burst inserted {inserted} keys"));
        }
        KeyCode::Char('s') => {
            app.run_bench();
            app.log.push_back(format!(
                "bench: scalar {:.2} ns, SIMD {:.2} ns",
                app.bench_scalar_ns, app.bench_avx2_ns
            ));
        }
        KeyCode::Char('r') => match app.trigger_recluster() {
            Ok(()) => app
                .log
                .push_back(format!("recluster gen {}", app.recluster_gen)),
            Err(e) => app.log.push_back(format!("recluster failed: {e}")),
        },
        KeyCode::Char('n') => match app.new_chat_prefix() {
            Ok(len) => app
                .log
                .push_back(format!("new chat prefix ({len} tokens)")),
            Err(e) => app.log.push_back(format!("new prefix failed: {e}")),
        },
        KeyCode::Char('e') => {
            let evicted = app.evict_step(16);
            app.log.push_back(format!("clock evicted {evicted} slots"));
        }
        KeyCode::Char('c') => app.launch_storm(),
        KeyCode::Char('k') => app.launch_crash(),
        KeyCode::Char('m') => app.launch_pressure(),
        KeyCode::Left => app.tab = app.tab.saturating_sub(1),
        KeyCode::Right => app.tab = (app.tab + 1).min(4),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::handle_key;
    use crate::config::Config;
    use crate::tui::app::TuiApp;
    use ratatui::crossterm::event::KeyCode;

    /// Verifies that tab keys switch tabs without panicking.
    #[test]
    fn tab_keys_switch_tabs() {
        let mut app = TuiApp::new(1 << 22, 0, true, 30, Config::default()).expect("app construction succeeds");
        handle_key(&mut app, KeyCode::Char('2'));
        assert_eq!(app.tab, 1);
        handle_key(&mut app, KeyCode::Char('5'));
        assert_eq!(app.tab, 4);
        handle_key(&mut app, KeyCode::Left);
        assert_eq!(app.tab, 3);
        handle_key(&mut app, KeyCode::Right);
        assert_eq!(app.tab, 4);
        app.shutdown();
    }

    /// Verifies that q and Esc request a quit.
    #[test]
    fn quit_keys_request_shutdown() {
        let mut app = TuiApp::new(1 << 22, 0, true, 30, Config::default()).expect("app construction succeeds");
        assert!(!app.quit);
        handle_key(&mut app, KeyCode::Char('q'));
        assert!(app.quit);
        app.shutdown();
    }

    /// Verifies that the space key toggles the pause flag, starting paused.
    #[test]
    fn space_toggles_pause() {
        let mut app = TuiApp::new(1 << 22, 0, true, 30, Config::default()).expect("app construction succeeds");
        assert!(app.paused, "workload must start paused");
        handle_key(&mut app, KeyCode::Char(' '));
        assert!(!app.paused);
        handle_key(&mut app, KeyCode::Char(' '));
        assert!(app.paused);
        app.shutdown();
    }

    /// Verifies the action keys do not panic on a fresh app.
    #[test]
    fn action_keys_are_graceful() {
        let mut app = TuiApp::new(1 << 22, 0, true, 30, Config::default()).expect("app construction succeeds");
        for code in [
            KeyCode::Char('i'),
            KeyCode::Char('s'),
            KeyCode::Char('r'),
            KeyCode::Char('n'),
            KeyCode::Char('e'),
            KeyCode::Char('c'),
            KeyCode::Char('k'),
            KeyCode::Char('m'),
        ] {
            handle_key(&mut app, code);
        }
        app.pump_log();
        app.pump_chaos();
        app.shutdown();
    }
}
