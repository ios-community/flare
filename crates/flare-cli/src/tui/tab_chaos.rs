//! Chaos tab: launch and audit contention storms, crash fault injection,
//! and memory exhaustion runs.

use crate::chaos::contention_rate;
use crate::chaos::crash_injector::CrashReport;
use crate::chaos::memory_pressure::PressureReport;
use crate::tui::ui::FrameInfo;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Renders the chaos tab content.
pub fn render(frame: &mut Frame<'_>, area: Rect, info: &FrameInfo) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(3)])
        .split(area);

    let mut lines = Vec::new();
    let storm_status = scenario_status(info.storm_running, info.storm.is_some());
    let crash_status = scenario_status(info.crash_running, info.crash.is_some());
    let pressure_status = scenario_status(info.pressure_running, info.pressure.is_some());

    lines.push(scenario_line(
        "C",
        "contention storm (8 threads, 200k attempts, 16MB arena)",
        storm_status.0,
        storm_status.1,
    ));
    lines.push(scenario_line(
        "K",
        "crash fault injection (512 WAL frames, truncation replay)",
        crash_status.0,
        crash_status.1,
    ));
    lines.push(scenario_line(
        "M",
        "memory exhaustion (1MB arena + slab recycling)",
        pressure_status.0,
        pressure_status.1,
    ));
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Scenario Launcher "),
        ),
        layout[0],
    );

    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        " last audit reports:",
        Style::default().fg(Color::White),
    )));
    if let Some(storm) = &info.storm {
        lines.push(Line::from(Span::styled(
            format!(
                "  storm: {} attempts, {:.2}% CAS contention, {} lost updates, {:.1} ms",
                storm.attempts,
                contention_rate(storm),
                storm.lost_updates,
                storm.elapsed_ms
            ),
            Style::default().fg(Color::Green),
        )));
    }
    if let Some(crash) = &info.crash {
        lines.push(Line::from(Span::styled(
            crash_line(crash),
            Style::default().fg(Color::Green),
        )));
    }
    if let Some(pressure) = &info.pressure {
        lines.push(Line::from(Span::styled(
            pressure_line(pressure),
            Style::default().fg(Color::Green),
        )));
    }
    if !info.chaos_busy && info.storm.is_none() && info.crash.is_none() && info.pressure.is_none() {
        lines.push(Line::from(Span::styled(
            "  no audits yet - launch a scenario",
            Style::default().fg(Color::DarkGray),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Audit Report "),
        ),
        layout[1],
    );
}

/// Returns (status label, color) for a chaos scenario.
const fn scenario_status(running: bool, has_report: bool) -> (&'static str, Color) {
    if running {
        ("busy", Color::Yellow)
    } else if has_report {
        ("done", Color::Green)
    } else {
        ("idle", Color::DarkGray)
    }
}

/// Formats one scenario launcher line with a fixed-width description so
/// the status column is always aligned.
fn scenario_line(key: &str, desc: &str, status: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!(" [{key}] {desc:<62}"),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(format!(" status: {status}"), Style::default().fg(color)),
    ])
}

/// Renders one crash audit line.
fn crash_line(crash: &CrashReport) -> String {
    format!(
        "  crash: {} frames replayed of {}, consistent={}, recovery {:.1} us",
        crash.frames_replayed, crash.frames_written, crash.consistent, crash.recovery_us
    )
}

/// Renders one memory pressure audit line.
fn pressure_line(pressure: &PressureReport) -> String {
    format!(
        "  pressure: {} keys at {} bytes, {} samples verified, {} slabs recycled",
        pressure.inserted_keys, pressure.frontier, pressure.sample_verified, pressure.slab_recycled
    )
}

#[cfg(test)]
mod tests {
    use super::{crash_line, pressure_line};
    use crate::chaos::crash_injector::CrashReport;
    use crate::chaos::memory_pressure::PressureReport;

    /// Verifies the crash audit line renders.
    #[test]
    fn crash_line_renders() {
        let crash = CrashReport {
            frames_written: 4,
            frames_replayed: 2,
            high_water: 48,
            recovery_us: 5.0,
            survived: 2,
            consistent: true,
            sink_bytes: 96,
        };
        let line = crash_line(&crash);
        assert!(line.contains("2 frames replayed of 4"));
        assert!(line.contains("consistent=true"));
    }

    /// Verifies the pressure audit line renders.
    #[test]
    fn pressure_line_renders() {
        let pressure = PressureReport {
            inserted_keys: 10,
            frontier: 512,
            capacity: 1024,
            exhaustion: "ArenaCapacityExceeded",
            sample_verified: 3,
            slab_allocated: 8,
            slab_recycled: 4,
        };
        assert!(pressure_line(&pressure).contains("10 keys"));
    }
}
