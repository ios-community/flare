//! Frame assembly for the TUI: header tabs, tab content, and the event
//! log footer. All rendering uses only the stable ratatui primitives
//! (`Paragraph`, `Block`, `Layout`, `Line`, `Span`) and a small custom
//! text-bar helper, so the codebase stays portable across ratatui
//! releases.

use crate::chaos::crash_injector::CrashReport;
use crate::chaos::memory_pressure::PressureReport;
use crate::chaos::storm::StormReport;
use crate::tui::app::TuiApp;
use crate::tui::tab_chaos;
use crate::tui::tab_dashboard;
use crate::tui::tab_kvcache;
use crate::tui::tab_memory;
use crate::tui::tab_vector;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Names of the five dashboard tabs, numbered 1-5 for the user.
pub const TAB_NAMES: [&str; 5] = [
    " 1 | Dashboard ",
    " 2 | Memory    ",
    " 3 | Vector    ",
    " 4 | KV-Cache  ",
    " 5 | Chaos     ",
];

/// A point-in-time snapshot of every value the tabs render.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct FrameInfo {
    /// Shared engine counters.
    pub snapshot: crate::tui::app::CounterSnapshot,
    /// Live per-second rates: inserts, hits, misses.
    pub rates: [f64; 3],
    /// Decoded telemetry lines for the footer.
    pub log: Vec<String>,
    /// Workload paused indicator.
    pub paused: bool,
    /// Selected tab index.
    pub tab: usize,
    /// Tree arena frontier in bytes.
    pub tree_frontier: u64,
    /// Tree arena capacity in bytes.
    pub tree_capacity: u64,
    /// Whether the vector index is trained.
    pub vector_trained: bool,
    /// Vectors appended to the vector index.
    pub vector_count: u64,
    /// KV-cache slot capacity.
    pub kv_slots: usize,
    /// KV-cache byte capacity.
    pub kv_capacity: usize,
    /// SIMD benchmark result in nanoseconds per distance.
    pub bench_scalar_ns: f64,
    /// SIMD benchmark result in nanoseconds per distance.
    pub bench_avx2_ns: f64,
    /// Published re-clustering generation.
    pub recluster_gen: u64,
    /// Current hazard era.
    pub hazard_era: u64,
    /// Retired slots awaiting reclamation.
    pub hazard_retired: usize,
    /// WAL frames buffered in the sink.
    pub sink_frames: u64,
    /// Chaos scenario in flight indicator.
    pub chaos_busy: bool,
    /// Whether the contention storm scenario is running.
    pub storm_running: bool,
    /// Whether the crash fault-injection scenario is running.
    pub crash_running: bool,
    /// Whether the memory exhaustion scenario is running.
    pub pressure_running: bool,
    /// Last contention storm report.
    pub storm: Option<StormReport>,
    /// Last crash fault-injection report.
    pub crash: Option<CrashReport>,
    /// Last memory exhaustion report.
    pub pressure: Option<PressureReport>,
}

impl TuiApp {
    /// Renders the full application frame.
    pub fn draw(&self, frame: &mut Frame<'_>) {
        let info = self.frame_info();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(6),
            ])
            .split(frame.area());
        Self::draw_header(frame, layout[0], &info);
        match info.tab {
            0 => tab_dashboard::render(frame, layout[1], &info),
            1 => tab_memory::render(frame, layout[1], &info),
            2 => tab_vector::render(frame, layout[1], &info),
            3 => tab_kvcache::render(frame, layout[1], &info),
            _ => tab_chaos::render(frame, layout[1], &info),
        }
        Self::draw_footer(frame, layout[2], &info);
    }

    /// Collects the per-frame snapshot of engine state.
    fn frame_info(&self) -> FrameInfo {
        let tree_arena = self.tree.arena();
        FrameInfo {
            snapshot: self.counters.snapshot(),
            rates: self.rates,
            log: self.log.iter().cloned().collect(),
            paused: self.paused,
            tab: self.tab,
            tree_frontier: tree_arena.frontier(),
            tree_capacity: tree_arena.capacity(),
            vector_trained: self.vector.is_trained(),
            vector_count: self.vector.vector_count().unwrap_or(0),
            kv_slots: self.kv.slot_count(),
            kv_capacity: self.kv.capacity_bytes(),
            bench_scalar_ns: self.bench_scalar_ns,
            bench_avx2_ns: self.bench_avx2_ns,
            recluster_gen: self.recluster_gen,
            hazard_era: self.hazard.current_era(),
            hazard_retired: self.hazard.retired_len(),
            sink_frames: self.sink.frame_count(),
            chaos_busy: self.chaos_inflight > 0,
            storm_running: self.storm_running,
            crash_running: self.crash_running,
            pressure_running: self.pressure_running,
            storm: self.storm,
            crash: self.crash,
            pressure: self.pressure,
        }
    }

    /// Renders the tab bar header.
    fn draw_header(frame: &mut Frame<'_>, area: Rect, info: &FrameInfo) {
        let mut spans = Vec::with_capacity(TAB_NAMES.len());
        for (index, name) in TAB_NAMES.iter().enumerate() {
            let (fg, bg) = if index == info.tab {
                (Color::Black, Color::Cyan)
            } else {
                (Color::White, Color::DarkGray)
            };
            spans.push(Span::styled(
                *name,
                Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD),
            ));
        }
        let status = if info.paused { " PAUSED " } else { " RUNNING " };
        spans.push(Span::styled(
            status,
            Style::default()
                .fg(if info.paused {
                    Color::Yellow
                } else {
                    Color::Green
                })
                .add_modifier(Modifier::BOLD),
        ));
        frame.render_widget(
            Paragraph::new(Line::from(spans))
                .block(Block::default().borders(Borders::ALL).title(" FLARE ")),
            area,
        );
    }

    /// Renders the event log footer.
    fn draw_footer(frame: &mut Frame<'_>, area: Rect, info: &FrameInfo) {
        let mut lines = Vec::with_capacity(4);
        let log_count = info.log.len().min(3);
        let start = info.log.len().saturating_sub(log_count);
        for line in &info.log[start..] {
            lines.push(Line::from(Span::styled(
                line,
                Style::default().fg(Color::DarkGray),
            )));
        }
        while lines.len() < 3 {
            lines.insert(
                0,
                Line::from(Span::styled(
                    "no events yet",
                    Style::default().fg(Color::DarkGray),
                )),
            );
        }
        let controls = tab_controls(info.tab);
        lines.push(Line::from(Span::styled(
            controls,
            Style::default().fg(Color::Cyan),
        )));
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Events / Controls "),
            ),
            area,
        );
    }
}

/// Returns the controls string for the footer, showing only global
/// keys and keys relevant to the current tab.
fn tab_controls(tab: usize) -> String {
    let mut parts = vec![
        "[1-5] tabs",
        "[SPACE] pause",
    ];
    match tab {
        0 => parts.push("[I] burst"),
        2 => {
            parts.push("[S] bench");
            parts.push("[R] recluster");
        }
        3 => {
            parts.push("[N] new chat");
            parts.push("[E] evict");
        }
        4 => {
            parts.push("[C] storm");
            parts.push("[K] crash");
            parts.push("[M] pressure");
        }
        _ => {}
    }
    parts.push("[Q] quit");
    format!(" {}", parts.join("  "))
}

/// Renders a horizontal text gauge: `[▓▓▓░░░] 64.3%`.
///
/// The bar spans `width` terminal columns and is drawn with the full-width
/// block and light-shade characters.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn bar(fraction: f64, width: u16) -> String {
    let inner = usize::from(width.saturating_sub(2));
    let inner = inner.max(1);
    let filled = (fraction.clamp(0.0, 1.0) * inner as f64) as usize;
    let mut rendered = String::with_capacity(inner + 8);
    rendered.push('[');
    for index in 0..inner {
        if index < filled {
            rendered.push('▓');
        } else {
            rendered.push('░');
        }
    }
    rendered.push(']');
    rendered
}

/// Formats a fraction as a percentage string with one decimal.
#[allow(clippy::cast_precision_loss)]
pub fn percent(fraction: f64) -> String {
    format!("{:.1}%", fraction * 100.0)
}

#[cfg(test)]
mod tests {
    use super::{TAB_NAMES, bar, percent};

    /// Verifies that the bar renders exactly the requested width.
    #[test]
    fn bar_fills_exact_width() {
        let rendered = bar(0.5, 12);
        assert_eq!(rendered.chars().count(), 12);
        assert!(rendered.starts_with('['));
        assert!(rendered.ends_with(']'));
    }

    /// Verifies that fractions clamp into [0, 1].
    #[test]
    fn bar_clamps_fraction() {
        assert!(bar(2.0, 8).starts_with("[▓▓▓▓▓▓"));
        assert!(bar(-1.0, 8).starts_with("[░░░░░░"));
    }

    /// Verifies the percentage formatting helper.
    #[test]
    fn percent_formats_one_decimal() {
        assert_eq!(percent(0.256), "25.6%");
        assert_eq!(percent(1.0), "100.0%");
    }

    /// Verifies the tab names cover all five indices, numbered 1-5.
    #[test]
    fn tab_names_cover_five_tabs() {
        assert_eq!(TAB_NAMES.len(), 5);
        assert!(TAB_NAMES[0].contains('1'));
        assert!(TAB_NAMES[4].contains("Chaos"));
    }
}
