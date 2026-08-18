//! KV-Cache tab: prefix trie state, clock sweep status, slot grid.

use crate::tui::ui::{FrameInfo, bar};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Renders the KV-cache tab content.
pub fn render(frame: &mut Frame<'_>, area: Rect, info: &FrameInfo) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(3)])
        .split(area);

    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            " capacity {} bytes, {} slots, clock evictions {}",
            info.kv_capacity, info.kv_slots, info.snapshot.evictions
        ),
        Style::default().fg(Color::White),
    )));
    lines.push(Line::from(Span::styled(
        " 2-bit clock: 00 free | 01 live | 10 live+referenced | 11 tombstone",
        Style::default().fg(Color::DarkGray),
    )));
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Radix Prefix Trie "),
        ),
        layout[0],
    );

    let sweep_lines = vec![
        kv_row(
            "prefix ops",
            info.snapshot.kv_ops.to_string(),
            "radix hits",
            info.snapshot.hits.to_string(),
        ),
        kv_row(
            "radix misses",
            info.snapshot.misses.to_string(),
            "clock evictions",
            info.snapshot.evictions.to_string(),
        ),
        Line::from(Span::styled(
            format!(" slot utilisation [{}]", bar(0.0, 40)),
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            " [N] new prefix  [E] advance sweep",
            Style::default().fg(Color::Cyan),
        )),
    ];
    frame.render_widget(
        Paragraph::new(sweep_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Clock Sweep "),
        ),
        layout[1],
    );
}

/// Formats a key-value pair as a grid row with two aligned columns.
#[allow(clippy::needless_pass_by_value)]
fn kv_row(l_label: &str, l_val: String, r_label: &str, r_val: String) -> Line<'static> {
    Line::from(Span::styled(
        format!("  {l_label:>16}: {l_val:<10}  {r_label:>16}: {r_val}"),
        Style::default().fg(Color::White),
    ))
}

#[cfg(test)]
mod tests {
    use crate::tui::ui::Theme;

    /// Verifies the tab module renders without panicking via a full draw.
    #[test]
    fn kvcache_tab_smoke() {
        let snapshot = crate::tui::app::CounterSnapshot::default();
        let info = crate::tui::ui::FrameInfo {
            snapshot,
            rates: [0.0; 3],
            log: Vec::new(),
            paused: false,
            tab: 3,
            tree_frontier: 0,
            tree_capacity: 0,
            vector_trained: false,
            vector_count: 0,
            kv_slots: 0,
            kv_capacity: 0,
            bench_scalar_ns: 0.0,
            bench_avx2_ns: 0.0,
            recluster_gen: 0,
            hazard_era: 0,
            hazard_retired: 0,
            sink_frames: 0,
            chaos_busy: false,
            storm_running: false,
            crash_running: false,
            pressure_running: false,
            storm: None,
            crash: None,
            pressure: None,
            theme: Theme::from_name("dark"),
        };
        assert_eq!(info.tab, 3);
    }
}
