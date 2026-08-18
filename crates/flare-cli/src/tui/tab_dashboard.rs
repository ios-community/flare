//! Dashboard tab: live gauges for the three engines plus a counters grid.

use crate::tui::ui::{FrameInfo, bar, percent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Renders the dashboard tab content.
pub fn render(frame: &mut Frame<'_>, area: Rect, info: &FrameInfo) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(3)])
        .split(area);
    let gauges = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(layout[0]);

    frame.render_widget(
        gauge_block(
            format!(
                " Tree Arena  {} / {}",
                format_bytes(info.tree_frontier),
                format_bytes(info.tree_capacity)
            ),
            gauge_fraction(info.tree_frontier, info.tree_capacity),
        ),
        gauges[0],
    );

    frame.render_widget(
        gauge_block(
            format!(
                " Vector Index  trained={} vectors={}",
                if info.vector_trained { "yes" } else { "no" },
                info.vector_count
            ),
            0.0,
        ),
        gauges[1],
    );

    frame.render_widget(
        gauge_block(
            format!(
                " KV-Cache Slab  slots={} capacity={}",
                info.kv_slots,
                format_bytes(info.kv_capacity as u64)
            ),
            0.0,
        ),
        gauges[2],
    );

    let counters = Paragraph::new(counter_lines(info)).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Live Counters "),
    );
    frame.render_widget(counters, layout[1]);
}

/// Computes the fraction to draw for a gauge.
#[allow(clippy::cast_precision_loss)]
fn gauge_fraction(frontier: u64, capacity: u64) -> f64 {
    if capacity == 0 {
        return 0.0;
    }
    frontier as f64 / capacity as f64
}

/// Builds a gauge block: label, text bar and percentage.
fn gauge_block(label: String, fraction: f64) -> Paragraph<'static> {
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        label,
        Style::default().fg(Color::Cyan),
    )));
    lines.push(Line::from(Span::styled(
        format!(" [{}] {}", bar(fraction, 24), percent(fraction)),
        Style::default().fg(Color::White),
    )));
    Paragraph::new(lines).block(Block::default().borders(Borders::ALL))
}

/// Builds the counter lines for the dashboard body: 2-column grid with
/// aligned labels and values.
#[allow(clippy::cast_precision_loss)]
fn counter_lines(info: &FrameInfo) -> Vec<Line<'static>> {
    let s = info.snapshot;
    let hit_rate = if s.hits + s.misses == 0 {
        0.0
    } else {
        s.hits as f64 / (s.hits + s.misses) as f64
    };
    vec![
        kv_row(
            "inserts/s",
            format!("{:.0}", info.rates[0]),
            "total inserts",
            s.inserts.to_string(),
        ),
        kv_row(
            "hits/s",
            format!("{:.0}", info.rates[1]),
            "total hits",
            s.hits.to_string(),
        ),
        kv_row(
            "misses/s",
            format!("{:.0}", info.rates[2]),
            "total misses",
            s.misses.to_string(),
        ),
        kv_row(
            "hit rate",
            percent(hit_rate),
            "contended CAS",
            s.contended.to_string(),
        ),
        kv_row(
            "vector ops",
            s.vector_ops.to_string(),
            "kv ops",
            s.kv_ops.to_string(),
        ),
        kv_row(
            "wal frames",
            s.wal_frames.to_string(),
            "errors",
            s.errors.to_string(),
        ),
        kv_row(
            "clock evictions",
            s.evictions.to_string(),
            "hazard era",
            format!("{} (ret {})", info.hazard_era, info.hazard_retired),
        ),
    ]
}

/// Formats a key-value pair as a grid row with two aligned columns.
#[allow(clippy::needless_pass_by_value)]
fn kv_row(l_label: &str, l_val: String, r_label: &str, r_val: String) -> Line<'static> {
    Line::from(Span::styled(
        format!("  {l_label:>16}: {l_val:<10}  {r_label:>16}: {r_val}"),
        Style::default().fg(Color::White),
    ))
}

/// Formats a byte count with a binary suffix.
#[allow(clippy::cast_precision_loss)]
fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    let value = bytes as f64;
    if value >= KIB * KIB * KIB {
        format!("{:.1} GiB", value / (KIB * KIB * KIB))
    } else if value >= KIB * KIB {
        format!("{:.1} MiB", value / (KIB * KIB))
    } else if value >= KIB {
        format!("{:.1} KiB", value / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::{format_bytes, gauge_fraction};

    /// Verifies the binary byte formatting.
    #[test]
    fn bytes_format_with_suffixes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1 << 20), "1.0 MiB");
        assert_eq!(format_bytes(1 << 30), "1.0 GiB");
    }

    /// Verifies gauge fractions stay in [0, 1].
    #[test]
    fn gauge_fractions_are_bounded() {
        assert!(gauge_fraction(0, 0).abs() < 1e-9);
        assert!((gauge_fraction(50, 100) - 0.5).abs() < 1e-9);
    }
}
