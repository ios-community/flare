//! Memory tab: arena allocator map, hazard eras, and tagged pointer map.

use crate::tui::ui::{FrameInfo, bar, percent};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Renders the memory tab content.
pub fn render(frame: &mut Frame<'_>, area: Rect, info: &FrameInfo) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(3)])
        .split(area);

    let fraction = fraction_of(info.tree_frontier, info.tree_capacity);
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            " arena frontier {} / {} ({} free)",
            info.tree_frontier,
            info.tree_capacity,
            info.tree_capacity - info.tree_frontier
        ),
        Style::default().fg(Color::White),
    )));
    lines.push(Line::from(Span::styled(
        format!(" [{}] {}", bar(fraction, 40), percent(fraction)),
        Style::default().fg(Color::White),
    )));
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Tree Arena Allocator "),
        ),
        layout[0],
    );

    let slab_lines = vec![
        kv_row(
            "hazard era",
            info.hazard_era.to_string(),
            "kv slots",
            info.kv_slots.to_string(),
        ),
        kv_row(
            "retired slots",
            info.hazard_retired.to_string(),
            "wal frames",
            info.sink_frames.to_string(),
        ),
        Line::from(Span::styled(
            " slabs: 4 KiB chunks, 2-bit clock, fetch_or(1) refcount",
            Style::default().fg(Color::DarkGray),
        )),
    ];
    frame.render_widget(
        Paragraph::new(slab_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Slab Pool / Hazard Eras "),
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

/// Fraction of the tree arena currently consumed.
#[allow(clippy::cast_precision_loss)]
fn fraction_of(frontier: u64, capacity: u64) -> f64 {
    if capacity == 0 {
        0.0
    } else {
        frontier as f64 / capacity as f64
    }
}

#[cfg(test)]
mod tests {
    use super::fraction_of;

    /// Verifies the arena fraction helper.
    #[test]
    fn arena_fraction_is_bounded() {
        assert!(fraction_of(0, 0).abs() < 1e-9);
        assert!(fraction_of(0, 100).abs() < 1e-9);
        assert!((fraction_of(100, 100) - 1.0).abs() < 1e-9);
    }
}
