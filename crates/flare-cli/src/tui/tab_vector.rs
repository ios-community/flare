//! Vector tab: IVF centroid map, SIMD-vs-scalar benchmark, re-clustering
//! status.

use crate::tui::ui::{FrameInfo, bar};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

/// Renders the vector tab content.
pub fn render(frame: &mut Frame<'_>, area: Rect, info: &FrameInfo) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(3)])
        .split(area);

    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!(
            " index trained: {}   vectors: {}   re-cluster generation: {}",
            if info.vector_trained { "yes" } else { "no" },
            info.vector_count,
            info.recluster_gen
        ),
        Style::default().fg(Color::White),
    )));
    lines.push(Line::from(Span::styled(
        " centroids: 16 clusters, 64-dim vectors, 4 sub-vectors, ADC distance",
        Style::default().fg(Color::DarkGray),
    )));
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" IVF-PQ Index "),
        ),
        layout[0],
    );

    let mut lines = Vec::new();
    if info.bench_avx2_ns > 0.0 {
        let scalar_frac = info.bench_avx2_ns / info.bench_scalar_ns.max(info.bench_avx2_ns);
        lines.push(Line::from(Span::styled(
            format!(
                " l2_sq (scalar)        {:>8.2} ns/op  [{}]",
                info.bench_scalar_ns,
                bar(scalar_frac, 30)
            ),
            Style::default().fg(Color::White),
        )));
        lines.push(Line::from(Span::styled(
            format!(
                " l2_sq_dispatch (SIMD) {:>8.2} ns/op  [{}]",
                info.bench_avx2_ns,
                bar(1.0 - scalar_frac, 30)
            ),
            Style::default().fg(Color::Cyan),
        )));
        let speedup = info.bench_scalar_ns / info.bench_avx2_ns.max(1e-9);
        let delta_percent = (speedup - 1.0) * 100.0;
        lines.push(Line::from(Span::styled(
            format!(" speedup {speedup:.2}x    ({delta_percent:+.0}%)"),
            Style::default().fg(Color::Green),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            " press [S] to run the SIMD-vs-scalar benchmark",
            Style::default().fg(Color::DarkGray),
        )));
    }
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" SIMD vs Scalar (ADC) "),
        ),
        layout[1],
    );
}

#[cfg(test)]
mod tests {
    use crate::tui::ui::percent;

    /// Verifies the percent helper stays consistent.
    #[test]
    fn percent_helper_is_shared() {
        assert_eq!(percent(0.0), "0.0%");
    }
}
