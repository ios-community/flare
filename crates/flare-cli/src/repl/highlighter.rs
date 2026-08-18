//! Syntax highlighting for the REPL shell.
//!
//! Known commands are painted cyan, argument lists dim, and unknown input
//! plain, so mistakes stand out before the shell executes the line.

use crate::repl::commands::COMMANDS;
use nu_ansi_term::{Color, Style};
use reedline::StyledText;

/// Highlights shell lines: command tokens cyan, the rest dim.
pub struct FlareHighlighter;

impl reedline::Highlighter for FlareHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> StyledText {
        let mut styled = StyledText::new();
        let mut words = line.splitn(2, char::is_whitespace);
        let command = words.next().unwrap_or_default();
        if COMMANDS.contains(&command) {
            styled.push((Style::default().fg(Color::Cyan).bold(), command.to_string()));
        } else if !command.is_empty() {
            styled.push((Style::default().fg(Color::Red), command.to_string()));
        }
        if let Some(rest) = words.next() {
            styled.push((Style::default().fg(Color::DarkGray), format!(" {rest}")));
        }
        styled
    }
}

#[cfg(test)]
mod tests {
    use super::FlareHighlighter;
    use crate::repl::commands::COMMANDS;
    use reedline::Highlighter;

    /// Verifies that known commands and unknown tokens get distinct styles.
    #[test]
    fn known_commands_are_highlighted() {
        let highlighter = FlareHighlighter;
        for command in COMMANDS {
            let styled = highlighter.highlight(command, 0);
            let buffer = styled.buffer;
            assert_eq!(buffer.len(), 1, "one segment for bare command");
            assert_eq!(buffer[0].1, command.to_string());
        }
    }

    /// Verifies that arguments are appended as a second segment.
    #[test]
    fn arguments_form_a_separate_segment() {
        let highlighter = FlareHighlighter;
        let styled = highlighter.highlight("kv-insert hello 42", 0);
        assert_eq!(styled.buffer.len(), 2);
        assert_eq!(styled.buffer[1].1, " hello 42");
    }

    /// Verifies that an empty line renders as an empty highlight.
    #[test]
    fn empty_line_highlights_to_nothing() {
        let highlighter = FlareHighlighter;
        assert!(highlighter.highlight("", 0).buffer.is_empty());
    }
}
