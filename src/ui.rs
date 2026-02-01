use std::time::{Duration, Instant};

use ratatui::prelude::*;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, Theme};
use syntect::parsing::SyntaxSet;

use anyhow::Result;
use futures::StreamExt;

use crate::copilot_client::{ChunkType, StreamChunk};

const TOOL_PREFIX: &str = "🔧";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    Thinking,
    Responding,
    Done,
}


pub struct Spinner {
    frames: &'static [&'static str],
    index: usize,
    last_tick: Instant,
    interval: Duration,
}

impl Spinner {
    pub fn new(_ascii: bool) -> Self {
        Self {
            frames: &["-", "\\", "|", "/"],
            index: 0,
            last_tick: Instant::now(),
            interval: Duration::from_millis(80),
        }
    }

    pub fn tick(&mut self) {
        if self.last_tick.elapsed() >= self.interval {
            self.index = (self.index + 1) % self.frames.len();
            self.last_tick = Instant::now();
        }
    }

    pub fn frame(&self) -> &'static str {
        self.frames[self.index]
    }
}

pub struct UiState {
    pub activity: Activity,
    pub message: String,
    pub reasoning: String,
    pub tool_log: Vec<String>,
    pub start_time: Instant,
    pub tokens_used: u64,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            activity: Activity::Thinking,
            message: String::new(),
            reasoning: String::new(),
            tool_log: Vec::new(),
            start_time: Instant::now(),
            tokens_used: 0,
        }
    }
}

pub struct CopexUi {
    spinner: Spinner,
    pub state: UiState,
    syntax_set: SyntaxSet,
    theme: Theme,
}

impl CopexUi {
    pub fn new(theme_name: Option<&str>, ascii: bool) -> Self {
        let syntax_set = SyntaxSet::load_defaults_newlines();
        let themes = syntect::highlighting::ThemeSet::load_defaults();
        let theme = theme_name
            .and_then(|name| themes.themes.get(name).cloned())
            .or_else(|| themes.themes.get("base16-ocean.dark").cloned())
            .unwrap_or_else(|| themes.themes.values().next().cloned().unwrap());
        Self {
            spinner: Spinner::new(ascii),
            state: UiState::new(),
            syntax_set,
            theme,
        }
    }

    pub fn tick(&mut self) {
        self.spinner.tick();
    }

    pub fn render(&mut self, frame: &mut Frame) {
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5)].as_ref())
            .split(frame.area());

        let status = self.status_line();
        let status_block = Paragraph::new(status)
            .block(Block::default().borders(Borders::ALL).title("Status"));
        frame.render_widget(status_block, layout[0]);

        let body = Paragraph::new(self.highlight_message())
            .wrap(Wrap { trim: false })
            .block(Block::default().borders(Borders::ALL).title("Response"));
        frame.render_widget(body, layout[1]);
    }

    fn status_line(&self) -> String {
        let elapsed = self.state.start_time.elapsed().as_secs();
        let label = match self.state.activity {
            Activity::Thinking => "Thinking",
            Activity::Responding => "Responding",
            Activity::Done => "Done",
        };
        format!(
            "{} {} - {}s - {} tokens",
            self.spinner.frame(),
            label,
            elapsed,
            self.state.tokens_used
        )
    }

    fn highlight_message(&self) -> Text<'_> {
        let mut combined = String::new();
        if !self.state.reasoning.is_empty() {
            combined.push_str("[Reasoning]\n");
            combined.push_str(&self.state.reasoning);
            combined.push_str("\n\n");
        }
        if !self.state.tool_log.is_empty() {
            combined.push_str("[Tools]\n");
            for line in &self.state.tool_log {
                combined.push_str(line);
                combined.push('\n');
            }
            combined.push('\n');
        }
        combined.push_str(&self.state.message);

        if combined.is_empty() {
            return Text::from(Line::from(""));
        }
        let syntax = self
            .syntax_set
            .find_syntax_by_extension("md")
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text());
        let mut highlighter = HighlightLines::new(syntax, &self.theme);
        let mut lines = Vec::new();
        for line in combined.lines() {
            let ranges = highlighter
                .highlight_line(line, &self.syntax_set)
                .unwrap_or_default();
            let spans: Vec<Span> = ranges
                .into_iter()
                .map(|(style, text)| Span::styled(text.to_string(), to_ratatui_style(style)))
                .collect();
            lines.push(Line::from(spans));
        }
        Text::from(lines)
    }
}

pub struct StreamUiResult {
    #[allow(dead_code)]
    pub message: String,
    #[allow(dead_code)]
    pub tokens_used: u64,
}

pub async fn run_stream_ui<S>(
    stream: S,
    theme: Option<&str>,
    ascii: bool,
    show_reasoning: bool,
) -> Result<StreamUiResult>
where
    S: futures::Stream<Item = Result<StreamChunk>>,
{
    let mut terminal = setup_terminal()?;
    let mut ui = CopexUi::new(theme, ascii);
    let mut stream = Box::pin(stream);
    let mut ticker = tokio::time::interval(Duration::from_millis(80));

    let mut done = false;
    draw(&mut terminal, &mut ui)?;
    while !done {
        let mut stream_ref = stream.as_mut();
        let next_chunk = stream_ref.next();
        tokio::select! {
            _ = ticker.tick() => {
                ui.tick();
                draw(&mut terminal, &mut ui)?;
            }
            chunk = next_chunk => {
                match chunk {
                    Some(chunk) => {
                        let chunk = chunk?;
                        if ui.state.activity == Activity::Thinking {
                            ui.state.activity = Activity::Responding;
                        }
                        handle_chunk(&mut ui, &chunk, show_reasoning);
                        if chunk.is_final {
                            ui.state.activity = Activity::Done;
                            draw(&mut terminal, &mut ui)?;
                            done = true;
                        }
                    }
                    None => {
                        ui.state.activity = Activity::Done;
                        draw(&mut terminal, &mut ui)?;
                        done = true;
                    }
                }
            }
        }
    }
    let message = ui.state.message.clone();
    let tokens_used = ui.state.tokens_used;
    teardown_terminal(&mut terminal)?;
    Ok(StreamUiResult { message, tokens_used })
}

fn handle_chunk(ui: &mut CopexUi, chunk: &StreamChunk, show_reasoning: bool) {
    if !chunk.delta.is_empty() {
        ui.state.tokens_used += estimate_tokens(&chunk.delta);
    }
    match chunk.chunk_type {
        ChunkType::Message => {
            if !chunk.delta.is_empty() {
                ui.state.message.push_str(&chunk.delta);
            }
        }
        ChunkType::Reasoning => {
            if !show_reasoning {
                return;
            }
            if !chunk.delta.is_empty() {
                ui.state.reasoning.push_str(&chunk.delta);
            }
        }
        ChunkType::ToolCall => {
            let line = if chunk.delta.starts_with(TOOL_PREFIX) {
                chunk.delta.clone()
            } else {
                format!("{} {}", TOOL_PREFIX, chunk.delta)
            };
            if ui.state.tool_log.last().map(|v| v.as_str()) != Some(line.as_str()) {
                ui.state.tool_log.push(line);
            }
        }
        ChunkType::ToolResult => {
            if !chunk.delta.is_empty() {
                ui.state.tool_log.push(chunk.delta.clone());
            }
        }
        ChunkType::System => {
            if !chunk.delta.is_empty() {
                ui.state.tool_log.push(format!("[{}]", chunk.delta));
            }
        }
    }
}

fn estimate_tokens(text: &str) -> u64 {
    text.split_whitespace().count() as u64
}

fn to_ratatui_style(style: syntect::highlighting::Style) -> ratatui::style::Style {
    let mut rat_style = ratatui::style::Style::default()
        .fg(ratatui::style::Color::Rgb(
            style.foreground.r,
            style.foreground.g,
            style.foreground.b,
        ))
        .bg(ratatui::style::Color::Rgb(
            style.background.r,
            style.background.g,
            style.background.b,
        ));
    if style.font_style.contains(FontStyle::BOLD) {
        rat_style = rat_style.add_modifier(ratatui::style::Modifier::BOLD);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        rat_style = rat_style.add_modifier(ratatui::style::Modifier::ITALIC);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        rat_style = rat_style.add_modifier(ratatui::style::Modifier::UNDERLINED);
    }
    rat_style
}

fn setup_terminal() -> Result<ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>> {
    use crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};
    use std::io::stdout;

    enable_raw_mode()?;
    let mut stdout = stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let terminal = ratatui::Terminal::new(backend)?;
    Ok(terminal)
}

fn teardown_terminal(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
) -> Result<()> {
    use crossterm::terminal::{disable_raw_mode, LeaveAlternateScreen};
    disable_raw_mode().ok();
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn draw(
    terminal: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    ui: &mut CopexUi,
) -> Result<()> {
    terminal.draw(|frame| ui.render(frame))?;
    Ok(())
}
