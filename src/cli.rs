use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "copex", version, about = "Copilot Extended CLI")]
pub struct Cli {
    #[arg(long, short, global = true)]
    pub verbose: bool,
    #[arg(long, short, global = true)]
    pub quiet: bool,
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Chat(ChatArgs),
    Plan(PlanArgs),
    Ralph(RalphArgs),
    Interactive(InteractiveArgs),
    Session(SessionArgs),
    Models,
    Themes,
    UiDemo(UiDemoArgs),
    Init(InitArgs),
    Login,
    Logout,
    Status,
}

#[derive(Debug, Args)]
pub struct ChatArgs {
    pub prompt: Option<String>,
    #[arg(long, short)]
    pub model: Option<String>,
    #[arg(long, short)]
    pub reasoning: Option<String>,
    #[arg(long, default_value_t = 5)]
    pub max_retries: u32,
    #[arg(long)]
    pub no_stream: bool,
    #[arg(long, default_value_t = true)]
    pub show_reasoning: bool,
    #[arg(long, conflicts_with = "show_reasoning")]
    pub no_reasoning: bool,
    #[arg(long, short)]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub raw: bool,
    #[arg(long)]
    pub no_color: bool,
    #[arg(long)]
    pub plain: bool,
    #[arg(long)]
    pub log_level: Option<String>,
    #[arg(long)]
    pub log_file: Option<PathBuf>,
    #[arg(long)]
    pub ui_theme: Option<String>,
    #[arg(long)]
    pub ui_density: Option<String>,
}

#[derive(Debug, Args)]
pub struct PlanArgs {
    pub task: Option<String>,
    #[arg(long, short)]
    pub execute: bool,
    #[arg(long, short = 'R')]
    pub review: bool,
    #[arg(long)]
    pub resume: bool,
    #[arg(long, short)]
    pub output: Option<PathBuf>,
    #[arg(long, short = 'f', default_value_t = 1)]
    pub from_step: usize,
    #[arg(long, short = 'l')]
    pub load: Option<PathBuf>,
    #[arg(long, short = 'n', default_value_t = 10)]
    pub max_iterations: usize,
    #[arg(long, short)]
    pub model: Option<String>,
    #[arg(long, short)]
    pub reasoning: Option<String>,
    #[arg(long, default_value = "terminal", value_parser = ["terminal", "rich", "json", "quiet"])]
    pub progress: String,
    #[arg(long)]
    pub log_level: Option<String>,
    #[arg(long)]
    pub log_file: Option<PathBuf>,
    #[arg(long)]
    pub mcp_config: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct RalphArgs {
    pub prompt: String,
    #[arg(long, short = 'n', default_value_t = 30)]
    pub max_iterations: usize,
    #[arg(long, short)]
    pub promise: Option<String>,
    #[arg(long, short)]
    pub model: Option<String>,
    #[arg(long, short)]
    pub reasoning: Option<String>,
    #[arg(long)]
    pub log_level: Option<String>,
    #[arg(long)]
    pub log_file: Option<PathBuf>,
    #[arg(long)]
    pub mcp_config: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct InitArgs {
    #[arg(long, short, default_value = "~/.copex/config.toml")]
    pub path: String,
}

#[derive(Debug, Args)]
pub struct InteractiveArgs {
    #[arg(long, short)]
    pub model: Option<String>,
    #[arg(long, short)]
    pub reasoning: Option<String>,
    #[arg(long)]
    pub ui_theme: Option<String>,
    #[arg(long)]
    pub ui_density: Option<String>,
    #[arg(long)]
    pub no_color: bool,
    #[arg(long)]
    pub plain: bool,
    #[arg(long, short = 't')]
    pub theme: Option<String>,
    #[arg(long)]
    pub log_level: Option<String>,
    #[arg(long)]
    pub log_file: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct UiDemoArgs {
    #[arg(long, short)]
    pub theme: Option<String>,
    #[arg(long)]
    pub plain: bool,
    #[arg(long, short)]
    pub component: Option<String>,
}

#[derive(Debug, Args)]
pub struct SessionArgs {
    #[command(subcommand)]
    pub command: Option<SessionCommand>,
    #[arg(long)]
    pub export: Option<String>,
    #[arg(long, default_value = "json")]
    pub format: String,
    #[arg(long, short)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub enum SessionCommand {
    /// List saved sessions
    List,
    /// Export a session to a file
    Export {
        /// Session ID to export
        session_id: String,
        /// Output file path
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Export format (json or md)
        #[arg(long, short, default_value = "json")]
        format: String,
    },
    /// Delete a session
    Delete {
        /// Session ID to delete
        session_id: String,
    },
}
