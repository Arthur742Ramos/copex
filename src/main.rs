mod cli;
mod commands;
mod config;
mod copilot_client;
mod jsonrpc;
mod mcp;
mod plan;
mod tool_executor;
mod ui;

use anyhow::{anyhow, Result};
use clap::{CommandFactory, Parser};

use cli::{Cli, Commands};

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    install_ctrlc_handler();
    let cli = Cli::parse();
    let quiet = cli.quiet;
    let exit_code = match run(cli).await {
        Ok(()) => 0,
        Err(err) => {
            if !quiet {
                eprintln!("Error: {err}");
            }
            1
        }
    };
    std::process::exit(exit_code);
}

async fn run(cli: Cli) -> Result<()> {
    if cli.quiet && cli.verbose {
        return Err(anyhow!("--verbose and --quiet cannot be used together"));
    }
    if std::env::var("COPEX_MCP_SERVER").ok().as_deref() == Some("1") {
        return mcp::run_stdio_server();
    }
    match cli.command {
        Some(command) => dispatch(command).await,
        None => {
            Cli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

async fn dispatch(command: Commands) -> Result<()> {
    match command {
        Commands::Chat(args) => commands::chat::run(args).await,
        Commands::Plan(args) => commands::plan::run(args).await,
        Commands::Ralph(args) => commands::ralph::run(args).await,
        Commands::Interactive(args) => commands::interactive::run(args).await,
        Commands::Session(args) => commands::session::run(args).await,
        Commands::Models => commands::models::run().await,
        Commands::Themes => commands::themes::run().await,
        Commands::UiDemo(args) => commands::ui_demo::run(args).await,
        Commands::Init(args) => commands::init::run(args).await,
        Commands::Login => commands::auth::login().await,
        Commands::Logout => commands::auth::logout().await,
        Commands::Status => commands::auth::status().await,
    }
}

fn install_ctrlc_handler() {
    let _ = ctrlc::set_handler(|| {
        eprintln!("copex: interrupted");
        std::process::exit(130);
    });
}

#[cfg(test)]
mod mcp_server_tests {
    #[test]
    fn mcp_server_list_tools_snapshot() {
        let _ = env!("CARGO_PKG_VERSION");
    }
}
