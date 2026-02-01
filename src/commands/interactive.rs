use std::pin::pin;

use anyhow::Result;
use futures::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Config as RustyConfig};

use crate::cli::InteractiveArgs;
use crate::config::Config;
use crate::copilot_client::{Client, ChunkType};

pub async fn run(args: InteractiveArgs) -> Result<()> {
    // Build config from args
    let mut config = Config::load()?;
    if let Some(model) = args.model {
        config.model = model;
    }
    if let Some(reasoning) = args.reasoning {
        config.reasoning_effort = reasoning;
    }
    if let Some(theme) = args.ui_theme {
        config.ui_theme = theme;
    }
    if let Some(density) = args.ui_density {
        config.ui_density = density;
    }
    if args.no_color || args.plain {
        config.ui_ascii_icons = true;
    }
    
    println!("╭──────────────────────────────────────────────────────────────────────────────╮");
    println!("│                          Copex Interactive Mode                              │");
    println!("│                                                                              │");
    println!("│  Model: {:<60}  │", config.model);
    println!("│  Reasoning: {:<56}  │", config.reasoning_effort);
    println!("│                                                                              │");
    println!("│  Commands: /exit, /clear, /help, /model <name>, /reasoning <level>          │");
    println!("╰──────────────────────────────────────────────────────────────────────────────╯");
    println!();
    
    // Create client
    let client = Client::new(config.clone())?;
    
    // Set up rustyline editor
    let rl_config = RustyConfig::builder()
        .history_ignore_space(true)
        .auto_add_history(true)
        .build();
    let mut rl = DefaultEditor::with_config(rl_config)?;
    
    // Load history if exists
    let history_path = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("copex")
        .join("history.txt");
    let _ = rl.load_history(&history_path);
    
    loop {
        let readline = rl.readline("copex> ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                
                // Handle commands
                if trimmed.starts_with('/') {
                    match handle_command(trimmed, &mut config) {
                        CommandResult::Exit => break,
                        CommandResult::Clear => {
                            // Clear screen
                            print!("\x1B[2J\x1B[1;1H");
                            continue;
                        }
                        CommandResult::Help => {
                            print_help();
                            continue;
                        }
                        CommandResult::Continue => continue,
                        CommandResult::NotACommand => {
                            // Not a command, treat as prompt
                        }
                    }
                }
                
                // Send to model
                match client.chat(trimmed).await {
                    Ok(stream) => {
                        let mut stream = pin!(stream);
                        let mut first_message = true;
                        while let Some(chunk_result) = stream.next().await {
                            match chunk_result {
                                Ok(chunk) => {
                                    match chunk.chunk_type {
                                        ChunkType::Message => {
                                            if first_message && !chunk.delta.is_empty() {
                                                print!("\n");
                                                first_message = false;
                                            }
                                            print!("{}", chunk.delta);
                                            std::io::Write::flush(&mut std::io::stdout()).ok();
                                        }
                                        ChunkType::Reasoning => {
                                            // Show reasoning in dim
                                            print!("\x1B[2m{}\x1B[0m", chunk.delta);
                                            std::io::Write::flush(&mut std::io::stdout()).ok();
                                        }
                                        ChunkType::ToolCall => {
                                            println!("\n\x1B[33m{}\x1B[0m", chunk.delta);
                                        }
                                        ChunkType::ToolResult => {
                                            println!("\x1B[32m{}\x1B[0m", chunk.delta);
                                        }
                                        ChunkType::System => {
                                            println!("\x1B[36m{}\x1B[0m", chunk.delta);
                                        }
                                    }
                                    if chunk.is_final {
                                        println!("\n");
                                        break;
                                    }
                                }
                                Err(e) => {
                                    eprintln!("\n\x1B[31mError: {}\x1B[0m\n", e);
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("\x1B[31mError: {}\x1B[0m\n", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }
    
    // Save history
    if let Some(parent) = history_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = rl.save_history(&history_path);
    
    Ok(())
}

enum CommandResult {
    Exit,
    Clear,
    Help,
    Continue,
    NotACommand,
}

fn handle_command(cmd: &str, config: &mut Config) -> CommandResult {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts.first().map(|s| s.to_lowercase()).as_deref() {
        Some("/exit") | Some("/quit") => CommandResult::Exit,
        Some("/clear") => CommandResult::Clear,
        Some("/help") => CommandResult::Help,
        Some("/model") => {
            if let Some(model) = parts.get(1) {
                config.model = model.to_string();
                println!("Model set to: {}", config.model);
            } else {
                println!("Current model: {}", config.model);
            }
            CommandResult::Continue
        }
        Some("/reasoning") => {
            if let Some(level) = parts.get(1) {
                config.reasoning_effort = level.to_string();
                println!("Reasoning set to: {}", config.reasoning_effort);
            } else {
                println!("Current reasoning: {}", config.reasoning_effort);
            }
            CommandResult::Continue
        }
        _ => CommandResult::NotACommand,
    }
}

fn print_help() {
    println!();
    println!("Commands:");
    println!("  /model <name>      - Change model (e.g., /model claude-opus-4.5)");
    println!("  /reasoning <level> - Change reasoning (low, medium, high, xhigh)");
    println!("  /clear             - Clear screen");
    println!("  /help              - Show this help");
    println!("  /exit              - Exit");
    println!();
}
