use std::path::PathBuf;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::cli::{SessionArgs, SessionCommand};

/// A saved session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedSession {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub model: String,
    pub messages: Vec<SessionMessage>,
}

/// A message in a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tool_calls: Vec<ToolCallRecord>,
}

/// A tool call record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub name: String,
    pub arguments: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

fn get_sessions_dir() -> Result<PathBuf> {
    let data_dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("copex")
        .join("sessions");
    std::fs::create_dir_all(&data_dir)?;
    Ok(data_dir)
}

fn list_sessions() -> Result<Vec<SavedSession>> {
    let dir = get_sessions_dir()?;
    let mut sessions = Vec::new();
    
    if dir.exists() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let content = std::fs::read_to_string(&path)?;
                if let Ok(session) = serde_json::from_str::<SavedSession>(&content) {
                    sessions.push(session);
                }
            }
        }
    }
    
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(sessions)
}

fn export_session(session_id: &str, output: Option<PathBuf>, format: &str) -> Result<()> {
    let dir = get_sessions_dir()?;
    let session_path = dir.join(format!("{}.json", session_id));
    
    if !session_path.exists() {
        anyhow::bail!("Session not found: {}", session_id);
    }
    
    let content = std::fs::read_to_string(&session_path)?;
    let session: SavedSession = serde_json::from_str(&content)?;
    
    let output_content = match format {
        "json" => serde_json::to_string_pretty(&session)?,
        "md" | "markdown" => {
            let mut md = String::new();
            md.push_str(&format!("# Session: {}\n\n", session.id));
            md.push_str(&format!("**Model:** {}\n", session.model));
            md.push_str(&format!("**Created:** {}\n", session.created_at));
            md.push_str(&format!("**Updated:** {}\n\n", session.updated_at));
            md.push_str("---\n\n");
            
            for msg in &session.messages {
                let role_icon = match msg.role.as_str() {
                    "user" => "👤",
                    "assistant" => "🤖",
                    _ => "📝",
                };
                md.push_str(&format!("## {} {}\n\n", role_icon, msg.role));
                md.push_str(&format!("{}\n\n", msg.content));
                
                if let Some(reasoning) = &msg.reasoning {
                    md.push_str(&format!("<details>\n<summary>Reasoning</summary>\n\n{}\n\n</details>\n\n", reasoning));
                }
                
                if !msg.tool_calls.is_empty() {
                    md.push_str("### Tool Calls\n\n");
                    for tool in &msg.tool_calls {
                        md.push_str(&format!("- **{}**: `{}`\n", tool.name, tool.arguments));
                        if let Some(result) = &tool.result {
                            md.push_str(&format!("  - Result: {}\n", result));
                        }
                    }
                    md.push_str("\n");
                }
            }
            md
        }
        _ => anyhow::bail!("Unsupported format: {}. Use 'json' or 'md'", format),
    };
    
    let output_path = output.unwrap_or_else(|| {
        PathBuf::from(format!("session_{}.{}", session_id, format))
    });
    
    std::fs::write(&output_path, &output_content)
        .with_context(|| format!("Failed to write to {}", output_path.display()))?;
    
    println!("Exported session to: {}", output_path.display());
    Ok(())
}

fn delete_session(session_id: &str) -> Result<()> {
    let dir = get_sessions_dir()?;
    let session_path = dir.join(format!("{}.json", session_id));
    
    if !session_path.exists() {
        anyhow::bail!("Session not found: {}", session_id);
    }
    
    std::fs::remove_file(&session_path)?;
    println!("Deleted session: {}", session_id);
    Ok(())
}

pub async fn run(args: SessionArgs) -> Result<()> {
    if let Some(export_id) = args.export.as_deref() {
        export_session(export_id, args.output.clone(), &args.format)?;
        return Ok(());
    }
    match args.command {
        Some(command) => match command {
            SessionCommand::List => {
                let sessions = list_sessions()?;
                if sessions.is_empty() {
                    println!("No saved sessions found.");
                } else {
                    println!("╭────────────────────────────────────────────────────────────────────────────╮");
                    println!("│ {:36} │ {:20} │ {:10} │", "Session ID", "Updated", "Messages");
                    println!("├────────────────────────────────────────────────────────────────────────────┤");
                    for session in sessions {
                        let updated = session.updated_at.format("%Y-%m-%d %H:%M").to_string();
                        println!(
                            "│ {:36} │ {:20} │ {:10} │",
                            session.id,
                            updated,
                            session.messages.len()
                        );
                    }
                    println!("╰────────────────────────────────────────────────────────────────────────────╯");
                }
            }
            SessionCommand::Export {
                session_id,
                output,
                format,
            } => {
                export_session(&session_id, output, &format)?;
            }
            SessionCommand::Delete { session_id } => {
                delete_session(&session_id)?;
            }
        },
        None => {
            anyhow::bail!("Provide --export <session-id> or a session subcommand");
        }
    }
    
    Ok(())
}

/// Save a session to disk
#[allow(dead_code)]
pub fn save_session(session: &SavedSession) -> Result<()> {
    let dir = get_sessions_dir()?;
    let path = dir.join(format!("{}.json", session.id));
    let content = serde_json::to_string_pretty(session)?;
    std::fs::write(&path, &content)?;
    Ok(())
}

/// Load a session from disk
#[allow(dead_code)]
pub fn load_session(session_id: &str) -> Result<SavedSession> {
    let dir = get_sessions_dir()?;
    let path = dir.join(format!("{}.json", session_id));
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Session not found: {}", session_id))?;
    let session: SavedSession = serde_json::from_str(&content)?;
    Ok(session)
}
