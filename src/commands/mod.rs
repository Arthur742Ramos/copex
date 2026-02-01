use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use futures::StreamExt;

use crate::config::Config;
use serde_json::{json, Value};
use crate::copilot_client::SessionHandle;
use crate::mcp::McpToolClient;

use crate::copilot_client::{ChunkType, Client, StreamChunk};

pub mod auth;
pub mod chat;
pub mod init;
pub mod interactive;
pub mod models;
pub mod plan;
pub mod ralph;
pub mod session;
pub mod themes;
pub mod ui_demo;

pub fn load_config(path: Option<PathBuf>) -> Result<Config> {
    match path {
        Some(path) => Config::load_from_file(path),
        None => Config::load(),
    }
}

pub fn build_client(config: &Config) -> Result<Client> {
    Client::new(config.clone())
}

pub fn build_client_with_mcp(config: &Config, mcp_config: Option<Value>) -> Result<Client> {
    let client = Client::new(config.clone())?;
    Ok(client.with_mcp_config(mcp_config))
}

pub fn load_mcp_config(path: Option<std::path::PathBuf>) -> Result<Option<serde_json::Value>> {
    let path = match path {
        Some(path) => path,
        None => return Ok(None),
    };
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read MCP config {}", path.display()))?;
    let json = if content.trim_start().starts_with('{') {
        serde_json::from_str(&content)?
    } else {
        serde_json::from_str(&content)?
    };
    Ok(Some(json))
}

#[derive(Debug, Clone)]
pub struct ToolCallRequest {
    pub name: String,
    pub arguments: Value,
    pub call_id: Option<String>,
}

pub fn parse_tool_call(chunk: &StreamChunk) -> Option<ToolCallRequest> {
    if chunk.chunk_type != ChunkType::ToolCall {
        return None;
    }
    let data = chunk.tool_data.as_ref()?;
    if let Some(event_type) = data
        .get("eventType")
        .or_else(|| data.get("type"))
        .and_then(|v| v.as_str())
    {
        if event_type != "tool.call" {
            return None;
        }
    }
    let name = data
        .get("name")
        .or_else(|| data.get("tool"))
        .or_else(|| data.get("toolName"))
        .and_then(|v| v.as_str())?
        .to_string();
    let call_id = data
        .get("callId")
        .or_else(|| data.get("toolCallId"))
        .or_else(|| data.get("id"))
        .and_then(|v| v.as_str())
        .map(|value| value.to_string());
    let mut arguments = data
        .get("arguments")
        .or_else(|| data.get("args"))
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    if let Some(arg_string) = arguments.as_str() {
        if let Ok(parsed) = serde_json::from_str::<Value>(arg_string) {
            arguments = parsed;
        }
    }
    if arguments.is_null() {
        arguments = Value::Object(Default::default());
    }
    Some(ToolCallRequest {
        name,
        arguments,
        call_id,
    })
}

pub async fn execute_tool_call(
    session: &SessionHandle,
    tool_call: &ToolCallRequest,
    mcp_client: &dyn McpToolClient,
) -> Result<()> {
    let call_id = tool_call
        .call_id
        .as_deref()
        .ok_or_else(|| anyhow!("Tool call missing id"))?;
    if tool_call.name == "report_intent" {
        session
            .send_tool_call_update(call_id, "success", Some(json!({ "success": true })), None)
            .await?;
        return Ok(());
    }

    let mut tool_name = tool_call.name.as_str();
    let mut arguments = tool_call.arguments.clone();
    if arguments.is_string() {
        let value = arguments
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_default();
        arguments = match tool_name {
            "view" | "read_file" => json!({ "path": value }),
            "create" | "write_file" => json!({ "content": value }),
            "bash" | "shell" | "run_command" => json!({ "command": value }),
            _ => json!({}),
        };
    }

    match tool_name {
        "view" => tool_name = "read_file",
        "create" => tool_name = "write_file",
        "bash" | "shell" => tool_name = "run_command",
        _ => {}
    }

    if tool_name == "write_file" {
        if arguments.get("content").is_none() {
            let file_text = arguments
                .get("file_text")
                .or_else(|| arguments.get("text"))
                .cloned();
            if let Some(value) = file_text {
                match arguments.as_object_mut() {
                    Some(map) => {
                        map.entry("content".to_string()).or_insert(value);
                    }
                    None => {
                        arguments = json!({ "content": value });
                    }
                }
            }
        }
    }

    if tool_name == "run_command" && arguments.get("command").is_none() {
        if let Some(cmd) = arguments.get("cmd").cloned() {
            match arguments.as_object_mut() {
                Some(map) => {
                    map.entry("command".to_string()).or_insert(cmd);
                }
                None => {
                    arguments = json!({ "command": cmd });
                }
            }
        }
    }

    let result = mcp_client.call_tool(tool_name, arguments);

    match result {
        Ok(raw_output) => {
            session
                .send_tool_call_update(call_id, "success", Some(raw_output), None)
                .await?;
        }
        Err(err) => {
            let message = err.to_string();
            session
                .send_tool_call_update(
                    call_id,
                    "error",
                    Some(json!({ "error": message })),
                    Some(message),
                )
                .await?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub async fn collect_stream<S>(stream: S) -> Result<String>
where
    S: futures::Stream<Item = Result<StreamChunk>>,
{
    let mut stream = Box::pin(stream);
    let mut output = String::new();
    while let Some(chunk) = stream.as_mut().next().await {
        let chunk = chunk?;
        output.push_str(&chunk.delta);
        if chunk.is_final {
            break;
        }
    }
    Ok(output)
}

pub async fn stream_to_stdout<S>(stream: S, show_reasoning: bool) -> Result<()>
where
    S: futures::Stream<Item = Result<StreamChunk>>,
{
    use std::io::Write;

    let mut stream = Box::pin(stream);
    let mut stdout = std::io::stdout();
    while let Some(chunk) = stream.as_mut().next().await {
        let chunk = chunk?;
        if !show_reasoning && chunk.chunk_type == ChunkType::Reasoning {
            if chunk.is_final {
                break;
            }
            continue;
        }
        if !chunk.delta.is_empty() {
            write!(stdout, "{}", chunk.delta)?;
            stdout.flush().ok();
        }
        if chunk.is_final {
            break;
        }
    }
    writeln!(stdout)?;
    Ok(())
}
