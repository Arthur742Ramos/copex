use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::config::Config;
use crate::jsonrpc::{JsonRpcClient, SessionEvent};

const TOOL_PREFIX: &str = "🔧";

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub chunk_type: ChunkType,
    pub delta: String,
    pub is_final: bool,
    pub tool_data: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Message,
    Reasoning,
    ToolCall,
    ToolResult,
    #[allow(dead_code)]
    System,
}

pub struct Session {
    pub session_id: String,
    client: Arc<Mutex<JsonRpcClient>>,
    event_rx: mpsc::Receiver<SessionEvent>,
}

#[derive(Clone)]
pub struct SessionHandle {
    #[allow(dead_code)]
    pub session_id: String,
    #[allow(dead_code)]
    pub client: Arc<Mutex<JsonRpcClient>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub version: String,
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthStatus {
    #[serde(rename = "isAuthenticated")]
    pub is_authenticated: bool,
    pub login: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub supports_reasoning: bool,
}

pub struct Client {
    config: Config,
    rpc: Arc<Mutex<JsonRpcClient>>,
    session_events: Arc<Mutex<HashMap<String, mpsc::Sender<SessionEvent>>>>,
    mcp_config: Option<serde_json::Value>,
}

impl Client {
    pub fn new(config: Config) -> Result<Self> {
        let cli_path = resolve_copilot_cli()?;
        let log_level = std::env::var("COPEX_COPILOT_LOG_LEVEL").unwrap_or_else(|_| "error".to_string());
        let rpc = JsonRpcClient::spawn(&cli_path, &log_level)?;

        let session_events: Arc<Mutex<HashMap<String, mpsc::Sender<SessionEvent>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let session_events_clone = Arc::clone(&session_events);

        rpc.set_event_handler(move |session_id, event| {
            if let Some(sender) = session_events_clone.lock().unwrap().get(&session_id) {
                let _ = sender.try_send(event);
            }
        });

        Ok(Self {
            config,
            rpc: Arc::new(Mutex::new(rpc)),
            session_events,
            mcp_config: None,
        })
    }

    pub fn with_mcp_config(mut self, mcp_config: Option<serde_json::Value>) -> Self {
        self.mcp_config = mcp_config;
        self
    }

    pub async fn get_status(&self) -> Result<StatusResponse> {
        let result = self
            .rpc
            .lock()
            .unwrap()
            .request("status.get", json!({}))
            .await?;
        let status: StatusResponse = serde_json::from_value(result)?;
        Ok(status)
    }

    pub async fn get_auth_status(&self) -> Result<AuthStatus> {
        let result = self
            .rpc
            .lock()
            .unwrap()
            .request("auth.getStatus", json!({}))
            .await?;
        let auth: AuthStatus = serde_json::from_value(result)?;
        Ok(auth)
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let result = self
            .rpc
            .lock()
            .unwrap()
            .request("models.list", json!({}))
            .await?;
        let models = result
            .get("models")
            .and_then(|value| value.as_array())
            .ok_or_else(|| anyhow!("Invalid models.list response"))?;
        let mut parsed = Vec::new();
        for model in models {
            let id = model
                .get("id")
                .and_then(|value| value.as_str())
                .unwrap_or_default()
                .to_string();
            if id.is_empty() {
                continue;
            }
            let name = model
                .get("name")
                .and_then(|value| value.as_str())
                .unwrap_or(&id)
                .to_string();
            let supports_reasoning = model
                .get("capabilities")
                .and_then(|value| value.get("supports"))
                .and_then(|value| value.get("reasoningEffort"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            parsed.push(ModelInfo {
                id,
                name,
                version: String::new(),
                supports_reasoning,
            });
        }
        if parsed.is_empty() {
            parse_models_from_help().await
        } else {
            Ok(parsed)
        }
    }

    pub async fn create_session(&self) -> Result<Session> {
        let session_params = self.build_session_params()?;
        let result = self
            .rpc
            .lock()
            .unwrap()
            .request("session.create", session_params)
            .await?;
        let session_id = result
            .get("sessionId")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow!("Missing sessionId"))?
            .to_string();
        let (tx, rx) = mpsc::channel(1024);
        self.session_events
            .lock()
            .unwrap()
            .insert(session_id.clone(), tx);
        Ok(Session {
            session_id,
            client: Arc::clone(&self.rpc),
            event_rx: rx,
        })
    }

    pub async fn chat(
        &self,
        prompt: &str,
    ) -> Result<impl Stream<Item = Result<StreamChunk>> + 'static> {
        let prompt = prompt.to_string();
        let client = Arc::clone(&self.rpc);
        let session_events = Arc::clone(&self.session_events);
        let session_params = self.build_session_params()?;
        let stream = try_stream! {
            let result = {
                let rpc = client.lock().unwrap();
                rpc.request("session.create", session_params).await?
            };
            let session_id = result
                .get("sessionId")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("Missing sessionId"))?
                .to_string();
            let (tx, rx) = mpsc::channel(1024);
            session_events
                .lock()
                .unwrap()
                .insert(session_id.clone(), tx);
            let response = {
                let rpc = client.lock().unwrap();
                rpc.request(
                    "session.send",
                    json!({
                        "sessionId": session_id,
                        "prompt": prompt,
                    }),
                )
                .await?
            };
            let message_id = response
                .get("messageId")
                .and_then(|value| value.as_str())
                .map(|value| value.to_string());
            let mut stream = Box::pin(build_stream(rx, message_id));
            while let Some(chunk) = stream.as_mut().next().await {
                yield chunk?;
            }
        };
        Ok(stream)
    }

    fn build_session_params(&self) -> Result<Value> {
        let mut params = serde_json::Map::new();
        params.insert("streaming".to_string(), Value::Bool(self.config.streaming));
        params.insert("model".to_string(), Value::String(self.config.model.clone()));
        if !self.config.reasoning_effort.is_empty()
            && self.config.reasoning_effort != "none"
        {
            params.insert(
                "modelReasoningEffort".to_string(),
                Value::String(self.config.reasoning_effort.clone()),
            );
        }
        params.insert(
            "workingDirectory".to_string(),
            Value::String(std::env::current_dir()?.display().to_string()),
        );
        if let Some(mcp_config) = &self.mcp_config {
            params.insert("mcpServers".to_string(), mcp_config.clone());
        }
        Ok(Value::Object(params))
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if let Ok(mut rpc) = self.rpc.lock() {
            let _ = futures::executor::block_on(rpc.stop());
        }
    }
}

impl Session {
    pub fn handle(&self) -> SessionHandle {
        SessionHandle {
            session_id: self.session_id.clone(),
            client: Arc::clone(&self.client),
        }
    }

    pub async fn send(
        &mut self,
        prompt: &str,
    ) -> Result<impl Stream<Item = Result<StreamChunk>> + Send + 'static> {
        let params = json!({
            "sessionId": self.session_id,
            "prompt": prompt,
        });
        let response = self
            .client
            .lock()
            .unwrap()
            .request("session.send", params)
            .await?;
        let message_id = response
            .get("messageId")
            .and_then(|value| value.as_str())
            .map(|value| value.to_string());
        let rx = std::mem::replace(&mut self.event_rx, mpsc::channel(1).1);
        Ok(build_stream(rx, message_id))
    }
}

impl SessionHandle {
    pub async fn send_tool_call_update(
        &self,
        _call_id: &str,
        _status: &str,
        _result: Option<Value>,
        _content_text: Option<String>,
    ) -> Result<()> {
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn destroy(&self) -> Result<()> {
        let payload = json!({"sessionId": self.session_id});
        self.client
            .lock()
            .unwrap()
            .request("session.destroy", payload)
            .await?;
        Ok(())
    }
}

fn build_stream(
    rx: mpsc::Receiver<SessionEvent>,
    message_id: Option<String>,
) -> impl Stream<Item = Result<StreamChunk>> + 'static {
    try_stream! {
        let mut rx = rx;
        let mut saw_message_delta = false;
        let mut saw_reasoning_delta = false;
        let mut final_message = None;
        let mut final_reasoning = None;
        let mut seen_tool_calls: HashSet<String> = HashSet::new();

        while let Some(event) = rx.recv().await {
            let event_type = event.event_type.as_str();
            let data = event.data.clone().unwrap_or_else(|| json!({}));

            match event_type {
                "assistant.message_delta" => {
                    if !matches_message_id(&data, &message_id) {
                        continue;
                    }
                    let delta = data
                        .get("deltaContent")
                        .or_else(|| data.get("content"))
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .to_string();
                    saw_message_delta = true;
                    if !delta.is_empty() {
                        yield StreamChunk {
                            chunk_type: ChunkType::Message,
                            delta,
                            is_final: false,
                            tool_data: None,
                        };
                    }
                }
                "assistant.reasoning_delta" => {
                    let delta = data
                        .get("deltaContent")
                        .or_else(|| data.get("content"))
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .to_string();
                    saw_reasoning_delta = true;
                    if !delta.is_empty() {
                        yield StreamChunk {
                            chunk_type: ChunkType::Reasoning,
                            delta,
                            is_final: false,
                            tool_data: None,
                        };
                    }
                }
                "assistant.message" => {
                    if !matches_message_id(&data, &message_id) {
                        continue;
                    }
                    let content = data
                        .get("content")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !content.is_empty() {
                        final_message = Some(content.clone());
                    }
                    if let Some(requests) = data.get("toolRequests") {
                        if let Some(array) = requests.as_array() {
                            for request in array {
                                if let Some(chunk) = build_tool_call_chunk(request, &mut seen_tool_calls) {
                                    yield chunk;
                                }
                            }
                        }
                    }
                }
                "assistant.reasoning" => {
                    let content = data
                        .get("content")
                        .and_then(|value| value.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !content.is_empty() {
                        final_reasoning = Some(content);
                    }
                }
                "tool.execution_start" => {
                    if let Some(chunk) = build_tool_execution_chunk(&data, &mut seen_tool_calls) {
                        yield chunk;
                    }
                }
                "tool.execution_partial_result" | "tool.execution_complete" => {
                    let tool_name = data.get("toolName").and_then(|v| v.as_str()).unwrap_or("tool");
                    let payload = json!({
                        "eventType": event_type,
                        "toolCallId": data.get("toolCallId"),
                        "toolName": data.get("toolName"),
                        "success": data.get("success"),
                        "result": data.get("result"),
                    });
                    yield StreamChunk {
                        chunk_type: ChunkType::ToolResult,
                        delta: format!("{} {} result", TOOL_PREFIX, tool_name),
                        is_final: false,
                        tool_data: Some(payload),
                    };
                }
                "assistant.turn_end" | "session.idle" => {
                    if !saw_reasoning_delta {
                        if let Some(content) = final_reasoning.take() {
                            yield StreamChunk {
                                chunk_type: ChunkType::Reasoning,
                                delta: content,
                                is_final: false,
                                tool_data: None,
                            };
                        }
                    }
                    if !saw_message_delta {
                        if let Some(content) = final_message.take() {
                            yield StreamChunk {
                                chunk_type: ChunkType::Message,
                                delta: content,
                                is_final: false,
                                tool_data: None,
                            };
                        }
                    }
                    yield StreamChunk {
                        chunk_type: ChunkType::Message,
                        delta: String::new(),
                        is_final: true,
                        tool_data: None,
                    };
                    break;
                }
                "session.error" | "error" => {
                    let message = data
                        .get("message")
                        .and_then(|value| value.as_str())
                        .unwrap_or("Session error")
                        .to_string();
                    Err(anyhow!(message))?;
                }
                _ => {}
            }
        }
    }
}

fn build_tool_call_chunk(
    request: &Value,
    seen: &mut HashSet<String>,
) -> Option<StreamChunk> {
    let name = request.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
    let call_id = request
        .get("toolCallId")
        .or_else(|| request.get("callId"))
        .or_else(|| request.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if !call_id.is_empty() && !seen.insert(call_id.clone()) {
        return None;
    }
    let arguments = request.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let payload = json!({
        "eventType": "tool.call",
        "name": name,
        "arguments": arguments,
        "callId": call_id,
        "toolCallId": call_id,
    });
    Some(StreamChunk {
        chunk_type: ChunkType::ToolCall,
        delta: format!("{} {}", TOOL_PREFIX, name),
        is_final: false,
        tool_data: Some(payload),
    })
}

fn build_tool_execution_chunk(
    data: &Value,
    seen: &mut HashSet<String>,
) -> Option<StreamChunk> {
    let tool_name = data.get("toolName").and_then(|v| v.as_str()).unwrap_or("tool");
    let call_id = data
        .get("toolCallId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if !call_id.is_empty() && !seen.insert(call_id.clone()) {
        return None;
    }
    let payload = json!({
        "eventType": "tool.call",
        "name": tool_name,
        "arguments": data.get("arguments").cloned().unwrap_or_else(|| json!({})),
        "callId": call_id,
        "toolCallId": call_id,
    });
    Some(StreamChunk {
        chunk_type: ChunkType::ToolCall,
        delta: format!("{} {}", TOOL_PREFIX, tool_name),
        is_final: false,
        tool_data: Some(payload),
    })
}

fn matches_message_id(data: &Value, message_id: &Option<String>) -> bool {
    if let Some(expected) = message_id {
        if let Some(actual) = data.get("messageId").and_then(|value| value.as_str()) {
            return actual == expected;
        }
    }
    true
}

fn resolve_copilot_cli() -> Result<String> {
    if let Ok(path) = std::env::var("COPEX_COPILOT_CLI") {
        if !path.trim().is_empty() {
            return Ok(path);
        }
    }
    if let Ok(path) = std::env::var("COPILOT_CLI_PATH") {
        if !path.trim().is_empty() {
            return Ok(path);
        }
    }
    if let Ok(path) = std::env::var("COPILOT_PATH") {
        if !path.trim().is_empty() {
            return Ok(path);
        }
    }
    if let Some(path) = which::which("copilot").ok() {
        return Ok(path.display().to_string());
    }
    let candidates = [
        "/opt/homebrew/bin/copilot",
        "/usr/local/bin/copilot",
        "/usr/bin/copilot",
    ];
    for candidate in candidates {
        if std::path::Path::new(candidate).exists() {
            return Ok(candidate.to_string());
        }
    }
    Err(anyhow!(
        "Copilot CLI not found; install GitHub Copilot CLI or set COPEX_COPILOT_CLI"
    ))
}

async fn parse_models_from_help() -> Result<Vec<ModelInfo>> {
    let output = tokio::process::Command::new("copilot")
        .arg("--help")
        .output()
        .await?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        let message = if stderr.trim().is_empty() {
            stdout.trim().to_string()
        } else {
            format!("{}\n{}", stdout.trim(), stderr.trim())
        };
        return Err(anyhow!("copilot --help failed: {}", message.trim()));
    }
    let regex = regex::Regex::new(r#""([^"]+)""#).expect("regex");
    let name_pattern = regex::Regex::new(r"^[a-z0-9]+(?:-[a-z0-9.]+)+$").expect("regex");
    let mut seen = HashSet::new();
    let mut models = Vec::new();
    for caps in regex.captures_iter(&stdout) {
        if let Some(value) = caps.get(1).map(|m| m.as_str()) {
            let value = value.trim();
            if name_pattern.is_match(value) && seen.insert(value.to_string()) {
                models.push(ModelInfo {
                    id: value.to_string(),
                    name: value.to_string(),
                    version: String::new(),
                    supports_reasoning: true,
                });
            }
        }
    }
    if models.is_empty() {
        Err(anyhow!("Failed to parse models from copilot --help"))
    } else {
        Ok(models)
    }
}
