//! Minimal async JSON-RPC 2.0 client for stdio transport
//!
//! This communicates with the GitHub Copilot CLI via stdin/stdout using JSON-RPC 2.0.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

/// JSON-RPC request
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: String,
    pub method: String,
    pub params: Value,
}

/// JSON-RPC response
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcResponse {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<String>,
    pub result: Option<Value>,
    pub error: Option<JsonRpcError>,
    pub method: Option<String>,
    pub params: Option<Value>,
}

/// JSON-RPC error
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[allow(dead_code)]
    pub data: Option<Value>,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC Error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for JsonRpcError {}

/// Session event from the CLI
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: Option<Value>,
}

/// Message sent to the CLI
enum ClientMessage {
    Request {
        id: String,
        method: String,
        params: Value,
        #[allow(dead_code)]
        response_tx: oneshot::Sender<Result<Value>>,
    },
    Stop,
}

/// JSON-RPC client that spawns and communicates with the Copilot CLI
pub struct JsonRpcClient {
    process: Child,
    #[allow(dead_code)]
    stdin: Arc<Mutex<ChildStdin>>,
    message_tx: mpsc::Sender<ClientMessage>,
    #[allow(dead_code)]
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value>>>>>,
    event_handler: Arc<Mutex<Option<Box<dyn Fn(String, SessionEvent) + Send + Sync>>>>,
}

impl JsonRpcClient {
    /// Spawn the copilot CLI and create a client
    pub fn spawn(cli_path: &str, log_level: &str) -> Result<Self> {
        let mut process = Command::new(cli_path)
            .args(["--server", "--stdio", "--log-level", log_level])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("Failed to spawn copilot CLI: {}", cli_path))?;

        let stdin = process.stdin.take().context("Failed to get stdin")?;
        let stdout = process.stdout.take().context("Failed to get stdout")?;

        let stdin = Arc::new(Mutex::new(stdin));
        let pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value>>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let event_handler: Arc<Mutex<Option<Box<dyn Fn(String, SessionEvent) + Send + Sync>>>> =
            Arc::new(Mutex::new(None));

        let (message_tx, message_rx) = mpsc::channel::<ClientMessage>(100);

        // Spawn reader thread
        let pending_clone = Arc::clone(&pending);
        let event_handler_clone = Arc::clone(&event_handler);
        std::thread::spawn(move || {
            read_loop(stdout, pending_clone, event_handler_clone);
        });

        // Spawn writer task
        let stdin_clone = Arc::clone(&stdin);
        tokio::spawn(async move {
            write_loop(message_rx, stdin_clone).await;
        });

        Ok(Self {
            process,
            stdin,
            message_tx,
            pending,
            event_handler,
        })
    }

    /// Set a handler for session events
    pub fn set_event_handler<F>(&self, handler: F)
    where
        F: Fn(String, SessionEvent) + Send + Sync + 'static,
    {
        let mut guard = self.event_handler.lock().unwrap();
        *guard = Some(Box::new(handler));
    }

    /// Send a JSON-RPC request and wait for response
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = Uuid::new_v4().to_string();
        let (response_tx, response_rx) = oneshot::channel();

        // Register pending request
        {
            let mut pending = self.pending.lock().unwrap();
            pending.insert(id.clone(), response_tx);
        }

        // Send request
        self.message_tx
            .send(ClientMessage::Request {
                id: id.clone(),
                method: method.to_string(),
                params,
                response_tx: {
                    // This is unused now, we use the pending map
                    let (tx, _) = oneshot::channel();
                    tx
                },
            })
            .await
            .map_err(|_| anyhow!("Failed to send request"))?;

        // Wait for response
        match tokio::time::timeout(std::time::Duration::from_secs(120), response_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow!("Response channel closed")),
            Err(_) => {
                // Remove from pending
                self.pending.lock().unwrap().remove(&id);
                Err(anyhow!("Request timed out"))
            }
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    #[allow(dead_code)]
    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let id = Uuid::new_v4().to_string();
        self.message_tx
            .send(ClientMessage::Request {
                id,
                method: method.to_string(),
                params,
                response_tx: {
                    let (tx, _) = oneshot::channel();
                    tx
                },
            })
            .await
            .map_err(|_| anyhow!("Failed to send notification"))?;
        Ok(())
    }

    /// Stop the client and terminate the CLI process
    pub async fn stop(&mut self) -> Result<()> {
        let _ = self.message_tx.send(ClientMessage::Stop).await;
        self.process.kill().ok();
        self.process.wait().ok();
        Ok(())
    }
}

impl Drop for JsonRpcClient {
    fn drop(&mut self) {
        self.process.kill().ok();
    }
}

/// Read messages from the CLI stdout
fn read_loop(
    stdout: ChildStdout,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<Result<Value>>>>>,
    event_handler: Arc<Mutex<Option<Box<dyn Fn(String, SessionEvent) + Send + Sync>>>>,
) {
    let mut reader = BufReader::new(stdout);

    loop {
        // Read Content-Length header
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 {
            break;
        }

        let content_length: usize = header
            .strip_prefix("Content-Length: ")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        if content_length == 0 {
            continue;
        }

        // Read empty line
        let mut empty = String::new();
        reader.read_line(&mut empty).ok();

        // Read content
        let mut content = vec![0u8; content_length];
        if std::io::Read::read_exact(&mut reader, &mut content).is_err() {
            break;
        }

        // Parse JSON
        let message: JsonRpcResponse = match serde_json::from_slice(&content) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to parse JSON-RPC message: {}", e);
                continue;
            }
        };

        // Handle response or notification
        if let Some(id) = &message.id {
            // It's a response
            if let Some(tx) = pending.lock().unwrap().remove(id) {
                let result = if let Some(error) = message.error {
                    Err(anyhow!("{}", error))
                } else if let Some(result) = message.result {
                    Ok(result)
                } else {
                    Ok(Value::Null)
                };
                let _ = tx.send(result);
            }
        } else if let Some(method) = &message.method {
            // It's a notification
            if method == "session.event" {
                if let Some(params) = message.params {
                    let session_id = params
                        .get("sessionId")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if let Some(event_data) = params.get("event") {
                        if let Ok(event) = serde_json::from_value::<SessionEvent>(event_data.clone())
                        {
                            if let Some(handler) = event_handler.lock().unwrap().as_ref() {
                                handler(session_id, event);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Write messages to the CLI stdin
async fn write_loop(mut rx: mpsc::Receiver<ClientMessage>, stdin: Arc<Mutex<ChildStdin>>) {
    while let Some(msg) = rx.recv().await {
        match msg {
            ClientMessage::Request {
                id,
                method,
                params,
                ..
            } => {
                let request = JsonRpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id,
                    method,
                    params,
                };
                let content = serde_json::to_string(&request).unwrap();
                let header = format!("Content-Length: {}\r\n\r\n", content.len());

                let mut stdin = stdin.lock().unwrap();
                stdin.write_all(header.as_bytes()).ok();
                stdin.write_all(content.as_bytes()).ok();
                stdin.flush().ok();
            }
            ClientMessage::Stop => break,
        }
    }
}
