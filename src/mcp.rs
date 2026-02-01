use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: HashMap<String, String>,
}

impl McpServerConfig {
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            command: command.into(),
            args: Vec::new(),
            cwd: None,
            env: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: String,
    #[serde(default)]
    pub result: Option<serde_json::Value>,
    #[serde(default)]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

pub struct McpClient {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
}

pub trait McpToolClient {
    fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<serde_json::Value>;
}

impl McpClient {
    pub fn spawn(config: &McpServerConfig) -> Result<Self> {
        let mut cmd = Command::new(&config.command);
        cmd.args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(cwd) = &config.cwd {
            cmd.current_dir(cwd);
        }
        if !config.env.is_empty() {
            cmd.envs(&config.env);
        }
        let mut child = cmd.spawn().with_context(|| "Failed to spawn MCP server")?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("Missing stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("Missing stdout"))?;

        Ok(Self {
            child,
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(BufReader::new(stdout))),
        })
    }

    pub fn initialize(&self, client_name: &str, client_version: &str) -> Result<()> {
        let params = serde_json::json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": { "tools": {}, "resources": {} },
            "clientInfo": { "name": client_name, "version": client_version },
        });
        let response = self.request("initialize", Some(params))?;
        if response.error.is_some() {
            return Err(anyhow!("MCP initialize failed: {:?}", response.error));
        }
        self.notify("notifications/initialized", None)?;
        Ok(())
    }

    pub fn list_tools(&self) -> Result<Vec<McpTool>> {
        let response = self.request("tools/list", Some(serde_json::json!({})))?;
        if let Some(error) = response.error {
            return Err(anyhow!("tools/list failed: {}", error.message));
        }
        let result = response.result.unwrap_or_default();
        let tools = result
            .get("tools")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        Ok(tools)
    }

    pub fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<serde_json::Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });
        let response = self.request("tools/call", Some(params))?;
        if let Some(error) = response.error {
            return Err(anyhow!("tools/call failed: {}", error.message));
        }
        Ok(response.result.unwrap_or_default())
    }

    pub fn shutdown(&mut self) -> Result<()> {
        let _ = self.notify("shutdown", None);
        let _ = self.child.kill();
        Ok(())
    }

    fn request(&self, method: &str, params: Option<serde_json::Value>) -> Result<JsonRpcResponse> {
        let id = Uuid::new_v4().to_string();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: id.clone(),
            method: method.into(),
            params,
        };
        self.write_message(&request)?;
        self.read_response(&id)
    }

    fn notify(&self, method: &str, params: Option<serde_json::Value>) -> Result<()> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Uuid::new_v4().to_string(),
            method: method.into(),
            params,
        };
        self.write_message(&request)
    }

    fn write_message<T: Serialize>(&self, value: &T) -> Result<()> {
        let payload = serde_json::to_string(value)?;
        let mut stdin = self.stdin.lock().expect("stdin lock");
        stdin
            .write_all(payload.as_bytes())
            .and_then(|_| stdin.write_all(b"\n"))
            .with_context(|| "Failed to write MCP message")?;
        stdin.flush().ok();
        Ok(())
    }

    fn read_response(&self, id: &str) -> Result<JsonRpcResponse> {
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > Duration::from_secs(10) {
                return Err(anyhow!("Timed out waiting for MCP response"));
            }
            let mut line = String::new();
            let mut stdout = self.stdout.lock().expect("stdout lock");
            let bytes = stdout.read_line(&mut line)?;
            if bytes == 0 {
                continue;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let response: JsonRpcResponse = serde_json::from_str(trimmed)
                .with_context(|| format!("Invalid MCP response: {}", trimmed))?;
            if response.id == id {
                return Ok(response);
            }
        }
    }
}

impl McpToolClient for McpClient {
    fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<serde_json::Value> {
        McpClient::call_tool(self, name, arguments)
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpClientInfo {
    #[allow(dead_code)]
    pub name: Option<String>,
    #[allow(dead_code)]
    pub version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpInitializeParams {
    pub protocol_version: String,
    #[allow(dead_code)]
    pub client_info: Option<McpClientInfo>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolCallParams {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

pub fn run_stdio_server() -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let request: JsonRpcRequest = serde_json::from_str(trimmed)
            .with_context(|| format!("Invalid MCP request: {}", trimmed))?;
        let id = request.id.clone();
        let response = match request.method.as_str() {
            "initialize" => {
                let params: McpInitializeParams = request
                    .params
                    .as_ref()
                    .ok_or_else(|| anyhow!("Missing initialize params"))
                    .and_then(|value| serde_json::from_value(value.clone()).map_err(anyhow::Error::from))?;
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "protocolVersion": params.protocol_version,
                        "capabilities": {
                            "tools": {},
                            "resources": {}
                        },
                        "serverInfo": {
                            "name": "copex-mcp",
                            "version": env!("CARGO_PKG_VERSION")
                        }
                    }
                })
            }
            "notifications/initialized" => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {}
                })
            }
            "tools/list" => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [
                            {
                                "name": "read_file",
                                "description": "Read a file from disk",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": { "path": { "type": "string" } },
                                    "required": ["path"]
                                }
                            },
                            {
                                "name": "write_file",
                                "description": "Write a file to disk",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": { "type": "string" },
                                        "content": { "type": "string" }
                                    },
                                    "required": ["path", "content"]
                                }
                            },
                            {
                                "name": "run_command",
                                "description": "Run a shell command",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": { "command": { "type": "string" } },
                                    "required": ["command"]
                                }
                            }
                        ]
                    }
                })
            }
            "tools/call" => {
                let params: McpToolCallParams = request
                    .params
                    .as_ref()
                    .ok_or_else(|| anyhow!("Missing tools/call params"))
                    .and_then(|value| serde_json::from_value(value.clone()).map_err(anyhow::Error::from))?;
                match dispatch_tool_call(&params.name, &params.arguments) {
                    Ok(result) => serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": result.output.to_string()
                                }
                            ],
                            "isError": result.is_error,
                            "output": result.output
                        }
                    }),
                    Err(err) => serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32603,
                            "message": err.to_string()
                        }
                    }),
                }
            }
            _ => serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Method not found: {}", request.method)
                }
            }),
        };
        let payload = serde_json::to_string(&response)?;
        stdout.write_all(payload.as_bytes())?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct ToolDispatchResult {
    pub output: serde_json::Value,
    pub is_error: bool,
}

pub fn dispatch_tool_call(name: &str, arguments: &serde_json::Value) -> Result<ToolDispatchResult> {
    match name {
        "read_file" => {
            let path = arguments
                .get("path")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("read_file requires a path"))?;
            let contents = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read file {}", path))?;
            Ok(ToolDispatchResult {
                output: serde_json::json!({ "content": contents }),
                is_error: false,
            })
        }
        "write_file" => {
            let path = arguments
                .get("path")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("write_file requires a path"))?;
            let content = arguments
                .get("content")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("write_file requires content"))?;
            std::fs::write(path, content)
                .with_context(|| format!("Failed to write file {}", path))?;
            Ok(ToolDispatchResult {
                output: serde_json::json!({ "success": true }),
                is_error: false,
            })
        }
        "run_command" => {
            let command = arguments
                .get("command")
                .and_then(|value| value.as_str())
                .ok_or_else(|| anyhow!("run_command requires command"))?;
            let output = Command::new("sh")
                .arg("-c")
                .arg(command)
                .output()
                .with_context(|| format!("Failed to run command {}", command))?;
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            Ok(ToolDispatchResult {
                output: serde_json::json!({
                    "stdout": stdout,
                    "stderr": stderr,
                    "status": output.status.code()
                }),
                is_error: !output.status.success(),
            })
        }
        _ => Err(anyhow!("Unsupported tool: {}", name)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn mcp_round_trip_initialize() {
        let temp = tempfile::tempdir().expect("tempdir");
        let server_path = temp.path().join("server.py");
        let script = r#"
import json
import sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    payload = json.loads(line)
    method = payload.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": payload["id"], "result": {"capabilities": {}}})
    elif method == "notifications/initialized":
        send({"jsonrpc": "2.0", "id": payload["id"], "result": {}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}})
"#;
        let mut file = std::fs::File::create(&server_path).expect("create");
        file.write_all(script.as_bytes()).expect("write");
        let mut config = McpServerConfig::new("test", "python3");
        config.args.push(server_path.to_string_lossy().to_string());
        let client = McpClient::spawn(&config).expect("spawn");
        client.initialize("copex", "0.1.0").expect("init");
        let tools = client.list_tools().expect("list");
        assert!(tools.is_empty());
    }
}
