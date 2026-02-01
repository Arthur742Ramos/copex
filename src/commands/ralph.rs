use anyhow::Result;
use futures::StreamExt;

use crate::cli::RalphArgs;
use crate::commands::{
    build_client_with_mcp,
    execute_tool_call,
    load_config,
    load_mcp_config,
    parse_tool_call,
};
use crate::mcp::{dispatch_tool_call, McpServerConfig, McpToolClient, ToolDispatchResult};
use serde_json::{json, Value};
use crate::copilot_client::ChunkType;

pub async fn run(args: RalphArgs) -> Result<()> {
    let mut config = load_config(None)?;
    if let Some(model) = args.model {
        config.model = model;
    }
    if let Some(reasoning) = args.reasoning {
        config.reasoning_effort = reasoning;
    }

    let mcp_config = load_mcp_config(args.mcp_config.clone())?;
    let client = build_client_with_mcp(&config, mcp_config)?;
    let mcp_client = spawn_mcp_client().unwrap_or_else(|_| LocalMcpClient::fallback());

    // Ralph Wiggum mode: iterative loop until success
    let base_prompt = args.prompt.clone();
    let promise = args.promise.clone();

    for iteration in 1..=args.max_iterations {
        eprintln!("\n[Ralph Wiggum Loop - Iteration {}]", iteration);
        eprintln!("Max iterations: {}", args.max_iterations);
        if let Some(ref p) = promise {
            eprintln!("\nTo complete this loop, output: <promise>{}</promise>", p);
        } else {
            eprintln!("\nNo completion promise set - loop runs until max iterations or cancelled.");
        }
        eprintln!("\nYou can see your previous work in the conversation. Continue improving.\n");

        let prompt = if let Some(ref p) = promise {
            format!(
                "[Ralph Wiggum Loop - Iteration {}]\nMax iterations: {}\n\nTo complete this loop, output: <promise>{}</promise>\nONLY output this when the statement is genuinely TRUE.\n\nYou can see your previous work in the conversation. Continue improving.\n\n{}",
                iteration,
                args.max_iterations,
                p,
                base_prompt
            )
        } else {
            format!(
                "[Ralph Wiggum Loop - Iteration {}]\nMax iterations: {}\n\nNo completion promise set - loop runs until max iterations or cancelled.\n\nYou can see your previous work in the conversation. Continue improving.\n\n{}",
                iteration,
                args.max_iterations,
                base_prompt
            )
        };
        
        let mut session = client.create_session().await?;
        let session_handle = session.handle();
        let stream = session.send(&prompt).await?;
        let mut stream = Box::pin(stream);

        use std::io::Write;
        let mut stdout = std::io::stdout();
        let mut response = String::new();
        
        let mut printed_any = false;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            match chunk.chunk_type {
                ChunkType::Message | ChunkType::Reasoning => {
                    if !chunk.delta.is_empty() {
                        printed_any = true;
                        write!(stdout, "{}", chunk.delta)?;
                        stdout.flush().ok();
                        response.push_str(&chunk.delta);
                    }
                }
                ChunkType::ToolCall => {
                    printed_any = true;
                    writeln!(stdout, "\n{}", chunk.delta)?;
                    if let Some(tool_call) = parse_tool_call(&chunk) {
                        writeln!(stdout, "tool call: {} {}", tool_call.name, tool_call.arguments)?;
                        execute_tool_call(&session_handle, &tool_call, &mcp_client).await?;
                    }
                }
                ChunkType::ToolResult => {
                    printed_any = true;
                    writeln!(stdout, "{}", chunk.delta)?;
                }
                ChunkType::System => {
                    printed_any = true;
                    writeln!(stdout, "[{}]", chunk.delta)?;
                }
            }
            if chunk.is_final {
                break;
            }
        }
        if printed_any {
            writeln!(stdout)?;
        }

        // Check if task seems complete
        if let Some(ref p) = promise {
            let response_lower = response.to_lowercase();
            let promise_lower = p.to_lowercase();
            if response_lower.contains(&format!("<promise>{}</promise>", promise_lower)) {
                eprintln!("\n✅ Ralph completed the task!");
                return Ok(());
            }
        } else {
            let response_lower = response.to_lowercase();
            if response_lower.contains("task complete")
                || response_lower.contains("successfully completed")
                || response_lower.contains("all done")
                || response_lower.contains("finished successfully")
            {
            eprintln!("\n✅ Ralph completed the task!");
            return Ok(());
        }
        }
    }

    eprintln!("\n⚠️ Ralph reached max iterations without clear success");
    Ok(())
}

enum LocalMcpClient {
    External(crate::mcp::McpClient),
    Local,
}

impl LocalMcpClient {
    fn fallback() -> Self {
        LocalMcpClient::Local
    }
}

impl McpToolClient for LocalMcpClient {
    fn call_tool(&self, name: &str, arguments: Value) -> Result<Value> {
        match self {
            LocalMcpClient::External(client) => client.call_tool(name, arguments),
            LocalMcpClient::Local => match dispatch_tool_call(name, &arguments) {
                Ok(result) => Ok(adapt_tool_output(result)),
                Err(err) => Err(err),
            },
        }
    }
}

fn adapt_tool_output(result: ToolDispatchResult) -> Value {
    json!({
        "content": [{ "type": "text", "text": result.output.to_string() }],
        "isError": result.is_error,
        "output": result.output
    })
}

fn spawn_mcp_client() -> Result<LocalMcpClient> {
    let mut config = McpServerConfig::new(
        "copex-mcp",
        std::env::current_exe()?.display().to_string(),
    );
    config.env.insert("COPEX_MCP_SERVER".to_string(), "1".to_string());
    let client = crate::mcp::McpClient::spawn(&config)?;
    client.initialize("copex", env!("CARGO_PKG_VERSION"))?;
    Ok(LocalMcpClient::External(client))
}
