use anyhow::{anyhow, Result};
use futures::StreamExt;

use crate::cli::PlanArgs;
use crate::plan::{Plan, PlanExecutor, PlanState};
use crate::commands::{
    build_client_with_mcp,
    execute_tool_call,
    load_config,
    load_mcp_config,
    parse_tool_call,
};
use crate::mcp::{dispatch_tool_call, McpServerConfig, McpToolClient, ToolDispatchResult};
use serde_json::{json, Value};
use crate::copilot_client::{ChunkType, StreamChunk};

pub async fn run(args: PlanArgs) -> Result<()> {
    let mut config = load_config(None)?;
    if let Some(model) = args.model {
        config.model = model;
    }
    if let Some(reasoning) = args.reasoning {
        config.reasoning_effort = reasoning;
    }

    if args.task.is_none() && !args.resume && args.load.is_none() {
        return Err(anyhow!("Provide a task, --resume, or --load"));
    }
    let mcp_config = load_mcp_config(args.mcp_config.clone())?;
    let client = build_client_with_mcp(&config, mcp_config)?;
    let mcp_client = spawn_mcp_client().unwrap_or_else(|_| LocalMcpClient::fallback());

    if args.resume && args.load.is_none() {
        let state = PlanState::load(None)?
            .ok_or_else(|| anyhow!("No saved plan state found for --resume"))?;
        let mut plan = state.plan;
        let max_iterations = if args.max_iterations == 0 {
            10
        } else {
            args.max_iterations
        };
        let executor = PlanExecutor::new_with_max(max_iterations);
        let from_step = std::cmp::max(args.from_step, state.current_step);
        execute_plan(&executor, &client, &mcp_client, &mut plan, &state.task, from_step).await?;
        return Ok(());
    }
    let task = args.task.clone().unwrap_or_default();

    if let Some(load_path) = args.load.clone() {
        let mut plan = Plan::load(load_path)?;
        if let Some(output_path) = args.output.clone() {
            plan.save(output_path)?;
        }
        if args.review && !confirm_execute(&plan)? {
            return Ok(());
        }
        if args.execute || args.review {
            let max_iterations = if args.max_iterations == 0 {
                10
            } else {
                args.max_iterations
            };
            let executor = PlanExecutor::new_with_max(max_iterations);
            let task = plan.task.clone();
            execute_plan(&executor, &client, &mcp_client, &mut plan, &task, args.from_step)
                .await?;
        } else {
            println!("{}", format_plan(&plan));
        }
        return Ok(());
    }

    let mut session = client.create_session().await?;

    // Build the planning prompt
    let prompt = if args.execute || args.review {
        if args.review || args.resume || args.from_step > 1 || args.load.is_some() {
            format!(
                "I need you to plan and execute the following task step by step:\n\n{}\n\n\
                Resume from step {} if prior work exists. If a plan is provided, continue execution.\n\n\
                For each step:\n\
                1. Explain what you're about to do\n\
                2. Execute the step using available tools\n\
                3. Verify the result before moving on\n\n\
                If any step fails, try to fix it before moving to the next step.\n\
                Continue until the task is complete.",
                task,
                args.from_step
            )
        } else {
            format!(
                "I need you to plan and execute the following task step by step:\n\n{}\n\n\
                For each step:\n\
                1. Explain what you're about to do\n\
                2. Execute the step using available tools\n\
                3. Verify the result before moving on\n\n\
                If any step fails, try to fix it before moving to the next step.\n\
                Continue until the task is complete.",
                task
            )
        }
    } else {
        format!(
            "I need you to create a detailed plan for the following task:\n\n{}\n\n\
            Please provide:\n\
            1. A numbered list of steps\n\
            2. What tools/actions each step requires\n\
            3. Expected outcomes for each step\n\
            4. Potential issues and how to handle them\n\n\
            Do NOT execute the plan yet - just create it.",
            task
        )
    };

    let stream = session.send(&prompt).await?;
    let plan_text = collect_message(stream).await?;
    let mut plan = Plan::new(task.clone(), PlanExecutor::new().parse_steps(&plan_text));
    if let Some(output_path) = args.output.clone() {
        plan.save(output_path)?;
    }
    let should_execute = args.execute || args.review;
    if args.review && !confirm_execute(&plan)? {
        return Ok(());
    }
    if should_execute {
        let max_iterations = if args.max_iterations == 0 {
            10
        } else {
            args.max_iterations
        };
        let executor = PlanExecutor::new_with_max(max_iterations);
        execute_plan(&executor, &client, &mcp_client, &mut plan, &task, args.from_step).await?;
    } else {
        println!("{}", plan_text);
    }
    Ok(())
}

async fn collect_message<S>(stream: S) -> Result<String>
where
    S: futures::Stream<Item = Result<StreamChunk>>,
{
    let mut stream = Box::pin(stream);
    let mut output = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if chunk.chunk_type == ChunkType::Message && !chunk.delta.is_empty() {
            output.push_str(&chunk.delta);
        }
        if chunk.is_final {
            break;
        }
    }
    Ok(output)
}

async fn execute_plan(
    executor: &PlanExecutor,
    client: &crate::copilot_client::Client,
    mcp_client: &dyn McpToolClient,
    plan: &mut Plan,
    task: &str,
    from_step: usize,
) -> Result<()> {
    let step_template = format!(
        "You are executing a plan step.\n\nOVERALL TASK: {}\n\nCURRENT STEP:",
        task
    );
    executor.execute_plan(
        plan,
        |step| {
            let prompt = format!("{}\n{}", step_template, step.description);
            let response = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let mut session = client.create_session().await?;
                    let session_handle = session.handle();
                    let stream = session.send(&prompt).await?;
                    let mut stream = Box::pin(stream);
                    let mut output = String::new();
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk?;
                        if let Some(tool_call) = parse_tool_call(&chunk) {
                            execute_tool_call(&session_handle, &tool_call, mcp_client).await?;
                        }
                        if chunk.chunk_type == ChunkType::Message && !chunk.delta.is_empty() {
                            output.push_str(&chunk.delta);
                        }
                        if chunk.is_final {
                            break;
                        }
                    }
                    Ok::<_, anyhow::Error>(output)
                })
            })?;
            Ok((response, 0))
        },
        from_step,
        None,
    )
}

fn format_plan(plan: &Plan) -> String {
    let mut lines = vec![format!("Plan: {}", plan.task)];
    for step in &plan.steps {
        lines.push(format!("Step {}: {}", step.number, step.description));
    }
    lines.join("\n")
}

fn confirm_execute(plan: &Plan) -> Result<bool> {
    println!("{}", format_plan(plan));
    println!();
    print!("Execute this plan? [y/N]: ");
    std::io::Write::flush(&mut std::io::stdout())?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes"))
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
