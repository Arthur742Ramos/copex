use anyhow::{anyhow, Result};

use crate::cli::ChatArgs;

use super::{build_client, load_config, stream_to_stdout};
use crate::ui::run_stream_ui;

pub async fn run(args: ChatArgs) -> Result<()> {
    let mut config = load_config(args.config)?;
    if args.no_color || args.plain {
        config.ui_ascii_icons = true;
    }
    if let Some(model) = args.model {
        config.model = model;
    }
    if let Some(reasoning) = args.reasoning {
        config.reasoning_effort = reasoning;
    }
    if args.no_stream {
        config.streaming = false;
    }
    if let Some(theme) = args.ui_theme {
        config.ui_theme = theme;
    }
    if let Some(density) = args.ui_density {
        config.ui_density = density;
    }
    let show_reasoning = if args.no_reasoning {
        false
    } else {
        args.show_reasoning
    };

    let prompt = args
        .prompt
        .ok_or_else(|| anyhow!("Missing prompt. Provide prompt or use interactive mode."))?;
    
    let client = build_client(&config)?;
    let mut session = client.create_session().await?;
    let stream = session.send(&prompt).await?;
    
    // Check if we're in a TTY and not in raw mode
    let use_tui =
        config.streaming && !args.raw && !args.plain && atty::is(atty::Stream::Stdout);
    
    if use_tui {
        // Try TUI, fallback to raw on error
        match run_stream_ui(
            stream,
            Some(&config.ui_theme),
            config.ui_ascii_icons,
            show_reasoning,
        )
        .await {
            Ok(_) => Ok(()),
            Err(_) => {
                // TUI failed, retry with raw output
                let mut session = client.create_session().await?;
                let stream = session.send(&prompt).await?;
                stream_to_stdout(stream, show_reasoning).await
            }
        }
    } else {
        stream_to_stdout(stream, show_reasoning).await
    }
}
