use anyhow::{anyhow, Result};

use crate::cli::UiDemoArgs;

pub async fn run(args: UiDemoArgs) -> Result<()> {
    if let Some(component) = &args.component {
        let valid = ["code", "diff", "tokens", "tools", "progress", "collapsible", "streaming", "all"];
        if !valid.contains(&component.as_str()) {
            return Err(anyhow!(
                "Unknown component: {}. Available: {}",
                component,
                valid.join(", ")
            ));
        }
    }

    let theme = args.theme.as_deref().unwrap_or("default");
    let mode = if args.plain { "plain" } else { "rich" };
    let component = args.component.as_deref().unwrap_or("all");

    println!("Copex UI Demo");
    println!("  Theme: {}", theme);
    println!("  Mode: {}", mode);
    println!("  Component: {}", component);
    println!();
    println!("(UI demo components are not yet implemented in the Rust UI stack.)");
    Ok(())
}
