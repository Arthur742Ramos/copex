use anyhow::Result;

use crate::commands::{build_client, load_config};

pub async fn run() -> Result<()> {
    let config = load_config(None)?;
    let client = build_client(&config)?;
    
    let models = client.list_models().await?;
    
    println!("Available Models:");
    println!();
    for model in models {
        println!("  • {}", model.id);
    }
    
    Ok(())
}
