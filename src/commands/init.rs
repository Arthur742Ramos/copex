use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::cli::InitArgs;
use crate::config::Config;

pub async fn run(args: InitArgs) -> Result<()> {
    let path = expand_path(&args.path);
    let config = Config::default();
    let payload = toml::to_string_pretty(&config)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    std::fs::write(&path, payload).with_context(|| format!("Failed to write {}", path.display()))?;
    println!("Wrote config to {}", path.display());
    Ok(())
}

fn expand_path(raw: &str) -> PathBuf {
    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(raw)
}
