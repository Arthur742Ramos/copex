use anyhow::Result;

use crate::commands::{build_client, load_config};

pub async fn login() -> Result<()> {
    if which::which("gh").is_err() {
        println!("Error: GitHub CLI (gh) not found.");
        println!("Install it from: https://cli.github.com/");
        return Ok(());
    }
    println!("Opening browser for GitHub authentication...");
    let status = std::process::Command::new("gh").args(["auth", "login"]).status()?;
    if status.success() {
        println!("Logged in.");
    } else {
        println!("Login may have failed. Check status with: copex status");
    }
    Ok(())
}

pub async fn logout() -> Result<()> {
    if which::which("gh").is_err() {
        println!("Error: GitHub CLI (gh) not found.");
        return Ok(());
    }
    let status = std::process::Command::new("gh").args(["auth", "logout"]).status()?;
    if status.success() {
        println!("Logged out.");
    }
    Ok(())
}

pub async fn status() -> Result<()> {
    let config = load_config(None)?;
    let client = build_client(&config)?;
    let copilot_version = client
        .get_status()
        .await
        .map(|status| status.version)
        .unwrap_or_else(|_| "N/A".to_string());
    let auth_status = client.get_auth_status().await.ok();
    let gh_path = which::which("gh").ok();

    println!("Copex Status");
    println!("  Copex Version: 0.1.0 (Rust)");
    println!("  Copilot CLI Version: {}", copilot_version);
    if let Some(login) = auth_status.and_then(|auth| auth.login) {
        println!("  GitHub Login: {}", login);
    }
    if gh_path.is_none() {
        println!("  GitHub CLI: not found");
        println!("  Install: https://cli.github.com/");
    } else {
        let _ = std::process::Command::new("gh").args(["auth", "status"]).status();
    }
    Ok(())
}
