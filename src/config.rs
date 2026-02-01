use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub max_auto_continues: u32,
    pub base_delay: f64,
    pub max_delay: f64,
    pub exponential_base: f64,
    pub retry_on_any_error: bool,
    pub retry_on_errors: Vec<String>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            max_auto_continues: 3,
            base_delay: 1.0,
            max_delay: 30.0,
            exponential_base: 2.0,
            retry_on_any_error: true,
            retry_on_errors: vec![
                "500".into(),
                "502".into(),
                "503".into(),
                "504".into(),
                "Internal Server Error".into(),
                "rate limit".into(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: String,
    pub reasoning_effort: String,
    pub streaming: bool,
    pub retry: RetryConfig,
    pub timeout: f64,
    pub auto_continue: bool,
    pub continue_prompt: String,
    pub recovery_prompt_max_chars: usize,
    pub auth_refresh_interval: f64,
    pub auth_refresh_buffer: f64,
    pub auth_refresh_on_error: bool,
    pub ui_theme: String,
    pub ui_density: String,
    pub ui_ascii_icons: bool,
    pub stream_queue_max_size: usize,
    pub stream_drop_mode: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "claude-opus-4.5".into(),
            reasoning_effort: "xhigh".into(),
            streaming: true,
            retry: RetryConfig::default(),
            timeout: 300.0,
            auto_continue: true,
            continue_prompt: "Keep going".into(),
            recovery_prompt_max_chars: 8000,
            auth_refresh_interval: 3300.0,
            auth_refresh_buffer: 300.0,
            auth_refresh_on_error: true,
            ui_theme: "default".into(),
            ui_density: "extended".into(),
            ui_ascii_icons: false,
            stream_queue_max_size: 1000,
            stream_drop_mode: "drop_oldest".into(),
        }
    }
}

impl Config {
    pub fn default_path() -> Result<PathBuf> {
        let home = dirs::home_dir().context("Failed to resolve home directory")?;
        Ok(home.join(".copex").join("config.toml"))
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config TOML: {}", path.display()))?;
        Ok(config)
    }

    pub fn load() -> Result<Self> {
        let path = Self::default_path()?;
        if path.exists() {
            return Self::load_from_file(path);
        }
        Ok(Self::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_loads_from_toml() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("config.toml");
        let toml = r#"
model = "gpt-5.1-codex"
reasoning_effort = "high"
streaming = false
timeout = 120.0
auto_continue = false
continue_prompt = "Continue"
recovery_prompt_max_chars = 4000
auth_refresh_interval = 0.0
auth_refresh_buffer = 0.0
auth_refresh_on_error = false
ui_theme = "mono"
ui_density = "compact"
ui_ascii_icons = true
stream_queue_max_size = 50
stream_drop_mode = "drop_newest"

[retry]
max_retries = 2
max_auto_continues = 0
base_delay = 0.5
max_delay = 5.0
exponential_base = 2.0
retry_on_any_error = true
retry_on_errors = ["500"]
"#;
        fs::write(&path, toml).expect("write");
        let loaded = Config::load_from_file(&path).expect("load");
        assert_eq!(loaded.model, "gpt-5.1-codex");
        assert_eq!(loaded.ui_theme, "mono");
        assert_eq!(loaded.retry.max_retries, 2);
    }
}
