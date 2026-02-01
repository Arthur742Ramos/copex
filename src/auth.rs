use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Duration, TimeZone, Utc};
use serde::{Deserialize, Deserializer, Serialize};

const DEFAULT_REFRESH_URL: &str = "https://api.github.com/copilot_internal/v2/token";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

impl AuthToken {
    pub fn new(token: impl Into<String>, expires_at: DateTime<Utc>) -> Self {
        Self {
            token: token.into(),
            expires_at,
        }
    }

    pub fn is_expired(&self, buffer: Duration) -> bool {
        Utc::now() + buffer >= self.expires_at
    }
}

#[derive(Debug, Clone)]
pub struct AuthManager {
    cache_path: PathBuf,
    refresh_url: String,
}

impl AuthManager {
    pub fn new(cache_path: PathBuf) -> Self {
        Self {
            cache_path,
            refresh_url: DEFAULT_REFRESH_URL.to_string(),
        }
    }

    pub fn default() -> Result<Self> {
        Ok(Self::new(default_cache_path()?))
    }

    pub fn load_cached(&self) -> Result<Option<AuthToken>> {
        if !self.cache_path.exists() {
            return Ok(None);
        }
        let contents = fs::read_to_string(&self.cache_path).with_context(|| {
            format!("Failed to read token cache: {}", self.cache_path.display())
        })?;
        let token = serde_json::from_str(&contents).with_context(|| {
            format!("Failed to parse token cache: {}", self.cache_path.display())
        })?;
        Ok(Some(token))
    }

    pub fn save_token(&self, token: &AuthToken) -> Result<()> {
        if let Some(parent) = self.cache_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create cache dir: {}", parent.display())
            })?;
        }
        let payload = serde_json::to_string_pretty(token)?;
        fs::write(&self.cache_path, payload).with_context(|| {
            format!("Failed to write token cache: {}", self.cache_path.display())
        })?;
        Ok(())
    }

    pub async fn ensure_token(&self, buffer: Duration) -> Result<AuthToken> {
        self.ensure_token_with(buffer, || async {
            let github_token = resolve_github_token()?;
            self.refresh_token(&github_token).await
        })
        .await
    }

    pub async fn ensure_token_with<F, Fut>(&self, buffer: Duration, refresh: F) -> Result<AuthToken>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<AuthToken>>,
    {
        if let Some(cached) = self.load_cached()? {
            if !cached.is_expired(buffer) {
                return Ok(cached);
            }
        }
        let token = refresh().await?;
        self.save_token(&token)?;
        Ok(token)
    }

    pub async fn refresh_token(&self, github_token: &str) -> Result<AuthToken> {
        let client = reqwest::Client::new();
        let response = client
            .get(&self.refresh_url)
            .header("Authorization", format!("token {}", github_token))
            .header("Accept", "application/json")
            .header("User-Agent", "copex-rust")
            .send()
            .await?
            .error_for_status()?;
        let payload: TokenResponse = response.json().await?;
        Ok(AuthToken::new(payload.token, payload.expires_at))
    }

    pub fn refresh_url(&self) -> &str {
        &self.refresh_url
    }

    pub fn set_refresh_url(&mut self, url: impl Into<String>) {
        self.refresh_url = url.into();
    }
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    token: String,
    #[serde(alias = "expiresAt", deserialize_with = "deserialize_expires_at")]
    expires_at: DateTime<Utc>,
}

fn deserialize_expires_at<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::String(raw) => DateTime::parse_from_rfc3339(&raw)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(serde::de::Error::custom),
        serde_json::Value::Number(num) => {
            let mut seconds = num
                .as_i64()
                .ok_or_else(|| serde::de::Error::custom("expires_at must be integer"))?;
            if seconds > 2_000_000_000_000 {
                seconds /= 1000;
            }
            Utc.timestamp_opt(seconds, 0)
                .single()
                .ok_or_else(|| serde::de::Error::custom("expires_at timestamp out of range"))
        }
        _ => Err(serde::de::Error::custom(
            "expires_at must be string or integer",
        )),
    }
}

fn default_cache_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Failed to resolve home directory")?;
    Ok(home.join(".copex").join("token.json"))
}

fn resolve_github_token() -> Result<String> {
    if let Ok(token) = std::env::var("GITHUB_TOKEN") {
        if !token.trim().is_empty() {
            return Ok(token);
        }
    }
    if let Ok(token) = std::env::var("GH_TOKEN") {
        if !token.trim().is_empty() {
            return Ok(token);
        }
    }
    let output = Command::new("gh")
        .args(["auth", "token"])
        .output()
        .map_err(|err| anyhow!("Failed to run gh auth token: {err}"))?;
    if !output.status.success() {
        return Err(anyhow!(
            "gh auth token failed with status: {}",
            output.status
        ));
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err(anyhow!("gh auth token returned empty output"));
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn ensure_token_uses_cached_when_valid() {
        let temp = tempfile::tempdir().expect("tempdir");
        let cache_path = temp.path().join("token.json");
        let manager = AuthManager::new(cache_path.clone());
        let token = AuthToken::new("cached", Utc::now() + Duration::minutes(30));
        manager.save_token(&token).expect("save token");
        let called = Arc::new(AtomicUsize::new(0));
        let called_clone = Arc::clone(&called);
        let loaded = manager
            .ensure_token_with(Duration::minutes(5), move || {
                let called_clone = Arc::clone(&called_clone);
                async move {
                    called_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(AuthToken::new("refreshed", Utc::now() + Duration::minutes(30)))
                }
            })
            .await
            .expect("ensure token");
        assert_eq!(loaded.token, "cached");
        assert_eq!(called.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn ensure_token_refreshes_when_expired() {
        let temp = tempfile::tempdir().expect("tempdir");
        let cache_path = temp.path().join("token.json");
        let manager = AuthManager::new(cache_path.clone());
        let token = AuthToken::new("cached", Utc::now() - Duration::minutes(10));
        manager.save_token(&token).expect("save token");
        let called = Arc::new(AtomicUsize::new(0));
        let called_clone = Arc::clone(&called);
        let loaded = manager
            .ensure_token_with(Duration::minutes(5), move || {
                let called_clone = Arc::clone(&called_clone);
                async move {
                    called_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(AuthToken::new("refreshed", Utc::now() + Duration::minutes(30)))
                }
            })
            .await
            .expect("ensure token");
        assert_eq!(loaded.token, "refreshed");
        assert_eq!(called.load(Ordering::SeqCst), 1);
    }
}
