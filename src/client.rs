use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, USER_AGENT};
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

use crate::auth::AuthManager;
use crate::config::Config;

const DEFAULT_API_URL: &str = "https://api.githubcopilot.com/chat/completions";

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceDelta {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamResponse {
    pub choices: Vec<ChoiceDelta>,
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub delta: String,
    pub is_final: bool,
}

#[derive(Debug, Clone)]
pub struct Client {
    config: Config,
    auth: AuthManager,
    api_url: String,
}

impl Client {
    pub fn new(config: Config, auth: AuthManager) -> Self {
        Self {
            config,
            auth,
            api_url: DEFAULT_API_URL.to_string(),
        }
    }

    pub fn set_api_url(&mut self, url: impl Into<String>) {
        self.api_url = url.into();
    }

    pub async fn stream_chat(
        &self,
        messages: Vec<ChatMessage>,
    ) -> Result<impl Stream<Item = Result<StreamChunk>>> {
        let buffer_ms = (self.config.auth_refresh_buffer * 1000.0) as i64;
        let token = self
            .auth
            .ensure_token(chrono::Duration::milliseconds(buffer_ms))
            .await?;
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: true,
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", token.token))?,
        );
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        headers.insert(USER_AGENT, HeaderValue::from_static("copex-rust"));

        let client = reqwest::Client::new();
        let builder = client
            .post(&self.api_url)
            .headers(headers)
            .json(&request)
            .timeout(Duration::from_secs_f64(self.config.timeout));

        let event_source = EventSource::new(builder)?;
        Ok(parse_sse(event_source))
    }
}

fn parse_sse<S>(mut event_source: S) -> impl Stream<Item = Result<StreamChunk>>
where
    S: Stream<Item = Result<Event, reqwest_eventsource::Error>> + Unpin,
{
    try_stream! {
        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => continue,
                Ok(Event::Message(message)) => {
                    let chunks = parse_message(&message.data)?;
                    for chunk in chunks {
                        let done = chunk.is_final;
                        yield chunk;
                        if done {
                            return;
                        }
                    }
                }
                Err(err) => {
                    Err(anyhow!("SSE error: {err}"))?;
                }
            }
        }
    }
}

fn parse_message(data: &str) -> Result<Vec<StreamChunk>> {
    if data == "[DONE]" {
        return Ok(vec![StreamChunk {
            delta: String::new(),
            is_final: true,
        }]);
    }
    let payload: StreamResponse = serde_json::from_str(data)
        .with_context(|| format!("Failed to parse SSE payload: {}", data))?;
    let mut chunks = Vec::new();
    for choice in payload.choices {
        if let Some(content) = choice.delta.content {
            chunks.push(StreamChunk {
                delta: content,
                is_final: false,
            });
        }
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_message_emits_chunks() {
        let data = r#"{"choices":[{"delta":{"content":"Hello "}}]}"#;
        let chunks = parse_message(data).expect("parse");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].delta, "Hello ");
        assert!(!chunks[0].is_final);
    }

    #[test]
    fn parse_message_done_is_final() {
        let chunks = parse_message("[DONE]").expect("parse");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
    }
}
