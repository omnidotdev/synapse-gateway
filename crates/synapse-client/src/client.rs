use std::pin::Pin;

use bytes::Bytes;
use futures::stream::{self, Stream, StreamExt};
use reqwest::header::AUTHORIZATION;
use url::Url;

use crate::error::{Result, SynapseClientError};
use crate::types::{
    ChatEvent, ChatRequest, ChatResponse, McpTool, Model, ModelList, SpeechRequest, StreamChunk,
    ToolResult, ToolSearchResult, Transcription,
};

/// Typed HTTP client for the Synapse AI router
#[derive(Debug, Clone)]
pub struct SynapseClient {
    base_url: Url,
    http: reqwest::Client,
    api_key: Option<String>,
}

impl SynapseClient {
    /// Create a new client pointing at the given base URL
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is invalid
    pub fn new(base_url: &str) -> Result<Self> {
        let base_url = Url::parse(base_url)
            .map_err(|e| SynapseClientError::Config(format!("invalid base URL: {e}")))?;

        Ok(Self {
            base_url,
            http: reqwest::Client::new(),
            api_key: None,
        })
    }

    /// Set the API key for authentication
    #[must_use]
    pub fn with_api_key(mut self, api_key: String) -> Self {
        self.api_key = Some(api_key);
        self
    }

    /// Get the base URL
    #[must_use]
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    // -- LLM --

    /// Send a chat completion request (non-streaming)
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the response cannot be parsed
    pub async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let url = self.url("/v1/chat/completions");

        let mut request = ChatRequest {
            stream: false,
            ..req.clone()
        };
        request.stream = false;

        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&request)
            .send()
            .await?;

        self.handle_error(response).await?.json().await.map_err(Into::into)
    }

    /// Send a streaming chat completion request
    ///
    /// Returns a stream of `ChatEvent`s parsed from SSE
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails
    pub async fn chat_completion_stream(
        &self,
        req: &ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        let url = self.url("/v1/chat/completions");

        let mut request = req.clone();
        request.stream = true;

        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&request)
            .send()
            .await?;

        let response = self.handle_error(response).await?;
        let byte_stream = response.bytes_stream();

        let event_stream = parse_sse_stream(byte_stream);

        Ok(Box::pin(event_stream))
    }

    /// List available models
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails
    pub async fn list_models(&self) -> Result<Vec<Model>> {
        let url = self.url("/v1/models");

        let response = self.request(reqwest::Method::GET, &url).send().await?;

        let list: ModelList = self.handle_error(response).await?.json().await?;
        Ok(list.data)
    }

    // -- STT --

    /// Transcribe audio
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails
    pub async fn transcribe(
        &self,
        audio: Bytes,
        filename: &str,
        model: &str,
    ) -> Result<Transcription> {
        let url = self.url("/v1/audio/transcriptions");

        let part = reqwest::multipart::Part::bytes(audio.to_vec())
            .file_name(filename.to_owned())
            .mime_str("audio/wav")
            .map_err(|e| SynapseClientError::Config(format!("invalid mime type: {e}")))?;

        let form = reqwest::multipart::Form::new()
            .text("model", model.to_owned())
            .part("file", part);

        let response = self
            .request(reqwest::Method::POST, &url)
            .multipart(form)
            .send()
            .await?;

        self.handle_error(response).await?.json().await.map_err(Into::into)
    }

    // -- TTS --

    /// Synthesize text to speech
    ///
    /// Returns raw audio bytes
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails
    pub async fn synthesize(&self, req: &SpeechRequest) -> Result<Bytes> {
        let url = self.url("/v1/audio/speech");

        let response = self
            .request(reqwest::Method::POST, &url)
            .json(req)
            .send()
            .await?;

        Ok(self.handle_error(response).await?.bytes().await?)
    }

    // -- MCP --

    /// List available MCP tools
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails
    pub async fn list_tools(&self, server: Option<&str>) -> Result<Vec<McpTool>> {
        let url = self.url("/mcp/tools/list");

        let body = serde_json::json!({
            "server": server,
        });

        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&body)
            .send()
            .await?;

        #[derive(Deserialize)]
        struct Resp {
            tools: Vec<McpTool>,
        }

        let resp: Resp = self.handle_error(response).await?.json().await?;
        Ok(resp.tools)
    }

    /// Call an MCP tool
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolResult> {
        let url = self.url("/mcp/tools/call");

        let body = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let response = self
            .request(reqwest::Method::POST, &url)
            .json(&body)
            .send()
            .await?;

        self.handle_error(response).await?.json().await.map_err(Into::into)
    }

    /// Search MCP tools by query
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails
    pub async fn search_tools(
        &self,
        query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ToolSearchResult>> {
        let mut url = self.url("/mcp/search");
        url.query_pairs_mut().append_pair("q", query);
        if let Some(limit) = limit {
            url.query_pairs_mut()
                .append_pair("limit", &limit.to_string());
        }

        let response = self.request(reqwest::Method::GET, &url).send().await?;

        #[derive(Deserialize)]
        struct Resp {
            results: Vec<ToolSearchResult>,
        }

        let resp: Resp = self.handle_error(response).await?.json().await?;
        Ok(resp.results)
    }

    // -- Helpers --

    fn url(&self, path: &str) -> Url {
        let mut url = self.base_url.clone();
        url.set_path(path);
        url
    }

    fn request(&self, method: reqwest::Method, url: &Url) -> reqwest::RequestBuilder {
        let mut builder = self.http.request(method, url.as_str());

        if let Some(ref key) = self.api_key {
            builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
        }

        builder
    }

    async fn handle_error(&self, response: reqwest::Response) -> Result<reqwest::Response> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        // Try to parse error body
        let body = response.text().await.unwrap_or_default();
        let (error_type, message) = parse_error_body(&body);

        Err(SynapseClientError::Api {
            status: status.as_u16(),
            error_type,
            message,
        })
    }
}

/// Parse an error response body into (type, message)
fn parse_error_body(body: &str) -> (String, String) {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        let error = &json["error"];
        let error_type = error["type"]
            .as_str()
            .unwrap_or("unknown")
            .to_owned();
        let message = error["message"]
            .as_str()
            .unwrap_or(body)
            .to_owned();
        (error_type, message)
    } else {
        ("unknown".to_owned(), body.to_owned())
    }
}

/// Parse a byte stream of SSE data into `ChatEvent`s
fn parse_sse_stream<S>(byte_stream: S) -> impl Stream<Item = Result<ChatEvent>>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send + 'static,
{
    // Buffer for incomplete lines across chunks
    let line_stream = byte_stream
        .map(|result| result.map_err(SynapseClientError::Http))
        .scan(String::new(), |buffer, result| {
            let bytes = match result {
                Ok(b) => b,
                Err(e) => return std::future::ready(Some(vec![Err(e)])),
            };

            let text = String::from_utf8_lossy(&bytes);
            buffer.push_str(&text);

            let mut events = Vec::new();
            while let Some(pos) = buffer.find("\n\n") {
                let chunk = buffer[..pos].to_owned();
                *buffer = buffer[pos + 2..].to_owned();

                for line in chunk.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        events.push(Ok(data.to_owned()));
                    }
                }
            }

            std::future::ready(Some(events))
        })
        .flat_map(stream::iter);

    // Parse each SSE data line into ChatEvents
    line_stream.filter_map(|result| async move {
        match result {
            Err(e) => Some(Err(e)),
            Ok(data) => {
                if data == "[DONE]" {
                    return Some(Ok(ChatEvent::Done {
                        finish_reason: None,
                        usage: None,
                    }));
                }

                match serde_json::from_str::<StreamChunk>(&data) {
                    Ok(chunk) => chunk_to_event(chunk),
                    Err(e) => Some(Err(SynapseClientError::Parse(format!(
                        "failed to parse stream chunk: {e}"
                    )))),
                }
            }
        }
    })
}

/// Convert a parsed stream chunk into a `ChatEvent`
fn chunk_to_event(chunk: StreamChunk) -> Option<Result<ChatEvent>> {
    let choice = chunk.choices.into_iter().next()?;

    // Check for finish
    if let Some(reason) = choice.finish_reason {
        return Some(Ok(ChatEvent::Done {
            finish_reason: Some(reason),
            usage: chunk.usage,
        }));
    }

    // Check for tool calls
    if let Some(tool_calls) = choice.delta.tool_calls {
        for tc in tool_calls {
            if let Some(ref func) = tc.function {
                if let (Some(id), Some(name)) = (&tc.id, &func.name) {
                    return Some(Ok(ChatEvent::ToolCallStart {
                        index: tc.index,
                        id: id.clone(),
                        name: name.clone(),
                    }));
                }
                if let Some(ref args) = func.arguments {
                    return Some(Ok(ChatEvent::ToolCallDelta {
                        index: tc.index,
                        arguments: args.clone(),
                    }));
                }
            }
        }
    }

    // Text content
    if let Some(content) = choice.delta.content {
        return Some(Ok(ChatEvent::ContentDelta(content)));
    }

    None
}

use serde::Deserialize;
