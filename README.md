# Synapse

AI router for unified LLM, MCP, STT, and TTS provider management.

## Features

- **LLM Routing** -- Unified interface to OpenAI, Anthropic, Google, AWS Bedrock
- **MCP Aggregation** -- Aggregate Model Context Protocol servers through a single endpoint
- **STT/TTS** -- Speech-to-text and text-to-speech provider routing (coming soon)
- **Rate Limiting** -- In-memory and Redis-backed rate limiting
- **Authentication** -- OAuth2/JWT authentication
- **Telemetry** -- OpenTelemetry metrics, tracing, and logs

## Quick Start

```bash
cargo build --release -p synapse
./target/release/synapse --config synapse.toml
```

## Configuration

See `crates/synapse-config/` for configuration options.

## License

[MIT](LICENSE)
