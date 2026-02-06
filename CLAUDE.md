# Claude Instructions

## Project Overview

Synapse is an AI router for unified LLM, MCP, STT, and TTS provider management. Built in Rust.

## Architecture

Rust workspace with binary crate (`synapse/`) and library crates (`crates/`).

## Commands

```bash
cargo build                       # Build all
cargo build --release -p synapse  # Build binary
cargo test                        # Run all tests
cargo clippy                      # Lint
cargo run -p synapse -- --help    # Show CLI help
```

## Code Style

Follow Omni Rust conventions:
- No trailing punctuation on comments
- Use `thiserror` for library errors, `anyhow` for application errors
- Use `tracing` for logging
- Document public items with `///`
