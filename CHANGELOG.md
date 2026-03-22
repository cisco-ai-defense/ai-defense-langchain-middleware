# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-22

### Added

- **`AIDefenseMiddleware`** — LLM inspection middleware using `ChatInspectionClient`.
  Inspects user messages (`before_model`) and AI responses (`after_model`) against
  Cisco AI Defense security policies. Supports enforce/monitor/off modes, fail-open/closed
  semantics, configurable rules, and violation callbacks.

- **`AIDefenseAgentsecMiddleware`** — LLM inspection middleware using agentsec's
  `LLMInspector`. Provides retry with exponential backoff, fail-open/closed semantics,
  and agentsec configuration inheritance.

- **`AIDefenseToolMiddleware`** — Tool/MCP inspection middleware using
  `MCPInspectionClient`. Inspects tool call arguments before execution and tool
  results after execution via the `wrap_tool_call` hook.

- **`AIDefenseAgentsecToolMiddleware`** — Tool/MCP inspection middleware using
  agentsec's `MCPInspector`. Adds retry, backoff, and fail-open support for
  tool call inspection.

- **`from_env()` class methods** on all middleware for configuration via environment
  variables (`AIDEFENSE_API_KEY`, `AIDEFENSE_REGION`, `AIDEFENSE_AGENTSEC_ENDPOINT`, etc.).

- **Full async support** — All middleware implement both sync and async hooks
  (`before_model`/`abefore_model`, `after_model`/`aafter_model`,
  `wrap_tool_call`/`awrap_tool_call`).

- **Region normalization** — Short aliases (`"us"`, `"eu"`, `"apj"`) are automatically
  expanded to full AWS region names (e.g., `"us-east-1"`, `"eu-west-1"`, `"ap-southeast-1"`).

- 8 example scripts covering enforce mode, monitor mode, custom rules, composed
  middleware, side-by-side comparison, and tool/MCP inspection.

- 44 unit tests covering all four middleware classes with mocked AI Defense clients.

- Apache 2.0 license headers on all Python source files.

- Pre-commit hooks for license header enforcement (`addlicense`) and code formatting (`black`).

- GitHub Actions CI workflow running tests across Python 3.10–3.13.

- Open-source governance files: `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`.

### Dependencies

- Requires `cisco-aidefense-sdk >= 2.1.0`.
- Requires `langchain >= 1.0.0` and `langgraph >= 0.2.0`.
