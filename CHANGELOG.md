# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-20

### Added

- **`AIDefenseMiddleware`** — LLM inspection middleware using `ChatInspectionClient`.
  Inspects user messages (`before_model`) and AI responses (`after_model`) against
  Cisco AI Defense security policies. Supports enforce/monitor/off modes, fail-open,
  configurable rules, and violation callbacks.

- **`AIDefenseAgentsecMiddleware`** — LLM inspection middleware using agentsec's
  `LLMInspector`. Provides retry with exponential backoff, fail-open/closed semantics,
  and agentsec configuration inheritance.

- **`AIDefenseToolMiddleware`** — Tool/MCP inspection middleware using
  `MCPInspectionClient`. Inspects tool call arguments before execution and tool
  results after execution via `wrap_tool_call` hook.

- **`AIDefenseAgentsecToolMiddleware`** — Tool/MCP inspection middleware using
  agentsec's `MCPInspector`. Adds retry, backoff, and fail-open support for
  tool call inspection.

- **`from_env()` class methods** — Configure any middleware from environment
  variables (`AIDEFENSE_API_KEY`, `AIDEFENSE_REGION`, etc.).

- **Async support** — All middleware implement both sync and async hooks
  (`before_model`/`abefore_model`, `after_model`/`aafter_model`,
  `wrap_tool_call`/`awrap_tool_call`).

- **Region aliases** — Short aliases `"us"`, `"eu"`, `"apj"` are normalized
  to full region names automatically.

- 8 examples covering enforce mode, monitor mode, custom rules, composed
  middleware, side-by-side comparison, and tool inspection.

- Unit tests for all four middleware classes with mocked AI Defense clients.
