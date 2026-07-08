# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-07-07

### Changed

- Raised minimum `cisco-aidefense-sdk` version to `>=2.1.2` so installs no longer
  pull in the obsolete PyPI `uuid` package (removed from the SDK in 2.1.2).

## [1.1.0] - 2026-06-29

### Added

- **`create_react_agent` support** — Cisco AI Defense can now protect agents
  built with LangGraph's `create_react_agent`, which does not use the
  `before_model` / `after_model` hook API available to `create_agent`.

- **`AIDefenseHooks`** — Provides `pre_model_hook` and `post_model_hook`
  callables that plug into `create_react_agent`'s native hook parameters to
  inspect every LLM input and output.

- **`AIDefenseToolNode`** — A `ToolNode` subclass that inspects every tool
  call request (name + arguments) and response using `ToolNode`'s native
  `wrap_tool_call` / `awrap_tool_call` interceptors.

- **`create_aidefense_react_agent`** — Drop-in replacement for
  `create_react_agent`. Change one function name and add AI Defense
  credentials — both LLM and tool inspection are wired automatically.

- **`AIDefenseViolationError`** — Raised in `"enforce"` mode when AI Defense
  blocks a request or response. Carries `.direction` and `.response`.

- **Azure OpenAI example** (`examples/10_azure_openai_create_react_agent.py`)
  — Demonstrates both usage patterns with `AzureChatOpenAI`, including a
  macOS corporate-CA SSL workaround.

### Changed

- Minimum `langgraph` version raised from `>=0.2.0` to `>=0.2.27` to require
  `pre_model_hook`, `post_model_hook`, and `ToolNode.wrap_tool_call`.

### Fixed

- `inspect_messages` now handles multimodal / structured message content
  (e.g. `HumanMessage(content=[{"type": "text", "text": "..."}])`). Previously
  only plain string content was inspected; list-of-blocks content was silently
  skipped.

- `_awrap_tool_call` now correctly `await`s the async `execute` callable and
  offloads blocking SDK calls to `asyncio.to_thread()`.

- Provider built-in tool dicts are now bound via `model.bind_tools()` instead
  of being passed to `AIDefenseToolNode`, which would raise `ValueError`.

- A failing `on_violation` callback no longer suppresses
  `AIDefenseViolationError` in enforce mode.

- The `ImportError` guard in `__init__.py` now only rewrites errors that
  originate from `langgraph`; other import failures propagate unchanged.

- Passing a plain `ToolNode` to `create_aidefense_react_agent` now emits a
  `UserWarning` noting that tool-call inspection is disabled.

---

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
  expanded to full AWS region names (e.g., `"us"` → `"us-west-2"`, `"eu"` → `"eu-central-1"`, `"apj"` → `"ap-northeast-1"`).

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
