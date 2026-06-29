# Changelog

All notable changes to `langchain-cisco-aidefense` are documented here.

---

## [1.1.0] — 2026-06-29

### Added

- **`create_react_agent` support** — Cisco AI Defense can now protect agents
  built with LangGraph's `create_react_agent`, which does not expose the
  `before_model` / `after_model` hook API used by the existing middleware.

- **`AIDefenseHooks`** — Supplies `pre_model_hook` and `post_model_hook`
  callables that plug directly into `create_react_agent`'s native hook
  parameters to inspect every LLM input and output.

- **`AIDefenseToolNode`** — A `ToolNode` subclass that inspects every tool
  call request (name + arguments) and response using `ToolNode`'s native
  `wrap_tool_call` / `awrap_tool_call` interceptors.

- **`create_aidefense_react_agent`** — Drop-in replacement for
  `create_react_agent`. Change one function name and add your AI Defense
  config — both LLM and tool inspection are wired automatically.

- **`AIDefenseViolationError`** — Raised in `"enforce"` mode when AI Defense
  blocks a request or response. Carries `.direction` (`"input"`, `"output"`,
  `"tool '<name>' input"`, `"tool '<name>' output"`) and `.response`
  (the full `InspectResponse` from the SDK).

- **Azure OpenAI example** (`examples/10_azure_openai_create_react_agent.py`)
  — Shows both usage patterns with `AzureChatOpenAI`, including a macOS
  corporate-CA SSL fix for Cisco laptops.

### Changed

- Minimum `langgraph` version raised from `>=0.2.0` to `>=0.2.27` to ensure
  `pre_model_hook`, `post_model_hook`, and `ToolNode.wrap_tool_call` are
  available.

### Fixed

- `inspect_messages` now handles multimodal / structured message content
  (e.g. `HumanMessage(content=[{"type": "text", "text": "..."}])`). Previously
  only plain string content was inspected; list-of-blocks content was silently
  skipped, creating a security bypass.

- `_awrap_tool_call` now correctly `await`s the async `execute` callable and
  offloads blocking SDK HTTP calls to `asyncio.to_thread()` so concurrent async
  tool calls do not serialize on the event loop.

- Provider built-in tool dicts (e.g. OpenAI / Anthropic native tools) are now
  bound to the model via `model.bind_tools()` instead of being passed to
  `AIDefenseToolNode`, which would raise `ValueError` because `ToolNode` cannot
  convert them to executable tools.

- A failing `on_violation` callback no longer suppresses
  `AIDefenseViolationError` in enforce mode or interrupts monitor-mode flow.
  Callback exceptions are caught and logged via `logger.exception`.

- The `ImportError` guard in `__init__.py` now only rewrites errors that
  actually originate from `langgraph`; other import failures are re-raised
  unchanged so the real cause is not obscured.

- Passing a plain `ToolNode` to `create_aidefense_react_agent` now emits a
  `UserWarning` explaining that tool-call inspection is disabled, instead of
  silently skipping it.

---

## [1.0.0] — 2026-04-01

Initial release.

- `AIDefenseMiddleware` — LLM input/output inspection for `create_agent`
  (`before_model` / `after_model` hooks).
- `AIDefenseAgentsecMiddleware` — agentsec-based LLM inspection.
- `AIDefenseToolMiddleware` — MCP tool call and response inspection.
- `AIDefenseAgentsecToolMiddleware` — agentsec-based tool inspection.
