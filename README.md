# langchain-cisco-aidefense

[![PyPI version](https://img.shields.io/pypi/v/langchain-cisco-aidefense.svg)](https://pypi.org/project/langchain-cisco-aidefense/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

LangChain agent middleware for [Cisco AI Defense](https://developer.cisco.com/docs/ai-defense/overview/), providing runtime security inspection of LLM inputs/outputs **and** tool/MCP calls.

Detects prompt injection, jailbreaks, PII leakage, toxic content, and unsafe tool usage — directly within the LangChain agent loop.

## Installation

```bash
pip install langchain-cisco-aidefense
```

## Middleware Overview

Four middleware implementations are provided:

### LLM Inspection (`before_model` / `after_model`)

| Middleware | Built on | Best for |
|---|---|---|
| `AIDefenseMiddleware` | `ChatInspectionClient` | New integrations — lightweight, no global state |
| `AIDefenseAgentsecMiddleware` | agentsec `LLMInspector` | When you need agentsec's retry/backoff machinery |

### Tool / MCP Inspection (`wrap_tool_call`)

| Middleware | Built on | Best for |
|---|---|---|
| `AIDefenseToolMiddleware` | `MCPInspectionClient` | New integrations — tool/MCP call inspection |
| `AIDefenseAgentsecToolMiddleware` | agentsec `MCPInspector` | When you need agentsec's retry/backoff machinery |

## Quick Start

```bash
pip install langchain-cisco-aidefense langchain-openai
```

```python
from aidefense_langchain import AIDefenseMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    middleware=[
        AIDefenseMiddleware(
            api_key="your-cisco-ai-defense-api-key",
            region="us-west-2",
            mode="enforce",
        ),
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

## LLM Inspection

### `AIDefenseMiddleware` (Recommended)

Uses `ChatInspectionClient` directly. Self-contained configuration, no global state, no monkey-patching.

```python
from aidefense_langchain import AIDefenseMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    middleware=[
        AIDefenseMiddleware(
            api_key="your-api-key",
            region="us-west-2",
            mode="enforce",       # "enforce" | "monitor" | "off"
            fail_open=True,
            inspection_scope="latest_turn",  # default
        ),
    ],
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Cisco AI Defense API key |
| `region` | `str` | `"us-west-2"` | AI Defense region (`"us-west-2"`, `"eu-central-1"`, `"ap-northeast-1"`) |
| `mode` | `str` | `"enforce"` | `"enforce"` (block), `"monitor"` (log only), `"off"` |
| `fail_open` | `bool` | `True` | Allow on inspection API errors |
| `timeout` | `int` | `30` | Inspection timeout in seconds |
| `rules` | `list` | `None` | Rules to enable (e.g. `["PII", "Prompt Injection"]`) |
| `user` | `str` | `None` | User identity for audit |
| `src_app` | `str` | `None` | Source application name |
| `on_violation` | `callable` | `None` | `(InspectResponse, direction) -> None` callback |
| `inspection_scope` | `str` | `"latest_turn"` | `"latest_turn"` inspects only the newest conversation window for each hook; `"thread"` reinspects the full retained transcript |

#### Inspection scope

- `latest_turn` (default)
  - `before_model` inspects only the new messages added since the previous assistant output.
  - `after_model` inspects only the current turn window, including the newly generated assistant response.
  - This avoids an older violation poisoning later safe turns forever.
- `thread`
  - Reinspects the full retained `state["messages"]` transcript on every hook.
  - Use this only when you explicitly want transcript-wide enforcement.

#### How it works

```
User message
    |
    v
+--------------------------------------------------+
|  before_model hook                               |
|  -> ChatInspectionClient.inspect_conversation()  |
|  -> if not safe and mode="enforce": jump_to=end  |
+--------------------------------------------------+
    |
    v
  LLM call
    |
    v
+--------------------------------------------------+
|  after_model hook                                |
|  -> ChatInspectionClient.inspect_conversation()  |
|  -> if not safe and mode="enforce": jump_to=end  |
+--------------------------------------------------+
    |
    v
Agent response
```

### `AIDefenseAgentsecMiddleware`

Uses agentsec's `LLMInspector` — gets retry with exponential backoff and fail-open semantics.

```python
from aidefense_langchain import AIDefenseAgentsecMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    middleware=[
        AIDefenseAgentsecMiddleware(
            mode="enforce",
            api_key="your-api-key",
            endpoint="https://us.api.inspect.aidefense.security.cisco.com",
            retry_total=3,
            retry_backoff=1.0,
            inspection_scope="latest_turn",  # default
        ),
    ],
)
```

> **Do not call `agentsec.protect()` when using middleware.** The middleware
> handles all inspection directly. Calling `protect()` would activate
> monkey-patching on the underlying LLM SDK, causing every request to be
> inspected **twice** — once by the patched SDK and once by the middleware —
> doubling latency and API calls with no security benefit.

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `str` | `"enforce"` | `"enforce"`, `"monitor"`, or `"off"` |
| `api_key` | `str` | from state/env | AI Defense API key |
| `endpoint` | `str` | from state/env | AI Defense API endpoint |
| `fail_open` | `bool` | `True` | Allow on inspection errors |
| `timeout_ms` | `int` | from state | Timeout in milliseconds |
| `retry_total` | `int` | `1` | Retry attempts |
| `retry_backoff` | `float` | `0.0` | Backoff factor in seconds |
| `rules` | `list` | `None` | Inspection rules |
| `user` | `str` | `None` | User identity |
| `src_app` | `str` | `None` | Source application name |
| `on_violation` | `callable` | `None` | `(Decision, direction) -> None` callback |
| `inspection_scope` | `str` | `"latest_turn"` | `"latest_turn"` inspects only the newest conversation window for each hook; `"thread"` reinspects the full retained transcript |

## Tool / MCP Inspection

### `AIDefenseToolMiddleware` (Recommended for tools)

Uses `MCPInspectionClient` to inspect tool call requests (name + arguments) and tool call results.

```python
from aidefense_langchain import AIDefenseMiddleware, AIDefenseToolMiddleware
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[search_db, send_email],
    middleware=[
        AIDefenseMiddleware(api_key="your-key", mode="enforce"),
        AIDefenseToolMiddleware(api_key="your-key", mode="enforce"),
    ],
)
```

### `AIDefenseAgentsecToolMiddleware`

Uses agentsec's `MCPInspector` with retry, backoff, and fail-open support.

```python
from aidefense_langchain import AIDefenseAgentsecMiddleware, AIDefenseAgentsecToolMiddleware

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[read_file, execute_query],
    middleware=[
        AIDefenseAgentsecMiddleware(mode="enforce", api_key="your-key"),
        AIDefenseAgentsecToolMiddleware(mode="enforce", api_key="your-key"),
    ],
)
```

### Tool Middleware Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required / from env | Cisco AI Defense API key |
| `region` | `str` | `"us-west-2"` | AI Defense region (ChatClient variant only) |
| `mode` | `str` | `"enforce"` | `"enforce"`, `"monitor"`, or `"off"` |
| `fail_open` | `bool` | `True` | Allow on inspection API errors |
| `inspect_requests` | `bool` | `True` | Inspect tool call requests before execution |
| `inspect_responses` | `bool` | `True` | Inspect tool results after execution |
| `on_violation` | `callable` | `None` | Violation callback |

### How tool inspection works

```
Tool call (from LLM)
    |
    v
+--------------------------------------------------+
|  wrap_tool_call -- PRE-CALL inspection           |
|  -> MCPInspectionClient.inspect_tool_call()      |
|  -> if not safe and mode="enforce": return block |
+--------------------------------------------------+
    |
    v
  Tool executes
    |
    v
+--------------------------------------------------+
|  wrap_tool_call -- POST-CALL inspection          |
|  -> MCPInspectionClient.inspect_response()       |
|  -> if not safe and mode="enforce": return block |
+--------------------------------------------------+
    |
    v
Tool result returned to agent
```

This covers **all** tool types:
- LangChain tools (`@tool` decorated functions)
- MCP tools registered via LangChain's MCP integration
- Any tool executed through the agent's tool node

## Environment Variables

The middleware can also be configured via environment variables (used by `from_env()` class methods):

```bash
export AIDEFENSE_API_KEY=your-api-key
export AIDEFENSE_REGION=us-west-2
export AIDEFENSE_MODE=enforce
export AIDEFENSE_FAIL_OPEN=true
export AIDEFENSE_TIMEOUT=30
export AIDEFENSE_INSPECTION_SCOPE=latest_turn
```

```python
from aidefense_langchain import AIDefenseMiddleware

middleware = AIDefenseMiddleware.from_env()
```

## Comparison

| Criteria | `AIDefenseMiddleware` | `AIDefenseAgentsecMiddleware` |
|---|:---:|:---:|
| No global state / side effects | Yes | No (uses `_state`) |
| Self-contained config | Yes | Yes (pass explicitly) |
| Built-in retry + backoff | Via `Config` | Custom |
| Built-in fail-open | In middleware | In inspector |
| `inspect_prompt` / `inspect_response` | Yes | No (`inspect_conversation` only) |
| Dependency footprint | Lighter (`aidefense.runtime`) | Heavier (`aidefense.runtime.agentsec`) |

**Recommendation**: Use `AIDefenseMiddleware` for new projects. Use `AIDefenseAgentsecMiddleware` when you need agentsec's retry/backoff machinery.

## Enforcement Modes

| Mode | Behavior |
|---|---|
| `enforce` | Block violations — agent returns a "blocked" message via `jump_to: "end"` |
| `monitor` | Log violations and invoke `on_violation` callback; never blocks |
| `off` | Skip inspection entirely |

## Fail-Open Behavior

When `fail_open=True` (default) and the AI Defense inspection API is unreachable:
- The request is **allowed** to proceed
- A warning is logged

When `fail_open=False`:
- The request is **blocked** (or an exception is raised)

## Examples

| # | File | Description |
|---|---|---|
| 1 | `01_chat_client_enforce.py` | ChatClient middleware — enforce mode (block violations) |
| 2 | `02_chat_client_monitor.py` | ChatClient middleware — monitor mode with violation callback |
| 3 | `03_chat_client_with_rules.py` | ChatClient middleware — specific rules (PII, Prompt Injection) |
| 4 | `04_agentsec_enforce.py` | Agentsec middleware — enforce mode with retry config |
| 5 | `05_composed_middleware.py` | AI Defense + custom logging middleware composed together |
| 6 | `06_side_by_side.py` | Same request through both middleware — side-by-side comparison |
| 7 | `07_tool_inspection_enforce.py` | Tool inspection — LLM + tool call inspection combined |
| 8 | `08_tool_inspection_agentsec.py` | Agentsec tool inspection — MCPInspector with retry |

## Development

```bash
git clone https://github.com/cisco-ai-defense/ai-defense-langchain-middleware.git
cd ai-defense-langchain-middleware
pip install -e ".[dev,examples]"
pytest
```

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.
