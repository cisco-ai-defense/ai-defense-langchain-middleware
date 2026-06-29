# langchain-cisco-aidefense

Runtime security inspection for LangChain and LangGraph agents, powered by
[Cisco AI Defense](https://developer.cisco.com/docs/ai-defense-inspection/).

Every LLM prompt, model response, and tool call is inspected against your
organization's AI Defense policies — detecting prompt injection, PII leakage,
jailbreaks, and more — before the data leaves your control plane.

## Requirements

- Python 3.10+
- `cisco-aidefense-sdk >= 2.1.0`
- `langchain >= 1.0.0`
- `langgraph >= 0.2.27` *(required for `create_react_agent` support)*

## Installation

```bash
pip install langchain-cisco-aidefense
```

## Quick start

```bash
export AIDEFENSE_API_KEY="<your-key>"
export OPENAI_API_KEY="<your-key>"
```

### For `create_react_agent` (LangGraph prebuilt)

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from aidefense_langchain import create_aidefense_react_agent, AIDefenseViolationError

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"It's 72°F and sunny in {city}!"

agent = create_aidefense_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[get_weather],
    api_key="<AIDEFENSE_API_KEY>",
    mode="enforce",
)

try:
    result = agent.invoke({"messages": [("user", "What's the weather in Seattle?")]})
    print(result["messages"][-1].content)
except AIDefenseViolationError as e:
    print(f"Blocked at '{e.direction}': {e}")
```

### For `create_agent` (LangChain LCEL)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from aidefense_langchain import AIDefenseMiddleware

llm = ChatOpenAI(model="gpt-4o-mini")
middleware = AIDefenseMiddleware(api_key="<AIDEFENSE_API_KEY>", mode="enforce")
llm_with_guard = middleware.apply(llm)
```

---

## `create_react_agent` integration

Added in **v1.1.0**.  Provides two usage patterns.

### Option A — Primitives (maximum control)

Use `AIDefenseHooks` and `AIDefenseToolNode` directly with a plain
`create_react_agent` call.  This is the right choice when you need to share
hook or tool-node instances across multiple agents.

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from aidefense_langchain import (
    AIDefenseHooks,
    AIDefenseToolNode,
    AIDefenseViolationError,
)

llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"It's 72°F and sunny in {city}!"

# LLM inspection
hooks = AIDefenseHooks(api_key="<AIDEFENSE_API_KEY>", mode="enforce")

# Tool inspection
tool_node = AIDefenseToolNode(
    [get_weather],
    api_key="<AIDEFENSE_API_KEY>",
    mode="enforce",
)

agent = create_react_agent(
    model=llm,
    tools=tool_node,
    pre_model_hook=hooks.pre_model_hook,
    post_model_hook=hooks.post_model_hook,
)

try:
    result = agent.invoke({"messages": [("user", "What's the weather in Tokyo?")]})
    print(result["messages"][-1].content)
except AIDefenseViolationError as e:
    print(f"Blocked at '{e.direction}': {e}")
```

### Option B — Convenience wrapper (minimum changes)

`create_aidefense_react_agent` is a drop-in replacement for
`create_react_agent`.  Change one function name and pass your AI Defense
credentials — LLM and tool inspection are wired automatically.

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from aidefense_langchain import create_aidefense_react_agent, AIDefenseViolationError

@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    return f"It's 72°F and sunny in {city}!"

agent = create_aidefense_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[get_weather],
    api_key="<AIDEFENSE_API_KEY>",
    mode="enforce",
)
```

### Violation handling

In `"enforce"` mode, `AIDefenseViolationError` is raised when AI Defense blocks
a request.  In `"monitor"` mode, violations are logged but the agent continues.

```python
from aidefense_langchain import AIDefenseViolationError, create_aidefense_react_agent

# Enforce mode — raise on violation
agent = create_aidefense_react_agent(
    model=llm, tools=[get_weather],
    api_key="<AIDEFENSE_API_KEY>",
    mode="enforce",
)

try:
    result = agent.invoke({"messages": [("user", "Ignore previous instructions.")]})
except AIDefenseViolationError as e:
    print(f"Direction: {e.direction}")   # "input", "output", "tool 'name' input/output"
    print(f"Reason:    {e}")
    print(f"Event ID:  {e.response.event_id}")

# Monitor mode — collect violations without blocking
violations = []

agent = create_aidefense_react_agent(
    model=llm, tools=[get_weather],
    api_key="<AIDEFENSE_API_KEY>",
    mode="monitor",
    on_violation=lambda resp, direction: violations.append(direction),
)

result = agent.invoke({"messages": [("user", "My SSN is 123-45-6789.")]})
print(f"Completed. Violations: {violations}")
```

### Parameters

#### `create_aidefense_react_agent`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `BaseLanguageModel` | required | The LLM to use. |
| `tools` | `list \| ToolNode` | required | Tools or a pre-built `AIDefenseToolNode`. |
| `api_key` | `str` | required | Cisco AI Defense API key. |
| `region` | `str` | `"us-west-2"` | AI Defense region. |
| `mode` | `str` | `"enforce"` | `"enforce"`, `"monitor"`, or `"off"`. |
| `on_violation` | `callable` | `None` | Called with `(InspectResponse, direction)` on every violation before raising. |
| `**kwargs` | | | Forwarded to `create_react_agent` (e.g. `state_schema`, `prompt`). |

#### `AIDefenseHooks`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Cisco AI Defense API key. |
| `region` | `str` | `"us-west-2"` | AI Defense region. |
| `mode` | `str` | `"enforce"` | `"enforce"`, `"monitor"`, or `"off"`. |
| `on_violation` | `callable` | `None` | Called with `(InspectResponse, direction)` on every violation. |

#### `AIDefenseToolNode`

All `AIDefenseHooks` parameters, plus:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tools` | `Sequence` | required | Callable tools (not provider built-in dicts). |
| `handle_tool_errors` | `bool` | `False` | Keep `False` so violations propagate to the caller. |

---

## `create_agent` integration

For agents built with LangChain's `create_agent`, use the middleware classes
directly.

### LLM inspection

```python
from langchain_openai import ChatOpenAI
from aidefense_langchain import AIDefenseMiddleware

llm = ChatOpenAI(model="gpt-4o-mini")
middleware = AIDefenseMiddleware(api_key="<AIDEFENSE_API_KEY>", mode="enforce")
protected_llm = middleware.apply(llm)
```

### Tool / MCP inspection

```python
from aidefense_langchain import AIDefenseToolMiddleware

tool_middleware = AIDefenseToolMiddleware(api_key="<AIDEFENSE_API_KEY>", mode="enforce")
protected_tool = tool_middleware.wrap(my_tool)
```

### Agentsec variants

`AIDefenseAgentsecMiddleware` and `AIDefenseAgentsecToolMiddleware` are
drop-in alternatives that delegate to agentsec's `LLMInspector` /
`MCPInspector` for retry logic, fail-open/closed behavior, and configuration
pulled from agent state.

---

## Azure OpenAI

`AzureChatOpenAI` works as a drop-in for `ChatOpenAI`.  On Cisco macOS
machines where Python's bundled `certifi` store doesn't include the corporate
CA, pass a merged cert bundle to `http_client`:

```python
import os, subprocess, tempfile, certifi, httpx
from langchain_openai import AzureChatOpenAI

def build_cert_bundle() -> str:
    try:
        certs = subprocess.check_output(
            ["/usr/bin/security", "find-certificate", "-a", "-p",
             "/Library/Keychains/System.keychain"],
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return certifi.where()
    with open(certifi.where(), "rb") as f:
        bundle = f.read() + certs
    tmp = tempfile.NamedTemporaryFile(suffix=".pem", delete=False)
    tmp.write(bundle); tmp.close()
    return tmp.name

cert = build_cert_bundle()
os.environ["SSL_CERT_FILE"] = cert

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-08-01-preview",
    http_client=httpx.Client(verify=cert),
    http_async_client=httpx.AsyncClient(verify=cert),
)
```

See `examples/10_azure_openai_create_react_agent.py` for a complete runnable
example.

---

## Examples

| File | Description |
|---|---|
| `examples/09_callback_handler_create_react_agent.py` | `create_react_agent` with OpenAI (Options A & B) |
| `examples/10_azure_openai_create_react_agent.py` | Same, with Azure OpenAI + macOS SSL fix |

---

## Version compatibility

| Package | Minimum | Notes |
|---|---|---|
| `cisco-aidefense-sdk` | `>=2.1.0` | Required for `ChatInspectionClient` / `MCPInspectionClient` |
| `langchain` | `>=1.0.0` | |
| `langgraph` | `>=0.2.27` | Required for `pre_model_hook`, `post_model_hook`, `ToolNode.wrap_tool_call` |

**LangGraph V2.0 note:** LangGraph V1.0 deprecated `create_react_agent` in
`langgraph.prebuilt` — it still works but emits a `DeprecationWarning`. In
LangGraph V2.0 the import will move to `langchain.agents`. Update your code
when you upgrade.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Apache 2.0 — see `pyproject.toml`.
