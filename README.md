# Cisco AI Defense — LangChain Middleware

LangChain agent middleware for [Cisco AI Defense](https://developer.cisco.com/docs/ai-defense/overview/), providing runtime security inspection of LLM inputs and outputs.

Two middleware implementations are provided to demonstrate different integration approaches:

| Middleware | Built on | Best for |
|---|---|---|
| `AIDefenseMiddleware` | `ChatInspectionClient` | New integrations — lightweight, no global state |
| `AIDefenseAgentsecMiddleware` | agentsec `LLMInspector` | Apps that already use `agentsec.protect()` |

## Quick Start

```bash
pip install -e ".[examples]"
cp .env.example .env   # fill in your API keys
python examples/01_chat_client_enforce.py
```

## Approach 1: `AIDefenseMiddleware` (Recommended)

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
            region="us",
            mode="enforce",       # "enforce" | "monitor" | "off"
            fail_open=True,
        ),
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | required | Cisco AI Defense API key |
| `region` | `str` | `"us"` | AI Defense region (`"us"`, `"eu"`, `"apj"`) |
| `mode` | `str` | `"enforce"` | `"enforce"` (block), `"monitor"` (log only), `"off"` |
| `fail_open` | `bool` | `True` | Allow on inspection API errors |
| `timeout` | `int` | `30` | Inspection timeout in seconds |
| `rules` | `list` | `None` | Rules to enable (e.g. `["PII", "Prompt Injection"]`) |
| `user` | `str` | `None` | User identity for audit |
| `src_app` | `str` | `None` | Source application name |
| `on_violation` | `callable` | `None` | `(InspectResponse, direction) -> None` callback |

### How it works

```
User message
    │
    ▼
┌──────────────────────────────────────────────────┐
│  before_model hook                               │
│  → ChatInspectionClient.inspect_conversation()   │
│  → if not safe and mode="enforce": jump_to=end   │
└──────────────────────────────────────────────────┘
    │
    ▼
  LLM call
    │
    ▼
┌──────────────────────────────────────────────────┐
│  after_model hook                                │
│  → ChatInspectionClient.inspect_conversation()   │
│  → if not safe and mode="enforce": jump_to=end   │
└──────────────────────────────────────────────────┘
    │
    ▼
Agent response
```

## Approach 2: `AIDefenseAgentsecMiddleware`

Uses agentsec's `LLMInspector` — gets retry with exponential backoff, fail-open semantics, and can inherit config from `agentsec.protect()`.

```python
from aidefense_langchain import AIDefenseAgentsecMiddleware
from langchain.agents import create_agent

# Option A: explicit config
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
        ),
    ],
)

# Option B: inherit from protect()
from aidefense.runtime import agentsec

agentsec.protect(config="agentsec.yaml", patch_clients=False)

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    middleware=[
        AIDefenseAgentsecMiddleware(mode="enforce"),  # config from protect()
    ],
)
```

### Parameters

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

## Comparison

| Criteria | `AIDefenseMiddleware` | `AIDefenseAgentsecMiddleware` |
|---|:---:|:---:|
| No global state / side effects | ✅ | ❌ (uses `_state`) |
| Self-contained config | ✅ | ⚠️ (can use `protect()`) |
| Built-in retry + backoff | ⚠️ (via `Config`) | ✅ (custom) |
| Built-in fail-open | ✅ (in middleware) | ✅ (in inspector) |
| `inspect_prompt` / `inspect_response` | ✅ | ❌ (`inspect_conversation` only) |
| Inherits from `protect()` YAML | ❌ | ✅ |
| Dependency footprint | Lighter (`aidefense.runtime`) | Heavier (`aidefense.runtime.agentsec`) |

**Recommendation**: Use `AIDefenseMiddleware` for new projects. Use `AIDefenseAgentsecMiddleware` if your app already relies on `agentsec.protect()` config.

## Examples

| # | File | Description |
|---|---|---|
| 1 | `01_chat_client_enforce.py` | ChatClient middleware — enforce mode (block violations) |
| 2 | `02_chat_client_monitor.py` | ChatClient middleware — monitor mode with violation callback |
| 3 | `03_chat_client_with_rules.py` | ChatClient middleware — specific rules (PII, Prompt Injection) |
| 4 | `04_agentsec_enforce.py` | Agentsec middleware — enforce mode with retry config |
| 5 | `05_agentsec_with_protect.py` | Agentsec middleware — config inherited from `protect()` |
| 6 | `06_composed_middleware.py` | AI Defense + custom logging middleware composed together |
| 7 | `07_side_by_side.py` | Same request through both middleware — side-by-side comparison |

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

## Project Structure

```
ai-defense-langchain-middleware/
├── aidefense_langchain/
│   ├── __init__.py
│   ├── middleware_chat_client.py   # AIDefenseMiddleware (ChatInspectionClient)
│   └── middleware_agentsec.py      # AIDefenseAgentsecMiddleware (LLMInspector)
├── examples/
│   ├── 01_chat_client_enforce.py
│   ├── 02_chat_client_monitor.py
│   ├── 03_chat_client_with_rules.py
│   ├── 04_agentsec_enforce.py
│   ├── 05_agentsec_with_protect.py
│   ├── 06_composed_middleware.py
│   └── 07_side_by_side.py
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## License

Apache-2.0
