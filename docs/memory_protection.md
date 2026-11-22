# Local memory protection

The memory middleware uses open-source Quarry rules to inspect memory at the
point a LangChain agent writes or reads it. This is local inspection. Today's
AI Defense `/inspect` API does not accept Quarry extension inputs or these
memory rules.

## Installation

Use Python 3.11–3.13 with the pinned Quarry commit. Its declared-input API is in its public source
but is not yet in the 0.1.0 PyPI release. Until 0.1.1 is released:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install 'git+https://github.com/cisco-ai-defense/project-quarry.git@9c5db4aa7cf843e1e2a78bf973fe106a30d446a0'
pip install -e '.[memory,examples]'
```

The `memory` extra will install Quarry directly after that release. The other
LangChain middleware classes remain usable without this extra.

## Use it with a LangChain agent

```python
from aidefense_langchain.memory import AIDefenseMemoryMiddleware, memory_origin
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

guard = AIDefenseMemoryMiddleware(
    profile="balanced",
    protected_keys={(('reports', 'preferences'), 'report_recipient')},
    on_decision=lambda decision: print(decision.status, decision.operation, decision.rules),
)
store = guard.protect_store(InMemoryStore())
agent = create_agent(
    model=model,
    tools=tools,
    store=store,
    checkpointer=guard.protect_checkpointer(InMemorySaver()),
    middleware=[guard],
)

# Only trusted application code may assign origin and trust. A saved note
# cannot declare itself trusted.
with memory_origin("user", "trusted"):
    store.put(("reports", "preferences"), "report_recipient", {"content": "reports@example.com"})
```

Use the wrapped Store everywhere an agent or tool accesses that memory. The
checkpointer wrapper is separate because LangGraph keeps conversation history
there, not in the Store.

For a tool that can perform a side effect based on remembered content, use
`profile="strict", gated_tools={"send_report"}`. The bundled flow rule checks
whether an unverified address read earlier in that agent run is copied into
`send_report(to=...)`. The middleware returns an error `ToolMessage` before the
tool runs when the strict rule matches. This is an exact-copy signal, not proof
of causation or a general authorization check. Only text actually returned by
an earlier memory-reading tool counts; parallel read/send calls from one model
response do not create a false ordering. The guard checks that the next model
call still sees the inspected tool result. Place this middleware last in
`middleware=[...]` so its model hook sees the final request. Adapt the flow rule to your
tool name and arguments with `tool_rule_texts=[your_quarry_rule]`.

Flow evidence lives in the middleware instance for one agent invocation. A
checkpoint resumed in another process has no prior trace; a gated tool fails
closed until that run establishes fresh evidence. Persisting a cross-process
trace is separate work.

The [live demo](../examples/memory_poisoning_demo/README.md) runs the same
agent without and with protection, using a local simulated report sink.

## What is covered

| Resource | Writes | Reads | Origin across runs | Pre-tool flow |
| --- | --- | --- | --- | --- |
| LangGraph `BaseStore` (tested: `InMemoryStore`) | `put`, `delete`, `batch`, sync and async | `get`, `search`, sync and async | SHA-256-bound sidecar | Yes, for reads inside the guarded agent run |
| LangGraph checkpointer (tested: `InMemorySaver`) | New/changed `messages` and pending message writes | Saved messages on replay | No durable origin on replay | No |
| LangChain `BaseChatMessageHistory` | Add and clear, sync and async | Saved messages | Unknown | No |

The chat-history wrapper is for existing `RunnableWithMessageHistory` apps;
LangChain now recommends LangGraph persistence for new agents. Use
`guard.protect_history(backing, session_id)` in a history factory.

Read matches are **observational** in every profile. They do not erase memory
or silently return “not found.” A value the scanner cannot safely represent
causes an explicit inspection error. Blocked Store writes do not reach the
backing Store; inside an agent tool they become a recoverable error result.

## Rules and sensitivity

The bundled [rules](../aidefense_langchain/memory/quarry/rules/) are Quarry `.rule`
files selected by `profiles.json`. Profiles add increasingly broad signals:

| Profile | Enforced writes | Observed writes/reads | Pre-tool flow |
| --- | --- | --- | --- |
| Permissive | Unverified writes to host-designated protected keys | Role spoofing, authority laundering, protected reads, provenance mismatch | Off |
| Balanced (default) | Permissive plus precise role spoofing and authority laundering | Future-behavior, sensitive-tool and secret transfer, instruction and encoded override, protected reads | Observe exact copied recipient |
| Strict | Balanced plus broader instruction, persistence, tool redirection, exfiltration, encoded and punctuated patterns | Protected reads and provenance mismatch | Block exact copied recipient |

`mode="monitor"` observes matches without blocking; `mode="off"` delegates
unchanged. Custom `rule_texts=[...]` replace the memory-stage profile rules.
Custom `tool_rule_texts=[...]` replace its tool-stage flow rules. Keep rules
about your application's protected keys and side-effect tools specific to
their verified facts. The included signatures can match benign quoted attack
examples, especially in strict mode.

An optional `include_semantic_judge=True` adds a **monitor-only** model rule.
The caller must supply Quarry `model_config` and `model_adapters`; no endpoint,
key, or model call is included by default. A judge timeout leaves deterministic
decisions intact and sets `decision.monitor_incomplete=True` for observability.

## Boundaries and layout

`aidefense_langchain/memory/backends/` wraps Store, checkpoint, and chat
history objects and tracks provenance. `memory/middleware.py` owns LangChain
hooks, while `memory/flow.py` records released reads for one agent run.
`memory/quarry/` converts those facts to Quarry declared inputs, runs the
rules, and packages the profile and rule files. No internal Augur package or
AgentSec patcher is required.

The Store sidecar records the writer origin, tool name, and a canonical value
digest. A missing or mismatched digest lowers trust to `unknown` on read. The
digest detects an out-of-band value change **only when the sidecar was not also
changed**; it does not authenticate a compromised backing Store. Keep the raw
Store reference away from agent tools. The current tests validate in-memory
backends; durable Stores and savers need their own atomicity, paging, and TTL
checks. The sidecar requests the same TTL as its value, but expiration timing
still depends on the backing Store.
