# Local memory protection

The memory middleware uses open-source Quarry rules to inspect memory at the
point a LangChain agent writes or reads it. This is local inspection. Today's
AI Defense `/inspect` API does not accept Quarry declared inputs or these
memory rules.

## Installation

Use Python 3.11–3.13. Memory protection needs Quarry's declared-input API, which
is on Quarry's `main` branch but not yet in a PyPI release (0.1.1 predates it and
reports the same version number). Until a release includes it, install the pinned
commit first:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install 'git+https://github.com/cisco-ai-defense/project-quarry.git@9c5db4aa7cf843e1e2a78bf973fe106a30d446a0'
pip install -e '.[memory,examples]'
```

With an older Quarry build, creating the middleware raises an error that points
here. The other LangChain middleware classes remain usable without this extra.

## Use it with a LangChain agent

```python
from aidefense_langchain.memory import AIDefenseMemoryMiddleware, memory_origin
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

guard = AIDefenseMemoryMiddleware(
    profile="balanced",
    protected_keys={(("reports", "preferences"), "report_recipient")},
    on_decision=lambda decision: print(decision.status, decision.kind, decision.operation, decision.rules),
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
whether an email address from unverified memory, read earlier in that agent run,
appears in `send_report(to=...)`. The middleware returns an error `ToolMessage`
before the tool runs when the strict rule matches.

This is a copy signal, not proof of causation or a general authorization check.
Quarry cannot compare a captured value for equality, so a recipient that merely
contains the address (`xops@example.com` for `ops@example.com`) also matches.
Only text actually returned by an earlier memory-reading tool counts; parallel
read/send calls from one model response do not create a false ordering.
Evidence follows the tool result the next model call actually sees: if other
middleware, such as `PIIMiddleware` with `apply_to_tool_results=True`, rewrites
that result, only memory text that survives the rewrite counts. A redacted
address is no longer evidence, so a send to it is not flagged by this rule.
Place this middleware last in `middleware=[...]` so its model hook sees the
final request. Adapt the flow rule to your tool name and arguments with
`tool_rule_texts=[your_quarry_rule]`.

Flow evidence lives in the middleware instance for one agent invocation. Each
new invocation starts with no evidence, even on an existing thread: a memory
read from an earlier turn does not count, so a gated call in a later turn is
only flagged if that turn reads the value again. An interrupted run resumed in
another process, or by a new middleware instance, has no trace at all; every
gated call in the rest of that run fails closed in enforce mode. Persisting a
cross-process trace is separate work.

The [live demo](../examples/memory_poisoning_demo/README.md) runs the same
agent without and with protection, using a local simulated report sink.

## What is covered

| Resource | Writes | Reads | Origin across runs | Pre-tool flow |
| --- | --- | --- | --- | --- |
| LangGraph `BaseStore` (tested: `InMemoryStore`) | `put`, `delete`, `batch`, sync and async | `get`, `search`, sync and async | SHA-256-bound sidecar | Yes, for reads inside the guarded agent run |
| LangGraph checkpointer (tested: `InMemorySaver`) | New/changed `messages` and pending message writes, including `RemoveMessage`, from dict, pydantic, dataclass, `Command`, and `Overwrite` inputs | Saved messages on replay | No durable origin on replay | No |
| LangChain `BaseChatMessageHistory` | Add and clear, sync and async | Saved messages | Unknown | No |

The chat-history wrapper is for existing `RunnableWithMessageHistory` apps;
LangChain now recommends LangGraph persistence for new agents. Use
`guard.protect_history(backing, session_id)` in a history factory. Provider
fields such as `additional_kwargs` and non-text content blocks are scanned as
data; empty fields are ignored. A read scans the newest 4096 messages and
reports one `error` decision for any older ones, which are still returned.

Read matches are **observational** in every profile. They do not erase memory
or silently return “not found.” A value the scanner cannot safely represent
causes an explicit inspection error. Blocked Store writes do not reach the
backing Store; inside an agent tool they become a recoverable error result.

## Memory events (schema v1)

Every backend describes what it does as a framework-neutral **memory event**,
validated by the Pydantic models `MemoryMutation` and `MemoryRead` (exported from
`aidefense_langchain.memory`). Quarry sees two declared input sections, always
both present (an empty one is `[]`):

| Section | Model | Operations |
| --- | --- | --- |
| `memory.mutation_requested` | `MemoryMutation` | `write`, `append`, `delete`, `clear` |
| `memory.read_returned` | `MemoryRead` | `get`, `list`, `search` |

Each event has these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Always `1` |
| `operation` | One of the operations above |
| `target` | `kind` (`store`, `checkpoint`, `session`), `framework`, `scope` (`item`, `collection`, `all`), `namespace`, `namespace_path` (`/a/b/`), `key`, `session_id`, `item_index` |
| `source` | Who acts now: `{type, name, call_id}`; a tool, the application, or a framework component |
| `content` | `text` (readable text for rules), `format` (`text`, `json`, `message`, `none`), `truncated` |
| `provenance` | `origin`, `trust`, `status` (`new`, `verified`, `missing`, `mismatch`), `writer`, `writer_run_id` |
| `target_class` | `protected`, `ordinary`, or `unknown` |
| `query` | Search text, on `search` reads only |

How each backend fills them:

| Backend | Writes | Reads | Target identity |
| --- | --- | --- | --- |
| Store | `write`, `delete` | `get`, `search` | `namespace`, `key` |
| Checkpointer | `write`, `delete` (a `RemoveMessage`) | `list` (replay) | `session_id` = thread ID, `namespace` = `[checkpoint_ns or "root", channel]`, `key` = message ID when present, `item_index` |
| Chat history | `append`, `clear` (`scope: "collection"`) | `list` | `session_id`, `key` = message ID when present, `item_index` |

`delete` and `clear` carry no content (`format: "none"`). A read keeps origin
and trust only when its provenance is `verified`; `missing` and `mismatch` reads
are `unknown`.

### Content modes

By default, long values keep only their first 12 KB and last 4 KB of text and
are marked `content.truncated`. Padding could hide a payload in the skipped
middle, so two rules report shortened content that is not trusted: on a
protected target (enforced in balanced and strict) and on any other target
(monitor). Pass `full_text_scan=True` to `AIDefenseMemoryMiddleware` to scan all
of it; memory values are capped at 1 MiB either way.

### Writing a rule

Rules read the nested paths directly. Skip trusted content with
`e.provenance.trust != "trusted"`, and match namespaces through `namespace_path`:

```text
rule "Vendor importer cannot change the report recipient"
condition:
    any e in memory.mutation_requested where
        e.target.namespace_path == "/reports/preferences/" and
        e.target.key == "report_recipient" and
        e.source.type == "tool" and
        e.source.name == "ingest_vendor_bulletin"
```

A rule that should look at both writes and reads checks both sections:
`(any e in memory.mutation_requested where ...) or (any e in memory.read_returned where ...)`.

### Decisions

`on_decision` receives a `MemoryDecision` for each boundary check. Memory
content is never included. `operation` is the v1 operation, `kind` is the
target kind (`store`, `checkpoint`, `session`, or `tool` for a pre-tool flow
check), and `session_id` and `item_index` identify checkpoint and history
items. `status` is `allow`, `monitor`, `block`, or `error`.

## Rules and sensitivity

The bundled [rules](../aidefense_langchain/memory/quarry/rules/) are Quarry `.rule`
files selected by `profiles.json`. Profiles add increasingly broad signals:

| Profile | Enforced writes | Observed writes/reads | Pre-tool flow |
| --- | --- | --- | --- |
| Permissive | Unverified writes to host-designated protected keys | Role spoofing, authority laundering, protected reads, provenance mismatch, shortened content | Off |
| Balanced (default) | Permissive plus shortened content on protected keys, precise role spoofing and authority laundering | Future-behavior, sensitive-tool and secret transfer, instruction and encoded override, protected reads, provenance mismatch, other shortened content | Observe copied recipient |
| Strict | Balanced plus broader instruction, persistence, tool redirection, exfiltration, encoded and punctuated patterns | Protected reads, provenance mismatch, other shortened content | Block copied recipient |

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

`aidefense_langchain/memory/events.py` defines the v1 models and content
normalization. `memory/backends/` wraps Store, checkpoint, and chat history
objects and tracks provenance. `memory/middleware.py` owns LangChain hooks,
while `memory/flow.py` records released reads for one agent run.
`memory/quarry/` declares the v1 inputs to Quarry, runs the rules, and packages
the profile and rule files.

The Store sidecar records the writer's origin, trust, and actor, plus a
SHA-256 digest of the canonical value. A missing or mismatched digest lowers
trust to `unknown` on read. The digest is not keyed: it detects an out-of-band
value change **only when the sidecar was not also changed**, and does not
authenticate a compromised backing Store. Keep the raw Store reference away
from agent tools. The current tests validate in-memory backends; durable
Stores and savers need their own atomicity, paging, and TTL checks. The sidecar
requests the same TTL as its value, but expiration timing still depends on the
backing Store.
