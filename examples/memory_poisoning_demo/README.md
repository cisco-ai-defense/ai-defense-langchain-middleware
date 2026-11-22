# Memory poisoning demo: two boundaries

A LangChain agent stores a report recipient and later uses it in `send_report`. An external vendor bulletin proposes changing the recipient from `reports@company.example` to `audit@attacker.invalid`. The demo shows two separate protections:

1. **Store write:** a custom Quarry rule rejects the bulletin importer's attempted change before it reaches the Store.
2. **Memory to tool:** an untrusted recipient is already in Store. The agent reads it and proposes `send_report(to=...)`; the strict profile stops the tool before its side effect.

Each comparison uses the same real model and task. `send_report` only records a local simulated send; it never sends email.

## Run

Use Python 3.11–3.13 with the pinned Quarry commit. This example uses Python 3.13. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
# Temporary until the Quarry declared-input release is published (the memory
# extra resolves against this installation):
pip install 'git+https://github.com/cisco-ai-defense/project-quarry.git@9c5db4aa7cf843e1e2a78bf973fe106a30d446a0'
pip install -e '.[examples,memory]'
export OPENAI_API_KEY='your-own-key'
python examples/memory_poisoning_demo/run.py
```

For Azure OpenAI, set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and `OPENAI_API_VERSION` instead of `OPENAI_API_KEY`. Set `OPENAI_MODEL` or pass `--model` to choose a direct OpenAI model; for Azure, `--model` overrides the deployment. For another OpenAI-compatible endpoint, set `OPENAI_BASE_URL` too. The CLI exits nonzero if the model skips a required tool or either expected outcome does not occur.

For the optional local UI:

```bash
pip install streamlit
streamlit run examples/memory_poisoning_demo/app.py
```

For a shorter, step-by-step walkthrough, open the [Jupyter notebook](live_memory_demo.ipynb):

```bash
pip install jupyterlab
jupyter lab examples/memory_poisoning_demo/live_memory_demo.ipynb
```

Run its cells in order. It uses the same model credentials, shows the middleware
wiring and Quarry rule, then runs the same agent task without and with protection.

## Expected results

| Boundary | Comparison | Expected result |
| --- | --- | --- |
| Store write | Unguarded vs guarded | Guard rejects the proposed change; later send uses the safe address. |
| Memory to tool | Strict monitor vs strict enforce | Both see the untrusted stored address and propose the same send; enforce blocks the local sink. |

The [custom write rule](rules.py) targets the **external import tool writing a preference**. It does not depend on the attacker's address or a generic phrase. The second case uses the packaged `memory_to_tool` rule selected by `profile="strict"`. A host-asserted `memory_origin("external", "untrusted")` is bound to the stored value. Quarry compares the exact address read from memory with the proposed tool call. Reads remain visible to the agent; the tool call is the enforcement point.

## Integration shape

```python
from aidefense_langchain.memory import AIDefenseMemoryMiddleware, memory_origin
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

guard = AIDefenseMemoryMiddleware(profile="strict", gated_tools={"send_report"})
store = guard.protect_store(InMemoryStore())
with memory_origin("external", "untrusted"):
    store.put(("reports", "preferences"), "report_recipient", {"content": recipient})
agent = create_agent(model=model, tools=tools, store=store, middleware=[guard])
```

The Store wrapper inspects writes and reads. The middleware records reads within one invocation, adds tool provenance, and checks selected tool calls before dispatch. This example runs local Quarry rules; it does not assume `/inspect` supports memory rules.
