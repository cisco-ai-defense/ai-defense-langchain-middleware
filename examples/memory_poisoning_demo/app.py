# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Local Streamlit view of the two real-model memory boundaries."""

from __future__ import annotations

import os

import streamlit as st

from flow_scenario import FlowResult, flow_rule_text, run_flow_comparison
from rules import PROTECT_REPORT_RECIPIENT
from scenario import (
    ATTACKER_RECIPIENT, SAFE_RECIPIENT, VENDOR_BULLETIN,
    DemoResult, model_configured, run_comparison,
)


st.set_page_config(page_title="Memory protection demo", page_icon="🛡️", layout="wide")
st.markdown(
    """<style>
    .block-container {max-width: 1180px; padding-top: 2.4rem}
    h1, h2, h3 {letter-spacing: -0.025em}
    .eyebrow {font-size: .75rem; font-weight: 700; letter-spacing: .14em;
              text-transform: uppercase; color: #5588bd; margin-bottom: .45rem}
    .intro {font-size: 1.08rem; color: #8793a4; max-width: 830px}
    .step {padding: 1rem 1.2rem; border: 1px solid #38506b; border-radius: 12px;
           min-height: 112px; background: #13202d}
    .step b {color: #e7f1fb}
    .step small {display: block; color: #a9b8c8; margin-top: .35rem}
    </style>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">LangChain · Store · Quarry</div>', unsafe_allow_html=True)
st.title("Protect memory at two boundaries")
st.markdown(
    '<p class="intro">A real agent imports an external bulletin, then uses a stored '
    'report recipient. See how Quarry can reject an unsafe Store write and, separately, '
    'stop a tool action influenced by untrusted memory.</p>',
    unsafe_allow_html=True,
)

steps = st.columns(4)
for column, (title, detail) in zip(steps, [
    ("1 · Agent", "Imports a bulletin and sends weekly reports"),
    ("2 · Memory", "A LangGraph Store holds the report recipient"),
    ("3 · Attack", f"External input proposes {ATTACKER_RECIPIENT}"),
    ("4 · Guards", "Check the write and the later tool call"),
]):
    column.markdown(
        f'<div class="step"><b>{title}</b><small>{detail}</small></div>',
        unsafe_allow_html=True,
    )

st.write("")
model = st.text_input(
    "OpenAI model or Azure deployment",
    value=os.getenv("OPENAI_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
)
if not model_configured():
    st.info("Set OpenAI or Azure OpenAI credentials in the shell that starts Streamlit.")
if st.button("Run both defenses", type="primary", disabled=not model_configured()):
    st.session_state.pop("results", None)
    with st.spinner("Running the agent with the same model in each case…"):
        try:
            write = run_comparison(model=model)
            flow = run_flow_comparison(model=model)
            st.session_state.results = (write, flow)
        except Exception as error:
            st.error(f"The run failed ({type(error).__name__}). Run the CLI for details.")


def show_write(result: DemoResult) -> None:
    st.subheader("Guarded" if result.protected else "Unguarded")
    if not result.complete:
        st.warning("Inconclusive: the model skipped an expected tool.")
    elif result.protected and result.stored_recipient == SAFE_RECIPIENT:
        st.success("The proposed write was rejected.")
    elif not result.protected and result.stored_recipient == ATTACKER_RECIPIENT:
        st.error("The bulletin changed the preference.")
    else:
        st.warning("The outcome differed from the expected scenario.")
    st.metric("Proposed write", result.proposed_recipient or "None")
    st.metric("Value in Store", result.stored_recipient or "Missing")
    st.metric("Local simulated send", ", ".join(result.report_recipients) or "None")
    st.caption("Tools: " + (" → ".join(result.tool_calls) or "none"))
    for decision in result.decisions:
        if decision["status"] == "block":
            st.write(f"Quarry: **block** · {', '.join(decision['rules'])}")


def show_flow(result: FlowResult) -> None:
    st.subheader("Enforce" if result.protected else "Monitor only")
    blocked = any(d["operation"] == "tool_call" and d["status"] == "block" for d in result.decisions)
    if not result.complete:
        st.warning("Inconclusive: the model did not read and propose a send.")
    elif result.protected and blocked and not result.simulated_sends:
        st.success("Quarry stopped send_report before the local sink.")
    elif not result.protected and result.simulated_sends == [ATTACKER_RECIPIENT]:
        st.error("The report tool used the untrusted stored address.")
    else:
        st.warning("The outcome differed from the expected scenario.")
    st.metric("Value in Store", result.stored_recipient or "Missing")
    st.metric("Proposed send_report(to=)", ", ".join(result.proposed_sends) or "None")
    st.metric("Local simulated send", ", ".join(result.simulated_sends) or "None")
    read = next((d for d in result.decisions if d["reader_tool"] == "read_report_recipient"), None)
    if read:
        st.caption(
            "Read provenance: " + read["origin_kind"] + " · " + read["trust_level"]
            + " · " + read["provenance_status"]
        )
    for decision in result.decisions:
        if decision["operation"] == "tool_call":
            st.write(f"Quarry: **{decision['status']}** · {', '.join(decision['rules']) or 'no match'}")
    with st.expander("Agent reply"):
        st.write(result.reply)


write_tab, flow_tab = st.tabs(["1 · Protect the Store write", "2 · Protect the tool action"])
with write_tab:
    st.markdown(
        "An external bulletin proposes replacing a safe preference. The application import "
        "tool attempts that write in both runs; the guarded Store scans before commit."
    )
    with st.expander("Bulletin and custom Quarry rule"):
        st.code(VENDOR_BULLETIN, language="text")
        st.code(PROTECT_REPORT_RECIPIENT, language="text")
    if "results" in st.session_state:
        left, right = st.columns(2, gap="large")
        with left:
            show_write(st.session_state.results[0][0])
        with right:
            show_write(st.session_state.results[0][1])

with flow_tab:
    st.markdown(
        f"A prior write has left **{ATTACKER_RECIPIENT}** in Store, marked external and "
        "untrusted. The agent reads it and proposes `send_report(to=...)`. The strict "
        "profile compares the proposed address with the recorded memory read before "
        "the local sink executes."
    )
    with st.expander("Built-in strict-profile Quarry flow rule"):
        st.code(flow_rule_text(), language="text")
    if "results" in st.session_state:
        left, right = st.columns(2, gap="large")
        with left:
            show_flow(st.session_state.results[1][0])
        with right:
            show_flow(st.session_state.results[1][1])

st.caption("The .invalid address cannot receive email. send_report only records a local event; this demo sends no email.")
