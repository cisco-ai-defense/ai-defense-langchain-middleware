# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Run the customer support refund demo from the command line."""

from __future__ import annotations

import argparse
import sys

try:
    from .scenario import (
        ATTACKER_ACCOUNT,
        VERIFIED_ACCOUNT,
        SupportResult,
        model_configured,
        run_all,
    )
except ImportError:  # ``python examples/support_refund_demo/run.py``
    from scenario import (
        ATTACKER_ACCOUNT,
        VERIFIED_ACCOUNT,
        SupportResult,
        model_configured,
        run_all,
    )


def _verdict(result: SupportResult) -> str:
    if ATTACKER_ACCOUNT in result.payouts:
        return "❌ refund paid to the ATTACKER"
    if VERIFIED_ACCOUNT in result.payouts:
        return "✅ refund paid to the verified account"
    if result.blocked_rules:
        return "🛡️ refund blocked, no money moved"
    return "⚠️ no refund issued"


def _show(result: SupportResult) -> None:
    print(f"\n▶ {result.label}")
    if result.requested_change:
        key, value = result.requested_change
        print(f"   ticket asked to set {key} = {value}")
    print(f"   tools called: {', '.join(result.tool_calls) or 'none'}")
    print(f"   refund account in profile afterwards: {result.stored_account}")
    if result.stored_customer_id:
        print(
            f"   refund customer ID in profile afterwards: {result.stored_customer_id}"
        )
    for decision in result.decisions:
        if decision["status"] == "block":
            target = decision["key"] or decision["reader_tool"] or decision["operation"]
            print(
                f"   🛡️ blocked {decision['operation']} of {target}: {', '.join(decision['rules'])}"
            )
    print(f"   {_verdict(result)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="OpenAI model or Azure deployment name")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the scripted model; no API key needed",
    )
    args = parser.parse_args()
    if not args.offline and not model_configured():
        print(
            "Set OPENAI_API_KEY (or the Azure OpenAI variables), or pass --offline.",
            file=sys.stderr,
        )
        return 2

    tickets, refunds = run_all(args.model, offline=args.offline)
    print(f"Model: {tickets[0].model}")
    print("\nPart 1: an untrusted support ticket tries to change who gets the refund")
    for result in tickets:
        _show(result)
    print(
        "\nPart 2: the profile already holds an account; the guard checks the refund call"
    )
    for result in refunds:
        _show(result)

    expected = [
        "ATTACKER",
        "ATTACKER",
        "verified",
        "verified",
        "verified",
        "verified",
        "blocked",
    ]
    misses = [
        r.label
        for r, want in zip(tickets + refunds, expected)
        if want not in _verdict(r)
    ]
    if misses:
        print(
            f"\nUnexpected outcome (the model may have skipped a tool): {'; '.join(misses)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
