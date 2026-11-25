# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""The support desk's Quarry policy.

Both rules describe a trust boundary rather than an attacker's account number:
the public ticket importer may not change where refunds are paid, and a refund
may not be paid to an account that came from unverified memory.
"""

PROTECT_REFUND_DETAILS = '''rule "Support tickets cannot change refund payout details"
metadata:
    category = "memory_integrity"
    severity = "high"
condition:
    any e in memory.mutation_requested where
        (e.operation == "write" or e.operation == "delete") and
        (e.target.key == "refund_account" or e.target.key == "refund_customer_id") and
        e.source.type == "tool" and
        e.source.name == "ingest_support_ticket"'''

# The account captured from memory is a substring match on the call argument.
# The window spans the longest flow: 64 reads, the model message, then the call.
REFUND_TO_UNVERIFIED_ACCOUNT = """rule "Refund paid to an unverified account from memory"
metadata:
    category = "memory_to_tool"
    severity = "high"
strings:
    $account = "ACCT-[0-9]{8}" regex
condition:
    (any call in get_tool_calls("issue_refund") where
        call.arguments.account contains acct)
    after(within: 65)
    (any r in memory.read_returned where
        r.provenance.origin != "user" and r.provenance.origin != "system" and
        r.provenance.trust != "trusted" and
        $account in r.content.text as acct)"""
