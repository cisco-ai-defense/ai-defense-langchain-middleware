# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""The demo's application-specific Quarry policy.

Each Store write reaches Quarry as a v1 ``memory.mutation_requested`` event.
This rule is deliberately about a trust boundary, not an attacker email
address: an external bulletin importer may not change the report recipient
preference.
"""

PROTECT_REPORT_RECIPIENT = '''rule "External bulletin cannot change report recipient"
metadata:
    category = "memory_integrity"
    severity = "high"
condition:
    any e in memory.mutation_requested where
        e.operation == "write" and
        e.target.key == "report_recipient" and
        e.source.type == "tool" and
        e.source.name == "ingest_vendor_bulletin"'''
