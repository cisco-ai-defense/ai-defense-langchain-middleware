# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""The demo's application-specific Quarry policy.

The memory extension supplies ``memory.events``. This rule is deliberately
about a trust boundary, not an attacker email address: an external bulletin
importer may not change the report recipient preference.
"""

PROTECT_REPORT_RECIPIENT = '''rule "External bulletin cannot change report recipient"
metadata:
    category = "memory_integrity"
    severity = "high"
condition:
    any event in memory.events where
        event.operation == "write" and
        event.key == "report_recipient" and
        event.writer_tool == "ingest_vendor_bulletin"'''
