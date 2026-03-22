from __future__ import annotations

import json
from typing import Any


def flatten_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        pieces = []
        for item in content:
            maybe_text = _content_part_to_text(item)
            if maybe_text:
                pieces.append(maybe_text)
        if pieces:
            return "\n".join(pieces)
        return json.dumps(content, ensure_ascii=False)

    if isinstance(content, dict):
        maybe_text = _content_part_to_text(content)
        if maybe_text:
            return maybe_text
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def tool_result_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"content": [{"type": "text", "text": content}]}

    if isinstance(content, dict):
        return content

    flattened = flatten_content_text(content)
    return {"content": [{"type": "text", "text": flattened}]}


def _content_part_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item

    if not isinstance(item, dict):
        return ""

    direct_text = item.get("text")
    if isinstance(direct_text, str):
        return direct_text

    inner_content = item.get("content")
    if isinstance(inner_content, str):
        return inner_content

    if isinstance(inner_content, list):
        nested = []
        for part in inner_content:
            maybe_text = _content_part_to_text(part)
            if maybe_text:
                nested.append(maybe_text)
        if nested:
            return "\n".join(nested)

    return ""
