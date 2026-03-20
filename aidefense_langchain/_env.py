from __future__ import annotations

import os
from typing import Any, Mapping


_REGION_ALIASES = {
    "us": "us-west-2",
    "eu": "eu-central-1",
    "apj": "ap-northeast-1",
}


def _env_values(env: Mapping[str, str] | None = None) -> dict[str, str]:
    return dict(os.environ if env is None else env)


def normalize_region(region: str | None) -> str:
    if not region:
        return "us-west-2"

    raw = region.strip()
    return _REGION_ALIASES.get(raw.lower(), raw)


def direct_kwargs_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = _env_values(env)
    api_key = values.get("AIDEFENSE_API_KEY")
    if not api_key:
        raise ValueError("AIDEFENSE_API_KEY is required")

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "region": normalize_region(values.get("AIDEFENSE_REGION", "us-west-2")),
    }

    mode = values.get("AIDEFENSE_MODE")
    if mode:
        kwargs["mode"] = mode

    fail_open = _parse_bool(values.get("AIDEFENSE_FAIL_OPEN"))
    if fail_open is not None:
        kwargs["fail_open"] = fail_open

    timeout = _parse_int(values.get("AIDEFENSE_TIMEOUT"))
    if timeout is not None:
        kwargs["timeout"] = timeout

    return kwargs


def agentsec_kwargs_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = _env_values(env)
    kwargs: dict[str, Any] = {}

    api_key = values.get("AIDEFENSE_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key

    endpoint = values.get("AIDEFENSE_ENDPOINT")
    if endpoint:
        kwargs["endpoint"] = endpoint

    mode = values.get("AIDEFENSE_MODE")
    if mode:
        kwargs["mode"] = mode

    fail_open = _parse_bool(values.get("AIDEFENSE_FAIL_OPEN"))
    if fail_open is not None:
        kwargs["fail_open"] = fail_open

    timeout_ms = _parse_int(values.get("AIDEFENSE_TIMEOUT_MS"))
    if timeout_ms is not None:
        kwargs["timeout_ms"] = timeout_ms

    retry_total = _parse_int(values.get("AIDEFENSE_RETRY_TOTAL"))
    if retry_total is not None:
        kwargs["retry_total"] = retry_total

    retry_backoff = _parse_float(values.get("AIDEFENSE_RETRY_BACKOFF"))
    if retry_backoff is not None:
        kwargs["retry_backoff"] = retry_backoff

    return kwargs


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _parse_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)
