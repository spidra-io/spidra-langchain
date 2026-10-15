"""Shared utilities for the Spidra integration.

This module centralizes API key / URL resolution, client construction, and
shared helpers. Every loader and tool routes through here so credential
handling is applied uniformly.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.utils import convert_to_secret_str
from spidra import AsyncSpidra, Spidra

#: Default Spidra API base URL when none is supplied via argument or env var.
DEFAULT_API_URL: str = "https://api.spidra.io/api"


def resolve_api_key(api_key: str | None) -> str:
    """Resolve the Spidra API key from the argument or ``SPIDRA_API_KEY`` env var."""
    return api_key or os.environ.get("SPIDRA_API_KEY") or ""


def resolve_api_url(api_url: str | None) -> str:
    """Resolve the Spidra API URL, falling back to :data:`DEFAULT_API_URL`."""
    return api_url or os.environ.get("SPIDRA_API_URL") or DEFAULT_API_URL


def create_client(api_key: str | None = None, api_url: str | None = None) -> Spidra:
    """Build a Spidra client from resolved credentials.

    Args:
        api_key: Explicit API key, or ``None`` to read from the environment.
        api_url: Explicit API URL, or ``None`` to read from the environment.

    Returns:
        A configured :class:`spidra.Spidra` client.
    """
    return Spidra(
        api_key=resolve_api_key(api_key),
        api_url=resolve_api_url(api_url),
    )


def create_async_client(api_key: str | None = None, api_url: str | None = None) -> AsyncSpidra:
    """Build an AsyncSpidra client from resolved credentials.

    Args:
        api_key: Explicit API key, or ``None`` to read from the environment.
        api_url: Explicit API URL, or ``None`` to read from the environment.

    Returns:
        A configured :class:`spidra.AsyncSpidra` client.
    """
    return AsyncSpidra(
        api_key=resolve_api_key(api_key),
        api_url=resolve_api_url(api_url),
    )


def initialize_client(values: dict) -> dict:
    """Validate environment and instantiate the client for a pydantic tool.

    Resolves the API key into a ``SecretStr``, resolves the API URL, and
    stores ready sync and async clients. Intended for use from a
    ``model_validator(mode="before")``.

    Args:
        values: The raw init values for a pydantic model.

    Returns:
        The updated values dict with ``api_key``, ``api_url``, ``client`` and ``async_client`` set.
    """
    api_key = resolve_api_key(values.get("api_key"))
    values["api_key"] = convert_to_secret_str(api_key)
    api_url = resolve_api_url(values.get("api_url"))
    values["api_url"] = api_url
    raw_key = values["api_key"].get_secret_value()
    values["client"] = Spidra(
        api_key=raw_key,
        api_url=api_url,
    )
    values["async_client"] = AsyncSpidra(
        api_key=raw_key,
        api_url=api_url,
    )
    return values


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values from a params dict.

    Args:
        params: Keyword arguments destined for a Spidra client method.

    Returns:
        A new mapping without ``None`` values.
    """
    return {k: v for k, v in params.items() if v is not None}


def to_dict(obj: Any) -> dict[str, Any]:
    """Coerce a Spidra response field into a plain ``dict``.

    Args:
        obj: A dataclass, pydantic model, dict, or arbitrary object (or ``None``).

    Returns:
        A plain dict; empty when ``obj`` is ``None``.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if v is not None}
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return obj.model_dump(exclude_none=True)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if v is not None}
    return {"value": str(obj)}
