"""Shared Ultralytics HUB client factory for the cloud pipeline."""

from __future__ import annotations

from hub_sdk import HUBClient

from config import settings


def hub_client() -> HUBClient:
    """Return an authenticated :class:`hub_sdk.HUBClient`.

    Raises:
        RuntimeError: If the Ultralytics API key is not configured.
    """
    api_key = settings.ultralytics.api_key
    if not api_key or api_key == "YOUR_ULTRALYTICS_API_KEY":
        raise RuntimeError(
            "Missing Ultralytics API key. Copy .secrets.toml.example to .secrets.toml "
            "and set ultralytics.api_key (or export SHARP_ULTRALYTICS__API_KEY)."
        )
    return HUBClient({"api_key": api_key})
