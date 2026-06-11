"""Ultralytics Platform helpers shared by both pipelines.

Replaces the legacy Ultralytics HUB SDK (``hub_sdk``), which is wound down at the
end of July 2026. The Platform exposes two complementary surfaces:

* the ``ultralytics`` package itself, which resolves ``ul://`` URIs (datasets and
  models) and streams local training to the Platform when ``project`` is set to
  ``<username>/<project>``;
* a REST API at ``https://platform.ultralytics.com/api`` authenticated with an
  ``Authorization: Bearer <ul_...>`` header, used to create projects/models,
  launch cloud-GPU jobs, poll training status and fetch weight download URLs.

This module centralises both: API-key handling (:func:`configure_ultralytics_auth`
for the package, :class:`PlatformClient` for REST) and the ``ul://`` URI builders
that turn the configured slugs into the references the Platform expects.

Platform integration requires ``ultralytics>=8.4.60``.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from config import settings
from sharp.logging_utils import get_logger

logger = get_logger(__name__)

#: Base URL for every Platform REST endpoint.
API_BASE = "https://platform.ultralytics.com/api"

#: Placeholder values shipped in the example config that must be replaced.
_PLACEHOLDERS = {
    "",
    "YOUR_ULTRALYTICS_API_KEY",
    "YOUR_DATASET_ID",
    "your-username",
    "your-dataset",
    "your-project",
    "your-model",
}


def require_api_key(api_key: str | None = None) -> str:
    """Return a usable Platform API key or raise with actionable guidance.

    Args:
        api_key: Explicit key; defaults to ``settings.ultralytics.api_key``.

    Returns:
        The validated key.

    Raises:
        RuntimeError: If no real key is configured.
    """
    key = api_key or settings.ultralytics.get("api_key", "")
    if not key or key in _PLACEHOLDERS:
        raise RuntimeError(
            "Missing Ultralytics Platform API key. Create one at "
            "https://platform.ultralytics.com -> Settings -> API Keys, then copy "
            ".secrets.toml.example to .secrets.toml and set ultralytics.api_key "
            "(or export SHARP_ULTRALYTICS__API_KEY)."
        )
    if not key.startswith("ul_"):
        logger.warning(
            "Platform API keys start with 'ul_' (got one that does not); "
            "a legacy HUB key will stop working when HUB is wound down."
        )
    return key


def configure_ultralytics_auth(api_key: str | None = None) -> None:
    """Register the API key so the ``ultralytics`` package can reach the Platform.

    Sets both the documented ``ULTRALYTICS_API_KEY`` environment variable and the
    persisted ``api_key`` setting (the programmatic equivalent of
    ``yolo settings api_key=...``), so ``ul://`` resolution and metric streaming
    work regardless of which mechanism the installed build reads.

    Args:
        api_key: Explicit key; defaults to ``settings.ultralytics.api_key``.
    """
    key = require_api_key(api_key)
    os.environ.setdefault("ULTRALYTICS_API_KEY", key)
    from ultralytics import settings as ul_settings

    ul_settings.update({"api_key": key})


def _require_slug(name: str, value: str | None) -> str:
    """Return a configured slug or raise if it is missing/placeholder."""
    value = (value or "").strip()
    if not value or value in _PLACEHOLDERS:
        raise RuntimeError(
            f"settings.ultralytics.{name} is not set. Fill it in settings.toml "
            f"(or export SHARP_ULTRALYTICS__{name.upper()}); it is the '{name}' "
            "segment of your ul:// references on the Platform."
        )
    return value


def dataset_uri(dataset: str | None = None) -> str:
    """Build the ``ul://<username>/datasets/<dataset>`` URI for the source dataset.

    Args:
        dataset: A dataset slug, or a full ``ul://`` URI / local path to pass
            through unchanged. Defaults to ``settings.ultralytics.dataset``.

    Returns:
        A ``ul://`` dataset URI (or ``dataset`` verbatim when it is already a URI
        or an existing local path).
    """
    if dataset and (dataset.startswith("ul://") or os.path.exists(dataset)):
        return dataset
    user = _require_slug("username", settings.ultralytics.username)
    slug = _require_slug("dataset", dataset or settings.ultralytics.dataset)
    return f"ul://{user}/datasets/{slug}"


def project_ref(project: str | None = None) -> str:
    """Return ``<username>/<project>`` for ``model.train(project=...)`` streaming."""
    user = _require_slug("username", settings.ultralytics.username)
    slug = _require_slug("project", project or settings.ultralytics.project)
    return f"{user}/{slug}"


def model_uri(project: str | None = None, model: str | None = None) -> str:
    """Build the ``ul://<username>/<project>/<model>`` URI for a trained model.

    Args:
        project: Project slug; defaults to ``settings.ultralytics.project``.
        model: Model slug, or a full ``ul://`` URI to pass through. Defaults to
            ``settings.ultralytics.model``.

    Returns:
        A ``ul://`` model URI loadable with ``YOLO(...)``.
    """
    if model and model.startswith("ul://"):
        return model
    user = _require_slug("username", settings.ultralytics.username)
    proj = _require_slug("project", project or settings.ultralytics.project)
    slug = _require_slug("model", model or settings.ultralytics.model)
    return f"ul://{user}/{proj}/{slug}"


class PlatformClient:
    """Thin authenticated REST client over ``https://platform.ultralytics.com/api``."""

    def __init__(self, api_key: str | None = None) -> None:
        """Build a client, validating the API key eagerly.

        Args:
            api_key: Explicit key; defaults to ``settings.ultralytics.api_key``.
        """
        self.api_key = require_api_key(api_key)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def url(self, path: str) -> str:
        """Return the absolute URL for an API ``path`` (with or without a leading slash)."""
        return f"{API_BASE}/{path.lstrip('/')}"

    def request(
        self, method: str, path: str, *, timeout: float = 60, **kwargs: Any
    ) -> dict[str, Any]:
        """Perform a request and return the parsed JSON body.

        Args:
            method: HTTP verb.
            path: API path relative to :data:`API_BASE`.
            timeout: Per-request timeout in seconds.
            **kwargs: Extra arguments forwarded to ``requests`` (``json``, ``data``,
                ``files``, ``params`` ...).

        Returns:
            The decoded JSON object (``{}`` for an empty body, ``{"raw": ...}`` for
            a non-JSON body).

        Raises:
            RuntimeError: On any non-2xx response, surfacing the status and body.
        """
        response = self.session.request(method, self.url(path), timeout=timeout, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Platform API {method} {path} failed "
                f"({response.status_code}): {response.text[:500]}"
            )
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    def get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send a GET request (see :meth:`request`)."""
        return self.request("GET", path, **kwargs)

    def post(self, path: str, json: Any | None = None, **kwargs: Any) -> dict[str, Any]:
        """Send a POST request (see :meth:`request`)."""
        return self.request("POST", path, json=json, **kwargs)


def platform_client(api_key: str | None = None) -> PlatformClient:
    """Return an authenticated :class:`PlatformClient`."""
    return PlatformClient(api_key)
