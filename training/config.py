"""Central configuration loader for the SHARP training pipelines.

Uses dynaconf so values can come from ``settings.toml`` (non-secret),
``.secrets.toml`` (API keys, gitignored), or ``SHARP_``-prefixed environment
variables. Import the singleton ``settings`` everywhere instead of hard-coding
values::

    from config import settings

    print(settings.train.epochs)
    print(settings.ultralytics.dataset_id)
"""

from __future__ import annotations

from pathlib import Path

from dynaconf import Dynaconf

# Directory containing this file, so the app works regardless of the cwd.
BASE_DIR = Path(__file__).resolve().parent

settings = Dynaconf(
    envvar_prefix="SHARP",
    root_path=str(BASE_DIR),
    settings_files=["settings.toml", ".secrets.toml"],
    merge_enabled=True,
)
"""Singleton settings object shared by both pipelines."""
