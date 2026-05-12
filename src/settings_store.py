from __future__ import annotations

import json
from typing import Any

from src.config import CACHE_DIR, PROP_MARKET_MAP


SETTINGS_PATH = CACHE_DIR / "user_settings.json"

DEFAULT_SETTINGS = {
    "api_key": "",
    "bookmakers": ["draftkings", "fanduel", "betmgm", "betrivers", "espnbet"],
    "prop_types": list(PROP_MARKET_MAP.keys()),
    "days_ahead": 1,
    "auto_refresh": True,
    "auto_refresh_minutes": 20,
}


def load_user_settings() -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        return DEFAULT_SETTINGS.copy()
    try:
        payload = json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return DEFAULT_SETTINGS.copy()

    settings = DEFAULT_SETTINGS.copy()
    if isinstance(payload, dict):
        settings.update(payload)
    return settings


def save_user_settings(settings: dict[str, Any]) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged = DEFAULT_SETTINGS.copy()
    merged.update(settings)
    SETTINGS_PATH.write_text(json.dumps(merged, indent=2))
    return merged
