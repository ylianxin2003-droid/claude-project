"""
Application configuration.

Loads settings from (in order of priority):
1. Streamlit Cloud Secrets (``st.secrets``) — used when deployed
2. Local ``.env`` file — used for development
3. System environment variables

No API token is hardcoded in this file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_PATH = PROJECT_ROOT / ".env"

SERENE_API_BASE_URL: str = ""
SERENE_API_TOKEN: str = ""
SERENE_API_TIMEOUT: int = 30
SERENE_AUTH_SCHEME: str = "Token"

# ── Map / grid / cache settings ──────────────────────────────────────────
MAP_RESOLUTION_DEFAULT: float = 2.5
CACHE_DIR: str = str(PROJECT_ROOT / "data" / "cache")
HISTORY_DIR: str = str(PROJECT_ROOT / "data" / "history")
DEFAULT_REGION: str = "global"
SERENE_BATCH_SIZE: int = 500
DEFAULT_TIME_STEP_HOURS: int = 12

# ── Hazard detection thresholds (prototype, not official ICAO) ───────────
# value_*  → threshold on the variable value itself
# gradient_* → threshold on spatial gradient magnitude (value per degree)
# change_rate_* → threshold on temporal change rate (value per hour)
HAZARD_THRESHOLDS: dict[str, dict[str, float]] = {
    "TEC": {
        "value_watch": 50.0,
        "value_warning": 80.0,
        "value_severe": 120.0,
        "gradient_watch": 5.0,
        "gradient_warning": 10.0,
        "gradient_severe": 20.0,
        "change_rate_watch": 10.0,
        "change_rate_warning": 20.0,
        "change_rate_severe": 40.0,
    },
    "MUF3000_depression": {
        "value_watch": 0.2,
        "value_warning": 0.4,
        "value_severe": 0.6,
        "gradient_watch": 0.05,
        "gradient_warning": 0.10,
        "gradient_severe": 0.20,
        "change_rate_watch": 0.1,
        "change_rate_warning": 0.2,
        "change_rate_severe": 0.4,
    },
    "foF2_depression": {
        "value_watch": 0.2,
        "value_warning": 0.4,
        "value_severe": 0.6,
        "gradient_watch": 0.05,
        "gradient_warning": 0.10,
        "gradient_severe": 0.20,
        "change_rate_watch": 0.1,
        "change_rate_warning": 0.2,
        "change_rate_severe": 0.4,
    },
}


def _load_env_file() -> None:
    """Load .env from this repository root only."""
    if not _ENV_PATH.exists():
        return

    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le"):
        try:
            load_dotenv(_ENV_PATH, encoding=encoding)
            return
        except UnicodeDecodeError:
            continue

    logger.warning(
        "Could not decode .env — re-save as UTF-8. "
        "Using environment / Streamlit secrets only."
    )


def _read_os_env() -> None:
    """Read from OS environment (and values set by dotenv)."""
    global SERENE_API_BASE_URL, SERENE_API_TOKEN, SERENE_API_TIMEOUT, SERENE_AUTH_SCHEME

    SERENE_API_BASE_URL = os.getenv("SERENE_API_BASE_URL", SERENE_API_BASE_URL).strip()
    SERENE_API_TOKEN = os.getenv("SERENE_API_TOKEN", SERENE_API_TOKEN).strip()
    SERENE_API_TIMEOUT = int(os.getenv("SERENE_API_TIMEOUT", str(SERENE_API_TIMEOUT)) or "30")
    SERENE_AUTH_SCHEME = (
        os.getenv("SERENE_AUTH_SCHEME", SERENE_AUTH_SCHEME).strip() or "Token"
    )


def _get_secret(secrets: object, key: str) -> str | None:
    """Read a key from flat or ``[serene]`` nested Streamlit secrets."""
    try:
        if key in secrets:
            return str(secrets[key]).strip()
        if "serene" in secrets and key in secrets["serene"]:
            return str(secrets["serene"][key]).strip()
    except Exception:
        return None
    return None


def _load_streamlit_secrets() -> None:
    """Override with Streamlit Cloud secrets when the app is running on Streamlit."""
    try:
        import streamlit as st
    except ImportError:
        return

    try:
        secrets = st.secrets
    except Exception:
        return

    global SERENE_API_BASE_URL, SERENE_API_TOKEN, SERENE_API_TIMEOUT, SERENE_AUTH_SCHEME

    base = _get_secret(secrets, "SERENE_API_BASE_URL")
    token = _get_secret(secrets, "SERENE_API_TOKEN")
    timeout = _get_secret(secrets, "SERENE_API_TIMEOUT")
    scheme = _get_secret(secrets, "SERENE_AUTH_SCHEME")

    if base:
        SERENE_API_BASE_URL = base
    if token:
        SERENE_API_TOKEN = token
    if timeout:
        SERENE_API_TIMEOUT = int(timeout)
    if scheme:
        SERENE_AUTH_SCHEME = scheme


def reload_config() -> None:
    """Reload settings (.env + Streamlit secrets). Call once at app startup."""
    _load_env_file()
    _read_os_env()
    _load_streamlit_secrets()


reload_config()


def validate_config() -> list[str]:
    """Return user-facing warnings when SERENE settings are missing."""
    messages: list[str] = []

    if not SERENE_API_BASE_URL:
        messages.append(
            "SERENE_API_BASE_URL is not set. "
            "For local dev: copy .env.example to .env. "
            "For Streamlit Cloud: add secrets in the app settings."
        )

    if not SERENE_API_TOKEN:
        messages.append(
            "SERENE_API_TOKEN is not set. "
            "This dashboard requires SERENE API access or existing cache. "
            "Local sample fallback is disabled."
        )

    return messages
