# src/ai_summary.py
"""
FIREWATCH - short plain-language summary of OUR computed numbers, via Gemini.

summarize(area_label, facts) returns 3-4 sentences, or None when there is no
API key or anything goes wrong. The model only sees the facts we pass in
(latest month vs median, unusual flag, trend, validation MAE/MAPE) and is told
not to add any number, cause or prediction. The app labels the result
"AI-generated".

Key:   GEMINI_API_KEY  (environment, .env in the repo root, or st.secrets)
Model: GEMINI_MODEL    (optional, default "gemini-2.5-flash")
The key is sent in a request header, never in the URL, and is never printed.
"""
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"
TIMEOUT_S = 30

INSTRUCTION = ("Write 3-4 plain sentences for a farmer or official. "
               "Use only these numbers; do not add any number, cause or prediction.")


def _from_dotenv(name, path=None):
    """Read one value from the repo's .env file (simple KEY=VALUE lines)."""
    path = Path(path) if path else ROOT / ".env"
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip('"').strip("'") or None
    except OSError:
        pass
    return None


def _from_streamlit_secrets(name):
    try:
        import streamlit as st
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:                    # no Streamlit, or no secrets file
        return None


def get_setting(name, default=None):
    """Environment first, then .env, then Streamlit secrets."""
    return (os.environ.get(name) or _from_dotenv(name)
            or _from_streamlit_secrets(name) or default)


def build_prompt(area_label, facts):
    """The full text sent to the model: instruction + our facts as JSON."""
    facts_json = json.dumps(facts, indent=2, sort_keys=True, default=str)
    return (f"{INSTRUCTION}\n\n"
            f"Area: {area_label}\n"
            "All counts are satellite-detected fire hotspots (NASA FIRMS), "
            "not burned area.\n"
            f"FIREWATCH facts:\n{facts_json}")


def _extract_text(payload):
    parts = payload["candidates"][0]["content"]["parts"]
    text = "".join(part.get("text", "") for part in parts).strip()
    return text or None


def summarize(area_label, facts):
    """3-4 sentence summary of `facts`, or None if unavailable."""
    api_key = get_setting("GEMINI_API_KEY")
    if not api_key:
        return None
    model = get_setting("GEMINI_MODEL", DEFAULT_MODEL)
    body = {"contents": [{"parts": [{"text": build_prompt(area_label, facts)}]}]}
    try:
        response = requests.post(
            API_URL.format(model=model),
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json=body,
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        return _extract_text(response.json())
    except Exception:                    # network, HTTP, quota or format problem
        return None
