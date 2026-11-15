# tests/test_ai_summary.py
"""
Tests for src/ai_summary.py. The Gemini API is MOCKED: no network calls and no
real key are needed. The fake key below is a placeholder, not a credential.
Run from the repo root:   pytest -v tests/test_ai_summary.py
"""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ai_summary  # noqa: E402

FAKE_KEY = "test-placeholder-key"
FACTS = {
    "latest_month": {"month": "2025-11", "hotspots": 120.0,
                      "historical_median_same_month": 80.0, "unusual_above_p90": True},
    "yearly_trend": {"label": "Decreasing", "years": "2003-2025"},
    "harmonization_validation": {"mae_hotspots_per_month": 10.0, "mape_pct": 5.0},
}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _gemini_reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture
def no_key(monkeypatch, tmp_path):
    """No key anywhere: env cleared, .env pointed at an empty folder, no secrets."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(ai_summary, "ROOT", tmp_path)
    monkeypatch.setattr(ai_summary, "_from_streamlit_secrets", lambda name: None)


@pytest.fixture
def with_key(no_key, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)


@pytest.fixture
def posts(with_key, monkeypatch):
    """Replace requests.post inside ai_summary.py and record each call."""
    record = []

    def fake_post(url, headers=None, json=None, timeout=None):
        record.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _FakeResponse(_gemini_reply("Fire activity was above the usual level."))

    monkeypatch.setattr(ai_summary.requests, "post", fake_post)
    return record


def test_no_key_returns_none_without_calling_api(no_key, monkeypatch):
    def must_not_call(*a, **k):
        raise AssertionError("API must not be called without a key")
    monkeypatch.setattr(ai_summary.requests, "post", must_not_call)
    assert ai_summary.summarize("Punjab", FACTS) is None


def test_returns_model_text(posts):
    assert ai_summary.summarize("Punjab", FACTS) == "Fire activity was above the usual level."


def test_key_in_header_not_url_and_default_model(posts):
    ai_summary.summarize("Punjab", FACTS)
    sent = posts[0]
    assert sent["headers"]["x-goog-api-key"] == FAKE_KEY
    assert FAKE_KEY not in sent["url"]
    assert sent["url"].endswith("/models/gemini-2.5-flash:generateContent")


def test_model_from_env(posts, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-other")
    ai_summary.summarize("Punjab", FACTS)
    assert "/models/gemini-other:generateContent" in posts[0]["url"]


def test_prompt_has_instruction_and_only_our_facts(posts):
    ai_summary.summarize("Punjab", FACTS)
    prompt = posts[0]["json"]["contents"][0]["parts"][0]["text"]
    assert ("Write 3-4 plain sentences for a farmer or official. "
            "Use only these numbers; do not add any number, cause or prediction.") in prompt
    assert "Punjab" in prompt
    assert '"historical_median_same_month": 80.0' in prompt
    assert FAKE_KEY not in prompt


def test_key_read_from_dotenv(no_key, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"OTHER=1\nGEMINI_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    assert ai_summary.get_setting("GEMINI_API_KEY") == FAKE_KEY


@pytest.mark.parametrize("make_response", [
    lambda: _FakeResponse({}, status=429),                  # quota / HTTP error
    lambda: _FakeResponse({"candidates": []}),              # empty reply
    lambda: _FakeResponse(_gemini_reply("   ")),            # blank text
])
def test_bad_replies_return_none(with_key, monkeypatch, make_response):
    monkeypatch.setattr(ai_summary.requests, "post", lambda *a, **k: make_response())
    assert ai_summary.summarize("Punjab", FACTS) is None


def test_network_error_returns_none(with_key, monkeypatch):
    def broken_post(*a, **k):
        raise requests.Timeout("slow")
    monkeypatch.setattr(ai_summary.requests, "post", broken_post)
    assert ai_summary.summarize("Punjab", FACTS) is None
