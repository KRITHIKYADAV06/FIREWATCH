# tests/test_weather.py
"""
Tests for src/weather.py. The NASA POWER API is MOCKED: no network calls.
Run from the repo root:   pytest -v tests/test_weather.py
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import weather  # noqa: E402


def _power_payload():
    """A tiny made-up POWER reply: Jan-Feb 2020 plus the annual "202013" key."""
    return {"properties": {"parameter": {
        "PRECTOTCORR": {"202001": 1.5, "202002": 0.0, "202013": 0.8},
        "T2M": {"202001": 12.3, "202002": -999.0, "202013": 20.0},
        "RH2M": {"202001": 70.0, "202002": 65.0, "202013": 60.0},
        "WS2M": {"202001": 2.1, "202002": 2.4, "202013": 2.2},
    }}}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def calls(monkeypatch):
    """Replace requests.get inside weather.py and record each call."""
    record = []

    def fake_get(url, params=None, timeout=None):
        record.append({"url": url, "params": params, "timeout": timeout})
        return _FakeResponse(_power_payload())

    monkeypatch.setattr(weather.requests, "get", fake_get)
    return record


def test_parses_columns_and_drops_annual_month_13(calls):
    df = weather.monthly_weather(30.1, 75.7, 2020, 2020)
    assert list(df.columns) == ["year", "month", "rain_mm_day", "temp_c", "humidity_pct", "wind_ms"]
    assert df[["year", "month"]].values.tolist() == [[2020, 1], [2020, 2]]   # no month 13
    assert df.loc[0, "rain_mm_day"] == pytest.approx(1.5)
    assert df.loc[0, "temp_c"] == pytest.approx(12.3)


def test_fill_value_becomes_missing(calls):
    df = weather.monthly_weather(30.1, 75.7, 2020, 2020)
    assert pd.isna(df.loc[1, "temp_c"])             # -999 is POWER's "no data"


def test_request_uses_spec_parameters_and_timeout(calls):
    weather.monthly_weather(30.1, 75.7, 2019, 2020)
    sent = calls[0]
    assert sent["url"] == "https://power.larc.nasa.gov/api/temporal/monthly/point"
    assert sent["timeout"] == 20
    assert sent["params"]["parameters"] == "PRECTOTCORR,T2M,RH2M,WS2M"
    assert sent["params"]["community"] == "AG"
    assert (sent["params"]["start"], sent["params"]["end"]) == (2019, 2020)
    assert sent["params"]["format"] == "JSON"


def test_network_error_returns_none(monkeypatch):
    def broken_get(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(weather.requests, "get", broken_get)
    assert weather.monthly_weather(30.1, 75.7, 2020, 2020) is None


def test_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(weather.requests, "get",
                        lambda *a, **k: _FakeResponse({}, status=500))
    assert weather.monthly_weather(30.1, 75.7, 2020, 2020) is None


def test_bad_json_returns_none(monkeypatch):
    monkeypatch.setattr(weather.requests, "get",
                        lambda *a, **k: _FakeResponse({"unexpected": True}))
    assert weather.monthly_weather(30.1, 75.7, 2020, 2020) is None


def test_reversed_years_returns_none(calls):
    assert weather.monthly_weather(30.1, 75.7, 2021, 2020) is None
    assert calls == []                              # no request was even made
