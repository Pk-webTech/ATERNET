"""A truncated/corrupted API body must be retried, not crash the pipeline."""

import json

from src.phase1 import pvlive_openmeteo as om


class _Resp:
    def __init__(self, body: bytes):
        self._b = body

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_truncated_json_is_retried(monkeypatch):
    good = json.dumps({"hourly": {"time": ["2024-01-01T00:00"]}}).encode()
    bodies = [good[:20], good]  # first response cut off mid-document
    monkeypatch.setattr(om.urllib.request, "urlopen", lambda *a, **k: _Resp(bodies.pop(0)))
    monkeypatch.setattr(om.time, "sleep", lambda s: None)
    out = om._http_get_json("https://example.invalid/x", {"a": 1}, cache=False)
    assert out["hourly"]["time"] == ["2024-01-01T00:00"]
    assert not bodies  # both responses were consumed


def test_failing_range_is_split_and_recovers(monkeypatch):
    """A big request that always fails must be halved until the pieces succeed."""
    import pandas as pd

    calls = []

    def fake_get(url, params, cache, retries=4, timeout=120):
        a, b = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
        calls.append((a, b))
        if (b - a).days > 40:  # "large responses get cut short"
            raise RuntimeError(f"Request to {url} failed after {retries} attempts: cut off")
        return {"start": params["start_date"], "end": params["end_date"]}

    monkeypatch.setattr(om, "_http_get_json", fake_get)
    out = om.fetch_range_adaptive("https://example.invalid", {}, pd.Timestamp("2024-01-01"),
                                  pd.Timestamp("2024-06-30"), cache=False, chunk_days=183)
    covered = sorted((pd.Timestamp(p["start"]), pd.Timestamp(p["end"])) for p in out)
    assert covered[0][0] == pd.Timestamp("2024-01-01") and covered[-1][1] == pd.Timestamp("2024-06-30")
    assert all((b - a).days <= 40 for a, b in covered)
    # contiguous, no gaps or overlaps
    assert all(covered[i][1] + pd.Timedelta(days=1) == covered[i + 1][0] for i in range(len(covered) - 1))


def test_client_errors_are_not_split(monkeypatch):
    import pandas as pd
    import pytest

    def fake_get(url, params, cache, retries=4, timeout=120):
        raise RuntimeError("HTTP 400 from x: bad request")

    monkeypatch.setattr(om, "_http_get_json", fake_get)
    with pytest.raises(RuntimeError):
        om.fetch_range_adaptive("https://example.invalid", {}, pd.Timestamp("2024-01-01"),
                                pd.Timestamp("2024-06-30"), cache=False)
