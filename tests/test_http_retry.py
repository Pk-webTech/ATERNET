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
