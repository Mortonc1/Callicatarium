import time

import pytest
from fastapi.testclient import TestClient

from musicgen_personas.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setattr("musicgen_personas.personas.DEFAULT_REGISTRY_PATH", registry_path)
    web_app._jobs.clear()
    return TestClient(web_app.app)


def test_index_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "MusicHub" in res.text


def test_presets_are_listed(client):
    res = client.get("/api/presets")
    assert res.status_code == 200
    names = {p["name"] for p in res.json()}
    assert "Aria" in names


def test_persona_lifecycle_via_api(client):
    assert client.get("/api/personas").json() == []

    created = client.post(
        "/api/personas",
        json={"name": "Nova", "voice": "v2/en_speaker_5", "genre": "synthwave", "description": "cool voice"},
    )
    assert created.status_code == 200
    assert created.json()["name"] == "Nova"

    listed = client.get("/api/personas").json()
    assert [p["name"] for p in listed] == ["Nova"]

    dup = client.post(
        "/api/personas",
        json={"name": "Nova", "voice": "v2/en_speaker_5"},
    )
    assert dup.status_code == 409


def test_generate_unknown_persona_404s(client):
    res = client.post("/api/generate", json={"persona": "Nope", "lyrics": "hello"})
    assert res.status_code == 404


def test_generate_rejects_empty_lyrics(client):
    client.post("/api/personas", json={"name": "Aria2", "voice": "v2/en_speaker_9"})
    res = client.post("/api/generate", json={"persona": "Aria2", "lyrics": "   "})
    assert res.status_code == 400


def test_generate_job_lifecycle_reaches_terminal_state(client, monkeypatch):
    # Real generation needs Bark's model weights (network access this test
    # environment doesn't have, and which would make this test slow and
    # network-dependent regardless). What's under test here is this
    # project's own job pipeline -- queued -> running -> terminal state,
    # correctly surfaced over the API -- so the model-loading boundary is
    # mocked to fail fast rather than actually reaching Bark/HF Hub.
    def _fail_to_load(*args, **kwargs):
        raise RuntimeError("model weights unavailable in test environment")

    monkeypatch.setattr(web_app, "_ensure_models_loaded", _fail_to_load)

    client.post("/api/personas", json={"name": "Aria3", "voice": "v2/en_speaker_9"})
    res = client.post("/api/generate", json={"persona": "Aria3", "lyrics": "hello there"})
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.1)

    assert status["status"] == "error"
    assert "model weights unavailable" in status["error"]

    jobs = client.get("/api/jobs").json()
    assert any(j["job_id"] == job_id for j in jobs)
