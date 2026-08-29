import time

import pytest
from fastapi.testclient import TestClient

from musicgen_personas.web import app as web_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    monkeypatch.setattr("musicgen_personas.personas.DEFAULT_REGISTRY_PATH", registry_path)
    monkeypatch.setattr("musicgen_personas.song.DEFAULT_SONGS_DIR", tmp_path / "songs")
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


def _wait_for_job(client, job_id, timeout=5):
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.05)
    return status


def test_song_create_and_get(client):
    client.post("/api/personas", json={"name": "Aria4", "voice": "v2/en_speaker_9"})
    res = client.post(
        "/api/songs", json={"title": "My Song", "persona": "Aria4", "lyrics": "verse one\n\nchorus one"}
    )
    assert res.status_code == 200
    song = res.json()
    assert song["title"] == "My Song"
    assert len(song["sections"]) == 2
    assert all(s["is_stale"] for s in song["sections"])

    fetched = client.get(f"/api/songs/{song['id']}").json()
    assert fetched["id"] == song["id"]


def test_song_create_rejects_unknown_persona(client):
    res = client.post("/api/songs", json={"title": "Song", "persona": "Nope", "lyrics": "hi"})
    assert res.status_code == 404


def test_song_list_shows_summary(client):
    client.post("/api/personas", json={"name": "Aria5", "voice": "v2/en_speaker_9"})
    client.post("/api/songs", json={"title": "Song A", "persona": "Aria5", "lyrics": "a\n\nb"})
    listed = client.get("/api/songs").json()
    assert len(listed) == 1
    assert listed[0]["title"] == "Song A"
    assert listed[0]["section_count"] == 2
    assert listed[0]["stale_count"] == 2
    assert listed[0]["has_full_render"] is False


def test_song_section_editing_and_reordering(client):
    client.post("/api/personas", json={"name": "Aria6", "voice": "v2/en_speaker_9"})
    song = client.post(
        "/api/songs", json={"title": "Song", "persona": "Aria6", "lyrics": "verse one"}
    ).json()
    section_id = song["sections"][0]["id"]

    updated = client.put(f"/api/songs/{song['id']}/sections/{section_id}", json={"lyrics": "new words"})
    assert updated.status_code == 200
    assert updated.json()["sections"][0]["lyrics"] == "new words"

    added = client.post(
        f"/api/songs/{song['id']}/sections", json={"label": "Bridge", "lyrics": "bridge text", "position": 0}
    ).json()
    assert [s["label"] for s in added["sections"]] == ["Bridge", "Section 1"]

    ids = [s["id"] for s in added["sections"]]
    reordered = client.post(
        f"/api/songs/{song['id']}/sections/reorder", json={"section_ids": list(reversed(ids))}
    ).json()
    assert [s["id"] for s in reordered["sections"]] == list(reversed(ids))

    removed = client.delete(f"/api/songs/{song['id']}/sections/{ids[0]}").json()
    assert len(removed["sections"]) == 1


def test_song_reorder_rejects_bad_ids(client):
    client.post("/api/personas", json={"name": "Aria7", "voice": "v2/en_speaker_9"})
    song = client.post("/api/songs", json={"title": "Song", "persona": "Aria7", "lyrics": "a"}).json()
    res = client.post(f"/api/songs/{song['id']}/sections/reorder", json={"section_ids": ["bogus"]})
    assert res.status_code == 400


def test_song_regenerate_section_job_lifecycle(client, monkeypatch):
    def _fail_to_load(*args, **kwargs):
        raise RuntimeError("model weights unavailable in test environment")

    monkeypatch.setattr(web_app, "_ensure_models_loaded", _fail_to_load)

    client.post("/api/personas", json={"name": "Aria8", "voice": "v2/en_speaker_9"})
    song = client.post("/api/songs", json={"title": "Song", "persona": "Aria8", "lyrics": "verse"}).json()
    section_id = song["sections"][0]["id"]

    res = client.post(f"/api/songs/{song['id']}/sections/{section_id}/regenerate", json={})
    assert res.status_code == 200
    status = _wait_for_job(client, res.json()["job_id"])
    assert status["status"] == "error"
    assert "model weights unavailable" in status["error"]

    # section audio still unset since regeneration never got past model loading
    audio_res = client.get(f"/api/songs/{song['id']}/sections/{section_id}/audio")
    assert audio_res.status_code == 404


def test_song_render_full_pipeline_without_bark(client, tmp_path):
    import numpy as np
    from scipy.io.wavfile import write as write_wav

    from musicgen_personas.song import SongStore

    client.post("/api/personas", json={"name": "Aria9", "voice": "v2/en_speaker_9"})
    song_resp = client.post(
        "/api/songs", json={"title": "Song", "persona": "Aria9", "lyrics": "verse one\n\nverse two"}
    ).json()
    song_id = song_resp["id"]

    # Fabricate rendered section audio directly (bypassing Bark entirely) so
    # the render endpoint's own stitching logic can be exercised for real.
    store = SongStore()
    song = store.get(song_id)
    for section in song.sections:
        song.dir.mkdir(parents=True, exist_ok=True)
        audio = np.ones(24000, dtype=np.float32)
        filename = f"{section.id}.wav"
        write_wav(song.dir / filename, 24000, audio)
        section.audio_file = filename
        section.rendered_lyrics = section.lyrics
    store.save(song)

    res = client.post(f"/api/songs/{song_id}/render")
    assert res.status_code == 200
    status = _wait_for_job(client, res.json()["job_id"])
    assert status["status"] == "done"

    audio_res = client.get(f"/api/songs/{song_id}/audio")
    assert audio_res.status_code == 200
    assert audio_res.headers["content-type"] == "audio/wav" or audio_res.content[:4] == b"RIFF"

    summary = next(s for s in client.get("/api/songs").json() if s["id"] == song_id)
    assert summary["has_full_render"] is True

    # The detail endpoint (what the song editor page actually reads to
    # decide whether to show a full-song player) must reflect this too.
    detail = client.get(f"/api/songs/{song_id}").json()
    assert detail["has_full_render"] is True
