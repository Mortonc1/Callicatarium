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


def _fabricate_stems(song, names=("vocals", "drums", "bass", "other"), rate=24000, amplitude=0.2):
    import numpy as np
    from scipy.io.wavfile import write as write_wav

    stems_dir = song.dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        write_wav(stems_dir / f"{name}.wav", rate, np.full(rate, amplitude, dtype=np.float32))
    song.stem_levels = {name: {"gain": 1.0, "muted": False} for name in names}


def test_stem_separation_job_lifecycle(client, monkeypatch):
    # Real separation needs Demucs' model weights over the network -- same
    # class of dependency as Bark, and just as undesirable in a fast unit
    # test. What's under test here is this project's own job wiring, so the
    # separation call itself is mocked to fail fast.
    def _fail_to_separate(store, song):
        raise RuntimeError("demucs unavailable in test environment")

    monkeypatch.setattr("musicgen_personas.song_stems.separate_song_stems", _fail_to_separate)

    client.post("/api/personas", json={"name": "AriaS", "voice": "v2/en_speaker_9"})
    song = client.post("/api/songs", json={"title": "Song", "persona": "AriaS", "lyrics": "verse"}).json()

    res = client.post(f"/api/songs/{song['id']}/stems/separate")
    assert res.status_code == 200
    status = _wait_for_job(client, res.json()["job_id"])
    assert status["status"] == "error"
    assert "demucs unavailable" in status["error"]


def test_stem_level_update_and_audio_serving(client):
    from musicgen_personas.song import SongStore

    client.post("/api/personas", json={"name": "AriaT", "voice": "v2/en_speaker_9"})
    song_resp = client.post("/api/songs", json={"title": "Song", "persona": "AriaT", "lyrics": "verse"}).json()
    song_id = song_resp["id"]

    store = SongStore()
    song = store.get(song_id)
    _fabricate_stems(song)
    store.save(song)

    updated = client.put(f"/api/songs/{song_id}/stems/vocals", json={"gain": 0.5, "muted": True})
    assert updated.status_code == 200
    assert updated.json()["stem_levels"]["vocals"] == {"gain": 0.5, "muted": True}

    missing = client.put(f"/api/songs/{song_id}/stems/banjo", json={"gain": 1.0})
    assert missing.status_code == 404

    audio_res = client.get(f"/api/songs/{song_id}/stems/vocals/audio")
    assert audio_res.status_code == 200

    no_such_stem = client.get(f"/api/songs/{song_id}/stems/banjo/audio")
    assert no_such_stem.status_code == 404


def test_stem_mix_job_and_audio(client):
    from musicgen_personas.song import SongStore

    client.post("/api/personas", json={"name": "AriaU", "voice": "v2/en_speaker_9"})
    song_resp = client.post("/api/songs", json={"title": "Song", "persona": "AriaU", "lyrics": "verse"}).json()
    song_id = song_resp["id"]

    store = SongStore()
    song = store.get(song_id)
    _fabricate_stems(song)
    store.save(song)

    no_mix_yet = client.get(f"/api/songs/{song_id}/stems/mix/audio")
    assert no_mix_yet.status_code == 404

    res = client.post(f"/api/songs/{song_id}/stems/mix")
    assert res.status_code == 200
    status = _wait_for_job(client, res.json()["job_id"])
    assert status["status"] == "done"

    mix_res = client.get(f"/api/songs/{song_id}/stems/mix/audio")
    assert mix_res.status_code == 200


def test_from_reference_job_lifecycle(client, monkeypatch, tmp_path):
    # Transcription needs Whisper's weights over the network; what's under
    # test is this project's upload -> job -> song wiring, so the transcribe
    # step is mocked to fail fast.
    def _fail(*args, **kwargs):
        raise RuntimeError("whisper unavailable in test environment")

    monkeypatch.setattr("musicgen_personas.recreate.create_song_from_reference", _fail)

    client.post("/api/personas", json={"name": "AriaR", "voice": "v2/en_speaker_9"})
    res = client.post(
        "/api/songs/from-reference",
        data={"title": "Recreated", "persona": "AriaR", "model_size": "base", "isolate_vocals": "false"},
        files={"reference": ("ref.mp3", b"not really audio", "audio/mpeg")},
    )
    assert res.status_code == 200
    status = _wait_for_job(client, res.json()["job_id"])
    assert status["status"] == "error"
    assert "whisper unavailable" in status["error"]


def test_from_reference_rejects_unknown_persona(client):
    res = client.post(
        "/api/songs/from-reference",
        data={"title": "Recreated", "persona": "Nope"},
        files={"reference": ("ref.mp3", b"x", "audio/mpeg")},
    )
    assert res.status_code == 404


def test_section_instrumental_job_and_audio(client, monkeypatch):
    from musicgen_personas.song import SongStore

    client.post("/api/personas", json={"name": "AriaI", "voice": "v2/en_speaker_9"})
    song = client.post("/api/songs", json={"title": "Song", "persona": "AriaI", "lyrics": "verse"}).json()
    section_id = song["sections"][0]["id"]

    no_instrumental = client.get(f"/api/songs/{song['id']}/sections/{section_id}/instrumental/audio")
    assert no_instrumental.status_code == 404

    def fake_generate(store, s, sid, prompt="x", duration=None):
        import numpy as np
        from scipy.io.wavfile import write as write_wav

        section = s.section(sid)
        s.dir.mkdir(parents=True, exist_ok=True)
        filename = f"{sid}_instrumental.wav"
        write_wav(s.dir / filename, 24000, np.zeros(1000, dtype=np.float32))
        section.instrumental_file = filename
        store.save(s)
        return s

    monkeypatch.setattr("musicgen_personas.recreate.generate_section_instrumental", fake_generate)

    res = client.post(
        f"/api/songs/{song['id']}/sections/{section_id}/instrumental",
        json={"prompt": "moody synthwave"},
    )
    assert res.status_code == 200
    status = _wait_for_job(client, res.json()["job_id"])
    assert status["status"] == "done"

    audio = client.get(f"/api/songs/{song['id']}/sections/{section_id}/instrumental/audio")
    assert audio.status_code == 200


def test_split_and_merge_sections_via_api(client):
    client.post("/api/personas", json={"name": "AriaSp", "voice": "v2/en_speaker_9"})
    song = client.post(
        "/api/songs", json={"title": "Song", "persona": "AriaSp", "lyrics": "hold the line tonight"}
    ).json()
    section_id = song["sections"][0]["id"]

    split = client.post(
        f"/api/songs/{song['id']}/sections/{section_id}/split", json={"granularity": "words"}
    )
    assert split.status_code == 200
    pieces = split.json()["sections"]
    assert [s["lyrics"] for s in pieces] == ["hold", "the", "line", "tonight"]

    # change one word, then merge back -- the line should read correctly
    client.put(f"/api/songs/{song['id']}/sections/{pieces[3]['id']}", json={"lyrics": "forever"})
    merged = client.post(
        f"/api/songs/{song['id']}/sections/merge",
        json={"section_ids": [s["id"] for s in pieces]},
    )
    assert merged.status_code == 200
    assert merged.json()["sections"][0]["lyrics"] == "hold the line forever"


def test_split_unsplittable_returns_400(client):
    client.post("/api/personas", json={"name": "AriaSp2", "voice": "v2/en_speaker_9"})
    song = client.post(
        "/api/songs", json={"title": "Song", "persona": "AriaSp2", "lyrics": "single"}
    ).json()
    res = client.post(
        f"/api/songs/{song['id']}/sections/{song['sections'][0]['id']}/split",
        json={"granularity": "lines"},
    )
    assert res.status_code == 400


def test_merge_non_adjacent_returns_400(client):
    client.post("/api/personas", json={"name": "AriaSp3", "voice": "v2/en_speaker_9"})
    song = client.post(
        "/api/songs", json={"title": "Song", "persona": "AriaSp3", "lyrics": "a\n\nb\n\nc"}
    ).json()
    ids = [s["id"] for s in song["sections"]]
    res = client.post(
        f"/api/songs/{song['id']}/sections/merge", json={"section_ids": [ids[0], ids[2]]}
    )
    assert res.status_code == 400


def test_styles_endpoint_lists_presets(client):
    res = client.get("/api/styles")
    assert res.status_code == 200
    keys = {s["key"] for s in res.json()}
    assert {"synthwave", "lofi", "rock", "hiphop"} <= keys
    hiphop = next(s for s in res.json() if s["key"] == "hiphop")
    assert hiphop["sung"] is False


def test_create_song_with_style(client):
    client.post("/api/personas", json={"name": "AriaSt", "voice": "v2/en_speaker_9"})
    song = client.post(
        "/api/songs",
        json={"title": "Song", "persona": "AriaSt", "lyrics": "a line", "style_key": "synthwave"},
    ).json()
    assert song["style_key"] == "synthwave"


def test_set_and_clear_song_style(client):
    client.post("/api/personas", json={"name": "AriaSt2", "voice": "v2/en_speaker_9"})
    song = client.post(
        "/api/songs", json={"title": "Song", "persona": "AriaSt2", "lyrics": "a line"}
    ).json()
    assert song["style_key"] is None

    set_res = client.put(f"/api/songs/{song['id']}/style", json={"style_key": "lofi"})
    assert set_res.status_code == 200
    assert set_res.json()["style_key"] == "lofi"

    cleared = client.put(f"/api/songs/{song['id']}/style", json={"style_key": None})
    assert cleared.json()["style_key"] is None


def test_set_unknown_style_returns_400(client):
    client.post("/api/personas", json={"name": "AriaSt3", "voice": "v2/en_speaker_9"})
    song = client.post(
        "/api/songs", json={"title": "Song", "persona": "AriaSt3", "lyrics": "a line"}
    ).json()
    res = client.put(f"/api/songs/{song['id']}/style", json={"style_key": "not-a-genre"})
    assert res.status_code == 400
