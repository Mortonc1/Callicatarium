import numpy as np
import pytest
from scipy.io.wavfile import write as write_wav

from musicgen_personas.recreate import (
    build_sections_from_reference,
    generate_section_instrumental,
    import_reference,
)
from musicgen_personas.song import SongStore
from musicgen_personas.transcribe import TranscriptSegment


@pytest.fixture
def reference_file(tmp_path):
    path = tmp_path / "reference.wav"
    write_wav(path, 24000, np.full(24000 * 5, 0.2, dtype=np.float32))
    return path


def test_import_reference_copies_into_song_dir(tmp_path, reference_file):
    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    dest = import_reference(store, song, reference_file)

    assert dest.exists()
    assert dest.parent == song.dir / "reference"
    # survives a round-trip through disk, and resolves back to a real file
    reloaded = store.get(song.id)
    assert reloaded.reference_file == song.reference_file
    assert reloaded.reference_path().exists()


def test_import_reference_missing_file_raises(tmp_path):
    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    with pytest.raises(FileNotFoundError):
        import_reference(store, song, tmp_path / "nope.wav")


def test_build_sections_from_reference_uses_transcript_timing(tmp_path, reference_file, monkeypatch):
    # Transcription itself needs Whisper's weights over the network; what's
    # under test here is this project's own segment -> section mapping, so
    # the transcribe call is stubbed with a known segment list.
    fake_segments = [
        TranscriptSegment(0.0, 2.0, "first line"),
        TranscriptSegment(2.1, 4.0, "second line"),
        TranscriptSegment(12.0, 14.0, "after a break"),
    ]
    monkeypatch.setattr("musicgen_personas.transcribe.transcribe", lambda *a, **k: fake_segments)

    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    import_reference(store, song, reference_file)
    song = build_sections_from_reference(store, song, use_isolated_vocals=False)

    assert len(song.sections) == 2
    assert song.sections[0].lyrics == "first line\nsecond line"
    assert song.sections[0].ref_start == 0.0
    assert song.sections[0].ref_end == 4.0
    assert song.sections[1].lyrics == "after a break"
    assert song.sections[1].ref_start == 12.0
    # and it persisted
    assert store.get(song.id).sections[1].ref_start == 12.0


def test_build_sections_requires_a_reference(tmp_path):
    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    with pytest.raises(ValueError, match="no reference track"):
        build_sections_from_reference(store, song)


def test_build_sections_empty_transcript_raises(tmp_path, reference_file, monkeypatch):
    monkeypatch.setattr("musicgen_personas.transcribe.transcribe", lambda *a, **k: [])
    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    import_reference(store, song, reference_file)
    with pytest.raises(RuntimeError, match="no lyrics"):
        build_sections_from_reference(store, song, use_isolated_vocals=False)


def test_generate_section_instrumental_conditions_on_matching_slice(
    tmp_path, reference_file, monkeypatch
):
    calls = {}

    def fake_generate(prompt, reference_path, out_path, duration, reference_offset):
        calls.update(
            prompt=prompt, duration=duration, reference_offset=reference_offset, out_path=out_path
        )
        write_wav(out_path, 24000, np.zeros(1000, dtype=np.float32))
        return out_path

    monkeypatch.setattr("musicgen_personas.melody.generate_melody_conditioned", fake_generate)

    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    import_reference(store, song, reference_file)
    section = song.sections[0]
    section.ref_start = 30.0
    section.ref_end = 45.0
    store.save(song)

    song = generate_section_instrumental(store, song, section.id, prompt="moody synthwave")

    # conditioned on this section's own slice of the reference, not the start
    assert calls["reference_offset"] == 30.0
    assert calls["duration"] == 15.0
    assert calls["prompt"] == "moody synthwave"
    assert store.get(song.id).sections[0].instrumental_file is not None


def test_generate_section_instrumental_clamps_to_musicgen_limit(
    tmp_path, reference_file, monkeypatch
):
    calls = {}

    def fake_generate(prompt, reference_path, out_path, duration, reference_offset):
        calls["duration"] = duration
        write_wav(out_path, 24000, np.zeros(1000, dtype=np.float32))
        return out_path

    monkeypatch.setattr("musicgen_personas.melody.generate_melody_conditioned", fake_generate)

    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    import_reference(store, song, reference_file)
    section = song.sections[0]
    section.ref_start = 0.0
    section.ref_end = 120.0  # far longer than MusicGen can generate in one call
    store.save(song)

    generate_section_instrumental(store, song, section.id)
    assert calls["duration"] == 30.0


def test_generate_section_instrumental_without_reference_timing_raises(tmp_path, reference_file):
    store = SongStore(root=tmp_path / "songs")
    song = store.create("Song", "Aria", "placeholder")
    import_reference(store, song, reference_file)
    with pytest.raises(ValueError, match="no reference timing"):
        generate_section_instrumental(store, song, song.sections[0].id)
