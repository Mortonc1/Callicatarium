import numpy as np
import pytest
from scipy.io.wavfile import write as write_wav

from musicgen_personas.song import SongStore
from musicgen_personas.song_render import _crossfade_concat, render_song

SAMPLE_RATE = 24000


def _write_section_audio(song, section, seconds=1.0, value=1.0, rate=SAMPLE_RATE):
    audio = np.full(int(rate * seconds), value, dtype=np.float32)
    song.dir.mkdir(parents=True, exist_ok=True)
    filename = f"{section.id}.wav"
    write_wav(song.dir / filename, rate, audio)
    section.audio_file = filename
    section.rendered_lyrics = section.lyrics


def test_crossfade_concat_preserves_total_length_minus_overlap():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(1000, dtype=np.float32) * 2
    result = _crossfade_concat(a, b, fade_len=100)
    assert len(result) == 1000 + 1000 - 100


def test_crossfade_concat_short_clips_dont_crash():
    a = np.ones(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    result = _crossfade_concat(a, b, fade_len=100)  # fade_len longer than either clip
    assert len(result) > 0


def test_render_song_raises_when_section_never_rendered(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    with pytest.raises(ValueError, match="need regenerating"):
        render_song(song, tmp_path / "out.wav")


def test_render_song_raises_when_section_edited_since_render(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    for section in song.sections:
        _write_section_audio(song, section)
    store.save(song)
    song.sections[0].lyrics = "edited lyrics"  # now stale again
    with pytest.raises(ValueError, match="need regenerating"):
        render_song(song, tmp_path / "out.wav")


def test_render_song_stitches_sections_in_order(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    for section in song.sections:
        _write_section_audio(song, section, seconds=0.5)
    store.save(song)

    out_path = render_song(song, tmp_path / "out.wav")
    assert out_path.exists()

    from scipy.io.wavfile import read as read_wav

    rate, audio = read_wav(out_path)
    assert rate == SAMPLE_RATE
    # Two 0.5s clips crossfaded lose one fade-length of overlap (120ms) --
    # not simply doubled, and not collapsed to a single clip's length either.
    fade_len = int(SAMPLE_RATE * 0.120)
    expected_len = int(SAMPLE_RATE * 0.5) * 2 - fade_len
    assert len(audio) == expected_len


def test_render_song_rejects_sample_rate_mismatch(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    _write_section_audio(song, song.sections[0], rate=24000)
    _write_section_audio(song, song.sections[1], rate=16000)
    store.save(song)
    with pytest.raises(ValueError, match="Sample rate mismatch"):
        render_song(song, tmp_path / "out.wav")
