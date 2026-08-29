import numpy as np
import pytest
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav

from musicgen_personas.song import SongStore
from musicgen_personas.song_stems import mix_stems, separate_song_stems, set_stem_level

SAMPLE_RATE = 24000


def _make_song_with_stems(tmp_path, levels=None):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one")
    song.dir.mkdir(parents=True, exist_ok=True)
    stems_dir = song.dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    stem_names = ["vocals", "drums", "bass", "other"]
    song.stem_levels = levels or {name: {"gain": 1.0, "muted": False} for name in stem_names}
    for name in stem_names:
        audio = np.full(SAMPLE_RATE, 0.25, dtype=np.float32)
        write_wav(stems_dir / f"{name}.wav", SAMPLE_RATE, audio)
    store.save(song)
    return store, song


def test_separate_song_stems_requires_full_render(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one")
    with pytest.raises(ValueError, match="Render the full song first"):
        separate_song_stems(store, song)


def test_mix_stems_requires_separation_first(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one")
    with pytest.raises(ValueError, match="No stems yet"):
        mix_stems(song, tmp_path / "mix.wav")


def test_mix_stems_sums_all_unmuted_stems(tmp_path):
    _, song = _make_song_with_stems(tmp_path)
    out_path = mix_stems(song, tmp_path / "mix.wav")
    rate, audio = read_wav(out_path)
    assert rate == SAMPLE_RATE
    # four stems at 0.25 amplitude and gain 1.0 sum to ~1.0
    assert np.isclose(float(audio.max()), 1.0, atol=0.01)


def test_mix_stems_respects_mute(tmp_path):
    levels = {
        "vocals": {"gain": 1.0, "muted": False},
        "drums": {"gain": 1.0, "muted": True},
        "bass": {"gain": 1.0, "muted": True},
        "other": {"gain": 1.0, "muted": True},
    }
    _, song = _make_song_with_stems(tmp_path, levels=levels)
    out_path = mix_stems(song, tmp_path / "mix.wav")
    _, audio = read_wav(out_path)
    # only vocals (0.25 amplitude) survives
    assert np.isclose(float(audio.max()), 0.25, atol=0.01)


def test_mix_stems_all_muted_produces_silence(tmp_path):
    levels = {name: {"gain": 1.0, "muted": True} for name in ["vocals", "drums", "bass", "other"]}
    _, song = _make_song_with_stems(tmp_path, levels=levels)
    out_path = mix_stems(song, tmp_path / "mix.wav")
    _, audio = read_wav(out_path)
    assert float(np.abs(audio).max()) == 0.0


def test_mix_stems_clips_prevented_when_gain_boosted(tmp_path):
    levels = {name: {"gain": 3.0, "muted": False} for name in ["vocals", "drums", "bass", "other"]}
    _, song = _make_song_with_stems(tmp_path, levels=levels)
    out_path = mix_stems(song, tmp_path / "mix.wav")
    _, audio = read_wav(out_path)
    assert float(np.abs(audio).max()) <= 1.0 + 1e-6


def test_set_stem_level_updates_gain_and_mute(tmp_path):
    store, song = _make_song_with_stems(tmp_path)
    set_stem_level(store, song, "vocals", gain=0.5)
    set_stem_level(store, song, "drums", muted=True)
    reloaded = store.get(song.id)
    assert reloaded.stem_levels["vocals"]["gain"] == 0.5
    assert reloaded.stem_levels["drums"]["muted"] is True
    # untouched fields keep their defaults
    assert reloaded.stem_levels["vocals"]["muted"] is False
    assert reloaded.stem_levels["drums"]["gain"] == 1.0


def test_set_stem_level_unknown_stem_raises(tmp_path):
    store, song = _make_song_with_stems(tmp_path)
    with pytest.raises(KeyError):
        set_stem_level(store, song, "banjo", gain=0.5)
