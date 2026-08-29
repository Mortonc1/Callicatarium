"""Splits a song's full render into stems and lets you remix them.

This is the closest thing this project has to a real multitrack editor:
once a song has been rendered to `full.wav`, `separate_song_stems` splits
it into independent vocals/drums/bass/other tracks (see `stems.py` for how
and why), and `mix_stems` lets you render a custom mix from independent
per-stem gain and mute settings -- adjust the vocals without touching the
drums, mute the bass, and so on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .song import Song, SongStore


def separate_song_stems(store: SongStore, song: Song) -> Song:
    """Split `song`'s full render into stems, stored under song.dir/stems/."""
    from .stems import cleanup_work_dir, separate_stems

    full_path = song.dir / "full.wav"
    if not full_path.exists():
        raise ValueError("Render the full song first -- no full.wav to separate")

    work_dir = song.dir / "_demucs_work"
    stems_dir = song.dir / "stems"
    try:
        produced = separate_stems(full_path, work_dir)
        stems_dir.mkdir(parents=True, exist_ok=True)
        stem_levels = {}
        for name, path in produced.items():
            path.replace(stems_dir / f"{name}.wav")
            stem_levels[name] = {"gain": 1.0, "muted": False}
    finally:
        cleanup_work_dir(work_dir)

    song.stem_levels = stem_levels
    store.save(song)
    return song


def set_stem_level(
    store: SongStore, song: Song, stem_name: str, gain: float | None = None, muted: bool | None = None
) -> Song:
    if stem_name not in song.stem_levels:
        raise KeyError(f"No stem '{stem_name}' on this song -- run separate-stems first")
    if gain is not None:
        song.stem_levels[stem_name]["gain"] = gain
    if muted is not None:
        song.stem_levels[stem_name]["muted"] = muted
    store.save(song)
    return song


def mix_stems(song: Song, out_path: str | Path) -> Path:
    """Render a custom mix from the song's current per-stem gain/mute settings."""
    from scipy.io.wavfile import read as read_wav
    from scipy.io.wavfile import write as write_wav

    if not song.stem_levels:
        raise ValueError("No stems yet -- run separate-stems first")

    sample_rate = None
    mix: np.ndarray | None = None
    for name, level in song.stem_levels.items():
        stem_path = song.dir / "stems" / f"{name}.wav"
        if not stem_path.exists():
            raise FileNotFoundError(f"Missing stem file for '{name}': {stem_path}")
        if level["muted"]:
            continue
        rate, audio = read_wav(stem_path)
        audio = audio.astype(np.float32) * level["gain"]
        if sample_rate is None:
            sample_rate = rate
            mix = audio
        else:
            n = min(len(mix), len(audio))
            mix = mix[:n] + audio[:n]

    if mix is None:
        # Everything muted: still produce a valid silent file at the right length/rate.
        any_stem = song.dir / "stems" / f"{next(iter(song.stem_levels))}.wav"
        rate, sample_audio = read_wav(any_stem)
        sample_rate = rate
        mix = np.zeros_like(sample_audio, dtype=np.float32)

    peak = float(np.abs(mix).max()) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak  # avoid clipping when stems are boosted

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, sample_rate, mix.astype(np.float32))
    return out_path
