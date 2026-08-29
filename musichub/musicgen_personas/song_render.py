"""Turns a Song's sections into actual audio.

Each section is rendered independently via `generate_song` (a plain,
stateless call from the persona's base voice -- no chaining from other
sections), then sections are stitched into one final track with a short
crossfade at each boundary so joins don't click.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .personas import PersonaRegistry
from .song import Song, SongStore

_CROSSFADE_MS = 120


def regenerate_section(store: SongStore, song: Song, section_id: str, seed: int | None = None) -> Song:
    """Render (or re-render) exactly one section's audio. Other sections are untouched."""
    from .generate import generate_song

    section = song.section(section_id)
    persona = PersonaRegistry().get(song.persona_name)
    effective_seed = seed if seed is not None else section.seed

    audio_filename = f"{section.id}.wav"
    out_path = song.dir / audio_filename
    generate_song(persona, section.lyrics, out_path, seed=effective_seed, style_key=song.style_key)

    section.audio_file = audio_filename
    section.rendered_lyrics = section.lyrics
    if seed is not None:
        section.seed = seed
    store.save(song)
    return song


def _crossfade_concat(a: np.ndarray, b: np.ndarray, fade_len: int) -> np.ndarray:
    fade_len = min(fade_len, len(a), len(b))
    if fade_len <= 0:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    mixed = a[-fade_len:] * fade_out + b[:fade_len] * fade_in
    return np.concatenate([a[:-fade_len], mixed, b[fade_len:]])


def render_song(song: Song, out_path: str | Path) -> Path:
    """Stitch every section's already-rendered audio into one final track.

    Raises ValueError if any section hasn't been rendered yet, or has been
    edited since it was last rendered (`regenerate_section` first).
    """
    from scipy.io.wavfile import read as read_wav
    from scipy.io.wavfile import write as write_wav

    if not song.sections:
        raise ValueError("Song has no sections")
    stale = [s.label for s in song.sections if s.is_stale]
    if stale:
        raise ValueError(f"These sections need regenerating before render: {', '.join(stale)}")

    sample_rate = None
    clips: list[np.ndarray] = []
    for section in song.sections:
        rate, audio = read_wav(song.dir / section.audio_file)
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise ValueError(f"Sample rate mismatch in section '{section.label}'")
        clips.append(audio.astype(np.float32))

    fade_len = int(sample_rate * _CROSSFADE_MS / 1000)
    full_audio = clips[0]
    for clip in clips[1:]:
        full_audio = _crossfade_concat(full_audio, clip, fade_len)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, sample_rate, full_audio.astype(np.float32))
    return out_path
