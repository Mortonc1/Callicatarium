"""Song generation: renders lyrics through a Persona's voice into a .wav file.

This module lazily imports `bark` (and `torch`) so that persona management
(list/create/remove) works even without the heavy generation dependencies
installed. Install them with:

    pip install -e ".[generate]"

The first call to `generate_song` downloads Bark's model weights (several
GB) and runs noticeably faster on a GPU than a CPU.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .personas import Persona

_SILENCE_SECONDS = 0.35
_MAX_CHUNK_CHARS = 200


def _resolve_history_prompt(persona: Persona) -> str:
    if persona.voice_source_type == "preset":
        return persona.voice_source_value
    if persona.voice_source_type == "npz":
        npz_path = Path(persona.voice_source_value)
        if not npz_path.is_absolute():
            npz_path = Path(__file__).resolve().parent.parent / npz_path
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Custom voice file not found for persona '{persona.name}': {npz_path}"
            )
        return str(npz_path)
    raise ValueError(
        f"Unknown voice_source_type '{persona.voice_source_type}' on persona '{persona.name}'"
    )


def _chunk_lyrics(lyrics: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split lyrics into Bark-sized chunks (Bark caps out around ~13s/chunk)."""
    lines = [line.strip() for line in re.split(r"[\n]+", lyrics) if line.strip()]
    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = f"{current} {line}".strip() if current else line
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [lyrics]


def generate_song(
    persona: Persona,
    lyrics: str,
    out_path: str | Path,
    seed: int | None = None,
) -> Path:
    """Generate a song for `lyrics` in `persona`'s voice and write it to `out_path`.

    Wrap lines you want sung (rather than spoken) in musical notes, e.g.
    "♪ Walking down the street tonight ♪" -- this is Bark's own convention
    for cueing singing/music.
    """
    try:
        from bark import SAMPLE_RATE as BARK_SAMPLE_RATE
        from bark import generate_audio
        from bark.generation import preload_models
    except ImportError as exc:
        raise RuntimeError(
            "The 'bark' package (and torch) must be installed to generate audio.\n"
            'Install with: pip install -e ".[generate]"'
        ) from exc

    if seed is not None:
        import torch

        torch.manual_seed(seed)
        np.random.seed(seed)

    preload_models()
    history_prompt = _resolve_history_prompt(persona)

    silence = np.zeros(int(_SILENCE_SECONDS * BARK_SAMPLE_RATE), dtype=np.float32)
    segments: list[np.ndarray] = []
    for chunk in _chunk_lyrics(lyrics):
        audio = generate_audio(
            chunk,
            history_prompt=history_prompt,
            text_temp=persona.text_temp,
            waveform_temp=persona.waveform_temp,
        )
        segments.append(audio)
        segments.append(silence)

    full_audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)

    from scipy.io.wavfile import write as write_wav

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, BARK_SAMPLE_RATE, full_audio)
    return out_path
