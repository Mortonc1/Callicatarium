"""Register a custom cloned voice as a reusable persona.

Producing the voice clip itself isn't done here -- capture one with a
voice-cloning tool that outputs Bark's own `history_prompt` .npz format, e.g.
https://github.com/serp-ai/bark-with-voice-clone (a community project built
directly on Bark). Once you have that .npz file, register it here and it
becomes a persona you can reuse by name forever after, exactly like the
built-in presets.
"""
from __future__ import annotations

from pathlib import Path

from .personas import Persona, PersonaRegistry


def save_cloned_persona(
    registry: PersonaRegistry,
    name: str,
    npz_path: str | Path,
    genre: str = "",
    description: str = "",
    overwrite: bool = False,
) -> Persona:
    npz_path = Path(npz_path).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"No such voice file: {npz_path}")
    if npz_path.suffix != ".npz":
        raise ValueError("Custom voices must be Bark history-prompt .npz files")

    persona = Persona(
        name=name,
        voice_source_type="npz",
        voice_source_value=str(npz_path),
        genre=genre,
        description=description,
    )
    registry.add(persona, overwrite=overwrite)
    return persona
