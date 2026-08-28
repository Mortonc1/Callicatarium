"""Persona registry: named, reusable voice profiles for song generation.

A Persona bundles a voice (either one of Bark's built-in speaker presets, or
a custom cloned voice saved as a Bark `history_prompt` .npz file) together
with genre/description metadata and default generation parameters, so the
same voice can be invoked by name across many songs.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "personas" / "registry.json"


@dataclass
class Persona:
    name: str
    voice_source_type: str  # "preset" (built-in Bark speaker) or "npz" (custom cloned voice)
    voice_source_value: str  # e.g. "v2/en_speaker_6" or "personas/custom/aria.npz"
    genre: str = ""
    description: str = ""
    text_temp: float = 0.7
    waveform_temp: float = 0.7
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Persona":
        return cls(**data)


class PersonaRegistry:
    """Load/save/manage a JSON-backed collection of reusable voice personas."""

    def __init__(self, path: Path | str = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)
        self._personas: dict[str, Persona] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = self.path.read_text().strip()
        data = json.loads(raw) if raw else {}
        self._personas = {name: Persona.from_dict(p) for name, p in data.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: p.to_dict() for name, p in self._personas.items()}
        self.path.write_text(json.dumps(data, indent=2) + "\n")

    def add(self, persona: Persona, overwrite: bool = False) -> None:
        if persona.name in self._personas and not overwrite:
            raise ValueError(
                f"Persona '{persona.name}' already exists. Pass overwrite=True to replace it."
            )
        self._personas[persona.name] = persona
        self.save()

    def get(self, name: str) -> Persona:
        try:
            return self._personas[name]
        except KeyError:
            known = ", ".join(sorted(self._personas)) or "(none)"
            raise KeyError(f"No persona named '{name}'. Known personas: {known}") from None

    def list(self) -> list[Persona]:
        return sorted(self._personas.values(), key=lambda p: p.name.lower())

    def remove(self, name: str) -> None:
        if name not in self._personas:
            raise KeyError(f"No persona named '{name}'.")
        del self._personas[name]
        self.save()
