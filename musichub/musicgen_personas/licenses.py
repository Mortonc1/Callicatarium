"""What each model's weights are licensed under, and whether you can sell the output.

Model *code* and model *weights* are often licensed differently -- MusicGen's
code is MIT while its weights are CC-BY-NC -- so what matters for "can I sell
this track" is the weights. This module records that per model and provides a
gate you can switch on to make commercially-unusable models refuse to run at
all, rather than silently producing audio you can't release.

Turn the gate on by setting MUSICHUB_COMMERCIAL_ONLY=1 in the environment.

None of this is legal advice, and license terms change. The `url` on each
entry points at the authoritative source -- read it before building a
catalogue on any of this.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

COMMERCIAL_ONLY_ENV = "MUSICHUB_COMMERCIAL_ONLY"

# commercial_use values
YES = "yes"  # permissive; sell what you make
NO = "no"  # non-commercial weights; personal/research only
CONDITIONAL = "conditional"  # allowed, but with strings attached -- read `notes`


@dataclass(frozen=True)
class ModelLicense:
    key: str
    name: str
    used_for: str
    weights_license: str
    commercial_use: str
    notes: str
    url: str


MODEL_LICENSES: dict[str, ModelLicense] = {
    "bark": ModelLicense(
        key="bark",
        name="Bark (suno/bark)",
        used_for="vocals",
        weights_license="MIT",
        commercial_use=YES,
        notes="Code and weights both MIT.",
        url="https://github.com/suno-ai/bark",
    ),
    "demucs": ModelLicense(
        key="demucs",
        name="Demucs (htdemucs)",
        used_for="stem separation",
        weights_license="MIT",
        commercial_use=YES,
        notes="Code and weights both MIT.",
        url="https://github.com/facebookresearch/demucs",
    ),
    "whisper": ModelLicense(
        key="whisper",
        name="Whisper via faster-whisper",
        used_for="transcription",
        weights_license="MIT",
        commercial_use=YES,
        notes="OpenAI released Whisper's weights under MIT; faster-whisper is MIT too.",
        url="https://github.com/openai/whisper",
    ),
    "musicgen": ModelLicense(
        key="musicgen",
        name="MusicGen (facebook/musicgen-melody)",
        used_for="melody-conditioned instrumentals",
        weights_license="CC-BY-NC-4.0",
        commercial_use=NO,
        notes=(
            "Audiocraft's CODE is MIT but the WEIGHTS are CC-BY-NC 4.0 -- "
            "non-commercial only. Meta's license does not cover commercially "
            "releasing music the model generates."
        ),
        url="https://huggingface.co/facebook/musicgen-melody",
    ),
    "stable-audio-open": ModelLicense(
        key="stable-audio-open",
        name="Stable Audio Open",
        used_for="melody-conditioned instrumentals (alternative to MusicGen)",
        weights_license="Stability AI Community License",
        commercial_use=CONDITIONAL,
        notes=(
            "Free commercial use for individuals/organisations under US$1M annual "
            "revenue; above that Stability requires an enterprise licence. Verify "
            "the current terms yourself before releasing commercially."
        ),
        url="https://huggingface.co/stabilityai/stable-audio-open-1.0",
    ),
}


def commercial_only() -> bool:
    return os.environ.get(COMMERCIAL_ONLY_ENV, "").strip().lower() in ("1", "true", "yes", "on")


class NonCommercialModelError(RuntimeError):
    """Raised when a non-commercial model is requested while the gate is on."""


def assert_commercial_ok(model_key: str) -> None:
    """Refuse to run a non-commercially-licensed model when the gate is on.

    Models marked CONDITIONAL are allowed through -- their terms permit
    commercial use, just with strings -- so read `notes` for those.
    """
    if not commercial_only():
        return
    entry = MODEL_LICENSES.get(model_key)
    if entry is None:
        raise NonCommercialModelError(
            f"{COMMERCIAL_ONLY_ENV} is set and '{model_key}' has no recorded licence, "
            "so it's being refused rather than assumed safe."
        )
    if entry.commercial_use == NO:
        raise NonCommercialModelError(
            f"{entry.name} is refused because {COMMERCIAL_ONLY_ENV} is set.\n"
            f"  Weights licence: {entry.weights_license}\n"
            f"  {entry.notes}\n"
            f"  {entry.url}\n"
            f"Unset {COMMERCIAL_ONLY_ENV} to use it for personal/non-commercial work, "
            f"or pick a model whose weights permit commercial use."
        )


def summary_rows() -> list[ModelLicense]:
    order = {YES: 0, CONDITIONAL: 1, NO: 2}
    return sorted(MODEL_LICENSES.values(), key=lambda m: (order[m.commercial_use], m.name))
