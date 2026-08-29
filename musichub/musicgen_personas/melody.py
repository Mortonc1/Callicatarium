"""Melody-conditioned instrumental generation via MusicGen.

MusicGen (Meta, MIT licensed: https://github.com/facebookresearch/audiocraft)
has a `melody` variant that accepts a reference clip alongside a text
prompt. It extracts a coarse chromagram -- roughly, which pitch classes are
sounding over time -- and generates *new* audio that follows that contour.

What this is and isn't: the reference's actual audio is never copied,
sampled, or remixed; only a low-resolution pitch/rhythm outline informs a
fresh generation. That also caps how close the result can get -- expect
"recognisably the same tune, clearly a different recording", not a
near-duplicate. MusicGen's melody model also generates instrumental music
and does not sing lyrics; vocals come from Bark separately.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_MODEL = "facebook/musicgen-melody"
MAX_SEGMENT_SECONDS = 30.0  # MusicGen's practical generation ceiling per call


def generate_melody_conditioned(
    prompt: str,
    reference_path: str | Path,
    out_path: str | Path,
    duration: float = 30.0,
    model_name: str = DEFAULT_MODEL,
    reference_offset: float = 0.0,
) -> Path:
    """Generate instrumental audio from `prompt`, guided by `reference_path`'s melody.

    `reference_offset` picks which part of the reference to condition on,
    so a long track can guide section-by-section rather than only from its
    opening.
    """
    try:
        import torchaudio
        from audiocraft.models import MusicGen
    except ImportError as exc:
        raise RuntimeError(
            "The 'audiocraft' package must be installed for melody-conditioned generation.\n"
            'Install with: pip install -e ".[melody]"'
        ) from exc

    reference_path = Path(reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(f"No such reference audio: {reference_path}")
    if duration > MAX_SEGMENT_SECONDS:
        raise ValueError(
            f"MusicGen generates at most {MAX_SEGMENT_SECONDS}s per call; "
            f"asked for {duration}s. Generate section by section instead."
        )

    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration)

    waveform, sample_rate = torchaudio.load(str(reference_path))
    start_frame = int(reference_offset * sample_rate)
    end_frame = start_frame + int(duration * sample_rate)
    excerpt = waveform[:, start_frame:end_frame]
    if excerpt.shape[-1] == 0:
        raise ValueError(
            f"reference_offset {reference_offset}s is past the end of {reference_path.name}"
        )

    output = model.generate_with_chroma([prompt], excerpt[None], sample_rate)

    audio = output[0].cpu().numpy()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)  # collapse to mono for consistency with the rest of the pipeline

    from scipy.io.wavfile import write as write_wav

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_path, model.sample_rate, audio.astype(np.float32))
    return out_path
