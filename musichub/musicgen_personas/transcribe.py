"""Transcribes a reference track's vocals into timestamped lyric segments.

Uses faster-whisper (https://github.com/SYSTRAN/faster-whisper, MIT), a
reimplementation of OpenAI's Whisper. Whisper is trained on speech, not
singing -- expect to hand-correct the result, especially where vocals sit
low in a mix or the delivery is stylised. Running separation first and
transcribing the isolated vocal stem helps a lot; `transcribe_song_vocals`
in `recreate.py` does exactly that.

The timestamps matter as much as the words here: they're what lets a
reference track seed a song project whose sections line up with the
original's actual structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .licenses import assert_commercial_ok

DEFAULT_MODEL = "base"


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


def transcribe(
    audio_path: str | Path,
    model_size: str = DEFAULT_MODEL,
    language: str | None = None,
) -> list[TranscriptSegment]:
    """Transcribe `audio_path` into timestamped segments."""
    assert_commercial_ok("whisper")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "The 'faster-whisper' package must be installed to transcribe.\n"
            'Install with: pip install -e ".[transcribe]"'
        ) from exc

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"No such audio file: {audio_path}")

    try:
        model = WhisperModel(model_size, device="auto", compute_type="default")
    except Exception as exc:  # noqa: BLE001 -- re-raised below with context
        # The first run downloads weights from Hugging Face. A bare
        # "403 Forbidden" here reads as a bug in this project; it usually
        # means the network (proxy, firewall, offline machine) blocked that.
        raise RuntimeError(
            f"Couldn't load the Whisper '{model_size}' model: {exc}\n"
            "The first run downloads it from huggingface.co -- check that host is "
            "reachable, or pre-download the model on a machine that can reach it."
        ) from exc
    segments, _info = model.transcribe(str(audio_path), language=language)
    return [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments
        if s.text.strip()
    ]


def group_into_sections(
    segments: list[TranscriptSegment], gap_threshold: float = 2.0, max_lines: int = 4
) -> list[list[TranscriptSegment]]:
    """Group segments into section-sized blocks.

    Starts a new block on a long instrumental gap (a natural verse/chorus
    boundary) or once a block reaches `max_lines`, so sections come out at
    a size that's actually workable to regenerate one at a time.
    """
    if not segments:
        return []

    blocks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = [segments[0]]
    for prev, seg in zip(segments, segments[1:]):
        gap = seg.start - prev.end
        if gap >= gap_threshold or len(current) >= max_lines:
            blocks.append(current)
            current = [seg]
        else:
            current.append(seg)
    blocks.append(current)
    return blocks


def segments_to_lyrics(segments: list[TranscriptSegment]) -> str:
    return "\n".join(s.text for s in segments)
