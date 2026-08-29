"""Builds a new, editable song project guided by a reference track.

The flow, end to end:

1. Optionally split the reference with Demucs and keep the vocal stem --
   Whisper transcribes isolated vocals far more reliably than a full mix.
2. Transcribe it into timestamped lines.
3. Group those lines into sections at natural gaps, and create a song
   project whose sections carry the *original's* timing (`ref_start` /
   `ref_end`).
4. Each section can then be regenerated as vocals (Bark, via the normal
   section pipeline) and/or as a melody-conditioned instrumental bed
   (MusicGen), conditioned on the matching slice of the reference.

Nothing here copies the reference's audio into the output. The transcript
is text, and melody conditioning uses only a coarse pitch contour -- see
`melody.py`. What you keep is your own lyrics and song structure; what
gets regenerated is entirely new audio.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from .song import Section, Song, SongStore

DEFAULT_INSTRUMENTAL_PROMPT = "instrumental backing track"


def _reference_dir(song: Song) -> Path:
    return song.dir / "reference"


def import_reference(store: SongStore, song: Song, reference_path: str | Path) -> Path:
    """Copy a reference track into the song's own directory and record it."""
    import shutil

    reference_path = Path(reference_path)
    if not reference_path.exists():
        raise FileNotFoundError(f"No such reference audio: {reference_path}")

    ref_dir = _reference_dir(song)
    ref_dir.mkdir(parents=True, exist_ok=True)
    dest = ref_dir / f"reference{reference_path.suffix}"
    shutil.copy2(reference_path, dest)

    song.reference_file = str(dest.relative_to(song.dir))
    store.save(song)
    return dest


def isolate_reference_vocals(song: Song) -> Path:
    """Split the reference and return its vocal stem, for cleaner transcription."""
    from .stems import cleanup_work_dir, separate_stems

    reference = song.reference_path()
    if reference is None:
        raise ValueError("This song has no reference track imported")

    work_dir = _reference_dir(song) / "_demucs_work"
    try:
        produced = separate_stems(reference, work_dir)
        if "vocals" not in produced:
            raise RuntimeError(f"Demucs produced no vocal stem (got: {', '.join(produced)})")
        vocals_dest = _reference_dir(song) / "reference_vocals.wav"
        produced["vocals"].replace(vocals_dest)
    finally:
        cleanup_work_dir(work_dir)
    return vocals_dest


def build_sections_from_reference(
    store: SongStore,
    song: Song,
    use_isolated_vocals: bool = True,
    model_size: str = "base",
    gap_threshold: float = 2.0,
    max_lines: int = 4,
) -> Song:
    """Transcribe the reference and replace the song's sections with what it finds."""
    from .transcribe import group_into_sections, segments_to_lyrics, transcribe

    reference = song.reference_path()
    if reference is None:
        raise ValueError("This song has no reference track imported")

    source = isolate_reference_vocals(song) if use_isolated_vocals else reference
    segments = transcribe(source, model_size=model_size)
    if not segments:
        raise RuntimeError(
            "Transcription found no lyrics. If the vocals are quiet or heavily "
            "processed, try a larger model_size, or paste the lyrics in by hand."
        )

    sections = []
    for i, block in enumerate(group_into_sections(segments, gap_threshold, max_lines)):
        sections.append(
            Section(
                id=uuid.uuid4().hex[:8],
                label=f"Section {i + 1}",
                lyrics=segments_to_lyrics(block),
                ref_start=block[0].start,
                ref_end=block[-1].end,
            )
        )

    song.sections = sections
    store.save(song)
    return song


def create_song_from_reference(
    store: SongStore,
    title: str,
    persona_name: str,
    reference_path: str | Path,
    use_isolated_vocals: bool = True,
    model_size: str = "base",
) -> Song:
    """One-shot: new song project seeded from a reference track's lyrics and structure."""
    song = store.create(title, persona_name, lyrics="(pending transcription)")
    import_reference(store, song, reference_path)
    return build_sections_from_reference(
        store, song, use_isolated_vocals=use_isolated_vocals, model_size=model_size
    )


def generate_section_instrumental(
    store: SongStore,
    song: Song,
    section_id: str,
    prompt: str = DEFAULT_INSTRUMENTAL_PROMPT,
    duration: float | None = None,
) -> Song:
    """Generate a melody-conditioned instrumental bed for one section.

    Conditions on the slice of the reference that section came from, so
    each part of the new track follows the corresponding part of the old.
    """
    from .melody import MAX_SEGMENT_SECONDS, generate_melody_conditioned

    reference = song.reference_path()
    if reference is None:
        raise ValueError("This song has no reference track imported")

    section = song.section(section_id)
    if section.ref_start is None:
        raise ValueError(
            f"Section '{section.label}' has no reference timing -- it wasn't built from "
            "the reference track, so there's no matching slice to condition on."
        )

    if duration is None:
        duration = (section.ref_end or section.ref_start) - section.ref_start
    duration = max(1.0, min(duration, MAX_SEGMENT_SECONDS))

    filename = f"{section.id}_instrumental.wav"
    generate_melody_conditioned(
        prompt=prompt,
        reference_path=reference,
        out_path=song.dir / filename,
        duration=duration,
        reference_offset=section.ref_start,
    )
    section.instrumental_file = filename
    store.save(song)
    return song
