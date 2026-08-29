"""Song editor: a song is an ordered list of independently-editable sections.

Each section renders to its own audio file. Editing and regenerating one
section's lyrics never touches any other section's audio -- unlike feeding
a whole song through as one lyrics blob, where a small text change can
shift chunk boundaries and regenerate audio well beyond the edited word.
This module holds the data model and CRUD only; no Bark/torch dependency,
so it works even without the generation extras installed. See
`song_render.py` for actually producing audio.
"""
from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_SONGS_DIR = Path(__file__).resolve().parent.parent / "songs"


@dataclass
class Section:
    id: str
    label: str
    lyrics: str
    seed: int | None = None
    rendered_lyrics: str | None = None  # lyrics text as of the last successful render
    audio_file: str | None = None  # filename within the song's directory
    # Where this section sits in an imported reference track, when the song
    # was built from one -- used to condition generation on the matching slice.
    ref_start: float | None = None
    ref_end: float | None = None
    instrumental_file: str | None = None  # melody-conditioned backing, if generated
    # How this section was carved out, when it came from split_section --
    # lets merge_sections rejoin words with spaces and lines with newlines.
    split_kind: str | None = None

    @property
    def is_stale(self) -> bool:
        return self.audio_file is None or self.rendered_lyrics != self.lyrics

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_stale"] = self.is_stale
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Song:
    id: str
    title: str
    persona_name: str
    sections: list[Section] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # {stem_name: {"gain": float, "muted": bool}}, populated by separate_song_stems.
    stem_levels: dict[str, dict] = field(default_factory=dict)
    # Path (relative to song.dir) of an imported guide track, if this song
    # was built from one. See recreate.py.
    reference_file: str | None = None
    songs_dir: Path = field(default=DEFAULT_SONGS_DIR, repr=False, compare=False)

    @property
    def dir(self) -> Path:
        return self.songs_dir / self.id

    def reference_path(self) -> Path | None:
        if not self.reference_file:
            return None
        return self.dir / self.reference_file

    def section(self, section_id: str) -> Section:
        for s in self.sections:
            if s.id == section_id:
                return s
        raise KeyError(f"No section '{section_id}' in song '{self.title}'")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "persona_name": self.persona_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sections": [s.to_dict() for s in self.sections],
            "has_full_render": (self.dir / "full.wav").exists(),
            "stem_levels": self.stem_levels,
            "reference_file": self.reference_file,
            "has_reference": self.reference_file is not None,
        }

    @classmethod
    def from_dict(cls, data: dict, songs_dir: Path = DEFAULT_SONGS_DIR) -> "Song":
        return cls(
            id=data["id"],
            title=data["title"],
            persona_name=data["persona_name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            sections=[Section.from_dict(s) for s in data["sections"]],
            stem_levels=data.get("stem_levels", {}),
            reference_file=data.get("reference_file"),
            songs_dir=songs_dir,
        )


def split_into_sections(lyrics: str) -> list[Section]:
    """Split on blank lines into labeled sections (Section 1, Section 2, ...)."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", lyrics) if b.strip()]
    if not blocks:
        blocks = [lyrics.strip()]
    return [
        Section(id=uuid.uuid4().hex[:8], label=f"Section {i + 1}", lyrics=block)
        for i, block in enumerate(blocks)
    ]


class SongStore:
    """JSON-backed collection of songs, one subdirectory per song for audio files."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else DEFAULT_SONGS_DIR

    def _path(self, song_id: str) -> Path:
        return self.root / song_id / "song.json"

    def create(self, title: str, persona_name: str, lyrics: str) -> Song:
        song = Song(id=uuid.uuid4().hex[:12], title=title, persona_name=persona_name, songs_dir=self.root)
        song.sections = split_into_sections(lyrics)
        self.save(song)
        return song

    def get(self, song_id: str) -> Song:
        path = self._path(song_id)
        if not path.exists():
            raise KeyError(f"No song '{song_id}'")
        return Song.from_dict(json.loads(path.read_text()), songs_dir=self.root)

    def list(self) -> list[Song]:
        songs = []
        if not self.root.exists():
            return songs
        for song_dir in self.root.iterdir():
            song_file = song_dir / "song.json"
            if song_file.exists():
                songs.append(Song.from_dict(json.loads(song_file.read_text()), songs_dir=self.root))
        return sorted(songs, key=lambda s: s.updated_at, reverse=True)

    def save(self, song: Song) -> None:
        song.updated_at = time.time()
        song.dir.mkdir(parents=True, exist_ok=True)
        self._path(song.id).write_text(json.dumps(song.to_dict(), indent=2) + "\n")

    def delete(self, song_id: str) -> None:
        song_dir = self.root / song_id
        if not song_dir.exists():
            raise KeyError(f"No song '{song_id}'")
        shutil.rmtree(song_dir)

    def update_section_lyrics(self, song: Song, section_id: str, lyrics: str) -> Song:
        section = song.section(section_id)
        section.lyrics = lyrics
        self.save(song)
        return song

    def add_section(self, song: Song, label: str, lyrics: str, position: int | None = None) -> Song:
        section = Section(id=uuid.uuid4().hex[:8], label=label, lyrics=lyrics)
        if position is None:
            song.sections.append(section)
        else:
            song.sections.insert(position, section)
        self.save(song)
        return song

    def split_section(self, song: Song, section_id: str, granularity: str = "lines") -> Song:
        """Replace one section with several smaller ones.

        `granularity` is "lines" (one section per line) or "words" (one per
        word). Splitting finer makes edits more surgical -- regenerating one
        word touches only that word -- at the cost of flow, since each
        section is generated as its own standalone utterance. Choose per
        edit: split down to fix something, merge back for a smoother take.

        Reference timing, when present, is divided across the new sections
        in proportion to their text length, so each still knows roughly
        which slice of a guide track it came from.
        """
        if granularity not in ("lines", "words"):
            raise ValueError(f"granularity must be 'lines' or 'words', got {granularity!r}")

        index = next((i for i, s in enumerate(song.sections) if s.id == section_id), None)
        if index is None:
            raise KeyError(f"No section '{section_id}' in song '{song.title}'")
        section = song.sections[index]

        if granularity == "lines":
            pieces = [ln.strip() for ln in section.lyrics.splitlines() if ln.strip()]
        else:
            pieces = section.lyrics.split()
        if len(pieces) <= 1:
            raise ValueError(f"Section '{section.label}' can't be split any further by {granularity}")

        total_chars = sum(len(p) for p in pieces) or 1
        span = None
        if section.ref_start is not None and section.ref_end is not None:
            span = section.ref_end - section.ref_start

        new_sections = []
        consumed = 0
        for i, piece in enumerate(pieces):
            ref_start = ref_end = None
            if span is not None:
                ref_start = section.ref_start + span * (consumed / total_chars)
                ref_end = section.ref_start + span * ((consumed + len(piece)) / total_chars)
            consumed += len(piece)
            new_sections.append(
                Section(
                    id=uuid.uuid4().hex[:8],
                    label=f"{section.label}.{i + 1}",
                    lyrics=piece,
                    seed=section.seed,
                    split_kind=granularity,
                    ref_start=ref_start,
                    ref_end=ref_end,
                )
            )

        # The old section's rendered audio no longer corresponds to any one
        # new section, so drop it; the pieces start stale.
        if section.audio_file:
            (song.dir / section.audio_file).unlink(missing_ok=True)

        song.sections[index : index + 1] = new_sections
        self.save(song)
        return song

    def merge_sections(self, song: Song, section_ids: list[str], label: str | None = None) -> Song:
        """Merge adjacent sections back into one, so it regenerates as a single take."""
        if len(section_ids) < 2:
            raise ValueError("Merging needs at least two sections")

        positions = []
        for sid in section_ids:
            pos = next((i for i, s in enumerate(song.sections) if s.id == sid), None)
            if pos is None:
                raise KeyError(f"No section '{sid}' in song '{song.title}'")
            positions.append(pos)
        positions.sort()
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise ValueError("Only adjacent sections can be merged")

        merged_from = [song.sections[p] for p in positions]
        ref_starts = [s.ref_start for s in merged_from if s.ref_start is not None]
        ref_ends = [s.ref_end for s in merged_from if s.ref_end is not None]

        # Rejoin the way the pieces were split: words back into a line,
        # lines back into a block. Anything else falls back to newlines.
        joiner = " " if all(s.split_kind == "words" for s in merged_from) else "\n"

        merged = Section(
            id=uuid.uuid4().hex[:8],
            label=label or merged_from[0].label.split(".")[0],
            lyrics=joiner.join(s.lyrics for s in merged_from),
            seed=merged_from[0].seed,
            ref_start=min(ref_starts) if ref_starts else None,
            ref_end=max(ref_ends) if ref_ends else None,
        )

        for s in merged_from:
            if s.audio_file:
                (song.dir / s.audio_file).unlink(missing_ok=True)

        song.sections[positions[0] : positions[-1] + 1] = [merged]
        self.save(song)
        return song

    def remove_section(self, song: Song, section_id: str) -> Song:
        song.section(section_id)  # raises KeyError if missing
        removed = song.section(section_id)
        song.sections = [s for s in song.sections if s.id != section_id]
        if removed.audio_file:
            (song.dir / removed.audio_file).unlink(missing_ok=True)
        self.save(song)
        return song

    def reorder_sections(self, song: Song, section_ids: list[str]) -> Song:
        if sorted(section_ids) != sorted(s.id for s in song.sections):
            raise ValueError("reorder must include exactly the song's current section ids, no more/fewer")
        by_id = {s.id: s for s in song.sections}
        song.sections = [by_id[sid] for sid in section_ids]
        self.save(song)
        return song
