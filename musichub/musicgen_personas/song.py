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
from dataclasses import asdict, dataclass, field
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

    @property
    def is_stale(self) -> bool:
        return self.audio_file is None or self.rendered_lyrics != self.lyrics

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_stale"] = self.is_stale
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Section":
        data = {k: v for k, v in data.items() if k != "is_stale"}
        return cls(**data)


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
    songs_dir: Path = field(default=DEFAULT_SONGS_DIR, repr=False, compare=False)

    @property
    def dir(self) -> Path:
        return self.songs_dir / self.id

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
