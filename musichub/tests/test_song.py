import pytest

from musicgen_personas.song import Section, SongStore, split_into_sections


def test_split_into_sections_on_blank_lines():
    lyrics = "Verse line one\nVerse line two\n\nChorus line one\nChorus line two"
    sections = split_into_sections(lyrics)
    assert [s.lyrics for s in sections] == [
        "Verse line one\nVerse line two",
        "Chorus line one\nChorus line two",
    ]
    assert [s.label for s in sections] == ["Section 1", "Section 2"]


def test_split_into_sections_no_blank_lines_is_one_section():
    assert len(split_into_sections("just one block of lyrics")) == 1


def test_section_is_stale_when_never_rendered():
    section = Section(id="a", label="Section 1", lyrics="hello")
    assert section.is_stale


def test_section_is_stale_after_lyrics_edited_post_render():
    section = Section(id="a", label="Section 1", lyrics="hello", rendered_lyrics="hello", audio_file="a.wav")
    assert not section.is_stale
    section.lyrics = "hello there"
    assert section.is_stale


def test_create_and_get(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("My Song", "Aria", "verse one\n\nchorus one")
    assert len(song.sections) == 2

    reloaded = store.get(song.id)
    assert reloaded.title == "My Song"
    assert [s.lyrics for s in reloaded.sections] == ["verse one", "chorus one"]


def test_persists_across_instances(tmp_path):
    song_id = SongStore(root=tmp_path).create("Song", "Aria", "hi").id
    reloaded = SongStore(root=tmp_path).get(song_id)
    assert reloaded.title == "Song"


def test_get_missing_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        SongStore(root=tmp_path).get("nope")


def test_list_sorted_by_updated_at_desc(tmp_path):
    store = SongStore(root=tmp_path)
    first = store.create("First", "Aria", "a")
    second = store.create("Second", "Aria", "b")
    assert [s.id for s in store.list()] == [second.id, first.id]


def test_update_section_lyrics(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "original lyrics")
    section_id = song.sections[0].id
    store.update_section_lyrics(song, section_id, "changed lyrics")
    assert store.get(song.id).section(section_id).lyrics == "changed lyrics"


def test_add_section_appends_by_default(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one")
    store.add_section(song, "Chorus", "chorus lyrics")
    assert [s.label for s in store.get(song.id).sections] == ["Section 1", "Chorus"]


def test_add_section_at_position(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    store.add_section(song, "Intro", "intro lyrics", position=0)
    assert [s.label for s in store.get(song.id).sections] == ["Intro", "Section 1", "Section 2"]


def test_remove_section(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    to_remove = song.sections[0].id
    store.remove_section(song, to_remove)
    remaining = store.get(song.id).sections
    assert len(remaining) == 1
    assert remaining[0].id != to_remove


def test_remove_missing_section_raises(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one")
    with pytest.raises(KeyError):
        store.remove_section(song, "nonexistent")


def test_reorder_sections(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    ids = [s.id for s in song.sections]
    store.reorder_sections(song, list(reversed(ids)))
    assert [s.id for s in store.get(song.id).sections] == list(reversed(ids))


def test_reorder_rejects_mismatched_ids(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "verse one\n\nverse two")
    with pytest.raises(ValueError):
        store.reorder_sections(song, ["not", "the", "right", "ids"])
