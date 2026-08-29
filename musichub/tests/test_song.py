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


def test_split_section_by_lines(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "line one\nline two\nline three")
    section_id = song.sections[0].id
    store.split_section(song, section_id, granularity="lines")

    reloaded = store.get(song.id)
    assert [s.lyrics for s in reloaded.sections] == ["line one", "line two", "line three"]
    assert [s.label for s in reloaded.sections] == ["Section 1.1", "Section 1.2", "Section 1.3"]
    assert all(s.is_stale for s in reloaded.sections)


def test_split_section_by_words(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "hold the line")
    store.split_section(song, song.sections[0].id, granularity="words")
    assert [s.lyrics for s in store.get(song.id).sections] == ["hold", "the", "line"]


def test_split_only_affects_the_target_section(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "a one\na two\n\nchorus here")
    first, second = song.sections[0].id, song.sections[1].id
    store.split_section(song, first, granularity="lines")

    sections = store.get(song.id).sections
    assert [s.lyrics for s in sections] == ["a one", "a two", "chorus here"]
    assert sections[-1].id == second  # untouched, same identity


def test_split_divides_reference_timing(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "aaa\nbbb")
    section = song.sections[0]
    section.ref_start, section.ref_end = 10.0, 20.0
    store.save(song)
    store.split_section(song, section.id, granularity="lines")

    sections = store.get(song.id).sections
    assert sections[0].ref_start == 10.0
    assert sections[0].ref_end == 15.0  # equal-length lines split the span evenly
    assert sections[1].ref_start == 15.0
    assert sections[1].ref_end == 20.0


def test_split_unsplittable_section_raises(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "single")
    with pytest.raises(ValueError, match="can't be split"):
        store.split_section(song, song.sections[0].id, granularity="lines")


def test_split_rejects_bad_granularity(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "a\nb")
    with pytest.raises(ValueError, match="granularity"):
        store.split_section(song, song.sections[0].id, granularity="syllables")


def test_merge_sections_recombines(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "one\n\ntwo\n\nthree")
    ids = [s.id for s in song.sections]
    store.merge_sections(song, ids[:2])

    sections = store.get(song.id).sections
    assert [s.lyrics for s in sections] == ["one\ntwo", "three"]


def test_merge_preserves_outer_reference_bounds(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "one\n\ntwo")
    song.sections[0].ref_start, song.sections[0].ref_end = 5.0, 8.0
    song.sections[1].ref_start, song.sections[1].ref_end = 8.0, 12.0
    store.save(song)
    store.merge_sections(song, [s.id for s in song.sections])

    merged = store.get(song.id).sections[0]
    assert merged.ref_start == 5.0
    assert merged.ref_end == 12.0


def test_merge_rejects_non_adjacent(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "one\n\ntwo\n\nthree")
    ids = [s.id for s in song.sections]
    with pytest.raises(ValueError, match="adjacent"):
        store.merge_sections(song, [ids[0], ids[2]])


def test_merge_needs_two_sections(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "one\n\ntwo")
    with pytest.raises(ValueError, match="at least two"):
        store.merge_sections(song, [song.sections[0].id])


def test_split_then_merge_round_trips_lyrics(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "line one\nline two")
    store.split_section(song, song.sections[0].id, granularity="lines")
    store.merge_sections(song, [s.id for s in song.sections])
    assert store.get(song.id).sections[0].lyrics == "line one\nline two"


def test_split_then_merge_round_trips_words(tmp_path):
    # Words must rejoin with spaces, not newlines -- otherwise splitting a
    # line to fix one word and merging back mangles it into separate lines.
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "hold the line tonight")
    store.split_section(song, song.sections[0].id, granularity="words")
    store.merge_sections(song, [s.id for s in song.sections])
    assert store.get(song.id).sections[0].lyrics == "hold the line tonight"


def test_edit_one_word_then_merge_back(tmp_path):
    # The workflow this is all for: split to isolate a word, change it,
    # merge back, and the line reads correctly with only that word changed.
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "hold the line tonight")
    store.split_section(song, song.sections[0].id, granularity="words")
    store.update_section_lyrics(song, song.sections[3].id, "forever")
    store.merge_sections(song, [s.id for s in song.sections])
    assert store.get(song.id).sections[0].lyrics == "hold the line forever"


def test_song_style_persists(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "a line", style_key="synthwave")
    assert store.get(song.id).style_key == "synthwave"


def test_set_style_validates_and_persists(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "a line")
    assert store.get(song.id).style_key is None
    store.set_style(song, "lofi")
    assert store.get(song.id).style_key == "lofi"


def test_set_style_rejects_unknown(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "a line")
    with pytest.raises(KeyError):
        store.set_style(song, "not-a-real-genre")


def test_set_style_can_be_cleared(tmp_path):
    store = SongStore(root=tmp_path)
    song = store.create("Song", "Aria", "a line", style_key="rock")
    store.set_style(song, None)
    assert store.get(song.id).style_key is None
