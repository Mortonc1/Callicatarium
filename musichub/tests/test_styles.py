import pytest

from musicgen_personas.styles import (
    MUSIC_NOTE,
    STYLES,
    apply_vocal_style,
    get,
    instrumental_prompt_for,
)


def test_get_known_style():
    assert get("synthwave").name == "Synthwave"


def test_get_unknown_style_lists_options():
    with pytest.raises(KeyError, match="Known styles"):
        get("dubstep-polka")


def test_every_style_has_an_instrumental_prompt():
    for key, style in STYLES.items():
        assert style.instrumental_prompt.strip(), key
        assert style.key == key


def test_sung_style_wraps_lines_in_music_notes():
    out = apply_vocal_style("hold the line\nwe are alive", "pop")
    assert out.splitlines() == [
        f"{MUSIC_NOTE} hold the line {MUSIC_NOTE}",
        f"{MUSIC_NOTE} we are alive {MUSIC_NOTE}",
    ]


def test_spoken_style_does_not_add_singing_cue():
    out = apply_vocal_style("just say this plainly", "spoken")
    assert MUSIC_NOTE not in out


def test_hiphop_is_spoken_not_sung():
    assert get("hiphop").sung is False
    assert MUSIC_NOTE not in apply_vocal_style("rhythm over the beat", "hiphop")


def test_existing_music_notes_are_not_doubled():
    already = f"{MUSIC_NOTE} hold the line {MUSIC_NOTE}"
    assert apply_vocal_style(already, "pop") == already


def test_no_style_leaves_lyrics_untouched():
    assert apply_vocal_style("plain lyrics", None) == "plain lyrics"


def test_vocal_cues_are_prepended_when_a_style_defines_them():
    out = apply_vocal_style("slow and sad", "ballad")
    assert out.startswith("[sighs]")


def test_blank_lines_are_dropped():
    out = apply_vocal_style("one\n\n\ntwo", "pop")
    assert len(out.splitlines()) == 2


def test_instrumental_prompt_uses_the_style():
    assert "synthwave" in instrumental_prompt_for("synthwave")


def test_instrumental_prompt_appends_extra_detail():
    out = instrumental_prompt_for("rock", extra="120 bpm, minor key")
    assert out.startswith("driving rock band")
    assert out.endswith("120 bpm, minor key")


def test_instrumental_prompt_without_style_is_generic():
    assert instrumental_prompt_for(None) == "instrumental backing track"
