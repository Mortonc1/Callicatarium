from musicgen_personas.generate import _chunk_lyrics, _next_history_prompt


def test_chunk_lyrics_splits_on_max_chars():
    lyrics = "\n".join(["This is a line of lyrics that is somewhat long"] * 6)
    chunks = _chunk_lyrics(lyrics, max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)  # allows one line to slightly overshoot


def test_chunk_lyrics_short_text_is_single_chunk():
    assert _chunk_lyrics("♪ one short line ♪") == ["♪ one short line ♪"]


def test_chunk_lyrics_ignores_blank_lines():
    chunks = _chunk_lyrics("line one\n\n\nline two")
    assert chunks == ["line one line two"]


def test_next_history_prompt_first_chunk_uses_base():
    assert _next_history_prompt(0, "base", "previous", reset_every=4) == "base"


def test_next_history_prompt_chains_previous_generation():
    assert _next_history_prompt(1, "base", "previous", reset_every=4) == "previous"
    assert _next_history_prompt(3, "base", "previous", reset_every=4) == "previous"


def test_next_history_prompt_resets_on_schedule():
    assert _next_history_prompt(4, "base", "previous", reset_every=4) == "base"
    assert _next_history_prompt(8, "base", "previous", reset_every=4) == "base"


def test_next_history_prompt_falls_back_to_base_when_no_prior_generation():
    assert _next_history_prompt(1, "base", None, reset_every=4) == "base"


def test_next_history_prompt_reset_disabled():
    assert _next_history_prompt(4, "base", "previous", reset_every=0) == "previous"
