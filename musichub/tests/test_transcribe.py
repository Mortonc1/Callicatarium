from musicgen_personas.transcribe import (
    TranscriptSegment,
    group_into_sections,
    segments_to_lyrics,
)


def seg(start, end, text):
    return TranscriptSegment(start=start, end=end, text=text)


def test_group_empty():
    assert group_into_sections([]) == []


def test_group_splits_on_long_gap():
    segments = [
        seg(0.0, 2.0, "line one"),
        seg(2.1, 4.0, "line two"),
        seg(10.0, 12.0, "after the instrumental break"),
    ]
    blocks = group_into_sections(segments, gap_threshold=2.0)
    assert len(blocks) == 2
    assert [s.text for s in blocks[0]] == ["line one", "line two"]
    assert [s.text for s in blocks[1]] == ["after the instrumental break"]


def test_group_splits_on_max_lines():
    segments = [seg(i, i + 0.5, f"line {i}") for i in range(9)]
    blocks = group_into_sections(segments, gap_threshold=99.0, max_lines=4)
    assert [len(b) for b in blocks] == [4, 4, 1]


def test_group_keeps_tight_lines_together():
    segments = [seg(0.0, 1.0, "a"), seg(1.1, 2.0, "b")]
    blocks = group_into_sections(segments, gap_threshold=2.0, max_lines=4)
    assert len(blocks) == 1


def test_gap_measured_from_end_not_start():
    # A long line followed closely by the next is NOT a section break, even
    # though the two start times are far apart.
    segments = [seg(0.0, 9.0, "a long held line"), seg(9.2, 10.0, "next line")]
    blocks = group_into_sections(segments, gap_threshold=2.0)
    assert len(blocks) == 1


def test_segments_to_lyrics_joins_with_newlines():
    assert segments_to_lyrics([seg(0, 1, "one"), seg(1, 2, "two")]) == "one\ntwo"
