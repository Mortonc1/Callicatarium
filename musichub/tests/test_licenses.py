import pytest

from musicgen_personas.licenses import (
    COMMERCIAL_ONLY_ENV,
    MODEL_LICENSES,
    NO,
    NonCommercialModelError,
    assert_commercial_ok,
    commercial_only,
    summary_rows,
)


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv(COMMERCIAL_ONLY_ENV, raising=False)
    assert commercial_only() is False
    assert_commercial_ok("musicgen")  # allowed when the gate is off


@pytest.mark.parametrize("value", ["1", "true", "yes", "ON"])
def test_gate_recognises_truthy_values(monkeypatch, value):
    monkeypatch.setenv(COMMERCIAL_ONLY_ENV, value)
    assert commercial_only() is True


@pytest.mark.parametrize("value", ["0", "false", "no", ""])
def test_gate_recognises_falsy_values(monkeypatch, value):
    monkeypatch.setenv(COMMERCIAL_ONLY_ENV, value)
    assert commercial_only() is False


def test_gate_blocks_non_commercial_model(monkeypatch):
    monkeypatch.setenv(COMMERCIAL_ONLY_ENV, "1")
    with pytest.raises(NonCommercialModelError, match="CC-BY-NC"):
        assert_commercial_ok("musicgen")


@pytest.mark.parametrize("key", ["bark", "demucs", "whisper"])
def test_gate_allows_mit_models(monkeypatch, key):
    monkeypatch.setenv(COMMERCIAL_ONLY_ENV, "1")
    assert_commercial_ok(key)  # must not raise


def test_gate_allows_conditional_models(monkeypatch):
    # Conditional licences permit commercial use, just with strings attached.
    monkeypatch.setenv(COMMERCIAL_ONLY_ENV, "1")
    assert_commercial_ok("stable-audio-open")


def test_gate_refuses_unknown_model_rather_than_assuming_safe(monkeypatch):
    monkeypatch.setenv(COMMERCIAL_ONLY_ENV, "1")
    with pytest.raises(NonCommercialModelError, match="no recorded licence"):
        assert_commercial_ok("some-new-model")


def test_every_model_the_project_uses_is_recorded():
    for key in ("bark", "demucs", "whisper", "musicgen"):
        assert key in MODEL_LICENSES


def test_summary_lists_commercial_safe_first():
    rows = summary_rows()
    assert rows[0].commercial_use != NO
    assert rows[-1].commercial_use == NO
