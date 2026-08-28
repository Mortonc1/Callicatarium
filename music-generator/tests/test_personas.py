import pytest

from musicgen_personas.personas import Persona, PersonaRegistry


def make_persona(name="Aria"):
    return Persona(
        name=name,
        voice_source_type="preset",
        voice_source_value="v2/en_speaker_9",
        genre="pop",
        description="test voice",
    )


def test_add_and_get(tmp_path):
    registry = PersonaRegistry(path=tmp_path / "registry.json")
    registry.add(make_persona())
    assert registry.get("Aria").genre == "pop"


def test_persists_across_instances(tmp_path):
    path = tmp_path / "registry.json"
    PersonaRegistry(path=path).add(make_persona())
    reloaded = PersonaRegistry(path=path)
    assert [p.name for p in reloaded.list()] == ["Aria"]


def test_duplicate_without_overwrite_raises(tmp_path):
    registry = PersonaRegistry(path=tmp_path / "registry.json")
    registry.add(make_persona())
    with pytest.raises(ValueError):
        registry.add(make_persona())


def test_overwrite_replaces(tmp_path):
    registry = PersonaRegistry(path=tmp_path / "registry.json")
    registry.add(make_persona())
    updated = make_persona()
    updated.genre = "rock"
    registry.add(updated, overwrite=True)
    assert registry.get("Aria").genre == "rock"


def test_get_missing_raises_key_error(tmp_path):
    registry = PersonaRegistry(path=tmp_path / "registry.json")
    with pytest.raises(KeyError):
        registry.get("Nope")


def test_remove(tmp_path):
    registry = PersonaRegistry(path=tmp_path / "registry.json")
    registry.add(make_persona())
    registry.remove("Aria")
    assert registry.list() == []


def test_list_sorted(tmp_path):
    registry = PersonaRegistry(path=tmp_path / "registry.json")
    registry.add(make_persona("Zed"))
    registry.add(make_persona("Aria"))
    assert [p.name for p in registry.list()] == ["Aria", "Zed"]
