"""Command-line interface for the persona-based song generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clone import save_cloned_persona
from .personas import Persona, PersonaRegistry
from .presets import CURATED_PRESETS
from .song import SongStore


def _cmd_persona_list(args: argparse.Namespace) -> None:
    registry = PersonaRegistry()
    personas = registry.list()
    if not personas:
        print("No personas yet. Run 'persona seed-defaults' to add starter voices.")
        return
    for p in personas:
        print(f"{p.name:12} [{p.genre:10}] {p.description}  ({p.voice_source_type}:{p.voice_source_value})")


def _cmd_persona_seed_defaults(args: argparse.Namespace) -> None:
    registry = PersonaRegistry()
    existing = {p.name for p in registry.list()}
    added = 0
    for preset in CURATED_PRESETS:
        if preset["name"] in existing and not args.overwrite:
            continue
        registry.add(
            Persona(
                name=preset["name"],
                voice_source_type="preset",
                voice_source_value=preset["voice"],
                genre=preset["genre"],
                description=preset["description"],
            ),
            overwrite=args.overwrite,
        )
        added += 1
    print(f"Added {added} default persona(s).")


def _cmd_persona_create(args: argparse.Namespace) -> None:
    registry = PersonaRegistry()
    persona = Persona(
        name=args.name,
        voice_source_type="preset",
        voice_source_value=args.voice,
        genre=args.genre or "",
        description=args.description or "",
        text_temp=args.text_temp,
        waveform_temp=args.waveform_temp,
    )
    registry.add(persona, overwrite=args.overwrite)
    print(f"Saved persona '{persona.name}'.")


def _cmd_persona_save_clone(args: argparse.Namespace) -> None:
    registry = PersonaRegistry()
    persona = save_cloned_persona(
        registry,
        name=args.name,
        npz_path=args.npz,
        genre=args.genre or "",
        description=args.description or "",
        overwrite=args.overwrite,
    )
    print(f"Saved cloned persona '{persona.name}' -> {persona.voice_source_value}")


def _cmd_persona_remove(args: argparse.Namespace) -> None:
    registry = PersonaRegistry()
    registry.remove(args.name)
    print(f"Removed persona '{args.name}'.")


def _cmd_generate(args: argparse.Namespace) -> None:
    from .generate import generate_song  # lazy: only 'generate' needs bark/torch

    registry = PersonaRegistry()
    persona = registry.get(args.persona)

    lyrics = Path(args.lyrics_file).read_text() if args.lyrics_file else args.lyrics
    if not lyrics:
        print("Provide --lyrics or --lyrics-file", file=sys.stderr)
        sys.exit(1)

    out_path = generate_song(
        persona, lyrics, args.out, seed=args.seed, continuity_reset_every=args.reset_every
    )
    print(f"Wrote {out_path}")


def _cmd_song_create(args: argparse.Namespace) -> None:
    store = SongStore()
    lyrics = Path(args.lyrics_file).read_text() if args.lyrics_file else args.lyrics
    if not lyrics:
        print("Provide --lyrics or --lyrics-file", file=sys.stderr)
        sys.exit(1)
    song = store.create(args.title, args.persona, lyrics)
    print(f"Created song '{song.title}' ({song.id}) with {len(song.sections)} section(s):")
    for s in song.sections:
        print(f"  {s.id}  {s.label}")


def _cmd_song_list(args: argparse.Namespace) -> None:
    songs = SongStore().list()
    if not songs:
        print("No songs yet. Run 'song create' to start one.")
        return
    for song in songs:
        stale = sum(1 for s in song.sections if s.is_stale)
        print(f"{song.id}  {song.title:30}  persona={song.persona_name}  sections={len(song.sections)}  stale={stale}")


def _cmd_song_show(args: argparse.Namespace) -> None:
    song = SongStore().get(args.song_id)
    print(f"{song.title} ({song.id})  persona={song.persona_name}")
    for s in song.sections:
        flag = "STALE" if s.is_stale else "ok"
        print(f"  [{flag:5}] {s.id}  {s.label}\n          {s.lyrics}")


def _cmd_song_edit_section(args: argparse.Namespace) -> None:
    store = SongStore()
    song = store.get(args.song_id)
    lyrics = Path(args.lyrics_file).read_text() if args.lyrics_file else args.lyrics
    store.update_section_lyrics(song, args.section_id, lyrics)
    print(f"Updated section '{args.section_id}'. It's now stale -- run 'song regenerate-section' to render it.")


def _cmd_song_add_section(args: argparse.Namespace) -> None:
    store = SongStore()
    song = store.get(args.song_id)
    lyrics = Path(args.lyrics_file).read_text() if args.lyrics_file else args.lyrics
    store.add_section(song, args.label, lyrics, position=args.position)
    print(f"Added section '{args.label}' to '{song.title}'.")


def _cmd_song_remove_section(args: argparse.Namespace) -> None:
    store = SongStore()
    song = store.get(args.song_id)
    store.remove_section(song, args.section_id)
    print(f"Removed section '{args.section_id}' from '{song.title}'.")


def _cmd_song_reorder(args: argparse.Namespace) -> None:
    store = SongStore()
    song = store.get(args.song_id)
    store.reorder_sections(song, args.section_ids)
    print(f"Reordered sections for '{song.title}'.")


def _cmd_song_regenerate_section(args: argparse.Namespace) -> None:
    from .song_render import regenerate_section  # lazy: needs bark/torch

    store = SongStore()
    song = store.get(args.song_id)
    regenerate_section(store, song, args.section_id, seed=args.seed)
    print(f"Regenerated section '{args.section_id}'.")


def _cmd_song_render(args: argparse.Namespace) -> None:
    from .song_render import render_song  # lazy: only needs scipy/numpy, but keep lazy for consistency

    song = SongStore().get(args.song_id)
    out_path = render_song(song, args.out)
    print(f"Wrote {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="musicgen-personas", description="Reusable-voice song generator")
    sub = parser.add_subparsers(dest="command", required=True)

    persona_parser = sub.add_parser("persona", help="Manage reusable voice personas")
    persona_sub = persona_parser.add_subparsers(dest="persona_command", required=True)

    p_list = persona_sub.add_parser("list", help="List saved personas")
    p_list.set_defaults(func=_cmd_persona_list)

    p_seed = persona_sub.add_parser("seed-defaults", help="Add curated starter personas")
    p_seed.add_argument("--overwrite", action="store_true")
    p_seed.set_defaults(func=_cmd_persona_seed_defaults)

    p_create = persona_sub.add_parser("create", help="Create a persona from a built-in Bark voice preset")
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--voice", required=True, help="e.g. v2/en_speaker_6")
    p_create.add_argument("--genre", default="")
    p_create.add_argument("--description", default="")
    p_create.add_argument("--text-temp", type=float, default=0.7, dest="text_temp")
    p_create.add_argument("--waveform-temp", type=float, default=0.7, dest="waveform_temp")
    p_create.add_argument("--overwrite", action="store_true")
    p_create.set_defaults(func=_cmd_persona_create)

    p_clone = persona_sub.add_parser("save-clone", help="Register a custom cloned voice (.npz) as a persona")
    p_clone.add_argument("--name", required=True)
    p_clone.add_argument("--npz", required=True)
    p_clone.add_argument("--genre", default="")
    p_clone.add_argument("--description", default="")
    p_clone.add_argument("--overwrite", action="store_true")
    p_clone.set_defaults(func=_cmd_persona_save_clone)

    p_remove = persona_sub.add_parser("remove", help="Delete a saved persona")
    p_remove.add_argument("name")
    p_remove.set_defaults(func=_cmd_persona_remove)

    g = sub.add_parser("generate", help="Generate a song using a saved persona")
    g.add_argument("--persona", required=True)
    g.add_argument("--lyrics", default="")
    g.add_argument("--lyrics-file", default="")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument(
        "--reset-every",
        type=int,
        default=4,
        dest="reset_every",
        help="Snap back to the persona's base voice every N lyric chunks (0 disables resets)",
    )
    g.set_defaults(func=_cmd_generate)

    song_parser = sub.add_parser("song", help="Build a song as independently-editable sections")
    song_sub = song_parser.add_subparsers(dest="song_command", required=True)

    s_create = song_sub.add_parser("create", help="Create a song, splitting lyrics into sections on blank lines")
    s_create.add_argument("--title", required=True)
    s_create.add_argument("--persona", required=True)
    s_create.add_argument("--lyrics", default="")
    s_create.add_argument("--lyrics-file", default="")
    s_create.set_defaults(func=_cmd_song_create)

    s_list = song_sub.add_parser("list", help="List songs")
    s_list.set_defaults(func=_cmd_song_list)

    s_show = song_sub.add_parser("show", help="Show a song's sections")
    s_show.add_argument("song_id")
    s_show.set_defaults(func=_cmd_song_show)

    s_edit = song_sub.add_parser("edit-section", help="Change a section's lyrics (marks it stale)")
    s_edit.add_argument("song_id")
    s_edit.add_argument("section_id")
    s_edit.add_argument("--lyrics", default="")
    s_edit.add_argument("--lyrics-file", default="")
    s_edit.set_defaults(func=_cmd_song_edit_section)

    s_add = song_sub.add_parser("add-section", help="Add a new section")
    s_add.add_argument("song_id")
    s_add.add_argument("--label", required=True)
    s_add.add_argument("--lyrics", default="")
    s_add.add_argument("--lyrics-file", default="")
    s_add.add_argument("--position", type=int, default=None, help="0-based insert index; default appends")
    s_add.set_defaults(func=_cmd_song_add_section)

    s_remove = song_sub.add_parser("remove-section", help="Delete a section")
    s_remove.add_argument("song_id")
    s_remove.add_argument("section_id")
    s_remove.set_defaults(func=_cmd_song_remove_section)

    s_reorder = song_sub.add_parser("reorder", help="Reorder sections")
    s_reorder.add_argument("song_id")
    s_reorder.add_argument("section_ids", nargs="+", help="All section ids in the new order")
    s_reorder.set_defaults(func=_cmd_song_reorder)

    s_regen = song_sub.add_parser("regenerate-section", help="Render (or re-render) exactly one section")
    s_regen.add_argument("song_id")
    s_regen.add_argument("section_id")
    s_regen.add_argument("--seed", type=int, default=None)
    s_regen.set_defaults(func=_cmd_song_regenerate_section)

    s_render = song_sub.add_parser("render", help="Stitch all sections into one final track")
    s_render.add_argument("song_id")
    s_render.add_argument("--out", required=True)
    s_render.set_defaults(func=_cmd_song_render)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
