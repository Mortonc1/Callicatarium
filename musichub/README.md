# MusicHub

A self-hosted song generator with **reusable named voice personas** — the
same idea Suno's product is built around (pick a voice, generate as many
songs with it as you want), but built independently on top of
[Bark](https://github.com/suno-ai/bark), the generative audio model Suno's
own research team released as open source (MIT licensed). Everything here
runs on your own machine against weights you download yourself — there's no
dependency on Suno's paid API or any workaround of it.

## How personas work

A **persona** is a saved, named voice profile:

```json
{
  "name": "Aria",
  "voice_source_type": "preset",
  "voice_source_value": "v2/en_speaker_9",
  "genre": "pop",
  "description": "Bright, energetic female vocalist",
  "text_temp": 0.7,
  "waveform_temp": 0.7
}
```

`voice_source_type` is one of:

- `"preset"` — one of Bark's ~100 built-in speakers (`v2/en_speaker_0` .. `v2/en_speaker_9`,
  plus other languages). Eight are pre-seeded in `personas/registry.json` with
  friendlier names and genre tags (see `musicgen_personas/presets.py`).
- `"npz"` — a custom voice you cloned yourself, saved in Bark's own
  `history_prompt` format. Once registered, it's reusable by name exactly
  like a built-in preset, forever.

Personas persist in `personas/registry.json` (git-ignored per-clone files
live in `personas/custom/`), so you build up a personal voice library over
time instead of re-specifying a voice for every song.

## Setup

```bash
cd musichub
pip install -e .              # persona management only (lightweight)
pip install -e ".[generate]"  # + Bark/torch, needed to actually generate audio
```

The first generation call downloads Bark's model weights (a few GB). A GPU
is strongly recommended — Bark runs on CPU but is slow.

## Usage

```bash
# See the starter voices
musichub persona list

# Add your own voice built on a Bark preset
musichub persona create \
  --name "Nova" --voice v2/en_speaker_5 \
  --genre "synthwave" --description "Cool, detached female synth voice"

# Register a custom cloned voice (produced with a separate cloning tool,
# e.g. https://github.com/serp-ai/bark-with-voice-clone, which outputs
# Bark history_prompt .npz files)
musichub persona save-clone \
  --name "MyVoice" --npz personas/custom/myvoice.npz \
  --genre "rock" --description "My own cloned singing voice"

# Generate a song with a saved persona -- reuse it as many times as you like
musichub generate \
  --persona Aria \
  --lyrics "♪ Walking down the street tonight, city lights are burning bright ♪" \
  --out output/aria_song_1.wav

musichub generate \
  --persona Aria \
  --lyrics-file lyrics/verse2.txt \
  --out output/aria_song_2.wav \
  --seed 42
```

Wrap lines in `♪ ... ♪` to cue Bark to sing rather than speak — that's
Bark's own convention, not something this project adds.

Long lyrics are automatically chunked (Bark caps out around ~13 seconds per
generation) and stitched into one `.wav`. Each chunk after the first
continues from the previous chunk's generated state rather than restarting
from the bare voice preset, so a multi-line song flows as one take instead
of sounding cut-and-spliced. Every `--reset-every` chunks (default 4) it
snaps back to the persona's base voice, since chaining state indefinitely
lets Bark's voices drift off-character over a long generation.

## Phone / web UI

No terminal needed day-to-day: a small mobile-friendly web app wraps the
same persona system with a page you can drive from your phone's browser —
pick a voice, write lyrics, tap Generate, listen to (and re-play) your
recent songs.

The Bark model still has to actually run somewhere with real compute
(ideally a GPU) — a phone can't do that itself. So this is a small server
you start once on a capable machine (your desktop, a home server, a cloud
GPU box), and your phone just talks to it over the browser.

```bash
pip install -e ".[generate,web]"
musichub-web
```

Then from your phone, on the same wifi as that machine, open
`http://<that machine's LAN IP>:8000` (find the IP with `ipconfig`/`ifconfig`
on the host). For access from outside your home network, put something like
[Tailscale](https://tailscale.com) or an SSH tunnel in front of it rather
than exposing port 8000 directly to the internet.

The web UI can list personas, create new ones from a curated preset, submit
a generation job, and poll it to completion with an inline audio player —
everything the CLI does except registering a custom voice clone, which
still needs the `persona save-clone` command since that involves a file on
disk. Generation runs one job at a time in the background so the page stays
responsive while Bark works.

## Song projects: editing one section without wrecking the rest

Feeding a whole song through as one lyrics blob means Bark's own internal
chunking decides where the boundaries fall -- so changing a single word can
shift a chunk boundary and regenerate several seconds of audio you liked
and didn't touch. A **song project** fixes this by making the boundaries
yours: a song is an ordered list of independently-editable **sections**
(verse, chorus, bridge, ...), each rendered to its own audio file.

- Editing a section's lyrics only marks *that* section stale -- every other
  section's audio is untouched on disk until you explicitly ask to
  regenerate it.
- **Regenerate** re-renders exactly one section, independently, from the
  persona's base voice.
- **Render** stitches the current audio for every section into one final
  track, with a short crossfade at each join so splices don't click.
  It refuses to run while any section is stale, so you always know whether
  what you're about to render matches the lyrics on screen.

There's a real tradeoff here versus the single-call `generate` path: the
old path chains Bark's generated state across chunks for extra flow within
one generation, but that chaining is exactly what makes editing fragile
(regenerating an earlier chunk would ripple through everything after it).
Sections deliberately don't chain across each other for that reason --
each one is a clean, independent, re-editable unit, at a small cost to
inter-section flow.

CLI:

```bash
musichub song create --title "Midnight Drive" --persona Aria \
  --lyrics "$(printf 'verse line one\nverse line two\n\nchorus line one\nchorus line two')"
musichub song list
musichub song show <song-id>
musichub song edit-section <song-id> <section-id> --lyrics "verse line one, changed"
musichub song regenerate-section <song-id> <section-id>
musichub song add-section <song-id> --label Bridge --lyrics "bridge lyrics" --position 2
musichub song reorder <song-id> <section-id-1> <section-id-2> ...
musichub song remove-section <song-id> <section-id>
musichub song render <song-id> --out output/midnight_drive.wav
```

The web UI's **Song Projects** tab covers the same workflow: a section list
with inline lyrics editing, per-section Regenerate, up/down reordering, add
and remove, and a Render button for the stitched final track -- all from
your phone.

## Project layout

```
musichub/
  musicgen_personas/
    cli.py                # command-line interface
    web/                   # phone-friendly web UI (FastAPI + static page)
    song.py                 # song/section data model and CRUD (no bark needed)
    song_render.py          # renders sections and stitches them with a crossfade
    generate.py, personas.py, clone.py, presets.py
  personas/
    registry.json         # your saved personas (seeded with 8 starter voices)
    custom/                 # your cloned-voice .npz files (git-ignored)
  songs/                    # song projects: sections + rendered audio (git-ignored)
  output/                   # quick-clip generated songs (git-ignored)
  tests/                    # unit tests -- persona registry, chunking/continuity,
                              # song/section CRUD, stitching, and the web API's job
                              # pipeline (no GPU needed for any of it)
```

## Notes and limitations

- This targets Bark's speech/song generation, which handles vocals plus
  simple accompaniment in one model — it is not a full multitrack music
  production system.
- Voice *cloning* (turning a reference audio clip into a new `.npz`) isn't
  implemented in this project; it registers and reuses clones produced by
  an external tool that speaks Bark's `history_prompt` format.
- Nothing here calls or scrapes Suno's service. It's an independent,
  self-hosted pipeline you run against a model MIT-licensed for exactly
  this kind of use.

## Verification status

The persona registry (create/list/remove/reuse voices) is fully tested and
CLI-verified. The generation path (`generate_song`) was verified as far as
its sandbox allows: `bark`/`torch` install correctly, imports work, and a
real call reaches all the way into Bark's actual `preload_models()` →
`hf_hub_download()` — i.e. this project's integration code is correct and
calling Bark's real API properly. What couldn't be verified here is actual
audio output, because that sandbox's network policy blocks
`huggingface.co` (where Bark's model weights live), so the weight download
itself fails with a 403 in that environment. On a normal machine with open
network access, this should proceed straight through to producing a
`.wav`. If you hit anything past that point, it's a genuinely new issue
worth reporting.

The web app was verified for real: started as an actual server and hit
over real HTTP (not just in-process), confirming the page loads, the
persona/preset APIs return real data, and a submitted generation job
correctly moves through queued → running → a terminal state visible from
the page. The one thing it inherits, unverified for the same reason as
above, is real audio at the end of that pipeline.

Song projects are fully tested without needing Bark at all for the parts
that don't need it: section CRUD (create/edit/add/remove/reorder,
staleness tracking) and the crossfade-stitching logic in `render_song` are
covered directly, including a full render exercised end-to-end against
fabricated section audio (real `write_wav`/`read_wav` round-trips, not
mocks) to confirm the stitched output has the right length and plays back.
The song API was also hit over real HTTP the same way as above: created a
song, edited a section (confirmed it correctly flips to stale), and
confirmed the page serves the new Song Projects tab. The only unverified
step, same root cause as everywhere else in this project, is a section's
actual Bark-rendered audio.
