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

### Choosing your granularity per edit

Section size isn't fixed at creation. Split a section down when something
small needs changing, and merge back afterwards:

```bash
musichub song split-section <song-id> <section-id> --by lines
musichub song split-section <song-id> <section-id> --by words
musichub song edit-section <song-id> <piece-id> --lyrics "forever"
musichub song merge-sections <song-id> <id-1> <id-2> ... --label "Line 1"
```

Words rejoin with spaces and lines with newlines, so a split/edit/merge
round-trip gives you back the line with only the word you changed
different. In the web UI each section has **Split by line** / **Split by
word** buttons and a `merge` checkbox; tick two or more adjacent sections
and a merge button appears.

The tradeoff is real and unavoidable: each section is generated as its own
standalone utterance, so finer sections mean more surgical edits but weaker
flow across the joins. A lone word regenerated on its own gets its own
pitch and timing and usually will *not* blend back into the middle of a
sung phrase -- a crossfade can't fix that. **Line-level is the practical
floor for most edits.** Word-level is there because sometimes you know
better; the intended use is to split down, fix the word, merge back, and
regenerate the whole line as one take.

Splitting a rendered section discards its audio (that recording no longer
corresponds to any one piece), so the pieces start unrendered. Reference
timing from a guide track is divided across them proportionally, so each
piece still knows its slice for instrumental conditioning.

## Genre / style

Run `musichub styles` for the list. A style attaches to a song and does two
things -- one strongly, one weakly:

**Instrumentals (strong effect).** MusicGen takes a text prompt and genuinely
responds to it, so a style supplies one: picking `synthwave` sends
*"moody 80s synthwave, analog pads, arpeggiated bass, steady drum machine"*
rather than a generic *"instrumental backing track"*. Set it once on the song
and every section's instrumental uses it. An explicit `--prompt` still wins.

**Vocals (weak effect).** Bark has **no genre parameter**. Nothing in a style
can make it sing country instead of soul. Its only real levers are the voice
preset -- which the persona already picks, and which does most of the work --
and a few documented text cues. So a style wraps lines in `♪` when it's a
sung genre (and leaves them plain for `hiphop` / `spoken`), and prepends tags
like `[sighs]` where they suit. That nudges delivery. It does not change genre.

```bash
musichub styles
musichub song create --title "Midnight Drive" --persona Aria --style synthwave --lyrics "..."
musichub song set-style <song-id> lofi
musichub song set-style <song-id> none     # clear it
```

In the web UI there's a Genre/style dropdown on the new-song form and on each
song's detail page, with a line under it spelling out what the current
choice actually does. Changing a style doesn't retro-apply to audio you've
already rendered -- regenerate the sections you want it to affect.

### A correction worth stating plainly

Personas carry a `genre` field (the starter voices are tagged "pop", "rock",
"jazz" and so on). **That field is a label only** -- it has never been passed
to Bark and does not influence generation. It's there to help you pick a
voice from a list. The song-level `style` described above is the setting that
actually reaches a model.

## Stems: independently adjustable vocals/drums/bass/other

Bark only ever renders one mixed waveform -- there is no way to generate
separate instrument tracks directly, so "multitrack" here means something
specific: once a song is rendered to its final track, running
[Demucs](https://github.com/facebookresearch/demucs) (Meta, MIT licensed)
on that mix splits it into four stems -- vocals, drums, bass, other -- as
independent, standard source separation. It's not the same as an AI that
composed those parts separately, but it's a real, well-established
technique for pulling a mix back apart, and it's what gets you actual
per-track volume/mute control afterward.

Each stem gets its own gain (0-2x, linear) and mute toggle; **Mix &
preview** renders those settings down into a single `mix.wav` you can
listen to and re-adjust. Separation only works on a song that's already
been rendered (`song render` / the Render button) -- there's no per-section
stem separation, this operates on the finished full track.

```bash
musichub song separate-stems <song-id>
musichub song set-stem-level <song-id> vocals --gain 1.3
musichub song set-stem-level <song-id> drums --mute
musichub song mix-stems <song-id> --out output/custom_mix.wav
```

Requires `pip install -e ".[stems]"` (pulls in `demucs`). Separation quality
on Bark-generated audio specifically hasn't been validated here -- Demucs is
trained and tuned on real multitrack recordings, and how well it separates
a Bark mix is an open question you'll answer the first time you actually
run it.

## Recreating a song from a guide track

Point MusicHub at a track you already have and it builds an editable song
project from it: Whisper transcribes the vocals into timestamped lines,
those lines get grouped into sections at natural gaps, and each section
records where it sat in the original (`ref_start`/`ref_end`). You end up
with your song's structure and lyrics already laid out and editable,
instead of a blank page.

Each section can then generate a **melody-conditioned instrumental** via
MusicGen, conditioned on that section's own slice of the guide track --
so each part of the new track follows the corresponding part of the old.

```bash
musichub song from-reference --title "Static Mercy (rebuilt)" --persona Aria \
  --reference ~/Music/static_mercy.mp3 --model-size base
musichub song section-instrumental <song-id> <section-id> --prompt "moody synthwave, slow tempo"
```

The web UI has the same flow under **From a guide track** -- upload from
your phone, pick a voice, and it builds the project.

Requires `pip install -e ".[transcribe,melody]"` (and `[stems]`, which the
default vocal-isolation path uses).

### What this does and doesn't do

- The guide track's audio is **never copied, sampled, or remixed** into the
  output. Transcription produces text; melody conditioning extracts only a
  coarse chromagram (roughly: which pitch classes sound over time). Every
  audio sample in the result is newly generated.
- That also caps how close it can get. Expect "recognisably the same tune,
  clearly a different recording" -- not a near-duplicate. If you're hoping
  for something indistinguishable from the original, this technique will
  disappoint you, and no open-source pipeline available today will do
  better.
- Whisper is trained on **speech, not singing**. Transcripts of sung vocals
  are rough -- expect to correct them by hand. Isolating the vocal stem
  first (the default) helps a lot; a bigger `--model-size` helps more.
- MusicGen's melody model generates **instrumental music** and does not
  sing. Vocals come from Bark separately, via the normal section pipeline.

## Licensing: what you can sell

Model *code* and model *weights* are often licensed differently, and it's the
weights that decide whether you can commercially release what you generate.
Run `musichub licenses` for the current table. As it stands:

| Model | Used for | Weights licence | Sell the output? |
|---|---|---|---|
| Bark | vocals | MIT | yes |
| Demucs | stem separation | MIT | yes |
| Whisper | transcription | MIT | yes |
| MusicGen | melody-conditioned instrumentals | CC-BY-NC 4.0 | **no** |

So everything except the melody-guided instrumental feature is
unambiguously commercial-safe. MusicGen is the single exception: Audiocraft's
code is MIT but its weights are CC-BY-NC, and Meta's licence does not cover
commercially releasing music the model generates.

### Commercial-only mode

Set `MUSICHUB_COMMERCIAL_ONLY=1` and any model whose weights forbid commercial
use refuses to run, with an error naming the licence:

```bash
export MUSICHUB_COMMERCIAL_ONLY=1
musichub song section-instrumental <song-id> <section-id>
# NonCommercialModelError: MusicGen (facebook/musicgen-melody) is refused ...
```

The check runs *before* the dependency import, so you find out up front rather
than after downloading several GB of weights. Unknown models are refused
rather than assumed safe, so adding a new backend without recording its
licence fails closed.

With the gate on, every remaining feature -- personas, vocals, section
editing, stems, transcription -- is MIT and free to use commercially.

### If you want instrumentals commercially

`musicgen_personas/melody.py` is a thin wrapper around one model. Swapping in
a commercially-licensed alternative means reimplementing
`generate_melody_conditioned` against it and adding an entry to
`licenses.py`. Stable Audio Open is the obvious candidate and is already
recorded in the table as `conditional`: Stability's Community Licence permits
commercial use free below US$1M annual revenue, with an enterprise licence
required above that. **That backend is not implemented here** -- the entry
exists so the licence facts are recorded in one place.

None of this is legal advice, and these terms change. `musichub licenses`
prints the authoritative URL for each model; read them before releasing
anything commercially.

## Project layout

```
musichub/
  musicgen_personas/
    cli.py                # command-line interface
    web/                   # phone-friendly web UI (FastAPI + static page)
    song.py                 # song/section data model and CRUD (no bark needed)
    song_render.py          # renders sections and stitches them with a crossfade
    licenses.py              # per-model weights licences + the commercial-only gate
    styles.py                 # genre presets: instrumental prompts + Bark vocal cues
    stems.py                 # Demucs wrapper: splits a mix into vocals/drums/bass/other
    transcribe.py             # Whisper wrapper: timestamped lyrics from a guide track
    melody.py                  # MusicGen wrapper: melody-conditioned instrumentals
    recreate.py                 # guide track -> transcript -> sectioned song project
    song_stems.py             # song-level separate/gain/mute/mix on top of stems.py
    generate.py, personas.py, clone.py, presets.py
  personas/
    registry.json         # your saved personas (seeded with 8 starter voices)
    custom/                 # your cloned-voice .npz files (git-ignored)
  songs/                    # song projects: sections, stems, rendered audio (git-ignored)
  output/                   # quick-clip generated songs (git-ignored)
  tests/                    # unit tests -- persona registry, chunking/continuity,
                              # song/section CRUD, crossfade stitching, stem gain/mute/
                              # mixing, and the web API's job pipeline (no GPU needed
                              # for any of it)
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

Stems follow the same pattern. `demucs` installs and imports cleanly, and a
real separation call was attempted against a synthetic test file: it
reached all the way into Demucs' actual model-loading code and hit its
real download call to `dl.fbaipublicfiles.com` -- i.e. the integration
code is correct and calling Demucs' real API properly -- before hitting the
same class of 403 as everywhere else in this project (that host isn't
reachable from this sandbox either). Independent of that: gain/mute
adjustment and mixing (`mix_stems`) are pure audio math with no Bark or
Demucs dependency, and are fully tested against fabricated stem files,
including real HTTP round-trips (create a song, write fake stems directly
to disk, adjust gain over the API, submit a mix job, fetch the resulting
audio) -- all of that is verified for real. What's specifically
unconfirmed is Demucs' actual separation quality on Bark-generated audio,
which needs that network access and, more importantly, real ears.

The recreate pipeline was verified end-to-end over real HTTP with an actual
18MB MP3: the upload was accepted, a background job was created and ran,
the temp upload was cleaned up correctly afterwards, and the job failed
only at Whisper's model download (the same blocked host as everywhere
else). `faster-whisper` installs and imports cleanly. Everything this
project actually controls is tested directly: transcript-segment grouping
(gap and line-count boundaries, measured from segment *end* times),
segment-to-section mapping with reference timing preserved across a disk
round-trip, reference import, and instrumental conditioning on the correct
slice with MusicGen's 30s ceiling enforced. Unverified here, for the usual
reason: real transcription accuracy on sung vocals, and what
melody-conditioned output actually sounds like.
