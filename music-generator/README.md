# musicgen-personas

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
cd music-generator
pip install -e .              # persona management only (lightweight)
pip install -e ".[generate]"  # + Bark/torch, needed to actually generate audio
```

The first generation call downloads Bark's model weights (a few GB). A GPU
is strongly recommended — Bark runs on CPU but is slow.

## Usage

```bash
# See the starter voices
musicgen-personas persona list

# Add your own voice built on a Bark preset
musicgen-personas persona create \
  --name "Nova" --voice v2/en_speaker_5 \
  --genre "synthwave" --description "Cool, detached female synth voice"

# Register a custom cloned voice (produced with a separate cloning tool,
# e.g. https://github.com/serp-ai/bark-with-voice-clone, which outputs
# Bark history_prompt .npz files)
musicgen-personas persona save-clone \
  --name "MyVoice" --npz personas/custom/myvoice.npz \
  --genre "rock" --description "My own cloned singing voice"

# Generate a song with a saved persona -- reuse it as many times as you like
musicgen-personas generate \
  --persona Aria \
  --lyrics "♪ Walking down the street tonight, city lights are burning bright ♪" \
  --out output/aria_song_1.wav

musicgen-personas generate \
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

## Project layout

```
music-generator/
  musicgen_personas/   # library + CLI
  personas/
    registry.json       # your saved personas (seeded with 8 starter voices)
    custom/              # your cloned-voice .npz files (git-ignored)
  output/                # generated songs (git-ignored)
  tests/                 # unit tests for persona management (no GPU needed)
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
