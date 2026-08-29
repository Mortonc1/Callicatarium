"""Genre/style presets, and an honest account of what they can actually change.

Style control here is asymmetric, because the two models are:

* **Instrumentals (MusicGen)** take a free-text prompt and genuinely respond
  to it -- "moody synthwave, analog pads, slow tempo" produces meaningfully
  different music from "aggressive thrash metal". This is real conditioning.

* **Vocals (Bark)** have *no* genre parameter. Nothing you write here can
  tell Bark to sing country instead of soul. Its only real levers are the
  voice preset (which is what a Persona already picks) and a handful of
  documented text cues inside the prompt itself -- `♪` to sing rather than
  speak, bracketed tags like `[sighs]`, capitals for emphasis. A style can
  apply those consistently, which nudges delivery. It does not transform
  genre, and pretending otherwise would be a lie.

So: picking a style strongly shapes the backing track and only lightly
shapes the vocal performance. If you want a vocal to sit in a genre, the
persona's voice preset is doing most of that work.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    key: str
    name: str
    instrumental_prompt: str  # fed to MusicGen -- real, strong effect
    sung: bool  # wrap lines in ♪ so Bark sings rather than speaks
    vocal_cues: tuple[str, ...] = ()  # Bark tags prepended, e.g. "[sighs]"
    description: str = ""


STYLES: dict[str, Style] = {
    "synthwave": Style(
        key="synthwave",
        name="Synthwave",
        instrumental_prompt="moody 80s synthwave, analog pads, arpeggiated bass, steady drum machine",
        sung=True,
        description="Neon, nocturnal, mid-tempo",
    ),
    "lofi": Style(
        key="lofi",
        name="Lo-fi",
        instrumental_prompt="lo-fi hip hop, dusty piano, soft vinyl crackle, laid-back drums, warm bass",
        sung=True,
        description="Relaxed, hazy, unhurried",
    ),
    "rock": Style(
        key="rock",
        name="Rock",
        instrumental_prompt="driving rock band, distorted electric guitars, punchy drums, electric bass",
        sung=True,
        description="Loud guitars, forward momentum",
    ),
    "folk": Style(
        key="folk",
        name="Folk",
        instrumental_prompt="acoustic folk, fingerpicked guitar, gentle upright bass, brushed drums",
        sung=True,
        description="Acoustic, intimate, organic",
    ),
    "hiphop": Style(
        key="hiphop",
        name="Hip hop",
        instrumental_prompt="hip hop beat, heavy 808 bass, crisp hi-hats, sampled soul chords",
        sung=False,
        description="Rhythmic delivery over a beat -- spoken rather than sung",
    ),
    "orchestral": Style(
        key="orchestral",
        name="Orchestral",
        instrumental_prompt="cinematic orchestra, sweeping strings, brass swells, timpani, epic build",
        sung=True,
        description="Big, cinematic, dramatic",
    ),
    "jazz": Style(
        key="jazz",
        name="Jazz",
        instrumental_prompt="smoky jazz trio, brushed drums, walking upright bass, soft rhodes piano",
        sung=True,
        description="Loose, smoky, late-night",
    ),
    "ambient": Style(
        key="ambient",
        name="Ambient",
        instrumental_prompt="ambient soundscape, slow evolving pads, deep reverb, no percussion",
        sung=True,
        description="Formless, atmospheric, no beat",
    ),
    "pop": Style(
        key="pop",
        name="Pop",
        instrumental_prompt="bright modern pop, catchy synth hooks, tight drums, punchy bass",
        sung=True,
        description="Bright, hooky, radio-shaped",
    ),
    "ballad": Style(
        key="ballad",
        name="Ballad",
        instrumental_prompt="slow emotional ballad, grand piano, subtle strings, sparse arrangement",
        sung=True,
        vocal_cues=("[sighs]",),
        description="Slow, spacious, emotive",
    ),
    "metal": Style(
        key="metal",
        name="Metal",
        instrumental_prompt="heavy metal, downtuned palm-muted guitars, double kick drums, aggressive",
        sung=True,
        description="Heavy, fast, aggressive",
    ),
    "spoken": Style(
        key="spoken",
        name="Spoken word",
        instrumental_prompt="sparse minimal underscore, low drone, occasional piano notes",
        sung=False,
        description="Plain speech, no singing cue",
    ),
}

MUSIC_NOTE = "♪"


def get(style_key: str) -> Style:
    try:
        return STYLES[style_key]
    except KeyError:
        known = ", ".join(sorted(STYLES))
        raise KeyError(f"No style '{style_key}'. Known styles: {known}") from None


def apply_vocal_style(lyrics: str, style_key: str | None) -> str:
    """Decorate lyrics with the Bark text cues a style implies.

    This is the *only* lever Bark exposes short of changing voice, and its
    effect is modest: `♪` cues singing over speech, bracketed tags nudge
    delivery. Lines already wrapped in ♪ are left alone so hand-written
    cues aren't doubled up.
    """
    if style_key is None:
        return lyrics
    style = get(style_key)

    out_lines = []
    for line in lyrics.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if style.sung and not stripped.startswith(MUSIC_NOTE):
            stripped = f"{MUSIC_NOTE} {stripped} {MUSIC_NOTE}"
        out_lines.append(stripped)

    text = "\n".join(out_lines)
    if style.vocal_cues:
        text = f"{' '.join(style.vocal_cues)} {text}"
    return text


def instrumental_prompt_for(style_key: str | None, extra: str = "") -> str:
    """Build the MusicGen prompt for a style, with optional extra detail appended."""
    base = get(style_key).instrumental_prompt if style_key else "instrumental backing track"
    extra = extra.strip()
    return f"{base}, {extra}" if extra else base
