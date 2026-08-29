"""Splits a mixed track into instrument/vocal stems using Demucs.

Demucs (Meta, MIT licensed: https://github.com/facebookresearch/demucs) is
a source-separation model -- it doesn't generate anything, it pulls apart
audio that's already mixed. Bark only ever renders one mixed waveform, so
there's no way to get separate instrument tracks directly out of
generation; this is how a multitrack result gets approximated after the
fact, which is the standard technique for pulling stems out of a
recording that was never multitrack to begin with.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .licenses import assert_commercial_ok

DEFAULT_MODEL = "htdemucs"  # 4 stems: vocals, drums, bass, other


def separate_stems(audio_path: str | Path, work_dir: str | Path, model: str = DEFAULT_MODEL) -> dict[str, Path]:
    """Run Demucs on `audio_path`, using `work_dir` as scratch space.

    Returns {stem_name: path}, e.g. {"vocals": ..., "drums": ..., "bass": ..., "other": ...}.
    Caller owns `work_dir` and should clean it up once done with the paths.
    """
    assert_commercial_ok("demucs")

    try:
        import demucs.separate
    except ImportError as exc:
        raise RuntimeError(
            "The 'demucs' package must be installed to separate stems.\n"
            'Install with: pip install -e ".[stems]"'
        ) from exc

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"No such audio file: {audio_path}")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    demucs.separate.main(["-n", model, "-o", str(work_dir), str(audio_path)])

    track_dir = work_dir / model / audio_path.stem
    stems = {p.stem: p for p in track_dir.glob("*.wav")}
    if not stems:
        raise RuntimeError(f"Demucs produced no output in {track_dir}")
    return stems


def cleanup_work_dir(work_dir: str | Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
