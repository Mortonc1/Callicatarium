"""A phone-friendly web UI for generating songs with saved personas.

This still runs the real Bark model, so it needs to run on a machine with
real compute (ideally a GPU) and network access -- your phone is just the
client, talking to this server over the browser. Start it with:

    musicgen-personas-web

then open http://<that machine's LAN IP>:8000 from your phone (same wifi),
or point a tunnel (e.g. Tailscale, ngrok) at it for access from anywhere.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..personas import Persona, PersonaRegistry
from ..presets import CURATED_PRESETS
from ..song import SongStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"

app = FastAPI(title="musicgen-personas")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# A single-user personal tool: locks around file access, one worker so we
# never run two heavy Bark generations at once, an in-memory job table
# (fine to lose on restart -- generated .wav files and song.json stay on
# disk regardless).
_registry_lock = threading.Lock()
_songs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_models_lock = threading.Lock()
_models_ready = False


def _ensure_models_loaded() -> None:
    global _models_ready
    with _models_lock:
        if _models_ready:
            return
        from bark.generation import preload_models

        preload_models()
        _models_ready = True


class CreatePersonaRequest(BaseModel):
    name: str
    voice: str
    genre: str = ""
    description: str = ""


class GenerateRequest(BaseModel):
    persona: str
    lyrics: str
    seed: Optional[int] = None
    reset_every: int = 4


class CreateSongRequest(BaseModel):
    title: str
    persona: str
    lyrics: str


class UpdateSectionRequest(BaseModel):
    lyrics: str


class AddSectionRequest(BaseModel):
    label: str
    lyrics: str
    position: Optional[int] = None


class ReorderSectionsRequest(BaseModel):
    section_ids: list[str]


class RegenerateSectionRequest(BaseModel):
    seed: Optional[int] = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/api/personas")
def list_personas():
    with _registry_lock:
        return [p.to_dict() for p in PersonaRegistry().list()]


@app.get("/api/presets")
def list_presets():
    return CURATED_PRESETS


@app.post("/api/personas")
def create_persona(req: CreatePersonaRequest):
    with _registry_lock:
        registry = PersonaRegistry()
        persona = Persona(
            name=req.name,
            voice_source_type="preset",
            voice_source_value=req.voice,
            genre=req.genre,
            description=req.description,
        )
        try:
            registry.add(persona)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return persona.to_dict()


def _new_job(label: str) -> str:
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued", "label": label, "created_at": time.time()}
    return job_id


def _run_generation(
    job_id: str, persona: Persona, lyrics: str, seed: Optional[int], reset_every: int
) -> None:
    from ..generate import generate_song

    job = _jobs[job_id]
    job["status"] = "running"
    try:
        _ensure_models_loaded()
        out_path = OUTPUT_DIR / f"{job_id}.wav"
        generate_song(persona, lyrics, out_path, seed=seed, continuity_reset_every=reset_every)
        job["status"] = "done"
        job["out_path"] = str(out_path)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the client via /api/jobs/{id}
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    with _registry_lock:
        try:
            persona = PersonaRegistry().get(req.persona)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not req.lyrics.strip():
        raise HTTPException(status_code=400, detail="Lyrics can't be empty")

    job_id = _new_job(persona.name)
    _executor.submit(_run_generation, job_id, persona, req.lyrics, req.seed, req.reset_every)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    jobs = sorted(_jobs.items(), key=lambda kv: kv[1]["created_at"], reverse=True)
    return [
        {"job_id": jid, "status": j["status"], "label": j["label"], "created_at": j["created_at"]}
        for jid, j in jobs[:20]
    ]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {
        "job_id": job_id,
        "status": job["status"],
        "label": job["label"],
        "error": job.get("error"),
    }


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(
        job["out_path"], media_type="audio/wav", filename=f"{job['label']}_{job_id[:8]}.wav"
    )


# ---- Song projects: sections that can be edited/regenerated independently ----


def _song_summary(song) -> dict:
    return {
        "id": song.id,
        "title": song.title,
        "persona_name": song.persona_name,
        "updated_at": song.updated_at,
        "section_count": len(song.sections),
        "stale_count": sum(1 for s in song.sections if s.is_stale),
        "has_full_render": (song.dir / "full.wav").exists(),
    }


@app.get("/api/songs")
def list_songs():
    with _songs_lock:
        return [_song_summary(s) for s in SongStore().list()]


@app.post("/api/songs")
def create_song(req: CreateSongRequest):
    if not req.lyrics.strip():
        raise HTTPException(status_code=400, detail="Lyrics can't be empty")
    with _songs_lock:
        try:
            PersonaRegistry().get(req.persona)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        song = SongStore().create(req.title, req.persona, req.lyrics)
        return song.to_dict()


@app.get("/api/songs/{song_id}")
def get_song(song_id: str):
    with _songs_lock:
        try:
            return SongStore().get(song_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/songs/{song_id}")
def delete_song(song_id: str):
    with _songs_lock:
        try:
            SongStore().delete(song_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"deleted": song_id}


@app.put("/api/songs/{song_id}/sections/{section_id}")
def update_section(song_id: str, section_id: str, req: UpdateSectionRequest):
    with _songs_lock:
        store = SongStore()
        try:
            song = store.get(song_id)
            store.update_section_lyrics(song, section_id, req.lyrics)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return song.to_dict()


@app.post("/api/songs/{song_id}/sections")
def add_section(song_id: str, req: AddSectionRequest):
    with _songs_lock:
        store = SongStore()
        try:
            song = store.get(song_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store.add_section(song, req.label, req.lyrics, position=req.position)
        return song.to_dict()


@app.delete("/api/songs/{song_id}/sections/{section_id}")
def remove_section(song_id: str, section_id: str):
    with _songs_lock:
        store = SongStore()
        try:
            song = store.get(song_id)
            store.remove_section(song, section_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return song.to_dict()


@app.post("/api/songs/{song_id}/sections/reorder")
def reorder_sections(song_id: str, req: ReorderSectionsRequest):
    with _songs_lock:
        store = SongStore()
        try:
            song = store.get(song_id)
            store.reorder_sections(song, req.section_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return song.to_dict()


def _run_section_regeneration(job_id: str, song_id: str, section_id: str, seed: Optional[int]) -> None:
    from ..song_render import regenerate_section

    job = _jobs[job_id]
    job["status"] = "running"
    try:
        _ensure_models_loaded()
        with _songs_lock:
            store = SongStore()
            song = store.get(song_id)
        regenerate_section(store, song, section_id, seed=seed)
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001 -- surfaced to the client via /api/jobs/{id}
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/songs/{song_id}/sections/{section_id}/regenerate")
def regenerate_section_endpoint(song_id: str, section_id: str, req: RegenerateSectionRequest):
    with _songs_lock:
        try:
            song = SongStore().get(song_id)
            section = song.section(section_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    job_id = _new_job(f"{song.title} — {section.label}")
    _executor.submit(_run_section_regeneration, job_id, song_id, section_id, req.seed)
    return {"job_id": job_id}


@app.get("/api/songs/{song_id}/sections/{section_id}/audio")
def section_audio(song_id: str, section_id: str):
    with _songs_lock:
        try:
            song = SongStore().get(song_id)
            section = song.section(section_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not section.audio_file:
        raise HTTPException(status_code=404, detail="Section hasn't been rendered yet")
    return FileResponse(song.dir / section.audio_file, media_type="audio/wav")


def _run_song_render(job_id: str, song_id: str) -> None:
    from ..song_render import render_song

    job = _jobs[job_id]
    job["status"] = "running"
    try:
        with _songs_lock:
            song = SongStore().get(song_id)
        render_song(song, song.dir / "full.wav")
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001 -- surfaced to the client via /api/jobs/{id}
        job["status"] = "error"
        job["error"] = str(exc)


@app.post("/api/songs/{song_id}/render")
def render_song_endpoint(song_id: str):
    with _songs_lock:
        try:
            song = SongStore().get(song_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    job_id = _new_job(f"{song.title} (full render)")
    _executor.submit(_run_song_render, job_id, song_id)
    return {"job_id": job_id}


@app.get("/api/songs/{song_id}/audio")
def song_audio(song_id: str):
    with _songs_lock:
        try:
            song = SongStore().get(song_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    full_path = song.dir / "full.wav"
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Song hasn't been rendered yet")
    return FileResponse(full_path, media_type="audio/wav", filename=f"{song.title}.wav")


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
