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

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"

app = FastAPI(title="musicgen-personas")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# A single-user personal tool: one lock around registry file access, one
# worker so we never run two heavy Bark generations at once, an in-memory
# job table (fine to lose on restart -- generated .wav files stay on disk).
_registry_lock = threading.Lock()
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

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "queued",
        "persona": persona.name,
        "lyrics": req.lyrics,
        "created_at": time.time(),
    }
    _executor.submit(_run_generation, job_id, persona, req.lyrics, req.seed, req.reset_every)
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    jobs = sorted(_jobs.items(), key=lambda kv: kv[1]["created_at"], reverse=True)
    return [
        {"job_id": jid, "status": j["status"], "persona": j["persona"], "created_at": j["created_at"]}
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
        "persona": job["persona"],
        "error": job.get("error"),
    }


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str):
    job = _jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(status_code=404, detail="Not ready")
    return FileResponse(
        job["out_path"], media_type="audio/wav", filename=f"{job['persona']}_{job_id[:8]}.wav"
    )


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
