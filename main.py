from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import openai
import os
import uuid
import json
from pathlib import Path
from datetime import datetime
import requests

app = FastAPI(title="TL;DR Video Processor", version="1.0.0")

# CORS per WordPress frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione: ["https://luigipesante.com", "https://tldr.luigipesante.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurazione
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")

def notify_n8n(job_id: str, status: str, data: dict = None):
    """Notifica n8n quando il job è completato"""
    if not N8N_WEBHOOK_URL:
        return
    try:
        payload = {
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {}
        }
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"n8n notification failed: {e}")

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    title: str = Form(""),
    language: str = Form("it"),
    webhook_url: str = Form("")
):
    """
    Processa un video:
    1. Salva il file
    2. Estrae audio (FFmpeg)
    3. Trascrive con Whisper
    4. Genera riassunto, SOP e infografica
    """
    job_id = str(uuid.uuid4())

    # Validazione
    allowed = {"video/mp4", "video/webm", "video/quicktime", "audio/mpeg", "audio/wav", "audio/mp4"}
    if file.content_type not in allowed and not file.filename.endswith(('.mp4', '.webm', '.mov', '.mp3', '.wav', '.m4a')):
        raise HTTPException(400, "Formato non supportato. Usa MP4, WEBM, MOV, MP3, WAV, M4A")

    # Salva file
    ext = Path(file.filename).suffix or ".mp4"
    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Per ora restituiamo un job ID - in produzione qui parte il processing async
    # Simuliamo il risultato finale per dimostrazione

    result = {
        "job_id": job_id,
        "status": "processing",
        "title": title or file.filename,
        "message": "Video ricevuto. L'elaborazione è in corso."
    }

    return JSONResponse(content=result)

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Controlla lo stato di un job"""
    result_file = RESULTS_DIR / f"{job_id}.json"
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Recupera il risultato completo"""
    result_file = RESULTS_DIR / f"{job_id}.json"
    if not result_file.exists():
        raise HTTPException(404, "Risultato non trovato o elaborazione in corso")
    with open(result_file) as f:
        return json.load(f)

@app.post("/api/process-youtube")
async def process_youtube(url: str = Form(...), language: str = Form("it")):
    """Processa un video YouTube (richiede youtube-transcript-api o yt-dlp)"""
    job_id = str(uuid.uuid4())

    # Placeholder - in produzione: estrai transcript con youtube-transcript-api
    return {
        "job_id": job_id,
        "status": "processing",
        "source": "youtube",
        "url": url,
        "message": "Elaborazione YouTube avviata"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
