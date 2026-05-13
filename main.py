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
import subprocess
import re

# YouTube transcript
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

app = FastAPI(title="TL;DR Video Processor", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

openai.api_key = os.getenv("OPENAI_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")

def notify_n8n(job_id: str, status: str, data: dict = None):
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

def extract_youtube_id(url: str) -> str:
    """Estrae l'ID video da vari formati YouTube"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id: str, language: str = None) -> tuple:
    """Estrae transcript da YouTube. Ritorna (testo, lingua_trovata)"""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        if language and language != "auto":
            try:
                transcript = transcript_list.find_transcript([language])
            except:
                transcript = transcript_list.find_transcript(["it", "en", "es", "fr", "de"])
        else:
            # Auto-detect: prendi il primo disponibile
            transcript = transcript_list.find_transcript(["it", "en", "es", "fr", "de"])

        fetched = transcript.fetch()
        formatter = TextFormatter()
        text = formatter.format_transcript(fetched)
        detected_lang = transcript.language_code
        return text, detected_lang

    except Exception as e:
        raise Exception(f"Impossibile estrarre il transcript: {str(e)}")

def extract_audio(video_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "libmp3lame",
        "-q:a", "2", "-y", str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def transcribe_audio(audio_path: Path, language: str = None) -> tuple:
    with open(audio_path, "rb") as audio:
        kwargs = {"model": "whisper-1", "file": audio, "response_format": "text"}
        if language and language != "auto":
            kwargs["language"] = language
        transcript = openai.audio.transcriptions.create(**kwargs)
    # Whisper non ritorna lingua, assumiamo quella richiesta o "auto"
    return transcript, language or "auto"

def generate_content(transcript: str, language: str = "it") -> dict:
    lang_instruction = f"in {language}" if language != "auto" else "nella lingua del transcript"

    summary_prompt = f"""
Sei un esperto di analisi video e knowledge management. 
Analizza la seguente trascrizione e crea un riassunto strutturato {lang_instruction}.

TRASCRIZIONE:
{transcript[:15000]}

OUTPUT RICHIESTO:
1. RIASSUNTO ESECUTIVO (3-5 punti chiave)
2. CONCETTI FONDAMENTALI (elenco dettagliato)
3. AZIONI CONSIGLIATE (step pratici)
4. TAKEAWAY (insight principali)

Formatta in Markdown.
"""
    summary_response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
        max_tokens=2000
    )
    summary = summary_response.choices[0].message.content

    sop_prompt = f"""
Crea uno STANDARD OPERATING PROCEDURE (SOP) da questa trascrizione {lang_instruction}.

TRASCRIZIONE:
{transcript[:15000]}

FORMATO:
- TITOLO
- SCOPO  
- MATERIALE NECESSARIO
- PASSAGGI numerati con sotto-step
- TEMPO STIMATO
- CONTROLLI QUALITÀ
- TROUBLESHOOTING
"""
    sop_response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": sop_prompt}],
        temperature=0.2,
        max_tokens=2500
    )
    sop = sop_response.choices[0].message.content

    infographic_prompt = f"""
Crea un'INFOGRAFICA TESTUALE (ASCII/Markdown) da questa trascrizione {lang_instruction}.

TRASCRIZIONE:
{transcript[:15000]}

FORMATO:
┌─────────────────────────────────────┐
│         [TITOLO]                    │
├─────────────────────────────────────┤
│  📊 STATISTICHE CHIAVE              │
│  • Concetti: X                      │
│  • Parole: Y                        │
├─────────────────────────────────────┤
│  🎯 OBIETTIVO                       │
├─────────────────────────────────────┤
│  🔑 PILASTRI (1-5)                  │
├─────────────────────────────────────┤
│  ⚡ ACTION STEPS                    │
├─────────────────────────────────────┤
│  💡 QUOTE CHIAVE                     │
└─────────────────────────────────────┘
"""
    infographic_response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": infographic_prompt}],
        temperature=0.4,
        max_tokens=2500
    )
    infographic = infographic_response.choices[0].message.content

    return {
        "summary": summary,
        "sop": sop,
        "infographic": infographic,
        "raw_transcript": transcript,
        "word_count": len(transcript.split())
    }

@app.get("/api/status/test")
async def health_check():
    return {"status": "ok", "version": "1.1.0", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/process")
async def process_video(
    file: UploadFile = File(...),
    title: str = Form(""),
    language: str = Form("auto")
):
    job_id = str(uuid.uuid4())

    allowed = {".mp4", ".webm", ".mov", ".mp3", ".wav", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Formato non supportato. Usa: {', '.join(allowed)}")

    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        audio_path = UPLOAD_DIR / f"{job_id}.mp3"
        extract_audio(input_path, audio_path)
        transcript, detected_lang = transcribe_audio(audio_path, language)
        results = generate_content(transcript, detected_lang or language)

        result_data = {
            "job_id": job_id,
            "status": "completed",
            "title": title or file.filename,
            "language": detected_lang or language,
            **results
        }

        with open(RESULTS_DIR / f"{job_id}.json", "w") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        notify_n8n(job_id, "completed", {"title": title, "word_count": results["word_count"]})
        return result_data

    except Exception as e:
        error_data = {"job_id": job_id, "status": "error", "error": str(e)}
        with open(RESULTS_DIR / f"{job_id}.json", "w") as f:
            json.dump(error_data, f)
        raise HTTPException(500, str(e))

@app.post("/api/process-youtube")
async def process_youtube(url: str = Form(...), language: str = Form("auto")):
    job_id = str(uuid.uuid4())

    video_id = extract_youtube_id(url)
    if not video_id:
        raise HTTPException(400, "URL YouTube non valido. Formati supportati: youtube.com/watch?v=...,youtu.be/...")

    try:
        transcript, detected_lang = get_youtube_transcript(video_id, language)
        results = generate_content(transcript, detected_lang or language)

        result_data = {
            "job_id": job_id,
            "status": "completed",
            "source": "youtube",
            "video_id": video_id,
            "url": url,
            "language": detected_lang or language,
            **results
        }

        with open(RESULTS_DIR / f"{job_id}.json", "w") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        notify_n8n(job_id, "completed", {
            "title": f"YouTube: {video_id}", 
            "word_count": results["word_count"],
            "url": url
        })

        return result_data

    except Exception as e:
        error_data = {"job_id": job_id, "status": "error", "error": str(e)}
        with open(RESULTS_DIR / f"{job_id}.json", "w") as f:
            json.dump(error_data, f)
        raise HTTPException(500, str(e))

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    result_file = RESULTS_DIR / f"{job_id}.json"
    if result_file.exists():
        with open(result_file) as f:
            return json.load(f)
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    result_file = RESULTS_DIR / f"{job_id}.json"
    if not result_file.exists():
        raise HTTPException(404, "Risultato non trovato")
    with open(result_file) as f:
        return json.load(f)
