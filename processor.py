"""
TL;DR Video Processor Core
Gestisce: estrazione audio, trascrizione, summarization, SOP, infografica
"""
import os
import json
import openai
from pathlib import Path
from typing import Dict, List
import subprocess
import tempfile

class VideoProcessor:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "gpt-4-1106-preview"  # o gpt-4-turbo

    def extract_audio(self, video_path: Path, output_dir: Path) -> Path:
        """Estrae audio dal video usando FFmpeg"""
        audio_path = output_dir / f"{video_path.stem}.mp3"
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "libmp3lame",
            "-q:a", "2", "-y", str(audio_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return audio_path

    def transcribe(self, audio_path: Path, language: str = "it") -> str:
        """Trascrive audio con Whisper API"""
        with open(audio_path, "rb") as audio:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language=language,
                response_format="text"
            )
        return transcript

    def generate_summary(self, transcript: str, language: str = "it") -> Dict:
        """Genera riassunto strutturato, SOP e infografica"""

        # 1. Riassunto puntato
        summary_prompt = f"""
Sei un esperto di analisi video e knowledge management. 
Analizza la seguente trascrizione e crea un riassunto strutturato.

TRASCRIZIONE:
{transcript[:15000]}  # Limite per token

OUTPUT RICHIESTO (in {language}):
1. RIASSUNTO ESECUTIVO (3-5 punti chiave)
2. CONCETTI FONDAMENTALI (elenco dettagliato)
3. AZIONI CONSIGLIATE (step pratici)
4. TAKEAWAY (insight principali)

Formatta in Markdown.
"""

        summary_response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Sei un analista senior che crea riassunti strutturati e actionable."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        summary = summary_response.choices[0].message.content

        # 2. SOP (Standard Operating Procedure)
        sop_prompt = f"""
Basandoti sulla trascrizione, crea uno STANDARD OPERATING PROCEDURE (SOP).
Deve essere un protocollo passo-passo che chiunque possa seguire.

TRASCRIZIONE:
{transcript[:15000]}

FORMATO SOP:
- TITOLO: [Nome procedura]
- SCOPO: [Perché esiste questa procedura]
- MATERIALE NECESSARIO: [Lista]
- PASSaggi (1, 2, 3... con sotto-step a, b, c...)
- TEMPO STIMATO: [Durata]
- CONTROLLI QUALITÀ: [Come verificare che sia fatto bene]
- TROUBLESHOOTING: [Problemi comuni e soluzioni]

Scrivi in italiano professionale.
"""

        sop_response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Sei un operations manager che crea SOP aziendali dettagliati."},
                {"role": "user", "content": sop_prompt}
            ],
            temperature=0.2,
            max_tokens=2500
        )
        sop = sop_response.choices[0].message.content

        # 3. Infografica testuale
        infographic_prompt = f"""
Crea un'INFOGRAFICA TESTUALE basata sulla trascrizione.
Deve essere visualizzabile in testo puro (ASCII/Markdown).

TRASCRIZIONE:
{transcript[:15000]}

FORMATO:
┌─────────────────────────────────────┐
│         [TITOLO INFOGRAFICA]        │
├─────────────────────────────────────┤
│  📊 STATISTICHE CHIAVE              │
│  • Numero concetti: X               │
│  • Tempo video: Y                   │
│  • Difficoltà: Z                    │
├─────────────────────────────────────┤
│  🎯 OBIETTIVO                       │
│  [Testo breve]                      │
├─────────────────────────────────────┤
│  🔑 5 PILASTRI                      │
│  1. [Icona] [Concetto]              │
│     └─> [Dettaglio]                 │
│  2. ...                             │
├─────────────────────────────────────┤
│  ⚡ ACTION STEPS                     │
│  □ Step 1                           │
│  □ Step 2                           │
├─────────────────────────────────────┤
│  💡 QUOTE CHIAVE                     │
│  "..."                              │
└─────────────────────────────────────┘

Aggiungi anche un FLOWCHART TESTUALE del processo.
"""

        infographic_response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Sei un designer di infografiche che lavora solo con testo e ASCII art."},
                {"role": "user", "content": infographic_prompt}
            ],
            temperature=0.4,
            max_tokens=2500
        )
        infographic = infographic_response.choices[0].message.content

        return {
            "summary": summary,
            "sop": sop,
            "infographic": infographic,
            "raw_transcript": transcript,
            "word_count": len(transcript.split()),
            "processing_time": "calculated"
        }

    def process_video(self, video_path: Path, output_dir: Path, language: str = "it") -> Dict:
        """Pipeline completa"""
        # Estrai audio
        audio = self.extract_audio(video_path, output_dir)

        # Trascrivi
        transcript = self.transcribe(audio, language)

        # Genera contenuti
        results = self.generate_summary(transcript, language)

        # Salva risultato
        result_path = output_dir / "result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results
