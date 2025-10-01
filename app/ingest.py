from typing import List, Dict
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from .utils import clean_text
import tempfile, os, re
import json, time, random
from pathlib import Path


# Caché local de transcripciones
CACHE_DIR = Path("data/transcripts")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.json"

def _load_cached_transcript(video_id: str):
    p = _cache_path(video_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _save_cached_transcript(video_id: str, rows):
    try:
        _cache_path(video_id).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# Parser de VTT a filas {text, start, duration}
def _parse_vtt_to_rows(vtt_text: str):
    def ts_to_seconds(ts):
        parts = ts.replace(",", ".").split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        else:
            h, m, s = 0.0, parts[0], parts[1]
        return h*3600 + m*60 + s

    rows = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            start_ts, end_ts = [p.strip() for p in line.split("-->")]
            start = ts_to_seconds(start_ts)
            end = ts_to_seconds(end_ts)
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                text_lines.append(lines[i].strip())
                i += 1
            text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
            if text:
                rows.append({"text": text, "start": start, "duration": max(0.0, end - start)})
        i += 1
    return rows


# yt-dlp (fallback) SOLO MANUALES (no autosubs)
def _get_transcript_via_ytdlp(
    video_url: str,
    lang_priority=("es", "es-419", "en", "en-GB", "pt-BR", "pt"),
    cookiefile: str | None = None,
    cookiesfrombrowser: tuple | None = None,   
):
    from yt_dlp import YoutubeDL

    common = {
        "skip_download": True,
        "writesubtitles": True,          
        "writeautomaticsub": False,      
        "subtitlesformat": "vtt",
        "quiet": True,
        "no_warnings": True,
        "http_headers": {"User-Agent": "Mozilla/5.0"},
        "retries": 8,
        "retry_sleep": "exp",
        "sleep_interval_requests": 1.0,
        "max_sleep_interval_requests": 3.0,
    }
    if cookiefile and os.path.exists(cookiefile):
        common["cookiefile"] = cookiefile
    if cookiesfrombrowser:
        common["cookiesfrombrowser"] = cookiesfrombrowser

    # Intentar idiomas en orden
    for attempt_lang, lang in enumerate(lang_priority, start=1):
        with tempfile.TemporaryDirectory() as tmpd:
            opts = dict(common, outtmpl=os.path.join(tmpd, "%(id)s.%(ext)s"), subtitleslangs=[lang])
            try:
                with YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    ydl.download([video_url])
                vid_id = info["id"]
                vtts = [p for p in os.listdir(tmpd) if p.startswith(vid_id) and p.endswith(".vtt")]
                if not vtts:
                    time.sleep(0.8 + random.random() * 0.7)
                    continue
                with open(os.path.join(tmpd, vtts[0]), "r", encoding="utf-8") as f:
                    vtt_txt = f.read()
                rows = _parse_vtt_to_rows(vtt_txt)
                if rows:
                    return rows
            except Exception:
                time.sleep(min(3.0, 0.8 * (1.6 ** attempt_lang)) + random.random() * 0.9)
                continue

    # No intentamos 'all' porque reintroduciría autosubs
    raise RuntimeError("yt-dlp no encontró subtítulos MANUALES en los idiomas solicitados.")


# Preferencia de idioma y selección de pista
def _prefer_lang_code(lang_code: str, preferred=("es", "en")) -> int:
    lang_code = (lang_code or "").lower()
    for i, pref in enumerate(preferred):
        if lang_code == pref or lang_code.startswith(pref + "-") or pref.startswith(lang_code + "-"):
            return i
    return 10_000  

def _best_track(transcripts, preferred=("es", "en")):
    manuals = []
    for tr in transcripts:
        if getattr(tr, "is_generated", False):
            continue  # ignoramos autogenerados
        prio = _prefer_lang_code(tr.language_code, preferred)
        manuals.append((prio, tr))
    manuals.sort(key=lambda x: x[0])
    if manuals and manuals[0][0] < 10_000:
        return manuals[0][1]
    return None


# Comprobar si hay subtítulos manuales 
def has_manual_subs(video_id: str, cookies: str | None = None) -> bool:
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies)
        for tr in transcripts:
            if not getattr(tr, "is_generated", False):
                return True
        return False
    except Exception:
        return False


# Descarga subtítulos (manuales)
def get_transcript_auto(
    video_id: str,
    preferred_langs: tuple = ("es", "en"),
    cookies: str | None = None,             
    cookiefile: str | None = None,          
    cookiesfrombrowser: tuple | None = None,
    fallback_url: str | None = None,
    max_retries: int = 4,
    backoff_base: float = 1.5,
):
    # 0) Cache local
    cached = _load_cached_transcript(video_id)
    if cached:
        return cached

    # 1) API con reintentos (solo manuales)
    for attempt in range(max_retries):
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies)
            tr = _best_track(transcripts, preferred=preferred_langs)  # ignora autogenerados
            if tr is None:
                raise RuntimeError("Este vídeo no tiene subtítulos manuales disponibles en los idiomas preferidos.")
            rows = tr.fetch()
            _save_cached_transcript(video_id, rows)
            return rows
        except (TranscriptsDisabled, NoTranscriptFound):
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)
                continue
            break

    # 2) Fallback con yt-dlp SOLO manuales
    if fallback_url:
        rows = _get_transcript_via_ytdlp(
            fallback_url,
            lang_priority=("es", "es-419", "en", "en-GB", "pt-BR", "pt"),
            cookiefile=cookiefile,
            cookiesfrombrowser=cookiesfrombrowser,
        )
        _save_cached_transcript(video_id, rows)
        return rows

    raise RuntimeError("No se pudieron obtener subtítulos manuales (API + fallback).")


# Segmentación/chunking de los subtítulos
def segment_transcript(
    rows: List[Dict],
    window: int = 60,
    overlap: int = 12
) -> List[Dict]:
    if not rows:
        return []

    end_total = max(r["start"] + r["duration"] for r in rows)
    segments: List[Dict] = []
    start = 0.0

    while start < end_total:
        end = start + window

        # Reunimos todas las líneas que intersectan [start, end)
        texts = []
        for r in rows:
            line_start = r["start"]
            line_end = r["start"] + r["duration"]
            if line_start < end and line_end > start:
                texts.append(r["text"])

        chunk_text = clean_text(" ".join(texts))

        if chunk_text:
            segments.append({
                "start_sec": float(start),
                "end_sec": float(min(end, end_total)),
                "text": chunk_text
            })

        start += (window - overlap)

    return segments
