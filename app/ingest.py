from typing import List, Dict
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from .utils import clean_text
import tempfile, os, re
import json, time
from pathlib import Path


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


# Parsea subtítulos en formato VTT a la lista de filas estándar
def _parse_vtt_to_rows(vtt_text: str):
    def ts_to_seconds(ts):
        h, m, s = ts.split(':')
        return int(h)*3600 + int(m)*60 + float(s.replace(',', '.'))
    rows = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            start_ts, end_ts = [p.strip() for p in line.split('-->')]
            start = ts_to_seconds(start_ts)
            end = ts_to_seconds(end_ts)
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                text_lines.append(lines[i].strip())
                i += 1
            text = re.sub(r'\s+', ' ', ' '.join(text_lines)).strip()
            if text:
                rows.append({'text': text, 'start': start, 'duration': end - start})
        i += 1
    return rows


# Intenta obtener subtítulos usando yt-dlp (si falla YouTubeTranscriptApi)
def _get_transcript_via_ytdlp(
    video_url: str,
    lang_priority=('es','es-419','es-US','en','en-GB','pt-BR','pt'),
    cookiefile: str | None = None,
    cookiesfrombrowser: tuple | None = None,   # ej. ('chrome',) o ('edge',)
):
    from yt_dlp import YoutubeDL

    common = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'vtt',
        'quiet': True,
        'no_warnings': True,
        'http_headers': {'User-Agent': 'Mozilla/5.0'},
    }
    if cookiefile and os.path.exists(cookiefile):
        common['cookiefile'] = cookiefile
    if cookiesfrombrowser:
        common['cookiesfrombrowser'] = cookiesfrombrowser

    # 1) Probar cada idioma en orden
    for lang in lang_priority:
        with tempfile.TemporaryDirectory() as tmpd:
            opts = dict(common, outtmpl=os.path.join(tmpd, '%(id)s.%(ext)s'), subtitleslangs=[lang])
            try:
                with YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    ydl.download([video_url])
                vid_id = info['id']
                vtts = [p for p in os.listdir(tmpd) if p.startswith(vid_id) and p.endswith('.vtt')]
                if not vtts:
                    continue
                with open(os.path.join(tmpd, vtts[0]), 'r', encoding='utf-8') as f:
                    vtt_txt = f.read()
                rows = _parse_vtt_to_rows(vtt_txt)
                if rows:
                    return rows
            except Exception:
                continue

    # 2) Último recurso: bajar TODOS los idiomas y quedarnos con el primero válido
    with tempfile.TemporaryDirectory() as tmpd:
        opts = dict(common, outtmpl=os.path.join(tmpd, '%(id)s.%(ext)s'), subtitleslangs=['all'])
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                ydl.download([video_url])
            vid_id = info['id']
            vtts = [os.path.join(tmpd, p) for p in os.listdir(tmpd) if p.startswith(vid_id) and p.endswith('.vtt')]
            for vtt_path in vtts:
                try:
                    with open(vtt_path, 'r', encoding='utf-8') as f:
                        vtt_txt = f.read()
                    rows = _parse_vtt_to_rows(vtt_txt)
                    if rows:
                        return rows
                except Exception:
                    continue
        except Exception:
            pass

    raise RuntimeError("yt-dlp no pudo obtener subtítulos (ni por idiomas ni con 'all').")


# Devuelve una prioridad numérica según lista
def _prefer_lang_code(lang_code: str, preferred=('es','en')) -> int:
    lang_code = (lang_code or '').lower()
    for i, pref in enumerate(preferred):
        if lang_code == pref or lang_code.startswith(pref + '-') or pref.startswith(lang_code + '-'):
            return i
    return 10_000  # muy bajo


# Selecciona la mejor pista de subtítulos 
def _best_track(transcripts, preferred=('es','en')):
    manuals = []
    autos = []
    for tr in transcripts:
        prio = _prefer_lang_code(tr.language_code, preferred)
        item = (prio, tr)
        if getattr(tr, 'is_generated', False):
            autos.append(item)
        else:
            manuals.append(item)

    manuals.sort(key=lambda x: x[0])
    autos.sort(key=lambda x: x[0])

    if manuals and manuals[0][0] < 10_000:
        return manuals[0][1]
    if autos and autos[0][0] < 10_000:
        return autos[0][1]
    # Si no hay nada “preferido”, intenta cualquiera (manual > auto)
    if manuals:
        return manuals[0][1]
    if autos:
        return autos[0][1]
    return None


# Descarga subtítulos
def get_transcript_auto(
    video_id: str,
    preferred_langs: tuple = ('es','en'),
    cookies: str | None = None,             # cabecera Cookie (opcional)
    cookiefile: str | None = None,          # ruta a cookies.txt (Netscape)
    cookiesfrombrowser: tuple | None = None,# ej. ('chrome',) o ('edge',)
    fallback_url: str | None = None,
    max_retries: int = 4,
    backoff_base: float = 1.5
):
    # 0) Cache local
    cached = _load_cached_transcript(video_id)
    if cached:
        return cached

    # 1) API con reintentos
    for attempt in range(max_retries):
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies)
            tr = _best_track(transcripts, preferred=preferred_langs)
            if tr is None:
                raise RuntimeError("No hay pistas de subtítulos disponibles para este vídeo.")
            rows = tr.fetch()
            _save_cached_transcript(video_id, rows)
            return rows
        except (TranscriptsDisabled, NoTranscriptFound):
            # No hay transcripciones “oficiales” -> saltamos a fallback
            break
        except Exception as e:
            # 429 u otros: backoff y reintento
            if attempt < max_retries - 1:
                sleep_s = backoff_base ** attempt
                time.sleep(sleep_s)
                continue
            # agotados reintentos -> pasamos a fallback
            break

    # 2) Fallback con yt-dlp si hay URL
    if fallback_url:
        rows = _get_transcript_via_ytdlp(
            fallback_url,
            lang_priority=('es','es-419','es-US','en','en-GB','pt-BR','pt'),
            cookiefile=cookiefile,
            cookiesfrombrowser=cookiesfrombrowser
        )
        _save_cached_transcript(video_id, rows)
        return rows

    raise RuntimeError("No se pudieron obtener subtítulos (API + fallback).")


# Agrupa los subtítulos en chunks 
def segment_transcript(
    rows: List[Dict],
    window: int = 60,
    overlap: int = 12
) -> List[Dict]:
    if not rows:
        return []

    end_total = max(r['start'] + r['duration'] for r in rows)

    segments: List[Dict] = []
    start = 0.0

    while start < end_total:
        end = start + window

        # Reunimos todas las líneas de subtítulo que intersectan [start, end)
        texts = []
        for r in rows:
            line_start = r['start']
            line_end = r['start'] + r['duration']
            if line_start < end and line_end > start:
                texts.append(r['text'])

        chunk_text = clean_text(" ".join(texts))

        if chunk_text:
            segments.append({
                'start_sec': float(start),
                'end_sec': float(min(end, end_total)),
                'text': chunk_text
            })

        start += (window - overlap)

    return segments
