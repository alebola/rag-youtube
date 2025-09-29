import re


# Extrae el VIDEO_ID de un enlace de YouTube
def yt_id_from_url(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    if not m:
        raise ValueError("No se pudo extraer VIDEO_ID del enlace")
    return m.group(1)


# Convierte segundos a formato HH:MM:SS o MM:SS
def hhmmss(seconds: float) -> str:
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# Genera un enlace de YouTube que comienza en un segundo concreto
def time_url(video_id: str, start_sec: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}"


# Limpia el texto
def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t