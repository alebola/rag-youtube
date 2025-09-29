from app.utils import yt_id_from_url, hhmmss
from app.ingest import get_transcript_auto, segment_transcript


url = "https://www.youtube.com/watch?v=zxQyTK8quyY"
vid = yt_id_from_url(url)

rows = get_transcript_auto(
    vid,
    preferred_langs=('es','en'),
    fallback_url=url,
    # descomenta UNA de estas dos si tienes 429:
    cookiefile="cookies.txt",
    #cookiesfrombrowser=('chrome',),  # o ('edge',)
)
print("líneas:", len(rows))
chunks = segment_transcript(rows, window=60, overlap=12)
print("chunks:", len(chunks), "rango 0:", hhmmss(chunks[0]['start_sec']), "→", hhmmss(chunks[0]['end_sec']))
