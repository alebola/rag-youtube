from app.utils import yt_id_from_url, hhmmss
from app.ingest import get_transcript_auto, segment_transcript
from app.embeddings import embed_chunks


url = "https://www.youtube.com/watch?v=zxQyTK8quyY"
vid = yt_id_from_url(url)

# 1. Cargamos subtítulos desde caché (no golpea YouTube otra vez)
rows = get_transcript_auto(vid, fallback_url=url)
chunks = segment_transcript(rows, window=60, overlap=12)

# 2. Generamos embeddings
chunks_with_embs = embed_chunks(chunks)

print("Ejemplo de chunk con embedding:")
print("Texto:", chunks_with_embs[0]["text"][:80], "...")
print("Vector dimensión:", len(chunks_with_embs[0]["embedding"]))
