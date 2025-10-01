from app.utils import yt_id_from_url, hhmmss
from app.ingest import get_transcript_auto, segment_transcript
from app.embeddings import embed_chunks

url = "https://www.youtube.com/watch?v=7JQLiQJzirw"   
vid = yt_id_from_url(url)

rows = get_transcript_auto(
    vid,
    preferred_langs=("es","es-419","en"),
    fallback_url=url,
    cookiefile="cookies.txt",    
    max_retries=7,
    backoff_base=2.0
)

chunks = segment_transcript(rows, window=60, overlap=12)
chunks_with_embs = embed_chunks(chunks)

print("Ejemplo de chunk con embedding:")
print("Texto:", chunks_with_embs[0]["text"][:80], "...")
print("Vector dimensión:", len(chunks_with_embs[0]["embedding"]))
