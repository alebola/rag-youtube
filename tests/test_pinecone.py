from app.embeddings import embed_chunks
from app.ingest import get_transcript_auto, segment_transcript
from app.utils import yt_id_from_url, hhmmss
from app import pinecone_store

# Config 
url = "https://www.youtube.com/watch?v=zxQyTK8quyY"
vid = yt_id_from_url(url)

# Ingest subtítulos
rows = get_transcript_auto(vid, preferred_langs=("es","en"), fallback_url=url)
chunks = segment_transcript(rows, window=60, overlap=12)

# Embeddings 
chunks_with_embs = embed_chunks(chunks)

# Pinecone 
pinecone_store.ensure_index()
inserted = pinecone_store.upsert_chunks(chunks_with_embs, video_id=vid, title="Test video")
print("Subidos a Pinecone:", inserted)

# Query 
q_vec = embed_chunks([{"text":"explicación de transformers", "start_sec":0, "end_sec":0}])[0]["embedding"]
hits = pinecone_store.query(q_vec, top_k=3, video_id=vid)

print("Resultados:")
for h in hits:
    print("-", hhmmss(h["start_sec"]), "→", hhmmss(h["end_sec"]), "|", h["text"][:100])
