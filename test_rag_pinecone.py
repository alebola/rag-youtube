from sentence_transformers import SentenceTransformer
from app.utils import yt_id_from_url
from app.ingest import get_transcript_auto, segment_transcript
from app.embeddings import embed_chunks
from app.pinecone_store import ensure_index, upsert_chunks, query
from app.rag_answer import rag_answer_with_citations


url = "https://www.youtube.com/watch?v=zxQyTK8quyY"
vid = yt_id_from_url(url)

# 1) Prepara datos (usa caché para no tocar YouTube)
rows = get_transcript_auto(vid, fallback_url=url)
chunks = segment_transcript(rows, window=60, overlap=12)
chunks_with_embs = embed_chunks(chunks)

# 2) Asegura índice
ensure_index()

# 3) Sube vectores 
n = upsert_chunks(chunks_with_embs, video_id=vid, title="StatQuest: Transformers", lang="en")
print("Vectores upsertados:", n)

# 4) Embedding de la pregunta y query en Pinecone
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

question = "Explain briefly what self-attention is and why it's important."
q_vec = model.encode([question], convert_to_numpy=True)[0]
hits = query(q_vec, top_k=4, video_id=vid)

# 5) RAG: respuesta breve + citas con links
answer, citations = rag_answer_with_citations(vid, question, hits)
print("\nQ:", question)
print("A:", answer)
print("\nCitas:")
for c in citations:
    print("-", c["range"], "->", c["url"])
    print("  snippet:", c["snippet"])
