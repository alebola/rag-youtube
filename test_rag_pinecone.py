from sentence_transformers import SentenceTransformer
from app.utils import yt_id_from_url
from app.ingest import get_transcript_auto, segment_transcript
from app.embeddings import embed_chunks
from app.pinecone_store import ensure_index, upsert_chunks, query
from app.rag_answer import rag_answer_with_citations, dedup_hits_by_time


URL = "https://www.youtube.com/watch?v=zxQyTK8quyY"
VID = yt_id_from_url(URL)

QUESTIONS = [
    # --- Inglés, sobre el vídeo ---
    "Explain briefly what self-attention is and why it's important.",
    "What are queries, keys and values in a transformer?",
    "What is positional encoding used for?",
    "How is multi-head attention different from single-head?",
    "Why do transformers scale dot products in attention?",

    # --- Español, sobre el vídeo ---
    "Explica qué es la autoatención y por qué es importante.",
    "¿Para qué sirve el codificador posicional en los transformadores?",
    "¿Cuál es la diferencia entre multi-head attention y single-head?",
    "¿Qué representan las consultas, claves y valores en un transformador?",

    # --- Preguntas fuera de contexto ---
    "Who won the FIFA World Cup in 2022?",
    "¿Cuál es la capital de Francia?",
    "What is the recipe for paella?",
    "¿Qué nota sacó el autor del vídeo en la universidad?",
]


# 1) ingest + chunks 
rows = get_transcript_auto(VID, fallback_url=URL)
chunks = segment_transcript(rows, window=60, overlap=12)
chunks_with_embs = embed_chunks(chunks)

# 2) pinecone
ensure_index()
upsert_chunks(chunks_with_embs, video_id=VID, title="StatQuest: Transformers", lang="en")

# 3) preguntas
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

for q in QUESTIONS:
    q_vec = model.encode([q], convert_to_numpy=True)[0]
    hits = query(q_vec, top_k=8, video_id=VID)
    hits = dedup_hits_by_time(hits, min_gap_sec=60.0)  # reduce trozos pegados
    print("Top score:", round(hits[0]["score"], 3))
    ans, cites = rag_answer_with_citations(VID, q, hits, ctx_max=4, cite_k=3)
    print("\nQ:", q)
    print("A:", ans)
    print("Citas:")
    for c in cites:
        print(f"- {c['minute']} -> {c['url']}")
