from sentence_transformers import SentenceTransformer
from app.utils import yt_id_from_url
from app.ingest import get_transcript_auto, segment_transcript
from app.embeddings import embed_chunks
from app.pinecone_store import ensure_index, upsert_chunks, query
from app.rag_answers import rag_answer_with_citations, dedup_hits_by_time


URL = "https://www.youtube.com/watch?v=zxQyTK8quyY"
VID = yt_id_from_url(URL)

QUESTIONS = [
    "¿Qué dice el video sobre por qué los transformadores usan autoatención?",
    "¿Qué explica el video sobre la diferencia entre atención y autoatención?",
    "¿Qué dice el video sobre consultas (queries), claves (keys) y valores (values)?",
    "¿Qué comenta el video sobre codificación posicional y para qué sirve?",
    "¿El video menciona algo sobre el entrenamiento de transformadores?",
    "¿Qué explica el video sobre multi-head attention y su ventaja respecto a una sola cabeza?",
    "¿Qué explica el video sobre el papel del softmax en la atención?",
    "¿Qué dice el video sobre el flujo de información desde embeddings hasta la salida del modelo?",
    "What does the video say about why transformers use self-attention?",
    "What does the video say about why the dot product is scaled in attention?",
    "What does the video explain about multi-head attention and its advantage over a single head?",
    "What does the video say about reusing the weight matrices when computing queries, keys, and values?",
    "What does the video explain about the similarity (attention score) matrix across words?",
    "What does the video explain about the role of softmax in attention?",
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
    hits = dedup_hits_by_time(hits, min_gap_sec=60.0)  
    print("Top score:", round(hits[0]["score"], 3))
    ans, cites = rag_answer_with_citations(VID, q, hits, ctx_max=4, cite_k=2)
    print("\nQ:", q)
    print("A:", ans)
    print("Citas:")
    for c in cites:
        print(f"- {c['minute']} -> {c['url']}")
