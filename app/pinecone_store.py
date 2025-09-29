from typing import List, Dict, Optional, Tuple
import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from .utils import hhmmss, time_url

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-youtube-idx")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

DIM = 384        # MiniLM L12 v2
METRIC = "cosine"


# Verificamos que la configuración está bien
def _client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("Falta PINECONE_API_KEY en el entorno (.env).")
    return Pinecone(api_key=PINECONE_API_KEY)


# Crea el índice serverless si no existe
def ensure_index(index_name: str = PINECONE_INDEX, dim: int = DIM) -> None:
    pc = _client()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name in existing:
        return
    pc.create_index(
        name=index_name,
        dimension=dim,
        metric=METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


# Acceso al índice
def _index():
    pc = _client()
    return pc.Index(PINECONE_INDEX)


# Sube los chunks al índice
def upsert_chunks(
    chunks_with_embs: List[Dict],
    video_id: str,
    title: Optional[str] = None,
    lang: Optional[str] = None,
) -> int:
    vecs = []
    for i, c in enumerate(chunks_with_embs):
        vecs.append({
            "id": f"{video_id}:{i}",   # id único por chunk
            "values": c["embedding"] if isinstance(c["embedding"], list) else c["embedding"].tolist(),
            "metadata": {
                "video_id": video_id,
                "start_sec": float(c["start_sec"]),
                "end_sec": float(c["end_sec"]),
                "text": c["text"],
                **({"title": title} if title else {}),
                **({"lang": lang} if lang else {}),
            }
        })
    idx = _index()
    # idx.upsert(vectors=vecs, namespace=video_id) # opcional: namespace por video
    idx.upsert(vectors=vecs)
    return len(vecs)


# Asegura que el vector es una lista de floats
def _as_list(vec):
    import numpy as np
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float32).tolist()
    if isinstance(vec, list):
        return vec
    raise TypeError(f"query_embedding debe ser list o numpy.ndarray, no {type(vec)}")


# Consulta k vecinos más cercanos
def query(query_embedding, top_k: int = 4, video_id: Optional[str] = None) -> List[Dict]:
    idx = _index()
    flt = {"video_id": {"$eq": video_id}} if video_id else None
    res = idx.query(
        vector=_as_list(query_embedding),
        top_k=top_k,
        include_metadata=True,
        filter=flt
    )
    out = []
    for m in res["matches"]:
        md = m["metadata"]
        out.append({
            "score": float(m["score"]),
            "video_id": md.get("video_id"),
            "start_sec": md.get("start_sec"),
            "end_sec": md.get("end_sec"),
            "text": md.get("text"),
            "title": md.get("title"),
            "lang": md.get("lang"),
        })
    return out
