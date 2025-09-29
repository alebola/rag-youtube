from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np


# Cargamos el modelo una sola vez
_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# Devuelve un array [n, dim] con embeddings para cada texto.
def embed_texts(texts: List[str]) -> np.ndarray:
    return _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


# Añade embeddings a cada chunk del transcript.
def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    texts = [c["text"] for c in chunks]
    embs = embed_texts(texts)
    out = []
    for c, e in zip(chunks, embs):
        c2 = dict(c)
        c2["embedding"] = e.tolist()  # guardamos como lista para JSON/DB
        out.append(c2)
    return out
