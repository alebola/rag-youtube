import streamlit as st
from sentence_transformers import SentenceTransformer
from app.utils import yt_id_from_url
from app.ingest import get_transcript_auto, segment_transcript
from app.embeddings import embed_chunks
from app.pinecone_store import ensure_index, upsert_chunks, query
from app.rag_answer import rag_answer_with_citations
from pathlib import Path


APP_TITLE = "YouTube al grano"   # título de la app
WINDOW_SEC = 60                  # tamaño de chunk
OVERLAP_SEC = 12                 # solape de chunk
PREFERRED_LANGS = ("es", "en")   # prioridad de lenguajes
TOP_K = 8                        # vecinos a recuperar
CTX_MAX = 4                      # trozos al LLM
CITE_K = 2                       # nº de citas a mostrar 
MIN_SCORE = 0.40                 # umbral de similitud para filtrar hits

EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_emb_model = None


# Carga el modelo de embeddings 
@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMB_MODEL_NAME)


# Configuración de página + CSS
st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="centered")

st.markdown(
    """
    <style>
    .main { background-color: #fbfbfb; }
    h1, h2, h3 { font-family: "Segoe UI", system-ui, -apple-system, Roboto, "Helvetica Neue", Arial; }
    h1 { text-align:center; color:#ff4b4b; font-weight: 800; letter-spacing: 0.2px; }
    .stButton>button {
        background-color: #ff4b4b; color: white; border-radius: 8px;
        font-weight: 700; padding: 0.6rem 1rem; border: none;
    }
    .stButton>button:hover { background-color: #ff1f1f; color: white; }
    .card {
        background: white; border: 1px solid #eee; border-radius: 10px;
        padding: 1rem 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .cite-item {
        background: #f6f6f9; border-radius: 8px; padding: 0.45rem 0.6rem; margin-bottom: 0.3rem;
        border: 1px solid #ededf6;
    }
    .muted { color:#6b7280; font-size:0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"# {APP_TITLE}")
st.caption("Indexa un video, haz preguntas y obtén respuestas breves con enlaces al minuto exacto.")


# Sección: Indexar
st.subheader("1) Pega un link e indexa 🔗")
col_url, col_btn = st.columns([0.75, 0.25])
with col_url:
    url = st.text_input("URL del vídeo de YouTube", placeholder="https://www.youtube.com/watch?v=...")
with col_btn:
    index_btn = st.button("Indexar", type="primary")

if index_btn:
    if not url.strip():
        st.error("Por favor, pega una URL de YouTube válida.")
        st.stop()

    # Extraer el VIDEO_ID
    try:
        vid = yt_id_from_url(url)
    except Exception as e:
        st.error(f"No se pudo extraer el VIDEO_ID: {e}")
        st.stop()

    # Mostrar miniatura 
    st.image(
        f"https://img.youtube.com/vi/{vid}/0.jpg",
        caption="Miniatura del video",
        use_column_width=True  
    )


    # Ingesta + chunking + embeddings 
    with st.spinner("Descargando y segmentando subtítulos..."):
        # 1) Intento anónimo
        try:
            rows = get_transcript_auto(
                vid,
                preferred_langs=("es","es-419","en"),
                fallback_url=url,
                cookiefile="cookies.txt",   
                max_retries=6,
                backoff_base=2.0,
            )
        except Exception as e1:
            cookie_txt = Path("cookies.txt")
            tried_cookiefile = False
            if cookie_txt.exists():
                try:
                    rows = get_transcript_auto(
                        vid,
                        preferred_langs=PREFERRED_LANGS,
                        fallback_url=url,
                        cookiefile=str(cookie_txt),
                    )
                    tried_cookiefile = True
                except Exception:
                    pass

            if not tried_cookiefile:
                # 3) Intentar cookies del navegador (Chrome, luego Edge)
                got = False
                for browser in (('chrome',), ('edge',)):
                    try:
                        st.warning(
                            f"No pude obtener subtítulos de forma anónima ({e1}). "
                            f"Intentando usar cookies de {browser[0]}…"
                        )
                        rows = get_transcript_auto(
                            vid,
                            preferred_langs=PREFERRED_LANGS,
                            fallback_url=url,
                            cookiesfrombrowser=browser,
                        )
                        got = True
                        break
                    except Exception:
                        continue
                if not got:
                    st.error(
                        "No se pudieron obtener subtítulos (API + yt-dlp). "
                        "Prueba a exportar cookies a 'cookies.txt' o cierra el navegador/usa el flag "
                        "--disable-features=LockProfileCookieDatabase y reintenta."
                    )
                    st.stop()

    # 2) Segmentación
    chunks = segment_transcript(rows, window=WINDOW_SEC, overlap=OVERLAP_SEC)
    # 3) Embeddings de chunks
    chunks_with_embs = embed_chunks(chunks)


    # Pinecone: asegurar índice + upsert
    with st.spinner("🚀 Subiendo embeddings al índice vectorial..."):
        ensure_index()
        n = upsert_chunks(
            chunks_with_embs,
            video_id=vid,
            title="YouTube video",
            lang="auto",
        )

    # Guardar estado de sesión 
    st.session_state["last_video_id"] = vid
    st.session_state["last_url"] = url

    st.success(f"✅ Indexado OK. Vectores subidos: {n}.")
    st.info("Ya puedes preguntar justo debajo 👇")


# Sección: Preguntar
st.subheader("2) Pregunta sobre el vídeo 🎤")

last_vid = st.session_state.get("last_video_id")
if not last_vid:
    st.warning("Primero indexa un vídeo arriba ⬆️")
else:
    # Re-mostramos miniatura (por claridad visual)
    st.image(f"https://img.youtube.com/vi/{last_vid}/0.jpg", use_column_width=True)

    question = st.text_area("Tu pregunta", placeholder="¿Qué dice el video sobre ...", height=120)
    ask_btn = st.button("Responder", type="primary", key="ask_button")

    if ask_btn:
        if not question.strip():
            st.error("Escribe una pregunta.")
            st.stop()

        # 1) Embedding de la pregunta
        with st.spinner("🔎 Buscando fragmentos relevantes..."):
            embedder = get_embedder()
            q_vec = embedder.encode([question], convert_to_numpy=True)[0]

            # 2) Consulta a Pinecone 
            hits_all = query(q_vec, top_k=TOP_K, video_id=last_vid)
            hits = [h for h in hits_all if float(h.get("score", 0)) >= MIN_SCORE]

            if not hits:
                st.info("No encontré fragmentos suficientemente relevantes en este video.")
                st.stop()

        # 3) RAG: respuesta + citas 
        with st.spinner("🧠 Generando respuesta..."):
            answer, citations = rag_answer_with_citations(
                last_vid, question, hits, ctx_max=CTX_MAX, cite_k=CITE_K
            )

        # 4) Render bonito
        st.markdown("**Respuesta**")
        st.markdown(f'<div class="card">{answer}</div>', unsafe_allow_html=True)

        st.markdown("**Citas (enlace al minuto exacto)**")
        if not citations:
            st.write("—")
        else:
            for c in citations:
                st.markdown(
                    f'<div class="cite-item">⏱️ <strong>{c["minute"]}</strong> — '
                    f'<a href="{c["url"]}" target="_blank">{c["url"]}</a></div>',
                    unsafe_allow_html=True,
                )
