import streamlit as st
import json
import streamlit.components.v1 as components
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


# Efecto máquina de escribir para la respuesta
def typewriter_card(text: str, height: int | None = None):
    if height is None:
        chars_per_line = 95
        import math
        lines = max(1, math.ceil(len(text) / chars_per_line))
        height = 90 + lines * 26 

    html = f"""
    <div class="card typewriter-card">
        <div class="card-title">Respuesta</div>
        <div id="tw" style="font-size:1rem; line-height:1.6; color:#111827; font-family:'Poppins', sans-serif;"></div>
    </div>
    <script>
        const el = document.getElementById('tw');
        const txt = {json.dumps(text)};
        let i = 0;
        const speed = 14;
        function step() {{
            if (i <= txt.length) {{
                el.textContent = txt.slice(0, i);
                i++;
                setTimeout(step, speed);
            }}
        }}
        step();
    </script>
    """
    components.html(html, height=height, scrolling=False)


@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMB_MODEL_NAME)


st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        :root{
            --primary:#ff4b4b;          
            --primary-hover:#ff1f1f;
            --secondary:#1f2937;        
            --muted:#6b7280;
            --panel:#ffffff;
            --panel-alt:#f3f4f6;        
            --ring:rgba(0,0,0,0.04);
        }

        .main { background-color:#fbfbfb; }
        .block-container { max-width:1280px; }

        h1 { text-align:left; color:var(--primary); font-weight:800; letter-spacing:.2px; }
        h2, h3 { color:var(--secondary); font-weight:800; }

        .stButton>button {
            background-color:var(--primary) !important; 
            color:#fff !important; 
            border-radius:10px;
            font-weight:700; 
            padding:.6rem 1rem; 
            border:none;
            box-shadow:0 4px 12px var(--ring);
        }
        .stButton>button:hover { background-color:var(--primary-hover) !important; }

        .card {
            background:var(--panel); 
            border:1px solid #eee; 
            border-radius:12px;
            padding:1rem 1.1rem; 
            box-shadow:0 6px 20px var(--ring);
        }
        .card-title { color:var(--secondary); font-weight:800; margin-bottom:.35rem; }

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif !important;
        }
        .typewriter-card, .typewriter-card * {
            font-family: 'Poppins', sans-serif !important;
            font-size: 1.05rem;
            line-height: 1.5;
        }

        .cites { margin-top:.6rem; }
        .cite-item {
            background:var(--panel-alt); 
            border-radius:10px; 
            padding:.48rem .65rem; 
            margin-bottom:.35rem;
            border:1px solid #e5e7eb;
        }
        .cite-link a{ 
            text-decoration:none; 
            color:#2563eb; 
            font-weight:600; 
        }
        .muted { color:var(--muted); font-size:.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"# {APP_TITLE}")
st.caption("Indexa un video, haz preguntas y obtén respuestas breves con enlaces al minuto exacto.")

col_left, col_right = st.columns([0.48, 0.52], gap="large")

# IZQUIERDA: indexar + miniatura 
with col_left:
    st.subheader("1) Pega un link e indexa")
    st.markdown("**URL del vídeo de YouTube**")  
    c1, c2 = st.columns([0.78, 0.22])
    with c1:
        url = st.text_input(
            "URL del vídeo de YouTube", 
            placeholder="https://www.youtube.com/watch?v=...", 
            label_visibility="collapsed"
        )

    with c2:
        index_btn = st.button("Indexar", type="primary")

    if index_btn:
        if not url.strip():
            st.error("Por favor, pega una URL de YouTube válida.")
            st.stop()
        try:
            vid = yt_id_from_url(url)
        except Exception as e:
            st.error(f"No se pudo extraer el VIDEO_ID: {e}")
            st.stop()

        with st.spinner("Descargando y segmentando subtítulos..."):
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

        chunks = segment_transcript(rows, window=WINDOW_SEC, overlap=OVERLAP_SEC)
        chunks_with_embs = embed_chunks(chunks)

        with st.spinner("🚀 Subiendo embeddings al índice vectorial..."):
            ensure_index()
            n = upsert_chunks(
                chunks_with_embs,
                video_id=vid,
                title="YouTube video",
                lang="auto",
            )

        st.session_state["last_video_id"] = vid
        st.session_state["last_url"] = url
        st.success(f"✅ Indexado OK. Vectores subidos: {n}.")

    last_vid_for_thumb = st.session_state.get("last_video_id")
    if last_vid_for_thumb:
        thumb_url = f"https://img.youtube.com/vi/{last_vid_for_thumb}/0.jpg"
        st.markdown(
            f'''
            <div style="display:flex;justify-content:center;margin: 10px 0 6px;">
              <img src="{thumb_url}" alt="Miniatura del video"
                     style="max-width: 520px; width: 100%; border-radius: 12px;
                          box-shadow: 0 8px 24px rgba(0,0,0,.08); border:1px solid #eee;">
            </div>
            ''',
            unsafe_allow_html=True
        )

# DERECHA: preguntas -> respuesta + citas 
with col_right:
    st.subheader("2) Pregunta sobre el vídeo")

    last_vid = st.session_state.get("last_video_id")
    if not last_vid:
        st.warning("Primero indexa un vídeo a la izquierda ⬅️")
    else:
        question = st.text_area("Tu pregunta", placeholder="¿Qué dice el video sobre ...", height=120)
        ask_btn = st.button("Responder", type="primary", key="ask_button")

        if ask_btn:
            if not question.strip():
                st.error("Escribe una pregunta.")
                st.stop()

            with st.spinner("🔎 Buscando fragmentos relevantes..."):
                embedder = get_embedder()
                q_vec = embedder.encode([question], convert_to_numpy=True)[0]
                hits_all = query(q_vec, top_k=TOP_K, video_id=last_vid)
                hits = [h for h in hits_all if float(h.get("score", 0)) >= MIN_SCORE]

            if not hits:
                st.info("No encontré fragmentos suficientemente relevantes en este video.")
                st.stop()

            with st.spinner("🧠 Generando respuesta..."):
                answer, citations = rag_answer_with_citations(
                    last_vid, question, hits, ctx_max=CTX_MAX, cite_k=CITE_K
                )

            typewriter_card(answer)

            st.markdown("**Citas**")
            if not citations:
                st.write("—")
            else:
                for c in citations:
                    st.markdown(
                        f'<div class="cite-item">⏱️ <strong>{c["minute"]}</strong> — '
                        f'<a href="{c["url"]}" target="_blank">{c["url"]}</a></div>',
                        unsafe_allow_html=True,
                    )
