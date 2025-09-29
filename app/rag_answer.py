from typing import List, Dict, Tuple
from .utils import hhmmss, time_url
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch


_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# _DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Cargamos el modelo una sola vez
_tokenizer = None
_model = None
_pipe: TextGenerationPipeline | None = None


def _load_llm(model_name: str = _DEFAULT_MODEL) -> TextGenerationPipeline:
    global _tokenizer, _model, _pipe
    if _pipe is not None:
        return _pipe
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   # CPU
        low_cpu_mem_usage=True
    )
    _pipe = TextGenerationPipeline(
        model=_model,
        tokenizer=_tokenizer,
        device=-1,                   # CPU
        return_full_text=False
    )
    return _pipe


# Construye un prompt conciso
def build_prompt(question: str, hits: List[Dict], max_snippets: int = 3) -> str:
    context_lines = []
    for h in hits[:max_snippets]:
        # Limitar tamaño de snippet para no reventar contexto
        txt = h["text"].strip()
        if len(txt) > 400:
            txt = txt[:400] + "..."
        rng = f"[{hhmmss(h['start_sec'])}–{hhmmss(h['end_sec'])}]"
        context_lines.append(f"{rng} {txt}")
    context = "\n".join(context_lines) if context_lines else "(sin contexto disponible)"

    system = (
        "Eres un asistente conciso. Responde SOLO con la información del contexto.\n"
        "No inventes. Si no está en el contexto, di que no aparece.\n"
        "Responde en 1-3 frases como máximo."
    )
    user = (
        f"Pregunta: {question}\n\n"
        f"Contexto (fragmentos de subtítulos):\n{context}\n\n"
        "Responde de forma breve y directa basándote solo en lo anterior."
    )
    # Formato simple estilo chat
    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    return prompt


# Genera respuesta breve con LLM y contexto
def generate_rag_answer(question: str, hits: List[Dict], model_name: str = _DEFAULT_MODEL) -> str:
    pipe = _load_llm(model_name)
    prompt = build_prompt(question, hits)
    out = pipe(
        prompt,
        max_new_tokens=160,
        do_sample=False,     # determinista
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.05
    )
    text = out[0]["generated_text"].strip()
    # Un pequeño saneo
    return text.replace("\n\n", " ").replace("  ", " ")


# Devuelve citas listas para renderizar
def format_citations(video_id: str, hits: List[Dict]) -> List[Dict]:
    citations = []
    for h in hits:
        citations.append({
            "range": f"{hhmmss(h['start_sec'])}–{hhmmss(h['end_sec'])}",
            "url": time_url(video_id, h["start_sec"]),
            "snippet": (h["text"][:140] + "...") if len(h["text"]) > 140 else h["text"]
        })
    return citations


# Orquesta respuesta + citas
def rag_answer_with_citations(
    video_id: str,
    question: str,
    hits: List[Dict],
    model_name: str = _DEFAULT_MODEL
) -> Tuple[str, List[Dict]]:
    if not hits:
        return "No encontré fragmentos relevantes en los subtítulos.", []
    answer = generate_rag_answer(question, hits, model_name=model_name)
    citations = format_citations(video_id, hits)
    return answer, citations
