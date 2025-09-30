from typing import List, Dict, Tuple
from .utils import hhmmss, time_url
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import re


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


# Detiene la generación si el modelo empieza a escribir '[End of answer]'.
class StopOnEnd(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = tokenizer("[End of answer]", add_special_tokens=False, return_tensors="pt").input_ids[0]
    def __call__(self, input_ids, scores, **kwargs):
        L = self.stop_ids.shape[0]
        if input_ids.shape[1] >= L and (input_ids[0, -L:] == self.stop_ids).all():
            return True
        return False


# Genera respuesta breve con LLM y contexto
def generate_rag_answer(question: str, hits: List[Dict], model_name: str = _DEFAULT_MODEL) -> str:
    pipe = _load_llm(model_name)

    ctx_lines = []
    for h in hits[:4]:
        txt = h["text"].strip()
        if len(txt) > 400:
            txt = txt[:400] + "..."
        rng = f"[{hhmmss(h['start_sec'])}–{hhmmss(h['end_sec'])}]"
        ctx_lines.append(f"{rng} {txt}")
    has_context = len(ctx_lines) > 0
    context = "\n".join(ctx_lines) if has_context else "(no context)"

    system = (
        "You are a factual, concise QA assistant for YouTube videos. "
        "You must follow the RULES exactly.\n"
        "RULES:\n"
        "1) Use ONLY information inside the <context> block.\n"
        "2) If the answer is not explicitly supported by <context>, reply with EXACTLY ONE of these strings, "
        "matching the QUESTION language, with no extra words:\n"
        "   - If the question is in english → Not found in the subtitles.\n"
        "   - If the question is in spanish → No se encuentra en los subtítulos.\n"
        "3) Do NOT use outside knowledge. Do NOT guess.\n"
        "4) Write 1–2 sentences, entirely in the SAME language as the QUESTION (no code-switching).\n"
        "5) Do NOT include timestamps, links, citations, lists, bullet points, labels, or closing markers.\n"
        "6) Output must be ONLY the final answer text. No preambles, no language labels."
    )

    user = (
        "QUESTION:\n"
        f"{question}\n\n"
        "CONTEXT (YouTube subtitles) delimited by tags:\n"
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        "INSTRUCTIONS:\n"
        "- Answer briefly (1–2 sentences) using ONLY facts from <context>.\n"
        "- If <context> is empty or does not contain the answer, reply EXACTLY with the single matching string specified in RULE 2.\n"
        "- Do not add anything else."
    )


    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stop_criteria = StoppingCriteriaList([StopOnEnd(_tokenizer)])

    out = _pipe(
        prompt,
        max_new_tokens=160,
        do_sample=False,
        repetition_penalty=1.05,
        stopping_criteria=stop_criteria,
        eos_token_id=_tokenizer.eos_token_id,
    )

    text = out[0]["generated_text"].strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text



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
    hits: list[dict],
    model_name: str = _DEFAULT_MODEL,
    ctx_max: int = 4,
    cite_k: int = 2,
    min_gap_sec: float = 45.0,
    min_top_score: float = 0.35,   
):
    if not hits:
        return "Not found in the subtitles.", []

    # Ordena por score desc y aplica umbral ANTES de todo
    hits_sorted = sorted(hits, key=lambda h: h["score"], reverse=True)
    if hits_sorted[0]["score"] < min_top_score:
        return "Not found in the subtitles.", []  # sin citas

    # De-dup temporal + preparar contexto
    hits_dedup = dedup_hits_by_time(hits_sorted, min_gap_sec=min_gap_sec)
    context_hits = hits_dedup[:ctx_max]

    answer = generate_rag_answer(question, context_hits, model_name=model_name)

    # Si el modelo niega, no mostramos citas
    if answer.strip() in ("Not found in the subtitles.", "No se encuentra en los subtítulos."):
        return answer, []

    # Citas: top por score (post-dedupe), luego cronológico
    top_for_citation = sorted(hits_dedup, key=lambda h: h["score"], reverse=True)[:cite_k]
    top_for_citation = sorted(top_for_citation, key=lambda h: h["start_sec"])
    citations = [{"minute": hhmmss(h["start_sec"]), "url": time_url(video_id, h["start_sec"])} for h in top_for_citation]

    return answer, citations



# Elimina hits muy cercanos en el tiempo (por solapamiento de chunks)
def dedup_hits_by_time(hits: list[dict], min_gap_sec: float = 30.0) -> list[dict]:
    if not hits:
        return []
    hits_sorted = sorted(hits, key=lambda h: h["score"], reverse=True)
    kept = []
    last_kept_start = None
    for h in hits_sorted:
        s = float(h["start_sec"])
        if last_kept_start is None or abs(s - last_kept_start) >= min_gap_sec:
            kept.append(h)
            last_kept_start = s
    return kept