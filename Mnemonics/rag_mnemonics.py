from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Tuple

# Paths
HERE = Path(__file__).parent
KANJI_JSON = HERE / "merged_kanji.json"

# Try to locate the radicals CSV by pattern (fallback to None)
def _find_radicals_csv() -> Path | None:
    candidates = sorted(HERE.glob("radicals_with_visual_form_*.csv"))
    return candidates[-1] if candidates else None

# Globals initialized lazily
_KANJI_DATA = None
_RADICAL_DOCS: List[str] | None = None
_EMBEDDER = None
_FAISS_INDEX = None
_PIPE = None

_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+")

def _strip_cjk(s: str) -> str:
    return _CJK_RE.sub("", s or "")

def _one_sentence(text: str, max_chars: int = 120) -> str:
    t = (text or "").strip()
    t = re.sub(r"<[^>]*>", "", t)
    t = t.split("→", 1)[0]
    parts = re.split(r"(?<=[.!?])\s+", t)
    one = parts[0] if parts else t
    one = re.sub(r"\s+", " ", one).strip()
    if len(one) > max_chars:
        one = one[:max_chars].rstrip(" ,;:") + "…"
    return one

def _to_ascii(text: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", " ", str(text)).strip()

def _ensure_kanji_data():
    global _KANJI_DATA
    if _KANJI_DATA is None:
        with KANJI_JSON.open("r", encoding="utf-8") as f:
            _KANJI_DATA = json.load(f)

def _ensure_radical_index():
    global _RADICAL_DOCS, _EMBEDDER, _FAISS_INDEX
    if _RADICAL_DOCS is not None and _EMBEDDER is not None and _FAISS_INDEX is not None:
        return
    # Build radical docs from CSV if present; else use empty context
    csv_path = _find_radicals_csv()
    docs: List[str] = []
    if csv_path is not None:
        import pandas as pd
        import numpy as np
        import faiss
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            radical = _to_ascii(row.get("Radical", ""))
            meaning = _to_ascii(row.get("Meaning", ""))
            text = f"Radical: {radical}\nMeaning: {meaning}"
            docs.append(text)
        # Embeddings and FAISS
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        radical_embeddings = _EMBEDDER.encode(docs, show_progress_bar=False)
        dim = radical_embeddings.shape[1]
        _FAISS_INDEX = faiss.IndexFlatL2(dim)
        _FAISS_INDEX.add(radical_embeddings)
    else:
        # Keep empty docs and no index; retrieval will return empty list
        _EMBEDDER = None
        _FAISS_INDEX = None
    _RADICAL_DOCS = docs

def _retrieve_relevant_radicals(query_text: str, top_k: int = 3) -> List[str]:
    if not _RADICAL_DOCS:
        return []
    if _EMBEDDER is None or _FAISS_INDEX is None:
        return []
    import numpy as np
    query_emb = _EMBEDDER.encode([query_text])
    D, I = _FAISS_INDEX.search(np.array(query_emb), top_k)
    return [_RADICAL_DOCS[i] for i in I[0] if 0 <= i < len(_RADICAL_DOCS)]

def _ensure_pipe():
    global _PIPE
    if _PIPE is not None:
        return
    from transformers import pipeline
    _PIPE = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype="auto",
        device_map="auto",
    )

def generate_mnemonic(kanji: str) -> str:
    """Generate a single-line arrow-style English mnemonic using radicals as context."""
    _ensure_kanji_data()
    _ensure_radical_index()
    _ensure_pipe()

    k = kanji.strip()
    details = _KANJI_DATA.get(k)
    if not details:
        return f"Kanji {k} not found."

    meanings = ", ".join(details.get("meanings", []))
    radicals_en = details.get("wk_radicals", [])
    lhs = " + ".join(radicals_en) if radicals_en else meanings

    # Retrieve strictly ASCII-only context
    radical_context = "\n\n".join(_retrieve_relevant_radicals(" ".join(radicals_en)))

    prompt = f"""
You are a Kanji mnemonic generator.
Combine the radicals’ meanings to create a short, logical English mnemonic.

Follow this exact one-line format:
{k} = {lhs} → <short, clear mnemonic>

Rules:
- Use ONLY English words and ASCII punctuation. Never include Japanese/Chinese (kanji, kana, hanzi) in the mnemonic.
- Ignore any non-English visual descriptions; translate their ideas into simple English or omit them.
- Keep it under 120 characters.
- Output exactly one line. No extra text.

Example:
買 = Net + Shell → Buying involves catching valuable shells in a net.

Context about the radicals (English-only):
{radical_context}

Now generate for {k} ("{meanings}").
"""

    try:
        resp = _PIPE(
            prompt,
            max_new_tokens=80,
            temperature=0.4,
            top_p=0.8,
            do_sample=True,
            return_full_text=False,
        )
        text = resp[0]["generated_text"].strip() if resp and isinstance(resp, list) else str(resp).strip()
        text = re.sub(r"\s+", " ", text.replace("|", " "))

        # Try to extract a single arrow-style line
        match = re.search(rf"{re.escape(k)}\s*=\s*.*?→.*", text)
        if match:
            line = match.group(0).strip()
        else:
            # fallback: build manually
            line = f"{k} = {lhs} → Represents {meanings.lower()} through its parts."

        # Sanitize RHS: remove CJK from the RHS while preserving the Kanji on the left; keep one sentence
        if "→" in line:
            left, right = line.split("→", 1)
            right = _strip_cjk(right)
            right = _one_sentence(right, max_chars=120)
            right = re.sub(r"^[=\s]+", "", right)
            if lhs:
                lhs_pat = re.escape(lhs.strip())
                right = re.sub(rf"^(?:{lhs_pat}\b\s*:?\s*)+", "", right, flags=re.I)
            if meanings:
                meanings_pat = re.escape(meanings.strip())
                right = re.sub(rf"^(?:{meanings_pat}\b\s*:?\s*)+", "", right, flags=re.I)
            right = right.strip()
            if not right:
                right = _one_sentence(f"Represents {meanings.lower()}.", max_chars=120)
            if right and right[-1] not in ".!?…":
                right += "."
            line = f"{left.strip()} → {right}"
        return line
    except Exception as e:
        # As a last resort, return a simple fallback line
        fallback_lhs = lhs or (meanings or "—")
        return f"{k} = {fallback_lhs} → Represents {meanings.lower() or 'meaning'} through its parts."
