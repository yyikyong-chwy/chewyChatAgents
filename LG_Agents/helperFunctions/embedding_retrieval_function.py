
from __future__ import annotations
import os, json, math, re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# OpenAI for answer synthesis + query embeddings
OPENAI_MODEL_ANS = os.getenv("CJ_INSURANCE_ANSWER_MODEL", "gpt-4o-mini")
OPENAI_MODEL_EMB = os.getenv("CJ_INSURANCE_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TOP_K = int(os.getenv("CJ_INSURANCE_TOP_K", "6"))
MAX_TOK_CONTEXT = int(os.getenv("CJ_INSURANCE_MAX_CONTEXT", "1800"))  # soft guard
INSURANCE_EMBEDDINGS_INDEX_PATH = os.getenv("CJ_INSURANCE_INDEX", "C:/genAIProjects/new-pet-parent-poc/backend/LG_Agents/embeddings/pet_insurance_index.jsonl")


# ------------------------
# In-memory index cache
# ------------------------
@dataclass
class Chunk:
    id: str
    section: str
    text: str
    vector: Optional[List[float]] = None
    tokens: Optional[List[str]] = None

_INDEX: List[Chunk] = []
_HAS_VECTORS = False

def _load_index(embeddings_index_path: str) -> None:
    global _INDEX, _HAS_VECTORS
    if _INDEX:
        return
    try:
        with open(embeddings_index_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                _INDEX.append(Chunk(
                    id=rec.get("id"),
                    section=rec.get("section") or "Document",
                    text=rec.get("text") or "",
                    vector=rec.get("vector"),
                    tokens=rec.get("tokens"),
                ))
        _HAS_VECTORS = any(ch.vector for ch in _INDEX)
    except FileNotFoundError:
        _INDEX, _HAS_VECTORS = [], False

# ------------------------
# Retrieval
# ------------------------
def _cos_sim(a: List[float], b: List[float]) -> float:
    if not a or not b: return -1.0
    s, na, nb = 0.0, 0.0, 0.0
    for x, y in zip(a, b):
        s += x*y; na += x*x; nb += y*y
    if na == 0 or nb == 0: return -1.0
    return s / math.sqrt(na*nb)

def _embed_query(q: str) -> Optional[List[float]]:
    if not OPENAI_API_KEY:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    emb = client.embeddings.create(model=OPENAI_MODEL_EMB, input=q)
    return emb.data[0].embedding

_WORD = re.compile(r"[a-z0-9]{3,}", re.I)
def _tok(s: str) -> List[str]:
    return _WORD.findall((s or "").lower())

def _keyword_score(qtoks: List[str], doc_toks: List[str]) -> float:
    if not qtoks or not doc_toks: return 0.0
    qset, dset = set(qtoks), set(doc_toks)
    inter = len(qset & dset)
    return inter / (1 + len(qset))

def retrieve(query: str, k: int = TOP_K, embeddings_index_path: str = INSURANCE_EMBEDDINGS_INDEX_PATH) -> List[Chunk]:
    _load_index(embeddings_index_path)
    if not _INDEX:
        return []

    # Try vector search if vectors exist and we can embed the query
    qvec = _embed_query(query) if _HAS_VECTORS else None

    scored: List[Tuple[float, Chunk]] = []
    if qvec is not None:
        for ch in _INDEX:
            sc = _cos_sim(qvec, ch.vector or [])
            scored.append((sc, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
    else:
        # Fallback: keyword overlap scoring
        qt = _tok(query)
        for ch in _INDEX:
            sc = _keyword_score(qt, ch.tokens or _tok(ch.text))
            scored.append((sc, ch))
        scored.sort(key=lambda x: x[0], reverse=True)

    # Deduplicate by near-identical text
    out: List[Chunk] = []
    seen = set()
    for _, ch in scored:
        sig = hash(ch.text[:200])
        if sig in seen: 
            continue
        seen.add(sig)
        out.append(ch)
        if len(out) >= k:
            break
    return out


#helps to determine if the agent response is grounded, complete, and safe
def grounded_score_from_evidence(answer: str, chunks: List[Chunk], used_vectors: bool) -> float:
    """
    Heuristic groundedness score in [0,1].
    Signals:
      - Any evidence retrieved?
      - Does the answer cite sections that were actually retrieved?
      - Small boost if vector search was used (vs. keyword fallback).
    """
    ans = (answer or "").lower()
    if not chunks:
        return 0.2  # no evidence to lean on

    # Look for bracketed section mentions like: [Section: Claims > Filing a claim]
    # and normalize both sides for lenient matching.
    cited_sections = []
    for token in re.findall(r"\[section:\s*([^\]]+)\]", ans, flags=re.I):
        cited_sections.append(token.strip().lower())

    retrieved_sections = [ (ch.section or "").strip().lower() for ch in chunks if ch.section ]
    retrieved_sections = [s for s in retrieved_sections if s]

    if not cited_sections:
        # Evidence exists but user answer didn't anchor to it
        base = 0.5
        return base + (0.05 if used_vectors else 0.0)

    # Overlap ratio between cited and retrieved sections
    cited_set = set(cited_sections)
    retr_set = set(retrieved_sections)
    overlap = len(cited_set & retr_set)
    denom = max(1, len(cited_set))
    overlap_ratio = overlap / denom  # 0..1

    # Scale into upper band; nudge for vector usage
    base = 0.75 + 0.2 * overlap_ratio     # 0.75..0.95
    if used_vectors:
        base += 0.05                      # small bump for semantic retrieval
    return max(0.0, min(1.0, base))    