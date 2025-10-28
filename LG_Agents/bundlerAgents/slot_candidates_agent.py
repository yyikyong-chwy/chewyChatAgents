"""
slot_candidates_agent.py

LangGraph node that retrieves top-N candidate products for each planned slot
by searching a local embedding index (catalog_embeds.jsonl) and fusing scores.

Pipeline (per slot):
  1) Build dense embedding for slot.query_text (and soft_terms).
  2) Cosine similarity vs precomputed product vectors from JSONL.
  3) Add a small lexical bonus (token overlap against product text fields).
  4) (Optional) MMR for diversity.
  5) Return top-N as SlotSpec.candidates (List[CandidateProduct]).

Environment:
  CATALOG_EMBEDS_PATH   (default: ./catalog_embeds.jsonl)
  EMBEDDING_MODEL       (default: text-embedding-3-large)
  OPENAI_API_KEY        (required to embed queries)

Depends on your Pydantic models in nppState.py.

JSONL EXPECTED FIELDS (best-effort; flexible):
  - id | sku  (unique id; if missing, we synthesize one)
  - title | name
  - brand (optional)
  - category | product_type (optional)
  - price (optional)
  - rating, review_count (optional)
  - embedding | vector | embed | embedding_1536 (list[float])

If your schema differs, adjust `_extract_product_fields`.
"""

from __future__ import annotations

import os
import json
import math
import re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from openai import OpenAI

from LG_Agents.states.sessionState import SessionStateModel
from LG_Agents.states.bundleState import SlotSpec, CandidateProduct


# ----------------------------
# Config / Globals
# ----------------------------
_CATALOG_CACHE: Optional[List[Dict[str, Any]]] = None
_VECTOR_KEY_CANDIDATES = ["embedding", "vector", "embed", "embedding_1536"]
_TEXT_KEY_CANDIDATES = ["title", "name"]
_ID_KEY_CANDIDATES = ["sku", "id", "product_id"]


# ----------------------------
# Catalog loading
# ----------------------------
def _load_catalog(path: str) -> List[Dict[str, Any]]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    if not os.path.exists(path):
        raise FileNotFoundError(f"catalog file not found: {path}")

    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # normalize vector
            vec = None
            for k in _VECTOR_KEY_CANDIDATES:
                if k in obj and isinstance(obj[k], list):
                    vec = obj[k]
                    break
            if not vec:
                # skip items without vectors
                continue
            # ensure numpy
            try:
                vec = np.array(vec, dtype=np.float32)
            except Exception:
                continue
            obj["_embedding_np"] = vec
            items.append(obj)
    if not items:
        raise RuntimeError("No valid embedded items found in catalog.")

    _CATALOG_CACHE = items
    return items


def _extract_product_fields(prod: Dict[str, Any]) -> Tuple[str, str, Optional[str], Optional[str], Optional[float], Optional[float], Optional[int]]:
    # id/sku
    sku = None
    for k in _ID_KEY_CANDIDATES:
        if k in prod and prod[k]:
            sku = str(prod[k])
            break
    if sku is None:
        # synthesize a stable-ish id from position/hash
        sku = str(abs(hash(json.dumps(prod, sort_keys=True))) % (10**12))

    # title/name
    title = None
    for k in _TEXT_KEY_CANDIDATES:
        if k in prod and prod[k]:
            title = str(prod[k])
            break
    if not title:
        title = "Untitled product"

    brand = prod.get("brand")
    category = prod.get("category") or prod.get("product_type")
    price = prod.get("price")
    rating = prod.get("rating")
    review_count = prod.get("review_count")

    return sku, title, brand, category, price, rating, review_count


def _product_lexical_text(prod: Dict[str, Any]) -> str:
    fields = []
    for k in ["title", "name", "brand", "category", "description", "bullets", "tags", "attributes"]:
        v = prod.get(k)
        if isinstance(v, str):
            fields.append(v)
        elif isinstance(v, list):
            fields.extend([str(x) for x in v])
        elif isinstance(v, dict):
            fields.extend([f"{kk}:{vv}" for kk, vv in v.items()])
    return " ".join(fields).lower()


# ----------------------------
# Embeddings & scoring
# ----------------------------
def _embed_query(text: str) -> np.ndarray:
    client = OpenAI()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    # OpenAI embeddings expect a single string or list; returns array of floats
    resp = client.embeddings.create(model=model, input=text)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    # Normalize to unit length for fast cosine
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _cosine_sim(q: np.ndarray, m: np.ndarray) -> float:
    # m is assumed normalized at load-time? (not guaranteed) -> normalize defensively
    denom = (np.linalg.norm(m) + 1e-8)
    return float(np.dot(q, m / denom))


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())


def _lexical_bonus(query_text: str, soft_terms: List[str], product_text: str) -> float:
    # very lightweight overlap score in [0, ~1]
    q_tokens = set(_tokenize(query_text) + sum([_tokenize(t) for t in soft_terms], []))
    if not q_tokens:
        return 0.0
    p_tokens = set(_tokenize(product_text))
    overlap = q_tokens.intersection(p_tokens)
    return min(len(overlap) / max(len(q_tokens), 1), 1.0)  # normalized


def _fuse_scores(vec_score: float, lex_bonus: float, alpha: float = 0.85) -> float:
    # alpha near 1 favors vector; (1-alpha) is lexical bonus contribution
    return alpha * vec_score + (1.0 - alpha) * lex_bonus


def _mmr(
    candidates: List[Tuple[int, float, np.ndarray]],
    lambda_mult: float,
    k: int,
) -> List[int]:
    """
    Maximal Marginal Relevance:
      candidates: list of (idx_in_catalog, fused_score, embed)
      returns list of selected indices into 'candidates'
    """
    if not candidates:
        return []
    # sort by initial score descending
    order = sorted(range(len(candidates)), key=lambda i: candidates[i][1], reverse=True)
    selected = [order[0]]
    selected_vecs = [candidates[order[0]][2]]

    while len(selected) < min(k, len(candidates)):
        best_i = None
        best_val = -1e9
        for i in range(len(candidates)):
            if i in selected:
                continue
            sim_to_selected = 0.0
            if selected_vecs:
                # diversity penalty = max cosine to any selected
                sims = [float(np.dot(candidates[i][2], v)) for v in selected_vecs]
                sim_to_selected = max(sims)
            mmr_val = lambda_mult * candidates[i][1] - (1.0 - lambda_mult) * sim_to_selected
            if mmr_val > best_val:
                best_val = mmr_val
                best_i = i
        selected.append(best_i)
        selected_vecs.append(candidates[best_i][2])
    return selected


# ----------------------------
# Main agent
# ----------------------------
def slot_candidates_agent(state: SessionStateModel) -> SessionStateModel:
    """
    For each slot in state.slots that has a query_text, retrieve top-N candidates
    from the local embedding catalog and write them into slot.candidates.

    Uses state.retrieval_params for k and MMR options. Soft-terms are optional.
    """
    # Load catalog (cached)
    catalog_path = os.getenv("CATALOG_EMBEDS_PATH", "catalog_embeds.jsonl")
    catalog = _load_catalog(catalog_path)

    # Precompute product texts for lexical bonus
    prod_texts = [""] * len(catalog)
    prod_vecs = [None] * len(catalog)
    for i, prod in enumerate(catalog):
        prod_texts[i] = _product_lexical_text(prod)
        prod_vecs[i] = prod["_embedding_np"]
        # normalize vectors for stable cosine
        nv = np.linalg.norm(prod_vecs[i])
        if nv > 0:
            prod_vecs[i] = prod_vecs[i] / nv

    # Retrieval params
    rp = state.bundle_state.retrieval_params
    fuse_k = getattr(rp, "fuse_k", 100)           # candidates to consider for fusion/mmr
    top_k = getattr(rp, "top_k", 20)             # final candidates returned
    apply_mmr = getattr(rp, "apply_mmr", True)
    mmr_lambda = getattr(rp, "mmr_lambda", 0.7)

    for slot in state.bundle_state.slots:
        if not isinstance(slot, SlotSpec):
            continue
        if not getattr(slot, "query_text", None):
            # if no query text, skip this slot
            slot.candidates = []
            continue

        qtext = slot.query_text.strip()
        soft_terms = getattr(slot, "soft_terms", []) or []
        # Embed the main query; you can optionally append soft terms to input text,
        # but we prefer to keep dense signal clean and use soft terms lexically.
        q_vec = _embed_query(qtext)

        # Score all products (vector + lexical)
        scored: List[Tuple[int, float, np.ndarray, float, float]] = []  # (idx, fused, vec, vec_score, lex_bonus)
        for i, pv in enumerate(prod_vecs):
            vec_score = _cosine_sim(q_vec, pv)
            lex_bonus = _lexical_bonus(qtext, soft_terms, prod_texts[i])
            fused = _fuse_scores(vec_score, lex_bonus, alpha=0.88)
            scored.append((i, fused, pv, vec_score, lex_bonus))

        # Take top fuse_k by fused score
        scored.sort(key=lambda x: x[1], reverse=True)
        pool = scored[: max(fuse_k, top_k)]

        # Optional: MMR for diversity
        if apply_mmr and len(pool) > top_k:
            cand_for_mmr = [(i, fused, vec) for (i, fused, vec, _, _) in pool]
            sel_indices_local = _mmr(cand_for_mmr, lambda_mult=mmr_lambda, k=top_k)
            final = [pool[i] for i in sel_indices_local]
        else:
            final = pool[:top_k]

        # Build CandidateProduct list
        cands: List[CandidateProduct] = []
        for (idx, fused, _vec, vec_score, lex_bonus) in final:
            prod = catalog[idx]
            sku, title, brand, category, price, rating, review_count = _extract_product_fields(prod)
            cp = CandidateProduct(
                sku=sku,
                title=title,
                price=price,
                brand=brand,
                category=category,
                rating=rating,
                review_count=review_count,
                score_vector=round(float(vec_score), 6),
                score_bm25=None,  # not used here
                score_fused=round(float(fused), 6),
                score_rerank=None,
            )
            cands.append(cp)

        slot.candidates = cands

    return state

