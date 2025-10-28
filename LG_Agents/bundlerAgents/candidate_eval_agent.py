"""
candidate_eval_agent.py (LLM-driven)

LangGraph node that *evaluates* per-slot candidates against the pet profile,
adds human-readable reasoning for each product, and LLM-reranks the list.

Writes back to:
- CandidateProduct.reasoning      (why it fits this pet/slot)
- CandidateProduct.score_rerank   (final score used for ordering, 0..1)
- CandidateProduct.score_rerank2  (raw "fit" score before price/budget adjustment)

Environment:
  CANDIDATE_EVAL_MODEL   (default: OPENAI_MODEL or gpt-4o-mini)
  CATALOG_EMBEDS_PATH    (default: ./catalog_embeds.jsonl)
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI

import LG_Agents.states.sessionState as sessionState
from LG_Agents.states.bundleState import SlotSpec, CandidateProduct
import LG_Agents.helperFunctions.agent_helper_function as agent_helper_function  # pet_context_helper
# (Optional) if you want to share helpers with slot_candidates_agent, you can import there.
# We re-implement minimal text extraction here for decoupling.

# ----------------------------
# Lightweight catalog reader (to fetch per-candidate text fields, esp. descriptions)
# ----------------------------

_ID_KEYS = ["sku", "id", "product_id"]
_TEXT_KEYS = ["title", "name"]
_DESC_KEYS = ["description", "bullets", "attributes"]
_CAT_KEYS = ["category", "product_type"]
_BRAND_KEYS = ["brand"]

def _pick(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None

def _catalog_index(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a {sku_or_id: product_dict} map from JSONL.
    Safe on big files: one pass, no vectors loaded into RAM.
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return index

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            pid = _pick(obj, _ID_KEYS)
            if not pid:
                continue
            pid = str(pid)
            # keep only light, human-useful fields; ignore embedding to save memory
            index[pid] = {
                "id": pid,
                "sku": str(_pick(obj, ["sku"]) or pid),
                "title": _pick(obj, _TEXT_KEYS),
                "brand": _pick(obj, _BRAND_KEYS),
                "category": _pick(obj, _CAT_KEYS),
                "description": _pick(obj, _DESC_KEYS),
                "price": obj.get("price"),
            }
    return index

def _candidate_payload(c: CandidateProduct, cat: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge CandidateProduct (from retrieval) with any catalog text we can find for the same sku/id.
    """
    pid = c.sku or ""
    raw = cat.get(str(pid), {})
    return {
        "sku": c.sku,
        "title": c.title or raw.get("title"),
        "brand": c.brand or raw.get("brand"),
        "category": c.category or raw.get("category"),
        "price": c.price if c.price is not None else raw.get("price"),
        "rating": c.rating,
        "review_count": c.review_count,
        "description": (raw.get("description") or "")[:1200],  # keep tokens in check
        # retrieval audit fields can help the LLM arbitrate close calls
        "score_vector": c.score_vector,
        "score_fused": c.score_fused,
    }

# ----------------------------
# LLM plumbing
# ----------------------------

SYSTEM_PROMPT = """You are the *Candidate Evaluation Agent* for a new-pet-parent bundle builder.
Given:
- a specific pet profile (species, life stage signals from age/weight, habits, recent conditions, geo events),
- the slot being filled (e.g., leash, puppy food, calming aid) with its reason_for_suggestion, and
- a list of candidate products (with title/brand/category/description/price/ratings),

do ALL of the following, focusing on practical shopping guidance and safety-first choices:

1) For EACH candidate, write a short 1–2 sentence *reasoning* explaining why this product matches (or does not match) THIS pet’s context.
   - Be specific: mention size/weight fit (e.g., small leash for 9–12 lb puppy), behaviors (aggressive chewer, teething, anxious), and any dosing/age constraints.
   - If relevant, note red flags (e.g., wrong weight band, adult-only formula for a puppy, ingredients that might irritate sensitive skin).
   - Keep it neutral and policy-safe; do not invent medical claims.

2) Score each candidate:
   - fit_score: 0.0–1.0 (semantic/contextual fit only; ignore price)
   - price_adjustment: -0.3..+0.3 (how the price affects desirability given any budget hints; mild reward for good value, mild penalty for poor value)
   - final_score: clamp(fit_score + price_adjustment, 0.0, 1.0)

   Guidance:
   - When no budget is given, slightly prefer mid-range value over extreme prices.
   - If product clearly mismatches size/weight/dosing, fit_score should be low regardless of price.

3) Return a best-first ranking by final_score.

Return STRICT JSON with this shape (no commentary outside JSON):
{
  "slot_id": "<slot_id>",
  "evaluations": [
    {
      "sku": "<sku>",
      "fit_score": 0.0,
      "price_adjustment": 0.0,
      "final_score": 0.0,
      "reasoning": "<1–2 sentences>"
    }
  ],
  "reranked_skus": ["<sku-best>", "<sku-2>", "..."]
}
"""

def _call_llm_eval(pet_ctx: dict, slot_ctx: dict, cands: List[Dict[str, Any]], *, temperature: float = 0.2, max_tokens: int = 1800) -> str:
    client = OpenAI()    
    model = os.getenv("CANDIDATE_EVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # keep message compact but structured
    user_payload = {
        "pet_profile": pet_ctx,
        "slot": slot_ctx,
        "candidates": cands,
        "budget_hint": slot_ctx.get("budget", {}),  # may be {}
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"

def _repair_llm_json(pet_ctx: dict, slot_ctx: dict, cands: List[Dict[str, Any]], err: str, raw: str) -> str:
    client = OpenAI()
    model = os.getenv("CANDIDATE_EVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Your previous JSON did not validate. Return ONLY corrected JSON.\n\n"
                f"Validation error: {err}\n\n"
                f"Previous JSON:\n{raw}\n\n"
                f"Context again:\n{json.dumps({'pet_profile': pet_ctx, 'slot': slot_ctx, 'candidates': cands}, ensure_ascii=False)}"
            )},
        ],
        temperature=0,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"

def _coerce_result(raw_json: str) -> Dict[str, Any]:
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object.")
    if "evaluations" not in data or not isinstance(data["evaluations"], list):
        raise ValueError("JSON must contain 'evaluations' array.")
    return data

# ----------------------------
# Main LangGraph node
# ----------------------------

def candidate_eval_agent(state: sessionState) -> sessionState:
    """
    LLM-rerank + reasoning pass over each slot's candidates.
    - Uses RetrievalParams.rerank_top_k to limit how many are sent to the LLM.
    - Writes reasoning + scores back onto CandidateProduct objects.
    - Reorders slot.candidates by final_score (score_rerank) desc.
    """
    if not getattr(state.bundle_state, "slots", None):
        return state

    # 1) Build a lightweight catalog index for text enrichment (optional but helpful for the LLM).
    catalog_path = os.getenv("CATALOG_EMBEDS_PATH", "catalog_embeds.jsonl")
    cat_index = _catalog_index(catalog_path) 

    # 2) Pet + slot context for the LLM
    pet_ctx = agent_helper_function.pet_context_helper(state.pet_profile)

    rp = getattr(state, "retrieval_params", None)
    rerank_cap = int(getattr(rp, "rerank_top_k", 50) or 50)

    updated_slots: List[SlotSpec] = []
    for slot in state.bundle_state.slots:
        # skip if no candidates
        if not getattr(slot, "candidates", None):
            updated_slots.append(slot)
            continue

        # 2a) choose top-N to evaluate (use fused score order if present; else keep original order)
        cands_sorted = list(slot.candidates)
        try:
            cands_sorted.sort(key=lambda c: (c.score_fused if c.score_fused is not None else -1.0), reverse=True)
        except Exception:
            pass
        eval_list = cands_sorted[: min(len(cands_sorted), rerank_cap)]

        # 2b) build per-candidate payloads (merge with catalog text)
        cand_payloads: List[Dict[str, Any]] = [_candidate_payload(c, cat_index) for c in eval_list]

        # 2c) slot summary for the LLM
        slot_ctx = {
            "slot_id": slot.slot_id,
            "name": slot.name,
            "product_type": slot.product_type,
            "reason_for_suggestion": slot.reason_for_suggestion,
            "query_text": slot.query_text,
            "budget": getattr(slot, "budget", None).model_dump() if getattr(slot, "budget", None) else {},
            "filters": getattr(slot, "filters", None).model_dump() if getattr(slot, "filters", None) else {},
        }

        # 2d) call the LLM (with one repair try if needed)
        raw = _call_llm_eval(pet_ctx, slot_ctx, cand_payloads)
        tries = 0
        while True:
            tries += 1
            try:
                parsed = _coerce_result(raw)
                break
            except Exception as e:
                if tries >= 2:
                    # give up gracefully: leave fused order
                    parsed = {"evaluations": [], "reranked_skus": []}
                    break
                raw = _repair_llm_json(pet_ctx, slot_ctx, cand_payloads, str(e), raw)

        # 2e) write back reasoning + scores
        # index by sku for quick join
        eval_by_sku: Dict[str, Dict[str, Any]] = {}
        for ev in parsed.get("evaluations", []):
            sku = str(ev.get("sku", "")).strip()
            if not sku:
                continue
            eval_by_sku[sku] = ev

        # update objects in the original slot.candidates (preserve all objects)
        for c in slot.candidates:
            ev = eval_by_sku.get(str(c.sku), None)
            if ev:
                # raw fit score before price tweak
                c.score_rerank2 = float(ev.get("fit_score")) if ev.get("fit_score") is not None else None
                # final score used for ordering
                c.score_rerank = float(ev.get("final_score")) if ev.get("final_score") is not None else None
                # human explanation
                c.reasoning = str(ev.get("reasoning") or "").strip() or None

        # 2f) reorder: final_score desc, fall back to fused, then vector
        def _key(c: CandidateProduct):
            if c.score_rerank is not None:
                return (1, float(c.score_rerank))
            if c.score_fused is not None:
                return (0, float(c.score_fused))
            return (0, float(c.score_vector or 0.0))

        slot.candidates = sorted(slot.candidates, key=_key, reverse=True)
        updated_slots.append(slot)

    state.bundle_state.slots = updated_slots
    return state

