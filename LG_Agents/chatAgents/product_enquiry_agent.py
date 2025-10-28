# product_enquiry_agent.py
from __future__ import annotations

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.messages import AIMessage

# --- State & models
from LG_Agents.states.sessionState import SessionStateModel  # container
from LG_Agents.states.ChewyJourneyChatState import ChewyJourneyChatState
from LG_Agents.states.bundleState import SlotSpec, CandidateProduct, FilterSpec, Budget

# --- Reuse your retrieval + eval nodes
from LG_Agents.bundlerAgents.slot_candidates_agent import slot_candidates_agent as _retrieve_candidates
from LG_Agents.bundlerAgents.candidate_eval_agent import candidate_eval_agent as _eval_candidates


# =========================
# Config
# =========================
TOP_N = int(os.getenv("PRODUCT_ENQUIRY_TOP_N", 3))  # how many to show in the reply
SLOT_ID = os.getenv("PRODUCT_ENQUIRY_SLOT_ID", "one_off_product")


# =========================
# Lightweight NLP helpers
# =========================
_BRANDS_HINT = [
    "Frisco", "MidWest", "Petmate", "IRIS", "PetSafe", "KONG", "Van Ness", "Arm & Hammer",
    "Catit", "Litter-Robot", "Furminator", "PetSafe", "Blue Buffalo", "Hill's", "Purina",
]

_PRODUCT_SYNONYMS = [
    # (regex, normalized product_type, soft terms)
    (r"\b(crates?|cages?)\b", "crate", ["wire", "folding", "double door"]),
    (r"\b(carriers?)\b", "carrier", ["airline approved", "hard-sided", "soft-sided"]),
    (r"\b(litter\s*boxes?|litterbox|litter-box)\b", "litter box", ["high-sided", "covered", "self-cleaning"]),
    (r"\b(beds?)\b", "bed", ["orthopedic", "bolster", "washable"]),
    (r"\b(scratching\s*posts?|scratchers?)\b", "scratching post", ["sisal", "stability"]),
    (r"\b(harness(es)?|leashes?|collars?)\b", "harness/leash/collar", ["no-pull", "adjustable"]),
    (r"\b(fountains?|water\s*fountain)\b", "water fountain", ["quiet", "stainless steel"]),
    (r"\b(toys?)\b", "toy", ["interactive", "tough chew"]),
    (r"\b(foods?|kibble|dry\s*food|wet\s*food)\b", "food", ["grain-free", "sensitive stomach"]),
    (r"\b(shampoos?|grooming)\b", "grooming", ["hypoallergenic", "oatmeal"]),
]

_PRICE_RE = re.compile(
    r"(?:under|below|less than|<=?)\s*\$?\s*(\d+(?:\.\d{1,2})?)|"
    r"\$+\s*(\d+(?:\.\d{1,2})?)", re.I
)

_SIZE_TERMS = ["xs", "extra small", "small", "medium", "large", "xl", "extra large", "xxl"]


def _life_stage(species: Optional[str], age_months: Optional[float]) -> Optional[str]:
    if species is None or age_months is None:
        return None
    s = species.lower()
    a = float(age_months)
    if s == "dog":
        if a < 12: return "puppy"
        if a >= 96: return "senior"
    if s == "cat":
        if a < 12: return "kitten"
        if a >= 120: return "senior"
    return "adult"


def _extract_product_intent(text: str, species: Optional[str]) -> Tuple[str, List[str]]:
    """Return (product_type, soft_terms) from the user's ask."""
    t = text.lower()
    for pat, norm, extras in _PRODUCT_SYNONYMS:
        if re.search(pat, t):
            # Special-case: people say 'cage' for cats but usually mean 'carrier'
            if norm == "crate" and species and species.lower() == "cat":
                return "carrier", extras
            return norm, extras
    # default: just echo last noun-ish token fallback
    return "product", []


def _extract_budget(text: str) -> Optional[float]:
    m = _PRICE_RE.search(text or "")
    if not m:
        return None
    val = m.group(1) or m.group(2)
    try:
        return float(val)
    except Exception:
        return None


def _extract_brands(text: str) -> List[str]:
    found = []
    for b in _BRANDS_HINT:
        if re.search(fr"\b{re.escape(b)}\b", text, flags=re.I):
            found.append(b)
    return found


def _extract_size_terms(text: str) -> List[str]:
    hits = []
    for s in _SIZE_TERMS:
        if re.search(fr"\b{s}\b", text, flags=re.I):
            hits.append(s.lower())
    return hits


def _build_query_text(species: Optional[str],
                      breed: Optional[str],
                      life_stage: Optional[str],
                      product_type: str,
                      weight_lb: Optional[float],
                      user_text: str,
                      size_terms: List[str]) -> str:
    parts = []
    if species: parts.append(species)
    if breed: parts.append(breed)
    if life_stage: parts.append(life_stage)
    parts.append(product_type)
    if weight_lb:
        parts.append(f"{int(round(weight_lb))} lb")
    # keep user ask for extra specificity
    parts.append(user_text)
    # sprinkle size words lightly
    if size_terms:
        parts.extend(size_terms[:2])
    return " ".join([p for p in parts if p]).strip()


# =========================
# Main node
# =========================
def product_enquiry_agent(state: SessionStateModel) -> SessionStateModel:
    """
    Builds a single-slot retrieval for the user's described product,
    runs embedding search + LLM rerank, then writes a concise recommendation
    into ChewyJourneyChatState.agent_response.
    """
    cjs: ChewyJourneyChatState = state.chewy_journey_chat_state
    user_msg = (cjs.last_user_message or "").strip()

    # 1) Understand the ask
    species = getattr(state.pet_profile, "species", None)
    breed = getattr(state.pet_profile, "breed", None)
    weight_lb = getattr(state.pet_profile, "weight_lb", None)
    age_mo = getattr(state.pet_profile, "age_months", None)
    stage = _life_stage(species, age_mo)

    product_type, soft_terms_syn = _extract_product_intent(user_msg, species)
    size_terms = _extract_size_terms(user_msg)
    brands = _extract_brands(user_msg)
    price_cap = _extract_budget(user_msg)

    # 2) Build a SlotSpec that your retriever understands (see slot_candidates_agent)
    #    - query_text drives the dense embedding query
    #    - soft_terms contribute to lexical bonus (light bias)
    qtext = _build_query_text(species, breed, stage, product_type, weight_lb, user_msg, size_terms)

    filters = FilterSpec(
        species=species if species in {"dog", "cat"} else None,
        weight_lb=weight_lb,
        categories=[product_type] if product_type not in {"product"} else None,
        brands=brands or None,
        price_max=price_cap if price_cap else None,
        availability_required=True,
    )
    budget = Budget(max=price_cap) if price_cap else Budget()

    slot = SlotSpec(
        slot_id=SLOT_ID,
        role="single_purchase",
        name=f"{product_type.title()} for {species}" if species else product_type.title(),
        must_have=True,
        allow_substitution=True,
        product_type=product_type,
        reason_for_suggestion=f"User asked: {user_msg}",
        query_text=qtext,
        soft_terms=(soft_terms_syn + size_terms + brands)[:6],
        filters=filters,
        budget=budget,
    )

    # 3) Place/replace this single slot on state and run retrieval → eval pipeline
    #    (retriever reads SlotSpec.query_text & fills SlotSpec.candidates)  ← ref guide
    #    (evaluator LLM-reranks + adds CandidateProduct.reasoning)
    state.bundle_state.slots = [slot]
    try:
        state = _retrieve_candidates(state)   # uses catalog_embeds.jsonl; env-configured. 
        state = _eval_candidates(state)       # adds reasoning + score_rerank
    except Exception as e:
        # Graceful fallback text if catalog or API is unavailable
        msg = (
            "I ran into an issue searching our catalog just now. "
            "Want me to narrow by brand, size, or budget and try again?"
        )
        cjs.agent_response = msg
        cjs.route = "product"
        cjs.domain = "product"
        cjs.debug.setdefault("product_enquiry", {})
        cjs.debug["product_enquiry"].update({
            "error": str(e),
            "query_text": qtext,
            "filters": filters.model_dump() if hasattr(filters, "model_dump") else {},
        })
        state.chewy_journey_chat_state = cjs
        return state

    # 4) Compose a concise recommendation (top N)
    # Fetch the (now reranked) slot & top picks
    slot_out = next((s for s in state.bundle_state.slots if s.slot_id == SLOT_ID), None)
    picks: List[CandidateProduct] = (slot_out.candidates or [])[: max(1, TOP_N)] if slot_out else []

    if not picks:
        reply = (
            "I didn’t find good matches yet. Would you like me to try a different style or brand, "
            "or set a price range?"
        )
    else:
        header_bits = []
        if stage and species:
            header_bits.append(f"{stage} {species}")
        elif species:
            header_bits.append(species)
        if product_type:
            header_bits.append(product_type)

        header = "Top picks for " + " ".join(header_bits).strip()
        lines = [header + ":"]
        for i, p in enumerate(picks, 1):
            price_txt = f"${p.price:,.2f}" if (getattr(p, "price", None) is not None) else ""
            brand_txt = f"{p.brand} — " if getattr(p, "brand", None) else ""
            reason = (p.reasoning or "").strip()
            bullet = f"{i}) {brand_txt}{p.title or 'Product'} {price_txt}".strip()
            lines.append(bullet)
            if reason:
                lines.append(f"   • {reason}")
        lines.append("Would you like one of these, or should I refine by size, brand, or budget?")
        reply = "\n".join(lines)

    # # 5) Write response + debug
    # cjs.agent_response = reply    
    # cjs.route = "product"
    # cjs.debug.setdefault("product_enquiry", {})
    # cjs.debug["product_enquiry"].update({
    #     "query_text": qtext,
    #     "soft_terms": slot.soft_terms,
    #     "filters": slot.filters.model_dump() if hasattr(slot.filters, "model_dump") else {},
    #     "budget": slot.budget.model_dump() if hasattr(slot.budget, "model_dump") else {},
    #     "top_skus": [getattr(p, "sku", None) for p in picks],
    # })
    # state.chewy_journey_chat_state = cjs
    return {
        "messages": [AIMessage(content=reply)],
        "chewy_journey_chat_state": {
            "agent_response": reply,
            "route": "product",
            "debug": {
                "query_text": qtext,
                "soft_terms": slot.soft_terms,
                "filters": slot.filters.model_dump() if hasattr(slot.filters, "model_dump") else {},
                "budget": slot.budget.model_dump() if hasattr(slot.budget, "model_dump") else {},
                "top_skus": [getattr(p, "sku", None) for p in picks],
            },
        }
    }


# -------------- Local smoke test
if __name__ == "__main__":
    
    from pathlib import Path
    from LG_Agents.states.sessionState import load_state

    # Minimal smoke test without real state loader; replace with your state loader as needed.
    json_path = Path(__file__).resolve().parent.parent / "gradio_interface" / "last_session-3.json"
    state = load_state(json_path)
    last_user_message="I am looking for a cat carrier for Luna"
    #last_user_message="Is pet insurance worth it?"
    state.chewy_journey_chat_state=ChewyJourneyChatState(last_user_message=last_user_message)

    state = product_enquiry_agent(state)
    print("\n--- AGENT REPLY ---")
    print(state.chewy_journey_chat_state.agent_response)
    print("\nDEBUG:", state.chewy_journey_chat_state.debug.get("insurance"))